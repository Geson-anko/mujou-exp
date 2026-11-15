r"""DIAGNOSIS ONLY (M17, 2026-11-12): the target tokens, the completion errors
and the SIGReg of $z$ of checkpoints of a run, on the held-out Zurich val
stacks and on training batches. Not a program of the runs, and nothing of the
runs imports it.

It backs, with ``mini_train``, the audit of 「監査を受けた直し（2026-11-12、回す
前）」 of ``exp/video/results/m17.md`` (the cause of the pilot's collapse, the
premise of 「診断（2026-11-12、BP、seed 100）」): under the preset's loss of
that time (SIGReg of $z$ alone, no sg, no EMA) the target tokens become a
constant of each stack, and the mean of the context then predicts the masked
tokens.

For the target tokens $s \in \mathbb{R}^{n \times T \times D}$ of a batch
(``spread``; $T$ = slots x ``per`` spatial positions, slot first):

- ``within``: $\operatorname{mean} (s - \bar s_x)^2$, $\bar s_x$ the stack's
  mean token (0: tokens constant within each stack)
- ``between``: the variance over the stacks of $\bar s_x$, mean over the
  dimensions
- ``temporal``: the squared deviation of a token from the mean over the slots
  of its spatial position (0: tokens constant in time)
- ``norm2``: $\operatorname{mean} s^2$

and for each context view ``v`` with masked tokens $\Omega$
(``view_errors``), the mean squared error against the masked targets of
``v<v>_pred`` the predictor, ``v<v>_ctxmean`` the mean of the visible tokens
(the encoder's tokens of the context view), ``v<v>_tgtmean`` the stack's mean
target token $\bar s_x$ (an oracle constant of the stack), and
``v<v>_self`` the mean of the masked targets themselves (what an ideal
constant of the stack misses).

One JSON line per checkpoint: ``val_tokens`` (the means over the val batches
in order), ``train_tokens`` (one stream batch at its 9000th step, its ready
stacks, masks of a CUDA generator of seed 9), ``sigreg`` (``weighted_sigreg``
of $z$ along ``vision.sigreg_directions`` directions of a CPU generator of
seed 11: the val batches in order and shuffled (seed 5), stream batches at
the steps 3000, 9000, 18000 and 36000, and ``train_one_video_stretch``: B
stacks of one training video within 1/24 of it (``stretch_ends``), for the
first six videos, seeds 0..5) and ``videos_in_batch``: the stacks of the
stream batch at step 9000 in each training video (``videos_of``).

It was written for and run on the code of 2026-11-12 before the audit's
fixes (``fdb2ce2``); it runs on the code now, not checked bit for bit
against that. Run on a GPU host, read-only (it loads checkpoints)::

    python -m exp.video.diag.diag_ckpt <run_dir> [<ckpt> ...]

``<ckpt>`` is a checkpoint's name under ``ckpt/`` without ``.pt`` or
``init`` (default: ``init 00300.0 01800.0``). The data comes from
``$MUJOU_EXP_DATA`` (default: the fleet's copy).
"""

import argparse
import dataclasses
import json
import os
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import torch
from torch import Tensor

from exp.lib import ckpt
from exp.lib.video_io import U8Reader
from exp.video.compose import from_dict
from exp.video.data.frames import StackedTourFrames
from exp.video.data.walking_tours import videos
from exp.video.jepa.loss import view_ids, weighted_sigreg
from exp.video.jepa.masks import multiblock_masks, pack_views
from exp.video.jepa.nodes import draw_packed_directions
from exp.video.jepa.vit import VisionModules, build_vision
from exp.video.programs.vision import PackedMeasures, tours_root

DATA = "/data/geson/mujou-exp/.fleet/data"  # the fleet hosts' copy of the data

Predictor = Callable[[Tensor, Tensor, Tensor], Tensor]


def spread(target: Tensor, per: int) -> dict[str, Tensor]:
    """``within``, ``between``, ``temporal`` and ``norm2`` (module docstring)
    of the target tokens ``target`` ``(n, T, D)`` of ``per`` spatial
    positions a slot."""
    n, t, d = target.shape
    mean = target.mean(1, keepdim=True)
    tv = target.view(n, t // per, per, d)
    return {
        "within": (target - mean).pow(2).mean(),
        "between": mean.squeeze(1).var(0, unbiased=False).mean(),
        "temporal": (tv - tv.mean(1, keepdim=True)).pow(2).mean(),
        "norm2": target.pow(2).mean(),
    }


def view_errors(
    tokens: Tensor, y: Tensor, keeps: Sequence[int], predictor: Predictor
) -> list[dict[str, Tensor]]:
    """``pred``, ``ctxmean``, ``tgtmean`` and ``self`` (module docstring) of
    each context view of the packed tokens ``tokens`` ``(n, T + sum(keeps),
    D)`` (the target view's $T$, then ``keeps[v]`` of view ``v``) with the
    masks ``y`` ``(n, views * T)`` (1: masked)."""
    t = tokens.shape[1] - sum(keeps)
    tgt = tokens[:, :t]
    d = tgt.shape[-1]
    mean = tgt.mean(1, keepdim=True)
    out = []
    at = t
    for vi, k in enumerate(keeps):
        ctx_ids, tgt_ids = view_ids(y[:, vi * t : (vi + 1) * t], k)
        ctx = tokens[:, at : at + k]
        at += k
        goal = tgt.gather(1, tgt_ids[..., None].expand(-1, -1, d))
        out.append(
            {
                "pred": (predictor(ctx, ctx_ids, tgt_ids) - goal).pow(2).mean(),
                "ctxmean": (ctx.mean(1, keepdim=True) - goal).pow(2).mean(),
                "tgtmean": (mean - goal).pow(2).mean(),
                "self": (goal - goal.mean(1, keepdim=True)).pow(2).mean(),
            }
        )
    return out


def stretch_ends(
    length: int, n: int, *, frac: float, first: int, generator: torch.Generator
) -> Tensor:
    """``n`` distinct last frames ``(n,)``, sorted, of stacks within one
    stretch of ``int(length * frac)`` frames of a video of ``length``
    frames, the stretch starting at or after frame ``first``."""
    span = int(length * frac)
    s0 = int(torch.randint(first, length - span, (1,), generator=generator))
    return (s0 + torch.randperm(span, generator=generator)[:n]).sort().values


def videos_of(ends: Tensor, firsts: Tensor, videos: int) -> list[int]:
    """The count of the frames ``ends`` ``(n,)`` of the loop of ``videos``
    videos in each video, ``firsts`` ``(videos,)`` the first frame of each in
    the loop."""
    vid = torch.searchsorted(firsts, ends, right=True) - 1
    return torch.bincount(vid, minlength=videos).tolist()


def mean_dicts(ds: Sequence[dict[str, Tensor]]) -> dict[str, float]:
    """The mean over ``ds`` of each key."""
    return {k: float(torch.stack([x[k] for x in ds]).mean()) for k in ds[0]}


def main(argv: Sequence[str] | None = None) -> None:
    """One JSON line per checkpoint of a run (module docstring)."""
    p = argparse.ArgumentParser(description="Tokens and SIGReg of checkpoints (M17).")
    p.add_argument("run", type=Path)
    p.add_argument("times", nargs="*", default=["init", "00300.0", "01800.0"])
    args = p.parse_args(argv)
    os.environ.setdefault("MUJOU_EXP_DATA", DATA)
    torch.backends.cuda.matmul.allow_tf32 = False
    run: Path = args.run
    job = json.loads((run / "job.json").read_text())
    conf = dataclasses.replace(from_dict(job["config"]), device="cuda")
    v, tours = conf.vision, conf.data.tours
    assert v is not None and tours is not None
    dev = torch.device("cuda")
    pm = PackedMeasures(conf)
    root = tours_root(tours)
    b = conf.data.streams
    per = pm.grid[0] * pm.grid[1]

    def mods_at(t: str) -> VisionModules:
        mods = build_vision(conf, seed=conf.seed)
        if t != "init":
            ckpt.load_eval(run / "ckpt" / f"{t}.pt", mods.named())
        for m in mods.named().values():
            m.eval()
        return mods

    @torch.no_grad()
    def token_stats(
        mods: VisionModules, chunks: Tensor, masks: list[Tensor]
    ) -> dict[str, Tensor]:
        h, y = pack_views(
            chunks, masks, keeps=pm.keeps, patch=v.patch, tubelet=v.tubelet
        )
        tok = mods.encoder(h, mods.segments)
        out = spread(tok[:, : mods.tokens_per_view], per)
        keeps = list(mods.segments[1:])
        for vi, errs in enumerate(view_errors(tok, y, keeps, mods.predictor)):
            for key, val in errs.items():
                out[f"v{vi}_{key}"] = val
        return out

    def val_batches(shuffle: bool = False) -> Iterator[tuple[Tensor, list[Tensor]]]:
        w, masks = pm.windows["val"], pm.masks["val"]
        n = len(w)
        g = torch.Generator().manual_seed(5)
        order = torch.randperm(n, generator=g) if shuffle else torch.arange(n)
        for i in range(0, n, b):
            idx = order[i : i + b]
            yield w[idx, : v.frames].to(dev), [m[idx].to(dev) for m in masks]

    def streams() -> StackedTourFrames:
        return StackedTourFrames(
            root,
            n=b,
            held_out=tours.held_out,
            frames=v.frames,
            stride=v.stride,
            advance=tours.advance,
        )

    def train_stream(k: int) -> tuple[Tensor, Tensor]:
        s = streams()
        s.load_state_dict({"k": k})
        x, ready, _ = s.next()
        return x, ready

    def one_video_stretch(name: str, seed: int) -> Tensor:
        r = U8Reader(root / name)
        g = torch.Generator().manual_seed(seed)
        ends = stretch_ends(
            len(r), b, frac=1 / 24, first=v.frames * v.stride, generator=g
        )
        back = torch.arange(v.frames - 1, -1, -1) * v.stride
        return torch.stack([r[e - back] for e in ends.tolist()])

    dz, _ = draw_packed_directions(
        v.sigreg_directions,
        v.z_width,
        v.dim,
        tokens=False,
        generator=torch.Generator().manual_seed(11),
    )
    dz = dz.to(dev)

    def sig(z: Tensor) -> float:
        return float(weighted_sigreg(z, torch.ones(len(z), device=z.device), dz))

    gm = torch.Generator(device="cuda").manual_seed(9)
    names = [n for n in videos(root) if n != tours.held_out]
    for t in args.times:
        mods = mods_at(t)
        res: dict[str, object] = {"t": t}
        res["val_tokens"] = mean_dicts(
            [token_stats(mods, c, m) for c, m in val_batches()]
        )
        with torch.no_grad():
            x, ready = train_stream(9000)
            x = x[ready]
            masks = [
                multiblock_masks(len(x), pm.grid, mc, keep=k, generator=gm)
                for mc, k in zip(v.context_masks, pm.keeps, strict=True)
            ]
            res["train_tokens"] = mean_dicts([token_stats(mods, x.to(dev), masks)])
            s_sorted = [sig(mods.encode(c)) for c, _ in val_batches()]
            s_shuf = [sig(mods.encode(c)) for c, _ in val_batches(shuffle=True)]
            s_train = []
            for k in (3000, 9000, 18000, 36000):
                x, ready = train_stream(k)
                s_train.append(sig(mods.encode(x[ready].to(dev))))
            s_onevid = [
                sig(mods.encode(one_video_stretch(nm, i).to(dev)))
                for i, nm in enumerate(names[:6])
            ]
            res["sigreg"] = {
                "val_sorted": sum(s_sorted) / len(s_sorted),
                "val_shuffled": sum(s_shuf) / len(s_shuf),
                "train_streams": s_train,
                "train_one_video_stretch": s_onevid,
            }
            s = streams()
            ends = (s._offsets + 9000 * s._advance) % s._total
            res["videos_in_batch"] = videos_of(ends, s._firsts, len(names))
        print(json.dumps(res, default=float), flush=True)


if __name__ == "__main__":
    main()
