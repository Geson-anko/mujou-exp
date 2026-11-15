r"""DIAGNOSIS ONLY (M17, 2026-11-12): a BP replica of the vision's learning on
the ``m17/wt_n16`` preset (one optimizer step per call, ``time.frame`` =
1/30 s of elapsed time per step), with the order of the data, the rate, the
warmup, sg and what SIGReg sees as arguments, logging the held-out collapse
of the target tokens and their completion error. Not a program of the runs,
and nothing of the runs imports it.

It backs, with ``diag_ckpt``, the audit of 「監査を受けた直し（2026-11-12、回す
前）」 of ``exp/video/results/m17.md`` (the premise of 「診断（2026-11-12、BP、
seed 100）」): the collapse of the pilot came from the objective, not from the
rate, the warmup or the order of the data. With SIGReg of $z$ alone, the
same collapse (tokens constant within a stack) came with i.i.d. batches and at
a rate of 1e-5.

The data: ``order`` ``stream`` takes the next batch of the B streams of
``StackedTourFrames`` (as the runs); ``iid`` puts the B streams at uniformly
random frames of the loop of the training videos (a CPU generator of seed
123) and takes their next stacks, the not ``ready`` ones (reaching before
their video's first frame) masked out as in the runs. The context masks come
from a CUDA generator of seed ``seed + 1``, the SIGReg directions from one of
seed ``seed + 2``; the network is built with ``seed`` (100).

Every 500 steps, a JSON line of the mean training ``pred`` and ``sigreg``
since the last line and of the held-out val stacks (the means over batches
of B in a fixed order, seed 5; ``eval_row``): ``within``, ``between`` and
``temporal`` of ``diag_ckpt.spread``, and for each context view ``pred<v>``,
the predictor's mean squared error on the masked tokens, and
``ncomp<v>`` = ``pred<v>`` / ``self`` (``diag_ckpt.view_errors``): the
completion error over the masked targets' own spread about their mean (1:
the error of the best constant of the stack, 0: exact); ``sig_shuf``, the
SIGReg of $z$ of the batch along the val directions of ``PackedMeasures``.

It was written for and run on the code of 2026-11-12 before the audit's
fixes (``fdb2ce2``, whose preset had ``vision.sigreg_on=z``); it runs on the
code now, not checked bit for bit against that. Run on a GPU host::

    python -m exp.video.diag.mini_train <tag> <stream|iid> <steps> <lr> \
        <warmup_s> <sg: 0|1> [<sigreg_on>, default z]

The data comes from ``$MUJOU_EXP_DATA`` (default: the fleet's copy).
"""

import argparse
import json
import os
import time
from collections.abc import Sequence

import torch
from torch import Tensor

from exp.video.compose import compose
from exp.video.data.frames import StackedTourFrames
from exp.video.diag.diag_ckpt import DATA, Predictor, spread, view_errors
from exp.video.dyn.nets import make_optimizer
from exp.video.jepa.loss import jepa_loss_packed, weighted_sigreg
from exp.video.jepa.masks import multiblock_masks, pack_views
from exp.video.jepa.nodes import draw_packed_directions
from exp.video.jepa.vit import build_vision
from exp.video.programs.vision import PackedMeasures, tours_root

EVERY = 500  # steps between two lines


def eval_row(
    tokens: Tensor, y: Tensor, keeps: Sequence[int], per: int, predictor: Predictor
) -> dict[str, Tensor]:
    """``within``, ``between``, ``temporal``, ``pred<v>`` and ``ncomp<v>``
    (module docstring) of the packed tokens ``tokens`` with the masks ``y``
    (``diag_ckpt.view_errors``), ``per`` spatial positions a slot."""
    t = tokens.shape[1] - sum(keeps)
    s = spread(tokens[:, :t], per)
    row = {k: s[k] for k in ("within", "between", "temporal")}
    for vi, e in enumerate(view_errors(tokens, y, keeps, predictor)):
        row[f"pred{vi}"] = e["pred"]
        # completion error / within-stack spread; the floor only keeps a
        # spread of exactly 0 (all masked targets equal) from dividing by 0
        row[f"ncomp{vi}"] = e["pred"] / e["self"].clamp(min=1e-12)
    return row


def main(argv: Sequence[str] | None = None) -> None:
    """The replica's JSON lines (module docstring)."""
    p = argparse.ArgumentParser(description="BP replica of the M17 vision.")
    p.add_argument("tag")
    p.add_argument("order", choices=["stream", "iid"])
    p.add_argument("steps", type=int)
    p.add_argument("lr", type=float)
    p.add_argument("warmup", type=float)
    p.add_argument("sg", choices=["0", "1"])
    p.add_argument("sigreg_on", nargs="?", default="z")
    args = p.parse_args(argv)
    os.environ.setdefault("MUJOU_EXP_DATA", DATA)
    sg = args.sg == "1"
    conf = compose(
        [
            "+experiment=m17/wt_n16",
            "learner=bp",
            "device=cuda",
            "seed=100",
            f"optim_vision.lr={args.lr}",
            f"optim_vision.warmup={args.warmup}",
            f"vision.sg={sg}",
            f"vision.sigreg_on={args.sigreg_on}",
        ]
    )
    v, tours, optim = conf.vision, conf.data.tours, conf.optim_vision
    assert v is not None and tours is not None and optim is not None
    dev = torch.device("cuda")
    b = conf.data.streams
    mods = build_vision(conf, seed=conf.seed)
    params = [*mods.encoder.parameters(), *mods.predictor.parameters()]
    opt = make_optimizer(optim, conf.time.frame).bind(params)
    pm = PackedMeasures(conf)
    root = tours_root(tours)
    streams = StackedTourFrames(
        root,
        n=b,
        held_out=tours.held_out,
        frames=v.frames,
        stride=v.stride,
        advance=tours.advance,
    )
    gi = torch.Generator().manual_seed(123)
    gm = torch.Generator(device="cuda").manual_seed(conf.seed + 1)
    gd = torch.Generator(device="cuda").manual_seed(conf.seed + 2)
    keeps = list(mods.segments[1:])
    per = pm.grid[0] * pm.grid[1]

    def batch() -> tuple[Tensor, Tensor]:
        if args.order == "iid":  # B uniformly random frames of the loop
            streams._offsets = torch.randint(0, streams._total, (b,), generator=gi)
            streams.load_state_dict({"k": 1})
        x, ready, _ = streams.next()
        return x, ready

    @torch.no_grad()
    def evaluate() -> dict[str, float]:
        acc: dict[str, float] = {}
        nb = 0
        dz = pm._directions["val"][0]
        w, masks = pm.windows["val"], pm.masks["val"]
        perm = torch.randperm(len(w), generator=torch.Generator().manual_seed(5))
        for i in range(0, len(w), b):
            idx = perm[i : i + b]
            c = w[idx, : v.frames].to(dev)
            h, y = pack_views(
                c,
                [m[idx].to(dev) for m in masks],
                keeps=pm.keeps,
                patch=v.patch,
                tubelet=v.tubelet,
            )
            tok = mods.encoder(h, mods.segments)
            r = eval_row(tok, y, keeps, per, mods.predictor)
            z = mods.z_of(tok[:, : mods.tokens_per_view])
            r["sig_shuf"] = weighted_sigreg(z, torch.ones(len(z), device=dev), dz)
            for k, x in r.items():
                acc[k] = acc.get(k, 0.0) + float(x)
            nb += 1
        return {k: round(x / nb, 5) for k, x in acc.items()}

    print(
        json.dumps(
            {
                "tag": args.tag,
                "order": args.order,
                "lr": args.lr,
                "warmup": args.warmup,
                "sg": sg,
                "sigreg_on": args.sigreg_on,
            }
        ),
        flush=True,
    )
    print(json.dumps({"step": 0, **evaluate()}), flush=True)
    tr: list[tuple[float, float]] = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        x, ready = batch()
        x = x.to(dev)
        masks = [
            multiblock_masks(b, pm.grid, mc, keep=k, generator=gm)
            for mc, k in zip(v.context_masks, pm.keeps, strict=True)
        ]
        h, y = pack_views(x, masks, keeps=pm.keeps, patch=v.patch, tubelet=v.tubelet)
        dz, dt = draw_packed_directions(
            v.sigreg_directions, v.z_width, v.dim, tokens=False, generator=gd
        )
        terms = jepa_loss_packed(
            mods.encoder(h, mods.segments),
            y,
            ready.to(dev),
            dz,
            dt,
            mods.predictor,
            keeps=keeps,
            proj=mods.proj,
            lam=v.sigreg,
            sg=v.sg,
            sigreg_on=v.sigreg_on,
            pred_loss=v.pred_loss,
        )
        grads = torch.autograd.grad(terms.loss, params, materialize_grads=True)
        opt.step_(list(grads), conf.time.frame)
        tr.append((float(terms.pred), float(terms.sigreg)))
        if step % EVERY == 0:
            pred = sum(a for a, _ in tr) / len(tr)
            sig = sum(s for _, s in tr) / len(tr)
            tr = []
            line = {
                "step": step,
                "t_virtual": round(step / 30, 1),
                "train_pred": round(pred, 5),
                "train_sig": round(sig, 3),
                "wall": round(time.time() - t0),
            }
            print(json.dumps({**line, **evaluate()}), flush=True)


if __name__ == "__main__":
    main()
