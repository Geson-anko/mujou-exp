r"""V18d's selection numbers on the dyn/copy validation video
(plans/v18d.md, 「学習の実装」 の「検証の補完の損失」, and
results/v18d/README.md, 「短い run の判定」): the validation completion loss
and the collapse numbers, none of Zurich.

- **The clips** (``val_clips``): ``CLIPS`` clips of ``TIMES`` stacks
  $n$ apart, the lags spread evenly over ``LAGS`` (clip $j$ takes
  ``LAGS[j % 13]``), each clip's first stack placed uniformly (by
  ``VAL_SEED``) over every place of the validation video(s) where the whole
  clip fits: the whole video, not ``time_val``'s 128 s blocks, which a clip
  of $n = 512$ ($3 \cdot 512 + $ a stack's span frames) does not fit.
- **The masks** (``val_masks``): ``TUBES`` tubes, distinct rows of
  $F_k$ (the run's family, V18c's $F_8$) drawn by ``VAL_SEED`` after the
  clips, spread to the grid and hidden at every time, then the time side's
  rows of ``jepa.time_masks``, all on every clip.
- **The loss** (``rel_losses``): the target side's tokens $t$ and the
  predictor's $\hat t$ of ``jepa.st_predict``; of each side $X$ (the
  tubes ``now``, the time side ``time``) $L^{\mathrm{rel}}_X = \sum \lVert
  \hat t - t \rVert^2 / \sum \lVert t - \bar t_p \rVert^2$ over its masks'
  hidden tokens of every clip, $\bar t_p$ the mean of position $p$ over all
  the clips' stacks (the predictor's $R_{\mathrm{clim}}$); ``st`` their
  mean. Of each time-side row (``CUTS``) and each lag, the same ratio, as
  descriptions.
- **The collapse numbers**: ``time_val.stats_rows`` (``rank``,
  ``token_r2`` of the validation video's held stacks, ...). Its
  ``sigreg_heldin`` (of single training stacks) is kept as
  ``sigreg_heldin_stack``; ``sigreg_heldin`` is the registered one, the
  mean over the times $i$ of SIGReg of time $i$'s codes of
  ``HELDIN_BATCHES`` batches of ``HELDIN_BATCH`` training clips
  (``iid.Units.draw_clips`` from ``seed + HELDIN_SEED``, lags of
  ``iid.draw_lags`` on the run's ``st_lags`` from ``seed + HELDIN_LAG_SEED``),
  along ``CodeStats``' directions. ``pred_ratio_calib`` is V18c's
  single-stack predictor's, which a run of $K > 1$ (its positions with
  $\phi$ and $\psi$) does not have; $L^{\mathrm{rel}}_{\mathrm{now}}$ takes
  its place and it is not written.

``evaluate_run`` appends a row per step to
``<run>/posthoc_v18d/st_select.jsonl``::

    python -m exp.vision.st_val <run_dir> [--steps 6000]
"""

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from exp.lib import ckpt
from exp.lib.provenance import configure_determinism
from exp.lib.rundir import write_json
from exp.lib.sigreg import sigreg
from exp.lib.tours.stacks import tours_root
from exp.lib.video_io import U8Reader
from exp.vision.dyn_eval import DynConf
from exp.vision.geom_eval import _conf_of
from exp.vision.iid import Units, draw_lags
from exp.vision.jepa import (
    COARSE_GRID,
    JEPAModules,
    directions,
    expand_masks,
    mask_family,
    st_predict,
    time_masks,
)
from exp.vision.jepa_diag import run_ckpts
from exp.vision.jepa_eval import (
    EVAL_DIRECTION_SEED,
    EVAL_DIRECTIONS,
    HELDIN_BATCH,
    HELDIN_BATCHES,
    HELDIN_SEED,
    build_run_jepa,
)
from exp.vision.jepa_iid import ST_LAGS
from exp.vision.stream_eval import append_jsonl
from exp.vision.time_val import LAGS, OUT, setup, stats_rows

__all__ = [
    "CLIPS",
    "CUTS",
    "HELDIN_LAG_SEED",
    "ROWS",
    "TIMES",
    "TUBES",
    "VAL_SEED",
    "clip_stacks",
    "completion_errors",
    "evaluate_run",
    "heldin_sigreg",
    "main",
    "rel_losses",
    "val_clips",
    "val_masks",
]

ROWS = "st_select.jsonl"
CLIPS = 512
TIMES = 4  # K
TUBES = 8
VAL_SEED = 4201
HELDIN_LAG_SEED = 4203  # added to the seed: the held-in clips' lags
CLIP_BATCH = 32  # clips a time through the JEPA
# the time side's rows of time_masks(4), in its order (the hidden times, from 0)
CUTS = ("past", "interp_past", "outward", "middle", "interp_future", "future")


def val_clips(
    lengths: Sequence[int],
    span: int,
    *,
    count: int = CLIPS,
    lags: Sequence[int] = LAGS,
    times: int = TIMES,
    seed: int = VAL_SEED,
) -> tuple[Tensor, Tensor, Tensor, torch.Generator]:
    r"""The clips (module docstring) of the videos of ``lengths`` frames, a
    stack of ``span`` + 1 frames: each clip's video, first frame $f$ (its
    stack $i$ the frames $f + i n$ to $f + i n + $ ``span``) and lag $n$,
    ``(count,)`` int64 each, and the generator after the draw (``val_masks``
    draws on). The places of a lag are every $(v, f)$ with $f + (\mathrm{times}
    - 1) n + \mathrm{span} < L_v$, drawn uniformly.

    Raises:
        ValueError: A clip of a lag fits in no video.
    """
    gen = torch.Generator().manual_seed(seed)
    n = torch.tensor([lags[j % len(lags)] for j in range(count)])
    video = torch.empty(count, dtype=torch.long)
    first = torch.empty(count, dtype=torch.long)
    size = torch.tensor(lengths)
    for j, lag in enumerate(n.tolist()):
        fits = (size - (times - 1) * lag - span).clamp(min=0)  # places per video
        total = int(fits.sum())
        if total == 0:
            raise ValueError(f"a clip of lag {lag} fits in no video")
        u = int(torch.randint(total, (1,), generator=gen))
        v = int(torch.searchsorted(fits.cumsum(0), torch.tensor(u), right=True))
        video[j], first[j] = v, u - int(fits[:v].sum())
    return video, first, n, gen


def clip_stacks(
    readers: Sequence[Any],
    video: Tensor,
    first: Tensor,
    n: Tensor,
    *,
    frames: int,
    stride: int,
    times: int = TIMES,
) -> Tensor:
    """The clips' stacks u8 ``(B, times, frames, C, H, W)`` of ``val_clips``'
    places, read from ``readers`` (one per video, indexable by frames)."""
    offsets = torch.arange(frames) * stride
    out = []
    for v, f, lag in zip(video.tolist(), first.tolist(), n.tolist(), strict=True):
        r = readers[v]
        out.append(torch.stack([r[f + i * lag + offsets] for i in range(times)]))
    return torch.stack(out)


def val_masks(
    family: Tensor, s: int, gen: torch.Generator, *, tubes: int = TUBES,
    times: int = TIMES
) -> Tensor:  # fmt: skip
    """The masks (module docstring) bool ``(tubes + C(times, times/2),
    times, s^2 F_T)``: ``tubes`` distinct rows of ``family`` ``(F, F_T)``
    (a ``randperm`` by ``gen``) spread by ``s`` and hidden at every time,
    then the rows of ``time_masks(times)`` hiding whole times."""
    pick = family[torch.randperm(len(family), generator=gen)[:tubes]]
    tube = expand_masks(pick.cpu(), s)  # (tubes, T)
    count = tube.shape[1]
    whole = time_masks(times)[:, :, None].expand(-1, -1, count)
    return torch.cat([tube[:, None].expand(-1, times, -1), whole])


@torch.no_grad()
def completion_errors(
    mods: JEPAModules, x: Tensor, n: Tensor, masks: Tensor
) -> tuple[Tensor, Tensor]:
    r"""Of the clips ``x`` u8 ``(B, K, N, 3, H, W)`` at lags ``n`` ``(B,)``
    and each mask of ``masks`` ``(M, K, T)`` on every clip: $\sum \lVert
    \hat t - t \rVert^2$ over the hidden tokens ``(B, M)`` float64
    (``jepa.st_predict``), and the target side's tokens ``(B, K, T, d)``."""
    b, k = x.shape[:2]
    err = []
    tokens = None
    for mask in masks:
        z_hat, hide, tok = st_predict(mods, x, n, mask[None].expand(b, -1, -1))
        target = tok.unflatten(0, (b, k)).flatten(1, 2)  # (B, K T, d)
        e = z_hat - target.gather(1, hide[..., None].expand(-1, -1, z_hat.shape[-1]))
        err.append(e.double().square().sum((1, 2)))
        tokens = tok.unflatten(0, (b, k))
    assert tokens is not None
    return torch.stack(err, 1), tokens


def rel_losses(
    err: Tensor, tokens: Tensor, masks: Tensor, n: Tensor, tubes: int = TUBES
) -> dict[str, Any]:
    r"""$L^{\mathrm{rel}}$ (module docstring) of ``completion_errors``' sums
    ``err`` ``(B, M)`` and tokens ``(B, K, T, d)`` of all the clips, under
    ``masks`` ``(M, K, T)`` (the first ``tubes`` the tubes) at lags ``n``
    ``(B,)``: ``now``, ``time``, ``st`` = their mean, ``cuts`` (of each
    time-side row) and ``lags`` (``{"n": [...], "now": [...], "time":
    [...]}``)."""
    t = tokens.double()
    tbar = t.flatten(0, 1).mean(0)  # (T, d): of each position
    dev = (t - tbar).square().sum(-1)  # (B, K, T)
    den = torch.einsum("bip,mip->bm", dev, masks.to(dev))  # (B, M)
    err = err.double().to(den.device)

    def ratio(rows: Tensor | slice, cols: slice | int) -> float:
        return float(err[rows][:, cols].sum() / den[rows][:, cols].sum())

    every = slice(None)
    now, time = slice(0, tubes), slice(tubes, None)
    out: dict[str, Any] = {"now": ratio(every, now), "time": ratio(every, time)}
    out["st"] = (out["now"] + out["time"]) / 2
    names = (
        CUTS if masks.shape[0] - tubes == len(CUTS) else range(masks.shape[0] - tubes)
    )
    out["cuts"] = {str(c): ratio(every, tubes + j) for j, c in enumerate(names)}
    lags = sorted(set(n.tolist()))
    out["lags"] = {
        "n": lags,
        "now": [ratio(n == g, now) for g in lags],
        "time": [ratio(n == g, time) for g in lags],
    }
    return out


@torch.no_grad()
def heldin_sigreg(
    mu: Callable[[Tensor], Tensor], clips: Sequence[Tensor], dirs: Tensor
) -> dict[str, Any]:
    r"""``sigreg_heldin`` (module docstring) of the batches of clips
    ``clips`` (each ``(B, K, N, 3, H, W)``) and the codes ``mu`` (a stack
    batch to ``(B, D)``) along ``dirs``: the mean over the batches and the
    times of SIGReg of time $i$'s codes, and ``sigreg_heldin_times`` the
    times' means."""
    per = []
    for x in clips:
        per.append([float(sigreg(z, z[:0], dirs)) for z in
                    (mu(x[:, i]) for i in range(x.shape[1]))])  # fmt: skip
    each = torch.tensor(per, dtype=torch.float64).mean(0)
    return {"sigreg_heldin": float(each.mean()), "sigreg_heldin_times": each.tolist()}


def evaluate_run(
    run_dir: Path,
    *,
    steps: Sequence[int] | None = None,
    root: Path | None = None,
    device: str | None = None,
) -> Path:
    """The rows (module docstring) of the run's checkpoints of ``steps``
    (default the last) into ``run_dir / OUT / ROWS`` (appended). Returns the
    file.

    Raises:
        ValueError: A step has no checkpoint, or the run is not of $K$ =
            ``TIMES`` stacks.
    """
    conf = _conf_of(run_dir, device)
    settings = json.loads((run_dir / "jepa.json").read_text())
    if settings.get("st") != TIMES:
        raise ValueError(
            f"st_val is of V18d's runs of st {TIMES}, not {settings.get('st')}"
        )
    ckpts = run_ckpts(run_dir)
    todo = [max(ckpts)] if steps is None else list(steps)
    missing = [s for s in todo if s not in ckpts]
    if missing:
        raise ValueError(f"no checkpoint of steps {missing} (of {list(ckpts)})")
    determinism = configure_determinism(conf.device)
    dev = torch.device(conf.device)
    root = tours_root(conf.data.tours.data_version) if root is None else root
    out = run_dir / OUT
    out.mkdir(parents=True, exist_ok=True)
    vs, stats, names, _ = setup(conf, root, out)
    t = conf.data.tours
    span = (t.frames - 1) * t.stride
    readers = [U8Reader(root / v) for v in names]
    video, first, n, gen = val_clips([len(r) for r in readers], span)
    family = mask_family(settings["k"], COARSE_GRID)
    masks = val_masks(family, settings.get("expand", 1), gen)
    x = clip_stacks(readers, video, first, n, frames=t.frames, stride=t.stride)
    lo, hi = settings.get("st_lags") or ST_LAGS
    units = Units(conf, root, "all", seed=conf.seed + HELDIN_SEED)
    lag_gen = torch.Generator().manual_seed(conf.seed + HELDIN_LAG_SEED)
    held = [units.draw_clips(draw_lags(HELDIN_BATCH, lo, hi, lag_gen), TIMES)[0]
            for _ in range(HELDIN_BATCHES)]  # fmt: skip
    write_json(
        out / "st_config.json",
        {
            "val_videos": list(names),
            "clips": CLIPS,
            "times": TIMES,
            "lags": list(LAGS),
            "tubes": TUBES,
            "val_seed": VAL_SEED,
            "cuts": list(CUTS),
            "video": video.tolist(),
            "first": first.tolist(),
            "n": n.tolist(),
            "tube_rows": masks[:TUBES, 0].long().tolist(),
            "heldin": [HELDIN_BATCHES, HELDIN_BATCH],
            "heldin_lags": [lo, hi],
            "split_seed": DynConf().split_seed,
            "determinism": determinism,
        },  # fmt: skip
    )
    mods = build_run_jepa(conf, settings)
    for step in todo:
        head = ckpt.load_eval(ckpts[step], mods.modules())
        row: dict[str, Any] = {"step": step, "t": head["t"], "code": "z"}
        errs, toks = [], []
        for c in range(0, len(x), CLIP_BATCH):
            xb = x[c : c + CLIP_BATCH].to(dev)
            e, tok = completion_errors(mods, xb, n[c : c + CLIP_BATCH].to(dev),
                                       masks.to(dev))  # fmt: skip
            errs.append(e.cpu())
            toks.append(tok.double().cpu())
        rel = rel_losses(torch.cat(errs), torch.cat(toks), masks, n)
        row |= {"L_now": rel["now"], "L_time": rel["time"], "L_st": rel["st"],
                "L_cuts": rel["cuts"], "L_lags": rel["lags"]}  # fmt: skip
        base = stats_rows(mods, stats, vs, dev)
        base["sigreg_heldin_stack"] = base.pop("sigreg_heldin")
        row |= base
        dim = mods.mu(held[0][:2, 0].to(dev)).shape[1]
        gen_d = torch.Generator().manual_seed(conf.seed + EVAL_DIRECTION_SEED)
        a = directions(EVAL_DIRECTIONS, dim, gen_d).to(dev)  # CodeStats' directions
        row |= heldin_sigreg(lambda s: mods.mu(s.to(dev)).float(), held, a)
        append_jsonl(out / ROWS, [row])
    return out / ROWS


def main(argv: Sequence[str] | None = None) -> Path:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("run_dir", type=Path)
    p.add_argument("--steps", type=lambda s: [int(x) for x in s.split(",")])
    p.add_argument("--device", default=None)
    p.add_argument("--root", type=Path, default=None)
    a = p.parse_args(argv)
    return evaluate_run(a.run_dir, steps=a.steps, root=a.root, device=a.device)


if __name__ == "__main__":
    print(main())
