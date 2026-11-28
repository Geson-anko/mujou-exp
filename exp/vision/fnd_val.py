r"""The foundation's numbers of the validation video (``plans/foundation.md``
3, 5; F1 to F3): what the short runs' choice of the long runs reads, none of
Zurich. ``jepa_iid.train_fnd`` writes a row of ``val_rows`` at each
checkpoint.

The validation video is the dyn/copy validation video
(``dyn_eval.split_videos`` of ``DynConf.split_seed``; a training video, as
V18c's data are kept; neither Zurich nor dyn/copy's record video).

- **The completion loss**: its ``VAL_COUNT`` stacks of the arm's spacings
  (``fnd_jepa.draw_spacings``), each one's last frame uniform where it fits
  in the video, and their masks (``fnd_jepa.draw_masks``), all of
  ``VAL_SEED``: ``val_pred`` (the loss's form, the mean over the stacks of
  each one's mean squared error over its hidden tokens) and ``val_rel`` $=
  \sum \lVert \hat t - t \rVert^2 / \sum \lVert t - \bar t_\omega
  \rVert^2$ over the hidden tokens of every stack ($\bar t_\omega$ the mean
  of token $\omega$ over the stacks: the predictor's $R_{\mathrm{clim}}$),
  and ``val_rel_short`` and ``val_rel_long`` of each kind's stacks.
- **The collapse numbers**: of its ``HELD_COUNT`` stacks of $s$ = 1 whose
  last frames are spread evenly over the video (the codes the evaluation
  reads): ``rank`` (K2, ``metrics.effective_rank`` of $z$),
  ``participation``, ``token_r2`` (K3, ``jepa_eval.token_r2`` fit on the
  even stacks, held the odd), ``time_r2`` (``fnd_jepa.time_r2``), ``t_var``
  and ``z_var`` (the mean over the coordinates of the variances) and
  ``sigreg_val`` (the mean of SIGReg over batches of ``EVAL_BATCH``); of
  ``HELDIN_BATCHES`` batches of training stacks (``iid.Units.draw`` of
  ``HELDIN_SEED``) ``sigreg_heldin`` (K1), and of as many of $N(0, I)$
  ``sigreg_gauss``, along ``EVAL_DIRECTIONS`` directions of ``EVAL_SEED``.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from exp.lib.metrics import effective_rank
from exp.lib.sigreg import sigreg
from exp.lib.tours.walking_tours import videos
from exp.lib.video_io import U8Reader
from exp.vision.dyn_eval import DynConf, split_videos
from exp.vision.fnd_jepa import (
    EVAL_BATCH,
    KINDS,
    Arm,
    complete,
    context_index,
    draw_masks,
    draw_spacings,
    encode,
    targets,
    time_r2,
    uniform_int,
)
from exp.vision.iid import Units
from exp.vision.jepa import JEPAModules, directions
from exp.vision.jepa_eval import participation, token_r2
from exp.vision.models import early_fusion
from exp.vision.schema.run import RunConf

__all__ = [
    "EVAL_DIRECTIONS",
    "EVAL_SEED",
    "HELDIN_BATCHES",
    "HELDIN_SEED",
    "HELD_COUNT",
    "VAL_COUNT",
    "VAL_SEED",
    "ValSet",
    "val_rows",
    "val_set",
]

VAL_SEED = 4301  # the validation stacks, their spacings and masks
HELDIN_SEED = 4303  # the held-in training stacks (Units' generator)
EVAL_SEED = 4307  # the directions and the Gaussian reference of the numbers
VAL_COUNT = 512  # the validation stacks of the completion loss
HELD_COUNT = 1024  # the validation stacks of s = 1 of the collapse numbers
HELDIN_BATCHES = 4  # the batches of training stacks of sigreg_heldin
EVAL_DIRECTIONS = 1024  # M of the numbers' SIGReg


@dataclass(frozen=True)
class ValSet:
    """The validation video's stacks (module docstring): ``stacks`` u8 ``(V,
    n, 3, S, S)`` of the ``spacing`` ``(V,)``, their ``masks`` ``(V, |Ω|)``
    and ``kinds`` ``(V,)``; ``held`` u8 ``(C, n, 3, S, S)`` of $s$ = 1;
    ``heldin`` ``HELDIN_BATCHES`` batches of training stacks; ``video`` its
    name."""

    video: str
    stacks: Tensor
    spacing: Tensor
    masks: Tensor
    kinds: Tensor
    held: Tensor
    heldin: list[Tensor]


def val_set(
    conf: RunConf,
    root: Path,
    arm: Arm,
    *,
    s_max: int,
    times: int,
    count: int = VAL_COUNT,
    held: int = HELD_COUNT,
    batch: int = EVAL_BATCH,
) -> ValSet:
    """The ``ValSet`` of the run ``conf`` (its stacks; module docstring):
    ``count`` stacks of ``arm``'s spacings, each one's last frame uniform
    where it fits in the validation video, then their masks, all of
    ``VAL_SEED``; ``held`` stacks of $s$ = 1 whose last frames are spread
    evenly over the video; ``HELDIN_BATCHES`` x ``batch`` training stacks
    (``Units.draw`` of ``HELDIN_SEED``)."""
    t = conf.data.tours
    names = [v for v in videos(root) if v != t.held_out]
    video = split_videos(names, DynConf().split_seed)["val"][0]
    reader = U8Reader(root / video)
    span = (t.frames - 1) * t.stride
    back = torch.arange(t.frames - 1, -1, -1) * t.stride  # (n,)
    gen = torch.Generator().manual_seed(VAL_SEED)
    spacing = draw_spacings(count, arm, s_max, gen)
    ends = spacing * span + uniform_int(len(reader) - spacing * span, gen)
    stacks = torch.stack([reader[e - s * back] for e, s in zip(ends, spacing)])
    grid = t.size // conf.model.patch
    masks, kinds = draw_masks(count, times, grid, arm.timed, gen)
    last = torch.linspace(span, len(reader) - 1, held).round().long()
    units = Units(conf, root, "all", seed=HELDIN_SEED)
    heldin = [units.draw(batch) for _ in range(HELDIN_BATCHES)]
    return ValSet(
        video,
        stacks,
        spacing,
        masks,
        kinds,
        torch.stack([reader[e - back] for e in last]),
        heldin,
    )


@torch.no_grad()
def val_rows(
    mods: JEPAModules, vs: ValSet, *, heads: int, times: int, scaled: bool
) -> dict[str, float]:
    """The numbers of the validation video (module docstring) of the JEPA
    ``mods`` (``scaled``: $\\psi(\\ell)$ of the stacks' spacings to the
    predictor)."""
    dev = mods.parameters()[0].device
    sq, per, ts = [], [], []
    for i in range(0, len(vs.stacks), EVAL_BATCH):
        x = vs.stacks[i : i + EVAL_BATCH].to(dev)
        hidden = vs.masks[i : i + EVAL_BATCH].to(dev)
        s = vs.spacing[i : i + EVAL_BATCH].to(dev)
        images = early_fusion(x)
        t = mods.proj(encode(mods, images, heads))
        keep, valid, slot = context_index(hidden)
        context = encode(mods, images, heads, keep, valid)
        ell = s.double().log2() if scaled else None
        err = (complete(mods, context, slot, hidden, times, ell) - t).square()
        sq.append((err.sum(-1) * hidden).double())  # (b, |Ω|)
        per.append(sq[-1].sum(1) / (hidden.sum(1) * t.shape[-1]))
        ts.append(t)
    t_all = torch.cat(ts).double()  # (V, |Ω|, d)
    hidden = vs.masks.to(dev)
    dev2 = (t_all - t_all.mean(0)).square().sum(-1) * hidden  # (V, |Ω|)
    sq_all = torch.cat(sq)
    row: dict[str, float] = {
        "val_pred": float(torch.cat(per).mean()),
        "val_rel": float(sq_all.sum() / dev2.sum()),
    }
    kinds = vs.kinds.to(dev)
    for i, kind in enumerate(KINDS):
        on = kinds == i
        row[f"val_rel_{kind.name}"] = float(sq_all[on].sum() / dev2[on].sum())
    t_held, z_held = targets(mods, vs.held, heads)
    dirs = directions(
        EVAL_DIRECTIONS,
        z_held.shape[1],
        torch.Generator().manual_seed(EVAL_SEED),
    ).to(dev)
    gen = torch.Generator().manual_seed(EVAL_SEED)

    def mean_sigreg(batches: list[Tensor]) -> float:
        return sum(float(sigreg(b, b[:0], dirs)) for b in batches) / len(batches)

    b = len(vs.heldin[0])
    gauss = [torch.randn(b, z_held.shape[1], generator=gen) for _ in vs.heldin]
    row |= {
        "sigreg_heldin": mean_sigreg([targets(mods, x, heads)[1] for x in vs.heldin]),
        "sigreg_val": mean_sigreg(list(z_held.split(b))),
        "sigreg_gauss": mean_sigreg([g.to(dev) for g in gauss]),
        "rank": effective_rank(z_held),
        "participation": float(participation(z_held)),
        "token_r2": float(token_r2(t_held[0::2], t_held[1::2])),
        "time_r2": float(time_r2(t_held, times)),
        "t_var": float(t_held.double().var(0).mean()),
        "z_var": float(z_held.double().var(0).mean()),
    }
    return row
