r"""The completion error of the packed layout (M17): how well the predictor
completes the masked tokens of a stack from its visible ones, on held-out
data, made scale-free.

A mask kind hides a set $\Omega$ of (stack, token) pairs: ``short`` and
``long`` (the context views of ``vision.context_masks``, tubes of spatial
blocks, as in training) and ``future`` (``future_tokens``: every token of
the last ``future_slots`` time slots, the future of the stack). As in
training, the context (the visible tokens only) goes through the encoder and
the predictor, and the target tokens $s$ are those of the whole stack through
the same weights (``VisionModules.tokens``). For a predictor $\hat s$,

$$
E = \frac{\sum_{(i,t) \in \Omega} \lVert \hat s_{i,t} - s_{i,t} \rVert^2}
         {\sum_{(i,t) \in \Omega} \lVert s_{i,t} - \bar s_\Omega \rVert^2}
  = 1 - R^2,
$$

$\bar s_\Omega$ the mean of the masked target tokens over the set (per
dimension). Scaling every token by $c$ leaves $E$ as it is, so shrinking the
tokens cannot lower it (the loss ``pred`` can); predicting $\bar s_\Omega$ is
1, a perfect prediction 0. The same normalization gives the trivial
predictors: ``_vis``, the mean of the visible target tokens of the same stack
(it reads the target view, whose visible tokens attended to the masked ones:
a lenient bar), and ``_mean``, a fixed token, the per-dimension mean of the
target tokens of a reference set (the held-in stacks of the training videos;
near 1 on the reference itself, above 1 by how far the held-out set's mean
is from it).

$E$ is normalized by the spread of the masked tokens over the whole set,
which counts the spread between the stacks too: tokens that collapse to a
constant per stack (every token of a stack the same) still have it, and the
mean of a stack's visible tokens then predicts its masked ones almost
exactly. The error within the stacks (M17 audit, 2026-11-12),

$$
E^{\mathrm{n}} = \frac{\sum_{(i,t) \in \Omega} \lVert \hat s_{i,t} - s_{i,t} \rVert^2}
         {\sum_{(i,t) \in \Omega} \lVert s_{i,t} - \bar s_{i,\Omega} \rVert^2},
$$

$\bar s_{i,\Omega}$ the mean of the masked target tokens of stack $i$,
counts only what is to be completed within a stack: it grows without bound
as the tokens of a stack collapse, and predicting each stack's own mean of
its masked tokens is 1 (``ncomp_<kind>``; also for the trivial predictors).
"""

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor

from exp.video.jepa.loss import view_ids
from exp.video.jepa.masks import view_tokens
from exp.video.jepa.vit import VisionModules, patchify_tubelets

__all__ = [
    "FUTURE",
    "Unexplained",
    "complete",
    "completion",
    "future_slots",
    "future_tokens",
    "mean_token",
    "tube_tokens",
]

FUTURE = "future"


def future_slots(slots: int) -> int:
    """The last time slots the ``future`` kind hides of ``slots``: a quarter
    of them, at least one; 0 (no ``future`` kind) with one slot, which would
    leave nothing visible."""
    return 0 if slots < 2 else max(1, slots // 4)


def future_tokens(n: int, grid3: tuple[int, int, int], hidden: int) -> Tensor:
    """The mask ``(n, T)`` float (1: masked) of every token of the last
    ``hidden`` time slots, in the id order (time slot, row, column)."""
    slots, gh, gw = grid3
    per = gh * gw
    m = torch.zeros(slots * per)
    m[(slots - hidden) * per :] = 1
    return m.expand(n, -1).clone()


def tube_tokens(mask: Tensor, slots: int) -> Tensor:
    """The mask ``(n, T)`` float of the spatial ``mask`` ``(n, gh, gw)`` in
    every time slot (a tube, as ``masks.pack_views``)."""
    return view_tokens(mask, slots)


class Unexplained:
    r"""$E$ of the module docstring of several predictors of the same target
    tokens, summed over batches (float64): ``add`` a batch of targets ``(n,
    M, D)`` and each predictor's prediction (broadcast to it), then
    ``values`` ($E$) or ``within`` ($E^{\mathrm{n}}$, the spread within each
    of the ``n`` samples). Not finite when the targets do not vary."""

    def __init__(self, names: Sequence[str]) -> None:
        self._sse = dict.fromkeys(names, 0.0)
        self._sum: Tensor | None = None
        self._sq = 0.0
        self._n = 0
        self._within = 0.0

    def add(self, target: Tensor, preds: Mapping[str, Tensor]) -> None:
        t = target.double()
        self._within += float(((t - t.mean(-2, keepdim=True)) ** 2).sum())
        s = t.flatten(0, -2)  # (n M, D)
        for name, p in preds.items():
            diff = p.double().expand_as(target).flatten(0, -2) - s
            self._sse[name] += float((diff**2).sum())
        total = s.sum(0)
        self._sum = total if self._sum is None else self._sum + total
        self._sq += float((s**2).sum())
        self._n += len(s)

    def values(self) -> dict[str, float]:
        if self._sum is None:
            return {k: float("nan") for k in self._sse}
        var = self._sq - float((self._sum**2).sum()) / self._n
        out = {}
        for k, sse in self._sse.items():
            out[k] = sse / var if var > 0 else float("nan")
        return out

    def within(self) -> dict[str, float]:
        within = self._within
        return {
            k: sse / within if within > 0 else float("nan")
            for k, sse in self._sse.items()
        }


@torch.no_grad()
def complete(
    mods: VisionModules, chunks: Tensor, target: Tensor, masked: Tensor, keep: int
) -> tuple[Tensor, Tensor, Tensor]:
    """The prediction of the masked tokens of the stacks ``chunks`` ``(n, N,
    C, H, W)`` (u8 or in ``[0, 1]``) with the token mask ``masked`` ``(n,
    T)`` (1: masked, ``keep`` visible in every row): the context through the
    encoder (``segments`` ``(keep,)``) and the predictor, as in training.

    Args:
        mods: The modules of the packed layout.
        chunks: The stacks.
        target: Their target tokens ``(n, T, D)`` (``mods.tokens``).
        masked: The mask.
        keep: The visible tokens of a row.

    Returns:
        The predictions ``(n, M, D)``, the target tokens at the masked ids
        ``(n, M, D)``, and the mean of the visible target tokens ``(n, D)``.
    """
    x = chunks.float() / 255 if chunks.dtype == torch.uint8 else chunks.float()
    pixels = patchify_tubelets(x, mods.patch, mods.tubelet)  # (n, T, P)
    ctx_ids, tgt_ids = view_ids(masked.to(x.device), keep)
    picked = pixels.gather(1, ctx_ids[..., None].expand(-1, -1, pixels.shape[2]))
    h = torch.cat([picked, ctx_ids[..., None].to(picked)], 2)
    pred = mods.predictor(mods.encoder(h, (keep,)), ctx_ids, tgt_ids)
    d = target.shape[2]
    goal = target.gather(1, tgt_ids[..., None].expand(-1, -1, d))
    visible = target.gather(1, ctx_ids[..., None].expand(-1, -1, d)).mean(1)
    return pred, goal, visible


@torch.no_grad()
def mean_token(
    mods: VisionModules, stacks: Tensor, *, batch: int, device: torch.device
) -> Tensor:
    """The per-dimension mean ``(D,)`` of all target tokens of the stacks
    ``stacks`` ``(n, N, C, H, W)`` (u8, CPU): the fixed token of the
    ``_mean`` predictor."""
    total, count = None, 0
    for i in range(0, len(stacks), batch):
        target = mods.tokens(stacks[i : i + batch].to(device))  # (b, T, D)
        s = target.double().sum((0, 1))
        total = s if total is None else total + s
        count += target.shape[0] * target.shape[1]
    assert total is not None, "no stacks"
    return (total / count).float()


@torch.no_grad()
def completion(
    mods: VisionModules,
    stacks: Tensor,
    kinds: Mapping[str, tuple[Tensor, int]],
    *,
    mean_token: Tensor | None,
    batch: int,
    device: torch.device,
) -> dict[str, float]:
    r"""$E$ of each mask kind on the stacks ``stacks`` ``(n, N, C, H, W)``
    (u8, CPU), a batch of ``batch`` at a time on ``device``.

    Args:
        mods: The modules of the packed layout.
        stacks: The stacks.
        kinds: Per kind, the token mask ``(n, T)`` and the visible tokens of
            a row.
        mean_token: The fixed token ``(D,)`` of the ``_mean`` predictor
            (``mean_token`` of a reference set), or None (no ``_mean``).
        batch: The stacks of a batch.
        device: Where to compute.

    Returns:
        ``comp_<kind>`` (the predictor), ``comp_<kind>_vis`` and, with
        ``mean_token``, ``comp_<kind>_mean``; and the same of
        $E^{\mathrm{n}}$ as ``ncomp_<kind>``, ``ncomp_<kind>_vis``,
        ``ncomp_<kind>_mean``.
    """
    names = ["pred", "vis", *(("mean",) if mean_token is not None else ())]
    acc = {k: Unexplained(names) for k in kinds}
    for i in range(0, len(stacks), batch):
        x = stacks[i : i + batch].to(device)
        target = mods.tokens(x)  # (b, T, D)
        for kind, (masked, keep) in kinds.items():
            pred, goal, visible = complete(mods, x, target, masked[i : i + batch], keep)
            preds = {"pred": pred, "vis": visible[:, None]}
            if mean_token is not None:
                preds["mean"] = mean_token.to(goal)
            acc[kind].add(goal, preds)
    out: dict[str, float] = {}
    for kind, a in acc.items():
        for key, values in (("comp", a.values()), ("ncomp", a.within())):
            out[f"{key}_{kind}"] = values["pred"]
            for name in names[1:]:
                out[f"{key}_{kind}_{name}"] = values[name]
    return out
