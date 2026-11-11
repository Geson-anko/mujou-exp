r"""The masks of the context views (I-JEPA's blocks; V-JEPA's tubes for
$N > 1$) and the views they make of a chunk.

A mask is ``(n, gh, gw)`` bool over the grid of patches, True where the patch
is masked. ``block_masks`` draws each as the union of ``blocks`` rectangles of
about ``ratio * T / blocks`` patches each (``T = gh gw``), of aspect ratio
(height / width) log-uniform in ``ASPECT``, at uniform positions: a mask covers
at most about ``ratio`` of the grid (the blocks may overlap). A token is a tube
of all $N$ frames of its patch (``vit.patchify``), so the same mask covers the
patch in every frame: the context cannot copy a masked patch from a frame
next to it.

The packed layout (M17) draws each context view by ``multiblock_masks`` (V-JEPA's
multi-block mask of a ``schema.vision.MaskConf``, made to keep exactly
``expected_keep`` patches visible, so that every sample has the same shape)
and packs a stack and its views into one sequence (``pack_views``).
"""

import math
from collections.abc import Sequence

import torch
from torch import Tensor

from exp.video.schema.vision import MaskConf

__all__ = [
    "ASPECT",
    "block_masks",
    "block_sides",
    "expected_keep",
    "make_views",
    "multiblock_masks",
    "pack_views",
    "paint",
]

ASPECT = (0.75, 1.5)  # the range of a block's height / width


def block_sides(
    aspect: Tensor, grid: tuple[int, int], *, ratio: float, blocks: int
) -> tuple[Tensor, Tensor]:
    r"""The height and width of blocks of aspect ratio ``aspect`` (any shape):
    $h = \mathrm{round}(\sqrt{a s})$, $w = \mathrm{round}(\sqrt{a / s})$ with
    $a$ = ``ratio * T / blocks``, each clamped to ``[1, side]``. Long tensors of
    the shape of ``aspect``."""
    gh, gw = grid
    area = ratio * gh * gw / blocks
    h = torch.sqrt(area * aspect).round().clamp(1, gh).long()
    w = torch.sqrt(area / aspect).round().clamp(1, gw).long()
    return h, w


def paint(
    top: Tensor, left: Tensor, h: Tensor, w: Tensor, grid: tuple[int, int]
) -> Tensor:
    """The union of the rectangles ``[top, top + h) x [left, left + w)``
    (each ``(n, blocks)``) as masks ``(n, gh, gw)``."""
    gh, gw = grid
    r = torch.arange(gh, device=top.device).view(1, 1, gh, 1)
    c = torch.arange(gw, device=top.device).view(1, 1, 1, gw)
    t, lf = top[..., None, None], left[..., None, None]
    inside = (r >= t) & (r < t + h[..., None, None])
    inside = inside & (c >= lf) & (c < lf + w[..., None, None])
    return inside.any(1)


def block_masks(
    n: int,
    grid: tuple[int, int],
    *,
    ratio: float,
    blocks: int,
    generator: torch.Generator,
) -> Tensor:
    """``n`` masks ``(n, gh, gw)`` of ``blocks`` blocks (module docstring),
    drawn from ``generator`` (on its device): the aspect ratios, then the
    rows and the columns of the corners."""
    gh, gw = grid
    dev = generator.device
    lo, hi = math.log(ASPECT[0]), math.log(ASPECT[1])
    u = torch.rand(n, blocks, generator=generator, device=dev)
    h, w = block_sides(torch.exp(lo + (hi - lo) * u), grid, ratio=ratio, blocks=blocks)
    top = (torch.rand(n, blocks, generator=generator, device=dev) * (gh - h + 1)).long()
    left = (
        torch.rand(n, blocks, generator=generator, device=dev) * (gw - w + 1)
    ).long()
    return paint(top, left, h, w, grid)


def make_views(chunks: Tensor, masks: Tensor, patch: int) -> tuple[Tensor, Tensor]:
    """The views of the chunks ``(B, N, C, H, W)`` with the context masks
    ``(B, V - 1, gh, gw)``: view 0 the target (no mask), view ``v`` the chunk
    with the patches of mask ``v - 1`` set to 0 in every frame. Each view gets
    the mask as an extra channel (1 where masked, the same in every frame).

    Returns:
        ``h`` ``(B V, N, C + 1, H, W)`` and the masks of the tokens ``y``
        ``(B V, T)`` float (0 for the target view), the views of a chunk next
        to each other (sample ``b V + v``).
    """
    b, f, c, hh, ww = chunks.shape
    views = masks.shape[1] + 1
    none = masks.new_zeros(b, 1, *masks.shape[2:])
    m = torch.cat([none, masks], 1)  # (B, V, gh, gw)
    pixels = m.repeat_interleave(patch, -2).repeat_interleave(patch, -1)
    pixels = pixels[:, :, None, None].float()  # (B, V, 1, 1, H, W)
    x = chunks[:, None] * (1 - pixels)  # (B, V, N, C, H, W)
    channel = pixels.expand(b, views, f, 1, hh, ww)
    h = torch.cat([x, channel], 3).flatten(0, 1)
    return h, m.flatten(2).float().flatten(0, 1)


def _union(
    n: int, grid: tuple[int, int], mask: MaskConf, generator: torch.Generator
) -> Tensor:
    """``n`` unions ``(n, gh, gw)`` of ``mask.blocks`` rectangles of
    ``mask.scale`` of the grid each, aspect log-uniform in ``mask.aspect``, at
    uniform positions (drawn as in ``block_masks``)."""
    gh, gw = grid
    dev = generator.device
    b = mask.blocks
    lo, hi = math.log(mask.aspect[0]), math.log(mask.aspect[1])
    u = torch.rand(n, b, generator=generator, device=dev)
    h, w = block_sides(
        torch.exp(lo + (hi - lo) * u), grid, ratio=mask.scale * b, blocks=b
    )
    top = (torch.rand(n, b, generator=generator, device=dev) * (gh - h + 1)).long()
    left = (torch.rand(n, b, generator=generator, device=dev) * (gw - w + 1)).long()
    return paint(top, left, h, w, grid)


def multiblock_masks(
    n: int,
    grid: tuple[int, int],
    mask: MaskConf,
    *,
    keep: int,
    generator: torch.Generator,
) -> Tensor:
    r"""``n`` masks ``(n, gh, gw)`` bool (True: masked) of the context view
    ``mask``, each with exactly ``keep`` visible patches: the union of its
    blocks (``_union``), then, among the patches, the ``keep`` of the largest
    score $2 [\text{visible}] + u$, $u \sim U[0, 1)$, stay visible: random
    visible ones are masked too when the union leaves more than ``keep``, and
    random masked ones shown when it leaves fewer (V-JEPA's collator evens the
    counts of a batch for the same reason). Drawn from ``generator``: the
    blocks, then the scores."""
    gh, gw = grid
    union = _union(n, grid, mask, generator).flatten(1)  # (n, T_s)
    u = torch.rand(n, gh * gw, generator=generator, device=generator.device)
    score = 2 * (~union).float() + u
    shown = score.topk(keep, 1).indices
    out = torch.ones_like(union)
    out.scatter_(1, shown, False)
    return out.view(n, gh, gw)


def expected_keep(
    grid: tuple[int, int], mask: MaskConf, *, draws: int = 4096, seed: int = 0
) -> int:
    """The visible patches of the context view ``mask`` (``multiblock_masks``'s
    ``keep``): the mean visible count of ``draws`` unions of its blocks drawn
    from a CPU generator of ``seed``, rounded, within ``[1, T_s - 1]``."""
    g = torch.Generator().manual_seed(seed)
    visible = (~_union(draws, grid, mask, g)).flatten(1).sum(1).double().mean()
    return int(min(max(round(float(visible)), 1), grid[0] * grid[1] - 1))


def pack_views(
    chunks: Tensor, masks: Sequence[Tensor], *, patch: int, tubelet: int
) -> tuple[Tensor, Tensor]:
    """The packed sequences of the stacks ``chunks`` ``(B, N, C, H, W)`` (u8,
    or float in ``[0, 1]``) with the context masks ``masks`` (one ``(B, gh,
    gw)`` per context view, each of a fixed count of visible patches, the same
    in every time slot: a tube).

    Returns:
        ``h`` ``(B, L, P + 1)``: the target view's $T$ tubelets
        (``vit.patchify_tubelets``) in id order, then the visible tubelets of
        each view in id order, the last column each token's id (a float), and
        ``y`` ``(B, (V - 1) T)``: per view, 1 at the tokens to predict.
    """
    from exp.video.jepa.vit import patchify_tubelets  # vit imports this module

    x = chunks.float() / 255 if chunks.dtype == torch.uint8 else chunks.float()
    b, f = x.shape[:2]
    tokens = patchify_tubelets(x, patch, tubelet)  # (B, T, P)
    t = tokens.shape[1]
    slots = f // tubelet
    ids = torch.arange(t, device=x.device).expand(b, t)
    parts = [torch.cat([tokens, ids[..., None].to(x)], 2)]
    ys = []
    for m in masks:
        masked = m.flatten(1).repeat(1, slots)  # (B, T): the tube
        keep = t - int(masked[0].sum())  # the same in every row
        visible = masked.float().sort(dim=1, stable=True).indices[:, :keep]  # id order
        picked = tokens.gather(1, visible[..., None].expand(-1, -1, tokens.shape[2]))
        parts.append(torch.cat([picked, visible[..., None].to(x)], 2))
        ys.append(masked.float())
    return torch.cat(parts, 1), torch.cat(ys, 1)
