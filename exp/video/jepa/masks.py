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
"""

import math

import torch
from torch import Tensor

__all__ = ["ASPECT", "block_masks", "block_sides", "make_views", "paint"]

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
