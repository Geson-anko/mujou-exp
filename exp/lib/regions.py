r"""The evaluation's masks of the unobserved parts (V18c's E1, E1e, E1g;
``exp/vision/plans/jepa.md``, 「未観測の部分: 評価のマスクと領域」), on the
coarse 4 x 4 grid (row $a$, column $b$, the position $4a + b$; True hidden),
and the images of their regions.

- ``Q_tl``, ``Q_tr``, ``Q_bl``, ``Q_br``: a quarter (2 x 2), $k$ = 4
- ``H_l``, ``H_r``, ``H_t``, ``H_b``: a half, $k$ = 8
- ``R``: the rim (all but the centre $a, b \in \{1, 2\}$), $k$ = 12
- ``S_1`` = ``Q_tl`` $\cup$ ``Q_br``, ``S_2`` = ``Q_tr`` $\cup$ ``Q_bl``:
  two diagonal quarters (``SWAP_PAIRS``), $k$ = 8

A region's image (``region_image``) is its pixels' bounding box with the
pixels of the box outside the region grey (u8 128). ``harmonic_fill`` fills
the hidden pixels $U$ of a frame by the discrete Laplace equation with the
visible $V$ as its boundary: with $L$ the Laplacian of the 4-neighbour graph
of the pixels (at the frame's edges only the neighbours there: Neumann's
condition), $x_U = -L_{UU}^{-1} L_{UV} x_V$, so each filled pixel is the
mean of its neighbours.

``transplant`` puts a region's pixels of one stack into another (V18c's
locality of the self-space identification): the intervention leaves every
pixel natural and changes only which patches sit together.
"""

import torch
from torch import Tensor

__all__ = [
    "COARSE",
    "EVAL_MASKS",
    "GREY",
    "SWAP_PAIRS",
    "harmonic_fill",
    "harmonic_matrix",
    "pixel_mask",
    "region_image",
    "transplant",
]

COARSE = 4  # the masks' grid
GREY = 128


def _cells(rows: set[int], cols: set[int]) -> Tensor:
    m = torch.zeros(COARSE, COARSE, dtype=torch.bool)
    for a in rows:
        for b in cols:
            m[a, b] = True
    return m.flatten()


_TOP, _BOTTOM, _LEFT, _RIGHT, _ALL = {0, 1}, {2, 3}, {0, 1}, {2, 3}, {0, 1, 2, 3}

EVAL_MASKS: dict[str, Tensor] = {
    "Q_tl": _cells(_TOP, _LEFT),
    "Q_tr": _cells(_TOP, _RIGHT),
    "Q_bl": _cells(_BOTTOM, _LEFT),
    "Q_br": _cells(_BOTTOM, _RIGHT),
    "H_l": _cells(_ALL, _LEFT),
    "H_r": _cells(_ALL, _RIGHT),
    "H_t": _cells(_TOP, _ALL),
    "H_b": _cells(_BOTTOM, _ALL),
    "R": ~_cells({1, 2}, {1, 2}),
}
EVAL_MASKS["S_1"] = EVAL_MASKS["Q_tl"] | EVAL_MASKS["Q_br"]
EVAL_MASKS["S_2"] = EVAL_MASKS["Q_tr"] | EVAL_MASKS["Q_bl"]
SWAP_PAIRS = {"S_1": ("Q_tl", "Q_br"), "S_2": ("Q_tr", "Q_bl")}


def pixel_mask(mask: Tensor, size: int) -> Tensor:
    """The coarse mask ``(16,)`` as the pixels ``(size, size)`` bool of a
    frame: each cell ``size / 4`` pixels a side."""
    s = size // COARSE
    grid = mask.reshape(COARSE, COARSE)
    return grid.repeat_interleave(s, 0).repeat_interleave(s, 1)


def region_image(frames: Tensor, pixels: Tensor) -> Tensor:
    """The images u8 ``(B, 3, h, w)`` of the region ``pixels`` ``(H, W)``
    bool of the frames u8 ``(B, 3, H, W)``: the bounding box of the region,
    its pixels outside the region ``GREY``."""
    rows, cols = pixels.any(1).nonzero().flatten(), pixels.any(0).nonzero().flatten()
    r0, r1, c0, c1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
    box = frames[:, :, r0:r1, c0:c1].clone()
    box[:, :, ~pixels[r0:r1, c0:c1]] = GREY
    return box


def harmonic_matrix(pixels: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    r"""The hidden pixels $U$ and the visible $V$ (flat indices, row-major)
    of ``pixels`` ``(H, W)`` bool (True hidden) and $-L_{UU}^{-1} L_{UV}$
    ``(|U|, |V|)`` float64 (module docstring)."""
    h, w = pixels.shape
    n = h * w
    idx = torch.arange(n).reshape(h, w)
    lap = torch.zeros(n, n, dtype=torch.float64)
    for a, b in ((idx[:, :-1], idx[:, 1:]), (idx[:-1], idx[1:])):  # the edges
        a, b = a.flatten(), b.flatten()
        lap[a, b] = lap[b, a] = -1.0
    lap -= torch.diag(lap.sum(1))  # the degree on the diagonal
    flat = pixels.flatten()
    u, v = flat.nonzero().flatten(), (~flat).nonzero().flatten()
    m = -torch.linalg.solve(lap[u][:, u], lap[u][:, v])
    return u, v, m


def harmonic_fill(frames: Tensor, pixels: Tensor) -> Tensor:
    """The frames ``(B, C, H, W)`` with the hidden pixels of ``pixels``
    ``(H, W)`` filled by the harmonic interpolation of the visible ones, in
    float64 (``harmonic_matrix``)."""
    u, v, m = harmonic_matrix(pixels)
    x = frames.double().flatten(2)  # (B, C, H W)
    out = x.clone()
    out[:, :, u] = x[:, :, v] @ m.T.to(x.device)
    return out.reshape(frames.shape)


def transplant(dst: Tensor, src: Tensor, pixels: Tensor) -> Tensor:
    """The stacks u8 ``dst`` ``(B, n, 3, S, S)`` with the pixels ``pixels``
    ``(S, S)`` bool (True taken) of every frame those of ``src``; ``dst``
    and ``src`` broadcast (one stack of either to a batch of the other)."""
    return torch.where(pixels.to(dst.device), src, dst)
