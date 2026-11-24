r"""A fixed random linear map: $W \in \mathbb{R}^{d_{\mathrm{out}} \times
d_{\mathrm{in}}}$ of i.i.d. $N(0, 1/d_{\mathrm{in}})$ entries, drawn from a
seed and never learned (V18c's $z_W = W \cdot \mathrm{vec}(h)$).

For a fixed $v$, each coordinate of $W v$ is $N(0, \lVert v \rVert^2 /
d_{\mathrm{in}})$, so $\frac{1}{d_{\mathrm{out}}} \lVert W v \rVert^2$ is
$\lVert v \rVert^2 / d_{\mathrm{in}}$ times a $\chi^2_{d_{\mathrm{out}}} /
d_{\mathrm{out}}$ variable.
"""

import math

import torch
from torch import Tensor

__all__ = ["gaussian_projection", "project"]


def gaussian_projection(d_out: int, d_in: int, seed: int) -> Tensor:
    r"""$W$ ``(d_out, d_in)`` float32 on the CPU: ``torch.randn(d_out, d_in)``
    of a CPU generator seeded ``seed``, drawn at once, times $1 /
    \sqrt{d_{\mathrm{in}}}$. The same for every caller of the same
    arguments (the global RNG is not used)."""
    gen = torch.Generator().manual_seed(seed)
    w = torch.randn(d_out, d_in, generator=gen)
    return w.mul_(1 / math.sqrt(d_in))


def project(v: Tensor, w: Tensor, chunk: int | None = None) -> Tensor:
    r"""$v W^\top$ of the rows ``v`` ``(B, d_in)`` by ``w`` ``(d_out,
    d_in)``: ``(B, d_out)``. With ``chunk``, the sum over the input's
    coordinates is made of the partial products of ``chunk`` columns at a
    time, added in order (less memory at once; the rounding differs from the
    single product's)."""
    if chunk is None:
        return v @ w.T
    out = v.new_zeros(v.shape[0], w.shape[0])
    for i in range(0, w.shape[1], chunk):
        out += v[:, i : i + chunk] @ w[:, i : i + chunk].T
    return out
