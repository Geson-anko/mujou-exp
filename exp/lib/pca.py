r"""The principal components of samples (a code of $d$ numbers that needs no
learning, the yardstick of a learned code of the same $d$).

``fit_pca`` finds the mean $\mu$ and the top ``d`` principal directions $W$
``(d, F)`` (orthonormal rows) of ``x`` ``(n, F)`` by a randomized SVD of the
centered samples; ``encode`` is $z = (x - \mu) W^\top$ and ``decode`` $\hat x =
z W + \mu$, the projection onto the affine subspace that is the best of rank
``d`` in squared error (Eckart-Young) up to the randomized SVD's error.
"""

import torch
from torch import Tensor

__all__ = ["decode", "encode", "fit_pca"]


def fit_pca(
    x: Tensor, d: int, *, gen: torch.Generator, niter: int = 4
) -> tuple[Tensor, Tensor]:
    """The mean ``(F,)`` and the top ``d`` principal directions ``(d, F)`` of
    ``x`` ``(n, F)`` (float), by ``torch.svd_lowrank`` with ``q = d + 16``
    and ``niter`` power iterations, its random draws seeded from ``gen`` (a
    CPU generator; the global RNGs are left as they were).

    Raises:
        ValueError: ``d`` is not in ``[1, min(n, F)]``.
    """
    n, f = x.shape
    if not 1 <= d <= min(n, f):
        raise ValueError(f"d must be in [1, {min(n, f)}], not {d}")
    seed = int(torch.randint(2**62, (1,), generator=gen))
    mean = x.mean(0)
    devices = [x.device] if x.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        _, _, v = torch.svd_lowrank(x - mean, q=min(d + 16, n, f), niter=niter)
    return mean, v[:, :d].T.contiguous()


def encode(x: Tensor, mean: Tensor, comps: Tensor) -> Tensor:
    """The codes ``(n, d)`` of ``x`` ``(n, F)``: $(x - \\mu) W^\\top$."""
    return (x - mean) @ comps.T


def decode(z: Tensor, mean: Tensor, comps: Tensor) -> Tensor:
    """The samples ``(n, F)`` of the codes ``z`` ``(n, d)``: $z W + \\mu$."""
    return z @ comps + mean
