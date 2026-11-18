r"""The loss of the deterministic autoencoder (M18, stage 1).

For a stack $x$ (u8, $P = N C H W$ values) the target is $y = x / 256$ and

$$\ell_{AE}(x) = \tfrac12 \lVert y - g(f(x)) \rVert^2,$$

summed over the $P$ values (not divided by $P$): the negative log-likelihood
of the Gaussian decoder of stage 2 with $\sigma = 1$, without its constant,
the KL and the noise. A batch of $B$ stacks learns their mean
$L = \frac1B \sum_b \ell_{AE}(x_b)$ over the stacks that count, so that $B$ = 1
and $B$ > 1 estimate the same objective (the expected $\ell_{AE}$).
"""

import torch
from torch import Tensor

__all__ = ["LEVELS", "ae_loss", "batch_loss", "stack_mse_sums", "target"]

LEVELS = 256  # the values of an 8-bit pixel: y = x / LEVELS


def target(stacks: Tensor) -> Tensor:
    """$y = x / 256$ ``(n, N, C, H, W)`` float32 of the u8 stacks."""
    return stacks.float() / LEVELS


def ae_loss(y: Tensor, y_hat: Tensor) -> Tensor:
    r"""$\ell_{AE}$ ``(n,)``: $\frac12 \lVert y - \hat y \rVert^2$ of each
    sample of ``y`` and ``y_hat`` ``(n, ...)``, summed over all but the first
    dimension."""
    return 0.5 * (y - y_hat).pow(2).flatten(1).sum(1)


def batch_loss(losses: Tensor, valid: Tensor) -> Tensor:
    r"""$L$: the mean of ``losses`` ``(B,)`` over ``valid`` ``(B,)`` (0 when
    none is valid)."""
    w = valid.to(losses.dtype)
    return (losses * w).sum() / w.sum().clamp(min=1)


def stack_mse_sums(y: Tensor, y_hat: Tensor, mean: Tensor) -> Tensor:
    r"""The sums $\sum \lVert y - \hat y \rVert^2$ and $\sum \lVert y - m
    \rVert^2$ over the stacks ``y``, ``y_hat`` ``(n, N, C, H, W)``, with the
    mean stack ``mean`` ``(N, C, H, W)``: their ratio over a set of stacks is
    ``stack_mse`` (float64 ``(2,)``)."""
    err = (y - y_hat).double().pow(2).sum()
    base = (y - mean).double().pow(2).sum()
    return torch.stack([err, base])
