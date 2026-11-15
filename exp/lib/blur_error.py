r"""A reconstruction error that looks at the structure of an image, not its
exact pixels: the squared error after a Gaussian blur, for evaluation only.

For images $\hat x, x$ ``(..., H, W)`` and the Gaussian $G_\sigma$ (a
separable kernel of radius $\lceil 3\sigma \rceil$, normalized to sum 1,
borders reflected),

$$
e_\sigma(\hat x, x) = \operatorname{mean}\big( (G_\sigma * (\hat x - x))^2 \big),
\qquad
\mathrm{recon}_\sigma = \frac{e_\sigma(\hat x, x)}{e_\sigma(\bar x, x)}
$$

with $\bar x$ the mean image: 1 is the mean image, 0 the true frame, and
$\sigma = 0$ the per-pixel squared error. On a set of pixels $\Omega$ (the
masked region of an image), the mean is over $\Omega$ alone, of the error
blurred over the whole image (``blur_mse_masked``, ``blur_recon_masked``):
it reads the pixels within the kernel's radius of $\Omega$, no others, and
is ``blur_mse`` at $\Omega$ every pixel. The blur is linear, so a cosine of
angular frequency $\omega$ is scaled by the kernel's response
$K(\omega) = \sum_j k_j \cos(\omega j)$ and its error by $K(\omega)^2$: the
error of a shift by one pixel, $2(1 - \cos\omega)$ times the energy of the
image at $\omega$, counts only the frequencies the blur keeps, where
$2(1 - \cos\omega) \approx \omega^2$ is small.

No training loss may use it (M17 learns no reconstruction term): it is a
yardstick of decoded images.
"""

import math

import torch
from torch import Tensor
from torch.nn import functional as F

__all__ = [
    "SIGMA",
    "blur_mse",
    "blur_mse_masked",
    "blur_recon",
    "blur_recon_masked",
    "gaussian_blur",
    "gaussian_kernel",
]

SIGMA = 2.0  # pixels, for 128 x 128 frames (memory/agents/recon-metric/)


def gaussian_kernel(sigma: float, dtype: torch.dtype = torch.float32) -> Tensor:
    """The 1-d Gaussian ``(2r + 1,)``, $r = \\lceil 3\\sigma \\rceil$, summing
    to 1; ``[1]`` at ``sigma`` 0."""
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")
    r = math.ceil(3 * sigma)
    if r == 0:
        return torch.ones(1, dtype=dtype)
    j = torch.arange(-r, r + 1, dtype=torch.float64)
    k = torch.exp(-(j**2) / (2 * sigma**2))
    return (k / k.sum()).to(dtype)


def gaussian_blur(x: Tensor, sigma: float) -> Tensor:
    """``x`` ``(..., H, W)`` blurred by $G_\\sigma$ along both axes, the borders
    reflected; ``x`` itself at ``sigma`` 0."""
    k = gaussian_kernel(sigma, x.dtype).to(x.device)
    r = len(k) // 2
    if r == 0:
        return x
    h, w = x.shape[-2:]
    if r >= min(h, w):
        raise ValueError(f"the kernel radius {r} needs images over {r} pixels")
    y = x.reshape(-1, 1, h, w)
    y = F.conv2d(F.pad(y, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    y = F.conv2d(F.pad(y, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))
    return y.view(x.shape)


def blur_mse(a: Tensor, b: Tensor, sigma: float = SIGMA) -> Tensor:
    """$e_\\sigma(a, b)$: the mean squared value of the blurred difference of
    ``a`` and ``b`` ``(..., H, W)``, a scalar."""
    return (gaussian_blur(a - b, sigma) ** 2).mean()


def blur_recon(x_hat: Tensor, x: Tensor, mean: Tensor, sigma: float = SIGMA) -> float:
    """$\\mathrm{recon}_\\sigma$: $e_\\sigma$ of the reconstructions ``x_hat``
    against the frames ``x`` ``(n, C, H, W)``, over that of the mean image
    ``mean`` ``(C, H, W)`` (1 is the mean image)."""
    return float(blur_mse(x_hat, x, sigma) / blur_mse(mean.expand_as(x), x, sigma))


def blur_mse_masked(a: Tensor, b: Tensor, mask: Tensor, sigma: float = SIGMA) -> Tensor:
    """$e_\\sigma$ on the pixels ``mask`` ``(n, H, W)`` bool: the mean squared
    value of the blurred difference of ``a`` and ``b`` ``(n, C, H, W)`` over
    the channels of the masked pixels, a scalar (not finite without one)."""
    w = mask[:, None].to(a)
    err = gaussian_blur(a - b, sigma) ** 2
    return (err * w).sum() / (w.sum() * a.shape[1])


def blur_recon_masked(
    x_hat: Tensor, x: Tensor, mean: Tensor, mask: Tensor, sigma: float = SIGMA
) -> float:
    """``blur_recon`` on the pixels ``mask`` ``(n, H, W)`` bool:
    ``blur_mse_masked`` of the reconstructions ``x_hat`` against the frames
    ``x`` ``(n, C, H, W)``, over that of the mean image ``mean`` ``(C, H,
    W)``."""
    err = blur_mse_masked(x_hat, x, mask, sigma)
    return float(err / blur_mse_masked(mean.expand_as(x), x, mask, sigma))
