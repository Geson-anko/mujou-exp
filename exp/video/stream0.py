r"""The stream of stage 0: a linear regression whose inputs drift in time.

$x_k = \mu_k + \sigma_\xi \xi_k$, where the mean $\mu_k$ is an
Ornstein-Uhlenbeck process of correlation time $\tau_x$ and stationary
standard deviation $\sigma_\mu$, stepped exactly over each frame:

$$
\mu_{k+1} = e^{-\Delta t/\tau_x} \mu_k
    + \sigma_\mu \sqrt{1 - e^{-2\Delta t/\tau_x}}\ \epsilon_k,
$$

and $y_k = w_*^\top x_k + \sigma_y \eta_k$, with $w_* \sim N(0, I/d)$ fixed.
The i.i.d. stream draws each $\mu_k$ anew from the stationary $N(0,
\sigma_\mu^2 I)$: the same marginal distribution, no correlation in time.
"""

import math

import torch

__all__ = ["LinearStream"]


class LinearStream:
    """The frames ``(x, y)`` of the regression, one per ``next()``.

    Args:
        d: The dimension of ``x``.
        tau_x: The correlation time of the mean, in seconds.
        frame: The time between two frames, in seconds.
        noise: The standard deviation of the noise of ``y``.
        mean_std: The stationary standard deviation of each element of the mean.
        jitter: The standard deviation of ``x`` around its mean.
        iid: Draw each mean anew (no correlation in time).
        seed: Seeds ``w_*``, the first mean and every draw.
    """

    def __init__(
        self,
        *,
        d: int,
        tau_x: float,
        frame: float,
        noise: float,
        mean_std: float = 1.0,
        jitter: float = 0.3,
        iid: bool = False,
        seed: int,
    ) -> None:
        self._g = torch.Generator().manual_seed(seed)
        self.w = torch.randn(d, generator=self._g) / math.sqrt(d)  # w_*
        self._decay = math.exp(-frame / tau_x)
        self._kick = mean_std * math.sqrt(-math.expm1(-2 * frame / tau_x))
        self._mean_std, self._jitter, self._noise = mean_std, jitter, noise
        self._iid = iid
        self.mean = mean_std * torch.randn(d, generator=self._g)  # stationary start

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Move the mean by one frame and return ``x`` (``(d,)``) and ``y``
        (``(1,)``)."""
        eps = torch.randn(self.mean.shape, generator=self._g)
        if self._iid:
            self.mean = self._mean_std * eps
        else:
            self.mean = self._decay * self.mean + self._kick * eps
        x = self.mean + self._jitter * torch.randn(self.mean.shape, generator=self._g)
        y = x @ self.w + self._noise * torch.randn((), generator=self._g)
        return x, y.reshape(1)

    def sample_iid(self, n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        """``n`` samples ``(x, y)`` of the marginal distribution, without noise
        on ``y``, from a generator of their own (the stream does not move)."""
        g = torch.Generator().manual_seed(seed)
        d = self.w.shape[0]
        std = math.sqrt(self._mean_std**2 + self._jitter**2)
        x = std * torch.randn(n, d, generator=g)
        return x, (x @ self.w).unsqueeze(1)
