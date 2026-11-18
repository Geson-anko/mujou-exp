r"""The evaluation of the autoencoder in the run (M18): the squared error of
the stacks over that of the mean stack.

For a set of stacks $x_i$ with targets $y_i = x_i / 256$,

$$\mathrm{stack\_mse} = \frac{\sum_i \lVert y_i - g(f(x_i)) \rVert^2}
{\sum_i \lVert y_i - m \rVert^2},$$

with $m$ the mean stack of the training stacks (``training_moments``: the
``iter_stream_windows`` of M17's PCA, ``WINDOWS`` windows of ``STREAM_STEPS +
1`` stacks of the nine training videos): 1 is the mean stack, 0 a perfect
reconstruction. The held-out stacks are the first stack of each window of
``PackedMeasures`` (``walking_tours.eval_windows`` of ``TOURS_SEED``, the
first third calib, the second val, the last test); for the overfit control
(``data.tours.sampling`` ``"pool"``) the row of val also has the same of the
pool's stacks (``pool_stack_mse``), on which it learns.

The VAE's row (``VAEMeasures``, plans/m18-vae-stream.md 「段 2 の評価と判定」)
has, of the same stacks, its negative ELBO with a fixed noise (``u`` and
$\varepsilon$ drawn from ``EVAL_NOISE_SEED`` the same way at every
evaluation): ``bpd`` ($-\mathcal L / (P \ln 2) + 8$, the objective),
``loss`` ($-\mathcal L$ in nats a stack), the rate ``kl`` (nats a stack) and
``rate_bpd`` (bits a value: ``bpd`` less the distortion), $\sigma$, the
held-out optimum $\sigma^* = (\sum_i \lVert y_i - g(z_i) \rVert^2 /
(n P))^{1/2}$ of the same samples and the nats a stack lost to $\sigma \ne
\sigma^*$ (``sigma_gap``, ``exp.lib.elbo``), the quantiles over the $D$
latents of their mean KL (``kl_q<percent>``; no threshold), and
``stack_mse`` of $g(\mu)$ (the decoder at the posterior mean, against $y = x
/ 256$ as the autoencoder's). The overfit control's row of val also has
``pool_bpd`` and ``pool_stack_mse``.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import override

import torch
from torch import Tensor

from exp.lib.elbo import bits_per_dim, sigma_gap
from exp.video.data.walking_tours import eval_windows
from exp.video.measure.encoder_eval import STREAM_STEPS, WINDOWS, iter_stream_windows
from exp.video.programs.vision import TOURS_SEED
from exp.video.schema.run import RunConf
from exp.video.vae.elbo import elbo_terms
from exp.video.vae.loss import LEVELS, ae_loss, stack_mse_sums, target
from exp.video.vae.modules import AEModules, VAEModules

__all__ = [
    "EVAL_BATCH",
    "EVAL_NOISE_SEED",
    "KL_QUANTILES",
    "SPLITS",
    "AEMeasures",
    "StackMoments",
    "VAEMeasures",
    "elbo_errors",
    "held_out_stacks",
    "stack_errors",
    "training_moments",
]

SPLITS = ("val", "test")
EVAL_BATCH = 32  # the stacks through the autoencoder at once
EVAL_NOISE_SEED = 12345  # the VAE's noise of the evaluations (one per set)
KL_QUANTILES = (0, 10, 25, 50, 75, 90, 100)  # percents over the D latents


@dataclass(frozen=True)
class StackMoments:
    r"""The moments of the training stacks' targets $y$: the mean stack
    ``mean`` $m$ ``(N, C, H, W)`` float64, $\sigma_0^2 = \frac{1}{n P}
    \sum_i \lVert y_i - m \rVert^2$ (``var``: the mean of the values'
    variances, the maximum-likelihood variance of the mean stack's Gaussian)
    and the number of stacks $n$ (``count``)."""

    mean: Tensor
    var: float
    count: int


def training_moments(
    conf: RunConf, root: Path, *, windows: int = WINDOWS, steps: int = STREAM_STEPS
) -> StackMoments:
    """The ``StackMoments`` of the stacks of ``iter_stream_windows``
    (``steps + 1`` stacks a window: stack ``k`` its frames ``k .. k + N -
    1``), read a window at a time and summed in float64."""
    v = conf.vision
    assert v is not None
    n = v.frames
    s1 = s2 = None
    count = 0
    for w in iter_stream_windows(conf, root, windows=windows, steps=steps):
        y = w.double() / LEVELS  # (N + steps, C, H, W)
        k = len(y) - n + 1
        # slot s of the k stacks: frames s .. s + k - 1, summed by a cumulative sum
        c1 = torch.cat([torch.zeros_like(y[:1]), y.cumsum(0)])
        c2 = torch.cat([torch.zeros_like(y[:1]), (y * y).cumsum(0)])
        a1, a2 = c1[k : k + n] - c1[:n], c2[k : k + n] - c2[:n]  # (N, C, H, W)
        s1 = a1 if s1 is None else s1 + a1
        s2 = a2 if s2 is None else s2 + a2
        count += k
    if s1 is None or s2 is None:
        raise ValueError("no training window: the videos are shorter than a window")
    mean = s1 / count
    var = float((s2 / count - mean * mean).sum()) / mean.numel()
    return StackMoments(mean, var, count)


def held_out_stacks(conf: RunConf, root: Path) -> dict[str, Tensor]:
    """The held-out stacks u8 ``(n, N, C, H, W)`` of ``"val"`` and ``"test"``:
    the first stack of each window of ``PackedMeasures`` (module docstring),
    ``measure.clips`` x ``measure.windows`` each."""
    t, v, m = conf.data.tours, conf.vision, conf.measure
    assert t is not None and v is not None
    n = m.clips * m.windows
    u8 = eval_windows(
        root,
        held_out=t.held_out,
        clips=3 * n,
        frames=v.frames,
        stride=v.stride,
        steps=m.steps,
        seed=TOURS_SEED,
    )
    return {s: u8[(i + 1) * n : (i + 2) * n, : v.frames] for i, s in enumerate(SPLITS)}


@torch.no_grad()
def stack_errors(
    mods: AEModules | VAEModules,
    stacks: Tensor,
    mean: Tensor,
    *,
    batch: int = EVAL_BATCH,
) -> tuple[float, float]:
    r"""``stack_mse`` (module docstring) of ``mods`` on the stacks u8 ``(n,
    N, C, H, W)`` (any device) with the mean stack ``mean`` ``(N, C, H,
    W)``, and the mean $\ell_{AE}$ of the stacks; a batch at a time on the
    modules' device."""
    dev = mean.device
    sums = torch.zeros(2, dtype=torch.float64, device=dev)
    loss = torch.zeros((), dtype=torch.float64, device=dev)
    m = mean.float()
    for chunk in stacks.split(batch):
        x = chunk.to(dev)
        y, y_hat = target(x), mods(x)
        sums += stack_mse_sums(y, y_hat, m)
        loss += ae_loss(y, y_hat).double().sum()
    err, base = sums.tolist()
    return err / base, float(loss) / len(stacks)


class AEMeasures:
    """The evaluation of the run (module docstring): the held-out stacks of
    each split, the training stacks' ``moments`` and, for the overfit
    control, the ``pool`` of stacks it learns on.

    Args:
        conf: The run; its vision, Walking Tours, ``measure`` and device.
        root: The local copy of the data.
        pool: The pool's stacks u8 ``(n, N, C, H, W)``, or None.
    """

    def __init__(self, conf: RunConf, root: Path, *, pool: Tensor | None) -> None:
        self._dev = torch.device(conf.device)
        self.stacks = held_out_stacks(conf, root)
        self.moments = training_moments(conf, root)
        self._mean = self.moments.mean.float().to(self._dev)
        self._pool = pool

    def evaluate(self, mods: AEModules | VAEModules, split: str) -> dict[str, float]:
        """The row of ``split``: ``stack_mse`` and ``loss`` (the mean
        $\\ell_{AE}$) of its stacks and, in the row of val of the overfit
        control, ``pool_stack_mse`` and ``pool_loss``."""
        mse, loss = stack_errors(mods, self.stacks[split], self._mean)
        row = {"stack_mse": mse, "loss": loss}
        if split == "val" and self._pool is not None:
            mse, loss = stack_errors(mods, self._pool, self._mean)
            row |= {"pool_stack_mse": mse, "pool_loss": loss}
        return row


@torch.no_grad()
def elbo_errors(
    mods: VAEModules,
    stacks: Tensor,
    mean: Tensor,
    *,
    seed: int,
    batch: int = EVAL_BATCH,
) -> dict[str, float]:
    """The VAE's keys (module docstring) of the stacks u8 ``(n, N, C, H, W)``
    (any device) with the mean stack ``mean`` ``(N, C, H, W)``, a batch at a
    time on the modules' device: the noise of the stacks is drawn in their
    order from a generator of ``seed`` on that device, so that the same seed
    and ``batch`` give the same noise."""
    dev = mean.device
    gen = torch.Generator(device=dev).manual_seed(seed)
    m = mean.float()
    loss, kl, sq = torch.zeros(3, dtype=torch.float64, device=dev).unbind()
    kl_dims: Tensor | None = None
    sums = torch.zeros(2, dtype=torch.float64, device=dev)
    for chunk in stacks.split(batch):
        x = chunk.to(dev)
        eps = torch.randn(len(x), *mods.code_shape, generator=gen, device=dev)
        u = torch.rand(x.shape, generator=gen, device=dev)
        t = elbo_terms(mods, x, u, eps)
        loss += t.loss.double().sum()
        kl += t.kl.double().sum()
        sq += t.sq.double().sum()
        k = t.kl.double().sum(0)
        kl_dims = k if kl_dims is None else kl_dims + k
        sums += stack_mse_sums(target(x), mods.decoder(t.mu), m)
    n, p = len(stacks), mean.numel()
    assert kl_dims is not None
    nll = float(loss) / n
    sigma_sq = math.exp(2 * float(mods.log_sigma))
    star_sq = float(sq) / (n * p)
    err, base = sums.tolist()
    q = torch.tensor(KL_QUANTILES, dtype=torch.float64, device=dev) / 100
    quant = (kl_dims / n).quantile(q).tolist()
    return {
        "bpd": float(bits_per_dim(nll, p)),
        "loss": nll,
        "kl": float(kl) / n,
        "rate_bpd": float(kl) / n / (p * math.log(2)),
        "sigma": math.sqrt(sigma_sq),
        "sigma_star": math.sqrt(star_sq),
        "sigma_gap": sigma_gap(star_sq, sigma_sq, p),
        "stack_mse": err / base,
    } | {f"kl_q{c}": x for c, x in zip(KL_QUANTILES, quant, strict=True)}


class VAEMeasures(AEMeasures):
    """The evaluation of the VAE's run (module docstring): that of
    ``AEMeasures`` with ``elbo_errors`` in place of ``stack_errors``. The
    noise of val, test and the pool are of ``EVAL_NOISE_SEED`` plus 0, 1
    and 2."""

    _SEEDS = {"val": 0, "test": 1, "pool": 2}

    @override
    def evaluate(self, mods: AEModules | VAEModules, split: str) -> dict[str, float]:
        """The row of ``split`` and, in the row of val of the overfit
        control, ``pool_bpd`` and ``pool_stack_mse``."""
        assert isinstance(mods, VAEModules)
        row = elbo_errors(
            mods,
            self.stacks[split],
            self._mean,
            seed=EVAL_NOISE_SEED + self._SEEDS[split],
        )
        if split == "val" and self._pool is not None:
            pool = elbo_errors(
                mods, self._pool, self._mean, seed=EVAL_NOISE_SEED + self._SEEDS["pool"]
            )
            row |= {"pool_bpd": pool["bpd"], "pool_stack_mse": pool["stack_mse"]}
        return row
