r"""The loss of the VAE (M18, stage 2): the negative ELBO of a stack in nats,
summed over its $P$ values and its $D$ latents (not divided by $P$).

For a stack $x$ (u8, $P = N C H W$ values), the uniform noise $u$ of its
dequantization and the standard normal $\varepsilon$ of its code ``(T, d)``,

$$-\mathcal L(x) = \frac{\lVert y - g(z) \rVert^2}{2 \sigma^2} + P \ln \sigma
+ \frac P2 \ln 2\pi + \mathrm{KL}\big(q(z \mid x) \,\Vert\, N(0, I_D)\big),$$

$y = (x + u) / 256$, $(\mu, \ln s^2)$ the encoder's posterior of $x$ and $z =
\mu + s \odot \varepsilon$ the reparameterized sample: an unbiased estimate of
$-\mathcal L$ and of its gradient with respect to every parameter (Kingma &
Welling 2014; ``exp.lib.elbo``). $u$ and $\varepsilon$ are inputs, drawn by
the caller, so that the loss is a deterministic function of the stack, the
noise and the weights (the latent node of PC-ALM takes $\varepsilon$ the
same way). A batch learns the mean over its stacks that count
(``loss.batch_loss``), as the autoencoder.
"""

from dataclasses import dataclass

from torch import Tensor

from exp.lib.elbo import dequantize, gauss_kl, gauss_nll
from exp.video.vae.loss import LEVELS
from exp.video.vae.modules import VAEModules

__all__ = ["ELBOTerms", "elbo_terms"]


@dataclass(frozen=True)
class ELBOTerms:
    r"""The terms of $-\mathcal L$ of ``n`` stacks, in the graph: ``nll``
    $-\log p(y \mid z)$ ``(n,)``, ``kl`` of each latent ``(n, D)``, the
    squared error ``sq`` $\lVert y - g(z) \rVert^2$ ``(n,)``, and the
    posterior mean ``mu`` ``(n, T, d)``."""

    nll: Tensor
    kl: Tensor
    sq: Tensor
    mu: Tensor

    @property
    def loss(self) -> Tensor:
        r"""$-\mathcal L$ ``(n,)``: ``nll`` plus the sum of ``kl``."""
        return self.nll + self.kl.sum(1)


def elbo_terms(mods: VAEModules, x: Tensor, u: Tensor, eps: Tensor) -> ELBOTerms:
    r"""The ``ELBOTerms`` of the stacks ``x`` u8 ``(n, N, C, H, W)`` with
    the noise ``u`` in $[0, 1)$ of their shape and ``eps`` ``(n, T, d)``
    (module docstring), at the weights as they are now."""
    mu, logvar = mods.posterior(x)
    z = mu + (0.5 * logvar).exp() * eps
    y = dequantize(x, u, LEVELS)
    y_hat = mods.decoder(z)
    sq = (y - y_hat).square().flatten(1).sum(1)
    return ELBOTerms(
        nll=gauss_nll(y, y_hat, mods.log_sigma),
        kl=gauss_kl(mu, logvar).flatten(1),
        sq=sq,
        mu=mu,
    )
