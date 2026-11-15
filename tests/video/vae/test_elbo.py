import math
from dataclasses import replace

import torch
from torch.distributions import Normal, kl_divergence

from exp.video.schema import AEConf
from exp.video.vae.elbo import elbo_terms
from exp.video.vae.modules import build_vae
from tests.video.packed import packed_conf, stacks


def _vae():
    conf = replace(
        packed_conf("bp", token_dim=4),
        program="vae",
        ae=AEConf(decoder_depth=1, sigma0_sq=0.05),
    )
    return build_vae(conf, seed=0)


def _noise(vae, x, seed=1):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.rand(x.shape, generator=g),
        torch.randn(len(x), *vae.code_shape, generator=g),
    )


class TestELBOTerms:
    def test_the_negative_elbo_by_torch_distributions(self):
        """-L of a stack: -log N(y; g(z), sigma^2) summed over the P values
        of y = (x + u) / 256 at z = mu + s eps, plus the KL of q = N(mu, s^2)
        from N(0, 1) summed over the D latents. The posterior's log variance
        is moved off 0 first, so that s enters."""
        vae = _vae()
        out = vae.head
        d = vae.code_shape[1]
        with torch.no_grad():
            out.bias[d:] = torch.linspace(-1.0, 0.5, d)
            vae.likelihood.log_sigma.fill_(-1.7)
        x = stacks(2)
        u, eps = _noise(vae, x)
        with torch.no_grad():
            t = elbo_terms(vae, x, u, eps)
            mu, logvar = vae.posterior(x)
            s = (0.5 * logvar).exp()
            z = mu + s * eps
            y = (x.float() + u) / 256
            y_hat = vae.decoder(z)
        logp = Normal(y_hat.double(), math.exp(-1.7)).log_prob(y.double())
        nll = -logp.flatten(1).sum(1)
        kl = kl_divergence(Normal(mu.double(), s.double()), Normal(0.0, 1.0))
        torch.testing.assert_close(t.nll.double(), nll, rtol=1e-5, atol=0)
        torch.testing.assert_close(t.kl.double(), kl.flatten(1), rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(t.loss, t.nll + t.kl.sum(1))
        sq = (y - y_hat).square().flatten(1).sum(1)
        torch.testing.assert_close(t.sq, sq)
        assert torch.equal(t.mu, mu)

    def test_the_same_noise_the_same_loss(self):
        vae = _vae()
        x = stacks(2)
        u, eps = _noise(vae, x)
        with torch.no_grad():
            a = elbo_terms(vae, x, u, eps).loss
            b = elbo_terms(vae, x, u, eps).loss
            c = elbo_terms(vae, x, *_noise(vae, x, seed=2)).loss
        assert torch.equal(a, b) and not torch.equal(a, c)
