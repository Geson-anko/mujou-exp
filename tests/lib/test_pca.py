import pytest
import torch

from exp.lib.pca import decode, encode, fit_pca


def _data(seed: int = 0, n: int = 60, f: int = 20):
    g = torch.Generator().manual_seed(seed)
    scales = torch.logspace(1, -2, f, dtype=torch.float64)
    return torch.randn(n, f, generator=g, dtype=torch.float64) * scales + 5.0


class TestFitPCA:
    def test_the_subspace_is_the_top_singular_one(self):
        x = _data()
        mean, comps = fit_pca(x, 4, gen=torch.Generator().manual_seed(0), niter=8)
        torch.testing.assert_close(
            comps @ comps.T, torch.eye(4, dtype=x.dtype), atol=1e-10, rtol=0
        )
        _, _, vh = torch.linalg.svd(x - x.mean(0), full_matrices=False)
        p_fit = comps.T @ comps
        p_svd = vh[:4].T @ vh[:4]
        assert (p_fit - p_svd).abs().max() < 1e-5

    def test_the_error_is_the_dropped_singular_values_eckart_young(self):
        x = _data(1)
        mean, comps = fit_pca(x, 5, gen=torch.Generator().manual_seed(0), niter=8)
        x_hat = decode(encode(x, mean, comps), mean, comps)
        s = torch.linalg.svdvals(x - x.mean(0))
        err = float(((x - x_hat) ** 2).sum())
        assert err == pytest.approx(float((s[5:] ** 2).sum()), rel=1e-6)

    def test_the_generator_fixes_the_result_and_d_is_checked(self):
        x = _data(2)
        a = fit_pca(x, 3, gen=torch.Generator().manual_seed(7))
        b = fit_pca(x, 3, gen=torch.Generator().manual_seed(7))
        assert torch.equal(a[1], b[1])
        with pytest.raises(ValueError, match="d must"):
            fit_pca(x, 0, gen=torch.Generator())
