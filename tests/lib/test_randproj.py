"""``exp.lib.randproj``: the shape and the seed of $W$, the norm of $W v$
against its chi-square law, and the chunked product against the single one."""

import math

import torch

from exp.lib.randproj import gaussian_projection, project


class TestGaussianProjection:
    def test_the_shape_and_the_same_values_of_the_same_seed(self):
        w = gaussian_projection(8, 300, 61)
        assert w.shape == (8, 300) and w.dtype == torch.float32
        assert torch.equal(w, gaussian_projection(8, 300, 61))
        assert not torch.equal(w, gaussian_projection(8, 300, 62))

    def test_the_global_rng_is_left_as_it_was(self):
        torch.manual_seed(5)
        want = torch.rand(3)
        torch.manual_seed(5)
        gaussian_projection(4, 10, 61)
        assert torch.equal(torch.rand(3), want)

    def test_the_mean_square_of_w_v_is_that_of_v_within_the_chi_square(self):
        d_out, d_in = 1024, 4096
        w = gaussian_projection(d_out, d_in, 61).double()
        v = torch.randn(d_in, generator=torch.Generator().manual_seed(1)).double()
        ratio = float((w @ v).square().mean() / (v.square().sum() / d_in))
        # chi^2_d / d has mean 1 and sd sqrt(2 / d)
        assert abs(ratio - 1) < 4 * math.sqrt(2 / d_out)


class TestProject:
    def test_the_chunked_product_is_the_single_one(self):
        g = torch.Generator().manual_seed(0)
        w = gaussian_projection(16, 1000, 61)
        v = torch.randn(5, 1000, generator=g)
        torch.testing.assert_close(project(v, w, chunk=128), v @ w.T)
        torch.testing.assert_close(project(v, w), v @ w.T, rtol=0, atol=0)
