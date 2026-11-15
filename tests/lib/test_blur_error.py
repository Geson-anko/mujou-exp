import math

import pytest
import torch

from exp.lib.blur_error import (
    blur_mse,
    blur_mse_masked,
    blur_recon,
    blur_recon_masked,
    gaussian_blur,
    gaussian_kernel,
)

W = 64


def response(sigma: float, omega: float) -> float:
    """The kernel's response $K(\\omega) = \\sum_j k_j \\cos(\\omega j)$."""
    k = gaussian_kernel(sigma, torch.float64)
    j = torch.arange(len(k), dtype=torch.float64) - len(k) // 2
    return float((k * torch.cos(omega * j)).sum())


def cosine(omega: float, shift: float = 0.0, h: int = W, w: int = W) -> torch.Tensor:
    """$\\cos(\\omega (x - \\mathrm{shift}))$ along the width ``(h, w)``."""
    x = torch.arange(w, dtype=torch.float64) - shift
    return torch.cos(omega * x).expand(h, w).clone()


def images(seed: int, n: int = 4) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, 3, W, W, generator=g, dtype=torch.float64)


class TestGaussianKernel:
    @pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0, 3.3])
    def test_it_sums_to_one_over_three_sigma(self, sigma):
        k = gaussian_kernel(sigma, torch.float64)
        assert len(k) == 2 * math.ceil(3 * sigma) + 1
        assert float(k.sum()) == pytest.approx(1.0, abs=1e-15)
        assert torch.equal(k, k.flip(0))

    def test_sigma_zero_is_the_identity(self):
        assert torch.equal(gaussian_kernel(0.0), torch.ones(1))

    def test_a_negative_sigma_is_rejected(self):
        with pytest.raises(ValueError, match="sigma"):
            gaussian_kernel(-1.0)


class TestGaussianBlur:
    @pytest.mark.parametrize("sigma", [1.0, 2.0])
    @pytest.mark.parametrize("m", [3, 10])
    def test_a_cosine_is_scaled_by_the_kernel_response(self, sigma, m):
        # omega (W - 1) = m pi: the cosine is even about both borders, so the
        # reflected padding continues it exactly and the blur is K(omega) x.
        omega = m * math.pi / (W - 1)
        x = cosine(omega)
        y = gaussian_blur(x, sigma)
        torch.testing.assert_close(y, response(sigma, omega) * x, rtol=0, atol=1e-12)

    def test_a_constant_image_is_kept(self):
        x = torch.full((2, 3, W, W), 0.37, dtype=torch.float64)
        torch.testing.assert_close(gaussian_blur(x, 2.0), x, rtol=0, atol=1e-15)

    def test_the_shape_is_kept_and_the_axes_are_blurred_alike(self):
        x = images(0)
        y = gaussian_blur(x, 2.0)
        assert y.shape == x.shape
        torch.testing.assert_close(
            gaussian_blur(x.transpose(-1, -2), 2.0), y.transpose(-1, -2)
        )

    def test_a_kernel_wider_than_the_image_is_rejected(self):
        with pytest.raises(ValueError, match="radius"):
            gaussian_blur(torch.zeros(1, 3, 6, 6), 2.0)


class TestBlurMse:
    def test_the_same_image_is_zero(self):
        x = images(1)
        assert float(blur_mse(x, x)) == 0.0
        assert blur_recon(x, x, images(2, 1)[0]) == 0.0

    def test_sigma_zero_is_the_squared_error(self):
        a, b, mean = images(3), images(4), images(5, 1)[0]
        mse = ((a - b) ** 2).mean()
        assert float(blur_mse(a, b, 0.0)) == pytest.approx(float(mse), rel=1e-15)
        base = ((mean - b) ** 2).mean()
        assert blur_recon(a, b, mean, 0.0) == pytest.approx(float(mse / base), 1e-14)

    @pytest.mark.parametrize("sigma", [0.0, 1.0, 2.0, 4.0])
    def test_the_mean_image_is_one(self, sigma):
        x, mean = images(6), images(7, 1)[0]
        assert blur_recon(mean.expand_as(x), x, mean, sigma) == pytest.approx(1.0)

    @pytest.mark.parametrize("sigma", [1.0, 2.0])
    def test_a_cosine_counts_its_response_squared(self, sigma):
        omega = 7 * math.pi / (W - 1)
        x = cosine(omega)
        mse = float((x**2).mean())
        expected = response(sigma, omega) ** 2 * mse
        assert float(blur_mse(x, torch.zeros_like(x), sigma)) == pytest.approx(
            expected, rel=1e-12
        )

    @pytest.mark.parametrize("wavelength", [8.0, 16.0, 32.0])
    def test_a_one_pixel_shift_counts_the_response_squared(self, wavelength):
        # Away from the borders the blur is a convolution, so the blurred
        # difference of a cosine and its 1 px shift is K(omega) times the
        # difference: the shift's error is scaled by K(omega)^2 against the
        # plain one, 2 (1 - cos omega) times the cosine's energy.
        sigma, omega, r = 2.0, 2 * math.pi / wavelength, 6
        x, shifted = cosine(omega), cosine(omega, shift=1.0)
        d = gaussian_blur(shifted - x, sigma)[:, r:-r]
        plain = (shifted - x)[:, r:-r]
        torch.testing.assert_close(
            d, response(sigma, omega) * plain, rtol=0, atol=1e-12
        )
        ratio = float((d**2).mean() / (plain**2).mean())
        assert ratio == pytest.approx(response(sigma, omega) ** 2, rel=1e-10)
        # A structure of a patch (16 px) keeps most of its error, finer
        # detail little: K^2 = exp(-sigma^2 omega^2) for the continuous kernel,
        # off by the 0.27% of its mass beyond 3 sigma.
        assert ratio == pytest.approx(math.exp(-(sigma**2) * omega**2), rel=1e-2)


def block(n: int, top: int, left: int, side: int) -> torch.Tensor:
    """The mask ``(n, W, W)`` of a square of ``side`` at ``(top, left)``."""
    m = torch.zeros(n, W, W, dtype=torch.bool)
    m[:, top : top + side, left : left + side] = True
    return m


class TestMasked:
    def test_every_pixel_is_the_unmasked_error(self):
        a, b, mean = images(0), images(1), images(2)[0]
        every = torch.ones(4, W, W, dtype=torch.bool)
        torch.testing.assert_close(blur_mse_masked(a, b, every), blur_mse(a, b))
        assert blur_recon_masked(a, b, mean, every) == pytest.approx(
            blur_recon(a, b, mean), rel=1e-12
        )

    def test_sigma_zero_is_the_squared_error_on_the_mask(self):
        a, b = images(0), images(1)
        m = block(4, 10, 20, 8)
        expect = ((a - b) ** 2)[:, :, 10:18, 20:28].mean()
        torch.testing.assert_close(blur_mse_masked(a, b, m, 0.0), expect)

    @pytest.mark.parametrize("sigma", [1.0, 2.0])
    def test_pixels_beyond_the_kernels_radius_of_the_mask_are_not_read(self, sigma):
        """The blurred error at a pixel reads the pixels within Chebyshev
        distance r = ceil(3 sigma) (the reflected borders included); a change
        farther from every masked pixel leaves the error as it is, one at the
        radius does not."""
        r = math.ceil(3 * sigma)
        a, b = images(0), images(1)
        m = block(4, 20, 20, 10)  # rows and columns 20 .. 29
        base = blur_mse_masked(a, b, m, sigma)
        far = a.clone()
        near = torch.zeros(W, W, dtype=torch.bool)
        near[20 - r : 30 + r, 20 - r : 30 + r] = True
        far[:, :, ~near] = torch.rand(int((~near).sum()), dtype=a.dtype)
        assert float(blur_mse_masked(far, b, m, sigma)) == pytest.approx(
            float(base), rel=1e-12
        )
        edge = a.clone()
        edge[:, :, 20 - r, 25] += 1.0  # at the radius
        assert float(blur_mse_masked(edge, b, m, sigma)) != pytest.approx(
            float(base), rel=1e-12
        )
