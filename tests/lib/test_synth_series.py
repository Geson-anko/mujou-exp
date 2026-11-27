r"""``exp.lib.synth_series``: the series follow the autocorrelations they
are built with.

The oracles: the unbiased products $\overline{x_t x_{t+\ell}}$ (unit
variance by construction, no sample mean) of a long OU series against
$r^\ell$, and of the kernel's series against ``kernel_acf``, within 4.5 of
their standard errors over independent columns; the FFT's convolution
against the direct sum to $10^{-12}$; ``kernel_acf`` of a hand-made kernel;
the Chebyshev columns orthonormal on a whole period's grid (the discrete
orthogonality of $\cos j\theta$, exact) and their phases of huge $j$ against
exact integer arithmetic. A rotation's triples have $G = \sin^2(\omega
n/2)$ to $10^{-12}$; a chained rotation is each block's own rotation; the
OU integral's mean squared increments against the closed form $D(n)$
within 4.5 standard errors over columns. The state-space model's covariance
solves its Lyapunov equation to $10^{-12}$, its samples' products meet
$\Gamma(\ell)$ within 5 standard errors of batch means, and the Kalman
filter of a fully observed AR(1) is $\mathrm{tr}(\Sigma - F^\Delta \Sigma
F^{\Delta\top})$ to $10^{-10}$.
"""

import math

import pytest
import torch

from exp.lib.midpoint import triple_terms
from exp.lib.synth_series import (
    block_rotation,
    box_ar1_acf,
    box_mean,
    chained_rotation,
    chebyshev,
    complex_ou,
    complex_ou_gamma,
    gauss_kernel,
    kalman_errors,
    kernel_acf,
    lgssm,
    lgssm_cov,
    lgssm_gamma,
    ou,
    ou_integral,
    ou_integral_d,
    random_walk,
    rotation,
    rotation_gamma,
    smooth,
    spectral,
)

T_MAX = 4.5


def products_t(x: torch.Tensor, lag: int, want: float) -> float:
    """The t of the columns' means of $x_t x_{t+\\ell}$ against ``want``."""
    p = (x[:-lag] * x[lag:]).mean(0) if lag else x.square().mean(0)
    return float((p.mean() - want) / (p.std() / math.sqrt(len(p))))


class TestOU:
    @pytest.mark.parametrize("lag", [0, 1, 3, 10])
    def test_autocorrelation_is_r_to_the_lag(self, lag):
        x = ou(4000, 64, 0.8, torch.Generator().manual_seed(0))
        assert abs(products_t(x, lag, 0.8**lag)) < T_MAX


class TestSmooth:
    def test_kernel_acf_by_hand(self):
        k = torch.tensor([1.0, 2.0, 2.0]) / 3  # sum of squares 1
        got = kernel_acf(k, torch.tensor([0, 1, 2, 3, -1]))
        want = torch.tensor([1.0, 6 / 9, 2 / 9, 0.0, 6 / 9], dtype=torch.float64)
        torch.testing.assert_close(got, want, rtol=1e-15, atol=1e-15)

    def test_fft_is_the_direct_convolution(self):
        k = gauss_kernel(2.0)
        x = smooth(30, 3, k, torch.Generator().manual_seed(1))
        e = torch.randn(30 + len(k) - 1, 3, generator=torch.Generator().manual_seed(1),
                        dtype=torch.float64)  # fmt: skip
        direct = torch.stack(
            [(k[:, None] * e[t : t + len(k)]).sum(0) for t in range(30)]
        )
        torch.testing.assert_close(x, direct, rtol=0, atol=1e-12)

    @pytest.mark.parametrize("lag", [0, 2, 5, 12])
    def test_autocorrelation_is_the_kernels(self, lag):
        k = gauss_kernel(4.0)
        x = smooth(4000, 64, k, torch.Generator().manual_seed(2))
        want = float(kernel_acf(k, torch.tensor([lag]))[0])
        assert abs(products_t(x, lag, want)) < T_MAX


class TestRandomWalk:
    def test_increments_of_lag_l_have_variance_l(self):
        x = random_walk(3000, 64, torch.Generator().manual_seed(3))
        d = (x[7:] - x[:-7]).square().mean(0) / 7  # each column's, mean 1
        assert abs(float((d.mean() - 1) / (d.std() / 8))) < T_MAX


class TestSpectral:
    def test_columns_are_root2_cosines_of_their_frequency(self):
        w = torch.tensor([0.1, 0.7])
        x = spectral(50, w, torch.Generator().manual_seed(4))
        # cos(a + w) + cos(a - w) = 2 cos(a) cos(w), for every t inside
        torch.testing.assert_close(
            x[2:] + x[:-2], 2 * x[1:-1] * torch.cos(w.double()), rtol=0, atol=1e-13
        )
        assert float(x.abs().max()) <= math.sqrt(2) + 1e-15


class TestChebyshev:
    def test_columns_are_orthonormal_on_a_whole_period(self):
        p = 256
        theta = 2 * math.pi * torch.arange(p, dtype=torch.float64) / p
        x = chebyshev(theta, torch.arange(1, 65))
        torch.testing.assert_close(x.T @ x / p, torch.eye(64, dtype=torch.float64),
                                   rtol=0, atol=1e-12)  # fmt: skip

    def test_huge_frequencies_keep_their_phase(self):
        # theta = 2 pi a / 2^20: j theta mod 2 pi = 2 pi (j a mod 2^20) / 2^20
        a = torch.tensor([1, 12345, 987654])
        js = torch.tensor([10**8, 10**8 + 999_999, 123_456_789])
        theta = 2 * math.pi * a.double() / 2**20
        got = chebyshev(theta, js)
        turns = (a[:, None] * js) % 2**20
        want = math.sqrt(2) * torch.cos(2 * math.pi * turns.double() / 2**20)
        torch.testing.assert_close(got, want, rtol=0, atol=1e-6)


class TestRotation:
    @pytest.mark.parametrize("lag", [1, 7, 20, 41, 100])
    def test_each_triples_g_is_sin_squared(self, lag):
        period = 123.0
        x = rotation(600, 8, period, torch.Generator().manual_seed(0))
        t = triple_terms(x, torch.arange(lag, 600 - lag), lag)
        want = math.sin(math.pi * lag / period) ** 2
        torch.testing.assert_close(
            t.num / t.den, torch.full_like(t.num, want), rtol=1e-12, atol=0
        )

    def test_d_must_be_even(self):
        with pytest.raises(ValueError, match="even d"):
            rotation(10, 3, 5.0, torch.Generator())

    def test_a_chain_is_each_blocks_own_rotation(self):
        blocks = torch.tensor([0] * 50 + [1] * 60 + [2] * 40)
        x = chained_rotation(blocks, 4, (10.0, 30.0), torch.Generator().manual_seed(1))
        for b, period in ((0, 10.0), (1, 30.0), (2, 10.0)):
            rows = (blocks == b).nonzero().flatten()
            u = rows[1:-1]
            t = triple_terms(x, u, 1)
            want = math.sin(math.pi / period) ** 2
            assert float((t.num / t.den - want).abs().max()) <= 1e-12


class TestOUIntegral:
    @pytest.mark.parametrize("lag", [1, 5, 20])
    def test_mean_squared_increments_are_d(self, lag):
        r = math.exp(-1 / 8)
        x = ou_integral(4000, 256, r, torch.Generator().manual_seed(2))
        per = (x[lag:] - x[:-lag]).square().mean(0)  # (d,) independent columns
        t = float((per.mean() - ou_integral_d(lag, r)) / (per.std() / math.sqrt(256)))
        assert abs(t) <= T_MAX

    def test_d_by_hand(self):
        r = 0.5  # D(2) = 2 + 2 r for unit variance: Var(v1 + v2)
        assert ou_integral_d(2, r) == pytest.approx(2 + 2 * r, rel=1e-12)


class TestDynProcesses:
    """The processes of ``plans/dyn.md``'s V-p1 against their autocovariances."""

    def test_block_rotation_of_one_period_is_the_chained_rotation(self):
        blocks = torch.arange(3000) // 700
        a = block_rotation(blocks, 6, (37.3, 81.1), torch.Generator().manual_seed(4))
        b = chained_rotation(blocks, 6, (37.3, 81.1), torch.Generator().manual_seed(4))
        assert torch.equal(a, b)

    def test_block_rotation_of_radii_is_the_mixed_gamma(self):
        # every block of one realization: a phase-averaged plane has
        # E[x_{t+l} x_t^T] = (rho / rho_bar)^2 R(omega rho l) exactly, so the
        # products over many blocks average to rotation_gamma at the weights
        radii, period, lag = (0.6, 1.0, 1.4), 50.0, 7
        blocks = torch.arange(300 * 64) // 64
        x = block_rotation(
            blocks, 2, (period,), torch.Generator().manual_seed(5), radii
        )
        within = (blocks[lag:] == blocks[:-lag]).nonzero().flatten()
        prod = x[within + lag, :, None] * x[within, None, :]  # (P, 2, 2)
        kinds = blocks[within] % 3
        bar2 = sum(r * r for r in radii) / 3
        for kind in range(3):
            rho = radii[kind]
            want = rotation_gamma(lag, 2, torch.tensor([2 * math.pi / period * rho]),
                                  torch.ones(1), torch.tensor([rho * rho / bar2]))  # fmt: skip
            got = prod[kinds == kind]
            se = got.std(0) / math.sqrt(100)  # one phase per block
            assert ((got.mean(0) - want).abs() <= T_MAX * se + 1e-12).all()

    @pytest.mark.parametrize("lag", [0, 1, 5, 20])
    def test_complex_ou_autocovariance(self, lag):
        lam, period = 0.05, 23.0
        x = complex_ou(40000, 32, lam, period, torch.Generator().manual_seed(6))
        want = complex_ou_gamma(lag, 2, lam, period)
        p = (
            x[lag:].reshape(-1, 16, 2)[:, :, :, None]
            * x[: len(x) - lag].reshape(-1, 16, 2)[:, :, None, :]
        )
        per_plane = p.mean(0)  # (16, 2, 2): independent planes
        t = (per_plane.mean(0) - want) / (per_plane.std(0) / 4)
        assert (t.abs() < T_MAX).all()

    def test_box_ar1_acf_is_the_double_sum(self):
        r, w = 0.9, 5
        for lag in (0, 1, 3, 7):
            want = sum(r ** abs(lag + i - j) for i in range(w) for j in range(w)) / w**2
            got = float(box_ar1_acf(r, w, torch.tensor([lag]))[0])
            assert got == pytest.approx(want, rel=1e-14)

    @pytest.mark.parametrize("lag", [0, 2, 6])
    def test_box_mean_of_ou_follows_box_ar1_acf(self, lag):
        r, w = 0.8, 4
        y = box_mean(ou(20000, 64, r, torch.Generator().manual_seed(7)), w)
        assert len(y) == 20000 - w + 1
        want = float(box_ar1_acf(r, w, torch.tensor([lag]))[0])
        assert abs(products_t(y, lag, want)) < T_MAX

    def test_box_mean_by_hand(self):
        x = torch.arange(6, dtype=torch.float64)[:, None]
        torch.testing.assert_close(box_mean(x, 3)[:, 0],
                                   torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64))  # fmt: skip


class TestStateSpace:
    """The state-space model: its covariance solves the Lyapunov equation,
    its samples follow ``lgssm_gamma``, and the Kalman filter's error of a
    fully observed AR(1) is the closed form."""

    @staticmethod
    def _model():
        f = torch.tensor(
            [[0.7, 0.2, 0.0], [-0.1, 0.5, 0.3], [0.0, 0.2, 0.6]], dtype=torch.float64
        )
        q = torch.tensor(
            [[1.0, 0.3, 0.0], [0.3, 1.0, 0.0], [0.0, 0.0, 0.5]], dtype=torch.float64
        )
        h = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float64)
        r = 0.4 * torch.eye(2, dtype=torch.float64)
        return f, q, h, r

    def test_the_covariance_is_stationary(self):
        f, q, _, _ = self._model()
        s = lgssm_cov(f, q)
        torch.testing.assert_close(f @ s @ f.T + q, s, rtol=0, atol=1e-12)

    @pytest.mark.parametrize("lag", [0, 1, 3])
    def test_samples_follow_the_autocovariance(self, lag):
        """$\\overline{w_{t+\\ell} w_t^\\top}$ of a long series against
        $\\Gamma(\\ell)$, within 5 standard errors of batch means (blocks
        of 1000 steps, far beyond the memory $0.8^{1000}$)."""
        f, q, h, r = self._model()
        w = lgssm(200_000, f, q, h, r, torch.Generator().manual_seed(3))
        prod = (w[lag:, :, None] * w[: len(w) - lag, None, :])[:199_000]
        per = prod.reshape(199, 1000, 2, 2).mean(1)
        se = per.std(0) / math.sqrt(199)
        assert ((per.mean(0) - lgssm_gamma(lag, f, q, h, r)).abs() <= 5 * se).all()

    def test_negative_lags_transpose(self):
        f, q, h, r = self._model()
        torch.testing.assert_close(
            lgssm_gamma(-2, f, q, h, r), lgssm_gamma(2, f, q, h, r).T
        )

    def test_kalman_of_a_fully_observed_ar1(self):
        """$H = I$, $R = 0$: the state is seen, so $E_\\Delta =
        \\mathrm{tr}(\\Sigma - F^\\Delta \\Sigma F^{\\Delta\\top})$."""
        f, q, _, _ = self._model()
        eye = torch.eye(3, dtype=torch.float64)
        s = lgssm_cov(f, q)
        want = []
        for lag in (1, 2, 5):
            fd = torch.linalg.matrix_power(f, lag)
            want.append(float((s - fd @ s @ fd.T).trace()))
        got = kalman_errors(f, q, eye, 0 * eye, (1, 2, 5))
        torch.testing.assert_close(
            got, torch.tensor(want, dtype=torch.float64), rtol=1e-10, atol=0
        )

    def test_noise_raises_the_error_by_at_least_its_own(self):
        """Seen through noise $R$, the error is at least the noiseless one
        plus $\\mathrm{tr}\\,R$ (the noise of the target is unpredictable)."""
        f, q, h, r = self._model()
        noisy = kalman_errors(f, q, h, r, (1, 4))
        clean = kalman_errors(f, q, h, 0 * r, (1, 4))
        assert (noisy >= clean + r.trace() - 1e-12).all()
