import pytest
import torch

from exp.lib.linear_dyn import errors, fit, rollout, select_alpha


def _system(seed: int = 0, d: int = 4, n: int = 50, h: int = 6):
    """Windows of a noiseless linear system z_{k+1} = A z_k + b, A of spectral
    radius 0.9 (an orthogonal matrix times 0.9)."""
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
    a = 0.9 * q
    b = torch.randn(d, generator=g, dtype=torch.float64)
    z = [torch.randn(n, d, generator=g, dtype=torch.float64) * 3]
    for _ in range(h):
        z.append(z[-1] @ a.T + b)
    return a, b, torch.stack(z, 1)  # (n, 1 + h, d)


class TestFit:
    def test_no_ridge_is_least_squares_with_a_column_of_ones(self):
        g = torch.Generator().manual_seed(1)
        z0 = torch.randn(40, 3, generator=g, dtype=torch.float64)
        z1 = torch.randn(40, 3, generator=g, dtype=torch.float64)
        a, b = fit(z0, z1, alpha=0.0)
        x = torch.cat([z0, torch.ones(40, 1, dtype=torch.float64)], 1)
        w = torch.linalg.lstsq(x, z1).solution  # (4, 3)
        torch.testing.assert_close(a, w[:3].T, rtol=0, atol=1e-10)
        torch.testing.assert_close(b, w[3], rtol=0, atol=1e-10)

    def test_a_noiseless_system_is_recovered_and_predicted_exactly(self):
        a, b, zs = _system()
        z0, z1 = zs[:, :-1].flatten(0, 1), zs[:, 1:].flatten(0, 1)
        a_hat, b_hat = fit(z0, z1, alpha=0.0)
        torch.testing.assert_close(a_hat, a, rtol=0, atol=1e-8)
        torch.testing.assert_close(b_hat, b, rtol=0, atol=1e-8)
        e = errors(a_hat, b_hat, zs, [1, 3, 6])
        for h in (1, 3, 6):
            assert e[f"dyn_{h}"] < 1e-10

    def test_the_ridge_shrinks_a(self):
        _, _, zs = _system()
        z0, z1 = zs[:, :-1].flatten(0, 1), zs[:, 1:].flatten(0, 1)
        small, _ = fit(z0, z1, alpha=1.0)
        big, _ = fit(z0, z1, alpha=1e6)
        assert big.norm() < small.norm()


class TestErrors:
    def test_the_identity_is_the_copy(self):
        _, _, zs = _system(2)
        eye, zero = (
            torch.eye(4, dtype=torch.float64),
            torch.zeros(4, dtype=torch.float64),
        )
        e = errors(eye, zero, zs, [1, 4])
        assert e["dyn_1/copy_1"] == pytest.approx(1.0, abs=1e-12)
        assert e["dyn_4"] == pytest.approx(e["copy_4"], abs=1e-12)

    def test_a_scale_of_z_changes_nothing(self):
        """dyn_h is 1 - R^2: shrinking z cannot make it smaller."""
        _, _, zs = _system(3)
        noisy = zs + 0.1 * torch.randn(
            zs.shape, generator=torch.Generator().manual_seed(0), dtype=torch.float64
        )
        z0, z1 = noisy[:, :-1].flatten(0, 1), noisy[:, 1:].flatten(0, 1)
        e = errors(*fit(z0, z1, alpha=0.0), noisy, [1, 2])
        c = 1e-3
        f = errors(*fit(c * z0, c * z1, alpha=0.0), c * noisy, [1, 2])
        for k in e:
            assert f[k] == pytest.approx(e[k], rel=1e-8)

    def test_the_rollout_iterates_a(self):
        a, b, zs = _system(4)
        out = rollout(a, b, zs[:, 0], 6)
        torch.testing.assert_close(out, zs[:, 1:], rtol=0, atol=1e-10)

    def test_a_horizon_beyond_the_window_raises(self):
        a, b, zs = _system(h=2)
        with pytest.raises(ValueError, match="horizons"):
            errors(a, b, zs, [3])


class TestWhitenedErrors:
    def test_by_hand_with_a_diagonal_covariance(self):
        """Training codes (+-2, 0), (0, +-1): C = diag(2, 0.5). The windows go
        from 0 to the same four codes; the prediction b = (1, 0) misses by
        (1, 0), (-3, 0), (-1, +-1). Whitened: errors 0.5, 4.5, 2.5, 2.5 (mean
        2.5) over the variance 2, 2, 2, 2: dynw 1.25; raw: 3.5 / 2.5 = 1.4.
        The copy (0) misses by the codes themselves: copyw 1."""
        codes = torch.tensor([[2.0, 0.0], [-2.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        zs = torch.stack([torch.zeros(4, 2), codes], 1).double()
        a = torch.zeros(2, 2, dtype=torch.float64)
        b = torch.tensor([1.0, 0.0], dtype=torch.float64)
        e = errors(a, b, zs, [1], train=codes)
        assert e["dyn_1"] == pytest.approx(1.4, rel=1e-12)
        assert e["dynw_1"] == pytest.approx(1.25, rel=1e-5)
        assert e["copyw_1"] == pytest.approx(1.0, rel=1e-5)

    def test_an_invertible_linear_map_of_z_changes_nothing(self):
        """z -> M z with A -> M A M^-1, b -> M b: the same rollout in other
        coordinates, so dynw and copyw stay (dyn_h does not, in general)."""
        a, b, zs = _system(5)
        noisy = zs + 0.3 * torch.randn(
            zs.shape, generator=torch.Generator().manual_seed(1), dtype=torch.float64
        )
        g = torch.Generator().manual_seed(2)
        m = torch.randn(4, 4, generator=g, dtype=torch.float64) + 2 * torch.eye(
            4, dtype=torch.float64
        )
        train = noisy[:, :-1].flatten(0, 1)
        e = errors(a, b, noisy, [1, 3], train=train)
        f = errors(
            m @ a @ torch.linalg.inv(m), m @ b, noisy @ m.T, [1, 3], train=train @ m.T
        )
        for k in ("dynw_1", "dynw_3", "copyw_1", "copyw_3"):
            assert f[k] == pytest.approx(e[k], rel=1e-5)
        assert "dynw_1" not in errors(a, b, noisy, [1])


def test_select_alpha_picks_the_least_ridge_of_a_noiseless_system():
    """With no noise, more ridge only biases A: the least scale wins."""
    _, _, zs = _system(5)
    z0, z1 = zs[:, :-1].flatten(0, 1), zs[:, 1:].flatten(0, 1)
    unit = float(((z0 - z0.mean(0)) ** 2).sum()) / 4
    alpha = select_alpha(z0, z1, zs, scales=[1e-6, 1e-2, 1.0])
    assert alpha == pytest.approx(1e-6 * unit)


def test_select_alpha_is_the_least_dyn_1_of_the_fits_by_definition():
    """The choice is that of fitting each scale and measuring dyn_1 on calib."""
    _, _, zs = _system(6, d=5, n=40)
    g = torch.Generator().manual_seed(7)
    noisy = zs + 0.5 * torch.randn(zs.shape, generator=g, dtype=torch.float64)
    z0, z1 = noisy[:, :-1].flatten(0, 1), noisy[:, 1:].flatten(0, 1)
    scales = [1e-6, 1e-2, 1e-1, 1.0, 10.0]
    unit = float(((z0 - z0.mean(0)) ** 2).sum()) / 5
    errs = [
        errors(*fit(z0, z1, alpha=c * unit), noisy[:, :2], [1])["dyn_1"] for c in scales
    ]
    best = scales[min(range(len(scales)), key=errs.__getitem__)]
    assert select_alpha(z0, z1, noisy, scales=scales) == pytest.approx(best * unit)


def test_float64_inputs_are_left_as_they_were():
    _, _, zs = _system(8)
    z0, z1 = zs[:, :-1].flatten(0, 1), zs[:, 1:].flatten(0, 1)
    before = [z0.clone(), z1.clone()]
    fit(z0, z1, alpha=0.1)
    select_alpha(z0, z1, zs)
    errors(*fit(z0, z1, alpha=0.1), zs, [1], train=z0)
    assert torch.equal(z0, before[0]) and torch.equal(z1, before[1])
