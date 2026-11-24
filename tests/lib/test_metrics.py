import math

import pytest
import torch

from exp.lib.metrics import (
    bootstrap_se,
    effective_rank,
    identification,
    probe_accuracy,
    probe_r2,
    swap,
)


class TestEffectiveRank:
    def test_isotropic_gaussian_is_about_d(self):
        g = torch.Generator().manual_seed(0)
        z = torch.randn(20_000, 64, generator=g)
        assert abs(effective_rank(z) - 64) < 1.5

    def test_data_of_rank_r_is_r(self):
        g = torch.Generator().manual_seed(1)
        q, _ = torch.linalg.qr(torch.randn(64, 5, generator=g))
        z = torch.randn(20_000, 5, generator=g) @ q.T
        assert abs(effective_rank(z) - 5) < 0.1


class TestProbe:
    def _split(self, x, y):
        n = len(x) // 2
        return x[:n], y[:n], x[n:], y[n:]

    def test_a_linear_function_of_the_state_is_read_out(self):
        g = torch.Generator().manual_seed(2)
        y = torch.randn(4000, 6, generator=g)
        x = y @ torch.randn(6, 32, generator=g) + 0.01 * torch.randn(
            4000, 32, generator=g
        )
        assert probe_r2(*self._split(x, y)).min() > 0.99

    def test_unrelated_noise_reads_nothing(self):
        g = torch.Generator().manual_seed(3)
        x, y = torch.randn(4000, 32, generator=g), torch.randn(4000, 6, generator=g)
        assert probe_r2(*self._split(x, y)).abs().max() < 0.05


class TestBootstrapSE:
    @pytest.mark.parametrize(("n", "sigma"), [(200, 1.0), (1024, 3.0)])
    def test_it_is_the_se_of_the_mean_of_normal_values(self, n, sigma):
        """For n values of N(0, sigma^2) the standard error of their mean is
        sigma / sqrt(n); 4000 resamples estimate it within 10% (the sampling
        error is about 1/sqrt(2n) + 1/sqrt(2 * 4000), below 6%)."""
        g = torch.Generator().manual_seed(4)
        x = sigma * torch.randn(n, generator=g, dtype=torch.float64)
        got = bootstrap_se(x, gen=g, resamples=4000)
        assert abs(got / (sigma / math.sqrt(n)) - 1) < 0.1

    def test_the_same_generator_gives_the_same_value(self):
        x = torch.randn(100, generator=torch.Generator().manual_seed(5))
        a = bootstrap_se(x, gen=torch.Generator().manual_seed(6))
        b = bootstrap_se(x, gen=torch.Generator().manual_seed(6))
        assert a == b

    def test_constant_values_have_no_error(self):
        g = torch.Generator().manual_seed(7)
        assert bootstrap_se(torch.full((50,), 2.5), gen=g) == 0.0


def test_effective_rank_is_that_of_the_eigenvalues_of_the_covariance():
    """The singular values give the covariance's eigenvalues, also for n < D."""
    g = torch.Generator().manual_seed(3)
    for n, d in ((50, 8), (6, 20)):
        z = torch.randn(n, d, generator=g, dtype=torch.float64) * torch.arange(1, d + 1)
        values = torch.linalg.eigvalsh(torch.cov(z.T)).clamp(min=0)
        p = values / values.sum()
        p = p[p > 1e-12]
        expected = float(torch.exp(-(p * p.log()).sum()))
        assert effective_rank(z) == pytest.approx(expected, rel=1e-9)


class TestProbeAccuracy:
    def test_linearly_separable_classes_are_read_without_error(self):
        """Three classes at the corners of a simplex, 100 times further apart
        than their spread: the least-squares one-hot readout separates them."""
        g = torch.Generator().manual_seed(0)
        y = torch.arange(3).repeat(200)
        centres = 10 * torch.eye(3, 4)
        f = centres[y] + 0.1 * torch.randn(600, 4, generator=g)
        assert probe_accuracy(f[:400], y[:400], f[400:], y[400:], 3) == 1.0

    def test_features_without_the_label_are_at_chance(self):
        g = torch.Generator().manual_seed(1)
        y = torch.arange(4).repeat(500)
        f = torch.randn(2000, 4, generator=g)
        acc = probe_accuracy(f[:1000], y[:1000], f[1000:], y[1000:], 4)
        assert abs(acc - 0.25) < 4 * math.sqrt(0.25 * 0.75 / 1000)


def _features(n: int, f: int = 6, seed: int = 0) -> torch.Tensor:
    return torch.randn(n, f, generator=torch.Generator().manual_seed(seed))


class TestIdentification:
    def test_the_real_features_themselves_are_all_found(self):
        real = _features(10)
        out = identification(real, real, center=real.mean(0))
        assert out["top1"] == out["top5"] == 1.0
        assert out["hits"] == [1.0] * 10 and out["top1_se"] == 0.0
        assert out["top1_adj"] is None  # no miss
        assert out["pick_max"] == pytest.approx(1 / 10)  # every row once

    def test_one_prediction_for_all_finds_exactly_1_and_5_of_n(self):
        real = _features(20)
        pred = _features(1, seed=1).expand(20, -1)
        out = identification(pred, real, center=real.mean(0))
        assert out["top1"] == pytest.approx(1 / 20)
        assert out["top5"] == pytest.approx(5 / 20)
        assert out["pick_max"] == 1.0  # a hub: every pick on one row

    def test_a_permutation_finds_its_fixed_points(self):
        real = _features(12)
        perm = torch.tensor([0, 2, 1, 3, 5, 4, 6, 7, 9, 8, 11, 10])
        out = identification(real[perm], real, center=real.mean(0))
        fixed = int((perm == torch.arange(12)).sum())
        assert out["top1"] == pytest.approx(fixed / 12)
        # each miss picked the swapped neighbour
        assert out["top1_adj"] == 1.0

    def test_positive_scales_change_nothing_and_top5_is_at_least_top1(self):
        real = _features(15)
        pred = real[:, None] + 0.8 * torch.randn(
            15, 4, 6, generator=torch.Generator().manual_seed(3)
        )
        center = real.mean(0)
        a = identification(pred, real, center=center)
        scale = torch.linspace(0.5, 3, 15)[:, None, None]
        b = identification(center + scale * (pred - center), real, center=center)
        assert a == b
        assert a["top5"] >= a["top1"]
        assert len(a["hits"]) == 15
        assert a["top1"] == pytest.approx(sum(a["hits"]) / 15)

    def test_among_its_own_alone_every_query_is_found(self):
        real = _features(9)
        pred = _features(4, seed=2)  # nowhere near
        truth = torch.tensor([7, 1, 4, 0])
        out = identification(
            pred, real, center=real.mean(0), truth=truth, among=truth[:, None]
        )
        assert out["top1"] == 1.0 and "top1_adj" not in out

    def test_among_a_set_ranks_only_within_it(self):
        real = _features(9)
        truth = torch.tensor([2, 5])
        pred = real[[3, 6]]  # each the real of another row
        among = torch.tensor([[2, 4], [5, 6]])
        out = identification(
            pred, real, center=torch.zeros(6), truth=truth, among=among
        )
        # query 0 (row 3) is not among {2, 4} so picks one of them; query 1 picks 6
        assert out["hits"][1] == 0.0

    def test_a_prediction_at_the_centre_finds_none(self):
        # no direction: every real is as near as its own, and a tie is no hit
        real = _features(9)
        center = real.mean(0)
        pred = center.expand(4, -1)
        truth = torch.tensor([7, 1, 4, 0])
        among = torch.stack([truth, (truth + 1) % 9, (truth + 2) % 9], 1)
        assert identification(pred, real, center=center, truth=truth)["top1"] == 0
        out = identification(pred, real, center=center, truth=truth, among=among)
        assert out["top1"] == 0.0  # its own first in among changes nothing

    def test_a_tie_with_another_row_is_no_hit(self):
        real = _features(5)
        real[3] = real[1]  # two candidates of the same features
        out = identification(real, real, center=torch.zeros(6))
        assert out["hits"] == [1.0, 0.0, 1.0, 0.0, 1.0]


class TestSwap:
    def test_its_own_real_against_one_of_no_direction_is_always_chosen(self):
        t, zero = _features(30), torch.zeros(30, 6)
        value, each = swap(t, zero, t)
        assert value == 1.0 and each.shape == (30,)
        assert swap(zero, t, t)[0] == 0.0
        assert swap(t, -t, t)[0] == 1.0

    def test_equal_rows_and_rows_of_no_direction_tie_at_exactly_one_half(self):
        a, t = _features(30), _features(30, seed=1)
        value, each = swap(a, a.clone(), t)
        assert value == 0.5 and torch.all(each == 0.5)
        zero = torch.zeros(30, 6)
        assert swap(zero, zero, t)[0] == 0.5
        assert swap(a + 1e3, a + 1e3, t)[0] == 0.5

    def test_exchangeable_pairs_unrelated_to_the_real_are_half(self):
        n = 10_000
        a, b, t = _features(n, seed=4), _features(n, seed=5), _features(n, seed=6)
        value, _ = swap(a, b, t)
        assert abs(value - 0.5) < 4 * math.sqrt(0.25 / n)

    def test_a_noisier_other_unrelated_to_the_real_is_still_half(self):
        n = 10_000
        common, t = _features(n, seed=7), _features(n, seed=8)
        t = t + common  # the common part points to the real
        a = common + 0.1 * _features(n, seed=9)
        b = common + 3.0 * _features(n, seed=10)
        value, _ = swap(a, b, t)
        assert abs(value - 0.5) < 4 * math.sqrt(0.25 / n)
