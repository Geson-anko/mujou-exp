import math

import pytest
import torch

from exp.lib.metrics import (
    block_bootstrap,
    block_counts,
    bootstrap_se,
    contiguous_blocks,
    effective_rank,
    identification,
    integrated_time,
    max_t_critical,
    pair_wins,
    probe_accuracy,
    probe_r2,
    ratio_resamples,
    scene_bootstrap,
    scene_resamples,
    swap,
    video_jackknife,
)
from exp.lib.readout import balanced_accuracy, scene_m


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


class TestContiguousBlocks:
    def test_by_hand(self):
        seg = torch.tensor([0] * 7 + [1] * 3 + [2] * 2)
        got = contiguous_blocks(seg, 3)
        # 7 = 3 + 4 (the remainder joins the block before), 3, and 2 < 3
        assert got.tolist() == [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3]


class TestBlockBootstrap:
    """With $v_i$ = 1, $m_c = 1 - \bar e_c$, so the contrast $m_A - m_B$ is
    minus the mean of $d_i = e_{Ai} - e_{Bi}$: its standard error is that of
    a mean, of the items where they are independent and of the blocks' means
    where the items are equal within a block."""

    def test_independent_items_give_the_se_of_their_mean(self):
        n, sigma = 1000, 2.0
        g = torch.Generator().manual_seed(1)
        d = sigma * torch.randn(n, generator=g, dtype=torch.float64)
        e = torch.stack([d, torch.zeros(n, dtype=torch.float64)])
        blocks = torch.arange(n)
        got = block_bootstrap(
            e, torch.ones(n), blocks, torch.tensor([[1.0, -1.0]]), 4000, g
        )
        want = float(d.std(unbiased=False)) / math.sqrt(n)
        assert abs(float(got[0]) / want - 1) < 0.06  # 1/sqrt(2 * 4000) is 1.1%

    def test_equal_items_within_blocks_count_as_one_per_block(self):
        b, length = 400, 8
        g = torch.Generator().manual_seed(2)
        per_block = torch.randn(b, generator=g, dtype=torch.float64)
        d = per_block.repeat_interleave(length)
        e = torch.stack([d, torch.zeros_like(d)])
        blocks = contiguous_blocks(torch.zeros(b * length), length)
        got = block_bootstrap(
            e, torch.ones(b * length), blocks, torch.tensor([[1.0, -1.0]]), 4000, g
        )
        want = float(per_block.std(unbiased=False)) / math.sqrt(b)
        assert abs(float(got[0]) / want - 1) < 0.06

    def test_a_code_against_itself_is_exactly_0(self):
        g = torch.Generator().manual_seed(3)
        e = torch.rand(1, 50, generator=g, dtype=torch.float64).expand(2, -1)
        v = 1 + torch.rand(50, generator=g, dtype=torch.float64)
        got = block_bootstrap(
            e, v, torch.arange(50) // 5, torch.tensor([[1.0, -1.0]]), 100, g
        )
        assert float(got[0]) == 0.0


class TestIntegratedTime:
    @pytest.mark.parametrize("phi", [0.5, 0.9])
    def test_an_ar1_has_tau_of_1_plus_phi_over_1_minus_phi(self, phi):
        t, k = 50000, 8
        g = torch.Generator().manual_seed(4)
        noise = torch.randn(t, k, generator=g, dtype=torch.float64)
        x = torch.empty_like(noise)
        x[0] = noise[0] / math.sqrt(1 - phi**2)  # stationary from the start
        for i in range(1, t):
            x[i] = phi * x[i - 1] + noise[i]
        want = (1 + phi) / (1 - phi)
        assert integrated_time(x) == pytest.approx(want, rel=0.05)


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


class TestPairWins:
    def test_a_small_case_by_hand(self):
        # cos: s(n, m) = <e_n, r_m> / |r_m|; query 0's own (1, 0) against
        # (0, 1) (s 0.6 < 0.8: a win) and (2, 0) (s 0.8, a tie)
        real = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]], dtype=torch.float64)
        pred = torch.tensor([[0.8, 0.6]], dtype=torch.float64)
        mean, each, ties = pair_wins(pred, real, truth=torch.tensor([0]))
        assert mean == each[0] == 0.75 and ties.tolist() == [1]
        # l2: s(n, m) = -|e_n - r_m|^2: 0.4, 0.8 and 1.6 away
        mean, each, ties = pair_wins(pred, real, truth=torch.tensor([0]), metric="l2")
        assert mean == 1.0 and ties.tolist() == [0]

    def test_one_prediction_of_no_direction_ties_with_all(self):
        real = _features(12).double()
        zero = torch.zeros(12, 6, dtype=torch.float64)
        mean, each, ties = pair_wins(zero, real)
        assert mean == 0.5 and torch.all(each == 0.5)
        assert ties.tolist() == [11] * 12

    def test_the_real_rows_themselves_win_every_pair(self):
        real = _features(12).double()
        for metric in ("cos", "l2"):
            mean, each, ties = pair_wins(real, real, metric=metric)
            assert mean == 1.0 and torch.all(each == 1.0) and int(ties.sum()) == 0

    def test_the_order_of_the_candidates_and_a_positive_scale_change_nothing(self):
        real, pred = _features(30).double(), _features(10, seed=1).double()
        g = torch.Generator().manual_seed(2)
        among = torch.stack([torch.randperm(30, generator=g)[:7] for _ in range(10)])
        truth = among[:, 3]
        _, each, _ = pair_wins(pred, real, truth=truth, among=among)
        flipped = among.flip(1)
        _, again, _ = pair_wins(3.5 * pred, real, truth=truth, among=flipped)
        assert torch.equal(each, again)

    def test_per_query_candidates_are_the_shared_ones_taken_by_among(self):
        real, pred = _features(30).double(), _features(10, seed=1).double()
        g = torch.Generator().manual_seed(3)
        among = torch.stack([torch.randperm(30, generator=g)[:7] for _ in range(10)])
        for metric in ("cos", "l2"):
            _, shared, _ = pair_wins(
                pred, real, truth=among[:, 0], among=among, metric=metric
            )
            _, own, _ = pair_wins(pred, real[among], metric=metric)  # own column 0
            torch.testing.assert_close(shared, own, rtol=0, atol=1e-12)

    def test_a_row_of_among_without_its_own_is_refused(self):
        real, pred = _features(30).double(), _features(2, seed=1).double()
        among = torch.tensor([[0, 1, 2], [3, 4, 5]])
        with pytest.raises(AssertionError, match="holds its own"):
            pair_wins(pred, real, truth=torch.tensor([1, 9]), among=among)

    def test_its_top1_is_identifications(self):
        # a hit (no candidate as near, a tie not a hit) is a share of 1
        real = _features(40).double()
        pred = real + 0.8 * _features(40, seed=5).double()
        _, each, _ = pair_wins(pred, real)
        out = identification(pred, real, center=torch.zeros(6))
        assert out["hits"] == (each == 1.0).double().tolist()

    def test_the_metric_is_checked(self):
        with pytest.raises(ValueError, match="cos"):
            pair_wins(_features(3), _features(3), metric="dot")


class TestSceneBootstrap:
    """The stratified block bootstrap of ``readout.scene_m`` (V18c's main
    measure, 3rd remake)."""

    @staticmethod
    def labelled(n: int, videos: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        y = torch.randint(3, (n,), generator=g)
        video = torch.arange(n) * videos // n
        p1 = torch.where(
            torch.rand(n, generator=g) < 0.6, y, torch.randint(3, (n,), generator=g)
        )
        p2 = torch.where(
            torch.rand(n, generator=g) < 0.5, y, torch.randint(3, (n,), generator=g)
        )
        return y, video, torch.stack([p1, p2]), g

    def test_a_code_against_itself_has_no_error(self):
        y, video, preds, _ = self.labelled(600, 3, 0)
        blocks = contiguous_blocks(video, 10)
        twice = torch.stack([preds[0], preds[0]])
        se = scene_bootstrap(
            twice, y, video, blocks, torch.tensor([[1.0, -1.0]]), 200,
            torch.Generator().manual_seed(1),
        )  # fmt: skip
        assert float(se[0]) == 0.0

    def test_a_prediction_constant_in_each_video_has_no_error(self):
        # every resample keeps each video's own blocks, so m stays exactly 0
        y, video, _, g = self.labelled(600, 4, 2)
        per_video = torch.randint(3, (4,), generator=g)[video]
        blocks = contiguous_blocks(video, 7)
        se = scene_bootstrap(
            per_video[None], y, video, blocks, torch.eye(1), 200,
            torch.Generator().manual_seed(3),
        )  # fmt: skip
        assert float(se[0]) == 0.0

    def test_independent_items_give_the_binomial_error_of_ba(self):
        # blocks of one item, 2 classes of n_k = 1000: sd(BA) = sqrt(sum p_k
        # (1 - p_k) / n_k) / K to first order
        g = torch.Generator().manual_seed(4)
        n_k, p = 1000, (0.8, 0.6)
        y = torch.arange(2 * n_k) % 2
        hit = torch.rand(2 * n_k, generator=g) < torch.tensor(p)[y]
        pred = torch.where(hit, y, 1 - y)
        video = torch.zeros_like(y)
        se = scene_bootstrap(
            pred[None], y, video, torch.arange(2 * n_k), torch.eye(1), 2000,
            torch.Generator().manual_seed(5), measure="ba",
        )  # fmt: skip
        want = math.sqrt(sum(q * (1 - q) / n_k for q in p)) / 2
        assert float(se[0]) == pytest.approx(want, rel=0.1)

    def test_parts_alike_are_their_mean(self):
        y, video, preds, _ = self.labelled(600, 3, 6)
        blocks = contiguous_blocks(video, 10)
        c = torch.tensor([[1.0, -1.0]])
        one = scene_bootstrap(
            preds, y, video, blocks, c, 300, torch.Generator().manual_seed(7)
        )
        two = scene_bootstrap(
            torch.stack([preds, preds], 1), torch.stack([y, y]), video, blocks, c, 300,
            torch.Generator().manual_seed(7),
        )  # fmt: skip
        torch.testing.assert_close(one, two, rtol=1e-12, atol=0)

    def test_a_resample_of_all_the_blocks_once_is_the_measure(self):
        # the m of a resample is that of readout.scene_m on its rows: with
        # one block per video drawn, every resample is the sample itself
        y, video, preds, _ = self.labelled(300, 3, 8)
        blocks = video.clone()  # one block per video
        m = [scene_m(p, y, video, 3) for p in preds]
        se = scene_bootstrap(
            preds, y, video, blocks, torch.tensor([[1.0, 0.0]]), 50,
            torch.Generator().manual_seed(9),
        )  # fmt: skip
        assert float(se[0]) == pytest.approx(0.0, abs=1e-15)
        assert balanced_accuracy(preds[0], y, 3) > m[0]

    def test_excluded_rows_do_not_count(self):
        y, video, preds, _ = self.labelled(600, 3, 10)
        blocks = contiguous_blocks(video, 10)
        c = torch.tensor([[1.0, -1.0]])
        keep = torch.ones_like(y, dtype=torch.bool)
        keep[::5] = False
        y_ex = torch.where(keep, y, torch.full_like(y, -1))
        a = scene_bootstrap(
            preds, y_ex, video, blocks, c, 100, torch.Generator().manual_seed(11)
        )
        preds2 = preds.clone()
        preds2[:, ~keep] = (preds2[:, ~keep] + 1) % 3  # what they predict is ignored
        b = scene_bootstrap(
            preds2, y_ex, video, blocks, c, 100, torch.Generator().manual_seed(11)
        )
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_min_count_leaves_out_the_classes_scene_m_leaves_out(self):
        # each video one block: every resample is the sample itself, so each
        # resampled m is scene_m of the sample with the same min_count
        y, video, preds, _ = self.labelled(600, 3, 12)
        y = y.clone()
        y[y == 2] = 1
        y[:5] = 2  # class 2: 5 rows, below min_count 10
        blocks = contiguous_blocks(video, 600)
        for mc in (1, 10):
            got = scene_resamples(
                preds, y, video, blocks, 20, torch.Generator().manual_seed(13),
                min_count=mc,
            )  # fmt: skip
            for c in range(2):
                want = scene_m(preds[c], y, video, 3, min_count=mc)
                torch.testing.assert_close(
                    got[:, c], torch.full((20,), want, dtype=torch.float64),
                    rtol=0, atol=1e-12,
                )  # fmt: skip
        assert scene_m(preds[0], y, video, 3) != scene_m(
            preds[0], y, video, 3, min_count=10
        )


class TestVideoJackknife:
    def test_a_mean_of_video_means_has_the_textbook_error(self):
        g = torch.Generator().manual_seed(0)
        a = torch.randn(9, generator=g, dtype=torch.float64)
        theta = torch.stack([torch.cat([a[:u], a[u + 1 :]]).mean() for u in range(9)])
        values, se = video_jackknife(theta)
        assert torch.equal(values, theta)
        assert se == pytest.approx(float(a.std()) / 3, rel=1e-12)


class TestRatioResamples:
    def test_block_counts_draw_each_resample_b_blocks(self):
        counts = block_counts(7, 50, torch.Generator().manual_seed(3))
        assert counts.shape == (50, 7)
        assert torch.equal(counts.sum(1), torch.full((50,), 7.0, dtype=torch.float64))

    def test_a_common_denominator_is_block_bootstraps(self):
        # block_bootstrap's m_c = 1 - sum e / sum v of the same draws
        g = torch.Generator().manual_seed(0)
        e = torch.rand(3, 40, generator=g, dtype=torch.float64)
        v = 1 + torch.rand(40, generator=g, dtype=torch.float64)
        blocks = torch.arange(40) // 5
        contrasts = torch.tensor([[1.0, -1, 0], [0, 1, -1], [1, 0, 0]])
        want = block_bootstrap(
            e, v, blocks, contrasts, 300, torch.Generator().manual_seed(9)
        )
        e_b = torch.zeros(3, 8, dtype=torch.float64).index_add_(1, blocks, e)
        v_b = torch.zeros(8, dtype=torch.float64).index_add_(0, blocks, v)
        counts = block_counts(8, 300, torch.Generator().manual_seed(9))
        m = 1 - ratio_resamples(e_b, v_b.expand(3, -1), counts)
        got = (m @ contrasts.double().T).std(0)
        torch.testing.assert_close(got, want, rtol=1e-12, atol=0)

    def test_each_code_keeps_its_own_denominator(self):
        num = torch.tensor([[1.0, 2.0], [3.0, 1.0]])
        den = torch.tensor([[2.0, 2.0], [1.0, 4.0]])
        counts = torch.tensor([[2.0, 0.0], [1.0, 1.0]])
        got = ratio_resamples(num, den, counts)
        want = torch.tensor([[2 / 4, 6 / 2], [3 / 4, 4 / 5]], dtype=torch.float64)
        torch.testing.assert_close(got, want, rtol=1e-15, atol=0)


class TestMaxTCritical:
    def test_one_value_is_the_quantile_of_abs_t(self):
        g = torch.Generator().manual_seed(1)
        r = torch.randn(1000, 1, generator=g, dtype=torch.float64)
        got = max_t_critical(r, torch.zeros(1), torch.full((1,), 2.0), 0.9)
        assert got == pytest.approx(
            float(torch.quantile(r.abs()[:, 0] / 2, 0.9)), rel=1e-12
        )

    def test_the_maximum_over_values_is_wider(self):
        g = torch.Generator().manual_seed(2)
        r = torch.randn(4000, 6, generator=g, dtype=torch.float64)
        one = max_t_critical(r[:, :1], torch.zeros(1), torch.ones(1))
        six = max_t_critical(r, torch.zeros(6), torch.ones(6))
        # independent normals: P(max |t| <= c) = (2 Phi(c) - 1)^6 = 0.95
        c6 = math.sqrt(2) * torch.special.erfinv(torch.tensor(0.95 ** (1 / 6))).item()
        assert one == pytest.approx(1.96, abs=0.08)
        assert six == pytest.approx(c6, abs=0.1)
