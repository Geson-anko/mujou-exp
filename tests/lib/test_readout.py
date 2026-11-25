r"""``exp.lib.readout``: the exact linear relation recovered, independent
targets read as nothing, the calibration's choice, and the constant
prediction's identification at chance; ``centered_r2``'s values of the
targets, a constant and permutations, and Eckart-Young's ``rank_bound``.

The scene categories' readout (V18c's main measure, 3rd remake): the
weighted ridge equals the unweighted one of the rows copied by integer
weights; ``nested_cv``'s eigen path equals a ridge solved per fold and scale
(the codes standardized on each fit set), leaks nothing of the scored video,
and breaks ties to the larger scale; ``scene_m`` is exactly 0 for a
prediction constant within each video and $1 - B$ for the labels, 0 in
expectation for one independent of the labels within each video, and $q (1 -
B(\mathrm{label}))$ in expectation for the synthetic signal $S_q$."""

import math

import pytest
import torch

from exp.lib.metrics import identification
from exp.lib.readout import (
    CONST,
    SCENE_SCALES,
    balanced_accuracy,
    centered_r2,
    centered_terms,
    class_auc,
    class_weights,
    fit_ridge,
    fit_weighted_ridge,
    nested_cv,
    nested_cv_jackknife,
    predict,
    r2,
    rank_bound,
    scene_m,
    select_ridge,
    video_chance,
)


def data(n: int, d: int = 5, k: int = 3, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g, dtype=torch.float64)
    b = torch.randn(d, k, generator=g, dtype=torch.float64)
    return x, b, torch.randn(k, generator=g, dtype=torch.float64)


class TestRidge:
    def test_a_noiseless_linear_relation_is_recovered(self):
        x, b, c = data(50)
        fit = fit_ridge(x, x @ b + c, 1e-12)
        torch.testing.assert_close(fit.b, b, rtol=0, atol=1e-8)
        xt, _, _ = data(20, seed=1)
        assert r2(predict(fit, xt), xt @ b + c) == pytest.approx(1.0, abs=1e-10)

    def test_independent_targets_read_nothing_under_a_large_ridge(self):
        x, _, _ = data(200)
        g = torch.Generator().manual_seed(2)
        y = torch.randn(200, 4, generator=g)
        fit = fit_ridge(x, y, 1e6)
        held = torch.randn(200, 4, generator=g)
        assert r2(predict(fit, x), held) <= 0.01

    def test_the_calibrations_best_scale_is_chosen(self):
        x, b, c = data(30, d=20)
        g = torch.Generator().manual_seed(3)
        y = x @ b + c + 3 * torch.randn(30, 3, generator=g, dtype=torch.float64)
        xc, _, _ = data(30, d=20, seed=4)
        yc = xc @ b + c
        scales = (1e-8, 1e-2, 1.0)
        best = select_ridge(x, y, xc, yc, scales)
        scores = [r2(predict(fit_ridge(x, y, s), xc), yc) for s in scales]
        want = fit_ridge(x, y, scales[max(range(3), key=lambda i: scores[i])])
        assert best.alpha == want.alpha
        assert best.scale == want.scale

    def test_a_constant_input_identifies_at_exactly_chance(self):
        x, b, c = data(16)
        y = x @ b + c
        fit = fit_ridge(x, y, 1e-3)
        mean_in = x.mean(0, keepdim=True).expand(16, -1)  # E1's mean stage
        # a centre other than the fit set's mean (fill_eval's: the held-out
        # grid's), so the constant prediction has a direction
        out = identification(predict(fit, mean_in), y, center=torch.zeros(y.shape[1]))
        assert out["top1"] == pytest.approx(1 / 16)


class TestCenteredR2:
    """``centered_r2``: 1 for the targets themselves, exactly 0 for a
    constant, blind to a common shift and to a common positive scale, and
    the theory's value for a permutation of the targets."""

    def test_the_targets_are_1_and_a_constant_exactly_0(self):
        g = torch.Generator().manual_seed(5)
        y = torch.randn(40, 6, generator=g, dtype=torch.float64)
        assert centered_r2(y, y) == 1.0
        const = torch.full((40, 6), 3.7, dtype=torch.float64)
        assert centered_r2(const, y) == 0.0
        e, v = centered_terms(const, y)
        assert torch.equal(e, v)

    def test_a_shift_and_a_common_scale_change_nothing(self):
        g = torch.Generator().manual_seed(6)
        y = torch.randn(30, 4, generator=g, dtype=torch.float64)
        p = y + 0.7 * torch.randn(30, 4, generator=g, dtype=torch.float64)
        m = centered_r2(p, y)
        shift = torch.randn(4, generator=g, dtype=torch.float64)
        assert centered_r2(p + shift, y) == pytest.approx(m, abs=1e-12)
        assert centered_r2(p, y + shift) == pytest.approx(m, abs=1e-12)
        assert centered_r2(2.5 * p, 2.5 * y) == pytest.approx(m, abs=1e-12)

    def test_permutations_average_minus_1_and_derangements_less(self):
        # m(pi) = -1 + 2 sum_i <y~_pi(i), y~_i> / |Y~|^2; a uniform pi has
        # E y~_pi(i) = 0, one without fixed points -y~_i / (n - 1)
        n, draws = 12, 10000
        g = torch.Generator().manual_seed(7)
        y = torch.randn(n, 3, generator=g, dtype=torch.float64)
        uniform, deranged = [], []
        while len(deranged) < draws:
            pi = torch.randperm(n, generator=g)
            m = centered_r2(y[pi], y)
            if len(uniform) < draws:
                uniform.append(m)
            if not bool((pi == torch.arange(n)).any()):
                deranged.append(m)
        for got, want in ((uniform, -1.0), (deranged, -1 - 2 / (n - 1))):
            x = torch.tensor(got, dtype=torch.float64)
            se = float(x.std()) / math.sqrt(draws)
            assert abs(float(x.mean()) - want) < 4 * se


class TestRankBound:
    def test_no_prediction_of_rank_r_passes_it_and_the_svd_reaches_it(self):
        g = torch.Generator().manual_seed(8)
        y = torch.randn(60, 10, generator=g, dtype=torch.float64)
        y = y @ torch.randn(10, 10, generator=g, dtype=torch.float64)
        r = 3
        bound = rank_bound(y, r)
        for _ in range(20):
            a = torch.randn(60, r, generator=g, dtype=torch.float64)
            b = torch.randn(r, 10, generator=g, dtype=torch.float64)
            assert centered_r2(a @ b, y) <= bound
        yc = y - y.mean(0)
        u, s, vh = torch.linalg.svd(yc, full_matrices=False)
        best = u[:, :r] @ torch.diag(s[:r]) @ vh[:r]
        assert centered_r2(best, y) == pytest.approx(bound, abs=1e-12)


class TestSelectRidgeScore:
    def test_another_score_chooses_its_own_best(self):
        x, b, c = data(30, d=20)
        g = torch.Generator().manual_seed(9)
        y = x @ b + c + 3 * torch.randn(30, 3, generator=g, dtype=torch.float64)
        xc, _, _ = data(30, d=20, seed=4)
        yc = xc @ b + 5.0  # a shift only centred_r2 forgives
        scales = (1e-8, 1e-2, 1.0)
        best = select_ridge(x, y, xc, yc, scales, score=centered_r2)
        scores = [centered_r2(predict(fit_ridge(x, y, s), xc), yc) for s in scales]
        want = fit_ridge(x, y, scales[max(range(3), key=lambda i: scores[i])])
        assert best.alpha == want.alpha


# -- the scene categories -----------------------------------------------------------


def scenes(n: int, k: int, videos: int, seed: int, d: int = 6):
    """Codes ``(n, d)`` whose first ``k`` coordinates carry the label with
    noise, labels of unequal shares and videos of unequal lengths in order."""
    g = torch.Generator().manual_seed(seed)
    y = torch.multinomial(torch.linspace(1.0, 2.0, k), n, replacement=True, generator=g)
    video = torch.sort(torch.randint(videos, (n,), generator=g)).values
    x = torch.randn(n, d, generator=g, dtype=torch.float64)
    x[:, :k] += 1.5 * torch.nn.functional.one_hot(y, k).double()
    x[:, -1] += video.double()  # a video's offset: standardization must cope
    return x, y, video


def brute_scores(x, y, video, k, fit_videos, held, scale):
    """The weighted ridge solved directly on the fit videos' rows, the codes
    standardized on them, its outputs on ``held``."""
    on = torch.isin(video, torch.tensor(fit_videos))
    mu, sd = x[on].mean(0), x[on].std(0, unbiased=False)
    sd = torch.where(sd <= 1e-12, torch.ones_like(sd), sd)
    w = class_weights(y[on], k)
    t = torch.nn.functional.one_hot(y[on], k).double()
    fit = fit_weighted_ridge((x[on] - mu) / sd, t, w, scale)
    return predict(fit, (x[held] - mu) / sd)


class TestWeightedRidge:
    def test_integer_weights_are_rows_copied(self):
        x, b, c = data(30, d=4)
        y = x @ b + c
        reps = torch.tensor([1, 2, 3] * 10)
        got = fit_weighted_ridge(x, y, reps.double(), 0.3)
        want = fit_ridge(
            x.repeat_interleave(reps, 0), y.repeat_interleave(reps, 0), 0.3
        )
        torch.testing.assert_close(got.b, want.b, rtol=0, atol=1e-10)
        assert got.alpha == pytest.approx(want.alpha, rel=1e-12)
        xt, _, _ = data(5, d=4, seed=3)
        torch.testing.assert_close(predict(got, xt), predict(want, xt))

    def test_uniform_weights_are_the_plain_ridge(self):
        x, b, c = data(40, d=6)
        y = x @ b + c
        got = fit_weighted_ridge(x, y, torch.full((40,), 0.7, dtype=torch.float64), 1.0)
        want = fit_ridge(x, y, 1.0)
        torch.testing.assert_close(got.b, want.b, rtol=0, atol=1e-12)

    def test_class_weights_give_each_class_the_same_total(self):
        y = torch.tensor([0, 0, 0, 1, 2, 2])
        w = class_weights(y, 3)
        totals = torch.zeros(3, dtype=torch.float64).index_add_(0, y, w)
        torch.testing.assert_close(totals, torch.full((3,), 1 / 3, dtype=torch.float64))


class TestSceneScores:
    def test_a_hand_counted_confusion(self):
        y = torch.tensor([0, 0, 0, 1, 1, 2])
        pred = torch.tensor([0, 1, 0, 1, 2, 2])
        assert balanced_accuracy(pred, y, 3) == pytest.approx((2 / 3 + 1 / 2 + 1) / 3)
        assert balanced_accuracy(y, y, 3) == 1.0
        perm = torch.tensor([2, 0, 1])  # renaming the classes changes nothing
        assert balanced_accuracy(perm[pred], perm[y], 3) == pytest.approx(
            balanced_accuracy(pred, y, 3), abs=1e-15
        )

    def test_a_hand_counted_chance(self):
        # video 0: labels 0, 0, 1, predictions 0, 1, 1; video 1: 1, 1 / 0, 0
        y = torch.tensor([0, 0, 1, 1, 1])
        video = torch.tensor([0, 0, 0, 1, 1])
        pred = torch.tensor([0, 1, 1, 0, 0])
        # B = 1/2 [P(v0|0) P(0|v0) + P(v0|1) P(1|v0) + P(v1|1) P(1|v1)]
        want = 0.5 * (1 * (1 / 3) + (1 / 3) * (2 / 3) + (2 / 3) * 0)
        assert video_chance(pred, y, video, 2) == pytest.approx(want, abs=1e-15)

    def test_a_prediction_constant_in_each_video_scores_exactly_0(self):
        _, y, video = scenes(300, 4, 5, seed=1)
        g = torch.Generator().manual_seed(2)
        per_video = torch.randint(4, (5,), generator=g)
        assert scene_m(per_video[video], y, video, 4) == 0.0
        const = torch.zeros_like(y)
        assert balanced_accuracy(const, y, 4) == 0.25
        assert scene_m(const, y, video, 4) == 0.0

    def test_the_labels_score_one_less_their_chance(self):
        _, y, video = scenes(300, 4, 5, seed=3)
        b = video_chance(y, y, video, 4)
        assert scene_m(y, y, video, 4) == pytest.approx(1 - b, abs=1e-15)

    def test_independent_within_each_video_is_0_in_expectation(self):
        _, y, video = scenes(200, 3, 4, seed=4)
        g = torch.Generator().manual_seed(5)
        share = torch.rand(4, 3, generator=g)  # each video's own shares
        draws = 3000
        ms = torch.tensor(
            [
                scene_m(
                    torch.multinomial(share[video], 1, generator=g)[:, 0], y, video, 3
                )
                for _ in range(draws)
            ]
        )
        assert abs(float(ms.mean())) < 4 * float(ms.std()) / math.sqrt(draws)

    def test_the_synthetic_signal_recovers_q_times_one_less_chance(self):
        # in the population m*(S_q) = q (1 - B(label)): the mean over draws
        _, y, video = scenes(400, 3, 4, seed=6)
        b = video_chance(y, y, video, 3)
        g = torch.Generator().manual_seed(7)
        q, draws = 0.4, 2000
        ms = []
        for _ in range(draws):
            keep = torch.rand(len(y), generator=g) < q
            s = torch.where(keep, y, torch.randint(3, y.shape, generator=g))
            ms.append(scene_m(s, y, video, 3))
        mt = torch.tensor(ms)
        se = float(mt.std()) / math.sqrt(draws)
        assert abs(float(mt.mean()) - q * (1 - b)) < 4 * se

    def test_a_class_below_min_count_is_left_out(self):
        y = torch.tensor([0, 0, 1, 1, 2])
        pred = torch.tensor([0, 0, 1, 0, 2])
        assert balanced_accuracy(pred, y, 3, min_count=2) == pytest.approx(0.75)


class TestNestedCV:
    def test_each_fold_is_the_ridge_solved_on_its_fit_videos(self):
        x, y, video = scenes(240, 3, 4, seed=8)
        for c in (1e-3, 1.0):
            got = nested_cv(x, y, video, 3, scales=(c,))
            for v in range(4):
                held = video == v
                fit = [u for u in range(4) if u != v]
                want = brute_scores(x, y, video, 3, fit, held, c)
                torch.testing.assert_close(got.scores[held], want, rtol=0, atol=1e-9)
                assert torch.equal(got.pred[held], want.argmax(1))
            assert set(got.scale_by_video.values()) == {c}

    def test_the_inner_choice_is_the_best_inner_m(self):
        x, y, video = scenes(240, 3, 4, seed=9)
        scales = (1e-6, 1e-2, 10.0, CONST)
        got = nested_cv(x, y, video, 3, scales=scales)
        for v in range(4):
            fit = [u for u in range(4) if u != v]
            on = torch.isin(video, torch.tensor(fit))
            inner = []
            for c in scales:
                p = torch.zeros(len(y), dtype=torch.long)
                for u in fit:
                    rest = [w for w in fit if w != u]
                    if c != CONST:
                        p[video == u] = brute_scores(
                            x, y, video, 3, rest, video == u, c
                        ).argmax(1)
                inner.append(scene_m(p[on], y[on], video[on], 3))
            torch.testing.assert_close(
                torch.tensor(got.inner[v]), torch.tensor(inner), rtol=0, atol=1e-12
            )
            best = max(range(len(scales)), key=lambda i: (inner[i], i))
            assert got.scale_by_video[v] == scales[best]

    def test_a_tie_goes_to_the_larger_scale(self):
        # pure noise codes: every scale's inner m below that of the constant,
        # which is exactly 0, or tied with it
        x, y, video = scenes(120, 2, 3, seed=10)
        same = torch.zeros(len(y), 1, dtype=torch.float64)  # no information
        got = nested_cv(same, y, video, 2, scales=(1e-3, 1.0, CONST))
        assert set(got.scale_by_video.values()) == {CONST}
        assert torch.equal(got.pred, torch.zeros_like(y))

    def test_the_scored_videos_labels_do_not_reach_its_predictions(self):
        x, y, video = scenes(240, 3, 4, seed=11)
        a = nested_cv(x, y, video, 3, scales=SCENE_SCALES)
        g = torch.Generator().manual_seed(12)
        held = (video == 2).nonzero().flatten()
        y2 = y.clone()
        y2[held] = y[held[torch.randperm(len(held), generator=g)]]
        b = nested_cv(x, y2, video, 3, scales=SCENE_SCALES)
        assert torch.equal(a.pred[held], b.pred[held])

    def test_the_video_code_scores_exactly_0(self):
        _, y, video = scenes(240, 3, 4, seed=13)
        x = torch.nn.functional.one_hot(video, 4).double()
        got = nested_cv(x, y, video, 3, scales=(*SCENE_SCALES, CONST))
        assert abs(scene_m(got.pred, y, video, 3)) <= 1e-12

    def test_fixed_scales_are_the_outer_fits_at_them(self):
        x, y, video = scenes(240, 3, 4, seed=15)
        chosen = nested_cv(x, y, video, 3, scales=(1e-4, 1e-1, 10.0))
        again = nested_cv(x, y, video, 3, fixed=chosen.scale_by_video)
        assert torch.equal(again.pred, chosen.pred) and again.inner == {}
        const = nested_cv(x, y, video, 3, fixed=dict.fromkeys(range(4), CONST))
        assert torch.equal(const.pred, torch.zeros_like(y))

    def test_the_jackknife_refits_without_each_video(self):
        x, y, video = scenes(300, 3, 5, seed=14)
        got = nested_cv(x, y, video, 3, scales=(1e-2, 1.0))
        jack = nested_cv_jackknife(x, y, video, 3, got.scale_by_video)
        assert jack.shape == (5, len(y))
        for u in range(5):
            assert bool((jack[u][video == u] == -1).all())
            for v in range(5):
                if v == u:
                    continue
                rest = [w for w in range(5) if w not in (u, v)]
                want = brute_scores(
                    x, y, video, 3, rest, video == v, got.scale_by_video[v]
                ).argmax(1)
                assert torch.equal(jack[u][video == v], want)


class TestClassAuc:
    def test_a_perfect_and_a_reversed_score(self):
        y = torch.tensor([0] * 25 + [1] * 25)
        video = torch.zeros(50, dtype=torch.long)
        s = torch.nn.functional.one_hot(y, 2).double()
        assert class_auc(s, y, video, 2, min_each=20) == 1.0
        assert class_auc(-s, y, video, 2, min_each=20) == 0.0
        assert class_auc(torch.zeros(50, 2), y, video, 2, min_each=20) == 0.5

    def test_too_few_of_a_class_in_a_video_is_not_scored(self):
        y = torch.tensor([0] * 25 + [1] * 5)
        video = torch.zeros(30, dtype=torch.long)
        assert class_auc(torch.zeros(30, 2), y, video, 2, min_each=20) is None
