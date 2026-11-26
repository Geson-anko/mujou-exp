r"""``exp.lib.fidelity`` (the observation's content a code keeps).

The oracles: the pooled observation is each cell's mean by hand; $E$ and
$E^d$ are 1 and 1 of the observation itself and 0 and 0 of a constant
readout at the anchors' mean, and by hand of a few numbers; proposition 1
in closed form: of an orthogonal design $y = x$, the least squares from the
code with directions dropped explains exactly the kept variances' share, so
$E$ falls with each dropped direction; the Gaussian shrinkage $E[f \mid f +
n] = f / (1 + \sigma^2)$ in the large-sample limit of the least squares
(the conditional mean is linear); the ratio of a code with itself is 0 with
no spread, and its interval $\pm 2\sigma_{\mathrm{seed}}$; a readout
explaining nothing is $-\infty$ and dropped; the seeds' deviation by hand;
the front's verdicts by hand; an MLP of one seed the same twice and fitting
a linear map; the level error's lag-1 autocorrelation by hand, with $E^d$'s
error $\sum \lVert e_t \rVert^2 + \sum \lVert e_{t-1} \rVert^2 - 2\sum
\langle e_t, e_{t-1} \rangle$, and a readout of white error that keeps a
slow level ($E \approx 1$) at $E^d = 1 - 2 n \sigma^2 / \sum_t \lVert
d_t \rVert^2 < 0$ (outside propositions 1 and 2).

v3's oracles: the ridge of an orthogonal centred design shrinks the least
squares by $1 / (1 + \alpha)$ and keeps the intercept the mean; a path's
last snapshot is ``fit_mlp``'s net; the choice is the least score, a tie
the first in the order (44 candidates); of $y$ linear in $c$ the pair's
least squares reads the change exactly ($E^d = 1$: v2's difference $[B,
-B]$ lies in the pair's linear family), and the difference readout is v2's
$\hat d$; the contrast $D = E_a - 2^{-\psi_*}E_b$ is positive exactly where
$\psi > -\psi_*$ ($E_b > 0$), its interval by hand, finite above where
$\psi$'s upper end is $-\infty$ ($E_a$'s point below 0); the per-seed
$D_s$'s deviation by hand (0 of two linear readouts); the verdicts by hand;
the non-inferiority of a readout against itself (0) and of a uniform loss
$\delta$ per block (every draw $-\delta$).
"""

import math

import pytest
import torch

from exp.lib.fidelity import (
    ContentSums,
    MLPConf,
    candidate_names,
    comparable,
    content_sums,
    contrast_interval,
    contrast_seed_sd,
    contrast_seeds,
    contrast_verdict,
    diff_predictor,
    error_lag1,
    fit_linear,
    fit_mlp,
    fit_mlp_path,
    fit_ridge,
    front_verdict,
    kind_of,
    linear_predictor,
    noninferior,
    pair_inputs,
    pool_obs,
    predict,
    predict_linear,
    ratio_interval,
    run_errors,
    seed_sigma,
    select_candidate,
    snapshot_predictor,
)
from exp.lib.metrics import block_counts

CPU = torch.device("cpu")
F64 = torch.float64


def _counts(
    nb: int, seed: int = 101, n: int = 400
) -> tuple[torch.Tensor, torch.Tensor]:
    return (block_counts(nb, n, torch.Generator().manual_seed(seed)),
            block_counts(nb // 2, n, torch.Generator().manual_seed(seed)))  # fmt: skip


class TestObservation:
    def test_each_cell_is_its_mean(self):
        frames = torch.arange(2 * 3 * 4 * 4, dtype=torch.uint8).reshape(2, 3, 4, 4)
        got = pool_obs(frames, 2)
        f = frames.double() / 255
        want = f.reshape(2, 3, 2, 2, 2, 2).mean((3, 5)).flatten(1)
        assert got.shape == (2, 12)
        assert torch.allclose(got.double(), want, atol=1e-6)


def _series(n: int = 60, k: int = 3) -> torch.Tensor:
    return torch.randn(n, k, generator=torch.Generator().manual_seed(0), dtype=F64)


class TestQuantities:
    def test_the_observation_itself_and_a_constant(self):
        y = _series()
        t = torch.arange(1, 60)
        blocks = t // 20
        mean = y[t].mean(0)
        cs = content_sums(y, [y, mean.expand_as(y)], mean, t, blocks, 3)
        e, ed = cs.per_seed(), cs.per_seed(change=True)
        assert e.tolist() == pytest.approx([1.0, 0.0], abs=1e-12)
        assert ed.tolist() == pytest.approx([1.0, 0.0], abs=1e-12)
        # the value is the mean over the seeds (one denominator)
        assert cs.value() == pytest.approx(0.5, abs=1e-12)

    def test_by_hand(self):
        y = torch.tensor([[0.0], [1.0], [3.0], [6.0]], dtype=F64)
        p = torch.tensor([[0.0], [2.0], [2.0], [6.0]], dtype=F64)
        t = torch.tensor([1, 2, 3])
        cs = content_sums(
            y,
            [p],
            torch.tensor([2.0], dtype=F64),
            t,
            torch.zeros(3, dtype=torch.long),
            1,
        )
        # err: 1 + 1 + 0; tot: 1 + 1 + 16; d: 1, 2, 3; dhat: 2, 0, 4
        assert cs.err.tolist() == [[2.0]] and cs.tot.tolist() == [18.0]
        assert cs.derr.tolist() == [[1.0 + 4.0 + 1.0]] and cs.dtot.tolist() == [14.0]
        assert cs.value() == pytest.approx(1 - 2 / 18)
        assert cs.value(True) == pytest.approx(1 - 6 / 14)

    def test_the_errors_lag1_by_hand(self):
        y = torch.tensor([[0.0], [1.0], [3.0], [6.0]], dtype=F64)
        p = torch.tensor([[0.0], [2.0], [2.0], [6.0]], dtype=F64)
        t = torch.tensor([1, 2, 3])
        # e = y - p = 0, -1, 1, 0: cross -1, now 2, before 2
        assert error_lag1(y, [p, y], t)[0] == pytest.approx(-0.5)
        assert math.isnan(error_lag1(y, [p, y], t)[1])
        cs = content_sums(y, [p], y.mean(0), t, torch.zeros(3, dtype=torch.long), 1)
        assert cs.derr.sum().item() == pytest.approx(2 + 2 - 2 * -1)

    def test_a_white_error_on_a_kept_level(self):
        # a slow ramp read with white error s: E ~ 1, E^d = 1 - 2 n s2 / sum |d|^2
        g = torch.Generator().manual_seed(3)
        n, s2 = 100000, 0.1
        y = (torch.arange(n, dtype=F64) * 0.05).sin()[:, None] * 10
        p = y + math.sqrt(s2) * torch.randn(n, 1, generator=g, dtype=F64)
        t = torch.arange(1, n)
        cs = content_sums(y, [p], y.mean(0), t, torch.zeros(n - 1, dtype=torch.long), 1)
        d = float((y[1:] - y[:-1]).square().sum())
        assert cs.value() > 0.99
        assert cs.value(True) == pytest.approx(1 - 2 * (n - 1) * s2 / d, abs=0.02)
        assert cs.value(True) < 0
        assert error_lag1(y, [p], t)[0] == pytest.approx(0, abs=0.02)

    def test_proposition_1_closed_form(self):
        # an orthogonal design of variances v: y = x, the code x P (some
        # directions dropped); the least squares in sample explains the kept
        # variances exactly, so E falls with each dropped direction
        g = torch.Generator().manual_seed(1)
        n, d = 400, 5
        # columns orthogonal to each other and to the constant (centred)
        ones = torch.ones(n, 1, dtype=F64) / math.sqrt(n)
        q, _ = torch.linalg.qr(
            torch.cat([ones, torch.randn(n, d, generator=g, dtype=F64)], 1)
        )
        x = q[:, 1:] * torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0], dtype=F64)
        v = x.square().sum(0)
        es = []
        for r in range(d):
            keep = torch.ones(d, dtype=F64)
            keep[:r] = 0  # the largest r directions dropped
            xc = x * keep
            b = fit_linear(xc, x)
            cs = content_sums(x, [predict_linear(b, xc).double()], x.mean(0),
                              torch.arange(n), torch.zeros(n, dtype=torch.long), 1)  # fmt: skip
            want = float(v[r:].sum() / v.sum())
            assert cs.value() == pytest.approx(want, abs=1e-9)
            es.append(cs.value())
        assert all(b < a for a, b in zip(es, es[1:], strict=False))

    def test_gaussian_shrinkage(self):
        # E[f | f + n] = f / (1 + s2): the share explained of a unit f is 1/(1+s2)
        g = torch.Generator().manual_seed(2)
        n, s2 = 200000, 0.5
        f = torch.randn(n, 1, generator=g, dtype=F64)
        c = f + math.sqrt(s2) * torch.randn(n, 1, generator=g, dtype=F64)
        b = fit_linear(c, f)
        cs = content_sums(f, [predict_linear(b, c).double()], f.mean(0), torch.arange(1, n),
                          torch.zeros(n - 1, dtype=torch.long), 1)  # fmt: skip
        assert cs.value() == pytest.approx(1 / (1 + s2), abs=5e-3)


def _sums(errs: list[list[float]], tot: list[float]) -> ContentSums:
    e = torch.tensor(errs, dtype=F64)
    t = torch.tensor(tot, dtype=F64)
    return ContentSums(e, t, e.clone(), t.clone())


class TestRatio:
    def test_a_code_with_itself(self):
        g = torch.Generator().manual_seed(3)
        tot = (1 + torch.rand(8, generator=g, dtype=F64)).tolist()
        errs = [[0.3 * x for x in tot], [0.31 * x for x in tot]]
        a = _sums(errs, tot)
        c, c2 = _counts(8)
        got = ratio_interval(a, a, c, c2, 0.0)
        assert got["point"] == 0.0 and got["se"] == pytest.approx(0.0, abs=1e-15)
        got = ratio_interval(a, a, c, c2, 0.05)
        assert got["lo"] == pytest.approx(-0.1) and got["hi"] == pytest.approx(0.1)

    def test_the_point_and_the_total_spread(self):
        g = torch.Generator().manual_seed(4)
        tot = 1 + torch.rand(8, generator=g, dtype=F64)
        ea = tot * (0.2 + 0.1 * torch.rand(8, generator=g, dtype=F64))
        eb = tot * 0.1
        a, b = _sums([ea.tolist()], tot.tolist()), _sums([eb.tolist()], tot.tolist())
        c, c2 = _counts(8)
        got = ratio_interval(a, b, c, c2, 0.03)
        want = math.log2((1 - float(ea.sum() / tot.sum())) / 0.9)
        assert got["point"] == pytest.approx(want)
        sds = []
        for sa, sb, cc in ((a, b, c), (a.paired(), b.paired(), c2)):
            r = torch.log2(sa.draws(cc) / sb.draws(cc))
            sds.append(float(r.std()))
        tot_se = math.sqrt(max(sds) ** 2 + 0.03**2)
        assert got["se"] == pytest.approx(max(sds))
        assert got["lo"] == pytest.approx(want - 2 * tot_se)
        assert got["hi"] == pytest.approx(want + 2 * tot_se)

    def test_nothing_explained_is_dropped(self):
        tot = [1.0] * 8
        a = _sums([[1.2] * 8], tot)  # E = -0.2
        b = _sums([[0.1] * 8], tot)
        c, c2 = _counts(8)
        got = ratio_interval(a, b, c, c2, 0.01)
        assert (
            got["point"] == -math.inf
            and got["lo"] == -math.inf
            and got["hi"] == -math.inf
        )
        undefined = ratio_interval(b, a, c, c2, 0.01)
        assert undefined["point"] is None and undefined["lo"] is None

    def test_the_seeds_deviation_by_hand(self):
        tot = [1.0, 1.0]
        a = _sums([[0.25, 0.25], [0.5, 0.5], [0.0, 0.0]], tot)  # E_s 0.75, 0.5, 1
        b = _sums([[0.5, 0.5]] * 3, tot)  # E_s 0.5
        r = [math.log2(1.5), 0.0, 1.0]
        m = sum(r) / 3
        want = math.sqrt(sum((x - m) ** 2 for x in r) / 2)
        assert seed_sigma(a, b) == pytest.approx(want)

    def test_comparable(self):
        c, c2 = _counts(8)
        tot = [1.0] * 8
        assert not comparable(_sums([[1.0] * 8], tot), c, c2)["ok"]  # E = 0
        assert comparable(_sums([[0.5] * 8, [0.5] * 8], tot), c, c2)["ok"]
        # the seeds' spread counts: E_s 0.9 and -0.1 average to 0.4, not sure > 0
        assert not comparable(_sums([[0.1] * 8, [1.1] * 8], tot), c, c2)["ok"]


class TestVerdict:
    @staticmethod
    def iv(lo: float | None, hi: float | None) -> dict[str, float | None]:
        return {"lo": lo, "hi": hi}

    def test_by_hand(self):
        ok, low, mid = self.iv(-0.05, 0.05), self.iv(-0.5, -0.2), self.iv(-0.2, 0.0)
        wide, narrow = (0.6, 1.0), (0.4, 1.0)
        assert front_verdict(narrow, ok, ok, True, 0.5, 0.1) == "width"
        assert front_verdict((None, 3.0), ok, ok, True, 0.5, 0.1) == "width"
        assert front_verdict(wide, ok, ok, False, 0.5, 0.1) == "incomparable"
        assert front_verdict(wide, ok, ok, True, 0.5, 0.1) == "kept"
        assert front_verdict(wide, ok, low, True, 0.5, 0.1) == "dropped"
        assert front_verdict(wide, low, ok, True, 0.5, 0.1) == "dropped"
        assert front_verdict(wide, mid, ok, True, 0.5, 0.1) == "unresolved"
        assert (
            front_verdict(wide, self.iv(None, None), ok, True, 0.5, 0.1) == "unresolved"
        )


class TestReadout:
    def test_one_seed_twice_and_a_linear_map(self):
        g = torch.Generator().manual_seed(5)
        x = torch.randn(512, 3, generator=g)
        y = x @ torch.randn(3, 2, generator=g)
        conf = MLPConf(hidden=32, steps=600, batch=64, lr=3e-3)
        a, la = fit_mlp(x, y, conf, 7, CPU)
        b, lb = fit_mlp(x, y, conf, 7, CPU)
        assert la == lb
        for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
            assert torch.equal(pa, pb)
        _, lc = fit_mlp(x, y, conf, 8, CPU)
        assert lc != la
        p = predict(a, x, CPU)
        cs = content_sums(y, [p], y.mean(0), torch.arange(1, 512),
                          torch.zeros(511, dtype=torch.long), 1)  # fmt: skip
        assert cs.value() > 0.98
        assert not a.training


# -- v3 ------------------------------------------------------------------------------------


def _centred_orthogonal(n: int, d: int, seed: int) -> torch.Tensor:
    """``(n, d)`` columns orthogonal to each other and to the constant, of
    $x^\\top x = n I$."""
    g = torch.Generator().manual_seed(seed)
    ones = torch.ones(n, 1, dtype=F64) / math.sqrt(n)
    q, _ = torch.linalg.qr(
        torch.cat([ones, torch.randn(n, d, generator=g, dtype=F64)], 1)
    )
    return q[:, 1:] * math.sqrt(n)


class TestRidge:
    def test_an_orthogonal_design_shrinks_by_one_over_one_plus_alpha(self):
        n = 300
        x = _centred_orthogonal(n, 4, 6)
        g = torch.Generator().manual_seed(7)
        y = x @ torch.randn(4, 3, generator=g, dtype=F64) + 2.5
        y = y + 0.1 * torch.randn(n, 3, generator=g, dtype=F64)
        b0 = fit_linear(x, y)
        assert torch.allclose(fit_ridge(x, y, 0.0), b0, atol=1e-10)
        for alpha in (0.01, 0.1, 1.0):
            b = fit_ridge(x, y, alpha)
            assert torch.allclose(b[:-1], b0[:-1] / (1 + alpha), atol=1e-10)
            # the intercept is not shrunk: the mean of y (x is centred)
            assert torch.allclose(b[-1], y.mean(0), atol=1e-10)


class TestPath:
    def test_the_last_snapshot_is_fit_mlps_net(self):
        g = torch.Generator().manual_seed(8)
        x = torch.randn(200, 3, generator=g)
        y = torch.randn(200, 2, generator=g)
        conf = MLPConf(hidden=8, steps=40, batch=16)
        net, snaps = fit_mlp_path(x, y, conf, 3, CPU, 10)
        ref, _ = fit_mlp(x, y, conf, 3, CPU)
        assert len(snaps) == 4
        last = snapshot_predictor(net, snaps[-1], CPU)(x)
        assert torch.equal(last, predict(ref, x, CPU))
        first = snapshot_predictor(net, snaps[0], CPU)(x)
        assert not torch.equal(first, last)
        assert torch.equal(snapshot_predictor(net, snaps[-1], CPU)(x), last)


class TestChoice:
    def test_the_order_of_the_candidates(self):
        names = candidate_names((0.0, 0.01, 0.1, 1.0), (256, 1024), 20000, 1000)
        assert len(names) == 44
        assert names[:5] == ["lin-0", "lin-0.01", "lin-0.1", "lin-1", "mlp-256-1000"]
        assert names[-1] == "mlp-1024-20000" and names[23] == "mlp-256-20000"
        assert [kind_of(n) for n in ("lin-0.1", "mlp-256-1000", "diff")] == [
            "lin", "mlp", "diff"]  # fmt: skip

    def test_the_least_wins_and_ties_go_to_the_first(self):
        order = ["lin-0", "lin-1", "mlp-4-10", "mlp-4-20"]
        assert (
            select_candidate(dict(zip(order, [3.0, 2.0, 1.0, 1.5])), order)
            == "mlp-4-10"
        )
        assert (
            select_candidate(dict(zip(order, [2.0, 1.0, 1.0, 1.0])), order) == "lin-1"
        )
        nan = dict(zip(order, [math.nan, 5.0, math.nan, 6.0]))
        assert select_candidate(nan, order) == "lin-1"
        assert select_candidate(dict.fromkeys(order, math.nan), order) == "lin-0"

    def test_run_errors(self):
        y = torch.tensor([[1.0], [2.0], [3.0]], dtype=F64)
        p = torch.zeros(3, 1)
        got = run_errors(p, y, torch.tensor([0, 1, 1]), 3)
        assert got.tolist() == [1.0, 13.0, 0.0]


class TestPairReadout:
    def test_a_linear_observation_is_read_back_from_the_pair(self):
        # y linear in c: the pair's least squares reads d_t = y_t - y_{t-1}
        # exactly, E^d = 1 (the difference [B, -B] lies in the pair's family)
        g = torch.Generator().manual_seed(9)
        c = torch.randn(400, 3, generator=g, dtype=F64).cumsum(0) / 10
        y = c @ torch.randn(3, 5, generator=g, dtype=F64) + 1.0
        xp = pair_inputs(c[None])[0]
        d = y - torch.cat([y[:1], y[:-1]])
        t = torch.arange(1, 400)
        b = fit_ridge(xp[1:], d[1:], 0.0)
        cs = content_sums(y, [y], y.mean(0), t, torch.zeros(399, dtype=torch.long), 1,
                          dpreds=[predict_linear(b, xp).double()])  # fmt: skip
        assert cs.value(True) == pytest.approx(1.0, abs=1e-9)

    def test_the_difference_readout_is_v2s(self):
        g = torch.Generator().manual_seed(10)
        c = torch.randn(50, 2, generator=g, dtype=F64)
        y = torch.randn(50, 3, generator=g, dtype=F64)
        b = fit_linear(c, y)
        lvl = linear_predictor(b)
        dp = diff_predictor(lvl, 2)(pair_inputs(c[None])[0])
        t = torch.arange(1, 50)
        blocks = torch.zeros(49, dtype=torch.long)
        a = content_sums(y, [lvl(c)], y.mean(0), t, blocks, 1)
        bb = content_sums(y, [lvl(c)], y.mean(0), t, blocks, 1, dpreds=[dp])
        assert torch.allclose(a.derr, bb.derr, atol=1e-5)
        assert torch.equal(a.err, bb.err)


def _blocks(e: list[float], tot: list[float]) -> ContentSums:
    """Sums of $E$ per block ``e`` (the error ``(1 - e) tot``), $E^d$ the
    same."""
    t = torch.tensor(tot, dtype=F64)
    err = ((1 - torch.tensor(e, dtype=F64)) * t)[None]
    return ContentSums(err, t, err.clone(), t.clone())


class TestContrast:
    def test_the_bound_is_psis(self):
        # E_b > 0: D > 0 exactly where psi > -psi*
        c, c2 = _counts(8)
        b = _blocks([0.5] * 8, [1.0] * 8)
        rho = 2**-0.1
        for ea in (0.2, 0.5 * rho - 1e-6, 0.5 * rho + 1e-6, 0.6):
            a = _blocks([ea] * 8, [1.0] * 8)
            d = contrast_interval(a, b, c, c2, 0.0, 0.1)
            psi = ratio_interval(a, b, c, c2, 0.0)["point"]
            assert psi is not None
            assert (d["point"] > 0) == (psi > -0.1)
            assert d["point"] == pytest.approx(ea - rho * 0.5)

    def test_the_interval_by_hand_and_through_e_a_below_0(self):
        c, c2 = _counts(8)
        tot = [1.0] * 8
        # E_a's point below 0, many of its draws above: psi's upper end is
        # -inf, D's a finite number above the point
        a = _blocks([0.3, 0.3, 0.3, 0.3, -0.4, -0.4, -0.4, -0.4], tot)
        b = _blocks([0.5] * 8, tot)
        assert a.value() < 0
        assert ratio_interval(a, b, c, c2, 0.02)["hi"] == -math.inf
        d = contrast_interval(a, b, c, c2, 0.02, 0.1)
        sds = [float((sa.draws(cc) - 2**-0.1 * sb.draws(cc)).std())
               for sa, sb, cc in ((a, b, c), (a.paired(), b.paired(), c2))]  # fmt: skip
        tot_se = math.sqrt(max(sds) ** 2 + 0.02**2)
        assert d["se"] == pytest.approx(max(sds))
        assert d["hi"] == pytest.approx(d["point"] + 2 * tot_se)
        assert math.isfinite(d["hi"]) and d["hi"] > d["point"]

    def test_the_seeds_deviation(self):
        tot = [1.0, 1.0]
        a = ContentSums(torch.tensor([[0.5, 0.5], [0.3, 0.3], [0.1, 0.1]], dtype=F64),
                        torch.tensor(tot, dtype=F64),
                        torch.tensor([[0.5, 0.5], [0.3, 0.3], [0.1, 0.1]], dtype=F64),
                        torch.tensor(tot, dtype=F64))  # fmt: skip
        lin = _blocks([0.5, 0.5], tot)  # one seed
        rho = 2**-0.1
        # E_a per seed 0.5, 0.7, 0.9; against one value the deviation is E_a's
        d = contrast_seeds(a, lin, 0.1)
        assert d.tolist() == pytest.approx(
            [0.5 - rho * 0.5, 0.7 - rho * 0.5, 0.9 - rho * 0.5]
        )
        assert contrast_seed_sd(a, lin, 0.1) == pytest.approx(0.2)
        assert contrast_seed_sd(lin, lin, 0.1) == 0.0
        assert contrast_seed_sd(a, a, 0.1) == pytest.approx(0.2 * (1 - rho))
        two = ContentSums(a.err[:2], a.tot, a.derr[:2], a.dtot)
        with pytest.raises(ValueError, match="seeds"):
            contrast_seeds(a, two, 0.1)

    def test_the_verdicts(self):
        def iv(lo: float, hi: float) -> dict[str, float]:
            return {"lo": lo, "hi": hi}

        wide, narrow = (1.0, 2.0), (0.2, 2.0)
        ok, low, mid = iv(0.01, 0.2), iv(-0.3, -0.01), iv(-0.1, 0.1)
        assert contrast_verdict(narrow, ok, ok, True, 0.5) == "width"
        assert contrast_verdict((None, 3.0), ok, ok, True, 0.5) == "width"
        assert contrast_verdict(wide, ok, ok, False, 0.5) == "incomparable"
        assert contrast_verdict(wide, ok, ok, True, 0.5) == "kept"
        assert contrast_verdict(wide, ok, low, True, 0.5) == "dropped"
        assert contrast_verdict(wide, low, ok, True, 0.5) == "dropped"
        assert contrast_verdict(wide, mid, ok, True, 0.5) == "unresolved"


class TestNoninferior:
    def test_the_same_and_a_uniform_loss(self):
        g = torch.Generator().manual_seed(11)
        tot = (1 + torch.rand(16, generator=g, dtype=F64)).tolist()
        e = (0.5 + 0.2 * torch.rand(16, generator=g, dtype=F64)).tolist()
        lin = _blocks(e, tot)
        c = block_counts(16, 500, torch.Generator().manual_seed(102))
        same = noninferior(lin, lin, 0.01, c)
        assert same["point"] == 0.0 and same["lo"] == 0.0 and same["ok"]
        # every block loses 0.03 of its E: every draw's difference is -0.03
        worse = _blocks([x - 0.03 for x in e], tot)
        got = noninferior(worse, lin, 0.05, c)
        assert got["point"] == pytest.approx(-0.03) and got["lo"] == pytest.approx(
            -0.03
        )
        assert got["ok"]
        assert not noninferior(worse, lin, 0.02, c)["ok"]
        assert not noninferior(worse, lin, 0.02, c, change=True)["ok"]
