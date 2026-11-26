r"""``exp.lib.dynpred``: the direct prediction's ratios, fits and oracle.

The oracles: the half-octave grid by hand; the oracle of a 1-D OU at $(1 +
r)/2$ and $1 - r^2$, of a 1-D process at the closed FVU, of a deterministic
rotation at 0 (the pseudo-inverse), a fixed map at the least of the oracle
and never below it; the ridge's normal equations at the fit; a series of
constant speed (in the family) predicted to 0; the ratio of the two ratios
the code's own $V / D$; every ratio unchanged by an orthogonal map (the ridge
included); the MLP at step 0 the linear to the bit, the chosen time the
least of the seeds' mean (the earliest on a tie), the models side by side
the models alone; $\overline{\log R}$ of hand-made trapezoids.
"""

import math

import pytest
import torch

from exp.lib.dynpred import (
    GRID,
    MLPConf,
    Triples,
    best_linear,
    best_linear_one,
    choose,
    dens,
    fit_linear,
    fixed_error,
    half_octaves,
    mean_log,
    mean_log_draws,
    num_sums,
    pair_blocks,
    population_dens,
    record_preds,
    train_mlp,
)
from exp.lib.metrics import block_counts
from exp.lib.synth_series import block_rotation, ou, rotation_gamma

F64 = torch.float64


def _blocked(w: torch.Tensor, block: int, reach: int) -> Triples:
    """The triples of ``w`` cut in blocks of ``block`` rows, every centre at
    least ``reach`` from its block's ends."""
    n = len(w)
    nb = n // block
    cen = torch.cat(
        [torch.arange(b * block + reach, (b + 1) * block - reach) for b in range(nb)]
    )
    return Triples(w, cen, cen // block, nb)


def _ou_gamma(r: float, lag: int, k: int = 1) -> torch.Tensor:
    return torch.eye(k, dtype=F64) * r ** abs(lag)


class TestGrid:
    def test_half_octaves_by_hand(self):
        assert GRID == (1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 91, 128, 181, 256)
        assert half_octaves(4) == (1, 2, 3, 4)


class TestOracle:
    @pytest.mark.parametrize("r", [0.3, 0.9])
    def test_ou_is_half_of_one_plus_r_and_one_less_r_squared(self, r):
        g0, g1, g2 = _ou_gamma(r, 0), _ou_gamma(r, 1), _ou_gamma(r, 2)
        e = best_linear(g0, g1, g2)
        copy, clim = population_dens(g0, g1, torch.zeros(1, dtype=F64))
        assert e / copy == pytest.approx((1 + r) / 2, rel=1e-12)
        assert e / clim == pytest.approx(1 - r * r, rel=1e-12)

    def test_one_dimension_is_the_closed_fvu(self):
        r1, r2 = 0.7, 0.2  # a valid pair: |r2| < 1, 2 r1^2 - 1 <= r2
        e = best_linear(
            _ou_gamma(1, 0),
            torch.tensor([[r1]], dtype=F64),
            torch.tensor([[r2]], dtype=F64),
        )
        fvu = (1 - 2 * r1**2 + 2 * r1**2 * r2 - r2**2) / (1 - r1**2)
        assert e == pytest.approx(fvu, rel=1e-12)

    def test_a_deterministic_rotation_is_predicted_exactly(self):
        om = torch.tensor([2 * math.pi / 12.34])
        g = [rotation_gamma(lag, 4, om, torch.ones(1)) for lag in (0, 3, 6)]
        assert abs(best_linear(*g)) < 1e-10
        # one input: w_{t+n} = R(omega n) w_t is linear in w_t too
        assert abs(best_linear_one(g[0], g[1])) < 1e-10

    def test_a_fixed_map_reaches_the_least_only_at_the_best(self):
        r = 0.8
        g = [_ou_gamma(r, lag, 3) for lag in (0, 2, 4)]
        best = best_linear(*g)
        m_star = torch.cat(
            [r**2 * torch.eye(3, dtype=F64), torch.zeros(3, 3, dtype=F64)], 1
        )
        assert fixed_error(m_star, torch.zeros(3, dtype=F64), *g) == pytest.approx(
            best, rel=1e-12
        )
        gen = torch.Generator().manual_seed(0)
        for _ in range(5):
            m = m_star + 0.1 * torch.randn(3, 6, generator=gen, dtype=F64)
            b = 0.1 * torch.randn(3, generator=gen, dtype=F64)
            assert fixed_error(m, b, *g) > best

    def test_fixed_error_of_an_ou_by_hand(self):
        # y = w_{t+n}, x = (w_t, w_{t-n}), E[y w_t] = r^n, E[y w_{t-n}] = r^2n,
        # E[w_t w_{t-n}] = r^n: E(y - m1 w_t - m2 w_{t-n} - b)^2 by hand
        r, m1, m2, b = 0.7, 0.4, -0.3, 0.2
        g = [_ou_gamma(r, lag) for lag in (0, 1, 2)]
        want = (
            1 - 2 * (m1 * r + m2 * r * r) + m1 * m1 + m2 * m2 + 2 * m1 * m2 * r + b * b
        )
        m = torch.tensor([[m1, m2]], dtype=F64)
        assert fixed_error(m, torch.tensor([b], dtype=F64), *g) == pytest.approx(
            want, rel=1e-13
        )


class TestLinear:
    def _ou_triples(self, seed: int) -> tuple[Triples, Triples]:
        w = ou(6000, 3, 0.9, torch.Generator().manual_seed(seed))
        return _blocked(w[:4000], 400, 20), _blocked(w[4000:], 400, 20)

    def test_the_fit_solves_the_ridge_normal_equations(self):
        train, val = self._ou_triples(0)
        f = fit_linear(train, val, 3)
        now, past, fut = train.at(3)
        x = torch.cat([now, past], 1)
        res = x @ f.a.T + f.b - fut
        assert float(res.sum(0).abs().max()) < 1e-9  # the intercept's
        grad = x.T @ res + f.alpha * f.a.T  # (2k, k); centred x: sum res = 0
        assert float(grad.abs().max()) < 1e-8 * float(
            x.abs().max() * fut.abs().max() * len(x)
        )

    def test_the_chosen_scale_has_the_least_validation_error(self):
        train, val = self._ou_triples(1)
        f = fit_linear(train, val, 5, scales=(1e-6, 1e-2, 10.0, 1e3))
        assert f.val[(1e-6, 1e-2, 10.0, 1e3).index(f.scale)] == min(f.val)
        pred, y = f.predict(val)
        assert float((pred - y).square().sum()) == pytest.approx(min(f.val), rel=1e-12)

    def test_a_tie_takes_the_smallest_scale(self):
        train, val = self._ou_triples(2)
        f = fit_linear(train, val, 2, scales=(1e-3, 1e-3))
        assert f.val[0] == f.val[1] and f.alpha == pytest.approx(f.alpha_min)

    def test_constant_speed_is_in_the_family(self):
        # w_t = v_b t + c_b per block: w_{t+n} = 2 w_t - w_{t-n} exactly
        gen = torch.Generator().manual_seed(3)
        blocks = torch.arange(200 * 50) // 50
        v = torch.randn(200, 4, generator=gen, dtype=F64)[blocks]
        c = torch.randn(200, 4, generator=gen, dtype=F64)[blocks]
        w = v * torch.arange(len(blocks), dtype=F64)[:, None] / 1000 + c
        train, val = _blocked(w[:6000], 50, 10), _blocked(w[6000:], 50, 10)
        f = fit_linear(train, val, 4, scales=(1e-14,))
        pred, y = f.predict(val)
        copy = dens(val, 4, w.mean(0))["copy_b"].sum()
        assert float((pred - y).square().sum() / copy) < 1e-8

    def test_the_two_ratios_are_the_codes_own_v_over_d(self):
        train, val = self._ou_triples(4)
        mean = train.w.mean(0)
        f = fit_linear(train, val, 4)
        pred, y = f.predict(val)
        num = num_sums(pred - y, val)["num_b"].sum()
        d = dens(val, 4, mean)
        r_copy, r_clim = num / d["copy_b"].sum(), num / d["clim_b"].sum()
        assert float(r_copy / r_clim) == pytest.approx(
            float(d["clim_b"].sum() / d["copy_b"].sum()), rel=1e-12
        )

    def test_an_orthogonal_map_changes_no_ratio_the_ridge_included(self):
        train, val = self._ou_triples(5)
        q, _ = torch.linalg.qr(
            torch.randn(3, 3, generator=torch.Generator().manual_seed(9), dtype=F64)
        )
        rot = [
            Triples(s.w @ q.T, s.centres, s.blocks, s.n_blocks) for s in (train, val)
        ]
        for two in (True, False):
            a = fit_linear(train, val, 6, two=two, scales=(1e-3, 1.0, 30.0))
            b = fit_linear(rot[0], rot[1], 6, two=two, scales=(1e-3, 1.0, 30.0))
            assert a.scale == b.scale
            torch.testing.assert_close(
                torch.tensor(a.val), torch.tensor(b.val), rtol=1e-10, atol=0
            )
            assert b.train_min == pytest.approx(a.train_min, rel=1e-10)

    def test_record_rows(self):
        train, val = self._ou_triples(6)
        mean = train.w.mean(0)
        p = record_preds(train, val, 2, mean)
        now, past, _ = val.at(2)
        torch.testing.assert_close(p["copy"], now)
        torch.testing.assert_close(p["vel"], 2 * now - past)
        # the scale c is the least squares of w_{t+n} - mean on w_t - mean
        nt, _, ft = train.at(2)
        c = float(((ft - mean) * (nt - mean)).sum() / (nt - mean).square().sum())
        torch.testing.assert_close(p["scale"], mean + c * (now - mean))


class TestMLP:
    def _sets(self) -> tuple[Triples, Triples]:
        blocks = torch.arange(40 * 64) // 64
        w = block_rotation(
            blocks, 4, (20.0,), torch.Generator().manual_seed(0), (0.6, 1.4)
        )
        w = w + 0.1 * torch.randn(
            w.shape, generator=torch.Generator().manual_seed(1), dtype=F64
        )
        return _blocked(w[: 30 * 64], 64, 8), _blocked(w[30 * 64 :], 64, 8)

    def _run(self, lags, seeds, steps=40):
        train, val = self._sets()
        fits = {n: fit_linear(train, val, n) for n in lags}
        conf = MLPConf(width=16, batch=32, steps=steps, every=10, seeds=seeds, lr=1e-2)
        return (
            fits,
            train,
            val,
            train_mlp(
                fits,
                train,
                val,
                {"val": val, "train": train},
                conf,
                torch.device("cpu"),
            ),
        )

    def test_step_zero_is_the_linear_to_the_bit(self):
        fits, train, val, run = self._run((2, 4), (11, 12), steps=0)
        assert run.times == (0,) and run.chosen.tolist() == [0, 0]
        for i, n in enumerate((2, 4)):
            for name, s in (("val", val), ("train", train)):
                pred, y = fits[n].predict(s)
                want = num_sums(pred - y, s)["num_b"]
                for j in range(2):
                    assert torch.equal(run.sums[name]["num_b"][i, j], want)

    def test_the_validation_numerator_is_never_above_the_linear(self):
        fits, _, val, run = self._run((2, 4), (11, 12))
        for i, n in enumerate((2, 4)):
            pred, y = fits[n].predict(val)
            lin = float(num_sums(pred - y, val)["num_b"].sum())
            assert float(run.sums["val"]["num_b"][i].sum(-1).mean()) <= lin
            assert float(run.val[i, :, 0].mean()) == lin

    def test_side_by_side_is_each_alone(self):
        _, _, _, both = self._run((2, 4), (11, 12))
        _, _, _, one = self._run((4,), (12,))
        torch.testing.assert_close(both.val[1, 1], one.val[0, 0], rtol=1e-5, atol=0)

    def test_choose_takes_the_least_mean_and_the_earliest_tie(self):
        val = torch.tensor(
            [
                [[3.0, 1.0, 2.0], [3.0, 3.0, 0.0]],  # means 3, 2, 1
                [[1.0, 2.0, 1.0], [1.0, 0.0, 1.0]],
            ],
            dtype=F64,
        )  # 1, 1, 1
        assert choose(val).tolist() == [2, 0]


class TestMeanLog:
    def test_trapezoids_by_hand(self):
        lags = (1, 2, 4, 8)
        r = torch.tensor(
            [math.exp(0.5), math.exp(1.0), math.exp(-1.0), math.exp(2.0)], dtype=F64
        )
        got = float(mean_log(lags, r, 2, 8))
        want = (
            (1.0 - 1.0) / 2 * math.log(2) + (-1.0 + 2.0) / 2 * math.log(2)
        ) / math.log(4)
        assert got == pytest.approx(want, rel=1e-14)

    def test_a_zero_ratio_is_nan(self):
        r = torch.tensor([1.0, 0.0, 1.0], dtype=F64)
        assert math.isnan(float(mean_log((1, 2, 4), r, 1, 4)))

    def test_draws_of_the_whole_sample_are_the_point(self):
        gen = torch.Generator().manual_seed(0)
        num, den = (
            torch.rand(3, 6, generator=gen, dtype=F64),
            torch.rand(3, 6, generator=gen, dtype=F64) + 1,
        )
        counts = torch.ones(1, 6, dtype=F64)
        got = float(mean_log_draws((1, 2, 4), num, den, counts, 1, 4)[0])
        assert got == pytest.approx(
            float(mean_log((1, 2, 4), num.sum(1) / den.sum(1), 1, 4)), rel=1e-14
        )
        draws = mean_log_draws((1, 2, 4), num, den, block_counts(6, 50, gen), 1, 4)
        assert draws.shape == (50,)

    def test_pair_blocks(self):
        b = torch.arange(5, dtype=F64)
        torch.testing.assert_close(pair_blocks(b), torch.tensor([1.0, 9.0], dtype=F64))
