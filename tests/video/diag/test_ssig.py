"""The diagnosis of the SIGReg of the data points (M17): the closed form of
E S against exp.lib.sigreg, the collapse's projections, lambda*, and the
bench."""

import math

import pytest
import torch

from exp.lib.sigreg import epps_pulley, sigreg
from exp.video.diag.ssig import SCALES, bench, bound, expected_s, spread_of_sums
from tests.video.packed import tiny


def _unit(a: torch.Tensor) -> torch.Tensor:
    return a / a.norm(dim=-1, keepdim=True)


@pytest.mark.parametrize("v", [1.0, 0.4, 2.5])
def test_the_expected_statistic_of_iid_normals_is_the_mean_of_epps_pulley(v):
    """E S of n iid N(0, v) along a direction is exact for the trapezoid
    statistic: the mean of 4000 draws within 4 standard errors."""
    g = torch.Generator().manual_seed(0)
    n, draws = 16, 4000
    x = math.sqrt(v) * torch.randn(n, draws, generator=g, dtype=torch.float64)
    s = epps_pulley(x)  # one statistic per column
    se = float(s.std()) / math.sqrt(draws)
    want = float(expected_s(torch.tensor(v), n))
    assert abs(float(s.mean()) - want) < 4 * se


def test_the_sums_of_a_unit_direction_have_mean_one_and_t_1_is_one():
    g = torch.Generator().manual_seed(1)
    s2 = spread_of_sums(8, 3, 20000, g)
    assert float(s2.mean()) == pytest.approx(1.0, abs=0.02)
    # T |P a|^2 with |P a|^2 ~ Beta(d/2, (T-1) d/2): variance 2 (T - 1) / (T d + 2)
    assert float(s2.var()) == pytest.approx(2 * 7 / (8 * 3 + 2), rel=0.05)
    torch.testing.assert_close(
        spread_of_sums(1, 3, 5, g), torch.ones(5, dtype=torch.float64)
    )


def test_the_collapse_costs_the_mean_over_directions_of_the_closed_form():
    """Data points of one N(0, I_d) token repeated T times: exp.lib.sigreg of
    n of them along fresh unit directions has the mean E_a E S(|s_a|^2),
    within 4 standard errors."""
    g = torch.Generator().manual_seed(2)
    t, d, n, draws = 4, 2, 16, 3000
    vals, s2 = [], []
    for _ in range(draws):
        c = torch.randn(n, 1, d, generator=g, dtype=torch.float64)
        x = c.expand(n, t, d).reshape(n, t * d)
        a = _unit(torch.randn(1, t * d, generator=g, dtype=torch.float64))
        vals.append(float(sigreg(x, x[:0], a)))
        s2.append(float((a.view(t, d).sum(0) ** 2).sum()))
    v = torch.tensor(vals)
    want = float(expected_s(torch.tensor(s2), n).mean())
    assert abs(float(v.mean()) - want) < 4 * float(v.std()) / math.sqrt(draws)


def test_at_lambda_star_the_collapse_and_iid_tokens_cost_the_same():
    b = bound(16, 4, 33, draws=2000)
    lam = b["lam_star"]
    collapse = lam * b["s_collapse"]
    iid = (1 - lam) * 1.0 + lam * b["s_iid"]
    assert collapse == pytest.approx(iid, rel=1e-12)
    assert b["s_collapse"] > b["s_iid"] and 0 < lam < 1


def test_at_the_scaled_lambda_star_the_collapse_costs_the_cheapest_iid_tokens():
    """lambda*_v: the collapse ties with the iid tokens of the cheapest scale
    on SCALES, below it is cheaper than all of them, and it is at most
    lambda* (the scale 1)."""
    n = 33
    b = bound(16, 4, n, draws=2000)
    v = SCALES.to(torch.float64)
    s_v = expected_s(v, n)

    def iid(lam: float) -> float:
        return float(((1 - lam) * v + lam * s_v).min())

    lam = b["lam_star_scaled"]
    assert lam * b["s_collapse"] == pytest.approx(iid(lam), rel=1e-12)
    below = 0.99 * lam
    assert below * b["s_collapse"] < iid(below)
    assert 0 < lam <= b["lam_star"]


def test_the_bench_runs_both_stores_with_the_buffer_full():
    over = tiny("none") + [
        "+experiment=m17/ssig_t128",
        "vision.token_dim=8",
        "vision.datapoint_buffer=4",
    ]
    over = [o for o in over if not o.startswith("+experiment=m17/wt_n16")]
    for store in ("z", "input"):
        out = bench(over, store=store, frames=2, warmup=1)
        assert out["store"] == store and out["buffer"] == 4
        assert out["steps_per_s"] > 0


def test_the_bench_runs_the_gauss_preset_with_its_moments():
    """m17/gkl_t128 (the stored representations only): the full buffer makes
    its moments at each step's insertion."""
    over = tiny("none") + [
        "+experiment=m17/gkl_t128",
        "vision.sigreg=0.5",  # mandatory in the preset; the cost does not see it
        "vision.token_dim=1",
        "vision.datapoint_buffer=40",
    ]
    over = [o for o in over if not o.startswith("+experiment=m17/wt_n16")]
    out = bench(over, store="z", frames=2, warmup=1)
    assert out["buffer"] == 40 and out["steps_per_s"] > 0
