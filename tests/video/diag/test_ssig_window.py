"""The window of lambda of the SIGReg of the data points on one stream (M17
diagnosis): the pair weights against every ordered pair and against a
simulated buffer (jepa.nodes.PastPoints), E S of correlated normals against
Monte Carlo of epps_pulley, the table of G against its direct value, the first
order, the interpolation of the curves, the ends of the window, and the
command on small curves."""

import json
import math

import pytest
import torch

from exp.lib.sigreg import epps_pulley
from exp.video.diag.ssig import expected_s
from exp.video.diag.ssig_window import (
    FPRIME0,
    Stream,
    common,
    kappa_of,
    load_curves,
    pair_table,
    pair_weights,
    solutions,
    table,
    windows,
)
from exp.video.jepa.nodes import PastPoints


@pytest.mark.parametrize(("n_buf", "k"), [(8, 1), (8, 3), (256, 24)])
def test_the_pair_weights_count_every_ordered_pair_of_the_samples(n_buf, k):
    lags, w = pair_weights(n_buf, k)
    assert float(w.sum()) == pytest.approx(n_buf * (n_buf + 1), rel=1e-12)
    assert float(lags.min()) == 1.0  # the newest row is a frame old
    # lag 1: the current with the newest row at phase 0 (2 pairs, 1 / k of
    # the frames), and for k = 1 two rows one insertion apart (N q of them)
    w1 = 2 / k if k > 1 else 2 + n_buf * (1 - 1 / n_buf)
    assert float(w[0]) == pytest.approx(w1, rel=1e-12)


def test_the_pair_weights_are_those_of_a_simulated_buffer():
    """PastPoints with the frame index as its data point: the mean over the
    frames (buffer full) of the sum over the ordered pairs of phi^lag equals
    sum_tau W(tau) phi^tau, within 4 standard errors of batch means."""
    n_buf, k, phi = 6, 3, 0.9
    pts = PastPoints(
        capacity=n_buf,
        every=k,
        streams=1,
        width=1,
        dtype=torch.float64,
        directions=1,
        device="cpu",
    )
    g = torch.Generator().manual_seed(0)
    one = torch.ones(1, dtype=torch.bool)
    per_frame = []
    for j in range(24000):
        pts.next_frame(None, g)
        if pts.count == n_buf and j > 40 * k:
            x = torch.cat([torch.tensor([float(j)]), pts.filled[:, 0]])
            lag = (x[:, None] - x[None, :]).abs()
            off = ~torch.eye(len(x), dtype=torch.bool)
            assert float(lag[off].min()) >= 1
            per_frame.append(float((phi ** lag[off]).sum()))
        pts.keep(torch.tensor([[float(j)]], dtype=torch.float64), one)
    sims = torch.tensor(per_frame)
    batches = sims[: len(sims) // 60 * 60].view(60, -1).mean(1)
    se = float(batches.std()) / math.sqrt(len(batches))
    lags, w = pair_weights(n_buf, k)
    want = float((w * phi**lags).sum())
    assert abs(float(sims.mean()) - want) < 4 * se


def test_the_expected_statistic_of_correlated_normals_is_the_mean_of_epps_pulley():
    """n samples along a line of variance v, correlated phi^|j - l|: E S of
    Stream equals the mean of epps_pulley within 4 standard errors."""
    n, v, phi, draws = 12, 0.8, 0.6, 6000
    j = torch.arange(n, dtype=torch.float64)
    cov = v * phi ** (j[:, None] - j[None, :]).abs()
    chol = torch.linalg.cholesky(cov)
    g = torch.Generator().manual_seed(1)
    x = chol @ torch.randn(n, draws, generator=g, dtype=torch.float64)
    s = epps_pulley(x)
    lags = torch.arange(1, n, dtype=torch.float64)
    want = Stream(torch.tensor([v]), n)(lags, 2 * (n - lags), lambda t: phi**t)
    se = float(s.std()) / math.sqrt(draws)
    assert abs(float(s.mean()) - want["s"]) < 4 * se
    assert want["iid"] == pytest.approx(float(expected_s(torch.tensor(v), n)))
    # correlation raises the statistic of the same marginals
    assert want["pairs"] > 0.5


def test_the_table_is_g_between_its_nodes():
    v = 0.7 + 0.6 * torch.rand(50, generator=torch.Generator().manual_seed(2))
    st = Stream(v.double(), 9)
    kap = torch.tensor([-0.73, -0.1, 0.0, 0.031, 0.5, 0.987, 1.0], dtype=torch.float64)
    torch.testing.assert_close(
        st.g(kap), pair_table(v.double(), kap), atol=1e-6, rtol=0
    )
    assert float(st.g(torch.tensor([0.0], dtype=torch.float64))) == pytest.approx(
        0.0, abs=1e-12
    )


def test_a_small_correlation_costs_fprime0_n_minus_1_rbar():
    n, r = 257, 1e-4
    lags, w = pair_weights(256, 6)
    out = Stream(torch.ones(1, dtype=torch.float64), n)(lags, w, lambda t: r + 0 * t)
    assert out["rbar"] == pytest.approx(r, rel=1e-9)
    assert out["first_order"] == pytest.approx(FPRIME0 * (n - 1) * r)
    assert out["pairs"] == pytest.approx(out["first_order"], rel=1e-3)


def test_kappa_is_linear_between_the_lags_and_held_or_zero_beyond():
    lags, rho = [1, 2, 4, 8], [0.8, 0.6, 0.2, 0.1]
    tau = torch.tensor([1.0, 3.0, 6.0, 8.0, 9.0, 100.0], dtype=torch.float64)
    hold = kappa_of(lags, rho, "hold")(tau)
    zero = kappa_of(lags, rho, "zero")(tau)
    want = torch.tensor([0.8, 0.4, 0.15, 0.1, 0.1, 0.1], dtype=torch.float64)
    torch.testing.assert_close(hold, want)
    torch.testing.assert_close(zero[:4], want[:4])
    assert zero[4:].tolist() == [0.0, 0.0]
    with pytest.raises(ValueError):
        kappa_of([2, 4], [0.1, 0.0], "hold")


def test_at_the_ends_of_the_window_the_losses_are_equal():
    rho, s_h, s_c, s_cw, s_0 = 0.12, 2.0, 105.0, 30.0, 1.06
    out = solutions(s_h, rho, s_c, s_cw, s_0)

    def loss(lam: float, l_x: float, s_x: float) -> float:
        return (1 - lam) * l_x + lam * s_x

    h = 1 - rho
    for key, l_x, s_x in (
        ("lam_C", 0.0, s_c),
        ("lam_Cw", 0.0, s_cw),
        ("lam_I", 1.0, s_0),
    ):
        lam = out[key]
        assert loss(lam, h, s_h) == pytest.approx(loss(lam, l_x, s_x), rel=1e-12)
    lo, mid, hi = out["grid"]
    assert lo == out["lam_Cw"] and hi == out["lam_I"]
    assert mid == pytest.approx(math.sqrt(lo * hi))
    # the window is not empty iff s_h < s_max = rho min(S_C, S_Cw) + (1 - rho) S_0
    assert out["s_max"] == pytest.approx(rho * s_cw + (1 - rho) * s_0)
    for s in (out["s_max"] - 1e-9, out["s_max"] + 1e-9):
        got = solutions(s, rho, s_c, s_cw, s_0)
        assert (got["grid"] is not None) == (s < out["s_max"])
    # a collapse cheaper than H at any lambda: no window
    empty = solutions(31.0, rho, s_c, s_cw, s_0)
    assert math.isinf(empty["lam_Cw"]) and empty["grid"] is None


def _curve(rho: list[float]) -> dict:
    return {"rho": rho, "se": [0.01] * len(rho)}


def test_the_command_reads_the_curves_and_gives_a_row_per_arm_and_span(tmp_path):
    lags = [3, 6, 12]  # video frames, stride 3: 1, 2, 4 steps of 0.1 s
    corr = {
        "stride": 3,
        "lags": lags,
        "t95": 2.0,
        "curves": {"g": _curve([0.5, 0.3, 0.1])},
    }
    rep = {
        "codes": {
            "m/tokens": {
                "stride": 3,
                "lags": lags,
                "t95": 2.0,
                "curves": {"white": _curve([0.2, 0.0, 0.0])},
            }
        }
    }
    (tmp_path / "corr.json").write_text(json.dumps(corr))
    (tmp_path / "repr-m.json").write_text(json.dumps(rep))
    curves = load_curves(tmp_path, ["pixels/g", "m/tokens/white"])
    assert curves["pixels/g"]["lags"] == [1.0, 2.0, 4.0]
    with pytest.raises(KeyError):
        load_curves(tmp_path, ["nope/g"])
    rows = windows(tmp_path, ["pixels/g", "m/tokens/white"], [0.12], [1, 3], 8, 200)
    assert len(rows) == 3 * 2 * 2  # arms x curves x spans
    for r in rows:
        assert r["low"]["rbar"] <= r["mid"]["rbar"] <= r["high"]["rbar"]
        assert r["mid_zero_tail"]["rbar"] <= r["mid"]["rbar"]
    white = [r for r in rows if r["curve"] == "m/tokens/white"]
    pix = [r for r in rows if r["curve"] == "pixels/g"]
    assert all(a["mid"]["s"] < b["mid"]["s"] for a, b in zip(white, pix, strict=True))
    assert len(table(rows).splitlines()) == 2 + len(rows)


def test_the_common_grid_is_the_intersection_of_the_windows_not_empty():
    def row(k, grid):
        return {
            "arm": "a",
            "rho": 0.1,
            "curve": "c",
            "k": k,
            "rbar_max": 0.01,
            "mid": {"grid": grid},
        }

    rows = [row(1, None), row(6, [0.1, 0.2, 0.4]), row(24, [0.05, 0.3, 0.6])]
    (out,) = common(rows)
    assert out["empty"] == [1]
    lo, mid, hi = out["grid"]
    assert (lo, hi) == (0.1, 0.4) and mid == pytest.approx(0.2)
    (none,) = common([row(1, [0.1, 0.12, 0.2]), row(6, [0.3, 0.4, 0.5])])
    assert none["empty"] == [] and none["grid"] is None
