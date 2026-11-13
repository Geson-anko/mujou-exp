"""The stream's statistics (diagnosis code, M17): the sums against brute
force, the autocovariance of an AR(1) video against its closed form, the
reference of two videos against all their pairs, and the arithmetic of the
budget and the drift."""

import json
import math

import pytest
import torch

from exp.video.diag.stream_stats import (
    between_reference,
    budget,
    curve,
    drift,
    first_reach,
    lag_sums,
    log_lags,
    run_speed,
    stack_products,
)


def test_log_lags_are_a_third_of_an_octave_apart():
    lags = log_lags(64)
    assert lags[:4] == [1, 2, 3, 4]  # round(2^(1/3)) = 1, round(2^(2/3)) = 2
    assert lags[-1] == 64
    assert lags == sorted(set(lags))


def test_stack_products_are_the_dot_of_the_flattened_stacks():
    g = torch.Generator().manual_seed(0)
    y = torch.randn(60, 5, generator=g, dtype=torch.float64)
    stride, n, lag = 3, 4, 7
    grid = torch.arange(0, 60, stride)
    anchors = grid[grid + lag < 60]
    p = (y[anchors] * y[anchors + lag]).sum(1)
    got = stack_products(p, n)
    want = [
        float(
            (
                y[int(t) - stride * (n - 1) : int(t) + 1 : stride].flatten()
                * y[
                    int(t) + lag - stride * (n - 1) : int(t) + lag + 1 : stride
                ].flatten()
            ).sum()
        )
        for t in anchors[n - 1 :]
    ]
    torch.testing.assert_close(got, torch.tensor(want, dtype=torch.float64))


def test_lag_sums_are_the_sums_of_the_products_by_brute_force():
    g = torch.Generator().manual_seed(1)
    x = torch.rand(50, 6, generator=g, dtype=torch.float64)
    m = torch.rand(6, generator=g, dtype=torch.float64)
    s = lag_sums(x, m, [1, 5], stride=2, frames=3, blocks=2)
    y, u = x - m, x - x.mean(0)
    for c, z in (("g", y), ("w", u)):
        for i, lag in enumerate([1, 5]):
            t = torch.arange(0, 50, 2)
            t = t[t + lag < 50]
            p = (z[t] * z[t + lag]).sum(1)
            assert sum(s[f"frame_{c}"][i]) == pytest.approx(float(p.sum()), rel=1e-5)
            stacks = sum(
                float((z[t[j - 2 : j + 1]] * z[t[j - 2 : j + 1] + lag]).sum())
                for j in range(2, len(t))
            )
            assert sum(s[f"stack_{c}"][i]) == pytest.approx(stacks, rel=1e-5)
            assert sum(s["frame_n"][i]) == len(t)
        grid = torch.arange(0, 50, 2)
        assert s[f"v_frame_{c}"] == pytest.approx(float((z[grid] ** 2).sum()), rel=1e-5)


def _ar1_video(n: int, dim: int, phi: float, offset: float, seed: int) -> torch.Tensor:
    """Frames offset + z_t, z_t = phi z_{t-1} + sqrt(1 - phi^2) e_t (unit
    variance per component, from the stationary start)."""
    g = torch.Generator().manual_seed(seed)
    e = torch.randn(n, dim, generator=g, dtype=torch.float64)
    z = torch.empty_like(e)
    z[0] = e[0]
    for t in range(1, n):
        z[t] = phi * z[t - 1] + math.sqrt(1 - phi * phi) * e[t]
    return offset + z


def test_the_curves_of_two_ar1_videos_are_their_closed_forms():
    # Within a video rho_w = phi^lag; centred by the pooled mean, video a has
    # rho_g = (phi^lag + delta^2) / (1 + delta^2) for offsets +-delta, and two
    # frames of the two videos -delta^2 / (1 + delta^2).
    phi, delta, dim, n = 0.9, 1.0, 1024, 3000
    xs = [_ar1_video(n, dim, phi, s * delta, seed) for s, seed in ((1, 2), (-1, 3))]
    m = torch.cat(xs).mean(0)
    lags = [1, 3, 10, 30, 100]
    per = [lag_sums(x, m, lags, stride=3, frames=4, blocks=5) for x in xs]
    phis = torch.tensor([phi**lag for lag in lags], dtype=torch.float64)
    want_g = (phis + delta**2) / (1 + delta**2)
    for kind in ("frame", "stack"):  # the stack's curve is the frame's
        w = curve(per, kind, "w")
        torch.testing.assert_close(
            torch.tensor(w["rho"], dtype=torch.float64), phis, atol=0.03, rtol=0
        )
        g = curve(per, kind, "g")
        torch.testing.assert_close(
            torch.tensor(g["rho"], dtype=torch.float64), want_g, atol=0.03, rtol=0
        )
        for v in g["videos"]:
            torch.testing.assert_close(
                torch.tensor(v["rho"], dtype=torch.float64), want_g, atol=0.03, rtol=0
            )
        assert all(s > 0 for s in w["se"])
    d = torch.stack([x.mean(0) for x in xs]) - m
    g = curve(per, "frame", "g")
    v = torch.tensor([x["v"] for x in g["videos"]])
    ref = between_reference(d @ d.T, torch.tensor([n, n]), v, g["v"])
    assert ref["ref"] == pytest.approx(-(delta**2) / (1 + delta**2), abs=0.03)
    assert ref["share_pooled"] == pytest.approx(delta**2 / (1 + delta**2), abs=0.03)
    w = curve(per, "frame", "w")
    assert first_reach(lags, w["rho"], w["se"], 0.0) == 100  # 0.9^30 = 0.04


def test_the_reference_is_the_mean_over_all_pairs_of_two_videos():
    g = torch.Generator().manual_seed(4)
    vids = [torch.randn(k, 3, generator=g, dtype=torch.float64) + k for k in (4, 5, 6)]
    allx = torch.cat(vids)
    m = allx.mean(0)
    v_pool = float(((allx - m) ** 2).sum(1).mean())
    pairs = torch.cat(
        [
            ((a - m) @ (b - m).T).flatten()
            for i, a in enumerate(vids)
            for j, b in enumerate(vids)
            if i != j
        ]
    )
    d = torch.stack([x.mean(0) for x in vids]) - m
    n = torch.tensor([4.0, 5.0, 6.0])
    ref = between_reference(d @ d.T, n, torch.ones(3), v_pool)
    assert ref["ref"] == pytest.approx(float(pairs.mean()) / v_pool)
    closed = -float((n**2 * (d**2).sum(1)).sum()) / float(n.sum() ** 2 - (n**2).sum())
    assert ref["ref"] == pytest.approx(closed / v_pool)
    assert ref["ref"] < 0


def test_first_reach_is_the_first_lag_within_the_interval():
    lags = [1, 2, 4, 8]
    rho = [0.9, 0.5, 0.1, 0.0]
    assert first_reach(lags, rho, [0.01] * 4, 0.0) == 8
    assert first_reach(lags, rho, [0.06] * 4, 0.0) == 4  # 0.1 <= 1.96 * 0.06
    assert first_reach(lags, rho, [0.0] * 4, 0.2) == 4
    assert first_reach(lags, rho, [0.0] * 4, -0.1) is None


def test_drift_is_the_change_over_the_spread():
    g = torch.Generator().manual_seed(5)
    a = torch.randn(100, 4, generator=g, dtype=torch.float64)
    a = a - a.mean(0)
    assert drift(a, a) == 0.0
    assert drift(a, 2 * a) == pytest.approx(0.25)  # |a|^2 / |2a|^2
    assert drift(a, a + 3.0) == pytest.approx(36 / float((a**2).sum(1).mean()))


def test_budget_of_a_run(tmp_path):
    run = tmp_path / "b" / "r"
    (run / "exit").mkdir(parents=True)
    config = {"data": {"streams": 4, "tours": {"advance": 3, "fps": 30.0}}}
    (run / "job.json").write_text(json.dumps({"config": config}))
    (run / "result.json").write_text(json.dumps({"frames": 6000}))
    for i, (a, b) in enumerate(((0.0, 200.0), (500.0, 600.0))):
        (run / "exit" / f"attempt-{i + 1}.json").write_text(
            json.dumps({"started": a, "ended": b})
        )
    sp = run_speed(run)
    assert sp["steps_per_s"] == pytest.approx(20.0)  # 6000 steps in 300 s
    out = budget(sp, 60.0, wall=30.0, loop=30 * 1000, frames=16, pixels=10, dims=[8])
    assert out["video_per_stream"] == pytest.approx(600.0)  # 6000 x 0.1 s
    assert out["video_per_stream_wall"] == pytest.approx(60.0)  # 20 x 30 x 0.1
    assert out["points_per_stream"] == 10
    assert out["points_total"] == 40
    assert out["points_distinct"] == 16  # the loop of 1000 s
    assert out["bytes_repr"] == {"8": 40 * 8 * 4}
    assert out["bytes_raw"] == 40 * 16 * 10
