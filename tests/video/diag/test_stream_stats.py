"""The stream's statistics (diagnosis code, M17): the sums against brute
force, the autocovariance of an AR(1) video against its closed form, the
jackknife against leaving each video out, the reference of two videos against
all their pairs, Student's t against its closed forms, the effective sample
size, the drift against all pairs, the arithmetic of the budget, the
whitening against its definition and against independent AR(1) components of
unequal scales, and the window's mean correlation against all its pairs."""

import json
import math
from types import SimpleNamespace

import pytest
import torch

from exp.video.diag.stream_stats import (
    budget,
    crossing,
    curve,
    drift,
    lag_spectrum,
    lag_sums,
    log_lags,
    n_eff,
    reference,
    run_speed,
    settled,
    stack_codes,
    stack_products,
    t_quantile,
    whitener,
    window_mean,
)


def test_log_lags_are_a_third_of_an_octave_apart():
    lags = log_lags(64)
    assert lags[:4] == [1, 2, 3, 4]  # round(2^(1/3)) = 1, round(2^(2/3)) = 2
    assert lags[-1] == 64
    assert lags == sorted(set(lags))


def test_stack_products_are_the_sums_of_consecutive_values():
    p = torch.arange(10, dtype=torch.float64)
    torch.testing.assert_close(
        stack_products(p, 3),
        torch.tensor(
            [float(j + j - 1 + j - 2) for j in range(2, 10)], dtype=torch.float64
        ),
    )


def test_lag_sums_are_the_sums_over_the_stacks_by_brute_force():
    g = torch.Generator().manual_seed(1)
    x = torch.rand(40, 6, generator=g, dtype=torch.float64)
    own = x.mean(0)
    m = torch.rand(6, generator=g, dtype=torch.float64)
    n, lags = 3, [1, 5]
    s = lag_sums(x, own, m, lags, frames=n)
    for c, y in (("g", x - m), ("w", x - own)):
        stacks = [y[j - n + 1 : j + 1].flatten() for j in range(n - 1, 40)]
        assert s[f"sq_{c}"] == pytest.approx(sum(float(v @ v) for v in stacks))
        for i, k in enumerate(lags):
            pairs = [(stacks[j], stacks[j + k]) for j in range(len(stacks) - k)]
            assert s["anchors"][i] == len(pairs)
            assert s[f"num_{c}"][i] == pytest.approx(
                sum(float(a @ b) for a, b in pairs)
            )
            assert s[f"den0_{c}"][i] == pytest.approx(
                sum(float(a @ a) for a, _ in pairs)
            )
            assert s[f"den1_{c}"][i] == pytest.approx(
                sum(float(b @ b) for _, b in pairs)
            )
    assert s["points"] == 40 - n + 1


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
    # data points of the two videos -delta^2 / (1 + delta^2). A stack's curve
    # is the frame's.
    phi, delta, dim, n = 0.9, 1.0, 1024, 1000
    xs = [_ar1_video(n, dim, phi, s * delta, seed) for s, seed in ((1, 2), (-1, 3))]
    own = [x.mean(0) for x in xs]
    m = torch.cat(xs).mean(0)
    lags = [1, 3, 10, 30]
    phis = torch.tensor([phi**lag for lag in lags], dtype=torch.float64)
    want_g = (phis + delta**2) / (1 + delta**2)
    for frames in (1, 4):
        per = [
            lag_sums(x, o, m, lags, frames=frames) for x, o in zip(xs, own, strict=True)
        ]
        w = torch.tensor(curve(per, "w")["rho"], dtype=torch.float64)
        torch.testing.assert_close(w, phis, atol=0.03, rtol=0)
        g = curve(per, "g")
        torch.testing.assert_close(
            torch.tensor(g["rho"]).double(), want_g, atol=0.03, rtol=0
        )
        for v in g["videos"]:
            torch.testing.assert_close(
                torch.tensor(v).double(), want_g, atol=0.03, rtol=0
            )
    d = torch.stack(own) - m
    sq = torch.tensor([s["sq_g"] for s in per], dtype=torch.float64)
    pts = torch.tensor([s["points"] for s in per], dtype=torch.float64)
    ref = reference(d @ d.T, pts, sq, 4.0)
    assert ref["ref"] == pytest.approx(-(delta**2) / (1 + delta**2), abs=0.03)
    assert ref["share_pooled"] == pytest.approx(delta**2 / (1 + delta**2), abs=0.03)


def test_the_error_is_the_jackknife_over_the_videos():
    g = torch.Generator().manual_seed(6)
    xs = [
        torch.randn(30 + 5 * a, 4, generator=g, dtype=torch.float64) + a
        for a in range(4)
    ]
    own = [x.mean(0) for x in xs]
    m = torch.cat(xs).mean(0)
    per = [lag_sums(x, o, m, [1, 2], frames=2) for x, o in zip(xs, own, strict=True)]
    cur = curve(per, "g")

    def pooled(keep: list[int], i: int) -> float:
        num = sum(per[a]["num_g"][i] for a in keep)
        d0 = sum(per[a]["den0_g"][i] for a in keep)
        d1 = sum(per[a]["den1_g"][i] for a in keep)
        return num / math.sqrt(d0 * d1)

    for i in range(2):
        assert cur["rho"][i] == pytest.approx(pooled([0, 1, 2, 3], i))
        loo = torch.tensor(
            [pooled([b for b in range(4) if b != a], i) for a in range(4)]
        )
        want = math.sqrt(3 / 4 * float(((loo - loo.mean()) ** 2).sum()))
        assert cur["se"][i] == pytest.approx(want)


def test_the_reference_is_the_mean_over_all_pairs_of_two_videos():
    g = torch.Generator().manual_seed(4)
    vids = [torch.randn(k, 3, generator=g, dtype=torch.float64) + k for k in (4, 5, 6)]
    allx = torch.cat(vids)
    m = allx.mean(0)
    sq = torch.tensor([float(((x - m) ** 2).sum()) for x in vids])
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
    ref = reference(d @ d.T, n, sq)
    v = float(sq.sum() / n.sum())
    assert ref["ref"] == pytest.approx(float(pairs.mean()) / v)
    closed = -float((n**2 * (d**2).sum(1)).sum()) / float(n.sum() ** 2 - (n**2).sum())
    assert ref["ref"] == pytest.approx(closed / v)
    assert ref["ref"] < 0
    assert ref["se"] > 0


def test_t_quantile_has_the_closed_forms_of_one_and_two_degrees():
    p = 0.975
    assert t_quantile(p, 1) == pytest.approx(math.tan(math.pi * (p - 0.5)), rel=1e-4)
    assert t_quantile(p, 2) == pytest.approx(
        (2 * p - 1) / math.sqrt(2 * p * (1 - p)), rel=1e-6
    )
    assert t_quantile(p, 10**6) == pytest.approx(1.959964, rel=1e-5)  # the normal's


def test_crossing_and_the_effective_sample_size():
    assert crossing([1, 2, 4, 8], [0.9, 0.5, 0.1, 0.0], 0.1) == 4
    assert crossing([1, 2], [0.9, 0.5], 0.1) is None
    assert n_eff(0.0, 256) == 256
    assert n_eff(1.0, 256) == pytest.approx(1.0)
    assert n_eff(1 / 255, 256) == pytest.approx(128.0)  # the bias f'(0) at N/2
    assert n_eff(-1 / 255, 256) is None


def test_drift_averaged_over_every_shift_of_the_stacks_is_one():
    g = torch.Generator().manual_seed(5)
    b = torch.randn(7, 4, generator=g, dtype=torch.float64)
    assert drift(b, b) == 0.0
    # every shift s = 1..n-1 pairs each stack with another: all ordered pairs
    mean = sum(drift(b.roll(s, 0), b) for s in range(1, 7)) / 6
    assert mean == pytest.approx(1.0)


def test_stack_codes_code_consecutive_frames():
    grid = torch.arange(10 * 2, dtype=torch.uint8).reshape(10, 2, 1, 1)
    mods = SimpleNamespace(
        tokens=lambda x: x.float().flatten(2),  # (n, N, C) as the tokens
        z_of=lambda t: t.mean(1),
    )
    out = stack_codes(mods, grid, 3, ["tokens", "z"], batch=4)  # type: ignore[arg-type]
    assert out["tokens"].shape == (8, 6)
    for i in range(8):
        want = grid[i : i + 3].float().flatten(1)
        torch.testing.assert_close(out["tokens"][i], want.flatten())
        torch.testing.assert_close(out["z"][i], want.mean(0))


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


def test_whitener_makes_the_second_moment_the_identity_on_its_range():
    g = torch.Generator().manual_seed(7)
    x = torch.randn(500, 3, generator=g, dtype=torch.float64) * torch.tensor(
        [10.0, 1.0, 0.1], dtype=torch.float64
    )
    x = torch.cat([x, x[:, :1]], 1)  # a repeated column: rank 3 of 4
    rows = [x[:200], x[200:]]
    m = x.mean(0)
    w, info = whitener(rows, m, dev=torch.device("cpu"), chunk=64)
    assert info["rank"] == 3 and w.shape == (4, 3)
    y = (x - m) @ w
    torch.testing.assert_close(y.T @ y / len(x), torch.eye(3, dtype=torch.float64))
    assert sum(info["eig_top"]) == pytest.approx(1.0)


def test_the_whitened_curve_is_the_mean_over_directions_not_the_trace_ratio():
    # Independent AR(1) components of correlations phi_i and scales s_i: the
    # trace ratio is sum s_i^2 phi_i^k / sum s_i^2, the whitened mean is the
    # plain mean of phi_i^k, and the directions' extremes are min and max.
    phis, scales, n = [0.99, 0.5], [10.0, 1.0], 40000
    cols = [
        s * _ar1_video(n, 1, phi, 0.0, seed)
        for phi, s, seed in zip(phis, scales, (8, 9), strict=True)
    ]
    x = torch.cat(cols, 1)
    rows, m = [x[: n // 2], x[n // 2 :]], x.mean(0)
    lags = [1, 3]
    w, _ = whitener(rows, m, dev=torch.device("cpu"))
    white = [(r - m) @ w for r in rows]

    def rho(rs):
        per = [lag_sums(r, r.mean(0), m * 0, lags, frames=1) for r in rs]
        return curve(per, "g")["rho"]

    for i, k in enumerate(lags):
        pk = [phi**k for phi in phis]
        trace = sum(s * s * p for s, p in zip(scales, pk, strict=True)) / sum(
            s * s for s in scales
        )
        assert rho([r - m for r in rows])[i] == pytest.approx(trace, abs=0.03)
        assert rho(white)[i] == pytest.approx(sum(pk) / 2, abs=0.03)
    spec = lag_spectrum(white, lags, dev=torch.device("cpu"))
    for (lo, hi), k in zip(spec, lags, strict=True):
        assert lo == pytest.approx(min(phi**k for phi in phis), abs=0.03)
        assert hi == pytest.approx(max(phi**k for phi in phis), abs=0.03)
    exact = rho(white)  # the mean eigenvalue (of r = 2) is the curve's rho
    for (lo, hi), r in zip(spec, exact, strict=True):
        assert (lo + hi) / 2 == pytest.approx(r, rel=1e-9)


def test_window_mean_is_the_mean_over_all_pairs_of_the_window():
    phi, k = 0.9, 12
    lags = list(range(1, 20))
    rho = [phi**lag for lag in lags]
    pairs = [phi ** abs(i - j) for i in range(k) for j in range(k) if i != j]
    assert window_mean(lags, rho, k) == pytest.approx(sum(pairs) / len(pairs))
    # a rho linear in the lag is interpolated exactly between sparse lags
    sparse = [1, 4, 16]
    lin = [1 - lag / 32 for lag in sparse]
    pairs = [1 - abs(i - j) / 32 for i in range(k) for j in range(k) if i != j]
    assert window_mean(sparse, lin, k) == pytest.approx(sum(pairs) / len(pairs))
    with pytest.raises(ValueError, match="do not cover"):
        window_mean(sparse, lin, 18)


def test_settled_is_where_rho_stays_below_the_bound():
    lags, rho = [1, 2, 4, 8, 16], [0.5, 0.05, 0.2, 0.05, 0.01]
    assert crossing(lags, rho, 0.1) == 2
    assert settled(lags, rho, 0.1) == 8
    assert settled([1, 2], [0.5, 0.2], 0.1) is None
