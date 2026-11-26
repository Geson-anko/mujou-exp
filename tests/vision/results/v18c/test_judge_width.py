r"""``exp.vision.results.v18c.judge_width`` (the description of the linear
horizon).

The oracles: blocks of one curve each resample to that curve, so a paired
ratio's interval is the point ratio by hand and its wording follows
(``distinct``, ``small``, ``unresolved`` of hand-placed widths); a $p > 0$
at 25.6 s is a cut; a $\Delta_{\mathrm{typ}}$ ratio is given only of
windows in every draw; the first window and its reappearance of a
hand-made periodic $p$; the survival of moments of monotone widths is $p$
itself; the guard's verdicts of hand-placed $c$, $f$, $e$, $\mathrm{SE}^*$,
$\sigma_{\mathrm{dec}}$ and $L$ (passed, unverifiable, broken, pending, no
lag); $\sigma_{\mathrm{dec}}$ of hand-made seeds; V-h2's families; the
checks counted by hand.
"""

import math
from collections.abc import Sequence

import pytest
import torch

from exp.vision.geom_eval import GeomConf
from exp.vision.results.v18c.judge_width import (
    cheat_pass,
    checks_stage,
    guard_verdict,
    pair_ratio,
    sigma_dec,
)
from exp.vision.width_eval import LinSums, WidthConf, point

W = WidthConf(max_lag=8, main_lo=2, step=1.0)
G = GeomConf(boot=50)
B = 8  # blocks


def sums_of(p: Sequence[float], n_per: float = 10.0) -> LinSums:
    """Lags 1..8 whose every block has the share ``p`` of winners (and the
    survivors of monotone widths: p itself, running min)."""
    lags = torch.arange(1, len(p) + 1)
    pt = torch.tensor(p, dtype=torch.float64)
    n = torch.full((len(p), B), n_per, dtype=torch.float64)
    win = pt[:, None] * n
    surv = pt.cummin(0).values[W.main_lo - 1 :, None] * n[W.main_lo - 1 :]
    g = 0.3 * torch.ones(len(p), B, dtype=torch.float64)
    return LinSums(lags, g, torch.ones_like(g), n, {"25": win, "50": win, "asym": win},
                   {"25": surv, "50": surv})  # fmt: skip


def mean_of(p: Sequence[float]) -> float:
    x = torch.arange(W.main_lo, len(p) + 1, dtype=torch.float64)
    q = torch.tensor(p[W.main_lo - 1 :], dtype=torch.float64)
    return float(x[0] + ((q[1:] + q[:-1]) / 2).sum())  # unit spacing


class TestPairRatio:
    def test_one_curve_per_block_resamples_to_the_point(self):
        a = [1, 1, 1, 1, 1, 0.5, 0, 0]
        b = [1, 1, 0.5, 0, 0, 0, 0, 0]
        got = pair_ratio(sums_of(a), sums_of(b), W, G)
        want = math.log2(mean_of(a) / mean_of(b))
        lo, hi = got["mean"]
        assert lo == pytest.approx(want) and hi == pytest.approx(want)
        assert got["mean_cls"] == "distinct" and want > 0.5
        assert not got["cut"]
        same = pair_ratio(sums_of(a), sums_of(a), W, G)
        assert same["mean"] == (0.0, 0.0) and same["mean_cls"] == "small"
        assert same["typ_cls"] == "small"

    def test_a_p_at_the_last_lag_is_a_cut(self):
        a = [1, 1, 1, 1, 1, 1, 1, 0.2]
        assert pair_ratio(sums_of(a), sums_of(a), W, G)["cut"]

    def test_no_window_gives_no_typ_ratio(self):
        a = [1, 1, 1, 1, 0, 0, 0, 0]
        none = [0.4] * 8
        got = pair_ratio(sums_of(a), sums_of(none), W, G)
        assert got["typ"] is None and got["typ_cls"] is None
        assert got["windowed"][0] == (1.0, 0.0)


class TestPoint:
    def test_the_first_window_and_its_reappearance(self):
        p = [1, 1, 1, 0, 0, 1, 1, 0]
        pt = point(sums_of(p), W)
        assert pt["25"]["p"]["first"]["lo_cut"]
        assert 3 < pt["25"]["typ"] < 4
        assert len(pt["25"]["p"]["again"]) == 1
        again = pt["25"]["p"]["again"][0]
        assert 5 < again["hi"] < 8 and 4 < again["lo"] < 6

    def test_the_survival_of_monotone_widths_is_p(self):
        p = [1, 1, 0.9, 0.7, 0.4, 0.2, 0.1, 0.0]
        pt = point(sums_of(p), W)
        assert pt["25"]["surv"]["first"]["hi"] == pytest.approx(pt["25"]["typ"])
        assert pt["25"]["surv_mean"] == pytest.approx(pt["25"]["mean"])


def values(c: float, f: float, e: float, se: float = 0.01) -> dict:
    return {"c": c, "f": f, "e": e, "se_cf": se, "se_ef": se}


class TestGuardVerdict:
    @pytest.mark.parametrize(
        ("vals", "won", "want"),
        [
            ({8: values(0.6, 0.2, 0.5)}, [8], "passed"),
            (
                {8: values(0.6, 0.2, 0.5), 16: values(0.23, 0.2, 0.5)},
                [8, 16],
                "unverifiable",
            ),
            ({8: values(0.6, 0.2, 0.1)}, [8], "broken"),
            ({8: values(0.6, 0.2, 0.21)}, [8], "pending"),
            ({8: values(0.6, 0.2, 0.5)}, [], "no_lag"),
            ({8: values(0.6, 0.2, 0.5), 16: values(0.6, 0.2, 0.1)}, [8, 16], "broken"),
        ],
    )
    def test_by_hand(self, vals, won, want):
        assert guard_verdict(vals, won, sigma=0.02)["verdict"] == want

    def test_sigma_dec_widens_the_interval(self):
        # c - f = 0.06: clears 2 SE_tot of SE 0.01 alone (0.02), not with sigma 0.03
        v = {8: values(0.26, 0.2, 0.5)}
        assert guard_verdict(v, [8], 0.0)["verdict"] == "passed"
        assert guard_verdict(v, [8], 0.03)["verdict"] == "unverifiable"


class TestSigmaDec:
    def test_by_hand(self):
        by_seed = {
            0: {8: values(0.5, 0.1, 0), 16: values(0.5, 0.2, 0)},
            1: {8: values(0.6, 0.1, 0), 16: values(0.5, 0.2, 0)},
            2: {8: values(0.7, 0.1, 0), 16: values(0.5, 0.2, 0)},
        }
        # lag 8: c - f = 0.4, 0.5, 0.6 (variance 0.01); lag 16: 0.3 x 3 (0)
        assert sigma_dec(by_seed) == pytest.approx(math.sqrt(0.005))


def cheat(lo: float, verdict: str, mean: float) -> dict:
    return {
        "ratio": {"mean": (lo, lo + 0.2)},
        "guard": {"verdict": verdict},
        "mean": mean,
    }


class TestCheats:
    def test_the_families(self):
        cheats = {
            "base": cheat(0.0, "no_lag", 1.0),
            "mean-8": cheat(0.1, "passed", 1.1),
            "mean-32": cheat(0.6, "broken", 2.0),
            "drop-16": cheat(0.6, "broken", 1.6),
            "drop-32": cheat(0.7, "pending", 1.5),
        }
        got = cheat_pass(cheats)
        assert got["mean"]["result"] == "pass" and got["mean"]["claimable"] == [
            "mean-32"
        ]
        assert got["drop"]["result"] == "fail" and not got["drop"]["monotone"]
        none = cheat_pass({"mean-8": cheat(0.1, "passed", 1.0)})
        assert none["mean"]["result"] == "none_claimable"


class TestChecks:
    def test_counted_by_hand(self):
        rows = [
            {"part": "check", "name": "V-h0a", "pass": True},
            {"part": "check", "name": "V-h0a", "pass": False},
            {"part": "record", "name": "V-h0j"},
        ]
        got = checks_stage(rows)
        assert got["checks"] == {"V-h0a": {"pass": 1, "fail": 1}}
        assert len(got["failures"]) == 1 and len(got["records"]) == 1
