r"""``exp.vision.results.v18c.judge_width`` (the description of the linear
horizon).

The oracles: blocks of one curve each resample to that curve, so a paired
ratio's interval is the point ratio by hand and its wording follows
(``distinct``, ``small``, ``unresolved`` of hand-placed widths), of the main
$\bar\Delta_1$ (the integral to the first window's upper end) and of
$\bar\Delta$ beside it; two codes of no window are not called alike (C6);
the main width of a first window and its reappearance by hand; a $p > 0$
at 25.6 s is a cut; a $\Delta_{\mathrm{typ}}$ ratio is given only of
windows in every draw; the first window and its reappearance of a
hand-made periodic $p$; the survival of moments of monotone widths is $p$
itself; the guard's verdicts of hand-placed $c$, $f$, $e$, $\mathrm{SE}^*$,
$\sigma_{\mathrm{dec}}$ and $L$ (passed, unverifiable, broken, pending, no
lag) and a missing row an error; $\sigma_{\mathrm{dec}}$ of hand-made
seeds, of $c - f$ and of $e - f$; V-h2's families (a claimable strength of
no guard lag out of range); the notes of the guard on a wording; the main
whitening's rows alone among the yard's (whose sensitivity has a whitening
of the same $k$ and reference); the checks counted by hand; the models'
seed and step pairs and the values at a lag by hand.
"""

import math
from collections.abc import Sequence
from dataclasses import replace

import pytest
import torch

from exp.vision.geom_eval import GeomConf
from exp.vision.results.v18c.judge_width import (
    NO_WINDOW,
    _sums,
    at_lags,
    cheat_pass,
    checks_stage,
    guard_note,
    guard_verdict,
    model_pairs,
    pair_ratio,
    sigma_dec,
)
from exp.vision.width_eval import LinSums, WidthConf, code_rows, lin_sums, point

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
        # the main: a's window ends at 6 (p = 1/2 there), b's at 3
        # (2 + 1 + 1 + 1 + 0.75 = 5.75 and 2 + 0.75 = 2.75)
        want1 = math.log2(5.75 / 2.75)
        lo, hi = got["mean1"]
        assert lo == pytest.approx(want1) and hi == pytest.approx(want1)
        assert got["mean1_cls"] == "distinct"
        assert not got["cut"]
        same = pair_ratio(sums_of(a), sums_of(a), W, G)
        assert same["mean"] == (0.0, 0.0) and same["mean_cls"] == "small"
        assert same["typ_cls"] == "small"

    def test_a_window_to_the_last_lag_is_a_cut_a_p_there_a_tail(self):
        a = [1, 1, 1, 1, 1, 1, 1, 0.2]
        got = pair_ratio(sums_of(a), sums_of(a), W, G)
        assert not got["cut"] and got["tail"][0][0] == pytest.approx(0.2)
        b = [1, 1, 1, 1, 1, 1, 1, 0.6]
        assert pair_ratio(sums_of(b), sums_of(a), W, G)["cut"]

    def test_no_window_gives_no_typ_ratio(self):
        a = [1, 1, 1, 1, 0, 0, 0, 0]
        none = [0.4] * 8
        got = pair_ratio(sums_of(a), sums_of(none), W, G)
        assert got["typ"] is None and got["typ_cls"] is None
        assert got["windowed"][0] == (1.0, 0.0)
        # a window against none: the main width of none is 2 (the grid's
        # first lag); the window ends at sqrt(20) (between 4 and 5, linear
        # in log), its width 2 + 2 + (1 + 1/2)/2 (sqrt(20) - 4)
        w = 4 + 0.75 * (math.sqrt(20) - 4)
        lo, hi = got["mean1"]
        assert lo == pytest.approx(math.log2(w / 2)) and got["mean1_cls"] == "distinct"

    def test_two_codes_of_no_window_are_not_alike(self):
        # their main widths are 2 alike, the ratio 0: not 「小さい」 (C6)
        got = pair_ratio(sums_of([0.4] * 8), sums_of([0.3] * 8), W, G)
        assert got["mean1"] == (0.0, 0.0) and got["mean1_cls"] == NO_WINDOW


class TestPoint:
    def test_the_first_window_and_its_reappearance(self):
        p = [1, 1, 1, 0, 0, 1, 1, 0]
        pt = point(sums_of(p), W)
        # the main width stops at the first window's end (the crossing of
        # 1/2 between 3 and 4 linear in log: sqrt(12)); the mean goes on
        hi = math.sqrt(12)
        assert pt["25"]["mean1"] == pytest.approx(2 + 1 + (1 + 0.5) / 2 * (hi - 3))
        assert pt["25"]["mean"] == pytest.approx(mean_of(p))
        assert pt["25"]["mean"] > pt["25"]["mean1"]
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
        assert guard_verdict(vals, won, 0.02, 0.02)["verdict"] == want

    def test_sigma_dec_widens_the_interval(self):
        # c - f = 0.06: clears 2 SE_tot of SE 0.01 alone (0.02), not with sigma 0.03
        v = {8: values(0.26, 0.2, 0.5)}
        assert guard_verdict(v, [8], 0.0, 0.0)["verdict"] == "passed"
        assert guard_verdict(v, [8], 0.03, 0.0)["verdict"] == "unverifiable"
        # e - f = 0.03: clears 2 SE_tot of SE 0.01 alone (0.02), not with
        # sigma_ef 0.02 (2 sqrt(0.0005) = 0.045), nor does it lose
        w = {8: values(0.6, 0.2, 0.23)}
        assert guard_verdict(w, [8], 0.0, 0.0)["verdict"] == "passed"
        assert guard_verdict(w, [8], 0.0, 0.02)["verdict"] == "pending"

    def test_a_missing_row_is_an_error_not_an_empty_l(self):
        with pytest.raises(ValueError, match="no guard row"):
            guard_verdict({8: values(0.6, 0.2, 0.5)}, [8, 16], 0.02, 0.02)


class TestSigmaDec:
    def test_by_hand(self):
        by_seed = {
            0: {8: values(0.5, 0.1, 0), 16: values(0.5, 0.2, 0)},
            1: {8: values(0.6, 0.1, 0), 16: values(0.5, 0.2, 0)},
            2: {8: values(0.7, 0.1, 0), 16: values(0.5, 0.2, 0)},
        }
        # lag 8: c - f = 0.4, 0.5, 0.6 (variance 0.01); lag 16: 0.3 x 3 (0)
        assert sigma_dec(by_seed) == pytest.approx(math.sqrt(0.005))
        by_seed = {
            0: {8: values(0, 0.1, 0.1), 16: values(0, 0.2, 0.2)},
            1: {8: values(0, 0.1, 0.3), 16: values(0, 0.2, 0.2)},
            2: {8: values(0, 0.1, 0.5), 16: values(0, 0.2, 0.2)},
        }
        # lag 8: e - f = 0, 0.2, 0.4 (variance 0.04); lag 16: 0 x 3
        assert sigma_dec(by_seed, "e") == pytest.approx(math.sqrt(0.02))


def cheat(lo: float, verdict: str, mean1: float) -> dict:
    """A cheat whose main width's ratio is ``(lo, lo + 0.2)``; its
    $\bar\\Delta$, beside it, the other way (never read by the verdict)."""
    return {
        "ratio": {"mean1": (lo, lo + 0.2), "mean": (-lo - 0.2, -lo)},
        "guard": {"verdict": verdict},
        "mean1": mean1,
        "mean": -mean1,
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
        # a claimable strength whose latent wins at no guard lag: out of range
        out = cheat_pass({"mean-8": cheat(0.6, "broken", 1.5),
                          "mean-32": cheat(0.8, "no_lag", 2.0)})  # fmt: skip
        assert out["mean"]["result"] == "out_of_range"
        # a guard that passed a cheat fails, whatever the others
        out = cheat_pass({"mean-8": cheat(0.6, "passed", 1.5),
                          "mean-32": cheat(0.8, "no_lag", 2.0)})  # fmt: skip
        assert out["mean"]["result"] == "fail"


class TestGuardNote:
    def test_by_hand(self):
        assert guard_note("passed", False) == ""
        assert guard_note("unverifiable", False) == "ガードで確かめられない"
        both = guard_note("passed", True)
        assert both == "6.4 秒より長い刻みはガードの範囲外"
        assert "ガードの行が無い" in guard_note(None, False)
        assert guard_note("broken", True).count("。") == 1


class TestMainWhitening:
    def test_the_yards_rows_are_of_the_main_whitening_alone(self):
        from tests.vision.test_width_eval import WTINY, codes, tiny_sets

        sets, ws = tiny_sets()
        ev, ref = codes(sets.n)
        g = replace(G, k=2, k_side=(), k_sens=(3,), block=40)
        rows = code_rows(ev, ref, ws, g, WTINY, sens=True, labels={"place": "own"})
        # k = 2 and the reference: of c_eps and of c_eps_sens, two whitenings
        with pytest.raises(ValueError, match="two lin rows"):
            lin_sums(rows, place="own", k=2, ref="all")
        got = _sums(rows, "own", g)
        assert got is not None
        assert got.lags.tolist() == list(range(1, WTINY.max_lag + 1))
        want = lin_sums(rows, place="own", k=2, eps=g.c_eps, ref="all")
        assert torch.equal(got.num, want.num) and torch.equal(got.n, want.n)


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


class TestModelPairs:
    def test_by_hand(self):
        labels = ["a-s0@6000", "a-s1@6000", "a-s0@60000", "a-s1@60000",
                  "b-s0@60000", "v16", "c-s1@6000", "c-s1@60000"]  # fmt: skip
        got = model_pairs(labels)
        assert got["seeds"] == [("a-s0@6000", "a-s1@6000"),
                                ("a-s0@60000", "a-s1@60000")]  # fmt: skip
        assert got["steps"] == [("a-s0@60000", "a-s0@6000"),
                                ("a-s1@60000", "a-s1@6000"),
                                ("c-s1@60000", "c-s1@6000")]  # fmt: skip


class TestAtLags:
    def test_by_hand(self):
        ls = sums_of([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
        got = at_lags(ls, (2, 5, 9))
        assert set(got) == {2, 5}  # no lag 9
        assert got[2]["p25"] == pytest.approx(0.8)
        assert got[5]["p50"] == pytest.approx(0.5)
        assert got[5]["G"] == pytest.approx(0.3)
