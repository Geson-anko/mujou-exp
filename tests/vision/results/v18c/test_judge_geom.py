r"""``exp.vision.results.v18c.judge_geom`` (the description of V18c's time
linearity and information width).

The oracles: the stationary values of the synthetic controls are the ones
registered in the README before any number (``ou_mix_1`` and ``ou_mix_30``,
``cheb_a``; Codex's and Claude's computations), the permutation's $E[s] =
N/(N - 1)$ and $G$ 3/4; the flips of order counted by hand; and the checks'
passes and failures counted by hand. The apply stage's: the wording's
classes of hand-placed intervals, the judged lags of hand-made V-d2 rows, a
paired difference made exact by shifting a numerator by a multiple of its
denominator (SE 0), the simultaneous critical value of a curve repeated at
every lag (the one-lag quantile of $\lvert t \rvert$), the TwoNN's paired
difference counted by hand, and the provisional bounds of the spread's
diagnosis.
"""

import pytest
import torch

from exp.lib.metrics import block_counts, ratio_resamples
from exp.vision.geom_eval import GeomConf, ratio_se
from exp.vision.results.v18c.judge_geom import (
    _flips,
    _init_of,
    _lean,
    check_keys,
    checks_stage,
    classify,
    confirmed_lags,
    diag_flags,
    judged_lags,
    k3_hmean_same,
    paired_d,
    paired_g,
    seed_pairs,
    step0_inits,
    step_pairs,
    theory,
)

G = GeomConf()


class TestTheory:
    @pytest.mark.parametrize(
        ("code", "s", "g"),
        [
            ("ou_mix_1", (0.126, None, None, None, None, 1.000),
             (0.493, 0.498, 0.578, 0.714, 0.748, 0.750)),
            ("ou_mix_30", (0.111, None, None, None, None, 0.517),
             (0.555, 0.598, 0.657, 0.714, 0.739, 0.726)),
            ("cheb_a", (0.041, 0.159, 0.547, 1.197, 0.882, 1.024),
             (0.037, 0.139, 0.453, 0.816, 0.710, 0.758)),
        ],
    )  # fmt: skip
    def test_the_registered_values(self, code, s, g):
        for lag, want_s, want_g in zip(G.lags, s, g, strict=True):
            got = theory(code, lag, G, 38989)
            assert got is not None
            if want_s is not None:
                assert got[0] == pytest.approx(want_s, abs=6e-4)
            assert got[1] == pytest.approx(want_g, abs=6e-4)

    def test_the_permutation(self):
        got = theory("pca_perm", 8, G, 38989)
        assert got == pytest.approx((38989 / 38988, 0.75), rel=1e-15)

    def test_the_k3_has_none(self):
        assert theory("k3_z", 8, G, 100) is None


class TestFlips:
    def test_by_hand(self):
        main = {"a": 0.1, "b": 0.2, "c": 0.3}
        assert _flips(main, {"a": 0.1, "b": 0.2, "c": 0.3}) == 0
        assert _flips(main, {"a": 0.25, "b": 0.2, "c": 0.3}) == 1
        assert _flips(main, {"a": 0.3, "b": 0.2, "c": 0.1}) == 3
        assert _flips(main, {"a": 0.3, "c": 0.1}) == 1


class TestChecks:
    def test_counts(self, capsys):
        rows = [
            {"part": "check", "name": "V-d0a", "pass": True},
            {"part": "check", "name": "V-d0a", "pass": False, "lag": 8},
            {"part": "check", "name": "V-d0b", "pass": True},
            {"part": "record", "name": "gauss", "dim": 8, "d": 7.5, "each": [7.0, 8.0]},
        ]
        out = checks_stage(rows)
        assert out["checks"] == {
            "V-d0a": {"pass": 1, "fail": 1},
            "V-d0b": {"pass": 1, "fail": 0},
        }
        assert out["failures"] == [rows[1]] and out["records"] == [rows[3]]
        assert "ここで止める" in capsys.readouterr().out


# -- the apply stage -----------------------------------------------------------------

G2 = GeomConf(lags=(8, 16, 32))


def _mid_row(lag: int, num: list[float], den: list[float], place: str = "own") -> dict:
    return {
        "part": "mid", "place": place, "lag": lag, "stratum": "in", "k": 64,
        "eps": 1e-6, "ref": "all", "G": sum(num) / sum(den), "num_b": num,
        "den_b": den, "n_b": [1.0] * len(num),
    }  # fmt: skip


def _twonn_rows(ds: list[float], sub: list[float], place: str = "own") -> list[dict]:
    base = {"place": place, "k": 64, "eps": 1e-6, "ref": "all", "metric": "pre",
            "spacing": 8}  # fmt: skip
    rows = [base | {"part": "twonn", "offset": i, "d": d} for i, d in enumerate(ds)]
    return rows + [base | {"part": "twonn_sub", "d": sub}]


class TestClassify:
    """README 「言葉づかいの規則」: distinct where the interval clears
    $\\Delta_*$ wholly on one side, small where it lies in $[-\\Delta_*,
    \\Delta_*]$, unresolved otherwise."""

    @pytest.mark.parametrize(
        ("lo", "hi", "want"),
        [
            (0.031, 0.05, "distinct"), (-0.08, -0.031, "distinct"),
            (-0.02, 0.029, "small"), (-0.03, 0.03, "small"),
            (0.02, 0.04, "unresolved"), (-0.04, 0.0, "unresolved"),
            (0.03, 0.05, "unresolved"), (-0.05, 0.05, "unresolved"),
        ],
    )  # fmt: skip
    def test_by_hand(self, lo, hi, want):
        assert classify(lo, hi, 0.03) == want


class TestJudgedLags:
    def test_the_smallest_delta_at_or_above(self):
        # Delta* 0.03 -> the registered delta 0.04: lag 8 holds it apart from
        # 0, lag 16 holds it but not apart, lag 32 cannot make it (no row).
        rows = [
            {"part": "check", "name": "V-d2", "lag": 8, "delta": 0.04, "apart_from_0": True},
            {"part": "check", "name": "V-d2", "lag": 8, "delta": 0.02, "apart_from_0": True},
            {"part": "check", "name": "V-d2", "lag": 16, "delta": 0.04, "apart_from_0": False},
            {"part": "check", "name": "V-d2", "lag": 16, "delta": 0.08, "apart_from_0": True},
            {"part": "check", "name": "V-d2", "lag": 32, "delta": 0.02, "apart_from_0": True},
        ]  # fmt: skip
        assert judged_lags(rows, 0.03, G2) == (8,)
        # Delta* 0.02 -> delta 0.02 itself.
        assert judged_lags(rows, 0.02, G2) == (8, 32)

    def test_the_confirmed_lags_read_twice_delta(self):
        # The plan's note on V-d2: whether an interval clears Delta* wholly
        # is seen with twice its delta. 2 x 0.03 -> the registered 0.08: only
        # lag 16 has it apart from 0.
        rows = [
            {"part": "check", "name": "V-d2", "lag": 8, "delta": 0.04, "apart_from_0": True},
            {"part": "check", "name": "V-d2", "lag": 16, "delta": 0.08, "apart_from_0": True},
            {"part": "check", "name": "V-d2", "lag": 32, "delta": 0.04, "apart_from_0": True},
        ]  # fmt: skip
        assert confirmed_lags(rows, 0.03, G2) == (16,)


class TestPairedG:
    def test_a_shift_of_the_numerator_is_exact(self):
        # b's numerator is a's plus c times its denominator, block by block:
        # every resample's G_b - G_a is c exactly, so the SE is 0 and the
        # interval is the point c.
        den = [4.0, 5.0, 6.0, 3.0]
        num = [1.0, 3.0, 2.0, 2.5]
        out = {}
        for c in (0.05, 0.01):
            a = [_mid_row(lag, num, den) for lag in G2.lags]
            b = [_mid_row(lag, [n + c * d for n, d in zip(num, den, strict=True)], den)
                 for lag in G2.lags]  # fmt: skip
            out[c] = paired_g(b, a, "own", G2, (8, 16), 0.03)
        for lag in G2.lags:
            e = out[0.05]["lags"][lag]
            assert e["diff"] == pytest.approx(0.05, abs=1e-12)
            assert e["se"] == pytest.approx(0.0, abs=1e-12)
            assert e["lo"] == pytest.approx(0.05) and e["hi"] == pytest.approx(0.05)
        assert out[0.05]["lags"][8]["cls"] == "distinct"
        assert out[0.01]["lags"][8]["cls"] == "small"
        # lag 32 is not judged: described only.
        assert out[0.05]["lags"][32]["cls"] is None

    def test_a_distinct_lag_not_confirmed(self):
        # Judged at 8 and 16, confirmed (twice delta) at 8 only: the same
        # exact shift 0.05 is distinct at 8 and distinct_unconfirmed at 16
        # (「解像度が足りず、確かめられていない」); a small one stays small.
        den = [4.0, 5.0, 6.0, 3.0]
        num = [1.0, 3.0, 2.0, 2.5]
        a = [_mid_row(lag, num, den) for lag in G2.lags]
        for c, want in ((0.05, ("distinct", "distinct_unconfirmed")),
                        (-0.05, ("distinct", "distinct_unconfirmed")),
                        (0.01, ("small", "small"))):  # fmt: skip
            b = [_mid_row(lag, [n + c * d for n, d in zip(num, den, strict=True)], den)
                 for lag in G2.lags]  # fmt: skip
            got = paired_g(b, a, "own", G2, (8, 16), 0.03, confirmed=(8,))["lags"]
            assert (got[8]["cls"], got[16]["cls"]) == want
            assert (got[8]["cls_sim"], got[16]["cls_sim"]) == want
            assert got[32]["cls"] is None
        # Without ``confirmed`` every judged lag is confirmed (the old reading).
        b = [_mid_row(lag, [n + 0.05 * d for n, d in zip(num, den, strict=True)], den)
             for lag in G2.lags]  # fmt: skip
        assert paired_g(b, a, "own", G2, (8, 16), 0.03)["lags"][16]["cls"] == "distinct"

    def test_the_se_is_ratio_se_of_the_contrast(self):
        a = [
            _mid_row(lag, [1.0, 2.0, 1.5, 0.5], [3.0, 4.0, 2.0, 2.0]) for lag in G2.lags
        ]
        b = [
            _mid_row(lag, [2.0, 1.0, 1.0, 1.5], [3.0, 3.0, 3.0, 3.0]) for lag in G2.lags
        ]
        got = paired_g(a, b, "own", G2, (8,), 0.03)["lags"][8]
        num = torch.tensor([a[0]["num_b"], b[0]["num_b"]], dtype=torch.float64)
        den = torch.tensor([a[0]["den_b"], b[0]["den_b"]], dtype=torch.float64)
        want = ratio_se(num, den, torch.tensor([1.0, -1.0], dtype=torch.float64), G2)
        assert got["se"] == pytest.approx(want, rel=1e-12)
        assert got["diff"] == pytest.approx(a[0]["G"] - b[0]["G"], rel=1e-15)
        assert got["lo"] == pytest.approx(got["diff"] - 2 * want, rel=1e-12)

    def test_the_simultaneous_critical_value(self):
        # The same curve at every lag: the max over lags of identical
        # columns is one column's |t|, so c is the one-lag quantile.
        a1 = [_mid_row(lag, [1.0, 2.0, 1.5, 0.5], [3.0, 4.0, 2.0, 2.0]) for lag in (8,)]
        b1 = [_mid_row(lag, [2.0, 1.0, 1.0, 1.5], [3.0, 3.0, 3.0, 3.0]) for lag in (8,)]
        one = paired_g(a1, b1, "own", GeomConf(lags=(8,)), (8,), 0.03)
        a = [_mid_row(lag, a1[0]["num_b"], a1[0]["den_b"]) for lag in G2.lags]
        b = [_mid_row(lag, b1[0]["num_b"], b1[0]["den_b"]) for lag in G2.lags]
        three = paired_g(a, b, "own", G2, (8,), 0.03)
        assert three["c"] == pytest.approx(one["c"], rel=1e-12)
        r = ratio_resamples(
            torch.tensor([a1[0]["num_b"], b1[0]["num_b"]], dtype=torch.float64),
            torch.tensor([a1[0]["den_b"], b1[0]["den_b"]], dtype=torch.float64),
            block_counts(4, G2.boot, torch.Generator().manual_seed(G2.boot_seed)),
        )
        d = r[:, 0] - r[:, 1]
        t = (d - one["lags"][8]["diff"]).abs() / d.std()
        assert one["c"] == pytest.approx(float(torch.quantile(t, 0.95)), rel=1e-12)
        e = three["lags"][8]
        assert e["lo_sim"] == pytest.approx(e["diff"] - three["c"] * e["se"], rel=1e-12)

    def test_the_side_k_reads_its_own_rows(self):
        # The k = 16 rows differ from the main ones by a shift of 0.05; the
        # main ones are equal: the variant picks the k = 16 rows only.
        den, num = [4.0, 5.0, 6.0, 3.0], [1.0, 3.0, 2.0, 2.5]
        shifted = [n + 0.05 * d for n, d in zip(num, den, strict=True)]
        a = [_mid_row(8, num, den), _mid_row(8, shifted, den) | {"k": 16}]
        b = [_mid_row(8, num, den), _mid_row(8, num, den) | {"k": 16}]
        g1 = GeomConf(lags=(8,))
        assert paired_g(a, b, "own", g1, (8,))["lags"][8]["diff"] == 0.0
        side = paired_g(a, b, "own", g1, (8,), k=16)["lags"][8]
        assert side["diff"] == pytest.approx(0.05, abs=1e-12)
        assert side["cls"] == "distinct"


def _pd(a: list[dict], b: list[dict]) -> dict:
    got = paired_d(a, b, "own", G)
    assert got is not None
    return got


class TestPairedD:
    def test_by_hand(self):
        # mean d 10 and 16: -6; half-sample differences -6.5, -6, -5.5 with
        # standard deviation 0.5: the interval -6 +- 1 clears -4.
        a = _twonn_rows([9.0, 11.0], [9.5, 10.0, 10.5])
        b = _twonn_rows([15.0, 17.0], [16.0, 16.0, 16.0])
        got = _pd(a, b)
        assert got["diff"] == pytest.approx(-6.0)
        assert got["sd"] == pytest.approx(0.5)
        assert (got["lo"], got["hi"]) == pytest.approx((-7.0, -5.0))
        assert got["cls"] == "distinct"

    def test_small_and_above_20(self):
        a = _twonn_rows([10.0, 10.0], [10.0, 11.0, 12.0])
        b = _twonn_rows([11.0, 11.0], [11.0, 11.0, 11.0])
        assert _pd(a, b)["cls"] == "small"  # -1 +- 2
        hi_a = _twonn_rows([22.0, 22.0], [22.0, 22.5, 23.0])
        hi_b = _twonn_rows([30.0, 30.0], [30.0, 30.0, 30.0])
        assert _pd(hi_a, hi_b)["cls"] == "above_20"
        # one above 20 only: classified as usual (-20 +- 2, distinct).
        assert _pd(a, hi_b)["cls"] == "distinct"


class TestDiagFlags:
    @pytest.mark.parametrize(
        ("diag", "want"),
        [
            ({"V_over_k": 0.4, "var_max": 2.0, "top_share": 0.2}, ["V_over_k"]),
            ({"V_over_k": 2.1, "var_max": 2.0, "top_share": 0.2}, ["V_over_k"]),
            ({"V_over_k": 0.9, "var_max": 12.0, "top_share": 0.5}, ["var_max", "top_share"]),
            ({"V_over_k": 0.5, "var_max": 10.0, "top_share": 0.4}, []),
        ],
    )  # fmt: skip
    def test_the_provisional_bounds(self, diag, want):
        assert diag_flags(diag) == want


class TestApplyGuards:
    """The apply stage's stops and its labels, by hand."""

    def test_the_key_stops(self):
        check_keys("k1", {"a": [{"key": "k1"}, {"key": "k1"}]})
        with pytest.raises(SystemExit, match="b: key"):
            check_keys(
                "k1", {"a": [{"key": "k1"}], "b": [{"key": "k1"}, {"key": "k2"}]}
            )
        with pytest.raises(SystemExit, match="c: key"):
            check_keys("k1", {"c": []})

    def test_v_d0h_is_bit_for_bit_but_the_labels(self):
        h = [{"part": "mid", "place": "h_mean", "code": "z", "run": "r", "G": 0.5,
              "num_b": [1.0, 2.0]}]  # fmt: skip
        k3 = [h[0] | {"code": "k3_hmean", "place": "own", "run": "other"}]
        assert k3_hmean_same(k3, h)
        off = [k3[0] | {"num_b": [1.0, 2.0 + 2**-40]}]
        assert not k3_hmean_same(off, h)
        assert not k3_hmean_same([], [])

    def test_the_step_0_of_v18c_is_read_from_the_yard(self):
        vd = [
            {"part": "record", "name": "step0_same", "init": "init_c", "same": True,
             "runs": ["/x/v18c-lr1e-4-lam0.5-r0.5-s0"]},
            {"part": "record", "name": "step0_same", "init": "init_c1", "same": False,
             "runs": ["/x/v18c-lr1e-4-lam0.5-r0.5-s1"]},
        ]  # fmt: skip
        step0 = step0_inits(vd)
        assert step0 == {"v18c-lr1e-4-lam0.5-r0.5-s0": "init_c"}
        c0 = {"run": "v18c-lr1e-4-lam0.5-r0.5-s0", "family": "v18c"}
        assert _init_of(c0, step0) == "init_c"
        with pytest.raises(SystemExit, match="step 0"):
            _init_of({"run": "v18c-lr1e-4-lam0.5-r0.5-s1", "family": "v18c"}, step0)
        assert _init_of({"run": "v16", "family": "vae"}, step0) == "init_v16"
        assert _init_of({"run": "v17", "family": "vae"}, step0) is None
        v18 = {"run": "v18-long-lr1e-4-lam0.5-r0.5-s0", "family": "v18"}
        assert _init_of(v18, step0) is None

    def test_the_pairs_of_steps_and_seeds(self):
        finals = ["v18c-a-s0@60000", "v18c-a-s1@60000"]
        assert step_pairs(finals) == [
            ("v18c-a-s0@60000", "v18c-a-s0@6000"),
            ("v18c-a-s1@60000", "v18c-a-s1@6000"),
        ]
        assert seed_pairs(finals) == [("v18c-a-s0@6000", "v18c-a-s1@6000")]

    def test_lean_drops_only_the_bounds_of_a_paired_g(self):
        e = {"diff": 0.1234567, "se": 0.01, "lo": 0.1, "hi": 0.2, "lo_sim": 0.0,
             "hi_sim": 0.3, "cls": "small", "cls_sim": "small"}  # fmt: skip
        d = {"diff": 1.0, "lo": 0.5, "hi": 1.5, "cls": "small"}
        assert _lean({"lags": {8: e}, "d": d}) == {
            "lags": {8: {"diff": 0.123457, "se": 0.01, "cls": "small",
                         "cls_sim": "small"}},
            "d": d,
        }  # fmt: skip
