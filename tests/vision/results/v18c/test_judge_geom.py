r"""``exp.vision.results.v18c.judge_geom`` (the description of V18c's time
linearity and information width).

The oracles: the stationary values of the synthetic controls are the ones
registered in the README before any number (``ou_mix_1`` and ``ou_mix_30``,
``cheb_a``; Codex's and Claude's computations), the permutation's $E[s] =
N/(N - 1)$ and $G$ 3/4; the flips of order counted by hand; and the checks'
passes and failures counted by hand.
"""

import pytest

from exp.vision.geom_eval import GeomConf
from exp.vision.results.v18c.judge_geom import _flips, checks_stage, theory

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
