r"""``results/dyn/judge_dyn.py``: the words of hand-made intervals, the
$\sigma_{ab}$ rule, the Bonferroni factor, and the interval of pairs whose
$\psi$ is known exactly (a numerator scaled by $e^{c}$ at every block: every
draw is $c$, the standard error 0): of the MLP $\pm 2\sigma_{ab}$ with
$s_{ab}$ of the pair's seeds counted and each denominator apart, of the
linear no spread, the shifted blocks beside; the unmeasurable codes; V-p2d's
invalid draws."""

import math

import pytest
import torch

from exp.lib.dynpred import GRID
from exp.vision.results.dyn.judge_dyn import (
    Interval,
    bonferroni_z,
    pair,
    psi_interval,
    sigma_ab,
    summary,
    unmeasurable,
    vp2d,
    words,
)

F64 = torch.float64


def iv(lo, hi):
    return Interval(
        None if lo is None else (lo + hi) / 2, lo, hi, 0.0, 0.0, 0.0, 2.0, 0.0
    )


class TestWords:
    @pytest.mark.parametrize(
        ("copy", "clim", "want"),
        [
            ((-2, -1), (-3, -0.5), "a_better"),
            ((1, 2), (0.1, 3), "b_better"),
            ((-2, -1), (0.5, 1), "split"),
            ((0.5, 1), (-2, -1), "split"),
            ((-2, 1), (-3, -1), "none"),
            ((-2, -1), (None, None), "none"),
        ],
    )
    def test_the_table(self, copy, clim, want):
        assert words(iv(*copy), iv(*clim)) == want


class TestRules:
    def test_sigma_ab_is_the_larger_sample_sd(self):
        assert sigma_ab([0.0, 1.0, 2.0], [0.0, 0.0, 3.0]) == pytest.approx(math.sqrt(3))
        assert sigma_ab([0.0, 2.0, 4.0], [1.0, 1.0, 1.0]) == pytest.approx(2.0)

    def test_bonferroni(self):
        assert bonferroni_z(1) == pytest.approx(1.959964, abs=1e-6)
        assert bonferroni_z(4) == pytest.approx(2.497705, abs=1e-6)


class TestInterval:
    def _sums(self, seed: int = 0, blocks: int = 30):
        gen = torch.Generator().manual_seed(seed)
        num = torch.rand(len(GRID), blocks, generator=gen, dtype=F64) + 0.1
        den = torch.rand(len(GRID), blocks, generator=gen, dtype=F64) + 1
        return num, den

    def test_a_scaled_numerator_is_its_log(self):
        num, den = self._sums()
        got = psi_interval(GRID, num * math.exp(-0.7), den, num, den, boot=200)
        assert got.psi == pytest.approx(-0.7, abs=1e-12)
        assert got.lo == pytest.approx(-0.7, abs=1e-9) and got.hi == pytest.approx(
            -0.7, abs=1e-9
        )
        wide = psi_interval(
            GRID, num * math.exp(-0.7), den, num, den, boot=200, sigma=0.5, z=2.0
        )
        assert wide.lo == pytest.approx(-1.7, abs=1e-9) and wide.hi == pytest.approx(
            0.3, abs=1e-9
        )

    def test_independent_numerators_have_a_spread(self):
        a, den = self._sums(1)
        b, _ = self._sums(2)
        got = psi_interval(GRID, a, den, b, den, boot=200)
        assert got.lo is not None and got.hi is not None and got.psi is not None
        assert got.lo < got.psi < got.hi

    def test_zero_denominators_give_no_interval(self):
        num, den = self._sums()
        den[:, 1:] = 0.0  # a draw misses block 0 with probability (29/30)^30 = 0.36
        got = psi_interval(GRID, num, den, num, den, boot=200)
        assert got.lo is None and got.invalid > 0.025


def _rows(num, den, family, seed=None, set_="zurich", shifted=False):
    """Rows of one family on a set: ``num``, ``den`` ``(L, B)`` by lag of
    ``GRID`` (both denominators ``den``)."""
    out = []
    for i, n in enumerate(GRID):
        r = {"part": "dyn", "set": set_, "family": family, "lag": n, "seed": seed,
             "num_b": num[i].tolist(), "copy_b": den[i].tolist(),
             "clim_b": den[i].tolist()}  # fmt: skip
        if shifted:
            r |= {"num_b_s": num[i, ::2].tolist(), "copy_b_s": den[i, ::2].tolist(),
                  "clim_b_s": den[i, ::2].tolist()}  # fmt: skip
        out.append(r)
    return out


class TestPair:
    """``pair`` of MLP rows whose seed $s$ scales code a's numerator by
    $e^{c_s}$ at every block and lag: $\\psi_s = c_s$ exactly, every draw of
    the seeds' mean the same (SE* 0), so the interval is $\\hat\\psi \\pm 2
    \\sigma_{ab}$ with $\\sigma_{ab} = \\max(\\mathrm{sd}(\\psi_{\\mathrm{rot}}),
    \\mathrm{sd}(c))$ of each denominator apart."""

    C = (-0.1, 0.0, 0.1)  # s_ab = 0.1

    def _codes(self, shifted=False):
        gen = torch.Generator().manual_seed(0)
        num = torch.rand(len(GRID), 30, generator=gen, dtype=F64) + 0.1
        den = torch.rand(len(GRID), 30, generator=gen, dtype=F64) + 1
        a, b = [], []
        for s, c in zip((1, 2, 3), self.C, strict=True):
            a += _rows(num * math.exp(c), den, "mlp", s, shifted=shifted)
            b += _rows(num, den, "mlp", s, shifted=shifted)
        a += _rows(num * math.exp(-0.3), den, "lin", shifted=shifted)
        b += _rows(num, den, "lin", shifted=shifted)
        return a, b

    def test_s_ab_widens_past_sigma_rot(self):
        a, b = self._codes()
        rot = {"copy": [0.0, 0.5, 1.0], "clim": [0.0, 0.01, 0.02]}  # sd 0.5, 0.01
        got = pair(a, b, "mlp", rot)
        assert got is not None
        psi = math.log(sum(math.exp(c) for c in self.C) / 3)
        # copy: sigma_rot 0.5 > s_ab 0.1; clim: s_ab 0.1 > sigma_rot 0.01
        for d, sig in (("copy", 0.5), ("clim", 0.1)):
            assert got["sigma"][d]["s_ab"] == pytest.approx(0.1, abs=1e-12)
            assert got[d]["sigma"] == pytest.approx(sig, abs=1e-12)
            assert got[d]["psi"] == pytest.approx(psi, abs=1e-12)
            assert got[d]["lo"] == pytest.approx(psi - 2 * sig, abs=1e-9)
            assert got[d]["hi"] == pytest.approx(psi + 2 * sig, abs=1e-9)
        # sigma_rot alone (s_ab not counted) would have given clim 0.01
        narrow = pair(a, b, "mlp", {"copy": [0.0] * 3, "clim": [0.0] * 3})
        assert narrow is not None
        assert narrow["clim"]["hi"] - narrow["clim"]["lo"] == pytest.approx(
            0.4, abs=1e-9
        )

    def test_the_linear_has_no_training_spread(self):
        a, b = self._codes()
        got = pair(a, b, "lin", {"copy": [0.0, 0.5, 1.0], "clim": [0.0, 0.5, 1.0]})
        assert got is not None
        assert got["copy"]["sigma"] == 0.0
        assert got["copy"]["lo"] == pytest.approx(-0.3, abs=1e-9)
        assert got["words"] == "a_better" and "shifted" not in got

    def test_the_shifted_blocks_beside(self):
        a, b = self._codes(shifted=True)
        got = pair(a, b, "lin")
        assert got is not None
        assert got["shifted"]["copy"]["psi"] == pytest.approx(-0.3, abs=1e-12)
        assert got["shifted"]["words"] == "a_better"


class TestUnmeasurable:
    def test_a_zero_copy_denominator(self):
        num = torch.ones(len(GRID), 4, dtype=F64)
        den = torch.ones(len(GRID), 4, dtype=F64)
        den[3] = 0.0
        a = _rows(num, den, "lin")
        assert unmeasurable(a) == {"why": "zero_copy", "lags": [GRID[3]]}
        assert summary(a)["unmeasurable"]["lags"] == [GRID[3]]
        assert pair(a, _rows(num, torch.ones_like(den), "lin"), "lin") == {
            "unmeasurable": {"a": {"why": "zero_copy", "lags": [GRID[3]]}}
        }

    def test_the_whitener(self):
        assert unmeasurable([{"part": "unmeasurable", "why": "flat"}]) == {
            "why": "flat"
        }


def test_vp2d_gives_no_se_past_the_invalid_share():
    gen = torch.Generator().manual_seed(3)
    num = torch.rand(len(GRID), 30, generator=gen, dtype=F64) + 0.1
    den = torch.rand(len(GRID), 30, generator=gen, dtype=F64) + 1
    rows = _rows(num, den, "lin", set_="rec") + _rows(num, den, "mlp", 1, set_="rec")
    ok = vp2d(rows)
    assert ok is not None and ok["pass"] and ok["se"] is not None
    den[:, 1:] = 0.0  # as test_zero_denominators_give_no_interval: ~36% invalid
    rows = _rows(num, den, "lin", set_="rec") + _rows(num, den, "mlp", 1, set_="rec")
    got = vp2d(rows)
    assert got is not None and got["se"] is None and not got["pass"]
    assert got["invalid"] > 0.025
