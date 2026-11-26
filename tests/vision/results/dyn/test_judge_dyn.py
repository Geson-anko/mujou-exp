r"""``results/dyn/judge_dyn.py``: the words of hand-made intervals, the
$\sigma_{ab}$ rule, the Bonferroni factor, and the interval of pairs whose
$\psi$ is known exactly (a numerator scaled by $e^{c}$ at every block: every
draw is $c$, the standard error 0)."""

import math

import pytest
import torch

from exp.lib.dynpred import GRID
from exp.vision.results.dyn.judge_dyn import (
    Interval,
    bonferroni_z,
    psi_interval,
    sigma_ab,
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
