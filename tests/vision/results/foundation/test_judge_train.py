r"""``results/foundation/judge_train.py``: the lines, the judgement of a
short run and the choice, on numbers made by hand; the pixel yardsticks on
stacks of known structure.

The oracles: each line is its formula of the given numbers; a run crosses
exactly the line its number is put across; a run not finished or with a
non-finite number is diverged; the choice is the least ``val_pred`` of the
runs neither collapsed nor diverged, none when all collapsed. Stacks whose
frames are the same along the time have pixel ``time_r2`` 1; ``color_tube``
of stacks whose tubelets are each one colour holds that colour, so
``token_r2`` of tokens that are one affine map of one shared variable is 1.
"""

import math

import pytest
import torch

from exp.vision.results.foundation.judge_train import (
    STEPS,
    choose,
    judge,
    lines,
    pixel_yard,
    run_name,
)

YARD = {"rank_color": 12.0, "token_r2_color_tube": 0.6, "time_r2_pixel": 0.8,
        "sigreg_const": 30.0}  # fmt: skip
INIT = {"sigreg_gauss": 0.5, "rank": 40.0, "token_r2": 0.4, "time_r2": 0.85}
LN = lines(YARD, INIT)
GOOD = {"step": STEPS, "val_pred": 0.3, "val_rel": 0.5, "sigreg_heldin": 1.0,
        "rank": 50.0, "token_r2": 0.5, "time_r2": 0.7}  # fmt: skip
TRAIN = [{"step": s, "loss": 1.0, "pred": 1.0, "sigreg": 1.0, "grad_norm": 1.0,
          "time_r2": 0.5} for s in (100, STEPS)]  # fmt: skip


def test_lines() -> None:
    assert LN == pytest.approx(
        {"k1": 15.25, "rank_floor": 12.0, "theta3": 0.8, "time_line": 0.9}
    )


def test_run_name() -> None:
    assert run_name("S1", 1e-4, 0.8) == "fnd-S1-lr1e-4-lam0.8-s0"
    assert run_name("Ss", 1e-5, 0.95, seed=1) == "fnd-Ss-lr1e-5-lam0.95-s1"


class TestJudge:
    def test_good(self) -> None:
        x = judge(TRAIN, [GOOD], LN)
        assert not x["diverged"] and not x["collapsed"]
        assert [x[k] for k in ("K1", "K2", "K3", "KT")] == [False] * 4

    @pytest.mark.parametrize(
        ("key", "value", "flag"),
        [
            ("sigreg_heldin", 15.25, "K1"),
            ("rank", 11.9, "K2"),
            ("token_r2", 0.8, "K3"),
            ("time_r2", 0.91, "KT"),
        ],  # fmt: skip
    )
    def test_each_line(self, key: str, value: float, flag: str) -> None:
        x = judge(TRAIN, [GOOD | {key: value}], LN)
        assert x["collapsed"] and x[flag]
        assert [k for k in ("K1", "K2", "K3", "KT") if x[k]] == [flag]

    def test_time_on_the_line_is_inside(self) -> None:
        assert not judge(TRAIN, [GOOD | {"time_r2": 0.9}], LN)["KT"]

    @pytest.mark.parametrize(
        ("train", "val"),
        [
            (TRAIN[:1], [GOOD]),
            (TRAIN, [GOOD | {"step": 5000}]),
            (TRAIN, [GOOD | {"rank": math.nan}]),
            ([TRAIN[0] | {"loss": math.inf}, TRAIN[1]], [GOOD]),
        ],  # fmt: skip
    )
    def test_diverged(self, train: list, val: list) -> None:
        x = judge(train, val, LN)
        assert x["diverged"] and x["collapsed"] and x["K1"] is None


class TestChoose:
    def _row(self, name: str, pred: float, collapsed: bool = False) -> dict:
        return {"name": name, "val_pred": pred, "collapsed": collapsed,
                "diverged": False}  # fmt: skip

    def test_least_of_the_healthy(self) -> None:
        rows = [self._row("a", 0.2, True), self._row("b", 0.4), self._row("c", 0.3)]
        pick = choose(rows)
        assert pick is not None and pick["name"] == "c"

    def test_none_when_all_collapsed(self) -> None:
        assert choose([self._row("a", 0.2, True)]) is None


class TestPixelYard:
    def test_constant_along_time(self) -> None:
        g = torch.Generator().manual_seed(0)
        frame = torch.randint(
            0, 256, (16, 1, 3, 64, 64), generator=g, dtype=torch.uint8
        )
        y = pixel_yard(frame.expand(16, 8, 3, 64, 64).contiguous(), 1.0)
        assert y["time_r2_pixel"] == pytest.approx(1.0, abs=1e-12)
        assert y["sigreg_const"] == 1.0

    def test_tube_colour_of_one_variable(self) -> None:
        # every tubelet one colour a_w v + b_w of one variable v per stack:
        # each token an affine map of any other, token_r2 = 1
        g = torch.Generator().manual_seed(1)
        v = torch.rand(64, generator=g)
        a = torch.rand(4, 3, 8, 8, generator=g)  # (T, c, gy, gx)
        b = torch.rand(4, 3, 8, 8, generator=g) * 0.5
        col = (a * v[:, None, None, None, None] * 0.5 + b) * 255  # (N, T, c, gy, gx)
        x = col.round().to(torch.uint8)
        x = x.repeat_interleave(2, 1).repeat_interleave(8, 3).repeat_interleave(8, 4)
        y = pixel_yard(x, 0.0)
        assert y["token_r2_color_tube"] == pytest.approx(1.0, abs=1e-3)
