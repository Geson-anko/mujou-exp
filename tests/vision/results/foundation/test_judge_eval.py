"""``exp.vision.results.foundation.judge_eval``: the rules by hand.

The oracles: the names map to their runs and steps; each collapse line
fires on its side of the line only; the ratios are the sums' quotients; a
pair whose copy interval lies above 0 and clim interval below 0 is *slow*;
good needs ``a_better`` in both directions and the guards; an arm is good
when both seeds are and split when one is.
"""

import pytest

from exp.vision.results.foundation.judge_eval import (
    arm_verdict,
    cheats,
    collapse,
    good,
    ratios,
    run_of,
)

LN = {"k1": 26.7, "rank_floor": 13.8, "theta3": 0.54, "time_line": 0.87}
OK_ROW = {"sigreg_heldin": 1.2, "rank": 200.0, "token_r2": 0.1, "time_r2": 0.4}


@pytest.mark.parametrize(
    "code,want",
    [
        ("init", ("fnd-T0-lr1e-4-lam0.5-s0", 0)),
        ("init-s1", ("fndL-T0-lr1e-4-lam0.5-s1", 0)),
        ("L-Ss-lr1e-4-lam0.5-s1-45000", ("fndL-Ss-lr1e-4-lam0.5-s1", 45000)),
        ("T0-lr1e-5-lam0.95", ("fnd-T0-lr1e-5-lam0.95-s0", 6000)),
        ("pca", None),
        ("v16", None),
    ],
)
def test_run_of(code, want):
    assert run_of(code) == want


@pytest.mark.parametrize(
    "key,value,flag",
    [
        ("sigreg_heldin", 26.7, "K1"),
        ("rank", 13.7, "K2"),
        ("token_r2", 0.54, "K3"),
        ("time_r2", 0.871, "KT"),
    ],  # fmt: skip
)
def test_each_line(key, value, flag):
    assert not any(collapse(OK_ROW, LN).values())
    got = collapse(OK_ROW | {key: value}, LN)
    assert got[flag] and sum(got.values()) == 1


def test_ratios():
    rows = [{"part": "half", "src": "left", "dst": "right", "num_b": [1.0, 2.0],
             "copy_b": [3.0, 3.0], "clim_b": [6.0, 6.0]}]  # fmt: skip
    assert ratios(rows) == {"left-right": {"copy": 0.5, "clim": 0.25}}


def _iv(lo, hi):
    return {"lo": lo, "hi": hi}


def test_cheats():
    slow = {"copy": _iv(0.01, 0.03), "clim": _iv(-0.05, -0.01)}
    fine = {"copy": _iv(-0.03, -0.01), "clim": _iv(-0.05, -0.01)}
    unsure = {"copy": _iv(-0.01, 0.03), "clim": _iv(-0.05, -0.01)}
    got = cheats({"future": slow, "past": fine}, True, True)
    assert got["slow"] == {"future": True, "past": False}
    assert cheats({"future": unsure, "past": None}, True, True)["slow"] == {
        "future": False, "past": False}  # fmt: skip
    assert cheats({}, False, False)["constant_in_time"]
    assert cheats({}, None, False)["content_dropped"]
    assert not cheats({}, None, True)["constant_in_time"]


def test_good():
    both = {"future": {"words": "a_better"}, "past": {"words": "a_better"}}
    assert good(both, True)
    assert not good(both, False)
    assert not good(both | {"past": {"words": "split"}}, True)


def test_arm_verdict():
    def entry(g):
        return {"good": {"pca": g, "v16": True, "v17": False, "mean8": False}}

    codes = {"L-T0-lr1e-4-lam0.5-s0-45000": entry(True),
             "L-T0-lr1e-4-lam0.5-s1-45000": entry(False)}  # fmt: skip
    got = arm_verdict(codes, "T0")
    assert got == {"pca": "split", "v16": "good", "v17": "not", "mean8": "not"}
