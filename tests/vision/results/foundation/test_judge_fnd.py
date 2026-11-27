r"""``results/foundation/judge_fnd.py``: the main count of layers, the
pairs' words and the checks' summary, on rows made by hand.

The oracles: the count of the least validation loss is chosen (fewer on a
tie); a code whose numerators are half of the other's at every lag, block
and seed has $\psi = \log \tfrac12$ by both denominators and the words
「a の予測が b より良いと区別できた」; doubling its copy denominators and
halving its mean's splits the denominators; the summary of the checks is
their conjunction.
"""

import json
import math

import pytest
import torch

from exp.vision.results.foundation.judge_fnd import (
    check_summary,
    gru_summary,
    judge,
    pair,
)

LAGS = [1, 2, 4, 8, 16, 32]


def _row(code_scale: float, seed: int, layers: int = 1, val: float = 1.0,
         copy: float = 1.0, clim: float = 1.0, direction: str = "future") -> dict:  # fmt: skip
    g = torch.Generator().manual_seed(seed)
    base = 1 + torch.rand(len(LAGS), 20, generator=g, dtype=torch.float64)
    z = {"num_b": (code_scale * base).tolist(), "copy_b": (2 * copy * base).tolist(),
         "clim_b": (3 * clim * base).tolist(), "n_b": [10.0] * 20}  # fmt: skip
    return {"part": "gru", "dir": direction, "layers": layers, "seed": seed, "lags": LAGS,
            "val_curve": [2.0, val], "best": 1, "sets": {"zurich": z}}  # fmt: skip


class TestSummary:
    @pytest.mark.parametrize(
        ("v1", "v2", "main"), [(1.0, 0.9, 2), (0.9, 1.0, 1), (1.0, 1.0, 1)]
    )
    def test_the_main_count_is_the_least_validation(self, v1, v2, main):
        rows = [_row(1, s, 1, v1) for s in (1, 2, 3)] + [
            _row(1, s, 2, v2) for s in (1, 2, 3)
        ]
        got = gru_summary(rows)["future"]
        assert got["main"] == main
        assert got["layers"]["1"]["mean_log"]["copy"] == pytest.approx(math.log(0.5))


class TestPair:
    def test_half_the_error_is_better(self):
        a = [_row(0.5, s) for s in (1, 2, 3)]
        b = [_row(1.0, s) for s in (1, 2, 3)]
        got = pair(a, b, "future", 1)
        assert got is not None and got["words"] == "a_better"
        assert got["copy"]["psi"] == pytest.approx(math.log(0.5))
        assert got["clim"]["psi"] == pytest.approx(math.log(0.5))

    def test_the_denominators_split(self):
        a = [_row(1.0, s, copy=2.0, clim=0.5) for s in (1, 2, 3)]
        b = [_row(1.0, s) for s in (1, 2, 3)]
        got = pair(a, b, "future", 1)
        assert got is not None and got["words"] == "split"

    def test_a_missing_side_is_none(self):
        assert pair([_row(1, 1)], [], "future", 1) is None


class TestJudge:
    def test_checks_and_a_directory(self, tmp_path):
        assert check_summary([{"part": "check", "check": "ou", "layers": 1, "ok": True},
                              {"part": "check", "check": "iid", "layers": 2, "ok": False}])["all"] is False  # fmt: skip
        for code, scale in (("pca", 1.0), ("x", 0.5)):
            d = tmp_path / "gru" / code
            d.mkdir(parents=True)
            for s in (1, 2, 3):
                (d / f"L1-future-s{s}.json").write_text(json.dumps([_row(scale, s)]))
        (tmp_path / "checks").mkdir()
        (tmp_path / "checks" / "ou-L1-s1.json").write_text(
            json.dumps([{"part": "check", "check": "ou", "layers": 1, "ok": True}])
        )
        out = judge(tmp_path)
        assert out["checks"]["all"] is True
        assert out["codes"]["x"]["pairs"]["future-L1"]["words"] == "a_better"
