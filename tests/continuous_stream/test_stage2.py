import math

import pytest

from experiments.continuous_stream.stage2 import judge


def _row(preq: float, rollout: float, retention: float) -> dict[str, list[float]]:
    return {
        "preq": [preq] * 3,
        "rollout10": [rollout] * 3,
        "retention": [retention] * 3,
    }


class TestJudge:
    def test_a_clear_win_supports_h2_to_h4(self):
        t = {
            "sgd": _row(1.0, 1.0, 1.0),
            "adamw": _row(1.2, 0.9, 1.1),  # the baseline of rollout10 is 0.9
            "white-lr1": _row(0.8, 0.72, 0.9),  # -20% on both
            "white-l1": _row(0.95, 0.85, 1.0),
            "orth": _row(0.95, 0.88, 1.0),  # less than half of the gain
        }
        lines = judge(t)
        assert lines[0].startswith("H2: supported ['white-lr1']")
        assert any(x.startswith("H3 (against white-lr1): supported") for x in lines)
        assert any(x.startswith("H4 (white-lr1): supported") for x in lines)

    def test_a_gain_on_one_metric_only_rejects_h2(self):
        t = {
            "sgd": _row(1.0, 1.0, 1.0),
            "white-lr1": _row(0.8, 0.95, 0.9),
            "orth": _row(0.7, 0.9, 1.0),
        }
        lines = judge(t)
        assert lines[0].startswith("H2: rejected")
        assert any(x.startswith("H3 (against white-lr1): rejected") for x in lines)

    def test_a_collapsed_baseline_is_left_out(self):
        """AdamW with a collapsed z (low rank) has a tiny error; it is not the
        baseline, and a collapsed candidate improves nothing."""
        t = {
            "sgd": {**_row(1.0, 1.0, 1.0), "rank": [40.0] * 3},
            "adamw": {**_row(0.01, 0.01, 0.01), "rank": [2.0, 2.0, 50.0]},
            "white-lr1": {**_row(0.8, 0.8, 0.9), "rank": [40.0] * 3},
            "proj4": {**_row(0.1, 0.1, 0.1), "rank": [40.0, 40.0, 3.0]},
        }
        t["adamw"]["preq"][2] = math.nan  # did not finish
        lines = judge(t)
        assert lines[0] == "H2: supported ['white-lr1']"
        assert (
            "Collapsed baselines, left out: ['adamw-s0', 'adamw-s1', 'adamw-s2']"
            in lines
        )

    @pytest.mark.parametrize(("rollout1", "met"), [(0.5, "met"), (1.5, "not met")])
    def test_the_precondition_is_a_prediction_better_than_the_mean(self, rollout1, met):
        t = {"sgd": {**_row(1.0, 1.0, 1.0), "rollout1": [0.5, 0.5, rollout1]}}
        t["white-lr1"] = _row(0.8, 0.8, 0.9)
        assert any(
            x.startswith(f"Precondition (sgd rollout1 < 1): {met};") for x in judge(t)
        )
