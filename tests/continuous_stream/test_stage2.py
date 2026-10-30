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
