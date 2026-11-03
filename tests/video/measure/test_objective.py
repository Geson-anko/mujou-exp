import math

import pytest

from exp.video.measure.objective import FLOOR, PENALTY, objective


def _row(
    r5: float, r10: float, cv5: float = 0.2, cv10: float = 0.4
) -> dict[str, float]:
    return {"roll5": r5, "cv5": cv5, "roll10": r10, "cv10": cv10}


class TestObjective:
    def test_constant_velocity_itself_is_zero(self):
        assert objective(_row(0.2, 0.4)) == 0.0

    def test_it_is_the_mean_of_the_log_ratios(self):
        """J = (log(0.1 / 0.2) + log(0.1 / 0.4)) / 2 = (log 0.5 + log 0.25) / 2."""
        want = (math.log(0.5) + math.log(0.25)) / 2
        assert objective(_row(0.1, 0.1)) == pytest.approx(want, rel=1e-12)

    def test_a_run_that_diverged_takes_ten_times_constant_velocity(self):
        assert PENALTY == pytest.approx(math.log(10))
        assert objective(_row(0.1, 0.1), diverged=True) == PENALTY

    @pytest.mark.parametrize("bad", [math.nan, math.inf, None, -1.0])
    def test_a_value_that_is_no_error_takes_the_penalty(self, bad):
        assert objective(_row(bad, 0.1)) == PENALTY  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", [math.nan, math.inf, None, 0.0])
    def test_a_baseline_that_is_no_error_takes_the_penalty(self, bad):
        assert objective(_row(0.1, 0.1, cv5=bad)) == PENALTY  # type: ignore[arg-type]

    def test_a_perfect_rollout_ranks_first(self):
        """roll = 0 counts as FLOOR: J = (log(FLOOR / 0.2) + log(0.1 / 0.4)) / 2,
        below any rollout with an error."""
        want = (math.log(FLOOR / 0.2) + math.log(0.25)) / 2
        assert objective(_row(0.0, 0.1)) == pytest.approx(want, rel=1e-12)
        assert objective(_row(0.0, 0.1)) < objective(_row(1e-9, 0.1))

    def test_no_run_that_went_on_ranks_below_one_that_diverged(self):
        assert objective(_row(200.0, 400.0)) == PENALTY

    def test_a_row_without_the_horizons_is_an_error(self):
        with pytest.raises(KeyError):
            objective({"roll5": 0.1, "cv5": 0.2})
