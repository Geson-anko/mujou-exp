"""The value of a trial: the mean over its seeds of J_s (design 5.2)."""

import math

import pytest
from conftest import row
from sweep.objective import DIVERGED, SeedRun, reports, seed_value, trial_value

LOG10 = math.log(10.0)


def running(*rows: dict) -> SeedRun:
    return SeedRun(status="running", rows=list(rows))


class TestSeedValue:
    def test_the_penalty_is_ten_times_the_constant_velocity_error(self):
        # J = mean over h of log(E_h / CV_h): an error 10 x CV is log 10.
        assert DIVERGED == pytest.approx(LOG10)

    def test_a_completed_seed_is_the_objective_of_its_result(self):
        run = SeedRun(
            status="completed",
            rows=[row(60.0, -0.1), row(1800.0, -0.3)],
            result={"status": "completed", "objective": -0.5},
        )
        assert seed_value(run) == -0.5

    def test_without_one_in_the_result_the_last_val_row(self):
        run = SeedRun(
            status="completed",
            rows=[row(60.0, -0.1), row(1800.0, -0.3), row(1800.0, -9.0, "test")],
            result={"status": "completed"},
        )
        assert seed_value(run) == -0.3

    def test_a_diverged_seed_is_log_10(self):
        run = SeedRun(
            status="diverged",
            rows=[row(60.0, -0.1)],
            result={"status": "diverged", "objective": -0.1, "t_end": 90.0},
        )
        assert seed_value(run) == pytest.approx(LOG10)

    def test_a_nonfinite_objective_is_log_10(self):
        run = SeedRun(status="completed", rows=[], result={"objective": float("nan")})
        assert seed_value(run) == pytest.approx(LOG10)

    def test_a_completed_seed_without_any_objective_is_an_error(self):
        with pytest.raises(ValueError, match="objective"):
            seed_value(SeedRun(status="completed", rows=[], result={}))

    def test_the_trial_value_is_the_mean_over_seeds(self):
        a = SeedRun(status="completed", rows=[], result={"objective": -0.5})
        b = SeedRun(status="diverged", rows=[], result={"t_end": 10.0})
        assert trial_value([a, b]) == pytest.approx((-0.5 + LOG10) / 2)


class TestReports:
    def test_only_the_times_every_seed_reached(self):
        a = running(row(60.0, -0.2), row(120.0, -0.4))
        b = running(row(60.0, -0.6))
        assert reports([a, b], after=-1) == [(60, pytest.approx(-0.4))]

    def test_only_after_the_last_step_reported(self):
        a = running(row(60.0, -0.2), row(120.0, -0.4))
        b = running(row(60.0, -0.6), row(120.0, -0.2))
        assert reports([a, b], after=60) == [(120, pytest.approx(-0.3))]

    def test_a_diverged_seed_counts_log_10_from_its_end_on(self):
        a = running(row(60.0, -0.2), row(120.0, -0.4))
        b = SeedRun(
            status="diverged",
            rows=[row(60.0, -0.6), row(90.0, 50.0)],
            result={"status": "diverged", "t_end": 90.0},
        )
        assert reports([a, b], after=-1) == [
            (60, pytest.approx(-0.4)),
            (120, pytest.approx((-0.4 + LOG10) / 2)),
        ]

    def test_a_queued_seed_holds_every_report(self):
        a = running(row(60.0, -0.2))
        assert reports([a, SeedRun(status="queued", rows=[])], after=-1) == []

    def test_a_row_without_an_objective_value_is_skipped(self):
        null = {"t": 60.0, "split": "val", "objective": None}
        assert reports([running(null, row(120.0, -0.2))], after=-1) == [
            (120, pytest.approx(-0.2))
        ]

    def test_the_test_rows_are_not_the_objective(self):
        a = running(row(60.0, -0.2), row(60.0, 5.0, "test"))
        assert reports([a], after=-1) == [(60, pytest.approx(-0.2))]

    def test_the_step_is_the_time_rounded(self):
        # 300 s of frames of 1/30 s may sum to 299.99999...; the rung at 300
        # must still see it.
        a = running(row(sum([1 / 30] * 9000), -0.2))
        assert reports([a], after=-1) == [(300, pytest.approx(-0.2))]
