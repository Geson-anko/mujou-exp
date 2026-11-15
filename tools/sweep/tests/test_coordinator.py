"""The coordinator against optuna's in-memory storage and a NAS of fake runs.

The oracles are the rules of design 5.2: a trial reports the mean over its
seeds only at the times all of them reached; a diverged seed is told as log 10;
an exception (a failed run) is a FAIL; ASHA keeps the best 1/3 at each rung
(300 s, then 900 s), and a pruned trial's runs are all stopped.
"""

import math
import threading
from typing import Any

import optuna
import pytest
from conftest import SNAP_B, FakeFleet, code_info, fake_compose, row
from sweep.coordinator import Coordinator, StepError
from sweep.study import StudyConf, StudyError, open_study

from exp.lib.fleet.contract import read_json
from exp.lib.fleet.queue import Nas

LOG10 = math.log(10.0)
State = optuna.trial.TrialState


def make_conf(**study: Any) -> StudyConf:
    d: dict[str, Any] = {
        "study": {
            "name": "m14-s1-test-v1",
            "overrides": ["+experiment=m14/s1", "learner=bp", "phase=explore"],
            "seeds": [100, 101],
            "parallel": 2,
            "max_trials": 4,
            "res": "cpu",
            "n_startup_trials": 2,
            **study,
        },
        "space": {
            "optim": {"categorical": ["sgd", "adamw"]},
            "optim.lr": {
                "float": [0.001, 0.3],
                "log": True,
                "when": "optim != 'adamw'",
            },
            "optim.lr@adamw": {
                "float": [1e-5, 1e-2],
                "log": True,
                "when": "optim == 'adamw'",
            },
        },
        "enqueue": [{"optim": "sgd", "optim.lr": 0.03}],
    }
    return StudyConf.from_dict(d)


def start(nas: Nas, conf: StudyConf | None = None, storage=None, code=None):
    conf = conf or make_conf()
    storage = storage or optuna.storages.InMemoryStorage()
    code = code or code_info()
    study = open_study(conf, storage, code=code)
    coord = Coordinator(conf, study, nas=nas, code=code, composer=fake_compose)
    return coord, study, storage


def runs_of(study: optuna.Study, number: int) -> list[str]:
    return list(study.trials[number].user_attrs["runs"])


class TestSubmit:
    def test_parallel_trials_of_all_seeds_are_queued(self, nas, fleet: FakeFleet):
        coord, study, _ = start(nas)
        coord.step()
        jobs = fleet.queued()
        assert len(jobs) == 2 * 2  # parallel x seeds
        assert [t.state for t in study.trials] == [State.RUNNING] * 2
        for t in study.trials:
            mine = [j for j in jobs if j["trial"] == t.number]
            assert [j["run_id"] for j in mine] == runs_of(study, t.number)
            assert [j["seed"] for j in mine] == [100, 101]
            for j in mine:
                assert j["study"] == "m14-s1-test-v1"
                assert j["batch"] == "m14-s1-test-v1"
                assert j["code"]["snap"] == code_info()["snap"]
                assert j["res"]["class"] == "cpu"
                assert j["overrides"][:3] == conf_overrides()
                assert j["overrides"][-1] == f"seed={j['seed']}"
        # The enqueued trial (M13's best) comes first.
        first = [j for j in jobs if j["trial"] == 0][0]
        assert "optim=sgd" in first["overrides"]
        assert "optim.lr=0.03" in first["overrides"]

    def test_nothing_more_while_the_trials_run(self, nas, fleet):
        coord, _, _ = start(nas)
        coord.step()
        coord.step()
        assert len(fleet.queued()) == 4

    def test_a_config_that_does_not_compose_fails_the_trial(self, nas, fleet):
        conf = make_conf(overrides=["+experiment=m14/s1", "bad=1"], max_trials=2)
        coord, study, _ = start(nas, conf)
        coord.step()
        assert [t.state for t in study.trials] == [State.FAIL] * 2
        assert fleet.queued() == []
        assert coord.finished()


def conf_overrides() -> list[str]:
    return ["+experiment=m14/s1", "learner=bp", "phase=explore"]


class TestReport:
    def test_only_when_every_seed_reached_the_time(self, nas, fleet):
        coord, study, _ = start(nas)
        coord.step()
        a, b = runs_of(study, 0)
        fleet.claim_all([a, b])
        fleet.live(a, [row(60.0, -0.2), row(120.0, -0.4)])
        fleet.live(b, [row(60.0, -0.6)])
        coord.step()
        assert study.trials[0].intermediate_values == {60: pytest.approx(-0.4)}
        fleet.live(b, [row(60.0, -0.6), row(120.0, -0.2)])
        coord.step()
        assert study.trials[0].intermediate_values == {
            60: pytest.approx(-0.4),
            120: pytest.approx(-0.3),
        }

    def test_the_finished_runs_are_read_from_runs(self, nas, fleet):
        coord, study, _ = start(nas)
        coord.step()
        a, b = runs_of(study, 0)
        fleet.claim_all([a, b])
        fleet.end(a, "completed", [row(60.0, -0.2), row(120.0, -0.4)], objective=-0.4)
        fleet.live(b, [row(60.0, -0.6), row(120.0, -0.2)])
        coord.step()
        assert study.trials[0].intermediate_values == {
            60: pytest.approx(-0.4),
            120: pytest.approx(-0.3),
        }
        assert study.trials[0].state == State.RUNNING


class TestTell:
    def test_a_diverged_seed_is_told_as_log_10(self, nas, fleet):
        coord, study, _ = start(nas)
        coord.step()
        a, b = runs_of(study, 0)
        fleet.claim_all([a, b])
        fleet.end(a, "completed", [row(60.0, -0.3)], objective=-0.5)
        fleet.end(b, "diverged", [row(60.0, -0.1)], objective=7.0, t_end=60.0)
        coord.step()
        t = study.trials[0]
        assert t.state == State.COMPLETE
        assert t.value == pytest.approx((-0.5 + LOG10) / 2)

    def test_a_failed_run_fails_the_trial_and_stops_the_others(self, nas, fleet):
        coord, study, _ = start(nas)
        coord.step()
        a, b = runs_of(study, 0)
        fleet.claim_all([a, b])
        fleet.end(a, "failed", reason="exit 1")
        coord.step()
        assert study.trials[0].state == State.FAIL
        assert fleet.stops()[b]["reason"] == "trial-failed"
        # Its place goes to a new trial.
        assert study.trials[-1].state == State.RUNNING
        assert len(study.trials) == 3

    def test_a_run_pruned_before_a_restart_prunes_the_trial(self, nas, fleet):
        coord, study, _ = start(nas)
        coord.step()
        a, b = runs_of(study, 0)
        fleet.claim_all([a, b])
        fleet.end(a, "stopped", reason="pruned")
        fleet.end(b, "stopped", reason="pruned")
        coord.step()
        assert study.trials[0].state == State.PRUNED

    def test_max_trials_then_finished(self, nas, fleet):
        coord, study, _ = start(nas, make_conf(max_trials=2))
        coord.step()
        assert not coord.finished()
        for n in (0, 1):
            fleet.claim_all(runs_of(study, n))
            for r in runs_of(study, n):
                fleet.end(r, "completed", [row(60.0, -0.1 * n)], objective=-0.1 * n)
        coord.step()
        assert [t.state for t in study.trials] == [State.COMPLETE] * 2
        assert coord.finished()


class TestPrune:
    def test_asha_stops_every_seed_of_the_worse_trial_at_the_rung(self, nas, fleet):
        coord, study, _ = start(nas)
        coord.step()
        t0, t1 = runs_of(study, 0), runs_of(study, 1)
        fleet.claim_all(t0)
        fleet.claim(t1[0])  # t1[1] is still in the queue
        for r in t0:
            fleet.live(r, [row(300.0, -1.0)])
        coord.step()
        # Alone at the rung of 300 s, trial 0 is the best 1/3 and goes on.
        assert study.trials[0].state == State.RUNNING
        assert fleet.stops() == {}
        fleet.live(t1[0], [row(300.0, 0.0)])
        nas.ensure("std-02")
        fleet.claim(t1[1], "std-02")
        fleet.live(t1[1], [row(300.0, 0.0)])
        coord.step()
        # Of [-1, 0] at the rung, only the smaller is promoted (1 // 3 -> top 1).
        assert study.trials[1].state == State.PRUNED
        assert study.trials[0].state == State.RUNNING
        stops = fleet.stops()
        assert set(stops) == set(t1)
        assert all(s["reason"] == "pruned" and s["now"] for s in stops.values())

    def test_a_diverged_seed_counts_log_10_at_the_rung(self, nas, fleet):
        coord, study, _ = start(nas)
        coord.step()
        t0, t1 = runs_of(study, 0), runs_of(study, 1)
        fleet.claim_all([*t0, *t1])
        for r in t0:
            fleet.live(r, [row(300.0, -1.0)])
        coord.step()
        fleet.end(t1[0], "diverged", [row(60.0, 1.0)], t_end=60.0)
        coord.step()
        assert study.trials[1].state == State.RUNNING  # seed 101 has no row yet
        fleet.live(t1[1], [row(60.0, -2.0), row(300.0, -2.0)])
        coord.step()
        # At 300 s: (log 10 - 2) / 2 = 0.15 > -1, so trial 1 is pruned.
        t = study.trials[1]
        assert t.intermediate_values[300] == pytest.approx((LOG10 - 2.0) / 2)
        assert t.state == State.PRUNED
        assert set(fleet.stops()) == {t1[1]}


class TestQueuedStop:
    def test_stopping_a_queued_run_moves_it_to_done(self, nas, fleet):
        coord, study, _ = start(nas, make_conf(seeds=[100, 101, 102]))
        coord.step()
        t0, t1 = runs_of(study, 0), runs_of(study, 1)
        fleet.claim_all(t0)
        fleet.claim_all(t1[:2])  # t1[2] stays queued
        fleet.end(t1[0], "failed", reason="exit 1")
        coord.step()
        assert study.trials[1].state == State.FAIL
        rec = read_json(nas.done / f"{t1[2]}.json")
        assert rec["status"] == "stopped"
        assert rec["reason"] == "trial-failed"
        assert t1[2] not in {j["run_id"] for j in fleet.queued()}
        assert fleet.stops()[t1[1]]["reason"] == "trial-failed"


class TestRestart:
    def test_a_new_coordinator_goes_on_reporting(self, nas, fleet):
        coord, study, storage = start(nas)
        coord.step()
        a, b = runs_of(study, 0)
        fleet.claim_all([a, b])
        for r in (a, b):
            fleet.live(r, [row(60.0, -0.2)])
        coord.step()
        n_jobs = len(fleet.queued())
        del coord

        coord2, study2, _ = start(nas, storage=storage)
        coord2.step()
        assert len(fleet.queued()) == n_jobs  # nothing submitted twice
        assert len(study2.trials) == 2
        for r in (a, b):
            fleet.live(r, [row(60.0, -0.2), row(120.0, -0.6)])
        coord2.step()
        assert study2.trials[0].intermediate_values == {
            60: pytest.approx(-0.2),
            120: pytest.approx(-0.6),
        }

    def test_a_restart_mends_what_a_crash_left(self, nas, fleet):
        coord, study, storage = start(nas)
        coord.step()
        # Stopped after queueing trial 0 but one run (an absent queue file),
        # and after asking trial 2 but before queueing it (no runs).
        lost = runs_of(study, 0)[1]
        next(p for p in nas.entries() if lost in p.name).unlink()
        study.ask()
        coord2, study2, _ = start(nas, storage=storage)
        coord2.step()
        assert study2.trials[2].state == State.FAIL
        # Absent once: a worker may be between two renames of it.
        assert lost not in {j["run_id"] for j in fleet.queued()}
        coord2.step()
        assert lost in {j["run_id"] for j in fleet.queued()}
        assert len(fleet.queued()) == 4

    def test_a_restart_draws_other_random_trials(self, nas):
        # In the random startup the sampler's seed is all there is: a new
        # coordinator seeded as the first would ask the same parameters again.
        conf = make_conf(n_startup_trials=100, parallel=8, max_trials=8)
        storage = optuna.storages.InMemoryStorage()

        def asked(n: int) -> list[dict[str, Any]]:
            study = open_study(conf, storage, code=code_info())
            out = []
            for _ in range(n):
                t = study.ask()
                out.append(conf.space.suggest(t))
            return out

        first = asked(4)[1:]  # trial 0 is the enqueued one
        again = asked(3)
        assert all(p not in first for p in again)

    def test_the_keys_of_how_it_runs_may_change(self, nas):
        _, _, storage = start(nas)
        _, study, _ = start(
            nas, make_conf(parallel=8, max_trials=40, prio=50), storage=storage
        )
        got = StudyConf.of_study(study)
        assert (got.parallel, got.max_trials, got.prio) == (8, 40, 50)

    def test_the_study_refuses_another_snapshot(self, nas):
        _, _, storage = start(nas)
        with pytest.raises(StudyError, match="snapshot"):
            start(nas, storage=storage, code=code_info(SNAP_B))

    def test_the_study_refuses_another_definition(self, nas):
        _, _, storage = start(nas)
        with pytest.raises(StudyError, match="definition"):
            start(nas, make_conf(seeds=[100, 101, 102]), storage=storage)

    def test_the_coordinator_refuses_another_snapshot(self, nas):
        _, study, _ = start(nas)
        with pytest.raises(StudyError, match="snapshot"):
            Coordinator(
                make_conf(),
                study,
                nas=nas,
                code=code_info(SNAP_B),
                composer=fake_compose,
            )


class TestDuplicates:
    def test_trials_of_the_same_params_share_their_runs(self, nas, fleet):
        conf = StudyConf.from_dict(
            {
                "study": {
                    "name": "dup-v1",
                    "overrides": ["+experiment=m14/s0"],
                    "seeds": [100],
                    "parallel": 2,
                    "max_trials": 2,
                },
                "space": {"optim": {"categorical": ["sgd", "adamw"]}},
                "enqueue": [{"optim": "sgd"}, {"optim": "sgd"}],
            }
        )
        coord, study, _ = start(nas, conf)
        coord.step()
        assert runs_of(study, 0) == runs_of(study, 1)
        assert len(fleet.queued()) == 1
        (r,) = runs_of(study, 0)
        fleet.claim(r)
        fleet.live(r, [row(300.0, 0.0)])
        coord.step()
        # Equal values at the rung: both are promoted (value <= the best).
        assert fleet.stops() == {}
        fleet.end(r, "completed", [row(300.0, 0.0)], objective=-0.7)
        coord.step()
        assert [t.value for t in study.trials] == [-0.7, -0.7]

    def test_a_run_finished_in_another_batch_is_not_run_again(self, nas, fleet):
        conf = make_conf(max_trials=1, parallel=1)
        coord, study, _ = start(nas, conf)
        coord.step()
        runs = runs_of(study, 0)
        fleet.claim_all(runs)
        for r in runs:
            fleet.end(r, "completed", [row(60.0, -0.4)], objective=-0.4)
        coord.step()
        # The same trial in a new study (a new batch) on the same snapshot.
        conf2 = make_conf(name="m14-s1-test-v2", max_trials=1, parallel=1)
        coord2, study2, _ = start(nas, conf2)
        coord2.step()
        assert runs_of(study2, 0) == runs
        assert fleet.queued() == []
        coord2.step()
        assert study2.trials[0].value == pytest.approx(-0.4)


class TestFleetErrors:
    """What breaks in the fleet (the compose of the snapshot, the NAS) costs
    no trial: the trial waits, and the next step does what is left."""

    def flaky(self, fail_at: set[int], exc: Exception):
        calls = {"n": 0}

        def compose(lists):
            calls["n"] += 1
            if calls["n"] in fail_at:
                raise exc
            return fake_compose(lists)

        return compose

    def coordinator(self, nas, conf, composer):
        study = open_study(conf, optuna.storages.InMemoryStorage(), code=code_info())
        coord = Coordinator(conf, study, nas=nas, code=code_info(), composer=composer)
        return coord, study

    @pytest.mark.parametrize(
        "exc", [FileNotFoundError("bin/python"), RuntimeError("compose failed")]
    )
    def test_a_broken_compose_is_composed_again(self, nas, fleet, exc):
        coord, study = self.coordinator(nas, make_conf(), self.flaky({2}, exc))
        with pytest.raises(StepError):
            coord.step()
        assert [t.state for t in study.trials] == [State.RUNNING] * 2
        assert "runs" not in study.trials[1].user_attrs
        coord.step()
        assert [t.state for t in study.trials] == [State.RUNNING] * 2
        assert {j["run_id"] for j in fleet.queued()} == {
            *runs_of(study, 0),
            *runs_of(study, 1),
        }

    def test_a_compose_that_stays_broken_asks_no_more_trials(self, nas, fleet):
        conf = make_conf(max_trials=50)
        coord, study = self.coordinator(
            nas, conf, self.flaky(set(range(1, 100)), RuntimeError("compose failed"))
        )
        for _ in range(5):
            with pytest.raises(StepError):
                coord.step()
        assert [t.state for t in study.trials] == [State.RUNNING]
        assert not coord.finished()
        assert fleet.queued() == []

    def test_serve_goes_on_after_a_broken_compose(self, nas, fleet):
        coord, study = self.coordinator(
            nas, make_conf(), self.flaky({1}, RuntimeError("compose failed"))
        )
        stop = threading.Event()
        steps = []

        def after() -> None:
            steps.append(1)
            if len(steps) == 2:
                stop.set()

        coord.serve(stop, every=0.0, after_step=after)
        assert len(steps) == 2
        assert len(fleet.queued()) == 4

    def test_a_run_whose_submit_failed_is_queued_again(self, nas, fleet, monkeypatch):
        coord, study = self.coordinator(nas, make_conf(), fake_compose)
        submit, calls = nas.submit, {"n": 0}

        def eio(job, **kw):
            calls["n"] += 1
            if calls["n"] == 2:
                raise OSError(5, "Input/output error")
            return submit(job, **kw)

        monkeypatch.setattr(nas, "submit", eio)
        with pytest.raises(OSError):
            coord.step()
        a, b = runs_of(study, 0)
        assert {j["run_id"] for j in fleet.queued()} == {a}
        coord.step()  # absent once
        assert b not in {j["run_id"] for j in fleet.queued()}
        coord.step()  # absent twice
        assert b in {j["run_id"] for j in fleet.queued()}
        assert study.trials[0].state == State.RUNNING


class TestStopping:
    def test_a_trial_waits_for_the_stop_of_its_run_then_runs_it(self, nas, fleet):
        conf = StudyConf.from_dict(
            {
                "study": {
                    "name": "dup-v1",
                    "overrides": ["+experiment=m14/s0"],
                    "seeds": [100, 101],
                    "parallel": 1,
                    "max_trials": 2,
                },
                "space": {"optim": {"categorical": ["sgd", "adamw"]}},
                "enqueue": [{"optim": "sgd"}, {"optim": "sgd"}],
            }
        )
        coord, study, _ = start(nas, conf)
        coord.step()
        a, b = runs_of(study, 0)
        fleet.claim_all([a, b])
        fleet.end(a, "failed", reason="exit 1")
        coord.step()
        # Trial 0 failed and asked b to stop; trial 1 (the same runs) waits.
        assert study.trials[0].state == State.FAIL
        assert fleet.stops()[b]["reason"] == "trial-failed"
        assert study.trials[1].state == State.RUNNING
        assert "runs" not in study.trials[1].user_attrs
        fleet.end(b, "stopped", reason="trial-failed")
        coord.step()
        assert runs_of(study, 1) == [a, b]
        assert {j["run_id"] for j in fleet.queued()} == {a, b}
        assert all(j["trial"] == 1 for j in fleet.queued())


class TestLate:
    def test_a_trial_seen_only_at_its_end_is_told_not_pruned(self, nas, fleet):
        coord, study, _ = start(nas)
        coord.step()
        t0, t1 = runs_of(study, 0), runs_of(study, 1)
        fleet.claim_all([*t0, *t1])
        for r in t0:
            fleet.live(r, [row(300.0, -1.0)])
        for r in t1:
            fleet.end(r, "completed", [row(300.0, 0.0)], objective=0.0)
        coord.step()
        # At the rung of 300 s 0 > -1 would prune it; it ran to its end.
        t = study.trials[1]
        assert t.state == State.COMPLETE
        assert t.value == pytest.approx(0.0)
        assert t.intermediate_values == {300: pytest.approx(0.0)}


def _post_final(objective: float | None) -> dict[str, Any]:
    val: dict[str, Any] = {"split": "val", "recon": 0.5}
    if objective is not None:
        val["objective"] = objective
    return {"decode": {"val": val, "test": {"split": "test"}}}


class TestPosthoc:
    """objective_from = "posthoc" (M17): a trial's value is the mean over its
    seeds of the objective of the posthoc job after each completed run."""

    def conf(self, **post: Any) -> StudyConf:
        return make_conf(
            objective_from="posthoc",
            posthoc={"res": "4090", "gpu_mem": 6.0, **post},
            max_trials=1,
            parallel=1,
        )

    def test_the_posthoc_jobs_follow_the_runs_and_tell_the_mean(self, nas, fleet):
        coord, study, _ = start(nas, self.conf())
        coord.step()
        runs = runs_of(study, 0)
        fleet.claim_all(runs)
        fleet.end(runs[0], "completed", [row(300.0, 9.0)])
        coord.step()
        assert not fleet.queued()  # not before every seed ended
        fleet.end(runs[1], "completed", [row(300.0, 9.0)])
        coord.step()
        posts = fleet.queued()
        assert len(posts) == 2 and study.trials[0].state == State.RUNNING
        for post, run in zip(sorted(posts, key=lambda j: j["after"]), sorted(runs)):
            assert post["after"] == [run]
            assert post["batch"] == "m14-s1-test-v1-post"
            assert post["config"]["task"] == "posthoc"
            assert post["overrides"][-1] == "task=posthoc"
            assert post["parent"] == str(nas.runs / "m14-s1-test-v1" / run)
            assert post["decode"] == {"ckpts": "last"}
            assert post["res"]["class"] == "4090" and post["res"]["gpu_mem_gb"] == 6.0
        assert study.trials[0].user_attrs["post"] == {
            j["after"][0]: j["run_id"] for j in posts
        }
        assert study.trials[0].last_step is None  # no report: ASHA never prunes
        for j, value in zip(posts, (0.8, 1.2)):
            fleet.claim(j["run_id"])
            fleet.end(j["run_id"], "completed", final=_post_final(value))
        coord.step()
        assert study.trials[0].state == State.COMPLETE
        assert study.trials[0].value == pytest.approx(1.0)
        coord.step()
        assert len(fleet.queued()) == 0  # nothing queued twice

    def test_a_diverged_seed_counts_log10_and_gets_no_posthoc(self, nas, fleet):
        coord, study, _ = start(nas, self.conf())
        coord.step()
        runs = runs_of(study, 0)
        fleet.claim_all(runs)
        fleet.end(runs[0], "completed", [row(300.0, 1.0)])
        fleet.end(runs[1], "diverged", [], t_end=120.0)
        coord.step()
        (post,) = fleet.queued()
        assert post["after"] == [runs[0]]
        fleet.claim(post["run_id"])
        fleet.end(post["run_id"], "completed", final=_post_final(0.5))
        coord.step()
        assert study.trials[0].value == pytest.approx((0.5 + LOG10) / 2)

    @pytest.mark.parametrize("how", ["failed", "no-objective"])
    def test_a_posthoc_without_a_value_fails_the_trial(self, nas, fleet, how):
        coord, study, _ = start(nas, self.conf())
        coord.step()
        runs = runs_of(study, 0)
        fleet.claim_all(runs)
        for r in runs:
            fleet.end(r, "completed", [row(300.0, 1.0)])
        coord.step()
        post = fleet.queued()[0]
        fleet.claim(post["run_id"])
        if how == "failed":
            fleet.end(post["run_id"], "failed", reason="boom")
        else:
            fleet.end(post["run_id"], "completed", final=_post_final(None))
        coord.step()
        assert study.trials[0].state == State.FAIL

    def test_the_settings_are_checked(self):
        with pytest.raises(ValueError, match="objective_from"):
            make_conf(objective_from="probe")
        with pytest.raises(ValueError, match="unknown keys"):
            make_conf(objective_from="posthoc", posthoc={"color": 1})
        with pytest.raises(ValueError, match="ckpts"):
            make_conf(objective_from="posthoc", posthoc={"ckpts": "some"})
        conf = make_conf(objective_from="posthoc", posthoc={"ckpts": "all"})
        assert StudyConf.from_dict(conf.to_dict()).to_dict() == conf.to_dict()
        assert "objective_from" not in make_conf().to_dict()["study"]
