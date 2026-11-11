"""``sweep confirm`` and ``sweep bracket``: the runs whose numbers go to the README.

Confirm: the top k trials by their value x learners x seeds, with
``phase=confirm``, on a clean snapshot. Bracket: one trial's parameter times
each factor, x learners x seeds. Both queue through the fleet's duplicate
check, so the x1 runs of a bracket are the confirm runs themselves.
"""

from typing import Any

import optuna
import pytest
from conftest import code_info, fake_compose
from sweep.followup import FollowupError, bracket, confirm
from sweep.study import open_study
from test_coordinator import make_conf

RES = {"class": "4090", "ram_gb": 2.0, "gpu_mem_gb": 2.0}


def told(study: optuna.Study, params: dict[str, Any], value: float) -> None:
    conf_space = make_conf().space
    study.enqueue_trial(params)
    t = study.ask()
    conf_space.suggest(t)
    study.tell(t, value)


@pytest.fixture
def study() -> optuna.Study:
    s = open_study(make_conf(), optuna.storages.InMemoryStorage(), code=code_info())
    # The first trial is the enqueued sgd 0.03 of the study; tell it too.
    t = s.ask()
    make_conf().space.suggest(t)
    s.tell(t, -0.5)
    told(s, {"optim": "sgd", "optim.lr": 0.01}, -0.2)
    told(s, {"optim": "adamw", "optim.lr@adamw": 1e-4}, -0.9)
    return s


def follow(fn, study, nas, **kw):
    args: dict[str, Any] = {
        "nas": nas,
        "code": code_info(),
        "composer": fake_compose,
        "learners": ["bp", "pc"],
        "seeds": [0, 1, 2],
        "batch": "m14-s1-confirm-v1",
        "res": RES,
    }
    return fn(study, **{**args, **kw})


class TestConfirm:
    def test_the_top_k_by_value_x_learners_x_seeds(self, study, nas, fleet):
        out = follow(confirm, study, nas, top=2)
        assert len(out) == 2 * 2 * 3
        assert all(state == "new" for state, _ in out)
        jobs = fleet.queued()
        assert {j["run_id"] for j in jobs} == {j["run_id"] for _, j in out}
        for j in jobs:
            o = j["overrides"]
            assert o.count("phase=confirm") == 1 and "phase=explore" not in o
            assert sum(x.startswith("learner=") for x in o) == 1
            assert j["batch"] == "m14-s1-confirm-v1"
            assert j["res"] == RES
            assert "confirm" in j["tags"]
        # Trials -0.9 (adamw 1e-4) and -0.5 (sgd 0.03); not -0.2 (sgd 0.01).
        lrs = {x for j in jobs for x in j["overrides"] if x.startswith("optim.lr=")}
        assert lrs == {"optim.lr=0.0001", "optim.lr=0.03"}
        triples = {
            (j["config"]["learner"]["kind"], j["seed"], j["config"]["optim"]["lr"])
            for j in jobs
        }
        assert triples == {
            (le, s, lr) for le in ("bp", "pc") for s in (0, 1, 2) for lr in (1e-4, 0.03)
        }

    def test_twice_is_all_duplicates(self, study, nas, fleet):
        follow(confirm, study, nas, top=2)
        again = follow(confirm, study, nas, top=2)
        assert {state for state, _ in again} == {"duplicate"}
        assert len(fleet.queued()) == 12

    def test_dry_queues_nothing(self, study, nas, fleet):
        out = follow(confirm, study, nas, top=1, dry=True)
        assert len(out) == 6
        assert fleet.queued() == []

    def test_a_dirty_snapshot_is_refused(self, study, nas, fleet):
        with pytest.raises(FollowupError, match="clean"):
            follow(confirm, study, nas, top=1, code=code_info(dirty=True))
        unpushed = {**code_info(), "mujou_pushed": False}
        with pytest.raises(FollowupError, match="clean"):
            follow(confirm, study, nas, top=1, code=unpushed)
        assert fleet.queued() == []

    def test_any_gpu_is_refused(self, study, nas):
        # A confirmation fixes the device class of all its runs (design 3.4).
        with pytest.raises(FollowupError, match="device class"):
            follow(confirm, study, nas, top=1, res={**RES, "class": "any_gpu"})

    def test_more_than_the_completed_trials_is_refused(self, study, nas):
        with pytest.raises(FollowupError, match="3 completed"):
            follow(confirm, study, nas, top=4)

    def test_the_top_k_are_of_distinct_parameters(self, study, nas):
        # A second trial of the best parameters (adamw 1e-4) is not a second
        # of the top 2: the top 2 are still adamw 1e-4 and sgd 0.03.
        told(study, {"optim": "adamw", "optim.lr@adamw": 1e-4}, -0.8)
        out = follow(confirm, study, nas, top=2, learners=["bp"], seeds=[0])
        lrs = {j["config"]["optim"]["lr"] for _, j in out}
        assert lrs == {1e-4, 0.03}
        with pytest.raises(FollowupError, match="3 completed"):
            follow(confirm, study, nas, top=4)

    @pytest.mark.parametrize("fn", [confirm, bracket])
    def test_the_search_seeds_are_refused(self, fn, study, nas, fleet):
        # phase is out of a run's hash: a confirm run on seed 100 would be
        # the search's run of seed 100, skipped as its duplicate.
        kw: dict[str, Any] = {"top": 1}
        if fn is bracket:
            kw = {"trial": None, "param": "optim.lr", "factors": [1.0]}
        with pytest.raises(FollowupError, match="100"):
            follow(fn, study, nas, seeds=[0, 100], **kw)
        assert fleet.queued() == []


class TestBracket:
    def test_the_best_trials_rate_times_the_factors(self, study, nas, fleet):
        out = follow(
            bracket,
            study,
            nas,
            trial=None,
            param="optim.lr",
            factors=[0.5, 1.0, 2.0],
            batch="m14-s1-bracket-v1",
        )
        assert len(out) == 3 * 2 * 3
        lrs = sorted({j["config"]["optim"]["lr"] for _, j in out})
        assert lrs == pytest.approx([0.5e-4, 1e-4, 2e-4])
        for _, j in out:
            assert "phase=confirm" in j["overrides"]
            assert sum(x.startswith("optim.lr=") for x in j["overrides"]) == 1
            assert "optim=adamw" in j["overrides"]

    def test_the_x1_runs_are_the_confirm_runs(self, study, nas, fleet):
        follow(confirm, study, nas, top=1)
        out = follow(
            bracket,
            study,
            nas,
            trial=None,
            param="optim.lr",
            factors=[0.5, 1.0, 2.0],
            batch="m14-s1-bracket-v1",
        )
        dup = [j for state, j in out if state == "duplicate"]
        assert len(dup) == 2 * 3
        assert {j["config"]["optim"]["lr"] for j in dup} == {1e-4}
        assert len(fleet.queued()) == 6 + 12

    def test_a_named_trial(self, study, nas):
        out = follow(
            bracket,
            study,
            nas,
            trial=1,
            param="optim.lr",
            factors=[0.5, 2.0],
            learners=["pc"],
            seeds=[0],
        )
        assert sorted(j["config"]["optim"]["lr"] for _, j in out) == pytest.approx(
            [0.005, 0.02]
        )

    def test_a_parameter_the_trial_lacks_is_refused(self, study, nas):
        with pytest.raises(FollowupError, match="optim.tau_m"):
            follow(
                bracket,
                study,
                nas,
                trial=None,
                param="optim.tau_m",
                factors=[0.5, 1.0, 2.0],
            )


def test_a_posthoc_study_follows_each_confirm_run_by_its_posthoc(nas, fleet):
    conf = make_conf(objective_from="posthoc", posthoc={"res": "any_gpu", "ram": 9.0})
    s = open_study(conf, optuna.storages.InMemoryStorage(), code=code_info())
    t = s.ask()
    conf.space.suggest(t)
    s.tell(t, 0.7)
    out = follow(confirm, s, nas, top=1, learners=["pc"], seeds=[0, 1])
    assert [state for state, _ in out] == ["new"] * 4
    runs, posts = out[0::2], out[1::2]
    for (_, run), (_, post) in zip(runs, posts, strict=True):
        assert post["after"] == [run["run_id"]]
        assert post["batch"] == "m14-s1-confirm-v1-post"
        assert post["decode"] == {"ckpts": "all"}
        assert post["parent"] == str(nas.runs / "m14-s1-confirm-v1" / run["run_id"])
        assert post["res"] == {**RES, "ram_gb": 9.0}  # the confirm's class
    again = follow(confirm, s, nas, top=1, learners=["pc"], seeds=[0, 1])
    assert [state for state, _ in again] == ["duplicate"] * 4
