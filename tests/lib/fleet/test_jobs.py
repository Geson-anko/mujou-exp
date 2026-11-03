import sys
from pathlib import Path

import pytest

from exp.lib.fleet.jobs import (
    compose_configs,
    expand,
    group_of,
    label_of,
    make_job,
    parse_set,
)
from tests.lib.fleet.fleetkit import REPO

CONFIG = {
    "program": "dynamics",
    "name": "m14/s1",
    "seed": 0,
    "device": "cpu",
    "phase": "explore",
    "ctl": {"log_every": 1.0},
    "optim": {"kind": "sgd", "lr": 0.03},
}
CODE = {"snap": "a" * 40, "envkey": "e" * 12}
RES = {"class": "cpu", "ram_gb": 2.0, "gpu_mem_gb": 0.0}


def _job(config=CONFIG, code=CODE, res=RES, **kw):
    return make_job(batch="b", config=config, overrides=[], code=code, res=res, **kw)


class TestNames:
    def test_the_same_input_gives_the_same_run_id(self):
        assert _job()["run_id"] == _job(prio=10, tags=["x"])["run_id"]
        assert _job()["run_id"].startswith("m14-s1-")
        assert _job()["run_id"].endswith("-s0")

    def test_seed_device_ctl_phase_do_not_change_the_group(self):
        other = {
            **CONFIG,
            "seed": 2,
            "device": "cuda",
            "ctl": {"log_every": 5.0},
            "phase": "confirm",
        }
        a, b = _job(), _job(config=other)
        assert a["group"] == b["group"]
        assert a["run_id"] != b["run_id"]
        assert b["run_id"].endswith("-s2")

    @pytest.mark.parametrize(
        "change",
        [
            {"config": {**CONFIG, "optim": {"kind": "sgd", "lr": 0.01}}},
            {"code": {**CODE, "snap": "b" * 40}},
            {"res": {**RES, "class": "4090"}},
            {"entry": "exp.lib.fleet.fake"},
        ],
    )
    def test_config_snapshot_class_and_entry_change_the_group(self, change):
        assert _job(**change)["group"] != _job()["group"]

    def test_the_group_hashes_canonical_json(self):
        reordered = dict(reversed(list(CONFIG.items())))
        assert group_of(
            program="dynamics", config=reordered, snap="s", res_class="cpu"
        ) == group_of(program="dynamics", config=CONFIG, snap="s", res_class="cpu")

    def test_label(self):
        overrides = [
            "+experiment=m14/s1",
            "learner=pc",
            "dynamics=big",
            "optim.lr=0.02",
            "seed=3",
        ]
        assert label_of(overrides) == "pc big lr0.02"

    def test_extra_cannot_replace_the_jobs_keys(self):
        with pytest.raises(ValueError, match="run_id"):
            _job(extra={"run_id": "x"})


class TestExpand:
    def test_the_product_with_seeds_fastest(self):
        got = expand(
            ["+experiment=m14/s0"], ["learner=pc,bp", "optim.lr=0.1,0.2"], [0, 1]
        )
        assert len(got) == 8
        assert got[0] == ["+experiment=m14/s0", "learner=pc", "optim.lr=0.1", "seed=0"]
        assert got[1] == ["+experiment=m14/s0", "learner=pc", "optim.lr=0.1", "seed=1"]
        assert got[2] == ["+experiment=m14/s0", "learner=pc", "optim.lr=0.2", "seed=0"]
        assert got[-1] == ["+experiment=m14/s0", "learner=bp", "optim.lr=0.2", "seed=1"]

    def test_no_sweep_is_one_job(self):
        assert expand(["a=1"]) == [["a=1"]]

    def test_parse_set(self):
        assert parse_set(["fake.exit=3", "fake.err=boom", "x=[1,2]"]) == {
            "fake": {"exit": 3, "err": "boom"},
            "x": [1, 2],
        }


class TestCompose:
    def test_composes_in_the_snapshot_with_its_python(self):
        (cfg,) = compose_configs(
            Path(sys.executable), REPO, [["+experiment=m14/s0", "seed=4"]]
        )
        assert cfg["name"] == "m14/s0"
        assert cfg["seed"] == 4
        assert cfg["program"] == "dynamics"

    def test_names_the_lists_that_do_not_compose(self):
        with pytest.raises(ValueError, match="optim.kind=foo"):
            compose_configs(
                Path(sys.executable),
                REPO,
                [["+experiment=m14/s0"], ["+experiment=m14/s0", "optim.kind=foo"]],
            )
