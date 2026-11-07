import json
from pathlib import Path

import pytest

import exp.video.conf
from exp.video.compose import compose, from_dict, to_dict
from exp.video.schema import RunConf

CONF = Path(exp.video.conf.__file__).parent
EXPERIMENTS = sorted(
    p.relative_to(CONF / "experiment").with_suffix("").as_posix()
    for p in CONF.glob("experiment/**/*.yaml")
)
GROUPS = ["dynamics", "vision", "learner", "optim", "phase"]
OPTIONS = [(g, p.stem) for g in GROUPS for p in sorted((CONF / g).glob("*.yaml"))]
# The settings a group option needs to make a valid run: without dynamics,
# the program is the vision one, which needs an encoder and its optimizer.
NEEDS = {
    ("dynamics", "none"): [
        "program=vision",
        "vision=vit_t",
        "+optim@optim_vision=sgd",
    ],
}


def test_every_yaml_is_an_experiment_or_a_group_option() -> None:
    listed = {f"experiment/{e}" for e in EXPERIMENTS} | {f"{g}/{o}" for g, o in OPTIONS}
    found = {
        p.relative_to(CONF).with_suffix("").as_posix() for p in CONF.glob("**/*.yaml")
    }
    assert found - {"run"} == listed


@pytest.mark.parametrize("experiment", EXPERIMENTS)
class TestExperiment:
    def test_composes_to_run_conf(self, experiment: str) -> None:
        conf = compose([f"+experiment={experiment}"])
        assert isinstance(conf, RunConf)
        assert conf.name == experiment

    def test_round_trips_through_json(self, experiment: str) -> None:
        conf = compose([f"+experiment={experiment}"])
        back = from_dict(json.loads(json.dumps(to_dict(conf))))
        assert isinstance(back, RunConf)
        assert back == conf


# One field per group that tells which option was applied.
WIDTHS = {"vit_t": 128, "vit_s": 256, "small": 256, "big": 512}


def _applied(conf: RunConf, group: str) -> object:
    match group:
        case "learner":
            return conf.learner.kind
        case "optim":
            return conf.optim.kind
        case "phase":
            return conf.phase
        case "vision":
            return "none" if conf.vision is None else conf.vision.dim
        case "dynamics":
            return "none" if conf.dynamics is None else conf.dynamics.d_s
        case _:
            raise AssertionError(f"no field for the group {group}")


@pytest.mark.parametrize(("group", "option"), OPTIONS)
def test_group_option_composes(group: str, option: str) -> None:
    extra = NEEDS.get((group, option), [])
    conf = compose(["+experiment=m14/s0", f"{group}={option}", *extra])
    assert _applied(conf, group) == WIDTHS.get(option, option)
    assert from_dict(to_dict(conf)) == conf


class TestM14S0:
    """Stage 0 of M14: one ball, positions and velocities, the physics of M13."""

    def test_data(self) -> None:
        conf = compose(["+experiment=m14/s0"])
        assert conf.program == "dynamics"
        assert conf.data.streams == 128
        assert conf.data.tours is None
        balls = conf.data.balls
        assert balls is not None
        assert balls.radii == [0.10]
        assert balls.obs == "pos_vel"
        p = balls.physics
        assert (p.gamma0, p.gamma_amp, p.e0, p.e_amp, p.drag) == (
            1.0,
            0.0,
            1.0,
            0.0,
            0.0,
        )
        assert p.conserve
        assert p.substeps == 8

    def test_small_and_big_dynamics(self) -> None:
        small = compose(["+experiment=m14/s0"]).dynamics
        big = compose(["+experiment=m14/s0", "dynamics=big"]).dynamics
        assert small is not None and big is not None
        assert (small.d_s, small.nodes) == (256, 1)
        assert (big.d_s, big.nodes) == (512, 3)

    def test_time_and_optimizer(self) -> None:
        conf = compose(["+experiment=m14/s0"])
        assert conf.time.frame == pytest.approx(1 / 30)
        assert conf.time.learn_interval == 25e-3
        assert (conf.optim.kind, conf.optim.lr, conf.optim.warmup) == (
            "sgd",
            0.03,
            300.0,
        )
        assert conf.learner.kind == "bp"
        assert conf.phase == "explore"
