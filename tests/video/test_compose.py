import dataclasses
import sys
import sysconfig

import pytest
from hydra import initialize_config_module
from hydra.errors import ConfigCompositionException

from exp.video.compose import compose, from_dict, to_dict
from exp.video.schema.data import ToursConf

S0 = "+experiment=m14/s0"


class TestOverrides:
    def test_unknown_optimizer_fails_in_post_init(self) -> None:
        with pytest.raises(ValueError, match="foo"):
            compose([S0, "optim.kind=foo"])

    def test_optimizer_group(self) -> None:
        conf = compose([S0, "optim=adamw"])
        assert (conf.optim.kind, conf.optim.lr) == ("adamw", 1e-4)

    def test_optimizer_kind_keeps_the_rest(self) -> None:
        conf = compose([S0, "optim.kind=adamw"])
        assert (conf.optim.kind, conf.optim.lr) == ("adamw", 0.03)

    def test_seed_and_device(self) -> None:
        conf = compose([S0, "seed=7", "device=cuda:1"])
        assert (conf.seed, conf.device) == (7, "cuda:1")

    def test_unknown_device_fails(self) -> None:
        with pytest.raises(ValueError, match="gpu"):
            compose([S0, "device=gpu"])

    def test_unknown_key_fails(self) -> None:
        with pytest.raises(ConfigCompositionException):
            compose([S0, "optim.foo=1"])

    def test_learner_and_phase(self) -> None:
        conf = compose([S0, "learner=pc", "phase=confirm"])
        assert conf.learner.kind == "pc"
        assert conf.phase == "confirm"
        assert conf.ctl.eval_test
        assert (conf.ctl.ckpt_every, conf.ctl.viz_every) == (300.0, 300.0)


class TestSubtrees:
    def test_encoder_optimizer_from_the_optim_group(self) -> None:
        conf = compose(
            [
                S0,
                "program=joint",
                "vision=vit_s",
                "+optim@optim_vision=adamw",
                "optim_vision.lr=2e-3",
            ]
        )
        assert conf.vision is not None and conf.vision.dim == 384
        assert conf.optim_vision is not None
        assert (conf.optim_vision.kind, conf.optim_vision.lr) == ("adamw", 2e-3)
        assert conf.optim.kind == "sgd"

    def test_dynamics_program_needs_dynamics(self) -> None:
        with pytest.raises(ValueError, match="dynamics"):
            compose([S0, "dynamics=none"])

    def test_vision_program_needs_vision(self) -> None:
        with pytest.raises(ValueError, match="vision"):
            compose([S0, "program=vision"])

    def test_no_data_source_fails(self) -> None:
        d = to_dict(compose([S0]))
        d["data"]["balls"] = None
        with pytest.raises(ValueError, match="exactly one"):
            from_dict(d)

    def test_two_data_sources_fail(self) -> None:
        d = to_dict(compose([S0]))
        d["data"]["tours"] = dataclasses.asdict(
            ToursConf(data_version="v1", held_out="a")
        )
        with pytest.raises(ValueError, match="exactly one"):
            from_dict(d)

    @pytest.mark.parametrize(
        ("override", "subtree"),
        [
            ("offline.steps=100", "offline"),
            ("data.tours.size=32", "data.tours"),
            ("optim_vision.lr=1e-3", "optim_vision"),
            ("vision.dim=64", "vision"),
        ],
    )
    def test_override_into_a_none_subtree_fails(
        self, override: str, subtree: str
    ) -> None:
        with pytest.raises(ValueError, match=f"{subtree} is None"):
            compose([S0, override])

    def test_override_after_its_group_fills_the_subtree(self) -> None:
        conf = compose([S0, "vision.dim=64", "vision=vit_t"])
        assert conf.vision is not None and conf.vision.dim == 64

    def test_vision_needs_a_node(self) -> None:
        with pytest.raises(ValueError, match="nodes"):
            compose([S0, "vision=vit_t", "vision.nodes=0"])


class TestRepeated:
    def test_same_result_twice(self) -> None:
        assert compose([S0, "seed=3"]) == compose([S0, "seed=3"])

    def test_after_hydra_was_left_initialized(self) -> None:
        initialize_config_module(config_module="exp.video.conf", version_base="1.3")
        assert compose([S0]).name == "m14/s0"


@pytest.mark.skipif(
    not sysconfig.get_config_var("Py_GIL_DISABLED"), reason="not a free-threaded build"
)
def test_gil_stays_disabled_after_compose() -> None:
    compose([S0, "optim=adamw"])
    assert not sys._is_gil_enabled()
