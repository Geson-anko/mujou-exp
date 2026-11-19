import sys
import sysconfig

import pytest
from hydra import initialize_config_module
from hydra.errors import ConfigCompositionException

from exp.vision.compose import compose, from_dict, to_dict

BASE = "+experiment=v0/base"
TINY = "+experiment=v0/tiny"


class TestOverrides:
    def test_the_base_preset_is_the_spec_of_v0(self) -> None:
        c = compose([BASE])
        assert (c.program, c.task, c.name, c.phase) == (
            "vae",
            "online",
            "v0/base",
            "explore",
        )
        t = c.data.tours
        assert (t.data_version, t.held_out, t.size, t.fps) == (
            "ad6e56b5995e",
            "Zurich",
            64,
            10.0,
        )
        assert (t.frames, t.stride, t.advance, c.data.streams) == (8, 1, 1, 1)
        assert (c.model.arch, c.model.size, c.model.latent) == ("vit", "S", 256)
        assert (c.model.enc_nodes, c.model.dec_nodes) == (4, 2)
        assert (c.model.width, c.model.head_count, c.model.patch) == (256, 4, 16)
        assert (c.time.frame, c.time.h, c.time.lam, c.time.r) == (0.1, 4e-3, 2e-3, 4e-3)
        assert (c.optim.kind, c.optim.tau) == ("sgd", 1e5)  # stays finite
        assert c.time.learn_interval == 0.09
        assert c.measure.ckpt_times == [0.0, 60.0, 300.0, 600.0, 1200.0, 1800.0]
        assert (c.measure.duration, c.measure.blur_sigma) == (1800.0, 1.0)
        assert c.measure.posthoc_seeds == 4

    def test_the_tiny_preset_composes(self) -> None:
        c = compose([TINY])
        assert c.name == "v0/tiny" and c.data.tours.size == 32

    @pytest.mark.parametrize("arch", ["vit", "resnet"])
    @pytest.mark.parametrize("size", ["s", "m", "l"])
    def test_model_group(self, arch: str, size: str) -> None:
        c = compose([BASE, f"model={arch}_{size}"])
        assert (c.model.arch, c.model.size) == (arch, size.upper())

    @pytest.mark.parametrize(
        "kind",
        [
            "sgd",
            "sgd_momentum",
            "rmsprop",
            "adamw",
            "nesterov",
            "rmsprop_centered",
            "adadelta",
            "rprop",
            "muon",
        ],
    )
    def test_optimizer_group(self, kind: str) -> None:
        assert compose([BASE, f"optim={kind}"]).optim.kind == kind

    @pytest.mark.parametrize(
        ("kind", "key"),
        [("rprop", "tau_m"), ("nesterov", "tau_v"), ("rprop", "eps")],
    )
    def test_a_setting_the_kind_does_not_use_fails(self, kind: str, key: str) -> None:
        with pytest.raises(ValueError, match=key):
            compose([BASE, f"optim={kind}", f"optim.{key}=3"])

    def test_dec_scale_is_of_every_kind_and_positive(self) -> None:
        assert (
            compose([BASE, "optim=rprop", "optim.dec_scale=10"]).optim.dec_scale == 10
        )
        with pytest.raises(ValueError, match="dec_scale"):
            compose([BASE, "optim.dec_scale=0"])

    def test_ogd_is_off_by_default_and_of_sgd_rmsprop_adamw(self) -> None:
        assert compose([BASE]).optim.ogd_n == 0
        for kind in ("sgd", "rmsprop", "adamw"):
            c = compose([BASE, f"optim={kind}", "optim.ogd_n=16", "optim.ogd_every=1"])
            assert (c.optim.ogd_n, c.optim.ogd_every) == (16, 1.0)
        with pytest.raises(ValueError, match="ogd_n"):
            compose([BASE, "optim=muon", "optim.ogd_n=16"])
        with pytest.raises(ValueError, match="ogd_n"):
            compose([BASE, "optim=sgd_momentum", "optim.ogd_n=16"])

    def test_ogd_every_without_ogd_or_not_positive_fails(self) -> None:
        with pytest.raises(ValueError, match="ogd_every"):
            compose([BASE, "optim.ogd_every=10"])
        with pytest.raises(ValueError, match="ogd_every"):
            compose([BASE, "optim.ogd_n=4", "optim.ogd_every=0"])
        with pytest.raises(ValueError, match="ogd_n"):
            compose([BASE, "optim.ogd_n=-1"])

    def test_the_heads_are_of_64_dimensions_by_default(self) -> None:
        assert compose([BASE]).model.head_count == 4
        assert compose([BASE, "model.width=512"]).model.head_count == 8
        two = compose([BASE, "model.width=512", "model.heads=2"])
        assert two.model.head_count == 2
        with pytest.raises(ValueError, match="heads"):
            compose([BASE, "model.width=96"])

    def test_scale_decoder_scales_the_decoders_channels(self) -> None:
        assert compose([BASE, "model.width=512"]).model.decoder_widths == [
            256,
            128,
            64,
            32,
            32,
        ]
        c = compose([BASE, "model.width=128", "model.scale_decoder=true"])
        assert c.model.decoder_widths == [128, 64, 32, 16, 16]
        with pytest.raises(ValueError, match="decoder"):
            compose(
                [
                    BASE,
                    "model.width=64",
                    "model.scale_decoder=true",
                    "model.decoder_channels=[256,128,64,32,30]",
                ]
            )

    def test_mup_is_of_the_vit_of_64_dimensional_heads(self) -> None:
        assert (
            compose([BASE, "model.param=mup", "model.width=512"]).model.param == "mup"
        )
        for bad in (
            ["model=resnet_s", "model.param=mup"],
            ["model.param=mup", "model.width=512", "model.heads=4"],
            ["model.param=mup", "optim.ogd_n=4"],
            ["model.param=mu"],
        ):
            with pytest.raises(ValueError, match="param"):
                compose([BASE, *bad])

    def test_grad_stats_is_off_by_default(self) -> None:
        assert not compose([BASE]).measure.grad_stats
        assert compose([BASE, "measure.grad_stats=true"]).measure.grad_stats

    def test_unknown_optimizer_fails_in_post_init(self) -> None:
        with pytest.raises(ValueError, match="foo"):
            compose([BASE, "optim.kind=foo"])

    def test_a_time_constant_of_another_kind_fails(self) -> None:
        with pytest.raises(ValueError, match="tau_v"):
            compose([BASE, "optim.tau_v=10"])

    def test_seed_and_device(self) -> None:
        c = compose([BASE, "seed=7", "device=cuda:1"])
        assert (c.seed, c.device) == (7, "cuda:1")

    def test_unknown_device_fails(self) -> None:
        with pytest.raises(ValueError, match="gpu"):
            compose([BASE, "device=gpu"])

    def test_unknown_key_fails(self) -> None:
        with pytest.raises(ConfigCompositionException):
            compose([BASE, "optim.foo=1"])

    def test_phase_confirm_turns_on_the_test_stacks(self) -> None:
        c = compose([BASE, "phase=confirm"])
        assert c.phase == "confirm" and c.ctl.eval_test

    @pytest.mark.parametrize("learn_interval", [0.032, 0.056, 0.102])
    def test_a_learn_interval_outside_the_window_fails(
        self, learn_interval: float
    ) -> None:
        """(0.058, 0.1] at 0.1 s frames, 2 ms steps and 8 hops (exp.vision.pc;
        tests/vision/test_pc.py counts the learns on both sides)."""
        with pytest.raises(ValueError, match="learn_interval"):
            compose([BASE, f"time.learn_interval={learn_interval}"])

    def test_the_window_moves_with_the_frame(self) -> None:
        with pytest.raises(ValueError, match="learn_interval"):
            compose([BASE, "time.frame=0.2"])
        c = compose([BASE, "time.frame=0.2", "time.learn_interval=0.2"])
        assert c.time.learn_interval == 0.2

    def test_the_node_counts_are_fixed(self) -> None:
        with pytest.raises(ValueError, match="fixed"):
            compose([BASE, "model.enc_nodes=3"])

    def test_the_size_must_fit_the_patch(self) -> None:
        with pytest.raises(ValueError, match="patch"):
            compose([BASE, "data.tours.size=60"])

    def test_a_checkpoint_off_the_frames_fails(self) -> None:
        with pytest.raises(ValueError, match="whole number of frames"):
            compose([BASE, "measure.ckpt_times=[0,60.05]"])

    def test_a_checkpoint_beyond_the_run_fails(self) -> None:
        with pytest.raises(ValueError, match="beyond"):
            compose([BASE, "measure.duration=100"])


class TestRoundTrip:
    def test_to_dict_from_dict(self) -> None:
        c = compose([BASE, "model=resnet_m", "optim=adamw", "seed=3"])
        assert from_dict(to_dict(c)) == c

    def test_from_dict_checks_the_schema(self) -> None:
        d = to_dict(compose([BASE]))
        d["task"] = "nope"
        with pytest.raises(ValueError, match="nope"):
            from_dict(d)

    def test_provenance_finds_the_data_version(self) -> None:
        d = to_dict(compose([BASE]))
        assert d["data"]["tours"]["data_version"] == "ad6e56b5995e"


class TestRepeated:
    def test_same_result_twice(self) -> None:
        assert compose([BASE, "seed=3"]) == compose([BASE, "seed=3"])

    def test_after_hydra_was_left_initialized(self) -> None:
        initialize_config_module(config_module="exp.vision.conf", version_base="1.3")
        assert compose([BASE]).name == "v0/base"


@pytest.mark.skipif(
    not sysconfig.get_config_var("Py_GIL_DISABLED"), reason="not a free-threaded build"
)
def test_gil_stays_disabled_after_compose() -> None:
    compose([BASE, "optim=adamw"])
    assert not sys._is_gil_enabled()
