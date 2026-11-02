import pytest

from exp.video.balls import Physics
from exp.video.config import Config, DataConfig, OptimConfig


class TestConfig:
    def test_json_round_trip_of_the_defaults(self):
        config = Config()
        assert Config.from_json(config.to_json()) == config

    def test_json_round_trip_of_changed_values(self):
        config = Config(
            name="x",
            seed=3,
            optim=OptimConfig(kind="whitened", tau_l=1.0, tau_r=None, graft=True),
            data=DataConfig(kind="linear", tau_x=3.0, iid=True),
        )
        assert Config.from_json(config.to_json()) == config


class TestStreams:
    def test_the_batch_is_empty_for_one_stream_and_b_for_more(self):
        assert DataConfig().batch == ()
        assert DataConfig(streams=4).batch == (4,)

    def test_json_round_trip(self):
        config = Config(data=DataConfig(streams=8))
        assert Config.from_json(config.to_json()) == config

    @pytest.mark.parametrize(
        "changes",
        [
            {"streams": 0},
            {"streams": 2, "kind": "egodex"},
            {"streams": 2, "kind": "linear"},
            {"streams": 2, "order": "shuffled"},
        ],
    )
    def test_streams_are_for_the_synthetic_video_in_stream_order(self, changes):
        with pytest.raises(ValueError, match="streams"):
            DataConfig(**changes)


class TestPhysics:
    def test_the_default_is_the_physics_of_the_video(self):
        assert DataConfig().physics() == Physics()

    def test_each_field_reaches_the_physics(self):
        d = DataConfig(
            gamma0=2.0,
            gamma_amp=0.0,
            e0=1.0,
            e_amp=0.0,
            drag=0.0,
            conserve=True,
            physics_period=60.0,
        )
        assert d.physics() == Physics(
            gamma0=2.0,
            gamma_amp=0.0,
            e0=1.0,
            e_amp=0.0,
            drag=0.0,
            conserve=True,
            period=60.0,
        )

    def test_json_round_trip(self):
        config = Config(data=DataConfig(e0=1.0, drag=0.0, redraw=False, radii=(0.1,)))
        assert Config.from_json(config.to_json()) == config


class TestCheckpoints:
    @pytest.mark.parametrize("seconds", [-1.0, 0.5])
    def test_the_interval_is_whole_seconds(self, seconds):
        with pytest.raises(ValueError, match="whole seconds"):
            Config(checkpoint_interval=seconds)

    def test_a_real_time_run_takes_none(self):
        with pytest.raises(ValueError, match="real-time"):
            Config(checkpoint_interval=60.0, realtime=True)


class TestDynamicsOptim:
    def test_json_round_trip(self):
        config = Config(dynamics_optim=OptimConfig(kind="adamw", tau=33.0))
        assert Config.from_json(config.to_json()) == config
