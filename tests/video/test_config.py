import pytest

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
