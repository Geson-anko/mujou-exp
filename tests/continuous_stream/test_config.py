import pytest

from experiments.continuous_stream.config import Config, DataConfig, OptimConfig


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

    def test_an_int_in_json_reads_as_float(self):
        config = Config.from_dict({"optim": {"tau": 2}})
        assert isinstance(config.optim.tau, float)
        assert config.optim.tau == 2.0

    def test_an_unknown_key_is_an_error(self):
        with pytest.raises(TypeError, match="lr"):
            Config.from_dict({"optim": {"lr": 0.1}})
