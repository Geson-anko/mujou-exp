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
