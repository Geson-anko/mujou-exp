import json

import pytest

from experiments.continuous_stream import run
from experiments.continuous_stream.config import Config, DataConfig, OptimConfig


def _run(tmp_path, config: Config):
    path = tmp_path / f"{config.name}.json"
    path.write_text(config.to_json())
    out = run.main(["--config", str(path), "--out", str(tmp_path)])
    lines = (out / "metrics.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


class TestRunLinear:
    def test_the_same_seed_writes_the_same_metrics(self, tmp_path):
        data = DataConfig(kind="linear", seconds=2.0)
        a = _run(tmp_path, Config(name="a", eval_interval=1.0, data=data))
        b = _run(tmp_path, Config(name="b", eval_interval=1.0, data=data))
        assert len(a) == 60
        assert a == b

    @pytest.mark.slow
    def test_it_converges_to_w_star_on_the_iid_stream(self, tmp_path):
        config = Config(
            name="iid",
            eval_interval=10.0,
            optim=OptimConfig(tau=10.0),
            data=DataConfig(kind="linear", seconds=60.0, iid=True),
        )
        records = _run(tmp_path, config)
        assert records[0]["dist"] > 0.5
        tail = [r["dist"] for r in records[-300:]]
        assert sum(tail) / len(tail) < 0.1
