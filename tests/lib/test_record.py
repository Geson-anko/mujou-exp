import json
import subprocess
from dataclasses import asdict, dataclass
from typing import Any

from exp.lib.record import REPO_ROOT, Recorder


@dataclass(frozen=True)
class _Config:
    name: str = "t"
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout.strip()


class TestRecorder:
    def test_config_json_holds_the_settings_and_the_commits(self, tmp_path):
        config = _Config(name="t", seed=7)
        with Recorder(config, tmp_path) as recorder:
            head = json.loads((recorder.dir / "config.json").read_text())
        assert head["config"] == config.to_dict()
        assert head["mujou"] == _git(
            "-C", str(REPO_ROOT / "mujou"), "rev-parse", "HEAD"
        )
        assert head["repo"] == _git("-C", str(REPO_ROOT), "rev-parse", "HEAD")
        assert isinstance(head["dirty"], bool)

    def test_metrics_are_one_json_object_per_line(self, tmp_path):
        with Recorder(_Config(), tmp_path) as recorder:
            recorder.log(0.5, loss=1.0)
            recorder.log(1.0, loss=0.25, phase="b")
        lines = (recorder.dir / "metrics.jsonl").read_text().splitlines()
        assert [json.loads(line) for line in lines] == [
            {"time": 0.5, "loss": 1.0},
            {"time": 1.0, "loss": 0.25, "phase": "b"},
        ]
