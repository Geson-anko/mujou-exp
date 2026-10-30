import json
import subprocess

from experiments.continuous_stream import run
from experiments.continuous_stream.config import Config
from experiments.continuous_stream.record import REPO_ROOT, Recorder


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout.strip()


class TestRecorder:
    def test_config_json_holds_the_settings_and_the_commits(self, tmp_path):
        config = Config(name="t", seed=7)
        with Recorder(config, tmp_path) as recorder:
            head = json.loads((recorder.dir / "config.json").read_text())
        assert Config.from_dict(head["config"]) == config
        assert head["mujou"] == _git(
            "-C", str(REPO_ROOT / "mujou"), "rev-parse", "HEAD"
        )
        assert head["repo"] == _git("-C", str(REPO_ROOT), "rev-parse", "HEAD")
        assert isinstance(head["dirty"], bool)

    def test_metrics_are_one_json_object_per_line(self, tmp_path):
        with Recorder(Config(), tmp_path) as recorder:
            recorder.log(0.5, loss=1.0)
            recorder.log(1.0, loss=0.25, phase="b")
        lines = (recorder.dir / "metrics.jsonl").read_text().splitlines()
        assert [json.loads(line) for line in lines] == [
            {"time": 0.5, "loss": 1.0},
            {"time": 1.0, "loss": 0.25, "phase": "b"},
        ]


class TestRun:
    def test_an_empty_run_writes_config_json(self, tmp_path):
        path = tmp_path / "c.json"
        path.write_text(Config(name="empty").to_json())
        out = run.main(
            ["--config", str(path), "--out", str(tmp_path / "runs"), "--dry"]
        )
        assert (out / "config.json").is_file()
        assert out.name.endswith("-empty")

    def test_help(self):
        result = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "-m",
                "experiments.continuous_stream.run",
                "--help",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--config" in result.stdout
