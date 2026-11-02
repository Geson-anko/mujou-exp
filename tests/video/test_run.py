import json
import subprocess

import pytest
import torch

from exp.lib.record import REPO_ROOT
from exp.video import run
from exp.video.balls import Video
from exp.video.config import Config, DataConfig, OptimConfig
from exp.video.stage0 import RUN_MODULE


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


class TestSource:
    def test_the_default_physics_gives_the_frames_of_the_video(self):
        """M13 made the physics a setting; left at its defaults, the stream is
        the one of the runs before it."""
        video, _ = run.source(Config(seed=3))
        want = Video(seed=3)
        for _ in range(300):  # 10 s, across episodes
            a, b = video.next(), want.next()
            assert torch.equal(a.image, b.image) and a.reset == b.reset

    def test_the_settings_reach_the_stream_and_the_clips(self):
        data = DataConfig(
            gamma_amp=0.0, e0=1.0, e_amp=0.0, drag=0.0, redraw=False, streams=2
        )
        video, clips = run.source(Config(data=data))
        resets = torch.stack([torch.as_tensor(video.next().reset) for _ in range(300)])
        assert resets[0].all() and not resets[1:].any()
        torch.testing.assert_close(
            clips.params, torch.tensor([1.0, 1.0]).expand_as(clips.params)
        )


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
                RUN_MODULE,
                "--help",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--config" in result.stdout
