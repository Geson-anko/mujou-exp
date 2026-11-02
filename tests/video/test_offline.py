import json

import pytest
import torch

from exp.video import offline
from exp.video.config import Config, DataConfig, ModelConfig


class TestWindowStarts:
    def test_windows_never_cross_a_reset(self):
        """A window of w frames from (t, b) holds frames t..t+w-1 of stream b;
        it is taken iff none of frames t+1..t+w-1 starts an episode."""
        resets = torch.zeros(10, 2, dtype=torch.bool)
        resets[0] = True
        resets[4, 0] = True  # stream 0 starts anew at frame 4
        got = {tuple(x) for x in offline.window_starts(resets, 3).tolist()}
        want = {(t, 0) for t in (0, 1, 4, 5, 6, 7)} | {(t, 1) for t in range(8)}
        assert got == want

    def test_a_stream_without_resets_gives_every_window(self):
        resets = torch.zeros(6, 1, dtype=torch.bool)
        resets[0] = True
        got = offline.window_starts(resets, 6)
        assert got.tolist() == [[0, 0]]


@pytest.mark.slow
class TestMain:
    def test_it_learns_and_saves_the_model_of_the_config(self, tmp_path, monkeypatch):
        config = Config(
            name="tiny",
            model=ModelConfig(d_z=8, d_s=12, width=8, dyn_nodes=2, z_norm=True),
            data=DataConfig(
                gamma_amp=0.0, e0=1.0, e_amp=0.0, drag=0.0, redraw=False, streams=4
            ),
        )
        path = tmp_path / "c.json"
        path.write_text(config.to_json())
        save = tmp_path / "out"
        argv = ["offline", "--config", str(path), "--learn-encoder", "--local"]
        argv += ["--batch", "4", "--frames", "256", "--steps", "2", "--every", "2"]
        argv += ["--device", "cpu", "--save", str(save)]
        monkeypatch.setattr("sys.argv", argv)
        offline.main()
        head = json.loads((save / "config.json").read_text())
        assert Config.from_dict(head["config"]) == config
        snap = torch.load(save / "ckpt" / "00002.pt", weights_only=False)
        assert snap.step.d_state == 24
