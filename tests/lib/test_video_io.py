import hashlib
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
import pytest
import torch

from exp.lib.video_io import (
    FRAMES,
    INDEX,
    U8Reader,
    U8Writer,
    ffmpeg_frames,
    ffmpeg_version,
)


def make_video(path: Path, graph: str) -> Path:
    """Encode the lavfi ``graph`` losslessly (x264, qp 0) to ``path``."""
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y", "-f", "lavfi"]
    cmd += ["-i", graph, "-pix_fmt", "yuv420p", "-c:v", "libx264", "-qp", "0"]
    subprocess.run([*cmd, str(path)], check=True)
    return path


def rgb_seconds(path: Path) -> Path:
    """Red for 1 s, lime for 1 s, blue for 1 s, at 30 fps, 320x180."""
    c = "s=320x180:r=30:d=1"
    graph = f"color=c=red:{c}[a];color=c=lime:{c}[b];color=c=blue:{c}[c];"
    return make_video(path, graph + "[a][b][c]concat=n=3[out0]")


def frames(path: Path, **kw) -> torch.Tensor:
    kw = {"fps": 10.0, "width": 64, "height": 64} | kw
    return torch.stack(list(ffmpeg_frames(path, **kw))).float() / 255


# the dominant channel of each frame: 0 red, 1 green, 2 blue
def channel(x: torch.Tensor) -> list[int]:
    return x.mean((2, 3)).argmax(1).tolist()


@pytest.fixture(scope="module")
def rgb(tmp_path_factory) -> Path:
    return rgb_seconds(tmp_path_factory.mktemp("video") / "rgb.mp4")


class TestFfmpegFrames:
    def test_shape_and_dtype(self, rgb):
        x = next(ffmpeg_frames(rgb, fps=10.0, width=64, height=48))
        assert x.shape == (3, 48, 64)
        assert x.dtype == torch.uint8

    def test_frame_count_is_seconds_times_fps(self, rgb):
        assert len(frames(rgb, fps=10.0)) == 30
        assert len(frames(rgb, fps=5.0)) == 15

    def test_frames_follow_the_clock(self, rgb):
        # the output frame at t = k / 10 shows the second floor(t): k // 10
        assert channel(frames(rgb)) == [0] * 10 + [1] * 10 + [2] * 10

    def test_colors_are_rgb_in_chw_order(self, rgb):
        x = frames(rgb)[:10]
        assert x[:, 0].min() > 0.95
        assert x[:, 1:].max() < 0.05

    def test_start_and_duration_cut_the_interval(self, rgb):
        x = frames(rgb, start=1.0, duration=1.0)
        assert channel(x) == [1] * 10

    def test_square_crop_is_centered(self, tmp_path):
        # 320x180: the largest centered square is x in [70, 250); lime there,
        # red outside, so a centered crop is all lime and any other is not
        graph = "color=c=red:s=320x180:r=30:d=0.2,"
        graph += "drawbox=x=70:y=0:w=180:h=180:color=lime:t=fill"
        x = frames(make_video(tmp_path / "a.mp4", graph))
        assert x[:, 1].min() > 0.9
        assert x[:, 0].max() < 0.1

    def test_wide_crop_cuts_the_top_and_bottom(self, tmp_path):
        # aspect 2:1 of 320x180 is 320x160 at y in [10, 170): red bands on
        # rows [0, 10) and [170, 180) are cut away
        graph = "color=c=lime:s=320x180:r=30:d=0.2,"
        graph += "drawbox=x=0:y=0:w=320:h=10:color=red:t=fill,"
        graph += "drawbox=x=0:y=170:w=320:h=10:color=red:t=fill"
        x = frames(make_video(tmp_path / "a.mp4", graph), width=64, height=32)
        assert x.shape[1:] == (3, 32, 64)
        assert x[:, 0].max() < 0.1

    def test_closing_early_stops_ffmpeg(self, rgb):
        it = ffmpeg_frames(rgb, fps=10.0, width=64, height=64)
        next(it)
        it.close()  # must not hang or raise

    def test_a_missing_file_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="ffmpeg failed"):
            next(ffmpeg_frames(tmp_path / "none.mp4", fps=10, width=8, height=8))

    def test_version_names_ffmpeg(self):
        assert ffmpeg_version().startswith("ffmpeg version")


class TestU8:
    def write(self, path: Path, n: int = 5) -> torch.Tensor:
        x = torch.randint(0, 256, (n, 3, 4, 6), dtype=torch.uint8)
        with U8Writer(path, (3, 4, 6), {"video": "a"}) as w:
            for frame in x:
                w.append(frame)
        return x

    def test_round_trip(self, tmp_path):
        x = self.write(tmp_path / "d")
        r = U8Reader(tmp_path / "d")
        assert len(r) == 5 and r.shape == (3, 4, 6)
        assert torch.equal(r[2], x[2])
        assert torch.equal(r[1:4], x[1:4])
        assert r.index["meta"] == {"video": "a"}

    def test_index_records_the_sha256_of_the_bytes(self, tmp_path):
        self.write(tmp_path / "d")
        index = json.loads((tmp_path / "d" / INDEX).read_text())
        data = (tmp_path / "d" / FRAMES).read_bytes()
        assert index["sha256"] == hashlib.sha256(data).hexdigest()
        assert index["frames"] == 5 and index["shape"] == [3, 4, 6]

    def test_same_frames_give_the_same_index(self, tmp_path):
        x = self.write(tmp_path / "a")
        with U8Writer(tmp_path / "b", (3, 4, 6), {"video": "a"}) as w:
            for frame in x:
                w.append(frame)
        assert (tmp_path / "a" / INDEX).read_bytes() == (
            tmp_path / "b" / INDEX
        ).read_bytes()

    def test_verify_catches_one_changed_byte(self, tmp_path):
        self.write(tmp_path / "d")
        U8Reader(tmp_path / "d").verify()
        data = bytearray((tmp_path / "d" / FRAMES).read_bytes())
        data[7] ^= 1
        (tmp_path / "d" / FRAMES).write_bytes(bytes(data))
        with pytest.raises(ValueError, match="sha256"):
            U8Reader(tmp_path / "d").verify()

    def test_opening_a_truncated_file_raises(self, tmp_path):
        self.write(tmp_path / "d")
        data = (tmp_path / "d" / FRAMES).read_bytes()
        (tmp_path / "d" / FRAMES).write_bytes(data[:-1])
        with pytest.raises(ValueError, match="bytes"):
            U8Reader(tmp_path / "d")

    def test_wrong_shape_dtype_or_device_is_refused(self, tmp_path):
        with U8Writer(tmp_path / "d", (3, 4, 6)) as w:
            with pytest.raises(ValueError):
                w.append(torch.zeros(3, 4, 5, dtype=torch.uint8))
            with pytest.raises(ValueError):
                w.append(torch.zeros(3, 4, 6))
            # a tensor off the CPU has no bytes behind its pointer here
            with pytest.raises(ValueError, match="CPU"):
                w.append(torch.empty(3, 4, 6, dtype=torch.uint8, device="meta"))
        assert w.frames == 0

    def test_an_error_in_the_block_writes_no_index(self, tmp_path):
        with pytest.raises(KeyError), U8Writer(tmp_path / "d", (3, 4, 6)):
            raise KeyError
        assert not (tmp_path / "d" / INDEX).exists()

    def test_no_frames_reads_as_empty(self, tmp_path):
        with U8Writer(tmp_path / "d", (3, 4, 6)):
            pass
        assert U8Reader(tmp_path / "d")[:].shape == (0, 3, 4, 6)
