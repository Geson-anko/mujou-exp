import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
import pytest
import torch

from exp.video import egodex
from exp.video.egodex import (
    EgoDexStream,
    calibration_images,
    decode,
    egodex_clips,
    plan_clips,
    plan_stream,
    prepare,
)


def _make(path: Path, source: str, frames: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y", "-f", "lavfi"]
    cmd += ["-i", f"{source}", "-frames:v", str(frames)]
    cmd += ["-pix_fmt", "yuv420p", "-c:v", "libx264", str(path)]
    subprocess.run(cmd, check=True)
    return path


def _color(path: Path, c: str, frames: int) -> Path:
    return _make(path, f"color=c={c}:s=320x180:r=30", frames)


class TestDecode:
    def test_shape_dtype_and_frame_count(self, tmp_path):
        x = decode(_color(tmp_path / "a.mp4", "red", 10))
        assert x.shape == (10, 3, 64, 64)
        assert x.dtype == torch.uint8

    def test_red_video_is_red_in_chw_order(self, tmp_path):
        x = decode(_color(tmp_path / "a.mp4", "red", 3)).float() / 255
        assert x[:, 0].min() > 0.85
        assert x[:, 1:].max() < 0.15

    def test_first_frames_keeps_only_the_head(self, tmp_path):
        p = _color(tmp_path / "a.mp4", "red", 10)
        assert len(decode(p, frames=4)) == 4

    def test_center_is_kept(self, tmp_path):
        # green in x in [40, 280) of 320, red outside; after shrinking to height
        # 64 (width 114) the green spans columns 14..99, so the central 64
        # columns (25..88) are all green and a left-aligned crop would not be
        src = (
            "color=c=red:s=320x180:r=30,drawbox=x=40:y=0:w=240:h=180:color=lime:t=fill"
        )
        x = decode(_make(tmp_path / "a.mp4", src, 2)).float() / 255
        assert x[:, 1].min() > 0.8
        assert x[:, 0].max() < 0.2

    def test_wide_edge_is_cut(self, tmp_path):
        # red only at the far left and right 20 pixels of 320: cut away
        src = "color=c=lime:s=320x180:r=30,drawbox=x=0:y=0:w=20:h=180:color=red:t=fill"
        src += ",drawbox=x=300:y=0:w=20:h=180:color=red:t=fill"
        x = decode(_make(tmp_path / "a.mp4", src, 1)).float() / 255
        assert x[:, 0].max() < 0.2


def _fake_meta() -> list[dict]:
    def m(split, task, i, n):
        return {"relative_path": f"{split}/{task}/{i}.mp4", "frame_count": n}

    meta = []
    for task in ("b", "a", "c"):
        meta += [m("part1", task, i, 70) for i in range(8)]
        meta += [m("part1", task, 100 + i, 10) for i in range(2)]  # too short
    meta += [m("part1", "few", i, 70) for i in range(3)]
    for task, n in (("t2", 200), ("t1", 100), ("t0", 130), ("t3", 128)):
        meta += [m("test", task, 0, n)]
    return meta


class TestPlan:
    def plan(self, **kw):
        return plan_stream(
            _fake_meta(), block=4.0, passes=2, min_episodes=5, seed=0, **kw
        )

    def test_tasks_in_name_order_and_few_episode_tasks_dropped(self):
        plan = self.plan()
        firsts = [e.task for e in plan if e.round == 0]
        assert sorted(set(firsts)) == ["a", "b", "c"]
        order = [t for i, t in enumerate(firsts) if i == 0 or t != firsts[i - 1]]
        assert order == ["a", "b", "c"]

    def test_short_episodes_are_dropped(self):
        assert all(e.frames >= 64 for e in self.plan())
        assert not any(int(e.path.split("/")[-1][:-4]) >= 100 for e in self.plan())

    def test_block_exceeds_the_length_by_less_than_one_episode(self):
        plan = self.plan()
        for b in {e.block for e in plan}:
            n = sum(e.frames for e in plan if e.block == b)
            assert 120 < n <= 120 + 70  # 4 s = 120 frames, episodes of 70

    def test_second_pass_uses_unused_episodes_and_returns_to_the_task(self):
        plan = self.plan()
        for task in "abc":
            p0 = {e.path for e in plan if e.task == task and e.round == 0}
            p1 = {e.path for e in plan if e.task == task and e.round == 1}
            assert p0 and p1 and not p0 & p1
        tasks = [e.task for e in plan]
        assert tasks.count("a") > 0 and [e.round for e in plan][-1] == 1
        assert [e.block for e in plan] == sorted(e.block for e in plan)
        assert max(e.block for e in plan) == 5  # 3 tasks x 2 passes

    def test_seed_fixes_the_plan(self):
        assert self.plan() == self.plan()
        other = plan_stream(_fake_meta(), block=4.0, min_episodes=5, seed=1)
        assert other != self.plan()

    def test_runs_out_of_episodes(self):
        with pytest.raises(ValueError):
            plan_stream(_fake_meta(), block=100.0, min_episodes=5)

    def test_clips_from_long_tasks_in_name_order(self):
        plan = plan_clips(_fake_meta(), tasks=2, length=128)
        assert [e.task for e in plan] == ["t0", "t2"]
        assert [e.task for e in plan_clips(_fake_meta(), tasks=9)] == [
            "t0",
            "t2",
            "t3",
        ]


@pytest.fixture(scope="module")
def dataset(tmp_path_factory) -> tuple[Path, Path]:
    """A fake EgoDex whose episodes are told apart by their color."""
    tmp_path = tmp_path_factory.mktemp("egodex")
    root = tmp_path / "egodex"
    colors = ["red", "green", "blue", "yellow", "white", "black"]
    meta = []
    k = 0
    for task in ("a", "b"):
        for i in range(6):
            n = 70 + i
            _color(root / "part1" / task / f"{i}.mp4", colors[(k := k + 1) % 6], n)
            meta.append({"relative_path": f"part1/{task}/{i}.mp4", "frame_count": n})
    for task in ("x", "y"):
        _color(root / "test" / task / "0.mp4", "red", 40)
        meta.append({"relative_path": f"test/{task}/0.mp4", "frame_count": 40})
    (root / "video_metadata.json").write_text(json.dumps(meta))
    return root, tmp_path / "out"


@pytest.fixture(scope="module")
def prepared(dataset) -> tuple[Path, Path, dict]:
    root, out = dataset
    index = prepare(
        out,
        root=root,
        jobs=2,
        block=3.0,
        passes=2,
        min_episodes=3,
        clips=2,
        length=32,
    )
    return root, out, index


class TestPrepareAndLoad:
    def test_round_trip(self, prepared):
        root, out, index = prepared
        s = EgoDexStream(out)
        assert s.frames == index["frames"]
        assert not list(out.glob("*.tmp"))
        n = sum(e["frames"] for e in index["episodes"])
        assert n == s.frames
        for e in index["episodes"]:
            assert e["frames"] == e["metadata_frames"]
            ref = decode(root / e["path"]).float() / 255
            first = s.k = e["start"]
            got = torch.stack([s.next().image for _ in range(e["frames"])])
            torch.testing.assert_close(got, ref)
            assert first == e["start"]

    def test_frame_fields_and_reset_at_episode_starts(self, prepared):
        root, out, index = prepared
        starts = {e["start"] for e in index["episodes"]}
        s = EgoDexStream(out)
        for k in range(s.frames):
            f = s.next()
            assert f.reset == (k in starts)
            assert f.image.shape == (3, 64, 64) and f.image.dtype == torch.float32
            assert 0 <= float(f.image.min()) and float(f.image.max()) <= 1
            assert f.state.shape == (0,) and f.params.shape == (0,)
        with pytest.raises(IndexError):
            s.next()

    def test_seconds_and_offset(self, prepared):
        root, out, index = prepared
        s = EgoDexStream(out, seconds=1.0, offset=5)
        for _ in range(30):
            s.next()
        with pytest.raises(IndexError):
            s.next()

    def test_task_and_pass_lookup(self, prepared):
        root, out, index = prepared
        s = EgoDexStream(out)
        for e in index["episodes"]:
            for k in (e["start"], e["start"] + e["frames"] - 1):
                assert s.task_of(k) == e["task"]
                assert s.pass_of(k) == e["round"]
        assert s.pass_of(0) == 0 and s.pass_of(s.frames - 1) == 1

    def test_clips(self, prepared):
        root, out, index = prepared
        assert index["clips"]["shape"] == [2, 32, 3, 64, 64]
        c = egodex_clips(out)
        assert c.image.shape == (1, 2, 32, 3, 64, 64)
        assert c.image.dtype == torch.float32
        assert c.state.shape == (1, 2, 32, 0) and c.params.shape == (1, 2, 32, 0)
        assert c.reset is False
        assert c.image[..., 0, :, :].min() > 0.85  # the red clips

    def test_calibration_images_from_the_first_half(self, prepared):
        root, out, index = prepared
        x = calibration_images(out, n=16, seed=0)
        assert x.shape == (16, 3, 64, 64) and x.dtype == torch.float32
        torch.testing.assert_close(x, calibration_images(out, n=16, seed=0))
        s = EgoDexStream(out)
        half = torch.stack([s.next().image for _ in range(s.frames // 2)])
        for img in x:
            assert (half == img).flatten(1).all(1).any()

    def test_index_records_the_setup(self, prepared):
        root, out, index = prepared
        index = json.loads((out / "index.json").read_text())
        assert "ffmpeg" in index and index["settings"]["seed"] == 0
        assert {"path", "task", "split", "start", "frames", "block", "round"} <= set(
            index["episodes"][0]
        )


def test_real_dataset_metadata_is_planned():
    root = Path("/datasets/egodex")
    if not (root / "video_metadata.json").exists():
        pytest.skip("EgoDex is not available")
    meta = egodex._metadata(root)
    plan = plan_stream(meta)
    assert len({e.task for e in plan}) == 24
    assert len(plan_clips(meta)) == 48
