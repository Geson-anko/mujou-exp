import hashlib
import json
from pathlib import Path

import pytest
import torch

from exp.lib.video_io import INDEX, U8Reader, U8Writer
from exp.video.data.walking_tours import (
    MANIFEST,
    data_version,
    eval_clips,
    finalize,
    prepare,
    videos,
)
from tests.lib.test_video_io import make_video


def color_video(path: Path, color: str, seconds: float = 2.0) -> Path:
    return make_video(path, f"color=c={color}:s=160x90:r=30:d={seconds}")


def entry(path: Path, status: str = "ok") -> dict:
    return {
        "file": path.name,
        "url": "https://www.youtube.com/watch?v=x",
        "dataset_license": "CC-BY-4.0",
        "upload_license": "unverified",
        "blur": "unverified",
        "status": status,
        "bytes": path.stat().st_size,
    }


@pytest.fixture(scope="module")
def src(tmp_path_factory) -> tuple[Path, Path]:
    """Videos Red, Lime (2 s each, 30 fps) and an excluded Blue, with their
    manifest."""
    d = tmp_path_factory.mktemp("src")
    red = color_video(d / "Red.mp4", "red")
    lime = color_video(d / "Lime.mp4", "lime")
    blue = color_video(d / "Blue.mp4", "blue")
    blue_entry = entry(blue, "excluded") | {"reason": "a test"}
    manifest = {"videos": {"Red": entry(red), "Lime": entry(lime), "Blue": blue_entry}}
    path = d / "manifest.json"
    path.write_text(json.dumps(manifest))
    return d, path


class TestManifest:
    def test_every_video_records_its_source_license_and_blur(self):
        entries = json.loads(MANIFEST.read_text())["videos"]
        assert len(entries) == 10
        for video, e in entries.items():
            assert e["file"] == f"{video}.mp4"
            assert e["url"].startswith("https://www.youtube.com/watch?v=")
            assert e["dataset_license"] == "CC-BY-4.0"
            assert e["upload_license"] in ("CC-BY-4.0", "youtube", "unverified")
            assert e["blur"] in ("yes", "no", "unverified")
            assert e["status"] in ("ok", "excluded")
            assert e["status"] == "ok" or e["reason"]
            assert e["bytes"] > 0


class TestPrepare:
    def test_frames_are_the_video_at_fps_and_size(self, src, tmp_path):
        d, manifest = src
        prepare("Red", d, tmp_path, fps=10.0, size=16, manifest=manifest)
        r = U8Reader(tmp_path / "Red")
        assert len(r) == 20  # 2 s at 10 fps
        assert r.shape == (3, 16, 16)
        x = r[:].float() / 255
        assert x[:, 0].min() > 0.95 and x[:, 1:].max() < 0.05
        r.verify()
        assert r.index["meta"]["video"] == "Red"
        assert r.index["meta"]["blur"] == "unverified"
        assert not (tmp_path / "Red.tmp").exists()

    def test_a_video_not_in_the_manifest_is_refused(self, src, tmp_path):
        d, manifest = src
        with pytest.raises(ValueError, match="not in"):
            prepare("Atlantis", d, tmp_path, manifest=manifest)
        with pytest.raises(ValueError, match="not in"):
            prepare("Atlantis", d, tmp_path)  # the real manifest
        assert not list(tmp_path.iterdir())

    def test_an_excluded_video_is_refused(self, src, tmp_path):
        d, manifest = src
        with pytest.raises(ValueError, match="excluded"):
            prepare("Blue", d, tmp_path, manifest=manifest)
        assert not list(tmp_path.iterdir())

    def test_a_file_of_another_size_is_refused(self, src, tmp_path):
        d, manifest = src
        partial = tmp_path / "src"
        partial.mkdir()
        data = (d / "Red.mp4").read_bytes()
        (partial / "Red.mp4").write_bytes(data[: len(data) // 2])
        with pytest.raises(ValueError, match="bytes"):
            prepare("Red", partial, tmp_path / "out", manifest=manifest)

    def test_an_existing_output_is_kept(self, src, tmp_path):
        d, manifest = src
        prepare("Red", d, tmp_path, size=8, manifest=manifest)
        with pytest.raises(FileExistsError):
            prepare("Red", d, tmp_path, size=8, manifest=manifest)

    def test_a_stale_tmp_is_replaced(self, src, tmp_path):
        d, manifest = src
        (tmp_path / "Red.tmp").mkdir()
        (tmp_path / "Red.tmp" / "junk").write_text("x")
        prepare("Red", d, tmp_path, size=8, manifest=manifest)
        assert len(U8Reader(tmp_path / "Red")) == 20


class TestFinalize:
    def test_version_is_the_hash_of_the_index_and_names_the_dir(self, src, tmp_path):
        d, manifest = src
        root = tmp_path / "staging"
        for v in ("Red", "Lime"):
            prepare(v, d, root, size=8, manifest=manifest)
        out = finalize(root, manifest=manifest)  # every ok video of it
        assert not root.exists()
        digest = hashlib.sha256((out / INDEX).read_bytes()).hexdigest()
        assert out.name == digest[:12] == data_version(out / INDEX)
        assert videos(out) == ["Lime", "Red"]
        index = json.loads((out / INDEX).read_text())
        red = hashlib.sha256((out / "Red" / INDEX).read_bytes()).hexdigest()
        assert index["videos"]["Red"] == {"frames": 20, "index_sha256": red}
        assert finalize(out, manifest=manifest) == out  # again: no change

    def test_same_inputs_give_the_same_version(self, src, tmp_path):
        d, manifest = src
        names = []
        for k in range(2):
            prepare("Red", d, tmp_path / f"s{k}", size=8, manifest=manifest)
            names.append(finalize(tmp_path / f"s{k}", expected=["Red"]).name)
            (tmp_path / names[-1]).rename(tmp_path / f"done{k}")
        assert names[0] == names[1]

    def test_another_frame_gives_another_version(self, tmp_path):
        x = torch.zeros(2, 3, 4, 4, dtype=torch.uint8)
        versions = []
        for k in range(2):
            meta = {"video": "A", "fps": 10, "size": 4}
            with U8Writer(tmp_path / f"s{k}" / "A", (3, 4, 4), meta) as w:
                w.append(x[0])
                w.append(x[1] + k)
            versions.append(finalize(tmp_path / f"s{k}", expected=["A"]).name)
        assert versions[0] != versions[1]

    def test_mixed_settings_are_refused(self, tmp_path):
        for v, size in (("A", 4), ("B", 8)):
            meta = {"video": v, "fps": 10, "size": size}
            with U8Writer(tmp_path / v, (3, size, size), meta):
                pass
        with pytest.raises(ValueError, match="differ"):
            finalize(tmp_path, expected=["A", "B"])

    def test_a_missing_video_is_refused(self, src, tmp_path):
        d, manifest = src
        prepare("Red", d, tmp_path, size=8, manifest=manifest)  # Lime is not
        with pytest.raises(ValueError, match=r"missing \['Lime'\]"):
            finalize(tmp_path, manifest=manifest)
        with pytest.raises(ValueError, match=r"unexpected \['Red'\]"):
            finalize(tmp_path, expected=["Lime"])
        assert not (tmp_path / INDEX).exists()

    def test_an_unfinished_prep_is_refused(self, src, tmp_path):
        d, manifest = src
        for v in ("Red", "Lime"):
            prepare(v, d, tmp_path, size=8, manifest=manifest)
        (tmp_path / "Lime.tmp").mkdir()  # a second prep of Lime still running
        with pytest.raises(ValueError, match="Lime.tmp"):
            finalize(tmp_path, manifest=manifest)

    def test_an_index_of_another_video_is_refused(self, tmp_path):
        with U8Writer(tmp_path / "A", (3, 4, 4), {"video": "B", "fps": 10, "size": 4}):
            pass
        with pytest.raises(ValueError, match="of video 'B'"):
            finalize(tmp_path, expected=["A"])


def numbered(root: Path, lengths: dict[str, int], size: int = 2) -> Path:
    """Finalized data whose frame j of video v has pixels (v, j % 256, j // 256)
    with v the index of the video in name order."""
    for v, (name, n) in enumerate(sorted(lengths.items())):
        meta = {"video": name, "fps": 10.0, "size": size}
        with U8Writer(root / name, (3, size, size), meta) as w:
            for j in range(n):
                px = torch.tensor([v, j % 256, j // 256], dtype=torch.uint8)
                w.append(px.view(3, 1, 1).expand(3, size, size))
    return finalize(root, expected=list(lengths))


class TestEvalClips:
    @pytest.fixture
    def root(self, tmp_path) -> Path:
        return numbered(tmp_path / "d", {"A": 50, "H": 100})

    def frame_numbers(self, clips: torch.Tensor) -> torch.Tensor:
        px = clips[..., 0, 0].long()  # (clips, length, 3)
        return px[..., 1] + 256 * px[..., 2]

    def test_clips_are_consecutive_frames_of_the_held_out_video(self, root):
        x = eval_clips(root, held_out="H", clips=4, length=10, seed=0)
        assert x.shape == (4, 10, 3, 2, 2) and x.dtype == torch.uint8
        assert (x[:, :, 0] == 1).all()  # video H only
        j = self.frame_numbers(x)
        assert (j[:, 1:] - j[:, :-1] == 1).all()
        assert (j[:, 0] % 10 == 0).all()  # starts on the slots
        assert (j[1:, 0] > j[:-1, 0]).all()  # in time order, no shared frame

    def test_seed_fixes_the_clips(self, root):
        a = eval_clips(root, held_out="H", clips=4, length=10, seed=0)
        b = eval_clips(root, held_out="H", clips=4, length=10, seed=0)
        c = eval_clips(root, held_out="H", clips=4, length=10, seed=1)
        assert torch.equal(a, b) and not torch.equal(a, c)

    def test_all_slots(self, root):
        j = self.frame_numbers(
            eval_clips(root, held_out="H", clips=10, length=10, seed=3)
        )
        assert j[:, 0].tolist() == list(range(0, 100, 10))

    def test_too_many_clips_or_an_unknown_video_raise(self, root):
        with pytest.raises(ValueError, match="clips"):
            eval_clips(root, held_out="H", clips=11, length=10, seed=0)
        with pytest.raises(ValueError, match="not in"):
            eval_clips(root, held_out="Z", clips=1, length=10, seed=0)
        with pytest.raises(ValueError, match="positive"):
            eval_clips(root, held_out="H", clips=1, length=0, seed=0)
