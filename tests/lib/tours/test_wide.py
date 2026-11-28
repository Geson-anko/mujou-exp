"""``exp.lib.tours.wide``: the bands' columns of the plan's 854 x 480 videos,
a video too narrow refused, and on synthetic lossless videos (``slow``) the
square byte for byte the prepared data's, each band column the area mean of
its source columns, a mismatched data refused, and the index."""

import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
import pytest
import torch

from exp.lib.tours import walking_tours
from exp.lib.tours.wide import BAND, band_filter, band_geometry, finalize, prepare
from exp.lib.video_io import INDEX, U8Reader

W, H = 854, 480  # the three videos' (manifest)


class TestGeometry:
    def test_the_plans_columns_and_filter(self):
        g = band_geometry(W, H, size=64)
        assert g == {
            "side": 480,
            "x0": 187,
            "band_src": 180,
            "left": 7,
            "right": 667,
            "size": 64,
            "band": BAND,
        }
        assert band_filter(g, fps=10) == (
            "fps=10,split=3[a][b][c];"
            "[a]crop=180:480:7:0,scale=24:64:flags=area[l];"
            "[b]crop=480:480:187:0,scale=64:64:flags=area[m];"
            "[c]crop=180:480:667:0,scale=24:64:flags=area[r];"
            "[l][m][r]hstack=inputs=3"
        )

    def test_a_video_narrower_than_its_bands_is_refused(self):
        with pytest.raises(ValueError, match="fewer than a band"):
            band_geometry(640, 480, size=64)
        with pytest.raises(ValueError, match="whole columns"):
            band_geometry(855, 480, size=64)


def lossless(path: Path, graph: str) -> Path:
    """The lavfi ``graph`` in RGB, losslessly (FFV1), at ``path``."""
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y", "-f", "lavfi"]
    cmd += ["-i", graph, "-pix_fmt", "bgr0", "-c:v", "ffv1", str(path)]
    subprocess.run(cmd, check=True)
    return path


def decoded(path: Path) -> torch.Tensor:
    """The frames u8 ``(N, 3, H, W)`` of ``path`` at 10 fps, full size."""
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(path)]
    cmd += ["-vf", "fps=10", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return (
        torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        .view(-1, H, W, 3)
        .permute(0, 3, 1, 2)
    )


def entry(path: Path) -> dict:
    return {
        "file": path.name,
        "url": "https://www.youtube.com/watch?v=x",
        "dataset_license": "CC-BY-4.0",
        "upload_license": "unverified",
        "blur": "unverified",
        "status": "ok",
        "bytes": path.stat().st_size,
        "width": W,
        "height": H,
    }


@pytest.fixture(scope="module")
def src(tmp_path_factory) -> tuple[Path, Path]:
    """Moving test patterns ``Moving``, ``Other`` and vertical stripes
    ``Stripes`` (2 s, 30 fps, 854 x 480), and their manifest."""
    d = tmp_path_factory.mktemp("src")
    moving = lossless(d / "Moving.mkv", f"testsrc=s={W}x{H}:r=30:d=2")
    other = lossless(d / "Other.mkv", f"testsrc2=s={W}x{H}:r=30:d=2")
    stripes = lossless(
        d / "Stripes.mkv",
        f"color=c=black:s={W}x{H}:r=30:d=2,format=gbrp,"
        "geq=r='128+100*sin(X/9)':g='128+100*sin(X/23)':b='X/4'",
    )
    videos = {"Moving": entry(moving), "Other": entry(other), "Stripes": entry(stripes)}
    path = d / "manifest.json"
    path.write_text(json.dumps({"videos": videos}))
    return d, path


def prepared(d: Path, manifest: Path, out: Path, videos: list[str]) -> Path:
    """The finalized prepared data (64 px, 10 fps) of ``videos``."""
    for v in videos:
        walking_tours.prepare(v, d, out, fps=10.0, size=64, manifest=manifest)
    return walking_tours.finalize(out, expected=videos, manifest=manifest)


def area_means(row: torch.Tensor, x0: int, src_cols: int, out: int) -> torch.Tensor:
    """The mean of the source columns ``row`` ``(3, W)`` over each of
    ``out`` equal spans of ``src_cols`` from ``x0`` (the boundary columns
    weighted by their overlap): ``(3, out)``."""
    s = src_cols / out
    cols = []
    for j in range(out):
        a, b = x0 + j * s, x0 + (j + 1) * s
        total = torch.zeros(3, dtype=torch.float64)
        for c in range(int(a), min(int(b) + 1, row.shape[1])):
            total += max(0.0, min(b, c + 1) - max(a, c)) * row[:, c].double()
        cols.append(total / s)
    return torch.stack(cols, 1)


@pytest.mark.slow
class TestPrepare:
    def test_the_square_is_the_prepared_data_and_the_index(self, src, tmp_path):
        d, manifest = src
        data = prepared(d, manifest, tmp_path / "data", ["Moving", "Other"])
        out = tmp_path / "wide"
        for v in ("Moving", "Other"):
            prepare(v, d, data, out, manifest=manifest)
        wide = U8Reader(out / "Moving")
        ref = U8Reader(data / "Moving")
        assert wide.shape == (3, 64, 64 + 2 * BAND) and len(wide) == len(ref) == 20
        assert torch.equal(
            wide[0 : len(wide)][..., BAND : BAND + 64], ref[0 : len(ref)]
        )
        root = finalize(out)
        assert root.name.startswith("wide-") and not out.exists()
        index = json.loads((root / INDEX).read_text())
        assert set(index["videos"]) == {"Moving", "Other"}
        assert index["geometry"] == band_geometry(W, H, size=64)

    def test_each_band_column_is_the_area_mean_of_its_source(self, src, tmp_path):
        d, manifest = src
        data = prepared(d, manifest, tmp_path / "data", ["Stripes"])
        frame = U8Reader(
            prepare("Stripes", d, data, tmp_path / "wide", manifest=manifest)
        )[0]
        row = decoded(d / "Stripes.mkv")[0, :, 0]  # (3, W): the stripes are vertical
        g = band_geometry(W, H, size=64)
        for x0, cols in ((g["left"], slice(0, BAND)), (g["right"], slice(-BAND, None))):
            want = area_means(row, x0, g["band_src"], BAND)
            got = frame[:, 0, cols].double()
            assert float((got - want).abs().max()) <= 1.0

    def test_another_videos_data_is_refused(self, src, tmp_path):
        d, manifest = src
        data = prepared(d, manifest, tmp_path / "data", ["Moving"])
        (data / "Moving").rename(data / "Other")  # Other's data: Moving's frames
        with pytest.raises(ValueError, match="not the prepared data"):
            prepare("Other", d, data, tmp_path / "wide", manifest=manifest)
        assert not (tmp_path / "wide" / "Other").exists()
