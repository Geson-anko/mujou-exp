"""``exp.lib.tours.hires``: the plan's filter of the 854 x 480 videos, and on
synthetic lossless videos (``slow``) the 5 x 5 means of the 320 px frames are
the prepared 64 px frames by the checks (a) and (b), only the listed frames
are written and read back by their places, and prepared data one frame off
in time fails (b) and writes nothing."""

import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from exp.lib.tours.hires import (
    MAX_DIFF,
    HiresFrames,
    finalize,
    hires_filter,
    hires_geometry,
    position_check,
    prepare,
)
from exp.lib.video_io import INDEX, U8Reader, U8Writer
from tests.lib.tours.test_wide import H, W, entry, lossless, prepared


@pytest.fixture(scope="module")
def src(tmp_path_factory) -> tuple[Path, Path]:
    """Smooth waves that move fast (2 radians a frame at 10 fps) ``Waves``
    and a test pattern ``Other`` (2 s, 30 fps, 854 x 480), and their
    manifest."""
    d = tmp_path_factory.mktemp("src")
    waves = lossless(
        d / "Waves.mkv",
        f"color=c=black:s={W}x{H}:r=30:d=2,format=gbrp,"
        "geq=r='128+100*sin(X/13+20*T)':g='128+100*sin(Y/17-15*T)'"
        ":b='128+100*sin((X+Y)/29+9*T)'",
    )
    other = lossless(d / "Other.mkv", f"testsrc2=s={W}x{H}:r=30:d=2")
    path = d / "manifest.json"
    path.write_text(
        json.dumps({"videos": {"Waves": entry(waves), "Other": entry(other)}})
    )
    return d, path


class TestGeometry:
    def test_the_plans_filter(self):
        g = hires_geometry(W, H)
        assert g == {"side": 480, "x0": 187, "size": 320}
        assert hires_filter(g, fps=10.0) == (
            "fps=10,crop=480:480:187:0,scale=320:320:flags=area"
        )

    def test_a_square_off_whole_columns_is_refused(self):
        with pytest.raises(ValueError, match="whole columns"):
            hires_geometry(855, 480)


@pytest.mark.slow
class TestPrepare:
    def test_the_listed_frames_pass_the_checks_and_are_read_back(self, src, tmp_path):
        d, manifest = src
        data = prepared(d, manifest, tmp_path / "data", ["Waves", "Other"])
        out = tmp_path / "hires"
        lists = {"Waves": [7, 0, 19, 3, 7], "Other": [5]}
        for v, frames in lists.items():
            prepare(v, d, data, frames, out, manifest=manifest)
        reader = U8Reader(out / "Waves")
        assert len(reader) == 4 and reader.shape == (3, 320, 320)
        ref = U8Reader(data / "Waves")
        moved = 0
        for i, f in enumerate([0, 3, 7, 19]):
            check = position_check(reader[i], f, ref)
            assert check["a"] and check["b"] and check["diff"] <= MAX_DIFF
            moved += len(check["neighbours"])
        assert moved > 0  # (b) was tried
        root = finalize(out)
        assert root.name.startswith("hires-") and not out.exists()
        index = json.loads((root / INDEX).read_text())
        assert index["videos"]["Waves"]["frames"] == [0, 3, 7, 19]
        hires = HiresFrames(root)
        got = hires.frames([("Waves", 19), ("Other", 5), ("Waves", 0)])
        assert torch.equal(got[0], reader[3]) and torch.equal(got[2], reader[0])
        down = F.avg_pool2d(got[1:2].double(), 5)[0]
        other = U8Reader(data / "Other")[5].double()
        assert float((down - other).abs().mean()) <= MAX_DIFF
        with pytest.raises(ValueError, match="lacks"):
            hires.frames([("Waves", 1)])

    def test_data_one_frame_off_fails_b_and_writes_nothing(self, src, tmp_path):
        d, manifest = src
        data = prepared(d, manifest, tmp_path / "data", ["Waves"])
        ref = U8Reader(data / "Waves")
        late = tmp_path / "late"
        late.mkdir()
        (late / INDEX).write_text("{}")
        with U8Writer(late / "Waves", ref.shape, ref.index["meta"]) as w:
            w.append(ref[0].clone())
            for i in range(len(ref) - 1):  # frame f is the true f - 1
                w.append(ref[i].clone())
        with pytest.raises(ValueError, match=r"\(b\)"):
            prepare(
                "Waves", d, late, list(range(1, 20)), tmp_path / "h", manifest=manifest
            )
        assert not (tmp_path / "h" / "Waves").exists()
