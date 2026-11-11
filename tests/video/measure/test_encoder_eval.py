"""The evaluation afterwards of an encoder of the packed layout (M17) on a
tiny run of a small Walking Tours: the files, the keys, the objective, the
checkpoints it reads, the resume, and the training windows."""

from pathlib import Path

import pytest

from exp.lib.rundir import Exit
from exp.video import loop
from exp.video.main import local_job
from exp.video.measure.encoder_eval import EncoderEval, stream_windows
from exp.video.measure.objective import PENALTY
from exp.video.measure.posthoc import Parent
from exp.video.programs.vision import DATA_ENV, PROGRAM
from tests.video.data.test_walking_tours import numbered
from tests.video.packed import packed_conf, smooth_tours, tiny

LENGTHS = {"A": 120, "B": 80, "H": 200}


@pytest.fixture(scope="module")
def parent(tmp_path_factory) -> tuple[Path, Path]:
    """A finished tiny run with two checkpoints, and the data's root."""
    tmp = tmp_path_factory.mktemp("m17")
    root = smooth_tours(tmp / "data" / "d", LENGTHS)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(DATA_ENV, str(root.parent))
        mp.setattr(loop, "assert_free_threaded", lambda where: None)
        job = local_job(tiny(root.name), tmp / "run")
        assert loop.run(PROGRAM, job, attempt=1) == Exit.OK
    return tmp / "run", root


def _eval(parent, out: Path, **kw) -> EncoderEval:
    run_dir, root = parent
    p = Parent.read(run_dir, device="cpu")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(DATA_ENV, str(root.parent))
        return EncoderEval(p, p.conf, out, root=root, windows=16, steps=3, **kw)


def _files(out: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(out.iterdir())}


class TestEncoderEval:
    def test_the_units_lines_and_pictures(self, parent, tmp_path):
        e = _eval(parent, tmp_path / "o")
        assert len(e) == 3  # the yardsticks, two checkpoints
        lines = [ln for _ in range(len(e)) for ln in e.step()]
        assert [ln["code"] for ln in lines] == ["pca"] * 2 + ["init"] * 2 + ["vit"] * 4
        vit = lines[-2]
        assert vit["split"] == "val" and vit["t"] == 1.0
        for k in (
            "recon",
            "recon_fit",
            "dyn_1",
            "dyn_2",
            "copy_1",
            "dyn_1/copy_1",
            "img_dyn_1",
            "img_dyn_1/copy",
            "restore_short_mask",
            "restore_short_all",
            "restore_long_mask",
            "restore_long_all",
            "restore_target_all",
            "rank",
            "z_std_min",
            "z_spread",
            "alpha",
            "objective",
        ):
            assert k in vit, k
        assert "objective" not in lines[-1]  # test
        pca = lines[0]
        assert "recon_lin" in pca and "restore_short_all" not in pca
        want = max(vit["recon"] / pca["recon"], vit["dyn_1"] / pca["dyn_1"])
        assert vit["objective"] == pytest.approx(min(want, PENALTY))
        names = set(_files(tmp_path / "o"))
        for stem in ("pca", "init", "00000.5", "00001.0"):
            assert {f"{stem}.png", f"{stem}-dyn.png"} <= names
        assert "init-restore.png" in names and "pca-restore.png" not in names
        final = e.final()
        assert final["decode"]["val"] == vit
        assert set(final["baselines"]["test"]) == {"pca", "init"}

    def test_ckpts_last_reads_the_last_checkpoint_only(self, parent, tmp_path):
        e = _eval(parent, tmp_path / "o", ckpts="last")
        assert len(e) == 2
        e.step()
        assert [ln["t"] for ln in e.step()] == [1.0, 1.0]

    def test_a_resume_writes_the_bytes_of_one_never_stopped(self, parent, tmp_path):
        full = _eval(parent, tmp_path / "a")
        for _ in range(len(full)):
            full.step()
        part = _eval(parent, tmp_path / "b")
        part.step()
        part.step()
        with (tmp_path / "b" / "metrics.jsonl").open("a") as f:
            f.write('{"cut": 1}\n')  # a line of the unit that was cut
        rest = _eval(parent, tmp_path / "b", done=2)
        rest.step()
        a, b = _files(tmp_path / "a"), _files(tmp_path / "b")
        del a["config.json"], b["config.json"]
        assert a == b


def test_the_stream_windows_start_evenly_and_stay_within_a_video(tmp_path):
    """The loop A (0..29), B (30..49); window j starts at j 50 / 10 = 5 j, its
    frames are 2 apart, and a window of 4 + 3 frames (a span of 13) that
    would leave its video is left out: starts 0, 5, 10, 15 of A, 30, 35 of B."""
    root = numbered(tmp_path / "d", {"A": 30, "B": 20, "H": 10})
    conf = packed_conf(data_version=root.name, stride=2)
    w = stream_windows(conf, root, windows=10, steps=3)
    assert w.shape == (6, 7, 3, 2, 2)
    px = w[..., 0, 0].long()  # (6, 7, 3): video, j % 256, j // 256
    video, j = px[..., 0], px[..., 1] + 256 * px[..., 2]
    assert video[:, 0].tolist() == [0, 0, 0, 0, 1, 1]
    assert (video == video[:, :1]).all()
    assert j[:, 0].tolist() == [0, 5, 10, 15, 0, 5]
    assert (j[:, 1:] - j[:, :-1] == 2).all()
