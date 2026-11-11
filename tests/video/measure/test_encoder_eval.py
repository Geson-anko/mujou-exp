"""The evaluation afterwards of an encoder of the packed layout (M17) on a
tiny run of a small Walking Tours: the files, the keys, the objective, the
checkpoints it reads, the resume, and the training windows."""

import math
from pathlib import Path

import pytest
import torch

from exp.lib.rundir import Exit
from exp.video import loop
from exp.video.main import local_job
from exp.video.measure.encoder_eval import (
    EncoderEval,
    _objective,
    _Standardized,
    stream_windows,
)
from exp.video.measure.objective import FLOOR, PENALTY
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
            "dynw_1",
            "copyw_1",
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
        assert vit["objective"] == pytest.approx(_objective(vit, pca))
        names = set(_files(tmp_path / "o"))
        for stem in ("pca", "init", "00000.5", "00001.0"):
            assert {f"{stem}.png", f"{stem}-dyn.png"} <= names
        assert "init-restore.png" in names and "pca-restore.png" not in names
        final = e.final()
        assert final["decode"]["val"] == vit
        assert set(final["baselines"]["test"]) == {"pca", "init"}

    def test_the_encoder_is_held_bit_for_bit(self, parent, tmp_path, monkeypatch):
        """Every module of each code (the ViT before it learns, each
        checkpoint) has the same bytes after its evaluation (the decoders,
        the ridge and the token decoder fit) as before."""
        held = []
        evaluate = EncoderEval._evaluate

        def watched(self, code, *args, **kw):
            if code.mods is None:
                return evaluate(self, code, *args, **kw)
            before = {
                f"{n}.{k}": x.detach().clone()
                for n, m in code.mods.named().items()
                for k, x in m.state_dict().items()
            }
            out = evaluate(self, code, *args, **kw)
            after = {
                f"{n}.{k}": x
                for n, m in code.mods.named().items()
                for k, x in m.state_dict().items()
            }
            assert before.keys() == after.keys()
            for k, x in before.items():
                assert torch.equal(x, after[k]), k
            held.append(code.name)
            return out

        monkeypatch.setattr(EncoderEval, "_evaluate", watched)
        e = _eval(parent, tmp_path / "o")
        for _ in range(len(e)):
            e.step()
        assert held == ["init", "vit", "vit"]

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


def test_the_decoder_sees_z_standardized_on_the_training_codes():
    """Components of std 10 and 0.1 about means 5 and -1 reach the decoder at
    mean 0 and std 1; a constant one stays 0."""
    g = torch.Generator().manual_seed(0)
    z = torch.randn(4096, 3, generator=g) * torch.tensor([10.0, 0.1, 0.0])
    z = z + torch.tensor([5.0, -1.0, 2.0])
    std = _Standardized(z).standardize(z)
    torch.testing.assert_close(std.mean(0), torch.zeros(3), atol=1e-5, rtol=0)
    torch.testing.assert_close(std[:, :2].std(0), torch.ones(2), atol=1e-5, rtol=0)
    assert (std[:, 2] == 0).all()


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


PCA = {"recon": 0.2, "dyn_1/copy_1": 0.5}


def _line(**kw) -> dict:
    return {"recon": 0.3, "dyn_1/copy_1": 0.5, "z_spread": 1.0} | kw


class TestObjective:
    """J = log max(recon / recon_PCA, r_1 / r_1_PCA), r_1 = dyn_1/copy_1, by
    hand on lines of round numbers."""

    def test_the_log_of_the_worse_ratio(self):
        assert _objective(_line(), PCA) == pytest.approx(math.log(1.5))
        line = _line(**{"recon": 0.1, "dyn_1/copy_1": 1.0})
        assert _objective(line, PCA) == pytest.approx(math.log(2.0))

    def test_better_than_the_pca_on_both_is_negative(self):
        line = _line(**{"recon": 0.1, "dyn_1/copy_1": 0.125})
        assert _objective(line, PCA) == pytest.approx(math.log(0.5))

    def test_ratios_up_to_ten_stay_apart_and_beyond_take_the_penalty(self):
        """Ratios 3.7 and 9 (the ViT before it learns is about 3.7) rank
        apart, below log 10; 20 is log 10 as a collapse."""
        values = [_objective(_line(recon=0.2 * r), PCA) for r in (3.7, 9.0, 20.0)]
        assert values[0] == pytest.approx(math.log(3.7))
        assert values[1] == pytest.approx(math.log(9.0))
        assert values[2] == PENALTY

    def test_a_collapse_is_the_penalty_however_good_the_ratios(self):
        line = _line(recon=0.01, z_spread=1e-4)
        assert _objective(line, PCA) == PENALTY
        assert _objective({"recon": 0.01, "dyn_1/copy_1": 0.1}, PCA) == PENALTY

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), None])
    def test_a_value_that_is_not_finite_is_the_penalty(self, value):
        assert _objective(_line(recon=value), PCA) == PENALTY
        assert _objective(_line(), PCA | {"dyn_1/copy_1": value}) == PENALTY

    def test_a_perfect_code_counts_as_the_floor(self):
        line = _line(**{"recon": 0.0, "dyn_1/copy_1": 0.0})
        assert _objective(line, PCA) == pytest.approx(math.log(FLOOR))
