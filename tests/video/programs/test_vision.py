"""The program of M15 run by the loop on small jobs: the files of the run
contract, determinism, resume (the Chunker's ring and generator among the
state), the prequential record, Walking Tours as the stream, and the GIL."""

import json
import os
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Any

import pytest
import torch

from exp.lib import ckpt
from exp.lib.rundir import Exit, verify_manifest
from exp.video import loop
from exp.video.compose import compose, from_dict
from exp.video.jepa.vit import build_vision
from exp.video.main import local_job
from exp.video.measure.objective import PENALTY
from exp.video.programs.vision import (
    COLLAPSE,
    DATA_ENV,
    PROGRAM,
    Measures,
    rebuild,
    vision_objective,
)
from tests.video.data.test_walking_tours import numbered

ROOT = Path(__file__).resolve().parents[3]
TINY = [
    "data.streams=2",
    "data.balls.size=16",
    "measure.clips=4",
    "measure.every=0.5",
    "seconds=1",
    "vision.dim=16",
    "vision.depth=2",
    "vision.heads=2",
    "vision.pred_depth=1",
    "vision.sigreg_directions=16",
]


@pytest.fixture(autouse=True)
def _no_gil_check(monkeypatch):
    """The GIL of this test process is pytest's (TestGIL checks it in a run of
    its own)."""
    monkeypatch.setattr(loop, "assert_free_threaded", lambda where: None)


def _run(
    out: Path,
    overrides: list[str],
    program: Any = PROGRAM,
    attempt: int = 1,
    experiment: str = "m15/balls_n1",
    tiny: list[str] = TINY,
) -> int:
    job = local_job([f"+experiment={experiment}", *tiny, *overrides], out)
    return loop.run(program, job, attempt=attempt)


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


class _Stopping:
    """The program, with ``stop.request`` made after frame ``at``."""

    def __init__(self, at: int) -> None:
        self.at = at

    def build(self, conf, run_dir: Path, resume):
        built = PROGRAM.build(conf, run_dir, resume)
        advance, count = built.advance, [0]

        def stopping() -> None:
            advance()
            count[0] += 1
            if count[0] == self.at:
                (run_dir / "stop.request").touch()

        built.advance = stopping
        return built


class TestContract:
    def test_a_small_job_writes_every_file_of_the_contract(self, tmp_path):
        out = tmp_path / "run"
        confirm = ["ctl.ckpt_every=0.5", "ctl.viz_every=1", "ctl.eval_test=true"]
        assert _run(out, ["vision.frames=2", *confirm]) == Exit.OK
        for name in (
            "job.json",
            "config.yaml",
            "CODE.json",
            "code.diff",
            "provenance/attempt-1.json",
            "status.json",
            "timing.jsonl",
            "stream.jsonl",
            "evals.jsonl",
            "baselines.json",
            "result.json",
            "manifest.json",
            "ckpt/00000.5.pt",
            "ckpt/00001.0.pt",
            "viz/00001.0/views.png",
        ):
            assert (out / name).is_file(), name
        assert not (out / "resume").exists()
        verify_manifest(out)
        stream = _lines(out / "stream.jsonl")
        assert [x["frames"] for x in stream] == [30]
        assert set(stream[0]) == {"t", "frames", "loss", "pred", "sigreg"}
        evals = _lines(out / "evals.jsonl")
        assert [(x["t"], x["split"]) for x in evals] == [
            (0.5, "val"),
            (0.5, "test"),
            (1.0, "val"),
            (1.0, "test"),
        ]
        val = evals[0]
        for key in (
            "loss",
            "pred",
            "sigreg",
            "rank",
            "z_std_min",
            "z_spread",
            "thumb",
            "probe_x2",
            "probe_vy0",
            "probe_pos_min",
            "probe_vel_min",
            "objective",
        ):
            assert key in val, key
        assert "objective" not in evals[1]
        result = json.loads((out / "result.json").read_text())
        assert result["status"] == "completed"
        assert result["final"]["test"] == evals[-1]
        assert result["objective"] == evals[-2]["objective"]
        base = json.loads((out / "baselines.json").read_text())
        assert {
            "init_probe_pos_min",
            "pixels_probe_pos_min",
            "init_rank",
            "init_z_spread",
        } <= set(base["val"])
        conf = from_dict(json.loads((out / "job.json").read_text())["config"])
        header = ckpt.load_eval(out / "ckpt" / "00001.0.pt", rebuild(conf))
        assert header["frame"] == 30 and header["program"] == "vision"

    def test_the_same_job_twice_writes_the_same_bytes(self, tmp_path):
        for d in ("a", "b"):
            assert _run(tmp_path / d, ["learner=pc"]) == Exit.OK
        for name in ("evals.jsonl", "stream.jsonl", "baselines.json"):
            assert (tmp_path / "a" / name).read_bytes() == (
                tmp_path / "b" / name
            ).read_bytes()

    def test_the_bench_task_evaluates_nothing(self, tmp_path):
        out = tmp_path / "run"
        assert _run(out, ["task=bench", "seconds=0.5"]) == Exit.OK
        result = json.loads((out / "result.json").read_text())
        assert set(result["final"]) == {"timing"}
        assert result["final"]["timing"]["fps"] > 0
        assert not (out / "baselines.json").exists()


class TestRecon:
    def test_off_by_default_it_builds_no_decoder_and_records_no_recon(self, tmp_path):
        conf = compose(["+experiment=m15/balls_n1"])
        assert conf.vision is not None and conf.vision.recon == 0
        assert build_vision(conf, seed=0).decoder is None
        assert set(rebuild(conf)) == {"encoder", "predictor"}
        assert _run(tmp_path, ["learner=pc"]) == Exit.OK
        for name in ("evals.jsonl", "stream.jsonl"):
            assert "recon" not in (tmp_path / name).read_text()

    def test_the_other_weights_do_not_depend_on_the_decoder(self):
        """The decoder is built after the encoder and the predictor, so a run
        with recon starts from the weights of the run without."""
        off = compose(["+experiment=m15/balls_n1"])
        on = compose(["+experiment=m15/balls_n1", "vision.recon=1.0"])
        a, b = build_vision(off, seed=3), build_vision(on, seed=3)
        for x, y in zip(
            [*a.encoder.parameters(), *a.predictor.parameters()],
            [*b.encoder.parameters(), *b.predictor.parameters()],
            strict=True,
        ):
            assert torch.equal(x, y)
        assert b.decoder is not None

    @pytest.mark.parametrize("learner", ["pc", "bp"])
    def test_a_run_records_recon_and_checkpoints_the_decoder(self, tmp_path, learner):
        out = tmp_path / "run"
        extra = [f"learner={learner}", "vision.recon=1.0", "ctl.ckpt_every=0.5"]
        assert _run(out, extra) == Exit.OK
        for name in ("evals.jsonl", "stream.jsonl"):
            rows = [r for r in _lines(out / name) if r.get("split", "val") == "val"]
            assert rows and all(r["recon"] > 0 for r in rows[-1:])
        conf = from_dict(json.loads((out / "job.json").read_text())["config"])
        mods = rebuild(conf)
        assert set(mods) == {"encoder", "predictor", "decoder"}
        ckpt.load_eval(out / "ckpt" / "00001.0.pt", mods)

    @pytest.mark.parametrize("learner", ["pc", "bp"])
    def test_resume_equals_uninterrupted_with_recon(self, tmp_path, learner):
        extra = [
            f"learner={learner}",
            "vision.recon=1.0",
            "vision.recon_hidden=16",
            "vision.frames=2",
            "optim_vision.warmup=0.5",
            "optim_vision.lr=1e-3",
            "ctl.resume_every=0.3",
            "ctl.log_every=0.2",
        ]
        assert _run(tmp_path / "whole", extra) == Exit.OK
        cut = tmp_path / "cut"
        assert _run(cut, extra, _Stopping(23)) == Exit.STOPPED
        (cut / "stop.request").unlink()
        assert _run(cut, extra, attempt=2) == Exit.OK
        for name in ("evals.jsonl", "stream.jsonl"):
            assert (cut / name).read_bytes() == (tmp_path / "whole" / name).read_bytes()
        assert '"recon"' in (cut / "stream.jsonl").read_text()


class TestResume:
    @pytest.mark.parametrize("learner", ["pc", "bp"])
    def test_resume_equals_uninterrupted(self, tmp_path, learner):
        """Stopped at frame 23 (past the first evaluation, between two resume
        checkpoints) and resumed, the run writes the bytes of one never
        stopped: the nodes' states, the Chunker's ring of N = 2 frames and its
        generator (the masks), the head's directions, the optimizers (AdamW in
        a warm-up), the streams and the sums since the last line resume."""
        extra = [
            f"learner={learner}",
            "vision.frames=2",
            "optim_vision.warmup=0.5",
            "optim_vision.lr=1e-3",
            "ctl.resume_every=0.3",
            "ctl.log_every=0.2",
        ]
        assert _run(tmp_path / "whole", extra) == Exit.OK
        cut = tmp_path / "cut"
        assert _run(cut, extra, _Stopping(23)) == Exit.STOPPED
        assert (cut / "resume" / "00000023.pt").is_file()
        (cut / "stop.request").unlink()
        assert _run(cut, extra, attempt=2) == Exit.OK
        for name in ("evals.jsonl", "stream.jsonl"):
            assert (cut / name).read_bytes() == (tmp_path / "whole" / name).read_bytes()


class TestRecords:
    @pytest.mark.parametrize("learner", ["pc", "bp"])
    def test_the_loss_recorded_is_the_one_before_learning(self, tmp_path, learner):
        """With N = 2, frame 1 has the first full chunk, and its loss is
        computed before anything learns from it: its record is the same at any
        learning rate; later frames' are not."""
        lines = []
        for lr in ("1e-5", "0.3"):
            out = tmp_path / lr
            ctl = [
                "ctl.log_every=0.03",
                "optim@optim_vision=sgd",
                f"optim_vision.lr={lr}",
                "optim_vision.warmup=0",
                f"learner={learner}",
                "measure.every=0",
                "vision.frames=2",
            ]
            assert _run(out, ["seconds=0.2", *ctl]) == Exit.OK
            lines.append(_lines(out / "stream.jsonl"))
        a, b = lines
        assert a[0]["loss"] is None  # frame 0 is not counted (nor full)
        assert a[1]["frames"] == 2 and a[1]["loss"] == b[1]["loss"]
        assert a[4]["loss"] != b[4]["loss"]


class TestTours:
    def test_walking_tours_is_a_stream_of_the_run(self, tmp_path, monkeypatch):
        """Prepared data under $MUJOU_EXP_DATA/<data_version>: the run streams
        the videos but the held-out one and evaluates on the held-out one."""
        root = numbered(tmp_path / "data" / "d", {"A": 40, "B": 30, "H": 96}, size=8)
        monkeypatch.setenv(DATA_ENV, str(root.parent))
        tours = [
            f"data.tours.data_version={root.name}",
            "data.tours.held_out=H",
            "data.tours.size=8",
            "vision.patch=4",
        ]
        tiny = [o for o in TINY if not o.startswith("data.balls")]
        out = tmp_path / "run"
        assert (
            _run(out, ["vision.frames=2", *tours], experiment="m15/wt_n8", tiny=tiny)
            == Exit.OK
        )
        row = _lines(out / "evals.jsonl")[-1]
        assert {"loss", "rank", "z_std_min", "z_spread", "thumb", "objective"} <= set(
            row
        )
        assert not any(k.startswith("probe") for k in row)

    def test_calib_val_and_test_are_the_thirds_of_the_held_out_video_in_time(
        self, tmp_path, monkeypatch
    ):
        """No chunk of val is next to one of test (nor calib to val): the
        frames of each split come after those of the split before."""
        root = numbered(tmp_path / "d", {"A": 40, "H": 200}, size=8)
        monkeypatch.setenv(DATA_ENV, str(root.parent))
        tours = [
            f"data.tours.data_version={root.name}",
            "data.tours.held_out=H",
            "data.tours.size=8",
            "vision.patch=4",
            "vision.frames=2",
        ]
        tiny = [o for o in TINY if not o.startswith("data.balls")]
        job = local_job(["+experiment=m15/wt_n8", *tiny, *tours], tmp_path / "run")
        conf = from_dict(json.loads(job.read_text())["config"])
        sets = Measures(conf).sets
        px = {k: (s.chunks[..., 1, 0, 0] * 255).round().long() for k, s in sets.items()}
        assert all(len(x) == 16 for x in px.values())
        assert px["calib"].max() < px["val"].min()
        assert px["val"].max() < px["test"].min()

    def test_without_the_data_the_run_is_an_environment_error(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv(DATA_ENV, str(tmp_path / "nowhere"))
        out = tmp_path / "run"
        tiny = [o for o in TINY if not o.startswith("data.balls")]
        assert _run(out, [], experiment="m15/wt_n1", tiny=tiny) == Exit.ENV


def test_the_initial_z_of_the_balls_is_spread():
    """The ViT of m15/balls_n1 before it learns gives z that differ from
    image to image: z_spread 0.033 on val (with the patch embedding of std
    0.02 on the raw pixels it was 9e-4, nearly the same z for every image,
    where SIGReg on z cannot pull the samples apart)."""
    conf = compose(["+experiment=m15/balls_n1", "measure.clips=64"])
    row = Measures(conf).evaluate(build_vision(conf, seed=0), "val", version=0)
    assert row["z_spread"] > 0.01 and row["z_std_min"] > 1e-3


@pytest.mark.cuda
@pytest.mark.parametrize("learner", ["pc", "bp"])
def test_a_run_on_cuda_completes(tmp_path, learner):
    """The stats the host reads back (into host memory) add to the sums of
    stream.jsonl."""
    assert _run(tmp_path / "run", ["device=cuda", f"learner={learner}"]) == Exit.OK


@pytest.mark.cuda
@pytest.mark.parametrize("learner", ["pc", "bp"])
def test_a_run_with_recon_on_cuda_completes(tmp_path, learner):
    out = tmp_path / "run"
    assert (
        _run(out, ["device=cuda", f"learner={learner}", "vision.recon=1.0"]) == Exit.OK
    )
    assert '"recon"' in (out / "stream.jsonl").read_text()


def test_the_objective():
    """1 - the probe of the balls' positions, else 1 - that of the thumbnail;
    PENALTY when z collapsed or a value is missing, at most PENALTY."""
    ok = {"z_spread": 0.1}
    assert vision_objective(ok | {"probe_pos_min": 0.9, "thumb": 0.2}) == (
        pytest.approx(0.1)
    )
    assert vision_objective(ok | {"thumb": 0.75}) == pytest.approx(0.25)
    assert vision_objective(ok | {"probe_pos_min": None}) == PENALTY
    assert vision_objective(ok | {"probe_pos_min": -5.0}) == PENALTY
    collapsed = {"z_spread": COLLAPSE / 2, "probe_pos_min": 0.9}
    assert vision_objective(collapsed) == PENALTY
    assert vision_objective({"z_spread": None, "thumb": 0.9}) == PENALTY


class TestGIL:
    def test_a_local_run_stays_free_threaded(self, tmp_path):
        """``python -m exp.video.main +experiment=m15/balls_n1 ...`` runs with
        the GIL disabled through the import, the build and the evaluations
        (the loop checks each and would exit 4)."""
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("PYTHON_GIL", "OMP_NUM_THREADS")
        }
        out = tmp_path / "run"
        args = ["+experiment=m15/balls_n1", *TINY, "seconds=0.5"]
        proc = subprocess.run(
            [sys.executable, "-m", "exp.video.main", *args, "--out", str(out)],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert proc.returncode == Exit.OK, proc.stderr
        prov = json.loads((out / "provenance" / "attempt-1.json").read_text())
        if sysconfig.get_config_var("Py_GIL_DISABLED"):
            assert prov["gil_enabled"] is False
        verify_manifest(out)
