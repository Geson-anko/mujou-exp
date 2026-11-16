import json
import math
import os
from pathlib import Path
from typing import Any

import pytest

from exp.lib import ckpt
from exp.lib.fleet.jobs import make_job
from exp.lib.rundir import Exit, verify_manifest
from exp.video import loop
from exp.video.compose import compose, to_dict
from exp.video.main import local_job, main as video_main
from exp.video.measure.posthoc import Parent
from exp.video.programs.vision import DATA_ENV
from exp.video.vae.eval import AEEval
from exp.video.vae.program import PROGRAM, rebuild
from tests.video.packed import smooth_tours, tiny


def tiny_ae(version: str) -> list[str]:
    """A run of ``m18/ae_t128`` of a second on 16 x 16 frames of the data
    ``version`` (held out ``H``): the small ViT of ``tests.video.packed.tiny``
    with tokens of 4 and a decoder of one block, two streams."""
    kept = [o for o in tiny(version) if not o.startswith(("+experiment", "learner"))]
    return [
        "+experiment=m18/ae_t128",
        *kept,
        "vision.token_dim=4",
        "ae.decoder_depth=1",
        "optim_vision.lr=1e-3",
    ]


def tiny_vae(version: str) -> list[str]:
    """``tiny_ae`` of the VAE (``m18/vae_t128``): the same small run."""
    kept = [o for o in tiny_ae(version) if not o.startswith("+experiment")]
    return ["+experiment=m18/vae_t128", *kept]


@pytest.fixture(autouse=True)
def _no_gil_check(monkeypatch):
    monkeypatch.setattr(loop, "assert_free_threaded", lambda where: None)


@pytest.fixture
def tours(tmp_path, monkeypatch) -> list[str]:
    root = smooth_tours(tmp_path / "data" / "d", {"A": 120, "B": 80, "H": 200})
    monkeypatch.setenv(DATA_ENV, str(root.parent))
    return tiny_ae(root.name)


def _run(out: Path, overrides: list[str], program: Any = PROGRAM) -> int:
    return loop.run(program, local_job(overrides, out), attempt=1)


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

        built.advance = stopping  # type: ignore[method-assign]  # the test's hook
        return built


class TestOnline:
    @pytest.mark.parametrize("sampling", ["stream", "shuffle", "pool"])
    def test_a_run_writes_the_rows_of_the_autoencoder(self, tmp_path, tours, sampling):
        out = tmp_path / "run"
        extra = [f"data.tours.sampling={sampling}", "data.tours.pool=5"]
        assert _run(out, [*tours, *extra]) == Exit.OK
        rows = _lines(out / "evals.jsonl")
        assert [(r["t"], r["split"]) for r in rows] == [(0.5, "val"), (1.0, "val")]
        pool = {"pool_stack_mse", "pool_loss"} if sampling == "pool" else set()
        assert set(rows[-1]) == {"t", "frame", "split", "stack_mse", "loss"} | pool
        assert all(math.isfinite(rows[-1][k]) for k in ("stack_mse", "loss", *pool))
        logged = _lines(out / "stream.jsonl")
        assert logged and all(r["loss"] is not None for r in logged)
        base = json.loads((out / "baselines.json").read_text())
        assert set(base) == {"val", "test"} and "stack_mse" in base["val"]
        conf = compose(tours)
        head = ckpt.load_eval(out / "ckpt" / "00001.0.pt", rebuild(conf))
        assert head["t"] == 1.0
        result = json.loads((out / "result.json").read_text())
        assert result["objective"] == rows[-1]["stack_mse"]

    def test_the_bench_task_evaluates_nothing(self, tmp_path, tours):
        out = tmp_path / "run"
        assert _run(out, [*tours, "task=bench"]) == Exit.OK
        assert not (out / "evals.jsonl").read_text()
        assert not (out / "baselines.json").exists()

    @pytest.mark.parametrize("sampling", ["stream", "shuffle"])
    def test_resume_equals_uninterrupted(self, tmp_path, tours, sampling):
        o = [*tours, f"data.tours.sampling={sampling}", "ctl.resume_every=0.2"]
        whole, parts = tmp_path / "whole", tmp_path / "parts"
        assert _run(whole, o) == Exit.OK
        job = local_job(o, parts)
        assert loop.run(_Stopping(13), job, attempt=1) == Exit.STOPPED
        (parts / "stop.request").unlink()
        assert loop.run(PROGRAM, job, attempt=2) == Exit.OK
        for name in ("evals.jsonl", "stream.jsonl"):
            assert (whole / name).read_bytes() == (parts / name).read_bytes(), name


CODE = {
    "snap": "unknown",
    "base_head": "unknown",
    "dirty": False,
    "mujou_snap": "unknown",
    "mujou_pushed": False,
    "envkey": "local",
}
RES = {"class": "cpu", "ram_gb": 2.0, "gpu_mem_gb": 0.0}


class TestPosthoc:
    """The posthoc job of a run of the autoencoder: M17's evaluation of the
    flattened tokens, the autoencoder's own decoder and the mean color."""

    def _job(self, runs: Path, o: list[str], name: str, task: str, **extra) -> Path:
        o = [*o, f"task={task}"]
        job = make_job(
            batch="b",
            config=to_dict(compose(o)),
            overrides=o,
            code=CODE,
            res=RES,
            extra=extra,
        )
        run_dir = runs / "b" / name
        run_dir.mkdir(parents=True)
        (run_dir / "job.json").write_text(json.dumps(job))
        return run_dir

    def test_every_checkpoint_then_resume(self, tmp_path, tours):
        runs = tmp_path / "runs"
        assert _run(runs / "b" / "parent", tours) == Exit.OK
        decode = {"windows": 16, "steps": 3}
        run_dir = self._job(
            runs, tours, "post", "posthoc", parent="b/parent", decode=decode
        )
        assert (
            video_main(["--job", str(run_dir / "job.json"), "--attempt", "1"])
            == Exit.OK
        )
        out = run_dir / "post" / "decode"
        lines = _lines(out / "metrics.jsonl")
        assert [(x["t"], x["split"]) for x in lines] == [
            (0.5, "val"),
            (0.5, "test"),
            (1.0, "val"),
            (1.0, "test"),
        ]
        own = {"stack_mse", "loss", "own_recon", "own_recon_blur"}
        m17 = {"recon", "recon_blur", "dyn_1/copy_1", "rank", "z_spread"}
        assert own | m17 <= set(lines[-1]) and lines[-1]["code"] == "ae"
        assert not any(k.startswith(("restore", "comp")) for k in lines[-1])
        assert all(math.isfinite(lines[-1][k]) for k in own | m17)
        refs = _lines(out / "refs.jsonl")
        assert [(r["code"], r["split"]) for r in refs] == [
            ("tube", "val"),
            ("tube", "test"),
        ]
        final = json.loads((run_dir / "result.json").read_text())["final"]
        assert set(final["baselines"]["val"]) == {"pca", "init"}
        assert (
            final["decode"]["val"]["t"] == 1.0
            and "tube" == final["refs"]["val"]["code"]
        )
        assert (out / "00001.0-own.png").is_file()
        config = json.loads((out / "config.json").read_text())
        assert config["parent"]["group"] and config["ckpts"] == [
            "00000.5.pt",
            "00001.0.pt",
        ]
        verify_manifest(run_dir)
        # a resume after the first checkpoint keeps its lines and redoes the rest
        conf = compose([*tours, "task=posthoc"])
        parent = Parent.read(runs / "b" / "parent", device="cpu")
        again = AEEval(
            parent, conf, out, root=_root(tours), windows=16, steps=3, done=2
        )
        again.step()
        assert _lines(out / "metrics.jsonl") == lines
        assert _lines(out / "refs.jsonl") == refs


def _root(overrides: list[str]) -> Path:
    version = next(o for o in overrides if o.startswith("data.tours.data_version="))
    return Path(os.environ[DATA_ENV]) / version.split("=", 1)[1]


VAE_KEYS = {
    "bpd",
    "loss",
    "kl",
    "rate_bpd",
    "sigma",
    "sigma_star",
    "sigma_gap",
    "stack_mse",
} | {f"kl_q{c}" for c in (0, 10, 25, 50, 75, 90, 100)}


@pytest.fixture
def vae_tours(tmp_path, monkeypatch) -> list[str]:
    root = smooth_tours(tmp_path / "data" / "d", {"A": 120, "B": 80, "H": 200})
    monkeypatch.setenv(DATA_ENV, str(root.parent))
    return tiny_vae(root.name)


class TestVAE:
    @pytest.mark.parametrize("sampling", ["stream", "pool"])
    def test_a_run_writes_the_rows_of_the_vae(self, tmp_path, vae_tours, sampling):
        out = tmp_path / "run"
        extra = [f"data.tours.sampling={sampling}", "data.tours.pool=5"]
        assert _run(out, [*vae_tours, *extra]) == Exit.OK
        rows = _lines(out / "evals.jsonl")
        assert [(r["t"], r["split"]) for r in rows] == [(0.5, "val"), (1.0, "val")]
        pool = {"pool_bpd", "pool_stack_mse"} if sampling == "pool" else set()
        assert set(rows[-1]) == {"t", "frame", "split"} | VAE_KEYS | pool
        assert all(math.isfinite(rows[-1][k]) for k in VAE_KEYS | pool)
        logged = _lines(out / "stream.jsonl")
        assert logged and all(
            r[k] is not None for r in logged for k in ("loss", "bpd", "kl", "log_sigma")
        )
        base = json.loads((out / "baselines.json").read_text())
        assert set(base) == {"val", "test"} and "bpd" in base["val"]
        assert base["val"]["sigma"] == pytest.approx(math.sqrt(0.0743143460753926))
        head = ckpt.load_eval(out / "ckpt" / "00001.0.pt", rebuild(compose(vae_tours)))
        assert head["t"] == 1.0
        result = json.loads((out / "result.json").read_text())
        assert result["objective"] == rows[-1]["bpd"]

    def test_resume_equals_uninterrupted(self, tmp_path, vae_tours):
        """The noise's generator is in the resume checkpoint: the resumed run
        draws what the whole run drew."""
        o = [*vae_tours, "ctl.resume_every=0.2"]
        whole, parts = tmp_path / "whole", tmp_path / "parts"
        assert _run(whole, o) == Exit.OK
        job = local_job(o, parts)
        assert loop.run(_Stopping(13), job, attempt=1) == Exit.STOPPED
        (parts / "stop.request").unlink()
        assert loop.run(PROGRAM, job, attempt=2) == Exit.OK
        for name in ("evals.jsonl", "stream.jsonl"):
            assert (whole / name).read_bytes() == (parts / name).read_bytes(), name

    def test_a_pc_run_writes_the_rows_and_resumes(self, tmp_path, vae_tours):
        """learner=pc (stage 4, ``pc.VAEPC``): the rows of the BP run, and a
        resumed run (the nodes' r, lam, clocks and optimizers in the state)
        writes what the whole run wrote."""
        o = [*vae_tours, "learner=pc", "time.learn_interval=0.029"]
        o = [*o, "ctl.resume_every=0.2"]
        whole, parts = tmp_path / "whole", tmp_path / "parts"
        assert _run(whole, o) == Exit.OK
        rows = _lines(whole / "evals.jsonl")
        assert [(r["t"], r["split"]) for r in rows] == [(0.5, "val"), (1.0, "val")]
        assert set(rows[-1]) == {"t", "frame", "split"} | VAE_KEYS
        assert all(math.isfinite(rows[-1][k]) for k in VAE_KEYS)
        logged = _lines(whole / "stream.jsonl")
        assert logged and all(
            r[k] is not None for r in logged for k in ("loss", "bpd", "kl", "log_sigma")
        )
        head = ckpt.load_eval(whole / "ckpt" / "00001.0.pt", rebuild(compose(o)))
        assert head["t"] == 1.0
        job = local_job(o, parts)
        assert loop.run(_Stopping(13), job, attempt=1) == Exit.STOPPED
        (parts / "stop.request").unlink()
        assert loop.run(PROGRAM, job, attempt=2) == Exit.OK
        for name in ("evals.jsonl", "stream.jsonl"):
            assert (whole / name).read_bytes() == (parts / name).read_bytes(), name

    def test_pc_learns_as_bp_at_the_same_learn_interval(self, tmp_path, vae_tours):
        """The same run by BP and by PC-ALM, both learning at 29 ms (the
        step from which this net's nodes learn by BP's gradient): the same
        stream rows and evaluations up to the rounding, through the second
        of learning (QP3 of 「段 4」, extended)."""
        o = [*vae_tours, "time.learn_interval=0.029"]
        bp, pc_ = tmp_path / "bp", tmp_path / "pc"
        assert _run(bp, o) == Exit.OK
        assert _run(pc_, [*o, "learner=pc"]) == Exit.OK
        for name, keys in (("stream.jsonl", ("loss", "kl")), ("evals.jsonl", ("bpd",))):
            a, b = _lines(bp / name), _lines(pc_ / name)
            assert len(a) == len(b) > 0
            for ra, rb in zip(a, b, strict=True):
                for k in keys:
                    assert rb[k] == pytest.approx(ra[k], rel=1e-6), (name, k)

    def test_posthoc_reads_the_code_mu(self, tmp_path, vae_tours):
        """The posthoc of a VAE's run: M17's evaluation of mu flattened and
        the VAE's own decoder at mu, each line's keys finite (that the ViT
        before learning of unit 0 is the VAE's untrained mu is
        ``test_modules.py``'s)."""
        runs = tmp_path / "runs"
        assert _run(runs / "b" / "parent", vae_tours) == Exit.OK
        run_dir = TestPosthoc()._job(
            runs,
            vae_tours,
            "post",
            "posthoc",
            parent="b/parent",
            decode={"windows": 16, "steps": 3, "ckpts": "last"},
        )
        assert (
            video_main(["--job", str(run_dir / "job.json"), "--attempt", "1"])
            == Exit.OK
        )
        lines = _lines(run_dir / "post" / "decode" / "metrics.jsonl")
        assert [(x["t"], x["split"]) for x in lines] == [(1.0, "val"), (1.0, "test")]
        own = {"stack_mse", "loss", "own_recon", "own_recon_blur"}
        m17 = {"recon", "recon_blur", "dyn_1/copy_1", "rank", "z_spread"}
        assert own | m17 <= set(lines[-1])
        assert all(math.isfinite(lines[-1][k]) for k in own | m17)
