"""The program ``"vae"`` on synthetic stacks (``bench.RandomStacks``): its
first logged loss against the float64 -ELBO of the noise it drew, its
schedules, and a stopped-and-resumed run against one never stopped."""

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from exp.lib import ckpt
from exp.lib.rundir import Exit
from exp.vision import loop
from exp.vision.bench import RandomStacks
from exp.vision.compose import compose, to_dict
from exp.vision.loop import Built
from exp.vision.models import VAEModules, build_vae
from exp.vision.program import NOISE_SEED, Online, VAEProgram, rebuild, schedule_of
from exp.vision.schema.run import RunConf
from tests.vision.test_pc import _oracle

TINY = ["+experiment=v0/tiny"]  # its optim.tau 1e4: a step that stays finite


@pytest.fixture(autouse=True)
def _no_gil_check(monkeypatch):
    """The GIL of this test process is pytest's (tests/vision/test_main.py
    checks it in a run of its own)."""
    monkeypatch.setattr(loop, "assert_free_threaded", lambda where: None)


class _CodeSquare:
    """Measures that read the live modules: the mean square of the code of
    fixed stacks (as ``"stack_mse"``, the objective)."""

    def __init__(self, conf: RunConf) -> None:
        self._x = RandomStacks(conf, seed=99).next()[0]

    def evaluate(self, mods: VAEModules, split: str) -> dict[str, float]:
        with torch.no_grad():
            return {"stack_mse": float(mods.mu(self._x).square().mean())}


class _Stopping:
    """``VAEProgram`` that asks to stop after frame ``stop_at``."""

    def __init__(self, stop_at: int | None) -> None:
        self._inner = VAEProgram(streams=RandomStacks, measures=_CodeSquare)
        self.stop_at = stop_at

    def build(self, conf: RunConf, run_dir: Path, resume: Any) -> Built:
        built = self._inner.build(conf, run_dir, resume)
        assert isinstance(built, Online)
        advance = built.advance

        def stopping() -> None:
            advance()
            if built.host.frames == self.stop_at:
                (run_dir / "stop.request").touch()

        built.advance = stopping  # type: ignore[method-assign]
        return built


def _job(run_dir: Path, overrides: list[str]) -> Path:
    run_dir.mkdir(parents=True)
    job = {
        "run_id": "vae-s0",
        "group": "g",
        "program": "vae",
        "config": to_dict(compose(overrides)),
        "code": {"snap": "x"},
    }
    path = run_dir / "job.json"
    path.write_text(json.dumps(job))
    return path


class TestOnline:
    def test_the_first_loss_is_bps_elbo_of_the_noise_drawn_from_the_seed(self):
        """The first line's loss is the mean -ELBO of the first stacks at the
        initial weights, with u then eps drawn from seed + NOISE_SEED, as
        float64 BP computes it (the run is float32)."""
        conf = compose([*TINY, "data.streams=2", "seed=3"])
        online = Online(conf, None, RandomStacks(conf))
        try:
            online.advance()
            row = online.log()
        finally:
            online.close()
        x = RandomStacks(conf).next()[0]
        g = torch.Generator().manual_seed(conf.seed + NOISE_SEED)
        u = torch.rand(x.shape, generator=g)
        eps = torch.randn(2, conf.model.latent, generator=g)
        double = build_vae(conf, seed=conf.seed)
        for m in double.modules().values():
            m.double()
        loss, _ = _oracle(
            double, x, u.double(), eps.double(), torch.ones(2, dtype=torch.bool)
        )
        assert row["loss"] == pytest.approx(float(loss.detach()), rel=1e-5)

    def test_the_schedules(self):
        """Online: the checkpoints of measure.ckpt_times (0, 0.5, 1 s are
        frames 0, 5, 10), evaluations only with measures; bench: neither."""
        conf = compose(TINY)
        s = schedule_of(conf, evaluate=False)
        assert (s.units, s.seconds, s.log, s.resume) == (10, 0.1, 5, 5)
        assert s.ckpt_at == (0, 5, 10) and s.evaluate == 0 and not s.test
        assert schedule_of(conf, evaluate=True).evaluate == 5
        bench = schedule_of(compose([*TINY, "task=bench"]), evaluate=True)
        assert bench.ckpt_at == () and bench.evaluate == 0

    def test_posthoc_is_not_a_task(self):
        """It is ``python -m exp.vision.eval`` (tests/vision/test_eval.py)."""
        with pytest.raises(ValueError, match="task"):
            compose([*TINY, "task=posthoc"])


class TestResume:
    @pytest.mark.parametrize("stop_at", [3, 5])
    def test_a_stopped_and_resumed_run_is_one_never_stopped_bit_for_bit(
        self, tmp_path, stop_at
    ):
        """On the CPU: stopped after frame ``stop_at`` (between two resume
        checkpoints, on one) and resumed, the run writes the bytes of
        stream.jsonl and evals.jsonl of a run never stopped, and its last
        checkpoint holds the same weights."""
        overrides = [*TINY, "ctl.log_every=0.1"]
        whole = _job(tmp_path / "whole", overrides)
        assert loop.run(_Stopping(None), whole, attempt=1) == Exit.OK
        cut = _job(tmp_path / "cut", overrides)
        assert loop.run(_Stopping(stop_at), cut, attempt=1) == Exit.STOPPED
        (cut.parent / "stop.request").unlink()
        assert loop.run(_Stopping(None), cut, attempt=2) == Exit.OK
        for name in ("stream.jsonl", "evals.jsonl"):
            a = (whole.parent / name).read_bytes()
            assert a and a == (cut.parent / name).read_bytes(), name
        conf = compose(overrides)
        a, b = rebuild(conf), rebuild(conf)
        ckpt.load_eval(ckpt.eval_path(whole.parent, 1.0), a)
        ckpt.load_eval(ckpt.eval_path(cut.parent, 1.0), b)
        start = rebuild(conf)
        for name in a:
            sa, sb = a[name].state_dict(), b[name].state_dict()
            assert all(torch.equal(sa[k], sb[k]) for k in sa)
        moved = start["encoder"].state_dict()
        assert any(
            not torch.equal(moved[k], a["encoder"].state_dict()[k]) for k in moved
        )
        assert ckpt.eval_path(whole.parent, 0.0).exists()  # the initialization
