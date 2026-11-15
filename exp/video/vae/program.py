r"""The program ``"ae"`` (M18): the autoencoder of ``exp.video.vae`` on stacks
of Walking Tours, for the tasks

- ``"online"``: the autoencoder (BP, ``bp.AENet``) learns from the ``B``
  stacks of each frame of the run's source (``ae_streams``); every
  ``measure.every`` seconds it is evaluated on the held-out stacks
  (``measures.AEMeasures``: ``stack_mse`` and ``loss``);
- ``"bench"``: the same without the evaluations, to time it;
- ``"posthoc"`` and ``"baselines"``: M17's evaluation of the code afterwards
  (``exp.video.programs.posthoc``, ``eval.AEEval``).

The source of the stacks is ``data.tours.sampling``: ``"stream"``, the
streams of M17 (``StackedTourFrames``); ``"shuffle"``, the stacks the
streams give over the whole run in an order drawn from the seed ``seed +
SOURCE_SEED`` (``ShuffledStackedTourFrames``: the same stacks, only the order
differs); ``"pool"``, the overfit control, ``data.tours.pool`` stacks drawn
once from the same seed and then drawn with replacement
(``PooledStackedTourFrames``); ``"iid"``, independent uniform stacks
(``IIDStackedTourFrames``).

A frame: the stacks written to the node, and $L$ it sends first for them read
back (of the weights before it learned from them), added to the sums of
``stream.jsonl`` (``loss``: the mean over the frames with a stack that
counts). Frame 0 is not counted, as in the vision program.
"""

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mujou.pc as pc
import mujou.testing
import torch
from mujou.testing import ManualRun
from torch import nn

from exp.lib.host import Host
from exp.lib.rundir import write_json
from exp.video.data.frames import (
    IIDStackedTourFrames,
    PooledStackedTourFrames,
    ShuffledStackedTourFrames,
    StackedTourFrames,
    StackStreams,
)
from exp.video.loop import Built, Schedule, every
from exp.video.measure.objective import PENALTY
from exp.video.programs.vision import tours_root
from exp.video.schema.run import RunConf
from exp.video.vae.bp import build_ae_net
from exp.video.vae.measures import AEMeasures
from exp.video.vae.modules import build_ae

__all__ = ["PROGRAM", "SOURCE_SEED", "AEProgram", "ae_streams", "rebuild"]

SOURCE_SEED = 6  # added to the run's seed: the shuffle's order, the pool's draws


def ae_streams(conf: RunConf) -> StackStreams:
    """The source of the stacks of ``conf`` (module docstring).

    Raises:
        ValueError: The run does not stream Walking Tours, or has no vision.
        EnvError: The data is not here (``tours_root``).
    """
    t, v = conf.data.tours, conf.vision
    if t is None or v is None:
        raise ValueError(f"the run {conf.name!r} needs a vision on Walking Tours")
    root, b, seed = tours_root(t), conf.data.streams, conf.seed + SOURCE_SEED
    common = {"n": b, "held_out": t.held_out, "frames": v.frames, "stride": v.stride}
    match t.sampling:
        case "stream":
            return StackedTourFrames(root, advance=t.advance, **common)
        case "shuffle":
            units = round(conf.seconds / conf.time.frame)
            return ShuffledStackedTourFrames(
                root, advance=t.advance, units=units, seed=seed, **common
            )
        case "pool":
            return PooledStackedTourFrames(root, pool=t.pool, seed=seed, **common)
        case sampling:
            assert sampling == "iid"  # ToursConf checks the choice
            return IIDStackedTourFrames(root, seed=seed, **common)


class _Online:
    """The tasks ``"online"`` and ``"bench"`` (no evaluations)."""

    def __init__(
        self, conf: RunConf, run_dir: Path, resume: Mapping[str, Any] | None
    ) -> None:
        self.conf = conf
        bench = conf.task == "bench"
        self.net = build_ae_net(conf)
        self.streams = ae_streams(conf)
        self.measures: AEMeasures | None = None
        if not bench:
            assert conf.data.tours is not None  # ae_streams checks
            pool = None
            if isinstance(self.streams, PooledStackedTourFrames):
                pool = self.streams.stacks
            self.measures = AEMeasures(conf, tours_root(conf.data.tours), pool=pool)
            base = {
                s: self.measures.evaluate(self.net.mods, s) for s in ("val", "test")
            }
            write_json(
                run_dir / "baselines.json", {s: _clean(r) for s, r in base.items()}
            )
        b = conf.data.streams
        dev = torch.device(conf.device)
        self._y = torch.zeros(b, 1, device=dev)
        # loss and the frames since the last log, in host memory (Host.show)
        self._acc = torch.zeros(2, dtype=torch.float64)
        state = None if resume is None else resume["mujou"]
        self._run: ManualRun = mujou.testing.run(self.net, state=state)
        frame = conf.time.frame
        self.host = Host(self._run, frame=frame, period=self.net.tau.period)
        if resume is not None:
            h = resume["host"]
            self.host.load_state_dict(h["host"])
            self.streams.load_state_dict(h["streams"])
            self._acc.copy_(h["acc"])
        ctl, m = conf.ctl, conf.measure
        self._schedule = Schedule(
            units=round(conf.seconds / frame),
            unit="frame",
            seconds=frame,
            log=every(ctl.log_every, frame),
            evaluate=0 if bench else every(m.every, frame),
            ckpt=every(ctl.ckpt_every, frame),
            viz=0,
            resume=every(ctl.resume_every, frame),
            test=ctl.eval_test and not bench,
        )

    @property
    def schedule(self) -> Schedule:
        return self._schedule

    def advance(self) -> None:
        k = self.host.frames
        stacks, ready, reset = self.streams.next()
        dev = self._y.device
        sample = pc.msg.Sample(
            x=stacks.to(dev), y=self._y, mask=ready.to(dev), reset=reset.to(dev)
        )
        p, _ = self.host.show({self.net.prev: sample}, self.net.stats)
        if k == 0:
            return
        counted = p["n"] > 0
        terms = torch.stack([p["loss"], torch.ones_like(p["n"])])
        self._acc += torch.where(counted, terms.double(), 0.0)

    def log(self) -> dict[str, float | None]:
        loss, n = self._acc.tolist()
        self._acc.zero_()
        return {"loss": loss / n if n else None}

    def evaluate(self, split: str) -> dict[str, float]:
        assert self.measures is not None  # no evaluations in a bench
        return self.measures.evaluate(self.net.mods, split)

    def diverged(self, row: Mapping[str, float]) -> str | None:
        bad = [k for k, x in row.items() if not math.isfinite(x)]
        return f"{', '.join(bad)} not finite" if bad else None

    def objective(
        self, row: Mapping[str, Any] | None, *, diverged: bool
    ) -> float | None:
        if diverged:
            return PENALTY
        return None if row is None else row.get("stack_mse")

    def modules(self) -> dict[str, nn.Module]:
        return self.net.mods.named()

    def visualize(self, out: Path) -> None:
        pass

    def resume_state(self) -> tuple[Mapping[str, Any], dict[str, Any]]:
        host = {
            "host": self.host.state_dict(),
            "streams": self.streams.state_dict(),
            "acc": self._acc.cpu().clone(),
        }
        return self._run.state_dict(), host

    def final(self) -> dict[str, dict[str, Any]]:
        return {}

    def close(self) -> None:
        self._run.shutdown()


def _clean(d: Mapping[str, float]) -> dict[str, float | None]:
    return {k: (x if math.isfinite(x) else None) for k, x in d.items()}


class AEProgram:
    """The program ``"ae"`` (module docstring), a ``loop.Program``."""

    def build(
        self, conf: RunConf, run_dir: Path, resume: Mapping[str, Any] | None
    ) -> Built:
        match conf.task:
            case "online" | "bench":
                return _Online(conf, run_dir, resume)
            case "posthoc" | "baselines":  # imported here: it imports the programs
                from exp.video.programs.posthoc import Posthoc

                return Posthoc(conf, run_dir, resume)
            case task:
                raise ValueError(f"the ae program has no task {task!r}")


def rebuild(conf: RunConf) -> dict[str, nn.Module]:
    """Fresh modules of ``conf`` by the names of the evaluation checkpoint, to
    load one into (``exp.lib.ckpt.load_eval``)."""
    return build_ae(conf, seed=conf.seed).named()


PROGRAM = AEProgram()
