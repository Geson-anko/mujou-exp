r"""The program ``"vae"`` (V0, ``plans/v0-foundation.md``): the VAE of
``exp.vision.models`` learned by PC-ALM alone (``exp.vision.pc``) from the
Walking Tours stream, for the tasks

- ``"online"``: the run of ``measure.duration`` seconds of virtual time, a
  frame of ``time.frame`` a unit; evaluations every ``measure.every``
  seconds when the program has measures (``VAEProgram(measures=...)``), the
  evaluation checkpoints at ``measure.ckpt_times`` (0, the initialization,
  included);
- ``"bench"``: the same without evaluations or checkpoints, to time it
  (``exp.vision.bench`` times every arch and size).

The evaluation of the checkpoints afterwards is ``python -m exp.vision.eval
<run_dir>`` (``exp.vision.eval.evaluate_run``), not a task.

A frame: the host (``exp.lib.host.Host``, on ``mujou.run_manually``) writes
the stacks of the stream (``StackStreams``: ``B`` = ``data.streams``, with
the stream's ``reset``), with the noise of the frame, the dequantization's
$u$ ``(B, n, 3, H, W)`` and the code's $\varepsilon$ ``(B, D)``, drawn here
from one generator on the device seeded with ``seed + NOISE_SEED`` (kept in
the resume checkpoint), and steps the run through the frame by
``tau.period``. It reads back the first $-\log p(y \mid z)$, squared error
and $\ln \sigma$ of ``out`` and KL of ``latent`` made from this write (of the
weights before they learned from it: prequential), added to the sums of
``stream.jsonl`` over the frames with a stack that counts. With
``measure.grad_stats`` the nodes' gradient statistics go to ``grad.jsonl``
(``exp.lib.grad_stats``), collected after each frame.
"""

import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import nn

import mujou
from exp.lib.elbo import bits_per_dim
from exp.lib.host import Host
from exp.lib.tours.stacks import StackedTourFrames, StackStreams, tours_root
from exp.vision.eval import check_splits
from exp.vision.loop import Built, Schedule, every
from exp.vision.models import VAEModules, build_vae
from exp.vision.pc import VAEPC, build_net
from exp.vision.schema.run import RunConf

__all__ = [
    "NOISE_SEED",
    "PROGRAM",
    "Measures",
    "Online",
    "VAEProgram",
    "rebuild",
    "schedule_of",
    "tour_streams",
]

NOISE_SEED = 13  # added to the run's seed: the noise of the stream


class Measures(Protocol):
    """The evaluation of the live modules on the held-out stacks."""

    def evaluate(self, mods: VAEModules, split: str) -> dict[str, float]:
        """The metrics of ``split`` (``"val"`` or ``"test"``); those of
        ``"val"`` may carry ``"stack_mse"``, the run's objective."""
        ...


def tour_streams(conf: RunConf) -> StackStreams:
    """The stream of ``conf``: ``data.streams`` streams of stacks of Walking
    Tours (``data.tours``) without its held-out video. For ``"online"`` it
    first checks that the held-out video has the windows of the posthoc
    evaluation (``eval.check_splits``), so that a run that could not be
    evaluated fails before it runs, not after.

    Raises:
        EnvError: The data is not here (``tours_root``).
        ValueError: The held-out video is too short (``check_splits``).
    """
    t = conf.data.tours
    root = tours_root(t.data_version)
    if conf.task == "online":
        check_splits(conf, root)
    return StackedTourFrames(
        root,
        n=conf.data.streams,
        held_out=t.held_out,
        frames=t.frames,
        stride=t.stride,
        advance=t.advance,
    )


def schedule_of(conf: RunConf, *, evaluate: bool) -> Schedule:
    """The loop's schedule of ``conf``: frames of ``time.frame`` over
    ``measure.duration``; for ``"online"`` the evaluations (with
    ``evaluate``) and the checkpoints of ``measure.ckpt_times``, none of
    them for ``"bench"``."""
    frame, ctl, m = conf.time.frame, conf.ctl, conf.measure
    online = conf.task == "online"
    return Schedule(
        units=round(m.duration / frame),
        unit="frame",
        seconds=frame,
        log=every(ctl.log_every, frame),
        evaluate=every(m.every, frame) if online and evaluate else 0,
        ckpt_at=tuple(round(t / frame) for t in m.ckpt_times) if online else (),
        viz=0,
        resume=every(ctl.resume_every, frame),
        test=ctl.eval_test and online and evaluate,
    )


class Online:
    """The tasks ``"online"`` and ``"bench"`` (module docstring), a
    ``loop.Built``.

    Args:
        conf: The run.
        resume: The resume checkpoint's tree, or ``None`` for a fresh start.
        streams: The stream of stacks (``tour_streams`` of ``conf``).
        measures: The evaluation, or ``None`` for none.
    """

    def __init__(
        self,
        conf: RunConf,
        resume: Mapping[str, Any] | None,
        streams: StackStreams,
        measures: Measures | None = None,
    ) -> None:
        self.conf = conf
        self.net: VAEPC = build_net(conf)
        self.streams = streams
        self.measures = measures
        dev = torch.device(conf.device)
        self._noise = torch.Generator(device=dev).manual_seed(conf.seed + NOISE_SEED)
        # nll, kl, sq and the frames counted since the last log, on the device
        self._acc = torch.zeros(4, dtype=torch.float64, device=dev)
        self._log_sigma: torch.Tensor | None = None  # of the last frame
        state = None if resume is None else resume["mujou"]
        self._run = mujou.run_manually(self.net, state=state)
        self.host = Host(self._run, frame=conf.time.frame, period=self.net.tau.period)
        if resume is not None:
            h = resume["host"]
            self.host.load_state_dict(h["host"])
            self.streams.load_state_dict(h["streams"])
            self._acc.copy_(h["acc"])
            self._noise.set_state(h["noise"])
            ls = h["log_sigma"]
            self._log_sigma = None if ls is None else ls.to(dev)
        self._schedule = schedule_of(conf, evaluate=measures is not None)
        t = conf.data.tours
        self._values = t.frames * 3 * t.size * t.size  # P of a stack

    @property
    def schedule(self) -> Schedule:
        return self._schedule

    def advance(self) -> None:
        stacks, ready, reset = self.streams.next()
        dev = self._acc.device
        x = stacks.to(dev)
        u = torch.rand(x.shape, generator=self._noise, device=dev)
        eps = torch.randn(
            len(x), self.net.mods.latent, generator=self._noise, device=dev
        )
        net = self.net
        sample = net.sample(x, u, eps, ready.to(dev), reset.to(dev))
        (out, latent), _ = self.host.show_all({net.prev: sample}, [net.stats, net.kl])
        counted = (out["n"] > 0).to(dev)
        terms = torch.stack(
            [out["nll"], latent["kl"], out["sq"], torch.ones_like(out["n"])]
        ).to(dev)
        self._acc += torch.where(counted, terms.double(), 0.0)
        self._log_sigma = out["log_sigma"].to(dev)

    def log(self) -> dict[str, float | None]:
        nll, kl, sq, n = self._acc.tolist()
        self._acc.zero_()
        if not n:
            return dict.fromkeys(["loss", "bpd", "kl", "sq", "log_sigma"])
        loss = (nll + kl) / n
        return {
            "loss": loss,
            "bpd": float(bits_per_dim(loss, self._values)),
            "kl": kl / n,
            "sq": sq / n,
            "log_sigma": None if self._log_sigma is None else float(self._log_sigma),
        }

    def rows(self) -> dict[str, list[dict[str, Any]]]:
        rows = self.net.grad_rows()
        return {"grad.jsonl": rows} if rows else {}

    def evaluate(self, split: str) -> dict[str, float]:
        assert self.measures is not None  # the schedule has no evaluations
        return self.measures.evaluate(self.net.mods, split)

    def diverged(self, row: Mapping[str, float]) -> str | None:
        bad = [k for k, x in row.items() if not math.isfinite(x)]
        return f"{', '.join(bad)} not finite" if bad else None

    def objective(
        self, row: Mapping[str, Any] | None, *, diverged: bool
    ) -> float | None:
        if diverged or row is None:
            return None
        return row.get("stack_mse")

    def modules(self) -> dict[str, nn.Module]:
        return self.net.mods.modules()

    def visualize(self, out: Path) -> None:
        pass

    def resume_state(self) -> tuple[Mapping[str, Any], dict[str, Any]]:
        ls = self._log_sigma
        host = {
            "host": self.host.state_dict(),
            "streams": self.streams.state_dict(),
            "acc": self._acc.cpu().clone(),
            "noise": self._noise.get_state(),
            "log_sigma": None if ls is None else ls.cpu().clone(),
        }
        return self._run.state_dict(), host

    def final(self) -> dict[str, dict[str, Any]]:
        return {}

    def close(self) -> None:
        self._run.shutdown()


class VAEProgram:
    """The program ``"vae"`` (module docstring), a ``loop.Program``.

    Args:
        streams: The stream of a run's config (``tour_streams``; a test gives
            synthetic stacks).
        measures: The evaluation of a run's config, or ``None`` for none.
    """

    def __init__(
        self,
        streams: Callable[[RunConf], StackStreams] = tour_streams,
        measures: Callable[[RunConf], Measures] | None = None,
    ) -> None:
        self._streams = streams
        self._measures = measures

    def build(
        self, conf: RunConf, run_dir: Path, resume: Mapping[str, Any] | None
    ) -> Built:
        measures = None  # "online" or "bench" (RunConf checks the task)
        if conf.task == "online" and self._measures is not None:
            measures = self._measures(conf)
        return Online(conf, resume, self._streams(conf), measures)


def rebuild(conf: RunConf) -> dict[str, nn.Module]:
    """Fresh modules of ``conf`` by the names of the evaluation checkpoint, to
    load one into (``exp.lib.ckpt.load_eval``)."""
    return build_vae(conf, seed=conf.seed).modules()


PROGRAM = VAEProgram()
