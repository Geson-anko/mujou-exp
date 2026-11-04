r"""The program of M14: the dynamics on the states of the balls (``exp.video.dyn``),
for the tasks

- ``"online"``: the world model (PC-ALM or BP, ``conf.learner``) learns from
  ``B`` streams frame by frame; every ``measure.every`` seconds it is
  evaluated on the validation clips and from the live streams
  (``measure.dynamics``);
- ``"bench"``: the same without the evaluations, to time it;
- ``"baselines"``: only the baselines of the clips (``baselines.json`` and
  ``result.json``), nothing learns;
- ``"offline"``: the bound of the same modules fit by Adam on windows of the
  streams (``dyn.offline``), evaluated as the online run.

Every task writes ``baselines.json``: the standardization (fitted on the
calibration clips) and the baselines of the validation and test clips. The
clips have ``measure.windows`` windows each, ``STRIDE`` frames apart.

A frame of the online task: the next state of each stream, observed and
standardized, written as a ``pc.msg.Sample`` (``reset`` on the first frame
only: the streams never draw new balls); the prequential prediction read back
(``Host.show``); its squared error added on the device to the sums of the
predictions from the observation (``preq``) and from the model's own
prediction (``preq_open``, the frames ``nodes.is_open`` says), read once per
line of ``stream.jsonl``.
"""

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mujou.pc as pc
import mujou.testing
import torch
from mujou.testing import ManualRun
from torch import Tensor, nn

from exp.lib.host import Host
from exp.lib.rundir import write_json
from exp.video.data.clips import (
    STENCIL,
    ClipSet,
    contact_mask,
    fit_mean_image,
    fit_standardizer,
    make_clips,
    window_states,
)
from exp.video.data.observe import PIXELS, Standardizer, observe
from exp.video.data.physics import BallStreams
from exp.video.dyn.modules import DynamicsModules, build_dynamics
from exp.video.dyn.nets import build_world, live_states, modules
from exp.video.dyn.nodes import is_open
from exp.video.dyn.offline import OfflineFit, pixel_encoder, stream_windows
from exp.video.loop import Built, Schedule, every
from exp.video.measure.dynamics import (
    dynamics_baselines,
    evaluate_dynamics,
    fork_eval,
)
from exp.video.measure.objective import PENALTY, objective
from exp.video.measure.viz import pick_windows, visualize
from exp.video.schema.data import BallsConf
from exp.video.schema.run import RunConf

__all__ = ["PROGRAM", "STRIDE", "DynamicsProgram", "Measures"]

STRIDE = 30  # frames between two windows of a clip
_SPLITS = ("val", "test")


def _clean(d: Mapping[str, float]) -> dict[str, float | None]:
    return {k: (v if math.isfinite(v) else None) for k, v in d.items()}


class Measures:
    """What every task measures with: the standardization, the clips of each
    split and their baselines, and the windows to draw.

    Args:
        conf: The run; its balls, ``measure`` and frame.
    """

    def __init__(self, conf: RunConf) -> None:
        balls = conf.data.balls
        if balls is None:
            raise ValueError(f"the run {conf.name!r} has no balls")
        m = conf.measure
        if m.observed < STENCIL:
            raise ValueError(f"observed={m.observed} is fewer than STENCIL={STENCIL}")
        self.balls: BallsConf = balls
        self.m = m
        self.k = len(balls.radii)
        self.frame = conf.time.frame
        self.std: Standardizer = fit_standardizer(balls, frame=self.frame)
        pixels = balls.obs == PIXELS
        self.reference = "ca" if pixels else "cv"  # the bar of the objective
        mean_image = fit_mean_image(balls, frame=self.frame) if pixels else None
        horizon = max(m.horizons)
        length = m.observed + horizon + STRIDE * (m.windows - 1)
        self.clips: dict[str, ClipSet] = {
            split: make_clips(
                balls,
                split=split,
                clips=m.clips,
                length=length,
                observed=m.observed,
                horizon=horizon,
                stride=STRIDE,
                frame=self.frame,
            )
            for split in _SPLITS
        }
        self.base = {
            s: dynamics_baselines(c, balls, self.std, m, mean_image=mean_image)
            for s, c in self.clips.items()
        }
        h = 10 if 10 in m.horizons else horizon
        self.key = h  # the horizon of the divergence and the pictures
        val = self.clips["val"]
        contact = contact_mask(
            window_states(val), m.observed, [h], radii=balls.radii, frame=self.frame
        )
        self.picked = pick_windows(contact[:, 0].any(-1))
        self.width = self.observe(val.states[:1, 0]).shape[-1]

    def observe(self, states: Tensor) -> Tensor:
        """The raw observation of ``states`` ``(..., 4K)``."""
        return observe(
            states, self.balls.obs, self.k, radii=self.balls.radii, size=self.balls.size
        )

    def write(self, run_dir: Path) -> None:
        """``baselines.json``: the standardization and the baselines of each
        split."""
        write_json(
            run_dir / "baselines.json",
            {
                "obs": self.balls.obs,
                "standardizer": {
                    "mean": self.std.mean.tolist(),
                    "std": self.std.std.tolist(),
                },
                **{s: _clean(b) for s, b in self.base.items()},
            },
        )

    def evaluate(
        self, mods: DynamicsModules, split: str, *, residual: bool
    ) -> dict[str, float]:
        return evaluate_dynamics(
            mods,
            self.clips[split],
            self.std,
            self.base[split],
            self.m,
            obs=self.balls.obs,
            balls=self.k,
            residual=residual,
            size=self.balls.size,
        )

    def diverged(self, row: Mapping[str, float]) -> str | None:
        """``roll{h}`` (``h`` 10, or the largest horizon) is not finite or
        above ``measure.diverge`` times ``mean{h}``, or a prediction is not
        finite."""
        h = self.key
        roll, mean = row[f"roll{h}"], row[f"mean{h}"]
        if row["nonfinite"]:
            return f"{row['nonfinite']:g} windows predicted values that are not finite"
        if not math.isfinite(roll) or roll > self.m.diverge * mean:
            return f"roll{h} {roll:.4g} > {self.m.diverge:g} x mean{h} {mean:.4g}"
        return None

    def visualize(self, out: Path, mods: DynamicsModules, *, residual: bool) -> None:
        visualize(
            out,
            mods,
            self.clips["val"],
            self.picked,
            self.std,
            self.m,
            obs=self.balls.obs,
            balls=self.k,
            residual=residual,
            size=self.balls.size,
        )


class _Evaluated:
    """What the tasks that learn share: the evaluation of their modules."""

    conf: RunConf
    measures: Measures
    mods: DynamicsModules

    def _row(self, split: str) -> dict[str, float]:
        residual = self.conf.dynamics is not None and self.conf.dynamics.residual
        return self.measures.evaluate(self.mods, split, residual=residual)

    def diverged(self, row: Mapping[str, float]) -> str | None:
        return self.measures.diverged(row)

    def objective(
        self, row: Mapping[str, Any] | None, *, diverged: bool
    ) -> float | None:
        if diverged:
            return PENALTY
        return (
            None if row is None else objective(row, reference=self.measures.reference)
        )

    def modules(self) -> dict[str, nn.Module]:
        return {"step": self.mods.step, "readout": self.mods.readout}

    def visualize(self, out: Path) -> None:
        residual = self.conf.dynamics is not None and self.conf.dynamics.residual
        self.measures.visualize(out, self.mods, residual=residual)

    def final(self) -> dict[str, dict[str, Any]]:
        return {}


class _Online(_Evaluated):
    """The tasks ``"online"`` and ``"bench"`` (no evaluations)."""

    def __init__(
        self, conf: RunConf, measures: Measures, resume: Mapping[str, Any] | None
    ) -> None:
        self.conf, self.measures = conf, measures
        balls, m = measures.balls, measures.m
        dyn = conf.dynamics
        if dyn is None:
            raise ValueError(f"the run {conf.name!r} has no dynamics")
        self._dyn = dyn
        frame, b = conf.time.frame, conf.data.streams
        self.net = build_world(conf, measures.width)
        self.mods = modules(self.net)
        self.streams = BallStreams.of(balls, b, frame=frame, seed=conf.seed)
        dev = torch.device(conf.device)
        std = measures.std
        self._mean, self._scale = std.mean.float(), std.std.float()
        self._y = torch.zeros(b, 1, device=dev)
        self._first = torch.ones(b, dtype=torch.bool, device=dev)
        self._later = torch.zeros(b, dtype=torch.bool, device=dev)
        # The sums since the last log: error and count from the observation,
        # then from the model's own prediction. On the CPU, where the host
        # reads the predictions back (Host.show).
        self._acc = torch.zeros(4, dtype=torch.float64)
        # The last frames, raw, on the CPU: the observations, but for pixels
        # the states (the extrapolation baselines move them and render).
        pixels = balls.obs == PIXELS
        self._history = torch.zeros(
            b, STENCIL, 4 * measures.k if pixels else measures.width
        )
        state = None if resume is None else resume["mujou"]
        self._run: ManualRun = mujou.testing.run(self.net, state=state)
        self.host = Host(self._run, frame=frame, period=self.net.tau.period)
        if resume is not None:
            h = resume["host"]
            self.host.load_state_dict(h["host"])
            self.streams.load_state_dict(h["streams"])
            self._acc.copy_(h["acc"])
            self._history.copy_(h["history"])
        bench = conf.task == "bench"
        ctl = conf.ctl
        self._schedule = Schedule(
            units=round(conf.seconds / frame),
            unit="frame",
            seconds=frame,
            log=every(ctl.log_every, frame),
            evaluate=0 if bench else every(m.every, frame),
            ckpt=every(ctl.ckpt_every, frame),
            viz=0 if bench else every(ctl.viz_every, frame),
            resume=every(ctl.resume_every, frame),
            test=ctl.eval_test and not bench,
        )

    @property
    def schedule(self) -> Schedule:
        return self._schedule

    def advance(self) -> None:
        k = self.host.frames
        states = self.streams.next()
        raw = self.measures.observe(states)
        kept = states if self.measures.balls.obs == PIXELS else raw
        self._history = torch.cat([self._history[:, 1:], kept[:, None]], 1)
        x = ((raw - self._mean) / self._scale).to(self._y.device)  # Standardizer.apply
        reset = self._first if k == 0 else self._later
        sample = pc.msg.Sample(x=x, y=self._y, mask=~reset, reset=reset)
        p, _ = self.host.show({self.net.prev: sample}, self.net.next)
        if k == 0:
            return
        err = ((p["y_hat"] - p["y"]) ** 2).mean(-1)
        mask = p["mask"]
        opened = k > 1 and is_open(k, self._dyn.open_loop, self._dyn.open_cycle)
        i = 2 if opened else 0
        sums = torch.stack(
            [torch.where(mask, err, 0.0).sum().double(), mask.sum().double()]
        )
        self._acc[i : i + 2] += sums

    def log(self) -> dict[str, float | None]:
        e, n, e_open, n_open = self._acc.tolist()
        self._acc.zero_()
        return {
            "preq": e / n if n else None,
            "preq_open": e_open / n_open if n_open else None,
        }

    def evaluate(self, split: str) -> dict[str, float]:
        row = self._row(split)
        if split == "val":
            ms = self.measures
            row |= fork_eval(
                self.mods,
                live_states(self.net),
                self.streams,
                self._history,
                ms.std,
                ms.m,
                obs=ms.balls.obs,
                balls=ms.k,
                residual=self._dyn.residual,
                frame=ms.frame,
                radii=ms.balls.radii,
                size=ms.balls.size,
            )
            row["objective"] = objective(row, reference=self.measures.reference)
        return row

    def resume_state(self) -> tuple[Mapping[str, Any], dict[str, Any]]:
        host = {
            "host": self.host.state_dict(),
            "streams": self.streams.state_dict(),
            "acc": self._acc.cpu().clone(),
            "history": self._history.clone(),
        }
        return self._run.state_dict(), host

    def close(self) -> None:
        self._run.shutdown()


class _Offline(_Evaluated):
    """The task ``"offline"``: ``OfflineFit`` of the modules on the windows of
    ``offline.frames`` frames of each stream, evaluated every
    ``offline.every`` steps."""

    def __init__(
        self, conf: RunConf, measures: Measures, resume: Mapping[str, Any] | None
    ) -> None:
        self.conf, self.measures = conf, measures
        dyn, off = conf.dynamics, conf.offline
        if dyn is None or off is None:
            raise ValueError(f"the run {conf.name!r} needs dynamics and offline")
        reach = measures.m.observed + max(measures.m.horizons)
        if off.window < reach:  # a horizon the fit would never predict
            raise ValueError(
                f"offline.window={off.window} is shorter than observed + the "
                f"largest horizon = {reach}"
            )
        d = measures.width
        self.mods = build_dynamics(dyn, d, d, seed=conf.seed, device=conf.device)
        pixels = measures.balls.obs == PIXELS
        encode = pixel_encoder(conf, measures.std) if pixels else None
        # pixels are rendered per batch: all of them would be tens of GB
        windows = stream_windows(
            conf, off.frames, off.window, std=measures.std, states=pixels
        )
        self._fit = OfflineFit(
            self.mods,
            windows,
            batch=off.batch,
            lr=off.lr,
            local=off.local,
            observed=measures.m.observed,
            residual=dyn.residual,
            gen=torch.Generator().manual_seed(conf.seed),
            encode=encode,
        )
        self._acc = torch.zeros(2, dtype=torch.float64, device=conf.device)
        if resume is not None:
            h = resume["host"]
            for name, module in self.modules().items():
                module.load_state_dict(h["modules"][name])
            self._fit.load_state_dict(h["fit"])
            self._acc.copy_(h["acc"])
        ctl = conf.ctl
        self._schedule = Schedule(
            units=off.steps,
            unit="step",
            seconds=None,
            log=max(1, off.every // 10),
            evaluate=off.every,
            ckpt=off.every if ctl.ckpt_every > 0 else 0,
            viz=off.every if ctl.viz_every > 0 else 0,
            resume=off.every,
            test=ctl.eval_test,
        )

    @property
    def schedule(self) -> Schedule:
        return self._schedule

    def advance(self) -> None:
        loss = self._fit.step()
        self._acc += torch.stack(
            [loss.double(), loss.new_ones((), dtype=torch.float64)]
        )

    def log(self) -> dict[str, float | None]:
        total, n = self._acc.tolist()
        self._acc.zero_()
        return {"loss": total / n if n else None}

    def evaluate(self, split: str) -> dict[str, float]:
        row = self._row(split)
        if split == "val":
            row["objective"] = objective(row, reference=self.measures.reference)
        return row

    def resume_state(self) -> tuple[Mapping[str, Any], dict[str, Any]]:
        host = {
            "modules": {
                name: {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
                for name, m in self.modules().items()
            },
            "fit": self._fit.state_dict(),
            "acc": self._acc.cpu().clone(),
        }
        return {}, host

    def close(self) -> None:
        pass


class _Baselines:
    """The task ``"baselines"``: nothing runs; ``result.json``'s ``final`` is
    the baselines of each split."""

    def __init__(self, measures: Measures) -> None:
        self._measures = measures
        self._schedule = Schedule(units=0)

    @property
    def schedule(self) -> Schedule:
        return self._schedule

    def advance(self) -> None:
        raise RuntimeError("the baselines task has no units")

    def log(self) -> dict[str, float | None]:
        return {}

    def evaluate(self, split: str) -> dict[str, float]:
        return dict(self._measures.base[split])

    def diverged(self, row: Mapping[str, float]) -> str | None:
        return None

    def objective(
        self, row: Mapping[str, Any] | None, *, diverged: bool
    ) -> float | None:
        return None

    def modules(self) -> dict[str, nn.Module]:
        return {}

    def visualize(self, out: Path) -> None:
        pass

    def resume_state(self) -> tuple[Mapping[str, Any], dict[str, Any]]:
        return {}, {}

    def final(self) -> dict[str, dict[str, Any]]:
        return {s: _clean(b) for s, b in self._measures.base.items()}

    def close(self) -> None:
        pass


class DynamicsProgram:
    """The program ``"dynamics"`` (module docstring), a ``loop.Program``."""

    def build(
        self, conf: RunConf, run_dir: Path, resume: Mapping[str, Any] | None
    ) -> Built:
        measures = Measures(conf)
        measures.write(run_dir)
        match conf.task:
            case "online" | "bench":
                return _Online(conf, measures, resume)
            case "offline":
                return _Offline(conf, measures, resume)
            case "baselines":
                return _Baselines(measures)
            case task:
                raise ValueError(f"the dynamics program has no task {task!r}")


PROGRAM = DynamicsProgram()
