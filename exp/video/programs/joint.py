r"""The program of M16: the vision of M15 and the dynamics of M14 learned
together, for the tasks

- ``"online"`` (and ``"bench"``, without the evaluations): the encoder learns
  by the vision's loss alone (``exp.video.jepa``: JEPA, and with
  ``vision.recon`` > 0 the reconstruction of the frame) and the dynamics
  (``exp.video.dyn``)
  learns to predict the encoder's next $z$ from the past $z$, both frame by
  frame from ``B`` streams of balls (PC-ALM or BP, ``conf.learner``). Every
  ``measure.every`` seconds both are evaluated on fixed windows of the balls
  (``measure.joint``);
- ``"posthoc"``: the decoder learned afterwards of each evaluation checkpoint,
  which also decodes the dynamics' rollouts into images
  (``exp.video.programs.posthoc``, ``measure.joint.image_rows``).

The dynamics' prediction error sends no credit to the encoder: $z$ is a
constant for the dynamics, as input and as target. In PC-ALM the dynamics is
a ``DynamicsBlock`` on the head's ``z_out``, a port that only sends; in BP the
world reads the vision's ``z_out`` (detached). So the encoder learns what it
learns alone: with the same seed and data its weights are those of the vision
program (tested). The dynamics' weights start from ``conf.seed + 1`` and the
vision's from ``conf.seed``.

The control (``vision.frozen``): the encoder is that of the last evaluation
checkpoint of an M15 run, the job's top-level ``"parent"`` (as the posthoc's:
``"<batch>/<run_id>"`` relative to the runs' root, or an absolute path; on
this host), and never learns. ``fleet submit ... --set parent=<batch>/<run_id>
--after <run_id> --host <host>``, one job per parent: ``parent_ckpt`` requires
the parent to be a finished vision run of the same ``vision`` and
``optim_vision`` settings, seed, learner and balls (a mismatch raises), so
``--seeds`` needs a parent per seed.
The checkpoint read is written to ``frozen_from.json``. In the nodes the freeze
is ``learn_interval = inf``.

A frame: as the vision program's (the images written as a ``pc.msg.Sample``,
the JEPA terms read back), and the dynamics' prequential prediction of $z$
read back from the same frame: its squared error added on the device to the
sums of ``stream.jsonl`` (``preq`` from the past $z$, ``preq_open`` from the
model's own prediction, the frames ``nodes.is_open`` says). The frames before
the first full chunk ($N > 1$) are not marked as such to the dynamics: it
takes the $z$ of the encoder for the zero-padded chunk then (the balls'
streams never reset, so it happens once).

Known difference between PC and BP once the encoder learns: the dynamics' past
$z$ and its target are those the encoder sent at different points of the frame
(PC: the past $z$ is the last one of the previous frame, the target the one
before the encoder's step of this frame; BP: the first one of the previous
frame, the target the one after the encoder's step). The shift is the
encoder's change within one frame, small at the learning rate of the presets.

The row of an evaluation has the vision's (``programs.vision.Measures``: the
JEPA loss, the rank of $z$, the probes of the balls ``probe_pos_min`` ...) and
the latent errors of ``measure.joint.latent_errors`` (``roll{h}``, ``copy{h}``,
``mean{h}``). The objective of a val row is ``roll{h} / mean{h}`` at ``h`` 10
(or the largest horizon), at most ``PENALTY`` (as ``vision_objective`` when $z$
collapsed). The judgement in images is the posthoc's.
"""

import dataclasses
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mujou.pc as pc
import mujou.testing
import torch
from mujou.pc.msg import Activity
from mujou.testing import ManualRun
from torch import nn

from exp.lib import ckpt
from exp.lib.host import Host
from exp.lib.rundir import EnvError, write_json
from exp.video.compose import from_dict
from exp.video.data.frames import BallFrames
from exp.video.dyn.bp import DynamicsBP
from exp.video.dyn.modules import DynamicsModules
from exp.video.dyn.nets import make_optimizer
from exp.video.dyn.nodes import DynamicsBlock, is_open
from exp.video.jepa.nets import VisionBP, VisionPC
from exp.video.jepa.nodes import z_spec
from exp.video.jepa.vit import VisionModules, build_vision
from exp.video.loop import Built, Schedule, every
from exp.video.measure.joint import Windows, ball_windows, joint_dynamics, latent_row
from exp.video.measure.objective import PENALTY
from exp.video.programs.posthoc import parent_dir
from exp.video.programs.vision import COLLAPSE, STRIDE, Measures
from exp.video.schema.run import RunConf
from mujou import DictSpec

__all__ = [
    "PROGRAM",
    "JointBP",
    "JointPC",
    "JointProgram",
    "build_joint",
    "joint_objective",
    "parent_ckpt",
    "rebuild",
]


def _vision_conf(conf: RunConf) -> RunConf:
    """``conf`` for the vision's nodes: with a frozen encoder they never learn
    (``learn_interval`` inf)."""
    assert conf.vision is not None
    if not conf.vision.frozen:
        return conf
    time = dataclasses.replace(conf.time, learn_interval=math.inf)
    return dataclasses.replace(conf, time=time)


def _upstream(conf: RunConf) -> DictSpec[Activity]:
    """The spec of what the vision sends to the dynamics on ``z_out``."""
    assert conf.vision is not None
    return z_spec(conf.data.streams, conf.vision.dim)


class JointPC(VisionPC):
    """``VisionPC`` and, on the head's ``z_out``, the ``DynamicsBlock`` of
    ``dyn``: the vision's ports (``prev``, ``stats``, ``z_out``) and the
    block's ``next`` (the prequential prediction of $z$) and ``error``.

    Args:
        mods: The vision's modules, on ``conf.device``.
        dyn: The dynamics' modules.
        conf: The run: its vision, dynamics, time, streams, seed and device.
            With ``vision.frozen`` the vision's nodes never learn.
    """

    def __init__(
        self, mods: VisionModules, dyn: DynamicsModules, *, conf: RunConf
    ) -> None:
        super().__init__(mods, conf=_vision_conf(conf))
        assert conf.dynamics is not None
        self.dyn = dyn
        self.block = DynamicsBlock(
            dyn,
            act_spec=_upstream(conf),
            conf=conf.dynamics,
            tau=self.tau,
            optimizer=make_optimizer(conf.optim, conf.time.frame),
            learn_interval=conf.time.learn_interval,
            device=conf.device,
        )
        self.block.connect(self.z_out)  # the port only sends: no credit back
        self.next, self.error = self.block.next, self.block.error


class JointBP(VisionBP):
    """``VisionBP`` and, reading its ``z_out`` (detached), the ``DynamicsBP`` of
    ``dyn``, with the ports of ``JointPC``."""

    def __init__(
        self, mods: VisionModules, dyn: DynamicsModules, *, conf: RunConf
    ) -> None:
        super().__init__(mods, conf=_vision_conf(conf))
        assert conf.dynamics is not None
        self.dyn = dyn
        self.world = DynamicsBP(
            dyn,
            act_spec=_upstream(conf),
            conf=conf.dynamics,
            tau=self.tau,
            optimizer=make_optimizer(conf.optim, conf.time.frame),
            learn_interval=conf.time.learn_interval,
            device=conf.device,
        )
        self.z_out.connect(self.world.prev)
        self.next, self.error = self.world.next, self.world.error


def build_joint(conf: RunConf, *, frozen_from: Path | None = None) -> JointPC | JointBP:
    """The joint network of ``conf`` in the nodes of ``conf.learner`` (``"pc"``
    or ``"bp"``): the vision of ``build_vision`` with seed ``conf.seed`` (the
    encoder's initial weights are those of the vision program of the same
    seed) and the dynamics of ``measure.joint.joint_dynamics`` (seed ``+ 1``).
    With ``vision.frozen``, the encoder and the predictor are loaded from the
    evaluation checkpoint ``frozen_from`` of an M15 run.

    Raises:
        ValueError: ``conf`` has no vision, dynamics or ``optim_vision``, its
            learner is ``"none"``, ``vision.frozen`` has no ``frozen_from``,
            or ``frozen_from`` is not the checkpoint of a vision run.
    """
    v = conf.vision
    if v is None or conf.dynamics is None or conf.optim_vision is None:
        raise ValueError(
            f"the run {conf.name!r} needs vision, optim_vision and dynamics"
        )
    kinds = {"pc": JointPC, "bp": JointBP}
    if conf.learner.kind not in kinds:
        raise ValueError(f"learner {conf.learner.kind!r} learns nothing; 'pc' or 'bp'")
    if v.frozen and frozen_from is None:
        raise ValueError("vision.frozen needs the checkpoint to load the encoder from")
    mods = build_vision(conf, seed=conf.seed)
    if v.frozen:
        assert frozen_from is not None
        head = ckpt.load_eval(frozen_from, mods.named())
        if head["program"] != "vision":
            raise ValueError(
                f"{frozen_from} is a checkpoint of {head['program']!r}, not 'vision'"
            )
    return kinds[conf.learner.kind](mods, joint_dynamics(conf), conf=conf)


def parent_ckpt(run_dir: Path, conf: RunConf) -> Path:
    """The last evaluation checkpoint of the M15 run that the job in ``run_dir``
    names as its ``"parent"`` (``programs.posthoc.parent_dir``), for the frozen
    run ``conf``.

    Raises:
        ValueError: The job names no parent, the parent is not a vision run
            that finished, or its ``vision`` settings (but ``frozen``), seed,
            learner or balls differ from ``conf``'s.
        EnvError: The parent has no checkpoint here.
    """
    job = json.loads((run_dir / "job.json").read_text())
    parent = parent_dir(job, run_dir)
    if not (parent / "job.json").is_file():
        raise EnvError(f"the parent run {parent} is not here")
    pconf = from_dict(json.loads((parent / "job.json").read_text())["config"])
    if pconf.program != "vision":
        raise ValueError(f"the parent {parent} is of {pconf.program!r}, not 'vision'")
    if not (parent / "result.json").is_file():
        raise ValueError(f"the parent {parent} has not finished (no result.json)")
    assert conf.vision is not None and pconf.vision is not None
    same = {
        "vision": dataclasses.replace(pconf.vision, frozen=conf.vision.frozen)
        == conf.vision,
        "seed": pconf.seed == conf.seed,
        "learner": pconf.learner.kind == conf.learner.kind,
        "optim_vision": pconf.optim_vision == conf.optim_vision,
        "data.balls": pconf.data.balls == conf.data.balls,
    }
    if bad := [k for k, ok in same.items() if not ok]:
        raise ValueError(
            f"the parent {parent} differs from this run in {', '.join(bad)}: the "
            "control needs the encoder of the run it controls"
        )
    ckpts = sorted((parent / "ckpt").glob("*.pt"))
    if not ckpts:
        raise EnvError(f"the parent run {parent} has no evaluation checkpoint here")
    return ckpts[-1]


def _horizon(conf: RunConf) -> int:
    """The horizon of the objective and the divergence: 10, or the largest."""
    h = conf.measure.horizons
    return 10 if 10 in h else max(h)


def joint_objective(row: Mapping[str, Any], horizon: int) -> float:
    """``roll{horizon} / mean{horizon}`` of a val row, at most ``PENALTY``;
    ``PENALTY`` when it is not finite or $z$ collapsed (``z_spread`` below
    ``COLLAPSE``).

    It scores the dynamics in the latent space the encoder made, so sweep only
    the dynamics' settings with it: over the vision's it would select encoders
    whose $z$ is easy to predict, the distortion M16 keeps out of the credit."""
    spread = row.get("z_spread")
    if spread is None or not spread >= COLLAPSE:
        return PENALTY
    ratio = row.get(f"roll{horizon}/mean{horizon}")
    if ratio is None or not math.isfinite(ratio):
        return PENALTY
    return min(ratio, PENALTY)


class _Online:
    """The tasks ``"online"`` and ``"bench"`` (no evaluations)."""

    def __init__(
        self, conf: RunConf, run_dir: Path, resume: Mapping[str, Any] | None
    ) -> None:
        balls, v, dyn = conf.data.balls, conf.vision, conf.dynamics
        if v is None or dyn is None:
            raise ValueError(f"the run {conf.name!r} needs vision and dynamics")
        if balls is None or balls.obs != "frames":
            raise ValueError("the joint program streams balls with obs 'frames'")
        self.conf = conf
        bench = conf.task == "bench"
        frozen = parent_ckpt(run_dir, conf) if v.frozen else None
        if frozen is not None:
            write_json(
                run_dir / "frozen_from.json",
                {"parent": frozen.parent.parent.name, "ckpt": frozen.name},
            )
        self.net = build_joint(conf, frozen_from=frozen)
        self.mods, self.dyn = self.net.mods, self.net.dyn
        self.measures = None if bench else Measures(conf)
        self._windows: dict[str, Windows] = {}
        if self.measures is not None:  # of the initial weights, before a resume
            base = self.measures.baselines(self.mods)
            write_json(
                run_dir / "baselines.json",
                {
                    s: {k: x if math.isfinite(x) else None for k, x in b.items()}
                    for s, b in base.items()
                },
            )
        frame, b = conf.time.frame, conf.data.streams
        self.streams = BallFrames.of(balls, b, frame=frame, seed=conf.seed)
        dev = torch.device(conf.device)
        self._y = torch.zeros(b, 1, device=dev)
        self._mask = torch.ones(b, dtype=torch.bool, device=dev)
        # The sums since the last log, in host memory, where Host.show reads
        # the messages: the vision's loss, pred, sigreg, recon and frames; then
        # the dynamics' error and count from the observation and from its own
        # prediction.
        self._recon = v.recon > 0
        self._acc = torch.zeros(9, dtype=torch.float64)
        state = None if resume is None else resume["mujou"]
        self._run: ManualRun = mujou.testing.run(self.net, state=state)
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
            viz=0 if bench else every(ctl.viz_every, frame),
            resume=every(ctl.resume_every, frame),
            test=ctl.eval_test and not bench,
        )

    @property
    def schedule(self) -> Schedule:
        return self._schedule

    def advance(self) -> None:
        k = self.host.frames
        dyn = self.conf.dynamics
        assert dyn is not None
        images, _, reset = self.streams.next()
        dev = self._y.device
        sample = pc.msg.Sample(
            x=images.to(dev), y=self._y, mask=self._mask, reset=reset.to(dev)
        )
        (p, q), _ = self.host.show_all(
            {self.net.prev: sample}, [self.net.stats, self.net.next]
        )
        if k == 0:
            return
        terms = torch.stack(
            [p["loss"], p["pred"], p["sigreg"], p["recon"], torch.ones_like(p["n"])]
        )
        self._acc[:5] += torch.where(p["n"] > 0, terms.double(), 0.0)
        err = ((q["y_hat"] - q["y"]) ** 2).mean(-1)
        mask = q["mask"]
        i = 7 if k > 1 and is_open(k, dyn.open_loop, dyn.open_cycle) else 5
        sums = torch.stack(
            [torch.where(mask, err, 0.0).sum().double(), mask.sum().double()]
        )
        self._acc[i : i + 2] += sums

    def log(self) -> dict[str, float | None]:
        loss, pred, sig, rec, n, e, ne, e_open, n_open = self._acc.tolist()
        self._acc.zero_()
        return {
            "loss": loss / n if n else None,
            "pred": pred / n if n else None,
            "sigreg": sig / n if n else None,
            **({"recon": rec / n if n else None} if self._recon else {}),
            "preq": e / ne if ne else None,
            "preq_open": e_open / n_open if n_open else None,
        }

    def evaluate(self, split: str) -> dict[str, float]:
        assert self.measures is not None  # no evaluations in a bench
        conf = self.conf
        assert conf.dynamics is not None
        row = self.measures.evaluate(self.mods, split, version=self.host.frames)
        if split not in self._windows:
            self._windows[split] = ball_windows(conf, split, stride=STRIDE)
        row |= latent_row(
            self.mods,
            self.dyn,
            self._windows[split],
            conf.measure.horizons,
            residual=conf.dynamics.residual,
        )
        if split == "val":
            row["objective"] = joint_objective(row, _horizon(conf))
        return row

    def diverged(self, row: Mapping[str, float]) -> str | None:
        """A term of the loss or ``roll{h}`` is not finite, or a rollout is not
        finite (``h`` as in ``_horizon``). Not ``measure.diverge``: $z$ is not
        standardized and its spread is tiny early on, so ``roll{h}`` far above
        ``mean{h}`` is not yet a run to stop; the rows and the objective tell it."""
        h = _horizon(self.conf)
        keys = [
            "loss",
            "pred",
            "sigreg",
            *(["recon"] if self._recon else []),
            f"roll{h}",
        ]
        bad = [k for k in keys if not math.isfinite(row[k])]
        if bad:
            return f"{', '.join(bad)} not finite"
        if row["nonfinite"]:
            return f"{row['nonfinite']:g} windows predicted values that are not finite"
        return None

    def objective(
        self, row: Mapping[str, Any] | None, *, diverged: bool
    ) -> float | None:
        if diverged:
            return PENALTY
        return None if row is None else joint_objective(row, _horizon(self.conf))

    def modules(self) -> dict[str, nn.Module]:
        return self.mods.named() | {"step": self.dyn.step, "readout": self.dyn.readout}

    def visualize(self, out: Path) -> None:
        if self.measures is not None:
            self.measures.visualize(out, self.mods.patch)

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


class JointProgram:
    """The program ``"joint"`` (module docstring), a ``loop.Program``."""

    def build(
        self, conf: RunConf, run_dir: Path, resume: Mapping[str, Any] | None
    ) -> Built:
        match conf.task:
            case "online" | "bench":
                return _Online(conf, run_dir, resume)
            case "posthoc":  # imported here: it imports the vision program
                from exp.video.programs.posthoc import Posthoc

                return Posthoc(conf, run_dir, resume)
            case task:
                raise ValueError(f"the joint program has no task {task!r}")


def rebuild(conf: RunConf) -> dict[str, nn.Module]:
    """Fresh modules of ``conf`` by the names of the evaluation checkpoint, to
    load one into (``exp.lib.ckpt.load_eval``)."""
    mods, dyn = build_vision(conf, seed=conf.seed), joint_dynamics(conf)
    return mods.named() | {"step": dyn.step, "readout": dyn.readout}


PROGRAM = JointProgram()
