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

The program ``"vae"`` (stage 2) is the same with the VAE (``bp.VAENet``,
``measures.VAEMeasures``): the host draws each frame's noise, the
dequantization's $u$ and the code's $\varepsilon$, from one generator on the
device seeded with ``seed + NOISE_SEED`` (kept in the resume checkpoint), and
``stream.jsonl`` has the means of $-\mathcal L$ (``loss``, nats a stack, and
``bpd``) and of its KL (``kl``), and the last $\ln \sigma$
(``log_sigma``). With ``learner=pc`` (stage 4) the VAE learns by PC-ALM
(``pc.VAEPC``): the host writes the same stacks and noise, packed with the
target and the noise in ``y`` (``VAEPC.sample``, the stream's ``reset``
with them), and reads $-\mathcal L$ as the sum of the output's
$-\log p(y \mid z)$ and the latent's KL.
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

from exp.lib.elbo import bits_per_dim
from exp.lib.host import Host
from exp.lib.rundir import write_json
from exp.video.data.frames import (
    IIDStackedTourFrames,
    PooledStackedTourFrames,
    ShuffledStackedTourFrames,
    StackedTourFrames,
    StackStreams,
)
from exp.video.jepa.vit import image_shape
from exp.video.loop import Built, Schedule, every
from exp.video.measure.objective import PENALTY
from exp.video.programs.vision import tours_root
from exp.video.schema.run import RunConf
from exp.video.vae.bp import (
    AENet,
    VAENet,
    VAESample,
    build_ae_net,
    build_vae_net,
)
from exp.video.vae.measures import AEMeasures, VAEMeasures
from exp.video.vae.modules import build_modules
from exp.video.vae.pc import VAEPC, build_vae_pc_net

__all__ = [
    "NOISE_SEED",
    "PROGRAM",
    "SOURCE_SEED",
    "AEProgram",
    "ae_streams",
    "rebuild",
]

SOURCE_SEED = 6  # added to the run's seed: the shuffle's order, the pool's draws
NOISE_SEED = 13  # added to the run's seed: the VAE's noise of the stream


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
        self.vae = conf.program == "vae"
        self.net: AENet | VAENet | VAEPC
        if self.vae and conf.learner.kind == "pc":
            self.net = build_vae_pc_net(conf)
        elif self.vae:
            self.net = build_vae_net(conf)
        else:
            self.net = build_ae_net(conf)
        self._code_shape = (0, 0)  # (T, d) of the VAE's code
        if isinstance(self.net, VAENet | VAEPC):
            self._code_shape = self.net.mods.code_shape
        self.streams = ae_streams(conf)
        self.measures: AEMeasures | None = None
        if not bench:
            assert conf.data.tours is not None  # ae_streams checks
            pool = None
            if isinstance(self.streams, PooledStackedTourFrames):
                pool = self.streams.stacks
            kind = VAEMeasures if self.vae else AEMeasures
            self.measures = kind(conf, tours_root(conf.data.tours), pool=pool)
            base = {
                s: self.measures.evaluate(self.net.mods, s) for s in ("val", "test")
            }
            write_json(
                run_dir / "baselines.json", {s: _clean(r) for s, r in base.items()}
            )
        b = conf.data.streams
        dev = torch.device(conf.device)
        self._y = torch.zeros(b, 1, device=dev)
        # the VAE's noise of the stream
        self._noise = torch.Generator(device=dev).manual_seed(conf.seed + NOISE_SEED)
        # loss (and the VAE's kl) and the frames since the last log, in host
        # memory (Host.show); the VAE's last log sigma
        self._acc = torch.zeros(3, dtype=torch.float64)
        self._log_sigma: float | None = None
        v = conf.vision
        assert v is not None  # ae_streams checks
        channels, size = image_shape(conf)
        self._values = v.frames * channels * size * size  # P of a stack
        state = None if resume is None else resume["mujou"]
        self._run: ManualRun = mujou.testing.run(self.net, state=state)
        frame = conf.time.frame
        self.host = Host(self._run, frame=frame, period=self.net.tau.period)
        if resume is not None:
            h = resume["host"]
            self.host.load_state_dict(h["host"])
            self.streams.load_state_dict(h["streams"])
            self._acc.copy_(h["acc"])
            if self.vae:
                self._noise.set_state(h["noise"])
                self._log_sigma = h["log_sigma"]
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
        x, mask = stacks.to(dev), ready.to(dev)
        p = self._show(x, mask, reset.to(dev))
        if k == 0:
            return
        counted = p["n"] > 0
        kl = p["kl"] if self.vae else torch.zeros_like(p["n"])
        terms = torch.stack([p["loss"], kl, torch.ones_like(p["n"])])
        self._acc += torch.where(counted, terms.double(), 0.0)
        if self.vae:
            self._log_sigma = float(p["log_sigma"])

    def _show(self, x: torch.Tensor, mask: torch.Tensor, reset: torch.Tensor) -> Any:
        """Show the frame's stacks ``x`` (with the VAE's noise drawn here)
        and return what the net sent first of them: ``n``, ``loss`` (and the
        VAE's ``kl`` and ``log_sigma``)."""
        dev = x.device
        net = self.net
        if not self.vae:
            sample = pc.msg.Sample(x=x, y=self._y, mask=mask, reset=reset)
            return self.host.show({net.prev: sample}, net.stats)[0]
        shape = (len(x), *self._code_shape)
        u = torch.rand(x.shape, generator=self._noise, device=dev)
        eps = torch.randn(shape, generator=self._noise, device=dev)
        if not isinstance(net, VAEPC):
            sample = VAESample(x=x, u=u, eps=eps, mask=mask)
            return self.host.show({net.prev: sample}, net.stats)[0]
        (out, latent), _ = self.host.show_all(
            {net.prev: net.sample(x, u, eps, mask, reset)}, [net.stats, net.kl]
        )
        return {
            "loss": out["nll"] + latent["kl"],
            "kl": latent["kl"],
            "log_sigma": out["log_sigma"],
            "n": out["n"],
        }

    def log(self) -> dict[str, float | None]:
        loss, kl, n = self._acc.tolist()
        self._acc.zero_()
        if not self.vae:
            return {"loss": loss / n if n else None}
        p = self._values
        return {
            "loss": loss / n if n else None,
            "bpd": float(bits_per_dim(loss / n, p)) if n else None,
            "kl": kl / n if n else None,
            "log_sigma": self._log_sigma,
        }

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
        return None if row is None else row.get("bpd" if self.vae else "stack_mse")

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
        if self.vae:
            host |= {"noise": self._noise.get_state(), "log_sigma": self._log_sigma}
        return self._run.state_dict(), host

    def final(self) -> dict[str, dict[str, Any]]:
        return {}

    def close(self) -> None:
        self._run.shutdown()


def _clean(d: Mapping[str, float]) -> dict[str, float | None]:
    return {k: (x if math.isfinite(x) else None) for k, x in d.items()}


class AEProgram:
    """The programs ``"ae"`` and ``"vae"`` (module docstring), a
    ``loop.Program``."""

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
                raise ValueError(f"the {conf.program} program has no task {task!r}")


def rebuild(conf: RunConf) -> dict[str, nn.Module]:
    """Fresh modules of ``conf`` by the names of the evaluation checkpoint, to
    load one into (``exp.lib.ckpt.load_eval``)."""
    return build_modules(conf, seed=conf.seed).named()


PROGRAM = AEProgram()
