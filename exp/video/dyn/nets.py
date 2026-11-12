r"""The world models of M14 on the state of the balls: ``Source -> DynamicsBlock``
(PC-ALM) or ``Source -> DynamicsBP`` (BP), made from a ``RunConf`` from the
same initial weights, and the optimizer of a ``OptimConf``."""

import math
from typing import cast

import mujou.pc as pc
import torch
from mujou.pc.optim import Optimizer

import mujou
from exp.lib.optim import AdamDirection, Grafted, StepAdamW, Warmup
from exp.video.dyn.bp import DynamicsBP
from exp.video.dyn.modules import DynamicsModules, build_dynamics
from exp.video.dyn.nodes import DynamicsBlock, DynamicsLayer, Source
from exp.video.schema.dyn import DynamicsConf
from exp.video.schema.learn import OptimConf
from exp.video.schema.run import RunConf
from mujou import TensorSpec

__all__ = [
    "StateWorldBP",
    "StateWorldPC",
    "build_world",
    "live_states",
    "make_optimizer",
    "modules",
    "window_stats",
]


def make_optimizer(c: OptimConf, frame: float) -> Optimizer:
    r"""The optimizer of ``c``, with the time constant $\tau$ = ``frame / c.lr``
    of the parameters (a step of one frame moves SGD by ``lr`` times the
    gradient): ``pc.optim.SGD``, ``pc.optim.AdamW`` or ``Grafted(AdamDirection)``,
    in ``Warmup`` over ``c.warmup`` seconds unless it is 0; or, for
    ``"adamw_step"``, ``StepAdamW`` of ``c.lr`` per step and its schedules
    (``frame`` unused).

    Raises:
        ValueError: ``c.lr`` is not > 0, or ``c.weight_decay`` is not 0 for a
            kind without weight decay (``"sgd"``, ``"graft_adam"``).
    """
    if not c.lr > 0:
        raise ValueError(f"optim.lr must be > 0, got {c.lr}")
    if c.kind not in ("adamw", "adamw_step") and c.weight_decay != 0:
        raise ValueError(
            f"optim.kind {c.kind!r} has no weight decay; "
            f"got weight_decay {c.weight_decay} (use 'adamw')"
        )
    if c.kind == "adamw_step":
        b1, b2 = c.betas
        return StepAdamW(
            lr=c.lr,
            betas=(b1, b2),
            eps=c.eps,
            weight_decay=c.weight_decay,
            weight_decay_end=c.weight_decay_end,
            warmup_steps=c.warmup_steps,
            decay=c.decay,
            decay_steps=c.decay_steps,
            min_lr=c.min_lr,
            clip=c.clip,
        )
    tau = frame / c.lr
    inner: Optimizer
    if c.kind == "sgd":
        inner = pc.optim.SGD(tau=tau)
    elif c.kind == "adamw":
        inner = pc.optim.AdamW(
            tau=tau, tau_m=c.tau_m, tau_v=c.tau_v, weight_decay=c.weight_decay
        )
    else:  # "graft_adam", as OptimConf checks
        inner = Grafted(inner=AdamDirection(tau=tau, tau_m=c.tau_m, tau_v=c.tau_v))
    return inner if c.warmup == 0 else Warmup(inner=inner, seconds=c.warmup)


def _parts(conf: RunConf) -> tuple[DynamicsConf, pc.TimeConstants, int]:
    """The dynamics, the time constants and the batch B of ``conf``."""
    if conf.dynamics is None:
        raise ValueError(f"the run {conf.name!r} has no dynamics")
    t = conf.time
    return conf.dynamics, pc.TimeConstants(h=t.h, lam=t.lam, r=t.r), conf.data.streams


class StateWorldPC(mujou.Network):
    """``Source -> DynamicsBlock`` of ``mods``: the host writes a
    ``pc.msg.Sample`` (``x`` ``(B, d_in)``, ``y`` zeros ``(B, 1)``, ``mask``
    and ``reset`` ``(B,)``) to ``prev`` and reads the prequential prediction
    from ``next``; ``tau`` is the time constants of the nodes.

    Args:
        mods: The modules, on ``conf.device``.
        conf: The run: its dynamics, time, streams and device.
        optimizer: The optimizer of every node.
    """

    def __init__(
        self, mods: DynamicsModules, *, conf: RunConf, optimizer: Optimizer
    ) -> None:
        super().__init__()
        dyn, tau, b = _parts(conf)
        self.tau = tau
        d_in = mods.step.first.gate.in_features
        self.source = Source(
            x=TensorSpec((b, d_in)), period=tau.period, batch=b, device=conf.device
        )
        self.block = DynamicsBlock(
            mods,
            act_spec=self.source.act_spec,
            conf=dyn,
            tau=tau,
            optimizer=optimizer,
            learn_interval=conf.time.learn_interval,
            device=conf.device,
        )
        self.block.connect(self.source.out)
        self.prev, self.next, self.error = (
            self.source.prev,
            self.block.next,
            self.block.error,
        )


class StateWorldBP(mujou.Network):
    """``Source -> DynamicsBP`` of ``mods``, with the ports of
    ``StateWorldPC``."""

    def __init__(
        self, mods: DynamicsModules, *, conf: RunConf, optimizer: Optimizer
    ) -> None:
        super().__init__()
        dyn, tau, b = _parts(conf)
        self.tau = tau
        d_in = mods.step.first.gate.in_features
        self.source = Source(
            x=TensorSpec((b, d_in)), period=tau.period, batch=b, device=conf.device
        )
        self.world = DynamicsBP(
            mods,
            act_spec=self.source.act_spec,
            conf=dyn,
            tau=tau,
            optimizer=optimizer,
            learn_interval=conf.time.learn_interval,
            device=conf.device,
        )
        self.source.out.connect(self.world.prev)
        self.prev, self.next, self.error = (
            self.source.prev,
            self.world.next,
            self.world.error,
        )


def build_world(conf: RunConf, d_obs: int) -> StateWorldPC | StateWorldBP:
    """The world model of ``conf`` on observations of ``d_obs``: the modules of
    ``build_dynamics`` (seed ``conf.seed``, predicting the next observation)
    in the nodes of ``conf.learner`` (``"pc"`` or ``"bp"``), with the
    optimizer of ``conf.optim``.

    Raises:
        ValueError: ``conf`` has no dynamics, or its learner is ``"none"``.
    """
    dyn, _, _ = _parts(conf)
    kinds = {"pc": StateWorldPC, "bp": StateWorldBP}
    if conf.learner.kind not in kinds:
        raise ValueError(
            f"learner {conf.learner.kind!r} learns no world model; use 'pc' or 'bp'"
        )
    mods = build_dynamics(dyn, d_obs, d_obs, seed=conf.seed, device=conf.device)
    optimizer = make_optimizer(conf.optim, conf.time.frame)
    return kinds[conf.learner.kind](mods, conf=conf, optimizer=optimizer)


def modules(net: StateWorldPC | StateWorldBP) -> DynamicsModules:
    """The live modules of ``net`` (no copy): read them between the steps of
    its run."""
    return net.block.mods if isinstance(net, StateWorldPC) else net.world.mods


def live_states(net: StateWorldPC | StateWorldBP) -> torch.Tensor:
    r"""The state $(s_1, h_2, \dots, h_L)_k$ ``(B, L d_s)`` of each stream after
    the frame the nodes saw last (that of ``DynamicsStack``) at the weights as
    they are now (after the frame's learning, for PC and BP alike), read
    between the steps of its run: the state a rollout from the next input
    starts from. It is not the state of the prequential prediction of that
    frame when the weights learned in the frame."""
    if isinstance(net, StateWorldBP):
        return net.world.s_last.clone()
    block = net.block
    layers = [cast(DynamicsLayer, getattr(block, n)) for n in block.names[1:]]
    return torch.cat([block.dynamics.mu, *(layer.h_last for layer in layers)], -1)


def window_stats(dyn: DynamicsBP | DynamicsBlock) -> dict[str, float | None]:
    """The learning on windows of ``dyn`` (one with ``dynamics.window``), read
    between the steps of its run, for the records only: ``win_loss`` and
    ``win_gnorm`` of the last window learned (None before the first),
    ``win_steps``, ``win_skipped`` and ``win_masked`` so far.

    For ``DynamicsBP`` they are its own. For the nodes of ``DynamicsBlock``,
    ``win_steps`` is the least of the nodes', ``win_skipped`` the sum of the
    nodes' (a window skipped by two nodes counts twice), ``win_masked``
    ``Dynamics``'s and ``win_loss`` ``Compare``'s; ``win_gnorm`` is the norm
    over the nodes of the gradient each learned last (BP's ``win_gnorm`` at
    the equilibrium, when they learned the same window); it also has
    ``win_unsettled`` and ``win_vjps`` (the windows learned with inputs that
    changed shortly before, the computations of the window functions), summed
    over the nodes.

    Raises:
        ValueError: ``dyn`` does not learn on windows.
    """
    if dyn.window is None:
        raise ValueError("the dynamics does not learn on windows")
    if isinstance(dyn, DynamicsBP):
        loss, gnorm = dyn.win_loss.item(), dyn.win_gnorm.item()
        return {
            "win_loss": loss if math.isfinite(loss) else None,
            "win_gnorm": gnorm if math.isfinite(gnorm) else None,
            "win_steps": dyn.win_steps,
            "win_skipped": dyn.win_skipped,
            "win_masked": dyn.win_masked,
        }
    nodes = dyn.nodes
    loss = dyn.compare.win_loss.item()
    gnorm = math.sqrt(sum(n.win_gsq.item() for n in nodes))
    return {
        "win_loss": loss if math.isfinite(loss) else None,
        "win_gnorm": gnorm if math.isfinite(gnorm) else None,
        "win_steps": min(n.win_steps for n in nodes),
        "win_skipped": sum(n.win_skipped for n in nodes),
        "win_masked": dyn.dynamics.win_masked,
        "win_unsettled": sum(n.win_unsettled for n in nodes),
        "win_vjps": sum(n.win_vjps for n in nodes),
    }
