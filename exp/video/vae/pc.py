r"""The VAE of stage 2 learned by PC-ALM (M18, stage 4; plans/m18-vae-stream.md
「PC-ALM への載せ方」): the modules, the initial weights, the objective and the
optimizer of ``bp.VAENet``, the learning rule replaced.

``inp -> enc1 .. encL -> latent -> dec1 .. decK -> out``, in this order (the
order of the tree, in which the host steps them):

- ``inp`` (``pc.Input``): the host's ``pc.msg.Sample`` (``VAEPC.sample``),
  ``x`` the stacks packed as the encoder reads them (``VisionModules.whole``,
  ``(B, T, P_t + 1)``) and ``y`` the target $y = (x + u) / 256$ of each stack
  ($P$ values) followed by its code's noise $\varepsilon$ ($D = T d$): the
  noise and the target travel with the activity, so ``latent`` and ``out``
  use those of the write their activity was computed from;
- ``enc1`` .. ``encL`` (``pc.Layer``): the encoder's stages
  (``vit.Segmented`` of the whole stack), the last one to $h_4 = (\mu, \ln
  s^2)$ ``(B, T, 2d)``;
- ``latent`` (``Latent``): the constraint $z = \mu + e^{\ell / 2} \odot
  \varepsilon$ and the KL of $q$ as a local energy;
- ``dec1`` .. ``decK`` (``pc.Layer``): the decoder's blocks, ``per`` a node as
  the encoder's (``TokenDecoder.parts``);
- ``out`` (``GaussOut``): the readout and the Gaussian likelihood of the
  learned $\ln \sigma$.

With $r_i = h_i - f_i(h_{i-1})$ the augmented Lagrangian is

$$\mathcal F = E_{\mathrm{out}}(h_7) + E_{\mathrm{KL}}(h_4) + \sum_i
\big(\lambda_i^\top r_i + \tfrac12 \lVert r_i \rVert^2\big),$$

$E_{\mathrm{out}} + E_{\mathrm{KL}} = -\mathcal L$ (``elbo.elbo_terms``) at
the forward values. At its fixed point $\lambda_i = -\partial(-\mathcal L) /
\partial h_i$ and every node's $\partial \mathcal F / \partial \theta$ is
BP's $\partial(-\mathcal L) / \partial \theta$. ``latent`` and ``out`` weigh
their energies by $B / n$ ($n$ the stacks that count, from the activity's
``mask``), so that a node's $g_\theta / B$ is the gradient of BP's loss, the
mean over the stacks that count (``loss.batch_loss``).

Under the host's steps (``exp.lib.host``: all nodes once each ``tau.period``,
in the order of the tree) and the default time constants ($\tau_\lambda =
\tau_r / 2 = \tau_h / 2$, steps of $\tau_h / 2$), an update moves $\delta =
r + \lambda$ to $-g$ exactly, the forward reaches ``out`` in the step of the
write, and the $k$-th node from the output has BP's credit after step $2k$:
a node learns by BP's gradient (up to the rounding) from step $2 \cdot 7 + 1
= 15$ on with 4 encoder and 2 decoder nodes (``time.learn_interval`` 29 ms).
"""

import math
from typing import TypedDict, cast, override

import mujou.pc as pc
import torch
from mujou.pc.msg import Activity, rows
from mujou.pc.optim import Optimizer
from mujou.time import Timestamp
from mujou.types import Device
from torch import Tensor

import mujou
from exp.lib.elbo import dequantize, gauss_kl, gauss_nll
from exp.video.dyn.nets import make_optimizer
from exp.video.jepa.vit import Segmented, Stage, image_shape
from exp.video.schema.run import RunConf
from exp.video.vae.loss import LEVELS
from exp.video.vae.modules import DecoderPart, LogSigma, VAEModules, build_vae
from mujou import DictSpec, MultiPort, Node, Port, TensorSpec, TimestampSpec

__all__ = [
    "GaussOut",
    "KLStats",
    "KL_STATS",
    "Latent",
    "NLLStats",
    "NLL_STATS",
    "VAEPC",
    "build_vae_pc_net",
]

_SCALAR = TensorSpec(())


class KLStats(TypedDict):
    """What ``Latent`` sends on ``stats``: the mean over the stacks that count
    of the KL (nats a stack) of the activity it received (0 when none counts),
    the number ``n`` of them (0-d float32), and its ``time``."""

    kl: Tensor
    n: Tensor
    time: Timestamp


KL_STATS = DictSpec(KLStats, kl=_SCALAR, n=_SCALAR, time=TimestampSpec())


class NLLStats(TypedDict):
    r"""What ``GaussOut`` sends on ``stats``: the mean over the stacks that
    count of $-\log p(y \mid z)$ (nats a stack) of the activity it received,
    before it learned from it (0 when none counts), $\ln \sigma$ then, the
    number ``n`` of them (0-d float32), and its ``time``."""

    nll: Tensor
    log_sigma: Tensor
    n: Tensor
    time: Timestamp


NLL_STATS = DictSpec(
    NLLStats, nll=_SCALAR, log_sigma=_SCALAR, n=_SCALAR, time=TimestampSpec()
)


def _count(valid: Tensor, like: Tensor) -> tuple[Tensor, Tensor]:
    """``(n, B / n)``: the stacks that count of ``valid`` ``(B,)`` (0-d, of
    ``like``'s dtype) and the weight of the energies, $n$ taken as 1 when
    none counts (their energy is 0 then)."""
    n = valid.to(like.dtype).sum()
    return n, len(valid) / n.clamp(min=1)


class Latent(Node):
    r"""The latent of the VAE as a node of PC-ALM: the constraint $z = f(h;
    \varepsilon) = \mu + e^{\ell / 2} \odot \varepsilon$ of $h = (\mu, \ell)$
    ``(B, T, 2d)`` ($\ell = \ln s^2$), with the KL of $q = N(\mu,
    \mathrm{diag}\, e^\ell)$ to $N(0, I)$ as a local energy. $\varepsilon$
    is the last $T d$ values of each stack's ``y`` (the values before it are
    the next nodes' target), taken as 0 for a stack whose ``mask`` is False.

    It holds $r$ and $\lambda$ of $z$ ``(B, T, d)`` as ``pc.Layer`` does: it
    sends $f(h)$ on, with the ``y``, ``mask``, ``reset`` and ``time`` it
    received; it sets $r$ and $\lambda$ of the stacks to ``reset`` to 0 in
    the update that first sees an input; it moves them by ``relax_`` with
    the credit $g$ of the next nodes and the capped ``dt``. It sends back
    the gradient of its energy

    $$E = \sum_b \big(-f(h_b)^\top \delta_b\big) + \frac Bn \sum_{b:\,
    \mathrm{mask}_b} \mathrm{KL}(h_b),$$

    $g_{\mathrm{in}} = -J^\top \delta + \frac Bn \partial \mathrm{KL} /
    \partial h$ ($\delta = r + \lambda$ before the update, 0 for a stack
    whose ``mask`` is False), from one autograd of its own computation from
    the detached $h$. It has no parameters, so it never learns, and sends the
    mean KL on ``stats`` in every update.

    Args:
        act_spec: The spec of what the encoder's last node sends: ``h`` ``(B,
            T, 2d)``, ``y`` ``(B, P + T d)``.
        tau: The time constants of $r$ and $\lambda$.
        device: Where the node computes.

    Raises:
        ValueError: ``y`` holds fewer than $T d$ values a stack.
    """

    def __init__(
        self,
        *,
        act_spec: DictSpec[Activity],
        tau: pc.TimeConstants,
        device: Device = "cpu",
    ) -> None:
        super().__init__(device=device, period=tau.period)
        self._tau = tau
        # no parameters: the clock caps dt and tells a new input, never learns
        self.clock = pc.clock.LearnClock.of(
            tau, period=self.period, learn_interval=math.inf
        )
        h = act_spec.field("h", TensorSpec)
        y = act_spec.field("y", TensorSpec)
        b, t, two_d = h.shape
        self._d = two_d // 2
        self._values = y.shape[1] - t * self._d  # P: the target's values
        if self._values < 0:
            raise ValueError(
                f"y of {y.shape[1]} values holds no noise of {t} x {self._d}"
            )
        z = TensorSpec((b, t, self._d), h.dtype)
        self.r, self.lam = z.zeros(device=self.device), z.zeros(device=self.device)
        self.act_spec = pc.msg.activity_spec(z, y, b)
        self.prev = Port(send=h, recv=act_spec)
        self.next = MultiPort(send=self.act_spec, recv=z)
        self.stats = Port(send=KL_STATS)

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        act = self.prev.recv()
        tick = self.clock.tick(dt, act["time"], self.timer.now())
        if tick.new:
            reset = rows(act["reset"], self.r)
            self.r.masked_fill_(reset, 0.0)
            self.lam.masked_fill_(reset, 0.0)
        valid = act["mask"]
        delta = torch.where(rows(valid, self.r), self.r + self.lam, 0.0)
        z, g_in, kl, n = self._terms(act["h"], act["y"], valid, delta)
        self.prev.send(g_in)
        g = sum(self.next.recv().values(), torch.zeros_like(self.r))
        pc.functional.relax_(self.r, self.lam, g, tick.dt, self._tau)
        self.next.send({**act, "h": z})
        # float32, whatever the messages' dtype
        self.stats.send({"kl": kl.float(), "n": n.float(), "time": act["time"]})

    def _terms(
        self, h: Tensor, y: Tensor, valid: Tensor, delta: Tensor
    ) -> list[Tensor]:
        """``[z, g_in, kl, n]`` (class docstring)."""
        b, t, _two = h.shape
        d = self._d
        eps = y[:, self._values :].reshape(b, t, d)
        # a stack that does not count adds nothing, whatever its y (a NaN in
        # it would give 0 * NaN in the gradient), as pc.functional.loss_vjp
        eps = torch.where(rows(valid, eps), eps, 0.0)
        n, weight = _count(valid, h)
        with torch.enable_grad():
            leaf = h.detach().requires_grad_()
            mu, logvar = leaf[..., :d], leaf[..., d:]
            z = mu + (0.5 * logvar).exp() * eps
            kl = torch.where(valid, gauss_kl(mu, logvar).flatten(1).sum(1), 0.0)
            # -J^T delta from z (delta held fixed), as pc.functional.vjp's
            energy = (z * -delta).sum() + weight * kl.sum()
            # The gradient of this node's own energy, from a detached leaf.
            (g_in,) = torch.autograd.grad(energy, [leaf])  # mujou: local-autograd
        return [z.detach(), g_in, kl.detach().sum() / n.clamp(min=1), n]


class GaussOut(Node):
    r"""The output of the VAE as a node of PC-ALM: the decoder's readout
    ``module`` (``TokenDecoder.parts``, the last: $\hat y$ ``(B, N, C, H,
    W)`` of the tokens) and the Gaussian likelihood of the learned $\ln
    \sigma$ (``likelihood``), whose energy is

    $$E = \frac Bn \sum_{b:\, \mathrm{mask}_b} -\log N(y_b; \hat y_b,
    \sigma^2 I)$$

    (``exp.lib.elbo.gauss_nll``), $y_b$ the first $P$ values of the stack's
    ``y`` (the target the host wrote with it). In every update it sends back
    $g_{\mathrm{in}} = \partial E / \partial h$ and the mean
    $-\log p(y \mid z)$ on ``stats``, from one autograd of its own
    computation from the detached $h$; when its clock says (as
    ``pc.Layer``'s) and a stack counts, it learns the readout's parameters
    and $\ln \sigma$ by $g_\theta / B = \partial L / \partial \theta$ of the
    same autograd, stepped by the clock's ``elapsed``.

    Args:
        module: The readout, on ``device``.
        likelihood: $\ln \sigma$, on ``device``.
        act_spec: The spec of what the decoder's last node sends.
        tau: The time constants (the period and ``max_dt``).
        optimizer: The optimizer of the readout's parameters and $\ln
            \sigma$.
        learn_interval: As in ``pc.clock.LearnClock``.
        device: Where the node computes.
    """

    def __init__(
        self,
        module: DecoderPart,
        likelihood: LogSigma,
        *,
        act_spec: DictSpec[Activity],
        tau: pc.TimeConstants,
        optimizer: Optimizer,
        learn_interval: float | None,
        device: Device = "cpu",
    ) -> None:
        super().__init__(device=device, period=tau.period)
        self.module, self.likelihood = module, likelihood
        self.clock = pc.clock.LearnClock.of(
            tau, period=self.period, learn_interval=learn_interval
        )
        self._params = [*module.parameters(), *likelihood.parameters()]
        self.optimizer = optimizer.bind(self._params)
        h = act_spec.field("h", TensorSpec)
        with torch.no_grad():
            self._shape = tuple(module(h.zeros(device=self.device)).shape)
        self._b, self._values = self._shape[0], math.prod(self._shape[1:])
        self.prev = Port(send=h, recv=act_spec)
        self.stats = Port(send=NLL_STATS)

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        act = self.prev.recv()
        tick = self.clock.tick(dt, act["time"], self.timer.now())
        valid = act["mask"]
        learn = tick.learn and bool(valid.any())
        g_in, nll, n, *g_theta = self._terms(learn, act["h"], act["y"], valid)
        self.prev.send(g_in)
        self.stats.send(  # float32, whatever the messages' dtype
            {
                "nll": nll.float(),
                "log_sigma": self.likelihood.log_sigma.detach().float(),
                "n": n.float(),
                "time": act["time"],
            }
        )
        if learn:
            # theta moves by dL/dtheta = (1 / B) dE/dtheta.
            self.optimizer.step_([g / self._b for g in g_theta], tick.elapsed)

    def _terms(self, learn: bool, h: Tensor, y: Tensor, valid: Tensor) -> list[Tensor]:
        """``[g_in, nll, n, *g_theta]`` (``g_theta`` only when it learns)."""
        params = self._params if learn else []
        target = y[:, : self._values].reshape(self._shape)
        # as Latent's eps: a stack that does not count has no target
        target = torch.where(rows(valid, target), target, 0.0)
        n, weight = _count(valid, h)
        with torch.enable_grad():
            leaf = h.detach().requires_grad_()
            nll = gauss_nll(target, self.module(leaf), self.likelihood.log_sigma)
            nll = torch.where(valid, nll, 0.0)
            # The gradient of this node's own energy, from a detached leaf.
            g_in, *g_theta = torch.autograd.grad(  # mujou: local-autograd
                weight * nll.sum(), [leaf, *params], materialize_grads=True
            )
        return [g_in, nll.detach().sum() / n.clamp(min=1), n, *g_theta]


class VAEPC(mujou.Network):
    r"""The VAE of ``conf`` in the nodes of PC-ALM (module docstring): the
    host writes ``sample(...)`` to ``prev`` and reads ``stats`` (``out``'s
    ``NLLStats``) and ``kl`` (``latent``'s ``KLStats``); $-\mathcal L$ is
    the sum of their ``nll`` and ``kl``.

    The encoder's nodes are its stages (``vision.nodes``), the decoder's of
    as many blocks a node (``vision.depth / vision.nodes``); the ``pc.Layer``
    nodes replay their updates from CUDA graphs with ``ctl.cuda_graph`` on
    CUDA (``latent`` and ``out`` do not). The messages are of the modules'
    dtype. One ``optimizer`` (built with all the modules,
    for ``scale="fan_in"``) is bound by each node to its own parameters.

    Args:
        mods: The modules, on ``conf.device``.
        conf: The run: its vision, time, streams, ``ctl.cuda_graph`` and
            device.
        optimizer: The optimizer of all the parameters (``optim_vision``).

    Raises:
        ValueError: The decoder's depth is not a multiple of the encoder's
            blocks a node (``TokenDecoder.parts``).
    """

    def __init__(
        self, mods: VAEModules, *, conf: RunConf, optimizer: Optimizer
    ) -> None:
        super().__init__()
        v, t = conf.vision, conf.time
        assert v is not None  # build_vae checks
        self.mods = mods
        self.tau = tau = pc.TimeConstants(h=t.h, lam=t.lam, r=t.r)
        b, dtype, dev = conf.data.streams, mods.log_sigma.dtype, conf.device
        channels, size = image_shape(conf)
        n_tokens, d = mods.code_shape
        values = v.frames * channels * size * size  # P
        tubelet = v.tubelet * channels * v.patch**2
        self.inp = pc.Input(
            x=TensorSpec((b, n_tokens, tubelet + 1), dtype),
            y=TensorSpec((b, values + n_tokens * d), dtype),
            period=tau.period,
            batch=b,
            device=dev,
        )
        spec, port = self.inp.act_spec, self.inp.next
        layers: list[tuple[str, torch.nn.Module]] = [
            (f"enc{i + 1}", Segmented(cast(Stage, stage), (n_tokens,)))
            for i, stage in enumerate(mods.vision.encoder.stages)
        ]
        parts = mods.decoder.parts(v.depth // v.nodes)
        decoder = [(f"dec{i + 1}", part) for i, part in enumerate(parts[:-1])]
        self.names = tuple(name for name, _ in layers + decoder)
        for name, module in layers:
            spec, port = self._layer(name, module, spec, port, conf, optimizer)
        self.latent = Latent(act_spec=spec, tau=tau, device=dev)
        port.connect(self.latent.prev)
        spec, port = self.latent.act_spec, self.latent.next
        for name, module in decoder:
            spec, port = self._layer(name, module, spec, port, conf, optimizer)
        self.out = GaussOut(
            parts[-1],
            mods.likelihood,
            act_spec=spec,
            tau=tau,
            optimizer=optimizer,
            learn_interval=t.learn_interval,
            device=dev,
        )
        port.connect(self.out.prev)
        self.prev, self.stats, self.kl = (
            self.inp.prev,
            self.out.stats,
            self.latent.stats,
        )

    def _layer(
        self,
        name: str,
        module: torch.nn.Module,
        spec: DictSpec[Activity],
        port: MultiPort[Activity, Tensor],
        conf: RunConf,
        optimizer: Optimizer,
    ) -> tuple[DictSpec[Activity], MultiPort[Activity, Tensor]]:
        """Make the ``pc.Layer`` ``name`` of ``module`` after ``port``, set it
        as an attribute (in the order of the tree), and return its spec and
        port."""
        node = pc.Layer(
            module,
            act_spec=spec,
            tau=self.tau,
            optimizer=optimizer,
            learn_interval=conf.time.learn_interval,
            device=conf.device,
            cuda_graph=conf.ctl.cuda_graph and conf.device.startswith("cuda"),
        )
        setattr(self, name, node)
        port.connect(node.prev)
        return node.act_spec, node.next

    def sample(
        self, x: Tensor, u: Tensor, eps: Tensor, mask: Tensor, reset: Tensor
    ) -> pc.msg.Sample:
        r"""The write of the stacks ``x`` u8 ``(B, N, C, H, W)`` with their
        dequantization's noise ``u`` (of their shape) and their code's
        ``eps`` ``(B, T, d)``, ``mask`` and ``reset`` ``(B,)``: ``x`` packed
        (``VisionModules.whole``), ``y`` the target $(x + u) / 256$ and
        ``eps``, each stack's flattened (module docstring)."""
        dtype = self.mods.log_sigma.dtype
        y = dequantize(x, u, LEVELS).flatten(1)
        return pc.msg.Sample(
            x=self.mods.vision.whole(x).to(dtype),
            y=torch.cat([y, eps.flatten(1)], 1).to(dtype),
            mask=mask,
            reset=reset,
        )


def build_vae_pc_net(conf: RunConf) -> VAEPC:
    r"""The VAE of ``conf`` learned by PC-ALM: the modules of ``build_vae``
    (seed ``conf.seed``, those of ``bp.build_vae_net``) with the optimizer
    of ``conf.optim_vision`` (``scale`` ``"fan_in"`` by the roles in all the
    modules, as there).

    Raises:
        ValueError: ``conf`` has no ``optim_vision``, its learner is not
            ``"pc"``, or ``build_vae`` or ``VAEPC`` raises.
    """
    if conf.optim_vision is None:
        raise ValueError(f"the run {conf.name!r} needs optim_vision")
    if conf.learner.kind != "pc":
        raise ValueError(f"VAEPC learns by learner=pc, not {conf.learner.kind!r}")
    mods = build_vae(conf, seed=conf.seed)
    optimizer = make_optimizer(
        conf.optim_vision, conf.time.frame, list(mods.named().values())
    )
    return VAEPC(mods, conf=conf, optimizer=optimizer)
