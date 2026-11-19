r"""The VAE of V0 learned by PC-ALM alone (``plans/v0-foundation.md``), as
M18's (``exp/video/vae/pc.py``) with a fixed node count and the Jacobi steps
of ``mujou.run_manually``.

``inp -> enc1 .. enc4 -> latent -> dec1 -> dec2 -> out``:

- ``inp`` (``pc.Input``): the host's ``pc.msg.Sample`` (``VAEPC.sample``),
  ``x`` the stacks as images ``(B, 3n, H, W)`` of $x / 255$
  (``models.early_fusion``) and ``y`` each stack's target $y = (x + u) / 256$
  ($P = 3nHW$ values) followed by its code's noise $\varepsilon$ ($D$
  values): the noise and the target travel with the activity, so ``latent``
  and ``out`` pair each activity with the noise of its own write, however
  the frames are staggered along the chain;
- ``enc1`` .. ``enc4`` (``pc.Layer``): the encoder's node modules, the last
  to $h_4 = (\mu, \ell)$ ``(B, 2D)``, $\ell = \ln s^2$;
- ``latent`` (``Latent``): the constraint $z = \mu + e^{\ell / 2} \odot
  \varepsilon$ and the KL of $q$ as a local energy;
- ``dec1``, ``dec2`` (``pc.Layer``): the decoder's node modules;
- ``out`` (``GaussOut``): the readout and the Gaussian likelihood of the
  learned $\ln \sigma$.

With $r_i = h_i - f_i(h_{i-1})$ the augmented Lagrangian is

$$\mathcal F = E_{\mathrm{out}}(h_7) + E_{\mathrm{KL}}(h_4) + \sum_i
\big(\lambda_i^\top r_i + \tfrac12 \lVert r_i \rVert^2\big),$$

$E_{\mathrm{out}} + E_{\mathrm{KL}} = B \cdot (-\mathcal L)$ at the forward
values, $-\mathcal L$ the mean negative ELBO over the stacks that count. At
its fixed point $\lambda_i = -\partial E / \partial h_i$ and every node's
$g_\theta / B$ is BP's $\partial(-\mathcal L) / \partial \theta$. ``latent``
and ``out`` weigh their energies by $B / n$ ($n$ the stacks whose ``mask``
is True), so that the mean is over the stacks that count.

When a node learns (``learn_interval``)
----------------------------------------

The host (``exp.lib.host.Host``) writes a frame every ``time.frame`` = 0.1 s
and steps the run by ``tau.period`` = $h / (1 + h / r)$ = 2 ms: 50 steps a
frame, each updating every node once from the values at the start of the
step (Jacobi). With ``pc.Layer`` and ``Latent`` relaxing before they compute
$\delta$ and $g_{\mathrm{in}}$, a write at $t_w$ goes as follows, step $s$
at $t_w + s \cdot$ period, the $j$-th node after ``inp`` (``enc1`` = 1 ..
``enc4`` = 4, ``latent`` = 5, ``dec1``, ``dec2`` = 6, 7, ``out`` = 8 =
``HOPS``):

1. ``inp`` sends the write in step 1, node $j$ receives it in step $j + 1$:
   ``out`` in step ``HOPS`` + 1 = 9, where it sends its exact credit back;
2. a node relaxes $\delta = r + \lambda$ to $-g$ in one update (the default
   time constants, $\tau_\lambda = \tau_r / 2 = \tau_h / 2$, steps of
   $\tau_h / 2$), so node $j$ holds BP's credit from step $9 + (8 - j)$ on:
   ``enc1`` last, from step $2 \cdot$ ``HOPS`` = 16 ($t_w$ + 32 ms), and it
   keeps it while the input stays (nothing learns in between);
3. ``LearnClock`` learns in the first update at or after $t_w + k \cdot$
   ``learn_interval``, $k = 1, 2, \ldots$, while the node still holds this
   write. Node $j$ holds it until it receives the next write, in step $j +
   1$ after $t_w$ + frame, i.e. through $t_w$ + frame + $j \cdot$ period.

So every node learns exactly once per frame, from BP's gradient, when

- ``learn_interval`` $\ge 2 \cdot$ ``HOPS`` $\cdot$ period = 32 ms (the
  credit has arrived; 2.);
- $2 \cdot$ ``learn_interval`` > frame + ``HOPS`` $\cdot$ period = 116 ms
  (the second point is past the last update of ``out``, the node that holds
  a write the longest; 3.), i.e. ``learn_interval`` > 58 ms;
- ``learn_interval`` $\le$ frame = 100 ms (``enc1``, $j$ = 1, learns before
  the next write reaches it; 3.). The bound is that of frame 0: the host
  steps once before its first write (``Host``), so that write is at $t_w$ =
  period and the next one comes frame - period after it; ``enc1`` holds it
  through frame - period + period = frame after $t_w$ (in a later frame,
  through frame + period).

``LEARN_INTERVAL`` = 90 ms, step 45 of the 50, is in the window $(58, 100]$
ms, late in the frame (29 steps after the credit is exact) and 5 steps from
the upper end. Between two learns of a node there is one frame, so in the
steady state ``elapsed`` is 0.1 s and ``SGD(tau)`` steps by $0.1 / \tau$ of
the gradient a frame. ``tests/vision/test_pc.py`` checks that every node
learns once per frame by BP's float64 gradient, and that at 30 ms (step 15)
``enc1`` does not. ``RunConf`` refuses a ``learn_interval`` outside the
window of its frame, time constants and hops
(``exp.vision.schema.run.check_learn_interval``); ``VAEPC`` itself takes any,
as these tests do.
"""

import math
from typing import TypedDict, override

import mujou.pc as pc
import torch
from mujou.pc.msg import Activity, rows
from mujou.pc.optim import Optimizer
from mujou.time import Timestamp
from mujou.types import Device
from torch import Tensor

import mujou
from exp.lib.elbo import LEVELS, dequantize, gauss_kl, gauss_nll
from exp.vision.models import LogSigma, VAEModules, build_vae, early_fusion
from exp.vision.schema.optim import OptimConf
from exp.vision.schema.run import RunConf
from mujou import DictSpec, MultiPort, Node, Port, TensorSpec, TimestampSpec

__all__ = [
    "HOPS",
    "KL_STATS",
    "LEARN_INTERVAL",
    "NLL_STATS",
    "VAEPC",
    "GaussOut",
    "KLStats",
    "Latent",
    "NLLStats",
    "build_net",
    "make_optimizer",
]

HOPS = 8  # inp -> enc1..4 -> latent -> dec1..2 -> out
LEARN_INTERVAL = 0.09  # module docstring: in (58, 100] ms at 0.1 s frames of 2 ms
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
    before it learned from it (0 when none counts), the mean squared error
    ``sq`` $\lVert y - \hat y \rVert^2$ of them, $\ln \sigma$ then, the
    number ``n`` of them (0-d float32), and its ``time``."""

    nll: Tensor
    sq: Tensor
    log_sigma: Tensor
    n: Tensor
    time: Timestamp


NLL_STATS = DictSpec(
    NLLStats,
    nll=_SCALAR,
    sq=_SCALAR,
    log_sigma=_SCALAR,
    n=_SCALAR,
    time=TimestampSpec(),
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
    ``(B, 2D)``, with the KL of $q = N(\mu, \mathrm{diag}\, e^\ell)$ to
    $N(0, I)$ as a local energy. $\varepsilon$ is the last $D$ values of each
    stack's ``y`` (the values before it are ``out``'s target), taken as 0 for
    a stack whose ``mask`` is False.

    It holds $r$ and $\lambda$ of $z$ ``(B, D)`` and updates as ``pc.Layer``
    does: it sets $r$ and $\lambda$ of the stacks to ``reset`` to 0 in the
    update that first sees an input, relaxes them by ``relax_`` with the
    credit $g$ of the next nodes and the capped ``dt``, and then, from the
    relaxed $\delta = r + \lambda$ (0 for a stack whose ``mask`` is False),
    sends back the gradient of its energy

    $$E = \sum_b \big(-f(h_b)^\top \delta_b\big) + \frac Bn \sum_{b:\,
    \mathrm{mask}_b} \mathrm{KL}(h_b),$$

    $g_{\mathrm{in}} = -J^\top \delta + \frac Bn \partial \mathrm{KL} /
    \partial h$, from one autograd of its own computation from the detached
    $h$, and sends $f(h)$ on with the ``y``, ``mask``, ``reset`` and ``time``
    it received. It has no parameters, so it never learns; it sends the mean
    KL on ``stats`` in every update.

    Args:
        act_spec: The spec of what ``enc4`` sends: ``h`` ``(B, 2D)``, ``y``
            ``(B, P + D)``.
        tau: The time constants of $r$ and $\lambda$.
        device: Where the node computes.

    Raises:
        ValueError: ``y`` holds fewer than $D$ values a stack.
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
        b, two_d = h.shape
        self._d = two_d // 2
        self._values = y.shape[1] - self._d  # P: the target's values
        if self._values < 0:
            raise ValueError(f"y of {y.shape[1]} values holds no noise of {self._d}")
        z = TensorSpec((b, self._d), h.dtype)
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
        g = sum(self.next.recv().values(), torch.zeros_like(self.r))
        pc.functional.relax_(self.r, self.lam, g, tick.dt, self._tau)
        valid = act["mask"]
        delta = torch.where(rows(valid, self.r), self.r + self.lam, 0.0)
        z, g_in, kl, n = self._terms(act["h"], act["y"], valid, delta)
        self.prev.send(g_in)
        self.next.send({**act, "h": z})
        # float32, whatever the messages' dtype
        self.stats.send({"kl": kl.float(), "n": n.float(), "time": act["time"]})

    def _terms(
        self, h: Tensor, y: Tensor, valid: Tensor, delta: Tensor
    ) -> list[Tensor]:
        """``[z, g_in, kl, n]`` (class docstring)."""
        d = self._d
        eps = y[:, self._values :]
        # a stack that does not count adds nothing, whatever its y (a NaN in
        # it would give 0 * NaN in the gradient), as pc.functional.loss_vjp
        eps = torch.where(rows(valid, eps), eps, 0.0)
        n, weight = _count(valid, h)
        with torch.enable_grad():
            leaf = h.detach().requires_grad_()
            mu, logvar = leaf[:, :d], leaf[:, d:]
            z = mu + (0.5 * logvar).exp() * eps
            kl = torch.where(valid, gauss_kl(mu, logvar).sum(1), 0.0)
            # -J^T delta from z (delta held fixed), as pc.functional.vjp's
            energy = (z * -delta).sum() + weight * kl.sum()
            # The gradient of this node's own energy, from a detached leaf.
            (g_in,) = torch.autograd.grad(energy, [leaf])  # mujou: local-autograd
        return [z.detach(), g_in, kl.detach().sum() / n.clamp(min=1), n]


class GaussOut(Node):
    r"""The output of the VAE as a node of PC-ALM: the decoder's readout
    ``module`` ($\hat y$ ``(B, 3n, H, W)`` of ``dec2``'s map) and the Gaussian
    likelihood of the learned $\ln \sigma$ (``likelihood``), whose energy is

    $$E = \frac Bn \sum_{b:\, \mathrm{mask}_b} -\log N(y_b; \hat y_b,
    \sigma^2 I)$$

    (``exp.lib.elbo.gauss_nll``), $y_b$ the first $P$ values of the stack's
    ``y`` (the target the host wrote with it). In every update it sends back
    $g_{\mathrm{in}} = \partial E / \partial h$ and the mean $-\log p(y \mid
    z)$ on ``stats``, from one autograd of its own computation from the
    detached $h$; when its clock says (as ``pc.Layer``'s) and a stack counts,
    it learns the readout's parameters and $\ln \sigma$ by $g_\theta / B$ of
    the same autograd, stepped by the clock's ``elapsed``.

    Args:
        module: The readout, on ``device``.
        likelihood: $\ln \sigma$, on ``device``.
        act_spec: The spec of what ``dec2`` sends.
        tau: The time constants (the period and ``max_dt``).
        optimizer: The optimizer of the readout's parameters and $\ln
            \sigma$.
        learn_interval: As in ``pc.clock.LearnClock``.
        device: Where the node computes.
    """

    def __init__(
        self,
        module: torch.nn.Module,
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
        self._params = [
            *module.parameters(),
            *(p for p in likelihood.parameters() if p.requires_grad),
        ]
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
        g_in, nll, sq, n, *g_theta = self._terms(learn, act["h"], act["y"], valid)
        self.prev.send(g_in)
        self.stats.send(  # float32, whatever the messages' dtype
            {
                "nll": nll.float(),
                "sq": sq.float(),
                "log_sigma": self.likelihood.log_sigma.detach().float(),
                "n": n.float(),
                "time": act["time"],
            }
        )
        if learn:
            # theta moves by dL/dtheta = (1 / B) dE/dtheta.
            self.optimizer.step_([g / self._b for g in g_theta], tick.elapsed)

    def _terms(self, learn: bool, h: Tensor, y: Tensor, valid: Tensor) -> list[Tensor]:
        """``[g_in, nll, sq, n, *g_theta]`` (``g_theta`` only when it learns)."""
        params = self._params if learn else []
        target = y[:, : self._values].reshape(self._shape)
        # as Latent's eps: a stack that does not count has no target
        target = torch.where(rows(valid, target), target, 0.0)
        n, weight = _count(valid, h)
        with torch.enable_grad():
            leaf = h.detach().requires_grad_()
            y_hat = self.module(leaf)
            nll = gauss_nll(target, y_hat, self.likelihood.log_sigma)
            nll = torch.where(valid, nll, 0.0)
            # The gradient of this node's own energy, from a detached leaf.
            g_in, *g_theta = torch.autograd.grad(  # mujou: local-autograd
                weight * nll.sum(), [leaf, *params], materialize_grads=True
            )
        sq = torch.where(
            valid, (target - y_hat.detach()).square().flatten(1).sum(1), 0.0
        )
        per = n.clamp(min=1)
        return [g_in, nll.detach().sum() / per, sq.sum() / per, n, *g_theta]


def make_optimizer(c: OptimConf) -> Optimizer:
    """The ``mujou.pc.optim`` optimizer of ``c`` (every time constant in
    seconds of virtual time)."""
    match c.kind:
        case "sgd":
            return pc.optim.SGD(tau=c.tau)
        case "sgd_momentum":
            return pc.optim.SGD(tau=c.tau, tau_m=c.tau_m)
        case "rmsprop":
            return pc.optim.RMSProp(tau=c.tau, tau_v=c.tau_v, eps=c.eps)
        case kind:
            assert kind == "adamw"  # OptimConf checks the choice
            return pc.optim.AdamW(
                tau=c.tau,
                tau_m=c.tau_m,
                tau_v=c.tau_v,
                eps=c.eps,
                weight_decay=c.weight_decay,
            )


class VAEPC(mujou.Network):
    r"""The VAE of ``conf`` in the nodes of PC-ALM (module docstring): the
    host writes ``sample(...)`` to ``prev`` and reads ``stats`` (``out``'s
    ``NLLStats``) and ``kl`` (``latent``'s ``KLStats``); $-\mathcal L$ is
    the sum of their ``nll`` and ``kl``.

    The ``pc.Layer`` nodes replay their updates from CUDA graphs with
    ``ctl.cuda_graph`` on CUDA (``latent`` and ``out`` do not). The messages
    are of the modules' dtype. ``optimizer`` is bound by each node to its own
    parameters.

    Args:
        mods: The modules, on ``conf.device``.
        conf: The run: its time, streams, ``ctl.cuda_graph`` and device.
        optimizer: The optimizer of every node.
    """

    def __init__(
        self, mods: VAEModules, *, conf: RunConf, optimizer: Optimizer
    ) -> None:
        super().__init__()
        t = conf.time
        self.mods = mods
        self.tau = tau = pc.TimeConstants(h=t.h, lam=t.lam, r=t.r)
        b, dtype, dev = conf.data.streams, mods.log_sigma.dtype, conf.device
        frames, size = conf.data.tours.frames, conf.data.tours.size
        values = frames * 3 * size * size  # P
        self.inp = pc.Input(
            x=TensorSpec((b, 3 * frames, size, size), dtype),
            y=TensorSpec((b, values + mods.latent), dtype),
            period=tau.period,
            batch=b,
            device=dev,
        )
        graph = conf.ctl.cuda_graph and conf.device.startswith("cuda")
        spec, port = self.inp.act_spec, self.inp.next
        encoder = [(f"enc{i + 1}", m) for i, m in enumerate(mods.encoder.parts)]
        decoder = [(f"dec{i + 1}", m) for i, m in enumerate(mods.decoder.parts)]
        self.names = tuple(name for name, _ in encoder + decoder)
        for name, module in encoder:
            spec, port = self._layer(name, module, spec, port, conf, optimizer, graph)
        self.latent = Latent(act_spec=spec, tau=tau, device=dev)
        port.connect(self.latent.prev)
        spec, port = self.latent.act_spec, self.latent.next
        for name, module in decoder:
            spec, port = self._layer(name, module, spec, port, conf, optimizer, graph)
        self.out = GaussOut(
            mods.decoder.readout,
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
        graph: bool,
    ) -> tuple[DictSpec[Activity], MultiPort[Activity, Tensor]]:
        """Make the ``pc.Layer`` ``name`` of ``module`` after ``port``, set it
        as an attribute, and return its spec and port."""
        node = pc.Layer(
            module,
            act_spec=spec,
            tau=self.tau,
            optimizer=optimizer,
            learn_interval=conf.time.learn_interval,
            device=conf.device,
            cuda_graph=graph,
        )
        setattr(self, name, node)
        port.connect(node.prev)
        return node.act_spec, node.next

    def sample(
        self, x: Tensor, u: Tensor, eps: Tensor, mask: Tensor, reset: Tensor
    ) -> pc.msg.Sample:
        r"""The write of the stacks ``x`` u8 ``(B, n, 3, H, W)`` with their
        dequantization's noise ``u`` (of their shape) and their code's
        ``eps`` ``(B, D)``, ``mask`` and ``reset`` ``(B,)``: ``x`` as images
        of $x / 255$, ``y`` the target $(x + u) / 256$ and ``eps``, each
        stack's flattened (module docstring)."""
        dtype = self.mods.log_sigma.dtype
        y = dequantize(x, u, LEVELS).flatten(1)
        return pc.msg.Sample(
            x=early_fusion(x).to(dtype),
            y=torch.cat([y, eps.flatten(1)], 1).to(dtype),
            mask=mask,
            reset=reset,
        )


def build_net(conf: RunConf) -> VAEPC:
    """The VAE of ``conf`` learned by PC-ALM: the modules of ``build_vae``
    (seed ``conf.seed``) with the optimizer of ``conf.optim``."""
    mods = build_vae(conf, seed=conf.seed)
    return VAEPC(mods, conf=conf, optimizer=make_optimizer(conf.optim))
