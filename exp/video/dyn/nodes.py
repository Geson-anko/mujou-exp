r"""The PC-ALM nodes of the dynamics, written with the public API of
``mujou.pc`` (``pc.functional``, ``pc.clock.LearnClock``, ``pc.msg``).

``Source`` shows the host's samples downstream; ``Dynamics`` predicts the
minGRU state $s_k$ of the frame from the remembered $x_{k-1}, s_{k-1}$;
``DynamicsLayer`` stacks a layer of the same frame on it; ``Compare`` reads
$\hat x_k = W_o y_L$ out and compares it with $x_k$. ``DynamicsBlock`` is
``Dynamics -> DynamicsLayer x (nodes - 1) -> Compare``, fed by one upstream
port that only sends.

No credit reaches the input of the dynamics, by the types of the connections
rather than by a promise to send zeros: ``Source.out`` only sends, and
``Dynamics.z`` and ``Compare.target`` only receive, so a port that sends
credit cannot be connected to them (``Port.connect`` raises ``TypeError``).
The input $x_{k-1}$ and the target $x_k$ are constants of the energy.

The nodes take one stream (the batch is ``()``) or ``B`` independent streams
side by side (``(B,)``, that of the ``act_spec`` they are given): each memory
below is then one per sample, and each sample resets, counts its predictions
and carries its credit on its own. A node sends the credit of each sample (the
gradient of that sample's energy), and learns from the mean over the batch
(``g_theta / B``, as ``pc.Layer`` does). Each node differentiates only its own
computation, from detached inputs, and sends the gradients of its energy back
as messages.

With a window (``DynamicsConf.window``, M16) the nodes learn as ``DynamicsBP``
does on windows, by episodes of PC-ALM over window-shaped messages
(``WindowActivity``, ``WindowMessage``) between neighbours: each node
differentiates only its window function (a ``scan`` of its minGRU over the
window, or the readout) from the detached windows it received, the open loop
of the window is solved by iterating the predictions through ``Compare ->
Dynamics``, and with ``feedback="grad"`` the credit of a fed-back prediction
goes back over the same port. At the equilibrium each node's
$g_\theta / (B P)$ is its part of the gradient of ``offline.window_loss``
(``DynamicsBlock``).
"""

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypedDict, cast, override

import mujou.pc as pc
import torch
from mujou.pc.msg import Activity, rows
from mujou.pc.optim import Optimizer
from mujou.time import Timestamp
from mujou.types import Device, Size
from torch import Tensor, nn

import mujou
from exp.lib.mingru import MinGRU, MinGRUTrace, ResidualMinGRU
from exp.lib.ring import WindowRing
from exp.video.dyn.modules import DynamicsModules
from exp.video.schema.dyn import DynamicsConf, WindowConf
from mujou import (
    DictSpec,
    FloatSpec,
    IntSpec,
    MultiPort,
    Node,
    Port,
    Stateful,
    TensorSpec,
    TimestampSpec,
)

__all__ = [
    "Compare",
    "Dynamics",
    "DynamicsBlock",
    "DynamicsLayer",
    "Source",
    "WindowActivity",
    "WindowMessage",
    "compare_energy",
    "is_open",
    "open_mask",
    "prediction_spec",
    "window_activity_spec",
    "window_energy",
    "window_message_spec",
]


def prediction_spec(x_spec: DictSpec[Activity]) -> DictSpec[pc.msg.Prediction]:
    r"""The spec of ``Compare.next``: $\hat x_k$, $x_k$, ``mask`` and ``time``."""
    x = x_spec.field("h", TensorSpec)
    return DictSpec(
        pc.msg.Prediction,
        y_hat=x,
        y=x,
        mask=x_spec.field("mask", TensorSpec),
        time=x_spec.field("time", TimestampSpec),
    )


def is_open(count: int, open_loop: float, cycle: int) -> bool:
    """Whether the ``count``-th prediction of a stream's video (1: the first,
    from its first frame) starts from the model's own prediction rather than
    the observed input: the last ``open_loop`` of every ``cycle`` predictions."""
    observed = round(cycle * (1 - open_loop))
    return open_loop > 0 and (count - 1) % cycle >= observed


def open_mask(counts: torch.Tensor, open_loop: float, cycle: int) -> torch.Tensor:
    """``is_open`` of each of ``counts`` (the number of predictions of each
    stream's video so far, ``(*batch,)``), as a bool tensor."""
    observed = round(cycle * (1 - open_loop))
    if open_loop <= 0:
        return torch.zeros_like(counts, dtype=torch.bool)
    return (counts - 1) % cycle >= observed


# --- The window messages ------------------------------------------------------


class WindowActivity(TypedDict):
    """The forward message of a window, ``Dynamics -> DynamicsLayer -> ...
    -> Compare``: the window ``h`` ``(B, P, d)`` of the sender's output
    ($S_1$ or $Y_n$, position $p$ for the prediction $p + 1$), the targets
    ``y`` ``(B, P, D)`` ($x_{p+1}$, passed on as they are), the ``mask``
    ``(B,)`` of the windows to learn from, the window time ``time`` $t_w$ (the
    write time of the frame the window is due at; ``Timestamp(ns=0)``: none
    yet), the ``elapsed`` of the window (``WindowRing.elapsed``) and the
    sender's revision ``rev`` of ``h`` (it grows by one whenever ``h`` or
    ``time`` changes)."""

    h: Tensor
    y: Tensor
    mask: Tensor
    time: Timestamp
    elapsed: float
    rev: int


class WindowMessage(TypedDict):
    r"""Every other message of a window: ``x`` ``(B, P, k)`` (a credit, the
    input window $C$ or the predictions $\hat X$), the window time ``time``
    $t_w$ it was computed in, and the sender's revision ``rev``."""

    x: Tensor
    time: Timestamp
    rev: int


def window_activity_spec(
    h: TensorSpec, y: TensorSpec, mask: TensorSpec
) -> DictSpec[WindowActivity]:
    """The spec of a ``WindowActivity`` of ``h`` ``(B, P, d)``, ``y`` ``(B,
    P, D)`` and ``mask`` ``(B,)``."""
    return DictSpec(
        WindowActivity,
        h=h,
        y=y,
        mask=mask,
        time=TimestampSpec(),
        elapsed=FloatSpec(),
        rev=IntSpec(),
    )


def window_message_spec(x: TensorSpec) -> DictSpec[WindowMessage]:
    """The spec of a ``WindowMessage`` of ``x``."""
    return DictSpec(WindowMessage, x=x, time=TimestampSpec(), rev=IntSpec())


def window_energy(
    readout: Callable[[Tensor], Tensor],
    y_top: Tensor,
    target: Tensor,
    mask: Tensor,
    base: Tensor | None = None,
) -> Tensor:
    r"""$m_b \sum_p \frac1D \lVert \hat X_b[p] - T_b[p] \rVert^2$ of each window
    ``(B,)``: the predictions $\hat X = W_o Y_L$ (``readout(y_top)``,
    ``y_top`` ``(B, P, d)``), or $\hat X = \mathrm{base} + W_o Y_L$ with a
    ``base`` ``(B, P, D)``, against ``target`` $T$ ``(B, P, D)``, 0 for the
    windows out of ``mask`` ``(B,)``. Its sum over the windows is $E = B P
    L_\text{win}$ of ``offline.window_loss``."""
    x_hat = readout(y_top) if base is None else base + readout(y_top)
    return _window_error(x_hat, target, mask)


def _window_error(x_hat: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """``window_energy`` of the predictions ``x_hat``."""
    return torch.where(mask, ((x_hat - target) ** 2).mean(-1).sum(-1), 0.0)


class _Outbox(Stateful):
    """The last window message a node sent on each of its window ports: it
    sends again only a message whose ``time`` or tensor (bit for bit) differs
    from the last, with the revision one more. A node's state, so that a
    resumed node sends what it would have."""

    def __init__(self, device: Device) -> None:
        self._device = device
        # port name -> (time_ns, rev, the tensor sent)
        self.sent: dict[str, tuple[int, int, Tensor]] = {}

    def send(
        self, name: str, port: Port[Any, Any], key: str, msg: Mapping[str, Any]
    ) -> None:
        """Send ``msg`` (with its ``rev`` set) on ``port`` (named ``name``)
        unless it is the last one sent there; ``key`` names the tensor
        compared."""
        x, time = msg[key], msg["time"].ns
        last = self.sent.get(name)
        if last is not None and last[0] == time and torch.equal(last[2], x):
            return
        rev = 0 if last is None else last[1] + 1
        port.send({**msg, "rev": rev})
        self.sent[name] = (time, rev, x)

    @override
    def state_dict(self) -> dict[str, Any]:
        return {
            name: {"time_ns": t, "rev": rev, "x": x}
            for name, (t, rev, x) in self.sent.items()
        }

    @override
    def load_state_dict(self, state: Mapping[str, Any], /) -> None:
        self.sent = {
            name: (int(s["time_ns"]), int(s["rev"]), s["x"].to(self._device, copy=True))
            for name, s in state.items()
        }


def _key(msg: Mapping[str, Any]) -> tuple[int, int]:
    """The window time and revision of a window message: what a node compares
    to know whether an input changed."""
    return msg["time"].ns, msg["rev"]


class _WindowNode(Node):
    r"""What the nodes of the dynamics share for learning on windows
    (``window``): the episode of a window time $t_w$, ``r_w`` and ``lam_w``
    of the output window, the recomputation only on a change, and the
    learning at $t_w$ + ``relax``.

    A node keeps ``episode`` (``t_ns`` $t_w$, ``at_ns`` $t_w$ + ``relax``,
    ``elapsed``, ``done``; None before the first), ``win_seen`` (the time
    and revision of each window input it read last), ``win_partial`` (whether
    the last update relaxed by less than ``max_dt``), ``win_changed_ns`` (the
    time of the last update whose inputs changed or that relaxed by less than
    ``max_dt``), ``win_out`` (``_Outbox``),
    the counters ``win_steps``, ``win_skipped``, ``win_unsettled`` and
    ``win_vjps``, and ``win_gsq`` ($\lVert g_\theta / (B P) \rVert^2$ of the
    last window learned, NaN before). All of it is state."""

    def _init_window(
        self,
        window: WindowConf,
        *,
        tau: pc.TimeConstants,
        batch: int,
        hops: int,
        out: TensorSpec | None,
        dtype: torch.dtype,
    ) -> None:
        """The state of the class docstring, ``r_w`` and ``lam_w`` of ``out``
        ``(B, P, d)`` (none for ``Compare``), and the settings; ``hops`` is
        the number of hops of a loop of the window messages (``nodes + 1``):
        a node that learns less than ``hops`` periods after its inputs last
        changed counts the window as unsettled."""
        self._window = window
        self._win_tau = tau
        self._max_dt = tau.max_dt
        self._win_batch = batch  # B
        self._positions = window.window - 1  # P
        self._relax_ns = round(window.relax * 1e9)
        self._settle_ns = hops * round(tau.max_dt * 1e9)
        dev = self.device
        if out is not None:
            self.r_w, self.lam_w = out.zeros(device=dev), out.zeros(device=dev)
        self.episode: dict[str, Any] | None = None
        self.register_state("episode")
        self.win_seen: tuple[tuple[int, int], ...] | None = None
        self.register_state("win_seen")
        self.win_partial = False
        self.register_state("win_partial")
        self.win_changed_ns = 0
        self.register_state("win_changed_ns")
        self.win_out = _Outbox(dev)
        self.win_steps, self.win_skipped = 0, 0
        self.win_unsettled, self.win_vjps = 0, 0
        for name in ("win_steps", "win_skipped", "win_unsettled", "win_vjps"):
            self.register_state(name)
        self.win_gsq = torch.full((), math.nan, dtype=dtype, device=dev)
        # The forward graph of the window function at the input it was built
        # from (key, leaf, output): reused while only the credit changes. Not
        # state: a resumed node builds it again, to the same values.
        self._fwd: tuple[Any, Tensor, Tensor] | None = None

    def _begin(self, t_ns: int, elapsed: float, now_ns: int) -> dict[str, Any]:
        """Open the episode of $t_w$ = ``t_ns``: the one before counts as
        skipped unless it is done; ``r_w`` and ``lam_w`` start at 0."""
        ep = self.episode
        if ep is not None and not ep["done"]:
            self.win_skipped += 1
        self.episode = {
            "t_ns": t_ns,
            "at_ns": t_ns + self._relax_ns,
            "elapsed": elapsed,
            "done": False,
        }
        if hasattr(self, "r_w"):
            self.r_w.zero_()
            self.lam_w.zero_()
        self.win_seen, self._fwd, self.win_changed_ns = None, None, now_ns
        return self.episode

    def _due(
        self, reads: Sequence[Mapping[str, Any]], dt: float, now_ns: int, learn: bool
    ) -> bool:
        r"""Whether this update computes: the inputs ``reads`` changed (their
        time or revision), it learns, or ``r_w`` and ``lam_w`` are not at the
        fixed point: this ``dt`` or the last falls short of ``max_dt`` (where
        one ``relax_`` does not reach $\delta_w = -g$, so the credit sent in
        that update is off until a full update relaxes again). Such an update
        counts as a change for ``win_unsettled``. ``Compare`` has no ``r_w``:
        its ``dt`` does not matter."""
        keys = tuple(_key(m) for m in reads)
        partial = hasattr(self, "r_w") and not self._full(dt)
        changed = keys != self.win_seen or self.win_partial or partial
        self.win_seen, self.win_partial = keys, partial
        if changed:
            self.win_changed_ns = now_ns
        return changed or learn

    def _full(self, dt: float) -> bool:
        """Whether the update moves by all of ``max_dt`` (to the rounding of
        the timer's ns)."""
        return dt >= self._max_dt * (1 - 1e-6)

    def _delta(self, g: Tensor, dt: float) -> Tensor:
        r"""Relax ``r_w`` and ``lam_w`` by the credit ``g`` of the output
        window and return $\delta_w = r_w + \lambda_w$.

        With the time constants of one-step relaxation (``_check_window``:
        $\tau_\lambda = \tau_r^2 / (\tau_h + \tau_r)$ and ``max_dt`` =
        ``period`` = $\tau_h \tau_r / (\tau_h + \tau_r)$; the defaults,
        $\tau_r = \tau_h$, $\tau_\lambda = \tau_h / 2$) and ``dt`` =
        ``max_dt``, one ``relax_`` moves $\delta_w$ to $-g$ whatever ``r_w``
        and ``lam_w`` were; they are then set to $(0, -g)$, the fixed point of
        ``relax_`` with ``g`` held, which has the same $\delta_w$ but does not
        round again in the next update, so that the messages settle bit for
        bit. A shorter ``dt`` leaves $\delta_w$ short of $-g$; ``_due``
        relaxes again in the next update."""
        pc.functional.relax_(self.r_w, self.lam_w, g, dt, self._win_tau)
        if self._full(dt):
            self.r_w.zero_()
            torch.neg(g, out=self.lam_w)
        return self.r_w + self.lam_w

    def _window_vjp(
        self,
        fn: Callable[[Tensor], Tensor],
        key: Any,
        x: Tensor,
        delta: Tensor,
        *,
        credit: bool,
        params: Sequence[Tensor],
    ) -> tuple[Tensor, Tensor | None, list[Tensor]]:
        r"""``(fn(x), g_in, g_theta)`` with $g_\text{in} = -J^\top \delta$
        (with ``credit``, else None) and $g_\theta = -J_\theta^\top \delta$
        for ``params``, as ``pc.functional.vjp`` gives them, from the detached
        ``x``. The forward graph is kept for the same ``key`` (the same input
        at the same weights) and reused while only ``delta`` changes."""
        if self._fwd is None or self._fwd[0] != key:
            with torch.enable_grad():
                leaf = x.detach().requires_grad_()
                self._fwd = (key, leaf, fn(leaf))
        _, leaf, out = self._fwd
        wrt = [leaf, *params] if credit else list(params)
        if not wrt:
            return out.detach(), None, []
        # The gradient of this node's own window function, from a detached
        # leaf: it stays inside the node.
        grads = torch.autograd.grad(  # mujou: local-autograd
            out,
            wrt,
            grad_outputs=-delta,
            retain_graph=not params,
            materialize_grads=True,
        )
        if not credit:
            return out.detach(), None, list(grads)
        return out.detach(), grads[0], list(grads[1:])

    def _learn(
        self,
        ep: dict[str, Any],
        g_theta: list[Tensor],
        ok: bool,
        now_ns: int,
    ) -> None:
        """End the episode: step the optimizer by ``g_theta / (B P)`` and the
        episode's ``elapsed`` if all the inputs read were of its $t_w$
        (``ok``), else count it skipped. The node sends nothing more of it."""
        ep["done"] = True
        self._fwd = None
        if not ok:
            self.win_skipped += 1
            return
        if now_ns - self.win_changed_ns < self._settle_ns:
            self.win_unsettled += 1
        n = self._win_batch * self._positions
        grads = [g / n for g in g_theta]
        with torch.no_grad():
            self.win_gsq.copy_(torch.stack([g.square().sum() for g in grads]).sum())
        self.optimizer.step_(grads, ep["elapsed"])
        self.win_steps += 1

    optimizer: pc.optim.BoundOptimizer  # bound in each node's __init__


# --- Source --------------------------------------------------------------------


class Source(Node):
    """Shows the host's samples downstream and takes no credit back.

    The host writes a ``pc.msg.Sample`` to ``prev``: ``x`` ``(*batch, *x)``,
    ``y`` unused zeros ``(*batch, 1)`` (the target of the dynamics is ``x``
    itself), ``mask`` and ``reset`` ``batch``. In each ``on_update`` the node
    sends from ``out`` an ``Activity`` whose ``h`` is ``x``, with the time the
    host wrote it, as ``pc.Input`` does. ``out`` only sends: it connects to any
    number of ports that only receive, and to none that sends credit back.

    Args:
        x: The spec of ``x``, ``(*batch, *x)``.
        period: The period of the node in seconds.
        batch: The batch ``x`` starts with: ``()`` or ``B``.
        device: Where the node computes.
    """

    def __init__(
        self, *, x: TensorSpec, period: float, batch: Size, device: Device = "cpu"
    ) -> None:
        super().__init__(device=device, period=period)
        y = TensorSpec((*pc.msg.batch_shape(batch), 1))
        self.act_spec = pc.msg.activity_spec(x, y, batch)
        mask = self.act_spec.field("mask", TensorSpec)
        self.prev = Port(
            recv=DictSpec(pc.msg.Sample, x=x, y=y, mask=mask, reset=mask)  # the host
        )
        self.out = Port(send=self.act_spec)

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        sample, written_at = self.prev.recv_with_time()
        self.out.send(
            {
                "h": sample["x"],
                "y": sample["y"],
                "mask": sample["mask"],
                "reset": sample["reset"],
                "time": Timestamp(ns=0) if written_at is None else written_at,
            }
        )


# --- Dynamics ------------------------------------------------------------------


class Dynamics(_WindowNode):
    r"""$s_k = \mathrm{minGRU}(x_{k-1}, s_{k-1})$: a node of PC-ALM whose input
    is the past.

    It receives $x$ on ``z`` (the observation in M14, an encoder's $z$ in
    M16), a port that only receives. In the update that first sees a new frame
    (a new ``time``), it remembers the $x$ it received last (that of the frame
    before) as $x_{k-1}$ and the ``mu`` it sent last as $s_{k-1}$, and sets
    ``r`` and ``lam`` to zero. Both are constants for the frame: the credit
    stops here. On the first frame of a video (``reset``), $s := 0$ and there
    is no $x_{k-1}$: it sends ``mu = 0`` with ``mask`` False, predicting
    nothing.

    Otherwise it relaxes ``r`` and ``lam`` by the ``g`` that the next nodes
    send and learns as ``pc.Layer`` does, from ``delta = r + lam``. The state
    carried to the next frame is ``mu``, not ``mu + r``.

    With ``open_loop`` > 0, it also receives ``Compare``'s prediction on
    ``z_hat`` and, for the predictions that ``is_open`` says, remembers its own
    last $\hat x_{k-1}$ instead of $x_{k-1}$: it learns to predict on from its
    own predictions, as a rollout runs. The target stays the observed $x_k$.

    With ``residual``, it also sends the $x_{k-1}$ it remembered on ``base``,
    for ``Compare`` to predict $\hat x_k = x_{k-1} + W_o y_L$.

    With ``trace``, it learns through time: in the update that first sees a
    frame, it moves the eligibility traces of its minGRU to $s_k$
    (``MinGRUTrace``, with the $x$ it remembered held) and learns by
    $-\delta^\top \partial s_k / \partial \theta$ (the $g_\theta$ of
    ``pc.functional.vjp``) with them, the gradient through $s_{k-1}, s_{k-2},
    \dots$ since the video began.

    With a batch of streams, ``has_prev`` and ``predictions`` (the count that
    ``is_open`` reads) are per stream, and a stream that resets does not touch
    the memory of the others.

    With ``window`` (``WindowConf``; the batch is then flattened to ``B``
    windows) it learns on windows instead of every frame, as ``DynamicsBP``
    does: in every update it predicts as above (forward only: no ``r``,
    ``lam``, traces or learning of the frame, nothing received on ``next``),
    and in the update that first sees a frame it pushes the frame's input into
    a ``WindowRing`` (``ring``). When a window is due with a stream in its
    mask, it opens the episode of the window (``DynamicsBlock``'s docstring):
    its window function is $F_1: C \mapsto S_1 = \mathrm{scan}(C, s_0 = 0)$
    on the input window $C$ ``(B, P, D)``: $C[p] = x_p$ for the ``observed``
    positions and the prediction $\hat X[p - 1]$ received on ``win_hat``
    after (0 until it comes). It sends $S_1$ with the targets $x_{1..P}$ on
    ``win_next``, $C$ on ``win_base`` (with ``residual``) and, with
    ``feedback="grad"``, the credit of the fed-back predictions on
    ``win_hat``: $\mathrm{FB}[q] = \bar a(C[q + 1])$, the credit of
    $C[q + 1]$ through $F_1$ ($g_C = -J^\top \delta_w$) and through the base
    of ``Compare`` (received on ``win_base``), for the open positions. The
    ring and ``win_masked`` (the streams out of the masks of the windows due)
    are state.
    """

    def __init__(
        self,
        module: MinGRU,
        *,
        act_spec: DictSpec[Activity],
        tau: pc.TimeConstants,
        optimizer: Optimizer,
        learn_interval: float | None,
        open_loop: float = 0.0,
        open_cycle: int = 20,
        residual: bool = False,
        trace: bool = False,
        window: WindowConf | None = None,
        hops: int = 2,
        device: Device = "cpu",
    ) -> None:
        super().__init__(device=device, period=tau.period)
        self._tau = tau
        self._open_loop, self._open_cycle = open_loop, open_cycle
        self._residual = residual
        self.module = module
        self.clock = pc.clock.LearnClock.of(
            tau, period=self.period, learn_interval=learn_interval
        )
        self._params = list(module.parameters())
        self.optimizer = optimizer.bind(self._params)
        batch = act_spec.field("mask", TensorSpec).shape
        self._samples = math.prod(batch)  # B
        self.trace = MinGRUTrace(module, tuple(batch)) if trace else None
        x = act_spec.field("h", TensorSpec)
        s = TensorSpec((*batch, module.gate.out_features), x.dtype)
        dev = self.device
        self.r, self.lam = s.zeros(device=dev), s.zeros(device=dev)
        self.z_prev, self.s_prev = x.zeros(device=dev), s.zeros(device=dev)  # kept
        self.z_last, self.mu = x.zeros(device=dev), s.zeros(device=dev)  # seen, sent
        # Per stream: whether z_prev is of this video, and the predictions of it.
        self.has_prev = torch.zeros(batch, dtype=torch.bool, device=dev)
        self.register_state("has_prev")
        self.predictions = torch.zeros(batch, dtype=torch.int64, device=dev)
        self.register_state("predictions")
        self.z_hat_last = x.zeros(device=dev)  # the prediction of the frame
        # Whether z_hat_last is the prediction of the frame z_last is of.
        self.z_hat_ok = torch.zeros(batch, dtype=torch.bool, device=dev)
        self.register_state("z_hat_ok")
        self.act_spec = pc.msg.activity_spec(s, act_spec.field("y", TensorSpec), batch)
        self.z = Port(recv=act_spec)  # receives only: no credit to the past
        self.next = MultiPort(send=self.act_spec, recv=s)
        if open_loop > 0:
            self.z_hat = Port(recv=prediction_spec(act_spec))
        if residual:  # the x_{k-1} it remembered, as the base of the prediction
            self.base = Port(send=act_spec)
        self._window = window
        if window is not None:
            self._init_dynamics_window(window, x, s, hops)

    def _init_dynamics_window(
        self, window: WindowConf, x: TensorSpec, s: TensorSpec, hops: int
    ) -> None:
        """The ring, the state and the window ports of ``window``."""
        n, w, dev = self._samples, window.window, self.device
        p = w - 1
        self._feedback = window.feedback
        # The positions of C that are observed (all with "teacher").
        self._observed = p if window.feedback == "teacher" else min(window.observed, p)
        h = TensorSpec((n, p, s.shape[-1]), x.dtype)
        frames = TensorSpec((n, p, x.shape[-1]), x.dtype)
        self._init_window(
            window, tau=self._tau, batch=n, hops=hops, out=h, dtype=x.dtype
        )
        self.ring = WindowRing(
            n, w, x.shape[-1], stride=window.stride, device=dev, dtype=x.dtype
        )
        self.win_masked = 0
        self.register_state("win_masked")
        grad = window.feedback == "grad"
        msg_h, msg_x = window_message_spec(h), window_message_spec(frames)
        self.win_spec = window_activity_spec(h, frames, TensorSpec((n,), torch.bool))
        self.win_next = Port(send=self.win_spec, recv=msg_h)
        if self._residual:  # C, the base; its credit back with "grad"
            self.win_base = Port(send=msg_x, recv=msg_x if grad else None)
        if window.feedback != "teacher":  # X^ in; with "grad" its credit back
            self.win_hat = Port(recv=msg_x, send=msg_x if grad else None)

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        if self._window is not None:
            self._update_on_windows(dt)
            return
        act = self.z.recv()
        tick = self.clock.tick(dt, act["time"], self.timer.now())
        if tick.new:
            self._new_frame(act["reset"])
        mask = act["mask"] & self.has_prev
        learn = tick.learn and bool(mask.any())
        if self._open_loop > 0:
            self._take_z_hat(act)
        if bool(self.has_prev.any()):
            delta = torch.where(rows(mask, self.r), self.r + self.lam, 0.0)
            s_prev = self.s_prev
            direct = learn and self.trace is None
            mu, _, g_theta = pc.functional.vjp(
                lambda z: self.module(z, s_prev),
                self.z_prev,
                lambda _: delta,
                self._params if direct else (),
            )
            if learn and self.trace is not None:
                # dE/dtheta = -J^T delta, as vjp's; the trace gives +J^T delta.
                g_theta = self.trace.grads(-delta)
            mu = torch.where(rows(self.has_prev, mu), mu, 0.0)
        else:
            mu, g_theta = torch.zeros_like(self.r), []
        g = sum(self.next.recv().values(), torch.zeros_like(self.r))
        pc.functional.relax_(self.r, self.lam, g, tick.dt, self._tau)
        self.mu = mu
        self.z_last = act["h"]
        self.next.send({**act, "h": mu, "mask": mask})
        if self._residual:
            self.base.send({**act, "h": self.z_prev, "mask": mask})
        if learn:
            # theta moves by the mean gradient of the batch, g_theta / B.
            self.optimizer.step_([g / self._samples for g in g_theta], tick.elapsed)

    def _take_z_hat(self, act: Activity) -> None:
        """Remember ``Compare``'s prediction of this frame, if it came."""
        p = self.z_hat.recv()
        if p["time"] == act["time"]:
            got = p["mask"]
            self.z_hat_last = torch.where(
                rows(got, p["y_hat"]), p["y_hat"], self.z_hat_last
            )
            self.z_hat_ok = self.z_hat_ok | got

    def _update_on_windows(self, dt: float) -> None:
        """``on_update`` with a window: the prediction of the frame (forward
        only), then the window's episode."""
        act = self.z.recv()
        tick = self.clock.tick(dt, act["time"], self.timer.now())
        if tick.new:
            self._new_frame(act["reset"])
        mask = act["mask"] & self.has_prev
        if self._open_loop > 0:
            self._take_z_hat(act)
        mu = torch.zeros_like(self.r)
        if bool(self.has_prev.any()):
            with torch.no_grad():
                mu = self.module(self.z_prev, self.s_prev)
            mu = torch.where(rows(self.has_prev, mu), mu, 0.0)
        self.mu = mu
        self.z_last = act["h"]
        self.next.send({**act, "h": mu, "mask": mask})
        if self._residual:
            self.base.send({**act, "h": self.z_prev, "mask": mask})
        due = self._push(act) if tick.new else None
        now = self.timer.now().ns
        ep = self.episode
        if ep is not None and not ep["done"]:
            # Not when the window due replaces it before its time: it is
            # skipped then.
            if due is None or now >= ep["at_ns"]:
                self._window_step(tick.dt, now)
        if due is not None:
            frames, wmask = due
            ep = self._begin(act["time"].ns, self.ring.elapsed(), now)
            ep["x"], ep["mask"] = frames, wmask
            self._window_step(tick.dt, now)

    def _push(self, act: Activity) -> tuple[Tensor, Tensor] | None:
        """Push the frame's input into the ring; the window (frames ``(B, W,
        D)`` and mask ``(B,)``) if one is due with a stream in its mask."""
        n = self._samples
        ring = self.ring
        ring.push_(act["h"].reshape(n, -1), act["time"].ns, act["reset"].reshape(n))
        if not ring.due():
            return None
        frames, mask = ring.window()
        self.win_masked += int((~mask).sum())
        return (frames, mask) if bool(mask.any()) else None

    def _window_step(self, dt: float, now: int) -> None:
        """One update of the episode: read the credits and the predictions,
        relax, compute $F_1$ and its credits if an input changed, send; learn
        at $t_w$ + ``relax``."""
        ep = self.episode
        assert ep is not None
        t, obs, p = ep["t_ns"], self._observed, self._positions
        learn = now >= ep["at_ns"]
        g_msg = self.win_next.recv()
        hat = self.win_hat.recv() if self._feedback != "teacher" else None
        grad = self._feedback == "grad"
        g_base = self.win_base.recv() if grad and self._residual else None
        reads = [m for m in (g_msg, hat, g_base) if m is not None]
        if not self._due(reads, dt, now, learn):
            return
        self.win_vjps += 1
        g = g_msg["x"] if g_msg["time"].ns == t else torch.zeros_like(self.r_w)
        delta = self._delta(g, dt)
        x = ep["x"]
        c = x[:, :p].clone()
        got = hat is not None and hat["time"].ns == t
        if hat is not None:  # the open positions: the predictions fed back
            c[:, obs:] = hat["x"][:, obs - 1 : p - 1] if got else 0.0
        key = (t, _key(hat) if got and hat is not None else None)
        zero = c.new_zeros(c.shape[0], self.r_w.shape[-1])
        s1, g_c, g_theta = self._window_vjp(
            lambda c: self.module.scan(c, zero),
            key,
            c,
            delta,
            credit=grad,
            params=self._params if learn else [],
        )
        time = Timestamp(ns=t)
        self.win_out.send(
            "win_next",
            self.win_next,
            "h",
            {
                "h": s1,
                "y": x[:, 1:],
                "mask": ep["mask"],
                "time": time,
                "elapsed": ep["elapsed"],
            },
        )
        if self._residual:
            self.win_out.send("win_base", self.win_base, "x", {"x": c, "time": time})
        if grad:
            assert g_c is not None
            a_c = g_c[:, obs:]  # the credit of the fed-back C[p], p >= obs
            if g_base is not None and g_base["time"].ns == t:
                a_c = a_c + g_base["x"][:, obs:]
            fb = torch.zeros_like(c)
            fb[:, obs - 1 : p - 1] = a_c
            self.win_out.send("win_hat", self.win_hat, "x", {"x": fb, "time": time})
        if learn:
            ok = all(m["time"].ns == t for m in reads)
            self._learn(ep, g_theta, ok, now)

    def _new_frame(self, reset: torch.Tensor) -> None:
        self.r.zero_()
        self.lam.zero_()
        # The streams that reset start a new video: s := 0, and no x before.
        predictions = torch.where(self.has_prev, self.predictions + 1, 1)
        # Its own prediction of the frame before, if it came in that frame.
        opened = self.z_hat_ok & open_mask(
            predictions, self._open_loop, self._open_cycle
        )
        z_prev = torch.where(rows(opened, self.z_last), self.z_hat_last, self.z_last)
        self.z_prev = torch.where(rows(reset, z_prev), self.z_prev, z_prev)
        self.mu = torch.where(rows(reset, self.mu), 0.0, self.mu)
        self.s_prev = self.mu
        if self.trace is not None:  # to s_k; a video that starts has none
            self.trace.step_(self.z_prev, self.s_prev)
            self.trace.reset_(reset)
        self.has_prev = ~reset
        self.predictions = predictions.masked_fill(reset, 0)
        self.z_hat_ok = torch.zeros_like(self.z_hat_ok)


class DynamicsLayer(_WindowNode):
    r"""$y_k = x_k + h_k$ with $h_k = \mathrm{minGRU}(\mathrm{GELU}(x_k),
    h_{k-1})$ (``ResidualMinGRU``): a node of PC-ALM stacked on ``Dynamics`` or
    on another of it, whose input $x_k$ is of the same frame.

    In the update that first sees a new frame (a new ``time`` of its input), it
    remembers the state $h$ it computed last as $h_{k-1}$ (0 on the first frame
    of a video, ``reset``) and sets ``r`` and ``lam`` to zero. From the second
    frame of a video on, it sends $y_k$; while the frame has a target
    (``mask``), it also sends the credit of $x_k$ back on ``prev`` (``g_in``:
    the input is of this frame, so the credit goes on), relaxes ``r`` and
    ``lam`` by the ``g`` from ``next`` and learns as ``pc.Layer`` does, from
    ``delta = r + lam``. On the first frame, it passes $x_k$ (0) on and keeps
    its state at 0. With a batch of streams, ``first`` is per stream. With
    ``trace``, it learns through time as ``Dynamics`` does, by the traces of
    its cell on $\mathrm{GELU}(x_k)$ (as $\partial y_k / \partial \theta =
    \partial h_k / \partial \theta$), moved when it first sees the frame.

    With ``window`` (and ``win_spec``, the ``WindowActivity`` of the node
    below), it predicts every frame forward only, as ``Dynamics`` does, and
    takes part in the episodes of the windows: its window function is $F_n:
    Y_{n-1} \mapsto Y_n = \mathrm{scan}(Y_{n-1}, h_0 = 0)$ on the window
    received on ``win_prev``; it sends $Y_n$ on ``win_next`` (the targets,
    mask, time and elapsed passed on) and the credit $g_\text{in} =
    -J^\top \delta_w$ of $Y_{n-1}$ back on ``win_prev``.
    """

    def __init__(
        self,
        module: ResidualMinGRU,
        *,
        act_spec: DictSpec[Activity],
        tau: pc.TimeConstants,
        optimizer: Optimizer,
        learn_interval: float | None,
        trace: bool = False,
        window: WindowConf | None = None,
        win_spec: DictSpec[WindowActivity] | None = None,
        hops: int = 2,
        device: Device = "cpu",
    ) -> None:
        super().__init__(device=device, period=tau.period)
        self._tau = tau
        self.module = module
        self.clock = pc.clock.LearnClock.of(
            tau, period=self.period, learn_interval=learn_interval
        )
        self._params = list(module.parameters())
        self.optimizer = optimizer.bind(self._params)
        s = act_spec.field("h", TensorSpec)
        batch = act_spec.field("mask", TensorSpec).shape
        self._samples = math.prod(batch)  # B
        self.trace = MinGRUTrace(module.cell, tuple(batch)) if trace else None
        dev = self.device
        self.r, self.lam = s.zeros(device=dev), s.zeros(device=dev)
        self.h_prev, self.h_last = s.zeros(device=dev), s.zeros(device=dev)
        # Per stream: the first frame of a video, with nothing to predict from.
        self.first = torch.ones(batch, dtype=torch.bool, device=dev)
        self.register_state("first")
        self.act_spec = act_spec
        self.prev = Port(send=s, recv=act_spec)
        self.next = MultiPort(send=act_spec, recv=s)
        self._window = window
        if window is not None:
            if win_spec is None:
                raise ValueError("a DynamicsLayer with a window needs its win_spec")
            h = win_spec.field("h", TensorSpec)
            self._init_window(
                window, tau=tau, batch=h.shape[0], hops=hops, out=h, dtype=h.dtype
            )
            self.win_spec = win_spec
            msg = window_message_spec(h)
            self.win_prev = Port(send=msg, recv=win_spec)
            self.win_next = Port(send=win_spec, recv=msg)

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        if self._window is not None:
            self._update_on_windows(dt)
            return
        act = self.prev.recv()
        tick = self.clock.tick(dt, act["time"], self.timer.now())
        if tick.new:
            self.r.zero_()
            self.lam.zero_()
            self.first = act["reset"]
            self.h_last = torch.where(rows(self.first, self.h_last), 0.0, self.h_last)
            self.h_prev = self.h_last
            if self.trace is not None:
                self.trace.step_(nn.functional.gelu(act["h"]), self.h_prev)
                self.trace.reset_(self.first)
        x = act["h"]
        mask = act["mask"] & ~self.first
        learn = tick.learn and bool(mask.any())
        if bool((~self.first).any()):
            delta = torch.where(rows(mask, self.r), self.r + self.lam, 0.0)
            h_prev = self.h_prev
            direct = learn and self.trace is None
            mu, g_in, g_theta = pc.functional.vjp(
                lambda x: self.module(x, h_prev),
                x,
                lambda _: delta,
                self._params if direct else (),
            )
            if learn and self.trace is not None:
                # dE/dtheta = -J^T delta, as vjp's; the trace gives +J^T delta.
                g_theta = self.trace.grads(-delta)
            first = rows(self.first, x)
            mu = torch.where(first, x, mu)  # the first frame passes x on
            self.h_last = torch.where(first, self.h_last, mu - x)
        else:
            mu, g_in, g_theta = x, torch.zeros_like(x), []
        g = sum(self.next.recv().values(), torch.zeros_like(self.r))
        pc.functional.relax_(self.r, self.lam, g, tick.dt, self._tau)
        self.prev.send(g_in)
        self.next.send({**act, "h": mu})
        if learn:
            self.optimizer.step_([g / self._samples for g in g_theta], tick.elapsed)

    def _update_on_windows(self, dt: float) -> None:
        """``on_update`` with a window: the output of the frame (forward
        only), then the window's episode."""
        act = self.prev.recv()
        tick = self.clock.tick(dt, act["time"], self.timer.now())
        if tick.new:
            self.first = act["reset"]
            self.h_last = torch.where(rows(self.first, self.h_last), 0.0, self.h_last)
            self.h_prev = self.h_last
        x = act["h"]
        mu = x
        if bool((~self.first).any()):
            with torch.no_grad():
                mu = self.module(x, self.h_prev)
            first = rows(self.first, x)
            mu = torch.where(first, x, mu)  # the first frame passes x on
            self.h_last = torch.where(first, self.h_last, mu - x)
        self.next.send({**act, "h": mu})
        self._window_step(tick.dt)

    def _window_step(self, dt: float) -> None:
        """One update of the episode of the window received on ``win_prev``
        (a new $t_w$ opens one): relax by the credit from above, compute
        $F_n$ and its credit if an input changed, send; learn at $t_w$ +
        ``relax``. Once it learned, it reads only ``win_prev``, for the next
        $t_w$."""
        now = self.timer.now().ns
        fwd = self.win_prev.recv()
        t = fwd["time"].ns
        ep = self.episode
        if t != 0 and (ep is None or ep["t_ns"] != t):
            ep = self._begin(t, fwd["elapsed"], now)
        if ep is None or ep["done"]:
            return
        learn = now >= ep["at_ns"]
        g_msg = self.win_next.recv()
        if not self._due([fwd, g_msg], dt, now, learn):
            return
        self.win_vjps += 1
        g = g_msg["x"] if g_msg["time"].ns == t else torch.zeros_like(self.r_w)
        delta = self._delta(g, dt)
        zero = self.r_w.new_zeros(self.r_w.shape[0], self.r_w.shape[-1])
        y, g_in, g_theta = self._window_vjp(
            lambda y: self.module.scan(y, zero),
            _key(fwd),
            fwd["h"],
            delta,
            credit=True,
            params=self._params if learn else [],
        )
        assert g_in is not None
        time = Timestamp(ns=t)
        self.win_out.send("win_next", self.win_next, "h", {**fwd, "h": y, "time": time})
        self.win_out.send("win_prev", self.win_prev, "x", {"x": g_in, "time": time})
        if learn:
            self._learn(ep, g_theta, g_msg["time"].ns == t, now)


# --- Compare -------------------------------------------------------------------


def compare_energy(
    readout: Callable[[torch.Tensor], torch.Tensor],
    s: torch.Tensor,
    target: torch.Tensor,
    base: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""$\frac{1}{D} \lVert \hat x - x \rVert^2$ of each sample, ``(*batch,)``:
    the prediction $\hat x = W_o s$ (``readout(s)``, ``s`` ``(*batch, d_s)``),
    or $\hat x = b + W_o s$ from a ``base`` $b$ ``(*batch, D)`` (the residual
    prediction), against ``target`` $x$ ``(*batch, D)``, averaged over the last
    axis."""
    x_hat = readout(s) if base is None else base + readout(s)
    return ((x_hat - target) ** 2).mean(-1)


class Compare(_WindowNode):
    r"""The output node of the prediction: $\hat x_k = W_o y_k$ against $x_k$.

    Its energy is ``compare_energy`` of the top output $y_k$ received on ``s``
    and the target $x_k$ received on ``target``; it sends its gradient with
    respect to $y_k$ back on ``s`` and learns $W_o$ by the one with respect to
    $W_o$, when both inputs are of the same frame and the dynamics predicted
    (their ``mask``). ``target`` only receives: the target is a constant of the
    energy. With ``residual``, it predicts $\hat x_k = x_{k-1} + W_o y_k$ from
    the $x_{k-1}$ that ``Dynamics`` sends on ``base`` (a constant too). It sends
    the prediction ``y_hat`` $= \hat x_k$, the target ``y`` $= x_k$, their
    ``mask`` and ``time`` (a ``pc.msg.Prediction``) on ``next``, in every
    update whose inputs are of the same frame (none while ``s`` is still of
    the frame before), and the prediction error $\frac{1}{D} \lVert \hat x_k -
    x_k \rVert^2$ on ``error``. The first prediction of a frame on ``next`` is
    the prequential one; the later ones are of the weights as they learn (what
    ``Dynamics`` remembers with ``open_loop``). ``error`` is sent once a frame,
    from the first prediction: the prequential error.

    With a batch of streams, the energy is the sum of the samples that have a
    prediction (so each sample gets the gradient of its own energy), $W_o$
    learns from the mean over the batch, and ``error`` is the mean over the
    samples that have one.

    With ``window`` (and ``win_spec``, the ``WindowActivity`` of the top node),
    it predicts every frame as above but learns nothing from it and sends no
    credit on ``s``; it takes part in the episodes of the windows instead:
    from the window $Y_L$ received on ``win_s`` (and the base $C$ on
    ``win_base`` with ``residual``; nothing is computed before it is of the
    same $t_w$), it predicts $\hat X = \mathrm{base} + W_o Y_L$, sends it on
    ``win_next`` (for ``Dynamics`` to feed back; not with ``"teacher"``) and
    sends back the gradients of

    $$
    \tilde E_C = \sum_b \texttt{window\_energy}_b + \sum_{b,p}
    \mathrm{FB}_b[p] \cdot \hat X_b[p]
    $$

    (FB, the credit of the fed-back predictions, received on ``win_next``
    with ``"grad"``; 0 otherwise): $\partial \tilde E_C / \partial Y_L$ on
    ``win_s`` and, with ``"grad"`` and ``residual``, $\partial \tilde E_C /
    \partial\,\mathrm{base}$ on ``win_base``. It is the output node: no
    ``r_w`` or ``lam_w``. ``win_loss`` is $E_C / (B P)$ of the last window it
    learned, at the weights before (a 0-d tensor, NaN before the first).
    """

    def __init__(
        self,
        readout: nn.Module,
        *,
        s_spec: DictSpec[Activity],
        target_spec: DictSpec[Activity],
        tau: pc.TimeConstants,
        optimizer: Optimizer,
        learn_interval: float | None,
        residual: bool = False,
        window: WindowConf | None = None,
        win_spec: DictSpec[WindowActivity] | None = None,
        hops: int = 2,
        device: Device = "cpu",
    ) -> None:
        super().__init__(device=device, period=tau.period)
        self.module = readout
        self._residual = residual
        self.clock = pc.clock.LearnClock.of(
            tau, period=self.period, learn_interval=learn_interval
        )
        self._params = list(readout.parameters())
        self.optimizer = optimizer.bind(self._params)
        self._samples = math.prod(target_spec.field("mask", TensorSpec).shape)  # B
        self.s = Port(send=s_spec.field("h", TensorSpec), recv=s_spec)
        self.target = Port(recv=target_spec)  # receives only: a constant
        self.next = Port(send=prediction_spec(target_spec))
        if residual:  # the x_{k-1} that Dynamics remembered
            self.base = Port(recv=target_spec)
        dtype = target_spec.field("h", TensorSpec).dtype
        self.error = Port(
            send=TensorSpec((), dtype), initial=torch.tensor(math.nan, dtype=dtype)
        )
        self.error_sent = False  # the error of the frame of the target was sent
        self.register_state("error_sent")
        self._window = window
        if window is not None:
            if win_spec is None:
                raise ValueError("a Compare with a window needs its win_spec")
            self._init_compare_window(window, win_spec, tau, hops)

    def _init_compare_window(
        self,
        window: WindowConf,
        win_spec: DictSpec[WindowActivity],
        tau: pc.TimeConstants,
        hops: int,
    ) -> None:
        """The state and the window ports of ``window``."""
        h, y = win_spec.field("h", TensorSpec), win_spec.field("y", TensorSpec)
        self._init_window(
            window, tau=tau, batch=h.shape[0], hops=hops, out=None, dtype=h.dtype
        )
        self._feedback = window.feedback
        grad = window.feedback == "grad"
        msg_x = window_message_spec(y)
        self.win_s = Port(send=window_message_spec(h), recv=win_spec)
        if self._residual:  # C, the base; with "grad" its credit back
            self.win_base = Port(recv=msg_x, send=msg_x if grad else None)
        if window.feedback != "teacher":  # X^ to feed back; its credit back
            self.win_next = Port(send=msg_x, recv=msg_x if grad else None)
        self.win_loss = torch.full((), math.nan, dtype=h.dtype, device=self.device)

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        if self._window is not None:
            self._update_on_windows(dt)
            return
        s, x = self.s.recv(), self.target.recv()
        tick = self.clock.tick(dt, x["time"], self.timer.now())
        if tick.new:
            self.error_sent = False
        synced = s["time"] == x["time"]  # the inputs are of the same frame
        mask = s["mask"] & x["mask"]
        base = None
        if self._residual:
            b = self.base.recv()
            synced = synced and b["time"] == x["time"]
            mask = mask & b["mask"]
            base = b["h"]
        if not synced:
            mask = torch.zeros_like(mask)
        any_mask = bool(mask.any())
        learn = tick.learn and any_mask
        with torch.no_grad():
            y_hat = self.module(s["h"])
            if base is not None:
                y_hat = base + y_hat
        if any_mask:
            g_s, g_theta = self._vjp(s["h"], x["h"], mask, learn, base)
            self.s.send(g_s)
        else:
            self.s.send(torch.zeros_like(s["h"]))
            g_theta = []
        if synced:  # no prediction of x_k from the s of another frame
            y = x["h"]
            self.next.send({"y_hat": y_hat, "y": y, "mask": mask, "time": x["time"]})
        if any_mask and not self.error_sent:  # the prequential error, once
            self.error.send(((y_hat - x["h"]) ** 2)[mask].mean())
            self.error_sent = True
        if learn:
            self.optimizer.step_([g / self._samples for g in g_theta], tick.elapsed)

    def _update_on_windows(self, dt: float) -> None:
        """``on_update`` with a window: the prequential prediction and error
        of the frame (no credit, no learning), then the window's episode."""
        s, x = self.s.recv(), self.target.recv()
        tick = self.clock.tick(dt, x["time"], self.timer.now())
        if tick.new:
            self.error_sent = False
        synced = s["time"] == x["time"]
        mask = s["mask"] & x["mask"]
        base = None
        if self._residual:
            b = self.base.recv()
            synced = synced and b["time"] == x["time"]
            mask = mask & b["mask"]
            base = b["h"]
        if not synced:
            mask = torch.zeros_like(mask)
        with torch.no_grad():
            y_hat = self.module(s["h"])
            if base is not None:
                y_hat = base + y_hat
        if synced:
            self.next.send(
                {"y_hat": y_hat, "y": x["h"], "mask": mask, "time": x["time"]}
            )
        if bool(mask.any()) and not self.error_sent:
            self.error.send(((y_hat - x["h"]) ** 2)[mask].mean())
            self.error_sent = True
        self._window_step(tick.dt)

    def _window_step(self, dt: float) -> None:
        r"""One update of the episode of the window received on ``win_s``:
        compute $\hat X$ and the gradients of $\tilde E_C$ if an input
        changed, send; learn at $t_w$ + ``relax``."""
        now = self.timer.now().ns
        fwd = self.win_s.recv()
        t = fwd["time"].ns
        ep = self.episode
        if t != 0 and (ep is None or ep["t_ns"] != t):
            ep = self._begin(t, fwd["elapsed"], now)
        if ep is None or ep["done"]:
            return
        learn = now >= ep["at_ns"]
        grad = self._feedback == "grad"
        base = self.win_base.recv() if self._residual else None
        fb = self.win_next.recv() if grad else None
        reads = [m for m in (fwd, base, fb) if m is not None]
        if not self._due(reads, dt, now, learn):
            return
        if base is not None and base["time"].ns != t:  # nothing before the base
            if learn:
                self._learn(ep, [], False, now)
            return
        self.win_vjps += 1
        params = self._params if learn else []
        with torch.enable_grad():
            y = fwd["h"].detach().requires_grad_()
            leaves = [y]
            b = None if base is None else base["x"].detach()
            if b is not None and grad:
                leaves.append(b.requires_grad_())
            x_hat = self.module(y) if b is None else b + self.module(y)
            energy = _window_error(x_hat, fwd["y"], fwd["mask"]).sum()
            total = energy
            if fb is not None and fb["time"].ns == t:  # the credit fed back
                total = total + (fb["x"] * x_hat).sum()
            # The gradients of this node's own energy, from detached leaves.
            grads = torch.autograd.grad(  # mujou: local-autograd
                total, [*leaves, *params], materialize_grads=True
            )
        time = Timestamp(ns=t)
        self.win_out.send("win_s", self.win_s, "x", {"x": grads[0], "time": time})
        if len(leaves) > 1:
            g_base = {"x": grads[1], "time": time}
            self.win_out.send("win_base", self.win_base, "x", g_base)
        if self._feedback != "teacher":
            x_hat = x_hat.detach()
            self.win_out.send(
                "win_next", self.win_next, "x", {"x": x_hat, "time": time}
            )
        if learn:
            ok = all(m["time"].ns == t for m in reads)
            if ok:
                with torch.no_grad():
                    self.win_loss.copy_(energy / (self._win_batch * self._positions))
            self._learn(ep, list(grads[len(leaves) :]), ok, now)

    def _vjp(
        self,
        s: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        learn: bool,
        base: torch.Tensor | None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        params = self._params if learn else []
        with torch.enable_grad():
            s = s.detach().requires_grad_()
            energy = compare_energy(self.module, s, target, base)
            energy = torch.where(mask, energy, 0.0).sum()
            # The gradient of this node's own energy, from a detached leaf.
            g_s, *g_theta = torch.autograd.grad(  # mujou: local-autograd
                energy, [s, *params]
            )
        return g_s, g_theta


# --- The block ----------------------------------------------------------------


def _check_window(window: WindowConf, tau: pc.TimeConstants) -> None:
    """Refuse the windows the nodes cannot learn (``DynamicsBlock``)."""
    if window.credit == "own":
        raise ValueError(
            "dynamics.window.credit='own' is an ablation of BP only: "
            "learner=pc learns with credit='bptt'"
        )
    # One explicit step of dt = period = h r / (h + r) moves r to -(r / (h +
    # r)) (lam + g), and lam + r to -g from any (r, lam) iff lam = r^2 / (h +
    # r) (inf without the decay); max_dt must be that period (2 r >= h).
    one_step = tau.r**2 / (tau.h + tau.r) if math.isfinite(tau.r) else math.inf
    if not math.isclose(tau.lam, one_step, rel_tol=1e-9) or not math.isclose(
        tau.max_dt, tau.period, rel_tol=1e-12
    ):
        raise ValueError(
            "the PC-ALM window learning needs the time constants of one-step "
            f"relaxation, time.lam = r^2 / (h + r) = {one_step} (not {tau.lam}) "
            "with max_dt = period (2 r >= h)"
        )


class DynamicsBlock(mujou.Network):
    r"""``Dynamics -> DynamicsLayer x (conf.nodes - 1) -> Compare`` of the
    modules ``mods``, fed by one upstream port that only sends.

    The nodes are ``dynamics``, ``dynamics2``, ... (``names``) and
    ``compare``. ``connect(upstream)`` connects the upstream to
    ``dynamics.z`` (the input) and to ``compare.target`` (the target); with
    ``conf.open_loop`` > 0, ``compare.next`` also feeds ``dynamics.z_hat``,
    and with ``conf.residual``, ``dynamics.base`` feeds ``compare.base``.
    ``next`` is ``compare.next`` (the prequential prediction) and ``error``
    ``compare.error``. Every node binds its own copy of ``optimizer``.

    With ``conf.window`` (``WindowConf``) the nodes learn on windows: the
    rule of ``DynamicsBP`` (the same windows, mask, times and elapsed; the
    gradient of ``offline.window_loss`` with ``credit="bptt"``, normalized by
    $B P$), distributed over the nodes. The window ports are wired
    ``dynamics.win_next -> dynamics2.win_prev -> ... -> compare.win_s``,
    ``dynamics.win_base <-> compare.win_base`` (``residual``) and
    ``compare.win_next <-> dynamics.win_hat`` (not ``"teacher"``); they carry
    windows ``(B, P, .)`` and connect to nothing outside the block (their
    specs are no ``Activity``). The episode of a window, by the local rules of
    each node:

    - ``Dynamics`` opens it when its ring has a window due with a stream in
      its mask, at $t_w$ (the write time of the frame); the other nodes, when
      the window on their forward input has a new $t_w$. Each sets ``r_w`` and
      ``lam_w`` to 0, and counts the episode before as skipped
      (``win_skipped``) unless it ended.
    - In each update a node reads its window inputs, relaxes ``r_w`` and
      ``lam_w`` by the credit of its output window (with the default time
      constants one update reaches $\delta_w = -g$; ``_WindowNode._delta``),
      computes its window function and its credits from the detached inputs,
      and sends them; only when an input changed (its ``time`` or ``rev``),
      and it sends only what changed. An input of another $t_w$ is not there
      yet: 0. The credit graph is triangular in time, so the messages settle,
      bit for bit, after $(L + 1) K + L$ updates of the tree order ($K + L$
      with ``"detach"``, $L$ with ``"teacher"``; $K = W - 1 - O$, $L$ =
      ``conf.nodes``): 79 updates (0.158 s at a period of 2 ms) for the
      default window; ``relax`` (0.3 s) leaves room for it.
    - At the first update at or after $t_w$ + ``relax`` it learns: the step of
      its optimizer by $g_\theta / (B P)$ (at the equilibrium the matching
      part of BP's gradient of $L_\text{win}$) and the window's elapsed, if
      every window input it read is of $t_w$ (``win_steps``; else
      ``win_skipped``), counting it in ``win_unsettled`` when an input changed
      (or an update relaxed by less than ``max_dt``) less than $L + 1$
      periods before. That bound is a loop of the messages in the tree order
      or with the nodes' periods within a factor of 2; with a node much slower
      than the others, a window learned before the messages settled may go
      uncounted. The settling compares the messages bit for bit, so it assumes
      deterministic kernels: otherwise the nodes recompute in every update and
      count their windows as unsettled. It then sends nothing more of the
      window, and reads no credit, until the next $t_w$: a node that learns
      later reads what the others sent before they learned.
    - ``learn_interval`` is not read, the frames learn nothing, and every
      frame is still predicted (forward only).

    Args:
        mods: The modules, on ``device``.
        act_spec: The spec of what the upstream sends, ``h`` ``(*batch,
            d_in)``.
        conf: The dynamics; ``d_s`` and ``nodes`` are those of ``mods``.
        tau: The time constants of PC-ALM.
        optimizer: The optimizer of every node.
        learn_interval: As in ``pc.clock.LearnClock``.
        device: Where the nodes compute.

    Raises:
        ValueError: ``conf.window`` with ``credit="own"`` (an ablation of BP
            only), or with time constants under which one update does not
            reach $\delta = -g$ (``tau.lam`` other than ``tau.r**2 / (tau.h
            + tau.r)``, or ``tau.max_dt`` other than ``tau.period``): the
            settling and the counts above assume it.
    """

    dynamics2: DynamicsLayer  # set by name: those of conf.nodes only
    dynamics3: DynamicsLayer

    def __init__(
        self,
        mods: DynamicsModules,
        *,
        act_spec: DictSpec[Activity],
        conf: DynamicsConf,
        tau: pc.TimeConstants,
        optimizer: Optimizer,
        learn_interval: float | None,
        device: Device = "cpu",
    ) -> None:
        if conf.window is not None:
            _check_window(conf.window, tau)
        super().__init__()
        self._window = conf.window
        hops = conf.nodes + 1
        self.mods = mods
        self.dynamics = Dynamics(
            mods.step.first,
            act_spec=act_spec,
            tau=tau,
            optimizer=optimizer,
            learn_interval=learn_interval,
            open_loop=conf.open_loop,
            open_cycle=conf.open_cycle,
            residual=conf.residual,
            trace=conf.trace,
            window=conf.window,
            hops=hops,
            device=device,
        )
        self.names = ("dynamics",) + tuple(
            f"dynamics{i + 2}" for i in range(len(mods.step.rest))
        )
        top: Dynamics | DynamicsLayer = self.dynamics
        for name, layer in zip(self.names[1:], mods.step.rest, strict=True):
            node = DynamicsLayer(
                cast(ResidualMinGRU, layer),
                act_spec=top.act_spec,
                tau=tau,
                optimizer=optimizer,
                learn_interval=learn_interval,
                trace=conf.trace,
                window=conf.window,
                win_spec=top.win_spec if conf.window is not None else None,
                hops=hops,
                device=device,
            )
            setattr(self, name, node)
            top.next.connect(node.prev)
            if conf.window is not None:
                top.win_next.connect(node.win_prev)
            top = node
        self.compare = Compare(
            mods.readout,
            s_spec=top.act_spec,
            target_spec=act_spec,
            tau=tau,
            optimizer=optimizer,
            learn_interval=learn_interval,
            residual=conf.residual,
            window=conf.window,
            win_spec=top.win_spec if conf.window is not None else None,
            hops=hops,
            device=device,
        )
        top.next.connect(self.compare.s)
        if conf.open_loop > 0:
            self.compare.next.connect(self.dynamics.z_hat)
        if conf.residual:
            self.dynamics.base.connect(self.compare.base)
        if conf.window is not None:
            top.win_next.connect(self.compare.win_s)
            if conf.residual:
                self.dynamics.win_base.connect(self.compare.win_base)
            if conf.window.feedback != "teacher":
                self.compare.win_next.connect(self.dynamics.win_hat)
        self.next, self.error = self.compare.next, self.compare.error

    @property
    def window(self) -> WindowConf | None:
        """The learning on windows, or None (learning from every frame)."""
        return self._window

    @property
    def nodes(self) -> list[Dynamics | DynamicsLayer | Compare]:
        """``dynamics``, ``dynamics2``, ... and ``compare``, in this order."""
        layers = [cast(DynamicsLayer, getattr(self, n)) for n in self.names[1:]]
        return [self.dynamics, *layers, self.compare]

    def connect(self, upstream: Port[Activity]) -> None:
        """Feed the block from ``upstream`` (``Source.out`` in M14, the
        encoder's output in M16): the input and the target of the dynamics."""
        upstream.connect(self.dynamics.z)
        upstream.connect(self.compare.target)
