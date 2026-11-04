r"""The offline bound of M14 (``task="offline"``): the modules of the dynamics
fit by Adam on windows cut from the training streams, with all the data at
hand and as many passes as it takes.

On a window $x_0 .. x_{W-1}$ the modules predict as a rollout of the
evaluation does: teacher forced for the first ``observed`` frames, then from
their own predictions, $s_j = \mathrm{step}(c_{j-1}, s_{j-1})$ from $s_0 = 0$
with $c_{j-1} = x_{j-1}$ for $j \le$ ``observed`` and $\hat x_{j-1}$ after,
$\hat x_j = W_o\,\mathrm{Top}(s_j)$ (plus $c_{j-1}$ with ``residual``). The
loss is the mean squared error of all the predictions $\hat x_1 ..
\hat x_{W-1}$. Its gradient runs through time (BPTT), or with ``local`` stops
at one frame as in the nodes: $c_{j-1}$ (an observation or a fed-back
prediction) and $s_{j-1}$ are constants of the loss of frame $j$.

``feedback`` says what the fed-back prediction is: ``"grad"`` carries the
gradient of the later frames back through it, ``"detach"`` cuts it
($c_{j-1} = \mathrm{sg}(\hat x_{j-1})$), ``"teacher"`` has no open loop. With
``credit="own"`` the gradient through time is cut as the eligibility traces
(``MinGRUTrace``) cut it: each layer's state is carried only along its own
recurrence, the path across layers is that of the same frame, and the
feedback is detached. A ``mask`` drops the windows that are not to be learned
from, under the fixed normalization $n (W - 1)$.

``OfflineFit`` draws its windows at random (``order="random"``), or takes
the windows $[tS, tS + W)$ of every stream at step $t$ (``order="stream"``),
as a run that learns every $S$ frames from the stream would, stepping an
optimizer of its own by ``elapsed``.

The optimizer is Adam as a ``pc.optim.AdamW`` (``exp.lib.optim.adam``), not
``torch.optim``, which brings the GIL back; an offline run passes the same
checks and the same ``evaluate_dynamics`` as an online one.
"""

from collections.abc import Callable, Mapping
from typing import Any, cast

import torch
from mujou.pc.optim import Optimizer
from torch import Tensor, nn

from exp.lib.mingru import MinGRU, ResidualMinGRU
from exp.lib.optim import adam
from exp.video.data.clips import fit_standardizer
from exp.video.data.observe import Standardizer, observe
from exp.video.data.physics import BallStreams
from exp.video.dyn.modules import DynamicsModules, DynamicsStack, Top
from exp.video.schema.learn import CREDITS, FEEDBACKS, ORDERS
from exp.video.schema.run import RunConf

__all__ = [
    "OfflineFit",
    "fit_offline",
    "pixel_encoder",
    "stream_windows",
    "window_loss",
]


def _state_windows(conf: RunConf, frames: int, window: int) -> Tensor:
    """The states of the first ``frames`` frames of each training stream,
    cut as ``stream_windows`` cuts them: ``(B (frames // window), window, 4K)``
    float64."""
    balls = conf.data.balls
    if balls is None:
        raise ValueError(f"the run {conf.name!r} has no balls")
    streams = BallStreams.of(
        balls, conf.data.streams, frame=conf.time.frame, seed=conf.seed
    )
    states = torch.stack([streams.next() for _ in range(frames)], 1)
    n = frames // window
    return states[:, : n * window].reshape(-1, window, states.shape[-1])


def stream_windows(
    conf: RunConf,
    frames: int,
    window: int = 32,
    *,
    std: Standardizer | None = None,
    states: bool = False,
) -> Tensor:
    """The first ``frames`` frames of each training stream of ``conf`` (the
    ``BallStreams`` of its seed), standardized by ``std`` (default: fitted on
    the calibration clips) and cut into windows of ``window`` frames, stream
    after stream: window ``j`` of stream ``b`` is row ``b (frames // window) +
    j``, its frames ``j window .. (j + 1) window - 1``. A window never spans two
    streams; the last ``frames % window`` frames of each stream are left out.

    With ``states`` the windows are those of the states ``(.., 4K)`` float64,
    not observed or standardized (``pixel_encoder`` does both): the images of
    all the frames of the full length of a run (128 streams x 54000 frames x
    1024 pixels) would not fit in memory.

    Returns:
        ``(B (frames // window), window, d)`` float32.

    Raises:
        ValueError: ``conf`` has no balls.
    """
    windows = _state_windows(conf, frames, window)
    if states:
        return windows
    balls = conf.data.balls
    assert balls is not None
    raw = observe(
        windows, balls.obs, len(balls.radii), radii=balls.radii, size=balls.size
    )
    std = std if std is not None else fit_standardizer(balls, frame=conf.time.frame)
    return std.apply(raw)


def pixel_encoder(conf: RunConf, std: Standardizer) -> Callable[[Tensor], Tensor]:
    """The map from windows of states ``(n, W, 4K)`` to the standardized
    images ``(n, W, size^2)`` float32 the nodes see (``observe`` and ``std``)."""
    balls = conf.data.balls
    if balls is None:
        raise ValueError(f"the run {conf.name!r} has no balls")
    k = len(balls.radii)

    def encode(x: Tensor) -> Tensor:
        raw = observe(x, balls.obs, k, radii=balls.radii, size=balls.size)
        return std.apply(raw)

    return encode


def _through(
    cell: MinGRU, x: Tensor, carried: Tensor, *, held: bool
) -> tuple[Tensor, Tensor]:
    """The output and the carried state of ``cell`` at ``x`` after the carried
    state ``carried``, with the gradient that of the eligibility traces.

    The carried state is built from the detached input, so its graph is the
    cell's own recurrence with the input held. The output has the carried
    state's value and the gradients of both: the carried state's, and that of
    ``cell(x, carried.detach())`` through ``x`` only (the term
    ``cell(x.detach(), carried.detach())`` takes its gradient for the
    parameters back). With ``held`` the input has no graph: the output is the
    carried state."""
    new = cell(x.detach(), carried)
    if held:
        return new, new
    h = carried.detach()
    return new + cell(x, h) - cell(x.detach(), h), new


def _own_step(
    stack: DynamicsStack, c: Tensor, carried: list[Tensor]
) -> tuple[list[Tensor], list[Tensor]]:
    """One frame of ``stack`` with the credit of ``credit="own"``: the new
    states of the nodes (the parts of ``DynamicsStack``'s state) and the
    carried states."""
    out, new = _through(stack.first, c, carried[0], held=True)
    parts, news, y = [out], [new], out
    for layer, h in zip(stack.rest, carried[1:], strict=True):
        cell = cast(ResidualMinGRU, layer).cell
        part, new = _through(cell, nn.functional.gelu(y), h, held=False)
        parts.append(part)
        news.append(new)
        y = y + part
    return parts, news


def window_loss(
    mods: DynamicsModules,
    x: Tensor,
    *,
    observed: int,
    local: bool,
    residual: bool,
    feedback: str = "grad",
    credit: str = "bptt",
    mask: Tensor | None = None,
) -> Tensor:
    r"""The loss of the module docstring on the windows ``x`` ``(n, W, d)``
    (``W >= 2``), a 0-d tensor in the graph of the parameters.

    With ``mask`` ``(n,)`` (bool or 0/1) it is $\sum_b m_b \sum_j
    \frac1D \lVert \hat x_j - x_j \rVert^2 / (n P)$ with $P = W - 1$: the
    windows out of ``mask`` cost 0 and the others keep the weight they have
    without it. Without it, the mean of all the predictions.

    Args:
        feedback: ``"grad"``, ``"detach"`` or ``"teacher"`` (the module
            docstring); ``"teacher"`` is ``observed = W``.
        credit: ``"bptt"``, or ``"own"``: the gradient through time of each
            layer only along its own state, the feedback detached (whatever
            ``feedback`` says).

    Raises:
        ValueError: ``feedback`` or ``credit`` is not one of its choices,
            ``credit="own"`` with ``local`` (it already has no credit through
            time), or ``mask`` is not of shape ``(n,)``.
    """
    if feedback not in FEEDBACKS:
        raise ValueError(f"feedback must be one of {FEEDBACKS}, not {feedback!r}")
    if credit not in CREDITS:
        raise ValueError(f"credit must be one of {CREDITS}, not {credit!r}")
    if credit == "own" and local:
        raise ValueError("credit='own' and local are two ways to cut the credit")
    own = credit == "own"
    if mask is not None and mask.shape != x.shape[:1]:
        raise ValueError(f"mask must be ({x.shape[0]},), got {tuple(mask.shape)}")
    if feedback == "teacher":
        observed = x.shape[1]
    cut = feedback == "detach" or own
    top = Top(mods.step.d_s)
    s = x.new_zeros(x.shape[0], mods.step.d_state)
    carried = list(s.split(mods.step.d_s, -1))
    c = x[:, 0]
    losses = []
    for j in range(1, x.shape[1]):
        if local:  # the input and the state are constants of frame j
            c, s = c.detach(), s.detach()
        if own:
            parts, carried = _own_step(mods.step, c, carried)
            y = mods.readout(torch.stack(parts).sum(0))
        else:
            s = mods.step(c, s)
            y = mods.readout(top(s))
        x_hat = c + y if residual else y
        losses.append(((x_hat - x[:, j]) ** 2).mean(-1))
        c = x[:, j] if j < observed else (x_hat.detach() if cut else x_hat)
    per = torch.stack(losses, 1)  # (n, P)
    if mask is None:
        return per.mean()
    return (per.sum(1) * mask.to(per.device, per.dtype)).sum() / per.numel()


class OfflineFit:
    """The fit of ``mods`` on ``windows`` (on the CPU; with ``encode`` the
    windows of something it turns to ``(batch, W, d)``, each batch as drawn)
    one step at a time (``step``): the gradient of ``window_loss`` on a batch
    of windows, and a step of the optimizer by ``elapsed``.

    With ``order="random"`` (default) ``windows`` is ``(n, W, d)``, ``batch``
    windows are drawn with replacement by ``gen`` (a CPU generator), and the
    optimizer is ``adam(lr)`` (``elapsed = 1``) unless ``optimizer`` says.

    With ``order="stream"`` ``windows`` is the data of each stream ``(B,
    frames, d)`` and nothing is drawn: step $t$ takes the windows $[tS, tS +
    W)$ of every stream (``stride`` $S$, ``window`` $W$; a batch of $B$
    windows), so that the steps go over the streams as a run that learns on
    the last $W$ frames every $S$ frames does. There are ``(frames - W) // S +
    1`` steps. The optimizer is ``optimizer`` stepped by ``elapsed`` (the time
    $S$ frames take), or ``adam(lr)`` (``elapsed = 1``). ``batch`` is not
    used.

    ``grad_norm`` is the norm of the gradient of the last step, a 0-d tensor
    (no synchronization).

    Its state (``state_dict``) is that of the optimizer, the generator and the
    count of steps (which says the next window of ``order="stream"``); the
    weights are the modules'.

    Raises:
        ValueError: ``order`` is not one of its choices, ``order="stream"``
            without ``window`` and a ``stride`` >= 1, or windows that do not
            fit in ``windows``; or (in ``step``) more steps than there are
            windows.
    """

    def __init__(
        self,
        mods: DynamicsModules,
        windows: Tensor,
        *,
        batch: int,
        lr: float,
        local: bool,
        observed: int,
        residual: bool,
        gen: torch.Generator,
        encode: Callable[[Tensor], Tensor] | None = None,
        order: str = "random",
        window: int | None = None,
        stride: int = 1,
        optimizer: Optimizer | None = None,
        elapsed: float = 1.0,
        feedback: str = "grad",
        credit: str = "bptt",
    ) -> None:
        if order not in ORDERS:
            raise ValueError(f"order must be one of {ORDERS}, not {order!r}")
        self._count = len(windows)  # the windows of "random", the steps of "stream"
        if order == "stream":
            if window is None or window < 2 or stride < 1:
                raise ValueError(
                    f"order='stream' needs window >= 2 and stride >= 1, "
                    f"got {window!r} and {stride}"
                )
            frames = windows.shape[1]
            if frames < window:
                raise ValueError(f"windows of {window} frames in {frames} frames")
            self._count = (frames - window) // stride + 1
        self.mods = mods
        self._encode = encode
        self._windows = windows
        self._batch, self._local = batch, local
        self._observed, self._residual = observed, residual
        self._gen = gen
        self._order, self._window, self._stride = order, window or 0, stride
        self._elapsed = elapsed
        self._feedback, self._credit = feedback, credit
        self._params = [*mods.step.parameters(), *mods.readout.parameters()]
        self._optimizer = (optimizer or adam(lr)).bind(self._params)
        self._device = mods.readout.weight.device
        self.steps = 0
        self.grad_norm = torch.zeros((), device=self._device)

    def _batch_at(self) -> Tensor:
        """The windows of the step now, before the encoder."""
        if self._order == "random":
            idx = torch.randint(len(self._windows), (self._batch,), generator=self._gen)
            return self._windows[idx]
        if self.steps >= self._count:
            raise ValueError(
                f"step {self.steps + 1}: the data has windows for {self._count} steps"
            )
        start = self.steps * self._stride
        return self._windows[:, start : start + self._window]

    def step(self) -> Tensor:
        """One step; returns its loss before the step, a 0-d tensor on the
        modules' device (no synchronization)."""
        x = self._batch_at()
        if self._encode is not None:
            x = self._encode(x)
        x = x.to(self._device)
        with torch.enable_grad():
            loss = window_loss(
                self.mods,
                x,
                observed=self._observed,
                local=self._local,
                residual=self._residual,
                feedback=self._feedback,
                credit=self._credit,
            )
            grads = torch.autograd.grad(loss, self._params, materialize_grads=True)
        self.grad_norm = torch.linalg.vector_norm(
            torch.stack([torch.linalg.vector_norm(g.detach()) for g in grads])
        )
        self._optimizer.step_(grads, self._elapsed)
        self.steps += 1
        return loss.detach()

    def state_dict(self) -> dict[str, Any]:
        return {
            "optimizer": self._optimizer.state_dict(),
            "gen": self._gen.get_state(),
            "steps": self.steps,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._optimizer.load_state_dict(state["optimizer"])
        self._gen.set_state(state["gen"])
        self.steps = int(state["steps"])


def fit_offline(
    mods: DynamicsModules,
    windows: Tensor,
    *,
    steps: int,
    batch: int,
    lr: float,
    local: bool,
    observed: int,
    gen: torch.Generator,
    every: int,
    on_eval: Callable[[int, DynamicsModules], None],
    residual: bool = False,
) -> None:
    """Fit ``mods`` for ``steps`` steps of ``OfflineFit``, calling
    ``on_eval(step, mods)`` after every ``every`` steps (0: never)."""
    fit = OfflineFit(
        mods,
        windows,
        batch=batch,
        lr=lr,
        local=local,
        observed=observed,
        residual=residual,
        gen=gen,
    )
    for _ in range(steps):
        fit.step()
        if every and fit.steps % every == 0:
            on_eval(fit.steps, mods)
