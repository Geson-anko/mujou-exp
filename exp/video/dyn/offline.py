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

The optimizer is Adam as a ``pc.optim.AdamW`` (``exp.lib.optim.adam``), not
``torch.optim``, which brings the GIL back; an offline run passes the same
checks and the same ``evaluate_dynamics`` as an online one.
"""

from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import Tensor

from exp.lib.optim import adam
from exp.video.data.clips import fit_standardizer
from exp.video.data.observe import Standardizer, observe
from exp.video.data.physics import BallStreams
from exp.video.dyn.modules import DynamicsModules, Top
from exp.video.schema.run import RunConf

__all__ = ["OfflineFit", "fit_offline", "stream_windows", "window_loss"]


def stream_windows(
    conf: RunConf, frames: int, window: int = 32, *, std: Standardizer | None = None
) -> Tensor:
    """The first ``frames`` frames of each training stream of ``conf`` (the
    ``BallStreams`` of its seed), standardized by ``std`` (default: fitted on
    the calibration clips) and cut into windows of ``window`` frames, stream
    after stream: window ``j`` of stream ``b`` is row ``b (frames // window) +
    j``, its frames ``j window .. (j + 1) window - 1``. A window never spans two
    streams; the last ``frames % window`` frames of each stream are left out.

    Returns:
        ``(B (frames // window), window, d)`` float32.

    Raises:
        ValueError: ``conf`` has no balls.
    """
    balls = conf.data.balls
    if balls is None:
        raise ValueError(f"the run {conf.name!r} has no balls")
    frame, k = conf.time.frame, len(balls.radii)
    streams = BallStreams.of(balls, conf.data.streams, frame=frame, seed=conf.seed)
    raw = torch.stack([observe(streams.next(), balls.obs, k) for _ in range(frames)], 1)
    std = std if std is not None else fit_standardizer(balls, frame=frame)
    n = frames // window
    x = std.apply(raw[:, : n * window])  # (B, n window, d)
    return x.reshape(x.shape[0] * n, window, x.shape[-1])


def window_loss(
    mods: DynamicsModules,
    x: Tensor,
    *,
    observed: int,
    local: bool,
    residual: bool,
) -> Tensor:
    """The loss of the module docstring on the windows ``x`` ``(n, W, d)``
    (``W >= 2``), a 0-d tensor in the graph of the parameters."""
    top = Top(mods.step.d_s)
    s = x.new_zeros(x.shape[0], mods.step.d_state)
    c = x[:, 0]
    losses = []
    for j in range(1, x.shape[1]):
        if local:  # the input and the state are constants of frame j
            c, s = c.detach(), s.detach()
        s = mods.step(c, s)
        y = mods.readout(top(s))
        x_hat = c + y if residual else y
        losses.append(((x_hat - x[:, j]) ** 2).mean(-1))
        c = x[:, j] if j < observed else x_hat
    return torch.stack(losses, 1).mean()


class OfflineFit:
    """The fit of ``mods`` on ``windows`` ``(n, W, d)`` (on the CPU) by Adam,
    one step at a time (``step``): ``batch`` windows drawn with replacement
    by ``gen`` (a CPU generator), the gradient of ``window_loss`` on them, and
    a step of ``adam(lr)`` (``elapsed = 1``).

    Its state (``state_dict``) is that of the optimizer, the generator and the
    count of steps; the weights are the modules'.
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
    ) -> None:
        self.mods = mods
        self._windows = windows
        self._batch, self._local = batch, local
        self._observed, self._residual = observed, residual
        self._gen = gen
        self._params = [*mods.step.parameters(), *mods.readout.parameters()]
        self._optimizer = adam(lr).bind(self._params)
        self._device = mods.readout.weight.device
        self.steps = 0

    def step(self) -> Tensor:
        """One step; returns its loss before the step, a 0-d tensor on the
        modules' device (no synchronization)."""
        idx = torch.randint(len(self._windows), (self._batch,), generator=self._gen)
        x = self._windows[idx].to(self._device)
        with torch.enable_grad():
            loss = window_loss(
                self.mods,
                x,
                observed=self._observed,
                local=self._local,
                residual=self._residual,
            )
            grads = torch.autograd.grad(loss, self._params, materialize_grads=True)
        self._optimizer.step_(grads, 1.0)
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
