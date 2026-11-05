"""The learning on windows of the PC-ALM nodes (``DynamicsBlock`` with
``dynamics.window``): at the equilibrium of each window's episode, each node
learns its part of BP's gradient of ``offline.window_loss``, at the time
``DynamicsBP`` learns it, from the same frames."""

import math
import re
from pathlib import Path
from typing import Any, override

import mujou.pc as pc
import mujou.testing
import pytest
import torch
from torch import nn

import mujou
from exp.lib.mingru import ResidualMinGRU
from exp.video.dyn.bp import DynamicsBP
from exp.video.dyn.modules import DynamicsModules, build_dynamics
from exp.video.dyn.nets import window_stats
from exp.video.dyn.nodes import DynamicsBlock, DynamicsLayer, Source
from exp.video.dyn.offline import window_loss
from exp.video.schema.dyn import DynamicsConf, WindowConf
from mujou import TensorSpec

W, O, S = 6, 3, 4
B, D, D_S = 3, 4, 5
P = W - 1
PERIOD = 2e-3
STEPS = 17  # a frame
FRAME = STEPS * PERIOD
NS = round(PERIOD * 1e9)
TAU = pc.TimeConstants(h=4e-3, lam=2e-3, r=4e-3)
RELAX = 0.05  # 25 steps


class _World64(mujou.Network):
    """``Source -> DynamicsBlock`` (PC) or ``Source -> DynamicsBP`` in
    float64."""

    def __init__(self, mods, dyn: DynamicsConf, optimizer, *, learner: str) -> None:
        super().__init__()
        assert TAU.period == PERIOD
        self.tau = TAU
        self.source = Source(
            x=TensorSpec((B, D), dtype=torch.float64), period=PERIOD, batch=B
        )
        kind = DynamicsBlock if learner == "pc" else DynamicsBP
        self.dyn = kind(
            mods,
            act_spec=self.source.act_spec,
            conf=dyn,
            tau=TAU,
            optimizer=optimizer,
            learn_interval=25e-3,
        )
        if isinstance(self.dyn, DynamicsBlock):
            self.dyn.connect(self.source.out)
        else:
            self.source.out.connect(self.dyn.prev)
        self.prev = self.source.prev


def _mods64(dyn: DynamicsConf) -> DynamicsModules:
    mods = build_dynamics(dyn, D, D, seed=0)
    mods.step.double()
    mods.readout.double()
    with torch.no_grad():  # a readout that is not 0, so that every weight moves
        g = torch.Generator().manual_seed(5)
        mods.readout.weight.copy_(
            0.3 * torch.randn(mods.readout.weight.shape, generator=g)
        )
    return mods


def _dyn(
    *, nodes: int = 3, residual: bool = True, relax: float = RELAX, **window
) -> DynamicsConf:
    win = WindowConf(
        **(
            {
                "window": W,
                "observed": O,
                "stride": S,
                "relax": relax,
                "feedback": "grad",
            }
            | window
        )
    )
    return DynamicsConf(
        d_s=D_S, nodes=nodes, residual=residual, open_loop=0.0, window=win
    )


def _stream(frames: int, seed: int = 3) -> torch.Tensor:
    """(frames, B, D) float64."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(frames, B, D, generator=g, dtype=torch.float64)


def _write(run, net: _World64, xs: torch.Tensor, k: int, resets=None) -> None:
    reset = torch.full((B,), k == 0)
    if resets is not None:
        reset = reset | resets[k]
    sample = pc.msg.Sample(x=xs[k], y=torch.zeros(B, 1), mask=~reset, reset=reset)
    run.write({net.prev: sample})


def _params(module: nn.Module) -> list[torch.Tensor]:
    return list(module.parameters())


class _Recording(pc.optim.Optimizer):
    """``inner`` (or nothing, without it) bound to each node, recording for
    each binding the gradients, the elapsed and the time of every step."""

    def __init__(self, inner: pc.optim.Optimizer | None = None) -> None:
        self.inner = inner
        self.bound: list[tuple[list[torch.Tensor], list[Any]]] = []

    @override
    def bind(self, params):
        params = list(params)
        steps: list[Any] = []
        self.bound.append((params, steps))
        inner = None if self.inner is None else self.inner.bind(params)

        class _Bound(pc.optim.BoundOptimizer):
            @override
            def step_(self, grads, elapsed):
                steps.append(([g.clone() for g in grads], elapsed))
                if inner is not None:
                    inner.step_(grads, elapsed)

            @override
            def state_dict(self):
                return {} if inner is None else inner.state_dict()

            @override
            def load_state_dict(self, state, /):
                if inner is not None:
                    inner.load_state_dict(state)

        return _Bound()

    def steps_of(self, params: list[torch.Tensor]) -> list[Any]:
        """The steps of the binding of exactly ``params``."""
        ids = [id(p) for p in params]
        for bound, steps in self.bound:
            if [id(p) for p in bound] == ids:
                return steps
        raise KeyError("no binding of these parameters")


def _node_params(mods: DynamicsModules) -> list[list[torch.Tensor]]:
    """The parameters of each node: Dynamics, dynamics2.., Compare."""
    return [
        _params(mods.step.first),
        *(_params(layer) for layer in mods.step.rest),
        _params(mods.readout),
    ]


def _reference(
    mods: DynamicsModules,
    x: torch.Tensor,
    *,
    observed: int,
    residual: bool,
    feedback: str,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], torch.Tensor]:
    r"""The window loss unrolled node by node on ``x`` ``(B, W, D)``: the loss,
    the predictions $\hat x_{1..P}$ ``(B, P, D)``, a zero leaf ``eps_n``
    ``(B, P, d)`` added to each node's output where the next node reads it
    (not in its own recurrence), and a zero leaf ``eps_c`` ``(B, P, D)``
    added to the fed-back inputs $c_p$, $p \ge$ ``observed`` (with
    ``"grad"``), where both the minGRU and the base read them. Checked against
    ``window_loss`` to 1e-12."""
    n, w, _ = x.shape
    p_ = w - 1
    obs = w if feedback == "teacher" else observed
    first, rest = mods.step.first, list(mods.step.rest)
    eps = [
        torch.zeros(n, p_, D_S, dtype=x.dtype, requires_grad=True)
        for _ in range(1 + len(rest))
    ]
    eps_c = torch.zeros(n, p_, x.shape[-1], dtype=x.dtype, requires_grad=True)
    s1 = x.new_zeros(n, D_S)
    hs = [x.new_zeros(n, D_S) for _ in rest]
    c = x[:, 0]
    losses, hats = [], []
    for j in range(1, w):
        p = j - 1
        used = c + eps_c[:, p] if feedback == "grad" and p >= obs else c
        s1 = first(used, s1)
        y = s1 + eps[0][:, p]
        for i, layer in enumerate(rest):
            assert isinstance(layer, ResidualMinGRU)
            hs[i] = layer.cell(nn.functional.gelu(y), hs[i])
            y = y + hs[i] + eps[i + 1][:, p]
        out = mods.readout(y)
        x_hat = used + out if residual else out
        hats.append(x_hat)
        losses.append(((x_hat - x[:, j]) ** 2).mean(-1))
        c = x[:, j] if j < obs else (x_hat.detach() if feedback == "detach" else x_hat)
    per = torch.stack(losses, 1)
    loss = (per.sum(1) * mask.to(per.dtype)).sum() / per.numel()
    want = window_loss(
        mods,
        x,
        observed=observed,
        local=False,
        residual=residual,
        feedback=feedback,
        mask=mask,
    )
    torch.testing.assert_close(loss, want, rtol=0, atol=1e-12)
    return loss, torch.stack(hats, 1), eps, eps_c


def _until_learning(run, net: _World64, xs: torch.Tensor) -> int:
    """Show the frames ``xs`` (a window due at the last), then step without a
    new frame up to the step before ``Dynamics`` learns; the steps taken in
    all."""
    block = net.dyn
    assert isinstance(block, DynamicsBlock)
    steps = 0
    for k in range(len(xs)):
        _write(run, net, xs, k)
        for _ in range(STEPS):
            run.step(PERIOD)
            steps += 1
    ep = block.dynamics.episode
    assert ep is not None and not ep["done"]
    while run.timer.now().ns + NS < ep["at_ns"]:
        run.step(PERIOD)
        steps += 1
    return steps


FEEDBACKS = ["grad", "detach", "teacher"]


class TestEquilibrium:
    @pytest.mark.parametrize("residual", [True, False], ids=["res", "plain"])
    @pytest.mark.parametrize("nodes", [1, 3])
    @pytest.mark.parametrize("feedback", FEEDBACKS)
    def test_each_node_learns_its_part_of_the_bp_gradient(
        self, feedback, nodes, residual
    ):
        """One window held: in the update before the learning, each node's
        ``r_w + lam_w`` is $-B P\\,\\partial L_\\text{win} / \\partial$ (its
        output window, as the next node reads it) and, with ``"grad"``, the
        credit ``Dynamics`` sends back on ``win_hat`` is that of the fed-back
        predictions; at the learning, each node's gradient (``g_theta / (B
        P)``) is its part of autograd's gradient of ``window_loss``, with the
        elapsed of S frames, in the first update at or after $t_w$ +
        ``relax``, settled and skipping nothing, and ``Compare``'s
        ``win_loss`` is the loss."""
        dyn = _dyn(nodes=nodes, residual=residual, feedback=feedback)
        rec = _Recording()
        mods = _mods64(dyn)
        net = _World64(mods, dyn, rec, learner="pc")
        block = net.dyn
        assert isinstance(block, DynamicsBlock)
        xs = _stream(W)
        frames = xs.transpose(0, 1)
        mask = torch.ones(B, dtype=torch.bool)
        ref = _mods64(dyn)
        loss, _, eps, eps_c = _reference(
            ref, frames, observed=O, residual=residual, feedback=feedback, mask=mask
        )
        g_eps = torch.autograd.grad(
            loss, [*eps, eps_c], allow_unused=True, retain_graph=True
        )
        with mujou.testing.run(net) as run:
            _until_learning(run, net, xs)
            nodes_ = block.nodes
            for node, g in zip(nodes_[:-1], g_eps, strict=False):
                torch.testing.assert_close(
                    node.r_w + node.lam_w, -B * P * g, rtol=0, atol=1e-10
                )
            if feedback == "grad":
                fb = run.read(block.dynamics.win_hat)["x"]
                g_c = g_eps[-1]
                assert g_c is not None
                want = torch.zeros_like(fb)
                want[:, O - 1 : P - 1] = B * P * g_c[:, O:]
                torch.testing.assert_close(fb, want, rtol=0, atol=1e-10)
            at_ns = block.dynamics.episode["at_ns"]  # type: ignore[index]
            assert all(len(s) == 0 for _, s in rec.bound)
            run.step(PERIOD)
            assert run.timer.now().ns >= at_ns > run.timer.now().ns - NS
        want = torch.autograd.grad(loss, _params(ref.step) + _params(ref.readout))
        got: list[torch.Tensor] = []
        for params in _node_params(mods):
            (steps,) = [rec.steps_of(params)]
            assert len(steps) == 1
            grads, elapsed = steps[0]
            assert elapsed == pytest.approx(S * FRAME, rel=1e-12)
            got += grads
        for g, r in zip(got, want, strict=True):
            torch.testing.assert_close(g, r, rtol=0, atol=1e-10)
        stats = window_stats(block)
        assert stats["win_steps"] == 1
        assert stats["win_skipped"] == stats["win_unsettled"] == 0
        assert stats["win_loss"] == pytest.approx(loss.item(), rel=0, abs=1e-12)


def _show(run, net: _World64, xs: torch.Tensor, after: int = 0, resets=None) -> None:
    """Show the frames ``xs`` (with the ``resets`` ``(frames, B)`` of the
    streams), then ``after`` steps without a new frame."""
    for k in range(len(xs)):
        _write(run, net, xs, k, resets)
        for _ in range(STEPS):
            run.step(PERIOD)
    for _ in range(after):
        run.step(PERIOD)


def _optimizer(kind: str) -> pc.optim.Optimizer:
    if kind == "sgd":
        return pc.optim.SGD(tau=0.5)
    return pc.optim.AdamW(tau=2.0, tau_m=0.3, tau_v=5.0)


def _weights(mods: DynamicsModules) -> list[torch.Tensor]:
    return _params(mods.step) + _params(mods.readout)


def _resets(frames: int) -> torch.Tensor:
    """``(frames, B)``: stream 1 starts a video again at frame 8 and stream 2
    at frame 3, so that the windows over them are out of the mask."""
    resets = torch.zeros(frames, B, dtype=torch.bool)
    resets[8, 1] = resets[3, 2] = True
    return resets


class TestTrajectory:
    @pytest.mark.parametrize("masked", [False, True], ids=["", "masked"])
    @pytest.mark.parametrize("kind", ["sgd", "adamw"])
    @pytest.mark.parametrize("feedback", ["grad", "detach"])
    def test_pc_and_bp_learn_the_same_weights_over_three_windows(
        self, feedback, kind, masked
    ):
        """W + 2S frames, three windows learned: the nodes of PC-ALM and
        ``DynamicsBP`` (the same modules, frames and optimizer) end with the
        same weights, ``win_loss`` and ``win_gnorm`` to 1e-8, none skipped or
        unsettled. With streams that reset inside windows (``masked``), the
        windows out of the mask count nothing and the normalization stays
        $B P$: the same as BP, with the same ``win_masked``."""
        dyn = _dyn(feedback=feedback)
        xs = _stream(W + 2 * S)
        resets = _resets(len(xs)) if masked else None
        nets = {}
        for learner in ("pc", "bp"):
            net = _World64(_mods64(dyn), dyn, _optimizer(kind), learner=learner)
            with mujou.testing.run(net) as run:
                _show(run, net, xs, after=round(RELAX / PERIOD) + 1, resets=resets)
            nets[learner] = net
        pc_, bp_ = nets["pc"].dyn, nets["bp"].dyn
        assert isinstance(pc_, DynamicsBlock) and isinstance(bp_, DynamicsBP)
        for a, b in zip(_weights(pc_.mods), _weights(bp_.mods), strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=1e-8)
        assert not torch.equal(pc_.mods.readout.weight, _mods64(dyn).readout.weight)
        got, want = window_stats(pc_), window_stats(bp_)
        assert got["win_steps"] == want["win_steps"] == 3
        assert got["win_skipped"] == got["win_unsettled"] == 0
        assert got["win_masked"] == want["win_masked"]
        assert (got["win_masked"] != 0) == masked
        for key in ("win_loss", "win_gnorm"):
            assert got[key] == pytest.approx(want[key], rel=0, abs=1e-8)


class TestOpenLoop:
    @pytest.mark.parametrize("nodes", [1, 3])
    @pytest.mark.parametrize("feedback", ["grad", "detach"])
    def test_the_predictions_fed_back_are_the_rollout_of_bp(self, feedback, nodes):
        """At the learning, the window of predictions ``Compare`` sends (and
        ``Dynamics`` fed back) is the rollout of ``window_loss``: from s = 0,
        O frames observed, then from its own predictions, to 1e-12."""
        dyn = _dyn(nodes=nodes, feedback=feedback)
        mods = _mods64(dyn)
        net = _World64(mods, dyn, _Recording(), learner="pc")
        block = net.dyn
        assert isinstance(block, DynamicsBlock)
        xs = _stream(W)
        with mujou.testing.run(net) as run:
            _until_learning(run, net, xs)
            run.step(PERIOD)
            got = run.read(block.compare.win_next)["x"]
        mask = torch.ones(B, dtype=torch.bool)
        with torch.no_grad():
            _, hats, _, _ = _reference(
                _mods64(dyn),
                xs.transpose(0, 1),
                observed=O,
                residual=True,
                feedback=feedback,
                mask=mask,
            )
        torch.testing.assert_close(got, hats, rtol=0, atol=1e-12)


def _schedule(name: str, nodes: list[mujou.Node], slow: int | None):
    """The nodes to update in step ``i``: in the tree order, or reversed;
    with ``slow``, the node of that index only in every other step (its period
    doubled)."""
    order = nodes[::-1] if name.startswith("reverse") else nodes

    def at(i: int) -> list[mujou.Node]:
        if slow is None or i % 2 == 1:  # in the steps that end at even periods
            return order
        return [n for n in order if n is not nodes[slow]]

    return at


# With relax half a period short of S frames, the time of a window comes 68
# periods after it, one before the next frame is first seen; a node at half the
# rate updates at the even periods only, so its time comes in the update that
# first sees the next window (69 periods after it). A layer or Compare then
# reads the next window from below first and skips the one before (but the
# last, which no window follows); Dynamics learns it first, in the same update
# (as DynamicsBP), and in the reverse order the layer updates before Dynamics
# opens the next one.
LATE_SKIPS = {"slow-dynamics2": [0, 3, 0, 0], "slow-compare": [0, 0, 0, 3]}

SCHEDULES = [
    ("slow-dynamics", 1),
    ("slow-dynamics2", 2),
    ("slow-compare", 4),
    ("reverse", None),
    ("reverse-slow-dynamics2", 2),
]


class TestAsynchrony:
    @pytest.mark.parametrize(("name", "slow"), SCHEDULES, ids=[n for n, _ in SCHEDULES])
    # 0.12 s is 60 periods; the messages of these schedules settle within 17
    # periods after t_w (13 in the reverse order, 17 with Dynamics or a layer
    # at half the rate).
    @pytest.mark.parametrize(
        "relax", [0.12, S * FRAME - PERIOD / 2], ids=["0.12", "late"]
    )
    def test_a_window_is_learned_as_bp_or_skipped_never_misaligned(
        self, name, slow, relax
    ):
        """With one node at half the rate, or the nodes in the reverse order,
        or both, every window a node learns is BP's gradient of that window
        to 1e-8 (the weights do not move, so it is that of the initial
        weights), and each node learns or skips every window whose time came.
        With ``relax`` 0.12 s, every node learns every window, settled; with
        ``relax`` half a period short of S frames, a slow node after
        ``Dynamics`` sees the next window before its time and skips
        (``LATE_SKIPS``)."""
        dyn = _dyn(relax=relax)
        rec = _Recording()
        mods = _mods64(dyn)
        net = _World64(mods, dyn, rec, learner="pc")
        block = net.dyn
        assert isinstance(block, DynamicsBlock)
        nodes: list[mujou.Node] = [net.source, *block.nodes]
        at = _schedule(name, nodes, slow)
        windows = 4
        xs = _stream(W + (windows - 1) * S)
        learned: dict[int, list[int]] = {i: [] for i in range(len(block.nodes))}
        held: dict[int, torch.Tensor] = {}  # t_w -> the frames Dynamics held
        i = 0
        with mujou.testing.run(net) as run:
            end = (len(xs) * STEPS + round(relax / PERIOD) + 20) * NS
            k = 0
            while run.timer.now().ns < end:
                if k < len(xs) and i % STEPS == 0:
                    _write(run, net, xs, k)
                    k += 1
                before = [
                    (n.win_steps, None if n.episode is None else n.episode["t_ns"])
                    for n in block.nodes
                ]
                run.step(PERIOD, *at(i))
                i += 1
                ep = block.dynamics.episode
                if ep is not None and ep["t_ns"] not in held:
                    held[ep["t_ns"]] = ep["x"].clone()
                for j, (n, (steps, t)) in enumerate(zip(block.nodes, before)):
                    if n.win_steps > steps:
                        assert t is not None
                        learned[j].append(t)
        assert len(held) == windows, sorted(held)
        refs = {}
        for w, (t, frames) in enumerate(sorted(held.items())):
            last = W - 1 + w * S
            if not name.startswith("reverse"):
                # (In the reverse order the first frame, written at time 0, is
                # not new to Dynamics: it took the source's initial message,
                # also of time 0, as a frame before it; DynamicsBP would too.)
                assert torch.equal(frames, xs[last - W + 1 : last + 1].transpose(0, 1))
            loss = window_loss(
                mods, frames, observed=O, local=False, residual=True, feedback="grad"
            )
            refs[t] = torch.autograd.grad(loss, _weights(mods))
        cuts = [0, 4, 8, 12, 14]  # the parameters of each node in _weights
        skipped = 0
        for j, (n, params) in enumerate(zip(block.nodes, _node_params(mods))):
            steps = rec.steps_of(params)
            assert len(steps) == len(learned[j]) == n.win_steps
            for (grads, _), t in zip(steps, learned[j], strict=True):
                want = refs[t][cuts[j] : cuts[j + 1]]
                for g, r in zip(grads, want, strict=True):
                    torch.testing.assert_close(g, r, rtol=0, atol=1e-8)
            assert n.win_steps + n.win_skipped == windows
            skipped += n.win_skipped
        assert window_stats(block)["win_unsettled"] == 0
        skips = [n.win_skipped for n in block.nodes]
        if relax == 0.12:
            assert skips == [0, 0, 0, 0]
        else:
            assert skips == LATE_SKIPS.get(name, [0, 0, 0, 0])


def _run_schedule(net: _World64, xs: torch.Tensor, at, steps: int) -> None:
    """``steps`` steps of ``at`` (``_schedule``; all nodes for BP), a frame of
    ``xs`` written every ``STEPS``."""
    with mujou.testing.run(net) as run:
        k = 0
        for i in range(steps):
            if k < len(xs) and i % STEPS == 0:
                _write(run, net, xs, k)
                k += 1
            run.step(PERIOD, *at(i))


LEARNING_SCHEDULES = [("slow-dynamics", 1), ("slow-dynamics2", 2), ("slow-compare", 4)]


class TestAsynchronyLearning:
    @pytest.mark.parametrize(
        ("name", "slow"), LEARNING_SCHEDULES, ids=[n for n, _ in LEARNING_SCHEDULES]
    )
    def test_a_slow_node_learns_the_trajectory_of_bp(self, name, slow):
        """With one node at half the rate and the weights moving (AdamW), a
        node that learns a window after the others reads only what they sent
        before they learned: four windows end with BP's weights to 1e-8,
        none skipped or unsettled."""
        relax = 0.12
        dyn = _dyn(relax=relax)
        windows = 4
        xs = _stream(W + (windows - 1) * S)
        steps = len(xs) * STEPS + round(relax / PERIOD) + 20
        pc_net = _World64(_mods64(dyn), dyn, _optimizer("adamw"), learner="pc")
        block = pc_net.dyn
        assert isinstance(block, DynamicsBlock)
        _run_schedule(
            pc_net, xs, _schedule(name, [pc_net.source, *block.nodes], slow), steps
        )
        bp_net = _World64(_mods64(dyn), dyn, _optimizer("adamw"), learner="bp")
        _run_schedule(bp_net, xs, lambda i: [], steps)
        bp_ = bp_net.dyn
        assert isinstance(bp_, DynamicsBP)
        for a, b in zip(_weights(block.mods), _weights(bp_.mods), strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=1e-8)
        got = window_stats(block)
        assert got["win_steps"] == window_stats(bp_)["win_steps"] == windows
        assert got["win_skipped"] == got["win_unsettled"] == 0


def _window_ports(block: DynamicsBlock) -> dict[str, Any]:
    """The window ports each node sends on, by node."""
    out: dict[str, Any] = {}
    for name in block.names:
        node = getattr(block, name)
        ports = [node.win_next] + ([node.win_prev] if name != "dynamics" else [])
        if name == "dynamics":
            ports += [node.win_base, node.win_hat]
        out[name] = ports
    c = block.compare
    out["compare"] = [c.win_s, c.win_base, c.win_next]
    return out


class TestFreeze:
    def test_a_node_sends_nothing_of_a_window_after_learning_it(self):
        """From the update a node learns a window in to the next window, the
        window messages it sends stay as they were, bit for bit and at the
        time sent; and from the time the messages settled to the learning, no
        node computes its window function again."""
        dyn = _dyn(relax=0.08)  # 40 periods; the messages settle in 12
        net = _World64(_mods64(dyn), dyn, pc.optim.SGD(tau=0.5), learner="pc")
        block = net.dyn
        assert isinstance(block, DynamicsBlock)
        ports = _window_ports(block)
        nodes = dict(zip([*block.names, "compare"], block.nodes, strict=True))
        xs = _stream(W + S)
        frozen: dict[str, list[Any]] = {}
        vjps: list[list[int]] = []
        checked = 0
        with mujou.testing.run(net) as run:
            for k in range(len(xs)):
                _write(run, net, xs, k)
                for _ in range(STEPS):
                    run.step(PERIOD)
                    ep = block.dynamics.episode
                    if ep is None or ep["t_ns"] != (W - 1) * round(FRAME * 1e9):
                        continue  # the first window only
                    since = (run.timer.now().ns - ep["t_ns"]) // NS
                    if 20 <= since < 40:
                        vjps.append([n.win_vjps for n in block.nodes])
                    for name, node in nodes.items():
                        if node.win_steps == 1 and name not in frozen:
                            frozen[name] = [run.read_with_time(p) for p in ports[name]]
                        elif name in frozen:
                            now = [run.read_with_time(p) for p in ports[name]]
                            for (a, ta), (b, tb) in zip(frozen[name], now):
                                assert ta == tb
                                assert a.keys() == b.keys()
                                for key in a:
                                    if isinstance(a[key], torch.Tensor):
                                        assert torch.equal(a[key], b[key])
                                    else:
                                        assert a[key] == b[key]
                            checked += 1
        assert set(frozen) == set(nodes)
        assert checked > 4 * 10
        assert len(vjps) == 20 and all(v == vjps[0] for v in vjps)


class TestResume:
    def test_a_run_resumed_mid_episode_is_one_never_stopped(self):
        """Stopped after a window's time and before its learning (the episode
        open, ``r_w`` not 0) and restored from ``run.state_dict()``, the run
        ends as one never stopped, bit for bit: the weights, the window
        messages and ``window_stats``, with AdamW."""
        dyn = _dyn()
        xs = _stream(W + 2 * S)
        after = round(RELAX / PERIOD) + 1
        whole = _World64(_mods64(dyn), dyn, _optimizer("adamw"), learner="pc")
        with mujou.testing.run(whole) as run:
            _show(run, whole, xs, after=after)
            ref = {
                n: [run.read(p) for p in ps]
                for n, ps in _window_ports(whole.dyn).items()  # type: ignore[arg-type]
            }
        cut = W + S  # the second window is due at frame cut - 1
        first = _World64(_mods64(dyn), dyn, _optimizer("adamw"), learner="pc")
        block = first.dyn
        assert isinstance(block, DynamicsBlock)
        with mujou.testing.run(first) as run:
            _show(run, first, xs[: cut - 1])
            _write(run, first, xs, cut - 1)
            for _ in range(6):  # into the episode of the frame's window
                run.step(PERIOD)
            ep = block.dynamics.episode
            assert ep is not None and not ep["done"]
            assert bool(block.dynamics.r_w.any() or block.dynamics.lam_w.any())
            state = run.state_dict()
        second = _World64(_mods64(dyn), dyn, _optimizer("adamw"), learner="pc")
        with mujou.testing.run(second, state=state) as run:
            # the rest of the frame of the cut, then the frames after it
            for _ in range(STEPS - 6):
                run.step(PERIOD)
            for k in range(cut, len(xs)):
                _write(run, second, xs, k)
                for _ in range(STEPS):
                    run.step(PERIOD)
            for _ in range(after):
                run.step(PERIOD)
            got = {
                n: [run.read(p) for p in ps]
                for n, ps in _window_ports(second.dyn).items()  # type: ignore[arg-type]
            }
        a, b = whole.dyn, second.dyn
        assert isinstance(a, DynamicsBlock) and isinstance(b, DynamicsBlock)
        for p, q in zip(_weights(a.mods), _weights(b.mods), strict=True):
            assert torch.equal(p, q)
        for name, msgs in ref.items():
            for m, n in zip(msgs, got[name], strict=True):
                for key in m:
                    if isinstance(m[key], torch.Tensor):
                        assert torch.equal(m[key], n[key]), (name, key)
                    else:
                        assert m[key] == n[key], (name, key)
        assert window_stats(a) == window_stats(b)
        assert window_stats(a)["win_steps"] == 3


class TestLocality:
    def test_no_parameter_gets_a_grad_and_only_the_optimizers_move_them(self):
        """After windows learned, no parameter holds ``.grad``; SGD moved every
        node's weights, and an optimizer that does not step moves none."""
        dyn = _dyn()
        moved = {}
        for name, optimizer in (("sgd", pc.optim.SGD(tau=0.5)), ("none", _Recording())):
            mods = _mods64(dyn)
            before = [p.detach().clone() for p in _weights(mods)]
            net = _World64(mods, dyn, optimizer, learner="pc")
            with mujou.testing.run(net) as run:
                _show(run, net, _stream(W + S), after=round(RELAX / PERIOD) + 1)
            after = _weights(mods)
            assert all(p.grad is None for p in after)
            moved[name] = [not torch.equal(a, b) for a, b in zip(before, after)]
        assert all(moved["sgd"])
        assert not any(moved["none"])

    @pytest.mark.parametrize("path", ["exp/video/dyn/nodes.py", "exp/lib/mingru.py"])
    def test_the_code_has_no_backward_and_no_grad_attribute(self, path):
        """In the code (docstrings and comments left out): no ``.backward(``,
        no ``.grad`` attribute but ``autograd.grad``, and every
        ``autograd.grad(`` marked ``# mujou: local-autograd``."""
        root = Path(__file__).resolve().parents[3]
        lines = (root / path).read_text().splitlines()
        code = _code_lines(lines)
        for i, line in code:
            assert ".backward(" not in line, (path, i + 1)
            assert not re.search(r"(?<!autograd)\.grad\b", line), (path, i + 1)
            if "autograd.grad(" in line:
                assert "# mujou: local-autograd" in lines[i], (path, i + 1)


def _code_lines(lines: list[str]) -> list[tuple[int, str]]:
    """The lines of code with their index, without docstrings and comments."""
    out, doc = [], None
    for i, line in enumerate(lines):
        text = line.strip()
        if doc is not None:
            if doc in text:
                doc = None
            continue
        for quote in ('r"""', '"""'):
            if text.startswith(quote):
                rest = text[len(quote) :]
                if quote[-3:] not in rest:
                    doc = '"""'
                break
        else:
            out.append((i, line.split("#", 1)[0]))
    return out


class TestBlock:
    def test_credit_own_is_refused(self):
        dyn = _dyn(feedback="detach", credit="own")
        with pytest.raises(ValueError, match="ablation of BP only"):
            _World64(_mods64(dyn), dyn, _Recording(), learner="pc")

    def test_time_constants_without_one_step_relaxation_are_refused(self):
        dyn = _dyn()
        source = Source(
            x=TensorSpec((B, D), dtype=torch.float64), period=PERIOD, batch=B
        )
        tau = pc.TimeConstants(h=4e-3, lam=1e-3, r=4e-3)
        with pytest.raises(ValueError, match="one-step"):
            DynamicsBlock(
                _mods64(dyn),
                act_spec=source.act_spec,
                conf=dyn,
                tau=tau,
                optimizer=_Recording(),
                learn_interval=None,
            )

    def test_without_a_window_the_nodes_have_no_window_ports(self):
        dyn = DynamicsConf(d_s=D_S, nodes=2, residual=True)
        source = Source(x=TensorSpec((B, D)), period=PERIOD, batch=B)
        block = DynamicsBlock(
            build_dynamics(dyn, D, D, seed=0),
            act_spec=source.act_spec,
            conf=dyn,
            tau=TAU,
            optimizer=_Recording(),
            learn_interval=None,
        )
        assert block.window is None
        for node in block.nodes:
            assert not any(k.startswith("win_") for k in vars(node))
        assert isinstance(block.dynamics2, DynamicsLayer)


def _one_window(
    dyn: DynamicsConf, dts=None, *, until: int | None = None
) -> tuple[list[float], list[int], int, int]:
    """One window of the default stream learned by PC-ALM (the recording
    optimizer), with the step ``i`` of ``dts(i)`` seconds (one period without
    it): the largest error of each node's gradient against autograd's of
    ``window_loss``, each node's ``win_unsettled``, the window time and the
    time of the last update that computed a window function before the
    learning (all in ns)."""
    win = dyn.window
    assert win is not None
    rec = _Recording()
    mods = _mods64(dyn)
    net = _World64(mods, dyn, rec, learner="pc")
    block = net.dyn
    assert isinstance(block, DynamicsBlock)
    xs = _stream(win.window)
    dt = dts if dts is not None else (lambda i: PERIOD)
    last, i = 0, 0
    with mujou.testing.run(net) as run:
        steps = win.window * STEPS + (until or round(win.relax / PERIOD) + 2)
        for i in range(steps):
            if i % STEPS == 0 and i // STEPS < win.window:
                _write(run, net, xs, i // STEPS)
            before = sum(n.win_vjps for n in block.nodes)
            run.step(dt(i))
            learned = any(n.win_steps or n.win_skipped for n in block.nodes)
            if sum(n.win_vjps for n in block.nodes) > before and not learned:
                last = run.timer.now().ns
    ep = block.dynamics.episode
    assert ep is not None
    ref = _mods64(dyn)
    loss = window_loss(
        ref,
        xs.transpose(0, 1),
        observed=win.observed,
        local=False,
        residual=dyn.residual,
        feedback=win.feedback,
    )
    want = torch.autograd.grad(loss, _weights(ref))
    errs, k = [], 0
    for params in _node_params(mods):
        steps_ = rec.steps_of(params)
        if steps_:
            (grads, _), *_ = steps_
            errs.append(
                max(
                    (g - w).abs().max().item()
                    for g, w in zip(grads, want[k : k + len(params)], strict=True)
                )
            )
        else:
            errs.append(math.nan)  # skipped
        k += len(params)
    return errs, [n.win_unsettled for n in block.nodes], ep["t_ns"], last


class TestSettling:
    @pytest.mark.parametrize(
        ("feedback", "nodes", "settle"),
        [("grad", 3, 79), ("grad", 1, 39), ("detach", 3, 22), ("teacher", 3, 3)],
    )
    def test_the_default_window_settles_within_the_default_relax(
        self, feedback, nodes, settle
    ):
        """The default window (W 36, O 16, relax 0.3 s = 150 periods): the
        messages settle, bit for bit, in $(L + 1) K + L$ updates of the tree
        order after the episode opens ($K + L$ with ``"detach"``, $L$ with
        ``"teacher"``; $K = 19$), one period after $t_w$; then each node
        learns BP's gradient to 1e-10, none unsettled."""
        dyn = DynamicsConf(
            d_s=D_S,
            nodes=nodes,
            residual=True,
            open_loop=0.0,
            window=WindowConf(window=36, observed=16, feedback=feedback),
        )
        assert dyn.window is not None and math.isclose(dyn.window.relax, 0.3)
        errs, unsettled, t_w, last = _one_window(dyn)
        assert (last - t_w) // NS == settle + 1
        assert max(errs) < 1e-10
        assert unsettled == [0] * (nodes + 1)

    @pytest.mark.parametrize("periods", [4, 6, 8, 10, 12, 14])
    def test_a_window_learned_before_it_settles_is_counted(self, periods):
        """W 6 (K 2, 3 nodes: settled 12 periods after $t_w$) with ``relax``
        too short: every node that learns a gradient other than BP's to 1e-10
        counts it in ``win_unsettled`` (up to 10 periods, some do)."""
        dyn = _dyn(relax=periods * PERIOD)
        errs, unsettled, _, _ = _one_window(dyn)
        for err, count in zip(errs, unsettled, strict=True):
            if not err < 1e-10:  # NaN: skipped, nothing learned
                assert count == 1 or math.isnan(err), (errs, unsettled)
        if periods <= 10:  # the oracle above is not empty
            assert max(errs) > 1e-10

    # (In the rule before the fix, offsets 8 and 9 learned an error of 1.5e-3
    # uncounted: the credit of the short update stayed.)
    @pytest.mark.parametrize("offset", range(22))
    def test_an_update_shorter_than_max_dt_does_not_leave_a_wrong_credit(self, offset):
        """One update of half a period ``offset`` periods after the window
        opens: ``r_w + lam_w`` falls short of $-g$ there, so the node relaxes
        again in its next update; each node still learns BP's gradient to
        1e-10, or counts the window as unsettled."""
        dyn = _dyn()
        win = dyn.window
        assert win is not None
        at = (win.window - 1) * STEPS + offset + 1

        def dts(i: int) -> float:
            return PERIOD / 2 if i == at else PERIOD

        errs, unsettled, _, _ = _one_window(dyn, dts)
        for err, count in zip(errs, unsettled, strict=True):
            assert err < 1e-10 or count == 1, (errs, unsettled)
        assert max(errs) < 1e-10


class TestOneStepRelaxation:
    @pytest.mark.parametrize("r", [2e-3, 4e-3, 8e-3, math.inf])
    def test_one_update_reaches_minus_g_with_the_accepted_constants(self, r):
        """With ``lam`` = $\tau_r^2 / (\tau_h + \tau_r)$ and ``dt`` =
        ``period``, one ``relax_`` from any ``(r, lam)`` gives ``r + lam`` =
        $-g$ (what ``_check_window`` accepts); with ``lam`` = $\tau_h \tau_r
        / (\tau_h + \tau_r)$ and $\tau_r \ne \tau_h$ it does not, and the
        block refuses it."""
        h = 4e-3
        lam = r * r / (h + r) if math.isfinite(r) else math.inf
        tau = pc.TimeConstants(h=h, lam=lam, r=r)
        g = torch.Generator().manual_seed(0)
        r_w, lam_w, cred = (
            torch.randn(7, generator=g, dtype=torch.float64) for _ in range(3)
        )
        if not math.isfinite(lam):
            lam_w.zero_()  # lam stays at 0 without the dual
        pc.functional.relax_(r_w, lam_w, cred, tau.period, tau)
        torch.testing.assert_close(r_w + lam_w, -cred, rtol=0, atol=1e-12)
        block = _block(tau)  # accepted
        assert block.window is not None
        if math.isfinite(r) and r != h:
            wrong = pc.TimeConstants(h=h, lam=h * r / (h + r), r=r)
            with pytest.raises(ValueError, match="one-step"):
                _block(wrong)

    def test_a_max_dt_shorter_than_the_period_is_refused(self):
        """``2 r < h``: ``max_dt`` = ``2 lam`` < ``period``, so a full update
        does not reach $-g$."""
        h, r = 4e-3, 1e-3
        tau = pc.TimeConstants(h=h, lam=r * r / (h + r), r=r)
        assert tau.max_dt < tau.period
        with pytest.raises(ValueError, match="one-step"):
            _block(tau)


def _block(tau: pc.TimeConstants) -> DynamicsBlock:
    dyn = _dyn()
    source = Source(x=TensorSpec((B, D), dtype=torch.float64), period=PERIOD, batch=B)
    return DynamicsBlock(
        _mods64(dyn),
        act_spec=source.act_spec,
        conf=dyn,
        tau=tau,
        optimizer=_Recording(),
        learn_interval=None,
    )
