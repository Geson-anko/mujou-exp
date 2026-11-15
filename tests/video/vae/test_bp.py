import copy
import math
from collections.abc import Iterable, Sequence
from dataclasses import replace
from typing import Any, override

import mujou.pc as pc
import mujou.testing
import pytest
import torch

from exp.lib.host import Host
from exp.video.schema import AEConf, LearnerConf, OptimConf, RunConf
from exp.video.vae.bp import AENet, build_ae_net
from exp.video.vae.loss import ae_loss, batch_loss, target
from exp.video.vae.modules import AEModules, build_ae
from tests.video.packed import packed_conf, stacks

FRAME = 1 / 30
LR = 0.1  # per frame: SGD of tau = FRAME / LR


def _conf(**kw: Any) -> RunConf:
    conf = packed_conf("bp", streams=2, learn_interval=25e-3, token_dim=4)
    return replace(
        conf,
        program="ae",
        ae=AEConf(decoder_depth=1),
        optim_vision=OptimConf(kind="sgd", lr=LR, warmup=0.0),
        **kw,
    )


class _Recorded(pc.optim.Optimizer):
    """SGD of ``tau`` that records the ``elapsed`` of each step."""

    def __init__(self, tau: float) -> None:
        self.inner, self.elapsed = pc.optim.SGD(tau=tau), []

    @override
    def bind(self, params: Iterable[torch.Tensor]) -> pc.optim.BoundOptimizer:
        bound, record = self.inner.bind(params), self.elapsed

        class _Bound(pc.optim.BoundOptimizer):
            @override
            def step_(self, grads: Sequence[torch.Tensor], elapsed: float) -> None:
                record.append(elapsed)
                bound.step_(grads, elapsed)

            @override
            def state_dict(self) -> dict[str, Any]:
                return bound.state_dict()

            @override
            def load_state_dict(self, state: Any, /) -> None:
                bound.load_state_dict(state)

        return _Bound()


def _host(net: AENet) -> Host:
    run = mujou.testing.run(net)
    return Host(run, frame=FRAME, period=net.tau.period)


def _sample(x: torch.Tensor, valid: torch.Tensor) -> pc.msg.Sample:
    b = len(x)
    return pc.msg.Sample(
        x=x, y=torch.zeros(b, 1), mask=valid, reset=torch.zeros(b, dtype=torch.bool)
    )


def _loss(mods: AEModules, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return batch_loss(ae_loss(target(x), mods(x)), valid)


class TestAEBP:
    def test_one_frame_sends_l_before_learning_and_steps_by_its_gradient(self):
        """The loss on stats is L at the initial weights; the frame's one step
        is SGD on dL/dtheta of all the encoder's and decoder's parameters with
        the rate elapsed / tau, elapsed the time of the frame the clock
        counted. The stack that does not count (mask false) adds nothing."""
        conf = _conf()
        mods = build_ae(conf, seed=conf.seed)
        before = copy.deepcopy(mods)
        opt = _Recorded(tau=FRAME / LR)
        net = AENet(mods, conf=conf, optimizer=opt)
        x, valid = stacks(2), torch.tensor([True, False])
        p, _ = _host(net).show({net.prev: _sample(x, valid)}, net.stats)
        want = _loss(before, x, valid)
        assert float(p["loss"]) == pytest.approx(float(want), rel=1e-6)
        assert float(p["n"]) == 1.0
        assert len(opt.elapsed) == 1
        grads = torch.autograd.grad(want, before.parameters())
        rate = opt.elapsed[0] / (FRAME / LR)
        for p_new, p_old, g in zip(mods.parameters(), before.parameters(), grads):
            torch.testing.assert_close(p_new, p_old - rate * g, rtol=1e-5, atol=1e-7)

    def test_the_clock_learns_once_a_frame_over_about_a_frame(self):
        """With updates every 2 ms and learn_interval 25 ms the clock learns
        at the first update 25 ms after each write, once in each frame of 16
        or 17 updates: the first step counts 26 ms, the next ones the time
        between two learnings, 32 or 34 ms, at most 1.02 frames."""
        conf = _conf()
        opt = _Recorded(tau=FRAME / LR)
        net = AENet(build_ae(conf, seed=0), conf=conf, optimizer=opt)
        host = _host(net)
        valid = torch.ones(2, dtype=torch.bool)
        for k in range(12):
            host.show({net.prev: _sample(stacks(2, seed=k), valid)}, net.stats)
        assert len(opt.elapsed) == 12
        assert opt.elapsed[0] == pytest.approx(26e-3)
        assert {round(e * 1e3) for e in opt.elapsed[1:]} == {32, 34}
        assert max(opt.elapsed) / FRAME == pytest.approx(1.02, abs=1e-9)

    def test_no_stack_that_counts_learns_nothing(self):
        conf = _conf()
        mods = build_ae(conf, seed=conf.seed)
        before = copy.deepcopy(mods)
        net = AENet(mods, conf=conf, optimizer=pc.optim.SGD(tau=FRAME / LR))
        none = torch.zeros(2, dtype=torch.bool)
        p, _ = _host(net).show({net.prev: _sample(stacks(2), none)}, net.stats)
        assert float(p["loss"]) == 0.0 and float(p["n"]) == 0.0
        for a, b in zip(mods.parameters(), before.parameters(), strict=True):
            assert torch.equal(a, b)

    def test_build_needs_bp(self):
        conf = replace(_conf(), learner=LearnerConf(kind="pc"))
        with pytest.raises(ValueError, match="learner=bp"):
            build_ae_net(conf)

    def test_build_with_fan_in_binds_every_parameter(self):
        rms = OptimConf(kind="rmsprop", lr=1e-3, warmup=0.0, scale="fan_in")
        net = build_ae_net(replace(_conf(), optim_vision=rms))
        assert math.isfinite(net.tau.period)
