import math

import mujou.pc as pc
import mujou.testing
import pytest
import torch
from torch import nn

from exp.video.data.clips import fit_standardizer, make_clips
from exp.video.data.observe import observe
from exp.video.data.physics import BallStreams
from exp.video.dyn.modules import DynamicsModules, build_dynamics
from exp.video.dyn.nets import build_world, modules
from exp.video.measure.dynamics import (
    ball_contact_errors,
    dynamics_baselines,
    evaluate_dynamics,
    fork_eval,
    rollout,
)
from exp.video.schema.data import BallsConf, DataConf, PhysicsConf
from exp.video.schema.dyn import DynamicsConf
from exp.video.schema.learn import LearnerConf, OptimConf
from exp.video.schema.measure import MeasureConf
from exp.video.schema.run import RunConf, TimeConf

FRAME = 1 / 30
M = MeasureConf(observed=16, horizons=[1, 2, 5, 10, 20], clips=8)
STAGES = {
    "s0": BallsConf(radii=[0.10], obs="pos_vel"),
    "s2": BallsConf(radii=[0.06, 0.08, 0.10], obs="pos"),
}


def _width(balls: BallsConf) -> int:
    return len(balls.radii) * (4 if balls.obs == "pos_vel" else 2)


def _setting(balls: BallsConf):
    clips = make_clips(
        balls, split="val", clips=M.clips, length=100, horizon=20, frame=FRAME
    )
    std = fit_standardizer(balls, frame=FRAME, clips=16)
    return clips, std, dynamics_baselines(clips, balls, std, M)


def _copying(d: int, nodes: int = 2) -> DynamicsModules:
    """Residual modules with W_o = 0: every prediction is the input before."""
    conf = DynamicsConf(d_s=8, nodes=nodes, residual=True)
    return build_dynamics(conf, d, d, seed=0)


def _gated(d: int, a: torch.Tensor, *, gate: float) -> DynamicsModules:
    r"""One minGRU node of d_s = d with the gate saturated, W_s = ``a``, b_s = 0
    and the readout the identity: with ``gate`` +50, u = 1 and $s_k = A
    x_{k-1}$, so the open loop predicts $A^h x_{O-1}$; with -50, u = 0 and the
    state is carried, so every prediction is the readout of the state."""
    mods = build_dynamics(DynamicsConf(d_s=d, nodes=1), d, d, seed=0)
    with torch.no_grad():
        cell = mods.step.first
        cell.gate.weight.zero_()
        cell.gate.bias.fill_(gate)
        cell.candidate.weight.copy_(a)
        cell.candidate.bias.zero_()
        mods.readout.weight.copy_(torch.eye(d))
        mods.readout.bias.zero_()
    return mods


class TestEvaluate:
    @pytest.mark.parametrize("stage", sorted(STAGES))
    def test_a_rollout_that_copies_is_the_copy_baseline(self, stage):
        """With W_o = 0 a residual rollout predicts the last observation at
        every horizon: each error equals the copy baseline's, over all the
        windows, those with and without a contact and each ball's columns."""
        balls = STAGES[stage]
        clips, std, base = _setting(balls)
        mods = _copying(_width(balls))
        row = evaluate_dynamics(
            mods,
            clips,
            std,
            base,
            M,
            obs=balls.obs,
            balls=len(balls.radii),
            residual=True,
        )
        suffixes = ["", "@contact", "@free", "@contactball"]
        suffixes += [f"@ball{i}" for i in range(len(balls.radii))]
        for h in M.horizons:
            for suffix in suffixes:
                want = base[f"copy{h}{suffix}"]
                got = row[f"roll{h}{suffix}"]
                if math.isnan(want):  # no window of that kind
                    assert math.isnan(got)
                else:
                    assert got == pytest.approx(want, rel=1e-6, abs=1e-9)
            assert row[f"roll{h}/cv{h}"] == pytest.approx(
                base[f"copy{h}"] / base[f"cv{h}"], rel=1e-6
            )
        assert row["motion2"] == 0.0  # the prediction stands still
        assert row["nonfinite"] == 0
        for h in M.horizons:  # copy is one of the bars: it learned nothing
            for key in ("bounce", "bounceball", "gravity"):
                assert not row[f"{key}{h}"] < 1.0 - 1e-6  # float32 against 64

    @pytest.mark.parametrize("stage", sorted(STAGES))
    def test_the_bars_of_the_bounces_and_gravity_are_the_best_baselines(self, stage):
        """bounce{h} = roll@contact / min(mean, copy, cv, ca)@contact,
        bounceball{h} the same @contactball, gravity{h} = roll@free /
        min(mean, copy, cv)@free."""
        balls = STAGES[stage]
        clips, std, base = _setting(balls)
        mods = build_dynamics(
            DynamicsConf(d_s=8, nodes=2), _width(balls), _width(balls), seed=0
        )
        row = evaluate_dynamics(
            mods,
            clips,
            std,
            base,
            M,
            obs=balls.obs,
            balls=len(balls.radii),
            residual=False,
        )
        h = 10
        bars = {
            "bounce": ("@contact", ["mean", "copy", "cv", "ca"]),
            "bounceball": ("@contactball", ["mean", "copy", "cv", "ca"]),
            "gravity": ("@free", ["mean", "copy", "cv"]),
        }
        for key, (suffix, names) in bars.items():
            best = min(base[f"{n}{h}{suffix}"] for n in names)
            assert row[f"{key}{h}"] == pytest.approx(row[f"roll{h}{suffix}"] / best)

    def test_with_one_ball_a_contact_of_the_ball_is_one_of_the_window(self):
        balls = STAGES["s0"]
        clips, std, base = _setting(balls)
        for h in M.horizons:
            for name in ("mean", "copy", "cv", "ca"):
                want = base[f"{name}{h}@contact"]
                assert base[f"{name}{h}@contactball"] == pytest.approx(want)

    def test_the_errors_of_the_contacts_count_the_balls_that_touch(self):
        """Two windows, two balls of two columns, h = 1: the squared errors of
        the balls are (1, 4) and (9, 16) (each the mean of its columns); with
        the contacts (yes, no) and (yes, yes) the error is (1 + 9 + 16) / 3."""
        target = torch.zeros(2, 1, 4)
        pred = torch.tensor([[[1.0, 1.0, 2.0, 2.0]], [[3.0, 3.0, 4.0, 4.0]]])
        contact = torch.tensor([[[True, False]], [[True, True]]])
        got = ball_contact_errors(pred, target, contact, [1])
        assert got[1] == pytest.approx(26 / 3)
        assert math.isnan(ball_contact_errors(pred, target, contact & False, [1])[1])

    def test_the_open_loop_feeds_its_own_predictions(self):
        """With s_k = A x_{k-1} and the identity readout, the prediction h
        frames ahead is A^h x_{O-1}."""
        balls = STAGES["s0"]
        clips, std, base = _setting(balls)
        theta = 0.3
        rot = torch.tensor(
            [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
        )
        a = torch.block_diag(rot, 0.9 * rot)
        x = torch.randn(5, 16, 4, generator=torch.Generator().manual_seed(0))
        pred = rollout(
            _gated(4, a, gate=50.0), x, residual=False, observed=16, horizon=6
        )
        last = x[:, 15]
        for h in range(1, 7):
            want = last @ torch.linalg.matrix_power(a, h).T
            torch.testing.assert_close(pred[:, h - 1], want, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("learner", ["pc", "bp"])
    def test_the_teacher_forced_prediction_is_the_nets(self, learner):
        """Shown a window as a stream with learning off, the network's
        prequential prediction of frame j is the rollout's first prediction
        from the j frames before (``observed = j``)."""
        conf = RunConf(
            name="test",
            learner=LearnerConf(kind=learner),
            data=DataConf(streams=3),
            dynamics=DynamicsConf(d_s=8, nodes=3, open_loop=0.0),
            time=TimeConf(learn_interval=math.inf),
            optim=OptimConf(warmup=0.0),
        )
        net = build_world(conf, 4)
        with torch.no_grad():  # a readout that is not 0, so the test sees it
            nn.init.normal_(
                modules(net).readout.weight, generator=torch.Generator().manual_seed(1)
            )
        x = torch.randn(3, 8, 4, generator=torch.Generator().manual_seed(2))
        got = []
        with mujou.testing.run(net) as run:
            for k in range(8):
                reset = torch.full((3,), k == 0)
                sample = pc.msg.Sample(
                    x=x[:, k], y=torch.zeros(3, 1), mask=~reset, reset=reset
                )
                run.write({net.prev: sample})
                for _ in range(17):
                    run.step(net.tau.period)
                got.append(run.read(net.next)["y_hat"])
        mods = modules(net)
        for j in range(1, 8):
            want = rollout(mods, x, residual=False, observed=j, horizon=1)[:, 0]
            torch.testing.assert_close(got[j], want, rtol=1e-5, atol=1e-6)

    def test_a_prediction_that_is_not_finite_is_counted(self):
        balls = STAGES["s0"]
        clips, std, base = _setting(balls)
        mods = _copying(4)
        with torch.no_grad():
            mods.readout.bias.fill_(math.inf)
        row = evaluate_dynamics(
            mods, clips, std, base, M, obs="pos_vel", balls=1, residual=True
        )
        assert row["nonfinite"] == len(clips.starts)
        assert not math.isfinite(row["roll10"])

    def test_clips_of_another_length_are_an_error(self):
        balls = STAGES["s0"]
        clips, std, base = _setting(balls)
        far = MeasureConf(observed=16, horizons=[40])
        with pytest.raises(ValueError, match="do not fit"):
            evaluate_dynamics(
                _copying(4),
                clips,
                std,
                base,
                far,
                obs="pos_vel",
                balls=1,
                residual=True,
            )


def _flying_streams(balls: BallsConf, n: int) -> BallStreams:
    """``n`` streams without gravity whose balls fly slowly through the middle
    of the box: no wall within the frames of the test."""
    streams = BallStreams.of(balls, n, frame=FRAME, seed=0)
    k = len(balls.radii)
    pos = torch.full((n, k, 2), 0.5, dtype=torch.float64)
    vel = torch.linspace(-0.2, 0.2, n * k * 2, dtype=torch.float64).view(n, k, 2)
    streams.load_state_dict({"pos": pos, "vel": vel, "frame": torch.tensor(0)})
    return streams


class TestFork:
    @pytest.mark.parametrize("stage", sorted(STAGES))
    def test_the_future_is_the_streams_and_they_go_on_unchanged(self, stage):
        """Without gravity and walls, constant velocity from the last two
        frames shown is the stream's future exactly, so ``forkcv`` is 0: the
        future starts at the frame after the last one shown. The streams are
        left where they were."""
        balls = BallsConf(
            radii=STAGES[stage].radii,
            obs=STAGES[stage].obs,
            physics=PhysicsConf(gamma0=0.0),
        )
        k, d = len(balls.radii), _width(balls)
        streams = _flying_streams(balls, 4)
        history = torch.stack(
            [observe(streams.next(), balls.obs, k) for _ in range(3)], 1
        )
        after = streams.fork()
        std = fit_standardizer(STAGES[stage], frame=FRAME, clips=16)
        mods = _copying(d)
        s = torch.zeros(4, mods.step.d_state)
        row = fork_eval(
            mods,
            s,
            streams,
            history,
            std,
            M,
            obs=balls.obs,
            balls=k,
            residual=True,
            frame=FRAME,
        )
        for h in M.horizons:
            assert row[f"forkcv{h}"] < 1e-10
            assert row[f"fork{h}"] == pytest.approx(row[f"forkcopy{h}"], rel=1e-6)
        torch.testing.assert_close(streams.next(), after.next(), rtol=0, atol=0)

    def test_the_rollout_starts_from_the_live_state(self):
        """With the gate shut (u = 0) the state is carried as it is and every
        prediction is W_o s = s: the error of each horizon is that of the
        given state against the future."""
        balls = BallsConf(radii=[0.10], obs="pos_vel", physics=PhysicsConf(gamma0=0.0))
        streams = _flying_streams(balls, 3)
        history = torch.stack(
            [observe(streams.next(), "pos_vel", 1) for _ in range(3)], 1
        )
        std = fit_standardizer(STAGES["s0"], frame=FRAME, clips=16)
        s = torch.randn(3, 4, generator=torch.Generator().manual_seed(5))
        mods = _gated(4, torch.eye(4), gate=-50.0)
        future = streams.fork()
        truth = torch.stack(
            [std.apply(observe(future.next(), "pos_vel", 1)) for _ in range(20)], 1
        )
        row = fork_eval(
            mods,
            s,
            streams,
            history,
            std,
            M,
            obs="pos_vel",
            balls=1,
            residual=False,
            frame=FRAME,
        )
        for h in M.horizons:
            want = float(((s - truth[:, h - 1]) ** 2).mean())
            assert row[f"fork{h}"] == pytest.approx(want, rel=1e-6)
