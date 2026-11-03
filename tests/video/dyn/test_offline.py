import pytest
import torch

from exp.video.data.clips import fit_standardizer
from exp.video.data.observe import observe
from exp.video.data.physics import BallStreams
from exp.video.dyn.modules import DynamicsModules, Top, build_dynamics
from exp.video.dyn.offline import OfflineFit, fit_offline, stream_windows, window_loss
from exp.video.programs.dynamics import PROGRAM
from exp.video.schema.data import BallsConf, DataConf
from exp.video.schema.dyn import DynamicsConf
from exp.video.schema.learn import OfflineConf
from exp.video.schema.measure import MeasureConf
from exp.video.schema.run import RunConf

FRAME = 1 / 30


def _conf(task: str = "offline", **dynamics) -> RunConf:
    return RunConf(
        name="test",
        task=task,
        data=DataConf(streams=3, balls=BallsConf(radii=[0.06, 0.10], obs="pos")),
        dynamics=DynamicsConf(d_s=8, **dynamics),
        offline=OfflineConf(frames=72, window=36, steps=4, batch=5, every=2),
        measure=MeasureConf(clips=4, windows=2),
    )


def _mods(residual: bool, nodes: int = 2) -> DynamicsModules:
    conf = DynamicsConf(d_s=5, nodes=nodes, residual=residual)
    mods = build_dynamics(conf, 4, 4, seed=3)
    mods.step.double()
    mods.readout.double()
    if residual:  # a readout that is not 0, so the gradients reach the step
        with torch.no_grad():
            g = torch.Generator().manual_seed(4)
            mods.readout.weight.copy_(
                0.3 * torch.randn(4, 5, generator=g, dtype=torch.float64)
            )
    return mods


def _params(mods: DynamicsModules) -> list[torch.Tensor]:
    return [*mods.step.parameters(), *mods.readout.parameters()]


def _cut_loss(mods: DynamicsModules, x, consts, *, residual: bool) -> torch.Tensor:
    """The loss with the input and the state of each frame held at
    ``consts``: frame j costs (f(c_{j-1}, s_{j-1}) - x_j)^2."""
    top = Top(mods.step.d_s)
    losses = []
    for j, (c, s) in enumerate(consts, start=1):
        s_j = mods.step(c, s)
        y = mods.readout(top(s_j))
        x_hat = c + y if residual else y
        losses.append(((x_hat - x[:, j]) ** 2).mean(-1))
    return torch.stack(losses, 1).mean()


def _consts(mods: DynamicsModules, x, observed: int, residual: bool):
    """The input and the state each frame starts from, at the weights now."""
    top = Top(mods.step.d_s)
    s = x.new_zeros(x.shape[0], mods.step.d_state)
    c = x[:, 0]
    out = []
    with torch.no_grad():
        for j in range(1, x.shape[1]):
            out.append((c, s))
            s = mods.step(c, s)
            y = mods.readout(top(s))
            x_hat = c + y if residual else y
            c = x[:, j] if j < observed else x_hat
    return out


def _finite_difference(loss_of, params, entries: int = 4, eps: float = 1e-6):
    """Central differences of ``loss_of()`` at a few entries of each
    parameter: (indices, values) per parameter."""
    g = torch.Generator().manual_seed(0)
    out = []
    for p in params:
        idx = torch.randint(p.numel(), (entries,), generator=g)
        vals = []
        flat = p.data.view(-1)
        for i in idx.tolist():
            old = float(flat[i])
            flat[i] = old + eps
            up = float(loss_of())
            flat[i] = old - eps
            down = float(loss_of())
            flat[i] = old
            vals.append((up - down) / (2 * eps))
        out.append((idx, torch.tensor(vals, dtype=torch.float64)))
    return out


class TestStreamWindows:
    def test_a_window_is_consecutive_frames_of_one_stream(self):
        conf = _conf()
        balls = conf.data.balls
        assert balls is not None
        std = fit_standardizer(balls, frame=FRAME, clips=16)
        w = stream_windows(conf, 20, 8, std=std)  # 2 windows per stream, 4 frames left
        assert w.shape == (3 * 2, 8, 4)
        streams = BallStreams.of(balls, 3, frame=FRAME, seed=conf.seed)
        frames = torch.stack([observe(streams.next(), "pos", 2) for _ in range(20)], 1)
        frames = std.apply(frames)
        for b in range(3):
            for j in range(2):
                torch.testing.assert_close(
                    w[b * 2 + j], frames[b, 8 * j : 8 * (j + 1)], rtol=0, atol=0
                )


class TestWindowLoss:
    @pytest.mark.parametrize("residual", [False, True], ids=["", "residual"])
    def test_local_is_the_gradient_of_the_loss_cut_at_one_frame(self, residual):
        """With ``local``, the gradient is that of the loss whose input and
        state of each frame are constants (held at their values now), by
        central differences in float64."""
        mods = _mods(residual)
        x = torch.randn(
            3, 7, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(1)
        )
        params = _params(mods)
        loss = window_loss(mods, x, observed=4, local=True, residual=residual)
        grads = torch.autograd.grad(loss, params)
        consts = _consts(mods, x, 4, residual)
        with torch.no_grad():
            fd = _finite_difference(
                lambda: _cut_loss(mods, x, consts, residual=residual), params
            )
        for g, (idx, want) in zip(grads, fd, strict=True):
            torch.testing.assert_close(g.reshape(-1)[idx], want, rtol=1e-5, atol=1e-9)

    @pytest.mark.parametrize("residual", [False, True], ids=["", "residual"])
    def test_through_time_is_the_gradient_of_the_whole_loss(self, residual):
        mods = _mods(residual)
        x = torch.randn(
            3, 7, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(1)
        )
        params = _params(mods)
        loss = window_loss(mods, x, observed=4, local=False, residual=residual)
        grads = torch.autograd.grad(loss, params)
        with torch.no_grad():
            fd = _finite_difference(
                lambda: window_loss(
                    mods, x, observed=4, local=False, residual=residual
                ),
                params,
            )
        for g, (idx, want) in zip(grads, fd, strict=True):
            torch.testing.assert_close(g.reshape(-1)[idx], want, rtol=1e-5, atol=1e-9)
        local = window_loss(mods, x, observed=4, local=True, residual=residual)
        assert float(local.detach()) == pytest.approx(
            float(loss.detach()), rel=1e-12
        )  # the same value


class TestOfflineFit:
    def test_a_fit_resumed_from_its_state_goes_on_as_one_never_stopped(self):
        x = torch.randn(10, 6, 4, generator=torch.Generator().manual_seed(2))

        def fit(mods):
            return OfflineFit(
                mods,
                x,
                batch=3,
                lr=1e-2,
                local=False,
                observed=3,
                residual=False,
                gen=torch.Generator().manual_seed(0),
            )

        a = build_dynamics(DynamicsConf(d_s=6), 4, 4, seed=1)
        whole = fit(a)
        losses = [float(whole.step()) for _ in range(4)]
        b = build_dynamics(DynamicsConf(d_s=6), 4, 4, seed=1)
        first = fit(b)
        for _ in range(2):
            first.step()
        state = first.state_dict()
        weights = [p.detach().clone() for p in _params(b)]
        c = build_dynamics(DynamicsConf(d_s=6), 4, 4, seed=9)
        with torch.no_grad():
            for p, w in zip(_params(c), weights, strict=True):
                p.copy_(w)
        second = fit(c)
        second.load_state_dict(state)
        assert [float(second.step()) for _ in range(2)] == losses[2:]
        for p, q in zip(_params(a), _params(c), strict=True):
            assert torch.equal(p, q)

    def test_fit_offline_calls_back_every_steps(self):
        x = torch.randn(10, 6, 4, generator=torch.Generator().manual_seed(2))
        mods = build_dynamics(DynamicsConf(d_s=6), 4, 4, seed=1)
        seen = []
        fit_offline(
            mods,
            x,
            steps=5,
            batch=3,
            lr=1e-2,
            local=True,
            observed=3,
            gen=torch.Generator().manual_seed(0),
            every=2,
            on_eval=lambda step, m: seen.append((step, m is mods)),
        )
        assert seen == [(2, True), (4, True)]


class TestOfflineEvaluation:
    def test_the_same_weights_evaluate_as_online(self, tmp_path):
        """The offline task evaluates with the clips, the standardization and
        the rollout of the online one: the same weights, the same row."""
        online = PROGRAM.build(_conf("online", nodes=2), tmp_path, None)
        offline = PROGRAM.build(_conf("offline", nodes=2), tmp_path, None)
        try:
            src, dst = online.modules(), offline.modules()
            with torch.no_grad():
                for name, m in src.items():
                    for p in m.parameters():
                        p.normal_(0, 0.3, generator=torch.Generator().manual_seed(7))
                    dst[name].load_state_dict(m.state_dict())
            a, b = online.evaluate("test"), offline.evaluate("test")
        finally:
            online.close()
            offline.close()
        rolls = [k for k in a if k.startswith("roll")]
        assert len(rolls) > 10
        torch.testing.assert_close(
            torch.tensor([a[k] for k in rolls]),
            torch.tensor([b[k] for k in rolls]),
            rtol=0,
            atol=0,
            equal_nan=True,
        )
