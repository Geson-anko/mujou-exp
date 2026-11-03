import time

import pytest
import torch

from exp.video.data.physics import (
    STREAM_SEED_STRIDE,
    BallState,
    BallStreams,
    Physics,
    advance_,
    draw,
    render,
)
from exp.video.schema.data import BallsConf, PhysicsConf
from tests.video.data import reference_balls as ref

FRAME = 1 / 30

# The physics of the reference test: it changes in time (a short period, so
# 2000 frames see it change), with a drag and a lossy restitution.
CHANGING = dict(gamma0=1.0, gamma_amp=0.5, e0=0.6, e_amp=0.5, drag=0.1, period=20.0)
CONSERVED = dict(gamma0=1.0, gamma_amp=0.0, e0=1.0, e_amp=0.0, drag=0.0, period=600.0)


def _streams(n: int, radii, physics: dict, conserve: bool, seed: int) -> BallStreams:
    return BallStreams(
        n,
        radii=radii,
        physics=Physics(**physics, conserve=conserve),
        frame=FRAME,
        substeps=8,
        seed=seed,
    )


def _reference(i: int, radii, physics: dict, conserve: bool, seed: int) -> ref.Video:
    return ref.Video(
        radii=tuple(radii),
        frame=FRAME,
        physics=ref.Physics(**physics, conserve=conserve),
        episode=None,
        substeps=8,
        seed=seed + STREAM_SEED_STRIDE * i,
    )


@pytest.mark.parametrize("radii", [(0.10,), (0.06, 0.08, 0.10)])
@pytest.mark.parametrize("conserve", [False, True])
@pytest.mark.parametrize("physics", [CHANGING, CONSERVED], ids=["changing", "kept"])
def test_stream_i_is_the_reference_video_of_its_seed(radii, conserve, physics):
    """Every operation is the reference's on the same doubles, so the states
    are equal bit for bit: the float32 frames and the float64 state."""
    n, seed, frames = 3, 7, 2000
    streams = _streams(n, radii, physics, conserve, seed)
    videos = [_reference(i, radii, physics, conserve, seed) for i in range(n)]
    bounces = 0  # side walls, and the floor
    previous = None
    for _ in range(frames):
        got = streams.next()
        want = torch.stack([v.next().state for v in videos])
        assert torch.equal(got, want)
        if previous is not None:
            side = previous[:, 2::4] * got[:, 2::4] < 0
            floor = (previous[:, 3::4] < 0) & (got[:, 3::4] > 0)
            bounces += int(side.sum() + floor.sum())
        previous = got
    assert bounces > 20  # both branches of the bounces were taken
    d = streams.state_dict()
    for i, v in enumerate(videos):
        assert d["pos"][i].tolist() == v._pos  # the reference's lists
        assert d["vel"][i].tolist() == v._vel


def test_draw_draws_one_clip_after_the_other_as_the_reference():
    radii = (0.06, 0.10)
    g, g_ref = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
    s = draw(5, radii, g)
    for i in range(5):
        pos, vel = ref.draw_balls(radii, g_ref)
        assert s.pos[i].tolist() == pos
        assert s.vel[i].tolist() == vel


@pytest.mark.parametrize("wall", ["left", "right", "bottom", "top"])
def test_a_ball_hitting_a_wall_keeps_e_of_its_speed(wall):
    """No gravity, no drag: a ball at 1 per second toward a wall 0.4 away
    bounces once in a second and comes back at e of its speed."""
    physics = Physics(**{**CONSERVED, "gamma0": 0.0, "e0": 0.5}, conserve=False)
    axis, sign = {"left": (0, -1), "right": (0, 1), "bottom": (1, -1), "top": (1, 1)}[
        wall
    ]
    vel = torch.zeros(1, 1, 2, dtype=torch.float64)
    vel[0, 0, axis] = sign * 1.0
    s = BallState(torch.full((1, 1, 2), 0.5, dtype=torch.float64), vel)
    advance_(s, torch.tensor([0.1], dtype=torch.float64), physics, 0.0, 1.0, 240)
    assert float(s.vel[0, 0, axis]) == pytest.approx(-sign * 0.5)
    assert float(s.vel[0, 0, 1 - axis]) == 0.0
    assert 0.1 <= float(s.pos[0, 0, axis]) <= 0.9


def _modified_energy(pos: torch.Tensor, vel: torch.Tensor, h: float) -> torch.Tensor:
    r"""$\tilde E = |v|^2 / 2 + \gamma y - \gamma h v_y / 2$ with $\gamma = 1$."""
    return 0.5 * (vel**2).sum(-1) + pos[..., 1] - 0.5 * h * vel[..., 1]


@pytest.mark.slow
def test_conserve_keeps_the_modified_energy_of_each_ball_across_the_bounces():
    r"""Semi-implicit Euler keeps $\tilde E$ exactly between the bounces, and with
    ``conserve`` the bounces keep it too: it holds to rounding over 10
    minutes, for every ball of every stream. The float64 state keeps it to
    about 5e-14 (the reference's float32 frames to 1e-5)."""
    streams = _streams(4, (0.06, 0.08, 0.10), CONSERVED, True, seed=1)
    h = FRAME / 8
    streams.next()
    d = streams.state_dict()
    e0 = _modified_energy(d["pos"], d["vel"], h)
    for _ in range(18_000):
        streams.next()
    d = streams.state_dict()
    torch.testing.assert_close(
        _modified_energy(d["pos"], d["vel"], h), e0, rtol=0, atol=1e-10
    )


def test_without_conserve_the_bounces_change_the_modified_energy():
    streams = _streams(4, (0.06, 0.08, 0.10), CONSERVED, False, seed=1)
    h = FRAME / 8
    streams.next()
    d = streams.state_dict()
    e0 = _modified_energy(d["pos"], d["vel"], h)
    for _ in range(1800):
        streams.next()
    d = streams.state_dict()
    assert float((_modified_energy(d["pos"], d["vel"], h) - e0).abs().max()) > 1e-4


def test_the_future_of_a_fork_is_that_of_the_original():
    streams = _streams(3, (0.06, 0.10), CHANGING, False, seed=2)
    for _ in range(50):
        streams.next()
    fork = streams.fork()
    ahead = [fork.next() for _ in range(200)]
    for want in ahead:
        assert torch.equal(streams.next(), want)


def test_streams_loaded_from_a_state_dict_continue_as_if_never_stopped():
    straight = _streams(3, (0.06, 0.10), CHANGING, True, seed=2)
    stopped = _streams(3, (0.06, 0.10), CHANGING, True, seed=2)
    for _ in range(100):
        straight.next()
        stopped.next()
    saved = stopped.state_dict()
    resumed = _streams(3, (0.06, 0.10), CHANGING, True, seed=99)  # other balls
    resumed.load_state_dict(saved)
    for _ in range(100):
        assert torch.equal(resumed.next(), straight.next())


def test_load_state_dict_rejects_other_shapes():
    d = _streams(3, (0.10,), CONSERVED, True, seed=0).state_dict()
    with pytest.raises(ValueError, match="shape"):
        _streams(2, (0.10,), CONSERVED, True, seed=0).load_state_dict(d)


def test_of_builds_the_streams_of_a_config():
    conf = BallsConf(radii=[0.06, 0.10], physics=PhysicsConf(substeps=4, drag=0.1))
    got = BallStreams.of(conf, 2, frame=FRAME, seed=5)
    want = BallStreams(
        2,
        radii=[0.06, 0.10],
        physics=Physics.from_conf(conf.physics),
        frame=FRAME,
        substeps=4,
        seed=5,
    )
    for _ in range(30):
        assert torch.equal(got.next(), want.next())


def test_render_is_the_reference_render():
    states = _streams(6, (0.06, 0.08, 0.10), CHANGING, False, seed=4).next()
    radii = (0.06, 0.08, 0.10)
    assert torch.equal(render(states, 32, radii), ref.render(states, 32, radii))


@pytest.mark.slow
def test_thirty_minutes_of_128_streams_take_less_than_a_minute():
    """M14 streams 30 minutes of B = 128; the physics must not be the cost."""
    streams = _streams(128, (0.06, 0.08, 0.10), CHANGING, True, seed=0)
    start = time.perf_counter()
    for _ in range(30 * 60 * 30):
        streams.next()
    assert time.perf_counter() - start < 60.0
