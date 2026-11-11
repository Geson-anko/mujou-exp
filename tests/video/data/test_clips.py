import math

import pytest
import torch

from exp.video.data.clips import (
    SPLIT_SEEDS,
    STENCIL,
    contact_mask,
    fit_standardizer,
    make_clips,
    simulate,
    windows,
)
from exp.video.data.observe import Standardizer, observe
from exp.video.data.physics import BallState
from exp.video.schema.data import BallsConf, PhysicsConf
from tests.video.data import reference_balls as ref

FRAME = 1 / 30
SUBSTEPS = 8
H = FRAME / SUBSTEPS
OBSERVED = 16
HORIZONS = list(range(1, 21))
CHANGING = PhysicsConf(
    gamma0=1.0, gamma_amp=0.5, e0=0.6, e_amp=0.5, drag=0.1, period=20.0, conserve=False
)


def _balls(radii=(0.06, 0.10), physics: PhysicsConf = CHANGING, obs="pos_vel"):
    return BallsConf(radii=list(radii), obs=obs, physics=physics)


def test_the_clips_are_reference_balls_drawn_one_after_the_other():
    balls = _balls()
    c = make_clips(balls, split="val", clips=3, length=40, frame=FRAME)
    g = torch.Generator().manual_seed(SPLIT_SEEDS["val"])
    physics = ref.Physics(
        gamma0=1.0, gamma_amp=0.5, e0=0.6, e_amp=0.5, drag=0.1, period=20.0
    )
    radii = (0.06, 0.10)
    for i in range(3):
        pos, vel = ref.draw_balls(radii, g)
        for k in range(40):
            if k:
                ref.advance(pos, vel, radii, physics, k * FRAME - FRAME, FRAME, 8)
            want = [x for p, v in zip(pos, vel, strict=True) for x in (*p, *v)]
            assert c.states[i, k].tolist() == want


def test_the_windows_are_four_per_clip_within_it():
    c = make_clips(_balls(), split="test", clips=5, frame=FRAME)
    assert c.states.shape == (5, 128, 8)
    assert c.starts.shape == (20, 2)
    assert c.starts[:4, 1].tolist() == [0, 30, 60, 90]
    assert int((c.starts[:, 1] + c.span).max()) <= 128
    std = Standardizer.fit(observe(c.states, "pos", 2))
    x, states = windows(c, "pos", std)
    assert x.shape == (20, 36, 4) and x.dtype == torch.float32
    for w in (0, 7, 19):
        clip, first = c.starts[w].tolist()
        assert torch.equal(states[w], c.states[clip, first : first + 36])
    torch.testing.assert_close(x, std.apply(observe(states, "pos", 2)).float())


def test_each_split_has_its_own_fixed_clips():
    def states(split: str) -> torch.Tensor:
        return make_clips(_balls(), split=split, clips=2, length=40, frame=FRAME).states

    assert torch.equal(states("val"), states("val"))
    assert not torch.equal(states("val"), states("test"))
    assert not torch.equal(states("val"), states("calib"))
    with pytest.raises(ValueError, match="split"):
        states("train")


def test_the_standardization_is_fitted_on_the_calibration_clips_only():
    balls = _balls()
    calib = make_clips(balls, split="calib", clips=16, frame=FRAME)
    got = fit_standardizer(balls, frame=FRAME, clips=16)
    want = Standardizer.fit(observe(calib.states, "pos_vel", 2))
    assert torch.equal(got.mean, want.mean) and torch.equal(got.std, want.std)


def test_simulate_from_a_frame_of_a_clip_gives_the_clip():
    balls = _balls()
    c = make_clips(balls, split="val", clips=4, length=60, frame=FRAME)
    s = c.states[:, 25].unflatten(-1, (2, 4))
    start = BallState(s[..., :2].clone(), s[..., 2:].clone())
    future = simulate(start, balls, frames=35, frame=FRAME, first=25)
    assert torch.equal(future, c.states[:, 25:])


def _first_substep(y0: float, v0: float, g: float, past) -> int:
    r"""The first substep $n$ whose free-flight position past the wall:
    semi-implicit Euler gives $v_n = v_0 - g h n$ and
    $y_n = y_0 + h n v_0 - g h^2 n (n + 1) / 2$."""
    n = 1
    while not past(y0 + H * n * v0 - g * H * H * n * (n + 1) / 2):
        n += 1
    return n


# (axis, position, velocity, gravity, the test that a position is past the wall)
FLIGHTS = {
    "floor": (1, 0.5, 0.0, 1.0, lambda y: y < 0.1),  # dropped from rest
    "ceiling": (1, 0.2, 1.5, 1.0, lambda y: y > 0.9),  # thrown up hard
    "right": (0, 0.5, 0.55, 0.0, lambda x: x > 0.9),  # no gravity along x
}


def _flight_states(axis: int, p0: float, v0: float, g: float, e: float = 1.0):
    """One ball of radius 0.1 at ``p0`` with velocity ``v0`` along ``axis``
    (0.5 and 0 along the other), without drag: the states ``(1, 36, 4)``."""
    pos = torch.full((1, 1, 2), 0.5, dtype=torch.float64)
    vel = torch.zeros(1, 1, 2, dtype=torch.float64)
    pos[0, 0, axis], vel[0, 0, axis] = p0, v0
    physics = PhysicsConf(gamma0=g, e0=e, drag=0.0, conserve=e == 1.0)
    balls = _balls(radii=(0.1,), physics=physics)
    return simulate(BallState(pos, vel), balls, frames=36, frame=FRAME)


@pytest.mark.parametrize("e", [1.0, 0.5])
@pytest.mark.parametrize("flight", FLIGHTS)
def test_the_first_contact_is_the_hand_computed_frame(flight, e):
    """The bounce happens in the substep n where the free flight leaves the box;
    frame ceil(n / substeps) is the first to show it, so the first horizon
    flagged is that frame minus the last observed one (with any restitution:
    the bounce reverses the velocity, -e v)."""
    axis, p0, v0, g, past = FLIGHTS[flight]
    n = _first_substep(p0, v0, g, past)
    first = math.ceil(n / SUBSTEPS) - (OBSERVED - 1)
    assert 1 <= first <= 20  # the contact is within the horizons
    states = _flight_states(axis, p0, v0, g, e)
    mask = contact_mask(states, OBSERVED, HORIZONS, radii=(0.1,), frame=FRAME)
    assert mask.shape == (1, 20, 1)
    assert mask[0, :, 0].tolist() == [h >= first for h in HORIZONS]


def _drop_height(frame: int) -> float:
    """A height (on a grid of 1e-3) from which a ball dropped from rest under
    a gravity of 1 first passes the floor (y < 0.1) in ``frame``."""
    for i in range(110, 900):
        y0 = i / 1000
        n = _first_substep(y0, 0.0, 1.0, lambda y: y < 0.1)
        if math.ceil(n / SUBSTEPS) == frame:
            return y0
    raise AssertionError(frame)


@pytest.mark.parametrize("frame", range(OBSERVED - STENCIL - 1, OBSERVED))
def test_a_bounce_among_the_frames_the_extrapolation_reads_is_a_contact(frame):
    """A bounce shown first by frame f >= observed - STENCIL + 1 lies between
    two of the last STENCIL observed frames, which constant acceleration reads:
    a contact at every horizon. One shown by frame observed - STENCIL lies
    before them: no contact at h = 1 (the ball, e = 1, is on its way back up
    to the drop height)."""
    states = _flight_states(1, _drop_height(frame), 0.0, 1.0)
    mask = contact_mask(states, OBSERVED, HORIZONS, radii=(0.1,), frame=FRAME)
    if frame > OBSERVED - STENCIL:
        assert mask[0, :, 0].all()
    else:
        assert not mask[0, 0, 0]


def test_contact_mask_rejects_fewer_observed_frames_than_the_stencil():
    states = _flight_states(1, 0.5, 0.0, 1.0)
    with pytest.raises(ValueError, match="STENCIL"):
        contact_mask(states, STENCIL - 1, [1], radii=(0.1,), frame=FRAME)


def test_the_clips_know_the_balls_they_were_made_of():
    balls = _balls()
    c = make_clips(balls, split="val", clips=2, length=40, frame=FRAME)
    c.check(_balls(obs="pos"))  # the observation does not matter
    with pytest.raises(ValueError, match="radii"):
        c.check(_balls(radii=(0.06, 0.12)))
    with pytest.raises(ValueError, match="physics|radii"):
        c.check(_balls(physics=PhysicsConf()))


@pytest.mark.parametrize(
    ("y0", "vy", "g"),
    [(0.5, 0.05, 0.0), (0.3, 0.6, 1.0)],
    ids=["slow-in-the-middle", "top-of-a-flight"],
)
def test_a_ball_far_from_the_walls_touches_none(y0, vy, g):
    """Slow in the middle of the box, or thrown up to a top at 0.48 (far below
    the ceiling at 0.9): the flip of v_y at the top is not a contact."""
    pos = torch.tensor([[[0.4, y0]]], dtype=torch.float64)
    vel = torch.tensor([[[0.1, vy]]], dtype=torch.float64)
    physics = PhysicsConf(gamma0=g, e0=1.0, drag=0.0, conserve=True)
    states = simulate(
        BallState(pos, vel), _balls((0.1,), physics), frames=36, frame=FRAME
    )
    if g:
        assert (states[0, 16:, 3] < 0).any()  # the top is within the horizons
    assert not contact_mask(states, OBSERVED, HORIZONS, radii=(0.1,), frame=FRAME).any()


def test_the_contacts_are_per_ball():
    pos = torch.tensor([[[0.5, 0.5], [0.4, 0.5]]], dtype=torch.float64)
    vel = torch.tensor([[[0.55, 0.0], [0.1, 0.05]]], dtype=torch.float64)
    physics = PhysicsConf(gamma0=0.0, e0=1.0, drag=0.0, conserve=True)
    balls = _balls((0.1, 0.1), physics)
    states = simulate(BallState(pos, vel), balls, frames=36, frame=FRAME)
    mask = contact_mask(states, OBSERVED, HORIZONS, radii=(0.1, 0.1), frame=FRAME)
    assert mask[0, -1].tolist() == [True, False]
