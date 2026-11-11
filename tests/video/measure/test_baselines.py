import math

import pytest
import torch

from exp.video.data.clips import contact_mask, make_clips, simulate, windows
from exp.video.data.observe import Standardizer, observe
from exp.video.data.physics import BallState
from exp.video.measure.baselines import (
    baselines,
    const_accel,
    const_velocity,
    const_velocity_observed,
    copy_pred,
    errors,
    mean_pred,
    true_physics,
)
from exp.video.schema.data import BallsConf, PhysicsConf
from exp.video.schema.measure import MeasureConf

FRAME = 1 / 30
OBSERVED, HORIZON = 16, 20
RADII = (0.06, 0.10)
CHANGING = PhysicsConf(
    gamma0=1.0, gamma_amp=0.5, e0=0.6, e_amp=0.5, drag=0.1, period=20.0, conserve=False
)


def _flight(gamma: float) -> tuple[BallsConf, torch.Tensor]:
    """Two balls that stay off the walls for the 36 frames of a window (with
    gravity, both are thrown up, to tops at 0.68 and 0.48), with no drag and a
    constant gravity: the states ``(1, 36, 8)``."""
    physics = PhysicsConf(gamma0=gamma, e0=1.0, drag=0.0, conserve=True)
    balls = BallsConf(radii=list(RADII), physics=physics)
    pos = torch.tensor([[[0.4, 0.5], [0.4, 0.3]]], dtype=torch.float64)
    vy = (0.6, 0.6) if gamma else (0.05, 0.2)
    vel = torch.tensor([[[0.1, vy[0]], [0.1, vy[1]]]], dtype=torch.float64)
    states = simulate(
        BallState(pos, vel), balls, frames=OBSERVED + HORIZON, frame=FRAME
    )
    assert not contact_mask(states, 3, [33], radii=RADII, frame=FRAME).any()
    return balls, states


@pytest.mark.parametrize("obs", ["pos_vel", "pos"])
def test_constant_velocity_is_exact_without_gravity_and_walls(obs):
    _, states = _flight(gamma=0.0)
    raw = observe(states, obs, 2)
    pred = const_velocity(raw, obs, 2, OBSERVED, HORIZON, FRAME)
    torch.testing.assert_close(pred, raw[:, OBSERVED:], rtol=0, atol=1e-6)
    if obs == "pos_vel":
        pred = const_velocity_observed(raw, 2, OBSERVED, HORIZON, FRAME)
        torch.testing.assert_close(pred, raw[:, OBSERVED:], rtol=0, atol=1e-6)


@pytest.mark.parametrize("obs", ["pos_vel", "pos"])
def test_constant_acceleration_is_exact_under_a_constant_gravity(obs):
    r"""Semi-implicit Euler under a constant gravity without drag or walls puts
    the ball at $y_n = y_0 + h n v_0 - g h^2 n (n + 1) / 2$ after $n$
    substeps, a quadratic in $n$, so the positions of the frames are a
    quadratic in the frame and three of them extrapolate exactly. (The
    velocity columns are differences of positions, not the velocity of the
    state, so only the positions are exact.)"""
    _, states = _flight(gamma=1.0)
    raw = observe(states, obs, 2)
    pred = const_accel(raw, obs, 2, OBSERVED, HORIZON, FRAME)
    cols = [0, 1, 4, 5] if obs == "pos_vel" else [0, 1, 2, 3]
    torch.testing.assert_close(
        pred[..., cols], raw[:, OBSERVED:, cols], rtol=0, atol=1e-6
    )
    # and constant velocity misses the fall: (h (h + 1) / 2) g dt^2 at h = 20
    cv = const_velocity(raw, obs, 2, OBSERVED, HORIZON, FRAME)
    miss = float((cv[0, -1, cols[1]] - raw[0, -1, cols[1]]).abs())
    assert miss == pytest.approx(20 * 21 / 2 * FRAME**2, rel=1e-6)


def test_the_velocity_columns_of_constant_acceleration_are_its_differences():
    _, states = _flight(gamma=1.0)
    raw = observe(states, "pos_vel", 2)
    pred = const_accel(raw, "pos_vel", 2, OBSERVED, HORIZON, FRAME)
    pos = pred[..., [0, 1, 4, 5]]
    prev = torch.cat([raw[:, OBSERVED - 1 : OBSERVED, [0, 1, 4, 5]], pos[:, :-1]], 1)
    torch.testing.assert_close(
        pred[..., [2, 3, 6, 7]], (pos - prev) / FRAME, rtol=0, atol=1e-9
    )


def test_copy_repeats_the_last_observed_frame():
    x = torch.randn(3, 36, 4)
    pred = copy_pred(x, OBSERVED, HORIZON)
    assert pred.shape == (3, HORIZON, 4)
    assert torch.equal(pred, x[:, OBSERVED - 1 : OBSERVED].expand(-1, HORIZON, -1))


def test_the_mean_has_error_1_on_the_set_it_was_fitted_on():
    """0 in the standardized space against data of mean 0 and variance 1 per
    dimension: the mean squared error over every frame is exactly 1."""
    balls = BallsConf(radii=list(RADII), physics=CHANGING)
    x = observe(
        make_clips(balls, split="calib", clips=16, frame=FRAME).states, "pos_vel", 2
    )
    std = Standardizer.fit(x)
    n, length, d = x.shape
    errs = errors(mean_pred(n, length, d), std.apply(x), range(1, length + 1))
    assert sum(errs.values()) / length == pytest.approx(1.0, abs=1e-6)


def test_errors_average_over_the_masked_windows():
    target = torch.zeros(3, 2, 2, dtype=torch.float64)
    pred = torch.tensor([1.0, 2.0, 3.0]).double().view(3, 1, 1).expand(3, 2, 2)
    pred = pred * torch.tensor([1.0, 2.0]).double().view(1, 2, 1)  # h = 2: twice
    assert errors(pred, target, [1, 2]) == pytest.approx({1: 14 / 3, 2: 56 / 3})
    mask = torch.tensor([[True, False], [False, False], [True, True]])
    got = errors(pred, target, [1, 2], mask)
    assert got[1] == 5.0 and got[2] == 36.0
    assert math.isnan(errors(pred, target, [1], mask[:, 1:2] & False)[1])


def _clips(obs: str):
    balls = BallsConf(radii=list(RADII), obs=obs, physics=CHANGING)
    clips = make_clips(balls, split="val", clips=16, frame=FRAME)
    std = Standardizer.fit(observe(clips.states, obs, 2))
    return balls, clips, std


def test_the_true_physics_is_the_future_of_the_clips():
    balls, clips, _ = _clips("pos_vel")
    _, states = windows(clips, "pos_vel", Standardizer.fit(clips.states))
    future = true_physics(clips, balls, OBSERVED, HORIZON)
    assert torch.equal(future, states[:, OBSERVED:])


@pytest.mark.parametrize("obs", ["pos_vel", "pos"])
def test_baselines_names_every_error_and_true_physics_is_zero(obs):
    balls, clips, std = _clips(obs)
    m = MeasureConf()
    out = baselines(clips, balls, std, m)
    names = ["mean", "copy", "cv", "ca", "true"] + (
        ["cvobs"] if obs == "pos_vel" else []
    )
    suffixes = ["", "@contact", "@free", "@ball0", "@ball1"]
    want = {f"{n}{h}{s}" for n in names for h in m.horizons for s in suffixes}
    assert set(out) == want
    for h in m.horizons:
        assert out[f"true{h}"] == 0.0


def test_contact_and_free_and_the_balls_split_the_error():
    """The error is a mean over the windows, so it is the mean of those with
    and without a contact weighted by their counts; and over the columns, so
    it is the mean of the balls' (two balls of 4 columns each)."""
    balls, clips, std = _clips("pos_vel")
    m = MeasureConf()
    out = baselines(clips, balls, std, m)
    _, states = windows(clips, "pos_vel", std)
    contact = contact_mask(states, OBSERVED, m.horizons, radii=RADII, frame=FRAME)
    contact = contact.any(-1)
    n = len(states)
    for i, h in enumerate(m.horizons):
        k = int(contact[:, i].sum())
        assert 0 < k < n  # both kinds of window exist
        for name in ("copy", "cv", "ca"):
            mixed = (
                k * out[f"{name}{h}@contact"] + (n - k) * out[f"{name}{h}@free"]
            ) / n
            assert mixed == pytest.approx(out[f"{name}{h}"], rel=1e-9)
            balls_mean = (out[f"{name}{h}@ball0"] + out[f"{name}{h}@ball1"]) / 2
            assert balls_mean == pytest.approx(out[f"{name}{h}"], rel=1e-9)


def test_baselines_rejects_a_measure_the_clips_do_not_fit():
    balls, clips, std = _clips("pos")
    with pytest.raises(ValueError, match="fit"):
        baselines(clips, balls, std, MeasureConf(horizons=[1, 40]))
    with pytest.raises(ValueError, match="STENCIL"):
        baselines(clips, balls, std, MeasureConf(observed=2))


def test_the_true_physics_rejects_balls_the_clips_were_not_made_of():
    balls, clips, std = _clips("pos_vel")
    other = BallsConf(radii=list(RADII), physics=PhysicsConf())
    with pytest.raises(ValueError, match="clips were made of"):
        true_physics(clips, other, OBSERVED, HORIZON)
    with pytest.raises(ValueError, match="clips were made of"):
        baselines(clips, other, std, MeasureConf())


def test_constant_acceleration_is_exact_on_the_free_windows():
    """Under the constant gravity of M14 (no drag, e = 1), a window without a
    contact from the first frame constant acceleration reads to the predicted
    one is a free flight, whose positions are a quadratic in the frame (see
    above): ``ca`` is exact there up to rounding, while the windows with a
    contact (some with a bounce among the three frames read) are not."""
    balls = BallsConf(radii=[0.10], obs="pos", physics=PhysicsConf())
    clips = make_clips(balls, split="val", clips=64, frame=FRAME)
    std = Standardizer.fit(observe(clips.states, "pos", 1))
    m = MeasureConf()
    out = baselines(clips, balls, std, m)
    for h in m.horizons:
        assert out[f"ca{h}@free"] < 1e-20
        assert out[f"ca{h}@contact"] > 1e-4
        assert out[f"cv{h}@free"] > 1e-10  # the gravity cv misses
