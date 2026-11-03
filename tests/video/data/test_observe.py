import pytest
import torch

from exp.video.data.clips import make_clips
from exp.video.data.observe import Standardizer, observe, position_columns, to_states
from exp.video.schema.data import BallsConf, PhysicsConf

BALLS = BallsConf(radii=[0.06, 0.08, 0.10], physics=PhysicsConf(drag=0.1, e0=0.8))


def _calib() -> torch.Tensor:
    return make_clips(BALLS, split="calib", clips=32, frame=1 / 30).states


def test_pos_vel_is_the_state_and_pos_its_position_columns():
    states = torch.arange(2 * 5 * 12, dtype=torch.float64).view(2, 5, 12)
    assert torch.equal(observe(states, "pos_vel", 3), states)
    xy = states.view(2, 5, 3, 4)[..., :2].reshape(2, 5, 6)
    assert torch.equal(observe(states, "pos", 3), xy)
    for obs in ("pos_vel", "pos"):
        cols = position_columns(obs, 3)  # (3, 2)
        assert torch.equal(observe(states, obs, 3)[..., cols].flatten(-2), xy)


def test_to_states_undoes_observe_with_zero_velocities_for_pos():
    states = _calib()
    assert torch.equal(to_states(observe(states, "pos_vel", 3), "pos_vel", 3), states)
    back = to_states(observe(states, "pos", 3), "pos", 3).view(*states.shape[:2], 3, 4)
    full = states.view(*states.shape[:2], 3, 4)
    assert torch.equal(back[..., :2], full[..., :2])
    assert not back[..., 2:].any()


@pytest.mark.parametrize("balls", [1, 3])
def test_observe_and_to_states_never_alias_their_input(balls):
    for obs in ("pos_vel", "pos"):
        states = torch.ones(2, 5, 4 * balls, dtype=torch.float64)
        observe(states, obs, balls).zero_()
        assert states.all()
        raw = observe(states, obs, balls)
        to_states(raw, obs, balls).zero_()
        assert raw.all()


def test_other_observations_are_rejected():
    with pytest.raises(ValueError, match="obs"):
        observe(torch.zeros(1, 4), "pixels", 1)


@pytest.mark.parametrize("obs", ["pos_vel", "pos"])
def test_the_calibration_set_has_mean_0_and_variance_1(obs):
    x = observe(_calib(), obs, 3)
    std = Standardizer.fit(x)
    y = std.apply(x).reshape(-1, x.shape[-1])
    torch.testing.assert_close(
        y.mean(0), torch.zeros(x.shape[-1], dtype=y.dtype), rtol=0, atol=1e-6
    )
    torch.testing.assert_close(
        y.var(0, correction=0),
        torch.ones(x.shape[-1], dtype=y.dtype),
        rtol=0,
        atol=1e-6,
    )


def test_stage_0_back_from_the_standardized_space_is_the_state():
    states = _calib()
    std = Standardizer.fit(observe(states, "pos_vel", 3))
    for dtype in (torch.float64, torch.float32):
        y = std.apply(observe(states, "pos_vel", 3).to(dtype))
        assert y.dtype == dtype
        back = to_states(std.invert(y), "pos_vel", 3)
        torch.testing.assert_close(back.double(), states, rtol=0, atol=1e-6)


def test_a_constant_dimension_is_rejected():
    x = torch.randn(4, 5, 3, dtype=torch.float64)
    x[..., 1] = 0.5
    with pytest.raises(ValueError, match="constant"):
        Standardizer.fit(x)
