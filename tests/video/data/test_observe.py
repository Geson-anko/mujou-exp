import pytest
import torch

from exp.video.data.clips import fit_standardizer, make_clips
from exp.video.data.observe import Standardizer, observe, position_columns, to_states
from exp.video.data.physics import render
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
        observe(torch.zeros(1, 4), "frames", 1)
    with pytest.raises(ValueError, match="radii"):
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


class TestPixels:
    def test_the_observation_is_the_flattened_render(self):
        states = _calib()[:2, :5]  # (2, 5, 12) float64
        got = observe(states, "pixels", 3, radii=BALLS.radii, size=16)
        assert got.shape == (2, 5, 256) and got.dtype == torch.float32
        want = render(states.reshape(-1, 12), 16, BALLS.radii).reshape(2, 5, 256)
        torch.testing.assert_close(got, want.float(), rtol=0, atol=0)

    def test_float32_states_are_rendered_as_float64_is(self):
        states = _calib()[:1, :4].float().double()  # the values of float32 states
        got = observe(states.float(), "pixels", 3, radii=BALLS.radii)
        assert torch.equal(got, observe(states, "pixels", 3, radii=BALLS.radii))

    def test_chunks_do_not_change_the_images(self, monkeypatch):
        states = _calib()[:2, :9]
        whole = observe(states, "pixels", 3, radii=BALLS.radii)
        monkeypatch.setattr("exp.video.data.observe._CHUNK", 4)
        assert torch.equal(observe(states, "pixels", 3, radii=BALLS.radii), whole)

    def test_unstandardizing_gives_the_image_of_the_renderer(self):
        states = _calib()[:4, :6]
        x = observe(states, "pixels", 3, radii=BALLS.radii)
        std = Standardizer.fit(x, pooled=True)
        back = std.invert(std.apply(x).double())
        torch.testing.assert_close(back.float(), x, rtol=0, atol=1e-6)

    def test_the_calibration_set_has_mean_0_and_variance_1_over_all_pixels(self):
        x = observe(_calib(), "pixels", 3, radii=BALLS.radii)  # (32, 128, 1024)
        std = Standardizer.fit(x, pooled=True)
        assert std.mean.shape == std.std.shape == (1024,)
        assert (std.mean == std.mean[0]).all() and (std.std == std.std[0]).all()
        y = std.apply(x).double()
        assert float(y.mean()) == pytest.approx(0.0, abs=1e-6)
        assert float(y.var(correction=0)) == pytest.approx(1.0, abs=1e-5)
        assert float(x.double().mean()) == pytest.approx(float(std.mean[0]), abs=1e-6)

    def test_fit_standardizer_pools_pixels_and_only_pixels(self):
        pixels = BallsConf(radii=[0.06, 0.08, 0.10], obs="pixels")
        std = fit_standardizer(pixels, frame=1 / 30, clips=8)
        assert std.mean.shape == (1024,) and bool((std.std == std.std[0]).all())
        pos = fit_standardizer(BALLS, frame=1 / 30, clips=8)  # pos_vel, per dimension
        assert not bool((pos.std == pos.std[0]).all())

    def test_pixels_cannot_be_turned_back_to_states(self):
        with pytest.raises(ValueError, match="obs"):
            to_states(torch.zeros(1, 1024), "pixels", 3)
