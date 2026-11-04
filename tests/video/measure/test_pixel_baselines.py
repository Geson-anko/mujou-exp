import math

import pytest
import torch

from exp.video.data.clips import ClipSet, contact_mask, make_clips, simulate
from exp.video.data.observe import Standardizer, render_flat
from exp.video.data.physics import BallState
from exp.video.measure.pixel_baselines import baselines, disc_masks, region_sq
from exp.video.schema.data import BallsConf, PhysicsConf
from exp.video.schema.measure import MeasureConf

FRAME = 1 / 30
OBSERVED, HORIZON = 16, 20
RADII = (0.06, 0.10)
M = MeasureConf(observed=OBSERVED, horizons=[1, 5, 10, 20])


def _flight(gamma: float) -> tuple[BallsConf, ClipSet]:
    """A clip of one window of two balls that stay off the walls (no drag,
    constant gravity ``gamma``, with gravity thrown up to tops 0.68 and 0.48)."""
    physics = PhysicsConf(gamma0=gamma, e0=1.0, drag=0.0, conserve=True)
    balls = BallsConf(radii=list(RADII), obs="pixels", physics=physics)
    pos = torch.tensor([[[0.4, 0.5], [0.4, 0.3]]], dtype=torch.float64)
    vy = (0.6, 0.6) if gamma else (0.05, 0.2)
    vel = torch.tensor([[[0.1, vy[0]], [0.1, vy[1]]]], dtype=torch.float64)
    states = simulate(
        BallState(pos, vel), balls, frames=OBSERVED + HORIZON, frame=FRAME
    )
    assert not contact_mask(states, 3, [HORIZON], radii=RADII, frame=FRAME).any()
    clips = ClipSet(
        states,
        torch.zeros(1, 2, dtype=torch.long),
        OBSERVED,
        HORIZON,
        FRAME,
        RADII,
        balls.physics,
    )
    return balls, clips


def _std(balls: BallsConf, clips: ClipSet) -> Standardizer:
    """The standardization of the images of some other balls."""
    other = make_clips(balls, split="calib", clips=8, frame=FRAME)
    return Standardizer.fit(
        render_flat(other.states, balls.radii, balls.size), pooled=True
    )


class TestBars:
    def test_the_rendered_constant_velocity_is_exact_without_gravity_and_walls(self):
        balls, clips = _flight(gamma=0.0)
        row = baselines(clips, balls, _std(balls, clips), M)
        for h in M.horizons:
            assert row[f"cv{h}"] < 1e-12
            assert row[f"ca{h}"] < 1e-12
            assert row[f"true{h}"] == 0.0
            assert row[f"copy{h}"] > 0  # the balls do move
        assert row["cv20@ball1"] < 1e-12

    def test_the_rendered_constant_acceleration_is_exact_under_a_constant_gravity(
        self,
    ):
        balls, clips = _flight(gamma=1.0)
        row = baselines(clips, balls, _std(balls, clips), M)
        for h in M.horizons:
            assert row[f"ca{h}"] < 1e-12
            assert row[f"true{h}"] == 0.0
        assert row["cv20"] > 1e-3  # constant velocity does not know gravity

    def test_the_mean_has_error_1_on_the_pixels_of_the_calibration_clips(self):
        balls = BallsConf(radii=list(RADII), obs="pixels")
        clips = make_clips(balls, split="val", clips=8, frame=FRAME)
        std = _std(balls, clips)
        row = baselines(clips, balls, std, M)
        for h in M.horizons:
            assert row[f"mean{h}"] == pytest.approx(1.0, abs=0.05)
        for suffix in ("", "@contact", "@free", "@contactball", "@ball0", "@ball1"):
            assert row[f"true10{suffix}"] in (0.0,) or math.isnan(
                row[f"true10{suffix}"]
            )

    def test_copy_is_the_last_image_and_the_keys_are_those_of_the_state_bars(self):
        balls, clips = _flight(gamma=1.0)
        std = _std(balls, clips)
        row = baselines(clips, balls, std, M)
        states = clips.states
        last = std.apply(render_flat(states[:, OBSERVED - 1], RADII, 32))
        for h in M.horizons:
            target = std.apply(render_flat(states[:, OBSERVED + h - 1], RADII, 32))
            assert row[f"copy{h}"] == pytest.approx(
                float(((last - target) ** 2).mean()), rel=1e-5
            )
            for name in ("mean", "copy", "cv", "ca", "true"):
                for suffix in ("", "@contact", "@free", "@contactball", "@ball0"):
                    assert f"{name}{h}{suffix}" in row
        assert math.isnan(row["copy5@contact"])  # no window with a contact


class TestRegions:
    def test_the_disc_pixels_by_hand(self):
        """On 8 x 8 pixels (centers at (i + 1/2) / 8) a ball of radius 0.2 at
        the center covers, in each quadrant around it, the offsets (1/16, 1/16),
        (1/16, 3/16) and (3/16, 1/16) (squared distances 0.0078, 0.0391, 0.0391
        < 0.04) but not (3/16, 3/16) (0.0703): 12 pixels."""
        states = torch.tensor([[0.5, 0.5, 0.0, 0.0]], dtype=torch.float64)
        mask = disc_masks(states, [0.2], 8).view(8, 8)
        assert int(mask.sum()) == 12
        for row in range(8):
            for col in range(8):
                dx, dy = (col + 0.5) / 8 - 0.5, (1 - (row + 0.5) / 8) - 0.5
                assert bool(mask[row, col]) == (dx * dx + dy * dy <= 0.04)
        assert not mask[0, 0] and mask[3, 3] and mask[4, 4]

    def test_row_0_is_the_top_where_y_is_1(self):
        states = torch.tensor([[0.5, 0.95, 0.0, 0.0]], dtype=torch.float64)
        mask = disc_masks(states, [0.06], 32).view(32, 32)
        assert mask[:3].any() and not mask[16:].any()

    def test_the_region_error_of_a_vanished_ball_by_hand(self):
        """Two balls far apart; the prediction is the image of ball 0 only. The
        error is 0 on the disc of ball 0 and, on the disc of ball 1, the mean
        of the squared brightness of ball 1 there."""
        radii = [0.1, 0.1]
        states = torch.tensor([[0.25, 0.5, 0, 0, 0.75, 0.5, 0, 0]], dtype=torch.float64)
        both = render_flat(states, radii, 16)
        first = render_flat(states[:, :4], radii[:1], 16)
        sq = (first - both) ** 2  # (1, 256)
        masks = disc_masks(states, radii, 16)  # (1, 2, 256)
        got = region_sq(sq, masks)[0]
        want = [0.0, 0.0]
        for b in range(2):
            idx = [i for i in range(256) if bool(masks[0, b, i])]
            want[b] = sum(float(sq[0, i]) for i in idx) / len(idx)
        assert got[0].item() == pytest.approx(want[0], abs=1e-12)
        assert got[1].item() == pytest.approx(want[1], rel=1e-9)
        assert want[0] < 1e-6 < 0.01 < want[1]  # the vanished ball shows

    def test_an_empty_region_is_nan(self):
        sq = torch.ones(1, 4)
        masks = torch.zeros(1, 2, 4, dtype=torch.bool)
        assert region_sq(sq, masks).isnan().all()

    def test_the_ball_errors_of_a_baseline_are_its_errors_on_the_discs(self):
        balls, clips = _flight(gamma=1.0)
        std = _std(balls, clips)
        row = baselines(clips, balls, std, M)
        states = clips.states
        h = 10
        last = std.apply(render_flat(states[:, OBSERVED - 1], RADII, 32))
        target = std.apply(render_flat(states[:, OBSERVED + h - 1], RADII, 32))
        masks = disc_masks(states[:, OBSERVED + h - 1], RADII, 32)  # (1, 2, 1024)
        sq = ((last - target) ** 2)[0]
        for b in range(2):
            want = float(sq[masks[0, b]].double().mean())
            assert row[f"copy{h}@ball{b}"] == pytest.approx(want, rel=1e-5)


class TestMeanImage:
    def test_the_mean_image_is_standardized_and_the_same_for_every_window(self):
        balls, clips = _flight(gamma=0.0)
        std = _std(balls, clips)
        image = torch.linspace(0.0, 0.2, balls.size**2, dtype=torch.float64)
        row = baselines(clips, balls, std, M, mean_image=image)
        states = clips.states
        target = std.apply(render_flat(states[:, OBSERVED:], balls.radii, balls.size))
        want = (target - std.apply(image.float())) ** 2  # (1, HORIZON, size^2)
        for h in M.horizons:
            assert row[f"mean{h}"] == pytest.approx(
                float(want[:, h - 1].mean()), rel=1e-5
            )

    def test_without_one_the_mean_is_the_uniform_image_of_the_one_mean(self):
        balls, clips = _flight(gamma=0.0)
        std = _std(balls, clips)
        uniform = torch.full((balls.size**2,), float(std.mean[0]), dtype=torch.float64)
        a = baselines(clips, balls, std, M)
        b = baselines(clips, balls, std, M, mean_image=uniform)
        assert a["mean10"] == pytest.approx(b["mean10"], rel=1e-5)

    def test_a_ball_with_no_pixel_does_not_make_the_mean_of_balls_nan(self):
        """At size 2 no pixel center is within 0.06 of a ball: its region is
        empty (NaN) and only it drops out."""
        balls, clips = _flight(gamma=0.0)
        balls.size = 2
        std = _std(balls, clips)
        clips.states[..., 0:2] = 0.5  # 0.35 from every pixel center
        clips.states[..., 4:6] = torch.tensor([0.26, 0.74], dtype=torch.float64)
        masks = disc_masks(clips.states[:, OBSERVED], RADII, 2)
        assert not masks[0, 0].any() and masks[0, 1].any()
        row = baselines(clips, balls, std, M)
        assert math.isnan(row["copy10@ball0"])
        assert math.isfinite(row["copy10@ball1"])
