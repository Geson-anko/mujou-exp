"""The probes of a representation: exact on a linear map of the targets, about
0 on noise, the columns of each ball where they belong, the pixel baseline kept
from overfitting, and the metrics of an encoder on a set of chunks."""

import pytest
import torch

from exp.lib.metrics import effective_rank
from exp.video.compose import compose
from exp.video.jepa.vit import build_vision
from exp.video.measure.vision import (
    ProbeSet,
    ball_probe,
    baselines,
    thumbnail_probe,
    vision_metrics,
)

N = 1200


def _gen(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _states(n: int, balls: int, gen: torch.Generator) -> torch.Tensor:
    """Random states ``(n, 4 balls)``: positions in the box, velocities about 1."""
    s = torch.rand(n, balls, 4, generator=gen)
    s[..., 2:] = torch.randn(n, balls, 2, generator=gen)
    return s.flatten(1)


def _half(n: int) -> torch.Tensor:
    return torch.arange(n) < n // 2


class TestBallProbe:
    @pytest.mark.parametrize("velocity", [False, True])
    def test_a_linear_map_of_the_states_is_probed_exactly(self, velocity):
        """$z = s A + c$ with $A$ of full row rank: the probe recovers each
        column, $R^2 = 1$."""
        g = _gen()
        s = _states(N, 3, g)
        z = s @ torch.randn(12, 20, generator=g) + 3.0
        out = ball_probe(z, s, 3, _half(N), velocity=velocity)
        names = ["x", "y", "vx", "vy"] if velocity else ["x", "y"]
        want = {f"probe_{a}{k}" for k in range(3) for a in names}
        want |= {"probe_pos_min"} | ({"probe_vel_min"} if velocity else set())
        assert set(out) == want
        for key, value in out.items():
            assert value == pytest.approx(1.0, abs=1e-6), key

    def test_noise_is_probed_at_about_zero(self):
        """$z$ independent of the states: the best linear prediction is the mean,
        so $R^2 \\approx 0$; a probe chosen by cross-validation does not
        overfit the 32 features of noise to much below it."""
        g = _gen(1)
        s = _states(N, 2, g)
        z = torch.randn(N, 32, generator=g)
        out = ball_probe(z, s, 2, _half(N), velocity=True)
        for key, value in out.items():
            assert -0.05 < value < 0.02, (key, value)

    def test_each_ball_is_probed_from_its_own_columns(self):
        """$z$ holds the x of ball 1 and the y of ball 0 only: those two are
        exact and the rest is about 0, so the names follow the columns."""
        g = _gen(2)
        s = _states(N, 2, g)
        z = torch.stack([s[:, 4 * 1 + 0], s[:, 4 * 0 + 1]], 1)
        z = torch.cat([z, torch.randn(N, 4, generator=g)], 1)
        out = ball_probe(z, s, 2, _half(N))
        assert out["probe_x1"] == pytest.approx(1.0, abs=1e-6)
        assert out["probe_y0"] == pytest.approx(1.0, abs=1e-6)
        assert abs(out["probe_x0"]) < 0.05 and abs(out["probe_y1"]) < 0.05
        assert out["probe_pos_min"] == min(out["probe_x0"], out["probe_y1"])

    def test_it_is_fit_on_half_and_scored_on_the_rest(self):
        """$z = (x, y)$ on the fit half and $(y, x)$ on the rest: the probe
        learns $x$ from the first column and predicts $y$ on the rest, so for
        independent uniform $x, y$, $R^2 = 1 - E[(y - x)^2] / V[x] = -1$."""
        g = _gen(3)
        s = _states(4 * N, 1, g)
        half = _half(4 * N)
        z = s[:, :2].clone()
        z[~half] = z[~half].flip(1)
        out = ball_probe(z, s, 1, half)
        assert out["probe_x0"] == pytest.approx(-1.0, abs=0.1)
        assert out["probe_y0"] == pytest.approx(-1.0, abs=0.1)

    def test_more_features_than_samples_do_not_overfit(self):
        """2000 features of noise and 600 samples to fit on (as the raw pixels
        of the chunks): least squares would interpolate the targets and score
        far below 0; the cross-validated penalty keeps $R^2 \approx 0$."""
        g = _gen(4)
        x = torch.randn(N, 2000, generator=g)
        out = ball_probe(x, _states(N, 1, g), 1, _half(N))
        for key, value in out.items():
            assert -0.05 < value < 0.02, (key, value)

    def test_features_rarely_seen_on_the_fit_half_do_not_hurt(self):
        """20 noisy copies of $(x, y)$ (so that the cross-validation picks a
        moderate penalty), and 100 small features each nonzero at 2 fit
        samples and at 10% of the scored ones (pixels the balls of calib
        almost never lit): they carry nothing about the targets, so the probe
        with them scores as the probe without them. Standardizing each
        feature would blow them up to about $\\sqrt{n / 2}$ on the scored
        samples and score them far below 0."""
        g = _gen(12)
        s = _states(N, 1, g)
        dense = s[:, :2].repeat(1, 10) + torch.randn(N, 20, generator=g)
        rare = torch.zeros(N, 100)
        for j in range(100):
            rare[torch.randint(N // 2, (2,), generator=g), j] = 0.05
        rare[N // 2 :][torch.rand(N // 2, 100, generator=g) < 0.1] = 0.05
        alone = ball_probe(dense, s, 1, _half(N))
        both = ball_probe(torch.cat([dense, rare], 1), s, 1, _half(N))
        for key, value in both.items():
            assert alone[key] > 0.3, (key, alone[key])
            assert value == pytest.approx(alone[key], abs=0.05), key

    def test_a_rotation_and_a_scale_of_z_change_nothing(self):
        """The features are centered and divided by one common scale, and the
        ridge penalty is of the norm of the weights: for $Q$ orthogonal and
        $c > 0$, the probe of $c\\,z Q$ is that of $z$."""
        g = _gen(13)
        s = _states(N, 2, g)
        z = s @ torch.randn(8, 6, generator=g) + torch.randn(N, 6, generator=g)
        q, _ = torch.linalg.qr(torch.randn(6, 6, generator=g, dtype=torch.float64))
        turned = 7.0 * (z.double() @ q)
        got = ball_probe(turned, s, 2, _half(N), velocity=True)
        want = ball_probe(z, s, 2, _half(N), velocity=True)
        for key, value in want.items():
            assert got[key] == pytest.approx(value, abs=1e-5), key

    def test_a_wrong_number_of_balls_is_refused(self):
        with pytest.raises(ValueError, match="columns"):
            ball_probe(torch.zeros(10, 2), torch.zeros(10, 8), 3, _half(10))


class TestThumbnailProbe:
    def _images(self, n: int, gen: torch.Generator) -> tuple[torch.Tensor, ...]:
        """Images ``(n, 3, 32, 32)`` whose grey 8x8 block means are ``pattern``
        ``(n, 64)``: each block constant, the channels around it."""
        pattern = torch.rand(n, 8, 8, generator=gen)
        tilt = torch.tensor([-0.1, 0.0, 0.1]).view(1, 3, 1, 1)
        images = pattern[:, None].repeat_interleave(4, -1).repeat_interleave(4, -2)
        return images + tilt, pattern.flatten(1)

    def test_a_linear_map_of_the_thumbnail_is_probed_exactly(self):
        g = _gen(5)
        images, pattern = self._images(N, g)
        z = pattern @ torch.randn(64, 80, generator=g)
        assert thumbnail_probe(z, images, _half(N)) == pytest.approx(1.0, abs=1e-6)

    def test_noise_is_probed_at_about_zero(self):
        g = _gen(6)
        images, _ = self._images(N, g)
        z = torch.randn(N, 16, generator=g)
        assert -0.05 < thumbnail_probe(z, images, _half(N)) < 0.02

    def test_the_r2_is_pooled_over_the_pixels(self):
        """$z$ knows the left half of the thumbnail exactly and nothing of the
        right half, whose variance is 3 times as large (std x sqrt 3): the
        pooled $R^2$ is $1 - 3 / 4 = 0.25$, not the mean 0.5 of the pixels'."""
        g = _gen(7)
        images, pattern = self._images(4 * N, g)
        grid = pattern.view(-1, 8, 8)
        grid[:, :, 4:] = 0.5 + (grid[:, :, 4:] - 0.5) * 3**0.5
        images = grid[:, None].repeat_interleave(4, -1).repeat_interleave(4, -2)
        z = grid[:, :, :4].flatten(1)
        got = thumbnail_probe(z, images, _half(4 * N))
        assert got == pytest.approx(0.25, abs=0.03)


class TestProbeSet:
    def test_halved_fits_on_the_first_half(self):
        p = ProbeSet.halved(torch.zeros(5, 1, 1, 4, 4))
        assert p.half.tolist() == [True, True, False, False, False]
        assert p.balls == 0

    def test_shapes_are_checked(self):
        with pytest.raises(ValueError, match="chunks"):
            ProbeSet(torch.zeros(4, 1, 4, 4), _half(4))
        with pytest.raises(ValueError, match="states"):
            ProbeSet(torch.zeros(4, 1, 1, 4, 4), _half(4), torch.zeros(4, 6))


class TestVisionMetrics:
    def _clips(self, n: int = 200, frames: int = 2) -> ProbeSet:
        g = _gen(8)
        chunks = torch.rand(n, frames, 1, 8, 8, generator=g)
        return ProbeSet.halved(chunks, _states(n, 2, g))

    def test_the_metrics_of_an_encoder(self):
        """``rank`` and ``z_std_min`` are those of all the $z$; the probes are
        ``ball_probe`` (with the velocity, $N = 2$) and ``thumbnail_probe`` on
        the same $z$; encoding in batches changes only the rounding."""
        clips = self._clips()
        proj = torch.randn(128, 16, generator=_gen(9))

        def encode(c: torch.Tensor) -> torch.Tensor:
            return c.flatten(1) @ proj

        row = vision_metrics(encode, clips, batch=7)
        z = encode(clips.chunks)
        assert row["rank"] == pytest.approx(effective_rank(z), rel=1e-6)
        assert row["z_std_min"] == pytest.approx(float(z.std(0).min()), rel=1e-6)
        assert clips.states is not None
        probes = ball_probe(z, clips.states, 2, clips.half, velocity=True)
        for key, value in probes.items():
            assert row[key] == pytest.approx(value, abs=1e-5), key
        thumb = thumbnail_probe(z, clips.chunks[:, -1], clips.half)
        assert row["thumb"] == pytest.approx(thumb, abs=1e-5)

    def test_a_collapse_shows_as_a_zero_std(self):
        clips = self._clips(frames=1)
        row = vision_metrics(lambda c: torch.ones(len(c), 4), clips)
        assert row["z_std_min"] == 0.0 and "probe_vel_min" not in row

    def test_without_states_there_is_no_ball_probe(self):
        clips = ProbeSet.halved(torch.rand(40, 1, 3, 8, 8, generator=_gen(10)))
        row = vision_metrics(lambda c: c.flatten(1)[:, :6], clips)
        assert set(row) == {"rank", "z_std_min", "thumb"}


class TestBaselines:
    def test_the_untrained_vit_and_the_pixels(self):
        """``init_*`` are the metrics of ``build_vision`` at the run's seed
        (as the run starts); ``pixels_*`` the ball probes from the raw pixels
        of the chunks."""
        conf = compose(
            [
                "+experiment=m15/balls_n1",
                "data.balls.size=16",
                "vision.dim=16",
                "vision.depth=2",
                "vision.heads=2",
                "vision.pred_depth=1",
                "seed=3",
            ]
        )
        g = _gen(11)
        clips = ProbeSet.halved(torch.rand(60, 1, 1, 16, 16, generator=g))
        clips = ProbeSet(clips.chunks, clips.half, _states(60, 3, g))
        out = baselines(conf, clips)
        init = vision_metrics(build_vision(conf, seed=3).encode, clips)
        for key, value in init.items():
            assert out[f"init_{key}"] == pytest.approx(value, abs=1e-9), key
        assert clips.states is not None
        pixels = ball_probe(clips.chunks.flatten(1), clips.states, 3, clips.half)
        for key, value in pixels.items():
            assert out[f"pixels_{key}"] == pytest.approx(value, abs=1e-9), key
        assert not any(k.startswith("pixels_") and "thumb" in k for k in out)
