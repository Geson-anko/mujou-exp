import math
import time
from dataclasses import replace

import pytest
import torch

from exp.video.balls import (
    Physics,
    Streams,
    Video,
    advance,
    eval_clips,
    ordered,
    render,
)

CONSERVED = Physics(
    gamma0=1.0, gamma_amp=0.0, e0=1.0, e_amp=0.0, drag=0.0, conserve=True
)
STILL = Physics(gamma0=0.0, gamma_amp=0.0, e0=1.0, e_amp=0.0, drag=0.0)
FRAME = 1 / 30


def _states(video: Video, n: int) -> torch.Tensor:
    return torch.stack([video.next().state for _ in range(n)])


def _energy(states: torch.Tensor, gamma: float) -> torch.Tensor:
    s = states.reshape(len(states), -1, 4).double()
    return (0.5 * (s[..., 2] ** 2 + s[..., 3] ** 2) + gamma * s[..., 1]).sum(-1)


class TestPhysics:
    def test_kinetic_energy_is_kept_without_gravity_drag_and_loss(self):
        video = Video(physics=STILL, episode=(1e9, 1e9), seed=0)
        e = _energy(_states(video, 3000), gamma=0.0)  # 100 s, many bounces
        torch.testing.assert_close(e, e[:1].expand_as(e), rtol=1e-6, atol=0)

    def test_semi_implicit_euler_keeps_the_energy_bounded(self):
        physics = Physics(gamma0=1.0, gamma_amp=0.0, e0=1.0, e_amp=0.0, drag=0.0)
        video = Video(physics=physics, episode=(1e9, 1e9), seed=0)
        e = _energy(_states(video, 18_000), gamma=1.0)  # 10 minutes
        dev = (e - e[0]).abs()
        early = float(dev[:1800].max())  # the first minute
        assert float(dev.max()) < 2 * early + 1e-3  # does not keep growing
        assert float(dev.max()) < 0.05 * float(e[0])

    @pytest.mark.parametrize("wall", ["left", "right", "bottom", "top"])
    def test_a_ball_hitting_a_wall_keeps_e_of_its_speed(self, wall):
        physics = Physics(gamma0=0.0, gamma_amp=0.0, e0=0.5, e_amp=0.0, drag=0.0)
        axis, sign = {
            "left": (0, -1),
            "right": (0, 1),
            "bottom": (1, -1),
            "top": (1, 1),
        }[wall]
        pos, vel = [[0.5, 0.5]], [[0.0, 0.0]]
        vel[0][axis] = sign * 1.0
        advance(pos, vel, (0.1,), physics, 0.0, 1.0, 240)  # hits once, at 0.4 s
        assert vel[0][axis] == pytest.approx(-sign * 0.5)
        assert vel[0][1 - axis] == 0.0
        assert 0.1 <= pos[0][axis] <= 0.9

    @pytest.mark.slow
    def test_the_energy_stays_bounded_for_thirty_minutes(self):
        """M13: with e = 1, no drag and a constant gravity, one stream runs
        for 30 minutes without new balls; its energy does not drift."""
        video = Video(physics=CONSERVED, episode=None, seed=1)
        e = _energy(_states(video, 54_000), gamma=1.0)
        dev = (e - e[0]).abs()
        early = float(dev[:1800].max())  # the first minute
        assert float(dev.max()) < 2 * early + 1e-3
        assert float(dev.max()) < 0.05 * float(e[0])

    def test_conserve_keeps_the_modified_energy_of_each_ball_exactly(self):
        """Semi-implicit Euler keeps $\tilde E = v^2/2 + \\gamma y - \\gamma h v_y / 2$
        exactly between the bounces ($h$ the step); with ``conserve``, the
        bounces keep it too, so it holds to rounding over any time."""
        video = Video(physics=CONSERVED, episode=None, seed=1)
        s = _states(video, 18_000).reshape(18_000, -1, 4).double()  # 10 minutes
        h = FRAME / 8
        e = 0.5 * (s[..., 2] ** 2 + s[..., 3] ** 2) + s[..., 1] - 0.5 * h * s[..., 3]
        torch.testing.assert_close(e, e[:1].expand_as(e), rtol=0, atol=1e-5)

    def test_without_conserve_the_bounces_change_the_energy(self):
        """The reflection at the floor changes $\tilde E$ by
        $\\gamma (2 \\delta - h |v_y|)$ ($\\delta$ how far the ball went past the
        wall), so it is kept only with ``conserve``: the default is the
        physics of the runs before M13."""
        physics = replace(CONSERVED, conserve=False)
        video = Video(physics=physics, episode=None, seed=1)
        s = _states(video, 1800).reshape(1800, -1, 4).double()
        h = FRAME / 8
        e = 0.5 * (s[..., 2] ** 2 + s[..., 3] ** 2) + s[..., 1] - 0.5 * h * s[..., 3]
        assert float((e - e[0]).abs().max()) > 1e-4

    @pytest.mark.parametrize("wall", ["bottom", "top"])
    def test_conserve_keeps_e_of_the_speed_at_a_wall(self, wall):
        """Off the floor or the ceiling with gravity, the speed after the
        bounce is about e times that before (up to O(gamma h))."""
        physics = Physics(gamma0=1.0, gamma_amp=0.0, e0=0.5, e_amp=0.0, drag=0.0)
        physics = replace(physics, conserve=True)
        sign = -1 if wall == "bottom" else 1
        pos, vel = [[0.5, 0.5]], [[0.0, sign * 3.0]]
        before = vel[0][1]
        while (before > 0) == (vel[0][1] > 0):  # hits within 0.2 s
            before = vel[0][1]
            advance(pos, vel, (0.1,), physics, 0.0, 1 / 240, 1)
        assert vel[0][1] == pytest.approx(-0.5 * before, abs=0.02)

    def test_the_stream_keeps_moving(self):
        """With the default episodes, the balls do not come to rest: the
        stream is not mostly still frames."""
        video = Video(physics=Physics(period=600.0), seed=0)
        frames = [video.next() for _ in range(3600)]  # 2 minutes
        moved = [
            float((b.image - a.image).abs().max()) > 1e-2
            for a, b in zip(frames, frames[1:], strict=False)
            if not b.reset
        ]
        assert sum(moved) / len(moved) > 0.95

    def test_the_physics_follows_its_schedule_across_episodes(self):
        physics = Physics(period=10.0)
        video = Video(physics=physics, episode=(1.0, 2.0), seed=0)
        frames = [video.next() for _ in range(600)]
        assert sum(f.reset for f in frames) >= 10
        for k, f in enumerate(frames):
            t = k * FRAME
            want = torch.tensor([physics.gamma(t), physics.restitution(t)])
            torch.testing.assert_close(f.params, want)
        assert physics.gamma(2.5) == pytest.approx(1.5)  # gamma0 (1 + 0.5 sin(pi/2))
        assert physics.restitution(0.0) == pytest.approx(0.9)  # e0 (1 + 0.5)

    def test_the_same_seed_gives_the_same_frames(self):
        a, b = Video(seed=4), Video(seed=4)
        for _ in range(100):
            fa, fb = a.next(), b.next()
            assert torch.equal(fa.image, fb.image) and torch.equal(fa.state, fb.state)
            assert fa.reset == fb.reset


class TestNoRedraw:
    """M13: ``episode=None`` draws the balls once, so the frames change
    continuously."""

    PHYSICS = CONSERVED

    def test_only_the_first_frame_resets(self):
        video = Video(physics=self.PHYSICS, episode=None, seed=0)
        resets = [bool(video.next().reset) for _ in range(900)]  # 30 s
        assert resets[0] and not any(resets[1:])

    def test_each_frame_moves_the_balls_by_at_most_one_frame_of_motion(self):
        """The speed of a ball never exceeds sqrt(2 E) (E its energy per unit
        mass, kept at e = 1 without drag), so two frames are no farther apart
        than that speed times the frame, at the walls too (the bounce reflects
        the position)."""
        video = Video(physics=self.PHYSICS, episode=None, seed=0)
        s = _states(video, 900).reshape(900, -1, 4).double()
        radii = torch.tensor(video.radii, dtype=torch.float64)
        energy = 0.5 * (s[0, :, 2] ** 2 + s[0, :, 3] ** 2) + (s[0, :, 1] - radii)
        v_max = (2 * energy).sqrt()  # the speed with the ball on the floor
        step = (s[1:, :, :2] - s[:-1, :, :2]).norm(dim=-1)  # (899, balls)
        assert (step <= 1.01 * v_max * FRAME).all()
        assert (step > 0).all()  # and they keep moving

    def test_the_default_still_redraws(self):
        video = Video(seed=0)
        assert sum(bool(video.next().reset) for _ in range(900)) >= 5


class TestRadii:
    def test_the_radii_set_the_balls(self):
        video = Video(radii=(0.1, 0.1), seed=0)
        f = video.next()
        assert video.radii == (0.1, 0.1) and f.state.shape == (8,)
        clips = eval_clips(radii=(0.1,), length=4)
        assert clips.state.shape == (8, 4, 4, 4)

    def test_without_radii_the_first_balls_of_radii(self):
        a, b = Video(balls=2, seed=3), Video(radii=(0.06, 0.08), seed=3)
        for _ in range(30):
            assert torch.equal(a.next().image, b.next().image)


class TestRender:
    @pytest.mark.parametrize("xy", [(0.4, 0.6), (0.25, 0.3), (0.7, 0.75)])
    def test_the_centroid_is_the_ball(self, xy):
        state = torch.tensor([[*xy, 0.0, 0.0]], dtype=torch.float64)
        image = render(state, 32, (0.1,))[0, 0]
        idx = torch.arange(32, dtype=torch.float64)
        col = float((image.sum(0) * idx).sum() / image.sum())
        row = float((image.sum(1) * idx).sum() / image.sum())
        assert abs(col - (xy[0] * 32 - 0.5)) < 1
        assert abs(row - ((1 - xy[1]) * 32 - 0.5)) < 1

    def test_values_are_in_the_unit_interval(self):
        video = Video(seed=0)
        images = torch.stack([video.next().image for _ in range(100)])
        assert images.shape == (100, 1, 32, 32)
        assert images.min() >= 0 and images.max() <= 1


class TestEvalClips:
    def test_shapes_and_phases(self):
        physics = Physics()
        clips = eval_clips(physics=physics, length=16)
        assert clips.image.shape == (8, 4, 16, 1, 32, 32)
        assert clips.state.shape == (8, 4, 16, 12)
        t = 3 * physics.period / 8
        assert float(clips.params[3, 0, 0, 0]) == pytest.approx(physics.gamma(t))

    def test_fixed_by_the_seed(self):
        a, b = eval_clips(length=8), eval_clips(length=8)
        assert torch.equal(a.image, b.image)


@pytest.mark.slow
def test_thirty_minutes_take_less_than_a_minute():
    video = Video(seed=0)
    start = time.perf_counter()
    for _ in range(54_000):
        video.next()
    assert time.perf_counter() - start < 60
    assert math.isfinite(float(video.next().state.sum()))


def _frames(n: int) -> list:
    video = Video(episode=(1e9, 1e9), seed=0)
    return [video.next() for _ in range(n)]


class TestOrdered:
    def _ordered(self, order: str, frames: int = 90):
        return list(
            ordered(
                Video(episode=(1e9, 1e9), seed=0),
                frames=frames,
                order=order,
                clip=10,
                replay=30,
                seed=0,
            )
        )

    def test_stream_is_the_video(self):
        want = _frames(20)
        got = self._ordered("stream", 20)
        assert all(live for _, live in got)
        for (f, _), w in zip(got, want, strict=True):
            assert torch.equal(f.state, w.state) and f.reset == w.reset

    def test_shuffled_shows_each_clip_once_from_a_reset(self):
        want = _frames(90)
        got = self._ordered("shuffled")
        assert len(got) == 90
        clips = [got[i : i + 10] for i in range(0, 90, 10)]
        firsts = []
        for c in clips:
            assert c[0][0].reset and not any(f.reset for f, _ in c[1:])
            start = next(
                i for i, w in enumerate(want) if torch.equal(w.state, c[0][0].state)
            )
            for j, (f, _) in enumerate(c):
                assert torch.equal(f.state, want[start + j].state)
            firsts.append(start)
        assert sorted(firsts) == list(range(0, 90, 10))
        assert firsts != sorted(firsts)

    def test_replay_follows_each_live_clip_with_a_past_one(self):
        want = _frames(90)
        got = self._ordered("replay")
        live = [f for f, is_live in got if is_live]
        assert len(live) == 90 and len(got) == 180
        for f, w in zip(live, want, strict=True):
            assert torch.equal(f.state, w.state)
        for i in range(0, 180, 20):  # 10 live frames, then 10 replayed
            assert all(is_live for _, is_live in got[i : i + 10])
            assert not any(is_live for _, is_live in got[i + 10 : i + 20])
            replayed = got[i + 10][0]
            seen = want[: i // 2 + 10]
            assert replayed.reset
            assert any(torch.equal(replayed.state, w.state) for w in seen[::10])


class TestStreams:
    def test_stream_0_is_the_video_of_the_seed_and_the_others_differ(self):
        streams = Streams(3, seed=5)
        alone = Video(seed=5)
        stacks = [streams.next() for _ in range(4)]
        for f in stacks:
            want = alone.next()
            assert f.image.shape == (3, 1, 32, 32)
            assert torch.as_tensor(f.reset).shape == (3,)
            assert torch.equal(f.image[0], want.image)
            assert torch.equal(f.state[0], want.state)
        last = stacks[-1]
        assert not torch.equal(last.image[0], last.image[1])
        assert not torch.equal(last.image[1], last.image[2])

    def test_each_stream_starts_its_episodes_on_its_own(self):
        streams = Streams(2, seed=0, episode=(0.1, 0.5))
        resets = torch.stack([torch.as_tensor(streams.next().reset) for _ in range(60)])
        assert resets[0].all()  # every video starts with a reset
        assert resets[1:, 0].any() and resets[1:, 1].any()
        assert (resets[1:, 0] != resets[1:, 1]).any()

    def test_spread_streams_start_at_their_phase_of_the_physics(self):
        """Stream i of n starts at i T / n: its first parameters are those of
        the physics at that time, and it goes on from there."""
        physics = Physics(period=60.0)
        streams = Streams(4, seed=0, spread=True, physics=physics, frame=0.5)
        for k in range(3):
            f = streams.next()
            for i in range(4):
                t = 15.0 * i + 0.5 * k
                expected = torch.tensor([physics.gamma(t), physics.restitution(t)])
                torch.testing.assert_close(f.params[i], expected)

    def test_without_spread_every_stream_starts_at_zero(self):
        physics = Physics(period=60.0)
        f = Streams(3, seed=0, physics=physics).next()
        torch.testing.assert_close(f.params, f.params[:1].expand(3, 2))
