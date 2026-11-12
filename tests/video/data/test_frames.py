from pathlib import Path

import pytest
import torch

from exp.video.data.frames import (
    BallFrames,
    IIDStackedTourFrames,
    StackedTourFrames,
    TourFrames,
)
from exp.video.data.physics import BallStreams, Physics, render
from exp.video.schema import PhysicsConf
from tests.video.data.test_walking_tours import numbered

# A, B, C of 7, 5, 9 frames; H held out. The loop: A 0..6, B 7..11, C 12..20.
LENGTHS = {"A": 7, "B": 5, "C": 9, "H": 4}
FIRSTS = {0: "A", 7: "B", 12: "C"}
N = 21


def decode(images: torch.Tensor) -> list[tuple[int, int]]:
    """(video, frame) of each stream's image of ``numbered``."""
    px = (images[:, :, 0, 0] * 255).round().long()  # (n, 3)
    return [(int(v), int(a) + 256 * int(b)) for v, a, b in px.tolist()]


def loop_frame(g: int) -> tuple[int, int]:
    """(video, frame) of frame g of the loop A, B, C (video numbers by name,
    so H, the 4th, is never shown)."""
    first = max(f for f in FIRSTS if f <= g)
    return "ABC".index(FIRSTS[first]), g - first


@pytest.fixture
def root(tmp_path) -> Path:
    return numbered(tmp_path / "d", LENGTHS)


class TestTourFrames:
    def test_stream_i_shows_frame_offset_plus_k_of_the_loop(self, root):
        n = 4
        frames = TourFrames(root, n=n, held_out="H")
        offsets = [i * N // n for i in range(n)]  # 0, 5, 10, 15
        for k in range(30):  # past the end of the loop
            images, _, _ = frames.next()
            expect = [loop_frame((o + k) % N) for o in offsets]
            assert decode(images) == expect

    def test_reset_at_the_first_frame_and_at_each_video(self, root):
        frames = TourFrames(root, n=4, held_out="H")
        offsets = [0, 5, 10, 15]
        for k in range(30):
            _, _, reset = frames.next()
            expect = [k == 0 or (o + k) % N in FIRSTS for o in offsets]
            assert reset.tolist() == expect

    def test_shapes_and_values(self, root):
        images, states, reset = TourFrames(root, n=3, held_out="H").next()
        assert images.shape == (3, 3, 2, 2) and images.dtype == torch.float32
        assert states.shape == (3, 0)
        assert reset.dtype == torch.bool and reset.shape == (3,)
        assert 0.0 <= images.min() and images.max() <= 1.0

    def test_the_held_out_video_is_never_shown(self, root):
        frames = TourFrames(root, n=5, held_out="A")
        for _ in range(N):
            assert all(v != 0 for v, _ in decode(frames.next()[0]))

    def test_unknown_held_out_raises(self, root):
        with pytest.raises(ValueError, match="not in"):
            TourFrames(root, n=2, held_out="Z")

    def test_no_frames_besides_the_held_out_raises(self, tmp_path):
        only = numbered(tmp_path / "d", {"A": 0, "H": 5})
        with pytest.raises(ValueError, match="no frames"):
            TourFrames(only, n=2, held_out="H")

    def test_state_dict_continues_exactly(self, root):
        a = TourFrames(root, n=3, held_out="H")
        for _ in range(8):
            a.next()
        b = TourFrames(root, n=3, held_out="H")
        b.load_state_dict(a.state_dict())
        for _ in range(10):
            for x, y in zip(a.next(), b.next()):
                assert torch.equal(x, y)


class TestBallFrames:
    KW = {"radii": [0.06, 0.1], "frame": 1 / 30, "substeps": 8, "seed": 3}

    def physics(self) -> Physics:
        return Physics.from_conf(PhysicsConf())

    def test_images_render_the_states_of_the_ball_streams(self):
        frames = BallFrames(5, physics=self.physics(), size=16, **self.KW)
        streams = BallStreams(5, physics=self.physics(), **self.KW)
        for k in range(4):
            images, states, reset = frames.next()
            expect = streams.next()
            assert torch.equal(states, expect)
            assert torch.equal(images, render(expect, 16, self.KW["radii"]))
            assert images.shape == (5, 1, 16, 16)
            assert reset.tolist() == [k == 0] * 5

    def test_state_dict_continues_exactly(self):
        a = BallFrames(3, physics=self.physics(), size=8, **self.KW)
        for _ in range(5):
            a.next()
        b = BallFrames(3, physics=self.physics(), size=8, **self.KW)
        b.load_state_dict(a.state_dict())
        for _ in range(5):
            for x, y in zip(a.next(), b.next()):
                assert torch.equal(x, y)


class TestStackedTourFrames:
    def test_one_frame_a_step_is_tour_frames(self, root):
        a = TourFrames(root, n=4, held_out="H")
        b = StackedTourFrames(root, n=4, held_out="H", frames=1, stride=1, advance=1)
        for _ in range(30):
            images, _, reset = a.next()
            stack, ready, reset_b = b.next()
            assert stack.dtype == torch.uint8 and stack.shape == (4, 1, 3, 2, 2)
            assert torch.equal(stack[:, 0].float() / 255, images)
            assert torch.equal(reset_b, reset) and ready.all()

    def test_the_stack_is_g_minus_j_stride_and_ready_within_a_video(self, root):
        """Stream i ends at g = offset_i + 3k of the loop; its stack is frames
        g - 2j (j = 2, 1, 0) of g's video, ready when g - 4 is in it."""
        n, frames, stride, advance = 3, 3, 2, 3
        s = StackedTourFrames(
            root, n=n, held_out="H", frames=frames, stride=stride, advance=advance
        )
        offsets = [i * N // n for i in range(n)]  # 0, 7, 14
        for k in range(12):
            stack, ready, reset = s.next()
            for i in range(n):
                g = (offsets[i] + k * advance) % N
                video, local = loop_frame(g)
                want = [(video, max(local - stride * j, 0)) for j in (2, 1, 0)]
                assert decode(stack[i].float() / 255) == want
                assert bool(ready[i]) == (local - stride * (frames - 1) >= 0)
                assert bool(reset[i]) == (k == 0 or local < advance)

    def test_state_dict_continues_exactly(self, root):
        kw = {"n": 2, "held_out": "H", "frames": 2, "stride": 2, "advance": 3}
        a = StackedTourFrames(root, **kw)
        for _ in range(5):
            a.next()
        b = StackedTourFrames(root, **kw)
        b.load_state_dict(a.state_dict())
        for x, y in zip(a.next(), b.next(), strict=True):
            assert torch.equal(x, y)

    def test_non_positive_settings_raise(self, root):
        with pytest.raises(ValueError, match="positive"):
            StackedTourFrames(root, n=1, held_out="H", frames=2, stride=0, advance=1)


class TestIIDStackedTourFrames:
    KW: dict = {"n": 64, "held_out": "H", "frames": 3, "stride": 2}

    def test_every_stack_is_whole_within_a_video_and_reset(self, root):
        """A stack of 3 frames 2 apart ends at local frame >= 4 of its video
        and holds frames end - 4, end - 2, end of it."""
        s = IIDStackedTourFrames(root, seed=0, **self.KW)
        for _ in range(5):
            stack, ready, reset = s.next()
            assert ready.all() and reset.all()
            for i in range(64):
                (v0, a), (v1, b), (v2, c) = decode(stack[i].float() / 255)
                assert v0 == v1 == v2 and a + 2 == b and b + 2 == c and a >= 0

    def test_the_ends_are_uniform_over_the_frames_that_end_a_stack(self, root):
        """A: 7 frames, 3 ends (4, 5, 6); B: 5, 1 end; C: 9, 5 ends; 9 in all,
        each drawn about 1 / 9 of the time; the held-out video never."""
        s = IIDStackedTourFrames(root, seed=1, **self.KW)
        counts: dict[tuple[int, int], int] = {}
        draws = 0
        for _ in range(60):
            stack, _, _ = s.next()
            for i in range(64):
                end = decode(stack[i].float() / 255)[-1]
                counts[end] = counts.get(end, 0) + 1
                draws += 1
        want = {(0, 4), (0, 5), (0, 6), (1, 4), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8)}
        assert set(counts) == want
        for c in counts.values():  # 3840 draws: 427 each, sd about 19
            assert abs(c - draws / 9) < 6 * (draws / 9) ** 0.5

    def test_the_seed_and_k_fix_the_draws(self, root):
        a = IIDStackedTourFrames(root, seed=2, **self.KW)
        a.next()
        b = IIDStackedTourFrames(root, seed=2, **self.KW)
        b.load_state_dict(a.state_dict())
        assert torch.equal(a.next()[0], b.next()[0])
        c = IIDStackedTourFrames(root, seed=3, **self.KW)
        c.load_state_dict({"k": 1})
        assert not torch.equal(b.next()[0], c.next()[0])
