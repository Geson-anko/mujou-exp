from pathlib import Path

import pytest
import torch

from exp.video.data.frames import BallFrames, TourFrames
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
