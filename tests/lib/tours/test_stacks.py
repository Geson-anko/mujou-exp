from pathlib import Path

import pytest
import torch

from exp.lib.rundir import EnvError
from exp.lib.tours.stacks import DATA_ENV, StackedTourFrames, tours_root
from tests.lib.tours.test_walking_tours import numbered

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


class TestStackedTourFrames:
    def test_one_frame_a_step_walks_the_loop(self, root):
        """Stream i shows frame offset_i + k of the loop; reset at the first
        stack and at the first frame of each video."""
        n = 4
        s = StackedTourFrames(root, n=n, held_out="H", frames=1, stride=1, advance=1)
        offsets = [i * N // n for i in range(n)]
        for k in range(30):
            stack, ready, reset = s.next()
            assert stack.dtype == torch.uint8 and stack.shape == (n, 1, 3, 2, 2)
            gs = [(o + k) % N for o in offsets]
            assert decode(stack[:, 0].float() / 255) == [loop_frame(g) for g in gs]
            assert ready.all()
            assert reset.tolist() == [k == 0 or g in FIRSTS for g in gs]

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

    def test_locate_names_the_frame_a_stack_ends_at(self, root):
        n = 3
        s = StackedTourFrames(root, n=n, held_out="H", frames=2, stride=2, advance=3)
        for k in range(12):
            stack, _, _ = s.next()
            for i in range(n):
                video, frame = s.locate(k, i)
                assert decode(stack[i : i + 1, -1].float() / 255) == [
                    ("ABC".index(video), frame)
                ]

    def test_non_positive_settings_raise(self, root):
        with pytest.raises(ValueError, match="positive"):
            StackedTourFrames(root, n=1, held_out="H", frames=2, stride=0, advance=1)

    def test_unknown_layout_raises(self, root):
        with pytest.raises(ValueError, match="layout"):
            StackedTourFrames(
                root, n=2, held_out="H", frames=1, stride=1, advance=1, layout="x"
            )


class TestChunkLayout:
    """``layout="chunk"``: the k-th ``next()`` of n streams is the single
    stream's next() calls k n .. k n + n - 1, in order (the oracle is the
    stream of n = 1)."""

    @pytest.mark.parametrize(
        ("n", "frames", "stride", "advance"), [(4, 1, 1, 1), (3, 3, 2, 2)]
    )
    def test_the_chunk_is_the_single_streams_next_n_stacks(
        self, root, n, frames, stride, advance
    ):
        kw = {"held_out": "H", "frames": frames, "stride": stride, "advance": advance}
        chunk = StackedTourFrames(root, n=n, layout="chunk", **kw)
        single = StackedTourFrames(root, n=1, **kw)
        for k in range(15):  # past the loop of 21 frames at least once
            stacks, ready, reset = chunk.next()
            for i in range(n):
                one, one_ready, _ = single.next()
                assert torch.equal(stacks[i], one[0])
                assert bool(ready[i]) == bool(one_ready[0])
                # its previous stack is n advance frames back: reset when a
                # video starts within them
                g = ((k * n + i) * advance) % N
                _, local = loop_frame(g)
                assert bool(reset[i]) == (k == 0 or local < n * advance)

    def test_state_dict_continues_exactly(self, root):
        kw = {"n": 3, "held_out": "H", "frames": 2, "stride": 1, "advance": 1}
        a = StackedTourFrames(root, layout="chunk", **kw)
        for _ in range(4):
            a.next()
        b = StackedTourFrames(root, layout="chunk", **kw)
        b.load_state_dict(a.state_dict())
        for x, y in zip(a.next(), b.next(), strict=True):
            assert torch.equal(x, y)


class TestToursRoot:
    def test_env_var_names_the_parent_of_the_version(self, tmp_path, monkeypatch):
        (tmp_path / "v1").mkdir()
        (tmp_path / "v1" / "index.json").write_text("{}")
        monkeypatch.setenv(DATA_ENV, str(tmp_path))
        assert tours_root("v1") == tmp_path / "v1"

    def test_missing_data_is_an_env_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv(DATA_ENV, str(tmp_path))
        with pytest.raises(EnvError, match="no prepared Walking Tours"):
            tours_root("nope")
