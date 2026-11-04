import pytest
import torch

from exp.lib.ring import WindowRing

B, W, D, S = 3, 5, 2, 2
FRAME_NS = 33_333_333


def _frames(n: int) -> torch.Tensor:
    """Frame k of stream b is the vector (k, b): (n, B, D)."""
    k = torch.arange(n, dtype=torch.float64)[:, None].expand(n, B)
    b = torch.arange(B, dtype=torch.float64)[None, :].expand(n, B)
    return torch.stack([k, b], -1)


def _ring(stride: int = S) -> WindowRing:
    return WindowRing(B, W, D, stride=stride, dtype=torch.float64)


def _push_all(ring: WindowRing, x: torch.Tensor, resets=None, times=None) -> None:
    for k in range(len(x)):
        reset = torch.zeros(B, dtype=torch.bool)
        if resets is not None:
            reset = resets[k]
        t = k * FRAME_NS if times is None else times[k]
        ring.push_(x[k], t, reset)


class TestWindowRing:
    @pytest.mark.parametrize("n", [W, W + 1, 2 * W + 3])
    def test_each_stream_holds_its_last_w_frames_oldest_first(self, n):
        ring = _ring()
        x = _frames(n)
        _push_all(ring, x)
        held, mask = ring.window()
        torch.testing.assert_close(held, x[n - W :].transpose(0, 1), rtol=0, atol=0)
        assert mask.all()

    def test_the_window_is_a_copy(self):
        ring = _ring()
        _push_all(ring, _frames(W))
        held, _ = ring.window()
        ring.push_(
            torch.full((B, D), -1.0, dtype=torch.float64),
            W * FRAME_NS,
            torch.zeros(B, dtype=torch.bool),
        )
        assert (held >= 0).all()

    def test_a_window_with_a_reset_after_its_first_frame_is_masked(self):
        """A reset starts a video at its frame: the window is of one video
        iff the reset is at its oldest frame or before."""
        n = 12
        resets = torch.zeros(n, B, dtype=torch.bool)
        resets[0] = True
        resets[6, 1] = True  # stream 1: a new video from frame 6
        ring = _ring()
        for k in range(n):
            ring.push_(_frames(n)[k], k * FRAME_NS, resets[k])
            if k + 1 >= W:
                _, mask = ring.window()
                first = k + 1 - W  # the oldest frame of the window
                one_video = first >= 6 or k < 6
                assert mask.tolist() == [True, one_video, True], k

    def test_too_few_pushes_since_the_start_are_masked_as_a_reset(self):
        ring = _ring()
        _push_all(ring, _frames(W))
        resets = torch.tensor([False, True, False])
        ring.push_(_frames(1)[0], W * FRAME_NS, resets)
        assert ring.window()[1].tolist() == [True, False, True]

    @pytest.mark.parametrize(
        ("gap", "masked"), [(0, True), (2, True), (1.4, False), (0.6, False)]
    )
    def test_a_time_gap_or_a_repeat_masks_every_stream(self, gap, masked):
        """The gap of the 3rd frame is ``gap`` frames: a repeat (0) or a missed
        frame (2) is out of (0.5, 1.5) of the median gap, a jitter is not."""
        times = [k * FRAME_NS for k in range(W)]
        shift = round((gap - 1) * FRAME_NS)
        times = times[:2] + [t + shift for t in times[2:]]
        ring = _ring()
        _push_all(ring, _frames(W), times=times)
        assert ring.window()[1].tolist() == [not masked] * B
        # once the gap has left the window, it is learned from again
        for k in range(W, 2 * W):
            ring.push_(
                _frames(1)[0],
                times[-1] + (k - W + 1) * FRAME_NS,
                torch.zeros(B, dtype=torch.bool),
            )
        assert ring.window()[1].all()

    @pytest.mark.parametrize("stride", [1, 2, 3, W, W + 2])
    @pytest.mark.parametrize("n", [W - 1, W, W + 1, 4 * W + 1])
    def test_the_windows_due_over_n_pushes_are_floor_n_minus_w_over_s_plus_1(
        self, n, stride
    ):
        ring = _ring(stride)
        due = []
        for k in range(n):
            ring.push_(_frames(1)[0], k * FRAME_NS, torch.zeros(B, dtype=torch.bool))
            if ring.due():
                due.append(k + 1)
        count = (n - W) // stride + 1 if n >= W else 0
        assert len(due) == count
        assert due == [W + i * stride for i in range(count)]

    @pytest.mark.parametrize("stride", [1, S, W - 1, W, W + 2])
    def test_elapsed_is_s_frames(self, stride):
        ring = _ring(stride)
        for k in range(3 * W + 2 * stride):
            ring.push_(_frames(1)[0], k * FRAME_NS, torch.zeros(B, dtype=torch.bool))
            if ring.due():
                assert ring.elapsed() == stride * FRAME_NS / 1e9

    def test_elapsed_is_the_time_since_the_window_before(self):
        """Frames at irregular times: the elapsed of a window is the time of
        its last frame minus that of the window before."""
        g = torch.Generator().manual_seed(0)
        times = torch.cumsum(
            torch.randint(30_000_000, 36_000_000, (20,), generator=g), 0
        ).tolist()
        ring = _ring()
        ends = []
        for k, t in enumerate(times):
            ring.push_(_frames(1)[0], t, torch.zeros(B, dtype=torch.bool))
            if ring.due():
                if ends:
                    assert ring.elapsed() == (t - ends[-1]) / 1e9
                else:  # the first: the push S before, from the ring
                    assert ring.elapsed() == (t - times[k - S]) / 1e9
                ends.append(t)

    def test_the_state_round_trip_restores_the_ring_exactly(self):
        a = _ring()
        resets = torch.zeros(9, B, dtype=torch.bool)
        resets[3, 2] = True
        _push_all(a, _frames(9), resets)
        b = _ring()
        b.load_state_dict(
            {
                k: (v.clone() if isinstance(v, torch.Tensor) else v)
                for k, v in a.state_dict().items()
            }
        )
        for ring in (a, b):
            ring.push_(_frames(10)[9], 9 * FRAME_NS, torch.zeros(B, dtype=torch.bool))
        xa, ma = a.window()
        xb, mb = b.window()
        assert torch.equal(xa, xb) and torch.equal(ma, mb)
        assert a.due() == b.due() and a.elapsed() == b.elapsed() and a.k == b.k == 10

    def test_a_window_before_w_pushes_is_an_error(self):
        ring = _ring()
        _push_all(ring, _frames(W - 1))
        assert not ring.due()
        with pytest.raises(ValueError, match="after 4 pushes"):
            ring.window()
        with pytest.raises(ValueError, match="window >= 2"):
            WindowRing(B, 1, D, stride=1)
