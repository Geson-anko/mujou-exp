import torch

from exp.lib.png import read_png
from exp.video.data.observe import render_flat
from exp.video.measure.viz import (
    BOX,
    COLORS,
    frames_png,
    pick_windows,
    pixel_frames_png,
    to_pixels,
    trajectory_png,
)

SIZE = 32


def _states(xy: list[tuple[float, float]]) -> torch.Tensor:
    """One ball at each position, as states ``(n, 4)`` (no velocity)."""
    s = torch.zeros(len(xy), 4, dtype=torch.float64)
    s[:, :2] = torch.tensor(xy, dtype=torch.float64)
    return s


def _centroid(frame: torch.Tensor) -> tuple[float, float]:
    """The box position of the centroid of the brightness of a grey frame
    ``(SIZE, SIZE)``: pixel (row i, column j) is centered at x = (j + 0.5) /
    SIZE, y = 1 - (i + 0.5) / SIZE."""
    w = frame.double()
    rows = torch.arange(SIZE, dtype=torch.float64)
    total = w.sum()
    i = float((w.sum(1) * rows).sum() / total)
    j = float((w.sum(0) * rows).sum() / total)
    return (j + 0.5) / SIZE, 1 - (i + 0.5) / SIZE


class TestFramesPng:
    def test_the_balls_are_where_the_states_put_them(self, tmp_path):
        """Decoded by hand, each frame of the picture (row clip x
        {true, pred, cv}, column a frame) has its blob centered at the ball's
        position (balls well inside the box, so no wall cuts a blob)."""
        true = _states([(0.3, 0.4), (0.35, 0.45), (0.4, 0.5)]).view(1, 3, 4)
        pred = _states([(0.3, 0.4), (0.6, 0.3), (0.65, 0.35)]).view(1, 3, 4)
        cv = _states([(0.3, 0.4), (0.5, 0.7), (0.45, 0.75)]).view(1, 3, 4)
        path = tmp_path / "frames.png"
        frames_png(path, true, pred, cv, [0.1], size=SIZE, columns=(0, 2), scale=1)
        image = read_png(path)
        cell = SIZE + 2  # the frame and its border of 1 pixel
        assert image.shape == (3 * cell, 2 * cell, 3)
        for r, states in enumerate((true, pred, cv)):
            for c, k in enumerate((0, 2)):
                grey = image[
                    r * cell + 1 : (r + 1) * cell - 1,
                    c * cell + 1 : (c + 1) * cell - 1,
                    0,
                ]
                x, y = _centroid(grey)
                want = states[0, k, :2].tolist()
                assert abs(x - want[0]) < 0.01 and abs(y - want[1]) < 0.01


class TestPixelFramesPng:
    def test_the_predicted_image_is_drawn_as_it_is(self, tmp_path):
        """Row clip x {true, pred, cv}: the prediction row shows the given image
        (here a ball at a place no state gives), the others the rendered
        states."""
        true = _states([(0.3, 0.4), (0.35, 0.45)]).view(1, 2, 4)
        cv = _states([(0.3, 0.4), (0.4, 0.5)]).view(1, 2, 4)
        pred = render_flat(_states([(0.3, 0.4), (0.7, 0.2)]).view(1, 2, 4), [0.1], SIZE)
        path = tmp_path / "frames.png"
        pixel_frames_png(
            path, true, pred, cv, [0.1], size=SIZE, columns=(0, 1), scale=1
        )
        image = read_png(path)
        cell = SIZE + 2
        assert image.shape == (3 * cell, 2 * cell, 3)
        want = {(0, 1): (0.35, 0.45), (1, 0): (0.3, 0.4), (1, 1): (0.7, 0.2)}
        want |= {(2, 1): (0.4, 0.5)}
        for (r, c), (x0, y0) in want.items():
            grey = image[
                r * cell + 1 : (r + 1) * cell - 1, c * cell + 1 : (c + 1) * cell - 1, 0
            ]
            x, y = _centroid(grey)
            assert abs(x - x0) < 0.01 and abs(y - y0) < 0.01


class TestTrajectoryPng:
    def test_each_path_ends_in_its_color(self, tmp_path):
        """The true future ends in green, constant velocity in blue and the
        prediction in red, at the pixels of their last positions; the box is
        drawn on the border and the observed path in grey."""
        canvas = 64
        true_xy = torch.tensor([[0.2, 0.2], [0.3, 0.3], [0.4, 0.4], [0.5, 0.45]])
        true_xy = true_xy.view(1, 4, 1, 2)  # observed 2, horizon 2
        pred_xy = torch.tensor([[0.35, 0.6], [0.3, 0.8]]).view(1, 2, 1, 2)
        cv_xy = torch.tensor([[0.6, 0.3], [0.8, 0.2]]).view(1, 2, 1, 2)
        path = tmp_path / "traj.png"
        trajectory_png(path, true_xy, pred_xy, cv_xy, observed=2, canvas=canvas)
        image = read_png(path)
        assert image.shape == (canvas, canvas, 3)

        def at(xy: list[float]) -> tuple[int, ...]:
            px, py = (
                to_pixels(torch.tensor(xy, dtype=torch.float64), canvas).round().long()
            )
            return tuple(image[int(py), int(px)].tolist())

        assert at([0.5, 0.45]) == COLORS["true"]
        assert at([0.3, 0.8]) == COLORS["pred"]
        assert at([0.8, 0.2]) == COLORS["cv"]
        assert at([0.2, 0.2]) == COLORS["observed"]
        assert tuple(image[0, 0].tolist()) == BOX
        assert tuple(image[canvas - 1, canvas // 2].tolist()) == BOX


class TestPickWindows:
    def test_half_with_a_contact_half_without(self):
        contact = torch.tensor([False, True, False, False, True, True])
        assert pick_windows(contact) == [1, 4, 0, 2]  # contacts first

    def test_the_other_kind_fills_in(self):
        contact = torch.tensor([False, True, False, False, False])
        assert pick_windows(contact) == [1, 0, 2, 3]
