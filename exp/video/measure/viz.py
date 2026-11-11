"""The pictures of M14 to look at: the predicted balls rendered next to the true
ones (``frames_png``) and the predicted paths over the true ones
(``trajectory_png``), of a few fixed windows of the validation clips: the
first half with a wall contact up to 10 frames ahead, the others without
(``pick_windows``, ``visualize``). For the observation ``"pixels"`` (stage 3)
the prediction is itself an image, drawn as it is (``pixel_frames_png``,
clipped to [0, 1] by ``tile``), and there is no ``traj.png``: the model
predicts no positions.

Torch and ``exp.lib.png`` only. Positions are in the box $[0, 1]^2$ with y up;
on a canvas of ``c`` pixels, $(x, y)$ is the pixel of column $x (c - 1)$ and
row $(1 - y)(c - 1)$.
"""

from collections.abc import Sequence
from pathlib import Path

import torch
from torch import Tensor

from exp.lib.png import draw_points_, draw_polyline_, tile, write_png
from exp.video.data.clips import ClipSet, windows
from exp.video.data.observe import (
    PIXELS,
    Standardizer,
    position_columns,
    render_flat,
    to_states,
)
from exp.video.data.physics import render
from exp.video.dyn.modules import DynamicsModules
from exp.video.measure.baselines import const_velocity
from exp.video.measure.dynamics import rollout
from exp.video.schema.measure import MeasureConf

__all__ = [
    "BOX",
    "COLORS",
    "frames_png",
    "pick_windows",
    "pixel_frames_png",
    "to_pixels",
    "trajectory_png",
    "visualize",
]

COLORS = {
    "observed": (128, 128, 128),
    "true": (0, 200, 0),
    "cv": (40, 90, 255),
    "pred": (230, 30, 30),
}
BOX = (90, 90, 90)  # the walls a ball's center cannot cross are inside it


def frames_png(
    path: Path,
    true_states: Tensor,
    pred_states: Tensor,
    cv_states: Tensor,
    radii: Sequence[float],
    *,
    size: int = 32,
    columns: Sequence[int] = (0, 1, 2, 5, 10, 20),
    clips: int = 4,
    scale: int = 3,
) -> None:
    """Write the balls of ``clips`` windows rendered (``physics.render``) at
    the frames ``columns`` past the last observed one (0: that frame): per
    window a row of the truth, of the prediction and of constant velocity.

    Args:
        true_states: ``(n, 1 + H, 4K)``: index 0 the last observed frame,
            ``h`` the frame ``h`` ahead (``H >= max(columns)``).
        pred_states: ``(n, 1 + H, 4K)``, index 0 the last observed frame too.
        cv_states: ``(n, 1 + H, 4K)``, the same.
    """
    cols = list(columns)
    rows = []
    for i in range(min(clips, len(true_states))):
        for states in (true_states, pred_states, cv_states):
            s = states[i, cols].double()  # (len(cols), 4K)
            rows.append(render(s, size, radii))  # (len(cols), 1, size, size)
    _write_tiles(path, rows, scale)


def _write_tiles(path: Path, rows: Sequence[Tensor], scale: int) -> None:
    """Write ``rows`` (each ``(columns, 1, size, size)``) as one tiled PNG."""
    image = tile(torch.stack(list(rows)).float(), scale)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_png(path, image)


def pixel_frames_png(
    path: Path,
    true_states: Tensor,
    pred_images: Tensor,
    cv_states: Tensor,
    radii: Sequence[float],
    *,
    size: int = 32,
    columns: Sequence[int] = (0, 1, 2, 5, 10, 20),
    clips: int = 4,
    scale: int = 3,
) -> None:
    """``frames_png`` for a model that predicts images: the same rows (truth,
    prediction, constant velocity) and columns, but the prediction is
    ``pred_images`` as they are.

    Args:
        true_states: ``(n, 1 + H, 4K)``, as ``frames_png``.
        pred_images: ``(n, 1 + H, size * size)`` raw (not standardized)
            images, index 0 the last observed frame.
        cv_states: ``(n, 1 + H, 4K)``.
    """
    cols = list(columns)
    rows = []
    for i in range(min(clips, len(true_states))):
        truth = render(true_states[i, cols].double(), size, radii)
        pred = pred_images[i, cols].reshape(len(cols), 1, size, size)
        cv = render(cv_states[i, cols].double(), size, radii)
        rows += [truth, pred, cv]
    _write_tiles(path, rows, scale)


def to_pixels(xy: Tensor, canvas: int) -> Tensor:
    """Box positions ``(..., 2)`` as pixels ``(..., 2)`` (x right, y down) of a
    canvas of ``canvas`` pixels (module docstring)."""
    px = xy[..., 0] * (canvas - 1)
    py = (1 - xy[..., 1]) * (canvas - 1)
    return torch.stack([px, py], -1)


def trajectory_png(
    path: Path,
    true_xy: Tensor,
    pred_xy: Tensor,
    cv_xy: Tensor,
    *,
    observed: int,
    canvas: int = 256,
    clips: int = 4,
) -> None:
    """Write ``clips`` panels side by side, one per window, of ``canvas``
    pixels each on black: the box (``BOX``), and per ball the observed path
    (grey), the true future (green), constant velocity (blue) and the
    prediction (red), each future from the last observed position.

    Args:
        true_xy: ``(n, observed + H, K, 2)`` positions in the box.
        pred_xy: ``(n, H, K, 2)``.
        cv_xy: ``(n, H, K, 2)``.
    """
    n = min(clips, len(true_xy))
    image = torch.zeros(canvas, canvas * n, 3, dtype=torch.uint8)
    for i in range(n):
        panel = torch.zeros(canvas, canvas, 3, dtype=torch.uint8)
        c = canvas - 1
        corners = torch.tensor([[0, 0], [c, 0], [c, c], [0, c], [0, 0]])
        draw_polyline_(panel, corners.double(), BOX)
        last = true_xy[i, observed - 1 : observed]  # (1, K, 2)
        paths = {
            "observed": true_xy[i, :observed],
            "true": true_xy[i, observed - 1 :],
            "cv": torch.cat([last, cv_xy[i]]),
            "pred": torch.cat([last, pred_xy[i]]),
        }
        for name, xy in paths.items():  # the prediction last, on top
            px = to_pixels(xy.double(), canvas)  # (T, K, 2)
            for k in range(px.shape[1]):
                draw_polyline_(panel, px[:, k], COLORS[name])
            if name != "observed":
                draw_points_(panel, px[-1], COLORS[name], radius=2)
        image[:, canvas * i : canvas * (i + 1)] = panel
    path.parent.mkdir(parents=True, exist_ok=True)
    write_png(path, image)


def pick_windows(contact: Tensor, clips: int = 4) -> list[int]:
    """The windows to draw, in the order of the pictures: the first
    ``clips // 2`` with a contact (``contact`` ``(n,)`` bool), then the first
    without; when one kind runs short, the other fills in (at the end)."""
    with_contact = contact.nonzero().squeeze(1).tolist()
    without = (~contact).nonzero().squeeze(1).tolist()
    half = clips // 2
    picked = with_contact[:half] + without[: clips - half]
    rest = with_contact[half:] + without[clips - half :]
    return picked + rest[: clips - len(picked)]


@torch.no_grad()
def visualize(
    out: Path,
    mods: DynamicsModules,
    clips: ClipSet,
    picked: Sequence[int],
    std: Standardizer,
    m: MeasureConf,
    *,
    obs: str,
    balls: int,
    residual: bool,
    size: int,
) -> None:
    """``out/frames.png`` and ``out/traj.png`` of the rollout of ``mods`` on
    the windows ``picked`` of ``clips`` (``pick_windows``), up to the largest
    of ``m.horizons``; for ``"pixels"`` ``frames.png`` only (module
    docstring)."""
    o, horizon = m.observed, max(m.horizons)
    x, states = windows(clips, obs, std, size=size)
    idx = torch.tensor(list(picked))
    x, states = x[idx], states[idx]
    device = mods.readout.weight.device
    pred = rollout(mods, x.to(device), residual=residual, observed=o, horizon=horizon)
    pred_raw = std.invert(pred.cpu().double())
    if obs == PIXELS:
        _pixel_pictures(out, pred_raw, states, clips, m, size=size)
        return
    raw = std.invert(x.double())
    cv_raw = const_velocity(raw, obs, balls, o, horizon, clips.frame)
    last = states[:, o - 1 : o]
    span = slice(o - 1, o + horizon)
    pred_states = torch.cat([last, to_states(pred_raw, obs, balls)], 1)
    cv_states = torch.cat([last, to_states(cv_raw, obs, balls)], 1)
    columns = [c for c in (0, 1, 2, 5, 10, 20) if c <= horizon]
    frames_png(
        out / "frames.png",
        states[:, span],
        pred_states,
        cv_states,
        clips.radii,
        size=size,
        columns=columns,
        clips=len(idx),
    )
    xy = position_columns(obs, balls)  # (K, 2)
    true_xy = states[:, : o + horizon].unflatten(-1, (balls, 4))[..., :2]
    trajectory_png(
        out / "traj.png",
        true_xy,
        pred_raw[..., xy],
        cv_raw[..., xy],
        observed=o,
        clips=len(idx),
    )


def _pixel_pictures(
    out: Path,
    pred_raw: Tensor,
    states: Tensor,
    clips: ClipSet,
    m: MeasureConf,
    *,
    size: int,
) -> None:
    """``out/frames.png`` of the predicted images ``pred_raw`` ``(n, H, size^2)``
    of the windows with the true ``states``."""
    o, horizon = m.observed, max(m.horizons)
    k = len(clips.radii)
    last = states[:, o - 1 : o]
    cv = const_velocity(states, "pos_vel", k, o, horizon, clips.frame)
    pred = torch.cat([render_flat(last, clips.radii, size), pred_raw.float()], 1)
    pixel_frames_png(
        out / "frames.png",
        states[:, o - 1 : o + horizon],
        pred,
        torch.cat([last, cv], 1),
        clips.radii,
        size=size,
        columns=[c for c in (0, 1, 2, 5, 10, 20) if c <= horizon],
        clips=len(states),
    )
