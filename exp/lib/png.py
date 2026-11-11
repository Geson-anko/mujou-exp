"""Pictures to look at, with torch and zlib only: an RGB PNG written and read
back, frames tiled into one image, and points and lines drawn into it.

An image is a ``(H, W, 3)`` uint8 tensor. Drawing takes pixel coordinates
``xy`` with ``xy[..., 0]`` the column (x, to the right) and ``xy[..., 1]`` the
row (y, downwards) from the top-left pixel, rounded to the nearest pixel; what
falls outside the image, or is not finite, is not drawn.
"""

import math
import struct
import zlib
from pathlib import Path

import torch
from torch import nn

__all__ = ["draw_points_", "draw_polyline_", "read_png", "tile", "write_png"]

_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def write_png(path: Path, image: torch.Tensor) -> None:
    """Write ``image`` ``(H, W, 3)`` uint8 as an 8-bit RGB PNG (no filter)."""
    h, w, _ = image.shape
    rows = image.cpu().reshape(h, w * 3).tolist()
    data = b"".join(b"\0" + bytes(row) for row in rows)

    def chunk(kind: bytes, body: bytes) -> bytes:
        crc = zlib.crc32(kind + body)
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", crc)

    header = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    png = _SIGNATURE + chunk(b"IHDR", header)
    png += chunk(b"IDAT", zlib.compress(data)) + chunk(b"IEND", b"")
    path.write_bytes(png)


def read_png(path: Path) -> torch.Tensor:
    """The image ``(H, W, 3)`` uint8 of a PNG that ``write_png`` wrote (8-bit
    RGB, not interlaced, every row of filter 0).

    Raises:
        ValueError: The file is not such a PNG.
    """
    data = path.read_bytes()
    if not data.startswith(_SIGNATURE):
        raise ValueError(f"{path} is not a PNG")
    pos, idat, size = len(_SIGNATURE), b"", (0, 0)
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        kind, body = data[pos + 4 : pos + 8], data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if kind == b"IHDR":
            w, h, depth, color, _, _, interlace = struct.unpack(">IIBBBBB", body)
            if (depth, color, interlace) != (8, 2, 0):
                raise ValueError(f"{path} is not an 8-bit RGB PNG without interlace")
            size = (h, w)
        elif kind == b"IDAT":
            idat += body
    h, w = size
    raw = zlib.decompress(idat)
    stride = 1 + 3 * w
    rows = [raw[i * stride : (i + 1) * stride] for i in range(h)]
    if any(row[0] != 0 for row in rows):
        raise ValueError(f"{path} has rows of a filter other than 0")
    pixels = b"".join(row[1:] for row in rows)
    return torch.frombuffer(bytearray(pixels), dtype=torch.uint8).reshape(h, w, 3)


def tile(frames: torch.Tensor, scale: int = 1) -> torch.Tensor:
    """Frames ``(rows, cols, C, H, W)`` in [0, 1] (``C`` 1 or 3) as one image
    ``(rows (H scale + 2), cols (W scale + 2), 3)`` uint8: each frame scaled
    up ``scale`` times, framed by a gap of 1 grey pixel on each side."""
    rows, cols, c, _, _ = frames.shape
    x = frames.clamp(0, 1).expand(-1, -1, 3, -1, -1) if c == 1 else frames.clamp(0, 1)
    x = x.repeat_interleave(scale, -2).repeat_interleave(scale, -1)
    x = nn.functional.pad(x, (1, 1, 1, 1), value=0.5)  # (rows, cols, 3, H'', W'')
    x = x.permute(0, 3, 1, 4, 2).reshape(rows * x.shape[3], cols * x.shape[4], 3)
    return (x * 255).round().to(torch.uint8)


def _color(canvas: torch.Tensor, color: tuple[int, int, int]) -> torch.Tensor:
    return torch.tensor(color, dtype=torch.uint8, device=canvas.device)


def _put_(canvas: torch.Tensor, rc: torch.Tensor, color: torch.Tensor) -> None:
    """Set the pixels ``rc`` ``(n, 2)`` (row, column; long) that are inside."""
    h, w, _ = canvas.shape
    inside = (rc[:, 0] >= 0) & (rc[:, 0] < h) & (rc[:, 1] >= 0) & (rc[:, 1] < w)
    rc = rc[inside]
    canvas[rc[:, 0], rc[:, 1]] = color


def draw_points_(
    canvas: torch.Tensor,
    xy: torch.Tensor,
    color: tuple[int, int, int],
    *,
    radius: int = 0,
) -> None:
    """Draw a disc of ``radius`` pixels (0: one pixel) at each point ``xy``
    ``(n, 2)`` (module docstring) into ``canvas`` ``(H, W, 3)`` uint8, in
    place: the pixels within ``radius`` of the point's pixel."""
    h, w, _ = canvas.shape
    xy = xy.detach().reshape(-1, 2).to(canvas.device, torch.float64)
    xy = xy[torch.isfinite(xy).all(1)]
    far = float(max(h, w) + radius + 1)  # beyond: drawn nowhere, as there
    centers = xy.clamp(-far, far).round().long()
    span = torch.arange(-radius, radius + 1, device=canvas.device)
    dy, dx = torch.meshgrid(span, span, indexing="ij")
    disc = (dx * dx + dy * dy) <= radius * radius
    offsets = torch.stack([dy[disc], dx[disc]], 1)  # (k, 2) as (row, column)
    rc = (centers.flip(1)[:, None, :] + offsets[None]).reshape(-1, 2)
    _put_(canvas, rc, _color(canvas, color))


def _clip(
    a: tuple[float, float], b: tuple[float, float], w: int, h: int
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """The part of the segment ``a``-``b`` (x, y) within the pixels of a
    ``w`` by ``h`` image (Liang-Barsky), or ``None`` if none is."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    t0, t1 = 0.0, 1.0
    for p, q in (
        (-dx, a[0] + 0.5),
        (dx, w - 0.5 - a[0]),
        (-dy, a[1] + 0.5),
        (dy, h - 0.5 - a[1]),
    ):
        if p == 0:
            if q < 0:
                return None
            continue
        r = q / p
        if p < 0:
            t0 = max(t0, r)
        else:
            t1 = min(t1, r)
    if t0 > t1:
        return None
    return (a[0] + t0 * dx, a[1] + t0 * dy), (a[0] + t1 * dx, a[1] + t1 * dy)


def draw_polyline_(
    canvas: torch.Tensor, xy: torch.Tensor, color: tuple[int, int, int]
) -> None:
    """Draw the line through the points ``xy`` ``(n, 2)`` (module docstring),
    one pixel wide, into ``canvas`` ``(H, W, 3)`` uint8, in place.

    Each segment is cut to the image, its ends rounded to pixels, and sampled
    at ``max(|dx|, |dy|) + 1`` evenly spaced points rounded to pixels, so that
    its pixels touch (8-connected). A segment with an end that is not finite
    is not drawn.
    """
    h, w, _ = canvas.shape
    pts = [(float(x), float(y)) for x, y in xy.detach().reshape(-1, 2).tolist()]
    if len(pts) == 1:
        draw_points_(canvas, xy, color)
        return
    out = []
    for a, b in zip(pts[:-1], pts[1:], strict=True):
        if not all(math.isfinite(v) for v in (*a, *b)):
            continue
        cut = _clip(a, b, w, h)
        if cut is None:
            continue
        p0 = torch.tensor(cut[0], dtype=torch.float64).round()
        p1 = torch.tensor(cut[1], dtype=torch.float64).round()
        n = int((p1 - p0).abs().max()) + 1
        s = torch.linspace(0, 1, n, dtype=torch.float64)
        out.append(p0 + s[:, None] * (p1 - p0))
    if out:
        rc = torch.cat(out).round().long().flip(1).to(canvas.device)
        _put_(canvas, rc, _color(canvas, color))
