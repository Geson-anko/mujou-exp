import struct
import zlib

import pytest
import torch

from exp.lib.png import draw_points_, draw_polyline_, read_png, tile, write_png

RED = (255, 0, 0)


def _canvas(h: int = 9, w: int = 12) -> torch.Tensor:
    return torch.zeros(h, w, 3, dtype=torch.uint8)


def _drawn(canvas: torch.Tensor) -> set[tuple[int, int]]:
    """The (row, column) of the pixels that are not black."""
    return {(int(r), int(c)) for r, c in canvas.any(-1).nonzero().tolist()}


class TestWritePng:
    def test_the_png_holds_the_pixels(self, tmp_path):
        """Read back by hand: the signature, the size in IHDR, and the filtered
        rows (filter 0) in IDAT."""
        g = torch.Generator().manual_seed(0)
        image = torch.randint(0, 256, (3, 5, 3), generator=g, dtype=torch.uint8)
        path = tmp_path / "a.png"
        write_png(path, image)
        data = path.read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n"
        w, h = struct.unpack(">II", data[16:24])
        assert (w, h) == (5, 3)
        start = data.index(b"IDAT") + 4
        (length,) = struct.unpack(">I", data[start - 8 : start - 4])
        raw = zlib.decompress(data[start : start + length])
        rows = [raw[i * 16 : (i + 1) * 16] for i in range(3)]
        assert all(r[0] == 0 for r in rows)
        assert [list(r[1:]) for r in rows] == image.reshape(3, 15).tolist()

    def test_the_crc_of_each_chunk_holds(self, tmp_path):
        path = tmp_path / "a.png"
        write_png(path, torch.full((2, 2, 3), 7, dtype=torch.uint8))
        data, pos = path.read_bytes(), 8
        kinds = []
        while pos < len(data):
            (length,) = struct.unpack(">I", data[pos : pos + 4])
            kind_body = data[pos + 4 : pos + 8 + length]
            (crc,) = struct.unpack(">I", data[pos + 8 + length : pos + 12 + length])
            assert zlib.crc32(kind_body) == crc
            kinds.append(kind_body[:4])
            pos += 12 + length
        assert kinds == [b"IHDR", b"IDAT", b"IEND"]


class TestReadPng:
    def test_it_reads_back_what_was_written(self, tmp_path):
        g = torch.Generator().manual_seed(1)
        image = torch.randint(0, 256, (7, 4, 3), generator=g, dtype=torch.uint8)
        write_png(tmp_path / "a.png", image)
        assert torch.equal(read_png(tmp_path / "a.png"), image)

    def test_another_file_raises(self, tmp_path):
        (tmp_path / "a.png").write_bytes(b"GIF89a")
        with pytest.raises(ValueError, match="not a PNG"):
            read_png(tmp_path / "a.png")


class TestTile:
    def test_the_tile_places_each_frame_with_a_gap(self):
        frames = torch.zeros(2, 3, 1, 4, 4)
        frames[1, 2] = 1.0
        out = tile(frames, scale=2)
        assert out.shape == (2 * 10, 3 * 10, 3)
        assert out[0, 0, 0] == 128  # the gap
        assert (out[11:19, 21:29] == 255).all()  # frame (1, 2), inside its gap
        assert (out[1:9, 1:9] == 0).all()

    def test_a_colour_frame_keeps_its_channels(self):
        frames = torch.zeros(1, 1, 3, 2, 2)
        frames[0, 0, 1] = 1.0  # green
        out = tile(frames)
        assert out[1:3, 1:3].tolist() == [[[0, 255, 0]] * 2] * 2


class TestDrawPoints:
    def test_a_point_is_its_nearest_pixel(self):
        """x is the column and y the row: (3.4, 5.6) is row 6, column 3."""
        canvas = _canvas()
        draw_points_(canvas, torch.tensor([[3.4, 5.6]]), RED)
        assert _drawn(canvas) == {(6, 3)}
        assert canvas[6, 3].tolist() == list(RED)

    def test_a_disc_is_the_pixels_within_its_radius(self):
        canvas = _canvas(11, 11)
        draw_points_(canvas, torch.tensor([[5.0, 5.0]]), RED, radius=2)
        want = {
            (r, c)
            for r in range(11)
            for c in range(11)
            if (r - 5) ** 2 + (c - 5) ** 2 <= 4
        }
        assert _drawn(canvas) == want

    def test_points_outside_or_not_finite_are_not_drawn(self):
        canvas = _canvas()
        xy = torch.tensor(
            [[-3.0, 2.0], [13.0, 0.0], [1e30, 1.0], [float("nan"), 1.0], [0.0, 8.0]]
        )
        draw_points_(canvas, xy, RED, radius=1)
        assert _drawn(canvas) == {(8, 0), (7, 0), (8, 1)}


class TestDrawPolyline:
    def test_a_horizontal_line_covers_the_pixels_between_its_ends(self):
        canvas = _canvas()
        draw_polyline_(canvas, torch.tensor([[2.0, 4.0], [9.0, 4.0]]), RED)
        assert _drawn(canvas) == {(4, c) for c in range(2, 10)}

    def test_a_diagonal_line_is_8_connected_from_end_to_end(self):
        canvas = _canvas(20, 20)
        draw_polyline_(
            canvas, torch.tensor([[1.0, 2.0], [17.0, 9.0], [4.0, 18.0]]), RED
        )
        drawn = _drawn(canvas)
        assert {(2, 1), (9, 17), (18, 4)} <= drawn
        # Walking from the first end through touching pixels reaches all of it.
        seen, todo = set(), [(2, 1)]
        while todo:
            r, c = todo.pop()
            if (r, c) in seen or (r, c) not in drawn:
                continue
            seen.add((r, c))
            todo += [(r + i, c + j) for i in (-1, 0, 1) for j in (-1, 0, 1)]
        assert seen == drawn
        # One pixel per column of the first segment (its major axis is x).
        first = {c for r, c in drawn if c <= 17 and r <= 9}
        assert first == set(range(1, 18))

    def test_a_segment_is_cut_to_the_image(self):
        """From far left to the middle of row 3: the pixels of columns 0..5,
        drawn at the cost of the image, not of the segment."""
        canvas = _canvas()
        draw_polyline_(canvas, torch.tensor([[-1e12, 3.0], [5.0, 3.0]]), RED)
        assert _drawn(canvas) == {(3, c) for c in range(6)}

    def test_a_segment_with_an_end_not_finite_is_skipped(self):
        canvas = _canvas()
        xy = torch.tensor([[0.0, 0.0], [float("nan"), 1.0], [3.0, 1.0], [5.0, 1.0]])
        draw_polyline_(canvas, xy, RED)
        assert _drawn(canvas) == {(1, c) for c in range(3, 6)}
