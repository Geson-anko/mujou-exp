"""Videos as u8 frames: decoding with ffmpeg, and frames on disk.

``ffmpeg_frames`` streams the frames of a video through the ffmpeg of
``imageio-ffmpeg``, resampled in time and shrunk to a centered crop, without
holding the video in memory. ``U8Writer`` appends such frames to a raw file and
records it in ``index.json``; ``U8Reader`` memory-maps the file back.

A frame directory holds:

- ``frames.u8``: the frames ``(N, 3, H, W)`` uint8 in C order
- ``index.json``: ``frames`` (N), ``shape`` ([3, H, W]), ``sha256`` (of
  ``frames.u8``) and ``meta`` (what the writer was given). Nothing in it
  depends on the clock, so the same frames give the same bytes.
"""

import ctypes
import hashlib
import json
import subprocess
import tempfile
from collections.abc import Generator, Mapping
from pathlib import Path
from types import TracebackType
from typing import IO, Any, Self

import imageio_ffmpeg
import torch
from torch import Tensor

__all__ = [
    "FRAMES",
    "INDEX",
    "U8Reader",
    "U8Writer",
    "ffmpeg_filter",
    "ffmpeg_frames",
    "ffmpeg_version",
    "sha256_file",
]

FRAMES = "frames.u8"
INDEX = "index.json"


def ffmpeg_version() -> str:
    """The first line of ``ffmpeg -version`` of the ffmpeg in use."""
    out = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-version"],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.splitlines()[0]


def ffmpeg_filter(*, fps: float, width: int, height: int) -> str:
    """The ffmpeg filter of ``ffmpeg_frames``: ``fps`` frames per second (the
    input frame nearest to each output time), the largest centered crop of
    aspect ``width : height``, then an area-average shrink to ``width`` x
    ``height``."""
    crop = f"crop='min(iw,ih*{width}/{height})':'min(ih,iw*{height}/{width})'"
    return f"fps={fps},{crop},scale={width}:{height}:flags=area"


def ffmpeg_frames(
    path: Path,
    *,
    fps: float,
    width: int,
    height: int,
    start: float = 0.0,
    duration: float | None = None,
    threads: int = 4,
) -> Generator[Tensor]:
    """The frames of the video at ``path``, one u8 ``(3, height, width)``
    tensor (RGB) per frame, as ``ffmpeg_filter`` makes them.

    Args:
        path: The video.
        fps: The frames per second of the output.
        width: The width of the output, in pixels.
        height: The height of the output, in pixels.
        start: The first second of the video to decode.
        duration: The seconds to decode (``None``: to the end).
        threads: ffmpeg's decoding threads.

    Frames are read from ffmpeg's pipe one at a time; closing the iterator
    early stops ffmpeg.

    Raises:
        RuntimeError: ffmpeg failed (its messages are in the error), or its
            output ended inside a frame.
    """
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error"]
    cmd += ["-threads", str(threads)]
    if start:
        cmd += ["-ss", repr(float(start))]
    if duration is not None:
        cmd += ["-t", repr(float(duration))]
    cmd += ["-i", str(path), "-an", "-sn"]
    cmd += ["-vf", ffmpeg_filter(fps=fps, width=width, height=height)]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    n = 3 * width * height
    # stderr goes to a file: a pipe nobody reads could fill and block ffmpeg
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err)
        assert proc.stdout is not None
        try:
            while buf := proc.stdout.read(n):
                if len(buf) < n:
                    code = proc.wait()  # stdout is closed: ffmpeg is exiting
                    raise RuntimeError(
                        f"ffmpeg's output of {path} ended inside a frame "
                        f"({len(buf)} of {n} bytes, exit {code}): {_read(err)}"
                    )
                hwc = torch.frombuffer(bytearray(buf), dtype=torch.uint8)
                yield hwc.view(height, width, 3).permute(2, 0, 1).contiguous()
            code = proc.wait()
        finally:
            if proc.poll() is None:  # closed early, or an error above
                proc.kill()
                proc.wait()
            proc.stdout.close()
        if code != 0:
            raise RuntimeError(f"ffmpeg failed on {path} (exit {code}): {_read(err)}")


def _read(err: IO[bytes]) -> str:
    """The text written to ``err`` so far."""
    err.seek(0)
    return err.read().decode(errors="replace").strip()


def sha256_file(path: Path) -> str:
    """The sha256 of the file at ``path``, in hex."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def _bytes(x: Tensor) -> bytes:
    """The raw C-order bytes of a uint8 tensor."""
    x = x.contiguous()
    return ctypes.string_at(x.data_ptr(), x.numel())


class U8Writer:
    """Appends u8 frames of ``shape`` (3, H, W) to ``directory/frames.u8``;
    ``close`` writes ``index.json`` with ``meta`` (JSON values). Use it as a
    context manager: leaving the block without an error closes it, leaving it
    with one writes no index.

    The directory is created and must not hold frames already.
    """

    def __init__(
        self,
        directory: Path,
        shape: tuple[int, int, int],
        meta: Mapping[str, Any] | None = None,
    ) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self._dir = directory
        self._shape = tuple(shape)
        self._meta = dict(meta or {})
        self._file = open(directory / FRAMES, "xb")
        self._hash = hashlib.sha256()
        self.frames = 0  # appended so far

    def append(self, frame: Tensor) -> None:
        """Append one frame, u8 ``(3, H, W)``.

        Raises:
            ValueError: ``frame`` is not a uint8 CPU tensor of the writer's
                shape.
        """
        if (
            frame.dtype != torch.uint8
            or tuple(frame.shape) != self._shape
            or frame.device.type != "cpu"  # its bytes are read by pointer
        ):
            raise ValueError(
                f"expected a uint8 CPU frame of shape {self._shape}, "
                f"got {frame.dtype} {tuple(frame.shape)} on {frame.device}"
            )
        data = _bytes(frame)
        self._file.write(data)
        self._hash.update(data)
        self.frames += 1

    def close(self) -> dict[str, Any]:
        """Close ``frames.u8``, write ``index.json`` and return its content."""
        self._file.close()
        index = {
            "frames": self.frames,
            "shape": list(self._shape),
            "sha256": self._hash.hexdigest(),
            "meta": self._meta,
        }
        (self._dir / INDEX).write_text(json.dumps(index, indent=1, sort_keys=True))
        return index

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self.close()
        else:
            self._file.close()


class U8Reader:
    """The frames of a directory of ``U8Writer``, memory-mapped (read-only in
    effect: writes to the tensors do not reach the file).

    ``reader[k]`` is frame ``k``, u8 ``(3, H, W)``; ``reader[a:b]`` is
    ``(b - a, 3, H, W)``. ``index`` is the content of ``index.json``.

    Raises:
        ValueError: ``frames.u8`` does not have the size of the index.
    """

    def __init__(self, directory: Path) -> None:
        self._dir = directory
        self.index: dict[str, Any] = json.loads((directory / INDEX).read_text())
        c, h, w = self.index["shape"]
        self.shape: tuple[int, int, int] = (c, h, w)
        n = self.index["frames"]
        size = n * c * h * w
        actual = (directory / FRAMES).stat().st_size
        if actual != size:
            raise ValueError(
                f"{directory / FRAMES} has {actual} bytes, the index says {size}"
            )
        if n:
            flat = torch.from_file(
                str(directory / FRAMES), shared=False, size=size, dtype=torch.uint8
            )
        else:  # from_file cannot map an empty file
            flat = torch.zeros(0, dtype=torch.uint8)
        self._data = flat.view(n, *self.shape)

    def __len__(self) -> int:
        return self._data.shape[0]

    def __getitem__(self, k: int | slice | Tensor) -> Tensor:
        return self._data[k]

    def verify(self) -> None:
        """Check ``frames.u8`` against the sha256 of the index (the size is
        checked on opening).

        Raises:
            ValueError: the file differs from the index.
        """
        path = self._dir / FRAMES
        digest = sha256_file(path)
        if digest != self.index["sha256"]:
            raise ValueError(
                f"{path} has sha256 {digest}, the index says {self.index['sha256']}"
            )
