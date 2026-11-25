r"""The prepared Walking Tours' square at a higher resolution, at chosen
frames only (V18c's main measure, ``exp/vision/plans/jepa.md``, 「高解像度の
フレーム」): data for evaluation only, the prepared data (its
``data_version``) left as it is.

The prepared frames are the largest centered square of a video (``video_io.
ffmpeg_filter``): for a source of $W$ x $c$ ($W \ge c$) the columns $[x_0,
x_0 + c)$, $x_0 = (W - c) / 2$, shrunk by the area average to the data's
size $s$. Here the same square is shrunk to ``size`` $= f s$ ($f$ a whole
number, 5 for 320 of 64; ``hires_filter``), so that the $f \times f$ means
of a frame here are the prepared frame up to rounding: the area average of
$c / s$ source pixels a side is the mean of the $f \times f$ area averages
of $c / (f s)$ pixels whose bounds fall on its own. The two differ by the u8
rounding of each (at most 1/2 a level, less on average) and ffmpeg's
placement of the 4:2:0 colors.

``prepare`` decodes a video, keeps the frames of a list (``frames.json``
of ``python -m exp.vision.sem_eval frames``) into ``<out>/<video>/``
(``video_io.U8Writer``; ``frames.json`` there: the numbers and the largest
$d(f)$), and checks
the place of every kept frame $f$ against the prepared data $x$ before it
is written (``position_check``; else it stops, writing no index): with $d(g)$
the mean absolute difference of the frame's $f \times f$ means (unrounded)
and $x_g$,

- (a) $d(f) \le$ ``MAX_DIFF`` (1.0 levels);
- (b) for each neighbour $g = f \pm 1$ that the video moves to
  ($\langle \lvert x_f - x_g \rvert \rangle >$ ``MOVING`` = 2 levels),
  $d(f) < d(g)$: a frame one off in time is told apart.

Once the videos are there, ``finalize`` writes ``<out>/index.json`` (each
video's frame numbers and the sha256 of its index; the source data, fps,
size and filter) and renames ``<out>`` to ``hires-<12 hex>`` of that index.
``HiresFrames`` reads the frames at (video, frame) places::

    python -m exp.lib.tours.hires prepare <video> --src /datasets/walking_tours \
        --data $MUJOU_EXP_DATA/<data_version> --frames <frames.json> --out <dir>
    python -m exp.lib.tours.hires finalize <dir>
"""

import argparse
import hashlib
import json
import os
import shutil
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from exp.lib.tours.walking_tours import MANIFEST, data_version, manifest_entry
from exp.lib.video_io import INDEX, U8Reader, U8Writer, ffmpeg_frames, ffmpeg_version

__all__ = [
    "FRAME_LIST",
    "MAX_DIFF",
    "MOVING",
    "PREFIX",
    "SIZE",
    "HiresFrames",
    "finalize",
    "hires_filter",
    "hires_geometry",
    "position_check",
    "prepare",
]

SIZE = 320  # 5 x 64
PREFIX = "hires-"
FRAME_LIST = "frames.json"  # a video's kept frames and d(f)'s largest, in its directory
MAX_DIFF = 1.0  # (a): levels of u8
MOVING = 2.0  # (b): levels of u8 between neighbouring prepared frames


def hires_geometry(width: int, height: int, *, size: int = SIZE) -> dict[str, int]:
    r"""The square of a ``width`` x ``height`` video (module docstring):
    ``side`` $c$, ``x0`` $x_0$, ``size``.

    Raises:
        ValueError: The video is taller than wide, or the square is not
            centered on a whole column.
    """
    c = height
    if width < c or (width - c) % 2:
        raise ValueError(f"a {width} x {height} video has no square on whole columns")
    return {"side": c, "x0": (width - c) // 2, "size": size}


def hires_filter(geometry: dict[str, int], *, fps: float) -> str:
    """The ffmpeg filter of ``geometry`` (``hires_geometry``): ``fps``
    frames per second, the square cropped and shrunk by the area average."""
    c, x0, size = geometry["side"], geometry["x0"], geometry["size"]
    return f"fps={fps:g},crop={c}:{c}:{x0}:0,scale={size}:{size}:flags=area"


def _diff(a: Tensor, b: Tensor) -> float:
    return float((a.double() - b.double()).abs().mean())


def position_check(frame: Tensor, f: int, ref: U8Reader) -> dict[str, Any]:
    """The checks (a) and (b) (module docstring) of the frame u8 ``(3,
    size, size)`` as frame ``f`` of the prepared frames ``ref``:
    ``diff`` $d(f)$, ``neighbours`` $d(g)$ of the moving neighbours $g$
    (by frame), ``a`` and ``b`` whether each holds.

    Raises:
        ValueError: ``size`` is not a whole multiple of the prepared size.
    """
    s = ref.shape[-1]
    k = frame.shape[-1] // s
    if k * s != frame.shape[-1]:
        raise ValueError(f"a frame of {frame.shape[-1]} is not a multiple of {s}")
    down = F.avg_pool2d(frame.double()[None], k)[0]
    d = _diff(down, ref[f])
    moving = {
        g: _diff(down, ref[g])
        for g in (f - 1, f + 1)
        if 0 <= g < len(ref) and _diff(ref[f], ref[g]) > MOVING
    }
    return {
        "diff": d,
        "neighbours": moving,
        "a": d <= MAX_DIFF,
        "b": all(d < dg for dg in moving.values()),
    }


def prepare(
    video: str,
    src: Path,
    data: Path,
    frames: Sequence[int],
    out: Path,
    *,
    size: int = SIZE,
    threads: int = 4,
    manifest: Path = MANIFEST,
) -> Path:
    """Decode ``video`` (its manifest entry's file under ``src``) into
    ``out/<video>/`` (module docstring): the frames ``frames`` (of the
    prepared data ``data/<video>``, its fps; the numbers sorted, once
    each), each checked before it is written. Written as ``out/<video>.tmp/``
    (a stale one removed), then renamed. Returns the directory.

    Raises:
        ValueError: The manifest refuses ``video``, the file's size is not
            the manifest's, a frame number is not of the prepared data, the
            video has another number of frames than the prepared data, or a
            frame fails (a) or (b).
        FileExistsError: ``out/<video>`` exists.
    """
    entry = manifest_entry(video, manifest)
    path = src / entry["file"]
    if path.stat().st_size != entry["bytes"]:
        raise ValueError(
            f"{path} has {path.stat().st_size} bytes, the manifest says {entry['bytes']}"
        )
    ref = U8Reader(data / video)
    keep = sorted(set(frames))
    if keep and not 0 <= keep[0] <= keep[-1] < len(ref):
        raise ValueError(f"frames of {video} outside [0, {len(ref)})")
    fps = ref.index["meta"]["fps"]
    geometry = hires_geometry(entry["width"], entry["height"], size=size)
    vf = hires_filter(geometry, fps=fps)
    final = out / video
    if final.exists():
        raise FileExistsError(final)
    tmp = out / f"{video}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    meta = {
        "video": video,
        "file": entry["file"],
        "source_bytes": entry["bytes"],
        "data": data.name,
        "data_index_sha256": hashlib.sha256((data / INDEX).read_bytes()).hexdigest(),
        "fps": fps,
        "geometry": geometry,
        "filter": vf,
        "ffmpeg": ffmpeg_version(),
    }
    wanted = set(keep)
    worst_a, total = 0.0, 0
    decoded = ffmpeg_frames(
        path, fps=fps, width=size, height=size, threads=threads, vf=vf
    )
    with closing(decoded), U8Writer(tmp, (3, size, size), meta) as w:
        for i, frame in enumerate(decoded):
            total = i + 1
            if i not in wanted:
                continue
            check = position_check(frame, i, ref)
            if not (check["a"] and check["b"]):
                failed = [c for c in ("a", "b") if not check[c]]
                raise ValueError(
                    f"frame {i} of {video} fails the position check "
                    f"{', '.join(f'({c})' for c in failed)}: d(f) {check['diff']:.3f}, "
                    f"moving neighbours {check['neighbours']}"
                )
            worst_a = max(worst_a, check["diff"])
            w.append(frame)
        if total != len(ref):
            raise ValueError(f"{video}: {total} frames, the prepared data {len(ref)}")
        if w.frames != len(keep):
            raise ValueError(f"{video}: {w.frames} of {len(keep)} frames kept")
    (tmp / FRAME_LIST).write_text(json.dumps({"frames": keep, "max_diff": worst_a}))
    os.rename(tmp, final)
    return final


def finalize(root: Path) -> Path:
    """Write ``root/index.json`` over the videos under ``root`` (module
    docstring) and rename ``root`` to ``hires-<12 hex>`` of it beside it.
    Returns the new path.

    Raises:
        ValueError: A ``.tmp`` is left, there is no video, a video's index
            names another video, or the videos differ in their source data,
            fps or size.
        FileExistsError: A directory of that name exists already.
    """
    tmps = sorted(d.name for d in root.glob("*.tmp"))
    if tmps:
        raise ValueError(f"unfinished preps under {root}: {', '.join(tmps)}")
    dirs = sorted(d for d in root.iterdir() if (d / INDEX).is_file())
    if not dirs:
        raise ValueError(f"no video under {root}")
    entries: dict[str, Any] = {}
    shared: set[str] = set()
    for d in dirs:
        raw = (d / INDEX).read_bytes()
        meta = json.loads(raw)["meta"]
        if meta.get("video") != d.name:
            raise ValueError(
                f"{d / INDEX} is of video {meta.get('video')!r}, not {d.name!r}"
            )
        size = meta["geometry"]["size"]
        keys = ("data", "data_index_sha256", "fps")
        shared.add(json.dumps({k: meta[k] for k in keys} | {"size": size}))
        kept = json.loads((d / FRAME_LIST).read_text())
        entries[d.name] = {
            "frames": kept["frames"],
            "max_diff": kept["max_diff"],
            "index_sha256": hashlib.sha256(raw).hexdigest(),
            "filter": meta["filter"],
        }
    if len(shared) != 1:
        raise ValueError(f"the videos under {root} differ in their data or size")
    top = json.loads(shared.pop()) | {"videos": entries}
    (root / INDEX).write_text(json.dumps(top, indent=1, sort_keys=True))
    target = root.parent / f"{PREFIX}{data_version(root / INDEX)}"
    if target == root:
        return root
    if target.exists():
        raise FileExistsError(target)
    os.rename(root, target)
    return target


class HiresFrames:
    """The data of ``finalize`` at ``root``: ``index`` (its
    ``index.json``), ``size``, and the frames at (video, frame) places."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.index: dict[str, Any] = json.loads((root / INDEX).read_text())
        self.size: int = self.index["size"]
        self._readers: dict[str, U8Reader] = {}
        self._at: dict[str, dict[int, int]] = {
            v: {f: i for i, f in enumerate(e["frames"])}
            for v, e in self.index["videos"].items()
        }

    def frames(self, places: Sequence[tuple[str, int]]) -> Tensor:
        """The frames u8 ``(N, 3, size, size)`` at ``places``.

        Raises:
            ValueError: A place is not in the data.
        """
        missing = [(v, f) for v, f in places if f not in self._at.get(v, {})]
        if missing:
            raise ValueError(
                f"the data {self.root} lacks {len(missing)} places, e.g. {missing[0]}"
            )
        for v, _ in places:
            if v not in self._readers:
                self._readers[v] = U8Reader(self.root / v)
        return torch.stack([self._readers[v][self._at[v][f]] for v, f in places])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("prepare", help="decode one video's listed frames")
    q.add_argument("video")
    q.add_argument("--src", type=Path, default=Path("/datasets/walking_tours"))
    q.add_argument("--data", type=Path, required=True, help="the prepared data")
    q.add_argument("--frames", type=Path, required=True, help="sem_eval frames' list")
    q.add_argument("--out", type=Path, required=True)
    q.add_argument("--size", type=int, default=SIZE)
    q.add_argument("--threads", type=int, default=4)
    f = sub.add_parser("finalize", help="index OUT and rename it to its version")
    f.add_argument("root", type=Path)
    args = p.parse_args()
    if args.cmd == "prepare":
        lists = json.loads(args.frames.read_text())
        print(
            prepare(
                args.video,
                args.src,
                args.data,
                lists[args.video],
                args.out,
                size=args.size,
                threads=args.threads,
            )
        )
    else:
        print(finalize(args.root))


if __name__ == "__main__":
    main()
