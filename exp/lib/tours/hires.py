r"""The prepared Walking Tours' square at a higher resolution, at chosen
frames only (V18c's main measure, ``exp/vision/plans/jepa.md``, 「高解像度の
フレーム」): data for evaluation only, the prepared data (its
``data_version``) left as it is.

The prepared frames are the largest centered square of a video (``video_io.
ffmpeg_filter``): for a source of $W$ x $c$ ($W \ge c$) the columns $[x_0,
x_0 + c)$, $x_0 = (W - c) / 2$, shrunk by the area average to the data's
size $s$. Here the same square is turned into RGB first and then shrunk to
``size`` $= f s$ ($f$ a whole number, 5 for 320 of 64; ``hires_filter``).
The reference of the check (a) is the same square shrunk the same way to
$s$ (``hires_filter`` of ``size`` $s$): the area average of $c / s$ source
pixels a side is the mean of the $f \times f$ area averages of $c / (f s)$
pixels whose bounds fall on its own, so the $f \times f$ means of a frame
here are the reference up to the u8 rounding of each (at most 1/2 a level,
less on average). The prepared data shrink the 4:2:0 YUV before turning it
into RGB, so their colours differ from both by the conversion's order (2 to
3 levels on average on the three videos; README 「V18c: 作り直した主の指標の
有効性」): the data are checked only relatively, (b) and (c).

``prepare`` decodes a video, keeps the frames of a list (``frames.json``
of ``python -m exp.vision.sem_eval frames``) into ``<out>/<video>/``
(``video_io.U8Writer``; ``frames.json`` there: the numbers, the largest
$d$ of (a) and each kept frame's checks, ``CHECK_LOG``), and checks the
place of every kept frame $f$ before it is written (``position_check``;
else it stops, writing no index). With
$\bar x$ the frame's $f \times f$ means (unrounded), $r_f$ the reference,
$x_g$ the prepared data, and $d(u, v)$ the mean absolute difference:

- (a) $d(\bar x, r_f) \le$ ``MAX_DIFF`` (1.0 levels);
- (b) for each neighbour $g = f \pm 1$ that the video moves to
  ($d(x_f, x_g) >$ ``MOVING`` = 2 levels), $d(\bar x, x_f) < d(\bar x,
  x_g)$: a frame one off in time is told apart;
- (c) for each shift $\sigma$ of ``SHIFTS`` (1 pixel up, down, left or
  right at the frame's size before the means, and at the data's size after
  them) that changes the means ($d(\bar x^\sigma, \bar x) >$ ``MOVING`` on
  the interior), $d(\bar x, x_f) < d(\bar x^\sigma, x_f)$ on the interior
  (``BORDER`` data pixels off each edge, where the roll wraps): a place
  one pixel off is told apart. A shift that leaves the means as they are
  cannot be told apart, as a neighbour that does not move in (b).

Once the videos are there, ``finalize`` writes ``<out>/index.json`` (each
video's frame numbers, the checks in brief and the sha256 of its index; the
source data, fps, size and filter) and renames ``<out>`` to ``hires-<12 hex>`` of that index.
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
    "BORDER",
    "MAX_DIFF",
    "MOVING",
    "PREFIX",
    "SHIFTS",
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
MOVING = 2.0  # (b), (c): levels of u8 that make a neighbour or a shift tell
SHIFTS = (  # (c): (where, dy, dx); "big" before the means, "small" after
    *(("big", dy, dx) for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))),
    *(("small", dy, dx) for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))),
)
BORDER = 2  # (c): data pixels off each edge, past the rolls' wrap
CHECK_LOG = ("diff", "data_diff", "b_margin", "c_margin", "c_margin_all", "c_compared")


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


def hires_filter(
    geometry: dict[str, int], *, fps: float, size: int | None = None
) -> str:
    """The ffmpeg filter of ``geometry`` (``hires_geometry``): ``fps``
    frames per second, the square cropped, turned into RGB and shrunk by the
    area average to ``size`` (default the geometry's; the data's size for
    the reference of (a))."""
    c, x0 = geometry["side"], geometry["x0"]
    s = geometry["size"] if size is None else size
    return f"fps={fps:g},crop={c}:{c}:{x0}:0,format=rgb24,scale={s}:{s}:flags=area"


def _diff(a: Tensor, b: Tensor) -> float:
    return float((a.double() - b.double()).abs().mean())


def position_check(
    frame: Tensor, f: int, ref: U8Reader, reference: Tensor
) -> dict[str, Any]:
    r"""The checks (a), (b) and (c) (module docstring) of the frame u8 ``(3,
    size, size)`` as frame ``f`` of the prepared frames ``ref``, of the
    reference u8 ``(3, s, s)`` (``hires_filter`` at the data's size):
    ``diff`` $d(\bar x, r_f)$, ``data_diff`` $d(\bar x, x_f)$,
    ``neighbours`` $d(\bar x, x_g)$ of the moving neighbours $g$ (by frame),
    ``shifts`` (by ``where,dy,dx``) each shift's ``change`` $d(\bar
    x^\sigma, \bar x)$ and ``margin`` $d(\bar x^\sigma, x_f) - d(\bar x,
    x_f)$ on the interior ((c) reads the margins of the changes over
    ``MOVING``); ``a``, ``b`` and ``c`` whether each holds.

    Raises:
        ValueError: ``size`` is not a whole multiple of the prepared size.
    """
    s = ref.shape[-1]
    k = frame.shape[-1] // s
    if k * s != frame.shape[-1]:
        raise ValueError(f"a frame of {frame.shape[-1]} is not a multiple of {s}")
    big = frame.double()[None]
    down = F.avg_pool2d(big, k)[0]
    x = ref[f]
    d_data = _diff(down, x)
    moving = {
        g: _diff(down, ref[g])
        for g in (f - 1, f + 1)
        if 0 <= g < len(ref) and _diff(x, ref[g]) > MOVING
    }
    inner = (..., slice(BORDER, -BORDER), slice(BORDER, -BORDER))
    d0 = _diff(down[inner], x[inner])
    shifts = {}
    for where, dy, dx in SHIFTS:
        if where == "big":
            moved = F.avg_pool2d(torch.roll(big, (dy, dx), (2, 3)), k)[0]
        else:
            moved = torch.roll(down, (dy, dx), (1, 2))
        shifts[f"{where},{dy},{dx}"] = {
            "change": _diff(moved[inner], down[inner]),
            "margin": _diff(moved[inner], x[inner]) - d0,
        }
    d = _diff(down, reference)
    return {
        "diff": d,
        "data_diff": d_data,
        "neighbours": moving,
        "shifts": shifts,
        "a": d <= MAX_DIFF,
        "b": all(d_data < dg for dg in moving.values()),
        "c": all(v["margin"] > 0 for v in shifts.values() if v["change"] > MOVING),
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
            frame fails (a), (b) or (c).
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
    small = ref.shape[-1]
    vf_ref = hires_filter(geometry, fps=fps, size=small)
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
        "reference_filter": vf_ref,
        "ffmpeg": ffmpeg_version(),
    }
    wanted = set(keep)
    total = 0
    log: dict[str, list[Any]] = {k: [] for k in CHECK_LOG}
    decoded = ffmpeg_frames(
        path, fps=fps, width=size, height=size, threads=threads, vf=vf
    )
    refs = ffmpeg_frames(
        path, fps=fps, width=small, height=small, threads=threads, vf=vf_ref
    )
    with closing(decoded), closing(refs), U8Writer(tmp, (3, size, size), meta) as w:
        for i, (frame, reference) in enumerate(zip(decoded, refs, strict=True)):
            total = i + 1
            if i not in wanted:
                continue
            check = position_check(frame, i, ref, reference)
            if not (check["a"] and check["b"] and check["c"]):
                failed = [c for c in ("a", "b", "c") if not check[c]]
                raise ValueError(
                    f"frame {i} of {video} fails the position check "
                    f"{', '.join(f'({c})' for c in failed)}: d(f) {check['diff']:.3f}, "
                    f"to the data {check['data_diff']:.3f}, moving neighbours "
                    f"{check['neighbours']}, shifts {check['shifts']}"
                )
            _log(log, check)
            w.append(frame)
        if total != len(ref):
            raise ValueError(f"{video}: {total} frames, the prepared data {len(ref)}")
        if w.frames != len(keep):
            raise ValueError(f"{video}: {w.frames} of {len(keep)} frames kept")
    worst_a = max(log["diff"], default=0.0)
    (tmp / FRAME_LIST).write_text(
        json.dumps({"frames": keep, "max_diff": worst_a, "checks": log})
    )
    os.rename(tmp, final)
    return final


def _log(log: dict[str, list[Any]], check: dict[str, Any]) -> None:
    r"""Append a kept frame's checks to ``log`` (``CHECK_LOG``): (a)'s
    ``diff``, ``data_diff``, (b)'s smallest margin $d(\bar x, x_g) - d(\bar
    x, x_f)$ (None without a moving neighbour), (c)'s smallest margin over
    the shifts it reads (None without one) and over all shifts, and the
    number of shifts it reads."""
    d = check["data_diff"]
    near = [v - d for v in check["neighbours"].values()]
    read = [v["margin"] for v in check["shifts"].values() if v["change"] > MOVING]
    log["diff"].append(check["diff"])
    log["data_diff"].append(d)
    log["b_margin"].append(min(near, default=None))
    log["c_margin"].append(min(read, default=None))
    log["c_margin_all"].append(min(v["margin"] for v in check["shifts"].values()))
    log["c_compared"].append(len(read))


def _summary(log: dict[str, list[Any]] | None) -> dict[str, Any] | None:
    """The kept frames' checks (``CHECK_LOG``) in brief: (a)'s mean,
    ``data_diff``'s mean and largest, (b)'s and (c)'s smallest margins and
    the frames each read, (c)'s smallest margin over all shifts."""
    if log is None:
        return None
    n = len(log["diff"])
    b = [m for m in log["b_margin"] if m is not None]
    c = [m for m in log["c_margin"] if m is not None]
    return {
        "frames": n,
        "diff_mean": sum(log["diff"]) / n if n else None,
        "data_diff_mean": sum(log["data_diff"]) / n if n else None,
        "data_diff_max": max(log["data_diff"], default=None),
        "b_margin_min": min(b, default=None),
        "b_frames": len(b),
        "c_margin_min": min(c, default=None),
        "c_frames": len(c),
        "c_margin_all_min": min(log["c_margin_all"], default=None),
    }


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
            "checks": _summary(kept.get("checks")),
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
