"""Walking Tours (M15, M16): the videos shrunk to u8 frames, and their clips.

``prepare`` decodes one video of ``MANIFEST`` (one job of ``task=prep``) into
``<out>/<video>/``, a frame directory of ``exp.lib.video_io``: ``fps`` frames
per second of the largest centered square, shrunk to ``size`` x ``size``. Once
every video is there, ``finalize`` writes ``<out>/index.json`` (the videos, and
the sha256 of each video's index) and renames ``<out>`` to its
``data_version``, the first 12 hex digits of the sha256 of that index. The
data of a run is named by its ``data_version`` alone.

``MANIFEST`` records, per video, the source URL, the licenses (of the dataset,
and of the upload), whether faces and plates are blurred, and the size of the
file; ``prepare`` refuses a video that is not in it, is excluded, or whose file
has another size, and ``finalize`` refuses a set of videos that is not every
``ok`` video of it.

Run as ``python -m exp.video.data.walking_tours prepare VIDEO --src DIR --out
DIR`` and ``python -m exp.video.data.walking_tours finalize DIR``.
"""

import argparse
import hashlib
import json
import os
import shutil
from collections.abc import Collection
from contextlib import closing
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from exp.lib.video_io import (
    INDEX,
    U8Reader,
    U8Writer,
    ffmpeg_filter,
    ffmpeg_frames,
    ffmpeg_version,
)

__all__ = [
    "MANIFEST",
    "data_version",
    "eval_clips",
    "finalize",
    "manifest_entry",
    "prepare",
    "videos",
]

MANIFEST = Path(__file__).with_name("walking_tours") / "manifest.json"


def manifest_entry(video: str, manifest: Path = MANIFEST) -> dict[str, Any]:
    """The entry of ``video`` in ``manifest``.

    Raises:
        ValueError: ``video`` is not in the manifest or is excluded.
    """
    entries = json.loads(manifest.read_text())["videos"]
    if video not in entries:
        raise ValueError(
            f"{video!r} is not in {manifest}: record its source, licenses and "
            f"blur there first (known: {', '.join(sorted(entries))})"
        )
    entry = entries[video]
    if entry["status"] != "ok":
        reason = entry.get("reason", "no reason given")
        raise ValueError(f"{video!r} is {entry['status']} in {manifest}: {reason}")
    return entry


def prepare(
    video: str,
    src: Path,
    out: Path,
    *,
    fps: float = 10.0,
    size: int = 64,
    threads: int = 4,
    manifest: Path = MANIFEST,
) -> None:
    """Decode ``video`` (the file of its manifest entry under ``src``) into the
    frame directory ``out/<video>/``: u8 ``(N, 3, size, size)``, ``fps``
    frames per second. Written as ``out/<video>.tmp/`` (a stale one is
    removed), then renamed.

    Raises:
        ValueError: the manifest refuses ``video`` (see ``manifest_entry``),
            or the file's size is not the manifest's.
        FileExistsError: ``out/<video>`` exists.
        RuntimeError: ffmpeg failed or gave no frames.
    """
    entry = manifest_entry(video, manifest)
    path = src / entry["file"]
    if path.stat().st_size != entry["bytes"]:
        raise ValueError(
            f"{path} has {path.stat().st_size} bytes, the manifest says "
            f"{entry['bytes']}: incomplete, or another file"
        )
    final = out / video
    if final.exists():
        raise FileExistsError(final)
    tmp = out / f"{video}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    meta = {
        "video": video,
        "file": entry["file"],
        "url": entry["url"],
        "dataset_license": entry["dataset_license"],
        "upload_license": entry["upload_license"],
        "blur": entry["blur"],
        "source_bytes": entry["bytes"],
        "fps": fps,
        "size": size,
        "filter": ffmpeg_filter(fps=fps, width=size, height=size),
        "ffmpeg": ffmpeg_version(),
    }
    frames = ffmpeg_frames(path, fps=fps, width=size, height=size, threads=threads)
    with closing(frames), U8Writer(tmp, (3, size, size), meta) as w:
        for frame in frames:
            w.append(frame)
        if w.frames == 0:
            raise RuntimeError(f"ffmpeg gave no frames of {path}")
    os.rename(tmp, final)


def data_version(index: Path) -> str:
    """The first 12 hex digits of the sha256 of the file ``index``."""
    return hashlib.sha256(index.read_bytes()).hexdigest()[:12]


def finalize(
    root: Path,
    *,
    expected: Collection[str] | None = None,
    manifest: Path = MANIFEST,
) -> Path:
    """Write ``root/index.json`` over the prepared videos under ``root`` and
    rename ``root`` to ``root.parent / data_version``; returns the new path.

    The index holds ``fps``, ``size`` and, per video in name order, its
    ``frames`` and the sha256 of its ``index.json`` (which holds the sha256 of
    its frames), so the version names the bytes of every frame.

    The videos under ``root`` must be exactly ``expected`` (default: every
    ``ok`` video of ``manifest``, the held-out one included), with no prep
    left running or failed (a ``<video>.tmp``), so a missing video cannot
    silently shorten the data.

    Raises:
        ValueError: a ``.tmp`` is left, the videos are not ``expected``, a
            video's index names another video, or the videos differ in fps
            or size.
        FileExistsError: a directory of that version exists already.
    """
    if expected is None:
        entries_ = json.loads(manifest.read_text())["videos"]
        expected = [v for v, e in entries_.items() if e["status"] == "ok"]
    tmps = sorted(d.name for d in root.glob("*.tmp"))
    if tmps:
        raise ValueError(f"unfinished preps under {root}: {', '.join(tmps)}")
    dirs = sorted(d for d in root.iterdir() if (d / INDEX).is_file())
    found = {d.name for d in dirs}
    if found != set(expected):
        missing = sorted(set(expected) - found)
        extra = sorted(found - set(expected))
        raise ValueError(
            f"the videos under {root} are not the expected ones: "
            f"missing {missing}, unexpected {extra}"
        )
    entries: dict[str, Any] = {}
    settings: set[tuple[float, int]] = set()
    for d in dirs:
        raw = (d / INDEX).read_bytes()
        index = json.loads(raw)
        if index["meta"].get("video") != d.name:
            raise ValueError(
                f"{d / INDEX} is of video {index['meta'].get('video')!r}, "
                f"not {d.name!r}"
            )
        settings.add((index["meta"]["fps"], index["meta"]["size"]))
        entries[d.name] = {
            "frames": index["frames"],
            "index_sha256": hashlib.sha256(raw).hexdigest(),
        }
    if len(settings) != 1:
        raise ValueError(f"the videos under {root} differ in (fps, size): {settings}")
    ((fps, size),) = settings
    top = {"fps": fps, "size": size, "videos": entries}
    (root / INDEX).write_text(json.dumps(top, indent=1, sort_keys=True))
    target = root.parent / data_version(root / INDEX)
    if target == root:
        return root
    if target.exists():
        raise FileExistsError(target)
    os.rename(root, target)
    return target


def videos(root: Path) -> list[str]:
    """The videos of the finalized data at ``root``, in name order."""
    return sorted(json.loads((root / INDEX).read_text())["videos"])


def eval_clips(
    root: Path, *, held_out: str, clips: int, length: int, seed: int
) -> Tensor:
    """``clips`` clips of ``length`` consecutive frames of the held-out video,
    u8 ``(clips, length, 3, size, size)``.

    The video is cut into ``frames // length`` slots of ``length`` frames; the
    clips are the slots of a random subset of ``clips`` of them (drawn by
    ``seed``), in time order, so no two clips share a frame.

    Raises:
        ValueError: ``length`` is not positive, ``held_out`` is not in the
            data, or it has fewer than ``clips`` slots.
    """
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    if held_out not in videos(root):
        raise ValueError(f"{held_out!r} is not in {root} ({videos(root)})")
    reader = U8Reader(root / held_out)
    slots = len(reader) // length
    if clips > slots:
        raise ValueError(
            f"{held_out} has {len(reader)} frames: {slots} clips of {length}, "
            f"not {clips}"
        )
    g = torch.Generator().manual_seed(seed)
    chosen = torch.randperm(slots, generator=g)[:clips].sort().values
    return torch.stack([reader[s * length : (s + 1) * length] for s in chosen.tolist()])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("prepare", help="decode one video into OUT/VIDEO")
    q.add_argument("video")
    q.add_argument("--src", type=Path, default=Path("/datasets/walking_tours"))
    q.add_argument("--out", type=Path, required=True)
    q.add_argument("--fps", type=float, default=10.0)
    q.add_argument("--size", type=int, default=64)
    q.add_argument("--threads", type=int, default=4)
    f = sub.add_parser("finalize", help="index ROOT and rename it to its version")
    f.add_argument("root", type=Path)
    f.add_argument("--expected", nargs="+", help="the videos (default: every ok one)")
    a = p.parse_args()
    if a.cmd == "prepare":
        prepare(a.video, a.src, a.out, fps=a.fps, size=a.size, threads=a.threads)
    else:
        print(finalize(a.root, expected=a.expected))


if __name__ == "__main__":
    main()
