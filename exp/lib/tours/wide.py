r"""The bands beside the prepared Walking Tours' square (V18c's E2,
``exp/vision/plans/jepa.md``, 「E2: 画面の外」): data for evaluation only, the
prepared data (its ``data_version``) left as it is.

The prepared frames are the largest centered square of a video (``video_io.
ffmpeg_filter``): for a source of $W$ x $c$ ($W > c$) the columns $[x_0, x_0
+ c)$, $x_0 = (W - c) / 2$, shrunk by the area average to ``size`` x
``size``. At the same scale a band of ``band`` output columns is $b = $
``band`` $\cdot c /$ ``size`` source columns (180 of 480 for 24 of 64). A
frame here is u8 ``(3, size, band + size + band)``: the columns $[x_0 - b,
x_0)$ shrunk to ``band`` (the left band), the square as the prepared data
has it, and $[x_0 + c, x_0 + c + b)$ shrunk to ``band`` (the right band),
by one ffmpeg filter (``band_filter``) so that the three are of the same
source frame.

``prepare`` decodes a video into ``<out>/<video>/`` (``video_io.U8Writer``)
and checks that its middle columns are the prepared data's frames, byte for
byte, as many as they are (else it writes nothing). Once the videos are
there, ``finalize`` writes ``<out>/index.json`` (the videos and the sha256 of
each one's index, the source data, the geometry) and renames ``<out>`` to
``wide-<12 hex>`` of that index::

    python -m exp.lib.tours.wide prepare <video> --src /datasets/walking_tours \
        --data $MUJOU_EXP_DATA/<data_version> --out <dir>
    python -m exp.lib.tours.wide finalize <dir>
"""

import argparse
import hashlib
import json
import os
import shutil
from contextlib import closing
from pathlib import Path
from typing import Any

import torch

from exp.lib.tours.walking_tours import MANIFEST, data_version, manifest_entry
from exp.lib.video_io import INDEX, U8Reader, U8Writer, ffmpeg_frames, ffmpeg_version

__all__ = ["BAND", "PREFIX", "band_filter", "band_geometry", "finalize", "prepare"]

BAND = 24  # output columns of a band at 64 px: 180 source columns of 480
PREFIX = "wide-"


def band_geometry(
    width: int, height: int, *, size: int, band: int = BAND
) -> dict[str, int]:
    r"""The source columns of a ``width`` x ``height`` video (module
    docstring): ``side`` $c$, ``x0`` $x_0$, ``band_src`` $b$, the bands'
    first columns ``left`` $x_0 - b$ and ``right`` $x_0 + c$, and ``size``
    and ``band``.

    Raises:
        ValueError: The square is not centered on a whole column, $b$ is not
            a whole number of columns, or the video is too narrow for its
            bands ($x_0 < b$).
    """
    c = height
    if width < c or (width - c) % 2:
        raise ValueError(f"a {width} x {height} video has no square on whole columns")
    if (band * c) % size:
        raise ValueError(f"a band of {band} of {size} is not whole columns of {c}")
    x0, b = (width - c) // 2, band * c // size
    if x0 < b:
        raise ValueError(
            f"a {width} x {height} video has {x0} columns beside its square, "
            f"fewer than a band's {b}"
        )
    return {
        "side": c,
        "x0": x0,
        "band_src": b,
        "left": x0 - b,
        "right": x0 + c,
        "size": size,
        "band": band,
    }


def band_filter(geometry: dict[str, int], *, fps: float) -> str:
    """The ffmpeg filter of ``geometry`` (``band_geometry``): ``fps`` frames
    per second, split into the left band, the square and the right band,
    each cropped and shrunk by the area average, side by side."""
    g = geometry
    c, b, size, band = g["side"], g["band_src"], g["size"], g["band"]
    left = f"crop={b}:{c}:{g['left']}:0,scale={band}:{size}:flags=area"
    mid = f"crop={c}:{c}:{g['x0']}:0,scale={size}:{size}:flags=area"
    right = f"crop={b}:{c}:{g['right']}:0,scale={band}:{size}:flags=area"
    return (
        f"fps={fps},split=3[a][b][c];[a]{left}[l];[b]{mid}[m];[c]{right}[r];"
        "[l][m][r]hstack=inputs=3"
    )


def prepare(
    video: str,
    src: Path,
    data: Path,
    out: Path,
    *,
    band: int = BAND,
    threads: int = 4,
    manifest: Path = MANIFEST,
) -> Path:
    """Decode ``video`` (its manifest entry's file under ``src``) into
    ``out/<video>/`` (module docstring), checked against the prepared data
    ``data`` (``data/<video>``, its fps and size). Written as
    ``out/<video>.tmp/`` (a stale one removed), then renamed. Returns the
    directory.

    Raises:
        ValueError: The manifest refuses ``video``, the file's size is not
            the manifest's, the geometry is refused (``band_geometry``), or
            the middle columns are not the prepared frames.
        FileExistsError: ``out/<video>`` exists.
    """
    entry = manifest_entry(video, manifest)
    path = src / entry["file"]
    if path.stat().st_size != entry["bytes"]:
        raise ValueError(
            f"{path} has {path.stat().st_size} bytes, the manifest says {entry['bytes']}"
        )
    ref = U8Reader(data / video)
    fps, size = ref.index["meta"]["fps"], ref.index["meta"]["size"]
    geometry = band_geometry(entry["width"], entry["height"], size=size, band=band)
    vf = band_filter(geometry, fps=fps)
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
    width = size + 2 * band
    frames = ffmpeg_frames(
        path, fps=fps, width=width, height=size, threads=threads, vf=vf
    )
    with closing(frames), U8Writer(tmp, (3, size, width), meta) as w:
        for i, frame in enumerate(frames):
            if i >= len(ref) or not torch.equal(
                frame[:, :, band : band + size], ref[i]
            ):
                raise ValueError(
                    f"frame {i} of {video}: its middle columns are not the "
                    f"prepared data's of {data} ({len(ref)} frames)"
                )
            w.append(frame)
        if w.frames != len(ref):
            raise ValueError(
                f"{video}: {w.frames} frames, the prepared data {len(ref)}"
            )
    os.rename(tmp, final)
    return final


def finalize(root: Path) -> Path:
    """Write ``root/index.json`` over the videos under ``root`` (module
    docstring) and rename ``root`` to ``wide-<12 hex>`` of it beside it.
    Returns the new path.

    Raises:
        ValueError: A ``.tmp`` is left, there is no video, a video's index
            names another video, or the videos differ in their source data,
            fps or geometry.
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
        keys = ("data", "data_index_sha256", "fps", "geometry")
        shared.add(json.dumps({k: meta[k] for k in keys}, sort_keys=True))
        entries[d.name] = {
            "frames": json.loads(raw)["frames"],
            "index_sha256": hashlib.sha256(raw).hexdigest(),
        }
    if len(shared) != 1:
        raise ValueError(f"the videos under {root} differ in their data or geometry")
    top = json.loads(shared.pop()) | {"videos": entries}
    (root / INDEX).write_text(json.dumps(top, indent=1, sort_keys=True))
    target = root.parent / f"{PREFIX}{data_version(root / INDEX)}"
    if target == root:
        return root
    if target.exists():
        raise FileExistsError(target)
    os.rename(root, target)
    return target


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("prepare", help="decode one video into OUT/VIDEO")
    q.add_argument("video")
    q.add_argument("--src", type=Path, default=Path("/datasets/walking_tours"))
    q.add_argument("--data", type=Path, required=True, help="the prepared data")
    q.add_argument("--out", type=Path, required=True)
    q.add_argument("--band", type=int, default=BAND)
    q.add_argument("--threads", type=int, default=4)
    f = sub.add_parser("finalize", help="index OUT and rename it to its version")
    f.add_argument("root", type=Path)
    args = p.parse_args()
    if args.cmd == "prepare":
        print(
            prepare(
                args.video,
                args.src,
                args.data,
                args.out,
                band=args.band,
                threads=args.threads,
            )
        )
    else:
        print(finalize(args.root))


if __name__ == "__main__":
    main()
