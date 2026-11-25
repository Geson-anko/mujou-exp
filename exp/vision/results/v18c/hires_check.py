r"""Why the hires frames of V18c's main measure fail the position check (a)
(exp/vision/README.md, "V18c: 作り直した主の指標の有効性（2026-11-25）"): the
check of ``exp.lib.tours.hires`` over every frame ``sem_eval frames`` lists,
with nothing written, and its parts.

For one video, decodes the hires frames as ``hires.prepare`` did then
(``yuv_filter``: the 4:2:0 YUV shrunk before RGB; ``hires_filter`` turns the
square into RGB first since, README 「V18c: 作り直した主の指標の有効性」) and, for each listed frame $f$, the check's $d(f)$ (the
mean absolute difference of the frame's 5 x 5 means, unrounded, and the
prepared frame $x_f$), $d(f \pm 1)$ of the moving neighbours (b), the mean
signed difference per channel and the difference of the luma (BT.601
weights). On the first ``--first`` frames it also decomposes $d$:

- ``fresh64``: the prepared filter decoded again against the data (0 if the
  decoder is the data's, bit for bit);
- ``rgb``: both sizes scaled in RGB (``format=rgb24`` before ``scale``), the
  5 x 5 means of 320 against 64: the plan's theory says they differ by the
  u8 rounding only;
- ``rgb64_data``: the RGB-scaled 64 against the data (the part of the
  colour conversion's order in the data itself);
- ``shift``: $d$ of the hires frames rolled by $(dy, dx)$ of $-5..5$ pixels
  at 320 before the means, the interior only; the place is right if the
  smallest is at $(0, 0)$.

Reads no target and no code. Runs where the sources and the prepared data
are (a host's container)::

    python -m exp.vision.results.v18c.hires_check <video> --frames frames.json \
        --data $MUJOU_EXP_DATA/ad6e56b5995e --out <dir> [--first 60]
"""

import argparse
import json
from collections.abc import Generator
from contextlib import closing
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from exp.lib.tours.hires import MAX_DIFF, MOVING, hires_geometry
from exp.lib.tours.walking_tours import MANIFEST, manifest_entry
from exp.lib.video_io import U8Reader, ffmpeg_frames

FACTOR = 5  # 320 / 64
LUMA = (0.299, 0.587, 0.114)  # BT.601
SHIFTS = range(-5, 6)


def yuv_filter(geometry: dict[str, int], *, fps: float) -> str:
    """The hires filter of the run this script diagnosed (module docstring)."""
    c, x0, size = geometry["side"], geometry["x0"], geometry["size"]
    return f"fps={fps:g},crop={c}:{c}:{x0}:0,scale={size}:{size}:flags=area"


def _d(a: Tensor, b: Tensor) -> float:
    return float((a.double() - b.double()).abs().mean())


def _luma(x: Tensor) -> Tensor:
    w = torch.tensor(LUMA, dtype=torch.float64).view(3, 1, 1)
    return (x.double() * w).sum(-3)


def _decode(
    path: Path, vf: str, size: int, fps: float, n: int | None
) -> Generator[Tensor]:
    duration = None if n is None else (n + 0.5) / fps
    return ffmpeg_frames(
        path, fps=fps, width=size, height=size, vf=vf, duration=duration
    )


def _parts(
    path: Path, ref: U8Reader, geometry: dict[str, int], fps: float, n: int
) -> dict[str, Any]:
    """The decomposition of $d$ on the first ``n`` frames (module docstring)."""
    c, x0 = geometry["side"], geometry["x0"]
    base = f"fps={fps:g},crop={c}:{c}:{x0}:0"
    small = ref.shape[-1]

    def take(vf: str, size: int) -> Tensor:
        with closing(_decode(path, vf, size, fps, n)) as it:
            return torch.stack(list(it))[:n].double()

    data = torch.stack([ref[i] for i in range(n)]).double()
    fresh = take(f"{base},scale={small}:{small}:flags=area", small)
    hires = take(yuv_filter(geometry, fps=fps), geometry["size"])
    rgb_big = take(
        f"{base},format=rgb24,scale={geometry['size']}:{geometry['size']}:flags=area",
        geometry["size"],
    )
    rgb_small = take(f"{base},format=rgb24,scale={small}:{small}:flags=area", small)
    mean = F.avg_pool2d(hires, FACTOR)
    inner = data[..., 2:-2, 2:-2]
    shift = {}
    for dy in SHIFTS:
        for dx in SHIFTS:
            rolled = F.avg_pool2d(torch.roll(hires, (dy, dx), (2, 3)), FACTOR)
            shift[f"{dy},{dx}"] = _d(rolled[..., 2:-2, 2:-2], inner)
    best = min(shift, key=lambda k: shift[k])
    return {
        "frames": n,
        "check": _d(mean, data),
        "fresh64": _d(fresh, data),
        "rgb": _d(F.avg_pool2d(rgb_big, FACTOR), rgb_small),
        "rgb64_data": _d(rgb_small, data),
        "shift_best": best,
        "shift": {
            k: shift[k] for k in ("0,0", "0,1", "1,0", "0,-1", "-1,0", "0,5", "5,0")
        },
    }


def check(
    video: str, frames: list[int], data: Path, src: Path, first: int
) -> dict[str, Any]:
    """The check of every listed frame of ``video`` and the parts (module
    docstring)."""
    entry = manifest_entry(video, MANIFEST)
    path = src / entry["file"]
    ref = U8Reader(data / video)
    fps = ref.index["meta"]["fps"]
    geometry = hires_geometry(entry["width"], entry["height"])
    wanted = set(frames)
    rows: list[dict[str, Any]] = []
    with closing(
        _decode(path, yuv_filter(geometry, fps=fps), geometry["size"], fps, None)
    ) as it:
        for i, frame in enumerate(it):
            if i not in wanted:
                continue
            mean = F.avg_pool2d(frame.double()[None], FACTOR)[0]
            x = ref[i].double()
            near = {
                g: _d(mean, ref[g])
                for g in (i - 1, i + 1)
                if 0 <= g < len(ref) and _d(x, ref[g]) > MOVING
            }
            d = _d(mean, x)
            rows.append({
                "f": i,
                "d": d,
                "near_min": min(near.values()) if near else None,
                "b": all(d < v for v in near.values()),
                "signed": (mean - x).mean((1, 2)).tolist(),
                "luma": _d(_luma(mean), _luma(x)),
            })  # fmt: skip
    ds = torch.tensor([r["d"] for r in rows], dtype=torch.float64)
    lum = torch.tensor([r["luma"] for r in rows], dtype=torch.float64)
    signed = torch.tensor([r["signed"] for r in rows], dtype=torch.float64)
    return {
        "video": video,
        "frames": len(rows),
        "listed": len(wanted),
        "max_diff": MAX_DIFF,
        "a_fail": int((ds > MAX_DIFF).sum()),
        "b_fail": sum(not r["b"] for r in rows),
        "d": {
            "mean": float(ds.mean()),
            "median": float(ds.median()),
            "q95": float(ds.quantile(0.95)),
            "max": float(ds.max()),
        },
        "luma": {"mean": float(lum.mean()), "max": float(lum.max())},
        "signed_mean": signed.mean(0).tolist(),
        "near_min_min": min(r["near_min"] for r in rows if r["near_min"] is not None),
        "parts": _parts(path, ref, geometry, fps, first),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video")
    ap.add_argument("--frames", type=Path, required=True, help="sem_eval frames' list")
    ap.add_argument("--data", type=Path, required=True, help="the prepared data")
    ap.add_argument("--src", type=Path, default=Path("/datasets/walking_tours"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--first", type=int, default=60)
    args = ap.parse_args()
    frames = json.loads(args.frames.read_text())[args.video]
    got = check(args.video, frames, args.data, args.src, args.first)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.video}.json").write_text(json.dumps(got, indent=1))


if __name__ == "__main__":
    main()
