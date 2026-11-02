r"""The real video of stage 3 (PLAN.md, section 6): EgoDex, downscaled.

EgoDex (CC BY-NC-ND) is read from a local copy and shrunk to height 64, then
cropped to the central 64x64. The shrunk data stays on this machine only.

``prepare`` writes, under one directory:

- ``stream.u8``: the training stream, all frames as ``(N, 3, 64, 64)`` uint8
  (C order). It is built from the ``part1`` split: for each task (in name
  order) a *block* of consecutive episodes of about ``block`` seconds, and
  ``passes`` rounds over the tasks; a later round uses episodes the task has
  not shown yet, so the stream returns to the same task and retention can be
  measured.
- ``clips.u8``: evaluation clips from the ``test`` split, ``(T, L, 3, 64, 64)``
  uint8: the first ``L`` frames of one episode of each of up to ``T`` tasks.
- ``index.json``: the episodes (path, task, split, first frame, frames, block,
  pass), the shape of the clips, the settings, and the ffmpeg version.

Run it as ``python -m exp.video.egodex prepare --out datasets/egodex64``.
"""

import argparse
import ctypes
import json
import os
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import imageio_ffmpeg
import torch

from exp.video.balls import Frame

__all__ = [
    "Episode",
    "EgoDexStream",
    "calibration_images",
    "decode",
    "egodex_clips",
    "plan_clips",
    "plan_stream",
    "prepare",
]

SIZE = 64
FPS = 30
ROOT = Path("/datasets/egodex")


@dataclass(frozen=True)
class Episode:
    """One episode of the plan."""

    path: str  # relative to the dataset root
    task: str
    split: str
    frames: int  # according to the metadata
    block: int = -1  # the block of the stream (-1 for a clip)
    round: int = -1  # the pass over the tasks (-1 for a clip)


def _ffmpeg() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def ffmpeg_version() -> str:
    out = subprocess.run([_ffmpeg(), "-version"], capture_output=True, text=True)
    return out.stdout.splitlines()[0]


def decode(path: str | Path, *, frames: int | None = None) -> torch.Tensor:
    """The frames of the video as ``(n, 3, 64, 64)`` uint8: shrunk to height 64
    (area average), then the central 64 columns. ``frames`` keeps the first ones
    only."""
    cmd = [_ffmpeg(), "-v", "error", "-threads", "1", "-i", str(path)]
    if frames is not None:
        cmd += ["-frames:v", str(frames)]
    vf = f"scale=-2:{SIZE}:flags=area,crop={SIZE}:{SIZE}"
    cmd += ["-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    flat = torch.frombuffer(bytearray(out), dtype=torch.uint8)
    return flat.view(-1, SIZE, SIZE, 3).permute(0, 3, 1, 2).contiguous()


def _bytes(x: torch.Tensor) -> bytes:
    """The raw C-order bytes of a contiguous uint8 tensor."""
    x = x.contiguous()
    return ctypes.string_at(x.data_ptr(), x.numel())


def _metadata(root: Path) -> list[dict]:
    with open(root / "video_metadata.json") as f:
        return json.load(f)


def _by_task(meta: list[dict], split: str) -> dict[str, list[dict]]:
    tasks: dict[str, list[dict]] = defaultdict(list)
    for m in meta:
        s, task, _ = m["relative_path"].split("/")
        if s == split:
            tasks[task].append(m)
    return {t: sorted(v, key=lambda m: m["relative_path"]) for t, v in tasks.items()}


def plan_stream(
    meta: list[dict],
    *,
    block: float = 75.0,
    passes: int = 2,
    min_frames: int = SIZE,
    min_episodes: int = 50,
    seed: int = 0,
) -> list[Episode]:
    """The episodes of the training stream, in order.

    Tasks of ``part1`` with fewer than ``min_episodes`` episodes (of at least
    ``min_frames`` frames) are dropped. Each task's episodes are shuffled by
    ``seed``; in each pass every task (in name order) takes the next episodes
    of its order until the block exceeds ``block`` seconds. Raises if a task
    runs out of episodes."""
    tasks = _by_task(meta, "part1")
    g = torch.Generator().manual_seed(seed)
    orders: dict[str, list[dict]] = {}
    for task in sorted(tasks):
        eps = [m for m in tasks[task] if m["frame_count"] >= min_frames]
        if len(eps) < min_episodes:
            continue
        perm = torch.randperm(len(eps), generator=g).tolist()
        orders[task] = [eps[i] for i in perm]
    plan: list[Episode] = []
    cursor = dict.fromkeys(orders, 0)
    n_block = 0
    for r in range(passes):
        for task, order in orders.items():
            total = 0
            while total <= block * FPS:
                if cursor[task] >= len(order):
                    raise ValueError(f"task {task} has no unused episodes left")
                m = order[cursor[task]]
                cursor[task] += 1
                total += m["frame_count"]
                plan.append(
                    Episode(
                        m["relative_path"], task, "part1", m["frame_count"], n_block, r
                    )
                )
            n_block += 1
    return plan


def plan_clips(
    meta: list[dict], *, tasks: int = 48, length: int = 128, seed: int = 0
) -> list[Episode]:
    """One episode of at least ``length`` frames from each of the first
    ``tasks`` tasks (in name order) of ``test`` that have one, chosen by
    ``seed``."""
    g = torch.Generator().manual_seed(seed)
    plan: list[Episode] = []
    for task, eps in sorted(_by_task(meta, "test").items()):
        long = [m for m in eps if m["frame_count"] >= length]
        if not long or len(plan) == tasks:
            continue
        m = long[int(torch.randint(len(long), (), generator=g))]
        plan.append(Episode(m["relative_path"], task, "test", m["frame_count"]))
    return plan


def prepare(
    out: str | Path,
    *,
    root: str | Path = ROOT,
    jobs: int = 16,
    block: float = 75.0,
    passes: int = 2,
    min_episodes: int = 50,
    clips: int = 48,
    length: int = 128,
    seed: int = 0,
) -> dict:
    """Decode the plan into ``out`` (see the module docstring); returns the
    index. Files are written under temporary names, then renamed."""
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    meta = _metadata(root)
    stream_plan = plan_stream(
        meta, block=block, passes=passes, min_episodes=min_episodes, seed=seed
    )
    clip_plan = plan_clips(meta, tasks=clips, length=length, seed=seed)
    start = time.time()

    episodes: list[dict] = []
    start_frame = 0
    with ThreadPoolExecutor(jobs) as pool, open(out / "stream.u8.tmp", "wb") as f:
        results = pool.map(lambda e: decode(root / e.path), stream_plan)
        for e, x in zip(stream_plan, results):
            if len(x) != e.frames:
                print(f"{e.path}: decoded {len(x)} frames, metadata {e.frames}")
            f.write(_bytes(x))
            episodes.append(
                asdict(e)
                | {"start": start_frame, "frames": len(x), "metadata_frames": e.frames}
            )
            start_frame += len(x)
    clip_x = []
    with ThreadPoolExecutor(jobs) as pool:
        for e, x in zip(
            clip_plan,
            pool.map(lambda e: decode(root / e.path, frames=length), clip_plan),
        ):
            if len(x) != length:
                raise RuntimeError(f"{e.path}: decoded {len(x)} < {length} frames")
            clip_x.append(x)
    clip_t = torch.stack(clip_x)
    (out / "clips.u8.tmp").write_bytes(_bytes(clip_t))

    index = {
        "episodes": episodes,
        "frames": start_frame,
        "clips": {
            "shape": list(clip_t.shape),
            "episodes": [asdict(e) for e in clip_plan],
        },
        "settings": {
            "block": block,
            "passes": passes,
            "min_episodes": min_episodes,
            "clips": clips,
            "length": length,
            "seed": seed,
            "size": SIZE,
            "fps": FPS,
            "root": str(root),
        },
        "ffmpeg": ffmpeg_version(),
        "seconds": time.time() - start,
    }
    (out / "index.json.tmp").write_text(json.dumps(index, indent=1))
    for name in ("stream.u8", "clips.u8", "index.json"):
        os.replace(out / f"{name}.tmp", out / name)
    return index


def _index(path: str | Path) -> dict:
    return json.loads((Path(path) / "index.json").read_text())


class EgoDexStream:
    """The stream of frames, one per ``next()``, memory-mapped from ``path``.

    Args:
        path: The directory ``prepare`` wrote.
        seconds: Stop after this many seconds of frames (``None``: the end).
        offset: The first frame.

    ``next()`` raises ``IndexError`` at the end. ``reset`` is ``True`` at the
    first frame of an episode; ``state`` and ``params`` are empty.
    """

    def __init__(
        self, path: str | Path, *, seconds: float | None = None, offset: int = 0
    ) -> None:
        index = _index(path)
        self.episodes: list[dict] = index["episodes"]
        self.frames: int = index["frames"]
        data = torch.from_file(
            str(Path(path) / "stream.u8"),
            shared=False,
            size=self.frames * 3 * SIZE * SIZE,
            dtype=torch.uint8,
        )
        self._data = data.view(self.frames, 3, SIZE, SIZE)
        self._starts = torch.tensor([e["start"] for e in self.episodes])
        self._firsts = set(self._starts.tolist())
        self.k = offset  # the frame to come
        end = self.frames if seconds is None else offset + round(seconds * FPS)
        self._end = min(end, self.frames)

    def _episode(self, k: int) -> dict:
        i = int(torch.searchsorted(self._starts, k, right=True)) - 1
        return self.episodes[i]

    def task_of(self, k: int) -> str:
        return self._episode(k)["task"]

    def pass_of(self, k: int) -> int:
        return self._episode(k)["round"]

    def next(self) -> Frame:
        if self.k >= self._end:
            raise IndexError("the stream is exhausted")
        image = self._data[self.k].float() / 255
        reset = self.k in self._firsts
        self.k += 1
        return Frame(image, torch.zeros(0), torch.zeros(0), reset)


def egodex_clips(path: str | Path) -> Frame:
    """The evaluation clips: ``image`` is ``(1, T, L, 3, 64, 64)`` float in
    [0, 1] (the leading 1 is the phase, as in ``balls.eval_clips``); ``state``
    and ``params`` are ``(1, T, L, 0)``; ``reset`` is ``False``."""
    shape = _index(path)["clips"]["shape"]
    x = torch.from_file(
        str(Path(path) / "clips.u8"),
        shared=False,
        size=int(torch.Size(shape).numel()),
        dtype=torch.uint8,
    ).view(shape)
    t, length = shape[:2]
    empty = torch.zeros(1, t, length, 0)
    return Frame(x.float().div(255).unsqueeze(0), empty, empty.clone(), False)


def calibration_images(path: str | Path, n: int = 256, seed: int = 0) -> torch.Tensor:
    """``n`` frames drawn uniformly (without replacement) from the first half of
    the stream, ``(n, 3, 64, 64)`` float in [0, 1]."""
    stream = EgoDexStream(path)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(stream.frames // 2, generator=g)[:n].sort().values
    return stream._data[idx].float() / 255


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("prepare", help="downscale EgoDex into the stream and clips")
    q.add_argument("--out", default="datasets/egodex64")
    q.add_argument("--root", default=str(ROOT))
    q.add_argument("--jobs", type=int, default=16)
    q.add_argument("--block", type=float, default=75.0)
    q.add_argument("--passes", type=int, default=2)
    q.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    index = prepare(
        a.out,
        root=a.root,
        jobs=a.jobs,
        block=a.block,
        passes=a.passes,
        seed=a.seed,
    )
    tasks = {e["task"] for e in index["episodes"]}
    n = index["frames"]
    print(
        f"{n} frames ({n / FPS / 60:.1f} min), {len(tasks)} tasks, "
        f"clips {index['clips']['shape']}, {index['seconds']:.0f} s"
    )


if __name__ == "__main__":
    main()
