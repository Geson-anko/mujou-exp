r"""The decoder learned afterwards (M15, M16; design 8.5): for each evaluation
checkpoint of a vision or joint run, a decoder from $z$ to the frame, learned by BP
with Adam (``exp.lib.optim.adam``, not ``torch.optim``) while the encoder is
held, decodes the $z$ of the evaluation chunks.

For each ``ckpt/<t>.pt`` of the run (``Parent``), in time order, the vision
is rebuilt from its ``job.json`` (``compose.from_dict``, ``vit.build_vision``)
and the checkpoint loaded into it; it encodes ``frames`` chunks of the run's
stream (``stream_chunks``: chunk ends spread evenly over all the frames of all
the streams of the run, as the run showed them); a ``decoder`` learns the last
frame of each chunk from its $z$ (``fit_decoder``, from the generator of
``seed + DECODER_SEED``, the same for every checkpoint); and the decoder
decodes the evaluation chunks of each split (``evals``: the program gives
those of ``evals.jsonl``). ``Decoding`` does one checkpoint a step and writes
into its directory ``out``

- ``<t>.png``: ``SHOWN`` chunks of the first split (val), the true last
  frames in the first row and their reconstructions in the second;
- ``metrics.jsonl``: a line per checkpoint and split, ``{"t", "frame",
  "split", "recon", "pix_mean", "recon_fit"}``, appended and synced after
  each checkpoint: the mean squared error per pixel of the reconstructions of
  the split's last frames (``recon``), the same of the mean image of those
  frames (``pix_mean``, the bar a decoder of a $z$ that says nothing reaches)
  and of the reconstructions of the frames the decoder learned on
  (``recon_fit``, to see it overfit);
- ``config.json``: the settings of the decoding and the parent run (its
  ``run_id``, ``group`` and ``code``).

On the CPU the files are the same bytes for the same run and settings, and a
``Decoding`` started at checkpoint ``done`` writes those of one that did all.
The task ``posthoc`` of the program (``exp.video.programs.posthoc``) runs it as
a job of the fleet.
"""

import bisect
import dataclasses
import json
import math
import os
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, override

import torch
from torch import Tensor, nn

from exp.lib import ckpt
from exp.lib.optim import adam
from exp.lib.png import tile, write_png
from exp.lib.rundir import atomic_open, write_json
from exp.lib.video_io import U8Reader
from exp.video.compose import from_dict
from exp.video.data.physics import BallStreams, render
from exp.video.data.walking_tours import videos
from exp.video.jepa.vit import VisionModules, build_vision
from exp.video.measure.joint import (
    Windows,
    decode,
    image_rows,
    joint_dynamics,
    latent_errors,
)
from exp.video.schema.run import RunConf

__all__ = [
    "BATCH",
    "DECODER_SEED",
    "FIT_BATCH",
    "FRAMES",
    "LR",
    "SHOWN",
    "STEPS",
    "ChunkSource",
    "Decoding",
    "Parent",
    "decode_run",
    "decoder",
    "fit_decoder",
    "stream_chunks",
]

DECODER_SEED = 3  # added to the run's seed (CHUNKER_SEED 1, HEAD_SEED 2)
SHOWN = 8  # val chunks in a PNG
FRAMES, STEPS = 16384, 2000  # the chunks a decoder learns on, and its steps
BATCH = 256  # chunks encoded or decoded at once
FIT_BATCH, LR = 256, 1e-3  # of the decoder's fit
_BASE = 32  # the channels of the decoder's last transposed convolution


def decoder(d_z: int, shape: tuple[int, int, int]) -> nn.Module:
    """$z$ ``(n, d_z)`` to frames ``(n, C, H, W)`` for ``shape = (C, H, W)``
    with ``H = W`` a power of 2 from 8 (as M11's): a linear map to 4 x 4 maps
    of ``32 * 2^u`` channels, then ``u = log2(H / 4)`` transposed
    convolutions that double the side and halve the channels (ReLU before
    each), then a 3 x 3 convolution to ``C`` channels.

    Raises:
        ValueError: ``H`` is not such a side, or ``W != H``.
    """
    c, h, w = shape
    if h != w or h < 8 or h & (h - 1):
        raise ValueError(f"frames of a side that is a power of 2 from 8, not {h}x{w}")
    ups = int(math.log2(h // 4))
    width = _BASE * 2**ups
    layers: list[nn.Module] = [
        nn.Linear(d_z, width * 16),
        nn.Unflatten(1, (width, 4, 4)),
    ]
    for _ in range(ups):
        layers += [nn.ReLU(), nn.ConvTranspose2d(width, width // 2, 4, 2, 1)]
        width //= 2
    layers += [nn.ReLU(), nn.Conv2d(width, c, 3, padding=1)]
    return nn.Sequential(*layers)


def fit_decoder(
    z: Tensor,
    images: Tensor,
    *,
    steps: int = STEPS,
    batch: int = FIT_BATCH,
    lr: float = LR,
    gen: torch.Generator,
) -> nn.Module:
    r"""A ``decoder`` of ``z`` ``(n, D)`` to ``images`` ``(n, C, H, W)`` (on
    one device), fit by ``steps`` steps of ``adam(lr)`` on the mean squared
    error of ``batch`` pairs drawn with replacement.

    ``gen`` (a CPU generator) draws the seed of the initial weights (built on
    the CPU, so they do not depend on the device; the global RNG is left as
    it was) and then the batches. The gradient is taken by autograd within
    the decoder alone: it is a measurement, outside the network of the run.
    """
    seed = int(torch.randint(2**62, (1,), generator=gen))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        dec = decoder(z.shape[1], (images.shape[1], images.shape[2], images.shape[3]))
    dec = dec.to(z.device)
    params = list(dec.parameters())
    optimizer = adam(lr).bind(params)
    z, images = z.detach(), images.detach()
    for _ in range(steps):
        idx = torch.randint(len(z), (batch,), generator=gen).to(z.device)
        with torch.enable_grad():
            loss = ((dec(z[idx]) - images[idx]) ** 2).mean()
            grads = torch.autograd.grad(loss, params)
        optimizer.step_(grads, 1.0)
    return dec


class ChunkSource(Protocol):
    """Chunks of a run's stream, read in slices: ``source[a:b]`` is ``(b - a,
    N, C, H, W)`` float32 in ``[0, 1]`` on the CPU, as the run's ``Chunker``
    held them."""

    def __len__(self) -> int: ...

    def __getitem__(self, s: slice, /) -> Tensor: ...


def _ends(units: int, streams: int, n: int, frames: int) -> list[tuple[int, int]]:
    """``min(frames, S)`` chunk ends (frame, stream) spread evenly over the
    ``S = (units - n + 1) streams`` ends of full chunks of ``streams`` streams
    of ``units`` frames, in time order: end ``m`` is position ``m S / M``
    (rounded down) of the frames of the streams in (frame, stream) order."""
    total = (units - n + 1) * streams
    if total <= 0:
        raise ValueError(f"a stream of {units} frames has no chunk of {n}")
    m = min(frames, total)
    return [
        (n - 1 + p // streams, p % streams) for p in (j * total // m for j in range(m))
    ]


class _BallChunks(ChunkSource):
    """The chunks of the balls' streams of ``conf`` (``BallFrames`` of the run's
    seed, replayed frame by frame up to the last end) at ``ends``, kept as the
    states ``(M, N, 4K)`` and rendered when read."""

    def __init__(self, conf: RunConf, ends: Sequence[tuple[int, int]]) -> None:
        balls, v = conf.data.balls, conf.vision
        assert balls is not None and v is not None
        self._size, self._radii = balls.size, list(balls.radii)
        streams = BallStreams.of(
            balls, conf.data.streams, frame=conf.time.frame, seed=conf.seed
        )
        ring: deque[Tensor] = deque(maxlen=v.frames)  # (B, 4K) each
        out: list[Tensor] = []
        at = 0
        for k in range(ends[-1][0] + 1):
            ring.append(streams.next())
            while at < len(ends) and ends[at][0] == k:
                out.append(torch.stack(list(ring))[:, ends[at][1]])  # (N, 4K)
                at += 1
        self._states = torch.stack(out)  # (M, N, 4K) float32

    @override
    def __len__(self) -> int:
        return len(self._states)

    @override
    def __getitem__(self, s: slice, /) -> Tensor:
        states = self._states[s]
        images = render(states.flatten(0, 1), self._size, self._radii)
        return images.float().unflatten(0, states.shape[:2])


class _TourChunks(ChunkSource):
    r"""The chunks of the Walking Tours streams of ``conf`` at ``ends``, read
    from the local copy of the data ``root`` when read.

    As ``TourFrames``: the videos but the held-out one are joined in name order
    into a loop of $L$ frames, frame $k$ of stream $i$ is frame
    $(\lfloor i L / B \rfloor + k) \bmod L$ of the loop. A stream resets at the
    first frame of a video, so an end fewer than $N - 1$ frames into its video
    has no full chunk and is left out.
    """

    def __init__(
        self, conf: RunConf, ends: Sequence[tuple[int, int]], root: Path
    ) -> None:
        t, v = conf.data.tours, conf.vision
        assert t is not None and v is not None
        self._readers = [
            U8Reader(root / name) for name in videos(root) if name != t.held_out
        ]
        lengths = [len(r) for r in self._readers]
        firsts = [sum(lengths[:j]) for j in range(len(lengths))]
        loop, b, n = sum(lengths), conf.data.streams, v.frames
        self._n = n
        self._at: list[tuple[int, int]] = []  # (video, frame of the end)
        for k, i in ends:
            g = (i * loop // b + k) % loop
            video = bisect.bisect_right(firsts, g) - 1
            local = g - firsts[video]
            if local >= n - 1:
                self._at.append((video, local))

    @override
    def __len__(self) -> int:
        return len(self._at)

    @override
    def __getitem__(self, s: slice, /) -> Tensor:
        n = self._n
        u8 = [self._readers[v][e - n + 1 : e + 1] for v, e in self._at[s]]
        return torch.stack(u8).float() / 255  # (b, N, 3, size, size)


def stream_chunks(
    conf: RunConf, frames: int, *, root: Path | None = None
) -> ChunkSource:
    """About ``frames`` chunks of the stream of the run ``conf`` (all of its
    ``seconds``), their ends spread evenly over the frames and the streams
    (a few fewer for Walking Tours, ``_TourChunks``, read from the local copy
    of the data ``root``).

    Raises:
        ValueError: The run has no vision, or streams Walking Tours and
            ``root`` is ``None``.
    """
    v = conf.vision
    if v is None:
        raise ValueError(f"the run {conf.name!r} has no vision")
    units = round(conf.seconds / conf.time.frame)
    ends = _ends(units, conf.data.streams, v.frames, frames)
    if conf.data.balls is not None:
        return _BallChunks(conf, ends)
    if root is None:
        raise ValueError(f"the run {conf.name!r} streams Walking Tours: give root")
    return _TourChunks(conf, ends, root)


@torch.no_grad()
def _encode(
    mods: VisionModules, source: ChunkSource | Tensor, device: torch.device
) -> Tensor:
    """$z$ ``(M, D)`` of all the chunks of ``source`` (or of chunks ``(M, N, C,
    H, W)``), ``BATCH`` at a time."""
    zs = [
        mods.encode(source[i : i + BATCH].to(device))
        for i in range(0, len(source), BATCH)
    ]
    return torch.cat(zs)


def _mse(a: Tensor, b: Tensor) -> float:
    return float(((a - b) ** 2).mean())


@dataclass(frozen=True)
class Parent:
    """The vision run whose checkpoints are decoded: its directory, its job,
    its config with ``device`` replaced by that of the decoding, and its
    evaluation checkpoints in time order (``%07.1f`` names sort so)."""

    run_dir: Path
    job: dict[str, Any]
    conf: RunConf
    ckpts: list[Path]

    @classmethod
    def read(cls, run_dir: Path, *, device: str) -> Parent:
        """The run in ``run_dir``, decoded on ``device``.

        Raises:
            ValueError: The run is not of the program ``"vision"`` or ``"joint"``, or has no
                evaluation checkpoint.
        """
        job = json.loads((run_dir / "job.json").read_text())
        conf = dataclasses.replace(from_dict(job["config"]), device=device)
        if conf.program not in ("vision", "joint"):
            raise ValueError(
                f"the run {run_dir} is of {conf.program!r}, not 'vision' or 'joint'"
            )
        ckpts = sorted((run_dir / "ckpt").glob("*.pt"))
        if not ckpts:
            raise ValueError(f"the run {run_dir} has no evaluation checkpoint")
        return cls(run_dir, job, conf, ckpts)


class Decoding:
    """The decoding of the checkpoints of ``parent`` (module docstring) into
    the directory ``out``, a checkpoint per ``step``, from checkpoint
    ``done`` (0: a fresh start, which writes ``config.json`` and empties
    ``metrics.jsonl``; else the lines of the first ``done`` checkpoints are
    kept and the rest cut).

    Args:
        evals: The evaluation chunks ``(n, N, C, H, W)`` in ``[0, 1]`` of each
            split, in the order of the lines; the PNG shows the first.
        root: The local copy of the data, for a run of Walking Tours.
        windows: For a run of the program ``"joint"``, the evaluation windows
            of each split (``measure.joint.Windows``, the keys of ``evals``):
            the dynamics of the checkpoint rolls out from them, the decoder
            decodes the predicted $z$, and the lines also have the image keys
            of ``measure.joint.image_rows`` (the PNG then shows the true frames
            and the decoded rollouts instead).
        frames: The chunks of the stream each decoder learns on.
        steps: The steps of each decoder's fit.
    """

    def __init__(
        self,
        parent: Parent,
        out: Path,
        *,
        evals: Mapping[str, Tensor],
        root: Path | None = None,
        windows: Mapping[str, Windows] | None = None,
        frames: int = FRAMES,
        steps: int = STEPS,
        done: int = 0,
    ) -> None:
        if not 0 <= done <= len(parent.ckpts):
            raise ValueError(f"done {done} of {len(parent.ckpts)} checkpoints")
        conf = self._conf = parent.conf
        self.parent, self.done = parent, done
        self.out, self._steps = out, steps
        self._evals = {s: x.to(conf.device) for s, x in evals.items()}
        self._dev = torch.device(conf.device)
        self._mods = build_vision(conf, seed=conf.seed)
        self._windows = windows
        self._dyn = None if windows is None else joint_dynamics(conf)
        self._source = stream_chunks(conf, frames, root=root)
        self._targets = torch.cat(
            [
                self._source[i : i + BATCH][:, -1]
                for i in range(0, len(self._source), BATCH)
            ]
        ).to(self._dev)  # (M, C, H, W)
        out.mkdir(parents=True, exist_ok=True)
        if done == 0:
            write_json(out / "config.json", self._config(frames))
        metrics = out / "metrics.jsonl"
        kept = metrics.read_text().splitlines(True)[: done * len(evals)] if done else []
        with atomic_open(metrics) as f:
            f.write("".join(kept).encode())

    def _rollouts(
        self, mods: VisionModules, dec: nn.Module, w: Windows
    ) -> tuple[dict[str, float], Tensor]:
        """The image keys and picture of the joint run on the windows ``w``."""
        assert self._dyn is not None and self._conf.dynamics is not None
        z = w.z(mods)
        horizons = self._conf.measure.horizons
        _, pred = latent_errors(
            self._dyn,
            z,
            observed=w.observed,
            horizons=horizons,
            residual=self._conf.dynamics.residual,
        )
        return image_rows(dec, w, z, pred, horizons)

    def _config(self, frames: int) -> dict[str, Any]:
        job = self.parent.job
        return {
            "frames": frames,
            "chunks": len(self._source),
            "steps": self._steps,
            "batch": FIT_BATCH,
            "lr": LR,
            "seed": self._conf.seed + DECODER_SEED,
            "device": self._conf.device,
            "ckpts": [p.name for p in self.parent.ckpts],
            "parent": {
                "run_id": job.get("run_id"),
                "group": job["group"],
                "code": job.get("code"),
            },
        }

    def __len__(self) -> int:
        """The checkpoints of the parent."""
        return len(self.parent.ckpts)

    def step(self) -> list[dict[str, Any]]:
        """Decode checkpoint ``done``, write its PNG and lines, and return the
        lines (a line per split)."""
        path = self.parent.ckpts[self.done]
        mods, dev = self._mods, self._dev
        # The training decoder (``vision.recon`` > 0) is loaded and ignored: the
        # decoder fit here is its own.
        named: dict[str, nn.Module] = dict(mods.named())
        if self._dyn is not None:
            named |= {"step": self._dyn.step, "readout": self._dyn.readout}
        head = ckpt.load_eval(path, named, group=self.parent.job["group"])
        z = _encode(mods, self._source, dev)
        gen = torch.Generator().manual_seed(self._conf.seed + DECODER_SEED)
        dec = fit_decoder(z, self._targets, steps=self._steps, gen=gen)
        recon_fit = _mse(decode(dec, z), self._targets)
        lines: list[dict[str, Any]] = []
        picture: Tensor | None = None  # of the first split
        for split, chunks in self._evals.items():
            x = chunks[:, -1]  # (n, C, H, W)
            x_hat = decode(dec, _encode(mods, chunks, dev))
            lines.append(
                {
                    "t": head["t"],
                    "frame": head["frame"],
                    "split": split,
                    "recon": _mse(x_hat, x),
                    "pix_mean": _mse(x.mean(0, keepdim=True).expand_as(x), x),
                    "recon_fit": recon_fit,
                }
            )
            if self._windows is None:
                frames = torch.stack([x[:SHOWN], x_hat[:SHOWN]])  # (2, SHOWN, C, H, W)
            else:  # (2 SHOWN, 1 + horizons, C, H, W)
                row, frames = self._rollouts(mods, dec, self._windows[split])
                lines[-1] |= row
            picture = frames if picture is None else picture
        assert picture is not None
        grid = picture.cpu()
        scale = max(1, 96 // grid.shape[-1])
        write_png(self.out / f"{path.stem}.png", tile(grid, scale=scale))
        with (self.out / "metrics.jsonl").open("a") as f:
            f.write("".join(json.dumps(line) + "\n" for line in lines))
            f.flush()
            os.fsync(f.fileno())
        self.done += 1
        return lines


def decode_run(
    parent: Parent,
    out: Path,
    *,
    evals: Mapping[str, Tensor],
    root: Path | None = None,
    frames: int = FRAMES,
    steps: int = STEPS,
) -> list[dict[str, Any]]:
    """Decode every evaluation checkpoint of ``parent`` into ``out``
    (``Decoding``, from a fresh start); returns the lines of
    ``metrics.jsonl``."""
    d = Decoding(parent, out, evals=evals, root=root, frames=frames, steps=steps)
    return [line for _ in range(len(d)) for line in d.step()]
