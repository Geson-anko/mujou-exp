"""The latent space seen as frames (README: M11, the reconstruction): from
each checkpoint of a run, a decoder from $z$ to the frame, learned by BP with
Adam on frames of the run's stream while the encoder is held, decodes the $z$
of the evaluation clips and the rollouts from them.

    uv run python -m exp.video.decode <run dir> [--steps 2000] [--device cuda:0]

For each checkpoint ``ckpt/<seconds>.pt`` of the run (``checkpoint_interval``),
writes ``decode/<seconds>.png`` and a line of ``decode/metrics.jsonl``, with the
settings of the decoding and the commits in ``decode/config.json``. The PNG
has, for a few clips, the rows: the frames, their $z$ decoded, the rollout
decoded and the shrunk last $z$ (``evaluate.shrink_predictions``) decoded, at
the last observed frame and ``COLUMNS`` frames after it. The line has the mean
squared error per pixel of the decoded frames over all the clips: ``recon``
(the $z$ of the frames), and ``img_rollout{n}``, ``img_persist{n}``,
``img_mean{n}``, ``img_shrink{n}`` (the predictions $n$ frames ahead), and the
bars without the decoder, ``pix_copy{n}`` (the last observed frame again) and
``pix_mean{n}`` (the mean of the frames the decoder learned on); and the
metrics of ``evaluate`` of the checkpoint.
"""

import argparse
import json
import math
import struct
import zlib
from pathlib import Path

import torch
from torch import nn

from exp.lib.record import REPO_ROOT, git_dirty, git_sha
from exp.video.balls import Frame
from exp.video.config import Config
from exp.video.evaluate import (
    HORIZONS,
    Snapshot,
    evaluate,
    halves,
    run_states,
    shrink_predictions,
)
from exp.video.run import load_checkpoint, source

OBSERVED = 16  # the frames observed before the rollout, as in evaluate
LR, BATCH, SEED = 1e-3, 256, 0  # of the decoder
COLUMNS = (1, 2, 5, 10, 20)  # frames after the last observed one
SHOWN = 4  # clips in the PNG


def decoder(d_z: int, shape: tuple[int, ...]) -> nn.Module:
    """$z$ ``(n, d_z)`` to frames ``(n, *shape)``, ``shape = (C, H, W)`` with
    ``H = W`` a power of 2 from 8: a linear map to 4 x 4, then transposed
    convolutions that double the size."""
    c, h, _ = shape
    ups = int(math.log2(h // 4))
    width = 32 * 2**ups
    layers: list[nn.Module] = [
        nn.Linear(d_z, width * 16),
        nn.Unflatten(1, (width, 4, 4)),
    ]
    for _ in range(ups):
        layers += [nn.ReLU(), nn.ConvTranspose2d(width, width // 2, 4, 2, 1)]
        width //= 2
    layers += [nn.ReLU(), nn.Conv2d(width, c, 3, padding=1)]
    return nn.Sequential(*layers)


@torch.no_grad()
def encode(encoder: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """The $z$ ``(n, D)`` of the frames ``(n, C, H, W)``, in chunks."""
    return torch.cat([encoder(x) for x in images.split(1024)])


def stream_frames(config: Config, n: int) -> torch.Tensor:
    """The first ``n`` frames ``(n, C, H, W)`` of the run's stream (all the
    streams of a frame, with ``data.streams`` > 1)."""
    video, _ = source(config)
    out, count = [], 0
    while count < n:
        image = video.next().image
        image = image.reshape(-1, *image.shape[-3:])
        out.append(image)
        count += image.shape[0]
    return torch.cat(out)[:n]


def fit_decoder(z: torch.Tensor, images: torch.Tensor, steps: int) -> nn.Module:
    """A ``decoder`` of ``z`` ``(n, D)`` to ``images`` ``(n, C, H, W)`` by
    Adam on the mean squared error, from ``SEED``."""
    torch.manual_seed(SEED)  # the initial weights
    g = torch.Generator(z.device).manual_seed(SEED)
    dec = decoder(z.shape[1], tuple(images.shape[1:])).to(z.device)
    opt = torch.optim.Adam(dec.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        idx = torch.randint(len(z), (BATCH,), device=z.device, generator=g)
        loss = ((dec(z[idx]) - images[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return dec.eval()


def write_png(path: Path, image: torch.Tensor) -> None:
    """Write ``image`` ``(H, W, 3)`` uint8 as an RGB PNG."""
    h, w, _ = image.shape
    rows = image.cpu().reshape(h, w * 3).tolist()
    data = b"".join(b"\0" + bytes(row) for row in rows)

    def chunk(kind: bytes, body: bytes) -> bytes:
        crc = zlib.crc32(kind + body)
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", crc)

    header = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
    png += chunk(b"IDAT", zlib.compress(data)) + chunk(b"IEND", b"")
    path.write_bytes(png)


def _tile(frames: torch.Tensor, scale: int) -> torch.Tensor:
    """Frames ``(rows, cols, C, H, W)`` in [0, 1] as one image ``(H', W', 3)``
    uint8, each frame scaled up ``scale`` times, with a gap of 2 pixels."""
    rows, cols, c, h, w = frames.shape
    x = frames.clamp(0, 1).expand(-1, -1, 3, -1, -1) if c == 1 else frames.clamp(0, 1)
    x = x.repeat_interleave(scale, -2).repeat_interleave(scale, -1)
    x = nn.functional.pad(x, (1, 1, 1, 1), value=0.5)  # (rows, cols, 3, H'', W'')
    x = x.permute(0, 3, 1, 4, 2).reshape(rows * x.shape[3], cols * x.shape[4], 3)
    return (x * 255).round().to(torch.uint8)


@torch.no_grad()
def decode_checkpoint(
    snap: Snapshot,
    config: Config,
    clips: Frame,
    train: torch.Tensor,
    out: Path,
    seconds: int,
    steps: int,
) -> dict[str, float]:
    """Fit a decoder to the encoder of ``snap`` on the frames ``train``
    (on its device), write the PNG of the checkpoint at ``seconds`` to ``out``,
    and return its line of ``decode/metrics.jsonl`` (module docstring) on the
    ``clips`` (on its device)."""
    p, c, t = clips.image.shape[:3]
    x = clips.image.reshape(p * c, t, *clips.image.shape[3:])
    d_state = config.model.d_s * config.model.dyn_nodes
    with torch.enable_grad():
        dec = fit_decoder(encode(snap.encoder, train), train, steps)
    z = encode(snap.encoder, x.flatten(0, 1)).reshape(p * c, t, -1)
    steps_ahead = max(HORIZONS)
    _, preds = run_states(
        z, snap.step, snap.readout, d_state, OBSERVED, steps_ahead, snap.residual
    )
    shrunk = shrink_predictions(
        z, halves(p, c, z.device), OBSERVED, sorted({*HORIZONS, *COLUMNS})
    )
    m = z.mean((0, 1))

    def err(z_hat: torch.Tensor, target: torch.Tensor) -> float:
        return float(((dec(z_hat) - target) ** 2).mean())

    line: dict[str, float] = {"time": float(seconds)}
    line["recon"] = err(z.flatten(0, 1), x.flatten(0, 1))
    last = z[:, OBSERVED - 1]
    for h in HORIZONS:
        target = x[:, OBSERVED - 1 + h]
        line[f"img_rollout{h}"] = err(preds[:, h - 1], target)
        line[f"img_persist{h}"] = err(last, target)
        line[f"img_mean{h}"] = err(m.expand_as(last), target)
        line[f"img_shrink{h}"] = err(shrunk[h], target)
        # Without the decoder: the last frame again, and the mean frame.
        line[f"pix_copy{h}"] = float(((x[:, OBSERVED - 1] - target) ** 2).mean())
        line[f"pix_mean{h}"] = float(((train.mean(0) - target) ** 2).mean())
    line.update(evaluate(snap, clips, d_state))
    # A few clips, from different phases of the physics.
    shown = range(0, p * c, max(1, p * c // SHOWN))[:SHOWN]
    cols = [OBSERVED - 1 + h for h in (0, *COLUMNS)]
    rows = []
    for i in shown:
        rollout = [z[i, OBSERVED - 1]] + [preds[i, h - 1] for h in COLUMNS]
        shrink = [z[i, OBSERVED - 1]] + [shrunk[h][i] for h in COLUMNS]
        rows += [
            x[i, cols],
            dec(z[i, cols]),
            dec(torch.stack(rollout)),
            dec(torch.stack(shrink)),
        ]
    scale = max(1, 96 // x.shape[-1])
    write_png(out / f"{seconds:05d}.png", _tile(torch.stack(rows), scale))
    return line


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode the z of checkpoints.")
    parser.add_argument("run", type=Path, help="the directory of a run")
    parser.add_argument("--steps", type=int, default=2000, help="of the decoder")
    parser.add_argument("--frames", type=int, default=16384, help="to fit it on")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    head = json.loads((args.run / "config.json").read_text())
    config = Config.from_dict(head["config"])
    dev = args.device
    train = stream_frames(config, args.frames).to(dev)
    _, clips = source(config)
    clips = clips._replace(
        image=clips.image.to(dev),
        state=clips.state.to(dev),
        params=clips.params.to(dev),
    )
    out = args.run / "decode"
    out.mkdir(exist_ok=True)
    settings = {
        "steps": args.steps,
        "frames": args.frames,
        "lr": LR,
        "batch": BATCH,
        "seed": SEED,
        "repo": git_sha(REPO_ROOT),
        "mujou": git_sha(REPO_ROOT / "mujou"),
        "dirty": git_dirty(REPO_ROOT),
    }
    (out / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
    with (out / "metrics.jsonl").open("w") as f:
        for path in sorted((args.run / "ckpt").glob("*.pt")):
            snap = load_checkpoint(path, dev)
            line = decode_checkpoint(
                snap, config, clips, train, out, int(path.stem), args.steps
            )
            f.write(json.dumps(line) + "\n")
            f.flush()
            keys = ("recon", "img_rollout10", "img_shrink10", "img_mean10")
            print(path.stem, {k: round(line[k], 4) for k in keys}, flush=True)


if __name__ == "__main__":
    main()
