"""M11: the bound of rollout10 offline, by BP through time on windows of the
synthetic video drawn at random, with Adam (README: M11, the offline bound).

    uv run python -m exp.video.offline [--config <json>] [--learn-encoder]
        [--sigreg 15] [--local] [--batch 128] [--blur 0] [--sgd --lr 0.03]
        [--device cuda:0] [--save runs/<dir>]

The model (sizes, ``z_norm``, ``residual``, the nodes of the dynamics) and the
stream (the physics, the redraws, the ``streams``) are those of ``--config``
(a ``Config`` as for ``run.py``; left out, ``DEFAULT``: $D=64$, $d_s=256$,
``z_norm`` and the default stream). The first ``--frames`` frames of the
stream (over all its streams) are kept, and each window has 32 frames of one
stream within an episode (``window_starts``): the model observes the first 16
and then runs open loop on its own predictions, and the loss is the mean
prediction error over the 31 predictions. With ``--learn-encoder``, the
encoder learns too, with the full credit to the targets, and SIGReg on the z
of the batch weighted by ``--sigreg`` per sample (about kappa / (1 - kappa)
of the nodes); ``--local`` stops the gradient where the nodes stop it, at the
context $z_{k-1}$ and at the prediction fed back. Prints the evaluation on the
clips every ``--every`` steps; with ``--save``, also a checkpoint in
``<dir>/ckpt/<step>.pt`` (named by the step, not by seconds) with a
``config.json`` of the model and the stream, for ``decode.py``.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn

from exp.lib.sigreg import epps_pulley
from exp.video.config import Config, DataConfig, ModelConfig
from exp.video.evaluate import Snapshot, evaluate
from exp.video.model import (
    DynamicsStack,
    Top,
    calibration_images,
    dynamics_modules,
    encoder_stages,
    standardize_,
)
from exp.video.run import save_checkpoint, source

KEYS = ("rank", "z_var", "rollout1", "rollout10", "persist10", "mean10")
WINDOW, OBSERVED = 32, 16
DEFAULT = Config(
    name="offline",
    model=ModelConfig(d_z=64, d_s=256, z_norm=True),
    data=DataConfig(kind="video"),
)


def window_starts(resets: torch.Tensor, window: int) -> torch.Tensor:
    """The windows ``(n, 2)`` of ``window`` frames, as (first frame, stream),
    in the frames of the streams whose ``resets`` are ``(T, B)``: those that
    no episode starts within after their first frame."""
    count = resets.long().cumsum(0)  # the resets up to each frame
    within = count[window - 1 :] - count[: len(resets) - window + 1]
    return (within == 0).nonzero()


def stream_windows(config: Config, frames: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The first ``frames`` frames of the stream of ``config`` (over its
    streams), as images ``(T, B, C, H, W)``, and their ``window_starts``."""
    video, _ = source(config)
    b = config.data.streams
    images, resets = [], []
    for _ in range(frames // b):
        f = video.next()
        images.append(f.image.reshape(b, *f.image.shape[-3:]))
        resets.append(torch.as_tensor(f.reset).reshape(b))
    return torch.stack(images), window_starts(torch.stack(resets), WINDOW)


def main() -> None:
    parser = argparse.ArgumentParser(description="The offline bound of rollout10.")
    parser.add_argument("--config", type=Path, help="the model and the stream")
    parser.add_argument("--learn-encoder", action="store_true")
    parser.add_argument("--sigreg", type=float, default=15.0)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--blur", type=float, help="instead of that of the config")
    parser.add_argument("--sgd", action="store_true", help="SGD instead of Adam")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--frames", type=int, default=30000, help="of the stream")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--every", type=int, default=1000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save", type=Path, help="a directory for checkpoints")
    args = parser.parse_args()
    config = DEFAULT if args.config is None else Config.load(args.config)
    if args.blur is not None:
        config = replace(config, model=replace(config.model, blur=args.blur))
    m = config.model
    if args.save is not None:  # the model as a run of the nodes would have it
        args.save.mkdir(parents=True, exist_ok=True)
        shown = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        head = {"config": config.to_dict(), "offline": shown | {"save": None}}
        (args.save / "config.json").write_text(json.dumps(head, indent=2) + "\n")
    torch.manual_seed(config.seed)
    dev, b = args.device, args.batch
    encoder = nn.Sequential(*encoder_stages(config))
    standardize_(encoder, calibration_images(config))
    images, starts = stream_windows(config, args.frames)
    images = images.to(dev)
    encoder = encoder.to(dev)
    first, layers = dynamics_modules(m)
    step = DynamicsStack(first, layers).to(dev)
    linear = nn.Linear(m.d_s, m.d_z)
    if m.residual:  # the prediction starts as z_{k-1}, the change as 0
        nn.init.zeros_(linear.weight)
        nn.init.zeros_(linear.bias)
    readout = nn.Sequential(Top(m.d_s), linear).to(dev)
    params = [*step.parameters(), *readout.parameters()]
    if args.learn_encoder:
        params += list(encoder.parameters())
    optimizer = (
        torch.optim.SGD(params, lr=args.lr)
        if args.sgd
        else torch.optim.Adam(params, lr=args.lr)
    )
    _, clips = source(config)
    clips = clips._replace(
        image=clips.image.to(dev),
        state=clips.state.to(dev),
        params=clips.params.to(dev),
    )

    def held(x: torch.Tensor) -> torch.Tensor:  # where the nodes stop the credit
        return x.detach() if args.local else x

    for i in range(args.steps):
        t, which = starts[torch.randint(len(starts), (b,))].unbind(1)
        x = images[t[:, None] + torch.arange(WINDOW), which[:, None]]  # (B, T, C, H, W)
        with torch.set_grad_enabled(args.learn_encoder):
            z = encoder(x.flatten(0, 1)).reshape(b, WINDOW, m.d_z)
        s = z.new_zeros(b, step.d_state)
        z_in = held(z[:, 0])
        loss = z.new_zeros(())
        for k in range(1, WINDOW):
            s = step(z_in, s)
            z_hat = z_in + readout(s) if m.residual else readout(s)
            loss = loss + ((z_hat - z[:, k]) ** 2).mean() / (WINDOW - 1)
            z_in = held(z[:, k]) if k < OBSERVED else held(z_hat)
        if args.learn_encoder:  # the mean over samples, times the weight
            a = torch.randn(m.d_z, m.directions, device=dev)
            a = a / a.norm(dim=0)
            samples = z.flatten(0, 1)
            loss = loss + args.sigreg * epps_pulley(samples @ a).mean() / len(samples)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % args.every == 0:
            snap = Snapshot(encoder, step, readout, m.residual)
            e = evaluate(snap, clips, step.d_state)
            if args.save is not None:
                save_checkpoint(snap, args.save, i + 1)
            row = {k: round(e[k], 3) for k in KEYS}
            print(json.dumps({"step": i + 1, **row}), flush=True)


if __name__ == "__main__":
    main()
