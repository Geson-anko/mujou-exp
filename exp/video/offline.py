"""M11: the bound of rollout10 offline, by BP through time on windows of the
synthetic video drawn at random, with Adam (README: M11, the offline bound).

    uv run python -m exp.video.offline [--learn-encoder] [--sigreg 15]
        [--local] [--batch 128] [--blur 0] [--sgd --lr 0.03] [--device cuda:0]
        [--save runs/<dir>]

Each window has 32 frames within an episode: the model observes the first 16
and then runs open loop on its own predictions, and the loss is the mean
prediction error over the 31 predictions. With ``--learn-encoder``, the
encoder learns too, with the full credit to the targets, and SIGReg on the z
of the batch weighted by ``--sigreg`` per sample (about kappa / (1 - kappa)
of the nodes); ``--local`` stops the gradient where the nodes stop it, at the
context $z_{k-1}$ and at the prediction fed back. Prints the evaluation on the
clips every ``--every`` steps; with ``--save``, also a checkpoint in
``<dir>/ckpt/<step>.pt`` (named by the step, not by seconds) with a
``config.json`` of the same model, for ``decode.py``.
"""

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from exp.lib.mingru import MinGRU
from exp.lib.sigreg import epps_pulley
from exp.video.balls import Video, eval_clips
from exp.video.config import Config, DataConfig, ModelConfig
from exp.video.evaluate import Snapshot, evaluate
from exp.video.model import DynamicsStack, Top, encoder_modules, standardize_
from exp.video.run import save_checkpoint

KEYS = ("rank", "z_var", "rollout1", "rollout10", "persist10", "mean10")
WINDOW, OBSERVED = 32, 16
D_Z, D_S = 64, 256


def main() -> None:
    parser = argparse.ArgumentParser(description="The offline bound of rollout10.")
    parser.add_argument("--learn-encoder", action="store_true")
    parser.add_argument("--sigreg", type=float, default=15.0)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--blur", type=float, default=0.0)
    parser.add_argument("--sgd", action="store_true", help="SGD instead of Adam")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--frames", type=int, default=30000, help="of the stream")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--every", type=int, default=1000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save", type=Path, help="a directory for checkpoints")
    args = parser.parse_args()
    if args.save is not None:  # the model as a run of the nodes would have it
        model = ModelConfig(d_z=D_Z, d_s=D_S, z_norm=True, blur=args.blur)
        config = Config(name="offline", model=model, data=DataConfig(kind="video"))
        args.save.mkdir(parents=True, exist_ok=True)
        head = {"config": config.to_dict(), "offline": vars(args) | {"save": None}}
        (args.save / "config.json").write_text(json.dumps(head, indent=2) + "\n")
    torch.manual_seed(0)
    dev, b = args.device, args.batch
    g1, z_mod = encoder_modules(32, D_Z, z_norm=True, blur=args.blur)
    encoder = nn.Sequential(g1, z_mod)
    calibration = Video(seed=10_000)
    standardize_(encoder, torch.stack([calibration.next().image for _ in range(256)]))
    video = Video(seed=0)
    frames = [video.next() for _ in range(args.frames)]
    images = torch.stack([f.image for f in frames]).to(dev)
    resets = [f.reset for f in frames]
    starts = torch.tensor(
        [i for i in range(len(frames) - WINDOW) if not any(resets[i + 1 : i + WINDOW])]
    )
    encoder = encoder.to(dev)
    step = DynamicsStack(MinGRU(D_Z, D_S), []).to(dev)
    readout = nn.Sequential(Top(D_S), nn.Linear(D_S, D_Z)).to(dev)
    params = [*step.parameters(), *readout.parameters()]
    if args.learn_encoder:
        params += list(encoder.parameters())
    optimizer = (
        torch.optim.SGD(params, lr=args.lr)
        if args.sgd
        else torch.optim.Adam(params, lr=args.lr)
    )
    clips = eval_clips()
    clips = clips._replace(
        image=clips.image.to(dev),
        state=clips.state.to(dev),
        params=clips.params.to(dev),
    )

    def held(x: torch.Tensor) -> torch.Tensor:  # where the nodes stop the credit
        return x.detach() if args.local else x

    for i in range(args.steps):
        idx = starts[torch.randint(len(starts), (b,))]
        x = images[idx[:, None] + torch.arange(WINDOW)]  # (B, T, 1, H, W)
        with torch.set_grad_enabled(args.learn_encoder):
            z = encoder(x.flatten(0, 1)).reshape(b, WINDOW, D_Z)
        s = z.new_zeros(b, D_S)
        z_in = held(z[:, 0])
        loss = z.new_zeros(())
        for k in range(1, WINDOW):
            s = step(z_in, s)
            z_hat = readout(s)
            loss = loss + ((z_hat - z[:, k]) ** 2).mean() / (WINDOW - 1)
            z_in = held(z[:, k]) if k < OBSERVED else held(z_hat)
        if args.learn_encoder:  # the mean over samples, times the weight
            a = torch.randn(D_Z, 256, device=dev)
            a = a / a.norm(dim=0)
            samples = z.flatten(0, 1)
            loss = loss + args.sigreg * epps_pulley(samples @ a).mean() / len(samples)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % args.every == 0:
            snap = Snapshot(encoder, step, readout)
            e = evaluate(snap, clips, D_S)
            if args.save is not None:
                save_checkpoint(snap, args.save, i + 1)
            row = {k: round(e[k], 3) for k in KEYS}
            print(json.dumps({"step": i + 1, **row}), flush=True)


if __name__ == "__main__":
    main()
