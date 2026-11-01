"""M11: the world model by BP with the encoder held at its initial weights, so
that only the dynamics and the readout learn (README: the bar of rollout10).

    uv run python -m exp.video.frozen --lr 0.03 --open-loop 0.5 [--live] [--dyn 3]

Prints the rollout errors on the clips every two minutes of the stream.
"""

import argparse
import copy
import json

import mujou.pc as pc
import mujou.testing
import torch

from exp.lib.host import Host
from exp.video.balls import Physics, Video, eval_clips
from exp.video.bp import WorldBPNetwork
from exp.video.config import (
    Config,
    DataConfig,
    ModelConfig,
    OptimConfig,
    make_optimizer,
)
from exp.video.evaluate import Snapshot, evaluate

KEYS = ("rank", "z_var", "rollout1", "rollout5", "rollout10", "rollout20", "mean10")


def main() -> None:
    parser = argparse.ArgumentParser(description="The dynamics on a frozen encoder.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--open-loop", type=float, default=0.0)
    parser.add_argument("--dyn", type=int, default=1, help="the dynamics' layers")
    parser.add_argument("--live", action="store_true", help="the encoder learns too")
    parser.add_argument("--seconds", type=float, default=600.0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    model = ModelConfig(
        kappa=0.8,
        stop_target=True,
        z_norm=True,
        open_loop=args.open_loop,
        open_cycle=20,
        dyn_nodes=args.dyn,
    )
    config = Config(
        seed=args.seed,
        bp=True,
        model=model,
        optim=OptimConfig(tau=(1 / 30) / args.lr),
        data=DataConfig(seconds=args.seconds),
    )
    net = WorldBPNetwork(config)
    w = net.world
    if not args.live:  # the optimizer of the dynamics and the readout only
        w._params = [*w.step.parameters(), *w.readout.parameters()]
        w.optimizer = make_optimizer(config.optim).bind(w._params)
    physics = Physics(period=config.data.physics_period)
    video = Video(seed=args.seed, physics=physics)
    clips = eval_clips(physics=physics)
    every = round(120 * 30)
    with mujou.testing.run(net) as run:
        host = Host(run, frame=1 / 30, period=net.tau.period)
        for k in range(round(args.seconds * 30)):
            f = video.next()
            sample = pc.msg.Sample(
                x=f.image,
                y=torch.zeros(1),
                mask=torch.tensor(not f.reset),
                reset=torch.tensor(f.reset),
            )
            host.show({net.prev: sample}, net.next)
            if (k + 1) % every == 0:
                snap = Snapshot(
                    copy.deepcopy(w.encoder),
                    copy.deepcopy(w.step),
                    copy.deepcopy(w.readout),
                )
                e = evaluate(snap, clips, model.d_s * model.dyn_nodes)
                row = {key: round(e[key], 3) for key in KEYS}
                print(json.dumps({"time": (k + 1) / 30, **row}), flush=True)


if __name__ == "__main__":
    main()
