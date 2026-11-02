"""How far the credit gets within a frame, for the nodes of the network of
EgoDex cut into more or fewer nodes (``enc_nodes``, ``dyn_nodes``).

    uv run python -m exp.video.latency --device cuda:1 --out runs/latency.jsonl

With learning off, the stream of EgoDex runs frame by frame; after each step of
a frame, each node's ``delta = r + lam`` is compared with the credit
$-\\partial L_k / \\partial h$ of its activity that autograd computes through the
same modules, memory, buffer and directions at that moment. A node whose
credit is still far from it when the node learns (``learn_interval`` after the
write) learns from a credit that has not come round. The steps go in the order
of the tree (Gauss-Seidel: the activities reach every node within a step, the
credit one node per step) or from the output back (Jacobi: one node per step
both ways, as in a run in real time).
"""

import argparse
import json
import math
import statistics
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import mujou.pc as pc
import mujou.testing
import torch

from exp.lib.sigreg import sigreg
from exp.video.config import Config, DataConfig, ModelConfig, OptimConfig, TimeConfig
from exp.video.egodex import EgoDexStream
from exp.video.model import Network, target_weight
from exp.video.nodes import compare_energy

LAYOUTS = ((1, 1), (2, 1), (3, 1), (2, 2), (3, 2), (3, 3), (2, 3))


def base_config(device: str) -> Config:
    """The network of EgoDex with the setting M11 starts from."""
    return Config(
        name="latency",
        device=device,
        model=ModelConfig(
            d_z=128,
            d_s=512,
            kappa=0.8,
            stop_target=True,
            target_weight=0.1,
            z_norm=True,
        ),
        optim=OptimConfig(tau=(1 / 30) / 0.01),
        data=DataConfig(kind="egodex", size=64),
        time=TimeConfig(learn_interval=math.inf),
    )


def ideal_credits(net: Network, image: torch.Tensor, config: Config) -> list[Any]:
    """$-\\partial L_k / \\partial h$ of each node's activity by autograd, in the
    order of ``net.encoder_names`` then ``net.dynamics_names``."""
    m = config.model
    with torch.enable_grad():
        hs = []
        h = image
        for i, name in enumerate(net.encoder_names):
            h = getattr(net, name).module(h)
            if i == 0:
                h = h.detach().requires_grad_()
            else:
                h.retain_grad()
            hs.append(h)
        z = h
        d = net.dynamics
        y = d.module(d.z_prev, d.s_prev).detach().requires_grad_()
        ys = [y]
        for name in net.dynamics_names[1:]:
            node = getattr(net, name)
            y = node.module(y, node.h_prev)
            y.retain_grad()
            ys.append(y)
        w = target_weight(m)
        target = w * z + (1 - w) * z.detach()
        sr = net.sigreg
        loss = compare_energy(net.compare.module, y, target, m.kappa)
        loss = loss + m.kappa * sigreg(z, sr.buffer[: sr.count], sr.directions)
        loss.backward()
    return [-t.grad for t in (*hs, *ys)]


def measure(config: Config, *, frames: int, warmup: int, order: str) -> dict:
    """The median over ``frames`` frames (after ``warmup``) of the relative error
    of each node's ``r + lam`` against ``ideal_credits``, after each step of the
    frame, and the wall time per frame."""
    net = Network(config)
    stream = EgoDexStream(config.data.path)
    period = net.tau.period
    steps = round(config.time.frame / period)  # 16 or 17: 16 is enough here
    names = [*net.encoder_names, *net.dynamics_names]
    nodes = [getattr(net, n) for n in names]
    if order == "jacobi":
        tree = [net.input, *nodes[: len(net.encoder_names)], net.sigreg]
        tree += nodes[len(net.encoder_names) :] + [net.compare]
        stepping = tuple(reversed(tree))
    else:
        stepping = ()
    errors: list[list[list[float]]] = [[[] for _ in range(steps)] for _ in names]
    dev = config.device
    wall = 0.0
    with mujou.testing.run(net) as run:
        for k in range(warmup + frames):
            f = stream.next()
            sample = pc.msg.Sample(
                x=f.image,
                y=torch.zeros(1),
                mask=torch.tensor(not f.reset),
                reset=torch.tensor(f.reset),
            )
            run.write({net.prev: sample})
            t0 = time.perf_counter()
            for i in range(steps):
                run.step(period, *stepping)
                if k < warmup or f.reset or not net.dynamics.has_prev:
                    continue
                ideal = ideal_credits(net, f.image.to(dev), config)
                for j, (node, want) in enumerate(zip(nodes, ideal, strict=True)):
                    got = node.r + node.lam
                    err = float((got - want).norm() / want.norm().clamp_min(1e-12))
                    errors[j][i].append(err)
            if k >= warmup:
                wall += time.perf_counter() - t0
    return {
        "layout": [config.model.enc_nodes, config.model.dyn_nodes],
        "order": order,
        "nodes": names,
        "median": [
            [statistics.median(e) if e else math.nan for e in per] for per in errors
        ],
        "frames": frames,
        "wall_per_frame": wall / frames,  # with the ideal credits and learning off
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="The credit within a frame.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    with args.out.open("a") as out:
        for enc, dyn in LAYOUTS:
            for order in ("gauss-seidel", "jacobi"):
                config = base_config(args.device)
                config = replace(
                    config, model=replace(config.model, enc_nodes=enc, dyn_nodes=dyn)
                )
                result = measure(
                    config, frames=args.frames, warmup=args.warmup, order=order
                )
                out.write(json.dumps(result) + "\n")
                out.flush()
                at = 12  # the update that learns: 26 ms after the write
                worst = max(row[at] for row in result["median"])
                print(enc, dyn, order, f"worst at 26 ms: {worst:.3f}", flush=True)
                print(
                    "  ",
                    [
                        f"{n}={row[at]:.3f}"
                        for n, row in zip(
                            result["nodes"], result["median"], strict=True
                        )
                    ],
                    flush=True,
                )


if __name__ == "__main__":
    main()
