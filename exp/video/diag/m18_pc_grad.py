r"""DIAGNOSIS ONLY (M18 stage 4): does the VAE of PC-ALM (``vae.pc.VAEPC``)
learn by BP's gradient at a run's checkpoints?

At given weights, ``pc_grads`` shows ``n`` stacks to a ``VAEPC`` of those
weights for one frame under the host's steps (``exp.lib.host``) and records
the gradient each node steps by at its first learning (``GradRecorder``:
nothing moves), $g_\theta / B$; ``bp_grads`` is the gradient of BP's loss,
the mean $-\mathcal L$ of the same stacks with the same noise
(``elbo.elbo_terms``, ``loss.batch_loss``). Under the run's
``time.learn_interval`` (29 ms: the 15th step) they are equal up to the
rounding (plans/m18-vae-stream.md 「有限の歩数での一致」). ``compare`` gives,
per parameter tensor, the norm of BP's gradient, that of the difference,
their ratio (the relative error) and the cosine.

``main`` loads each checkpoint into ``build_vae(conf, seed=conf.seed)`` and
writes ``compare`` at the initial weights (``init``) and at each checkpoint
on the first ``--stacks`` held-out stacks of val (``measures.held_out_stacks``)
with $u$ and $\varepsilon$ of ``EVAL_NOISE_SEED`` drawn as the evaluations
draw them (the diagnosis of results/m18.md 「段 4」, QP4). Needs the data and
the checkpoints (run it on a host).

Run from the repository root: ``python -m exp.video.diag.m18_pc_grad --out F
--ckpt NAME=PATH [--ckpt ...] +experiment=m18/vae_t128_pc seed=100
device=cuda``.
"""

import argparse
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, override

import mujou.pc as pc
import mujou.testing
import torch
from torch import Tensor

from exp.lib import ckpt
from exp.lib.host import Host
from exp.video.compose import compose
from exp.video.programs.vision import tours_root
from exp.video.schema.run import RunConf
from exp.video.vae.elbo import elbo_terms
from exp.video.vae.loss import batch_loss
from exp.video.vae.measures import EVAL_NOISE_SEED, held_out_stacks
from exp.video.vae.modules import VAEModules, build_vae
from exp.video.vae.pc import VAEPC

__all__ = ["GradRecorder", "bp_grads", "compare", "named", "pc_grads"]


class GradRecorder(pc.optim.Optimizer):
    """An optimizer that moves nothing and records, by parameter, the last
    gradient a node stepped by (``grads``) and each step's ``elapsed``."""

    def __init__(self) -> None:
        self.grads: dict[Tensor, Tensor] = {}
        self.elapsed: list[float] = []

    @override
    def bind(self, params: Iterable[Tensor]) -> pc.optim.BoundOptimizer:
        mine, rec = list(params), self

        class _Bound(pc.optim.BoundOptimizer):
            @override
            def step_(self, grads: Sequence[Tensor], elapsed: float) -> None:
                for p, g in zip(mine, grads, strict=True):
                    rec.grads[p] = g.detach().clone()
                rec.elapsed.append(elapsed)

            @override
            def state_dict(self) -> dict[str, Any]:
                return {}

            @override
            def load_state_dict(self, state: Any, /) -> None:
                pass

        return _Bound()


def named(mods: VAEModules) -> dict[str, Tensor]:
    """The parameters of ``mods`` by the names of the checkpoints
    (``encoder.stages.0.embed.proj.weight``, ...)."""
    return {
        f"{k}.{n}": p for k, m in mods.named().items() for n, p in m.named_parameters()
    }


def pc_grads(
    mods: VAEModules, conf: RunConf, x: Tensor, u: Tensor, eps: Tensor
) -> dict[str, Tensor]:
    r"""$g_\theta / B$ of each parameter of ``mods`` (``named``) at the first
    learning of a ``VAEPC`` of ``mods`` and ``conf`` (its batch set to the
    stacks') shown the stacks ``x`` u8 ``(B, N, C, H, W)`` with the noise
    ``u`` and ``eps`` ``(B, T, d)`` for one frame, every stack counting and
    reset. ``mods`` does not move.

    Raises:
        RuntimeError: A parameter was not stepped within the frame (no node
            learned: ``time.learn_interval`` longer than the frame).
    """
    b = len(x)
    conf = replace(conf, data=replace(conf.data, streams=b))
    rec = GradRecorder()
    net = VAEPC(mods, conf=conf, optimizer=rec)
    run = mujou.testing.run(net)
    try:
        host = Host(run, frame=conf.time.frame, period=net.tau.period)
        ones = torch.ones(b, dtype=torch.bool, device=x.device)
        host.show_all({net.prev: net.sample(x, u, eps, ones, ones)}, [net.stats])
    finally:
        run.shutdown()
    missing = [k for k, p in named(mods).items() if p not in rec.grads]
    if missing:
        raise RuntimeError(f"no step of {missing[:3]} ... within the frame")
    return {k: rec.grads[p] for k, p in named(mods).items()}


def bp_grads(mods: VAEModules, x: Tensor, u: Tensor, eps: Tensor) -> dict[str, Tensor]:
    r"""$\partial L / \partial \theta$ of each parameter of ``mods``
    (``named``), $L$ the mean $-\mathcal L$ of the stacks ``x`` with the
    noise ``u`` and ``eps`` (``bp.VAEBP``'s loss, every stack counting)."""
    params = named(mods)
    valid = torch.ones(len(x), dtype=torch.bool, device=x.device)
    loss = batch_loss(elbo_terms(mods, x, u, eps).loss, valid)
    grads = torch.autograd.grad(loss, list(params.values()), materialize_grads=True)
    return dict(zip(params, grads, strict=True))


def compare(
    bp: Mapping[str, Tensor], pc_: Mapping[str, Tensor]
) -> list[dict[str, Any]]:
    """Per tensor of ``bp`` (by name, in order): the norms of BP's gradient
    (``norm``) and of the difference (``diff``), ``rel`` = ``diff / norm``
    (None where ``norm`` is 0) and the cosine ``cos`` (None where either is
    0), in float64."""
    out = []
    for name, g in bp.items():
        a, b = g.double().flatten(), pc_[name].double().flatten()
        norm, diff = float(a.norm()), float((b - a).norm())
        nb = float(b.norm())
        out.append(
            {
                "name": name,
                "norm": norm,
                "diff": diff,
                "rel": diff / norm if norm > 0 else None,
                "cos": float(a @ b) / (norm * nb) if norm > 0 and nb > 0 else None,
            }
        )
    return out


def _worst(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The largest ``rel`` and the smallest ``cos`` of ``compare``'s rows."""
    rels = [r["rel"] for r in rows if r["rel"] is not None]
    coss = [r["cos"] for r in rows if r["cos"] is not None]
    return {
        "max_rel": max(rels, default=math.nan),
        "min_cos": min(coss, default=math.nan),
    }


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m exp.video.diag.m18_pc_grad")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ckpt", action="append", default=[], help="NAME=PATH")
    ap.add_argument("--stacks", type=int, default=8)
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args(argv)
    conf = compose(a.overrides)
    assert conf.data.tours is not None
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device(conf.device)
    x = held_out_stacks(conf, tours_root(conf.data.tours))["val"][: a.stacks].to(dev)
    mods = build_vae(conf, seed=conf.seed)
    gen = torch.Generator(device=dev).manual_seed(EVAL_NOISE_SEED)
    eps = torch.randn(len(x), *mods.code_shape, generator=gen, device=dev)
    u = torch.rand(x.shape, generator=gen, device=dev)
    points: list[tuple[str, Path | None]] = [("init", None)]
    for spec in a.ckpt:
        name, path = spec.split("=", 1)
        points.append((name, Path(path)))
    result: dict[str, Any] = {}
    for name, path in points:
        mods = build_vae(conf, seed=conf.seed)
        head = {} if path is None else ckpt.load_eval(path, mods.named())
        rows = compare(bp_grads(mods, x, u, eps), pc_grads(mods, conf, x, u, eps))
        result[name] = {"t": head.get("t", 0.0), **_worst(rows), "tensors": rows}
        print(
            name, json.dumps({k: v for k, v in result[name].items() if k != "tensors"})
        )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
