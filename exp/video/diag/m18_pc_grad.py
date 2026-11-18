r"""DIAGNOSIS ONLY (M18 stage 4): does the VAE of PC-ALM (``vae.pc.VAEPC``)
learn by BP's gradient at a run's checkpoints?

At given weights, ``pc_grads`` shows ``n`` stacks to a ``VAEPC`` of those
weights for one frame under the host's steps (``exp.lib.host``) and records
the gradient each node steps by at its first learning (``GradRecorder``:
nothing moves), $g_\theta / B$, and the frame's first $-\mathcal L$ (``out``'s
nll and ``latent``'s kl, at those weights); ``bp_grads`` is the gradient of
BP's loss, the mean $-\mathcal L$ of the same stacks with the same noise
(``elbo.elbo_terms``, ``loss.batch_loss``), and that loss. Under the run's
``time.learn_interval`` (29 ms: the 15th step) they are equal up to the
rounding (plans/m18-vae-stream.md 「有限の歩数での一致」). ``compare`` gives,
per parameter tensor, the norm of a reference gradient, that of the
difference, their ratio (the relative error) and the cosine.

The bound of the check is the run's own precision: the runs compute in
float32, so a difference of PC-ALM from BP smaller than float32's error in
BP's own gradient is one no run can show. ``main`` therefore computes, at
each point, BP's gradient and loss in float64 (``exact``), BP's and the
PC-ALM net's in float32 (the run's path, CUDA graphs as ``ctl.cuda_graph``)
and the PC-ALM net's in float64, each compared with ``exact``. ``failures``
names what breaks the bound: a tensor whose float64 PC-ALM error exceeds
float32 BP's (QP4; by the theory the former is float64's rounding, some
1e-9 of the latter), and the loss (QP3) when float64 PC-ALM's $-\mathcal L$
is off the exact by more than the stats' own rounding: the nodes send their
nll and kl in float32, half an ulp each, at most $2^{-24}$ of the value,
and the bound $2^{-23} (|\mathrm{nll}| + |\mathrm{kl}|)$ leaves as much
again for float64's rounding before it.

``main`` loads each checkpoint into ``build_vae(conf, seed=conf.seed)`` and
writes these at the initial weights (``init``) and at each checkpoint on the
first ``--stacks`` held-out stacks of val (``measures.held_out_stacks``) with
$u$ and $\varepsilon$ of ``EVAL_NOISE_SEED`` drawn as the evaluations draw
them (the diagnosis of results/m18.md 「段 4」, QP3 and QP4), and under
``provenance`` the snapshot, the host and the settings
(``provenance.of_diagnosis``). Needs the data and the checkpoints (run it
on a host).

Run from the repository root: ``python -m exp.video.diag.m18_pc_grad --out F
--ckpt NAME=PATH [--ckpt ...] +experiment=m18/vae_t128_pc seed=100
device=cuda``.
"""

import argparse
import copy
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
from exp.lib.provenance import of_diagnosis
from exp.video.compose import compose
from exp.video.programs.vision import tours_root
from exp.video.schema.run import RunConf
from exp.video.vae.elbo import elbo_terms
from exp.video.vae.loss import batch_loss
from exp.video.vae.measures import EVAL_NOISE_SEED, held_out_stacks
from exp.video.vae.modules import VAEModules, build_vae
from exp.video.vae.pc import VAEPC

__all__ = ["GradRecorder", "bp_grads", "compare", "failures", "named", "pc_grads"]


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
) -> tuple[dict[str, Tensor], dict[str, float]]:
    r"""$g_\theta / B$ of each parameter of ``mods`` (``named``) at the first
    learning of a ``VAEPC`` of ``mods`` and ``conf`` (its batch set to the
    stacks') shown the stacks ``x`` u8 ``(B, N, C, H, W)`` with the noise
    ``u`` and ``eps`` ``(B, T, d)`` (of ``mods``' dtype) for one frame, every
    stack counting and reset, and the frame's first stats the net sends
    (``nll`` and ``kl``, float32: their sum is $-\mathcal L$). ``mods`` does
    not move.

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
        (out, latent), _ = host.show_all(
            {net.prev: net.sample(x, u, eps, ones, ones)}, [net.stats, net.kl]
        )
    finally:
        run.shutdown()
    missing = [k for k, p in named(mods).items() if p not in rec.grads]
    if missing:
        raise RuntimeError(f"no step of {missing[:3]} ... within the frame")
    grads = {k: rec.grads[p] for k, p in named(mods).items()}
    return grads, {"nll": float(out["nll"]), "kl": float(latent["kl"])}


def bp_grads(
    mods: VAEModules, x: Tensor, u: Tensor, eps: Tensor
) -> tuple[dict[str, Tensor], float]:
    r"""$\partial L / \partial \theta$ of each parameter of ``mods``
    (``named``) and $L$, the mean $-\mathcal L$ of the stacks ``x`` with the
    noise ``u`` and ``eps`` (``bp.VAEBP``'s loss, every stack counting), in
    ``mods``' dtype."""
    params = named(mods)
    valid = torch.ones(len(x), dtype=torch.bool, device=x.device)
    loss = batch_loss(elbo_terms(mods, x, u, eps).loss, valid)
    grads = torch.autograd.grad(loss, list(params.values()), materialize_grads=True)
    return dict(zip(params, grads, strict=True)), float(loss)


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


def failures(
    floor: Sequence[Mapping[str, Any]],
    check: Sequence[Mapping[str, Any]],
    exact: float,
    stats: Mapping[str, float],
) -> list[str]:
    r"""What breaks the bound (module docstring): the tensors whose ``rel`` in
    ``check`` (float64 PC-ALM against the exact gradient) exceeds that in
    ``floor`` (float32 BP against it), both of ``compare`` in the same
    order, and ``"loss"`` when float64 PC-ALM's ``stats`` (``nll`` and
    ``kl``) sum farther from the exact loss ``exact`` than $2^{-23}
    (|\mathrm{nll}| + |\mathrm{kl}|)$. Empty: the check holds."""
    out = [
        c["name"]
        for f, c in zip(floor, check, strict=True)
        if f["rel"] is not None and c["rel"] > f["rel"]
    ]
    nll, kl = stats["nll"], stats["kl"]
    if abs(nll + kl - exact) > 2.0**-23 * (abs(nll) + abs(kl)):
        out.append("loss")
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
    settings = {"overrides": a.overrides, "ckpt": a.ckpt, "stacks": a.stacks}
    result: dict[str, Any] = {"provenance": of_diagnosis(settings)}
    u64, eps64 = u.double(), eps.double()
    for name, path in points:
        mods = build_vae(conf, seed=conf.seed)
        head = {} if path is None else ckpt.load_eval(path, mods.named())
        wide = copy.deepcopy(mods)
        for m in wide.named().values():
            m.double()
        exact, l_exact = bp_grads(wide, x, u64, eps64)
        bp, l_bp = bp_grads(mods, x, u, eps)
        pc32, l_pc = pc_grads(mods, conf, x, u, eps)
        pc64, l_pc64 = pc_grads(wide, conf, x, u64, eps64)
        # -L of BP; nll and kl of PC-ALM's stats
        loss = {"exact": l_exact, "bp": l_bp, "pc": l_pc, "pc64": l_pc64}
        rows = {
            "bp": compare(exact, bp),
            "pc": compare(exact, pc32),
            "pc64": compare(exact, pc64),
            "pc_bp": compare(bp, pc32),
        }
        result[name] = {
            "t": head.get("t", 0.0),
            "failures": failures(rows["bp"], rows["pc64"], l_exact, l_pc64),
            "loss": loss,
            **{k: _worst(v) for k, v in rows.items()},
            "tensors": rows,
        }
        print(
            name, json.dumps({k: v for k, v in result[name].items() if k != "tensors"})
        )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
