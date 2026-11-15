r"""DIAGNOSIS ONLY (M18 stage 1): does the decoder's last map keep its
initial weights, and how much of the reconstruction error is theirs?

The last map of the decoder (``TokenDecoder.stage.out``, $W \in
\mathbb{R}^{\tau C p^2 \times d}$, $d$ = 256 to 6144) starts at LeCun's
random $W_0$ (std about $0.88/\sqrt{d}$ after the truncation). With
``scale=fan_in`` an Adam-type step moves each element of $W$ by about $\eta /
d$ (``exp.lib.optim``), at most about $\eta F / d$ in $F$ steps (6.1e-3 for
$\eta$ = 2.9e-5, $F$ = 54000), a tenth of $W_0$'s std: the steps can add a
few coherent directions (a sum over the $d$ inputs moves the output by up to
$\eta F$) but cannot overwrite $W_0$. Its output $\sigma W_0 h$ ($h$ the
final ``LayerNorm``'s output, $\sigma$ the pixels' std) is then a random
projection of $h$ left in every reconstruction.

- ``movement``: per parameter of a module, its fan-in $n$ (``fan_ins``), the
  root mean square of its initial value and of its change, and the largest
  change of an element.
- ``without_init_readout``: a copy of the decoder whose last map is
  $W - W_0$ (``out.weight`` less ``w0``): the decoder with the initial
  readout's term $\sigma W_0 h$ taken out, all else as learned.
- ``readout_errors``: ``stack_mse`` (``vae.measures``) of the autoencoder and
  of the one without the initial readout on the same stacks, and the energy
  of the removed term $\sum \lVert \hat y - \hat y_{-0} \rVert^2 / \sum \lVert
  y - m \rVert^2$ (``e0``) on the same scale.

``main`` loads each checkpoint into ``build_ae(conf, seed=conf.seed)`` (the
run's initial weights are those of its seed) and prints, per checkpoint and
for the initial weights (``init``), the decoder's ``movement`` and the
encoder's, and ``readout_errors`` on the held-out stacks of val
(``measures.held_out_stacks``, the mean stack of ``training_moments``): the
check pre-registered in results/m18.md, 「段 1 の追加の確かめ」. Needs the data and
the checkpoints (run it on a host).

Run from the repository root: ``python -m exp.video.diag.m18_readout --out
F --ckpt NAME=PATH [--ckpt ...] +experiment=m18/ae_t128 seed=100
device=cuda``.
"""

import argparse
import copy
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from exp.lib import ckpt
from exp.lib.optim import fan_ins
from exp.video.compose import compose
from exp.video.programs.vision import tours_root
from exp.video.vae.loss import stack_mse_sums, target
from exp.video.vae.measures import EVAL_BATCH, held_out_stacks, training_moments
from exp.video.vae.modules import AEModules, TokenDecoder, build_ae

__all__ = ["movement", "readout_errors", "without_init_readout"]


@torch.no_grad()
def movement(before: nn.Module, after: nn.Module) -> list[dict[str, Any]]:
    r"""Per parameter of ``after`` (by name, in order): its fan-in ``n``
    (``fan_ins`` of ``after``), ``rms0`` the root mean square of its value in
    ``before``, ``rms_delta`` and ``max_delta`` those of the change
    $\theta_{after} - \theta_{before}$ (root mean square, largest absolute
    value of an element)."""
    n_of = fan_ins(after)
    start = dict(before.named_parameters())
    out = []
    for name, p in after.named_parameters():
        p0 = start[name].to(p.device)
        d = (p - p0).double()
        out.append(
            {
                "name": name,
                "shape": list(p.shape),
                "n": n_of[p],
                "rms0": float(p0.double().pow(2).mean().sqrt()),
                "rms_delta": float(d.pow(2).mean().sqrt()),
                "max_delta": float(d.abs().max()),
            }
        )
    return out


@torch.no_grad()
def without_init_readout(decoder: TokenDecoder, w0: Tensor) -> TokenDecoder:
    r"""A copy of ``decoder`` whose last map's weight is $W - W_0$ (``w0``:
    $W_0$, of the weight's shape): its output is $\mu + \sigma((W - W_0) h +
    b)$ where ``decoder``'s is $\mu + \sigma(W h + b)$."""
    out = copy.deepcopy(decoder)
    last = out.stage.out
    assert last is not None
    last.weight -= w0.to(last.weight.device)
    return out


@torch.no_grad()
def readout_errors(
    full: Callable[[Tensor], Tensor],
    ablated: Callable[[Tensor], Tensor],
    stacks: Tensor,
    mean: Tensor,
    *,
    batch: int = EVAL_BATCH,
) -> dict[str, float]:
    r"""``stack_mse`` of ``full`` and of ``ablated`` ($x \mapsto \hat y$) on
    the u8 stacks ``(n, N, C, H, W)`` with the mean stack ``mean`` ``(N, C,
    H, W)`` (on the modules' device), and ``e0`` $= \sum \lVert \hat y_{full}
    - \hat y_{ablated} \rVert^2 / \sum \lVert y - m \rVert^2$; a batch at a
    time."""
    dev = mean.device
    sums = torch.zeros(4, dtype=torch.float64, device=dev)
    m = mean.float()
    for chunk in stacks.split(batch):
        x = chunk.to(dev)
        y, y_full, y_abl = target(x), full(x), ablated(x)
        e_full, base = stack_mse_sums(y, y_full, m)
        e_abl, _ = stack_mse_sums(y, y_abl, m)
        e0 = (y_full - y_abl).double().pow(2).sum()
        sums += torch.stack([e_full, e_abl, e0, base])
    e_full, e_abl, e0, base = sums.tolist()
    return {"stack_mse": e_full / base, "ablated": e_abl / base, "e0": e0 / base}


def _check(
    mods: AEModules, init: AEModules, stacks: Tensor, mean: Tensor
) -> dict[str, Any]:
    last = init.decoder.stage.out
    assert last is not None
    ablated = AEModules(mods.vision, without_init_readout(mods.decoder, last.weight))
    return {
        "decoder": movement(init.decoder, mods.decoder),
        "encoder": movement(init.vision.encoder, mods.vision.encoder),
        "val": readout_errors(mods, ablated, stacks, mean),
    }


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m exp.video.diag.m18_readout")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ckpt", action="append", default=[], help="NAME=PATH")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args(argv)
    conf = compose(a.overrides)
    assert conf.data.tours is not None
    torch.backends.cuda.matmul.allow_tf32 = False
    root = tours_root(conf.data.tours)
    dev = torch.device(conf.device)
    stacks = held_out_stacks(conf, root)["val"]
    mean = training_moments(conf, root).mean.float().to(dev)
    init = build_ae(conf, seed=conf.seed)
    result = {"init": _check(init, init, stacks, mean)}
    for spec in a.ckpt:
        name, path = spec.split("=", 1)
        mods = build_ae(conf, seed=conf.seed)
        head = ckpt.load_eval(Path(path), mods.named())
        result[name] = {"t": head["t"], "path": path} | _check(mods, init, stacks, mean)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1))
    for name, r in result.items():
        out = next(x for x in r["decoder"] if x["name"] == "stage.out.weight")
        norm = next(x for x in r["decoder"] if x["name"] == "stage.norm.weight")
        print(name, json.dumps(r["val"]), json.dumps(out), json.dumps(norm))


if __name__ == "__main__":
    main()
