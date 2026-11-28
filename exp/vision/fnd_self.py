r"""The foundation's self-identification (``plans/foundation.md`` 4.2): the
learned predictor's own completion of a half of the grid, in its own
representation space, cut by cut (a description beside ``fnd_guard``'s half
view, which reads the codes from outside).

**A cut** $(a, b)$ of ``fnd_guard.PAIRS``: the encoder sees the positions of
the half $a$ only (``fnd_guard.half_keep``), the predictor completes the
other half $b$ (``fnd_jepa.complete``, the hidden mask $b$; of a scaled arm
the spacing $s = 1$, $\ell = 0$), and its $\hat t_\omega$ is scored against
the target side's tokens $t_\omega$ (every position seen,
``fnd_jepa.targets``). The spatial cuts are the present, ``early`` ->
``late`` the future and ``late`` -> ``early`` the past inside the stack.

**The sums** (``cut_terms``) of each stack over $\omega \in b$: the
numerator $\lVert \hat t_\omega - t_\omega \rVert^2$, the copy's $\lVert
t_{n(\omega)} - t_\omega \rVert^2$ ($n(\omega)$ the nearest position of $a$
along the cut's axis, ``nearest_seen``: the last time seen of the future,
the column next to the cut of the present) and the mean's $\lVert \bar
t_\omega - t_\omega \rVert^2$ ($\bar t_\omega$ the mean of the reference's
tokens). $R_{\mathrm{copy}}$ and $R_{\mathrm{clim}}$ are the ratios of their
sums over Zurich's non-overlapping stacks (``fnd_dyn``'s, blocks of 128 s
and the shifted ones); the denominators are of the code's own tokens, so
the ratios are compared across codes only as a description::

    python -m exp.vision.fnd_self --config <v18c run> --sets runs/v18c-tlin \
        --out runs/fnd-eval --places NAME <ckpt> [--device cuda]
"""

import argparse
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from exp.lib.dynpred import block_sums
from exp.lib.gru_dyn import step_blocks
from exp.lib.provenance import configure_determinism
from exp.vision.eval import BATCH
from exp.vision.fill_eval import FillConf
from exp.vision.fnd_dyn import FndConf, encode_stacks, stack_starts
from exp.vision.fnd_guard import PAIRS, half_keep
from exp.vision.fnd_jepa import (
    EVAL_BATCH,
    complete,
    context_index,
    encode,
    load_fnd,
    targets,
)
from exp.vision.geom_eval import GeomConf, _conf_of, _start, _write
from exp.vision.jepa import JEPAModules
from exp.vision.models import early_fusion
from exp.vision.width_eval import WidthConf, sets_and_ref

__all__ = ["CUTS", "cut_terms", "main", "nearest_seen", "predict_half"]

CUTS: tuple[tuple[str, str, str], ...] = tuple(
    (axis, a, b) for axis, pairs in PAIRS.items() for a, b in pairs
)

_F64 = torch.float64


def nearest_seen(seen: str, grid: tuple[int, int, int]) -> Tensor:
    """Of each position of the grid ``(T, H, W)`` (time-major, row-major),
    the flat index ``(T H W,)`` of the nearest position of the half
    ``seen`` along the cut's axis (itself if seen)."""
    t, h, w = grid
    jt, jr, jc = torch.meshgrid(
        torch.arange(t), torch.arange(h), torch.arange(w), indexing="ij"
    )
    clamp = {"early": (jt.clamp(max=t // 2 - 1), jr, jc),
             "late": (jt.clamp(min=t // 2), jr, jc),
             "top": (jt, jr.clamp(max=h // 2 - 1), jc),
             "bottom": (jt, jr.clamp(min=h // 2), jc),
             "left": (jt, jr, jc.clamp(max=w // 2 - 1)),
             "right": (jt, jr, jc.clamp(min=w // 2))}  # fmt: skip
    if seen not in clamp:
        raise ValueError(f"no half {seen!r}")
    a, b, c = clamp[seen]
    return ((a * h + b) * w + c).flatten()


def cut_terms(
    t_hat: Tensor, t: Tensor, dst: Tensor, near: Tensor, mean: Tensor
) -> Tensor:
    r"""Each stack's sums ``(B, 3)`` float64 over the positions ``dst``
    ``(|Ω|,)`` bool of the numerator $\lVert \hat t - t \rVert^2$, the
    copy's $\lVert t_{n(\omega)} - t_\omega \rVert^2$ (``near`` ``(|Ω|,)``)
    and the mean's $\lVert \bar t_\omega - t_\omega \rVert^2$ (``mean``
    ``(|Ω|, d)``) of the tokens ``t_hat``, ``t`` ``(B, |Ω|, d)``."""
    t_hat, t, mean = t_hat.double(), t.double(), mean.double()
    on = dst.to(t.device)
    num = (t_hat - t).square().sum(-1)[:, on].sum(1)
    copy = (t[:, near.to(t.device)] - t).square().sum(-1)[:, on].sum(1)
    clim = (mean[None] - t).square().sum(-1)[:, on].sum(1)
    return torch.stack([num, copy, clim], 1)


@torch.no_grad()
def predict_half(
    mods: JEPAModules, x: Tensor, heads: int, times: int, seen: Tensor, scaled: bool
) -> Tensor:
    r"""The predictor's $\hat t$ ``(B, |Ω|, d)`` of the stacks u8 ``x``
    whose encoder sees the positions ``seen`` ``(|Ω|,)`` bool only (the
    others hidden; of a ``scaled`` arm $\ell = 0$, $s$ = 1)."""
    dev = mods.parameters()[0].device
    hidden = (~seen.to(dev)).expand(len(x), -1)
    images = early_fusion(x.to(dev))
    keep, valid, slot = context_index(hidden)
    context = encode(mods, images, heads, keep, valid)
    ell = torch.zeros(len(x), dtype=_F64, device=dev) if scaled else None
    return complete(mods, context, slot, hidden, times, ell)


def main(argv: Sequence[str] | None = None) -> Path:
    p = argparse.ArgumentParser(prog="python -m exp.vision.fnd_self")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--sets", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--root", type=Path)
    p.add_argument("--device")
    p.add_argument("--places", nargs=2, metavar=("NAME", "CKPT"), required=True)
    a = p.parse_args(argv)
    gconf, fconf = GeomConf(), FndConf()
    conf = _conf_of(a.config, a.device)
    determinism = configure_determinism(conf.device)
    name, path = a.places
    run_conf, settings, mods = load_fnd(Path(path), conf.device)
    heads = run_conf.model.head_count
    times, g = settings["times"], settings["grid"]
    grid = (times, g, g)
    scaled = bool(settings.get("scaled", False))
    _, zurich, ref, _ = sets_and_ref(conf, a.sets, a.root, gconf, WidthConf(),
                                     FillConf(), None, None)  # fmt: skip
    mean = torch.cat(
        [targets(mods, x, heads)[0].double().cpu() for x in ref.split(BATCH)]
    ).mean(0)
    cuts = [(axis, s, d, half_keep(s, grid).flatten(), half_keep(d, grid).flatten(),
             nearest_seen(s, grid)) for axis, s, d in CUTS]  # fmt: skip

    def fn(x: Tensor) -> Tensor:
        out = []
        for c in x.split(EVAL_BATCH):
            t = targets(mods, c, heads)[0]
            out.append(torch.cat([
                cut_terms(predict_half(mods, c, heads, times, seen, scaled), t, dst,
                          near, mean.to(t.device))
                for _, _, _, seen, dst, near in cuts], 1))  # fmt: skip
        return torch.cat(out)

    t = conf.data.tours
    zs = stack_starts(len(zurich.frames), t.frames, t.stride)
    terms = encode_stacks(fn, lambda i, j: zurich.frames[i:j], zs, t.frames, t.stride,
                          torch.device(conf.device), lambda z: z, fconf.read).double()  # fmt: skip
    blocks = {"": step_blocks([len(zs)], fconf.block, 0),
              "_s": step_blocks([len(zs)], fconf.block, fconf.block // 2)}  # fmt: skip
    rows: list[dict[str, Any]] = []
    for i, (axis, s, d, *_) in enumerate(cuts):
        row: dict[str, Any] = {"part": "self", "axis": axis, "src": s, "dst": d}
        for tail, b in blocks.items():
            nb = int(b.max()) + 1
            for j, key in enumerate(("num", "copy", "clim")):
                row[f"{key}_b{tail}"] = block_sums(terms[:, 3 * i + j], b, nb).tolist()
        row["r_copy"] = sum(row["num_b"]) / sum(row["copy_b"])
        row["r_clim"] = sum(row["num_b"]) / sum(row["clim_b"])
        rows.append(row)
    _start(a.out / "self", {"geom": asdict(gconf)}, determinism)
    target = a.out / "self" / f"{name}.json"
    _write(target, rows, {"code": name, "ckpt": path, "scaled": scaled})
    return target


if __name__ == "__main__":
    main()
