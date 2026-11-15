r"""DIAGNOSIS ONLY (M17, 2026-11-14): reconstructions of the evaluation's frames
from mean colors alone, no learned features, to see how much of a code's
``recon_blur`` (the untrained ViT's above all) mean color explains. Not a
program of the runs, and nothing of the runs imports it.

Two codes of a stack $x$ ``(n, N, C, H, W)`` in $[0, 1]$, functions of the
data alone:

- ``tube``: the mean color of each tubelet ($\tau$ frames x $p \times p$
  pixels, the ViT's tokens in ``patchify_tubelets``' order), $D = T C$
  (``tubelet_means``);
- ``stack``: the mean color of the whole stack, $D = C$ (``stack_means``),
  the mean of the ``tube`` code's tubelets (they are of one size).

Three reconstructions of the stack's last frame $f$ (the evaluation's target)
from each:

- ``direct`` (no fit): ``tube`` fills each patch of $f$ with the mean color
  of its tubelet of the last time slot (``tube_picture``), ``stack`` every
  pixel with the stack's mean color (``stack_picture``);
- ``linear``: the least-squares affine map $\hat f = c W + b$ of the code
  $c$ on the evaluation's training stacks (``encoder_eval.stream_windows``,
  ``fit_affine``: closed form, no ridge, no draw), clamped to $[0, 1]$ as the
  evaluation clamps its decoded frames. ``direct`` is a fixed linear map of
  the code and the mean image the map $W = 0$: on the training stacks the
  ``linear`` squared error is at most either's;
- ``decoder``: the evaluation's own decoder of the code
  (``EncoderEval.evaluate_code``: the code standardized, ``posthoc.decoder``
  from the seed ``seed + DECODER_SEED``, ``steps`` steps of rate
  ``DECODER_LR``), the path of the untrained ViT's and the PCA's lines, with
  the linear dynamics of the code (``dyn_h/copy_h``) and, the ``linear`` map
  handed to it, its ``recon_lin``, ``recon_lin_blur`` (the same as the
  ``linear`` here, measured by the evaluation).

Every reconstruction is measured as the evaluation measures one, on the first
stack of each window of val and test (``PackedMeasures``): ``recon`` and
``recon_blur`` ($\sigma$ = 2 px), over the error of the mean image of the
training stacks' last frames.

``python -m exp.video.diag.mean_color_ref --out DIR [--yardsticks]
+experiment=m17/ssig_t128 seed=100 device=cuda`` writes ``DIR/mean_color_ref.json``,
``DIR/refs.png`` (val: the true last frames, then ``direct`` and ``linear``
of ``tube``, then of ``stack``) and the evaluation's files in ``DIR/eval``
(``<code>.png``, ``<code>-dyn.png``, ``codes.jsonl``; with ``--yardsticks``
also unit 0, the PCA and the untrained ViT, rerun on this host:
``baselines.jsonl``).
"""

import argparse
import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from exp.lib import pca
from exp.lib.blur_error import blur_recon
from exp.lib.png import tile, write_png
from exp.lib.rundir import write_json
from exp.video.compose import compose
from exp.video.measure.encoder_eval import (
    DECODER_LR,
    SHOWN,
    WINDOWS,
    EncoderEval,
    stream_windows,
)
from exp.video.measure.posthoc import BATCH, DECODER_SEED, STEPS, as_float
from exp.video.programs.vision import PackedMeasures, tours_root
from exp.video.schema.run import RunConf

__all__ = [
    "CODES",
    "fit_affine",
    "measure",
    "reference",
    "stack_means",
    "stack_picture",
    "tube_picture",
    "tubelet_means",
]

CODES = ("tube", "stack")
SPLITS = ("val", "test")
# The keys of the evaluation's lines kept in the JSON (all in codes.jsonl).
_KEPT = (
    "recon",
    "recon_blur",
    "recon_fit",
    "recon_lin",
    "recon_lin_blur",
    "rank",
    "z_spread",
    "alpha",
)


def tubelet_means(x: Tensor, patch: int, tubelet: int) -> Tensor:
    """The ``tube`` code ``(n, T C)`` of stacks ``x`` ``(n, N, C, H, W)``
    float: the mean color of each tubelet, the tubelets in
    ``patchify_tubelets``' order (time slot, row, column), the $C$ colors of
    one together."""
    n, f, c, h, w = x.shape
    gt, gh, gw = f // tubelet, h // patch, w // patch
    m = x.reshape(n, gt, tubelet, c, gh, patch, gw, patch).mean((2, 5, 7))
    return m.permute(0, 1, 3, 4, 2).reshape(n, gt * gh * gw * c)  # (n, gt gh gw C)


def stack_means(x: Tensor) -> Tensor:
    """The ``stack`` code ``(n, C)`` of stacks ``x`` ``(n, N, C, H, W)``
    float: the mean color over all its frames and pixels."""
    return x.mean((1, 3, 4))


def tube_picture(code: Tensor, grid: tuple[int, int, int], patch: int) -> Tensor:
    """The ``direct`` frames ``(n, C, gh p, gw p)`` of the ``tube`` code
    ``(n, gt gh gw C)`` of the grid ``(gt, gh, gw)``: each patch filled with
    the mean color of its tubelet of the last time slot."""
    gt, gh, gw = grid
    m = code.reshape(len(code), gt, gh, gw, -1)[:, -1]  # (n, gh, gw, C)
    m = m.permute(0, 3, 1, 2)  # (n, C, gh, gw)
    return m.repeat_interleave(patch, -2).repeat_interleave(patch, -1)


def stack_picture(code: Tensor, side: int) -> Tensor:
    """The ``direct`` frames ``(n, C, side, side)`` of the ``stack`` code
    ``(n, C)``: every pixel its stack's mean color."""
    return code[:, :, None, None].expand(-1, -1, side, side)


def fit_affine(z: Tensor, y: Tensor, *, batch: int = BATCH) -> tuple[Tensor, Tensor]:
    r"""$(b, W)$, ``(F,)`` and ``(D, F)`` float32, of the least squares
    $\min \sum_i \lVert z_i W + b - y_i \rVert^2$ for codes ``z`` ``(n, D)``
    and targets ``y`` ``(n, ...)`` of $F$ values (u8 frames read as ``x /
    255``), on the device of ``z``.

    Closed form in float64: with $\bar z, \bar y$ the means and $Z_c$ the
    centred codes, $W$ solves $Z_c^\top Z_c W = Z_c^\top Y$ (``lstsq``: the
    least-norm solution if $Z_c$ is not of full rank) and $b = \bar y - \bar
    z W$. $Z_c^\top Y = Z_c^\top (Y - \bar y)$, as the columns of $Z_c$ sum
    to 0, so $Y$ is read a ``batch`` of rows at a time, never centred.
    """
    if len(z) != len(y):
        raise ValueError(f"{len(z)} codes for {len(y)} targets")
    z = z.double()
    mz = z.mean(0)
    zc = z - mz
    czz = zc.T @ zc  # (D, D)
    f = y[0].numel()
    sy = torch.zeros(f, dtype=torch.float64, device=z.device)
    czy = torch.zeros(len(mz), f, dtype=torch.float64, device=z.device)  # (D, F)
    for i in range(0, len(z), batch):
        yb = as_float(y[i : i + batch].to(z.device)).flatten(1).double()  # (b, F)
        sy += yb.sum(0)
        czy += zc[i : i + batch].T @ yb
    w = torch.linalg.lstsq(czz.cpu(), czy.cpu()).solution.to(z.device)
    b = sy / len(z) - mz @ w
    return b.float(), w.float()


def measure(x_hat: Tensor, f: Tensor, mean: Tensor) -> dict[str, float]:
    """``recon`` and ``recon_blur`` of the reconstructions ``x_hat`` of the
    frames ``f`` ``(n, C, H, W)``, over the errors of the mean image ``mean``
    ``(1, C, H, W)``, as the evaluation's lines."""
    m = mean.expand_as(f)
    recon = float(((x_hat - f) ** 2).mean() / ((m - f) ** 2).mean())
    return {"recon": recon, "recon_blur": blur_recon(x_hat, f, mean)}


def _codes(conf: RunConf) -> dict[str, Callable[[Tensor], Tensor]]:
    """The codes by name, of stacks u8 or floats in $[0, 1]$."""
    v = conf.vision
    assert v is not None
    return {
        "tube": lambda s: tubelet_means(as_float(s), v.patch, v.tubelet),
        "stack": lambda s: stack_means(as_float(s)),
    }


def _picture(conf: RunConf, name: str, code: Tensor) -> Tensor:
    """The ``direct`` frames of the code ``name``."""
    v, t = conf.vision, conf.data.tours
    assert v is not None and t is not None
    if name == "stack":
        return stack_picture(code, t.size)
    side = t.size // v.patch
    return tube_picture(code, (v.frames // v.tubelet, side, side), v.patch)


def _train_fits(
    conf: RunConf, root: Path, windows: int, dev: torch.device
) -> tuple[Tensor, dict[str, tuple[Tensor, Tensor]], int]:
    """The mean image ``(1, C, H, W)`` of the training stacks' last frames
    (as ``EncoderEval``'s), the ``linear`` map of each code and the number of
    training stacks."""
    v = conf.vision
    assert v is not None
    train = stream_windows(conf, root, windows=windows)  # (w, N + K - 1, ...) u8
    k = train.shape[1] - v.frames + 1
    targets = train[:, v.frames - 1 :].flatten(0, 1).to(dev)  # (w K, C, H, W) u8
    sums = torch.stack([as_float(c).sum(0) for c in targets.split(BATCH)])
    mean = (sums.sum(0) / len(targets))[None]
    codes = _codes(conf)
    fits = {}
    for name, of in codes.items():
        z = []
        for window in train.split(max(1, BATCH // k)):
            x = window.to(dev)
            stacks = torch.stack([x[:, i : i + v.frames] for i in range(k)], 1)
            z.append(of(stacks.flatten(0, 1)))  # (window, stack) order
        fits[name] = fit_affine(torch.cat(z), targets)
    return mean, fits, len(targets)


def _held_out(
    conf: RunConf,
    mean: Tensor,
    fits: Mapping[str, tuple[Tensor, Tensor]],
    dev: torch.device,
    out: Path,
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """``direct`` and ``linear`` of each code by code and split; the
    pictures of val (``refs.png``)."""
    v = conf.vision
    assert v is not None
    measures = PackedMeasures(conf)
    got: dict[str, dict[str, dict[str, dict[str, float]]]] = {c: {} for c in CODES}
    for split in SPLITS:
        windows = measures.windows[split]
        x = windows[:, : v.frames].to(dev)  # the first stacks, u8
        f0 = as_float(windows[:, v.frames - 1].to(dev))
        rows = [f0[:SHOWN]]
        for name, of in _codes(conf).items():
            code = of(x)
            direct = _picture(conf, name, code).clamp(0, 1)
            linear = pca.decode(code, *fits[name]).view_as(f0).clamp(0, 1)
            got[name][split] = {
                "direct": measure(direct, f0, mean),
                "linear": measure(linear, f0, mean),
            }
            rows += [direct[:SHOWN], linear[:SHOWN]]
        if split == "val":
            write_png(out / "refs.png", tile(torch.stack(rows).cpu(), scale=1))
    return got


def _clean(x: Any) -> Any:
    return None if isinstance(x, float) and not math.isfinite(x) else x


def reference(
    conf: RunConf,
    root: Path,
    out: Path,
    *,
    windows: int = WINDOWS,
    steps: int = STEPS,
    yardsticks: bool = False,
) -> dict[str, Any]:
    """The reconstructions of every code (module docstring) of the run
    ``conf`` on the data ``root``, its files in ``out``; returns the JSON."""
    dev = torch.device(conf.device)
    out.mkdir(parents=True, exist_ok=True)
    mean, fits, stacks = _train_fits(conf, root, windows, dev)
    got = _held_out(conf, mean, fits, dev, out)
    e = EncoderEval(None, conf, out / "eval", root=root, windows=windows, steps=steps)
    result: dict[str, Any] = {
        "config": {
            "name": conf.name,
            "seed": conf.seed,
            "decoder_seed": conf.seed + DECODER_SEED,
            "decoder_lr": DECODER_LR,
            "steps": steps,
            "windows": windows,
            "stacks": stacks,
            "data_version": None
            if conf.data.tours is None
            else conf.data.tours.data_version,
            "device": conf.device,
        },
        "codes": {},
    }
    if yardsticks:
        lines = e.step()  # unit 0: the PCA and the untrained ViT, baselines.jsonl
        result["yardsticks"] = {
            f"{ln['code']}/{ln['split']}": {
                k: _clean(ln[k]) for k in (*_KEPT, "dyn_1/copy_1") if k in ln
            }
            for ln in lines
        }
    dyn = [f"dyn_{h}/copy_{h}" for h in conf.measure.horizons]
    with (out / "eval" / "codes.jsonl").open("w") as fh:
        for name, of in _codes(conf).items():
            lines = e.evaluate_code(name, of, linear=fits[name])
            entry: dict[str, Any] = {"D": len(fits[name][1])}
            for ln in lines:
                fh.write(json.dumps({k: _clean(x) for k, x in ln.items()}) + "\n")
                kept = {k: _clean(ln[k]) for k in (*_KEPT, *dyn) if k in ln}
                entry[ln["split"]] = got[name][ln["split"]] | {"decoder": kept}
            result["codes"][name] = entry
    return result


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="mean-color reconstructions")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--windows", type=int, default=WINDOWS)
    p.add_argument("--steps", type=int, default=STEPS)
    p.add_argument("--yardsticks", action="store_true", help="rerun unit 0 too")
    p.add_argument("overrides", nargs="*")
    args = p.parse_args(argv)
    conf = compose(args.overrides)
    if conf.data.tours is None:
        raise SystemExit("the run streams no Walking Tours")
    result = reference(
        conf,
        tours_root(conf.data.tours),
        args.out,
        windows=args.windows,
        steps=args.steps,
        yardsticks=args.yardsticks,
    )
    write_json(args.out / "mean_color_ref.json", result)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
