r"""Why every yardstick's $m$ of V18c's remade main measure is near 0
(exp/vision/README.md, "V18c: 作り直した主の指標の有効性（2026-11-25）"): a
diagnosis, not a rule. Reads the targets' cache and the yardsticks' inputs
as ``sem_eval`` makes them, and for ``CODES``:

- ``fit_cv``: the centred $R^2$ (``readout.centered_r2``) on the fit set's
  half 1 of a ridge fit on its half 0 (``SemSets.halves``), at each scale of
  ``ALPHA_SCALES``: whether the targets are readable inside the fitting
  videos at all (the fit set is of the training videos, the grid of Zurich);
- ``zurich_cv``: the same within Zurich, the ridge fit on calib and scored
  on sel: whether they are readable inside the held-out video;
- ``transfer``: fit on the fit set, scored on sel (``m_sel`` at each scale);
- the share of the targets' variance on sel that the fit set's mean
  misses (``shift``: $\lVert \bar y_{\mathrm{sel}} - \bar y_{\mathrm{fit}}
  \rVert^2 / \langle \lVert y - \bar y_{\mathrm{sel}} \rVert^2
  \rangle_{\mathrm{sel}}$), and the share of sel's variance in the fit set's
  top 16 whitened components.

Also V-m0's shift identification with the query's own stack left out of the
candidates (``shift_top1_wo_self``) and with every candidate but the truth
within 1 s of the query left out (``shift_top1_wo_near``): described only,
the registered ``shift_top1`` keeps the query among its candidates and is 0
by construction (its own cosine is 1).

Runs on a host with the data and the cache::

    python -m exp.vision.results.v18c.sem_probe --sem <cache> --out <json>
"""

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from exp.lib.linear_dyn import ALPHA_SCALES
from exp.lib.readout import centered_r2, fit_ridge, predict
from exp.vision.sem_eval import (
    SPAN,
    SemConf,
    _Context,
    _yard_conf,
    _yard_inputs,
)

CODES = ("color", "pca16", "pca1024", "inc64")


def _sweep(x_a, y_a, x_b, y_b) -> dict[str, float]:
    out = {}
    for c in ALPHA_SCALES:
        g = fit_ridge(x_a, y_a, c)
        out[f"{c:g}"] = centered_r2(predict(g, x_b).cpu(), y_b.cpu())
    return out


def _shift(y: torch.Tensor, sel: torch.Tensor, drop_near: bool) -> float:
    """The share of the sel stacks $g$ found from $g + 1$'s target by the
    nearest cosine (centred on the grid's mean) among sel's but $g + 1$ (and
    $g + 2$ with ``drop_near``)."""
    members = sel.tolist()
    unit = y - y.mean(0)
    unit = unit / unit.norm(dim=1, keepdim=True)
    hits, n = 0, 0
    for g in members:
        if g + 1 not in members:
            continue
        drop = {g + 1} | ({g + 2} if drop_near else set())
        cols = [h for h in members if h not in drop]
        sims = unit[torch.tensor(cols)] @ unit[g + 1]
        hits += int(cols[int(sims.argmax())] == g)
        n += 1
    return hits / n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sem", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    conf = _yard_conf(None)
    ctx = _Context(
        conf, args.out.parent / "probe-unused", sconf=SemConf(), root=None,
        sem=args.sem, cache=None, spacing=None, span=SPAN,
    )  # fmt: skip
    t = ctx.targets
    split = t.meta["sets_split"]
    idx = {s: torch.tensor(split[s]["idx"]) for s in ("calib", "sel", "final")}
    y_fit, y_grid = t.y("sem", "fit"), t.y("sem", "grid")
    halves = ctx.sets.halves
    res: dict[str, Any] = {}
    ys = y_grid[idx["sel"]]
    shift = float((ys.mean(0) - y_fit.mean(0)).square().sum())
    spread = float((ys - ys.mean(0)).square().sum(1).mean())
    ev = torch.linalg.eigvalsh(torch.cov(ys.T)).flip(0)
    res["shift_over_spread"] = shift / spread
    res["sel_top16_share"] = float(ev[:16].sum() / ev.sum())
    res["sel_total_var"] = spread
    res["fit_total_var"] = float((y_fit - y_fit.mean(0)).square().sum(1).mean())
    res["shift_top1_wo_self"] = _shift(y_grid, idx["sel"], drop_near=False)
    res["shift_top1_wo_near"] = _shift(y_grid, idx["sel"], drop_near=True)
    pca: dict = {}
    for code in CODES:
        x_fit, x_grid = _yard_inputs(code, ctx, {}, pca)  # type: ignore[misc]
        x_fit, x_grid = x_fit.double().cpu(), x_grid.double().cpu()
        h0, h1 = halves == 0, halves == 1
        res[code] = {
            "fit_cv": _sweep(x_fit[h0], y_fit[h0], x_fit[h1], y_fit[h1]),
            "zurich_cv": _sweep(
                x_grid[idx["calib"]],
                y_grid[idx["calib"]],
                x_grid[idx["sel"]],
                y_grid[idx["sel"]],
            ),  # fmt: skip
            "transfer": _sweep(x_fit, y_fit, x_grid[idx["sel"]], ys),
        }
        print(code, json.dumps(res[code]), flush=True)
    args.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in CODES}))


if __name__ == "__main__":
    main()
