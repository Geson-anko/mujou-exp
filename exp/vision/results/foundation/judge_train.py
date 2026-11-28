r"""The judgement of the foundation's short runs (F1 to F3) and the long
runs' choice (``results/foundation/README.md``, 「学習の記録」), the rules
written before any trained number, none of Zurich.

**The lines** (``lines``) are V18c's constructions on the held stacks of the
validation video (``fnd_val``: $s$ = 1, the codes the evaluation reads), of
the untrained JEPA (``init``: the step 0's row of ``val.jsonl``; every arm's
encoder and $W$ of a seed are the same) and of pixel yardsticks
(``pixel_yard``):

- K1 (a constant): ``sigreg_heldin`` $\ge$ (``sigreg_gauss`` +
  ``sigreg_const``) / 2, the midpoint of $N(0, I)$'s and a constant code's
  SIGReg along the same directions
- K2 (low rank): ``rank`` $<$ $\min$(``rank`` of ``color``, of ``init``),
  ``color`` the mean colours of the 16 x 16 patches of each frame
  (``jepa_eval.ColorCode(16)``)
- K3 (every token one code): ``token_r2`` $\ge \theta_3 = (1 + \max($
  ``token_r2`` of ``color_tube``, of ``init``$))/2$, ``color_tube`` the
  mean colour of each frame of each 2 x 8 x 8 tubelet (the tokens' grid)
- the time's K3 (tokens constant along the times): ``time_r2`` $>$
  $\max$(of ``init``, of the pixel tubelets) + 0.05
  (``fnd_guard.time_r2_threshold``, F0's rule on these stacks)

**A short run** (``judge``) is diverged when it did not reach ``STEPS`` or
a number of ``train.jsonl`` or of its last ``val.jsonl`` row is not finite;
collapsed when diverged or any line is crossed at ``STEPS``.

**The choice** (``choose``): of each arm, the run of the least ``val_pred``
(the completion loss of the validation video) at ``STEPS`` of those neither
collapsed nor diverged; none when every run collapsed. ``val_rel`` (its
$R_{\mathrm{clim}}$) is recorded beside.

::

    python -m exp.vision.results.foundation.judge_train yard \
        --run <runs>/fnd-train/<a seed 0 run> --out <runs>/fnd-train/lines.json
    uv run python -m exp.vision.results.foundation.judge_train judge \
        --dir runs/fnd-train [--json out.json]
"""

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor

__all__ = ["choose", "judge", "lines", "main", "pixel_yard", "run_name"]

type Row = dict[str, Any]

ARMS = ("T0", "S1", "Ss")
LRS = (1e-4, 1e-5)
LAMS = (0.5, 0.8, 0.95)
STEPS = 6000
TIME_MARGIN = 0.05  # fnd_guard.TIME_R2_MARGIN
TRAIN_KEYS = ("loss", "pred", "sigreg", "grad_norm", "time_r2")
VAL_KEYS = ("val_pred", "val_rel", "sigreg_heldin", "rank", "token_r2", "time_r2")


def run_name(arm: str, lr: float, lam: float, seed: int = 0) -> str:
    """The short run's directory: ``fnd-S1-lr1e-4-lam0.8-s0``."""
    return f"fnd-{arm}-lr{lr:.0e}-lam{lam:g}-s{seed}".replace("e-0", "e-")


def pixel_yard(held: Tensor, sigreg_const: float) -> Row:
    """The pixel yardsticks of the held stacks u8 ``held`` ``(C, n, 3, S,
    S)`` (module docstring), with ``sigreg_const`` beside."""
    from exp.lib.metrics import effective_rank
    from exp.vision.fnd_guard import tubelets
    from exp.vision.fnd_jepa import time_r2
    from exp.vision.jepa_eval import ColorCode, token_r2

    tube = tubelets(held)  # (C, T, P, f 3 p^2)
    c, t, p, _ = tube.shape
    pixel = tube.reshape(c, t * p, -1)  # time-major as fnd_jepa's tokens
    colour = tube.reshape(c, t * p, 6, -1).mean(-1)  # (C, |Ω|, f 3)
    return {
        "rank_color": effective_rank(ColorCode(16).mu(held)),
        "token_r2_color_tube": float(token_r2(colour[0::2], colour[1::2])),
        "time_r2_pixel": float(time_r2(pixel, t)),
        "sigreg_const": sigreg_const,
    }


def lines(yard: Mapping[str, float], init: Mapping[str, float]) -> dict[str, float]:
    """The lines (module docstring) of the yardsticks ``yard`` (of
    ``pixel_yard``) and the untrained JEPA's row ``init``."""
    return {
        "k1": (init["sigreg_gauss"] + yard["sigreg_const"]) / 2.0,
        "rank_floor": min(yard["rank_color"], init["rank"]),
        "theta3": (1.0 + max(yard["token_r2_color_tube"], init["token_r2"])) / 2.0,
        "time_line": max(init["time_r2"], yard["time_r2_pixel"]) + TIME_MARGIN,
    }


def _finite(row: Mapping[str, Any], keys: Sequence[str]) -> bool:
    return all(
        k in row and row[k] is not None and math.isfinite(float(row[k])) for k in keys
    )


def judge(train: Sequence[Row], val: Sequence[Row], ln: Mapping[str, float]) -> Row:
    """Divergence, the lines and the numbers of a short run of rows
    ``train`` (``train.jsonl``) and ``val`` (``val.jsonl``) at ``STEPS``."""
    last = {r["step"]: r for r in val}.get(STEPS, {})
    finished = bool(train) and train[-1]["step"] == STEPS
    diverged = (
        not finished
        or not all(_finite(r, TRAIN_KEYS) for r in train)
        or not _finite(last, VAL_KEYS)
    )
    row: Row = {k: last.get(k) for k in (*VAL_KEYS, "val_rel_short", "val_rel_long")}
    row |= {"finished": finished, "diverged": diverged}
    if diverged:
        return row | {"K1": None, "K2": None, "K3": None, "KT": None, "collapsed": True}
    row["K1"] = last["sigreg_heldin"] >= ln["k1"]
    row["K2"] = last["rank"] < ln["rank_floor"]
    row["K3"] = last["token_r2"] >= ln["theta3"]
    row["KT"] = last["time_r2"] > ln["time_line"]
    row["collapsed"] = bool(row["K1"] or row["K2"] or row["K3"] or row["KT"])
    return row


def choose(rows: Sequence[Row]) -> Row | None:
    """Of one arm's judged ``rows`` (each with ``name``), the one of the
    least ``val_pred`` of those neither collapsed nor diverged."""
    ok = [x for x in rows if not x["collapsed"] and not x["diverged"]]
    return min(ok, key=lambda x: x["val_pred"]) if ok else None


def _jsonl(path: Path) -> list[Row]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x]


def evaluate(root: Path) -> dict[str, Any]:
    """The lines, every short run's row and each arm's choice of the runs
    under ``root`` (with ``lines.json`` of ``yard``)."""
    got = json.loads((root / "lines.json").read_text())
    ln = got["lines"]
    rows: dict[str, list[Row]] = {}
    choice: dict[str, Any] = {}
    for arm in ARMS:
        rows[arm] = []
        for lr in LRS:
            for lam in LAMS:
                name = run_name(arm, lr, lam)
                d = root / name
                x = judge(_jsonl(d / "train.jsonl"), _jsonl(d / "val.jsonl"), ln)
                rows[arm].append({"name": name, "lr": lr, "lam": lam} | x)
        pick = choose(rows[arm])
        choice[arm] = None if pick is None else pick["name"]
    return {"yard": got["yard"], "init": got["init"], "lines": ln, "rows": rows,
            "choice": choice}  # fmt: skip


def _fmt(x: Any, d: int = 4) -> str:
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "×" if x else "·"
    if isinstance(x, float):
        return f"{x:.{d}f}" if math.isfinite(x) else "nan"
    return str(x)


def _print(out: Mapping[str, Any]) -> None:
    ln = out["lines"]
    print(f"K1 ≥ {ln['k1']:.4f}、K2 の rank の床 {ln['rank_floor']:.2f}、"
          f"K3 の θ3 = {ln['theta3']:.4f}、time_r2 の線 {ln['time_line']:.4f}\n")  # fmt: skip
    print("| run | val_pred | val_rel | short | long | sigreg_heldin | rank "
          "| token_r2 | time_r2 | K1 | K2 | K3 | KT | 発散 | 選択 |")  # fmt: skip
    print("| -- " * 16 + "|")
    for arm, rows in out["rows"].items():
        for x in rows:
            mark = "○" if out["choice"][arm] == x["name"] else ""
            print(f"| `{x['name']}` | {_fmt(x['val_pred'])} | {_fmt(x['val_rel'])} "
                  f"| {_fmt(x['val_rel_short'])} | {_fmt(x['val_rel_long'])} "
                  f"| {_fmt(x['sigreg_heldin'])} | {_fmt(x['rank'], 1)} "
                  f"| {_fmt(x['token_r2'])} | {_fmt(x['time_r2'])} | {_fmt(x['K1'])} "
                  f"| {_fmt(x['K2'])} | {_fmt(x['K3'])} | {_fmt(x['KT'])} "
                  f"| {_fmt(x['diverged'])} | {mark} |")  # fmt: skip
    print(f"\n- choice: {out['choice']}")


def _yard(run: Path, device: str) -> Row:
    """``lines.json``'s content of the seed 0 run ``run`` (on the fleet)."""
    from exp.lib.sigreg import sigreg
    from exp.lib.tours.stacks import tours_root
    from exp.vision.compose import from_dict
    from exp.vision.fnd_jepa import ARMS as ARM_KINDS
    from exp.vision.fnd_val import EVAL_DIRECTIONS, EVAL_SEED, val_set
    from exp.vision.jepa import directions

    d = yaml.safe_load((run / "config.yaml").read_text())
    d["device"] = device
    conf = from_dict(d)
    settings = json.loads((run / "jepa.json").read_text())
    root = tours_root(conf.data.tours.data_version)
    vs = val_set(conf, root, ARM_KINDS["S1"], s_max=1, times=settings["times"])
    dim = settings["latent_dim"]
    dirs = directions(EVAL_DIRECTIONS, dim, torch.Generator().manual_seed(EVAL_SEED))
    b = len(vs.heldin[0])
    const = float(sigreg(torch.zeros(b, dim), torch.zeros(0, dim), dirs))
    init = {r["step"]: r for r in _jsonl(run / "val.jsonl")}[0]
    yard = pixel_yard(vs.held, const)
    return {"run": run.name, "video": vs.video, "held": len(vs.held), "yard": yard,
            "init": init, "lines": lines(yard, init)}  # fmt: skip


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    y = sub.add_parser("yard")
    y.add_argument("--run", type=Path, required=True)
    y.add_argument("--out", type=Path, required=True)
    y.add_argument("--device", default="cpu")
    j = sub.add_parser("judge")
    j.add_argument("--dir", type=Path, required=True)
    j.add_argument("--json", type=Path)
    a = p.parse_args(argv)
    if a.cmd == "yard":
        out = _yard(a.run, a.device)
        a.out.write_text(json.dumps(out, indent=1) + "\n")
        print(json.dumps(out["lines"], indent=1))
        return out
    out = evaluate(a.dir)
    _print(out)
    if a.json is not None:
        a.json.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    return out


if __name__ == "__main__":
    main()
