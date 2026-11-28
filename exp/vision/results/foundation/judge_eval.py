r"""The verdicts of the foundation's evaluation stage (``plans/foundation.md``
4; ``results/foundation/README.md``, 「評価の段」), the rules written
before any learned code's evaluation was read.

**The inputs**: the yardsticks' outputs of F0 (``--f0``: ``pca``,
``mean8``, ``v16``, ``v17``, ``rot``; their content, ``pca``'s half view,
the pixel tubelets' and ``init``'s ``time_r2``), the learned codes' and
``init``'s (``--train``: the same layout), the predictor's own completion
(``--self``, ``fnd_self``) and the training runs (``--runs``: each run's
``val.jsonl`` and ``lines.json``).

**A code** (``code_entry``): ``judge_fnd.gru_summary`` (the main count of
layers chosen on the validation video), its pairs against each yardstick of
``AGAINST`` at its main count of layers (``judge_fnd.pair``,
$\sigma_{\mathrm{rot}}$ of ``rot`` against ``pca``), the two-point linear's
pair against ``pca``, the half view's and the self-identification's
$R_{\mathrm{copy}}$, $R_{\mathrm{clim}}$ (the evaluation's sums), and the
guards:

- the collapse (``collapse``): K1 to K3 and the time's line of the run's
  ``val.jsonl`` row at the code's step, against ``lines.json``
  (``judge_train``'s rules); ``init`` is untrained and only recorded
- ``time_r2`` (the reference's tokens, ``fnd_guard``) $\le$ the line fixed
  before (``time_line``: $\max$(``init``, pixels) + 0.05)
- the content: $U = 1 - E^{\mathrm{lin}} \le 1.10 \max(U_{\texttt{pca}},
  U_{\texttt{init}})$ (``fnd_guard.noninferior``)

**Good** (``good``) against a yardstick $b$: in the future and in the past
the words are ``a_better`` (both denominators' intervals below 0), and every
guard holds. **The cheats** (``cheats``): *slow* where a direction's pair
against ``pca`` has $\psi_{\mathrm{copy}} > 0$ and $\psi_{\mathrm{clim}} < 0$
with the intervals apart from 0 (the code is easier than ``pca`` only
against the mean: a slower code), *constant in time* where ``time_r2``
exceeds its line, *content dropped* where $U$ exceeds its bound.

**An arm** (``arm_verdict``): its long runs' last step, each seed apart;
the arm is good against $b$ when both seeds are, split when one is. The
short runs, the long runs' steps 0, 6000 and 30000 are descriptions.

::

    uv run python -m exp.vision.results.foundation.judge_eval --f0 runs/fnd-f0-eval \
        --train runs/fnd-train-eval --self runs/fnd-eval --runs runs/fnd-train [--json out.json]
"""

import argparse
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from exp.vision.fnd_guard import NONINFERIOR, TIME_R2_MARGIN, fidelity_of
from exp.vision.results.foundation.judge_fnd import (
    gru_summary,
    lin_pair,
    lin_summary,
    pair,
)

__all__ = [
    "AGAINST",
    "arm_verdict",
    "cheats",
    "code_entry",
    "collapse",
    "good",
    "judge",
    "main",
    "ratios",
    "run_of",
]

type Row = dict[str, Any]

AGAINST = ("pca", "v16", "v17", "mean8")
ARMS = ("T0", "S1", "Ss")
LAST = 45000  # the long runs' last step (LONG_STEPS)
DIRS = ("future", "past")


def run_of(code: str) -> tuple[str, int] | None:
    """The training run and step of a learned code's name (``init`` the
    seed 0's step 0, ``init-s1`` the seed 1's); None of a yardstick."""
    if code == "init":
        return "fnd-T0-lr1e-4-lam0.5-s0", 0
    if code == "init-s1":
        return "fndL-T0-lr1e-4-lam0.5-s1", 0
    m = re.fullmatch(r"L-(.+-s\d)-(\d+)", code)
    if m:
        return f"fndL-{m[1]}", int(m[2])
    if re.fullmatch(r"(T0|S1|Ss)-lr[\de.-]+-lam[\d.]+", code):
        return f"fnd-{code}-s0", 6000
    return None


def collapse(row: Mapping[str, float], ln: Mapping[str, float]) -> dict[str, bool]:
    """K1 to K3 and the time's line of a ``val.jsonl`` row
    (``judge_train.judge``'s)."""
    return {
        "K1": row["sigreg_heldin"] >= ln["k1"],
        "K2": row["rank"] < ln["rank_floor"],
        "K3": row["token_r2"] >= ln["theta3"],
        "KT": row["time_r2"] > ln["time_line"],
    }


def ratios(rows: Sequence[Row], key: str = "src") -> dict[str, dict[str, float]]:
    """$R_{\\mathrm{copy}}$ and $R_{\\mathrm{clim}}$ of each half-view or
    self row (``"<src>-<dst>"``) of the evaluation's sums."""
    out = {}
    for r in rows:
        if r.get("part") not in ("half", "self"):
            continue
        num = sum(r["num_b"])
        out[f"{r[key]}-{r['dst']}"] = {
            "copy": num / sum(r["copy_b"]),
            "clim": num / sum(r["clim_b"]),
        }
    return out


def cheats(
    pairs_pca: Mapping[str, Any], time_ok: bool | None, content_ok: bool
) -> dict[str, Any]:
    """The cheats' signs (module docstring) of a code's pairs against
    ``pca`` by direction."""
    slow = {}
    for d in DIRS:
        p = pairs_pca.get(d)
        lo, hi = (p["copy"]["lo"], p["clim"]["hi"]) if p else (None, None)
        slow[d] = lo is not None and hi is not None and lo > 0 and hi < 0
    return {"slow": slow, "constant_in_time": time_ok is False,
            "content_dropped": not content_ok}  # fmt: skip


def good(pairs: Mapping[str, Any], guards_ok: bool) -> bool:
    """Whether the future's and the past's words are ``a_better`` and the
    guards hold."""
    return guards_ok and all(pairs.get(d, {}).get("words") == "a_better" for d in DIRS)


def code_entry(
    code: str, gru: Mapping[str, list[Row]], lin: Mapping[str, list[Row]],
    content: Mapping[str, Row], now: Mapping[str, list[Row]],
    self_rows: Mapping[str, list[Row]], time_r2: Mapping[str, float],
    val: Mapping[str, list[Row]], ln: Mapping[str, float], bounds: Mapping[str, float],
) -> dict[str, Any]:  # fmt: skip
    """The numbers, guards and verdicts of ``code`` (module docstring)."""
    e: dict[str, Any] = {"run": run_of(code)}
    s = gru_summary(gru[code])
    e["dyn"] = {d: {"main": s[d]["main"], "mean_log": s[d]["layers"][str(s[d]["main"])]["mean_log"],
                    "sd": s[d]["layers"][str(s[d]["main"])]["sd"],
                    "by_layers": {n: v["mean_log"] for n, v in s[d]["layers"].items()},
                    "val": {n: v["val"] for n, v in s[d]["layers"].items()},
                    "best": {n: v["best"] for n, v in s[d]["layers"].items()}}
                for d in DIRS}  # fmt: skip
    e["pairs"] = {}
    for b in AGAINST:
        if b == code or b not in gru:
            continue
        e["pairs"][b] = {d: pair(gru[code], gru[b], d, s[d]["main"], gru.get("rot"),
                                 gru.get("pca")) for d in DIRS}  # fmt: skip
    if code in lin and "pca" in lin and code != "pca":
        e["lin"] = lin_summary(lin[code])
        e["lin_pair_pca"] = {d: lin_pair(lin[code], lin["pca"], d) for d in DIRS}
    if code in now:
        e["now"] = ratios(now[code])
    if code in self_rows:
        e["self"] = ratios(self_rows[code])
    g: dict[str, Any] = {}
    if code in content:
        el, ed = fidelity_of(content[code])
        g["E_lin"], g["U"], g["E_d_lin"] = el, 1 - el, ed
        g["content_ok"] = 1 - el <= bounds["u_max"]
    if code in time_r2:
        g["time_r2"] = time_r2[code]
        g["time_ok"] = time_r2[code] <= bounds["time_line"]
    run = e["run"]
    if run is not None and run[0] in val:
        row = next((r for r in val[run[0]] if r["step"] == run[1]), None)
        if row is not None:
            g["val_row"] = {k: row[k] for k in ("sigreg_heldin", "rank", "token_r2", "time_r2",
                                                "val_pred", "val_rel")}  # fmt: skip
            g["collapse"] = collapse(row, ln)
    e["guards"] = g
    learned = run is not None and run[1] > 0
    if learned:
        ok = (not any(g.get("collapse", {}).values()) and g.get("time_ok", False)
              and g.get("content_ok", False))  # fmt: skip
        g["all_ok"] = ok
        e["good"] = {b: good(p, ok) for b, p in e["pairs"].items()}
        if "pca" in e["pairs"]:
            e["cheats"] = cheats(e["pairs"]["pca"], g.get("time_ok"),
                                 g.get("content_ok", False))  # fmt: skip
    return e


def arm_verdict(codes: Mapping[str, Any], arm: str) -> dict[str, str]:
    """Of each yardstick: ``good`` when both seeds' last steps are,
    ``split`` when one is, else ``not``."""
    out = {}
    for b in AGAINST:
        got = [codes.get(f"L-{arm}-lr1e-4-lam0.5-s{s}-{LAST}", {}).get("good", {}).get(b)
               for s in (0, 1)]  # fmt: skip
        n = sum(bool(x) for x in got)
        out[b] = "good" if n == 2 else ("split" if n == 1 else "not")
    return out


def _rows(path: Path) -> list[Row]:
    return json.loads(path.read_text())


def _jsonl(path: Path) -> list[Row]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def judge(f0: Path, train: Path, self_dir: Path, runs: Path) -> dict[str, Any]:
    """Every code's entry and each arm's verdict (module docstring)."""
    gru: dict[str, list[Row]] = {}
    lin: dict[str, list[Row]] = {}
    content: dict[str, Row] = {}
    now: dict[str, list[Row]] = {}
    tr2: dict[str, float] = {}
    for root in (f0, train):
        for f in sorted((root / "gru").glob("*/*.json")):
            gru.setdefault(f.parent.name, []).extend(_rows(f))
        lin |= {f.stem: _rows(f) for f in sorted((root / "lin").glob("*.json"))}
        content |= {
            f.stem: _rows(f)[0] for f in sorted((root / "content").glob("*.json"))
        }
        now |= {f.stem: _rows(f) for f in sorted((root / "now").glob("*.json"))
                if f.name != "config.json"}  # fmt: skip
        for f in sorted((root / "guard").glob("time_r2_*.json")):
            tr2[f.stem.removeprefix("time_r2_")] = _rows(f)[0]["value"]
    self_rows = {f.stem: _rows(f) for f in sorted((self_dir / "self").glob("*.json"))
                 if f.name != "config.json"}  # fmt: skip
    ln = json.loads((runs / "lines.json").read_text())["lines"]
    val = {p.parent.name: _jsonl(p) for p in sorted(runs.glob("fnd*/val.jsonl"))}
    u = {c: 1 - fidelity_of(content[c])[0] for c in ("pca", "init")}
    bounds = {"time_line": max(tr2["init"], tr2["pixel"]) + TIME_R2_MARGIN,
              "u_max": NONINFERIOR * max(u.values())}  # fmt: skip
    codes = {c: code_entry(c, gru, lin, content, now, self_rows, tr2, val, ln, bounds)
             for c in sorted(gru) if c != "rot"}  # fmt: skip
    return {"bounds": bounds, "lines": ln, "codes": codes,
            "arms": {a: arm_verdict(codes, a) for a in ARMS}}  # fmt: skip


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    p = argparse.ArgumentParser(
        prog="python -m exp.vision.results.foundation.judge_eval"
    )
    p.add_argument("--f0", type=Path, required=True)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--self", dest="self_dir", type=Path, required=True)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--json", type=Path)
    a = p.parse_args(argv)
    out = judge(a.f0, a.train, a.self_dir, a.runs)
    text = json.dumps(out, indent=1, default=lambda x: None if isinstance(x, float)
                      and not math.isfinite(x) else x)  # fmt: skip
    if a.json:
        a.json.write_text(text + "\n")
    else:
        print(text)
    return out


if __name__ == "__main__":
    main()
