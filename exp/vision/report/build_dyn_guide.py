"""Build ``exp/vision/dyn-guide.html``: a guide to the main evaluation dyn/copy.

The prose lives in ``dyn_guide.html.in``. The page draws two kinds of figures:

- schematics and closed forms (the OU explainer is exact: a stationary OU
  process is Markov, so the best two-point linear prediction equals the
  one-point one, ``R_copy = (1 + r) / 2`` and ``R_clim = 1 - r^2``; white
  directions have ``R_copy = 1/2``, ``R_clim = 1`` and pool by sums);
- the real data of ``results/dyn/README.md``, extracted from the judge files
  ``runs/v18c-dyn/judge_apply.json`` (sha256 ``b78cb104a88e...``) and
  ``judge_yard.json`` into ``dyn_guide_data.json``.

Subcommands:

- ``data <judge_apply.json> <judge_yard.json>``: write ``dyn_guide_data.json``.
- ``build``: write the HTML (template + data).
- ``extract <out.json>`` / ``apply <in.json>``: the prose review (yomiyasu),
  matched by block index and checked against the original text.

Run with ``uv run python -m exp.vision.report.build_dyn_guide build``.
"""

import json
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "dyn_guide.html.in"
DATA = HERE / "dyn_guide_data.json"
OUT = HERE.parent / "dyn-guide.html"

BLOCK = re.compile(
    r"<(h1|h2|h3|p|li|figcaption|dt|dd|td|th)(\s[^>]*)?>(.*?)</\1>", re.DOTALL
)

# The yard codes drawn next to the learned codes (own reading only).
YARD = ["pca", "mean8", "color", "clock", "init", "init_c", "init_f", "init_m"]


def _r(x: float | None, digits: int = 4) -> float | None:
    return None if x is None else round(float(x), digits)


def _label(name: str) -> tuple[str, str]:
    """A short label and the group of an applied model's key."""
    if name in ("v16", "v17"):
        return f"VAE {name}", "VAE"
    run, step = name.split("@")
    k = "60k" if step == "60000" else "6k"
    if run.startswith("v18c-"):
        m = re.fullmatch(r"v18c-lr(\S+?)-lam(\S+?)-r\S+?-(s\d)", run)
        assert m is not None, run
        lr, lam, s = m.groups()
        return f"V18c {lr} λ{lam} {s} @{k}", f"V18c {lr}"
    if run.startswith("v18b-long-f"):
        return f"V18b F {run[-2:]} @{k}", "V18b F"
    if run.startswith("v18b-long-m"):
        return f"V18b M {run[-2:]} @{k}", "V18b M（崩壊）"
    r = run.split("-")[-2]
    return f"V18 {r} {run[-2:]} @{k}", "V18"


def _curve(c: dict[str, Any]) -> dict[str, list[float | None]]:
    return {
        "lags": c["lags"],
        "copy": [_r(v) for v in c["copy"]],
        "clim": [_r(v) for v in c["clim"]],
    }


def _summary(s: dict[str, Any]) -> dict[str, Any]:
    main = {
        f: {d: _r(s["main"][f][d]) for d in ("copy", "clim")}
        for f in ("lin", "mlp", "lin1")
    }
    below = {
        f: {d: _r(s["below"][f][d]) for d in ("copy", "clim")} for f in ("lin", "mlp")
    }
    lin = s["curves"]["lin"]
    return {
        "main": main,
        "below": below,
        "r256": {d: _r(lin[d][lin["lags"].index(256)]) for d in ("copy", "clim")},
        "curves": {"lin": _curve(lin), "mlp": _curve(s["curves"]["mlp"])},
        "steps": s["steps"],
    }


def _pair(p: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for fam in ("lin", "mlp"):
        q = p[fam]
        out[fam] = {"w": q["words"]}
        for d in ("copy", "clim"):
            out[fam][d] = [_r(q[d]["psi"]), _r(q[d]["lo"]), _r(q[d]["hi"])]
    return out


def extract_data(apply_path: Path, yard_path: Path) -> dict[str, Any]:
    ap = json.loads(apply_path.read_text())
    yd = json.loads(yard_path.read_text())
    yard = {}
    for code in YARD:
        c = yd["codes"][f"{code}/own"]
        entry: dict[str, Any] = {"summary": _summary(c["summary"])}
        if code != "pca":
            entry["vs_pca"] = _pair(yd["pairs"][f"{code}/pca"])
        yard[code] = entry
    models = []
    for name, m in ap["models"].items():
        label, group = _label(name)
        row: dict[str, Any] = {"key": name, "label": label, "group": group}
        row["meta"] = m["meta"]
        for place in ("own", "h_mean"):
            s = _summary(m[place]["summary"])
            if place == "h_mean":
                s.pop("curves")
            row[place] = {
                "summary": s,
                "pairs": {b: _pair(p) for b, p in m[place]["pairs"].items()},
            }
        models.append(row)
    steps = []
    for k, v in ap["steps"].items():
        a, b = k.split("/")
        steps.append({"a": a, "b": b, "own": _pair(v["own"])})
    return {
        "source": {
            "judge_apply": "runs/v18c-dyn/judge_apply.json (sha256 b78cb104a88e…)",
            "note": ap["note"],
        },
        "yard": yard,
        "models": models,
        "steps": steps,
    }


def build() -> None:
    html = TEMPLATE.read_text()
    data = json.dumps(
        json.loads(DATA.read_text()), ensure_ascii=False, separators=(",", ":")
    )
    html = html.replace("/*DATA*/{}", data.replace("</", "<\\/"))
    OUT.write_text(html)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


def _blocks(html: str) -> list[str]:
    return [m.group(3) for m in BLOCK.finditer(html)]


def extract_text(out: Path) -> None:
    blocks = _blocks(TEMPLATE.read_text())
    out.write_text(
        json.dumps(
            [{"i": i, "text": t} for i, t in enumerate(blocks)],
            ensure_ascii=False,
            indent=1,
        )
    )
    print(f"{len(blocks)} blocks -> {out}")


def apply_text(inp: Path) -> None:
    html = TEMPLATE.read_text()
    edits = {e["i"]: e for e in json.loads(inp.read_text())}
    pieces = []
    last = 0
    for i, m in enumerate(BLOCK.finditer(html)):
        e = edits.get(i)
        if e is None or e.get("new") is None:
            continue
        if e["text"] != m.group(3):
            raise SystemExit(f"block {i} does not match the template")
        pieces.append(html[last : m.start(3)])
        pieces.append(e["new"])
        last = m.end(3)
    pieces.append(html[last:])
    TEMPLATE.write_text("".join(pieces))
    print(f"applied {sum(1 for e in edits.values() if e.get('new') is not None)} edits")


def main(argv: list[str]) -> None:
    cmd = argv[0] if argv else "build"
    if cmd == "data":
        DATA.write_text(
            json.dumps(
                extract_data(Path(argv[1]), Path(argv[2])),
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        print(f"wrote {DATA} ({DATA.stat().st_size} bytes)")
    elif cmd == "build":
        build()
    elif cmd == "extract":
        extract_text(Path(argv[1]))
    elif cmd == "apply":
        apply_text(Path(argv[1]))
    else:
        raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    main(sys.argv[1:])
