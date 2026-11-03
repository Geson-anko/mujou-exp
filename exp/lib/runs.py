"""The index and the tables of the runs: what the fleet left on disk, read back.

Standard library only: it runs on the Mac's plain Python and inside the
fleet's CLI, without torch or hydra.

A run is a directory ``<runs>/<batch>/<run_id>/`` of the run contract:
``job.json`` (always), ``evals.jsonl`` (one flat line per evaluation, with
``t`` and ``split``), ``result.json`` (when it completed or diverged),
``manifest.json`` (the mark of completion, written last) and
``provenance/attempt-<n>.json``. ``index`` reads either such a directory of
batches, or the fleet's NAS root, whose ``queue/``, ``claimed/<host>/`` and
``done/`` hold the job files of the runs that have not finished or that left
no directory there (pruned, failed).

    python -m exp.lib.runs table --root /nas/geson/mujou-exp --batch B \\
        --rows config.dynamics.d_s,config.dynamics.trace \\
        --cols config.learner.kind --value roll10/cv10
"""

import argparse
import json
import math
import re
import shlex
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

__all__ = [
    "UNFINISHED",
    "RunRecord",
    "export",
    "index",
    "main",
    "series",
    "table",
    "where",
]

# The statuses of a run that has not finished (``table`` refuses them).
UNFINISHED = frozenset({"queued", "running", "incomplete"})

_SHOWN = frozenset({"completed", "diverged", "pruned"})  # statuses a cell reports
_COPIED = frozenset({"completed", "diverged"})  # statuses the fleet copies to the NAS
_MARKS = {"diverged": "発散", "pruned": "打切"}
_MISSING = "欠"  # a seed without a value: failed, cancelled, or never submitted
_PENDING = "未"  # a seed not finished yet (only with allow_unfinished)
_EMPTY = "—"  # a cell with no run at all
_ABSENT = object()  # a path that does not resolve
_COND = re.compile(r"^([^=!~]+)(!=|~=|=)(.*)$")
_AT = re.compile(r"^t=(.+)$")


@dataclass(frozen=True)
class RunRecord:
    """One run: its directory, its ``job.json``, its ``result.json`` (None
    until it completes or diverges), and its ``status``.

    ``status`` is the fleet's when ``done/`` records anything but completed or
    diverged (``"pruned"``, ``"failed"``, ``"corrupt"``, ...; a corrupt copy
    keeps its directory and manifest), or when the job is in ``queue/``
    (``"queued"``) or ``claimed/`` (``"running"``); else ``result.json``'s
    (``"completed"``, ``"diverged"``) when the directory holds a
    ``manifest.json``; else ``"incomplete"`` for a run ``done/`` calls
    completed or diverged, or ``"running"``.
    """

    dir: Path
    job: dict[str, Any]
    result: dict[str, Any] | None
    status: str = "completed"

    @property
    def run_id(self) -> str:
        return str(self.job["run_id"])

    @property
    def batch(self) -> str:
        return str(self.job.get("batch"))

    @property
    def seed(self) -> int:
        return int(self.job["seed"])

    @property
    def snap(self) -> str:
        """The snapshot of the code (40 hex), ``"?"`` if unrecorded."""
        return str(self.job.get("code", {}).get("snap", "?"))

    @property
    def mujou(self) -> str:
        """The snapshot of mujou (40 hex), ``"?"`` if unrecorded."""
        return str(self.job.get("code", {}).get("mujou_snap", "?"))

    @property
    def dirty(self) -> bool:
        """Whether the snapshot held uncommitted changes."""
        return bool(self.job.get("code", {}).get("dirty", False))

    @cached_property
    def measured_device_class(self) -> str | None:
        """Where the run ran by its provenance: ``"cpu"`` or the GPU's model
        (``"4090"``); None without one (a pruned or failed run has no
        directory on the NAS).

        From the last ``provenance/attempt-<n>.json``: its ``device_class``,
        else ``"cpu"`` for a ``device`` of ``"cpu"``, else the 4 digits of the
        name in ``gpu``.
        """
        attempts = sorted(
            self.dir.glob("provenance/attempt-*.json"), key=_attempt_number
        )
        if not attempts:
            return None
        prov = _read_json(attempts[-1])
        if "device_class" in prov:
            return str(prov["device_class"])
        if str(prov.get("device", "")).startswith("cpu"):
            return "cpu"
        gpu = prov.get("gpu")
        if isinstance(gpu, str):
            model = re.search(r"\d{4}", gpu)
            return model.group() if model else gpu
        return None

    @property
    def device_class(self) -> str:
        """``measured_device_class``, else the class the job asked for
        (``job.res.class``, which may be ``"any_gpu"``)."""
        measured = self.measured_device_class
        if measured is not None:
            return measured
        return str(self.job.get("res", {}).get("class", "?"))

    def evals(self, split: str = "val") -> list[dict[str, Any]]:
        """The lines of ``evals.jsonl`` of ``split``, in the order written."""
        return [row for row in self._eval_lines() if row.get("split") == split]

    def at(self, when: str = "last", split: str = "test") -> dict[str, Any] | None:
        """The metrics of ``split`` at ``when``: ``"last"`` (``result.json``'s
        ``final[split]``, else the last evaluation) or ``"t=600"`` (the
        evaluation at 600 s of virtual time). None if there is none.
        """
        if when == "last":
            final = (self.result or {}).get("final", {}).get(split)
            if final is not None:
                return final
            rows = self.evals(split)
            return rows[-1] if rows else None
        target = _parse_at(when)
        for row in self.evals(split):
            if math.isclose(float(row["t"]), target, rel_tol=1e-9, abs_tol=1e-6):
                return row
        return None

    def _eval_lines(self) -> list[dict[str, Any]]:
        path = self.dir / "evals.jsonl"
        if not path.is_file():
            return []
        lines = path.read_text().splitlines()
        rows: list[dict[str, Any]] = []
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if i == len(lines) - 1:  # a live copy may end mid-line
                    break
                raise
        return rows


def index(root: Path, *, batch: str | None = None) -> list[RunRecord]:
    """The runs under ``root``, sorted by ``(batch, run_id)``; only ``batch``'s
    if given.

    ``root`` is a directory of ``<batch>/<run_id>/`` (the Mac's ``runs/``), or
    the fleet's NAS root (it has ``runs/``, ``queue/``, ``claimed/`` or
    ``done/``). On the NAS root a job in ``queue/`` is ``"queued"`` whatever
    else is there, a job in ``claimed/<host>/`` is ``"running"`` unless
    ``done/`` has it, and a job ``done/`` records as anything but completed
    or diverged takes that status, whatever its directory holds.
    """
    root = Path(root)
    fleet = any((root / d).is_dir() for d in ("runs", "queue", "claimed", "done"))
    runs_dir = root / "runs" if fleet else root
    done = _jobs(root / "done", batch) if fleet else {}
    found: dict[tuple[str, str], RunRecord] = {}
    for path in sorted(runs_dir.glob("*/*/job.json")):
        job = _read_json(path)
        if batch is not None and job.get("batch") != batch:
            continue
        d = path.parent
        result = (
            _read_json(d / "result.json") if (d / "result.json").is_file() else None
        )
        entry = done.get(_key(job))
        if entry is not None and entry.get("status") not in _COPIED:
            status = _done_status(entry)  # e.g. a corrupt copy with its manifest
        elif result is not None and (d / "manifest.json").is_file():
            status = str(result.get("status", "completed"))
        elif entry is not None:
            status = _done_status(entry)
        else:
            status = "running"
        found[_key(job)] = RunRecord(d, job, result, status)
    if fleet:
        for key, entry in done.items():
            if key not in found:
                d = runs_dir / key[0] / key[1]
                found[key] = RunRecord(d, entry, None, _done_status(entry))
        active = [("running", _jobs(root / "claimed", batch, hosts=True))]
        active.append(("queued", _jobs(root / "queue", batch)))
        for status, jobs in active:
            for key, job in jobs.items():
                if status == "running" and key in done:
                    continue  # the worker is moving the claim to done/
                old = found.get(key)
                d = old.dir if old else runs_dir / key[0] / key[1]
                found[key] = RunRecord(d, old.job if old else job, None, status)
    return sorted(found.values(), key=lambda r: (r.batch, r.run_id))


def where(runs: Sequence[RunRecord], *conds: str) -> list[RunRecord]:
    """The runs that meet every condition.

    A condition is ``path=value``, ``path!=value`` or ``path~=value``. A path
    is dotted into ``job.json`` (``config.learner.kind``, ``seed``,
    ``code.dirty``), or into ``result.json`` after ``result.``, or is
    ``status`` or ``device_class``. ``=`` compares the value's text as the
    table prints it (``true``, ``null``, ``64``), or numbers as numbers
    (``lr=0.02``, ``d_s=64.0``); ``~=`` is membership in a list (``tags~=confirm``)
    or a substring of a string. A path that does not resolve fails ``=`` and
    ``~=`` and passes ``!=``.
    """
    parsed: list[tuple[str, str, str]] = []
    for cond in conds:
        m = _COND.match(cond)
        if m is None:
            raise ValueError(
                f"a condition is path=value, path!=value or path~=value, not {cond!r}"
            )
        parsed.append((m[1].strip(), m[2], m[3].strip()))
    return [r for r in runs if all(_meets(r, *c) for c in parsed)]


def table(
    runs: Iterable[RunRecord],
    *,
    rows: Sequence[str],
    cols: Sequence[str],
    value: str,
    split: str = "test",
    at: str = "last",
    digits: int = 2,
    allow_mixed: bool = False,
    allow_unfinished: bool = False,
    command: str | None = None,
) -> str:
    """A markdown table of ``value`` (a key of the metrics of ``split`` at
    ``at``, see ``RunRecord.at``) with a row per combination of the values of
    the paths ``rows`` and a column per combination of those of ``cols``
    (paths as in ``where``), and a footnote.

    A cell lists the value of each seed, in the order of the seeds of the
    whole table, as ``"a / b / c"`` with ``digits`` significant figures
    (never in exponent notation: 0.0042, 0.62, 12, 1234): ``"発散"`` for a
    run that diverged, ``"打切"`` for one pruned, ``"欠"`` for a seed without a
    value (failed, or not submitted for that cell); nothing is averaged. A
    trailing ``" *"`` marks a cell with a dirty snapshot; ``"—"`` a cell with
    no run.

    The footnote gives the value, the order of the seeds, the batches, the
    snapshots, the mujou SHAs and the device classes, the count of runs by
    status, and ``command`` (the call that made the table).

    Raises ``ValueError`` for a run that has not finished (unless
    ``allow_unfinished``, which prints ``"未"``), for a cell whose reported
    runs mix snapshots, mujou versions or measured device classes (unless
    ``allow_mixed``; a run without provenance, such as a pruned one, has no
    measured class and is left out of that check), for two runs of the same
    seed in one cell (a row or a column is missing), and when no completed
    run has ``value``.
    """
    runs = list(runs)
    if not runs:
        raise ValueError("no runs to tabulate")
    if digits < 1:
        raise ValueError(f"digits is a count of significant figures, not {digits}")
    if not allow_unfinished:
        waiting = [r.run_id for r in runs if r.status in ("queued", "running")]
        broken = [r.run_id for r in runs if r.status == "incomplete"]
        reasons = []
        if waiting:
            reasons.append(f"{len(waiting)} run(s) have not finished: {_ids(waiting)}")
        if broken:
            reasons.append(
                f"{len(broken)} run(s) are done but their directory has no "
                f"manifest.json (requeue them): {_ids(broken)}"
            )
        if reasons:
            raise ValueError("; ".join(reasons) + "; or pass allow_unfinished")

    cells: dict[tuple[tuple[str, ...], tuple[str, ...]], dict[int, RunRecord]] = {}
    sort_keys: dict[str, tuple[Any, ...]] = {}
    for r in runs:
        key = (_labels(r, rows, sort_keys), _labels(r, cols, sort_keys))
        cell = cells.setdefault(key, {})
        if r.seed in cell:
            raise ValueError(
                f"cell {_cell_name(rows, cols, key)} has two runs of seed {r.seed}: "
                f"{cell[r.seed].run_id}, {r.run_id}; split them with rows or cols, "
                "or filter with where()"
            )
        cell[r.seed] = r

    mixed = _mixed(cells, rows, cols)
    if mixed and not allow_mixed:
        raise ValueError("; ".join(mixed) + "; pass allow_mixed to tabulate anyway")

    seeds = sorted({r.seed for r in runs})
    completed = [r for r in runs if r.status == "completed"]
    if completed and all(_value(r, value, split, at) is None for r in completed):
        raise ValueError(f"no completed run has {value!r} in split {split!r} at {at!r}")

    def order(labels: tuple[str, ...]) -> tuple[Any, ...]:
        return tuple(sort_keys[label] for label in labels)

    row_keys = sorted({k[0] for k in cells}, key=order)
    col_keys = sorted({k[1] for k in cells}, key=order)
    header = [_short(p) for p in rows] + [
        ", ".join(f"{_short(p)}={v}" for p, v in zip(cols, c, strict=True)) or value
        for c in col_keys
    ]
    lines = [_md_row(header), _md_row(["---"] * len(header))]
    for rk in row_keys:
        body = [
            _cell(cells.get((rk, ck)), seeds, value, split, at, digits)
            for ck in col_keys
        ]
        lines.append(_md_row([*rk, *body]))

    shown = [r for r in runs if r.status in _SHOWN]
    counts = Counter(r.status for r in runs)
    if command is None:
        command = (
            f"runs.table(rows={list(rows)!r}, cols={list(cols)!r}, value={value!r}, "
            f"split={split!r}, at={at!r})"
        )
    notes = [
        f"値: `{value}`（{split}、{at}）。セルの並びは seed {' / '.join(map(str, seeds))}",
        "batch " + "、".join(f"`{b}`" for b in sorted({r.batch for r in runs})),
        f"snapshot {_shas(r.snap for r in shown)}、mujou {_shas(r.mujou for r in shown)}、"
        f"デバイス {'、'.join(sorted(_measured(shown))) or '-'}",
        f"{len(runs)} run: " + "、".join(f"{s} {n}" for s, n in sorted(counts.items())),
    ]
    if any(r.dirty for r in shown):
        notes.append("`*`: 汚れた snapshot（commit されていない変更を含む）")
    if mixed:
        notes.append("混在を許した: " + "; ".join(mixed))
    notes.append(f"コマンド: `{command}`")
    return "\n".join([*lines, "", *(f"- {n}" for n in notes)]) + "\n"


def series(run: RunRecord, key: str, split: str = "val") -> list[tuple[float, float]]:
    """``(t, value)`` of ``key`` over the evaluations of ``split``, in time
    order; evaluations without a number for ``key`` are left out."""
    return _series(run.evals(split), key)


def export(runs: Iterable[RunRecord], keys: Sequence[str]) -> dict[str, Any]:
    """The runs as plain data for ``report.html``: per run its identity, its
    config and code, its status and device class, ``final[split][key]`` from
    ``result.json``, and ``series[split][key]`` as ``[[t, value], ...]`` for
    every split in ``evals.jsonl``. JSON-serializable."""
    out: list[dict[str, Any]] = []
    for r in runs:
        lines = r._eval_lines()
        splits = sorted({str(row.get("split")) for row in lines})
        final = (r.result or {}).get("final", {})
        code = r.job.get("code", {})
        out.append(
            {
                "run_id": r.run_id,
                "batch": r.batch,
                "group": r.job.get("group"),
                "seed": r.seed,
                "status": r.status,
                "device_class": r.device_class,
                "tags": r.job.get("tags", []),
                "overrides": r.job.get("overrides", []),
                "config": r.job.get("config", {}),
                "code": {k: code.get(k) for k in ("snap", "mujou_snap", "dirty")},
                "result": None
                if r.result is None
                else {
                    k: r.result.get(k)
                    for k in ("status", "reason", "t_end", "objective")
                },
                "final": {
                    s: {k: m[k] for k in keys if k in m} for s, m in final.items()
                },
                "series": {
                    s: {
                        k: [
                            list(p)
                            for p in _series(
                                [x for x in lines if x.get("split") == s], k
                            )
                        ]
                        for k in keys
                    }
                    for s in splits
                },
            }
        )
    return {"schema": 1, "keys": list(keys), "runs": out}


def main(argv: Sequence[str] | None = None, *, command: str | None = None) -> int:
    """``python -m exp.lib.runs table ...``: print a table (see ``table``).

    ``command`` is the command line the footnote shows (a CLI that forwards
    here passes what its user typed); by default ``python -m exp.lib.runs``
    and ``argv``. Returns 0, or 2 when the table is refused or a condition or
    a file cannot be read (the reason on stderr)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="python -m exp.lib.runs")
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("table", help="a markdown table of a metric over the runs")
    t.add_argument(
        "--root", type=Path, default=Path("runs"), help="runs/ or the NAS root"
    )
    t.add_argument("--batch", default=None)
    t.add_argument("--where", action="append", default=[], metavar="COND")
    t.add_argument("--rows", type=_csv, default=[])
    t.add_argument("--cols", type=_csv, default=[])
    t.add_argument("--value", required=True)
    t.add_argument("--split", default="test")
    t.add_argument("--at", default="last")
    t.add_argument("--digits", type=int, default=2, help="significant figures")
    t.add_argument("--allow-mixed", action="store_true")
    t.add_argument("--allow-unfinished", action="store_true")
    args = parser.parse_args(argv)
    if command is None:
        command = shlex.join(["python", "-m", "exp.lib.runs", *argv])
    try:
        runs = where(index(args.root, batch=args.batch), *args.where)
        text = table(
            runs,
            rows=args.rows,
            cols=args.cols,
            value=args.value,
            split=args.split,
            at=args.at,
            digits=args.digits,
            allow_mixed=args.allow_mixed,
            allow_unfinished=args.allow_unfinished,
            command=command,
        )
    except ValueError as e:
        print(f"runs table: {e}", file=sys.stderr)
        return 2
    sys.stdout.write(text)
    return 0


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _attempt_number(path: Path) -> int:
    m = re.search(r"attempt-(\d+)", path.name)
    return int(m[1]) if m else -1


def _key(job: dict[str, Any]) -> tuple[str, str]:
    return (str(job.get("batch")), str(job["run_id"]))


def _jobs(
    d: Path, batch: str | None, *, hosts: bool = False
) -> dict[tuple[str, str], dict[str, Any]]:
    """The job files in ``d`` (or in ``d/<host>/``) by ``(batch, run_id)``."""
    paths = d.glob("*/*.json") if hosts else d.glob("*.json")
    jobs: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(paths):
        job = _read_json(path)
        if "run_id" in job and (batch is None or job.get("batch") == batch):
            jobs[_key(job)] = job
    return jobs


def _done_status(entry: dict[str, Any]) -> str:
    """The status of a run from its ``done/`` entry when its directory does
    not give it: a completed or diverged run without a manifest is
    ``"incomplete"``."""
    status = str(entry.get("status", "?"))
    return "incomplete" if status in ("completed", "diverged") else status


def _parse_at(when: str) -> float:
    m = _AT.match(when)
    if m is None:
        raise ValueError(f"when is 'last' or 't=<seconds>', not {when!r}")
    return float(m[1])


def _lookup(run: RunRecord, path: str) -> Any:
    if path == "status":
        return run.status
    if path == "device_class":
        return run.device_class
    head, _, rest = path.partition(".")
    node: Any = run.job
    parts = path.split(".")
    if head == "result":
        node, parts = run.result, rest.split(".") if rest else []
    for part in parts:
        if not isinstance(node, dict) or part not in node:
            return _ABSENT
        node = node[part]
    return node


def _label(v: Any) -> str:
    """``v`` as text, as an override would spell it."""
    if v is _ABSENT or v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float, str)):
        return str(v)
    return json.dumps(v, sort_keys=True, ensure_ascii=False)


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _sort_key(v: Any) -> tuple[Any, ...]:
    """Numbers by value, then booleans, then text, then null."""
    if _is_number(v):
        return (0, float(v))
    if isinstance(v, bool):
        return (1, v)
    if v is _ABSENT or v is None:
        return (3, "")
    return (2, _label(v))


def _equal(actual: Any, text: str) -> bool:
    if _label(actual) == text:
        return True
    if not _is_number(actual):
        return False
    try:
        return float(actual) == float(text)
    except ValueError:
        return False


def _meets(run: RunRecord, path: str, op: str, text: str) -> bool:
    actual = _lookup(run, path)
    if op == "~=":
        if isinstance(actual, list):
            return any(_equal(x, text) for x in actual)
        return isinstance(actual, str) and text in actual
    hit = actual is not _ABSENT and _equal(actual, text)
    return hit if op == "=" else not hit


def _labels(
    run: RunRecord, paths: Sequence[str], sort_keys: dict[str, tuple[Any, ...]]
) -> tuple[str, ...]:
    out: list[str] = []
    for p in paths:
        v = _lookup(run, p)
        label = _label(v)
        sort_keys[label] = _sort_key(v)
        out.append(label)
    return tuple(out)


def _short(path: str) -> str:
    return path.removeprefix("config.")


def _cell_name(
    rows: Sequence[str],
    cols: Sequence[str],
    key: tuple[tuple[str, ...], tuple[str, ...]],
) -> str:
    pairs = zip([*rows, *cols], [*key[0], *key[1]], strict=True)
    return "(" + ", ".join(f"{_short(p)}={v}" for p, v in pairs) + ")"


def _mixed(
    cells: dict[tuple[tuple[str, ...], tuple[str, ...]], dict[int, RunRecord]],
    rows: Sequence[str],
    cols: Sequence[str],
) -> list[str]:
    """What each cell mixes among the runs it reports."""
    out: list[str] = []
    for key, cell in cells.items():
        shown = [r for r in cell.values() if r.status in _SHOWN]
        for what, values in (
            ("snapshots", {r.snap for r in shown}),
            ("mujou versions", {r.mujou for r in shown}),
            ("device classes", _measured(shown)),
        ):
            if len(values) > 1:
                out.append(
                    f"cell {_cell_name(rows, cols, key)} mixes {what} "
                    + ", ".join(v[:12] for v in sorted(values))
                )
    return out


def _value(run: RunRecord, value: str, split: str, at: str) -> Any:
    metrics = run.at(at, split)
    return None if metrics is None else metrics.get(value)


def _cell(
    cell: dict[int, RunRecord] | None,
    seeds: Sequence[int],
    value: str,
    split: str,
    at: str,
    digits: int,
) -> str:
    if not cell:
        return _EMPTY
    marks: list[str] = []
    for seed in seeds:
        r = cell.get(seed)
        if r is None:
            marks.append(_MISSING)
        elif r.status in _MARKS:
            marks.append(_MARKS[r.status])
        elif r.status in UNFINISHED:
            marks.append(_PENDING)
        elif r.status == "completed":
            v = _value(r, value, split, at)
            marks.append(_MISSING if v is None else _number(v, digits))
        else:
            marks.append(_MISSING)
    dirty = any(r.dirty for r in cell.values() if r.status in _SHOWN)
    return " / ".join(marks) + (" *" if dirty else "")


def _number(v: Any, digits: int) -> str:
    """``v`` to ``digits`` significant figures in fixed notation."""
    if not _is_number(v):
        return _label(v)
    x = float(v)
    if not math.isfinite(x):
        return str(x)
    if x == 0:
        return f"{x:.{max(digits - 1, 0)}f}"
    exponent = math.floor(math.log10(abs(x)))
    if abs(round(x, digits - 1 - exponent)) >= 10 ** (exponent + 1):
        exponent += 1  # 0.0996 rounds up to 0.10
    return f"{x:.{max(digits - 1 - exponent, 0)}f}"


def _measured(runs: Iterable[RunRecord]) -> set[str]:
    """The device classes the runs' provenance records."""
    return {c for r in runs if (c := r.measured_device_class) is not None}


def _ids(run_ids: Sequence[str]) -> str:
    return ", ".join(run_ids[:5]) + (" ..." if len(run_ids) > 5 else "")


def _shas(shas: Iterable[str]) -> str:
    unique = sorted({s[:12] for s in shas})
    return "、".join(f"`{s}`" for s in unique) or "-"


def _md_row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |"


def _csv(text: str) -> list[str]:
    return [p.strip() for p in text.split(",") if p.strip()]


def _series(rows: Sequence[dict[str, Any]], key: str) -> list[tuple[float, float]]:
    return [
        (float(row["t"]), float(row[key])) for row in rows if _is_number(row.get(key))
    ]


if __name__ == "__main__":
    raise SystemExit(main())
