"""The shape of a job (``job.json``, design 3.3) and its names (design 2.4).

``group`` hashes what decides a run's metrics apart from its seed: the
program, the resolved config without ``seed``, ``device``, ``ctl`` and
``phase``, the code snapshot and the resource class. ``run_id`` is
``<name>-<group[:8]>-s<seed>``, so submitting the same thing twice names the
same run and the second is skipped as a duplicate.
"""

import hashlib
import itertools
import json
import re
import subprocess
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_ENTRY",
    "GPU_ANY",
    "canonical_json",
    "compose_configs",
    "expand",
    "group_of",
    "label_of",
    "make_job",
    "parse_set",
    "run_id_of",
    "seed_of",
]

DEFAULT_ENTRY = "exp.video.main"
GPU_ANY = "any_gpu"
_ENTRY = re.compile(r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)*")
_HASH_DROP = ("seed", "device", "ctl", "phase")


def canonical_json(obj: Any) -> str:
    """``obj`` as JSON with sorted keys and no spaces: equal values, equal text."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def group_of(
    *,
    program: str,
    config: Mapping[str, Any],
    snap: str,
    res_class: str,
    entry: str = DEFAULT_ENTRY,
) -> str:
    """12 hex of ``sha256(canonical_json({program, config without seed, device,
    ctl, phase, snap, res_class}))``. A non-default ``entry`` joins the hash."""
    key: dict[str, Any] = {
        "program": program,
        "config": {k: v for k, v in config.items() if k not in _HASH_DROP},
        "snap": snap,
        "res_class": res_class,
    }
    if entry != DEFAULT_ENTRY:
        key["entry"] = entry
    return hashlib.sha256(canonical_json(key).encode()).hexdigest()[:12]


def run_id_of(name: str, group: str, seed: int) -> str:
    """``f"{name with '/' as '-'}-{group[:8]}-s{seed}"``, e.g. ``m14-s1-3fa29c01-s0``."""
    return f"{name.replace('/', '-')}-{group[:8]}-s{seed}"


def label_of(overrides: Sequence[str]) -> str:
    """A short name to read, from the overrides other than ``seed`` and the
    presets: ``["learner=pc", "dynamics=big", "optim.lr=0.02"]`` ->
    ``"pc big lr0.02"``. Not used to identify anything."""
    parts: list[str] = []
    for o in overrides:
        key, _, value = o.partition("=")
        key = key.lstrip("+~")
        if key == "seed" or key.startswith("experiment") or not value:
            continue
        last = key.rsplit(".", 1)[-1]
        parts.append(value if "." not in key and "@" not in key else f"{last}{value}")
    return " ".join(parts)


def seed_of(overrides: Sequence[str], default: int = 0) -> int:
    """The value of the last ``seed=`` override, else ``default``."""
    seed = default
    for o in overrides:
        if o.startswith("seed="):
            seed = int(o.removeprefix("seed="))
    return seed


def expand(
    overrides: Sequence[str],
    sweep: Sequence[str] = (),
    seeds: Sequence[int] | None = None,
) -> list[list[str]]:
    """The override lists of a submission: ``overrides``, then one ``k=v`` for
    each sweep ``"k=a,b,..."`` (the cartesian product, the first sweep varying
    slowest), then ``seed=s`` for each of ``seeds`` (the fastest)."""
    axes: list[list[str]] = []
    for s in sweep:
        key, sep, values = s.partition("=")
        if not sep or not values:
            raise ValueError(f"a sweep is 'key=a,b,...', not {s!r}")
        axes.append([f"{key}={v}" for v in values.split(",")])
    if seeds is not None:
        axes.append([f"seed={s}" for s in seeds])
    return [list(overrides) + list(combo) for combo in itertools.product(*axes)]


def parse_set(items: Sequence[str]) -> dict[str, Any]:
    """``["fake.exit=3", "tags=[\\"x\\"]"]`` -> ``{"fake": {"exit": 3}, "tags":
    ["x"]}``: dotted keys, values parsed as JSON (else kept as strings)."""
    out: dict[str, Any] = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"a --set is 'key=value', not {item!r}")
        try:
            value: Any = json.loads(raw)
        except ValueError:
            value = raw
        node = out
        *parents, last = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[last] = value
    return out


def make_job(
    *,
    batch: str,
    config: Mapping[str, Any],
    overrides: Sequence[str],
    code: Mapping[str, Any],
    res: Mapping[str, Any],
    entry: str = DEFAULT_ENTRY,
    prio: int = 50,
    after: Sequence[str] = (),
    host: str | None = None,
    avoid_host: str | None = None,
    tags: Sequence[str] = (),
    retry: Mapping[str, int] | None = None,
    study: str | None = None,
    trial: int | None = None,
    extra: Mapping[str, Any] | None = None,
    submitted: str | None = None,
) -> dict[str, Any]:
    """A job (the content of ``job.json``) for the resolved ``config``, run by
    ``python -m <entry>`` of the snapshot ``code["snap"]`` on a slot of
    ``res["class"]``. ``extra`` adds top-level keys (e.g. ``fake`` for the fake
    program); it may not replace the keys made here."""
    if not _ENTRY.fullmatch(entry):
        raise ValueError(f"entry must be a module name, not {entry!r}")
    if not 0 <= prio <= 999:
        raise ValueError(f"prio must be in 0..999, not {prio}")
    program = str(config.get("program", "fake"))
    seed = int(config.get("seed", 0))
    group = group_of(
        program=program,
        config=config,
        snap=code["snap"],
        res_class=res["class"],
        entry=entry,
    )
    job: dict[str, Any] = {
        "schema": 1,
        "run_id": run_id_of(str(config["name"]), group, seed),
        "batch": batch,
        "group": group,
        "program": program,
        "name": str(config["name"]),
        "label": label_of(overrides),
        "seed": seed,
        "entry": entry,
        "overrides": list(overrides),
        "config": dict(config),
        "code": dict(code),
        "res": dict(res),
        "prio": prio,
        "after": list(after),
        "host": host,
        "avoid_host": avoid_host,
        "study": study,
        "trial": trial,
        "tags": list(tags),
        "retry": dict(retry or {"oom": 1, "lost": 3}),
        "retried": {"oom": 0, "lost": 0},
        "attempt": 1,
        "submitted": submitted or datetime.now().astimezone().isoformat(),
    }
    for k, v in (extra or {}).items():
        if k in job:
            raise ValueError(f"--set may not replace the job's {k!r}")
        job[k] = v
    return job


# Run by the venv python of a snapshot, in that snapshot: hydra stays out of the
# fleet's own process. stdin: a JSON list of override lists; stdout: a JSON
# list of {"config": dict} or {"error": str}.
_COMPOSE_SRC = """
import json, sys
from exp.video.compose import compose, to_dict
out = []
for overrides in json.load(sys.stdin):
    try:
        out.append({"config": to_dict(compose(overrides))})
    except Exception as e:
        out.append({"error": f"{type(e).__name__}: {e}"})
json.dump(out, sys.stdout)
"""


def compose_configs(
    python: Path, cwd: Path, override_lists: Sequence[Sequence[str]]
) -> list[dict[str, Any]]:
    """The resolved config of each override list, composed by ``exp.video.compose``
    under ``python`` in the snapshot ``cwd``. ``ValueError`` lists the lists
    that do not compose."""
    proc = subprocess.run(
        [str(python), "-c", _COMPOSE_SRC],
        cwd=cwd,
        input=json.dumps([list(o) for o in override_lists]),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"compose failed:\n{proc.stderr}")
    results = json.loads(proc.stdout)
    errors = [
        f"{' '.join(o)}: {r['error']}"
        for o, r in zip(override_lists, results, strict=True)
        if "error" in r
    ]
    if errors:
        raise ValueError("these do not compose:\n" + "\n".join(errors))
    return [r["config"] for r in results]
