"""The run directories of the fleet (``exp/lib/fleet/README.md``) as TensorBoard
event files: one directory of events per run, ``<logdir>/<batch>/<run_id>/``.

A run directory gives (each only when its file is there)

- ``stream/<key>``: the lines of ``stream.jsonl``, at step ``frames``;
- ``eval/<split>/<key>``: the lines of ``evals.jsonl``, at step ``round(t)``;
- ``base/<split>/run_init/<key>``: ``baselines.json`` (the vision before it
  learns, ``init_*``), at step 0 and at the last step (a level line);
- the text ``run``: the run_id, the batch, the snapshot, the mujou SHA, the
  data version, the seed and the overrides (``job.json``, ``CODE.json``,
  ``provenance/``);
- the hyperparameters (the HParams tab): the config's ``vision.*``,
  ``optim_vision.*``, ``learner.kind``, ``seed`` (and ``name``), with the
  last val numbers as its metrics (``hp/<key>``) and those of the last
  ``vit`` val line of its posthoc (``hp/post/<key>``: ``objective``,
  ``recon``, ``dyn_1``, ...) when the posthoc is in the same export (or in the
  run's own ``post/decode``).

The directory of a ``task=posthoc`` job (its ``job.json`` names a
``parent``) goes into its parent's directory of events (the parent's batch and
run_id: of its ``job.json`` when the parent is here, else the last two parts
of its path, ``<batch>/<run_id>`` on the NAS), so a run and its evaluation
afterwards show together:

- ``post/<split>/<key>``: the ``vit`` lines of ``post/decode/metrics.jsonl``,
  at step ``round(t)``;
- ``base/<split>/<code>/<key>``: the lines of ``post/decode/baselines.jsonl``
  (``pca``, ``init``), at step 0 and at the last step of the checkpoints;
- images ``post/<kind>`` (``recon``, ``restore``, ``dyn``) of ``<t>*.png`` at
  step ``round(t)``, and ``base/<code>/<kind>`` of ``init*.png`` and
  ``pca*.png`` at step 0, the PNG bytes as they are.

Non-numeric values and nulls are left out. The events are written anew: an
export removes the event files it wrote before into the same directory.
"""

import json
import math
import re
import struct
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from tensorboardX import SummaryWriter
from tensorboardX.proto import summary_pb2
from tensorboardX.summary import hparams

__all__ = ["HPARAM_PREFIXES", "export", "export_run", "png_size"]

HPARAM_PREFIXES = ("vision.", "optim_vision.", "learner.kind", "seed", "name")
# protobuf's generated module has no stubs: its message classes are untyped.
Summary: Any = vars(summary_pb2)["Summary"]
_SUFFIX = {"run": ".run", "post": ".post"}  # the event files of an export
_STAMP = re.compile(r"^(\d+(?:\.\d+)?)(?:-(\w+))?$")


def png_size(data: bytes) -> tuple[int, int]:
    """The width and height of the PNG ``data`` (its IHDR chunk)."""
    if data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise ValueError("not a PNG")
    w, h = struct.unpack(">II", data[16:24])
    return w, h


def _lines(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out = []
    for raw in path.read_text().splitlines():
        try:
            out.append(json.loads(raw))
        except ValueError:
            continue  # a partial last line of a live copy
    return out


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.is_file() else {}


def _numbers(row: Mapping[str, Any], skip: Iterable[str] = ()) -> dict[str, float]:
    """The finite numbers of ``row`` (not bools), but the keys of ``skip``."""
    out = {}
    for k, v in row.items():
        if k in skip or isinstance(v, bool) or not isinstance(v, int | float):
            continue
        if math.isfinite(v):
            out[k] = float(v)
    return out


def _flat(d: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, Mapping):
            out |= _flat(v, f"{key}.")
        else:
            out[key] = v
    return out


def _hparam(v: Any) -> bool | str | float | int:
    if isinstance(v, bool | int | float | str):
        return v
    return json.dumps(v)


def _target(run_dir: Path, job: Mapping[str, Any], logdir: Path) -> Path:
    """The directory of events of ``run_dir``: its parent run's for a posthoc
    job, else ``<batch>/<run_id>``."""
    parent = job.get("parent")
    if isinstance(parent, str) and parent:
        p = Path(parent)
        if not p.is_absolute():  # relative to the runs' root (programs.posthoc)
            p = run_dir.parents[1] / p
        if (p / "job.json").is_file():  # here: its own batch and run_id
            return _target(p, _json(p / "job.json"), logdir)
        return logdir / p.parent.name / p.name  # <batch>/<run_id> on the NAS
    batch = job.get("batch") or run_dir.parent.name
    return logdir / str(batch) / str(job.get("run_id") or run_dir.name)


def _writer(target: Path, kind: str) -> SummaryWriter:
    target.mkdir(parents=True, exist_ok=True)
    for old in target.glob(f"events.out.tfevents.*{_SUFFIX[kind]}"):
        old.unlink()
    return SummaryWriter(str(target), filename_suffix=_SUFFIX[kind], flush_secs=1)


def _add(w: SummaryWriter, summary: Any, step: int | None = None) -> None:
    """Write the protobuf ``summary`` with ``w``'s event file."""
    assert w.file_writer is not None  # a writer that writes to disk
    w.file_writer.add_summary(summary, step)


def _levels(
    w: SummaryWriter, tag: str, row: Mapping[str, float], steps: Sequence[int]
) -> None:
    for k, v in row.items():
        for s in sorted(set(steps)):
            w.add_scalar(f"{tag}/{k}", v, s)


def _image(w: SummaryWriter, tag: str, path: Path, step: int) -> None:
    data = path.read_bytes()
    width, height = png_size(data)
    img = Summary.Image(
        height=height, width=width, colorspace=3, encoded_image_string=data
    )
    _add(w, Summary(value=[Summary.Value(tag=tag, image=img)]), step)


def _text(run_dir: Path, job: Mapping[str, Any]) -> str:
    code = _json(run_dir / "CODE.json") or dict(job.get("code") or {})
    prov = sorted((run_dir / "provenance").glob("attempt-*.json"))
    last = _json(prov[-1]) if prov else {}
    config = dict(job.get("config") or {})
    tours = ((config.get("data") or {}).get("tours") or {}).get("data_version")
    rows = [
        ("run_id", job.get("run_id")),
        ("batch", job.get("batch")),
        ("snapshot", code.get("snap")),
        ("dirty", code.get("dirty")),
        ("mujou", code.get("mujou_snap")),
        ("data_version", last.get("data_version") or tours),
        ("seed", job.get("seed")),
        ("host", last.get("host")),
        ("overrides", " ".join(job.get("overrides") or [])),
    ]
    return "\n".join(f"- {k}: `{v}`" for k, v in rows if v is not None)


def _post_last(run_dir: Path) -> dict[str, float]:
    """The numbers of the last ``vit`` val line of the posthoc in ``run_dir``."""
    lines = [
        x
        for x in _lines(run_dir / "post" / "decode" / "metrics.jsonl")
        if x.get("code", "vit") == "vit" and x.get("split") == "val"
    ]
    return _numbers(lines[-1], ("t", "frame")) if lines else {}


def _run_events(
    run_dir: Path, job: Mapping[str, Any], target: Path, post: Mapping[str, float]
) -> None:
    w = _writer(target, "run")
    try:
        for line in _lines(run_dir / "stream.jsonl"):
            step = int(line.get("frames", 0))
            for k, v in _numbers(line, ("t", "frames")).items():
                w.add_scalar(f"stream/{k}", v, step)
        evals = _lines(run_dir / "evals.jsonl")
        steps = [0]
        for line in evals:
            step = round(float(line["t"]))
            steps.append(step)
            skip = ("t", "frame", "frames", "step")
            for k, v in _numbers(line, skip).items():
                w.add_scalar(f"eval/{line['split']}/{k}", v, step)
        for split, row in _json(run_dir / "baselines.json").items():
            base = {k.removeprefix("init_"): v for k, v in _numbers(row).items()}
            _levels(w, f"base/{split}/run_init", base, [0, max(steps)])
        w.add_text("run", _text(run_dir, job), 0)
        config = _flat(dict(job.get("config") or {}))
        hp = {
            k: _hparam(v)
            for k, v in config.items()
            if k.startswith(HPARAM_PREFIXES) and v is not None
        }
        last = next((r for r in reversed(evals) if r.get("split") == "val"), {})
        metrics = {f"hp/{k}": v for k, v in _numbers(last, ("t", "frame")).items()}
        metrics |= {f"hp/post/{k}": v for k, v in post.items()}
        if hp and metrics:
            for s in hparams(hp, metrics):
                _add(w, s)
            for k, v in metrics.items():
                w.add_scalar(k, v, max(steps))
    finally:
        w.close()


def _post_events(run_dir: Path, target: Path) -> None:
    decode = run_dir / "post" / "decode"
    w = _writer(target, "post")
    try:
        lines = [
            x for x in _lines(decode / "metrics.jsonl") if x.get("code", "vit") == "vit"
        ]
        steps = [0]
        for line in lines:
            step = round(float(line["t"]))
            steps.append(step)
            for k, v in _numbers(line, ("t", "frame")).items():
                w.add_scalar(f"post/{line['split']}/{k}", v, step)
        for line in _lines(decode / "baselines.jsonl"):
            row = _numbers(line, ("t", "frame"))
            _levels(w, f"base/{line['split']}/{line['code']}", row, [0, max(steps)])
        for png in sorted(decode.glob("*.png")):
            stem, _, kind = png.stem.partition("-")
            kind = kind or "recon"
            m = _STAMP.match(png.stem)
            if m is not None:
                _image(w, f"post/{kind}", png, round(float(m.group(1))))
            else:
                _image(w, f"base/{stem}/{kind}", png, 0)
    finally:
        w.close()


def export_run(
    run_dir: Path, logdir: Path, *, post: Mapping[str, float] | None = None
) -> Path:
    """Write the events of the run directory ``run_dir`` (module docstring);
    returns their directory. ``post``: the numbers of its posthoc for the
    HParams (``_post_last``; None: of its own ``post/decode``)."""
    job = _json(run_dir / "job.json")
    target = _target(run_dir, job, logdir)
    if job.get("parent") or (run_dir / "post" / "decode").is_dir():
        _post_events(run_dir, target)
    if not job.get("parent"):
        _run_events(run_dir, job, target, _post_last(run_dir) if post is None else post)
    return target


def export(runs: Iterable[Path], logdir: Path) -> list[Path]:
    """``export_run`` of each run directory of ``runs`` (a directory without
    a ``job.json`` is a batch: each of its run directories)."""
    dirs: list[Path] = []
    for path in runs:
        dirs += (
            [path]
            if (path / "job.json").is_file()
            else sorted(d for d in path.iterdir() if (d / "job.json").is_file())
        )
    posts: dict[Path, dict[str, float]] = {}
    for d in dirs:
        job = _json(d / "job.json")
        if job.get("parent") and (last := _post_last(d)):
            posts[_target(d, job, logdir)] = last  # the last of them wins
    return [
        export_run(d, logdir, post=posts.get(_target(d, _json(d / "job.json"), logdir)))
        for d in dirs
    ]
