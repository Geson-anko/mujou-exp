"""The contract of a run's directory: the files a run writes, how it writes
them so that a crash leaves no half-written file, the exit codes, the stop
request and the manifest that marks a run complete.

A run writes JSON files whole and atomically (``write_json``, by
``atomic_open``: a temporary file, then ``os.replace``) and the ``.jsonl`` files
line by line (``Jsonl``: each line flushed, ``fsync`` at an evaluation and at a
resume checkpoint). A resumed run cuts its metrics files (``RunDir.metrics``,
opened yet or not) back to the byte offsets recorded in the checkpoint
(``RunDir.truncate``), so that its lines are those of a run never stopped. The last file it writes is ``manifest.json`` (``write_manifest``):
without it a run is not complete, whatever its exit code.

``manifest.json`` is ``{"schema": 1, "files": {path: {"sha256": hex, "size":
bytes}}}`` with the paths relative to the run's directory, in POSIX form and
sorted, of every file but those of ``UNLISTED``, those below
``UNLISTED_DIRS`` and the temporary ``.*.tmp`` of a write under way.

Standard library only.
"""

import hashlib
import json
import os
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from enum import IntEnum
from pathlib import Path
from types import TracebackType
from typing import IO, Any, Self

__all__ = [
    "MANIFEST",
    "MANIFEST_SCHEMA",
    "METRICS",
    "STOP_REQUEST",
    "UNLISTED",
    "UNLISTED_DIRS",
    "EnvError",
    "Exit",
    "Jsonl",
    "ManifestError",
    "RunDir",
    "atomic_open",
    "verify_manifest",
    "write_json",
    "write_manifest",
]

MANIFEST = "manifest.json"
MANIFEST_SCHEMA = 1
STOP_REQUEST = "stop.request"
# Written by the worker or the wall clock, or local only: not in the manifest.
UNLISTED = frozenset({MANIFEST, STOP_REQUEST, "status.json", "timing.jsonl"})
UNLISTED_DIRS = frozenset({"resume", "logs", "exit"})
# The metrics files of a run: functions of (code, settings), cut back on resume.
METRICS = ("stream.jsonl", "evals.jsonl")


class Exit(IntEnum):
    """The exit code of a run, as the worker reads it."""

    OK = 0  # completed: result.json and manifest.json written
    ERROR = 1  # an exception: an error of the code, not retried
    DIVERGED = 3  # a result of the science: result.json and manifest.json
    ENV = 4  # the environment is wrong (GIL, OMP, CUDA, data): retried elsewhere
    STOPPED = 75  # stopped on request, after a resume checkpoint


class EnvError(RuntimeError):
    """The environment cannot run this job (the GIL is enabled, a thread count
    is wrong, a device or the data is missing): the run exits with
    ``Exit.ENV``, and the worker puts the job back for another host."""


class ManifestError(ValueError):
    """The files of a run do not match its manifest."""


def _is_temporary(name: str) -> bool:
    """Whether ``name`` is the temporary file of a write under way."""
    return name.startswith(".") and name.endswith(".tmp")


@contextmanager
def atomic_open(path: Path) -> Iterator[IO[bytes]]:
    """A binary file to write the new content of ``path`` to.

    It is ``.<name>.<pid>.tmp`` of ``path``'s directory, created with the
    umask's mode as ``open`` does; on leaving the block it is synced and moved
    to ``path`` (``os.replace``), so that a reader sees the old file or the new
    one, never a part. On an exception it is removed and ``path`` is left as
    it was.
    """
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("wb") as f:
            yield f
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write_json(path: Path, obj: Any) -> None:
    """Write ``obj`` as indented JSON to ``path`` atomically (``atomic_open``)."""
    data = (json.dumps(obj, indent=2) + "\n").encode()
    with atomic_open(path) as f:
        f.write(data)


class Jsonl:
    """A ``.jsonl`` file a run appends to: one JSON object per line, each line
    flushed to the OS as it is written.

    A line is ``json.dumps`` of the record with its keys in their order, so the
    same records make the same bytes. ``sync`` makes the lines durable and
    returns the offset to cut back to on resume.
    """

    def __init__(self, path: Path) -> None:
        self._file: IO[bytes] = path.open("ab")

    def append(self, record: Mapping[str, Any]) -> None:
        """Write ``record`` as one line and flush it."""
        self._file.write(json.dumps(record).encode() + b"\n")
        self._file.flush()

    def sync(self) -> int:
        """``fsync`` the file and return its size in bytes."""
        self._file.flush()
        os.fsync(self._file.fileno())
        return self._file.tell()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class RunDir:
    """The directory of one run (``job.json`` in it is the worker's).

    It opens the ``.jsonl`` files for appending (``jsonl``), cuts them back on
    resume (``truncate``), writes the JSON files atomically (``write_json``),
    tells whether the worker asked the run to stop (``stop_requested``) and
    writes the manifest (``write_manifest``).

    ``metrics`` names the ``.jsonl`` files the manifest lists. They are fixed
    up front, so that a resume checkpoint records an offset for each, also for
    one the run has not opened yet (a checkpoint before the first evaluation),
    and a resumed run cuts each back. A ``.jsonl`` file the run opens must be
    one of them, or one of ``UNLISTED`` (``timing.jsonl``, appended to across
    attempts and never cut back).
    """

    def __init__(self, root: Path, *, metrics: Iterable[str] = METRICS) -> None:
        self.root = root
        self.metrics = tuple(metrics)
        self._open: dict[str, Jsonl] = {}

    def read_job(self) -> dict[str, Any]:
        """The ``job.json`` of the run."""
        return json.loads((self.root / "job.json").read_text())

    def write_json(self, name: str, obj: Any) -> None:
        """Write ``obj`` to ``name`` (relative, its directory made) atomically."""
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, obj)

    def jsonl(self, name: str) -> Jsonl:
        """The ``.jsonl`` file ``name``, opened for appending once and kept
        open until ``close``.

        Raises:
            ValueError: ``name`` is neither one of ``metrics`` nor unlisted.
        """
        if name not in self.metrics and name not in UNLISTED:
            raise ValueError(
                f"{name} is not one of the metrics files {self.metrics} nor"
                " unlisted: a resume would not cut it back"
            )
        if name not in self._open:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            self._open[name] = Jsonl(path)
        return self._open[name]

    def sync(self) -> dict[str, int]:
        """``fsync`` every open ``.jsonl`` file and return the sizes in bytes
        of the ``metrics`` files by name, the offsets a resume checkpoint
        records: of a file not opened yet, its size on disk (0 if missing)."""
        sizes = {name: f.sync() for name, f in self._open.items()}
        offsets = {}
        for name in self.metrics:
            path = self.root / name
            if name in sizes:
                offsets[name] = sizes[name]
            else:
                offsets[name] = path.stat().st_size if path.exists() else 0
        return offsets

    def truncate(self, offsets: Mapping[str, int]) -> None:
        """Cut each file of ``offsets`` back to its offset (before opening it),
        dropping the lines written after the checkpoint that recorded them.

        A missing file with offset 0 is left missing: the checkpoint was
        taken before its first line.

        Raises:
            ValueError: A file is shorter than its offset (it lost lines the
                checkpoint counted), or is open already.
        """
        for name, offset in offsets.items():
            if name in self._open:
                raise ValueError(f"{name} is open: truncate before writing")
            path = self.root / name
            size = path.stat().st_size if path.exists() else 0
            if size < offset:
                raise ValueError(f"{name} has {size} bytes, fewer than {offset}")
            if not path.exists():
                continue
            with path.open("r+b") as f:
                f.truncate(offset)
                f.flush()
                os.fsync(f.fileno())

    def stop_requested(self) -> bool:
        """Whether the worker asked the run to stop (``stop.request`` exists):
        then it writes a resume checkpoint and exits with ``Exit.STOPPED``."""
        return (self.root / STOP_REQUEST).exists()

    def close(self) -> None:
        """Close the open ``.jsonl`` files."""
        for f in self._open.values():
            f.close()
        self._open.clear()

    def write_manifest(self) -> dict[str, Any]:
        """Sync and close the ``.jsonl`` files, then write ``manifest.json``
        (the module docstring), the last file of a run; returns it."""
        self.sync()
        self.close()
        return write_manifest(self.root)


def _listed(root: Path) -> list[str]:
    """The paths of the files of ``root`` that a manifest lists, sorted."""
    out = []
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if rel.parts[0] in UNLISTED_DIRS or not path.is_file():
            continue
        if rel.as_posix() in UNLISTED or _is_temporary(path.name):
            continue
        out.append(rel.as_posix())
    return sorted(out)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_manifest(root: Path) -> dict[str, Any]:
    """Write ``root/manifest.json`` (the module docstring) and return it."""
    files = {
        rel: {"sha256": _sha256(root / rel), "size": (root / rel).stat().st_size}
        for rel in _listed(root)
    }
    manifest = {"schema": MANIFEST_SCHEMA, "files": files}
    write_json(root / MANIFEST, manifest)
    return manifest


def verify_manifest(root: Path) -> None:
    """Check every file listed in ``root/manifest.json`` against its size and
    sha256. A file the manifest does not list is not checked.

    Raises:
        ManifestError: The manifest is missing or of another schema, or a
            listed file is missing or differs; the message names every such
            file.
    """
    path = root / MANIFEST
    if not path.exists():
        raise ManifestError(f"no {MANIFEST} in {root}")
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ManifestError(
            f"{path} is of schema {manifest.get('schema')!r}, not {MANIFEST_SCHEMA}"
        )
    problems = []
    for rel, want in manifest["files"].items():
        file = root / rel
        if not file.is_file():
            problems.append(f"{rel}: missing")
        elif file.stat().st_size != want["size"]:
            problems.append(f"{rel}: {file.stat().st_size} bytes, not {want['size']}")
        elif _sha256(file) != want["sha256"]:
            problems.append(f"{rel}: sha256 differs")
    if problems:
        raise ManifestError(f"{root}: " + "; ".join(problems))
