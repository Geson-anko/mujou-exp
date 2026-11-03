"""The run contract as the fleet sees it (design 3.1, 3.2, 3.6): the exit codes,
the files of a run directory, and the manifest.

A run writes ``result.json`` and then ``manifest.json`` last; without the
manifest it has not finished, whatever its exit code. The manifest lists the
sha256 and size of every file of the run directory except the ones that change
after the run or never leave the host (``MANIFEST_EXCLUDE``).
"""

import hashlib
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

__all__ = [
    "EXIT_DIVERGED",
    "EXIT_ENV",
    "EXIT_OK",
    "EXIT_STOPPED",
    "MANIFEST_EXCLUDE",
    "copy_run",
    "read_json",
    "sha256_file",
    "verify_manifest",
    "write_json",
    "write_manifest",
]

EXIT_OK = 0  # completed: result.json (status "completed") and manifest.json
EXIT_DIVERGED = 3  # diverged, a result of science: result.json and manifest.json
EXIT_ENV = 4  # the environment is wrong (GIL, OMP, CUDA, data): the job goes back
EXIT_STOPPED = 75  # stopped on request (stop.request), after a checkpoint to resume

# Top-level names left out of the manifest (and resume/ out of the archive).
MANIFEST_EXCLUDE = frozenset(
    {
        "manifest.json",
        "resume",
        "logs",
        "exit",
        "status.json",
        "timing.jsonl",
        "stop.request",
    }
)


def read_json(path: Path) -> Any:
    """The JSON value of ``path``."""
    with path.open("rb") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    """Write ``obj`` to ``path`` atomically: a temporary file in the same
    directory, then ``os.replace``."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    """The hex sha256 of the bytes of ``path``."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _manifested(run_dir: Path) -> Iterator[Path]:
    for top in sorted(run_dir.iterdir()):
        if top.name in MANIFEST_EXCLUDE or top.name.startswith("."):
            continue
        if top.is_dir():
            yield from sorted(p for p in top.rglob("*") if p.is_file())
        else:
            yield top


def write_manifest(run_dir: Path) -> None:
    """Write ``manifest.json``: ``{"schema": 1, "files": {relpath: {"sha256",
    "size"}}}`` of the files of ``run_dir`` outside ``MANIFEST_EXCLUDE``."""
    files = {
        p.relative_to(run_dir).as_posix(): {
            "sha256": sha256_file(p),
            "size": p.stat().st_size,
        }
        for p in _manifested(run_dir)
    }
    write_json(run_dir / "manifest.json", {"schema": 1, "files": files})


def verify_manifest(run_dir: Path) -> str | None:
    """``None`` when every file the manifest of ``run_dir`` lists is there with
    its size and sha256; otherwise what is wrong. ``files`` may be a mapping of
    relpath to ``{"sha256", "size"}`` or a list of ``{"path", "sha256",
    "size"}``."""
    try:
        manifest = read_json(run_dir / "manifest.json")
    except (OSError, ValueError) as e:
        return f"manifest.json: {e}"
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if isinstance(files, list):
        if not all(
            isinstance(f, dict) and isinstance(f.get("path"), str) for f in files
        ):
            return "manifest.json: a file entry without a path"
        files = {f["path"]: f for f in files}
    if not isinstance(files, dict):
        return "manifest.json has no files"
    for rel, want in files.items():
        if not isinstance(want, dict) or not isinstance(want.get("sha256"), str):
            return f"manifest.json: {rel}: no sha256"
        p = run_dir / rel
        try:
            size = p.stat().st_size
            digest = sha256_file(p)
        except OSError as e:
            return f"{rel}: {e}"
        if "size" in want and size != want["size"]:
            return f"{rel}: size {size} != {want['size']}"
        if digest != want["sha256"]:
            return f"{rel}: sha256 differs"
    return None


def copy_run(src: Path, dst: Path) -> None:
    """Copy the run directory ``src`` into ``dst`` without ``resume/``: the
    bytes only (no modes, owners or times, which a CIFS mount refuses)."""
    for d, dirs, files in os.walk(src):
        rel = Path(d).relative_to(src)
        if rel == Path("."):
            dirs[:] = [x for x in dirs if x != "resume"]
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for name in files:
            if name.startswith(".") and name.endswith(".tmp"):
                continue
            with (
                (Path(d) / name).open("rb") as fin,
                (dst / rel / name).open("wb") as out,
            ):
                while chunk := fin.read(1 << 20):
                    out.write(chunk)
