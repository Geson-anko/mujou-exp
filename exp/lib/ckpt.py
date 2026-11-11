"""The two checkpoints of a run.

An evaluation checkpoint ``ckpt/<t:07.1f>.pt`` holds the weights of the
modules only, ``{"schema", "program", "group", "t", "frame", "modules": {name:
state_dict}}``: it is read with ``torch.load(weights_only=True)`` into modules
rebuilt from the settings (``load_eval``), and copied to the NAS.

A resume checkpoint ``resume/<frame:08d>.pt`` holds all a run needs to go on
as if it had never stopped, ``{"schema", "group", "frame", "mujou":
run.state_dict(), "host": host state, "offsets": {file: bytes}}``: the tree of
mujou (the nodes' states, ports and timer), the host's state (the streams and
their generators, ``Host.frames``, the running sums) and the sizes of the
``.jsonl`` files (``RunDir.sync``) to cut them back to. ``resume/latest`` names
the newest (its file name and a newline); the two newest are kept, on the host
only, and removed when the run completes (``clear_resume``).

Each file is written to a temporary name and moved into place, so a crash
leaves the previous checkpoint whole.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from exp.lib.rundir import atomic_open

__all__ = [
    "CKPT_SCHEMA",
    "clear_resume",
    "eval_path",
    "load_eval",
    "load_resume",
    "save_eval",
    "save_resume",
]

CKPT_SCHEMA = 1
_KEEP = 2  # resume checkpoints kept


def _save(obj: Any, path: Path) -> None:
    """``torch.save`` to ``path`` atomically (``atomic_open``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_open(path) as f:
        torch.save(obj, f)


def _load(path: Path) -> dict[str, Any]:
    """The tree of the checkpoint ``path``, on the CPU.

    Raises:
        ValueError: It is of another ``schema`` than ``CKPT_SCHEMA``.
    """
    tree = torch.load(path, map_location="cpu", weights_only=True)
    if tree.get("schema") != CKPT_SCHEMA:
        raise ValueError(
            f"{path} is of schema {tree.get('schema')!r}, not {CKPT_SCHEMA}"
        )
    return tree


def eval_path(run_dir: Path, t: float) -> Path:
    """``ckpt/<t:07.1f>.pt`` of the run in ``run_dir``."""
    return run_dir / "ckpt" / f"{t:07.1f}.pt"


def save_eval(
    run_dir: Path,
    *,
    program: str,
    group: str,
    t: float,
    frame: int,
    modules: Mapping[str, nn.Module],
) -> Path:
    """Write the evaluation checkpoint at ``t`` seconds (frame ``frame``) with
    the weights of ``modules`` (copied to the CPU) and return its path."""
    tree = {
        "schema": CKPT_SCHEMA,
        "program": program,
        "group": group,
        "t": t,
        "frame": frame,
        "modules": {
            name: {k: v.detach().cpu() for k, v in m.state_dict().items()}
            for name, m in modules.items()
        },
    }
    path = eval_path(run_dir, t)
    _save(tree, path)
    return path


def load_eval(
    path: Path, modules: Mapping[str, nn.Module], *, group: str | None = None
) -> dict[str, Any]:
    """Load the weights of the evaluation checkpoint ``path`` into ``modules``
    (built from the run's settings, on any device) and return its header
    (``schema``, ``program``, ``group``, ``t``, ``frame``).

    Raises:
        ValueError: The checkpoint's modules are not those of ``modules`` by
            name, ``group`` is given and is not the checkpoint's, or the
            checkpoint is of another schema.
    """
    tree = _load(path)
    if group is not None and tree["group"] != group:
        raise ValueError(f"{path} is of group {tree['group']!r}, not {group!r}")
    saved = tree["modules"]
    if set(saved) != set(modules):
        raise ValueError(f"{path} has modules {sorted(saved)}, not {sorted(modules)}")
    for name, m in modules.items():
        m.load_state_dict(saved[name])
    return {k: v for k, v in tree.items() if k != "modules"}


def save_resume(
    run_dir: Path,
    *,
    group: str,
    frame: int,
    mujou_state: Mapping[str, Any],
    host: Mapping[str, Any],
    offsets: Mapping[str, int],
) -> Path:
    """Write ``resume/<frame:08d>.pt``, point ``resume/latest`` at it, and
    remove all but the ``2`` newest; returns its path.

    Args:
        mujou_state: ``run.state_dict()`` of the run, taken between steps.
        host: The host's state (the program's ``host_state``), of tensors
            (generators' states among them) and plain values only.
        offsets: The sizes of the ``.jsonl`` files (``RunDir.sync()``) to cut
            them back to on resume.
    """
    tree = {
        "schema": CKPT_SCHEMA,
        "group": group,
        "frame": frame,
        "mujou": dict(mujou_state),
        "host": dict(host),
        "offsets": dict(offsets),
    }
    folder = run_dir / "resume"
    path = folder / f"{frame:08d}.pt"
    _save(tree, path)
    with atomic_open(folder / "latest") as f:
        f.write(f"{path.name}\n".encode())
    for old in sorted(folder.glob("*.pt"))[:-_KEEP]:
        old.unlink()
    return path


def load_resume(run_dir: Path, *, group: str) -> dict[str, Any] | None:
    """The newest resume checkpoint of the run in ``run_dir`` (the tree of
    ``save_resume``), or ``None`` if there is none.

    The newest is the one ``resume/latest`` names, or, if that is missing,
    empty or names no file (a crash between the two writes, a power loss),
    the last ``*.pt`` by name (their names are the zero-padded frames).

    It is read with ``weights_only=True``, as the tree of mujou allows: the
    host's state must be of tensors (a generator's ``get_state()`` is one) and
    plain values.

    Raises:
        ValueError: The checkpoint is of another group (other settings) or
            another schema.
    """
    folder = run_dir / "resume"
    latest = folder / "latest"
    name = latest.read_text().strip() if latest.is_file() else ""
    path = folder / name
    if not name or not path.is_file():
        saved = sorted(folder.glob("*.pt"))
        if not saved:
            return None
        path = saved[-1]
    tree = _load(path)
    if tree["group"] != group:
        raise ValueError(f"{path} is of group {tree['group']!r}, not {group!r}")
    return tree


def clear_resume(run_dir: Path) -> None:
    """Remove the resume checkpoints of a run that completed."""
    folder = run_dir / "resume"
    if not folder.exists():
        return
    for path in folder.iterdir():
        path.unlink()
    folder.rmdir()
