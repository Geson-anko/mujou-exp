"""A snapshot on a host (design 4.4, steps 2-4): the bare repos, the read-only
code tree and the venv of its envkey.

::

    <root>/git/{repo,mujou}.git        bare repos the Mac pushes the snapshots to
    <root>/code/<snap12>/              git archive of the snapshot, mujou inside; read-only
        CODE.json code.diff .envkey .ready
    <root>/code/latest -> <snap12>     the code the CLI and new workers start from
    <root>/venv/<envkey>/  .ready      uv sync --frozen --no-editable (mujou as a wheel)

``envkey = sha256(uv.lock, pyproject.toml, mujou's tree, python's version)[:12]``,
so snapshots that change only the experiment's code share a venv, and a new
mujou gets a new venv next to the old one: a run keeps its code and its mujou
for its whole life. ``EXTRACT_SH`` (bash, in the container) puts the code tree
in place; then ``provision`` (this module, run from that tree by the
interpreter ``uv python find`` gives) does the rest.
"""

import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from exp.lib.fleet.contract import write_json

__all__ = ["EXTRACT_SH", "SYNC", "envkey", "provision", "venv_python"]

SYNC = ("uv", "sync", "--frozen", "--no-editable")

# argv: root snap mujou_snap snap12. Extracts the snapshot to <root>/code/<snap12>
# unless it is there and ready, under the host's provision lock (released when
# the shell exits); prints the directory.
EXTRACT_SH = r"""
set -euo pipefail
root=$1 snap=$2 msnap=$3 s12=$4
mkdir -p "$root/code" "$root/venv" "$root/locks"
exec 9>"$root/locks/provision.lock"
if command -v flock > /dev/null; then flock 9; fi
dst="$root/code/$s12"
if [ ! -e "$dst/.ready" ]; then
  tmp="$root/code/.$s12.tmp"
  for d in "$tmp" "$dst"; do
    if [ -e "$d" ]; then chmod -R u+w "$d"; rm -rf "$d"; fi
  done
  mkdir -p "$tmp/mujou"
  git --git-dir="$root/git/repo.git" archive "$snap" | tar -x -C "$tmp"
  git --git-dir="$root/git/mujou.git" archive "$msnap" | tar -x -C "$tmp/mujou"
  mv "$tmp" "$dst"
fi
echo "$dst"
"""

_VERIFY_SRC = """
import json, sys
import mujou
out = {"mujou": mujou.__file__, "gil": sys._is_gil_enabled()}
try:
    import torch
    out["cuda"] = torch.cuda.device_count()
except ImportError:
    out["cuda"] = None
print(json.dumps(out))
"""


def envkey(tree: Path, mujou_tree: str, python_version: str) -> str:
    """12 hex of ``sha256`` over ``uv.lock`` and ``pyproject.toml`` of the code
    ``tree``, mujou's git tree sha and ``python_version``."""
    h = hashlib.sha256()
    for name in ("uv.lock", "pyproject.toml"):
        h.update(name.encode() + b"\0" + (tree / name).read_bytes() + b"\0")
    h.update(mujou_tree.encode() + b"\0" + python_version.encode())
    return h.hexdigest()[:12]


def venv_python(root: Path, key: str) -> Path:
    """The python of the venv ``key`` on a host with fleet root ``root``."""
    return root / "venv" / key / "bin" / "python"


def _git(git_dir: Path, *args: str) -> str:
    return subprocess.run(
        ["git", f"--git-dir={git_dir}", *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def provision(
    root: Path,
    tree: Path,
    info: Mapping[str, Any],
    *,
    sync: Sequence[str] = SYNC,
    gpus: int | None = None,
) -> str:
    """Finish the extracted code ``tree`` of the snapshot ``info`` (a
    ``Snapshot.to_dict()``) on the host with fleet root ``root``; returns its
    envkey.

    Makes the venv of the envkey unless it is ready: ``sync`` in ``tree`` with
    ``UV_PROJECT_ENVIRONMENT`` set to it, then the venv's python, run from
    ``/``, must import mujou from inside the venv with the GIL disabled and (when
    ``gpus`` is given) see ``gpus`` CUDA devices. Then writes ``CODE.json``,
    ``code.diff`` and ``.envkey``, marks the tree ready, makes it read-only and
    points ``code/latest`` at it. Idempotent.
    """
    git_dir = root / "git"
    mujou_tree = _git(
        git_dir / "mujou.git", "rev-parse", f"{info['mujou_snap']}^{{tree}}"
    )
    key = envkey(tree, mujou_tree.strip(), sys.version)
    venv = root / "venv" / key
    if not (venv / ".ready").exists():
        env = {
            **os.environ,
            "UV_PROJECT_ENVIRONMENT": str(venv),
            "CARGO_TARGET_DIR": str(root / "cargo-target"),
        }
        subprocess.run(list(sync), cwd=tree, env=env, check=True)
        _verify(venv, gpus)
        (venv / ".ready").write_text(key + "\n")
    if not (tree / ".ready").exists():
        diff = ""
        if info["snap"] != info["base_head"]:
            diff += _git(git_dir / "repo.git", "diff", info["base_head"], info["snap"])
        if info["mujou_snap"] != info["mujou_base"]:
            diff += _git(
                git_dir / "mujou.git",
                "diff",
                "--src-prefix=a/mujou/",
                "--dst-prefix=b/mujou/",
                info["mujou_base"],
                info["mujou_snap"],
            )
        write_json(tree / "CODE.json", {**info, "envkey": key})
        (tree / "code.diff").write_text(diff)
        (tree / ".envkey").write_text(key + "\n")
        (tree / ".ready").write_text(info["snap"] + "\n")
        subprocess.run(["chmod", "-R", "a-w", str(tree)], check=True)
    latest = root / "code" / "latest"
    tmp = latest.with_name(".latest.tmp")
    tmp.unlink(missing_ok=True)
    tmp.symlink_to(tree.name)
    os.replace(tmp, latest)
    return key


def _verify(venv: Path, gpus: int | None) -> None:
    proc = subprocess.run(
        [str(venv / "bin" / "python"), "-I", "-c", _VERIFY_SRC],
        cwd="/",
        capture_output=True,
        text=True,
        check=True,
    )
    got = json.loads(proc.stdout)
    if not got["mujou"].startswith(str(venv)):
        raise RuntimeError(f"mujou is imported from {got['mujou']}, outside {venv}")
    if got["gil"]:
        raise RuntimeError(f"the GIL is enabled in {venv}")
    if gpus is not None and got["cuda"] != gpus:
        raise RuntimeError(f"{venv} sees {got['cuda']} CUDA devices, not {gpus}")
