"""A git snapshot of the working tree, mujou included (design 4.4, step 1).

With a clean tree the snapshot is ``HEAD``. Otherwise every file git does not
ignore, untracked ones included, goes into a commit on top of ``HEAD`` made
through a temporary index: ``HEAD``, the branch and the real index stay where
they are. mujou gets the same treatment, and the gitlink of the snapshot points
at mujou's snapshot. The author, committer and dates are fixed (the dates of
``HEAD``), so the same tree gives the same snapshot. ``refs/fleet/snap/<sha>``
keeps it from git's gc.
"""

import os
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

__all__ = ["Snapshot", "git", "make"]

_REF = "refs/fleet/snap/"


def git(repo: Path, *args: str, env: Mapping[str, str] | None = None) -> str:
    """The stripped stdout of ``git -C repo args``; ``CalledProcessError`` on failure."""
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
        env=None if env is None else {**os.environ, **env},
    ).stdout.strip()


@dataclass(frozen=True, kw_only=True)
class Snapshot:
    """``snap`` (40 hex) on top of ``base_head``, and mujou's ``mujou_snap`` on
    top of ``mujou_base``. ``dirty`` when either differs from its base;
    ``mujou_pushed`` when mujou's snapshot is on one of mujou's remote branches."""

    snap: str
    base_head: str
    dirty: bool
    mujou_snap: str
    mujou_base: str
    mujou_pushed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _commit(repo: Path, gitlink: tuple[str, str] | None = None) -> tuple[str, str]:
    """``(snapshot, HEAD)`` of ``repo``: the tree of the working files (and the
    ``gitlink`` ``(path, sha)``) committed on ``HEAD``, or ``HEAD`` itself when
    the tree is ``HEAD``'s."""
    head = git(repo, "rev-parse", "HEAD")
    with tempfile.TemporaryDirectory() as tmp:
        env = {"GIT_INDEX_FILE": str(Path(tmp) / "index")}
        git(repo, "read-tree", "HEAD", env=env)
        git(repo, "add", "-A", env=env)
        if gitlink is not None:
            path, sha = gitlink
            git(
                repo,
                "update-index",
                "--add",
                "--cacheinfo",
                f"160000,{sha},{path}",
                env=env,
            )
        tree = git(repo, "write-tree", env=env)
    if tree == git(repo, "rev-parse", "HEAD^{tree}"):
        return head, head
    date = git(repo, "log", "-1", "--format=%cI", "HEAD")
    who = {
        "GIT_AUTHOR_NAME": "fleet",
        "GIT_AUTHOR_EMAIL": "fleet@localhost",
        "GIT_COMMITTER_NAME": "fleet",
        "GIT_COMMITTER_EMAIL": "fleet@localhost",
        "GIT_AUTHOR_DATE": date,
        "GIT_COMMITTER_DATE": date,
    }
    snap = git(repo, "commit-tree", tree, "-p", head, "-m", "fleet snapshot", env=who)
    return snap, head


def make(repo: Path, mujou: str = "mujou") -> Snapshot:
    """The snapshot of the checkout ``repo`` and its submodule at ``repo/mujou``,
    with refs under ``refs/fleet/snap/`` in both."""
    sub = repo / mujou
    mujou_snap, mujou_base = _commit(sub)
    snap, head = _commit(repo, gitlink=(mujou, mujou_snap))
    for r, sha in ((repo, snap), (sub, mujou_snap)):
        git(r, "update-ref", _REF + sha, sha)
    pushed = bool(git(sub, "branch", "-r", "--contains", mujou_snap))
    return Snapshot(
        snap=snap,
        base_head=head,
        dirty=snap != head or mujou_snap != mujou_base,
        mujou_snap=mujou_snap,
        mujou_base=mujou_base,
        mujou_pushed=pushed,
    )
