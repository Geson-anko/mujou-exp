import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from exp.lib.fleet.contract import read_json
from exp.lib.fleet.provision import EXTRACT_SH, envkey, provision, venv_python
from exp.lib.fleet.snapshot import Snapshot, git, make
from tests.lib.fleet.test_snapshot import checkout  # noqa: F401 (a fixture)

# A stand-in for `uv sync`: a venv at $UV_PROJECT_ENVIRONMENT with a package `mujou`.
_SYNC = """
import os, sys, sysconfig, venv
from pathlib import Path
env = Path(os.environ["UV_PROJECT_ENVIRONMENT"])
venv.create(env, symlinks=True, with_pip=False)
site = Path(sysconfig.get_path("purelib", vars={"base": str(env), "platbase": str(env)}))
(site / "mujou").mkdir(parents=True, exist_ok=True)
(site / "mujou" / "__init__.py").write_text("")
"""
SYNC = (sys.executable, "-c", _SYNC)


def _push(snap: Snapshot, repo: Path, root: Path) -> Path:
    """What ``fleet push`` does for one host, over the local file system."""
    for name, src, sha in (
        ("repo", repo, snap.snap),
        ("mujou", repo / "mujou", snap.mujou_snap),
    ):
        git_dir = root / "git" / f"{name}.git"
        subprocess.run(["git", "init", "-q", "--bare", str(git_dir)], check=True)
        git(src, "push", "-q", str(git_dir), f"{sha}:refs/fleet/snap/{sha}")
    out = subprocess.run(
        [
            "bash",
            "-c",
            EXTRACT_SH,
            "_",
            str(root),
            snap.snap,
            snap.mujou_snap,
            snap.snap[:12],
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return Path(out.strip())


@pytest.fixture
def project(checkout: Path) -> Path:  # noqa: F811
    (checkout / "uv.lock").write_text("lock 1\n")
    (checkout / "pyproject.toml").write_text("[project]\n")
    git(checkout, "add", "-A")
    git(checkout, "commit", "-q", "-m", "project")
    return checkout


class TestEnvkey:
    def test_changes_with_the_lock_mujou_and_python(self, tmp_path: Path):
        (tmp_path / "uv.lock").write_text("a")
        (tmp_path / "pyproject.toml").write_text("p")
        k = envkey(tmp_path, "t1", "3.14.0")
        assert k == envkey(tmp_path, "t1", "3.14.0")
        assert k != envkey(tmp_path, "t2", "3.14.0")
        assert k != envkey(tmp_path, "t1", "3.14.1")
        (tmp_path / "uv.lock").write_text("b")
        assert k != envkey(tmp_path, "t1", "3.14.0")


class TestProvision:
    def test_a_read_only_tree_with_its_venv_and_latest(
        self, project: Path, tmp_path: Path
    ):
        (project / "mujou" / "dirty.txt").write_text("dirty\n")
        (project / "untracked.txt").write_text("u\n")
        snap = make(project)
        root = tmp_path / "root"
        tree = _push(snap, project, root)
        key = provision(root, tree, snap.to_dict(), sync=SYNC)
        assert tree == root / "code" / snap.snap[:12]
        assert (tree / "untracked.txt").read_text() == "u\n"
        assert (tree / "mujou" / "dirty.txt").read_text() == "dirty\n"
        assert (tree / ".ready").exists() and (root / "venv" / key / ".ready").exists()
        code = read_json(tree / "CODE.json")
        assert code["snap"] == snap.snap and code["envkey"] == key and code["dirty"]
        assert "b/mujou/dirty.txt" in (tree / "code.diff").read_text()
        assert "b/untracked.txt" in (tree / "code.diff").read_text()
        assert not os.access(tree / "a.txt", os.W_OK)
        assert not os.access(tree, os.W_OK)
        assert (root / "code" / "latest").resolve() == tree
        assert venv_python(root, key).exists()

    def test_idempotent_and_a_venv_shared_by_snapshots_with_the_same_envkey(
        self, project: Path, tmp_path: Path
    ):
        root = tmp_path / "root"
        first = make(project)
        key = provision(root, _push(first, project, root), first.to_dict(), sync=SYNC)
        assert (
            provision(
                root, _push(first, project, root), first.to_dict(), sync=("false",)
            )
            == key
        )
        (project / "exp.py").write_text("changed code\n")
        second = make(project)
        tree = _push(second, project, root)
        assert provision(root, tree, second.to_dict(), sync=("false",)) == key
        assert (root / "code" / "latest").resolve() == tree
        (project / "uv.lock").write_text("lock 2\n")
        third = make(project)
        other = provision(root, _push(third, project, root), third.to_dict(), sync=SYNC)
        assert other != key
        assert sorted(p.name for p in (root / "venv").iterdir()) == sorted([key, other])
        # the earlier trees stay as they were
        assert (
            root / "code" / second.snap[:12] / "exp.py"
        ).read_text() == "changed code\n"


@pytest.mark.skipif(shutil.which("flock") is None, reason="flock(1) is Linux's")
class TestProvisionLock:
    def test_provision_runs_under_the_hosts_provision_lock(self, tmp_path: Path):
        from exp.lib.fleet.cli import _PROVISION_SH

        root, tree = tmp_path / "root", tmp_path / "tree"
        tree.mkdir()
        lock = root / "locks" / "provision.lock"
        probe = tmp_path / "probe.sh"  # stands in for the python that provisions
        probe.write_text(
            f'#!/bin/sh\nif flock -n "{lock}" true; then echo free; else echo held; fi\n'
        )
        probe.chmod(0o755)
        out = subprocess.run(
            ["bash", "-c", _PROVISION_SH, "_", str(tree), str(root), "{}", "0"],
            env={**os.environ, "FLEET_PYTHON": str(probe)},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        assert out.strip() == "held"
        assert subprocess.run(["flock", "-n", str(lock), "true"]).returncode == 0
