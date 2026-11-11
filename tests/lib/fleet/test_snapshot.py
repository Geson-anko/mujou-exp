import subprocess
from pathlib import Path

import pytest

from exp.lib.fleet.snapshot import git, make


def _sh(repo: Path, *args: str) -> str:
    return git(repo, *args)


@pytest.fixture
def checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A checkout with a submodule-like ``mujou`` (a clone with a remote branch)."""
    for k, v in {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
        "GIT_CONFIG_GLOBAL": "/dev/null",
    }.items():
        monkeypatch.setenv(k, v)
    origin = tmp_path / "mujou-origin"
    origin.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(origin)], check=True)
    (origin / "m.txt").write_text("m\n")
    _sh(origin, "add", "-A")
    _sh(origin, "commit", "-q", "-m", "m")
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    subprocess.run(["git", "clone", "-q", str(origin), str(repo / "mujou")], check=True)
    (repo / "a.txt").write_text("a\n")
    (repo / ".gitignore").write_text("ignored.txt\n")
    _sh(repo, "add", "-A")  # mujou goes in as a gitlink
    _sh(repo, "commit", "-q", "-m", "a")
    return repo


class TestSnapshot:
    def test_a_clean_tree_is_head(self, checkout: Path):
        s = make(checkout)
        assert s.snap == _sh(checkout, "rev-parse", "HEAD")
        assert not s.dirty
        assert s.mujou_snap == s.mujou_base
        assert s.mujou_pushed

    def test_a_dirty_tree_with_untracked_files_and_a_dirty_mujou(self, checkout: Path):
        head = _sh(checkout, "rev-parse", "HEAD")
        mujou_head = _sh(checkout / "mujou", "rev-parse", "HEAD")
        (checkout / "a.txt").write_text("changed\n")
        (checkout / "new.txt").write_text("new\n")
        (checkout / "ignored.txt").write_text("x\n")
        (checkout / "mujou" / "m.txt").write_text("changed m\n")
        (checkout / "mujou" / "n.txt").write_text("new in mujou\n")
        _sh(checkout, "add", "a.txt")  # something staged in the real index
        status = _sh(checkout, "status", "--porcelain")
        s = make(checkout)
        assert s.dirty and s.base_head == head and s.mujou_base == mujou_head
        assert not s.mujou_pushed
        assert _sh(checkout, "show", f"{s.snap}:a.txt") == "changed"
        assert _sh(checkout, "show", f"{s.snap}:new.txt") == "new"
        assert "ignored.txt" not in _sh(checkout, "ls-tree", "--name-only", s.snap)
        gitlink = _sh(checkout, "ls-tree", s.snap, "mujou").split()
        assert gitlink[:3] == ["160000", "commit", s.mujou_snap]
        assert (
            _sh(checkout / "mujou", "show", f"{s.mujou_snap}:n.txt") == "new in mujou"
        )
        assert _sh(checkout, "rev-parse", f"{s.snap}^") == head
        # HEAD, the branch and the real index stay as they were
        assert _sh(checkout, "rev-parse", "HEAD") == head
        assert _sh(checkout / "mujou", "rev-parse", "HEAD") == mujou_head
        assert _sh(checkout, "status", "--porcelain") == status
        assert _sh(checkout, "rev-parse", f"refs/fleet/snap/{s.snap}") == s.snap

    def test_the_same_tree_gives_the_same_snapshot(self, checkout: Path):
        (checkout / "new.txt").write_text("new\n")
        assert make(checkout) == make(checkout)
