"""The output of one run: ``runs/<date>-<name>/`` with ``config.json`` and
``metrics.jsonl``."""

import datetime
import json
import subprocess
from pathlib import Path
from types import TracebackType
from typing import IO, Any, Self

from experiments.continuous_stream.config import Config

__all__ = ["REPO_ROOT", "Recorder", "git_sha", "git_dirty"]

REPO_ROOT = Path(__file__).resolve().parents[2]


def git_sha(repo: Path) -> str:
    """The commit checked out in ``repo``."""
    return _git(repo, "rev-parse", "HEAD").strip()


def git_dirty(repo: Path) -> bool:
    """Whether ``repo`` has changes not committed (the submodule's included)."""
    return bool(_git(repo, "status", "--porcelain").strip())


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout


class Recorder:
    """Writes the settings of a run once and its metrics as they come.

    ``config.json`` holds the settings (``"config"``), the commits of mujou and
    of this repository (``"mujou"``, ``"repo"``), whether either has changes not
    committed (``"dirty"``) and when the run started (``"created"``, wall
    clock). ``metrics.jsonl`` has one JSON object per record, with the time of
    the run's timer (``"time"``, seconds).

    Raises:
        FileExistsError: The directory of the run exists already.
    """

    def __init__(self, config: Config, root: Path = REPO_ROOT / "runs") -> None:
        now = datetime.datetime.now().astimezone()
        self.dir = root / f"{now:%Y-%m-%d}-{config.name}"
        self.dir.mkdir(parents=True, exist_ok=False)
        head = {
            "config": config.to_dict(),
            "mujou": git_sha(REPO_ROOT / "mujou"),
            "repo": git_sha(REPO_ROOT),
            "dirty": git_dirty(REPO_ROOT) or git_dirty(REPO_ROOT / "mujou"),
            "created": now.isoformat(),
        }
        (self.dir / "config.json").write_text(json.dumps(head, indent=2) + "\n")
        self._metrics: IO[str] = (self.dir / "metrics.jsonl").open("w")

    def log(self, time: float, **values: Any) -> None:
        """Append one record at ``time`` seconds of the run's timer."""
        self._metrics.write(json.dumps({"time": time, **values}) + "\n")

    def close(self) -> None:
        self._metrics.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
