"""Tests for the Claude Code hooks under .claude/hooks/."""

import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
HOOKS_DIR = REPO_ROOT / ".claude" / "hooks"


def run_hook(
    script: str, payload: Mapping[str, object], env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    path = HOOKS_DIR / script
    command = (
        [sys.executable, str(path)] if path.suffix == ".py" else ["bash", str(path)]
    )
    return subprocess.run(
        command,
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
        timeout=30,
    )


class TestPreToolUseGuardBash:
    @staticmethod
    def run_command(command: str) -> subprocess.CompletedProcess[str]:
        return run_hook(
            "pretooluse-guard-bash.py",
            {"tool_name": "Bash", "tool_input": {"command": command}},
        )

    @pytest.mark.parametrize(
        "command",
        [
            "pip install numpy",
            "pip3 install -U numpy",
            "python -m pip install numpy",
            "uv pip install numpy",
            "cd src && pip install -e .",
            "FOO=1 pip install numpy",
            "python3.14t -m pip install numpy",
            "uv run pip install numpy",
            "uv run --frozen python -m pip install numpy",
            ".venv/bin/pip install numpy",
        ],
    )
    def test_blocks_pip_install(self, command: str):
        result = self.run_command(command)

        assert result.returncode == 2
        assert "uv add" in result.stderr

    @pytest.mark.parametrize(
        "command",
        [
            "git commit --no-verify -m 'x'",
            "git commit -n -m 'x'",
            "git commit -anm 'x'",
            "git -c core.editor=true commit -n -m 'x'",
            "git -C repo --no-pager commit --no-verify -m 'x'",
        ],
    )
    def test_blocks_commit_without_verification(self, command: str):
        result = self.run_command(command)

        assert result.returncode == 2
        assert "pre-commit" in result.stderr

    @pytest.mark.parametrize(
        "command",
        [
            "uv add numpy",
            "uv pip list",
            "pip list",
            "git commit -m 'feat(core): add -n flag docs'",
            "git commit -am 'fix: x'",
            "echo 'pip install numpy'",
            "uv run pytest -q",
            "git -c core.editor=true commit -m 'x'",
            "git -C repo log -n 3",
        ],
    )
    def test_allows_other_commands(self, command: str):
        result = self.run_command(command)

        assert result.returncode == 0
        assert result.stderr == ""

    def test_fails_open_on_malformed_input(self):
        result = run_hook("pretooluse-guard-bash.py", {"tool_input": {}})

        assert result.returncode == 0


class TestCompactionRecovery:
    def test_post_compact_marker_triggers_one_shot_recovery(self, tmp_path: Path):
        env = {"TMPDIR": str(tmp_path)}
        payload = {"session_id": "sid-123"}

        run_hook("compaction-recovery.sh", payload, env)
        first = run_hook("userpromptsubmit-compaction-recovery.sh", payload, env)
        second = run_hook("userpromptsubmit-compaction-recovery.sh", payload, env)

        context = json.loads(first.stdout)["hookSpecificOutput"]["additionalContext"]
        assert "COMPACTION RECOVERY" in context
        assert second.stdout == ""

    def test_warn_marker_triggers_one_shot_compact_prep_reminder(self, tmp_path: Path):
        env = {"TMPDIR": str(tmp_path)}
        payload = {"session_id": "sid-456"}
        warn_dir = tmp_path / "claude-compact-warn"
        warn_dir.mkdir()
        (warn_dir / "sid-456").write_text("63\n")

        first = run_hook("userpromptsubmit-compact-prep-reminder.sh", payload, env)
        second = run_hook("userpromptsubmit-compact-prep-reminder.sh", payload, env)

        context = json.loads(first.stdout)["hookSpecificOutput"]["additionalContext"]
        assert "63%" in context
        assert "/compact-prep" in context
        assert second.stdout == ""
