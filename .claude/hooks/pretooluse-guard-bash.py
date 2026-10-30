#!/usr/bin/env python3
"""PreToolUse hook (matcher: Bash): block commands that bypass project tooling.

- ``pip install`` 系: 依存は uv で管理する（``uv add`` / ``uv sync``）。venv 外への
  install や ``uv.lock`` とのずれを防ぐ。``.venv/bin/pip``・``python3.14t -m pip``・
  ``uv run pip`` の形も含む
- ``git commit --no-verify`` / ``-n``: pre-commit を飛ばしたコミットを防ぐ。
  ``git -c k=v commit -n`` のようにグローバルオプションを挟んだ形も含む

ブロック時は exit 2 + stderr で理由を Claude に返す。解析に失敗したら fail-open
（exit 0）。依存は標準ライブラリのみ。
"""

import json
import os
import re
import shlex
import sys

# Command separators after which a new simple command starts.
_SEPARATORS = {"&&", "||", ";", "|", "&"}

_PIP_HINT = (
    "pip での install は禁止。依存は uv で管理する: "
    "ランタイム依存は `uv add <pkg>`、開発依存は `uv add --dev <pkg>`、"
    "環境の再現は `uv sync`。"
)
_NO_VERIFY_HINT = (
    "`git commit --no-verify` は禁止。pre-commit を通してからコミットする"
    "（`just format` で事前に整形できる）。"
)


def _split_commands(command: str) -> list[list[str]]:
    """Split a shell line into simple commands (token lists)."""
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|")
    lexer.whitespace_split = True
    commands: list[list[str]] = [[]]
    for token in lexer:
        if token in _SEPARATORS:
            commands.append([])
        else:
            commands[-1].append(token)
    return [c for c in commands if c]


def _strip_env_prefix(tokens: list[str]) -> list[str]:
    """Drop leading ``VAR=value`` assignments."""
    index = 0
    while index < len(tokens) and re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*=.*", tokens[index]
    ):
        index += 1
    return tokens[index:]


def _strip_uv_run(tokens: list[str]) -> list[str]:
    """Drop a leading ``uv run`` and its flags (``uv run --frozen pip ...``)."""
    if tokens[:2] != ["uv", "run"]:
        return tokens
    index = 2
    while index < len(tokens) and tokens[index].startswith("-"):
        index += 1
    return tokens[index:]


def _is_pip_install(tokens: list[str]) -> bool:
    # Compare basenames so that `.venv/bin/pip` and `/usr/bin/python3` count.
    program = os.path.basename(tokens[0]) if tokens else ""
    if len(tokens) >= 2 and re.fullmatch(r"pip3?(\.\d+)?", program):
        return tokens[1] == "install"
    if len(tokens) >= 3 and program == "uv" and tokens[1] == "pip":
        return tokens[2] == "install"
    if len(tokens) >= 4 and re.fullmatch(r"python3?(\.\d+)?t?", program):
        return tokens[1:4] == ["-m", "pip", "install"]
    return False


def _strip_git_options(tokens: list[str]) -> list[str]:
    """Drop the global options between ``git`` and the subcommand."""
    index = 1
    while index < len(tokens) and tokens[index].startswith("-"):
        # These take their value as the next token.
        index += 2 if tokens[index] in ("-c", "-C") else 1
    return ["git", *tokens[index:]]


def _is_commit_no_verify(tokens: list[str]) -> bool:
    if not tokens or tokens[0] != "git":
        return False
    tokens = _strip_git_options(tokens)
    if len(tokens) < 2 or tokens[1] != "commit":
        return False
    for token in tokens[2:]:
        if token == "--no-verify":
            return True
        # Short option cluster such as -n or -an (but not --long options).
        if re.fullmatch(r"-[a-zA-Z]+", token) and "n" in token[1:]:
            return True
    return False


def main() -> int:
    try:
        data = json.load(sys.stdin)
        command = data["tool_input"]["command"]
        commands = _split_commands(command)
    except Exception:
        return 0

    for raw in commands:
        tokens = _strip_uv_run(_strip_env_prefix(raw))
        if _is_pip_install(tokens):
            print(_PIP_HINT, file=sys.stderr)
            return 2
        if _is_commit_no_verify(tokens):
            print(_NO_VERIFY_HINT, file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
