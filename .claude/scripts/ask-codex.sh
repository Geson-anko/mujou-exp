#!/usr/bin/env bash
# Ask Codex (OpenAI Codex CLI) a question from Claude Code, read-only.
#
# Usage:
#   ask-codex.sh [-r THREAD_ID] [-e EFFORT] [PROMPT...]
#   echo "PROMPT" | ask-codex.sh [-r THREAD_ID] [-e EFFORT]
#
#   -r THREAD_ID  continue an earlier discussion (the id printed on the last line)
#   -e EFFORT     model_reasoning_effort (low|medium|high|xhigh); default: codex config
#
# Env: CODEX_MODEL (default gpt-6.2-sol).
# Prints Codex's final answer, then a last line "codex-thread: <id>".
# Codex runs in the read-only sandbox at the git toplevel of the current directory.
set -euo pipefail

model="${CODEX_MODEL:-gpt-6.2-sol}"
thread=""
effort=""
while getopts "r:e:h" opt; do
  case "$opt" in
    r) thread="$OPTARG" ;;
    e) effort="$OPTARG" ;;
    h) sed -n '2,13p' "$0"; exit 0 ;;
    *) exit 2 ;;
  esac
done
shift $((OPTIND - 1))

command -v codex >/dev/null || { echo "ask-codex: codex CLI not found" >&2; exit 127; }

if [ $# -gt 0 ]; then
  question="$*"
elif [ ! -t 0 ]; then
  question="$(cat)"
else
  echo "ask-codex: no prompt (pass it as arguments or on stdin)" >&2
  exit 2
fi
[ -n "$question" ] || { echo "ask-codex: empty prompt" >&2; exit 2; }

root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

opts=(-m "$model" -c 'sandbox_mode="read-only"' --json -o "$work/answer.md")
[ -n "$effort" ] && opts+=(-c "model_reasoning_effort=\"$effort\"")

if [ -z "$thread" ]; then
  # First turn: tell Codex who is asking and how to answer.
  cat >"$work/prompt.md" <<EOF
You are being consulted by Claude Code, another AI agent working in this
repository (${root}). Act as a peer reviewer and discussion partner.

- Read CLAUDE.md first; it holds the project's conventions. Read any files you
  need, but do not modify anything (read-only sandbox).
- Think independently. If Claude's premise, hypothesis or reading of results
  looks wrong, say so plainly and explain why. Do not just agree.
- Ground claims in the code, theory or numbers; say what you are unsure of and
  what experiment or check would settle it.
- Answer in Japanese (keep code, identifiers and paths as-is). Be concise.

--- Question from Claude Code ---
${question}
EOF
  codex exec "${opts[@]}" -C "$root" - <"$work/prompt.md" >"$work/events.jsonl" 2>"$work/stderr.log" || {
    echo "ask-codex: codex exec failed" >&2; tail -20 "$work/stderr.log" >&2; exit 1; }
  thread="$(python3 -c 'import json,sys
for l in open(sys.argv[1]):
    try: e=json.loads(l)
    except ValueError: continue
    if e.get("type")=="thread.started": print(e["thread_id"]); break' "$work/events.jsonl")"
else
  printf '%s\n' "$question" >"$work/prompt.md"
  (cd "$root" && codex exec resume "$thread" "${opts[@]}" - <"$work/prompt.md") \
    >"$work/events.jsonl" 2>"$work/stderr.log" || {
    echo "ask-codex: codex exec resume failed" >&2; tail -20 "$work/stderr.log" >&2; exit 1; }
fi

cat "$work/answer.md"
echo
echo "codex-thread: ${thread}"
