#!/bin/bash
# 現セッションの session_id を返す。
#   1) CLAUDE_CODE_SESSION_ID 環境変数を最優先 (Claude Code が主セッションに注入)
#   2) 無ければ現 project の transcript ディレクトリの最新 .jsonl の basename
# 取得不能なら空を返し exit 1 (呼び出し側は推測名で state file を作らないこと)。
#
# jq 非依存 (pure bash)。skill から `bash .claude/scripts/get-session-id.sh` で使う。

set -uo pipefail

if [[ -n "${CLAUDE_CODE_SESSION_ID:-}" ]]; then
  printf '%s\n' "$CLAUDE_CODE_SESSION_ID"
  exit 0
fi

# fallback: cwd を Claude Code の transcript slug (/ と . を - に) に変換し、最新 jsonl を探す
slug=$(printf '%s' "$PWD" | sed 's#[/.]#-#g')
newest=""
for f in "$HOME/.claude/projects/$slug"/*.jsonl; do
  [[ -e "$f" ]] || continue
  if [[ -z "$newest" || "$f" -nt "$newest" ]]; then
    newest="$f"
  fi
done

if [[ -n "$newest" ]]; then
  basename "$newest" .jsonl
  exit 0
fi

exit 1
