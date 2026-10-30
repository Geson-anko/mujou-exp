#!/bin/bash
# PostCompact hook (matcher: ""): コンテキスト圧縮の発生を marker file で記録する。
# PostCompact は additionalContext を返せないため、圧縮直後の指示注入は
# UserPromptSubmit 側 (userpromptsubmit-compaction-recovery.sh) が担う。
#
# fail-open (常に exit 0)。依存: python3 のみ。

set -uo pipefail

source "${BASH_SOURCE[0]%/*}/_common.sh" 2>/dev/null || exit 0
sid=$(read_session_id) || sid=""
[[ -z "$sid" ]] && exit 0

# 圧縮発生 marker を書く (UserPromptSubmit が検出して指示注入 → 削除する)
marker_dir="${TMPDIR:-/tmp}/claude-compacted"
mkdir -p "$marker_dir" 2>/dev/null || true
date +%s >"$marker_dir/$sid" 2>/dev/null || true

# 60% 警告の cooldown をリセット (圧縮後は再び警告してよい)
rm -f "${TMPDIR:-/tmp}/claude-compact-warned/$sid" 2>/dev/null || true

exit 0
