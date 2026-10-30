#!/bin/bash
# UserPromptSubmit hook: statusLine が書いた compact-warn marker を検出し、additionalContext で
# /compact-prep の実行を促す (one-shot)。
#
# フロー: statusline.sh が ctx% >= 閾値 で warn marker 書込
#   → 本 hook が検出 → 指示注入 → warn marker 削除 + warned(cooldown) marker 作成
#   → PostCompact hook (compaction-recovery.sh) が warned marker を削除 (cooldown リセット)
#
# overhead: marker が無ければ即 exit。fail-open (常に exit 0)。依存: python3 のみ。

set -uo pipefail

source "${BASH_SOURCE[0]%/*}/_common.sh" 2>/dev/null || exit 0
sid=$(read_session_id) || sid=""
[[ -z "$sid" ]] && exit 0

warn_marker="${TMPDIR:-/tmp}/claude-compact-warn/$sid"
[[ -f "$warn_marker" ]] || exit 0

pct=$(cat "$warn_marker" 2>/dev/null)
pct=${pct:-"?"}
rm -f "$warn_marker" 2>/dev/null || true # one-shot

# cooldown marker (statusLine が再び warn marker を書くのを抑止; PostCompact でリセット)
warned_dir="${TMPDIR:-/tmp}/claude-compact-warned"
mkdir -p "$warned_dir" 2>/dev/null || true
date +%s >"$warned_dir/$sid" 2>/dev/null || true

ctx="[COMPACT PREP REMINDER] context 使用率が ${pct}% に達した。"
ctx+=$'\n'"- 作業の区切りで、ユーザーに \`/compact-prep\` の実行を提案せよ"
ctx+=$'\n'"- \`/compact-prep\` 実行後、続けて \`/compact\` の実行を案内せよ"
ctx+=$'\n'"- scope 縮小や別セッション化ではなく、圧縮前 state 保存で対処せよ"
ctx+=$'\n'"- 自動 compact に先を越される前に、区切りを作ることを優先せよ"

printf '%s' "$ctx" | print_prompt_context || true
exit 0
