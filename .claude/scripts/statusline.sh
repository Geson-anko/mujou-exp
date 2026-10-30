#!/bin/bash
# Claude Code statusLine: model / git branch / context 使用率(%) を 1 行で表示し、
# 使用率が閾値以上になったら compact-prep 警告 marker を書く (cooldown 中でなければ)。
#
# stdin = Claude Code が渡す statusLine JSON。context 使用率は transcript 末尾側の
# 最新 usage エントリ (input + cache_read + cache_creation + output) を context
# window で割って算出する。window は model id に "1m" を含むか 200k を超えていれば
# 1,000,000、それ以外は 200,000。
#
# fail-open: 何が失敗しても最低限の 1 行を返し exit 0。依存: python3 のみ。

set -uo pipefail

input=$(cat)

# JSON 解析と usage 集計を 1 回の python3 呼び出しにまとめる (毎ターン走るため)。
# 出力は US (0x1f) 区切り 1 行: sid, model, cwd, window, pct
# 区切りに TAB を使わないのは、TAB が bash の IFS whitespace 扱いになり
# 空フィールドが脱落・連結してしまうため。
parsed=$(
  printf '%s' "$input" | python3 -c '
import json
import sys


SEP = "\x1f"


def clean(value):
    text = str(value or "")
    for bad in ("\n", "\r", "\t", SEP):
        text = text.replace(bad, " ")
    return text.strip()


try:
    data = json.load(sys.stdin)
except Exception:
    data = {}
if not isinstance(data, dict):
    data = {}

model = data.get("model") or {}
if not isinstance(model, dict):
    model = {}
workspace = data.get("workspace") or {}
if not isinstance(workspace, dict):
    workspace = {}

sid = clean(data.get("session_id"))
name = clean(model.get("display_name") or model.get("id")) or "?"
model_id = str(model.get("id") or "").lower()
cwd = clean(workspace.get("current_dir") or data.get("cwd"))

window = 1000000 if "1m" in model_id or data.get("exceeds_200k_tokens") else 200000

used = 0
path = data.get("transcript_path") or ""
if path:
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            tail = handle.readlines()[-400:]
        for line in tail:
            try:
                record = json.loads(line)
            except Exception:
                continue
            if not isinstance(record, dict):
                continue
            message = record.get("message") or {}
            usage = message.get("usage") if isinstance(message, dict) else None
            if not isinstance(usage, dict):
                continue
            used = (
                (usage.get("input_tokens") or 0)
                + (usage.get("cache_read_input_tokens") or 0)
                + (usage.get("cache_creation_input_tokens") or 0)
                + (usage.get("output_tokens") or 0)
            )
    except Exception:
        used = 0

pct = used * 100 // window if window > 0 and used > 0 else 0
print(SEP.join([sid, name, cwd, str(window), str(pct)]))
' 2>/dev/null
) || parsed=""

sid=""
model="?"
cwd=""
window=200000
pct=0
if [[ -n "$parsed" ]]; then
  IFS=$'\x1f' read -r sid model cwd window pct <<<"$parsed"
fi
[[ "$window" =~ ^[0-9]+$ ]] || window=200000
[[ "$pct" =~ ^[0-9]+$ ]] || pct=0
[[ -n "$model" ]] || model="?"

# git branch (cwd があれば)
branch=""
if [[ -n "$cwd" ]]; then
  branch=$(git -C "$cwd" rev-parse --abbrev-ref HEAD 2>/dev/null || true)
fi

# 閾値超で 60% 警告 marker を書く (cooldown 中でなければ)
threshold=60
if [[ -n "$sid" && "$pct" -ge "$threshold" ]]; then
  warned="${TMPDIR:-/tmp}/claude-compact-warned/$sid"
  if [[ ! -f "$warned" ]]; then
    warn_dir="${TMPDIR:-/tmp}/claude-compact-warn"
    mkdir -p "$warn_dir" 2>/dev/null || true
    printf '%s\n' "$pct" >"$warn_dir/$sid" 2>/dev/null || true
  fi
fi

# 表示行
line="⏺ ${model}"
if [[ -n "$branch" ]]; then
  line="${line} | ⎇ ${branch}"
fi
line="${line} | ctx ${pct}% of $((window / 1000))k"
printf '%s\n' "$line"
exit 0
