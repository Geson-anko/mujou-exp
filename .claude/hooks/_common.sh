# shell の hook が source する共通の関数。hook ではない (settings.json に登録しない)。
# fail-open: 失敗しても空文字を返すか何も出力しない。依存: python3 のみ。

# stdin の hook 入力 JSON から session_id を出力する (無い・壊れていれば空)。
read_session_id() {
  python3 -c '
import json
import sys

try:
    data = json.load(sys.stdin)
    print(data.get("session_id") or "" if isinstance(data, dict) else "")
except Exception:
    pass
' 2>/dev/null
}

# stdin のテキストを UserPromptSubmit の additionalContext として出力する。
print_prompt_context() {
  python3 -c '
import json
import sys

print(
    json.dumps(
        {
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": sys.stdin.read(),
            }
        }
    )
)
' 2>/dev/null
}
