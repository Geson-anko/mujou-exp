---
name: compact-prep
description: 'Claude Code の /compact 実行前に、現セッションの作業状態を一時 state file へ退避する。圧縮サマリーに残りにくい判断構造 (採用/却下した案・現在フェーズ・委譲したサブエージェント) を保存し、圧縮後の復旧 hook が読み戻す。MANDATORY TRIGGERS: /compact-prep, compact-prep, 圧縮準備, compact 準備, コンパクト準備, 圧縮前状態保存。DO NOT TRIGGER: compact 後の復旧、通常の進捗報告、plan 作成、context 使用率の雑談。'
argument-hint: '[復旧メモ]'
allowed-tools: Read, Write, Bash(bash .claude/scripts/get-session-id.sh), Bash(mkdir:*), Bash(date:*), Bash(pwd)
---

# compact-prep (圧縮前の状態退避)

Claude Code の `/compact` 直前に実行する。圧縮サマリーへ残りにくい作業状態を
`${TMPDIR:-/tmp}/claude-compact-state/<session_id>.md` へ保存する。圧縮後は
`.claude/hooks/userpromptsubmit-compaction-recovery.sh` がこの file を読み戻すよう指示する。

この skill の成果物は、`<種別>/<日付>/...` ブランチでのコード変更ではなく、退避したセッション状態そのもの。
圧縮の要約が「何をやったか」を残す一方で、この file は「なぜその選択をしたか / 却下した案 /
今どのフェーズか / どのサブエージェントに何を委譲したか」という、要約から落ちやすい
判断構造を残す。

## 進め方 (この順で)

1. **session_id を取得する。** `bash .claude/scripts/get-session-id.sh` を実行する。
    - 空 (exit 1) なら state file を推測名で作らず、「session_id を取得できないため準備未完了」と
        報告して停止する (Hard gate)。
2. 保存先を `${TMPDIR:-/tmp}/claude-compact-state/<session_id>.md` に決め、
    `mkdir -p "${TMPDIR:-/tmp}/claude-compact-state"` でディレクトリを用意する。
3. 現在の状態を棚卸しする。
    - TodoWrite の TaskList (in-progress / 残タスク)
    - plan mode の計画 (あれば) と現在フェーズ・ステップ
    - このセッションで採用した案・却下した案とその理由
    - 起動中・委譲済みのサブエージェント (implementation-planner / spec-test-author /
        plan-implementer / code-reviewer / algorithm-reviewer / code-simplifier、エージェントチーム、
        do-on-worktree のバックグラウンド worktree) と担当
    - 編集中の file と、未保存・未検証（`just run` 未通過）の注意点
4. state file に次の見出しを **この順で** Write する。
    - `# Compact Prep State`
    - `## Active Plan` (plan / 現在フェーズ・ステップ。無ければ「なし」)
    - `## TaskList Summary` (in-progress タスクと補足)
    - `## Session Decisions` (採用/却下した案と理由)
    - `## Constraints and Blockers` (制約・ブロッカー・未完了の検証)
    - `## Subagent Topology` (委譲先サブエージェント / worktree。無ければ「なし」)
    - `## Editing Files` (編集中 file と未保存・未検証の注意)
    - `## Recovery Notes` (圧縮後の自分に向けた引き継ぎメモ。引数の復旧メモがあれば含める)
5. **保存後に state file を Read し直し**、上記見出しがすべて存在することを確認する (Forcing function)。
6. ユーザーに「準備完了。`/compact` を実行してください。」と伝える。

## 完了時に報告すること (Completion receipt)

- state file のパス
- 保存した主要項目
- 未確認・空欄にした項目とその理由
- `準備完了。/compact を実行してください。`

## 関連

- 圧縮後の復旧は `.claude/hooks/userpromptsubmit-compaction-recovery.sh` が
    この state file と TaskList / CLAUDE.md の決定事項を読み戻すよう指示する。
- 60% 通知 (`.claude/hooks/userpromptsubmit-compact-prep-reminder.sh`) がこの skill の実行を促す。
    context 管理の全体像は [CLAUDE.md](../../../CLAUDE.md) の「コンテキスト管理 (compact)」を参照。
