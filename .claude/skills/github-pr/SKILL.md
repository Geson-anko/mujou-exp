---
name: github-pr
description: 現在のブランチを GitHub に push し、gh で対象ブランチ（デフォルト main）へのプルリクエスト（PR）を作成する手順。「PR を出して」「プルリクを送って」「ブランチをプッシュして PR」と言われたときに参照する。
---

# GitHub へ push して PR を作成する

現在のブランチを remote に push し、`gh` CLI で対象ブランチへの PR を作成する。

引数で対象ブランチを指定できる。**指定がなければ `main`**。

## 手順

### 1. 事前チェック

```bash
git branch --show-current   # 現在ブランチの確認
git status --short          # 未コミット変更の有無
git log <target>..HEAD --oneline  # 対象ブランチに対する差分コミット
```

- 現在ブランチが `main` の場合は中止してユーザーに確認する（main から直接 PR は出さない）
- 未コミットの変更がある場合は中止し、先にコミットするかユーザーに確認する
- 対象ブランチに対する差分コミットが 0 件なら PR を出しても意味がないので、その旨を報告して中止する
- `git fetch origin <target>` の後 `git log HEAD..origin/<target> --oneline` に commit が並ぶなら、push の前に skill [merge-main](../merge-main/SKILL.md) で取り込む

既存 PR と重複しないか確認する。

```bash
gh pr list --head "$(git branch --show-current)" --state open
```

既に open な PR があれば新規作成せず、その PR の URL を報告する（push のみ行えば PR は自動更新される）。

### 2. push

```bash
git push -u origin "$(git branch --show-current)"
```

### 3. PR 作成

`gh pr create` は引数が足りないと対話モードに入るので、必ずフラグを渡して非対話で実行する。

```bash
gh pr create \
  --base <target> \
  --head "$(git branch --show-current)" \
  --title "<タイトル>" \
  --body "<説明>"
```

- **タイトル**：コミット規約と同じ `<種別>(<スコープ>): <内容>` 形式。コミットが 1 件ならそのメッセージをそのまま使う
- **説明**：変更の要約（何を・なぜ）。複数コミットなら主要な変更点を箇条書きにする
- 説明に改行を含める場合は `--body "$(cat <<'EOF' ... EOF)"` 形式を使う

### 4. 結果報告

作成された PR の URL をユーザーに報告する。

## 注意点

- **マージはしない**。`main` への merge はユーザー判断（CLAUDE.md の Git 運用に従う）。`gh pr merge` は実行しない
- 検証（`just run`）が未実施の変更を含む場合は、PR 作成前に実行する
- push が reject された場合（remote が先行）は `git pull --rebase` の要否をユーザーに確認する。`--force` 系は使わない
- `.github/workflows/` を変更する PR を push するには `gh auth` の token に `workflow` scope が要る
