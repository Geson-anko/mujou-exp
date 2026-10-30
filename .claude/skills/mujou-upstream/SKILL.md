---
name: mujou-upstream
description: 実験中に mujou（git submodule の mujou/）へ機能追加・修正が要ると分かったときの手順。submodule 側でブランチを切って実装し、mujou へ PR を送りながら、実験側はその変更を使い続ける。submodule のポインタの扱い（未 push のコミットを指さない）も含む。「mujou に足りない」「mujou を直して」「mujou に PR」と言われたとき、または experiments/ から mujou の機能を足したくなったときに参照する。
---

# mujou に変更を入れて PR を送る

実験で mujou に足りないものが見つかったら、実験側に代用品を作らず、`mujou/` に直接足して upstream へ PR を送る。実験は `mujou/` の作業ブランチを使って止まらずに進める。

## 前提

- `mujou/` は https://github.com/Geson-anko/mujou の submodule。`pyproject.toml` の `[tool.uv.sources]` で editable に入っているので、`mujou/` の編集は次の `uv run` で反映される（Rust を変えたら再ビルドが走る）
- `mujou/` の中の変更は mujou 本体の変更。[mujou/CLAUDE.md](../../../mujou/CLAUDE.md) と [mujou/memory/MEMORY.md](../../../mujou/memory/MEMORY.md) を先に読み、その規約に従う（局所性・非同期性、オラクルによるテスト、`CHANGELOG.md`、`docs/` の en / ja、bindings を変えたら `just stub`）
- mujou の skill（`mujou/.claude/skills/`）は自動では見えない。`locality-and-async`・`testing-strategy`・`refactor-conventions` は必要に応じて直接 Read する
- 大きな変更（公開 API の変更や追加・依存の追加・実行モデルの変更）は、mujou の CONTRIBUTING に従い先に issue で設計を合意する。ユーザーに確認してから進める

## 手順

### 1. 状態の確認

```bash
just mujou-status        # checkout 中のコミット、ブランチ、未コミットの変更
git -C mujou fetch origin
```

未 push のコミットや未コミットの変更が残っていたら、先にそれを片付ける（どの作業の続きかユーザーに確認する）。

### 2. ブランチを切る

submodule は通常 detached HEAD。そのまま編集しない。

```bash
cd mujou
git switch -c <種別>/<日付>/<内容> origin/main     # 例: feature/2026-10-30/pc-layer-hook
```

すでに別の mujou PR が未 merge で、その上に積む必要があるときだけ、そのブランチから切る（PR の base も合わせる）。

### 3. 実装と検証

mujou の規約どおりに、テスト → 実装 →（必要なら）docs / CHANGELOG / stub を行う。検証は mujou 側で通す。

```bash
cd mujou && just run
```

mujou の `just run` は docs のビルドまで含む。反復中は `just test-fast` / `uv run pytest <path>`。実験側の `just run` では mujou のテストは走らない。

### 4. mujou のコミットと PR

mujou の規約（`<種別>(<スコープ>): <内容>`、1 commit 1 関心事、main へ直接 commit しない）でコミットし、skill [github-pr](../github-pr/SKILL.md) の手順で `mujou/` の中から push して PR を作る（base は `main`）。PR の本文は mujou の PULL_REQUEST_TEMPLATE に従う。マージはしない。

### 5. 実験側でその変更を使う

PR の merge を待たずに実験を進められる。ただし、このリポジトリの submodule ポインタを更新するコミットは **mujou のコミットを push してから**行う。

```bash
just mujou-status        # "OK: ... is on a remote branch" を確認
git add mujou
git commit -m "chore(mujou): <何を取り込んだか> (<PR の URL か番号>)"
```

`mujou-status` が失敗するときは push 漏れ。ポインタをコミットしない（他の clone / container で submodule を取得できなくなる）。

### 6. PR が merge されたら

```bash
just mujou-update        # origin/main へ進めて uv sync
just mujou-status
git add mujou && git commit -m "chore(mujou): bump to main (<PR>)"
```

squash merge された場合、作業ブランチのコミットは main に無いので、必ず `origin/main` に進めてからポインタをコミットする。作業ブランチは `git -C mujou branch -d` で消してよい。

## 注意

- mujou の変更とそれを使う実験の変更は、コミットを分ける（mujou は mujou のリポジトリ、実験はこのリポジトリ）
- 実験の結果を再現するのに必要な mujou の版は、このリポジトリの submodule ポインタが決める。実験の記録にも SHA を残す
- `git submodule update` は、ローカルのブランチ上の未 push のコミットから離れて detached HEAD にする。作業中に実行しない（コミット自体は reflog に残る）
- mujou の変更が実験の都合だけの暫定物なら upstream に送らず、実験側に置く。判断が割れたらユーザーに確認する
