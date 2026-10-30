# CLAUDE.md

このファイルは Claude Code がこのリポジトリを扱う際のガイダンス。常時ロードされるので、手続きの詳細は skill に置き、ここには判断の拠り所だけを書く。

## 応答言語

ユーザーへの応答は日本語で行う。

- 散文・説明・要約・確認・質問・計画は日本語で書く
- コード・コマンド・ファイルパス・識別子・ログ/エラーの引用は原語のまま残す
- コミットメッセージ・PR タイトルは「Git 運用」の規約に従い、日本語化しない
- README（ルートと `experiments/` 配下の実験ごとのもの）・計画書・skill・memory は日本語で書く。docstring とコードのコメントは英語

## プロジェクト概要

`mujou-exp` は、深層学習ライブラリ [mujou](https://github.com/Geson-anko/mujou)（推論と学習が同時に走り続ける、非同期・分散の継続学習ライブラリ）を**使って実験する**プロジェクト。

- mujou は git submodule（`mujou/`）で、そこから editable でインストールする（`pyproject.toml` の `[tool.uv.sources]`）。PyPI 版は使わない
- 実験で mujou に足りない機能が見つかったら、**その場で `mujou/` に追加し、mujou へ PR を送りながら**実験を進める。実験側にライブラリの代用品を溜めない
- 実験は `experiments/<name>/` に 1 実験 1 パッケージで置く（[experiments/README.md](experiments/README.md)）

## 開発原則

慎重さを速度に優先する。trivial なタスクでは判断で簡略化してよい。

1. **実装前に考える**: 仮定を明示し、不確かな点は確認する。より単純な解決策があれば提示する
2. **シンプルさを優先**: 要求されていない機能・抽象化を足さない。問題を解く最小限のコードにする
3. **必要な範囲だけ変更**: 周辺コードをついでに整形・リファクタしない。diff の各行を要求へトレースできる状態にする
4. **ゴール駆動**: 検証可能な成功条件を先に置き、満たすまで実装と検証を繰り返す。実験では「何が分かれば仮説が支持/棄却されるか」を実行前に書く
5. **再現性**: 乱数 seed・設定・mujou のコミット（submodule の SHA）を実験の記録に残す。結果の数字は実験の `README.md` に書き、checkpoint やログは git に入れない

## mujou を変更するとき

`mujou/` の中の変更は mujou 本体の変更であり、**mujou 側の規約が適用される**。着手前に [mujou/CLAUDE.md](mujou/CLAUDE.md) と [mujou/memory/MEMORY.md](mujou/memory/MEMORY.md) を読む。特に次は mujou の不変の原則で、実験の都合で崩さない。

- ノード間の局所性・非同期性（ノードをまたぐ `.backward()`、グローバルなバリア等を持ち込まない）
- テストは理論から導いたオラクルで検証する。公開 API を変えたら `docs/`（en / ja）、`CHANGELOG.md`、`just stub` まで揃える

mujou の hook と skill は、このリポジトリのルートで起動した Claude Code には**自動では効かない**（`.claude/` はルートのものだけが読まれる）。局所性の hook による警告がないので、変更を自分で `mujou/.claude/skills/locality-and-async/SKILL.md` と突き合わせる。

手順の詳細は skill `mujou-upstream`。要点:

- submodule は detached HEAD になりがち。**必ずブランチを切ってから**編集する（`mujou/` で `main` から `<種別>/<日付>/<内容>`）
- mujou の PR は `Geson-anko/mujou` の `main` 宛。mujou の検証は `cd mujou && just run`
- **push していない mujou のコミットを指す submodule の更新を、このリポジトリでコミットしない**（`just mujou-status` で確認）。他の環境で clone / checkout できなくなる
- PR が merge されたら `just mujou-update` で `origin/main` に戻し、ポインタの更新を `chore(mujou): ...` でコミットする

## ツーリング

- パッケージ・環境管理: `uv`（`uv.lock` をコミットする）。**`pip install` は使わない**（hook がブロックする）。依存追加は `uv add` / `uv add --dev`。mujou の依存を変えるときは mujou 側の `pyproject.toml` を直す
- Python: 3.14 以上。mujou に合わせて**開発は free-threaded 版の 3.14t**（`.python-version`）
- タスクランナー: `just`（1.58 以上）
- 型チェッカー: `pyright`（`experiments/` と `tests/`、standard）。`mujou/` は対象外（mujou 自身が strict で検査する）
- リンター/フォーマッター: `ruff`（line-length 88）、pre-commit。`mujou/` は pre-commit の対象外
- `uv sync` は mujou の Rust 拡張を maturin でビルドするので Rust（rustup）が要る。ホストに無ければ dev container で作業する

## コマンド

| レシピ                 | 内容                                                                |
| ---------------------- | ------------------------------------------------------------------- |
| `just setup`           | submodule の取得 + `uv venv` + `uv sync` + pre-commit install       |
| `just format`          | pre-commit フックを全ファイルに実行                                 |
| `just test`            | このプロジェクトの pytest（mujou のテストは含まない）               |
| `just test-fast`       | `slow` を除いた pytest。反復中に使う                                |
| `just type`            | pyright                                                             |
| `just run`             | format → test → type。**コミット前の検証はこれ**                    |
| `just mujou-status`    | submodule の状態。checkout 中のコミットが origin に無ければ失敗する |
| `just mujou-update`    | submodule を `origin/main` の最新へ進める                           |
| `just container <cmd>` | dev container の中で、今いる checkout の `<cmd>` を実行する         |

単一テスト: `uv run pytest tests/test_x.py -v` / キーワード: `uv run pytest -k "<expr>"`

## Git 運用

- `main` から `<種別>/<日付>/<内容>` で分岐する（例: `feature/2026-10-30/first-pc-experiment`）。種別は `feature`, `fix`, `refactor`, `docs`, `chore`、実験は `exp`
- `main` に直接 commit しない。main への merge はユーザー判断
- commit は `<種別>(<スコープ>): <内容>`、1 commit 1 関心事。`just run` が通る前に commit しない
- force push、`git reset --hard`、`--no-verify`、未確認の破壊的操作を行わない

## エージェント運用

- 既定はシングルエージェント。サブエージェントやチームは、ユーザーが明示したときだけ使う
- ユーザーへの質問は作業の区切りにまとめて出し、推奨案で進める（[memory/feedback_questions_at_end.md](memory/feedback_questions_at_end.md)）
- 中間メモは `memory/agents/<agent-name>/<task-slug>.md`

## Hooks（`.claude/settings.json`）

| hook                               | 役割                                                   |
| ---------------------------------- | ------------------------------------------------------ |
| `pretooluse-guard-bash.py`         | `pip install` 系と `git commit --no-verify` をブロック |
| `compaction-recovery.sh` ほか 2 本 | compact 前後の状態退避と復旧                           |

hook は `python3` と `bash` だけに依存し、fail-open。テストは `tests/test_claude_hooks.py`。statusLine（`.claude/scripts/statusline.sh`）が context 使用率を表示し、60% を超えると `/compact-prep` を促す。

## メモリ参照

プロジェクト固有の規約・知見・ユーザーの好み・未決定事項は repo ルートの [memory/](memory/) に保存する（git 管理対象）。セッション開始時に [memory/MEMORY.md](memory/MEMORY.md) を確認する。新しい規約・フィードバック・決定が判明したら 1 メモリ 1 ファイルで足し、`MEMORY.md` から 1 行リンクを張る。

## 索引（skill）

- mujou への機能追加と PR: `/mujou-upstream`
- Git: `/github-pr` / `/merge-main` / `/do-on-worktree`
- 並列化: `/maximize-parallels` / `.claude/` 編集: `/edit-dot-claude`
- 日本語の推敲: `/yomiyasu`（日本語ドキュメントを書く・直すときに使う）
- context 管理: `/compact-prep` / 言語切替: `/japanese`
