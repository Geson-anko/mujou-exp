# mujou-exp

継続学習のための非同期・分散深層学習ライブラリ [mujou](https://github.com/Geson-anko/mujou) を使って実験するプロジェクト。

mujou は git submodule（`mujou/`）で、editable でインストールする。実験で mujou に足りない機能が見つかったら、submodule 側に追加し、mujou へ PR を送りながら実験を進める。

## セットアップ

```sh
git clone --recurse-submodules https://github.com/Geson-anko/mujou-exp.git
cd mujou-exp
just setup   # git submodule update --init, uv venv --clear, uv sync, pre-commit install
```

`uv sync` は mujou のネイティブモジュール（Rust、maturin 経由）をビルドする。そのため [uv](https://docs.astral.sh/uv/)、[just](https://just.systems/) 1.58 以上、[rustup](https://rustup.rs/) が要る。dev container には全部入っている。

## Dev container

`.devcontainer/` は mujou と同じ環境で、Ubuntu に uv、just、gh、Rust ツールチェーン、Claude Code が入っている。チェックアウトを VS Code で開いて "Reopen in Container" を選ぶか、[Dev Container CLI](https://github.com/devcontainers/cli) 経由で任意のコマンドを実行する。

```sh
just container just run
```

コンテナはホストの `.venv/` と衝突しないよう、専用の仮想環境 `.venv-devcontainer/` を使う。

## コマンド

| レシピ              | 内容                                                              |
| ------------------- | ----------------------------------------------------------------- |
| `just setup`        | submodule の取得、仮想環境の作成、pre-commit フックのインストール |
| `just format`       | pre-commit フックを全ファイルに実行                               |
| `just test`         | このプロジェクトの pytest                                         |
| `just test-fast`    | `slow` を除いた pytest                                            |
| `just type`         | `experiments/` と `tests/` への pyright                           |
| `just run`          | format → test → type。**コミット前に実行する**                    |
| `just mujou-status` | submodule の状態。コミットが origin に無ければ失敗する            |
| `just mujou-update` | submodule を `origin/main` の最新へ進める                         |
| `just container …`  | dev container の中でコマンドを実行する                            |

## mujou を変更するとき

[CLAUDE.md](CLAUDE.md) と skill `mujou-upstream`（`.claude/skills/mujou-upstream/SKILL.md`）を参照。
