# exp/

実験のコードは `exp/` に置く。1 実験 1 パッケージの `exp/<name>/`（`__init__.py`、コード、`README.md`）と、実験に共通の `exp/lib/` からなる。`README.md` には仮説、実行方法、結果を書く。実行は `uv run python -m exp.<name>.<script>`。テストは `tests/` に同じ形でミラーする（`exp/<name>/` ↔ `tests/<name>/`、`exp/lib/` ↔ `tests/lib/`）。

## 3 層の棲み分け

1. **mujou に足す**: ノード・ポート・実行・学習則・optimizer の基盤など、mujou の利用者一般が使う機能と、mujou の抽象に足りないもの。mujou の規約（オラクルテスト・docs・CHANGELOG）で保守する価値があるもの。実験側にライブラリの代用品を溜めない（skill `mujou-upstream`）
2. **`exp/lib/` に置く**: 2 つ以上の実験が使う、または特定の実験に閉じない、実験を回すための補助コード（記録・掃引・ホスト・設定・指標）と、研究中の部品。実験の `Config`・データ・モデルに依存せず、mujou・torch・標準ライブラリだけに依存する。mujou の抽象に属すると分かったら mujou に移し、lib からは消す
3. **`exp/<name>/` に置く**: その実験の設定・データ・モデル・スクリプトと、実験の都合だけの暫定物

迷ったときの判定は「mujou の利用者に説明できる機能か（→ mujou）、実験の運用の道具か（→ lib）」。判断が割れたらユーザーに確認する。

依存の向きは `exp/<name>/` → `exp/lib/` → mujou。逆向きの import と、実験どうしの import はしない。

## 実験

| パッケージ       | 内容                                                   |
| ---------------- | ------------------------------------------------------ |
| [video/](video/) | 動画のストリームから世界のシミュレータを PC-ALM で学ぶ |

## exp/lib/

`record`（run の出力）、`sweep`（設定ごとのサブプロセス掃引）、`host`（フレーム単位のループ）、`config`（JSON から dataclass）、`metrics`（実効ランク・probe）、`optim`・`sigreg`・`mingru`・`linear_stream`（研究中の部品と試験台）。使われなかった部品は、次の実験の時点で戻すか消す。

## 成果物

checkpoint、ログ、プロットは commit しない（`runs/`、`outputs/`、`logs/` は無視される）。残したい数字は実験の `README.md` に書く。
