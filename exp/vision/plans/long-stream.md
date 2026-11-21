# ストリームを 600 万 stack まで続ける（V17、計画、2026-11-21）

予想・判定・止める規則は [README の V17](../README.md) に登録した。この文書は、決めた経緯と、実行の手順をまとめたもの。

## 決めた経緯（2026-11-21、ユーザーと）

1. B = 1 の単一ストリームで 600 万 stack を見せると、2〜4 週間かかる（幅 1024 は 2.65 単位/秒）。そのまま回すのは長すぎるので、やめた
2. 100 本のストリームを約 479 秒ずつ離して並べる案（`layout=spread`）も考えた。しかし、この形ではほぼランダムな batch になり、「単一ストリームを長く続けたら」という問いとは別のものになる。そのため外した
3. ユーザーの案: **単一ストリームの連続する 100 stack を 1 回の batch にする**（`layout=chunk`）。B = 1 の run で、10 秒ごとに勾配を平均して 1 回だけ学ぶことと同じになる。データの順序は B = 1 と同じで、計算は 60000 単位で済む
4. RMSProp の一歩は正規化されている（1 回の更新の学習率は δ/τ）。1 回の更新が動画の 10 秒を表すので、時定数は B = 1 の 100 分の 1 にする。$\tau$ の倍率（線形の 100 分の 1 か、平方根の 10 分の 1 か）は勾配の相関で決まるので、3 点を回して B = 1 と比べて選ぶ
5. 比べるための B = 1 の run を、6000 秒まで回す（`v17-b1`）
6. V16 の値に合わせて設計する必要はない。V16 の値は参考にとどめる

## 実装（このブランチ）

- `exp.lib.tours.stacks.StackedTourFrames(layout="chunk")`: ストリーム $i$ は $i \cdot \mathrm{advance}$ から始まり、1 回の `next()` で $n \cdot \mathrm{advance}$ 進む。テストのオラクルは $n = 1$ のストリーム（`tests/lib/tours/test_stacks.py`）
- `data.layout`（`exp.vision.schema.DataConf`、既定は `"spread"`）
- `exp.vision.eval.stream_view`: chunk の run を、それが歩く単一ストリームの run として読む。時刻と frame は $c = B$ 倍になる。`stacks_at` に chunk の設定をそのまま渡すと、例外を出す
- `stream_eval --span --probe-at` と `z_use --span --at`。テストのオラクルは「chunk の run の出力が、同じ重みを $c$ 倍の時刻に持つ B = 1 の run の出力と一致する」と「`--span s` の出力が、duration = s の run の出力と一致する」
- `stream_eval` の `summary.jsonl` に `zurich`（Zurich の val の窓での誤差の比）を足した

## 実行の手順（workflow）

ルーティンは sonnet のエージェントに任せる（[feedback_routine_to_sonnet](../../../memory/feedback_routine_to_sonnet.md)）。計画・判定・結果の読みはメインで行う。

| 段 | 担当 | 内容 |
| -- | -- | -- |
| 1 | Workflow `v17-launch`（sonnet） | `just fleet push` → `up --restart`、std-00 の 4090 で bench（幅 1024、B = 1 と B = 100）、`v17-b1` と `v17-chunk` を投入し、最初の checkpoint まで見届ける。fps・GPU メモリ・所要時間の見込みを返す |
| 2 | メイン（バックグラウンドの待ち） | `just fleet status` を見て、`v17-chunk` が run の 1800 秒を過ぎるのを待つ |
| 3 | Workflow `v17-midcheck`（sonnet） | 止める規則の判定のため、run の 600 秒と 1800 秒の checkpoint を NAS の作業用の場所に写し、`stream_eval --span 6000` を掛けて span を返す |
| 4 | メイン | 止める規則を判定する |
| 5 | Workflow `v17-posthoc`（sonnet） | 終わった run に `stream_eval` と `z_use` を掛け、`just fleet pull` で手元に持ってくる |
| 6 | メイン | 判定（`exp/vision/results/v17/`）、README の結果、資料（`/experiment-report`、本文は `/yomiyasu`） |

### checkpoint（run の秒）

`[0, 0.6, 3, 6, …, 60（3 秒ごと）, 120, 180, 300, 480, 600, 960, 1200, 1800, 2400, 3000, 3600, 4200, 4800, 5400, 6000]`（37 個）。動画の時刻では 0、60、300〜6000 秒（300 秒ごと）、12000、18000、30000、48000（約 1 周）、60000、96000（約 2 周）、120000、180000〜600000 秒（60000 秒ごと）。1 個 284 MB なので、1 本 10.5 GB になる。
