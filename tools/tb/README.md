# TensorBoard のビュー（`tools/tb/`）

run のディレクトリ（`exp/lib/fleet/README.md` の契約のファイル）を TensorBoard の event ファイルに書き出し、ブラウザで見る。run の中では何も書かない。書き出しは run が終わった後に、手元へ持ってきたディレクトリから行う。

学習の venv とは別の uv プロジェクトにしてある（Python 3.13、`tensorboard==2.21.0`、`tensorboardX==2.6.5`）。tensorboard が使う grpcio などが free-threaded の 3.14t に wheel を持たないため。torch は入れない。

## 使い方

```bash
just fleet pull --batch m17-wt-confirm-v1 --batch m17-wt-confirm-v1-post   # ./runs/ へ
just tb export --batch m17-wt-confirm-v1 m17-wt-confirm-v1-post            # ./tb/ へ書く
just tb view                                                                # http://localhost:6006
```

- `export` は `--batch B...`（`--root` の下、既定は `runs/`）か `--runs DIR...`（run か batch のディレクトリ）を受け取り、`--out`（既定は `tb/`）の下に run ごとのディレクトリ `<batch>/<run_id>/` を作る。同じ run を書き出し直すと、前の event ファイルを消して書き直す
- `view` は `tensorboard --logdir tb --port 6006` を起動する。`tb/` は git に入れない
- std-00 で書き出して Mac で見るときは `ssh -L 6006:localhost:6006 std-00` で転送する

## 書き出すもの

| TensorBoard の tag                            | 元のファイル                                                                                                                                                                                           | step                       |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------- |
| `stream/<key>`                                | `stream.jsonl`                                                                                                                                                                                         | `frames`                   |
| `eval/<split>/<key>`                          | `evals.jsonl`                                                                                                                                                                                          | `round(t)`（仮想時間の秒） |
| `base/<split>/run_init/<key>`                 | `baselines.json`（学習前の視覚）                                                                                                                                                                       | 0 と最後の step（横線）    |
| `post/<split>/<key>`                          | posthoc の `post/decode/metrics.jsonl`（`code` が `vit` の行）                                                                                                                                         | `round(t)`                 |
| `base/<split>/<code>/<key>`                   | posthoc の `post/decode/baselines.jsonl`（`pca`・`init`）                                                                                                                                              | 0 と最後の step（横線）    |
| 画像 `post/recon`・`post/restore`・`post/dyn` | posthoc の `<t>.png`・`<t>-restore.png`・`<t>-dyn.png`                                                                                                                                                 | `round(t)`                 |
| 画像 `base/<code>/<kind>`                     | `init*.png`・`pca*.png`                                                                                                                                                                                | 0                          |
| テキスト `run`                                | `job.json`・`CODE.json`・`provenance/`                                                                                                                                                                 | 0                          |
| HParams                                       | config の `vision.*`・`optim_vision.*`・`learner.kind`・`seed`・`name`、指標は最後の val の行（`hp/<key>`）と posthoc の最後の `vit` の val の行（`hp/post/<key>`、同じ export に posthoc があるとき） | 最後の step                |

- `task=posthoc` の job のディレクトリ（`job.json` に `parent` がある）は、親の run と同じディレクトリに書く。run と後からの評価が 1 つの run として並ぶ
- 値が null のもの、数でないものは書かない。PNG はそのままのバイト列で載せる（幅と高さは IHDR から読む）
- Walking Tours のフレームは deface を通していない。ビューは手元だけで見て、外に出す画像（`report.html`）はぼかしてから載せる

## テスト

```bash
just test-tb   # 偽の run と posthoc を書き出し、tensorboard の EventAccumulator で読み戻す。pyright も
```
