# M14〜M16 の再編の制約（orchestrator、2026-11-03）

ユーザーの指示（/goal）: exp/video/MILESTONES.md の M16 まで全て達成する。実験のコードベースと機能は完全に再編してよい（互換性は無視）。ハイパーパラメータの管理に hydra、最適化に optuna を使ってよい。実験管理のツールを導入してもよい。必要なのは多くの実験を効率的に行えること。全てのマシンを使う。

## 計算資源

- std-00〜03（192.168.100.100〜103）。各 32 コア、61 GB、GPU は RTX 4090（24 GB）と RTX 4060（8 GB）
- 操作は Mac（オーケストレーター）から `ssh std-0X` だけ。ホスト同士は SSH できない
- 各ホストの checkout は `/data/geson/mujou-exp`（Mac は `/Users/geson/Projects/mujou-exp`）。実行は dev container の中（`just container <cmd>`、または `docker exec -u vscode -w /data/geson/mujou-exp mujou-exp-dev-1 ...`。コンテナ名は 4 台とも `mujou-exp-dev-1`）
- NAS が各ホストの `/nas` に CIFS でマウントされている。データセットは `/nas/datasets/{egodex,walking_tours,ego4d,div2k,lsdir}`。コンテナには今 egodex と walking_tours だけがマウントされる（compose.yaml と .env）。自分の領域は `/nas/geson/`
- 1 run = 1 プロセス、仮想時間（`mujou.testing.run` の手動 step）で決定的。CPU 1 コアが律速（GPU のカーネル起動）。M13 の BIG の PC-ALM + トレースは 30 分のストリームで 5〜7 時間以上。よって効率の鍵は「多数の run を全ホストに並べる」こと

## Python の制約

- 学習のプロセスは free-threaded の 3.14t。GIL を戻すものを学習のプロセスに入れない: grpc（`grpc._cython`）、`torch.optim` の生成（`torch._dynamo` を読む）。trackio は orjson がビルドできず入らない
- hydra-core 1.3.8 と optuna 5.0.0 は 3.14t に入り、import できる（確認済み）
- よって optuna は学習のプロセスに入れず、**中央の調整役（Mac か 1 台のホスト）が ask/tell** する。学習のプロセスは hydra の override を受けて走り、結果（中間の評価と最終の目的値）を run のディレクトリに書くだけ
- 依存の追加は `uv add`（pip は禁止）。pyright standard、ruff、docstring は英語、README と計画書は日本語

## 研究の規約（変えない）

- 学習則の中心は PC-ALM（`mujou.pc`）。BP は基準。BP で探索し、PC-ALM と BP の両方で確認する
- ノード間の局所性と非同期性（ノードをまたぐ `.backward()`、グローバルなバリアを持ち込まない）
- テストは理論から導いたオラクル。「損失が下がった」は完了の条件にしない
- 判定の基準は回す前に README に書き、回した後に数字を書く。数値の条件は、見た目と合うことを確かめてから使う（予測を画像に戻して見る）
- seed・設定・mujou の SHA・コードの版を run の記録に残す。checkpoint とログは git に入れない
- 1 マイルストーン 1 PR が原則。ブランチは `exp/<日付>/continuous-stream-<mN>`。インフラの再編は別の PR にしてよい

## 地図

詳細は同じディレクトリの `map-model.md`、`map-run.md`、`map-data.md`、`map-lib.md`、`map-mujou.md`、`map-history.md`、`map-tests.md`。

## 実測のスループット（std-00、2026-11-03）

M13 の BIG（玉 1 つ、B=128）、ストリーム 10 秒分（300 フレーム）、起動の数秒を含む壁時計:

| | CPU | cuda (4090) |
| -- | -- | -- |
| BP | 50 s | 6.2 s |
| BP + trace | 149 s | 8.0 s |
| PC-ALM | 336 s | 24 s |
| PC-ALM + trace | 441 s | 26 s |

PC-ALM + trace を 2 枚の GPU に半分ずつ並べたとき: 4 本で 60 s、8 本で 88 s、16 本で 126 s（1 本 26 s）。16 本でスループットは 1 本の 3.3 倍で、まだ伸びている。1 ホスト 16 スロット前後（全 64）が出発点。30 分のストリームの PC-ALM + trace は単独で約 75 分、16 並列で約 6 時間。
