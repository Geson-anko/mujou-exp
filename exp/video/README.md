# video（旧 continuous_stream）: 動画のストリームから世界のシミュレータを PC-ALM で学ぶ

2026-11-01 に `experiments/continuous_stream/` から `exp/video/` に移し、`video.py` を `balls.py` に改名し、汎用部を `exp/lib/` に移した。振る舞いは変えていない（移動前後で `metrics.jsonl` が一致）。[results/](results/) の M12 までのファイル名は当時のもの。

計画は [PLAN.md](PLAN.md)、実装と検証の単位は [MILESTONES.md](MILESTONES.md)。結果の数字はマイルストーンごとに [results/](results/) に置く（下の表）。M6〜M10 の結果を、入力の動画とグラフで読める 1 枚のページにまとめたものが [report.html](report.html)（ブラウザで開く。データは埋め込み済みで、`runs/` から作った）。M11〜M12（|z| の発散、`rollout10`、予測のデコード、EgoDex）は [report-m11-m12.html](report-m11-m12.html) に同じ形でまとめた。

## 今の方針（2026-11-02）

動力学と視覚を分けて 1 つずつ確かめ、最後に合わせる（MILESTONES.md の「これからの流れ」）。M14 で玉の状態を観測にして動力学だけを学ぶ。M15 で画像・動画のエンコーダを JEPA で学ぶ。M16 でその 2 つを同時に学ぶ。M13 では、潜在空間で予測誤差をとって同時に学ぶと、潜在空間が予測しやすいように歪んだため。

## 決定事項（2026-11-01）

段 3 はこの設定から始める（PLAN.md 12 節）。

- optimizer は SGD、学習率 0.01。AdamW・白色化・射影はまだ使わない
- `stop_target` を入れる
- SIGReg の重みは $\kappa=0.8$
- エンコーダの初期化は `model.standardize_`
- 学習規則は PC-ALM、`learn_interval` は 25 ms、relaxer は `pc.relax.Euler`

M11（同日）で次を足した。段 3 にはこれを持ち込む。

- $z$ を半径 $\sqrt D$ の球面に載せる（`z_norm`）。SIGReg による $\lvert z\rvert$ の押し広げと発散がなくなった
- 動力学は、半分の時間は自分の予測から先を予測して学ぶ（`open_loop=0.5`, `open_cycle=20`）。`rollout10` が 1.6〜1.9 から 1.29 に下がった
- 学習率は 0.03（`z_norm` のもとでは 0.1 でも発散しない）

残る課題は、`rollout10` が定数の予測（`mean10` 約 1.1）にまだ届かないこと。EgoDex（M12）では、エンコーダを 3 ノード、動力学を 3 ノードに分ける。

## 仮説

PLAN.md 8 節の H0〜H4。各段の冒頭で閾値を確定させ、結果のファイルに写す。

## 実行方法

M13 までのスクリプト（`run.py`・`easy.py` など）は 2026-11-03 に消した（コードは git の 433a72e に残る）。設定は hydra の preset（`conf/experiment/`、型は `schema/`）で書き、`exp.video.compose.compose` で解決する。run の入口は `main.py`、プログラム（`programs/`）は設定の `program` で選ぶ。

```bash
# 手元で 1 本（<out> が run のディレクトリ。止めた run は同じコマンドで再開する）
uv run python -m exp.video.main +experiment=m14/s2 learner=pc seconds=60 --out /tmp/m14
# 4 台で（exp/lib/fleet/README.md）: snapshot を配り、投げ、結果を ./runs/ へ
just fleet push
just fleet submit --batch m14-s2-v1 +experiment=m14/s2 learner=pc --seeds 0,1,2 --res cpu
just fleet pull --batch m14-s2-v1
# optuna の探索と、上位の設定の確認（tools/sweep/README.md、study は studies/）
just sweep run exp/video/studies/m14_s3_bp.toml
just sweep confirm m14-s3-bp-v1 --top 2 --learners bp,pc --seeds 0,1,2 --res cpu --batch m14-s3-confirm-v1
```

run のディレクトリの中身（`config.yaml`・`evals.jsonl`・`stream.jsonl`・`result.json`・`ckpt/`・`viz/` など）は `exp/lib/rundir.py` と fleet の README の契約のとおり。checkpoint とログは git に入れない。

## 結果

| | 結果 | 結論 |
| -- | ---- | ---- |
| M5 | [計算時間](results/m05.md) | SGD は CPU 1 スレッドで実時間の 1.07 倍。段 2 の大きさでは GPU は速くない |
| M6 | [段 0: 線形回帰のストリーム](results/m06.md) | H0 は棄却（前半は支持、後半が棄却）。段 2 は grafting ありで比べる |
| M7 | [段 1: 合成動画での勾配の観測](results/m07.md) | 前提（ランク 32 以上）を満たさず、H1 はどちらとも言えない |
| M8 | [段 2: 世界のシミュレータと optimizer の比較](results/m08.md) | 表現が崩壊していたので中断し、M9 の後に M10 として回し直した |
| M9 | [表現の崩壊への対処](results/m09.md) | `stop_target`、$\kappa=0.8$、学習率 0.01 で 3 seed とも条件を満たす |
| M10 | [ストリームに合う optimizer と relaxer](results/m10.md) | どの optimizer も SGD を超えない。段 3 には SGD と `pc.relax.Euler` |
| M11 | [SIGReg による \|z\| の押し広げと `rollout10`](results/m11.md) | \|z\| は `z_norm` で直った。潜在の `rollout10` < 1 は動力学の証拠にならず、条件を画像空間に改めた |
| M12 | [段 3: EgoDex](results/m12.md) | 残差の予測で `rollout10` 0.37〜0.50。写すだけ（0.10〜0.14）には届かない |
| M13 | [合成動画を簡単にした土台](results/m13.md) | 未達のまま中断。玉 1 つのオンラインは適格度トレースで 5・10 フレーム先 0.49・0.92 倍（平均の画像との比） |
| M14 | [動力学だけを学ぶ（玉の状態を観測に）](results/m14.md) | 段 1・2 は達成（段 2 の #45 で test の `roll5/cv5` 0.38〜0.44、`roll10/cv10` 0.25〜0.30、PC-ALM と BP、3 seed）。画素を観測にした段 3 は等加速度の外挿の 1.9〜2.2 倍で届かない |
| M15 | [画像のエンコーダを JEPA で学ぶ](results/m15.md) | 達成。JEPA に spatial broadcast decoder による最後のフレームの再構成を足し、test の `probe_pos_min` 0.957〜0.979（PC-ALM と BP、3 seed）。JEPA だけでは 0.05〜0.2 で、玉の位置が線形に読めなかった |
