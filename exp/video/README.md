# video（旧 continuous_stream）: 動画のストリームから世界のシミュレータを PC-ALM で学ぶ

2026-11-01 に `experiments/continuous_stream/` から `exp/video/` に移し、`video.py` を `balls.py` に改名し、汎用部を `exp/lib/` に移した。振る舞いは変えていない（移動前後で `metrics.jsonl` が一致）。[results/](results/) の M12 までのファイル名は当時のもの。

計画は [PLAN.md](PLAN.md)、実装と検証の単位は [MILESTONES.md](MILESTONES.md)。結果の数字はマイルストーンごとに [results/](results/) に置く（下の表）。M6〜M10 の結果を、入力の動画とグラフで読める 1 枚のページにまとめたものが [report.html](report.html)（ブラウザで開く。データは埋め込み済みで、`runs/` から作った）。M11〜M12（|z| の発散、`rollout10`、予測のデコード、EgoDex）は [report-m11-m12.html](report-m11-m12.html) に同じ形でまとめた。M14〜M16（動力学だけ、視覚だけ、同時の学習、予測だけで回した 5 秒、速さ）は [report-m14-m16.html](report-m14-m16.html)。Walking Tours のフレームは Deface で顔をぼかしてから載せた。

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

M17（Walking Tours でエンコーダを詰める）は、preset `m17/wt_n16`・`wt_n8`・`wt_n1`（`vision.layout=packed`、`vision.recon` は 0 で固定）で回し、エンコーダの評価は後から `task=posthoc` で行う。posthoc は物差し（画素の PCA と学習前の ViT）、$z$ からの再構成、$z$ の上の線形の動力学、予測器を通した復元を測る（`measure/encoder_eval.py`）。探索は `studies/m17_wt_bp.toml`（`objective_from = "posthoc"`、tools/sweep/README.md）。run の 1 フレーム（仮想時間の 1/30 秒）にストリームは動画を 3 フレーム（0.1 秒）進める（`data.tours.advance=3`）。動画は仮想時間に対して 3 倍速で流れ、間の 2 フレームはどのノードも推論しない。

トークン列を表現にする preset は `m17/tok_n16`（2026-11-12 の決定）。`wt_n16` と同じ積み・エンコーダ・optimizer で、最後の段が各トークンを `vision.token_dim` = 16 次元に線形に写す（32 次元は `vision.token_dim=32` で上書きする）。SIGReg は、トークンを 16 次元の 1 つの分布の標本とみてかける（`token_sigreg=raw`。$T$ で割らず、積みの平均も引かない）。標本には、いまのトークンに加えて、過去のターゲットのトークンも入れる。過去のトークンのバッファの大きさと、毎フレーム入れる数は、ストリーム 1 本あたりで決める（`vision.token_buffer` = 2048、`vision.token_insert` = 512。$B$ 本なら全体で $2048B$ 行、毎フレーム $512B$ 個）。512 は 1 積みのトークンの数 $T$ で、有効なストリームのターゲットのトークンをすべて入れる（`token_insert` は $T$ 以下）。空いた行を埋めたあとは、ランダムに選んだ行と入れ替える（`jepa/nodes.py` の `PastTokens`）。行が 1 フレームを生き残る確率は $1 - 512/2048 = 3/4$ で、行の年齢は平均 4 フレーム（ストリームの時間で 0.4 秒）の幾何分布になる。この分布は $B$ によらない。シングルストリームは `data.streams=1` で回す。held-out の評価の損失は、`data.streams` によらず 32 積みずつの batch でとる（`programs/vision.py` の `PACKED_BATCH`）。SIGReg の値は標本数とともに増えるので、$B$ の違う run を同じ標本数で比べるためである。run の長さは 1800 秒で、300 秒ごとにチェックポイントを残す。$\lambda$（`vision.sigreg`）の 0.003 は仮の値で、実際のトークンが積みの中でどれだけ相関するかを測ってから、掃引で決める。posthoc は積みの符号を、平らにしたトークン列（512 × 16 = 8192 次元）にする（`measure.posthoc_code=tokens`）。デコーダ・線形の動力学・`rank`・`z_spread` も、学習前の ViT も、この符号で測る。PCA も同じ次元にするが、学習のフレーム数がその次元に足りないときは、取れる最大の成分数にし、行に `pca_rank` として書く。

SIGReg の代わりに I-JEPA の EMA のターゲットエンコーダを使う preset は `m17/ema_t512`・`ema_t128`・`ema_t64`（2026-11-13 の決定）。単一ストリームでは SIGReg に要る i.i.d. な標本を集められないためである。予測のターゲットは、エンコーダの EMA のコピーに積み全体を通したトークンになる。コピーは勾配を受けず、学ぶエンコーダを時間で追う。$\tau\,\dot{\bar\theta} = \theta - \bar\theta$ を、ステップの間 $\theta$ を一定として厳密に解くと、`elapsed` 秒のステップでは $\bar\theta \leftarrow a\bar\theta + (1 - a)\theta$、$a = m^{\mathrm{elapsed}/\mathrm{frame}}$ になる（`jepa/ema.py`、`vision.ema` = $m$ = 0.996 はフレームあたりの値）。1 フレームのステップでは I-JEPA の式そのものになる。mujou の optimizer と同じく、時定数は秒で決まり、学ぶ回数によらない。1 フレームでストリームは 0.1 秒進むので、時定数 $-\mathrm{frame}/\ln m$ ≈ 250 フレームはストリームの 25 秒で、$B$ にも `time.learn_interval` にもよらない。I-JEPA は $m$ を決まったエポック数で 1 まで上げるが、続くストリームにはエポックがないので、M17 では一定にする。`vision.ema` = 0 はターゲットエンコーダを作らない（2026-11-13 より前の run の設定と checkpoint はそのまま読める）。SIGReg の項はない（`vision.sigreg=0`。stats の `sigreg` は重み 0 の監視の値）。過去のトークンのバッファも使わない。ターゲットに LayerNorm はかけない。$d$ = 2 のトークンに affine なしの LayerNorm をかけると ±1 になり、中身が残らないからである。表現の大きさは $D = Td$ = 1024 で、3 つの腕で $T$ と $d$ の分け方を変える。

| preset         | patch | tubelet | $T$                   | $d$ | short / long の 1 slot の見える patch |
| -------------- | ----- | ------- | --------------------- | --- | ------------------------------------- |
| `m17/ema_t512` | 16    | 2       | 512（8 slot × 8 × 8） | 2   | 64 のうち 18 / 8（隠す割合 0.72 / 0.88） |
| `m17/ema_t128` | 32    | 2       | 128（8 slot × 4 × 4） | 8   | 16 のうち 4 / 1（0.75 / 0.94）           |
| `m17/ema_t64`  | 32    | 4       | 64（4 slot × 4 × 4）  | 16  | 16 のうち 4 / 1（0.75 / 0.94）           |

マスクの設定は `wt_n16` のまま変えない（`tests/video/jepa/test_masks.py` の `TestTheFourByFourGrid`）。ストリームは 1 本（`data.streams=1`）。それ以外は、optimizer・1800 秒・300 秒ごとのチェックポイントも含めて `tok_n16` と同じ。PC-ALM では、各エンコーダノードが自分の段のコピーを持ち、自分の optimizer のステップの直後に動かす（`ema.TargetStage`・`ema.Following`）。ノードの活動は、詰めたトークンのあとにコピーの経路の $T$ トークンを運び、ヘッドはそれをターゲットとして読む。経路には credit が流れない。held-out の評価の `pred` と補完の誤差（`comp_*`）も、学習と同じくターゲットエンコーダのトークンをターゲットにする。posthoc も、ターゲットエンコーダのトークンで測る。デコーダと線形の動力学の符号、`rank`・`z_spread`、復元のトークンのデコーダとその上限（`restore_target_*`）がこれにあたる。I-JEPA は下流の評価にターゲットエンコーダを使う。予測器の出力もターゲットエンコーダのトークンへ向けて学ぶので、学ぶエンコーダのトークンで学んだデコーダでは、別の空間の値を描くことになる。復元で見える位置は、文脈の view をターゲットエンコーダに通したトークンで描く（予測器が読むのは、学ぶエンコーダの文脈のトークン）。ターゲットエンコーダのない run は変わらない。チェックポイントにはターゲットエンコーダも `target` として入る。

```bash
# 3 つの腕（BP、seed 100）。PC-ALM は learner=pc
for arm in t512 t128 t64; do
  just fleet submit --batch m17-ema-v1 +experiment=m17/ema_$arm learner=bp --seeds 100 --res 4090 --gpu-mem 20
done
# posthoc は下と同じく task=posthoc で親の run を指す
```

単一ストリームのデータポイントの SIGReg の preset は `m17/ssig_t512`・`ssig_t128`・`ssig_t64`（2026-11-13、ユーザーの仮説）。トークン化は同じ名前の EMA の腕と同じで、ターゲットエンコーダはない（`vision.ema` = 0）。SIGReg は、積み 1 つを 1 標本としてかける（`vision.sigreg_on=datapoints`）。標本は、ターゲットのトークンを平らにした列（$D = Td$ = 1024）である。SIGReg が見る標本は、いまの積みと、過去のデータポイントのバッファ $N$ = 256 個（`vision.datapoint_buffer`、ストリーム 1 本あたり）を合わせたもの。バッファには、ヘッドが見た $k$ フレームに 1 つ（`vision.datapoint_every`）を入れる。1 フレームはストリームの 0.1 秒である。run は仮想時間で、どのノードもすべてのフレームを通る（`mujou.testing.run`）。そのため、ヘッドが見たフレームはストリームのフレームと同じになる。満ちたあとは、ランダムに選んだ行と入れ替える（`jepa/nodes.py` の `PastPoints`）。行が 1 回の挿入を生き残る確率は $1 - 1/N$ で、行の年齢は平均 $Nk$ フレームの幾何分布になる（$k$ = 1・6・24 で 25.6 秒・2.6 分・10.2 分）。バッファに貯めるものは 2 通りある（`vision.datapoint_store`）。

- `z`（既定）: 表現を detach して貯める。勾配はいまの標本にだけ届く。PC-ALM では、ヘッドのノードが自分のバッファを持つ。貯めた行は、それを作ったときのエンコーダのもので、平均 $Nk$ フレーム（学習ステップ）前の重みで作られている。$k$ を変えると、バッファが覆う時間とこの古さが一緒に変わる
- `input`: 積みのターゲットの view の画素（u8）を貯め、損失を計算するたびに今のエンコーダで符号化し直す。勾配はすべての標本に届く（LeJEPA と同じ）。BP だけで使える。PC-ALM のヘッドはエンコーダを持たないので、`learner=pc` では設定のエラーになる

$\lambda$（`vision.sigreg`）の下限は次のように導いた。積みの中でトークンが一定の解では、予測の誤差が 0 になる。ところが、平らにした列を乱数の向きに射影すると、この解は $d$ が大きいほど $N(0, 1)$ に近く見える。そのため、この解が i.i.d. なトークンより高くつくのは、$\lambda > \lambda^*$ のときだけである。$n$ = 257 標本では、$\lambda^*$ は $d$ = 2・8・16 で 0.037・0.116・0.213 になる（`python -m exp.video.diag.ssig bound`。テストは `tests/video/schema/test_presets.py`）。i.i.d. なトークンの尺度もエンコーダが選べるとすると、0.030・0.109・0.206 になる（`lam_star_scaled`）。この下限が除くのは、積み 1 つにつき $d$ 個の数の符号（$m = d$）だけである。文脈から決まる $m > d$ 個の数の潜在から作るトークンも、予測の誤差は 0 になる。$m$ が大きいほど、その統計量は $S_0$ に近づく。そのため、どの $\lambda < 1$ でも、この解は健全な解より安い。この解が崩壊かどうかは $m$ による。それは学んだ符号の rank でしか分からない。preset の $\lambda$（t512 0.0442、t128 0.134、t64 0.233）は、腕ごとの格子の真ん中である。格子は、健全な解が崩壊と i.i.d. なデータポイントの両方に勝つ窓から決めた。窓には、バッファの標本どうしのストリームの相関を入れた。相関は、各腕の学習前の ViT（同じトークン化の EMA の run の `init`）の、白色化したトークン列で測った（`python -m exp.video.diag.ssig_window --own`、results/m17.md の「各腕の学習前の相関で導き直した格子」）。格子の両端は `vision.sigreg` で上書きする（t512 0.0348・0.056、t128 0.112・0.162、t64 0.207・0.263）。窓の下端も、$m = d$ の符号を除くだけである。前の格子（真ん中が 0.0542・0.182・0.271）は、別のトークン化の曲線（$D$ = 8192）を代わりに使って導いたもので、この格子で置き換えた。

posthoc には、情報を持たない入力の変化に対する表現の感度を足した（`measure/sensitivity.py`）。`sens_dither` は量子化のディザ（画素ごとに $U(-\frac12, \frac12)/255$）、`sens_shift` は 1 画素の平行移動（端の列は複製）に対する感度で、どちらも $E\lVert f(x') - f(x) \rVert^2 / E_{i \ne j}\lVert f(x_i) - f(x_j) \rVert^2$ である。分母は `diag/stream_stats` の `drift` と同じ。`_pix` は同じ比を画素で測った値である。表現の値がこれを上回るなら、入力ではほとんど変わらない差を、表現が増幅している（判定の規則は results/m17.md の「ストリームの SIGReg（回す前に決めたこと）」）。最後の checkpoint の表現の、ラグごとの白色化した相関は、`diag/stream_stats` の `repr` で測る。

```bash
# λ は preset の格子の真ん中（両端は vision.sigreg で上書きする）
# BP、seed 100、k = 1（k = 6・24 は vision.datapoint_every=6 / 24 で上書きする）
for arm in t512 t128 t64; do
  just fleet submit --batch m17-ssig-v1 +experiment=m17/ssig_$arm learner=bp --seeds 100 --res 4090 --gpu-mem 20
done
# 入力を貯める腕（BP だけ）は vision.datapoint_store=input
# 速さ（random な積み、バッファは満杯）: python -m exp.video.diag.ssig bench --arm t128 --device cpu
# 各腕の学習前の相関（格子の導出。EMA の run の init、t128・t64 も同じ形）と窓
python -m exp.video.diag.stream_stats repr --root DATA --runs /nas/geson/mujou-exp/runs \
    --model t512=m17-ema-v1:m17-ema_t512-35d806f3-s100:init:tokens --out repr-t512.json
python -m exp.video.diag.ssig_window --own --curves t512/tokens/white t128/tokens/white t64/tokens/white
# 最後の checkpoint の表現の相関（NAS の run から）
python -m exp.video.diag.stream_stats repr --root DATA --runs /nas/geson/mujou-exp/runs \
    --model ssig=m17-ssig-v1:<RUN_ID>:01800.0:tokens --spectrum tokens --out repr-ssig.json
```

補完の誤差（`measure/completion.py`）には、自明な予測の物差しとして `comp_<kind>_pos`（と `ncomp_<kind>_pos`）もある。学ぶ 9 本の held-in の積みのターゲットのトークンを、位置（time slot と空間の位置）ごと・次元ごとに平均し、それで当てる。文脈を見ずに位置だけを知る予測である。どの積みでも同じ位置を隠す `future` では、参照の集合で二乗誤差を最小にするので、参照の上では `_mean` 以下になる。積みごとにマスクを引く `short`・`long` では、この大小は保証されない。読み方は 2 つある。`comp_<kind>` が `comp_<kind>_pos` を下回れば、予測器は位置より多くを文脈から得ている。トークンが位置だけの関数 $g(p)$ に近い解（P）では、`comp_<kind>_pos` が 0 に近くなり、`ncomp_<kind>_pos` も 0 に近くなる。`comp_<kind>` と `comp_<kind>_pos` が同じ程度だというだけでは、予測器が文脈から位置以上のものを得ていないことしか言えない。両方が 1 に近ければ、トークンは互いに独立か位置による構造を持たない解で、P ではない。

予測器を通した復元（`measure/encoder_eval.py`）には、上限として `restore_target_<view>_mask` も載せる。隠さないトークン（エンコーダに積み全体を通したもの）を同じトークンのデコーダで描き戻し、`restore_<view>_mask` と同じ隠した画素で測る。復元の鍵はすべて、誤差をぼかしてから測った `_blur` 版も持つ（σ = 2 px）。隠した画素の鍵では、ぼかした誤差の画像を隠した画素で平均し、平均の画像の同じ値で割る（`exp/lib/blur_error.py` の `blur_recon_masked`）。復元の画像（`<name>-restore.png`）は val の 10 窓で、行は上から順に、真の最後のフレーム、隠さないトークンから描き戻した上限、各 context view の隠した入力とその予測器を通した復元。

```bash
# 物差しだけ（親の run なし）
just fleet submit --batch m17-base-v1 +experiment=m17/wt_n16 task=baselines --res 4090
# run の後に、その run の posthoc（親は NAS の run のディレクトリ）
just fleet submit --batch m17-post-v1 +experiment=m17/wt_n16 task=posthoc --res 4090 \
    --after <RUN_ID> --set parent=/nas/geson/mujou-exp/runs/<batch>/<RUN_ID> --set decode.ckpts=all
# TensorBoard のビュー（tools/tb/README.md）
just fleet pull --batch <batch> --batch m17-post-v1
just tb export --batch <batch> m17-post-v1 && just tb view
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
| M16 | [視覚と動力学を同時に学ぶ](results/m16.md) | 達成。動力学を 0.6 秒ごとの窓の BPTT（W=36、feedback=grad、PC-ALM は窓のエピソード）で学び、test の後から学ぶデコーダで 10 フレーム先が平均の画像の 0.66〜0.79 倍（PC-ALM と BP、3 seed）。毎フレームの学習（最初の確認、BP）では 1.3〜1.5 倍で届かなかった。PC-ALM を 5 更新/フレームに縮めると学習が安定せず、採らなかった |
| M17 | [Walking Tours でエンコーダを詰める](results/m17.md) | 実施中。物差し・判定基準・計算時間・試し・診断まで。preset の JEPA（BP）は $N$ = 16・8・1 とも学ぶほど学習前の ViT より悪くなり（val の `recon` 0.63 → 0.75〜0.81、PCA-256 は 0.15）、9 つの変更でも学習前を下回らず、study の前で止めている |
