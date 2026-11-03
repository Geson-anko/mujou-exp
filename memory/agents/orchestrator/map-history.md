# 研究の経緯と教訓の地図（M14〜M16 の再設計のため、2026-11-03）

読んだもの: `exp/video/{MILESTONES,README,PLAN}.md`、`exp/video/results/m10〜m13.md`、`memory/*.md`、`memory/agents/**`、確認のため `exp/video/{config,easy,run,decode,offline,balls,evaluate}.py` の一部、`.claude/skills/experiment-report/SKILL.md`、`gh pr list`。コードは変更していない。

注意: このチェックアウトでは `mujou/` submodule が未初期化（`git submodule status` が `-881c4ee...`）。`runs/` もローカルに無い（run は他のマシンにある）。

---

## 1. 研究の流れ（1 行ずつ）

| M | 結論 | PR / ブランチ |
| -- | ---- | ------------- |
| M0〜M5 | 足場・optimizer・段 0・合成動画・ノード（SIGReg/Compare/Dynamics）・評価と BP 版 | #3〜#8 `exp/2026-10-30/continuous-stream-m{0..5}-*` |
| M6 | 段 0: H0 棄却。grafting を使う、`learn_interval` 25 ms（PC-ALM と BP が 3 桁一致、`None` より 1〜5% 良い） | #9 |
| M7 | 段 1: 前提（ランク 32 以上）未達。LSUV 初期化 `model.standardize_` | #10 |
| M8〜M10 | 表現崩壊 → `stop_target`・κ=0.8・SGD 0.01。白色化・射影・Muon・Adam 系どれも SGD を超えず（Adam 系は 9 本すべて崩壊/発散）。relaxer は `pc.relax.Euler` のまま | #11 `exp/2026-10-30/continuous-stream-m8-stage2` |
| M11 | `z_norm`（z を半径 √D の球面へ）で |z| 発散が止まる。開ループ（`open_loop=0.5, open_cycle=20`）。潜在 `rollout10` < 1 は動力学の証拠にならないと判明 → 条件を画像空間へ | #18 `exp/2026-11-01/stage3-egodex` |
| M12 | EgoDex（3+3 ノード、残差の予測）`rollout10` 0.37〜0.50、写し 0.10〜0.14 に届かず。保持なし | #18 |
| M13 | 簡単な合成動画。未達のまま中断。玉 1 つのオンライン最良は適格度トレースで 5/10 フレーム先 0.49/0.92 倍（平均の画像比） | #19 `exp/2026-11-02/continuous-stream-m14-dynamics`（M13 の結果と M14〜M16 の計画を同じ PR で） |
| M14 | 計画: 動力学だけ（観測の上の誤差） | 未着手 |
| M15 | 計画: 画像・動画の JEPA（Walking Tours） | 未着手 |
| M16 | 計画: 同時の学習（動力学はエンコーダに credit を送らない） | 未着手 |

方針の要（`memory/project_dynamics_only.md`）: 潜在空間で予測誤差をとって同時に学ぶと、潜在空間が予測しやすく歪む（当てにくい玉を捨てる、折り畳む）。表現の問題と、動力学をオンラインで学ぶ問題（時間の局所性、optimizer）を分けて確かめる。**M16 で動力学からエンコーダへ credit を戻す設計を提案しない。**

---

## 2. 最良の既知の設定（数字は正確に）

### 2.1 共通の時間設定（`exp/video/config.py:35` `TimeConfig`、全実験で不変）

- `frame = 1/30`、`TimeConstants(h=4e-3, lam=2e-3, r=4e-3)`（周期 2 ms、1 フレーム約 16.7 周期）、`learn_interval = 25e-3`
- 実行は `mujou.testing.run` で手で `step`（仮想時間、決定的）。実時間 `mujou.run` は多ノード・GPU では追いつかない（M12: 9 ノードで 47% のフレームが間に合わず）

### 2.2 学習率の表し方（gotcha）

- `OptimConfig.tau` で表す。SGD の学習率 LR は `tau = FRAME / LR`（`easy.py:69`: `OptimConfig(kind="sgd", tau=FRAME / LR, warmup=WARMUP)`）。LR=0.03 → tau≈1.111 秒
- `OptimConfig.warmup`（秒、`exp.lib.optim.Warmup`）: 300 秒。B=128 で LR ≥0.02 のとき最初の 1 分の崩壊を止めた（M11）
- `Config.dynamics_optim`（M13）: 動力学と読み出しだけ別の optimizer

### 2.3 M13 のモデルの「小」と「大」（`exp/video/easy.py:42-45`）

```python
SMALL = ModelConfig(kappa=0.13, z_norm=True, open_loop=0.5, open_cycle=20, buffer=1023)
BIG = replace(SMALL, width=64, d_s=512, dyn_nodes=3)
LR, WARMUP, CHECKPOINT = 0.03, 300.0, 300.0
STREAMS = 128
```

| | 小（SMALL） | 大（BIG） |
| - | ----------- | --------- |
| `d_z`（D） | 64（既定） | 64 |
| `width`（エンコーダの最初の conv、w/2w/4w） | 32 | 64 |
| `d_s` | 256 | 512 |
| `dyn_nodes`（`Dynamics` + `dyn_nodes-1` 個の `DynamicsLayer`） | 1 | 3 |
| `enc_nodes` | 2（既定） | 2 |
| `kappa` | 0.13 | 0.13（玉 1 つの探索では 0.0154 / 0.0078 に変えた） |
| `buffer` / `buffer_every` / `directions` | 1023 / 1 / 256 | 同じ |
| `z_norm`, `open_loop`, `open_cycle` | True, 0.5, 20 | 同じ |
| `stop_target` / `target_weight` | False / 0（= w=1） | 同じ |

- `DynamicsLayer` は $y = x + \mathrm{minGRU}(\mathrm{GELU}(x), h_{k-1})$（残差の minGRU）。評価と BP では `DynamicsStack` にまとめる
- M14 は「$d_s$ と動力学のノードの数は M13 の大と小」= 小 (d_s 256, 1 ノード) / 大 (d_s 512, 3 ノード)。エンコーダが無いので width は関係ない
- 「モデルの大きさは効く」: 小は z が情報を保っても 10 フレーム先を当てられない（オフライン、画像 10 で 1.05 倍）

### 2.4 κ と B の換算（gotcha）

`DataConfig.streams = B > 1` では SIGReg の項が B 本の和、予測誤差は平均で届く。サンプルあたりの SIGReg の重みは $\kappa B / (1-\kappa)$（`config.py` の `DataConfig.streams` の docstring）。

- B=128: κ=0.13 → 約 19、κ=0.0154 → 約 2.0（オフラインの λ=2）、κ=0.0078 → 約 1.0
- B=256: κ=0.0078 → 約 2.0
- オフラインの `--sigreg`（λ）はこのサンプルあたりの重み

### 2.5 データ `easy.EASY`（`easy.py:29`）

```python
EASY = DataConfig(kind="video", seconds=1800.0, gamma_amp=0.0, e0=1.0, e_amp=0.0,
                  drag=0.0, conserve=True, redraw=False, streams=128)
```

- 32×32 グレースケール、玉 3 つ `RADII = (0.06, 0.08, 0.10)`（`balls.py:32`）、`gamma0=1`、8 小刻みの半陰的 Euler、`physics_period=600`（EASY では一定なので無関係）
- 玉 1 つ: `radii=(0.10,)`。玉 2 つ: `(0.08, 0.10)`
- `conserve=True` が必要: e=1 でも位置の反射が $\tilde E$ を $\gamma(2\delta - h|v_y|)$ ずらし、共鳴で 30 分に玉ごと最大 10% 溜まる
- 玉の速さ: 平均 0.026/フレーム（約 0.8 画素）。速度の尺度は約 0.03/フレーム、位置は [0,1]（M14 で標準化が要る理由）
- `spread`（ストリーム i を i·T/B から始める）は物理が周期で変わるときだけ意味がある。EASY では物理が一定
- `Frame.state` は `(…, 4 * balls)`（位置と速度）。`Frame.params` は `(…, 2)`（γ, e）

### 2.6 M13 のオンラインの最良（玉 1 つ、BP、30 分、seed 0）

設定: 大（width 64、d_s 512、dyn_nodes 3、D=64）、B=128、SGD LR 0.03（tau=1/30/0.03）、warmup 300 秒、checkpoint 300 秒、`z_norm`、開ループ (0.5, 20)、buffer 1023、w=1、**κ=0.0154、再構成 10（`recon=10`）、`trace=True`**、`radii=(0.10,)`、`EASY` の物理。

| 設定 | 画像 5 | 画像 10 |
| ---- | ------ | ------- |
| SGD 0.03（トレースなし） | 0.69 | 1.17 |
| 残差の予測、SGD 0.03 | 0.69 | 0.89 |
| **トレース、SGD 0.03** | **0.49** | 0.92 |
| トレース + 残差 | 0.68 | 0.95（20 分の時点で 4 クリップ中 3 つが 20 フレーム先まで追う） |
| トレース、B=256（κ=0.0078） | 0.50 | 0.92 |
| トレース、SGD 0.05 | 0.62 | 1.28 |
| トレース、AdamW 1e-4（全体） | 0.67 | 1.00（300 秒で潜在 0.44 と速いがランク 7→17 で悪化） |
| トレース、AdamW 3e-4 | 1.29 | 1.47 |
| トレース、動力学だけ AdamW 1e-4 | 0.67 | 1.04 |
| トレース、動力学だけ SGD 0.1 / 0.3 | 0.81 / 0.72 | 1.23 / 1.38（不安定、20 先 2.6 倍） |
| トレース、`adam_direction`+`graft`（全体） | 0.61 | 0.89 |
| 開ループ 0.25 / 0.75（残差、SGD 0.03） | 0.65 / 0.85 | 1.01 / 1.08 |

（値は予測をデコードした画素誤差 / 平均の画像 `pix_mean{n}`）

- PC-ALM のトレース版は 30 分で 5〜7 時間超の見込みで、**一度も回していない**。M13 のオンライン探索はすべて BP（`WorldBP`）

### 2.7 M13 のオフラインの上限

共通: 128 窓（`WINDOW=32`, `OBSERVED=16`、`offline.py:49`）、Adam 1e-3、`--local`（ノードと同じく文脈の $z_{k-1}$ と戻す予測で勾配を止める）、131,072 フレーム、値は最後のチェックポイントをデコード。

玉 3 つ:
- 最良 10 フレーム先: 大、λ=1、再構成 100、20000 ステップ → 再構成 0.0009、画像 5 / 10 = **0.59 / 0.59**（目安 0.5 未達）。小さい玉は予測の 1〜2 フレームで消える
- 大、λ=5、w=1、6000 → 0.60 / 0.63 だが小さい玉 2 つを捨てる（再構成 0.0077）
- 小、λ=15、w=1 → 0.83 / 1.05
- 画素の予測誤差を足す（λ=2、再構成 100、画素 1000）→ 0.56 / 0.81
- 玉 3 つを同じ大きさにしても良くならない（0.72 / 1.00）→ 複数の玉そのものの難しさ

玉 2 つ（0.08, 0.10）: 10000 ステップ 0.44 / 0.50、20000 で 0.43 / 0.46（小さい玉が再構成でもぼける）

玉 1 つ（0.10）、大、λ=2、再構成 100、6000:
- **0.24 / 0.43、再構成 0.0000、見た目も合う（床での跳ね返りまで）**
- $s$ の勾配を 1 フレームで切る（ノードと同じ）と 0.39 / 0.77 → **10 フレーム先の律速は時間の局所性** → 適格度トレースの動機
- SGD 0.1（勾配を切らない）では潜在 0.37 対 Adam 0.17 → SGD の標本効率も律速

### 2.8 オンライン（玉 3 つ、大、B=128、30 分、seed 0、BP）

| 設定 | 再構成 | 画像 5 | 画像 10 | 潜在 rollout10/shrink10 | ランク |
| ---- | ------ | ------ | ------- | ----------------------- | ------ |
| M11 の設定（κ=0.13、w=1） | 0.0015 | 1.00 | 1.10 | 1.00 | 41 |
| κ=0.0078、再構成 10 | 0.0009 | 1.02 | 1.16 | 0.73 | 4.2 |
| κ=0.0078、再構成 30 | 0.0006 | 1.06 | 1.17 | 0.77 | 4.4 |
| κ=0.0078、再構成 99 | 110 秒で NaN（SGD では重みがデコーダの歩幅になる） | | | | |

### 2.9 M11 の到達点（B=128、`spread`、参考）

κ=0.13、LR 0.03、warmup 300 秒、B=128、`spread`、w=1、`z_norm`、開ループ (0.5, 20)、buffer 1023 で、PC-ALM・BP 3 seed とも `rollout10` 0.94〜0.99、ランク 33.8〜35.9。**しかしデコードすると玉が止まって薄れるだけ（`amp10` 0.38）で、動力学ではなかった**。潜在の `rollout10` と画像の 10 先の誤差の相関は −0.71。

### 2.10 M12 EgoDex（参考、M15/M16 の実動画の比較材料）

- 3+3 ノード（エンコーダ 3 段 → D=128、`Dynamics` d_s 512 + `DynamicsLayer`×2）、`z_norm`、開ループ (0.5,20)、SGD 0.03、κ=0.8、`stop_target`、buffer 1023・`buffer_every=8`、`residual`
- `rollout10` PC-ALM 0.43/0.40/0.37、BP 0.48/0.39/0.50、`persist10` 0.10〜0.14、`mean10` 0.93
- `z_norm` なしは 2〜4 秒で発散（似たフレームが続くと SIGReg のバッファの自分のコピーから斥力）
- 実動画は隣のフレームがほぼ同じ → 予測は残差（`residual`、$W_o$ はゼロから）が自然、SIGReg のバッファは数分に広げる

---

## 3. 評価の手順と閾値

### 3.1 評価クリップ（`balls.eval_clips`、`balls.py:283`）

- 既定: `phases=8`、`clips=4`、`length=128`、`seed=12345` → 32 クリップ × 128 フレーム、`(phases, clips, length, ...)` で積む。各クリップは新しい玉から始める。run の物理（`DataConfig.physics()`）と `radii` で作る（`run.py:135`）
- rollout: 16 フレームを観測（$s=0$ から）、その後は開ループで自分の予測を入力に戻す。`HORIZONS = (1, 5, 10, 20)`（`evaluate.py:29`）
- `halves(phases, clips)`: クリップを半分に分けて probe・`shrink` の係数を互いに当てる
- 評価は `eval_interval`（既定 60 秒）ごとにモジュールを複製し素の PyTorch で順伝播。学習は止めない

### 3.2 物差し（動力学を知らない基準と必ず並べる）

- 潜在: `persist{n}`（最後の z を写す）、`mean{n}`（z の平均）、`shrink{n}`（$m + \lambda_n(z_\text{last}-m)$、平均へ縮めた写し）、`amp{n}`・`corr{n}`
- 画像（`decode.py`）: `img_rollout{n}`、`img_persist{n}`、`img_mean{n}`、`img_shrink{n}`、デコーダなしの `pix_copy{n}`・`pix_mean{n}`、`recon`
- **`rollout10` は「クリップ間の同時刻の分散」で割るので、定数でも約 1.1。「1 未満」を条件にしない**

M13 データ（玉 3 つ）の画素の物差し（1 画素あたりの二乗誤差）:

| n | `pix_copy` | `pix_mean` | 0.5 倍の目安 |
| - | ---------- | ---------- | ------------ |
| 1 | 0.00312 | 0.01462 | |
| 5 | 0.0221 | 0.0144 | 0.0072 |
| 10 | 0.0289 | 0.0149 | 0.0074 |
| 20 | 0.0292 | 0.0145 | |

### 3.3 M13 の完了の条件（当時）

30 分、PC-ALM と BP、seed 0/1/2 で: 発散しない、再構成 ≤ 0.001、画像の予測誤差が 5・10 フレーム先で `pix_mean` の 0.5 倍未満、潜在で `rollout{n}` < `shrink{n}`、見た目で玉が正しく動き跳ね返る。

### 3.4 M14 の物差しと条件（MILESTONES.md の計画）

- 観測は評価クリップの平均と標準偏差で各次元を標準化。誤差は観測の空間での 1 次元あたりの二乗誤差
- 物差し: 平均（評価クリップの平均の観測）、写し（最後の観測）、**等速の外挿**（最後の 2 フレームの位置差を速度に、重力も壁も無しで進める）、真の物理（誤差 0、上限の確認だけ）
- 段 0: 玉 1 つの $(x,y,v_x,v_y)$（マルコフ）、段 1: 玉 1 つの $(x,y)$、段 2: 玉 3 つの位置、段 3: 画像 32×32（誤差は画素、潜在の誤差は使わない。段 1・2 が通ってから決める）
- 完了の条件（段 1・2）: 30 分、PC-ALM と BP、seed 0/1/2 すべてで、発散しない、5・10 フレーム先が平均・写し・等速の外挿のどれよりも十分に小さい（**目安: 等速の外挿の 0.5 倍未満**）、20 フレーム先も記録、予測した軌跡を見て正しい向きと壁での跳ね返り、局所性・非同期性を崩さない。**閾値は手順 1 で物差しを測ってから確定し、回す前に README に書く**
- 手順: 1 物差し → 2 オフラインの上限（窓、Adam、BPTT。M13 の `offline.py` と同じ窓の作り方）→ 3 オンライン 段 0→1→2、30 分、BP で探索してから PC-ALM と BP の seed 0/1/2 → 4 段 3
- オラクル: 観測の逆標準化 = `Frame.state`、等速の外挿は重力 0・壁なしで誤差 0、エンコーダなしの平衡点で `lam` が BP の credit と一致（M5 と同じ）、トレースの勾配 = 入力固定の BPTT
- 適格度トレースはあり・なしを比べる。オフラインとオンラインの差、トレースと optimizer の効果も README に書く

### 3.5 M15 / M16 の判定（計画）

- M15: 各チェックポイントに学んだデコーダでの再構成の見た目、合成動画なら玉の状態への線形 probe、実効ランク。詳細は着手時に決める
- M16: 動力学が予測した z をデコーダで画像に戻し、5・10・20 フレーム先を真のフレームと比べる（`pix_mean`、写し）。見た目で玉が正しく動き跳ね返るか

---

## 4. 定性確認の方針（必須）

- `memory/feedback_metrics_need_qualitative_check.md`: 数値指標は、デコードした画像の見た目と合うことを確かめてから条件に使う。きっかけは M11 で潜在 `rollout10` < 1 を達成と報告し、実は平均へ縮むだけだったこと
- 動力学を知らない基準（写し・平均・shrink・平均の画像、M14 では等速の外挿）と並べる。絶対の閾値だけで達成と言わない
- `memory/project_decoder_for_debug.md`: デコーダはデバッグの道具。**各チェックポイントに後から BP で学ぶ（エンコーダは止めたまま、Adam 1e-3、batch 256、2000 ステップ、ストリームの 16384 フレーム、1 ckpt 約 9 秒）**。システムにも mujou の実行にも入れない。M13 の `nodes.Reconstruct` / `ModelConfig.recon` は M15・M16 の学習に入れない
- `decode.py` の出力: `runs/<run>/decode/<秒>.png`（行: フレーム、z の再構成、rollout、shrink。列は `COLUMNS = (1, 2, 5, 10, 20)` フレーム後、4 クリップ）と `decode/metrics.jsonl`、`decode/config.json`
- M14 は観測が状態なのでデコーダ不要: 予測した状態を `balls.render` で描き、真のフレームと並べる。予測軌跡と真の軌跡を重ねた図も描く

---

## 5. 記録と報告の規約

- 実行前に判定基準を README（または results/mXX.md の「回す前に確定した判定基準」）に書き、実行後に数字を書く
- `exp/video/README.md` の「結果」表に 1 行（`| Mxx | [題](results/mxx.md) | 結論 |`）。数字の正は `results/mxx.md`
- `results/mxx.md` の冒頭: `# 題（Mxx、日付）` と「[README](../README.md) の結果の 1 節。ファイル名とパスは当時のもの。」。表は seed 0/1/2 を ` / ` 区切りで 1 セルに。崩壊・発散も残す（「崩壊」「発散」と書く）。「M?? の判定」節で結論
- `MILESTONES.md`: 各マイルストーンに **状態**（計画 / 未達のまま中断 など、日付つき）。終わったら「終わったマイルストーン」表の 1 行に畳む
- `PLAN.md` 12 節「決めたこと」に日付つきで決定を足す
- report.html（skill `experiment-report`）: `exp/<name>/report.html`、1 ファイル完結（JSON を `<script type="application/json" id="data">` に埋め込み、素の JS + SVG、外部は Google Fonts だけ）、数 MB 以下（手本約 300 KB）、構成は ヘッダー / 分かったこと / 実験の仕組み / 入力 / 進め方 / 結果の節 / 仮説の判定 / 次に持ち込むもの / 付録: 再現。生成スクリプトと抜き出した JSON は commit しない（scratchpad）。**日本語の本文は必ず skill `yomiyasu` で推敲**。既存: `report.html`（M6〜M10）、`report-m11-m12.html`
- 再現性: `runs/<日付>-<名前>/` に `config.json`（設定、mujou とこのリポジトリの SHA、未コミット変更の有無）と `metrics.jsonl`。runs・checkpoint は git に入れない。seed は 0/1/2、判定は 3 seed すべてで同じ向きのときだけ支持
- memory: 知見は `memory/project_mXX_findings.md` 形式で 1 ファイル、`MEMORY.md` から 1 行リンク。中間メモは `memory/agents/<agent>/<task>.md`
- 文書は日本語、docstring・コメントは英語

## 6. ブランチと PR の規約

- 1 マイルストーン 1 PR が原則。ブランチ `exp/<日付>/continuous-stream-<マイルストーン>`（例 `exp/2026-11-02/continuous-stream-m14-dynamics`、M13 は `...-m13-easy-video` の予定だったが、実際は PR #19 の m14-dynamics ブランチに M13 の結果と M14〜M16 の計画が入った）。実際には複数マイルストーンを 1 PR にした例（#11 M8〜M10、#18 M11〜M12）もある
- PR タイトル `exp(video): ...`。commit は `feat(video)`・`feat(lib)`・`docs(video)`・`fix(lib)`・`docs(memory)`・`chore(...)`。1 commit 1 関心事、`just run` が通ってから
- main に直接 commit しない、main への merge はユーザー判断。PR 本文末尾に attribution
- mujou を変えるときは mujou 側でブランチを切り、`Geson-anko/mujou` の main 宛に PR。未 push の mujou コミットを指すポインタはコミットしない（`just mujou-status`）。mujou の SHA は 881c4ee

## 7. 実行環境と運用上の落とし穴

- PLAN 11 節の環境: RTX 4090（24 GB）と RTX 4060（8 GB）、32 コア（ユーザーは今回 4 台の GPU マシンを使うと言っている）
- `torch.set_num_threads(1)`。複数プロセスでフレームを作るときは **`OMP_NUM_THREADS=1`**（OpenMP の取り合いで数十分止まる）。テストも OpenMP を 1 スレッドに固定（PR #17）
- `pkill -f <パターン>` は自分のシェルにも当たるので PID で止める
- 手で回す実行は CPU 1 コアが律速、GPU は空く。GPU に複数本を並べてよい（4090 に 2 本・4060 に 1 本の割合、M12）。4060 は 8 GB で評価を分割
- 時間の見積もり: オフライン 6000 ステップは GPU に 3〜4 本並べて 30〜50 分。M12 の 3+3 ノードは 4090 で実時間の約 6 倍（30 分で 3〜3.5 時間）。M13 の PC-ALM + トレースは 30 分で 5〜7 時間超
- `easy.py` はデバイスを `cuda:0` / `cuda:1` に交互に割り当てている（1 台 2 GPU 前提のハードコード）
- 既存の掃引は `exp/lib/sweep.py` `run_all(module, configs, out, jobs)`（サブプロセスで `-m <module> --config --out`、`done` で判定、失敗しても終了コード 0）。設定は frozen dataclass + JSON（`exp.lib.config.from_dict`、未知のキーは TypeError）
- 既知の非決定性: `tests/video/test_model.py::test_evaluating_does_not_change_the_learning` が全体実行で稀に落ちた（free-threaded・並列の疑い、未調査）

---

## 8. 持ち越す教訓（設計判断に効くもの）

1. Adam 系は SIGReg と組むと崩壊・発散（M10 の 9 本、M11 の 8 本）。M13 でもオンラインの AdamW はランクが上がって悪化。**ただし M14 にはエンコーダも SIGReg も無いので、Adam の害の原因（表現の崩れ）が消える可能性がある → optimizer の比較をやり直す価値がある**
2. オフラインでも SGD は Adam より遅い（潜在 0.37 対 0.17）。オンラインの律速は「時間の局所性」→ トレースで部分的に解消 →「SGD の標本効率」
3. 適格度トレース（`ModelConfig.trace`、`exp.lib.mingru.MinGRUTrace`、RTRL、重みと同じ大きさ × サンプル数のメモリ）は minGRU の再帰が対角なので局所性を保ったまま BPTT と同じ勾配。PC-ALM では遅い（5〜7 時間 / 30 分）
4. 残差の予測は実動画で必須（M12）。合成動画ではトレースと組むと 5 フレーム先が悪化（0.49 → 0.68）
5. 開ループ (0.5, 20) は M11 で `rollout10` を下げた。玉 1 つでは 0.25/0.75 にしても良くならない
6. B 本のストリーム（B=128）は `rollout1` を大きく下げる（標本効率）。B=256 は B=128 と同じ
7. SGD の学習率: 0.03 が最良、0.05 以上は悪化。動力学だけ 0.1・0.3 は不安定
8. 再構成の重みは SGD ではデコーダの歩幅（99 で NaN）
9. 実動画の隣接フレームはほぼ同じ → `persist` が非常に強い基準（EgoDex `persist10` 0.10〜0.14）
10. PC-ALM と BP は、測ったどの設定でもほぼ同じ値（M6・M11・M12）。探索は BP で、確認は両方

---

## 9. M14〜M16 に関わる未決事項

M14:
- 閾値の確定（等速の外挿の 0.5 倍が目安。手順 1 の物差しを見て README に書く）
- 置き場所と名前（「実装のときに決める」。`exp/video/` の中、`balls.py`・`nodes.py`・`bp.py`・ホストのループを使い回す）
- エンコーダなしの `Network` と `WorldBP` の組み方。`Input` → `Dynamics` → `Compare`（読み出し $W_o$ で次の観測を予測）。目標への credit が無いので `stop_target` 等は不要
- 観測の空間で残差の予測（$\hat o_k = o_{k-1} + W_o s_k$）を使うか
- optimizer: SIGReg が無いので Adam / AdamW / grafting を改めて試すか
- 小・大のどちらで探索するか、トレースのあり・なし、B
- 段 3（画像）の組み方と条件（段 1・2 が通ってから。誤差は画素の上）
- PC-ALM + トレースの実行時間（5〜7 時間 / 30 分）をどう捌くか（並列・高速化）
- 通った後、物理の難しさ（重力の時間変化、反発係数、抵抗）を戻すか（M16 のとき決める）

M15:
- マスクと予測器（I-JEPA / V-JEPA のブロックのマスク）、ターゲットのエンコーダ（EMA か、LeJEPA のように SIGReg だけで EMA と stop-gradient を外すか）
- ストリームへの載せ方: 1 フレームずつ届くストリームで N フレームのまとまりをどう作るか、ノードの分け方（エンコーダ、予測器、SIGReg）と局所性。ViT を PC-ALM のノードにどう分けるか（M12 の credit の回り切りの制約: 3+3 で Jacobi 20 ms、`learn_interval` 25 ms）
- データ: 合成動画（玉）から始めるか、最初から Walking Tours か。解像度とフレームの間引き（4K・60 fps から）。Walking Tours は `WALKING_TOURS_DIR`（`.env`）→ dev container の `/datasets/walking_tours`（書き込み可、無ければ `datasets/walking_tours`）。YouTube から取得（`WTour.txt`）、動画ごとのライセンスと顔・ナンバーのぼかしを確認、レポートに出典を書く。物を扱う動きが要るなら EPIC-KITCHENS-100（CC BY-NC 4.0、非商用なので可）
- 判定の基準（デコーダの再構成の見た目、合成なら線形 probe、実効ランク）の数値の閾値
- SIGReg の |z| 発散（`z_norm` の要否）とバッファの長さ（実動画では数分）

M16:
- 動力学は M14 の設定、エンコーダは M15 の設定から。credit はエンコーダに送らない（`target_weight=0` 相当、`Dynamics` は $z_{k-1}$ へ `g_in` を返さない）
- 実動画での判定の物差し（写しが非常に強い）
- 物理の難しさを戻すか
- M14 の観測（状態）から潜在 z への移行で、標準化や残差がどう効くか
