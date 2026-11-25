# 潜在空間の disentanglement の評価に向けた、手元の資産の棚卸し（2026-11-25）

役割: 読むだけの棚卸し。コード・README・plans には触れていない。GPU は使っていない。NAS は `ssh std-00` で `ls` しただけ。

読んだもの: `exp/vision/eval-guide.html`、`exp/vision/README.md` の V16〜V18c の各節、`exp/vision/plans/jepa.md` の見出しと冒頭、`memory/project_jepa_plan.md`・`memory/feedback_eval_matches_objective.md`・`memory/project_m17_wt_encoder.md`、`memory/agents/fable/`・`memory/agents/codex/` の見出しと要点、`exp/vision/*.py` と `exp/lib/*.py` の docstring と定義の一覧、`exp/lib/tours/manifest.json`。

確かさは 3 段で書く。**確かめた**（ファイルか NAS で見た）、**コードから**（docstring・定義を読んで判断した。実行はしていない）、**推測**。

## 0. 要点

1. **入力の形は、VAE（V16・V17）と JEPA（V18・V18b・V18c）でまったく同じ。** どれも 8 フレーム（0.8 秒、10 fps）× 3 × 64 × 64 の stack を early fusion（24 チャネル）で ViT-S（幅 1024、4 ブロック、μP）に入れる。同じ config（V14 の `b2006f0a-s0`）から作り、ブロックの初期値も同じ（確かめた）。違うのは**パッチの大きさ**（16・4・8）と、**潜在の取り出し方**（大域か、トークンか）だけである
2. **潜在の形はモデルごとに大きく違う。** VAE の $\mu$ は大域の 256 次元。JEPA の符号はトークンの列で、V18 の P16 は 16 × 64（学習した $P$）、V18b の F は 256 × 4、M は 256 × 64、V18c は 64 × 16（固定の $R$）。SIGReg をかける $z$ は、どれも 1024 次元（M だけ 64 次元のトークン平均）。比べるときは「どこで読むか」をそろえる必要がある（1.3 節）
3. **VAE の頭の前にも、JEPA の P16 と同じ形のトークン $h$（16 × 1024、最後の LayerNorm の後）がある**（コードから）。VAE の $\mu$ は $\mathrm{vec}(h)$ の線形写像である。`jepa.TokenEncoder` は `build_vae` の ViT から頭を外したものなので、VAE の checkpoint からも同じ関数で $h$ を取り出せる見込み（未実行）。トークンの水準で VAE と JEPA を同じ形で比べられる
4. **checkpoint はすべて NAS にそろっている**（確かめた）。V16 の 6 本、V17 の本番 3 本と追加の調査、V18 の掃引 18 本と長い run 4 本、V18b の掃引と長い run 4 本、V18c の掃引 15 本と長い run 12 本（12 本とも 9 個の checkpoint がそろい、学習と既存の評価のジョブは done）
5. **画素で作れる介入の材料は多い。** 明るさ・白の偏り・ぼかし（`sem_eval`）、調和補間と移植（`exp/lib/regions.py`）に加えて、**本物の平行移動**に使える左右の帯のデータ（`wide`、64 × 112、Amsterdam・Bangkok・Zurich の全フレーム）、**本物の拡大**に使える 320 px のフレーム（`hires`、選んだフレームだけ）と 128 px・30 fps の全フレーム（M17・M18 のデータ）、**時間**（stack の間隔 0.1 秒から）がある
6. **評価の部品は「符号 = stack からベクトルへの関数」で統一されている**（`stream_eval.Code` の `modules()` と `mu(x)`）。新しいモデル（β-VAE・MAE・VICReg・公開モデル）も、この 2 つを書けば既存の評価（線形の動力学、実効ランク、`token_r2`、同定、ブートストラップ）にそのまま載る（コードから）
7. **他の自己教師あり学習**は、同じデータ・同じ ViT-S・同じ BP の i.i.d. 学習（`jepa_iid.py` の形）で作るのが最も公平で、手間は MAE が最小（マスク・`TokenEncoder` の `keep`・予測器が既にある）、VICReg・SimCLR が中、DINO が最大。公開の学習済みモデルは学習の手間が無いが、データ・解像度・入力の形（1 枚か 8 枚か）・大きさが違い、とくに **拡張（色の揺らしなど）で人が決めた不変性が、画素の介入の試験の答えを先に決めてしまう**（4 節）

## 1. 比べられるモデル

### 1.1 一覧（checkpoint の在りかは `/nas/geson/mujou-exp/runs/` からの相対）

| 名前 | 何か | 学び方 | コード | checkpoint（確かめた） | 見た stack |
| -- | -- | -- | -- | -- | -- |
| `v16` | VAE（$\mu$） | BP・AdamW、i.i.d. な batch 100、60000 step、ELBO（学習する $\ln\sigma$） | `exp/vision/iid.py`、`models.build_vae` | `v16-iid/iid-all-lr3e-4-s0/ckpt/600000.0.pt`（物差しに使ったもの）。ほかに `iid-all-lr{1e-4,1e-3}`、`iid-span-lr{1e-4,3e-4,1e-3}`（1800 秒分だけで学んだ、覚えた VAE）。各 run の ckpt は $t$ = 0・60・300・600・1200・1800・6000・18000・60000・180000・600000 秒（$t$ = 見た stack × 0.1） | 600 万 |
| `v17` | VAE（$\mu$） | 単一ストリームの PC-ALM（連続する 100 stack を 1 batch、$\tau = 10^3$） | `exp/vision/main.py`（`program=vae`）、`pc.py` | `v17-chunk/v0-base-49b81800-s0/ckpt/04800.0.pt`（動画の時刻 480000 秒）。ほかに $\tau = 3\cdot10^3$・$10^4$ の 2 本、`v17-b1*`（B = 1）、`v17-c10-long/v0-base-966b62e1-s0`（$c$ = 10、run の 12000 秒まで） | 480 万（ストリーム） |
| `v14`（参考） | VAE（$\mu$） | 単一ストリーム B = 1、1800 秒 | 同上 | `v14-width/v0-base-b2006f0a-s0/ckpt/01800.0.pt`（V16〜V18c の config の元） | 1.8 万 |
| V18 P16 | JEPA | BP・AdamW、i.i.d. な batch 128、LeJEPA の形（EMA なし・stop-gradient なし）、$(1-\lambda)L_{\mathrm{pred}} + \lambda S$ | `exp/vision/jepa.py`・`jepa_iid.py` | 長い run: `v18-long/v18-long-lr1e-4-lam0.5-r{0.25,0.5,0.75}-s0`、`-r0.25-s1`（60000 step）。掃引: `v18-screen/v18-lr<lr>-lam<λ>-r<r>-s0/`（18 本、6000 step、K3 に崩れた run を含む） | 768 万（長い run） |
| V18b F | JEPA | 同上、パッチ 4 | 同上 | `v18b-long/v18b-long-f-lr1e-4-lam0.5-r0.25-s{0,1}`、掃引は `v18b-screen/` | 768 万 |
| V18b M | JEPA（K3 に崩れた負の対照） | 同上、SIGReg をトークン平均にかける | 同上 | `v18b-long/v18b-long-m-lr3e-5-lam0.8-r0.25-s{0,1}` | 768 万 |
| V18c | JEPA | 同上、パッチ 8、固定の $R$・$W$ | 同上 | 長い run: `v18c-long/v18c-lr{1e-4,1e-5}-lam{0.5,0.8,0.95}-r0.5-s{0,1}`（12 本、9 個ずつ）。掃引: `v18c-screen/` の 15 本（6000 step） | 768 万 |
| 学習前 | 各腕の step 0 | — | — | 各 run の `ckpt/00000.0.pt`（`init`・`init_f`・`init_m`・`init_c`）。V18c の掃引の 15 本の step 0 は同じ重み（README） | 0 |
| M18 の AE・VAE（参考、優先度は低い） | `exp/video` の決定的な自己符号化器と VAE | 単一ストリーム | tag `video-final`（`exp/video/vae/`） | `m18-ae-v1`、`m18-vae-v1` など | 128 px・30 fps のデータ（M18 は学べなかった段がある。入力の形が違う） |

- 長い run の checkpoint のファイル名は $t$ = 12.8 × step 秒: 0・25600（2000）・76800（6000）・128000（10000）・256000（20000）・384000・512000・640000・768000（60000）（確かめた）
- 1 つの checkpoint は 217〜303 MB（確かめた）
- どの run にも `config.yaml` と、`jepa.json`（JEPA）または `iid.json`（V16）に provenance（snapshot・mujou のコミット・host・コマンド行）がある（README）

### 1.2 物差し（学習の要らない符号と、派生の符号）

コードは `exp/vision/jepa_eval.py`（`ColorCode`・`PCACode`・`TokenCode`・`RandProjCode`）、`sem_eval.py`（`K3Code`、`inc64`）、`self_eval.py`（`MirrorColorCode`）、`gen_eval.py`（`ZeroCode`）。

| 名前 | 形 | 次元 | 何を表すか |
| -- | -- | -- | -- |
| `color`（`ColorCode(16)`） | 16 トークン × 24 | 384 | パッチ 16 ごと・チャネルごとの画素の平均（$x/255$） |
| `color8`・`color4` | 64 × 24・256 × 24 | 1536・6144 | パッチ 8・4 の同じもの |
| `pca` | 大域 | 1024 | 画素（8 × 3 × 64 × 64 = 98304）の PCA。分散は成分の順に下がる |
| `init256` | 大域 | 256 | `init` の $z$ を PCA で 256 次元に（VAE の $\mu$ と次元をそろえた版） |
| `z_w`（`RandProjCode`） | 大域 | 1024 | $W \cdot \mathrm{vec}(h)$、$W$ は固定のガウス（`W_SEED`）。$Tw$ が違えば別の写像 |
| `h_mean`・`h_flat`・`h_pool`・`zbar`・`zflat`・`zpca` | | 1024・$Tw$・16384・64・16384・1024 | `TokenCode` のプール |
| `z_k3` | 大域 | 1024 | 全トークンを位置の平均で置き換えた $W\mathrm{vec}(1 \otimes \bar t)$（場面だけ） |
| `inc64` | 大域 | 2048 | 64 px の最後のフレームの Inception pool3（外部の特徴。主の正解にはしない方針） |
| `uncond` | 大域 | 1024 | 0（条件なしのデコーダの床） |
| `MirrorColorCode(16)` | トークン | | 180° 反対のマスの平均の色（逃げ道の合成の例） |

V18 の物差しの値は `v18-baselines/posthoc_jepa/`、V18b は `v18b-baselines/`、V18c は `v18c-baselines/`、生成（G）の物差しは `v18c-gen/yardsticks/`、`sem` は `v18c-sem/yardsticks/`、場面は `v18c-scene/yardsticks/`（確かめた）。

### 1.3 潜在の形と尺度（比べるときにそろえるべきもの）

| モデル | トークン格子 | トークンの次元（$h$ → 符号） | 大域の符号 | 尺度・分布 |
| -- | -- | -- | -- | -- |
| VAE（V16・V17） | 4 × 4（パッチ 16） | $h$: 1024（頭の前、LayerNorm の後） | $\mu = $ 頭$(\mathrm{vec}\,h) \in \mathbb{R}^{256}$。事後の $\ell = \ln s^2$ も出る | KL で各座標が $N(0,1)$ に引かれる。使われない座標は分散が小さく $s^2 \approx 1$ になりうる（推測。active units で確かめられる） |
| V18 P16 | 4 × 4 | 1024 → $z_p = P h_p \in \mathbb{R}^{64}$（学習した $P$） | $z = \mathrm{vec}(z_p) \in \mathbb{R}^{1024}$ | SIGReg で $z \approx N(0, I)$（`sigreg_heldin` 1.18〜1.24、60000 step） |
| V18b F | 16 × 16（パッチ 4） | 1024 → 4 | $z \in \mathbb{R}^{1024}$ | 同上 |
| V18b M | 16 × 16 | 1024 → 64 | 平らな $z \in \mathbb{R}^{16384}$（評価は PCA の 1024 次元 `zpca`）。SIGReg は $\bar z \in \mathbb{R}^{64}$ にだけかかる | K3 に崩れた |
| V18c | 8 × 8（パッチ 8） | 1024 → $t_p = R h_p \in \mathbb{R}^{16}$（固定の $R$、seed 67） | $z = W\mathrm{vec}(t) \in \mathbb{R}^{1024}$（固定の $W$、seed 61） | $z$ は SIGReg で等方に近い。$t$ そのものは等方でない（SIGReg の理想は $\mathrm{Cov}(\mathrm{vec}\,t) = (W^\top W)^{-1}$、README の V18c の反証の節） |
| 学習前（各腕） | 同じ | 同じ | 同じ | `sigreg_heldin` は V18 の `init` で 66.4、F の `init_f` で 61.4（定数の値 52.3 の付近。README） |

違いとして書いておくこと:

- **大域かトークンか**: VAE の $\mu$ は位置を混ぜた 256 次元。JEPA の符号は位置ごとのトークンを並べたもの。「線形に切り分けやすいか」を測るとき、VAE は $\mu$、JEPA は $z$ で比べると、次元（256 と 1024）と構造（位置の添字の有無）が同時に違う。共通に読める点は 2 つある: (a) 頭・射影の前の $h$（VAE・P16 は 16 × 1024 で形が同一、F は 256 × 1024、V18c は 64 × 1024）、(b) $h$ の位置の平均 `h_mean`（1024）
- **次元の偏り**（V18 の教訓）: 後から当てはめるデコーダ・ridge・線形の動力学の容量は次元とともに増える。`init` と `init256` の `recon_blur` の差は 0.059 だった。次元の違う符号を、当てはめを含む量で比べるときは、PCA で次元をそろえるか、当てはめを含まない量を使う
- **基底と尺度**: VAE の $\mu$ は座標ごとに KL がかかるので、座標軸に意味がありうる（軸に沿った disentanglement の文献の前提）。JEPA の $z$ は SIGReg が回転に不変なので、軸に特別な意味はない（コードから: SIGReg は方向をランダムに引き直す）。軸に沿った指標（MIG・SAP など）は、JEPA には回転の任意性があって不利に働く。比べるなら、可逆な線形写像に不変な量（例: `linear_dyn` の白色化した `dynw_h`）を選ぶ
- **パッチの大きさ**: 同じ画素の領域でも、トークンの数が 16・64・256 と違う。V18b は「粗い 1 マス = 細かい格子の $s \times s$ ブロック」の写像（`jepa.mask_expansion`・`h_pool`）で画素の領域をそろえた
- **的の違い**: VAE は 8 フレームすべての画素を描き戻すように学ぶ。JEPA は隠した位置の表現を当てるように学ぶ。どちらも同じ 0.8 秒の stack を 1 点として扱う
- **見た量**: V16 は 600 万 stack、JEPA の長い run は 768 万 stack（batch 128 × 60000）、V17 はストリームで 480 万 stack

### 1.4 デコーダ（潜在の中を動いたときに画像で見るための部品）

- **VAE は自前のデコーダを持つ**（`VAEModules.decoder`、$z$ から 8 フレームの平均の画像）。$\mu$ の補間や加減の結果を、学習した本人のデコーダで画像にできる（コードから）
- **JEPA には自前のデコーダが無い。** 後から学ぶ 2 種類がある
  - 決定的なデコーダ（`eval.fit_decoder`、6000 組、4 seed）: `recon_blur` を測るたびに当てはめ直し、**保存していない**（確かめた: `posthoc_jepa/` には png と jsonl だけ）
  - 条件つきの flow matching のデコーダ（`exp/lib/flow_decoder.py` の `FlowUNet`、`gen_eval`）: **保存してある**（確かめた）。V18・V18b の長い run の `posthoc_gen/ckpt/dec-<step>-{z,z_w}-d<j>.pt`（step 0・6000・20000・60000）、物差しの `v18c-gen/yardsticks/posthoc_gen/ckpt/dec-yardstick-{color,pca,v16,v17,init_m,uncond}-d0.pt`。符号の標準化（`Standardizer`）も同じファイルに入っている（docstring）。V18c（パッチ 8）の符号の flow のデコーダは無い
  - 注意: flow のデコーダは符号に無い細部をもっともらしく描く。補間の途中の符号が学習の分布から外れても、それらしい画像を描きうる。画像の見た目は裏づけにとどめ、主の量にしない（1 回目の失敗の教訓とも合う）

## 2. データ

### 2.1 学習と評価の動画（確かめた）

- 準備したデータ `ad6e56b5995e`（`/data/geson/mujou-exp/.fleet/data/` と NAS の `/nas/geson/mujou-exp/data/`）: Walking Tours の 10 本（Amsterdam・Bangkok・Chiang_Mai・Istanbul・Kuala_Lampur・Singapore・Stockholm・Venice・Wildlife・Zurich）。元は 854 × 480・30 fps の YouTube の 480p（Wildlife だけ 640 × 360）。中央の最大の正方形を 64 × 64 に面積平均で縮め、10 fps にした u8
- Zurich（38996 フレーム、65 分）は学習に使わない評価専用。学習は残りの 9 本（Wildlife を含む、479350 フレーム）。`scene_eval` は Wildlife を外した 9 都市
- 元の動画は dev container の `/datasets/walking_tours`（std-00 のホストからは見えなかった。`memory/project_video_data.md` によれば container にマウント）

### 2.2 時間の並び

- **stack** = 連続する 8 フレーム（0.8 秒）。stack の間隔（`data.tours.stride`）は 1 フレーム = 0.1 秒なので、**隣り合う stack は 8 枚のうち 7 枚を共有する**
- i.i.d. の学習（V16・V18 系）は、学習の 9 本の全 stack から一様に、重複を許して引く（`iid.Units` の `"all"`）
- 評価の組（README・eval-guide）
  - Zurich の格子: 1 秒（10 フレーム）おきの 3899 stack（`gen_eval.Zurich`）。易しい候補は等間隔の 512 個（7〜8 秒おき）、難しい候補は前後 ±2〜±32 秒の 2 秒おき 33 個（`gen_eval.hard_sets`）
  - Zurich の val の 1024 窓（`eval.eval_splits`、中ほどの約 22 分に平均 1.3 秒おき）と calib
  - 当てはめの組（`StreamEvaluator.fit_stacks`、学習のストリームの 0〜6000 秒の奇数番目の 10 秒区間、約 6000 stack。Amsterdam 4905 + Bangkok 1093）
  - `sem_eval` の calib・sel・final（Zurich の格子を 6 区間・間隔 30 で分けたもの）
- 相関の時間: 画素の相関は 11〜36 分以上かけて下がる（M17、`memory/project_m17_wt_encoder.md`）。ブートストラップは連続したブロック（8 個 ≈ 61 秒と 16 個）で引く（`metrics.block_bootstrap`、`integrated_time`）

### 2.3 評価専用のデータ（確かめた）

| データ | 置き場所（std-00） | 形 | 中身 | 使える介入 |
| -- | -- | -- | -- | -- |
| `wide-61bf4460e101` | `/data/geson/mujou-exp/` | u8 `(3, 64, 112)`、10 fps、全フレーム（Zurich 38996） | 正方形の左右に 24 px ずつの帯。中央の 64 列は準備したデータとバイトまで同じ（`exp/lib/tours/wide.py` が確かめる）。Amsterdam・Bangkok・Zurich だけ | **本物の横の平行移動**（±24 px まで、端の折り返しなし、8 フレームとも同じ移動） |
| `hires-165822412e75` | 同上 | 320 × 320、選んだフレームだけ | Amsterdam 4905・Bangkok 1093・Zurich 3899（当てはめの組と Zurich の格子の最後のフレーム） | **拡大**（320 から切り出して 64 に縮める）、1 画素未満の平行移動。ただし stack の最後の 1 枚だけ |
| `hires-3e5bc8de6196` | 同上 | 320 × 320 | 9 都市の 2 秒おきの格子（Zurich 1950 など） | 同上 |
| `3c3e6805fdaa`（M17・M18 のデータ） | `/data/...fleet/data/` と NAS の `data/` | 128 × 128、30 fps、全フレーム（Zurich 116989） | 10 本すべて | **2 倍の拡大**（128 の中の 64 を切り出す）、**30 fps の細かい時間**、8 フレームとも本物の画素。準備したデータとはバイトでは一致しない見込み（変換の順が違う。要確認） |
| Imagenette | std-01 の `/data/geson/mujou-exp/.fleet/imagenette/u8/{train,val}.pt` | 128 px、train 9469・val 3925 | 静止画（`jepa_diag.still_stacks` で同じ画像を 8 枚並べる） | ラベルがあるので、方針上は主にしない |

注意: `hires` の色は準備したデータと 2〜3 階調ずれる（YUV と RGB の変換の順。`hires.py` の docstring）。`wt128-30fps` は作りかけ（`Stockholm.tmp`）で、使えるのは `3c3e6805fdaa` のほう。

### 2.4 画素で作れる介入（既存の関数）

| 介入 | 関数 | 連続な強さ | 画素の空間での性質（注意） |
| -- | -- | -- | -- |
| 明るさ（× 0.8）・白の偏り（1.1, 1.0, 0.9）・ぼかし（σ = 1 px） | `sem_eval.white_balance`・`blur`・`_nuisance`（`BRIGHT`・`WHITE`・`BLUR_SIGMA`） | 係数を連続に動かせる | 明るさと白の偏りは画素の**線形写像**。画素の PCA でもそのまま線形になる |
| 統計の合わせ | `sem_eval.match_stats` | | |
| 調和補間（隠した画素を周りから埋める） | `exp/lib/regions.harmonic_fill`・`harmonic_matrix` | | 線形 |
| 移植（ある領域の画素を別の stack のものに替える） | `regions.transplant`、`regions.pixel_mask` | 領域の選び方 | 画素の凸結合ではない入れ替え |
| マスク（4 × 4 の粗い格子の族 $F_k$、評価のマスク Q・H・R・S） | `jepa.mask_family`・`expand_masks`、`regions` | 隠すマスの数 | |
| 横の平行移動（本物） | `wide` のデータを切り出す（関数は無い。切り出すだけ） | 0〜±24 px | 画素の空間では強く非線形（軌道が曲がる） |
| 拡大・切り出し | `hires`・`3c3e6805fdaa` を切り出して縮める（`jepa_diag.shrink` は 2 × 2 の平均） | 倍率 | 非線形 |
| 時間のずれ | 格子の stack を $\Delta$ 秒ずらす（`gen_eval.Zurich`、`eval.stacks_at`） | 0.1 秒刻み（30 fps のデータなら 1/30 秒） | 自然な変化。画素では非線形 |
| 止めた stack（同じ画像を 8 枚） | `jepa_diag.still_stacks` | | 動きのない端 |
| 左右の反転・円環のずらし | 無い（`torch.flip`・`torch.roll` で 1 行） | | 画素の置換なので画素の空間で線形（自明な基準になる） |

## 3. 流用できる評価の部品

### 3.1 `exp/lib`

| モジュール | 中身 | disentanglement の評価での使い道（推測） |
| -- | -- | -- |
| `metrics.py` | `effective_rank`、`probe_r2`（標準化した ridge の $R^2$）、`probe_accuracy`、`bootstrap_se`、`block_bootstrap`・`contiguous_blocks`・`integrated_time`（時間相関のある標本の揺れ）、`scene_bootstrap`、`video_jackknife`、`identification`（中心化した cos の最近傍、`top1`・`pick_max`）、`pair_wins`（対の勝率 $A$）、`swap` | 介入の量を線形に読み出す $R^2$、揺れの見積もり、同定 |
| `readout.py` | 中心化した ridge（`fit_ridge`・`predict`・`select_ridge`）、`centered_r2`、`rank_bound`（Eckart–Young の天井）、重みつき ridge・`nested_cv` | 線形の読み出しの当てはめ |
| `linear_dyn.py` | 閉じた形の ridge の線形動力学 $\hat z_{k+1} = Az_k + b$、`rollout`、`errors`（`dyn_h`・`copy_h`、**白色化した `dynw_h`・`copyw_h` は可逆な線形写像に不変**）、`select_alpha` | 時間の線形性。空間の違う符号どうしを比べるときの不変な量 |
| `pca.py` | 乱択 SVD の PCA（`fit_pca`・`encode`・`decode`） | 次元をそろえる、`pca` の物差し |
| `randproj.py` | 固定のガウスの写像 | 次元をそろえる別の手 |
| `regions.py` | 評価のマスク・領域の画像・調和補間・移植 | 局所の介入 |
| `sigreg.py` | Epps–Pulley（`epps_pulley`・`sigreg`）、`expected_epps_pulley`（相関のあるガウス過程の期待値） | 潜在の分布がガウス・等方に近いかの量 |
| `gauss_kl.py` | ガウスの 2 次のモーメントの KL（$\frac12[\mathrm{tr}\Sigma + \lVert\mu\rVert^2 - D - \log\det\Sigma]$） | 共分散だけで見た「座標の独立さ」（ガウスの total correlation $-\frac12\log\det(\text{相関行列})$ に近い量）を出す材料 |
| `elbo.py` | VAE の ELBO の部品と、確率的 PCA の閉じた対数尤度 | VAE の KL を座標ごとに見る（active units）材料 |
| `flow_decoder.py` | 条件つき flow matching の U-Net、`fit_flow`・`euler_sample` | 潜在の補間を画像で見る（裏づけ） |
| `blur_error.py` | ぼかした二乗誤差（`recon_blur`） | 方針上は主にしない |
| `inception.py`・`places365.py` | 外部のネットワーク（重みは std-00 の `/data/.../.fleet/data/inception`、`places365`） | 方針上は主にしない。公開モデルを読み込む前例として使える |
| `tours/`（`walking_tours`・`hires`・`wide`・`stacks`） | データの準備と読み出し | 介入のデータ |
| `ckpt.py` | 評価の checkpoint の読み書き（`load_eval`） | |

### 3.2 `exp/vision`

| モジュール | 中身 | 使い道 |
| -- | -- | -- |
| `stream_eval.py` | `Code` プロトコル（`modules()`・`mu(x)`）、`StreamEvaluator.probes`（後から学ぶデコーダ + 線形の動力学、6000 組） | **新しいモデルを評価に載せる共通の口** |
| `jepa_eval.py` | `ColorCode`・`TokenCode`・`RandProjCode`・`PCACode`、`participation`・`token_r2`・`drift`、`build_run_jepa`・`arm_codes`・`vae_at`、`CodeStats`、`evaluate_yardsticks` | run と物差しを同じ統計に通す。VAE の読み込み（`vae_at`） |
| `jepa_diag.py` | `tokens`（$h$ の probe）、`sigreg`（Zurich の格子・`partition`・`autocov`）、`imagenette`（`shrink`・`still_stacks`） | $h$ を読む前例、時間の自己相関 |
| `jepa.py` | マスクの族・予測器・`predict`・`jepa_terms`・`rho_in`・`sigreg_points` | 予測器の出力（補完）を潜在の中の操作として使う |
| `fill_eval.py` | E1・E1e・E1g・E2（凍結した ridge の読み出しで隠した部分の特徴を当てる）、`e1f_masks`、`swap` の部分 | 読み出しを全体を見た符号で当てはめて凍結する手順 |
| `self_eval.py` | 自分の表現空間での補完の同定（`stage_scores`・`pair_wins`・`swap`・移植・局所性 $a_U$・$a_V$・$L$・`band_ratios`・`expected_pair_wins`） | 介入（移植）に対してトークンがどこで動くかを測る前例。局所性の量は「どの位置の画素を変えたら、どのトークンが動くか」の線形の分解そのもの |
| `sem_eval.py` | 高解像度の的、`sensitivity`（明るさ・白・ぼかしで的がどれだけ動くか）、`far_pairs`、`K3Code`、`zurich_split` | 介入の関数と「物差しの感度」を先に測る手順 |
| `gen_eval.py` | flow のデコーダの学習と生成、Zurich の格子・候補・`hard_sets`、`ZeroCode` | 潜在の中を動いた先を画像にする |
| `scene_eval.py` | 場面のカテゴリ（Places365）、`random_labels`（ランダムのラベルの帰無）、`synthetic_codes`（合成の符号で回収を確かめる） | 帰無と合成の回収試験の手順は流用できる。ラベルそのものは方針上使わない |
| `z_use.py` | 符号をずらしたときのデコーダの誤差 | VAE のデコーダが符号を読むかの確かめ |
| `iid.py` | VAE の BP・i.i.d. 学習（`neg_elbo`、`Units`） | β-VAE を作る土台（KL の重みを足すだけ、推測） |
| `jepa_iid.py` | JEPA の BP・i.i.d. 学習 | MAE・VICReg などを作る土台 |
| `results/v18c/*.py` | 判定の集計（`judge*.py`、`self_extra.py` など） | 集計の書き方の手本 |

## 4. 他の自己教師あり学習を足すとしたら

### 4.1 選択肢 A: 同じデータ・同じ ViT-S・同じ BP の i.i.d. 学習で作る

`jepa_iid.py` は mujou のノードではなく、素の PyTorch の BP で学ぶ（V16 の `iid.py` と同じ形）。比べるための学習なら、これに合わせるのが手間も公平さも最もよい。mujou のノード（ストリーム・PC-ALM）で作るのは、V19 以降でストリームでの比べが要るときにする（推測）。CLAUDE.md の 3 層の判定では、損失そのもの（VICReg・InfoNCE など）は複数の実験が使うなら `exp/lib/`、mujou の利用者一般に説明できるノードにするなら mujou になる。

| 手法 | 既にあるもの | 足すもの | 手間（推測） | 計算（推測、4090） |
| -- | -- | -- | -- | -- |
| β-VAE | `iid.py`（ELBO、学習する $\ln\sigma$）、`models.build_vae` | KL の重み $\beta$ の引数。V7 で $\sigma$ と KL の釣り合いを扱った前例がある | 小（数十行 + テスト） | V16 と同じ 1 本 73〜87 分 |
| 決定的な AE | 同上 | KL を外し、$\mu$ をそのまま使う | 小 | 同上 |
| MAE | マスクの族、`TokenEncoder` の `keep`（見えるトークンだけを通す）、予測器 | 的を「隠したパッチの正規化した画素」に替え、予測器の出力の次元を変える（パッチ 8 なら 1536） | 小〜中（100〜200 行 + テスト） | V18c と同程度（パッチ 8 で 0.115 秒 / step、60000 step で約 2 時間） |
| VICReg | SIGReg の口（`sigreg_points`）、学習ループ | 2 つの view を作る部分、分散・共分散・不変の 3 項 | 中 | view が 2 つなのでエンコーダの計算は約 2 倍 |
| SimCLR（InfoNCE） | 同上 | view と InfoNCE。batch 128 では負例が少ない | 中 | 同上 |
| LeJEPA の view の形（SIGReg + view の不変） | SIGReg、学習ループ | view を作る部分 | 中。いまの JEPA と目的の残りが同じなので、差が view だけになる | 同上 |
| BYOL・DINO | `exp/video/jepa/ema.py` の `Following`（M17 の EMA、tag `video-final`） | EMA の教師、DINO は中心化・温度・multi-crop | 大（ハイパーパラメータが多い） | multi-crop で数倍 |

view の作り方は、公平さの論点になる。

- **拡張**（切り出し・色の揺らし・反転）: 人が「何に不変であるべきか」を決める。色の揺らしで学んだモデルは明るさに不変になるので、2.4 節の明るさの介入の試験は学習の前に答えが決まる。**拡張で入れた不変性と同じ介入で disentanglement を測ると、循環になる**（導いた）。ユーザーの方針「人のドメイン知識を入れない」にも触れうる
- **時間の view**: 同じ動画の $\Delta$ 秒離れた 2 つの stack を view にする。人が選ぶのは $\Delta$ だけで、時間で変わるものに不変になる。データ（Walking Tours の連続した動画）にもともとある構造を使う。ただし、時間の線形性を測る試験とは循環しうる（同じ理由）

公平にそろえるもの: データ（学習の 9 本、Zurich を外す）、入力（8 フレーム × 64 px、early fusion）、エンコーダ（同じ ViT-S・μP・初期値の seed）、batch と step の数（見た stack の数）、optimizer と学習率の掃引の幅、設定の選び方（disentanglement の量で選ぶと循環するので、各手法の自分の目的の値か、方針で決めた別の量で選ぶ）、読む位置（$h$・`h_mean`・各手法の出力）。

### 4.2 選択肢 B: 学習済みの公開モデルを使う

| モデル（推測を含む） | 学習のデータ・形 | 合わない点 |
| -- | -- | -- |
| DINOv2 ViT-S/14 | LVD-142M、224 px の 1 枚 | パッチ 14、64 px を拡大して入れる |
| MAE ViT-B/16、I-JEPA ViT-H/14 | ImageNet、224 px の 1 枚 | 大きさが 1〜2 桁違う |
| V-JEPA・VideoMAE | 動画、16 フレーム × 224 px | 8 フレーム・10 fps と合わない |
| SimCLR・VICReg・BYOL の ResNet-50 | ImageNet | 構造が違う |
| **DoRA ViT-S/16**（Venkataramanan ら、ICLR 2024） | **Walking Tours そのもの**で学んだ（DINO の形） | 同じデータの領域で最も近い。ただし学習に Zurich を含むか（10 本すべてで学んだ重みか、Venice だけか）を**要確認**。含むなら Zurich は評価の外の動画にならない。224 px・30 fps・multi-crop の拡張つき |

- 依存: いまの依存は `hydra-core`・`imageio-ffmpeg`・`mujou`・`torchvision` だけ（`pyproject.toml`）。公開モデルは `torch.hub` か `uv add timm` で入れることになる（`pip install` は使わない規約）
- 前例: `exp/lib/inception.py`・`places365.py` が外部の重みを読み込んでいる。同じ形で `Code`（`modules()`・`mu(x)`）を書けば評価に載る
- **公平さの問題**: データ・解像度・入力の形（1 枚か 8 枚か）・モデルの大きさ・計算量・拡張がすべて違い、差をどれかに帰せない。64 px を拡大して入れると、モデル側の癖が先に出る（1 回目の試みで Inception に拡大した 64 px を入れて目盛りが 0.002〜0.024 に狭まった教訓）。8 フレームの stack をどう入れるか（最後の 1 枚、フレームごとの平均）も決めなければならない。参考の上側・別の系として並べる用途に向き、同じ条件での比べには向かない（推測）

## 5. 棚卸しから見えた、評価を設計するときの注意（理論の裏づけつき）

1. **画素の空間そのものが、いくつかの介入に対して線形である。** 明るさ・白の偏りは画素の線形写像で、反転・円環のずらしは画素の置換（線形の作用）である。これらの介入で「潜在の中で線形に動く」ことを測ると、画素の PCA（`pca`）が自明に満点に近くなる（導いた）。ユーザーの言う「画像を足しても意味がない」が当てはまるのは、本物の平行移動（端から新しい内容が入る）・拡大・時間の変化（視差・遮蔽）のように、画素の空間で軌道が曲がる変化である。`wide`・`hires`・`3c3e6805fdaa`・時間のずれは、この種類の介入の材料になる
2. **隣り合う stack は 7 枚を共有する。** 時間の線形性を 0.1 秒刻みで測ると、`color` のような入力をほぼ写す符号が自明に良くなる（V18 で (`dyn_1/copy_1`, `color`) が 0.109 になり使えなかった）。時間の介入は 0.8 秒以上離れた stack で作るか、`color` と `pca` を下側の物差しとして必ず並べる
3. **次元と当てはめの容量をそろえる**（1.3 節）。VAE の $\mu$ は 256、JEPA は 1024、$h$ は 16384 以上
4. **軸に沿った disentanglement の指標は、回転に不変な目的（SIGReg）で学んだ符号に不利である**（1.3 節）。比べるなら、可逆な線形写像に不変な量を選ぶか、両方の版を並べる
5. **教師なしの disentanglement は、帰納的な偏りか何らかの教師なしには識別できない**（Locatello ら 2019 の不可能性の結果。文献の記憶で、原文はこの棚卸しでは読んでいない）。既存の指標（β-VAE・FactorVAE・MIG・DCI・SAP）は真の因子のラベルを前提にする。人のラベルを使わない方針のもとでは、「因子」の候補は、こちらが画素で作れる介入の強さ（移動量・倍率・明るさの係数・時間のずれ）になる。これは人が決めた変換だが、意味のラベルではない。これを因子として使ってよいかは、ユーザーの判断が要る
6. **介入の強さの読み出しは、学習した読み出しを含む。** 方針（崩壊や当てやすい的への逃げを対照で塞ぐ）に合わせて、`color`・`pca`・学習前・K3 の合成（`K3Code`）を同じ量に通す。3 回の作り直しの教訓（物差しの値の幅が狭い、的が当てやすいものに変わる）から、介入の量の目盛りを先に物差しで確かめる（`sem_eval.sensitivity` の手順が手本になる）
7. **VAE は自前のデコーダで潜在の補間を画像にできるが、JEPA は後から学ぶデコーダが要る。** 画像で見る比べは、同じ種類の後から学ぶデコーダ（flow のデコーダは VAE の $\mu$ の版も保存済み）にそろえる。flow のデコーダは分布の外の符号でもそれらしく描くので、裏づけにとどめる
