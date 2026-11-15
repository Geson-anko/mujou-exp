# M17: Walking Tours でエンコーダを詰める（実装計画）

## 概要

128×128×3・30 fps の Walking Tours で、再構成の項なしの JEPA（V-JEPA・V-JEPA 2 の入力と masking、LeJEPA の SIGReg）でエンコーダを PC-ALM と BP で学ぶ。$z$ からの再構成・予測器を通した復元・$z$ の上の線形の動力学は posthoc だけで測る。合意済みの計画は `exp/video/MILESTONES.md` の「## M17」、決定は `exp/video/PLAN.md` 12 節「M17 の前に決めたこと」。この計画はそれをシグネチャの粒度に落とす。

## 仮説と判定基準

- 完了の条件は MILESTONES.md の M17「完了の条件」をそのまま使う。閾値は手順 3（物差し）を見て確定し、回す前に `results/m17.md` に書く
- 探索の目的（val、最後のチェックポイント、小さいほどよい）は $J = \max(\mathrm{recon}/\mathrm{recon}_{\mathrm{PCA}},\ \mathrm{dyn}_1/\mathrm{dyn}_{1,\mathrm{PCA}})$。`z_spread` < 1e-3（`programs.vision.COLLAPSE`）か非有限のとき `measure.objective.PENALTY`（$\log 10$）
- PCA は同じ $D$、同じデコーダ、同じ ridge、同じデータを通した値（線形の PCA の再構成 `recon_lin` は参考に併記するだけ）
- 速さの規則（ユーザー）: PC-ALM の確認の 1 run が 4090 で 7 時間程度に収まらなければ $N=8$ に落とす。判断は手順 4 の `task=bench` の実測で行う（下の見積もりでは収まらない見込み。確認事項 1）

## 原典から採るもの

| 項目 | 原典 | M17 の既定 | 探索の軸 | 理由 |
| -- | -- | -- | -- | -- |
| patch・tubelet | V-JEPA §3.3・App. B（2×16×16）、V-JEPA 2 §2.1（同） | patch 16、$\tau=2$ | なし（$N=1$ の対照だけ $\tau=1$） | ユーザーの指定と原典が一致 |
| 積み方 | V-JEPA App. B: 16 フレーム・stride 4（30 fps で約 2 秒）。V-JEPA 2 Table 9: 16 フレーム・4 fps（4 秒） | $N=16$、$\Delta=3$ フレーム（0.1 s、1.5 秒） | $N\in\{8,16\}$、$\Delta\in\{2,3,6\}$ フレーム（0.067・0.1・0.2 s） | 0.2 s は V-JEPA 2 の 4 fps に近い側 |
| マスク | V-JEPA §3.2・App. C: 短い範囲 = 8 ブロック・各 0.15、長い範囲 = 2 ブロック・各 0.7、aspect (0.75, 1.5)、時間方向は全部（tube）、平均で約 90% を隠す。2 種類を別の mask にする（App. B の multi-mask、E.4: 1 種類より良い）。V-JEPA 2 Table 9 も同じ（scale [0.15, 0.7]、temporal [1, 1]） | context view を 2 つ（短い範囲・長い範囲）、tube | 各 view の scale、view の組（短・長・両方） | ユーザーの「0.5〜0.9 を探索」は scale の範囲で覆う（下の `MaskConf`） |
| context の encoder | V-JEPA §3.2: 隠したトークンは x-encoder の入力から落とす | 落とす（`layout="packed"`） | なし | 128×128×16 で計算が収まる唯一の形（「見積もり」）。M15 の「隠した patch を 0 にして mask の channel を足す」形は M15/M16 の preset のために残す |
| 予測器 | V-JEPA §3.3: context のトークン + 学習する mask token（共有のベクトル + 3D の位置）、出力は隠した位置だけ。12 block・幅 384。V-JEPA 2 Table 12: ViT-s（幅 384・深さ 12） | 幅 = エンコーダの幅、深さ 2（M15） | 深さ $\{2, 4\}$ | 深い予測器は計算が重い（予測器だけで 2 view × 512 トークン）。M15 の 2 から始める |
| 位置の符号 | V-JEPA App. B: 絶対の 3D sincos。V-JEPA 2 §2.1: 3D-RoPE（次元を時間・縦・横の 3 つに分け、各々 1D の回転）。「最大のモデルの学習を安定にする」 | 3D sincos（Embed で足す、予測器でも足す） | なし | 下の「採らないもの」 |
| 損失 | V-JEPA Eq. 2・V-JEPA 2 Eq. 1: L1。LeJEPA §5.1 Eq. 6〜8: 二乗 | 二乗（M15 と同じ `prediction_errors`） | `pred_loss` $\in\{$l2, l1$\}$ | 1 行の分岐で原典の両方を試せる |
| target | V-JEPA §3.1・V-JEPA 2 §2.1: EMA + stop-gradient。LeJEPA: どちらも使わない | EMA なし、同じ重み。`sg` は軸 | `sg` | ユーザーの規則（EMA なし） |
| SIGReg | LeJEPA §4.2（Def. 2）: Epps-Pulley、方向は毎 step 引き直す（§4.3）、17 点の台形則（§5.1）、$\lambda=0.05$（§6.1）、batch ≥ 128 を推奨（§6.1）、minibatch の偏りは batch 16 でも小さい（§4.2.3） | `exp.lib.sigreg` のまま、$z$ にかける | かけ先・重み | M15 と同じ |
| pool | V-JEPA §4.3 は attentive probe（平均 pool より大きく良い） | target view の全パッチトークンの平均、必要なら線形の射影で $D$ へ | 幅と $D$ | ユーザーの制約（1 本のベクトル、$D \le 1024$）。attentive probe は学ぶ読み出しなので posthoc 以外に置けない |

### 採らないもの（1 行ずつ）

- 3D-RoPE: `pc.Layer` の module は `h` しか受け取らないので、各 block に位置を届けるには位置を `h` の列として全ノードに運ぶ必要がある。原典の効果は最大のモデルの安定（V-JEPA 2 §2.1）で、幅 256 の ViT には小さいとみる。results に「採らなかった」と書く
- 進む解像度の学習・cooldown（V-JEPA 2 §2.4）: 継続学習のストリームに終わりの段が無い
- 予測器の幅を絞る `pred_dim`（V-JEPA の 384）: 幅 = エンコーダで足りる。必要になってから
- データの拡張（V-JEPA App. C の flip・resize）: 中央の正方形をそのまま使う（M15 と同じ）

## 公開インターフェース案

`T_s = (S/p)^2`（空間の patch 数、128/16 → 64）、`T_t = N/τ`、`T = T_t T_s`（N=16 で 512）、`P = τ C p^2`（1536）、`K_v` は context view $v$ の残すトークン数（固定、`K_v = k_v T_t`、$k_v$ は空間の数）、`L = T + Σ_v K_v`。

### スキーマ（`exp/video/schema/`）

- `ToursConf`（data.py）に `advance: int = 1` を足す。1 run frame（`time.frame`）にストリームが進む動画のフレーム数。M17 は 3（30 fps の動画を 10 fps で学ぶ）。`fps` は 30.0、`size` は 128
- `VisionConf`（vision.py）に足す:
  - `layout: str = "pixels"`（`LAYOUTS = ("pixels", "packed")`）。`"pixels"` は今の M15/M16 の形で、振る舞いを変えない
  - `tubelet: int = 0`（0: $\tau = N$、今の tube。`packed` では必須で `frames % tubelet == 0`）
  - `stride: int = 1`（積む間隔 $\Delta$、動画のフレーム数）
  - `z_dim: int | None = None`（None: 幅と同じで射影なし。違えば `Linear(dim, z_dim)`。`1 ≤ z_dim ≤ 1024`。射影ありで `sigreg_on == "tokens"` は拒む: 射影に勾配が届かない）
  - `context_masks: list[MaskConf]`（`packed` だけ。`views == 1 + len(context_masks)` を検査）
  - `pred_loss: str = "l2"`（`("l2", "l1")`）
  - `packed` では `z_pool == "mean"` だけを許す（V-JEPA は CLS を使わない、App. C）。`dim % 16 == 0`（3D sincos の分割）
- `MaskConf`（vision.py、新規の dataclass）: `blocks: int`、`scale: float`（1 ブロックの面積 / 空間の格子、V-JEPA の spatial scale）、`aspect: tuple[float, float] = (0.75, 1.5)`。残す数 $k_v$ は `expected_keep` で決める（下）
- `MeasureConf` に `steps: int = 10`（評価の窓の先の step 数、線形の動力学の最大の地平）を足す。`horizons` は M17 の preset で `[1, 5, 10]`

### データ（`exp/video/data/`）

- `walking_tours.prepare(..., fps=30.0, size=128)` はそのまま使う（引数だけ）。新しい data_version は `finalize` の出力
- `walking_tours.eval_windows(root, *, held_out, clips, frames, stride, steps, seed) -> Tensor`: u8 `(clips, frames + steps, 3, S, S)`。窓 $c$ の $j$ 枚目は $s_c + j\Delta$。slot の長さは `(frames + steps - 1) * stride + 1`、slot の選び方は `eval_clips` と同じ（`stride=1, steps=0` で `eval_clips(length=frames)` とバイト単位で同じ）
- `frames.StackedTourFrames(root, *, n, held_out, frames, stride, advance)`（`FrameStreams` と同じ protocol だが `next()` の戻りが違うので別 protocol `StackStreams`）:
  - `next() -> tuple[Tensor, Tensor, Tensor]`: u8 の積み `(n, N, 3, S, S)`（古い順）、`ready (n,)` bool（積みが 1 本の動画の中に収まる）、`reset (n,)` bool（最初の run frame、または前の run frame から動画が変わった）
  - ストリーム $i$ の run frame $k$ の最後のフレームはループの $(\lfloor iL/n \rfloor + k\cdot\mathrm{advance}) \bmod L$（`TourFrames` と同じ並べ方）。`state_dict` は `{"k"}` だけ
  - 積みはディスクから直接読む（リングを持たない）。ホストの役目なので局所性に触れない

### ViT（`exp/video/jepa/vit.py`）

- `patchify_tubelets(x: Tensor, patch: int, tubelet: int) -> Tensor`: `(n, N, C, H, W)` → `(n, T, P)`、トークンの順は (時間の slot, 行, 列)、各トークンは `(τ, C, p, p)` を平らにしたもの
- `sincos_3d(grid: tuple[int, int, int], dim: int) -> Tensor`: `(T, dim)`。次元を時間 `dim/4`、縦 `3dim/8`、横 `3dim/8` に分け、各々 `sincos_2d` と同じ式の 1D（採用理由: V-JEPA の 3D sincos と同じ考え方で、`dim % 16 == 0` で割り切れる）
- `PackedEmbed(*, patch_dim: int, dim: int, grid3: tuple[int, int, int], mean: float, std: float)`: 入力 `(n, L, P + 1)`（最後の列はトークンの id、float）→ `(n, L, dim)` = `Linear((x - μ)/σ) + sincos_3d[id]`。id の列は detach して読む
- `Block(dim, heads)` の `forward(x, segments: Sequence[int] | None = None)`: `segments` を与えると、トークンの軸を区間に切って区間の中だけで attention（block 対角）。LN と MLP はトークンごと。None は今と同じ
- `Stage.forward(x, segments=None)` と `Encoder.forward(x, segments=None)` に同じ引数を通す
- `Segmented(stage: Stage, segments: tuple[int, ...])`（nn.Module）: `pc.Layer` に渡す包み。`forward(x) = stage(x, segments)`。パラメタは stage のもの（checkpoint の名前は `Encoder` のまま）
- `PackedPredictor(blocks, dim, grid3)`: `forward(ctx: Tensor, ctx_ids: Tensor, tgt_ids: Tensor) -> Tensor`。`ctx (n, K, D)`・`ctx_ids (n, K)`・`tgt_ids (n, M)`（long）から、隠した位置の予測 `(n, M, D)`。入力は `[ctx + pos[ctx_ids], mask_token + pos[tgt_ids]]`、blocks・LN・Linear のあと後ろの M 個を返す。`mask_token` は `(dim,)` の学ぶパラメタ（std 0.02）
- `VisionModules` に `proj: nn.Linear | None` と `layout`・`tubelet`・`segments` を足す。`named()` は射影があれば `"proj"` を含む。`encode(chunks)` は packed では target view だけ（`segments=(T,)`）を通して平均 pool → 射影。`tokens(chunks) -> (n, T, dim)`（target view のトークン、posthoc のトークンのデコーダ用）を足す
- `build_vision`: packed では `PackedEmbed`・`PackedPredictor`・射影を作る。初期化の順は stages → predictor → proj（decoder は M17 では作らない）

### マスク（`exp/video/jepa/masks.py`）

- `multiblock_masks(n: int, grid: tuple[int, int], mask: MaskConf, *, keep: int, generator) -> Tensor`: `(n, gh, gw)` bool、True が隠す。`blocks` 個の矩形（面積 `scale * gh * gw`、aspect は log 一様）の和集合を引き、見える数がちょうど `keep` になるよう、多ければ見える patch を乱数で隠し、少なければ隠した patch を乱数で戻す（固定の形にするため。V-JEPA の collator が batch の中で数をそろえるのと同じ目的）
- `expected_keep(grid, mask: MaskConf, *, draws: int = 4096, seed: int = 0) -> int`: 調整前の見える数の平均を丸めたもの（固定の generator で決定的）。$k_v$ はこれ
- `pack_views(chunks_u8: Tensor, masks: Sequence[Tensor], *, patch: int, tubelet: int) -> tuple[Tensor, Tensor]`: 積み `(B, N, C, H, W)` と各 context view の mask `(B, gh, gw)` から、`h` `(B, L, P + 1)`（target の T トークンを id 順、続いて各 view の残すトークンを id 順。最後の列が id）と `y` `(B, (V-1) T)`（各 view の、予測するトークンが 1 の float）。mask は時間の全 slot に同じ（tube）

### ノード（`exp/video/jepa/nodes.py`・`bp.py`・`nets.py`）

- `StackChunker(Node)`: `prev` は `pc.msg.Sample`（`x` u8 `(B, N, C, H, W)`、`y` zeros `(B, 1)`、`mask`、`reset`）。新しい書き込みを見た update で、各 context view の mask を自分の generator で引き（view の順、`multiblock_masks`）、`pack_views` を一度だけ送る。sample は stream（`B` 行、view を行に並べない）。`mask` は host の `mask`（= `ready`）。パラメタなし、`next` の credit は捨てる（今の `Chunker` と同じ）
- `JEPAHead` を packed でも動くようにするか、別クラス `PackedJEPAHead(Node)` にする（推奨: 別クラス。既存の `JEPAHead` の CUDA graph と分岐を増やさない）。受け取るのは `(B, L, D)`。中で target・各 context に切り、`PackedPredictor`、`jepa_loss_packed`、SIGReg、射影。credit は自分のエネルギー $E = B L$ の detach した leaf からの `torch.autograd.grad` だけ（今と同じ）。`z_out` は射影後の $z$ `(B, z_dim)`。射影と予測器はこのノードのパラメタ
- `JEPABP` は packed の modules でも `terms` を計算できるよう、`jepa_loss_packed` を呼ぶ分岐を足す（1 ノードの中の BP。今と同じ）
- `nets.build_vision_net`: `layout == "packed"` で `StackChunker → Segmented(stage) の pc.Layer × nodes → PackedJEPAHead`（PC）か `StackChunker → JEPABP`（BP）

### 損失（`exp/video/jepa/loss.py`）

- `jepa_loss_packed(tokens: Tensor, y: Tensor, valid: Tensor, directions_z: Tensor, directions_t: Tensor | None, predictor: PackedPredictor, *, keeps: Sequence[int], proj: nn.Module | None, lam: float, sg: bool, sigreg_on: str, pred_loss: str) -> JEPATerms`
  - `tokens (B, L, D)`、`y (B, (V-1) T)`、`valid (B,)`
  - $L^{\mathrm{pred}}_b = \frac{1}{V-1}\sum_v \frac{1}{M_v}\sum_{t \in \mathrm{masked}_v} \ell(\hat s_{b,v,t}, s_{b,t})$、$\ell$ は `l2` で $\frac1D\lVert\cdot\rVert^2$、`l1` で $\frac1D\lVert\cdot\rVert_1$。$M_v = T - K_v$ は固定
  - SIGReg と全体の式は `jepa_loss` と同じ（$\frac1B\sum_{b\in M}[(1-\lambda)L^{\mathrm{pred}}_b + \lambda S]$）。`"z"` は射影後の $z$ に `directions_z (M, z_dim)`、`"tokens"` は target のトークンに `directions_t (M, dim)`
  - `JEPATerms.recon` は常に 0（M17 は再構成の項を持たない）

### 評価（run の中、`exp/video/programs/vision.py`）

- `Measures` の packed の分岐: split ごとに `eval_windows`（`clips × windows` = 256 窓、u8 のまま device に置く）と、view ごとの固定の mask（split の seed）。行は `loss`・`pred`・`sigreg`・`rank`・`z_std_min`・`z_spread`（窓の最初の積み $k=0$ で）。`thumb`（線形の probe）は M17 では出さない（読み出しを学ぶのは posthoc だけ、という規則に寄せる）。val の行の `objective` は書かない（posthoc が書く）
- `vision_objective` は packed の行に `None` を返す

### posthoc（`exp/video/measure/posthoc.py`・新規 `exp/video/measure/linear_dyn.py`・`exp/video/measure/pca.py`・`exp/video/measure/restore.py`）

- `pca.fit_pca(x: Tensor, d: int, *, gen: torch.Generator, niter: int = 4) -> tuple[Tensor, Tensor]`: `x (n, F)` から平均 `(F,)` と主成分 `(d, F)`（行が正規直交）。`torch.svd_lowrank`（`q = d + 16`）を `fork_rng` の中で `gen` から引いた seed で
- `pca.encode(x, mean, comps) -> (n, d)`、`pca.decode(z, mean, comps) -> (n, F)`
- `linear_dyn.fit(z0: Tensor, z1: Tensor, *, alpha: float) -> tuple[Tensor, Tensor]`: $\min_{A,b}\sum\lVert z_1 - A z_0 - b\rVert^2 + \alpha\lVert A\rVert_F^2$ の閉じた形（float64、平均を引いて $A = C_{10}(C_{00} + \alpha I)^{-1}$、$b = \bar z_1 - A\bar z_0$、`torch.linalg.solve`）。`A (D, D)`・`b (D,)`
- `linear_dyn.rollout(A, b, z0: Tensor, steps: int) -> Tensor`: `(n, steps, D)`、$\hat z_{h} = A\hat z_{h-1} + b$
- `linear_dyn.errors(A, b, zs: Tensor, horizons: Sequence[int]) -> dict[str, float]`: `zs (n, 1 + H, D)`（窓の $k=0..H$）。`dyn_h` = $\overline{\lVert\hat z_h - z_h\rVert^2} / \overline{\lVert z_h - \bar z_h\rVert^2}$、`copy_h` = 同じ分母で $\hat z_h = z_0$、`dyn_h/copy_h`
- `linear_dyn.select_alpha(z0, z1, calib: Tensor, *, alphas: Sequence[float]) -> float`: calib の `dyn_1` が最小の $\alpha$。格子は $c\cdot\mathrm{tr}(C_{00})/D$、$c \in \{10^{-8}, 10^{-7}, \dots, 10^{0}\}$（probe が calib で ridge を選ぶのと同じ扱い。採用理由: Zurich の val/test を選択に使わない）
- `restore.token_decoder(dim: int, patch: int, channels: int, hidden: int = 512) -> nn.Module`: トークン `(n, dim)` → その tubelet の最後のフレームの patch `(n, C, p, p)`。Linear-GELU-Linear-GELU-Linear
- `restore.fit_token_decoder(tokens, patches, *, steps, batch, lr, gen) -> nn.Module`: `fit_decoder` と同じ Adam・同じ乱数の扱い
- `restore.restore(mods: VisionModules, chunks: Tensor, mask: Tensor, dec: nn.Module, *, keep: int) -> tuple[Tensor, Tensor]`: context view（`mask` で落とした積み）をエンコーダ（`segments=(K,)`）と予測器に通し、最後の時間の slot の各位置を、見える位置は context のトークン、隠した位置は予測器の出力として `dec` で描く。戻りは画像 `(n, C, H, W)` と隠した画素の mask `(n, H, W)`。target view の積みも隠した画素も読まない
- `posthoc.stream_windows(conf: RunConf, root: Path, *, windows: int = 2048, steps: int = 8) -> Tensor`: 学ぶ 9 本のループから等間隔に始点を取り、動画の境をまたぐ窓を除いた u8 `(n, N + steps, 3, S, S)`（`eval_windows` と同じ $\Delta$）。デコーダ（$z$ とトークン）と ridge はこれだけで学ぶ（9 本から、Zurich は測るだけ）
- `Decoding` の packed の分岐（`_PackedDecoding` に分けてよい）。単位 0 は物差し、単位 1.. はチェックポイント:
  - 単位 0: PCA-$D$（学ぶ窓の最後のフレームで `fit_pca`、`gen = seed + PCA_SEED`）と学習前の ViT（`build_vision(conf, seed=conf.seed)`、ckpt を読まない）を、下のチェックポイントと同じ評価に通し、`baselines.jsonl` に書く
  - 各チェックポイント: 学ぶ窓を符号化 → $z$ のデコーダ（今の `decoder`・`fit_decoder`、128×128 も対応済み）→ ridge（α は calib で選ぶ）→ トークンのデコーダ（target view の最後の slot のトークン）→ val・test で指標と画像
  - 1 行の鍵: `t`・`frame`・`split`・`code`（`"vit"`・`"init"`・`"pca"`）・`recon`（$z$ のデコーダの画素の二乗誤差 / 学ぶフレームの平均の画像の誤差）・`recon_fit`・`recon_lin`（pca だけ）・`dyn_{1,5,10}`・`copy_{1,5,10}`・`dyn_h/copy_h`・`img_dyn_h`（$g(\hat z_h)$ と真のフレームの画素の誤差 / 平均の画像の誤差）・`img_dyn_h/copy`（同じ誤差 / 最初のフレームを写した誤差）・`restore_{short,long}_{mask,all}`（隠した領域・画像全体、それぞれ平均の画像の同じ領域の誤差で割る）・`restore_target_all`（target view のトークンを描いた天井）・`rank`・`z_std_min`・`z_spread`・`alpha`・`objective`（val の `vit` だけ）
  - 画像（`post/decode/`）: `<t>.png`（val 8 窓の真の最後のフレームと再構成）、`<t>-restore.png`（真、短い範囲の context、その復元、長い範囲の context、その復元、target の天井の 6 行）、`<t>-dyn.png`（4 窓、$h = 0, 1, 5, 10$ の真・$g(\hat z_h)$・$g(z_h)$）。`init*.png`・`pca*.png` も同じ形
  - `result.json` の `final`: `{"decode": {split: 最後の vit の行}, "baselines": {split: {code: 行}}}`
  - 平均の画像は学ぶ窓の最後のフレームの画素ごとの平均（今の `pix_mean` は split 自身の平均。M17 では val・test を物差しの計算に使わない）
  - $z$ のデコーダの標本は学ぶ窓の全 $z$（2048 × 9 = 18432、今の `FRAMES` 16384 とほぼ同じ）、`STEPS` は 2000 のまま。手順 4 の試しで `recon_fit` と `recon` の差を見て足りなければ、study の前に一度だけ変える（以後は固定）
  - 学ぶフレームの画素は u8 のまま device に置き、batch ごとに float にする（18432 × 49152 B ≈ 0.9 GB。float では 3.6 GB）
- job の `decode` 設定に `"ckpts": "all" | "last"` を足す（study は `last`、確認は `all`）
- `programs/vision.py` に `task="baselines"`: 単位 0 だけを回す（親の run なし、job の config で）。手順 3 の物差しに使う

### 掃引（`tools/sweep/`）

- study の TOML に `[study] objective_from = "evals" | "posthoc"`（既定 `"evals"`）と `[study.posthoc]`（`res`、`gpu_mem`、`ram`、`ckpts`）を足す
- `"posthoc"` では: trial の run が completed になったら、`task=posthoc` の job（`parent` = NAS の `runs/<batch>/<run_id>` の絶対パス、`after` = run_id、batch `<study>-post`）を入れ、その `result.json` の `final.decode.val.objective` を seed で平均して tell する。途中の報告はしない（ASHA は働かない）。diverged の seed は `PENALTY`
- `followup.confirm` は `objective_from == "posthoc"` の study では、確認の各 run の後にも posthoc（`ckpts=all`）を入れる

### TensorBoard（新規 `tools/tb/`）

- uv の別プロジェクト（`tools/sweep` と同じ形）。依存は `tensorboard` と `tensorboardX` を `uv lock` のときの版で固定（採用理由: grpcio などが free-threaded の 3.14t に wheel を持たない見込み。`requires-python = "==3.12.*"` から始め、入らなければ 3.13）。torch を入れない
- `python -m tb export --runs <dir>... --out <logdir>`: run ごとに `<logdir>/<batch>/<run_id>/` へ event を書く
  - scalars: `stream/<key>`（step = frames）、`eval/<split>/<key>`（step = round(t)）、`post/<split>/<key>`（step = round(t)）、`base/<split>/<code>/<key>`（step 0 と最後の t に同じ値、横線の物差し）
  - images: `post/decode/*.png` を PNG のまま（`Summary.Image(encoded_image_string=...)`、幅と高さは IHDR から標準ライブラリで読む）
  - hparams: config の `vision.*`・`optim_vision.*`・`learner.kind`・`seed` と、最後の `objective`・`recon`・`dyn_1` など
  - text: snapshot、mujou の SHA、data_version、seed（`CODE.json`・`provenance/`）
- `just tb export --batch <B>...` と `just tb view <logdir>`（`tensorboard --logdir <logdir> --port 6006`、Mac で `http://localhost:6006`）。`tb/` は `.gitignore` に足す

## 理論上の対応

- 学習則は PC-ALM のまま。エンコーダの各 stage が `pc.Layer`（制約 $h = f(h_{\mathrm{in}};\theta)$）、損失は `PackedJEPAHead` のエネルギー $E = B L$ だけ。credit は $-J_h^\top\delta$ のメッセージだけでノードをまたぐ（`mujou/.claude/skills/pc-alm/SKILL.md` の式から変えない）
- ノードの分け方: `StackChunker`（パラメタなし）→ Enc1..Enc4（`Segmented(stage)`）→ `PackedJEPAHead`（予測器・射影・SIGReg）。target と context を 1 つの sample の区間に並べ、attention を区間の中に閉じるので、view どうしは同じノードの中でもまじらない。区間の長さは設定から決まる定数で、メッセージから何も読み戻さない
- 局所性: ヘッドの `torch.autograd.grad` は自分の detach した leaf から（今と同じ）。`StackChunker` は host の書き込みだけを読む。ノードをまたぐ `.backward()`・`.grad`・グローバルなバリアを足さない（`mujou/.claude/skills/locality-and-async/SKILL.md`）
- posthoc は run の外。エンコーダは checkpoint から読んで止め、デコーダ・ridge・PCA は mujou の network に入らない

## mujou に足すもの

なし。`pc.Layer` は `nn.Module` を 1 つ受け取るので、`Segmented` で足りる。速さのために「入力と $\theta$ が変わらない update では線形化を使い回して vjp だけを計算する」最適化を mujou に入れる案はあるが、M17 の範囲に入れない（想定リスク）。

## 置き場所

- `exp/video/`: 上のスキーマ・データ・ViT・ノード・損失・評価・posthoc・preset・study（M17 の実験に閉じる）
- `exp/lib/`: `linear_dyn`（ridge の線形の動力学）と `pca` は実験に閉じない道具なので `exp/lib/linear_dyn.py`・`exp/lib/pca.py` に置く（torch だけに依存）。`restore` は ViT に依存するので `exp/video/measure/restore.py`
- `tools/tb/`: TensorBoard の書き出し
- mujou: なし

## preset と study

- `conf/vision/vit_m17.yaml`: `dim 256`、`heads 4`、`depth 8`、`nodes 4`、`pred_depth 2`、`patch 16`
- `conf/experiment/m17/wt_n16.yaml`（主）: `program vision`、`data.tours = {data_version: <新>, held_out: Zurich, size: 128, fps: 30.0, advance: 3}`、`data.streams`（確認事項 1、暫定 32）、`vision = {layout: packed, frames: 16, tubelet: 2, stride: 3, views: 3, context_masks: [{blocks: 8, scale: 0.15}, {blocks: 2, scale: 0.7}], z_dim: null, recon: 0.0, sigreg_on: z, pixel_mean/std: 学ぶ 9 本の 500 フレームごとに測った値}`、`seconds 600`（確認事項 1）、`measure = {clips: 64, windows: 4, every: 60, steps: 10, horizons: [1, 5, 10]}`、`ctl.ckpt_every 300`（explore でも最後のチェックポイントを残す。experiment は phase の後に読まれるので上書きできる）、`optim_vision` は adamw
- `wt_n8.yaml`: `frames 8`（256 トークン）。`wt_n1.yaml`: `frames 1`、`tubelet 1`（64 トークン）、`stride` は使わない
- `vision.recon` は 3 つとも 0 固定。preset のテスト（`tests/video/schema/test_presets.py`）で、m17 の preset の `recon == 0` を確かめる
- `exp/video/studies/m17_wt_bp.toml`: `name m17-wt-bp-v1`、`overrides ["+experiment=m17/wt_n16", "learner=bp", "phase=explore"]`、`seeds [100, 101]`、`objective_from "posthoc"`、`[study.posthoc] ckpts "last"`、`res any_gpu`（4060 に載るかは手順 4 で確かめ、載らなければ `4090`）。`[space]`:
  - `vision.frames` {8, 16}、`vision.stride` {2, 3, 6}
  - `vision.dim` {192, 256, 384}、`vision.z_dim` {128, 256, 512}
  - `vision.context_masks.0.scale` [0.05, 0.25]、`vision.context_masks.1.scale` [0.3, 0.8]
  - `vision.sigreg_on` {z, both}（射影ありで tokens は拒むので外す）、`vision.sigreg` [0.01, 0.9] log、`vision.sg` {false, true}
  - `vision.pred_loss` {l2, l1}、`vision.pred_depth` {2, 4}
  - `optim_vision.lr` [3e-5, 1e-3] log、`optim_vision.warmup` {0, 60, 300}
  - `[[enqueue]]`: preset のまま（$N=16$、$\Delta=3$、幅 256、$D$ 256、V-JEPA のマスク）
  - hydra の list の中の override（`context_masks.0.scale`）が compose で通るかを最初に確かめる。通らなければ `MaskConf` を `mask_short`・`mask_long` の 2 つの field にする

## 実装ステップ

1. データ: 10 本を `prepare --fps 30 --size 128` で作る（4 台に `just fleet x` で配分、出力は NAS の作業ディレクトリ）→ `finalize` → 各ホストの `.fleet/data/<version>/` へ写す（76 GB × 4。先に `fleet doctor` で空きを確かめる）。学ぶ 9 本の画素の平均・標準偏差を測る
2. スキーマ（`layout`・`tubelet`・`stride`・`z_dim`・`MaskConf`・`pred_loss`・`advance`・`steps`）と検査
3. `patchify_tubelets`・`sincos_3d`・`Block`/`Stage`/`Encoder` の `segments`・`Segmented`・`PackedEmbed`・`PackedPredictor`・射影・`build_vision` の packed
4. `multiblock_masks`・`expected_keep`・`pack_views`
5. `jepa_loss_packed`
6. `StackedTourFrames`・`eval_windows`
7. `StackChunker`・`PackedJEPAHead`・`JEPABP` の packed・`nets` の packed。`_Online` は packed で `StackedTourFrames` を使い、ホストは run frame ごとに積みを書く
8. `Measures` の packed の行
9. `exp/lib/pca.py`・`exp/lib/linear_dyn.py`・`measure/restore.py`
10. posthoc の packed（単位 0 の物差し、チェックポイント、画像、`ckpts` 設定）と `task=baselines`
11. preset（`vit_m17`、`m17/wt_n16`・`wt_n8`・`wt_n1`）
12. `tools/sweep` の `objective_from = "posthoc"` と followup
13. `tools/tb`（export・view・just のレシピ）
14. `just run` と `just test-sweep`、tools/tb のテスト。ここまでで commit を関心ごとに分ける
15. 物差し: `task=baselines` を $N=16$・8・1 の preset で（PCA は $D=256$。study の $D$ 128・512 も）。完了の条件の閾値を results に書く
16. 試し（BP、seed 100、`wt_n16`）: 崩壊しないか、`<t>.png` が平均の画像より細部を描くか。`task=bench`（PC-ALM と BP、$N$ 16・8、B 128・64・32 の 4090 での fps、空いた GPU で 3〜5 秒の仮想時間）。下の見積もりと比べ、`data.streams` と `seconds` を確定する（確認事項 1）
17. study → 確認（PC-ALM と BP、seed 0/1/2、test、posthoc は all）→ `results/m17.md`・TensorBoard のビュー・`report.html`（skill `experiment-report`、Walking Tours の画像は deface でぼかしてから）

## テスト観点

ミラー配置（`tests/video/...`、`tests/lib/...`、`tools/tb/tests/`）。

- 正常系:
  - `patchify_tubelets(x, p, tubelet=N)` が今の `patchify(x, p)` と一致する（$\tau = N$ は今の tube）
  - `patchify_tubelets(x, p, 1)` の時間の slot $s$ のトークンが、フレーム $s$ だけの `patchify(x[:, s:s+1], p)` と一致する
  - `sincos_3d`: 時間の部分は同じ slot の中で一定、縦・横の部分は `sincos_2d` と同じ式、全トークンの行が互いに異なる
  - `Block(segments=(a, b))` の出力の前の $a$ 個が、前の $a$ 個だけを `Block` に通したものと一致する（block 対角。context が target を変えない）
  - packed の target view の $z$ が `VisionModules.encode`（target だけ）と一致する（rtol 1e-6）
  - `multiblock_masks`: 見える数がちょうど `keep`、乱数の generator が同じなら同じ mask
  - `pack_views`: target の T 行が全トークンを id 順に持つ、context の行が mask の見える位置だけを id 順に持つ、`y` が mask と一致
  - `StackedTourFrames(frames=1, stride=1, advance=1)` の画像（u8/255）が `TourFrames.next()` と一致し、`reset` も一致する
  - `StackedTourFrames` の積みの index が $g - j\Delta$（小さな合成のデータで手計算）
  - `eval_windows(stride=1, steps=0)` が `eval_clips(length=frames)` とバイト単位で一致
- 数値的オラクル:
  - `linear_dyn.fit(alpha=0)` が、1 の列を足した最小二乗の正規方程式（`torch.linalg.lstsq`）の解と一致する
  - 雑音のない合成の線形の系 $z_{k+1} = A z_k + b$（スペクトル半径 < 1）で、`fit` が $A$・$b$ を 1e-8 で回復し、`dyn_h` が全ての $h$ で 0（1e-10）
  - $A = I$、$b = 0$ で `dyn_h/copy_h` がちょうど 1
  - `dyn_h` が $z \mapsto c z$（$c \ne 0$）で変わらない（尺度を縮めても小さくならない）
  - `fit_pca` の主成分の張る部分空間が、小さなデータでの `torch.linalg.svd` の上位 $d$ と一致し（射影行列の差 < 1e-5）、再構成の二乗誤差の和が捨てた特異値の二乗の和に一致する（Eckart–Young）
  - `jepa_loss_packed` の SIGReg・全体の式が、同じ数値を手で並べた `jepa_loss` の式と一致する（`l2` で $M_v$ の平均）
  - `PackedJEPAHead` の credit が、エネルギー $E = B L$ の `torch.autograd.grad` と一致（今の `test_nodes.py` 186 行の型）
  - PC-ALM の固定点の credit が BP の勾配と一致する（今の `test_nodes.py` 217 行の型を packed で）
- 局所性:
  - packed の PC-ALM のコード（`StackChunker`・`PackedJEPAHead`・`Segmented`）にノードをまたぐ `.backward()`・`.grad` が無い（今の `test_nodes.py` の検査を packed にも）
  - `restore.restore` の出力が、隠した tubelet の画素を書き換えても変わらない（context と予測器だけを使う）。見える位置の画素は、context のトークンを `dec` で描いたものと一致
- 異常系:
  - `layout="packed"` で `tubelet` が `frames` を割り切らない、`views != 1 + len(context_masks)`、`z_pool != "mean"`、`dim % 16 != 0`、`z_dim > 1024`、射影ありで `sigreg_on == "tokens"` を `ValueError`
  - `eval_windows` で slot が足りない、`stride < 1`
  - m17 の preset で `vision.recon != 0` を拒む（preset のテスト）
- エッジケース:
  - $N=1$・$\tau=1$（64 トークン）で全経路が動く
  - 動画の境にかかる積み（`ready` が False、その stream の credit が 0）
  - `expected_keep` が 0 や T にならない（scale の範囲の端）
  - posthoc の `ckpts="last"`、単位 0 から再開
  - tools/tb: 小さな偽の run ディレクトリから event が書け、scalar の tag と step が仕様どおり（`tensorboard` の `EventAccumulator` で読み戻す）
- slow / cuda: bench と packed の小さな run（数秒）は `slow`

## 再現性と記録

- seed: 探索 100/101、確認 0/1/2。Chunker・Head・decoder は既存の `CHUNKER_SEED`・`HEAD_SEED`・`DECODER_SEED`、PCA は `PCA_SEED = 4`（run の seed に足す）
- results/m17.md に書くもの: data_version と prep の ffmpeg の版、pixel_mean/std、snapshot と mujou の SHA、preset と override、`data.streams`・`seconds`、物差しの表（PCA・学習前・平均・写し、$D$ ごと）、閾値（回す前）、study の上位、確認の表（`recon`・`dyn_{1,5,10}`・`dyn_h/copy_h`・`img_dyn_h`・`restore_*`・`rank`・`z_std_min`・`z_spread`）、$N$・$\Delta$ ごとの 2 つの誤差、Pareto の散布図、見た PNG のファイル名、計算時間、採らなかった V-JEPA 2 の変更
- checkpoint・ログ・`tb/` は git に入れない

## 見積もり（メモリと時間）

前提: M15 の玉 $N=1$・B=128 の PC-ALM 5.0 fps・BP 35 fps（空いた 4090、TF32 なし。`results/m15.md` の計算時間）から、実効 17.5 TFLOPS（PC）・15 TFLOPS（BP）の計算律速とみて外挿する。PC は 1 run frame に 16.7 update、各 update で各ノードが forward + vjp（forward の約 2.5 倍）、ヘッドはトークンが変わったときだけ。BP は 1 run frame に forward・backward・target の forward（約 3.5 倍）。小さい B では GPU の効率が落ちるので、B=32 の値は楽観側。

1 stream あたりの forward（幅 256、8 block、予測器 2 block、context 2 view で V-JEPA のマスク: 短い範囲 約 27%・長い範囲 約 9% を残す）:

| $N$ | トークン（target + context） | エンコーダ GMAC | 予測器 GMAC |
| -- | -- | -- | -- |
| 16 | 512 + 138 + 48 = 698 | 9.9 | 3.8 |
| 8 | 256 + 69 + 24 = 349 | 4.7 | 1.7 |
| 1 | 64 + 17 + 6 = 87 | 1.1 | 0.4 |
| M15 の玉（参考） | 2 view × 64 | 0.31 | 0.05 |

1 run frame の時間と、600 秒（18000 run frame）・1800 秒（54000）の所要（4090 1 枚を占める）:

| 設定 | PC s/frame | PC 600 s | PC 1800 s | BP s/frame | BP 600 s |
| -- | -- | -- | -- | -- | -- |
| $N=16$、B=128 | 6.8 | 34 h | 102 h | 0.82 | 4.1 h |
| $N=16$、B=32 | 1.7 | 8.5 h | 26 h | 0.21 | 1.0 h |
| $N=8$、B=128 | 3.2 | 16 h | 48 h | 0.38 | 1.9 h |
| $N=8$、B=32 | 0.81 | 4.1 h | 12 h | 0.10 | 0.5 h |
| $N=1$、B=128 | 0.76 | 3.8 h | 11 h | 0.09 | 0.5 h |

メモリ（BP の活性、1 トークン 1 block あたり約 $15d$ 個の float と attention の確率）: $N=16$・B=128 で約 22 GB（エンコーダ 16 GB、予測器 6 GB、入力 0.6 GB）で、24 GB の 4090 に余裕が無い。B=32 で約 6 GB、4060（8 GB）にも載る見込み。PC-ALM も各ノードの vjp と CUDA graph の pool の和でほぼ同じ。

読み: B=128 では $N$ によらず PC-ALM の確認が 7 時間に収まらない。ユーザーの規則（$N=8$ に落とす）だけでは足りず、B か run の長さも下げる必要がある。暫定は B=32・600 秒で、$N=16$ が約 8.5 時間（規則で $N=8$ へ、約 4 時間）。手順 16 の bench の実測で確定する。速くする手段（採るかは bench の後）: attention を `scaled_dot_product_attention` にする（メモリ。CUDA で決定的かを確かめる）、mujou の `pc.Layer` で線形化を使い回す（vjp の forward を省く、約 2 倍）。

## 想定リスク・トレードオフ

- 見積もりは計算律速の外挿で、小さい B では効率が落ちる。bench を study の前に必ず回す
- token を落とす形（packed）は M15 の「隠した patch を 0 にして通す」形と別物で、M15 の結果とは直接比べられない。M15/M16 の preset は `layout="pixels"` のまま変えない
- `multiblock_masks` の数をそろえる調整で、ブロックの形が少し崩れる（見える数が平均より多い mask では、乱数の patch が追加で隠れる）
- packed の sequence にした分、1 ノードの入力は 1 本の `h` のまま。区間の長さは設定の定数なので、`expected_keep` の結果が変わると checkpoint の互換が切れる（`group` の hash に入るので同じ run とは扱われない）
- 30 fps の動画を 1 run frame に 3 フレーム進める（下の確認事項 2）。M15 の Walking Tours と同じ形で、時定数の意味（1 入力に 16.7 update）を保つ
- study は posthoc を待って tell するので ASHA が効かない。1 trial = 2 seed × run + posthoc。run が短い（BP 1 時間程度）前提で、32 trial に 4 台で 1〜2 日
- 4060 は 8 GB で、$N=16$・B=32 の BP と posthoc（$z$ のデコーダの 128×128 の fit）が載るかは手順 16 で確かめる
- $z$ のデコーダ（2000 step、128×128）は 1 回に約 8 分の見込み。確認の posthoc は 1 run あたり 30 分程度
- TensorBoard の画像は deface を通していない Walking Tours のフレームを含む。ビューは手元だけで見て、外に出す画像（report.html）は manifest の注記どおりぼかしてから載せる
- `thumb` の probe を M17 の行から外すので、M15 の Walking Tours の `thumb` とは比べない
- 採らなかった案: 3D-RoPE（上）、run の中でデコーダを学ぶ（ユーザーの規則に反する）、SDPA を既定にする（決定性が未確認）

## 確認事項

1. **B と run の長さ**: 見積もりでは B=128 では PC-ALM の確認が $N=8$ でも 16 時間（600 秒）〜48 時間（1800 秒）かかり、ユーザーの「7 時間を超えたら $N=8$」だけでは収まらない。暫定案: M17 全体（study と確認）を B=32・600 秒の仮想時間にし、bench で $N=16$ が 7 時間を超えれば規則どおり $N=8$ に落とす。B を study と確認でそろえるのは、SIGReg の強さが B で変わるため（`loss.jepa_loss` の注記）。LeJEPA §6.1 は batch ≥ 128 を推奨するが、§4.2.3 は batch 16 でも偏りが小さいとする
2. **「30 fps で進み、3 フレームに 1 回学ぶ」の時間の扱い**: 暫定案は、1 run frame（仮想 1/30 秒、16.7 update）に動画を 3 フレーム進め、run frame ごとに新しい積みで学ぶ（M15 の Walking Tours と同じ形）。もう 1 つの読み方（仮想時間も 30 fps で、新しい積みが 3 run frame ごと、1 つの積みに 50 update）では PC-ALM の時間が 3 倍になる。`time.frame` と `data.tours.advance` の組で後から切り替えられる
3. **探索の目的を posthoc から取るための掃引の拡張**: 暫定案は `tools/sweep` に `objective_from = "posthoc"` を足し、trial の run の後に posthoc の job を入れて、その値で tell する（ASHA なし）。代わりに study をやめ、`fleet submit --sweep` の小さな格子と posthoc の表で選ぶ道もある

## 参照

- 合意した計画: `exp/video/MILESTONES.md` の「## M17」、`exp/video/PLAN.md` 12 節、`memory/project_m17_wt_encoder.md`、`memory/project_decoder_for_debug.md`、`memory/feedback_realtime_speed.md`、`memory/feedback_metrics_need_qualitative_check.md`
- 今のコード: `exp/video/jepa/vit.py`（`patchify` 54 行、`Embed`、`Predictor`、`build_vision`）、`jepa/masks.py`（`block_masks`・`make_views`）、`jepa/loss.py`（`prediction_errors`・`jepa_loss`）、`jepa/nodes.py`（`Chunker`・`JEPAHead`）、`jepa/bp.py`、`jepa/nets.py`、`data/walking_tours.py`（`prepare`・`eval_clips`）、`data/frames.py`（`TourFrames`）、`measure/posthoc.py`（`decoder`・`fit_decoder`・`_TourChunks`・`Decoding`）、`programs/vision.py`（`Measures`・`vision_objective`）、`programs/posthoc.py`、`exp/lib/sigreg.py`、`exp/lib/host.py`、`tools/sweep/README.md`、`exp/lib/fleet/README.md`
- M15・M16 の実装と数字: `exp/video/results/m15.md`（計算時間の表）、`m16.md`（速さの節）
- 原典: V-JEPA（arXiv:2404.08471）§3.1〜3.4・§4.3・§4.4・App. B・C・E.4。V-JEPA 2（arXiv:2506.09985）§2.1〜2.4・Table 9・12。LeJEPA（arXiv:2511.08544）§4.2・4.3・5.1・6.1
- 理論: `mujou/.claude/skills/pc-alm/SKILL.md`、`mujou/.claude/skills/locality-and-async/SKILL.md`
- 規約: skill `refactor-conventions`、`testing-strategy`
