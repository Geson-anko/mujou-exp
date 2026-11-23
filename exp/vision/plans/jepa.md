# JEPA: 単一ストリームから SIGReg で JEPA を学ぶ（計画、2026-11-23）

V17 で VAE の検証を締めくくり、検証する対象を JEPA に移す（[memory/project_v17_to_jepa.md](../../../memory/project_v17_to_jepa.md)）。`exp/video` の M17 で決着しなかった JEPA の問いに、VAE と同じ順序で答えを出す。番号は `exp/vision` の V18 から続ける。

## 背景

- **VAE は、普通の学習を先に確かめる順序で決着した。** V16 で、BP と i.i.d. なミニバッチなら同じモデルが学べる（span 0.164）ことを確かめた。そのうえで V17 で、ストリームの PC-ALM でも長く学べば下がる（480 万 stack で span 0.211）ことを示した
- **M17 の JEPA は、この順序を踏んでいなかった。** BP と i.i.d. なミニバッチで JEPA + SIGReg が学べることを確かめないまま、B = 1 のストリームで崩壊と戦っていた。そのため、学べない原因が (1) 実装・モデル・評価、(2) 学習の設定、(3) ストリームの相関のどれなのかを分けられなかった
- **評価にも穴があった。** M17 の再構成の基準は、tubelet ごとの平均の色だけの符号でも満たせた。V16 では、後から学ぶデコーダの `recon_blur` が頭打ちになり、符号の差を映さなかった
- **ユーザーの仮説**: SIGReg は、標本が i.i.d. であることを前提にした検定統計量である。ストリームでは時間相関のためにデータポイントを独立に扱えず、前提が崩れる。一方で、時間相関はそれ自体が手がかりにもなる

## 全体の流れ

段ごとに 1 つの要素だけ変える。

| 段 | 学習則 | データの並べ方 | 正則化 | 問い |
| -- | -- | -- | -- | -- |
| V18 | BP + AdamW | i.i.d. なランダムの stack のミニバッチ | 標準の SIGReg | モデル・目的・評価は正しいか。JEPA の上限の目安（V16 にあたる） |
| V19 | BP | 単一ストリーム（B = 1、長い run は c = 10） | 標準の SIGReg と FIFO のバッファ | 相関で何が壊れるかを、V18 と同じ物差しで再現して測る |
| V20 | BP | 単一ストリーム | 時間相関の SIGReg | 仮説の検証。V19 との差が、相関の扱いの効果になる |
| V21 | PC-ALM | 単一ストリーム | V20 で良かったもの | システムの学習則でも同じになるか |
| V22 | PC-ALM | 長いストリーム（c = 10、数百万 stack） | 同上 | V17 にあたる長い run |

各段の前に、判定の基準と予想を `exp/vision/README.md` に書いてから回す。前の段の判定が外れたら、次の段に進まず原因を調べる。

## 共通の設計

- **モデル**: V14・V16・V17 と同じ ViT-S（幅 1024、μP、patch 16、64×64、8 枚の stack を early fusion）。VAE の頭（`2D` への写像）を除いたエンコーダを使い、VAE と同じ物差しで比べる。トークンは 4×4 = 16 個
- **表現**: トークンごとに $d$ = 64 次元に写し、平らにした列 $z \in \mathbb{R}^{D}$（$D$ = 1024）とする（2026-11-13 の決定。$D$ が十分小さいことだけが制約）
- **目的**: LeJEPA の形にする。エンコーダは 1 つで、EMA のターゲットも stop-gradient も入れない。予測器が、見えるトークンから隠した位置のトークンの埋め込みを当てる。損失は $(1 - \lambda) L_{\mathrm{pred}} + \lambda\, \mathrm{SIGReg}$
- **マスク**: 空間のブロックを stack の全フレームに同じ位置で掛ける（V-JEPA の tube）。early fusion なので、課題は空間の補完だけで、未来のフレームの補完は含まない
- **SIGReg**: データポイント（stack）ごとに $z$ にかける（トークンごとにはかけない。位置だけの解が安く通るため、2026-11-13 に棄却）。実装は `exp/lib/sigreg.py` の Epps–Pulley を使う
- **バッファ**: 過去の $z$ のバッファは FIFO にする（2026-11-23、ユーザーの決定）。これまでのランダムな置き換え（`insert_`）は、各行の生存時間が確率的になる。FIFO なら、バッファは直近の決まった時間の幅をちょうど覆う

## V18: BP と i.i.d. なミニバッチで学べるか（詳細設計、2026-11-23）

run の一覧・判定・長い run の規則・予想・反証は [README の「V18」の節](../README.md)（回す前に決めたこと）に書いた。ここには、実装できる粒度の設計と、目的の式の自明な解の導出を書く。

### ユーザーが決めたこと

- early fusion のまま（8 枚 × 3 チャネル = 24 チャネル、64×64、patch 16 → 4×4 = 16 トークン）。patch 8 にはしない
- **マスク率に注意する**: トークンが 16 個しかないので、隠すトークンの数をちょうど $k$ 個に固定する。マスク率 $r \in \{0.25, 0.5, 0.75\}$（$k$ = 4、8、12）を掃引の軸にする。$r$ = 0.75 でも見えるトークンが 4 個残る
- batch は 128。SIGReg は batch の中の標本だけで計算する（バッファは V19 から）
- 掃引: 学習率 $\{10^{-4}, 10^{-3}\}$ × $\lambda$ $\{0.005, 0.05, 0.5\}$ × $r$ の 18 本、seed 0、6000 step。長い run は 60000 step

### ファイルの分け方

| ファイル | 中身 |
| -- | -- |
| `exp/vision/jepa.py` | マスクの族と引き方（`mask_family`、`draw_masks`）、モジュール（`JEPAModules`: エンコーダ・射影・予測器、`build_jepa`、`jepa_mup_lr`）、損失（`jepa_terms`） |
| `exp/vision/jepa_iid.py` | 学習。`exp/vision/iid.py` の `Units` と `lr_at` を import して使う |
| `exp/vision/jepa_eval.py` | 評価（run の checkpoint と物差し）。`stream_eval`・`eval` の部品を使う |
| `tests/vision/test_jepa.py`、`test_jepa_iid.py`、`test_jepa_eval.py` | 下の「実装のテスト」 |

`exp/video/jepa` は import しない。M17 の `block_masks` と `multiblock_masks` は使わない。`block_masks` は平面の上に一様に置くので、中央の位置ほど多く隠れる。`multiblock_masks` は、ちょうどの数にするために見える patch をランダムに足して隠すので、ブロックの形が崩れる。移すのは、矩形を格子に塗る考え方（`paint`）だけである。

### エンコーダと表現

- **エンコーダ**: `build_vae(conf, seed=conf.seed)` と同じ手順・同じ seed で ViT（`_vit_parts`）を作り、node 4 の `head` だけを使わない（最後の `LayerNorm` は使う）。head は node 4 の最後に引かれるので、ブロックの初期値は V16 の VAE（`b2006f0a-s0` の config、seed 0）とちょうど一致する（テストで確かめる）。config は V16 と同じ `config.yaml` を読む。デコーダと $\ln\sigma$ は作らない
- **射影**: $P$ = `Linear(1024 → 64)`、全位置で共有。トークン $h_p$（最後の `LayerNorm` の後）から $z_p = P h_p \in \mathbb{R}^{64}$。$z = (z_1, \dots, z_{16}) \in \mathbb{R}^{1024}$（位置は行優先）
- **標的側**: stack の 16 トークンすべてをエンコーダに通して $z$ を作る。stop-gradient も EMA もない
- **文脈側**: 同じエンコーダに、見える $16 - k$ 個のトークンだけを通す。パッチの埋め込みに位置の符号を足した後で、隠したトークンを落とす（V-JEPA と同じ）。注意は見えるトークンの中だけで働くので、隠した patch の画素は文脈側の出力に影響しない（テストで確かめる）。$k$ は batch の中で同じなので、形は `(B, 16 - k, 1024)` にそろう
- **予測器**: 幅 256、4 head（`HEAD_DIM` = 64）、pre-norm の `Block` を 4 段（エンコーダと同じ段数）。V-JEPA の予測器と同じ形で、次の順に計算する
  1. 文脈側の出力（幅 1024）を `pred_embed` = `Linear(1024 → 256)` で写し、その位置の符号 `sincos_2d((4, 4), 256)` を足す
  2. 隠した位置には、学習する 1 本のベクトル（mask token、256 次元）にその位置の符号を足したものを置く
  3. 16 個を `Block` 4 段に通し、`LayerNorm` の後、`pred_out` = `Linear(256 → 64)` で写す。隠した位置の出力を $\hat z_p$ とする
- 予測器の大きさは掃引しない。下の導出では、予測器が弱いほど健全な解の予測の損失 $\ell_H$ が大きくなり、stack の中で一定の解が有利になる。そこで段数はエンコーダにそろえ、弱くはしない

### μP での役割

エンコーダは VAE と同じ規則（`mup_roles`・`mup_lr`・`_mup_init_`、$k_w$ = 1024 / 256 = 4）。新しいパラメータは次のとおり。予測器の幅は 256（= `BASE_WIDTH`）に固定し、エンコーダの幅とともには変えない。

| パラメータ | 役割 | 学習率の倍率 | 初期値 |
| -- | -- | -- | -- |
| エンコーダ（パッチの埋め込み・ブロック・`LayerNorm`） | VAE と同じ（input・hidden・vector） | `mup_lr` と同じ | `build_vae` と同じ |
| 射影 $P$（幅 1024 → 64） | output（VAE の head と同じ、幅から固定の大きさへ） | $k_w^{-1}$ = 1/4 | LeCun（std $1/\sqrt{1024}$）× $k_w^{-1/2}$、bias 0 |
| `pred_embed`（幅 1024 → 256） | output（エンコーダの幅から固定の大きさへ） | $k_w^{-1}$ = 1/4 | 同上 |
| 予測器のブロック（256 → 256） | hidden（幅 256 なので $k$ = 1） | 1 | `_init_vit` の std 0.02 |
| `pred_out`（256 → 64） | output（$k$ = 1） | 1 | LeCun |
| mask token・bias・`LayerNorm` | vector | 1 | mask token は std 0.02 の truncated normal、他は `_init_vit` |

### マスク

- **族 $F_k$**: 4×4 の格子をトーラス（上下・左右の端をつないだもの）とみる。高さと幅が 1〜4 の矩形を 1 つ、または 2 つ置いた和集合のうち、ちょうど $k$ 個の位置を覆うものを、重複を除いた集合として列挙する。$|F_4|$ = 632、$|F_8|$ = 436、$|F_{12}|$ = 232（うち矩形 1 つのものは 24、8、8）
- **引き方**: 各 stack に、$F_k$ から一様に 1 つ選ぶ（batch の中で独立、`seed + MASK_SEED` の生成器）。early fusion なので、同じ位置が stack の 8 枚すべてで隠れる（V-JEPA の tube にあたる）
- **位置ごとに一様**: $F_k$ はトーラスの平行移動で閉じているので、族の上の一様分布では、各位置が隠される確率はちょうど $k/16$ になる。族の上で数えると、各位置を覆う要素は $k$ = 4 で 158、8 で 218、12 で 174 個（どれも $k|F_k|/16$）
- **トーラスにした理由**: 平面の上に一様に置くと、中央の位置ほど多く隠れる。代わりに、端をまたぐブロックは、平面では反対側の端にある 2 つの矩形になる
- **形の分布**（トーラスの上でつながった成分の数）: $k$ = 4 は 1 成分 280・2 成分 352、$k$ = 8 は 392・44、$k$ = 12 は 232・0。テストで、この数を別の書き方の総当たりと照合する
- **マスク率と形の交絡**: 2 成分の割合は、$k$ = 4 で 56%（1×1 の孤立したセルを含む）、8 で 10%、12 で 0% である。そのため $r$ の効果には、隠したブロックがまとまっているかの効果が混ざる。V18 では分けない（$r$ ごとに族を変えずに一様に引くことを優先した）。$r$ の差を読むときは、この交絡を併記する

### 損失

$$
L = (1 - \lambda)\, L_{\mathrm{pred}} + \lambda S, \qquad
L_{\mathrm{pred}} = \frac{1}{B k d} \sum_{n=1}^{B} \sum_{p \in M_n} \lVert \hat z_{n,p} - z_{n,p} \rVert^2
$$

- $M_n$ は stack $n$ の隠した位置（ちょうど $k$ 個）、$d$ = 64。$L_{\mathrm{pred}}$ は座標あたりの二乗誤差で、LeJEPA と同じ尺度である。SIGReg が満たされると $z$ の座標の分散は 1 なので、位置ごとの平均で当てる自明な予測の $L_{\mathrm{pred}}$ は約 1 になる
- $S$ = `exp.lib.sigreg.sigreg(z, z[:0], A)`。batch の $N = B$ = 128 個の $z \in \mathbb{R}^{1024}$ の Epps–Pulley の統計量（`T_POINTS` の 17 点）を、$M$ = 1024 本の方向 $A$ で平均したもの。$A$ は step ごとに `seed + DIRECTION_SEED` の生成器から引き直す（標準正規を引いて長さ 1 にする。batch の中では同じ方向）
- 統計量は $N$ 倍の形（LeJEPA と同じ）で、$z \sim N(0, I)$ なら $N$ によらず約 1.05 になる

### 自明な解の導出（$N$ = 128、$D$ = 1024）

統計量の期待値（$t$ は全実数で積分。台形則の 17 点でもほぼ同じ値になる）:

- **健全な解** $z \sim N(0, I_D)$: $E[N\lvert\hat\varphi - \varphi\rvert^2] = 1 - e^{-t^2}$ なので、$E[S] = \int (1 - e^{-t^2})\, e^{-t^2/2}\, dt = \sqrt{2\pi} - \sqrt{2\pi/3} = 1.059$。$N$ によらない
- **定数** $z_n = c$: 方向 $a$ への射影は定数 $s = a^\top c$ で、$s = 0$ のとき最小。$S = N \int (1 - e^{-t^2/2})^2 e^{-t^2/2}\, dt = N(\sqrt{2\pi} - 2\sqrt{\pi} + \sqrt{2\pi/3}) = 0.409 N$ = 52.3
- **共分散 $\Sigma$（$\mathrm{tr}\,\Sigma = D$）のガウス**: 射影の分散 $a^\top \Sigma a$ が方向によって揺れる分だけ罰を受ける。揺れが小さいとして展開すると、超過は $\frac{N}{4} \int t^4 e^{-3t^2/2}\, dt \cdot \mathrm{Var}_a(a^\top \Sigma a) \approx 0.241\, N\, (1/r_2 - 1/D)$。$r_2 = D^2 / \mathrm{tr}\,\Sigma^2$（participation ratio）
- **数値で確かめた値**（`epps_pulley`、$M$ = 1024、$N$ = 128、方向を引き直した平均）: 健全 1.05、定数 52.3、stack の中で一定 1.55、$\Sigma$ のランク 1 で 22.4、16 で 3.4、64 で 1.38、128 で 1.32

健全な解の予測の損失を $\ell_H$ とする（座標あたり、0〜1 程度）。3 つの自明な解は次のとおり。

1. **定数**: $L_{\mathrm{pred}} = 0$、$S$ の超過は 51.3。目的の式が健全な解を選ぶのは $\lambda > \ell_H / (\ell_H + 51.3)$ のとき。$\ell_H \le 1$ なら $\lambda > 0.019$ で排除される。**$\lambda$ = 0.005 では、$\ell_H > 0.26$ なら排除されない**
2. **位置だけの関数** $z_{n,p} = g(p)$: stack の単位で見ると、$z_n$ は入力によらない定数なので、1 と同じ（超過 51.3）
3. **stack の中で一定** $z_{n,p} = h(x_n)$、$h \sim N(0, I_{64})$: $\Sigma = (\mathbf{1}\mathbf{1}^\top) \otimes I_{64}$ で $r_2$ = 64、超過は約 0.45（数値では 0.50）。$h$ を、どの見える部分からも推定できる量（場面全体の明るさ・色・種類）にすれば $L_{\mathrm{pred}} \approx 0$ になる。排除されるのは $\lambda > \ell_H / (\ell_H + 0.50)$ のときで、$\ell_H$ = 0.3、0.5、0.8 では 0.38、0.50、0.62。**$\lambda \in \{0.005, 0.05\}$ では排除されない。$\lambda$ = 0.5 でも、$\ell_H > 0.5$ なら排除されない**
   - **この解は族をなす**: 位置ごとの可逆な線形写像 $A_p$（直交行列 $R_p$ など）を掛けた $z_{n,p} = A_p h(x_n)$ でも、$L_{\mathrm{pred}} = 0$ のままにできる。予測器は位置を知っているので、$A_q A_p^{-1}$ で写せばよい。$A_p$ が直交なら $\Sigma$ のスペクトルは変わらず（$r_2$ = 64）、SIGReg の超過も同じになる。共有の $P$ の前のトークン $h_p$ は位置の符号を持つので、エンコーダはこの形を作れる。数値（$N$ = 128、$D$ = 1024、$M$ = 1024）: 恒等の版は $S$ = 1.62・`rho_in` = 0.000・実効ランク 49.5、$R_p$ で回した版は $S$ = 1.63・`rho_in` = 0.932・実効ランク 49.5。独立なガウスでも `rho_in` は 0.937 である。したがって `rho_in` は恒等の版しか捕まえられない。評価では、族全体で 1 になる、基底によらない量 `token_r2`（下）で測る

一般に、エンコーダは、見える部分から予測できない位置ごとの内容を捨てて $r_2$ を下げると、$L_{\mathrm{pred}}$ を下げる代わりに約 $0.241 N\, \Delta(1/r_2)$ の罰を払う。罰は $N$ に比例するので、$N$ = 128 の batch の SIGReg は、$r_2$ を $N$ 程度まで下げることをほとんど罰しない。$N$ = 1024 なら、stack の中で一定の解の超過は 3.46（数値）で、$\ell_H$ = 0.5 なら $\lambda > 0.13$ で排除される。V18 では $N$ を変えない（バッファで標本を増やすのは V19）。この解は目的の式では排除されないので、評価で直接測る（下の `token_r2`）。崩壊が起きたら、まずこの導出と照らし合わせる（[memory/feedback_simple_fundamental.md](../../../memory/feedback_simple_fundamental.md)）。

この結論は 3 の解に限らない。「トークンごとに違う線形の読み出しで、どの見える部分からも推定できる約 128 次元の大域的な潜在を表す」符号（$r_2 \approx N$ = 128）の超過は、約 $0.241 \cdot 128 \cdot (1/128 - 1/1024) \approx 0.21$ しかない。この符号は $L_{\mathrm{pred}} \approx 0$ と両立し、`token_r2` ≈ 1、`rho_in` ≈ 15/16 になる。さらに、位置ごとの内容を一部だけ捨てた符号は、`token_r2` が 1 より小さいまま、同じ方向に寄る。したがって、崩壊の極端な形（3 の族）は `token_r2` で捕まえられるが、どこまで位置ごとの内容を残したかは、SIGReg の値でも `token_r2` でもなく、probe（`recon_blur`）でしか分からない。

上の比較は、目的の式の大域的な最小どうしの値の比較であって、学習の動きの議論ではない。勾配法の初期の動きでは、SIGReg は $N$ 倍の形なので 1 標本あたりの勾配が $O(1)$、$L_{\mathrm{pred}}$ は平均なので 1 標本あたり $1/B$ になる（LeJEPA と同じ尺度）。1 標本あたりの実効的な重みの比は $\lambda B / (1 - \lambda)$ で、例えば $\lambda$ = 0.005 でも約 0.64 である。どちらの解に向かって動くかは、この導出だけでは決まらない。

### 学習（`jepa_iid.py`）

- **データ**: `iid.Units(conf, root, "all", seed=conf.seed + DRAW_SEED)` から 128 stack を、一様に、重複を許して引く（V16 の `all`）。次の batch はスレッドで先に読む（V16 と同じ）
- **optimizer**: AdamW（$\beta$ = 0.9・0.999、eps 1e-8、weight decay 0）。`jepa_mup_lr` の倍率ごとに param group を分ける。clip しない。学習率は `iid.lr_at`（最初の 1000 step で線形に上げ、`--steps` で cosine で 0 に下げる）
- **乱数**: 生成器を分ける。初期値（エンコーダは seed、射影は seed + `PROJ_SEED`、予測器は seed + `PREDICTOR_SEED`）、stack（seed + `DRAW_SEED`）、マスク（seed + `MASK_SEED`）、方向（seed + `DIRECTION_SEED`）。`configure_determinism` を使う
- **checkpoint**: step $s$ の時刻は、見た stack × 0.1 秒（V16 と同じ）で、$t = 12.8 s$ 秒。短い run は step 0、1000、2000、4000、6000（$t$ = 0、12800、25600、51200、76800 秒）。長い run は 0、2000、6000、10000、20000、30000、40000、50000、60000（$t$ = 768000 秒まで）。長い run の 6000 step は、短い run の 6000 step とは学習率の予定が違うので同じではない
- **出力** `<out>/`:
  - `config.yaml`（読んだ config、`device` を書き換えたもの）
  - `jepa.json`: $r$、$k$、$\lambda$、$d$、$M$、予測器の形、学習率、steps、batch、warmup、seed たち、optimizer、族の大きさ $|F_k|$、units の数、determinism
  - `train.jsonl`: 100 step ごとの平均。`loss`、`pred`、`sigreg`、`grad_norm`（clip の前）、batch の `z_var`（座標ごとの分散の平均）と `rho_in`、`lr`、`sec`
  - `jepa.json` の `provenance`: `provenance.of_diagnosis` の記録（host の `code/<snap>` から走らせれば、その `CODE.json`。snapshot と mujou のコミットを含む）、host、コマンド行、環境
  - `ckpt/<t>.pt`（`ckpt.save_eval`、program `"jepa"`、modules は `encoder`・`proj`・`predictor`）
- 損失が有限でなくなったら止める（`FloatingPointError`、V16 と同じ）

### 評価（`jepa_eval.py`）

符号は、どれも「stack → 符号のベクトル」の関数として扱う。JEPA の符号は標的側の平らな $z$（1024 次元）。VAE の `VaeCode` の `decode`（システムの描き戻し）は使わない。

- **probe**（`stream_eval.StreamEvaluator` の部品、`--span 6000` の視点）: 学習のストリームの 0〜6000 秒の奇数番目の 10 秒の区間の約 6000 組で、後から学ぶデコーダ（4 seed × 2000 step）と線形の動力学を当てはめる。span・after・zurich（val の 1024 窓）で、`recon_blur`、`dyn_1/copy_1`、`rank` を出す
  - V16 は 1800 秒ぶん（1800 組）で当てはめた。V17 は 6000 組で、同じ学習前のモデルの zurich の `recon_blur` が 0.502 から 0.404 に下がった。当てはめる組の数が値を決めていたので、V17 と同じ 6000 秒にそろえる
  - `StreamEvaluator.probes` のうち `decode` を使う部分（絵の 2 段目）を、符号だけのものでは外せるようにする。VAE の行は変えない（`tests/vision/test_stream_eval.py` はそのまま通す）
- **JEPA の量**（zurich の val の各窓の最初の stack、$N$ = 1024、checkpoint ごと）
  - `pred`: 固定したマスク（各 stack に $F_k$ から 8 個、`EVAL_MASK_SEED`）での $L_{\mathrm{pred}}$。`pred_triv`: 同じマスクで、隠した位置を「位置ごとの $z_p$ の平均」（当てはめの組の stack で求めたもの）で当てたときの値。`pred_ratio` = `pred` / `pred_triv`。`pred_triv` は学習のストリームの平均で zurich を当てるので、ドメインのずれの分だけ大きくなる。読みの幅を見るため、zurich 自身の calib の窓の位置ごとの平均で当てた `pred_triv_calib` と `pred_ratio_calib` も並べる
  - `sigreg_zurich`: zurich の 1024 個の $z$ の $S$（固定した 1024 本の方向、`EVAL_DIRECTION_SEED`）。同じ方向・同じ $N$ での、$N(0, I)$ の標本の値（約 1.06）と定数の値（0.409 $N$ = 419）を同じ行に書く。zurich は 1 本の動画なので、健全でも $N(0, I)$ の値には届かない。並べるだけにする
  - `sigreg_heldin`: 学習の units から一様に引いた 128 stack × 16 batch の $S$ の平均（固定した方向と生成器）。学習中の $S$ と同じ尺度で、参照は 1.05 と 52.3（評価が同じ方向で計算して行に書く）
  - `token_r2`: 1 つのトークン $z_p$ から別のトークン $z_q$ を線形回帰（切片つきの ridge、$\alpha_p$ = `TOKEN_RIDGE` × $\mathrm{tr}(X_p^\top X_p)/d$）で当てたときの $R^2$ を、順序つきの組 $p \ne q$（240 組）で平均したもの。回帰は当てはめの組の stack で求め、zurich で評価する。導出の族 $z_{n,p} = A_p h(x_n)$ なら、どの可逆な $A_p$ でも 1、位置どうしが独立なら 0。K3 に使う
  - `participation`: zurich の $z$ の共分散の participation ratio $r_2 = (\mathrm{tr}\,\Sigma)^2 / \mathrm{tr}\,\Sigma^2$（超過の式の $r_2$）。標本の値は低く偏る（$N(0, I_D)$ で約 $N D/(N+D)$ = 512）。並べるだけ
  - `rho_in`: stack の中の分散の割合。$u_{n,p} = z_{n,p} - \bar z_p$（$\bar z_p$ は位置 $p$ の、stack をまたいだ平均）、$\bar u_n = \frac{1}{16} \sum_p u_{n,p}$ として、$\rho_{\mathrm{in}} = \sum_{n,p} \lVert u_{n,p} - \bar u_n \rVert^2 / \sum_{n,p} \lVert u_{n,p} \rVert^2$。stack の中で一定なら 0、位置どうしが無相関で分散が等しければ 15/16。基底に依存するので（3 の族を $R_p$ で回すと 0.93）、恒等の版の診断として並べるだけにする
  - `z_var`: 座標ごとの分散の平均（SIGReg の目標は 1）
- **絵**: probe の checkpoint ごとに `<t>-zurich.png` と `<t>-span.png`（本物の最後のフレームと、後から学ぶデコーダの seed 0 の描き戻し）
- **物差し**（`python -m exp.vision.jepa_eval --yardsticks`、1 回だけ）: 同じ probe に通す。トークンを持つもの（`init`・`color`）は `token_r2` と `rho_in` も出す
  - `init`: JEPA の step 0（seed 0。18 本とも同じ初期値）
  - `init256`: `init` の $z$ を、当てはめの組の stack での PCA で 256 次元（VAE の $\mu$ と同じ）に落としたもの。符号の次元をそろえた参照（下の「次元の偏り」）
  - `color`: パッチごと・チャネルごとの画素の平均（16 トークン × 24 チャネル = 384 次元）。学習しない符号で、M17 の `tube` にあたる
  - `pca`: 当てはめの組の stack（平らにした 24×64×64）の画素の PCA、1024 成分（`exp.lib.pca`、seed + `PCA_SEED`）。上側の参照で、並べるだけ
  - `v16`: V16 の `all`・学習率 $3 \cdot 10^{-4}$ の最後の checkpoint の $\mu$（256 次元）
  - `v17`: V17 の本番（$\tau = 10^3$、`v0-base-49b81800-s0`）の `04800.0` の $\mu$。V17 の値（zurich の `recon_blur` 0.2136、`dyn_1/copy_1` 0.768）を再現するかで、評価の経路を確かめる
- **次元の偏り**: 後から学ぶデコーダと ridge の線形動力学に入る次元は、VAE の $\mu$ が 256、`init`・JEPA・`pca` が 1024、`color` が 384 である。入力は標準化されるが、デコーダの容量と動力学の自由度は次元とともに増える。そのため VAE は `init` を下回りにくく、物差しの有効性の規則（README）では組が「使えない」と判定されやすい（保守的な方向）。run と `init` の比較は同じ 1024 次元で公平である。`init256` との差で、この偏りの大きさを見る
- **span**: 既定は `SPAN` = 6000 秒（`--span` を省いても 6000 組で当てはめる）。V16 の config の `measure.duration`（1800 秒）に黙って戻らないようにした
- **出力**: run の `posthoc_jepa/`（`config.json`、`probes.jsonl`、`jepa.jsonl`、絵）。物差しは `runs/v18-jepa/yardsticks/posthoc_jepa/`（行に `code` の名前）

### 実行

- 先に mini-00 で 100 step の試しを回し、1 step の時間・GPU メモリ・`train.jsonl` の値が有限であることを確かめる
- 短い 18 本は、std-00〜03 の 4090 と 4060 に、1 GPU に 1 本ずつ（std-03 で同居させたとき illegal memory access が出た）。1 step 約 0.1 秒の見込みで（V16 の VAE は batch 100 で 0.073 秒）、6000 step は約 10 分
- 出力は NAS の `/nas/geson/mujou-exp/runs/v18-jepa/<name>/` に置き、ckpt 以外を `./runs/v18-jepa/` に pull する。名前は `jepa-r<r>-lam<λ>-lr<lr>-s<seed>`（例 `jepa-r0.5-lam0.05-lr1e-3-s0`）、長い run は `jepa-long-r<r>-lam<λ>-lr<lr>-s<seed>`

### 実装のテスト（理論のオラクル）

- **マスク**: 族のすべての要素がちょうど $k$ 個を覆う。族の大きさ（632・436・232）、矩形 1 つの数、成分の数が、別の書き方の総当たりと一致する。トーラスの平行移動で閉じている。族の上で、各位置を覆う要素の数がちょうど $k|F_k|/16$。$10^5$ 回引いた位置ごとの頻度が、二項分布の 4σ の中にある
- **文脈側**: 隠した patch の画素を変えても、文脈側の出力がまったく同じ
- **損失**: 手で計算できる小さな例と一致する。定数の $z$ で $S$ が $0.409 N$（台形則の誤差の中）。stack の中で一定のトークンと、見えるトークンを写すだけの予測器で $L_{\mathrm{pred}} = 0$。標的側の $z$ にも $L_{\mathrm{pred}}$ の勾配が届く（stop-gradient がない）
- **初期値**: エンコーダのブロックが、同じ seed の `build_vae` のものと一致する
- **μP**: 役割と倍率が上の表のとおり。エンコーダの倍率が `mup_lr` と一致する
- **学習**: 小さな設定で 2 step 回し、同じ seed で同じ値になる（決定性）。checkpoint の時刻が $t = 12.8 s$
- **評価**: `color` の符号が手計算と一致する。`rho_in` が、stack の中で一定なら 0、位置が独立な標本なら 15/16 に近い。`token_r2` が、stack の中で一定なら 1、それを位置ごとの直交行列で回しても 1（このとき `rho_in` は 0.85 より大きい）、位置が独立なら 0、半分を共有する $z_p = (s + e_p)/\sqrt 2$ なら 1/4。`participation` が、等方的な標本で $N D/(N+D)$、直線の上で 1、stack の中で一定で約 $d$
- **予測器の順序**: ブロックを恒等にした予測器の出力が、隠した位置の昇順で $\mathrm{out}(\mathrm{LN}(\text{mask token} + \mathrm{pos}_p))$ に一致する

## V19: ストリームで標準の SIGReg

- V18 で選んだ設定のまま、データだけを単一ストリームにする
- **対照**: B とバッファを同じにしたまま、stack の順だけをシャッフルした腕を入れる。「小さい batch・オンライン学習のせい」と「相関のせい」を分けるため。並べ替えは原因の切り分けにだけ使い、並べ替えで学べることを成果にはしない（M18 のときのユーザーの判断）
- 表現の自己相関の時定数と、FIFO の窓の長さとの関係を測る。何を測るかは、時間相関の SIGReg の調査（下）の結果で決める

## V20: 時間相関の SIGReg

- 手法は、Fable の指揮で文献調査（Deep Research）を行い、いくつか考案してから決める（2026-11-23、ユーザーの依頼）。調査の報告は [memory/agents/fable/temporal-sigreg.md](../../../memory/agents/fable/temporal-sigreg.md)
- 回す前に、理論のオラクルでテストする。定常な AR 過程のガウス標本で、(1) 相関が 0 のとき標準の SIGReg に戻ること、(2) 相関があっても統計量が偏らないことを確かめる。目的の式に自明な解がないかを導出で確かめ、Codex にも反論を求める（skill `ask-codex`）

## V21・V22: PC-ALM

- V20 で良かった正則化を、ノードの中で閉じた局所的な形にする。時定数は秒で定義し、更新の式にステップ数を含めない
- BP の勾配をオラクルにしたテスト（小さなネットで一致を確かめる）を先に書く
- V22 は V17 と同じく、`data.layout=chunk data.streams=10`、optimizer の時定数は B = 1 の 10 分の 1 で回す

## コードの置き場所

- JEPA の目的（マスク・予測器・損失）は `exp/vision` に置く
- 時間相関の SIGReg と FIFO のバッファは、研究中の部品なので `exp/lib` に置く
- `exp/video/jepa` からは、テストのある部品（マスクなど）だけを移す。`exp/video` は凍結したまま、import しない

## 運用

- 1 つの V ごとに 1 PR。ブランチは `exp/<日付>/vision-v<n>-<内容>`（V18 は `exp/2026-11-23/vision-v18-jepa-iid`）
- run の投入・監視・posthoc・集計は sonnet のサブエージェントに任せる。std-00〜03 で並列に回し、mini-00 は試しだけに使う

## 未決定

- 時間相関の SIGReg の手法（調査の後に決める）
- V18 の後: stack の中で一定の解（「V18」の導出）が実際に起きたとき、標本数 $N$（batch・バッファ）を軸にするか

決まったこと（2026-11-23）: patch 16 の 4×4 のまま（early fusion、ユーザー）。マスク率は $\{0.25, 0.5, 0.75\}$ を掃引し、隠す数はちょうど $k$ 個（ユーザー）。予測器は幅 256・4 段に固定。V18 の batch は 128。
