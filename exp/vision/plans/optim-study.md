# ストリームで学べる optimizer を探す（V11〜V14、2026-11-19）

2026-11-19 にユーザーと決めた計画。V0〜V10 で、学習の全体像が見えてきた。RMSProp $\tau = 10^4$ と AdamW $\tau = 10^5$ は学べる。SGD のデコーダは $z$ を使わず、重みでいまの場面を追う。ここからは、ストリームから学ぶ仕組みのうち最適化の部分を掘り下げる。

## 目的

ストリームの設定で学べる最適な optimizer を、mujou のフレームワークに合う形で見つける。実験の結果をもとに Fable・Codex と相談しながら、多くの実験を回して絞り込む。

- **「最適」の物差し**: V8 の判定（4 条件）を満たしたうえで、1800 秒の span の誤差が小さいもの。忘却と長い run（6000 秒）は、上位の確かめに使う
- **「mujou に合う」の条件**: ノードの中で閉じる（ノードをまたぐ計算もグローバルな同期も持ち込まない）、時間の形で書ける（`elapsed` で進む）、遅すぎない（`run_manually` の仮想の時計で進むので、リアルタイムは要件にしない。1 フレームの計算時間は bench で記録する）
- **置き場所**: optimizer はすべて `exp/lib` に作る。mujou への PR は、性能が出たものだけを後で送る

## 前提（V0〜V10 から引き継ぐもの）

- モデルは ViT だけ（ResNet は使わない）。基準は ViT-S（幅 256、ヘッド 4、1 ノード 1 ブロック）
- データ・目的の式・PC-ALM・`run_manually` は V0 のまま（[PLAN.md](../PLAN.md)）。1800 秒、checkpoint は 60 秒ごと（`measure.ckpt_every=60`）、seed 0 で絞り込む
- 評価は V8 と同じ `python -m exp.vision.stream_eval`。判定の 4 条件も V8 と同じ（[README の V8](../README.md)）
  1. span の誤差が 0 秒より小さく、1 より小さい
  2. after の誤差が 0 秒より小さい
  3. span の `recon_blur` が、0 秒より、標準誤差を合わせたものの 2 倍を超えて小さい
  4. span の `dyn_1/copy_1` が 0 秒より小さい
- 各ノードは 1 フレーム（$\delta = 0.1$ 秒）に 1 回学ぶ。時間の形の optimizer は、$\delta$ が一定なら、学習率 $\delta/\tau$、減衰 $\beta = e^{-\delta/\tau_x}$ の、ステップで数える optimizer と同じ更新になる

## 段

| 段 | 内容 | 回す時期 |
| -- | -- | -- |
| O0 | optimizer 群、ノードごとの $\tau$、勾配の計測、OGD、幅と μP の実装 | — |
| V11 | optimizer の総当たり（勾配の計測つき） | O0 の後すぐ |
| V12 | デコーダの速さ | V11 と同じ波 |
| V13 | 勾配の直交化（OGD） | V11 の勾配の計測を読んでから登録する |
| V14 | 幅と μP | O0 と coord check の後 |

## O0: 実装の仕様

### optimizer（`exp/lib/optim.py` か、その隣の新しいモジュール）

`mujou.pc.optim.Optimizer` を実装する（`bind(params)` が `BoundOptimizer` を返し、`step_(grads, elapsed)` で動かす）。`torch.optim` は使わない（free-threaded で GIL が戻るため）。テストのオラクルは `torch.optim` で、$\delta$ 一定の数ステップで、学習率 $\delta/\tau$、$\beta = e^{-\delta/\tau_x}$ の `torch.optim` と一致することを確かめる。

| 名前 | 定義 | torch との対応 |
| -- | -- | -- |
| `nesterov` | `pc.optim.SGD(tau, tau_m)` の momentum（勾配の指数移動平均 $m$、質量 $c_m$ で偏りを補正した $\hat m$）の Nesterov 形。$\beta = e^{-\text{elapsed}/\tau_m}$ として $\Delta\theta = -\frac{\text{elapsed}}{\tau}\big((1-\beta)\,g + \beta\,\hat m\big)$。$g$ が一定なら一歩は SGD と同じ | `SGD(lr=(δ/τ)(1−β), momentum=β, nesterov=True)`（torch の buffer は和なので $1-\beta$ 倍で揃う。初めの数歩は偏りの補正の分だけずれる。`pc.optim.SGD` の docstring と同じ扱い） |
| `rmsprop_centered` | $g / (\sqrt{\hat v - \hat m^2} + \varepsilon)$ | `RMSprop(centered=True)` |
| `adadelta` | $\Delta x = -\frac{\delta}{\tau} \frac{\sqrt{\hat u + \varepsilon}}{\sqrt{\hat v + \varepsilon}}\, g$、$v$ は $g^2$、$u$ は $\Delta x^2$ の指数移動平均（時定数 $\tau_v$、`eps` 1e-6） | `Adadelta(lr=δ/τ, rho=e^{-δ/τ_v})` |
| `rprop` | 要素ごとの一歩 $s$ を、勾配の符号が前と同じなら 1.2 倍、反対なら 0.5 倍にし（反対のときはその回は動かさない）、$[s_{\max} \cdot 10^{-3}, s_{\max}]$ に収める。$\Delta\theta = -\text{elapsed} \cdot s \cdot \mathrm{sign}(g)$、$s_{\max} = 1/\tau$、$s$ の初期値は $s_{\max}/10$ | `Rprop`（$\delta$ 一定のとき） |
| `muon` | 行列（`nn.Linear` の重み、`nn.Conv2d` の重みは `(out, in·k·k)` に平たくする）は Muon: momentum（$\tau_m$）の Nesterov 形を Newton–Schulz 5 回で直交化し、`match_rms_adamw`（$0.2\sqrt{\max(A, B)}$ 倍）で大きさを AdamW に揃える。それ以外（bias、LayerNorm、GroupNorm、$\ln\sigma$、トークン）は AdamW。weight decay はどちらも 0.01 | `Muon(adjust_lr_fn="match_rms_adamw")` と `AdamW` |

既存の `sgd`・`sgd_momentum`・`rmsprop`・`adamw`（`mujou.pc.optim`）はそのまま使う。ステップ数 $t$ を更新式に含むもの（Adagrad・RAdam・NAdam・ASGD・torch の形の Adafactor）と、Adam・Adamax は扱わない（2026-11-19 のユーザーの判断。Adam 系は AdamW で足りる）。LBFGS（同じデータで損失を何度も評価する）と SparseAdam（疎な勾配専用）も外す。

`OptimConf` の `kind` に上の 5 つを足し、使わない設定を拒む既存の規則（`_USED`）を広げる。

### ノードごとの $\tau$（V12）

`optim.dec_scale`（既定 1.0）: デコーダのノード（`dec1`・`dec2`・`out`）の $\tau$ を `optim.tau × dec_scale` にする。エンコーダと `latent` は `optim.tau` のまま。

### 勾配の計測（`exp/lib` の、どの optimizer も包めるラッパー）

`measure.grad_stats`（既定 false）で有効にする。各ノードの中で閉じた観測だけで、学習を変えない（有効でも無効でも、学習の数値がビット単位で一致することをテストする）。ノードのパラメータをつないだ 1 本のベクトルで測る。1 秒ごとに `grad.jsonl` へ書く。

- $\lVert g\rVert$、$\lVert\Delta\theta\rVert$、$\lVert\theta\rVert$（ノードごと）。テンソルごとの $\lVert\Delta\theta\rVert/\lVert\theta\rVert$ は 60 秒ごと
- 時間を隔てた勾配の cos: 時間の幅 $L \in \{1, 10, 100\}$ 秒ごとに、$L$ 秒ごとに錨の勾配 $g_a$ を取り直し、そこからの経過時間（age）とともに $\cos(g_t, g_a)$ を書く。直前の勾配との cos（age 0.1 秒）も書く
- OGD の射影の後に残る割合 $\rho_s = \lVert (I - QQ^\top) g\rVert / \lVert g\rVert$: 間隔 $s \in \{0.1, 1, 10\}$ 秒ごとに保持した直近 $n = 16$ 個の勾配が張る空間（$Q$ は正規直交基底）。V13 の一歩がどれだけ残るかを、回す前に見積もる

メモリは run あたり 2 GB 以内（必要なら保持する勾配を bf16 にする）。集計は `python -m exp.vision.grad_table <run_dir>...`: ノードごとに、age（0.1・1・10・100 秒）の cos の平均と $\rho_s$ の平均を、0〜300 秒と 300〜1800 秒に分けて出す。

### 実装の記録（optimizer 群・dec_scale・勾配の計測、2026-11-19）

- optimizer は `exp/lib/optim_zoo.py`（`Nesterov`・`RMSPropCentered`・`Adadelta`・`Rprop`・`Muon`）にある。設定は `optim=<kind>`（`conf/optim/<kind>.yaml`）。`torch.optim` との対応は各 docstring に書き、テストで一致を確かめた。torch は偏りを補正しないので、nesterov・rmsprop_centered・adadelta と muon の momentum は、質量を 1 にした状態から一致する
- `rmsprop_centered` の分母は $\sqrt{(v - m^2)/c} + \varepsilon$ にした（$c$ は質量）。表のとおり $\hat v - \hat m^2$ にすると、1 歩目で 0 になり、一歩が $g/\varepsilon$ まで大きくなるため。$c = 1$ では torch の centered と一致し、1 歩目の大きさは RMSProp と同じになる
- `adadelta` の $u$ は、学習率を掛ける前の方向の 2 乗の平均にした（torch の `acc_delta` と同じ）
- `muon` は、2 次元以上のパラメータを行列として扱う。ViT-S とデコーダでは、これが `nn.Linear` と `nn.Conv2d` の重みとちょうど一致する（テストで確かめた）。行列以外は AdamW で、`tau`・`tau_m`・`tau_v`・`eps`・weight decay は Muon と同じ値を使う
- 勾配の計測は `exp/lib/grad_stats.py` の `GradStats`。時刻はノード自身の `elapsed` の和で数える。周期 $P$ の事象（書き出し・錨・保持）は、$kP$ に最も近い歩で起こす。テンソルごとの $\lVert\Delta\theta\rVert/\lVert\theta\rVert$ は、60 秒ごとの歩の 1 歩分の値。錨と保持は最初の歩で取る
- 保持する勾配は bf16 にした。ViT-S（8.29M パラメータ）では 52 本 × 8.29M × 2 B = 0.86 GB で、$\rho_s$ の計算のときだけ、さらに最大 0.19 GB を使う（enc4 の 16 本を fp32 にする）。計測の状態は再開の checkpoint に入る。止めて再開しても、`grad.jsonl` は止めなかった run と同じバイトになる（テストで確かめた）
- `grad_table` の age の bin は、0.1・1・10・100 秒を中心とする対数の区間（$10^k$ の bin は $[10^{k-1/2}, 10^{k+1/2})$）。直前の勾配（0.1 秒）との cos と各錨との cos を、age ごとにまとめて平均する

### OGD（V13、`exp/lib`）

各ノードで、$s$ 秒ごとに（経過時間で数える）生の勾配 $g$ を保持し、直近 $n$ 個を $G$ とする。前処理の計量 $P$（SGD は $I$、RMSProp・AdamW は $\mathrm{diag}(1/(\sqrt{\hat v} + \varepsilon))$）のもとで、

$$\min_\Delta\; d^\top \Delta + \frac{1}{2\eta} \Delta^\top P^{-1} \Delta \quad \text{s.t.}\quad G^\top \Delta = 0
\;\Rightarrow\; \Delta = -\eta \big(P - PG (G^\top P G)^{-1} G^\top P\big)\, d$$

$d$ は SGD・RMSProp では $g$、AdamW では $\hat m$。$\hat v$ は射影の前の $g$ で更新する。AdamW の weight decay は射影の外で掛ける。$G^\top P G$ の階数落ち（連続する勾配はよく似る）は、固有値分解で最大の固有値の $10^{-6}$ 倍より小さい方向を捨てて扱う。

局所性: 射影はノードの中で閉じる。各ノードで $g_{\mathrm{old},i}^\top \Delta_i = 0$ なら、和も 0 なので、1 次の干渉がないという OGD の保証は局所に限っても失われない。

テスト: $G^\top\Delta = 0$、記憶が空なら元の optimizer と一致、$P = I$ なら $(I - QQ^\top) g$、保持の間隔が `elapsed` の刻み方によらないこと。

設定: `optim.ogd_n`（0 で無効）、`optim.ogd_every`（秒）。

### 幅と μP（V14）

- `model.width` を ViT の幅とし、ヘッドの次元を 64 に固定する（`heads = width / 64`）。ヘッドの次元が幅によらないので、注意の $1/\sqrt{d_{\mathrm{head}}}$ は変えなくてよい
- デコーダのチャネル `decoder_channels` も `width / 256` 倍にする（`model.scale_decoder`、既定は V14 の run だけ true）
- `model.param`: `sp`（既定、いまの初期化と学習率）か `mup`。基準の幅は 256 で、幅 256 の `mup` は `sp` と同じ run になる
- μP（Adam 形の規則、Tensor Programs V の表 3。幅の比 $k = w / 256$）
  - 入力の側（パッチの埋め込み、デコーダの $z$ からの `Linear`）: 初期化も学習率も変えない
  - 隠れ層（ViT のブロックの行列、デコーダの中間の Conv）: 初期化の分散 $\propto 1/\text{fan\_in}$（いまの初期化のまま）、学習率 $\times 1/k$
  - 出力の側（エンコーダの $(\mu, \ell)$ への head、readout の Conv）: 初期化の標準偏差 $\times 1/k$、学習率 $\times 1/k$
  - bias・LayerNorm・GroupNorm・$\ln\sigma$: 変えない
  - テンソルごとの学習率の倍率は、optimizer に渡す倍率の表で与える（`fan_ins` と同じ、パラメータを鍵にした辞書）
- coord check（`python -m exp.vision.coord_check`）: 幅 128・256・512・1024 で最初の 10 秒（100 フレーム）を学び、各ノードの出力の RMS と、その初期値からの変化を 0・10・50・100 フレームで出す。μP では幅によらず、SP では幅とともに変わるはず
- PC-ALM の確かめ: 幅 128 と 512 でも、各ノードの勾配が BP の勾配と一致すること（`tests/vision/test_pc.py` と同じ形）

### 実装の記録（OGD・幅と μP・coord check、2026-11-19）

- OGD は `exp/lib/ogd.py` の `OGD(base, n, every)`。base は `pc.optim` の `SGD`（momentum なし）・`RMSProp`・`AdamW` で、その更新を同じ演算の順で書き直した。記憶が空の最初の歩は base とビット単位で一致し、モーメントは射影の後も base と一致する（テストで確かめた）。設定は `optim.ogd_n`（0 で無効、sgd・rmsprop・adamw だけ）と `optim.ogd_every`（既定 1.0。`ogd_n` = 0 では既定のまま）
- 保持の規則は `GradStats` と同じ（ノード自身の時刻、$kP$ に最も近い歩、最初の歩で 1 回）。保持する勾配はパラメータの dtype（fp32）で、ViT-S では 16 本 × 8.29M × 4 B = 0.53 GB。$G^\top P G$（$k \times k$）の固有値分解は CPU の float64 で解く
- `grad_stats` の $\rho_s$ の固有値分解も CPU に移した。mini-00 の smoke で、CUDA の linalg の遅延ロードが複数のノードのスレッドで同時に走り、`lazy wrapper should be called at most once` で落ちていたため
- `model.heads` の既定を `None`（幅 / 64、`ModelConf.head_count`）にした。`__post_init__` で書き戻すと、OmegaConf が schema の既定値を作るときに 4 に固まるため、値は property で読む。v0/tiny のように `heads` を書いた設定はそのまま
- `model.scale_decoder=true` で、デコーダのチャネルを幅 / 256 倍にする（整数にならなければ拒む）。GroupNorm の群の数は `gcd(チャネル, 32)` のままで、幅 128〜1024 で割り切れる（テストで確かめた）
- `model.param=mup`（ViT だけ、ヘッドの次元 64 のときだけ、OGD とは併用しない）。学習率の倍率は `exp.vision.models.mup_lr`（パラメータを鍵にした表）を `exp/lib/optim.py` の `ScaledLR` に渡す。`ScaledLR` は倍率 $s$ ごとにパラメータを束ね、内側の optimizer の $\tau$ を $\tau / s$ にして bind する。モーメントの時定数は変えず、AdamW の weight decay は学習率と一緒に倍率がかかる。倍率がすべて 1 なら内側の optimizer そのものになる
- μP の初期化は、幅 256 に錨を置いた「隠れ層の分散 $\propto 1/\text{fan\_in}$、出力の分散 $\propto 1/\text{fan\_in}^2$」と読んだ。いまの ViT のブロックの行列は幅によらず std 0.02（$1/\text{fan\_in}$ に比例しない）なので、μP では $k^{-1/2}$ 倍する。head（LeCun）・readout とデコーダの Conv（torch の既定）はもともと std $\propto \text{fan\_in}^{-1/2}$ なので、隠れ層の Conv はそのまま、出力の 2 つは $k^{-1/2}$ 倍する（幅 256 の std の $1/k$ になる）。SP は、いまの初期化のまま（ブロックは幅によらず std 0.02）
- 幅 256 の μP は SP と同じ run になることを、初期値と RMSProp の 2 フレームの学習で、ビット単位で確かめた。幅 128・512 の μP でも、各ノードの勾配は BP の float64 の勾配と一致した
- coord check（`exp/vision/coord_check.py`）は、最初のフレームのスタックを probe にして、各ノードのモジュールに通した出力（デコーダには $z = \mu$）の RMS と、0 フレーム目からの変化の RMS を出す。μP では、出力の側（enc4 の head、out）の初期の RMS は $k^{-1/2}$ で小さくなる（μP の初期化どおり）ので、判定は変化（delta）で読む
- 手元の CPU で、合成のスタック（毎フレーム独立な乱数）で幅 128・256・512、20 フレームまで回すと、delta の傾きは SP で +0.15〜+0.78、μP の隠れ層（enc1〜3、dec1・dec2）で −0.03〜+0.15 だった。μP の出力の側は −0.4〜−0.5 で、学習の入力と probe が無相関なためと読んでいる（$\Delta W x$ が独立な符号の和になる）。実データ（フレームどうしが似ている）の GPU で確かめる

## V11・V12・V14 の腕・判定・予想

[README の V11・V12・V14](../README.md) に書く（回す前に決めたこと）。V13 は V11 の勾配の計測を読んでから README に登録する。
