# M16 の条件 2 を通すための実装計画: 窓の BPTT による学習則（BP と PC-ALM で同じ規則）

## 0. 結論と方針

3 つの設計から、**objective-and-schedule（窓の再生と、ノードの中の BPTT）を土台に選ぶ**。ほかの 2 案からは次を取り込む。

- tbptt から: optimizer の値を elapsed で読み替える方法、1 つの unroll を全員で使う作り、有限差分のオラクル
- better-traces から: どの credit が効いたかを分けるオフラインの分解（`credit="own"`）

監査の指摘は、下の各節で直す。

選ぶ理由は 3 つある。

1. **BP の規則は、オフラインの上限そのものになる**。学ぶのは、今いる位置で終わる 36 フレームの窓で、$s=0$ から始め、16 フレームを観測し、20 フレームを開ループで予測する。損失は 35 個の予測の平均。この形は評価の手順（$s=0$ から 16 フレームを teacher forcing）とも同じなので、tbptt の「途中の状態から始める窓と、$s=0$ から始める評価のずれ」が起きない。窓の境界は、ノード全体で 1 本のフレームカウンタで決める。ストリームごとの位相のずれも、開ループの周期への整列も問題にならない。
2. **オフラインの予測が、オンラインの BP の run とほぼ同じ計算になる**。`OfflineFit(order="stream")` と、エンコーダを固定したオンラインの BP の run は、同じ窓・同じ optimizer・同じ elapsed で同じ重みの軌跡をたどる。これはテストのオラクルとして固定する。安い事前の検査が、そのまま BP の結果の予言になる。
3. **PC-ALM が退化しない**。動力学の 3 ノードと `Compare` は残す。各ノードは自分の再帰だけを窓の上で微分し、層をまたぐ時間方向の credit は、隣どうしの窓の形のメッセージで運ぶ。平衡点では、各ノードの $g_\theta$ が BP の窓の勾配と一致する。tbptt の fused 案は「PC-ALM が 1 段遅れの BP になる」問題があり、両方の監査が却下を示唆していた。

better-traces は近似で（合格の見込み 0.15）、開ループの feedback の経路を扱えない。そこで、診断の分解だけを取り込む。

規約に照らして、監査の指摘を次のように直す。

- 全体の長さを使う cosine decay は使わない（不変条件 3）。オフラインで通った run も decay なしだった
- 更新量はステップ数ではなく、時刻の差の elapsed で決める
- `replay` は片方の学習則にしか実装できないので入れない
- 窓は `Dynamics` だけが持つ（目標は窓のメッセージの `y` に載せる）。フレームの抜けがある窓は mask する
- 正規化は 1 か所で決める: 固定の $B\cdot P$ で割り、mask した窓の寄与は 0
- 符号は mujou の規約に合わせて明記する

## 1. 学習則（数式）

### 記号

- ストリーム $b$ の最新の $W$ フレーム（既定 $W=36$）を $x_0..x_{W-1}$ とする。$P=W-1$ を予測の数、$O$ を観測するフレームの数とする（既定 $O=16$）
- 各層の状態は 0 から始める: $s_{1,0}=0$、$h_{n,0}=0$
- $j=1..P$ について、次のように計算する（現在の `window_loss` と同じ）

$$
c_{j-1}=\begin{cases}x_{j-1} & j\le O\\ \phi(\hat x_{j-1}) & j>O\end{cases},\quad s_j=\mathrm{step}(c_{j-1},s_{j-1}),\quad \hat x_j=[c_{j-1}]+W_o\,\mathrm{Top}(s_j)
$$

ここで、`feedback` によって $\phi$ が変わる。

| `feedback` | $\phi$ | 意味 |
| -- | -- | -- |
| `grad` | 恒等 | 今のオフラインの上限 |
| `detach` | `.detach()` | feedback の経路を切る |
| `teacher` | — | $O=W$（開ループなし） |

### 損失と正規化

$$
L_\text{win}=\frac{1}{B\,P}\sum_b m_b\sum_{j=1}^{P}\frac1D\lVert\hat x_{j,b}-x_{j,b}\rVert^2
$$

$m_b$ は、窓の $W$ フレームがすべてリセット後で、抜けも重複も無いことを表す。$B$ は固定の値。mask が全部 True なら、今の `window_loss` の `.mean()` と一致する。

### 更新の時刻

- ノードのフレームカウンタ $k$（ring に push した数）は、全ストリームで共通。$k\ge W$ かつ $(k-W)\bmod S=0$ のとき、窓 $t$ ができる。既定は $S=18$
- 学習するのは、窓の時刻（最後のフレームの時刻）$t_w$ から `relax` 秒たった後の最初の update。BP も PC と同じ時刻に学ぶ。こうすると、ライブの予測（preq）も含めて両者がそろう
- `optimizer.step_(g, elapsed)` に渡す elapsed は、ring に保存したフレームの時刻だけで決める。値は $t_w(t)-t_w(t-1)$ で、$t=0$ だけは窓の中の $x_{W-1}$ と $x_{W-1-S}$ の時刻の差を使う。フレームの周期が一定なら $S\cdot\text{frame}$ になる。ステップ数は使わない

### optimizer の写像

`make_optimizer` の写像（$\tau=\text{frame}/\text{lr}$、1 回の step で $\text{elapsed}/\tau$ だけ動く）をそのまま使う。オフラインの `adam(1e-3, 0.9, 0.999)` を 1 窓に 1 回の step として、$S=18$（elapsed 0.6 s）なら次の値になる。

| 値 | 式 | $S=18$ のとき |
| -- | -- | -- |
| `optim.lr` | $10^{-3}/S$ | 5.56e-5 |
| `tau_m` | $0.6/(-\ln0.9)$ | 5.69 s |
| `tau_v` | $0.6/(-\ln0.999)$ | 600 s |
| `warmup` | | 0 |
| `weight_decay` | | 0 |

`pc.optim.AdamW` は、elapsed が一定なら Adam と厳密に同じになる（`adam` の docstring による）。

### ライブの経路

毎フレームの予測は今のまま続ける。持ち越した状態を今の重みで使い、`next`/`error` で prequential の予測を送る。ただし毎フレームの学習は行わず、トレースも使わない。

### BP（`DynamicsBP`）

$g=\nabla_\theta L_\text{win}$ を、ノードの中の autograd で求める。入力は記憶した、detach 済みの $z$ の窓。

### PC-ALM（3 ノード + `Compare`、窓のエピソード）

**各ノードの窓の関数**

- `Dynamics` は $F_1:(c_0..c_{P-1})\mapsto(s_{1,1..P})$。入力の観測の位置は自分の ring の値、開ループの位置は `Compare` の $\hat x$ の窓のメッセージで、どちらも定数として扱う
- `DynamicsLayer` $n$ は $F_n: Y_{n-1}\mapsto Y_n$。$y_{n,j}=y_{n-1,j}+\mathrm{minGRU}(\mathrm{GELU}(y_{n-1,j}),h_{n,j-1})$ で、自分の再帰は窓の中で閉じている
- `Compare` のエネルギーは、サンプルと位置の和で次のとおり。予測の基点 $c$ は `win_base` のメッセージで受け取る

$$
E_C=\sum_b m_b\sum_j\frac1D\lVert c_{j-1}+W_oy_{L,j}-x_j\rVert^2
$$

**緩和と学習**

- 各ノードは、窓の形の `r_w` と `lam_w`（`(B,P,d)`）を `pc.functional.relax_` で緩和する
- $\delta=r_w+\lambda_w$ とし、`pc.functional.vjp` で $g_\text{in}=-J^\top\delta$ と $g_\theta=-J_\theta^\top\delta$ を求める
- 平衡点では $\delta=-g_\text{above}$ になり、$g_\text{in}=J^\top g_\text{above}=\partial E/\partial(\text{入力の窓})$ となる。$g_\theta/(BP)$ は BP の $\nabla_\theta L_\text{win}$ と一致する（`detach` と `teacher` の場合）
- 窓の credit のグラフは三角なので、有限回の update で厳密に収束する。今の時定数（dt = τ_h/2、τ_r = τ_h、τ_λ = τ_h/2）では、$g$ が一定なら 1 update で $\delta=-g$ にちょうど達する（監査が数値で確かめた）

**開ループ**

- `detach` の開ループは、予測のポートを通る Jacobi 反復で解く。1 巡で正しい位置が 1 つ伸び、1 巡に 4 hop かかる
- 20 位置 × 4 hop × 2 ms = 160 ms なので、`relax` の既定は 0.3 s とし、$S\cdot\text{frame}=0.6$ s より短くする。エピソードは重ならない

**エピソードの約束**（局所的な規則だけで決める）

- 窓のメッセージは `time`（$t_w$）を持つ。ノードは新しい $t_w$ を初めて見たら、`r_w` と `lam_w` を 0 にする
- 学ぶのは、$t_w+\text{relax}$ の update で、読んだ窓の入力がすべてその $t_w$ のものだったときだけ。そうでなければ学ばず、`win_skipped` を数える
- 直前の update から入力が変わっていたら、`win_unsettled` を数える
- 学んだ後は、そのエピソードの窓のメッセージを凍結する。再計算せず、同じ値を送り続ける。遅れて学ぶノードは、他のノードが学ぶ前のスナップショットを読むので、早く学んだノードが平衡点を動かさない

**`grad`（feedback を通る credit）**

予備として持ち、作るのはオフラインで `grad` が `detach` より h=10 で 0.03 を超えて良かったときだけにする。

- `Compare.win_base` と `Dynamics.win_hat` を、開ループの位置にだけ credit を運ぶ双方向のポートにする
- `Compare` は、$\partial E/\partial\hat x+g_\text{fb}$ を `g_s` と $g_\theta(W_o)$ の両方に使う
- credit は時間方向にはやはり三角なので、収束する。BP の基準も `grad` にそろえる

### 局所性

| 学習則 | 微分するもの | ノードをまたぐもの |
| -- | -- | -- |
| BP | 1 ノードの中で、記憶した detach 済みの履歴に対する自分のモジュールだけ（`# mujou: local-autograd`） | なし |
| PC | 各ノードが、detach した窓の入力に対する自分の窓の関数だけ | 宣言したポートを通る窓の形のメッセージだけ |

- `.backward()` も `.grad` も使わない
- グローバルなバリアも無い。境界は各ノードのカウンタと $t_w$ で、遅れたノードは自分の学習を飛ばすだけになる
- エンコーダへの credit は無い。`Dynamics.z` と `Compare.target` は受け取るだけのままで、窓のポートは動力学のブロックの内側にしか繋がらない
- フェーズもエポックも無い。予測は毎フレーム続き、学習は 0.6 s ごとに続く

PC の学習が「毎フレーム」から「0.6 s ごとの窓」に変わるので、M16 での「オンラインの PC-ALM」の意味が変わる。**着手前にユーザーの了承を取り、`memory/` と `results/m16.md` に記録する**（§7）。

## 2. 変更するもの（ファイル・シグネチャ）

mujou 本体の変更は要らない。使うのは既存の `Port`、`MultiPort`、`TensorSpec`、`pc.msg.activity_spec`、`pc.functional.vjp`/`relax_`、`register_state`/`Stateful`、`pc.optim.AdamW` だけ。

### `exp/lib/mingru.py`

`MinGRU.scan(x_seq, s0) -> Tensor` を足す。`x_seq` は `(*batch, T, d_x)` で、返すのは `(*batch, T, d_s)`。

- gate と candidate の matmul を時間方向にまとめて 1 回で計算し、要素ごとの再帰だけを T 歩まわす
- `ResidualMinGRU.scan(x_seq, h0)` も足す

### `exp/lib/ring.py`（新規）

`FrameRing(Stateful)` を作る。ストリームごとの、直近 W フレームの ring。

- 保持するもの: `x (B,W,D)`、フレームの時刻 `(W,)`、リセットからのフレーム数 `since (B,)`、push の総数 `k`
- `push_(x, time, reset)`: 1 フレームを足す
- `window() -> (x (B,W,D), mask (B,))`: `mask` は「W フレームがすべてリセット後」かつ「時刻の差が `(0.5, 1.5)·frame` に収まり、抜けも重複も無い」
- `due(stride) -> bool`: `(k-W) % S == 0` かつ `k ≥ W`
- `elapsed(stride) -> float`: §1 の時刻の差
- `state_dict` と `load_state_dict`

特定の実験に閉じない部品なので lib に置く。

### `exp/video/schema/dyn.py`

`WindowConf` を足す。

```python
@dataclass(kw_only=True)
class WindowConf:
    window: int = 36; observed: int = 16; stride: int = 18
    feedback: str = "detach"   # "detach" | "teacher" | "grad"
    credit: str = "bptt"       # "bptt" | "own" (BP only, ablation)
    relax: float = 0.3         # s from the window time to the learning
```

`DynamicsConf.window: WindowConf | None = None` も足す。既定は今の動作のまま。

`__post_init__` で次の設定を弾く。

- `window` と `trace` を同時に有効にする
- `observed >= window`
- `stride < 1`
- `feedback` か `credit` が選択肢の外

PC と `grad` の組（予備を作るまで）、PC と `credit="own"` の組は、組み立てのときに `ValueError` にする。

### `exp/video/schema/learn.py`

`OfflineConf` に次の 4 つを足す。

| フィールド | 既定 | 意味 |
| -- | -- | -- |
| `order` | `"random"` | `"random"` か `"stream"` |
| `stride` | `0` | 0 は `window` と同じ |
| `feedback` | `"grad"` | |
| `credit` | `"bptt"` | |

`order="stream"` のときは `lr` を使わず、`conf.optim` を `make_optimizer` で作り、elapsed は `stride·frame` にする（docstring に書く）。既存の run は、既定の値で変わらない。

### `exp/video/dyn/modules.py`

`DynamicsStack.scan(c_seq, s0) -> (states (n,T,L d_s), tops (n,T,d_s))` を足す。入力の列が定数のときの、時間方向にまとめた forward。

### `exp/video/dyn/offline.py`

`window_loss` を拡張する。

```python
window_loss(mods, x, *, observed, local, residual,
            feedback="grad", credit="bptt", mask=None) -> Tensor
```

- `mask` は `(n,)`。和を $nP$ で割る
- `credit="own"` は、トレースと同じ打ち切り。運ぶ状態は $h^c_j=\text{cell}(x_j^{\text{det}},h^c_{j-1})$、出力は $h^c_j+\text{cell}(x_j,h^c_{j-1}{}^{\text{det}})-\text{cell}(x_j^{\text{det}},h^c_{j-1}{}^{\text{det}})$ とする（straight-through）。feedback は detach する
- `detach` と `teacher` の BP は、no_grad の rollout で $c$ を確定してから、`scan` で 1 回微分する最適化を入れてよい（下のオラクルで値の一致を確かめる）

`OfflineFit` に `order` と `stride` を足し、`optimizer: Optimizer | None` と `elapsed: float` を受け取れるようにする。

- `order="stream"` では、ステップ $t$ に全ストリームの窓 `[tS, tS+W)` を取る。重複して切るので、データはストリームごとの状態 `(B, frames, 4K)` で持ち、ステップごとに encode する
- 返り値の loss に加えて、勾配のノルムも記録する

### `exp/video/dyn/bp.py`（`DynamicsBP`）

`conf.window` があるときの動作:

- `_new_frame` で `FrameRing.push_` する。入れるのは、前のフレームで最後に受け取った $z$（PC の `Dynamics.z_last` と同じ約束）
- `due` なら、窓と mask と $t_w$ を `pending` に置く
- $t_w+\text{relax}$ の update で `window_loss` → `autograd.grad` → `optimizer.step_(g, ring.elapsed)` → `_predict()`
- 毎フレームの `_grads` は呼ばない
- 状態は ring、`pending`、`win_steps`、`win_skipped`、`win_loss_last`、`win_gnorm_last`

### `exp/video/dyn/nodes.py`

`conf.window` があるときだけ、次を足す。既存の経路は変えない。

**`Dynamics`**

- `FrameRing` を持つ
- ポート:
  - `win_next = MultiPort(send=activity_spec(TensorSpec((B,P,d_s)), y=TensorSpec((B,P,D)), batch), recv=TensorSpec((B,P,d_s)))`。目標 $x_{1..P}$ は `y` に載せ、`DynamicsLayer` はそれを `{**act}` でそのまま上へ渡す
  - `win_base`（送るだけ。$c_{0..P-1}$）
  - `win_hat`（受け取るだけ。`observed < window` のとき）
- 状態: `r_w`、`lam_w`、`win_time`、`frozen`、`optimizer` の束縛（既存のものを流用）

**`DynamicsLayer`**

- ポート: `win_prev = Port(send=g, recv=act)`、`win_next = MultiPort(...)`
- 状態: `r_w`、`lam_w`。入力と δ が前の update と同じなら、vjp を飛ばして前の値を送り直す（局所的で、結果は厳密に同じ）

**`Compare`**

- ポート: `win_s = Port(send=g, recv=act)`、`win_base`（受け取る）、`win_next`（$\hat x$ の窓を送る → `Dynamics.win_hat`）
- `xbuf` は持たない

**共通**

- 新しい関数 `window_energy(readout, Y, target, base, mask) -> (B,)` を、`compare_energy` の隣に置く
- 窓を使うとき、ライブの経路は forward（mu）だけにし、vjp と relax と毎フレームの学習を飛ばす。`Compare` は prequential の予測と誤差を送り続ける
- `DynamicsBlock` は `conf.window` のとき、窓のポートを配線する: `dynamics.win_next → dynamics2.win_prev → … → compare.win_s`、`dynamics.win_base → compare.win_base`、`compare.win_next → dynamics.win_hat`

### その他

- `exp/video/dyn/nets.py`: 変更なし。`live_states` と `modules` は今の属性で足りる。`win_*` のカウンタを読む関数 `window_stats(net) -> dict` を足す。計測のためにステップの間に読むだけで、更新則には使わない
- `exp/video/programs/joint.py`:
  - `_Offline` は `offline.order` によって窓の作り方と optimizer を切り替える
  - `_Online.log` に `win_loss`、`win_gnorm`（BP）、`win_steps`、`win_skipped`、`win_unsettled` を足す
  - `conf.dynamics.window` があるときは、`window ≥ measure.observed + max(horizons)` を確かめる（オフラインと同じ検査）
- 設定:
  - `exp/video/conf/experiment/m16/offline_stream.yaml`: `order: stream`、`stride: 18`、`dynamics: big`、`optim` に §1 の写像の値
  - `exp/video/conf/experiment/m16/balls_joint_win.yaml`: `big`、`residual`、`window: {window: 36, observed: 16, stride: 18, feedback: <C1 で選んだもの>}`、`trace: false`、optim は同じ写像の値
  - `exp/video/studies/m16_balls_joint_win_bp.toml`
- `exp/video/results/m16.md`: この段の判定基準を、回す前に書く（§5）。結果は後で書く

## 3. テストのオラクル（理論から導いたもの。float64・CPU・同期の `ManualRun`）

**`tests/lib/test_mingru.py`**

1. `MinGRU.scan` と `ResidualMinGRU.scan` は、step の繰り返しと値・勾配とも 1e-12 で一致する

**`tests/lib/test_ring.py`**

2. k 回 push した後、各ストリームは直近 W フレームを順に持つ
   - 窓の中にリセットや時刻の抜け・重複がある窓は mask される
   - N フレームでの窓の数は ⌊(N−W)/S⌋+1
   - `elapsed` は $S\cdot$frame になる
   - `state_dict` の往復で状態が保たれる

**`tests/video/dyn/test_offline.py`**

3. `mask=None` は今の `window_loss` とビット単位で同じ。`teacher` は `observed=W` と同じ。`detach` は、開ループの位置に $\hat x$`.detach()` を入れて手で組んだ損失と同じ
4. 有限差分: D=3、d_s=4、L=2、W=6、O=3、residual の極小の系で、`grad` と `detach` の各パラメータの勾配が中心差分と 1e-6 で一致する
5. トレースが特別な場合になること: `credit="own"` の勾配は、窓の中を $s=0$ から重みを固定して、`MinGRUTrace` を各層の保持した入力で進め、`_grads` と同じ規則でフレームごとに出した勾配の和に 1e-10 で一致する（オフラインの分解で `own` を診断に使う根拠）
6. `order="stream"`: ステップ $t$ のバッチは、状態の列を直接 `[tS,tS+W)` で切ったものと一致する

**`tests/video/dyn/test_bp.py`**

7. 生の勾配: 窓ができた最初の学習で、`DynamicsBP` に渡る勾配が `autograd.grad(window_loss(...))` と 1e-10 で一致する。SGD で回して、正規化の誤りを見逃さないようにする
8. run 全体の一致: M14 の `StateWorldBP`（観測の固定したストリーム）を N フレーム回した重みが、同じ $z$ で `OfflineFit(order="stream", stride=S)` を ⌊(N−W)/S⌋+1 ステップ回した重みと 1e-8 で一致する。SGD で 1 本、AdamW で 1 本（elapsed の写像も確かめる）
9. 退化: `window=None` の既存のテストがすべてそのまま通る

**`tests/video/dyn/test_nodes.py`（窓の PC）**

10. 平衡点: 1 つの窓を保ち、学習の時刻での各ノード（`Dynamics`、`DynamicsLayer`×2、`Compare`）の $g_\theta/(BP)$ が、BP の $\nabla_\theta L_\text{win}$ の対応する部分と 1e-10 で一致する。`detach` と `teacher`、nodes=1 と 3、residual の有無で確かめる。符号は $\delta_w=r_w+\lambda_w=-B P\,\partial L_\text{win}/\partial(\text{出力の窓})$ を直接 assert する
11. 軌跡: PC と BP の窓の学習を SGD で 3 窓まわした重みが 1e-8 で一致する（正規化の誤りを捕まえる）。AdamW は別に 1 本
12. 開ループ: 学習の時刻に `Compare.win_next` が送る $\hat x$ の窓は、BP の `detach` の rollout と 1e-12 で一致する
13. 非同期: 1 つのノードの周期だけ 2 倍にしたスケジュールでは、窓は (a) BP と 1e-8 で一致して学ばれるか、(b) 飛ばされて `win_skipped` に数えられるかのどちらかで、ずれた窓では決して学ばない
14. 凍結: 学んだ後、次の $t_w$ までそのエピソードの窓のメッセージがビット単位で変わらない
15. resume: エピソードの途中で止めて `run.state_dict()` から再開した run は、止めずに回した run とビット単位で同じ

**`tests/video/programs/test_joint.py`**

16. `TestNoCreditToTheEncoder` を窓を使う設定で回す（pc と bp）。エンコーダは視覚だけの run とビット単位で同じになる
17. 窓のポートを `z_out` に繋ぐと `TypeError` になる
18. 既存の静的な検査（`.backward`/`.grad` が無いこと）を、窓のコードにも広げる

## 4. 安い検査（回す前に判定基準を書いておく）

### C0（コードなし、すぐ）

`m16-offline-v1` の学習曲線（大の BPTT と大の 1 フレームの、250/500/1000/3000/6000/12000 ステップ）を読む。

- 大の BPTT が 3000 ステップで `rec_roll10/pix10` ≤ 0.90 に届かなければ、余裕が薄いことを先にユーザーに伝える。C1 は回す
- 大の 1 フレームが 1.2〜1.3 なら、オンラインの頭打ちと同じ形と記録する

### C1（offline.py の変更だけ、約 1 日 + GPU）

`task=offline`、大、val を使う。エンコーダは M15 の確認の BP の seed 0。`order=stream` で、optimizer は §1 の写像の値、cosine decay は無し。

| run | 設定 |
| -- | -- |
| a | S=18（2999 ステップ）× feedback ∈ {grad, detach, teacher} |
| b | S=36（1499 ステップ）× {grad, detach} |
| c | `order=random` で a と同じステップ数（ストリームの順番の効果を分ける） |
| d | `credit=own`、S=18（帰属を分ける診断。判定には使わない） |

判定（最終ステップの値）:

1. **G0a（作るかどうか）**: S=18 で、ある feedback が BP の s0 と PC の s0 のエンコーダの両方で `rec_roll5/pix5` ≤ 0.85 かつ `rec_roll10/pix10` ≤ 0.95 を満たせば、BP の窓の学習を作る
   - どれも満たさなければ止める。大きさや損失の重み（開ループの位置）を見直す案をユーザーに出す
   - PC の M15 のエンコーダでの run（PC の s0）は、a のうち選んだ feedback について回す
2. **feedback の選び方**: G0a を満たすもののうち、h=10 で最良から 0.03 以内に入る、いちばん単純なものを選ぶ。順は teacher、detach、grad。grad しか残らなければ、PC の予備（双方向のポート）を作る
3. **ストリームの順番**: `stream` が `random` より h=10 で 0.05 を超えて悪ければ、記録したうえで S=9 も試す。replay は両方の学習則で作れる形が決まるまで入れない
4. **seed の幅**: 選んだ設定を、BP と PC の seed 1・2 のエンコーダ（計 4 本）でも回す。どれかが h=10 で 1 を超えたら、PC を作る前にユーザーに報告する
5. **帰属**: d（`own`）も ≤ 0.95 なら、効いたのは層をまたぐ credit ではなく「$s=0$ の窓という目的」だと results に書く。d が約 1.2 なら、効いたのは時間を通した credit だと書く

### C2（BP のオンライン、§2 の bp.py の実装の後）

val の seed 100/101 で 1800 s 回す。設定は写像の値で、run は 2 本。

- 同時に学ぶ run（`m16/balls_joint_win`）
- 対照: エンコーダは seed 100/101 の M15 の run（m15 の broadcast の study の #13 の trial。残っていなければ、先に視覚だけの run を seed 100/101 で回す）

判定:

- **G1**: 同時に学ぶ run が、2 つの seed とも `rec_roll10/pix10` < 0.95 かつ `rec_roll5/pix5` < 0.85
- **等価性の検査**: 対照の seed 100 の最終の `rec_roll10/pix10` が、同じエンコーダでの C1 の stream fit と 0.01 以内に入る。テスト 8 の実機での確かめで、外れたら実装を疑う
- 2 本に後から学ぶデコーダを回し、`img_*` と `rec_*` の比の差を記録する（判定は `img`）

## 5. run の計画

1. **BP の study**（`m16-balls-joint-win-bp-v1`、同時に学ぶ run、val の seed 100/101、1800 s、約 20 trial）
   - 最初に写像の値を入れる
   - 探索の空間: S ∈ {9, 18, 36}（lr は 1 窓あたりの Adam の 3e-4〜3e-3 を lr = 値/S で読み替える）、`tau_m` は 1 窓あたりの β1 ∈ [0.8, 0.95] に相当する範囲、`tau_v` は β2 ∈ [0.99, 0.9995] に相当する範囲、warmup ∈ {0, 60, 300}
   - 固定: 大、residual、W=36、O=16、C1 で選んだ feedback
   - 目的は今の `image_objective`。G1 を満たしていれば、study は小さくてよい
2. **PC-ALM**（§7 の了承の後）
   - テスト 10〜15 を通す
   - `task=bench` で 60 s（仮想時間）回して fps を測る。2 fps を下回れば、確認の前に vjp の再利用と `scan` を詰める
   - PC の val の seed 100 で、BP と同じ設定の対照を回す。BP とは表示の 3〜4 桁まで一致する見込み。`win_skipped` と `win_unsettled` が 0 であることを確かめる
3. **確認**（`m16-confirm-v2`、study の 1 位）: `just sweep confirm … --learners bp,pc --seeds 0,1,2`、同時に学ぶ run と対照、4090、test の後に posthoc（`m16-confirm-post-v2`）
   - 条件 1〜5 は変えない。条件 2 は `img_roll{5,10}`、条件 3 は `post/decode/<t>.png` で判定する
   - 確認の run で `win_skipped`、`win_unsettled` が 0 であることを assert する
4. 結果は `results/m16.md` と README の表の行に書く。snapshot、mujou の SHA、seed も記録する

工数の見込み:

| 項目 | 日数 |
| -- | -- |
| C1 のコード | 1 |
| BP の窓 | 1.5 |
| PC の窓 | 3.5 |
| 設定と文書 | 0.5 |
| 計 | 約 6.5 日 + GPU の時間 |

PC の実装は、C1 と C2 を通ってから始める。BP だけで外れたときの損を小さくするため。

## 6. リスクと対処

- **余裕が薄い**: オフラインで通った 0.925 は 1 本のエンコーダ・val・1000 ステップでの値。C0 の曲線と C1 の 4 つの seed で余裕を先に測り、C2 の同時に学ぶ run で、エンコーダが動くことの損を直接測る
- **勾配の爆発**: 20 歩の開ループと残差を通るので、勾配が大きくなりうる。`win_gnorm` を記録する。クリップは入れない（オフラインも入れていない）。入れるなら、両方の学習則と offline.py に同じ規則を入れ、上限からの逸脱として書く
- **PC の重さ**: 前の項目の vjp の再利用、時間方向にまとめた `scan`、ライブの経路の forward だけの処理で抑える。それでも遅ければ、`relax` を縮めて Jacobi を短くする（`teacher` が選ばれれば約 16 ms で済む）
- **`grad` が要る場合**: PC の予備（双方向の feedback のポート）が要り、PC の工数は約 1.5 倍になる
- **M14 との共有**: M14 のコードの経路はクラスを共有する。`window=None` の既定で今の動作は変わらず、リセットを含む窓は mask する（テスト 2・9）

## 7. 判断が要ること（着手前にまとめて確認する）

1. **PC-ALM の学習の頻度が変わる**: 0.6 s ごとの窓のエピソード（局所的・非同期・継続的）にし、毎フレームの学習とトレースを使わない。これを M16 の「オンラインの PC-ALM」として認めるか。認める場合は `memory/` に 1 ファイルを足し、`results/m16.md` の判定の節に注記する
2. **学習率の decay を使わない**: run の長さを前提にする cosine decay は使わない（不変条件 3）。了承を確認したい
3. **選択の手順**: C1 は val で選ぶ。M15 の確認のエンコーダ（seed 0/1/2）は test の run とビット単位で同じエンコーダだが、C1 で使うのは val の窓だけで、確認の seed での設定の選択はしない。この扱いを results に明記してよいか

推奨は 1〜3 すべて「はい」で、この前提で C0・C1 から始める。
