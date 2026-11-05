# M16 段 C3: PC-ALM の窓の学習の設計（実装前）

計画は [m16-window-bptt.md](m16-window-bptt.md) の §1「PC-ALM」「grad」、§2 nodes.py、§3 の 10〜15。C1 で `feedback="grad"` を選んだ（[results/m16.md](../../../exp/video/results/m16.md)）。BP 側の実装は C2（`ffc4109`: `exp/video/dyn/bp.py` の `DynamicsBP` の window 経路、`exp/lib/ring.py` の `WindowRing`）。

**目標**: `learner=pc` と `dynamics.window` の組で、各ノードの $g_\theta/(BP)$ が、BP（`DynamicsBP` と `autograd.grad(window_loss)`）の同じ窓・同じ時刻の勾配の対応する部分と、平衡点で一致すること。対象は M14（`StateWorldPC`）と M16（`JointPC`）。

mujou 本体の変更は要らない。使うのは `Port`、`DictSpec`、`TensorSpec`、`IntSpec`、`FloatSpec`、`TimestampSpec`、`pc.functional.vjp`、`pc.functional.relax_`、`pc.clock.LearnClock`、`WindowRing`、各ノードの optimizer。

---

## 0. 記号と添字

- $B$: ストリーム数（`math.prod(batch)`）。$W$: 窓の長さ。$P=W-1$: 予測の数。$O$: 観測するフレーム数（`teacher` では $O:=W$）。$K=P-O$: 戻した予測を入力に使う位置の数（既定 $36-1-16=19$）。$L$: 動力学のノード数（`Dynamics` が 1 つと `DynamicsLayer` が $L-1$ 個）。$d$ は `d_s`、$D$ は $z$ の次元（joint は 128）
- 窓の frame は $x_0..x_{W-1}$ で、`(B, W, D)`。`WindowRing.window()` の返り値をそのまま使う
- **メッセージの位置の添字は $p=0..P-1$ にそろえる**
  - 出力の窓（$s_{1}$、$Y_n$、$\hat x$、目標）の $p$ は、予測 $j=p+1$ を指す: $Y_n[p]=y_{n,p+1}$、$\hat X[p]=\hat x_{p+1}$、目標 $T[p]=x_{p+1}$
  - 入力の窓 $C$ の $p$ は $c_p$ を指す（$s_{p+1}$ をつくる入力）: $p<O$ なら $C[p]=x_p$、$p\ge O$ なら $C[p]=\hat x_p=\hat X[p-1]$
  - base（残差の基点）は $\hat x_{p+1}=c_p+W_oY_L[p]$ なので、$\mathrm{base}[p]=C[p]$ で添字が同じ
- `window_loss`（`exp/video/dyn/offline.py:168-227`）と同じ規則: $c_j$ は `j < observed` なら $x_j$、それ以外は $\hat x_j$（`grad` はそのまま、`detach` は `.detach()`）。残差の base は $c$ そのもの（`detach` の開ループでは base も detach される）

$$
L_\text{win}=\frac{1}{BP}\sum_b m_b\sum_{p}\frac1D\lVert\hat X_b[p]-T_b[p]\rVert^2,\qquad E:=BP\,L_\text{win}
$$

## 1. 符号の規約（mujou に合わせる）

`pc.functional` の規約（`mujou/src/mujou/pc/functional.py:1-24, 44-87, 168-220`）を窓にそのまま当てはめる。

- 隣へ送る credit $g$ は、つねに**エネルギーの勾配** $\partial E/\partial(\text{送り手の出力})$ で、受け手はそれを `relax_` の `g` に足す
- 隠れノードは出力の窓に `r_w`、`lam_w`（出力の窓と同じ shape）を持つ。$\delta_w=r_w+\lambda_w$
- `vjp(f, h_in, lambda _: δ, params)` は $g_\text{in}=-J^\top\delta$、$g_\theta=-J_\theta^\top\delta$ を返す
- 平衡点では $\delta_w=-g$ なので、$g_\text{in}=J^\top g=\partial E/\partial(\text{入力の窓})$、$g_\theta=\partial E/\partial\theta$ になる
- 各ノードは `optimizer.step_([g/(B*P) for g in g_θ], elapsed)` で学ぶ（optimizer が勾配を引く）。$B\cdot P$ は固定で、mask した窓の寄与は 0（`window_loss(mask=...)` と同じ）
- テスト 10 で直接確かめる式: $\delta_w=r_w+\lambda_w=-BP\,\partial L_\text{win}/\partial(\text{出力の窓})$

### `relax_` の 1 update での到達

時定数は既定の $\tau_r=\tau_h$、$\tau_\lambda=\tau_h/2$、dt は `max_dt = period` $=\tau_h/2$。このとき 1 回の `relax_` は、**それまでの r、λ によらず** $\delta'=-g$ にする。

$r'=r-\tfrac12(2r+\lambda+g)=-\tfrac12(\lambda+g)$、$\lambda'=\lambda+r'=\tfrac12\lambda-\tfrac12 g$ なので、$r'+\lambda'=-g$。

- `tick.dt` は `max_dt` で頭打ちになる。Host（`exp/lib/host.py`）とテストの `ManualRun` は毎回ちょうど period で進むので、各 update の後に $\delta=-g_\text{いま受けた}$ が丸め誤差の範囲で成り立つ
- dt < period の update では一致は漸近的になる（固有値の絶対値は 1 未満、たとえば $a=0.4$ で $\sqrt{0.2}$）。点 4 の skip の規則は、この場合は再計算する

**窓のノードは update の中で「受信 → `relax_` → vjp → 送信」の順に処理する**（毎フレームの経路の「vjp → relax」とは逆）。こうすると credit の 1 hop が 1 update で済む。これはノードの内側の順序の選択で、局所性は変わらない。

## 2. 各ノードの窓の関数

### `Dynamics`（ノード 1）

$$
F_1:\;C\in\mathbb R^{B\times P\times D}\mapsto S_1=\text{MinGRU.scan}(C,\,s_0=0)\in\mathbb R^{B\times P\times d}
$$

- $C[p<O]$ は、自分が保持した窓の $x_p$（定数）
- $C[p\ge O]$ は、受け取った $\hat X[p-1]$（定数として vjp の入力に入れる）
- `teacher` では、すべて $x_p$

$S_1$ の credit $g$ は、次のノードから `win_next` で受け取る。

- `relax_(r_w, lam_w, g)` で緩和する
- `vjp(lambda c: scan(c, 0), C, δ)` は $(S_1,\ g_C=-J^\top\delta,\ g_\theta)$ を返す

**feedback の credit（`grad` のときだけ）**

$C[p]$（$p\ge O$）の全 credit は、2 つの使い道の和になる。

$$
\bar a(c_p)=g_C[p]+g_\text{base}[p]\qquad(p\ge O)
$$

- $g_C[p]$ は自分の $F_1$ を通る分
- $g_\text{base}[p]$ は Compare の base を通る分。Compare から `win_base` で受け取る（残差のときだけ）

これを $\hat x_p$ の credit として、`win_hat` で Compare に返す。添字は出力の窓にそろえ、$q=p-1$ に置く。

$$
\mathrm{FB}[q]=\bar a(c_{q+1})\ \ (O-1\le q\le P-2),\qquad \mathrm{FB}[q]=0\ \text{(それ以外)}
$$

- $\bar a(c_p)$ は自分の入力 $c_p$ のエネルギーの勾配で、`DynamicsLayer.prev` が送る $g_\text{in}$ と同じ種類の値。Dynamics は $c$ に r、λ を持たない（入力の和を中継するだけ）
- `detach` では FB を送らない（ポートも受け取るだけ）

### `DynamicsLayer` $n$（ノード 2..L）

$$
F_n:\;Y_{n-1}\mapsto Y_n=\text{ResidualMinGRU.scan}(Y_{n-1},\,h_0=0)=Y_{n-1}+\text{MinGRU.scan}(\mathrm{GELU}(Y_{n-1}),0)
$$

- 自分の再帰（$h_{n,j}=y_{n,j}-y_{n-1,j}$）は窓の中で閉じている
- 上からの $g$ で `relax_` し、`vjp(lambda y: scan(y, 0), Y_{n-1}, δ)` の $g_\text{in}$ を `win_prev` で下へ送る

### `Compare`（出力ノード）

$$
\hat X=\mathrm{base}+W_oY_L\quad(\text{残差でなければ }\mathrm{base}=0)
$$

$$
\tilde E_C(Y_L,\mathrm{base};\theta_C)=\underbrace{\sum_b m_b\sum_p\frac1D\lVert\hat X_b[p]-T_b[p]\rVert^2}_{E_C=\texttt{window\_energy}}+\underbrace{\sum_{b,p}\mathrm{FB}_b[p]\cdot\hat X_b[p]}_{\text{grad のときだけ}}
$$

FB は受け取った定数。線形項は、`vjp` の `(mu * -d).sum()` と同じく「受け取った credit を出力に掛けた 1 次の項」にあたる。ノードの内側の autograd（`# mujou: local-autograd`）で、detach した葉 $Y_L$、base から次を求める。

- $g_Y=\partial\tilde E_C/\partial Y_L=W_o^\top\bar a(\hat X)$。$\bar a(\hat X)=\tfrac2D m(\hat X-T)+\mathrm{FB}$ で、`win_s` で返す
- $g_\text{base}=\partial\tilde E_C/\partial\,\mathrm{base}=\bar a(\hat X)$。`win_base` で返す（`grad` かつ残差のとき）
- $g_\theta=\partial\tilde E_C/\partial(W_o,b_o)$

Compare は出力ノードなので r、λ を持たない（`loss_vjp` と同じく損失をそのまま使う）。FB も、受け取った値をそのまま使う。

### 平衡点が BP と一致する理由

**FB の項を入れる理由**

c は $\hat x$ の恒等写像なので、$\hat x_p$ の随伴は $e_p+\bar a(c_p)$ になる（$e_p$ は直接の損失の勾配）。$\tilde E_C$ の線形項は、これを出力ノードの中で表すものである。

**平衡点の性質**

credit の不動点の方程式は、`window_loss` の unroll した計算グラフをノードの境界で切ったときの reverse-mode の随伴方程式そのものになる。

- $\delta_n=-\bar a(Y_n)$
- $g_\theta$ = 各ノードの $J_\theta^\top\bar a$

時間方向には三角（位置 $p$ の随伴は位置 $\ge p$ の値にだけ依存する）なので、解は一意で、有限回の update で厳密に達する（§4）。

**`detach` と `teacher`**

- `detach`: FB = 0、$g_\text{base}$ は不要
- `teacher`: 開ループのポートが無い

**ノード境界の credit の定義**

$\partial L/\partial(\text{出力の窓})$ は「出力を次のノードが読むところにだけ足した 0 の摂動 $\varepsilon_n$ の勾配」と定義する。自分の再帰を通る分は含まない。これがテスト 10 の参照になる。

## 3. ポートとメッセージの spec

`nodes.py` に TypedDict と spec の関数を足す。dtype は `act_spec` の `h` に合わせる（テストは float64）。

```
class WindowActivity(TypedDict):   # the forward window, D -> layers -> Compare
    h: Tensor        # (B, P, d)   S_1 or Y_n
    y: Tensor        # (B, P, D)   targets T[p] = x_{p+1}, passed on as {**act}
    mask: Tensor     # (B,) bool   m_b of the ring
    time: Timestamp  # t_w: the write time of the frame the window is due at
    elapsed: float   # ring.elapsed() at the due push (FloatSpec)
    rev: int         # the sender's revision of h (IntSpec)

class WindowMessage(TypedDict):    # every other window message
    x: Tensor        # (B, P, ·)
    time: Timestamp  # t_w of the episode it was computed in
    rev: int
```

`window_activity_spec(h: TensorSpec, y: TensorSpec, batch) -> DictSpec[WindowActivity]` と `window_message_spec(x: TensorSpec) -> DictSpec[WindowMessage]` を足す。`WA(d)` は $h$ が `(B,P,d)` の WindowActivity の spec、`WM(k)` は $x$ が `(B,P,k)` の WindowMessage の spec を表す。

| ノード | ポート | send | recv | 条件 |
| -- | -- | -- | -- | -- |
| Dynamics | `win_next` | `WA(d)`（$S_1$） | `WM(d)`（$g$ of $S_1$） | window |
| Dynamics | `win_base` | `WM(D)`（$C$） | `WM(D)`（$g_\text{base}$） / なし | 残差。recv は grad のとき |
| Dynamics | `win_hat` | `WM(D)`（FB） / なし | `WM(D)`（$\hat X$） | feedback ≠ teacher。send は grad のとき |
| DynamicsLayer | `win_prev` | `WM(d)`（$g_\text{in}$） | `WA(d)` | window |
| DynamicsLayer | `win_next` | `WA(d)`（$Y_n$） | `WM(d)` | window |
| Compare | `win_s` | `WM(d)`（$g_Y$） | `WA(d)` | window |
| Compare | `win_base` | `WM(D)`（$g_\text{base}$） / なし | `WM(D)`（$C$） | 残差。send は grad のとき |
| Compare | `win_next` | `WM(D)`（$\hat X$） | `WM(D)`（FB） / なし | feedback ≠ teacher。recv は grad のとき |

**配線**（`DynamicsBlock.__init__` で `conf.window` のときだけ）

- `dynamics.win_next → dynamics2.win_prev`、…、`dynamics{L}.win_next → compare.win_s`
- `dynamics.win_base ↔ compare.win_base`
- `compare.win_next ↔ dynamics.win_hat`

**ポートの型**

- 窓のポートは消費者が 1 つなので `Port` にする（`MultiPort` にはしない）。初期値は zeros で、`time` は ns=0（窓が無いことを表す）。本物の $t_w$ は $(W-1)\cdot$frame 以上なので 0 にならない
- 窓のポートは窓の形の spec なので、`z_out`（`Port(send=act_spec)`）にも `Source.out` にも繋がらない（`TypeError`）。`Dynamics.z` と `Compare.target` は受け取るだけのまま。エンコーダへの credit は、ポートの型で無いことが保証される

## 4. エピソードの約束と各ノードの update

ノードは自分の update で、局所的な規則だけで動く。

### 共通の状態

| 状態 | 中身 |
| -- | -- |
| `r_w`, `lam_w` | `(B,P,d)` のテンソル（Dynamics と DynamicsLayer）。属性なので自動で state になる |
| `episode` | `dict \| None`、`register_state`。`t_ns`（$t_w$）、`at_ns`（$t_w$+relax）、`elapsed`、`done`（学んだか、飛ばした）。Dynamics はこれに `x`（`(B,W,D)`、保持した窓）と `mask`（`(B,)`）を足す |
| `seen` | tuple、`register_state`。直前の update で読んだ各窓入力の `(time_ns, rev)` |
| `rev_f`, `rev_b` | int、`register_state`。自分が送った forward と credit の版数（単調増加、リセットしない） |
| `win_steps`, `win_skipped`, `win_unsettled`, `win_vjps` | int、`register_state` |
| `win_gsq` | 0-d テンソル。最後に学んだ $\lVert g_\theta/(BP)\rVert^2$ |
| Dynamics だけ | `ring: WindowRing`（Stateful の属性として state になる）、`win_masked` |
| Compare だけ | `win_loss`（0-d、$E_C/(BP)$、学んだ update の学ぶ前の重みでの値） |

### 規則（全ノード）

1. **エピソードの $t_w$**
   - Dynamics: 自分の ring で決める
   - 他のノード: forward の窓入力（`win_prev`、`win_s`）の `time` で決める
   - 新しい $t_w$ を初めて見たら、前のエピソードが `done` でなければ `win_skipped += 1` とする。そのうえで `r_w`、`lam_w` を 0 にし、`episode` を作り直す（`at_ns = t_w + relax_ns`、`elapsed` はメッセージの値）
2. **凍結**: `episode` が `done` の間は、窓のメッセージを送らず、credit のポートも読まない
   - スロットは最後の値を保つ。これが「学ぶ前のスナップショット」になる
   - DynamicsLayer と Compare は新しい $t_w$ を見つけるために forward 入力だけを読む。Dynamics は何も読まない
3. **他の $t_w$ の入力**: `time` が自分の $t_w$ と違う窓入力は、まだ届いていないものとして扱う
   - credit と FB は 0、$\hat X$ は 0（開ループの位置の初期値）とする
   - Compare は、base が $Y$ と同じ $t_w$ になるまで計算しない
4. **再計算の省略**（コストの対策）: `keys` = 各窓入力の `(time_ns, rev)` の tuple とし、`changed = keys != seen`、`seen = keys` とする
   - `changed` か、学ぶ update か、`tick.dt < max_dt` のときだけ vjp を計算する（`win_vjps += 1`）
   - 計算したら送る。forward の入力が変わったときだけ forward を `rev_f+1` で、credit は `rev_b+1` で送る
   - 送らない update では、スロットが前の値を保つ
   - 省略しても、結果は「毎回計算する」場合と丸め誤差の範囲で同じで、決定的でもある（§1 で δ は 1 update で $-g$ に達するため）
5. **学ぶ時刻**: `now >= at_ns` の最初の update で、次を行う
   - 窓入力をすべて読み、`relax_` する
   - `params` 付きの vjp で $g_\theta$ を求める（そのエピソードで params 付きの vjp はこの 1 回だけ）
   - **読んだ窓入力の `time` がすべて $t_w$ なら** `step_([g/(B*P)], episode.elapsed)` を行い、`win_steps += 1`、`win_gsq` を更新する。`changed` なら `win_unsettled += 1` とする
   - そうでなければ学ばず、`win_skipped += 1` とする
   - どちらの場合も `done = True` にする（凍結）
   - この update の送信は、学ぶ前の重みで計算した値だけにする（optimizer の step は送信の後）
6. **毎フレームの経路**（`window` のとき）
   - `mu` の forward だけを no_grad で行う
   - 毎フレームの vjp、`relax_`、学習、トレース、毎フレームの credit のポート（`next` の recv、`prev` の send、Compare の `s` の send）は、使わず送らない（初期値の zeros のまま）
   - Compare は `next`（prequential の予測）と `error` を今のまま送る
   - `LearnClock` は dt と `new` のためだけに使い、`tick.learn` は読まない（`DynamicsBP` と同じ）

### Dynamics の 1 回の update

1. `act = z.recv()`、`tick = clock.tick(...)`
2. `tick.new` なら、毎フレームの `_new_frame`（今のまま）を行う
3. 毎フレームの経路の forward を行い、`next` と `base` を送る（今のまま、vjp なし）
4. 開いているエピソードがあれば、窓の 1 歩を進める（規則 4、5）
   - ただし、この update で新しい窓ができて、まだ学ぶ時刻でないなら計算を飛ばす。次の 5 で skip に数える
5. `tick.new` なら、`ring.push_(act["h"], act["time"].ns, act["reset"])` を行う。`DynamicsBP._push` と同じく、その frame を最初に見た update の $z$ を入れる
   - `ring.due()` なら、`frames, mask = ring.window()`、`win_masked += (~mask).sum()` とする
   - `mask.any()` が偽なら、エピソードは作らない（前のものが続く。BP の「pending を残す」と同じ）
   - 真なら、前のエピソードが `done` でなければ `win_skipped += 1` とする（BP の「置き換え」と同じ）。そのうえで新しいエピソード（`t_ns = act["time"].ns`、`at_ns`、`elapsed = ring.elapsed()`、`x = frames`、`mask`）を作り、r、λ を 0 にして、その場で forward を送る
   - 開ループの位置の $C$ は 0 で始める

$\hat X$ がまだ届いていない間の forward は、どれも正しくない。正しい値は §5 の掃引で決まり、初期値は収束までの回数に影響しない。

**BP との同期（M14 と、エンコーダを固定した run）**

- 窓の frame・mask・$t_w$・elapsed が `DynamicsBP` と同じになる
- 学ぶ時刻は、どちらも「`now >= t_w + relax` の最初の update」になる

**例外: BP の「pending の時刻が来たら、次を持つ前に学ぶ」**

- Dynamics は、4 → 5 の順なので、同じ update で前のエピソードを学べる
- 他のノードは、学ぶ update が次の $t_w$ を見た後なら学べず、skip に数える（規則 1）
- relax が $S\cdot$frame − 数 period 以下なら起きない。既定は 0.3 s と 0.6 s で、150 period の余裕がある

## 5. 収束までの update の数と relax

1 つの「掃引」は 1 hop が 1 update とする。$K=P-O$、$L$ = 動力学のノード数。

### Host とテストのスケジュール

`ManualRun.step` は木の順（Dynamics, dynamics2.., Compare）に Gauss-Seidel で更新する（`mujou/src/mujou/execution.py:559-568`）。forward の hop（D→L2→…→C、D→C の FB、D→C の base）は同じ step の中で届く。後ろ向きの hop（C→L_L→…→D、C→D の $g_\text{base}$）は 1 step かかる。

- **forward**: $\hat x_{O+k}$ は、エピソードの開始の step から $k$ step 後に正しくなる。$\hat x_P$ は $K$ step 後
- **backward（grad）**
  - $\bar a(\hat x_P)=e_P$ は $K$ step 後に正しくなる
  - feedback を 1 つ戻るごとに、C→L_L→…→D→(FB) C で $L$ step かかる
  - $\bar a(\hat x_O)$ は $K+LK$ step 後に正しくなる
  - その後、Dynamics の δ まで $L$ step かかる
- **全ノードの $g_\theta$ が厳密になるまで**: $T_\text{grad}=(L+1)K+L$ step
  - 既定（$L=3$、$K=19$）では **79 step = 158 ms**（period 2 ms）
  - 開始の step は書き込みの 1 period 後なので、$t_w+160$ ms に収束する
  - $L=1$ では $2K+1=39$ step
- `detach`: $T=K+L$（22 step）
- `teacher`: $T=L$

### 完全に非同期な最悪の場合

両向きの hop がそれぞれ受け手の 1 period かかる場合、ループの 1 周は $L+1$ hop になる。

- forward: $L+(L+1)K$
- backward: $(L+1)K$
- 最後の下り: $L$
- 開始: 1

合計は $2L+2(L+1)K+1$。既定では **159 period = 0.318 s** になる（「1 巡 4 hop × 20 巡 × 2 ms」の 160 ms が forward と backward に 1 回ずつ）。

### relax の決定

**既定の 0.3 s（150 step）のままにする。**

- この実験のプログラムはすべて Host の決定的な木の順で回る。必要なのは 80 step（0.16 s）で、余裕は 70 step（1.9 倍）
- C2 の BP の run（`m16/balls_joint_win`）も 0.3 s なので、BP の対照と学ぶ時刻が同じになる
- `RunConf` の検査 `relax < stride*frame`（0.6 s）も満たす
- 完全に非同期な実行では 0.318 s が要り、0.3 s では足りない。そのときは 0.4 s にする（§9 の確認事項）。足りなかった場合は `win_unsettled` が数える
- テストの大きさ（$W=6$、$O=3$、$K=2$）: $T=3L+2$ で、$L=1,2,3$ について 5、8、11 step。非同期の最悪で $L=3$ は 23 period。テストの relax は 0.05 s（25 step）、周期を 2 倍にするテスト 13 は 0.1 s にする（$S\cdot$FRAME = 4·34 ms = 0.136 s 未満）

## 6. 計算量と速度の対策

**1 回の窓の vjp**（B=128、P=35、d=512）

- gate と candidate の matmul は、時間方向にまとめて 1 回で行う（`MinGRU.scan`）。4480×512×1024 を fwd と bwd で、約 14 GFLOP
- 要素ごとの再帰は T=35 歩（`torch.addcmul` が 1 歩に 1 kernel）で、bwd もある
- 4090 で 1 回 1〜1.5 ms と見込む（kernel の起動が支配的）

**回数**

- 規則 4 の省略を入れると、1 エピソードで 1 ノードが vjp を計算するのは、forward で約 $K$ 回、backward で約 $K$ 回（$L$ step ごとに credit が変わる）
- 合計は約 40 回 × 4 ノード ≈ 160 回/窓 ≈ 9 回/frame、つまり **約 10〜15 ms/frame** を足す見込み
- 収束した後から学ぶまで（約 70 step）は、どのノードも vjp を計算しない。凍結の後もしない

**コピー**

- `recv` は毎回コピーする。窓のメッセージは 9.2 MB（`(128,35,512)` の float32）と 2.3 MB
- 凍結中は forward の入力だけを読む。1 frame で約 1 GB の device 内のコピー、つまり約 1〜3 ms/frame
- 足りなければ、目標 `y` を Dynamics → Compare の別のポートに分ける（後回し）

**その他**

- 毎フレームの経路は forward だけ。GPU の同期は、窓を作る update の `mask.any()` が 1 回だけ（rev は Python の int）
- **計測**: `task=bench` の joint を tiny の設定、CPU で、`learner ∈ {pc, bp}` × 窓の有無で回し、`result.json` の `final.timing` の fps を報告する

## 7. ファイルとシグネチャ

### `exp/lib/mingru.py`

- `MinGRU.scan(x: Tensor (*batch, T, d_z), s0: Tensor (*batch, d_s)) -> Tensor (*batch, T, d_s)`
  - $u=\sigma(W_ux)$ と $\tilde s=W_sx$ を全 T で 1 回に計算する
  - $s_t=(1-u_t)s_{t-1}+u_t\tilde s_t$ を T 歩まわして stack する
- `ResidualMinGRU.scan(x: Tensor (*batch, T, d), h0: Tensor (*batch, d)) -> Tensor (*batch, T, d)`
  - $y=x+\text{cell.scan}(\mathrm{GELU}(x),h_0)$

### `exp/video/dyn/nodes.py`

- 足すもの: `WindowActivity`、`WindowMessage`、`window_activity_spec`、`window_message_spec`
- `window_energy(readout, y_top: (B,P,d), target: (B,P,D), mask: (B,), base: (B,P,D) | None) -> Tensor (B,)`。これが $E_C$ の各サンプルの値で、`compare_energy` の隣に置く
- `Dynamics`、`DynamicsLayer`、`Compare` に `window: WindowConf | None = None` を足す
  - Dynamics はさらに `feedback` を `window` から読む
- 窓の処理はそれぞれ private の `_window_step(tick)` にまとめる
- `window is None` の経路は 1 行も順序を変えない。分岐は最初に置く
- `DynamicsBlock`
  - C2 のガード `ValueError("the PC-ALM window learning is not built yet ...")` を削除する
  - 代わりに、`conf.window.credit == "own"` を `ValueError`（「BP の ablation のみ」）にする
  - 窓のポートを配線する
  - `window` property を足す
- module docstring と各クラスの docstring を更新する

### `exp/video/dyn/nets.py`

`window_stats(dyn: DynamicsBP | DynamicsBlock) -> dict[str, float | None]` に広げる。BP の返り値は今のまま。PC の返り値は次のとおり。

| キー | 値 |
| -- | -- |
| `win_steps` | ノードの最小 |
| `win_skipped` | ノードの和 |
| `win_unsettled` | ノードの和 |
| `win_masked` | Dynamics の値 |
| `win_loss` | Compare の値 |
| `win_gnorm` | $\sqrt{\sum_n \texttt{win\_gsq}_n}$（平衡点で BP の `win_gnorm` と同じ） |

窓が無ければ `ValueError`。

### `exp/video/programs/joint.py`

- `_Online._window_stats` は、`JointPC` なら `net.block`、`JointBP` なら `net.world` を読む
- module docstring の「BP only; PC-ALM raises」を直す

### `exp/video/schema/dyn.py`

`WindowConf` の docstring の「BP only for now」を直す。PC の relax の目安として、§5 の式を 1 行で書く。

### テスト

| ファイル | 変更 |
| -- | -- |
| `tests/lib/test_mingru.py` | scan のテストを足す |
| `tests/video/dyn/test_window_pc.py`（新規） | 10〜15 |
| `tests/video/dyn/test_bp.py` | `test_the_pc_learner_has_no_windows_yet` を「PC と `credit="own"` は `ValueError`」に置き換える。ガードを消すので、この 1 本は変える |
| `tests/video/programs/test_joint.py` | `test_the_pc_learner_has_no_windows_yet` も同様に置き換える。16 を足す |
| ほかの既存テスト | 変えない |

## 8. テスト（float64、CPU、`mujou.testing.run` の `ManualRun`）

**共通の部品**（`test_window_pc.py`）

- `_WorldPC64`: `StateWorldPC` の float64 版。`_World64` と同じ Source と period 2 ms
- `_mods64`: test_bp と同じ。readout は 0 でない
- $W=6$、$O=3$、$S=4$、$B=3$、$D=4$、`d_s=5`、FRAME = 17 step
- `_RecordingByParams`: bind した params の id でノードを区別し、`(grads, elapsed, now_ns)` を記録する。内側の optimizer はそのまま、または動かさないもの
- `_reference(mods, x, *, observed, residual, feedback, mask) -> (loss, xhats, eps)`
  - ノードで分けた参照の unroll。各ノードの出力を次のノードが読むところにだけ、0 の葉 `eps_n (B,P,d)` を足す
  - `grad` では、戻した $c_p$（$p\ge O$）にも `eps_c (B,P,D)` を足す
  - 冒頭で、`loss` が `window_loss(mask=...)` と 1e-12 で一致することを assert する（参照の検査）

**テストの一覧**

| # | 内容 |
| -- | -- |
| 10 | 平衡点（`TestEquilibrium`） |
| 11 | 軌跡（`TestTrajectory`） |
| 12 | 開ループ |
| 13 | 非同期（`TestAsynchrony`） |
| 14 | 凍結と省略 |
| 15 | resume |
| 16 | joint（`test_joint.py`） |
| lib | `MinGRU.scan` |

**10. 平衡点（`TestEquilibrium`）**

- パラメータ: feedback ∈ {grad, detach, teacher} × nodes ∈ {1, 3} × residual ∈ {T, F}
- 手順: W フレームを見せ（最初は reset）、新しい frame を書かずに step を続けて、学ぶ時刻の直前で止める
- 直前の step の後に読み、次を確かめる
  - 各ノードの `r_w + lam_w` が、`-B*P*autograd.grad(loss, eps_n)` と 1e-10 で一致する（符号の規約を直接確かめる）
  - grad では、`run.read(dynamics.win_hat)["x"]` が、FB の定義どおり `B*P*∂loss/∂eps_c` を 1 つずらしたものと一致する
- 学ぶ step の後に、次を確かめる
  - 各ノードの記録した grads が、`autograd.grad(window_loss(ref, frames, mask), params)` の対応する部分と 1e-10 で一致する。対応は Dynamics = `step.first`、dynamics{n} = `step.rest[n-2]`、Compare = `readout`
  - elapsed は $S\cdot$FRAME
  - 学んだ時刻は BP と同じく `at_ns` 以上の最初の update
  - `win_unsettled == win_skipped == 0`
  - Compare の `win_loss` は BP の loss と 1e-12 で一致する

**11. 軌跡（`TestTrajectory`）**

- $W+2S$ フレームで 3 窓を学ぶ
- `StateWorldPC` と `StateWorldBP`（同じ `_mods64`、同じ `DynamicsConf`）の重みが 1e-8 で一致する。SGD で 1 本、AdamW で 1 本（別のテスト）
- `window_stats` の `win_steps` = 3、`win_skipped` = `win_unsettled` = 0
- `win_loss` と `win_gnorm` が BP と 1e-8 で一致する
- feedback は grad と detach、nodes は 3、残差
- prequential の予測の一致は求めない。学ぶ update が frame の最後の step に当たると、毎フレームの状態の再計算の時点が BP とずれるため（重みは同じ）

**12. 開ループ**

- 学ぶ時刻（10 と同じ手順）に、`run.read(compare.win_next)["x"]` が `_reference` の $\hat x_{1..P}$（no_grad で同じ重み）と 1e-12 で一致する
- grad と detach、nodes は 1 と 3

**13. 非同期（`TestAsynchrony`）**

- スケジュールは 3 種
  - 1 つのノード（`dynamics` / `dynamics2` / `compare` をパラメータにする）の周期を 2 倍にする。そのノードは 1 step おきの `run.step(PERIOD, *all)` だけで更新し、他の step は `run.step(PERIOD, *others)` にする
  - 逆順（`run.step(PERIOD, compare, dynamics3, dynamics2, dynamics, source)`）
  - 上の 2 つを合わせたもの
- optimizer は動かさない記録だけのもの。重みが一定なので、各窓の BP の参照は初期の重みでの `autograd.grad(window_loss)` になる
- 確かめること
  - (a) 記録された (ノード, 窓) の grads が、その窓の参照と 1e-8 で一致する
  - (b) 各ノードで、記録の数 + `win_skipped` = mask のある窓の数
  - これで「ずれた窓で学ぶ」ことが無いことを確かめる
- relax は 2 つ
  - 0.1 s: すべて学ぶ。`win_skipped == 0`
  - $S\cdot$FRAME − PERIOD/2: 次の $t_w$ が遅いノードの学ぶ時刻より先に届くので、そのノードは skip に数える。`win_skipped >= 1` を assert する。実装でどの組で起きるかを確かめ、テストに固定する
- 窓の数は 4（$W+3S$ フレーム）

**14. 凍結と省略**

- ノードが学んだ step から次の $t_w$ の直前までのすべての step で、そのノードが送る窓のポートの `run.read` が、`torch.equal` と int・Timestamp の比較でビット単位で変わらない
- 収束した後から学ぶまでの step で、各ノードの `win_vjps` が増えない（省略の検査）
- 省略の無い参照（`tick.dt < max_dt` の分岐を常に通すテスト用のフラグは作らない）とは、10 の一致で間接的に確かめる

**15. resume**

- エピソードの途中（$t_w$ の後、学ぶ前。Dynamics の `episode` が開いていて、`r_w` が 0 でない）で `run.state_dict()` を取る
- 新しいネットワークに `mujou.testing.run(net2, state=state)` で戻して続ける
- 止めずに回した run と比べ、次がビット単位で同じになる（`torch.equal`）
  - 重み
  - 窓のポートの `run.read`
  - `window_stats`
  - optimizer は AdamW
- grad、nodes は 3、残差

**16. joint（`test_joint.py`）**

- `TestNoCreditToTheEncoder` に `(1, JEPA, "pc", WINDOW)` を足す。nodes は 1。$W=4$、$O=2$ では $T=3$ step で、relax 0.01 s = 5 step 以内
  - `dynamics.nodes=3`、`dynamics.window.relax=0.02` の組も足す（$T=7$ step）
  - エンコーダと predictor が視覚だけの run とビット単位で同じ
  - `window_stats(joint.block)["win_steps"] == 3`
  - `win_skipped == win_unsettled == 0`
  - `p.grad is None`
- `TestResume::test_a_run_learning_on_windows_resumes_as_one_never_stopped` を `learner ∈ {pc, bp}` にパラメータ化する
  - pc では、止めた時点で `block.dynamics.episode` が開いていることを assert する
  - stream.jsonl と evals.jsonl がバイト単位で同じ
  - PC のログには `win_unsettled` がある
- `test_window_ports_cannot_connect_to_z_out`: `net.z_out.connect(p)` が `TypeError` になる。`p` は `block.dynamics.win_hat`、`block.dynamics.win_next`、`block.compare.win_base`、`block.compare.win_s`
- `TestDynamicsIsM14s::test_a_fixed_z_sequence_gives_m14s_predictions_and_weights` に窓（pc）の組を足す。エンコーダを固定した joint と、M14 の `StateWorldPC` が、ビット単位で同じ重みになる
- 静的な検査（`test_window_pc.py::TestLocality`）
  - 窓の学習の後、全パラメータの `.grad is None`、`SGD` で全ノードの重みが動き、`_Still` では動かない（`tests/video/dyn/test_nodes.py:279-306` の窓版）
  - 加えて、ソースの検査を足す。`exp/video/dyn/nodes.py` と `exp/lib/mingru.py` の本文に `.backward(` が無いこと、`autograd.grad` 以外の `.grad` の属性参照が無いこと（正規表現 `(?<!autograd)\.grad\b`）、`autograd.grad(` のある行がすべて `# mujou: local-autograd` を持つこと

**lib（`tests/lib/test_mingru.py`）**

- `MinGRU.scan` と `ResidualMinGRU.scan` は、step の繰り返しと値・入力の勾配・パラメータの勾配のすべてで 1e-12 で一致する（float64）
- `s0 ≠ 0` と batch の形 `()`、`(B,)` も確かめる

**退化**

- 既存のテスト（`window=None`）は、上の 2 本の置き換えを除いて変えずに通る
- 最後に `just run` が exit 0 になること

## 9. リスクと確認事項

- **relax 0.3 s は、Host の決定的なスケジュールでのみ十分**（§5）。非同期で使うなら 0.4 s にする。暫定案は 0.3 s のまま（BP の C2 と同じ時刻に学ぶ）で、確認の run では `win_unsettled == 0` を assert する
- **joint では、PC と BP の $z$ の時点が違う**（`joint.py:50-56` の既知の差）。PC と BP の一致のオラクルは、M14 とエンコーダを固定した run だけで成り立つ。同時に学ぶ run では、表示の 3〜4 桁での一致は見込まない
- **速度**: 見込みは +10〜15 ms/frame（4090）。bench で 2 fps を下回るか、BP の 2 倍を超えたら、次を順に試す
  1. `y` を別のポートに分ける
  2. 凍結中の forward 入力の読みを、小さい「エピソード」のメッセージに替える
- **`relax_` で dt が period より短い update**: δ が 1 update で $-g$ に達しないので、省略しない（規則 4）。Host では起きない

## Audit

algorithm-reviewer（2026-11-04）。対象は設計のみで、コードの差分は無い。`just run` は走らせていない（コードの変更が無く、fixer がこのファイル以外を書き換えるのを避けるため）。

**verdict: request-changes**。数理の核（符号・FB・正規化・平衡点の BP との一致・三角性による有限回の収束・(L+1)K+L の数え方）は正しい。ただし規則 4（再計算の省略）と `relax_` の組み合わせでは、窓のメッセージがビット単位で落ち着かない。そのため `win_unsettled`・`win_vjps`・速度の見込みと、テスト 10・11・14・16 の `win_unsettled == 0` が成り立たない。

### must-fix

1. **(time, rev) の鍵では、ループが永久に回る**（確信度: 高）
   - 規則 4 は「forward の入力が変わったら forward を rev+1 で送る」。forward のループ D→…→C→X̂→D と credit のループ D→FB→C→…→D があるので、rev は毎 update 進み続ける
   - その結果、全ノードが毎 step vjp を計算する。学ぶ update では毎回 `win_unsettled` を数え、`win_vjps` は止まらない
   - 数値例（W=36・O=16・L=3・Host の順）: detach でも、150 step すべてで 4 ノードとも vjp を計算した（`scratchpad/sim3.py` の rev モード）
   - 直し方: 送る値が前に送った値とビット単位で同じなら、rev を上げず送らない（`torch.equal`）。前に送った値は state に入れる（resume のため）
2. **値で比べても、grad では ulp の揺れが回り続ける**（確信度: 高）
   - δ=−g に達した後も、`relax_` は r→0、λ→−g へ半分ずつ動き続ける。どこか 1 位置でも入力が変わると、`relax_` が P 個の位置をすべて丸め直す。その揺れが credit のループを回る
   - 数値例: 値で比べても、grad で最後にメッセージが変わったのは float64 で step 188（relax 0.3 s の 150 step を超える）、float32 で step 134
   - テストの大きさ（relax 25 step、float64）では、学ぶ時点で落ち着かない
   - 直し方（推奨）: 1 update で厳密な領域（`dt == max_dt` かつ `tau.lam == h·r/(h+r)`）では、`relax_` の後に `(r_w, λ_w) := (0, −g)` とする。これは g を固定したときの `relax_` の不動点で、δ は数学的に変わらない
     - 検証: grad は step 78（= 設計の (L+1)K+L = 79 step）、detach は step 21（22 step）で、float64 と float32 の両方でビット単位で落ち着いた。BP との相対誤差は float64 で 6e-16、float32 で 8e-7（`scratchpad/sim5.py`）
     - 代案: 許容誤差つきで変化を判定する。この場合、ビット単位の再現は保てるが、鍵の意味が変わる

### should-fix

1. **速度の見込みが約 2 倍甘い**
   - 落ち着くまでの間、Host の順では各ノードの入力が毎 step どこかの位置で変わる。上の修正を入れても、1 窓で 1 ノードあたり約 (L+1)K+L ≈ 80 回の vjp になる（見込みは約 40 回）
   - 4 ノードで 320 回/窓、約 18 回/frame で、4090 では **+25〜30 ms/frame**
   - 対策: forward の入力が変わらず δ だけが変わる update（D と層では、80 回のうち約 60 回）は、`torch.func.vjp` の `vjp_fn` を使い回して forward を省く。`vjp_fn` は state にしない（resume の後に作り直す）。bench では vjps/窓 も報告する
2. **1 update で厳密な領域の検査を入れる**
   - 有限回の収束・step 数・省略・上の修正は、どれも `tau.lam = h·r/(h+r)` と `period = max_dt` を前提にする。`TimeConf` は変えられる
   - PC で window を使い、この領域にないときは `DynamicsBlock` で `ValueError` にする（または省略と修正を切り、その旨を docstring に書く）
3. **`win_unsettled` の判定（「この update で入力が変わった」）は弱い**
   - FB の段では、あるノードの credit が L step に 1 回しか変わらないスケジュールがありうる（周期を 2 倍にしたノードなど）。その間の update で時刻が来ると、収束前に学んでも数えない
   - 「最後に入力が変わった時刻から (L+1)·period 以内」などで判定する。Host の順では、落ち着くまで毎 step 変わるので今の判定でも捕まる
4. **計画 §7-1 の了承の記録**
   - 「PC の学習を 0.6 s ごとの窓のエピソードにし、これを M16 のオンライン PC-ALM とする」ことの了承を記録する `memory/` の 1 ファイルが無い（不変条件 3 にかかる判断）。実装の前に足す
5. **テストの余裕が小さい**
   - テスト 13 の周期 2 倍は、最悪で約 2×23=46 step 要る。relax 0.1 s（50 step）に対して余裕は 4 step。実装で数えて固定するか、0.12 s（< S·FRAME = 0.136 s）にする
   - テスト 16 の joint（3/5、7/10 step）は、エンコーダを通る開始の遅れを数えてから relax を決める

### nit

- テスト 13 の「記録 + skip = 窓の数」は、各ノードがすべての t_w を見る前提。最後に開いたままのエピソードはどちらにも入らないので、`at_ns` を過ぎた窓だけを数える
- ソースの検査の正規表現 `(?<!autograd)\.grad\b` は、docstring やコメントの `.grad` にも当たる。コメントと docstring を除いて検査する
- 凍結中も、新しい t_w を見つけるために 9.2 MB の forward の窓を毎 update 読む（§9 の後回しのままでよい）

### 照合メモ

- `relax_`（τ_r=τ_h、τ_λ=τ_h/2、dt=τ_h/2=`max_dt`）は r'=−(λ+g)/2、λ'=(λ−g)/2 で、r'+λ'=−g になる。`max_dt=min(period, 2λ)=period` を `time_constants.py` で確かめた
- `window_loss` の添字: C[p]=x_p（p<O）、X̂[p−1]（p≥O）。base=C。FB[q]=ā(c_{q+1})（O−1≤q≤P−2）。ā(c_p)=g_C[p]+g_base[p]、g_base[p]=ā(X̂[p])。Compare の Ẽ=E_C+ΣFB·X̂ から g_Y=W_oᵀ(e+FB)、g_θ=Σ(e+FB)Y_Lᵀ となり、reverse-mode の随伴と一致する
- 数値: Host の順の Gauss-Seidel を、mujou の `pc.functional.vjp`/`relax_` でそのまま模擬した（W=6、O=3、B=3、mask に False を 1 つ）
  - 12 の組（feedback×nodes {1,3}×residual）すべてで、g_θ/(BP) と `autograd.grad(window_loss)` の差が ≤ 9e-16、E_C/(BP) と L_win の差が ≤ 9e-16
  - W=36、O=16、L=3 でも差は 1e-12 以下
- step 数: 上の修正を入れると、grad で 79 step、detach で 22 step。設計の (L+1)K+L と K+L に一致する
- 局所性: ノードをまたぐ autograd は無い（Compare の葉 Y_L と base は detach、D の C は定数）。重みはそれぞれの所有ノードだけが書く。グローバルなバリアは無い（学ぶ時刻は run のタイマーと t_w で各ノードが決め、ずれたら skip）。凍結によって、遅れて学ぶノードも学ぶ前のスナップショットを読む。ずれた窓で学ぶ経路は見つからなかった（Dynamics は 4→5 の順、他のノードは新しい t_w を見たら古いエピソードを skip にする）
- 確認事項への意見: (1) relax は 0.3 s のままでよい。すべて Host の仮想時間で動き、上の修正の後は 79 step で 1.9 倍の余裕がある。(2) joint で PC と BP の一致を M14 とエンコーダを固定した run に限るのは妥当
