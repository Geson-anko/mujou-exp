# 地図: モデルとノード（M14〜M16 の再編のため）

対象: `exp/video/model.py`（416 行）、`exp/video/nodes.py`（533 行）、`exp/video/bp.py`（388 行）、`exp/lib/mingru.py`（108 行）、`exp/lib/sigreg.py`（216 行）。利用側として `exp/video/config.py`・`run.py`・`evaluate.py`・`tests/video/test_model.py` も参照した。2026-11-03、main 06a2a11。

**注意**: このマシン（darwin）では submodule `mujou/` が空（`git submodule status` が `-881c4ee...`）。`~/Projects/mujou` は古い版で、`pc.msg`・`pc.clock` がまだ無く API が違う。mujou の API の意味は exp 側の使い方と PLAN.md から読んだもので、再編の前に `just setup` で 881c4ee を取り、`pc.Input`・`pc.Layer`・`pc.clock.LearnClock`・`pc.functional.relax_` を確かめること。

---

## 1. 全体の配線（`model.Network`、model.py:268-416）

```text
host ─Sample─▶ Input ─▶ g1 ─▶ z ─┬──▶ dynamics(.z) ─▶ dynamics2 ─▶ dynamics3 ─▶ compare.s
                (pc.Input) (pc.Layer)│        ▲ (.z_hat, open_loop>0)               │
                                     ├──▶ sigreg.prev                               │
                                     ├──▶ compare.z ◀───────────────────────────────┘
                                     └──▶ reconstruct.z (recon>0)   input.next ─▶ reconstruct.x
                         dynamics.base ─▶ compare.base   (residual)
net.prev = input.prev, net.next = compare.next（pc.msg.Prediction）
```

組み立ての順（乱数の消費順が BP 版と一致する必要がある。§6）:

1. `tau = pc.TimeConstants(h=t.h, lam=t.lam, r=t.r)`（既定 4e-3 / 2e-3 / 4e-3。周期 `tau.period` は 2 ms。1 フレーム 1/30 s ≒ 16〜17 ステップ）
2. `optimizer_of(name)`（model.py:299）: 名前が `dynamics*` か `compare` なら `config.dynamics_optim or config.optim`、それ以外（g1, z, reconstruct）は `config.optim`。`correlation=True` なら `exp.lib.optim.Correlation` で包み `self.correlations[name]` に置く（run.py が `rho_<node>_<lag>` を記録）
3. `torch.manual_seed(config.seed)` → `encoder_stages(config)` → `standardize_(nn.Sequential(*stages), calibration_images(config))` → `dynamics_modules(m)` → `readout = nn.Linear(m.d_s, m.d_z)`（`residual` なら重みとバイアスを 0 初期化）→ `decoder(...)`（`recon>0` のとき）
4. **すべて CPU で作ってから `.to(config.device)`**（初期値をデバイスに依らせない、model.py:316）。ノードにも `device=dev`
5. `self.input = pc.Input(x=TensorSpec((*d.batch, *image_shape(d))), y=TensorSpec((*d.batch, 1)), period=tau.period, batch=d.batch, device=dev)`。`y` は使わないゼロ
6. エンコーダ: `encoder_names = ("g1", ..., "z")`、各段を `pc.Layer(stage, act_spec=prev.act_spec, tau, optimizer, learn_interval=li, device)` にして `setattr` し、`prev.next.connect(node.prev)` で直列につなぐ
7. `Dynamics(step, act_spec=self.z.act_spec, ...)`、その上に `DynamicsLayer` を `dyn_nodes-1` 個（名前 `dynamics2`, `dynamics3`, ...）。`top` が最上段
8. `SIGReg(act_spec=self.z.act_spec, kappa, directions, capacity=m.buffer, every=m.buffer_every, tau, seed=config.seed, device)`
9. `Compare(readout, s_spec=top.act_spec, z_spec=self.z.act_spec, kappa, tau, optimizer, learn_interval, target_weight=target_weight(m), residual, device)`
10. 接続: `z.next → dynamics.z, sigreg.prev, compare.z`（＋`reconstruct.z`）、`top.next → compare.s`、`open_loop>0` なら `compare.next → dynamics.z_hat`、`residual` なら `dynamics.base → compare.base`、`recon>0` なら `input.next → reconstruct.x`
11. `self.residual = m.residual`、`self.prev, self.next = self.input.prev, self.compare.next`

公開している属性（利用側が名前で読む）: `tau`, `correlations`, `input`, `g1`/`g2`/`z`, `encoder_names`, `dynamics`, `dynamics2..`, `dynamics_names`, `sigreg`, `compare`, `reconstruct`, `residual`, `prev`, `next`。`evaluate.Snapshot.of(net)`（evaluate.py:41）は `encoder_names` と `dynamics_names` と `compare.module` を `getattr` で集めて `Snapshot(encoder, step=DynamicsStack, readout=Sequential(Top, W_o), residual)` を deepcopy する。`latency.py` は `net.z`・`net.sigreg`・`net.compare.module` を直接使う。

### ホストとの約束（run.py:190-196）

- 毎フレーム `pc.msg.Sample(x=f.image, y=zeros(*batch,1), mask=~reset, reset=reset)` を `net.prev` に書き、`Host.show({net.prev: sample}, net.next)` で 1 フレーム分ステップし、書いた `time` の最初の `Prediction`（prequential、学ぶ前の予測）を受け取る
- **`mask = ~reset` が前提**（SIGReg は mask を「動画の最初のフレームでない」と読む）
- B 本のストリーム（`DataConfig.streams > 1`）なら全テンソルの先頭に `(B,)`、`batch = ()` か `(B,)`。`DataConfig.batch` プロパティ

---

## 2. 各ノードの役割

PC-ALM の共通の形（nodes.py の自作ノード）:

- 状態 `r`, `lam`（活動 `h` の shape）。`pc.clock.LearnClock.of(tau, period=self.period, learn_interval=li)` の `tick = clock.tick(dt, act["time"], timer.now())` が `tick.new`（新しい `time` を初めて見た更新）、`tick.learn`（学ぶ更新）、`tick.dt`、`tick.elapsed`（前回学んでからの秒。optimizer の歩幅）を返す
- `delta = r + lam` を credit とし、`pc.functional.vjp(f, x, lambda _: delta, params)` → `(mu, g_in, g_theta)`。`vjp` は入力を detach する（局所性）
- `g = sum(self.next.recv().values())`（MultiPort で受けた後段の credit の和）、`pc.functional.relax_(self.r, self.lam, g, tick.dt, self._tau)`
- 送る活動は **`mu`**（緩和の前の順伝播の値。`Activity.h = mu`）。平衡点で `r → 0`、`lam` が credit を持つ（test_model の「lam が BP の credit と一致」オラクル）
- 学ぶとき `optimizer.step_([g / B for g in g_theta], tick.elapsed)`（バッチ平均）。新しいフレームの更新で `r`, `lam` を 0 に戻す
- `learn_interval=25e-3`（既定。`None` で毎更新）: 書き込みから 25 ms 後、1 フレーム 1 回学ぶ。`math.inf` で学ばない（テストで平衡点を見るとき）
- `OptimConfig.tau`（既定 1.0）が SGD の時定数。学習率は `elapsed / tau`。M13 は `tau = FRAME / 0.03`、`warmup=300`

### `pc.Input`（mujou）
観測を `Activity{h=x, y, mask, reset, time}` にして `next`（MultiPort、複数の接続可）へ送る。後段から返る credit は使わない（観測は固定）。**要確認**: `Input.next` の recv spec が `x` の shape で、`Compare.z`・`Dynamics.z` の送り spec（`act_spec.field("h")`）と合うこと。

### `pc.Layer`（mujou）= `g1`, `z`
エンコーダの各段。`act_spec` は前段のもの。`z` の `mu` が $z_k$。`z.next` は MultiPort で、dynamics・sigreg・compare・reconstruct から credit を受け、和で緩和する。

### `Dynamics`（nodes.py:78-230）
$s_k = \mathrm{minGRU}(z_{k-1}, s_{k-1})$。

- コンストラクタ: `Dynamics(module: MinGRU, *, act_spec, tau, optimizer, learn_interval, open_loop=0.0, open_cycle=20, residual=False, trace=False, device="cpu")`
- ポート: `z = Port(send=z_spec, recv=act_spec)`（**送るのは常にゼロ**: 過去は定数なので `g_in` を返さない）、`next = MultiPort(send=self.act_spec, recv=s)`、`open_loop>0` で `z_hat = Port(recv=prediction_spec(act_spec))`、`residual` で `base = Port(send=act_spec)`
- `self.act_spec = pc.msg.activity_spec(s, act_spec.field("y"), batch)`（`h` が `(*batch, d_s)`）
- 状態（`register_state` は bool/int のもの）: `r`, `lam`, `z_prev`, `s_prev`, `z_last`, `mu`, `has_prev`(B), `predictions`(B, int), `z_hat_last`, `z_hat_ok`(B)
- `_new_frame(reset)`（nodes.py:211）: `r, lam = 0`、`predictions` を進め、`is_open` なら `z_prev := z_hat_last`（自分の予測）、そうでなければ `z_prev := z_last`（前のフレームで最後に受けた z）。reset した列は `z_prev` を据え置き `mu := 0`、`s_prev := mu`、`has_prev := ~reset`。trace があれば `trace.step_(z_prev, s_prev)` と `trace.reset_(reset)`
- 各更新: `mask = act["mask"] & has_prev`、`delta = where(mask, r + lam, 0)`、`mu = module(z_prev, s_prev)`（has_prev でない列は 0）。trace なしなら vjp で `g_theta`、ありなら `trace.grads(delta)`。`next.send({**act, "h": mu, "mask": mask})` — **`time` は今のフレームの `act["time"]` のまま**なので Compare で $z_k$ と揃う
- 引き継ぐ $s_k$ は `mu`（`mu + r` でない）。temporal PC の観測による補正は入らない（PLAN.md:117）
- `module.gate.out_features` を `d_s` として読む（MinGRU 専用）

### `DynamicsLayer`（nodes.py:233-325）
$y_k = x_k + \mathrm{minGRU}(\mathrm{GELU}(x_k), h_{k-1})$（`ResidualMinGRU`）。入力が同じフレームのものなので `g_in` を前段へ返す（`prev = Port(send=s, recv=act_spec)`）。動画の最初のフレーム（`first = reset`）は `x` を素通し。`h_last = mu - x`。trace は `MinGRUTrace(module.cell)` を `gelu(x)` で進める。`act_spec` は入力と同じ。

### `Compare`（nodes.py:345-452）
出力ノード。$\hat z_k = W_o s_k$（`residual` なら `base + W_o s_k`）と $z_k$ を比べる。

- `Compare(readout: nn.Linear, *, s_spec, z_spec, kappa, tau, optimizer, learn_interval, target_weight=1.0, residual=False, device)`。型注釈は `nn.Linear` だが中身は `self.module(s)` を呼ぶだけで任意の `nn.Module` で動く
- エネルギー `compare_energy(readout, s, z, kappa, base) = (1 - kappa) * ((z_hat - z)**2).mean(-1)`（サンプルごと。**最後の 1 軸の平均**なので画像の shape ではそのまま使えない）
- `mask = s.mask & z.mask`、`s.time != z.time` なら全部 False、`residual` なら `base.mask` と時刻も要る
- `_vjp`: `s`, `z` を detach→requires_grad、`where(mask, energy, 0).sum()` を `autograd.grad`（`# mujou: local-autograd`）→ `g_s`, `g_z`, `g_theta`
- `s.send(g_s)`、**`z.send(target_weight * g_z)`**、`next.send({"y_hat", "y": z, "mask", "time"})`、`error.send(mean of (y_hat - z)^2 over masked)`
- `y_hat` は `no_grad` で毎更新計算（学習の前の値）

### `SIGReg`（exp/lib/sigreg.py:113-216）
パラメータなし。`buffer (capacity, D)` に過去の z（`every` フレームに 1 回、`insert_` で埋めてから一様に置換）、`directions (M, D)` を毎フレーム生成器から引き直し、`ecf_sums` でバッファ側の特性関数の和を前計算。`count + B >= min_samples(16)` かつ `mask` のとき `kappa * B * dS/dz_b` を `prev` に返す。z が変わらない間は `g` をキャッシュ。**スケールの注意**: パラメータの上では B 個のサンプル項の和として効くので、同じ kappa でも B が大きいほど強い（per-sample で `kappa B / (1 - kappa)`）。関数版 `epps_pulley`, `sigreg(z, buffer, directions)`, `sigreg_from_sums` は BP 版とオフラインでも使う。

### `Reconstruct`（nodes.py:469-533）
M13 の再構成の項。`recon_energy = weight * mean_pixels((dec(z) - x)^2)`。`z` に credit を返し、`x`（Input）には返さない。**M14〜M16 では使わない方針**（MILESTONES.md:69）。

### `is_open` / `open_mask`（nodes.py:58-72）
動画の `count` 番目の予測が自分の予測から始まるか。各 `cycle` の後ろ `open_loop` の割合。

---

## 3. `ModelConfig` の関係するフィールド（config.py:72-119、既定値）

| フィールド | 既定 | 意味 |
| -- | -- | -- |
| `hidden` | 0 | 段 0（線形回帰）専用 |
| `d_z` | 64 | z の次元。**`dynamics_modules` の入力次元と readout の出力次元もこれ** |
| `d_s` | 256 | minGRU の状態の幅（各動力学ノード） |
| `kappa` | 0.05 | SIGReg の重み。Compare のエネルギーにも `(1-kappa)` で掛かる |
| `stop_target` | False | 目標 z への credit を止めるか |
| `target_weight` | 0.0 | `stop_target` のときだけ効く割合。`model.target_weight(m) = m.target_weight if m.stop_target else 1.0`（model.py:216）。**`stop_target=False` だと target_weight の値に関係なく 1.0** |
| `z_norm` | False | Z の最後に `nn.RMSNorm(d_z, elementwise_affine=False)` を足し、$\lVert z\rVert = \sqrt{d_z}$ にする |
| `width` | 32 | 合成動画の CNN の最初のチャンネル数 |
| `blur` | 0.0 | 入力のガウスぼかし（`GaussianBlur`） |
| `open_loop`, `open_cycle` | 0.0, 20 | 自分の予測から予測する割合 |
| `residual` | False | $\hat z_k = z_{k-1} + W_o s_k$、$W_o$ を 0 初期化（最初は「写し」） |
| `enc_nodes` | 2 | 合成動画は 2 固定（`g1`, `z`）。EgoDex は 1〜3 |
| `dyn_nodes` | 1 | `Dynamics` + `DynamicsLayer` × (n-1) |
| `recon` | 0.0 | 再構成の重み |
| `trace` | False | minGRU の適格度トレース（RTRL） |
| `directions` | 256 | SIGReg の方向の数 |
| `buffer` | 255 | SIGReg のバッファ |
| `buffer_every` | 1 | |

M13 の設定（easy.py）: `SMALL = ModelConfig(kappa=0.13, z_norm=True, open_loop=0.5, open_cycle=20, buffer=1023)`、`BIG = replace(SMALL, width=64, d_s=512, dyn_nodes=3)`。optimizer は SGD `tau=FRAME/0.03`, `warmup=300`。`config.dynamics_optim` で動力学と readout だけ別の optimizer にできる。

---

## 4. 適格度トレース（`exp/lib/mingru.py`）

- `MinGRU(d_z, d_s)`: `gate = Linear(d_z, d_s)`（$W_u$）、`candidate = Linear(d_z, d_s)`（$W_s$）、`s = (1-u) s_prev + u c`。**gate も candidate も入力 z だけの関数で、`s_prev` に依らない**（対角の線形回帰に入力依存のゲートが付いたもの）。1 ノードの minGRU + 線形 readout は「入力依存の時定数をもつ漏れ積分器の束」で、非線形な時間の計算（跳ね返り）は `dyn_nodes > 1`（層の間の GELU）か非線形の readout に頼る。M14 段 1（位置だけから速度を推す）はこの表現力の点で注意
- `ResidualMinGRU(d)`: `y = x + MinGRU(gelu(x), h_prev)`
- `MinGRUTrace(cell, batch)`: $e_k = \partial s_k / \partial\theta$ を自分の状態を通して（入力は固定）厳密に持つ。`step_(x, s_prev)` は**現在の重み**で進める（RTRL の通例の近似。重みが動くと厳密な BPTT ではない）、`reset_(which)`、`grads(delta) -> [gW_u, gb_u, gW_s, gb_s]`（バッチで和。パラメータの順は `cell.parameters()` と同じ）
- メモリ: `2 * B * d_s * d_x + 2 * B * d_s` 個の float。BIG で B=128: `Dynamics`（d_x=64, d_s=512）≒ 34 MB、`DynamicsLayer` 1 つ（d_x=d_s=512）≒ 268 MB。M14 の観測（d_x=2〜12）なら Dynamics 側は無視できる
- `Dynamics` は `_new_frame` で `trace.step_(z_prev, s_prev)`、学ぶ更新で `g_theta = trace.grads(delta)`（delta は今の `r + lam`）。trace のとき vjp には params を渡さない（`direct = learn and trace is None`）
- オラクル: `tests/lib/test_mingru.py::TestMinGRUTrace`、`tests/video/test_model.py::TestTrace`（K フレームの後、PC は `(r+lam)·s_K`、BP は loss の、z を固定した BPTT の勾配と一致。`monkeypatch.setattr(model, "make_optimizer", ...)` で optimizer を記録し、**作られた順の添字**（PC は `recorders[2]` が dynamics）で読むので、組み立ての順を変えるとテストも直す）

---

## 5. `WorldBP`（bp.py:53-331）と `WorldBPNetwork`（bp.py:334-388）

BP の基準。**1 ノード**（`Input → WorldBP`）で全モジュールを autograd で学ぶ。PC-ALM と同じ `LearnClock`・同じメモリ（$z_{k-1}, s_{k-1}$ は detach、勾配は 1 フレームで止まる）・同じバッファと方向・同じ optimizer。

- `WorldBP(modules=(encoder, step: DynamicsStack, readout: nn.Linear), *, act_spec, kappa, directions, capacity, tau, optimizer, learn_interval, min_samples=16, every=1, seed=0, target_weight=1.0, open_loop=0.0, open_cycle=20, residual=False, decoder=None, recon=0.0, trace=False, dynamics_optimizer=None, device="cpu")`
- `self.readout = nn.Sequential(Top(step.d_s), readout)`（多ノードの動力学の出力 $y_L = s_1 + h_2 + \dots$）
- `_params = [*encoder, *step, *readout, *decoder]` の順。`dynamics_optimizer` があれば `slice(n_enc, n_enc + n_dyn)` を別 optimizer に
- `_new_frame(act)`（bp.py:207）: バッファへ前の z を入れ、方向を引き、メモリを進め、**`no_grad` で `z = encoder(x)`、`s = step(z_prev, s_prev)`、`z_hat = readout(s)`** を計算して `next` へ送る（prequential）
- `on_update`: `tick.learn and mask.any()` で `_grads` → `optimizer.step_`（勾配は和ではなく `loss` が既にバッチ平均）
- `_loss`（bp.py:306）: `z = encoder(x)`、`s = _state(ys)`、`target = w z + (1-w) z.detach()`（`target_weight`）、`compare_energy(readout, s, target, kappa, base)` + 再構成、`where(mask).sum() / B`、`count + B >= min_samples` なら `+ kappa * sigreg(where(mask, z, z.detach()), buffer[:count], directions)`
- `_grads`: trace なしは `autograd.grad(loss, _params, materialize_grads=True)`。trace ありは各層の出力 `ys` への勾配を取り、`trace.grads(d)` で動力学の勾配を作る（PC と同じ近似）
- `WorldBPNetwork(config)`: `torch.manual_seed(seed)` → `encoder = Sequential(*encoder_stages(config))` → `standardize_` → `dynamics_modules(m)` → readout（residual で 0）→ `DynamicsStack(first, layers)` → decoder。**`model.Network` と同じ順で乱数を消費するので、同じ seed で同じ初期値**（`test_bp_starts_from_the_same_weights`）。属性 `tau`, `input`, `world`, `prev`, `next`
- `run.snapshot(net)`（run.py:86）が `WorldBPNetwork` なら `net.world.encoder/step/readout/residual` を deepcopy
- `bp_chain(*modules, loss, **kwargs)`: 段 0 の `pc.Chain` 用

**`DynamicsStack`（model.py:227）** は動力学ノードを 1 モジュールにしたもの。状態は `(s_1, h_2, ..., h_L)` を連結した `L d_s`、`d_state` プロパティ。`Top(d_s)` が和を取る。評価（`evaluate.run_states` / `rollout_errors`）と BP 版の共通の部品。

---

## 6. 汎用か、画像専用か

| 汎用（観測がベクトルなら何でも） | 画像専用 |
| -- | -- |
| `nodes.Dynamics`, `DynamicsLayer`, `Compare`, `compare_energy`, `prediction_spec`, `is_open`, `open_mask` | `model.GaussianBlur`, `encoder_modules`, `egodex_stages`, `encoder_stages`, `image_shape`, `calibration_images`（`Video` を描画）, `standardize_`（Conv/Linear 前提。**Identity では `layers[-1]` で IndexError**） |
| `lib.mingru.*`, `lib.sigreg.*`（z がベクトルなら） | `model.decoder`, `nodes.Reconstruct`, `recon_energy` |
| `model.DynamicsStack`, `Top`, `dynamics_modules`, `target_weight` | `Network.__init__` の Input の `x` spec（`image_shape`）と `encoder_stages` 呼び出し |
| `bp.WorldBP` の動力学・readout 部分（`encoder` 引数は任意の `nn.Module`、パラメータ 0 個でも動く） | `WorldBPNetwork` の encoder 作りと `standardize_` |

`ModelConfig.d_z` が「動力学の入力次元 = readout の出力次元 = SIGReg の次元」を兼ねている点が、エンコーダなしで最も素直に流用できる鍵。

---

## 7. M14: エンコーダのない Network（Input obs → Dynamics → Compare）

MILESTONES の要求: `Input`（観測）→ `Dynamics`（minGRU、M13 の大・小）→ `Compare`（$W_o$ で次の観測を予測）。エンコーダ・SIGReg・`z_norm`・再構成なし。目標は観測なので目標への credit なし。trace あり/なし。PC-ALM と `WorldBP`。

必要な変更（ノードは**変更不要**、組み立てだけ）:

1. `Input(x=TensorSpec((*batch, d_obs)), y=TensorSpec((*batch, 1)), ...)`。`d_obs` = 4（段 0）, 2（段 1）, 6（段 2、玉 3 つの位置）。`d_z := d_obs` にすれば `dynamics_modules(m)` と `readout = Linear(d_s, d_obs)` がそのまま使える
2. `Dynamics(step, act_spec=self.input.act_spec, ...)`、`Compare(readout, s_spec=top.act_spec, z_spec=self.input.act_spec, kappa=0.0, target_weight=0.0, ...)`
3. 接続: `input.next.connect(dynamics.z)`、`input.next.connect(compare.z)`、`top.next.connect(compare.s)`、(open_loop) `compare.next → dynamics.z_hat`、(residual) `dynamics.base → compare.base`。Input は返ってくる credit（Dynamics はゼロ、Compare は `target_weight * g_z`）を使わないはず（要確認）。念のため `target_weight=0`
4. **`kappa=0`**: `compare_energy` の `(1 - kappa)` を 1 にする。SIGReg ノードは作らない
5. `standardize_`・`calibration_images` を呼ばない。乱数の順は「`manual_seed` → dynamics_modules → readout」で PC と BP を揃える
6. `encoder_names = ()` にすると `Snapshot.of` の `nn.Sequential()`（空）が恒等写像になり、評価がそのまま動く。`self.z` を前提にした `latency.py` は壊れる
7. `WorldBP((nn.Identity(), DynamicsStack(...), readout), kappa=0.0, target_weight=0.0, ...)`。kappa=0 でも `sigreg` を毎回計算する（無駄。`count + B >= min_samples` の分岐をフラグで切るとよい）。`directions`/`capacity` は小さく
8. 観測の標準化（評価クリップの平均と標準偏差）はホスト／ストリーム側で行う。ノードには入れない
9. `residual=True`（$x_{k-1} + W_o s_k$、最初は写し）は M14 で特に自然。段 1（位置だけ）では s が速度を持つ必要がある
10. 評価: `evaluate.rollout_errors(z=obs, step, readout, d_s, residual)` がそのまま観測空間の rollout になる。probe・実効ランクは意味がない。`run_video` は `f.image` を書き `d_state = d_s * dyn_nodes` を使う → 観測を書くホストが要る
11. 平衡点のオラクル（test_model.py:87 `test_lam_is_the_credit_of_bp_at_the_fixed_point`、:552 の B 本版、:356 の多ノード版）は、エンコーダとSIGReg を抜いた形でそのまま移せる（1 フレームを長く保持し `learn_interval=inf`、各ノードの `lam` = autograd の `-dL/dh`）

段 3（画像を観測にして誤差も画素の上）: `MinGRU` の `Linear` と `compare_energy` の `mean(-1)` は最後の 1 軸だけを見るので、**観測を `(B, 1024)` に平らにして渡す**のが最小の変更（readout = `Linear(d_s, 1024)`）。CNN の入力・出力を挟むなら Dynamics の前段に「入力モジュール」を足すか、Compare の readout を任意モジュール（デコーダ）にして energy を `flatten(-3)` で平均する。

速度の目安: 30 分 = 54000 フレーム × 約 17 ステップ ≒ 92 万ステップ。各ノードは学ばない更新でも毎ステップ `vjp` で `mu` を計算する。M14 の小さい次元では Python のオーバーヘッドが支配的になり、GPU より CPU が速い可能性がある（測ること）。`balls.Video.next()` は毎フレーム画像を描く（`render`）ので、状態だけ要る M14 ではストリーム側でも無駄になる。

---

## 8. M16: 出力が動力学にとって定数のエンコーダ

MILESTONES の要求: 動力学の予測誤差はエンコーダに credit を送らない（目標の z も入力の z も定数）。

今のコードで既に満たしている部分:

- **入力の z**: `Dynamics.z` は常にゼロを送る（nodes.py:165, `z_prev` は記憶した定数）
- **目標の z**: `Compare.z.send(target_weight * g_z)`。`ModelConfig(stop_target=True, target_weight=0.0)` で 0（**`stop_target=True` を忘れると weight 1**）
- `residual` の `base`・`open_loop` の `z_hat` も credit を運ばない

したがって M16 の PC-ALM 版は「エンコーダ部分 → `z.next` → {動力学部分}」で、動力学部分は M14 のエンコーダなし Network と同じものを `Input.next` の代わりに `Z.next` につなげばよい。設計の勧め:

- 動力学の部分（Dynamics + DynamicsLayer + Compare）を「上流の活動ポートと act_spec を受け取って組み立てる関数」にし、M14（上流 = `Input`）と M16（上流 = エンコーダの出力ノード）で共有する
- Compare に `stop_target`/`target_weight` の二重のフラグを持たせるより、「z を受けるだけ（送らない or 常にゼロ）」の形を明示した方が読みやすい。Compare の `kappa` は SIGReg と無関係にする（`(1-kappa)` の結合を外す）
- エンコーダの学習は M15 の JEPA（V-JEPA 風、N フレーム、SIGReg、ViT 見込み）のノードが担う。今の `pc.Layer` の CNN + SIGReg ノードだけでは、M13 で w=0 のとき「エンコーダが SIGReg だけで動き、動力学が発散」した（memory/agents/orchestrator/m13-easy-video.md）。z の尺度を固定する `z_norm`（RMSNorm）は M16 で有用な可能性がある
- エンコーダの z は 1 フレームの中でも学習で変わる。Dynamics は `_new_frame` で「前のフレームで最後に受けた z」を `z_prev` にし、Compare は各更新の z を目標にする。エンコーダから z が届くまで数周期の遅れがあるが、`time` で揃えるので正しさは保たれる（予測が `Host.show` の 1 フレームに間に合うかは見る）
- BP 版: 今の `WorldBP` は 1 ノードで loss を結合しているため、`target_weight=0` でもエンコーダは SIGReg（と recon）で学ぶ。M16 は credit が流れないので、**「エンコーダの BP ノード（JEPA の損失）」と「動力学の BP ノード（= encoder を `nn.Identity()` にした `WorldBP`、入力が z）」の 2 ノードに分ける**のが素直で、M14 の BP 版をそのまま流用できる
- 判定のデコーダは系の外（`decode.py` と同じく、チェックポイントごとにエンコーダを止めて後から学ぶ）。`Snapshot(encoder, step, readout, residual)` の形は M16 の評価にも使える

---

## 9. 再編で気をつけること（まとめ）

- PC 版と BP 版の**初期値の一致**は「`manual_seed` の後に同じ順でモジュールを作る」ことに依存。再編では 1 つの `build_modules(config)` を両方から呼ぶ形にすると壊れにくい
- テストは `make_optimizer` を monkeypatch し、**optimizer の作られた順**で読む（`recorders[2]` など）
- 名前で `getattr` する作り（`encoder_names`, `dynamics_names`）を利用側（evaluate, latency, run, tests）が前提にしている
- `register_state` に入れるのは Python のスカラーと bool/int テンソル（`has_prev`, `predictions`, `z_hat_ok`, `first`, `count`, `seen`, `frames`, `cached`, `started`）
- 局所性: 自作ノードの autograd は自分のモジュールと detach した入力だけ（`# mujou: local-autograd` のコメント）。ノードをまたぐ `.backward()` は禁止。`mujou/.claude/skills/locality-and-async/SKILL.md` と突き合わせる（このリポジトリでは hook が効かない）
- デバイス: モジュールは CPU で作って移す。trace のテンソルは `w.new_zeros` でモジュールのデバイスに乗るので、**モジュールを移してからノードを作る**（今の順）。WorldBP と SIGReg は `Node(seed=...)` の `self.generator` を使う
- `y` の spec（`(*batch, 1)` のゼロ）は使っていないが Input と activity_spec が要求する
- run.py の `torch.set_num_threads(1)`（main）。複数ジョブを 1 マシンで回す前提
