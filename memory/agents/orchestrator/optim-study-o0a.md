# O0 の A（optimizer 群・dec_scale・勾配の計測・grad_table・bench）

仕様: exp/vision/plans/optim-study.md。腕と予想: exp/vision/README.md の V11・V12（編集しない）。

## 段階 1: 計画

### 公開 IF

- `exp/lib/optim_zoo.py`: `Nesterov(tau, tau_m)`、`RMSPropCentered(tau, tau_v, eps)`、
  `Adadelta(tau, tau_v, eps=1e-6)`、`Rprop(tau)`、`Muon(tau, tau_m, tau_v, eps, weight_decay)`、
  `newton_schulz(x, steps=5)`。すべて `mujou.pc.optim.Optimizer`
- `exp/lib/grad_stats.py`: `GradStats(inner, node, names, store=bf16, every=1, tensors_every=60,
  anchors=(1,10,100), spans=(0.1,1,10), keep=16)`、bound は `drain() -> list[dict]`。
  `residual_share(G, g)`（ρ。OGD でも使える）
- `exp/vision/schema/optim.py`: kind 5 つ、`_USED` 拡張、`dec_scale`（全 kind、> 0）
- `exp/vision/schema/measure.py`: `grad_stats: bool = False`
- `exp/vision/pc.py`: `make_optimizer(c, scale=1.0)`、`make_optimizers(conf, mods) -> dict[node, Optimizer]`
  （dec_scale と grad_stats）、`VAEPC(optimizer=Optimizer | Mapping[str, Optimizer])`、`VAEPC.grad_rows()`
- `exp/vision/loop.py`: `Built.rows() -> Mapping[file, list[row]]` を毎ユニット後に書く。
  `RECORDS = ("grad.jsonl",)` を metrics に足す（再開で切り戻す）
- `exp/vision/grad_table.py`: `table(rows) -> list[dict]`、`main(run_dirs)`
- `exp/vision/bench.py`: `--models vit_s,...`、行に `optim`

### 解釈（仕様の読み）

- rmsprop_centered: 仕様の $\hat v - \hat m^2$ そのままだと 1 歩目で 0（分散 0）になり g/eps で発散する。
  $(v - m^2)/c$（$c$ は質量。$\hat v - c\,\hat m^2$）にする。$c = 1$ で torch の centered と一致、
  1 歩目は RMSProp の $\hat v$ と同じ大きさ。CS で $v - m^2 \ge 0$
- adadelta: torch と同じく $u$ は lr を掛ける前の方向 $D$ の 2 乗。$\hat u$ は前の歩の質量で割る（初歩は 0）
- nesterov・muon: torch の buf は偏りの補正なし。$c_m = 1$ を load して torch と完全一致をテスト
- muon: 行列は ndim >= 2（ViT-S では Linear と Conv2d の重みだけであることをテスト）。AdamW の部分は
  `pc.optim.AdamW(tau, tau_m, tau_v, eps, weight_decay)`
- rprop: 一歩の大きさの更新（1.2 / 0.5）はステップごと。$\Delta\theta = -\text{elapsed}\cdot s\cdot\mathrm{sign}$
- 勾配の計測の時刻: ノード自身の elapsed の和。事象（ログ 1 s、テンソル 60 s、錨 L、保持 s）は
  `t + elapsed/2 > due` の歩（due に最も近い歩）で起こす。最初の歩で錨と保持を置く
- テンソルごとの $\lVert\Delta\theta\rVert/\lVert\theta\rVert$ はその歩の $\Delta\theta$（60 秒ごとの歩）
- age の bin: 0.1・1・10・100 を中心とする幾何の区間（端は $10^{k+0.5}$）
- 窓: 0〜300 は t <= 300、300〜1800 は 300 < t <= 1800
- 再開: 計測の状態（錨・保持・前の勾配）を optimizer の状態に入れ、grad.jsonl も切り戻す。
  stop/resume で grad.jsonl のバイトが一致することをテスト

### メモリ（ViT-S、8.29M パラメータ）

52 本（前 1 + 錨 3 + 保持 3×16）× 8.29M × 2 B（bf16）= 0.86 GB。ρ の計算で一時的に 16 × D の fp32
（最大のノード enc4 2.89M で 185 MB）。

## 段階の記録

- 段階 2: torch のオラクル（$c = 1$ を load して完全一致、1 歩目は手計算）、手計算の cos・ρ、ビット一致、
  再開で grad.jsonl のバイト一致、dec_scale のノードごとの tau、bench の --models
- 段階 3: Rprop の倍率を `torch.where(…, 1.2, 1.0)` で作ると float32 になり、torch と 1e-9 ずれた。
  `ones_like(s).masked_fill_` で dtype を揃えた
- 段階 3: Walking Tours のデータが手元に無いので、tiny の CPU run は `RandomStacks` で loop を通して確かめた
  （5 種 + sgd、grad_stats あり、3 秒、すべて exit 0、grad.jsonl に 7 ノード × 3 行）
- 段階 4: 自己レビュー。ノードをまたぐ計算なし（`.backward(` / `.grad` の grep で 0 件）。
  `dec_scale` と `grad_stats` が config に増えたので、同じ設定でも group の hash は V8 と変わる（比べるのは sq の値）
- 却下: 計測の状態を再開の checkpoint から外す案（checkpoint は小さいが、再開後の錨と保持がずれる）
