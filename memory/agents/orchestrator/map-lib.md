# 地図: optimizer と雑多なスクリプト（再編のため）

対象: `exp/lib/optim.py`, `exp/lib/linear_stream.py`, `exp/lib/__init__.py`, `exp/video/{stage0,stage1,stage2,stage3,stability,latency,frozen,collapse}.py`。
mujou の参照は submodule のピン `881c4ee` の `src/mujou/pc/optim.py`（この checkout の `mujou/` は空。`~/Projects/mujou` で `git show 881c4ee:...` で読んだ）。

## 0. 結論（再編の判断）

| ファイル | 用途 | M14-M16 で要るか |
| -------- | ---- | ---------------- |
| `exp/lib/optim.py` | 時間で積分する optimizer 群（`pc.optim.Optimizer` の実装） | **残す**。M13 の結論は「残りの律速は optimizer」。`Warmup`（easy.py が使用）と `AdamDirection`+`Grafted`（M13 で試行）は M14 の探索空間に入る。`Whitened`/`Projected`/`Orthogonalized` は M10 で SGD を超えなかったが小さくテスト済み。optuna の探索の選択肢として残すか、消すかは方針次第。`Correlation` は診断専用（下記の gotcha 参照）で opt-in にするか消す |
| `exp/lib/linear_stream.py` | 段 0（M6）の線形回帰ストリーム | **消してよい**。使用者は `run.run_linear` と `stage0.py` だけ。M14 段 0（玉の状態）が安価な試験台を置き換える |
| `exp/lib/__init__.py` | docstring 1 行のみ | 空でよい |
| `exp/video/stage0.py` | M6 の掃引 | **消す**。ただし `RUN_MODULE="exp.video.run"`, `SEEDS=(0,1,2)` を easy.py・stage1/2/3・stability・collapse・`tests/video/test_run.py:11` が import している。移設が要る |
| `exp/video/stage1.py` | M7 の掃引 | 消す（`summary` を collapse.py が import） |
| `exp/video/stage2.py` | M8/M10 の optimizer 比較と H2-H4 の判定 | 消す（`tests/video/test_stage2.py` も一緒に消す） |
| `exp/video/stage3.py` | M12 の EgoDex | 消す（M15 は EgoDex を使わないと明記） |
| `exp/video/stability.py` | M11 の掃引群（14 種類のフラグ） | 消す（`summary`/`KEYS` を stage3 が import） |
| `exp/video/latency.py` | フレーム内の credit の到達度の診断（EgoDex のネット） | 消す。考え方（ノードの `r+lam` と autograd の credit の比較、Gauss-Seidel/Jacobi）は M16 で多ノードを組むときに再利用しうるが、コードは EgoDex・SIGReg・`Network` に密結合 |
| `exp/video/frozen.py` | M11 の凍結エンコーダ + BP | 消す（M16 の「エンコーダに credit を送らない」は凍結とは別物） |
| `exp/video/collapse.py` | M9 の崩壊の掃引 | 消す |

旧スクリプトの共通パターン（再編で置き換える対象）: `configs() -> list[Config]` を Python で組み、`exp.lib.sweep.run_all(RUN_MODULE, configs, out, jobs)` で 1 run 1 サブプロセス（`ThreadPoolExecutor(jobs)`、`out/<name>.json` に設定を書き `python -m exp.video.run --config ... --out ...`、stdout の最終行が run dir、そこに `done` を touch）。完了判定は `out.glob(f"*-{name}/done")`。集計は `metrics.jsonl` を読み、run dir 名 `<date>-<time>-...-<name>` から `split("-", 3)[3]` で名前を取る（日付部の `-` の数に依存した脆い取り方）。`device` は設定に直書き（`"cuda:0"/"cuda:1"` を手で振り分け）。→ hydra multirun + optuna sweeper + fleet launcher でまとめて置換できる。

## 1. mujou 側の optimizer の契約（`mujou/src/mujou/pc/optim.py`, ピン 881c4ee）

- `class Optimizer(ABC)`: 設定値だけを持つ（state なし、ノード間で共有可）。`bind(params: Iterable[Tensor]) -> BoundOptimizer` が抽象。
- `class BoundOptimizer(Stateful)`: 1 ノードのパラメータに bind されて state を持つ。
  - `step_(grads: Sequence[Tensor], elapsed: float) -> None`: in-place、勾配記録なし。`grads` は bind した順・同 shape、**バッチ平均済み**（ノードが `1/B` を掛ける）。`elapsed` は `Tick.elapsed`（前回パラメータが動いてからの時間、上限あり）。
  - `state_dict()` は optimizer の tensor とメモリ共有、`load_state_dict` は copy-in。ノード属性として持てばノードの state に自動で入る。
- 組み込み: `SGD(tau)`: `p -= (elapsed/tau) g`。`AdamW(tau, tau_m, tau_v, eps=1e-8?, weight_decay)`: 時間で厳密に積分した moment と bias 補正 `c_m, c_v`（`tau dc/dt = 1-c`）。
- `torch.optim` は使わない（ステップを数える／free-threaded で GIL を戻すため）。
- 差し込み口: `pc.Layer(..., optimizer=...)` は `layer.py:204` で `self.optimizer = optimizer.bind(self._params)`（requires_grad のものだけ）、`layer.py:231` で `self.optimizer.step_([g / self._samples for g in g_theta], tick.elapsed)`。`pc.Output` も同様（`output.py:124,168`）。`pc.Chain(..., optimizer=..., learn_interval=...)` は全層に同じ設定を渡す。
- 実験側の差し込み: `exp/video/config.py:257 make_optimizer(c: OptimConfig) -> Optimizer`。使用者は `model.py:301`（`Network` の各ノード。`dynamics*`/`compare` は `config.dynamics_optim or config.optim`）、`bp.py:370,383`（`WorldBP(optimizer=..., dynamics_optimizer=...)`、`bp.py:130-137,198-205` で dynamics と readout の params を別に bind）、`run.py:54`（`run_linear`）。

## 2. `exp/lib/optim.py`（591 行）の詳細

共通: 方向 $D$ を勾配から作り、$\Delta W = -\frac{\text{elapsed}}{\tau} D$（`DirectionOptimizer`）、または grafting で $\Delta W = -\frac{\text{elapsed}}{\tau}\lVert G\rVert_F \frac{D}{\lVert D\rVert_F}$（`Grafted`）。2 次元以上の重みは行列 `(d_out, d_in)`（conv は `(c_out, c_in k^2)`、`_matrix(t) = t.reshape(t.shape[0], -1)` :126）、1 次元（bias）は `D = G`（Whitened/Projected/Orthogonalized の場合）。

`__all__`: `LAGS, AdamDirection, Correlation, BoundDirection, DirectionOptimizer, Grafted, Orthogonalized, Projected, Warmup, Whitened`。

### 基底
- `BoundDirection(Stateful)` :40 — 抽象 `direction(grads, elapsed) -> list[Tensor]`（state を elapsed 秒進め、各パラメータの方向を返す）。
- `@dataclass(frozen=True, kw_only=True) DirectionOptimizer(Optimizer)` :51 — フィールド `tau: float`（必須、SGD と同じ秒の時定数）。抽象 `bind_direction(params) -> BoundDirection`。`bind()` は `_Stepped(params, direction, tau, graft=False)`。
- `@dataclass(frozen=True, kw_only=True) Grafted(Optimizer)` :72 — フィールド `inner: DirectionOptimizer`。`bind()` は `_Stepped(..., inner.tau, graft=True)`。**`Grafted` 自体は DirectionOptimizer ではないので入れ子にできない**。SGD/AdamW（mujou の）は graft できない。
- `_Stepped(BoundOptimizer)` :86 — `step_`: 勾配数の不一致で `ValueError`、`elapsed == 0` なら何もしない（direction の state も進めない）、`eta = elapsed/tau`、`torch.no_grad()` 下で direction を取り、graft なら **パラメータ tensor ごとに**（docstring は「layer by layer」だが実装は tensor 単位、bias も別に）`d *= ||g||/||d||`（`||d||==0` ならその tensor を飛ばす）、`p.sub_(d, alpha=eta)`。state は direction の state そのもの。

### 内部ヘルパ
- `_ema_(avg, target, elapsed, tau)` :131 — `avg.lerp_(target, 1-exp(-elapsed/tau))`、target を保った厳密解。
- `_Factors(Stateful)` :140 — 各行列について `L = EMA(G G^T)`（`(d_out,d_out)`）、`R = EMA(G^T G)`（`(d_in,d_in)`）、およびそこから計算した行列 `of_l, of_r`。使わない側・非行列は空 tensor `torch.zeros(0)`。`since = inf` から始まり、`due(elapsed)` で `since += elapsed` し `>= refresh` で 0 に戻して True（**最初のステップで必ず計算**）。state: `l, r, of_l, of_r, since`。
  - **コスト注意**: L/R は O(d_out²)+O(d_in²) のメモリ。d_s=512 の minGRU や conv で大きくなる。
- `_eigh(a)` :204 — float64 で `torch.linalg.eigh`。LAPACK の収束失敗時は有限なら 1e-12 相対の対称ノイズ（seed 0 の Generator）を足して再計算、非有限なら raise。**GPU でも float64 eigh**（遅い）。stage2 は `refresh=5.0` で計算時間を抑えた。
- `_inverse_root(a, power, eps)` :253 — `(a + eps*λmax I)^-power`。`λmax==0`（まだ勾配なし）は単位行列。
- `_projector(a, rank, eps)` :362 — `I - U U^T`、U は上位 `rank` 個かつ `λ > eps λmax` の固有ベクトル。

### `Whitened(DirectionOptimizer)` :226
$D = (L+\epsilon I)^{-a} G (R+\epsilon I)^{-b}$。フィールドと既定: `tau`（必須）, `tau_l: float|None = None`, `tau_r: float|None = None`（None はその側を白色化しない）, `a=0.25`, `b=0.25`, `eps=1e-4`（λmax 比）, `refresh=1.0`（秒）。L/R は**現在の G を含めて**更新してから（`update_` → `due`）根を計算。`a=b=0` は SGD、`a=b=1/4, tau→0` で `U V^T`（Muon）。

### `Orthogonalized(DirectionOptimizer)` :299
`eps=1e-4`。SVD で `D = U V^T`（`σ > eps σ0` のみ）、state なし。Whitened の極限。1 次元は `D=G`。

### `Projected(DirectionOptimizer)` :341
$D = (I - U_L U_L^\top) G (I - U_R U_R^\top)$。フィールド: `tau`, `tau_l=None`, `tau_r=None`, `rank=4`, `eps=1e-4`, `refresh=1.0`。**射影子は現在の G を足す前の L/R から**（順序が Whitened と逆: `due` → 方向 → `update_`）。両側 None なら SGD。

### `AdamDirection(DirectionOptimizer)` :410
`tau_m`（必須）, `tau_v`（必須）, `eps=1e-8`。`pc.optim.AdamW` と同じ時間積分（`a_m = 1-exp(-elapsed/tau_m)`, `c_m += a_m (1-c_m)`）、方向 `m/c_m / (sqrt(v)/sqrt(c_v) + eps)`。単独で step すると weight decay なしの AdamW と一致（テストあり）。state: `m, v, c_m, c_v`。`Grafted(inner=AdamDirection(...))` が「Adam の向き + SGD の大きさ」（M6 で決め、M13 でも試行: 玉 1 つで 0.61・0.89、SGD 0.49・0.92 と同等）。

### `Correlation(Optimizer)` :473（診断）
- 通常クラス（**frozen dataclass ではない、可変、1 ノードに 1 個**）。`Correlation(inner)`、`bind` で `_BoundCorrelation` を作り自分に保持、`take() -> dict[int, float]` は前回の take からの各 lag の平均 cos。
- `LAGS = (1, 10, 100, 1000)`（ノードのステップ数での lag）。全パラメータを 1 本のベクトルに平坦化・正規化して保持。直近 100 ステップと、100 ごとの 1 個を 1000 ステップ分 → **ノードあたり約 110 本の勾配ベクトルを device 上に保持**（大きいノードではメモリ食い）。`_sums` は tensor のまま足す（device 同期を遅らせる）、`take()` で float 化。
- state_dict は inner のみ（相関のバッファは保存されない）。
- 使用: `model.Network(config, correlation=True)`（`model.py:295-303`）、`run.py:165` で **PC-ALM の run_video では常に True**、`run.py:225-227` で eval ごとに `rho_<node>_<lag>` を記録。BP 版には無い。M7/M10 の分析用。M14-M16 では不要と思われる → 消すか既定 off に。

### `Warmup(Optimizer)` :541
- `@dataclass(frozen=True, kw_only=True)`、`inner: Optimizer`, `seconds: float`（`0 < seconds < inf` でなければ `ValueError`）。
- ramp $r(t)=\min(1,t/T)$、elapsed のステップを $\int_t^{t+\text{elapsed}} r$ 秒の inner のステップに変換（`_ramped(t) = t²/2T` か `t - T/2`）。t は state（`torch.zeros((), float64)` tensor）でステップの刻み方に依らない。
- **inner の時間全体を伸ばす**: AdamW の moment、Whitened の EMA・`refresh` も ramp 中は時定数が $1/r$ 倍に見える。t はノードごとの学習時間（ノードが学習した elapsed の和）なので、PC-ALM と BP で同じ ramp（M11 で algorithm-reviewer が approve）。
- state: `{"t": tensor, "inner": ...}`。
- 使用: `OptimConfig.warmup > 0` で `make_optimizer` が最外に巻く（Grafted の外側）。`easy.py:69` が `warmup=WARMUP` を使う（M13 の現行設定）。

### テスト（`tests/lib/test_optim.py`, 363 行）
オラクル: Whitened の `a=b=0`/両側なしは SGD、`tau→0` で極分解、EMA が刻みに依らない、G 固定で `G G^T` に収束。Projected の更新が上位方向に直交、両側なしで SGD。AdamDirection 単独 = AdamW(wd=0)。Grafted のステップの大きさ = SGD。conv が行列として動く、state_dict の round trip、`pc.Chain` で学習が進む（`Grafted(inner=Whitened(...))`）。Orthogonalized = Whitened の極限、rank 1。`_eigh` の摂動の経路。Warmup の積分・境界・0 秒・刻み非依存・ramp 後は SGD・round trip。→ optim.py を残すならテストもそのまま残せる（他モジュールへの依存なし、mujou と torch だけ）。

## 3. `OptimConfig` と `make_optimizer`（`exp/video/config.py`）

`@dataclass OptimConfig` :46、フラット:
```
kind: str = "sgd"   # "sgd" | "adamw" | "adam_direction" | "whitened" | "orthogonalized" | "projected"
tau: float = 1.0    # 慣習: tau = FRAME / lr（lr はフレームあたり、FRAME = 1/30）
graft: bool = False # direction 系のみ有効。sgd/adamw では黙って無視される
tau_m: float = 0.3; tau_v: float = 30.0; weight_decay: float = 0.0
tau_l: float|None = None; tau_r: float|None = None; a = 0.25; b = 0.25
rank: int = 4; eps: float = 1e-4; refresh: float = 1.0
warmup: float = 0.0  # >0 で Warmup で巻く
```
`make_optimizer` :257: warmup>0 なら `Warmup(inner=make_optimizer(replace(c, warmup=0)), seconds=c.warmup)`。sgd→`pc_optim.SGD(tau)`, adamw→`pc_optim.AdamW(tau, tau_m, tau_v, weight_decay)`（**eps は渡さない**。また `eps=1e-4` の既定は Whitened 用で、AdamDirection にも渡さない → AdamDirection は eps=1e-8 固定）、それ以外は DirectionOptimizer を作り `graft` なら `Grafted(inner=...)`。未知の kind は `ValueError`。
`Config.dynamics_optim: OptimConfig | None = None` :227 — dynamics と readout だけ別の optimizer（M13）。

hydra 化のヒント: optimizer 群は全部 `kw_only` の frozen dataclass（`Correlation` を除く）なので、`_target_: exp.lib.optim.Grafted` + `inner: {_target_: exp.lib.optim.AdamDirection, tau: ..., tau_m: ..., tau_v: ...}` のような入れ子の `instantiate` がそのまま効き、`kind` 文字列の分岐（make_optimizer）を消せる。ただし `tau` は lr から計算する値なので、`lr`（フレームあたり）を探索変数にして `tau = frame / lr` を resolver か factory で作る方が optuna で扱いやすい。tests/video/test_model.py:232,233,275,317,331 が `make_optimizer` を monkeypatch しているので、消すならそこも直す。

## 4. `exp/lib/linear_stream.py`（77 行）

`LinearStream(*, d, tau_x, frame, noise, mean_std=1.0, jitter=0.3, iid=False, seed)`。$x_k=\mu_k+\sigma_\xi\xi_k$、$\mu$ は OU 過程（相関時間 `tau_x`、定常 std `mean_std`、フレームごとに厳密に）、`y = w_*^T x + noise·η`、`w_* ~ N(0, I/d)`。`next() -> (x (d,), y (1,))`、`sample_iid(n, seed)`（ノイズなし、別 generator）。`iid=True` は毎回定常分布から平均を引き直す。
使用者: `run.py:19,58`（`run_linear`, `DataConfig.kind == "linear"`、`run.py:250`）だけ。付随する設定: `DataConfig.d_x=32, tau_x=10.0, noise=0.1, iid=False`（iid は linear 以外で未使用）、`ModelConfig.hidden=0`（MLP の幅）。テスト `tests/lib/test_linear_stream.py`（55 行）。→ run_linear ごと消せる。

## 5. 旧スクリプト（`exp/video/`）

### `stage0.py`（M6、122 行）
`configs()`: seed×ストリーム（`x1, x10, x100, iid`）×optimizer（SGD `TAU=10.0` と `adam_direction` の `tau_m ∈ (0.1,0.3,1,3)`×graft/raw、`tau_v=30`）の線形回帰、加えて MLP（hidden 32）で `learn_interval ∈ (25e-3, None)`×PC/BP。`SECONDS=600`。`summarize`: 後半の `dist, iid_loss, loss` の平均。CLI `run|summarize --out --jobs 8`。**定数 `RUN_MODULE`, `SEEDS` の置き場になっている**のが唯一の残る依存。

### `stage1.py`（M7、76 行）
`video-k{kappa}-lr{lr}-s{seed}`、`RATES=(0.01,0.03,0.1)`、`KAPPAS=(0.05,0.5,0.8)`、600 秒。`summary(path)`: 後半 5 分の `error/z_var` と最後の eval の `rank*, rollout*, probe*`、`rho_*` の平均（Correlation の出力を読む唯一の集計）。

### `stage2.py`（M8/M10、283 行）
`LR=0.01, KAPPA=0.8, TAU=FRAME/LR, TAUS=(0.3,3.0), REFRESH=5.0, SECONDS=1800, PERIOD=600`。`optimizers()`: sgd, adam-m0.3/m1（graft）, adamw（lr 1e-3）, orth, white-l/r/lr0.3, white-lr3, proj4/16-lr0.3（全部 graft）。BP・shuffled・replay の基準。`realtime()` は `realtime=True` の確認。`summary`（preq, rollout1/5/10/20, retention, rank, probe）、`table`、`collapsed`、`judge`（H2-H4 の閾値判定）。`tests/video/test_stage2.py` が `judge` をテスト。結論: どの optimizer も SGD を超えない。

### `stage3.py`（M12、137 行）
EgoDex、`model(**changes)` の基準 `d_z=128, d_s=512, enc_nodes=3, dyn_nodes=3, kappa=0.8, stop_target=True`。`FIX = z_norm, open_loop=0.5, open_cycle=20, buffer=1023, buffer_every=8`。`VARIANTS`: m9/norm/fix/fix-bp/res/res-bp（lr 0.01/0.03）。`DEVICES=("cuda:0","cuda:0","cuda:1")`（4090 に 2 本、4060 に 1 本）。`retention(path, data)`: タスク切替時の誤差の跳ね（2 周目/1 周目）。`stability.summary/KEYS` を使う。

### `stability.py`（M11、489 行）
M11 の掃引を `--open --confirm --predictive --heavy --blur --long --adam --streams --span --spread --spread-confirm --spread-warm --spread-warm-fast --decode` の排他フラグで切替。中身の知見（すべて results/m11.md にある）: `z_norm`、`open_loop=0.5, open_cycle=20`、`streams=128` で SIGReg の重みは `kappa B/(1-kappa)`、`spread=True`（ストリーム i を周期の i/B から）、`warmup=300`、`checkpoint_interval=300`（decode 用）。`summary(path)`: 最後の eval の `rank, rollout*, persist*, mean*, z_var`、最後の 1/4 の `preq`、`z_rms_max`。`KEYS` タプル。**要るのは知見だけで、コードは不要**。

### `latency.py`（181 行）
学習を止めて（`learn_interval=inf`）EgoDex を流し、フレーム内の各ステップ後に各ノードの `r + lam` を autograd の $-\partial L/\partial h$（`ideal_credits`: encoder→dynamics→`compare_energy`+`kappa*sigreg`）と比べる。`LAYOUTS` の (enc, dyn) ノード数 × 順序（gauss-seidel / jacobi = 出力から逆順 `run.step(period, *reversed(tree))`）。`at=12` ステップ（書込みから 26 ms、学習のタイミング）での最悪の相対誤差を出す。`mujou.testing.run` を手で step。M16 で「ノードを増やすと credit が届くまで何ステップかかるか」を測るなら手法は参考になるが、ネットの組み方が変わるので書き直しになる。

### `frozen.py`（89 行）
M11: `WorldBPNetwork` の optimizer を `w.step` と `w.readout` の params だけに bind し直して（`w._params`, `w.optimizer` を外から差し替え、private 属性に触る）エンコーダを初期値で固定。2 分ごとに `evaluate(Snapshot(...))`。M16 の「エンコーダは学ぶが動力学の誤差は credit を送らない」とは違う。消してよい。

### `collapse.py`（M9、96 行）
`m9-{stop|full}-k{kappa}-lr{lr}-s0`、`KAPPAS=(0.05,0.2,0.5,0.8)`、`RATES=(0.03,0.1)`、`confirm()` は 30 分×seed 0-2×PC/BP×`CONFIRM_RATES=(0.1,0.03,0.01)`。`stage1.summary` を使う。消してよい。

## 6. 消すときに一緒に直す/消すもの（依存グラフ）

- `stage0.RUN_MODULE, SEEDS` ← `easy.py:25`, `tests/video/test_run.py:11`, stage1/2/3, stability, collapse。
- `stage1.summary` ← collapse。`stability.summary, KEYS` ← stage3。`stage2.judge` ← `tests/video/test_stage2.py`。
- `LinearStream` ← `run.py` の `run_linear`（と `DataConfig.kind=="linear"`, `d_x, tau_x, noise, iid`, `ModelConfig.hidden`, `tests/lib/test_linear_stream.py`, `tests/video/test_run.py` の linear のケースがあれば）。
- `Correlation` ← `model.py:23,295-303`, `run.py:165,225-227`。
- `egodex.py` は stage3・latency・`run.source`・`model.egodex_stages`・`config`（`path="datasets/egodex64"`, `enc_nodes` の EgoDex 用分岐）が使う。M15 で EgoDex を使わないので、まとめて消す候補。
- 文書: `exp/README.md:23`（lib の部品一覧）、`exp/video/PLAN.md:348`、results/*.md と report*.html の実行コマンド（過去の記録なので残してよい。git の履歴で再現できる）。
