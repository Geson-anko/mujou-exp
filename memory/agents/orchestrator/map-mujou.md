# map: mujou ライブラリ（submodule `mujou/`）— M14〜M16 の再編のための地図

作成 2026-11-03（orchestrator の調査サブエージェント）。コードは変更していない。

## 0. 前提と注意

- **作業ツリーの `mujou/` は空**（submodule が未初期化）。`git submodule status` は `-881c4ee...`。ポインタ `881c4ee`（PR #114 の merge、v0.1.0 の後）は **`origin/main` の HEAD と同じ**で、未 push のコミットは無い。調べるときは scratchpad に clone して `881c4ee` を checkout した（`/private/tmp/claude-501/.../scratchpad/src/mujou`）。実験を回す各マシンでは `just setup`（`git submodule update --init` + `uv sync`。Rust/maturin で `mujou._native` をビルドする）が要る
- `~/Projects/mujou` は古い別の checkout（`chore/2026-10-21/...` ブランチ）で、`881c4ee` を持っていない。使わない
- 依存: `torch>=2.14.1`（lock は 2.15.0 と 2.14.1 が混在）、Python 3.14 / 3.14t、ビルドは `maturin>=1.15`
- 公開 API は `mujou.*` の `__all__`。`mujou._native`（Rust）だけが private

## 1. パッケージの構成（`src/mujou/`、Python 約 6.3k 行）

| モジュール | 行 | 役割 |
| --- | --- | --- |
| `__init__.py` | 43 | `Node, Stateful, Port, MultiPort, Channel, Network, Group, run, Run, NetworkError, *Spec` を再輸出 |
| `node.py` | 501 | `Node`（hook と状態）、`Stateful`（ABC）、状態の収集 |
| `port.py` | 374 | `Port`, `MultiPort`, `Channel`, `Wire` |
| `spec.py` | 389 | `TensorSpec, DictSpec, IntSpec, FloatSpec, BoolSpec, TimestampSpec, MessageSpec, FieldSpec` |
| `network.py` | 73 | `Network`, `Group(**named)` |
| `execution.py` | 830 | `run()`, `Run`, `ManualRun`, 開始・停止・状態の取得 |
| `testing.py` | 82 | `mujou.testing.run` → `ManualRun` |
| `time.py` | 28 | `global_timer, Timer, VirtualTimer, ReadOnlyTimer, Timestamp`（実体は `_native`） |
| `types.py` | 28 | `Device`, `Size = int \| tuple[int, ...]`, `Message` |
| `viz.py` | 114 | `mujou.viz.to_html(root) -> str` |
| `driver.py`, `wiring.py`, `tree.py`, `state.py`, `codec.py`, `current.py`, `component.py`, `errors.py` | | 内部（Driver は hook ごとの CUDA device/stream 設定と同期） |
| `pc/` | 約 1.7k | PC-ALM（下の 4 節） |

## 2. ノードとポートの定義

### `Node`（`node.py:29`、`__init__` は `node.py:175`）

```python
Node.__init__(self, *, period: float = 0.01, seed: int | None = None,
              device: Device = "cpu", dedicated_thread: bool = False)
```

- 読み取り専用 property: `device`, `period`, `dedicated_thread`, `generator`（ノード専用の `torch.Generator`、`device` 上、`seed` で初期化、状態に入る）、`timer`（hook の中でだけ読める `ReadOnlyTimer`。`__init__` やホストで読むと `RuntimeError`）
- hook（すべて最初に `super()` を呼ぶ規約）: `on_start()` / `on_update(dt)` / `on_pause()` / `on_resume()` / `on_exception()` / `on_stop()`。順は `on_start → (load_state_dict) → on_update...`
- 状態: tensor 属性と `mujou.Stateful`（`nn.Module` を含む）の属性は自動で状態に入る。普通の値（int、list of tensor 等）は `self.register_state("name")`。`state_dict()` / `load_state_dict()` を override 可（`super()` に委ねる）。CUDA 上の状態は CPU のコピーになり、既定の `load_state_dict` がノードの device に戻す。乱数生成器の状態は CPU と CUDA で形式が違い、CUDA で保存したものは CUDA にしか戻らない
- device は `"cpu"` か CUDA だけ（`"mps"` 等は `ValueError`）。`"cuda"`（index なし）は作った時点のホストの current device に固定
- CUDA のノード: hook ごとにノード専用の stream を current にし、**`on_update` の後にその stream を同期する**（worker がその間待つ）。stream は device あたり 32 本のプールで、超えると共有
- CUDA のテンソルやモジュールは `on_start` 以降に作るのが推奨（caching allocator の stream 紐付けのため。`__init__` でも動く）
- **モジュールの既定の初期化はグローバル乱数を使う**ので、再現には `self.generator` で初期化し直す。ノードの中で `torch.manual_seed` を呼ばない

### `Port` / `MultiPort`（`port.py`）

```python
Port(send=SPEC_S | None, recv=SPEC_R | None, initial=...)      # 受け取る相手は高々 1、送るだけなら何本でも
MultiPort(send=..., recv=...)                                   # recv() -> dict[Channel, R]（connect 順）
a.connect(b) -> Channel
port.send(msg); port.recv(); port.recv_with_time() -> (value, Timestamp | None)
multiport.send_to(channel, msg)
```

- ポートはノードの属性に直接代入する（list / dict に入れると `TypeError`）。`__dict__` が唯一のレジストリ
- **connect は両向きの spec の一致を要求する**（`port.py:65`）: `self.send_spec == other.recv_spec` かつ `self.recv_spec == other.send_spec`。片方が送らない向きは `None` 同士で一致させる。`MultiPort` は全ピアに同じ recv spec を課す
- チャネルは queue ではなく**最新値 1 つ**。読みは待たない。送られる前は初期値（zeros か `initial=`）で時刻 `None`
- `send` は detach してコピー（DLPack）。ノードをまたぐ計算グラフは作れない
- 値はテンソルと Python スカラー、`DictSpec(TypedDict, **fields)` で 1 メッセージにまとめる（一緒に届く必要がある値は 1 つの DictSpec に）
- **shape は spec で固定**（バッチ次元も shape の一部）。可変 shape や構築後のバッチ変更は無い（`specs/open-decisions.md`）

### `Network` / `Group`（`network.py`）

- `Network` は組み立てだけ（hook 無し）。子ノード・子ネットワークを属性に代入、ポートを `self.prev = self.l1.prev` のように公開。パス `"body.l1"` が状態の木のキー
- `Group(**named)` は並べるだけ。1 つの `run` に独立なネットワークを並べる（1 本の時間軸を共有）

## 3. 実行（ホストの入出力、スレッド、時間）

### `mujou.run`（`execution.py:595`）

```python
mujou.run(root, *, timer=None, inputs=None, state=None, workers=None) -> Run
```

- バックグラウンドで全ノードを M:N のワーカープール（Rust）で回す。既定 `workers` は使える CPU 数（GIL 有効時は半分）、ノード数で頭打ち
- `Run`: `write({port: value}) -> Timestamp`（複数ポートは一斉に切り替わる。値はコピー）、`read(port)`, `read_with_time(port)`, `subscribe(port, timeout=)`, `wait_for(port, until, timeout=)`, `timer`（`sleep`, `pause`, `resume`, `set_time_scale`）, `state_dict(timeout=)`, `shutdown()`, `join()`, context manager
- ホストが書けるのは「受け取る・未接続の Port」だけ。読むのは送るポートなら何でも（接続中のポートも観測できる）。読み出しは常に CPU のコピー
- 1 つのノード・ネットワークは 1 回しか走らない（再実行は新しいインスタンス）
- ノードの hook の中で `Run` のメソッドや timer の操作を呼ぶと `RuntimeError`

### `mujou.testing.run` と `ManualRun.step`（`testing.py`, `execution.py:531`）

```python
mujou.testing.run(root, *, timer: VirtualTimer | None = None, inputs=None, state=None) -> ManualRun
ManualRun.step(dt: float, *nodes: Node) -> None   # timer を dt 進め、nodes を順に 1 回ずつ更新（省略で木の順 = Gauss-Seidel）
```

- `period` は無視される。`dt` は step が進めた時間そのもの。決定的
- **実験は現状すべてこれで回している**（`exp/lib/host.py` の `Host`: フレームごとに `write` → `step(period)` を `frame/period` 回 → `read`）。理由は実時間の `mujou.run` では間に合わないため（M12: 4090 で 9 ノード × 2 ms 周期 → フレームの 47% で予測が遅れた。1 周期の全ノードの更新が CPU のカーネル起動律速で約 1.5〜2 ms）
- `pc.show` は `ManualRun` では返らない（タイマーを誰も進めないため）
- ManualRun は workers を持たず、ノードは 1 つずつ順に更新される。**1 つの run の中は並列化されない**。並列性は「プロセス／スレッドを分けて複数の run を同時に回す」ことで得る（各 `testing.run` は自分の `VirtualTimer` を持つので、同じプロセスに複数の run を並べても時間軸は独立）
- step の合間はノードが動かないので、ホストがノードの属性（モジュール）を `copy.deepcopy` しても競合しない（`exp/video/run.py:snapshot` がそうしている）。実時間の run では禁止（競合する）

### 時間

- 実行ごとに時間軸は 1 本。`global_timer`（既定）、`Timer()`、`VirtualTimer()`（`advance(seconds)`）。`Timestamp` は ns の整数、差は秒の float
- `dt`・`period`・pc の時定数はすべてタイマーの秒。`timeout` だけ実時間

### スレッドとプロセス全体の設定

- ノードの hook は worker をまたいで移る。`threading.local` に頼らない。スレッドごとの torch 設定（`no_grad`, `autocast`）は `with` に閉じる
- `torch.manual_seed`、`torch.set_num_threads`、`torch.backends.*`、`torch.use_deterministic_algorithms` はホストが `run` の前に 1 回設定する。CPU の目安: (workers + 専用スレッド) × `set_num_threads` ≤ 物理コア
- **free-threaded 3.14t の落とし穴**: `torch.optim.*` を作ると `torch._dynamo` が import され、プロセス全体で GIL が戻る（`pc.optim` が torch.optim を使わない理由）。exp では `offline.py:122-124` と `decode.py:76-77` が `torch.optim` を使う。hydra / optuna / trackio などを入れるときも、import 後に `sys._is_gil_enabled()` を確かめる（free-threading 非対応の C 拡張は GIL を戻す）。ManualRun 1 本は単一スレッドなので GIL の影響は「1 プロセスに複数の run をスレッドで並べる」ときに効く

## 4. `mujou.pc`（PC-ALM）

`import mujou.pc as pc`。直下: `Chain, Shown, show, Input, Layer, Output, TimeConstants`。サブモジュール: `pc.optim, pc.relax, pc.loss, pc.msg, pc.clock, pc.functional`。

### メッセージ（`pc/msg.py`）

- `Sample{x, y, mask, reset}`（ホスト → `Input.prev`）
- `Activity{h, y, mask, reset, time: Timestamp}`（前向き。`y`・`mask`・`reset`・`time` は Input から終端まで同じメッセージで運ばれる）
- `Prediction{y_hat, y, mask, time}`（`Output.next` → ホスト）
- `activity_spec(h: TensorSpec, y: TensorSpec, batch: Size = ()) -> DictSpec[Activity]`。`mask`・`reset` は `TensorSpec(batch, bool)`
- `rows(mask, like)`（mask を `like` の shape に reshape）、`batch_shape(batch)`
- `g_in` は素のテンソル（`h_in` の shape）で後ろ向き

### 量と式（`pc/functional.py`, `pc/layer.py`）

- `mu = f(h_in; θ)`、`r = h - mu`（ノードが保持）、`lam`（双対）、`delta = r + lam`、`g_in = -J_hᵀ delta`、`g = Σ` 次のノードからの `g_in`
- 緩和: `τ_h ṙ = -(r + lam + g) - (τ_h/τ_r) r`、`τ_λ λ̇ = r`（`r` を動かしてから、動いた `r` で `lam`）
- **送るのは `mu`（固定予測の仮定）**、`mu + r` ではない。credit は後ろ向きにだけ流れる
- 平衡点で `r = 0`、`lam = -∂ℓ/∂h`（BP の credit）、重みの勾配も BP と一致
- `pc.functional.vjp(module, h_in, delta: Callable[[mu], Tensor], params=()) -> (mu, g_in, g_theta)`。`h_in` は detach される。`g_theta` はバッチで和（ノードが `/B` してから optimizer へ）
- `pc.functional.loss_vjp(module, h_in, y, mask, loss, params=()) -> (y_hat, g_in, g_theta)`。mask=False の y は 0 に置き換え
- `pc.functional.relax_(r, lam, g, dt, tau, relaxer=None)`（in place）
- `pc.functional.mse(y_hat, y, batch=())`（サンプルごとの平均、½ なし）、`cross_entropy(...)`
- ノード内 autograd は `_grad`（`functional.py`、`# mujou: local-autograd`）

### ノード

```python
pc.Input(*, x: TensorSpec, y: TensorSpec, period: float, batch: Size = (), device="cpu")
pc.Layer(module, *, act_spec, tau: TimeConstants, optimizer: Optimizer, relaxer: Relaxer = Euler(),
         learn_interval: float | None, period=None, max_dt=None, device="cpu")
pc.Output(module, *, act_spec, loss, tau, optimizer, learn_interval, period=None, max_dt=None, device="cpu")
pc.Chain(*modules, x, y, loss, tau, optimizer, relaxer=Euler(), learn_interval, batch=(), max_dt=None, period=None, device="cpu")
pc.show(run, chain: Shown, sample: Sample, *, seconds) -> Prediction   # 実時間の run 専用
```

- ポート: `Input.prev: Port(recv=Sample)`, `Input.next: MultiPort(send=Activity, recv=x)`（g_in は読まない）、`Layer.prev: Port(send=h_in, recv=Activity)`, `Layer.next: MultiPort(send=Activity, recv=out)`、`Output.prev: Port(send=h_in, recv=Activity)`, `Output.next: Port(send=Prediction)`
- 各ノードは `act_spec`（自分が送る Activity の spec）を公開し、次のノードがそれを受け取る
- `Layer.on_update`（`layer.py`）: `prev.recv` → `clock.tick` → 新入力なら `reset` の行の `r`・`lam` を 0 + `relaxer.reset_` → `vjp`（mask=False の行は delta=0）→ `prev.send(g_in)` → `relax_` → `next.send(mu, ...)` → `learn` なら `optimizer.step_([g/B ...], tick.elapsed)`
- `Layer` は作るときに zeros で module を 1 回呼んで shape を決める。module はサンプルごとに独立（BatchNorm の train、Dropout は不可。LayerNorm・GroupNorm は可）
- 状態のキー: `generator`, `module`, `r`, `lam`, `clock`, `optimizer`, `relaxer`

### `TimeConstants(h, lam, r)`（`pc/time_constants.py`）

- すべて必須。`period = h / (1 + h/r)`（r=h で h/2）、`max_dt = min(period, 2*lam)`
- 推奨の組: `r = h`, `lam = h/2`, 周期 `h/2` で、1 更新で delta が `-g` に移り、credit が 1 周期に 1 ノード戻る
- `lam = inf` は PC（λ 無し）、`r = inf` は減衰なし。周期 > `max_dt` は `ValueError`

### `pc.clock.LearnClock`（`pc/clock.py`）

- `LearnClock(*, period, learn_interval: float | None, max_dt)`、`LearnClock.of(tau, *, period, learn_interval, max_dt=None)`
- `tick(dt, time, now) -> Tick(dt, new, learn, elapsed)`。`learn_interval=None` で毎更新学習（**ストリーム向け**、M14 系はこれか有限値）。`math.inf` で学習しない。新しい入力を初めて見た更新では学ばない（有限の場合）

### optimizer / relaxer（設定 + bind の形）

- `pc.optim.SGD(tau)`: `p -= (elapsed/tau) g`。状態なし
- `pc.optim.AdamW(tau, tau_m, tau_v, eps=1e-8, weight_decay=1e-2)`: 時間で進む AdamW、質量 `c_m, c_v` で bias 補正。torch 換算 `tau = Δt/lr`, `tau_m = -Δt/ln β1`
- 自作: `pc.optim.Optimizer.bind(params) -> BoundOptimizer`（`step_(grads, elapsed)`, `state_dict`, `load_state_dict`）。exp/lib/optim.py に Whitened / Projected / AdamDirection / Grafted がある（M1、M10 で SGD を超えず）
- `pc.relax.Euler()`（既定、状態なし）。自作: `Relaxer.bind(r) -> BoundRelaxer`（`step_(r, force, dt, tau_h)`, `reset_(samples)`）。不動点は force=0
- `pc.loss.MSELoss(batch)`, `pc.loss.CrossEntropyLoss(batch)`（サンプルごとの損失を返す。バッチ平均の loss を渡すと `ValueError`）

### バッチ（B 本のストリーム）

- `batch=B`（または多次元）を `Input`/`Chain`/`activity_spec` と loss の両方に渡す。片方だけだと `Output` がエラー
- サンプル（= ストリーム）ごとに `mask`・`reset` を持つ。ストリームでは `reset=False`、エピソードの頭だけ True。`r`・`lam` はストリームごとに持ち越される
- `reset` はノードがその書き込みを見たときに 1 回働く。見られずに上書きされた書き込みのリセットは失われる（ManualRun で毎フレーム数ステップ回すなら問題にならない）
- 時間の次元（系列）はサンプルの次元であってバッチではない（`(B, T, D)` は `batch=B`）

## 5. CUDA / 複数 GPU

- ノードごとに `device="cuda:N"`。CUDA のノード同士のメッセージは device 上に置いたまま（GPU をまたぐと device-to-device コピー、peer access が無ければドライバがホスト経由にしうる）。CPU のノード・ホストが読むときは CPU のコピー
- ホストの `write` は、持ち主のノードの CUDA device のテンソルなら device 上でコピー
- 自分で同期するなら自分の stream だけ（`torch.cuda.synchronize()` は禁止。他ノードも待つ）
- **CUDA graph は使わない**（mujou のユーザー決定、2026-10-29）。速さはノードを少なく大きく切ることで稼ぐ
- **プロセスやマシンをまたぐ分散は無い**。1 つの run は 1 プロセスの中。DDP 相当の機能も無い。4 台のフリートは「独立な run（1 run = 1 seed × 1 設定）をプロセス単位で GPU に割り当てる」形を実験側で作る
- M5 の計測: バッチ 1・32×32 では GPU は CPU より速くない（2 ms ごとの小さなカーネル起動とノードごとの同期が支配）。大きいモデル・大きいバッチで GPU が効く

## 6. 状態の保存と復元

- `run.state_dict(timeout=inf) -> dict`（実行中は全ノードを一時停止して取り、正常終了後も取れる。例外後は `NetworkError`）。`ManualRun` でも使える
- 木: `{"timer": {"time_ns", "time_scale"}, "nodes": {path: {"state": ..., "last_update_ns": int, "ports": {"prev": {"in": E}, "next": {"out": E, "to": {peer_path: E}}}}}}`、`E = {"value", "time_ns"}`。すべて CPU のコピーで `torch.save` / `torch.load(weights_only=True)` で読める（独自クラスを入れなければ）
- 復元: **同じ構造の新しいインスタンス**を作り `mujou.run(net, state=tree)`（または `mujou.testing.run(net, state=tree)`）。パス・ポート・ピアが一致しないと `ValueError`。復元した run は保存時刻から始まる専用のタイマー（`timer=` と `state=` の併用は不可）
- 今の exp は mujou の state_dict を使わず、`exp/video/run.py:86-110` の `snapshot` が step の合間にモジュールを `deepcopy` して `torch.save`（pickle、`weights_only=False`）している。評価とデコーダの学習用で、学習の再開はできない

## 7. 局所性・非同期性の不変条件（`mujou/.claude/skills/locality-and-async/SKILL.md`）

この repo のルートで起動した Claude には mujou の hook が効かないので、mujou を変えるときは自分で照合する。

1. **局所性**: ノードは自分の状態と、隣から**メッセージで受け取った値**だけで更新する。他ノードの属性を覗かない。ノードをまたいで計算グラフを繋がない（`.backward()` はノード内だけ。`src/mujou/` 内では `# mujou: local-autograd`）。hook で変えたスレッドごとの設定は hook 内で戻す
2. **非同期性**: 全ノードをまたぐバリア（「全員 forward → 全員 backward」）を正しさの前提にしない。同期実行は非同期の特殊なスケジュールとして書き、同期専用の経路を作らない。停止は状態の取得と観測にだけ使い、学習アルゴリズムの一部にしない
3. **継続学習**: train/eval のモード、エポック前提の API を持ち込まない。教師の有無は mask で
4. **状態の所有**: 1 つのテンソルの書き手は 1 ノード。ノードをまたぐ結合の重みは後ろ側（受け取る側）が持つ
- ノード内の BP は自由（1 ノードの中に閉じていれば BP も 1 つの更新手法）
- 更新手法は差し替え可能に。ノード・メッセージ・スケジューラに PC-ALM 固有の量を焼き付けない
- ホストのループ（`exp/lib/host.py` の write → step × n → read）はホストのスケジュールなので不変条件に触れない。ただし「1 フレームの間に全ノードが平衡するまで回す」をアルゴリズムの正しさの前提にすると、非同期性の原則に沿った主張にならない点は README に明記する

## 8. M14〜M16 で要りそうなもの・足りないもの

| 要るもの | mujou にあるか | どうするか |
| --- | --- | --- |
| credit を後ろに送らない結合（M14 の目標＝観測、M16 の目標 z・入力 z が定数） | 仕組みとしてはある | 受け手が `g_in` を送らない（zeros を送る・送らない）だけで実現できる。**M13 の `exp/video/nodes.py` の `Dynamics` がすでにそうしている**（`z = Port(send=z, recv=act_spec)` で何も送らず zeros、`Compare` の `target_weight=0`）。注意: 送り手の `MultiPort(send=Activity, recv=X)` に繋ぐピアは、送らなくても `send=X` を宣言しないと connect が `TypeError`。代わりに送り手が送るだけの `Port(send=Activity)` を別に出せば、ピアは `Port(recv=Activity)` で繋げる（送るだけの Port はピアを何本でも持てる） |
| 観測が状態（低次元）の入力ノード | `pc.Input` で足りる | `x=TensorSpec((B, d_obs))`。Input は `g_in` を読まない（x は clamp）ので、M14 の「目標へ送る credit はない」は自動 |
| B=128 本のストリームのバッチ | ある | `batch=B`、`reset` は最初のフレームだけ True |
| ViT 級のノード（M15 のエンコーダ） | 制限は無い | module はサンプルごとに独立であればよい（トークン間の attention はサンプルの中なので可、LayerNorm 可、Dropout 不可）。PC-ALM の `Layer` は 1 周期ごとに forward + VJP を 1 回するので、1 フレーム S 周期なら BP の S 倍。**ViT はノード 1 つの中で BP（自前の node 内 autograd + `pc.optim` で step）にする**のが現実的。複数ノードに割ると 1 周期 1.5〜2 ms の起動律速が効く |
| JEPA の EMA ターゲット | 無い（必要なら） | 同じノードの中で持てば局所性に触れない（EMA の重みもそのノードの状態）。別ノードにすると重みのコピーを 2 か所に持つことになり、不変条件 4 と食い違う |
| N フレームのまとまり（V-JEPA） | 無い | ノードの中にリングバッファ（`register_state` するか `state_dict` を override）。`(B, N, C, H, W)` を 1 サンプルとして扱う（時間はサンプルの次元） |
| マスク（I-JEPA/V-JEPA のブロックマスク） | 無い | ノード内で `self.generator` から引く（seed 再現のため） |
| 複数 GPU・複数マシン | GPU 間メッセージはある。プロセス・マシン間は無い | フリートは run 単位の並列で実験側に作る。1 run 内で GPU を分けても、ManualRun は逐次なので速くならない |
| optuna の pruning・途中経過 | — | ホストのループで評価値を報告すればよい（mujou は関与しない）。`ManualRun` は host のスレッドで step するので、ループの中で `trial.report` を呼べる |
| 学習の再開（長い run の preemption 対策） | `state_dict` / `state=` | 今の exp は使っていない。フリートで長い run を回すなら mujou の木を定期保存し、同じ構造で再構築して `testing.run(net, state=tree)` で再開できる（データのストリームの位置と乱数はホスト側で別途保存が要る） |
| 1 ノードの「エネルギー」の監視ポート | 無い（open decision） | 実験のノードで `error = Port(send=TensorSpec(()))` を足す形（M13 の `Compare.error` と同じ） |
| 多入力の `Layer`（合流） | 無い（open decision） | `pc.functional` で自作（`examples/pc_merge.py` の `Merge` が手本） |
| 可変 shape・可変バッチ | 無い | 設定ごとにネットワークを作り直す |

### 再編での指針（要点）

- 実験の実行単位は「`mujou.testing.run` 1 本 + ホストのループ」。1 プロセス 1 run（GPU 1 枚を複数プロセスで共有）にすれば、GIL・`set_num_threads`・CUDA の stream プール（device あたり 32）の干渉を避けやすい。CPU で回す小さい M14 の設定は `torch.set_num_threads(1)` で 1 コア 1 run が目安
- `torch.optim` を使う BP の基準・オフラインの上限・デコーダの学習は、mujou の run と別プロセスに置くか、`pc.optim` で書く（3.14t で GIL を戻さないため）。hydra・optuna・trackio の import 後の GIL も確かめる
- 実時間の `mujou.run` は M12 で間に合わなかった。M14〜M16 も ManualRun（仮想時間）が前提。step の合間ならホストがモジュールを deepcopy してよい
