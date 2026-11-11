# 設計案: M14〜M16 のための再編（観点: 研究の速度）

2026-11-03、orchestrator 配下の設計サブエージェント。main 06a2a11、mujou 881c4ee。既存のファイルは変更していない。

この案の物差しは 2 つある。1 つは「1 日に回せる、判定に使える実験の数」。もう 1 つは、オーケストレーター（Mac の AI エージェント）が「結果を読む → 決める → 投げる」を 1 周する時間。インフラは最小にし、測れない複雑さは入れない。

---

## 0. 要旨と設計の原則

### 速度を決めるもの（地図から読んだ律速）

| 律速 | 今の値 | この案での手当て | 効き |
| ---- | ------ | ---------------- | ---- |
| 玉の物理が純 Python | B=128 で約 0.13 s/フレーム、30 分で物理だけ約 2 時間 | torch の float64 で `(B, balls)` をまとめて進める（`balls.py` の書き直し）。旧実装はテストの参照実装として残し、ビット一致を確かめる | 約 100〜200 倍。M14 の 1 run の大半が消える |
| 1 run = 1 プロセス = CPU 1 コア | 1 台で手書きの数本 | 4 台 × 約 28 スロットを常時埋めるワーカー | 並列の本数が 10 倍以上 |
| 探索が手作業（定数タプルの書き足し） | 1 周に数時間 | `fleet submit --grid` と optuna の ask/tell、pruning | 1 周が数分 |
| 結果を読むのが遅い | `metrics.jsonl` が毎フレーム 54k 行、run dir 名の glob | 評価は `evals.jsonl` に分ける。`fleet table` が seed を束ねた表を 1 コマンドで出す | 分析が 1 コマンド |
| 発散した run も最後まで回る | 30 分〜7 時間を浪費 | NaN か閾値超えで即終了（`diverged`）。pruner が STOP ファイルで止める | 無駄な計算を捨てる |
| 毎フレームの `.item()`（GPU 同期） | 毎フレーム | 和を device 上に貯め、`log_every` ごとに 1 回だけ取り出す | GPU の run が速くなる |

### 原則

1. **共有のバスは NAS のファイルだけ**。キュー・run の出力・STOP の合図・ホストの心拍は、すべて `/nas/geson/mujou-exp/`（コンテナでは `/fleet`）のファイルにする。サーバー・DB・メッセージキューを立てない。optuna の sqlite だけは std-00 のローカルディスクに置き、調整役のプロセスだけが触る
2. **学習のプロセスは 1 本の入口**（`python -m exp.video.run`）。online / offline / baseline / decode の別は設定の `task` で切り替える。ランチャーは 1 種類のコマンドしか知らない
3. **ランチャーは stdlib だけ**（`exp/lib/fleet/`）。Mac では `uv run --no-project --python 3.14` で、ホストではコンテナの venv で、同じコードが動く
4. **状態を問い合わせるコマンドは std-00 の上で実行する**。Mac の CLI は薄い転送役で、NAS が見える std-00 のコンテナに ssh で処理を渡す。Mac が自分でやるのは sync・pull・up だけ
5. **コードは commit せずに投げてよい**。投げるたびに `exp/` のスナップショットを各ホストへ配り、job はそのスナップショットに固定する。run は repo と mujou の SHA と差分（`code.diff`）を記録する。再現性を保ったまま、commit 待ちが無くなる
6. **ダッシュボードは入れない**。読むのは人ではなく AI エージェントなので、テキストの表と PNG が最速。trackio も optuna-dashboard も既定では入れない（§6）

---

## 1. パッケージ構成

### 1.1 再編後の `exp/`

```text
exp/
  README.md                      # 3 層の説明（lib の一覧を更新）
  lib/                           # mujou・torch・stdlib だけに依存
    __init__.py
    host.py          (残す)      # Host だけ。RealtimeHost は消す（M14〜M16 は仮想時間）
    mingru.py        (残す)      # MinGRU, ResidualMinGRU, MinGRUTrace
    sigreg.py        (残す)      # 関数（epps_pulley, sigreg, insert_）と SIGReg ノード。M15 が使う
    metrics.py       (残す)      # effective_rank, probe_r2
    optim.py         (縮める)    # AdamDirection, Grafted, Warmup だけ。Whitened/Projected/Orthogonalized/Correlation/_eigh/_Factors を消す
    rundir.py        (新規)      # run の契約（§3）。Recorder の後継。GIL の確認、code の版、STOP、ckpt
    png.py           (新規)      # write_png, tile（decode.py から移す。stdlib zlib + torch）
    runs.py          (新規, stdlib) # run dir の読み込み、seed の束ね、Markdown の表（§6）
    fleet/           (新規, stdlib) # ランチャー（§4）
      __init__.py
      __main__.py                # python -m exp.lib.fleet
      cli.py                     # argparse。Mac か std-00 かを見て、転送するか自分で実行する
      remote.py                  # ssh + docker exec、ControlMaster、並列実行
      sync.py                    # rsync のミラー、exp/ のスナップショット、uv sync の要否
      jobs.py                    # job の id・group・slug、--grid の展開、job.json の schema
      queue.py                   # NAS の上のキュー（queue/ → claimed/<host>/）と状態の導出
      worker.py                  # 各ホストの常駐プロセス（スロットの割り当て、起動、回収、心拍）
      slots.py                   # CPU コア・GPU・メモリの割り当て（純粋関数）
  video/
    README.md, PLAN.md, MILESTONES.md, results/, report*.html   (残す。記録)
    __init__.py
    config.py        (書き直し)  # structured config の dataclass（§2）
    conf/            (新規)      # hydra の YAML（§2）
    run.py           (書き直し)  # 唯一の入口。task で分岐
    balls.py         (書き直し)  # ベクトル化した物理、clips、render
    observe.py       (新規)      # 状態 → 観測（段 0〜3）、Standardizer、ObsStream、EvalSet
    dynamics.py      (nodes.py の後継) # Dynamics, DynamicsLayer, Compare（目標へ credit を送らない形に固定）
    world.py         (model.py + bp.py の後継) # 動力学のモジュール、DynamicsPart、ObsNetwork、DynamicsBP、ObsBPNetwork
    measure.py       (evaluate.py の後継) # 観測の空間の rollout と物差し（平均・写し・等速・真の物理）
    viz.py           (新規)      # 予測した状態を描いた PNG、軌跡の JSON
    offline.py       (書き直し)  # オフラインの上限（窓、BPTT、pc.optim で step）
    tune.py          (新規)      # optuna の調整役（§5）。optuna に依存するので lib ではなくここ
    studies/         (新規)      # 探索の仕様の YAML
    figs.py          (新規)      # matplotlib の図。学習の環境では import しない（別の環境で動かす、§6）
    # M15 以降に足す
    vit.py                       # ViT、パッチ、チューブ、マスク（nn.Module だけ）
    jepa.py                      # JEPA ノード（mujou の Node）と BP 版
    wt.py                        # Walking Tours の前処理とストリーム
    decode.py        (書き直し)  # チェックポイントごとに後から学ぶデコーダ（task=decode）
    joint.py                     # M16 の JointNetwork
fleet.toml           (新規, repo ルート) # ホスト・スロット・パス
```

### 1.2 消すもの

- `exp/lib/config.py`（`from_dict`。hydra に置き換え）、`exp/lib/sweep.py`（fleet に置き換え）、`exp/lib/linear_stream.py`（段 0 の線形回帰は M14 段 0 が置き換える）、`exp/lib/record.py`（`rundir.py` に置き換え）
- `exp/video/` の `easy.py`, `stage0.py`, `stage1.py`, `stage2.py`, `stage3.py`, `stability.py`, `latency.py`, `frozen.py`, `collapse.py`, `egodex.py`, `model.py`, `nodes.py`, `bp.py`, `evaluate.py`, `decode.py`, `offline.py`, `config.py`, `run.py`（旧版。必要な考え方は新しいモジュールに移す）
- 旧版から捨てる機能: エンコーダの CNN・LSUV（`standardize_`）・`Reconstruct`・`recon`・`z_norm`・`kappa` と Compare の結合・`stop_target`/`target_weight`・`Correlation`・`realtime`・`order`（shuffled/replay）・エピソードの引き直し・`spread`・EgoDex
- `compose.yaml` と `.env.example` の EgoDex のマウント（M15 は EgoDex を使わない）
- 記録（`README.md`, `PLAN.md`, `MILESTONES.md`, `results/`, `report*.html`）は残す。実行コマンドは当時のもので、git の履歴で再現できる

### 1.3 mujou に上げるもの

**最初は何も上げない**。M14 に要るものは mujou にすべてある（`pc.Input` は credit を読まない、`Port` の片側だけ送る結合、`run.state_dict()`、`pc.optim.AdamW`）。上流の作業で実験を止めない。

計測で律速と分かったときだけ、次を候補にする。

- `ManualRun.step` を n 回まとめる API（1 フレーム 17 回の Python→Rust の往復を減らす）。`fleet bench` のプロファイルで、ホストのループの割合が大きいときだけ
- M15 で ViT のノードを PC-ALM に分けるとき、多入力の `Layer`（open decision の合流）

### 1.4 テストの構成（1 モジュール 1 テストのミラー）

```text
tests/
  conftest.py                 # OMP_NUM_THREADS=1。共通の fixture: TAU, FRAME, fd（中心差分）, tiny_config(), sample()
  test_claude_hooks.py, test_mujou_install.py   (残す)
  lib/
    test_host.py, test_mingru.py, test_sigreg.py, test_metrics.py   (残す)
    test_optim.py             (縮める: AdamDirection, Grafted, Warmup)
    test_rundir.py, test_png.py, test_runs.py                       (新規)
    fleet/
      __init__.py
      test_jobs.py            # id が決定的、seed だけ違えば group が同じ、grid の直積
      test_queue.py           # claim が 1 回だけ成功する、状態の導出、requeue
      test_slots.py           # CPU・GPU・メモリの割り当て、優先度
      test_worker.py          # 偽の run（sleep する python -c）で起動・回収・STOP・KILL・心拍
      test_stdlib_only.py     # exp.lib.fleet と exp.lib.runs を -I で import しても torch が sys.modules に無い
  video/
    test_config.py            # すべての preset が RunConfig に compose できる、slug と override
    test_balls.py             # ベクトル化 = 旧スカラー実装（テスト内の参照実装）とビット一致、エネルギー、反射
    test_observe.py           # 逆標準化で Frame.state に戻る、段ごとの列、Standardizer
    test_dynamics.py          # Compare の有限差分、Dynamics が逐次の minGRU と一致、過去と目標へ credit を返さない
    test_world.py             # 平衡点の lam = BP の credit、PC と BP の初期値の一致、トレース = BPTT、局所性
    test_measure.py           # 恒等モデル = 写し、等速の外挿の誤差 0、真の物理の誤差 0、残差 0 初期化 = 写し
    test_viz.py               # 描いた PNG の重心が状態の位置
    test_offline.py           # 窓が reset をまたがない、小さい設定で最後まで走り evals を書く
    test_run.py               # 契約のファイルと schema、決定性、STOP、発散、GIL（subprocess）
    test_tune.py              # 目的値の束ね方、pruning の判定、confirm の job の生成（optuna の in-memory storage）
    # M15 以降: test_vit.py, test_jepa.py, test_wt.py, test_decode.py, test_joint.py
```

消すテスト: `tests/lib/test_config.py`, `test_linear_stream.py`, `test_record.py`、`tests/video/test_egodex.py`, `test_stage2.py`, `test_bp.py`, `test_model.py`, `test_nodes.py`, `test_evaluate.py`, `test_decode.py`（M15 で作り直す）。オラクルの型（平衡点、トレース、有限差分、局所性、物理の不変量、決定性）は新しいテストに移す（map-tests.md の 3 節の表）。

テストの実行はコンテナの中（Mac では mujou をビルドしない）: `just fleet x std-00 -- just test-fast`。

---

## 2. hydra の設定

### 2.1 方針

- **hydra の compose API だけを使う**（`@hydra.main` を使わない）。出力ディレクトリ・ログ・multirun・launcher の仕組みは要らない。run dir は §3 の契約が決める
- スキーマは `@dataclass` で書き、`ConfigStore` に登録する。preset は YAML。`OmegaConf.to_object` で dataclass に戻してから使うので、利用側は普通の型付きの dataclass を読む（pyright がそのまま効く）
- `Union` の dataclass は OmegaConf が扱えないので、**有無で表す**: `encoder: EncoderConfig | None`、`dynamics: DynamicsConfig | None`。M14 は encoder が None、M15 は dynamics が None、M16 は両方
- `Literal` は使わず `str` にし、`__post_init__` で検査する（`to_object` が dataclass のコンストラクタを呼ぶので走る）。tuple は避けて `list`
- 学習率は `tau` ではなく **フレームあたりの `lr`** で持ち、`OptimConfig.build(frame)` が `tau = frame / lr` に直す（optuna で対数一様に引くため）

### 2.2 dataclass（`exp/video/config.py`）

```python
@dataclass(kw_only=True)
class TimeConfig:
    frame: float = 1 / 30
    h: float = 4e-3
    lam: float = 2e-3
    r: float = 4e-3
    learn_interval: float | None = 25e-3
    def constants(self) -> pc.TimeConstants: ...

@dataclass(kw_only=True)
class OptimConfig:
    kind: str = "sgd"            # "sgd" | "adamw" | "adam_graft"
    lr: float = 0.03             # per frame: tau = frame / lr
    warmup: float = 300.0        # seconds; 0 for none
    tau_m: float = 0.3
    tau_v: float = 30.0
    weight_decay: float = 0.0
    def build(self, frame: float) -> pc.optim.Optimizer: ...

@dataclass(kw_only=True)
class BallsConfig:               # M13 の EASY の物理が既定
    radii: list[float] = field(default_factory=lambda: [0.10])
    obs: str = "pos_vel"         # "pos_vel" | "pos" | "image"
    size: int = 32
    streams: int = 128
    gamma0: float = 1.0
    gamma_amp: float = 0.0
    e0: float = 1.0
    e_amp: float = 0.0
    drag: float = 0.0
    conserve: bool = True
    physics_period: float = 600.0
    clips: int = 512             # 評価クリップ（ベクトル化したので増やせる）
    eval_seed: int = 12345
    def physics(self) -> Physics: ...

@dataclass(kw_only=True)
class VideoConfig:               # M15: 前処理した u8 memmap
    path: str = "/fleet/data/wt64"
    size: int = 64
    streams: int = 32
    chunk: int = 4               # N フレームのまとまり（1 で画像）

@dataclass(kw_only=True)
class DataConfig:
    balls: BallsConfig | None = field(default_factory=BallsConfig)
    video: VideoConfig | None = None   # ちょうど一方

@dataclass(kw_only=True)
class DynamicsConfig:
    d_s: int = 256
    nodes: int = 1
    trace: bool = False
    residual: bool = False
    open_loop: float = 0.0
    open_cycle: int = 20

@dataclass(kw_only=True)
class EncoderConfig:             # M15/M16。§8 で詰める
    patch: int = 4
    dim: int = 192
    depth: int = 6
    heads: int = 3
    d_z: int = 64
    mask_ratio: float = 0.75
    sigreg: float = 1.0
    directions: int = 256
    buffer: int = 1023

@dataclass(kw_only=True)
class OfflineConfig:
    steps: int = 6000
    batch: int = 128
    window: int = 32
    lr: float = 1e-3
    frames: int = 131072
    every: int = 500

@dataclass(kw_only=True)
class DecodeConfig:
    run: str = ""                # 対象の run id（同じ runs/<exp>/ の中）
    steps: int = 2000
    frames: int = 16384

@dataclass(kw_only=True)
class RunConfig:
    exp: str = "m14"             # マイルストーン。run dir の階層
    task: str = "online"         # "online" | "offline" | "baseline" | "decode" | "bench"
    learner: str = "bp"          # "pc" | "bp"
    seed: int = 0
    seconds: float = 600.0       # ストリームの長さ（仮想時間）
    eval_every: float = 60.0
    log_every: float = 10.0
    ckpt_every: float = 0.0      # 0: 最後に 1 回だけ
    viz_every: float = 0.0       # 0: 最後に 1 回だけ
    device: str = "cpu"          # ワーカーが上書きする
    diverge: float = 1e3         # err が平均の物差しのこの倍を超えたら発散として止める
    time: TimeConfig = field(default_factory=TimeConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    dyn_optim: OptimConfig | None = None
    data: DataConfig = field(default_factory=DataConfig)
    dynamics: DynamicsConfig | None = field(default_factory=DynamicsConfig)
    encoder: EncoderConfig | None = None
    offline: OfflineConfig = field(default_factory=OfflineConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)

def compose(preset: str, overrides: Sequence[str] = ()) -> RunConfig:
    """``conf/base.yaml`` + ``conf/experiment/<preset>.yaml`` + ``overrides``,
    as the dataclass (OmegaConf.to_object, checked with isinstance)."""
```

### 2.3 YAML の木（`exp/video/conf/`、`__init__.py` を置いて package にする）

```text
conf/
  __init__.py
  base.yaml                # defaults: [schema, dynamics: small, optim: sgd, _self_]
  dynamics/
    small.yaml             # d_s: 256, nodes: 1
    big.yaml               # d_s: 512, nodes: 3
  optim/
    sgd.yaml               # kind: sgd, lr: 0.03, warmup: 300
    adamw.yaml             # kind: adamw, lr: 1e-4, warmup: 300
    adam_graft.yaml        # kind: adam_graft, lr: 0.03, warmup: 300
  mode/
    explore.yaml           # seconds: 600, ckpt_every: 0
    confirm.yaml           # seconds: 1800, ckpt_every: 300, viz_every: 300
  experiment/              # @package _global_
    m14/
      s0.yaml              # exp: m14, data.balls: {radii: [0.10], obs: pos_vel}
      s1.yaml              # radii: [0.10], obs: pos
      s2.yaml              # radii: [0.06, 0.08, 0.10], obs: pos
      s3.yaml              # radii: [0.10]（段 2 の後に決める）, obs: image
    m15/
      balls.yaml           # dynamics: null, encoder: {...}, data.balls.obs: image
      wt.yaml              # dynamics: null, data: {balls: null, video: {...}}
    m16/
      balls.yaml           # encoder + dynamics
      wt.yaml
```

起動:

```bash
# 手元（コンテナ内）で 1 本
uv run python -m exp.video.run m14/s1 learner=pc dynamics=big optim.lr=0.02 dynamics.trace=true seed=1 --out runs/dev
# フリートへ（ワーカーが同じものを --job で起動する）
just fleet submit m14/s1 dynamics=big --grid learner=pc,bp dynamics.trace=true,false --seeds 0,1,2 +mode=confirm
```

第 1 引数の preset は `+experiment=<preset>` に直して compose する。残りは hydra の override の文法のまま（`key=value`、`+key=value`、`group=option`）。

### 2.4 override から run の名前と tag へ

`exp/lib/fleet/jobs.py` が、hydra を使わずに override の文字列だけから決める（Mac でも同じ id が出る）。

- `key(job) = sha1(json.dumps({"preset", "overrides": sorted(seed 以外), "snap"}))`
- `group = key[:6]`（seed だけ違う run が同じ group）、`run_id = f"{preset の / を -}-{slug}-s{seed}-{sha1(group+seed)[:4]}"`。例: `m14-s1-pc-big-lr0.02-tr1-s1-9c2e`
- `slug`: seed 以外の override を key の最後の成分で短くし、`-` でつなぐ（48 文字で切る）。短縮の表は `jobs.py` の定数（`learner=pc → pc`、`dynamics=big → big`、`optim.lr=0.02 → lr0.02`、`dynamics.trace=true → tr1`、`dynamics.residual → res`、`dynamics.open_loop → ol`、`optim.kind → kind の値`、`+mode=confirm → confirm`、他は `<leaf><value>`）。`[]`・`,`・`/` は除く
- tag: job の `overrides`（そのまま）と、`--tag` で付けた語（`explore`、`confirm`、study 名）。集計は名前ではなく `job.json` の `overrides` と `group` で行うので、slug は読みやすさのためだけ
- `snap`（コードのスナップショット）を key に入れるので、コードを変えて同じ override を投げると別の run になる。同じコードで同じ override を投げ直すと、done の run は飛ばす（冪等）。`--reuse` でスナップショットを無視して冪等にする

### 2.5 pyright に優しくする工夫

- `compose()` の戻り値は `RunConfig`（`to_object` の `Any` を `isinstance` で絞る）。他のモジュールは OmegaConf を import しない
- YAML の誤りは `tests/video/test_config.py` が全 preset を compose して捕まえる（`conf/experiment/**/*.yaml` を glob して parametrize）
- `__post_init__` で `obs`・`task`・`learner`・`kind` の値と、`data.balls` と `data.video` のちょうど一方を検査する

---

## 3. run の契約

### 3.1 ディレクトリ

```text
/fleet/runs/<exp>/<run_id>/       # = /nas/geson/mujou-exp/runs/...（手元では --out の下）
  job.json          # 投げたもの（ワーカーが起動前に書く）
  config.yaml       # 解決した設定（OmegaConf.to_yaml）
  meta.json         # 起動時の環境とコードの版
  code.diff         # repo と mujou の未コミットの差分（スナップショットから写す）
  log.txt           # stdout と stderr
  progress.json     # 上書き。{t, frames, of, fps, wall, eta, last}（status が読む）
  scalars.jsonl     # log_every ごとに 1 行（毎フレームではない）
  evals.jsonl       # eval_every ごとに 1 行
  baselines.json    # 物差し（モデルに依らないので 1 回だけ）
  result.json       # 最後に 1 回。これが有るかで終わったかを判定する
  ckpt/<t:07.1f>.pt # {"t", "tree": run.state_dict()}
  viz/<t>.png, viz/<t>.json   # 見た目の確認（§7.6）
  decode/           # M15/M16 の task=decode が後から書く
  STOP              # 誰かが置いたら、次の log_every で止まる（status: stopped / pruned）
  KILL              # ワーカーが見たら SIGTERM
```

### 3.2 schema

```jsonc
// job.json
{"id": "m14-s1-pc-big-lr0.02-tr1-s1-9c2e", "group": "4be1a0", "exp": "m14", "preset": "m14/s1",
 "overrides": ["learner=pc", "dynamics=big", "optim.lr=0.02", "dynamics.trace=true"], "seed": 1,
 "snap": "a1b2c3d4e5", "resources": {"device": "cpu", "gpu_mem": 0, "mem": 2},
 "prio": 50, "after": [], "study": null, "trial": null, "tags": ["explore"], "submitted": "2026-11-03T10:00:00+09:00"}

// meta.json
{"run_id": "...", "host": "std-02", "slot": {"core": 7, "gpu": null}, "pid": 12345,
 "started": "...", "argv": [...], "device": "cpu",
 "code": {"snap": "a1b2c3d4e5", "repo": "06a2a11...", "repo_dirty": true, "diff_sha256": "...",
          "mujou": "881c4ee...", "mujou_dirty": false},
 "python": "3.14.0 (free-threading)", "torch": "2.15.0", "gil": false, "threads": 1}

// scalars.jsonl（1 行）
{"t": 10.0, "frames": 300, "wall": 8.1, "err": 0.123, "open_err": 0.2}

// evals.jsonl（1 行）。err{h} は標準化した観測の空間の 1 次元あたりの二乗誤差
{"t": 60.0, "wall": 41.0, "err1": ..., "err5": ..., "err10": ..., "err20": ...,
 "pos5": ..., "pos10": ..., "pos20": ...,          // 位置の列だけ
 "r5_cv": ..., "r10_cv": ..., "r20_cv": ...,       // err / 等速の外挿
 "r10_copy": ..., "r10_mean": ...}

// baselines.json
{"mean": {"1": ..., "5": ..., "10": ..., "20": ...}, "copy": {...}, "cv": {...}, "physics": {...}}

// result.json
{"status": "done", "t": 1800.0, "wall": 5230.0, "frames": 54000, "final": {<最後の evals の行>},
 "error": null}      // status: done | stopped | pruned | diverged | failed。failed は traceback の末尾
```

- 評価の名前は M14〜M16 で揃える。M15/M16 の `decode/evals.jsonl` は `recon`, `img{h}`, `pix_copy{h}`, `pix_mean{h}`, `r{h}_pixmean` を足す
- オフライン（task=offline）も同じ `evals.jsonl` の鍵を書く。`t` の代わりに `step` を入れる。同じ表の道具で比べられる

### 3.3 `exp/lib/rundir.py`

```python
class RunDir:
    def __init__(self, path: Path, *, job: Mapping[str, Any], config_yaml: str) -> None
        # mkdir, config.yaml・meta.json を書き、code.diff を写す（スナップショットの CODE.json から）
    def scalars(self, **values: float) -> None      # 1 行追記して flush
    def eval(self, **values: float) -> None
    def progress(self, **values: float) -> None     # 一時ファイルに書いて os.replace
    def baselines(self, values: Mapping[str, Any]) -> None
    def should_stop(self) -> bool                   # STOP があるか（log_every ごとに 1 回 stat）
    def checkpoint(self, t: float, tree: Mapping[str, Any]) -> Path
    def finish(self, status: str, **extra: Any) -> None
def code_version(root: Path) -> dict[str, Any]     # スナップショットの CODE.json か、git から
def assert_free_threaded() -> None
    # sysconfig の Py_GIL_DISABLED が 1 で sys._is_gil_enabled() が True なら RuntimeError
def load_modules(tree: Mapping[str, Any], net: mujou.Network) -> None
    # ckpt の木からノードの "module" の state_dict だけを、同じ構造で作った net に入れる（評価・デコード用）
```

### 3.4 run.py の流れ

```python
def main(argv: Sequence[str] | None = None) -> Path:
    # 1. --job <job.json>（ワーカー）か、<preset> [overrides] --out <dir>（手元）
    # 2. config = compose(preset, [*overrides, f"seed={seed}", f"device={device}"])
    # 3. torch.set_num_threads(1)（OMP_NUM_THREADS=1 はワーカーが env で渡す）
    # 4. rd = RunDir(...)
    # 5. try: TASKS[config.task](config, rd) / except: rd.finish("failed", error=tail) ; raise
def run_online(config: RunConfig, rd: RunDir) -> None
def run_baseline(config: RunConfig, rd: RunDir) -> None   # 物差しだけ測って終わる（数秒）
def run_bench(config: RunConfig, rd: RunDir) -> None      # 300 フレームの fps を測る
# run_offline は offline.py、run_decode は decode.py
```

`run_online` のループ（M14）:

1. `ev = observe.eval_set(cfg)`、`rd.baselines(measure.baselines(ev))`
2. `net = world.build(config)`、`stream = observe.ObsStream(...)`
3. `assert_free_threaded()`（ネットワークを作った後。torch の lazy import も含めて確かめる）
4. `with mujou.testing.run(net) as run`、`Host(run, frame, period)`。各フレーム: `x, reset = stream.next()`、`Sample(x, y=zeros, mask=~reset, reset)`、`p = host.show(...)`
5. 予測誤差の和と数を device 上のテンソルに貯める。`log_every` ごとに `.item()` して `rd.scalars`、`rd.progress`。ここで `rd.should_stop()` と発散（非有限か `err > diverge × mean`）を見る
6. `eval_every` ごとに `measure.evaluate(world.modules(net), ev)` を `torch.no_grad` で、step の合間にモジュールを直接読む（deepcopy しない）。`rd.eval`
7. `ckpt_every`・`viz_every` ごと、と最後に、`rd.checkpoint(t, run.state_dict())` と `viz.write(...)`
8. 最後に `assert_free_threaded()` をもう一度、`rd.finish("done", final=...)`

### 3.5 GIL の確認

- 起動時と終了時に `assert_free_threaded()`。戻っていたら `failed`（理由: GIL）
- `tests/video/test_run.py::test_the_run_keeps_the_gil_off`: subprocess で `python -c "import exp.video.run, sys; ...; build a tiny net; assert not sys._is_gil_enabled()"`
- 規約: exp の中で `torch.optim` を使わない（オフラインの上限とデコーダも `pc.optim.AdamW` を `bind(...).step_(grads, elapsed=1.0)` で使う）。`tests/video/test_run.py` で `exp/` を grep して `torch.optim` が無いことも確かめる。optuna・matplotlib は学習のプロセスで import しない（`tune.py`・`figs.py` は別のプロセス）

---

## 4. フリートのランチャー（`exp/lib/fleet/`、stdlib だけ）

### 4.1 構成

```text
Mac (orchestrator)                       std-00..03（各ホスト、コンテナ mujou-exp-dev-1）
  just fleet <cmd>                         worker（常駐、1 ホスト 1 本）
   ├─ sync/up/pull: 自分で ssh・rsync        ├─ /fleet/queue から claim
   └─ その他: ssh std-00 docker exec ...      ├─ スロットに run を起動（taskset、CUDA_VISIBLE_DEVICES）
             python -m exp.lib.fleet <cmd>    ├─ 回収、心拍 /fleet/hosts/<host>.json
                                              └─ KILL を見て SIGTERM
            NAS /nas/geson/mujou-exp = コンテナの /fleet
              queue/  claimed/<host>/  finished/  hosts/  runs/<exp>/<id>/  studies/  data/
```

- **ホスト同士は通信しない**。すべて NAS のファイルで調整する
- Mac が落ちても（スリープしても）、キューに残った job はワーカーが回し続ける。Mac は投げて、後で読むだけ

### 4.2 `fleet.toml`（repo ルート）

```toml
nas = "/nas/geson/mujou-exp"      # ホストのパス
mount = "/fleet"                  # コンテナの中のパス
checkout = "/data/geson/mujou-exp"
container = "mujou-exp-dev-1"
user = "vscode"
head = "std-00"                   # 状態の問い合わせと optuna の調整役を置くホスト

[slots]                           # fleet bench の結果で決める（§9 の WP0）
cpu = 28                          # 1 run 1 コア。4 コアはワーカー・評価・OS に残す
mem = 56                          # GB
gpus = [ {id = 0, name = "4090", mem = 22, jobs = 8},
         {id = 1, name = "4060", mem = 7,  jobs = 3} ]

[[hosts]]
name = "std-00"
[[hosts]]
name = "std-01"
[[hosts]]
name = "std-02"
[[hosts]]
name = "std-03"
```

### 4.3 コードの同期（rsync のミラー + exp のスナップショット）

git は使わない（commit と push を待たない）。各ホストの checkout は Mac の作業ツリーの**片方向のミラー**で、ホストで編集しない。

`fleet sync`（`submit` が毎回先に呼ぶ。差分が無ければ数秒）:

1. Mac で `exp/` の内容のハッシュ `snap` を計算する（`git ls-files -co --exclude-standard -- exp` の各ファイルの path と bytes の sha256、先頭 10 桁）。`CODE.json`（repo と mujou の SHA、dirty、`git diff HEAD` と未追跡ファイルの差分を `code.diff` に、mujou も同じく）を作る
2. 4 台へ並列に `rsync -az --delete --filter=':- .gitignore' --exclude=mujou/target --exclude=.git/modules/mujou/... <repo>/ std-0X:/data/geson/mujou-exp/`。gitignore の filter で `.venv-devcontainer`・`.env`・`runs`・`datasets`・`.fleet` は送らず消さない。`.git` も送るので、ホストの `git status` は Mac と同じになる
3. 各ホストで `mkdir -p .fleet/snap/<snap> && cp -a exp .fleet/snap/<snap>/ && cp CODE.json code.diff ...`（既にあれば飛ばす）
4. `uv.lock`・`pyproject.toml`・`mujou/` のハッシュが前回の `.fleet/env-stamp` と違うホストだけ、コンテナで `uv sync`（mujou の Rust の再ビルド）。走っている run が使う mujou は変わるので、`--env` が要る変更のときは `fleet sync` が走っている run の数を表示して確認を求める（`--yes` で進む）

run は `cwd = .fleet/snap/<snap>` で、python は checkout の `.venv-devcontainer/bin/python`。`exp` はスナップショットから、mujou は checkout の editable から読む。ワーカーは job の `CODE.json` の mujou の版が自分のホストの版と違えば、その job を claim しない（`fleet status` に「mujou の版が違うホスト」と出る）。

前提: Mac の submodule を初期化しておく（今は空。WP0）。

### 4.4 キュー（`queue.py`）

- 投入: `queue/<prio:03d>-<submitted_ns>-<id>.json`（小さい prio が先）。同時に `runs/<exp>/<id>/job.json` は書かない（ワーカーが起動時に書く）
- claim: `os.rename(queue/<f>, claimed/<host>/<f>)`。SMB のサーバーで rename は原子的で、負けた側は `FileNotFoundError` になる。`test_queue.py` は 2 つのプロセスで同じ job を取り合い、片方だけが取れることを確かめる（CIFS の上の確認は WP0 で 1 回やる）
- `after`: 依存する run の `result.json` が `done` になるまで claim しない（task=decode を学習の run の後に回すため）
- 状態は保存せず、ファイルから導く: `queue/` にある → queued、`claimed/` にあり `result.json` が無い → running（心拍が 2 分以上古いホストなら lost）、`result.json` の `status`
- 終わった claim は `finished/<id>.json` へ移す（一覧のため）

### 4.5 ワーカー（`worker.py`）

```python
def main(host: str, config: FleetConfig) -> None:   # 5 秒ごとのループ
    # 1. 子プロセスを回収。returncode != 0 で result.json が無ければ、ワーカーが failed を書く
    #    （signal と 137 は 1 回だけ自動で queue へ戻す）
    # 2. claimed/<host>/ の KILL を見て SIGTERM（10 秒後に SIGKILL）
    # 3. 空いたスロットに入る job を優先度順に claim し、起動する
    # 4. hosts/<host>.json に心拍（時刻、スロットの使用、走っている id、loadavg、空きメモリ、nvidia-smi の空き）
def launch(job: Job, slot: Slot) -> subprocess.Popen[bytes]:
    # env: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPYCACHEPREFIX=/tmp/pyc
    #      CUDA_VISIBLE_DEVICES=<gpu id or "">
    # argv: taskset -c <core> <venv python> -m exp.video.run --job <run dir>/job.json --device <cpu|cuda:0>
    # cwd: .fleet/snap/<snap>、stdout/stderr → <run dir>/log.txt、start_new_session=True
```

- ワーカーを再起動しても run は死なない（`start_new_session`）。再起動したワーカーは `claimed/<host>/` の job を読み、`meta.json` の pid が生きていて `/proc/<pid>/cmdline` に id があれば引き取る（子ではないので終わりは `result.json` で知る）
- 起動は `fleet up`: 各ホストで心拍が古ければ `docker exec -d -u vscode -w <checkout> <container> bash -lc 'nohup python -m exp.lib.fleet worker --host std-0X >> .fleet/worker.log 2>&1'`

### 4.6 スロット（`slots.py`、純粋関数）

```python
@dataclass(frozen=True) class Slot: core: int; gpu: int | None
def allocate(free: Resources, job: JobResources, config: SlotConfig) -> Slot | None
```

- CPU の job: 空いたコアを 1 つ（`taskset` で固定）とメモリ `mem` GB
- GPU の job（`--gpu`）: コアを 1 つ + GPU を 1 枚。`gpu_mem` の宣言の和が GPU の `mem` 以下、本数が `jobs` 以下。4090 を先に埋める
- 実際の空き（`nvidia-smi`）は心拍に出すだけで、割り当ては宣言で行う（予測できて速い）。CUDA の OOM は failed になり、`fleet retry --gpu-mem 6` で投げ直す
- M14 の段 0〜2 は CPU の job が既定。段 3 と M15/M16 は GPU が既定（どちらが速いかは `task=bench` で決める）

### 4.7 CLI（オーケストレーターが使うもの）

```text
just fleet up                                   # コンテナとワーカーを確かめ、無ければ起動。ホストの表
just fleet sync [--env] [--yes]                 # §4.3
just fleet submit <preset> [overrides...] [--grid k=v1,v2 ...] [--seeds 0,1,2]
                 [--gpu] [--gpu-mem G] [--mem M] [--prio P] [--after ID] [--tag T] [--reuse] [--dry]
                                                # 投げた id を 1 行ずつ出す。--dry は展開だけ
just fleet status [--exp m14] [--tag T]         # 状態ごとの数、ホストごとのスロット、走っている run の進み（t/of、fps、ETA、最新の r10_cv）
just fleet ls <glob> [--state failed|running|...]
just fleet table <glob> [--keys r5_cv,r10_cv,r20_cv] [--at last|t=600] [--md] [--sort KEY]
                                                # group ごとに 1 行、seed を " / " で 1 セル、違う override を列に（§6）
just fleet log <id> [-n 80]                     # log.txt の末尾
just fleet stop <glob>                          # STOP を置く（次の log_every で止まる）
just fleet kill <glob>                          # KILL を置く（SIGTERM）
just fleet cancel <glob>                        # キューから外す
just fleet retry <glob> [--gpu-mem G]           # failed / lost を投げ直す
just fleet requeue --host std-03                # 落ちたホストの running を queue へ戻す
just fleet pull <glob> [--ckpt]                 # runs/<exp>/<id>/ を Mac の runs/ へ rsync（既定で ckpt を除く）
just fleet x <host|all> -- <cmd...>             # コンテナでコマンド（テスト、uv sync、python -m ...）
just fleet tune start|status|stop|confirm ...   # §5
```

- justfile: `fleet *args: uv run --no-project --python 3.14 python -m exp.lib.fleet {{args}}`
- ssh は ControlMaster を使う（`-o ControlMaster=auto -o ControlPath=~/.ssh/cm-%C -o ControlPersist=600`）。1 コマンド 0.1 秒台
- 出力は既定で短い表。`--json` で機械向け
- 典型の 1 周: `fleet table 'm14-s1-*' --at last` → 決める → `fleet submit ...` → `fleet status`（数分後）→ `fleet pull ... && Read viz/*.png`

### 4.8 出力の置き場所

**NAS に直接書く**（ローカルに書いて後で同期しない）。理由: 調整役（optuna）と `fleet status`・`table` が途中経過を同じ場所で読める。同期の仕組みも要らない。

- 書き込みの量を抑える: 毎フレームの行を無くした（`log_every` 10 秒ごと）。100 本並べても NAS への追記は毎秒数十行
- M14 の checkpoint は数 MB。M15 の ViT でも数十 MB で、ckpt は 300 秒ごとなので量は小さい
- データセット（M15 の Walking Tours の u8）は NAS の `/fleet/data/` に 1 回だけ前処理し、run は最初にホストのローカル（`/data/geson/cache/`）へ写してから memmap する（無ければ写す。一時ファイル + `os.replace`）
- 要 compose の変更: `${FLEET_DIR:-./fleet}:/fleet`（各ホストの `.env` に `FLEET_DIR=/nas/geson/mujou-exp`）。GPU をコンテナへ渡す設定（`deploy.resources.reservations.devices`）が無いので、WP0 で CUDA が見えるか確かめ、見えなければ足す

### 4.9 失敗の扱い

| 起きたこと | 検出 | 扱い |
| ---------- | ---- | ---- |
| run の例外 | run が `result.json` に failed と traceback | 自動では投げ直さない（コードの誤り）。`fleet ls --state failed` → `fleet log` |
| signal / OOM killer（137） | ワーカーが returncode を見る | 1 回だけ自動で queue へ戻す |
| CUDA OOM | traceback | `fleet retry --gpu-mem` |
| 発散 | run が自分で `diverged` として止まる | 失敗ではなく結果。表では「発散」 |
| ワーカーが落ちた | 心拍が古い | `fleet up` が起動し直し、走っている run を引き取る |
| ホスト・コンテナが落ちた | 心拍が古く pid も無い | `fleet status` に lost と出る。`fleet requeue --host` |
| NAS が一時的に書けない | run の書き込みが例外 | failed（signal ではないので手で retry） |
| スナップショットや mujou の版が無い・違う | ワーカーが claim しない | `fleet status` に出る。`fleet sync` |

学習の再開（mujou の state_dict から）は作らない。最長の run（PC-ALM + トレース、5〜7 時間）でも投げ直しで足り、作る手間の方が大きい。

---

## 5. optuna

### 5.1 置き場所と storage

- 調整役 `python -m exp.video.tune run <study.yaml>` は std-00 のコンテナで常駐する 1 プロセス（`fleet tune start` が `docker exec -d` で起動）。学習のプロセスとは別なので、optuna の import は GIL に関係しない
- storage は std-00 のローカルディスクの sqlite（`/data/geson/mujou-exp/.fleet/optuna/<study>.db`）。触るのは調整役だけなので、CIFS の上のロックの問題を避けられる。10 分ごとに `sqlite3` の backup API で `/fleet/studies/<study>/study.db` へ写す
- 調整役は job を NAS のキューに直接書く（`exp.lib.fleet.queue.submit` を呼ぶ）。run の中間値も NAS の `evals.jsonl` から読む（`exp.lib.runs`）。学習のプロセスは optuna を知らない

### 5.2 study の仕様（`exp/video/studies/m14_s1_bp.yaml`）

```yaml
name: m14-s1-bp
preset: m14/s1
fixed: [learner=bp, +mode=explore]          # 600 秒
seeds: [0, 1]                               # 1 trial = この seed の run の束
space:
  optim.lr:         {log: [0.003, 0.3]}
  optim.warmup:     {choice: [0, 60, 300]}
  optim:            {choice: [sgd, adamw, adam_graft]}   # group の選択も override の 1 つ
  dynamics:         {choice: [small, big]}
  dynamics.trace:   {choice: [true, false]}
  dynamics.residual:{choice: [true, false]}
  dynamics.open_loop: {choice: [0.0, 0.5]}
objective: {keys: [r5_cv, r10_cv], reduce: mean_log, at: last, diverged: 10.0}
pruner: {kind: median, startup: 10, warmup: 120}         # warmup は仮想時間の秒
trials: 96
parallel: 48                                             # 同時に走らせる trial（× seeds 本）
prio: 60
```

### 5.3 調整役のループ（30 秒ごと）

```python
def tick(study: optuna.Study, spec: StudySpec, inflight: dict[int, Trial]) -> None:
    # 1. 各 in-flight trial の run（seed ごと）の evals.jsonl を読む
    #    全 seed が届いた最新の t について v_t = mean_s log(r10_cv) などを report(v_t, step=int(t))
    #    should_prune() なら全 run に STOP（status: pruned）、tell(PRUNED)
    # 2. 全 run の result.json がそろったら value(spec, results) を tell
    # 3. in-flight が parallel を下回る分だけ ask → overrides を作り、seeds 本を submit（tags に study と trial 番号）
def value(spec: StudySpec, results: Sequence[Result]) -> float
    # seed ごとに keys の log の平均、seed で平均。diverged / failed の seed は log(spec.diverged) を入れる
```

- sampler: `TPESampler(seed=0, multivariate=True, group=True, constant_liar=True)`（並列の ask で同じ点を引かない）
- 1 trial は複数 seed の束なので、目的値が 1 つの seed の当たり外れに引きずられない。探索では 2 seed、確認では 3 seed
- trial と run の対応は `trial.user_attrs["runs"]` に置く。調整役を再起動すると、RUNNING の trial の run を読み直して続ける
- pruning は「止めた run が途中まで書いた `evals.jsonl`」を残す。表の上では pruned と出る

### 5.4 BP で探索し、PC-ALM と BP で確認する

1. 探索: 上の study（`learner=bp`、600 秒、2 seed）。段ごとに 1 study
2. 確認: `just fleet tune confirm m14-s1-bp --top 3 --grid learner=pc,bp --seeds 0,1,2 +mode=confirm --tag confirm`
   - 上位 3 trial の params を override に直し、`learner × seed` の直積を通常の job として投げる（optuna を通さない）
   - 表は `fleet table --tag confirm` で、同じ設定の PC-ALM と BP が隣り合う行になる（group の違いは `learner` だけ）
3. 探索の範囲を絞りたいとき（因子が少ないとき）は optuna を使わず `fleet submit --grid` で直積を投げる。M14 段 0 の最初の比較（small/big × trace × residual × lr 3 点）はこちらが速い

---

## 6. 追跡と集計

### 6.1 `exp/lib/runs.py`（stdlib）

```python
@dataclass class RunView:
    path: Path; job: dict[str, Any]; meta: dict[str, Any] | None; result: dict[str, Any] | None
    def evals(self) -> list[dict[str, float]]            # 遅延読み込み
    def at(self, when: str) -> dict[str, float] | None   # "last" か "t=600"（その t 以下で最後の行）
    @property def status(self) -> str                    # queued/running/... は queue と合わせて fleet が決める
def load(root: Path, pattern: str, *, tags: Sequence[str] = ()) -> list[RunView]
def table(runs: Sequence[RunView], keys: Sequence[str], *, at: str = "last") -> Table
    # group ごとに 1 行。seed の昇順に値を " / " でつなぐ（results/mXX.md の規約）。発散は「発散」、pruned は「打切」
    # 選んだ group の間で値が違う override だけを列にする
def markdown(t: Table, *, digits: int = 2) -> str
```

`fleet table` はこれを std-00 で呼び、出力をそのまま README・`results/mXX.md` に貼れる形で返す。例:

```text
| learner | dynamics | trace | r5_cv            | r10_cv           | r20_cv           |
| ------- | -------- | ----- | ---------------- | ---------------- | ---------------- |
| bp      | big      | true  | 0.31 / 0.33 / 0.30 | 0.42 / 0.45 / 0.41 | 0.70 / 0.74 / 0.69 |
| pc      | big      | true  | 0.32 / 0.33 / 0.31 | 0.43 / 0.44 / 0.42 | 発散 / 0.75 / 0.70 |
```

### 6.2 図

- 見た目の確認の PNG（予測した状態を `balls.render` で描いて真のフレームと並べたもの）は、**run 自身が書く**（`viz/<t>.png`、torch と `lib.png` だけ）。オーケストレーターは `fleet pull <glob>` の後に Read で画像を見る
- 折れ線（学習曲線、horizon ごとの比、軌跡の重ね描き）は `exp/video/figs.py`（matplotlib）。学習の環境（3.14t）には入れず、`uv run --no-project --python 3.14 --with matplotlib python -m exp.video.figs <cmd> runs/...` で、GIL ありの別の一時環境で動かす。入力は pull した `evals.jsonl` と `viz/<t>.json`（予測と真の軌跡の小さな配列）だけで、torch は要らない
- report.html は skill `experiment-report` のまま。データは `runs.load` で抜き出す

### 6.3 trackio・optuna-dashboard は入れない

- trackio: 3.14t に入らない。別の GIL ありの環境とサーバーが要り、毎フレームの指標を送る経路が学習のプロセスに入る。読むのが AI エージェントなので、ダッシュボードの利点が小さい
- optuna-dashboard: コードは要らないので、人が見たいときだけ `fleet x std-00 -- uvx --python 3.14 optuna-dashboard sqlite:///.fleet/optuna/<study>.db --port 8080` と `ssh -L 8080:localhost:8080 std-00` で開けばよい。設計には含めない
- 代わりに `fleet tune status <study>` が、trial の数・状態ごとの数・上位 5 trial の params と値・各パラメータの重要度（`optuna.importance.get_param_importances`）をテキストで出す

---

## 7. M14 のコード（シグネチャ）とオラクル

### 7.1 `exp/video/balls.py`（ベクトル化）

```python
@dataclass(frozen=True, kw_only=True)
class Physics:
    gamma0: float = 1.0; gamma_amp: float = 0.0; e0: float = 1.0; e_amp: float = 0.0
    drag: float = 0.0; period: float = 600.0; conserve: bool = True
    def gamma(self, t: float) -> float
    def restitution(self, t: float) -> float

def step_(pos: Tensor, vel: Tensor, radii: Tensor, h: float, gamma: float, e: float,
          drag: float, conserve: bool) -> None
    """One semi-implicit Euler step in place. pos, vel: (n, balls, 2) float64; radii: (balls,)."""
def advance_(pos: Tensor, vel: Tensor, radii: Tensor, physics: Physics, t: float,
             seconds: float, substeps: int = 8) -> None
def draw(n: int, radii: Tensor, g: torch.Generator) -> tuple[Tensor, Tensor]   # (n, balls, 2) ×2

class Balls:
    """n streams that never redraw. state() is (n, balls, 4): x, y, vx, vy (per second)."""
    def __init__(self, n: int, *, radii: Sequence[float], physics: Physics, frame: float,
                 substeps: int = 8, seed: int) -> None
    def state(self) -> Tensor
    def advance(self) -> Tensor       # one frame on; the new state
    t: float

def clips(n: int, length: int, *, radii: Sequence[float], physics: Physics, frame: float,
          substeps: int = 8, seed: int = 12345) -> Tensor    # (n, length, balls, 4)
def render(states: Tensor, size: int, radii: Sequence[float]) -> Tensor
    """states (n, balls, 4) -> images (n, 1, size, size)."""
```

- 状態のレイアウトを `(…, balls, 4)` に変える（旧: 連結の `4*balls`）
- 全ストリームが同じ時刻 `t` を共有する（`spread` を消した）ので、重力と反発係数は substep ごとのスカラー。玉ごと・軸ごとの壁の判定は `torch.where`
- 物理の時間変化（gamma_amp など）は残す（M16 で難しさを戻す判断のため）。既定は EASY

### 7.2 `exp/video/observe.py`

```python
OBS_KINDS = ("pos_vel", "pos", "image")
def observe(states: Tensor, kind: str, *, size: int, radii: Sequence[float]) -> Tensor
    """states (..., balls, 4) -> (..., d_obs): pos_vel (balls*4), pos (balls*2), image (size*size)."""
def d_obs(kind: str, balls: int, size: int) -> int

@dataclass(frozen=True)
class Layout:
    """Columns of the observation: pos (balls, 2) and vel (balls, 2) or None."""
    pos: Tensor
    vel: Tensor | None
def layout(kind: str, balls: int) -> Layout | None     # image は None

@dataclass(frozen=True)
class Standardizer:
    mean: Tensor     # (d,)
    std: Tensor      # (d,)；image は全画素で 1 つのスカラーを (d,) に広げる
    @classmethod
    def fit(cls, x: Tensor, *, per_dim: bool = True) -> Self     # x (n, T, d)
    def apply(self, x: Tensor) -> Tensor
    def invert(self, x: Tensor) -> Tensor

@dataclass(frozen=True)
class EvalSet:
    states: Tensor    # (n, T, balls, 4) raw
    obs: Tensor       # (n, T, d) standardized, float32
    std: Standardizer
    layout: Layout | None
    frame: float
    radii: tuple[float, ...]
def eval_set(cfg: BallsConfig, frame: float, *, observed: int = 16, steps: int = 20,
             device: str = "cpu") -> EvalSet

class ObsStream:
    def __init__(self, cfg: BallsConfig, *, frame: float, seed: int, std: Standardizer,
                 device: str = "cpu") -> None
    def next(self) -> tuple[Tensor, Tensor]   # x (B, d) float32 standardized, reset (B,) bool（最初だけ True）
```

標準化は評価クリップの統計（MILESTONES どおり）。ストリームも同じ `Standardizer` を使う。

### 7.3 `exp/video/dynamics.py`（ノード）

```python
def prediction_spec(target: DictSpec[Activity]) -> DictSpec[pc.msg.Prediction]
def is_open(count: int, open_loop: float, cycle: int) -> bool
def open_mask(counts: Tensor, open_loop: float, cycle: int) -> Tensor

class Dynamics(Node):        # s_k = minGRU(x_{k-1}, s_{k-1})。旧 nodes.Dynamics と同じ（z → x と読み替え）
    def __init__(self, module: MinGRU, *, act_spec: DictSpec[Activity], tau: pc.TimeConstants,
                 optimizer: Optimizer, learn_interval: float | None, open_loop: float = 0.0,
                 open_cycle: int = 20, residual: bool = False, trace: bool = False,
                 device: Device = "cpu") -> None
    z: Port        # recv=act_spec, send=x spec。何も送らない（zeros）: 過去は定数
    next: MultiPort
    base: Port     # residual のとき
    z_hat: Port    # open_loop > 0 のとき

class DynamicsLayer(Node):   # 旧と同じ

def compare_energy(readout: nn.Module, s: Tensor, target: Tensor, base: Tensor | None = None) -> Tensor
    """(*batch,): mean over the last axis of (readout(s) [+ base] - target)^2. (kappa の結合は消した)"""

class Compare(Node):
    """Predicts the target from s and learns W_o; never sends credit to the target."""
    def __init__(self, readout: nn.Module, *, s_spec: DictSpec[Activity],
                 target_spec: DictSpec[Activity], tau: pc.TimeConstants, optimizer: Optimizer,
                 learn_interval: float | None, residual: bool = False, device: Device = "cpu") -> None
    s: Port        # credit g_s を送る
    target: Port   # recv=target_spec, send=h spec。何も送らない（zeros）
    next: Port     # Prediction（ホストが読む）
    base: Port     # residual のとき
```

M16 の「エンコーダに credit を送らない」は、この 2 つのポートが**構造として送らない**ことで保証する（旧の `stop_target`/`target_weight` の二重のフラグを消した）。

### 7.4 `exp/video/world.py`（組み立て）

```python
@dataclass
class DynamicsModules:
    first: MinGRU
    layers: list[ResidualMinGRU]
    readout: nn.Linear
def dynamics_modules(cfg: DynamicsConfig, d_in: int, d_out: int, *, seed: int) -> DynamicsModules
    """torch.manual_seed(seed), then first, layers, readout in this order (zero readout when
    residual). PC と BP の両方がこれを呼ぶので初期値が一致する。CPU で作ってから .to(device)。"""

class DynamicsStack(nn.Module): ...   # (x_{k-1}, s_{k-1}) -> s_k。状態は L d_s を連結
class Top(nn.Module): ...

class DynamicsPart(mujou.Network):
    """Dynamics -> DynamicsLayer x (nodes-1) -> Compare. Fed by an upstream activity port;
    sends no credit to it. M14 は Input、M16 はエンコーダの出力につなぐ。"""
    def __init__(self, mods: DynamicsModules, *, upstream: DictSpec[Activity], cfg: DynamicsConfig,
                 tau: pc.TimeConstants, optimizer: Optimizer, dyn_optimizer: Optimizer | None,
                 learn_interval: float | None, device: Device) -> None
    def attach(self, port: MultiPort[Any, Any]) -> None   # port → dynamics.z と compare.target
    next: Port

class ObsNetwork(mujou.Network):     # PC-ALM: pc.Input(x=(B, d_obs)) -> DynamicsPart
    def __init__(self, config: RunConfig) -> None
    prev: Port; next: Port; tau: pc.TimeConstants; mods: DynamicsModules

class DynamicsBP(Node):
    """BP in one node: the same memory (x_{k-1}, s_{k-1} detached), trace, open loop, residual
    and LearnClock as DynamicsPart. 旧 WorldBP からエンコーダ・SIGReg・デコーダを抜いたもの。"""
    def __init__(self, stack: DynamicsStack, readout: nn.Linear, *, act_spec: DictSpec[Activity],
                 tau: pc.TimeConstants, optimizer: Optimizer, dyn_optimizer: Optimizer | None,
                 learn_interval: float | None, open_loop: float = 0.0, open_cycle: int = 20,
                 residual: bool = False, trace: bool = False, device: Device = "cpu") -> None
    prev: Port    # recv=act_spec。何も送らない
    next: Port    # Prediction

class ObsBPNetwork(mujou.Network):   # pc.Input -> DynamicsBP
    def __init__(self, config: RunConfig) -> None

def build(config: RunConfig) -> ObsNetwork | ObsBPNetwork
def modules(net: ObsNetwork | ObsBPNetwork) -> tuple[DynamicsStack, nn.Module, bool]
    # 評価が読む (stack, readout = Sequential(Top, W_o), residual)。モジュールそのもの（コピーしない）
```

### 7.5 `exp/video/measure.py`

```python
HORIZONS = (1, 5, 10, 20)
OBSERVED = 16
def rollout(stack: DynamicsStack, readout: nn.Module, obs: Tensor, *, residual: bool,
            observed: int = OBSERVED, steps: int = 20) -> Tensor
    """obs (n, T, d) -> predictions (n, steps, d) of frames observed..observed+steps-1:
    teacher-forced for k < observed, then open loop from the model's own predictions."""
def errors(pred: Tensor, ev: EvalSet, *, observed: int = OBSERVED,
           horizons: Sequence[int] = HORIZONS, columns: Tensor | None = None) -> dict[int, float]
    """Per-dimension mean squared error in the standardized space, at each horizon."""
def mean_baseline(ev: EvalSet, steps: int) -> Tensor        # 評価クリップの平均の観測
def copy_baseline(ev: EvalSet, steps: int) -> Tensor        # 最後の観測
def cv_baseline(ev: EvalSet, steps: int) -> Tensor
    """Constant velocity from the last two raw positions: p + h (p_last - p_prev), velocity
    columns (p_last - p_prev) / frame; no gravity, no walls; standardized. image は None。"""
def physics_baseline(ev: EvalSet, physics: Physics, steps: int) -> Tensor   # 真の物理（誤差 0 の確認）
def baselines(ev: EvalSet) -> dict[str, dict[int, float]]
def evaluate(stack: DynamicsStack, readout: nn.Module, residual: bool, ev: EvalSet,
             base: Mapping[str, Mapping[int, float]]) -> dict[str, float]
    # err{h}, pos{h}, r{h}_cv, r{h}_copy, r{h}_mean
def predicted_states(pred: Tensor, ev: EvalSet) -> Tensor    # 逆標準化し (n, steps, balls, 4) へ。pos だけなら速度 0
```

### 7.6 `exp/video/viz.py`

```python
COLUMNS = (1, 2, 5, 10, 20)
def strip(ev: EvalSet, pred: Tensor, *, clips: int = 4, scale: int = 3) -> Tensor
    """Rows: true frame / predicted state rendered, per clip; columns: COLUMNS frames ahead. (3, H, W)"""
def trajectories(ev: EvalSet, pred: Tensor, *, clips: int = 4) -> dict[str, list[Any]]
    """真の位置と予測した位置の列（JSON 用、figs.py が重ね描く）。"""
def write(rd: RunDir, t: float, ev: EvalSet, pred: Tensor) -> None   # viz/<t>.png と viz/<t>.json
```

段 3（画像）では `pred` が画素なので、そのまま画像として並べる（デコーダは要らない）。

### 7.7 `exp/video/offline.py`

```python
def windows(states: Tensor, window: int) -> Tensor
    """states (T, B, ...) of one Balls run -> starts (n, 2) (no reset: every start)."""
def run_offline(config: RunConfig, rd: RunDir) -> None
    # Balls で T = frames // B 時刻ぶん状態を作る → observe・標準化 → 窓 (batch, window, d)
    # 教師強制 k < 16、以降は開ループ、BPTT は開ループを通る。pc.optim.AdamW を step_(grads, 1.0)
    # every ステップごとに measure.evaluate を rd.eval(step=..., ...)、最後に ckpt
```

### 7.8 オラクル（テスト）

| テスト | 理論から言えること | ファイル |
| ------ | ------------------ | -------- |
| ベクトル化の物理 | 旧スカラー実装（テスト内に参照実装として写す）と float64 でビット一致（玉 1・3、conserve の有無、時間変化あり） | test_balls.py |
| エネルギー | `conserve` で $\tilde E = v^2/2 + g y - g h v_y/2$ が跳ね返りをまたいで保たれる（slow で 30 分） | test_balls.py |
| 重力 0 の等速 | gamma0=0・壁から遠い初期値で、位置が厳密に線形 | test_balls.py |
| 観測 | 段 0 は `invert(apply(observe(s)))` が `s` に戻る。段 1・2 は位置の列だけ。`fit` した後の平均 0・std 1 | test_observe.py |
| 等速の外挿 | 重力 0・壁に当たらない区間で `cv_baseline` の誤差が 0 | test_measure.py |
| 写しと恒等 | readout が直前の観測を返すモデルの rollout は `copy_baseline` と一致。residual で W_o=0 の初期値も同じ | test_measure.py |
| 真の物理 | `physics_baseline` の誤差は 0 | test_measure.py |
| Compare | `compare_energy` の勾配が有限差分と一致（float64）。target のポートは zeros のまま | test_dynamics.py |
| Dynamics | 出力が逐次の minGRU と一致。z のポートは zeros（過去へ credit を返さない） | test_dynamics.py |
| エンコーダなしの平衡点 | 1 フレームを長く保持し `learn_interval=inf`、各ノードの `lam` が BP の `-∂L/∂h` と一致（1 ノード・3 ノード、B 本のストリーム） | test_world.py |
| 同じ初期値 | 同じ seed で ObsNetwork と ObsBPNetwork のモジュールの重みが一致 | test_world.py |
| トレース | 重みを止めたとき、PC と BP の動力学の勾配が入力を固定した BPTT と一致 | test_world.py |
| 局所性 | どのノードも `p.grad` を残さない、重みは動く。Input に届く credit は 0 | test_world.py |
| 決定性 | 同じ job で `evals.jsonl` が一致。評価が学習を変えない | test_run.py |
| 契約 | 小さい設定で、§3 のファイルと鍵がそろう。STOP で stopped、NaN を入れると diverged | test_run.py |
| GIL | run のプロセスで GIL が戻らない（subprocess） | test_run.py |

---

## 8. M15・M16 のスケッチ

### 8.1 M15: JEPA のノード（局所性の中で）

**N フレームのまとまり**: ノードの中のリングバッファ `frames (B, N, C, H, W)`（`register_state` するテンソル）。新しいフレームを見た更新で 1 つずらす。`stride` フレームに 1 回だけ学ぶ（既定 `stride = N`、重ならないまとまり）。reset のストリームはバッファを 0 にし、N フレームそろうまで mask=False。時間はサンプルの次元なので mujou の制約（shape 固定）に収まる。

**第 1 段（BP、ノード 1 つ）**: `jepa.JEPA(Node)` が ViT エンコーダ・予測器・SIGReg（`lib.sigreg` の関数とバッファ）をすべて自分の状態に持ち、ノードの中で autograd（`# mujou: local-autograd` 相当）し、`pc.optim` で step する。

- LeJEPA の形: ターゲットも同じエンコーダで、EMA と stop-gradient を使わず、崩壊は SIGReg で防ぐ。EMA の重みのコピーが要らないので、状態の持ち主が 1 つで済む（不変条件 4）
- マスク: V-JEPA のチューブのブロックマスクを `self.generator` から引く（seed で再現）
- ポート: `prev`（Input の Activity、フレーム）、`next: MultiPort(send=Activity(h = z (B, d_z)))`。z はまとまりの最新のトークンの平均（M16 の動力学への入力）
- これが M15 の BP 版

**第 2 段（PC-ALM）**: ViT を段に切った `pc.Layer` の列 → `JEPAHead(Node)`（予測器、損失、SIGReg）。文脈の見方とターゲットの見方を**バッチの次元に並べる**（1 ストリームから 2 サンプル: マスクした文脈、マスクしないターゲット）。重みは 1 組のまま、両方の見方に credit が流れる（LeJEPA は stop-gradient なし）。credit が段を戻るのに 1 周期 1 ノードかかるので、段の数は少なく（2〜3）切る。第 1 段の結果を見てから着手する。

モジュール: `vit.py`（`PatchEmbed`, `TubeEmbed`, `Block`, `ViT`, `Predictor`, `tube_masks(g, B, grid, ratio) -> Tensor`）、`jepa.py`（`JEPA`, `JEPAHead`, `jepa_loss(...)`、`JEPANetwork(config)`、`JEPABPNetwork(config)`）。

**判定の道具**（各チェックポイント）: `decode.py` のデコーダの再構成（見た目）、合成の玉なら z から状態への線形の probe（`lib.metrics.probe_r2`）、実効ランク。

### 8.2 Walking Tours（`exp/video/wt.py`）

```python
def plan(src: Path) -> list[Video]                     # WTour.txt と動画ファイル、長さ、ライセンス・ぼかしの記録
def decode_stream(path: Path, *, size: int, fps: float, start: float = 0.0,
                  seconds: float | None = None) -> Iterator[Tensor]   # ffmpeg を Popen、固定長ずつ読む (3, size, size) uint8
def prepare(src: Path, out: Path, *, size: int = 64, fps: float = 10.0, clips: int = 64,
            length: int = 48) -> None                  # stream.u8 + clips.u8 + index.json（一時ファイル + os.replace）
class WTStream:                                         # B 本 = 長い動画の B 個の開始位置（i T / B）から並べて流す
    def __init__(self, path: Path, *, streams: int, seed: int, cache: Path | None = None) -> None
    def next(self) -> tuple[Tensor, Tensor]             # (B, 3, size, size) float, reset (B,)
```

- ffmpeg のフィルタ: `fps=<fps>,scale=-2:<size>:flags=area,crop=<size>:<size>`（中央を切る。16:9 の両端を捨てる）。4K のデコードは CPU を食うので、動画ごとに 1 job（`task=prepare`、11 本を並列）にして、フリートの CPU スロットで回す
- テストは ffmpeg で作る小さな色の動画（旧 test_egodex の作り方）。ダウンロードはテストに入れない

### 8.3 後から学ぶデコーダ（`exp/video/decode.py`、task=decode）

```python
def run_decode(config: RunConfig, rd: RunDir) -> None
    # 対象の run の config.yaml から同じネットワークを作り、ckpt/*.pt を順に load_modules
    # エンコーダを止めたまま、ストリームの先頭 frames フレームで z を計算 → fit_decoder → 評価クリップで
    # recon, img{h}（M16: 動力学が予測した z を復号）, pix_copy{h}, pix_mean{h} を decode/evals.jsonl へ
    # decode/<t>.png（行: 真 / z の復号 / 予測の復号、列: COLUMNS）
def fit_decoder(z: Tensor, images: Tensor, *, steps: int, seed: int = 0) -> nn.Module   # pc.optim.AdamW で step
```

`fleet submit m15/balls ... --then decode` が、学習の run の後に `after` 付きで decode の job を自動で積む（checkpoint ごとに独立なので、`decode.ckpt=<t>` で 1 ckpt 1 job に割ることもできる）。

### 8.4 M16（`exp/video/joint.py`）

```python
class JointNetwork(mujou.Network):
    """pc.Input(frames) -> JEPA (M15 の設定) -> z -> DynamicsPart (M14 の設定). The dynamics
    gets z as constants: Dynamics.z and Compare.target never send credit."""
    def __init__(self, config: RunConfig) -> None
class JointBPNetwork(mujou.Network):
    """pc.Input -> JEPA の BP ノード -> DynamicsBP（M14 のもの。入力が z）の 2 ノード。"""
```

- 観測の標準化の代わりに z の尺度をどう揃えるか（`z_norm` か、オンラインの running 統計）は M16 の着手時に決める
- **要のオラクル**: 同じ seed で K フレーム流したとき、エンコーダの重みが「動力学を付けた run」と「付けない run（JEPANetwork 単独）」でビット一致する。credit が動力学からエンコーダへ流れないことの直接の証拠
- 判定: decode の `img{h}` を `pix_mean{h}` と `pix_copy{h}` と比べ、PNG で玉が正しく動くかを見る

---

## 9. 実装の計画（実験を早くフリートに載せる順）

ブランチ: インフラは `refactor/2026-11-03/fleet-and-m14-infra`（PR 1 本）、M14 の実験は `exp/2026-11-0X/continuous-stream-m14-dynamics`。インフラの PR の中で M14 の部品まで作り、段 0 を回し始める。

### 段階 0（オーケストレーター、直列、半日弱）

1. Mac で `git submodule update --init`（sync のミラーの前提）
2. `uv add hydra-core optuna`、`uv.lock` を commit
3. `compose.yaml`: `/fleet` のマウント、GPU の reservation（要れば）、EgoDex のマウントを消す。各ホストの `.env` に `FLEET_DIR`。4 台のコンテナを作り直す
4. 確かめること（結果を `memory/agents/orchestrator/` に 1 枚）: コンテナから `/fleet` に vscode で書けるか、`os.rename` の claim が 2 ホストで取り合って片方だけ成功するか、コンテナで CUDA が見えるか、Mac の rsync が `--filter=':- .gitignore'` を解するか（openrsync なら `brew install rsync`）、32 コアが物理か論理か
5. 旧モジュールとテストを消す 1 commit（§1.2）。以後の WP は新しいファイルだけを作るので衝突しない
6. 下の WP の境界になる型とシグネチャ（§2.2、§3.2、§7）を `exp/video/config.py` と各ファイルのスタブ（docstring と `raise NotImplementedError`）で先に置く

### 段階 1（並列、WP ごとに別のエージェント。ファイルが重ならない）

| WP | 持つファイル | 依存 | 完了の条件 |
| -- | ------------ | ---- | ---------- |
| A フリート | `exp/lib/fleet/*`、`fleet.toml`、`justfile` の `fleet`、`tests/lib/fleet/*` | §3.2 の job.json | 偽の run で submit → claim → 起動 → result → status が回る。`fleet x all -- nproc` |
| B run の契約と設定 | `exp/lib/rundir.py`、`exp/video/config.py`、`exp/video/conf/**`、`exp/video/run.py`、`tests/lib/test_rundir.py`、`tests/video/test_config.py`、`test_run.py` | C・D・E のスタブ | 全 preset が compose できる、`task=baseline` が契約のファイルを書く、GIL のテスト |
| C データ | `exp/video/balls.py`、`exp/video/observe.py`、テスト | なし | ビット一致とオラクル。B=128・30 分の物理が 1 分未満 |
| D モデル | `exp/video/dynamics.py`、`exp/video/world.py`、`exp/lib/optim.py`（縮める）、テスト | C の `ObsStream` の形だけ | 平衡点・初期値・トレース・局所性のオラクル |
| E 物差しと見た目 | `exp/video/measure.py`、`exp/video/viz.py`、`exp/lib/png.py`、テスト | C | 等速・写し・真の物理のオラクル、PNG の重心 |
| F 集計 | `exp/lib/runs.py`、`tests/lib/test_runs.py` | §3.2 | 偽の run dir で seed を束ねた Markdown |
| J1 Walking Tours（M15 の先行） | `exp/video/wt.py`、`tests/video/test_wt.py` | B の `task` 分岐（`prepare` を足す） | 合成の mp4 で形・色・中央 crop。実データの前処理を job で回す |

A・B・C・D・E・F・J1 は同時に始められる。`run.py` の `TASKS` の表だけが B と他の WP の接点で、B が持つ。

### 段階 2（最初の実験。段階 1 の A・B・C・E がそろった時点）

1. `fleet bench`: `task=bench` を small/big × cpu/cuda で各ホストに投げ、fps と同時本数ごとの総スループット（1・8・16・28 本）を測る → `fleet.toml` の `slots` を決める
2. **M14 手順 1（物差し）**: `fleet submit m14/s0 task=baseline` と s1・s2（数秒）。表から閾値（目安: 等速の 0.5 倍）を確かめ、README に「回す前に確定した判定基準」を書く
3. D がそろったら **M14 段 0 のオンライン（BP、grid）**: `fleet submit m14/s0 --grid dynamics=small,big dynamics.trace=true,false dynamics.residual=true,false optim.lr=0.01,0.03,0.1 --seeds 0,1 --tag s0-grid`（48 本、600 秒）
4. 同時に **オフラインの上限**（WP G）を段 0〜2 で

### 段階 3（探索を自動に）

| WP | 持つファイル | 依存 |
| -- | ------------ | ---- |
| G オフライン | `exp/video/offline.py`、`tests/video/test_offline.py` | C・D・E |
| H optuna | `exp/video/tune.py`、`exp/video/studies/*.yaml`、`tests/video/test_tune.py`、`fleet tune` の転送（A に小さな追加） | A・B・F |
| I 図 | `exp/video/figs.py`、`justfile` の `fig` | F |

→ 段 1・2 の BP の study（各 96 trial × 2 seed）→ `tune confirm` で PC-ALM と BP × 3 seed × 30 分 → 段 1・2 が通ったら段 3（`obs=image`、GPU の job）の組み方と条件を README に書いて回す。

### 段階 4（M15・M16）

| WP | 持つファイル | 依存 |
| -- | ------------ | ---- |
| J2 ViT とマスク | `exp/video/vit.py`、`tests/video/test_vit.py` | なし（M14 と並べてよい） |
| J3 JEPA ノード | `exp/video/jepa.py`、`tests/video/test_jepa.py`、`conf/experiment/m15/*` | J2、B |
| J4 デコーダ | `exp/video/decode.py`、`tests/video/test_decode.py` | B、J3 の `JEPANetwork` |
| K 同時の学習 | `exp/video/joint.py`、`tests/video/test_joint.py`、`conf/experiment/m16/*` | D、J3 |

J2 は M14 の段階 1 と同時に始めてよい。M15 は合成の玉（`m15/balls`、probe が使える）から始め、J1 の前処理が終わったら Walking Tours に移る。

### 見積もり（測る前の目安）

- 段階 0〜1: 1〜1.5 日（WP を並列にして）。M14 の物差しは 1 日目に出る
- フリートの容量: 4 台 × 約 28 スロット ≈ 110 本を同時に。BP・600 秒の探索の run が 1 本 10〜30 分なら、1 時間に 200〜600 本。段 1 の study 1 つ（192 本）は 1〜2 時間
- PC-ALM + トレースの確認（30 分のストリームで 5〜7 時間）も、6 本 × 設定 3 つ = 18 本なら同時に回り、半日で終わる

---

## 10. 却下した案

| 案 | 却下の理由 |
| -- | ---------- |
| Ray / Slurm / Dask / Kubernetes | 4 台・1 run 1 プロセスには重い。ホスト同士が SSH できず、常駐のサーバーが要る |
| hydra の multirun と launcher（submitit、joblib）、hydra-optuna-sweeper | 1 台の中の掃引しかできず、run dir の作法が hydra に縛られる。optuna の pruning を外から効かせにくい |
| optuna を学習のプロセスに入れる | GIL のリスクと、学習のプロセスが storage に依存すること（NAS の sqlite のロック）。調整役 1 つに閉じる方が単純 |
| git で同期（commit → push → 各ホストで pull） | 1 周ごとに commit が要り、未完成のコードで試せない。差分を記録すれば再現性は同じ |
| run をローカルに書いて後で同期 | 途中経過（pruning、status）を読む経路が別に要る |
| trackio・MLflow・W&B | 3.14t に入らないか、サーバーが要る。読むのがエージェントで、表と PNG で足りる |
| 学習の再開（mujou の state_dict から） | 最長の run でも投げ直しで足りる。必要になったら `rundir.checkpoint` の木から足せる |
| 1 プロセスに複数の run をスレッドで並べる（free-threaded を活かす） | メモリは節約できるが、1 本の失敗が他を巻き込み、スロットの割り当てが複雑になる。1 run 1 プロセスのまま |

## 11. 未確定で、ユーザーに確認したいこと（推奨案で進める）

1. 探索のストリームを 600 秒（完了の条件は 30 分）にしてよいか。推奨: 探索 600 秒、確認 1800 秒。短い run の順位が 30 分でも保たれるかは、段 0 で 600 秒と 1800 秒の両方を回して確かめる
2. Whitened・Projected・Orthogonalized を消してよいか（M10 で SGD を超えなかった）。推奨: 消す。M14 で比べ直す optimizer は SGD・AdamW・Adam の向き + grafting
3. EgoDex のコードとマウントを消してよいか。推奨: 消す（git の履歴に残る）
4. NAS の `/nas/geson/mujou-exp/` を run の出力の正にしてよいか（Mac の `runs/` は pull した写し）
