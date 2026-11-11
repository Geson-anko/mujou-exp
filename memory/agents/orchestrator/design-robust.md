# 再編の設計案: 堅牢性と再現性を軸に（M14〜M16、4 台のフリート）

2026-11-03、orchestrator 配下の設計サブエージェント。既存のファイルは変更していない。読んだもの: `m14-m16-constraints.md`、`exp/video/MILESTONES.md`、`map-{model,run,data,lib,mujou,history,tests}.md`、`exp/lib/{record,sweep,host}.py`、`exp/video/{config,run}.py` の一部、`compose.yaml`、`.devcontainer/devcontainer.json`、`pyproject.toml`、`justfile`、`mujou/src/mujou/pc/{input,layer}.py` のポート定義。

**レンズ**: 何百本の run を 4 台に並べても (1) 黙って失敗する run が無い、(2) ホストが落ちてもきれいに戻る、(3) どの run も同じ結果を再び出せる、(4) 集計を信用できる。速さはこの 4 つを崩さない範囲で取る。

---

## 0. 設計を決めた事実（2026-11-03 に ssh で確かめた）

| 事実 | 設計への帰結 |
| ---- | ------------ |
| 4 台とも 32 コア、61〜62 GB、GPU0 = RTX 4090（24 GB）、GPU1 = RTX 4060（8 GB）。`/data` は btrfs のローカル NVMe（1.6 TB 以上空き） | 実行中の run の出力はローカルの `/data` に書く |
| NAS は `//192.168.100.190/public` を CIFS で `/nas` に。マウントの option は **`soft`、`nobrl`**、`actimeo=1` | `soft` は NAS が一瞬止まると書き込みが EIO で失敗する。`nobrl` だとバイト範囲のロックがホストの中でしか効かない。**sqlite（optuna・trackio）も、実行中の run の書き込みも NAS に置かない**。NAS は、完了した run の保管と、前処理したデータセットの配布元にだけ使う |
| コンテナ `mujou-exp-dev-1` から GPU は 2 枚見える。`/nas` はコンテナに無い（egodex と walking_tours だけをマウント）。ワークスペースはホストと同じパス `/data/geson/mujou-exp` で bind されている | フリート用のディレクトリを compose.yaml でマウントする（§4.2） |
| コンテナの venv は 3.14.8t、torch 2.15.0+cu132、`sys._is_gil_enabled()` は False。**numpy は入っていない**。コンテナに system の python3 は無い | 学習のプロセスに optuna（numpy に依存する）は入れない。ホスト側のエージェントはホストの python3（3.12）で動かす |
| ホストには `python3` 3.12、`rsync`、`flock`、`tmux` がある | ホストのエージェントは stdlib だけで書き、Python 3.11 以上で動くようにする |
| Mac の `/usr/bin/python3` は 3.9、`rsync` は openrsync（protocol 29）。uv には 3.13 と 3.14t がある | Mac で動くランチャーは `uv run --no-project --python 3.13` で起動する。rsync は Homebrew の GNU rsync に固定する（openrsync は `--partial` などの扱いが不確か） |
| `pc.Input.next` は `MultiPort(send=act_spec, recv=x)` | Input に繋ぐ相手は `send=x` を宣言しなければならない。「credit を送らない」ことを型で保証するには、送るだけのポートを持つ入力ノードを別に作る（§7.3） |
| `/nas/geson/trackio/` に別プロジェクトの trackio の sqlite がある | trackio の DB を CIFS に置くと上の `nobrl` の問題に当たる。採らない理由の 1 つ（§6） |

---

## 1. パッケージの構成

### 1.1 方針

- **3 層の規則はそのまま**。フリートのランチャーは「実験の運用の道具」だが、stdlib だけで動き、Python の版も違う（Mac 3.13 / ホスト 3.12）。そこで `exp/` の外に、トップレベルのパッケージ `fleet/` を置く。`fleet/` は `exp/` も torch も import しない。`exp/` も `fleet/` を import しない。両者の境界は「ジョブの仕様（`job.json`）」と「run のディレクトリの契約」（§3）というファイルの形だけにする
- 学習のプロセスが従う共通の骨格（心拍・評価・checkpoint・再開・発散・終了コード）は `exp/lib/program.py` に 1 つだけ置く。M14・M15・M16 の 3 つのプログラムが同じ骨格を通るので、堅牢性の性質を 1 か所で試験できる
- **mujou に先に移すものは無い**。M14 は mujou 881c4ee のままで組める。候補は 2 つ（§1.4）で、M16 と M15 で確かめてから PR にする

### 1.2 再編後の木

```text
fleet/                         # stdlib のみ。Python >= 3.11。Mac とホストの両方で動く
  __init__.py
  __main__.py                  # python -m fleet <cmd>
  README.md                    # 日本語。運用の手順
  hosts.toml                   # フリートの定義（commit する。§4.3）
  model.py                     # dataclass: Host, Gpu, Slot, Resources, JobSpec, Attempt, State(Enum), ExitClass(Enum)
  db.py                        # Mac の sqlite（WAL）。jobs / attempts / events / hosts / batches。schema の版と migration
  transport.py                 # Transport(Protocol).run(host, argv, stdin, timeout) -> Completed。Ssh(ControlMaster) と Local（テスト用）
  snapshot.py                  # 作業ツリー（未 commit の変更と mujou を含む）から snapshot commit を作り、ホストへ push
  provision.py                 # ホストの code/<snap> の worktree と venv/<envkey> を作る（ジョブとして流す）
  agent.py                     # ホストに置くエージェント: start / supervise / stop / status / archive / gc / health
  resolve.py                   # preset + override を、固定した code の中で hydra に解決させる（バッチ、ssh 1 回）
  schedule.py                  # 純関数: (queue, slots, running) -> placements。単体で試験する
  reconcile.py                 # 純関数: (DB の状態, エージェントの報告) -> 遷移と命令
  sync.py                      # 出力の pull（rsync）、manifest の検証、NAS への保管の指示
  daemon.py                    # tick: reconcile -> schedule -> start -> sync -> archive -> events
  index.py                     # runs/index.sqlite3 を、同期済みで検証済みの run から作り直す（キャッシュ）
  table.py                     # README の表、report.html 用の JSON、図のデータ
  cli.py                       # argparse。既定の出力は JSON（AI のオーケストレーターが読む）
  opt/                         # optuna を使う部分だけ。別の uv プロジェクト
    pyproject.toml             # requires-python = "==3.13.*"、optuna==5.0.0、optuna-dashboard
    uv.lock
    coordinator.py             # ask/tell のループ
    space.py                   # TOML の探索空間 -> optuna の分布 -> hydra の override

exp/
  README.md                    # 層の説明を更新（fleet/ との境界を足す）
  lib/
    host.py                    # Host（frames を復元できるように）。RealtimeHost は消す
    metrics.py                 # そのまま（effective_rank, probe_r2）
    mingru.py                  # MinGRUTrace を mujou.Stateful にする（再開で状態が欠けないように。§3.6）
    sigreg.py                  # そのまま（M15 の LeJEPA で使う）
    optim.py                   # Warmup, AdamDirection, Grafted だけ残す
    rundir.py                  # 新: run のディレクトリ（原子的な書き込み、jsonl と offset、状態、停止の要求、manifest）
    provenance.py              # 新: コードの版、環境、GIL の検査、デバイスの事実
    jobspec.py                 # 新: job.json の読み書き、正規化した JSON、cfg_hash / group_hash
    conf.py                    # 新: hydra の compose の薄い包み。`python -m exp.lib.conf resolve --batch`
    resume.py                  # 新: 再開用の checkpoint（mujou の state_dict + ホストの状態）、評価用の checkpoint（重みだけ）
    program.py                 # 新: 学習のプロセスの共通の骨格（§3.7）
    png.py                     # 新: write_png, tile, 点と線のラスタ（decode.py から移す）
    decoder.py                 # M15: 後から学ぶデコーダ（torch.optim を使う。補助のプロセス専用）
    video_io.py                # M15: ffmpeg のストリーミングの読み出しと u8 memmap への前処理（egodex.py を一般化）
  video/
    README.md  PLAN.md  MILESTONES.md  results/  report*.html   # 残す
    schema.py                  # 構造化された設定の dataclass（torch を import しない）
    conf/                      # hydra の yaml（§2）
    studies/                   # optuna の study の定義（TOML。§5）
    balls.py                   # 物理。Video は参照の実装として残し、ベクトル化した BallStreams を足す
    obs.py                     # M14: 状態から観測、標準化、逆変換
    baselines.py               # M14: 平均・写し・等速の外挿・真の物理
    nodes.py                   # Source, Dynamics, DynamicsLayer, Compare（書き直す。§7.3）
    dynamics.py                # build_modules, DynamicsStack, Top, DynamicsNet（PC-ALM）, DynamicsBP, DynamicsBPNet
    rollout.py                 # 観測の空間の rollout と物差し（evaluate.py の後継）
    run_dynamics.py            # プログラム: M14 のオンラインの run
    offline_dynamics.py        # 補助: M14 のオフラインの上限（torch.optim 可）
    viz_dynamics.py            # 補助: M14 の見た目の確認（PNG）
    chunks.py                  # M15: N フレームのリングバッファ
    vit.py                     # M15: 小さな ViT（サンプルごとに独立、LayerNorm）
    jepa.py                    # M15: JEPA ノード（エンコーダ + 予測器 + SIGReg、ノードの中の BP）
    walking_tours.py           # M15: 前処理の計画と WalkingToursStream
    run_jepa.py                # プログラム: M15
    decode_ckpt.py             # 補助: M15・M16 の各 checkpoint にデコーダを学んで PNG と指標
    run_joint.py               # プログラム: M16（JEPA ノード -> 動力学の部分）
```

### 1.3 消すもの

| 消す | 理由・引き継ぎ |
| ---- | -------------- |
| `exp/lib/{config,record,sweep,linear_stream}.py` | `from_dict` は hydra/OmegaConf に、`Recorder` は `rundir`・`provenance` に、`run_all` は `fleet` に置き換える。段 0 の線形回帰は M14 段 0 が置き換える |
| `exp/lib/optim.py` の `Whitened`・`Orthogonalized`・`Projected`・`Correlation`・`_Factors`・`_eigh` ほか | M10 で SGD を超えなかった。`Correlation` は診断専用で、可変の状態を持ち再開に乗らない。戻すときは git の履歴から |
| `exp/video/{model,bp,evaluate,decode,offline,config,run,easy,egodex,frozen,latency,stability,collapse,stage0,stage1,stage2,stage3}.py` | 動力学は `nodes`・`dynamics`・`rollout` に作り直す。エンコーダの CNN、`Reconstruct`、SIGReg を混ぜた `WorldBP`、EgoDex、M6〜M13 の掃引スクリプトは使わない。`easy.EASY` の物理は `conf/data/balls_easy.yaml` に移してから消す |
| 対応するテスト: `tests/lib/{test_config,test_record,test_linear_stream}.py`、`tests/video/{test_model,test_bp,test_evaluate,test_decode,test_offline,test_config,test_run,test_egodex,test_stage2}.py` | オラクルの型（平衡点の `lam` = BP の credit、トレース = BPTT、恒等モデル = 写し、`window_starts`、PNG の手での復号）は新しいテストに移す（§7.6） |
| `checkpoint` の pickle（`torch.save(Snapshot)`） | `weights_only=True` で読める state_dict だけにする |

`results/*.md` と `report*.html` の中のコマンドは当時の記録として残す（ファイル名とパスは当時のもの、と既に書いてある）。

### 1.4 mujou に送る候補（いまは送らない）

1. **credit を受け取らない入力**（`pc.Input(..., credit=False)` か、送るだけのポートを持つ `pc.Source`）。M14 と M16 で `exp/video/nodes.py` の `Source` として使い、「受け手が送り返せない」性質が役に立つと分かってから。mujou の利用者に「下流から credit を戻さない観測」と説明できる
2. **ノードの中の BP で自分の損失を学ぶノード**（M15 の JEPA ノードの一般形）。M15 で形が定まってから

どちらも局所性（ノードをまたぐ `.backward()` なし）と非同期性（バリアなし）に触れない。

### 1.5 テストの構成

```text
tests/
  conftest.py                  # OMP_NUM_THREADS=1。cuda(n) マーカーの skip の hook をここで実装する（今は無い）
  test_claude_hooks.py  test_mujou_install.py
  fleet/                       # stdlib だけ。ssh を使わない（LocalTransport と偽の docker）
    test_schedule.py  test_reconcile.py  test_agent.py  test_snapshot.py
    test_sync.py  test_db.py  test_index.py  test_table.py  test_cli.py
    opt/test_space.py          # optuna が無ければ importorskip
  lib/
    test_rundir.py  test_provenance.py  test_jobspec.py  test_conf.py  test_resume.py
    test_program.py  test_png.py  test_host.py  test_metrics.py  test_mingru.py
    test_sigreg.py  test_optim.py
  video/
    test_balls.py  test_obs.py  test_baselines.py  test_nodes.py  test_dynamics.py
    test_rollout.py  test_run_dynamics.py  test_offline_dynamics.py  test_viz_dynamics.py
    test_schema.py  test_conf_tree.py   # conf/ のすべての preset が解決でき、型が合う
    (M15) test_chunks.py  test_vit.py  test_jepa.py  test_walking_tours.py  test_decode_ckpt.py
    (M16) test_run_joint.py
```

`pyproject.toml` の `[tool.pyright] include` に `fleet` を足す。`fleet/opt/` は optuna が要るので pyright の対象から外すか、`fleet/opt/pyproject.toml` の環境で別に検査する。

---

## 2. hydra の設定の木

### 2.1 hydra の使い方を絞る

- **hydra は「合成（compose）」にだけ使う**。`@hydra.main`、hydra の出力ディレクトリ、logging、multirun、launcher・sweeper のプラグインは使わない。理由: (a) 出力ディレクトリの場所と名前はフリートが決める、(b) hydra の optuna sweeper は 1 プロセスの中で試行を回す前提で、§5 の中央の ask/tell と合わない、(c) 学習のプロセスに入れる依存を最小にする
- 合成は **`fleet submit` のときに、ジョブが走るのと同じ code snapshot の中で**行う（`fleet/resolve.py` が ssh 1 回でまとめて `python -m exp.lib.conf resolve --batch -` を呼ぶ）。結果の「解決済みの設定」（補間を解決した素の JSON）を `job.json` に入れる。こうすると
  - 設定の誤り（未知のキー、型の違い）は、キューで待つ前に submit の時点で落ちる
  - `cfg_hash` が投入の時点で決まるので、同じ run の二重投入を DB で弾ける
  - 学習のプロセスは hydra を import しない。`job.json` を OmegaConf で schema に当てて検証し、`OmegaConf.to_object` で dataclass にするだけ
- 手元で 1 本だけ試すときは `python -m exp.video.run_dynamics +experiment=m14/s0_explore optim.lr=0.01 --out /tmp/x` で、同じ関数で合成して `job.json` を書き、同じ経路で走る

### 2.2 構造化された設定（`exp/video/schema.py`、torch を import しない）

OmegaConf は dataclass の Union を扱えない。そこで **プログラムごとに根の schema を分け**、各フィールドの型を具体的な dataclass にする。種類は `Enum`（OmegaConf も pyright も扱える）で表し、`match` で分岐する。補間や独自の resolver は使わない（pyright から見えないため）。学習率から時定数への換算は、resolver ではなく property で書く。

```python
class Learner(Enum):
    PC = "pc"
    BP = "bp"
    NONE = "none"        # 学習しない（物差しだけ測る run）

class OptimKind(Enum):
    SGD = "sgd"
    ADAMW = "adamw"
    GRAFT_ADAM = "graft_adam"   # Grafted(inner=AdamDirection)

@dataclass
class TimeCfg:
    frame: float = 1 / 30
    h: float = 4e-3
    lam: float = 2e-3
    r: float = 4e-3
    learn_interval: float | None = 25e-3

@dataclass
class OptimCfg:
    kind: OptimKind = OptimKind.SGD
    lr: float = 0.03            # per frame; tau = frame / lr
    warmup: float = 300.0       # seconds of the run's timer
    tau_m: float = 0.3
    tau_v: float = 30.0
    weight_decay: float = 0.0

@dataclass
class BallsCfg:                 # easy.EASY のもの
    seconds: float = 1800.0
    streams: int = 128
    radii: list[float] = field(default_factory=lambda: [0.10])
    gamma0: float = 1.0
    gamma_amp: float = 0.0
    e0: float = 1.0
    e_amp: float = 0.0
    drag: float = 0.0
    conserve: bool = True
    substeps: int = 8
    size: int = 32              # stage 3 and rendering

class ObsKind(Enum):
    POS_VEL = "pos_vel"         # stage 0
    POS = "pos"                 # stage 1, 2
    IMAGE = "image"             # stage 3

@dataclass
class ObsCfg:
    kind: ObsKind = ObsKind.POS_VEL

@dataclass
class DynamicsCfg:
    d_s: int = 256
    nodes: int = 1
    trace: bool = False
    residual: bool = False
    open_loop: float = 0.0
    open_cycle: int = 20

@dataclass
class EvalCfg:
    interval: float = 60.0
    observed: int = 16
    horizons: list[int] = field(default_factory=lambda: [1, 5, 10, 20])
    phases: int = 8
    clips: int = 16             # 32 から増やす（物差しの標本を安定させる）
    length: int = 128
    seed: int = 12345           # run の seed に依らない: 物差しはどの seed でも同じ
    checkpoint_interval: float = 300.0

@dataclass
class CtlCfg:                   # 実行の制御（結果に影響しないものは hash から外す。§2.5）
    resume_interval: float = 300.0     # 仮想時間の秒ごとに再開用の checkpoint
    diverge_factor: float = 10.0       # err > factor * 平均の物差し が diverge_window 秒続いたら発散
    diverge_window: float = 10.0
    frame_log_every: float = 1.0       # 毎フレームの指標をこの秒で平均して 1 行

@dataclass
class Meta:
    name: str = MISSING
    milestone: str = MISSING           # "m14"
    phase: str = "explore"             # baseline | calibrate | explore | confirm | offline | audit
    tags: dict[str, str] = field(default_factory=dict)

@dataclass
class DynamicsRun:                     # program = exp.video.run_dynamics
    schema: int = 1
    seed: int = 0
    time: TimeCfg = field(default_factory=TimeCfg)
    data: BallsCfg = field(default_factory=BallsCfg)
    obs: ObsCfg = field(default_factory=ObsCfg)
    dynamics: DynamicsCfg = field(default_factory=DynamicsCfg)
    learner: Learner = Learner.BP
    optim: OptimCfg = field(default_factory=OptimCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    ctl: CtlCfg = field(default_factory=CtlCfg)
    meta: Meta = field(default_factory=Meta)
```

M15 は `JepaRun`（`data: VideoDataCfg`、`encoder: VitCfg`、`jepa: JepaCfg`、`optim`、`eval: JepaEvalCfg`）、M16 は `JointRun`（`JepaRun` の部分 + `dynamics` + `dyn_optim` + `learner_dyn` + `learner_enc`）。M16 の動力学の部分は `DynamicsCfg` と `OptimCfg` をそのまま使う（M14 の設定を持ち込める）。

`ConfigStore` への登録は `exp/video/schema.py` の `register()` 1 つにまとめる（`cs.store(name="dynamics_run_schema", node=DynamicsRun)` と、各 group の schema）。`exp.lib.conf` は「config のディレクトリ」と「register 関数」を受け取る汎用の関数にし、`exp/video/` を import しない。

### 2.3 yaml の木（`exp/video/conf/`）

```text
conf/
  dynamics_run.yaml           # defaults: [dynamics_run_schema, data: balls_easy, obs: s0, dynamics: small,
                              #            optim: sgd, _self_]   learner: bp
  jepa_run.yaml
  joint_run.yaml
  data/      balls_easy.yaml (玉 1 つ)  balls_easy3.yaml (玉 3 つ)  balls_video.yaml  walking_tours64.yaml
  obs/       s0.yaml (pos_vel, 玉 1)  s1.yaml (pos, 玉 1)  s2.yaml (pos, 玉 3。data も balls_easy3 に)  s3.yaml (image)
  dynamics/  small.yaml (d_s 256, nodes 1)  big.yaml (d_s 512, nodes 3)
  optim/     sgd.yaml  adamw.yaml  graft_adam.yaml
  encoder/   vit_t.yaml  vit_s.yaml
  jepa/      tube4.yaml  image.yaml (N=1)
  experiment/
    m14/ s0_baseline.yaml  s0_calibrate.yaml  s0_explore.yaml  s0_confirm.yaml
         s1_explore.yaml   s1_confirm.yaml    s2_explore.yaml  s2_confirm.yaml
         s3_explore.yaml   s3_confirm.yaml    offline.yaml
    m15/ balls_explore.yaml  wt_explore.yaml  wt_confirm.yaml
    m16/ balls.yaml  wt.yaml
```

段は `obs` の group で選ぶ。段 2 は玉が 3 つなので、`obs/s2.yaml` が `defaults: [override /data: balls_easy3]` を持つ。各 preset は `# @package _global_` で、例えば:

```yaml
# experiment/m14/s1_explore.yaml
# @package _global_
defaults:
  - override /obs: s1
  - override /dynamics: big
meta: {name: m14-s1-explore, milestone: m14, phase: explore}
learner: bp
data: {seconds: 600.0}
eval: {interval: 60.0, checkpoint_interval: 0.0}
```

`confirm` の preset は `seconds: 1800`、`checkpoint_interval: 300` で、learner と seed は投入するときに掃引する。

### 2.4 呼び出し方

```text
fleet submit m14/s1_confirm optim.lr=0.03 dynamics.trace=true \
    --sweep learner=pc,bp --seeds 0,1,2 --batch m14-s1-confirm-v1 --gpu 4090
```

- `--sweep k=a,b` は直積に展開する（optuna を使わない格子）。`--seeds` は `seed=` の掃引の略
- 学習のプロセスの起動: `python -m exp.video.run_dynamics --job <run_dir>/job.json --attempt <n>`（エージェントが起動する。人は直接呼ばない）

### 2.5 run の名前・タグ・hash

- **`cfg_hash`** = 解決済みの設定から `meta` と `ctl` を除き、正規化した JSON（`sort_keys`、`separators=(",", ":")`、float は `repr`、Enum は値）の sha256 の先頭 12 文字。`ctl` を除けるのは「結果に影響しない」ことを試験で保証するときだけ（`test_ctl_does_not_change_the_metrics`。`resume_interval` を変えても `evals.jsonl` のバイト列が同じ）。`diverge_*` は打ち切りの時刻を変えるので `ctl` には入れず、`eval` 側に置く案もある。ここでは「発散した run の最後の値は比べない」と決めて `ctl` に置く
- **`group_hash`** = `cfg_hash` の計算からさらに `seed` を除いたもの。seed の違う run を 1 つの「設定」にまとめる鍵
- **`run_key`** = `<meta.name>-<override の短い表記>-<cfg_hash[:8]>-<code_snap[:8]>`。override の表記は、利用者が書いた override をキーの順に並べ `.` 以外の記号を `-` に（例: `learner-pc_optim.lr-0.03_seed-1`）。60 文字を超えたら override の部分を省く（hash で一意）
- コードの版を `run_key` に入れる。同じ設定でもコードが違えば別の run である。集計は既定で `group_hash` ごとにまとめ、**1 つの表の中でコードの版が混ざったら止める**（§6.3）
- タグ（`meta.milestone`、`meta.phase`、`meta.tags`、それに `study`、`trial`、`batch`）は hash に入れない。DB と `index.sqlite3` に列として入り、検索に使う

---

## 3. run の契約

### 3.1 原則

1. **指標のファイルは (コード, 設定) の純関数**。壁時計の時刻、ホスト名、PID は `evals.jsonl` と `frames.jsonl` に入れない（`status.json` と `timing.jsonl` に分ける）。だから 2 つの run の指標を**バイト列の sha256 で比べられる**。再現の検査（§3.9）と再開の試験（§3.6）がこれに乗る
2. **書いたものは壊れない**。JSON は一時ファイルに書いて fsync してから `os.replace`。jsonl は行ごとに書き、評価と checkpoint のたびに flush と fsync。再開のときは checkpoint に記録した byte offset で切り詰める（重複行が出ない）
3. **完了の印は最後に書く `manifest.json`**。これが無い run は、終了コードが 0 でも完了とみなさない
4. **終了の理由を、run ではなく外側（supervisor）が書く**（`exit/attempt-<n>.json`）。segfault や OOM killer で run が自分では何も書けなくても、理由が残る

### 3.2 ディレクトリ

ホストの `/data/geson/fleet/runs/<batch>/<run_key>/`（コンテナからも同じパス）。

```text
<run_key>/
  job.json              不変。fleet が書く（§3.3）
  config.yaml           解決済みの設定（人が読む用。job.json の config と同じ内容）
  code.diff             base_head から snapshot までの差分（mujou の差分も）。人が読む用
  provenance/attempt-1.json   その試行の環境（§3.4）
  status.json           可変。心拍（§3.5）
  timing.jsonl          壁時計の記録（仮想時間 t と壁時計の経過、frames/s、RSS、GPU メモリ）
  metrics/frames.jsonl  1 行 = 仮想時間 frame_log_every 秒（30 フレーム）の平均
  metrics/evals.jsonl   1 行 = 1 回の評価
  result.json           終わったときだけ（completed / diverged）
  manifest.json         最後に書く。resume/ と logs/ 以外の全ファイルの sha256 とサイズ
  ckpt/<t:07.1f>.pt     評価用の checkpoint（モジュールの重みだけ、weights_only=True で読める）
  resume/<frame:08d>.pt 再開用（最新 2 つだけ残す。完了したら消す。同期しない）
  resume/latest         最新の再開用 checkpoint の名前（原子的に書き換える）
  logs/attempt-1.out  logs/attempt-1.err   標準出力と標準エラー（faulthandler の出力も）
  exit/attempt-1.json   supervisor が書く終了の記録
  stop.request          停止の要求（fleet が書く。§4.6）
  viz/  decode/         補助のジョブの出力
```

### 3.3 `job.json`

```json
{
  "schema": 1,
  "run_key": "m14-s1-confirm-learner-pc_seed-1-3fa29c01-8e1d2a77",
  "batch": "m14-s1-confirm-v1",
  "study": null, "trial": null,
  "program": "exp.video.run_dynamics",
  "code": {"snap": "<40 hex>", "base_head": "<40 hex>", "dirty": false,
           "mujou_snap": "<40 hex>", "envkey": "<12 hex>"},
  "config": { "...解決済み..." },
  "overrides": ["learner=pc", "seed=1", "optim.lr=0.03"],
  "cfg_hash": "3fa29c01ab12", "group_hash": "77c0e1d9aa03",
  "resources": {"cpus": 1, "ram_gb": 2.0, "gpu": "4090", "gpu_mem_gb": 1.5},
  "retry": {"max_infra": 3, "max_failed": 0, "resumable": true},
  "after": [], "same_host_as": null,
  "created": "2026-11-03T12:00:00+09:00"
}
```

同じ `run_key` の run ディレクトリが既にあれば、エージェントは `job.json` が**バイト単位で同じとき**だけ再開を許す（違えば `conflict` で起動を拒む）。

### 3.4 provenance（`exp/lib/provenance.py`）

試行の最初に書く。

```python
def collect(job: JobSpec, *, attempt: int) -> dict[str, Any]
def verify_code(job: JobSpec, cwd: Path) -> None         # HEAD == job.code.snap かつ git status が空。違えば EnvError
def assert_free_threaded(where: str) -> None             # §3.8
def configure_determinism(device: str) -> dict[str, Any] # 下の設定をして、その値を返す
```

中身: `attempt`、`host`（`FLEET_HOST`）、`container`、`python`（`sys.version`）、`gil_enabled: false`、`torch`、`cuda`、`cudnn`、GPU の名前とドライバ（`torch.cuda.get_device_name`）、CPU のモデル（`/proc/cpuinfo`）、`env`（`OMP_NUM_THREADS`、`MKL_NUM_THREADS`、`CUDA_VISIBLE_DEVICES`、`CUBLAS_WORKSPACE_CONFIG`、`PYTHON_GIL` の有無）、`taskset` の affinity（`os.sched_getaffinity(0)`）、`torch.get_num_threads()`、`deterministic`（TF32 を切ったか、`use_deterministic_algorithms`）、`data_version`（M15 の前処理したデータの hash）、`started`。

決定性の設定（`configure_determinism`）: `torch.set_num_threads(1)`、`torch.backends.cuda.matmul.allow_tf32 = False`、`torch.backends.cudnn.allow_tf32 = False`、`torch.use_deterministic_algorithms(True)`（決定的でない演算は**黙らずに例外**になる）、CUDA なら `CUBLAS_WORKSPACE_CONFIG=:4096:8` がエージェントから渡っていることを確かめる。`OMP_NUM_THREADS != "1"` なら起動を拒む（M13 で知られた数十分の停止の原因）。

**再現の単位**: 同じ (code snap, 設定, デバイスの種類) なら指標のバイト列が同じ、を保証する。CPU と CUDA、4090 と 4060 では違ってよい（カーネルが違う）。だからデバイスの種類（`cpu` / GPU の名前）を provenance と index に必ず残し、確認の batch は全 seed・全 learner を同じ種類に固定する（`--gpu 4090` か `--cpu`）。

### 3.5 `status.json`（心拍）

壁時計 30 秒ごと、評価のたび、終わるときに原子的に書き換える。

```json
{"state": "running", "attempt": 1, "pid": 412, "pgid": 412,
 "frame": 18000, "t": 600.0, "frames_total": 54000,
 "last_eval_t": 600.0, "last_ckpt_t": 600.0, "last_resume_frame": 18000,
 "fps": 41.2, "eta_s": 873.0, "rss_mb": 1310, "gpu_mem_mb": 912,
 "heartbeat_at": "2026-11-03T12:10:31+09:00", "beats": 41}
```

`frame` が進まずに `beats` だけが増えるなら「止まっている」（§4.7）。心拍の年齢はホストの時計で測る（Mac とホストの時計のずれに依らない）。

### 3.6 checkpoint と再開

2 種類を分ける。

| | 評価用 `ckpt/<t>.pt` | 再開用 `resume/<frame>.pt` |
| - | -------------------- | -------------------------- |
| 中身 | `{"schema", "program", "cfg_hash", "t", "frame", "modules": {name: state_dict}}`（M14 は `step`=`DynamicsStack`、`readout`。M15 は `encoder`、`predictor`） | `{"schema", "cfg_hash", "frame", "mujou": run.state_dict(), "host": HostState, "offsets": {"frames": int, "evals": int, "timing": int}}` |
| 読み方 | `torch.load(weights_only=True)` → `build_modules(cfg)` に `load_state_dict` | 同じ構造のネットワークを組み直し `mujou.testing.run(net, state=tree["mujou"])` |
| 頻度 | `eval.checkpoint_interval`（仮想時間） | `ctl.resume_interval`（仮想時間） |
| 保管 | 同期し、NAS にも保管 | ローカルだけ、最新 2 つ、完了で消す |

`HostState`（`exp/lib/resume.py`）: ストリームの状態（`BallStreams` の状態テンソルと `torch.Generator.get_state()`）、`Host.frames`、ストリームごとの予測の数 `count`、発散の判定の窓の状態、`frames.jsonl` の途中の平均の累積。

**欠けた状態は黙って結果を変える**。M13 の `MinGRUTrace` は `mujou.Stateful` でもテンソルでもない普通の属性なので、`run.state_dict()` に入らない。再開するとトレースが 0 から始まり、気づかれずに結果がずれる。そこで:

- `MinGRUTrace` を `mujou.Stateful` にする（`state_dict` / `load_state_dict`）
- **再開の試験を、すべてのプログラムで必須にする**: `test_resume_equals_uninterrupted`。K フレームで止めて再開し、さらに K フレーム回した `evals.jsonl`・`frames.jsonl` と、止めずに 2K フレーム回したものがバイト単位で一致する。どこかの状態が checkpoint から漏れていれば、この試験が落ちる

### 3.7 プログラムの骨格（`exp/lib/program.py`）

3 つのプログラムが同じ骨格を通る。プログラムは「組み立て」「1 フレーム」「評価」「保存する部品」だけを書く。

```python
class Program[C](Protocol):
    schema: type[C]
    def build(self, cfg: C, device: str) -> Built: ...           # ネットワーク、ストリーム、評価クリップ
    def frame(self, built: Built, host: Host) -> FrameRecord: ... # 1 フレーム見せて、その指標
    def evaluate(self, built: Built) -> dict[str, float]: ...    # step の合間に呼ぶ
    def modules(self, built: Built) -> dict[str, nn.Module]: ... # 評価用の checkpoint に入れるもの
    def host_state(self, built: Built) -> dict[str, Any]: ...
    def load_host_state(self, built: Built, state: dict[str, Any]) -> None: ...
    def objective(self, evals: dict[str, float]) -> dict[str, float]: ...  # §5 の目的値（README の物差しと同じ式）

def main(program: Program[Any], argv: list[str] | None = None) -> int   # 終了コードを返す
```

`main` がすること（順に）:

1. `job.json` を読み、schema で検証（`OmegaConf.merge(OmegaConf.structured(schema), config)` → `to_object`）。`verify_code`、`assert_free_threaded("import")`、`configure_determinism`、provenance を書く
2. `resume/latest` があれば再開（指標のファイルを offset で切り詰める）。無ければ新しく始める
3. `build` → `assert_free_threaded("build")`（optimizer を作った後。`torch.optim` が紛れ込めばここで落ちる）
4. フレームのループ: `frame` → 指標を累積して `frame_log_every` ごとに 1 行 → 発散の判定 → 評価の時刻なら `evaluate`（`assert_free_threaded("eval")` も）→ 評価用の checkpoint → 再開用の checkpoint → `stop.request` を見る → 心拍
5. 終わり: `result.json` → `manifest.json` → 終了コード

**終了コードの契約**（supervisor と reconcile がこれで分類する）:

| コード | 意味 | 書くもの | fleet の扱い |
| ------ | ---- | -------- | ------------ |
| 0 | 完了 | result（`status: completed`）、manifest | 完了（同期と検証の後に archived） |
| 3 | 発散（科学的な結果） | result（`status: diverged`、`reason`、`t`）、manifest | 完了の一種。表に「発散」と書く。再試行しない |
| 4 | 環境の誤り（GIL が戻った、コードの版が違う、データが無い、CUDA が見えない、`OMP_NUM_THREADS` が違う） | stderr に理由 | ホストの問題。そのホストでその種類のジョブを止め（cordon）、別のホストで再試行 |
| 75 | 停止の要求に従って止まった | 再開用の checkpoint | `stop.request` の理由に従う（pruned / 移動 / drain） |
| 1 など | 例外（コードの誤りの見込み） | stderr の traceback | 失敗。既定では再試行しない。同じエラーの印（例外の型 + 最後のフレーム）が batch の中で 3 回出たら batch を保留（§4.7） |
| シグナル | 外から殺された | supervisor が signal を記録 | 停止の要求が無い SIGKILL は OOM の見込み → ram を 1.5 倍にして 1 回だけ再試行 |

発散の判定: 予測が非有限、または `err > diverge_factor * 平均の物差し` が `diverge_window` 秒続く。**評価の値が非有限なら、それも発散として止める**（NaN を握りつぶさず、止めて記録する）。評価の中の例外は握りつぶさない（終了コード 1）。

### 3.8 GIL の検査

```python
def assert_free_threaded(where: str) -> None:
    """Raise EnvError when the GIL is enabled or a module known to enable it was imported."""
    # sys._is_gil_enabled() が True、または "torch._dynamo" が sys.modules にある
```

- 呼ぶところ: import の直後、ネットワークを組んだ後（optimizer の生成の後）、毎回の評価。失敗したら、直前の検査から増えた `sys.modules` の C 拡張の名前を stderr に出して終了コード 4
- エージェントは `PYTHON_GIL` を環境から消してから起動する（`PYTHON_GIL=0` で GIL を力ずくで止めると、非対応の拡張が壊れたまま動き、問題が隠れる）
- 補助のプログラム（`offline_dynamics`、`decode_ckpt`、`viz_dynamics`）は `torch.optim` を使うので、`Program` ではなく `aux_main` を通り、GIL の検査をしない。代わりに provenance に `gil_enabled` を記録する
- 試験: `tests/lib/test_provenance.py::test_gil_stays_disabled_through_the_program`（free-threaded のビルドでなければ skip）。小さなジョブを subprocess で走らせ、provenance の `gil_enabled` が false。`tests/lib/test_provenance.py::test_torch_optim_is_caught`（`torch.optim.SGD` を作ると `assert_free_threaded` が例外）

### 3.9 指標の schema

`metrics/evals.jsonl` の 1 行（M14）:

```json
{"t": 600.0, "frame": 18000,
 "m": {"err1": 0.0021, "err5": 0.031, "err10": 0.088, "err20": 0.21,
       "ratio_cv5": 0.41, "ratio_cv10": 0.38, "ratio_mean10": 0.09, "ratio_copy10": 0.12,
       "nonfinite": 0}}
```

- 物差し（`base_mean{h}`、`base_copy{h}`、`base_cv{h}`、`base_true{h}`）はモデルに依らないので、最初の評価の行と `result.json` にだけ書く
- `result.json`: `{"schema": 1, "status": "completed"|"diverged", "reason": null|"nonfinite"|"err>10x mean", "t_end", "frames", "final": {最後の評価}, "best": {...}, "baselines": {...}, "objective": {"log_ratio_cv_5_10": -0.93}, "metrics_sha256": {"evals": "...", "frames": "..."}, "peak": {"rss_mb", "gpu_mem_mb"}, "wall_s"}`
- `metrics_sha256` があるので、**監査の run**（§4.8）で、同じジョブを別の時刻・別のホストで回して指標が一致するかを hash だけで比べられる

---

## 4. フリートのランチャー

### 4.1 全体の形

```text
Mac（オーケストレーター、AI エージェント）
  fleet daemon（caffeinate の下、Bash の background）── 30 秒ごとの tick
    ├─ fleet.db（sqlite WAL、Mac のローカル）: jobs / attempts / events / hosts / batches
    ├─ ssh（ControlMaster で 4 本を張りっぱなし）→ std-0X のエージェント
    ├─ rsync pull → runs/（Mac の保管と index）
    └─ fleet opt run（別プロセス、optuna の環境）── fleet.db を読み書き
std-0X（ホスト）
  /data/geson/fleet/agent/agent.py（ホストの python3、stdlib）
    └─ supervisor（setsid で切り離す。1 run に 1 つ。core の flock を持つ）
         └─ docker exec mujou-exp-dev-1 taskset -c <cores> <venv>/bin/python -m exp.video.run_dynamics --job ...
```

- **押す方式**: ホストから Mac へも、ホストどうしも SSH できないので、Mac が押す。ホストには常駐のデーモンを置かない（ホストが再起動したら誰も起こさない常駐物は、壊れた状態を増やすだけ）。run 1 本ごとの supervisor だけが、その run が生きている間だけいる
- **真実の置き場**: 実行の真実はホストの run ディレクトリ（`job.json`、`status.json`、`exit/`）。`fleet.db` はそのキャッシュと、まだ起動していないジョブのキュー。`fleet rebuild-db` で、全ホストの run ディレクトリから実行の状態を作り直せる
- Mac が眠っても、daemon が落ちても、run は走り続ける。戻ったときの reconcile で追いつく

### 4.2 ホストのディレクトリと compose.yaml の変更

```text
/data/geson/fleet/                     # FLEET_DIR。コンテナにも同じパスでマウント
  agent/agent.py  agent/VERSION        # fleet bootstrap が置く。プロトコルの版を Mac と照合
  git/repo.git  git/mujou.git          # bare。Mac から snapshot を push する
  code/<snap12>/                       # snapshot ごとの worktree。準備が済んだら読み取り専用（chmod -R a-w）+ .ready
  venv/<envkey>/                       # uv の venv。.ready と env.json（uv pip freeze、python、torch、CUDA）
  uv-cache/                            # UV_CACHE_DIR
  data/<data_version>/                 # M15: 前処理したデータのローカルの複製（manifest で検証済み）
  runs/<batch>/<run_key>/
  locks/core-<n>.lock  locks/gpu-<i>.lock  locks/provision-<key>.lock
```

compose.yaml（1 回だけ変え、4 台でコンテナを作り直す。インフラの PR）:

```yaml
services:
  dev:
    restart: unless-stopped            # ホストの再起動の後にコンテナが戻るように
    volumes:
      - ${EGODEX_DIR:-./datasets/egodex}:/datasets/egodex:ro
      - ${WALKING_TOURS_DIR:-./datasets/walking_tours}:/datasets/walking_tours
      - ${FLEET_DIR:-./.fleet}:${FLEET_DIR:-/fleet}   # ホストでは .env に FLEET_DIR=/data/geson/fleet
```

（GPU は既にコンテナから見えている。Mac では `FLEET_DIR` を設定しないので、`./.fleet` を `.gitignore` に足す。）

### 4.3 フリートの定義（`fleet/hosts.toml`、Python 3.11 の `tomllib` で読む）

```toml
[defaults]
container = "mujou-exp-dev-1"
root = "/data/geson/fleet"
user = "vscode"
reserved_cores = [0, 1]          # OS、docker、エージェント、rsync、前処理の ffmpeg
ram_budget_gb = 52               # 61 GB のうち。ページキャッシュと余裕を残す
min_free_disk_gb = 100

[[host]]
name = "std-00"
ssh = "std-00"
gpus = [{index = 0, model = "4090", mem_gb = 22.0},
        {index = 1, model = "4060", mem_gb = 7.0}]
# std-01..03 も同じ
```

### 4.4 スロットの割り当て（`fleet/schedule.py`、純関数）

- 資源の単位: **core**（1 run に 1 core を `taskset` で専有。GPU の run は CUDA のドライバのスレッドのため 2 core）、**RAM**（GB）、**GPU のメモリ**（GB）。1 台あたり 30 core が使える
- ジョブの `resources` は preset ごとの見積もり。最初は保守的な既定（CPU の run: 1 core、2 GB。GPU の run: 2 core、3 GB、GPU 2 GB）で、**完了した run の `result.json` の `peak`（RSS と `torch.cuda.max_memory_allocated`）の最大 × 1.25 で見積もりを更新する**（`group_hash` を除いた preset 名ごと）。見積もりを超えた run は OOM で殺される前に supervisor が警告を events に出す
- GPU の種類: `gpu = "4090" | "4060" | "any" | null`。確認の batch は種類を固定する（§3.4）
- 置き方: 優先度 → 投入の古い順に、条件を満たすホストのうち空きの多いところへ（ホストの間に散らす。1 台が落ちたときの損を小さくする）。同じ `trial` の seed 違いは別のホストに散らす
- `after`（依存）と `same_host_as`（補助のジョブを親の run と同じホストに。ホストどうしでファイルを渡せないため）を守る
- **二重の割り当てを 2 か所で防ぐ**: Mac の scheduler は 1 つだけ（`fleet.db` の隣の flock）、ホストのエージェントは core と GPU の lock ファイルを `flock -n` で取れなければ起動を拒む（`busy`）
- 自分の測定で最初に決めること（WP1 の検証の一部）: GPU 1 枚に何本まで並べると 1 本の速さが落ち始めるか（M12 では 4090 に 5 本でほぼ落ちない）。M14 の小さいモデルは CPU の方が速い見込みなので、`s0_calibrate` の preset で CPU・4090・4060 の frames/s を測ってから、preset ごとの既定の `resources` を決める

### 4.5 コードの同期: rsync ではなく git の snapshot

**rsync で作業ツリーを送ると、run が「どのコードで走ったか」が名前を持たない**。送っている最中に Mac で編集すれば、半分だけ新しいコードで走る。長い run が走っている間に次のコードを送れば、走っている run の下でファイルが変わる。そこで:

1. `fleet snapshot`（Mac）:
   - 作業ツリーがきれいなら `snap = HEAD`
   - 汚れていれば、一時の index（`GIT_INDEX_FILE`）で `git add -A`（未追跡のファイルも、`.gitignore` に従って）→ `write-tree` → `commit-tree -p HEAD -m "fleet snapshot"`。ブランチも本物の index も動かさない。mujou も同じ手順で snapshot を作り、上の index の gitlink を `update-index --cacheinfo 160000,<mujou_snap>,mujou` で差し替える
   - `refs/fleet/snap/<sha>` を手元に作る（gc で消えないように）。各ホストの `git/repo.git` と `git/mujou.git` へ `git push`
   - `code.diff`（`base_head..snap` と mujou の差分）を作り、`job.json` と一緒に送る
   - 汚れた snapshot は既定で拒み、`--allow-dirty` のときだけ作る。確認の batch（`phase=confirm`）は**きれいな commit からしか投入できない**（README の数字はすべて commit から再現できる）
2. `provision`（ホストのジョブ、1 snapshot 1 回、flock で直列化）:
   - `git -C git/repo.git worktree add --detach code/<snap12> <snap>`、`submodule.mujou.url` をローカルの `git/mujou.git` に向けて `submodule update --init`
   - `envkey = sha256(uv.lock, pyproject.toml, mujou の tree の sha, python の版)[:12]`。`venv/<envkey>/.ready` が無ければ、コンテナの中で `UV_PROJECT_ENVIRONMENT=venv/<envkey> uv sync --frozen --no-editable`。**`--no-editable`** で mujou を wheel として入れるので、venv は特定の worktree に依存しない（worktree を消しても venv は壊れない）。`exp` は `package = false` で入らず、`cwd = code/<snap12>` から import される
   - 検証: venv の python で `import mujou; mujou.__file__` が venv の中、`sys._is_gil_enabled()` が False、`torch.cuda.is_available()`
   - `chmod -R a-w code/<snap12>`、`.ready` を書く
3. run の起動時に `verify_code`（§3.4）でもう一度確かめる
4. 掃除: `fleet gc` は、キューにも実行中にも参照されず 7 日を過ぎた `code/` を消す。venv は参照する snapshot が無くなったら消す。snapshot の commit そのものは、Mac の `refs/fleet/snap/*` と 4 台の bare repo、それに NAS の `git bundle`（`/nas/geson/mujou-exp/snapshots/<sha>.bundle`）に残す

こうすると、違うコードの run が同じホストで同時に走れる（開発を続けながら長い run を回せる）。

### 4.6 起動・停止・監視（ssh + docker exec）

**起動**: `ssh std-0X python3 /data/geson/fleet/agent/agent.py start`（stdin に job.json）。エージェントは:

1. `code/<snap12>/.ready` と `venv/<envkey>/.ready` を確かめる（無ければ `not_ready` を返し、Mac が provision を先に流す）
2. run ディレクトリを作る（既にあれば §3.3 の条件で再開、`attempt = n + 1`）
3. core と GPU の lock を `flock -n` で取る（取れなければ `busy`）
4. supervisor を二重 fork + `setsid` で切り離して起動し、すぐに `{"started": true, "attempt": n}` を返す（ssh が切れても run は残る）

**supervisor**（同じ `agent.py` の `supervise` サブコマンド）: lock を持ったまま

```text
docker exec -u vscode -w /data/geson/fleet/code/<snap12> \
  -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
  -e PYTHONUNBUFFERED=1 -e PYTHONFAULTHANDLER=1 -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  -e CUDA_VISIBLE_DEVICES=<i or ""> -e FLEET_HOST=std-0X -e FLEET_ATTEMPT=<n> \
  mujou-exp-dev-1 \
  setsid taskset -c <cores> /data/geson/fleet/venv/<envkey>/bin/python \
    -m exp.video.run_dynamics --job <run_dir>/job.json --attempt <n>
```

を実行し、標準出力と標準エラーを `logs/attempt-<n>.{out,err}` に流す。`docker exec` が返ったら `exit/attempt-<n>.json`（`returncode`、`signal`、`started`、`ended`、`stop_reason`）を原子的に書いて lock を放す。supervisor 自身も `supervisor.json` に 30 秒ごとに心拍を書く。

**停止**: `agent.py stop <run_dir> --reason pruned|drain|move|user [--now] [--grace 120]`

- `--now` でなければ `stop.request` を書く。run は次のフレームの境目で再開用の checkpoint を書き、終了コード 75 で止まる
- 猶予を過ぎたら、`status.json` の `pgid` に `docker exec mujou-exp-dev-1 kill -TERM -<pgid>`、さらに 10 秒後に `-KILL`。**`docker exec` のクライアントを殺しても中のプロセスは死なない**（docker の既知の挙動）ので、必ずコンテナの中の process group を殺す
- 枝刈り（pruned）は `--now`（再開しないので checkpoint は要らない）

**監視**: `agent.py status --json`（tick ごとに 1 ホスト 1 回の ssh）が返すもの:

- ホスト: 時刻、`nvidia-smi` の GPU ごとのメモリと使用率、空きメモリ、load、`/data` の空き、コンテナの `State.StartedAt`（変わっていればコンテナが再起動した）、エージェントの版
- run ごと（`runs/` の下で、まだ archived でないもの）: `status.json`、supervisor が生きているか（pid と lock を取れるかで判定）、`exit/` の最新、心拍の年齢、`frame` が最後に進んでからの時間、`evals.jsonl` の最後の行

### 4.7 失敗の扱い（`fleet/reconcile.py`、表で試験する純関数）

ジョブの状態: `queued → placed → starting → running → {completed, diverged, failed, pruned, stopped, lost, stalled} → (retry → queued) | archived`。

| 観測 | 判定 | 動作 |
| ---- | ---- | ---- |
| `exit` が 0 か 3、`result.json` と `manifest.json` が揃う | completed / diverged | 同期 → manifest の検証 → archived。検証に失敗したら再同期、3 回で `corrupt` として人に知らせる |
| `exit` が 0 なのに `result.json` か `manifest.json` が無い | failed（契約違反） | 再試行しない。events にエラーとして出す（「黙った成功」を許さない） |
| `exit` が 4 | infra | そのホストを、その種類（cpu/4090/4060）について cordon。別のホストで再試行（`max_infra` まで） |
| `exit` が 75 | stopped | 理由が pruned なら終わり。drain・move なら再開用の checkpoint から再投入 |
| `exit` が 1 | failed | 再試行しない。エラーの印（例外の型 + stderr の最後の行 + プログラム）を数え、batch の中で 3 回なら **batch を保留**（壊れたコードで 4 台を燃やさない） |
| SIGKILL、停止の要求なし | oom の見込み | `ram_gb` を 1.5 倍にして 1 回だけ再試行 |
| supervisor が死んでいて `exit` が無い、またはコンテナの `StartedAt` が変わった | lost | 再開用の checkpoint から再投入（同じ GPU の種類に限る。§3.4） |
| supervisor は生きているが `frame` が `stall_after`（その preset の 1 評価区間の予想の壁時計の 5 倍、最低 10 分）進まない | stalled | まず `kill -USR1`（faulthandler が全スレッドの traceback を stderr に出す。OpenMP の取り合いの診断になる）→ 停止 → 1 回だけ再試行、2 回目は failed |
| ホストに ssh が 3 tick 続けて届かない | host down | そのホストの run を `unknown` にする。**すぐには再投入しない**（ホストが戻って run が生きていれば二重になる）。2 時間戻らなければ別のホストで再開し、その試行を正（authoritative）として DB に記録する。戻ったホストで正でない試行を見つけたら殺し（fencing）、その出力は同期しない |
| `/data` の空きが `min_free_disk_gb` 未満 | disk | そのホストに新しい run を置かない。`fleet gc` を促す |

**はじめに燃やさない**: 新しい code snapshot で 20 本を超える batch を投入すると、まず **canary**（その batch の最初の設定を仮想時間 60 秒だけ、各 GPU の種類で 1 本ずつ）を回す。canary が completed にならなければ batch は `held` のまま。

すべての遷移は `events` 表に 1 行（`id`、時刻、run_key、前後の状態、理由、エラーの印）。AI のオーケストレーターは `fleet events --since <cursor>` で「前に見てから何が起きたか」だけを読む。

### 4.8 出力の置き場と同期

- 実行中: ホストのローカル `/data/geson/fleet/runs/`（btrfs）。NAS には書かない（§0）
- **Mac への pull**（`fleet sync`、daemon が tick ごと）:
  - 実行中の run: `status.json` と `metrics/evals.jsonl` だけ（小さい。optuna の中間値に使う）
  - 終わった run: `resume/` を除く全体を `rsync -a --partial`（Homebrew の rsync）で `runs/<batch>/<run_key>/` へ。その後 `manifest.json` の sha256 を Mac で検証。通ったら archived
- **NAS への保管**: archived になった run を、エージェントがホストから `/nas/geson/mujou-exp/runs/` へ `rsync -rt --no-perms`（CIFS なので所有者と権限は送らない）し、NAS の側でも manifest を検証する。checkpoint の全部はここに残す。Mac には `ckpt/` も含めて置くが、容量が問題になったら `fleet sync --no-ckpt` にして NAS を正にする
- ホストの run ディレクトリは、Mac と NAS の両方で検証が済んで 3 日経ったら `fleet gc` で消す
- `fleet.db` と optuna の DB は、tick ごとではなく 1 時間ごとに `sqlite3 .backup` で `runs/_backup/` に複製し、1 日 1 回ホスト経由で NAS にも置く
- **監査の run**: 完了した run の 2% を無作為に選び、同じ `job.json`（同じ snapshot、同じ GPU の種類）で別のホストに `phase=audit` として回す。`metrics_sha256` が一致しなければ events に `nondeterministic` を出す。再現性が壊れたことに、README を書く前に気づける

### 4.9 オーケストレーターの CLI（`python -m fleet ...`、`just fleet ...`）

既定の出力は JSON（1 行 1 オブジェクト）、`--table` で人が読む表。終了コードは 0 = 成功、1 = 誤った入力、2 = 一部が失敗、3 = フリートの問題。

```text
fleet doctor                        ssh、コンテナ、GPU、ディスク、エージェントの版、時計のずれ、rsync の版を点検
fleet bootstrap                     ホストにディレクトリとエージェントを置く（冪等）
fleet snapshot [--allow-dirty]      snap の sha を出し、ホストへ push し、provision を流す
fleet submit PRESET [OVERRIDES..] [--sweep k=a,b ..] [--seeds 0,1,2] --batch NAME
             [--gpu 4090|4060|any | --cpu] [--priority N] [--after RUN] [--tags k=v] [--dry-run]
                                    リモートで解決し、run_key ごとに new / duplicate を出す
fleet status [--batch B] [--host H]   状態ごとの数、実行中の進み具合と ETA、問題
fleet events [--since CURSOR]       状態の遷移と失敗（カーソルを返す）
fleet logs RUN [--err] [--tail N]   ホストから直接読む
fleet stop RUN | --batch B | --trial STUDY/N  [--now] [--reason R]
fleet retry RUN [--fresh]           再開用の checkpoint から、または最初から
fleet hold B / release B / cordon H [--kind cpu|4090|4060] / uncordon H / drain H
fleet daemon [--once] [--interval 30]
fleet sync [--live] / fleet gc [--dry-run] / fleet rebuild-db
fleet reproduce RUN                 監査の run を投入し、終わったら hash を比べる
fleet index / fleet query SQL / fleet table ... / fleet export ...      （§6）
fleet opt create SPEC / run SPEC / status STUDY / confirm STUDY ...     （§5、別の環境）
```

`justfile` には `fleet *args` を足す: `uv run --no-project --python 3.13 python -m fleet {{args}}`。`fleet opt` は `uv run --project fleet/opt python -m fleet.opt.coordinator ...`。

---

## 5. optuna

### 5.1 置き場所

- **coordinator は Mac の別プロセス**（`fleet opt run <spec>`、`fleet/opt/` の uv 環境、Python 3.13、optuna 5.0.0）。学習のプロセスは optuna を知らない。run は §3 の契約どおりにファイルを書くだけ
- **storage は Mac のローカルの sqlite**（`fleet/state/optuna.sqlite3`、RDBStorage）。NAS の CIFS は `nobrl` なので sqlite を置かない。JournalFileStorage もファイルのロックに頼るので同じ理由で置かない。書くのは coordinator だけ。optuna-dashboard は同じファイルを読むだけ
- coordinator と daemon は `fleet.db` を通して話す（coordinator は `fleet` のジョブ投入の関数を import して使う。stdlib なので optuna の環境からも import できる）

### 5.2 study の定義（`exp/video/studies/*.toml`、commit する）

```toml
[study]
name = "m14-s1-bp-explore-v1"     # code snapshot を変えたら v を上げる
preset = "m14/s1_explore"
direction = "minimize"
seeds = [100, 101]                 # 探索の seed。確認の seed（0/1/2）とは重ねない
parallel = 40                      # 同時に走らせる試行（× seeds 本の run）
max_trials = 240
resources = {gpu = "any"}
objective = {metric = "log_ratio_cv", horizons = [5, 10], seeds = "mean", diverged = 1.0}

[sampler]
kind = "tpe"
seed = 0
constant_liar = true
n_startup_trials = 24
multivariate = true

[pruner]
kind = "asha"                      # SuccessiveHalvingPruner。resource = 仮想時間の秒
min_resource = 300                 # warmup（300 秒）が終わるまでは枝刈りしない
reduction_factor = 3

[space]
"optim.lr"         = {float = [0.003, 0.3], log = true}
"optim.kind"       = {categorical = ["sgd", "adamw", "graft_adam"]}
"optim.tau_m"      = {float = [0.03, 3.0], log = true, when = "optim.kind != 'sgd'"}
"dynamics.trace"   = {categorical = [true, false]}
"dynamics.residual" = {categorical = [true, false]}
"dynamics.open_loop" = {categorical = [0.0, 0.25, 0.5]}

[[enqueue]]                        # 既知の良い点から始める（M13 の最良）
"optim.lr" = 0.03
"optim.kind" = "sgd"
"dynamics.trace" = true
```

### 5.3 ask/tell のループ

1. 走っている試行が `parallel` に満たなければ `trial = study.ask()`、`space.py` が define-by-run で `trial.suggest_*` を呼んで override の列にする（`when` の条件つき）。各 seed について `fleet submit`（`study`、`trial` のタグつき）。override の解決が失敗したら `tell(FAIL)`、その理由を `trial.user_attrs`
2. tick ごとに、同期された `evals.jsonl` を読む。評価の時刻 `t` について、**その試行の全 seed が `t` に達したときだけ**、seed の平均を `trial.report(value, step=int(t))`。一部の seed だけの平均は早い seed に偏るので使わない
3. `trial.should_prune()` → `fleet stop --trial <study>/<n> --now --reason pruned` → `tell(PRUNED)`
4. 全 seed が終わったら目的値を計算して `tell(value)`。**目的値は run が `result.json` の `objective` に書いた値をそのまま使う**（README の物差しと同じ関数 `Program.objective` で計算される。coordinator の側で式を別に持たない）
5. 失敗の扱い: seed の run が lost・stalled・infra で再試行中なら待つ。failed（例外）なら `tell(FAIL)`。diverged は `objective.diverged` の値（M14 では `log_ratio_cv` = 1.0 に相当する悪い値、つまり等速の外挿の e 倍）として tell し、TPE に「悪い領域」として学ばせる
6. `trial.user_attrs`: `run_keys`、`code_snap`、`group_hash`、seed ごとの目的値。`study.user_attrs`: spec ファイルの sha256、`code_snap`、preset、optuna の版
7. **study は 1 つの code snapshot に縛る**。coordinator は、study の `code_snap` と違う snapshot で試行を投入することを拒む（コードを変えたら `-v2` の新しい study にする）。これで 1 つの study の中で、違うコードの結果を比べてしまうことが無い
8. coordinator が落ちても、run は走り続ける。再起動すると、`RUNNING` の試行の run を `fleet.db` から探して報告を再開する（ask/tell と run の対応はタグで復元できる）

非同期の TPE の探索の道筋そのものは再現しない（完了の順に依る）。再現するのは各試行の run で、その点は §3 が保証する。README には study の spec、snapshot、試行の数を書く。

### 5.4 BP で探索してから PC-ALM で確認する

```text
fleet opt run exp/video/studies/m14_s1_bp_explore.toml       # learner=bp、600 秒、seed 100/101、枝刈りあり
fleet opt confirm m14-s1-bp-explore-v1 --top 3 \
    --preset m14/s1_confirm --learners pc,bp --seeds 0,1,2 --gpu 4090 --batch m14-s1-confirm-v1
```

- `confirm` は optuna の study ではなく、普通の batch（格子）を投入する。上位 k 個の試行のパラメータ × learner {pc, bp} × seed {0, 1, 2} × 30 分、枝刈りなし。各 run に `confirm_of = <study>/<trial>` のタグ
- 上位の選び方は探索の seed の目的値の平均。**README に書く数字は確認の seed（0/1/2）の run だけから取る**。探索で選んだ同じ seed の値を報告すると、選んだこと自体による楽観（winner's curse）が入るため
- PC-ALM のトレースは 30 分で 5〜7 時間かかる見込み。確認の batch は優先度を上げ、最初に置く（長い run を後に回すと、batch の終わりが最も遅い run で決まる）
- `confirm` は clean な snapshot を要求する（§4.5）

---

## 6. 追跡と集計

### 6.1 決めたこと: trackio は入れない、optuna-dashboard だけ入れる

| 候補 | 判断 |
| ---- | ---- |
| trackio（学習のプロセスの中） | 入らない（orjson が 3.14t でビルドできない）。入ったとしても、学習のプロセスに gradio 系の依存を入れることは GIL と停止の危険を増やす |
| trackio（別の環境で、ファイルから写す） | 2 つ目の「真実の置き場」になる。DB を NAS に置けば `nobrl` の問題に当たり、Mac に置くなら index と同じものを 2 つ持つことになる。**今は入れない**。曲線を対話的に見たくなったら、`fleet/index.py` から一方向に写す mirror として後で足せる（run の契約は変えずに済む） |
| optuna-dashboard | 入れる（`fleet/opt/` の環境だけ）。optuna の DB を読むだけで、探索の様子（重要度、平行座標、中間値の曲線）が手間なく見られる |
| index.sqlite3 + 表と JSON の書き出し | 入れる。README の表と report.html の正 |

### 6.2 index（`fleet/index.py`）

`runs/index.sqlite3` を、**manifest の検証が済んだ（archived の）run だけ**から作る。作り直しは冪等（消してから全部読み直す。数百本で数秒）。

```text
runs(run_key PK, batch, study, trial, program, milestone, phase, cfg_hash, group_hash, seed,
     code_snap, code_dirty, mujou_snap, learner, device_kind, host, status, reason,
     frames, t_end, wall_s, attempts, created, tags_json, config_json)
config_kv(run_key, key, value)        # 解決済みの設定を平らにしたもの（"optim.lr" = "0.03"）
evals(run_key, t, metric, value)
results(run_key, metric, value)       # final / objective / baselines
```

`fleet query "SELECT ..."` で直接読める。

### 6.3 README の表（`fleet/table.py`）

```text
fleet table --batch m14-s1-confirm-v1 --rows dynamics.size,dynamics.trace --cols learner \
    --metric final.ratio_cv10 --seeds 0,1,2 --format md
```

- セルは seed の値を ` / ` で区切る（README の規約）。発散は「発散」、欠けた seed は「欠」と書き、**欠けた seed を黙って平均から外さない**
- 次のときは止める（`--allow-mixed` で明示したときだけ続ける）: 1 つのセルの中でコードの版・mujou の版・デバイスの種類が混ざる、同じ (group, seed) に 2 本以上の archived な run がある、batch の中に archived でない run が残っている（「終わっていない batch を表にしない」）
- 表の下に脚注を自動で付ける: snapshot と mujou の SHA、デバイスの種類、run の数、この表を作ったコマンド。`results/mNN.md` にそのまま貼れる
- `fleet export --batch B --metrics ratio_cv5,ratio_cv10 --curves` → report.html（skill `experiment-report`）に埋め込む JSON。PNG（`viz/`、`decode/`）のパスも入れる

---

## 7. M14 のコード（シグネチャ）

### 7.1 データ（`exp/video/balls.py`、`obs.py`）

`Video` は純 Python で 1 ストリーム 1 フレーム約 1 ms。B=128 だと 30 分で物理だけで約 2 時間の CPU になり、M14 の run の大半が物理に消える。そこでベクトル化する。

```python
class BallStreams:
    """B streams of balls, vectorized over streams and balls in float64.

    Stream i draws its balls as Video(seed=seed + STREAM_SEED_STRIDE * i) does, and
    then moves exactly as it does: the states match the reference bit for bit.
    """
    def __init__(self, n: int, *, seed: int, radii: Sequence[float], physics: Physics,
                 frame: float, substeps: int = 8, render: int | None = None) -> None
    def next(self) -> Frame          # state: (n, 4 * balls); image: (n, 1, S, S) if render else (n, 0)
    def state_dict(self) -> dict[str, torch.Tensor]          # 再開用
    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None
```

- `render=None`（段 0〜2）なら画像を作らない
- 引き直しなしの EASY だけを扱う（`episode=None`）。時間で変わる物理（`gamma_amp` など）は `Physics.gamma(t)` を Python の float で計算して全ストリームに使う（今の `Video` と同じ）

```python
# exp/video/obs.py
@dataclass(frozen=True)
class ObsLayout:
    kind: ObsKind
    balls: int
    @property
    def dim(self) -> int                               # 4, 2, 6, size*size
    def select(self, state: torch.Tensor) -> torch.Tensor   # (..., 4 * balls) -> (..., dim)
    def to_state(self, obs: torch.Tensor) -> torch.Tensor   # 位置の列を戻し速度に 0 を入れる（描画用）

@dataclass(frozen=True)
class Standardizer:
    mean: torch.Tensor   # (dim,)
    std: torch.Tensor    # (dim,)
    @classmethod
    def of(cls, clips: torch.Tensor) -> Self           # clips: (P, C, L, dim)。全クリップ・全フレームの平均と標準偏差
    def __call__(self, x: torch.Tensor) -> torch.Tensor
    def inverse(self, x: torch.Tensor) -> torch.Tensor

def obs_clips(cfg: DynamicsRun) -> tuple[torch.Tensor, torch.Tensor, Standardizer]
    # (clips_obs (P, C, L, dim) standardized, clips_state (P, C, L, 4 * balls), standardizer)
```

段 3（画像）は `dim = size * size` に平らにした画素。標準化は画素ごとではなく、全画素で 1 つの平均と標準偏差（画素ごとだと背景の画素の std が 0 に近く、割ると壊れる）。

### 7.2 物差し（`exp/video/baselines.py`）

すべて標準化した観測の空間で、1 次元あたりの二乗誤差。`observed = 16`、horizon `h` はフレーム `observed - 1 + h` の予測（今の `evaluate.run_states` と同じ数え方）。

```python
def mean_errors(obs: torch.Tensor, observed: int, horizons: Sequence[int]) -> dict[int, float]
def copy_errors(obs: torch.Tensor, observed: int, horizons: Sequence[int]) -> dict[int, float]
def constant_velocity(state: torch.Tensor, layout: ObsLayout, observed: int, steps: int) -> torch.Tensor
    # 最後の 2 フレームの位置の差を速度として、重力も壁も入れずに進めた状態 (n, steps, 4 * balls)
    # 段 0 の速度の列は差分の速度（観測した速度ではない。MILESTONES の定義どおり）
def constant_velocity_errors(clips_state, layout, standardizer, observed, horizons) -> dict[int, float]
    # 段 3 は外挿した状態を render して画素で比べる
def true_physics_errors(clips_state, cfg, layout, standardizer, observed, horizons) -> dict[int, float]
    # 観測した最後の真の状態から advance で進める。0 になることの確認だけに使う
def baselines(...) -> dict[str, float]     # base_mean{h}, base_copy{h}, base_cv{h}, base_true{h}
```

### 7.3 ノードとネットワーク（`exp/video/nodes.py`、`dynamics.py`）

**credit を構造で止める**。上流の活動を受けるポートを「受けるだけ」にすれば、送り返す経路が型の上で存在しない。`target_weight = 0` のような重みで止めると、設定を 1 つ間違えただけで credit が流れる（M13 の `stop_target=False` だと `target_weight` が無視されて 1 になる罠がそれだった）。

```python
class Source(Node):
    """Shows the host's samples downstream and takes no credit back.

    Ports: prev = Port(recv=Sample); next = Port(send=Activity) (send only: a
    consumer cannot send credit to it, so none can reach the observation).
    """
    def __init__(self, *, x: TensorSpec, period: float, batch: Size = (), device: str = "cpu") -> None
    act_spec: DictSpec[Activity]

class Dynamics(Node):      # s_k = minGRU(z_{k-1}, s_{k-1})
    def __init__(self, module: MinGRU, *, act_spec: DictSpec[Activity], tau: pc.TimeConstants,
                 optimizer: Optimizer, learn_interval: float | None, open_loop: float = 0.0,
                 open_cycle: int = 20, residual: bool = False, trace: bool = False,
                 device: str = "cpu") -> None
    # ports: z = Port(recv=act_spec)（受けるだけ）、next = MultiPort(send=s_spec, recv=s)、
    #        z_hat = Port(recv=prediction_spec)（open_loop > 0）、base = Port(send=act_spec)（residual）

class DynamicsLayer(Node): ...     # 今と同じ（同じフレームの入力なので g_in を前段へ返す）

class Compare(Node):
    """The prediction error 1/D ||W_o s_k (+ base) - o_k||^2 on its last axis.

    Ports: s = Port(send=s, recv=s_spec)（credit を返す）、z = Port(recv=z_spec)（受けるだけ）、
    base = Port(recv=act_spec)、next = Port(send=Prediction)、error = Port(send=TensorSpec(()))
    """
    def __init__(self, readout: nn.Linear, *, s_spec, z_spec, tau, optimizer,
                 learn_interval, residual: bool = False, device: str = "cpu") -> None

def compare_energy(readout: nn.Module, s: Tensor, target: Tensor, base: Tensor | None) -> Tensor
    # (*batch,)。kappa と (1 - kappa) の結合は外す（SIGReg と無関係にする）
```

```python
# exp/video/dynamics.py
@dataclass
class DynamicsModules:
    step: DynamicsStack          # Dynamics + DynamicsLayer * (nodes - 1) を 1 つにしたもの
    readout: nn.Linear           # d_s -> d_obs（residual なら 0 で初期化）

def build_modules(d_in: int, d_out: int, cfg: DynamicsCfg, *, seed: int) -> DynamicsModules
    # torch.manual_seed(seed) の後に決まった順で作る。PC 版と BP 版はこれだけを呼ぶ（初期値の一致）

def build_dynamics(net: mujou.Network, upstream: Port[Any, Activity], act_spec: DictSpec[Activity],
                   mods: DynamicsModules, cfg: DynamicsCfg, *, tau, optimizer: Callable[[str], Optimizer],
                   learn_interval, device) -> Compare
    # Dynamics・DynamicsLayer・Compare を net の属性に置いて繋ぎ、Compare を返す。
    # upstream は M14 では Source.next、M16 では JEPA ノードの z ポート（どちらも送るだけ）

class DynamicsNet(mujou.Network):            # PC-ALM。Source -> build_dynamics
    def __init__(self, cfg: DynamicsRun, d_obs: int, device: str) -> None
    prev: Port; next: Port; source: Source; compare: Compare; tau: pc.TimeConstants

class DynamicsBP(Node):                      # BP の基準。1 ノードで動力学と readout を autograd で
    def __init__(self, mods: DynamicsModules, *, act_spec, tau, optimizer, learn_interval,
                 open_loop: float = 0.0, open_cycle: int = 20, residual: bool = False,
                 trace: bool = False, device: str = "cpu") -> None
    # 今の WorldBP からエンコーダ・SIGReg・再構成を除いたもの。z_{k-1}, s_{k-1} は detach（PC と同じメモリ）

class DynamicsBPNet(mujou.Network):          # Source -> DynamicsBP
    def __init__(self, cfg: DynamicsRun, d_obs: int, device: str) -> None

def make_optimizer(c: OptimCfg, frame: float) -> Optimizer   # Warmup(SGD | AdamW | Grafted(AdamDirection))
def snapshot(net: DynamicsNet | DynamicsBPNet) -> DynamicsModules   # step の合間に deepcopy
```

### 7.4 評価とプログラム（`rollout.py`、`run_dynamics.py`）

```python
# exp/video/rollout.py
def run_states(obs: Tensor, mods: DynamicsModules, observed: int, steps: int, residual: bool
               ) -> tuple[Tensor, Tensor]     # (states (n, observed+steps, d_state), preds (n, steps, d_obs))
def rollout_errors(obs, mods, observed, horizons, residual) -> dict[int, float]
def evaluate(mods: DynamicsModules, clips_obs: Tensor, base: dict[str, float], cfg: DynamicsRun
             ) -> dict[str, float]            # err{h}, ratio_cv{h}, ratio_mean{h}, ratio_copy{h}, nonfinite
def objective(m: dict[str, float], horizons: Sequence[int] = (5, 10)) -> float
    # mean_h log(err_h / base_cv_h)。0 で等速の外挿と同じ、log 0.5 = -0.69 が完了の目安
```

```python
# exp/video/run_dynamics.py
class DynamicsProgram(Program[DynamicsRun]): ...   # build / frame / evaluate / modules / host_state / objective
def main(argv: list[str] | None = None) -> int     # exp.lib.program.main(DynamicsProgram(), argv)
```

`learner = none` のときはネットワークを組まずに物差しだけを測り、`result.json` に書いて終わる（MILESTONES の手順 1）。

### 7.5 オフラインの上限と見た目（補助のプログラム）

```python
# exp/video/offline_dynamics.py（torch.optim.Adam。補助なので GIL の検査はしない）
def window_starts(resets: Tensor, window: int) -> Tensor        # 今の offline.py と同じ
def stream_windows(cfg: DynamicsRun, frames: int) -> Tensor     # (T, B, dim)、BallStreams から
def train(cfg: DynamicsRun, off: OfflineCfg, out: RunDir) -> None
    # 窓 32、観測 16、Adam、開ループの BPTT。評価は rollout.evaluate と同じ関数で、evals.jsonl に
    # OfflineCfg(steps, lr, batch, window, observed, local: bool) は DynamicsRun の offline フィールド
```

`local=True`（入力と帰還の勾配を 1 フレームで切る）と `False` の差が「時間の局所性」の上限の差になる（M13 と同じ切り分け）。

```python
# exp/video/viz_dynamics.py（run の完了の後、同じホストで）
def frames_png(path: Path, true_state: Tensor, pred_state: Tensor, radii, size: int, columns: Sequence[int]) -> None
    # 行: 真のフレーム / 予測の状態を balls.render で描いたもの。列: 1, 2, 5, 10, 20 フレーム後
def trajectory_png(path: Path, true_xy: Tensor, pred_xy: Tensor, scale: int = 8) -> None
    # 箱 [0,1]^2 に真の軌跡（緑）と予測（赤）を点の列で描く。exp.lib.png のラスタで、matplotlib は使わない
def main(run_dir: Path) -> int     # 各 ckpt/<t>.pt について viz/<t>-frames.png と viz/<t>-traj.png
```

### 7.6 オラクルの試験

| 試験 | 理論から言えること |
| ---- | ------------------ |
| `test_balls.py::TestBallStreams::test_matches_the_reference_video_bit_for_bit` | ストリーム i は `Video(seed=seed+STRIDE*i)` と同じ状態を 2000 フレーム、`torch.equal` で一致（float64 の同じ順の演算） |
| `test_balls.py::TestBallStreams::test_state_round_trips` | `state_dict` から戻したストリームが、止めずに進めたものと一致 |
| `test_obs.py::test_stage0_inverse_is_the_state` | 段 0 の観測を標準化から戻すと `Frame.state` に一致 |
| `test_obs.py::test_stage1_and_2_are_the_position_columns` | 段 1・2 は位置の列だけ（`state.reshape(..., balls, 4)[..., :2]`） |
| `test_obs.py::test_standardized_clips_have_zero_mean_and_unit_std` | 標準化した評価クリップの各次元の平均 0、標準偏差 1 |
| `test_baselines.py::test_constant_velocity_is_exact_without_gravity_away_from_walls` | `Physics(gamma0=0)`、壁から遠い初期値で、等速の外挿の誤差が 0（半陰的 Euler で重力 0 なら位置は厳密に線形） |
| `test_baselines.py::test_mean_error_is_the_variance` | 標準化した空間では、平均の物差しの誤差が 1（全クリップ・全フレームで平均を取った場合） |
| `test_baselines.py::test_true_physics_has_zero_error` | 真の物理の誤差は 0 |
| `test_nodes.py::TestSource::test_a_consumer_cannot_send_credit` | `Source.next` に `send=` を宣言したポートを connect すると `TypeError`（構造の上で credit が戻らない） |
| `test_nodes.py::TestCompare::test_gradients_match_finite_differences` | `compare_energy` の `s` 側の勾配が中心差分と一致（float64） |
| `test_nodes.py::TestDynamics::test_matches_the_sequential_mingru` | 1 フレームずつの `Dynamics` の `mu` が、逐次に回した minGRU と一致 |
| `test_nodes.py::test_no_node_leaves_a_grad` | どのノードも `p.grad` を残さず、重みは動く（局所性） |
| `test_dynamics.py::test_pc_and_bp_start_from_the_same_weights` | 同じ seed で `DynamicsNet` と `DynamicsBPNet` のモジュールの重みが一致 |
| `test_dynamics.py::test_lam_is_the_credit_of_bp_at_the_fixed_point[nodes=1,3][batch=(),(4,)]` | 1 フレームを長く保持し `learn_interval=inf` にすると、各ノードの `lam` が BP の `-dL/dh` に一致（`rtol=1e-3`） |
| `test_dynamics.py::test_trace_gradient_is_bptt_with_inputs_fixed[pc,bp]` | 重みを止めて K フレームの後、トレースの勾配が入力を固定した BPTT に一致 |
| `test_dynamics.py::test_one_frame_of_bp_is_autograd` | `DynamicsBP` の 1 回の学習の勾配が、同じ損失の autograd と一致 |
| `test_rollout.py::test_identity_model_gives_the_copy_errors` | readout が恒等で residual のとき、rollout の誤差は写しの誤差 |
| `test_run_dynamics.py::test_tiny_job_writes_the_contract` | 小さなジョブ（B=2、10 秒）が §3.2 の全ファイルを書き、schema が合い、manifest の sha256 が合う |
| `test_run_dynamics.py::test_same_job_same_metrics_bytes` | 同じ `job.json` を 2 回回すと `evals.jsonl`・`frames.jsonl` がバイト単位で一致 |
| `test_run_dynamics.py::test_resume_equals_uninterrupted[pc,bp][trace]` | §3.6 |
| `test_run_dynamics.py::test_ctl_does_not_change_the_metrics` | `resume_interval` と `frame_log_every` 以外の `ctl` を変えても指標が一致（`cfg_hash` から外してよい根拠） |
| `test_run_dynamics.py::test_divergence_ends_with_status_diverged` | 大きすぎる学習率で、終了コード 3、`result.status == "diverged"` |
| `test_run_dynamics.py::test_stop_request_exits_75_with_a_resume_point` | `stop.request` で 75、再開用の checkpoint がある |
| `test_offline_dynamics.py::test_windows_do_not_cross_resets` | 今の `window_starts` の試験 |
| `test_offline_dynamics.py::test_offline_eval_is_the_online_eval` | 同じ重みなら、オフラインの評価の値が `rollout.evaluate` の値と一致（物差しが 1 つであることの確認） |
| `test_viz_dynamics.py::test_png_decodes_to_the_drawn_pixels` | 書いた PNG を手で復号すると、描いた画素（今の `test_decode.py` の形） |

`fleet` の試験（ssh も docker も使わない）: `LocalTransport` と、`docker exec` の代わりに引数を受けてそのまま子プロセスを起動する偽の `docker` スクリプトを PATH に置く。

| 試験 | 言えること |
| ---- | ---------- |
| `test_schedule.py::test_never_books_a_core_twice` ほか | 置き方が core・RAM・GPU のメモリの予算を超えない、同じ入力で同じ置き方（決定的）、seed 違いを散らす |
| `test_reconcile.py`（表で 20 通り） | §4.7 の表どおりの遷移 |
| `test_agent.py::test_exit_is_recorded_when_the_child_is_killed` | 子を SIGKILL しても `exit/attempt-1.json` に signal 9 が残る |
| `test_agent.py::test_busy_core_is_refused` | lock を他が持っていれば `busy` |
| `test_agent.py::test_stop_kills_the_process_group` | 猶予の後、孫まで死ぬ |
| `test_snapshot.py::test_dirty_tree_is_captured_with_untracked_files` | 汚れたツリーの snapshot が、未追跡のファイルと mujou の gitlink を含み、本物の index と HEAD を動かさない |
| `test_sync.py::test_corrupted_file_fails_the_manifest` | 1 バイト変えると archived にならない |
| `test_table.py::test_missing_seed_is_marked_not_averaged`、`test_mixed_code_is_refused` | §6.3 |

---

## 8. M15 と M16 の部品（モジュールの単位）

### 8.1 M15: JEPA

- **ノードの分け方**: エンコーダ（ViT）、予測器、SIGReg の損失、（使うなら）EMA のターゲットを、**1 つのノード `JEPA` の中に閉じ、ノードの中の BP**（`# mujou: local-autograd`、optimizer は `pc.optim.AdamW` か SGD、`torch.optim` は使わない）で学ぶ。理由: ViT を PC-ALM の多数のノードに割ると、1 周期 1.5〜2 ms のカーネル起動が律速になり（M12）、credit が 1 フレームの中で回り切らない。EMA のターゲットを別ノードにすると重みのコピーを 2 か所に持ち、状態の所有の不変条件（1 つのテンソルの書き手は 1 ノード）と食い違う。ノードの中の BP は mujou の原則の上で許されている（「1 ノードの中に閉じていれば BP も 1 つの更新手法」）
- **ポート**: `prev = Port(recv=Activity)`（`Source` から 1 フレーム）、`z = Port(send=z_act_spec)`（**送るだけ**。M16 で動力学がここに繋がる）、`loss = Port(send=TensorSpec(()))`（監視）
- **N フレームのまとまり**（`exp/video/chunks.py`）: `class FrameRing(Stateful)`: `(B, N, C, H, W)` のリングと、各ストリームの「連続して入ったフレームの数」（`reset` で 0）。`push_(x, reset)`、`ready -> (B,) bool`（N 枚揃ったストリーム）、`chunk() -> (B, N, C, H, W)`（古い順）。`Stateful` なので再開に乗る。`N=1` で画像の I-JEPA になる
- **いつ学ぶか**: 新しいフレームで `ready` のストリームについて、`stride` フレームに 1 回（`JepaCfg.stride`）。`learn_interval` と同じ `LearnClock` で、学ぶのは 1 フレーム 1 回まで
- **マスク**: ブロックのマスク（V-JEPA の時空間のチューブ）をノードの `self.generator` から引く（再現と再開のため。ノードの状態に入る）
- **崩壊の防止**: `exp.lib.sigreg` の関数版（`sigreg(z, buffer, directions)`）を、B 本のストリームのエンコーダの出力に使う。バッファはノードの中、実動画では数分に広げる（M12 の知見）。`z_norm`（RMSNorm）はスイッチで試す
- **下流へ送る z**: 最新のフレームを含むまとまりのトークンの平均（`(B, D)`）。M16 の動力学の入力になる
- **データ**（`exp/lib/video_io.py`、`exp/video/walking_tours.py`）:
  - `video_io.stream_frames(path, *, fps, size, crop, start=None, duration=None) -> Iterator[np.ndarray]` ではなく torch の `uint8 (n, 3, H, W)` を固定長で `subprocess.Popen` の stdout から読む（動画全体をメモリに読まない）。`-vf fps=<fps>,scale=-2:<H>:flags=area,crop=<H>:<H>`
  - `walking_tours.prepare(src_dir, out_dir, *, fps=10, size=64, jobs=8) -> PrepIndex`: 動画ごとに並列で u8 の memmap（`stream.u8`）と `index.json`（元ファイルの sha256、ffmpeg の版、設定、フレーム数、動画の境界）、評価クリップ `clips.u8`（各動画の終わりから取る。学習のストリームと重ねない）。`data_version = sha256(index.json の内容)[:12]`
  - 前処理は**フリートの補助のジョブ**として 1 回だけ、NAS の `/nas/geson/mujou-exp-cache/walking_tours/<data_version>/` に書く（`.tmp` に書いてから rename）。各ホストは `fleet provision-data <data_version>` で `/data/geson/fleet/data/<data_version>/` に rsync し、manifest を検証してから `.ready`。**run はローカルの複製だけを読む**（`soft` の CIFS から 30 分の run の間ずっと読まない）
  - `WalkingToursStream(path, *, streams: int, seed: int)`: B 本のストリームを、各動画の中の異なる開始位置から流す。`state_dict` は各ストリームの位置
  - 取得のとき、動画ごとのライセンスと、顔とナンバーのぼかしを確かめて `index.json` に記録する（MILESTONES の要求）
- **合成動画から始める**: `data=balls_video`（`BallStreams(render=32)`、玉 1〜3 つ）で、`Frame.state` への線形の probe（`exp.lib.metrics.probe_r2`）と実効ランクが使える。Walking Tours は合成で形が固まってから
- **後から学ぶデコーダ**（`exp/lib/decoder.py`、`exp/video/decode_ckpt.py`、補助のジョブ、`same_host_as` で親の run のホスト）:
  - `fit_decoder(z: Tensor (n, D), images: Tensor (n, C, H, W), *, steps, lr, batch, seed) -> nn.Module`（Adam、エンコーダは止めたまま）
  - `decode_ckpt.main(run_dir, *, steps=2000, frames=16384)`: 各 `ckpt/<t>.pt` のエンコーダで、決まった学習用のフレーム（`data_version` と seed で決まる）を符号化してデコーダを学び、評価クリップを再構成して `decode/<t>.png` と `decode/metrics.jsonl`（`recon`、合成なら `probe_pos`、`probe_vel`、`rank`）。checkpoint ごとに独立なので、checkpoint の数だけ並べられる
  - デコーダは系の外（mujou の実行にも、学習のプロセスにも入れない）
- 評価用の checkpoint の中身: `{"encoder": state_dict, "predictor": state_dict}`

### 8.2 M16: 同時の学習

- `JointNet(mujou.Network)`: `Source → JEPA`、`build_dynamics(net, upstream=jepa.z, act_spec=jepa.z_spec, mods, cfg.dynamics, ...)`。`jepa.z` は送るだけ、`Dynamics.z` と `Compare.z` は受けるだけなので、**動力学の誤差がエンコーダへ戻る経路が型の上で無い**（MILESTONES: 目標の z も入力の z も定数）
- BP 版: `JEPA`（そのまま）→ `DynamicsBP`（入力が z）の 2 ノード。M13 の `WorldBP` のように 1 ノードで損失を足し合わせない（足し合わせると、重み 0 でも組み方の誤りで credit が混ざりうる）
- 動力学の観測の空間は z。標準化は `Standardizer.of` を評価クリップの z で毎評価やり直すのではなく、Compare の誤差は素の z の上でとる（`z_norm` を使えば尺度は固定される）。評価の物差しはデコードした画素の上（M13 の `img_rollout{n} / pix_mean{n}`、`pix_copy{n}`）
- オラクル:
  - `test_run_joint.py::test_encoder_is_blind_to_the_dynamics`: 同じ seed で、動力学をつないだ `JointNet` と、つながない `JEPA` だけのネットワークを同じフレームで回すと、エンコーダの重みがバイト単位で一致（credit が漏れていない最も直接の証拠）
  - `test_run_joint.py::test_dynamics_part_is_the_m14_one`: 上流に決まった z の列を流すと、`build_dynamics` の部分の振る舞いが M14 の `DynamicsNet` に同じ列を観測として流したものと一致
- 評価: `decode_ckpt` を流用し、動力学が予測した z をデコードして 5・10・20 フレーム先を真のフレームと比べる。checkpoint には `encoder`・`predictor`・`step`・`readout` を全部入れる

---

## 9. 実装の計画（並列の作業の単位）

### 9.1 順序の考え方

- 最初の実験を早く始めるため、**インフラ（fleet と run の契約）と M14 段 0 を同時に作り、合流したところで段 0 の物差しと calibrate を回す**。オフラインの上限・見た目・optuna はその後
- 堅牢性の部品（終了の記録、心拍、manifest、再開、監査）は「後で足す」にしない。最初の run から入れる。後から足すと、それまでの run の信頼性が分からなくなる
- 作業の単位ごとに**持ち主のファイル**を決め、ほかの単位のファイルに触らない。共有するのはこの文書の §3（契約）と §7 のシグネチャだけ。`pyproject.toml` と `uv.lock` に触るのは WP2 だけ、`compose.yaml` と `.env.example` は WP0 だけ

### 9.2 作業の単位

| WP | 内容 | 持ち主のファイル | 依存 | 目安 |
| -- | ---- | ---------------- | ---- | ---- |
| **WP0** | ホストの準備: compose.yaml（`FLEET_DIR`、`restart`）、`.env.example`、`.gitignore`（`.fleet/`、`fleet/state/`）、4 台でコンテナを作り直す、Mac に Homebrew の rsync、`~/.ssh/config` に ControlMaster | `compose.yaml`、`.env.example`、`.gitignore` | なし | 半日 |
| **WP1** | fleet の核: `model`、`db`、`transport`、`snapshot`、`provision`、`agent`、`schedule`、`reconcile`、`sync`、`daemon`、`cli`、`hosts.toml`、`fleet/README.md`、`tests/fleet/`。`justfile` の `fleet` レシピ | `fleet/**`（`opt/`・`index`・`table` を除く）、`tests/fleet/**`、`justfile` | WP0 は実機の確認のときだけ | 2 日 |
| **WP2** | run の契約: `exp/lib/{rundir,provenance,jobspec,conf,resume,program,png}.py` と試験。`hydra-core` を `uv add`。`exp/lib/{config,record,sweep,linear_stream}.py` とその試験を消す。`MinGRUTrace` を `Stateful` に。`optim.py` を削る。`host.py` から `RealtimeHost` を消し `frames` を復元可能に。`tests/conftest.py` に `cuda(n)` の skip | `exp/lib/**`、`tests/lib/**`、`pyproject.toml`、`uv.lock`、`tests/conftest.py` | なし | 2 日 |
| **WP3** | M14 のデータ: `BallStreams`（ビット一致）、`obs.py`、`baselines.py` と試験 | `exp/video/{balls,obs,baselines}.py`、`tests/video/test_{balls,obs,baselines}.py` | なし | 1.5 日 |
| **WP4** | M14 のモデル: `nodes.py`（`Source`、受けるだけのポート、`Compare` から kappa を外す）、`dynamics.py`、`rollout.py` と試験 | `exp/video/{nodes,dynamics,rollout}.py`、`tests/video/test_{nodes,dynamics,rollout}.py` | WP2 の `mingru`（`Stateful` 化）の interface だけ | 2 日 |
| **WP5** | M14 のプログラム: `schema.py`、`conf/**`、`run_dynamics.py`、試験（契約、決定性、再開、発散、停止、全 preset の解決）。古い `exp/video/*.py` と `tests/video/*` を消す（§1.3）。`exp/README.md` と `exp/video/README.md` の実行方法 | `exp/video/{schema,run_dynamics}.py`、`exp/video/conf/**`、`tests/video/test_{run_dynamics,schema,conf_tree}.py`、消すファイル | WP2・WP3・WP4 | 1.5 日 |
| **WP6** | optuna と集計: `fleet/opt/**`（別の uv プロジェクト）、`fleet/{index,table}.py`、`exp/video/studies/*.toml`、試験 | `fleet/opt/**`、`fleet/{index,table}.py`、`tests/fleet/{test_index,test_table}.py`、`tests/fleet/opt/**`、`exp/video/studies/**` | WP1 の `db` と `cli` の interface、§3 の契約（偽の run ディレクトリで試験できる） | 2 日 |
| **WP7** | M14 のオフラインの上限 | `exp/video/offline_dynamics.py`、試験 | WP5 | 1 日 |
| **WP8** | M14 の見た目: `viz_dynamics.py`、fleet の補助のジョブ（`--after`、`same_host_as`）で自動で流す | `exp/video/viz_dynamics.py`、試験 | WP5（png は WP2） | 1 日 |
| **WP9** | M14 段 3（画像の観測）: `obs` の `IMAGE`、`BallStreams(render=)`、画素の物差し、`conf/obs/s3.yaml` | `exp/video/{obs,baselines}.py` の追記（WP3 の後なので衝突しない）、`conf/obs/s3.yaml` | 段 1・2 が通ってから | 1 日 |
| **WP10** | M15 のデータ: `exp/lib/video_io.py`、`exp/video/walking_tours.py`、`fleet provision-data`、合成の mp4 による試験 | `exp/lib/video_io.py`、`exp/video/walking_tours.py`、`fleet/provision.py` への小さな追記（WP1 の後） | WP1・WP2 | 2 日 |
| **WP11** | M15 のモデル: `chunks.py`、`vit.py`、`jepa.py` と試験（`FrameRing` の再開、マスクの再現、ノードの中の勾配の中心差分、局所性） | `exp/video/{chunks,vit,jepa}.py`、試験 | WP2 | 3 日 |
| **WP12** | デコーダ: `exp/lib/decoder.py`、`exp/video/decode_ckpt.py` | 同左、試験 | WP2 | 1 日 |
| **WP13** | M15 のプログラム: `run_jepa.py`、`schema.py` と `conf/` への追記 | `exp/video/run_jepa.py`、`conf/{encoder,jepa,experiment/m15}/**` | WP10〜12 | 1.5 日 |
| **WP14** | M16: `run_joint.py`、no-credit のオラクル | `exp/video/run_joint.py`、`conf/experiment/m16/**`、試験 | M14・M15 の結論 | 1.5 日 |

並列にできる組: **WP1・WP2・WP3・WP4 を同時に**（ファイルが重ならない）。WP4 は `MinGRUTrace` の `Stateful` 化（WP2）を待たずに、interface（`state_dict` / `load_state_dict` を持つ）を仮定して書ける。WP5 と WP6 は同時に。WP10・WP11・WP12 は M14 の実験が走っている間に同時に。

### 9.3 PR の切り方

1. **PR-infra**（`chore/<日付>/fleet-and-run-contract`）: WP0・WP1・WP2（と WP6 の index・table）。M14 の前に merge できる形にする。mujou の変更は無い
2. **PR-M14**（`exp/<日付>/continuous-stream-m14-dynamics`）: WP3・WP4・WP5・WP7・WP8・WP9 と、README・`results/m14.md` の数字
3. **PR-M15**、**PR-M16**: 1 マイルストーン 1 PR

PR-infra を merge する前に PR-M14 の作業を始めるときは、PR-M14 のブランチを PR-infra のブランチから切る（`/merge-main` で後から追従）。

### 9.4 最初の 1 週間の実験の流れ

1. （WP0〜WP2 の後、1〜2 日目）`fleet doctor` → `fleet bootstrap` → `fleet snapshot` → 偽の小さなジョブ（`tests` と同じ極小の設定）を 4 台の各スロットの種類で 1 本ずつ。**わざと失敗させる**: ジョブを `kill -9`、コンテナを `docker restart`、ssh を切る、`stop.request` を出す。§4.7 の表どおりに分類され、再開した run の指標が止めなかった run と一致することを実機で確かめる（ここで通らなければ、何百本を回す前に直す）
2. （WP5 の後、3〜4 日目）M14 の手順 1: `fleet submit m14/s0_baseline --sweep obs=s0,s1,s2`（`learner=none`）で物差しを測る → README に閾値を書く
3. 同時に `m14/s0_calibrate`（小・大 × PC・BP × trace あり・なし × CPU・4090・4060、仮想時間 120 秒）で frames/s とメモリを測り、preset の既定の `resources` を決める。30 分の run の時間を README の計画に書く
4. 段 0 のオンライン: BP の探索（WP6 があれば optuna、無ければ `--sweep` の格子）→ `fleet opt confirm` で PC-ALM と BP、seed 0/1/2
5. WP7 のオフラインの上限（段 0〜2）を、段 0 のオンラインと並べて流す
6. 段 1・段 2 へ

---

## 10. 未確認のこと（実装の最初に確かめる）

- OmegaConf 2.3 が `float | None`（PEP 604）と `list[float]` の構造化された設定を扱うか（WP2 の最初の試験。駄目なら `Optional[float]` に）
- `run.state_dict()` を `ManualRun` の step の合間に取り、`mujou.testing.run(net, state=tree)` で戻したとき、ホストが書いた `Source.prev` の値と時刻が戻るか。戻らなければ、再開の最初のフレームの扱いを `HostState` で補う（再開の試験が教えてくれる）
- `uv sync --frozen --no-editable` で mujou が wheel として入り、`exp` が `cwd` から import されるか
- `torch.use_deterministic_algorithms(True)` で、M14 のノードの演算に例外になるものが無いか（あれば、その演算を決定的なものに置き換える。決定性を黙って緩めない）
- GPU 1 枚に並べる本数と 1 本の速さ（§4.4）
- Walking Tours の動画ごとのライセンスと、顔・ナンバーのぼかし（M15 の取得のとき）
