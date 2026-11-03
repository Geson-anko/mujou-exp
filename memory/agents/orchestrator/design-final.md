# 最終設計: M14〜M16 の再編（4 台のフリートで多数の実験を回す）

2026-11-03、lead architect（orchestrator 配下）。main 06a2a11、mujou 881c4ee。既存のファイルは変えていない。

入力: `m14-m16-constraints.md`、`exp/video/MILESTONES.md`、`map-*.md`、3 つの案（`design-velocity.md`・`design-robust.md`・`design-science.md`）と、2 人の審査の結果（std-00 と Mac で確かめた事実を含む）。

**土台**: science 案。2 人の審査の合計点は velocity と同点だったが、目的の後半「M14〜M16 に正しく答える」で最も強い。インフラは science の「Mac が配る」をやめ、velocity の**取りに行く worker**（NAS 上のキュー、std-00 の調整役）に替える。robust からは run の契約（終了コード、supervisor の記録、manifest、再開）、git の snapshot、envkey ごとの venv を移す。robust の Mac の DB・reconcile・daemon・fencing の機構は入れない。

---

## 0. 要点

### 0.1 設計を決めた事実（審査で確かめたもの）

| 事実 | 帰結 |
| -- | -- |
| ホストは Ryzen 9 7950X、**16 物理コア × SMT 2 = 32 論理 CPU**。M13 では 16 本を並べると 1 本が 4 倍遅くなった（当時は `OMP_NUM_THREADS` が未設定） | 枠の既定は 1 台 14 本（物理コア単位）。`fleet bench` で総スループットが頭打ちになる本数を測ってから増やす |
| `/nas` の CIFS は `soft,nobrl,cache=strict,actimeo=1` | 実行中の run の出力・checkpoint・sqlite は NAS に置かない。NAS には小さいキューのファイルと、完了した run の保管だけを置く。NAS の操作は EIO を再試行する |
| SMB の `rename` はサーバーの側で原子的（velocity の前提。2 台で取り合う確認は WP0） | キューの claim は `os.rename` |
| コンテナに `python3` は無い。venv の python を絶対パスで呼ぶ | worker・CLI・run はすべて `<venv>/bin/python` で起動する |
| GPU 2 枚は既にコンテナから見える（devcontainer CLI が DeviceRequests を付ける）。compose.yaml には GPU もワークスペースの bind も無い。RestartPolicy は `no` | コンテナの作り直しは素の `docker compose up` ではなく `devcontainer up`（`just container true`）。compose に `restart: unless-stopped` を足す |
| Mac の submodule は初期化済み（881c4ee） | snapshot は Mac の mujou の作業ツリーから作れる |
| OmegaConf 2.3（3.14t）の Enum / StrEnum は**メンバーの名前**でしか照合しない（`kind: sgd` も `obs: pos` も ValidationError）。`frozen=True` の dataclass は override が ReadonlyConfigError | 設定の種類は `str` + `__post_init__` の検査。schema は frozen にしない |
| `kw_only`、`float \| None`、`list[float]`、`X \| None` の dataclass のノードは OmegaConf 2.3 / hydra 1.3.8 on 3.14t で動く | 有無は `X \| None` で表す（Union や派生クラスの群は使わない） |
| `torch.use_deterministic_algorithms(True)` は torch 2.15 で `torch._dynamo`・`triton` を import し、**GIL が戻る**。`torch._C._set_deterministic_algorithms(True)` なら戻らない | 決定性は private の関数と TF32 の無効化で設定し、GIL の検査に `torch._dynamo` の見張りを足す |
| hydra-core・omegaconf・PyYAML の import では GIL は戻らない（macOS で確認。Linux は WP0 で確認） | 手元の 1 本は学習のプロセスで compose してよい |
| 学習の venv に numpy は無い。optuna は numpy・sqlalchemy を連れてくる | optuna は学習の venv に入れない。調整役は別の uv プロジェクト（`tools/sweep/`） |
| std-00 では今も別の bench の run（`exp.video.run`、GPU 83〜90%）が動いている | コンテナを作り直す前に、全ホストで走っている run が無いことを確かめる（WP0 の手順） |

### 0.2 審査で挙がった欠陥と、この設計での対処

| 欠陥 | 対処 |
| -- | -- |
| Mac が配る方式は Mac が寝ると止まる（robust・science） | worker が NAS のキューから取りに行く。optuna の調整役は std-00 のコンテナに常駐する。Mac は投げて、後で読むだけ（4 節） |
| run の出力を NAS に直接書く（velocity） | 実行中はホストのローカル（`/data/geson/mujou-exp/.fleet/runs/`）に書く。途中経過の小さいファイル（`status.json`・`evals.jsonl`）だけを worker が 60 秒ごとに NAS の `live/` へ写す。終わったら全体を NAS へ写して manifest を検証する（4.6） |
| 再開がない（velocity） | 再開用の checkpoint と `test_resume_equals_uninterrupted`（3.5） |
| 28〜30 枠の見積もりは過大 | 枠は物理コア単位で既定 14。`fleet bench`（1/4/8/14/16/24/30 本）で決める（4.9） |
| rsync のミラーで中身と hash がずれる、mujou の再ビルドが走っている run の下で起きる（velocity） | git の commit-tree の snapshot を各ホストへ push し、`git archive` で読み取り専用の木にする。venv は envkey ごと（4.4） |
| `use_deterministic_algorithms` で GIL が戻る（robust） | `torch._C._set_deterministic_algorithms`（0.1）。CUDA では `warn_only=True`（GPU の run のビット一致は保証しない） |
| Enum の名前照合、frozen の override（robust・science） | `str` + `__post_init__`、frozen にしない（2.2） |
| 発散を FAIL として tell する（science） | ペナルティの値（$J=\log 10$）で tell する。FAIL は例外のときだけ（5.2） |
| 探索と報告で同じクリップ・seed（velocity） | calib / val / test のクリップ、探索の seed 100 番台、確認の seed 0/1/2（3.7、5.4） |
| credit を「zeros を送る約束」で止める（velocity） | 送るだけの `Source` と受けるだけのポート。接続の型で credit の経路が無い（7.5） |
| run_id が時刻で、投げ直しの重複を弾けない（science） | run_id は (program, 設定, snapshot, 資源の種類) の hash + seed。同じものの投げ直しは飛ばす（2.4） |
| 長い run がある間は mujou を更新できない（science） | envkey ごとの venv（`--no-editable`）。違う mujou の版の run が同じホストで並ぶ（4.4） |
| 規模が大きすぎる（robust） | Mac の DB・reconcile・daemon・fencing の WP を持たない。状態はすべてファイルから導く。fleet は 8 モジュール（4.1） |
| M15 に PC-ALM の版がない（robust） | science の「ビューをサンプルとして並べる」PC-ALM 版と、同じ部品の BP 版を両方作る（8.2） |
| コンテナが再起動すると run が全部死ぬ | `restart: unless-stopped`。worker は起動時に自分のホストの走っていた run を再開用の checkpoint から戻す（4.6） |
| `chmod a-w` の木で `__pycache__` が書けない | `PYTHONPYCACHEPREFIX=/tmp/pycache` を worker が渡す |

### 0.3 科学の面で採るもの（science 案の要）

1. 探索と報告を分ける。optuna は val のクリップと seed 100/101 だけを見る。README の数字は test のクリップと seed 0/1/2 の確認の run だけから取る
2. M14 の物差しに**等加速度の外挿**を足す。誤差を `@contact` / `@free` と玉ごとに分ける。等速の外挿を下回れば「重力を学んだ」、接触のある窓で等加速度の外挿を下回れば「跳ね返りを学んだ」と分けて書く
3. 評価は 2 通り: 評価クリップで $s=0$ から 16 フレーム観測する評価と、学習中のストリームから分岐する評価（`fork`）。差で暖機不足と動力学の失敗を切り分ける
4. BP で選んだ値を PC-ALM に移すことの公平さを、最良の設定の学習率 ×0.5/1/2 の bracket で両方の learner について確かめる
5. M15→M16 の受け渡しの条件: 玉ごとの位置の probe の $R^2 \ge 0.9$（基準は学習前の同じ ViT と生の画素）。M16 には凍結したエンコーダの対照を置く。Walking Tours は 1 本を評価専用に取っておく
6. M16 のオラクル: 動力学を付けても付けなくても、同じ seed でエンコーダの重みがビット単位で同じ

---

## 1. パッケージの構成

### 1.1 方針

- 3 層の規則はそのまま。`exp/lib/` は mujou・torch・標準ライブラリだけに依存する。hydra を使う compose は `exp/video/` に、optuna を使う調整役は `tools/sweep/`（別の uv プロジェクト）に置く
- `exp/lib/fleet/` と `exp/lib/runs.py` は**標準ライブラリだけ**で、`exp.lib` の他のモジュールを import しない（Mac の素の Python 3.14 でも動く）。`exp/lib/__init__.py` は docstring だけのまま
- `exp/video/` は 1 実験 1 パッケージのまま、中を領域ごとの下位パッケージに分ける。並行する WP の担当がディレクトリで分かれる
- 互換は捨てる。M13 以前のスクリプト・EgoDex のコード・CNN のエンコーダ・`Reconstruct`・`Correlation`・線形回帰の段 0・`RealtimeHost` は消す（過去の数字は `results/*.md` と git の履歴に残る）

### 1.2 木

```text
exp/lib/
  host.py          Host だけ（RealtimeHost を消す。frames を state_dict に）
  mingru.py        MinGRU, ResidualMinGRU, MinGRUTrace（mujou.Stateful にする）
  sigreg.py        そのまま（M15 で使う）
  optim.py         Warmup, AdamDirection, Grafted と adam(lr, beta1, beta2) -> pc.optim.AdamW の換算。
                   Whitened・Projected・Orthogonalized・Correlation・_Factors・_eigh を消す
  metrics.py       effective_rank, probe_r2, bootstrap_se（新）
  rundir.py        run の契約（3 節）: RunDir, exit codes, stop request, manifest
  provenance.py    コードの版、環境、assert_free_threaded, configure_determinism
  ckpt.py          評価用の ckpt（重みだけ）と再開用の ckpt（mujou の木 + HostState + offsets）
  png.py           write_png, tile, draw_points_, draw_polyline_（decode.py から）
  video_io.py      ffmpeg_frames, U8Writer, U8Reader（M15）
  runs.py          [stdlib] run の索引と表（6 節）
  fleet/           [stdlib] ランチャー（4 節）
    __main__.py  cli.py  config.py（fleet.toml）  jobs.py（job の形、id、group）
    queue.py（NAS のキュー）  snapshot.py（Mac: commit-tree と push）
    provision.py（ホスト: archive と venv）  worker.py（ホストに常駐）  slots.py（純関数）
exp/video/
  README.md PLAN.md MILESTONES.md results/ report*.html   残す
  main.py          python -m exp.video.main --job <job.json> | <overrides...> --out DIR
  loop.py          汎用のホストのループ（Program の protocol、評価・ckpt・発散・停止・心拍）
  compose.py       hydra の Compose API の薄い包み（conf/ と名前がぶつからないよう compose.py）
  schema/          structured config（領域ごとのファイル。torch を import しない）
    __init__.py（ConfigStore への登録）  run.py  data.py  dyn.py  vision.py  learn.py  measure.py
  conf/            hydra の yaml（2.3）。__init__.py を置いて package にする
  programs/        dynamics.py（M14）  vision.py（M15）  joint.py（M16）
  data/            physics.py  observe.py  clips.py  frames.py  walking_tours.py
  dyn/             modules.py  nodes.py  bp.py  nets.py  offline.py  pixels.py（段 3）
  jepa/            vit.py  masks.py  loss.py  nodes.py  bp.py  nets.py
  measure/         baselines.py  dynamics.py  objective.py  viz.py  vision.py  posthoc.py
  studies/         optuna の study の定義（TOML、commit する）
tools/sweep/       optuna の調整役。pyproject.toml（optuna==5.0.0, hydra-core==1.3.8）、
                   sweep/{__main__,coordinator,space,objective}.py、tests/
fleet.toml         ホストと枠（repo のルート、commit する）
```

`eval` は組み込みの名前なので、評価の下位パッケージは `measure/` にする。

### 1.3 消すもの

- `exp/lib/{config,record,sweep,linear_stream}.py` と対応するテスト
- `exp/video/` の `easy, egodex, stage0..3, stability, latency, frozen, collapse, model, nodes, bp, evaluate, decode, offline, config, run, balls` の `.py`。物理の参照の実装（`balls._step` と `Video`）は `tests/video/data/reference_balls.py` に移し、ベクトル化した物理のビット一致の試験に使う
- 対応するテスト（`tests/video/test_*.py` の全部、`tests/lib/test_{config,record,linear_stream}.py`）。オラクルの型（平衡点の `lam` = BP の credit、トレース = BPTT、恒等モデル = 写し、`window_starts`、PNG の手での復号、物理の不変量）は新しいテストに移す

### 1.4 テストの構成

`exp/<pkg>/<mod>.py ↔ tests/<pkg>/test_<mod>.py` のミラー。`tools/sweep/tests/` は別。

```text
tests/
  conftest.py            OMP_NUM_THREADS=1。cuda(n) マーカーの skip の hook を実装する
  test_gil.py            exp.video.main・hydra・omegaconf を import し、小さいネットワークで step した後も GIL が無効
  test_claude_hooks.py test_mujou_install.py
  lib/  test_{host,mingru,sigreg,optim,metrics,rundir,provenance,ckpt,png,video_io,runs}.py
        fleet/ test_{jobs,queue,snapshot,provision,worker,slots,cli,stdlib_only}.py
  video/ test_{main,loop,compose}.py
         schema/ test_presets.py           全 preset が compose でき、to_object が往復する
         programs/ test_{dynamics,vision,joint}.py
         data/ test_{physics,observe,clips,frames,walking_tours}.py  reference_balls.py
         dyn/ test_{modules,nodes,bp,nets,offline,pixels}.py
         jepa/ test_{vit,masks,loss,nodes,bp,nets}.py
         measure/ test_{baselines,dynamics,objective,viz,vision,posthoc}.py
```

`pyproject.toml` の pyright と ruff の exclude に `.fleet` と `tools/sweep/.venv` を足す。

---

## 2. hydra の設定

### 2.1 使い方

- **Compose API だけ**（`initialize_config_module("exp.video.conf")` + `compose`）。`@hydra.main`・hydra の出力ディレクトリ・multirun・sweeper は使わない。並列はフリート、探索は optuna が受け持つ
- compose は**投入のとき**に、run が走るのと同じ code snapshot の中で行う（std-00 のコンテナで `fleet submit` が呼ぶ。4.7）。解決済みの設定を `job.json` に入れるので、設定の誤りはキューに入る前に落ち、`group` の hash も投入の時点で決まる
- 学習のプロセスは `job.json` の設定を `OmegaConf.merge(OmegaConf.structured(RunConf), cfg)` → `to_object` で dataclass に戻すだけ。手元で 1 本試すとき（`python -m exp.video.main +experiment=m14/s0 optim.lr=0.01 --out /tmp/x`）は、同じ `compose()` で `job.json` を作ってから同じ経路で走る
- コードが触るのは dataclass のインスタンスだけ。`DictConfig` を関数の間で渡さない。`compose()` は `isinstance` で `RunConf` に絞って返す
- 学習率は 1 フレームあたりの `lr` で持ち、optimizer を作る関数が `tau = frame / lr` に換算する（対数一様で探索でき、M13 の換算の取り違えが無くなる）

### 2.2 structured config（`exp/video/schema/`）

規則: 種類は `str`（`__post_init__` で値を検査）、Enum と `Literal` は使わない。tuple ではなく `list`。frozen にしない（`to_object` の後で書き換えない規約にする）。部分木の有無は `X | None`。補間と独自の resolver は使わない。

```python
# schema/run.py
@dataclass(kw_only=True)
class RunConf:
    program: str = "dynamics"        # "dynamics" | "vision" | "joint"
    task: str = "online"             # "online" | "offline" | "baselines" | "bench" | "posthoc" | "prep"
    name: str = MISSING              # set by the preset, e.g. "m14/s1"
    tags: list[str] = field(default_factory=list)
    seed: int = 0
    device: str = "cpu"              # the worker overrides it with the slot's device
    phase: str = "explore"           # "explore" | "confirm" | "audit"
    seconds: float = 1800.0          # length of the stream (virtual time)
    time: TimeConf = field(default_factory=TimeConf)
    data: DataConf = field(default_factory=DataConf)
    dynamics: DynamicsConf | None = None     # M14, M16
    vision: VisionConf | None = None         # M15, M16
    learner: LearnerConf = field(default_factory=LearnerConf)
    optim: OptimConf = field(default_factory=OptimConf)          # the dynamics' optimizer
    optim_vision: OptimConf | None = None                         # the encoder's (M15, M16)
    offline: OfflineConf | None = None
    measure: MeasureConf = field(default_factory=MeasureConf)
    ctl: CtlConf = field(default_factory=CtlConf)    # log/eval/ckpt/resume intervals; not in the group hash
    def __post_init__(self) -> None: ...             # the allowed strings; the program needs its subtrees

# schema/data.py
@dataclass(kw_only=True)
class PhysicsConf:                 # EASY of M13 by default
    gamma0: float = 1.0; gamma_amp: float = 0.0; e0: float = 1.0; e_amp: float = 0.0
    drag: float = 0.0; period: float = 600.0; conserve: bool = True; substeps: int = 8

@dataclass(kw_only=True)
class BallsConf:
    radii: list[float] = field(default_factory=lambda: [0.10])
    obs: str = "pos_vel"           # "pos_vel" (stage 0) | "pos" (1, 2) | "pixels" (3) | "frames" (M15, M16)
    size: int = 32
    physics: PhysicsConf = field(default_factory=PhysicsConf)

@dataclass(kw_only=True)
class ToursConf:
    data_version: str = MISSING    # the prepared u8 data (8.3)
    held_out: str = MISSING        # the video kept for evaluation only
    size: int = 64; fps: float = 10.0

@dataclass(kw_only=True)
class DataConf:
    streams: int = 128             # B
    balls: BallsConf | None = field(default_factory=BallsConf)
    tours: ToursConf | None = None # exactly one of balls and tours

# schema/learn.py
@dataclass(kw_only=True)
class OptimConf:
    kind: str = "sgd"              # "sgd" | "adamw" | "graft_adam"
    lr: float = 0.03               # per frame
    warmup: float = 300.0          # seconds of virtual time; 0 for none
    tau_m: float = 0.3; tau_v: float = 30.0; weight_decay: float = 0.0

@dataclass(kw_only=True)
class LearnerConf:
    kind: str = "bp"               # "pc" | "bp" | "none" (baselines only)
```

`dyn.py`（`DynamicsConf`: `d_s, nodes, trace, residual, open_loop, open_cycle`）、`vision.py`（`VisionConf`: ViT の大きさ、`frames` N、マスク、`sg`、`sigreg`、`z_pool`）、`measure.py`（`MeasureConf`: `observed=16`, `horizons=[1,5,10,20]`, `clips=256`, `windows=4`, `every=60.0`, `diverge=10.0`、確認の閾値 `criteria`）は 7・8 節のシグネチャに合わせる。`CtlConf` は `log_every=1.0`（仮想時間の秒）、`ckpt_every`、`resume_every=300.0`、`viz_every` を持つ。

### 2.3 yaml の木（`exp/video/conf/`）

```text
conf/
  __init__.py
  run.yaml                 defaults: [_self_, dynamics: small, optim: sgd, learner: bp, phase: explore]
  dynamics/ none.yaml small.yaml big.yaml          small: d_s 256, 1 node / big: d_s 512, 3 nodes
  vision/   none.yaml vit_t.yaml vit_s.yaml
  learner/  pc.yaml bp.yaml none.yaml
  optim/    sgd.yaml adamw.yaml graft_adam.yaml
  phase/    explore.yaml confirm.yaml audit.yaml   confirm: test も評価、ckpt 300 秒、viz 300 秒
  experiment/                                      "# @package _global_"
    m14/ s0.yaml s1.yaml s2.yaml s3.yaml baselines.yaml offline.yaml bench.yaml
    m15/ balls_n1.yaml balls_n4.yaml wt_n1.yaml wt_n8.yaml
    m16/ balls.yaml wt.yaml frozen.yaml
```

```yaml
# conf/experiment/m14/s2.yaml
# @package _global_
defaults:
  - override /dynamics: small
name: m14/s2
tags: [m14, s2]
data:
  balls: {radii: [0.06, 0.08, 0.10], obs: pos}
```

### 2.4 id・group・名前

- `group = sha256(canonical_json({program, config から seed・device・ctl・phase を除いたもの, snap, res_class}))[:12]`。seed だけ違う run が同じ group
- `run_id = f"{name の / を -}-{group[:8]}-s{seed}"`（例 `m14-s1-3fa29c01-s0`）。監査の run は元の run_id に `-audit` を付ける。同じ (設定, snapshot, 資源の種類, seed) の投げ直しは、NAS に `result.json` か queue / claimed のファイルがあれば飛ばす（`duplicate` と出す）
- `job.json` に override の列をそのまま残す。集計は名前ではなく設定の値と `group` で絞る。`fleet` は seed 以外の override から読む用の短い名前（`pc big tr1 lr0.02`）を作るが、識別には使わない

---

## 3. run の契約

### 3.1 原則

1. **指標のファイルは (コード, 設定) の純関数**。壁時計・ホスト・pid は `evals.jsonl` と `stream.jsonl` に入れず、`status.json` と `timing.jsonl` に分ける。2 つの run の指標をバイト列の sha256 で比べられる
2. JSON は一時ファイルに書いて `os.replace`。jsonl は行ごとに書いて flush、評価と再開用の checkpoint のときに fsync。再開のときは checkpoint に記録した byte offset で切り詰める
3. 完了の印は最後に書く `manifest.json`。これが無ければ、終了コードが 0 でも完了とみなさない
4. 終了の理由は外側（worker）が `exit/attempt-<n>.json` に書く。segfault や OOM killer でも理由が残る

### 3.2 ディレクトリ

実行中はホストの `/data/geson/mujou-exp/.fleet/runs/<batch>/<run_id>/`。完了後に NAS の `/nas/geson/mujou-exp/runs/<batch>/<run_id>/` へ写す（`resume/` を除く）。

```text
job.json           不変。worker が起動前に置く（3.3）
config.yaml        解決済みの設定（人が読む用）
CODE.json code.diff    snapshot の版と、base_head からの差分（mujou を含む）
provenance/attempt-<n>.json   環境（3.4）
status.json        心拍（壁時計 30 秒ごと）: state, attempt, pid, pgid, frame, t, of, fps, eta, rss_mb, gpu_mem_mb
timing.jsonl       壁時計の記録
stream.jsonl       仮想時間 log_every 秒ごと: {"t", "frames", "preq", "preq_open"}（毎フレームの行は無くす）
evals.jsonl        評価 1 回 1 行（3.7）
baselines.json     物差し（val と test の各 h）
result.json        完了か発散のときだけ
manifest.json      最後に書く。resume/ logs/ exit/ status.json timing.jsonl 以外の sha256 とサイズ
ckpt/<t:07.1f>.pt  評価用（モジュールの重みだけ）
resume/<frame:08d>.pt  resume/latest   再開用（最新 2 つ、ローカルだけ）
viz/<t>/           見た目の確認の PNG（frames.png, traj.png）
post/              後から学ぶデコーダ（M15・M16）
logs/attempt-<n>.{out,err}   exit/attempt-<n>.json   stop.request
```

### 3.3 `job.json`

```json
{"schema": 1, "run_id": "m14-s1-3fa29c01-s0", "batch": "m14-s1-confirm-v1", "group": "3fa29c01ab12",
 "program": "dynamics", "seed": 0, "overrides": ["+experiment=m14/s1", "learner=pc", "seed=0"],
 "config": {"...": "解決済み"},
 "code": {"snap": "<40 hex>", "base_head": "<40 hex>", "dirty": false,
          "mujou_snap": "<40 hex>", "mujou_pushed": true, "envkey": "<12 hex>"},
 "res": {"class": "cpu", "ram_gb": 2.0, "gpu_mem_gb": 0.0},
 "prio": 50, "after": [], "host": null, "avoid_host": null,
 "study": null, "trial": null, "audit_of": null, "tags": ["confirm"],
 "retry": {"oom": 1, "lost": 3, "stall": 1}, "submitted": "2026-11-03T12:00:00+09:00"}
```

`res.class` は `cpu` / `any_gpu` / `4090` / `4060`。`host` は再開を同じホストに縛るときだけ、`avoid_host` は監査の run が別のホストで走るために使う。

### 3.4 provenance と決定性（`exp/lib/provenance.py`）

```python
def collect(job: Mapping[str, Any], *, attempt: int) -> dict[str, Any]
def assert_free_threaded(where: str) -> None
    # EnvError when Py_GIL_DISABLED and sys._is_gil_enabled(), or "torch._dynamo" in sys.modules;
    # the message lists the C extensions imported since the previous check
def configure_determinism(device: str) -> dict[str, Any]
    # torch.set_num_threads(1); TF32 off (matmul, cudnn);
    # torch._C._set_deterministic_algorithms(True, warn_only=device.startswith("cuda"))
    #   (not torch.use_deterministic_algorithms: it imports torch._dynamo and enables the GIL);
    # refuses to start (EnvError) unless OMP_NUM_THREADS == "1"; checks CUBLAS_WORKSPACE_CONFIG on CUDA
```

中身: `python`、`gil_enabled`、`torch`、`cuda`、GPU の名前、CPU のモデル、`os.sched_getaffinity(0)`、環境（`OMP_NUM_THREADS` など、`PYTHON_GIL` の有無）、決定性の設定、`data_version`（M15）、`host`、`attempt`。

**再現の単位**: CPU では (snapshot, 設定) が同じなら指標のバイト列が同じ。CUDA では同じ GPU の種類でも保証しない（`warn_only`）。だからデバイスの種類を必ず記録し、確認の batch は全 seed・全 learner を同じ種類に固定する。監査（4.8）は CPU の run だけでビット一致を見る。

### 3.5 checkpoint と再開（`exp/lib/ckpt.py`）

| | 評価用 `ckpt/<t>.pt` | 再開用 `resume/<frame>.pt` |
| - | -- | -- |
| 中身 | `{"schema", "program", "group", "t", "frame", "modules": {name: state_dict}}` | `{"schema", "group", "frame", "mujou": run.state_dict(), "host": HostState, "offsets": {file: bytes}}` |
| 読み方 | `torch.load(weights_only=True)` → 設定からモジュールを組み直して `load_state_dict` | 同じ構造のネットワークを組み直し `mujou.testing.run(net, state=tree["mujou"])` |
| 頻度 | `ctl.ckpt_every`（confirm は 300 秒） | `ctl.resume_every`（300 秒） |
| 保管 | NAS に写す | ローカルだけ、最新 2 つ、完了したら消す |

- `HostState`: ストリームの状態（`BallStreams` の位置・速度・generator、Walking Tours の読み出し位置）、`Host.frames`、累積の途中の値、発散の判定の窓
- `MinGRUTrace` を `mujou.Stateful` にする（今は普通の属性で `Node.state_dict` に入らず、再開するとトレースが 0 から始まる）
- mujou の木はポートの最後の値と時刻まで持つ（map-mujou §state_dict）。それでも欠けがあれば、再開の試験が落ちて教える
- **すべての program に `test_resume_equals_uninterrupted` を置く**: K フレームで止めて再開し、さらに K フレーム回した `evals.jsonl`・`stream.jsonl` が、止めずに 2K フレーム回したものとバイト単位で一致（CPU）

### 3.6 終了コードと骨格（`exp/video/loop.py`）

```python
class Program[C](Protocol):
    def build(self, cfg: RunConf) -> Built: ...                       # network, streams, clip sets
    def frame(self, built: Built, host: Host) -> None: ...            # show one frame, accumulate on device
    def evaluate(self, built: Built, split: str) -> dict[str, float]: ...   # between steps, no_grad
    def modules(self, built: Built) -> dict[str, nn.Module]: ...      # for the evaluation ckpt
    def host_state(self, built: Built) -> dict[str, Any]: ...
    def load_host_state(self, built: Built, state: dict[str, Any]) -> None: ...
    def visualize(self, built: Built, out: Path) -> None: ...         # viz/<t>/

def run(program: Program[Any], job: Path, *, attempt: int) -> int     # returns the exit code
```

`run` の順: `job.json` を読んで検証 → `assert_free_threaded("import")` → `configure_determinism` → provenance → `resume/latest` があれば再開 → `build` → `assert_free_threaded("build")` → ループ（`frame`、`log_every` ごとに 1 回だけ `.item()` して `stream.jsonl`、発散の判定、`stop.request` の確認、評価、checkpoint、心拍）→ `result.json` → `manifest.json`。

| コード | 意味 | 書くもの | worker の扱い |
| -- | -- | -- | -- |
| 0 | 完了 | result（`completed`）、manifest | NAS へ写して検証、`done/` |
| 3 | 発散（科学の結果） | result（`diverged`、`reason`、`t`）、manifest | 完了の一種。再試行しない。表に「発散」 |
| 4 | 環境の誤り（GIL、OMP、CUDA が見えない、データが無い） | stderr | そのホストのその資源の種類を cordon（`hosts/<host>.cordon`）し、job をキューに戻す |
| 75 | 停止の要求に従った | 再開用の checkpoint | 理由が `pruned` なら終わり、`drain` / `move` なら再投入 |
| 1 ほか | 例外（コードの誤り） | traceback | 失敗。再試行しない。同じエラーの印が batch で 3 回なら batch を保留（4.6） |
| シグナル | 外から殺された | （worker が記録） | 停止の要求の無い SIGKILL は OOM の見込み。`ram_gb` を 1.5 倍にして 1 回だけ再試行 |

発散: 前向きの予測が非有限、または評価の `roll10 > measure.diverge × mean10`、または評価の値が非有限。評価の中の例外は握りつぶさない。

`exit 0` なのに `result.json` か `manifest.json` が無ければ、worker は failed（契約違反）とする。黙った成功を許さない。

### 3.7 指標の schema（M14）

`evals.jsonl` の 1 行（値は標準化した観測の空間の 1 次元あたりの二乗誤差。キーは平ら）:

```json
{"t": 600.0, "frame": 18000, "split": "val",
 "roll1": 0.01, "roll5": 0.05, "roll10": 0.12, "roll20": 0.40,
 "mean5": 1.0, "copy5": 0.31, "cv5": 0.08, "ca5": 0.04,
 "roll5/cv5": 0.62, "roll10/cv10": 0.60, "roll10/ca10": 0.9,
 "roll10@contact": 0.30, "roll10@free": 0.05, "ca10@contact": 0.45, "ca10@free": 0.0,
 "roll10@ball0": 0.1, "roll10@ball1": 0.12, "roll10@ball2": 0.15,
 "roll10_se": 0.004, "amp10": 0.95, "motion10": 0.98,
 "fork10": 0.11, "fork10/cv10": 0.6, "nonfinite": 0, "objective": -0.47}
```

- `amp{h}` = 予測の分散 / 真の分散、`motion{h}` = 予測の 1 フレームの移動量 / 真の移動量。「止まって薄れる」を数で拾う補助（見た目の確認の代わりにしない）
- `@contact` / `@free`: 観測の最後から $h$ フレーム先までに、真の軌跡で壁への接触（速度の成分の符号の反転）がある窓 / 無い窓
- `objective` は `measure/objective.py` の 1 つの関数が計算する（5.2）。val の行にだけ入れる。optuna はこの値をそのまま使う
- `result.json`: `{"schema", "status", "reason", "t_end", "frames", "final": {"val": {...}, "test": {...}}, "criteria": {...}, "objective", "metrics_sha256": {"evals", "stream"}, "peak": {"rss_mb", "gpu_mem_mb"}}`。`criteria` は確認の preset の閾値（README と同じ値）を機械で判定したもの。見た目の確認は orchestrator が PNG を見て README に書く

---

## 4. フリート（`exp/lib/fleet/`、標準ライブラリだけ）

### 4.1 形

```text
Mac（orchestrator）
  just fleet push      snapshot を作り 4 台へ git push、各ホストで provision（ssh、並列）
  just fleet <cmd>     状態の問い合わせと投入は std-00 のコンテナへ転送（ssh + docker exec）
  just fleet pull      完了した run の軽いファイルを NAS から（std-00 経由で）Mac の runs/ へ

NAS /nas/geson/mujou-exp/（コンテナにも同じパスでマウント）
  queue/<prio:03d>-<submitted_ns>-<run_id>.json   claimed/<host>/   done/   holds/   failures/<batch>/
  hosts/<host>.json（心拍）  hosts/<host>.cordon   live/<run_id>/{status.json,evals.jsonl}
  runs/<batch>/<run_id>/（完了した run）  studies/<study>/（optuna の DB の写し）  data/  snapshots/

std-0X のコンテナ mujou-exp-dev-1
  worker（1 ホスト 1 本、常駐）── run のプロセス（taskset、start_new_session）
  std-00 だけ: tools/sweep の調整役（study ごとに 1 本、常駐）
  /data/geson/mujou-exp/.fleet/{git/, code/<snap12>/, venv/<envkey>/, runs/, data/, optuna/, worker.log}
```

- ホスト同士は NAS のファイルだけで調整する。Mac が寝ても、キューの job は worker が消化し、optuna は std-00 で回り続ける
- 状態はファイルから導く（DB を持たない）: `queue/` にある → queued、`claimed/<host>/` にあって `done/` に無い → running（そのホストの心拍が 10 分より古ければ `lost?`）、`done/<run_id>.json` の `status`

### 4.2 `fleet.toml`

```toml
nas = "/nas/geson/mujou-exp"
root = "/data/geson/mujou-exp/.fleet"     # host-local, bind-mounted with the checkout
checkout = "/data/geson/mujou-exp"
container = "mujou-exp-dev-1"
user = "vscode"
head = "std-00"                           # state queries, compose at submit, the optuna coordinator
hosts = ["std-00", "std-01", "std-02", "std-03"]

[slots]                                   # set from `fleet bench` (4.9)
cpu = 14                                  # physical cores used for runs; 2 left for the worker, evals, ffmpeg
smt = false                               # true: two runs on the two threads of one core
ram_gb = 52
gpus = [{index = 0, model = "4090", mem_gb = 22.0, jobs = 6},
        {index = 1, model = "4060", mem_gb = 7.0,  jobs = 2}]
```

### 4.3 コンテナの変更（WP0、1 回だけ）

```yaml
services:
  dev:
    restart: unless-stopped
    volumes:
      - ${EGODEX_DIR:-./datasets/egodex}:/datasets/egodex:ro
      - ${WALKING_TOURS_DIR:-./datasets/walking_tours}:/datasets/walking_tours
      - ${FLEET_NAS:-./.fleet-nas}:${FLEET_NAS:-/fleet-nas}   # hosts: FLEET_NAS=/nas/geson/mujou-exp in .env
```

作り直しの手順（各ホスト）: (1) `ssh std-0X 'docker exec mujou-exp-dev-1 pgrep -af "python -m exp"'` で走っている run が無いことを確かめる（あれば終わるまで待つか、ユーザーに確かめる）、(2) `.env` に `FLEET_NAS` を足す、(3) `just container true`（`devcontainer up`。素の `docker compose up` は GPU と workspace の bind を落とす）、(4) コンテナの中で `/nas/geson/mujou-exp` に vscode で書けること、`nvidia-smi` で 2 枚見えることを確かめる。

`.fleet/` と `.fleet-nas/` は `.gitignore` に足す。

### 4.4 コードと環境: git の snapshot と envkey の venv

`fleet push`（Mac）:

1. **snapshot**: 作業ツリーがきれいなら `snap = HEAD`。汚れていれば、一時の `GIT_INDEX_FILE` で `git add -A`（`.gitignore` に従う）→ `write-tree` → `commit-tree -p HEAD`。HEAD もブランチも本物の index も動かさない。mujou も同じ手順で `mujou_snap` を作り、上の index の gitlink を `update-index --cacheinfo 160000,<mujou_snap>,mujou` で差し替える。`refs/fleet/snap/<sha>` を作って gc から守る
2. 4 台へ並列に `git push ssh://std-0X/data/geson/mujou-exp/.fleet/git/repo.git <snap>:refs/fleet/snap/<snap>`（mujou も `mujou.git` へ）。bare repo が無ければ作る
3. **provision**（各ホスト、flock で直列）: `git archive <snap> | tar -x -C code/<snap12>`、`git -C mujou.git archive <mujou_snap> | tar -x -C code/<snap12>/mujou`、`CODE.json` と `code.diff` を置く。`envkey = sha256(uv.lock, pyproject.toml, mujou の tree の sha, python の版)[:12]`。`venv/<envkey>/.ready` が無ければ、コンテナの中で `UV_PROJECT_ENVIRONMENT=.fleet/venv/<envkey> uv sync --frozen --no-editable`（mujou を wheel として入れるので、venv は `code/` の木に依存しない）。検証: venv の python で `mujou.__file__` が venv の中、`sys._is_gil_enabled()` が False、`torch.cuda.device_count() == 2`。`chmod -R a-w code/<snap12>`、`.ready` を書く
4. `code/latest` の symlink を新しい snap に向ける（CLI と worker の起動に使う）

- 違う snapshot・違う mujou の run が同じホストで同時に走れる。走っている 5〜7 時間の run の下でコードも mujou も変わらない
- `phase=confirm` の投入は、`dirty=false` かつ `mujou_pushed=true`（`git branch -r --contains` が空でない）の snapshot しか受け付けない。README の数字はすべて push 済みの commit から再現できる
- 掃除: `fleet gc` は、キューにも実行中にも参照されず 7 日を過ぎた `code/` と、参照の無くなった `venv/` を消す

### 4.5 キュー（`queue.py`）

- 投入: `queue/<prio:03d>-<submitted_ns>-<run_id>.json`（小さい prio が先）。一時の名前に書いてから rename
- claim: `os.rename(queue/<f>, claimed/<host>/<f>)`。負けた側は `FileNotFoundError` で次へ。NAS の操作はすべて `nas_retry`（EIO・ESTALE を指数の backoff で 5 回）で包み、尽きたら worker はその tick を飛ばす（走っている run には影響しない。run はローカルに書いている）
- claim の前に確かめる: `after` の run が `done/` で completed、`holds/<batch>` が無い、`host` / `avoid_host`、自分の `code/<snap12>/.ready` と `venv/<envkey>/.ready` がある（無ければ他のホストに任せる）、そのホストの資源の種類が cordon されていない、空きの枠がある
- 終わった job は `claimed/<host>/` から `done/<run_id>.json`（`status`、`host`、`attempt`、`exit` を足したもの）へ移す

### 4.6 worker（`worker.py`、コンテナの中に常駐）

起動: `fleet up` が各ホストで `docker exec -d -u vscode -w <root>/code/latest mujou-exp-dev-1 <root>/venv/<envkey of latest>/bin/python -m exp.lib.fleet worker --host std-0X`（心拍が 2 分より古いときだけ）。

5 秒ごとのループ:

1. **回収**: 終わった子の終了コードを `exit/attempt-<n>.json` に書く（`returncode`、`signal`、`started`、`ended`、`stop_reason`）。3.6 の表で分類する。completed / diverged は run の木を NAS へ写し（`rsync -rt --no-perms --exclude resume/`、無ければ `shutil.copytree`）、NAS の側で manifest の sha256 を検証してから `done/` に移す。検証に 3 回失敗したら `corrupt`
2. **失敗の印**: exit 1 なら印（例外の型 + traceback の最後の行 + program）を `failures/<batch>/<sig>/<run_id>` に置く。同じ印が 3 つなら `holds/<batch>` を書く（壊れたコードで 4 台を燃やさない。`fleet release` で解く）
3. **停止**: `claimed/<host>/<run_id>.stop`（`fleet stop`・調整役が置く）を見たら run の `stop.request` に写す。`--now` か猶予（120 秒）が過ぎたら、process group に SIGTERM、10 秒後に SIGKILL（worker はコンテナの中にいるので pgid を直接殺せる）
4. **止まりの検出**: `status.json` の `frame` が 20 分進まなければ SIGUSR1（faulthandler が全スレッドの traceback を出す。OpenMP の取り合いの診断）→ 停止 → 1 回だけ再試行
5. **fencing**: 走っている run の claim ファイルが `claimed/<host>/` から消えていたら（`fleet requeue` で他へ移された）、その run を殺し、出力を NAS へ写さない
6. **live**: 走っている run の `status.json` と `evals.jsonl` を 60 秒ごとに `live/<run_id>/` へ写す（調整役と `fleet status` が読む）
7. **起動**: 4.5 の条件で claim し、`slots.allocate` で枠を決めて起動する
8. **心拍**: `hosts/<host>.json`（時刻、枠の使用、走っている run、load、空きメモリ、`nvidia-smi` の空き、worker の snap）

起動の argv と環境:

```text
taskset -c <cpu> <root>/venv/<envkey>/bin/python -m exp.video.main --job <run_dir>/job.json --attempt <n>
cwd = <root>/code/<snap12>; start_new_session=True; stdout/stderr -> logs/attempt-<n>.{out,err}
env: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
     PYTHONPYCACHEPREFIX=/tmp/pycache CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=<i or "">
     FLEET_HOST=std-0X; PYTHON_GIL is removed
```

worker 自身も `PYTHONPYCACHEPREFIX=/tmp/pycache` で起動する（`code/` は読み取り専用）。

worker の再起動（`fleet up --restart`、コードを更新したとき）: 新しい worker は `claimed/<host>/` の run を読み、`status.json` の pid が生きていて `/proc/<pid>/cmdline` に run_id があれば引き取る（終わりは `result.json` と `/proc` で知る）。pid が死んでいて `exit/` も無ければ **lost**: `resume/latest` があれば `--resume` で attempt を 1 つ進めて同じホストで起動し直す（`retry.lost` まで）。コンテナやホストの再起動の後は、`restart: unless-stopped` でコンテナが戻り、orchestrator の `fleet up` で worker が戻って、この手順で run が戻る。

ホストが戻らないとき: `fleet requeue --host std-0X` が `claimed/std-0X/` の job をキューへ戻す（最初から。再開用の checkpoint はそのホストのローカルにあるため）。ホストが後で戻っても、5 の fencing で二重に走らない。

### 4.7 CLI（`just fleet <cmd>`）

```text
just fleet doctor                    ssh、コンテナ、GPU、NAS の書き込み、rename、時計、rsync、物理コアの配置
just fleet push                      4.4。snap を出す
just fleet up [--restart]            worker と調整役（studies/*/RUNNING）を確かめ、無ければ起動
just fleet submit [--snap S] --batch B <overrides...> [--sweep k=a,b ...] [--seeds 0,1,2]
                 [--res cpu|any_gpu|4090|4060] [--ram G] [--gpu-mem G] [--prio P] [--after ID] [--tag T] [--dry]
                                     std-00 の code/<snap> の中で compose し、run_id ごとに new / duplicate を出す
just fleet status [--batch B]        状態ごとの数、ホストごとの枠、走っている run の t/of・fps・ETA・最新の roll10/cv10
just fleet ls [--batch B] [--state S]    just fleet logs <run> [--err] [--tail N]
just fleet stop <run>|--batch B|--trial S/N [--now] [--reason R]    just fleet cancel <run>|--batch B
just fleet retry <run> [--fresh] [--ram G]    just fleet requeue --host H
just fleet hold B | release B | cordon H [--res R] | uncordon H
just fleet pull [--batch B] [--ckpt] [--live <run>]   完了した run（既定で ckpt を除く）、または走っている run をホストから直接
just fleet table ...                 6 節（std-00 で runs.py を呼ぶ）
just fleet bench --preset m14/bench --levels 1,4,8,14,16,24,30     4.9
just fleet x <host|all> -- <cmd...>  コンテナで、code/latest と venv で（テスト、python -m ...）
just fleet gc [--dry-run]
```

- justfile: `fleet *args: uv run --no-project --python 3.14 python -m exp.lib.fleet {{args}}`。Mac で自分が実行するのは `doctor`・`push`・`up`・`pull`・`x`・`gc`。他は `ssh std-00 docker exec ... <venv python> -m exp.lib.fleet <cmd>`（cwd は `code/<snap>` か `code/latest`）に転送する
- ssh は ControlMaster（`ControlPersist=600`）。出力は既定で短い表、`--json` で機械向け
- **canary**: 新しい snapshot で 20 本を超える投入をすると、`submit` は先に canary（最初の設定を仮想時間 60 秒、資源の種類ごとに 1 本）を投げ、残りに `after: [canary]` を付ける。canary が completed でなければ残りは走らない
- 典型の 1 周: `fleet table --batch B` → 決める → `fleet push` → `fleet submit ...` → `fleet status` → `fleet pull --batch B` → Read で `viz/*.png` を見る

### 4.8 監査

`fleet confirm`（5.4）は、確認の batch の CPU の run から 1 本を選び、同じ `job.json`（`phase=audit`、`avoid_host` = 元のホスト、`after` = 元の run）を投げる。`fleet table` は監査の `metrics_sha256` が元と違えば `nondeterministic` と表の下に出す。手で回すときは `fleet reproduce <run>`。

### 4.9 枠の較正（最初に 1 回、ホストの種類が変わったら再び）

`fleet bench` は `m14/bench`（`task=bench`: 段 0 の小・大 × PC・BP × トレースあり / なしを仮想時間 120 秒）を、1 台の上で同時に 1/4/8/14/16/24/30 本（`smt=false` と `true`）回し、1 本の frames/s と総量を表にする。GPU（4090・4060）でも 1/2/4/6/8 本を測る。総量が頭打ちになる本数を `fleet.toml` に書き、preset の既定の `res`（CPU か GPU か、RAM、GPU のメモリ）を決める。測る前の目安は「1 台 14 本の CPU の run + GPU の run 数本、4 台で同時 60〜80 本」。

### 4.10 失敗の表

| 起きたこと | 検出 | 扱い |
| -- | -- | -- |
| run の例外 | exit 1 | failed。再試行しない。同じ印が 3 回で batch を保留 |
| 発散 | exit 3 | 結果。表で「発散」 |
| 環境の誤り | exit 4 | そのホストのその資源の種類を cordon、キューへ戻す |
| OOM killer | SIGKILL、停止の要求なし | RAM を 1.5 倍にして 1 回だけ再試行 |
| CUDA OOM | traceback（exit 1） | `fleet retry --gpu-mem G` |
| 止まった | frame が 20 分進まない | SIGUSR1 → 停止 → 1 回だけ再試行 |
| worker・コンテナ・ホストの再起動 | pid が死に exit が無い | 再開用の checkpoint から同じホストで（4.6） |
| ホストが戻らない | 心拍が古い | `fleet requeue --host`。戻ったら fencing |
| NAS が一時的に止まる | EIO | worker は再試行。run はローカルに書いているので落ちない。live と保管が遅れるだけ |
| 契約違反（exit 0 で manifest が無い） | worker | failed として出す |
| manifest の不一致 | NAS の検証 | 写し直し、3 回で corrupt |

---

## 5. optuna（`tools/sweep/`）

### 5.1 置き場

- 調整役は **std-00 のコンテナに常駐**する 1 study 1 プロセス: `UV_PROJECT_ENVIRONMENT=<root>/venv/sweep-<lockhash> uv run --project tools/sweep --python 3.14 python -m sweep run exp/video/studies/<study>.toml`（cwd は study の `code/<snap12>`）。学習の venv とは別なので、optuna・numpy・sqlalchemy は学習のプロセスに入らない
- storage は std-00 のローカルの sqlite（`<root>/optuna/<study>.db`）。書くのは調整役だけ。1 時間ごとに `sqlite3` の backup API で NAS の `studies/<study>/` へ写す。`studies/<study>/RUNNING` があれば `fleet up` が調整役を起こし直す
- 調整役は job を `exp.lib.fleet.queue` で直接キューに書き、`exp.video.compose` で override を解決する（同じ snapshot の中）。途中経過は NAS の `live/` と `done/`・`runs/` から読む
- optuna-dashboard は依存に入れない。見たいときだけ `just fleet x std-00 -- uvx optuna-dashboard sqlite:///...` と `ssh -L`

### 5.2 目的と途中の報告

- M14 の目的: $J = \frac{1}{|S|}\sum_{s}\frac12\sum_{h\in\{5,10\}}\log\frac{E^{\mathrm{val}}_{h}}{\mathrm{CV}^{\mathrm{val}}_{h}}$（最小化。0 で等速の外挿と同じ、$\log 0.5=-0.69$ が完了の目安）。run が `evals.jsonl` と `result.json` に書いた値をそのまま使う
- 発散した seed は $J_s=\log 10$（等速の外挿の 10 倍）として平均に入れ、**値として tell** する。TPE が発散する領域を学ぶ。FAIL は例外（exit 1）のときだけで、そのときは batch の保留も効く
- 途中の報告: 60 秒ごとに `live/` の `evals.jsonl` を読み、**trial の全 seed がその時刻に達したときだけ** seed の平均を `trial.report(value, step=int(t))`
- pruner: `SuccessiveHalvingPruner(min_resource=300, reduction_factor=3)`（仮想時間の秒。段は 300・900 秒）。`should_prune()` なら全 seed の claim に `.stop --now`（`pruned`）を置き、`tell(PRUNED)`
- sampler: `TPESampler(multivariate=True, group=True, constant_liar=True, n_startup_trials=16, seed=0)`。`[[enqueue]]` で M13 の最良から始める
- **study は 1 つの snapshot に縛る**。コードを変えたら `-v2` の新しい study にする

### 5.3 探索の長さ

探索の run は 1800 秒が既定で、ASHA が悪い trial を 300 秒と 900 秒で止める。最初の段 0 の格子（E1）で、600 秒の時点の順位と 1800 秒の順位の Spearman の相関を測る。0.8 以上なら段 1・2 の探索の preset を 600 秒にし、その判断と数字を README に書く（回す前に）。

### 5.4 study の定義と、BP で探索して PC-ALM と BP で確認する

```toml
# exp/video/studies/m14_s1_big_trace_bp.toml
[study]
name = "m14-s1-big-trace-bp-v1"
overrides = ["+experiment=m14/s1", "learner=bp", "dynamics=big", "dynamics.trace=true", "phase=explore"]
seeds = [100, 101]
parallel = 32           # trials in flight (x seeds runs)
max_trials = 160
res = "any_gpu"

[space]
"optim"             = {categorical = ["sgd", "adamw", "graft_adam"]}
"optim.lr"          = {float = [0.001, 0.3], log = true, when = "optim != 'adamw'"}
"optim.lr@adamw"    = {float = [1e-5, 1e-2], log = true, when = "optim == 'adamw'"}
"optim.tau_m"       = {float = [0.03, 3.0], log = true, when = "optim != 'sgd'"}
"optim.warmup"      = {categorical = [0.0, 60.0, 300.0]}
"dynamics.residual" = {categorical = [true, false]}
"dynamics.open_loop" = {categorical = [0.0, 0.25, 0.5]}

[[enqueue]]
"optim" = "sgd"
"optim.lr" = 0.03
```

固定: B=128、`learn_interval=25e-3`、物理、モデルの大きさとトレース（小 / 大 × あり / なしは別の study。大はトレースありで探索し、なしは同じ値で回して README に書く）。M14 にはエンコーダも SIGReg も無いので、M10〜M13 で Adam 系が負けた理由が消えているかもしれない。optimizer は比べ直す。

```text
sweep run     exp/video/studies/m14_s1_big_trace_bp.toml
sweep confirm m14-s1-big-trace-bp-v1 --top 3 --learners bp,pc --seeds 0,1,2 --res 4090 --batch m14-s1-confirm-v1
              # 3 x 2 x 3 = 18 runs, phase=confirm, clean snapshot only, + 1 audit run (4.8)
sweep bracket m14-s1-big-trace-bp-v1 --best --param optim.lr --factors 0.5,1,2 --learners bp,pc --seeds 0,1,2
```

- 上位の選び方は探索の seed の目的値。README の数字は確認の seed（0/1/2）と test のクリップだけから取る
- `bracket` で PC-ALM の最良が ×1 でなければ README に書き、両方とも bracket の最良で比べ直す
- 判定は README の規約どおり、3 seed のすべてで同じ向きのときだけ支持
- 因子が少ないとき（段 0 の最初の比較）は optuna を使わず `fleet submit --sweep` の格子を投げる
- 確認の batch は優先度を上げて先に置く（PC-ALM + トレースは 5〜7 時間かかり、batch の終わりを決める）

---

## 6. 追跡と集計

- trackio・MLflow・W&B は入れない（3.14t に入らないか、サーバーと学習のプロセスからの通信が要る。読むのは AI の orchestrator なので、表と PNG で足りる）
- `exp/lib/runs.py`（標準ライブラリだけ）:

```python
@dataclass(frozen=True)
class RunRecord:
    dir: Path; job: dict[str, Any]; result: dict[str, Any] | None
    def evals(self, split: str = "val") -> list[dict[str, Any]]
    def at(self, when: str = "last", split: str = "test") -> dict[str, Any] | None   # "last" | "t=600"
def index(root: Path, *, batch: str | None = None) -> list[RunRecord]
def where(runs: Sequence[RunRecord], *conds: str) -> list[RunRecord]   # "config.learner.kind=pc", "tags~=confirm"
def table(runs, *, rows: Sequence[str], cols: Sequence[str], value: str, split: str = "test",
          at: str = "last", digits: int = 2, allow_mixed: bool = False) -> str
    # markdown. A cell lists seeds 0/1/2 as "a / b / c"; "発散" for diverged, "打切" for pruned,
    # "欠" for a missing seed (never averaged away silently); "*" for a dirty snapshot.
    # Refuses (unless allow_mixed) a cell that mixes snapshots, mujou versions or device classes,
    # or a batch with runs not yet done. A footnote: snapshot, mujou SHA, device class, count, the command.
def series(run: RunRecord, key: str, split: str = "val") -> list[tuple[float, float]]
def export(runs, keys: Sequence[str]) -> dict[str, Any]   # JSON for report.html
```

- `just fleet table --batch m14-s1-confirm-v1 --rows config.dynamics.d_s,config.dynamics.trace --cols config.learner.kind --value roll10/cv10` の出力を `results/m14.md` にそのまま貼る
- 見た目の確認の PNG は run 自身が `viz/` に書く（torch と `exp.lib.png` だけ）。report.html の図は `runs.export` の JSON から skill `experiment-report` の手順で描く。matplotlib は依存に入れない

---

## 7. M14 のコード（シグネチャ）とオラクル

### 7.1 物理のベクトル化（`data/physics.py`）

今の `balls.Video` は純 Python で、B=128・30 分のストリームの物理だけで約 2 時間かかる。torch の float64 で `(B, K)` をまとめて進める。

```python
@dataclass(frozen=True, kw_only=True)
class Physics: gamma0, gamma_amp, e0, e_amp, drag, period, conserve   # as now; gamma(t), restitution(t)

class BallState(NamedTuple):
    pos: Tensor   # (B, K, 2) float64
    vel: Tensor   # (B, K, 2) float64, per second

def draw(n: int, radii: Sequence[float], gen: torch.Generator) -> BallState
def step_(s: BallState, radii: Tensor, h: float, gamma: float, e: float, drag: float, conserve: bool) -> None
def advance_(s: BallState, radii: Tensor, physics: Physics, t: float, seconds: float, substeps: int) -> None

class BallStreams:
    """B streams that never redraw; stream i starts as Video(seed=seed + STRIDE * i) does."""
    def __init__(self, n: int, *, radii, physics, frame: float, substeps: int, seed: int) -> None
    def next(self) -> Tensor                 # (B, 4K) float32: x, y, vx, vy per ball
    def fork(self) -> "BallStreams"          # a deep copy: the same future
    def state_dict(self) -> dict[str, Tensor]; def load_state_dict(self, d: dict[str, Tensor]) -> None

def render(states: Tensor, size: int, radii: Sequence[float]) -> Tensor   # (n, 1, size, size), as now
```

全ストリームが同じ時刻を共有するので、重力と反発係数は substep ごとのスカラー。玉ごと・軸ごとの壁の判定は `torch.where`。

### 7.2 観測（`data/observe.py`）

```python
def observe(states: Tensor, obs: str, balls: int) -> Tensor      # (..., 4K) -> pos_vel (..., 4K) | pos (..., 2K)
def to_states(obs_raw: Tensor, obs: str, balls: int) -> Tensor   # back to (..., 4K), zero velocities for pos

@dataclass
class Standardizer:
    mean: Tensor; std: Tensor                                     # (d,)
    @classmethod
    def fit(cls, x: Tensor) -> "Standardizer"                     # x (n, T, d), population variance
    def apply(self, x: Tensor) -> Tensor; def invert(self, y: Tensor) -> Tensor
```

標準化は専用の**較正のクリップ**（seed 777、256 本）で決め、`baselines.json` に残す。val と test のクリップでは決めない。

### 7.3 評価クリップ（`data/clips.py`）

```python
SPLIT_SEEDS = {"calib": 777, "val": 1001, "test": 12345}
@dataclass(frozen=True)
class ClipSet:
    states: Tensor    # (n_clips, L, 4K) true states
    starts: Tensor    # (n_windows, 2): (clip, first frame)
def make_clips(balls: BallsConf, *, split: str, clips: int = 256, length: int = 128, observed: int = 16,
               horizon: int = 20, stride: int = 30, frame: float) -> ClipSet     # 4 windows per clip: 1024
def windows(c: ClipSet, obs: str, std: Standardizer) -> tuple[Tensor, Tensor]
    # (n_windows, observed + horizon, d) standardized observations, and the true states
def contact_mask(states: Tensor, observed: int, horizons: Sequence[int]) -> Tensor
    # (n_windows, len(horizons), K) bool: ball k touches a wall between frame observed-1 and observed-1+h
```

### 7.4 物差し（`measure/baselines.py`）

すべて**生の観測の空間で予測してから標準化**する。

```python
def mean_pred(n: int, horizon: int, d: int) -> Tensor                        # zeros in the standardized space
def copy_pred(obs: Tensor, observed: int, horizon: int) -> Tensor
def const_velocity(raw: Tensor, obs: str, balls: int, observed: int, horizon: int, frame: float) -> Tensor
    # p_h = p_last + h (p_last - p_prev); for pos_vel the velocity columns are (p_last - p_prev) / frame
def const_velocity_observed(raw: Tensor, balls: int, observed: int, horizon: int, frame: float) -> Tensor
    # stage 0 only: moves with the observed velocity (a stronger baseline that uses the whole observation)
def const_accel(raw: Tensor, obs: str, balls: int, observed: int, horizon: int, frame: float) -> Tensor
    # p_h = p_last + h v + h (h + 1) / 2 a, v = p_last - p_prev, a = p_last - 2 p_prev + p_prev2
def true_physics(states: Tensor, balls: BallsConf, observed: int, horizon: int, frame: float) -> Tensor
def errors(pred: Tensor, target: Tensor, horizons: Sequence[int], mask: Tensor | None = None) -> dict[int, float]
def baselines(clips: ClipSet, balls: BallsConf, std: Standardizer, m: MeasureConf) -> dict[str, float]
    # mean{h}, copy{h}, cv{h}, cvobs{h}, ca{h}, true{h}, and @contact / @free / @ball{i} variants
```

### 7.5 ノードと組み立て（`dyn/`）

```python
# dyn/modules.py
class DynamicsStack(nn.Module): first: MinGRU; rest: nn.ModuleList   # (x_{k-1}, s_{k-1}) -> s_k, as now
class Top(nn.Module): ...
@dataclass
class DynamicsModules: step: DynamicsStack; readout: nn.Linear
def build_dynamics(conf: DynamicsConf, d_in: int, d_out: int, *, seed: int) -> DynamicsModules
    # torch.manual_seed(seed), then first, rest, readout in this order (zero readout when residual);
    # built on CPU then .to(device): the one place that draws initial weights, for PC, BP and offline

# dyn/nodes.py
class Source(Node):
    """Shows the host's samples downstream and takes no credit back.
    prev = Port(recv=Sample); out = Port(send=act_spec) (send-only: any number of recv-only peers)."""
    def __init__(self, *, x: TensorSpec, period: float, batch: Size, device: Device) -> None
    act_spec: DictSpec[Activity]

class Dynamics(Node):       # s_k = minGRU(x_{k-1}, s_{k-1}); z = Port(recv=act_spec) (no send spec)
class DynamicsLayer(Node):  # as now (same frame, sends g_in to the previous layer)
def compare_energy(readout: nn.Module, s: Tensor, target: Tensor, base: Tensor | None = None) -> Tensor
    # (*batch,): mean over the last axis of (readout(s) [+ base] - target)^2; no kappa
class Compare(Node):        # target = Port(recv=target_spec) (no send spec); s sends credit; next, error

class DynamicsBlock(mujou.Network):
    """Dynamics -> DynamicsLayer x (nodes-1) -> Compare on an upstream send-only port."""
    def __init__(self, mods: DynamicsModules, *, act_spec, conf: DynamicsConf, tau, optimizer,
                 learn_interval, device) -> None
    def connect(self, upstream: Port[Activity, None]) -> None    # M14: Source.out / M16: JEPAHead.z_out

# dyn/bp.py
class DynamicsBP(Node):
    """BP in one node with the same memory (x_{k-1}, s_{k-1} detached), trace, open loop, residual
    and LearnClock as DynamicsBlock; prev = Port(recv=act_spec)."""

# dyn/nets.py
class StateWorldPC(mujou.Network): source: Source; block: DynamicsBlock
class StateWorldBP(mujou.Network): source: Source; world: DynamicsBP
def build_world(conf: RunConf, d_obs: int) -> StateWorldPC | StateWorldBP
def modules(net) -> DynamicsModules            # the live modules, read between steps (no copy)
def live_states(net) -> Tensor                 # (B, d_state): s_k of each stream, for fork_eval
def make_optimizer(c: OptimConf, frame: float) -> pc.optim.Optimizer   # Warmup(SGD | AdamW | Grafted(AdamDirection))
```

`Compare` から `kappa` を外す（SIGReg の無い M14 で `(1-κ)` が付くと学習率の意味がずれる）。`stop_target` と `target_weight` も消す。credit を止めるのは接続の型だけ。

### 7.6 評価・ループ・見た目（`measure/dynamics.py`、`programs/dynamics.py`、`measure/viz.py`）

```python
def rollout(mods: DynamicsModules, obs: Tensor, *, residual: bool, observed: int, horizon: int) -> Tensor
    # teacher-forced for k < observed, then open loop from its own predictions: (n, horizon, d)
def evaluate_dynamics(mods, clips: ClipSet, std, base: dict[str, float], m: MeasureConf, *, obs, balls,
                      residual) -> dict[str, float]         # the keys of 3.7 except fork*
def fork_eval(net, streams: BallStreams, std, m: MeasureConf, *, obs, balls, residual) -> dict[str, float]
    # s = live_states(net); the future = streams.fork() advanced horizon frames; open loop from s

class DynamicsProgram(Program[RunConf]): ...   # task online | baselines | bench | offline (dyn/offline.py)

def frames_png(path: Path, true_states: Tensor, pred_states: Tensor, cv_states: Tensor, radii, *,
               size: int = 32, columns=(0, 1, 2, 5, 10, 20), clips: int = 4, scale: int = 3) -> None
    # rows per clip: true / prediction rendered / constant velocity rendered
def trajectory_png(path: Path, true_xy: Tensor, pred_xy: Tensor, cv_xy: Tensor, *, observed: int,
                   canvas: int = 256, clips: int = 4) -> None
    # the box, observed path (grey), true future (green), prediction (red), CV (blue)
```

- 前向きの予測の誤差（prequential、そのフレームを学ぶ前の予測）は device の上に貯め、`log_every` ごとに 1 回だけ `.item()` する
- 評価（60 秒ごとに val、checkpoint のときに test も）は step の合間にモジュールを直接読む（deepcopy しない）
- 見た目の確認: checkpoint のたびに固定の 4 窓を描き、**接触を含む窓を必ず 2 つ入れる**（`contact_mask` で選ぶ）。数字を README に書く前に、orchestrator が `fleet pull` した PNG を Read で見て「正しい向きへ動く」「壁で跳ね返る（すり抜けない・止まらない）」「薄れない」を確かめ、見たファイル名を results に書く

### 7.7 オフラインの上限（`dyn/offline.py`）

```python
def stream_windows(conf: RunConf, frames: int, window: int = 32) -> Tensor    # (n, window, d) from BallStreams
def fit_offline(mods: DynamicsModules, windows: Tensor, *, steps: int, batch: int, lr: float, local: bool,
                observed: int, gen: torch.Generator, every: int,
                on_eval: Callable[[int, DynamicsModules], None]) -> None
    # teacher forcing then open loop; BPTT through all, or with local=True the input and the fed-back
    # prediction detached (one frame, as the nodes); Adam as pc.optim.AdamW via exp.lib.optim.adam
```

`torch.optim` を使わないので、オフラインの run も同じ GIL の検査と同じ `evaluate_dynamics` を通る（`t` の代わりに `step`）。2×2 の表（BPTT / 1 フレームで切る × オフライン、トレース / なし × オンライン）で、時間の局所性とオンラインの標本効率の損を分けて README に書く。

### 7.8 段 3（画像を観測に、`dyn/pixels.py`）

段 1・2 が通ってから決める（MILESTONES）。下準備: 観測は `render(states)` を平らにした `(B, 1024)`、標準化は全画素で 1 つの平均と標準偏差。物差しは写し・平均の画像・等速と等加速度の外挿を描いた画像、玉ごとの領域の誤差（小さい玉の予測が消えることを数で拾う）。

### 7.9 オラクル

| テスト | 理論から言えること | 許容 |
| -- | -- | -- |
| `data/test_physics.py` 参照 | ストリーム i が参照の `Video(seed=seed+STRIDE*i)` と同じ状態（玉 1・3、conserve の有無、時間変化あり、2000 フレーム） | `torch.equal` |
| 同 エネルギー | `conserve` で $\tilde E$ が跳ね返りをまたいで保たれる（slow） | 今と同じ |
| 同 分岐・往復 | `fork()` の未来 = 元の未来。`state_dict` から戻したストリーム = 止めずに進めたもの | 完全一致 |
| `data/test_observe.py` | 段 0 を標準化から戻すと状態。段 1・2 は位置の列。較正の集合で平均 0・分散 1 | 1e-6 |
| `data/test_clips.py` | 床へ落とした玉で、接触のフラグが立つ最初の $h$ が手計算と一致。壁から遠い玉は全て False | 完全一致 |
| `measure/test_baselines.py` 等速 | 重力 0・壁に当たらない区間で誤差 0 | 1e-6 |
| 同 等加速度 | 一定の重力・抵抗 0・壁なしでは、半陰的 Euler の位置が刻みの番号の 2 次式なので、フレームで間引いても 3 点の外挿が厳密 | 1e-6 |
| 同 真の物理 / 平均 | `true_physics` の誤差 0。標準化した空間で平均の物差しの誤差 1 | 0 / 1e-6 |
| `dyn/test_nodes.py` credit なし | `Source.out` に send を宣言したポートを connect すると `TypeError`。`Dynamics.z`・`Compare.target` は send の spec を持たない | — |
| 同 Compare | `compare_energy` の `s` 側の勾配が中心差分と一致（float64） | 1e-6 |
| 同 Dynamics | 1 フレームずつの `mu` が逐次の minGRU と一致 | 1e-6 |
| 同 局所性 | 数フレーム後、どのパラメータにも `.grad` が無く、重みは各ノードの optimizer だけで動く | — |
| `dyn/test_nets.py` 平衡点 | 1 フレームを長く保持し `learn_interval=inf` にすると、各ノードの `lam` = `DynamicsBP` の同じ損失の $-\partial L/\partial h$（小・大、B=1 と 3） | rtol 1e-3 |
| 同 初期値 | `StateWorldPC` と `StateWorldBP` が同じ seed で同じ重み | 完全一致 |
| 同 トレース | 重みを止めたとき、トレースの勾配 = 入力を固定した BPTT（PC と BP） | 今と同じ |
| 同 残差の初期 | `residual=True` の初期（$W_o=0$）の rollout の誤差 = 写しの誤差 | 1e-7 |
| `dyn/test_offline.py` | 窓がストリームの最初をまたがない。`local=True` の勾配 = 1 フレームで切った損失の有限差分。同じ重みでオフラインの評価 = オンラインの評価 | 1e-4 |
| `measure/test_dynamics.py` | 恒等の動力学なら `roll{h}` = 写し、`fork{h}` = 写し | 1e-6 |
| `programs/test_dynamics.py` 契約 | 小さい job（B=2、10 秒）が 3.2 の全ファイルを書き、manifest の sha256 が合う | — |
| 同 決定性 | 同じ `job.json` を 2 回回すと `evals.jsonl`・`stream.jsonl` がバイト一致（CPU） | 完全一致 |
| 同 再開 | `test_resume_equals_uninterrupted[pc,bp][trace]` | 完全一致 |
| 同 prequential | 記録する予測はそのフレームを学ぶ前のもの（学習率を変えてもフレーム $k$ の記録は変わらない） | 完全一致 |
| 同 終了コード | 大きすぎる学習率で exit 3 と `diverged`。`stop.request` で exit 75 と再開用の checkpoint | — |
| 同 ctl | `ctl` を変えても指標が一致（`group` から外してよい根拠） | 完全一致 |
| `test_gil.py` | `main` の subprocess で、import・build・評価の後も GIL が無効。`torch.optim.SGD` を作ると `assert_free_threaded` が例外 | — |

「PC-ALM の重みの軌跡が BP と K フレームで近い」は許容誤差を理論から出せないので、テストではなく較正の run として README に書く。

---

## 8. M15 と M16

### 8.1 フレームのストリームと N フレームのまとまり

```python
# data/frames.py
class FrameStreams(Protocol):
    def next(self) -> tuple[Tensor, Tensor, Tensor]   # images (B, C, H, W) float, states (B, S) or (B, 0), reset (B,)
    def state_dict(self) -> dict[str, Any]; def load_state_dict(self, d: dict[str, Any]) -> None
class BallFrames(FrameStreams)      # BallStreams + render
class TourFrames(FrameStreams)      # B readers of the local u8 copy, stream i from offset i * N / B

# jepa/nodes.py
class FrameRing(mujou.Stateful):    # (B, N, C, H, W) ring and the count since reset; push_, ready, chunk
class Chunker(Node):
    """Parameter-free. Keeps the last N frames (a FrameRing); on a new frame draws the masks from its
    generator and sends V views per stream as samples: view 0 the target (no mask), views 1.. the context
    (masked tubes zeroed, the mask as an extra channel). mask=False until N frames since the reset."""
    prev: Port[None, Sample]; next: MultiPort[Activity, Tensor]   # h: (B * V, N, C, H, W)
```

- 時間はサンプルの次元。まとまりは毎フレーム 1 つずれる（フレーム $k$ のまとまりは $k-N+1..k$）
- マスクはフレームの最初の更新で 1 回だけ引く。PC 版と BP 版は同じ `Chunker` を使うので、同じ seed で同じマスク
- $N>1$ ではマスクを時間方向に伸ばした筒にする（隣のフレームから写す近道を塞ぐ）

### 8.2 JEPA のノード（PC-ALM 版と BP 版）

```text
host ─Sample▶ Chunker ─(B·V samples)▶ Enc1 ─▶ … ─▶ EncL ─▶ JEPAHead ─ z_out (send-only) ─▶ [M16 の DynamicsBlock]
              (no params)              pc.Layer: blocks of the ViT        predictor + SIGReg
```

- エンコーダ `Enc1..L` は ViT のブロックを数個ずつ束ねた `pc.Layer`（サンプルごとに独立: attention はサンプルの中、LayerNorm は可、Dropout は不可）。ノードは 2〜4 個（1 周期 1.5〜2 ms の起動の律速と、credit が 1 周期に 1 ノードしか戻らないため）
- `JEPAHead`: `(B·V, T, d)` を `(B, V, T, d)` に戻し、予測器が文脈のビューのトークンからマスクの位置の目標のトークンを予測する。$L = (1-\lambda) L_{\mathrm{pred}} + \lambda\,\mathrm{SIGReg}$（SIGReg は目標のビューの埋め込みと、下流へ送る $z$ に）。自分の中の autograd で各サンプルの埋め込みへの勾配を `prev` に返す（`# mujou: local-autograd`）。サンプルをまたぐ損失はこのノードの中に閉じ、1/B の規約は今の `SIGReg` ノードと同じにする
- `z_out = Port(send=act_spec)` で目標のビューの $z_k$（`(B, D)`。CLS / 平均のプーリング / 少数のレジスタは probe で選ぶ）を送る
- 目標のエンコーダの重みを別に持たない（EMA なし、LeJEPA の形。重みの書き手は 1 ノード）。stop-gradient の有無は head の中の切り替え（`vision.sg`）で、両方を比べる
- `JEPABP`（`jepa/bp.py`）: `Chunker` → 1 ノードに同じ ViT・予測器・損失。同じ `build_vision(conf, seed)` で同じ初期値。探索は BP で、確認は PC-ALM と BP の両方
- optimizer は `pc.optim`（`torch.optim` は使わない）

### 8.3 Walking Tours

```python
# exp/lib/video_io.py
def ffmpeg_frames(path: Path, *, fps: float, width: int, height: int, start: float = 0.0,
                  duration: float | None = None, threads: int = 4) -> Iterator[Tensor]   # u8 (3, H, W)
class U8Writer: ...; class U8Reader: ...      # memmap of (N, 3, H, W) uint8 + index.json

# exp/video/data/walking_tours.py
MANIFEST: Path    # per video: the license, the blur of faces and plates, the source URL
def prepare(video: str, src: Path, out: Path, *, fps: float = 10.0, size: int = 64) -> None   # task=prep
def data_version(index: Path) -> str          # sha256 of index.json, 12 hex
def eval_clips(root: Path, *, held_out: str, clips: int, length: int, seed: int) -> Tensor
```

- 4K・60 fps → 10 fps・64×64（中央の正方形）から始める。1 本 1 job（`task=prep`、CPU 8 枠）。出力は NAS の `data/wt/<data_version>/`（`.tmp` に書いて rename）
- run は**ローカルの複製だけを読む**: worker が job の `data_version` を `.fleet/data/<data_version>/` へ写して manifest を検証してから起動する（soft の CIFS から 30 分読み続けない）
- **評価には学習に流さない 1 本を取っておく**（`ToursConf.held_out`）
- マニフェストに無い動画は `prepare` が拒む（ライセンスとぼかしの確認を強制）。テストは ffmpeg で作った色の合成の mp4 だけ

### 8.4 M15 の評価と、M16 への受け渡しの条件（`measure/vision.py`）

```python
def ball_probe(z: Tensor, states: Tensor, balls: int, half: Tensor) -> dict[str, float]   # R^2 of x, y (vx, vy) per ball
def thumbnail_probe(z: Tensor, images: Tensor, half: Tensor, size: int = 8) -> float     # R^2 of the 8x8 grey thumbnail
def vision_metrics(encode: Callable[[Tensor], Tensor], clips) -> dict[str, float]         # rank, z_std_min, probes
```

- 合成の玉: 玉ごとの位置の probe の $R^2$（$N \ge 2$ なら速度も）、実効ランク、次元ごとの標準偏差の最小（崩壊の検知）。Walking Tours: 縮小画像の probe、実効ランク、後から学ぶデコーダの再構成
- 基準: 学習していない同じ ViT と、生の画素からの probe。これを超えなければ学んだとは言わない
- **受け渡しの条件**（目安。回す前に README で確定）: 3 つの玉すべての位置が $R^2 \ge 0.9$（test、3 seed、BP と PC-ALM）、デコーダの再構成で 3 つの玉が見える
- 危険（M13 の再来）: マスクの予測は文脈から当てられない内容を捨てる向きに働き、SIGReg は 1 次元の射影の分布しか見ないので止めない。sg のあり・なし、マスクの比率、トークンへの SIGReg を比べ、玉ごとの probe で判定する。どれも通らなければ、別の視覚の損失を検討する（M16 で動力学から credit を戻す案は出さない）

### 8.5 後から学ぶデコーダ（`measure/posthoc.py`、`task=posthoc`）

```python
def decoder(d_z: int, shape: tuple[int, int, int]) -> nn.Module
def fit_decoder(z: Tensor, images: Tensor, *, steps: int = 2000, batch: int = 256, lr: float = 1e-3,
                gen: torch.Generator) -> nn.Module                  # exp.lib.optim.adam, no torch.optim
def decode_run(run_dir: Path, *, frames: int = 16384, device: str) -> None
    # per ckpt: rebuild the vision net, load the modules, encode training frames sampled uniformly over
    # the whole stream, fit, decode the eval clips; post/decode/<t>.png and post/decode/metrics.jsonl
    # (recon, img_roll{h}, pix_copy{h}, pix_mean{h}; pix_mean is the mean image of the eval clips)
```

`fleet submit` は M15・M16 の確認の run の後ろに、`after` と `host`（親の run と同じホスト。ckpt がローカルにあるうちに）を付けた posthoc の job を自動で積む。

### 8.6 M16（`programs/joint.py`）

```python
class JointPC(mujou.Network): chunker; enc1..encL; head: JEPAHead; block: DynamicsBlock   # block.connect(head.z_out)
class JointBP(mujou.Network): chunker; vision: JEPABP; world: DynamicsBP                  # world reads vision.z_out
def build_joint(conf: RunConf) -> JointPC | JointBP   # build_vision(seed), then build_dynamics(seed + 1)
```

- 動力学は M14 の `DynamicsBlock` を上流だけ替えて使う。上流のポートが送るだけなので、credit がエンコーダへ戻る経路が無い
- $z$ の尺度: SIGReg が $z$ の射影をおよそ標準正規にするので、M14 の標準化した観測に近い（M14 の学習率を持ち込みやすい）。固定の標準化は入れない
- 対照: **凍結したエンコーダ**（M15 の最後の ckpt を読み、エンコーダのノードを `learn_interval=inf`）+ 動力学。M14 の状態の観測の結果と、M14 段 3 の結果も同じ表に並べる
- 判定: 潜在で `roll{h}` を写し・平均と比べ、デコーダで画像に戻して `img_roll{h}/pix_mean{h}` と写しと比べる（5・10・20 フレーム先）。まとまりは隣と $N-1$ フレームを共有し潜在の写しが非常に強いので、画像の空間の判定を主にする

### 8.7 M15・M16 のオラクル

| テスト | 理論から言えること |
| -- | -- |
| `jepa/test_masks.py` | マスクの比率・ブロックの数、筒が時間方向に一定、同じ seed で同じ |
| `jepa/test_nodes.py` 平衡点 | 学習を止めて 1 フレームを保持すると、各エンコーダのノードの `lam` = `JEPABP` の同じ損失の credit（サンプルごと、ビューごと） |
| 同 ビューの独立 | エンコーダのノードの出力は他のサンプルの入力に依らない |
| 同 sg | stop-gradient ありで、目標のビューが受ける予測の項の credit は 0 |
| 同 FrameRing | 再開でリングが戻る（`test_resume_equals_uninterrupted` の vision 版） |
| `jepa/test_bp.py` 初期値 | `VisionPC` と `VisionBP` が同じ seed で同じ重み |
| `programs/test_joint.py` **credit なし** | 同じ seed・同じデータで K フレーム回したとき、`JointPC` のエンコーダの重みと `VisionPC` の重みがビット単位で同じ（CPU）。BP 版も同じ |
| 同 動力学は M14 のもの | 上流に決まった z の列を流すと、`DynamicsBlock` の振る舞いが M14 に同じ列を観測として流したものと一致 |
| `measure/test_vision.py` | 状態の線形写像を $z$ にすると probe の $R^2$ = 1、乱数の $z$ なら約 0 |
| `measure/test_posthoc.py` | $z$ が画像を決める合成の対で、デコーダが学べる |

mujou に送る候補（必要が出たときだけ、skill `mujou-upstream`）: `pc.Source`（送るだけのポートの観測のノード。M14 と M16 で同じ形が 2 回要れば）、多入力でサンプルをまたぐ損失のノードの一般形（M15 で形が定まってから）。M14 では mujou を変えない。

---

## 9. 作業の包み（順序つき）

規則:

- 並べる包みは、担当するファイルが重ならない。`schema/` と `conf/` の群の yaml は WP1 が全部書く。WP1 の後の持ち主は、`schema/{run,data,dyn,learn,measure}.py` と `conf/{dynamics,learner,optim,phase}/` が WP8、`schema/vision.py` と `conf/vision/` が WP10。段 B の間はだれも schema を変えない（要るときは WP1 の持ち主に頼み、1 commit で入れてから並べ直す）
- 完了の条件はどれも `just run`（format・test・type）が通ることを含む（下では省く）。テストは dev container の中で回す（Mac では `just container just run`、ホストでは `just fleet x std-00 -- just test-fast`）。CUDA の試験は `@pytest.mark.cuda`
- ブランチと PR: WP0〜WP3・WP6・WP7 を `refactor/2026-11-03/fleet-and-hydra`（PR 1 本）、WP4・WP5・WP8・WP13 と M14 の結果を `exp/<日付>/continuous-stream-m14-dynamics`、WP9〜WP11 と M15 を `exp/<日付>/continuous-stream-m15-jepa`、WP14 と M16 を `exp/<日付>/continuous-stream-m16-joint`。後の枝は前の枝から切り、`/merge-main` で追う
- 1 つの包みの中は skill `solo-dev-cycle` の 5 段階（計画 → テスト → 実装 → リファクタ → 文書）

### 段 A（直列）

**WP0 ホストの準備と事実の確認**（orchestrator）

- 目的: コンテナと NAS をフリートに使える状態にし、設計の前提を実機で確かめる
- 担当: `compose.yaml`、`.env.example`、`.gitignore`（`.fleet/`、`.fleet-nas/`）、`pyproject.toml`・`uv.lock`（`uv add hydra-core==1.3.8`、pyright と ruff の exclude に `.fleet`）、`memory/agents/orchestrator/wp0-facts.md`
- 依存: なし
- 手順: 4.3 の作り直しの手順（先に全ホストで走っている run が無いことを確かめる）。Mac に Homebrew の rsync、`~/.ssh/config` に ControlMaster
- 完了の条件（結果を `wp0-facts.md` に書く）:
  - コンテナの中で vscode が `/nas/geson/mujou-exp` に書ける。2 台から同じファイルを同時に `os.rename` で取り合い、片方だけが成功する（100 回）
  - コンテナの 3.14t で `import hydra, omegaconf, yaml` の後と、`torch._C._set_deterministic_algorithms(True)` の後に `sys._is_gil_enabled()` が False
  - `UV_PROJECT_ENVIRONMENT=/tmp/venv-x uv sync --frozen --no-editable` で mujou が wheel として入り、`cwd` から `exp` が import できる
  - `/sys/devices/system/cpu/cpu*/topology/thread_siblings_list` で物理コアと SMT の兄弟の対応が分かる
  - `docker inspect` で `RestartPolicy=unless-stopped`、`nvidia-smi` で 2 枚

**WP1 骨組みと設定**（1 人）

- 目的: 旧コードを消し、新しい木・schema・compose・preset の形を固める（以後の包みの境界）
- 担当: 1.3 の削除（1 commit）、`exp/video/**/__init__.py`、`exp/video/schema/*.py`（全ファイル）、`exp/video/compose.py`、`exp/video/conf/**`（`experiment/` を除く群の yaml と `run.yaml`、`experiment/m14/s0.yaml` だけ）、`tests/video/schema/test_presets.py`、`tests/video/test_compose.py`、`tests/conftest.py`（`cuda(n)` の skip）、`tests/video/data/reference_balls.py`（旧 `_step` と `Video` を移す）
- 依存: WP0（`hydra-core` が入っていること）
- 完了の条件: `uv run pytest tests/video/schema tests/video/test_compose.py` が通る（`conf/**/*.yaml` を glob して全 preset が compose でき、`to_object` が `RunConf` を返し、不正な `optim.kind=foo` が `__post_init__` で落ちる、`optim=adamw` と `optim.kind=adamw` の override が通る）。`just run` が通る（旧テストは消えている）

### 段 B（並列: WP2・WP3・WP4・WP5・WP6・WP9。担当が重ならない）

**WP2 run の契約と lib**

- 目的: 3 節の契約を 1 か所に置く
- 担当: `exp/lib/{rundir,provenance,ckpt,png,host,mingru,optim,metrics}.py`、`tests/lib/test_{rundir,provenance,ckpt,png,host,mingru,optim,metrics}.py`、`tests/test_gil.py`
- 依存: WP1
- 完了の条件: `test_rundir`（原子的な書き込み、offset での切り詰め、manifest の検証が 1 バイトの改変で落ちる）、`test_provenance`（`torch.optim.SGD` を作ると `assert_free_threaded` が EnvError、`configure_determinism` の後も GIL が無効、`OMP_NUM_THREADS=4` で EnvError）、`test_mingru`（`MinGRUTrace` の state_dict の往復で続きのトレースの勾配が一致）、`test_optim`（`adam(lr, b1, b2)` の 1 ステップが閉じた式の Adam と 1e-6 で一致）、`test_host`（`frames` が state_dict で戻る）、`test_metrics`（`bootstrap_se` が既知の分散の正規乱数で理論の SE と 10% 以内）

**WP3 フリート**

- 目的: 4 節の push・キュー・worker・CLI
- 担当: `exp/lib/fleet/**`、`fleet.toml`、`justfile`（`fleet` レシピ）、`tests/lib/fleet/**`
- 依存: WP1（木）、3.2〜3.6 の契約（ファイルの形だけ。偽の run で試す）
- 完了の条件:
  - 単体（ssh も docker も使わない。`LocalTransport` と一時ディレクトリの NAS）: `test_queue`（2 プロセスで同じ job を取り合い 1 回だけ成功、EIO を 3 回注入しても claim が通る、`after`・`holds`・`host`・`avoid_host`・cordon を守る）、`test_slots`（物理コアを 2 重に割り当てない、GPU のメモリと本数の予算を超えない、決定的）、`test_worker`（偽の run（`python -c` で sleep して exit 0/3/4/75/1、SIGKILL）で 4.10 の表どおりに分類、`exit/attempt-n.json` が残る、exit 0 で manifest が無ければ failed、stop で孫まで死ぬ、再起動した worker が生きた run を引き取り死んだ run を `--resume` で戻す、claim が消えたら殺す、同じ印 3 回で `holds/`）、`test_snapshot`（汚れたツリーの snapshot が未追跡のファイルと mujou の gitlink を含み、HEAD と index を動かさない）、`test_jobs`（同じ入力で同じ run_id、seed だけ違えば同じ group、`--sweep` の直積、duplicate の判定）、`test_stdlib_only`（`python -I -c "import exp.lib.fleet.cli, exp.lib.runs"` の後に `torch` も `hydra` も `sys.modules` に無い）
  - 実機: `just fleet doctor` が 4 台で OK。`just fleet push && just fleet up` の後、偽の job（`task=bench` の 5 秒版）を 4 台の各資源の種類で 1 本ずつ流して `done/` に入る。訓練: run を `kill -9`、`docker restart`、`fleet stop`、Mac をスリープ（30 分）して、表どおりに分類されることを確かめる

**WP4 M14 のデータと物差し**

- 目的: 7.1〜7.4
- 担当: `exp/video/data/{physics,observe,clips}.py`、`exp/video/measure/baselines.py`、`tests/video/data/test_{physics,observe,clips}.py`、`tests/video/measure/test_baselines.py`
- 依存: WP1
- 完了の条件: 7.9 の物理・観測・クリップ・物差しの行がすべて通る。`python -m timeit` 相当の試験で B=128・30 分の物理が 1 分未満（slow）

**WP5 M14 のノードと組み立て**

- 目的: 7.5
- 担当: `exp/video/dyn/{modules,nodes,bp,nets}.py`、`tests/video/dyn/test_{modules,nodes,bp,nets}.py`
- 依存: WP1、WP2 の `MinGRUTrace` と `optim` の interface（`state_dict` を持つことだけを仮定して並べてよい）
- 完了の条件: 7.9 の `dyn/test_nodes.py`・`dyn/test_nets.py` の行（credit なしの TypeError、Compare の中心差分、逐次の minGRU、局所性、平衡点の `lam` = BP の credit（小・大、B=1・3）、初期値の一致、トレース = BPTT、残差の初期 = 写し）

**WP6 集計**

- 目的: 6 節の `runs.py`
- 担当: `exp/lib/runs.py`、`tests/lib/test_runs.py`
- 依存: 3.2・3.7 の schema（偽の run ディレクトリで試す）
- 完了の条件: seed を ' / ' で束ねた markdown、発散・打切・欠の表記、欠けた seed を平均に入れない、セルに snapshot・mujou・デバイスの種類が混ざると例外（`allow_mixed` で通る）、終わっていない batch を拒む、監査の hash の不一致を脚注に出す

**WP9 Walking Tours**（M14 と並べて進める）

- 目的: 8.3。11 本の前処理を、M15 が始まるまでに終える
- 担当: `exp/lib/video_io.py`、`exp/video/data/{frames,walking_tours}.py`、`exp/video/data/walking_tours/manifest.json`、`tests/lib/test_video_io.py`、`tests/video/data/test_{frames,walking_tours}.py`
- 依存: WP1。`task=prep` の実行は WP3 と WP8 の後
- 完了の条件: ffmpeg で作った色の合成の mp4 で、形・色・中央の crop・fps が合う。マニフェストに無い動画を `prepare` が拒む。`TourFrames` の state_dict の往復。実データ: 11 本の prep の job が done、`data_version` が README に書かれ、ライセンスとぼかしがマニフェストに記録されている

### 段 C（WP2・WP4・WP5 の後）

**WP7 optuna の調整役**（WP8 と並列）

- 目的: 5 節
- 担当: `tools/sweep/**`（`pyproject.toml`、`uv.lock`、`sweep/*.py`、`tests/`）、`exp/video/studies/*.toml`
- 依存: WP3 の `queue` と `jobs` の interface、3.7 の `objective` のキー
- 完了の条件: `uv run --project tools/sweep pytest tools/sweep/tests` が通る。optuna の in-memory storage と偽の `live/`・`done/` で: 全 seed が届いた時刻だけ report、発散した seed が $\log 10$ として値で tell、exit 1 で FAIL、ASHA の判定で全 seed に `.stop` が置かれる、`confirm` が top k × learner × seed の job と監査 1 本を作り dirty な snapshot を拒む、`bracket` が ×0.5/1/2 を作る、study が別の snapshot の job を拒む、調整役を再起動すると RUNNING の trial の報告を続ける

**WP8 M14 の結合**

- 目的: `main`・`loop`・M14 の program・評価・見た目・オフライン。最初の実験を回せる状態
- 担当: `exp/video/{main,loop}.py`、`exp/video/programs/dynamics.py`、`exp/video/measure/{dynamics,objective,viz}.py`、`exp/video/dyn/offline.py`、`exp/video/conf/experiment/m14/*`（`s0.yaml` と `s3.yaml` 以外を足す）、`schema/{run,data,dyn,learn,measure}.py` の追記、`tests/video/test_{main,loop}.py`、`tests/video/programs/test_dynamics.py`、`tests/video/measure/test_{dynamics,objective,viz}.py`、`tests/video/dyn/test_offline.py`
- 依存: WP2、WP4、WP5
- 設計の注意: `main.py` の program の表は `{"dynamics": "exp.video.programs.dynamics", "vision": "exp.video.programs.vision", "joint": "exp.video.programs.joint"}` の文字列で持ち、遅延で import する。WP10・WP14 は `main.py` を触らない
- 完了の条件: 7.9 の `programs/test_dynamics.py`（契約、決定性、再開 [pc, bp] × [trace]、prequential、終了コード、ctl）、`measure/test_dynamics.py`、`dyn/test_offline.py`、`test_gil.py` が通る。`viz` の PNG を手で復号すると描いた画素で、玉の重心が状態の位置に合う。`python -m exp.video.main +experiment=m14/s0 seconds=60 --out /tmp/m14` がコンテナで走り、3.2 の全ファイルを書く

### 段 D（最初の実験。WP3・WP6・WP8 の後）

**E1 較正と M14 の手順 1・2**（orchestrator）

- 目的: 枠を決め、物差しを測り、判定の基準を README に書く
- 依存: WP0、WP3、WP6、WP8
- 手順: `fleet push`（clean な commit）→ canary → `fleet bench` → `fleet.toml` の `slots` と preset の `res` を更新 → `m14/baselines`（段 0〜2、val と test）→ PNG（等速・等加速度の外挿）を見る → README に閾値を書く → `m14/offline`（段 0〜2、BPTT と 1 フレーム）→ 段 0 の BP の格子（小・大 × トレース × residual × lr 3 点 × seed 100/101、1800 秒）で 600 秒と 1800 秒の順位の Spearman を測る（5.3）
- 完了の条件: `exp/video/README.md` に枠の表、物差しの表、回す前の判定基準、探索の長さの判断が書かれている。`results/m14.md` に bench・物差し・オフラインの上限の表（`fleet table` の出力とそのコマンド）

### 段 E（WP7 と E1 の後）

**E2 M14 段 0→1→2**（orchestrator）

- 目的: M14 の完了の条件を段 1・2 で判定する
- 依存: WP7、E1
- 手順: 段ごとに BP の study（小 / 大 × トレースあり / なし）→ `sweep confirm`（PC-ALM と BP、seed 0/1/2、test、デバイスの種類を固定）→ `sweep bracket` → 監査 → PNG を見る → `results/m14.md`。通らない段は、オフラインの上限と分岐の評価で原因を切り分けてから次へ
- 完了の条件: MILESTONES の M14 の完了の条件（段 1・2、30 分、PC-ALM と BP、seed 0/1/2）を README に判定とともに書く。「重力を学んだ」「跳ね返りを学んだ」を `cv`・`ca`・`@contact` で分けて書く。監査の hash が一致

**WP13 M14 段 3（画像の観測）**（段 1・2 が通ってから）

- 目的: 7.8
- 担当: `exp/video/dyn/pixels.py`、`exp/video/measure/pixel_baselines.py`、`exp/video/conf/experiment/m14/s3.yaml`、`tests/video/dyn/test_pixels.py`、`tests/video/measure/test_pixel_baselines.py`
- 依存: E2 の段 1・2
- 完了の条件: 恒等の読み出しで画素の rollout が写しと一致、等速の外挿を描いた画像の誤差が重力 0・壁なしで 0、玉ごとの領域の誤差の手計算と一致。E2 と同じ手順で段 3 を回して README に書く

### 段 F（M15。WP10・WP11 は段 B の後ならいつでも始めてよい）

**WP10 ViT と JEPA**

- 目的: 8.1・8.2 の PC-ALM 版と BP 版
- 担当: `exp/video/jepa/**`、`exp/video/programs/vision.py`、`exp/video/conf/{vision/*,experiment/m15/*}`、`tests/video/jepa/**`、`tests/video/programs/test_vision.py`
- 依存: WP1（`schema/vision.py` の形）、WP5 の `Source`、WP8 の `loop.Program`
- 完了の条件: 8.7 の `jepa/` の行、vision の `test_resume_equals_uninterrupted` と GIL の検査（`tests/video/programs/test_vision.py` の中。`tests/test_gil.py` は WP2 のもの）

**WP11 視覚の評価と後付けのデコーダ**

- 目的: 8.4・8.5
- 担当: `exp/video/measure/{vision,posthoc}.py`、`tests/video/measure/test_{vision,posthoc}.py`
- 依存: WP10 の `build_vision` の interface、WP2 の `ckpt`
- 完了の条件: 8.7 の `measure/` の行。`task=posthoc` が小さい run の ckpt から `post/decode/` を書く

**E3 M15**（orchestrator）

- 手順: 玉の $N=1$ → $N=4$ → Walking Tours の $N=1$ → $N=8$。BP で探索、sg あり / なし、確認は PC-ALM と BP、seed 0/1/2。各 ckpt のデコーダの PNG を見る
- 完了の条件: 受け渡しの条件（8.4）の判定を README に書く。M16 に持ち込む設定を決める

### 段 G（M16。E2 と E3 の後）

**WP14 同時の学習**

- 目的: 8.6
- 担当: `exp/video/programs/joint.py`、`exp/video/conf/experiment/m16/*`、`tests/video/programs/test_joint.py`
- 依存: WP5・WP8（`DynamicsBlock`、`DynamicsBP`）、WP10（`JEPAHead.z_out`、`JEPABP`）、E2 と E3 の設定
- 完了の条件: 8.7 の `programs/test_joint.py`（エンコーダの重みのビット一致、動力学は M14 のもの）と joint の再開の試験

**E4 M16**（orchestrator）

- 手順: 同時の学習と凍結の対照を、PC-ALM と BP、seed 0/1/2 で。posthoc のデコーダで 5・10・20 フレーム先の画像を比べ、PNG を見る
- 完了の条件: MILESTONES の M16 の判定を README に書く

### 並べ方のまとめ

```text
WP0 → WP1 → ┬ WP2 ┐
            ├ WP3 ┼──────────────┐
            ├ WP4 ┼→ WP8 ─┬→ E1 ─┴→ E2 → WP13
            ├ WP5 ┘       │       ↑
            ├ WP6 ────────┘   WP7 ┘（WP2・WP4・WP5 の後、WP8 と並列）
            └ WP9 ─────────────→ (prep は WP3・WP8 の後)
WP5・WP8 の後: WP10 → WP11 → E3 ; E2・E3 の後: WP14 → E4
```

---

## 10. 未確認のことと危険

1. **SMB の rename の原子性**（WP0 で 2 台の取り合いを確かめる）。成り立たなければ、claim を「`claimed/<host>/` へ `O_EXCL` で印を作り、`queue/` から消す」の 2 段にし、重複した claim は run_id の小さいホストを正にする
2. **`uv sync --no-editable` で mujou が wheel として入るか**（WP0）。駄目なら editable のまま `code/<snap12>/mujou` に向ける（`code/` は不変なので正しいが、その snapshot の `code/` を消すと venv が壊れる。gc が venv の参照を見る）
3. **mujou の state_dict がポートの最後の値と時刻を戻すか**。map では木に入る。足りなければ `HostState` で補う（再開の試験が教える）
4. **numpy の不在**: optuna を学習の venv に入れないので変わらない。posthoc のデコーダも `pc.optim` で書き、numpy を要しない
5. **M14 段 1 の表現力**: 1 ノードの minGRU はゲート付きの漏れ積分器の束で、跳ね返りの非線形は層の間の GELU（大）に頼る。オフラインの上限が先に答える
6. **評価の暖機**: 16 フレームで $s$ が定常に達するか。クリップの評価と分岐の評価の差で見る。差が大きければ、判定の主を分岐の評価に替えるかを回す前に README で決める
7. **PC-ALM + 大 + トレースの時間**: `DynamicsLayer` のトレースは B=128・$d=512$ で 1 本 268 MB、毎フレーム $O(Bd^2)$。GPU の job にし、`fleet bench` で見積もってから確認の本数を決める
8. **Mac に残る依存**: `fleet up`（worker の起こし直し）と `fleet push`（新しいコード）は Mac から。ホストやコンテナが再起動した後、worker が戻るのは orchestrator が次に `fleet up` を叩いたとき。必要になったら compose の `command` に worker の起動を足す（今は入れない）
9. **ASHA と 1800 秒の探索の費用**: E1 の相関で 600 秒に縮められるかを決める。縮められなければ、`parallel` と `max_trials` を枠に合わせて減らす
