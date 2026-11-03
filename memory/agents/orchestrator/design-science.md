# 再編の設計案（科学的な正しさの観点）: M14〜M16 を全ホストで多数回す

2026-11-03、orchestrator 配下の設計サブエージェント。既存のファイルは変えていない。読んだもの: `m14-m16-constraints.md`、`exp/video/MILESTONES.md`、`map-*.md`（7 本）、`m13-easy-video.md`、`exp/video/{nodes,bp,balls,evaluate,offline}.py`、`exp/lib/{host,record,mingru,sigreg}.py`、`mujou/src/mujou/pc/{input,clock,functional}.py`、`justfile`、`compose.yaml`、`pyproject.toml`、`.devcontainer/devcontainer.json`。

**この案の観点**: マイルストーンの問いに正しく答えられること。具体的には、基準（物差し）の選び方、理論から導いたオラクル、局所性の不変条件、BP と PC-ALM の公平な比較、見た目の確認、M15 と M16 が mujou の中で実際に動く設計。インフラ（フリート・optuna・集計）は、この観点で必要な性質（再現性、探索と報告の分離、途中の打ち切り）が満たせる最小の形にした。

---

## 0. 要点（先に結論）

1. **探索と報告を分ける**。optuna が見るのは **val** の評価クリップと探索用の seed（100 番台）。README に書く数字は **test** の評価クリップと seed 0/1/2 の確認の run だけから作る。同じクリップで選んで同じクリップで報告すると、選択の偏り（winner's curse）が数字に乗る
2. **M14 の基準を 1 つ足す**: 「等加速度の外挿」（最後の 3 フレームの位置から 2 階差分で外挿。重力は学ばずに当たる）。等速の外挿を下回るだけでは「重力を学んだ」までしか言えない。**跳ね返りを学んだ**と言うには、等加速度の外挿を下回り、かつ「予測の区間に壁への接触がある窓」に絞った誤差でも下回る必要がある。真の軌跡から接触の有無が分かるので、誤差を「接触あり / なし」に分けて記録する
3. **評価は 2 通り**: (a) 評価クリップで $s=0$ から 16 フレーム観測して開ループ（M13 と同じ。オフラインの上限と同じ物差し）、(b) **学習中のストリームから分岐**（ノードの今の $s$ を写し、物理のストリームを複製して 20 フレーム先まで開ループ）。EASY のストリームは引き直しが無く、モデルは $s=0$ から 16 フレームという状況を学習中に一度も見ないので、(a) だけでは「暖機が足りない」と「動力学を学べていない」を区別できない
4. **credit を送らないことを構造で保証する**: 動力学の部分（`Dynamics`・`DynamicsLayer`・`Compare`）が上流から受け取るポートを **recv だけの Port** にする。M14 の上流は送るだけの `Source` ノード、M16 の上流は JEPA のヘッドの送るだけのポート。`target_weight` のような「重み 0」の約束ではなく、接続の型で credit が戻れない。M16 では「動力学を付けても付けなくても、同じ seed でエンコーダの重みがビット単位で同じ」をオラクルのテストにする
5. **BP と PC-ALM の公平さ**: 同じ `build_*` 関数で同じ順に乱数を消費して初期値を一致させ、同じ optimizer・同じ `LearnClock`・同じメモリの近似（1 フレームで止める、または同じトレース）・同じデータと評価にする。ハイパーパラメータは BP で探索し、PC-ALM にはそのまま移す。そのうえで、最良の 1 設定について**両方に同じ幅の学習率の小さな格子（×0.5, ×1, ×2）**を回し、PC-ALM が BP 向けの値で不利になっていないかを確かめる
6. **M15 の設計の要**: ViT の JEPA を **「ビューをサンプルとして並べる」**形で mujou に載せる。1 本のストリームから毎フレーム、マスクした文脈のビューとマスクしない目標のビューを作り、バッチの次元に並べる（`(B, V, ...)`）。エンコーダはサンプルごとに独立なので `pc.Layer` のノードにそのまま割れ、目標のエンコーダの重みを別に持たない（EMA なし、LeJEPA の形。重みの書き手は 1 ノード）。ビューをまたぐ損失（予測器と SIGReg）は 1 つのヘッドのノードの中に閉じる
7. **M15 の危険（M13 の失敗の再来）を先に測る**: マスクした領域の予測は、文脈から当てられない内容（マスクの中の玉）を表現から捨てる向きに働く。SIGReg は 1 次元の射影の分布しか見ないので、それを止めない（M13 で確認済みの機構）。M15 の合格条件に「**動力学に渡す $z$ から、玉ごとの位置が線形に読める**」（玉ごとの probe の $R^2$）を入れ、目標側の stop-gradient のあり・なしを比べる。これが M15 → M16 の受け渡しの条件になる
8. **mujou への変更は M14 では要らない**。M15 も「ビューをサンプルに」の形なら `pc.Layer` と自作ノードで足りる。足りない機能が見つかったら skill `mujou-upstream` で送る（候補は 9 節）

---

## 1. パッケージの構成

### 1.1 方針

- `exp/video` は 1 つの実験（連続ストリームの世界モデル）なので 1 パッケージのまま、中を**領域ごとの下位パッケージ**に分ける。並行して実装するエージェントの担当がディレクトリで分かれ、ファイルの衝突が減る
- 設定のスキーマも領域ごとのファイルに分ける（1 つの `schema.py` を全員が編集すると衝突する）
- 互換は捨てる。M13 以前のスクリプト・EgoDex・CNN エンコーダ・`Reconstruct`・`Correlation`・線形回帰の段 0 は消す（過去の数字は `results/*.md` と git の履歴に残る）

### 1.2 `exp/lib/`（mujou・torch・標準ライブラリだけ。`exp/video` を import しない）

| モジュール | 中身 | 由来 |
| -- | -- | -- |
| `mingru.py` | `MinGRU`, `ResidualMinGRU`, `MinGRUTrace`（**`mujou.Stateful` にして**ノードの状態に入れる。今は private 属性で state_dict に入らず、再開するとトレースが消える） | 残す・直す |
| `sigreg.py` | `epps_pulley`, `sigreg`, `ecf_sums`, `sigreg_from_sums`, `insert_`, `SIGReg` ノード | 残す |
| `optim.py` | `DirectionOptimizer`, `AdamDirection`, `Grafted`, `Warmup`。**`Whitened`・`Projected`・`Orthogonalized`・`Correlation`・`_Factors`・`_eigh` は消す**（M10 で SGD を超えず、M14 の探索空間にも入れない）。`adam(lr, beta1, beta2)` → `pc.optim.AdamW` の換算（`tau = 1/lr`, `tau_m = -1/ln β1`, `elapsed=1` で 1 ステップ）を足す: オフラインの上限と後から学ぶデコーダで `torch.optim` を使わない（GIL を戻さず、オンラインと同じ optimizer の実装） | 残す・削る |
| `host.py` | `Host`（`RealtimeHost` は消す。M14〜M16 は仮想時間だけ） | 残す |
| `metrics.py` | `effective_rank`, `probe_r2`、足す: `bootstrap_se(x, n, gen)`（窓をまたいだ平均の標準誤差） | 残す・足す |
| `rundir.py` | run の契約（3 節）: `RunDir`, `CodeVersion`, `check_gil()` | `record.py` を置き換え |
| `ckpt.py` | `save(path, tree, host_state, config_yaml)`, `load(path) -> Checkpoint`, `module_state(tree, path) -> dict` | 新規 |
| `conf.py` | hydra の Compose API の薄い包み（2 節）: `compose(overrides) -> RunConf`, `run_name`, `tags` | `config.py` を置き換え |
| `png.py` | `write_png`, `tile`, `draw_points_(canvas, xy, color)`, `draw_polyline_(...)`（zlib だけの PNG。M14 の軌跡の図も torch で描く） | `decode.py` から |
| `video_io.py` | `ffmpeg_frames(path, *, fps, width, height, start, duration) -> Iterator[Tensor u8 (3,H,W)]`（`Popen` で固定長ずつ読む）、`U8Writer`, `U8Reader`（memmap） | `egodex.py` を一般化 |
| `runs.py` | **標準ライブラリだけ**。run の索引・絞り込み・表（6 節） | 新規 |
| `fleet/` | **標準ライブラリだけ**。ランチャー（4 節）: `__main__.py`（CLI）, `model.py`（`Job`, `Resources`, `HostSpec`）, `alloc.py`（純関数の割り当て）, `remote.py`（ssh・docker exec・rsync）, `agent.py`（ホストで動く常駐）, `snapshot.py`（コードの版） | 新規 |

消す: `config.py`（`from_dict`）、`record.py`、`sweep.py`、`linear_stream.py`。

### 1.3 `exp/video/`

```text
exp/video/
  README.md  PLAN.md  MILESTONES.md  results/  report*.html   (文書は残す)
  main.py            # python -m exp.video.main <overrides...> --out DIR [--resume]
  loop.py            # ホストのループ: 書く → step → 読む、評価・ckpt・発散の検知・SIGTERM
  schema/            # structured config（領域ごと）
    __init__.py      #   ConfigStore への登録を 1 か所で（各モジュールの register() を呼ぶ）
    run.py           #   RunConf, TimeConf, PhaseConf
    data.py          #   DataConf, BallsStateConf, BallsImageConf, WalkingToursConf, PhysicsConf
    dyn.py           #   DynamicsConf, NoDynamicsConf
    jepa.py          #   EncoderConf, NoEncoderConf, ViTConf, JEPAConf
    learn.py         #   LearnerConf (PC/BP/Offline), OptimConf
    eval.py          #   EvalConf, CriteriaConf
  conf/              # hydra の yaml（2 節）
  data/
    physics.py       # 玉の物理をベクトル化（torch、B×玉）: Physics, BallState, BallStreams, render
    observe.py       # 段 → 観測の列、Standardizer、観測 → render 用の状態
    clips.py         # ClipSet（val/test/calib）、開始位置、接触のマスク
    frames.py        # 画像のストリーム（玉の画像、Walking Tours）の共通の形
    walking_tours.py # 前処理（ffmpeg → u8）、マニフェスト、評価クリップ
  dyn/
    modules.py       # DynamicsStack, Top, build_dynamics（PC・BP・オフラインで共有）
    nodes.py         # Source, Dynamics, DynamicsLayer, Compare（kappa と target_weight を外す）
    bp.py            # DynamicsBP（WorldBP からエンコーダ・SIGReg・再構成を外した 1 ノード）
    nets.py          # StateWorldPC, StateWorldBP, PixelWorldPC/BP（段 3）
    offline.py       # オフラインの上限（窓、Adam、BPTT か 1 フレームで切る）
  jepa/
    vit.py           # PatchEmbed（tubelet）, Block, ViT, Predictor
    masks.py         # block_masks, tube_masks（node の generator から）
    loss.py          # jepa_energy, ヘッドの損失
    nodes.py         # Chunker, JEPAHead
    bp.py            # JEPABP（同じ部品を 1 ノードで BP）
    nets.py          # VisionPC, VisionBP
  joint.py           # M16: JointPC, JointBP
  eval/
    baselines.py     # mean, copy, const_velocity, const_accel, true_physics
    dynamics.py      # rollout, evaluate_dynamics（クリップ）, fork_eval（ストリームから分岐）
    vision.py        # 玉ごとの probe、縮小画像の probe、実効ランク
    posthoc.py       # チェックポイントごとに後から学ぶデコーダ（別のジョブ）
    viz.py           # 見た目の確認の PNG（フレームの並べ図、軌跡の重ね図）
```

消す: `stage0.py`〜`stage3.py`、`stability.py`、`collapse.py`、`frozen.py`、`latency.py`、`easy.py`、`egodex.py`、`config.py`、`model.py`、`nodes.py`（→ `dyn/nodes.py`）、`bp.py`（→ `dyn/bp.py`）、`run.py`（→ `main.py` + `loop.py`）、`decode.py`（→ `eval/posthoc.py`）、`offline.py`（→ `dyn/offline.py`）、`evaluate.py`（→ `eval/`）、`balls.py`（→ `data/physics.py`）。

### 1.4 `tools/sweep/`（optuna の調整役。別の uv プロジェクト）

`exp/lib` は mujou・torch・標準ライブラリだけに依存する規約なので、optuna を使う調整役は外に出す。

```text
tools/sweep/
  pyproject.toml     # optuna==5.0.0, hydra-core==1.3.8（submit 前に override を組んで検査する）。torch も mujou も入れない
  sweep/cli.py       # python -m sweep ...
  sweep/coordinator.py, spaces.py, objective.py
  tests/
```

Mac で `uv run --project tools/sweep python -m sweep ...` として動く（GIL ありの普通の CPython 3.14 でよい。学習のプロセスではないので GIL の制約は効かない）。`exp.lib.fleet` は `sys.path` にリポジトリのルートを足して import する（標準ライブラリだけなので torch は要らない）。

### 1.5 mujou へ移すもの

M14 には無い。`Source`（送るだけの観測のノード）は実験側に置き、M15・M16 でも同じ形が要ると分かったら `pc.Source` として提案する候補にする。9 節に候補を並べる。

### 1.6 テストの構成

`exp/<pkg>/<mod>.py ↔ tests/<pkg>/test_<mod>.py` のミラー（各ディレクトリに `__init__.py`）。

```text
tests/
  conftest.py               # OMP_NUM_THREADS=1（今のまま）
  test_gil.py               # exp.video.main と hydra を import した後で GIL が無効（free-threaded のときだけ）
  test_claude_hooks.py  test_mujou_install.py
  lib/  test_mingru.py test_sigreg.py test_optim.py test_host.py test_metrics.py
        test_rundir.py test_ckpt.py test_conf.py test_png.py test_video_io.py test_runs.py
        fleet/ test_alloc.py test_agent.py test_snapshot.py test_model.py
  video/ test_main.py test_loop.py
        schema/ test_compose.py        # 全プリセットが組めて to_object が往復する
        data/ test_physics.py test_observe.py test_clips.py test_walking_tours.py
        dyn/ test_modules.py test_nodes.py test_bp.py test_nets.py test_offline.py
        jepa/ test_vit.py test_masks.py test_loss.py test_nodes.py test_bp.py test_nets.py
        eval/ test_baselines.py test_dynamics.py test_vision.py test_posthoc.py test_viz.py
        test_joint.py
tools/sweep/tests/ test_objective.py test_coordinator.py   # 偽の evals.jsonl で集約・打ち切りを確かめる
```

---

## 2. hydra の設定

### 2.1 使い方の方針

- **Compose API だけを使う**（`hydra.compose` + `initialize_config_module("exp.video.conf")`）。`@hydra.main` は使わない: hydra の作業ディレクトリの移動・ログ・`multirun/` を避け、出力の場所は run の契約（3 節）とランチャーが決める。multirun の代わりはフリートと optuna
- コードが触るのは `compose()` が返す **dataclass のインスタンス**（`OmegaConf.to_object`）だけ。`DictConfig` を関数の間で渡さない。pyright はふつうの dataclass として型を見る
- 学習率は「1 フレームあたりの lr」で持ち、`tau = frame / lr` は optimizer を作る関数で計算する（optuna が対数一様で探索しやすい。M13 の換算の取り違えを無くす）

### 2.2 structured config（抜粋）

```python
# exp/video/schema/data.py
class Obs(StrEnum):
    POS_VEL = "pos_vel"   # stage 0
    POS = "pos"           # stages 1, 2
    PIXELS = "pixels"     # stage 3

@dataclass(kw_only=True)
class PhysicsConf:                       # the EASY physics of M13 by default
    gamma0: float = 1.0
    gamma_amp: float = 0.0
    e0: float = 1.0
    e_amp: float = 0.0
    drag: float = 0.0
    conserve: bool = True
    substeps: int = 8

@dataclass(kw_only=True)
class DataConf:
    kind: str = MISSING
    seconds: float = 1800.0
    streams: int = 128                   # B

@dataclass(kw_only=True)
class BallsStateConf(DataConf):
    kind: str = "balls_state"
    radii: list[float] = field(default_factory=lambda: [0.10])
    obs: Obs = Obs.POS_VEL
    size: int = 32                       # only to render (visual check, stage 3)
    physics: PhysicsConf = field(default_factory=PhysicsConf)
```

```python
# exp/video/schema/run.py
@dataclass(kw_only=True)
class RunConf:
    name: str = MISSING                  # set by the preset: "m14/s1"
    tags: list[str] = field(default_factory=list)
    seed: int = 0
    device: str = "cpu"                  # the launcher overrides it with the slot's device
    data: DataConf = MISSING             # group "data": a subclass of DataConf
    dynamics: DynamicsConf = MISSING     # group "dynamics": NoDynamicsConf | DynamicsConf
    encoder: EncoderConf = MISSING       # group "encoder": NoEncoderConf | ViTConf
    learner: LearnerConf = MISSING       # group "learner": PCConf | BPConf | OfflineConf
    optim: OptimConf = MISSING           # the dynamics' optimizer
    optim_enc: OptimConf | None = None   # the encoder's (M15, M16)
    eval: EvalConf = MISSING
    phase: PhaseConf = MISSING           # group "phase": explore | confirm
    time: TimeConf = field(default_factory=TimeConf)
```

- 群の選択肢は基底の dataclass の**派生クラス**にする（hydra の文書の「`db: DBConfig` に `MySQLConfig(DBConfig)`」の形）。コードは `match conf.data: case BallsStateConf(): ...` で分ける。pyright は `match` の絞り込みを理解する
- 列挙は `StrEnum`（OmegaConf は `Literal` を扱えない）。タプルは `list` にする
- **WP0 で確かめること**: OmegaConf 2.3 が `kw_only=True` と `frozen=True` の dataclass を `to_object` できるか。できなければ schema は `frozen` にせず、`to_object` の後で触らない規約にする（テスト `test_compose.py` が全プリセットで往復を確かめる）

### 2.3 yaml の木

```text
exp/video/conf/
  run.yaml                       # defaults: [_self_, data: s0, dynamics: small, encoder: none,
                                 #            learner: bp, optim: sgd, eval: m14, phase: explore]
  data/      s0.yaml s1.yaml s2.yaml s3.yaml balls_frames.yaml wt64.yaml
  dynamics/  none.yaml small.yaml big.yaml          # small: d_s 256, 1 node / big: d_s 512, 3 nodes
  encoder/   none.yaml vit_t.yaml vit_s.yaml
  learner/   pc.yaml bp.yaml offline.yaml
  optim/     sgd.yaml adamw.yaml graft_adam.yaml
  eval/      m14.yaml m15.yaml m16.yaml
  phase/     explore.yaml confirm.yaml              # explore: val のクリップ、ckpt なし / confirm: test も評価、ckpt 300 秒
  experiment/                                       # "# @package _global_"
    m14/ baselines.yaml s0.yaml s1.yaml s2.yaml s3.yaml offline.yaml
    m15/ balls_n1.yaml balls_n4.yaml wt_n1.yaml wt_n8.yaml
    m16/ balls.yaml wt.yaml
```

段のデータの yaml の例:

```yaml
# conf/data/s2.yaml  (stage 2: three balls, positions only)
defaults: [balls_state_schema]
radii: [0.06, 0.08, 0.10]
obs: pos
```

```yaml
# conf/experiment/m14/s1.yaml
# @package _global_
defaults:
  - override /data: s1
  - override /encoder: none
  - override /eval: m14
name: m14/s1
tags: [m14, s1]
```

### 2.4 起動と、override から名前・タグへの対応

```bash
python -m exp.video.main +experiment=m14/s1 phase=confirm learner=pc dynamics=big \
    dynamics.trace=true optim.lr=0.03 seed=0 --out /data/geson/mujou-exp/runs/<run_id>
```

- run のディレクトリ名は `run_id = <yyyymmdd>-<hhmmss>-<host>-<4 hex>`（一意。同名の衝突と日付の glob を無くす）
- `name` はプリセットが決める（`m14/s1`）。`tags` = プリセットの `tags` ∪ {`learner`, `phase`, 段} ∪ ランチャーが付けるもの（`study=...`, `confirm`）
- `meta.json` に **override の列をそのまま**（`["learner=pc", "dynamics=big", ...]`）と、組んだ後の設定の全体（`config.yaml`）を残す。集計は設定の値で絞るので、名前に値を埋め込まない
- `conf.run_name(overrides)` は人が読む短い名前（`m14/s1 pc big trace lr0.03 s0`）を作るだけで、識別には使わない

---

## 3. run の契約

1 つの run = 1 つのプロセス = 1 つの `mujou.testing.run`。どの種類（online / offline / baselines / posthoc / prep）も同じ骨組みで書く。

### 3.1 ディレクトリ

```text
runs/<run_id>/
  meta.json         # 起動のときに 1 回（下の schema）
  config.yaml       # 組んだ後の設定（OmegaConf.to_yaml）
  code.diff         # repo_sha からの未コミットの差分（ランチャーの snapshot が作る）
  baselines.json    # この run の評価クリップでの物差し（val と test、各 h）
  stream.jsonl      # 仮想時間 1 秒ごとの集約: {"t", "frames", "preq", "preq_open", "wall"}
  evals.jsonl       # 評価 1 回 1 行（平らな dict、下の schema）
  result.json       # 終わったときに 1 回
  status            # running | done | failed | diverged | pruned | killed（一時ファイル → os.replace）
  ckpt/<t:06d>.pt   # mujou の state_dict の木 + ホストの状態（3.3）
  viz/<t:06d>/      # 見た目の確認の PNG（frames.png, traj.png）
  post/decode/      # 後から学ぶデコーダ（M15・M16）の出力
  stdout.log stderr.log   # エージェントが書く
```

毎フレーム 1 行の `metrics.jsonl`（30 分で 54,000 行）はやめる。前向きの予測の誤差は 1 秒ごとに平均して `stream.jsonl` へ、評価は `evals.jsonl` へ。各行を書いたら flush する（途中経過を optuna と人が読むため）。

### 3.2 schema

`meta.json`:

```json
{
  "run_id": "20261103-153012-std-02-a3f1", "job_id": "j-...", "kind": "online",
  "name": "m14/s1", "tags": ["m14", "s1", "pc", "confirm"],
  "overrides": ["+experiment=m14/s1", "learner=pc", "seed=0"],
  "study": null, "trial": null,
  "host": "std-02", "device": "cuda:0", "cpu_core": 7,
  "code": {"version": "c-3f2a91d0", "repo_sha": "06a2a11...", "dirty": true,
           "diff_sha256": "...", "mujou_sha": "881c4ee...", "mujou_dirty": false},
  "env": {"python": "3.14.0", "free_threaded": true, "gil_enabled": false,
          "torch": "2.15.0", "cuda": "12.8", "omp_threads": "1"},
  "started": "2026-11-03T15:30:12+09:00", "argv": ["..."]
}
```

`evals.jsonl` の 1 行（M14。値はすべて標準化した観測の空間の 1 次元あたりの二乗誤差。キーは平ら）:

```json
{"t": 600.0, "frame": 18000, "split": "val", "wall": 812.3,
 "roll1": 0.01, "roll5": 0.05, "roll10": 0.12, "roll20": 0.40,
 "mean5": 1.0, "copy5": 0.31, "cv5": 0.08, "ca5": 0.04,
 "roll10/cv10": 0.62, "roll10/ca10": 0.9,
 "roll10@contact": 0.30, "roll10@free": 0.05, "ca10@contact": 0.45, "ca10@free": 0.0,
 "roll10@ball0": 0.1, "roll10@ball1": 0.12, "roll10@ball2": 0.15,
 "roll10_se": 0.004, "amp10": 0.95, "motion10": 0.98,
 "fork10": 0.11, "fork10/cv10": 0.6,
 "objective": -0.47}
```

- `amp{h}` = 予測の分散 / 真の分散、`motion{h}` = 予測の 1 フレームの移動量の平均 / 真の移動量の平均。M11 の「玉が止まって薄れる」を数で拾う補助（見た目の確認の代わりにはしない）
- `@contact` / `@free`: 観測の最後のフレームから $h$ フレーム先までの間に、真の軌跡で壁への接触（速度の成分の符号の反転）がある窓 / 無い窓。`@ball{i}`: 玉ごと（段 2）
- `objective` は optuna が見る値（5.2）。val の行にだけ入れる

`result.json`:

```json
{"status": "done", "frames": 54000, "wall_seconds": 6123.4, "frames_per_wall_second": 8.8,
 "diverged_at": null, "final": {"val": {...}, "test": {...}},
 "criteria": {"roll5/cv5<0.5": true, "roll10/cv10<0.5": false}, "objective": -0.47}
```

`criteria` は、確認のプリセットに書いた（README と同じ値の）閾値を機械で判定したもの。見た目の確認は人（orchestrator）が PNG を見て README に書く。

### 3.3 チェックポイント

- `ckpt.save` は `run.state_dict()`（mujou の木。ノードの module・optimizer・clock・`r`・`lam`・ポートの値）と、ホストの状態（`BallStreams.state_dict()`: 玉の位置・速度・generator、フレームの番号、`Host.frames`、1 秒の集約の途中の値）と、`config.yaml` の文字列を 1 つの dict にして `torch.save` する。`torch.load(weights_only=True)` で読める（独自クラスを入れない）
- 評価・デコーダ・図は、設定から**ネットワークを組み直して** `ckpt.module_state(tree, "world.dynamics")` を `load_state_dict` する。pickle の `Snapshot` をやめるので、コードを動かしても読める
- 再開: `main --resume <run_dir>` が最後の ckpt から `mujou.testing.run(net, state=tree)` で続ける（ランチャーが落ちた run を戻すとき）。`MinGRUTrace` を `Stateful` にするのはこのため

### 3.4 コードの版と GIL

- コードはランチャーの snapshot（4.4）から走る。snapshot のディレクトリの `CODE.json`（`repo_sha`, `dirty`, `diff_sha256`, `version`）と `code.diff` を `RunDir` が写す。手で回す run（snapshot の外）は git から同じものを作る
- `mujou_sha` は import した mujou のパスの git から取る（snapshot は exp/ だけで、mujou はホストの checkout の editable install を使うため）
- **GIL**: `rundir.check_gil()` を、すべての import（hydra・omegaconf を含む）の後、ネットワークを作る前に呼ぶ。`sysconfig.get_config_var("Py_GIL_DISABLED")` が真で `sys._is_gil_enabled()` も真なら、online の run は `RuntimeError` で止める（どの import が戻したかを `sys.modules` の差分で出す）。結果は `meta.json` に残す。offline と posthoc も `pc.optim` で書くので同じ検査を通るはず。`tests/test_gil.py` が同じことをテストで確かめる

---

## 4. フリートのランチャー（`exp/lib/fleet`、標準ライブラリだけ）

### 4.1 形

```text
Mac（orchestrator / optuna）           std-00..03（docker の中）
  fleet CLI ── ssh ──────────────▶  agent（常駐、1 ホスト 1 つ）
  queue/（Mac のローカル）            /data/geson/fleet/{inbox,jobs,code}/
  runs-index/（軽いファイルの写し）    /data/geson/mujou-exp/runs/<run_id>/   ← ローカルの disk に書く
                                      /nas/geson/mujou-exp/runs/<run_id>/   ← 終わったら agent が写す
```

- **割り当ては Mac、実行はホストの agent**。Mac の `fleet dispatch` が空きのあるホストにだけジョブを渡し（`ssh std-0X 'cat > inbox/<id>.json'`）、agent は自分の inbox にあるものを順に起動する。ホスト同士は通信しない。走っている run は Mac が眠っても止まらない
- **出力はホストのローカルの disk**（毎秒の flush、ckpt。CIFS への細かい追記を避ける）。agent が終わった run を `/nas/geson/mujou-exp/runs/` へ写す（compose.yaml に `/nas/geson:/nas/geson` のマウントを足す）。Mac は `fleet pull` で軽いファイル（meta・config・evals・stream・result・status・baselines・viz の PNG）だけを `rsync -e ssh` で取る。ckpt は NAS に残す

### 4.2 `fleet.toml`（リポジトリのルート、git に入れる）

```toml
[defaults]
checkout = "/data/geson/mujou-exp"
container = "mujou-exp-dev-1"
fleet_root = "/data/geson/fleet"
nas_runs = "/nas/geson/mujou-exp/runs"

[hosts.std-00]
ssh = "std-00"
cpus = 30          # 32 cores - 2 (agent, ssh, eval bursts); fixed by `fleet bench`
ram_gb = 56
gpus = [{index = 0, name = "4090", mem_gb = 22, max_jobs = 8},
        {index = 1, name = "4060", mem_gb = 7, max_jobs = 3}]
# std-01..03 likewise
```

### 4.3 ジョブ

```json
{"id": "j-20261103-153001-7c2e", "kind": "online",
 "module": "exp.video.main", "args": ["+experiment=m14/s1", "learner=bp", "seed=100"],
 "code": "c-3f2a91d0",
 "res": {"cpu": 1, "gpu": "none", "gpu_mem_gb": 0, "ram_gb": 2},
 "priority": 0, "tags": ["study=m14-s1-bp"], "study": "m14-s1-bp", "trial": 12,
 "after": [], "max_retries": 1, "resume": true}
```

- `res.gpu`: `none`（CPU だけ）、`any`、`4090`。どの GPU のジョブも CPU を 1 コア使う（律速は CPU 1 コアのカーネル起動）
- agent は 1 ジョブに 1 コアを決めて `taskset -c <core>`、環境に `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=<i>`、`PYTHONPATH=<code snapshot>` を入れる。設定の `device` は `cuda:0`（見える 1 枚）か `cpu` を override で渡す。`OMP_NUM_THREADS` を入れないと OpenMP の取り合いで数十分止まる（既知）
- GPU のメモリは**宣言した `gpu_mem_gb` の和**で数える（起動の前に `nvidia-smi` の空きも確かめる）。プリセットが既定の `res` を持ち、`fleet submit` が上書きできる

### 4.4 コードの同期: rsync の snapshot（git の checkout は動かさない）

- `fleet sync` は Mac の作業ツリーの `exp/` を内容で hash して版 `c-<8 hex>` を作り、`rsync -a --delete --exclude __pycache__ exp/ std-0X:/data/geson/fleet/code/<ver>/exp/` で全ホストへ送る。`CODE.json`（Mac の `git rev-parse HEAD`、`git status --porcelain` の有無、`git diff HEAD -- exp` の sha256）と `code.diff` も同じ所に置く
- **snapshot は書き換えない**。走っているジョブと、まだ始まっていないジョブのコードが、後の編集で変わらない（今の「checkout を直すと、待っているジョブのコードも変わる」という再現性の穴を塞ぐ）
- 依存と mujou はホストの checkout（`/data/geson/mujou-exp` の `.venv-devcontainer`）のものを使う。`fleet sync` は Mac とホストの `uv.lock` の hash と mujou の SHA を比べ、違えば止める。`fleet setup [--hosts]` がホストの checkout で `git fetch && git checkout <sha> && git submodule update && uv sync`（コンテナの中）をする。これは走っているジョブが無いときだけ許す（editable の mujou が入れ替わるため）
- 未コミットの変更で回してよい（探索の反復を速くする）。ただし **README に載せる確認の run は、コミットして push した版（`dirty=false`）だけ**にする。`runs.py table` は既定で dirty の run に印を付ける

### 4.5 agent

- 起動: `fleet up` が各ホストで `docker exec -d -u vscode -w /data/geson/mujou-exp mujou-exp-dev-1 python3 -m exp.lib.fleet.agent --host std-0X` を実行する（コンテナの python で動く。標準ライブラリだけなので venv も要らない）。`agent.pid` と 10 秒ごとの `heartbeat.json`（空きのコア・GPU のメモリ・走っているジョブ・`torch.cuda.device_count()` の値は起動時に 1 回）
- ループ（5 秒）: inbox を priority 順に見て、`after` が満たされ資源があれば `subprocess.Popen(..., start_new_session=True)` で起動し `jobs/<id>.json` に pid・コア・GPU を書く。終わったら終了コードで `done` / `failed` を決め（run が `status` に `diverged` / `pruned` を書いていればそちら）、NAS へ写す
- 止める: `fleet kill <id>` が `control/<id>.kill` を置き、agent がプロセスグループに SIGTERM、30 秒後に SIGKILL。run は SIGTERM で `status=killed`（optuna が頼んだときは `pruned`）を書いて flush してから終わる
- 失敗: 終了コードが 0 でない run は `failed`。決定的な失敗を何度も回さないため、自動の再試行は「agent かホストが落ちて消えた（`lost`）」ときだけ。agent が起動したとき、`jobs/` の pid が生きていなければ `lost` にし、`resume=true` なら最後の ckpt から `--resume` で inbox に戻す。Mac は heartbeat が 2 分途絶えたホストを `down` にし、そこの未開始のジョブを他へ移す

### 4.6 CLI（orchestrator が使う）

```bash
fleet hosts                                  # 各ホストの agent・空き・GPU・版の一致
fleet bench --preset m14/s0 --levels 1,4,8,16,24,30   # 同時に回す本数と 1 本の速さ（4.7）
fleet sync                                   # snapshot を送り、版 c-xxxx を出す
fleet setup --hosts all                      # ホストの checkout と venv を揃える（空いているときだけ）
fleet submit --preset m14/s1 --set learner=bp dynamics=big --seeds 100,101 \
             [--res gpu=any,gpu_mem_gb=3] [--priority 5] [--tag explore] [--after j-...]
fleet submit --file jobs.jsonl               # optuna や手で作ったジョブの列
fleet dispatch [--daemon 30]                 # 空きにジョブを渡す（submit も 1 回呼ぶ）
fleet ls [--state queued|running|done|failed] [--tag ...] [--study ...]
fleet logs <job> [--tail 50] [--err]
fleet kill <job> | --study <s> | --tag <t>
fleet retry <job>
fleet pull [--tag ...]                       # 軽いファイルを Mac の runs-index/ へ
fleet up | fleet down
```

`just fleet *args` を `python3 -m exp.lib.fleet` の別名にする。

### 4.7 本数の較正（最初に 1 回）

M13 では 16 本を並べると 1 本が CPU で 4 倍、CUDA で 5.3 倍遅くなった（メモリ帯域・クロック）。コア数の分だけ並べても総量が伸びるとは限らない。`fleet bench` は同じ短い run（M14 段 0、2 分）を 1, 4, 8, 16, 24, 30 本同時に回し、1 本の `frames_per_wall_second` と総量を出す。総量が頭打ちになる本数を `fleet.toml` の `cpus` にする。GPU のジョブ（大 + トレース）も同じように 4090 と 4060 で測る。

---

## 5. optuna

### 5.1 置き場

- 調整役は Mac の `tools/sweep`（ssh できるのは Mac だけ）。storage は `JournalStorage(JournalFileBackend("sweeps/<study>.log"))`（Mac のローカル、`.gitignore` に `sweeps/`）。`fleet` が NAS へ写して保全する。optuna-dashboard は同じファイルを読めるので、見たいときだけ `uvx optuna-dashboard sweeps/<study>.log`（依存には入れない）
- 1 trial = 1 つの設定 × 探索の seed（既定 2 本: 100, 101）。trial ごとにジョブを seed の数だけ投げる

### 5.2 目的と途中の報告

- M14 の目的: $J = \frac{1}{|S|}\sum_{s} \frac12 \sum_{h\in\{5,10\}} \log\frac{E^{\mathrm{val}}_h}{\mathrm{CV}^{\mathrm{val}}_h}$（最小化）。比の対数なので段と尺度に依らず、seed の平均が幾何平均になる。発散は `FAIL`（optuna は FAIL を比較に入れない。代わりに同じ設定の近くを避けるよう、`constant_liar` と組む）
- 途中の報告: run は val の評価（60 秒ごと）の行に `objective` を書く。調整役は 60 秒ごとに `fleet pull --study` で `evals.jsonl` を読み、**trial の全 seed がその時刻に達したら** seed の平均を `trial.report(value, step=t)` する
- 打ち切り: `MedianPruner(n_startup_trials=8, n_warmup_steps=300)`（仮想時間 5 分までは切らない）。`should_prune()` なら trial の全ジョブを `fleet kill`（`status=pruned`）して `tell(state=PRUNED)`
- **探索の run も 30 分で回す**。短いストリーム（10 分）で選ぶと、序盤に速い設定（大きい学習率）に偏る（M13 の AdamW: 300 秒で速いが後で悪くなった）。打ち切りが悪い trial を早く止めるので、良い trial だけが最後まで走る
- sampler: `TPESampler(multivariate=True, group=True, constant_liar=True, n_startup_trials=16, seed=...)`。並べる trial 数は CPU の空き（例 40 trial × 2 seed = 80 コア）

### 5.3 探索の空間（M14 の例、`spaces.py`）

```python
def m14_dynamics(trial) -> list[str]:
    kind = trial.suggest_categorical("optim", ["sgd", "adamw", "graft_adam"])
    lr = {"sgd": (1e-3, 0.3), "adamw": (1e-5, 1e-2), "graft_adam": (1e-3, 0.3)}[kind]
    out = [f"optim={kind}", f"optim.lr={trial.suggest_float('lr', *lr, log=True)}",
           f"optim.warmup={trial.suggest_categorical('warmup', [0.0, 60.0, 300.0])}",
           f"dynamics.residual={trial.suggest_categorical('residual', [True, False])}",
           f"dynamics.open_loop={trial.suggest_categorical('open_loop', [0.0, 0.25, 0.5])}"]
    if kind != "sgd":
        out.append(f"optim.tau_m={trial.suggest_float('tau_m', 0.03, 3.0, log=True)}")
    return out
```

固定するもの: B=128、`learn_interval=25e-3`（M6 で決めた）、物理、モデルの大きさ（小と大は別の study）、トレース（あり・なしは別の study。7.6）。M14 にはエンコーダも SIGReg も無いので、M10〜M13 で Adam 系が負けた理由（表現の崩れ）が消えている可能性がある。optimizer の比較はやり直す価値がある（map-history 8 節の教訓 1）。

### 5.4 BP で探索し、PC-ALM と BP で確認する

```bash
sweep run --study m14-s1-big-trace-bp --space m14_dynamics \
          --preset m14/s1 --set learner=bp dynamics=big dynamics.trace=true \
          --seeds 100,101 --trials 80 --parallel 40
sweep confirm --study m14-s1-big-trace-bp --top 3 --learners bp,pc --seeds 0,1,2 \
          --set phase=confirm                      # 3 x 2 x 3 = 18 jobs, tag confirm
sweep bracket --study m14-s1-big-trace-bp --best --param optim.lr --factors 0.5,1,2 \
          --learners bp,pc --seeds 0,1,2           # fairness check of the transfer
```

- 確認の run は **seed 0/1/2（探索の seed と別）**、**test のクリップ**で報告する。`phase=confirm` は val と test の両方を評価し、ckpt を 300 秒ごとに残す
- `bracket` は「BP で選んだ値を PC-ALM に移すのが公平か」を確かめる。PC-ALM の最良が ×1 でなければ、そのことを README に書き、両方とも bracket の最良で比べ直す
- 判定（README の規約）: 3 seed のすべてで同じ向きのときだけ支持

---

## 6. 追跡と集計

- **trackio は入れない**（3.14t に入らない。別の環境の trackio サーバに送ると、学習のプロセスに通信と依存が増える。orchestrator は AI なので、ファイルを直接読めば足りる）。optuna-dashboard は上の通り「見たいときだけ」
- `exp/lib/runs.py`（標準ライブラリだけ）:

```python
@dataclass(frozen=True) class RunRecord: dir: Path; meta: dict; config: dict; result: dict | None; evals: list[dict]
def index(root: Path) -> list[RunRecord]                 # meta.json がある run
def where(runs, **conds) -> list[RunRecord]               # "tags~=confirm", "config.learner.kind=pc", "code.dirty=false"
def final(run, split="test") -> dict                       # 最後の評価の行
def table(runs, *, rows: list[str], cols: list[str], value: str, seeds="seed", fmt=".2f") -> str
    # markdown. セルは seed 0/1/2 を " / " で並べる（規約）。発散は "発散"、dirty は "*"
def series(runs, key) -> list[tuple[float, float]]        # 時刻と値（図の入力）
```

- CLI: `python3 -m exp.lib.runs table --root runs-index --where tags~=m14 tags~=confirm --rows config.dynamics.name,config.dynamics.trace --cols config.learner.kind --value "roll10/cv10"`。出力を `results/m14.md` に貼る（表の上に生成したコマンドを書く）
- 図: run ごとの PNG（`viz/`）はその場で作る。README と `report.html` の図は、`runs.py` で JSON を抜き出し、`/experiment-report` の手順（SVG と素の JS）で描く。matplotlib は依存に入れない

---

## 7. M14 で書くコード（シグネチャ）とオラクル

### 7.1 物理のベクトル化（`data/physics.py`）

今の `balls.Video` は純 Python で 1 ストリーム 1 フレーム約 1 ms、B=128 で 1 フレーム約 0.1 秒（推定）。30 分のストリームで物理だけで約 1.5 時間かかり、M14 の小さいネットワークより遅い。torch でまとめて進める。

```python
@dataclass(frozen=True, kw_only=True)
class Physics: gamma0, gamma_amp, e0, e_amp, drag, period, conserve   # as now

class BallState(NamedTuple):
    pos: Tensor   # (B, K, 2), float64
    vel: Tensor   # (B, K, 2), float64, per second

def draw(n: int, radii: Sequence[float], gen: torch.Generator) -> BallState
def step_(s: BallState, radii: Tensor, h: float, gamma: float, e: float, drag: float,
          conserve: bool) -> None          # one semi-implicit Euler step, vectorized
def advance_(s: BallState, radii: Tensor, physics: Physics, t: float, seconds: float,
             substeps: int) -> None

class BallStreams:
    def __init__(self, n: int, *, radii, physics, frame, substeps, seed) -> None
    def next(self) -> Tensor               # states (B, 4K) float32, [x, y, vx, vy] per ball
    def fork(self) -> "BallStreams"        # a deep copy: the same future
    def state_dict(self) -> dict; def load_state_dict(self, d: dict) -> None

def render(states: Tensor, size: int, radii: Sequence[float]) -> Tensor   # (n, 1, size, size), as now
```

引き直しは EASY では起きないので実装しない（`reset` は最初のフレームだけ）。

### 7.2 観測（`data/observe.py`）

```python
def observe(states: Tensor, obs: Obs, balls: int) -> Tensor
    # (..., 4K) -> pos_vel: (..., 4K) / pos: (..., 2K), columns x0, y0, x1, y1, ...
def to_states(obs_raw: Tensor, obs: Obs, balls: int) -> Tensor
    # back to (..., 4K) with zero velocities for pos (render reads positions only)

@dataclass(frozen=True)
class Standardizer:
    mean: Tensor   # (d,)
    std: Tensor    # (d,)
    @classmethod
    def fit(cls, x: Tensor) -> "Standardizer"    # over all clips and frames of the calibration set
    def apply(self, x: Tensor) -> Tensor
    def invert(self, y: Tensor) -> Tensor
```

標準化は専用の**較正のクリップ**（seed 777、256 本）で決め、run の `baselines.json` に残す。val と test のクリップでは決めない（評価の集合を学習の前処理に使わない）。物理が一定なので、どの集合で決めても統計はほぼ同じ。

### 7.3 評価クリップ（`data/clips.py`）

```python
@dataclass(frozen=True)
class ClipSet:
    states: Tensor   # (n_clips, L, 4K), true states
    starts: Tensor   # (n_windows, 2): (clip, first frame) of each window of observed + max(H)
def make_clips(conf: BallsStateConf, *, split: Split, clips=256, length=128,
               observed=16, horizon=20, stride=30) -> ClipSet
    # split: CALIB (seed 777) / VAL (seed 1001) / TEST (seed 12345)
def windows(c: ClipSet, obs: Obs, std: Standardizer) -> tuple[Tensor, Tensor]
    # (n_windows, observed + horizon, d) standardized observations, and the true states
def contact_mask(states: Tensor, observed: int, horizons: Sequence[int]) -> Tensor
    # (n_windows, H, K) bool: whether ball k touches a wall between frame observed-1 and observed-1+h
    # (a sign change of a velocity component in the true trajectory)
```

M13 は 32 クリップ × 1 窓で、標本が少なかった。256 クリップ × 4 窓（開始 0, 30, 60, 90）= 1024 窓にし、窓をまたいだ標準誤差（`bootstrap_se`）を記録する。物理が一定なので「相」は要らない。

### 7.4 物差し（`eval/baselines.py`）

すべて**生の観測の空間で予測してから標準化**する（標準化した空間で差分を取らない）。

```python
def mean_pred(std: Standardizer, n: int, horizon: int, d: int) -> Tensor          # zeros (n, H, d)
def copy_pred(obs: Tensor, observed: int, horizon: int) -> Tensor                  # the last observation
def const_velocity(raw: Tensor, obs: Obs, balls: int, observed: int, horizon: int, frame: float) -> Tensor
    # p_hat_h = p_last + h (p_last - p_prev); for pos_vel, the velocity columns stay (p_last - p_prev) / frame
def const_accel(raw: Tensor, obs: Obs, balls: int, observed: int, horizon: int, frame: float) -> Tensor
    # p_hat_h = p_last + h v + h (h + 1) / 2 a, v = p_last - p_prev, a = p_last - 2 p_prev + p_prev2
def true_physics(states: Tensor, observed: int, horizon: int, ...) -> Tensor        # advance_ from the true state
def errors(pred: Tensor, target: Tensor, horizons) -> dict[int, float]             # mean over windows and dims
```

- 段 0 の等速の外挿では、速度の列も位置の差から作る（MILESTONES の定義）。段 0 は速度が見えているので「観測した速度で進める」版（`cv_obs`）も記録する（観測を使い切る、より強い基準）
- **等加速度の外挿（新規）**: 重力が一定で抵抗が無いので、壁に当たらない区間では厳密に当たる（下のオラクル）。モデルがこれを下回るには、跳ね返りを当てる必要がある
- 判定の目安（MILESTONES: 等速の外挿の 0.5 倍未満）は残す。そのうえで README には、等加速度の外挿との比と `@contact` の誤差を並べ、「重力を学んだ」「跳ね返りを学んだ」を分けて書く

### 7.5 動力学のノードと組み立て（`dyn/`）

```python
# dyn/modules.py
class DynamicsStack(nn.Module): first: MinGRU; rest: ModuleList[ResidualMinGRU]; d_s; d_state  # as now
class Top(nn.Module)                                                              # as now
@dataclass class DynamicsModules: step: DynamicsStack; readout: nn.Linear
def build_dynamics(conf: DynamicsConf, d_in: int, d_out: int, *, seed: int) -> DynamicsModules
    # torch.manual_seed(seed), then MinGRU(d_in, d_s), the layers, Linear(d_s, d_out)
    # (zeros when conf.residual): the one place that draws the initial weights, for PC, BP and offline

# dyn/nodes.py
class Source(Node):
    """pc.Input whose activity goes out on a send-only port: its peers cannot send credit."""
    def __init__(self, *, x: TensorSpec, y: TensorSpec, period: float, batch: Size, device: Device)
    prev: Port[None, Sample]; out: Port[Activity, None]; act_spec: DictSpec[Activity]

class Dynamics(Node):      # as now, but z = Port(recv=act_spec): no send spec at all
class DynamicsLayer(Node)  # as now
def compare_energy(readout, s, target, base=None) -> Tensor   # mean over the last axis; no kappa
class Compare(Node):       # target z = Port(recv=z_spec); sends credit to s only; no kappa, no target_weight

class DynamicsBlock(mujou.Network):
    """Dynamics -> DynamicsLayer x (n-1) -> Compare on an upstream activity; ports z_in (recv-only
    peers to connect to the upstream's send-only port), next (Prediction), error."""
    def __init__(self, mods: DynamicsModules, *, act_spec, tau, optimizer, learn_interval,
                 open_loop, open_cycle, residual, trace, device)
    def connect(self, upstream: Port[Activity, None]) -> None

# dyn/bp.py
class DynamicsBP(Node):
    """WorldBP without encoder, SIGReg or decoder: the same memory, trace, open loop, residual
    and LearnClock, learned by autograd in one node; recv-only input port."""

# dyn/nets.py
class StateWorldPC(mujou.Network): source: Source; block: DynamicsBlock
class StateWorldBP(mujou.Network): source: Source; world: DynamicsBP
def build_world(conf: RunConf, d_obs: int) -> StateWorldPC | StateWorldBP
def live_states(net) -> Tensor   # (B, d_state): the s_k of each stream, read by the host between steps
```

- `Compare` から `kappa` を外す: SIGReg の無い M14 で `(1-κ)` が付くと、学習率の意味がずれる
- `live_states` はホストが step の合間にノードの属性を読むだけ（評価の観測。学習には使わない）。M13 の `snapshot` の deepcopy と同じ扱い

### 7.6 オンラインのループ（`loop.py`）と分岐の評価

```python
def run_online(conf: RunConf, out: RunDir) -> dict
    # streams -> observe -> standardize -> Sample(x, y=zeros, mask=~reset, reset) -> Host.show
    # stream.jsonl each virtual second; evals each conf.eval.every (60 s) on VAL, at checkpoints also TEST;
    # viz at checkpoints; divergence: non-finite preq, or roll10 > conf.eval.diverge (10 x mean) -> status diverged
def fork_eval(net, streams: BallStreams, obs, std, horizons=(1, 5, 10, 20)) -> dict
    # s = live_states(net); future = streams.fork() advanced H frames; open-loop rollout from s
```

トレースのあり・なしは別の study にする（それぞれの最良で比べる）。大のモデルは費用が大きいので、トレースありで探索し、なしは同じ値で回して、そのことを README に書く。

### 7.7 オフラインの上限（`dyn/offline.py`）

```python
def stream_windows(conf: RunConf, frames: int, window: int = 32) -> Tensor   # (n, window, d) standardized, from BallStreams
def fit_offline(mods: DynamicsModules, windows: Tensor, *, steps: int, batch: int, lr: float,
                local: bool, observed: int = 16, gen: torch.Generator,
                every: int, on_eval: Callable[[int, DynamicsModules], None]) -> None
    # teacher forcing for observed-1 frames, then open loop; BPTT through all, or with local the
    # input and the fed-back prediction detached (the gradient stops at one frame, as the nodes);
    # Adam as pc.optim.AdamW (exp.lib.optim.adam), not torch.optim
```

`learner=offline` の run も同じ `evaluate_dynamics`・同じクリップ・同じ `evals.jsonl`（`t` の代わりに `step`）で書く。2×2 の表（BPTT / 1 フレームで切る × オフライン、トレース / なし × オンライン）で、「時間の局所性」と「オンラインの標本効率」の損を分けて README に書く。

### 7.8 見た目の確認（`eval/viz.py`）

```python
def frames_png(path: Path, true_states: Tensor, pred_states: Tensor, radii, *, size=32,
               columns=(0, 1, 2, 5, 10, 20), clips=4, scale=3) -> None
    # rows per clip: true frame / predicted (render of the prediction) / const-velocity (render)
def trajectory_png(path: Path, true_xy: Tensor, pred_xy: Tensor, cv_xy: Tensor, *, observed: int,
                   canvas=256, clips=4) -> None
    # the box, the observed path (grey), the true future (green), the prediction (red), CV (blue)
```

- 段 1・2 の予測は位置だけなので、`to_states` で速度 0 を挿して `render` する
- チェックポイントのたびに固定の 4 窓を描く。**接触を含む窓を必ず 2 つ入れる**（`contact_mask` で選ぶ）。跳ね返りの見た目を毎回確かめるため
- 確認の手順: 数字を README に書く前に `fleet pull` で PNG を取り、orchestrator が Read で見て「正しい向きへ動く」「壁で跳ね返る（すり抜けない・止まらない）」「薄れない」を確かめ、見たファイル名を results に書く

### 7.9 段 3（画像を観測に）

段 1・2 が通ってから決める（MILESTONES）。下準備として:

- 観測は `render(states)` を平らにした `(B, 1024)`、画素は標準化しない。`build_dynamics(d_in=1024, d_out=1024)`、`residual` の base は前のフレーム。誤差は画素の平均（`compare_energy` の `mean(-1)` のまま）
- 物差し: 写し・平均の画像（評価クリップの平均）・**等速の外挿を描いた画像**（真の状態から等速で外挿して `render`。状態を知っている基準だが、画像の空間で等速の外挿と比べるため）・等加速度の外挿を描いた画像
- 小さい玉は画素の二乗誤差で重みが小さい（面積が小さい）。玉ごとに描いた画像から玉の領域の誤差も記録する（M13 の「小さい玉の予測が消える」を数で拾う）

### 7.10 オラクルのテスト

| テスト | 理論から言えること | 許容 |
| -- | -- | -- |
| `data/test_physics.py` ベクトル版 = 参照 | 今の純 Python の `_step`（テストの中に参照として移す）と同じ軌跡。どちらも IEEE の倍精度で同じ順の演算 | 1e-12 |
| 同 エネルギー | `conserve` で $\tilde E$ が跳ね返りをまたいで保たれる（今のテストを移す） | 今と同じ |
| 同 分岐 | `fork()` した物理の未来 = 元の未来 | 完全一致 |
| `test_observe.py` | 段 0 の観測を標準化から戻すと `state` と一致。段 1・2 は位置の列だけ | 1e-6（float32） |
| `test_observe.py` 平均 | 較正の集合では標準化した観測の平均 0・分散 1（母分散で決める） | 1e-6 |
| `test_baselines.py` 等速 | 重力 0・抵抗 0 で壁に当たらない初期値なら誤差 0 | 1e-6 |
| 同 等加速度 | 一定の重力・抵抗 0・壁なしでは、半陰的 Euler の小刻みの位置が刻みの番号の 2 次式なので、フレームで間引いた位置も 2 次式。3 点の外挿は厳密 | 1e-6 |
| 同 真の物理 | `true_physics` の誤差は 0 | 0 |
| `test_clips.py` 接触 | 床へ落とした玉で、接触のフラグが立つ最初の $h$ が手計算と一致。壁から遠い玉は全て False | 完全一致 |
| `dyn/test_nets.py` 平衡点 | エンコーダの無いネットワークで、1 フレームを長く保持し `learn_interval=inf` にすると、各ノードの `lam` = `DynamicsBP` の同じ損失の autograd の $-\partial L/\partial h$（小・大、B=1 と B=3） | rtol 1e-3（今のテストと同じ） |
| 同 初期値 | `StateWorldPC` と `StateWorldBP` が同じ seed で同じ重み | 完全一致 |
| 同 トレース | 重みを止めたとき、トレースの勾配 = 入力を固定した BPTT（PC と BP） | 今のテストと同じ |
| 同 残差の初期 | `residual=True` の初期（$W_o=0$）の rollout の誤差 = 写しの誤差（すべての $h$） | 1e-7 |
| `dyn/test_nodes.py` credit なし | `Dynamics.z` と `Compare.z` は send の spec を持たない。`Source.out` は recv の spec を持たない（構造） | — |
| 同 局所性 | 数フレーム回した後、どのパラメータにも `.grad` が残らず、各ノードの重みは自分の optimizer だけで動く（`test_nodes.py:217` の形） | — |
| `dyn/test_offline.py` | 窓がストリームの最初をまたがない。`local=True` の勾配 = 1 フレームで切った損失の有限差分 | 1e-4（float64） |
| `test_loop.py` 決定性 | 同じ設定と seed で `evals.jsonl` が一致。ckpt から再開した続きが、通しの run と一致 | 完全一致（CPU） |
| 同 prequential | 記録する予測は、そのフレームを学ぶ前のもの（学習率を大きくしても、フレーム $k$ の記録はフレーム $k$ の学習に依らない） | 完全一致 |
| `test_dynamics.py` 分岐の評価 | 恒等の動力学（readout が写し）なら `fork{h}` = 写しの誤差 | 1e-6 |

「PC-ALM の重みの軌跡が BP と K フレームで近い」は理論のオラクルではない（許容誤差を理論から出せない）ので、テストではなく較正の run（M6 と同じ「3 桁一致」の確認）として README に書く。

---

## 8. M15 と M16 の部品

### 8.1 フレームのストリームと N フレームのまとまり

```python
# data/frames.py
class FrameStreams(Protocol):
    def next(self) -> tuple[Tensor, Tensor, Tensor]   # images (B, C, H, W) float, states (B, S) or (B, 0), reset (B,)
    def state_dict(self) -> dict; def load_state_dict(self, d: dict) -> None
class BallFrames(FrameStreams)        # BallStreams + render
class TourFrames(FrameStreams)        # B readers of the u8 videos, stream i from offset i * N / B

# jepa/nodes.py
class Chunker(Node):
    """Parameter-free. Keeps the last N frames of each stream (a ring buffer, register_state); when a
    frame is new it draws the masks from its generator and sends V views per stream as samples:
    view 0 the target (no mask), views 1.. the context (masked tubes replaced by zeros, with the mask
    as an extra input channel). A stream with fewer than N frames since its reset has mask False."""
    def __init__(self, *, frame_spec: TensorSpec, n_frames: int, views: int, masks: MaskConf,
                 tau, batch, seed, device)
    prev: Port[None, Sample]; next: MultiPort[Activity, Tensor]   # h: (B * V, N, C, H, W)
```

- 時間はサンプルの次元（mujou の規約）。$N$ フレームのまとまりは 1 つのサンプルの中の軸で、まとまりは毎フレーム 1 つずれる（因果: フレーム $k$ のまとまりは $k-N+1..k$）
- マスクはフレームの最初の更新で 1 回だけ引く（フレームの間は同じ問題を緩和する）。PC 版と BP 版は同じ `Chunker` を使うので、同じ seed で同じマスク
- V-JEPA と同じく、$N>1$ ではマスクを時間方向に伸ばした筒（tube）にする。隣のフレームから写して当てる近道を塞ぐ

### 8.2 JEPA のノード（mujou の局所性の中で）

```text
host ─Sample▶ Chunker ─(B·V samples)▶ Enc1 ─▶ Enc2 ─▶ … ─▶ EncL ─▶ JEPAHead ─ z_out (send-only) ─▶ [M16 の DynamicsBlock]
              (no params)             pc.Layer: blocks of the ViT        predictor + SIGReg
```

- エンコーダ `Enc1..L` は ViT のブロックを数個ずつ束ねた `pc.Layer`（サンプルごとに独立: attention はサンプルの中、LayerNorm は可、Dropout は不可）。ノードは少なく大きく（2〜4 ノード。1 周期 1.5〜2 ms の起動の律速と、credit が 1 周期に 1 ノードしか戻らないため。`learn_interval` 25 ms の間に head から Enc1 まで届く数にする）
- `JEPAHead`（自作ノード）: 受け取った `(B·V, T, d)` を `(B, V, T, d)` に戻し、
  - 予測器 $P_\phi$（小さい Transformer）が文脈のビューのトークンからマスクの位置の目標のトークンを予測: $L_{\mathrm{pred}} = \frac{1}{|M|}\sum_{t\in M}\lVert P_\phi(c)_t - \mathrm{sg?}(y_t)\rVert^2$
  - SIGReg を目標のビューの埋め込み（トークンと、下流へ送る $z$ の両方）に: $L = (1-\lambda) L_{\mathrm{pred}} + \lambda\,\mathrm{SIGReg}$
  - 自分の中の autograd で、各サンプルの埋め込みへの勾配を `prev` に返す（`# mujou: local-autograd`）。サンプルをまたぐ損失（ビューの組、SIGReg のバッチ）はこのノードの中に閉じ、下のノードはサンプルごとの credit を受けるだけ。1/B の規約は今の `SIGReg` ノード（`kappa * B * dS/dz_b`）と同じにする
  - `z_out = Port(send=act_spec)` で目標のビューの $z_k$（`(B, D)`、まずは CLS かプーリング。8.4 で選ぶ）を送る。送るだけのポートなので、下流から credit は戻れない
- **目標のエンコーダの重みを別に持たない**（EMA なし）。目標のビューも同じ `Enc1..L` を通る別のサンプル。重みの書き手は 1 ノード（不変条件 4）。stop-gradient（`sg`）の有無は head の中の切り替えで、両方を比べる
- 形をそろえるため、文脈のビューはトークンを落とさず、マスクのトークンを 0 にしてマスクの印の入力チャンネルを付ける（SimMIM の形）。計算は少し無駄だが、`(B·V, T, …)` の 1 つの spec で済む
- `JEPABP`（`jepa/bp.py`）: `Chunker` → 1 ノードに同じ ViT・予測器・損失。同じ `build_vision(conf, seed)` で同じ初期値。BP の基準と探索に使う

### 8.3 Walking Tours の取り込み

```python
# exp/lib/video_io.py
def ffmpeg_frames(path: Path, *, fps: float, width: int, height: int, crop: str = "center",
                  start: float = 0.0, duration: float | None = None, threads: int = 4) -> Iterator[Tensor]
class U8Writer: ...; class U8Reader: ...   # memmap of (N, 3, H, W) uint8 + index.json

# exp/video/data/walking_tours.py
MANIFEST: Path   # walking_tours/manifest.json: per video the license, the blur of faces/plates, the source URL
def prepare(video: str, out: Path, *, fps: float = 10.0, size: int = 64) -> None   # a fleet "prep" job per video
def eval_clips(root: Path, *, held_out: str, clips: int, length: int, seed: int) -> Tensor
```

- 4K・60 fps → 10 fps・64×64（中央の正方形）から始める。10 fps にすると 1 フレームの動きが 3 倍になり、写しの基準が少し弱くなる（M12: 実動画の写しは非常に強い）。16:9 を保つ 112×64 は 2 段目の候補
- 1 本ずつ fleet の `prep` ジョブ（`cpu: 8`）にし、出力は `/nas/geson/datasets/wt/64_10fps/`、学習の前に各ホストのローカルへ写す
- **評価には学習に流さない 1 本を取っておく**（例: 一覧の最後の街歩き）。学習のストリームと評価クリップが同じ動画だと、継続学習の記憶と汎化が混ざる
- マニフェストに無い動画は `prepare` が拒む（ライセンスとぼかしの確認を強制。MILESTONES の要求）。テストは ffmpeg で作った色の合成の mp4 だけ（`test_egodex.py` の手を移す）

### 8.4 M15 の評価（`eval/vision.py`）と、M16 への受け渡しの条件

```python
def ball_probe(z: Tensor, states: Tensor, balls: int, half: Tensor) -> dict   # R^2 of x, y (and vx, vy) of each ball
def thumbnail_probe(z: Tensor, images: Tensor, half: Tensor, size: int = 8) -> float   # R^2 of the 8x8 grey thumbnail
def vision_metrics(encode: Callable, clips) -> dict   # rank, z_std_min, ball_probe, thumbnail_probe
```

- 合成の玉: **玉ごとの位置の probe の $R^2$**（下流へ送る $z$ から。3 つの玉のどれも）、$N \ge 2$ なら速度も。実効ランク、次元ごとの標準偏差の最小（崩壊の検知）
- Walking Tours: 縮小画像（8×8）の probe の $R^2$、実効ランク、後から学ぶデコーダの再構成
- 基準: **初期値のままの（学習していない）同じ ViT** と、**生の画素**からの probe。自己教師の表現の評価で欠かせない比較で、これを超えなければ学んだとは言えない
- 下流に送る $z$ の形（CLS / 平均のプーリング / 少数のレジスタのトークンの連結）は、玉ごとの probe で選ぶ。**M15 → M16 の受け渡しの条件**（目安）: 3 つの玉すべての位置が $R^2 \ge 0.9$（test、3 seed、BP と PC-ALM）、デコーダの再構成で 3 つの玉が見える

### 8.5 後から学ぶデコーダ（`eval/posthoc.py`、別のジョブ）

```python
def decoder(d_z: int, shape: tuple[int, int, int]) -> nn.Module    # H, W any multiple of 4
def fit_decoder(z: Tensor, images: Tensor, *, steps=2000, batch=256, lr=1e-3, gen) -> nn.Module
    # exp.lib.optim.adam (pc.optim.AdamW), no torch.optim
def decode_run(run_dir: Path, *, every_ckpt=True, frames=16384, device) -> None
    # per ckpt: rebuild the vision net, load the modules, encode the training frames sampled
    # uniformly over the whole stream (not its first seconds), fit, decode the eval clips;
    # write post/decode/<t>.png and metrics.jsonl (recon, img_roll{h}, pix_copy{h}, pix_mean{h})
```

- 学習のフレームは**ストリーム全体から一様に**取る（今の `stream_frames` は B=128 で先頭の約 4 秒しか取らない）。評価クリップとは別
- `pix_mean` は評価クリップの平均の画像に統一する（M14 と同じ定義。今は学習のフレームの平均）
- fleet の `posthoc` ジョブ（`after: [<run の job>]`、`gpu: any`）。M15・M16 の確認の run は、ランチャーが自動で後ろに付ける

### 8.6 M16（`joint.py`）

```python
class JointPC(mujou.Network):  chunker; enc1..encL; head: JEPAHead; block: DynamicsBlock   # block.connect(head.z_out)
class JointBP(mujou.Network):  chunker; vision: JEPABP; world: DynamicsBP                  # world reads vision.z_out
def build_joint(conf: RunConf) -> JointPC | JointBP   # build_vision(seed) then build_dynamics(seed + 1)
```

- 動力学は M14 の `DynamicsBlock` をそのまま上流だけ替えて使う。上流のポートが送るだけなので、credit がエンコーダへ戻ることは接続の型で起こらない
- $z$ の尺度: SIGReg が $z$ の射影をおよそ標準正規にするので、M14 の標準化した観測とほぼ同じ尺度になる（M14 の学習率をそのまま持ち込みやすい理由）。$z$ はエンコーダの学習とともに動く（非定常な目標）ので、固定の標準化は入れない
- 対照: **凍結したエンコーダ**（M15 の最後の ckpt を読み、エンコーダのノードを `learn_interval=inf` にする）+ 動力学。同時の学習と比べ、「エンコーダが動くこと」の損を分ける。M14 の状態の観測の結果（上限の参考）と、M14 段 3（画素を観測）の結果も同じ表に並べる
- 判定: 潜在で `roll{h}` を写し・平均と比べ、デコーダで画像に戻して `img_roll{h}/pix_mean{h}` と写しと比べる（5・10・20 フレーム先）。$N$ フレームのまとまりは隣と $N-1$ フレームを共有するので、潜在の写しは非常に強い。画像の空間の判定を主にする

### 8.7 M15・M16 のオラクル

| テスト | 理論から言えること |
| -- | -- |
| `jepa/test_masks.py` | マスクの比率・ブロックの数・筒が時間方向に一定、同じ seed で同じ |
| `jepa/test_nodes.py` 平衡点 | `Chunker` → `Enc1..L` → `JEPAHead` で、学習を止めて 1 フレームを保持すると、各エンコーダのノードの `lam` = `JEPABP` の同じ損失の autograd の credit（サンプルごと、ビューごと） |
| 同 ビューの独立 | エンコーダのノードの出力は、他のサンプルの入力に依らない（入力の 1 サンプルを変えて他が変わらない） |
| 同 sg | stop-gradient ありで、目標のビューのサンプルが受ける予測の項の credit は 0（SIGReg の項だけ） |
| `jepa/test_bp.py` 初期値 | `VisionPC` と `VisionBP` が同じ seed で同じ重み |
| `test_joint.py` **credit なし** | 同じ seed・同じデータで K フレーム回したとき、`JointPC` のエンコーダの重みと `VisionPC` のエンコーダの重みがビット単位で同じ（CPU）。BP 版も同じ |
| `eval/test_vision.py` | 状態の線形写像を $z$ にすると probe の $R^2$ = 1、乱数の $z$ なら約 0 |
| `eval/test_posthoc.py` | $z$ が画像を決める合成の対で、デコーダが学べる（今のテストを移す） |

---

## 9. 実装の手順（作業の包み）

方針: **インフラの核と M14 段 0 を最初に通し、フリートで実験を始めてから残りを並べる**。各包みは担当のディレクトリが重ならないように切る。`schema/__init__.py` と `conf/run.yaml` は WP0 が全部の群の空の選択肢まで作り、以降は各包みが自分の群のファイルだけを足す。

| WP | 内容 | 担当するファイル | 依存 |
| -- | -- | -- | -- |
| **WP0** 核（1 人、最初） | 旧コードとテストの削除（1 コミット）、`uv add hydra-core`、`schema/` の骨組みと全群の登録、`conf/run.yaml`、`exp/lib/{conf,rundir,ckpt,png}.py`、`MinGRUTrace` の `Stateful` 化、`optim.py` の削減と `adam`、`tests/test_gil.py`、compose.yaml に `/nas/geson` のマウント、`.gitignore`（`sweeps/`, `runs-index/`）、OmegaConf の `kw_only`/`frozen` の確認 | `exp/lib/` の上記、`exp/video/schema/`、`exp/video/conf/run.yaml`、`pyproject.toml`、`compose.yaml` | — |
| **WP1** フリート | `exp/lib/fleet/*`、`fleet.toml`、`just fleet`、`tests/lib/fleet/*`。偽のコマンド（`python -c "import time; time.sleep(1)"`）で agent の起動・停止・lost・再開を試す | `exp/lib/fleet/`、`fleet.toml`、`justfile` | WP0 の `rundir` の status の約束だけ |
| **WP2** M14 データと物差し | `data/{physics,observe,clips}.py`、`eval/baselines.py`、`schema/data.py`、`conf/data/s{0,1,2}.yaml` とテスト | `exp/video/data/`、`exp/video/eval/baselines.py` | WP0 |
| **WP3** M14 ノードと組み立て | `dyn/{modules,nodes,bp,nets}.py`、`schema/{dyn,learn}.py`、`conf/{dynamics,learner,optim}/`、平衡点・初期値・トレース・局所性のテスト | `exp/video/dyn/`（offline を除く） | WP0 |
| **WP4** M14 の結合 | `main.py`、`loop.py`、`eval/{dynamics,viz}.py`、`dyn/offline.py`、`schema/{run,eval}.py` の中身、`conf/{eval,phase,experiment/m14}/`、`tests/video/test_{main,loop}.py` | 上記 | WP2, WP3 |
| → **最初の実験** | `fleet bench` → M14 手順 1（`m14/baselines`: 物差しの表、等速・等加速度の外挿の PNG）→ README に判定基準を書く → 手順 2（`m14/offline` 段 0〜2、BPTT と 1 フレーム）→ 段 0 の BP の手の格子（`fleet submit` で lr × optimizer） | — | WP1, WP4 |
| **WP5** optuna | `tools/sweep/*`（`run`, `confirm`, `bracket`、偽の `evals.jsonl` で集約と打ち切りを試す） | `tools/sweep/` | WP1 のジョブの形、3.2 の schema |
| **WP6** 集計 | `exp/lib/runs.py` と CLI、テスト | `exp/lib/runs.py` | 3.2 の schema |
| → **M14 段 0→1→2** | 段ごとに BP の study（小・大 × トレースあり/なし）→ confirm と bracket → PNG を見る → results/m14.md | — | WP5, WP6 |
| **WP7** Walking Tours | `exp/lib/video_io.py`、`data/{frames,walking_tours}.py`、マニフェスト、`prep` ジョブ | 上記 | WP0（いつでも始められる） |
| **WP8** JEPA | `jepa/*`、`schema/jepa.py`、`conf/encoder/`、`conf/experiment/m15/`、平衡点・ビューの独立・sg のテスト | `exp/video/jepa/` | WP0、WP4 の `loop.py`（フレームのストリームを流す形を足す: loop の所有者と調整） |
| **WP9** 視覚の評価 | `eval/{vision,posthoc}.py`、`posthoc` ジョブの自動の後付け | 上記 | WP8 の `build_vision` の約束 |
| → **M15** | 玉の $N=1$ → $N=4$ → WT の $N=1$ → $N=8$。BP で探索、sg あり/なし、受け渡しの条件 | — | WP7〜9 |
| **WP10** M16 | `joint.py`、`conf/experiment/m16/`、`test_joint.py`（credit なしのビット一致）、凍結の対照 | 上記 | M14 の設定、M15 の設定 |
| **WP11** M14 段 3 | `dyn/nets.py` の `PixelWorld*`、画素の物差し（等速・等加速度の外挿を描いた画像、玉ごとの領域の誤差） | `dyn/nets.py` の追記、`eval/baselines.py` の追記 | 段 1・2 が通ってから |

- WP1・WP2・WP3・WP7 は WP0 の直後に並べられる（ディレクトリが別）。WP5・WP6 は WP4 と並べられる（3.2 の schema を WP0 の時点で文書として固めておく）
- 衝突しやすいのは `loop.py`（M14 と M15 の両方が使う）だけ。WP4 が「ストリーム・ネットワーク・評価器」の 3 つを受け取る汎用の形で書き、WP8 は新しい評価器とストリームを足すだけにする
- ブランチ: インフラの再編を 1 本の PR（`refactor/2026-11-03/fleet-and-hydra`、WP0〜WP6）、M14 を 1 本（`exp/<日付>/continuous-stream-m14-dynamics`）、M15・M16 をそれぞれ 1 本

### 9.1 mujou に送る候補（必要が出たときだけ）

- `pc.Source`（送るだけのポートを持つ観測のノード）: M14・M16 で同じ形が 2 回要れば提案する
- `Stateful` のトレースのような「ノードの属性ではない状態」を集める仕組みは今の `Stateful` で足りる（追加不要の見込み）
- JEPA の head のような多入力・サンプルをまたぐ損失のノードは `pc.functional` で書ける（`examples/pc_merge.py` の形）。足りないものがあれば skill `mujou-upstream`

---

## 10. 危険と未決事項

1. **M14 段 1 の表現力**: 1 ノードの minGRU はゲートも候補も入力だけの関数で、状態はゲート付きの漏れ積分器の束。位置の差（速度）は時定数の違う積分器の差でおよそ作れるが、跳ね返りの非線形は層の間の GELU（大）に頼る。オフラインの上限が先に答える（通らない段はオンラインの前に原因を切り分ける。MILESTONES の手順 2）
2. **評価の暖機**: 16 フレームで $s$ が定常に達するか。クリップの評価と分岐の評価の差で見る。差が大きければ、判定の主を分岐の評価に替えるかを README で決める（回す前に）
3. **PC-ALM + 大 + トレースの時間**: M14 はエンコーダが無く、`Dynamics` のトレースは小さい（$d_x \le 6$）が、`DynamicsLayer` のトレースは B=128・$d=512$ で 1 本 268 MB、毎フレーム $O(B d^2)$。GPU のジョブにする。`fleet bench` で見積もってから確認の本数を決める
4. **M15 の情報の捨て方**: 8 節の通り。sg のあり・なし、マスクの比率、トークンの埋め込みへの SIGReg を比べ、玉ごとの probe で判定する。どれも通らないときは、マスクの予測の代わりに「目標を固定したランダムな射影」などの別の視覚の損失を検討する（M16 で動力学から credit を戻す案は出さない）
5. **hydra の structured config の制約**（`kw_only`・`frozen`・派生クラスの群・`StrEnum`）は WP0 で確かめる。駄目なものは 2.2 の代替（frozen をやめる等）で進める
6. **コンテナの GPU**: compose.yaml に GPU の予約が無い。過去の run は GPU を使えていたが、`fleet hosts` が各コンテナの `torch.cuda.device_count()` を出して確かめる
