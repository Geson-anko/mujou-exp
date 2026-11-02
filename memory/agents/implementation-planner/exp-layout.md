# 実験コードの再編（experiments/ → exp/、continuous_stream → video、exp/lib/ の新設）

## 概要

モジュールパスを短くし、実験に閉じない補助コードを `exp/lib/` にまとめる。`experiments/` → `exp/`、`experiments/continuous_stream/` → `exp/video/`、再利用できる部品は `exp/lib/` へ移す。破壊的変更は許されているが、学習の振る舞い・乱数の消費順・`Config` のフィールド・run 名は変えない。過去の結果を再現するコマンドは、パスだけ変えて同じ意味を保つ。

## 仮説と判定基準

なし（リファクタ）。完了条件は次の 4 つで、作業前に固定する。

1. `just run` がグリーン（テスト数は移動前と同じか、分割で増えるだけ。テストを消さない）
2. 移動の前後で同じ設定の run の `metrics.jsonl` がバイト単位で一致する（ステップ 0 と最終ステップを参照）
3. `grep -rn "experiments\|continuous_stream" --exclude-dir=mujou --exclude-dir=.venv --exclude-dir=.venv-devcontainer --exclude-dir=.git --exclude-dir=__pycache__ --exclude-dir=runs .` の結果が、末尾の「意図的に残すもの」だけになる
4. `exp/lib/` が `exp/video/` を import していない（`grep -rn "exp.video" exp/lib tests/lib` が 0 件）

## 新レイアウト（移動対応表）

`lib` に置く基準は次の 2 つ。両方を満たすものだけを移す。

- 実験の `Config`・データ・モデルに依存しない。mujou の公開 API・torch・標準ライブラリだけで書ける
- 別の実験（たとえば段 3 の EgoDex を別実験にした場合）が、そのまま import できる

| 旧 `experiments/continuous_stream/` | 新                                                                                    | 判断と根拠                                                                                                                                                                                                                                                                                                                                  |
| ----------------------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `__init__.py`（1 行の docstring）   | `exp/video/__init__.py`                                                               | そのまま                                                                                                                                                                                                                                                                                                                                    |
| `experiments/__init__.py`（空）     | `exp/__init__.py`                                                                     | そのまま                                                                                                                                                                                                                                                                                                                                    |
| （新規）                            | `exp/lib/__init__.py`                                                                 | 空。docstring 1 行（"Code shared by the experiments: ..."）                                                                                                                                                                                                                                                                                 |
| `record.py`                         | `exp/lib/record.py`                                                                   | **lib**。`Config` への依存は `config.name` と `config.to_dict()` だけ（record.py:47-52）。引数の型を Protocol `RunConfig` に変える。`REPO_ROOT = parents[2]` は `exp/lib/record.py` でもリポジトリのルートを指すので、変更しない                                                                                                           |
| `host.py`                           | `exp/lib/host.py`                                                                     | **lib**。mujou だけに依存する（`ManualRun`、`Run`、`Port`）。中身は変えない                                                                                                                                                                                                                                                                 |
| `config.py` の `_from_dict`（131-152） | `exp/lib/config.py` の `from_dict`                                                    | **lib**。入れ子の frozen dataclass を JSON から戻す汎用処理。名前の `_` を外して公開する。`Config.from_dict` はこれを呼ぶ                                                                                                                                                                                                                  |
| `config.py` の残り                  | `exp/video/config.py`                                                                 | 実験固有（`TimeConfig` 〜 `Config`）。フィールドと既定値は変えない（過去の `runs/*/config.json` を読み戻せるように）。`optim.py` の `from_config` をここへ移し、`make_optimizer` に改名する                                                                                                                                               |
| `optim.py`（`from_config` 以外）    | `exp/lib/optim.py`                                                                    | **lib**。`pc.optim.Optimizer` の実装で、`OptimConfig` に依存するのは `from_config`（472-506）だけ。`Correlation` も汎用の診断。`git mv` で移し、`from_config` だけを抜く。H2 は棄却されたが、比較の対照として今後も使う                                                                                                                  |
| `optim.py` の `from_config`         | `exp/video/config.py` の `make_optimizer`                                             | `OptimConfig` を解釈する実験固有の部分。`config.from_config` は名前が重複して読みにくいので改名する                                                                                                                                                                                                                                         |
| `stream0.py`                        | `exp/lib/linear_stream.py`                                                            | **lib**。torch だけに依存する、ドリフトする線形回帰のストリーム。optimizer の検証用の汎用の試験台。`stream0` は「段 0」という実験内の呼び名なので、クラス名に合わせて改名する                                                                                                                                                              |
| `nodes.py` の SIGReg 節（251-364）  | `exp/lib/sigreg.py`                                                                   | **lib**。`T_POINTS`、`epps_pulley`、`sigreg`、`insert_`、`SIGReg` ノード。他の節を参照しない独立した塊で、`z` を正則化するノードとしてどの実験でも使える                                                                                                                                                                                   |
| `nodes.py` の `MinGRU`（41-55）     | `exp/lib/mingru.py`                                                                   | **lib**。依存のない `nn.Module`                                                                                                                                                                                                                                                                                                             |
| `nodes.py` の残り                   | `exp/video/nodes.py`                                                                  | 実験固有。`Dynamics`（フレームの記憶を持つ世界モデルの部品）、`compare_energy`、`Compare`（$\kappa$ 込みのエネルギー）                                                                                                                                                                                                                       |
| `evaluate.py` の `effective_rank`・`probe_r2`（98-129） | `exp/lib/metrics.py`                                                                  | **lib**。テンソルだけを受け取る指標                                                                                                                                                                                                                                                                                                         |
| `evaluate.py` の残り                | `exp/video/evaluate.py`                                                               | 実験固有。`Snapshot`（`Network` 依存）、`run_states`・`rollout_errors`（$(z, s)$ の再帰モデル前提）、`evaluate`（`Frame` の state/params 前提）                                                                                                                                                                                             |
| `stage0.py` の `run_all`（84-105）  | `exp/lib/sweep.py`                                                                    | **lib**。設定ごとにサブプロセスで `-m <module> --config --out` を回す掃引。モジュール名を引数にする（現状は文字列 `"experiments.continuous_stream.run"` の直書き、stage0.py:95）。stage1・stage2・collapse が stage0 から import しているスクリプト間の依存も解ける                                                                      |
| `stage0.py` の残り                  | `exp/video/stage0.py`                                                                 | 実験固有。`RUN_MODULE = "exp.video.run"` を足し、stage1・stage2・collapse は `SEEDS` と一緒にこれを import する                                                                                                                                                                                                                              |
| `video.py`                          | `exp/video/balls.py`                                                                  | 実験固有。改名の理由は次節                                                                                                                                                                                                                                                                                                                  |
| `bp.py`, `model.py`, `run.py`, `stage1.py`, `stage2.py`, `collapse.py` | `exp/video/` に同名で                                                                 | 実験固有。import の書き換えだけ                                                                                                                                                                                                                                                                                                             |
| `README.md`, `PLAN.md`, `MILESTONES.md`, `report.html` | `exp/video/` に同名で                                                                 | 文書の書き換え範囲は後述                                                                                                                                                                                                                                                                                                                    |
| `experiments/README.md`             | `exp/README.md`                                                                       | 書き直す（後述）                                                                                                                                                                                                                                                                                                                            |

lib に**移さない**もの（汎用部を切り出すと複雑になる、または実験固有）:

- `model.encoder_modules`・`standardize_`: `standardize_` が合成動画（`Video`）に依存する。CNN の形も実験の設計
- `bp.py`: `bp_chain` は汎用に見えるが、ほぼ `pc.Chain(nn.Sequential(...))` で、`WorldBP` と同居させたほうが読みやすい
- `stage1.summary`・`stage2.summary/judge`・各スクリプトの `done` 走査（`done.parent.name.split("-", 3)[3]` が 4 か所）: 重複はあるが、指標名が実験固有。抽出しない（重複は許容）

### `exp/` の最終形

```text
exp/
  README.md  __init__.py
  lib/
    __init__.py  config.py  record.py  sweep.py  host.py
    linear_stream.py  optim.py  sigreg.py  mingru.py  metrics.py
  video/
    README.md  PLAN.md  MILESTONES.md  report.html  __init__.py
    config.py  balls.py  nodes.py  model.py  bp.py  evaluate.py
    run.py  stage0.py  stage1.py  stage2.py  collapse.py
```

## パッケージ名 `video` と同じ名前のモジュールの扱い

推奨: **`video.py` → `balls.py`**（合成の「箱の中のボール」）。クラス名（`Video`、`Physics`、`Frame`）と関数名は変えない。

- `exp.video.video` を避けられる。段 3 で `egodex.py`（実写の動画）が同じパッケージに入る予定（ルート README.md:31、PLAN.md:336）なので、`video.py` は「どの動画か」が曖昧になる
- `DataConfig.kind == "video"` は変えない（`config.json` の互換）。`kind` の値はモジュール名ではなく「動画のストリーム」という意味なので、そのままでよい
- クラス `Video` を `Balls` に改名する案は採らない。差分が `model.py`・`run.py`・テストに広がるだけで、読みやすさはあまり変わらない

## 公開インターフェース案

振る舞いは変えない。シグネチャが変わるのは 3 つだけ（★）。

- `exp/lib/config.py`
  - ★ `class RunConfig(Protocol)`: `@property def name(self) -> str` / `def to_dict(self) -> dict[str, Any]`（frozen dataclass のフィールドは読み取り専用なので、pyright のために `name` は property で宣言する）
  - `def from_dict[T](cls: type[T], d: dict[str, Any]) -> T`（旧 `_from_dict`。挙動は同じ。未知のキーは `TypeError`、float の注釈なら JSON の int を float に変える）
- `exp/lib/record.py`
  - `REPO_ROOT: Path`, `git_sha(repo: Path) -> str`, `git_dirty(repo: Path) -> bool`
  - ★ `Recorder(config: RunConfig, root: Path = REPO_ROOT / "runs")`（旧: `config: Config`）。`log(time: float, **values: Any) -> None`、`close()`、context manager は同じ
- `exp/lib/sweep.py`
  - ★ `def run_all(module: str, configs: Sequence[RunConfig], out: Path, jobs: int) -> None`（旧: `run_all(out, jobs, every=None)`。`None` なら stage0 の `configs()` という既定は stage0 に依存するので廃止）。設定ファイルは `json.dumps(c.to_dict(), indent=2)` で書く（`Config.to_json` と同じバイト列）。`done` の判定、ログの出力、`cwd=REPO_ROOT` は同じ
- `exp/lib/host.py`: `Host(run: ManualRun, *, frame: float, period: float)`, `RealtimeHost(run: Run, *, frame: float)`（変更なし）
- `exp/lib/linear_stream.py`: `LinearStream(*, d, tau_x, frame, noise, mean_std=1.0, jitter=0.3, iid=False, seed)`（変更なし）
- `exp/lib/optim.py`: `BoundDirection`, `DirectionOptimizer`, `Grafted`, `Whitened`, `Orthogonalized`, `Projected`, `AdamDirection`, `LAGS`, `Correlation`（変更なし。`__all__` から `from_config` を外す）
- `exp/lib/sigreg.py`: `T_POINTS`, `epps_pulley(x: (N, M), t=T_POINTS) -> (M,)`, `sigreg(z: (D,), buffer: (N-1, D), directions: (M, D)) -> ()`, `insert_(buffer: (capacity, D), count: int, z: (D,), g: torch.Generator) -> int`, `SIGReg(*, act_spec, kappa, directions, capacity, tau, min_samples=16, seed=0, device="cpu")`（変更なし）
- `exp/lib/mingru.py`: `MinGRU(d_z: int, d_s: int)`、`forward(z: (..., d_z), s_prev: (..., d_s)) -> (..., d_s)`（変更なし）
- `exp/lib/metrics.py`: `effective_rank(z: (n, D)) -> float`, `probe_r2(train_x: (n, F), train_y: (n, K), test_x, test_y, ridge=1e-3) -> (K,)`（変更なし）
- `exp/video/config.py`: `make_optimizer(c: OptimConfig) -> pc.optim.Optimizer`（旧 `optim.from_config`。中身は同じ）。`Config.from_dict` は `exp.lib.config.from_dict` を呼ぶ
- `exp/video/stage0.py`: `RUN_MODULE: str = "exp.video.run"`。`run_all` の定義を消し、`main` は `run_all(RUN_MODULE, configs(), args.out, args.jobs)`
- `exp/video/stage1.py`・`stage2.py`・`collapse.py`: `run_all(RUN_MODULE, <configs>, args.out, args.jobs)`

docstring で `PLAN.md` を指している箇所のうち、lib に移るもの（`optim.py` の冒頭「PLAN.md, section 5」、`SIGReg` の「PLAN.md, section 3」）は `exp/video/PLAN.md` と書き直す。`evaluate.py:3` の ``video.eval_clips`` と `run.py:102` の ``video.ordered`` は ``balls.eval_clips``・``balls.ordered`` に直す。

## 理論上の対応

なし。ノードのコード（`Dynamics`、`Compare`、`SIGReg`）は**中身を変えずに**移すだけで、ノードの分け方・メッセージ・局所的な autograd（`# mujou: local-autograd`）は変わらない。`SIGReg` を別モジュールに移しても、隣接ノードからのメッセージ以外を読む経路は増えない。

## mujou に足すもの

なし。

- 今回 `exp/lib/` に置くものは、実験を回すための補助（記録・掃引・ホストのループ・設定の JSON 化・評価指標）と、研究中で mujou に入れるほど固まっていない部品（optimizer の候補、SIGReg、minGRU、試験用ストリーム）。mujou の抽象（Node・Port・Run・`pc.*`）の不足を補う代用品ではない
- 将来の候補として 1 行だけ書いておく: `Host` は `mujou.testing.run` をフレーム単位で進めるループで、`mujou.testing` に入る可能性がある。今回は動かさない

## 方針の棲み分け（文書の書き直し案）

CLAUDE.md の「プロジェクト概要」と `exp/README.md`、`refactor-conventions` の「実験のパッケージ構成」に、次の 3 層を書く。

1. **mujou に足す**: ノード・ポート・実行・学習則・optimizer の基盤など、mujou の利用者一般が使う機能と、mujou の抽象に足りないもの。mujou の規約（オラクルテスト・docs・CHANGELOG）で保守する価値があるもの。「実験側にライブラリの代用品を溜めない」はこの層に対する原則として残す
2. **`exp/lib/` に置く**: 2 つ以上の実験が使う、または特定の実験に閉じない、実験を回すための補助コード（記録・掃引・ホスト・設定・指標）と、研究中の部品。mujou の公開 API・torch・標準ライブラリだけに依存し、`exp/<name>/` を import しない。mujou の抽象に属すると分かったら skill `mujou-upstream` で mujou に移し、lib からは消す
3. **`exp/<name>/` に置く**: その実験の設定・データ・モデル・スクリプトと、実験の都合だけの暫定物。他の実験のパッケージを import しない（lib は import してよい）

迷ったときの判定は「mujou の利用者に説明できる機能か（→ mujou）、実験の運用の道具か（→ lib）」。判断が割れたらユーザーに確認する、という既存の一文は残す。

同じ内容を 1 行で `memory/project_exp_lib.md`（新規、`MEMORY.md` から 1 行リンク）に置く。`memory/project_purpose.md` の「How to apply」には「実験横断の補助コードは `exp/lib/`（[[project-exp-lib]]）」と 1 文だけ足す。

## 実装ステップ

各コミットの前に `just run` を通す（CLAUDE.md「Git 運用」）。1 コミット 1 関心事で、次の 5 つに分ける。

0. **基準の採取**（コミットしない）: 移動前の HEAD で、スクラッチに 2 つの小さな設定を作り、run を 1 本ずつ回す。
   - `Config(name="ref-lin", data=DataConfig(kind="linear", seconds=2.0), eval_interval=1.0)`
   - `Config(name="ref-vid", data=DataConfig(kind="video", seconds=2.0), eval_interval=1.0)`
   - `uv run python -m experiments.continuous_stream.run --config <path> --out <scratch>/before`。`metrics.jsonl` を控える
1. **`refactor(exp): move experiments/continuous_stream to exp/video`**
   - `git mv experiments exp` → `git mv exp/continuous_stream exp/video`、`git mv tests/continuous_stream tests/video`
   - `exp/` と `tests/video/` の `experiments.continuous_stream` を `exp.video` に一括置換する（import、stage0.py:95 と test_record.py:55 の文字列、各スクリプトの docstring のコマンド）
   - `pyproject.toml:33,47`（コメントと pyright の `include = ["exp", "tests"]`）、`justfile:26`、`.pre-commit-config.yaml:58,60`（`^(memory/|\.claude/|exp/[^/]+/)`）
   - 追跡されていない `experiments/**/__pycache__` が残るので、`git status` で `experiments/` に追跡中のファイルが無いことを確かめてから消す
2. **`refactor(video): rename video.py to balls.py`**: `git mv exp/video/video.py exp/video/balls.py`、`git mv tests/video/test_video.py tests/video/test_balls.py`。import と docstring の ``video.`` 参照（evaluate.py:3、run.py:102）を直す
3. **`refactor(exp): move the code shared by experiments to exp/lib`**
   - `git mv` で履歴を残すもの（ファイルの大半が移る側を `git mv` する）: `record.py`・`host.py` → `exp/lib/`、`stream0.py` → `exp/lib/linear_stream.py`、`optim.py` → `exp/lib/optim.py`（`from_config` を `exp/video/config.py` の `make_optimizer` へ移す）
   - 新規ファイルに切り出すもの（元のファイルが残る側）: `config._from_dict` → `lib/config.py`（+ `RunConfig`）、`stage0.run_all` → `lib/sweep.py`、`nodes` の SIGReg 節 → `lib/sigreg.py`、`nodes.MinGRU` → `lib/mingru.py`、`evaluate` の 2 関数 → `lib/metrics.py`
   - `Recorder` の型を `RunConfig` にする。`run_all` の呼び出し 4 か所（stage0/1/2、collapse）を新しいシグネチャにする
   - `stage2.py:43` の未使用の `BASELINE = "sgd"` を消す（どこからも参照されない）
   - テストを次節のとおり移す・分ける
   - ステップ 0 と同じ設定を `uv run python -m exp.video.run` で回し、`metrics.jsonl` を `diff` する（一致しなければ止めて原因を調べる）
4. **`docs: follow the move to exp/`**: CLAUDE.md、ルート README.md、`exp/README.md`、`exp/video/` の文書、`memory/`（範囲は後述）
5. **`chore(claude): follow the move to exp/`**: `.claude/` の更新（skill `edit-dot-claude` の手順。対象は後述）

最後に完了条件 1〜4 を確かめる。

## テスト観点

テストの中身（アサーション）は変えない。import と配置だけを変える。lib のテストは `exp.video` を import しない。

| 旧 `tests/continuous_stream/`                                   | 新                                                                                                       |
| --------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `__init__.py`                                                   | `tests/video/__init__.py`、新規 `tests/lib/__init__.py`                                                  |
| `test_record.py` の `TestRecorder`                              | `tests/lib/test_record.py`（`git mv`）。`Config` の代わりに、テスト内の小さな frozen dataclass（`name` と `to_dict` を持つ）を渡す。`Config.from_dict(head["config"]) == config` は `head["config"] == config.to_dict()` にする |
| `test_record.py` の `TestRun`（`--dry`、`--help`）              | `tests/video/test_run.py` に移す（`run.main` のテストなので）。`--help` のモジュール名は `exp.video.run` |
| `test_host.py`                                                  | `tests/lib/test_host.py`                                                                                 |
| `test_stream0.py`                                               | `tests/lib/test_linear_stream.py`                                                                        |
| `test_optim.py`                                                 | `tests/lib/test_optim.py`（`_eigh` の import も含めてそのまま）                                          |
| `test_nodes.py` の `TestEppsPulley`・`TestSIGRegFunction`・`TestInsert`・`TestSIGRegNode`（と `_SIGRegNet`） | `tests/lib/test_sigreg.py`                                                                               |
| `test_nodes.py` の残り（`TestCompare`・`TestDynamics`・`TestLocality`） | `tests/video/test_nodes.py`。`TestLocality` の `_World` は `SIGReg` を `exp.lib.sigreg` から import する |
| `test_evaluate.py` の `TestEffectiveRank`・`TestProbe`          | `tests/lib/test_metrics.py`                                                                              |
| `test_evaluate.py` の `TestRollout`                             | `tests/video/test_evaluate.py`                                                                           |
| `test_config.py` の round trip 2 本                             | `tests/video/test_config.py`                                                                             |
| `test_config.py` の int→float・未知のキー                       | `tests/lib/test_config.py`。テスト内の入れ子の frozen dataclass に対して `from_dict` を呼ぶ形にする      |
| `test_bp.py`, `test_model.py`, `test_run.py`, `test_stage2.py`, `test_video.py` | `tests/video/` に（`test_video.py` は `test_balls.py`）                                                  |

- 正常系: 既存のテストがすべて通る。テストの総数は移動前と同じ
- 異常系: `from_dict` の未知のキー（`TypeError`）が lib のテストで引き続き検証されている
- エッジケース: `REPO_ROOT` が移動後もリポジトリのルートを指す（`TestRecorder` の SHA 比較で検出できる）
- 数値的オラクル: 既存のもの（有限差分、BP との一致、閉形式）を変えない。追加は、ステップ 0/3 の移動前後の `metrics.jsonl` の一致（手動の確認で、テストにはしない）
- ミラーの例外: `lib/mingru.py`（`TestDynamics` が逐次の minGRU と照合している）と `lib/sweep.py`（これまでもテストなし）にはテストファイルを作らない。testing-strategy の 1 source 1 test からの例外として、このメモに記録するだけにする
- `tests/lib/` と `tests/video/` に同じ名前のファイル（`test_config.py`）があるので、両方に `__init__.py` を置く（無いと pytest がモジュール名の衝突で失敗する）

## 再現性と記録

- seed・`Config` のフィールドと既定値・run 名・乱数の消費順は変えない。過去の `runs/*/config.json` は新しい `Config.load` でそのまま読める
- 過去の結果を再現するコマンドは、モジュールパスの置換だけで同じ意味になる（例: `uv run python -m exp.video.stage2 run --out runs/stage2-m10`）
- `exp/video/README.md` の「決定事項」に 1 行足す: 「2026-11-01: `experiments/continuous_stream/` を `exp/video/` に移し、`video.py` を `balls.py` に改名し、汎用部を `exp/lib/` に移した。振る舞いは変えていない（移動前後で `metrics.jsonl` が一致）」

## 文書の書き換え範囲

推奨: **パスとコマンドはすべて新しいものにそろえる。** 過去の記録の中のファイル名の記述（「`video.py` を作る」など）は当時のまま残し、冒頭の注記で対応を示す。古いコマンドはもう動かないので、コピーして使える状態を優先する。古いパスは git の履歴に残る。

- `exp/video/README.md`: 24、40 行目のコマンド。タイトルは `# video（旧 continuous_stream）: ...` にする（名前を変えたことの手がかりとして残す）
- `exp/video/PLAN.md`: 329-342 行目の構成図を実際の配置にする（`exp/video/`、`balls.py`、lib に移したものは `exp/lib/` と注記、`tests/video/`）
- `exp/video/MILESTONES.md`: 12、30、33 行目のパスとコマンドを直す。冒頭に注記を 1 つ:「2026-11-01 に `exp/video/` へ移した（旧 `experiments/continuous_stream/`）。`video.py` は `balls.py`、`record.py`・`host.py`・`stream0.py` は `exp/lib/`（`stream0.py` は `linear_stream.py`）。以下のファイル名は当時のもの」。7 行目のブランチ名の例 `continuous-stream-...` は過去のブランチ名なので変えない
- `exp/video/report.html`: 181 行目のパス、486 行目の README へのパス、488-492 行目のコマンド。1 行目の `<title>` と見出しの `continuous_stream` も `video` にする（公開済みの Artifact があれば再公開は別途。このリポジトリのファイルだけ直す）
- 各スクリプトの docstring のコマンド（stage0/1/2、collapse、run の冒頭）: ステップ 1 の一括置換で直る
- ルート `README.md:31`（EgoDex の段 3 の参照 → `exp/video/`）、`README.md:47`（`exp/` と `tests/` への pyright）
- `CLAUDE.md:12, 20, 53` のパス。「プロジェクト概要」の mujou の bullet に棲み分けの 3 層を足す（上の「方針の棲み分け」節）
- `exp/README.md`: `experiments/` → `exp/`、実行コマンド `uv run python -m exp.<name>.<script>`、`exp/lib/` の役割と 3 層の棲み分け、依存の向き（`exp/<name>/` → `exp/lib/` → mujou。逆向きと実験どうしの import はしない）

## `.claude/` の更新箇所（列挙のみ。編集は skill `edit-dot-claude` で）

- `agents/implementation-planner.md:31` — `experiments/<name>/` → `exp/<name>/`
- `agents/plan-implementer.md:27` — ミラーの説明（`exp/<name>/` ↔ `tests/<name>/`、`exp/lib/` ↔ `tests/lib/`）
- `agents/spec-test-author.md:17, 32, 50` — 書かない範囲を `exp/` に、ミラーの説明に lib を足す
- `agents/orchestrator.md:3`（description）、`:18`
- `agents/code-simplifier.md:17`
- `skills/refactor-conventions/SKILL.md:22-27` — 「実験のパッケージ構成」を 3 層の棲み分けで書き直す。25 行目の「`experiments/` に共通モジュールを作らず mujou に足す」は、今回の方針と矛盾するので必ず直す
- `skills/testing-strategy/SKILL.md:35-38` — ミラーの図に `exp/lib/<module>.py ↔ tests/lib/test_<module>.py` を足す
- `skills/agent-team-startup/SKILL.md:30, 41, 52, 102`
- `skills/agent-team-startup/reference.md:7, 42`（例の `video.py` は `balls.py`）、`:79`（「continuous_stream に」→「video に」）
- `skills/solo-dev-cycle/SKILL.md:46, 61, 78`（78 行目の `experiments/README.md` → `exp/README.md`）
- `skills/mujou-upstream/SKILL.md:3`（description の「experiments/ から」→「exp/ から」）。8 行目か 80 行目の近くに「実験横断の補助は `exp/lib/`、mujou の抽象の不足は mujou」と 1 文足す

`settings.json`・hooks・scripts・commands には該当がない（grep 済み）。

## 想定リスク・トレードオフ

- **subprocess のモジュール名の書き漏れ**: `run_all` は文字列でモジュールを指すので、import のエラーにならず、run が全部 `FAILED` になるだけ。`RUN_MODULE` を 1 か所に置き、`test_run.py` の `--help` テストでモジュール名を検証する
- **`REPO_ROOT` の深さ**: `parents[2]` は `exp/lib/record.py` でも正しいが、将来 lib を深くすると壊れる。既存の SHA 比較テストで検出できる
- **pytest のパッケージ名**: `tests/` に `__init__.py` が無いので、`tests/lib`・`tests/video` はトップレベルの `lib`・`video` として import される。`exp.lib`・`exp.video` とは名前空間が違うので衝突しない。`tests/__init__.py` は足さない（既存の `test_claude_hooks.py` などの挙動を変えないため）
- **`git mv` の履歴**: 分割したファイルは、`git mv` した側にだけ履歴がつながる。`git log --follow` で追えるのは record/host/linear_stream/optim/balls と、元の名前のまま残るファイル
- **`config.py` が torch を import するようになる**: `make_optimizer` を置くため。stage スクリプトの起動が少し遅くなるだけで、実害はない。`exp/video/optim.py` に `from_config` だけを残す案は、lib と同名のモジュールが 2 つになるので採らない
- **`Config` を基底クラス（`JsonConfig`）で汎用化する案**は採らない。`dataclasses.asdict(self)` の型が pyright で通らず、`type: ignore` が要る。`from_dict` 関数 1 つを共有するほうが単純
- **lib が膨らむ**: 「実験の Config に依存しない」だけで何でも入れると、lib が研究中の部品の置き場になる。基準の 2 つ目（別の実験がそのまま import できる）で絞る。次の実験で使われなかったものは、その時点で戻すか消す
- **`report.html` の公開版**: リポジトリのファイルを直しても、公開済みの Artifact は古いまま。必要なら別途再公開する

## 意図的に残すもの（grep の完了条件の例外）

- `exp/video/README.md` のタイトルの「旧 continuous_stream」と、`MILESTONES.md` 冒頭の注記の `experiments/continuous_stream/`（改名の手がかり）
- このメモ `memory/agents/implementation-planner/exp-layout.md`
- `uv.lock`・`runs/`（grep の対象外にしている）
- ハイフン表記の `continuous-stream`（MILESTONES.md:7 のブランチ名、`run.py` の docstring と argparse の description の「the continuous-stream experiment」）は grep に掛からない。`run.py` の文言は「the video experiment」に直すのが望ましいが、必須ではない

## 確認事項

1. **研究中の部品を lib に入れるか** — optimizer の候補（H2 は棄却）、`SIGReg`、`MinGRU`、`LinearStream` は、実験の Config に依存しないので lib に入れる案にした。「lib は実験の運用の補助（record・host・sweep・config・metrics）だけにし、研究中の部品は `exp/video/` に残す」という狭い解釈もある。暫定案: lib に入れる（ユーザーが候補に挙げたため）
2. **過去の記録の書き換え** — `MILESTONES.md`・`report.html`・README の結果節のパスとコマンドを新しいものにそろえ、ファイル名の記述は注記で対応を示す案にした。暫定案: そろえる
3. **合成動画のモジュール名** — `balls.py` を推奨した（段 3 の `egodex.py` と並ぶ）。`synthetic.py` なども考えられる。暫定案: `balls.py`

## 参照

- 規約: `.claude/skills/refactor-conventions/SKILL.md`（22-27 行目が今回の方針と矛盾する）、`.claude/skills/testing-strategy/SKILL.md:33-45`
- 方針の原文: `CLAUDE.md`「プロジェクト概要」、`experiments/README.md`、`memory/project_purpose.md`
- 移動元: `experiments/continuous_stream/`（record.py:15 の `REPO_ROOT`、stage0.py:84-105 の `run_all`、nodes.py:38/151/251 の節の区切り、optim.py:472-506 の `from_config`、evaluate.py:98-129、config.py:131-152）
- 局所性: `mujou/.claude/skills/locality-and-async/SKILL.md`（ノードのコードは中身を変えずに移すので影響なし）
