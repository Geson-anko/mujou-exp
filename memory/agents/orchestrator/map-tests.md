# map: tests とツーリング（M14〜M16 の再編のため）

2026-11-03、orchestrator 配下の調査。コードは変更していない。対象は `tests/`、`pyproject.toml`、`.pre-commit-config.yaml`、`justfile`、skill `testing-strategy` / `refactor-conventions` / `experiment-report`。

## 1. 全体像

- テストは 24 ファイル、`def test_` が 195 個（parametrize 展開前）。`tests/lib/` 8 ファイル、`tests/video/` 12 ファイル、ミラー外が 2 ファイル（`test_claude_hooks.py`、`test_mujou_install.py`）と `conftest.py`
- `just test-fast`（`-m "not slow"`）はコンテナで約 2 分。`just test` は slow も含む（slow は TestRunVideo などで数分）
- ホストの mac の `.venv` には pytest が入っていない（`uv run --no-sync pytest` が spawn に失敗）。テストは dev container（`just container just test-fast`）で回す前提
- CI（`.github/workflows`）は無い。検証は手元の `just run`（format → test → type）だけ

## 2. pytest の設定（`pyproject.toml`）

- `testpaths = ["tests"]`、`pythonpath = ["."]`（`exp` をトップレベルの package として import）
- `addopts = ["--color=yes", "--durations=10", "--strict-markers"]`
- `markers`: `slow`、`cuda(n=1)` の 2 つだけ。新しいマーカーは登録しないと `--strict-markers` で落ちる
- **gotcha**: `cuda(n)` マーカーの skip を実装した `conftest.py` の hook は無い。`tests/conftest.py` は `os.environ.setdefault("OMP_NUM_THREADS", "1")` だけ（mujou のワーカースレッドで OpenMP がコア数ぶん立ち上がり、丸めが run ごとに変わるのを防ぐ。torch の import より前に必要）。CUDA のテストはどれも明示の `@pytest.mark.skipif(not torch.cuda.is_available(), ...)` で skip している（`tests/video/test_model.py:26` の `DEVICES`、`test_model.py:147`、`test_decode.py:93-94`、`tests/lib/test_sigreg.py:272`）。`@pytest.mark.cuda` を実際に付けているのは `test_decode.py:93` だけ
- 複数 GPU のテストは無い

## 3. テストの書き方（実例）

- `class TestXxx:` にまとめ、docstring に「理論から言えること」を英語で書く。名前は振る舞いの文（`test_lam_is_the_credit_of_bp_at_the_fixed_point`）
- 乱数: `torch.Generator().manual_seed(k)`。モジュールの初期化だけ `torch.manual_seed(0)`
- mujou の実行: `mujou.testing.run(net[, inputs=...])` の仮想タイマーで `run_.step(net.tau.period)` を手で回す。フレームのループは `exp.lib.host.Host(run, frame=1/30, period=TAU.period).show(writes, port)`
- 定数: `TAU = pc.TimeConstants(h=4e-3, lam=2e-3, r=4e-3)`、`FRAME = 1/30`（`test_bp.py`・`test_nodes.py` で重複定義）
- ヘルパー（ファイルごとに重複している。conftest に fixture は無い）
  - `_fd(f, x, eps=1e-6)`: 中心差分（`test_nodes.py:26`、`test_sigreg.py:24`）
  - `_inputs(*names_dims)` → `{name: pc.Input(x=TensorSpec((d,)), y=TensorSpec((1,)), period=TAU.period)}`、`_sample(x, reset, mask)` → `pc.msg.Sample`（`test_nodes.py`、`test_sigreg.py`、`test_model.py`）
  - `_Spy` / `_Frozen` / `_Recording`: 勾配と elapsed を記録する `pc.optim.Optimizer`（`test_bp.py:27,143`、`test_model.py:190`）。`monkeypatch.setattr(model, "make_optimizer", recording)` と `"exp.video.bp.make_optimizer"` で差し込む（`test_model.py:224-225`）。モジュール内の名前 `make_optimizer` に依存しているので、再編で名前を変えるとこのテストが壊れる
  - `_video_run(tmp_path, name, streams, **changes)`（`test_model.py:736`）、`_run`（`test_run.py:14`）: `Config` を JSON に書き、`run.main(["--config", p, "--out", tmp])` を呼び、`metrics.jsonl` を読む。`run.main` が run のディレクトリを返す契約に依存
- fixture は `test_egodex.py:135-170` の `scope="module"` の 2 つ（ffmpeg で色の違う偽 EgoDex を作る）と、組み込みの `tmp_path` / `monkeypatch` だけ
- 小さい設定: `SMALL = Config(model=ModelConfig(d_z=8, d_s=12, directions=16, buffer=31), optim=OptimConfig(tau=1.0))`（`test_model.py:34`）

### オラクルの型（再編後も残すべき型）

| 型 | 実例 |
| -- | ---- |
| PC-ALM の 1 層 = BP の 1 ステップ | `test_bp.py:73` `TestOneLayer.test_pc_alm_steps_exactly_as_bp`（`_Spy` の各ステップの勾配が autograd と一致、elapsed ≈ 1 フレーム） |
| 平衡点の `lam` = BP の credit | `test_bp.py:110`（2 層）、`test_model.py:87`（エンコーダ・動力学・Compare・SIGReg・Reconstruct、`learn_interval=math.inf` で 3000 ステップ保持、`rtol=1e-3`）、`test_model.py:356`（動力学のノード数）、`test_model.py:552`（バッチのストリームごと） |
| 有限差分 | `test_nodes.py:73`（`compare_energy` の両側の勾配、float64、`rtol=1e-4, atol=1e-6`）、`test_sigreg.py:88` |
| 適格度トレース = 入力固定の BPTT | `tests/lib/test_mingru.py:20`（`MinGRUTrace`、batch `()` と `(3,)`）、`test_model.py:221`（ネットワーク全体、pc と bp） |
| 退化 | `test_optim.py:48`（`Whitened` で a=b=0 は SGD）、`test_evaluate.py:46`（恒等モデルの rollout = persistence）、`test_evaluate.py:64` |
| 局所性 | `test_nodes.py:217`（どのノードも `p.grad` を残さず、重みは動く）、`test_nodes.py:158`（`Dynamics` は過去へ credit を返さない） |
| 物理の不変量 | `test_balls.py:35-112`（エネルギー保存、`conserve`、壁での反発係数）、`test_balls.py:155`（1 フレームの移動 ≤ v_max·FRAME） |
| 決定性 | `test_run.py:23`（同じ seed で同じ metrics）、`test_model.py:749`（評価が学習を変えない）、`test_model.py:756`（checkpoint から同じ評価値、`rel=1e-5`） |
| CPU = CUDA | `test_model.py:149`（TF32 を切り、最初の 10 フレームの誤差が `rtol=1e-3`） |
| 手で読める出力 | `test_decode.py:34`（PNG のバイト列を手で復号） |

## 4. テスト ↔ モジュール

テストの無いモジュール: `exp/lib/sweep.py`、`exp/video/{easy,collapse,frozen,latency,stability,stage0,stage1,stage3}.py`（実験の掃引スクリプト）。`stage2.py` は `judge` だけテストがある。

| テスト | モジュール | 主な対象 | slow |
| ------ | ---------- | -------- | ---- |
| `tests/lib/test_config.py` (3) | `exp/lib/config.py` | `from_dict`（int→float、未知キーはエラー、Optional の dataclass） | |
| `tests/lib/test_host.py` (3) | `exp/lib/host.py` | `Host`（フレームが格子に乗る、書いたフレームの予測を返す）、`RealtimeHost` | |
| `tests/lib/test_linear_stream.py` (5) | `exp/lib/linear_stream.py` | OU 過程の自己相関・分散、iid、`y = w* x + noise` | |
| `tests/lib/test_metrics.py` (4) | `exp/lib/metrics.py` | `effective_rank`、`probe_r2` | |
| `tests/lib/test_mingru.py` (3) | `exp/lib/mingru.py` | `MinGRUTrace` = BPTT、reset、`ResidualMinGRU` | |
| `tests/lib/test_optim.py` (24) | `exp/lib/optim.py` | `Whitened`・`Projected`・`AdamDirection`・`Grafted`・`Orthogonalized`・`_eigh`・`Warmup`、state_dict の往復 | |
| `tests/lib/test_record.py` (2) | `exp/lib/record.py` | `Recorder`（config.json に設定と commit、metrics.jsonl） | |
| `tests/lib/test_sigreg.py` (17) | `exp/lib/sigreg.py` | Epps-Pulley、`sigreg` の勾配（FD）、バッファ `insert`、`SIGReg` ノード、`Every`、`Sums`、バッチ（cuda param あり） | |
| `tests/video/test_balls.py` (27) | `exp/video/balls.py` | `Physics`・`advance`・`Video`・`Streams`・`render`・`eval_clips`・`ordered`、`episode=None`（redraw なし）、`radii` | 2 件（30 分のエネルギー、30 分が 1 分以内） |
| `tests/video/test_bp.py` (4) | `exp/video/bp.py` | `bp_chain`、`WorldBP`（バッチの平均、再構成の項） | |
| `tests/video/test_config.py` (11) | `exp/video/config.py` | `Config` の JSON 往復、`streams`、`Physics` への反映、`checkpoint_interval`、`DynamicsOptim` | |
| `tests/video/test_decode.py` (6) | `exp/video/decode.py` | `decoder`、`fit_decoder`、`write_png`、`_tile`（private を直接テスト）、`decode_checkpoint` | 2 件（うち 1 件 CUDA） |
| `tests/video/test_egodex.py` (20) | `exp/video/egodex.py` | ffmpeg の decode、`plan_stream`・`plan_clips`、`prepare`・`EgoDexStream`、`/datasets/egodex` があれば実データのメタデータ | |
| `tests/video/test_evaluate.py` (12) | `exp/video/evaluate.py` | `rollout_errors`・`persistence_errors`・`mean_errors`・残差の rollout・`shrink_*`・`halves`・`rollout_shape` | |
| `tests/video/test_model.py` (26) | `exp/video/model.py`（+ `run`、`bp`、`evaluate`） | エンコーダの幅、平衡点の credit（5 変種 × device）、CUDA、BP と同じ初期重み、トレース、`DynamicsOptim`、ノード数、blur、residual、open loop、バッチのストリーム、`TestRunVideo` | `TestRunVideo` のクラス全体（7 件） |
| `tests/video/test_nodes.py` (8) | `exp/video/nodes.py` | `Compare`（FD、`target_weight`、予測なし）、`Dynamics`（逐次の minGRU と一致、過去へ credit なし）、局所性、バッチ版 | |
| `tests/video/test_offline.py` (3) | `exp/video/offline.py` | `window_starts`（reset をまたがない）、`main` の保存 | `TestMain` |
| `tests/video/test_run.py` (6) | `exp/video/run.py`（+ `stage0.RUN_MODULE`） | linear の決定性と収束、`source`、`--dry`、`--help`（subprocess で `uv run python -m`） | 1 件（iid 収束） |
| `tests/video/test_stage2.py` (4) | `exp/video/stage2.py` | `judge`（H2〜H4 の判定） | |
| `tests/test_claude_hooks.py` (6) | `.claude/hooks/` | `pretooluse-guard-bash.py` ほか | |
| `tests/test_mujou_install.py` (1) | — | mujou が submodule の editable install | |

時間を食う fast のテスト: `test_model.py` の平衡点（20 フレーム × 17 ステップ + 3000 ステップ、5 変種）、`TestStreams`、`TestTrace`（1500 ステップ保持）。slow でなくても数秒ずつかかる。

## 5. ツーリングの規約

- **pyright** standard（`typeCheckingMode` 未指定 = standard）、`include = ["exp", "tests"]`、`reportImplicitOverride = true`（オーバーライドに `@override` 必須。テストの `_Spy` などもすべて `@override` を付けている）。`mujou`、`.claude/worktrees/`、`.venv-devcontainer` は除外
- **ruff**: line-length 88、`select = ["E","F","W","I"]`、ignore `E203`・`E501`・`E741`（数式の `l`・`I`）、isort は `combine-as-imports`。format は double quote
- **pre-commit**（`exclude: ^(mujou/|\.claude/skills/yomiyasu/)`）
  - `check-docstring-first`、`check-yaml`（hydra の yaml にも効く）、`check-toml`、`check-json`、`debug-statements`、`check-added-large-files`（既定 500 KB。`report-m11-m12.html` は 429 KB で近い）、`check-executables-have-shebangs`
  - `uv-lock`（lock の更新。依存を足したら `uv.lock` を commit）
  - `ruff --fix`、`ruff-format`、`pyupgrade --py314-plus`
  - `mdformat`（`memory/`、`.claude/`、`exp/<name>/` は対象外。ルートの README と `exp/README.md` は対象）
  - `codespell`（`--ignore-words-list=alm,crate`）
  - pygrep: `python-check-blanket-noqa`、`python-check-blanket-type-ignore`（`# type: ignore[code]` の形で書く）、`python-no-log-warn`、`python-use-type-annotations`
- **Python 3.14t（free-threaded）**、`requires-python >=3.14`、`[tool.uv] package = false`。依存は `imageio-ffmpeg` と `mujou`（editable）だけ。dev は `pre-commit`・`pyright`・`pytest`・`ruff`
  - hydra-core（omegaconf、antlr4 runtime）は pure Python で入るはず。optuna は sqlalchemy・numpy・alembic を引く。trackio は gradio・pandas などを引き、3.14t の wheel が揃うか要確認（`uv add` で失敗したら tracking は別の手段にする）
  - `pip install` は hook がブロックする。`uv add` / `uv add --dev`
- **.gitignore**: `runs/`、`outputs/`（hydra の既定の出力先と一致）、`wandb/`、`logs/`、`*.pt` 等。hydra の `multirun/`、optuna の `*.db`（sqlite）、trackio の保存先は**まだ無い**ので足す必要がある
- `justfile`: `test`（`pytest -v`）、`test-fast`（`-m "not slow"`）、`type`、`run`（format test type）、`container`（dev container の中で実行）。GPU の fleet 用のレシピは無い

## 6. 再編で守ること

1. **コード**: docstring とコメントは英語。公開関数の docstring にテンソルの shape（`h: (batch, n_in)`、バッチ先頭）。数式の記号を出典に揃える（`a`, `r`, `lam`, `eps`, `e_out`）。`_` prefix のカプセル化、`type X = ...`、`@classmethod` のファクトリ、`Any` と `# type: ignore[...]` は理由のコメント付きのときだけ。検証は境界でだけ、NaN は握りつぶさず止めて記録
2. **文書**: README・計画書・skill・memory は日本語。commit メッセージと PR タイトルは英語の `<種別>(<スコープ>): ...`
3. **設定と再現性**: ハイパーパラメータ・seed・パスは設定に集める（hydra の structured config に移すなら dataclass のまま `ConfigStore` に載せるのが規約に近い）。`torch.Generator` を明示的に渡す。記録に seed・設定・mujou の submodule の SHA を残す（今は `exp/lib/record.py` の `Recorder` が `config.json` に書く。tracking を入れても、この 3 つが run ごとに残ること）。checkpoint・ログは git の外
4. **テスト**: `exp/<name>/<module>.py ↔ tests/<name>/test_<module>.py` の 1 対 1 のミラー、各ディレクトリに `__init__.py`。理論から導いたオラクルが最優先、「損失が下がった」だけを完了の条件にしない。float64 と根拠のある許容誤差。`class TestXxx:`、parametrize、例外メッセージは substring。mujou と torch はモックしない、private 関数もモックしない、ダウンロードを伴うテストは書かない（Walking Tours も小さな合成の mp4 で置き換える。`test_egodex.py` の ffmpeg で色の動画を作る手が使える）。数十秒以上は `@pytest.mark.slow`。本番の学習スクリプト（hydra の main など）は小さな設定で最後まで走ることだけを確かめる
5. **mujou の原則**: 局所性・非同期性（ノードをまたぐ `.backward()` やバリアを持ち込まない）。`test_nodes.py:217` の局所性のテストの形を新しいネットワークにも置く
6. **レポート**: `exp/<name>/report.html` は 1 ファイルで完結（データは `<script type="application/json" id="data">`、素の JS と SVG、外部は Google Fonts だけ）、数 MB 以下（手本 295 KB）、数字は README・`results/`・`runs/` にあるものだけ、生成スクリプトと JSON は commit しない、本文は必ず yomiyasu で推敲、`h1` は問い、構成は「分かったこと → 仕組み → 入力 → 進め方 → 結果 → 仮説の判定 → 次へ → 付録: 再現（mujou の SHA、コマンド）」。M14〜M16 は「見た目で確かめる」が条件なので、予測のフレームや軌跡を描く図が要る

## 7. 旧モジュールを消すと不要になるテスト

M14（状態の観測の動力学）・M15（JEPA、Walking Tours）・M16（同時の学習、動力学は credit を返さない）を前提にした分類。

### 消してよい（対応するモジュールごと）

- `tests/video/test_egodex.py`（20 件）: M15 で EgoDex は使わない。`exp/video/egodex.py`、`model.egodex_stages`・`calibration_images`、`run.py:118` の分岐、`latency.py`、`stage3.py` と一緒に消せる。ffmpeg の偽データの作り方だけは Walking Tours のテストに流用できる
- `tests/video/test_stage2.py`（4 件）: `stage2.judge`（H2〜H4）。M8/M10 の判定で、もう使わない
- `tests/video/test_run.py` の `TestRunLinear`（2 件）と `tests/lib/test_linear_stream.py`（5 件）: 段 0 の線形回帰のストリーム。ただし `test_bp.py:56` の `_stream_through` が `LinearStream` を使っているので、`TestOneLayer` を残すならテスト内の小さな乱数ストリームに置き換える。`test_run.py` は `stage0.RUN_MODULE` も import している
- `tests/lib/test_optim.py`（24 件）: `Whitened`・`Projected`・`AdamDirection`・`Grafted`・`Orthogonalized`・`Warmup`。M10 で「どの optimizer も SGD を超えない」。M13 は AdamW と grafting を試したが SGD を超えない。M14 で optimizer をまた比べるなら `Grafted`（と `AdamDirection`）だけ残す判断もある。`model.py:23` が `exp.lib.optim.Correlation` を、`config.py:13` が optim のクラスを import している
- `tests/video/test_model.py` のうち、M13 のエンコーダの変種に属すもの: `TestEncoderWidth`（3）、`TestBlur`（2）、`TestResidual`、`test_bp_starts_from_the_same_weights` の recon、平衡点のテストの `stop_target`・`target_weight`・`z_norm`・`residual`・`recon` の変種、`test_how_the_encoder_is_cut_does_not_change_the_function`。畳み込みのエンコーダと `Reconstruct` を消すなら不要
- `tests/video/test_nodes.py` の `Reconstruct` 関係、`tests/video/test_bp.py:225`（`WorldBP` の再構成の項）: M13 の `nodes.Reconstruct` は使わない方針（memory `project_decoder_for_debug.md`）
- `tests/video/test_evaluate.py` の `TestShrink`（4）・`TestShape`（3）: `shrink{n}`・`amp{n}` は M11〜M13 の潜在の物差し。M14 は観測の空間の誤差、M16 はデコードした画像の誤差で判定するので不要になる見込み
- テストの無いスクリプト（`collapse`・`frozen`・`latency`・`stability`・`stage0`〜`stage3`・`easy`）は消してもテストは減らない。`easy.EASY` の物理は M14・M16 のデータなので、設定（hydra の yaml など）に移してから消す

### 残す・作り直す（M14〜M16 で使う）

- `test_balls.py`（物理・`Streams`・`eval_clips`・`render`。M14 は `Frame.state` から観測を作り、`balls.render` で予測を描く）
- `test_nodes.py` の `Compare`・`Dynamics`・局所性（M14 の `Input → Dynamics → Compare`。M16 は `target_weight=0` と「`Dynamics` が `g_in` を返さない」のテストが要の性質）
- `test_bp.py` の `TestOneLayer`・`TestTwoLayers`・`TestWorldBPBatch` の平均の部分（エンコーダの無い `WorldBP` に作り直す）
- `test_model.py` の平衡点の credit・`TestTrace`・`TestMoreNodes`・`TestOpenLoop`・`TestStreams`・`TestRunVideo`（エンコーダの無いネットワークで作り直す。MILESTONES の M14 のオラクル「エンコーダなしの平衡点」「トレース」はこれの流用）
- `tests/lib/test_mingru.py`、`tests/lib/test_sigreg.py`（M15 の LeJEPA の SIGReg）、`tests/lib/test_metrics.py`（probe・実効ランク）、`tests/lib/test_host.py`、`tests/lib/test_record.py`、`tests/lib/test_config.py`（hydra に移すなら `from_dict` の扱いを決め直す）
- `test_decode.py`（M15・M16 の後から学ぶデコーダ、PNG）。`_tile` は private を直接テストしているので、再編のときに公開にするか、公開 API 経由に直す
- `test_evaluate.py` の `TestRollout`・`TestPersistence`・`TestMean`（M14 の物差し「平均」「写し」と同じ形）。M14 の「等速の外挿」は新しいオラクル（重力 0 で壁に当たらない区間で誤差 0）
- `test_offline.py` の `window_starts`（M14 のオフラインの上限で同じ窓の作り方を使う）
- `test_config.py`: 設定の往復のテストは hydra の config（`OmegaConf.structured` → `to_object` の往復）で書き直す
- `test_claude_hooks.py`、`test_mujou_install.py` はそのまま
