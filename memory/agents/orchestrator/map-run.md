# 地図: 実験の実行と記録（run / config / host / record / sweep / metrics / infra）

2026-11-03 調査。M14〜M16 に向けた再編のための現状の地図。コードは変更していない。

## 0. 前提として見つけたこと

- この Mac の checkout では `mujou/` submodule が**未初期化**（`git submodule status` が `-881c4ee...`、中身が空）。ここでは import も実行もできない。実験は GPU 機（Linux、dev container の `.venv-devcontainer`）で回してきた
- 既知の計算環境（`exp/video/PLAN.md:383`）: RTX 4090（24 GB）と RTX 4060（8 GB）、32 コアの 1 台。sweep スクリプトが `cuda:0` / `cuda:1` を交互に固定で書いている（`easy.py:67`、`stability.py:202-263`）。4 台の GPU 機の情報はリポジトリ内に無い
- 依存は `imageio-ffmpeg` と `mujou` だけ（`pyproject.toml`）。hydra / optuna / trackio は未導入。Python は **3.14t（free-threaded）**。hydra（omegaconf + antlr4 runtime）、optuna（sqlalchemy、alembic）、trackio（gradio、pandas 等）が 3.14t で入り・動くかは要確認（C 拡張の free-threaded wheel の有無）

## 1. 設定（`exp/video/config.py`）

すべて `@dataclass(frozen=True, kw_only=True)`。JSON と往復する（`Config.to_dict` = `dataclasses.asdict`、`Config.from_dict` = `exp.lib.config.from_dict`）。

### `Config`（config.py:209-254）

| field | default | 意味 |
| -- | -- | -- |
| `name` | `"run"` | 出力ディレクトリ名に使う。sweep では設定値を手で埋め込んだ名前（例 `m13-big-pc-s0`） |
| `seed` | `0` | 初期重み（`torch.manual_seed`）、ストリーム、評価の seed の基 |
| `bp` | `False` | True で BP 基準（`WorldBPNetwork`）、False で PC-ALM（`Network(config, correlation=True)`） |
| `eval_interval` | `60.0` | 実行のタイマー（仮想時間）の秒。何フレームごとに評価するか = `round(eval_interval / frame)` |
| `checkpoint_interval` | `0.0` | 0 で無し。整数秒必須、realtime とは併用不可（`__post_init__`） |
| `realtime` | `False` | True で `mujou.run`（実時間）、False で `mujou.testing.run`（手で step） |
| `device` | `"cpu"` | 全ノードと評価のデバイス文字列（`"cuda:0"` など） |
| `time` | `TimeConfig()` | |
| `optim` | `OptimConfig()` | 全ノードの optimizer（`dynamics_optim` があればエンコーダとデコーダだけ） |
| `dynamics_optim` | `None` | 動力学と readout の optimizer（M13） |
| `model` | `ModelConfig()` | |
| `data` | `DataConfig()` | |

### `TimeConfig`（:34-42）

`frame=1/30`、`h=4e-3`、`lam=2e-3`、`r=4e-3`（`pc.TimeConstants(h=, lam=, r=)`）、`learn_interval=25e-3`（None で毎更新学習）。host の step 周期は `tau.period`（docstring より 2 ms。1 フレームが 16 か 17 step）。30 分 = 54,000 フレーム ≈ 90 万 step。

### `OptimConfig`（:45-69）

`kind="sgd"`（`sgd`/`adamw`/`adam_direction`/`whitened`/`orthogonalized`/`projected`）、`tau=1.0`（パラメータの時定数。sweep では学習率 lr を `tau = FRAME / lr` で表す）、`graft=False`、`tau_m=0.3`、`tau_v=30.0`、`weight_decay=0.0`、`tau_l=None`、`tau_r=None`、`a=0.25`、`b=0.25`、`rank=4`、`eps=1e-4`、`refresh=1.0`、`warmup=0.0`（秒、>0 で `Warmup(inner, seconds)` で包む）。`make_optimizer(c)`（:257-294）が実体化。

### `ModelConfig`（:72-119）

`hidden=0`（段 0 の MLP 幅）、`d_z=64`、`d_s=256`、`kappa=0.05`（SIGReg 重み）、`stop_target=False`、`target_weight=0.0`、`z_norm=False`、`width=32`、`blur=0.0`、`open_loop=0.0`、`open_cycle=20`、`residual=False`、`enc_nodes=2`、`dyn_nodes=1`、`recon=0.0`、`trace=False`、`directions=256`、`buffer=255`、`buffer_every=1`。
エンコーダ前提の項目（kappa, z_norm, width, blur, enc_nodes, recon, directions, buffer*）と動力学の項目（d_s, dyn_nodes, trace, residual, open_loop）が 1 つのクラスに混在。M14（エンコーダなし）、M15（ViT JEPA）、M16 では分けるのが自然。

### `DataConfig`（:122-206）

`kind="video"`（`"linear"`/`"video"`/`"egodex"`）、`seconds=600.0`、`iid=False`、線形回帰: `d_x=32`、`tau_x=10.0`、`noise=0.1`。合成動画: `size=32`、`balls=3`、`radii=None`（tuple、JSON の list を `__post_init__` で tuple 化）、`gamma0=1.0`、`gamma_amp=0.5`、`e0=0.6`、`e_amp=0.5`、`drag=0.1`、`conserve=False`、`physics_period=600.0`、`episode_min=3.0`、`episode_max=6.0`、`redraw=True`、`order="stream"`（`stream`/`shuffled`/`replay`）、`clip=1.0`、`replay=4096`、EgoDex: `path="datasets/egodex64"`、`streams=1`（B。>1 は `kind=="video"` かつ `order=="stream"` のみ）、`spread=False`。`physics()` → `balls.Physics`、`batch` → `()` か `(streams,)`。
M14 の「状態から観測」の kind は無い（新設が要る）。M13/M14 のデータは `easy.EASY`（`seconds=1800, gamma_amp=0, e0=1, e_amp=0, drag=0, conserve=True, redraw=False, streams=128`）。

### `exp/lib/config.py`

- `RunConfig` Protocol: `name` プロパティと `to_dict()` だけ（Recorder と sweep が要求）
- `from_dict(cls, d)`: 入れ子の dataclass、`X | None` の dataclass、`float` 型への int 変換を処理。未知のキーは `TypeError`。tuple は変換しない（`radii` は `__post_init__` で対処）。hydra の structured config に置き換えれば不要になる

## 2. 起動（CLI）

`uv run python -m exp.video.run [--config <json>] [--out runs] [--dry]`（run.py:236-254）

- `--config` 省略で `Config()`。JSON は `Config.to_dict()` の形そのまま（Recorder の `config.json` の `"config"` とは別。`config.json` は head 付き）
- `torch.set_num_threads(1)`（:248）。**`OMP_NUM_THREADS` は設定しない**。複数プロセスで並べると OpenMP の取り合いで数十分止まる既知の問題あり（`memory/project_m13_findings.md:15`、`tests/conftest.py` は自分で 1 を設定）。sweep も env を渡さない
- `data.kind` で `run_linear`（段 0）か `run_video`（video / egodex）に分岐。`--dry` は `config.json` だけ書く
- 最後の行に run のディレクトリを print（sweep がこれを読む）

マイルストーンごとの「sweep スクリプト」が設定のリストを Python で組む: `stage0.py`（`RUN_MODULE="exp.video.run"`、`SEEDS=(0,1,2)`）、`stage1.py`、`stage2.py`、`collapse.py`（M9）、`stability.py`（M11、大量の定数タプルに device を手書き）、`easy.py`（M13）。形は `main()` に `what in {run, summarize, configs}`、`--out runs/<mN>`、`--jobs`。summarize はスクリプトごとに独自（`easy.summary` は `decode/metrics.jsonl` の最終行を物差しで割った比）。

別系統の CLI（Config を部分的にしか使わない）: `offline.py`（オフライン上限、argparse に `--lr --steps --batch --sigreg --local --sgd --frames --every --device --save` など独自の引数多数、`--config` でモデルとデータだけ取る）、`decode.py <run dir> [--steps 2000] [--frames 16384] [--device cuda:0]`、`frozen.py`、`latency.py`。

## 3. ホストのループ

### `exp/lib/host.py`

- `Host(run: ManualRun, *, frame, period)`（:20-64）: `show(writes, watch) -> (first_msg, written_timestamp)`。`run.write(writes)` → `steps = floor((k+1)·frame/period+1e-6) - floor(k·frame/period+1e-6)` 回 `run.step(period)`、各 step 後 `run.read(watch)` で `msg["time"] == written` の最初のメッセージ（= そのフレームから学ぶ前の予測、prequential）を返す。届かなければ `RuntimeError`。**仮想時間なので結果は計算時間に依存しない、決定的**
- `RealtimeHost(run: Run, *, frame)`（:67-108）: `run.wait_for(watch, pred, timeout)`、フレームの時間が過ぎるまで `timer.sleep`。間に合わなければ `None`

### `run_video`（run.py:141-233）

1. `source(config)`（:113-138）→ `(video, clips)`。`egodex` は `EgoDexStream(d.path), egodex_clips(d.path)`。それ以外は `Video(seed, ...)`（B=1）か `Streams(B, seed, spread, ...)`、`eval_clips(size, balls, radii, frame, physics)`（`(P, C, T, ...)`）。clips は device へ
2. `frames = round(seconds/frame)`、`every = round(eval_interval/frame)`、`ckpt_every = round(checkpoint_interval/frame)`
3. Streams なら `video.next()` を frames 回、そうでなければ `balls.ordered(video, frames, order, clip, replay, seed+1)` で `(Frame, live)` を流す
4. 各フレーム: `pc.msg.Sample(x=f.image, y=zeros(*batch,1), mask=~reset, reset=reset)` を `net.prev` に書き、`net.next` を watch。記録: `frame`、`error`（閉ループの予測の `1/D||ẑ-z||²` の B 平均）、`open_error`（`open_mask` で開ループの予測）、`z_rms`、realtime で間に合わなければ `missed`。replay のフレームは `replay_error` だけ
5. `(k+1) % every == 0` かつ非 realtime: `evaluate(snapshot(net), clips, d_state)` の全指標（`z_var, rank, rollout{1,5,10,20}, persist*, mean*, shrink*, amp*, corr*, probe_{z,s}_{pos,vel,phys}`）と PC-ALM なら `rho_<node>_<lag>` を同じ行に足す
6. `(k+1) % ckpt_every == 0` かつ非 realtime: `save_checkpoint(snapshot(net), run_dir, seconds)` → `ckpt/<秒:05d>.pt`
7. realtime なら最後に 1 回だけ評価

評価は**仮想時間で同期的に**、ホストのループの中でモジュールの `deepcopy`（`Snapshot`）を取って行う（モジュールは step の間でだけ読める）。評価 1 回は小モデルで約 1 秒（`results/m05.md`）。

`Snapshot`（evaluate.py:33-56）: `encoder`, `step`（`DynamicsStack`）, `readout`, `residual`。**checkpoint は `torch.save(Snapshot)` の pickle（モジュールのクラスパスごと）**で、`load_checkpoint` は `weights_only=False`。コードを再編（クラスの移動・改名）すると古い checkpoint は読めなくなる。

`run_linear`（:30-83）: 段 0 の線形回帰 / MLP、`bp_chain` か `pc.Chain`、`loss`・`dist`・`iid_loss` を記録。

### 時間的な見積もり（実測）

| 設定 | 30 分のストリームの壁時計 |
| -- | -- |
| 合成動画小（D=64, d_s=256）、SGD、CPU 1 スレッド、空いた CPU | 約 32 分（実時間の 1.07 倍、`results/m05.md`） |
| 同、16 本並列で CPU が埋まっているとき | CPU 4.0 倍 / CUDA 5.3 倍 |
| EgoDex 3+3 ノード、4090、仮想時間 | 実時間の約 6 倍 = 3〜3.5 時間（CPU 1 コア律速、GPU は空き。5 本並べても 1 本あたりほぼ同じ） |
| M13 大、B=128、PC-ALM | 2 時間超 |
| M13 トレースあり PC-ALM | 5〜7 時間超の見込み（未実行） |
| offline 6000 ステップ | GPU に 3〜4 本並べて 30〜50 分 |
| decode.py | ckpt ごとにデコーダを 2000 step 学習（ckpt 300 秒ごとなら 6 個） |

律速は Python の 2 ms ごとの step とノードごとの小さなカーネル起動（バッチ小では GPU は CPU より速くない）。**1 プロセス = CPU 1 コア**で、GPU 1 枚に複数本並べられる（4060 は 8 GB で評価をチャンクに分割済み）。M14（エンコーダなし、観測 4〜12 次元）は GPU 不要で CPU に多数並べるのが速い見込み（要測定）。

## 4. 記録（`exp/lib/record.py`）

- `Recorder(config: RunConfig, root=REPO_ROOT/"runs")`: `root/<YYYY-MM-DD>-<config.name>/` を `exist_ok=False` で作る（同じ日に同名を回すと `FileExistsError`）
- `config.json`: `{"config": to_dict(), "mujou": sha, "repo": sha, "dirty": bool, "created": iso8601}`。git を subprocess で叩く（`git` が無い環境・別 checkout では失敗する）
- `metrics.jsonl`: 1 レコード 1 行 `{"time": <実行のタイマーの秒>, ...}`。**毎フレーム 1 行**（30 分で 54,000 行）、評価の行は同じ行にキーが増える。flush しない（バッファ）ので途中経過は遅れて見える
- `ckpt/<秒>.pt`（run.py）、`done`（sweep が成功時に touch）、`decode/{config.json,metrics.jsonl,<秒>.png}`（decode.py が後から）
- `runs/` は git 管理外。現 Mac には runs/ が無い（GPU 機の上にある）

## 5. sweep の現状（`exp/lib/sweep.py`）

`run_all(module, configs, out, jobs)`: `out/*-<name>/done` があるものを飛ばし、残りを `ThreadPoolExecutor(jobs)` で 1 本 1 プロセス（`sys.executable -m module --config out/<name>.json --out out`、`cwd=REPO_ROOT`、`capture_output=True`）。失敗時は stderr の最後 2000 文字を print するだけ（ログファイルは残らない）。成功時に stdout の最終行のディレクトリに `done`。

- デバイスの割り当ては設定の `device` に手書き（`cuda:0`/`cuda:1` 交互）。GPU の空き・メモリを見たスケジューリングは無い
- 1 台のマシンの中だけ。複数マシンへの分配、結果の集約、途中再開（checkpoint からの resume）は無い
- 探索は手作業: 定数タプル（`PARALLEL`、`SPANS`、`SPREAD_MORE` など）を「Added after ...」のコメントとともに書き足していく。optuna のような逐次的な探索・早期打ち切りは無い
- 名前に設定値を手で埋め込む（`f"{name}-lr{lr:g}-s{seed}"`）。衝突・取り違えは名前の規約頼み

## 6. infra

- `justfile`: `setup`、`format`、`test`、`test-fast`、`type`、`run`、`clean`、`mujou-status`、`mujou-update`、`container *args`（`devcontainer up` + `devcontainer exec ... bash -lc 'cd <checkout> && exec "$@"'`）。実験を回すレシピは無い
- `compose.yaml`: service `dev`、`sleep infinity`、`/datasets/egodex:ro` と `/datasets/walking_tours`（rw）を mount。**GPU の device reservation（`deploy.resources.reservations.devices` / `runtime: nvidia`）が無い**。`devcontainer.json` の `hostRequirements.gpu: optional` は compose ベースでは GPU を渡さない可能性があるので、コンテナ内で CUDA が見えるかは要確認
- `devcontainer.json`: `UV_PROJECT_ENVIRONMENT=.venv-devcontainer`、`postCreateCommand` で submodule 取得 + `uv sync`、ホストの `~/.claude`、`~/.ssh`（ro）、gh 設定を mount
- `Dockerfile`: ubuntu 24.04、gh/git/tmux/ripgrep/jq、uv 0.13.0、just 1.58.0、rustup（mujou の Rust 拡張のビルド用）、Claude Code。CUDA ツールキットは入れていない（torch の wheel に同梱の想定）

## 7. `exp/lib/metrics.py`

`effective_rank(z (n,D)) -> float`（共分散の固有値の正規化エントロピーの exp）、`probe_r2(train_x, train_y, test_x, test_y, ridge=1e-3) -> (K,)`（標準化 + 切片、ridge 回帰の R²）。M15 の probe・ランクにそのまま使える。

## 8. 多数の実験を複数マシンで効率よく回すための痛点

1. **設定の上書きが CLI で出来ない**: 1 値を変えるにも Python で `replace` を重ねる sweep スクリプトを書く。→ hydra の structured config（現行 dataclass をそのまま ConfigStore に登録できる）と `key=value` の override、multirun
2. **ModelConfig/DataConfig が全段の項目の寄せ集め**: M14（観測 → 動力学）、M15（ViT JEPA）、M16 を 1 つの Config に押し込めない。hydra の config group（`data=balls_state|balls_image|walking_tours`、`model=dynamics|jepa|joint`、`learner=pc|bp`）で分ける
3. **探索が手作業**: optuna で、評価の行（`rollout10` 等）を中間値として `trial.report` し pruner で早期打ち切り。評価がホストのループ内で同期的なので、そこに hook を入れれば済む
4. **デバイス・マシンの割り当てが手書き**: `device` を設定から外し、ランチャーが `CUDA_VISIBLE_DEVICES` と並列数（CPU コア律速、GPU あたり N 本）を決める形に。4 台への分配と、`runs/` の集約（rsync か共有ストレージか trackio のサーバ）
5. **ログ**: 毎フレームの jsonl（54k 行）と評価が同じファイルに混在、flush なし、stderr は失敗時の 2000 文字だけ。trackio にはフレームの指標を間引いて（例: 1 秒平均）、評価は全部送る。stdout/stderr はファイルに残す
6. **出力ディレクトリ名が日付 + name**: 同日同名で衝突、日をまたぐと done 判定が `*-name` の glob 頼み。マシン名や一意 id、設定のハッシュを入れる
7. **checkpoint が pickle の Snapshot**: 再編で読めなくなる。state_dict + 設定から組み直す形に。途中再開（preemption）も state_dict なら可能
8. **`OMP_NUM_THREADS=1` をランチャーが設定しない**（既知のハング要因）。`torch.set_num_threads(1)` だけでは足りない
9. **デコーダ評価が別プロセスで後から手動**（`decode.py <run>`）: M15・M16 の判定はこれが主。ランチャーの後段ジョブとして自動で回し、PNG を trackio / レポートに載せる
10. **1 本が長い**（30 分のストリームで 30 分〜7 時間、CPU 1 コア律速）: 並列の本数で稼ぐしかない。探索は短いストリーム（例 5〜10 分）+ pruner、確定は 30 分 × seed 3 × {PC-ALM, BP}
11. **realtime モードは評価・ckpt ができない**。M14〜M16 は仮想時間で回す前提でよい
12. 3.14t 上で hydra / optuna / trackio が入るかは最初に確かめる（入らなければランチャー・探索側だけ別の Python で動かし、run は subprocess で 3.14t を呼ぶ構成も可能）
13. この Mac では submodule が未初期化。実行は GPU 機で行う前提で、ランチャーは ssh で各機の checkout（同じコミット、`just mujou-status` OK）を使う必要がある
