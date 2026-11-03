# 地図: データと評価（exp/video の balls / easy / evaluate / decode / offline / egodex）

2026-11-03。M14〜M16 に向けた再編のための読み取りメモ。コードは変更していない。行番号は main（06a2a11）時点。

## 0. 依存の向き（再編で切るところ）

```text
balls.py  ← (torch だけ)                         Frame, Physics, Video, Streams, render, eval_clips, ordered
egodex.py ← balls.Frame, imageio_ffmpeg          decode(), prepare(), EgoDexStream, egodex_clips
evaluate.py ← balls.Frame, model.(DynamicsStack, Network, Top), lib.metrics
run.py:source(config) ← balls, egodex, config    ★ストリームと評価クリップの唯一の工場
decode.py ← evaluate, model.decoder, run.(load_checkpoint, source), config
offline.py ← evaluate, model.*, run.(save_checkpoint, source), lib.sigreg
easy.py ← config, lib.sweep.run_all, stage0.(RUN_MODULE, SEEDS)
```

- `evaluate.Snapshot.of(net)` が `model.Network` の内部名（`encoder_names`, `dynamics_names`, `.compare`, `.residual`）に依存。M14 のエンコーダなし Network では作り直しが要る
- `decode.py` と `offline.py` はどちらも `run.source` を経由してデータを得る。データ層を独立させるなら `source` を data モジュールに移すのが自然
- checkpoint は `torch.save(Snapshot)`（pickle、`weights_only=False`、run.py:99-110）。モジュールのクラス名を変えると旧 checkpoint は読めない（互換不要なので問題なし）

## 1. 玉の物理（balls.py）

### 定数・型

- `RADII = (0.06, 0.08, 0.10)`（balls.py:32）。`balls=k` は先頭 k 個。`radii=` で上書き（M13 の玉 1 つは `radii=(0.10,)` などを `DataConfig.radii` で指定）
- `Physics`（:35-60, frozen, kw_only）: `gamma0=1.0, gamma_amp=0.5, e0=0.6, e_amp=0.5, drag=0.1, period=600.0, conserve=False`
  - `gamma(t) = gamma0 (1 + gamma_amp sin(2πt/T))`、`restitution(t) = e0 (1 + e_amp sin(2πt/T + π/2))`
  - `easy.EASY` は `gamma_amp=0, e0=1, e_amp=0, drag=0, conserve=True` → γ≡1, e≡1, 抵抗なし（時間変化なし）
- `Frame(NamedTuple)`（:63-69）: `image (1,size,size)`, `state (4*balls,)`, `params (2,)` = (γ, e), `reset: bool | Tensor`。スタック時は先頭に次元が付く
  - **`state` のレイアウト**: 玉ごとに `[x, y, vx, vy]` を連結（`[x0,y0,vx0,vy0,x1,y1,...]`）。`state.reshape(..., balls, 4)` で玉に分かれる。位置の列は `idx[:, :2]`、速度は `idx[:, 2:]`（evaluate.py:240-246 と同じ）
  - **速度の単位は「1 秒あたり」**（`p += h*v`, h は秒）。1 フレームの位置の変化は `v/30`。初速は各軸 std 0.5/s（`draw_balls`, :245-254）。M13 の README では平均 0.026/フレーム（約 0.8 画素@32px）。MILESTONES の「速度は約 0.03/フレーム」は位置差のことで、state の速度の値そのものは ~0.5〜1.5 の桁。標準化はこの違いを吸収するが、等速の外挿は「最後の 2 フレームの位置差」で作る（MILESTONES どおり）ので単位は問題にならない
  - 箱は $[0,1]^2$、y 上向き。玉の中心は `[r, 1-r]` の範囲
- 積分: 半陰的 Euler、`substeps=8`/フレーム（`advance`, :257-280; `_step`, :83-110）。速度 → 位置 → 壁の反射。玉どうしは**衝突しない**（玉ごとに独立 → 段 2 は玉ごとの動力学の重ね合わせ）
  - 反射: `q = wall + e (wall - p)`（位置の鏡映）。`conserve=False` は `v = -e v`、`conserve=True` は `_reflected`（:72-80）で $\tilde E = v^2/2 + g y - g h v/2$ を保つ根（`e` 倍）。x 軸は g=0
  - 各 substep は始点の時刻の物理を使う
- `Video`（:127-200）: `size=32, balls=3, radii=None, frame=1/30, physics=Physics(), episode=(3.0,6.0), substeps=8, seed (必須), start=0.0`
  - `episode=None` で 1 エピソード（引き直しなし、`reset` は最初のフレームだけ True）
  - `next()`: リセットなら新しい玉、そうでなければ 1 フレーム進める → state, `render`, params を返す
  - 純 Python の float リストで回す（ベクトル化なし）。テスト `test_thirty_minutes_take_less_than_a_minute`（tests/video/test_balls.py:220）で 1 本 54,000 フレーム < 60 s → **約 1 ms/フレーム/ストリーム**。B=128 だと 1 フレームあたり ~0.1 s 程度（推定。手元に torch がなく実測していない）→ 30 分 = 54,000 フレームで物理だけで ~1.5 時間の CPU。**多数の実験を回すなら torch でベクトル化した物理（B×balls をまとめて）か、軌跡の事前計算・キャッシュが必須**。ベクトル化する場合は既存の `Video` とビット一致させるテストがオラクルになる（`Streams` のストリーム i = `Video(seed=seed+1_000_000*i)`）
- `Streams`（:206-242）: `Streams(n, seed=, spread=False, **video)`。ストリーム i の seed は `seed + STREAM_SEED_STRIDE*i`（`STREAM_SEED_STRIDE = 1_000_000`）。`spread` は開始時刻を `i T / n` にずらす。`next()` → `image (n,1,S,S)`, `state (n,4*balls)`, `params (n,2)`, `reset (n,) bool`
  - 乱数の消費順: `draw_balls` は `torch.rand(2)` を玉ごと → `torch.randn(2)` を玉ごと。エピソード長も同じ Generator から（`episode=None` なら消費しない）
- `ordered(video, frames, order, clip, replay, seed)`（:326-372）: `"stream" | "shuffled" | "replay"`。B>1 の `Streams` では使わない（`DataConfig.__post_init__` が `streams>1` を `order=="stream"` に限定）

### 描画 `render(states (n,4*balls), size, radii) -> (n,1,size,size)`（:113-124）

- 玉ごとにガウスの塊、σ = r/2、画素中心 `(arange(size)+0.5)/size`、行 0 が上（y を反転）。和を `clamp(max=1)`。グレー 1ch、値は [0,1]
- 解像度は `DataConfig.size`（既定 32）。M14 段 3 は 32×32 を予定
- **予測した状態を描くのにそのまま使える**（M14 の見た目の確認: 予測 state → `render` → 真のフレームと並べる）。`render` は位置しか使わないので速度の列はダミーでよいが、入力は `(n, 4*balls)` のレイアウトが要る（段 1・2 の位置だけの観測から描くときは速度 0 を挿して並べ直す）

### 評価クリップ `eval_clips(...)`（:283-323）

- 既定 `size=32, balls=3, radii=None, frame=1/30, physics=Physics(), phases=8, clips=4, length=128, substeps=8, seed=12345`
- 各相 p について開始時刻 `t0 = p T / phases`、クリップごとに**新しい玉**（同じ Generator から `draw_balls`）を引き、128 フレーム進める（エピソードの引き直しはない）
- 戻り値 `Frame(image (P,C,L,1,S,S), state (P,C,L,4*balls), params (P,C,L,2), reset=False)` = 既定で (8,4,128,...) → 32 クリップ
- `run.source` は `eval_clips(size, balls, radii, frame, physics)` だけを渡す（phases/clips/length/seed は既定のまま。run.py:134-136）
- EASY では γ・e が時間で変わらないので 8 相は実質 32 本の独立なクリップ。`halves` で相ごとに前半 2 本 / 後半 2 本に分ける（→ 標本 16 本ずつの out-of-sample。少ない）
- 32 クリップ × 16 観測 → 評価の標本が少ない。M14 で物差しを安定させるなら `clips`/`phases` を増やす（純 Python なので 128 フレーム × クリップ数ぶんの時間）
- 注意: ストリーム（`Streams`）の玉とクリップの玉は別の seed 系列。`calibration_images`（model.py:155-164）は `Video(size=d.size, seed=seed+10_000)` で**既定の物理・玉 3 つ**を使い、EASY の物理も radii も反映しない（エンコーダの標準化用。M14 には無関係、M15/M16 で流用するなら直す）

## 2. 評価（evaluate.py）

- `HORIZONS = (1, 5, 10, 20)`（:29）。`observed=16` が既定
- `Snapshot`（:32-52）: `encoder (o→z)`, `step ((z_{k-1}, s_{k-1}) → s_k)`, `readout (s → ẑ or Δz)`, `residual=False`。`Snapshot.of(net)` は `Network` から deepcopy
- `run_states(z (n,T,D), step, readout, d_s, observed, steps, residual) -> (states (n, observed+steps, d_s), preds (n, steps, D))`（:59-87）
  - $s_0=0$。`k=1..observed-1` で `s = step(z[:,k-1], s)`（教師強制）。その後 `z_in = z[:, observed-1]` から開ループ: `s = step(z_in, s); z_in = readout(s)`（residual なら加算）
  - `preds[:, h-1]` が フレーム `observed-1+h` の予測。**h=1 は「最後の観測の次」**
  - `step` の d_s は状態全体の幅（`d_s * dyn_nodes`。decode.py:131 と run.py:162）
- `rollout_errors(...)`（:90-107）: 各 h で `mean((preds[:,h-1] - z[:,observed-1+h])^2)`（1 次元あたり）
- `persistence_errors(z, observed, horizons)`（:110-117）: 写し `z[:,observed-1]`
- `mean_errors`（:120-128）: 全クリップ・全フレームの平均 `z.mean((0,1))`
- `halves(phases, clips)`（:131-135）: 相ごとに前半のクリップ True、phase-major
- `shrink_errors` / `shrink_predictions(z, half, observed, horizons)`（:138-176）: $\hat z = m + \lambda_n (z_{last} - m)$、λ_n は片方の半分で最小二乗して他方を予測（out-of-sample）
- `rollout_shape`（:179-203）: `amp{n}` = 予測の分散 / 目標の分散、`corr{n}`
- `evaluate(snap, clips, d_s) -> dict`（:206-256）
  - 画像を 1024 ずつエンコード → `z (P*C, T, D)`。`var = z.var(0).mean()`（クリップ間・次元平均）
  - キー: `z_var`, `rank`（`effective_rank` on 全フレーム）, `rollout{h}`, `persist{h}`, `mean{h}`, `shrink{h}`（**すべて var で割った相対値**）, `amp{h}`, `corr{h}`
  - `clips.state.shape[-1] > 0` なら probe: `z` と `s`（全 T を教師強制した states）から `[state, params]` へのリッジ（`probe_r2`、前半クリップで学び後半で評価）。`probe_{z|s}_{pos|vel|phys}`
  - 注意: probe の `phys` は γ と e。EASY では定数なので R² は 0/NaN 付近で意味がない（total=0 → 除算）。M14 以降で probe を使うなら `phys` を除くか条件付きにする
- **M14 で使い回せる部分**: `run_states`・`rollout_errors`・`persistence_errors`・`mean_errors` は z を「観測」に置き換えれば観測空間の物差しになる（encoder=identity）。ただし `evaluate()` は var で割った相対値を返すので、M14 の「1 次元あたりの二乗誤差（標準化した観測空間）」とは尺度が違う。等速の外挿はまだ無い（新規）
  - 等速の外挿（新規）: 位置 `p_last = x[:, 15]`, `v = x[:, 15] - x[:, 14]`, `p̂_h = p_last + h v`。段 0 の観測（速度も観測）では速度の予測を何にするか決める必要がある（`v̂ = v_obs` 一定、または位置差）。オラクル「重力 0 で壁に当たらない区間では誤差 0」は `Physics(gamma0=0)` と壁から遠い初期値で作れる（半陰的 Euler、重力 0 なら位置は厳密に線形）
  - 観測の標準化は評価クリップの平均と std で各次元（MILESTONES）。`mean_errors` の平均と同じ統計を使えば整合する

## 3. デコーダ（decode.py、M15・M16 の判定の道具）

- CLI: `python -m exp.video.decode <run dir> [--steps 2000] [--frames 16384] [--device cuda:0]`（:178-217）
- 定数: `OBSERVED=16, LR=1e-3, BATCH=256, SEED=0, COLUMNS=(1,2,5,10,20), SHOWN=4`（:45-48）
- 流れ: `<run>/config.json` の `config` から `Config` を復元 → `stream_frames(config, n)` でストリームの先頭 n フレーム（**B ストリームを平らにして数える**。B=128, n=16384 なら時間方向は 128 ステップ＝約 4 秒分だけ）→ `source(config)` でクリップ → `ckpt/*.pt` を順に
- `fit_decoder(z (n,D), images (n,C,H,W), steps) -> nn.Module`（:70-85）: `model.decoder(d_z, (C,H,W))`、Adam(1e-3) + CosineAnnealing、batch 256、seed 0。エンコーダは止めたまま（z は `no_grad` で事前計算）
- `model.decoder(d_z, shape)`（model.py:198-213）: Linear → 4×4 → ConvTranspose で倍々。**H=W が 8 以上の 2 の冪**。出力に sigmoid なし（`_tile` で clamp）
- `decode_checkpoint(snap, config, clips, train, out, seconds, steps) -> dict`（:115-175）
  - `recon` = 全クリップのフレームの再構成 MSE（画素あたり）
  - 各 h: `img_rollout{h}`（rollout の ẑ を復号）, `img_persist{h}`, `img_mean{h}`（z 平均の復号）, `img_shrink{h}`、デコーダなしの物差し `pix_copy{h}`（最後の観測フレーム）と `pix_mean{h}`（**デコーダの学習フレーム `train` の平均画像**。評価クリップの平均ではない）
  - `evaluate()` の全キーも同じ行に入れる
  - PNG: `SHOWN` 本のクリップ × 4 行（真のフレーム / z の復号 / rollout の復号 / shrink の復号）× 列 (0, 1, 2, 5, 10, 20)。`_tile` で 96px 相当に拡大、`write_png` は zlib だけの自前 PNG 書き出し（:88-112）。カラー 3ch もそのまま扱える
  - `d_state = d_s * dyn_nodes`（config.model から）
- 出力: `<run>/decode/{config.json, metrics.jsonl, <seconds:05d>.png}`。`easy.summary(run)`（easy.py:79-90）が最後の行から `img/mean{n} = img_rollout/pix_mean`, `img/copy{n}`, `z/shrink{n}` を作る（HORIZONS=(5,10,20)）
- M13 の物差しの数値（results/m13.md）: EASY の玉 3 つ・32px で `pix_copy` 5/10/20 = 0.0221/0.0289/0.0292、`pix_mean` = 0.0144/0.0149/0.0145
- **M15・M16 への流用**: 「各チェックポイントに後からデコーダを学ぶ」仕組みはそのまま使える。直すべき点
  - `Snapshot` に依存（encoder/step/readout）。M15 は動力学がない（encoder だけ、ViT のトークン出力なら z の形も違う）→ `decoder` の入力 `(n, D)` を前提にしている点、`run_states` を必ず呼ぶ点を外す
  - `stream_frames` / `source` が `Config`（`DataConfig.kind ∈ {video, egodex}`）に縛られている。Walking Tours 用のデータ源を足すなら、デコーダの学習フレームをデータ層から受け取る形にする
  - デコーダ 2000 step × checkpoint 数を逐次に回すので、checkpoint が多いと遅い。GPU 1 枚で並列化の余地（checkpoint ごとに独立）
  - `pix_mean` を学習フレームの平均にしている点は M14 の「評価クリップの平均」と定義が違う。統一するなら決める

## 4. オフラインの上限（offline.py）

- CLI（:79-93）: `--config <json>`（`Config`。無ければ `DEFAULT = Config(name="offline", model=ModelConfig(d_z=64, d_s=256, z_norm=True), data=DataConfig(kind="video"))`）, `--learn-encoder`, `--sigreg 15.0`, `--local`, `--batch 128`, `--blur`, `--sgd`, `--lr 1e-3`, `--frames 30000`, `--steps 3000`, `--every 1000`, `--device cuda:0`, `--save <dir>`
- `WINDOW, OBSERVED = 32, 16`（:49）
- `window_starts(resets (T,B), window) -> (n,2)`（:57-63）: (先頭フレーム, ストリーム) のうち窓内の 2 フレーム目以降にリセットが無いもの。引き直しなしなら全位置
- `stream_windows(config, frames)`（:66-76）: `frames // B` 時刻ぶん `video.next()` → `images (T,B,C,H,W)` を GPU に全部載せる。**B=128 で `--frames 30000` なら T=234 時刻しかない**（M13 では 131,072 フレーム = 1024 時刻）。物理が遅いので前処理に時間がかかる
- 学習ループ（:141-168）: 窓をランダムに B 個 → `x (B, 32, C, H, W)` → `z = encoder(x)`（`--learn-encoder` でなければ no_grad）→ `s=0` から k=1..31 で `s = step(z_in, s)`, `ẑ = readout(s)` (+z_in if residual), 損失 = 31 予測の平均 MSE。k<16 は `z_in = z[:,k]`（教師強制）、k≥16 は `z_in = ẑ`（開ループ、**BPTT が開ループの予測を通る**）。`--local` で入力と帰還を detach
  - 目標への credit の重み `w = target_weight(m)`（`stop_target` なら `m.target_weight`、そうでなければ 1）
  - SIGReg: `epps_pulley(samples @ a)` をランダム方向 `m.directions` 本で（`--learn-encoder` のとき）
  - 最適化: Adam（既定）か SGD。スケジューラなし
  - `--every` ごとに `evaluate()` → `KEYS = (rank, z_var, rollout1, rollout10, persist10, mean10)` を print、`--save` なら `ckpt/<step:05d>.pt`（名前はステップ数。秒ではない）と `config.json`（`{"config": ..., "offline": args}`）
- エンコーダは `encoder_stages` + `standardize_(calibration_images)`（LSUV）で初期化。M14 ではエンコーダを外し、観測 x を直接 z とすればよい（`readout` の出力次元を観測次元に）
- **M14 への流用**: 窓の作り方（`window_starts`）、開ループ BPTT の損失ループ、`DynamicsStack`/`Top`/`dynamics_modules` の組み方はそのまま使える。画像の代わりに `Frame.state` を窓にする（`(T,B,4*balls)` で GPU に載せても小さい）。`dynamics_modules(m)` は `MinGRU(m.d_z, m.d_s)` なので `d_z` を観測次元（段 0: 4、段 1: 2、段 2: 6）に設定する。適格度トレース（`ModelConfig.trace`）はオフラインには関係しない（BPTT が上限）
- ログは stdout の JSON 行だけ（Recorder を使わない）。hydra/trackio に載せ替えるなら、ここも run と同じ記録経路にまとめる

## 5. EgoDex（egodex.py）と動画デコードの流用（M15 の Walking Tours）

- 定数: `SIZE=64, FPS=30, ROOT=Path("/datasets/egodex")`（:49-51）
- `decode(path, *, frames=None) -> (n,3,64,64) uint8`（:75-86）: `imageio_ffmpeg.get_ffmpeg_exe()` の ffmpeg を subprocess で起動、`-vf scale=-2:64:flags=area,crop=64:64`（高さ 64 に面積平均で縮小 → 中央 64 列）、`-pix_fmt rgb24 -f rawvideo -` を stdout で受ける。`-threads 1`。**動画全体をメモリに読む**（1〜3 時間の 4K 動画には使えない。ストリーミング読み出しが要る）
- `prepare(out, root, jobs=16, block=75.0, passes=2, min_episodes=50, clips=48, length=128, seed=0)`（:172-244）→ `datasets/egodex64/`:
  - `stream.u8`: `(N,3,64,64)` uint8 C 順の生バイト（全エピソードを連結）
  - `clips.u8`: `(T=48, L=128, 3, 64, 64)` uint8（test split の各タスクから 1 エピソードの先頭 128 フレーム）
  - `index.json`: `episodes`（path, task, split, frames, block, round, start, metadata_frames）, `frames`（総数）, `clips.shape`, `settings`, `ffmpeg`（バージョン文字列）, `seconds`
  - `.tmp` に書いてから `os.replace`（原子的）
- `EgoDexStream(path, seconds=None, offset=0)`（:251-298）: `torch.from_file(..., shared=False)` で memmap、`next()` → `Frame(image (3,64,64) float/255, state=zeros(0), params=zeros(0), reset=エピソード先頭)`。`IndexError` で終端。`task_of(k)`, `pass_of(k)`
- `egodex_clips(path) -> Frame(image (1,T,L,3,64,64), state/params (1,T,L,0), reset=False)`（:301-314）。先頭の 1 は相の次元（`eval_clips` に形を合わせる）→ `evaluate()` の probe は state 幅 0 で飛ばされる
- `calibration_images(path, n=256, seed=0)`: ストリームの前半から一様に n 枚
- **Walking Tours（M15）への流用**
  - `/datasets/walking_tours` は compose.yaml:13-14 で `${WALKING_TOURS_DIR:-./datasets/walking_tours}` をマウント（egodex と違い `:ro` なし）。devcontainer の `containerEnv.WALKING_TOURS_DIR=/datasets/walking_tours`
  - u8 の memmap + index.json の形式（`prepare` → `Stream` → `clips`）はそのまま流用できる。Walking Tours は 1 本 1〜3 時間・4K・60 fps なので、事前に縮小・間引き（`-vf fps=<n>,scale=...,crop=...`）して u8 に書くのが現実的。64×64×3 で 30 fps・1 時間 ≈ 108,000 フレーム ≈ 1.3 GB/本（128×128 なら 4 倍）
  - `decode()` は `SIZE` が定数で、全フレームを 1 回の `subprocess.run` で受ける。サイズ・fps・`-ss`/`-t`（区間）を引数にし、`subprocess.Popen` で stdout を固定長ずつ読む形に一般化すれば長い動画を流せる。4K の H.264/VP9 のデコードは CPU を食う（`-threads 1` を外す、または動画ごとに並列）
  - アスペクト: `scale=-2:H` + 中央 crop は 16:9 の両端を捨てる（Walking Tours でも同じ方針でよいかは要判断）
  - 評価クリップ: `(1, T, L, 3, H, W)` の `Frame`（state/params 幅 0）を作れば `evaluate()`・`decode.py` の形にそのまま乗る。N フレームのまとまり（V-JEPA のチューブ）は `(..., N, C, H, W)` を窓から切る形にすれば `window_starts` が使える（`reset` は動画の境界）
  - ライセンス確認・顔とナンバーのぼかしの有無は MILESTONES の要求（取得時）
  - 合成動画（玉）で JEPA を始める場合は `Streams` + `render` がそのまま使える（1ch、32px。ViT のパッチなら 4×4 や 8×8）。`Frame.state` があるので probe（位置・速度）が使える

## 6. easy.py（M13 の設定の置き場、再編で消してよい）

- `FRAME=1/30, STREAMS=128`。`EASY = DataConfig(kind="video", seconds=1800, gamma_amp=0, e0=1, e_amp=0, drag=0, conserve=True, redraw=False, streams=128)`（:29-39）。`radii` は既定（玉 3 つ）。`spread` なし
- `SMALL = ModelConfig(kappa=0.13, z_norm=True, open_loop=0.5, open_cycle=20, buffer=1023)`、`BIG = replace(SMALL, width=64, d_s=512, dyn_nodes=3)`（:42-44）。M14 の「d_s と動力学のノードの数は M13 の大と小」= (d_s=256, dyn_nodes=1) と (d_s=512, dyn_nodes=3)（`ModelConfig` 既定: d_z=64, d_s=256, width=32, dyn_nodes=1, stop_target=False, target_weight=0.0, trace=False。config.py:77-116）
- `LR, WARMUP, CHECKPOINT = 0.03, 300.0, 300.0`、`OptimConfig(kind="sgd", tau=FRAME/LR, warmup=WARMUP)`（学習率を tau で表す）
- `online_configs` は seed × {pc, bp} で `device="cuda:1" if (seed+bp)%2 else "cuda:0"`（GPU 2 枚前提をハードコード）→ フリートのランチャで置き換える
- `run_all(RUN_MODULE, todo, out, jobs)`（exp/lib/sweep.py）で並列実行。`summarize` は `*/decode/metrics.jsonl` の最後の行

## 7. テスト（オラクル）で流用できるもの

- tests/video/test_balls.py: エネルギー保存（`conserve` で $\tilde E$ 厳密）、壁での速さ e 倍、引き直しなし、radii、render の重心、eval_clips の形と seed 固定、`Streams` の seed 規約、`ordered`。**物理をベクトル化するなら、これらと「既存 `Video` とのビット一致」をオラクルに**
- tests/video/test_evaluate.py: 恒等モデル → 写しの誤差、傾きの n 倍、shrink の λ、rollout_shape（倍々で 4^n）。観測空間の物差しにも同じオラクルが効く
- tests/video/test_offline.py: `window_starts` がリセットをまたがない、`main` が学んで保存
- tests/video/test_decode.py: デコーダの形、z が決めるフレームを学ぶ、PNG の画素、`_tile` の配置、checkpoint のデコード（GPU 版あり）
- tests/video/test_egodex.py: ffmpeg で作った色の動画で `decode` の形・色順（CHW）・先頭 frames・中央 crop → Walking Tours 用の一般化デコーダにも同じ作り方が使える

## 8. 再編のための要点（まとめ）

1. データ層を `Config`/`run.source` から切り離す: 「ストリーム（`next() -> Frame`、B 本）」「評価クリップ（`Frame (P,C,L,...)`）」「デコーダ用の学習フレーム」の 3 つを返す工場を、kind（balls-state / balls-image / walking_tours / egodex）ごとに用意する
2. M14 の観測: `Frame.state` から列を選ぶ（段 0: 玉 1 つの 4 列、段 1: 位置 2 列、段 2: 玉 3 つの位置 6 列 = `state.reshape(...,3,4)[...,:2].flatten(-2)`）→ 評価クリップの平均・std で標準化。逆変換のテストが MILESTONES のオラクル
3. 物理の速度がボトルネック（純 Python、B=128 で ~0.1 s/フレームと推定）。多数の実験の前にベクトル化（torch、B×balls、ビット一致テスト）か、seed ごとの軌跡キャッシュ（state だけなら 54,000×128×12×4 B ≈ 330 MB）を作る
4. 物差し: 平均・写しは既存関数（z→観測）、等速の外挿は新規、真の物理は誤差 0（`advance` を使えば作れる）。尺度は var で割る相対値と、標準化した観測空間の絶対値のどちらで判定するかを統一する
5. 見た目: 予測 state → `render` → `_tile`/`write_png`（decode.py）で PNG。軌跡の重ね描きは新規（matplotlib は依存にない。PNG を自前で描くか、依存を足す）
6. M15: `egodex.decode`/`prepare`/`EgoDexStream` を、サイズ・fps・区間を引数に取るストリーミング版に一般化して Walking Tours を u8 memmap に前処理。`decode.py` の「checkpoint ごとに後からデコーダ」は Snapshot 依存を外して流用
7. M16: 動力学の誤差がエンコーダに credit を送らない（`target_weight=0` かつ入力 z も detach）。offline.py の `--local` + `w=0` と同じ形。評価は decode.py の `img_rollout{n}/pix_mean{n}`, `/pix_copy{n}` をそのまま使える
