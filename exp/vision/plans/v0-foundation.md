# V0: 土台（実装の仕様）

[../PLAN.md](../PLAN.md) の V0 を、実装できる粒度に落とした仕様。2026-11-18 に決めた。移す部品の地図は [memory/agents/main/vision-v0-map.json](../../../memory/agents/main/vision-v0-map.json)（`exp/video` のファイルと行の対応、mujou の新しい API、落とし穴）。

## 前提: mujou の変更（origin/main、c0e7ab7）

- `mujou.testing` は消えた。`mujou.run_manually(root, *, timer=None, inputs=None, state=None, workers=None) -> ManualRun`、`from mujou import ManualRun`
- `ManualRun.step(dt)` はノードを渡さないと全ノードを同時に更新する（Jacobi）。前向きは 1 ステップに 1 ホップずつ進む
- `pc.Layer` は緩和してから、その δ で `g_in` を返し、学ぶ。hops = ノード列のホップ数のとき、層 $i$ が BP と同じ credit を持つのはステップ $2\cdot\text{hops} - i$ から
- 時刻 0 の書き込みは、初期のゼロ Activity と同じ時刻 0 になる（Host が最初のフレームの予測を取り違える）

## 決めたこと

### データ

- データの版は `ad6e56b5995e`（Walking Tours、10 fps、64×64、全ホストの `/data/geson/mujou-exp/.fleet/data/` にある）。`stride = 1`、`advance = 1` で、0.1 秒間隔の stack が 0.1 秒ごとに 1 つ来る
- stack の枚数 $n \in \{4, 8, 16\}$、既定は 8。$B = 1$ の単一ストリーム
- held-out は Zurich。calib / val / test は `eval_windows`（`TOURS_SEED = 4242`）を時間の順に 3 等分する（`exp/video` と同じ）
- Walking Tours の読み込みは、実験によらない部品として `exp/lib/tours/` に移す（`walking_tours.py`・`manifest.json`・`StackStreams` と `StackedTourFrames` だけを持つ `stacks.py`）。玉の合成データと並べ替えの対照は移さない

### 時間

| 量 | 値 |
| -- | -- |
| frame | 0.1 秒（10 fps のストリームの 1 枚） |
| `pc.TimeConstants` | $h$ = 4e-3、$\lambda$ = 2e-3、$r$ = 4e-3（period 2 ms）。1 フレームは 50 ステップ |
| ノードの配置 | Input → エンコーダ 4 ノード → Latent → デコーダ 2 ノード → GaussOut（hops = 8、$2\cdot$hops = 16 ステップ = 32 ms） |
| 学習 | 各ノードはフレームごとに 1 回、credit が届いて緩和した後（フレームの終わり近く）に学ぶ。`learn_interval` の値は実装で導き、テストで「各ノードがフレームごとにちょうど 1 回学び、最初のフレームの勾配が float64 の BP の勾配と一致する」を確かめる |

ノード数（エンコーダ 4、デコーダ 2）は、モデルの種類と大きさによらず固定する。

### モデル

入力の stack $x \in \{0..255\}^{n \times 3 \times 64 \times 64}$ は $x/255$ にし、フレームをチャネルに積んで $3n$ チャネルの画像にする（early fusion）。ViT と ResNet で同じ入力の形にするため。

| | ViT | ResNet |
| -- | -- | -- |
| ノード 1 | patch 16 の埋め込み（$3n \cdot 16 \cdot 16 \to w$、16 トークン、固定の 2 次元 sincos の位置）+ $k$ ブロック | stem（3×3 conv、stride 2、64→32）+ $k$ 個の basic block |
| ノード 2〜3 | $k$ ブロック | stride 2 で縮めて $k$ 個の basic block（32→16→8） |
| ノード 4 | $k$ ブロック + LayerNorm + 平らにして Linear（$16 w \to 2D$） | stride 2 で縮めて $k$ 個の basic block（8→4）+ 平らにして Linear（$16 C_4 \to 2D$） |

- 大きさ: ノードあたりのブロック数 $k$ が S = 1、M = 2、L = 4
- 幅: ViT は $w$ = 256（4 head、MLP 4 倍、pre-norm）。ResNet のチャネルは、M で ViT のエンコーダのパラメータ数と ±10 % に収まるように選び、S・L でも同じチャネルを使う。パラメータ数の表をテストで出す（`exp/vision/models.py` の `param_table()`）
- ResNet の正規化は、ストリーム（$B$ = 1）で使えるように GroupNorm にする（BatchNorm は $B$ = 1 では統計が壊れる）
- 潜在 $D$ = 256。頭の Linear の $\log s^2$ の行と bias は 0 で始める（$s^2 = 1$、M18 と同じ）
- デコーダ（どの大きさでも同じ、軽い）: Latent の後に、ノード 1 は Linear（$D \to 256 \cdot 4 \cdot 4$）+ 2 段の上げ（4→8→16、conv 3×3 + GroupNorm + 活性化、128・64 チャネル）、ノード 2 は 2 段の上げ（16→32→64、32・32 チャネル）。GaussOut が 3×3 conv で $3n$ チャネルに読み出し、学ぶスカラーの $\log\sigma$ を持つ
- 尤度は dequantize したガウス（$y = (x + u)/256$、`exp/lib/elbo.py`）。目的の式は ELBO
- 初期化は torch の既定に従う（ViT は `exp/video/jepa/vit.py` の規則でよい）。上の $\log s^2$ の行だけ例外
- `Latent` と `GaussOut` は `exp/video/vae/pc.py` から移す。`Latent` は新しい `pc.Layer` と同じく、緩和してから δ と `g_in` を計算する。$u$ と $\varepsilon$ はホストが引き、活動の $y$ に載せて運ぶ（Jacobi でフレームがずれても対がずれない）

### 学習の設定（出発点）

- optimizer は `mujou.pc.optim.SGD(tau=...)`（全ノードで同じ）。$\tau$ は試しの run で安定する値を 10 倍刻みで探す
- 時定数の momentum と RMSProp は mujou の PR（`pc.optim.SGD(tau, tau_m)`、`pc.optim.RMSProp(tau, tau_v)`）が入ったら使う
- run の長さは仮想時間 1800 秒（18000 フレーム）。checkpoint は 0・60・300・600・1200・1800 秒

### 評価（posthoc、checkpoint ごと）

- **t = 0 の checkpoint を保存する**（`exp/video` には無かった）。「初期化時より下がる」はこれと比べる
- 再構成誤差: システムのデコーダの、val の stack の二乗誤差を平均の stack の二乗誤差で割ったもの（`stack_mse`）
- `recon_blur`: 後から学ぶデコーダ（`exp/video/measure/posthoc.py` の `decoder`・`fit_decoder`、学習率 3e-4、$z = \mu$ を 1 つのスカラーで標準化）で最後のフレームを復元する。64 px なので $\sigma$ = 1（128 px の $\sigma$ = 2 と同じ扱い）。当てはめの揺れ（M18 で sd 約 0.0035）があるので、デコーダの seed を 4 つ変えて平均と sd を出す。`provenance.configure_determinism` のもとで回す
- `dyn_1/copy_1`: `exp/lib/linear_dyn.py`（ridge は calib で選ぶ）
- 物差し: 画素の PCA（$D$ = 256）の符号の `recon_blur` と `dyn_1/copy_1`
- 画像: val の 10 窓の、真の最後のフレーム・システムのデコーダ・後から学ぶデコーダの 3 行（2 倍に拡大）
- Imagenette の linear probe は V0 の最後に足す（データを 64 px で作り直す。std-01 にしかない）
- ELBO 全体・KL・bpd は記録してよいが、判定には使わない

### 実行の土台

- `exp/vision/{main,compose,loop}.py`・`schema/`・`conf/` を `exp/video` から移し、プログラムは `vae` 1 つにする
- fleet の `submit` が job の entry に合わせて compose のモジュールを選ぶようにする（今は `exp.video.compose` に固定）
- `exp/lib` と `tests/lib` を `mujou.run_manually` に移す。時刻 0 の衝突を `Host` 側で直す
- `exp/video` は、移し終えたらコードとテストを外す（README・計画・results・レポートは残す）。コードは tag `video-final`（aa58248、mujou 7a8b72e）から辿れる

## 完了の条件

- `just run` が通る
- 小さなネットで、最初のフレームの各ノードの勾配が float64 の BP の勾配と一致する（Jacobi、上の `learn_interval`）
- 止めて再開した run が、止めない run とビットまで同じになる（CPU）
- bench: ViT と ResNet の S・M・L の、$B$ = 1 の速さ（単位/秒。リアルタイムは 10）と GPU メモリ
- mini-00 で出発点の設定の試しの run（120 秒）が走り、posthoc まで通る
