# vision: 単一ストリームから PC-ALM で VAE のエンコーダを学ぶ

計画は [PLAN.md](PLAN.md)、V0 の実装の仕様は [plans/v0-foundation.md](plans/v0-foundation.md)。`exp/video`（M0〜M18）の続きで、番号を V0 から振り直した。`exp/video` のコードは tag `video-final`（mujou 7a8b72e）にある。

## 実行方法

```bash
# 手元で 1 本
uv run python -m exp.vision.main +experiment=v0/base --out runs/x
# fleet（std-00〜03、mini-00 は試しだけ）
just fleet submit --batch <b> +experiment=v0/base --sweep optim.tau=1e5,1e6 --seeds 0
# posthoc（run の後、GPU のあるホストで。MUJOU_EXP_DATA はデータの置き場所）
MUJOU_EXP_DATA=/data/geson/mujou-exp/.fleet/data python -m exp.vision.eval <run_dir>
# bench（B = 1 の速さと GPU メモリ）
python -m exp.vision.bench --device cuda --frames 100
```

## V0: 土台の確認（2026-11-18）

snapshot `f0d3cbe8bc0d`（mujou 3d26a3c、PR Geson-anko/mujou#123 のブランチ）。

### bench（$B$ = 1、100 フレーム、単位/秒。リアルタイムは 10）

| | ViT-S | ViT-M | ViT-L | ResNet-S | ResNet-M | ResNet-L |
| -- | -- | -- | -- | -- | -- | -- |
| エンコーダのパラメータ | 6.83M | 9.99M | 16.31M | 5.91M | 9.44M | 16.50M |
| 4090（std-00） | 6.38 | 6.24 | 6.16 | 6.35 | 6.31 | 6.14 |
| 5070 Ti（mini-00） | 5.52 | 5.47 | 5.44 | 5.49 | 5.49 | 5.64 |
| GPU メモリ（4090、MB） | 121 | 155 | 221 | 115 | 155 | 231 |

速さはモデルの大きさによらずほぼ同じで、1 フレーム 50 ステップ × 9 ノードの、ステップごとのオーバーヘッドで決まっているとみられる（未確認）。リアルタイムには届かない。1800 秒の run は 4090 で約 48 分。

### 試しの run（mini-00、v0/base、SGD $\tau$ = 1e5、120 秒、seed 0）

`v0-base-5804bbe3-s0`。120 秒のあいだ値はすべて有限。loss は 110814 → −38725、ln σ は −0.85 → −1.81 と下がった。

| t（秒） | stack_mse | recon_blur（4 seed の平均 ± sd） | dyn_1/copy_1 |
| -- | -- | -- | -- |
| 0 | 6.707 | 0.2849 ± 0.0041 | 0.880 |
| 60 | 1.115 | 0.2568 ± 0.0015 | 0.874 |
| 120 | 2.036 | 0.2538 ± 0.0052 | 0.875 |
| 画素の PCA（$D$ = 256） | | 0.0615 ± 0.0027 | 0.897 |

目視（`posthoc/00120.0.png`）: システムのデコーダは、120 秒の時点でどの窓にもほぼ同じ画像（空と暗い地面）を出し、まだ $z$ を使っていない。`stack_mse` が 1 を上回るのはこのため。

## V1: 出発点の設定は安定して学べるか（回す前に決めたこと、2026-11-18）

**問い**: v0/base（ViT-S、8 枚、64×64、SGD）で、1800 秒のあいだ発散せず、初期化時より下がる $\tau$ はあるか。あれば、その中で最も小さい $\tau$（最も大きい学習率）を以降の出発点にする。

**腕**: SGD の $\tau \in \{10^4, 10^5, 10^6, 10^7\}$、seed 0、1800 秒、std-00・02・03 の 4090（std-01 は GPU が見えないので外す）。posthoc は 0・60・300・600・1200・1800 秒の checkpoint。

**判定**（val、1800 秒と 0 秒の比べ）

- 安定: stream の全行が有限
- 学べた: 次の 3 つをすべて満たす
  - `stack_mse` が 0 秒より小さい
  - `recon_blur` の差が、2 点の標準誤差を合わせたものの 2 倍より大きい（$\Delta > 2\sqrt{(s_0^2 + s_1^2)/4}$、$s$ は 4 seed の sd）
  - `dyn_1/copy_1` が 0 秒より小さい
- 目視: システムのデコーダが窓ごとに違う画像を出しているかを書く（判定には入れない）

**予想**: $10^4$ は発散する（合成データで発散した）。$10^5$ は安定して学べる。$10^6$ は学べるが遅い。$10^7$ はほとんど動かず `recon_blur` で外れる。

**反証**: どの $\tau$ も「学べた」を満たさなければ、次は学習則より前の段（目的の式の重み、デコーダの形、`learn_interval`）を疑う。
