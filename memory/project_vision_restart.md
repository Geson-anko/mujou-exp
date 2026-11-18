---
name: project-vision-restart
description: 2026-11-18、M18 の後に exp/vision として V0 から番号を振り直す。PC-ALM だけで VAE、ViT と ResNet、時定数の momentum、run_manually
metadata:
  type: project
---

2026-11-18、M18 の完了を受けて、ユーザーは基礎を詰め直すため、新しく `exp/vision` で V0 から番号を振ると決めた。計画は exp/vision/PLAN.md。

- 目的: Walking Tours の単一ストリームから、動力学モデルが予測しやすく、入力の情報もある程度保つ低次元の特徴量を学ぶエンコーダ
- 目的の式は VAE（ELBO）。潜在空間をガウス化するため（再構成誤差だけの自己符号化器ではない）
- 学習は PC-ALM だけ。BP の run は入れない。実行は `mujou.run_manually`（Jacobi）
- データ: 0.1 秒間隔で 4・8・16 枚を積む。解像度 64
- モデル: ViT と ResNet、ノード数は固定、S・M・L はノードの中の層の数。パラメータ数を揃える
- optimizer: SGD → 時定数 τ の momentum（勾配の指数移動平均の形）→ RMSProp・AdamW。学習率は安定する程度に選ぶ
- 判定: 評価で見る損失は再構成誤差だけ（ELBO 全体・KL・bpd は判定に使わない）。再構成誤差・`recon_blur`・`dyn_1/copy_1` が初期化時より下がること、目視。copy に勝つかは問わない。linear probe（Imagenette）は参考
- posthoc の拡張: 最後のフレームを flow matching の軽量の拡散デコーダで復元（目視・I(x;z) の推定・Inception V3 の特徴の誤差）

**Why:** M18 までマイルストーンを積み重ねて設定が入り組んだ。再確認も兼ねて、足元から固め直すため。

**How to apply:** 順番にはこだわらず、VAE が安定して学べる設定と限界を行き来して探る（ユーザー）。全部の組み合わせは回さず、1 軸ずつ動かす。初期化は安定しないときに考える。関連: [[project-vae-stream-next]]、[[feedback-simple-fundamental]]、[[feedback-learn-setting-first]]、[[feedback-realtime-speed]]
