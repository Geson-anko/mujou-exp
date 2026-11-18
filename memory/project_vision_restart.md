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

**V1 の結果（2026-11-18）**: SGD の τ は 1e5 が規則の上での出発点（1e4 は 2 秒で発散、1e5・1e6 は 3 条件を満たす、1e7 は dyn_1/copy_1 で外れ）。ただしシステムのデコーダは 1800 秒でも val のどの窓にも同じ画像（学習中の場面）を出し、z を使っていない。符号の実効ランクは 80.6 → 10.2。「初期化時より下がる」だけでは学べたかを分けられない。次の一歩はユーザーと決める（exp/vision/README.md の「V1: 結果」）

**V2〜V5 の結果（2026-11-18）**: 「学べた」（stack_mse < 1 を含む 4 条件）は RMSProp τ=1e4 の 1 本だけ（stack_mse 0.640、システムのデコーダが窓ごとに違う画像を出す）。SGD・momentum・モデルの大きさ・stack の枚数では、デコーダが z を無視したまま（dec_spread ≈ 0）。ただし走りながらの誤差は平均の stack の約 0.37 で、デコーダは z ではなく自分のパラメータでいまの場面を追っている（直前 64 秒の recent_stack_mse は場面をまたぐので、これを否定する根拠にならない）。KL は 300〜700 nat あり、つぶれてはいない。β は置かず学ぶ σ が釣り合いを決めるが、σ² は腕ごとに最大 9 倍違い（SGD 約 0.009、RMSProp 1e4 は 0.054）、optimizer の効果と交絡している（ユーザーが KL と再構成の釣り合いを気にしている、2026-11-18）。ResNet は ViT より悪い（S・L は τ=1e5 で発散）。AdamW・RMSProp の小さい τ は符号の実効ランクが 1〜3.5 に崩れ、dyn_1/copy_1 だけ下がる。次は V6（RMSProp 1e4 の seed 1・2、RMSProp・AdamW の τ=1e5）
