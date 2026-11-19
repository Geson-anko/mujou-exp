---
name: project-optim-study
description: 2026-11-19、exp/vision の V11〜V14 で、ストリームで学べる最適な optimizer を探す。ユーザーが決めた範囲（t を含む式は外す、Adam 系は AdamW、OGD の間隔は秒、μP は幅だけ、mujou への PR は性能が出たものだけ）
metadata:
  type: project
---

2026-11-19、ユーザーは exp/vision の VAE（PC-ALM、Walking Tours のストリーム）で、optimizer を掘り下げると決めた。目的は「ストリームの設定で学べる最適な optimizer を、mujou のフレームワークに合う形で見つける」。結果を踏まえて Fable や Codex にも相談しながら、多くの実験を回す。計画は exp/vision/plans/optim-study.md、腕と予想は exp/vision/README.md の V11〜V14。

ユーザーの決定:

- 評価は V8 と同じ `stream_eval`（ストリームから直接）。ViT だけで、ResNet は使わない
- 更新式にステップ数 t を含む optimizer（Adagrad・RAdam・NAdam・ASGD・torch の形の Adafactor）は扱わない。Adam 系は AdamW で足りる（Adam・Adamax も外す）
- OGD（直近 n 個の勾配と直交する向きに更新）の保持の間隔 s は秒で決める（リアルタイムなので）。RMSProp・AdamW との組み合わせも試す
- 学習率と幅: μP（Tensor Programs V）は幅だけで試す
- mujou への PR は、性能が出た optimizer だけにする。それまでは exp/lib に置く
- 速さ: リアルタイムを目指すが、run は run_manually の仮想タイマーで進めるので、遅すぎなければよい

**Why:** V8〜V10 で、SGD ではデコーダが z を使わず、要素ごとに一歩を揃える optimizer（RMSProp・AdamW）でだけ学べると分かった。そこで、何が効いているのかを optimizer の側から詰める。

**How to apply:** 新しい optimizer の腕は、時間の形（lr = elapsed/τ、β = e^{−elapsed/τ_x}）でノードの中に閉じて書き、exp/lib に置く。順位は V8 の 4 条件を満たしたうえでの 1800 秒の span で決め、1 seed の差が 0.07 より小さいものは区別しない。関連: [[project-vision-restart]]、[[feedback-learn-setting-first]]
