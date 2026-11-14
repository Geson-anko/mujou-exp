---
name: project-vae-stream-next
description: 2026-11-14 のユーザーの方針: M17 の JEPA は λ の実験で締めくくり、次は VAE でストリームから学ぶ計画を立てる
metadata:
  type: project
---

2026-11-14、層ごとの学習率の結果（どの設定でも最後は広がりが潰れて held-out の J が上がる）を受けて、ユーザーは「JEPA ではなく、一旦 VAE といったモデルで取り組む必要があるかもしれない」と判断した。

- M17 の JEPA は λ（目的の式の釣り合い、η = 1e-5 の層ごとの学習率に固定）の実験で締めくくる
- 評価の確認（学習前の ViT の recon_blur が平均の色だけで説明できるか）も締めくくりに含める
- 次のマイルストーンは、Walking Tours のストリームから VAE を学ぶ。計画はユーザーと詰める

**Why:** 再構成なしの JEPA では、optimizer・学習率・層ごとの学習率のどれでも崩壊を止められなかった。再構成の項がある VAE なら、目的の式そのもの（ELBO）が自明な解を罰する。

**How to apply:** VAE の run は M17 の「vision.recon を 0 以外にしない」（[[project-m17-wt-encoder]]）の対象外で、新しいマイルストーンとして扱う。β のアニーリングなど理論の裏付けのない工夫は入れない（[[feedback-simple-fundamental]]）。学習の設定を先に探す（[[feedback-learn-setting-first]]）。
