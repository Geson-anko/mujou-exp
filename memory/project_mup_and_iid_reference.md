---
name: project-mup-and-iid-reference
description: 2026-11-20、ユーザーの決定。exp/vision では今後も μP を使う。V16 では、ストリームで最もよく学べたモデルを、ランダムな stack と BP で学ぶ普通の VAE として学び、上限の目安にする
metadata:
  type: project
---

2026-11-20、V11〜V15（optimizer の研究）の報告を受けて、ユーザーは次の 2 つを決めた。

- μP は今後も使う。新しい run の出発点は、V14 で muTransfer が成り立った μP（`model.param=mup`、`model.scale_decoder=true`）にする
- V16: ストリームで最もよく学べた設定（V14 の幅 1024・μP・RMSProp $10^5$、`b2006f0a-s0`）のモデルを、ストリームではなく、ランダムに選んだ stack のミニバッチで普通の VAE として学ぶ（`exp/vision/iid.py`、BP と AdamW）。再構成誤差・`recon_blur`・描き戻した画像の綺麗さを見る

**Why:** ストリームの run の span（0.587）が、モデルの限界なのか、ストリームと PC-ALM で学ぶことの限界なのかを分けるため。

**How to apply:** V16 は BP の run だが、比べるための上限の目安で、[[project-vision-restart]] の「BP の run は回さない」の例外としてユーザーが頼んだもの。システムの学習則は PC-ALM のまま（[[project-pc-alm-central]]）。μP の学習率の倍率は `exp.vision.models.mup_lr` を使う。関連: [[project-optim-study]]
