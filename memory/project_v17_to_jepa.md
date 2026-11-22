---
name: project-v17-to-jepa
description: 2026-11-22、ユーザーの決定。V17（単一ストリームを 480 万 stack）で VAE の検証は締めくくり、次は検証する対象を VAE から JEPA に移す。長い単一ストリームは c = 10 の chunk で回す
metadata:
  type: project
---

2026-11-22、V17 の結果（単一ストリームの PC-ALM の VAE は、480 万 stack で span 0.211、場面を見分けられる）を受けて、ユーザーは次を決めた。

- V17 の本番は 600 万 stack まで待たず、480 万 stack で止める。V16（普通の学習）とは span の場面の範囲が違い、直接は比べられないため
- 結果を `exp/vision/report-long.html` にまとめ、main にマージする
- 次は、検証する対象を VAE から JEPA に移す

**Why:** VAE については「ストリームの PC-ALM でも、長く見せれば場面を描けるまで学べる」ことが分かり、問いに答えが出たため。

**How to apply:** 長い単一ストリームを安く回すときは `data.layout=chunk data.streams=10`、optimizer の時定数は B = 1 の 10 分の 1（V17 の追加の調査で、1 周まで B = 1 と差 0.01 以内）。c = 100 は B = 1 の代わりにしない（1 周目で 0.06〜0.12 遅れ、τ を下げても縮まない）。B = 1 の seed の揺れは span で約 0.025 なので、閾値 0.07 は控えめ。JEPA の過去の知見は [[project-m17-wt-encoder]]、[[project-decoder-for-debug]]。関連: [[project-mup-and-iid-reference]]
