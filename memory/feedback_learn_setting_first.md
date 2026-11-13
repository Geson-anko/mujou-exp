---
name: feedback-learn-setting-first
description: 本質的な手法がうまくいかないときは、手法を替える前に学習の仕方（optimizer と学習率などのコアなハイパーパラメータ）を指数オーダーで掃引して、損失が下がる条件を探す
metadata:
  type: feedback
---

本質的な手法（SIGReg など）がうまく働かないときは、まず学習の仕方が悪い可能性を疑う。ユーザー（2026-11-13）の指摘は次の 4 点。

- ストリームの学習では、momentum を持つ Adam 系がうまく動かないと言われている。RMSProp と純粋な SGD でも学習してみる
- ハイパーパラメータは指数オーダー（10 倍刻みなど）で変え、損失が下がる条件を探す
- 掃引は、学習率などコアなハイパーパラメータから始める
- 手法を探るときは、まずそれがうまくいく設定を考え、次に代替案を考える。考えるときは Fable に端的に相談してよい（コストがかかるので短く）

**Why:** M17 では、トークンの SIGReg、データポイントの SIGReg、EMA のどれも、Adam（学習率 1e-4/フレーム、warmup 300 秒）1 点のまま比べていた。手法の良し悪しと学習の設定の良し悪しを分けていなかった。

**How to apply:** 新しい手法を判定する前に、optimizer（SGD・RMSProp・Adam）× 学習率の 10 倍刻みの掃引で、held-out の損失が学習前より下がる設定があるかを見る。EMA は採らない。関連: [[feedback-simple-fundamental]]、[[project-m17-wt-encoder]]
