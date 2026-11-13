---
name: project-ema-following
description: M17 の EMA のターゲットエンコーダ（jepa/ema.py の Following）は時間で追い、置き場所は今は exp/video/jepa。mujou に上げるかは未決定
metadata:
  type: project
---

M17 の I-JEPA のターゲットエンコーダ（`exp/video/jepa/ema.py` の `Following`）について、2026-11-13 に決めたこと。

- **時間で追う**: `vision.ema` = $m$ はフレームあたりの momentum。`elapsed` 秒のステップでは $m^{\mathrm{elapsed}/\mathrm{frame}}$ で動かす。$\tau\dot{\bar\theta} = \theta - \bar\theta$ を、ステップの間 $\theta$ を一定として厳密に解いた形で、時定数は $-\mathrm{frame}/\ln m$ 秒。mujou の optimizer と同じく、学ぶ回数によらない（locality-and-async の「更新量をステップ数で決めない」）。1 フレームのステップでは I-JEPA の式そのものになる。ただし実際の `elapsed` はフレームちょうどではない（テストの時計では 26 ms・34 ms で、平均がフレーム）。そのため、ステップごとの値はステップで数えた版とビット単位では一致しない
- **置き場所は今は `exp/video/jepa`**: `Following` は mujou の `pc.optim.Optimizer` を包む、学習則の基盤に近い部品である。CLAUDE.md の判定では mujou の候補になる。今は M17 の 1 つの実験でしか使わず、EMA が M17 で役に立つかもまだ分からないので、実験の中に置く

**Why:** レビュー（2026-11-13）で、ステップで数える EMA は `time.learn_interval` を変えると時定数が変わり、非同期の実行でノードごとに速さがずれると指摘された。置き場所も、mujou に置くべき候補と指摘された。

**How to apply:** EMA を M17 の外でも使う、または M17 の本線に残すと決まったら、mujou の `pc.optim` に時間で追う EMA の optimizer として上げる（skill `mujou-upstream`、オラクルは閉じた形の指数の減衰）。それまでは `exp/video/jepa/ema.py` から動かさない。関連: [[project-m17-wt-encoder]]
