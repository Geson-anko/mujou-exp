---
name: feedback-routine-to-sonnet
description: 2026-11-18、run の投入・監視・posthoc・集計などのルーティンワークは sonnet のサブエージェントか Workflow に任せてよい
metadata:
  type: feedback
---

ルーティンワークは、Workflow でもよいので sonnet に任せる（ユーザー、2026-11-18、exp/vision の計画のとき）。run は std-00〜03 に ssh で並列に組み、mini-00（5070 Ti）は試しの run とデバッグに使う。

**Why:** 4 台への並列の投入や posthoc・集計は手順が決まっていて、メインの context と時間を使うほどではないため。

**How to apply:** CLAUDE.md の「サブエージェントはユーザーが明示したときだけ」に対する明示の許可として扱う。対象は run の投入・監視、posthoc、表の集計など、手順が決まった作業。計画・判定の基準・結果の読み・設計の判断はメインで行う。サブエージェントや Workflow の agent には `model: "sonnet"` を指定する。関連: [[project-vision-restart]]
