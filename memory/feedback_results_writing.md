---
name: feedback-results-writing
description: 結果のまとめは体系的に順を追って説明し、日本語は sonnet/haiku に /yomiyasu で推敲させ、文科省の HP の文体を手本にする
metadata:
  type: feedback
---

結果をまとめるときは、体系的に、丁寧に、1 つずつ順を追って説明する。Opus の日本語は直訳調でくどい言い回しがあり読みにくいので、まとめの文章は sonnet か haiku のサブエージェントに skill `yomiyasu` で推敲させ、文部科学省のホームページの文章（平易で、主語と述語が近く、1 文 1 事項）を参考にする（ユーザー、2026-11-18）。

2026-11-20 に重ねて指摘された。直訳的な単語と凝った比喩表現を避け、なるべく平易な文にする。レポートは図と表を使い、体系的に一歩ずつ書き、そのレポートだけで実験の全体が分かるようにする（手本は exp/vision/report.html）。

**Why:** ユーザーが Opus の日本語を「直訳の日本語、くどい言い回し」で読みにくいと指摘した。

**How to apply:** README の結果の節、報告、レポートを書いたら、推敲をサブエージェント（sonnet か haiku）に渡して /yomiyasu をかけてから出す。チャットの報告も同じ方針で、結論 → 根拠 → 次の順に短い文で書く。関連: [[feedback-docs-in-japanese]]、[[feedback-routine-to-sonnet]]
