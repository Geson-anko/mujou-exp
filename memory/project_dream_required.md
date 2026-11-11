---
name: project-dream-required
description: 2026-11-11 決定、夢（1 フレームと 16 フレームから 150 フレーム開ループ）は video の同時の学習の run の必須の評価
metadata:
  type: project
---

夢（動力学を観測なしで 150 フレーム = 5 秒先まで開ループで回し、後から学ぶデコーダで画像に戻したもの）は、`exp/video` の同時の学習（`program="joint"`）の run の必須の評価にする。posthoc（`measure.posthoc.Decoding`）が毎チェックポイントで必ず計算する（`measure.joint.dream`）。観測は 2 通り: 学習と同じ 16 フレーム（`dream_*`）と、最後の 1 フレームだけ・状態 0 から（`dream1_*`）。

**Why:** 2026-11-11、ユーザーが M16 の報告書に足した夢を見て「夢はこの実験において、必須の評価にしたい」と決めた。数字（`img_roll{h}`）だけでは見えない、長い開ループで玉が形を保って動き続けるかを毎回見るため。

**How to apply:**
- 同時の学習の run を報告するときは、`dream_lead`（平均の画像を下回る間のフレーム数）と `dream.jsonl` の曲線、`<t>-dream.png` の見た目を添える
- 報告書の動画は PC-ALM だけでよい（BP は数字と図だけ。ユーザー、2026-11-11）
- 夢を切るオプションは作らない

関連: [[feedback-metrics-need-qualitative-check]] [[project-decoder-for-debug]]
