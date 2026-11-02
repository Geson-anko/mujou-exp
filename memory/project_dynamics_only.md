---
name: project-dynamics-only
description: 2026-11-02、M13（画像のエンコーダ + 潜在空間の予測誤差）を中断し、玉の状態を直接観測にして動力学だけをオンラインで学ぶ M14 に移った
metadata:
  type: project
---

2026-11-02、ユーザーの判断で、画像のエンコーダと潜在空間の予測誤差をやめ、合成データ（玉の物理の状態）で動力学モデルの学習だけに絞った（exp/video/MILESTONES.md の M14）。M13 は未達のまま中断。

**Why:** M13 では表現の問題（エンコーダが当てにくい玉を捨てる、潜在の二乗誤差の重み）と、動力学をオンラインで学ぶ問題（時間の局所性、optimizer）が絡み、切り分けられなかった。動力学だけで通らなければ画像でも通らない。

**How to apply:** M14 ではエンコーダ・SIGReg・`z_norm`・再構成の項を使わない。判定は観測の空間で、平均・写し・等速の外挿と比べる。見た目の確認は予測した軌跡を真の軌跡と重ねた図で行う（[[feedback-metrics-need-qualitative-check]] の精神）。M13 で得た適格度トレースと optimizer の知見は [[project-m13-findings]]。方針の経緯は [[project-synthetic-first]]。
