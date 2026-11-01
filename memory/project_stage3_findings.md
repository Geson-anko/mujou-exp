---
name: project-stage3-findings
description: exp/video の M11・M12（2026-11-01）で分かった、SIGReg の |z| 発散・rollout10 の物差し・EgoDex の実行の制約
metadata:
  type: project
---

- SIGReg の |z| 発散は、似たフレームが続くとバッファに溜まった自分のコピーから斥力を受け、裾で引力が消えて止まらない機構。`z_norm`（z を半径 √D の球面へ）で止まる。EgoDex では `z_norm` なしだと 2 秒で発散する
- `rollout10` は「クリップ間の同時刻の分散」で割るので、定数（全体の平均 z）でも 1 にならない（約 1.1）。評価の `mean{n}` と並べて読む。「1 未満」を完了の条件にすると誤る
- 開ループの学習（`open_loop`）は rollout10 を下げるが、定数の水準は超えない。固定エンコーダでも同じなので、律速は動力学の容量ではなく、z の予測しやすさ
- EgoDex は隣のフレームがほぼ同じ。SIGReg のバッファは数分に広げる（`buffer_every`）。予測は残差（`residual`）が自然
- EgoDex の 3+3 ノードは実時間（mujou.run）では追いつかない（CUDA の起動が 1 GPU で直列）。手動の仮想時間で、4090 でも実時間の約 6 倍（CPU 1 コア律速）。1 本 30 分で 3〜3.5 時間。GPU には複数本を並べてよい（4060 は 8 GB なので評価を分割して計算する）

**Why:** 段 3 以降の判断（何を完了の条件にするか、どこで回すか）に毎回効く。
**How to apply:** rollout の判定は `mean{n}`・`persist{n}` と比べて書く。EgoDex の実験時間は 1 本 3 時間超で見積もる。関連: [[project-pc-alm-central]]
