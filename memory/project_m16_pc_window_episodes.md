---
name: project-m16-pc-window-episodes
description: 決定（2026-11-04、ユーザー）。M16 の PC-ALM は 0.6 s ごとの窓のエピソードで学ぶ。ただし実時間の 2 倍程度の速さで回せることが条件
metadata:
  type: project
---

2026-11-04、ユーザーが了承した（計画 [m16-window-bptt.md](agents/main/m16-window-bptt.md) の §7-1）。

`dynamics.window` のある PC-ALM は、毎フレームの学習とトレースを使わない。`stride` フレーム（既定 18 フレーム = 0.6 s）ごとに直近 `window` フレームの窓のエピソードを開き、各ノードは窓の形のメッセージを隣と交わして平衡点まで緩和し、$t_w$ + `relax` の最初の update で 1 回学ぶ。毎フレームの予測（preq）は続ける。これを M16 の「オンラインの PC-ALM」とする。

**条件（ユーザー）**: 実時間で回せなくなるのはダメ。実時間の 2 倍くらいの速さが欲しい（[[feedback-realtime-speed]]）。

**Why:** 毎フレームの学習（1 フレームで止まる credit、トレース）では、オフラインの上限（窓の BPTT）に届かなかった（results/m16.md の C1）。窓のエピソードでも、局所性と非同期性は崩さない。学習は 0.6 s ごとに続き、フェーズもエポックも無い。

**How to apply:** 窓のある PC の run の結果は、毎フレーム学ぶ PC の run（M14、M16 の最初の確認）と同じ列に並べず、「窓のエピソード」と注記する。`win_skipped` と `win_unsettled` が 0 であることを確認の run で確かめる。速さは [[feedback-realtime-speed]] を満たすこと。
