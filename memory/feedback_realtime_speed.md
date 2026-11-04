---
name: feedback-realtime-speed
description: 実時間で回せることが目標。カメラ 1 台（B=1）の 30 fps を学習込みで 60 fps 以上を目指す。2026-11-04 は PC-ALM 38.7 fps で「一旦 OK」
metadata:
  type: feedback
---

実験のシステム（視覚と動力学を同時に学ぶ run など）は、現実の時間で回せなければならない。目安は実時間の 2 倍くらいの速さで、**測るのは B=1（カメラ 1 台）**: 30 fps のストリーム 1 本を、学習を含めて 60 fps 以上で回せること。B=128 は学習の効率のための並列で、この条件には入れない（ユーザーが選んだ）。PC-ALM の時定数（1 フレームあたりの更新回数）は、速さのために変えてよい（M15・M16 の PC の確認は回し直す）。

**Why:** mujou は推論と学習が同時に走り続ける継続学習のライブラリで、仮想時間（`ManualRun`）でしか回らない手法は目的に合わない。2026-11-04、M16 の窓のエピソード（PC-ALM）を了承したときに、条件として示された。そのときの測定は 4090、B=128 で、PC-ALM の同時の run が 4.3 fps、BP が 30 fps、M15 の視覚だけの PC-ALM が 3 fps。

**2026-11-04 の到達点（ユーザーが「あくまでリアルタイムは目標なので、一旦これで OK」）**: B=1・4090 で BP 69 fps、PC-ALM 38.7 fps（5 更新/フレーム、視覚の `pc.Layer`・`JEPAHead`・動力学の窓の関数を CUDA graph）。60 fps へ残る手は、動力学のノードをまとめる（落ち着くのに要る更新が 80 → 39 回、4 更新/フレームにできる）と、mujou の送るたびの同期（S2）を外すこと。計画は [m16-realtime-plan.md](agents/main/m16-realtime-plan.md)。

**How to apply:** 手法を選ぶ・確認するときに、4090 で B=1（`data.streams=1`）の fps を測って記録する（`task=bench`）。60 fps に届かない設定は、届くまで速くする（プロファイルしてから、mujou 側の改善なら mujou に PR）。結果の表には fps を書く。[[project-m16-pc-window-episodes]]
