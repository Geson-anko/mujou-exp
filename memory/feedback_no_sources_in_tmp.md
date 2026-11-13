---
name: feedback-no-sources-in-tmp
description: 解析・較正・図のスクリプトなどのソースを tmp（scratchpad、NAS の tmp）だけに置かない。リポジトリに入れる
metadata:
  type: feedback
---

ソースは tmp に置かない。tmp は消えるので、解析・較正・シミュレーション・図を作るスクリプトは必ずリポジトリ（`exp/video/diag/` など、テストつき）に保存する（ユーザー、2026-11-13）。

**Why:** M17 の λ の較正と S_P のシミュレーションのスクリプトが、セッションの scratchpad と `/nas/geson/mujou-exp/tmp/` にしかなかった。そのため、判定のもとになった導出をリポジトリから再現できなかった。

**How to apply:** 結果や判定に使うスクリプトは、書いた時点でリポジトリの中に置き、結果と同じ流れでコミットする。scratchpad に置いてよいのは、捨ててよい中間出力だけ。サブエージェントに解析を頼むときも、置き場所をリポジトリに指定する。関連: [[feedback-simple-fundamental]]
