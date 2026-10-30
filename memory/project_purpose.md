---
name: project-purpose
description: mujou-exp の目的。mujou ライブラリを使った実験を行い、足りない機能は mujou に即時追加して PR を送りながら進める
metadata:
  type: project
---

`mujou-exp`（2026-10-30 に作成）は、[mujou](https://github.com/Geson-anko/mujou) を使って実験するプロジェクト。mujou は推論と学習が同時に走り続ける継続学習ライブラリ（既定の更新手法は PC-ALM）。

**Why:** ライブラリの設計は実際に使わないと固まらない。実験で足りない機能が見つかったらその場で mujou に足し、mujou へ PR を送り、実験を止めない。

**How to apply:** 実験側にライブラリ機能の代用品を溜めず、`mujou/` に足す（手順は skill `mujou-upstream`）。実験の都合だけの暫定物は実験側に置く。判断が割れたらユーザーに確認する。関連: [[project-mujou-submodule]]
