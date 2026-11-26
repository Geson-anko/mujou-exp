---
name: feedback_cpu_trials_on_fleet
description: CPU を使う試し（合成の確かめ・掃引・ブートストラップ・並列のスクリプト）は Mac で回さず std-00〜03 と mini-00 で回す（2026-11-26）
metadata:
  type: feedback
---

ユーザーの言葉（2026-11-26、dyn/copy の確かめを Mac で 7 プロセス並列に回していたとき）: 「できれば mac ではなく、std-00〜03、mini-00 とかで CPU 使うような試しは回してほしい」。

**Why:** Mac はユーザーの手元の作業機で、重い計算で埋まると作業の邪魔になる。fleet の方が速く、コア数も多い（std は 16 コア、mini-00 は 8 コア）。

**How to apply:**
- 数分を超える CPU の計算、複数プロセスの並列は、スクリプトをリポジトリか NAS（/nas/geson/mujou-exp/runs/...）に置き、`ssh std-0X` からコンテナ `mujou-exp-dev-1` の中で走らせる（ホストに python は無い）。結果は NAS に書く
- 試験的な run・軽い試しは mini-00 も使ってよい（共用のホストなので、他のコンテナとシステムには触れない。CPU は 6 プロセスまで）
- Mac で回してよいのは、`just run`（format・テスト・型）、git、資料の生成、数秒〜1 分程度の確認だけ
- scratchpad に置いたスクリプトを Mac で回さない（[feedback_no_sources_in_tmp.md](feedback_no_sources_in_tmp.md) とも重なる）
