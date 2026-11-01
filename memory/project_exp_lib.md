---
name: project-exp-lib
description: コードの 3 層（mujou / exp/lib / exp/<name>）の棲み分け。実験横断の補助と研究中の部品は exp/lib/ に置く
metadata:
  type: project
---

mujou の抽象の不足は mujou に足し、実験を回す補助と研究中の部品（記録・掃引・ホスト・設定・指標・optimizer の候補・SIGReg など）は `exp/lib/` に、その実験の設定・データ・モデルは `exp/<name>/` に置く。依存は `exp/<name>/` → `exp/lib/` → mujou の一方向。

**Why:** 「足りないものは mujou へ」だけでは、実験に閉じない補助コードの置き場がなく、実験パッケージが膨らむ。2026-11-01 のコード配置の再編で決めた。

**How to apply:** 迷ったら「mujou の利用者に説明できる機能か（→ mujou）、実験の運用の道具か（→ lib）」。詳細は `exp/README.md`。関連: [[project-purpose]]
