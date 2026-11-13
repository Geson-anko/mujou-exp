---
name: project-m17-gauss-kl
description: 2026-11-13、ユーザーの決定で、EMA も負例も使わず、データポイントの正則化をガウスのモーメント合わせ KL(N(mu, Sigma) || N(0, I)) にする。N = 4096、λ は 10 倍刻みで掃引してから回す
metadata:
  type: project
---

2026-11-13、ユーザーの決定: EMA のターゲットも対照の負例も使わない。データポイントの項を $F/D$、$F = \mathrm{KL}(N(\mu, \Sigma) \| N(0, I))$ にする（Fable と考え、主エージェントが確かめた案）。バッファは $N$ = 4096（決定）。

- 実装は `exp/lib/gauss_kl.py`、`vision.datapoint_law=gauss`、preset `m17/gkl_*`。作業はブランチ `exp/2026-11-13/m17-gauss-kl`（worktree）
- 勾配の推定は、いまの標本をモーメントに入れず、$\hat P = (n - D - 2)S^{-1}$ を使う。plug-in の $\hat\Sigma^{-1}$ は $D$ = 1024、$n$ = 4096 で 1.34 倍に偏る
- 「相関はばらつきだけを増やす」が成り立つのは、$\mathrm{tr}$ と平均の部分だけ。対数行列式の部分は、相関時間がバッファの平均の年齢 $Nk$ に近づくと期待値も動く（AR(1) の診断で $\tau = 10^4$ に 16%）
- $\lambda$ の preset の 0.5 は仮の値。始まりは $D + 3$ 行まで切る（推定が存在する最初の行数）

**Why:** EMA は 2 つのモデルを持ち、SIGReg は相関した標本で統計量が偏る。目的の式として書けて、1 次と 2 次のモーメントだけに頼る手法を選んだ（[[feedback-simple-fundamental]]）。

**How to apply:** 回す前に $\lambda$ を 10 倍刻みで掃引する格子を決め、optimizer と学習率の掃引と合わせる。背景は [[project-m17-wt-encoder]]、前の案は [[project-m17-stream-sigreg]]。
