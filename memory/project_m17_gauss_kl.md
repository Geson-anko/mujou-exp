---
name: project-m17-gauss-kl
description: 2026-11-13、ユーザーの決定で、EMA も負例も使わず、データポイントの正則化をガウスのモーメント合わせ KL(N(mu, Sigma) || N(0, I)) にする。N = 4096、λ は 10 倍刻み、optimizer は m17-opt-v1 から。回す前の決定は m17.md に登録済みで、λ の格子・opt の選び方・held-out の F・4090 の速さが未決
metadata:
  type: project
---

2026-11-13、ユーザーの決定: EMA のターゲットも対照の負例も使わない。データポイントの項を $F/D$、$F = \mathrm{KL}(N(\mu, \Sigma) \| N(0, I))$ にする（Fable と考え、主エージェントが確かめた案）。バッファは $N$ = 4096（決定）、$k$ = 1、seed 100。$\lambda$ は温度で 10 倍刻みに掃引し、optimizer と学習率は `m17-opt-v1` の結果から取る。

- 実装は `exp/lib/gauss_kl.py`、`vision.datapoint_law=gauss`、preset `m17/gkl_*`（$\lambda$ は `.nan` で必須。`vision.sigreg=<λ>` で指定する）。作業はブランチ `exp/2026-11-13/m17-gauss-kl`（worktree `mujou-exp-wt/gauss-kl`）
- 勾配の推定は、いまの標本をモーメントに入れず、$\hat P = (n - D - 2)S^{-1}$ を使う。不偏なのは行が i.i.d. なガウスで、エンコーダが止まっているときだけ。plug-in の $\hat\Sigma^{-1}$ は 1.34 倍、座標が動径を共有する重い裾（Student t、$\nu$ = 5）では 1.21 倍に偏る
- 始まりは $D + 5$ 行まで切る（$\hat P$ の平均と分散が有限になる最初の行数。確率近似の前提）。記録する `sigreg` は不偏な $\tilde F/D$。特異な散布行列は NaN になり、run は発散として終わる
- AR(1) の診断: 凍結した期待値の固定点 $c^*$（seed 8 本）は、本番と同じ $D/n$ = 0.25 で $\tau/(Nk)$ = 0.04 から 1% 広がり、$\tau \gtrsim Nk$ で 20% 縮む。「相関は期待値を動かさない」は $\hat P$ を通る部分では成り立たない。SIGReg よりは偏りが小さい
- 回す前の決定と判定の基準（L・U・C）は results/m17.md の「ガウスのモーメント整合（回す前に決めたこと）」。未決: $\lambda$ の格子、`m17-opt-v1` からの選び方（予測器の初期化と RMSProp を含むブランチを取り込む必要あり）、L の held-out の $F$（評価の 256 窓では測れない）、4090 での速さ

**Why:** EMA は 2 つのモデルを持ち、SIGReg は相関した標本で統計量が偏る。目的の式として書けて、1 次と 2 次のモーメントだけに頼る手法を選んだ（[[feedback-simple-fundamental]]）。

**How to apply:** 未決の 4 点をユーザーが決めてから回す。結果は L・U・C で判定する。背景は [[project-m17-wt-encoder]]、前の案は [[project-m17-stream-sigreg]]。
