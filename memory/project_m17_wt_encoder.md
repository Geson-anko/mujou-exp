---
name: project-m17-wt-encoder
description: 2026-11-11 のユーザーの方針: M17 は Walking Tours で再構成の項なしの JEPA のエンコーダを詰める。評価は再構成の誤差と線形の動力学の予測誤差の両方
metadata:
  type: project
---

2026-11-11、ユーザーの方針で M17（exp/video/MILESTONES.md）を始める。目的は PC-ALM で学ぶ一番よいエンコーダ。

- 損失は JEPA（V-JEPA）+ SIGReg だけ。M15 で足した再構成の項は、玉の画像が単純すぎたための手当てとみる。本来の JEPA は再構成の誤差で学ばない
- PC-ALM で学ぶのはエンコーダだけ。デコーダと線形の動力学は評価のときに BP で学ぶ
- 評価は 2 つで、両方が小さいことを求める: 特徴量からの再構成の誤差（重要な特徴を捉えている）、特徴量の上の線形の動力学の予測誤差（線形によい特徴量）。片方だけ小さいのは disentangle の失敗か崩壊を疑う
- 特徴量の次元は入力より十分に小さい。画像は 128×128 以上。フレームは 30 fps、V-JEPA の入力は 0.1 秒の間隔で 16 枚を積み、tubelet は 2。評価には予測器を通した復元も載せる

**Why:** 実動画で、再構成に頼らない表現が、描き戻せることと線形に予測できることを両立するかを確かめるため。

**How to apply:** M17 の run で `vision.recon` を 0 以外にしない。片方の指標だけで設定を選ばない。デコーダの扱いは [[project-decoder-for-debug]]、データは [[project-video-data]]、見た目との照合は [[feedback-metrics-need-qualitative-check]]。
