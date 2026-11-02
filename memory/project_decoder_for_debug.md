---
name: project-decoder-for-debug
description: 再構成の評価のデコーダはデバッグの道具。各チェックポイントに後から学び、システムにも mujou の実行にも入れない（2026-11-02、ユーザー）
metadata:
  type: project
---

どの実験も見た目（画像に戻した再構成・予測）で確かめる。エンコーダがある実験では、各チェックポイントに軽量のデコーダを後から学び（`exp/video/decode.py` と同じ。エンコーダは止めたまま）、それで $z$ と、動力学が予測した $z$ を画像に戻す。

**Why:** デコーダはデバッグのためのもので、本来のシステムには取り込まない前提（2026-11-02、ユーザー）。mujou の中で同時に学ぶのは疑問とされた。

**How to apply:** M13 の再構成のノード（`nodes.Reconstruct`、`ModelConfig.recon`）を M15・M16 の学習に入れない。再構成の評価は run の外で、チェックポイントごとに行う。数値の条件は見た目と合うことを確かめてから使う（[[feedback-metrics-need-qualitative-check]]）。方針の全体は [[project-dynamics-only]]。
