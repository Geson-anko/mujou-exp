---
name: project-video-data
description: 実動画は Walking Tours（CC BY 4.0）に決めた。研究は非商用なので CC BY-NC も可（2026-11-02、ユーザー）
metadata:
  type: project
---

M15・M16 の実動画は Walking Tours（DoRA、ICLR 2024、https://huggingface.co/datasets/shawshankvkt/Walking_Tours）。CC BY 4.0、一人称の街歩き 10 本 + 野生動物 1 本、1〜3 時間、4K・60 fps。動画は YouTube から取る（`WTour.txt`）。この研究は非商用（ユーザー）なので CC BY-NC のデータ（EPIC-KITCHENS-100 など）も使える。

**Why:** EgoDex はレポートにフレームを直接載せられない。Walking Tours は出典を書けば載せられ、1 本の長い連続したストリームとして流せる。Ego4D はフレームを載せてよいかが不明なので外した。

**How to apply:** 取るときに動画ごとの YouTube のライセンスと、顔・ナンバーのぼかしの有無を確かめる。レポートには出典（論文と動画）を書く。物を扱う動きが要るなら EPIC-KITCHENS-100 を候補にする。全体の方針は [[project-dynamics-only]]。
