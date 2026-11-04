# M16 を実時間の 2 倍（B=1 で 60 fps）で回す計画（2026-11-04）

ユーザーの条件: [feedback_realtime_speed.md](../../feedback_realtime_speed.md)。B=1、4090、学習込みで 60 fps 以上。PC の時定数は変えてよい（M15・M16 の PC の確認は回し直す）。CUDA graph は使ってよい（mujou の 2026-10-29 の「使わない」を覆した）。

## 測定（snapshot d5a8fd5、B=1、4090、同期なしの bench）

| 設定 | BP | PC-ALM |
| -- | -- | -- |
| 16.7 更新/フレーム（周期 2 ms、既定） | 60.6 fps | 8.5 fps |
| 5 更新/フレーム（h=r=13.3 ms、λ=6.7 ms、relax 0.55 s） | 69.9 fps | 19.4 fps |

PC の 1 更新は CPU のカーネル起動律速（`run_backward` が 26%、送るごとの `Event.synchronize`）。視覚のステージの vjp は CUDA graph で 2.06 → 0.47 ms、1.76 → 0.40 ms（ビット単位で同じ）。

## 手順

1. mujou: `pc.Layer(cuda_graph=True)`（vjp を CUDA graph で記録・再生、既定は切り）。issue → PR
2. exp: `JEPAHead` と動力学のノードを graph にする（`exp/lib` の helper）。動力学のノードは、新しいフレームと窓のエピソードの無い update では何もしない
3. 時定数: 5 更新/フレーム（周期 frame/5）、窓の relax 0.55 s（落ち着くのに 80 周期 = 0.53 s、stride 0.6 s 未満）
4. B=1 で 60 fps を確かめる（`task=bench`、`data.streams=1`）。届かなければ mujou の送るごとの同期を外す（issue から）
5. M15 の PC の確認（新しい時定数）、M16 の確認（PC と BP）を回し直す。結果の表に B=1 の fps を書く
