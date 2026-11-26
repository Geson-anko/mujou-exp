# 線形の幅を fleet で並列に回す計画（2026-11-26）

speedup-planner（サブエージェント）の調査。数値をビットまで変えない分割だけを扱う。

## 空き（14:51）

std-00〜03 はどれも Ryzen 9 7950X（16 物理コア）、RAM の空き 58 GB、4090 と 4060 は使用率 0%。コンテナは `mujou-exp-dev-1`。mini-00 は使わない。

## checks が遅い理由

`configure_determinism`（exp/lib/provenance.py）が 1 スレッドに固定し、`check_rows` が V-h0a〜m を 1 プロセスで逐次に回す。時間の約 85% は V-h0i と V-h0k の `_common_sums` 約 1800 回（1 回 2.09 秒）。ブートストラップ・BLAS・OU のループは効かない。1 回目の checks は 68 分（14:51 に rc=0、/nas/geson/mujou-exp/runs/v18c-width/checks/rows.json）。

## 並列化（1 プロセス 1 スレッドを守り、プロセスの数で並列にする）

刻みをまとめて大きなテンソル演算にするのはやめる（総和の形が変わり丸めが変わる）。

| 段 | 分け方 | コードの変更 | 見積もり |
|--|--|--|--|
| width_eval `checks` | 確かめごとにプロセスを分け、V-h0i と V-h0k は seed の範囲で分けて結合 | `--only`、`--seeds lo:hi`、結合。V-h0k の真値の `g_t` は r をまたいで 1 本の乱数列なので、前の r の 100 回分の `_band_noise` を捨てて状態を進める | 68 分 → 約 5 分 |
| width_eval `yard` | 符号ごとに 1 プロセス（9 個） | `evaluate_yard` と `main` に `--codes`、`emit` を絞る、`vh.json` は pca のときだけ | 35 分 → 約 5 分 |
| width_eval `run` / `vae` | run × step（6000 / 60000）ごとに約 46 本 | 不要（出力のある step を飛ばす） | 約 2 時間 → 5〜10 分 |
| width_guard `yard`（V-h3） | `--fit pca:0` / `pca:1` / `pca:2` / `v17:0` を 4 台に、`--dec`（uncond）は 1 本だけ | 不要 | 2.5 時間 → 約 40 分 |
| width_guard `cheat`（V-h2） | ずる 9 個（base、mean-8/32/64/128、drop-16/32/48/56）を GPU に | `evaluate_cheat` と `main` に `--only` | 5.3 時間 → 1〜1.7 時間 |
| width_guard `run` | run ごとに 1 プロセス | 不要 | 11 時間 → 約 1.5 時間（8 枠） |

- V-h2 は V-h3 と同時に先回りで回してよい（V-h3 が外れたら捨てる）。
- GPU に 2 本目を載せるのは、1 本目の使用率が 60% 未満のときだけ（std-03 で同居による illegal memory access の前例）。4060 は使わない。
- CPU のジョブは 1 台 14 プロセスまで。実データのジョブは 1 本の RSS を見てから本数を決める。
- `runs/v18c-tlin/scripts/apply.sh` の逐次の `while read` は `xargs -P` 等の並列に変える。
- 一致の確かめ: 同じ単位（例: yard の pca 1 符号）を逐次版と分割版で出し、`rows/<code>.json` がバイト単位で同じかを比べる。

## 段 A の割り当て

| host | 4090 の 1 本目 | 4090 の 2 本目 | CPU（yard `--codes`） |
|--|--|--|--|
| std-00 | guard yard `pca:0` + `--dec` | cheat base, mean-8 | pca（+vh.json）, color |
| std-01 | guard yard `pca:1` | cheat mean-32, mean-64 | mean8, init |
| std-02 | guard yard `pca:2` | cheat mean-128, drop-16 | init_f, init_m |
| std-03 | guard yard `v17:0` → cheat drop-56 | cheat drop-32, drop-48 | init_c, init_c1, init_v16 |

計測スクリプト（リポジトリ外）: /private/tmp/claude-501/-Users-geson-Projects-mujou-exp/644760cb-6885-4f8a-ab99-df9be1fa9bb1/scratchpad/bench.py、bench2.py
