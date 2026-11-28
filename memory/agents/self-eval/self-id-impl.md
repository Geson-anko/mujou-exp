# 自分の表現空間での補完の同定の実装（2026-11-25）

計画は exp/vision/plans/jepa.md「V18c: 自分の表現空間での補完の同定」、事前登録は README の同名の節。solo-dev-cycle で進めた記録。

## 計画（段階 1）

- 公開 IF: `metrics.pair_wins(pred, real, *, truth, among, metric) -> (mean, A_n, ties_n)`、`regions.transplant(dst, src, pixels)`、`self_eval` の `stage_scores`・`expected_pair_wins`・`band_ratios`・`quarter_cells`・`local_sums`・`local_terms`・`step_rows`・`check_rows`・`evaluate_run`・`evaluate_checks`・`MirrorColorCode`、集計 `results/v18c/judge_self.py`
- テストのオラクル: Codex さんの中心化の反例（固定の中心で 1・1、別々の中心で 0・5/6）、位置だけの項（2 のべき）を足しても値がビットまで同じ、const は全部が引き分け、V-s6 の閉じた式とモンテカルロ、帯の重みつき平均 = 全体・本物で 0・参照の平均で 1、Codex さんの継ぎ目の符号で a_U = α・a_V = β・ρ_V = β²・η_U = (α−1)²、K3 で L = 0、位置だけで None、小さな JEPA で V-s0〜V-s3 が通る、ColorCode と鏡で V-s4・V-s5 が通る、ジョブの端から端まで

## 計画外の判断

- `pair_wins` に引き分けの数を 3 つ目の戻り値として足した（V-s2 の「全部が引き分け」）。`hard_c` 用に問い合わせごとの候補 `(n, c, F)` も受ける
- `top1` は A_n = 1 の割合（identification と同じ定義で、丸めだけ違う）
- `flat`・`pool` の平均を「最初の値 + ずれの平均」にした（V-s3 をビットで成り立たせる）。swap の 2 つの四半分は予測器 1 回から取る（GPU の再実行の食い違いを避ける）
- 補空間の k ≥ F は行を出さない（テストの小さな符号でだけ起こる）
- 出力は `<step>.json` と `<step>-tr.json`。per-query のリストは cos の四半分の平均の行だけ

## 自己レビュー

- swap を予測器 2 回で計算していたのを 1 回にした（V-s3 の swap が GPU で偶然外れうる）
- step_rows の swap 部分を `_swap_part` に切り出した
- M（F = 4096）は参照の段のテンソルが 1 段 約 200 MB になる。記述だけの腕なので許容した

## 数値を読む前に見つけた登録の問題（条件は変えていない。ユーザーの判断）

- V-s6: 24 個 × |t15| ≤ 3 で、実装が正しくても checkpoint あたり 0.196 が外れる（約 156 個で約 30 個）。Bonferroni なら閾値 4.51
- V-s5 の swap = 1: ‖r̃_A‖² > ⟨r̃_B, r̃_A⟩ を全問い合わせで要し、恒等式ではない
