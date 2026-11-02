# MEMORY

このプロジェクトのメモリ索引。1 メモリ 1 ファイル、ここには 1 行のリンクだけを置く。詳細は各ファイルへ。

エージェント固有の中間メモは [agents/](agents/) 配下（`<agent-name>/<task-slug>.md`）。

## project

- [プロジェクトの目的](project_purpose.md) — mujou を使って実験するリポジトリ。足りない機能は mujou に足して PR を送る
- [コードの 3 層](project_exp_lib.md) — mujou に足す / exp/lib / exp/<name>。実験横断の補助は exp/lib/
- [mujou submodule の運用](project_mujou_submodule.md) — editable で入れる。未 push のコミットを指すポインタをコミットしない
- [PC-ALM が中心](project_pc_alm_central.md) — 学習則は mujou.pc。BP は基準だけ。手法は局所的な形で組む
- [段 3 の知見](project_stage3_findings.md) — |z| 発散の機構と z_norm、rollout10 の物差し（定数で約 1.1、潜在の < 1 は動力学の証拠にならない）、EgoDex の実行の制約
- [合成動画を先に](project_synthetic_first.md) — 2026-11-02、EgoDex は一旦止め、簡単にした合成動画で正確な予測の土台を作る（M13）
- [M13 の知見](project_m13_findings.md) — w=1 のエンコーダは当てにくい玉を捨てる、再構成の項で保てる、オフラインの上限は 10 フレーム先 0.59 倍（未達）

## feedback

- [質問は最後にまとめる](feedback_questions_at_end.md) — 推奨案で進み、質問は区切りにまとめて出す
- [ドキュメントは日本語](feedback_docs_in_japanese.md) — README・規約・計画書・skill・memory は日本語。docstring とコメントは英語
- [指標は定性評価と照合](feedback_metrics_need_qualitative_check.md) — 予測の数値はデコードした画像と合うか確かめてから条件に使う
