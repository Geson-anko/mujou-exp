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
- [動力学と視覚を分ける](project_dynamics_only.md) — 2026-11-02、M13 を中断。M14 動力学だけ → M15 JEPA → M16 同時の学習（動力学はエンコーダに credit を送らない）
- [デコーダはデバッグ用](project_decoder_for_debug.md) — 再構成の評価のデコーダは各チェックポイントに後から学ぶ。システムにも mujou の実行にも入れない
- [実動画のデータ](project_video_data.md) — Walking Tours（CC BY 4.0）に決めた。研究は非商用なので CC BY-NC も可
- [M16 の PC-ALM は窓のエピソード](project_m16_pc_window_episodes.md) — 2026-11-04 決定、0.6 s ごとの窓で学ぶ PC-ALM を M16 のオンラインの PC-ALM とする。速さは学習が現実的な時間で終わること
- [夢は必須の評価](project_dream_required.md) — 2026-11-11 決定、同時の学習の run は posthoc が毎チェックポイントで 1 フレームと 16 フレームから 150 フレームの夢を計算する。報告の動画は PC-ALM だけでよい
- [M17 は Walking Tours のエンコーダ](project_m17_wt_encoder.md) — 2026-11-11、再構成の項なしの JEPA。課題は時空間の補完、主な指標は held-out の補完の誤差と線形の動力学（画素の再構成は重視しすぎない）。PC-ALM で学ぶのはエンコーダだけ

## feedback

- [質問は最後にまとめる](feedback_questions_at_end.md) — 推奨案で進み、質問は区切りにまとめて出す
- [ドキュメントは日本語](feedback_docs_in_japanese.md) — README・規約・計画書・skill・memory は日本語。docstring とコメントは英語
- [指標は定性評価と照合](feedback_metrics_need_qualitative_check.md) — 予測の数値はデコードした画像と合うか確かめてから条件に使う
- [実時間はやめ、学習が終わる速さを守る](feedback_realtime_speed.md) — 2026-11-05、実時間の最適化は諦めた（5 更新/フレームで PC-ALM が不安定）。確認の run が 4090 で数時間で終わること
- [手法はシンプルで基礎的に](feedback_simple_fundamental.md) — 2026-11-12、理論の裏付けのないヒューリスティック（後付けの正規化・罰・拡張）は入れない。自明な解を目的の式から導出で確かめる
- [ソースを tmp に置かない](feedback_no_sources_in_tmp.md) — 解析・較正のスクリプトは必ずリポジトリへ（tmp は消える）
