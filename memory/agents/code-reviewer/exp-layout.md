# exp-layout（experiments/continuous_stream → exp/video + exp/lib）レビュー

対象: `git diff main...HEAD`（610ff40, 0738990, 1a358c3, 41b0ef8）と `.claude/` の未コミット変更。

## verdict: approve

must-fix はない。振る舞いは変わっていない（下の検証結果）。should-fix は文書の整合と、`RUN_MODULE` の検証漏れ。

## must-fix

なし。

## should-fix

1. `.claude/agents/implementation-planner.md:33`（未コミット）
   - 問題: 「複数の実験が使い、特定の実験に閉じない機能は mujou に足す」が残っている。新しい 3 層の方針（その種の補助コードは `exp/lib/`）と矛盾する。計画書が `.claude/` の更新箇所に挙げたのは 31 行目だけで、33 行目が漏れていた
   - 根拠: CLAUDE.md「プロジェクト概要」の 3 層、`exp/README.md`、`refactor-conventions` の「実験のパッケージ構成」
   - 確信度: 高

2. `tests/video/test_run.py:50-66`（`TestRun.test_help`）と `exp/video/stage0.py:30`（`RUN_MODULE`）
   - 問題: `--help` のテストはモジュール名を文字列 `"exp.video.run"` で直書きしていて、`RUN_MODULE` を参照していない。`RUN_MODULE` を書き間違えても、どのテストも落ちない。`run_all` は失敗した run を `FAILED` と表示するだけで、終了コードは 0 のまま
   - 根拠: 計画書「想定リスク」の「`RUN_MODULE` を 1 か所に置き、`test_run.py` の `--help` テストでモジュール名を検証する」。現状のテストではこの対策になっていない。`exp/lib/sweep.py` にテストがないこと（計画の例外）と重なり、文字列で渡すパスを守るテストが一つもない。今回は手動で確認した（検証結果を参照）
   - 確信度: 高（テストが `RUN_MODULE` を見ていないことは事実。should-fix にしたのは、現状の値が正しいため）

3. `exp/video/MILESTONES.md:3`（冒頭の注記）
   - 問題: 移した先の対応表が不完全。`optim.py` も `exp/lib/` に移した（M1 の「作るもの（`optim.py`）」、52 行目）のに、注記にない。SIGReg・MinGRU（`nodes.py` から）、`effective_rank`・`probe_r2`（`evaluate.py` から）、`_from_dict`（`config.py` から）の移動先も書かれていない
   - 根拠: 計画書「文書の書き換え範囲」の MILESTONES の項。読み手が当時のファイル名から今の場所を引けない
   - 確信度: 高

4. `exp/video/MILESTONES.md:14`
   - 問題: 「テストは `tests/video/` に置き、`exp/video/` のモジュールと 1 対 1 にする（`test_optim.py` など）」。パスは新しくしたのに、例の `test_optim.py` は今は `tests/lib/` にある。冒頭の注記は「以下のファイル名は当時のもの」と言っているので、パスだけ新しくしたこの行は当時の記録としても現状の説明としても合わない（32-35 行目も同じく新旧が混ざっている）
   - 根拠: 現在の配置（`tests/lib/test_optim.py`）
   - 確信度: 中

5. `exp/video/README.md:11`
   - 問題: レイアウト変更の記録が「決定事項（2026-11-01）」の箇条の途中（`stop_target` と SIGReg の重みの間）に入っている。この節の前置きは「段 3 はこの設定から始める」で、箇条は段 3 の設定を並べたもの。コードの配置の変更はこの並びに属さないので、段 3 の設定の一つのように読める
   - 根拠: README.md:5-14 の節の構成
   - 確信度: 中

6. `tests/video/test_model.py::TestRunVideo::test_evaluating_does_not_change_the_learning` の非決定性（実装メモに記録あり）
   - 問題: `just run` の全体実行でまれに落ちる（実装中に 2 回）。今回のレビューで回した `just run` では通った。純粋な移動だけのコミット 1 の後でも起きているので、このブランチが原因ではないと思われる。ただし、決定性を前提にしたテストが flaky なまま main に入る
   - 根拠: `memory/agents/plan-implementer/exp-layout.md`
   - 確信度: 低（再現していない）。このブランチの範囲外で、別タスクとして追う

## nit

1. `exp/video/nodes.py:34` — 節見出し `# --- minGRU and Dynamics ---` が残っている。MinGRU は `exp/lib/mingru.py` に移ったので、見出しが中身と合わない（確信度: 高）
2. `exp/video/PLAN.md:340-344` — 構成図の `exp/lib/` には `optim.py` しか書かれていない。`nodes.py`・`evaluate.py` の行では `sigreg.py`・`mingru.py`・`metrics.py` を参照しているので、これらも並べるとそろう。`tests/lib/` も図にない（確信度: 中）
3. `exp/lib/optim.py:1`、`exp/lib/sigreg.py:1,60` — lib の docstring が `exp/video/PLAN.md` を指している。import の依存ではないので完了条件 4 には反しないし、計画どおりでもある。ただ、lib の説明が特定の実験の文書に依存する形になる。optim.py:1 は 88 文字を超える（E501 は ignore 済み）（確信度: 中）
4. `exp/lib/mingru.py` — 24 行の `nn.Module` 1 つだけのモジュール。今は `exp/video/` からしか使われていない。計画の確認事項 1（研究中の部品を lib に入れるか）がユーザー判断待ちなら、それに従う。シンプルさの観点からは、`exp/video/nodes.py` に残しても十分だった（確信度: 低。好みの範囲に近い）
5. `.claude/skills/refactor-conventions/SKILL.md:27` の「mujou と torch だけに依存し」と、`exp/README.md:8`・CLAUDE.md:22 の「mujou の公開 API・torch・標準ライブラリだけに依存する」とで表現がずれている。同じく「複数の実験が使う研究中の部品」と「2 つ以上の実験が使う、または特定の実験に閉じない」もずれている（確信度: 中）
6. `.claude/skills/agent-team-startup/SKILL.md:30` — `experiments/` を `exp/` に縮めたことで、表の列の桁がずれた（`.claude/` は mdformat の対象外なので自動では直らない）（確信度: 高）
7. `exp/video/README.md:202` — 結果の節に `test_video.py`（現 `test_balls.py`）が残っている。過去の記録なので残す方針でよいが、README には MILESTONES のような「ファイル名は当時のもの」という注記がない（確信度: 低）
8. コミット 41b0ef8 の件名 `docs: follow the move to exp/` にスコープがない（CLAUDE.md の規約は `<種別>(<スコープ>): <内容>`）（確信度: 高。計画書の指定どおりではある）
9. `tests/lib/test_record.py` は `git mv` ではなく新規作成なので、`git log --follow` で旧 `test_record.py` の履歴をたどれない（計画では `git mv`）（確信度: 高）
10. CLAUDE.md:20 — 「1 実験 1 パッケージ」という記述が CLAUDE.md から消え、`exp/README.md` にだけ残った（確信度: 低）

## 検証結果

- just run: pass（130 passed、pyright 0 errors。format による差分なし）
- テストの数: `def test_` の一覧は main の `tests/continuous_stream/` と HEAD の `tests/lib/`・`tests/video/` とで完全に一致（78 関数）。アサーションの変更は `TestRecorder` の `Config.from_dict(head["config"]) == config` → `head["config"] == config.to_dict()` の 1 か所だけ。`Config` の round trip は `tests/video/test_config.py` が引き続き検証している
- 振る舞いの一致: main（4c5909a）で採った `ref-lin`・`ref-vid`（kappa=0.05 なので SIGReg の経路も通る）の `metrics.jsonl` と、**現 HEAD（41b0ef8）**で回し直した `uv run python -m exp.video.run` の出力は、どちらも `cmp` でバイト単位まで一致した。実装者の after は、コミット 3 の作業途中（0738990 + dirty）で採ったものだったので、最終コミットで取り直した。BP（`bp=True`）と SGD 以外の optimizer はこの比較に入っていない。どちらもコードの差分は import と名前の変更だけ
- 文字列のモジュール名: `run_all(RUN_MODULE, [1 設定], out, 1)` を実際に回すと `done sw-lin` になり、2 回目は `0 runs to go`（`done` の判定）。`exp.video.{run,stage0,stage1,stage2,collapse} --help` はすべて解決した
- 依存の向き: `exp/lib/` と `tests/lib/` からの `exp.video` の import は 0 件（docstring の `exp/video/PLAN.md` だけ）
- 古いパスの残り: `experiments|continuous_stream` の grep で残るのは、計画書の「意図的に残すもの」と `exp/lib/__init__.py` の docstring の "experiments"（一般名詞）だけ
- 中間コミット: 610ff40 では `exp.video.video` を、0738990 以降では `balls` を import していて、どちらも整合している。610ff40〜0738990 の間は CLAUDE.md のリンク `experiments/README.md` が切れているが、41b0ef8 で直る
- 成果物汚染: `</content>` の混入なし（orchestrator.md の該当は事故を説明する本文）
