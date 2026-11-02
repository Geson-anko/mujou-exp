# exp-layout 実装ログ

- コミット 1〜4 を実装（`.claude/` は未着手、orchestrator 担当）。IF 変更通知: `run_all(module, configs, out, jobs)`、`Recorder(config: RunConfig)`、`optim.from_config` は `exp.video.config.make_optimizer`、`nodes.MinGRU/SIGReg` は `exp.lib.mingru/sigreg`。
- 計画外の判断:
  - `tests/lib/test_sigreg.py` に `_fd`・`_inputs`・`_sample` を複製（lib のテストが `exp.video` を import しないため）。
  - `exp/video/run.py` の「continuous-stream experiment」を「video experiment」に直した（計画が望ましいとした項目）。
  - `exp/video/nodes.py` の docstring の SIGReg への言及を `exp.lib.sigreg` に直した。
  - 文書の書き換えは `exp/video/MILESTONES.md` の `ブランチ名の例` を残した（計画どおり）。
- `tests/video/test_model.py::test_evaluating_does_not_change_the_learning` が、`just run` の全体実行で 1 回だけ失敗した（コミット 1 後と 3 の途中の計 2 回。単独実行や再実行では通る）。移動と無関係の非決定性（並列実行・free-threaded）の疑い。要調査（別タスク）。
