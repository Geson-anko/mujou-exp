# experiments/

1 実験 1 ディレクトリで、Python パッケージとして置く: `experiments/<name>/` に `__init__.py`、コード、`README.md` を入れる。`README.md` には仮説、実行方法、結果を書く。実行は `uv run python -m experiments.<name>.<script>`。

- 複数の実験が使い、特定の実験に閉じないライブラリコードは、ここではなく mujou に入れる（skill `mujou-upstream` を参照）。
- checkpoint、ログ、プロットは commit しない（`runs/`、`outputs/`、`logs/` は無視される）。残したい数字は実験の `README.md` に書く。
