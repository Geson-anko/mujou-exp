# continuous_stream: 動画のストリームから世界のシミュレータを PC-ALM で学ぶ

計画は [PLAN.md](PLAN.md)、実装と検証の単位は [MILESTONES.md](MILESTONES.md)。

## 仮説

PLAN.md 8 節の H0〜H4。各段の冒頭で閾値を確定させ、ここに写す。

## 実行方法

```bash
uv run python -m experiments.continuous_stream.run --config <path>  # 設定は JSON（config.py の Config）
```

出力は `runs/<日付>-<名前>/` の `config.json`（設定、mujou とこのリポジトリの SHA、未コミットの変更の有無）と `metrics.jsonl`。commit しない。

## 結果

（段ごとに追記する）
