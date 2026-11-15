"""The tables of results/m17.md (diagnosis, M17): a cell is the seeds' mean then
each value, and a run tree gives its rows."""

import json
from pathlib import Path

from exp.video.diag.m17_tables import at, cell, fmt, load_buf


def test_cell_is_the_mean_then_each_seed():
    assert cell([1.0, 3.0]) == "2.00（1.00 / 3.00）"
    assert cell([None, 0.25]) == "0.250"
    assert cell([None]) == "–"
    assert fmt(0.00042) == "4.2e-04"
    assert fmt(1234.5) == "1234"


def test_load_buf_groups_the_seeds_of_a_setting(tmp_path: Path):
    for seed in (101, 100):
        run = tmp_path / "m17-buf-v1" / f"m17-tok_n16-abcd1234-s{seed}"
        run.mkdir(parents=True)
        config = {"data": {"streams": 32}, "vision": {"sigreg": 5e-4, "token_dim": 16}}
        job = {"run_id": run.name, "seed": seed, "config": config}
        (run / "job.json").write_text(json.dumps(job))
        row = {"split": "val", "t": 300.0, "pred": seed / 1000}
        (run / "evals.jsonl").write_text(json.dumps(row) + "\n")
    runs = load_buf(tmp_path)
    (key,) = runs.groups
    assert key == ("m17-buf-v1", 32, 16, 5e-4)
    rids = runs.groups[key]
    assert [runs.train[r]["seed"] for r in rids] == [100, 101]
    vals = [at(runs.train[r]["ev"], 300.0, "val", "pred") for r in rids]
    assert cell(vals) == "0.101（0.100 / 0.101）"
