"""The tables of results/m18.md stage 1 (diagnosis): the verdicts are those
written before the runs (a missing value is unknown, a diverged run fails),
the reading table's rows, Q5's condition, and the training rate without the
evaluations."""

import json
from pathlib import Path

import pytest

from exp.video.diag.m18_tables import (
    POOL_BAR,
    R_BAR,
    below,
    both,
    facts,
    load,
    mark,
    q2,
    q5,
    reading,
    train_rate,
)


def test_below_fails_a_diverged_run_and_leaves_a_missing_value_unknown():
    assert below(0.1, 0.2, diverged=False) is True
    assert below(0.2, 0.2, diverged=False) is False  # strict
    assert below(None, 0.2, diverged=False) is None
    assert below(None, 0.2, diverged=True) is False
    assert below(0.1, 0.2, diverged=True) is False


@pytest.mark.parametrize(
    ("a", "b", "out"),
    [
        (True, True, True),
        (True, False, False),
        (None, False, False),
        (True, None, None),
        (None, None, None),
    ],
)
def test_both_is_false_first_then_unknown(a, b, out):
    assert both(a, b) is out
    assert both(b, a) is out


def test_reading_gives_each_row_of_the_table():
    assert mark(True) == "○" and mark(False) == "×" and mark(None) == "不明"
    assert reading(False, True, True).startswith("実装かモデル")
    assert reading(True, False, False).startswith("学習の設定かモデル")
    assert reading(True, True, False) == "時間相関のある順序が原因"
    assert reading(True, True, True) == "段 2 に進む"
    assert reading(True, False, True).startswith("予想していない")
    assert reading(None, True, True).startswith("不明")
    assert reading(True, None, True).startswith("不明")


def test_q2_compares_r_and_u_each():
    assert q2((False, True), (False, True)) == "当たり"
    assert q2((False, True), (False, False)) == "外れ"  # the same "learned" (no)
    assert q2((None, True), (False, True)) == "不明"


def test_q5_is_judged_only_when_b1_went_below_its_time_0():
    assert q5(0.5, 0.9, 0.7) == "当たり"
    assert q5(0.8, 0.9, 0.7) == "外れ"
    assert q5(0.95, 0.9, 0.7).startswith("判定しない")
    assert q5(None, 0.9, 0.7) == "不明"


def test_train_rate_subtracts_the_evaluations():
    # 800 units in 10 s of wall, 2 s of which an evaluation: 100 units/s
    timing = [
        {"wall": "2026-11-15T00:00:00+00:00", "unit": 100, "fps": 1.0},
        {"wall": "2026-11-15T00:00:05+00:00", "t": 60.0, "eval_s": 2.0},
        {"wall": "2026-11-15T00:00:10+00:00", "unit": 900, "fps": 1.0},
    ]
    assert train_rate(timing) == pytest.approx(100.0)
    assert train_rate(timing[:1]) is None


def _run(root: Path, batch: str, rid: str, kind: str, sampling: str, **kw) -> Path:
    d = root / batch / rid
    (d / "post/decode").mkdir(parents=True)
    config = {
        "optim_vision": {"kind": kind},
        "data": {"streams": 1, "tours": {"sampling": sampling}},
    }
    job = {"run_id": rid, "config": config, "after": kw.get("after", [])}
    (d / "job.json").write_text(json.dumps(job))
    return d


def _lines(path: Path, *rows: dict) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_facts_reads_the_criteria_of_a_run_and_its_posthoc(tmp_path: Path):
    tr = _run(tmp_path, "m18-ae-v1", "a-s100", "rmsprop", "stream")
    (tr / "status.json").write_text(json.dumps({"state": "completed"}))
    (tr / "baselines.json").write_text(json.dumps({"val": {"stack_mse": 1.2}}))
    _lines(tr / "evals.jsonl", {"t": 1800.0, "split": "val", "stack_mse": R_BAR / 2})
    po = _run(
        tmp_path, "m18-ae-post-v1", "b-s100", "rmsprop", "stream", after=[tr.name]
    )
    row = {"t": 1800.0, "split": "val", "recon_blur": 0.1, "dyn_1/copy_1": 0.9}
    _lines(po / "post/decode/metrics.jsonl", row)
    pl = _run(tmp_path, "m18-ae-v1", "c-s100", "rmsprop", "pool")
    (pl / "status.json").write_text(json.dumps({"state": "diverged"}))
    _lines(pl / "evals.jsonl", {"t": 1800.0, "split": "val", "pool_stack_mse": 0.0})
    runs = load(tmp_path)
    assert runs.post[("rmsprop", "stream", 1)] == po
    vit = {"recon_blur": 0.2, "dyn_1/copy_1": 0.8}
    f = facts(runs, ("rmsprop", "stream", 1), vit)
    assert (f.t0, f.end, f.r) == (1.2, R_BAR / 2, True)
    assert f.u is False  # dyn 0.9 is not below 0.8
    g = facts(runs, ("rmsprop", "pool", 1), vit)
    assert g.overfit is False and g.r is False  # diverged, though 0 < POOL_BAR
    assert 0.0 < POOL_BAR
    assert facts(runs, ("sgd", "stream", 1), vit).state == "無い"
