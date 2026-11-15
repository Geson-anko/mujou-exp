"""The tables of results/m18.md's eta sweep and decoder axis (diagnosis): the
selection rule written before the runs, the first passing rate, the
interpolation of the step-size scaling, and the keys of the runs across the
batches."""

import json
from pathlib import Path

import pytest

from exp.video.diag.m18_sweep_tables import (
    Candidate,
    facts,
    first_pass,
    last_t,
    load,
    scaled,
    select,
)


def _c(
    key: str,
    rb: float | None,
    *,
    u: bool | None = True,
    diverged: bool = False,
) -> Candidate:
    return Candidate(key, rb, u, diverged)


class TestSelect:
    def test_the_lowest_recon_blur_wins_in_any_order(self):
        cands = [_c("a", 0.3), _c("b", 0.1), _c("c", 0.2)]
        assert select(cands) == "b"
        assert select(cands[::-1]) == "b"

    def test_a_diverged_run_or_one_without_u_is_not_eligible(self):
        cands = [
            _c("a", 0.01, diverged=True),
            _c("b", 0.02, u=False),
            _c("c", 0.03, u=None),
            _c("d", None),
            _c("e", 0.5),
        ]
        assert select(cands) == "e"
        assert select(cands[:4]) is None
        assert select([]) is None


def test_first_pass_is_the_lowest_passing_rate_in_any_order():
    assert first_pass([1e-2, 1e-4, 1e-3], [True, None, True]) == 1e-3
    assert first_pass([1e-5, 1e-4], [False, None]) is None


def test_scaled_interpolates_linearly_from_time_0():
    rows = [{"t": t, "split": "val", "m": v} for t, v in ((60.0, 1.0), (120.0, 0.4))]
    rows += [{"t": 90.0, "split": "test", "m": 9.0}]  # another split
    assert scaled(rows, 2.0, 30.0, "m") == pytest.approx(1.5)
    assert scaled(rows, 2.0, 0.0, "m") == pytest.approx(2.0)
    assert scaled(rows, 2.0, 100.0, "m") == pytest.approx(0.6)
    assert scaled(rows, None, 30.0, "m") is None  # no time 0
    assert scaled(rows, 2.0, 121.0, "m") is None  # beyond the evaluations


def test_scaled_follows_a_path_in_lr_times_t():
    """If the value is a function of lr t alone (one path at any rate), the
    reference at t lr / lr_ref is the value at the lower rate (here linear in
    lr t, so the interpolation is exact)."""
    ref, lr = 3e-5, 1e-5

    def value(rate: float, t: float) -> float:
        return 2.0 - 1e4 * rate * t

    rows = [{"t": t, "split": "val", "m": value(ref, t)} for t in (60.0, 120.0, 900.0)]
    t = 1800.0
    want = value(lr, t)
    assert scaled(rows, value(ref, 0.0), t * lr / ref, "m") == pytest.approx(want)


def test_last_t_skips_non_finite_values_and_other_splits():
    rows = [
        {"t": 60.0, "split": "val", "m": 1.0},
        {"t": 120.0, "split": "val", "m": float("nan")},
        {"t": 180.0, "split": "test", "m": 0.5},
        {"t": 240.0, "split": "val", "m": float("inf")},
    ]
    assert last_t(rows, "m") == 60.0
    assert last_t(rows[1:], "m") is None


def _run(root: Path, batch: str, rid: str, config: dict, after=()) -> Path:
    d = root / batch / rid
    (d / "post/decode").mkdir(parents=True)
    job = {"run_id": rid, "config": config, "after": list(after)}
    (d / "job.json").write_text(json.dumps(job))
    (d / "status.json").write_text(json.dumps({"state": "completed"}))
    return d


def _conf(kind: str, lr: float, sampling: str, *, b: int = 1, depth: int = 2) -> dict:
    return {
        "optim_vision": {"kind": kind, "lr": lr},
        "data": {"streams": b, "tours": {"sampling": sampling}},
        "ae": {"decoder_depth": depth},
    }


def test_load_keys_the_b1_runs_of_every_batch_and_their_posthocs(tmp_path: Path):
    s1 = _run(tmp_path, "m18-ae-v1", "a", _conf("rmsprop", 2.9e-5, "stream"))
    _run(tmp_path, "m18-ae-v1", "b32", _conf("rmsprop", 2.9e-5, "stream", b=32))
    sw = _run(tmp_path, "m18-sweep-v1", "c", _conf("sgd", 4.09e-7, "pool"))
    dec = _run(tmp_path, "m18-dec-v1", "d", _conf("rmsprop", 1e-4, "stream", depth=8))
    post = _run(
        tmp_path, "m18-dec-post-v1", "e", _conf("rmsprop", 1e-4, "stream"), [dec.name]
    )
    stale = _run(tmp_path, "m18-sweep-post-v1", "f", {}, ["b32"])  # B = 32's
    runs = load(tmp_path)
    assert runs.train == {
        ("rmsprop", 2.9e-5, "stream", 2): s1,
        ("sgd", 4.09e-7, "pool", 2): sw,
        ("rmsprop", 1e-4, "stream", 8): dec,
    }
    assert runs.post == {("rmsprop", 1e-4, "stream", 8): post}
    assert stale not in runs.post.values()
    assert runs.base is None


def test_facts_reads_r_u_and_the_overfit_check(tmp_path: Path):
    st = _run(tmp_path, "m18-sweep-v1", "a", _conf("adamw", 1e-3, "stream"))
    row = {"t": 1800.0, "split": "val", "stack_mse": 0.1, "pool_stack_mse": 0.9}
    (st / "evals.jsonl").write_text(json.dumps(row) + "\n")
    po = _run(tmp_path, "m18-sweep-post-v1", "b", {}, [st.name])
    m = {"t": 1800.0, "split": "val", "recon": 0.3, "recon_blur": 0.2}
    m |= {"dyn_1/copy_1": 0.4, "rank": 12.0, "own_recon": 0.6, "own_recon_blur": 0.5}
    (po / "post/decode/metrics.jsonl").write_text(json.dumps(m) + "\n")
    pl = _run(tmp_path, "m18-sweep-v1", "c", _conf("adamw", 1e-3, "pool"))
    (pl / "status.json").write_text(json.dumps({"state": "diverged"}))
    (pl / "evals.jsonl").write_text(json.dumps(row | {"pool_stack_mse": 0.0}) + "\n")
    runs = load(tmp_path)
    vit = {"recon_blur": 0.318, "dyn_1/copy_1": 0.653}
    f = facts(runs, ("adamw", 1e-3, "stream", 2), vit)
    assert (f.end, f.r, f.u, f.overfit) == (0.1, True, True, None)
    assert (f.recon, f.recon_blur, f.dyn, f.rank) == (0.3, 0.2, 0.4, 12.0)
    assert (f.own, f.own_blur) == (0.6, 0.5)
    g = facts(runs, ("adamw", 1e-3, "pool", 2), vit)
    assert g.overfit is False  # diverged, though 0 is below the bar
    assert facts(runs, ("adamw", 1e-2, "stream", 2), vit).state == "無い"
