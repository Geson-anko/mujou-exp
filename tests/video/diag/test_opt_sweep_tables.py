"""The tables of the optimizer sweep of M17 (diagnosis): the criteria L, U and
C on rows whose verdict follows from the definitions (the untrained network
does not learn; a non-finite objective fails), and the grouping of a run tree
by (kind, lr, warmup) beside the untrained ViT of the same arm and seed."""

import json
import math
from pathlib import Path
from typing import Any

from exp.video.diag.opt_sweep_tables import (
    criteria,
    drift_table,
    load,
    run_criteria,
    summary,
    train_loss,
)

HEALTHY_EV = {
    "comp_short": 0.3,
    "comp_short_pos": 0.9,
    "comp_long": 0.5,
    "comp_long_pos": 0.9,
    **{f"ncomp_{k}": 0.6 for k in ("short", "long", "future")},
    **{f"ncomp_{k}_vis": 1.1 for k in ("short", "long", "future")},
    **{f"ncomp_{k}_mean": 1.2 for k in ("short", "long", "future")},
}
INIT = {
    "recon_blur": 0.24,
    "dyn_1/copy_1": 0.87,
    "sens_dither": 0.02,
    "sens_shift": 0.3,
}
HEALTHY_POST = {
    "z_spread": 0.3,
    "restore_target_short_mask": 0.4,
    "restore_target_long_mask": 0.5,
    "recon_blur": 0.2,
    "dyn_1/copy_1": 0.8,
    "sens_dither": 0.01,
    "sens_shift": 0.2,
}


def test_a_healthy_run_below_its_start_and_the_untrained_vit_learns():
    c = criteria(1.0, 0.9, HEALTHY_EV, HEALTHY_POST, INIT)
    assert c["L"] and c["U"] and c["C"] and c["learns"]


def test_the_untrained_network_itself_does_not_learn():
    """At its own values J equals J(0) and the posthoc equals the untrained
    ViT: L and U are strict, so neither holds (C may: 'at most' in sens)."""
    c = criteria(1.0, 1.0, HEALTHY_EV, {**HEALTHY_POST, **INIT}, INIT)
    assert c["L"] is False and c["U"] is False and c["learns"] is False
    assert c["sens"] is True


def test_a_diverged_objective_fails_l():
    for j in (math.nan, math.inf):
        assert criteria(1.0, j, HEALTHY_EV, HEALTHY_POST, INIT)["L"] is False


def test_u_needs_both_the_reconstruction_and_the_dynamics():
    for key in ("recon_blur", "dyn_1/copy_1"):
        post = {**HEALTHY_POST, key: INIT[key] * 1.01}
        c = criteria(1.0, 0.9, HEALTHY_EV, post, INIT)
        assert c["U"] is False and c["learns"] is False


def test_a_collapse_fails_c_and_a_missing_value_is_unknown():
    c = criteria(1.0, 0.9, HEALTHY_EV, {**HEALTHY_POST, "z_spread": 0.0}, INIT)
    assert c["spread"] is False and c["C"] is False and c["learns"] is False
    c = criteria(None, 0.9, HEALTHY_EV, HEALTHY_POST, INIT)
    assert c["L"] is None and c["learns"] is None


def test_the_training_loss_is_the_mean_of_the_300_s_before():
    stream = [{"t": float(t), "loss": float(t)} for t in range(1, 601)]
    assert train_loss(stream, 300.0) == sum(range(1, 301)) / 300
    assert train_loss(stream, 600.0) == sum(range(301, 601)) / 300
    assert train_loss([], 300.0) is None


def test_the_training_loss_skips_the_null_rows_of_a_diverged_run():
    """The loop writes a non-finite loss as null; the mean is of the rest."""
    stream = [{"t": 400.0, "loss": 1.0}, {"t": 500.0, "loss": None}]
    assert train_loss(stream, 600.0) == 1.0


def _write(path: Path, job: dict[str, Any], **files: Any) -> None:
    path.mkdir(parents=True)
    (path / "job.json").write_text(json.dumps(job))
    for name, content in files.items():
        p = path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith(".jsonl"):
            p.write_text("".join(json.dumps(x) + "\n" for x in content))
        else:
            p.write_text(json.dumps(content))


def _sweep(root: Path) -> None:
    """Three runs of seed 100: sgd lr 1e-2 whose J drops, adamw lr 1e-4 at
    its start (J and the posthoc equal to time 0), and adamw lr 1e-4 with a
    warm-up whose posthoc is missing."""
    runs = [
        ("sgd", 1e-2, 0.0, 0.9, HEALTHY_POST),
        ("adamw", 1e-4, 0.0, 1.0, {**HEALTHY_POST, **INIT}),
        ("adamw", 1e-4, 300.0, 0.9, None),
    ]
    for i, (kind, lr, warmup, j, post) in enumerate(runs):
        rid = f"m17-ssig_t128-0000000{i}-s100"
        optim = {"kind": kind, "lr": lr, "warmup": warmup}
        _write(
            root / "m17-opt-v1" / rid,
            {
                "run_id": rid,
                "seed": 100,
                "name": "m17/ssig_t128",
                "config": {"optim_vision": optim},
            },
            **{
                "evals.jsonl": [{"split": "val", "t": 1800.0, "loss": j, **HEALTHY_EV}],
                "baselines.json": {"val": {"init_loss": 1.0}},
            },
        )
        if post is not None:
            _write(
                root / "m17-opt-post-v1" / f"p{i}",
                {"run_id": "p", "seed": 100, "parent": f"/nas/m17-opt-v1/{rid}"},
                **{
                    "post/decode/metrics.jsonl": [
                        {"split": "val", "t": 1800.0, "code": "vit", **post}
                    ]
                },
            )
    _write(
        root / "m17-opt-base-v1" / "b",
        {"run_id": "b", "seed": 100, "name": "m17/ssig_t128"},
        **{
            "post/decode/baselines.jsonl": [
                {"split": "val", "code": "init", **INIT},
                {"split": "val", "code": "pca", "recon_blur": 0.01},
            ]
        },
    )


def test_load_groups_the_runs_and_reads_the_untrained_vit_of_the_seed(
    tmp_path: Path,
):
    _sweep(tmp_path)
    runs = load(tmp_path)
    assert set(runs.groups) == {
        ("sgd", 1e-2, 0.0),
        ("adamw", 1e-4, 0.0),
        ("adamw", 1e-4, 300.0),
    }
    got = {k: run_criteria(runs, r[0], 1800.0) for k, r in runs.groups.items()}
    assert got[("sgd", 1e-2, 0.0)]["learns"] is True
    assert got[("adamw", 1e-4, 0.0)]["learns"] is False
    assert got[("adamw", 1e-4, 300.0)]["L"] is True
    assert got[("adamw", 1e-4, 300.0)]["learns"] is None  # no posthoc
    shown = summary(runs, 1800.0)
    assert "| `sgd` | 0.01 | – | – |" in shown
    assert "| `adamw` | なし | 0.0001（warmup 300 秒） | – |" in shown


def test_a_run_that_diverged_fails_l_and_learns_and_is_listed(tmp_path: Path):
    """As the loop leaves it: the last eval row has a null loss, the run
    stops before 1800 s and ``result.json`` says ``diverged``; there is no
    posthoc at 1800 s. J at 1800 s is not finite, so L and learns fail
    (not unknown), and the summary lists the lr as diverged."""
    _sweep(tmp_path)
    rid = "m17-ssig_t128-00000009-s100"
    _write(
        tmp_path / "m17-opt-v1" / rid,
        {
            "run_id": rid,
            "seed": 100,
            "name": "m17/ssig_t128",
            "config": {"optim_vision": {"kind": "sgd", "lr": 1.0, "warmup": 0.0}},
        },
        **{
            "evals.jsonl": [
                {"split": "val", "t": 60.0, "loss": 0.9, **HEALTHY_EV},
                {"split": "val", "t": 120.0, "loss": None},
            ],
            "stream.jsonl": [{"t": 100.0, "loss": 2.0}, {"t": 110.0, "loss": None}],
            "baselines.json": {"val": {"init_loss": 1.0}},
            "result.json": {
                "status": "diverged",
                "reason": "loss not finite",
                "t_end": 120.0,
            },
        },
    )
    runs = load(tmp_path)
    c = run_criteria(runs, rid, 1800.0)
    assert c["L"] is False and c["learns"] is False
    assert c["U"] is None  # no posthoc at 1800 s
    assert train_loss(runs.train[rid]["stream"], 300.0) == 2.0
    assert "| `sgd` | 0.01 | – | 1 |" in summary(runs, 1800.0)


def test_the_drift_table_takes_the_pairs_300_s_apart(tmp_path: Path):
    _sweep(tmp_path)
    runs = load(tmp_path)
    rid = runs.groups[("sgd", 1e-2, 0.0)][0]
    pairs = [
        {"t": 300.0, "t2": 600.0, "delta": 300.0, "drift": 0.5},
        {"t": 300.0, "t2": 900.0, "delta": 600.0, "drift": 9.0},
    ]
    path = tmp_path / "drift.json"
    path.write_text(json.dumps({"runs": [{"run": rid, "pairs": pairs}]}))
    line = next(x for x in drift_table(runs, [path]).splitlines() if "sgd" in x)
    assert line.split(" | ")[2] == "0.500" and "9.00" not in line
