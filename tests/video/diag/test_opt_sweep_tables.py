"""The tables of the optimizer sweep of M17 (diagnosis): the criteria L, U and
C on rows whose verdict follows from the definitions (the untrained network
does not learn; a non-finite objective fails), and the grouping of a run tree
by (kind, lr, warmup, scale, lambda) beside the untrained ViT of the same arm
and seed."""

import json
import math
from pathlib import Path
from typing import Any

from exp.video.diag.opt_sweep_tables import (
    collapse_table,
    comp_table,
    course,
    course_table,
    criteria,
    drift_parts_table,
    drift_table,
    label,
    load,
    lr_table,
    run_criteria,
    sens_table,
    summary,
    times_table,
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
        ("sgd", 1e-2, 0.0, "none", None),
        ("adamw", 1e-4, 0.0, "none", None),
        ("adamw", 1e-4, 300.0, "none", None),
    }
    got = {k: run_criteria(runs, r[0], 1800.0) for k, r in runs.groups.items()}
    assert got[("sgd", 1e-2, 0.0, "none", None)]["learns"] is True
    assert got[("adamw", 1e-4, 0.0, "none", None)]["learns"] is False
    assert got[("adamw", 1e-4, 300.0, "none", None)]["L"] is True
    assert got[("adamw", 1e-4, 300.0, "none", None)]["learns"] is None  # no posthoc
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
    rid = runs.groups[("sgd", 1e-2, 0.0, "none", None)][0]
    pairs = [
        {"t": 300.0, "t2": 600.0, "delta": 300.0, "drift": 0.5},
        {"t": 300.0, "t2": 900.0, "delta": 600.0, "drift": 9.0},
    ]
    path = tmp_path / "drift.json"
    path.write_text(json.dumps({"runs": [{"run": rid, "pairs": pairs}]}))
    line = next(x for x in drift_table(runs, [path]).splitlines() if "sgd" in x)
    assert line.split(" | ")[2] == "0.500" and "9.00" not in line


def test_the_lr_table_divides_by_time_0_and_the_untrained_vit(tmp_path: Path):
    """J(t)/J(0) and the posthoc over the untrained ViT; the untrained ViT's
    own row is 1 by definition; the drift is that of the pair ending at t."""
    _sweep(tmp_path)
    runs = load(tmp_path)
    rid = runs.groups[("sgd", 1e-2, 0.0, "none", None)][0]
    path = tmp_path / "drift.json"
    pairs = [{"t": 1500.0, "t2": 1800.0, "delta": 300.0, "drift": 0.25}]
    path.write_text(json.dumps({"runs": [{"run": rid, "pairs": pairs}]}))
    lines = lr_table(runs, 1800.0, [path]).splitlines()
    sgd = next(x for x in lines if "`sgd`" in x).strip("| ").split(" | ")
    want_recon = HEALTHY_POST["recon_blur"] / INIT["recon_blur"]
    want_dyn = HEALTHY_POST["dyn_1/copy_1"] / INIT["dyn_1/copy_1"]
    assert sgd[1:] == [
        "0.900",
        f"{want_recon:.3f}",
        f"{want_dyn:.3f}",
        "–",  # no rank in the posthoc of the fixture
        "0.300",
        "0.250",
    ]
    at_start = next(x for x in lines if "`adamw` lr=0.0001 |" in x)
    assert at_start.split(" | ")[1:4] == ["1.00", "1.00", "1.00"]
    assert "| 学習前の ViT（t128、seed 100） | 1 | 1 | 1 |" in lines[-1]


def _cells(table: str, needle: str) -> list[str]:
    line = next(x for x in table.splitlines() if needle in x)
    return line.strip("| ").split(" | ")


def _course_run(root: Path) -> str:
    """One sgd run whose held-out J, with J(0) = 1, is 0.5, 0.7, 0.6 and 1.2
    at 60, 120, 300 and 360 s; a run with no eval has no course."""
    _sweep(root)
    rid = "m17-ssig_t128-00000005-s100"
    js = [(60.0, 0.5), (120.0, 0.7), (300.0, 0.6), (360.0, 1.2)]
    _write(
        root / "m17-opt-v1" / rid,
        {
            "run_id": rid,
            "seed": 100,
            "name": "m17/ssig_t128",
            "config": {"optim_vision": {"kind": "sgd", "lr": 1e-3, "warmup": 0.0}},
        },
        **{
            "evals.jsonl": [
                {"split": "val", "t": t, "loss": j, "pred": j / 2, "tok_within": j}
                for t, j in js
            ]
            + [{"split": "train", "t": 60.0, "loss": 0.01}],
            "baselines.json": {
                "val": {"init_loss": 1.0, "init_pred": 0.8, "init_tok_within": 0.4}
            },
        },
    )
    return rid


def test_the_course_counts_the_evals_below_j0_and_the_rises(tmp_path: Path):
    """By hand: least 0.5 at 60 s, greatest 1.2 at 360 s, three evals below
    J(0) = 1, two rises in three pairs, 1.2 / 0.6 = 2 from 300 s; the train
    split is not read."""
    rid = _course_run(tmp_path)
    runs = load(tmp_path)
    c = course(runs, rid)
    assert c is not None
    assert c["least"] == (60.0, 0.5) and c["greatest"] == (360.0, 1.2)
    assert c["below"] == 3 and (c["rises"], c["pairs"]) == (2, 3)
    assert c["late_ratio"] == 2.0
    assert _cells(course_table(runs), "`sgd` lr=0.001")[1:] == [
        "4",
        "0.500（60）",
        "1.20（360）",
        "3",
        "2 / 3",
        "2.00",
    ]
    assert _cells(course_table(runs), "`sgd` lr=0.01")[1] == "1"


def test_the_collapse_table_reads_time_0_from_the_baselines(tmp_path: Path):
    rid = _course_run(tmp_path)
    runs = load(tmp_path)
    got = _cells(collapse_table(runs, 360.0), "`sgd` lr=0.001")
    assert got[1:] == ["0.800", "0.250", "0.600", "0.400", "0.500", "1.20"]
    assert rid in runs.train


def test_the_values_behind_c_are_those_the_verdict_compares(tmp_path: Path):
    """comp and ncomp from the eval row, restore_target from the posthoc;
    sens of the run beside the untrained ViT's (the yardstick of sens)."""
    _sweep(tmp_path)
    runs = load(tmp_path)
    comp = _cells(comp_table(runs, 1800.0), "`sgd` lr=0.01")
    assert comp[1] == "0.300 / 0.900 / –"  # no comp_short_vis in the fixture
    assert comp[2] == "0.600 / – / 1.10 / 1.20"
    assert comp[3] == "0.400" and comp[6] == "0.500"
    sens = _cells(sens_table(runs, 1800.0), "`sgd` lr=0.01")
    assert sens[1:] == ["0.010 / 0.020 / –", "0.200 / 0.300 / –"]


def test_the_times_table_is_the_verdict_at_each_checkpoint(tmp_path: Path):
    """The fixture has rows at 1800 s only: L, U and C hold there for the
    healthy sgd run; elsewhere J is missing (L unknown) and so is the
    posthoc."""
    _sweep(tmp_path)
    runs = load(tmp_path)
    got = _cells(times_table(runs, [300.0, 1800.0]), "`sgd` lr=0.01")
    assert got[1:] == ["???", "○○○"]


def test_the_drift_parts_are_the_numerator_and_denominator(tmp_path: Path):
    _sweep(tmp_path)
    runs = load(tmp_path)
    rid = runs.groups[("sgd", 1e-2, 0.0, "none", None)][0]
    pair = {"t": 1500.0, "t2": 1800.0, "delta": 300.0}
    pair |= {"change": 0.5, "apart": 2.0, "drift": 0.25}
    path = tmp_path / "drift.json"
    path.write_text(json.dumps({"runs": [{"run": rid, "pairs": [pair]}]}))
    got = _cells(drift_parts_table(runs, [path], 1800.0), "`sgd`")
    assert got[1:] == ["0.500", "2.00", "0.250"]


def test_a_fan_in_run_is_its_own_group_labelled_eta(tmp_path: Path):
    """``scale`` ``fan_in`` keys a run apart from the same lr without it, and
    its label names the rate eta."""
    for i, scale in enumerate((None, "fan_in", None)):
        optim = {"kind": "adamw", "lr": 1e-4, "warmup": 0.0}
        if scale is not None:
            optim["scale"] = scale
        rid = f"m17-ssig_t128-0000000{i}-s100"
        _write(
            tmp_path / "m17-opt-v1" / rid,
            {
                "run_id": rid,
                "seed": 100,
                "name": "m17/ssig_t128",
                "config": {"optim_vision": optim},
            },
            **{"evals.jsonl": [], "baselines.json": {"val": {"init_loss": 1.0}}},
        )
    runs = load(tmp_path)
    assert set(runs.groups) == {
        ("adamw", 1e-4, 0.0, "none", None),
        ("adamw", 1e-4, 0.0, "fan_in", None),
    }
    assert label(("adamw", 1e-4, 0.0, "fan_in", None)) == "`adamw` η=0.0001"
    assert (
        label(("adamw", 1e-4, 300.0, "none", None)) == "`adamw` lr=0.0001 warmup 300 秒"
    )
    # no evals: unknown, the fan_in run named by its eta
    assert "| `adamw` | なし | η=0.0001、0.0001 | – |" in summary(runs, 1800.0)


def test_lambda_keys_a_run_apart_and_batches_join_under_one_rate(tmp_path: Path):
    """``vision.sigreg`` keys a run apart from the same optimizer at another
    lambda, and its label names it; two training batches load together, and
    ``lr`` keeps the runs of that rate only."""
    cases = [
        ("m17-plr-v1", 1e-5, 0.134),
        ("m17-plr-v1", 1e-4, 0.134),
        ("m17-lam-v1", 1e-5, 0.373),
    ]
    for i, (batch, lr, lam) in enumerate(cases):
        optim = {"kind": "adamw", "lr": lr, "warmup": 0.0, "scale": "fan_in"}
        rid = f"m17-ssig_t128-0000000{i}-s100"
        _write(
            tmp_path / batch / rid,
            {
                "run_id": rid,
                "seed": 100,
                "name": "m17/ssig_t128",
                "config": {"optim_vision": optim, "vision": {"sigreg": lam}},
            },
            **{"evals.jsonl": [], "baselines.json": {"val": {"init_loss": 1.0}}},
        )
    runs = load(tmp_path, ["m17-lam-v1", "m17-plr-v1"], [], lr=1e-5)
    assert set(runs.groups) == {
        ("adamw", 1e-5, 0.0, "fan_in", 0.134),
        ("adamw", 1e-5, 0.0, "fan_in", 0.373),
    }
    assert label(("adamw", 1e-5, 0.0, "fan_in", 0.373)) == "`adamw` η=1e-05 λ=0.373"
    shown = summary(runs, 1800.0)
    assert "| `adamw` | なし | η=1e-05（λ=0.134）、η=1e-05（λ=0.373） | – |" in shown
    assert len(load(tmp_path, ["m17-lam-v1", "m17-plr-v1"], []).train) == 3
