"""``exp.vision.results.v18c.judge_long`` (L1 of V18c's long runs).

The oracles: the registered lines at their edges by hand (K1 and K3 collapse
at the line, ``sigreg_heldin`` >= K1 and ``token_r2`` >= theta_3; K2 only
below the floor, ``rank`` < floor); L1's premise is a strict fall of
``pred_ratio_calib`` from step 6000 to 60000, and a run without it is not
judged (L1 None) while its K1-K3 are still read; a run without step 60000's
numbers is not judged and one with a non-finite training mean does not meet
L1, both counted as collapsed; the seed mean
of two hand-picked values; the durations of the time file's marks, also when
the Imagenette marks fall inside the evaluation's.
"""

import json
import math
from pathlib import Path
from typing import Any

import pytest

from exp.vision.results.v18c.judge_long import (
    CKPTS,
    MID,
    PROBE_STEPS,
    STEPS,
    LongRun,
    judge,
    seed_means,
)

LINES = {"k1": 26.0, "rank_floor": 12.0, "theta3": 0.5}


def make_run(
    root: Path,
    name: str = "v18c-lr1e-4-lam0.5-r0.5-s0",
    *,
    s: float = 1.2,
    rank: float = 100.0,
    r2: float = 0.0,
    q6: float = 0.7,
    q60: float = 0.5,
    last: bool = True,
    loss: float = 1.0,
    times: str = "",
) -> LongRun:
    """A long run whose every checkpoint has the same numbers, except the
    ones named for step 6000 and 60000 (``last`` False drops step 60000)."""
    path = root / name
    (path / "posthoc_jepa").mkdir(parents=True)
    lr, lam = (1e-4, 0.5) if "lr1e-4" in name else (1e-5, 0.5)
    conf = {
        "lr": lr,
        "lam": lam,
        "mask_ratio": 0.5,
        "steps": STEPS,
        "provenance": {"code": {"snap": "0" * 40}},
    }
    (path / "jepa.json").write_text(json.dumps(conf))
    keys = ("loss", "pred", "sigreg", "mix_var", "rho_in", "grad_norm")
    train = [
        {
            "step": t,
            **{k: loss if k == "loss" else 1.0 for k in keys},
            "sec": t,
            "max_mem": 1,
        }
        for t in range(100, STEPS + 1, 100)
    ]
    (path / "train.jsonl").write_text("\n".join(map(json.dumps, train)))
    stats, probes = [], []
    for step in CKPTS:
        if step == STEPS and not last:
            continue
        row: dict[str, Any] = {"code": "z", "step": step}
        for k in (
            "sigreg_heldin",
            "token_r2",
            "rho_in",
            "participation",
            "pred_ratio",
            "pred_ratio_calib",
            "sigreg_zurich",
            "embed_drift",
            "block_drift",
        ):
            row[k] = 1.0
        row["sigreg_heldin"] = s if step == STEPS else 1.5
        row["token_r2"] = r2 if step == STEPS else 0.0
        row["pred_ratio_calib"] = q60 if step == STEPS else q6 if step == MID else 1
        stats.append(row)
        if step in PROBE_STEPS:
            probes.append(
                {
                    "code": "z",
                    "group": "zurich",
                    "step": step,
                    "rank": rank if step == STEPS else 100.0,
                    "recon_blur": 0.3,
                    "dyn_1/copy_1": 0.5,
                }
            )
    post = path / "posthoc_jepa"
    (post / "jepa.jsonl").write_text("\n".join(map(json.dumps, stats)))
    (post / "probes.jsonl").write_text("\n".join(map(json.dumps, probes)))
    if times:
        (root / "logs").mkdir(exist_ok=True)
        (root / "logs" / f"{name}.time").write_text(times)
    return LongRun(path)


def test_healthy_run_meets_l1(tmp_path: Path) -> None:
    x = judge(make_run(tmp_path), LINES)
    assert (x["K1"], x["K2"], x["K3"], x["collapsed"], x["L1"]) == (
        False,
        False,
        False,
        False,
        True,
    )
    assert x["premise"] is True and x["complete"] and not x["diverged"]


@pytest.mark.parametrize(
    ("kw", "flag", "hit"),
    [
        ({"s": 26.0}, "K1", True),
        ({"s": math.nextafter(26.0, 0.0)}, "K1", False),
        ({"rank": 12.0}, "K2", False),
        ({"rank": math.nextafter(12.0, 0.0)}, "K2", True),
        ({"r2": 0.5}, "K3", True),
        ({"r2": math.nextafter(0.5, 0.0)}, "K3", False),
    ],
)
def test_lines_at_their_edges(
    tmp_path: Path, kw: dict[str, float], flag: str, hit: bool
) -> None:
    run = make_run(
        tmp_path, s=kw.get("s", 1.2), rank=kw.get("rank", 100.0), r2=kw.get("r2", 0.0)
    )
    x = judge(run, LINES)
    assert x[flag] is hit
    assert x["L1"] is not hit


def test_premise_needs_a_strict_fall(tmp_path: Path) -> None:
    same = judge(make_run(tmp_path / "a", q6=0.7, q60=0.7), LINES)
    up = judge(make_run(tmp_path / "b", q6=0.7, q60=0.9), LINES)
    assert same["premise"] is False and up["premise"] is False
    # not judged, but the collapse is still detected (none here)
    for x in (same, up):
        assert x["L1"] is None and x["collapsed"] is False
    hit = judge(make_run(tmp_path / "c", q60=0.9, s=30.0), LINES)
    assert hit["L1"] is None and hit["K1"] is True and hit["collapsed"]


def test_missing_last_step_or_divergence_fails_l1(tmp_path: Path) -> None:
    gone = judge(make_run(tmp_path / "a", last=False), LINES)
    nan = judge(make_run(tmp_path / "b", loss=math.nan), LINES)
    for x in (gone, nan):
        assert x["collapsed"] is True and x["K1"] is None
    assert gone["premise"] is None and gone["L1"] is None and not gone["complete"]
    assert nan["premise"] is True and nan["L1"] is False
    assert nan["diverged"]


def test_seed_means(tmp_path: Path) -> None:
    a = judge(make_run(tmp_path, "v18c-lr1e-4-lam0.5-r0.5-s0", s=1.0), LINES)
    b = judge(make_run(tmp_path, "v18c-lr1e-4-lam0.5-r0.5-s1", s=2.0), LINES)
    (m,) = seed_means([a, b])
    assert (m["lr"], m["lam"], m["n"]) == (1e-4, 0.5, 2)
    assert m["trajectory"][str(STEPS)]["sigreg_heldin"] == pytest.approx(1.5)
    # step 0 has no training row, so neither seed has it
    assert m["trajectory"]["0"]["train_pred"] is None
    (one,) = seed_means([a])
    assert one["trajectory"][str(STEPS)]["sigreg_heldin"] is None


def test_times_with_imagenette_inside_the_evaluation(tmp_path: Path) -> None:
    marks = "\n".join(
        [
            "train_start 100 h-g0",
            "train_end 7100 h-g0",
            "eval_start 7200 h-g1",
            "img_start 7300 h2-g1",
            "img_end 7600 h2-g1",
            "eval_end 11200 h-g1",
        ]
    )
    run = make_run(tmp_path, times=marks)
    assert run.times == {"train": 7000.0, "eval": 4000.0, "img": 300.0}
    x = judge(run, LINES)
    assert (x["train_sec"], x["eval_sec"], x["img_sec"]) == (7000.0, 4000.0, 300.0)
