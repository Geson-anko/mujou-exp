"""The tables of the EMA runs of M17 (diagnosis): the verdict of the criteria
fixed before the runs, on rows whose answer follows from the definitions, and
the grouping of a run tree by (arm, B) and seed."""

import json
from pathlib import Path

from exp.video.diag.ema_tables import load, ph, table, verdict
from exp.video.programs.vision import COLLAPSE

HEALTHY_EV = {
    "comp_short": 0.3,
    "comp_short_pos": 0.9,
    "comp_long": 0.5,
    "comp_long_pos": 0.9,
    **{f"ncomp_{k}": 0.6 for k in ("short", "long", "future")},
    **{f"ncomp_{k}_vis": 1.1 for k in ("short", "long", "future")},
    **{f"ncomp_{k}_mean": 1.2 for k in ("short", "long", "future")},
}
HEALTHY_POST = {
    "z_spread": 0.3,
    "restore_target_short_mask": 0.4,
    "restore_target_long_mask": 0.5,
}


def test_a_representation_that_uses_the_context_passes_every_criterion():
    v = verdict(HEALTHY_EV, HEALTHY_POST)
    assert v == dict(context=True, spread=True, target=True, ncomp=True, all=True)


def test_the_position_only_solution_fails_the_context():
    # P: every token a function of its position, so the position's mean
    # predicts as well as anything (comp = comp_pos = 0).
    ev = {**HEALTHY_EV, "comp_long": 0.0, "comp_long_pos": 0.0}
    v = verdict(ev, HEALTHY_POST)
    assert v["context"] is False and v["all"] is False


def test_a_collapse_across_stacks_fails_the_spread_at_the_threshold():
    assert verdict(HEALTHY_EV, {**HEALTHY_POST, "z_spread": COLLAPSE})["spread"]
    below = verdict(HEALTHY_EV, {**HEALTHY_POST, "z_spread": COLLAPSE / 2})
    assert below["spread"] is False and below["all"] is False


def test_tokens_that_restore_no_better_than_the_mean_image_fail_the_target():
    v = verdict(HEALTHY_EV, {**HEALTHY_POST, "restore_target_short_mask": 1.0})
    assert v["target"] is False


def test_a_completion_no_better_than_the_visible_tokens_fails_the_ncomp():
    v = verdict({**HEALTHY_EV, "ncomp_future": 1.1}, HEALTHY_POST)
    assert v["ncomp"] is False


def test_a_missing_value_is_unknown_not_a_pass():
    ev = {k: v for k, v in HEALTHY_EV.items() if k != "comp_short_pos"}
    v = verdict(ev, {})
    assert v["context"] is None and v["spread"] is None and v["all"] is None


def test_load_groups_the_seeds_of_an_arm_and_b(tmp_path: Path):
    for seed in (101, 100):
        rid = f"m17-ema_t128-abcd1234-s{seed}"
        run = tmp_path / "m17-ema-v1" / rid
        run.mkdir(parents=True)
        job = {
            "run_id": rid,
            "seed": seed,
            "name": "m17/ema_t128",
            "config": {"data": {"streams": 32}},
        }
        (run / "job.json").write_text(json.dumps(job))
        post = tmp_path / "m17-ema-post-v1" / f"m17-ema_t128-ffff0000-s{seed}"
        (post / "post/decode").mkdir(parents=True)
        pj = {"run_id": post.name, "seed": seed, "name": "m17/ema_t128"}
        pj["parent"] = f"/nas/runs/m17-ema-v1/{rid}"
        (post / "job.json").write_text(json.dumps(pj))
        line = {
            "split": "val",
            "t": 1800.0,
            "code": "vit",
            "z_spread": {100: 0.1, 101: 0.3}[seed],
        }
        (post / "post/decode/metrics.jsonl").write_text(json.dumps(line) + "\n")
    runs = load(tmp_path)
    (key,) = runs.groups
    assert key == ("t128", 32)
    rids = runs.groups[key]
    assert [runs.train[r]["seed"] for r in rids] == [100, 101]
    assert [ph(runs, r, 1800.0, "z_spread", "val") for r in rids] == [0.1, 0.3]
    assert "0.200（0.100 / 0.300）" in table(runs, ["z_spread"], ph, [1800.0], "val")
