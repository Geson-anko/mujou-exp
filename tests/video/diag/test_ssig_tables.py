"""The tables of the stream SIGReg runs of M17 (diagnosis): the verdict of the
five criteria fixed before the runs, on rows whose answer follows from the
definitions, the direction rule of the seed pairs, and the grouping of a run
tree by (arm, k, lambda) and seed beside the EMA B=1 runs."""

import json
from pathlib import Path
from typing import Any

from exp.video.diag.ssig_tables import (
    direction,
    load,
    ph,
    run_verdict,
    summary,
    verdict,
)
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
    "sens_dither": 0.01,
    "sens_shift": 0.2,
}
INIT = {"sens_dither": 0.02, "sens_shift": 0.3}


def test_a_healthy_run_no_more_sensitive_than_its_start_passes_all_five():
    v = verdict(HEALTHY_EV, HEALTHY_POST, INIT)
    assert v == dict(
        context=True, spread=True, target=True, ncomp=True, sens=True, all=True
    )


def test_the_untrained_network_itself_passes_the_sensitivity():
    # Criterion 5 is "at most": the network it started from is not amplified.
    v = verdict(HEALTHY_EV, {**HEALTHY_POST, **INIT}, INIT)
    assert v["sens"] is True


def test_a_larger_ratio_for_the_dither_or_the_shift_fails_the_sensitivity():
    for kind in ("dither", "shift"):
        post = {**HEALTHY_POST, f"sens_{kind}": INIT[f"sens_{kind}"] * 1.01}
        v = verdict(HEALTHY_EV, post, INIT)
        assert v["sens"] is False and v["all"] is False


def test_the_four_collapse_criteria_are_those_of_the_ema_runs():
    # The position-only solution (comp = comp_pos = 0) fails the context.
    v = verdict(
        {**HEALTHY_EV, "comp_short": 0.0, "comp_short_pos": 0.0}, HEALTHY_POST, INIT
    )
    assert v["context"] is False and v["all"] is False
    # A collapse across stacks: z_spread half the threshold.
    v = verdict(HEALTHY_EV, {**HEALTHY_POST, "z_spread": COLLAPSE / 2}, INIT)
    assert v["spread"] is False
    # Unmasked tokens no better than the mean image.
    v = verdict(HEALTHY_EV, {**HEALTHY_POST, "restore_target_long_mask": 1.0}, INIT)
    assert v["target"] is False
    # A completion no better than copying the visible tokens.
    v = verdict({**HEALTHY_EV, "ncomp_short": 1.1}, HEALTHY_POST, INIT)
    assert v["ncomp"] is False


def test_a_missing_yardstick_is_unknown_not_a_pass():
    v = verdict(HEALTHY_EV, HEALTHY_POST, {})
    assert v["sens"] is None and v["all"] is None


def test_a_difference_needs_all_four_seed_pairs_to_agree():
    assert direction([0.1, 0.2], [0.3, 0.4]) == "<"
    assert direction([0.5, 0.6], [0.3, 0.4]) == ">"
    assert direction([0.1, 0.5], [0.3, 0.4]) == "–"  # 0.5 > 0.3: one pair flips
    assert direction([0.1, None], [0.3, 0.4]) == "–"  # a missing value


def _write(path: Path, job: dict[str, Any], **files: Any) -> None:
    path.mkdir(parents=True)
    (path / "job.json").write_text(json.dumps(job))
    for name, lines in files.items():
        p = path / name.replace("__jsonl", ".jsonl").replace("__", "/")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(json.dumps(x) + "\n" for x in lines))


def test_load_groups_seeds_and_reads_the_yardstick_and_ema_of_the_same_seed(
    tmp_path: Path,
):
    vis = {"datapoint_every": 6, "sigreg": 0.134}
    for seed in (101, 100):
        rid = f"m17-ssig_t128-abcd1234-s{seed}"
        _write(
            tmp_path / "m17-ssig-v1" / rid,
            {
                "run_id": rid,
                "seed": seed,
                "name": "m17/ssig_t128",
                "config": {"vision": vis, "data": {"streams": 1}},
            },
            **{"evals.jsonl": [{"split": "val", "t": 1800.0, **HEALTHY_EV}]},
        )
        # The healthy line, but the shift of seed 101 above its start.
        post = {**HEALTHY_POST, "sens_shift": {100: 0.2, 101: 0.4}[seed]}
        _write(
            tmp_path / "m17-ssig-post-v1" / f"m17-ssig_t128-ffff0000-s{seed}",
            {
                "run_id": "p",
                "seed": seed,
                "name": "m17/ssig_t128",
                "parent": f"/nas/runs/m17-ssig-v1/{rid}",
            },
            post__decode__metrics__jsonl=[
                {"split": "val", "t": 1800.0, "code": "vit", **post}
            ],
        )
        _write(
            tmp_path / "m17-ssig-base-v1" / f"m17-ssig_t128-0000aaaa-s{seed}",
            {"run_id": "b", "seed": seed, "name": "m17/ssig_t128"},
            post__decode__baselines__jsonl=[
                {"split": "val", "code": "init", **INIT},
                {"split": "val", "code": "pca", "sens_shift": 9.0},
            ],
        )
        erid = f"m17-ema_t128-4a082154-s{seed}"
        _write(
            tmp_path / "m17-ema-v1" / erid,
            {
                "run_id": erid,
                "seed": seed,
                "name": "m17/ema_t128",
                "config": {"vision": {"sigreg": 0.0}, "data": {"streams": 1}},
            },
        )
        # The EMA sensitivities from post-v2, the rest from post-v1.
        for batch, line in (
            ("m17-ema-post-v1", {"recon_blur": 0.3, "sens_shift": None}),
            ("m17-ema-post-v2", {"sens_shift": 0.7}),
        ):
            _write(
                tmp_path / batch / f"x-{seed}",
                {
                    "run_id": "e",
                    "seed": seed,
                    "name": "m17/ema_t128",
                    "parent": f"/nas/runs/m17-ema-v1/{erid}",
                },
                post__decode__metrics__jsonl=[
                    {"split": "val", "t": 1800.0, "code": "vit", **line}
                ],
            )
    runs = load(tmp_path)
    assert list(runs.groups) == [("t128", 6, 0.134)]
    rids = runs.groups[("t128", 6, 0.134)]
    assert [runs.train[r]["seed"] for r in rids] == [100, 101]
    assert [run_verdict(runs, r, 1800.0, "val")["all"] for r in rids] == [True, False]
    assert "| `t128` | 6 | 0.134 | 棄却 | sens（101） |" in summary(runs, 1800.0)
    ema = runs.ema["t128"]
    assert [ph(runs, r, 1800.0, "sens_shift", "val") for r in ema] == [0.7, 0.7]
    assert [ph(runs, r, 1800.0, "recon_blur", "val") for r in ema] == [0.3, 0.3]
