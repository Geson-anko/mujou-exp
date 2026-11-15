"""The predictions P1 to P3 of the M17 per-layer learning rate (diagnosis)
on rows written as ``early_shrink live`` writes them, whose verdicts follow
from the definitions: q is output_change / (lr F), P1 compares the
embedding's q with the other maps', P2 the reverts, P3 plr against opt."""

import json
import math
from pathlib import Path
from typing import Any

from exp.video.diag.plr_attribution import (
    EMBED,
    add_fleet,
    additivity,
    embed_below_others,
    parse_tag,
    revert_summary,
    summarize,
    tables,
)

OTHER = "encoder.stages.0.blocks.0.attn.qkv"


def _rows(
    out: Path, tw: dict[float, float], change: dict[str, float], att: dict | None
) -> None:
    rows: list[dict[str, Any]] = [
        {
            "t": 0.0,
            "frame": 0,
            "tok_within": tw[0.0],
            "step_bound": {EMBED: 2.0, OTHER: 4.0},
        }
    ]
    for t, v in tw.items():
        if t:
            rows.append(
                {
                    "t": t,
                    "frame": round(30 * t),
                    "tok_within": v,
                    "output_change": {k: c * t for k, c in change.items()},
                }
            )
    if att is not None:
        rows.append({"t": max(tw), **att})
    out.mkdir(parents=True)
    (out / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def _att(embed: float, other: float) -> dict:
    return {
        EMBED: {"revert": {"tok_within": embed}, "only": {"tok_within": 0.3}},
        OTHER: {"revert": {"tok_within": other}, "only": {"tok_within": 0.4}},
        "predictor": {"revert": {"tok_within": 0.0}, "only": {"tok_within": 0.4}},
    }


def test_parse_tag_reads_arm_kind_lr_and_seconds():
    assert parse_tag("plr-adamw-1e-05-s60") == ("plr", "adamw", 1e-5, 60.0)
    assert parse_tag("opt-rmsprop-0.01-s1") == ("opt", "rmsprop", 0.01, 1.0)


def test_embed_below_others_and_revert_summary():
    assert embed_below_others({EMBED: 1.0, OTHER: 2.0}) is True
    assert embed_below_others({EMBED: 2.0, OTHER: 2.0}) is False
    assert embed_below_others({OTHER: 2.0}) is None
    r = revert_summary(_att(0.2, 0.2))
    assert r["p2"] is True  # equal: "at most"
    assert r["revert_other_module"] == OTHER
    assert revert_summary(_att(0.3, 0.2))["p2"] is False


def test_additivity_is_one_when_the_moves_add_up():
    """tok_within = t0 + sum of the moved modules' deltas: reverting i gives
    moved - delta_i, moving i alone t0 + delta_i, so every encoder module's
    (revert + only) / (moved + t0) is 1; the predictor is left out."""
    t0, deltas = 0.5, {EMBED: -0.2, OTHER: -0.1}
    moved = t0 + sum(deltas.values())
    att = {
        m: {"revert": {"tok_within": moved - d}, "only": {"tok_within": t0 + d}}
        for m, d in deltas.items()
    }
    att["predictor"] = {"revert": {"tok_within": 9.0}, "only": {"tok_within": 9.0}}
    a = additivity(att, moved, t0)
    assert math.isclose(a["add_min"], 1.0) and math.isclose(a["add_max"], 1.0)
    assert math.isclose(a["add_embed"], 1.0)
    att[EMBED]["only"]["tok_within"] = 1.0  # above moved + t0: not additive
    a = additivity(att, moved, t0)
    assert a["add_max_module"] == EMBED
    assert math.isclose(a["add_max"], (moved + 0.2 + 1.0) / (moved + t0))
    assert a["add_min_module"] == OTHER


def test_summarize_computes_q_p1_p2_p3(tmp_path: Path):
    """plr at lr 1e-4: output_change c t at frame 30 t, so q = c / (30 lr);
    the embedding's c is below qkv's (P1 holds); revert 0.1 against 0.2
    (P2 holds); tok_within at 15 s 0.05 against opt's 0.01 (P3 holds).
    opt at the same lr has the opposite P1 and P2."""
    tw_plr = {0.0: 0.47, 1.0: 0.4, 10.0: 0.2, 15.0: 0.05, 60.0: 0.03}
    tw_opt = {0.0: 0.47, 1.0: 0.3, 10.0: 0.1, 15.0: 0.01, 60.0: 0.005}
    _rows(
        tmp_path / "plr-adamw-0.0001-s60",
        tw_plr,
        {EMBED: 0.003, OTHER: 0.006},
        _att(0.1, 0.2),
    )
    _rows(
        tmp_path / "opt-adamw-0.0001-s60",
        tw_opt,
        {EMBED: 0.06, OTHER: 0.006},
        _att(0.4, 0.05),
    )
    s = summarize(tmp_path)
    plr, opt = s["plr-adamw-0.0001"], s["opt-adamw-0.0001"]
    q1 = plr["q"]["1"]
    assert math.isclose(q1["q"][EMBED], 0.003 / (30 * 1e-4))
    assert math.isclose(q1["q_over_bound"][OTHER], 0.006 / (30 * 1e-4) / 4.0)
    assert q1["p1"] is True and plr["q"]["10"]["p1"] is None  # judged at 1 s
    assert opt["q"]["1"]["p1"] is False
    assert plr["att"]["60"]["p2"] is True and opt["att"]["60"]["p2"] is False
    assert plr["p3"] is True and plr["p3_values"] == {"plr": 0.05, "opt": 0.01}
    assert "p3" not in opt
    assert plr["values"]["0"]["tok_within"] == 0.47
    assert math.isclose(plr["att"]["60"]["add_embed"], (0.1 + 0.3) / (0.03 + 0.47))
    shown = tables(s)
    assert (
        "| `adamw` | 0.0001 | 0.4 / 0.2 / 0.05 / 0.03 | 0.3 / 0.1 / 0.01 / 0.005 | ○ |"
        in shown
    )
    assert (
        "| `adamw` | 0.0001 | 1 | 2（`s0.b0.attn.qkv`） | ○ | 20 | 2（`s0.b0.attn.qkv`） | × |"
        in shown
    )


def test_p3_is_unknown_without_the_sweep_and_false_on_nan(tmp_path: Path):
    _rows(
        tmp_path / "plr-rmsprop-0.01-s60",
        {0.0: 0.47, 15.0: math.nan},
        {EMBED: 1.0, OTHER: 1.0},
        None,
    )
    assert summarize(tmp_path)["plr-rmsprop-0.01"]["p3"] is None
    _rows(
        tmp_path / "opt-rmsprop-0.01-s60",
        {0.0: 0.47, 15.0: 0.001},
        {EMBED: 1.0, OTHER: 1.0},
        None,
    )
    assert summarize(tmp_path)["plr-rmsprop-0.01"]["p3"] is False


def test_add_fleet_finds_the_run_of_the_same_arm_kind_and_lr(tmp_path: Path):
    """The plr entry gets the fan_in run's row at 60 s, not the sweep's run of
    the same number nor one with a warm-up."""
    runs = tmp_path / "runs"
    for batch, rid, scale, warmup, tw in (
        ("m17-plr-v1", "a", "fan_in", 0.0, 0.3),
        ("m17-opt-v1", "b", None, 0.0, 0.002),
        ("m17-opt-v1", "c", None, 300.0, 0.9),
    ):
        d = runs / batch / rid
        d.mkdir(parents=True)
        o: dict[str, Any] = {"kind": "adamw", "lr": 1e-5, "warmup": warmup}
        if scale:
            o["scale"] = scale
        (d / "job.json").write_text(json.dumps({"config": {"optim_vision": o}}))
        rows = [
            {"split": "val", "t": 60.0, "tok_within": tw, "pred": 0.1},
            {"split": "train", "t": 60.0, "tok_within": 5.0, "pred": 5.0},
        ]
        (d / "evals.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    s = {
        "plr-adamw-1e-05": {"arm": "plr", "kind": "adamw", "lr": 1e-5},
        "opt-adamw-1e-05": {"arm": "opt", "kind": "adamw", "lr": 1e-5},
        "opt-adamw-0.01": {"arm": "opt", "kind": "adamw", "lr": 1e-2},
    }
    add_fleet(s, runs)
    assert s["plr-adamw-1e-05"]["fleet_60"] == {"tok_within": 0.3, "pred": 0.1}
    assert s["opt-adamw-1e-05"]["fleet_60"] == {"tok_within": 0.002, "pred": 0.1}
    assert s["opt-adamw-0.01"]["fleet_60"] is None
