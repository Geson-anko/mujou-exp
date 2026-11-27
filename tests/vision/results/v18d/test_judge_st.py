"""``results/v18d/judge_st``: the registered rules on hand-made rows.

The oracles: the reference of each lambda is its largest learning rate, a
run 0.05 over it in sigreg_heldin or L_st is pending; the pass needs each
side below 0.95 (one side's shortfall not made up by the other); the choice
is the least L_st among the content-eligible, a tie within 0.01 broken by
E^lin, and with none eligible the largest min(D_E, D_Ed); a pass whose
content was not measured is left out; collapse by K1 to K3 of the lines.
"""

import json

from exp.vision.results.v18d.judge_st import (
    apply_pass,
    apply_pending,
    choose,
    judge,
    run_name,
)


def _row(lr, lam, **kw):
    base = {"name": run_name(lr, lam), "lr": lr, "lam": lam, "collapsed": False,
            "diverged": False, "sigreg_heldin": 1.2, "L_now": 0.6, "L_time": 0.8,
            "L_st": 0.7}  # fmt: skip
    return base | kw


def test_names():
    assert run_name(1e-4, 0.5) == "v18d-st-lr1e-4-lam0.5-r0.5-s0"
    assert (
        run_name(3e-5, 0.95, perm=True, seed=1) == "v18d-stperm-lr3e-5-lam0.95-r0.5-s1"
    )


def test_pending_and_pass():
    rows = [
        _row(1e-4, 0.5),
        _row(1e-5, 0.5, L_st=0.76),  # 0.06 over: pending
        _row(3e-5, 0.5, sigreg_heldin=1.24),  # 0.04 over: not
        _row(1e-4, 0.8, L_now=0.3, L_time=0.96, L_st=0.63),  # one side over
        _row(1e-4, 0.95, collapsed=True),
        _row(1e-5, 0.95, sigreg_heldin=9.0),  # the reference: no pending
        _row(1e-5, 0.8, L_time=0.95, L_st=0.6),  # at the bound of a side
    ]
    apply_pending(rows)
    apply_pass(rows)
    got = [(r["pending"], r["passed"]) for r in rows]
    assert got == [(False, True), (True, False), (False, True), (False, False),
                   (False, False), (False, False), (False, False)]  # fmt: skip
    assert rows[5]["reference"] == rows[5]["name"]


def _content(e, d_e, d_ed, ok=True):
    return {"E": e, "Ed": 0.1, "D_E": d_e, "D_Ed": d_ed, "eligible": ok}


def test_the_choice():
    a = _row(1e-4, 0.5, L_st=0.60, content=_content(0.5, 0.1, 0.1))
    b = _row(1e-5, 0.8, L_st=0.605, content=_content(0.6, 0.1, 0.1))
    c = _row(1e-4, 0.8, L_st=0.4, content=_content(0.7, -0.1, 0.1, False))
    d = _row(3e-5, 0.5, L_st=0.1, content=None)
    for r in (a, b, c, d):
        r["passed"] = True
    got = choose([a, b, c, d])
    assert got["chosen"] == b["name"] and "tie" in got["why"]
    assert got["unmeasured"] == [d["name"]]
    got = choose(
        [c, _row(1e-5, 0.5, passed=True, content=_content(0.2, -0.3, 0.5, False))]
    )
    assert got["chosen"] == c["name"] and "適格なし" in got["why"]
    assert choose([_row(1e-4, 0.5, passed=False)])["chosen"] is None


def test_collapse_and_divergence(tmp_path):
    ln = {"k1": 26.7, "rank_floor": 20.0, "theta3": 0.6}
    run = tmp_path / run_name(1e-4, 0.5)
    (run / "posthoc_v18d").mkdir(parents=True)
    rows = [{"step": s, "loss": 1.0, "pred": 1.0, "sigreg": 1.0, "pred_now": 1.0,
             "pred_time": 1.0} for s in (100, 6000)]  # fmt: skip
    (run / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    sel = {"step": 6000, "sigreg_heldin": 1.0, "rank": 19.0, "token_r2": 0.2,
           "L_now": 0.5, "L_time": 0.7, "L_st": 0.6}  # fmt: skip
    (run / "posthoc_v18d" / "st_select.jsonl").write_text(json.dumps(sel) + "\n")
    got = judge(run, 1e-4, 0.5, ln)
    assert (got["K1"], got["K2"], got["K3"], got["collapsed"]) == (
        False,
        True,
        False,
        True,
    )
    rows[0]["pred_time"] = float("nan")
    (run / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    got = judge(run, 1e-4, 0.5, ln)
    assert got["diverged"] and got["collapsed"]
