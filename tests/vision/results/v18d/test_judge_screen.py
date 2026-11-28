"""``results/v18d/judge_screen``: the registered rules on hand-made rows.

The oracles: the reference of each arm and lambda is its largest learning
rate, a run 0.05 over it in any of its keys is pending (S+T's
pred_ratio_calib only for S+T); the pass's bounds; the choice is the least
time loss among the content-eligible, a tie within 0.01 broken by E^lin,
and with none eligible the largest min(D_E, D_Ed); D = E_a - 2^-0.1 E_b, a
quantity of an incomparable E_b left out.
"""

import pytest
import torch

from exp.lib.fidelity import ContentSums
from exp.vision.geom_eval import GeomConf
from exp.vision.results.v18d.judge_screen import (
    apply_pass,
    apply_pending,
    choose,
    eligibility,
    lines,
    run_name,
)

F64 = torch.float64


def _row(arm, w, lr, lam, **kw):
    base = {"name": run_name(arm, w, lr, lam), "arm": arm, "w": w, "lr": lr, "lam": lam,
            "collapsed": False, "diverged": False, "sigreg_heldin": 1.2,
            "time_val": 0.6, "pred_ratio_calib": 0.5}  # fmt: skip
    return base | kw


def test_names_and_lines():
    assert run_name("ST", 0.25, 1e-4, 0.5) == "v18d-ST-w0.25-lr1e-4-lam0.5-r0.5-s0"
    assert run_name("T", None, 3e-5, 0.95) == "v18d-T-lr3e-5-lam0.95-r0.5-s0"
    yard = {"init_c": {"sigreg_heldin_gauss": 1.0, "sigreg_heldin_const": 50.0,
                       "rank": 30.0, "token_r2": 0.1},
            "color": {"rank": 20.0}, "color8": {"token_r2": 0.3}}  # fmt: skip
    assert lines(yard) == {"k1": 25.5, "rank_floor": 20.0, "theta3": 0.65}


def test_pending_and_pass():
    rows = [
        _row("ST", 1.0, 1e-4, 0.5),
        _row("ST", 1.0, 1e-5, 0.5, pred_ratio_calib=0.56),  # 0.06 over: pending
        _row("ST", 1.0, 3e-5, 0.5, time_val=0.64),  # 0.04 over: not
        _row("T", None, 1e-4, 0.5, pred_ratio_calib=0.1),
        _row("T", None, 1e-5, 0.5, pred_ratio_calib=0.9),  # not a key of T
        _row("T", None, 3e-5, 0.5, time_val=0.96),  # pending and over the bound
        _row("ST", 1.0, 1e-4, 0.8, collapsed=True),
        _row("ST", 1.0, 1e-5, 0.8, sigreg_heldin=9.0),  # the reference: no pending
    ]
    apply_pending(rows)
    apply_pass(rows)
    got = [(r["pending"], r["passed"]) for r in rows]
    assert got == [(False, True), (True, False), (False, True), (False, True),
                   (False, True), (True, False), (False, False), (False, False)]  # fmt: skip
    assert rows[7]["reference"] == rows[7]["name"]


def _content(e, ed, d_e, d_ed, ok=True):
    return {"E": e, "Ed": ed, "D_E": d_e, "D_Ed": d_ed, "eligible": ok}


def test_the_choice():
    a = _row("ST", 1.0, 1e-4, 0.5, time_val=0.60, content=_content(0.5, 0.1, 0.1, 0.1))
    b = _row(
        "ST", 0.25, 1e-5, 0.8, time_val=0.605, content=_content(0.6, 0.1, 0.1, 0.1)
    )
    c = _row(
        "ST", 1.0, 1e-4, 0.8, time_val=0.4, content=_content(0.7, 0.1, -0.1, 0.1, False)
    )
    t1 = _row(
        "T", None, 1e-4, 0.5, time_val=0.3, content=_content(0.2, 0.0, -0.2, 0.1, False)
    )
    t2 = _row(
        "T", None, 1e-5, 0.5, time_val=0.2, content=_content(0.2, 0.0, -0.3, 0.5, False)
    )
    for r in (a, b, c, t1, t2):
        r["passed"] = True
    got = choose([a, b, c, t1, t2])
    assert got["ST"]["chosen"] == b["name"]  # within 0.01 of a, the larger E
    assert got["T"]["chosen"] == t1["name"] and "適格なし" in got["T"]["why"]
    assert choose([_row("ST", 1.0, 1e-4, 0.5, passed=False)])["ST"]["chosen"] is None


def test_the_eligibility_difference():
    nb = 6
    gen = torch.Generator().manual_seed(0)
    tot = torch.full((nb,), 10.0, dtype=F64)
    good = ContentSums(torch.full((1, nb), 4.0, dtype=F64), tot,
                       torch.full((1, nb), 6.0, dtype=F64), tot)  # fmt: skip
    a = {"E": 0.6, "Ed": 0.30}
    b = {"E": good.value(), "Ed": good.value(True), "sums": good}
    got = eligibility(a, b, GeomConf())
    rho = 2**-0.1
    assert got["D_E"] == pytest.approx(0.6 - rho * 0.6)
    assert got["D_Ed"] == pytest.approx(0.30 - rho * 0.4)
    assert got["checked"] == 2 and not got["eligible"]
    # E^d of b at 0 with noise: not comparable, its check left out
    noisy = ContentSums(
        good.err, tot, 10 + torch.randn(1, nb, generator=gen, dtype=F64), tot
    )
    b2 = {"E": noisy.value(), "Ed": noisy.value(True), "sums": noisy}
    got = eligibility(a, b2, GeomConf())
    assert got["checked"] == 1 and got["eligible"]
    assert not got["comparable_Ed"]["ok"]
