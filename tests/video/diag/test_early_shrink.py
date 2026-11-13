"""The weight statistics and the attribution of the early-shrink diagnosis
(M17): closed forms of scaled and rank-one changes, and an additive measure
whose reverted and lone parts are known exactly."""

import math

import torch
from torch import nn

from exp.video.diag.early_shrink import (
    attribution,
    output_change,
    owners,
    params_of,
    same_value,
    swapped,
    weight_change,
)


def _net() -> dict[str, nn.Module]:
    torch.manual_seed(0)
    return {"a": nn.Sequential(nn.Linear(4, 3), nn.LayerNorm(3)), "b": nn.Linear(3, 2)}


def test_weight_change_of_no_change_is_zero():
    p = params_of(_net())
    init = {k: v.detach().clone() for k, v in p.items()}
    for row in weight_change(init, p).values():
        assert row["rel"] == 0 or math.isnan(row["rel"])  # nan: a zero bias
        assert row["max_abs"] == 0
        assert row.get("rel_op", 0) == 0


def test_weight_change_of_a_scaled_tensor_is_the_scale():
    """theta = (1 + c) theta_0: rel = rel_op = |c|, max_abs = |c| max|theta_0|."""
    w0 = torch.randn(
        5, 7, generator=torch.Generator().manual_seed(1), dtype=torch.float64
    )
    c = -0.3
    row = weight_change({"w": w0}, {"w": (1 + c) * w0})["w"]
    assert math.isclose(row["rel"], abs(c), rel_tol=1e-12)
    assert math.isclose(row["rel_op"], abs(c), rel_tol=1e-6)
    assert math.isclose(row["max_abs"], abs(c) * float(w0.abs().max()), rel_tol=1e-6)


def test_weight_change_of_a_rank_one_sign_step():
    """Delta = eta sign(u) sign(v)^T of an m x n matrix: |Delta|_F =
    sigma_max(Delta) = eta sqrt(m n), max_abs = eta."""
    g = torch.Generator().manual_seed(2)
    m, n, eta = 16, 64, 1e-3
    w0 = 0.02 * torch.randn(m, n, generator=g, dtype=torch.float64)
    u = torch.randn(m, generator=g, dtype=torch.float64).sign()
    v = torch.randn(n, generator=g, dtype=torch.float64).sign()
    row = weight_change({"w": w0}, {"w": w0 + eta * torch.outer(u, v)})["w"]
    s0 = float(torch.linalg.matrix_norm(w0, ord=2))
    assert math.isclose(row["rel"] * float(w0.norm()), eta * math.sqrt(m * n))
    assert math.isclose(row["rel_op"] * s0, eta * math.sqrt(m * n), rel_tol=1e-9)
    assert math.isclose(row["max_abs"], eta, rel_tol=1e-9)


def test_owners_groups_by_module_in_order():
    names = ["a.0.weight", "a.0.bias", "a.1.weight", "b.weight"]
    assert owners(names) == {
        "a.0": ["a.0.weight", "a.0.bias"],
        "a.1": ["a.1.weight"],
        "b": ["b.weight"],
    }


def test_swapped_holds_the_values_inside_and_restores_bit_for_bit():
    p = params_of(_net())
    before = {k: v.detach().clone() for k, v in p.items()}
    zero = {"b.weight": torch.zeros_like(p["b.weight"])}
    with swapped(p, zero):
        assert torch.equal(p["b.weight"], zero["b.weight"])
        assert torch.equal(p["a.0.weight"], before["a.0.weight"])
    for k, v in p.items():
        assert torch.equal(v, before[k])


def test_attribution_of_an_additive_measure():
    """M = sum of all the parameters: reverting module o gives M(now) - d_o,
    only o gives M(init) + d_o, d_o the sum of o's change."""
    p = params_of(_net())
    init = {k: v.detach().clone() for k, v in p.items()}
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for v in p.values():
            v.add_(torch.randn(v.shape, generator=g))

    def total() -> dict[str, float]:
        return {"m": sum(float(v.double().sum()) for v in p.values())}

    m_now = total()["m"]
    m_init = sum(float(v.double().sum()) for v in init.values())
    out = attribution(p, init, total)
    assert list(out) == ["a.0", "a.1", "b"]
    for owner, names in owners(list(p)).items():
        d = sum(float((p[k].double() - init[k].double()).sum()) for k in names)
        assert math.isclose(out[owner]["revert"]["m"], m_now - d, rel_tol=1e-9)
        assert math.isclose(out[owner]["only"]["m"], m_init + d, rel_tol=1e-9)
    assert math.isclose(total()["m"], m_now, rel_tol=0)  # restored


def test_same_value_counts_two_nans_as_one():
    assert same_value(math.nan, math.nan)
    assert same_value(1.5, 1.5)
    assert not same_value(1.5, math.nextafter(1.5, 2))


def test_output_change_of_a_sign_step_on_a_constant_input_is_eta_n():
    """W_0 of std 1/sqrt(n), b_0 = 0, the step eta u 1^T (u = +-1) and the
    input c 1: |Delta W x| = eta c n sqrt(m) and |W_0 x| = c |W_0 1|, so the
    ratio is eta n sqrt(m) / |W_0 1|, about eta n."""
    torch.manual_seed(4)
    m, n, eta, c = 8, 6144, 1e-5, 0.7
    lin = nn.Linear(n, m).double()
    with torch.no_grad():
        lin.weight.normal_(0, n**-0.5)
        lin.bias.zero_()
    mods = {"e": nn.Sequential(lin)}
    init = {k: v.detach().clone() for k, v in params_of(mods).items()}
    u = torch.randn(m, dtype=torch.float64).sign()
    with torch.no_grad():
        lin.weight.add_(eta * torch.outer(u, torch.ones(n, dtype=torch.float64)))
    x = torch.full((3, n), c, dtype=torch.float64)
    r = output_change(mods, init, lambda: mods["e"](x))
    w1 = float(init["e.0.weight"].sum(1).norm())  # |W_0 1|
    assert list(r) == ["e.0"]
    assert math.isclose(r["e.0"], eta * n * math.sqrt(m) / w1, rel_tol=1e-9)
    assert 0.5 * eta * n < r["e.0"] < 2 * eta * n


def test_output_change_counts_the_bias_and_skips_unused_maps():
    """A bias moved by d with W fixed: |d| sqrt(k) / |W_0 x + b_0| over k
    inputs; a map that sees no input has no entry."""
    torch.manual_seed(5)
    used, unused = nn.Linear(3, 2).double(), nn.Linear(3, 2).double()
    mods = {"a": used, "b": unused}
    init = {k: v.detach().clone() for k, v in params_of(mods).items()}
    d = torch.tensor([0.3, -0.4], dtype=torch.float64)
    with torch.no_grad():
        used.bias.add_(d)
    x = torch.randn(5, 3, dtype=torch.float64)
    r = output_change(mods, init, lambda: used(x))
    base = float((x @ init["a.weight"].T + init["a.bias"]).norm())
    assert list(r) == ["a"]
    assert math.isclose(r["a"], float(d.norm()) * math.sqrt(5) / base, rel_tol=1e-9)
