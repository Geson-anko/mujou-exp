"""The posthoc's noise of M18 stage 4 (diagnosis): ``code_diff`` of
hand-made codes and ``jittered``'s size and determinism."""

import math

import torch

from exp.video.diag.m18_post_noise import code_diff, jittered


def test_code_diff_of_equal_codes_is_zero() -> None:
    b = torch.randn(10, 4, generator=torch.Generator().manual_seed(0))
    d = code_diff(b.clone(), b)
    assert d == {"rel": 0.0, "rel_spread": 0.0, "max_over_sd": 0.0}


def test_code_diff_by_hand() -> None:
    # b: two codes (1, 0) and (3, 0); mean (2, 0); the spread about it is
    # (-1, 0), (1, 0): norm sqrt 2; the components' variances (unbiased)
    # 2 and 0, their mean 1: RMS std 1. a = b + (0, 0.5) on the first code.
    b = torch.tensor([[1.0, 0.0], [3.0, 0.0]])
    a = b + torch.tensor([[0.0, 0.5], [0.0, 0.0]])
    d = code_diff(a, b)
    assert math.isclose(d["rel"], 0.5 / math.sqrt(10))
    assert math.isclose(d["rel_spread"], 0.5 / math.sqrt(2))
    assert math.isclose(d["max_over_sd"], 0.5)


def test_jittered_is_rel_sd_and_deterministic() -> None:
    z = torch.zeros(2000, 50)
    j1 = jittered(z, 1e-3, 2.0, torch.Generator().manual_seed(7))
    j2 = jittered(z, 1e-3, 2.0, torch.Generator().manual_seed(7))
    assert torch.equal(j1, j2)
    assert j1.dtype == z.dtype
    rms = float(j1.double().square().mean().sqrt())
    assert abs(rms / 2e-3 - 1) < 0.01  # 1e5 draws: the RMS within 1 %
