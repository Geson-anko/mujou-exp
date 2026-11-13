import pytest
import torch

from exp.video.diag.stream_stats import drift
from exp.video.measure.sensitivity import (
    KINDS,
    dither,
    ratio,
    sensitivities,
    shift,
)

CPU = torch.device("cpu")


def _stacks(n: int = 12, seed: int = 0) -> torch.Tensor:
    """u8 stacks ``(n, 2, 1, 4, 4)``: 32 values each."""
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (n, 2, 1, 4, 4), generator=g).to(torch.uint8)


def _sens(code, seed: int = 7) -> dict[str, float]:
    return sensitivities(
        code,
        _stacks(),
        gen=torch.Generator().manual_seed(seed),
        batch=5,
        device=CPU,
    )


def test_the_dither_is_within_half_a_level_and_the_shift_moves_one_column():
    x = _stacks().float() / 255
    d = dither(x, torch.Generator().manual_seed(0)) - x
    assert d.abs().max() <= 0.5 / 255 + 1e-7 and d.abs().min() >= 0
    # U(-1/2, 1/2) / 255 has mean square 1 / (12 255^2); of 384 draws the
    # relative sd of the mean is sqrt(1/80 - 1/144) / (1/12) / sqrt(384) = 4.6 %
    assert float((d * 255).pow(2).mean()) == pytest.approx(1 / 12, rel=0.2)
    s = shift(x)
    assert torch.equal(s[..., 1:], x[..., :-1]) and torch.equal(s[..., 0], x[..., 0])


def test_the_ratio_is_the_drift_of_stream_stats():
    g = torch.Generator().manual_seed(1)
    a, b = torch.randn(9, 4, generator=g), torch.randn(9, 4, generator=g)
    assert ratio(b, a) == pytest.approx(drift(a, b), rel=1e-12)


def test_the_identity_code_has_the_ratios_of_the_pixels():
    """f the identity on the stack: sens_<kind> equals sens_<kind>_pix (the
    same dither, drawn batch by batch from the same generator)."""
    out = _sens(lambda x: x.flatten(1))
    for k in KINDS:
        assert out[f"sens_{k}"] == pytest.approx(out[f"sens_{k}_pix"], rel=1e-9)
        assert 0 < out[f"sens_{k}"] < 1


def test_an_orthogonal_map_of_the_code_changes_nothing():
    q, _ = torch.linalg.qr(
        torch.randn(32, 32, generator=torch.Generator().manual_seed(2))
    )

    def rotated(x: torch.Tensor) -> torch.Tensor:
        return x.flatten(1) @ q.T

    def scaled(x: torch.Tensor) -> torch.Tensor:  # and a scale of the code
        return 3.0 * x.flatten(1)

    base = _sens(lambda x: x.flatten(1))
    for f in (rotated, scaled):
        out = _sens(f)
        for k in KINDS:
            assert out[f"sens_{k}"] == pytest.approx(base[f"sens_{k}"], rel=1e-5)


def test_a_code_blind_to_the_dither_is_zero_on_it():
    """The u8 levels of the stack (x rounded to its level) do not see the
    dither, which stays within half a level: sens_dither 0; they see the
    shift."""

    def rounded(x: torch.Tensor) -> torch.Tensor:  # the u8 level: blind to dither
        return (x * 255).round().flatten(1)

    out = _sens(rounded)
    assert out["sens_dither"] == 0.0 and out["sens_shift"] > 0
