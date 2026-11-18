"""The posthoc's noise of M18 stage 4 (diagnosis): ``code_diff`` of
hand-made codes, ``jittered``'s size and determinism, and the fit at the
posthoc's seed against the posthoc's own line."""

import math

import torch
from torch import Tensor

from exp.video.compose import compose
from exp.video.diag.m18_post_noise import _fit, code_diff, jittered
from exp.video.measure.encoder_eval import EncoderEval, _Code
from exp.video.measure.posthoc import DECODER_SEED, as_float
from exp.video.programs.vision import DATA_ENV
from tests.video.packed import smooth_tours, tiny


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


def _last_frame_code(stacks: Tensor) -> Tensor:
    """A code of stacks ``(n, N, C, H, W)``: every fifth value of the last
    frame, ``(n, D)``."""
    return as_float(stacks[:, -1]).flatten(1)[:, ::5]


def test_the_fit_at_the_posthocs_seed_is_the_posthocs_line(tmp_path, monkeypatch):
    """``_fit`` copies ``EncoderEval._evaluate``'s decoder: at the generator
    of ``seed + DECODER_SEED`` its ``recon``, ``recon_blur`` and
    ``recon_fit`` are those of the val line the evaluation writes for the
    same code (on a small Walking Tours, bit for bit on CPU). A change of
    the evaluation that the diagnosis does not follow breaks this."""
    root = smooth_tours(tmp_path / "data" / "d", {"A": 120, "B": 80, "H": 200})
    monkeypatch.setenv(DATA_ENV, str(root.parent))
    conf = compose([*tiny(root.name), "seed=7"])
    v = conf.vision
    assert v is not None
    ev = EncoderEval(None, conf, tmp_path / "o", root=root, windows=16, steps=3)
    line = next(
        x for x in ev.evaluate_code("c", _last_frame_code) if x["split"] == "val"
    )
    code = _Code("c", _last_frame_code)
    z_train, _ = ev._encode(code, ev._train)
    z_val = ev._split_z(code, "val")[:, 0]
    f0 = as_float(ev._measures.windows["val"][:, v.frames - 1])
    got = _fit(ev, conf.seed + DECODER_SEED, z_train.flatten(0, 1), z_val, f0)
    assert got == {k: line[k] for k in ("recon", "recon_blur", "recon_fit")}
