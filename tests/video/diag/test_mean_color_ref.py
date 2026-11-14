"""The mean-color reconstructions (M17 diagnosis): the codes are the means of
the ViT's tubelets and of the stack, a frame constant on each patch is drawn
back exactly, the tubelet mean is the least-squares constant, the affine fit
is the least squares, and the whole run on a small Walking Tours."""

import json

import pytest
import torch

from exp.lib.blur_error import blur_recon
from exp.video.compose import compose
from exp.video.diag.mean_color_ref import (
    fit_affine,
    main,
    measure,
    stack_means,
    stack_picture,
    tube_picture,
    tubelet_means,
)
from exp.video.jepa.vit import patchify_tubelets
from exp.video.programs.vision import DATA_ENV
from tests.video.packed import smooth_tours, tiny

N, C, S, P, TAU = 4, 3, 8, 4, 2  # 2 slots x 2 x 2 tubelets of 2 frames x 4 x 4
GRID = (N // TAU, S // P, S // P)


def _x(n: int = 5, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, N, C, S, S, generator=g, dtype=torch.float64)


def _blocks(colors: torch.Tensor) -> torch.Tensor:
    """Stacks ``(n, N, C, S, S)`` constant on each tubelet, of the colors
    ``(n, gt, gh, gw, C)``."""
    x = colors.permute(0, 1, 4, 2, 3)  # (n, gt, C, gh, gw)
    x = x.repeat_interleave(TAU, 1).repeat_interleave(P, -2).repeat_interleave(P, -1)
    return x


def test_the_tube_code_is_the_mean_of_each_vit_tubelet():
    """Token t of ``patchify_tubelets`` (its tubelet's (tau, C, p, p) values)
    averaged over its frames and pixels is the code's colors t C .. t C + C."""
    x = _x()
    tok = patchify_tubelets(x, P, TAU).reshape(len(x), -1, TAU, C, P, P)
    want = tok.mean((2, 4, 5)).flatten(1)
    torch.testing.assert_close(tubelet_means(x, P, TAU), want, rtol=0, atol=1e-12)


def test_the_stack_code_is_the_mean_of_the_tubelet_means():
    x = _x()
    tube = tubelet_means(x, P, TAU).reshape(len(x), -1, C)
    torch.testing.assert_close(stack_means(x), tube.mean(1), rtol=0, atol=1e-12)


def test_a_stack_constant_on_each_tubelet_is_drawn_back_exactly():
    """Colors constant on each tubelet: the last frame is its last slot's
    tubelet means, so the direct picture is the frame and its errors 0 (the
    mean image another); a stack of one color is its stack mean's picture."""
    g = torch.Generator().manual_seed(1)
    x = _blocks(torch.rand(4, *GRID, C, generator=g, dtype=torch.float64))
    f = x[:, -1]
    got = tube_picture(tubelet_means(x, P, TAU), GRID, P)
    torch.testing.assert_close(got, f, rtol=0, atol=1e-15)
    mean = torch.full((1, C, S, S), 0.5, dtype=torch.float64)
    assert all(v < 1e-28 for v in measure(got, f, mean).values())
    one = torch.rand(4, C, generator=g, dtype=torch.float64)
    y = one[:, None, :, None, None].expand(4, N, C, S, S)
    drawn = stack_picture(stack_means(y), S)
    torch.testing.assert_close(drawn, y[:, -1], rtol=0, atol=1e-15)
    assert blur_recon(drawn, y[:, -1], mean) < 1e-28


def test_the_tubelet_mean_is_the_least_squares_constant_of_its_tubelet():
    """Among images constant on each tubelet, the squared error of any colors
    c is that of the means m plus (values per tubelet) |c - m|^2 (Pythagoras:
    the residual of the mean sums to 0 on each tubelet), so the mean is the
    one minimum."""
    x = _x()
    m = tubelet_means(x, P, TAU).reshape(len(x), *GRID, C)
    resid = x - _blocks(m)
    sums = tubelet_means(resid, P, TAU)
    torch.testing.assert_close(sums, torch.zeros_like(sums), rtol=0, atol=1e-12)
    g = torch.Generator().manual_seed(2)
    c = torch.rand(m.shape, generator=g, dtype=torch.float64)
    err = ((x - _blocks(c)) ** 2).sum()
    want = (resid**2).sum() + TAU * P * P * ((c - m) ** 2).sum()
    torch.testing.assert_close(err, want, rtol=1e-12, atol=0)
    assert err > (resid**2).sum()


class TestFitAffine:
    def test_an_affine_target_is_recovered(self):
        g = torch.Generator().manual_seed(3)
        z = torch.randn(200, 4, generator=g)
        w = torch.randn(4, 6, generator=g)
        b = torch.randn(6, generator=g)
        got_b, got_w = fit_affine(z, (z.double() @ w.double() + b).float(), batch=7)
        torch.testing.assert_close(got_w, w, rtol=0, atol=1e-5)
        torch.testing.assert_close(got_b, b, rtol=0, atol=1e-5)

    def test_the_residual_is_orthogonal_to_the_code_and_sums_to_zero(self):
        """The normal equations of the least squares on any target (u8 frames
        read as x / 255): the residual sums to 0 and is orthogonal to every
        component of the code."""
        g = torch.Generator().manual_seed(4)
        z = torch.randn(300, 5, generator=g)
        y = torch.randint(0, 256, (300, C, 2, 2), generator=g).to(torch.uint8)
        b, w = fit_affine(z, y, batch=64)
        r = (y.double() / 255).flatten(1) - (z.double() @ w.double() + b.double())
        torch.testing.assert_close(
            r.sum(0), torch.zeros(12, dtype=r.dtype), atol=1e-4, rtol=0
        )
        torch.testing.assert_close(
            z.double().T @ r, torch.zeros(5, 12, dtype=r.dtype), atol=1e-3, rtol=0
        )

    def test_it_is_no_worse_than_the_direct_picture_on_its_stacks(self):
        """The direct picture is a linear map of the tube code: the affine fit
        of the last frames on the same stacks errs no more."""
        x = _x(n=64, seed=5).float()
        code, f = tubelet_means(x, P, TAU), x[:, -1]
        b, w = fit_affine(code, f)
        lin = (code @ w + b).view_as(f)
        direct = tube_picture(code, GRID, P)
        assert ((lin - f) ** 2).sum() <= ((direct - f) ** 2).sum()

    def test_a_rank_deficient_code_gets_a_fit(self):
        """A component repeated: the least-norm solution, still exact."""
        g = torch.Generator().manual_seed(6)
        z = torch.randn(50, 2, generator=g)
        z = torch.cat([z, z[:, :1]], 1)
        y = z[:, :1] * 2 + 1
        b, w = fit_affine(z, y)
        torch.testing.assert_close(z @ w + b, y, rtol=0, atol=1e-5)


def test_the_run_on_a_small_tours_writes_every_reference(tmp_path, monkeypatch):
    """The whole diagnosis on 16 x 16 frames: both codes, both splits, the
    three reconstructions; the linear map measured here and by the evaluation
    (``recon_lin_blur``) agree; the yardsticks rerun."""
    root = smooth_tours(tmp_path / "data" / "d", {"A": 120, "B": 80, "H": 200})
    monkeypatch.setenv(DATA_ENV, str(root.parent))
    overrides = [*tiny(root.name), "seed=7"]
    out = tmp_path / "o"
    main(
        [
            "--out",
            str(out),
            "--windows",
            "16",
            "--steps",
            "3",
            "--yardsticks",
            *overrides,
        ]
    )
    got = json.loads((out / "mean_color_ref.json").read_text())
    conf = compose(overrides)
    assert got["config"]["seed"] == 7 and got["config"]["decoder_seed"] == 10
    v = conf.vision
    assert v is not None
    assert (
        got["codes"]["tube"]["D"] == (v.frames // v.tubelet) * (16 // v.patch) ** 2 * 3
    )
    assert got["codes"]["stack"]["D"] == 3
    for code in ("tube", "stack"):
        for split in ("val", "test"):
            e = got["codes"][code][split]
            assert set(e) == {"direct", "linear", "decoder"}
            assert e["linear"]["recon_blur"] == pytest.approx(
                e["decoder"]["recon_lin_blur"], rel=1e-4
            )
            assert e["linear"]["recon"] == pytest.approx(
                e["decoder"]["recon_lin"], rel=1e-4
            )
            assert "dyn_1/copy_1" in e["decoder"]
    assert {"pca/val", "init/val", "pca/test", "init/test"} == set(got["yardsticks"])
    names = {p.name for p in (out / "eval").iterdir()}
    assert {"tube.png", "stack-dyn.png", "codes.jsonl", "baselines.jsonl"} <= names
    assert (out / "refs.png").is_file()
