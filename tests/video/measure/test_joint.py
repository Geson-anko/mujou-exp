"""The latent and image measures of M16 on hand-made windows."""

from typing import override

import pytest
import torch

from exp.video.dyn.modules import build_dynamics
from exp.video.measure.joint import Windows, image_rows, latent_errors
from exp.video.schema.dyn import DynamicsConf

N, O, H = 2, 3, 4  # frames of a chunk, observed chunks, horizon


class _Stub:
    """An encoder whose z is the first pixel of the last frame of a chunk."""

    def encode(self, chunks: torch.Tensor) -> torch.Tensor:
        return chunks[:, -1, 0, 0, 0][:, None]


def _windows(n: int = 5) -> Windows:
    t = N - 1 + O + H
    images = (
        torch.arange(t, dtype=torch.float32).view(1, t, 1, 1, 1).expand(n, t, 1, 2, 2)
    )
    images = images + 100 * torch.arange(n).view(n, 1, 1, 1, 1)
    return Windows(images.contiguous(), N, O, H)


class TestWindows:
    def test_chunk_j_ends_at_frame_n_minus_1_plus_j(self):
        w = _windows()
        z = w.z(_Stub())  # type: ignore[arg-type]
        assert z.shape == (5, O + H, 1)
        expected = (N - 1 + torch.arange(O + H)).float()[None] + 100 * torch.arange(5)[
            :, None
        ]
        assert torch.equal(z[..., 0], expected)

    def test_the_true_frame_of_a_horizon_is_that_of_the_chunk_it_predicts(self):
        w = _windows()
        z = w.z(_Stub())  # type: ignore[arg-type]
        for h in (0, 1, 4):
            assert torch.equal(w.frame(h)[:, 0, 0, 0], z[:, O + h - 1, 0])


class TestLatentErrors:
    def test_a_rollout_that_copies_has_the_copy_error(self):
        """A residual dynamics with the readout at 0 predicts the last
        observation again: ``roll{h}`` is ``copy{h}``."""
        d = 3
        dyn = build_dynamics(DynamicsConf(d_s=8, residual=True), d, d, seed=0)
        z = torch.randn(6, O + H, d)
        row, pred = latent_errors(dyn, z, observed=O, horizons=[1, 4], residual=True)
        assert pred.shape == (6, H, d)
        for h in (1, 4):
            assert row[f"roll{h}"] == pytest.approx(row[f"copy{h}"])
            assert row[f"roll{h}/copy{h}"] == pytest.approx(1.0)
        assert row["nonfinite"] == 0

    def test_a_constant_z_is_predicted_by_the_copy_and_the_mean(self):
        dyn = build_dynamics(DynamicsConf(d_s=8, residual=True), 3, 3, seed=0)
        z = torch.ones(6, O + H, 3)
        row, _ = latent_errors(dyn, z, observed=O, horizons=[2], residual=True)
        assert row["copy2"] == 0 and row["mean2"] == 0


class TestImageRows:
    def test_the_bars_and_the_picture(self):
        """A decoder that is the identity on z (the frame's value) given the
        true z as the rollout has no error; the copy and the mean have."""
        w = _windows()
        z = w.z(_Stub())  # type: ignore[arg-type]
        dec = _Fill()
        rows, picture = image_rows(dec, w, z, z[:, O:], [1, 4])  # type: ignore[arg-type]
        assert rows["img_roll1"] == pytest.approx(0) and rows[
            "img_roll4"
        ] == pytest.approx(0)
        assert rows["img_recon"] == pytest.approx(0)
        assert rows["pix_copy1"] == pytest.approx(1.0)  # one frame further
        assert rows["pix_copy4"] == pytest.approx(16.0)
        var = (
            (100 * torch.arange(5).float()).var(correction=0).item()
        )  # the windows' spread
        assert rows["pix_mean1"] == pytest.approx(var)
        assert picture.shape == (2 * 4, 3, 1, 2, 2)  # 4 windows: truth over decoded
        assert torch.equal(picture[0], picture[1])


class _Fill(torch.nn.Module):
    """z ``(n, 1)`` to a ``(n, 1, 2, 2)`` image of that value."""

    @override
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z[:, :, None, None].expand(-1, -1, 2, 2)
