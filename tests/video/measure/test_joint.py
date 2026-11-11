"""The latent and image measures of M16 on hand-made windows."""

from typing import override

import pytest
import torch

from exp.video.dyn.modules import build_dynamics
from exp.video.measure.dynamics import open_loop
from exp.video.measure.joint import Windows, dream, image_rows, latent_errors
from exp.video.schema.dyn import DynamicsConf

N, O, H = 2, 3, 4  # frames of a chunk, observed chunks, horizon


class _Stub:
    """An encoder whose z is the first pixel of the last frame of a chunk."""

    def encode(self, chunks: torch.Tensor) -> torch.Tensor:
        return chunks[:, -1, 0, 0, 0][:, None]


def _windows(n: int = 5, horizon: int = H, apart: float = 100) -> Windows:
    """Frames of value ``k + apart * i`` at frame ``k`` of window ``i``."""
    t = N - 1 + O + horizon
    images = (
        torch.arange(t, dtype=torch.float32).view(1, t, 1, 1, 1).expand(n, t, 1, 2, 2)
    )
    images = images + apart * torch.arange(n).view(n, 1, 1, 1, 1)
    return Windows(images.contiguous(), N, O, horizon)


class TestWindows:
    def test_chunk_j_ends_at_frame_n_minus_1_plus_j(self):
        w = _windows()
        z = w.z(_Stub())  # type: ignore[arg-type]
        assert z.shape == (5, O + H, 1)
        expected = (N - 1 + torch.arange(O + H)).float()[None] + 100 * torch.arange(5)[
            :, None
        ]
        assert torch.equal(z[..., 0], expected)

    def test_the_first_chunks_only(self):
        w = _windows()
        assert torch.equal(w.z(_Stub(), chunks=O), w.z(_Stub())[:, :O])  # type: ignore[arg-type]

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


class TestImageRowsPrefix:
    def test_the_prefix_names_the_decoded_keys_and_not_the_bars(self):
        w = _windows()
        z = w.z(_Stub())  # type: ignore[arg-type]
        rows, _ = image_rows(_Fill(), w, z, z[:, O:], [1, 4], prefix="rec_")  # type: ignore[arg-type]
        assert set(rows) == {
            "rec_recon",
            "rec_roll1",
            "pix_copy1",
            "pix_mean1",
            "rec_roll4",
            "pix_copy4",
            "pix_mean4",
        }


DREAMT = 20  # the dream's horizon in these tests


class TestDream:
    def test_a_dynamics_that_copies_dreams_the_copy(self):
        """A residual dynamics with the readout at 0 predicts the last
        observed z again, from any state: decoded by ``_Fill``, both dreams
        are the last observed frame. The windows' values are ``k + 3 i`` (5
        windows): the copy's error at h is h², the mean image's the spread
        of ``3 i``, 18, so the copy is better than the mean for h < 5."""
        w = _windows(horizon=DREAMT, apart=3)
        z = w.z(_Stub(), chunks=O)  # type: ignore[arg-type]
        dyn = build_dynamics(DynamicsConf(d_s=8, residual=True), 1, 1, seed=0)
        rows, curves, picture = dream(dyn, _Fill(), w, z, residual=True)
        h = torch.arange(1, DREAMT + 1, dtype=torch.float64)
        assert curves["copy"] == pytest.approx((h**2).tolist())
        assert curves["mean"] == pytest.approx([18.0] * DREAMT)
        assert curves["dream"] == pytest.approx(curves["copy"])
        assert curves["dream1"] == pytest.approx(curves["copy"])
        assert set(rows) == {
            *(f"{k}{h}" for k in ("dream_img", "dream1_img") for h in (1, 5, 10, 20)),
            *(f"dream_{k}{h}" for k in ("copy", "mean") for h in (1, 5, 10, 20)),
            "dream_lead",
            "dream1_lead",
            "dream_nonfinite",
            "dream1_nonfinite",
        }
        assert rows["dream_img5"] == pytest.approx(25.0)
        assert rows["dream_copy20"] == pytest.approx(400.0)
        assert rows["dream_lead"] == rows["dream1_lead"] == 4
        assert rows["dream_nonfinite"] == 0
        # 3 windows of truth, dream, dream1; at h = 0, 10, 20
        assert picture.shape == (9, 3, 1, 2, 2)
        for i in range(3):
            assert torch.equal(picture[3 * i, 1], w.frame(10)[i])
            for r in (1, 2):
                assert torch.equal(picture[3 * i + r, 2], w.frame(0)[i])

    def test_the_one_frame_dream_starts_from_the_state_0(self):
        """``dream1`` is the open loop from the last observed z and the state
        0; ``dream`` has seen all the observed z (as ``latent_errors``)."""
        w = _windows(horizon=DREAMT)
        z = w.z(_Stub(), chunks=O)  # type: ignore[arg-type]
        dyn = build_dynamics(DynamicsConf(d_s=8, residual=True), 1, 1, seed=0)
        with torch.no_grad():
            dyn.readout.weight.normal_(generator=torch.Generator().manual_seed(0))
            s0 = z.new_zeros(len(z), dyn.step.d_state)
            one = open_loop(dyn, z[:, -1], s0, residual=True, horizon=DREAMT)
        full = torch.cat([z, torch.zeros(len(z), DREAMT, 1)], 1)
        _, all_seen = latent_errors(
            dyn, full, observed=O, horizons=[DREAMT], residual=True
        )
        _, curves, _ = dream(dyn, _Fill(), w, z, residual=True)
        for key, pred in (("dream1", one), ("dream", all_seen)):
            expected = [
                float(((_Fill()(pred[:, h]) - w.frame(h + 1)) ** 2).double().mean())
                for h in range(DREAMT)
            ]
            assert curves[key] == pytest.approx(expected), key
        assert curves["dream"] != pytest.approx(curves["dream1"])


class _Fill(torch.nn.Module):
    """z ``(n, 1)`` to a ``(n, 1, 2, 2)`` image of that value."""

    @override
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z[:, :, None, None].expand(-1, -1, 2, 2)
