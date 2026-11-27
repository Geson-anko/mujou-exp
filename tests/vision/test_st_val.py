r"""``exp.vision.st_val``: V18d's validation completion loss.

The oracles: the clips fit their videos, take the lags evenly and their
first stacks uniformly over the places that fit (chi-square); a video whose
frames carry their index reads each stack of a clip to its own frames; the
masks are 8 distinct tubes hiding the same positions at every time and the
6 rows of ``time_masks``, each hiding half the clip; the errors of
``completion_errors`` are ``st_terms``' (the training's) clip errors of the
same masks; the prediction $\bar t_p$ has $L^{\mathrm{rel}} = 1$ exactly on
every side, cut and lag, and $t$ itself 0; ``sigreg_heldin`` of clips whose
times are one stack is that stack's SIGReg.
"""

import itertools

import pytest
import torch

from exp.lib.sigreg import sigreg
from exp.lib.video_io import U8Reader, U8Writer
from exp.vision.jepa import (
    LATENT_DIM,
    build_jepa,
    directions,
    mask_family,
    st_terms,
    time_masks,
)
from exp.vision.st_val import (
    CUTS,
    clip_stacks,
    completion_errors,
    heldin_sigreg,
    rel_losses,
    val_clips,
    val_masks,
)
from exp.vision.time_val import LAGS
from tests.vision.test_jepa import conf_of

F64 = torch.float64


class TestClips:
    def test_every_clip_fits_and_the_lags_are_even(self):
        lengths, span = [2000, 1700], 7
        video, first, n, _ = val_clips(lengths, span)
        size = torch.tensor(lengths)[video]
        assert (first >= 0).all() and (first + 3 * n + span < size).all()
        counts = torch.bincount(torch.searchsorted(torch.tensor(LAGS), n))
        assert counts.tolist() == [40] * (512 % 13) + [39] * (13 - 512 % 13)

    def test_the_places_are_uniform(self):
        lengths, span, lag = [12, 9], 1, 2  # places 12-6-1=5 and 2: seven in all
        video, first, _, _ = val_clips(lengths, span, count=50_000, lags=(lag,), seed=5)
        place = torch.where(video == 0, first, 5 + first)
        got = torch.bincount(place, minlength=7).double()
        assert got.sum() == 50_000 and (place < 7).all()
        want = 50_000 / 7
        chi2 = float(((got - want) ** 2 / want).sum())
        assert chi2 < 22.46  # chi-square of 6 degrees, p = 0.001

    def test_the_same_seed_the_same_clips(self):
        a, b = val_clips([3000], 7)[:3], val_clips([3000], 7)[:3]
        assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))

    def test_a_lag_that_fits_nowhere(self):
        with pytest.raises(ValueError, match="fits in no video"):
            val_clips([100], 7, lags=(40,))

    def test_each_stack_is_its_own_frames(self, tmp_path):
        for name, length in (("a", 60), ("b", 40)):
            with U8Writer(tmp_path / name, (3, 2, 2)) as w:
                for f in range(length):
                    w.append(torch.full((3, 2, 2), f + (100 if name == "b" else 0),
                                        dtype=torch.uint8))  # fmt: skip
        readers = [U8Reader(tmp_path / v) for v in ("a", "b")]
        frames, stride = 3, 2
        video, first, n, _ = val_clips([60, 40], 4, count=40, lags=(1, 3, 9))
        x = clip_stacks(readers, video, first, n, frames=frames, stride=stride)
        got = x[..., 0, 0, 0].long()  # (B, 4, frames)
        base = torch.where(video == 1, 100, 0)
        i, f = torch.arange(4), torch.arange(frames) * stride
        want = (base + first)[:, None, None] + (n[:, None] * i)[:, :, None] + f
        assert torch.equal(got, want)


class TestMasks:
    def test_tubes_then_the_time_side_each_hiding_half(self):
        family = mask_family(8, 4)
        masks = val_masks(family, 2, torch.Generator().manual_seed(0))
        assert masks.shape == (14, 4, 64)
        assert (masks.flatten(1).sum(1) == 128).all()
        tubes = masks[:8]
        assert (tubes == tubes[:, :1]).all()  # the same positions at every time
        assert len({tuple(r.tolist()) for r in tubes[:, 0]}) == 8
        assert torch.equal(masks[8:].all(2), time_masks(4))
        assert torch.equal(masks[8:].any(2), time_masks(4))

    def test_the_cuts_name_the_rows(self):
        hidden = [tuple(r.nonzero()[:, 0].tolist()) for r in time_masks(4)]
        assert hidden == list(itertools.combinations(range(4), 2))
        named = dict(zip(CUTS, hidden, strict=True))
        assert named["past"] == (0, 1) and named["future"] == (2, 3)
        assert named["middle"] == (1, 2) and named["outward"] == (0, 3)
        assert named["interp_past"] == (0, 2) and named["interp_future"] == (1, 3)


class TestLoss:
    def test_the_errors_are_the_trainings(self):
        mods = build_jepa(conf_of(width=64, patch=2), seed=0, token_dim=16,
                          proj="fixed", latent_dim=LATENT_DIM)  # fmt: skip
        g = torch.Generator().manual_seed(1)
        x = torch.randint(256, (3, 4, 2, 3, 16, 16), generator=g, dtype=torch.uint8)
        n = torch.tensor([8, 45, 512])
        masks = val_masks(mask_family(8, 4), 2, torch.Generator().manual_seed(2))
        err, tokens = completion_errors(mods, x, n, masks)
        dirs = directions(8, LATENT_DIM, torch.Generator().manual_seed(3))
        for m, mask in enumerate(masks):
            with torch.no_grad():
                want = st_terms(
                    mods, x, n, mask[None].expand(3, -1, -1), dirs, 0.5, "mix"
                )
            per = want["clip_err"].double() * 128 * 16  # the mean over m d
            torch.testing.assert_close(err[:, m], per, rtol=1e-5, atol=0)
            assert torch.equal(tokens, want["tokens"].unflatten(0, (3, 4)))

    def test_the_mean_predicts_one_and_the_target_zero(self):
        g = torch.Generator().manual_seed(0)
        b, t, d = 26, 64, 3
        tokens = torch.randn(b, 4, t, d, generator=g, dtype=F64)
        masks = val_masks(mask_family(8, 4), 2, torch.Generator().manual_seed(4))
        n = torch.tensor(LAGS * 2)
        tbar = tokens.flatten(0, 1).mean(0)
        dev = (tokens - tbar).square().sum(-1)
        err = torch.einsum("bip,mip->bm", dev, masks.double())  # predicting tbar
        got = rel_losses(err, tokens, masks, n)
        ones = [got["now"], got["time"], got["st"], *got["cuts"].values(),
                *got["lags"]["now"], *got["lags"]["time"]]  # fmt: skip
        assert ones == pytest.approx([1.0] * len(ones), rel=1e-12)
        assert got["lags"]["n"] == list(LAGS)
        zero = rel_losses(torch.zeros_like(err), tokens, masks, n)
        assert zero["now"] == zero["time"] == zero["st"] == 0.0

    def test_a_side_reads_its_own_masks(self):
        g = torch.Generator().manual_seed(0)
        tokens = torch.randn(13, 4, 64, 2, generator=g, dtype=F64)
        masks = val_masks(mask_family(8, 4), 2, torch.Generator().manual_seed(4))
        err = torch.zeros(13, 14, dtype=F64)
        err[:, 8 + CUTS.index("future")] = 1.0
        got = rel_losses(err, tokens, masks, torch.full((13,), 8))
        assert got["now"] == 0.0 and got["time"] > 0
        assert [k for k, v in got["cuts"].items() if v > 0] == ["future"]


class TestHeldinSigreg:
    def test_clips_of_one_stack_are_its_sigreg(self):
        g = torch.Generator().manual_seed(0)
        one = torch.randn(64, 5, generator=g, dtype=F64)
        clips = [one[:, None].expand(-1, 4, -1)]
        dirs = directions(16, 5, torch.Generator().manual_seed(1)).double()
        got = heldin_sigreg(lambda z: z, clips, dirs)
        want = float(sigreg(one, one[:0], dirs))
        assert got["sigreg_heldin"] == pytest.approx(want, rel=1e-12)
        assert got["sigreg_heldin_times"] == pytest.approx([want] * 4, rel=1e-12)
