r"""``exp.lib.inception``: the preprocessing by hand, the net's features
(random weights; FID's weights only where the file is, never downloaded),
the local features (2 x 2 cells of the last maps, whose mean is pool3),
and the Fréchet distance against its closed forms (1-d, diagonal, rotated,
swapped, equal, of low rank)."""

import math

import pytest
import torch
from torch.nn import functional as F

from exp.lib.inception import (
    FIDInception,
    fid,
    frechet_distance,
    load_fid_inception,
    preprocess,
    sha256,
    weights_path,
)


def images(b: int, side: int = 64, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(256, (b, 3, side, side), generator=g, dtype=torch.uint8)


class TestPreprocess:
    def test_the_bilinear_enlargement_and_2x_minus_1_by_hand(self):
        x = images(2, side=8)
        want = F.interpolate(
            x.float() / 255, size=(299, 299), mode="bilinear", align_corners=False
        )
        torch.testing.assert_close(preprocess(x), 2 * want - 1, rtol=0, atol=1e-6)

    def test_a_frame_that_is_not_square_is_stretched(self):
        x = torch.full((1, 3, 16, 32), 255, dtype=torch.uint8)
        assert preprocess(x).shape == (1, 3, 299, 299)
        assert float(preprocess(x).min()) == pytest.approx(1.0)


class TestFIDInception:
    def test_the_features_of_random_weights_are_2048_not_negative(self):
        torch.manual_seed(0)
        f = FIDInception()(images(2, side=32))
        assert f.shape == (2, 2048) and float(f.min()) >= 0

    def test_the_local_features_average_to_pool3(self):
        torch.manual_seed(0)
        net = FIDInception()
        x = images(2, side=32)
        local = net.local(x).reshape(2, 2048, 4)
        assert local.shape == (2, 2048, 4)
        torch.testing.assert_close(local.mean(2), net(x), rtol=1e-5, atol=1e-6)

    def test_fids_weights_where_the_file_is(self):
        path = weights_path()
        if not path.is_file():
            pytest.skip(f"no FID weights at {path}")
        assert sha256(path).startswith("6726825d")
        net = load_fid_inception(path)
        x = images(5)
        f = net(x)
        assert f.shape == (5, 2048) and float(f.min()) >= 0
        apart = torch.cat([net(x[:2]), net(x[2:])])
        torch.testing.assert_close(apart, f, rtol=0, atol=1e-5)

    def test_a_file_of_another_digest_is_refused(self, tmp_path):
        path = tmp_path / "w.pth"
        torch.save({}, path)
        with pytest.raises(ValueError, match="sha256"):
            load_fid_inception(path)


def spd(d: int, seed: int) -> torch.Tensor:
    a = torch.randn(d, d, generator=torch.Generator().manual_seed(seed)).double()
    return a @ a.T / d + 0.1 * torch.eye(d, dtype=torch.float64)


class TestFrechetDistance:
    def test_1d_is_the_squared_differences_of_the_means_and_the_sds(self):
        d = frechet_distance(
            torch.tensor([1.0]), torch.tensor([[4.0]]),
            torch.tensor([-0.5]), torch.tensor([[0.25]]),
        )  # fmt: skip
        assert d == pytest.approx(1.5**2 + (2 - 0.5) ** 2)

    def test_diagonal_covariances_are_the_sum_of_squared_root_differences(self):
        a, b = torch.tensor([1.0, 2.0, 9.0]), torch.tensor([4.0, 0.5, 1.0])
        m1, m2 = torch.tensor([0.0, 1.0, 2.0]), torch.tensor([1.0, 1.0, 0.0])
        want = float((m1 - m2).square().sum() + (a.sqrt() - b.sqrt()).square().sum())
        d = frechet_distance(m1, torch.diag(a), m2, torch.diag(b))
        assert d == pytest.approx(want, rel=1e-12)

    def test_a_common_rotation_and_a_swap_change_nothing(self):
        s1, s2 = spd(5, 0), spd(5, 1)
        g = torch.Generator().manual_seed(3)
        m1, m2 = (torch.randn(5, generator=g, dtype=torch.float64) for _ in range(2))
        q, _ = torch.linalg.qr(torch.randn(5, 5, generator=g, dtype=torch.float64))
        d = frechet_distance(m1, s1, m2, s2)
        assert frechet_distance(q @ m1, q @ s1 @ q.T, q @ m2, q @ s2 @ q.T) == (
            pytest.approx(d, rel=1e-9)
        )
        assert frechet_distance(m2, s2, m1, s1) == pytest.approx(d, rel=1e-9)
        assert frechet_distance(m1, s1, m1, s1) == pytest.approx(0, abs=1e-9)

    def test_features_fewer_than_their_dimensions_give_a_finite_value(self):
        g = torch.Generator().manual_seed(2)
        f1, f2 = torch.randn(10, 40, generator=g), torch.randn(12, 40, generator=g)
        d = fid(f1, f2)
        assert math.isfinite(d) and d >= 0
        assert fid(f1, f1) == pytest.approx(0, abs=1e-6)
