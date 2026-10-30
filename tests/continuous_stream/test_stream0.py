import math

import torch

from experiments.continuous_stream.stream0 import LinearStream

FRAME = 1 / 30


def _means(stream: LinearStream, n: int) -> torch.Tensor:
    out = []
    for _ in range(n):
        stream.next()
        out.append(stream.mean)
    return torch.stack(out)  # (n, d)


def _autocorrelation(m: torch.Tensor, lag: int) -> float:
    m = m - m.mean(0)
    return float((m[:-lag] * m[lag:]).mean() / (m * m).mean())


class TestLinearStream:
    def test_the_mean_decorrelates_as_exp_of_minus_lag_over_tau(self):
        stream = LinearStream(d=32, tau_x=1.0, frame=FRAME, noise=0.1, seed=0)
        m = _means(stream, 20_000)
        for lag in (1, 10, 30):
            want = math.exp(-lag * FRAME / 1.0)
            assert abs(_autocorrelation(m, lag) - want) < 0.05, lag

    def test_the_mean_keeps_its_stationary_variance(self):
        stream = LinearStream(d=32, tau_x=1.0, frame=FRAME, noise=0.1, seed=0)
        var = float(_means(stream, 20_000).var())
        assert abs(var - 1.0) < 0.1

    def test_iid_has_no_correlation_and_the_same_marginal(self):
        stream = LinearStream(d=32, tau_x=1.0, frame=FRAME, noise=0.1, iid=True, seed=0)
        m = _means(stream, 5_000)
        assert abs(_autocorrelation(m, 1)) < 0.02
        assert abs(float(m.var()) - 1.0) < 0.05

    def test_y_is_w_star_x_plus_noise(self):
        stream = LinearStream(d=8, tau_x=1.0, frame=FRAME, noise=0.1, seed=0)
        residuals = []
        for _ in range(5_000):
            x, y = stream.next()
            residuals.append(float(y[0] - x @ stream.w))
        assert abs(torch.tensor(residuals).std().item() - 0.1) < 0.01

    def test_the_same_seed_gives_the_same_frames(self):
        a = LinearStream(d=4, tau_x=1.0, frame=FRAME, noise=0.1, seed=3)
        b = LinearStream(d=4, tau_x=1.0, frame=FRAME, noise=0.1, seed=3)
        for _ in range(10):
            (xa, ya), (xb, yb) = a.next(), b.next()
            assert torch.equal(xa, xb) and torch.equal(ya, yb)
