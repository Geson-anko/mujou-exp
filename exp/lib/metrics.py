"""Metrics of representations, on tensors."""

import torch

__all__ = ["bootstrap_se", "effective_rank", "probe_r2"]


def effective_rank(z: torch.Tensor) -> float:
    """The exponential of the entropy of the normalized eigenvalues of the
    covariance of ``z`` (``(n, D)``): the squared singular values of the
    centered ``z`` over ``n - 1`` (no ``(D, D)`` matrix: ``D`` = 16384 for the
    flattened tokens of M17, whose eigensolver wanted 8 GiB)."""
    x = z.double()
    x = x - x.mean(0)
    values = torch.linalg.svdvals(x) ** 2 / (len(x) - 1)
    p = values / values.sum()
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def probe_r2(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    ridge: float = 1e-3,
) -> torch.Tensor:
    """The $R^2$ on the test set of each target of the ridge regression from
    ``x`` (``(n, F)``) to ``y`` (``(n, K)``), the features standardized on the
    train set. Returns ``(K,)``."""
    mean, std = train_x.mean(0), train_x.std(0).clamp(min=1e-8)
    a = torch.cat(
        [(train_x - mean) / std, train_x.new_ones(len(train_x), 1)], 1
    ).double()
    b = torch.cat([(test_x - mean) / std, test_x.new_ones(len(test_x), 1)], 1).double()
    eye = torch.eye(a.shape[1], dtype=a.dtype, device=a.device)
    eye[-1, -1] = 0  # no penalty on the intercept
    w = torch.linalg.solve(a.T @ a + ridge * len(a) * eye, a.T @ train_y.double())
    resid = ((b @ w - test_y.double()) ** 2).sum(0)
    total = ((test_y.double() - test_y.double().mean(0)) ** 2).sum(0)
    return (1 - resid / total).float()


def bootstrap_se(
    x: torch.Tensor, *, gen: torch.Generator, resamples: int = 1000
) -> float:
    r"""The bootstrap standard error of the mean of ``x`` ``(n,)`` (one value
    per independent window): the standard deviation of the means of
    ``resamples`` resamples of ``n`` values drawn with replacement by ``gen``
    (on the device of ``x``), in float64.

    For ``n`` values of standard deviation $\sigma$ it estimates
    $\sigma / \sqrt{n}$.
    """
    n = x.shape[0]
    idx = torch.randint(n, (resamples, n), generator=gen, device=x.device)
    means = x.double()[idx].mean(1)
    return float(means.std())
