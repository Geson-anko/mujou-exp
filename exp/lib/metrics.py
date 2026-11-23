"""Metrics of representations, on tensors."""

import torch
from torch import nn

__all__ = [
    "PROBE_RIDGES",
    "bootstrap_se",
    "effective_rank",
    "probe_accuracy",
    "probe_r2",
]

PROBE_RIDGES = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)  # of probe_accuracy, times n


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


def probe_accuracy(
    f_train: torch.Tensor,
    y_train: torch.Tensor,
    f_val: torch.Tensor,
    y_val: torch.Tensor,
    classes: int,
) -> float:
    """The accuracy on val of the ridge regression of the one-hot labels from
    the standardized features (with a bias), its penalty (``PROBE_RIDGES``
    times the samples) chosen by the accuracy on the last fifth of train,
    then fit on all of train (M17's linear probe of Imagenette)."""
    mu, sd = f_train.mean(0), f_train.std(0).clamp(min=1e-8)

    def design(f: torch.Tensor) -> torch.Tensor:
        z = (f.double() - mu) / sd
        return torch.cat([z, torch.ones(len(z), 1, dtype=z.dtype)], 1)

    def fit(x: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
        t = nn.functional.one_hot(y, classes).double()
        a = x.T @ x + alpha * len(x) * torch.eye(x.shape[1], dtype=x.dtype)
        return torch.linalg.solve(a, x.T @ t)

    x, xv = design(f_train), design(f_val)
    cut = len(x) * 4 // 5
    perm = torch.randperm(len(x), generator=torch.Generator().manual_seed(0))
    fit_i, hold_i = perm[:cut], perm[cut:]
    scores = []
    for alpha in PROBE_RIDGES:
        w = fit(x[fit_i], y_train[fit_i], alpha)
        acc = (x[hold_i] @ w).argmax(1).eq(y_train[hold_i]).double().mean()
        scores.append(float(acc))
    best = PROBE_RIDGES[max(range(len(PROBE_RIDGES)), key=lambda i: scores[i])]
    w = fit(x, y_train, best)
    return float((xv @ w).argmax(1).eq(y_val).double().mean())


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
