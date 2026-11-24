"""Metrics of representations, on tensors."""

import math
from typing import Any

import torch
from torch import nn

__all__ = [
    "PROBE_RIDGES",
    "bootstrap_se",
    "effective_rank",
    "identification",
    "probe_accuracy",
    "probe_r2",
    "swap",
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


def _unit(x: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    """The rows of ``x`` less ``center``, of length 1, in float64."""
    u = x.double() - center.double()
    return u / torch.linalg.vector_norm(u, dim=-1, keepdim=True).clamp(min=1e-300)


def identification(
    pred: torch.Tensor,
    real: torch.Tensor,
    *,
    center: torch.Tensor,
    truth: torch.Tensor | None = None,
    among: torch.Tensor | None = None,
    ks: tuple[int, ...] = (1, 5),
) -> dict[str, Any]:
    r"""How often the features ``pred`` ``(n, K, F)`` ($g_{ij}$, $K$ of a
    query $i$; or ``(n, F)`` for $K$ = 1) pick their own real features out of
    ``real`` ``(m, F)`` ($r_l$) by the nearest cosine, both less ``center``
    ``(F,)``.

    Query $i$'s own is ``real[truth[i]]`` (``truth`` ``(n,)``, by default
    $i$) among the rows ``among[i]`` (``among`` ``(n, c)``, by default all
    $m$; each row holds ``truth[i]``). A tie is not a hit: a row as near as
    its own ranks before it, so a prediction at ``center`` (of no direction)
    finds none, whatever the order of ``among[i]``, and one prediction for
    all the queries finds exactly one of them where the real rows are
    distinct. With $\hat l_{ij} = \arg\max_l
    \cos(g_{ij} - \text{center}, r_l - \text{center})$: ``top1`` $=
    \frac{1}{nK} \sum_{i,j} [\hat l_{ij} = \text{truth}_i]$; ``top<k>`` for
    each $k$ of ``ks``, the share of $g_{ij}$ whose own is among the $k$
    nearest; ``hits`` the list of $h_i = \frac1K \sum_j [\hat l_{ij} =
    \text{truth}_i]$; ``top1_se`` $= \mathrm{sd}_i(h_i) / \sqrt n$; and,
    without ``among``, ``top1_adj``, the share of the misses whose pick is a
    neighbour $\lvert \hat l_{ij} - \text{truth}_i \rvert = 1$ (None
    without a miss), and ``pick_max``, the largest share of the $nK$ picks
    $\hat l_{ij}$ on one real row (a hub: a common shift of the predictions
    draws the picks to the rows along it; $1/m$ if every row is picked
    alike). Chance is $1/m$ (or $1/c$) for ``top1``.
    """
    g = pred.reshape(pred.shape[0], -1, pred.shape[-1])  # (n, K, F)
    n, k_samples, _ = g.shape
    gu, ru = _unit(g, center), _unit(real, center)
    own = torch.arange(n, device=real.device) if truth is None else truth
    sims = gu @ ru.T  # (n, K, m)
    at = own  # the column of each query's own in sims
    if among is not None:
        sims = sims.gather(2, among[:, None, :].expand(-1, k_samples, -1))
        at = (among == own[:, None]).int().argmax(1)
    own_sim = sims.gather(2, at[:, None, None].expand(-1, k_samples, 1))
    # the place of its own in a descending sort: the nearer rows, and the
    # other rows as near before it (a tie is not a hit)
    cols = torch.arange(sims.shape[2], device=sims.device)
    ties = (sims == own_sim) & (cols != at[:, None, None])
    rank = (sims > own_sim).sum(2) + ties.sum(2)  # (n, K)
    hits = (rank == 0).double().mean(1)  # (n,)
    out: dict[str, Any] = {
        f"top{k}": float((rank < k).double().mean()) for k in sorted({1, *ks})
    }
    out["top1_se"] = float(hits.std(unbiased=True)) / math.sqrt(n) if n > 1 else 0.0
    out["hits"] = hits.tolist()
    if among is None:
        picks = sims.argmax(2)  # (n, K)
        miss = rank > 0
        adj = (picks - own[:, None]).abs() == 1
        misses = int(miss.sum())
        out["top1_adj"] = float((adj & miss).sum()) / misses if misses else None
        counts = torch.bincount(picks.flatten(), minlength=real.shape[0])
        out["pick_max"] = float(counts.max()) / picks.numel()
    return out


def swap(
    own: torch.Tensor, other: torch.Tensor, real: torch.Tensor
) -> tuple[float, torch.Tensor]:
    r"""The two-way choice of each row $i$: whether ``own`` $a_i$ points
    more to ``real`` $t_i$ than ``other`` $b_i$ does (all ``(n, F)``, taken
    as they are: the caller centres them), $(a_i - b_i) \cdot t_i > 0$: 1,
    0, or 1/2 on a tie. The mean over the rows and the values ``(n,)``
    float64.

    Equal rows $a_i = b_i$ tie (exactly 1/2 for a choice that cannot tell the
    two apart). Where $a_i - b_i$ is independent of $t_i$ and symmetric about
    0, the expectation is 1/2 whatever the scales of $a$ and $b$: a common
    part of the two cancels, and so does neither's size, unlike the cosine
    or the distance, which favour the less noisy one."""
    d = ((own.double() - other.double()) * real.double()).sum(1)
    v = (d > 0).double() + 0.5 * (d == 0).double()
    return float(v.mean()), v
