"""Metrics of representations, on tensors."""

import math
from typing import Any

import torch
from torch import nn

__all__ = [
    "PROBE_RIDGES",
    "block_bootstrap",
    "block_counts",
    "bootstrap_se",
    "contiguous_blocks",
    "effective_rank",
    "identification",
    "integrated_time",
    "max_t_critical",
    "pair_wins",
    "probe_accuracy",
    "probe_r2",
    "ratio_resamples",
    "scene_bootstrap",
    "scene_resamples",
    "swap",
    "video_jackknife",
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


def contiguous_blocks(segments: torch.Tensor, length: int) -> torch.Tensor:
    r"""The block of each item ``(N,)`` long: the items of each segment (``segments``
    ``(N,)``, the items in order, a segment's items consecutive) cut into
    blocks of ``length`` consecutive items, the remainder of fewer than
    ``length`` at a segment's end joined to the block before it (a segment
    shorter than ``length`` is one block). Blocks are numbered from 0 in
    order."""
    out = torch.empty_like(segments, dtype=torch.long)
    start, nxt = 0, 0
    n = len(segments)
    while start < n:
        end = start
        while end < n and segments[end] == segments[start]:
            end += 1
        count = max((end - start) // length, 1)
        within = torch.arange(end - start) // length
        out[start:end] = nxt + within.clamp(max=count - 1)
        nxt += count
        start = end
    return out


def block_bootstrap(
    e: torch.Tensor,
    v: torch.Tensor,
    blocks: torch.Tensor,
    contrasts: torch.Tensor,
    n: int,
    gen: torch.Generator,
) -> torch.Tensor:
    r"""The block bootstrap's standard errors ``(Q,)`` float64 of the
    contrasts of the pooled measures $m_c = 1 - \sum_i e_{ci} / \sum_i
    v_i$ of $C$ codes: ``e`` ``(C, N)``, ``v`` ``(N,)``, ``blocks`` ``(N,)``
    each item's block (``contiguous_blocks``), ``contrasts`` ``(Q, C)`` the
    coefficients $a_{qc}$ of $\sum_c a_{qc} m_c$.

    Each of the ``n`` resamples draws as many blocks as there are, uniformly
    with replacement (by ``gen``, a CPU generator), the same draws for every
    code (so the contrasts are paired); a resample's $m^*_c$ is made of the
    drawn items' sums. A block much longer than the items' correlation
    makes the blocks' sums nearly independent (Künsch 1989)."""
    ids, inverse = torch.unique(blocks.cpu(), return_inverse=True)
    b = len(ids)
    ed, vd = e.double().cpu(), v.double().cpu()
    e_b = torch.zeros(ed.shape[0], b, dtype=torch.float64).index_add_(1, inverse, ed)
    v_b = torch.zeros(b, dtype=torch.float64).index_add_(0, inverse, vd)
    draws = torch.randint(b, (n, b), generator=gen)
    counts = torch.zeros(n, b, dtype=torch.float64)
    counts.scatter_add_(1, draws, torch.ones(n, b, dtype=torch.float64))
    m = 1 - (counts @ e_b.T) / (counts @ v_b)[:, None]  # (n, C)
    return (m @ contrasts.double().cpu().T).std(0)


def block_counts(n_blocks: int, n: int, gen: torch.Generator) -> torch.Tensor:
    """How many times each of ``n_blocks`` blocks is drawn ``(n, B)``
    float64 in each of ``n`` resamples of ``n_blocks`` blocks drawn
    uniformly with replacement by ``gen`` (a CPU generator): the draws of
    ``block_bootstrap``, so one seed pairs the resamples of codes computed
    apart."""
    draws = torch.randint(n_blocks, (n, n_blocks), generator=gen)
    counts = torch.zeros(n, n_blocks, dtype=torch.float64)
    return counts.scatter_add_(1, draws, torch.ones(n, n_blocks, dtype=torch.float64))


def ratio_resamples(
    num_b: torch.Tensor, den_b: torch.Tensor, counts: torch.Tensor
) -> torch.Tensor:
    r"""The resampled ratios $\sum_b c_b\,\mathrm{num}_{cb} / \sum_b c_b\,
    \mathrm{den}_{cb}$ ``(n, C)`` float64 of $C$ codes' block sums ``num_b``
    and ``den_b`` ``(C, B)`` (each code its own denominator, unlike
    ``block_bootstrap``) and the ``counts`` ``(n, B)`` of ``block_counts``;
    a paired difference of two codes is the difference of their columns."""
    c = counts.double()
    return (c @ num_b.double().T) / (c @ den_b.double().T)


def max_t_critical(
    resamples: torch.Tensor,
    estimate: torch.Tensor,
    se: torch.Tensor,
    level: float = 0.95,
) -> float:
    r"""The critical value $c$ of the simultaneous intervals $\hat	heta_l
        \pm c\,\mathrm{SE}_l$ over $L$ values: the ``level`` quantile of $\max_l
        \lvert 	heta^*_l - \hat	heta_l
    vert / \mathrm{SE}_l$ over the
        ``resamples`` ``(n, L)`` of the ``estimate`` ``(L,)`` of standard errors
        ``se`` ``(L,)``. Of one value, the quantile of $\lvert t
    vert$."""
    t = (resamples.double() - estimate.double()).abs() / se.double()
    return float(torch.quantile(t.max(1).values, level))


def _block_draws(
    blocks: torch.Tensor, video: torch.Tensor, n: int, gen: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The items' block positions ``(N,)``, the blocks' videos ``(B,)`` and
    the draws' counts ``(n, B)`` float64: each video's blocks drawn as many
    times as it has, uniformly with replacement (videos in order)."""
    _, inverse = torch.unique(blocks.cpu(), return_inverse=True)
    nb = int(inverse.max()) + 1
    first = torch.full((nb,), len(inverse), dtype=torch.long)
    first.scatter_reduce_(0, inverse, torch.arange(len(inverse)), reduce="amin")
    of_video = video.cpu()[first]
    counts = torch.zeros(n, nb, dtype=torch.float64)
    for v in torch.unique(of_video).tolist():
        own = (of_video == v).nonzero().flatten()
        draws = own[torch.randint(len(own), (n, len(own)), generator=gen)]
        counts.scatter_add_(1, draws, torch.ones(n, len(own), dtype=torch.float64))
    return inverse, of_video, counts


def scene_resamples(
    preds: torch.Tensor,
    y: torch.Tensor,
    video: torch.Tensor,
    blocks: torch.Tensor,
    n: int,
    gen: torch.Generator,
    *,
    measure: str = "m",
    min_count: int = 1,
) -> torch.Tensor:
    r"""The stratified block bootstrap's resampled measures ``(n, C)``
    float64 of $C$ codes' scene measures (``readout.scene_m``, or BA with
    ``measure`` ``"ba"``; V18c's main measure, 3rd remake): ``preds``
    ``(C, N)`` the predictions of the labels ``y`` ``(N,)`` in the videos
    ``video`` ``(N,)``, or ``(C, P, N)`` of $P$ parts of their own labels
    ``(P, N)`` whose measures are averaged (E1f's quarters); a label -1
    leaves its row out. ``blocks`` ``(N,)`` each item's block, within one
    video (``contiguous_blocks`` of the videos).

    Each of the ``n`` resamples draws each video's blocks as many times as
    it has, uniformly with replacement (by ``gen``, a CPU generator; the
    videos in order), the same draws for every code and part, and for every
    call of the same ``blocks``, ``video``, ``n`` and seed whatever the
    labels (so two subsets' resamples pair); the confusion, $n_k$, $P(v
    \mid y = k)$ and $P(\hat y = k \mid v)$ are made again of the drawn
    rows. A class of fewer than ``min_count`` rows in a resample (at least
    absent) is left out of its sums, as ``readout.scene_m`` does."""
    p3 = preds if preds.dim() == 3 else preds[:, None]
    y2 = y if y.dim() == 2 else y[None]
    inverse, of_video, counts = _block_draws(blocks, video, n, gen)
    nb = counts.shape[1]
    vids = torch.unique(of_video)
    onehot_bv = (of_video[:, None] == vids[None]).double()  # (B, V)
    per_part = []
    for part in range(y2.shape[0]):
        t = y2[part].cpu().long()
        valid = t >= 0
        k = int(t.max()) + 1
        pr = p3[:, part].cpu().long()
        at = inverse[valid] * k
        ones = torch.ones(int(valid.sum()), dtype=torch.float64)
        a = torch.zeros(nb * k, dtype=torch.float64).index_add_(0, at + t[valid], ones)
        a = a.view(nb, k)
        h = torch.zeros(len(pr), nb * k, dtype=torch.float64)
        q = torch.zeros(len(pr), nb * k, dtype=torch.float64)
        for c in range(len(pr)):
            pc = pr[c][valid]
            inside = (pc >= 0) & (pc < k)
            h[c].index_add_(0, at + t[valid], (pc == t[valid]).double())
            q[c].index_add_(0, at + pc.clamp(0, k - 1), inside.double())
        h, q = h.view(-1, nb, k), q.view(-1, nb, k)
        rows_b = a.sum(1)
        nvk = torch.einsum("nb,bv,bk->nvk", counts, onehot_bv, a)[:, None]
        hit = torch.einsum("nb,bv,cbk->ncvk", counts, onehot_bv, h)
        pred_k = torch.einsum("nb,bv,cbk->ncvk", counts, onehot_bv, q)
        rows = torch.einsum("nb,bv,b->nv", counts, onehot_bv, rows_b)[:, None, :, None]
        nk = nvk.sum(2)  # (n, 1, K)
        kept = nk >= max(min_count, 1)
        kk = kept.sum(2)  # (n, 1)
        if measure == "ba":
            term = hit.sum(2)
        else:
            term = ((hit * rows - nvk * pred_k) / rows.clamp(min=1)).sum(2)
        val = torch.where(kept, term / nk.clamp(min=1), torch.zeros_like(term))
        per_part.append(val.sum(2) / kk)  # (n, C)
    return torch.stack(per_part).mean(0)


def scene_bootstrap(
    preds: torch.Tensor,
    y: torch.Tensor,
    video: torch.Tensor,
    blocks: torch.Tensor,
    contrasts: torch.Tensor,
    n: int,
    gen: torch.Generator,
    *,
    measure: str = "m",
) -> torch.Tensor:
    """The standard errors ``(Q,)`` float64 of the contrasts ``(Q, C)`` of
    the measures of ``scene_resamples`` (its arguments)."""
    m = scene_resamples(preds, y, video, blocks, n, gen, measure=measure)
    return (m @ contrasts.double().cpu().T).std(0)


def video_jackknife(theta: torch.Tensor) -> tuple[torch.Tensor, float]:
    r"""The values $\theta_{(u)}$ ``(U,)`` (each without video $u$) as they
    are and the jackknife's standard error $\sqrt{\frac{U - 1}{U} \sum_u
    (\theta_{(u)} - \bar\theta)^2}$ (listed only; V18c's main measure,
    3rd remake)."""
    t = theta.double()
    u = len(t)
    return theta, math.sqrt((u - 1) / u * float((t - t.mean()).square().sum()))


def integrated_time(x: torch.Tensor) -> float:
    r"""The integrated correlation time $\tau = 1 + 2 \sum_{\ell = 1}^{\ell^*}
    \rho(\ell)$ of the series ``x`` ``(T, K)`` (time first): $\rho(\ell)$
    the mean over the $K$ coordinates of their autocorrelations $\sum_t
    \tilde x_t \tilde x_{t + \ell} / \sum_t \tilde x_t^2$ ($\tilde x$
    less its mean), $\ell^*$ the lag before $\rho$ is first at most 0. An
    AR(1) of coefficient $\phi$ has $\tau = (1 + \phi) / (1 - \phi)$."""
    xc = x.double() - x.double().mean(0)
    t = len(xc)
    den = xc.square().sum(0)
    tau = 1.0
    for lag in range(1, t):
        rho = float(((xc[:-lag] * xc[lag:]).sum(0) / den).mean())
        if rho <= 0:
            break
        tau += 2 * rho
    return tau


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


def pair_wins(
    pred: torch.Tensor,
    real: torch.Tensor,
    *,
    truth: torch.Tensor | None = None,
    among: torch.Tensor | None = None,
    metric: str = "cos",
) -> tuple[float, torch.Tensor, torch.Tensor]:
    r"""The pairs' win rate of each query: how often ``pred`` $e_n$ ``(n,
    F)`` (taken as it is: the caller centres it) is nearer its own real row
    than another candidate, a tie a half.

    The candidates of query $n$: the rows ``among[n]`` (``among`` ``(n,
    c)``, by default all $m$) of ``real`` ``(m, F)``, its own
    ``real[truth[n]]`` (``truth`` ``(n,)``, by default $n$; each row of
    ``among`` holds it); or ``real`` ``(n, c, F)``, each query's own
    candidates, its own at column ``truth[n]`` (by default 0). The
    similarity $s(n, l)$ is ``"cos"`` $\langle e_n, r_l \rangle / \lVert
    r_l \rVert$ (a row of no length gives 0; $e_n$'s length orders nothing)
    or ``"l2"`` $-\lVert e_n - r_l \rVert^2$ (computed as $2 \langle e_n,
    r_l \rangle - \lVert r_l \rVert^2$, the query's own $\lVert e_n
    \rVert^2$ dropped). With $o$ its own,

    $$A_n = \frac{1}{c - 1} \sum_{l \ne o} \left( [s(n, o) > s(n, l)] +
    \tfrac12 [s(n, o) = s(n, l)] \right),$$

    in float64. Returns the mean of $A_n$, the values $A_n$ ``(n,)`` and the
    candidates tied with its own ``(n,)``. $A_n = 1$ exactly where no other
    candidate is as near (``identification``'s hit: a tie is not one); one
    prediction of no direction ties with all, $A$ = 1/2.

    Raises:
        ValueError: ``metric`` is not ``"cos"`` or ``"l2"``.
    """
    if metric not in ("cos", "l2"):
        raise ValueError(f"the metric is 'cos' or 'l2', not {metric!r}")
    e, r = pred.double(), real.double()
    n = e.shape[0]
    if r.dim() == 3:  # (n, c, F): each query's own candidates
        dots = torch.einsum("nf,ncf->nc", e, r)
        sq = r.square().sum(2)  # (n, c)
        at = torch.zeros(n, dtype=torch.long) if truth is None else truth
    else:
        dots, sq = e @ r.T, r.square().sum(1)[None].expand(n, -1)  # (n, m)
        own = torch.arange(n) if truth is None else truth
        at = own
        if among is not None:
            dots, sq = dots.gather(1, among), sq.gather(1, among)
            found = among == own[:, None]
            assert bool(found.any(1).all()), "each row of among holds its own"
            at = found.int().argmax(1)
    if metric == "cos":
        sims = dots / sq.sqrt().clamp(min=1e-300)
    else:
        sims = 2 * dots - sq
    at = at.to(sims.device)
    own_sim = sims.gather(1, at[:, None])  # (n, 1)
    other = torch.ones_like(sims, dtype=torch.bool)
    other.scatter_(1, at[:, None], False)
    wins = (sims < own_sim).double() + 0.5 * (sims == own_sim).double()
    each = (wins * other).sum(1) / (sims.shape[1] - 1)
    ties = ((sims == own_sim) & other).sum(1)
    return float(each.mean()), each, ties


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
