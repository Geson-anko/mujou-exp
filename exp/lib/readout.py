r"""A centred ridge readout from codes to targets (V18c's E1, E1e, E2;
``exp/vision/plans/jepa.md``, 「未観測の部分: 読み出し」).

With $X$ ``(n, D)`` and $Y$ ``(n, K)`` the fit set less their means
$\bar x$, $\bar y$: $B = (X^\top X + \alpha I)^{-1} X^\top Y$ in float64,
$\alpha = c \cdot \mathrm{tr}(X^\top X) / D$, and $\hat y = (x - \bar x) B +
\bar y$. ``r2`` is the mean over the $K$ coordinates of $1 - \sum_i (\hat
y_{ik} - y_{ik})^2 / \sum_i (y_{ik} - \langle y_k \rangle)^2$ (the
evaluated set's mean). ``select_ridge`` takes the $c$ of the best ``r2`` on
a calibration set (the first on a tie), or of another ``score``.

V18c's main measure (``plans/jepa.md``, 「V18c: 主の指標の作り直し」):
``centered_r2`` pools the error over the coordinates after centring both
the prediction and the target on the evaluated set $E$, $\tilde p_i =
\hat y_i - \langle \hat y \rangle_E$, $\tilde y_i = y_i - \langle y
\rangle_E$:

$$m_E = 1 - \frac{\sum_{i \in E} \lVert \tilde p_i - \tilde y_i
\rVert^2}{\sum_{i \in E} \lVert \tilde y_i \rVert^2},$$

exactly 0 for a constant prediction, unchanged by a shift common to the
set. ``rank_bound`` is its ceiling over predictions whose centred rows are
of rank at most $r$: with $\mu_1 \ge \mu_2 \ge \dots$ the eigenvalues
of the covariance of $\tilde Y$, Eckart-Young gives $\lVert \tilde Y -
\tilde P \rVert_F^2 \ge \lvert E \rvert \sum_{j > r} \mu_j$, so $m_E
\le b_r(E) = \sum_{j \le r} \mu_j / \sum_j \mu_j$, exactly on each set.

V18c's main measure of the scene categories (``plans/jepa.md``, 「V18c: 主の
指標（場面のカテゴリ）」, 「読み出し」「採点」): a code to $K$ classes by a
ridge weighted by class, $w_i = 1 / (K n_{y_i})$ (``class_weights``; the
weighted least squares is the linear stand-in for the posterior's arg max
under a uniform prior, which maximizes the balanced accuracy):
``fit_weighted_ridge`` centres on the weighted means and solves $B = (X_c^\top
W X_c + \alpha I)^{-1} X_c^\top W Y_c$, $\alpha = c\,\mathrm{tr}(X_c^\top W
X_c) / D$. The prediction is the arg max of the outputs (the first on a
tie). ``nested_cv`` takes the videos as units: each video $v$ is scored by
the readout fit on the others, its scale chosen among ``SCENE_SCALES`` and
``CONST`` (the constant prediction, class 0: $c = \infty$, its outputs 0) by
the best $m$ of the inner leave-one-video-out on those others (a tie to the
larger scale). Each fit standardizes the codes by its fit rows' mean and
(population) sd (an sd of at most $10^{-12}$ taken as 1); the weighted
centring cancels the mean, so a fit is that of $x / s$.

The scores of predictions $\hat y$ of labels $y$ in videos $v$, of $n_{vk}$
labels $k$ in video $v$, $n_k = \sum_v n_{vk}$ and $N_v$ the rows of $v$:
$\mathrm{BA} = \frac1K \sum_k \#(\hat y = k, y = k) / n_k$
(``balanced_accuracy``), $B = \frac1K \sum_k \sum_v (n_{vk} / n_k)\,
P(\hat y = k \mid v)$ (``video_chance``) and $m = \mathrm{BA} - B$
(``scene_m``), summed as $\frac1K \sum_k \sum_v (h_{vk} N_v - n_{vk}
c_{vk}) / (N_v n_k)$ ($h$ the hits, $c$ the predictions of $k$), integers
over integers: a prediction constant within each video gives exactly 0. If
$\hat y$ is independent of $y$ within each video, $\mathbb E\,h_{vk} =
n_{vk} P(\hat y = k \mid v)$ and $m$ is 0 in expectation. A class of
fewer than ``min_count`` rows is left out of the sums, $K$ read as the
classes kept (NaN if none is).
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from exp.lib.linear_dyn import ALPHA_SCALES

__all__ = [
    "CONST",
    "SCENE_SCALES",
    "NestedResult",
    "Ridge",
    "balanced_accuracy",
    "centered_r2",
    "centered_terms",
    "class_auc",
    "class_weights",
    "fit_ridge",
    "fit_weighted_ridge",
    "nested_cv",
    "nested_cv_jackknife",
    "predict",
    "r2",
    "rank_bound",
    "scene_m",
    "select_ridge",
    "video_chance",
]

SCENE_SCALES = tuple(10.0 ** (e / 2) for e in range(-12, 7))  # 1e-6 .. 1e3
CONST = math.inf  # the constant prediction (class 0) as the largest scale


@dataclass(frozen=True)
class Ridge:
    """A fit readout: the means ``x_mean`` ``(D,)``, ``y_mean`` ``(K,)``,
    the matrix ``b`` ``(D, K)`` (float64), its ``alpha`` and the ``scale``
    it was fit at (``fit_ridge``)."""

    x_mean: Tensor
    y_mean: Tensor
    b: Tensor
    alpha: float
    scale: float


def fit_ridge(x: Tensor, y: Tensor, scale: float) -> Ridge:
    r"""The readout of $\alpha$ = ``scale`` $\cdot \mathrm{tr}(X^\top X) /
    D$ from ``x`` ``(n, D)`` to ``y`` ``(n, K)`` (module docstring)."""
    xd, yd = x.double(), y.double()
    mx, my = xd.mean(0), yd.mean(0)
    xc, yc = xd - mx, yd - my
    gram = xc.T @ xc
    alpha = scale * float(gram.trace()) / x.shape[1]
    eye = torch.eye(x.shape[1], dtype=torch.float64, device=x.device)
    b = torch.linalg.solve(gram + alpha * eye, xc.T @ yc)
    return Ridge(mx, my, b, alpha, scale)


def predict(fit: Ridge, x: Tensor) -> Tensor:
    """$\\hat y$ ``(N, K)`` float64 of the codes ``x`` ``(N, D)``."""
    return (x.double() - fit.x_mean) @ fit.b + fit.y_mean


def r2(pred: Tensor, y: Tensor) -> float:
    """The mean over the coordinates of $R^2$ of ``pred`` against ``y``
    ``(N, K)`` (module docstring)."""
    yd = y.double()
    err = (pred.double() - yd).square().sum(0)
    total = (yd - yd.mean(0)).square().sum(0)
    return float((1 - err / total).mean())


def centered_terms(pred: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    r"""The items' terms of ``centered_r2`` of ``pred`` against ``y`` ``(N,
    K)``: $e_i = \lVert \tilde p_i - \tilde y_i \rVert^2$ and $v_i =
    \lVert \tilde y_i \rVert^2$, both ``(N,)`` float64 (module docstring),
    so that $m = 1 - \sum_i e_i / \sum_i v_i$."""
    pd, yd = pred.double(), y.double()
    pc, yc = pd - pd.mean(0), yd - yd.mean(0)
    return (pc - yc).square().sum(1), yc.square().sum(1)


def centered_r2(pred: Tensor, y: Tensor) -> float:
    """$m$ of ``pred`` against ``y`` ``(N, K)`` (module docstring)."""
    e, v = centered_terms(pred, y)
    return float(1 - e.sum() / v.sum())


def rank_bound(y: Tensor, r: int) -> float:
    r"""$b_r$ of the targets ``y`` ``(N, K)`` (module docstring): the share
    of the $r$ largest eigenvalues of their covariance."""
    yd = y.double()
    s2 = torch.linalg.svdvals(yd - yd.mean(0)).square()
    return float(s2[:r].sum() / s2.sum())


def select_ridge(
    x: Tensor,
    y: Tensor,
    x_cal: Tensor,
    y_cal: Tensor,
    scales: Sequence[float] = ALPHA_SCALES,
    score: Callable[[Tensor, Tensor], float] = r2,
) -> Ridge:
    """The readout fit on ``x``, ``y`` of the scale of ``scales`` whose
    ``score`` (``r2``, or ``centered_r2``) on ``x_cal``, ``y_cal`` is the
    best (the first on a tie)."""
    fits = [fit_ridge(x, y, c) for c in scales]
    scores = [score(predict(f, x_cal), y_cal) for f in fits]
    return fits[max(range(len(fits)), key=lambda i: scores[i])]


# -- the scene categories (module docstring) -----------------------------------------


def class_weights(y: Tensor, k: int) -> Tensor:
    """$w_i = 1 / (K n_{y_i})$ ``(n,)`` float64 of the labels ``y`` ``(n,)``
    of ``k`` classes."""
    n = torch.bincount(y, minlength=k).double()
    return 1.0 / (k * n[y])


def fit_weighted_ridge(x: Tensor, y: Tensor, w: Tensor, scale: float) -> Ridge:
    r"""The ridge of $\alpha$ = ``scale`` $\cdot \mathrm{tr}(X_c^\top W
    X_c) / D$ from ``x`` ``(n, D)`` to ``y`` ``(n, K)`` weighted by ``w``
    ``(n,)`` (module docstring); its means are the weighted ones."""
    xd, yd, wd = x.double(), y.double(), w.double().to(x.device)
    total = wd.sum()
    mx, my = (wd @ xd) / total, (wd @ yd) / total
    xc, yc = xd - mx, yd - my
    gram = xc.T @ (wd[:, None] * xc)
    alpha = scale * float(gram.trace()) / x.shape[1]
    eye = torch.eye(x.shape[1], dtype=torch.float64, device=x.device)
    b = torch.linalg.solve(gram + alpha * eye, xc.T @ (wd[:, None] * yc))
    return Ridge(mx, my, b, alpha, scale)


def _table(
    pred: Tensor, y: Tensor, video: Tensor, k: int
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """The counts ``(V, K)`` float64 of the labels $n_{vk}$, the hits
    $h_{vk}$ and the predictions $c_{vk}$, and the rows $N_v$ ``(V,)``, of
    the videos present in order."""
    _, vid = torch.unique(video.cpu(), return_inverse=True)
    nv = int(vid.max()) + 1 if len(vid) else 0
    p, t = pred.cpu().long(), y.cpu().long()
    flat = vid * k

    def count(at: Tensor, on: Tensor | None = None) -> Tensor:
        ones = torch.ones(len(at), dtype=torch.float64)
        if on is not None:
            ones = ones * on.double()
        return torch.zeros(nv * k, dtype=torch.float64).index_add_(0, at, ones)

    n = count(flat + t).view(nv, k)
    hits = count(flat + t, p == t).view(nv, k)
    c = count(flat + p.clamp(0, k - 1), (p >= 0) & (p < k)).view(nv, k)
    rows = torch.bincount(vid, minlength=nv).double()
    return n, hits, c, rows


def _scores(
    pred: Tensor, y: Tensor, video: Tensor, k: int, min_count: int
) -> tuple[float, float, float]:
    """BA, $B$ and $m$ (module docstring)."""
    n, hits, c, rows = _table(pred, y, video, k)
    nk = n.sum(0)
    kept = nk >= max(min_count, 1)
    kk = int(kept.sum())
    if kk == 0:
        return math.nan, math.nan, math.nan
    nk_ = nk[kept]
    ba = float((hits.sum(0)[kept] / nk_).sum()) / kk
    share = c / rows[:, None]  # P(yhat = k | v)
    chance = float(((n * share).sum(0)[kept] / nk_).sum()) / kk
    num = hits * rows[:, None] - n * c  # integers
    m = float(((num / rows[:, None]).sum(0)[kept] / nk_).sum()) / kk
    return ba, chance, m


def balanced_accuracy(pred: Tensor, y: Tensor, k: int, *, min_count: int = 1) -> float:
    """BA of the predictions ``pred`` ``(N,)`` of the labels ``y`` ``(N,)``
    of ``k`` classes (module docstring)."""
    return _scores(pred, y, torch.zeros_like(y), k, min_count)[0]


def video_chance(
    pred: Tensor, y: Tensor, video: Tensor, k: int, *, min_count: int = 1
) -> float:
    """$B$ of the predictions ``pred`` ``(N,)`` of ``y`` ``(N,)`` in the
    videos ``video`` ``(N,)`` (module docstring)."""
    return _scores(pred, y, video, k, min_count)[1]


def scene_m(
    pred: Tensor, y: Tensor, video: Tensor, k: int, *, min_count: int = 1
) -> float:
    r"""$m = \mathrm{BA} - B$ of the predictions ``pred`` ``(N,)`` of ``y``
    ``(N,)`` in the videos ``video`` ``(N,)`` (module docstring)."""
    return _scores(pred, y, video, k, min_count)[2]


def _auc(s: Tensor, pos: Tensor) -> float:
    """The Mann-Whitney AUC of the scores ``s`` ``(n,)`` of the positives
    ``pos`` ``(n,)`` against the others (a tie counts 1/2)."""
    a, b = s[pos], s[~pos]
    return float(((a[:, None] > b).double() + 0.5 * (a[:, None] == b).double()).mean())


def class_auc(
    scores: Tensor, y: Tensor, video: Tensor, k: int, *, min_each: int = 20
) -> float | None:
    """The one-against-the-rest AUC of each class's output column of
    ``scores`` ``(N, K)`` within each video $v$ where class $k$ has
    ``min_each`` positives and negatives or more, averaged over the videos
    by $n_{vk}$, then over the classes that have one (None if none)."""
    per_class = []
    for c in range(k):
        num = den = 0.0
        for v in torch.unique(video).tolist():
            on = video == v
            pos = y[on] == c
            npos = int(pos.sum())
            if npos < min_each or int((~pos).sum()) < min_each:
                continue
            num += npos * _auc(scores[on][:, c].double(), pos)
            den += npos
        if den:
            per_class.append(num / den)
    return sum(per_class) / len(per_class) if per_class else None


class _Moments:
    """The sufficient statistics of a code ``x`` ``(N, D)`` (less its mean,
    float64) by video and class: $n_{vk}$, the sums $s_{vk}$ ``(D,)`` and
    the second moments $M_{vk}$ ``(D, D)``; every fit of ``nested_cv`` is
    made of them."""

    def __init__(self, x: Tensor, y: Tensor, video: Tensor, k: int) -> None:
        xd = x.double()
        self.x = xd - xd.mean(0)
        self.k = k
        self.videos = torch.unique(video).tolist()
        dev = x.device
        y, video = y.to(dev), video.to(dev)
        d = x.shape[1]
        nv = len(self.videos)
        self.n = torch.zeros(nv, k, dtype=torch.float64, device=dev)
        self.s = torch.zeros(nv, k, d, dtype=torch.float64, device=dev)
        self.m = torch.zeros(nv, k, d, d, dtype=torch.float64, device=dev)
        for i, v in enumerate(self.videos):
            for c in range(k):
                r = self.x[(video == v) & (y == c)]
                self.n[i, c] = len(r)
                self.s[i, c] = r.sum(0)
                self.m[i, c] = r.T @ r

    def fit(self, use: Sequence[int]) -> _Fold:
        """The fit on the videos at positions ``use`` of ``videos``."""
        at = torch.tensor(list(use), device=self.n.device)
        n, s, m = self.n[at].sum(0), self.s[at].sum(0), self.m[at].sum(0)
        k, d = self.k, self.x.shape[1]
        rows = n.sum()
        mean = s.sum(0) / rows
        var = torch.diagonal(m.sum(0)) / rows - mean.square()
        sd = var.clamp(min=0).sqrt()
        sd = torch.where(sd <= 1e-12, torch.ones_like(sd), sd)
        present = n > 0
        w = torch.where(present, 1.0 / (k * n.clamp(min=1)), torch.zeros_like(n))
        total = (w * n).sum()
        xbar = (w[:, None] * s).sum(0) / total
        gram = (w[:, None, None] * m).sum(0) - total * torch.outer(xbar, xbar)
        gram = gram / torch.outer(sd, sd)
        r = (w[:, None] * (s - n[:, None] * xbar)).T / sd[:, None]  # (D, K)
        lam, u = torch.linalg.eigh(gram)
        return _Fold(
            xbar=xbar,
            sd=sd,
            u=u,
            lam=lam,
            st=u.T @ r,
            ybar=w * n / total,
            unit=float(lam.sum()) / d,
        )


@dataclass(frozen=True)
class _Fold:
    r"""A fit of ``_Moments.fit``: the weighted mean ``xbar``, the sds, the
    eigenvectors ``u`` and values ``lam`` of the standardized weighted gram,
    $U^\top X_c^\top W Y_c$ ``st``, the weighted means of the labels and
    $\mathrm{tr} / D$ ``unit`` ($\alpha = c$ ``unit``)."""

    xbar: Tensor
    sd: Tensor
    u: Tensor
    lam: Tensor
    st: Tensor
    ybar: Tensor
    unit: float

    def scores(self, x: Tensor, scale: float) -> Tensor:
        """The outputs ``(n, K)`` on the rows ``x`` (``_Moments.x``'s) at
        ``scale``; 0 for ``CONST``."""
        if scale == CONST:
            return torch.zeros(len(x), len(self.ybar), dtype=torch.float64)
        a = ((x - self.xbar) / self.sd) @ self.u
        out = a @ (self.st / (self.lam + scale * self.unit)[:, None]) + self.ybar
        return out.cpu()


@dataclass(frozen=True)
class NestedResult:
    """``nested_cv``'s: the outer predictions ``pred`` ``(N,)`` and outputs
    ``scores`` ``(N, K)`` float64, the scale chosen for each scored video
    (by its id) and the inner $m$ of each candidate (by scored video)."""

    pred: Tensor
    scores: Tensor
    scale_by_video: dict[int, float]
    inner: dict[int, list[float]]


def nested_cv(
    x: Tensor,
    y: Tensor,
    video: Tensor,
    k: int,
    *,
    scales: Sequence[float] = (*SCENE_SCALES, CONST),
    fixed: Mapping[int, float] | None = None,
) -> NestedResult:
    """The nested cross-validation by video (module docstring) of the code
    ``x`` ``(N, D)`` to the labels ``y`` ``(N,)`` of ``k`` classes in the
    videos ``video`` ``(N,)``, the candidates ``scales``; with ``fixed``
    (by scored video), each scale given and no inner choice (``inner``
    empty)."""
    mom = _Moments(x, y, video, k)
    vids = mom.videos
    yc, vc = y.cpu(), video.cpu()
    rows = {v: (vc == v).nonzero().flatten() for v in vids}
    pred = torch.zeros(len(y), dtype=torch.long)
    out = torch.zeros(len(y), k, dtype=torch.float64)
    chosen: dict[int, float] = {}
    inner: dict[int, list[float]] = {}
    for i, v in enumerate(vids):
        fit = [j for j in range(len(vids)) if j != i]
        if fixed is not None:
            chosen[int(v)] = fixed[int(v)]
            at = rows[v]
            s = mom.fit(fit).scores(mom.x[at.to(mom.x.device)], fixed[int(v)])
            out[at], pred[at] = s, s.argmax(1)
            continue
        on = torch.cat([rows[vids[j]] for j in fit])
        inner_pred = torch.zeros(len(scales), len(y), dtype=torch.long)
        for j in fit:
            f = mom.fit([u for u in fit if u != j])
            at = rows[vids[j]]
            xa = mom.x[at.to(mom.x.device)]
            for c, scale in enumerate(scales):
                inner_pred[c, at] = f.scores(xa, scale).argmax(1)
        ms = [scene_m(inner_pred[c, on], yc[on], vc[on], k) for c in range(len(scales))]
        best = max(range(len(scales)), key=lambda c: (ms[c], scales[c]))  # tie: larger
        chosen[int(v)], inner[int(v)] = scales[best], ms
        at = rows[v]
        s = mom.fit(fit).scores(mom.x[at.to(mom.x.device)], scales[best])
        out[at], pred[at] = s, s.argmax(1)
    return NestedResult(pred, out, chosen, inner)


def nested_cv_jackknife(
    x: Tensor, y: Tensor, video: Tensor, k: int, scale_by_video: Mapping[int, float]
) -> Tensor:
    """The outer predictions ``(U, N)`` long without each video $u$ (row
    $u$ of the videos in order): each other video $v$ by the readout fit on
    the rest at its scale ``scale_by_video[v]``; $u$'s own rows -1."""
    mom = _Moments(x, y, video, k)
    vids = mom.videos
    vc = video.cpu()
    rows = {v: (vc == v).nonzero().flatten() for v in vids}
    out = torch.full((len(vids), len(y)), -1, dtype=torch.long)
    for a in range(len(vids)):
        for b, v in enumerate(vids):
            if b == a:
                continue
            f = mom.fit([j for j in range(len(vids)) if j not in (a, b)])
            at = rows[v]
            out[a, at] = f.scores(mom.x[at.to(mom.x.device)], scale_by_video[v]).argmax(
                1
            )
    return out
