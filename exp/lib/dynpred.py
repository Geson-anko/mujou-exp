r"""Direct prediction of whitened codes, against the copy and the mean
(``exp/vision/plans/dyn.md``, registered 2026-11-26 before any number).

**The ratios.** Of a lag $n$ (steps) and the triples $(t - n, t, t + n)$ of a
set's centres $t$ (``Triples``), a predictor $\hat w_{t+n} = f(w_t, w_{t-n})$
has

$$R_{\mathrm{copy}} = \frac{\sum_t \lVert \hat w_{t+n} - w_{t+n} \rVert^2}
{\sum_t \lVert w_t - w_{t+n} \rVert^2}, \qquad R_{\mathrm{clim}} =
\frac{\sum_t \lVert \hat w_{t+n} - w_{t+n} \rVert^2}{\sum_t \lVert \bar w -
w_{t+n} \rVert^2},$$

$\bar w$ the training set's mean; the blocks' sums of the numerator
(``num_b``), of the two denominators (``copy_b``, ``clim_b``, ``dens``) and
the counts (``n_b``) are what a row holds. Their ratio $R_{\mathrm{copy}} /
R_{\mathrm{clim}}$ is the code's own, of no model.

**The linear family** (``fit_linear``): $\hat w = M_1 w_t + M_2 w_{t-n} +
b$, the inputs $[w_t, w_{t-n}]$ (``two``; else $w_t$ alone) solved by
``linear_dyn.fit_scales`` (ridge, intercept, float64) at every $\alpha$ of
``ALPHA_SCALES``, the least numerator on the validation set chosen (the
smallest $c$ on a tie). It holds the copy ($M_1 = I$), the constant speed
($2I, -I$), $\exp(nA)$ ($M_2 = 0$) and the mean ($M = 0$). Of an
orthogonal $Q$ of the codes, $w \mapsto Qw$, the moments map by $Q \oplus Q$,
$\mathrm{tr}(C_{00})$ and the penalty $\lVert M \rVert_F^2$ are unchanged,
so $\hat M \mapsto Q \hat M (Q \oplus Q)^\top$ and every ratio is unchanged,
the ridge included.

**The residual MLP** (``train_mlp``): $\hat w = \hat w^{\mathrm{lin}} +
g_\theta([w_t, w_{t-n}])$, $g$ of ``MLPConf`` (GELU, the output layer's
weight and bias 0, so $g \equiv 0$ at step 0), the linear fixed; Adam on the
mean squared residual, every lag and seed side by side (``(L S)`` models of
stacked weights: Adam is elementwise and each model's loss its own, so each
is the model learned alone). Each seed draws its initial weights, then its
batches ``every`` steps at a time, by its generator: one seed's batches are
the same centres at every lag. At the times ``0, every, .., steps`` the
numerators of every set; of each lag the time of the least validation
numerator (the mean of the seeds; the earliest on a tie, ``choose``). The
time 0 is the linear itself (its sums, not recomputed: $g \equiv 0$), so
the chosen time's validation numerator is never above the linear's.

**The main number** (``mean_log``): $\overline{\log R} = \frac{1}{\log(hi /
lo)} \int_{\log lo}^{\log hi} \log R(n)\,d\log n$ by the trapezoids of the
lags in $[lo, hi]$ (``GRID``'s: the integers $\mathrm{round}(2^{k/2})$,
``half_octaves``). Its draws (``mean_log_draws``) are of the blocks' counts
(``metrics.block_counts``), a draw of a zero denominator NaN.

**The oracle** (``best_linear``, ``fixed_error``): of a stationary process of
mean 0 and autocovariance $\Gamma(\ell) = E[w_{t+\ell} w_t^\top]$, the inputs
$x = [w_t; w_{t-n}]$ and the target $y = w_{t+n}$ have $K = E[y x^\top] =
[\Gamma(n), \Gamma(2n)]$ and $\Sigma = E[x x^\top] = \begin{bmatrix}\Gamma(0)
& \Gamma(n) \\ \Gamma(n)^\top & \Gamma(0)\end{bmatrix}$; the least error of
a linear map is $E^* = \mathrm{tr}(\Gamma(0) - K \Sigma^+ K^\top)$ (second
moments alone: of any process), and a fixed $(M, b)$ has $E =
\mathrm{tr}(\Gamma(0) - 2MK^\top + M\Sigma M^\top) + \lVert b \rVert^2$. The
copy's is $\mathrm{tr}(2\Gamma(0) - \Gamma(n) - \Gamma(n)^\top)$ and the
mean $\bar w$'s $\mathrm{tr}\,\Gamma(0) + \lVert \bar w \rVert^2$
(``population_dens``). Of one dimension, $\mathrm{FVU} = (1 - 2r_1^2 +
2r_1^2 r_2 - r_2^2) / (1 - r_1^2)$; of an OU, $R_{\mathrm{copy}} = (1 + r)/2$
and $R_{\mathrm{clim}} = 1 - r^2$.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from exp.lib.linear_dyn import ALPHA_SCALES, fit_scales
from exp.lib.metrics import ratio_resamples

__all__ = [
    "GRID",
    "MAIN",
    "LinFit",
    "MLPConf",
    "MLPRun",
    "Triples",
    "best_linear",
    "best_linear_one",
    "block_sums",
    "choose",
    "dens",
    "fit_linear",
    "fixed_error",
    "half_octaves",
    "mean_log",
    "mean_log_draws",
    "mlp_forward",
    "mlp_init",
    "num_sums",
    "pair_blocks",
    "population_dens",
    "record_preds",
    "train_mlp",
]

_F64 = torch.float64


def half_octaves(top: int) -> tuple[int, ...]:
    r"""The integers $\mathrm{round}(2^{k/2})$, $k = 0 .. 2\log_2$ ``top``,
    once each (1 is both $k = 0$ and $k = 1$)."""
    ks = range(round(2 * math.log2(top)) + 1)
    return tuple(sorted({round(2 ** (k / 2)) for k in ks}))


GRID = half_octaves(256)  # 0.1 s .. 25.6 s, 16 lags
MAIN = (8, 256)  # 0.8 s .. 25.6 s


@dataclass(frozen=True)
class Triples:
    """A set's triples: the codes ``w`` ``(N, k)`` (float64), the centres
    ``(C,)`` (every triple of every lag used lies in ``w``), each centre's
    block ``blocks`` ``(C,)`` of ``n_blocks``, and optionally a second
    blocking (``blocks_s`` of ``n_blocks_s``: the blocks shifted by half a
    block, a sensitivity)."""

    w: Tensor
    centres: Tensor
    blocks: Tensor
    n_blocks: int
    blocks_s: Tensor | None = None
    n_blocks_s: int = 0

    def at(self, lag: int) -> tuple[Tensor, Tensor, Tensor]:
        """$w_t$, $w_{t-n}$, $w_{t+n}$ ``(C, k)`` of the centres."""
        c = self.centres
        return self.w[c], self.w[c - lag], self.w[c + lag]

    def to(self, dev: torch.device) -> Triples:
        return Triples(self.w.to(dev), self.centres.to(dev), self.blocks,
                       self.n_blocks, self.blocks_s, self.n_blocks_s)  # fmt: skip


def block_sums(values: Tensor, blocks: Tensor, n_blocks: int) -> Tensor:
    """The sums ``(..., n_blocks)`` float64 (CPU) of ``values`` ``(..., C)``
    by their ``blocks`` ``(C,)``, added in the centres' order (the same for
    one model as for many side by side)."""
    v = values.double().cpu()
    v2 = v.reshape(-1, v.shape[-1])
    out = torch.zeros(v2.shape[0], n_blocks, dtype=_F64)
    out.index_add_(1, blocks.cpu(), v2)
    return out.reshape(*v.shape[:-1], n_blocks)


def num_sums(err: Tensor, s: Triples) -> dict[str, Tensor]:
    """``num_b`` (and ``num_b_s`` of the shifted blocks) of the errors
    ``err`` ``(..., C, k)``: the blocks' sums of the squared norms."""
    sq = err.double().square().sum(-1)
    out = {"num_b": block_sums(sq, s.blocks, s.n_blocks)}
    if s.blocks_s is not None:
        out["num_b_s"] = block_sums(sq, s.blocks_s, s.n_blocks_s)
    return out


def dens(s: Triples, lag: int, mean: Tensor) -> dict[str, Tensor]:
    r"""``copy_b`` $\sum \lVert w_t - w_{t+n} \rVert^2$, ``clim_b`` $\sum
    \lVert \bar w - w_{t+n} \rVert^2$ (``mean`` $\bar w$) and ``n_b`` of the
    blocks (and of the shifted blocks, ``*_s``)."""
    now, _, fut = s.at(lag)
    out: dict[str, Tensor] = {}
    for blocks, nb, tail in (
        (s.blocks, s.n_blocks, ""),
        (s.blocks_s, s.n_blocks_s, "_s"),
    ):
        if blocks is None:
            continue
        out[f"copy_b{tail}"] = block_sums((now - fut).square().sum(1), blocks, nb)
        out[f"clim_b{tail}"] = block_sums(
            (mean.to(fut) - fut).square().sum(1), blocks, nb
        )
        out[f"n_b{tail}"] = block_sums(torch.ones(len(fut)), blocks, nb)
    return out


# -- the linear family --------------------------------------------------------------


def _inputs(s: Triples, lag: int, two: bool) -> tuple[Tensor, Tensor]:
    now, past, fut = s.at(lag)
    return (torch.cat([now, past], 1) if two else now), fut


@dataclass(frozen=True)
class LinFit:
    r"""A fit of one lag: $M$ ``a`` ``(k, m)``, $b$ ``(k,)`` (float64), the
    chosen ``alpha`` and its ``scale`` $c$, the validation numerators of
    every scale ``val``, and the training numerator of the smallest scale
    ``train_min`` at ``alpha_min`` (``plans/dyn.md``'s V-p2b)."""

    lag: int
    two: bool
    a: Tensor
    b: Tensor
    alpha: float
    scale: float
    val: tuple[float, ...]
    train_min: float
    alpha_min: float

    def predict(self, s: Triples) -> tuple[Tensor, Tensor]:
        """$\\hat w_{t+n}$ and $w_{t+n}$ ``(C, k)`` float64 of the set."""
        x, y = _inputs(s, self.lag, self.two)
        return x.double() @ self.a.T.to(x.device) + self.b.to(x.device), y.double()


def fit_linear(
    train: Triples,
    val: Triples,
    lag: int,
    *,
    two: bool = True,
    scales: Sequence[float] = ALPHA_SCALES,
) -> LinFit:
    """The linear predictor of ``lag`` (module docstring): fit on ``train``
    at every scale, chosen on ``val``."""
    x, y = _inputs(train, lag, two)
    fits = fit_scales(x, y, scales)
    xv, yv = _inputs(val, lag, two)
    vals = []
    for _, a, b in fits:
        vals.append(float((xv.double() @ a.T + b - yv).square().sum()))
    best = min(range(len(fits)), key=lambda i: (vals[i], i))
    alpha, a, b = fits[best]
    a0, b0 = fits[0][1], fits[0][2]
    train_min = float((x.double() @ a0.T + b0 - y).square().sum())
    return LinFit(lag, two, a.cpu(), b.cpu(), alpha, scales[best], tuple(vals),
                  train_min, fits[0][0])  # fmt: skip


def record_preds(
    train: Triples, s: Triples, lag: int, mean: Tensor
) -> dict[str, Tensor]:
    r"""The predictions ``(C, k)`` of the record's rows on ``s``: ``copy``
    $w_t$, ``vel`` $2w_t - w_{t-n}$, ``mean`` $\bar w$, ``scale`` $\bar w +
    c(w_t - \bar w)$ of the least squares $c$ on ``train``."""
    now_t, _, fut_t = train.at(lag)
    m = mean.to(now_t)
    c = float(((fut_t - m) * (now_t - m)).sum() / (now_t - m).square().sum())
    now, past, _ = s.at(lag)
    m = mean.to(now)
    return {"copy": now, "vel": 2 * now - past, "mean": m.expand_as(now),
            "scale": m + c * (now - m)}  # fmt: skip


# -- the residual MLP ----------------------------------------------------------------


@dataclass(frozen=True)
class MLPConf:
    """The residual MLP's settings (``plans/dyn.md``)."""

    width: int = 512
    lr: float = 1e-3
    batch: int = 256
    steps: int = 20000
    every: int = 500
    seeds: tuple[int, ...] = (5001, 5002, 5003)
    chunk: int = 4096  # rows of a set evaluated at a time

    @property
    def times(self) -> tuple[int, ...]:
        return tuple(range(0, self.steps + 1, self.every))


def mlp_init(k_in: int, k_out: int, width: int, gen: torch.Generator) -> list[Tensor]:
    r"""One model's ``[W1, b1, W2, b2, W3, b3]`` (float32; ``W`` ``(in,
    out)``): the hidden layers uniform in $\pm 1/\sqrt{\mathrm{fan\_in}}$
    (``nn.Linear``'s default) by ``gen`` in that order, the output layer 0."""

    def uni(*shape: int, fan: int) -> Tensor:
        return (torch.rand(*shape, generator=gen) * 2 - 1) / math.sqrt(fan)

    return [uni(k_in, width, fan=k_in), uni(width, fan=k_in),
            uni(width, width, fan=width), uni(width, fan=width),
            torch.zeros(width, k_out), torch.zeros(k_out)]  # fmt: skip


def mlp_forward(p: Sequence[Tensor], x: Tensor) -> Tensor:
    """$g(x)$ ``(M, B, k)`` of the stacked models ``p`` (``mlp_init``'s, each
    with a leading ``M``) on ``x`` ``(M, B, m)``."""
    gelu = nn.functional.gelu
    h = gelu(torch.baddbmm(p[1][:, None], x, p[0]))
    h = gelu(torch.baddbmm(p[3][:, None], h, p[2]))
    return torch.baddbmm(p[5][:, None], h, p[4])


@dataclass(frozen=True)
class MLPRun:
    """The MLP of ``lags`` x ``seeds``: ``times``, the validation numerators
    ``val`` ``(L, S, T)``, the chosen time's index of each lag ``chosen``
    ``(L,)``, and of each set its blocks' sums at the chosen time ``sums``
    (by set, ``num_b`` ``(L, S, B)`` and ``num_b_s`` with shifted blocks)."""

    lags: tuple[int, ...]
    seeds: tuple[int, ...]
    times: tuple[int, ...]
    val: Tensor
    chosen: Tensor
    sums: Mapping[str, Mapping[str, Tensor]]


def choose(val: Tensor) -> Tensor:
    """The time ``(L,)`` of the least mean over the seeds of ``val`` ``(L,
    S, T)``, the earliest on a tie."""
    return val.double().mean(1).argmin(1)  # argmin: the first of the least


def _set_errors(
    p: Sequence[Tensor], s: Triples, lags: Sequence[int], e_lin: Tensor, n_seeds: int,
    chunk: int,
) -> Tensor:  # fmt: skip
    """The squared errors ``(L S, C)`` float64 of the models on the set
    (``e_lin`` ``(L, C, k)`` the linear's errors $\\hat w^{\\mathrm{lin}} -
    w$ on its device)."""
    c, k = len(s.centres), s.w.shape[1]
    out = torch.empty(len(lags) * n_seeds, c, dtype=_F64, device=s.w.device)
    w32 = s.w.float()
    for i in range(0, c, chunk):
        cen = s.centres[i : i + chunk]
        now = w32[cen]
        x = torch.stack([torch.cat([now, w32[cen - n]], 1) for n in lags])  # (L, b, 2k)
        x = (
            x[:, None]
            .expand(-1, n_seeds, -1, -1)
            .reshape(len(lags) * n_seeds, -1, 2 * k)
        )
        with torch.no_grad():
            g = mlp_forward(p, x).double().reshape(len(lags), n_seeds, -1, k)
        err = e_lin[:, None, i : i + chunk] + g
        out[:, i : i + chunk] = err.square().sum(-1).reshape(len(lags) * n_seeds, -1)
    return out


def _diff(pred: Tensor, y: Tensor) -> Tensor:
    return pred - y


def train_mlp(
    fits: Mapping[int, LinFit],
    train: Triples,
    val: Triples,
    sets: Mapping[str, Triples],
    conf: MLPConf,
    dev: torch.device,
) -> MLPRun:
    """The residual MLP of every lag of ``fits`` (the linear fits) and seed
    of ``conf``, learned on ``train``, the time chosen on ``val``, the
    numerators of ``sets`` (by name; ``val`` among them or not) at the
    chosen time (module docstring)."""
    lags = tuple(sorted(fits))
    ns, nl = len(conf.seeds), len(lags)
    train, val = train.to(dev), val.to(dev)
    every = {name: s.to(dev) for name, s in sets.items()}
    val_key = "val" if "val" in every else "__val__"
    every.setdefault(val_key, val)

    def errors_of(s: Triples) -> Tensor:
        return torch.stack([_diff(*fits[n].predict(s)) for n in lags])

    e_train = errors_of(train)
    e_sets = {name: errors_of(s) for name, s in every.items()}
    k = train.w.shape[1]
    gens = [torch.Generator().manual_seed(seed) for seed in conf.seeds]
    inits = [mlp_init(2 * k, k, conf.width, g) for g in gens]
    p = [torch.stack([inits[j][i] for _ in lags for j in range(ns)]).to(dev).requires_grad_()
         for i in range(6)]  # fmt: skip
    opt = torch.optim.Adam(p, lr=conf.lr)
    w32 = train.w.float()
    # the time 0: g = 0, the linear's own sums
    lin_sq = {name: e.square().sum(-1) for name, e in e_sets.items()}  # (L, C)
    sums_t: dict[str, list[dict[str, Tensor]]] = {name: [] for name in every}

    def record(sq: Mapping[str, Tensor]) -> None:
        for name, s in every.items():
            v = sq[name].reshape(nl, -1, sq[name].shape[-1])
            if v.shape[1] == 1:
                v = v.expand(-1, ns, -1)
            got = {"num_b": block_sums(v, s.blocks, s.n_blocks)}
            if s.blocks_s is not None:
                got["num_b_s"] = block_sums(v, s.blocks_s, s.n_blocks_s)
            sums_t[name].append(got)

    record({name: v[:, None] for name, v in lin_sq.items()})
    c_train = len(train.centres)
    for _ in range(0, conf.steps, conf.every):
        idx = [
            torch.randint(c_train, (conf.every, conf.batch), generator=g) for g in gens
        ]
        for j in range(conf.every):
            rows = torch.stack([ix[j] for ix in idx]).to(dev)  # (S, B)
            cen = train.centres[rows]
            now = w32[cen]  # (S, B, k)
            x = torch.stack([torch.cat([now, w32[cen - n]], -1) for n in lags])
            target = -torch.stack([e_train[i][rows] for i in range(nl)]).float()
            out = mlp_forward(p, x.reshape(nl * ns, conf.batch, 2 * k))
            loss = (
                (out - target.reshape(nl * ns, conf.batch, k))
                .square()
                .mean((1, 2))
                .sum()
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            record({name: _set_errors([q.detach() for q in p], s, lags, e_sets[name], ns,
                                      conf.chunk) for name, s in every.items()})  # fmt: skip
    val_t = torch.stack([d["num_b"].sum(-1) for d in sums_t[val_key]], -1)  # (L, S, T)
    chosen = choose(val_t)
    sums = {
        name: {key: torch.stack([sums_t[name][int(chosen[i])][key][i] for i in range(nl)])
               for key in sums_t[name][0]}
        for name in sets
    }  # fmt: skip
    return MLPRun(lags, conf.seeds, conf.times, val_t, chosen, sums)


# -- the main number and its draws ---------------------------------------------------


def mean_log(
    lags: Sequence[int], r: Tensor, lo: int = MAIN[0], hi: int = MAIN[1]
) -> Tensor:
    r"""$\overline{\log R}$ ``(...)`` of ``r`` ``(..., L)`` at ``lags``
    ``(L,)`` (module docstring): the trapezoids of $\log R$ over $\log n$ of
    the lags in $[lo, hi]$ over $\log(hi / lo)$ (NaN where an $R$ there is
    not positive or not finite).

    Raises:
        ValueError: ``lo`` or ``hi`` is not among the lags.
    """
    lag_list = list(lags)
    if lo not in lag_list or hi not in lag_list:
        raise ValueError(f"lo {lo} and hi {hi} must be lags of {lag_list}")
    keep = [i for i, n in enumerate(lag_list) if lo <= n <= hi]
    x = torch.tensor([math.log(lag_list[i]) for i in keep], dtype=_F64)
    y = r.double()[..., keep]
    bad = ~(torch.isfinite(y) & (y > 0)).all(-1)
    ly = y.clamp(min=torch.finfo(_F64).tiny).log()
    area = ((ly[..., 1:] + ly[..., :-1]) / 2 * (x[1:] - x[:-1])).sum(-1)
    out = area / math.log(hi / lo)
    return torch.where(bad, torch.full_like(out, math.nan), out)


def pair_blocks(b: Tensor) -> Tensor:
    """The sums ``(..., B // 2)`` of adjacent blocks (the last odd one joined
    to the pair before it; ``width_eval``'s 256 s blocks)."""
    n = b.shape[-1]
    to = torch.arange(n).div(2, rounding_mode="floor").clamp(max=n // 2 - 1)
    return b.new_zeros(*b.shape[:-1], n // 2).index_add_(-1, to, b)


def mean_log_draws(
    lags: Sequence[int], num_b: Tensor, den_b: Tensor, counts: Tensor,
    lo: int = MAIN[0], hi: int = MAIN[1],
) -> Tensor:  # fmt: skip
    r"""$\overline{\log R}$ ``(n,)`` of each draw of the blocks' ``counts``
    ``(n, B)`` of the sums ``num_b``, ``den_b`` ``(L, B)`` (NaN of a draw
    whose denominator is 0 at a lag of the range)."""
    return mean_log(lags, ratio_resamples(num_b, den_b, counts), lo, hi)


# -- the oracle ----------------------------------------------------------------------


def _k_sigma(g0: Tensor, g1: Tensor, g2: Tensor) -> tuple[Tensor, Tensor]:
    k = torch.cat([g1, g2], 1)
    sigma = torch.cat([torch.cat([g0, g1], 1), torch.cat([g1.T, g0], 1)], 0)
    return k.double(), sigma.double()


def best_linear(g0: Tensor, g1: Tensor, g2: Tensor) -> float:
    r"""$E^* = \mathrm{tr}(\Gamma(0) - K \Sigma^+ K^\top)$ of the two inputs
    (module docstring) of $\Gamma(0)$, $\Gamma(n)$, $\Gamma(2n)$ ``(k,
    k)``."""
    k, sigma = _k_sigma(g0, g1, g2)
    sym = (sigma + sigma.T) / 2
    return float(
        g0.double().trace() - (k @ torch.linalg.pinv(sym, hermitian=True) @ k.T).trace()
    )


def best_linear_one(g0: Tensor, g1: Tensor) -> float:
    r"""$\mathrm{tr}(\Gamma(0) - \Gamma(n) \Gamma(0)^+ \Gamma(n)^\top)$, the
    least error of one input $w_t$."""
    g0, g1 = g0.double(), g1.double()
    return float(
        (g0 - g1 @ torch.linalg.pinv((g0 + g0.T) / 2, hermitian=True) @ g1.T).trace()
    )


def fixed_error(m: Tensor, b: Tensor, g0: Tensor, g1: Tensor, g2: Tensor) -> float:
    r"""$E = \mathrm{tr}(\Gamma(0) - 2MK^\top + M\Sigma M^\top) + \lVert b
    \rVert^2$ of the fixed $M$ ``(k, 2k)``, $b$ ``(k,)`` (module
    docstring)."""
    k, sigma = _k_sigma(g0, g1, g2)
    m = m.double()
    return float(g0.double().trace() - 2 * (m * k).sum() + (m @ sigma * m).sum()
                 + b.double().square().sum())  # fmt: skip


def population_dens(g0: Tensor, g1: Tensor, mean: Tensor) -> tuple[float, float]:
    r"""The copy's $\mathrm{tr}(2\Gamma(0) - \Gamma(n) - \Gamma(n)^\top)$ and
    the mean's $\mathrm{tr}\,\Gamma(0) + \lVert \bar w \rVert^2$."""
    g0, g1 = g0.double(), g1.double()
    return (float(2 * g0.trace() - 2 * g1.trace()),
            float(g0.trace() + mean.double().square().sum()))  # fmt: skip
