r"""The observation's content a code keeps: the guard of the linear horizon
rebuilt (``exp/vision/plans/jepa.md``, 「線形の幅のガードの作り直し」; README,
「線形の幅のガードの作り直し（回す前に決めたこと）」).

**The readout** $g$ (``fit_mlp``): a fixed MLP from the code $c_t$ of a time
to that time's observation $y_t$, ``MLPConf`` (``layers`` hidden layers of
``hidden`` units, GELU, Adam, a mean squared error over batches drawn with
replacement by the seed's generator; its weights drawn from the same seed).
Of ``fit_linear`` the least squares with an intercept (the minimum-norm
solution of the normal equations, so a code with a constant coordinate has
one), a record.

**The two quantities** (``content_sums``, ``ContentSums``) at anchors $t$:
the present's $E = 1 - \sum_t \lVert y_t - g(c_t) \rVert^2 / \sum_t \lVert
y_t - \bar y \rVert^2$ ($\bar y$ the training set's mean) and the change's
$E^d = 1 - \sum_t \lVert d_t - \hat d_t \rVert^2 / \sum_t \lVert d_t
\rVert^2$, $d_t = y_t - y_{t-1}$, $\hat d_t = g(c_t) - g(c_{t-1})$; each
kept as the blocks' sums, one row of errors per readout seed. A quantity's
value is that of the seeds' mean error (the denominators are of the data
alone, so it is the mean over the seeds of each seed's value). A constant
readout $g = \bar y$ has $E = 0$ on a set whose mean is $\bar y$ and $E^d =
0$ exactly; $g = y$ has 1 and 1.

**The ratio** $\psi = \log_2(E_a / E_b)$ (``ratio_interval``): of the
blocks drawn by ``counts`` (and of their adjacent pairs by ``counts2``, the
256 s blocks) the draws $\psi^*$; $\mathrm{SE}^*$ the larger of the two
lengths' standard deviations, $\mathrm{SE}_{\mathrm{tot}} =
\sqrt{\mathrm{SE}^{*2} + \sigma_{\mathrm{seed}}^2}$ (the readout's training
spread, independent of the anchors': C12's form) and the interval $\hat\psi
\pm 2\,\mathrm{SE}_{\mathrm{tot}}$. A draw of $E_a^* \le 0 < E_b^*$ is
$-\infty$ and one of $E_b^* \le 0$ undefined; where more than 2.5% of the
draws are not finite the interval's lower end is $-\infty$ (and of more than
2.5% undefined also its upper end $+\infty$): a percentile interval would
reach them. $\sigma_{\mathrm{seed}}$ (``seed_sigma``): the sample standard
deviation over the seeds of one pair's per-seed $\psi_s$ (the same anchors
and data, so the anchors' spread is not in it).

**Comparable** (``comparable``): the other code's $\hat E_b - 2\sqrt{
\mathrm{SE}^{*2} + s_b^2} > 0$, $s_b$ the sample deviation of its per-seed
values: a ratio to an $E_b$ that may be 0 says nothing.

**The front's verdict** (``front_verdict``) of a pair $(a, b)$, $a$ the one
claimed wide: where the width's ratio clears $+\Delta_*$ wholly, ``kept``
(the observation's content kept within $\psi_*$: both lower ends of $\psi$,
$\psi^d$ above $-\psi_*$), ``dropped`` (an upper end below $-\psi_*$),
``unresolved`` (else), ``incomparable`` (the other code not comparable;
checked first); else ``width`` (the width's own rule).
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = [
    "VERDICTS",
    "ContentSums",
    "MLPConf",
    "comparable",
    "content_sums",
    "fit_linear",
    "fit_mlp",
    "front_verdict",
    "mlp",
    "pool_obs",
    "predict",
    "predict_linear",
    "ratio_interval",
    "seed_sigma",
    "stack_seeds",
]

_F64 = torch.float64
VERDICTS = ("kept", "dropped", "unresolved", "incomparable", "width")
_TAIL = 0.025  # the share of draws a 95% percentile interval leaves on a side


# -- the observation and the readouts ------------------------------------------------


def pool_obs(frames: Tensor, size: int, batch: int = 4096) -> Tensor:
    """The frames u8 ``(N, 3, S, S)`` averaged to ``size`` x ``size``
    (``adaptive_avg_pool2d``: the mean of each cell where ``size`` divides
    ``S``) in [0, 1], flattened ``(N, 3 size^2)`` float32."""
    out = [F.adaptive_avg_pool2d(c.float() / 255, size).flatten(1)
           for c in frames.split(batch)]  # fmt: skip
    return torch.cat(out) if out else frames.new_zeros(0, 3 * size * size).float()


@dataclass(frozen=True)
class MLPConf:
    """The readout's settings (``plans/jepa.md``): by default the real
    data's, 64 -> 1024 -> 1024 -> out."""

    hidden: int = 1024
    layers: int = 2
    steps: int = 20000
    batch: int = 256
    lr: float = 1e-3


def mlp(d_in: int, d_out: int, conf: MLPConf) -> nn.Sequential:
    """``layers`` hidden layers of ``hidden`` units and GELU, then linear."""
    mods: list[nn.Module] = []
    width = d_in
    for _ in range(conf.layers):
        mods += [nn.Linear(width, conf.hidden), nn.GELU()]
        width = conf.hidden
    mods.append(nn.Linear(width, d_out))
    return nn.Sequential(*mods)


def fit_mlp(
    x: Tensor, y: Tensor, conf: MLPConf, seed: int, dev: torch.device
) -> tuple[nn.Module, list[float]]:
    """The readout of ``seed`` learned on the codes ``x`` ``(N, D)`` and the
    observations ``y`` ``(N, K)`` (float32 on ``dev``): its weights drawn
    after ``torch.manual_seed(seed)`` (the global generator restored), its
    batches by a CPU generator of ``seed``. Returns the net (eval) and the
    loss of every 100th step."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        net = mlp(x.shape[1], y.shape[1], conf)
    net = net.to(dev)
    xd, yd = x.float().to(dev), y.float().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=conf.lr)
    gen = torch.Generator().manual_seed(seed)
    losses = []
    for step in range(conf.steps):
        idx = torch.randint(len(xd), (conf.batch,), generator=gen).to(dev)
        loss = (net(xd[idx]) - yd[idx]).square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 100 == 0:
            losses.append(float(loss.detach()))
    return net.eval(), losses


@torch.no_grad()
def predict(net: nn.Module, x: Tensor, dev: torch.device, batch: int = 8192) -> Tensor:
    """$g(x)$ ``(n, K)`` float32 on the CPU."""
    out = [net(c.float().to(dev)).cpu() for c in x.split(batch)]
    return torch.cat(out)


def fit_linear(x: Tensor, y: Tensor, batch: int = 8192) -> Tensor:
    r"""The least squares $[x, 1] B \approx y$ in float64: $B$ ``(D + 1,
    K)`` the minimum-norm solution of the normal equations (``gelsd``),
    accumulated ``batch`` rows at a time."""
    d = x.shape[1]
    xtx = torch.zeros(d + 1, d + 1, dtype=_F64)
    xty = torch.zeros(d + 1, y.shape[1], dtype=_F64)
    for xc, yc in zip(x.split(batch), y.split(batch), strict=True):
        a = torch.cat([xc.to(_F64), torch.ones(len(xc), 1, dtype=_F64)], 1)
        xtx += a.T @ a
        xty += a.T @ yc.to(_F64)
    return torch.linalg.lstsq(xtx, xty, driver="gelsd").solution


def predict_linear(b: Tensor, x: Tensor) -> Tensor:
    """$[x, 1] B$ ``(n, K)`` float32."""
    return (x.to(_F64) @ b[:-1] + b[-1]).float()


# -- the two quantities ----------------------------------------------------------------


@dataclass(frozen=True)
class ContentSums:
    r"""The blocks' sums of $E$ and $E^d$ (module docstring): ``err``,
    ``derr`` ``(S, B)`` of each readout seed, ``tot``, ``dtot`` ``(B,)``."""

    err: Tensor
    tot: Tensor
    derr: Tensor
    dtot: Tensor

    def num_den(self, change: bool) -> tuple[Tensor, Tensor]:
        """The seeds' errors ``(S, B)`` and the denominators ``(B,)`` of
        $E^d$ (``change``) or $E$."""
        return (self.derr, self.dtot) if change else (self.err, self.tot)

    def value(self, change: bool = False) -> float:
        """$E$ (or $E^d$) of the seeds' mean error."""
        num, den = self.num_den(change)
        return float(1 - num.mean(0).sum() / den.sum())

    def per_seed(self, change: bool = False) -> Tensor:
        """Each seed's $E$ (or $E^d$) ``(S,)``."""
        num, den = self.num_den(change)
        return 1 - num.sum(1) / den.sum()

    def paired(self) -> ContentSums:
        """The sums of adjacent blocks (256 s; an odd last block joined to the
        pair before it)."""
        return ContentSums(_pair(self.err), _pair(self.tot), _pair(self.derr),
                           _pair(self.dtot))  # fmt: skip

    def draws(self, counts: Tensor, change: bool = False) -> Tensor:
        r"""$E^*$ (or $E^{d*}$) ``(n,)`` of the draws' ``counts`` ``(n, B)``."""
        num, den = self.num_den(change)
        c = counts.to(_F64)
        return 1 - (c @ num.mean(0)) / (c @ den)

    def to_row(self) -> dict[str, list[float] | list[list[float]]]:
        return {"err_b": self.err.tolist(), "tot_b": self.tot.tolist(),
                "derr_b": self.derr.tolist(), "dtot_b": self.dtot.tolist()}  # fmt: skip

    @classmethod
    def of_row(cls, row: dict[str, object]) -> ContentSums:
        def t(k: str) -> Tensor:
            return torch.tensor(row[k], dtype=_F64)

        return cls(t("err_b"), t("tot_b"), t("derr_b"), t("dtot_b"))

    @classmethod
    def cat(cls, parts: Sequence[ContentSums]) -> ContentSums:
        """The blocks of ``parts`` side by side (a realization of segments)."""
        return cls(torch.cat([p.err for p in parts], 1), torch.cat([p.tot for p in parts]),
                   torch.cat([p.derr for p in parts], 1), torch.cat([p.dtot for p in parts]))  # fmt: skip


def _pair(b: Tensor) -> Tensor:
    n = b.shape[-1]
    to = torch.arange(n).div(2, rounding_mode="floor").clamp(max=max(n // 2 - 1, 0))
    return b.new_zeros(*b.shape[:-1], max(n // 2, 1)).index_add_(-1, to, b)


def content_sums(
    y: Tensor,
    preds: Sequence[Tensor],
    mean: Tensor,
    t: Tensor,
    blocks: Tensor,
    n_blocks: int,
    batch: int = 4096,
) -> ContentSums:
    r"""The sums (module docstring) at the anchors ``t`` ``(A,)`` (each with
    $t - 1 \ge 0$ of its video) of the blocks ``blocks`` ``(A,)``: ``y``
    ``(n, K)`` the observations of every time, ``preds`` each seed's
    $g(c)$ ``(n, K)``, ``mean`` $\bar y$ ``(K,)``. Float64."""

    def sq(v: Tensor) -> Tensor:
        return v.to(_F64).square().sum(1)

    acc = {"err": [], "tot": [], "derr": [], "dtot": []}
    m = mean.to(_F64)
    for tc in t.split(batch):
        y1, y0 = y[tc].to(_F64), y[tc - 1].to(_F64)
        acc["tot"].append(sq(y1 - m))
        acc["dtot"].append(sq(y1 - y0))
        d = y1 - y0
        acc["err"].append(torch.stack([sq(y1 - p[tc]) for p in preds]))
        acc["derr"].append(
            torch.stack([sq(d - (p[tc].to(_F64) - p[tc - 1].to(_F64))) for p in preds])
        )

    def bsum(v: Tensor) -> Tensor:
        return v.new_zeros(*v.shape[:-1], n_blocks).index_add_(-1, blocks, v)

    return ContentSums(
        bsum(torch.cat(acc["err"], -1)),
        bsum(torch.cat(acc["tot"])),
        bsum(torch.cat(acc["derr"], -1)),
        bsum(torch.cat(acc["dtot"])),
    )


def stack_seeds(parts: Sequence[ContentSums]) -> ContentSums:
    """One ``ContentSums`` of the seeds of ``parts`` (each of the same
    anchors and blocks)."""
    return ContentSums(torch.cat([p.err for p in parts]), parts[0].tot,
                       torch.cat([p.derr for p in parts]), parts[0].dtot)  # fmt: skip


# -- the ratio and its interval ------------------------------------------------------


def _log_ratio(ea: Tensor, eb: Tensor) -> Tensor:
    """$\\log_2(E_a / E_b)$: $-\\infty$ of $E_a \\le 0 < E_b$, NaN of $E_b
    \\le 0$."""
    out = torch.full_like(ea, math.nan)
    pos = eb > 0
    out[pos & (ea <= 0)] = -math.inf
    ok = pos & (ea > 0)
    out[ok] = torch.log2(ea[ok] / eb[ok])
    return out


def ratio_interval(
    a: ContentSums,
    b: ContentSums,
    counts: Tensor,
    counts2: Tensor,
    sigma: float,
    *,
    change: bool = False,
) -> dict[str, float | None]:
    r"""$\psi = \log_2(E_a / E_b)$ (of $E^d$ with ``change``) and its
    interval (module docstring): ``counts`` the draws of the blocks,
    ``counts2`` of their adjacent pairs, ``sigma`` $\sigma_{\mathrm{seed}}$.
    Returns ``point``, ``se`` ($\mathrm{SE}^*$), ``lo``, ``hi`` and the
    largest share of draws not finite (``nonfinite``); all None but
    ``nonfinite`` where $\hat E_b \le 0$."""
    ea, eb = a.value(change), b.value(change)
    sds, bad, undefined = [], 0.0, 0.0
    for sa, sb, c in ((a, b, counts), (a.paired(), b.paired(), counts2)):
        r = _log_ratio(sa.draws(c, change), sb.draws(c, change))
        fin = torch.isfinite(r)
        bad = max(bad, float((~fin).double().mean()))
        undefined = max(undefined, float(r.isnan().double().mean()))
        sds.append(float(r[fin].std()) if int(fin.sum()) >= 2 else math.nan)
    if not eb > 0:
        return {"point": None, "se": None, "lo": None, "hi": None, "nonfinite": bad}
    point = math.log2(ea / eb) if ea > 0 else -math.inf
    se = max(sds) if not any(math.isnan(s) for s in sds) else math.nan
    tot = math.sqrt((0.0 if math.isnan(se) else se * se) + sigma * sigma)
    lo, hi = point - 2 * tot, point + 2 * tot
    if bad > _TAIL or math.isnan(se):
        lo = -math.inf
    if undefined > _TAIL or (math.isnan(se) and point > -math.inf):
        hi = math.inf
    return {"point": point, "se": se, "lo": lo, "hi": hi, "nonfinite": bad}


def seed_sigma(a: ContentSums, b: ContentSums, *, change: bool = False) -> float:
    r"""$\sigma_{\mathrm{seed}}$: the sample standard deviation ($n - 1$) of
    the per-seed $\psi_s = \log_2(E_{a,s} / E_{b,s})$ (the seeds in the same
    order in both)."""
    r = _log_ratio(a.per_seed(change), b.per_seed(change))
    return float(r.std())


def comparable(
    b: ContentSums, counts: Tensor, counts2: Tensor, *, change: bool = False
) -> dict[str, float | bool]:
    r"""Whether $E_b$ (of $E^d$ with ``change``) is comparable (module
    docstring): its ``point``, $\mathrm{SE}^*$ (``se``, the larger of the two
    lengths'), the seeds' sample deviation (``seed_sd``), the lower end
    ``lo`` and ``ok`` ($lo > 0$)."""
    sds = [
        float(s.draws(c, change).std()) for s, c in ((b, counts), (b.paired(), counts2))
    ]
    per = b.per_seed(change)
    sd = float(per.std()) if len(per) >= 2 else 0.0
    point = b.value(change)
    lo = point - 2 * math.sqrt(max(sds) ** 2 + sd * sd)
    return {"point": point, "se": max(sds), "seed_sd": sd, "lo": lo, "ok": lo > 0}


def front_verdict(
    width: tuple[float | None, float | None],
    psi: dict[str, float | None],
    psi_d: dict[str, float | None],
    is_comparable: bool,
    delta_star: float,
    psi_star: float,
) -> str:
    r"""The front's verdict (module docstring) of the width's ratio
    interval ``width`` (of $\bar\Delta_1$; a None lower end below all) and
    the intervals ``psi``, ``psi_d`` of ``ratio_interval`` (a None end
    unbounded)."""
    lo_w = width[0]
    if lo_w is None or not lo_w > delta_star:
        return "width"
    if not is_comparable:
        return "incomparable"

    def lo(v: dict[str, float | None]) -> float:
        x = v["lo"]
        return -math.inf if x is None else x

    def hi(v: dict[str, float | None]) -> float:
        x = v["hi"]
        return math.inf if x is None else x

    if lo(psi) > -psi_star and lo(psi_d) > -psi_star:
        return "kept"
    if hi(psi) < -psi_star or hi(psi_d) < -psi_star:
        return "dropped"
    return "unresolved"
