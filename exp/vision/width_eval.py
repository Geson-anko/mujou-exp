r"""The linear horizon of a code (``plans/jepa.md``, 「線形の幅（2026-11-26）」;
README, 「線形の幅（回す前に決めたこと、2026-11-26）」, registered before any
number): of the triples of Zurich at every lag of 0.1 s up to 25.6 s, in
the whitened coordinates of 「V18c: 時間の線形性と情報の幅」, the share
$p_q(\Delta)$ of the triples whose constant-speed extrapolation beats the
copy ($\rho_t > 1 - 2q$; ``exp.lib.midpoint.wins``), its windows and widths
(``exp.lib.linwidth``), the pooled $G_{\mathrm{mid}}$ and its windows, and
the directions' widths $N_{\mathrm{lin}}(T)$. A description, not a verdict:
the wording is ``results/v18c/judge_width.py``'s.

**The sets** (``WidthSets``, ``width_sets``): the stacks, blocks and cuts of
``geom_eval``'s sets (``runs/v18c-tlin/sets.pt``), their rule spread over
the lags $n = 1 .. $ ``WidthConf.max_lag`` (``centres``: ``all`` the
centres whose triple's frames $u - n .. u + n + \text{span}$ lie in one
block, ``in`` those of no cut among them); ``common`` the ``in`` centres of
the largest lag, the same set at every lag (C1: a set changing with the lag
moves $p$ by its population alone); the pairs $(u, u + n)$, $n = 1 ..$
``max_pair``, of the same rule (``pairs``); and of ``strat_lags`` the
``in`` triples split in thirds by the pixels' displacement (``geom_eval``'s
``slow``, ``mid``, ``fast``; of the lags of the geom sets theirs).

**The rows of a whitened code** (``white_rows``): ``lin`` of each lag and
stratum (``common`` at every whitening; ``in`` and the strata at the main
one): the count, the blocks' count, $G$, $p_{1/4}$, $p_{1/2}$ and the
blocks' sums of ``exp.lib.midpoint.win_sums``; of ``common`` also the
sums of the blocks shifted by half a block (``*_b_s``; C5's sensitivity,
each centre in its own stack's shifted block), from ``main_lo`` on the
survivors' sums ``surv25_b``, ``surv50_b`` (the centres whose first run of
wins from ``main_lo`` on reaches the lag; ``linwidth.survival``) and the
quantiles of the margin $(\delta_t/4 - \nu_t)/\delta_t$ (``margin``);
``dpair`` of
each pair lag (``all``, ``in``): the blocks' sums of $\lVert w_{u+n} - w_u
\rVert^2$ and counts; ``dir`` (``direction_row``): the generalized
eigenproblem of the ``in`` triples of ``dir_lag`` on the even blocks
(``slow_directions``), each direction's pooled $G_i(\Delta)$ on the odd
blocks' ``common`` triples at the main lags, its width $W_i$ (the first
window's upper end of $G_i < 1/4$; 0 of none) and $N_{\mathrm{lin}}(T)$ at
``lin_ts``; ``spread`` $V$. A triple's block is its first stack's.

**The widths of rows** (``LinSums``, ``point``): $p$ and $G$ of the blocks'
sums; on the main range (lags ``main_lo`` .. ``max_lag``) the first window
and reappearances of $p_{1/4} \ge 1/2$ and of $G < 1/4$ (and of $1/2$),
$\Delta_{\mathrm{typ}}$, $\bar\Delta$, the survival's crossing, and below
the main range the windows of the whole grid (a description beside
``mean8``). The bootstrap (``resampled``, ``draws``) is ``geom_eval``'s: the
blocks drawn by ``metrics.block_counts`` (``boot``, ``boot_seed``) and their
adjacent pairs, the union of the two intervals; the shifted blocks' interval
beside them (a description).

**The checks** (``check_rows``, V-h0a .. V-h0m, synthetic, on the CPU), the
**yard** (``evaluate_yard``: the yardsticks of ``geom_eval``'s yard and
V-h0a on the real ``pca``), and the **runs** (``evaluate_run``,
``evaluate_vae``; the exploration, a description)::

    OMP_NUM_THREADS=1 python -m exp.vision.width_eval checks --out runs/v18c-width/checks
    python -m exp.vision.width_eval yard --sets runs/v18c-tlin --out runs/v18c-width \
        --config <v18c run> --init <..> --init-f <..> --init-m <..> --init-c <..> \
        --init-c1 <..> --init-v16 <..>
    python -m exp.vision.width_eval run <run_dir> --steps 6000,60000 --sets runs/v18c-tlin
    python -m exp.vision.width_eval vae <ckpt> --name v16 --sets runs/v18c-tlin \
        --out runs/v18c-width
"""

import argparse
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from exp.lib import ckpt
from exp.lib.linwidth import (
    Windows,
    draws_summary,
    mean_width,
    n_lin,
    paired_ratio,
    survival,
    union,
    windows,
)
from exp.lib.metrics import block_counts, contiguous_blocks, ratio_resamples
from exp.lib.midpoint import (
    TripleTerms,
    Whitener,
    block_sums,
    mid_covariances,
    pair_sq,
    slow_directions,
    triple_terms,
    win_sums,
    wins,
)
from exp.lib.provenance import configure_determinism
from exp.lib.synth_series import (
    chained_rotation,
    gauss_kernel,
    ou,
    ou_integral,
    ou_integral_d,
    random_walk,
    rotation,
    smooth,
    white,
)
from exp.lib.tours.stacks import tours_root
from exp.vision.eval import stream_view
from exp.vision.fill_eval import FillConf, fill_sets
from exp.vision.geom_eval import (  # the geom job's own helpers, reused as they are
    GeomConf,
    GeomSets,
    JepaPlaces,
    VaeTokenCode,
    _conf_of,
    _feature_code,
    _first_vae,
    _init_of,
    _jepa_of,
    _mean8,
    _sets_at,
    _stack_diffs,
    _start,
    _write,
    _zpca,
    encode_places,
    population_whitener,
    variants,
)
from exp.vision.jepa_diag import run_ckpts
from exp.vision.jepa_eval import PCA_SEED, ColorCode, PCACode, pixels, vae_at
from exp.vision.models import VAEModules, build_vae
from exp.vision.results.v18c.judge_geom import classify

__all__ = [
    "OUT",
    "TARGETS",
    "CheckConf",
    "LinSums",
    "WidthConf",
    "WidthSets",
    "check_rows",
    "code_rows",
    "conditions",
    "direction_row",
    "draws",
    "evaluate_checks",
    "evaluate_run",
    "evaluate_vae",
    "evaluate_yard",
    "lin_sums",
    "main",
    "point",
    "resampled",
    "slots",
    "synthetic_sets",
    "white_rows",
    "width_sets",
]

OUT = "posthoc_width"
TARGETS = {"ext": "future", "mid": "now"}  # each guard read's true candidate
YARD_INITS = ("init", "init_f", "init_m", "init_c", "init_c1")
QS = (0.25, 0.5)

type Row = dict[str, Any]
type Places = Callable[[Tensor], dict[str, Tensor]]


@dataclass(frozen=True)
class WidthConf:
    """The settings (``plans/jepa.md``), by default the plan's: lags in
    stacks of ``step`` seconds."""

    max_lag: int = 256  # 25.6 s
    max_pair: int = 512
    main_lo: int = 8  # 0.8 s
    step: float = 0.1
    strat_lags: tuple[int, ...] = (8, 12, 16, 24, 32, 48, 64, 128, 256)
    dir_lag: int = 8
    lin_ts: tuple[int, ...] = (8, 16, 32, 64, 128, 256)
    margin_qs: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)

    def seconds(self, lags: Tensor) -> Tensor:
        return lags.double() * self.step


def _qname(q: float) -> str:
    return str(round(100 * q))


# -- the sets ----------------------------------------------------------------------


@dataclass(frozen=True)
class WidthSets:
    """The sets (module docstring): ``n`` stacks of ``span`` frames past
    the first, ``frame_block`` ``(n + span,)`` each frame's block,
    ``before`` ``(n + span,)`` the count of cuts before each frame,
    ``blocks`` ``(n,)`` each stack's, ``strata`` the ``slow``,
    ``mid``, ``fast`` centres by lag, the sets' ``key``, and
    ``blocks_shift`` ``(n,)`` each stack's block of the blocks shifted by
    half a block (``shifted_blocks``; C5's sensitivity)."""

    n: int
    span: int
    frame_block: Tensor
    before: Tensor
    blocks: Tensor
    strata: Mapping[int, Mapping[str, Tensor]]
    key: str
    max_lag: int
    blocks_shift: Tensor

    @property
    def n_blocks(self) -> int:
        return int(self.frame_block.max()) + 1

    @property
    def n_blocks_shift(self) -> int:
        return int(self.blocks_shift.max()) + 1

    def centres(self, lag: int) -> tuple[Tensor, Tensor]:
        """The centres ``all`` and ``in`` of ``lag`` (module docstring)."""
        u = torch.arange(lag, max(lag, self.n - lag))
        u = u[self.frame_block[u - lag] == self.frame_block[u + lag + self.span]]
        crossed = self.before[u + lag + self.span] - self.before[u - lag] > 0
        return u, u[~crossed]

    @property
    def common(self) -> Tensor:
        """The ``in`` centres of the largest lag (every lag's triple of them
        is ``in``)."""
        return self.centres(self.max_lag)[1]

    def pairs(self, lag: int) -> tuple[Tensor, Tensor]:
        """The starts $u$ ``all`` and ``in`` of the pairs $(u, u + n)$."""
        u = torch.arange(0, max(0, self.n - lag))
        u = u[self.frame_block[u] == self.frame_block[u + lag + self.span]]
        crossed = self.before[u + lag + self.span] - self.before[u] > 0
        return u, u[~crossed]


def _before(cuts: Tensor, frames: int) -> Tensor:
    cut = torch.zeros(frames - 1, dtype=torch.long)
    cut[cuts] = 1
    return torch.cat([torch.zeros(1, dtype=torch.long), cut.cumsum(0)])


def shifted_blocks(frames: int, block: int) -> Tensor:
    """Each frame's block ``(frames,)`` of ``block``-frame blocks from the
    frame ``block // 2`` on (``metrics.contiguous_blocks``; the frames
    before it joined to the first block)."""
    half = block // 2
    rest = contiguous_blocks(torch.zeros(frames - half, dtype=torch.long), block)
    return torch.cat([torch.zeros(half, dtype=torch.long), rest])


def synthetic_sets(n: int, span: int, block: int, max_lag: int) -> WidthSets:
    """The sets of a synthetic series of ``n`` stacks: ``block``-frame
    blocks (``metrics.contiguous_blocks``), no cut, no strata."""
    f = n + span
    fb = contiguous_blocks(torch.zeros(f, dtype=torch.long), block)
    return WidthSets(n, span, fb, _before(torch.zeros(0, dtype=torch.long), f),
                     fb[:n].clone(), {}, "synthetic", max_lag,
                     shifted_blocks(f, block)[:n].clone())  # fmt: skip


def width_sets(
    sets: GeomSets,
    gconf: GeomConf,
    wconf: WidthConf,
    frames: Tensor | None = None,
    *,
    stack: int = 8,
    stride: int = 1,
    dev: torch.device = torch.device("cpu"),
) -> WidthSets:
    """The sets of ``geom_eval``'s ``sets`` (module docstring); with the
    video's ``frames`` u8 also the strata of ``strat_lags`` beyond the geom
    sets' lags (the stacks of ``stack`` frames ``stride`` apart)."""
    f = sets.n + sets.span
    fb = contiguous_blocks(torch.zeros(f, dtype=torch.long), gconf.block)
    ws = WidthSets(sets.n, sets.span, fb, _before(sets.cuts, f), sets.blocks, {},
                   sets.key, wconf.max_lag,
                   shifted_blocks(f, gconf.block)[: sets.n].clone())  # fmt: skip
    strata: dict[int, dict[str, Tensor]] = {}
    for lag in wconf.strat_lags:
        if lag in sets.triples:
            strata[lag] = {s: sets.triples[lag][s] for s in ("slow", "mid", "fast")}
        elif frames is not None:
            strata[lag] = _strata(ws, frames, lag, stack, stride, dev)
    return replace(ws, strata=strata)


def _strata(
    ws: WidthSets, frames: Tensor, lag: int, stack: int, stride: int, dev: torch.device
) -> dict[str, Tensor]:
    """``geom_eval.geom_sets``'s split of the ``in`` triples of ``lag``."""
    _, inside = ws.centres(lag)
    diffs = _stack_diffs(frames, lag, stack, stride, dev)
    m = (diffs[inside - lag] + diffs[inside]) / 2
    order = torch.sort(m, stable=True)
    rank = torch.empty_like(order.indices)
    rank[order.indices] = torch.arange(len(rank))
    third = 3 * rank // max(len(rank), 1)
    return {"slow": inside[third == 0], "mid": inside[third == 1],
            "fast": inside[third == 2]}  # fmt: skip


def slots(t: Tensor, lag: int, reach: int = 4) -> dict[str, Tensor]:
    r"""The places of the anchors $t$ ``(A,)`` at ``lag``: ``past`` $t - n$,
    ``now`` $t$, ``future`` $t + n$; ``ext`` the extrapolation's candidates
    $t + n + jn$ and ``mid`` the midpoint's $t + jn$, $j = -$``reach``
    .. ``reach`` ``(A, 2 reach + 1)`` (the true one in the middle column)."""
    j = torch.arange(-reach, reach + 1) * lag
    return {"past": t - lag, "now": t, "future": t + lag,
            "ext": (t + lag)[:, None] + j, "mid": t[:, None] + j}  # fmt: skip


def conditions(c: Tensor, sl: Mapping[str, Tensor]) -> dict[str, Tensor]:
    r"""The guard's conditions ``(A, D)`` of the codes ``c`` ``(n, D)`` at
    the places ``sl`` (``slots``): ``past``, ``now``, ``future``, ``ext``
    $2c_t - c_{t-\Delta}$ and ``mid`` $\tfrac12(c_{t-\Delta} +
    c_{t+\Delta})$; ``ext``'s candidates are read against ``TARGETS["ext"]``
    and ``mid``'s against ``TARGETS["mid"]`` (the middle columns of
    ``sl["ext"]``, ``sl["mid"]``)."""
    past, now, fut = c[sl["past"]], c[sl["now"]], c[sl["future"]]
    return {"past": past, "now": now, "future": fut, "ext": 2 * now - past,
            "mid": (past + fut) / 2}  # fmt: skip


# -- the rows of a whitened code -----------------------------------------------------


def _lin_row(
    t_sub: TripleTerms, blocks: Tensor, nb: int, lag: int, stratum: str
) -> Row:
    sums = win_sums(t_sub, blocks, nb, QS)
    n_t = len(t_sub.num)
    row: Row = {"part": "lin", "lag": lag, "stratum": stratum, "n_triples": n_t,
                "n_blocks": len(blocks.unique())}  # fmt: skip
    if n_t:
        row |= {
            "G": float(t_sub.num.sum() / t_sub.den.sum()),
            "p25": float(sums["win25_b"].sum()) / n_t,
            "p50": float(sums["win50_b"].sum()) / n_t,
        }
    return row | {k: v.tolist() for k, v in sums.items()}


SHIFT_KEYS = ("num_b", "den_b", "n_b", "win25_b", "win50_b")


def _subset(t: TripleTerms, pick: Tensor) -> TripleTerms:
    return TripleTerms(t.num[pick], t.den[pick], t.a2[pick], t.b2[pick])


def white_rows(w: Tensor, ws: WidthSets, wconf: WidthConf, *, full: bool) -> list[Row]:
    """The rows (module docstring) of the whitened codes ``w`` ``(n, k)`` of
    the sets' stacks; with ``full`` (the main whitening) also ``in``, the
    strata, ``dpair`` and ``dir``."""
    nb, nbs = ws.n_blocks, ws.n_blocks_shift
    common = ws.common
    rows: list[Row] = []
    won: dict[float, list[Tensor]] = {q: [] for q in QS}
    common_rows: list[Row] = []
    for lag in range(1, wconf.max_lag + 1):
        u_all, u_in = ws.centres(lag)
        t_all = triple_terms(w, u_all, lag)
        subsets = {"common": common}
        if full:
            subsets["in"] = u_in
            subsets |= dict(ws.strata.get(lag, {}))
        for stratum, centres in subsets.items():
            t = _subset(t_all, torch.searchsorted(u_all, centres))
            row = _lin_row(t, ws.blocks[centres - lag], nb, lag, stratum)
            if stratum == "common":
                shift = win_sums(t, ws.blocks_shift[centres], nbs, QS)
                row |= {f"{k}_s": shift[k].tolist() for k in SHIFT_KEYS}
                if lag >= wconf.main_lo:
                    for q in QS:
                        won[q].append(wins(t, q))
                moving = t.den > 0
                margin = (t.den[moving] / 4 - t.num[moving]) / t.den[moving]
                if len(margin):
                    qs = torch.tensor(wconf.margin_qs, dtype=torch.float64)
                    row["margin"] = torch.quantile(margin, qs).tolist()
                common_rows.append(row)
            rows.append(row)
    blocks = ws.blocks[common - wconf.max_lag]
    for q in QS:
        alive = survival(torch.stack(won[q], 1)).double()  # (Nc, L)
        sums = block_sums(alive, blocks, nb)  # (B, L)
        for i, row in enumerate(r for r in common_rows if r["lag"] >= wconf.main_lo):
            row[f"surv{_qname(q)}_b"] = sums[:, i].tolist()
    if full:
        rows += _pair_rows(w, ws, wconf, nb)
        rows.append(direction_row(w, ws, wconf))
    return rows


def _pair_rows(w: Tensor, ws: WidthSets, wconf: WidthConf, nb: int) -> list[Row]:
    rows = []
    for lag in range(1, wconf.max_pair + 1):
        u_all, u_in = ws.pairs(lag)
        sq = pair_sq(w, u_all, lag)
        for stratum, u in (("all", u_all), ("in", u_in)):
            pick = torch.searchsorted(u_all, u)
            blocks = ws.blocks[u]
            rows.append({"part": "dpair", "lag": lag, "stratum": stratum,
                         "sq_b": block_sums(sq[pick], blocks, nb).tolist(),
                         "n_b": block_sums(torch.ones(len(u)), blocks, nb).tolist()})  # fmt: skip
    return rows


def direction_row(w: Tensor, ws: WidthSets, wconf: WidthConf) -> Row:
    """``dir`` (module docstring): the directions of the even blocks'
    ``in`` triples of ``dir_lag`` and their widths on the odd blocks'
    ``common`` triples."""
    _, u_in = ws.centres(wconf.dir_lag)
    even = u_in[ws.blocks[u_in - wconf.dir_lag] % 2 == 0]
    lam, v = slow_directions(*mid_covariances(w, even, wconf.dir_lag))
    y = w.double() @ v  # (n, k): each direction's 1-d series
    common = ws.common
    odd = common[ws.blocks[common - wconf.max_lag] % 2 == 1]
    lags = torch.arange(wconf.main_lo, wconf.max_lag + 1)
    g = torch.empty(len(lags), y.shape[1], dtype=torch.float64)
    for i, lag in enumerate(lags.tolist()):
        y0, y1, y2 = y[odd - lag], y[odd], y[odd + lag]
        num = (y1 - (y0 + y2) / 2).square().sum(0)
        den = ((y1 - y0).square() + (y2 - y1).square()).sum(0) / 2
        g[i] = num / den
    x = wconf.seconds(lags)
    widths = []
    for j in range(y.shape[1]):
        hi = windows(x, g[:, j], 0.25, below=True).typical
        widths.append(0.0 if hi is None else hi)
    ts = [t * wconf.step for t in wconf.lin_ts]
    return {
        "part": "dir",
        "lam": lam.tolist(),
        "g": g.T.tolist(),
        "W": widths,
        "n_lin": dict(
            zip(
                [str(t) for t in wconf.lin_ts],
                n_lin(torch.tensor(widths), ts),
                strict=True,
            )
        ),  # fmt: skip
        "n_even": len(even),
        "n_odd": len(odd),
    }


def code_rows(
    z_eval: Tensor,
    z_ref: Tensor,
    ws: WidthSets,
    gconf: GeomConf,
    wconf: WidthConf,
    *,
    sens: bool,
    labels: Mapping[str, Any],
    population: bool = False,
) -> list[Row]:
    """The rows of a code's evaluation ``z_eval`` ``(n, D)`` and reference
    ``z_ref`` ``(m, D)`` at each of ``geom_eval.variants`` (the main one
    ``full``; a whitening of a code not scoreable has its spectrum only),
    each with ``labels``, ``k``, ``eps``, ``ref`` and the sets' ``key``;
    with ``population`` also the synthetic control's population whitening
    (``ref`` ``population``, full)."""
    rows: list[Row] = []
    half = len(z_ref) // 2
    refs = {"all": z_ref, "front": z_ref[:half], "back": z_ref[half : 2 * half]}
    todo = [(Whitener.fit(refs[v.ref], v.k, c_eps=v.c_eps), v.ref, v.c_eps, i == 0)
            for i, v in enumerate(variants(gconf, sens=sens))]  # fmt: skip
    if population:
        d = z_eval.shape[1]
        todo.append(
            (population_whitener(d, gconf.c_eps), "population", gconf.c_eps, True)
        )
    for wh, ref, eps, full in todo:
        lab = dict(labels) | {"k": wh.k, "eps": eps, "ref": ref, "key": ws.key}
        rows.append(lab | {"part": "spec"} | wh.spec())
        if not wh.scoreable:
            continue
        w = wh.white(z_eval)
        x = w - w.mean(0)
        rows.append(lab | {"part": "spread", "V": float(x.square().sum(1).mean())})
        rows += [lab | r for r in white_rows(w, ws, wconf, full=full)]
    return rows


# -- the widths of rows ------------------------------------------------------------


@dataclass(frozen=True)
class LinSums:
    """The ``lin`` rows of one stratum as tensors: ``lags`` ``(L,)``, the
    blocks' sums ``num``, ``den``, ``n``, ``win`` (by ``"25"``, ``"50"``,
    ``"asym"``) ``(L, B)``, and ``surv`` (by ``"25"``, ``"50"``) of the lags
    from ``main_lo`` (empty if not ``common``)."""

    lags: Tensor
    num: Tensor
    den: Tensor
    n: Tensor
    win: Mapping[str, Tensor]
    surv: Mapping[str, Tensor] = field(default_factory=dict)

    def p(self, q: str) -> Tensor:
        return self.win[q].sum(1) / self.n.sum(1)

    def g(self) -> Tensor:
        return self.num.sum(1) / self.den.sum(1)

    def restrict(self, lo: int) -> LinSums:
        keep = self.lags >= lo
        return LinSums(self.lags[keep], self.num[keep], self.den[keep],
                       self.n[keep], {k: v[keep] for k, v in self.win.items()},
                       self.surv)  # fmt: skip


def _f64(values: Any) -> Tensor:
    return torch.tensor(values, dtype=torch.float64)


def lin_sums(
    rows: Sequence[Row], stratum: str = "common", *, shift: bool = False,
    **labels: Any,
) -> LinSums:  # fmt: skip
    """The ``LinSums`` of the ``lin`` rows of ``stratum`` and ``labels``
    (such as ``place``, ``k``, ``eps``, ``ref``) in order of lag; with
    ``shift`` of the shifted blocks' sums (``common``; no ``asym``, no
    survivors).

    Raises:
        ValueError: No row matches, or two rows match at a lag (``labels``
            do not name one whitening).
    """
    got = sorted(
        (r for r in rows if r.get("part") == "lin" and r["stratum"] == stratum
         and all(r.get(k) == v for k, v in labels.items())),
        key=lambda r: r["lag"],
    )  # fmt: skip
    if not got:
        raise ValueError(f"no lin rows of {stratum} and {labels}")
    lags = [r["lag"] for r in got]
    if len(set(lags)) != len(lags):
        raise ValueError(f"two lin rows at a lag of {stratum} and {labels}")
    if shift:
        sh = {k: _f64([r[f"{k}_s"] for r in got]) for k in SHIFT_KEYS}
        return LinSums(torch.tensor(lags), sh["num_b"], sh["den_b"], sh["n_b"],
                       {"25": sh["win25_b"], "50": sh["win50_b"]})  # fmt: skip
    win = {q: _f64([r[f"win{q}_b"] for r in got]) for q in ("25", "50", "asym")}
    surv = {}
    if "surv25_b" in got[-1]:
        surv = {q: _f64([r[f"surv{q}_b"] for r in got if f"surv{q}_b" in r])
                for q in ("25", "50")}  # fmt: skip
    return LinSums(torch.tensor([r["lag"] for r in got]), _f64([r["num_b"] for r in got]),
                   _f64([r["den_b"] for r in got]), _f64([r["n_b"] for r in got]),
                   win, surv)  # fmt: skip


def _span(w: Windows) -> dict[str, Any]:
    def one(s: Any) -> dict[str, Any] | None:
        return None if s is None else asdict(s)

    return {"first": one(w.first), "again": [asdict(s) for s in w.again]}


def point(ls: LinSums, wconf: WidthConf) -> dict[str, Any]:
    r"""The point estimates (module docstring) of ``ls`` (all the lags):
    of each bound $q$ the main range's windows of $p_q$ and of $G$,
    $\Delta_{\mathrm{typ}}$ (``typ``), $\bar\Delta$ (``mean``), the
    survival's window (``surv``), and the whole grid's windows
    (``whole``, the description below the main range)."""
    main = ls.restrict(wconf.main_lo)
    x = wconf.seconds(main.lags)
    xa = wconf.seconds(ls.lags)
    out: dict[str, Any] = {"p_asym": ls.p("asym").tolist()}
    for q in ("25", "50"):
        fq = int(q) / 100
        p, g = main.p(q), main.g()
        wp = windows(x, p, 0.5, below=False)
        out[q] = {
            "p": _span(wp),
            "G": _span(windows(x, g, fq, below=True)),
            "typ": wp.typical,
            "mean": mean_width(x, p),
            "whole_p": _span(windows(xa, ls.p(q), 0.5, below=False)),
            "whole_G": _span(windows(xa, ls.g(), fq, below=True)),
        }
        if q in ls.surv:
            s = ls.surv[q].sum(1) / main.n.sum(1)
            out[q]["surv"] = _span(windows(x, s, 0.5, below=False))
            out[q]["surv_mean"] = mean_width(x, s)
    return out


def resampled(ls: LinSums, counts: Tensor) -> tuple[dict[str, Tensor], Tensor]:
    """The resampled $p$ (by ``"25"``, ``"50"``) and $G$ ``(n, L)`` of the
    draws' ``counts`` ``(n, B)`` (``metrics.ratio_resamples``)."""
    return (
        {q: ratio_resamples(ls.win[q], ls.n, counts) for q in ("25", "50")},
        ratio_resamples(ls.num, ls.den, counts),
    )


def _paired(b_sums: Tensor) -> Tensor:
    """The sums ``(..., B // 2)`` of adjacent blocks (``geom_eval``'s)."""
    b = b_sums.shape[-1]
    to = torch.arange(b).div(2, rounding_mode="floor").clamp(max=b // 2 - 1)
    return b_sums.new_zeros(*b_sums.shape[:-1], b // 2).index_add_(-1, to, b_sums)


def paired_blocks(ls: LinSums) -> LinSums:
    """``ls`` of 256 s blocks: adjacent blocks' sums."""
    return LinSums(ls.lags, _paired(ls.num), _paired(ls.den), _paired(ls.n),
                   {k: _paired(v) for k, v in ls.win.items()},
                   {k: _paired(v) for k, v in ls.surv.items()})  # fmt: skip


def counts_of(n_blocks: int, gconf: GeomConf) -> Tensor:
    """The draws of ``geom_eval``'s bootstrap of ``n_blocks`` blocks."""
    return block_counts(
        n_blocks, gconf.boot, torch.Generator().manual_seed(gconf.boot_seed)
    )


def draws(
    ls: LinSums, wconf: WidthConf, gconf: GeomConf, shifted: LinSums | None = None
) -> dict[str, Any]:
    r"""The intervals of the main range's widths (``point``'s) of 128 s and
    256 s blocks and their union: of each bound, $p$'s window and $\bar\Delta$
    (``linwidth.draws_summary``) and the pooled $G$'s window; with
    ``shifted`` (the blocks shifted by half a block) its 128 s interval of
    $p$ beside them (``shift``, a description)."""
    main = ls.restrict(wconf.main_lo)
    x = wconf.seconds(main.lags)
    out: dict[str, Any] = {}
    per = {}
    for name, part in (("128", main), ("256", paired_blocks(main))):
        ps, g = resampled(part, counts_of(part.n.shape[1], gconf))
        per[name] = {
            q: {
                "p": asdict(draws_summary(x, ps[q], 0.5, below=False)),
                "G": asdict(draws_summary(x, g, int(q) / 100, below=True)),
            }
            for q in ("25", "50")
        }
    out["by_block"] = per
    for q in ("25", "50"):
        out[q] = {
            "typ": union([per[b][q]["p"]["typ"] for b in per]),
            "mean": union([per[b][q]["p"]["mean"] for b in per]),
            "G_hi": union([per[b][q]["G"]["typ"] for b in per]),
            "none": max(per[b][q]["p"]["none"] for b in per),
            "G_none": max(per[b][q]["G"]["none"] for b in per),
        }
    if shifted is not None:
        sm = shifted.restrict(wconf.main_lo)
        ps, _ = resampled(sm, counts_of(sm.n.shape[1], gconf))
        out["shift"] = {q: asdict(draws_summary(x, ps[q], 0.5, below=False))
                        for q in ("25", "50")}  # fmt: skip
    return out


# -- the checks V-h0 ---------------------------------------------------------------


@dataclass(frozen=True)
class CheckConf:
    r"""The settings of V-h0 (``check_rows``; ``plans/jepa.md``): series of
    ``n`` stacks of ``span`` frames past the first and ``d`` columns,
    periods in steps, the seeds of each check. The rotations' periods are
    $N/5$ steps of an $N$ prime to 30, so off the grid (registered): no lag
    $\le 256$ is a whole number of turns (no triple still) nor 1/6 or 1/4 of
    a turn from one ($\Delta = N(6j \pm 1)/30$ and $N(4j \pm 1)/20$ are not
    whole), and every triple's win is the closed one."""

    n: int = 38989
    span: int = 7
    d: int = 64
    seed: int = 1001
    rotation_period: float = 123.4  # 12.34 s
    chain_periods: tuple[float, ...] = (63.4, 93.4, 123.4, 183.4, 243.4)
    band_period: float = 200.0  # 20 s
    band_nu: float = 1 / 32
    band_sigma: float = 400.0  # V-h0e's band: the envelopes' Gaussian kernel
    white_dirs: tuple[int, ...] = (3, 4)
    white_dirs_half: tuple[int, ...] = (21, 22)
    ou_tau: float = 32.0
    integral_tau: float = 32.0
    thin_period: float = 246.8
    thin_n: int = 38393  # n + span a whole number of blocks (reversible)
    leak_dims: int = 8
    cover_seeds: tuple[int, int] = (1101, 1301)
    cover_truth_seed: int = 1100
    cover_long: int = 100
    cover_min: float = 0.90
    known_rs: tuple[float, ...] = (1.0, math.sqrt(2), 2.0)
    known_seeds: tuple[int, int] = (1301, 1501)
    known_truth_seed: int = 1600
    leak_taus: tuple[float, float] = (2.0, 64.0)
    known_max: int = 13
    delta_star: float = 0.5
    ou8_taus: tuple[float, ...] = (8.0, 80.0)
    rel: float = 1e-12
    tie: float = 1e-12


def _check(name: str, ok: bool, **values: Any) -> Row:
    return {"part": "check", "name": name, "pass": bool(ok)} | values


def _record(name: str, **values: Any) -> Row:
    return {"part": "record", "name": name} | values


def _band_noise(
    n: int, d: int, period: float, nu: float, gen: torch.Generator
) -> Tensor:
    r"""One frequency in every plane of the columns $2j$, $2j + 1$
    (``rotation``, its own phase by ``gen``) plus white noise of variance
    ``nu``. In a plane the band's $\langle a, b \rangle$, $\lVert a
    \rVert^2$, $\lVert b \rVert^2$ are of no time: a column of its own phase
    alone adds $\cos(2\omega t + 2\phi)$ terms, so a triple's $\rho_t$ (and
    $p$, $\bar\Delta$) would follow the realization's phases, a quantity no
    block bootstrap of one realization sees."""
    return rotation(n, d, period, gen) + math.sqrt(nu) * white(n, d, gen)


def _band_noise_window(period: float, nu: float, q: float) -> tuple[float, float]:
    r"""The pooled window of $G_\nu < q$ in steps (``midpoint``'s injection
    of one frequency, $G = s/2$): $s \in q \pm \sqrt{q^2 - (\tfrac32 -
    2q)\nu}$, $s = 1 - \cos(2\pi n / T)$."""
    root = math.sqrt(q * q - (1.5 - 2 * q) * nu)
    lo, hi = (period / (2 * math.pi) * math.acos(1 - s) for s in (q - root, q + root))
    return lo, hi


def _band_columns(
    n: int, d: int, period: float, sigma: float, gen: torch.Generator
) -> Tensor:
    r"""``(n, d)`` independent columns of one band: $a_t \cos \omega t + b_t
    \sin \omega t$, $\omega = 2\pi$ / ``period``, of the slow envelopes
    $a$, $b$ (``smooth`` of the Gaussian kernel of ``sigma`` steps, unit
    variance): $\rho(n) = \rho_a(n) \cos \omega n$, $\rho_a(n) = e^{-n^2 /
    (2\sigma^2)}$, nearly one frequency at the lags of the window. One
    frequency in every column with its own phase (as ``_band_noise``) is of
    rank 2 over time, its $C_\delta$ singular for the directions."""
    k = gauss_kernel(sigma)
    a, b = smooth(n, d, k, gen), smooth(n, d, k, gen)
    t = torch.arange(n, dtype=torch.float64)[:, None] * (2 * math.pi / period)
    return a * t.cos() + b * t.sin()


def _common_sums(x: Tensor, ws: WidthSets, wconf: WidthConf) -> LinSums:
    """The ``common`` sums of the main range only (V-h0i, V-h0k)."""
    common = ws.common
    blocks = ws.blocks[common - wconf.max_lag]
    nb = ws.n_blocks
    lags = list(range(wconf.main_lo, wconf.max_lag + 1))
    acc: dict[str, list[Tensor]] = {"num_b": [], "den_b": [], "n_b": [], "win25_b": [],
                                    "win50_b": [], "winasym_b": []}  # fmt: skip
    for lag in lags:
        s = win_sums(triple_terms(x, common, lag), blocks, nb, QS)
        for k in acc:
            acc[k].append(s[k])
    st = {k: torch.stack(v) for k, v in acc.items()}
    return LinSums(torch.tensor(lags), st["num_b"], st["den_b"], st["n_b"],
                   {"25": st["win25_b"], "50": st["win50_b"], "asym": st["winasym_b"]})  # fmt: skip


def _cat(parts: Sequence[LinSums]) -> LinSums:
    return LinSums(parts[0].lags, torch.cat([p.num for p in parts], 1),
                   torch.cat([p.den for p in parts], 1), torch.cat([p.n for p in parts], 1),
                   {k: torch.cat([p.win[k] for p in parts], 1) for k in parts[0].win})  # fmt: skip


def _between(x: float | None, lo: float, hi: float) -> bool:
    return x is not None and lo < x < hi


def _grid_cell(v: float, step: float) -> tuple[float, float]:
    """The grid's interval $(k\\,\\text{step}, (k + 1)\\,\\text{step})$
    holding ``v`` (seconds)."""
    k = math.floor(v / step)
    return k * step, (k + 1) * step


def _contains(iv: tuple[float | None, float | None], v: float) -> bool:
    lo, hi = iv
    return hi is not None and (lo is None or lo <= v) and v <= hi


def check_rows(
    cc: CheckConf = CheckConf(),
    gconf: GeomConf = GeomConf(),
    wconf: WidthConf = WidthConf(),
    *,
    long: bool = True,
) -> list[Row]:
    """V-h0a .. V-h0m (``plans/jepa.md``; module docstring), synthetic, on
    the CPU: rows of ``part`` ``check`` (a failure stops) and ``record``;
    ``long`` the coverage (V-h0i) and the known differences (V-h0k)."""
    ws = synthetic_sets(cc.n, cc.span, gconf.block, wconf.max_lag)
    rows: list[Row] = []
    rows += _vh0a(cc, ws, wconf)
    rows += _vh0b(cc, ws, wconf)
    rows += _vh0c(cc, ws, wconf)
    rows += _vh0d(cc, ws, wconf)
    rows += _vh0e(cc, ws, wconf)
    rows += _vh0f(cc, ws, wconf)
    rows += _vh0g(cc, ws, wconf)
    rows += _vh0h(cc, gconf, wconf)
    rows += _vh0j(cc, ws, wconf)
    rows += _vh0l(cc, ws, wconf)
    rows += _vh0m(cc, ws, gconf, wconf)
    if long:
        rows += _vh0i(cc, ws, gconf, wconf)
        rows += _vh0k(cc, ws, gconf, wconf)
    return rows


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def identity_rows(w: Tensor, ws: WidthSets, cc: CheckConf, code: str) -> list[Row]:
    r"""V-h0a of the whitened ``w``: per triple $\nu_t / \delta_t = (1 -
    \rho_t)/2$ with $\rho_t$ of the vectors $a$, $b$ themselves; the pooled
    $G = 1 - \tilde D(2\Delta) / (4 \bar D)$ of the same triples' squared
    displacements; the win $\nu_t < q\delta_t$ is $\rho_t > 1 - 2q$ (the
    triples within ``tie`` of the bound counted and left out). Both sides of
    the first lie in $[0, 1]$: its error is of that range."""
    rows = []
    x = w.double()
    for lag in (1, 8, 32, 128, 256):
        u = ws.centres(lag)[1]
        if lag > ws.max_lag or not len(u):
            continue
        t = triple_terms(x, u, lag)
        a, b = x[u] - x[u - lag], x[u + lag] - x[u]
        moving = t.den > 0
        r = ((a * b).sum(1) / t.den)[moving]
        per = (t.num / t.den)[moving]
        worst = float((per - (1 - r) / 2).abs().max())
        far = pair_sq(x, u - lag, 2 * lag)
        g_d = float(1 - far.sum() / (4 * t.den.sum()))
        g = float(t.num.sum() / t.den.sum())
        flips, ties = 0, 0
        for q in QS:
            near = (r - (1 - 2 * q)).abs() < cc.tie
            ties += int(near.sum())
            flips += int(((wins(t, q)[moving] != (r > 1 - 2 * q)) & ~near).sum())
        ok = worst <= cc.rel and abs(g - g_d) <= cc.rel * abs(g) and flips == 0
        rows.append(_check("V-h0a", ok, code=code, lag=lag, worst_rel=worst, G=g,
                           G_of_D=g_d, flips=flips, ties=ties))  # fmt: skip
    return rows


def _vh0a(cc: CheckConf, ws: WidthSets, wconf: WidthConf) -> list[Row]:
    x = ou(cc.n, cc.d, math.exp(-1 / cc.ou_tau), _gen(cc.seed))
    return identity_rows(x, ws, cc, "ou")


def _rotation_truth(periods: Tensor, lag: int, half: float = 1 / 6) -> Tensor:
    r"""Of rotations of ``periods`` ``(P,)`` (steps) at ``lag``: whether
    every triple wins, 1 where $\Delta / T$ is less than ``half`` from an
    integer (1/6 at the bound 1/4, 1/4 at 1/2).

    Raises:
        ValueError: $\Delta / T$ is (within $10^{-6}$) an integer, the triple
            still and its win rounding's, or ``half`` from one, a tie
            (``CheckConf``'s periods are of neither).
    """
    f = lag / periods
    near = (f - f.round()).abs()
    if bool(((near < 1e-6) | ((near - half).abs() < 1e-6)).any()):
        raise ValueError(f"a period of {periods.tolist()} is on the grid at {lag}")
    return (near < half).double()


def _vh0b(cc: CheckConf, ws: WidthSets, wconf: WidthConf) -> list[Row]:
    r"""The rotation: $p$ and the survival exact; the windows and
    $\bar\Delta$ of the true $p$ to $10^{-10}$; the first window's upper
    ends between 2.0 and 2.1 s and 3.0 and 3.1 s (q = 1/4, 1/2)."""
    x = rotation(cc.n, cc.d, cc.rotation_period, _gen(cc.seed + 1))
    ls = lin_sums(white_rows(x, ws, wconf, full=False))
    pt = point(ls, wconf)
    main = ls.restrict(wconf.main_lo)
    period = torch.tensor([cc.rotation_period], dtype=torch.float64)
    x_s = wconf.seconds(main.lags)
    ok, out = True, {}
    for q, half in (("25", 1 / 6), ("50", 1 / 4)):
        want = torch.cat([_rotation_truth(period, lag, half)
                          for lag in main.lags.tolist()])  # fmt: skip
        exact = torch.equal(main.p(q), want)
        w = windows(x_s, want, 0.5, below=False)
        m_want = mean_width(x_s, want)
        typ = pt[q]["typ"]
        same = (
            typ is not None and w.typical is not None
            and abs(typ - w.typical) <= 1e-10 * w.typical
            and [(a["lo"], a["hi"]) for a in pt[q]["p"]["again"]]
            == [(a.lo, a.hi) for a in w.again]
            and abs(pt[q]["mean"] - m_want) <= 1e-10 * m_want
        )  # fmt: skip
        ok = ok and exact and same
        out[q] = {"p_exact": exact, "typ": typ, "typ_want": w.typical,
                  "again": pt[q]["p"]["again"], "mean": pt[q]["mean"],
                  "mean_want": m_want}  # fmt: skip
    surv = ls.surv["25"].sum(1) / main.n.sum(1)
    surv_ok = torch.equal(surv, (main.lags.double() < cc.rotation_period / 6).double())
    ok = (ok and surv_ok and _between(out["25"]["typ"], 2.0, 2.1)
          and _between(out["50"]["typ"], 3.0, 3.1))  # fmt: skip
    return [_check("V-h0b", ok, surv_exact=surv_ok, by_q=out)]


def _vh0c(cc: CheckConf, ws: WidthSets, wconf: WidthConf) -> list[Row]:
    r"""The chained rotations: $p$ the common centres' share of the blocks
    that win and the survival exact, the first window's upper end and the
    survival's crossing between 2.0 and 2.1 s, $\bar\Delta$ of the true $p$
    and $0.8 + \int S$ of the true $S$ to $10^{-10}$."""
    x = chained_rotation(ws.blocks, cc.d, cc.chain_periods, _gen(cc.seed + 2))
    ls = lin_sums(white_rows(x, ws, wconf, full=False))
    pt = point(ls, wconf)
    main = ls.restrict(wconf.main_lo)
    common = ws.common
    per_block = torch.bincount(ws.blocks[common], minlength=ws.n_blocks).double()
    periods = torch.tensor([cc.chain_periods[b % len(cc.chain_periods)]
                            for b in range(ws.n_blocks)], dtype=torch.float64)  # fmt: skip
    weight = per_block / per_block.sum()
    p_want = torch.stack([(weight * _rotation_truth(periods, lag)).sum()
                          for lag in main.lags.tolist()])  # fmt: skip
    s_want = torch.stack([(weight * (lag < periods / 6).double()).sum()
                          for lag in main.lags.tolist()])  # fmt: skip
    p = main.p("25")
    s = ls.surv["25"].sum(1) / main.n.sum(1)
    x_s = wconf.seconds(main.lags)
    exact = bool((p - p_want).abs().max() <= 1e-12)
    exact = exact and bool((s - s_want).abs().max() <= 1e-12)
    m_want, s_mean_want = mean_width(x_s, p_want), mean_width(x_s, s_want)
    typ, surv_w = pt["25"]["typ"], pt["25"]["surv"]["first"]
    ok = (exact and _between(typ, 2.0, 2.1) and surv_w is not None
          and 2.0 < surv_w["hi"] < 2.1
          and abs(pt["25"]["mean"] - m_want) <= 1e-10 * m_want
          and abs(pt["25"]["surv_mean"] - s_mean_want) <= 1e-10 * s_mean_want)  # fmt: skip
    return [_check("V-h0c", ok, p_exact=exact, typ=typ, surv_hi=surv_w and surv_w["hi"],
                   mean=pt["25"]["mean"], mean_want=m_want,
                   surv_mean=pt["25"]["surv_mean"], surv_mean_want=s_mean_want)]  # fmt: skip


def _vh0d(cc: CheckConf, ws: WidthSets, wconf: WidthConf) -> list[Row]:
    x = _band_noise(cc.n, cc.d, cc.band_period, cc.band_nu, _gen(cc.seed + 3))
    pt = point(lin_sums(white_rows(x, ws, wconf, full=False)), wconf)
    lo, hi = (
        v * wconf.step for v in _band_noise_window(cc.band_period, cc.band_nu, 0.25)
    )
    first = pt["25"]["G"]["first"]
    ok = (first is not None and not first["lo_cut"] and not first["hi_cut"]
          and _grid_cell(lo, wconf.step)[0] < first["lo"] < _grid_cell(lo, wconf.step)[1]
          and _grid_cell(hi, wconf.step)[0] < first["hi"] < _grid_cell(hi, wconf.step)[1])  # fmt: skip
    return [_check("V-h0d", ok, window=first, want=(lo, hi))]


def _vh0e(cc: CheckConf, ws: WidthSets, wconf: WidthConf) -> list[Row]:
    r"""A band and ``m`` white directions: the pooled window of 1/4 exists of
    the fewer (3) and not of the more (4); $N_{\mathrm{lin}}(T) = 64 - m$
    where $T < T_0/6$ (a band direction's window, $G = s/2 < 1/4$) and 0
    beyond. Of 21 and 22 white directions the window of 1/2 is recorded."""
    rows = []
    for m in cc.white_dirs + cc.white_dirs_half:
        g = _gen(cc.seed + 4 + m)
        band = _band_columns(cc.n, cc.d - m, cc.band_period, cc.band_sigma, g)
        x = torch.cat([band, white(cc.n, m, g)], 1)
        rs = white_rows(x, ws, wconf, full=m in cc.white_dirs)
        pt = point(lin_sums(rs), wconf)
        if m in cc.white_dirs_half:
            rows.append(_record("V-h0e-half", m=m, window=pt["50"]["G"]["first"]))
            continue
        has = pt["25"]["G"]["first"] is not None
        (d_row,) = (r for r in rs if r["part"] == "dir")
        inside = [t for t in wconf.lin_ts if t < cc.band_period / 6]
        beyond = [t for t in wconf.lin_ts if t > cc.band_period / 6]
        ok_n = all(d_row["n_lin"][str(t)] == cc.d - m for t in inside) and all(
            d_row["n_lin"][str(t)] == 0 for t in beyond
        )
        rows.append(_check("V-h0e-window", has == (m == min(cc.white_dirs)), m=m,
                           window=pt["25"]["G"]["first"]))  # fmt: skip
        rows.append(_check("V-h0e-nlin", ok_n, m=m, n_lin=d_row["n_lin"], inside=inside,
                           beyond=beyond, widths=sorted(d_row["W"])))  # fmt: skip
    return rows


def _vh0f(cc: CheckConf, ws: WidthSets, wconf: WidthConf) -> list[Row]:
    rows = []
    g = _gen(cc.seed + 5)
    series = {
        "ou": (ou(cc.n, cc.d, math.exp(-1 / cc.ou_tau), g), ("25", "50")),
        "white": (white(cc.n, cc.d, g), ("25", "50")),
        "walk": (random_walk(cc.n, cc.d, g), ("25",)),
    }
    for name, (x, bounds) in series.items():
        pt = point(lin_sums(white_rows(x, ws, wconf, full=False)), wconf)
        for q in ("25", "50"):
            none = pt[q]["p"]["first"] is None and pt[q]["G"]["first"] is None
            if q in bounds:
                rows.append(_check("V-h0f", none, code=name, q=q, p=pt[q]["p"],
                                   G=pt[q]["G"]))  # fmt: skip
            else:
                rows.append(_record("V-h0f-walk-half", code=name, q=q, p=pt[q]["p"],
                                    G=pt[q]["G"], mean=pt[q]["mean"]))  # fmt: skip
    return rows


def _vh0g(cc: CheckConf, ws: WidthSets, wconf: WidthConf) -> list[Row]:
    r = math.exp(-1 / cc.integral_tau)
    x = ou_integral(cc.n, cc.d, r, _gen(cc.seed + 6))
    pt = point(lin_sums(white_rows(x, ws, wconf, full=False)), wconf)
    lo, hi = 1.0, 4 * cc.integral_tau  # the discrete root of G = 1/4
    for _ in range(200):
        mid = (lo + hi) / 2
        g = 1 - ou_integral_d(2 * mid, r) / (4 * ou_integral_d(mid, r))
        lo, hi = (mid, hi) if g < 0.25 else (lo, mid)
    want = lo * wconf.step
    cell = _grid_cell(want, wconf.step)
    first, half = pt["25"]["G"]["first"], pt["50"]["G"]["first"]
    ok = (first is not None and cell[0] < first["hi"] < cell[1]
          and half is not None and half["hi_cut"])  # fmt: skip
    return [_check("V-h0g", ok, hi=first and first["hi"], want=want,
                   want_continuous=1.1514 * cc.integral_tau * wconf.step,
                   half=half)]  # fmt: skip


def _vh0h(cc: CheckConf, gconf: GeomConf, wconf: WidthConf) -> list[Row]:
    """The reversal on sets of whole blocks (each stack's mirror is a stack
    of the mirrored block) and the thinning of a rotation by 2."""
    ws = synthetic_sets(cc.thin_n, cc.span, gconf.block, wconf.max_lag)
    x = ou(cc.thin_n, cc.d, math.exp(-1 / cc.ou_tau), _gen(cc.seed + 7))
    f = lin_sums(white_rows(x, ws, wconf, full=False))
    b = lin_sums(white_rows(x.flip(0), ws, wconf, full=False))
    same_n = torch.equal(f.n.sum(1), b.n.sum(1))
    same_w = all(torch.equal(f.win[q].sum(1), b.win[q].sum(1)) for q in ("25", "50"))
    same_s = all(torch.equal(f.surv[q].sum(1), b.surv[q].sum(1)) for q in ("25", "50"))
    rel = float(((f.g() - b.g()).abs() / f.g().abs()).max())
    pf, pb = point(f, wconf), point(b, wconf)
    # p and S are equal sums, so their windows and widths are equal to the
    # bit; G is equal to rounding: the same runs, the ends to 1e-9
    same_widths = all(
        pf[q][k] == pb[q][k]
        for q in ("25", "50") for k in ("p", "typ", "mean", "surv", "surv_mean")
    )  # fmt: skip
    same_g = all(_same_spans(pf[q]["G"], pb[q]["G"], 1e-9) for q in ("25", "50"))
    ok = same_n and same_w and same_s and rel <= cc.rel and same_widths and same_g
    rows = [_check("V-h0h-reverse", ok, rel_G=rel, wins_equal=same_w,
                   surv_equal=same_s, widths_equal=same_widths, G_windows=same_g)]  # fmt: skip
    rot = rotation(cc.thin_n, cc.d, cc.thin_period, _gen(cc.seed + 8))
    thin = rot[::2]
    lags = torch.arange(wconf.main_lo // 2, wconf.max_lag // 2 + 1)

    def p_of(x: Tensor, lag: int) -> Tensor:
        t = triple_terms(x, torch.arange(lag, len(x) - lag), lag)
        return wins(t, 0.25).double().mean()

    p_full = torch.stack([p_of(rot, 2 * n) for n in lags.tolist()])
    p_thin = torch.stack([p_of(thin, n) for n in lags.tolist()])
    w_full = windows(2 * lags.double(), p_full, 0.5, below=False).typical
    w_thin = windows(lags.double(), p_thin, 0.5, below=False).typical
    ok = (torch.equal(p_full, p_thin) and w_full is not None and w_thin is not None
          and abs(w_full - 2 * w_thin) <= cc.rel * w_full)  # fmt: skip
    rows.append(_check("V-h0h-thin", ok, steps_full=w_full, steps_thin=w_thin))
    return rows


def _same_spans(a: Mapping[str, Any], b: Mapping[str, Any], rel: float) -> bool:
    """Two ``_span``s of the same runs, the ends equal to ``rel``."""
    sa = ([a["first"]] if a["first"] else []) + list(a["again"])
    sb = ([b["first"]] if b["first"] else []) + list(b["again"])
    return len(sa) == len(sb) and all(
        x["lo_cut"] == y["lo_cut"] and x["hi_cut"] == y["hi_cut"]
        and abs(x["lo"] - y["lo"]) <= rel * x["lo"]
        and abs(x["hi"] - y["hi"]) <= rel * x["hi"]
        for x, y in zip(sa, sb, strict=True)
    )  # fmt: skip


def _vh0j(cc: CheckConf, ws: WidthSets, wconf: WidthConf) -> list[Row]:
    rows = []
    for tau in cc.ou8_taus:
        x = ou(cc.n + 7, cc.d, math.exp(-1 / tau), _gen(cc.seed + 9))
        x8 = x.unfold(0, 8, 1).mean(2)  # (n, d): the mean of 8 consecutive
        pt = point(lin_sums(white_rows(x8, ws, wconf, full=False)), wconf)
        rows.append(_record("V-h0j", tau=tau, whole_p=pt["25"]["whole_p"],
                            whole_G=pt["25"]["whole_G"]))  # fmt: skip
    return rows


def _vh0l(cc: CheckConf, ws: WidthSets, wconf: WidthConf) -> list[Row]:
    r"""The places by the frame's number, through the guard's own code
    (``slots``, ``conditions``, ``TARGETS``): $w_u = (u, (u/100)^2)$, the
    first coordinate the time. The conditions' times by hand (``ext`` at $t
    + \Delta$, ``mid`` at $t$), each read's true candidate (the middle
    column) the place of its target and the extrapolation's miss of the
    second coordinate $2\Delta^2/10^4$, four times the triple's $\nu_t$."""
    u = torch.arange(ws.n, dtype=torch.float64)
    w = torch.stack([u, (u / 100) ** 2], 1)
    rows = []
    for lag in (1, 8, 64):
        c = ws.centres(lag)[1]
        c = c[(c >= 4 * lag) & (c + 5 * lag < ws.n)][:50]
        sl = slots(c, lag)
        cond = conditions(w, sl)
        cd = c.double()
        times = {"past": cd - lag, "now": cd, "future": cd + lag, "ext": cd + lag,
                 "mid": cd}  # fmt: skip
        ok = all(torch.equal(cond[k][:, 0], v) for k, v in times.items())
        for read in ("ext", "mid"):
            middle = sl[read][:, sl[read].shape[1] // 2]
            ok = ok and torch.equal(middle, sl[TARGETS[read]])
            ok = ok and torch.equal(w[middle, 0], cond[read][:, 0])
        ok = (ok and torch.equal(w[sl["ext"][:, 0], 0], cd - 3 * lag)
              and torch.equal(w[sl["mid"][:, 8], 0], cd + 4 * lag))  # fmt: skip
        t = triple_terms(w, c, lag)
        hand = (w[sl["future"]] - cond["ext"]).square().sum(1)
        ok = ok and bool(((4 * t.num - hand).abs() <= 1e-9 * hand.clamp(min=1)).all())
        ok = ok and bool(((hand - (2 * lag**2 / 1e4) ** 2).abs() <= 1e-9).all())
        rows.append(_check("V-h0l", ok, lag=lag))
    return rows


def _ou_columns(n: int, taus: Tensor, gen: torch.Generator) -> Tensor:
    """``(n, len(taus))`` independent unit OUs of the time constants
    ``taus`` (steps), as ``synth_series.ou`` column by column."""
    r = torch.exp(-1 / taus.double())
    e = white(n, len(taus), gen)
    x = torch.empty_like(e)
    x[0] = e[0]
    c = (1 - r * r).sqrt()
    for t in range(1, n):
        x[t] = r * x[t - 1] + c * e[t]
    return x


def _pooled_ou_g(taus: Tensor, weights: Tensor, lag: int) -> float:
    r"""The pooled $G$ of independent OUs of ``taus`` and variances
    ``weights``: $1 - \sum_i w_i (1 - r_i^{2n}) / (4 \sum_i w_i (1 - r_i^n))$
    (``midpoint``'s $G = 1 - D(2n) / 4D(n)$)."""
    r = torch.exp(-1 / taus.double())
    w = weights.double()
    return float(1 - (w * (1 - r ** (2 * lag))).sum() / (4 * (w * (1 - r**lag)).sum()))


def _vh0m(cc: CheckConf, ws: WidthSets, gconf: GeomConf, wconf: WidthConf) -> list[Row]:
    r"""The whitening of the reference alone: an evaluation of OUs of
    time constants ``leak_taus`` (log-spaced over its ``leak_dims``
    columns) and scales 4 .. 1, a white reference of scales 1 .. 4. The
    reference's whitening weighs column $i$ by $(s^{\mathrm{ev}}_i /
    s^{\mathrm{ref}}_i)^2$ (16 .. 1/16, the fast columns most), the
    evaluation's by 1: the pooled $G$ moves by the closed difference of
    ``_pooled_ou_g``. The rows' $G$ is the reference whitening's to
    ``rel``; whitened anew on the evaluation it moves the closed way (the
    lag of the largest closed move: the same sign and over half of it)."""
    g = _gen(cc.seed + 10)
    k = cc.leak_dims
    scale_ref = torch.linspace(1, 4, k, dtype=torch.float64)
    scale_ev = torch.linspace(4, 1, k, dtype=torch.float64)
    lo, hi = cc.leak_taus
    taus = torch.logspace(math.log10(lo), math.log10(hi), k, dtype=torch.float64)
    ref = white(cc.n, k, g) * scale_ref
    ev = _ou_columns(cc.n, taus, g) * scale_ev
    gc = replace(gconf, k=k, k_side=())
    rows = code_rows(ev, ref, ws, gc, wconf, sens=False, labels={})
    got = lin_sums(rows, k=k, eps=gc.c_eps, ref="all").g()
    want = lin_sums(
        white_rows(
            Whitener.fit(ref, k, c_eps=gc.c_eps).white(ev), ws, wconf, full=False
        )
    ).g()
    other = lin_sums(
        white_rows(Whitener.fit(ev, k, c_eps=gc.c_eps).white(ev), ws, wconf, full=False)
    ).g()
    rel = float(((got - want).abs() / want.abs()).max())
    w_ref = (scale_ev / scale_ref) ** 2
    closed = torch.tensor([_pooled_ou_g(taus, torch.ones(k), lag) - _pooled_ou_g(taus, w_ref, lag)
                           for lag in range(1, wconf.max_lag + 1)])  # fmt: skip
    i = int(closed.abs().argmax())
    moved = float(other[i] - want[i])
    ok_moved = moved * float(closed[i]) > 0 and abs(moved) > abs(float(closed[i])) / 2
    return [_check("V-h0m", rel <= cc.rel and ok_moved, rel=rel, lag=i + 1,
                   refit_moved=moved, closed_move=float(closed[i]))]  # fmt: skip


def _long_sums(make: Callable[[int], Tensor], parts: int, ws: WidthSets,
               wconf: WidthConf) -> LinSums:  # fmt: skip
    """A realization of ``parts`` segments (``make(i)``, each of the sets'
    layout) as one: the blocks of every segment side by side."""
    return _cat([_common_sums(make(i), ws, wconf) for i in range(parts)])


def _band_truth(cc: CheckConf, ws: WidthSets, wconf: WidthConf, period: float,
                gen: torch.Generator) -> dict[str, Any]:  # fmt: skip
    """``point`` of the band and noise of ``period`` on ``cover_long``
    independent segments (``_band_noise``: of no phase, so the segments are
    of one process)."""
    return point(
        _long_sums(
            lambda _: _band_noise(cc.n, cc.d, period, cc.band_nu, gen),
            cc.cover_long, ws, wconf,
        ),
        wconf,
    )  # fmt: skip


def _vh0i(cc: CheckConf, ws: WidthSets, gconf: GeomConf, wconf: WidthConf) -> list[Row]:
    truth = _band_truth(cc, ws, wconf, cc.band_period, _gen(cc.cover_truth_seed))
    hi_true = _band_noise_window(cc.band_period, cc.band_nu, 0.25)[1] * wconf.step
    t_mean, t_typ = truth["25"]["mean"], truth["25"]["typ"]
    hits = {"G_hi": 0, "mean": 0, "typ": 0}
    n = 0
    for seed in range(*cc.cover_seeds):
        x = _band_noise(cc.n, cc.d, cc.band_period, cc.band_nu, _gen(seed))
        d = draws(_common_sums(x, ws, wconf), wconf, gconf)["25"]
        hits["G_hi"] += _contains(d["G_hi"], hi_true)
        hits["mean"] += _contains(d["mean"], t_mean)
        hits["typ"] += t_typ is not None and _contains(d["typ"], t_typ)
        n += 1
    share = {k: v / n for k, v in hits.items()}
    # the Monte Carlo error of a share near 0.95 of n draws
    mc = math.sqrt(0.95 * 0.05 / n)
    return [_check("V-h0i", all(v >= cc.cover_min for v in share.values()),
                   share=share, n=n, truth={"G_hi": hi_true, "mean": t_mean, "typ": t_typ},
                   mc_se=mc)]  # fmt: skip


def _vh0k(cc: CheckConf, ws: WidthSets, gconf: GeomConf, wconf: WidthConf) -> list[Row]:
    """The known differences: of each ``known_rs`` the pairs' wording by
    ``judge_geom.classify`` (the rule of the comparisons) and the true
    $\\log_2$ ratio of $\\bar\\Delta$ (``_band_truth`` of each period)."""
    rows = []
    main = torch.arange(wconf.main_lo, wconf.max_lag + 1)
    x_s = wconf.seconds(main)
    g_t = _gen(cc.known_truth_seed)
    true_mean = {r: _band_truth(cc, ws, wconf, cc.band_period * r, g_t)["25"]["mean"]
                 for r in cc.known_rs}  # fmt: skip
    for r in cc.known_rs:
        distinct = 0
        for seed in range(*cc.known_seeds):
            g = _gen(seed)
            a = _band_noise(cc.n, cc.d, cc.band_period, cc.band_nu, g)
            b = _band_noise(cc.n, cc.d, cc.band_period * r, cc.band_nu, g)
            la, lb = _common_sums(a, ws, wconf), _common_sums(b, ws, wconf)
            ivs = []
            for pa, pb in ((la, lb), (paired_blocks(la), paired_blocks(lb))):
                counts = counts_of(pa.n.shape[1], gconf)
                ra = ratio_resamples(pa.win["25"], pa.n, counts)
                rb = ratio_resamples(pb.win["25"], pb.n, counts)
                ivs.append(paired_ratio(x_s, ra, rb, 0.5).mean)
            lo, hi = union(ivs)
            distinct += classify(-math.inf if lo is None else lo,
                                 math.inf if hi is None else hi,
                                 cc.delta_star) == "distinct"  # fmt: skip
        n = cc.known_seeds[1] - cc.known_seeds[0]
        true = math.log2(true_mean[1.0] / true_mean[r]) if 1.0 in true_mean else None
        if r == 1.0:
            rows.append(_check("V-h0k", distinct <= cc.known_max, r=r, distinct=distinct,
                               n=n, true_ratio=true))  # fmt: skip
        else:
            rows.append(_record("V-h0k-power", r=r, distinct=distinct, n=n,
                                true_ratio=true, true_means=[true_mean[1.0], true_mean[r]]))  # fmt: skip
    return rows


# -- the jobs ----------------------------------------------------------------------


def evaluate_checks(
    out: Path,
    cc: CheckConf = CheckConf(),
    gconf: GeomConf = GeomConf(),
    wconf: WidthConf = WidthConf(),
) -> Path:
    """V-h0 into ``out`` (``config.json``, ``rows.json``) on the CPU."""
    determinism = configure_determinism("cpu")
    _start(out, {"check": asdict(cc), "geom": asdict(gconf), "width": asdict(wconf)},
           determinism)  # fmt: skip
    _write(out / "rows.json", check_rows(cc, gconf, wconf), {})
    return out


def places_of(
    fn: Places, zurich: Any, ref: Tensor, n: int, dev: torch.device
) -> dict[str, tuple[Tensor, Tensor]]:
    """The places of ``fn`` of Zurich's ``n`` stacks and of the reference
    (``geom_eval``'s encoding)."""
    ev = encode_places(fn, zurich.stacks, n, dev)
    rf = encode_places(fn, lambda idx: ref[idx], len(ref), dev)
    return {p: (ev[p], rf[p]) for p in ev}


def sets_and_ref(
    conf: Any, sets_dir: Path, root: Path | None, gconf: GeomConf, wconf: WidthConf,
    fconf: FillConf, spacing: Mapping[str, Any] | None, span: float | None,
) -> tuple[WidthSets, Any, Tensor, Path]:  # fmt: skip
    root = tours_root(conf.data.tours.data_version) if root is None else root
    sets, zurich = _sets_at(sets_dir, conf, root, gconf)
    t = conf.data.tours
    ws = width_sets(sets, gconf, wconf, zurich.frames, stack=t.frames, stride=t.stride,
                    dev=torch.device(conf.device))  # fmt: skip
    kw: dict[str, Any] = {} if span is None else {"span": span}
    ref = fill_sets(conf, root, fconf, spacing=spacing, **kw).fit
    return ws, zurich, ref, root


def evaluate_yard(
    conf: Any,
    sets_dir: Path,
    out: Path,
    *,
    inits: Mapping[str, Path],
    init_v16: Path,
    root: Path | None = None,
    gconf: GeomConf = GeomConf(),
    wconf: WidthConf = WidthConf(),
    fconf: FillConf = FillConf(),
    spacing: Mapping[str, Any] | None = None,
    span: float | None = None,
) -> Path:
    """V-h1: the yardsticks' rows (``out / "yard/rows/<code>.json"``: ``pca``,
    ``color``, ``mean8``, the step 0 of ``inits`` (``YARD_INITS``) and V16's
    VAE of $t = 0$, each place, the sensitivity's whitenings too) on the geom
    sets of ``sets_dir``, and V-h0a on the real ``pca`` (``out /
    "yard/vh.json"``). Returns ``out``.

    Raises:
        ValueError: A run of ``YARD_INITS`` is not given (or as
            ``geom_eval``'s yard).
    """
    missing = [n for n in YARD_INITS if n not in inits]
    if missing:
        raise ValueError(f"the yard needs the runs {missing}")
    determinism = configure_determinism(conf.device)
    dev = torch.device(conf.device)
    ws, zurich, ref, _ = sets_and_ref(conf, sets_dir, root, gconf, wconf, fconf,
                                       spacing, span)  # fmt: skip
    yard = out / "yard"
    sources = {k: str(v) for k, v in inits.items()} | {"init_v16": str(init_v16)}
    _start(yard, {"geom": asdict(gconf), "width": asdict(wconf), "key": ws.key,
                  "sources": sources}, determinism)  # fmt: skip

    def emit(code: str, places: Mapping[str, tuple[Tensor, Tensor]],
             labels: Mapping[str, Any] | None = None) -> None:  # fmt: skip
        rows: list[Row] = []
        for place, (ev, rf) in places.items():
            lab = {"code": code, "place": place} | dict(labels or {})
            rows += code_rows(ev, rf, ws, gconf, wconf, sens=True, labels=lab)
        _write(yard / "rows" / f"{code}.json", rows, {})

    gen = torch.Generator().manual_seed(conf.seed + PCA_SEED)
    pca_code = PCACode.fit(pixels, ref, gconf.pca_dim, gen=gen, device=conf.device)
    pca = places_of(_feature_code(pca_code.mu), zurich, ref, ws.n, dev)
    emit("pca", pca)
    emit("color", places_of(_feature_code(ColorCode(16).mu), zurich, ref, ws.n, dev))
    gen = torch.Generator().manual_seed(conf.seed + PCA_SEED)
    mean8 = PCACode.fit(_mean8, ref, gconf.pca_dim, gen=gen, device=conf.device)
    emit("mean8", places_of(_feature_code(mean8.mu), zurich, ref, ws.n, dev))
    for name in YARD_INITS:
        mods, own, path = _init_of(inits[name], conf.device)
        places = places_of(JepaPlaces(mods).places, zurich, ref, ws.n, dev)
        if own == "zpca":
            places["own"] = _zpca(places["own"], _conf_of(inits[name], None).seed,
                                  gconf.pca_dim)  # fmt: skip
        emit(name, places, {"source": str(path), "t": float(path.stem)})
    first = _first_vae(init_v16)
    vae = vae_at(first, conf.device)
    assert isinstance(vae, VAEModules)
    emit("init_v16", places_of(VaeTokenCode(vae).places, zurich, ref, ws.n, dev),
         {"source": str(first), "t": float(first.stem)})  # fmt: skip
    ev, rf = pca["own"]
    w = Whitener.fit(rf, gconf.k, c_eps=gconf.c_eps).white(ev)
    checks = identity_rows(w, ws, CheckConf(), "pca")
    _write(yard / "vh.json", checks, {"key": ws.key})
    return out


def _apply(fn: Places, zurich: Any, ref: Tensor, ws: WidthSets, dev: torch.device,
           labels: Mapping[str, Any], gconf: GeomConf, wconf: WidthConf,
           zpca_seed: int | None = None) -> list[Row]:  # fmt: skip
    places = places_of(fn, zurich, ref, ws.n, dev)
    if zpca_seed is not None:
        places["own"] = _zpca(places["own"], zpca_seed, gconf.pca_dim)
    rows: list[Row] = []
    for place, (ev, rf) in places.items():
        rows += code_rows(ev, rf, ws, gconf, wconf, sens=False,
                          labels=dict(labels) | {"place": place})  # fmt: skip
    return rows


def evaluate_run(
    run_dir: Path,
    *,
    steps: Sequence[int],
    sets_dir: Path,
    root: Path | None = None,
    device: str | None = None,
    gconf: GeomConf = GeomConf(),
    wconf: WidthConf = WidthConf(),
    fconf: FillConf = FillConf(),
    spacing: Mapping[str, Any] | None = None,
    span: float | None = None,
) -> Path:
    """The rows of the JEPA run's checkpoints of ``steps`` into ``run_dir /
    OUT / "rows/<step>.json"`` (a step whose file is there is skipped), on
    the geom sets of ``sets_dir``. Returns the directory.

    Raises:
        ValueError: A step has no checkpoint, or the sets or the
            ``config.json`` there are of other settings.
    """
    conf = _conf_of(run_dir, device)
    ckpts = run_ckpts(run_dir)
    missing = [s for s in steps if s not in ckpts]
    if missing:
        raise ValueError(f"no checkpoint of steps {missing} (of {list(ckpts)})")
    determinism = configure_determinism(conf.device)
    ws, zurich, ref, _ = sets_and_ref(conf, sets_dir, root, gconf, wconf, fconf,
                                       spacing, span)  # fmt: skip
    mods, own = _jepa_of(run_dir, conf.device)
    out = run_dir / OUT
    _start(out, {"geom": asdict(gconf), "width": asdict(wconf), "key": ws.key,
                 "run": run_dir.name}, determinism)  # fmt: skip
    for step in steps:
        path = out / "rows" / f"{step:06d}.json"
        if path.exists():
            continue
        ckpt.load_eval(ckpts[step], mods.modules())
        rows = _apply(JepaPlaces(mods).places, zurich, ref, ws, torch.device(conf.device),
                      {"code": run_dir.name, "step": step}, gconf, wconf,
                      conf.seed if own == "zpca" else None)  # fmt: skip
        _write(
            path,
            rows,
            {"run": run_dir.name, "step": step, "t": float(ckpts[step].stem)},
        )
    return out


def evaluate_vae(
    path: Path,
    *,
    name: str,
    sets_dir: Path,
    out: Path,
    root: Path | None = None,
    device: str | None = None,
    gconf: GeomConf = GeomConf(),
    wconf: WidthConf = WidthConf(),
    fconf: FillConf = FillConf(),
    spacing: Mapping[str, Any] | None = None,
    span: float | None = None,
) -> Path:
    """The rows of the VAE checkpoint ``path`` (its run's ``config.yaml`` two
    levels up; read on ``eval.stream_view`` as ``geom_eval.evaluate_vae``)
    into ``out / "vae/rows/<name>.json"``. Returns the file."""
    conf = _conf_of(path.parent.parent, device)
    view, _ = stream_view(conf)
    determinism = configure_determinism(conf.device)
    ws, zurich, ref, _ = sets_and_ref(view, sets_dir, root, gconf, wconf, fconf,
                                       spacing, span)  # fmt: skip
    mods = build_vae(conf, seed=conf.seed)
    ckpt.load_eval(path, mods.modules())
    vae_out = out / "vae"
    _start(vae_out, {"geom": asdict(gconf), "width": asdict(wconf), "key": ws.key},
           determinism)  # fmt: skip
    rows = _apply(VaeTokenCode(mods).places, zurich, ref, ws, torch.device(conf.device),
                  {"code": name, "source": str(path)}, gconf, wconf)  # fmt: skip
    target = vae_out / "rows" / f"{name}.json"
    _write(target, rows, {"t": float(path.stem)})
    return target


def _ints(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split(",") if x)


def main(argv: Sequence[str] | None = None) -> Path:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = p.add_subparsers(dest="job", required=True)
    checks = sub.add_parser("checks")
    checks.add_argument("--out", type=Path, required=True)
    yard = sub.add_parser("yard")
    yard.add_argument("--config", type=Path, required=True)
    yard.add_argument("--out", type=Path, required=True)
    for name in YARD_INITS:
        yard.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    yard.add_argument("--init-v16", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("run_dir", type=Path)
    run.add_argument("--steps", type=_ints, required=True)
    vae = sub.add_parser("vae")
    vae.add_argument("ckpt", type=Path)
    vae.add_argument("--name", required=True)
    vae.add_argument("--out", type=Path, required=True)
    for q in (yard, run, vae):
        q.add_argument("--sets", type=Path, required=True)
        q.add_argument("--device", default=None)
        q.add_argument("--root", type=Path, default=None)
    a = p.parse_args(argv)
    if a.job == "checks":
        return evaluate_checks(a.out)
    if a.job == "yard":
        conf = _conf_of(a.config, a.device)
        inits = {n: getattr(a, n) for n in YARD_INITS}
        return evaluate_yard(conf, a.sets, a.out, inits=inits, init_v16=a.init_v16,
                             root=a.root)  # fmt: skip
    if a.job == "run":
        return evaluate_run(a.run_dir, steps=a.steps, sets_dir=a.sets, root=a.root,
                            device=a.device)  # fmt: skip
    return evaluate_vae(a.ckpt, name=a.name, sets_dir=a.sets, out=a.out, root=a.root,
                        device=a.device)  # fmt: skip


if __name__ == "__main__":
    main()
