r"""V18c's identification of the fill in the JEPA's own space
(``plans/jepa.md``, 「V18c: 自分の表現空間での補完の同定」; README, the section
of the same name, registered before any number): whether the predictor's
$\hat t_Q$ of a hidden quarter $Q_q$ picks the target side's own $t_Q$ out of
other frames' at the same place, in the space the predictor is trained in.

The sets (``self_sets``): the reference ``ref`` (``fill_eval.fill_sets``'s
fit stacks: training videos, not Zurich) fixes every centre, the bands'
basis and the transplants' contexts $j$ (the first stack of each of the first
``SelfConf.contexts`` videos in the order of the fit set); the grid of
Zurich (``gen_eval.Zurich``, 1 s apart) holds the candidates: ``easy``
(``FillConf.n_cand`` evenly spread, chance $1/n$, each other's candidates)
and ``hard`` (each query's own and its neighbours $\pm 2 \dots \pm 32$ s,
``gen_eval.hard_sets``).

A region's vector $v$ is $Q_q$'s fine positions' tokens in order of position
($4 s^2 d$ wide). The stages of a mask (``MASKS``, ``fill_eval.e1f_masks``:
$Q_q$ always hidden, $k$ = 4, 8, 12): ``pred`` the predictor's $\hat t_p$
($p \in Q_q$), ``pool`` the context side's tokens after the projection
averaged over the visible positions and put at every $p$, ``flat``
``pred``'s positions replaced by their mean, ``interp`` the target side's
tokens of the stack whose hidden pixels are filled harmonically from the
visible (``regions.harmonic_matrix``), ``oracle`` the target side's own
$t_p$, ``const`` 0. Each stage less its mean on ``ref`` (by stage, quarter
and mask; the real less its own by quarter): $e_n$, $r_m$, never centred on
the queries or the candidates (Codex's counterexample). The scores
(``stage_scores``): the pairs' win rate $A$ (``metrics.pair_wins``, a tie
1/2, chance 1/2) and ``top1`` (the share of $A_n = 1$: no candidate as
near) of the cosine and of ``l2``; ``hard_c`` the hard candidates each less
the mean of the query's 33.

``swap``: under ``S_1``, ``S_2`` (two diagonal quarters $A$, $B$ hidden),
$d_n = (p_{A,n} - p_{B,n}) - \overline{(p_A - p_B)}_{\mathrm{ref}}$ and
the value $[\langle d_n, r_{A,n} \rangle > 0] + \frac12 [\cdot = 0]$
(``metrics.swap``), the mean of both pairs and directions: exactly 1/2 for a
replicated K3, whose $p_A - p_B$ is 0 to the bit.

The transplant (``regions.transplant``): $x_{j \leftarrow m}$ (``fwd``: the
context $j$ with $Q_q$'s pixels of the grid's $m$) and $x_{m \leftarrow j}$
(``rev``) through the target side; their $v_Q$, less their mean over $m$ of
``ref``, are the candidates of the natural queries (``oracle_tr`` the
natural real). The locality (``local_sums``, ``local_terms``): with $\Delta_p
= t_p(x_{j \leftarrow m}) - t_p(x_j)$, $\delta_p = t_p(x_m) - t_p(x_j)$
over the grid's $m$, $a_U = \sum \langle \Delta_p, \delta_p \rangle /
\sum \lVert \delta_p \rVert^2$ over $p \in Q$, $a_V$ the same outside, $L
= a_U - a_V$, $\rho_V = \sum \lVert \Delta_p \rVert^2 / \sum \lVert
\delta_p \rVert^2$ outside, $\eta_U = \sum \lVert \Delta_p - \delta_p
\rVert^2 / \sum \lVert \delta_p \rVert^2$ inside, and $a_U$, $a_V$ of the
cells at the seam and the rest (``quarter_cells``).

The companions: ``spec`` the eigenvalues of ``ref``'s real $v$ and
``participation``; ``band`` the ratios of the squared errors of ``pred`` to
the real's variance in the bands of ``ref``'s principal components
(``band_ratios``); ``comp`` the identification in the complement of the top
``comp_k`` components (none of a $k$ that leaves nothing).

The implementation checks (README, V-s0 to V-s6; rows of ``part``
``check``, a failure stops the reading): V-s0 the stages unchanged to the
bit with the hidden pixels random; V-s1 ``oracle`` at exactly 1 (and the
grid's duplicate rows); V-s2 ``const`` all ties; V-s3 the replicated K3 of
the checkpoint's own tokens (every token its stack's mean over the
positions: the real, the predictor's hidden ones, the context side's from
the real's) has ``pred`` and ``flat`` the same values, ``pool`` and
``oracle`` the same, ``swap`` exactly 1/2, and $\lvert L \rvert \le
10^{-9}$; V-s6 the noisy real $r_n + \sigma \xi$ at the closed form
``expected_pair_wins``. V-s4 and V-s5 (``check_rows``, once on real data):
the local colour code (``jepa_eval.ColorCode``) and its mirror
(``MirrorColorCode``) with the real as the prediction.

Into ``<run_dir>/posthoc_self/`` (``config.json``, ``rows/<step>.json`` and,
of ``--transplant-steps``, ``rows/<step>-tr.json``; compact JSON)::

    python -m exp.vision.self_eval checks --config <run_dir> --out runs/v18c-self/checks
    python -m exp.vision.self_eval run <run_dir> --steps 0,6000,60000 \
        --transplant-steps 0,6000,60000
    python -m exp.vision.self_eval same <run_dir> ... --step 0
"""

import argparse
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, override

import torch
import yaml
from torch import Tensor

from exp.lib import ckpt
from exp.lib.metrics import identification, pair_wins, swap
from exp.lib.provenance import configure_determinism, of_diagnosis
from exp.lib.regions import (
    COARSE,
    EVAL_MASKS,
    SWAP_PAIRS,
    harmonic_matrix,
    pixel_mask,
    transplant,
)
from exp.lib.rundir import atomic_open
from exp.lib.tours.stacks import tours_root
from exp.vision.compose import from_dict
from exp.vision.eval import BATCH
from exp.vision.fill_eval import QUARTER_NAMES, FillConf, e1f_masks, fill_sets
from exp.vision.gen_eval import Zurich
from exp.vision.jepa import JEPAModules, expand_masks, mask_index
from exp.vision.jepa_diag import run_ckpts
from exp.vision.jepa_eval import SPAN, ColorCode, build_run_jepa, participation
from exp.vision.models import early_fusion
from exp.vision.schema import RunConf

__all__ = [
    "MASKS",
    "METRICS",
    "NOISE_SEED",
    "OUT",
    "PERTURB_SEED",
    "SETS",
    "STAGES",
    "MirrorColorCode",
    "SelfConf",
    "SelfSets",
    "band_ratios",
    "check_rows",
    "evaluate_checks",
    "evaluate_run",
    "expected_pair_wins",
    "local_sums",
    "local_terms",
    "main",
    "quarter_cells",
    "self_sets",
    "stage_scores",
    "step_rows",
]

OUT = "posthoc_self"
MASKS = e1f_masks()  # (quarter, k) -> the coarse mask
KS = (4, 8, 12)
STAGES = ("pred", "pool", "flat", "interp", "oracle", "const")
METRICS = ("cos", "l2")
SETS = ("easy", "hard", "hard_c")
SWAP_STAGES = ("pred", "oracle")
COMP_STAGES = ("pred", "pool", "oracle")
DIRECTIONS = ("fwd", "rev")
NOISE_SEED = 103  # V-s6's noise
PERTURB_SEED = 107  # V-s0's random hidden pixels
DIGITS = 6  # of the per-query lists in the rows

type Tokens = Callable[[Tensor], Tensor]
type Scores = dict[tuple[str, str], dict[str, Any]]


@dataclass(frozen=True)
class SelfConf:
    """The settings (module docstring), by default the plan's: the
    transplants' ``contexts``, the bands' upper ``edges`` (4 times apart),
    the complements' ``comp_k``, V-s6's targets, draws and tolerance (in
    standard errors), the spectrum's ``spec_top`` eigenvalues, $10^{-9}$ of
    V-s3 and V-s4, and the bootstrap's ``blocks``, ``boot``, ``boot_seed``
    (of ``results/v18c/judge_self.py``, recorded here)."""

    contexts: int = 4
    edges: tuple[int, ...] = (4, 16, 64, 256)
    comp_k: tuple[int, ...] = (4, 16)
    noise_targets: tuple[float, ...] = (0.6, 0.75, 0.9)
    noise_draws: int = 16
    noise_tol: float = 3.0
    spec_top: int = 64
    local_tol: float = 1e-9
    blocks: tuple[int, ...] = (8, 16)
    boot: int = 2000
    boot_seed: int = 101


@dataclass(frozen=True)
class SelfSets:
    """The stacks u8 ``(·, n, 3, S, S)`` of the reference ``ref`` and of the
    grid ``grid``; the candidates ``cand`` (grid indices), ``hard_q``
    (positions in ``cand``) and ``hard_among`` (grid indices, its own
    first); the contexts' indices in ``ref``; ``ident`` and the contexts'
    ``places`` (video, frame) for the record."""

    ref: Tensor
    grid: Tensor
    cand: Tensor
    hard_q: Tensor
    hard_among: Tensor
    contexts: Tensor
    ident: Mapping[str, Any]
    places: Sequence[tuple[str, int]] = ()


def self_sets(
    conf: RunConf,
    root: Path,
    fconf: FillConf,
    *,
    contexts: int,
    span: float = SPAN,
    spacing: Mapping[str, Any] | None = None,
) -> SelfSets:
    """The sets of ``conf``'s data at ``root`` (module docstring): those of
    ``fill_sets`` and the whole grid of Zurich.

    Raises:
        ValueError: The fit set has fewer videos than ``contexts``.
    """
    sets = fill_sets(conf, root, fconf, span=span, spacing=spacing)
    zurich = Zurich(conf, root, fconf.grid_step)
    grid = zurich.stacks(torch.arange(len(zurich)))
    assert torch.equal(grid[sets.cand], sets.cand_stacks)
    assert sets.places is not None
    fit = sets.places["fit"]
    first: dict[str, int] = {}
    for i, (video, _) in enumerate(fit):
        first.setdefault(video, i)
    idx = list(first.values())[:contexts]
    if len(idx) < contexts:
        raise ValueError(f"{contexts} contexts of a fit set of {len(first)} videos")
    return SelfSets(
        ref=sets.fit,
        grid=grid,
        cand=sets.cand,
        hard_q=sets.hard_q,
        hard_among=sets.hard_among,
        contexts=torch.tensor(idx),
        ident=sets.ident,
        places=[fit[i] for i in idx],
    )


class MirrorColorCode(ColorCode):
    """The mirror code: each cell's token is ``ColorCode``'s of the cell
    turned half a turn, $(a, b) \\to (g - 1 - a, g - 1 - b)$ (the reversed
    row-major positions). It is local to the cell opposite: a quarter's
    tokens are the pixels of the diagonal quarter."""

    @override
    def tokens(self, x: Tensor) -> Tensor:
        return super().tokens(x).flip(1)


# -- scores ------------------------------------------------------------------------------


def _scores(
    e: Tensor,
    r: Tensor,
    metric: str,
    *,
    truth: Tensor | None = None,
    among: Tensor | None = None,
) -> dict[str, Any]:
    """``pair_wins``'s values as a score: ``A``, ``top1`` (the share of
    $A_n = 1$), the per-query lists and ``n_tie``."""
    _, each, ties = pair_wins(e, r, truth=truth, among=among, metric=metric)
    if r.dim() == 3:
        c = r.shape[1]
    else:
        c = r.shape[0] if among is None else among.shape[1]
    hits = (each == 1.0).double()
    return {
        "A": float(each.mean()),
        "top1": float(hits.mean()),
        "A_i": each.tolist(),
        "hit_i": hits.tolist(),
        "n_tie": int((ties == c - 1).sum()),
    }


def stage_scores(
    query: Tensor,
    query_mean: Tensor,
    real: Tensor,
    real_mean: Tensor,
    *,
    cand: Tensor,
    hard_q: Tensor,
    hard_among: Tensor,
    metrics: Sequence[str] = METRICS,
    sets: Sequence[str] = SETS,
) -> Scores:
    r"""The scores by (set, metric) of a stage's vectors of the candidates
    ``query`` ``(n, F)`` less ``query_mean`` ``(F,)`` against the grid's
    ``real`` ``(G, F)`` less ``real_mean``: ``A``, ``top1``, the per-query
    ``A_i`` and ``hit_i`` and ``n_tie`` (the queries tied with every
    candidate); of ``easy`` the candidates' own (chance 1/n), of ``hard``
    and ``hard_c`` those of ``hard_among`` (``hard_c`` each less the mean of
    the query's candidates; its own first), the queries ``hard_q``. The easy
    cosine's also has ``pick_max`` (``metrics.identification``)."""
    e = query.double() - query_mean.double()
    r = real.double() - real_mean.double()
    out: Scores = {}
    for metric in metrics:
        if "easy" in sets:
            row = _scores(e, r[cand], metric)
            if metric == "cos":
                ident = identification(e, r[cand], center=torch.zeros(e.shape[1]))
                row["pick_max"] = ident["pick_max"]
            out[("easy", metric)] = row
        eh = e[hard_q]
        if "hard" in sets:
            out[("hard", metric)] = _scores(
                eh, r, metric, truth=cand[hard_q], among=hard_among
            )
        if "hard_c" in sets:
            assert torch.equal(hard_among[:, 0], cand[hard_q])  # its own first
            c = r[hard_among]
            out[("hard_c", metric)] = _scores(eh, c - c.mean(1, keepdim=True), metric)
    return out


def _pair_terms(
    real: Tensor, truth: Tensor | None, among: Tensor | None
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""Of each query $n$ and candidate $m$: $\langle r_n, \tilde r_n -
    \tilde r_m \rangle$, $\lVert \tilde r_n - \tilde r_m \rVert$ (``(n,
    c)``), the other candidates ``(n, c)`` bool and the queries' $r_n$."""
    r = real.double()
    unit = r / torch.linalg.vector_norm(r, dim=1, keepdim=True).clamp(min=1e-300)
    own = torch.arange(len(r)) if truth is None else truth
    cols = torch.arange(len(r)).expand(len(own), -1) if among is None else among
    rn, un = r[own], unit[own]
    dots = (un @ unit.T).gather(1, cols)  # (n, c)
    sq_n = un.square().sum(1, keepdim=True)
    sq_m = unit.square().sum(1)[cols]
    num = (rn * un).sum(1, keepdim=True) - (rn @ unit.T).gather(1, cols)
    den = (sq_n + sq_m - 2 * dots).clamp(min=0).sqrt()
    other = cols != own[:, None]
    return num, den, other, rn


def _expected(num: Tensor, den: Tensor, other: Tensor, sigma: float) -> float:
    z = num / (sigma * den.clamp(min=1e-300))
    phi = torch.where(den > 0, 0.5 * (1 + torch.erf(z / math.sqrt(2))), 0.5)
    each = (phi * other).sum(1) / other.sum(1)
    return float(each.mean())


def expected_pair_wins(
    real: Tensor,
    sigma: float,
    *,
    truth: Tensor | None = None,
    among: Tensor | None = None,
) -> float:
    r"""The expectation of ``metrics.pair_wins`` (cosine) of the noisy
    queries $r_n + \sigma \xi_n$ ($\xi_n$ standard normal) against the fixed
    candidates ``real`` ``(m, F)`` (``truth``, ``among`` as there): $\frac1n
    \sum_n \frac{1}{c - 1} \sum_{m \ne n} \Phi\big(\langle r_n, \tilde r_n -
    \tilde r_m \rangle / \sigma \lVert \tilde r_n - \tilde r_m \rVert\big)$,
    $\tilde r = r / \lVert r \rVert$, a term of $\tilde r_n = \tilde r_m$
    1/2 (V-s6). The query's normalization is a positive factor of every
    similarity, and $\langle r_n + \sigma \xi, \tilde r_n - \tilde r_m
    \rangle$ is normal of that mean and variance $\sigma^2 \lVert \tilde r_n
    - \tilde r_m \rVert^2$."""
    num, den, other, _ = _pair_terms(real, truth, among)
    return _expected(num, den, other, sigma)


def _sigma_at(
    num: Tensor, den: Tensor, other: Tensor, target: float, scale: float
) -> float:
    r"""The $\sigma$ of ``_expected`` = ``target`` by bisection on $\log
    \sigma$ (the expectation falls with $\sigma$: every numerator is $\lVert
    r_n \rVert (1 - \cos) \ge 0$)."""
    lo, hi = math.log(scale * 1e-8), math.log(scale * 1e8)
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _expected(num, den, other, math.exp(mid)) > target:
            lo = mid
        else:
            hi = mid
    return math.exp(0.5 * (lo + hi))


def _noisy(
    real: Tensor,
    truth: Tensor | None,
    among: Tensor | None,
    sconf: SelfConf,
    gen: torch.Generator,
) -> list[dict[str, float]]:
    r"""V-s6 of the centred real ``real`` (``truth``, ``among`` as in
    ``pair_wins``): of each target of ``sconf``, the $\sigma$ of the closed
    form at it, the mean $A$ of ``noise_draws`` draws of $r_n + \sigma \xi$
    (never centred again) and $z$, its deviation over its standard error."""
    num, den, other, rn = _pair_terms(real, truth, among)
    scale = float(rn.square().sum(1).mean().sqrt())
    out = []
    for target in sconf.noise_targets:
        sigma = _sigma_at(num, den, other, target, scale)
        want = _expected(num, den, other, sigma)
        got = []
        for _ in range(sconf.noise_draws):
            xi = torch.randn(rn.shape, generator=gen, dtype=torch.float64)
            got.append(pair_wins(rn + sigma * xi, real, truth=truth, among=among)[0])
        mean = sum(got) / len(got)
        sd = math.sqrt(sum((g - mean) ** 2 for g in got) / (len(got) - 1))
        z = (mean - want) / (sd / math.sqrt(len(got))) if sd > 0 else math.inf
        out.append(
            {"target": target, "sigma": sigma, "want": want, "got": mean, "z": z}
        )
    return out


def band_ratios(
    pred: Tensor, real: Tensor, mean: Tensor, basis: Tensor, edges: Sequence[int]
) -> dict[str, Any]:
    r"""The ratio of each band $b$ of the principal components ``basis``
    ``(F, F)`` (columns in order of falling variance; the bands' upper
    ``edges``, cut at $F$): $\sum_n \lVert U_b^\top (\hat v_n - v_n) \rVert^2
    / \sum_n \lVert U_b^\top (v_n - \mu) \rVert^2$ of ``pred`` $\hat v$ and
    ``real`` $v$ ``(n, F)``, ``mean`` $\mu$; the bands' shares of the
    variance; and the ``total`` $\sum \lVert \hat v - v \rVert^2 / \sum
    \lVert v - \mu \rVert^2$, the shares' weighted mean of the ratios."""
    f = basis.shape[0]
    err = ((pred.double() - real.double()) @ basis.double()).square().sum(0)
    var = ((real.double() - mean.double()) @ basis.double()).square().sum(0)
    lows = [0, *[e for e in edges if e < f]]
    highs = [*[e for e in edges if e < f], f]
    ratio = [
        float(err[a:b].sum() / var[a:b].sum()) for a, b in zip(lows, highs, strict=True)
    ]
    share = [
        float(var[a:b].sum() / var.sum()) for a, b in zip(lows, highs, strict=True)
    ]
    return {"ratio": ratio, "share": share, "total": float(err.sum() / var.sum())}


# -- locality ------------------------------------------------------------------------------


def quarter_cells(q: str, s: int) -> tuple[Tensor, Tensor, Tensor]:
    """Of the quarter ``q`` on the grid $4 s$ (fine positions ``(16 s^2,)``
    bool): its positions, those of its coarse cells with an edge on a cell
    outside (the seam inside), and those of the cells outside with an edge on
    it (the seam outside)."""
    inside = EVAL_MASKS[f"Q_{q}"].reshape(COARSE, COARSE)
    pad = torch.nn.functional.pad(inside, (1, 1, 1, 1))
    near = pad[:-2, 1:-1] | pad[2:, 1:-1] | pad[1:-1, :-2] | pad[1:-1, 2:]
    out = torch.nn.functional.pad(~inside, (1, 1, 1, 1))
    near_out = out[:-2, 1:-1] | out[2:, 1:-1] | out[1:-1, :-2] | out[1:-1, 2:]
    return (
        _fine(inside.flatten(), s),
        _fine((inside & near_out).flatten(), s),
        _fine((~inside & near).flatten(), s),
    )


def local_sums(t_tr: Tensor, t_j: Tensor, t_m: Tensor) -> dict[str, Tensor]:
    r"""The sums over the batch of each position ``(T,)`` float64 of $\langle
    \Delta_p, \delta_p \rangle$ (``num``), $\lVert \delta_p \rVert^2$
    (``den``), $\lVert \Delta_p \rVert^2$ (``dd``) and $\lVert \Delta_p -
    \delta_p \rVert^2$ (``eta``), $\Delta = $ ``t_tr`` $-$ ``t_j``, $\delta =
    $ ``t_m`` $-$ ``t_j`` (``t_tr``, ``t_m`` ``(B, T, d)``, ``t_j`` ``(T,
    d)``)."""
    big, small = t_tr.double() - t_j.double(), t_m.double() - t_j.double()
    return {
        "num": (big * small).sum((0, 2)),
        "den": small.square().sum((0, 2)),
        "dd": big.square().sum((0, 2)),
        "eta": (big - small).square().sum((0, 2)),
    }


def _groups(
    sums: Mapping[str, Tensor], inside: Tensor, seam_u: Tensor, seam_v: Tensor
) -> dict[str, float]:
    """The sums of ``local_sums`` over the positions of each part."""
    parts = {
        "U": inside,
        "V": ~inside,
        "U_seam": seam_u,
        "U_in": inside & ~seam_u,
        "V_seam": seam_v,
        "V_far": ~inside & ~seam_v,
    }
    out = {}
    for name, at in parts.items():
        for key in ("num", "den"):
            out[f"{key}_{name}"] = float(sums[key][at].sum())
    out["dd_V"] = float(sums["dd"][~inside].sum())
    out["eta_U"] = float(sums["eta"][inside].sum())
    return out


def _terms(g: Mapping[str, float]) -> dict[str, float | None]:
    def ratio(a: float, b: float) -> float | None:
        return a / b if b > 0 else None

    out = {
        f"a_{p}": ratio(g[f"num_{p}"], g[f"den_{p}"])
        for p in ("U", "V", "U_seam", "U_in", "V_seam", "V_far")
    }
    a_u, a_v = out["a_U"], out["a_V"]
    out["L"] = None if a_u is None or a_v is None else a_u - a_v
    out["rho_V"] = ratio(g["dd_V"], g["den_V"])
    out["eta_U"] = ratio(g["eta_U"], g["den_U"])
    return out


def local_terms(
    sums: Mapping[str, Tensor], inside: Tensor, seam_u: Tensor, seam_v: Tensor
) -> dict[str, float | None]:
    r"""$a_U$, $a_V$, $L$, $\rho_V$, $\eta_U$ and $a_U$, $a_V$ of the seams
    (``a_U_seam``, ``a_U_in``, ``a_V_seam``, ``a_V_far``) of the sums of
    ``local_sums`` and the parts of ``quarter_cells`` (module docstring);
    None where the denominator is 0 (a code that does not move)."""
    return _terms(_groups(sums, inside, seam_u, seam_v))


# -- the codes' vectors -----------------------------------------------------------------------


def _pmean(x: Tensor, dim: int) -> Tensor:
    """The mean over ``dim``, as the first plus the mean of the deviations
    from it: equal values give it to the bit (V-s3's exactness)."""
    x0 = x.narrow(dim, 0, 1)
    return (x0 + (x - x0).mean(dim, keepdim=True)).squeeze(dim)


def _flat(v: Tensor) -> Tensor:
    """Each of the positions ``(B, P, d)`` replaced by their mean."""
    return _pmean(v, 1)[:, None].expand_as(v)


def _k3(t: Tensor) -> Tensor:
    """The tokens ``(..., T, d)`` all replaced by their mean."""
    return _pmean(t, -2).unsqueeze(-2).expand_as(t)


@torch.no_grad()
def _tokens(fn: Tokens, x: Tensor, dev: torch.device) -> Tensor:
    """``fn`` of the stacks ``x`` (CPU), ``BATCH`` at a time on ``dev``:
    float32 on the CPU."""
    return torch.cat([fn(c.to(dev)).float().cpu() for c in x.split(BATCH)])


def _fine(mask: Tensor, s: int) -> Tensor:
    return expand_masks(mask[None], s)[0]


@dataclass(frozen=True)
class _Harmonic:
    """The harmonic fill's hidden and visible pixels and matrix (on the
    device) of a mask."""

    u: Tensor
    v: Tensor
    m: Tensor

    def __call__(self, x: Tensor) -> Tensor:
        """The stacks u8 ``(B, n, 3, S, S)`` filled, floats in ``[0, 1]``."""
        b, n, c, h, w = x.shape
        flat = x.reshape(b, n * c, h * w).double()
        out = flat.clone()
        out[:, :, self.u] = flat[:, :, self.v] @ self.m.T
        return (out / 255).float().reshape(x.shape)


@torch.no_grad()
def _mask_stages(
    mods: JEPAModules,
    x: Tensor,
    u: Tensor,
    mask: Tensor,
    q_pos: Tensor,
    harm: _Harmonic | None,
    dev: torch.device,
) -> dict[str, Tensor]:
    r"""The stages of the stacks ``x`` under the fine ``mask`` at ``q_pos``
    ``(N, P d)`` float64 (module docstring), with the replicated K3's
    ``pred_k3``, ``flat_k3`` (of the predictor's hidden positions' mean) and
    ``pool_k3`` (of the context replaced by ``u`` ``(N, d)``, the real's
    position means); ``interp`` without ``harm`` omitted."""
    keep, hide = mask_index(mask[None].to(dev))
    sel = torch.isin(hide[0].cpu(), q_pos)
    out: dict[str, list[Tensor]] = {}
    for xb, ub in zip(x.split(BATCH), u.split(BATCH), strict=True):
        xb = xb.to(dev)
        b = len(xb)
        kb, hb = keep.expand(b, -1), hide.expand(b, -1)
        ctx = mods.encoder(early_fusion(xb), kb)
        pred = mods.predictor(ctx, kb, hb).double().cpu()  # (b, k, d)
        pq = pred[:, sel]  # (b, P, d)
        pool = _pmean(mods.proj(ctx).double().cpu(), 1)[:, None].expand_as(pq)
        p3 = _pmean(pred, 1)[:, None].expand_as(pq)
        c3 = _pmean(ub[:, None].expand(b, kb.shape[1], -1), 1)[:, None].expand_as(pq)
        got = {
            "pred": pq,
            "pool": pool,
            "flat": _flat(pq),
            "pred_k3": p3,
            "flat_k3": _flat(p3),
            "pool_k3": c3,
        }
        if harm is not None:
            got["interp"] = mods.tokens(harm(xb))[:, q_pos].double().cpu()
        for k, v in got.items():
            out.setdefault(k, []).append(v.flatten(1))
    return {k: torch.cat(v) for k, v in out.items()}


@torch.no_grad()
def _swap_stages(
    mods: JEPAModules,
    x: Tensor,
    mask: Tensor,
    pos: Mapping[str, Tensor],
    dev: torch.device,
) -> dict[str, Tensor]:
    """Of ``swap``'s mask (both quarters hidden), one pass of the predictor:
    ``pred_<q>`` its outputs at each quarter's positions ``pos[q]`` and
    ``pred_k3_<q>`` its hidden positions' mean there ``(N, P d)`` float64,
    so that the K3's two quarters are the same tensor."""
    keep, hide = mask_index(mask[None].to(dev))
    out: dict[str, list[Tensor]] = {}
    for xb in x.split(BATCH):
        b = len(xb)
        kb, hb = keep.expand(b, -1), hide.expand(b, -1)
        ctx = mods.encoder(early_fusion(xb.to(dev)), kb)
        pred = mods.predictor(ctx, kb, hb).double().cpu()  # (b, k, d)
        mean = _pmean(pred, 1)
        for q, p in pos.items():
            pq = pred[:, torch.isin(hide[0].cpu(), p)]
            out.setdefault(f"pred_{q}", []).append(pq.flatten(1))
            k3 = mean[:, None].expand_as(pq).flatten(1)
            out.setdefault(f"pred_k3_{q}", []).append(k3)
    return {k: torch.cat(v) for k, v in out.items()}


@torch.no_grad()
def _transplanted(
    fn: Tokens,
    x: Tensor,
    x_j: Tensor,
    pixels: Tensor,
    q_pos: Tensor,
    direction: str,
    dev: torch.device,
    local: tuple[Tensor, Tensor] | None = None,
) -> tuple[Tensor, dict[str, Tensor] | None, dict[str, Tensor] | None]:
    r"""The vectors ``(N, P d)`` float64 at ``q_pos`` of the stacks ``x``
    with the context ``x_j`` ``(1, ·)`` transplanted (``"fwd"``: $x_{j
    \leftarrow m}$; ``"rev"``: $x_{m \leftarrow j}$); with ``local`` ($t_j$
    ``(T, d)``, $t_m$ ``(N, T, d)`` of ``x``) the sums of ``local_sums``,
    and those of the replicated K3."""
    x_j, pixels = x_j.to(dev), pixels.to(dev)
    vecs: list[Tensor] = []
    sums: dict[str, Tensor] | None = None
    sums3: dict[str, Tensor] | None = None
    start = 0
    for xb in x.split(BATCH):
        xb = xb.to(dev)
        xt = (
            transplant(x_j, xb, pixels)
            if direction == "fwd"
            else transplant(xb, x_j, pixels)
        )
        t = fn(xt).double().cpu()
        vecs.append(t[:, q_pos].flatten(1))
        if local is not None:
            t_j, t_m = local[0].double(), local[1][start : start + len(xb)].double()
            got, got3 = local_sums(t, t_j, t_m), local_sums(_k3(t), _k3(t_j), _k3(t_m))
            sums = got if sums is None else {k: sums[k] + got[k] for k in got}
            sums3 = got3 if sums3 is None else {k: sums3[k] + got3[k] for k in got3}
        start += len(xb)
    return torch.cat(vecs), sums, sums3


# -- rows --------------------------------------------------------------------------------


def _rounded(x: Sequence[float]) -> list[float]:
    return [round(v, DIGITS) for v in x]


def _average(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The per-query mean of the scores ``results`` (of the same queries):
    ``A_i`` and ``hit_i`` averaged, ``n_tie`` summed."""
    a = torch.tensor([r["A_i"] for r in results], dtype=torch.float64).mean(0)
    h = torch.tensor([r["hit_i"] for r in results], dtype=torch.float64).mean(0)
    return {
        "A": float(a.mean()),
        "top1": float(h.mean()),
        "A_i": a.tolist(),
        "hit_i": h.tolist(),
        "n_tie": sum(r["n_tie"] for r in results),
    }


def _id_rows(
    store: Mapping[tuple[Any, ...], Mapping[str, Scores]],
    labels: Callable[[tuple[Any, ...]], dict[str, Any]],
) -> list[dict[str, Any]]:
    """The rows of each key's scores by quarter (``store[key][q]``) and of
    their per-query mean over the quarters (``quarter`` ``"mean"``, with the
    per-query lists of the cosine)."""
    rows = []
    for key, by_q in store.items():
        for sm in next(iter(by_q.values())):
            base = labels(key) | {"set": sm[0], "metric": sm[1]}
            for q, scores in by_q.items():
                s = scores[sm]
                row = base | {"quarter": q, "A": s["A"], "top1": s["top1"]}
                row["n_tie"] = s["n_tie"]
                if "pick_max" in s:
                    row["pick_max"] = s["pick_max"]
                rows.append(row)
            mean = _average([scores[sm] for scores in by_q.values()])
            row = (
                base
                | {"quarter": "mean"}
                | {k: mean[k] for k in ("A", "top1", "n_tie")}
            )
            if sm[1] == "cos":
                row |= {"A_i": _rounded(mean["A_i"]), "hit_i": mean["hit_i"]}
            rows.append(row)
    return rows


def _same(a: Scores, b: Scores) -> bool:
    return all(a[k]["A_i"] == b[k]["A_i"] for k in a)


def _all_tie(s: Scores) -> bool:
    return all(v["n_tie"] == len(v["A_i"]) for v in s.values())


@torch.no_grad()
def _swap_values(
    p_a: Tensor, p_b: Tensor, ref_a: Tensor, ref_b: Tensor, r_a: Tensor
) -> Tensor:
    """``swap``'s values ``(n,)`` of one direction: $d$ of the queries'
    ``p_a``, ``p_b`` less the reference's mean of ``ref_a`` $-$ ``ref_b``
    against $A$'s centred real ``r_a``."""
    d = (p_a - p_b) - (ref_a - ref_b).mean(0)
    return swap(d, torch.zeros_like(d), r_a)[1]


@torch.no_grad()
def _swap_part(
    mods: JEPAModules,
    sets: SelfSets,
    t_grid: Tensor,
    t_ref: Tensor,
    x_cand: Tensor,
    perturbed: Callable[[Tensor], Tensor],
    v_s0: dict[str, bool],
    *,
    expand: int,
    k_run: int,
    dev: torch.device,
) -> tuple[list[dict[str, Any]], bool]:
    """``swap``'s rows of ``SWAP_STAGES`` (module docstring) and whether the
    replicated K3 ties every query (V-s3); V-s0 of the masks ``S`` into
    ``v_s0``."""
    vals: dict[str, list[Tensor]] = {st: [] for st in (*SWAP_STAGES, "pred_k3")}
    for name, (qa, qb) in SWAP_PAIRS.items():
        mask = _fine(EVAL_MASKS[name], expand)
        pos = {q: _fine(EVAL_MASKS[q], expand).nonzero().flatten() for q in (qa, qb)}
        c = _swap_stages(mods, x_cand, mask, pos, dev)
        r = _swap_stages(mods, sets.ref, mask, pos, dev)
        bad = _swap_stages(mods, perturbed(EVAL_MASKS[name]), mask, pos, dev)
        v_s0[name] = all(
            torch.equal(bad[f"pred_{q}"], c[f"pred_{q}"]) for q in (qa, qb)
        )
        real = {
            q: (
                t_grid[sets.cand][:, p].flatten(1).double(),
                t_ref[:, p].flatten(1).double(),
            )
            for q, p in pos.items()
        }
        for a, b in ((qa, qb), (qb, qa)):
            r_a = real[a][0] - real[a][1].mean(0)
            for st in ("pred", "pred_k3"):
                vals[st].append(
                    _swap_values(
                        c[f"{st}_{a}"],
                        c[f"{st}_{b}"],
                        r[f"{st}_{a}"],
                        r[f"{st}_{b}"],
                        r_a,
                    )
                )
            vals["oracle"].append(
                _swap_values(real[a][0], real[b][0], real[a][1], real[b][1], r_a)
            )
    in_dist = int(EVAL_MASKS["S_1"].sum()) == k_run
    rows = []
    for st in SWAP_STAGES:
        v = torch.stack(vals[st])
        per = v.mean(0)
        rows.append(
            {
                "part": "swap", "set": "easy", "stage": st, "k": 8,
                "in_dist": in_dist, "swap": float(per.mean()),
                "swap_i": per.tolist(), "n_tie": int((v == 0.5).all(0).sum()),
            }
        )  # fmt: skip
    return rows, bool((torch.stack(vals["pred_k3"]) == 0.5).all())


@torch.no_grad()
def step_rows(
    mods: JEPAModules,
    sets: SelfSets,
    sconf: SelfConf,
    *,
    expand: int,
    k_run: int,
    transplant: bool,
    dev: torch.device,
) -> list[dict[str, Any]]:
    """The rows of one checkpoint of the JEPA ``mods`` (module docstring;
    masks spread by ``expand``, ``in_dist`` where $k$ is ``k_run``), with
    ``transplant`` the transplanted candidates, the locality and V-s3's
    $L$."""
    s, size = expand, sets.grid.shape[-1]
    t_grid = _tokens(mods.tokens, sets.grid, dev)  # (G, T, d)
    t_ref = _tokens(mods.tokens, sets.ref, dev)
    u_grid, u_ref = _pmean(t_grid.double(), 1), _pmean(t_ref.double(), 1)
    x_cand = sets.grid[sets.cand]
    u_cand = u_grid[sets.cand]
    picks: dict[str, Any] = {
        "cand": sets.cand,
        "hard_q": sets.hard_q,
        "hard_among": sets.hard_among,
    }
    n = len(sets.cand)
    gen = torch.Generator().manual_seed(PERTURB_SEED)
    rows: list[dict[str, Any]] = []
    v_s0: dict[str, bool] = {}
    v_s3: dict[str, bool] = {}

    def at(t: Tensor, pos: Tensor) -> Tensor:
        return t[:, pos].flatten(1).double()

    def perturbed(coarse: Tensor) -> Tensor:
        hidden = pixel_mask(coarse, size)
        noise = torch.randint(256, x_cand.shape, generator=gen, dtype=torch.uint8)
        return torch.where(hidden, noise, x_cand)

    swap_rows, swap_k3 = _swap_part(
        mods,
        sets,
        t_grid,
        t_ref,
        x_cand,
        perturbed,
        v_s0,
        expand=s,
        k_run=k_run,
        dev=dev,
    )
    rows += swap_rows
    v_s3["swap"] = swap_k3

    # the quarters
    store: dict[tuple[Any, ...], dict[str, Scores]] = {}
    band: dict[int, list[dict[str, Any]]] = {}
    comp: dict[tuple[int, int, str], dict[str, Scores]] = {}
    comp_rest: dict[tuple[int, int], list[float]] = {}
    groups: dict[tuple[str, int], tuple[dict[str, float], dict[str, float]]] = {}
    v_s1: list[dict[str, Any]] = []
    v_s2: list[bool] = []
    v_s6: list[dict[str, Any]] = []
    noise = torch.Generator().manual_seed(NOISE_SEED)
    harms: dict[bytes, _Harmonic] = {}
    for q in QUARTER_NAMES:
        inside, seam_u, seam_v = quarter_cells(q, s)
        q_pos = inside.nonzero().flatten()
        real_g, real_r = at(t_grid, q_pos), at(t_ref, q_pos)
        mu_t = real_r.mean(0)
        dev_r = real_r - mu_t
        lam, basis = torch.linalg.eigh(dev_r.T @ dev_r / len(dev_r))
        lam, basis = lam.flip(0).clamp(min=0), basis.flip(1)
        rows.append(
            {
                "part": "spec", "quarter": q,
                "eig": lam[: sconf.spec_top].tolist(), "eig_sum": float(lam.sum()),
                "participation": float(participation(real_r)),
            }
        )  # fmt: skip
        r_c = real_g - mu_t
        v_s1.append(
            {"quarter": q, "duplicates": len(r_c) - len(torch.unique(r_c, dim=0))}
        )
        # V-s6: the noisy real
        hard = (sets.cand[sets.hard_q], sets.hard_among)
        for set_name, base, truth, among in (
            ("easy", r_c[sets.cand], None, None),
            ("hard", r_c, *hard),
        ):
            for row in _noisy(base, truth, among, sconf, noise):
                v_s6.append({"quarter": q, "set": set_name} | row)
        # the candidates: natural and transplanted
        cands: dict[tuple[str, int | None], tuple[Tensor, Tensor]] = {
            ("nat", None): (real_g, mu_t)
        }
        if transplant:
            pixels = pixel_mask(EVAL_MASKS[f"Q_{q}"], size)
            for j_i, j in enumerate(sets.contexts.tolist()):
                x_j, t_j = sets.ref[j : j + 1], t_ref[j]
                for direction in DIRECTIONS:
                    local = (t_j, t_grid) if direction == "fwd" else None
                    vec_g, sums, sums3 = _transplanted(
                        mods.tokens,
                        sets.grid,
                        x_j,
                        pixels,
                        q_pos,
                        direction,
                        dev,
                        local,
                    )
                    vec_r, _, _ = _transplanted(
                        mods.tokens, sets.ref, x_j, pixels, q_pos, direction, dev
                    )
                    cands[(direction, j_i)] = (vec_g, vec_r.mean(0))
                    if sums is not None and sums3 is not None:
                        groups[(q, j_i)] = (
                            _groups(sums, inside, seam_u, seam_v),
                            _groups(sums3, inside, seam_u, seam_v),
                        )
        # the masks of this quarter
        for k in KS:
            coarse = MASKS[(q, k)]
            mask = _fine(coarse, s)
            key = coarse.numpy().tobytes()
            if key not in harms:
                u, v, m = harmonic_matrix(pixel_mask(coarse, size))
                harms[key] = _Harmonic(u.to(dev), v.to(dev), m.to(dev))
            harm = harms[key]
            st_c = _mask_stages(mods, x_cand, u_cand, mask, q_pos, harm, dev)
            st_r = _mask_stages(mods, sets.ref, u_ref, mask, q_pos, harm, dev)
            st_p = _mask_stages(mods, perturbed(coarse), u_cand, mask, q_pos, harm, dev)
            v_s0[f"{q}/{k}"] = all(
                torch.equal(st_p[x], st_c[x])
                for x in ("pred", "pool", "flat", "interp")
            )
            queries = {
                x: (st_c[x], st_r[x].mean(0))
                for x in ("pred", "pool", "flat", "interp")
            }
            queries["oracle"] = (real_g[sets.cand], mu_t)
            zero = torch.zeros_like(mu_t)
            queries["const"] = (zero.expand(n, -1), zero)
            for (kind, j), (cvec, cmean) in cands.items():
                for stage, (qv, qm) in queries.items():
                    name = "oracle_tr" if kind != "nat" and stage == "oracle" else stage
                    got = stage_scores(qv, qm, cvec, cmean, **picks)
                    store.setdefault((k, kind, j, name), {})[q] = got
            nat = {x: store[(k, "nat", None, x)][q] for x in STAGES}
            v_s2.append(
                all(nat["const"][(st, "cos")]["n_tie"] == len(nat["const"][(st, "cos")]["A_i"])
                    and nat["const"][(st, "cos")]["A"] == 0.5 for st in SETS)
            )  # fmt: skip
            # V-s3: the replicated K3 of these tokens
            real3_g = u_grid[:, None].expand(-1, len(q_pos), -1).flatten(1)
            real3_r = u_ref[:, None].expand(-1, len(q_pos), -1).flatten(1)
            mu3 = real3_r.mean(0)
            k3 = {
                x: stage_scores(
                    st_c[x], st_r[x].mean(0), real3_g, mu3, metrics=("cos",), **picks
                )
                for x in ("pred_k3", "flat_k3", "pool_k3")
            }
            k3["oracle_k3"] = stage_scores(
                real3_g[sets.cand], mu3, real3_g, mu3, metrics=("cos",), **picks
            )
            v_s3[f"{q}/{k}"] = _same(k3["pred_k3"], k3["flat_k3"]) and _same(
                k3["pool_k3"], k3["oracle_k3"]
            )
            # the companions
            b = band_ratios(st_c["pred"], real_g[sets.cand], mu_t, basis, sconf.edges)
            band.setdefault(k, []).append({"quarter": q} | b)
            for kc in sconf.comp_k:
                if kc >= len(mu_t):  # nothing left: rounding alone would order
                    continue
                proj = (
                    torch.eye(len(mu_t), dtype=torch.float64)
                    - basis[:, :kc] @ basis[:, :kc].T
                )
                for stage in COMP_STAGES:
                    qv, qm = queries[stage]
                    e = (qv - qm) @ proj
                    zero_c = torch.zeros(len(mu_t), dtype=torch.float64)
                    got = stage_scores(
                        e, zero_c, r_c @ proj, zero_c, metrics=("cos",),
                        sets=("easy", "hard"), **picks,
                    )  # fmt: skip
                    comp.setdefault((k, kc, stage), {})[q] = got
                rest = float(lam[kc:].sum() / lam.sum())
                comp_rest.setdefault((k, kc), []).append(rest)

    # the mean over the contexts
    if transplant:
        for (k, kind, j, stage), by_q in list(store.items()):
            if kind == "nat" or j != 0:
                continue
            mean_q = {}
            for q in by_q:
                per_j = [
                    store[(k, kind, jj, stage)][q] for jj in range(len(sets.contexts))
                ]
                mean_q[q] = {sm: _average([p[sm] for p in per_j]) for sm in per_j[0]}
            store[(k, kind, "mean", stage)] = mean_q

    def id_labels(key: tuple[Any, ...]) -> dict[str, Any]:
        k, kind, j, stage = key
        return {
            "part": "id",
            "cand": kind,
            "j": j,
            "k": k,
            "in_dist": k == k_run,
            "stage": stage,
        }

    rows += _id_rows(store, id_labels)
    for k, per_q in band.items():
        for b in per_q:
            rows.append(
                {"part": "band", "stage": "pred", "k": k, "in_dist": k == k_run} | b
            )
        rows.append(
            {
                "part": "band", "stage": "pred", "k": k, "in_dist": k == k_run,
                "quarter": "mean",
                "ratio": torch.tensor([b["ratio"] for b in per_q]).mean(0).tolist(),
                "share": torch.tensor([b["share"] for b in per_q]).mean(0).tolist(),
                "total": sum(b["total"] for b in per_q) / len(per_q),
            }
        )  # fmt: skip

    def comp_labels(key: tuple[Any, ...]) -> dict[str, Any]:
        k, kc, stage = key
        share = comp_rest[(k, kc)]  # of the quarters
        return {
            "part": "comp", "cand": "nat", "k": k, "in_dist": k == k_run,
            "comp_k": kc, "stage": stage, "rest": sum(share) / len(share),
        }  # fmt: skip

    rows += _id_rows(comp, comp_labels)
    if transplant:
        rows += _local_rows(groups, len(sets.contexts))
        k3_groups = [g3 for _, g3 in groups.values()]
        worst = max(abs(_terms(g)["L"] or 0.0) for g in [*k3_groups, _pool(k3_groups)])
        rows.append(
            {
                "part": "check",
                "name": "V-s3-L",
                "pass": worst <= sconf.local_tol,
                "max_abs_L": worst,
            }
        )
    # the implementation checks
    ones = all(
        store[(k, "nat", None, "oracle")][q][(st, "cos")]["A"] == 1.0
        and store[(k, "nat", None, "oracle")][q][(st, "cos")]["top1"] == 1.0
        for k in KS for q in QUARTER_NAMES for st in ("easy", "hard")
    )  # fmt: skip
    z = [c["z"] for c in v_s6]
    rows += [
        {"part": "check", "name": "V-s0", "pass": all(v_s0.values()), "each": v_s0},
        {"part": "check", "name": "V-s1", "pass": ones, "duplicates": v_s1},
        {"part": "check", "name": "V-s2", "pass": all(v_s2)},
        {"part": "check", "name": "V-s3", "pass": all(v_s3.values()), "each": v_s3},
        {
            "part": "check", "name": "V-s6",
            "pass": all(abs(x) <= sconf.noise_tol for x in z), "z": z, "each": v_s6,
        },
    ]  # fmt: skip
    return rows


def _pool(gs: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """The sums of the parts' sums ``_groups`` (the pooled ratios)."""
    return {k: sum(g[k] for g in gs) for k in gs[0]}


def _local_rows(
    groups: Mapping[tuple[str, int], tuple[dict[str, float], dict[str, float]]],
    contexts: int,
    labels: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """The rows of the locality by (quarter, context), by quarter over the
    contexts, and pooled (the sums added before the ratios)."""
    extra = dict(labels or {})
    rows = []
    for (q, j), (g, _) in groups.items():
        rows.append({"part": "local", "quarter": q, "j": j} | extra | _terms(g))
    for q in QUARTER_NAMES:
        mine = [groups[(q, j)][0] for j in range(contexts) if (q, j) in groups]
        if mine:
            rows.append(
                {"part": "local", "quarter": q, "j": "mean"}
                | extra
                | _terms(_pool(mine))
            )
    every = [g for g, _ in groups.values()]
    rows.append(
        {"part": "local", "quarter": "mean", "j": "mean"} | extra | _terms(_pool(every))
    )
    return rows


@torch.no_grad()
def check_rows(
    sets: SelfSets, sconf: SelfConf, *, patch: int, dev: torch.device
) -> list[dict[str, Any]]:
    """V-s4 and V-s5 (module docstring): the local colour code of ``patch``
    (``jepa_eval.ColorCode``) and its mirror, the real as the prediction, on
    the natural and the transplanted candidates, ``swap`` and the
    locality."""
    size = sets.grid.shape[-1]
    s = size // patch // COARSE
    picks: dict[str, Any] = {
        "cand": sets.cand,
        "hard_q": sets.hard_q,
        "hard_among": sets.hard_among,
    }
    rows: list[dict[str, Any]] = []
    verdict: dict[str, dict[str, bool]] = {"local": {}, "mirror": {}}
    for code_name, code in (
        ("local", ColorCode(patch)),
        ("mirror", MirrorColorCode(patch)),
    ):
        t_grid = _tokens(code.tokens, sets.grid, dev)
        t_ref = _tokens(code.tokens, sets.ref, dev)
        ok = verdict[code_name]
        swaps = []
        for qa, qb in SWAP_PAIRS.values():
            pa = _fine(EVAL_MASKS[qa], s).nonzero().flatten()
            pb = _fine(EVAL_MASKS[qb], s).nonzero().flatten()
            real = {
                q: (
                    t_grid[sets.cand][:, p].flatten(1).double(),
                    t_ref[:, p].flatten(1).double(),
                )
                for q, p in ((qa, pa), (qb, pb))
            }
            for a, b in ((qa, qb), (qb, qa)):
                r_a = real[a][0] - real[a][1].mean(0)
                swaps.append(
                    _swap_values(real[a][0], real[b][0], real[a][1], real[b][1], r_a)
                )
        per = torch.stack(swaps).mean(0)
        rows.append({"part": "swap", "code": code_name, "set": "easy", "stage": "oracle",
                     "swap": float(per.mean())})  # fmt: skip
        if code_name == "mirror":
            ok["swap"] = bool((per == 1.0).all())
        groups = {}
        store: dict[tuple[Any, ...], dict[str, Scores]] = {}
        for q in QUARTER_NAMES:
            inside, seam_u, seam_v = quarter_cells(q, s)
            q_pos = inside.nonzero().flatten()
            real_g = t_grid[:, q_pos].flatten(1).double()
            mu_t = t_ref[:, q_pos].flatten(1).double().mean(0)
            query = (real_g[sets.cand], mu_t)
            nat = stage_scores(*query, real_g, mu_t, **picks)
            store.setdefault(("nat", None, "oracle"), {})[q] = nat
            if code_name == "mirror":
                ok[f"nat/{q}"] = all(
                    nat[(st, "cos")]["top1"] == 1.0 for st in ("easy", "hard")
                )
            pixels = pixel_mask(EVAL_MASKS[f"Q_{q}"], size)
            for j_i, j in enumerate(sets.contexts.tolist()):
                x_j, t_j = sets.ref[j : j + 1], t_ref[j]
                for direction in DIRECTIONS:
                    local = (t_j, t_grid) if direction == "fwd" else None
                    vec_g, sums, sums3 = _transplanted(
                        code.tokens,
                        sets.grid,
                        x_j,
                        pixels,
                        q_pos,
                        direction,
                        dev,
                        local,
                    )
                    vec_r, _, _ = _transplanted(
                        code.tokens, sets.ref, x_j, pixels, q_pos, direction, dev
                    )
                    got = stage_scores(*query, vec_g, vec_r.mean(0), **picks)
                    store.setdefault((direction, j_i, "oracle_tr"), {})[q] = got
                    tag = f"{direction}/{q}/{j_i}"
                    natural_like = (
                        code_name == "local"
                        if direction == "fwd"
                        else code_name == "mirror"
                    )
                    if natural_like:
                        ok[tag] = _same(got, nat)
                    else:
                        ok[tag] = _all_tie(got) and all(
                            v["A"] == 0.5 and v["top1"] == 0.0 for v in got.values()
                        )
                    if sums is not None and sums3 is not None:
                        g = _groups(sums, inside, seam_u, seam_v)
                        groups[(q, j_i)] = (g, _groups(sums3, inside, seam_u, seam_v))
                        a = _terms(g)
                        a_u, a_v = a["a_U"], a["a_V"]
                        if code_name == "local":
                            ok[f"a/{q}/{j_i}"] = (
                                a_u is not None and a_v is not None
                                and abs(a_u - 1) <= sconf.local_tol
                                and abs(a_v) <= sconf.local_tol
                            )  # fmt: skip
                        else:
                            ok[f"a/{q}/{j_i}"] = a_u == 0.0

        def labels(key: tuple[Any, ...], code_name: str = code_name) -> dict[str, Any]:
            kind, j, stage = key
            return {
                "part": "id",
                "code": code_name,
                "cand": kind,
                "j": j,
                "stage": stage,
            }

        rows += _id_rows(store, labels)
        rows += _local_rows(groups, len(sets.contexts), {"code": code_name})
    for name, code_name in (("V-s4", "local"), ("V-s5", "mirror")):
        each = verdict[code_name]
        rows.append(
            {"part": "check", "name": name, "pass": all(each.values()), "each": each}
        )
    return rows


# -- the jobs ------------------------------------------------------------------------------


def _write(
    path: Path, rows: Sequence[Mapping[str, Any]], stamp: Mapping[str, Any]
) -> None:
    """``rows`` with ``stamp`` as compact JSON to ``path`` atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps([dict(stamp) | dict(r) for r in rows], separators=(",", ":"))
    with atomic_open(path) as f:
        f.write((data + "\n").encode())


def _depends(
    fconf: FillConf, sconf: SelfConf, sets: SelfSets, extra: Mapping[str, Any]
) -> dict[str, Any]:
    """What the rows depend on beside the checkpoint, as JSON values."""
    return json.loads(
        json.dumps(
            {
                "fill": asdict(fconf),
                "self": asdict(sconf),
                "ident": dict(sets.ident),
                "contexts": sets.contexts.tolist(),
                "context_places": [list(p) for p in sets.places],
                "seeds": {"noise": NOISE_SEED, "perturb": PERTURB_SEED},
                "cand": sets.cand.tolist(),
                "n_hard": len(sets.hard_q),
            }
            | dict(extra)
        )
    )


def _start(
    out: Path, depends: Mapping[str, Any], determinism: Mapping[str, Any]
) -> None:
    """``out/config.json``: written if absent, else checked.

    Raises:
        ValueError: The ``config.json`` there is of other settings.
    """
    path = out / "config.json"
    if path.exists():
        old = json.loads(path.read_text())
        if {k: old.get(k) for k in depends} != dict(depends):
            raise ValueError(f"{path} is of other settings: remove it first")
        return
    out.mkdir(parents=True, exist_ok=True)
    record = dict(depends) | {
        "determinism": dict(determinism),
        "provenance": {k: v for k, v in of_diagnosis({}).items() if k != "settings"},
    }
    with atomic_open(path) as f:
        f.write((json.dumps(record, indent=2) + "\n").encode())


def _conf_of(run_dir: Path, device: str | None) -> RunConf:
    conf = from_dict(yaml.safe_load((run_dir / "config.yaml").read_text()))
    return conf if device is None else replace(conf, device=device)


def _of_transplant(row: Mapping[str, Any]) -> bool:
    """Whether the row is of the transplant (``<step>-tr.json``)."""
    return (
        row["part"] == "local"
        or row.get("cand") in DIRECTIONS
        or row.get("name") == "V-s3-L"
    )


def evaluate_run(
    run_dir: Path,
    *,
    steps: Sequence[int],
    transplant_steps: Sequence[int] = (),
    fconf: FillConf = FillConf(),
    sconf: SelfConf = SelfConf(),
    root: Path | None = None,
    device: str | None = None,
    spacing: Mapping[str, Any] | None = None,
    span: float = SPAN,
) -> Path:
    """The rows of the checkpoints of ``steps`` (and of the transplant, of
    ``transplant_steps``) of the JEPA run in ``run_dir`` into ``run_dir /
    OUT``; a step whose file is there is skipped. Returns the directory.

    Raises:
        ValueError: A step has no checkpoint, or the ``config.json`` there is
            of other settings.
    """
    conf = _conf_of(run_dir, device)
    ckpts = run_ckpts(run_dir)
    todo = sorted({*steps, *transplant_steps})
    missing = [x for x in todo if x not in ckpts]
    if missing:
        raise ValueError(f"no checkpoint of steps {missing} (of {list(ckpts)})")
    determinism = configure_determinism(conf.device)
    dev = torch.device(conf.device)
    root = tours_root(conf.data.tours.data_version) if root is None else root
    sets = self_sets(
        conf, root, fconf, contexts=sconf.contexts, span=span, spacing=spacing
    )
    settings = json.loads((run_dir / "jepa.json").read_text())
    expand, k_run = settings.get("expand", 1), settings["k"]
    mods = build_run_jepa(conf, settings)
    out = run_dir / OUT
    extra = {"run": run_dir.name, "k": k_run, "expand": expand}
    _start(out, _depends(fconf, sconf, sets, extra), determinism)
    for step in todo:
        nat_path = out / "rows" / f"{step:06d}.json"
        tr_path = out / "rows" / f"{step:06d}-tr.json"
        nat = step in steps and not nat_path.exists()
        tr = step in transplant_steps and not tr_path.exists()
        if not (nat or tr):
            continue
        ckpt.load_eval(ckpts[step], mods.modules())
        rows = step_rows(
            mods, sets, sconf, expand=expand, k_run=k_run, transplant=tr, dev=dev
        )
        stamp = {"run": run_dir.name, "step": step, "t": float(ckpts[step].stem)}
        if nat:
            _write(nat_path, [r for r in rows if not _of_transplant(r)], stamp)
        if tr:
            _write(tr_path, [r for r in rows if _of_transplant(r)], stamp)
    return out


def evaluate_checks(
    conf: RunConf,
    out: Path,
    *,
    fconf: FillConf = FillConf(),
    sconf: SelfConf = SelfConf(),
    root: Path | None = None,
    spacing: Mapping[str, Any] | None = None,
    span: float = SPAN,
) -> Path:
    """V-s4 and V-s5 on ``conf``'s data (``check_rows``; the colour codes of
    the patch of a 4 x 4 grid) into ``out`` (``config.json``,
    ``rows.json``). Returns ``out``."""
    determinism = configure_determinism(conf.device)
    root = tours_root(conf.data.tours.data_version) if root is None else root
    sets = self_sets(
        conf, root, fconf, contexts=sconf.contexts, span=span, spacing=spacing
    )
    patch = conf.data.tours.size // COARSE
    _start(out, _depends(fconf, sconf, sets, {"patch": patch}), determinism)
    rows = check_rows(sets, sconf, patch=patch, dev=torch.device(conf.device))
    _write(out / "rows.json", rows, {})
    return out


def same_weights(paths: Sequence[Path]) -> bool:
    """Whether the checkpoints ``paths`` hold the same modules' weights to
    the bit (the screen's step 0, ``plans/jepa.md``'s note)."""
    trees = [
        torch.load(p, map_location="cpu", weights_only=True)["modules"] for p in paths
    ]
    first = trees[0]
    for tree in trees[1:]:
        if tree.keys() != first.keys():
            return False
        for name, state in tree.items():
            if state.keys() != first[name].keys():
                return False
            if not all(torch.equal(state[k], first[name][k]) for k in state):
                return False
    return True


def _ints(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split(",") if x)


def main(argv: Sequence[str] | None = None) -> Path | bool:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = p.add_subparsers(dest="job", required=True)
    run = sub.add_parser("run")
    run.add_argument("run_dir", type=Path)
    run.add_argument("--steps", type=_ints, required=True)
    run.add_argument("--transplant-steps", type=_ints, default=())
    checks = sub.add_parser("checks")
    checks.add_argument("--config", type=Path, required=True)
    checks.add_argument("--out", type=Path, required=True)
    same = sub.add_parser("same")
    same.add_argument("run_dirs", type=Path, nargs="+")
    same.add_argument("--step", type=int, default=0)
    for q in (run, checks):
        q.add_argument("--device", default=None)
        q.add_argument("--root", type=Path, default=None)
    a = p.parse_args(argv)
    if a.job == "run":
        return evaluate_run(
            a.run_dir, steps=a.steps, transplant_steps=a.transplant_steps,
            root=a.root, device=a.device,
        )  # fmt: skip
    if a.job == "checks":
        conf = _conf_of(a.config, a.device)
        return evaluate_checks(conf, a.out, root=a.root)
    got = same_weights([run_ckpts(d)[a.step] for d in a.run_dirs])
    print(json.dumps({"same": got, "runs": [d.name for d in a.run_dirs]}))
    return got


if __name__ == "__main__":
    main()
