r"""DIAGNOSIS ONLY (M17, 2026-11-13): the window of $\lambda$ of the SIGReg of
the data points on one stream (``m17/ssig_*``), from the stream correlation
of the samples, for each span $k$ of the buffer.

The setting is that of ``exp.video.diag.ssig`` (one stream, $B$ = 1): SIGReg
sees $n = N + 1$ samples, the current data point $x_0$ and a buffer of $N$
past ones (``jepa.nodes.PastPoints``), each $x \in \mathbb{R}^{Td}$ the
target tokens of a stack flattened. The loss is $J = (1 - \lambda) L +
\lambda S$ (``jepa.loss.jepa_loss_packed``), $L$ the prediction error per
dimension. As ``exp.video.diag.lam_window`` did for the tokens, it compares a
healthy solution with the trivial ones at the level of the data points.

**The ages of the samples.** At frame $j$ of the stream (0.1 s each), with
phase $\phi = j \bmod k$, the buffer row inserted $m$ insertions ago holds
the data point of frame $j - \phi - 1 - km$ (``PastPoints.next_frame``
inserts the point of frame $j - 1$ when $j$ is a multiple of $k$). Full, an
insertion replaces a uniform row, so the rows of $m_1 < m_2$ insertions ago
are both there with probability $(1 - 2/N)^{m_1} (1 - 1/N)^{m_2 - m_1}$ and
one alone with $q^m$, $q = 1 - 1/N$. The expected number of ordered pairs of
the $n$ samples at a lag $\tau$ (frames) is then (``pair_weights``)

$$W(\tau) = \frac2k \sum_{m \ge 0} \sum_{\phi = 0}^{k - 1} q^m
[\tau = 1 + \phi + km] + N \sum_{d \ge 1} q^d [\tau = kd],$$

the current with a row (phase uniform), then two rows (their phases cancel;
$\sum_{m_1} (1 - 2/N)^{m_1} = N/2$), and $\sum_\tau W = N(N + 1) = n(n - 1)$.

**The statistic.** Along a unit direction $a$ the samples are taken jointly
Gaussian, of mean 0, variance $v_a$, and correlation $\kappa(\tau)$ for a
pair $\tau$ frames apart, the same in every direction (the correlation of
the whitened representation: ``stream_stats``' ``white``). With $w(t) =
e^{-t^2/2}$, $n E|\hat\varphi - w|^2 = (1 - e^{-v t^2}) + n (e^{-v t^2/2} -
w)^2 + \frac1n \sum_{j \ne l} (e^{-v (1 - \kappa_{jl}) t^2} - e^{-v t^2})$,
so on the trapezoid rule of ``exp.lib.sigreg`` (``Stream``)

$$E\,S = E_a\, E S_{\mathrm{iid}}(v_a) + \frac1n \sum_\tau W(\tau)\,
G(\kappa(\tau)),\quad G(\kappa) = E_a \int (e^{-v_a (1 - \kappa) t^2} -
e^{-v_a t^2}) w\, dt,$$

$E S_{\mathrm{iid}}$ = ``ssig.expected_s``. To first order in $\kappa$ and at
$v_a = 1$ the second term is $f'(0)(n - 1)\bar r$, $\bar r = \sum_\tau
W \kappa / (n(n - 1))$ the mean correlation of the pairs and $f'(0) =
\sqrt{2\pi}/3^{3/2}$ (results/m17.md, 「定義」).

**The solutions** (the scale is 1 where $L$ depends on it, and the one of
least $S$ where $L$ = 0):

- H($\rho$): tokens $\sqrt\rho\, c + \sqrt{1 - \rho}\, e_t$ of correlation
  $\rho$ within a stack, predicted from the context by the common part:
  $L_H = 1 - \rho$. $v_a = (1 - \rho) + \rho \lVert s_a \rVert^2$ (``ssig``,
  $s_a = \sum_t a_t$), the data points correlated over the stream by
  $\kappa$: $S_H$ as above.
- C: every data point the same $z$: $L = 0$, $S_C = E S_{\mathrm{iid}}(0) =
  n \int (1 - w)^2 w\, dt$ (the constant 0 is the cheapest).
- Cw: tokens constant within a stack, independent between data points: $L$
  = 0, $S_{Cw}$ = ``ssig.bound``'s $S_c$ (H(1) without the stream, the scale
  minimized).
- I: iid $N(0, I)$ data points, independent of the stream: $L = 1$, $S_0 =
  E S_{\mathrm{iid}}(1)$.

C, Cw and I are trivial: they are defined without the data. I and Cw are
independent between moments 0.1 s apart only if the encoder reads what
tells them apart, which on near-equal stacks is mostly nuisance; they are
the end point of the force of the hypothesis. H beats a solution X of $L$ =
0 iff $\lambda > \lambda_X = L_H / (S_X - S_H + L_H)$ (never if $S_X \le
S_H$), and beats I iff $\lambda < \lambda_I = \rho / (S_H - S_0 + \rho)$.
The window is $(\max(\lambda_C, \lambda_{Cw}), \lambda_I)$, the grid its ends
and their geometric mean (``lam_window.grid``). Since $L_H + \rho = 1$, it is
not empty iff $S_H < \rho \min(S_C, S_{Cw}) + (1 - \rho) S_0$: with $S_H$
its first term plus $f'(0)(n - 1)\bar r$, iff the mean correlation of the
pairs is below ``rbar_max`` $= (\rho \min(S_C, S_{Cw}) + (1 - \rho) S_0 -
E_a E S_{\mathrm{iid}}(v_a)) / (f'(0)(n - 1))$ (to first order), a number of
the arm and $\rho$ alone, against which the $\bar r$ of any representation's
curve can be read.

The lower end rules out the code of $d$ numbers per stack only (Cw, $m = d$
in ``ssig``): codes of a latent of $m > d$ numbers that the context
determines also have $L$ = 0 and a statistic that falls toward $S_0$ as $m$
grows, so no $\lambda < 1$ makes H cheaper than them; whether the learned
code is one, and of which $m$, only its rank tells. H and I are held at the
scale 1, though the encoder may rescale them too (``ssig``'s
``lam_star_scaled`` does it for the iid tokens against Cw).

It compares points of the loss, not its minima, and holds at the steady
state of the buffer whatever it stores (with ``z`` the gradient only reaches
the current sample, which changes the path, not the values compared; its
rows are of the encoders that made them, a mean of $N k$ learning steps
before, taken here to be of the current one). One
grid serves every span of an arm when it lies in every window that is not
empty: the ends of the
intersection of those windows and their geometric mean (``common``); the
spans whose window is empty are listed (no $\lambda$ makes H the cheapest
of the four there).

$\kappa$ comes from the curves of ``stream_stats`` (``corr`` and ``repr``,
results/m17-stream/), linear between the measured lags (``window_mean``),
beyond the last one either held at its value or 0 (``tail``: the two bound
it for a curve that decreases to a non-negative value); the band of a curve
is $\kappa \pm t_{0.975}\, \hat\sigma$ (its jackknife), clipped to $[-1, 1]$.
$\rho$ is that of the untrained ViT (``lam_calib pooled``, $d$ = 16, the
training stream: 0.122, results/m17-calib/lam_calib_d16.json).

Run (CPU, seconds; directions seed 0)::

    python -m exp.video.diag.ssig_window --curves pixels/g tok_init/tokens/white
"""

import argparse
import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from exp.lib.sigreg import T_POINTS
from exp.video.diag.ssig import ARMS, bound, expected_s, spread_of_sums

__all__ = [
    "SPANS",
    "Stream",
    "common",
    "kappa_of",
    "load_curves",
    "main",
    "pair_table",
    "pair_weights",
    "solutions",
    "table",
    "windows",
]

SPANS = (1, 6, 24)  # k of the arms of m17/ssig_*
N_BUF = 256  # the buffer of m17/ssig_*
FPRIME0 = math.sqrt(2 * math.pi) / 3**1.5  # f'(0)
RHO = 0.122  # the untrained ViT's rho (d = 16, training stream, lam_calib)
TOL = 1e-15  # the survival q^m below which the ages are dropped


def pair_weights(n_buf: int, k: int, tol: float = TOL) -> tuple[Tensor, Tensor]:
    """The lags $\\tau \\ge 1$ (frames) and $W(\\tau)$, the expected number of
    ordered pairs of the current sample and the ``n_buf`` rows of a full
    buffer of span ``k`` at each lag (module docstring); ages of survival
    below ``tol`` are dropped. Returns ``(lags (L,), weights (L,))`` float64,
    the lags without a pair left out."""
    q = 1 - 1 / n_buf
    m_max = math.ceil(math.log(tol) / math.log(q)) if n_buf > 1 else 0
    m = torch.arange(m_max + 1, dtype=torch.float64)
    surv = q**m
    phi = torch.arange(k, dtype=torch.float64)
    cur_lag = (1 + phi[None, :] + k * m[:, None]).flatten()
    cur_w = (2 / k * surv[:, None]).expand(-1, k).flatten()
    d = m[1:]
    row_lag, row_w = k * d, n_buf * surv[1:]
    lag = torch.cat([cur_lag, row_lag]).long()
    w = torch.zeros(int(lag.max()) + 1, dtype=torch.float64)
    w.index_add_(0, lag, torch.cat([cur_w, row_w]))
    keep = w > 0
    return torch.arange(len(w), dtype=torch.float64)[keep], w[keep]


def kappa_of(
    lags: Sequence[float], rho: Sequence[float], tail: str
) -> Callable[[Tensor], Tensor]:
    """$\\kappa(\\tau)$ linear between the measured ``lags`` (frames, from 1)
    and beyond the last ``"hold"`` (its value) or ``"zero"``."""
    if lags[0] != 1 or tail not in ("hold", "zero"):
        raise ValueError(f"the lags start at {lags[0]}, not 1, or tail {tail!r}")
    lag = torch.tensor(lags, dtype=torch.float64)
    r = torch.tensor(rho, dtype=torch.float64)

    def at(tau: Tensor) -> Tensor:
        i = torch.searchsorted(lag, tau).clamp(1, len(lag) - 1)
        t = ((tau - lag[i - 1]) / (lag[i] - lag[i - 1])).clamp(max=1.0)
        out = r[i - 1] + t * (r[i] - r[i - 1])
        beyond = tau > lag[-1]
        return torch.where(beyond, r[-1] if tail == "hold" else 0.0, out)

    return at


def pair_table(v: Tensor, kappas: Tensor, t: Tensor = T_POINTS) -> Tensor:
    """$G(\\kappa)$ (module docstring) of the variances ``v`` ``(A,)`` of the
    directions at each of ``kappas`` ``(K,)``, by the trapezoid rule on
    ``t``."""
    t2 = t.to(torch.float64) ** 2
    w = torch.exp(-t2 / 2)
    v = v.to(torch.float64)
    base = torch.exp(-v[:, None] * t2).mean(0)  # (T,)
    out = []
    for chunk in kappas.to(torch.float64).split(64):
        e = torch.exp(-v[None, :, None] * (1 - chunk)[:, None, None] * t2)
        out.append(torch.trapezoid((e.mean(1) - base) * w, t.to(t2), dim=-1))
    return torch.cat(out)


class Stream:
    """$E S$ of ``n`` samples of the variances ``v`` ``(A,)`` along the
    directions, correlated over the stream (``__call__``); $G$ (``pair_table``)
    is tabulated once on ``nodes`` points of $[-1, 1]$ and taken linear
    between them."""

    def __init__(self, v: Tensor, n: int, nodes: int = 4001) -> None:
        self.n = n
        self.iid = float(expected_s(v, n).mean())
        self.table = pair_table(v, torch.linspace(-1, 1, nodes, dtype=torch.float64))

    def g(self, kappa: Tensor) -> Tensor:
        """$G$ at ``kappa`` (clipped to $[-1, 1]$)."""
        last = len(self.table) - 1
        pos = (kappa.clamp(-1.0, 1.0) + 1) / 2 * last
        i = pos.floor().long().clamp(max=last - 1)
        return self.table[i] + (pos - i) * (self.table[i + 1] - self.table[i])

    def __call__(
        self, lags: Tensor, weights: Tensor, kappa: Callable[[Tensor], Tensor]
    ) -> dict[str, float]:
        """With the pair weights ``(lags, weights)`` and the correlation
        ``kappa`` of a lag (module docstring): ``iid`` the first term of $E
        S$, ``pairs`` the second, ``s`` their sum, ``rbar``, and
        ``first_order`` $f'(0)(n - 1)\\bar r$."""
        n = self.n
        k = kappa(lags).clamp(-1.0, 1.0)
        rbar = float((weights * k).sum() / (n * (n - 1)))
        pairs = float((weights * self.g(k)).sum() / n)
        return {
            "iid": self.iid,
            "pairs": pairs,
            "s": self.iid + pairs,
            "rbar": rbar,
            "first_order": FPRIME0 * (n - 1) * rbar,
        }


def solutions(
    s_h: float, rho: float, s_c: float, s_cw: float, s_0: float
) -> dict[str, Any]:
    """$\\lambda_C$, $\\lambda_{Cw}$, $\\lambda_I$ of H($\\rho$) of
    statistic ``s_h`` against C, Cw and I (module docstring), the window and
    its grid (None if empty), and ``s_max`` $= \\rho \\min(S_C, S_{Cw}) +
    (1 - \\rho) S_0$, the window being non-empty iff $S_H$ < ``s_max``."""
    l_h = 1 - rho

    def lo(s_x: float) -> float:
        return l_h / (s_x - s_h + l_h) if s_x > s_h else math.inf

    lam_c, lam_cw = lo(s_c), lo(s_cw)
    lam_i = rho / (s_h - s_0 + rho)
    low = max(lam_c, lam_cw)
    grid = [low, math.sqrt(low * lam_i), lam_i] if low < lam_i else None
    s_max = rho * min(s_c, s_cw) + l_h * s_0
    return {
        "lam_C": lam_c,
        "lam_Cw": lam_cw,
        "lam_I": lam_i,
        "grid": grid,
        "s_max": s_max,
    }


def load_curves(root: Path, names: Sequence[str]) -> dict[str, dict[str, Any]]:
    """The curves ``<model>/<code>/<centre>`` (``pixels/<centre>`` of
    ``corr.json``) of the ``stream_stats`` outputs in ``root``: lags in
    frames of 0.1 s, ``rho`` and ``se``, ``t95``."""
    found: dict[str, dict[str, Any]] = {}
    c = json.loads((root / "corr.json").read_text())
    tables = [("pixels", c)]
    for path in sorted(root.glob("repr-*.json")):
        r = json.loads(path.read_text())
        tables += list(r["codes"].items())
    for label, x in tables:
        for centre, cur in x["curves"].items():
            found[f"{label}/{centre}"] = {
                "lags": [lag / x["stride"] for lag in x["lags"]],
                "rho": cur["rho"],
                "se": cur["se"],
                "t95": x["t95"],
            }
    missing = [n for n in names if n not in found]
    if missing:
        raise KeyError(f"no curves {missing} in {root} ({sorted(found)})")
    return {n: found[n] for n in names}


def windows(
    root: Path,
    names: Sequence[str],
    rhos: Sequence[float],
    spans: Sequence[int],
    n_buf: int,
    draws: int,
) -> list[dict[str, Any]]:
    """The windows of every arm, $\\rho$, curve and span (module docstring):
    of the curve (``mid``, and ``mid_zero_tail``) and of the ends of its
    band (``low``, ``high``)."""
    n = n_buf + 1
    curves = load_curves(root, names)
    s_0 = float(expected_s(torch.tensor(1.0), n))
    s_c = float(expected_s(torch.tensor(0.0), n))
    weights = {k: pair_weights(n_buf, k) for k in spans}
    out: list[dict[str, Any]] = []
    for arm, (t, d) in ARMS.items():
        s2 = spread_of_sums(t, d, draws, torch.Generator().manual_seed(0))
        s_cw = bound(t, d, n, draws=draws)["s_collapse"]
        for rho in rhos:
            stream = Stream((1 - rho) + rho * s2, n)
            for name, cur in curves.items():
                band = [cur["t95"] * s for s in cur["se"]]
                for k in spans:
                    lags, w = weights[k]
                    row: dict[str, Any] = {
                        "arm": arm,
                        "rho": rho,
                        "curve": name,
                        "k": k,
                        "S_0": s_0,
                        "S_C": s_c,
                        "S_Cw": s_cw,
                        "rbar_max": (
                            rho * min(s_c, s_cw) + (1 - rho) * s_0 - stream.iid
                        )
                        / (FPRIME0 * (n - 1)),
                    }
                    for label, sign, tail in (
                        ("mid", 0, "hold"),
                        ("mid_zero_tail", 0, "zero"),
                        ("low", -1, "hold"),
                        ("high", 1, "hold"),
                    ):
                        r = [
                            x + sign * b for x, b in zip(cur["rho"], band, strict=True)
                        ]
                        s = stream(lags, w, kappa_of(cur["lags"], r, tail))
                        row[label] = {**s, **solutions(s["s"], rho, s_c, s_cw, s_0)}
                    out.append(row)
    return out


def _g(x: float) -> str:
    return "–" if math.isinf(x) else f"{x:.3g}"


def table(rows: Sequence[dict[str, Any]]) -> str:
    """The rows as a markdown table: $\\bar r$ (band), $S_H$ and its pair
    term against $f'(0)(n-1)\\bar r$, $\\lambda_C$, $\\lambda_{Cw}$,
    $\\lambda_I$ (over the band) and the grid ("–": empty)."""
    head = [
        "arm",
        "rho",
        "curve",
        "k",
        "rbar (band)",
        "S_H",
        "pairs (1st order)",
        "lam_C",
        "lam_Cw",
        "lam_I (band)",
        "grid",
    ]
    lines = ["| " + " | ".join(head) + " |", "|" + " -- |" * len(head)]
    for r in rows:
        m, lo, hi = r["mid"], r["low"], r["high"]
        grid = "–" if m["grid"] is None else ", ".join(f"{x:.3g}" for x in m["grid"])
        cells = [
            r["arm"],
            f"{r['rho']:.3g}",
            r["curve"],
            str(r["k"]),
            f"{m['rbar']:.4f} ({lo['rbar']:.4f}〜{hi['rbar']:.4f})",
            f"{m['s']:.2f}",
            f"{m['pairs']:.2f} ({m['first_order']:.2f})",
            _g(m["lam_C"]),
            _g(m["lam_Cw"]),
            f"{_g(m['lam_I'])} ({_g(hi['lam_I'])}〜{_g(lo['lam_I'])})",
            grid,
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def common(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per arm, $\\rho$ and curve: the spans whose window (``mid``) is empty,
    and the grid of the intersection of the others (module docstring; None
    if it is empty)."""
    groups: dict[tuple[str, float, str], list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((r["arm"], r["rho"], r["curve"]), []).append(r)
    out = []
    for (arm, rho, curve), rs in groups.items():
        full = [r["mid"]["grid"] for r in rs if r["mid"]["grid"] is not None]
        lo = max((g[0] for g in full), default=math.nan)
        hi = min((g[-1] for g in full), default=math.nan)
        out.append(
            {
                "arm": arm,
                "rho": rho,
                "curve": curve,
                "rbar_max": rs[0]["rbar_max"],
                "empty": [r["k"] for r in rs if r["mid"]["grid"] is None],
                "grid": [lo, math.sqrt(lo * hi), hi] if lo < hi else None,
            }
        )
    return out


def main(argv: Sequence[str] | None = None) -> None:
    """``windows`` and ``common`` as markdown tables, and as JSON into
    ``--out``."""
    p = argparse.ArgumentParser(description="The lambda windows of m17/ssig_*.")
    p.add_argument("--dir", type=Path, default=Path("exp/video/results/m17-stream"))
    p.add_argument("--curves", nargs="+", default=["pixels/g"])
    p.add_argument("--rho", type=float, nargs="+", default=[RHO])
    p.add_argument("--spans", type=int, nargs="+", default=list(SPANS))
    p.add_argument("--buffer", type=int, default=N_BUF)
    p.add_argument("--draws", type=int, default=20000)
    p.add_argument("--out", type=Path)
    args = p.parse_args(argv)
    rows = windows(args.dir, args.curves, args.rho, args.spans, args.buffer, args.draws)
    grids = common(rows)
    if args.out is not None:
        args.out.write_text(
            json.dumps({"rows": rows, "common": grids}, indent=1) + "\n"
        )
    print(table(rows))
    print()
    print(
        "| arm | rho | rbar_max | curve | spans without a window | grid of the others |"
    )
    print("| -- | -- | -- | -- | -- | -- |")
    for g in grids:
        grid = "–" if g["grid"] is None else ", ".join(f"{x:.3g}" for x in g["grid"])
        empty = ", ".join(map(str, g["empty"])) or "none"
        print(
            f"| {g['arm']} | {g['rho']:.3g} | {g['rbar_max']:.4f} | {g['curve']} "
            f"| {empty} | {grid} |"
        )


if __name__ == "__main__":
    main()
