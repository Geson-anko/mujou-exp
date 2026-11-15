r"""DIAGNOSIS ONLY (M17): the Gaussian model of the token SIGReg and the windows
of lambda it gives, from the measurements of ``lam_calib``.

Backs, in ``exp/video/results/m17.md``: the section 「$\lambda$ の格子の導出」 of
「トークンの SIGReg」 (``stack``: the window $[1.72, 3.79] \times 10^{-3}$ and
$[1.74, 3.83] \times 10^{-3}$ from the measured $\rho$, $S(1) = 542$, the
slope 246.5) and 「回す前に決めたこと」 of 「ストリームごとのバッファ」
(``buffer``: $k_C$ = 1225, $S_C$, $S_P$, the windows and grids of $B$ = 32
and 1, $k_P$). Run on the measurements kept in
``exp/video/results/m17-calib/``::

    python -m exp.video.diag.lam_window stack exp/video/results/m17-calib/lam_calib_d16.json exp/video/results/m17-calib/lam_calib_d32.json
    python -m exp.video.diag.lam_window buffer exp/video/results/m17-calib/lam_calib2_b32.json exp/video/results/m17-calib/lam_calib2_b1.json

A path is a JSON file of one result, or the log of ``lam_calib`` (its line
``RESULT {...}``). CPU, no randomness.

**The model.** Along a unit direction $a$, the samples $x_1, \dots, x_N$ of
SIGReg (``exp.lib.sigreg.epps_pulley``, weight $e^{-t^2/2}$) are taken as
marginally $N(0, 1)$ and jointly Gaussian with correlations $r_{jl}$. Then
$\mathbb{E}\,\lvert \hat\varphi_N(t) - e^{-t^2/2} \rvert^2 = 1/N + N^{-2}
\sum_{j \ne l} e^{-t^2 (1 - r_{jl})} - e^{-t^2}$, and integrating $N$ times it
against $e^{-t^2/2}$ over the line gives exactly

$$\mathbb{E} S = S_0 + \frac1N \sum_{j \ne l} f(r_{jl}),\quad
f(r) = \sqrt{2\pi}\Big(\frac{1}{\sqrt{3 - 2r}} - \frac{1}{\sqrt 3}\Big),\quad
S_0 = f(1) = \sqrt{2\pi}\,(1 - 1/\sqrt 3) = 1.0594 .$$

``stack`` (one stack of $T$ tokens, streams independent, no buffer): the
healthy solution H has $r = \rho$ within a stack and $L_H = 1 - \rho$, so
$S(\rho) = S_0 + (T - 1) f(\rho)$, whatever the number of stacks. The
collapse within the stack C has $S(1) = T S_0$, $L = 0$; the iid tokens I have
$S_0$, $L = 1$. H beats C iff $\lambda > \lambda_{lo} = (1 - \rho) / (S(1) -
S(\rho) + 1 - \rho)$ and beats I iff $\lambda < \lambda_{hi} = \rho / (S(\rho)
- S_0 + \rho)$ (the objective is $(1 - \lambda) L + \lambda S$).

``buffer`` (the per-stream past tokens: capacity ``CAP`` = 2048 and ``INS`` =
512 inserted per stream and frame, so $n_k = \mathrm{INS}\, q^{k-1}$ rows of
age $k$ frames, $q = 1 - \mathrm{INS}/\mathrm{CAP}$, $G = T + \mathrm{CAP}$
samples per stream): for a solution whose tokens have correlation $r_0$ within
a stack and $r_k$ between stacks $k$ frames apart, the sum runs over the
ordered pairs of one stream (``S_closed``). H($\rho$): $r_0 = \rho$, $r_k =
\rho \kappa_k$ ($\kappa_k$ the measured lag correlation of the stack sums over
the within-stack $\rho$, beyond the $K$ = 48 lags measured 0), $L = 1 -
\rho$. C: $r_0 = 1$, $r_k = \kappa_{C,k}$, $L = 0$. I: $S_0$, $L = 1$. P
(tokens of the position alone, $L = 0$): pairs of the same position have $r =
1$ over the whole sample of all streams, $\mathbb{E} S_P = S_0 + f(1) / N
\sum_p \mathbb{E}\, n_p (n_p - 1)$ (``S_P_closed``).
"""

import argparse
import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

SQ = math.sqrt(2 * math.pi)
T, CAP, INS = 512, 2048, 512  # the tokens of a stack, the per-stream buffer
G = T + CAP  # the samples of one stream
q = 1 - INS / CAP  # the chance that a buffer row survives a frame
KMAX = 200  # the ages summed (q^KMAX ~ 1e-25)


def f(r: float) -> float:
    """$f(r)$ of the module docstring: what a pair of correlation ``r`` adds
    to $N\\,\\mathbb{E} S$."""
    return SQ * (1 / math.sqrt(3 - 2 * r) - 1 / math.sqrt(3))


def fprime(r: float) -> float:
    """$f'(r)$."""
    return SQ / (3 - 2 * r) ** 1.5


S0 = f(1.0)  # E S of iid N(0, 1) samples, at any N
n = [INS * q ** (k - 1) for k in range(1, KMAX + 1)]  # rows of age k = 1..KMAX


def S_stack(rho: float, t: int = T) -> float:
    """$\\mathbb{E} S$ of stacks of ``t`` tokens of correlation ``rho``
    within a stack, independent between stacks, no buffer."""
    return S0 + (t - 1) * f(rho)


def stack_window(rho: float, t: int = T) -> tuple[float, float]:
    """$(\\lambda_{lo}, \\lambda_{hi})$ of H($\\rho$) against C and I, no
    buffer (module docstring)."""
    S_H, S_C = S_stack(rho, t), S_stack(1.0, t)
    return (1 - rho) / (S_C - S_H + 1 - rho), rho / (S_H - S0 + rho)


def S_closed(r0: float, rk: Callable[[int], float]) -> float:
    """E S of a solution with within-stack r0 and lag correlations rk(k)."""
    tot = T * (T - 1) * f(r0)
    tot += 2 * T * sum(n[k - 1] * f(rk(k)) for k in range(1, KMAX + 1))
    tot += sum(
        INS * (INS - 1) * (q ** (k - 1)) ** 2 * f(r0) for k in range(1, KMAX + 1)
    )
    for k1 in range(1, KMAX + 1):
        for k2 in range(1, KMAX + 1):
            if k1 != k2:
                tot += n[k1 - 1] * n[k2 - 1] * f(rk(abs(k1 - k2)))
    return S0 + tot / G


def S_closed_slope(rk_over_r: Callable[[int], float]) -> float:
    """dS_H/drho at rho = 0 (linear in the pair sums, f'(0) each)."""
    pairs = T * (T - 1) + sum(
        INS * (INS - 1) * (q ** (k - 1)) ** 2 for k in range(1, KMAX + 1)
    )
    pairs += 2 * T * sum(n[k - 1] * rk_over_r(k) for k in range(1, KMAX + 1))
    for k1 in range(1, KMAX + 1):
        for k2 in range(1, KMAX + 1):
            if k1 != k2:
                pairs += n[k1 - 1] * n[k2 - 1] * rk_over_r(abs(k1 - k2))
    return fprime(0.0) * pairs / G


def S_P_closed(B: int) -> float:
    """E S_P: n_p = B (current) + m_p (buffer), m_p = sum_k Bin(B, q^(k-1))."""
    N = B * G
    mean_m = B * sum(q ** (k - 1) for k in range(1, KMAX + 1))
    var_m = B * sum(q ** (k - 1) * (1 - q ** (k - 1)) for k in range(1, KMAX + 1))
    e_nn = (B + mean_m) ** 2 + var_m - (B + mean_m)
    return S0 + f(1.0) * T * e_nn / N


def window(S_H: float, S_C: float, S_P: float, rho: float) -> dict[str, Any]:
    """The bounds of lambda where H($\\rho$) beats C (``lo_C``), P (``lo_P``)
    and I (``hi_I``), and whether the windows are empty."""
    L_H = 1 - rho
    lo_C = L_H / (S_C - S_H + L_H)
    hi_I = rho / (S_H - S0 + rho)
    lo_P = L_H / (S_P - S_H + L_H) if S_P > S_H else float("inf")
    return {
        "lo_C": lo_C,
        "lo_P": lo_P,
        "hi_I": hi_I,
        "empty_vs_P": not lo_P < hi_I,
        "empty_vs_C": not lo_C < hi_I,
    }


def grid(lo: float, hi: float) -> list[float]:
    """The ends and their geometric mean."""
    return [lo, math.sqrt(lo * hi), hi]


def load_result(path: Path) -> dict[str, Any] | None:
    """The result of ``lam_calib`` in ``path``: its line ``RESULT {...}``,
    else the whole file as JSON (None if neither)."""
    text = path.read_text()
    for line in text.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def stack_report(res: dict[str, Any]) -> None:
    """Print $\\rho$, $S(\\rho)$ and the window of ``stack`` for the training
    stream and val of a result of ``lam_calib pooled``."""
    t = res["T"]
    print(f"===== d={res['d']} T={t}  S0={S0:.4f}  S(1)={S_stack(1.0, t):.1f}")
    slope = (t - 1) * fprime(0.0)
    print(f"  dS/drho at 0 = {slope:.1f}  1/S(1) = {1 / S_stack(1.0, t):.3e}")
    for split in ("train", "val"):
        rho = res[split]["rho"]
        lo, hi = stack_window(rho, t)
        print(
            f"  {split:5s} rho={rho:.4f}  S(rho)={S_stack(rho, t):.2f}  "
            f"window=[{lo:.3e}, {hi:.3e}]  grid={[f'{x:.2g}' for x in grid(lo, hi)]}"
        )


def buffer_report(res: dict[str, Any]) -> None:
    """Print the measurements, the closed forms and the windows of ``buffer``
    for a result of ``lam_calib per-stream``."""
    B = res["B"]
    N = B * G
    c = res["corr"]
    Sm = res["S"]
    print(
        f"===== B={B} d={res['d']}  N={N} (N/512={N / 512:.0f})  count={Sm['count']:.0f}"
    )
    for key in ("h_pop", "h_rec", "hw_pop", "hw_rec", "white_pop", "white_rec"):
        lag = c[key]["lag"]
        print(
            f"  {key:10s} rho={c[key]['rho']:.4f} var_u={c[key]['var_u']:.3f} "
            f"lag1..8={[round(x, 3) for x in lag[:8]]} lag16,32,48={[round(lag[i], 3) for i in (15, 31, 47)]}"
        )
    for k in ("h", "hw", "white", "P", "Popt", "I"):
        print(
            f"  S_{k:5s} nobuf {Sm['S_' + k]:9.2f} ({Sm['S_' + k + '_sd']:.1f})   "
            f"buf {Sm['S_' + k + '_buf']:9.2f} ({Sm['S_' + k + '_buf_sd']:.1f})"
        )
    print(f"  Popt S at N=T: {res['Popt_S_at_T']:.3f}")
    SPc = S_P_closed(B)
    print(
        f"  S_P closed (iid g) = {SPc:.1f}; simulated {Sm['S_P_buf']:.1f}; Popt simulated {Sm['S_Popt_buf']:.1f}"
    )
    kC_lag, kC_r = c["white_pop"]["lag"], c["white_pop"]["rho"]

    def kapC(k: int, L: list[float] = kC_lag, r: float = kC_r) -> float:
        """kappa_C = the lag correlation / the within (the variance seen)."""
        return min(L[k - 1] / r, 1.0) if k <= len(L) else 0.0

    S_C = S_closed(1.0, kapC)
    print(
        f"  S_C closed (kappa_C measured) = {S_C:.1f}; kappa=1: {S_closed(1.0, lambda k: 1.0):.1f}; measured S_white_buf {Sm['S_white_buf']:.1f}"
    )
    for rkey in ("h_rec", "h_pop", "hw_pop"):
        rho = c[rkey]["rho"]
        L = c["hw_pop"]["lag"]
        rho_hw = c["hw_pop"]["rho"]

        def kapH(k: int, L: list[float] = L, r: float = rho_hw) -> float:
            return (L[k - 1] / r) if k <= len(L) else 0.0

        def kapH_c(k: int) -> float:
            return min(kapH(k), 1.0)

        S_H = S_closed(rho, lambda k, rho=rho: rho * kapH_c(k))
        slope = S_closed_slope(kapH_c)
        w = window(S_H, S_C, Sm["S_P_buf"], rho)
        wPc = window(S_H, S_C, SPc, rho)
        wPopt = window(S_H, S_C, Sm["S_Popt_buf"], rho)
        print(
            f"  [rho from {rkey}] rho={rho:.4f}  S_H(rho) closed={S_H:.1f}  slope k_C={slope:.1f}  "
            f"(S_H-S0)/rho={(S_H - S0) / rho:.1f}  vs S_P-S0={Sm['S_P_buf'] - S0:.1f} (Popt {Sm['S_Popt_buf'] - S0:.1f})"
        )
        print(f"     window (S_P sim iid g): {w}")
        print(f"     window (S_P closed):    {wPc}")
        print(f"     window (S_P Popt):      {wPopt}")
        print(f"     C/I grid: {[f'{x:.3g}' for x in grid(w['lo_C'], w['hi_I'])]}")
    # measured values in place of the model
    rho = c["hw_pop"]["rho"]
    wm = window(Sm["S_hw_buf"], Sm["S_white_buf"], Sm["S_P_buf"], rho)
    print(f"  measured S (hw, white, P, buffer), rho hw_pop: {wm}")
    kP = fprime(0.0) * (N / T - 1)
    print(
        f"  marginal cost per unit variance at 0: position k_P = f'(0)(N/T-1) = {kP:.1f}; "
        f"P endpoint f(1)(N/T-1)={f(1.0) * (N / T - 1):.1f}"
    )


def main(argv: Sequence[str] | None = None) -> None:
    """``stack`` or ``buffer`` (module docstring) over the result files."""
    p = argparse.ArgumentParser(description="The lambda windows of M17 (diagnosis).")
    p.add_argument("mode", choices=("stack", "buffer"))
    p.add_argument("paths", nargs="+", type=Path)
    args = p.parse_args(argv)
    report = stack_report if args.mode == "stack" else buffer_report
    for path in args.paths:
        res = load_result(path)
        if res is None:
            print(path, "no RESULT")
            continue
        report(res)


if __name__ == "__main__":
    main()
