r"""DIAGNOSIS ONLY (M17): simulations of the SIGReg statistic $S$
(``exp.lib.sigreg``: the Epps-Pulley statistic of the samples along 256 random
unit directions, averaged) on synthetic samples, CPU, ``torch.manual_seed(0)``
at the start of each command (the numbers below are those of that seed).

Every sample is built so that its marginal along any direction is exactly
$N(0, 1)$; what differs is the dependence between the samples, which is all
that $S$ can see beyond the marginal (``lam_window``: $\mathbb{E} S = S_0 +
\frac1N \sum_{j \ne l} f(r_{jl})$, $S_0 = 1.0594$).

- ``n``: the tokens of the position alone P, $s(x, p) = g(p)$, $g(p)$ of the
  $T$ = 512 positions iid $N(0, I_d)$ ($d$ = 16) drawn once: $B$ stacks of the
  same 512 points and a buffer of ``cap`` rows drawn uniformly from them,
  for $(B, \mathrm{cap})$ in (1, 0), (1, 8192), (32, 0), (32, 8192),
  (32, 65536), (32, 262144); 3 draws each. Backs the table of $S_P$ against
  $N/512$ in 「P の SIGReg と標本数（導出）」 (``exp/video/results/m17.md``)
- ``position``: the same P at $B$ = 32, without and with a buffer of 8192
  rows, against $B \cdot 512$ iid $N(0, I_d)$ tokens, $d$ = 16 and 32, 5 draws
  each: the check behind the same section
- ``tokens``: the Gaussian model of the token SIGReg (「$\lambda$ の格子の導出」):
  $n$ stacks of $m$ = 512 tokens $s = \sqrt\rho\, c + \sqrt{1 - \rho}\, e$
  ($c$ one per stack, $e$ one per token, both $N(0, I_{16})$), so $\rho$ is
  the correlation of two tokens of a stack; $S$ of the $nm$ tokens as samples
  of a 16-dimensional law, for $n$ = 32 and 256 and $\rho$ from 0 to 1, 3
  draws each. Its mean is $S_0 + (m - 1) f(\rho)$ at any $n$
  (``lam_window.S_stack``); the numbers are not quoted in the results
- ``power``: SIGReg of one sample per data point: $N$ samples of $D = T d$
  dimensions ($T$ = 512, $d$ = 16 and 32), iid $N(0, I_D)$ against Gaussians
  of rank $r$ scaled to unit marginal variance (4096, 1024, 256) and the
  collapse within the stack (one $N(0, I_d)$ token repeated $T$ times,
  $r = d$), for $N$ = 32, 256, 2048, 8192; 3 draws each. The directions are
  drawn once per $d$. Not quoted in the results
- ``scale``: the token SIGReg with a buffer of 8192 rows ($d$ = 16), $B$ = 1
  and 32 stacks: healthy (iid) tokens and stacks collapsed within (one token
  per stack), with a buffer of healthy past tokens or of collapsed past stacks
  (64 rows per stack). Not quoted in the results

Run: ``python -m exp.video.diag.sigreg_sims <command>`` (``power`` takes a
few minutes and some GB of memory).
"""

import argparse
from collections.abc import Sequence

import torch

from exp.lib.sigreg import epps_pulley, sigreg

T = 512  # the tokens of a stack
M = 256  # the directions


def unit_rows(a: torch.Tensor) -> torch.Tensor:
    """The rows of ``a`` divided by their norms."""
    return a / a.norm(dim=1, keepdim=True)


def position_only(
    t: int, d: int, b: int, cap: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The points ``g`` ``(t, d)`` iid $N(0, I_d)$, ``M`` unit directions, the
    current samples of P (``g`` repeated for ``b`` stacks, ``(b t, d)``) and a
    buffer of ``cap`` rows of ``g`` drawn uniformly, in this order from the
    global generator."""
    g = torch.randn(t, d)
    dirs = unit_rows(torch.randn(M, d))
    cur = g.repeat(b, 1)
    buf = g[torch.randint(t, (cap,))]
    return g, dirs, cur, buf


def gauss_stacks(
    n: int,
    m: int,
    d: int,
    rho: float,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    r"""``n`` stacks of ``m`` tokens ``(n, m, d)`` of the Gaussian model
    $\sqrt\rho\, c + \sqrt{1 - \rho}\, e$ (module docstring, ``tokens``)."""
    c = torch.randn(n, 1, d, generator=generator, device=device)
    e = torch.randn(n, m, d, generator=generator, device=device)
    return rho**0.5 * c + (1 - rho) ** 0.5 * e


def cmd_n() -> None:
    torch.manual_seed(0)
    d = 16
    for B, cap in ((1, 0), (1, 8192), (32, 0), (32, 8192), (32, 65536), (32, 262144)):
        out = []
        for _ in range(3):
            _, dirs, cur, buf = position_only(T, d, B, cap)
            out.append(float(sigreg(cur, buf, dirs)))
        print(
            f"B={B} cap={cap} N/512={(B * T + cap) / T:.0f}: S_P={sum(out) / len(out):.1f}"
        )


def cmd_position() -> None:
    torch.manual_seed(0)
    B, cap = 32, 8192
    for d in (16, 32):
        out = []
        for _ in range(5):
            _, dirs, cur, buf = position_only(T, d, B, cap)
            s_nobuf = float(sigreg(cur, cur[:0], dirs))
            s_buf = float(sigreg(cur, buf, dirs))
            iid = float(sigreg(torch.randn(B * T, d), cur[:0], dirs))
            out.append((s_nobuf, s_buf, iid))
        m = torch.tensor(out).mean(0).tolist()
        print(
            f"d={d}: position-only S without buffer {m[0]:.1f}, with buffer {m[1]:.1f}; iid N(0,I) {m[2]:.2f}"
        )


def cmd_tokens() -> None:
    torch.manual_seed(0)
    d = 16
    a = unit_rows(torch.randn(M, d))
    for n in (32, 256):
        row = []
        for rho in (0.0, 0.001, 0.01, 0.05, 0.2, 0.5, 1.0):
            v = []
            for _ in range(3):
                s = gauss_stacks(n, T, d, rho).reshape(n * T, d)
                v.append(float(epps_pulley(s @ a.T).mean()))
            row.append(f"rho={rho}:{sum(v) / 3:.1f}")
        print(f"n={n} stacks, N={n * T} tokens  " + "  ".join(row), flush=True)


def cmd_power() -> None:
    torch.manual_seed(0)
    for d in (16, 32):
        D = T * d
        a = unit_rows(torch.randn(M, D))
        for N in (32, 256, 2048, 8192):
            out = []
            for r in (D, 4096, 1024, 256, d):
                reps = []
                for _ in range(3):
                    if r == D:
                        x = torch.randn(N, D)
                    elif r == d:  # collapse: one d-dim token repeated T times
                        x = torch.randn(N, d).repeat(1, T)
                    else:
                        basis = torch.linalg.qr(torch.randn(D, r))[0]  # (D, r)
                        x = torch.randn(N, r) @ basis.T * (D / r) ** 0.5
                    reps.append(float(epps_pulley(x @ a.T).mean()))
                out.append(f"r={r}:{sum(reps) / 3:.2f}")
            print(f"d={d} D={D} N={N}  " + "  ".join(out), flush=True)


def cmd_scale() -> None:
    torch.manual_seed(0)
    d, C, k = 16, 8192, 64

    def S(cur: torch.Tensor, buf: torch.Tensor) -> float:
        return sigreg(cur, buf, a).item()

    a = unit_rows(torch.randn(M, d))
    for B in (1, 32):
        healthy = torch.randn(B * T, d)
        coll = torch.randn(B, 1, d).expand(B, T, d).reshape(-1, d)
        bh = torch.randn(C, d)
        # past collapsed: C/k stacks each k copies
        bc = torch.randn(C // k, 1, d).expand(C // k, k, d).reshape(-1, d)
        print(
            f"B={B}: no buf healthy {S(healthy, bh[:0]):.1f} collapse {S(coll, bh[:0]):.1f} | "
            f"buf(healthy past) healthy {S(healthy, bh):.1f} collapse {S(coll, bh):.1f} | "
            f"buf(collapsed past) collapse {S(coll, bc):.1f}"
        )


COMMANDS = {
    "n": cmd_n,
    "position": cmd_position,
    "tokens": cmd_tokens,
    "power": cmd_power,
    "scale": cmd_scale,
}


def main(argv: Sequence[str] | None = None) -> None:
    """One command of the module docstring."""
    p = argparse.ArgumentParser(description="SIGReg simulations of M17 (diagnosis).")
    p.add_argument("command", choices=tuple(COMMANDS))
    COMMANDS[p.parse_args(argv).command]()


if __name__ == "__main__":
    main()
