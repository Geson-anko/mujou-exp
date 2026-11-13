r"""DIAGNOSIS ONLY (M17, 2026-11-13): does the temporal correlation of one
stream move the fixed point of the Gaussian moment matching
(``exp.lib.gauss_kl``) and that of SIGReg (``exp.lib.sigreg``), each
estimated from the same buffer of past samples? The falsification test of
the claim that the correlation inflates the spread of the moment matching's
estimate but not its expectation.

**Stream.** $s_t \in \mathbb{R}^m$ (``dim``), each coordinate an independent
stationary AR(1), $s_t = a s_{t-1} + \sqrt{1 - a^2}\, e_t$, $a = e^{-1/\tau}$
($\tau$ = 0: $a$ = 0, iid), $e_t$ iid $N(0, I)$, $s_0 \sim N(0, I)$: every
$s_t \sim N(0, I)$, of correlation $e^{-|\ell| / \tau}$ at lag $\ell$.

**Encoder.** $z = W s + b$, from $W = I$, $b = 0$: $z \sim N(0, I)$, the
optimum of $F$. As $\mathrm{Cov}\, s = I$, the encoder's moments are exact at
every step: $\Sigma = W W^\top$, $\mu = b$.

**Buffer.** A ``jepa.nodes.PastPoints`` of $N$ = ``buffer`` (4096, decided)
rows, the $z$ of the frame before inserted once every $k$ = ``every``
frames (1), at a uniform row once full, as the head of the runs has it.

**A step** ($B$ = 1, plain SGD, $W \leftarrow W - \eta\, g s^\top$, $b
\leftarrow b - \eta\, g$) of $g$:

- ``gauss``: $\hat g_t / m$, the credit of $F / m$ (``gauss_kl_grad`` of
  the buffer's moments; off until it holds $m + 3$ rows);
- ``sigreg``: $\partial S / \partial z_t$, $S$ the Epps-Pulley statistic of
  $z_t$ and the buffer along ``directions`` unit directions drawn anew each
  step (``exp.lib.sigreg.sigreg_from_sums``; its expectation over the
  directions does not depend on their number, only its noise does: $m$ of
  them).

**Learning rate.** Each law's $\eta$ makes the time constant of the
linearized relaxation of an isotropic $W = w I$ at $w = 1$ (iid samples)
$\tau_{\mathrm{learn}}$ steps. $\partial L / \partial W_{ii} = L'(w) / m$ for
an isotropic $W$, so $w$ relaxes at the rate $\eta L''(1) / m$:

- ``gauss``: $L = F / m = \tfrac12 (w^2 - 1 - 2 \log w)$, $L''(1) = 2$,
  $\eta = m / (2 \tau_{\mathrm{learn}})$;
- ``sigreg``: $E\,S(w^2)$ of $n = N + 1$ iid samples (the closed form
  ``exp.video.diag.ssig.expected_s``) of a stationary encoder, whose $w$
  moves all $n$ samples; the step's gradient reaches the current sample
  only, $1 / n$ of it (the samples are exchangeable), so $L(w) = E\,S(w^2)
  / n$, $L''(1) = (2 S'(1) + 4 S''(1)) / n$, $\eta = m / (L''(1)
  \tau_{\mathrm{learn}})$. (The moment matching's step needs no such
  factor: $\hat g_t$ is an estimate of the whole $\nabla F$.)

$\tau_{\mathrm{learn}}$ = ``tau_learn`` = $10^5$ steps, 10 times the slowest
$\tau$ ($10^4$) and 24 times the buffer's mean age $N k$: the encoder is
stationary on the times of the stream and of the buffer (the premise of the
expected gradient of ``exp.lib.gauss_kl``).

**Measured** over the second half of the steps: the time mean of
$\mathrm{tr}\, \Sigma / m - 1$ (the mean of the eigenvalues of $\Sigma -
I$), its sd over time (the spread), the time means of the smallest and the
largest eigenvalue of $\Sigma$ (every ``EIG_EVERY`` steps), and of $\lVert
\mu \rVert$. For a relaxation of time constant $\tau_{\mathrm{learn}}$ the
time mean over $T$ steps has a standard error of about $\mathrm{sd}
\sqrt{2 \tau_{\mathrm{learn}} / T}$ (``se``).

**Frozen** (``gauss`` only): $W = I$ held, $K = E[\hat P (s_t -
\hat\mu) s_t^\top]$ over the steps with the buffer full. The expected
gradient of an isotropic $W = \sqrt c\, I$ is $\sqrt c\,(I - K / c)$ ($\hat
P$ scales as $1 / c$, the rest as $\sqrt c$), so it vanishes at $c^* =
\mathrm{tr}\, K / m$: the fixed point of the expected dynamics, without a
learning rate. $c^*$ = 1 for iid rows (``exp.lib.gauss_kl``).

**Predictions.** The reading of 2026-11-13: for ``gauss`` the mean of
$\Sigma - I$ does not depend on $\tau$, only its spread does; for SIGReg it
does. The derivation of ``exp.lib.gauss_kl`` expects the log-determinant's
part to move with the correlation: $S$ of correlated rows underestimates
$\Sigma$ and $s_t - \hat\mu$ is smaller than $s_t$ along a slow stream, so
$\hat P (s_t - \hat\mu)$ is biased.

Run: ``python -m exp.video.diag.gkl_ar1 run --law gauss --tau 100 --out
<json>`` (one law and $\tau$ per process; ``--frozen`` for $c^*$), then
``python -m exp.video.diag.gkl_ar1 report <json> ...`` (a Markdown table).
"""

import argparse
import json
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from exp.lib.gauss_kl import gauss_kl_grad
from exp.lib.sigreg import ecf_sums, sigreg_from_sums
from exp.video.diag.ssig import expected_s
from exp.video.jepa.nodes import PastPoints

__all__ = [
    "EIG_EVERY",
    "LAWS",
    "ar1_coefficients",
    "eta_of",
    "main",
    "report",
    "run",
    "sigreg_curvature",
]

LAWS = ("gauss", "sigreg")
EIG_EVERY = 100  # steps between two eigendecompositions of Sigma


def ar1_coefficients(tau: float) -> tuple[float, float]:
    r"""$a = e^{-1/\tau}$ and $\sqrt{1 - a^2}$ of the stream (0 and 1 for
    $\tau$ = 0, iid; module docstring)."""
    a = math.exp(-1 / tau) if tau > 0 else 0.0
    return a, math.sqrt(1 - a * a)


def sigreg_curvature(n: int) -> float:
    r"""$L''(1)$ of $L(w) = E\,S(w^2) / n$, $n$ iid samples (module
    docstring)."""
    w = torch.ones((), dtype=torch.float64, requires_grad=True)
    loss = expected_s(w**2, n) / n
    (d1,) = torch.autograd.grad(loss, [w], create_graph=True)
    (d2,) = torch.autograd.grad(d1, [w])
    return float(d2)


def eta_of(law: str, *, dim: int, buffer: int, tau_learn: float) -> float:
    r"""The learning rate of ``law`` of the time constant ``tau_learn``
    (module docstring)."""
    if law == "gauss":
        return dim / (2 * tau_learn)
    if law == "sigreg":
        return dim / (sigreg_curvature(buffer + 1) * tau_learn)
    raise ValueError(f"law must be one of {LAWS}, not {law!r}")


def _grad(
    law: str, z: torch.Tensor, points: PastPoints, gen: torch.Generator
) -> torch.Tensor:
    """$g$ of a step at ``z`` ``(m,)`` with the buffer ``points`` (module
    docstring)."""
    m = z.shape[0]
    if law == "gauss":
        return gauss_kl_grad(z[None], points.moments)[0] / m
    # in float32, as the head of the runs computes it
    a = torch.randn(m, m, generator=gen)
    dirs = a / a.norm(dim=1, keepdim=True)
    sums = ecf_sums(points.filled.float(), dirs)
    leaf = z.detach().float().requires_grad_()
    s = sigreg_from_sums(leaf, sums, points.count + 1, dirs)
    (g,) = torch.autograd.grad(s, [leaf])
    return g.to(z.dtype)


def run(
    law: str,
    tau: float,
    *,
    dim: int = 16,
    buffer: int = 4096,
    every: int = 1,
    steps: int = 500_000,
    tau_learn: float = 1e5,
    seed: int = 0,
    frozen: bool = False,
) -> dict[str, Any]:
    """One stream of correlation time ``tau`` (0: iid) learned by ``law``
    (or, ``frozen``, the expected gradient at $W = I$; ``gauss`` only), and
    what the module docstring measures."""
    if frozen and law != "gauss":
        raise ValueError("frozen measures the moment matching's c* only")
    gen = torch.Generator().manual_seed(seed)
    f64 = torch.float64
    a, noise = ar1_coefficients(tau)
    eta = eta_of(law, dim=dim, buffer=buffer, tau_learn=tau_learn)
    points = PastPoints(
        capacity=buffer,
        every=every,
        streams=1,
        width=dim,
        dtype=f64,
        directions=0,
        device="cpu",
        moments=law == "gauss",
    )
    one = torch.ones(1, dtype=torch.bool)
    w = torch.eye(dim, dtype=f64)
    b = torch.zeros(dim, dtype=f64)
    s = torch.randn(dim, generator=gen, dtype=f64)
    half = steps // 2
    dev, eig_min, eig_max, mu = [], [], [], []
    k_sum, k_n = torch.zeros(dim, dim, dtype=f64), 0
    started = time.perf_counter()
    for t in range(steps):
        points.next_frame(None, gen)
        z = w @ s + b
        if frozen:
            if points.count == len(points.rows):
                m = points.moments
                k_sum += torch.outer(m.precision @ (s - m.mean), s)
                k_n += 1
        else:
            g = _grad(law, z, points, gen)
            w -= eta * torch.outer(g, s)
            b -= eta * g
        points.keep(z[None], one)
        if t >= half and not frozen:
            sigma = w @ w.T
            dev.append(float(sigma.trace()) / dim - 1)
            if t % EIG_EVERY == 0:
                e = torch.linalg.eigvalsh(sigma)
                eig_min.append(float(e[0]))
                eig_max.append(float(e[-1]))
                mu.append(float(b.norm()))
        s = a * s + noise * torch.randn(dim, generator=gen, dtype=f64)
    out: dict[str, Any] = {
        "law": law,
        "tau": tau,
        "dim": dim,
        "buffer": buffer,
        "every": every,
        "steps": steps,
        "tau_learn": tau_learn,
        "eta": eta,
        "seed": seed,
        "frozen": frozen,
        "seconds": time.perf_counter() - started,
    }
    if frozen:
        k = k_sum / max(k_n, 1)
        out |= {
            "c_star": float(k.trace()) / dim,
            "k_diag_sd": float(k.diagonal().std()),
            "k_offdiag_rms": float(
                (k - torch.diag(k.diagonal())).pow(2).sum().sqrt() / dim
            ),
            "k_steps": k_n,
        }
        return out
    d = torch.tensor(dev, dtype=f64)
    sd = float(d.std())
    out |= {
        "mean_dev": float(d.mean()),
        "sd_dev": sd,
        "se": sd * math.sqrt(2 * tau_learn / len(dev)),
        "eig_min": sum(eig_min) / len(eig_min),
        "eig_max": sum(eig_max) / len(eig_max),
        "mu_norm": sum(mu) / len(mu),
    }
    return out


def report(results: Sequence[dict[str, Any]]) -> str:
    """The Markdown table of ``run``'s results, by law, then ``tau``."""
    lines = [
        "| law | tau | mean(tr Sigma / m - 1) | se | sd over time | "
        "eig min | eig max | norm mu | c* (frozen) |",
        "| -- | --: | --: | --: | --: | --: | --: | --: | --: |",
    ]
    trained = [r for r in results if not r["frozen"]]
    frozen = {r["tau"]: r for r in results if r["frozen"]}
    for r in sorted(trained, key=lambda r: (r["law"], r["tau"])):
        c = frozen.get(r["tau"]) if r["law"] == "gauss" else None
        lines.append(
            f"| {r['law']} | {r['tau']:g} | {r['mean_dev']:+.4f} | "
            f"{r['se']:.4f} | {r['sd_dev']:.4f} | {r['eig_min']:.3f} | "
            f"{r['eig_max']:.3f} | {r['mu_norm']:.3f} | "
            f"{'-' if c is None else format(c['c_star'], '.4f')} |"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="AR(1) stream (diagnosis).")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--law", choices=LAWS, required=True)
    r.add_argument("--tau", type=float, required=True)
    r.add_argument("--dim", type=int, default=16)
    r.add_argument("--buffer", type=int, default=4096)
    r.add_argument("--every", type=int, default=1)
    r.add_argument("--steps", type=int, default=500_000)
    r.add_argument("--tau-learn", type=float, default=1e5)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--frozen", action="store_true")
    r.add_argument("--out", type=Path, required=True)
    q = sub.add_parser("report")
    q.add_argument("files", type=Path, nargs="+")
    args = p.parse_args(argv)
    if args.cmd == "run":
        torch.set_num_threads(1)
        out = run(
            args.law,
            args.tau,
            dim=args.dim,
            buffer=args.buffer,
            every=args.every,
            steps=args.steps,
            tau_learn=args.tau_learn,
            seed=args.seed,
            frozen=args.frozen,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=1) + "\n")
        print(json.dumps(out, indent=1))
    else:
        print(report([json.loads(f.read_text()) for f in args.files]))


if __name__ == "__main__":
    main()
