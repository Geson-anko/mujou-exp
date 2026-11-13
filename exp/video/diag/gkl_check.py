r"""DIAGNOSIS ONLY (M17, 2026-11-13): three checks of the Gaussian moment
matching's estimate (``exp.lib.gauss_kl``) that the results report.

- ``nongauss``: $\mathrm{tr}\, E \hat P / D$ for iid rows of covariance $I$
  of three laws, $\hat P = (n - D - 2) S^{-1}$ of $n$ rows. It is 1 for
  Gaussian rows (the Wishart identity the estimate is built on); for a
  non-Gaussian law $E S^{-1}$ depends on the fourth moments and it is not.
  The laws: ``gauss`` $N(0, I)$; ``student`` the elliptical Student t of
  $\nu$ = 5 degrees of freedom scaled to covariance $I$ (one radial
  factor shared by the coordinates of a row: $x = \sqrt{(\nu - 2) / \nu}\,
  e / \sqrt{w / \nu}$, $e \sim N(0, I)$, $w \sim \chi^2_\nu$); ``laplace``
  independent Laplace coordinates of variance 1 (scale $1 / \sqrt 2$, the
  difference of two exponentials). The reported spread is the standard
  error of the mean over the ``reps`` buffers.
- ``floor``: the bias of the closed form of the sample moments,
  $E[\mathrm{KL}(N(\hat\mu, S / (n - 1)) \,\|\, N(0, I))] - F$, per
  coordinate, at $\Sigma = I$ (``exp.lib.gauss_kl.plugin_bias``): what the
  plug-in would read at the optimum, where the logged $\tilde F$ reads 0 on
  average.
- ``bench``: the milliseconds of ``buffer_moments``' parts on ``n`` rows of
  ``D`` (float32 rows, the float64 scatter, its Cholesky factor, the
  ``cholesky_inverse``) and of the whole call, the median of ``repeats``
  after ``warmup`` (synchronized on CUDA): the cost of an insertion, which
  the runs pay once every $k$ frames.

Run: ``python -m exp.video.diag.gkl_check nongauss --dim 64 --rows 256
--reps 1500``, ``... floor --dim 1024 --rows 4096``, ``... bench --dim 1024
--rows 4096 --device cpu`` (``cuda`` on a GPU host).
"""

import argparse
import json
import math
import statistics
import time
from collections.abc import Callable, Sequence
from typing import Any

import torch

from exp.lib.gauss_kl import buffer_moments, plugin_bias

__all__ = ["LAWS", "STUDENT_DOF", "bench", "floor", "main", "nongauss", "sample"]

LAWS = ("gauss", "student", "laplace")
STUDENT_DOF = 5  # nu of the elliptical Student t (finite fourth moments: nu > 4)


def sample(law: str, n: int, d: int, gen: torch.Generator) -> torch.Tensor:
    """``n`` iid rows ``(n, d)`` of ``law`` (module docstring), of mean 0 and
    covariance $I$, float64."""
    f64 = torch.float64
    if law == "gauss":
        return torch.randn(n, d, generator=gen, dtype=f64)
    if law == "student":
        nu = STUDENT_DOF
        e = torch.randn(n, d, generator=gen, dtype=f64)
        w = torch.randn(n, nu, generator=gen, dtype=f64).pow(2).sum(1)
        return math.sqrt((nu - 2) / nu) * e / (w / nu).sqrt()[:, None]
    if law == "laplace":
        u = torch.rand(2, n, d, generator=gen, dtype=f64)
        return (u[1].log() - u[0].log()) / math.sqrt(2)
    raise ValueError(f"law must be one of {LAWS}, not {law!r}")


def nongauss(law: str, *, dim: int, rows: int, reps: int, seed: int) -> dict[str, Any]:
    r"""$\mathrm{tr}\, E \hat P / D$ of ``reps`` buffers of ``rows`` iid rows
    of ``law`` and its standard error (module docstring)."""
    gen = torch.Generator().manual_seed(seed)
    tr = torch.tensor(
        [
            float(buffer_moments(sample(law, rows, dim, gen)).precision.trace()) / dim
            for _ in range(reps)
        ],
        dtype=torch.float64,
    )
    return {
        "law": law,
        "dim": dim,
        "rows": rows,
        "reps": reps,
        "seed": seed,
        "trace_ratio": float(tr.mean()),
        "se": float(tr.std()) / math.sqrt(reps),
        "plugin_ratio": (rows - 1) / (rows - dim - 2),
    }


def floor(*, dim: int, rows: int) -> dict[str, Any]:
    r"""The plug-in's bias per coordinate at $\Sigma = I$ (module
    docstring)."""
    return {
        "dim": dim,
        "rows": rows,
        "plugin_bias_per_coordinate": plugin_bias(rows, dim, float(dim)) / dim,
    }


def _timed(
    fn: Callable[[], Any], device: torch.device, warmup: int, repeats: int
) -> float:
    """The median milliseconds of ``fn``."""
    out = []
    for k in range(warmup + repeats):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if k >= warmup:
            out.append(1e3 * (time.perf_counter() - t))
    return statistics.median(out)


def bench(
    *, dim: int, rows: int, device: str, warmup: int, repeats: int
) -> dict[str, Any]:
    """The milliseconds of ``buffer_moments`` and of its parts (module
    docstring)."""
    dev = torch.device(device)
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(rows, dim, generator=gen).to(dev)
    xd = x.double()
    xc = xd - xd.mean(0)
    scatter = xc.T @ xc
    factor = torch.linalg.cholesky_ex(scatter)[0]
    parts = {
        "scatter_ms": lambda: (lambda c: c.T @ c)(x.double() - x.double().mean(0)),
        "cholesky_ms": lambda: torch.linalg.cholesky_ex(scatter),
        "inverse_ms": lambda: torch.cholesky_inverse(factor),
        "total_ms": lambda: buffer_moments(x),
    }
    out: dict[str, Any] = {"dim": dim, "rows": rows, "device": device}
    for name, fn in parts.items():
        out[name] = _timed(fn, dev, warmup, repeats)
    return out


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Checks of the moment matching.")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("nongauss")
    a.add_argument("--dim", type=int, required=True)
    a.add_argument("--rows", type=int, required=True)
    a.add_argument("--reps", type=int, required=True)
    a.add_argument("--seed", type=int, default=0)
    f = sub.add_parser("floor")
    f.add_argument("--dim", type=int, required=True)
    f.add_argument("--rows", type=int, required=True)
    b = sub.add_parser("bench")
    b.add_argument("--dim", type=int, required=True)
    b.add_argument("--rows", type=int, required=True)
    b.add_argument("--device", default="cpu")
    b.add_argument("--warmup", type=int, default=3)
    b.add_argument("--repeats", type=int, default=20)
    args = p.parse_args(argv)
    out: Any
    if args.cmd == "nongauss":
        out = [
            nongauss(law, dim=args.dim, rows=args.rows, reps=args.reps, seed=args.seed)
            for law in LAWS
        ]
    elif args.cmd == "floor":
        out = floor(dim=args.dim, rows=args.rows)
    else:
        out = bench(
            dim=args.dim,
            rows=args.rows,
            device=args.device,
            warmup=args.warmup,
            repeats=args.repeats,
        )
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
