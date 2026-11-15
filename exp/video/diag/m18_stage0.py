r"""The measurements of M18's stage 0 (plans/m18-vae-stream.md, 「段 0」): the
yardsticks of stage 1 in closed form and the values its learning settings are
derived from, before any run.

``pca``: of the training stacks $y = x / 256$ (``vae.measures``: the
``iter_stream_windows`` of M17's PCA, $n$ = 2048 windows x 9 stacks), the
mean stack $m$ and $\sigma_0^2$ (the mean over the $P$ values of their
variance), the eigenvalues $\lambda_j$ of their covariance $\frac1n \sum_i
(y_i - m)(y_i - m)^\top$ from the $n \times n$ Gram matrix of the centred
stacks (float64, by blocks of stacks on the device: $P$ = 786432 is too many
for the $P \times P$ covariance), the number of them above $\sigma_0^2$, the
top-$k$ principal basis $U = Y_c^\top V_k \Lambda_k^{-1/2}$ ($k$ = $D$ =
1024, the linear optimum of the autoencoder's loss at the code's size:
Baldi & Hornik 1989), and the ``stack_mse`` of the PCA-$k$
reconstruction $m + U U^\top (y - m)$ on the held-out stacks of val and test
(``measures.held_out_stacks``) and on the overfit control's pool (the stacks
of ``PooledStackedTourFrames`` of the run's seed, ``program.ae_streams``):
$\sum \lVert r \rVert^2 / \sum \lVert y - m \rVert^2$ with the residual
$r = (y - m) - U U^\top (y - m)$ computed directly. It also keeps, per split,
the mean square of each coordinate $u_j^\top (y - m)$ and of the residual (the
held-out sums of the pPCA likelihood of stage 2).

``hessian``: the spectral radius $\rho$ (the largest $|\lambda|$) of the
Hessian of the loss of the untrained autoencoder (``modules.build_ae`` of the
seed) with respect to all its parameters, by power iteration on
Hessian-vector products (double backward, ``lambda_max``): of $\ell_{AE}$ of
the first stack of each of ``--windows`` training windows (``WINDOWS``, the
windows of the moments and the PCA above, spread evenly over the loop of the
nine videos; a step of $B$ = 1 sees one stack), with the quantiles of $\rho$
over them, and of the mean $\ell_{AE}$ of ``--batch`` of them, every
``windows // batch``-th (the first stacks of ``data.streams`` = ``batch``
streams, which start at the loop's ``batch`` equal divisions; a step of
$B$ = ``batch`` sees their mean). $\rho$ bounds every eigenvalue, and the
batch's $\rho$ is at most the mean of its stacks' (the triangle inequality of
the spectral norm): SGD's rate is derived from the largest $\rho$ of the
stacks (plans/m18-vae-stream.md).

Run as ``python -m exp.video.diag.m18_stage0 pca --out F +experiment=m18/ae_t128
seed=100 device=cuda`` (and ``hessian``), with ``$MUJOU_EXP_DATA`` the data's
parent.
"""

import argparse
import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from exp.lib.png import tile, write_png
from exp.video.compose import compose
from exp.video.data.frames import PooledStackedTourFrames
from exp.video.measure.encoder_eval import WINDOWS, iter_stream_windows
from exp.video.programs.vision import tours_root
from exp.video.schema.run import RunConf
from exp.video.vae.loss import LEVELS, ae_loss, target
from exp.video.vae.measures import held_out_stacks, training_moments
from exp.video.vae.modules import build_ae
from exp.video.vae.program import ae_streams

__all__ = [
    "gram",
    "lambda_max",
    "main",
    "pca_basis",
    "projection_sums",
]

POWER_SEED = 0  # the start of the power iteration
TOL = 1e-4  # the relative error lambda_max leaves in rho, far below the 2 of 2 / rho
STREAMS_B32 = 32  # the streams of stage 1's B = 32 runs


def _centred(
    fetch: Callable[[int, int], Tensor],
    a: int,
    b: int,
    m: Tensor,
    scale: float,
) -> Tensor:
    """Rows ``a .. b - 1`` of ``fetch`` flattened, times ``scale``, less ``m``
    (float64 on the device of ``m``)."""
    return fetch(a, b).to(m.device).double().flatten(1) * scale - m


def gram(
    fetch: Callable[[int, int], Tensor],
    n: int,
    mean: Tensor,
    *,
    block: int,
    device: torch.device,
    scale: float = 1.0,
) -> Tensor:
    r"""$G = Y_c Y_c^\top$ ``(n, n)`` float64 on ``device`` of the centred
    rows $y_i - m$: $y_i$ ``scale`` times row $i$ of ``fetch(a, b)`` (the
    rows $a .. b - 1$ ``(b - a, ...)``, any dtype; u8 stacks with ``scale``
    1 / 256), ``mean`` $m$ of a row's shape: by blocks of ``block`` rows,
    the lower blocks computed and mirrored."""
    m = mean.double().to(device).flatten()
    g = torch.zeros(n, n, dtype=torch.float64, device=device)
    starts = list(range(0, n, block))
    for i, a in enumerate(starts):
        b = min(a + block, n)
        yi = _centred(fetch, a, b, m, scale)
        for c in starts[: i + 1]:
            d = min(c + block, n)
            yj = yi if c == a else _centred(fetch, c, d, m, scale)
            blk = yi @ yj.T
            g[a:b, c:d] = blk
            if c != a:
                g[c:d, a:b] = blk.T
            del yj
        del yi
    return g


def pca_basis(
    fetch: Callable[[int, int], Tensor],
    n: int,
    mean: Tensor,
    k: int,
    *,
    block: int,
    device: torch.device,
    scale: float = 1.0,
) -> tuple[Tensor, Tensor]:
    r"""The eigenvalues ``(n,)`` (descending, float64, on the CPU) of the
    covariance $\frac1n Y_c^\top Y_c$ of the rows of ``fetch`` (``gram``),
    and the top-``k`` principal basis $U = Y_c^\top V_k \Lambda_k^{-1/2}$
    ``(P, k)`` float64 on ``device`` ($\Lambda_k$ the top eigenvalues of
    $G$, $V_k$ their eigenvectors): orthonormal columns, the top principal
    directions."""
    g = gram(fetch, n, mean, block=block, device=device, scale=scale)
    evals, evecs = torch.linalg.eigh(g)  # ascending
    del g
    evals, evecs = evals.flip(0), evecs.flip(1)
    vk = evecs[:, :k] / evals[:k].clamp(min=0).sqrt()  # (n, k)
    del evecs
    if device.type == "cuda":  # the Gram's blocks, cached, would not leave room for U
        torch.cuda.empty_cache()
    m = mean.double().to(device).flatten()
    u = torch.zeros(m.numel(), k, dtype=torch.float64, device=device)
    for a in range(0, n, block):
        b = min(a + block, n)
        u += _centred(fetch, a, b, m, scale).T @ vk[a:b]
    return (evals / n).cpu(), u


def projection_sums(
    u: Tensor, rows: Tensor, mean: Tensor, *, batch: int = 32
) -> dict[str, Any]:
    r"""Of the rows ``rows`` ``(r, ...)`` (in the units of ``mean``) and the
    orthonormal basis ``u`` ``(P, k)``: ``stack_mse`` $= \sum \lVert r_i
    \rVert^2 / \sum \lVert y_i - m \rVert^2$ with $r_i = c_i - U U^\top
    c_i$, $c_i = y_i - m$; ``coord_ms`` the mean over the rows of each
    $(u_j^\top c_i)^2$ ``(k,)``; ``residual_ms`` the mean of $\lVert r_i
    \rVert^2$; ``base_ms`` the mean of $\lVert c_i \rVert^2$."""
    dev = u.device
    m = mean.double().to(dev).flatten()
    err = base = 0.0
    coords = torch.zeros(u.shape[1], dtype=torch.float64, device=dev)
    for chunk in rows.split(batch):
        c = chunk.to(dev).double().flatten(1) - m  # (b, P)
        z = c @ u  # (b, k)
        r = c - z @ u.T
        err += float(r.pow(2).sum())
        base += float(c.pow(2).sum())
        coords += z.pow(2).sum(0)
    n = len(rows)
    return {
        "stack_mse": err / base,
        "coord_ms": (coords / n).tolist(),
        "residual_ms": err / n,
        "base_ms": base / n,
        "count": n,
    }


def lambda_max(
    loss: Callable[[], Tensor],
    params: Sequence[Tensor],
    *,
    iters: int = 200,
    tol: float = 1e-4,
    gen: torch.Generator,
) -> dict[str, Any]:
    r"""The spectral radius $\rho$ of the Hessian $H$ of ``loss()`` with
    respect to ``params``, by power iteration from a normal direction of
    ``gen``: $v \leftarrow Hv / \lVert Hv \rVert$, with $Hv$ by double
    backward of one graph of the gradient.

    For a unit $v$, $\lVert Hv \rVert \le \rho$, and it never decreases from
    one iteration to the next ($\lVert Hv \rVert^2 = v^\top H^2 v \le
    \lVert H^2 v \rVert$ by Cauchy-Schwarz): it is the power iteration of
    $H^2$, and converges to $\rho$ by the factor $q = (\lambda_2 / \rho)^2$
    an iteration ($\lambda_2$ the largest other $|\lambda|$) even when
    $\rho$ and $-\rho$ are both eigenvalues, where the Rayleigh quotient
    $v^\top H v$ need not converge. It stops when the rest of the geometric
    tail, $\delta_k q_k / (1 - q_k)$ with $\delta_k$ the last change of
    $\lVert Hv \rVert$ and $q_k = \delta_k / \delta_{k-1}$, is below ``tol``
    of it.

    Returns ``rho`` (the last $\lVert Hv \rVert$, a lower bound of $\rho$),
    ``lam`` (the last $v^\top H v$: the dominant eigenvalue with its sign
    when one sign dominates), ``iters``, ``converged`` and ``history`` (of
    $\lVert Hv \rVert$)."""
    grads = torch.autograd.grad(loss(), params, create_graph=True)
    v = [
        torch.randn(p.shape, generator=gen, dtype=p.dtype).to(p.device) for p in params
    ]
    norm = math.sqrt(sum(float(x.pow(2).sum()) for x in v))
    v = [x / norm for x in v]
    history: list[float] = []
    lam = math.nan
    converged = False
    for _ in range(iters):
        dot = torch.stack([(g * x).sum() for g, x in zip(grads, v, strict=True)]).sum()
        hv = torch.autograd.grad(dot, params, retain_graph=True)
        lam = float(sum(float((h * x).sum()) for h, x in zip(hv, v, strict=True)))
        norm = math.sqrt(sum(float(h.pow(2).sum()) for h in hv))
        history.append(norm)
        v = [h.detach() / norm for h in hv]
        if len(history) >= 3:
            d1 = abs(history[-1] - history[-2])
            d0 = abs(history[-2] - history[-3])
            q = d1 / d0 if d0 > 0 else 0.0
            if d1 == 0 or (q < 1 and d1 * q / (1 - q) < tol * norm):
                converged = True
                break
    return {
        "rho": history[-1],
        "lam": lam,
        "iters": len(history),
        "converged": converged,
        "history": history,
    }


def _stack_index(conf: RunConf, root: Path, windows: int) -> tuple[Tensor, Tensor]:
    """The training windows u8 ``(w, N + STREAM_STEPS, C, H, W)`` and the
    (window, first frame) of each of their stacks ``(n, 2)``."""
    v = conf.vision
    assert v is not None
    w8 = torch.stack(list(iter_stream_windows(conf, root, windows=windows)))
    k = w8.shape[1] - v.frames + 1
    w = torch.arange(len(w8)).repeat_interleave(k)
    return w8, torch.stack([w, torch.arange(k).repeat(len(w8))], 1)


def _fetcher(
    windows: Tensor, index: Tensor, frames: int
) -> Callable[[int, int], Tensor]:
    span = torch.arange(frames)

    def fetch(a: int, b: int) -> Tensor:  # u8 (b - a, N, C, H, W)
        w, k = index[a:b, 0], index[a:b, 1]
        return windows[w[:, None], k[:, None] + span]

    return fetch


def run_pca(
    conf: RunConf, out: Path, *, k: int, block: int, windows: int = WINDOWS
) -> dict[str, Any]:
    """The ``pca`` measurement (module docstring) into ``out`` (a JSON file;
    the mean stack's first and last frames beside it as a PNG), of the stacks
    of ``windows`` training windows."""
    v, t = conf.vision, conf.data.tours
    assert v is not None and t is not None
    dev = torch.device(conf.device)
    root = tours_root(t)
    t0 = time.monotonic()
    moments = training_moments(conf, root, windows=windows)
    w8, index = _stack_index(conf, root, windows)
    n = len(index)
    assert n == moments.count, (n, moments.count)
    fetch = _fetcher(w8, index, v.frames)
    evals, u = pca_basis(
        fetch, n, moments.mean, k, block=block, device=dev, scale=1 / LEVELS
    )
    ortho = float((u.T @ u - torch.eye(k, dtype=u.dtype, device=dev)).abs().max())
    sets = dict(held_out_stacks(conf, root))
    pooled = ae_streams(
        replace(conf, data=replace(conf.data, tours=replace(t, sampling="pool")))
    )
    assert isinstance(pooled, PooledStackedTourFrames)
    sets["pool"] = pooled.stacks
    splits = {s: projection_sums(u, target(x), moments.mean) for s, x in sets.items()}
    sigma0 = moments.var
    result = {
        "n": n,
        "P": moments.mean.numel(),
        "k": k,
        "sigma0_sq": sigma0,
        "mean_pixel": float(moments.mean.mean()),
        "mean_rgb": moments.mean.mean((0, 2, 3)).tolist(),
        "trace_over_P": float(evals.sum()) / moments.mean.numel(),
        "above_sigma0_sq": int((evals > sigma0).sum()),
        "top_k_share": float(evals[:k].sum() / evals.sum()),
        "orthonormal_max_dev": ortho,
        "eigenvalues": evals.tolist(),
        "splits": splits,
        "seconds": time.monotonic() - t0,
        "levels": LEVELS,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result) + "\n")
    mean = moments.mean.float()
    write_png(out.with_suffix(".png"), tile(mean[[0, -1]][None].clamp(0, 1), scale=2))
    return result


def run_hessian(
    conf: RunConf, out: Path, *, windows: int, batch: int, iters: int
) -> dict[str, Any]:
    """The ``hessian`` measurement (module docstring) into ``out``: a row of
    each stack (no history), the quantiles of their $\\rho$, and the batch's
    measurement with its history."""
    t, v = conf.data.tours, conf.vision
    assert t is not None and v is not None
    if windows % batch:
        raise ValueError(f"the windows {windows} are not a multiple of {batch}")
    dev = torch.device(conf.device)
    root = tours_root(t)
    mods = build_ae(conf, seed=conf.seed)
    params = mods.parameters()
    stacks = [
        w[: v.frames] for w in iter_stream_windows(conf, root, windows=windows, steps=0)
    ]
    if len(stacks) != windows:  # a window across two videos is skipped
        raise ValueError(f"{windows - len(stacks)} windows cross two videos")
    gen = torch.Generator().manual_seed(POWER_SEED)

    def measured(xs: Tensor) -> dict[str, Any]:
        x = xs.to(dev)

        def loss() -> Tensor:
            return ae_loss(target(x), mods(x)).mean()

        r = lambda_max(loss, params, iters=iters, tol=TOL, gen=gen)
        with torch.no_grad():
            return {"loss": float(loss())} | r

    t0 = time.monotonic()
    rows = []
    for i, s in enumerate(stacks):
        r = measured(s[None])
        del r["history"]
        rows.append({"stack": i} | r)
    members = list(range(0, windows, windows // batch))
    together = measured(torch.stack([stacks[i] for i in members]))
    rho = torch.tensor([r["rho"] for r in rows], dtype=torch.float64)
    result = {
        "seed": conf.seed,
        "power_seed": POWER_SEED,
        "tol": TOL,
        "stacks": rows,
        "rho_quantiles": {
            str(q): float(rho.quantile(q)) for q in (0.5, 0.9, 0.99, 0.999)
        },
        "max_stack_rho": float(rho.max()),
        "all_converged": all(r["converged"] for r in rows),
        "batch": together | {"members": members},
        "batch_members_mean_rho": float(rho[members].mean()),
        "params": sum(p.numel() for p in params),
        "seconds": time.monotonic() - t0,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result) + "\n")
    return result


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m exp.video.diag.m18_stage0")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pca", help="the moments, the spectrum and the PCA yardsticks")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--k", type=int, default=1024)
    p.add_argument("--block", type=int, default=1024)
    p.add_argument("overrides", nargs="*")
    h = sub.add_parser("hessian", help="lambda_max of the untrained loss")
    h.add_argument("--out", type=Path, required=True)
    h.add_argument("--windows", type=int, default=WINDOWS)
    h.add_argument("--batch", type=int, default=STREAMS_B32)
    h.add_argument("--iters", type=int, default=200)
    h.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    conf = compose(args.overrides)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.cmd == "pca":
        r = run_pca(conf, args.out, k=args.k, block=args.block)
        shown = {k: x for k, x in r.items() if k not in ("eigenvalues", "splits")}
        shown |= {s: d["stack_mse"] for s, d in r["splits"].items()}
    else:
        r = run_hessian(
            conf, args.out, windows=args.windows, batch=args.batch, iters=args.iters
        )
        shown = {k: x for k, x in r.items() if k != "stacks"}
        shown["batch"] = {k: x for k, x in r["batch"].items() if k != "history"}
    print(json.dumps(shown, indent=1))


if __name__ == "__main__":
    main()
