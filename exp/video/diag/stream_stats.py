r"""DIAGNOSIS ONLY (M17, 2026-11-13): what the stream offers a long-term buffer
of data points for a SIGReg of one sample per data point (a stack), measured
before such a buffer is built. Not a program of the runs, and nothing of the
runs imports it.

**Why the correlation of two data points.** SIGReg (``exp.lib.sigreg``) is
the Epps-Pulley statistic of $N$ samples projected on unit directions, against
$N(0, 1)$, raw (not centred, not scaled). For $N$ samples whose projections
are $N(0, 1)$ with correlation $r$ between any two,

$$E\,S = S_0 + f'(0)\,(N - 1)\,r + O(r^2), \quad S_0 = \sqrt{2\pi} -
\sqrt{2\pi / 3}, \quad f'(0) = \int t^2 e^{-3t^2/2}\, dt = \sqrt{2\pi} /
3^{3/2} = 0.4824,$$

since $N\,E|\hat\varphi(t) - e^{-t^2/2}|^2 = (1 - e^{-t^2}) + (N - 1)
(e^{-t^2(1 - r)} - e^{-t^2})$. The premise is a projection of unit
variance in every direction: the representation SIGReg drives to, isotropic
about 0 ($C_0 = I$, $C$ the second moments about the centre 0, here the mean
of the training set). There, averaged over the directions $a$, the
correlation of the projections $a^\top y$ is $\operatorname{tr} C_\Delta / D$
(to first order in $r$ the bias averages linearly over the directions). Of a
representation that is not yet isotropic, the same quantity is the whitened
mean $\operatorname{tr}(C_0^{-1/2} C_\Delta C_0^{-1/2}) / D$: the correlation
the stream keeps after any linear map that makes the representation
isotropic. The trace ratio $\operatorname{tr} C_\Delta / \operatorname{tr}
C_0$ is not it: it weighs each direction by its variance (of a few large
directions it is their correlation). ``corr`` measures the trace ratio of the
pixels (their whitening, of $16 \times 49152$ dimensions, is not computed):
a reference, not the premise. ``repr`` measures both of the representations,
the whitened mean as the curve ``white``.

The scale of the bias is $f'(0)(N-1)\bar r$; the criterion $\bar r \le
1/(N-1)$, a bias of at most $f'(0)$, an effective sample size $N / (1 + (N -
1)\bar r)$ of at least $N/2$, is a convention on that scale, not derived.
``crossings`` reports, for each $N$ of ``ns``, the smallest lag with
$\rho(\Delta) \le 1/(N-1)$; ``n_eff`` the effective size at each lag. The
mean correlation $\bar r$ of a buffer is over all its pairs, not that of its
nearest points: of the points on the grid filling a window of $K$ grid steps
of one stream, $\bar r(K) = \sum_{l=1}^{K-1} 2(K - l)\rho(l) / (K(K-1))$
(``window_mean``, $\rho$ linear between the measured lags), so a buffer of
$N$ points spread over that window has at most about $1/\bar r(K)$
effective samples, whatever $N$. Over the whole loop of the training videos,
centred by its own mean, the mean over all pairs is $-1/(M-1) \approx 0$: the
constraint is the length of stream a buffer spans, not the spacing.

- ``corr``: the pixels. A frame is flattened (128 x 128 x 3, its u8
  values: the ratios do not depend on the scale); only the frames of the
  grid $t = 0, s, 2s, \dots$ of each training video ($s$ = ``stride``, 3:
  the stacks' spacing and the stream's advance per learning step) enter. $m_a$ is the mean of video $a$'s grid frames, $m =
  \sum_a J_a m_a / \sum_a J_a$ ($J_a$ its grid frames), $d_a = m_a - m$, $y =
  x - m$ (``g``) and $u = x - m_a$ (``w``). A data point is a stack $Y_j =
  (y_{j-N+1}, \dots, y_j)$ of $N$ = ``frames`` consecutive grid frames
  ($j \ge N - 1$). At a lag of $k$ grid steps, over the anchors $j$ with
  $Y_j$ and $Y_{j+k}$ in the video,

  $$\rho_a(k) = \frac{\sum_j \langle Y_j, Y_{j+k} \rangle}{\sqrt{\sum_j
  \lVert Y_j \rVert^2 \sum_j \lVert Y_{j+k} \rVert^2}},$$

  numerator and denominator over the same pairs (``lag_sums``); the pooled
  $\rho$ is the same ratio of the sums over all videos. Its error is the
  jackknife over the videos (the videos are the independent units; $m$ held
  fixed): $\hat\sigma^2 = \frac{A-1}{A} \sum_a (\rho_{-a} - \bar\rho_{-})^2$,
  and the upper end $\rho + t_{A-1}\hat\sigma$ of the two-sided 95%
  interval, $t_{A-1}$ the 0.975 quantile of Student's t of $A - 1$ degrees of
  freedom (``t_quantile``). The lags are $\mathrm{round}(2^{i/3})$ grid steps
  up to the shortest video less a stack (so every video has anchors at every
  lag; the largest, $2^{15}$ steps or 54.6 min, is the largest of them at or
  below the 60.1 min of Wildlife less a stack).

  The reference of two data points of different videos (``g``): the mean over
  all pairs of a stack of video $a$ and one of $b$ of $\langle Y, Y'
  \rangle$ is $N \langle d_a, d_b \rangle$ (each slot's mean is $m_a$, up to
  the $N - 1$ stacks at the ends), so over pairs of different videos
  $N \sum_{a \ne b} n_a n_b \langle d_a, d_b \rangle / \sum_{a \ne b} n_a
  n_b$ over the pooled mean square $v$ ($n_a$ the stacks), with the same
  jackknife. ``share``: $N \lVert d_a \rVert^2$ over the video's mean square,
  the part of video $a$ that no lag within it decorrelates. Of ``w`` the
  reference is 0.

- ``repr``: the same of the representations of the stacks. Each model
  (``label=batch:run:ckpt:codes``, ``ckpt`` a checkpoint's name or ``init``)
  codes every stack $Y_j$ of the grid, by its flattened target tokens
  (``tokens``) and/or their pooled $z$ (``z``, ``VisionModules.z_of``); the
  codes are the rows, a data point is one row ($N = 1$ in the above), and the
  reference is that of one row. Each code has two curves: ``g`` (the trace
  ratio) and ``white``, the same of the codes whitened by $C_0^{-1/2}$
  (``whitener``, $C_0$ pooled over the training videos about $m$, held fixed
  in the jackknife as $m$ is): its $\rho$ is the whitened mean. Of the codes
  in ``--spectrum``, ``spectrum`` gives at each lag the least and the largest
  correlation over the directions of the whitened codes, the extreme
  eigenvalues of the symmetric part of their lag matrix (``lag_spectrum``,
  whose mean eigenvalue is the ``white`` $\rho$).

- ``budget``: from finished runs (``job.json``, ``result.json``,
  ``exit/attempt-*.json``), the learning steps per wall-clock second (one
  unit is one learning step, ``data.tours.advance`` video frames), and, for
  the lag $\Delta_N$ at which each curve crosses $1/(N-1)$, the points
  $\Delta_N$ apart one stream gives in the run and in ``--wall`` seconds, all
  $B$ streams give, and the loop of the training videos holds.

- ``drift``: the staleness of a stored representation. The same held-in
  stacks (``encoder_eval.stream_windows`` with no step after the first) are
  coded by each evaluation checkpoint of a run (the flattened target tokens
  when ``measure.posthoc_code`` is ``"tokens"``, else $z$), and for each pair
  of checkpoints $\Delta$ apart

  $$\mathrm{drift} = \frac{E_i \lVert f_t(x_i) - f_{t+\Delta}(x_i)
  \rVert^2}{E_{i \ne j} \lVert f_{t+\Delta}(x_i) - f_{t+\Delta}(x_j)
  \rVert^2}:$$

  1 when a stored code is as far from the current one as two different
  stacks are from each other.

- ``report``: the tables of the outputs as markdown.

Run: ``python -m exp.video.diag.stream_stats corr --root DATA --out JSON``,
``... repr --root DATA --runs RUNS --model LABEL=SPEC... --out JSON``,
``... budget --corr JSON... --runs DIR... --out JSON``, ``... drift --root
DATA --runs DIR... --out JSON``, ``... report --dir DIR --repr NAME... --out
MD``.
"""

import argparse
import dataclasses
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from exp.lib import ckpt
from exp.lib.video_io import U8Reader
from exp.video.compose import from_dict
from exp.video.data.walking_tours import videos
from exp.video.jepa.vit import VisionModules, build_vision
from exp.video.measure.encoder_eval import stream_windows

__all__ = [
    "budget",
    "crossing",
    "curve",
    "drift",
    "frame_mean",
    "lag_spectrum",
    "lag_sums",
    "log_lags",
    "main",
    "n_eff",
    "reference",
    "report",
    "run_speed",
    "settled",
    "stack_codes",
    "stack_products",
    "t_quantile",
    "whitener",
    "window_mean",
]

NS = (32, 256, 1024)  # the sample sizes of a SIGReg asked about


def log_lags(max_lag: int, per_octave: int = 3) -> list[int]:
    """The distinct integers $\\mathrm{round}(2^{i / \\mathrm{per\\_octave}})$
    from 1 up to ``max_lag``."""
    top = math.floor(per_octave * math.log2(max_lag) + 1e-9)
    return sorted({round(2 ** (i / per_octave)) for i in range(top + 1)})


def t_quantile(p: float, dof: int, steps: int = 20001) -> float:
    """The ``p`` quantile ($p > 1/2$) of Student's t of ``dof`` degrees of
    freedom: the $x$ with $1/2 + \\int_0^x f = p$, the density $f$
    integrated by the trapezoid rule on ``steps`` points, by bisection."""
    nu = float(dof)
    logc = (
        math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(nu * math.pi)
    )

    def cdf(x: float) -> float:
        t = torch.linspace(0, x, steps, dtype=torch.float64)
        f = torch.exp(logc - (nu + 1) / 2 * torch.log1p(t * t / nu))
        return 0.5 + float(torch.trapezoid(f, t))

    lo, hi = 0.0, 1.0
    while cdf(hi) < p:
        lo, hi = hi, 2 * hi
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if cdf(mid) < p else (lo, mid)
    return (lo + hi) / 2


@torch.no_grad()
def frame_mean(
    x: Tensor, chunk: int = 4096, *, dev: torch.device | None = None
) -> Tensor:
    """The mean ``(D,)`` float64 of the rows of ``x`` ``(L, D)``, summed a
    chunk at a time on ``dev`` (that of ``x`` by default)."""
    dev = x.device if dev is None else dev
    s = torch.zeros(x.shape[1], dtype=torch.float64, device=dev)
    for i in range(0, len(x), chunk):
        s += x[i : i + chunk].to(dev).double().sum(0)
    return s / len(x)


def stack_products(p: Tensor, frames: int) -> Tensor:
    """The sums $P_j = \\sum_{k < N} p_{j-k}$ of ``frames`` consecutive
    values of ``p`` ``(J,)``, for $j = N - 1, \\dots, J - 1$: ``(J - N + 1,)``
    float64 (the products, or squared norms, of the stacks from those of their
    frames)."""
    c = torch.cat([p.new_zeros(1, dtype=torch.float64), p.double().cumsum(0)])
    return c[frames:] - c[:-frames]


@torch.no_grad()
def lag_sums(
    x: Tensor,
    own: Tensor,
    mean: Tensor,
    lags: Sequence[int],
    *,
    frames: int,
    chunk: int = 2048,
) -> dict[str, Any]:
    """The sums of one video behind its curves (module docstring): ``x``
    ``(J, D)`` its rows on the grid (u8 or float, on any device), ``own``
    $m_a$ and ``mean`` $m$ ``(D,)``, ``lags`` in grid steps, a data point
    ``frames`` consecutive rows. For each centring ``c`` (``g`` by $m$, ``w``
    by $m_a$): ``sq_<c>`` the sum of the squared norms of the data points;
    ``num_<c>``, ``den0_<c>``, ``den1_<c>`` (lists over the lags) the sums of
    $\\langle Y_j, Y_{j+k} \\rangle$, $\\lVert Y_j \\rVert^2$ and $\\lVert
    Y_{j+k} \\rVert^2$ over the anchors; ``points`` the data points and
    ``anchors`` (list) their number at each lag."""
    dev = x.device
    own32 = own.to(dev).float()
    d = (own.double() - mean.double()).to(dev)
    d32 = d.float()
    norms, ed = [], []  # |u_j|^2 and <d, u_j> of every row
    for i in range(0, len(x), chunk):
        u = x[i : i + chunk].float() - own32
        norms.append((u * u).sum(1).double())
        ed.append((u @ d32).double())
    q_w, e = torch.cat(norms), torch.cat(ed)
    dd = float((d * d).sum())
    q = {"w": q_w, "g": q_w + 2 * e + dd}
    stacks = {c: stack_products(v, frames) for c, v in q.items()}
    rows = len(x)
    out: dict[str, Any] = {"points": rows - frames + 1, "anchors": []}
    for c in ("g", "w"):
        out[f"sq_{c}"] = float(stacks[c].sum())
        for key in ("num", "den0", "den1"):
            out[f"{key}_{c}"] = []
    for lag in lags:
        n = max(rows - lag - frames + 1, 0)
        out["anchors"].append(n)
        p_w = torch.zeros(max(rows - lag, 0), dtype=torch.float64, device=dev)
        for a in range(0, rows - lag, chunk):  # <u_j, u_{j+lag}>
            b = min(a + chunk, rows - lag)
            u0 = x[a:b].float() - own32
            p_w[a:b] = (u0 * (x[a + lag : b + lag].float() - own32)).sum(1).double()
        p = {"w": p_w, "g": p_w + e[: rows - lag] + e[lag:] + dd}
        for c in ("g", "w"):
            num = float(stack_products(p[c], frames).sum()) if n else 0.0
            out[f"num_{c}"].append(num)
            out[f"den0_{c}"].append(float(stacks[c][:n].sum()))
            out[f"den1_{c}"].append(float(stacks[c][lag : lag + n].sum()))
    return out


def curve(per_video: Sequence[Mapping[str, Any]], c: str) -> dict[str, Any]:
    """The curve of centring ``c`` from the ``lag_sums`` of each video: per
    video and pooled $\\rho$, the pooled one's jackknife error over the videos
    ``se``, and ``v`` the pooled mean square of a data point (module
    docstring)."""
    num = torch.tensor([s[f"num_{c}"] for s in per_video], dtype=torch.float64)
    d0 = torch.tensor([s[f"den0_{c}"] for s in per_video], dtype=torch.float64)
    d1 = torch.tensor([s[f"den1_{c}"] for s in per_video], dtype=torch.float64)
    a = len(per_video)
    rho = num.sum(0) / (d0.sum(0) * d1.sum(0)).sqrt()
    loo = (num.sum(0) - num) / ((d0.sum(0) - d0) * (d1.sum(0) - d1)).sqrt()
    se = ((a - 1) / a * ((loo - loo.mean(0)) ** 2).sum(0)).sqrt()
    per = num / (d0 * d1).sqrt()  # nan where a video has no anchors
    v = sum(s[f"sq_{c}"] for s in per_video) / sum(s["points"] for s in per_video)
    return {"v": v, "rho": rho.tolist(), "se": se.tolist(), "videos": per.tolist()}


def reference(
    gram: Tensor, n: Tensor, sq: Tensor, scale: float = 1.0
) -> dict[str, Any]:
    """The correlation of two data points of different videos (module
    docstring) from ``gram`` ``(A, A)`` $\\langle d_a, d_b \\rangle$, the data
    points ``n`` ``(A,)`` and the sums of their squared norms ``sq`` ``(A,)``;
    ``scale`` $N$ of a stack. The pooled ``ref`` and its jackknife error
    ``se`` over the videos, and ``share`` per video and pooled."""
    a = len(n)
    g, n, sq = gram.double(), n.double(), sq.double()

    def ref(keep: Tensor) -> float:
        w = n[:, None] * n[None, :] * keep[:, None] * keep[None, :]
        w = w * ~torch.eye(a, dtype=torch.bool)
        v = float((sq * keep).sum() / (n * keep).sum())
        return float((w * g).sum() / w.sum()) * scale / v

    every = torch.ones(a, dtype=torch.float64)
    loo = torch.tensor(
        [ref(1 - torch.eye(a, dtype=torch.float64)[i]) for i in range(a)]
    )
    se = math.sqrt((a - 1) / a * float(((loo - loo.mean()) ** 2).sum()))
    share = g.diagonal() * scale / (sq / n)
    return {
        "ref": ref(every),
        "se": se,
        "share": share.tolist(),
        "share_pooled": float((n * g.diagonal()).sum() * scale / sq.sum()),
    }


def crossing(lags: Sequence[int], rho: Sequence[float], bound: float) -> int | None:
    """The smallest lag with $\\rho \\le$ ``bound``, or None."""
    for lag, r in zip(lags, rho, strict=True):
        if r <= bound:
            return lag
    return None


def settled(lags: Sequence[int], rho: Sequence[float], bound: float) -> int | None:
    """The smallest lag from which $\\rho \\le$ ``bound`` at every larger
    measured lag, or None (``crossing`` is the first lag below it)."""
    at = None
    for lag, r in zip(lags, rho, strict=True):
        if r > bound:
            at = None
        elif at is None:
            at = lag
    return at


def n_eff(rho: float, n: int) -> float | None:
    """$N / (1 + (N - 1)\\rho)$, None where $1 + (N - 1)\\rho \\le 0$ (no $N$
    exchangeable points have that mean correlation)."""
    den = 1 + (n - 1) * rho
    return n / den if den > 0 else None


def drift(a: Tensor, b: Tensor) -> float:
    """$E_i \\lVert a_i - b_i \\rVert^2 / E_{i \\ne j} \\lVert b_i - b_j
    \\rVert^2$ over the rows of ``a`` and ``b`` ``(n, D)``."""
    a, b = a.double(), b.double()
    n = len(b)
    apart = 2 * n / (n - 1) * float(((b - b.mean(0)) ** 2).sum(1).mean())
    return float(((a - b) ** 2).sum(1).mean()) / apart


@torch.no_grad()
def whitener(
    rows: Sequence[Tensor], mean: Tensor, *, dev: torch.device, chunk: int = 4096
) -> tuple[Tensor, dict[str, Any]]:
    """The whitening $W$ ``(D, r)`` float64 on ``dev`` of the rows ``(J_a,
    D)`` of all videos about ``mean`` $m$: with $C_0 = \\sum_a \\sum_j (x_j -
    m)(x_j - m)^\\top / \\sum_a J_a = U \\Lambda U^\\top$, $W = U_r
    \\Lambda_r^{-1/2}$ over the $r$ eigenvalues above $D\\,\\epsilon\\,
    \\lambda_{\\max}$ ($\\epsilon$ of float64: the numerical rank of
    ``torch.linalg.matrix_rank``), so $W^\\top C_0 W = I_r$ and $(x - m) W$
    is $C_0^{-1/2}(x - m)$ in the eigenbasis. With it ``rank`` $r$, the
    participation ratio ``pr`` $(\\sum \\lambda)^2 / \\sum \\lambda^2$ and the
    largest eigenvalues ``eig_top`` over their sum."""
    d = rows[0].shape[1]
    m = mean.to(dev).double()
    c0 = torch.zeros(d, d, dtype=torch.float64, device=dev)
    count = 0
    for x in rows:
        for i in range(0, len(x), chunk):
            y = x[i : i + chunk].to(dev).double() - m
            c0 += y.T @ y
        count += len(x)
    lam, u = torch.linalg.eigh(c0 / count)
    lam, u = lam.flip(0), u.flip(1)  # descending
    keep = lam > d * torch.finfo(torch.float64).eps * lam[0]
    w = u[:, keep] * lam[keep].rsqrt()
    info = {
        "rank": int(keep.sum()),
        "pr": float(lam.sum() ** 2 / (lam**2).sum()),
        "eig_top": (lam[:64] / lam.sum()).tolist(),
    }
    return w, info


@torch.no_grad()
def lag_spectrum(
    rows: Sequence[Tensor], lags: Sequence[int], *, dev: torch.device
) -> list[list[float]]:
    """At each lag $k$ (in rows), the least and the largest eigenvalue of
    $M_k = r\\,\\mathrm{sym}(\\sum_a \\sum_j w_j w_{j+k}^\\top) /
    \\sqrt{\\sum \\lVert w_j \\rVert^2 \\sum \\lVert w_{j+k} \\rVert^2}$ over
    the centred whitened rows $w$ ``(J_a, r)`` of each video: the correlation
    along its eigen-directions, whose mean $\\operatorname{tr} M_k / r$ is the
    curve's $\\rho$ (``lag_sums`` of one row)."""
    r = rows[0].shape[1]
    out = []
    for k in lags:
        num = torch.zeros(r, r, dtype=torch.float64, device=dev)
        d0 = d1 = 0.0
        for x in rows:
            if len(x) <= k:
                continue
            y = x.to(dev).double()
            num += y[:-k].T @ y[k:]
            d0 += float((y[:-k] ** 2).sum())
            d1 += float((y[k:] ** 2).sum())
        e = torch.linalg.eigvalsh((num + num.T) / 2 * r / math.sqrt(d0 * d1))
        out.append([float(e[0]), float(e[-1])])
    return out


def window_mean(lags: Sequence[int], rho: Sequence[float], k: int) -> float:
    """$\\bar r(K) = \\sum_{l=1}^{K-1} 2(K - l)\\rho(l) / (K(K-1))$, the mean
    correlation over all pairs of the $K$ = ``k`` consecutive points of the
    grid, $\\rho(l)$ linear between the measured ``lags`` (``lags[0]`` = 1,
    ``lags[-1]`` $\\ge K - 1$)."""
    if lags[0] != 1 or lags[-1] < k - 1 or k < 2:
        raise ValueError(f"the lags {lags[0]}..{lags[-1]} do not cover {k - 1}")
    lag = torch.tensor(lags, dtype=torch.float64)
    r = torch.tensor(rho, dtype=torch.float64)
    ls = torch.arange(1, k, dtype=torch.float64)
    i = torch.searchsorted(lag, ls).clamp(1, len(lags) - 1)  # lag[i-1] < l <= lag[i]
    t = (ls - lag[i - 1]) / (lag[i] - lag[i - 1])
    at = r[i - 1] + t * (r[i] - r[i - 1])
    return float((2 * (k - ls) * at).sum() / (k * (k - 1)))


def run_speed(run_dir: Path) -> dict[str, Any]:
    """The speed of a finished run: its units (learning steps), the summed
    wall-clock seconds of its attempts, and its config's streams, advance and
    fps."""
    job = json.loads((run_dir / "job.json").read_text())
    steps = json.loads((run_dir / "result.json").read_text())["frames"]
    wall = 0.0
    for p in sorted((run_dir / "exit").glob("attempt-*.json")):
        e = json.loads(p.read_text())
        wall += e["ended"] - e["started"]
    data = job["config"]["data"]
    return {
        "run": run_dir.name,
        "batch": run_dir.parent.name,
        "streams": data["streams"],
        "advance": data["tours"]["advance"],
        "fps": data["tours"]["fps"],
        "steps": steps,
        "wall": wall,
        "steps_per_s": steps / wall,
    }


def budget(
    speed: Mapping[str, Any],
    tau: float,
    *,
    wall: float,
    loop: int,
    frames: int,
    pixels: int,
    dims: Sequence[int],
) -> dict[str, Any]:
    """The $\\tau$-spaced data points of a run of ``speed`` (``run_speed``),
    $\\tau$ in video seconds: per stream in the run and in ``wall`` seconds,
    all streams (``total``) and at most the loop of ``loop`` frames
    (``distinct``), with the bytes of the run's points as float32 vectors of
    each $D$ in ``dims`` and as u8 stacks of ``frames`` x ``pixels``."""
    per_step = speed["advance"] / speed["fps"]  # video seconds per step
    video_run = speed["steps"] * per_step
    video_wall = speed["steps_per_s"] * wall * per_step
    b = speed["streams"]
    per_stream = math.floor(video_run / tau)
    covered = min(b * video_run, loop / speed["fps"])
    distinct = min(b * per_stream, math.floor(covered / tau))
    return {
        "tau": tau,
        "video_per_stream": video_run,
        "video_per_stream_wall": video_wall,
        "points_per_stream": per_stream,
        "points_per_stream_wall": math.floor(video_wall / tau),
        "points_total": b * per_stream,
        "points_distinct": distinct,
        "bytes_repr": {str(dim): b * per_stream * dim * 4 for dim in dims},
        "bytes_raw": b * per_stream * frames * pixels,
    }


@torch.no_grad()
def stack_codes(
    mods: VisionModules, grid: Tensor, frames: int, codes: Sequence[str], batch: int
) -> dict[str, Tensor]:
    """The codes ``(J - N + 1, D)`` float32 on the CPU of the stacks of
    ``frames`` consecutive frames of ``grid`` ``(J, C, H, W)`` (row $i$: the
    stack of frames $i, \\dots, i + N - 1$), for each of ``codes``:
    ``tokens`` the flattened target tokens, ``z`` their ``z_of``."""
    n = len(grid) - frames + 1
    out: dict[str, list[Tensor]] = {c: [] for c in codes}
    span = torch.arange(frames, device=grid.device)
    for i in range(0, n, batch):
        idx = torch.arange(i, min(i + batch, n), device=grid.device)[:, None] + span
        tok = mods.tokens(grid[idx])
        for c in codes:
            code = tok.flatten(1) if c == "tokens" else mods.z_of(tok)
            out[c].append(code.float().cpu())
    return {c: torch.cat(v) for c, v in out.items()}


# -- the commands -----------------------------------------------------------------


def _grid(
    reader: U8Reader, stride: int, dev: torch.device, chunk: int = 4096
) -> Tensor:
    """The grid frames ``0, stride, ..`` of ``reader``, u8 ``(J, C, H, W)``
    on ``dev``."""
    j = math.ceil(len(reader) / stride)
    x = torch.empty(j, *reader.shape, dtype=torch.uint8, device=dev)
    for i in range(0, j, chunk):
        x[i : i + chunk] = reader[i * stride : (i + chunk) * stride : stride].to(dev)
    return x


def _curves(
    rows: Callable[[int], Tensor],
    names: Sequence[str],
    *,
    frames: int,
    stride: int,
    fps: float,
    ns: Sequence[int],
    centres: Sequence[str],
    dev: torch.device,
) -> dict[str, Any]:
    """The curves (module docstring) of the rows ``rows(a)`` ``(J_a, D)`` of
    each video, a data point ``frames`` consecutive rows."""
    started = time.time()
    means, lengths = [], []
    for a, name in enumerate(names):  # pass 1: the means
        x = rows(a)
        means.append(frame_mean(x, dev=dev).cpu())
        lengths.append(len(x))
        print(f"mean {name} {time.time() - started:.0f}s", flush=True)
    j = torch.tensor(lengths, dtype=torch.float64)
    m_all = torch.stack(means)  # (A, D) float64
    m = (j[:, None] * m_all).sum(0) / j.sum()
    lags = log_lags(min(lengths) - frames)
    per_video = []
    for a, name in enumerate(names):  # pass 2: the products
        per_video.append(lag_sums(rows(a), m_all[a], m, lags, frames=frames))
        print(f"lags {name} {time.time() - started:.0f}s", flush=True)
    d = m_all - m
    gram = d @ d.T
    n = torch.tensor([s["points"] for s in per_video], dtype=torch.float64)
    t95 = t_quantile(0.975, len(names) - 1)
    out: dict[str, Any] = {
        "videos": list(names),
        "rows": lengths,
        "points": n.tolist(),
        "stride": stride,
        "stack_frames": frames,
        "fps": fps,
        "lags": [stride * k for k in lags],  # in video frames
        "ns": list(ns),
        "t95": t95,
        "gram": gram.tolist(),
        "curves": {},
    }
    for c in centres:
        cur = curve(per_video, c)
        if c == "g":
            sq = torch.tensor([s["sq_g"] for s in per_video], dtype=torch.float64)
            cur["reference"] = reference(gram, n, sq, float(frames))
        else:
            cur["reference"] = {"ref": 0.0, "se": 0.0}
        upper = [r + t95 * s for r, s in zip(cur["rho"], cur["se"], strict=True)]
        cur["cross"] = {
            str(k): {
                "bound": 1 / (k - 1),
                "lag": crossing(out["lags"], cur["rho"], 1 / (k - 1)),
                "lag_upper": crossing(out["lags"], upper, 1 / (k - 1)),
                "videos": [
                    crossing(out["lags"], v, 1 / (k - 1)) for v in cur["videos"]
                ],
            }
            for k in ns
        }
        cur["neff"] = {str(k): [n_eff(r, k) for r in cur["rho"]] for k in ns}
        out["curves"][c] = cur
    return out


def _training(root: Path, held_out: str) -> tuple[list[str], list[U8Reader]]:
    names = [v for v in videos(root) if v != held_out]
    return names, [U8Reader(root / v) for v in names]


def corr(args: argparse.Namespace) -> dict[str, Any]:
    """The command ``corr`` (module docstring)."""
    dev = torch.device(args.device)
    names, readers = _training(args.root, args.held_out)

    def rows(a: int) -> Tensor:
        return _grid(readers[a], args.stride, dev).flatten(1)

    out = _curves(
        rows,
        names,
        frames=args.frames,
        stride=args.stride,
        fps=args.fps,
        ns=args.ns,
        centres=("g", "w"),
        dev=dev,
    )
    return {"root": str(args.root), "frames": [len(r) for r in readers]} | out


def _model(spec: str, runs: Path, device: str) -> tuple[VisionModules, list[str]]:
    """The modules of ``batch:run:ckpt:codes`` (``codes`` joined by ``+``)."""
    batch, rid, which, codes = spec.split(":")
    job = json.loads((runs / batch / rid / "job.json").read_text())
    conf = dataclasses.replace(from_dict(job["config"]), device=device)
    assert conf.vision is not None and conf.vision.layout == "packed"
    mods = build_vision(conf, seed=conf.seed)
    if which != "init":
        path = runs / batch / rid / "ckpt" / f"{which}.pt"
        ckpt.load_eval(path, mods.named(), group=job["group"])
    for m in mods.named().values():
        m.eval()
    return mods, codes.split("+")


def reprs(args: argparse.Namespace) -> dict[str, Any]:
    """The command ``repr`` (module docstring)."""
    dev = torch.device(args.device)
    names, readers = _training(args.root, args.held_out)
    out: dict[str, Any] = {"root": str(args.root), "frames": [len(r) for r in readers]}
    out["models"], out["codes"] = {}, {}
    for item in args.model:
        label, spec = item.split("=")
        mods, kinds = _model(spec, args.runs, args.device)
        frames = mods.chunk[0]
        assert frames == args.frames
        started = time.time()
        codes: dict[str, list[Tensor]] = {k: [] for k in kinds}
        for name, r in zip(names, readers, strict=True):
            grid = _grid(r, args.stride, dev)
            for k, v in stack_codes(mods, grid, frames, kinds, args.batch).items():
                codes[k].append(v)
            del grid
            print(f"codes {label} {name} {time.time() - started:.0f}s", flush=True)
        out["models"][label] = spec
        for k in kinds:
            rows = codes.pop(k)
            common: dict[str, Any] = {
                "frames": 1,
                "stride": args.stride,
                "fps": args.fps,
                "ns": args.ns,
                "centres": ("g",),
                "dev": dev,
            }
            cur = _curves(lambda a, rows=rows: rows[a].to(dev), names, **common)
            m = torch.cat([x.double().sum(0, keepdim=True) for x in rows]).sum(0) / sum(
                len(x) for x in rows
            )
            w, info = whitener(rows, m, dev=dev)
            m_dev = m.to(dev)

            def white(a: int, rows=rows, w=w, m_dev=m_dev) -> Tensor:
                return ((rows[a].to(dev).double() - m_dev) @ w).float()

            wcur = _curves(white, names, **common)
            cur["curves"]["white"] = wcur["curves"]["g"]
            cur["whitener"] = info
            if k in args.spectrum:
                grid = [lag // args.stride for lag in cur["lags"]]
                cur["spectrum"] = lag_spectrum(
                    [white(a) for a in range(len(names))], grid, dev=dev
                )
            cur["dim"] = rows[0].shape[1]
            out["codes"][f"{label}/{k}"] = cur
            print(f"white {label}/{k} rank {info['rank']}", flush=True)
            del rows, w
    return out


def budgets(args: argparse.Namespace) -> dict[str, Any]:
    """The command ``budget`` (module docstring)."""
    speeds = [run_speed(run) for run in args.runs]
    rows = []
    loop = None
    for path in args.corr:
        c = json.loads(path.read_text())
        loop = int(sum(c["frames"]))
        sets = c["codes"] if "codes" in c else {"pixels": c}
        for label, s in sets.items():
            for key, cur in s["curves"].items():
                for k, cr in cur["cross"].items():
                    tau = None if cr["lag"] is None else cr["lag"] / s["fps"]
                    rows.append(
                        {
                            "curve": f"{label}/{key}",
                            "n": int(k),
                            "tau": tau,
                            "loop_points": None
                            if tau is None
                            else math.floor(loop / s["fps"] / tau),
                            "runs": [
                                None
                                if tau is None
                                else budget(
                                    sp,
                                    tau,
                                    wall=args.wall,
                                    loop=loop,
                                    frames=s["stack_frames"],
                                    pixels=args.pixels,
                                    dims=args.dims,
                                )
                                for sp in speeds
                            ],
                        }
                    )
    return {"loop": loop, "wall": args.wall, "speeds": speeds, "rows": rows}


@torch.no_grad()
def drifts(args: argparse.Namespace) -> dict[str, Any]:
    """The command ``drift`` (module docstring)."""
    rows = []
    for run in args.runs:
        job = json.loads((run / "job.json").read_text())
        conf = dataclasses.replace(from_dict(job["config"]), device=args.device)
        mods = build_vision(conf, seed=conf.seed)
        tokens = conf.measure.posthoc_code == "tokens"
        windows = stream_windows(conf, args.root, windows=args.windows, steps=0)
        dev = torch.device(args.device)
        codes: dict[float, Tensor] = {}
        for path in sorted((run / "ckpt").glob("*.pt")):
            head = ckpt.load_eval(path, mods.named(), group=job["group"])
            zs = []
            for i in range(0, len(windows), args.batch):
                x = windows[i : i + args.batch].to(dev)
                z = mods.tokens(x).flatten(1) if tokens else mods.encode(x)
                zs.append(z.float().cpu())
            codes[float(head["t"])] = torch.cat(zs)
        ts = sorted(codes)
        n = len(windows)
        pairs = [
            {
                "t": a,
                "t2": b,
                "delta": b - a,
                "drift": drift(codes[a], codes[b]),
                # the numerator and the denominator of the drift, per number
                "change": float(((codes[a] - codes[b]).double() ** 2).mean()),
                "apart": 2
                * n
                / (n - 1)
                * float(codes[b].double().var(0, correction=0).mean()),
            }
            for i, a in enumerate(ts)
            for b in ts[i + 1 :]
        ]
        rows.append(
            {
                "run": run.name,
                "batch": run.parent.name,
                "code": "tokens" if tokens else "z",
                "dim": codes[ts[0]].shape[1],
                "stacks": n,
                "streams": conf.data.streams,
                "pairs": pairs,
            }
        )
        print(f"drift {run.name}", flush=True)
    return {"root": str(args.root), "windows": args.windows, "runs": rows}


# -- the report --------------------------------------------------------------------


def _dur(s: float | None) -> str:
    """Video seconds as ``x 秒`` or ``x 分``, ``–`` for None."""
    if s is None:
        return "–"
    return f"{s:.3g} 秒" if s < 60 else f"{s / 60:.3g} 分"


def _f(x: float | None, fmt: str = ".3g") -> str:
    return "–" if x is None or x != x else format(x, fmt)


def _table(head: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    out = ["| " + " | ".join(head) + " |", "| -- |" + " --: |" * (len(head) - 1)]
    return out + ["| " + " | ".join(r) + " |" for r in rows]


def report(args: argparse.Namespace) -> str:
    """The command ``report``: the tables of results/m17.md (markdown) from
    the ``corr.json``, the ``--repr`` files, ``budget.json`` and
    ``drift.json`` of ``--dir``."""
    c = json.loads((args.dir / "corr.json").read_text())
    curves = {f"pixels/{k}": v for k, v in c["curves"].items()}
    codes: dict[str, Any] = {}
    for name in args.repr:
        r = json.loads((args.dir / name).read_text())
        assert r["codes"] and all(x["lags"] == c["lags"] for x in r["codes"].values())
        for label, x in r["codes"].items():
            for centre, cur in x["curves"].items():
                curves[f"{label}/{centre}"] = cur
            codes[label] = x
    fps, lags, ns = c["fps"], c["lags"], c["ns"]
    octaves = [i for i in range(len(lags)) if i % 3 == 0 or i == len(lags) - 1]
    parts = [  # the trace ratios (and pixels/w), then the whitened means
        {k: v for k, v in curves.items() if not k.endswith("/white")},
        {k: v for k, v in curves.items() if k.endswith("/white")},
    ]
    lines = [f"t95 (dof {len(c['videos']) - 1}) = {c['t95']:.4f}", ""]
    for group in parts:
        lines += _table(
            ["ラグ", *group],
            [
                [_dur(lags[i] / fps)]
                + [f"{v['rho'][i]:.4f} ± {v['se'][i]:.4f}" for v in group.values()]
                for i in octaves
            ],
        )
        lines += [""]
    head = ["曲線", "別々の動画", "動画の平均の割合"]
    head += [f"N={k}（{1 / (k - 1):.4f}）" for k in ns]
    head += [f"以後ずっと N={k}" for k in ns] + ["最後のラグの ρ"]
    rows = []
    for key, v in curves.items():
        ref = v["reference"]
        cells = [
            key,
            f"{ref['ref']:.4f} ± {ref['se']:.4f}",
            _f(ref.get("share_pooled"), ".4f"),
        ]
        for k in ns:
            cr = v["cross"][str(k)]
            lo = None if cr["lag"] is None else cr["lag"] / fps
            hi = None if cr["lag_upper"] is None else cr["lag_upper"] / fps
            cells += [f"{_dur(lo)}（{_dur(hi)}）"]
        for k in ns:
            at = settled(lags, v["rho"], 1 / (k - 1))
            cells += [_dur(None if at is None else at / fps)]
        cells += [f"{v['rho'][-1]:.4f} ± {v['se'][-1]:.4f}"]
        rows.append(cells)
    lines += _table(head, rows) + [""]
    stride = c["stride"]
    grid = [lag // stride for lag in lags]
    for group in parts:
        g_curves = {k: v for k, v in group.items() if not k.endswith("/w")}
        lines += _table(
            ["ラグ", *g_curves],
            [
                [_dur(lags[i] / fps)]
                + [
                    " / ".join(_f(v["neff"][str(k)][i], ".0f") for k in ns)
                    for v in g_curves.values()
                ]
                for i in octaves
            ],
        )
        lines += [""]
    for group in parts:  # a window of K grid points, K - 1 steps wide
        g_curves = {k: v for k, v in group.items() if not k.endswith("/w")}
        lines += _table(
            ["窓", *g_curves],
            [
                [_dur(lags[i] / fps)]
                + [
                    f"{(rb := window_mean(grid, v['rho'], grid[i] + 1)):.4f}"
                    f"（{_f(1 / rb if rb > 0 else None, '.0f')}）"
                    for v in g_curves.values()
                ]
                for i in octaves
            ],
        )
        lines += [""]
    head = ["符号", "D", "rank", "参加比", "上位 5 の割合"]
    lines += _table(
        head,
        [
            [
                label,
                f"{x['dim']}",
                f"{x['whitener']['rank']}",
                f"{x['whitener']['pr']:.3g}",
                f"{sum(x['whitener']['eig_top'][:5]):.3f}",
            ]
            for label, x in codes.items()
        ],
    )
    spec = {label: x["spectrum"] for label, x in codes.items() if "spectrum" in x}
    lines += [""] + _table(
        ["ラグ", *spec],
        [
            [_dur(lags[i] / fps)]
            + [f"{v[i][0]:.4f} 〜 {v[i][1]:.4f}" for v in spec.values()]
            for i in octaves
        ],
    )
    lines += [""]
    px = c["curves"]["g"]
    rows = []
    for a, name in enumerate(c["videos"]):
        cells = [name, f"{c['frames'][a]}", f"{px['reference']['share'][a]:.3f}"]
        cells += [
            _dur(None if (x := px["cross"][str(k)]["videos"][a]) is None else x / fps)
            for k in ns
        ]
        rows.append(cells)
    lines += _table(
        ["動画", "フレーム", "動画の平均の割合", *[f"N={k}" for k in ns]], rows
    )
    bud = json.loads((args.dir / "budget.json").read_text())
    lines += ["", f"loop {bud['loop'] / fps:.0f} s"]
    by_b: dict[int, list[int]] = {}
    for i, sp in enumerate(bud["speeds"]):
        by_b.setdefault(sp["streams"], []).append(i)
    rows = []
    for row in bud["rows"]:
        if row["tau"] is None:
            continue
        cells = [row["curve"], f"{row['n']}", _dur(row["tau"]), f"{row['loop_points']}"]
        for _, idx in sorted(by_b.items(), reverse=True):
            bs = [row["runs"][i] for i in idx]
            for key in (
                "points_per_stream",
                "points_per_stream_wall",
                "points_distinct",
            ):
                lo, hi = min(b[key] for b in bs), max(b[key] for b in bs)
                cells += [str(lo) if lo == hi else f"{lo}–{hi}"]
        rows.append(cells)
    head = ["曲線", "N", "Δ_N", "ループの点"]
    for b in sorted(by_b, reverse=True):
        head += [f"B={b} 1 本/run", f"B={b} 1 本/壁時計", f"B={b} distinct"]
    lines += _table(head, rows)
    for b, idx in sorted(by_b.items(), reverse=True):
        lo = min(bud["speeds"][i]["steps_per_s"] for i in idx)
        hi = max(bud["speeds"][i]["steps_per_s"] for i in idx)
        lines += ["", f"B={b}: {len(idx)} runs, {lo:.2f}–{hi:.2f} step/s"]
    dr = json.loads((args.dir / "drift.json").read_text())
    groups: dict[tuple[str, str, int, int, float], list[dict[str, float]]] = {}
    for r in dr["runs"]:  # the runs of a batch, code, D and B together
        for p in r["pairs"]:
            if p["delta"] <= args.max_delta:
                key = (r["batch"], r["code"], r["dim"], r["streams"], p["delta"])
                groups.setdefault(key, []).append(p)
    rows = []
    for (batch, code, dim, b, delta), ps in sorted(groups.items()):
        rows.append(
            [batch, code, f"{dim}", f"{b}", f"{delta:.0f}"]
            + [
                f"{min(p[k] for p in ps):.3g}–{max(p[k] for p in ps):.3g}"
                for k in ("drift", "change", "apart")
            ]
        )
    lines += [""] + _table(
        ["batch", "表現", "D", "B", "Δ（秒）", "drift", "change", "apart"], rows
    )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    """The command line (module docstring)."""
    p = argparse.ArgumentParser(description="The stream's statistics (diagnosis).")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("corr")
    e = sub.add_parser("repr")
    for q in (c, e):
        q.add_argument("--root", type=Path, required=True)
        q.add_argument("--held_out", default="Zurich")
        q.add_argument("--stride", type=int, default=3)
        q.add_argument("--frames", type=int, default=16)
        q.add_argument("--fps", type=float, default=30.0)
        q.add_argument("--ns", type=int, nargs="+", default=list(NS))
        q.add_argument("--device", default="cuda")
    e.add_argument("--runs", type=Path, default=Path("/nas/geson/mujou-exp/runs"))
    e.add_argument("--model", nargs="+", required=True)  # label=batch:run:ckpt:codes
    e.add_argument("--batch", type=int, default=128)
    e.add_argument("--spectrum", nargs="*", default=["z"])  # codes of lag_spectrum
    b = sub.add_parser("budget")
    b.add_argument("--corr", type=Path, nargs="+", required=True)
    b.add_argument("--runs", type=Path, nargs="+", required=True)
    b.add_argument("--wall", type=float, default=1800.0)
    b.add_argument("--pixels", type=int, default=3 * 128 * 128)
    b.add_argument("--dims", type=int, nargs="+", default=[256, 512, 1024, 2048])
    d = sub.add_parser("drift")
    d.add_argument("--root", type=Path, required=True)
    d.add_argument("--runs", type=Path, nargs="+", required=True)
    d.add_argument("--windows", type=int, default=2048)
    d.add_argument("--batch", type=int, default=64)
    d.add_argument("--device", default="cuda")
    r = sub.add_parser("report")
    r.add_argument("--dir", type=Path, required=True)
    r.add_argument("--repr", nargs="*", default=[])
    r.add_argument("--max_delta", type=float, default=900.0)
    for q in (c, e, b, d, r):
        q.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.cmd == "report":
        args.out.write_text(report(args))
        return
    run = {"corr": corr, "repr": reprs, "budget": budgets, "drift": drifts}[args.cmd]
    args.out.write_text(json.dumps(run(args), indent=1))


if __name__ == "__main__":
    main()
