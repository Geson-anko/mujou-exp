r"""DIAGNOSIS ONLY (M17, 2026-11-13): what the stream offers a long-term buffer
of data points for a SIGReg of one sample per data point (a stack), measured
before such a buffer is built. Not a program of the runs, and nothing of the
runs imports it.

- ``corr``: the normalized autocovariance of the frames and of the stacks of
  the training videos against the lag. A frame $x_t$ is flattened (128 x 128
  x 3 in $[0, 1]$); $m$ is the mean frame of the training set (every frame of
  the nine videos), $m_a$ that of video $a$, $d_a = m_a - m$; $y_t = x_t -
  m$ (centred by the training set) and $u_t = x_t - m_a$ (by the video). On
  the grid $t = 0, s, 2s, \dots$ of a video ($s$ = ``stride``, 3: the stacks'
  spacing and the stream's advance per learning step)

  $$\rho_a(\Delta) = \frac{\operatorname{mean}_t \langle y_t, y_{t+\Delta}
  \rangle}{\operatorname{mean}_t \lVert y_t \rVert^2}$$

  over the grid points with $t + \Delta$ in the video (the denominator over
  the whole grid), ``g`` below; ``w`` the same of $u$ (centred by the
  video). A stack $Y_t = (y_{t - (N-1)s}, \dots, y_t)$ ($N$ = ``frames``,
  16) is a data point; its products are the sums of its frames' products,
  $\langle Y_t, Y_{t+\Delta} \rangle = \sum_{k<N} \langle y_{t-ks},
  y_{t+\Delta-ks} \rangle$ (``stack_products``), over the grid points with
  the whole stack in the video. The pooled curve is the same ratio over the
  grid points of all the videos. Its sampling error is by batch means: the
  grid points of a lag are cut into ``blocks`` contiguous blocks, the ratio of
  each block (over the same denominator) varies as a stretch of video does,
  and the error of the video's ratio is the blocks' std over
  $\sqrt{\mathrm{blocks}}$; that of the pooled ratio is
  $\sqrt{\sum_a w_a^2 \sigma_a^2}$, $w_a$ the video's share of the grid
  points.

  The i.i.d. reference: two frames of different videos. Since the mean over
  all pairs $(i, j)$ of video $a$ and $b$ of $\langle y_i, y_j \rangle$ is
  $\langle d_a, d_b \rangle$, the reference of the pooled ``g`` curve is
  $\sum_{a \ne b} n_a n_b \langle d_a, d_b \rangle / \sum_{a \ne b} n_a n_b$
  over its denominator ($n_a$ the frames of video $a$), $-\sum_a n_a^2
  \lVert d_a \rVert^2 / (n^2 - \sum_a n_a^2)$ by $\sum_a n_a d_a = 0$:
  slightly negative. Its spread is that of the 36 pairs of videos, and its
  error their std over $\sqrt{36}$ (the pairs share videos: a lower bound).
  Video $a$'s reference is the mean over the frames of the other videos,
  $\sum_{b \ne a} n_b \langle d_a, d_b \rangle / \sum_{b \ne a} n_b$ over its
  denominator. Of a stack, $N$ times these (a stack's mean is $m_a$ in each
  slot) over the stacks' denominator. The reference of the ``w`` curves is
  0 (two independent frames of one video have $E \langle u, u' \rangle = 0$).
  Where the frames of a video stop being correlated, $\rho_a \to \lVert d_a
  \rVert^2 / \operatorname{mean} \lVert y \rVert^2$ (``share``, the video's
  own mean): no lag within one video reaches the reference of two videos
  unless the video's content drifts.

  The orthogonal time $\tau$ is the smallest lag at which $\rho(\Delta) -
  \mathrm{ref} \le 1.96 \sqrt{\sigma(\Delta)^2 + \sigma_{\mathrm{ref}}^2}$
  (the two-sided 95% interval of the difference reaches 0), ``None`` if no
  measured lag does. Lags ``log_lags``: $2^{i/3}$ frames rounded, up to
  ``max_lag``.

- ``budget``: from finished runs (``job.json``, ``result.json``,
  ``exit/attempt-*.json``), the learning steps per wall-clock second
  (the run's units over the summed durations of its attempts; one unit is
  one learning step, ``data.tours.advance`` video frames), the video time a
  stream covers in the run and in ``--wall`` seconds, and, for each $\tau$
  of a ``corr`` output, the $\tau$-spaced data points a stream offers and all
  $B$ streams offer (``distinct``: of those, at most as many as the loop of
  the training videos holds), with the memory of their representations of
  $D$ numbers (float32) and of their raw stacks (u8); and ``videos``, the $\lfloor n_a / \tau_a \rfloor$
  data points of each whole video at its own $\tau_a$ and their sum, all the
  training set offers.

- ``drift``: the staleness of a stored representation. The same held-in
  stacks (``encoder_eval.stream_windows`` with no step after the first) are
  encoded by each evaluation checkpoint of a run (the flattened target
  tokens when ``measure.posthoc_code`` is ``"tokens"``, else $z$), and for
  each pair of checkpoints $\Delta$ apart

  $$\mathrm{drift} = \frac{E \lVert f_t(x) - f_{t+\Delta}(x) \rVert^2}{E
  \lVert f_{t+\Delta}(x) - E f_{t+\Delta} \rVert^2}$$

  (``drift``): the change of a representation over its spread across stacks.

- ``report``: the tables of the three outputs as markdown.

Run: ``python -m exp.video.diag.stream_stats corr --root DATA --out JSON``,
``... budget --corr JSON --runs DIR... --out JSON``, ``... drift --root DATA
--runs DIR... --out JSON``, ``... report --dir DIR --out MD``.
"""

import argparse
import dataclasses
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from exp.lib import ckpt
from exp.lib.video_io import U8Reader
from exp.video.compose import from_dict
from exp.video.data.walking_tours import videos
from exp.video.jepa.vit import build_vision
from exp.video.measure.encoder_eval import stream_windows

__all__ = [
    "Z95",
    "between_reference",
    "budget",
    "curve",
    "drift",
    "first_reach",
    "frame_mean",
    "lag_sums",
    "log_lags",
    "main",
    "report",
    "run_speed",
    "stack_products",
]

Z95 = 1.96  # the two-sided 95% quantile of the standard normal
KINDS = ("frame", "stack")
CENTRES = ("g", "w")  # by the training set's mean frame, by the video's


def log_lags(max_lag: int, per_octave: int = 3) -> list[int]:
    """The distinct integers $\\mathrm{round}(2^{i / \\mathrm{per\\_octave}})$
    from 1 up to ``max_lag``."""
    top = math.floor(per_octave * math.log2(max_lag) + 1e-9)
    return sorted({round(2 ** (i / per_octave)) for i in range(top + 1)})


@torch.no_grad()
def frame_mean(
    x: Tensor, chunk: int = 4096, *, dev: torch.device | None = None
) -> Tensor:
    """The mean ``(D,)`` float64 of the frames ``x`` ``(L, D)``, summed a
    chunk at a time on ``dev`` (that of ``x`` by default)."""
    dev = x.device if dev is None else dev
    s = torch.zeros(x.shape[1], dtype=torch.float64, device=dev)
    for i in range(0, len(x), chunk):
        s += x[i : i + chunk].to(dev).double().sum(0)
    return s / len(x)


def stack_products(p: Tensor, frames: int) -> Tensor:
    """The products of the stacks ending at grid points ``frames - 1 ..``
    from the products ``p`` ``(J,)`` of their frames on the grid: $P_j =
    \\sum_{k < N} p_{j-k}$, ``(J - N + 1,)`` float64."""
    c = torch.cat([p.new_zeros(1, dtype=torch.float64), p.double().cumsum(0)])
    return c[frames:] - c[:-frames]


def _blocks(p: Tensor, blocks: int) -> tuple[list[float], list[int]]:
    parts = torch.tensor_split(p, blocks)
    return [float(b.sum()) for b in parts], [len(b) for b in parts]


@torch.no_grad()
def lag_sums(
    x: Tensor,
    mean: Tensor,
    lags: Sequence[int],
    *,
    stride: int,
    frames: int,
    blocks: int,
    chunk: int = 2048,
) -> dict[str, Any]:
    """The sums of one video ``x`` ``(L, D)`` (u8 or float, on any device)
    behind its curves (module docstring), $m$ = ``mean`` ``(D,)``:

    - ``v_<kind>_<c>``, ``n_<kind>``: the sum of the squared norms over the
      grid (``kind`` ``frame`` or ``stack``; ``c`` ``g`` centred by $m$,
      ``w`` by the video's mean) and their number;
    - ``<kind>_<c>`` ``(lags, blocks)``, ``<kind>_n`` ``(lags, blocks)``: the
      sums of the products at each lag in each block, and their numbers;
    - ``frames``: $L$.
    """
    dev = x.device
    m_a = frame_mean(x)
    d = (m_a - mean.double().to(dev)).float()
    m32 = m_a.float()
    norms, ed = [], []  # |u_t|^2 and <d, u_t> of every frame
    for i in range(0, len(x), chunk):
        u = x[i : i + chunk].float() - m32
        norms.append((u * u).sum(1).double())
        ed.append((u @ d).double())
    n_t, e_t = torch.cat(norms), torch.cat(ed)
    dd = float((d.double() ** 2).sum())
    grid = torch.arange(0, len(x), stride, device=dev)
    nw = n_t[grid]
    ng = nw + 2 * e_t[grid] + dd
    out: dict[str, Any] = {
        "frames": len(x),
        "n_frame": len(grid),
        "n_stack": len(grid) - frames + 1,
        "v_frame_w": float(nw.sum()),
        "v_frame_g": float(ng.sum()),
        "v_stack_w": float(stack_products(nw, frames).sum()),
        "v_stack_g": float(stack_products(ng, frames).sum()),
    }
    for key in ("frame_g", "frame_w", "stack_g", "stack_w", "frame_n", "stack_n"):
        out[key] = []
    for lag in lags:
        anchors = grid[grid + lag < len(x)]
        p_w = torch.cat(
            [
                ((x[a].float() - m32) * (x[a + lag].float() - m32)).sum(1).double()
                for a in anchors.split(chunk)
            ]
        )
        p_g = p_w + e_t[anchors] + e_t[anchors + lag] + dd
        for c, p in (("g", p_g), ("w", p_w)):
            sums, counts = _blocks(p, blocks)
            out[f"frame_{c}"].append(sums)
            ssums, scounts = _blocks(stack_products(p, frames), blocks)
            out[f"stack_{c}"].append(ssums)
        out["frame_n"].append(counts)
        out["stack_n"].append(scounts)
    return out


def curve(per_video: Sequence[Mapping[str, Any]], kind: str, c: str) -> dict:
    """The curves of ``kind`` and centring ``c`` from the ``lag_sums`` of each
    video: per video and pooled, ``rho`` and its batch-means error ``se``
    (lists over the lags), and ``v`` the denominator (module docstring)."""
    v_pool = sum(s[f"v_{kind}_{c}"] for s in per_video) / sum(
        s[f"n_{kind}"] for s in per_video
    )
    videos_out = []
    num = torch.zeros(len(per_video[0][f"{kind}_{c}"]), dtype=torch.float64)
    cnt = torch.zeros_like(num)
    var_pool = torch.zeros_like(num)
    for s in per_video:
        sums = torch.tensor(s[f"{kind}_{c}"], dtype=torch.float64)  # (lags, blocks)
        n = torch.tensor(s[f"{kind}_n"], dtype=torch.float64)
        v = s[f"v_{kind}_{c}"] / s[f"n_{kind}"]
        rho = sums.sum(1) / n.sum(1) / v
        k = sums.shape[1]
        se = (sums / n).std(1) / math.sqrt(k) / v
        videos_out.append({"v": v, "rho": rho.tolist(), "se": se.tolist()})
        num += sums.sum(1)
        cnt += n.sum(1)
        var_pool += n.sum(1) ** 2 * (se * v / v_pool) ** 2
    rho = num / cnt / v_pool
    se = var_pool.sqrt() / cnt
    return {
        "v": v_pool,
        "rho": rho.tolist(),
        "se": se.tolist(),
        "videos": videos_out,
    }


def between_reference(
    gram: Tensor, n: Tensor, v: Tensor, v_pool: float, scale: float = 1.0
) -> dict[str, Any]:
    """The i.i.d. reference of two data points of different videos (module
    docstring) from ``gram`` ``(A, A)`` $\\langle d_a, d_b \\rangle$, the
    frames ``n`` ``(A,)``, the denominators ``v`` ``(A,)`` of the videos and
    ``v_pool`` of the pooled curve; ``scale`` $N$ for stacks. The pooled
    ``ref``, its error ``se`` and the ``pairs``' values; ``videos`` per video;
    ``share`` $N \\lVert d_a \\rVert^2$ over the video's denominator."""
    a = len(n)
    off = ~torch.eye(a, dtype=torch.bool)
    g, n = gram.double(), n.double()
    w = n[:, None] * n[None, :] * off
    pooled = float((w * g).sum() / w.sum()) * scale / v_pool
    iu = torch.triu_indices(a, a, 1)
    pairs = g[iu[0], iu[1]] * scale / v_pool
    per = [
        float((n * g[i] * off[i]).sum() / (n * off[i]).sum()) * scale / float(v[i])
        for i in range(a)
    ]
    return {
        "ref": pooled,
        "se": float(pairs.std() / math.sqrt(len(pairs))),
        "pairs": pairs.tolist(),
        "videos": per,
        "share": [float(g[i, i]) * scale / float(v[i]) for i in range(a)],
        "share_pooled": float((n * g.diagonal()).sum() / n.sum()) * scale / v_pool,
    }


def first_reach(
    lags: Sequence[int],
    rho: Sequence[float],
    se: Sequence[float],
    ref: float,
    se_ref: float = 0.0,
) -> int | None:
    """The smallest lag at which $\\rho - \\mathrm{ref} \\le 1.96
    \\sqrt{\\sigma^2 + \\sigma_{\\mathrm{ref}}^2}$, or None."""
    for lag, r, s in zip(lags, rho, se, strict=True):
        if r - ref <= Z95 * math.sqrt(s * s + se_ref * se_ref):
            return lag
    return None


def drift(a: Tensor, b: Tensor) -> float:
    """$E \\lVert a - b \\rVert^2 / E \\lVert b - E b \\rVert^2$ over the
    rows of ``a`` and ``b`` ``(n, D)``."""
    a, b = a.double(), b.double()
    return float(((a - b) ** 2).sum(1).mean() / ((b - b.mean(0)) ** 2).sum(1).mean())


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


# -- the commands -----------------------------------------------------------------


def _load(reader: U8Reader, dev: torch.device, chunk: int = 4096) -> Tensor:
    """All frames of ``reader`` as u8 ``(L, D)`` on ``dev``."""
    x = torch.empty(len(reader), math.prod(reader.shape), dtype=torch.uint8, device=dev)
    for i in range(0, len(reader), chunk):
        x[i : i + chunk] = reader[i : i + chunk].flatten(1).to(dev)
    return x


def corr(args: argparse.Namespace) -> dict[str, Any]:
    """The command ``corr`` (module docstring)."""
    dev = torch.device(args.device)
    names = [v for v in videos(args.root) if v != args.held_out]
    readers = [U8Reader(args.root / v) for v in names]
    n = torch.tensor([len(r) for r in readers], dtype=torch.float64)
    started = time.time()
    means = []
    for name, r in zip(names, readers, strict=True):  # pass 1: the mean frames
        means.append(frame_mean(r[:].flatten(1), dev=dev).cpu())
        print(f"mean {name} {time.time() - started:.0f}s", flush=True)
    m_all = torch.stack(means)  # (A, D) float64
    m = (n[:, None] * m_all).sum(0) / n.sum()
    lags = log_lags(args.max_lag)
    per_video = []
    for name, r in zip(names, readers, strict=True):  # pass 2: the products
        x = _load(r, dev)
        s = lag_sums(
            x, m, lags, stride=args.stride, frames=args.frames, blocks=args.blocks
        )
        del x
        per_video.append(s)
        print(f"lags {name} {time.time() - started:.0f}s", flush=True)
    d = m_all - m
    gram = d @ d.T
    out: dict[str, Any] = {
        "root": str(args.root),
        "videos": names,
        "frames": n.tolist(),
        "lags": lags,
        "stride": args.stride,
        "stack_frames": args.frames,
        "blocks": args.blocks,
        "fps": args.fps,
        "gram": gram.tolist(),
        "curves": {},
    }
    for kind in KINDS:
        for c in CENTRES:
            cur = curve(per_video, kind, c)
            v = torch.tensor([x["v"] for x in cur["videos"]])
            if c == "g":
                scale = args.frames if kind == "stack" else 1
                ref = between_reference(gram, n, v, cur["v"], scale)
            else:
                ref = {"ref": 0.0, "se": 0.0, "videos": [0.0] * len(names)}
            cur["reference"] = ref
            cur["tau"] = first_reach(lags, cur["rho"], cur["se"], ref["ref"], ref["se"])
            for vid, rv in zip(cur["videos"], ref["videos"], strict=True):
                vid["tau"] = first_reach(lags, vid["rho"], vid["se"], rv, ref["se"])
            out["curves"][f"{kind}_{c}"] = cur
    return out


def budgets(args: argparse.Namespace) -> dict[str, Any]:
    """The command ``budget`` (module docstring)."""
    c = json.loads(args.corr.read_text())
    loop = int(sum(c["frames"]))
    taus = {}
    for key, cur in c["curves"].items():
        taus[key] = None if cur["tau"] is None else cur["tau"] / c["fps"]
    per_video = {}
    for key, cur in c["curves"].items():
        pts = {
            name: None if v["tau"] is None else math.floor(n / v["tau"])
            for name, n, v in zip(c["videos"], c["frames"], cur["videos"], strict=True)
        }
        per_video[key] = pts | {"total": sum(p for p in pts.values() if p is not None)}
    rows = []
    for run in args.runs:
        sp = run_speed(run)
        sp["budgets"] = {
            key: None
            if tau is None
            else budget(
                sp,
                tau,
                wall=args.wall,
                loop=loop,
                frames=c["stack_frames"],
                pixels=args.pixels,
                dims=args.dims,
            )
            for key, tau in taus.items()
        }
        rows.append(sp)
    return {
        "loop": loop,
        "wall": args.wall,
        "taus": taus,
        "videos": per_video,
        "runs": rows,
    }


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
        pairs = [
            {
                "t": a,
                "t2": b,
                "delta": b - a,
                "drift": drift(codes[a], codes[b]),
                # the numerator and the denominator of the drift, per number
                "change": float(((codes[a] - codes[b]).double() ** 2).mean()),
                "spread": float(codes[b].double().var(0, correction=0).mean()),
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
                "stacks": len(windows),
                "streams": conf.data.streams,
                "pairs": pairs,
            }
        )
        print(f"drift {run.name}", flush=True)
    return {"root": str(args.root), "windows": args.windows, "runs": rows}


def _g(x: float | None) -> str:
    return "–" if x is None else f"{x:.3g}"


def report(args: argparse.Namespace) -> str:
    """The command ``report``: the tables of results/m17.md (markdown) from
    the ``corr.json``, ``budget.json`` and ``drift.json`` of ``--dir``."""
    c = json.loads((args.dir / "corr.json").read_text())
    bud = json.loads((args.dir / "budget.json").read_text())
    dr = json.loads((args.dir / "drift.json").read_text())
    fps, lines = c["fps"], []
    cur = c["curves"]
    lines += ["| lag (frames) | lag (s) | " + " | ".join(cur) + " |"]
    lines += ["| --: | --: |" + " --: |" * len(cur)]
    for i, lag in enumerate(c["lags"]):
        cells = [f"{v['rho'][i]:.4f} ± {v['se'][i]:.4f}" for v in cur.values()]
        lines += [f"| {lag} | {lag / fps:.4g} | " + " | ".join(cells) + " |"]
    lines += ["", "| video | frames | " + " | ".join(f"tau {k}" for k in cur)]
    lines[-1] += " | share | ref |"
    lines += ["| -- | --: |" + " --: |" * (len(cur) + 2)]
    ref = cur["frame_g"]["reference"]
    for a, name in enumerate(c["videos"]):
        taus = [v["videos"][a]["tau"] for v in cur.values()]
        cells = [_g(None if t is None else t / fps) for t in taus]
        lines += [
            f"| {name} | {c['frames'][a]:.0f} | "
            + " | ".join(cells)
            + f" | {ref['share'][a]:.3f} | {ref['videos'][a]:.4f} |"
        ]
    pooled = [_g(None if v["tau"] is None else v["tau"] / fps) for v in cur.values()]
    lines += [
        f"| pooled | {sum(c['frames']):.0f} | "
        + " | ".join(pooled)
        + f" | {ref['share_pooled']:.4f} | {ref['ref']:.4f} ± {ref['se']:.4f} |"
    ]
    pairs = ref["pairs"]
    lines += [
        "",
        f"pairs of videos: {len(pairs)}, min {min(pairs):.4f}, max {max(pairs):.4f},"
        f" std {torch.tensor(pairs).std():.4f}",
        "points of each whole video at its tau: "
        + ", ".join(f"{k} {v['total']}" for k, v in bud["videos"].items()),
        "",
        "| run | B | steps/s | video/run (s) | video/wall (s) | key | tau (s)"
        " | per stream | per stream (wall) | total | distinct | MB D=256..2048"
        " | MB raw |",
        "| --" + " | --" * 12 + " |",
    ]
    for r in bud["runs"]:
        for key, b in r["budgets"].items():
            if b is None or not key.startswith("stack"):
                continue
            mb = "/".join(f"{v / 2**20:.2g}" for v in b["bytes_repr"].values())
            lines += [
                f"| {r['batch']}/{r['run']} | {r['streams']} | {r['steps_per_s']:.2f}"
                f" | {b['video_per_stream']:.0f} | {b['video_per_stream_wall']:.0f}"
                f" | {key} | {b['tau']:.0f} | {b['points_per_stream']}"
                f" | {b['points_per_stream_wall']} | {b['points_total']}"
                f" | {b['points_distinct']} | {mb} | {b['bytes_raw'] / 2**20:.0f} |"
            ]
    lines += ["", "| run | code | D | B | delta | drift (min–max) | change | spread |"]
    lines += ["| -- | -- | --: | --: | --: | -- | -- | -- |"]
    for r in dr["runs"]:
        for delta in sorted({p["delta"] for p in r["pairs"]}):
            ps = [p for p in r["pairs"] if p["delta"] == delta]
            if delta > args.max_delta:
                continue
            lo = min(p["drift"] for p in ps)
            hi = max(p["drift"] for p in ps)
            ch = [p["change"] for p in ps]
            sp = [p["spread"] for p in ps]
            lines += [
                f"| {r['batch']}/{r['run']} | {r['code']} | {r['dim']} | {r['streams']}"
                f" | {delta:.0f} | {lo:.3g}–{hi:.3g} | {min(ch):.3g}–{max(ch):.3g}"
                f" | {min(sp):.3g}–{max(sp):.3g} |"
            ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    """The command line (module docstring)."""
    p = argparse.ArgumentParser(description="The stream's statistics (diagnosis).")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("corr")
    c.add_argument("--root", type=Path, required=True)
    c.add_argument("--held_out", default="Zurich")
    c.add_argument("--max_lag", type=int, default=2**16)  # 36 min at 30 fps
    c.add_argument("--stride", type=int, default=3)
    c.add_argument("--frames", type=int, default=16)
    c.add_argument("--blocks", type=int, default=10)
    c.add_argument("--fps", type=float, default=30.0)
    c.add_argument("--device", default="cuda")
    b = sub.add_parser("budget")
    b.add_argument("--corr", type=Path, required=True)
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
    r.add_argument("--max_delta", type=float, default=900.0)
    for q in (c, b, d, r):
        q.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.cmd == "report":
        args.out.write_text(report(args))
        return
    run = {"corr": corr, "budget": budgets, "drift": drifts}[args.cmd]
    args.out.write_text(json.dumps(run(args), indent=1))


if __name__ == "__main__":
    main()
