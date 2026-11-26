r"""The main evaluation dyn/copy (``plans/dyn.md``; ``results/dyn/README.md``,
「回す前に決めたこと」, registered before any number): of a code whitened
by its reference (``midpoint.Whitener``, PCA64), the direct predictors of
$w_{t+n}$ from $(w_t, w_{t-n})$ (``exp.lib.dynpred``: the linear family and
the residual MLP) learned on the training videos, their errors against the
copy's and the mean's on Zurich's common centres, at every lag. A record of
blocks' sums; the verdicts are ``results/dyn/judge_dyn.py``'s.

**The sets** (``VideoSets``, ``video_sets``): the training videos (all but
Zurich, in name order) permuted by ``DynConf.split_seed``, the first the
record (``rec``), the second the validation (``val``), the rest the
training set (``train``); each cut from its first frame into blocks of
``block`` frames (the tail dropped), each block's stacks the
``block - span`` whose frames lie in it (``SetLayout``). The centres are
those of every triple of ``max_lag`` in the block, the same at every lag
(C1). Zurich's are ``width_eval``'s common centres, their blocks and the
blocks shifted by half a block (a sensitivity).

**The rows of a code's place** (``code_rows``): of every lag $1 ..
256$ the linear (``lin``); of the grid (``dynpred.GRID``) also the one-input
linear (``lin1``), the copy, the constant speed (``vel``), the mean and the
one-scale linear (``scale``), and the residual MLP of each seed (``mlp``,
the chosen ``step``; its validation curve ``mlp_val``). Of ``rec`` and
``zurich`` the blocks' sums (``num_b``, ``copy_b``, ``clim_b``, ``n_b``,
Zurich's shifted ``*_s``), of ``train`` and ``val`` the totals. Beside:
the reference's spectrum (``spec``: the participation ratio of all its
eigenvalues, the directions below $\epsilon$) and the linear's numerator by
direction on Zurich (``dir_num``).

**The checks** (V-p1, synthetic: ``check_parts``, ``merge_checks``) and the
**yard** (V-p2, ``evaluate_yard``), the **runs** (``evaluate_run``,
``evaluate_vae``; the exploration)::

    OMP_NUM_THREADS=1 python -m exp.vision.dyn_eval checks --out runs/v18c-dyn/checks
    ... checks --out <part> --only long-S4,mlp-S4 | --only V-p1f,V-p1h --seeds 5101:5151
    ... merge-checks --out runs/v18c-dyn/checks <part> <part> ..
    python -m exp.vision.dyn_eval yard --config <v18c run> --sets runs/v18c-tlin \
        --out runs/v18c-dyn --init <..> --init-f <..> --init-m <..> --init-c <..> \
        --init-c1 <..> --init-v16 <..> [--codes pca,rot]
    python -m exp.vision.dyn_eval run <run_dir> --steps 6000,60000 --sets runs/v18c-tlin
    python -m exp.vision.dyn_eval vae <ckpt> --name v16 --sets runs/v18c-tlin \
        --out runs/v18c-dyn
"""

import argparse
import hashlib
import json
import math
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import stdev
from typing import Any

import torch
from torch import Tensor

from exp.lib import ckpt
from exp.lib.dynpred import (
    GRID,
    MAIN,
    LinFit,
    MLPConf,
    Triples,
    best_linear,
    dens,
    fit_linear,
    fixed_error,
    mean_log,
    num_sums,
    population_dens,
    record_preds,
    train_mlp,
)
from exp.lib.linear_dyn import ALPHA_SCALES
from exp.lib.metrics import block_counts
from exp.lib.midpoint import Whitener, triple_terms
from exp.lib.provenance import configure_determinism
from exp.lib.synth_series import (
    block_rotation,
    box_ar1_acf,
    box_mean,
    chebyshev,
    complex_ou,
    complex_ou_gamma,
    ou,
    rotation_gamma,
    white,
)
from exp.lib.tours.walking_tours import videos
from exp.lib.video_io import U8Reader
from exp.vision.content_eval import ContentConf, orthogonal
from exp.vision.eval import stream_view
from exp.vision.fill_eval import FillConf
from exp.vision.geom_eval import (  # the geom job's own helpers, reused as they are
    GeomConf,
    JepaPlaces,
    VaeTokenCode,
    _conf_of,
    _feature_code,
    _first_vae,
    _init_of,
    _jepa_of,
    _mean8,
    _start,
    _write,
    encode_places,
)
from exp.vision.jepa_diag import run_ckpts
from exp.vision.jepa_eval import PCA_SEED, ColorCode, PCACode, pixels, vae_at
from exp.vision.models import VAEModules, build_vae
from exp.vision.results.dyn.judge_dyn import Interval, psi_interval, words
from exp.vision.width_eval import (
    WidthConf,
    WidthSets,
    places_of,
    sets_and_ref,
    synthetic_sets,
)
from exp.vision.width_guard import _zpca_map

__all__ = [
    "OUT",
    "YARD_CODES",
    "CheckConf",
    "DynConf",
    "SetLayout",
    "VideoSets",
    "check_parts",
    "check_units",
    "clock_code",
    "clock_times",
    "code_rows",
    "evaluate_checks",
    "evaluate_run",
    "evaluate_vae",
    "evaluate_yard",
    "layout_of",
    "main",
    "merge_checks",
    "merge_checks_into",
    "split_videos",
    "synth_codes",
    "synth_gamma",
    "video_sets",
    "zurich_triples",
]

OUT = "posthoc_dyn"
YARD_INITS = ("init", "init_f", "init_m", "init_c", "init_c1")
YARD_CODES = ("pca", "rot", "color", "mean8", *YARD_INITS, "init_v16", "clock")
ROLES = ("train", "val", "rec")
BLOCKED = ("rec", "zurich")  # the sets whose rows hold blocks' sums

type Row = dict[str, Any]
type Places = Callable[[Tensor], dict[str, Tensor]]

_F64 = torch.float64


@dataclass(frozen=True)
class DynConf:
    """The settings (``plans/dyn.md``), by default the plan's: lags in
    stacks (0.1 s)."""

    grid: tuple[int, ...] = GRID
    max_lag: int = 256
    main: tuple[int, int] = MAIN
    below: tuple[int, int] = (1, 8)  # the description below the main range
    block: int = 1280  # 128 s
    split_seed: int = 4101
    scales: tuple[float, ...] = ALPHA_SCALES
    mlp: MLPConf = field(default_factory=MLPConf)
    rot_seed: int = ContentConf().rot_seed  # V-h4b's rot
    clock_dirs: int = 8
    clock_period: float = 64.0  # 6.4 s
    clock_seed: int = 5950
    clock_lag: int = 8
    rec_seed: int = 102  # V-p2d's draws of the record set


# -- the sets ----------------------------------------------------------------------


def split_videos(names: Sequence[str], seed: int) -> dict[str, tuple[str, ...]]:
    """The roles of the training videos ``names`` (in name order): of their
    permutation by ``seed``, the first ``rec``, the second ``val``, the rest
    ``train`` (in the permutation's order)."""
    order = [sorted(names)[i] for i in torch.randperm(len(names),
             generator=torch.Generator().manual_seed(seed)).tolist()]  # fmt: skip
    return {"rec": (order[0],), "val": (order[1],), "train": tuple(order[2:])}


@dataclass(frozen=True)
class SetLayout:
    """A set's stacks, in order: each one's ``video`` (index into the
    set's videos) and first frame ``frame`` ``(N,)``, each one's ``block``
    ``(N,)`` (numbered over the set), the ``centres`` ``(C,)`` (indices
    into the stacks) and their ``blocks`` ``(C,)``."""

    video: Tensor
    frame: Tensor
    block: Tensor
    centres: Tensor
    blocks: Tensor
    n_blocks: int

    def triples(self, w: Tensor) -> Triples:
        return Triples(w, self.centres, self.blocks, self.n_blocks)


def layout_of(lengths: Sequence[int], span: int, block: int, max_lag: int) -> SetLayout:
    """The layout of videos of ``lengths`` frames (module docstring): of each
    whole block its ``block - span`` stacks, the centres $u$ of $u - n \\ge$
    the block's first stack and $u + n$'s last frame in the block, $n$ =
    ``max_lag``."""
    per = block - span
    video, frame, blk, cen = [], [], [], []
    nb, n = 0, 0
    for v, length in enumerate(lengths):
        for b in range(length // block):
            j = torch.arange(per)
            video.append(torch.full((per,), v))
            frame.append(b * block + j)
            blk.append(torch.full((per,), nb))
            cen.append(n + j[max_lag : per - max_lag])
            nb += 1
            n += per
    cat = lambda xs: torch.cat(xs) if xs else torch.zeros(0, dtype=torch.long)  # noqa: E731
    centres = cat(cen)
    block_of = cat(blk)
    return SetLayout(cat(video), cat(frame), block_of, centres, block_of[centres], nb)


@dataclass(frozen=True)
class VideoSets:
    """The training videos' sets: ``roles`` (``split_videos``), the videos'
    ``lengths`` by name, the stacks' ``span`` and the ``layouts`` by role,
    and their ``key``."""

    roles: Mapping[str, tuple[str, ...]]
    lengths: Mapping[str, int]
    span: int
    layouts: Mapping[str, SetLayout]
    key: str


def video_sets(names: Sequence[str], lengths: Sequence[int], span: int,
               conf: DynConf) -> VideoSets:  # fmt: skip
    """The sets of the training videos ``names`` of ``lengths`` frames."""
    roles = split_videos(names, conf.split_seed)
    by = dict(zip(names, lengths, strict=True))
    layouts = {r: layout_of([by[v] for v in roles[r]], span, conf.block, conf.max_lag)
               for r in ROLES}  # fmt: skip
    ident = {"roles": roles, "lengths": by, "span": span, "block": conf.block,
             "max_lag": conf.max_lag, "split_seed": conf.split_seed}  # fmt: skip
    key = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:16]
    return VideoSets(roles, by, span, layouts, key)


def zurich_triples(w: Tensor, ws: WidthSets) -> Triples:
    """Zurich's triples: the common centres, their blocks and the shifted
    blocks."""
    c = ws.common
    return Triples(
        w, c, ws.blocks[c], ws.n_blocks, ws.blocks_shift[c], ws.n_blocks_shift
    )


@torch.no_grad()
def encode_layout(fn: Places, root: Path, names: Sequence[str], layout: SetLayout,
                  block: int, frames: int, stride: int,
                  dev: torch.device) -> dict[str, Tensor]:  # fmt: skip
    """The places of ``fn`` of the layout's stacks (``names`` its videos, in
    the layout's order; each video's whole blocks), a block's frames read at
    a time."""
    offsets = torch.arange(frames) * stride
    per = block - (frames - 1) * stride
    got: dict[str, list[Tensor]] = {}
    for v, name in enumerate(names):
        reader = U8Reader(root / name)
        for b in range(int((layout.video == v).sum()) // per):
            clip = reader[b * block : (b + 1) * block]
            places = encode_places(fn, lambda idx, clip=clip: clip[idx[:, None] + offsets],
                                   per, dev)  # fmt: skip
            for k, x in places.items():
                got.setdefault(k, []).append(x)
    return {k: torch.cat(v) for k, v in got.items()}


# -- the rows of a code -------------------------------------------------------------


def _sums_row(name: str, num: Mapping[str, Tensor], den: Mapping[str, Tensor]) -> Row:
    """The sums of a set's row: blocks' of ``BLOCKED``, else totals."""
    both = dict(num) | dict(den)
    if name in BLOCKED:
        return {k: v.tolist() for k, v in both.items()}
    return {
        k.removesuffix("_b"): float(v.sum())
        for k, v in both.items()
        if not k.endswith("_s")
    }


def spectrum(ref: Tensor, wh: Whitener) -> Row:
    """The reference's spectrum (module docstring) beside ``Whitener.spec``."""
    x = ref.double() - ref.double().mean(0)
    lam = torch.linalg.eigvalsh(x.T @ x / len(x)).clamp(min=0)
    pr = float(lam.sum() ** 2 / lam.square().sum()) if float(lam.sum()) > 0 else None
    raised = int((wh.values < wh.eps).sum())
    return {"part": "spec", "pr": pr, "raised": raised} | wh.spec()


def code_rows(
    sets: Mapping[str, Triples],
    conf: DynConf,
    dev: torch.device,
    *,
    full: bool = True,
    mlp: bool = True,
) -> list[Row]:
    """The rows of a whitened code's ``sets`` (``train``, ``val``, ``rec``,
    ``zurich``; module docstring): ``full`` the linear at every lag up to
    ``max_lag`` (else the grid's), ``mlp`` the residual MLP."""
    train, val = sets["train"].to(dev), sets["val"].to(dev)
    on = {name: s.to(dev) for name, s in sets.items()}
    mean = train.w.mean(0)
    rows: list[Row] = []
    lags = range(1, conf.max_lag + 1) if full else conf.grid
    fits: dict[int, LinFit] = {}
    den_of: dict[tuple[str, int], dict[str, Tensor]] = {}
    for n in lags:
        f = fit_linear(train, val, n, scales=conf.scales)
        if n in conf.grid:
            fits[n] = f
        for name, s in on.items():
            den_of[name, n] = dens(s, n, mean)
            pred, y = f.predict(s)
            row = {"part": "dyn", "set": name, "family": "lin", "lag": n, "seed": None,
                   "step": None, "alpha": f.alpha, "scale": f.scale}  # fmt: skip
            row |= _sums_row(name, num_sums(pred - y, s), den_of[name, n])
            if name == "train":
                row |= {"train_min": f.train_min, "alpha_min": f.alpha_min}
            if name == "val":
                row["val_scales"] = list(f.val)
            if name == "zurich":
                row["dir_num"] = (pred - y).square().sum(0).tolist()
            rows.append(row)
    for n in conf.grid:
        f1 = fit_linear(train, val, n, two=False, scales=conf.scales)
        for name, s in on.items():
            d = den_of.get((name, n)) or dens(s, n, mean)
            pred, y = f1.predict(s)
            rows.append({"part": "dyn", "set": name, "family": "lin1", "lag": n, "seed": None,
                         "step": None, "alpha": f1.alpha, "scale": f1.scale}
                        | _sums_row(name, num_sums(pred - y, s), d))  # fmt: skip
            _, _, fut = s.at(n)
            for fam, p in record_preds(train, s, n, mean).items():
                rows.append({"part": "dyn", "set": name, "family": fam, "lag": n, "seed": None,
                             "step": None} | _sums_row(name, num_sums(p - fut, s), d))  # fmt: skip
    if mlp:
        missing = [n for n in conf.grid if n not in fits]
        for n in missing:
            fits[n] = fit_linear(train, val, n, scales=conf.scales)
        run = train_mlp({n: fits[n] for n in conf.grid}, train, val, on, conf.mlp, dev)
        for i, n in enumerate(run.lags):
            step = run.times[int(run.chosen[i])]
            rows.append({"part": "mlp_val", "lag": n, "seeds": list(run.seeds),
                         "times": list(run.times), "val": run.val[i].tolist(),
                         "step": step})  # fmt: skip
            for j, seed in enumerate(run.seeds):
                for name, s in on.items():
                    d = den_of.get((name, n)) or dens(s, n, mean)
                    num = {k: v[i, j] for k, v in run.sums[name].items()}
                    rows.append({"part": "dyn", "set": name, "family": "mlp", "lag": n,
                                 "seed": seed, "step": step} | _sums_row(name, num, d))  # fmt: skip
    return rows


def white_sets(raw: Mapping[str, Tensor], ref: Tensor, ws: WidthSets, vs: VideoSets,
               gconf: GeomConf) -> tuple[Whitener, dict[str, Triples]]:  # fmt: skip
    """The whitening of the reference ``ref`` and the sets' triples of the
    raw codes ``raw`` (by ``train``, ``val``, ``rec``, ``zurich``)."""
    wh = Whitener.fit(ref, gconf.k, c_eps=gconf.c_eps)
    sets = {r: vs.layouts[r].triples(wh.white(raw[r])) for r in ROLES}
    sets["zurich"] = zurich_triples(wh.white(raw["zurich"]), ws)
    return wh, sets


def clock_times(vs: VideoSets, ws: WidthSets) -> dict[str, tuple[Tensor, Tensor, int]]:
    """Each set's stacks' time (the first frame in its video), block and
    count of blocks."""
    out = {r: (vs.layouts[r].frame, vs.layouts[r].block, vs.layouts[r].n_blocks)
           for r in ROLES}  # fmt: skip
    out["zurich"] = (torch.arange(ws.n), ws.blocks, ws.n_blocks)
    return out


def clock_code(sets: Mapping[str, Triples], times: Mapping[str, tuple[Tensor, Tensor, int]],
               conf: DynConf) -> dict[str, Triples]:  # fmt: skip
    r"""``clock`` (``plans/dyn.md``'s V-p2): of ``pca``'s whitened ``sets``,
    the ``clock_dirs`` directions of the largest $s_i$ at ``clock_lag`` on
    the training set ($s_i = \overline{(w_{t+n,i} - w_{t,i})^2} / (2
    \mathrm{Var}_i)$ of its centres; the reference is no series, so V-h2's
    choice on the training videos) replaced, in pairs of decreasing speed, by
    the planes $\sqrt2 (\cos, \sin)(2\pi t / T + \phi)$ of the stack's time
    $t$, $T$ = ``clock_period``, the phases drawn by ``clock_seed`` for each
    block of the sets in the order ``train``, ``val``, ``rec``,
    ``zurich``."""
    tr = sets["train"]
    c, lag = tr.centres, conf.clock_lag
    s = (tr.w[c + lag] - tr.w[c]).square().mean(0) / (2 * tr.w.var(0, unbiased=False))
    dirs = torch.argsort(s, descending=True, stable=True)[: conf.clock_dirs]
    gen = torch.Generator().manual_seed(conf.clock_seed)
    out = {}
    for name in ("train", "val", "rec", "zurich"):
        t, blk, nb = times[name]
        phi = 2 * math.pi * torch.rand(nb, len(dirs) // 2, generator=gen, dtype=_F64)
        theta = 2 * math.pi * t.double()[:, None] / conf.clock_period + phi[blk]
        w = sets[name].w.clone()
        w[:, dirs[0::2]] = math.sqrt(2) * theta.cos()
        w[:, dirs[1::2]] = math.sqrt(2) * theta.sin()
        x = sets[name]
        out[name] = Triples(
            w, x.centres, x.blocks, x.n_blocks, x.blocks_s, x.n_blocks_s
        )
    return out


# -- the jobs on real codes ---------------------------------------------------------


def _video_sets_of(conf: Any, root: Path, dconf: DynConf) -> VideoSets:
    t = conf.data.tours
    names = [v for v in videos(root) if v != t.held_out]
    lengths = [len(U8Reader(root / v)) for v in names]
    return video_sets(names, lengths, (t.frames - 1) * t.stride, dconf)


def _raw_sets(fn: Places, conf: Any, root: Path, vs: VideoSets, zurich: Any, ref: Tensor,
              n: int, block: int, dev: torch.device) -> dict[str, dict[str, Tensor]]:  # fmt: skip
    """The raw places of ``fn`` by place, then by set (``ROLES``, ``zurich``,
    ``ref``)."""
    t = conf.data.tours
    by_set = {r: encode_layout(fn, root, vs.roles[r], vs.layouts[r], block, t.frames,
                               t.stride, dev) for r in ROLES}  # fmt: skip
    zr = places_of(fn, zurich, ref, n, dev)
    out: dict[str, dict[str, Tensor]] = {}
    for place, (ev, rf) in zr.items():
        out[place] = {r: by_set[r][place] for r in ROLES} | {"zurich": ev, "ref": rf}
    return out


def _labels_rows(rows: Sequence[Row], labels: Mapping[str, Any]) -> list[Row]:
    return [dict(labels) | r for r in rows]


def _place_rows(raw: Mapping[str, Tensor], ws: WidthSets, vs: VideoSets, gconf: GeomConf,
                conf: DynConf, dev: torch.device, labels: Mapping[str, Any],
                pre: Callable[[Tensor], Tensor] | None = None) -> list[Row]:  # fmt: skip
    """The rows of one place's raw codes (``pre`` a map fit on the reference
    applied to every set first)."""
    raw = dict(raw) if pre is None else {k: pre(v) for k, v in raw.items()}
    wh, sets = white_sets(raw, raw["ref"], ws, vs, gconf)
    lab = dict(labels) | {
        "k": gconf.k,
        "eps": gconf.c_eps,
        "key": ws.key,
        "train_key": vs.key,
    }
    spec = spectrum(raw["ref"], wh)
    if not wh.scoreable:
        return [lab | spec | {"part": "unmeasurable", "why": wh.why}]
    return _labels_rows([spec] + code_rows(sets, conf, dev), lab)


def evaluate_yard(
    conf: Any,
    sets_dir: Path,
    out: Path,
    *,
    inits: Mapping[str, Path],
    init_v16: Path,
    root: Path | None = None,
    codes: Collection[str] | None = None,
    dconf: DynConf = DynConf(),
    gconf: GeomConf = GeomConf(),
    wconf: WidthConf = WidthConf(),
    fconf: FillConf = FillConf(),
) -> Path:
    """V-p2's rows (``out / "yard/rows/<code>.json"``) of ``YARD_CODES``
    (``codes`` only those; each code's file the same to the byte whichever
    run beside it): ``pca``, ``rot`` (``pca``'s times V-h4b's orthogonal
    matrix), ``clock`` (``clock_code``), ``color``, ``mean8``, the step 0 of
    ``inits`` and V16's VAE of $t = 0$. Returns ``out``.

    Raises:
        ValueError: A code is not of ``YARD_CODES`` or a run is not given.
    """
    bad = [c for c in codes or () if c not in YARD_CODES]
    if bad:
        raise ValueError(f"no yardsticks {bad} (of {list(YARD_CODES)})")
    todo = YARD_CODES if codes is None else tuple(c for c in YARD_CODES if c in codes)
    missing = [n for n in YARD_INITS if n in todo and n not in inits]
    if missing:
        raise ValueError(f"the yard needs the runs {missing}")
    determinism = configure_determinism(conf.device)
    dev = torch.device(conf.device)
    ws, zurich, ref, root = sets_and_ref(
        conf, sets_dir, root, gconf, wconf, fconf, None, None
    )
    vs = _video_sets_of(conf, root, dconf)
    yard = out / "yard"
    sources = {k: str(v) for k, v in inits.items()} | {"init_v16": str(init_v16)}
    _start(yard, {"dyn": _plain(dconf), "geom": asdict(gconf), "key": ws.key,
                  "train_key": vs.key, "roles": dict(vs.roles), "sources": sources},
           determinism)  # fmt: skip

    def emit(code: str, rows: list[Row]) -> None:
        _write(yard / "rows" / f"{code}.json", rows, {})

    def raw_of(fn: Places) -> dict[str, dict[str, Tensor]]:
        return _raw_sets(fn, conf, root, vs, zurich, ref, ws.n, dconf.block, dev)

    if {"pca", "rot", "clock"} & set(todo):
        gen = torch.Generator().manual_seed(conf.seed + PCA_SEED)
        pca_code = PCACode.fit(pixels, ref, gconf.pca_dim, gen=gen, device=conf.device)
        raw = raw_of(_feature_code(pca_code.mu))["own"]
        lab = {"code": "pca", "place": "own"}
        if "pca" in todo:
            emit("pca", _place_rows(raw, ws, vs, gconf, dconf, dev, lab))
        _, sets = white_sets(raw, raw["ref"], ws, vs, gconf)
        base = {"place": "own", "k": gconf.k, "eps": gconf.c_eps, "key": ws.key,
                "train_key": vs.key}  # fmt: skip
        if "rot" in todo:
            q = orthogonal(gconf.k, dconf.rot_seed)
            rot = {n: Triples(s.w @ q, s.centres, s.blocks, s.n_blocks, s.blocks_s, s.n_blocks_s)
                   for n, s in sets.items()}  # fmt: skip
            emit(
                "rot", _labels_rows(code_rows(rot, dconf, dev), base | {"code": "rot"})
            )
        if "clock" in todo:
            clock = clock_code(sets, clock_times(vs, ws), dconf)
            emit(
                "clock",
                _labels_rows(code_rows(clock, dconf, dev), base | {"code": "clock"}),
            )
    for name, fn in (("color", _feature_code(ColorCode(16).mu)),):
        if name in todo:
            emit(name, _place_rows(raw_of(fn)["own"], ws, vs, gconf, dconf, dev,
                                   {"code": name, "place": "own"}))  # fmt: skip
    if "mean8" in todo:
        gen = torch.Generator().manual_seed(conf.seed + PCA_SEED)
        m8 = PCACode.fit(_mean8, ref, gconf.pca_dim, gen=gen, device=conf.device)
        emit("mean8", _place_rows(raw_of(_feature_code(m8.mu))["own"], ws, vs, gconf, dconf,
                                  dev, {"code": "mean8", "place": "own"}))  # fmt: skip
    for name in YARD_INITS:
        if name not in todo:
            continue
        mods, own, path = _init_of(inits[name], conf.device)
        raws = raw_of(JepaPlaces(mods).places)
        rows: list[Row] = []
        for place, raw in raws.items():
            pre = None
            if place == "own" and own == "zpca":
                pre = _zpca_map(
                    raw["ref"], _conf_of(inits[name], None).seed, gconf.pca_dim
                )
            rows += _place_rows(raw, ws, vs, gconf, dconf, dev,
                                {"code": name, "place": place, "source": str(path),
                                 "t": float(path.stem)}, pre)  # fmt: skip
        emit(name, rows)
    if "init_v16" in todo:
        first = _first_vae(init_v16)
        vae = vae_at(first, conf.device)
        assert isinstance(vae, VAEModules)
        rows = []
        for place, raw in raw_of(VaeTokenCode(vae).places).items():
            rows += _place_rows(raw, ws, vs, gconf, dconf, dev,
                                {"code": "init_v16", "place": place, "source": str(first),
                                 "t": float(first.stem)})  # fmt: skip
        emit("init_v16", rows)
    return out


def _plain(dconf: DynConf) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(dconf)))


def _apply(fn: Places, conf: Any, root: Path, vs: VideoSets, zurich: Any, ref: Tensor,
           ws: WidthSets, gconf: GeomConf, dconf: DynConf, dev: torch.device,
           labels: Mapping[str, Any], zpca_seed: int | None = None) -> list[Row]:  # fmt: skip
    rows: list[Row] = []
    for place, raw in _raw_sets(
        fn, conf, root, vs, zurich, ref, ws.n, dconf.block, dev
    ).items():
        pre = None
        if place == "own" and zpca_seed is not None:
            pre = _zpca_map(raw["ref"], zpca_seed, gconf.pca_dim)
        rows += _place_rows(
            raw, ws, vs, gconf, dconf, dev, dict(labels) | {"place": place}, pre
        )
    return rows


def evaluate_run(
    run_dir: Path,
    *,
    steps: Sequence[int],
    sets_dir: Path,
    root: Path | None = None,
    device: str | None = None,
    dconf: DynConf = DynConf(),
    gconf: GeomConf = GeomConf(),
    wconf: WidthConf = WidthConf(),
    fconf: FillConf = FillConf(),
) -> Path:
    """The rows of the JEPA run's checkpoints of ``steps`` into ``run_dir /
    OUT / "rows/<step>.json"`` (a step whose file is there is skipped).
    Returns the directory.

    Raises:
        ValueError: A step has no checkpoint.
    """
    conf = _conf_of(run_dir, device)
    ckpts = run_ckpts(run_dir)
    missing = [s for s in steps if s not in ckpts]
    if missing:
        raise ValueError(f"no checkpoint of steps {missing} (of {list(ckpts)})")
    determinism = configure_determinism(conf.device)
    dev = torch.device(conf.device)
    ws, zurich, ref, root = sets_and_ref(
        conf, sets_dir, root, gconf, wconf, fconf, None, None
    )
    vs = _video_sets_of(conf, root, dconf)
    mods, own = _jepa_of(run_dir, conf.device)
    out = run_dir / OUT
    _start(out, {"dyn": _plain(dconf), "geom": asdict(gconf), "key": ws.key,
                 "train_key": vs.key, "run": run_dir.name}, determinism)  # fmt: skip
    for step in steps:
        path = out / "rows" / f"{step:06d}.json"
        if path.exists():
            continue
        ckpt.load_eval(ckpts[step], mods.modules())
        rows = _apply(JepaPlaces(mods).places, conf, root, vs, zurich, ref, ws, gconf, dconf,
                      dev, {"code": run_dir.name, "step": step},
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
    dconf: DynConf = DynConf(),
    gconf: GeomConf = GeomConf(),
    wconf: WidthConf = WidthConf(),
    fconf: FillConf = FillConf(),
) -> Path:
    """The rows of the VAE checkpoint ``path`` (as ``width_eval``'s) into
    ``out / "vae/rows/<name>.json"``. Returns the file."""
    conf = _conf_of(path.parent.parent, device)
    view, _ = stream_view(conf)
    determinism = configure_determinism(conf.device)
    dev = torch.device(conf.device)
    ws, zurich, ref, root = sets_and_ref(
        view, sets_dir, root, gconf, wconf, fconf, None, None
    )
    vs = _video_sets_of(view, root, dconf)
    mods = build_vae(conf, seed=conf.seed)
    ckpt.load_eval(path, mods.modules())
    vae_out = out / "vae"
    _start(vae_out, {"dyn": _plain(dconf), "geom": asdict(gconf), "key": ws.key,
                     "train_key": vs.key}, determinism)  # fmt: skip
    rows = _apply(VaeTokenCode(mods).places, view, root, vs, zurich, ref, ws, gconf, dconf, dev,
                  {"code": name, "source": str(path)})  # fmt: skip
    target = vae_out / "rows" / f"{name}.json"
    _write(target, rows, {"t": float(path.stem)})
    return target


# -- the checks V-p1 ----------------------------------------------------------------


@dataclass(frozen=True)
class CheckConf:
    r"""The settings of V-p1 (``plans/dyn.md``): series in steps of 0.1 s,
    ``d`` columns of unit variance (the population's whitening is the
    identity), the training series of ``train_n`` steps cut into ``videos``
    equal videos and split as the real ones (``split_videos``), the
    evaluation of ``eval_n`` stacks of ``span`` (``width_eval``'s synthetic
    sets), ``long_parts`` of them on the long realization."""

    train_n: int = 479350
    videos: int = 9
    eval_n: int = 38989
    span: int = 7
    d: int = 64
    long_seed: int = 5000
    long_parts: int = 10
    period: float = 123.4  # S1, S3, S8, S10: 12.34 s
    cheb_period: float = 617.0  # S2's theta: 61.7 s
    lam: float = 0.01  # S3: 0.1 / s
    ou_tau: float = 32.0  # S4: 3.2 s
    box_widths: tuple[int, ...] = (32, 128)  # S5: 3.2 s, 12.8 s
    white_cols: int = 32  # S6
    chain_periods: tuple[float, ...] = (63.4, 123.4, 243.4)  # S7
    radii: tuple[float, ...] = (0.6, 0.8, 1.0, 1.2, 1.4)  # S8
    rot_seed: int = 5900  # S9
    nus: tuple[float, ...] = (1 / 32, 1 / 8)  # S10
    cover_seeds: tuple[int, int] = (5101, 5301)
    null_seeds: tuple[int, int] = (5301, 5501)
    cover_min: float = 0.90
    null_max: int = 13
    se_mult: float = 4.0
    slack: float = 1e-8
    rel: float = 1e-10
    floor: float = 1e-9
    bound_rel: float = 1e-9  # the rounding of "the MLP's validation <= the linear's"
    boot: int = 2000
    boot_seed: int = 101

    @property
    def r(self) -> float:
        return math.exp(-1 / self.ou_tau)


# the synthetic codes by family: a family's codes come of one draw
FAMILIES: dict[str, tuple[str, ...]] = {
    "ou": ("S4", "S5a", "S5b", "S6", "S9-S4"),
    "rot": ("S1", "S10a", "S10b"),
    "cheb": ("S2",),
    "cou": ("S3",),
    "chain": ("S7",),
    "amp": ("S8", "S9-S8"),
    "ou2": ("S4b",),
}
LONG_CODES = ("S1", "S2", "S3", "S4", "S5a", "S5b", "S6", "S7", "S8", "S9-S4", "S9-S8",
              "S10a", "S10b")  # fmt: skip
MLP_CODES = ("S4", "S9-S4", "S8", "S9-S8")
V_P1A = ("S1", "S2", "S3", "S4", "S5a", "S5b", "S6", "S7", "S10a", "S10b")
E_PAIRS = (("S10a", "S10b"), ("S6", "S4"), ("S5a", "S4"), ("S5b", "S4"))


def _family_of(code: str) -> str:
    return next(f for f, cs in FAMILIES.items() if code in cs)


def synth_codes(
    family: str, blocks: Tensor, cc: CheckConf, gen: torch.Generator
) -> dict[str, Tensor]:
    """The family's codes ``(n, d)`` of one draw by ``gen`` over the rows of
    ``blocks`` ``(n,)`` (the 128 s blocks of the rotations' phases)."""
    n, d = len(blocks), cc.d
    if family in ("ou", "ou2"):
        pad = max(cc.box_widths) - 1
        x = ou(n + pad, d, cc.r, gen)
        if family == "ou2":
            return {"S4b": x[pad:]}
        out = {"S4": x[pad:]}
        for w, name in zip(cc.box_widths, ("S5a", "S5b"), strict=True):
            sd = math.sqrt(float(box_ar1_acf(cc.r, w, torch.tensor([0]))[0]))
            out[name] = box_mean(x[pad - w + 1 :], w) / sd
        s6 = out["S4"].clone()
        s6[:, d - cc.white_cols :] = white(n, cc.white_cols, gen)
        out["S6"] = s6
        out["S9-S4"] = out["S4"] @ orthogonal(d, cc.rot_seed).T
        return out
    if family == "rot":
        s1 = block_rotation(blocks, d, (cc.period,), gen)
        out = {"S1": s1}
        for nu, name in zip(cc.nus, ("S10a", "S10b"), strict=True):
            out[name] = (s1 + math.sqrt(nu) * white(n, d, gen)) / math.sqrt(1 + nu)
        return out
    if family == "cheb":
        theta0 = 2 * math.pi * float(torch.rand(1, generator=gen, dtype=_F64))
        theta = theta0 + 2 * math.pi / cc.cheb_period * torch.arange(n, dtype=_F64)
        return {"S2": chebyshev(theta, torch.arange(1, d + 1))}
    if family == "cou":
        return {"S3": complex_ou(n, d, cc.lam, cc.period, gen)}
    if family == "chain":
        return {"S7": block_rotation(blocks, d, cc.chain_periods, gen)}
    s8 = block_rotation(blocks, d, (cc.period,), gen, cc.radii)
    return {"S8": s8, "S9-S8": s8 @ orthogonal(d, cc.rot_seed).T}


def synth_gamma(code: str, lag: int, weights: Tensor, cc: CheckConf) -> Tensor:
    r"""$\Gamma(\ell)$ ``(d, d)`` of the code (``weights`` the share of the
    set's centres by block kind: by ``b % J`` of the chained periods for
    ``S7``, of the radii for ``S8``; unused otherwise)."""
    d, r = cc.d, cc.r
    eye = torch.eye(d, dtype=_F64)
    om = torch.tensor([2 * math.pi / cc.period])
    if code in ("S4", "S4b"):
        return eye * r ** abs(lag)
    if code in ("S5a", "S5b"):
        w = cc.box_widths[0 if code == "S5a" else 1]
        acf = box_ar1_acf(r, w, torch.tensor([lag, 0]))
        return eye * float(acf[0] / acf[1])
    if code == "S6":
        diag = torch.full((d,), r ** abs(lag), dtype=_F64)
        diag[d - cc.white_cols :] = 1.0 if lag == 0 else 0.0
        return torch.diag(diag)
    if code == "S1":
        return rotation_gamma(lag, d, om, torch.ones(1))
    if code in ("S10a", "S10b"):
        nu = cc.nus[0 if code == "S10a" else 1]
        return eye if lag == 0 else rotation_gamma(lag, d, om, torch.ones(1)) / (1 + nu)
    if code == "S2":
        j = torch.arange(1, d + 1, dtype=_F64)
        return torch.diag(torch.cos(j * 2 * math.pi / cc.cheb_period * lag))
    if code == "S3":
        return complex_ou_gamma(lag, d, cc.lam, cc.period)
    if code == "S7":
        oms = torch.tensor([2 * math.pi / p for p in cc.chain_periods])
        return rotation_gamma(lag, d, oms, weights)
    if code == "S8":
        rs = torch.tensor(cc.radii, dtype=_F64)
        bar2 = float(rs.square().mean())
        return rotation_gamma(
            lag, d, 2 * math.pi / cc.period * rs, weights, rs.square() / bar2
        )
    q = orthogonal(d, cc.rot_seed)
    return q @ synth_gamma(code.removeprefix("S9-"), lag, weights, cc) @ q.T


def _kinds(code: str, cc: CheckConf) -> int:
    base = code.removeprefix("S9-")
    return {"S7": len(cc.chain_periods), "S8": len(cc.radii)}.get(base, 1)


@dataclass(frozen=True)
class _Synth:
    """A realization's sets of one family: by code, the ``sets`` (``train``,
    ``val``, ``rec``, ``eval``) and each set's generation blocks of its
    centres (for the mixtures' weights)."""

    sets: Mapping[str, Mapping[str, Triples]]
    kinds: Mapping[str, Tensor]


def _realization(family: str, seed: int, parts: int, cc: CheckConf) -> _Synth:
    """The family's codes on a realization of ``seed``: the training series
    (one draw over the videos, their generation blocks each video's own 128 s
    blocks numbered over the series) and ``parts`` evaluation series (their
    blocks side by side)."""
    sizes = [len(p) for p in torch.arange(cc.train_n).tensor_split(cc.videos)]
    names = [f"v{i}" for i in range(cc.videos)]
    dconf = DynConf()
    vs = video_sets(names, sizes, cc.span, dconf)
    starts = [0]
    for s in sizes[:-1]:
        starts.append(starts[-1] + s)
    gblocks, off = [], 0
    for s in sizes:
        b = torch.arange(s) // dconf.block
        gblocks.append(b + off)
        off += int(b.max()) + 1
    gb = torch.cat(gblocks)
    gen = torch.Generator().manual_seed(seed * 100 + list(FAMILIES).index(family) * 10)
    series = synth_codes(family, gb, cc, gen)
    out: dict[str, dict[str, Triples]] = {c: {} for c in series}
    kinds: dict[str, Tensor] = {}
    for role in ROLES:
        lay = vs.layouts[role]
        vid = torch.tensor([names.index(v) for v in vs.roles[role]])[lay.video]
        rows = torch.tensor(starts)[vid] + lay.frame
        for c, x in series.items():
            out[c][role] = lay.triples(x[rows])
        kinds[role] = gb[rows[lay.centres]]
    ws = synthetic_sets(cc.eval_n, cc.span, dconf.block, dconf.max_lag)
    evs: dict[str, list[Tensor]] = {c: [] for c in series}
    for p in range(parts):
        g = torch.Generator().manual_seed(
            seed * 100 + list(FAMILIES).index(family) * 10 + 1 + p
        )
        for c, x in synth_codes(family, ws.blocks, cc, g).items():
            evs[c].append(x)
    cen, blk = ws.common, ws.blocks[ws.common]
    n, nb = ws.n, ws.n_blocks
    for c in series:
        w = torch.cat(evs[c])
        centres = torch.cat([cen + p * n for p in range(parts)])
        blocks = torch.cat([blk + p * nb for p in range(parts)])
        out[c]["eval"] = Triples(w, centres, blocks, parts * nb)
    kinds["eval"] = torch.cat([blk for _ in range(parts)])
    return _Synth(out, kinds)


def _weights(kinds: Tensor, n: int) -> Tensor:
    return torch.bincount(kinds % n, minlength=n).double() / len(kinds)


def _lin_unit(
    code: str, syn: _Synth, cc: CheckConf, *, population: bool
) -> dict[str, Any]:
    """The linear's eval sums of the grid, and with ``population`` its
    population ratios (V-p1a), the oracle's and the identities' terms."""
    s = syn.sets[code]
    train, val, ev = s["train"], s["val"], s["eval"]
    mean = train.w.mean(0)
    kinds = _kinds(code, cc)
    w_ev, w_tr = _weights(syn.kinds["eval"], kinds), _weights(syn.kinds["train"], kinds)
    out: dict[str, Any] = {"lags": list(GRID), "num_b": [], "copy_b": [], "clim_b": [],
                           "vel_b": [], "scale": []}  # fmt: skip
    if population:
        out |= {"pop_copy": [], "pop_clim": [], "star_copy": [], "star_clim": [],
                "trainbest_copy": [], "trainbest_clim": [], "mid": [], "vd": []}  # fmt: skip
    for n in GRID:
        f = fit_linear(train, val, n)
        pred, y = f.predict(ev)
        d = dens(ev, n, mean)
        out["num_b"].append(num_sums(pred - y, ev)["num_b"].tolist())
        out["copy_b"].append(d["copy_b"].tolist())
        out["clim_b"].append(d["clim_b"].tolist())
        now, past, fut = ev.at(n)
        out["vel_b"].append(num_sums(2 * now - past - fut, ev)["num_b"].tolist())
        out["scale"].append(f.scale)
        if not population:
            continue
        g = [synth_gamma(code, lag, w_ev, cc) for lag in (0, n, 2 * n)]
        e_fixed = fixed_error(f.a, f.b, *g)
        copy, clim = population_dens(g[0], g[1], mean)
        e_star = best_linear(*g)
        gt = [synth_gamma(code, lag, w_tr, cc) for lag in (0, n, 2 * n)]
        k = torch.cat([gt[1], gt[2]], 1)
        sig = torch.cat(
            [torch.cat([gt[0], gt[1]], 1), torch.cat([gt[1].T, gt[0]], 1)], 0
        )
        m_tr = k @ torch.linalg.pinv((sig + sig.T) / 2, hermitian=True)
        e_tb = fixed_error(m_tr, torch.zeros(cc.d, dtype=_F64), *g)
        out["pop_copy"].append(e_fixed / copy)
        out["pop_clim"].append(e_fixed / clim)
        out["star_copy"].append(e_star / copy)
        out["star_clim"].append(e_star / clim)
        out["trainbest_copy"].append(e_tb / copy)
        out["trainbest_clim"].append(e_tb / clim)
        t = triple_terms(ev.w, ev.centres, n)
        out["mid"].append([float(t.num.sum()), float(t.den.sum()), float(t.a2.sum()),
                           float(t.b2.sum())])  # fmt: skip
        out["vd"].append(
            float((mean - fut).square().sum() / (now - fut).square().sum())
        )
    return out


def _mlp_unit(code: str, syn: _Synth, cc: CheckConf, mconf: MLPConf,
              dev: torch.device) -> dict[str, Any]:  # fmt: skip
    """The MLP of the grid on the long realization: the eval and val sums of
    each seed, the linear's, the chosen steps, and of every set (``train``,
    ``val``, ``rec``, ``eval``) the totals of the linear's and each seed's
    numerator and of the two denominators (``totals``, V-p1g's record)."""
    s = syn.sets[code]
    train, val = s["train"], s["val"]
    mean = train.w.mean(0)
    fits = {n: fit_linear(train, val, n) for n in GRID}
    on = {name: s[name] for name in (*ROLES, "eval")}
    run = train_mlp(fits, train, val, on, mconf, dev)
    totals: dict[str, dict[str, list[Any]]] = {}
    for name, t in on.items():
        tot: dict[str, list[Any]] = {"lin": [], "copy": [], "clim": []}
        for n in GRID:
            pred, y = fits[n].predict(t)
            d = dens(t, n, mean)
            tot["lin"].append(float(num_sums(pred - y, t)["num_b"].sum()))
            tot["copy"].append(float(d["copy_b"].sum()))
            tot["clim"].append(float(d["clim_b"].sum()))
        tot["mlp"] = run.sums[name]["num_b"].sum(-1).tolist()  # (L, S)
        totals[name] = tot
    return {"lags": list(GRID), "seeds": list(run.seeds),
            "steps": [run.times[int(i)] for i in run.chosen],
            "num_b": run.sums["eval"]["num_b"].tolist(),  # (L, S, B)
            "val": totals["val"]["mlp"], "lin_val": totals["val"]["lin"],
            "totals": totals}  # fmt: skip


def check_units(cc: CheckConf = CheckConf()) -> tuple[str, ...]:
    """The units one process may run apart: ``long-<code>`` (the linear on
    the long realization), ``mlp-<code>`` (the MLP there, on a GPU),
    ``V-p1f`` and ``V-p1h`` (their realizations, split by ``--seeds``)."""
    del cc
    return (*(f"long-{c}" for c in LONG_CODES), *(f"mlp-{c}" for c in MLP_CODES), "V-p1f",
            "V-p1h")  # fmt: skip


def _pair_draw(a: str, b: str, seed: int, cc: CheckConf, *,
               population: bool) -> dict[str, Any]:  # fmt: skip
    """The two codes' linear eval sums on the realization of ``seed``, and
    with ``population`` the population ratios of the learned predictors
    (``_lin_unit``; V-p1f's target)."""
    fams = {_family_of(a), _family_of(b)}
    syns = {f: _realization(f, seed, 1, cc) for f in sorted(fams)}
    out: dict[str, Any] = {"seed": seed}
    for c in (a, b):
        out[c] = _lin_unit(c, syns[_family_of(c)], cc, population=population)
    return out


def check_parts(
    cc: CheckConf,
    units: Collection[str],
    seeds: tuple[int, int] | None = None,
    mconf: MLPConf = MLPConf(),
    dev: torch.device = torch.device("cpu"),
) -> list[Row]:
    """The rows of the ``units`` of one process, for ``merge_checks``:
    ``{"part": "unit", "unit", "value"}`` of each long unit and
    ``{"part": "draw", "unit", "value"}`` of each realization of V-p1f and
    V-p1h whose seed lies in ``seeds`` (``lo <= seed < hi``; None: all).

    Raises:
        ValueError: A unit is not one of ``check_units``.
    """
    known = check_units(cc)
    bad = [u for u in units if u not in known]
    if bad:
        raise ValueError(f"no check units {bad} (of {list(known)})")
    lo, hi = (-math.inf, math.inf) if seeds is None else seeds
    rows: list[Row] = []
    cache: dict[str, _Synth] = {}

    def long_of(code: str) -> _Synth:
        fam = _family_of(code)
        if fam not in cache:
            cache[fam] = _realization(fam, cc.long_seed, cc.long_parts, cc)
        return cache[fam]

    for u in known:
        if u not in units:
            continue
        if u.startswith("long-"):
            code = u.removeprefix("long-")
            rows.append({"part": "unit", "unit": u,
                         "value": _lin_unit(code, long_of(code), cc, population=True)})  # fmt: skip
        elif u.startswith("mlp-"):
            code = u.removeprefix("mlp-")
            rows.append({"part": "unit", "unit": u,
                         "value": _mlp_unit(code, long_of(code), cc, mconf, dev)})  # fmt: skip
        elif u == "V-p1f":
            rows += [{"part": "draw", "unit": u,
                      "value": _pair_draw("S10a", "S10b", s, cc, population=True)}
                     for s in range(*cc.cover_seeds) if lo <= s < hi]  # fmt: skip
        else:
            rows += [{"part": "draw", "unit": u,
                      "value": _pair_draw("S4", "S4b", s, cc, population=False)}
                     for s in range(*cc.null_seeds) if lo <= s < hi]  # fmt: skip
    return rows


def _check(name: str, ok: bool, **values: Any) -> Row:
    return {"part": "check", "name": name, "pass": bool(ok)} | values


def _record(name: str, **values: Any) -> Row:
    return {"part": "record", "name": name} | values


def _t(x: Any) -> Tensor:
    return torch.tensor(x, dtype=_F64)


def _counts(n_blocks: int, cc: CheckConf) -> Tensor:
    return block_counts(n_blocks, cc.boot, torch.Generator().manual_seed(cc.boot_seed))


def _psi(a: Mapping[str, Any], b: Mapping[str, Any], den: str, cc: CheckConf,
         sigma: float = 0.0, z: float = 2.0) -> Interval:  # fmt: skip
    """The interval of $\\psi$ (``judge_dyn.psi_interval``) of two codes'
    sums of one realization (``den`` ``copy`` or ``clim``)."""
    return psi_interval(GRID, _t(a["num_b"]), _t(a[f"{den}_b"]), _t(b["num_b"]),
                        _t(b[f"{den}_b"]), cc.boot, cc.boot_seed, sigma=sigma, z=z)  # fmt: skip


def _oracle_psi(a: str, b: str, cc: CheckConf) -> tuple[float, float]:
    """The oracle's $(\\psi_{\\mathrm{copy}}, \\psi_{\\mathrm{clim}})$ of the
    best linear predictors of two stationary codes (no mixture)."""
    out = []
    for den in (0, 1):
        ml = []
        for code in (a, b):
            r = []
            for n in GRID:
                g = [synth_gamma(code, lag, torch.ones(1), cc) for lag in (0, n, 2 * n)]
                e = best_linear(*g)
                copy, clim = population_dens(g[0], g[1], torch.zeros(cc.d, dtype=_F64))
                r.append(e / (copy if den == 0 else clim))
            ml.append(float(mean_log(GRID, _t(r))))
        out.append(ml[0] - ml[1])
    return out[0], out[1]


def _vp1a(units: Mapping[str, Any], cc: CheckConf) -> list[Row]:
    rows = []
    for code in V_P1A:
        v = units[f"long-{code}"]
        num, ok_all, ok_floor = _t(v["num_b"]), True, True
        counts = _counts(num.shape[1], cc)
        worst = []
        for den in ("copy", "clim"):
            d = _t(v[f"{den}_b"])
            r_hat = num.sum(1) / d.sum(1)
            draws = (counts @ num.T) / (counts @ d.T)
            se = draws.std(0)
            pop = _t(v[f"pop_{den}"])
            gap = (r_hat - pop).abs()
            ok = bool((gap <= cc.se_mult * se + cc.slack).all())
            ok_all &= ok
            ok_floor &= bool((pop >= _t(v[f"star_{den}"]) - cc.floor).all())
            worst.append({"den": den, "r_hat": r_hat.tolist(), "pop": pop.tolist(),
                          "se": se.tolist(), "star": v[f"star_{den}"],
                          "trainbest": v[f"trainbest_{den}"]})  # fmt: skip
        rows.append(_check("V-p1a", ok_all, code=code, values=worst))
        rows.append(_check("V-p1a'", ok_floor, code=code,
                           shortfall={den: (_t(v[f"pop_{den}"]) - _t(v[f"star_{den}"])).tolist()
                                      for den in ("copy", "clim")}))  # fmt: skip
    return rows


def _close(a: Tensor, b: Tensor, rel: float) -> bool:
    return bool(((a - b).abs() <= rel * b.abs().clamp(min=1e-300)).all())


def _vp1b(units: Mapping[str, Any], cc: CheckConf) -> list[Row]:
    rows = []
    for code in LONG_CODES:
        v = units[f"long-{code}"]
        num, copy, clim = (
            _t(v["num_b"]).sum(1),
            _t(v["copy_b"]).sum(1),
            _t(v["clim_b"]).sum(1),
        )
        vd = _close((num / copy) / (num / clim), _t(v["vd"]), cc.rel)
        mid = _t(v["mid"])  # (L, 4): sum nu, sum delta, sum a2, sum b2
        r_vel = _t(v["vel_b"]).sum(1) / copy
        g = mid[:, 0] / mid[:, 1]
        ident = _close(r_vel, 2 * g * (mid[:, 2] + mid[:, 3]) / mid[:, 3], cc.rel)
        row = {"code": code, "vd": vd, "vel_identity": ident}
        ok = vd and ident
        if code == "S1":
            om = 2 * math.pi / cc.period
            want = _t([2 * (1 - math.cos(om * n)) for n in GRID])
            row["vel_closed"] = _close(r_vel, want, cc.rel)
            ok &= row["vel_closed"]
        if code.startswith("S9-"):
            o = units[f"long-{code.removeprefix('S9-')}"]
            same = _close(
                num / copy, _t(o["num_b"]).sum(1) / _t(o["copy_b"]).sum(1), cc.rel
            )
            row["invariant"] = same and v["scale"] == o["scale"]
            ok &= row["invariant"]
        rows.append(_check("V-p1b", ok, **row))
    return rows


def _mlp_values(v: Mapping[str, Any]) -> tuple[Tensor, list[Tensor]]:
    """The MLP's eval numerators: the seeds' mean ``(L, B)`` and each
    seed's."""
    nb = _t(v["num_b"])  # (L, S, B)
    return nb.mean(1), [nb[:, j] for j in range(nb.shape[1])]


def _ml(num: Tensor, den: Tensor) -> float:
    return float(mean_log(GRID, num.sum(1) / den.sum(1)))


def _sigma_rot(units: Mapping[str, Any], base: str) -> dict[str, float]:
    """Of each denominator apart, the standard deviation over the seeds of
    the MLP's $\\psi_s$ of ``S9-<base>`` against ``base``."""
    a, b = units[f"long-S9-{base}"], units[f"long-{base}"]
    _, ra = _mlp_values(units[f"mlp-S9-{base}"])
    _, rb = _mlp_values(units[f"mlp-{base}"])
    out = {}
    for den in ("copy", "clim"):
        da, db = _t(a[f"{den}_b"]), _t(b[f"{den}_b"])
        out[den] = stdev([_ml(x, da) - _ml(y, db) for x, y in zip(ra, rb, strict=True)])
    return out


def _mlp_vs_lin(units: Mapping[str, Any], code: str, cc: CheckConf,
                sigma_rot: Mapping[str, float]) -> dict[str, Any]:  # fmt: skip
    """MLP (a) against the linear (b) of one code: the intervals and the
    words, $\\sigma_{ab} = \\max(\\sigma_{\\mathrm{rot}}, s_{ab})$."""
    lin, mv = units[f"long-{code}"], units[f"mlp-{code}"]
    mean_b, seeds = _mlp_values(mv)
    out: dict[str, Any] = {}
    ivs = []
    for den in ("copy", "clim"):
        d = _t(lin[f"{den}_b"])
        s_ab = stdev([_ml(x, d) - _ml(_t(lin["num_b"]), d) for x in seeds])
        sigma = max(sigma_rot[den], s_ab)
        iv = psi_interval(GRID, mean_b, d, _t(lin["num_b"]), d, cc.boot, cc.boot_seed,
                          sigma=sigma)  # fmt: skip
        ivs.append(iv)
        out[den] = {
            "psi": iv.psi,
            "lo": iv.lo,
            "hi": iv.hi,
            "sigma": sigma,
            "s_ab": s_ab,
        }
    out["words"] = words(ivs[0], ivs[1])
    out["val_ok"] = all(
        sum(v) / len(v) <= lv * (1 + cc.bound_rel)
        for v, lv in zip(mv["val"], mv["lin_val"], strict=True)
    )
    out["steps"] = mv["steps"]
    return out


# the design memo's table (section 3.2) of the OU of tau = 3.2 s (32 steps), at
# Delta / tau = 0.5, 1, 2, 4, 8 (the grid's n = 16 .. 256): of the code and the
# denominator, the values; S5a is the box of W = tau, S5b of W = 4 tau
S5_TABLE: dict[str, dict[str, tuple[float, ...]]] = {
    "copy": {"S4": (0.803, 0.684, 0.568, 0.509, 0.500),
             "S5a": (0.656, 0.757, 0.600, 0.514, 0.500),
             "S5b": (0.433, 0.590, 0.672, 0.580, 0.501)},
    "clim": {"S4": (0.632, 0.865, 0.982, 1.000, 1.000),
             "S5b": (0.030, 0.140, 0.483, 0.974, 1.000)},
}  # fmt: skip
S5_LAGS = (16, 32, 64, 128, 256)


def _s5_table(units: Mapping[str, Any]) -> dict[str, Any]:
    """V-p1g: the table's values beside the oracle's best (``star_*``) and
    the long realization's linear (estimated) at its lags."""
    out: dict[str, Any] = {}
    for den, codes in S5_TABLE.items():
        for code, want in codes.items():
            v = units[f"long-{code}"]
            idx = [GRID.index(n) for n in S5_LAGS]
            r_hat = _t(v["num_b"]).sum(1) / _t(v[f"{den}_b"]).sum(1)
            out[f"{code}/{den}"] = {"table": list(want),
                                    "oracle": [v[f"star_{den}"][i] for i in idx],
                                    "estimate": [float(r_hat[i]) for i in idx]}  # fmt: skip
    return out


def _sets_values(mv: Mapping[str, Any]) -> dict[str, Any]:
    """V-p1g: of each set, the linear's and the seeds' mean MLP's
    $R_{\\mathrm{copy}}$ and $R_{\\mathrm{clim}}$ by lag."""
    out: dict[str, Any] = {}
    for name, t in mv["totals"].items():
        lin, mlp = _t(t["lin"]), _t(t["mlp"]).mean(1)
        out[name] = {f"{fam}_{den}": (num / _t(t[den])).tolist()
                     for fam, num in (("lin", lin), ("mlp", mlp))
                     for den in ("copy", "clim")}  # fmt: skip
    return out


def _vp1cdg(units: Mapping[str, Any], cc: CheckConf) -> list[Row]:
    s4, s8 = _sigma_rot(units, "S4"), _sigma_rot(units, "S8")
    neg = _mlp_vs_lin(units, "S4", cc, s4)
    pos = _mlp_vs_lin(units, "S8", cc, s8)
    return [
        _check("V-p1c", neg["words"] != "a_better" and neg["val_ok"], **neg),
        _check("V-p1d", pos["words"] == "a_better", **pos),
        _record("V-p1g", sigma_rot={"S4": s4, "S8": s8},
                steps={c: units[f"mlp-{c}"]["steps"] for c in MLP_CODES},
                sets={c: _sets_values(units[f"mlp-{c}"]) for c in MLP_CODES},
                s5_table=_s5_table(units)),
    ]  # fmt: skip


def _direction(iv: Interval) -> int:
    """-1, +1 of an interval wholly below, above 0; 0 else."""
    return (
        -1
        if iv.hi is not None and iv.hi < 0
        else (1 if iv.lo is not None and iv.lo > 0 else 0)
    )


def _vp1e(units: Mapping[str, Any], cc: CheckConf) -> list[Row]:
    rows = []
    for a, b in E_PAIRS:
        truth = _oracle_psi(a, b, cc)
        va, vb = units[f"long-{a}"], units[f"long-{b}"]
        got, ok = {}, True
        for den, t in zip(("copy", "clim"), truth, strict=True):
            iv = _psi(va, vb, den, cc)
            se = (
                (iv.hi - iv.lo) / 4
                if iv.hi is not None and iv.lo is not None
                else math.inf
            )
            compared = abs(t) >= cc.se_mult * se
            agree = (not compared) or _direction(iv) == (1 if t > 0 else -1)
            ok &= agree
            got[den] = {"truth": t, "psi": iv.psi, "lo": iv.lo, "hi": iv.hi,
                        "compared": compared, "agree": agree}  # fmt: skip
        rows.append(_check("V-p1e", ok, a=a, b=b, values=got))
    return rows


def _pop_psi(a: Mapping[str, Any], b: Mapping[str, Any], den: str) -> float:
    r"""$\psi(\hat M_a, \hat M_b)$ of one realization: the difference of the
    population $\overline{\log R}$ of the two learned predictors, fixed
    (``_lin_unit``'s ``pop_<den>``, the eval weights' $\Gamma$)."""
    return float(
        mean_log(GRID, _t(a[f"pop_{den}"])) - mean_log(GRID, _t(b[f"pop_{den}"]))
    )


def _vp1f(draws: Sequence[Mapping[str, Any]], cc: CheckConf) -> Row:
    r"""V-p1f: the share of the realizations whose interval holds its own
    target $\psi(\hat M_a, \hat M_b)$ (``_pop_psi``; the interval is
    conditional on the learned predictors, as V-p1a's target). The share
    holding the oracle's best $\psi^*$ (``_oracle_psi``) is recorded: the
    excess error of the finite training differs by code, so it does not
    cancel in $\psi$ and $\psi^*$ is not the interval's target."""
    star = _oracle_psi("S10a", "S10b", cc)
    hits, hits_star = [0, 0], [0, 0]
    gap: list[list[float]] = [[], []]
    for dr in draws:
        for i, den in enumerate(("copy", "clim")):
            iv = _psi(dr["S10a"], dr["S10b"], den, cc)
            if iv.lo is None or iv.hi is None:
                continue
            t = _pop_psi(dr["S10a"], dr["S10b"], den)
            hits[i] += iv.lo <= t <= iv.hi
            hits_star[i] += iv.lo <= star[i] <= iv.hi
            gap[i].append(t - star[i])
    share = [h / len(draws) for h in hits]
    return _check(
        "V-p1f", min(share) >= cc.cover_min, share=share, n=len(draws),
        share_star=[h / len(draws) for h in hits_star], star=star,
        target_minus_star={d: {"mean": sum(g) / len(g) if g else None,
                               "sd": stdev(g) if len(g) > 1 else None}
                           for d, g in zip(("copy", "clim"), gap, strict=True)},
    )  # fmt: skip


def _vp1h(draws: Sequence[Mapping[str, Any]], cc: CheckConf) -> Row:
    told = 0
    for dr in draws:
        ivs = [_psi(dr["S4"], dr["S4b"], den, cc) for den in ("copy", "clim")]
        told += words(*ivs) in ("a_better", "b_better")
    return _check("V-p1h", told <= cc.null_max, told=told, n=len(draws))


def merge_checks(parts: Sequence[Row], cc: CheckConf = CheckConf()) -> list[Row]:
    """The rows of V-p1 of the parts' rows (``check_parts``, any split, any
    order).

    Raises:
        ValueError: A unit or a realization is missing or given twice.
    """
    units: dict[str, Any] = {}
    got: dict[str, dict[int, Any]] = {"V-p1f": {}, "V-p1h": {}}
    for r in parts:
        if r["part"] == "draw":
            s = r["value"]["seed"]
            if s in got[r["unit"]]:
                raise ValueError(f"{r['unit']} {s} given twice")
            got[r["unit"]][s] = r["value"]
        else:
            if r["unit"] in units:
                raise ValueError(f"{r['unit']} given twice")
            units[r["unit"]] = r["value"]
    want = {"V-p1f": list(range(*cc.cover_seeds)), "V-p1h": list(range(*cc.null_seeds))}
    missing = [u for u in check_units(cc) if u not in got and u not in units]
    missing += [f"{u} {s}" for u, ss in want.items() for s in ss if s not in got[u]]
    if missing:
        raise ValueError(f"missing {missing}")
    rows = _vp1a(units, cc) + _vp1b(units, cc) + _vp1cdg(units, cc) + _vp1e(units, cc)
    rows.append(_vp1f([got["V-p1f"][s] for s in want["V-p1f"]], cc))
    rows.append(_vp1h([got["V-p1h"][s] for s in want["V-p1h"]], cc))
    return rows


def _check_config(cc: CheckConf, mconf: MLPConf) -> dict[str, Any]:
    return {"check": asdict(cc), "mlp": asdict(mconf)}


def evaluate_checks(
    out: Path,
    cc: CheckConf = CheckConf(),
    mconf: MLPConf = MLPConf(),
    *,
    units: Collection[str] | None = None,
    seeds: tuple[int, int] | None = None,
    device: str = "cpu",
) -> Path:
    """A part of V-p1 into ``out`` (``config.json``, ``rows.json``):
    ``units`` (None: all) and the realizations of ``seeds``."""
    determinism = configure_determinism(device)
    _start(out, _check_config(cc, mconf), determinism)
    rows = check_parts(cc, check_units(cc) if units is None else units, seeds, mconf,
                       torch.device(device))  # fmt: skip
    _write(out / "rows.json", rows, {})
    return out


def merge_checks_into(out: Path, parts: Sequence[Path], cc: CheckConf = CheckConf(),
                      mconf: MLPConf = MLPConf()) -> Path:  # fmt: skip
    """V-p1's ``rows.json`` in ``out`` of the parts' directories.

    Raises:
        ValueError: A part is of other settings, or a unit or a realization
            is missing or given twice.
    """
    want = json.loads(json.dumps(_check_config(cc, mconf)))
    rows: list[Row] = []
    for d in parts:
        conf = json.loads((d / "config.json").read_text())
        if {k: conf.get(k) for k in want} != want:
            raise ValueError(f"{d} is of other settings")
        rows += json.loads((d / "rows.json").read_text())
    _start(out, _check_config(cc, mconf), configure_determinism("cpu"))
    _write(out / "rows.json", merge_checks(rows, cc), {})
    return out


# -- the CLI ------------------------------------------------------------------------


def _ints(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split(",") if x)


def _names(text: str) -> tuple[str, ...]:
    return tuple(x for x in text.split(",") if x)


def _range(text: str) -> tuple[int, int]:
    lo, _, hi = text.partition(":")
    return int(lo), int(hi)


def main(argv: Sequence[str] | None = None) -> Path:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = p.add_subparsers(dest="job", required=True)
    checks = sub.add_parser("checks")
    checks.add_argument("--out", type=Path, required=True)
    checks.add_argument("--only", type=_names, default=None,
                        help="a part: these units (check_units), comma-separated")  # fmt: skip
    checks.add_argument("--seeds", type=_range, default=None,
                        help="a part: V-p1f's and V-p1h's realizations of seeds lo:hi")  # fmt: skip
    checks.add_argument("--device", default="cpu")
    merge = sub.add_parser("merge-checks")
    merge.add_argument("--out", type=Path, required=True)
    merge.add_argument("parts", type=Path, nargs="+")
    yard = sub.add_parser("yard")
    yard.add_argument("--config", type=Path, required=True)
    yard.add_argument("--out", type=Path, required=True)
    for name in YARD_INITS:
        yard.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    yard.add_argument("--init-v16", type=Path, required=True)
    yard.add_argument("--codes", type=_names, default=None,
                      help="only these yardsticks (YARD_CODES), comma-separated")  # fmt: skip
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
        return evaluate_checks(a.out, units=a.only, seeds=a.seeds, device=a.device)
    if a.job == "merge-checks":
        return merge_checks_into(a.out, a.parts)
    if a.job == "yard":
        conf = _conf_of(a.config, a.device)
        inits = {n: getattr(a, n) for n in YARD_INITS}
        return evaluate_yard(conf, a.sets, a.out, inits=inits, init_v16=a.init_v16,
                             root=a.root, codes=a.codes)  # fmt: skip
    if a.job == "run":
        return evaluate_run(a.run_dir, steps=a.steps, sets_dir=a.sets, root=a.root,
                            device=a.device)  # fmt: skip
    return evaluate_vae(a.ckpt, name=a.name, sets_dir=a.sets, out=a.out, root=a.root,
                        device=a.device)  # fmt: skip


if __name__ == "__main__":
    main()
