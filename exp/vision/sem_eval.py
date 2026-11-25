r"""V18c's main measure $m$, remade (``plans/jepa.md``, 「V18c: 主の指標の作り
直しと E1 の修理」; README 「V18c: 主の指標を作り直す」): how much of a
scene's abstract features, from a higher resolution than any code saw, a
fixed linear readout takes out of a code.

The sets (``sem_sets``): the probes' fit stacks (``StreamEvaluator.
fit_stacks`` of the span ``SPAN``, about 6000) fit the readouts and the
targets' spaces; the held-out video's grid (``gen_eval.Zurich``, 1 s apart)
is cut by ``zurich_split`` into calib (the ridges' scales), sel (the choice
of the long runs' settings) and final (their judgement). The fit set's two
halves (``SemSets.halves``) alternate over the odd blocks of
``StreamEvaluator``'s span that its stacks come from.

The targets (``build_targets``, once into a cache ``v18c-sem-<key>``, of
the frames of ``exp.lib.tours.hires``, 320 px): of the last frame of a stack,
``sem`` (the main one: the pool3 of FID's Inception of the 4 quarters tl,
tr, bl, br, 160 px each, side by side, 8192 numbers, in their PCA of
``SemConf.sem_dim`` components fit on the fit set from ``SEM_PCA_SEED``,
each component over its sd on the fit set: whitened), ``quad/<q>`` (one
quarter's pool3, so, for E1's repair), ``scene`` (the whole frame's, of
``scene_dim``; listed only) and ``inc64`` (the pool3 of the 64 px last
frame, a yardstick's code). Also, in ``targets.json``: the split, the
items' $v_i$ and blocks of each set, the K3 bound $b_{16}$ of each set,
the integrated time $\tau$ of ``sem`` along the grid and the block length
$L = \min(\max(\lceil 2 \tau \rceil, L_{\min}), L_{\max})$, V-m0's shift
identification (``shift_top1``) and the targets' sensitivity
(``sensitivity``).

A code's row (``score``): the ridge (``readout.fit_ridge``) from its inputs
on the fit set to ``sem`` of the scale of ``ALPHA_SCALES`` whose
``centered_r2`` on calib is the best; ``m_calib``, ``m_sel``, ``m_final``
and the items' $e_i$ (``e_sel``, ``e_final``); ``m_sel_half`` (fit on each
half, its scale chosen again); ``m_scene_*`` (to ``scene``); ``id_easy_sel``
(the nearest cosine among sel's targets, centred on the grid's mean,
chance $1/n_{\mathrm{sel}}$); and of the K3 codes ``k3_bound_*``. ``mean``
is the constant (its $m$ exactly 0). No input is reduced: every code is at
most ``SEM_READOUT_DIM`` wide.

Codes of a run (``RUN_CODES``): ``z`` ($W \mathrm{vec}(t)$), ``z_k3`` (the
tokens all replaced by their mean over the positions, $W \mathrm{vec}(1
\otimes \bar t)$: the scene alone) and ``h_mean``. The yardsticks
(``YARDSTICKS``): ``mean``, ``color`` (``ColorCode(16)``), ``color8``
(``ColorCode(8)``), ``pca<k>`` (the first $k$ of one PCA of the pixels),
``v16``, ``v17``, ``init_c`` (a V18c run's step 0 $z$), ``init_c_k3``,
``inc64``.

Into ``<run_dir>/posthoc_sem/`` (``config.json``, ``rows/<step>-<code>.json``)
and the yardsticks' ``<out>/posthoc_sem/rows/yardstick-<code>.json``; a job
whose file is there is skipped, after the ``config.json`` there is checked
against what the rows depend on::

    python -m exp.vision.sem_eval frames --out runs/v18c-sem/frames.json
    python -m exp.vision.sem_eval targets --hires <dir> [--inception <path>]
    python -m exp.vision.sem_eval yardsticks --out runs/v18c-sem/yardsticks \
        --v16 <ckpt> --v17 <ckpt> --init-run <run_dir>
    python -m exp.vision.sem_eval run <run_dir> --steps 0,1000,2000,4000,6000 \
        [--codes z,z_k3,h_mean]
"""

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor
from torch.nn import functional as F

from exp.lib import ckpt
from exp.lib.linear_dyn import ALPHA_SCALES
from exp.lib.metrics import contiguous_blocks, identification, integrated_time
from exp.lib.provenance import configure_determinism, of_diagnosis
from exp.lib.readout import (
    Ridge,
    centered_r2,
    centered_terms,
    fit_ridge,
    predict,
    rank_bound,
)
from exp.lib.rundir import write_json
from exp.lib.tours.hires import HiresFrames
from exp.lib.tours.stacks import DATA_ENV, StackedTourFrames, tours_root
from exp.vision.compose import from_dict
from exp.vision.eval import PCA_SEED
from exp.vision.fill_eval import Space, Targets, _apply
from exp.vision.gen_eval import Features, Zurich, inception_features
from exp.vision.jepa import JEPAModules
from exp.vision.jepa_diag import GRID_STEP, run_ckpts
from exp.vision.jepa_eval import (
    SPAN,
    ColorCode,
    PCACode,
    TokenCode,
    build_run_jepa,
    pixels,
    vae_at,
)
from exp.vision.jepa_iid import v16_config
from exp.vision.schema import RunConf
from exp.vision.stream_eval import BLOCK, StreamEvaluator, spanned

__all__ = [
    "K3_CODES",
    "OUT",
    "QUARTERS",
    "RUN_CODES",
    "SEM_PCA_SEED",
    "SEM_READOUT_DIM",
    "SENS_SEED",
    "SPLITS",
    "YARDSTICKS",
    "K3Code",
    "SemConf",
    "SemSets",
    "blur",
    "build_targets",
    "evaluate_frames",
    "evaluate_run",
    "evaluate_targets",
    "evaluate_yardsticks",
    "far_pairs",
    "main",
    "match_stats",
    "quarters",
    "score",
    "sem_sets",
    "sensitivity",
    "white_balance",
    "zurich_split",
]

OUT = "posthoc_sem"
SEM_PCA_SEED = 97  # the PCA of every target here
SENS_SEED = 103  # the pairs of the content swap (sensitivity)
BOOT, BOOT_SEED = 2000, 101  # the block bootstrap's (judge_m), recorded here
SEM_READOUT_DIM = 2048  # no code is reduced: none is wider
QUARTERS = ("tl", "tr", "bl", "br")
SPLITS = ("calib", "sel", "final")
RUN_CODES = ("z", "z_k3", "h_mean")
YARDSTICKS = (
    "mean",
    "color",
    "color8",
    "pca16",
    "pca64",
    "pca256",
    "pca1024",
    "v16",
    "v17",
    "init_c",
    "init_c_k3",
    "inc64",
)
K3_CODES = ("z_k3", "init_c_k3")
SENS = ("bright", "white", "blur")  # the nuisances (sensitivity)
BRIGHT = 0.8
WHITE = (1.1, 1.0, 0.9)
BLUR_SIGMA = 1.0  # pixels at 64 px: SemConf's factor times this at the hires size
YARD_PREFIX = "yardstick"
CACHE_PREFIX = "v18c-sem-"


@dataclass(frozen=True)
class SemConf:
    """The settings (module docstring; ``plans/jepa.md``), by default the
    plan's."""

    grid_step: int = GRID_STEP  # frames between the grid's stacks (1 s)
    segments: int = 6  # zurich_split's
    gap: int = 30  # grid stacks dropped at each segment's start (30 s)
    sem_dim: int = 64
    scene_dim: int = 32
    sens_gap: int = 300  # grid stacks between a pair of the content swap (5 min)
    l_min: int = 16
    l_max: int = 155
    k3_rank: int = 16  # d of V18c's tokens: the rank of z_k3
    pca_max: int = 1024  # the pixels' PCA of pca<k>
    batch: int = 64  # hires frames through Inception at once


# -- the sets ---------------------------------------------------------------------


def zurich_split(n_grid: int, segments: int = 6, gap: int = 30) -> dict[str, Tensor]:
    r"""The grid's indices of each of ``SPLITS`` and ``segment`` ``(n_grid,)``
    the segment of each index (-1 where dropped): the grid cut into
    ``segments`` runs of $\lfloor G / \text{segments} \rfloor$ (the last
    one taking the remainder), the first ``gap`` of each dropped, segment
    $s$ to ``SPLITS[s % 3]``.

    Raises:
        ValueError: A segment is not longer than ``gap``.
    """
    size = n_grid // segments
    if size <= gap:
        raise ValueError(f"segments of {size} of a grid of {n_grid}, gap {gap}")
    segment = torch.full((n_grid,), -1, dtype=torch.long)
    for s in range(segments):
        end = n_grid if s == segments - 1 else (s + 1) * size
        segment[s * size + gap : end] = s
    out = {
        name: ((segment >= 0) & (segment % 3 == j)).nonzero().flatten()
        for j, name in enumerate(SPLITS)
    }
    return out | {"segment": segment}


@dataclass(frozen=True)
class SemSets:
    """The stacks u8 of the fit set ``fit`` ``(m, n, 3, S, S)`` and of the
    grid ``grid`` ``(G, n, 3, S, S)``, the fit set's ``halves`` ``(m,)``
    (0 or 1), ``places`` (the video and frame of each set's last frames) and
    ``ident`` (what names the sets)."""

    fit: Tensor
    grid: Tensor
    halves: Tensor
    places: Mapping[str, list[tuple[str, int]]]
    ident: Mapping[str, Any]

    def last(self, name: str) -> Tensor:
        """The last frames u8 ``(N, 3, S, S)`` of ``"fit"`` or ``"grid"``."""
        return (self.fit if name == "fit" else self.grid)[:, -1]


def sem_sets(
    conf: RunConf,
    root: Path,
    sconf: SemConf,
    *,
    span: float = SPAN,
    spacing: Mapping[str, Any] | None = None,
) -> SemSets:
    """The sets of ``conf``'s data at ``root`` (module docstring);
    ``spacing`` ``StreamEvaluator``'s spacings by name (tests)."""
    view = spanned(conf, span)
    spacing = dict(spacing or {})
    with tempfile.TemporaryDirectory() as tmp:  # StreamEvaluator's out, unused
        ev = StreamEvaluator(view, Path(tmp), root, Path(tmp), c=1, **spacing)
    t = conf.data.tours
    zurich = Zurich(conf, root, sconf.grid_step)
    stream = StackedTourFrames(
        root,
        n=1,
        held_out=t.held_out,
        frames=t.frames,
        stride=t.stride,
        advance=t.advance,
    )
    block = round(spacing.get("block", BLOCK) / view.time.frame)
    ks = torch.tensor(ev.fit_ks)
    places = {
        "fit": [stream.locate(k) for k in ev.fit_ks],
        "grid": [(t.held_out, int(s) + zurich.span) for s in zurich.starts],
    }
    ident = {
        "data_version": t.data_version,
        "held_out": t.held_out,
        "fit_ks": ev.fit_ks,
        "grid_starts": zurich.starts.tolist(),
    }
    return SemSets(
        fit=ev.fit_stacks,
        grid=zurich.stacks(torch.arange(len(zurich))),
        halves=(ks // block // 2) % 2,  # the odd blocks, alternately
        places=places,
        ident=ident,
    )


# -- the targets -------------------------------------------------------------------


def quarters(frames: Tensor) -> dict[str, Tensor]:
    """The 4 quarters (``QUARTERS``) of the frames ``(B, 3, H, W)``."""
    h, w = frames.shape[-2] // 2, frames.shape[-1] // 2
    return {
        "tl": frames[..., :h, :w],
        "tr": frames[..., :h, w:],
        "bl": frames[..., h:, :w],
        "br": frames[..., h:, w:],
    }


def _pool3(features: Features, frames: Tensor) -> dict[str, Tensor]:
    """The features (float64, CPU) of each quarter and ``whole`` of the u8
    frames ``(B, 3, H, W)``."""
    out = {
        q: features(x.contiguous()).double().cpu() for q, x in quarters(frames).items()
    }
    return out | {"whole": features(frames).double().cpu()}


def _explained(raw: Tensor, space: Space) -> float:
    """The share of the variance of ``raw`` ``(m, F)`` on the fit set that
    the space's components hold."""
    x = raw.double()
    assert space.comps is not None
    u = (x - space.mean) @ space.comps.T
    return float(u.var(0).sum() / x.var(0).sum())


def white_balance(x: Tensor) -> Tensor:
    """The u8 frames ``(B, 3, H, W)`` of each channel times ``WHITE``, cut
    at 255 and rounded."""
    gains = torch.tensor(WHITE, dtype=torch.float64, device=x.device)[:, None, None]
    return (x.double() * gains).clamp(max=255).round().to(torch.uint8)


def blur(x: Tensor, sigma: float) -> Tensor:
    r"""The u8 frames ``(B, 3, H, W)`` blurred by a Gaussian of ``sigma``
    pixels (its taps to $\lceil 4 \sigma \rceil$, made to sum to 1; the
    edges reflected), rounded."""
    r = math.ceil(4 * sigma)
    taps = torch.arange(-r, r + 1, dtype=torch.float64, device=x.device)
    k = torch.exp(-(taps**2) / (2 * sigma**2))
    k = k / k.sum()
    c = x.shape[1]
    y = F.pad(x.double(), (r, r, r, r), mode="reflect")
    y = F.conv2d(y, k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    y = F.conv2d(y, k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)
    return y.round().clamp(0, 255).to(torch.uint8)


def match_stats(src: Tensor, ref: Tensor) -> Tensor:
    """The frames ``src`` ``(B, 3, H, W)`` whose channels' means and sds
    (over each frame's pixels) are those of ``ref``'s, float64 (unrounded)."""
    s, r = src.double(), ref.double()
    ms, mr = s.mean((2, 3), keepdim=True), r.mean((2, 3), keepdim=True)
    ss, sr = s.std((2, 3), keepdim=True), r.std((2, 3), keepdim=True)
    return (s - ms) / ss.clamp(min=1e-12) * sr + mr


def _nuisance(name: str, x: Tensor, sigma: float) -> Tensor:
    """The nuisance ``name`` of ``SENS`` of the u8 frames ``x``."""
    if name == "bright":
        return (x.double() * BRIGHT).round().to(torch.uint8)
    if name == "white":
        return white_balance(x)
    return blur(x, sigma)


def far_pairs(idx: Tensor, min_gap: int, gen: torch.Generator) -> Tensor:
    """A one-to-one pairing of the grid indices ``idx`` ``(n,)``: the
    positions ``(n,)`` in ``idx`` of each one's partner, at least
    ``min_gap`` apart on the grid. A matching by augmenting paths (breadth
    first), the
    order of the indices and of each one's candidates random by ``gen``.

    Raises:
        ValueError: There is no such pairing.
    """
    n = len(idx)
    g = idx.tolist()
    order = torch.randperm(n, generator=gen).tolist()
    prefs = [torch.randperm(n, generator=gen).tolist() for _ in range(n)]
    owner = [-1] * n  # the index paired to each partner

    def augment(start: int) -> bool:
        """A shortest augmenting path from ``start`` (breadth first), its
        pairs flipped; False if none."""
        reached: dict[int, int] = {}  # partner -> the index that reached it
        via: dict[int, int | None] = {start: None}  # index -> its partner
        frontier = [start]
        while frontier:
            nxt = []
            for i in frontier:
                for j in prefs[i]:
                    if j in reached or abs(g[i] - g[j]) < min_gap:
                        continue
                    reached[j] = i
                    if owner[j] < 0:
                        at: int | None = j
                        while at is not None:
                            k = reached[at]
                            owner[at], at = k, via[k]
                        return True
                    if owner[j] not in via:
                        via[owner[j]] = j
                        nxt.append(owner[j])
            frontier = nxt
        return False

    for i in order:
        if not augment(i):
            raise ValueError(f"no pairing of {n} indices {min_gap} apart")
    pi = torch.empty(n, dtype=torch.long)
    pi[torch.tensor(owner)] = torch.arange(n)
    return pi


def sensitivity(
    sets: SemSets,
    hires: HiresFrames,
    features: Features,
    spaces: Mapping[str, Space],
    split: Mapping[str, Tensor],
    sconf: SemConf,
    *,
    dev: torch.device,
) -> dict[str, Any]:
    r"""The targets' sensitivity (``plans/jepa.md``, 「的の感度の確かめ」) on
    the grid's stacks of the split's sets: of each nuisance $T_j$ of
    ``SENS``, $d_j = \langle \lVert y(T_j x) - y(x) \rVert^2 \rangle$; of the
    content swap, $d_c$ of ``match_stats`` $(x', x)$ rounded, $x'$ a stack
    of the same set ``sens_gap`` or more away (``far_pairs``, ``SENS_SEED``);
    $\rho_j = d_j / d_c$. The same of the pixel target ``spaces["pix"]``
    (the 64 px frame, the hires frame shrunk by its means), ``*_pix``."""
    t = sets.places["grid"]
    factor = hires.size // sets.grid.shape[-1]
    sigma = BLUR_SIGMA * factor
    gen = torch.Generator().manual_seed(SENS_SEED)
    names = (*SENS, "content")
    d = {k: 0.0 for k in names}
    d_pix = {k: 0.0 for k in names}
    count = 0

    def sem(x: Tensor) -> Tensor:
        p = _pool3(features, x)
        return spaces["sem"](torch.cat([p[q] for q in QUARTERS], 1))

    def pix(x: Tensor) -> Tensor:
        small = F.avg_pool2d(x.double(), factor) / 255
        return spaces["pix"](small.flatten(1).cpu())

    for name in SPLITS:
        idx = split[name]
        partner = idx[far_pairs(idx, sconf.sens_gap, gen)]
        for i in range(0, len(idx), sconf.batch):
            own, other = idx[i : i + sconf.batch], partner[i : i + sconf.batch]
            x = hires.frames([t[j] for j in own.tolist()]).to(dev)
            x2 = hires.frames([t[j] for j in other.tolist()]).to(dev)
            swapped = match_stats(x2, x).round().clamp(0, 255).to(torch.uint8)
            y0, p0 = sem(x), pix(x)
            for k in names:
                xk = swapped if k == "content" else _nuisance(k, x, sigma)
                d[k] += float((sem(xk) - y0).square().sum())
                d_pix[k] += float((pix(xk) - p0).square().sum())
            count += len(own)
    d = {k: v / count for k, v in d.items()}
    d_pix = {k: v / count for k, v in d_pix.items()}
    return {
        "n": count,
        "sigma": sigma,
        "d": d,
        "rho": {k: d[k] / d["content"] for k in SENS},
        "d_pix": d_pix,
        "rho_pix": {k: d_pix[k] / d_pix["content"] for k in SENS},
    }


def targets_key(sets: SemSets, sconf: SemConf, hires_name: str) -> str:
    """The first 12 hex digits of the sha256 of the sets' ``ident``, the
    settings the targets depend on and the hires data's name: the cache's
    name ``v18c-sem-<key>``."""
    settings = {
        "sem": asdict(sconf) | {"batch": None},
        "seeds": {"pca": SEM_PCA_SEED, "sens": SENS_SEED},
        "sens": {"bright": BRIGHT, "white": WHITE, "blur_sigma": BLUR_SIGMA},
    }
    text = json.dumps(
        {"sets": dict(sets.ident), "settings": settings, "hires": hires_name},
        sort_keys=True,
    )
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def build_targets(
    sets: SemSets,
    hires: HiresFrames,
    features: Features,
    sconf: SemConf,
    *,
    dev: torch.device,
    record: Mapping[str, Any] | None = None,
) -> Targets:
    """The targets (module docstring) of ``sets`` from ``hires`` by
    ``features`` (FID's Inception in use, ``record`` its weights' path and
    sha256)."""
    raw: dict[str, dict[str, Tensor]] = {}
    for name in ("fit", "grid"):
        places = sets.places[name]
        parts = [
            _pool3(features, hires.frames(places[i : i + sconf.batch]).to(dev))
            for i in range(0, len(places), sconf.batch)
        ]
        raw[name] = {k: torch.cat([p[k] for p in parts]) for k in (*QUARTERS, "whole")}
    spaces: dict[str, Space] = {}
    values: dict[str, dict[str, Tensor]] = {}
    explained: dict[str, float] = {}

    def add(key: str, of: Callable[[dict[str, Tensor]], Tensor], dim: int) -> None:
        fit = of(raw["fit"])
        spaces[key] = space = Space.fit(fit, dim, seed=SEM_PCA_SEED, whiten=True)
        values[key] = {s: space(of(raw[s])) for s in ("fit", "grid")}
        explained[key] = _explained(fit, space)

    add("sem", lambda r: torch.cat([r[q] for q in QUARTERS], 1), sconf.sem_dim)
    for q in QUARTERS:
        add(f"quad/{q}", lambda r, q=q: r[q], sconf.sem_dim)
    add("scene", lambda r: r["whole"], sconf.scene_dim)
    values["inc64"] = {
        s: torch.cat([features(c).double().cpu() for c in sets.last(s).split(256)])
        for s in ("fit", "grid")
    }
    small = sets.last("fit").flatten(1).double() / 255
    pix = Space.fit(small, sconf.sem_dim, seed=SEM_PCA_SEED, whiten=True)
    y = values["sem"]["grid"]
    split = zurich_split(len(y), sconf.segments, sconf.gap)
    tau = integrated_time(y)
    length = min(max(math.ceil(2 * tau), sconf.l_min), sconf.l_max)
    per_set: dict[str, dict[str, Any]] = {}
    for name in SPLITS:
        idx = split[name]
        _, v = centered_terms(y[idx], y[idx])
        per_set[name] = {
            "idx": idx.tolist(),
            "v": v.tolist(),
            "blocks": contiguous_blocks(split["segment"][idx], length).tolist(),
            "k3_bound": rank_bound(y[idx], sconf.k3_rank),
        }
    sens = sensitivity(
        sets, hires, features, spaces | {"pix": pix}, split, sconf, dev=dev
    )
    meta = {
        "sets": dict(sets.ident),
        "sem": asdict(sconf),
        "inception": dict(record or {}),
        "hires": {"name": hires.root.name, "index": hires.index},
        "seeds": {"pca": SEM_PCA_SEED, "sens": SENS_SEED},
        "explained": explained,
        "segment": split["segment"].tolist(),
        "sets_split": per_set,
        "tau": tau,
        "L": length,
        "L_at_max": length == sconf.l_max,
        "shift_top1": _shift_top1(y, split["sel"]),
        "sens": sens,
    }
    return Targets(spaces, values, meta)


def _shift_top1(y: Tensor, sel: Tensor) -> float:
    """V-m0's: the share of the sel stacks found from the next grid stack's
    target (1 s on, in sel too) among sel's by the nearest cosine, centred on
    the grid's mean."""
    members = set(sel.tolist())
    pos = [i for i, g in enumerate(sel.tolist()) if g + 1 in members]
    query = y[sel[pos] + 1]
    out = identification(
        query, y[sel], center=y.mean(0), truth=torch.tensor(pos, dtype=torch.long)
    )
    return out["top1"]


def _default_base() -> Path:
    repo = Path(__file__).resolve().parents[2]
    return Path(os.environ.get(DATA_ENV, repo / ".fleet" / "data"))


def find_cache(base: Path, sets: SemSets, sconf: SemConf) -> Path:
    """The one cache under ``base`` of these sets and settings.

    Raises:
        FileNotFoundError: None (``sem_eval targets`` makes it).
        ValueError: More than one (name it).
    """
    found = []
    for d in sorted(base.glob(f"{CACHE_PREFIX}*")):
        meta_path = d / "targets.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text())
        same = meta["sets"] == json.loads(json.dumps(dict(sets.ident)))
        if same and meta["sem"] == json.loads(json.dumps(asdict(sconf))):
            found.append(d)
    if not found:
        raise FileNotFoundError(
            f"no targets under {base}: make them by `sem_eval targets`"
        )
    if len(found) > 1:
        raise ValueError(f"several targets of these sets: {found}; pass one by --sem")
    return found[0]


# -- the codes and the scores ---------------------------------------------------------


class K3Code:
    r"""``z_k3`` of a V18c JEPA ``mods``: the target side's tokens $t$ ``(B,
    T, d)`` all replaced by their mean over the positions $\bar t$, through
    $W$: $W \mathrm{vec}(1_T \otimes \bar t)$ ``(B, D)``. Its centred values
    span at most $d$ dimensions ($W (1_T \otimes I_d) \bar t$).

    Raises:
        ValueError: ``mods`` has no ``mix`` (not V18c's).
    """

    def __init__(self, mods: JEPAModules) -> None:
        if mods.mix is None:
            raise ValueError("z_k3 is of a JEPA with W (V18c's sigreg_on mix)")
        self._mods, self._mix = mods, mods.mix

    def modules(self) -> Mapping[str, torch.nn.Module]:
        return self._mods.modules()

    def mu(self, x: Tensor) -> Tensor:
        t = self._mods.tokens(x)
        return self._mix(t.mean(1, keepdim=True).expand_as(t).flatten(1))


def _choose(x: Tensor, y: Tensor, x_cal: Tensor, y_cal: Tensor) -> tuple[Ridge, float]:
    """The ridge of the scale of ``ALPHA_SCALES`` whose ``centered_r2`` on
    the calib set is the best (the first on a tie), and that scale."""
    best: tuple[Ridge, float] | None = None
    top = -math.inf
    for c in ALPHA_SCALES:
        fit = fit_ridge(x, y, c)
        s = centered_r2(predict(fit, x_cal), y_cal)
        if s > top:
            best, top = (fit, c), s
    assert best is not None
    return best


def score(
    x_fit: Tensor | None,
    x_grid: Tensor | None,
    targets: Targets,
    halves: Tensor,
) -> dict[str, Any]:
    """A code's row (module docstring) of its inputs on the fit set
    ``(m, D)`` and the grid ``(G, D)``; None for ``mean``.

    Raises:
        ValueError: The inputs are wider than ``SEM_READOUT_DIM``.
    """
    meta = targets.meta["sets_split"]
    idx = {s: torch.tensor(meta[s]["idx"]) for s in SPLITS}
    y_fit, y_grid = targets.y("sem", "fit"), targets.y("sem", "grid")
    row: dict[str, Any] = {}
    if x_fit is None or x_grid is None:  # the constant: exactly 0
        zero = {s: torch.zeros(len(i), 1, dtype=torch.float64) for s, i in idx.items()}
        for s in SPLITS:
            e, v = centered_terms(zero[s], y_grid[idx[s]])  # e = v, bit for bit
            row[f"m_{s}"] = float(1 - e.sum() / v.sum())
            if s != "calib":
                row[f"e_{s}"] = e.tolist()
        return row | {
            "dim": 0,
            "alpha_scale": None,
            "alpha": None,
            "m_sel_half": [0.0, 0.0],
            "m_scene_sel": 0.0,
            "m_scene_final": 0.0,
            "id_easy_sel": None,
        }
    if x_fit.shape[1] > SEM_READOUT_DIM:
        raise ValueError(f"a code of {x_fit.shape[1]} > {SEM_READOUT_DIM} dimensions")
    dev = x_fit.device
    yf, yg = y_fit.to(dev), y_grid.to(dev)
    cal = idx["calib"].to(dev)
    g, c = _choose(x_fit, yf, x_grid[cal], yg[cal])
    p = predict(g, x_grid).cpu()
    row |= {"dim": x_fit.shape[1], "alpha_scale": c, "alpha": g.alpha}
    for s in SPLITS:
        e, v = centered_terms(p[idx[s]], y_grid[idx[s]])
        row[f"m_{s}"] = float(1 - e.sum() / v.sum())
        if s != "calib":
            row[f"e_{s}"] = e.tolist()
    sel = idx["sel"]
    halves = halves.to(dev)
    row["m_sel_half"] = []
    for h in (0, 1):
        on = halves == h
        gh, _ = _choose(x_fit[on], yf[on], x_grid[cal], yg[cal])
        row["m_sel_half"].append(
            centered_r2(predict(gh, x_grid[sel.to(dev)]).cpu(), y_grid[sel])
        )
    sf, sg = targets.y("scene", "fit").to(dev), targets.y("scene", "grid")
    gs, _ = _choose(x_fit, sf, x_grid[cal], sg.to(dev)[cal])
    ps = predict(gs, x_grid).cpu()
    for s in ("sel", "final"):
        row[f"m_scene_{s}"] = centered_r2(ps[idx[s]], sg[idx[s]])
    found = identification(p[sel], y_grid[sel], center=y_grid.mean(0))
    return row | {"id_easy_sel": found["top1"]}


type Inputs = tuple[Tensor, Tensor] | None


def _inputs(
    fn: Callable[[Tensor], Tensor], sets: SemSets, dev: torch.device
) -> tuple[Tensor, Tensor]:
    """``fn`` of the fit set's and the grid's stacks."""
    return _apply(fn, sets.fit, dev), _apply(fn, sets.grid, dev)


def _max_mem(dev: torch.device) -> float | None:
    if dev.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(dev) / 2**30


# -- the jobs -----------------------------------------------------------------------


def _conf_of(run_dir: Path, device: str | None) -> RunConf:
    conf = from_dict(yaml.safe_load((run_dir / "config.yaml").read_text()))
    return conf if device is None else replace(conf, device=device)


def _yard_conf(device: str | None) -> RunConf:
    d = v16_config()
    d["device"] = device or "cuda"
    return from_dict(d)


class _Context:
    """The sets, the targets and the record of an evaluation of ``conf``."""

    def __init__(
        self,
        conf: RunConf,
        out: Path,
        *,
        sconf: SemConf,
        root: Path | None,
        sem: Path | None,
        cache: Path | None,
        spacing: Mapping[str, Any] | None,
        span: float,
    ) -> None:
        self.conf, self.sconf, self.out = conf, sconf, out
        self.dev = torch.device(conf.device)
        self.determinism = configure_determinism(conf.device)
        root = tours_root(conf.data.tours.data_version) if root is None else root
        self.root = root
        self.sets = sem_sets(conf, root, sconf, span=span, spacing=spacing)
        if sem is None:
            sem = find_cache(
                _default_base() if cache is None else cache, self.sets, sconf
            )
        self.path = sem
        self.targets = Targets.load(sem)
        if self.targets.meta["sets"] != json.loads(json.dumps(dict(self.sets.ident))):
            raise ValueError(f"the targets {sem} are not of these sets")
        if self.targets.meta["sem"] != json.loads(json.dumps(asdict(sconf))):
            raise ValueError(f"the targets {sem} are of other settings")

    def depends(self) -> dict[str, Any]:
        """What the rows depend on beside the checkpoint."""
        d = {
            "sem": asdict(self.sconf),
            "targets": self.path.name,
            "seeds": {"pca": SEM_PCA_SEED, "pixels_pca": PCA_SEED},
            "alpha_scales": list(ALPHA_SCALES),
            "L": self.targets.meta["L"],
        }
        d = json.loads(json.dumps(d))
        key = hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]
        return d | {"key": key}

    def start(self, extra: Mapping[str, Any]) -> None:
        """``out/config.json`` (if absent).

        Raises:
            ValueError: The ``config.json`` there is of other settings.
        """
        path = self.out / "config.json"
        depends = self.depends()
        if path.exists():
            old = json.loads(path.read_text())
            if {k: old.get(k) for k in depends} != depends:
                raise ValueError(f"{path} is of other settings: remove it first")
            return
        self.out.mkdir(parents=True, exist_ok=True)
        write_json(
            path,
            depends
            | {
                "determinism": self.determinism,
                "provenance": {
                    k: v for k, v in of_diagnosis({}).items() if k != "settings"
                },
            }
            | dict(extra),
        )

    def write(
        self, name: str, inputs: Inputs, stamp: Mapping[str, Any], t0: float
    ) -> None:
        """The row of the inputs ``inputs`` into ``rows/<name>.json``."""
        x_fit, x_grid = (None, None) if inputs is None else inputs
        row = dict(stamp) | score(x_fit, x_grid, self.targets, self.sets.halves)
        code = stamp["code"]
        if code in K3_CODES:
            split = self.targets.meta["sets_split"]
            row |= {f"k3_bound_{s}": split[s]["k3_bound"] for s in ("sel", "final")}
        row |= {"sec": time.time() - t0, "max_mem": _max_mem(self.dev)}
        row["key"] = self.depends()["key"]
        path = self.out / "rows" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, row)


def evaluate_frames(
    out: Path,
    *,
    conf: RunConf | None = None,
    sconf: SemConf = SemConf(),
    root: Path | None = None,
    spacing: Mapping[str, Any] | None = None,
    span: float = SPAN,
) -> Path:
    """``out``: each video's frames the targets need (the fit set's and the
    grid's last frames), sorted, once each."""
    conf = _yard_conf("cpu") if conf is None else conf
    root = tours_root(conf.data.tours.data_version) if root is None else root
    sets = sem_sets(conf, root, sconf, span=span, spacing=spacing)
    lists: dict[str, set[int]] = {}
    for name in ("fit", "grid"):
        for v, f in sets.places[name]:
            lists.setdefault(v, set()).add(f)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, {v: sorted(f) for v, f in sorted(lists.items())})
    return out


def evaluate_targets(
    hires: Path,
    cache: Path | None = None,
    *,
    conf: RunConf | None = None,
    sconf: SemConf = SemConf(),
    root: Path | None = None,
    features: Features | None = None,
    inception: Path | None = None,
    spacing: Mapping[str, Any] | None = None,
    span: float = SPAN,
) -> Path:
    """The targets' cache (module docstring) of ``conf`` (default
    ``v16_config`` on CUDA) from the hires data ``hires``, under ``cache``
    (default ``$MUJOU_EXP_DATA``). Returns its directory (an existing one is
    kept)."""
    conf = _yard_conf(None) if conf is None else conf
    configure_determinism(conf.device)
    root = tours_root(conf.data.tours.data_version) if root is None else root
    record: dict[str, Any] = {}
    if features is None:
        features, record = inception_features(inception, conf.device)
    sets = sem_sets(conf, root, sconf, span=span, spacing=spacing)
    base = _default_base() if cache is None else cache
    path = base / f"{CACHE_PREFIX}{targets_key(sets, sconf, hires.name)}"
    if (path / "targets.pt").is_file():
        return path
    data = HiresFrames(hires)
    got = build_targets(
        sets, data, features, sconf, dev=torch.device(conf.device), record=record
    )
    got.meta["key"] = path.name
    got.save(path)
    return path


def _step_inputs(
    mods: JEPAModules, code: str, sets: SemSets, dev: torch.device
) -> tuple[Tensor, Tensor]:
    """The inputs of a run's code ``code`` of ``RUN_CODES`` (``mods`` as
    loaded)."""
    if code == "z":
        return _inputs(mods.mu, sets, dev)
    if code == "z_k3":
        return _inputs(K3Code(mods).mu, sets, dev)
    return _inputs(TokenCode(mods, "h_mean").mu, sets, dev)


def evaluate_run(
    run_dir: Path,
    *,
    steps: Sequence[int],
    codes: Sequence[str] = RUN_CODES,
    sconf: SemConf = SemConf(),
    root: Path | None = None,
    sem: Path | None = None,
    cache: Path | None = None,
    device: str | None = None,
    spacing: Mapping[str, Any] | None = None,
    span: float = SPAN,
) -> Path:
    """The rows of ``codes`` of the checkpoints of ``steps`` of the JEPA run
    in ``run_dir`` (module docstring). Returns ``run_dir / OUT``.

    Raises:
        ValueError: A code is not of ``RUN_CODES``, or a step has no
            checkpoint.
    """
    unknown = [c for c in codes if c not in RUN_CODES]
    if unknown:
        raise ValueError(f"a run's codes are of {RUN_CODES}, not {unknown}")
    conf = _conf_of(run_dir, device)
    ckpts = run_ckpts(run_dir)
    missing = [s for s in steps if s not in ckpts]
    if missing:
        raise ValueError(f"no checkpoint of steps {missing} (of {list(ckpts)})")
    ctx = _Context(
        conf, run_dir / OUT, sconf=sconf, root=root, sem=sem, cache=cache,
        spacing=spacing, span=span,
    )  # fmt: skip
    settings = json.loads((run_dir / "jepa.json").read_text())
    mods = build_run_jepa(conf, settings)
    for step in steps:
        prefix = f"{step:06d}"
        todo = [
            c for c in codes if not (ctx.out / "rows" / f"{prefix}-{c}.json").exists()
        ]
        if not todo:
            continue
        ckpt.load_eval(ckpts[step], mods.modules())
        ctx.start({"run": run_dir.name})
        for code in todo:
            t0 = time.time()
            if ctx.dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats(ctx.dev)
            stamp = {"step": step, "t": float(ckpts[step].stem), "code": code}
            inputs = _step_inputs(mods, code, ctx.sets, ctx.dev)
            ctx.write(f"{prefix}-{code}", inputs, stamp, t0)
    return ctx.out


def _yard_inputs(
    name: str,
    ctx: _Context,
    given: Mapping[str, Path | None],
    pca: dict[str, tuple[Tensor, Tensor]],
) -> Inputs:
    """The inputs of the yardstick ``name`` (None for ``mean``); the pixels'
    PCA kept in ``pca`` for the next ``pca<k>``."""
    sets, dev, conf = ctx.sets, ctx.dev, ctx.conf
    if name == "mean":
        return None
    if name == "inc64":
        v = ctx.targets.values["inc64"]
        return v["fit"].float().to(dev), v["grid"].float().to(dev)
    if name in ("color", "color8"):
        return _inputs(ColorCode(16 if name == "color" else 8).mu, sets, dev)
    if name.startswith("pca"):
        k = int(name[3:])
        if "all" not in pca:
            gen = torch.Generator().manual_seed(conf.seed + PCA_SEED)
            code = PCACode.fit(
                pixels, sets.fit, ctx.sconf.pca_max, gen=gen, device=conf.device
            )
            pca["all"] = _inputs(code.mu, sets, dev)
        x_fit, x_grid = pca["all"]
        return x_fit[:, :k], x_grid[:, :k]
    if name in ("v16", "v17"):
        path = given[name]
        assert path is not None
        return _inputs(vae_at(path, conf.device).mu, sets, dev)
    run = given["init_c"]
    assert run is not None
    r_conf = _conf_of(run, conf.device)
    mods = build_run_jepa(r_conf, json.loads((run / "jepa.json").read_text()))
    return _inputs((K3Code(mods) if name == "init_c_k3" else mods).mu, sets, dev)


def evaluate_yardsticks(
    out_dir: Path,
    *,
    codes: Sequence[str] = YARDSTICKS,
    conf: RunConf | None = None,
    v16: Path | None = None,
    v17: Path | None = None,
    init_run: Path | None = None,
    sconf: SemConf = SemConf(),
    root: Path | None = None,
    sem: Path | None = None,
    cache: Path | None = None,
    spacing: Mapping[str, Any] | None = None,
    span: float = SPAN,
) -> Path:
    """The yardsticks ``codes`` (``YARDSTICKS``) of ``conf`` (default
    ``v16_config`` on CUDA) into ``out_dir / OUT``; ``init_c`` and
    ``init_c_k3`` of the V18c run ``init_run`` at its step 0. Returns
    ``out_dir / OUT``.

    Raises:
        ValueError: A code is not of ``YARDSTICKS``, or its checkpoint or
            run is not given.
    """
    unknown = [c for c in codes if c not in YARDSTICKS]
    if unknown:
        raise ValueError(f"the yardsticks are of {YARDSTICKS}, not {unknown}")
    given = {"v16": v16, "v17": v17, "init_c": init_run, "init_c_k3": init_run}
    lacking = [c for c in codes if c in given and given[c] is None]
    if lacking:
        raise ValueError(f"the yardsticks {lacking} need their checkpoint or run")
    conf = _yard_conf(None) if conf is None else conf
    ctx = _Context(
        conf, out_dir / OUT, sconf=sconf, root=root, sem=sem, cache=cache,
        spacing=spacing, span=span,
    )  # fmt: skip
    pca: dict[str, tuple[Tensor, Tensor]] = {}
    for name in codes:
        path = ctx.out / "rows" / f"{YARD_PREFIX}-{name}.json"
        if path.exists():
            continue
        ctx.start({"yardsticks": list(codes)})
        t0 = time.time()
        stamp = {"step": None, "t": None, "code": name}
        ctx.write(
            f"{YARD_PREFIX}-{name}", _yard_inputs(name, ctx, given, pca), stamp, t0
        )
    return ctx.out


# -- the command line ------------------------------------------------------------------


def _ints(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split(","))


def _names(text: str) -> tuple[str, ...]:
    return tuple(text.split(","))


def main(argv: Sequence[str] | None = None) -> Path:
    """``python -m exp.vision.sem_eval`` (module docstring). Returns the
    output."""
    ap = argparse.ArgumentParser(
        prog="python -m exp.vision.sem_eval",
        description="V18c's main measure m: a linear readout of hires features.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    fr = sub.add_parser("frames", help="the frames the hires data must hold")
    fr.add_argument("--out", type=Path, required=True)
    tg = sub.add_parser("targets", help="the targets' cache (once)")
    tg.add_argument("--hires", type=Path, required=True)
    tg.add_argument("--inception", type=Path, help="FID's weights")
    yard = sub.add_parser("yardsticks", help="the yardsticks' rows")
    yard.add_argument("--out", type=Path, required=True)
    yard.add_argument("--codes", type=_names, default=YARDSTICKS)
    yard.add_argument("--v16", type=Path, help="a VAE checkpoint")
    yard.add_argument("--v17", type=Path, help="a VAE checkpoint")
    yard.add_argument("--init-run", type=Path, help="a V18c run (init_c)")
    run = sub.add_parser("run", help="the rows of a JEPA run's checkpoints")
    run.add_argument("run_dir", type=Path)
    run.add_argument("--steps", type=_ints, required=True)
    run.add_argument("--codes", type=_names, default=RUN_CODES)
    for p in (fr, tg, yard, run):
        p.add_argument("--root", type=Path, help="the data (default: tours_root)")
        p.add_argument("--device", help="(default: the run's, or cuda)")
    for p in (tg, yard, run):
        p.add_argument("--cache", type=Path, help="(default: $MUJOU_EXP_DATA)")
    for p in (yard, run):
        p.add_argument("--sem", type=Path, help="the targets (default: found)")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        return evaluate_run(
            args.run_dir,
            steps=args.steps,
            codes=args.codes,
            root=args.root,
            sem=args.sem,
            cache=args.cache,
            device=args.device,
        )
    conf = _yard_conf(args.device)
    if args.cmd == "frames":
        return evaluate_frames(args.out, conf=conf, root=args.root)
    if args.cmd == "targets":
        return evaluate_targets(
            args.hires, args.cache, conf=conf, root=args.root, inception=args.inception
        )
    return evaluate_yardsticks(
        args.out,
        codes=args.codes,
        conf=conf,
        v16=args.v16,
        v17=args.v17,
        init_run=args.init_run,
        root=args.root,
        sem=args.sem,
        cache=args.cache,
    )


if __name__ == "__main__":
    print(main())
