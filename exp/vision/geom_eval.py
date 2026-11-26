r"""V18c's time linearity and information width (``plans/jepa.md``,
「V18c: 時間の線形性と情報の幅」; README, the section of the same name,
registered before any number): of a code, the midpoint's error of the
triples $(z_{u-n}, z_u, z_{u+n})$ of Zurich in whitened coordinates against
$s(n)$, and the TwoNN of Zurich's points 8 s or more apart. A description,
not a verdict.

**The sets** (``geom_sets``, ``zurich_sets``): every stack of Zurich (0.1 s
apart, ``gen_eval.Zurich`` of step 1); its frames cut into blocks of
``GeomConf.block`` frames from the start (the remainder joined to the last,
``metrics.contiguous_blocks``); of each lag $n$ (``GeomConf.lags``, steps of
0.1 s) the centres $u$ whose three stacks' frames $u - n .. u + n + 7$ lie
in one block (``block_triples``). The cuts (``cut_rule``): between frames
$i$, $i + 1$ where their mean squared difference ($x / 255$) is at least the
1% quantile $\theta$ of the same of ``cut_pairs`` random pairs of frames
``cut_gap`` or more apart (seed ``cut_seed``). The strata of a lag: ``all``,
``in`` (no cut in the triple's frames), ``cross`` (one or more), and ``in``
split in three by the rank of the pixels' displacement $M = \tfrac12
(\lVert x_u - x_{u-n} \rVert^2 + \lVert x_{u+n} - x_u \rVert^2)$ (the
stacks' 8 frames, ``x / 255``): ``slow``, ``mid``, ``fast``. The TwoNN's
grid: every ``grid_step``-th stack (1 s), thinned to every $S$ s
(``GeomConf.spacings``) from each offset.

**The rows of a code** (``code_rows``), of each place (``own``, the model's
latent; ``h_mean``, the mean over the positions of the encoder's last
``LayerNorm``'s tokens, pooled in float64) and whitening (``exp.lib.midpoint.
Whitener`` of $k$, $c_\epsilon$ and the reference ``ref``: ``all`` the
5998 fit stacks of ``fill_eval.fill_sets``, ``front`` and ``back`` its
halves in the stream's order): ``spec`` (the spectrum; the only row of a
code not scoreable at $k$), ``spread`` ($V$, ``spread_diagnosis`` and its
blocks' sums $\sum w$, $\sum \lVert w \rVert^2$, counts), ``mid`` of each
lag and stratum ($G$,
$s$, $\bar D$, the means of $\lVert a \rVert^2$, $\lVert b \rVert^2$, and
the blocks' sums ``num_b``, ``den_b``, ``n_b``), ``twonn`` of each metric
(``pre`` the PCA coordinates $y$, ``white`` $w$), spacing and offset (and of
8 s the share of points whose neighbour is under ``near`` s away and the
estimate without such neighbours ``d_far``), and ``twonn_sub``: of 8 s, the
offsets' mean $\hat d$ on ``sub_draws`` half-samples of the blocks drawn
without replacement (seed ``sub_seed``, the same for every code: paired).
A synthetic control also has the rows of ``ref`` ``population``: whitened by
its population's covariance $I$ (``population_whitener``), its main
comparison with the theory; its reference's rows record how the
reference's sampling enters the metric (README, 「実装のレビューを受けた
直し」).

**The checks** (``check_rows``, V-d0a to V-d0f, synthetic, on the CPU) and
**the yard** (``evaluate_yard``: the sets, the yardsticks ``pca``,
``color``, ``mean8``, the step 0 of the JEPAs and of V16's VAE, the
synthetic controls ``pca_perm``, ``ou_mix_<c>``, ``cheb_a``, ``cheb_b``,
``cheb_c``, ``k3_z``, ``k3_hmean``, whether the step 0 of the runs of a seed
are the same (``step0_same``), and V-d0h, V-d1, V-d2 and the TwoNN's
resolution into ``vd.json``). No learned checkpoint is evaluated before the
user decides $\Delta_*$ (``run`` and ``vae`` are the later step; ``run``
also makes the K3's rows of a V18c checkpoint)::

    OMP_NUM_THREADS=1 python -m exp.vision.geom_eval checks --out runs/v18c-geom/checks
    python -m exp.vision.geom_eval yard --config <v18c run> --out runs/v18c-geom \
        --init <v18 run> --init-f <..> --init-m <..> --init-c <..> --init-c1 <..> \
        --init-v16 <v16 run> --same-c <the 6 runs of seed 0> \
        --same-c1 <the 6 runs of seed 1>
    python -m exp.vision.geom_eval run <run_dir> --steps 6000,60000 --sets runs/v18c-geom
    python -m exp.vision.geom_eval vae <ckpt> --name v16 --sets runs/v18c-geom
"""

import argparse
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Self

import torch
import yaml
from torch import Tensor

from exp.lib import ckpt, pca
from exp.lib.metrics import block_counts, contiguous_blocks, ratio_resamples
from exp.lib.midpoint import (
    Whitener,
    injection_for,
    permutation_terms,
    random_walk_terms,
    spread,
    stationary_terms,
    triple_terms,
)
from exp.lib.provenance import configure_determinism, of_diagnosis
from exp.lib.rundir import atomic_open
from exp.lib.synth_series import (
    chebyshev,
    gauss_kernel,
    kernel_acf,
    ou,
    random_walk,
    smooth,
    spectral,
    white,
)
from exp.lib.tours.stacks import tours_root
from exp.lib.twonn import TwoNN, thinned, twonn
from exp.vision.compose import from_dict
from exp.vision.eval import BATCH, PCA_SEED, stream_view
from exp.vision.fill_eval import FillConf, fill_sets
from exp.vision.gen_eval import Zurich
from exp.vision.jepa import JEPAModules
from exp.vision.jepa_diag import run_ckpts
from exp.vision.jepa_eval import (
    ZPCA_DIM,
    ColorCode,
    PCACode,
    build_run_jepa,
    pixels,
    vae_at,
)
from exp.vision.models import VAEModules, ViTPart, build_vae, early_fusion
from exp.vision.schema import RunConf
from exp.vision.self_eval import same_weights

__all__ = [
    "OUT",
    "PLACES",
    "STRATA",
    "CheckConf",
    "GeomConf",
    "GeomSets",
    "JepaPlaces",
    "K3Places",
    "VaeTokenCode",
    "block_triples",
    "check_rows",
    "code_rows",
    "cut_rule",
    "evaluate_checks",
    "evaluate_run",
    "evaluate_vae",
    "evaluate_yard",
    "geom_sets",
    "main",
    "ratio_se",
    "s_resamples",
    "zurich_sets",
]

OUT = "posthoc_geom"
PLACES = ("own", "h_mean")
STRATA = ("all", "in", "cross", "slow", "mid", "fast")
METRICS = ("pre", "white")
YARD_INITS = ("init", "init_f", "init_m", "init_c", "init_c1")
SAME_INITS = ("init_c", "init_c1")  # whose step 0 stands for a seed's runs
DIAG_TOP = 5  # the directions of spread_diagnosis's top_share

type Places = Callable[[Tensor], dict[str, Tensor]]
type Row = dict[str, Any]


@dataclass(frozen=True)
class GeomConf:
    r"""The settings (module docstring; ``plans/jepa.md``), by default the
    plan's. Lags and blocks in steps of 0.1 s (stacks, frames); ``near`` and
    ``spacings`` in seconds of the 1 s grid; ``ou_tau``, ``cheb_period`` in
    steps; a synthetic code's reference is drawn from its seed plus
    ``ref_shift`` (an independent realization of the same length)."""

    lags: tuple[int, ...] = (8, 16, 32, 64, 128, 256)
    block: int = 1280  # 128 s
    k: int = 64
    k_side: tuple[int, ...] = (16, 256)
    k_sens: tuple[int, ...] = (48, 96)
    c_eps: float = 1e-6
    c_eps_sens: float = 1e-3
    cut_gap: int = 6000  # 600 s
    cut_pairs: int = 20000
    cut_q: float = 0.01
    cut_seed: int = 401
    grid_step: int = 10  # stacks between the TwoNN grid's points (1 s)
    spacings: tuple[int, ...] = (8, 16, 32, 64)
    near: int = 64
    discard: float = 0.1
    floor: float = 1e-6
    boot: int = 2000
    boot_seed: int = 101
    sub_draws: int = 200
    sub_seed: int = 103
    perm_seed: int = 211
    ou_tau: float = 32.0  # 3.2 s
    mix_cs: tuple[int, ...] = (1, 3, 10, 30)
    mix_seed: int = 301
    cheb_period: float = 200.0  # the carrier's, 20 s
    cheb_seed: int = 503
    cheb_c_seed: int = 7
    ref_shift: int = 1000
    inject_seed: int = 701
    deltas: tuple[float, ...] = (0.005, 0.01, 0.02, 0.04, 0.08)
    tau_vd2: float = 4.71
    res_dims: tuple[int, ...] = (4, 8, 16, 32, 64)
    res_phis: tuple[float, ...] = (0.0, 0.5, 0.9)
    res_draws: int = 16
    res_seed: int = 801
    res_points: int = 487
    pca_dim: int = ZPCA_DIM  # pca's, mean8's and zpca's components


@dataclass(frozen=True)
class CheckConf:
    r"""The settings of V-d0a to V-d0f (``check_rows``; ``plans/jepa.md``):
    series of ``n`` steps with stacks ``span`` frames long cut by
    ``GeomConf.block``'s rule, ``d`` dimensions, ``draws`` realizations of
    each of V-d0b's processes (drawn by ``seed`` plus the process's place in
    order), its threshold ``tau``, and the constants of the processes and of
    TwoNN's recovery (``twonn_seeds``) and records."""

    n: int = 38989
    span: int = 7
    d: int = 64
    draws: int = 64
    tau: float = 4.26
    seed: int = 901
    gauss_sigma: float = 32.0  # 3.2 s
    band_period: float = 200.0  # omega_0 = 2 pi / 20 s
    band_width: float = 0.05
    second_band: float = 5.0
    rotation_period: int = 200
    rotation_length: int = 38800  # 194 periods
    affine_k: int = 16
    affine_seed: int = 23
    affine_cond: float = 1e3
    grid_every: int = 80  # 8 s
    twonn_points: int = 487
    twonn_seeds: tuple[int, ...] = tuple(range(601, 609))
    gauss_dims: tuple[int, ...] = (4, 8, 16, 32, 64)
    noise_levels: tuple[float, ...] = (0.1, 0.3, 1.0)


# -- the sets ----------------------------------------------------------------------


def block_triples(frame_block: Tensor, n_stacks: int, span: int, lag: int) -> Tensor:
    """The centres $u$ ``(T,)`` of the stacks ``0 .. n_stacks - 1`` (stack
    $u$ the frames $u .. u + \\text{span}$) whose triple of ``lag`` has its
    frames $u - n .. u + n + \\text{span}$ in one block (``frame_block``
    ``(F,)``, each frame's)."""
    u = torch.arange(lag, n_stacks - lag)
    return u[frame_block[u - lag] == frame_block[u + lag + span]]


def cut_rule(
    frames: Tensor, *, gap: int, pairs: int, q: float, seed: int, dev: torch.device
) -> tuple[Tensor, float]:
    r"""The mean squared difference $d_i$ ``(F - 1,)`` float64 of the frames
    $i$, $i + 1$ of ``frames`` u8 ``(F, 3, S, S)`` (``x / 255``) and the
    ``q`` quantile $\theta$ of the same of ``pairs`` pairs drawn uniformly
    among those ``gap`` or more frames apart (by ``seed``)."""

    def msd(a: Tensor, b: Tensor) -> Tensor:
        out = []
        for i in range(0, len(a), BATCH):
            x = frames[a[i : i + BATCH]].to(dev).float() / 255
            y = frames[b[i : i + BATCH]].to(dev).float() / 255
            out.append((x - y).square().flatten(1).mean(1).double().cpu())
        return torch.cat(out)

    f = len(frames)
    near = torch.arange(f - 1)
    d = msd(near, near + 1)
    gen = torch.Generator().manual_seed(seed)
    got: list[Tensor] = []
    have = 0
    while have < pairs:  # rejection: uniform over the pairs far enough apart
        ij = torch.randint(f, (2, 4 * pairs), generator=gen)
        ij = ij[:, (ij[0] - ij[1]).abs() >= gap]
        got.append(ij)
        have += ij.shape[1]
    i, j = torch.cat(got, 1)[:, :pairs]
    return d, float(torch.quantile(msd(i, j), q))


def _stack_diffs(
    frames: Tensor, lag: int, n: int, stride: int, dev: torch.device
) -> Tensor:
    r"""$\lVert x_{u + n} - x_u \rVert^2$ ``(U,)`` float64 of the stacks $u
    = 0 .. U - 1$ ($U$ = frames $- (n_f - 1)\,\text{stride} - $ ``lag``; a
    stack the ``n`` frames $u + j\,\text{stride}$, ``x / 255``): the sums
    of the frames' $e(i) = \lVert f_{i + n} - f_i \rVert^2$."""
    f = len(frames)
    e = []
    for i in range(0, f - lag, BATCH):
        idx = torch.arange(i, min(i + BATCH, f - lag))
        x = frames[idx].to(dev).float() / 255
        y = frames[idx + lag].to(dev).float() / 255
        e.append((y - x).square().flatten(1).sum(1).double().cpu())
    each = torch.cat(e)
    count = f - (n - 1) * stride - lag
    return sum(
        (each[j * stride : j * stride + count] for j in range(n)),
        torch.zeros(count, dtype=torch.float64),
    )


@dataclass(frozen=True)
class GeomSets:
    """The sets (module docstring): ``n`` stacks of ``span`` frames past
    the first, ``blocks`` ``(n,)`` each stack's block (its first frame's),
    ``triples`` the centres by lag and stratum, ``grid`` ``(G,)`` the stack
    of each point of the 1 s grid, ``cut_d`` ``(F - 1,)``, ``theta``,
    ``cuts`` the frames $i$ cut from $i + 1$, ``edges`` the strata's
    displacement bounds by lag, ``ident`` and its ``key``."""

    n: int
    span: int
    blocks: Tensor
    triples: Mapping[int, Mapping[str, Tensor]]
    grid: Tensor
    cut_d: Tensor
    theta: float
    cuts: Tensor
    edges: Mapping[int, list[float]]
    ident: Mapping[str, Any]
    key: str

    @property
    def n_blocks(self) -> int:
        return int(self.blocks.max()) + 1

    def save(self, out: Path) -> None:
        """``out / "sets.pt"`` and ``out / "sets.json"`` (the record)."""
        out.mkdir(parents=True, exist_ok=True)
        with atomic_open(out / "sets.pt") as f:
            torch.save(asdict(self), f)
        record = {
            "key": self.key,
            "ident": dict(self.ident),
            "theta": self.theta,
            "cuts": self.cuts.tolist(),
            "edges": {str(k): v for k, v in self.edges.items()},
            "counts": {
                str(lag): {s: len(c) for s, c in by.items()}
                for lag, by in self.triples.items()
            },
        }
        with atomic_open(out / "sets.json") as f:
            f.write((json.dumps(record, indent=1) + "\n").encode())

    @classmethod
    def load(cls, out: Path) -> Self:
        return cls(**torch.load(out / "sets.pt", weights_only=False))


def _key(
    ident: Mapping[str, Any], triples: Mapping[int, Mapping[str, Tensor]], cuts: Tensor
) -> str:
    h = hashlib.sha256(json.dumps(dict(ident), sort_keys=True).encode())
    for lag in sorted(triples):
        for s in STRATA:
            h.update(f"{lag}:{s}".encode() + triples[lag][s].numpy().tobytes())
    h.update(cuts.numpy().tobytes())
    return h.hexdigest()[:16]


def geom_sets(
    frames: Tensor,
    *,
    stack: int,
    stride: int,
    gconf: GeomConf,
    dev: torch.device,
    ident: Mapping[str, Any],
) -> GeomSets:
    """The sets (module docstring) of the video's ``frames`` u8 ``(F, 3, S,
    S)`` and its stacks of ``stack`` frames ``stride`` apart, one per frame
    (``ident`` names the video)."""
    f = len(frames)
    span = (stack - 1) * stride
    n = f - span
    frame_block = contiguous_blocks(torch.zeros(f, dtype=torch.long), gconf.block)
    cut_d, theta = cut_rule(
        frames, gap=gconf.cut_gap, pairs=gconf.cut_pairs, q=gconf.cut_q,
        seed=gconf.cut_seed, dev=dev,
    )  # fmt: skip
    cut = cut_d >= theta
    before = torch.cat([torch.zeros(1, dtype=torch.long), cut.long().cumsum(0)])
    triples: dict[int, dict[str, Tensor]] = {}
    edges: dict[int, list[float]] = {}
    for lag in gconf.lags:
        u = block_triples(frame_block, n, span, lag)
        # cuts between frames i, i + 1 for i in [u - n, u + n + span - 1]
        crossed = before[u + lag + span] - before[u - lag] > 0
        diffs = _stack_diffs(frames, lag, stack, stride, dev)
        m = (diffs[u - lag] + diffs[u]) / 2
        inside = u[~crossed]
        order = torch.sort(m[~crossed], stable=True)
        rank = torch.empty_like(order.indices)
        rank[order.indices] = torch.arange(len(rank))
        third = 3 * rank // max(len(rank), 1)
        triples[lag] = {
            "all": u,
            "in": inside,
            "cross": u[crossed],
            "slow": inside[third == 0],
            "mid": inside[third == 1],
            "fast": inside[third == 2],
        }
        vals = order.values
        edges[lag] = (
            [float(vals[len(vals) // 3]), float(vals[2 * len(vals) // 3])]
            if len(vals) >= 3
            else []
        )
    full = dict(ident) | {"frames": f, "stack": stack, "stride": stride, "theta": theta}
    full |= {"geom": asdict(gconf)}
    cuts = cut.nonzero().flatten()
    return GeomSets(
        n=n,
        span=span,
        blocks=frame_block[:n].clone(),
        triples=triples,
        grid=torch.arange(0, n, gconf.grid_step),
        cut_d=cut_d,
        theta=theta,
        cuts=cuts,
        edges=edges,
        ident=json.loads(json.dumps(full)),
        key=_key(full, triples, cuts),
    )


def zurich_sets(conf: RunConf, root: Path, gconf: GeomConf) -> tuple[GeomSets, Zurich]:
    """The sets of ``conf``'s held-out video at ``root`` and its stacks."""
    t = conf.data.tours
    zurich = Zurich(conf, root, 1)
    ident = {"data_version": t.data_version, "held_out": t.held_out}
    sets = geom_sets(
        zurich.frames, stack=t.frames, stride=t.stride, gconf=gconf,
        dev=torch.device(conf.device), ident=ident,
    )  # fmt: skip
    return sets, zurich


# -- the codes ---------------------------------------------------------------------


def _pooled(h: Tensor) -> Tensor:
    """The mean over the positions of ``h`` ``(B, T, w)``, float64 ``(B,
    w)`` (float64 holds the sum of $T$ float32 tokens exactly unless their
    exponents spread over more than $29 - \\log_2 T$ bits)."""
    return h.double().mean(1)


class VaeTokenCode:
    r"""The VAE's places in one pass: ``own`` $\mu$ and ``h_mean``, the
    tokens of the encoder's last node (``ViTPart``) after its ``norm``,
    before ``head``, averaged over the positions."""

    def __init__(self, mods: VAEModules) -> None:
        self._mods = mods

    def places(self, x: Tensor) -> dict[str, Tensor]:
        parts = self._mods.encoder.parts
        y = early_fusion(x).to(self._mods.log_sigma.dtype)
        for part in parts[:-1]:
            y = part(y)
        last = parts[-1]
        assert (
            isinstance(last, ViTPart)
            and last.norm is not None
            and last.head is not None
        )
        if last.embed is not None:
            y = last.embed(y)
        for block in last.blocks:
            y = block(y)
        h = last.norm(y)
        out = last.head(h.flatten(1))
        return {"own": out[:, : out.shape[1] // 2], "h_mean": _pooled(h)}


class JepaPlaces:
    r"""A JEPA's places in one pass: ``own`` its $z$ (``JEPAModules.mu``:
    the flat tokens, or $W \mathrm{vec}(t)$ with ``mix``) and ``h_mean``."""

    def __init__(self, mods: JEPAModules) -> None:
        self._mods = mods

    def places(self, x: Tensor) -> dict[str, Tensor]:
        h = self._mods.encoder(early_fusion(x))
        flat = self._mods.proj(h).flatten(1)
        z = flat if self._mods.mix is None else self._mods.mix(flat)
        return {"own": z, "h_mean": _pooled(h)}


def _tree_mean(x: Tensor) -> Tensor:
    """The mean over dim 1 of ``x`` ``(B, T, w)`` float64 summed in pairs
    (``T`` a power of 2): of $T$ copies of a value every partial sum is a
    power of 2 times it, so the mean is the value to the bit (a summation
    of another order rounds $3v$, $5v$, ...)."""
    t = x.shape[1]
    if t & (t - 1):
        raise ValueError(f"the pairwise mean needs T a power of 2, not {t}")
    while x.shape[1] > 1:
        x = x[:, 0::2] + x[:, 1::2]
    return x[:, 0] / t


class K3Places:
    r"""The replicated K3 of a V18c JEPA (``sem_eval.K3Code``'s): ``own``
    $W \mathrm{vec}(1 \otimes \bar t)$ (``k3_z``) and ``h_mean`` the same
    replacement of $h$ before its mean (``k3_hmean``). The replacement of
    $h$ is of its float64 mean (``_pooled``), its mean ``_tree_mean``: the
    K3's ``h_mean`` is the plain one to the bit by arithmetic, so V-d0h can
    fail only where two passes of the encoder differ, never by rounding."""

    def __init__(self, mods: JEPAModules) -> None:
        if mods.mix is None:
            raise ValueError("k3_z is of a JEPA with W (V18c's sigreg_on mix)")
        self._mods, self._mix = mods, mods.mix

    def places(self, x: Tensor) -> dict[str, Tensor]:
        h = self._mods.encoder(early_fusion(x))
        t = self._mods.proj(h)
        z = self._mix(t.mean(1, keepdim=True).expand_as(t).flatten(1))
        hbar = _pooled(h)[:, None].expand(-1, h.shape[1], -1)
        return {"own": z, "h_mean": _tree_mean(hbar)}


@torch.no_grad()
def encode_places(
    fn: Places, stacks: Callable[[Tensor], Tensor], n: int, dev: torch.device
) -> dict[str, Tensor]:
    """The places of ``fn`` of the ``n`` stacks ``stacks(idx)`` u8 (CPU),
    ``BATCH`` at a time, computed on ``dev`` and returned on the CPU
    (float32, or float64 as given)."""
    got: dict[str, list[Tensor]] = {}
    for i in range(0, n, BATCH):
        idx = torch.arange(i, min(i + BATCH, n))
        for name, v in fn(stacks(idx).to(dev)).items():
            got.setdefault(name, []).append(
                v.cpu() if v.dtype == torch.float64 else v.float().cpu()
            )
    return {k: torch.cat(v) for k, v in got.items()}


# -- the bootstrap -----------------------------------------------------------------


def _block_sums(values: Tensor, blocks: Tensor, n_blocks: int) -> Tensor:
    """The sums ``(n_blocks, ...)`` float64 of ``values`` ``(N, ...)`` by
    ``blocks`` ``(N,)``."""
    v = values.double()
    out = torch.zeros(n_blocks, *v.shape[1:], dtype=torch.float64)
    return out.index_add_(0, blocks, v)


def _paired(b_sums: Tensor) -> Tensor:
    """The sums ``(..., B // 2)`` of adjacent blocks ``(..., B)`` (256 s of
    128 s; an odd last joined to the pair before it)."""
    b = b_sums.shape[-1]
    to = torch.arange(b).div(2, rounding_mode="floor").clamp(max=b // 2 - 1)
    out = b_sums.new_zeros(*b_sums.shape[:-1], b // 2)
    return out.index_add_(-1, to, b_sums)


def _counts(n_blocks: int, gconf: GeomConf) -> Tensor:
    return block_counts(
        n_blocks, gconf.boot, torch.Generator().manual_seed(gconf.boot_seed)
    )


def ratio_se(num_b: Tensor, den_b: Tensor, contrast: Tensor, gconf: GeomConf) -> float:
    r"""$\mathrm{SE}^*$ of the contrast ``(C,)`` of $C$ codes' ratios $\sum
    \mathrm{num} / \sum \mathrm{den}$ of the blocks' sums ``(C, B)``: the
    larger of the block bootstraps of the blocks and of their adjacent pairs
    (``metrics.ratio_resamples``, ``boot`` resamples of ``boot_seed``, the
    same draws for every code). A resample of no triple of a code (a
    denominator 0, as in a stratum of few blocks) has no ratio: the standard
    deviation is of the others (conditional on a triple; NaN of fewer than
    2)."""
    sds = []
    for n_, d_ in ((num_b, den_b), (_paired(num_b), _paired(den_b))):
        r = ratio_resamples(n_, d_, _counts(n_.shape[1], gconf))
        sds.append(_finite_std(r @ contrast.double()))
    return _larger(sds)


def _finite_std(r: Tensor) -> float:
    """The standard deviation of the finite values of ``r`` (NaN of fewer
    than 2)."""
    ok = r[torch.isfinite(r)]
    return float(ok.std()) if len(ok) >= 2 else math.nan


def _larger(sds: Sequence[float]) -> float:
    """The largest of ``sds``, NaN if one is (``max`` would drop it)."""
    return math.nan if any(math.isnan(x) for x in sds) else max(sds)


def s_resamples(
    den_b: Tensor, n_b: Tensor, w_b: Tensor, ww_b: Tensor, m_b: Tensor, counts: Tensor
) -> Tensor:
    r"""The resampled $s = \bar D / (2V)$ ``(n,)`` of one code: the triples'
    ``den_b`` and counts ``n_b`` ``(B,)``, the stacks' $\sum w$ ``w_b`` ``(B,
    k)``, $\sum \lVert w \rVert^2$ ``ww_b`` and counts ``m_b`` ``(B,)``, and
    the ``counts`` ``(n, B)`` of the draws."""
    c = counts.double()
    mean_d = (c @ den_b.double()) / (c @ n_b.double())
    m = c @ m_b.double()
    mean_w = (c @ w_b.double()) / m[:, None]
    v = (c @ ww_b.double()) / m - mean_w.square().sum(1)
    return mean_d / (2 * v)


def _f64(values: Any) -> Tensor:
    return torch.tensor(values, dtype=torch.float64)


def s_se(
    spread_row: Mapping[str, Any], mid_row: Mapping[str, Any], gconf: GeomConf
) -> float:
    """$\\mathrm{SE}^*$ of $s$ of a ``mid`` row and its ``spread`` row (a
    resample of no triple dropped, as in ``ratio_se``)."""
    den_b, n_b = _f64(mid_row["den_b"]), _f64(mid_row["n_b"])
    w_b = _f64(spread_row["w_b"])  # (B, k)
    ww_b, m_b = _f64(spread_row["ww_b"]), _f64(spread_row["m_b"])
    sds = []
    for paired in (False, True):
        if paired:
            den_b, n_b, ww_b, m_b = map(_paired, (den_b, n_b, ww_b, m_b))
            w_b = _paired(w_b.T).T
        r = s_resamples(den_b, n_b, w_b, ww_b, m_b, _counts(len(den_b), gconf))
        sds.append(_finite_std(r))
    return _larger(sds)


# -- the rows of a code --------------------------------------------------------------


@dataclass(frozen=True)
class Variant:
    """A whitening: ``k``, ``c_eps``, the reference ``ref``, and whether its
    TwoNN rows are made."""

    k: int
    c_eps: float
    ref: str
    twonn: bool


def variants(gconf: GeomConf, *, sens: bool) -> list[Variant]:
    """The main whitening and $k$ of ``k_side`` (with TwoNN), and with
    ``sens`` (the yardsticks of V-d3) $k$ of ``k_sens``, ``c_eps_sens`` and
    the reference's halves."""
    out = [Variant(gconf.k, gconf.c_eps, "all", True)]
    out += [Variant(k, gconf.c_eps, "all", True) for k in gconf.k_side]
    if sens:
        out += [Variant(k, gconf.c_eps, "all", False) for k in gconf.k_sens]
        out += [Variant(gconf.k, gconf.c_eps_sens, "all", False)]
        out += [Variant(gconf.k, gconf.c_eps, r, False) for r in ("front", "back")]
    return out


def _refs(z_ref: Tensor) -> dict[str, Tensor]:
    half = len(z_ref) // 2
    return {"all": z_ref, "front": z_ref[:half], "back": z_ref[half : 2 * half]}


def spread_diagnosis(w: Tensor, top: int = DIAG_TOP) -> dict[str, float]:
    r"""Of the whitened evaluation ``w`` ``(n, k)``: $V / k$, the largest
    variance of a direction ``var_max`` (the evaluation's own principal
    directions in $w$) and the share of $V$ of the ``top`` largest
    ``top_share``. A reference like the evaluation gives $V/k \approx 1$,
    ``var_max`` $\approx 1$ and ``top_share`` $\approx \text{top}/k$; far
    from these, the metric follows where the reference and the evaluation
    differ (README, 「言葉づかいの規則」)."""
    x = w.double() - w.double().mean(0)
    var = torch.linalg.eigvalsh(x.T @ x / len(x)).flip(0).clamp(min=0)
    total = float(var.sum())
    return {
        "V_over_k": total / w.shape[1],
        "var_max": float(var[0]),
        "top_share": float(var[:top].sum()) / total if total > 0 else math.nan,
    }


def mid_rows(w: Tensor, sets: GeomSets) -> tuple[Row, list[Row]]:
    """The ``spread`` row (with ``spread_diagnosis``) and the ``mid`` rows
    (module docstring) of the whitened codes ``w`` ``(n, k)`` of the sets'
    stacks."""
    nb = sets.n_blocks
    v = spread(w)
    spread_row = {
        "part": "spread",
        "V": v,
        **spread_diagnosis(w),
        "w_b": _block_sums(w, sets.blocks, nb).tolist(),
        "ww_b": _block_sums(w.double().square().sum(1), sets.blocks, nb).tolist(),
        "m_b": _block_sums(torch.ones(len(w)), sets.blocks, nb).tolist(),
    }
    rows = []
    for lag, by in sets.triples.items():
        t_all = triple_terms(w, by["all"], lag)
        for stratum in STRATA:
            centers = by[stratum]
            pick = torch.searchsorted(by["all"], centers)  # each a subset of all
            num, den = t_all.num[pick], t_all.den[pick]
            blocks = sets.blocks[centers - lag]
            n_t = len(centers)
            row: Row = {"part": "mid", "lag": lag, "stratum": stratum, "n_triples": n_t,
                        "n_blocks": len(blocks.unique())}  # fmt: skip
            if n_t:
                row |= {
                    "G": float(num.sum() / den.sum()),
                    "s": float(den.mean()) / (2 * v),
                    "D": float(den.mean()),
                    "a2": float(t_all.a2[pick].mean()),
                    "b2": float(t_all.b2[pick].mean()),
                }
            row |= {
                "V": v,
                "num_b": _block_sums(num, blocks, nb).tolist(),
                "den_b": _block_sums(den, blocks, nb).tolist(),
                "n_b": _block_sums(torch.ones(n_t), blocks, nb).tolist(),
            }
            rows.append(row)
    return spread_row, rows


def _radius(points: Tensor) -> float:
    p = points.double()
    return math.sqrt(2 * float((p - p.mean(0)).square().sum(1).mean()))


def _estimate(est: TwoNN) -> dict[str, Any]:
    return {
        "N": est.n, "n_kept": est.n_kept, "n_dup": est.n_dup, "n_exact": est.n_exact, "d": est.d,
        "d_ml": est.d_ml, "rho": est.rho, "tie_share": est.tie_share,
    }  # fmt: skip


def twonn_rows(
    points: Tensor,
    sets: GeomSets,
    gconf: GeomConf,
    *,
    sub: bool,
    times: Tensor | None = None,
) -> list[Row]:
    """The ``twonn`` rows of the grid's ``points`` ``(G, k)`` of one metric
    (module docstring) and with ``sub`` its ``twonn_sub`` row; ``times``
    ``(G,)`` the points' times in steps of the grid (by default their
    places, ``arange(G)``; a permuted code's, its stacks' own)."""
    radius = _radius(points)
    if times is None:
        times = torch.arange(len(points), dtype=torch.float64)
    rows: list[Row] = []
    for spacing in gconf.spacings:
        for offset, idx in enumerate(thinned(len(points), spacing)):
            est = twonn(
                points[idx], radius=radius, discard=gconf.discard, floor=gconf.floor
            )
            row: Row = {
                "part": "twonn",
                "spacing": spacing,
                "offset": offset,
            } | _estimate(est)
            if spacing == gconf.spacings[0] and est.d is not None:
                t = times[idx]
                close = (t[:, None] - t[None]).abs() < gconf.near
                nn_t = t[est.nn]  # (N', 2)
                own_t = t[est.kept][:, None]
                row["near_share"] = float(
                    ((nn_t - own_t).abs() < gconf.near).any(1).double().mean()
                )
                far = twonn(points[idx], radius=radius, discard=gconf.discard,
                            floor=gconf.floor, exclude=close)  # fmt: skip
                row["d_far"] = far.d
            rows.append(row)
    if sub:
        rows.append({"part": "twonn_sub", "spacing": gconf.spacings[0],
                     "d": _sub_estimates(points, sets, radius, gconf)})  # fmt: skip
    return rows


def _sub_estimates(
    points: Tensor, sets: GeomSets, radius: float, gconf: GeomConf
) -> list[float | None]:
    """The offsets' mean $\\hat d$ of the first spacing on each of
    ``sub_draws`` half-samples of the blocks (without replacement, by
    ``sub_seed``: the same draws for every code)."""
    blocks = sets.blocks[sets.grid]
    nb = sets.n_blocks
    gen = torch.Generator().manual_seed(gconf.sub_seed)
    sets_s = thinned(len(points), gconf.spacings[0])
    out: list[float | None] = []
    for _ in range(gconf.sub_draws):
        chosen = torch.zeros(nb, dtype=torch.bool)
        chosen[torch.randperm(nb, generator=gen)[: nb // 2]] = True
        ds = []
        for idx in sets_s:
            keep = idx[chosen[blocks[idx]]]
            ds.append(
                twonn(
                    points[keep],
                    radius=radius,
                    discard=gconf.discard,
                    floor=gconf.floor,
                ).d
            )
        out.append(
            None
            if any(d is None for d in ds)
            else sum(d for d in ds if d is not None) / len(ds)
        )
    return out


def population_whitener(d: int, c_eps: float) -> Whitener:
    r"""The whitening of a synthetic control by its population: mean 0 and
    covariance $I_d$ (independent columns of unit variance), all $k = d$
    directions, $\epsilon = c_\epsilon \mathrm{tr}(I)/d = c_\epsilon$."""
    f64 = torch.float64
    return Whitener(
        mean=torch.zeros(d, dtype=f64),
        basis=torch.eye(d, dtype=f64),
        values=torch.ones(d, dtype=f64),
        eps=c_eps,
        total=float(d),
        k=d,
        why=None,
    )


def _whitened_rows(
    wh: Whitener,
    z_eval: Tensor,
    sets: GeomSets,
    gconf: GeomConf,
    lab: Mapping[str, Any],
    *,
    twonn: bool,
    sub: bool,
    times: Tensor | None,
) -> list[Row]:
    rows: list[Row] = [dict(lab) | {"part": "spec"} | wh.spec()]
    if not wh.scoreable:
        return rows
    w = wh.white(z_eval)
    spread_row, mids = mid_rows(w, sets)
    rows += [dict(lab) | spread_row] + [dict(lab) | r for r in mids]
    if twonn:
        for metric, x in (("pre", wh.pca(z_eval)), ("white", w)):
            got = twonn_rows(x[sets.grid], sets, gconf, sub=sub, times=times)
            rows += [dict(lab) | {"metric": metric} | r for r in got]
    return rows


def code_rows(
    z_eval: Tensor,
    z_ref: Tensor,
    sets: GeomSets,
    gconf: GeomConf,
    *,
    sens: bool,
    labels: Mapping[str, Any],
    population: bool = False,
    times: Tensor | None = None,
) -> list[Row]:
    """The rows (module docstring) of a code's evaluation ``z_eval`` ``(n,
    D)`` (the sets' stacks) and reference ``z_ref`` ``(m, D)``, each with
    ``labels``, its ``k``, ``eps`` ($c_\\epsilon$) and ``ref``. With
    ``population`` (a synthetic control) also the rows of ``ref``
    ``population`` (``population_whitener``, $k = D$, with the TwoNN and its
    ``twonn_sub``); ``times`` the grid's times for ``twonn_rows``."""
    rows: list[Row] = []
    refs = _refs(z_ref)
    for v in variants(gconf, sens=sens):
        wh = Whitener.fit(refs[v.ref], v.k, c_eps=v.c_eps)
        lab = dict(labels) | {"k": v.k, "eps": v.c_eps, "ref": v.ref, "key": sets.key}
        rows += _whitened_rows(wh, z_eval, sets, gconf, lab, twonn=v.twonn,
                               sub=v.k == gconf.k, times=times)  # fmt: skip
    if population:
        d = z_eval.shape[1]
        lab = dict(labels) | {"k": d, "eps": gconf.c_eps, "ref": "population",
                              "key": sets.key}  # fmt: skip
        rows += _whitened_rows(population_whitener(d, gconf.c_eps), z_eval, sets,
                               gconf, lab, twonn=True, sub=True, times=times)  # fmt: skip
    return rows


# -- the synthetic controls --------------------------------------------------------


def mix_series(n: int, d: int, c: int, gconf: GeomConf, gen: torch.Generator) -> Tensor:
    r"""``ou_mix_c`` ``(n, d)``: $d/2$ OU columns of $\tau$ = ``ou_tau`` and
    $d/2$ of the Gaussian kernel of $\sigma = c\,\tau$."""
    r = math.exp(-1 / gconf.ou_tau)
    return torch.cat(
        [
            ou(n, d // 2, r, gen),
            smooth(n, d - d // 2, gauss_kernel(c * gconf.ou_tau), gen),
        ],
        1,
    )


def cheb_js(name: str, gconf: GeomConf) -> Tensor:
    """The integers $j$ of ``cheb_a`` ($1 .. 64$), ``cheb_b`` ($1000 ..
    1063$) or ``cheb_c`` (64 drawn from $[10^8, 10^8 + 10^6)$, seed
    ``cheb_c_seed``)."""
    if name == "cheb_a":
        return torch.arange(1, 65)
    if name == "cheb_b":
        return torch.arange(1000, 1064)
    g = torch.Generator().manual_seed(gconf.cheb_c_seed)
    return torch.randperm(10**6, generator=g)[:64] + 10**8


def cheb_series(name: str, n: int, seed: int, gconf: GeomConf) -> Tensor:
    r"""$\sqrt2 \cos(j \theta_t)$, $\theta_t = \Theta + \omega t$, $\omega =
    2\pi / (\text{period} \cdot \bar j)$, $\Theta$ uniform (by ``seed``)."""
    js = cheb_js(name, gconf)
    omega = 2 * math.pi / (gconf.cheb_period * float(js.double().mean()))
    theta0 = (
        2
        * math.pi
        * float(
            torch.rand(
                1, generator=torch.Generator().manual_seed(seed), dtype=torch.float64
            )
        )
    )
    return chebyshev(theta0 + omega * torch.arange(n, dtype=torch.float64), js)


def synthetic_codes(n: int, gconf: GeomConf) -> dict[str, tuple[Tensor, Tensor]]:
    """The synthetic controls' evaluation and reference ``(n, 64)`` each:
    ``ou_mix_<c>`` (seed ``mix_seed`` + $c$) and ``cheb_a``, ``cheb_b``,
    ``cheb_c`` (seed ``cheb_seed``); each reference of its seed plus
    ``ref_shift``."""
    out: dict[str, tuple[Tensor, Tensor]] = {}
    for c in gconf.mix_cs:
        seed = gconf.mix_seed + c
        ev = mix_series(n, 64, c, gconf, torch.Generator().manual_seed(seed))
        ref = mix_series(
            n, 64, c, gconf, torch.Generator().manual_seed(seed + gconf.ref_shift)
        )
        out[f"ou_mix_{c}"] = (ev, ref)
    for name in ("cheb_a", "cheb_b", "cheb_c"):
        out[name] = (
            cheb_series(name, n, gconf.cheb_seed, gconf),
            cheb_series(name, n, gconf.cheb_seed + gconf.ref_shift, gconf),
        )
    return out


# -- the checks V-d0a to V-d0f -------------------------------------------------------


def _check(name: str, ok: bool, **values: Any) -> Row:
    return {"part": "check", "name": name, "pass": bool(ok)} | values


def _pooled_gs(w: Tensor, centers: Tensor, lag: int) -> tuple[float, float]:
    t = triple_terms(w, centers, lag)
    return float(t.num.sum() / t.den.sum()), float(t.den.mean()) / (2 * spread(w))


def _close(a: float, b: float, rel: float) -> bool:
    return abs(a - b) <= rel * max(abs(a), abs(b))


def _processes(
    cc: CheckConf, gconf: GeomConf
) -> list[
    tuple[
        str, Callable[[torch.Generator], Tensor], Callable[[int], tuple[float, float]]
    ]
]:
    """V-d0b's processes: their name, their draw and the closed form of
    the mean per triple of $\\nu$ and $\\delta$ at a lag. ``two_bands``'s
    second band is the first's 32 frequencies times ``second_band``: centred
    on $5\\omega_0$ and 5 times as wide (the plan does not say its width;
    the closed form is of the same frequencies, so the check holds either
    way)."""
    n, d = cc.n, cc.d
    r = math.exp(-1 / gconf.ou_tau)

    def per_dim(
        rho: Callable[[int], float], dims: int
    ) -> Callable[[int], tuple[float, float]]:
        def terms(lag: int) -> tuple[float, float]:
            num, den = stationary_terms(torch.tensor(rho(lag), dtype=torch.float64),
                                        torch.tensor(rho(2 * lag), dtype=torch.float64))  # fmt: skip
            return dims * float(num), dims * float(den)

        return terms

    def acf(kernel: Tensor) -> Callable[[int], float]:
        return lambda lag: float(kernel_acf(kernel, torch.tensor([lag]))[0])

    def add(
        *parts: Callable[[int], tuple[float, float]],
    ) -> Callable[[int], tuple[float, float]]:
        return lambda lag: tuple(map(sum, zip(*(p(lag) for p in parts), strict=True)))  # type: ignore[return-value]

    def waves(omegas: Tensor) -> Callable[[int], tuple[float, float]]:
        def terms(lag: int) -> tuple[float, float]:
            num, den = stationary_terms(
                torch.cos(omegas * lag), torch.cos(omegas * 2 * lag)
            )
            return float(num.sum()), float(den.sum())

        return terms

    out: list[
        tuple[
            str,
            Callable[[torch.Generator], Tensor],
            Callable[[int], tuple[float, float]],
        ]
    ] = [
        ("white", lambda g: white(n, d, g), per_dim(lambda lag: 0.0, d)),
        ("walk", lambda g: random_walk(n, d, g), lambda lag: random_walk_terms(lag, d)),
        ("ou", lambda g: ou(n, d, r, g), per_dim(lambda lag: r**lag, d)),
    ]
    k = gauss_kernel(cc.gauss_sigma)
    out.append(("gauss", lambda g: smooth(n, d, k, g), per_dim(acf(k), d)))
    for c in gconf.mix_cs:
        kc = gauss_kernel(c * gconf.ou_tau)
        out.append((f"ou_mix_{c}", lambda g, c=c: mix_series(n, d, c, gconf, g),
                    add(per_dim(lambda lag: r**lag, d // 2), per_dim(acf(kc), d - d // 2))))  # fmt: skip
    w0 = 2 * math.pi / cc.band_period
    band = w0 * torch.linspace(
        1 - cc.band_width, 1 + cc.band_width, d, dtype=torch.float64
    )
    half = w0 * torch.linspace(
        1 - cc.band_width, 1 + cc.band_width, d // 2, dtype=torch.float64
    )
    two = torch.cat([half, cc.second_band * half])
    out.append(("band", lambda g: spectral(n, band, g), waves(band)))
    out.append(("two_bands", lambda g: spectral(n, two, g), waves(two)))
    out.append(("band_noise", lambda g: torch.cat([spectral(n, half, g), white(n, d - d // 2, g)], 1),
                add(waves(half), per_dim(lambda lag: 0.0, d - d // 2))))  # fmt: skip
    return out


def check_rows(cc: CheckConf = CheckConf(), gconf: GeomConf = GeomConf()) -> list[Row]:
    """V-d0a to V-d0f (``plans/jepa.md``; module docstring), synthetic, on
    the CPU: rows of ``part`` ``check`` (a failure stops) and ``record``."""
    rows: list[Row] = []
    frame_block = contiguous_blocks(
        torch.zeros(cc.n + cc.span, dtype=torch.long), gconf.block
    )
    centers = {
        lag: block_triples(frame_block, cc.n, cc.span, lag) for lag in gconf.lags
    }
    rows += _vd0a(cc, gconf, centers)
    rows += _vd0b(cc, gconf, centers)
    rows += _vd0c(cc, gconf, centers)
    rows += _vd0d(cc, gconf, centers)
    rows += _vd0e(cc, gconf)
    rows += _vd0f(cc, gconf)
    return rows


def _vd0a(cc: CheckConf, gconf: GeomConf, centers: Mapping[int, Tensor]) -> list[Row]:
    g = torch.Generator().manual_seed(cc.seed - 1)
    b = torch.randn(cc.d, generator=g, dtype=torch.float64)
    line = torch.arange(cc.n, dtype=torch.float64)[:, None] * b
    rows = []
    for lag, u in centers.items():
        gm, _ = _pooled_gs(line, u, lag)
        rows.append(_check("V-d0a-line", abs(gm) <= 1e-12, lag=lag, G=gm))
    omega = 2 * math.pi / cc.rotation_period
    phase = torch.arange(4, dtype=torch.float64) * math.pi / 2
    wh = Whitener.fit(torch.stack([phase.cos(), phase.sin()], 1), 2, c_eps=gconf.c_eps)
    t = torch.arange(cc.rotation_length, dtype=torch.float64) * omega
    w = wh.white(torch.stack([t.cos(), t.sin()], 1))
    n = cc.rotation_length
    for lag, u in centers.items():
        u = u[u < n - lag]
        terms = triple_terms(w, u, lag)
        c = math.cos(omega * lag)
        per = terms.num / terms.den
        worst = float((per - (1 - c) / 2).abs().max()) / ((1 - c) / 2)
        _, s = _pooled_gs(w, u, lag)
        ok = worst <= 1e-10 and _close(s, 1 - c, 1e-10)
        rows.append(
            _check("V-d0a-rotation", ok, lag=lag, worst_rel=worst, s=s, s_theory=1 - c)
        )
    return rows


def _vd0b(cc: CheckConf, gconf: GeomConf, centers: Mapping[int, Tensor]) -> list[Row]:
    rows = []
    procs = _processes(cc, gconf)
    for i, (name, draw, theory) in enumerate(procs):
        g = torch.Generator().manual_seed(cc.seed + i)
        means = {lag: ([], []) for lag in centers}
        for _ in range(cc.draws):
            x = draw(g)
            for lag, u in centers.items():
                t = triple_terms(x, u, lag)
                means[lag][0].append(float(t.num.mean()))
                means[lag][1].append(float(t.den.mean()))
        rows += _t_rows(name, means, theory, cc)
    # the permutation: one OU code, its order drawn again each time
    g = torch.Generator().manual_seed(cc.seed + len(procs))
    x = ou(cc.n, cc.d, math.exp(-1 / gconf.ou_tau), g)
    var = spread(x)
    means = {lag: ([], []) for lag in centers}
    for _ in range(cc.draws):
        xp = x[torch.randperm(cc.n, generator=g)]
        for lag, u in centers.items():
            t = triple_terms(xp, u, lag)
            means[lag][0].append(float(t.num.mean()))
            means[lag][1].append(float(t.den.mean()))
    rows += _t_rows("permutation", means, lambda lag: permutation_terms(var, cc.n), cc)
    return rows


def _t_rows(name: str, means: Mapping[int, tuple[list[float], list[float]]],
            theory: Callable[[int], tuple[float, float]], cc: CheckConf) -> list[Row]:  # fmt: skip
    rows = []
    for lag, (nums, dens) in means.items():
        want = theory(lag)
        for which, got, w in (("num", nums, want[0]), ("den", dens, want[1])):
            v = torch.tensor(got, dtype=torch.float64)
            t = float((v.mean() - w) / (v.std() / math.sqrt(len(v))))
            rows.append(_check("V-d0b", abs(t) <= cc.tau, process=name, lag=lag,
                               term=which, mean=float(v.mean()), theory=w, t=t))  # fmt: skip
    return rows


def _random_invertible(k: int, cond: float, gen: torch.Generator) -> Tensor:
    q1, _ = torch.linalg.qr(torch.randn(k, k, generator=gen, dtype=torch.float64))
    q2, _ = torch.linalg.qr(torch.randn(k, k, generator=gen, dtype=torch.float64))
    return (
        q1
        @ torch.diag(torch.logspace(0, math.log10(cond), k, dtype=torch.float64))
        @ q2
    )


def _grid_d(x: Tensor, every: int) -> float | None:
    return twonn(x[::every]).d


def _vd0c(cc: CheckConf, gconf: GeomConf, centers: Mapping[int, Tensor]) -> list[Row]:
    g = torch.Generator().manual_seed(cc.affine_seed)
    r = math.exp(-1 / gconf.ou_tau)
    ref, ev = ou(cc.n, cc.affine_k, r, g), ou(cc.n, cc.affine_k, r, g)
    a = _random_invertible(cc.affine_k, cc.affine_cond, g)
    shift = 100 * torch.randn(cc.affine_k, generator=g, dtype=torch.float64)
    plain = Whitener.fit(ref, cc.affine_k, c_eps=0.0)
    moved = Whitener.fit(ref @ a.T + shift, cc.affine_k, c_eps=0.0)
    w0, w1 = plain.white(ev), moved.white(ev @ a.T + shift)
    rows = []
    for lag, u in centers.items():
        p, m = _pooled_gs(w0, u, lag), _pooled_gs(w1, u, lag)
        ok = _close(p[0], m[0], 1e-8) and _close(p[1], m[1], 1e-8)
        rows.append(
            _check(
                "V-d0c-affine", ok, lag=lag, G=p[0], G_moved=m[0], s=p[1], s_moved=m[1]
            )
        )
    d0, d1 = _grid_d(w0, cc.grid_every), _grid_d(w1, cc.grid_every)
    ok = d0 is not None and d1 is not None and _close(d0, d1, 1e-8)
    rows.append(_check("V-d0c-twonn-white", ok, d=d0, d_moved=d1))
    q, _ = torch.linalg.qr(
        torch.randn(cc.affine_k, cc.affine_k, generator=g, dtype=torch.float64)
    )
    sim = Whitener.fit(3.7 * ref @ q.T + shift, cc.affine_k, c_eps=0.0)
    e0, e1 = (
        _grid_d(plain.pca(ev), cc.grid_every),
        _grid_d(sim.pca(3.7 * ev @ q.T + shift), cc.grid_every),
    )
    ok = e0 is not None and e1 is not None and _close(e0, e1, 1e-8)
    rows.append(_check("V-d0c-twonn-pre", ok, d=e0, d_moved=e1))
    return rows


def _vd0d(cc: CheckConf, gconf: GeomConf, centers: Mapping[int, Tensor]) -> list[Row]:
    x = ou(
        cc.n,
        cc.d,
        math.exp(-1 / gconf.ou_tau),
        torch.Generator().manual_seed(cc.seed - 2),
    )
    rows = []
    for lag, u in centers.items():
        f, b = _pooled_gs(x, u, lag), _pooled_gs(x.flip(0), cc.n - 1 - u, lag)
        ok = _close(f[0], b[0], 1e-12) and _close(f[1], b[1], 1e-12)
        rows.append(
            _check(
                "V-d0d", ok, lag=lag, G=f[0], G_reversed=b[0], s=f[1], s_reversed=b[1]
            )
        )
    return rows


def _vd0e(cc: CheckConf, gconf: GeomConf) -> list[Row]:
    g = torch.Generator().manual_seed(cc.seed - 3)
    k = gconf.k
    const = torch.ones(cc.n, cc.d, dtype=torch.float64)
    narrow = torch.randn(cc.n, k // 2, generator=g, dtype=torch.float64)
    low = torch.randn(cc.n, k - 4, generator=g, dtype=torch.float64) @ torch.randn(
        k - 4, k, generator=g, dtype=torch.float64
    )
    rows = []
    for name, z in (("constant", const), ("D<k", narrow), ("rank", low)):
        wh = Whitener.fit(z, k, c_eps=gconf.c_eps)
        rows.append(_check("V-d0e", not wh.scoreable, code=name, why=wh.why))
    rows.append(_check("V-d0e", twonn(const[:500]).d is None, code="constant-twonn"))
    return rows


# the hand count: unequal points on a line, each (r1, r2) counted by hand
_LINE = (0.0, 1.0, 3.0, 8.0, 20.0, 45.0, 100.0)
_LINE_MU = (3.0, 2.0, 1.5, 7 / 5, 17 / 12, 37 / 25, 80 / 55)


def _vd0f(cc: CheckConf, gconf: GeomConf) -> list[Row]:
    rows = []
    logs = sorted(math.log(m) for m in _LINE_MU)
    n = len(logs)
    fit = math.floor(0.9 * n)
    ys = [-math.log(1 - i / n) for i in range(1, fit + 1)]
    want = sum(x * y for x, y in zip(logs, ys, strict=False)) / sum(
        x * x for x in logs[:fit]
    )
    line = torch.tensor(_LINE, dtype=torch.float64)[:, None]
    got = twonn(line)
    ok = got.d is not None and _close(got.d, want, 1e-12) and got.n_fit == fit
    rows.append(_check("V-d0f-hand", ok, d=got.d, d_hand=want, n_fit=got.n_fit))
    dup = twonn(torch.cat([line, line[3:4], line[5:6] + 4e-6]), radius=10.0)
    ok = dup.n_dup == 2 and dup.d is not None and _close(dup.d, want, 1e-12)
    rows.append(
        _check("V-d0f-duplicates", ok, n_dup=dup.n_dup, n_exact=dup.n_exact, d=dup.d)
    )
    n487 = cc.twonn_points
    rows.append(_check("V-d0f-discard", twonn(torch.randn(n487, 3, generator=torch.Generator().manual_seed(0),
                       dtype=torch.float64)).n_fit == 438, n=n487))  # fmt: skip

    def mean_d(
        make: Callable[[torch.Generator], Tensor],
    ) -> tuple[float | None, list[float | None]]:
        ds = [twonn(make(torch.Generator().manual_seed(s))).d for s in cc.twonn_seeds]
        ok_ds = [d for d in ds if d is not None]
        return (sum(ok_ds) / len(ok_ds) if len(ok_ds) == len(ds) else None), ds

    def phases(name: str, noise: float = 0.0) -> Callable[[torch.Generator], Tensor]:
        def make(g: torch.Generator) -> Tensor:
            theta = 2 * math.pi * torch.rand(n487, generator=g, dtype=torch.float64)
            x = chebyshev(theta, cheb_js(name, gconf))
            return (
                x + noise * torch.randn(x.shape, generator=g, dtype=torch.float64)
                if noise
                else x
            )

        return make

    def gauss(d: int) -> Callable[[torch.Generator], Tensor]:
        return lambda g: torch.randn(n487, d, generator=g, dtype=torch.float64)

    m1, each = mean_d(phases("cheb_a"))
    rows.append(
        _check("V-d0f-cheb1", m1 is not None and 0.5 < m1 < 1.5, d=m1, each=each)
    )
    m4, each = mean_d(gauss(4))
    rows.append(
        _check("V-d0f-gauss4", m4 is not None and 3.5 < m4 < 4.5, d=m4, each=each)
    )
    for d in cc.gauss_dims[1:]:
        m, each = mean_d(gauss(d))
        rows.append({"part": "record", "name": "gauss", "dim": d, "d": m, "each": each})
    for name in ("cheb_b", "cheb_c"):
        m, each = mean_d(phases(name))
        rows.append({"part": "record", "name": f"{name}-phases", "d": m, "each": each})
    for eps in cc.noise_levels:
        m, each = mean_d(phases("cheb_b", eps))
        rows.append(
            {
                "part": "record",
                "name": "cheb_b-noise",
                "noise": eps,
                "d": m,
                "each": each,
            }
        )
    for name in ("cheb_a", "cheb_b", "cheb_c"):
        x = cheb_series(name, cc.n, gconf.cheb_seed, gconf)
        ds = [
            twonn(x[o :: cc.grid_every]).d
            for o in range(0, cc.grid_every, gconf.grid_step)
        ]
        rows.append({"part": "record", "name": f"{name}-grid", "each": ds})
    return rows


# -- the yard: V-d0h, V-d1, V-d2 ------------------------------------------------------


def _rows_of(rows: Sequence[Row], **labels: Any) -> list[Row]:
    return [r for r in rows if all(r.get(k) == v for k, v in labels.items())]


def _main(rows: Sequence[Row], gconf: GeomConf, **labels: Any) -> list[Row]:
    return _rows_of(rows, k=gconf.k, eps=gconf.c_eps, ref="all", **labels)


def _bits(a: Tensor, b: Tensor) -> dict[str, Any]:
    diff = (a.double() - b.double()).abs()
    return {"equal": bool(torch.equal(a, b)), "max_abs": float(diff.max()),
            "max_rel": float(diff.max() / a.double().abs().max().clamp(min=1e-300))}  # fmt: skip


def vd0h(
    hmean: tuple[Tensor, Tensor],
    k3: Mapping[str, tuple[Tensor, Tensor]],
    pca_eval: Tensor,
    pca_ref: Tensor,
    sets: GeomSets,
    gconf: GeomConf,
) -> list[Row]:
    """V-d0h: ``k3_hmean`` is ``init_c``'s ``h_mean`` to the bit (evaluation
    and reference), ``k3_z`` is not scoreable at $k$, and ``pca`` reversed
    in time (its centres mirrored) has the same $G$ and $s$."""
    rows = []
    for i, part in enumerate(("eval", "ref")):
        b = _bits(k3["h_mean"][i], hmean[i])
        rows.append(_check("V-d0h-k3_hmean", b["equal"], set=part, **b))
    wh = Whitener.fit(k3["own"][1], gconf.k, c_eps=gconf.c_eps)
    rows.append(_check("V-d0h-k3_z", not wh.scoreable, why=wh.why))
    w = Whitener.fit(pca_ref, gconf.k, c_eps=gconf.c_eps).white(pca_eval)
    n = len(w)
    for lag, by in sets.triples.items():
        u = by["all"]
        f, b = _pooled_gs(w, u, lag), _pooled_gs(w.flip(0), n - 1 - u, lag)
        ok = _close(f[0], b[0], 1e-12) and _close(f[1], b[1], 1e-12)
        rows.append(
            _check(
                "V-d0h-reverse",
                ok,
                lag=lag,
                G=f[0],
                G_reversed=b[0],
                s=f[1],
                s_reversed=b[1],
            )
        )
    return rows


def vd1(rows_by_code: Mapping[str, Sequence[Row]], gconf: GeomConf) -> list[Row]:
    """V-d1: of ``pca`` and ``init_c``'s own latent, the ``in`` stratum's
    $s$ at the shortest lag has $s + 2\\,\\mathrm{SE}^* < 1/2$."""
    out = []
    lag = gconf.lags[0]
    for code in ("pca", "init_c"):
        rows = rows_by_code[code]
        (spread_row,) = _main(rows, gconf, place="own", part="spread")
        (mid,) = _main(rows, gconf, place="own", part="mid", lag=lag, stratum="in")
        se = s_se(spread_row, mid, gconf)
        out.append(
            _check(
                "V-d1", mid["s"] + 2 * se < 0.5, code=code, lag=lag, s=mid["s"], se=se
            )
        )
    return out


def vd2(
    pca_eval: Tensor, pca_ref: Tensor, sets: GeomSets, gconf: GeomConf
) -> list[Row]:
    r"""V-d2: white noise of total variance $\nu V$ (seed ``inject_seed``)
    added to ``pca``'s whitened codes, $\nu$ = ``injection_for`` of each lag's
    ``in`` $G$, $s$ and each $\delta$ of ``deltas``; $\hat\delta = G_\nu - G$
    on the same triples and its paired $\mathrm{SE}^*$, a failure where
    $\lvert \hat\delta - \delta \rvert > \tau\,\mathrm{SE}^*$."""
    wh = Whitener.fit(pca_ref, gconf.k, c_eps=gconf.c_eps)
    w = wh.white(pca_eval)
    v = spread(w)
    xi = torch.randn(
        w.shape,
        generator=torch.Generator().manual_seed(gconf.inject_seed),
        dtype=torch.float64,
    )
    nb = sets.n_blocks
    rows = []
    for lag, by in sets.triples.items():
        u = by["in"]
        blocks = sets.blocks[u - lag]
        t = triple_terms(w, u, lag)
        g, s = float(t.num.sum() / t.den.sum()), float(t.den.mean()) / (2 * v)
        for delta in gconf.deltas:
            nu = injection_for(g, s, delta)
            if nu is None:
                rows.append(
                    {
                        "part": "record",
                        "name": "V-d2-unreachable",
                        "lag": lag,
                        "delta": delta,
                        "G": g,
                    }
                )
                continue
            tn = triple_terms(w + math.sqrt(nu * v / w.shape[1]) * xi, u, lag)
            num_b = torch.stack(
                [_block_sums(t.num, blocks, nb), _block_sums(tn.num, blocks, nb)]
            )
            den_b = torch.stack(
                [_block_sums(t.den, blocks, nb), _block_sums(tn.den, blocks, nb)]
            )
            hat = float(tn.num.sum() / tn.den.sum()) - g
            se = ratio_se(num_b, den_b, torch.tensor([-1.0, 1.0]), gconf)
            rows.append(_check("V-d2", abs(hat - delta) <= gconf.tau_vd2 * se, lag=lag, delta=delta,
                               nu=nu, G=g, s=s, hat=hat, se=se, recovery=hat / delta,
                               apart_from_0=abs(hat) > 2 * se))  # fmt: skip
    return rows


def twonn_resolution(gconf: GeomConf) -> list[Row]:
    """The TwoNN of isotropic Gaussians of ``res_dims`` with the AR(1)
    correlation ``res_phis`` between consecutive points (``res_points``,
    ``res_draws`` draws from ``res_seed``): records of V-d2's side."""
    rows = []
    for phi in gconf.res_phis:
        for d in gconf.res_dims:
            ds = []
            for i in range(gconf.res_draws):
                g = torch.Generator().manual_seed(gconf.res_seed + i)
                x = (
                    white(gconf.res_points, d, g)
                    if phi == 0
                    else ou(gconf.res_points, d, phi, g)
                )
                ds.append(twonn(x, discard=gconf.discard, floor=gconf.floor).d)
            rows.append(
                {
                    "part": "record",
                    "name": "twonn_resolution",
                    "phi": phi,
                    "dim": d,
                    "d": ds,
                }
            )
    return rows


# -- the jobs ----------------------------------------------------------------------


def _write(path: Path, rows: Sequence[Row], stamp: Mapping[str, Any]) -> None:
    """``rows`` with ``stamp`` and the snapshot ``snap`` as compact JSON to
    ``path`` atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    code = of_diagnosis({})["code"]
    stamp = {"snap": code and code.get("snap")} | dict(stamp)
    data = json.dumps([dict(stamp) | dict(r) for r in rows], separators=(",", ":"))
    with atomic_open(path) as f:
        f.write((data + "\n").encode())


def _start(
    out: Path, depends: Mapping[str, Any], determinism: Mapping[str, Any]
) -> None:
    """``out/config.json``: written if absent, else checked.

    Raises:
        ValueError: The ``config.json`` there is of other settings.
    """
    path = out / "config.json"
    plain: dict[str, Any] = json.loads(json.dumps(dict(depends)))
    if path.exists():
        old = json.loads(path.read_text())
        if {k: old.get(k) for k in plain} != plain:
            raise ValueError(f"{path} is of other settings: remove it first")
        return
    out.mkdir(parents=True, exist_ok=True)
    record = plain | {
        "determinism": dict(determinism),
        "provenance": {k: v for k, v in of_diagnosis({}).items() if k != "settings"},
    }
    with atomic_open(path) as f:
        f.write((json.dumps(record, indent=2) + "\n").encode())


def evaluate_checks(
    out: Path, cc: CheckConf = CheckConf(), gconf: GeomConf = GeomConf()
) -> Path:
    """V-d0a to V-d0f into ``out`` (``config.json``, ``rows.json``) on the
    CPU. Returns ``out``."""
    determinism = configure_determinism("cpu")
    _start(out, {"check": asdict(cc), "geom": asdict(gconf)}, determinism)
    _write(out / "rows.json", check_rows(cc, gconf), {})
    return out


def _conf_of(run_dir: Path, device: str | None) -> RunConf:
    conf = from_dict(yaml.safe_load((run_dir / "config.yaml").read_text()))
    return conf if device is None else replace(conf, device=device)


def _sets_at(
    out: Path, conf: RunConf, root: Path, gconf: GeomConf
) -> tuple[GeomSets, Zurich]:
    """The sets saved in ``out`` (made and saved there if absent) and
    Zurich's stacks.

    Raises:
        ValueError: The saved sets are of other settings.
    """
    if (out / "sets.pt").exists():
        sets = GeomSets.load(out)
        zurich = Zurich(conf, root, 1)
        t = conf.data.tours
        want = {"data_version": t.data_version, "held_out": t.held_out,
                "frames": len(zurich.frames), "stack": t.frames, "stride": t.stride,
                "geom": asdict(gconf)}  # fmt: skip
        got = {k: sets.ident.get(k) for k in want}
        if got != json.loads(json.dumps(want)):
            raise ValueError(f"{out / 'sets.pt'} is of other settings")
        return sets, zurich
    sets, zurich = zurich_sets(conf, root, gconf)
    sets.save(out)
    return sets, zurich


def _jepa_of(run_dir: Path, device: str) -> tuple[JEPAModules, str]:
    """The JEPA of the run (built from its seed) and its own latent's kind:
    ``zpca`` of the SIGReg of the mean (V18b M), else ``z``."""
    conf = _conf_of(run_dir, device)
    settings = json.loads((run_dir / "jepa.json").read_text())
    own = "zpca" if settings.get("sigreg_on", "flat") == "mean" else "z"
    return build_run_jepa(conf, settings), own


def _init_of(run_dir: Path, device: str) -> tuple[JEPAModules, str, Path]:
    """The JEPA of the run's step 0 checkpoint (``plans/jepa.md``, 「符号と
    checkpoint」), its own latent's kind and the checkpoint.

    Raises:
        ValueError: The run has no checkpoint of step 0.
    """
    ckpts = run_ckpts(run_dir)
    if 0 not in ckpts:
        raise ValueError(f"{run_dir} has no checkpoint of step 0 (of {list(ckpts)})")
    mods, own = _jepa_of(run_dir, device)
    ckpt.load_eval(ckpts[0], mods.modules())
    return mods, own, ckpts[0]


def _first_vae(run_dir: Path) -> Path:
    """The checkpoint of $t = 0$ of the VAE run (its untrained $\\mu$).

    Raises:
        ValueError: The run's first checkpoint is not of $t = 0$ (a learned
            VAE would be a yardstick before $\\Delta_*$).
    """
    first = min((run_dir / "ckpt").glob("*.pt"), key=lambda p: float(p.stem))
    if float(first.stem) != 0.0:
        raise ValueError(f"{run_dir}'s first checkpoint {first.name} is not of t = 0")
    return first


def _places_of(
    fn: Places, zurich: Zurich, ref: Tensor, sets: GeomSets, dev: torch.device
) -> dict[str, tuple[Tensor, Tensor]]:
    ev = encode_places(fn, zurich.stacks, sets.n, dev)
    rf = encode_places(fn, lambda idx: ref[idx], len(ref), dev)
    return {p: (ev[p], rf[p]) for p in ev}


def _zpca(pair: tuple[Tensor, Tensor], seed: int, dim: int) -> tuple[Tensor, Tensor]:
    """The PCA of ``dim`` components of $z$ fit on the reference (seed $+$
    ``PCA_SEED``) of both sets (``jepa_eval``'s ``zpca``)."""
    gen = torch.Generator().manual_seed(seed + PCA_SEED)
    mean, comps = pca.fit_pca(pair[1], dim, gen=gen)
    return pca.encode(pair[0], mean, comps), pca.encode(pair[1], mean, comps)


def _feature_code(fn: Callable[[Tensor], Tensor]) -> Places:
    return lambda x: {"own": fn(x)}


def _mean8(x: Tensor) -> Tensor:
    """The mean image of the stack's frames ``(B, 3 S S)`` (``x / 255``)."""
    return x.float().mean(1).flatten(1) / 255


def evaluate_yard(
    conf: RunConf,
    out: Path,
    *,
    inits: Mapping[str, Path],
    init_v16: Path,
    same: Mapping[str, Sequence[Path]],
    root: Path | None = None,
    gconf: GeomConf = GeomConf(),
    fconf: FillConf = FillConf(),
    spacing: Mapping[str, Any] | None = None,
    span: float | None = None,
) -> Path:
    """The sets, the yardsticks' and synthetic controls' rows (``out /
    "yard/rows/<code>.json"``), and V-d0h, V-d1, V-d2 and the TwoNN's
    resolution (``out / "yard/vd.json"``). ``inits`` the runs of
    ``YARD_INITS`` (their step 0 checkpoints); ``init_v16`` V16's VAE run
    (its checkpoint of $t = 0$); ``same`` of ``init_c`` and ``init_c1`` the
    runs of the seed whose step 0 stands for all (``plans/jepa.md``: a
    ``step0_same`` record of ``self_eval.same_weights``; where they differ,
    each run's step 0 is a yardstick too, ``<init>@<run>``). Every row of an
    init holds its checkpoint ``source`` and ``t``. Returns ``out``.

    Raises:
        ValueError: A run of ``YARD_INITS`` or of ``same`` is not given, a
            run has no step 0, or V16's first checkpoint is not of $t = 0$.
    """
    missing = [n for n in YARD_INITS if n not in inits]
    missing += [n for n in SAME_INITS if not same.get(n)]
    if missing:
        raise ValueError(f"the yard needs the runs {missing}")
    determinism = configure_determinism(conf.device)
    dev = torch.device(conf.device)
    root = tours_root(conf.data.tours.data_version) if root is None else root
    sets, zurich = _sets_at(out, conf, root, gconf)
    kw: dict[str, Any] = {} if span is None else {"span": span}
    ref = fill_sets(conf, root, fconf, spacing=spacing, **kw).fit
    yard = out / "yard"
    sources = {k: str(v) for k, v in inits.items()} | {"init_v16": str(init_v16)}
    sources |= {f"same_{k}": [str(p) for p in same[k]] for k in SAME_INITS}
    _start(
        yard, {"geom": asdict(gconf), "key": sets.key, "sources": sources}, determinism
    )
    rows_by: dict[str, list[Row]] = {}
    kept: dict[str, dict[str, tuple[Tensor, Tensor]]] = {}

    def emit(
        code: str,
        places: Mapping[str, tuple[Tensor, Tensor]],
        sens: bool,
        labels: Mapping[str, Any] | None = None,
        **kw: Any,
    ) -> None:
        rows: list[Row] = []
        for place, (ev, rf) in places.items():
            lab = {"code": code, "place": place} | dict(labels or {})
            rows += code_rows(ev, rf, sets, gconf, sens=sens, labels=lab, **kw)
        _write(yard / "rows" / f"{code}.json", rows, {})
        rows_by[code] = rows

    def init_places(
        run_dir: Path,
    ) -> tuple[dict[str, tuple[Tensor, Tensor]], JEPAModules, dict[str, Any]]:
        mods, own, path = _init_of(run_dir, conf.device)
        places = _places_of(JepaPlaces(mods).places, zurich, ref, sets, dev)
        if own == "zpca":
            places["own"] = _zpca(
                places["own"], _conf_of(run_dir, None).seed, gconf.pca_dim
            )
        return places, mods, {"source": str(path), "t": float(path.stem)}

    gen = torch.Generator().manual_seed(conf.seed + PCA_SEED)
    pca_code = PCACode.fit(pixels, ref, gconf.pca_dim, gen=gen, device=conf.device)
    kept["pca"] = _places_of(_feature_code(pca_code.mu), zurich, ref, sets, dev)
    emit("pca", kept["pca"], True)
    emit(
        "color",
        _places_of(_feature_code(ColorCode(16).mu), zurich, ref, sets, dev),
        True,
    )
    gen = torch.Generator().manual_seed(conf.seed + PCA_SEED)
    mean8 = PCACode.fit(_mean8, ref, gconf.pca_dim, gen=gen, device=conf.device)
    emit("mean8", _places_of(_feature_code(mean8.mu), zurich, ref, sets, dev), True)
    records: list[Row] = []
    for name in YARD_INITS:
        places, mods, source = init_places(inits[name])
        emit(name, places, True, labels=source)
        if name == "init_c":
            kept["init_c"] = places
            kept["k3"] = _places_of(K3Places(mods).places, zurich, ref, sets, dev)
    for name in SAME_INITS:
        runs = list(same[name])
        ok = same_weights([run_ckpts(d)[0] for d in runs])
        records.append({"part": "record", "name": "step0_same", "init": name,
                        "same": ok, "runs": [str(d) for d in runs]})  # fmt: skip
        if not ok:  # each run's step 0 a yardstick (plans/jepa.md)
            for run_dir in runs:
                places, _, source = init_places(run_dir)
                emit(f"{name}@{run_dir.name}", places, True, labels=source)
    first = _first_vae(init_v16)
    vae = vae_at(first, conf.device)
    assert isinstance(vae, VAEModules)
    emit("init_v16", _places_of(VaeTokenCode(vae).places, zurich, ref, sets, dev), True,
         labels={"source": str(first), "t": float(first.stem)})  # fmt: skip
    # the synthetic controls
    ev, rf = kept["pca"]["own"]
    perm = torch.randperm(
        len(ev), generator=torch.Generator().manual_seed(gconf.perm_seed)
    )
    # the TwoNN's near neighbours by the permuted stacks' own times
    emit("pca_perm", {"own": (ev[perm], rf)}, False,
         times=perm[sets.grid].double() / gconf.grid_step)  # fmt: skip
    for name, pair in synthetic_codes(sets.n, gconf).items():
        emit(name, {"own": pair}, False, population=True)
    emit("k3_z", {"own": kept["k3"]["own"]}, False)
    emit("k3_hmean", {"h_mean": kept["k3"]["h_mean"]}, False)
    checks = vd0h(
        kept["init_c"]["h_mean"], kept["k3"], *kept["pca"]["own"], sets, gconf
    )
    checks += vd1(rows_by, gconf)
    checks += vd2(*kept["pca"]["own"], sets, gconf)
    checks += twonn_resolution(gconf)
    _write(yard / "vd.json", records + checks, {"key": sets.key})
    return out


def _apply_rows(
    fn: Places, zurich: Zurich, ref: Tensor, sets: GeomSets, dev: torch.device,
    labels: Mapping[str, Any], gconf: GeomConf, own_zpca_seed: int | None = None,
) -> list[Row]:  # fmt: skip
    places = _places_of(fn, zurich, ref, sets, dev)
    if own_zpca_seed is not None:
        places["own"] = _zpca(places["own"], own_zpca_seed, gconf.pca_dim)
    rows: list[Row] = []
    for place, (ev, rf) in places.items():
        rows += code_rows(
            ev, rf, sets, gconf, sens=False, labels=dict(labels) | {"place": place}
        )
    return rows


def evaluate_run(
    run_dir: Path,
    *,
    steps: Sequence[int],
    sets_dir: Path,
    root: Path | None = None,
    device: str | None = None,
    gconf: GeomConf = GeomConf(),
    fconf: FillConf = FillConf(),
    spacing: Mapping[str, Any] | None = None,
    span: float | None = None,
) -> Path:
    """The rows of the JEPA run's checkpoints of ``steps`` into ``run_dir /
    OUT / "rows/<step>.json"`` (a step whose file is there is skipped), on
    the sets saved in ``sets_dir``; of a run with $W$ (V18c) also the
    replicated K3's (``K3Places``: ``code`` ``k3_z`` of ``own`` and
    ``k3_hmean`` of ``h_mean``, with ``run``) into ``k3/<step>.json``
    (``plans/jepa.md``, 「合成の対照」). Returns the directory.

    Raises:
        ValueError: A step has no checkpoint, or the sets or the
            ``config.json`` there are of other settings.
    """
    conf = _conf_of(run_dir, device)
    ckpts = run_ckpts(run_dir)
    missing = [s for s in steps if s not in ckpts]
    if missing:
        raise ValueError(f"no checkpoint of steps {missing} (of {list(ckpts)})")
    determinism = configure_determinism(conf.device)
    dev = torch.device(conf.device)
    root = tours_root(conf.data.tours.data_version) if root is None else root
    sets, zurich = _sets_at(sets_dir, conf, root, gconf)
    kw: dict[str, Any] = {} if span is None else {"span": span}
    ref = fill_sets(conf, root, fconf, spacing=spacing, **kw).fit
    mods, own = _jepa_of(run_dir, conf.device)
    out = run_dir / OUT
    _start(
        out, {"geom": asdict(gconf), "key": sets.key, "run": run_dir.name}, determinism
    )
    for step in steps:
        stamp = {"run": run_dir.name, "step": step, "t": float(ckpts[step].stem)}
        path = out / "rows" / f"{step:06d}.json"
        k3_path = out / "k3" / f"{step:06d}.json"
        todo = not path.exists() or (mods.mix is not None and not k3_path.exists())
        if not todo:
            continue
        ckpt.load_eval(ckpts[step], mods.modules())
        if not path.exists():
            rows = _apply_rows(JepaPlaces(mods).places, zurich, ref, sets, dev,
                               {"code": run_dir.name, "step": step}, gconf,
                               conf.seed if own == "zpca" else None)  # fmt: skip
            _write(path, rows, stamp)
        if mods.mix is not None and not k3_path.exists():
            k3 = _places_of(K3Places(mods).places, zurich, ref, sets, dev)
            rows = []
            for code, place in (("k3_z", "own"), ("k3_hmean", "h_mean")):
                ev, rf = k3[place]
                lab = {"code": code, "place": place, "run": run_dir.name, "step": step}
                rows += code_rows(ev, rf, sets, gconf, sens=False, labels=lab)
            _write(k3_path, rows, stamp)
    return out


def evaluate_vae(
    path: Path,
    *,
    name: str,
    sets_dir: Path,
    root: Path | None = None,
    device: str | None = None,
    gconf: GeomConf = GeomConf(),
    fconf: FillConf = FillConf(),
    spacing: Mapping[str, Any] | None = None,
    span: float | None = None,
) -> Path:
    """The rows of the VAE checkpoint ``path`` (its run's ``config.yaml``
    two levels up) into ``sets_dir / "vae/rows/<name>.json"``. Returns
    the file.

    The sets and the reference are read on the run's ``eval.stream_view``:
    of a ``"chunk"`` run (V17) the single stream as a spread run of the
    same ``data.tours`` and ``time.frame``, so the same data as every
    other code's (``fill_sets`` keeps only those and its own span)."""
    conf = _conf_of(path.parent.parent, device)
    view, _ = stream_view(conf)
    determinism = configure_determinism(conf.device)
    dev = torch.device(conf.device)
    root = tours_root(conf.data.tours.data_version) if root is None else root
    sets, zurich = _sets_at(sets_dir, view, root, gconf)
    kw: dict[str, Any] = {} if span is None else {"span": span}
    ref = fill_sets(view, root, fconf, spacing=spacing, **kw).fit
    mods = build_vae(conf, seed=conf.seed)
    ckpt.load_eval(path, mods.modules())
    out = sets_dir / "vae"
    _start(out, {"geom": asdict(gconf), "key": sets.key}, determinism)
    rows = _apply_rows(VaeTokenCode(mods).places, zurich, ref, sets, dev,
                       {"code": name, "source": str(path)}, gconf)  # fmt: skip
    target = out / "rows" / f"{name}.json"
    _write(target, rows, {"t": float(path.stem)})
    return target


def _ints(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split(",") if x)


def main(argv: Sequence[str] | None = None) -> Path:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = p.add_subparsers(dest="job", required=True)
    checks = sub.add_parser("checks")
    checks.add_argument("--out", type=Path, required=True)
    yard = sub.add_parser("yard")
    yard.add_argument("--config", type=Path, required=True)
    yard.add_argument("--out", type=Path, required=True)
    for name in YARD_INITS:
        yard.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    yard.add_argument("--init-v16", type=Path, required=True)
    for name in SAME_INITS:  # the runs of the seed, the init's own among them
        yard.add_argument(f"--same-{name.removeprefix('init_')}", type=Path,
                          nargs="+", required=True)  # fmt: skip
    run = sub.add_parser("run")
    run.add_argument("run_dir", type=Path)
    run.add_argument("--steps", type=_ints, required=True)
    vae = sub.add_parser("vae")
    vae.add_argument("ckpt", type=Path)
    vae.add_argument("--name", required=True)
    for q in (run, vae):
        q.add_argument("--sets", type=Path, required=True)
    for q in (yard, run, vae):
        q.add_argument("--device", default=None)
        q.add_argument("--root", type=Path, default=None)
    a = p.parse_args(argv)
    if a.job == "checks":
        return evaluate_checks(a.out)
    if a.job == "yard":
        conf = _conf_of(a.config, a.device)
        inits = {n: getattr(a, n) for n in YARD_INITS}
        same = {n: getattr(a, f"same_{n.removeprefix('init_')}") for n in SAME_INITS}
        return evaluate_yard(conf, a.out, inits=inits, init_v16=a.init_v16, same=same,
                             root=a.root)  # fmt: skip
    if a.job == "run":
        return evaluate_run(
            a.run_dir, steps=a.steps, sets_dir=a.sets, root=a.root, device=a.device
        )
    return evaluate_vae(
        a.ckpt, name=a.name, sets_dir=a.sets, root=a.root, device=a.device
    )


if __name__ == "__main__":
    main()
