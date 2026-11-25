r"""V18c's evaluation of the unobserved parts (``plans/jepa.md``, 「未観測の
部分」; README 「V18c: 未観測の部分の評価（E1・E1e・E1g・E2）」): whether a
JEPA's codes hold, in a form a fixed linear readout takes out, the features
of what its input did not show.

The sets, the same for every code (``fill_sets``): the probes' fit stacks
(``StreamEvaluator.fit_stacks`` of the span ``SPAN``) fit the readouts and
the targets' spaces; the first stacks of the held-out video's calib windows
choose the ridges; the held-out video's grid (``gen_eval.Zurich``, 1 s
apart) is scored, on G's candidates (``gen_eval.candidates``, ``id_easy``
among all of them, chance $1/n$) and their neighbours (``gen_eval.
hard_sets``, ``id_hard`` among 33, chance 1/33).

The targets (``build_targets``, once into a cache ``v18c-targets-<key>``):
of a region of the last frame of a stack, ``incep`` (FID's Inception of the
region's image, ``regions.region_image``, in the PCA of ``target_dim``
components fit on the fit set) and ``color`` (the mean colors of the
``color_block``-pixel blocks wholly in the region), each less the fit set's
mean and over one scale (the root mean square of the coordinates' sds on the
fit set). The keys: ``U/<mask>/<kind>`` the hidden region of each of
``E1_MASKS``, ``V/<mask>/<kind>`` the visible one, ``I/<mask>/<kind>`` the
hidden region of the frame filled harmonically (``regions.harmonic_fill``)
in U's space; of E2 (with the wide data, ``exp.lib.tours.wide``)
``B/<band>/<kind>`` of ``BANDS``, ``C/<kind>`` the whole frame (``ext``) and
``BI/<band>/<kind>`` the band of the square's edge column repeated
(``interp``).

The readout (``exp.lib.readout``): a centred ridge from inputs (their PCA of
``readout_dim`` from ``READOUT_PCA_SEED`` on the fit set where wider) to a
target, its scale chosen by ``r2`` on the calib set. A row of a readout's
prediction $\hat y$ of the candidates (``score``): ``r2``, ``sq_i`` (each
candidate's squared error), ``id_easy``, ``id_hard``, ``hits_easy``,
``hits_hard``, ``n_hard``, ``pick_max_easy`` (a hub's share of the picks)
(``metrics.identification``, centre the mean of
the targets of the held-out video's grid, as G's centre is $R$'s: the
targets are centred on the fit set, and the shift from the training videos
to Zurich is common to every candidate).

- E1 (``e1_rows``): of each mask $U$ and kind, the readout $g_U$ from the
  target side's tokens at $U$ (``JEPAModules.tokens``, positions in order)
  fit on the fit set, never on a prediction; then the stages ``oracle``
  (those tokens of the candidates), ``pred`` (the predictor's under $U$),
  ``mean`` (the fit set's mean), ``shuf`` (``pred`` of the candidate half
  the candidates on), and ``interp`` (``I`` itself, no readout). ``swap``
  (mask ``S``, ``_swap_rows``): under ``S_1`` and ``S_2`` both quarters
  hidden, whether $g_A$ takes the truth of quarter $A$ from the inputs at
  $A$ rather than from those at $B$: with $u_P$ a stage's inputs at $P$
  less their mean over the candidates (both in $A$'s reducer), whether
  $(u_A - u_B) B_A$ points to $A$'s target less the grid's mean (its inner
  product is positive; ``metrics.swap``), averaged over $(A, B)$ and $(B,
  A)$ of both, per candidate. ``gain`` $= (\text{pred} - \text{mean}) /
  (\text{oracle} - \text{mean})$ of the aggregation.
- E1e (``e1e_rows``): the context side's tokens (the visible positions'
  encoder outputs) to $U$'s targets, a readout of each code; the yardsticks
  ``color_V`` (the visible coarse cells' mean colors), ``pca_V`` (the
  stack's pixels with the hidden ones 0), ``ext`` ($V$'s target).
- E1g (``e1g_rows``): G's decoder of the run's ``z`` draws the code whose
  hidden positions are filled (``full`` all the target side's; ``oracle``,
  ``pred``, ``mean`` the visible positions the context side's and the hidden
  ones the target side's, the predictor's, the fit set's mean), from the
  noise of G's sample 0; the drawn hidden region in $U$'s space is
  identified among the candidates' (``id_easy``, ``id_hard``), and the
  visible one (``vis_easy``).
- E2 (``e2_rows``): a code of the plain stack to each band's targets; the
  yardsticks ``ext`` (``C``), ``interp`` (``BI``) and ``mean``. ``side`` and
  ``outer`` are the means of ``L``, ``R`` and of ``L_out``, ``R_out``.

E1's repair (``plans/jepa.md``, 「E1 の修理」):

- ``diag`` (``e1diag_rows``, on the sets and targets as they are): of
  ``swap``'s pairs, directions, kinds and the stages ``oracle`` and
  ``pred``, the curves of the inner products and the wins of every shift
  of the candidates (``shift_curves``), into ``rows/<step>-e1diag.json``
  and ``e1diag-<step>.png``.
- ``--split sem --sem <targets>`` (``evaluate_sem_split``): on the split of
  ``sem_eval``'s targets (``sem_split_sets``: the ridges' scales chosen on
  its calib grid stacks, the candidates of sel and final), E1 of the
  targets ``color`` and ``quad`` (the quarters' hires features) with the
  stages ``oracle_k3`` and ``pred_k3`` (the position means copied: ``swap``
  exactly 1/2, ``n_tie`` every candidate), and E1f (``e1f_rows``): the
  readouts fit on the predictor's outputs themselves, scored by
  ``centered_r2``, into ``<run_dir>/posthoc_fill_sem/`` (``OUT_SEM``: its
  own ``config.json``, never mixed with the plain E1's).

Into ``<run_dir>/posthoc_fill/`` (``config.json``, ``rows/<step>-<part>.json``
a list of rows, the pictures ``g-<step>-<mask>.png``) and
``<run_dir>/posthoc_side/`` (``rows/<step>-<code>.json``); the yardsticks
into ``<out>/posthoc_fill/`` and ``<out>/posthoc_side/`` (``yardstick-…``). A
job whose file is there is skipped, after the ``config.json`` there is
checked against the settings the rows depend on (``_Context.depends``)::

    python -m exp.vision.fill_eval targets [--wide <dir>] [--inception <path>]
    python -m exp.vision.fill_eval videos
    python -m exp.vision.fill_eval run <run_dir> --steps 0,6000,20000,60000 \
        [--parts e1,e1e,e2]
    python -m exp.vision.fill_eval gen <run_dir> --steps 0,6000,20000,60000
    python -m exp.vision.fill_eval yardsticks --out runs/v18c-fill/yardsticks \
        --v16 <ckpt> --v17 <ckpt> --m-run <run_dir>
    python -m exp.vision.fill_eval diag <run_dir> --steps 0,6000,60000
    python -m exp.vision.fill_eval run <run_dir> --steps 0,6000,60000 \
        --parts e1,e1f --split sem --sem <sem_eval's targets>
    python -m exp.vision.fill_eval run <run_dir> --steps 0,6000,60000 \
        --parts e1f --target scene    # scene_eval.evaluate_fill
"""

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor
from torch.nn import functional as F

from exp.lib import ckpt, pca
from exp.lib.flow_decoder import FlowUNet, euler_sample
from exp.lib.metrics import identification, swap
from exp.lib.png import draw_polyline_, tile, write_png
from exp.lib.provenance import configure_determinism, of_diagnosis
from exp.lib.readout import (
    Ridge,
    centered_r2,
    centered_terms,
    fit_ridge,
    predict,
    r2,
    select_ridge,
)
from exp.lib.regions import (
    COARSE,
    EVAL_MASKS,
    SWAP_PAIRS,
    harmonic_fill,
    pixel_mask,
    region_image,
)
from exp.lib.rundir import write_json
from exp.lib.tours.stacks import DATA_ENV, StackedTourFrames, tours_root
from exp.lib.video_io import INDEX, U8Reader
from exp.vision import gen_eval
from exp.vision.compose import from_dict
from exp.vision.eval import BATCH, PCA_SEED, eval_split_starts
from exp.vision.gen_eval import (
    SAMPLE_SEED,
    Features,
    GenConf,
    Zurich,
    candidates,
    hard_sets,
    inception_features,
    step_codes,
)
from exp.vision.jepa import (
    JEPAModules,
    expand_masks,
    mask_index,
    predict as jepa_predict,
)
from exp.vision.jepa_diag import GRID_STEP, run_ckpts
from exp.vision.jepa_eval import (
    SPAN,
    W_CHUNK,
    W_DIM,
    W_SEED,
    ZPCA_DIM,
    ColorCode,
    PCACode,
    RandProjCode,
    build_run_jepa,
    pixels,
    vae_at,
)
from exp.vision.jepa_iid import v16_config
from exp.vision.models import early_fusion
from exp.vision.schema import RunConf
from exp.vision.stream_eval import Code, StreamEvaluator, spanned

__all__ = [
    "BANDS",
    "E1F_K",
    "E1F_STAGES",
    "E1_MASKS",
    "KINDS",
    "OUT",
    "OUT_SEM",
    "OUT_SIDE",
    "PARTS",
    "QUARTER_NAMES",
    "READOUT_PCA_SEED",
    "SEM_PARTS",
    "TARGET_PCA_SEED",
    "FillConf",
    "FillSets",
    "Reducer",
    "Space",
    "Targets",
    "block_colors",
    "build_targets",
    "e1_rows",
    "e1diag_rows",
    "e1e_rows",
    "e1f_fits",
    "e1f_inputs",
    "e1f_masks",
    "e1f_rows",
    "e1g_rows",
    "e2_rows",
    "evaluate_diag",
    "evaluate_gen",
    "evaluate_run",
    "evaluate_sem_split",
    "evaluate_targets",
    "evaluate_yardsticks",
    "fill_sets",
    "gain",
    "main",
    "score",
    "sem_split_sets",
    "sem_split_targets",
    "shift_curves",
    "targets_key",
]

OUT, OUT_SIDE = "posthoc_fill", "posthoc_side"
OUT_SEM = "posthoc_fill_sem"  # E1 and E1f on sem_eval's split: its own config
TARGET_PCA_SEED = 79  # the PCA of incep's targets
READOUT_PCA_SEED = 83  # the PCA of a readout's inputs wider than readout_dim
E1_MASKS = ("Q_tl", "Q_tr", "Q_bl", "Q_br", "H_l", "H_r", "H_t", "H_b", "R")
MEANS = {"Q": ("Q_tl", "Q_tr", "Q_bl", "Q_br"), "H": ("H_l", "H_r", "H_t", "H_b")}
KINDS = ("incep", "color")
BANDS = ("L", "R", "L_out", "R_out")
BAND_MEANS = {"side": ("L", "R"), "outer": ("L_out", "R_out")}
PARTS = ("e1", "e1e", "e2")  # of a run; E1g is ``gen``
SEM_PARTS = ("e1", "e1f")  # of a run with the split of sem_eval (--split sem)
SETS = ("fit", "calib", "grid")
E1G_STAGES = ("full", "oracle", "pred", "mean")
PICTURE_MASKS = ("Q_br", "R")
YARD_PREFIX = "yardstick"
YARD_E1E = ("color_V", "pca_V", "ext")
YARD_E2 = ("color", "pca", "v16", "v17", "init_m", "ext", "interp", "mean")


@dataclass(frozen=True)
class FillConf:
    """The settings (module docstring; ``plans/jepa.md``), by default the
    plan's."""

    n_cand: int = 512
    grid_step: int = GRID_STEP  # frames between the grid's stacks (1 s)
    hard_reach: int = 16
    hard_stride: int = 2
    target_dim: int = 256  # incep's PCA
    readout_dim: int = 1024  # a readout's inputs wider than this are PCA'd
    color_block: int = 8  # pixels a side of color's blocks
    outer: int = 16  # columns of a band's outer part (E2's L_out, R_out)
    picture_rows: int = 16
    picture_scale: int = 2


# -- the sets ---------------------------------------------------------------------


@dataclass(frozen=True)
class FillSets:
    """The stacks of the evaluation (module docstring): ``fit`` and
    ``calib`` u8 ``(m, n, 3, S, S)``, the grid's last frames ``grid_last``
    ``(G, 3, S, S)``, the candidates' stacks ``cand_stacks``, their grid
    indices ``cand``, ``hard_q`` and ``hard_among`` (``gen_eval.hard_sets``),
    ``ident`` (what names the sets, for the cache's key) and ``places`` (the
    video and frame of each set's last frames, for the wide data; None if
    unknown)."""

    fit: Tensor
    calib: Tensor
    grid_last: Tensor
    cand_stacks: Tensor
    cand: Tensor
    hard_q: Tensor
    hard_among: Tensor
    ident: Mapping[str, Any]
    places: Mapping[str, list[tuple[str, int]]] | None = None

    def last(self, name: str) -> Tensor:
        """The last frames u8 ``(N, 3, S, S)`` of the set ``name`` of
        ``SETS``."""
        if name == "grid":
            return self.grid_last
        return (self.fit if name == "fit" else self.calib)[:, -1]


def fill_sets(
    conf: RunConf,
    root: Path,
    fconf: FillConf,
    *,
    span: float = SPAN,
    spacing: Mapping[str, Any] | None = None,
) -> FillSets:
    """The sets of ``conf``'s data at ``root`` (module docstring);
    ``spacing`` ``StreamEvaluator``'s spacings by name (tests)."""
    view = spanned(conf, span)
    with tempfile.TemporaryDirectory() as tmp:  # StreamEvaluator's out, unused
        ev = StreamEvaluator(view, Path(tmp), root, Path(tmp), c=1, **(spacing or {}))
    t = conf.data.tours
    zurich = Zurich(conf, root, fconf.grid_step)
    cand = candidates(len(zurich), fconf.n_cand)
    hard_q, among = hard_sets(cand, len(zurich), fconf.hard_reach, fconf.hard_stride)
    calib_first = eval_split_starts(view, root)["calib"]
    stream = StackedTourFrames(
        root,
        n=1,
        held_out=t.held_out,
        frames=t.frames,
        stride=t.stride,
        advance=t.advance,
    )
    grid = torch.arange(len(zurich))
    places = {
        "fit": [stream.locate(k) for k in ev.fit_ks],
        "calib": [(t.held_out, int(s) + zurich.span) for s in calib_first],
        "grid": [(t.held_out, int(s) + zurich.span) for s in zurich.starts],
    }
    ident = {
        "data_version": t.data_version,
        "held_out": t.held_out,
        "fit_ks": ev.fit_ks,
        "calib_starts": calib_first.tolist(),
        "grid_starts": zurich.starts.tolist(),
    }
    return FillSets(
        fit=ev.fit_stacks,
        calib=ev.zurich_calib_stacks,
        grid_last=zurich.last(grid),
        cand_stacks=zurich.stacks(cand),
        cand=cand,
        hard_q=hard_q,
        hard_among=among,
        ident=ident,
        places=places,
    )


# -- the targets -------------------------------------------------------------------


@dataclass(frozen=True)
class Space:
    """A target's space: raw features $f$ to $((f - \\mu) W^\\top) / s$
    (without $W$ for ``comps`` None), float64; $s$ one scale, or one per
    coordinate ``(K,)`` (whitened)."""

    mean: Tensor
    comps: Tensor | None
    scale: float | Tensor

    def __call__(self, raw: Tensor) -> Tensor:
        u = raw.double() - self.mean
        if self.comps is not None:
            u = u @ self.comps.T
        return u / self.scale

    @classmethod
    def fit(
        cls,
        raw: Tensor,
        dim: int | None,
        *,
        seed: int = TARGET_PCA_SEED,
        whiten: bool = False,
    ) -> Space:
        """The space of the fit set's raw features ``raw`` ``(m, F)``: their
        PCA of ``dim`` components from ``seed`` (or None: their mean only),
        and the root mean square of the coordinates' sds, or with ``whiten``
        each coordinate's sd."""
        x = raw.double()
        if dim is None:
            mean, comps = x.mean(0), None
        else:
            gen = torch.Generator().manual_seed(seed)
            mean, comps = pca.fit_pca(x, dim, gen=gen)
        u = x - mean if comps is None else (x - mean) @ comps.T
        if whiten:
            return cls(mean, comps, u.std(0))
        return cls(mean, comps, math.sqrt(float(u.var(0).mean())))

    def state(self) -> dict[str, Any]:
        return {"mean": self.mean, "comps": self.comps, "scale": self.scale}


class Targets:
    """The targets (module docstring): ``spaces`` by key, ``values`` by key
    and set (``SETS``; ``grid`` only for ``I``, ``BI``), ``meta``
    (``targets.json``)."""

    def __init__(
        self,
        spaces: dict[str, Space],
        values: dict[str, dict[str, Tensor]],
        meta: dict[str, Any],
    ) -> None:
        self.spaces, self.values, self.meta = spaces, values, meta

    def y(self, key: str, part: str) -> Tensor:
        """The targets ``key`` of the set ``part`` float64 ``(N, K)``."""
        return self.values[key][part].double()

    def has(self, prefix: str) -> bool:
        return any(k.startswith(prefix) for k in self.values)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "spaces": {k: s.state() for k, s in self.spaces.items()},
                "values": self.values,
            },
            path / "targets.pt",
        )
        write_json(path / "targets.json", self.meta)

    @classmethod
    def load(cls, path: Path) -> Targets:
        """The targets saved at ``path``.

        Raises:
            FileNotFoundError: None there (``fill_eval targets`` makes them).
        """
        if not (path / "targets.pt").is_file():
            raise FileNotFoundError(
                f"no targets at {path}: make them by `fill_eval targets`"
            )
        d = torch.load(path / "targets.pt", weights_only=True)
        spaces = {k: Space(**s) for k, s in d["spaces"].items()}
        meta = json.loads((path / "targets.json").read_text())
        return cls(spaces, d["values"], meta)


def targets_key(sets: FillSets, fconf: FillConf) -> str:
    """The first 12 hex digits of the sha256 of the sets' ``ident`` and the
    settings the targets depend on: the cache's name ``v18c-targets-<key>``."""
    settings = {
        "target_dim": fconf.target_dim,
        "color_block": fconf.color_block,
        "outer": fconf.outer,
        "grid_step": fconf.grid_step,
        "target_pca_seed": TARGET_PCA_SEED,
    }
    text = json.dumps({"sets": dict(sets.ident), "settings": settings}, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def cache_path(base: Path, key: str) -> Path:
    return base / f"v18c-targets-{key}"


def _default_base() -> Path:
    repo = Path(__file__).resolve().parents[2]
    return Path(os.environ.get(DATA_ENV, repo / ".fleet" / "data"))


def block_colors(frames: Tensor, pixels: Tensor, block: int) -> Tensor:
    """The mean colors in ``x / 255`` of the ``block``-pixel blocks of the
    u8 ``frames`` ``(B, 3, H, W)`` wholly in ``pixels`` ``(H, W)`` bool,
    ``(B, 3 n_blocks)`` float64 (row-major blocks, channel-major)."""
    inside = F.avg_pool2d(pixels[None, None].double(), block)[0, 0] == 1
    means = F.avg_pool2d(frames.double() / 255, block)  # (B, 3, H/b, W/b)
    return means[:, :, inside].flatten(1)


def _raw(kind: str, features: Features, images: Tensor, block: int) -> Tensor:
    """The raw features (float64, CPU) of a target ``kind`` of the u8
    images ``(N, 3, h, w)`` (a region's images, or of a ``GREY`` filling:
    color's blocks are those wholly off grey, the region's)."""
    if kind == "incep":
        return torch.cat([features(c).double().cpu() for c in images.split(BATCH)])
    return block_colors(images, torch.ones(images.shape[2:], dtype=torch.bool), block)


def _region_raw(
    kind: str, features: Features, frames: Tensor, pix: Tensor, block: int
) -> Tensor:
    """The raw features of the region ``pix`` of the u8 ``frames``: incep's
    of its image (``region_image``), color's of its blocks."""
    if kind == "incep":
        return _raw(kind, features, region_image(frames, pix), block)
    return block_colors(frames, pix, block)


def _dims(kind: str, fconf: FillConf) -> int | None:
    return fconf.target_dim if kind == "incep" else None


def _e1_targets(
    sets: FillSets, features: Features, fconf: FillConf
) -> tuple[dict[str, Space], dict[str, dict[str, Tensor]]]:
    """E1's targets (module docstring): U, V and I."""
    size, block = sets.grid_last.shape[-1], fconf.color_block
    spaces: dict[str, Space] = {}
    values: dict[str, dict[str, Tensor]] = {}
    for m in E1_MASKS:
        hidden = pixel_mask(EVAL_MASKS[m], size)
        filled = harmonic_fill(sets.grid_last, hidden).round().clamp(0, 255)
        filled = filled.to(torch.uint8)
        for kind in KINDS:
            for side, pix in (("U", hidden), ("V", ~hidden)):
                raw = {
                    s: _region_raw(kind, features, sets.last(s), pix, block)
                    for s in SETS
                }
                key = f"{side}/{m}/{kind}"
                spaces[key] = space = Space.fit(raw["fit"], _dims(kind, fconf))
                values[key] = {s: space(r).float() for s, r in raw.items()}
            interp = _region_raw(kind, features, filled, hidden, block)
            values[f"I/{m}/{kind}"] = {"grid": spaces[f"U/{m}/{kind}"](interp).float()}
    return spaces, values


class WideFrames:
    """The wide data at ``root`` (``exp.lib.tours.wide``): ``band`` (its
    columns a band) and the frames at places."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.index: dict[str, Any] = json.loads((root / INDEX).read_text())
        self.band: int = self.index["geometry"]["band"]
        self._readers: dict[str, U8Reader] = {}

    def frames(self, places: Sequence[tuple[str, int]]) -> Tensor:
        """The frames u8 ``(N, 3, S, S + 2 band)`` at ``places`` (video,
        frame).

        Raises:
            ValueError: A video is not in the data.
        """
        missing = sorted({v for v, _ in places} - set(self.index["videos"]))
        if missing:
            raise ValueError(f"the wide data {self.root} lacks the videos {missing}")
        for v, _ in places:
            if v not in self._readers:
                self._readers[v] = U8Reader(self.root / v)
        return torch.stack([self._readers[v][f] for v, f in places])


def _bands(wide: Tensor, band: int, outer: int) -> dict[str, Tensor]:
    """The bands (``BANDS``) and ``centre`` of wide frames ``(N, 3, S, S +
    2 band)``: ``L_out`` the left band's ``outer`` columns away from the
    square, ``R_out`` the right band's."""
    left, right = wide[..., :band], wide[..., -band:]
    return {
        "L": left,
        "R": right,
        "L_out": left[..., :outer],
        "R_out": right[..., band - outer :],
        "centre": wide[..., band:-band],
    }


def _e2_targets(
    sets: FillSets, features: Features, fconf: FillConf, wide: WideFrames
) -> tuple[dict[str, Space], dict[str, dict[str, Tensor]]]:
    """E2's targets (module docstring): B, C and BI.

    Raises:
        ValueError: The sets' places are unknown, or the wide frames' square
            is not the sets' last frames.
    """
    if sets.places is None:
        raise ValueError("E2's targets need the sets' places")
    block = fconf.color_block
    bands = {}
    for s in SETS:
        b = _bands(wide.frames(sets.places[s]), wide.band, fconf.outer)
        if not torch.equal(b["centre"], sets.last(s)):
            raise ValueError(f"the wide data's square is not the {s} set's last frames")
        bands[s] = b
    spaces: dict[str, Space] = {}
    values: dict[str, dict[str, Tensor]] = {}
    for kind in KINDS:
        for name in (*BANDS, "C"):
            part = "centre" if name == "C" else name
            raw = {s: _raw(kind, features, bands[s][part], block) for s in SETS}
            key = f"C/{kind}" if name == "C" else f"B/{name}/{kind}"
            spaces[key] = space = Space.fit(raw["fit"], _dims(kind, fconf))
            values[key] = {s: space(r).float() for s, r in raw.items()}
        centre = bands["grid"]["centre"]
        for name in BANDS:
            width = bands["grid"][name].shape[-1]
            edge = centre[..., :1] if name.startswith("L") else centre[..., -1:]
            raw = _raw(
                kind, features, edge.expand(-1, -1, -1, width).contiguous(), block
            )
            values[f"BI/{name}/{kind}"] = {
                "grid": spaces[f"B/{name}/{kind}"](raw).float()
            }
    return spaces, values


def build_targets(
    sets: FillSets,
    features: Features,
    fconf: FillConf,
    *,
    wide: WideFrames | None = None,
    have: Targets | None = None,
    record: Mapping[str, Any] | None = None,
) -> Targets:
    """The targets (module docstring) of ``sets`` by ``features`` (FID's
    Inception in use, ``record`` its weights' path and sha256): E1's, and
    E2's with ``wide``; those of ``have`` kept (E1's, or E2's when ``have``
    has them)."""
    spaces = dict(have.spaces) if have else {}
    values = dict(have.values) if have else {}
    meta = (
        dict(have.meta)
        if have
        else {
            "sets": dict(sets.ident),
            "fill": asdict(fconf),
            "inception": dict(record or {}),
            "target_pca_seed": TARGET_PCA_SEED,
        }
    )
    if not (have and have.has("U/")):
        s, v = _e1_targets(sets, features, fconf)
        spaces |= s
        values |= v
    if wide is not None and not (have and have.has("B/")):
        s, v = _e2_targets(sets, features, fconf, wide)
        spaces |= s
        values |= v
        meta["wide"] = {"root": wide.root.name, "index": wide.index}
    return Targets(spaces, values, meta)


# -- the readouts and the scores -----------------------------------------------------


class Reducer:
    """A readout's inputs as they are, or their PCA (``pca``) of ``dim``
    components fit on the fit set from ``READOUT_PCA_SEED`` where wider."""

    def __init__(self, mean: Tensor | None = None, comps: Tensor | None = None) -> None:
        self._mean, self._comps = mean, comps

    @classmethod
    def fit(cls, x: Tensor, dim: int) -> Reducer:
        if x.shape[1] <= dim:
            return cls()
        gen = torch.Generator().manual_seed(READOUT_PCA_SEED)
        return cls(*pca.fit_pca(x.float(), dim, gen=gen))

    def __call__(self, x: Tensor) -> Tensor:
        if self._mean is None or self._comps is None:
            return x
        return pca.encode(x.float(), self._mean, self._comps)


def score(y_hat: Tensor, targets: Targets, key: str, sets: FillSets) -> dict[str, Any]:
    """The row's scores (module docstring) of the candidates' predictions
    ``y_hat`` ``(n, K)`` of the targets ``key``."""
    grid = targets.y(key, "grid")
    y, pred = grid[sets.cand], y_hat.double().cpu()
    center = grid.mean(0)  # the held-out video's, as G's R
    easy = identification(pred, y, center=center)
    hard = identification(
        pred[sets.hard_q],
        grid,
        center=center,
        truth=sets.cand[sets.hard_q],
        among=sets.hard_among,
    )
    return {
        "r2": r2(pred, y),
        "sq_i": (pred - y).square().sum(1).tolist(),
        "id_easy": easy["top1"],
        "id_easy_top5": easy["top5"],
        "hits_easy": easy["hits"],
        "pick_max_easy": easy["pick_max"],
        "id_hard": hard["top1"],
        "hits_hard": hard["hits"],
        "n_hard": len(sets.hard_q),
    }


def gain(pred: float, oracle: float, mean: float) -> float:
    r"""$(\text{pred} - \text{mean}) / (\text{oracle} - \text{mean})$: 1 for
    a prediction as good as the oracle, 0 for one as good as the mean (NaN
    where the oracle is the mean)."""
    den = oracle - mean
    return (pred - mean) / den if den != 0 else math.nan


def _readout(x_fit: Tensor, x_cal: Tensor, targets: Targets, key: str) -> Ridge:
    """The ridge from the inputs to the targets ``key``, its scale chosen on
    the calib set."""
    dev = x_fit.device
    return select_ridge(
        x_fit, targets.y(key, "fit").to(dev), x_cal, targets.y(key, "calib").to(dev)
    )


def _mean_row(rows: Sequence[Mapping[str, Any]], **labels: Any) -> dict[str, Any]:
    """The mean of ``rows`` (numbers, and lists elementwise) with
    ``labels``; ``alpha`` dropped."""
    out: dict[str, Any] = dict(labels)
    for k, v in rows[0].items():
        if k in labels or k == "alpha":
            continue
        if isinstance(v, list):
            cols = zip(*(r[k] for r in rows), strict=True)
            out[k] = [sum(c) / len(rows) for c in cols]
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = sum(r[k] for r in rows) / len(rows)
        else:
            out[k] = v
    return out


def _means(
    rows: list[dict[str, Any]],
    group: str,
    means: Mapping[str, Sequence[str]],
    by: Sequence[str],
) -> list[dict[str, Any]]:
    """The rows of the means over the members of each of ``means`` (the
    rows' ``group``) of each value of the labels ``by``."""
    out = []
    keys = sorted({tuple(r[b] for b in by) for r in rows}, key=str)
    for name, members in means.items():
        for key in keys:
            part = [
                r
                for r in rows
                if r[group] in members and tuple(r[b] for b in by) == key
            ]
            if len(part) == len(members):
                labels = dict(zip(by, key, strict=True)) | {group: name}
                labels |= {k: part[0][k] for k in ("in_dist",) if k in part[0]}
                out.append(_mean_row(part, **labels))
    return out


# -- encoding -----------------------------------------------------------------------


@torch.no_grad()
def _apply(fn: Callable[[Tensor], Tensor], x: Tensor, dev: torch.device) -> Tensor:
    """``fn`` of the stacks u8 ``x`` (CPU), ``BATCH`` at a time on ``dev``,
    float32, without autograd (a code's graph of thousands of stacks is
    never kept, whatever the caller)."""
    return torch.cat([fn(c.to(dev)).float() for c in x.split(BATCH)])


def _hide(mask: Tensor, b: int) -> tuple[Tensor, Tensor]:
    """The visible and the hidden positions ``(b, ·)`` of the fine mask
    ``(T,)`` for a batch of ``b``."""
    return mask_index(mask[None].expand(b, -1))


def _predicted(mods: JEPAModules, mask: Tensor) -> Callable[[Tensor], Tensor]:
    """The predictor's $\\hat z$ ``(B, k d)`` (hidden positions in order) of
    stacks under the fine mask ``mask``."""

    def fn(x: Tensor) -> Tensor:
        z_hat, _ = jepa_predict(mods, early_fusion(x), mask[None].expand(len(x), -1))
        return z_hat.flatten(1)

    return fn


def _context(mods: JEPAModules, mask: Tensor) -> Callable[[Tensor], Tensor]:
    """The context side's tokens of the encoder ``(B, V w)`` (visible
    positions in order) under the fine mask ``mask``."""

    def fn(x: Tensor) -> Tensor:
        keep, _ = _hide(mask, len(x))
        return mods.encoder(early_fusion(x), keep).flatten(1)

    return fn


def _fine(name: str, expand: int, dev: torch.device) -> Tensor:
    """The mask ``name`` spread to the JEPA's grid ``(T,)``."""
    return expand_masks(EVAL_MASKS[name][None], expand)[0].to(dev)


# -- E1 ------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Fitted:
    """Of an E1 mask: its fine positions ``pos``, the reducer and the
    readouts by kind, the candidates' flat target side's tokens there
    ``cand`` ``(n, k d)``, the fit set's mean of them ``fit_mean`` and of
    them reduced ``x_mean`` ``(1, ·)``; with
    the K3 stages, the same of the tokens replaced by their mean over all
    the positions (``red_k3``, ``ridges_k3``, ``cand_k3``)."""

    pos: Tensor
    red: Reducer
    ridges: dict[str, Ridge]
    cand: Tensor
    fit_mean: Tensor
    x_mean: Tensor
    red_k3: Reducer | None = None
    ridges_k3: dict[str, Ridge] | None = None
    cand_k3: Tensor | None = None


def _k3_tokens(z: Tensor) -> Tensor:
    """The tokens ``(B, T, d)`` all replaced by their mean over the
    positions."""
    return z.mean(1, keepdim=True).expand_as(z)


def _target_tokens(
    mods: JEPAModules, sets: FillSets, dev: torch.device
) -> dict[str, Tensor]:
    """The target side's tokens ``(N, T, d)`` of ``fit``, ``calib`` and the
    candidates (``cand``)."""
    stacks = (("fit", sets.fit), ("calib", sets.calib), ("cand", sets.cand_stacks))
    return {s: _apply(mods.tokens, x, dev) for s, x in stacks}


def _fit_mask(
    m: str,
    tok: Mapping[str, Tensor],
    targets: Targets,
    fconf: FillConf,
    kinds: Sequence[str],
    *,
    expand: int,
    k3: bool,
    dev: torch.device,
) -> _Fitted:
    """The readouts $g_U$ of the mask ``m`` of each of ``kinds`` (module
    docstring), of the target side's tokens ``tok`` (and with ``k3``, of
    their position means')."""
    pos = _fine(m, expand, dev).nonzero().flatten()
    flat = {s: z[:, pos].flatten(1) for s, z in tok.items()}
    red = Reducer.fit(flat["fit"], fconf.readout_dim)
    x_fit, x_cal = red(flat["fit"]), red(flat["calib"])
    ridges = {k: _readout(x_fit, x_cal, targets, f"U/{m}/{k}") for k in kinds}
    fitted = _Fitted(
        pos, red, ridges, flat["cand"], flat["fit"].mean(0), x_fit.mean(0, keepdim=True)
    )
    if not k3:
        return fitted
    flat3 = {s: _k3_tokens(z)[:, pos].flatten(1) for s, z in tok.items()}
    red3 = Reducer.fit(flat3["fit"], fconf.readout_dim)
    x3_fit, x3_cal = red3(flat3["fit"]), red3(flat3["calib"])
    ridges3 = {k: _readout(x3_fit, x3_cal, targets, f"U/{m}/{k}") for k in kinds}
    return replace(fitted, red_k3=red3, ridges_k3=ridges3, cand_k3=flat3["cand"])


@torch.no_grad()
def e1_rows(
    mods: JEPAModules,
    sets: FillSets,
    targets: Targets,
    fconf: FillConf,
    *,
    expand: int,
    k_run: int,
    dev: torch.device,
    kinds: Mapping[str, Sequence[str]] | None = None,
    k3: bool = False,
) -> list[dict[str, Any]]:
    """E1's rows (module docstring) of the JEPA ``mods`` (masks spread by
    ``expand``; ``in_dist`` where a mask hides the run's ``k_run``); the
    targets of each mask ``kinds[mask]`` (default ``KINDS``); with ``k3``
    the stages ``oracle_k3`` (the tokens replaced by their mean over all the
    positions, its own readout) and ``pred_k3`` (the predictor's hidden
    positions replaced by their mean, $g_U$), and ``swap``'s ``n_tie``."""
    tok = _target_tokens(mods, sets, dev)
    n = len(sets.cand)
    rows: list[dict[str, Any]] = []
    fitted: dict[str, _Fitted] = {}
    for m in E1_MASKS:
        mask_kinds = KINDS if kinds is None else kinds[m]
        f = fitted[m] = _fit_mask(
            m, tok, targets, fconf, mask_kinds, expand=expand, k3=k3, dev=dev
        )
        mask = _fine(m, expand, dev)
        k_d = _apply(_predicted(mods, mask), sets.cand_stacks, dev)
        pred = f.red(k_d)
        stages = {
            "oracle": f.red(f.cand),
            "pred": pred,
            "mean": f.x_mean.expand(n, -1),
            "shuf": pred.roll(n // 2, 0),
        }
        if k3:
            z_hat = k_d.reshape(n, len(f.pos), -1)
            stages["pred_k3"] = f.red(_k3_tokens(z_hat).flatten(1))
        in_dist = int(EVAL_MASKS[m].sum()) == k_run
        for kind in mask_kinds:
            key = f"U/{m}/{kind}"
            g = f.ridges[kind]
            labels = {"target": kind, "mask": m, "in_dist": in_dist}
            for stage, x in stages.items():
                row = labels | {"stage": stage, "alpha": g.alpha}
                rows.append(row | score(predict(g, x), targets, key, sets))
            if k3:
                assert f.red_k3 is not None and f.ridges_k3 is not None
                assert f.cand_k3 is not None
                g3 = f.ridges_k3[kind]
                row = labels | {"stage": "oracle_k3", "alpha": g3.alpha}
                rows.append(
                    row | score(predict(g3, f.red_k3(f.cand_k3)), targets, key, sets)
                )
            if f"I/{m}/{kind}" in targets.values:
                interp = targets.y(f"I/{m}/{kind}", "grid")[sets.cand]
                row = labels | {"stage": "interp", "alpha": None}
                rows.append(row | score(interp, targets, key, sets))
    rows += _means(rows, "mask", MEANS, ("target", "stage"))
    swap_kinds = KINDS if kinds is None else kinds[SWAP_PAIRS["S_1"][0]]
    rows += _swap_rows(
        mods, sets, targets, fitted, swap_kinds, expand=expand, k_run=k_run,
        dev=dev, k3=k3,
    )  # fmt: skip
    return rows


def _deviation(x: Tensor) -> Tensor:
    """The rows of ``x`` ``(n, D)`` less their mean, float64; less the first
    row first, so that equal rows give exactly 0 (a constant stage ties in
    ``swap`` instead of choosing by its rounding)."""
    d = x.double() - x[:1].double()
    return d - d.mean(0)


def _swap_parts(
    mods: JEPAModules,
    sets: FillSets,
    targets: Targets,
    fitted: Mapping[str, _Fitted],
    kinds: Sequence[str],
    *,
    expand: int,
    dev: torch.device,
    stages: Sequence[str] = ("oracle", "pred", "mean", "shuf"),
) -> Iterator[tuple[str, str, Tensor, Tensor, Tensor]]:
    r"""Of each pair of ``SWAP_PAIRS``, each direction $(A, B)$, each kind
    and each of ``stages`` (``_swap_rows``): the kind, the stage, the
    readout's parts $u_A B_A$, $u_B B_A$ ``(n, K)`` and the truth $t$ ``(n,
    K)`` (CPU, float64). The K3 stages: ``oracle_k3`` of the tokens'
    position means through their own reducer and readout, ``pred_k3`` of the
    predictor's hidden positions replaced by their mean (both quarters'),
    through $A$'s; the inputs at $A$ and at $B$ are then the same tensor."""
    n = len(sets.cand)
    for s, (a, b) in SWAP_PAIRS.items():
        mask = _fine(s, expand, dev)
        hide = mask.nonzero().flatten()
        k_d = _apply(_predicted(mods, mask), sets.cand_stacks, dev)  # (n, k d)
        z_hat = k_d.reshape(n, len(hide), -1)
        z_k3 = _k3_tokens(z_hat)
        at = {q: z_hat[:, torch.isin(hide, fitted[q].pos)].flatten(1) for q in (a, b)}
        at3 = {q: z_k3[:, torch.isin(hide, fitted[q].pos)].flatten(1) for q in (a, b)}
        for u, w in ((a, b), (b, a)):
            own, other, red = fitted[u], fitted[w], fitted[u].red
            pred = {q: red(at[q]) for q in (u, w)}
            inputs: dict[str, tuple[Tensor, Tensor]] = {
                "oracle": (red(own.cand), red(other.cand)),
                "pred": (pred[u], pred[w]),
                "mean": (
                    red(own.fit_mean[None]).expand(n, -1),
                    red(other.fit_mean[None]).expand(n, -1),
                ),
                "shuf": (pred[u].roll(n // 2, 0), pred[w].roll(n // 2, 0)),
            }
            if "pred_k3" in stages:
                inputs["pred_k3"] = (red(at3[u]), red(at3[w]))
            if "oracle_k3" in stages:
                assert own.red_k3 is not None and own.cand_k3 is not None
                assert other.cand_k3 is not None
                inputs["oracle_k3"] = (
                    own.red_k3(own.cand_k3),
                    own.red_k3(other.cand_k3),
                )
            for kind in kinds:
                grid = targets.y(f"U/{u}/{kind}", "grid")
                truth = grid[sets.cand] - grid.mean(0)
                for stage in stages:
                    if stage == "oracle_k3":
                        assert own.ridges_k3 is not None
                        b_u = own.ridges_k3[kind].b
                    else:
                        b_u = own.ridges[kind].b
                    x_u, x_w = inputs[stage]
                    part_u, part_w = (
                        (_deviation(x) @ b_u.to(x.device)).cpu() for x in (x_u, x_w)
                    )
                    yield kind, stage, part_u, part_w, truth


def _swap_rows(
    mods: JEPAModules,
    sets: FillSets,
    targets: Targets,
    fitted: Mapping[str, _Fitted],
    kinds: Sequence[str] = KINDS,
    *,
    expand: int,
    k_run: int,
    dev: torch.device,
    k3: bool = False,
) -> list[dict[str, Any]]:
    r"""The rows of ``swap`` (mask ``S``; module docstring) of each kind and
    stage but ``interp`` (``_swap_parts``), and with ``k3`` of ``oracle_k3``
    and ``pred_k3`` too, with ``n_tie`` (the candidates tied in all four).

    Of the pair $(A, B)$ and a stage: the inputs at $A$ and at $B$ (flat in
    order of position, the two quarters of the same shape) through $A$'s
    reducer, each less its mean over the candidates ($u_A$, $u_B$; per
    coordinate, so a position's constant offset, the yardstick ``mean``
    among them, is out); the readout's parts $u_A B_A$, $u_B B_A$; the truth $t$, $A$'s
    targets less the grid's mean (``score``'s centre). A code whose inputs
    at $A$ and $B$ differ by a constant per position (K3: $z_{n,p} =
    h(x_n) + g(p)$) has $u_A = u_B$ and ties: 1/2, whatever the
    distributions of the two quarters (exactly where the tokens are equal,
    as in the stub's test). So does ``mean`` (0 for both). Parts at $A$ and
    $B$ that differ by what is unrelated to $t$ give 1/2 in expectation,
    whatever their sizes (``metrics.swap``)."""
    stages = ("oracle", "pred", "mean", "shuf")
    if k3:
        stages += ("oracle_k3", "pred_k3")
    values: dict[tuple[str, str], list[Tensor]] = {}
    for kind, stage, part_u, part_w, truth in _swap_parts(
        mods, sets, targets, fitted, kinds, expand=expand, dev=dev, stages=stages
    ):
        values.setdefault((kind, stage), []).append(swap(part_u, part_w, truth)[1])
    in_dist = int(EVAL_MASKS["S_1"].sum()) == k_run
    rows = []
    for (kind, stage), v in values.items():
        per = torch.stack(v).mean(0)
        row = {
            "target": kind,
            "mask": "S",
            "in_dist": in_dist,
            "stage": stage,
            "swap": float(per.mean()),
            "swap_i": per.tolist(),
        }
        if k3:
            row["n_tie"] = int((torch.stack(v) == 0.5).all(0).sum())
        rows.append(row)
    return rows


# -- E1's diagnosis of shuf ------------------------------------------------------------------


def _wins(x: Tensor) -> Tensor:
    """1 where positive, 1/2 where 0, 0 where negative (``metrics.swap``'s)."""
    return (x > 0).double() + 0.5 * (x == 0).double()


def shift_curves(part_u: Tensor, part_w: Tensor, truth: Tensor) -> dict[str, Any]:
    r"""The diagnosis of ``shuf`` (``plans/jepa.md``, 「ずらしの診断」) of one
    pair, direction, kind and stage: $D$ = ``part_u`` − ``part_w``, $T$ =
    ``truth`` ``(n, K)``, $P = D T^\top$ ``(n, n)`` float64 ($P_{ji} = d_j
    \cdot t_i$). ``C`` $C(\tau) = \frac1n \sum_i P_{(i + \tau) \bmod n,
    i}$ and ``s`` $s(\tau)$ the mean of their wins (1, 1/2 on 0, 0),
    $\tau = 0, \dots, n - 1$ ($s(0)$ is ``swap``); ``C_nw``, ``s_nw`` the
    same of $i + \tau < n$ only, $\tau \le n - 64$; ``C_off``, ``s_off``
    over all $i \ne j$; ``resid`` $\max_i \lvert \sum_j P_{ji} \rvert /
    \max \lvert P \rvert$ (0 up to rounding: ``_deviation`` centres $d$)."""
    d = part_u.double() - part_w.double()
    p = d @ truth.double().T  # (n, n)
    n = len(p)
    i = torch.arange(n)
    taus = torch.arange(n)
    shifted = p[(i[None] + taus[:, None]) % n, i[None].expand(n, -1)]  # (tau, i)
    c_nw, s_nw = [], []
    for tau in range(max(n - 63, 1)):
        diag = p[i[: n - tau] + tau, i[: n - tau]]
        c_nw.append(float(diag.mean()))
        s_nw.append(float(_wins(diag).mean()))
    off = ~torch.eye(n, dtype=torch.bool)
    return {
        "C": shifted.mean(1).tolist(),
        "s": _wins(shifted).mean(1).tolist(),
        "C_nw": c_nw,
        "s_nw": s_nw,
        "C_off": float(p[off].mean()),
        "s_off": float(_wins(p[off]).mean()),
        "resid": float(p.sum(0).abs().max() / p.abs().max().clamp(min=1e-300)),
    }


DIAG_STAGES = ("oracle", "pred")


@torch.no_grad()
def e1diag_rows(
    mods: JEPAModules,
    sets: FillSets,
    targets: Targets,
    fconf: FillConf,
    *,
    expand: int,
    dev: torch.device,
) -> list[dict[str, Any]]:
    """The rows of ``shift_curves`` of each kind (``KINDS``) and stage of
    ``DIAG_STAGES``, averaged over the 2 pairs and 2 directions of
    ``swap`` (``resid`` their largest), of the JEPA ``mods``."""
    tok = _target_tokens(mods, sets, dev)
    quarters = sorted({q for pair in SWAP_PAIRS.values() for q in pair})
    fitted = {
        m: _fit_mask(m, tok, targets, fconf, KINDS, expand=expand, k3=False, dev=dev)
        for m in quarters
    }
    got: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for kind, stage, part_u, part_w, truth in _swap_parts(
        mods, sets, targets, fitted, KINDS, expand=expand, dev=dev, stages=DIAG_STAGES
    ):
        got.setdefault((kind, stage), []).append(shift_curves(part_u, part_w, truth))
    rows = []
    for (kind, stage), curves in got.items():
        row: dict[str, Any] = {"target": kind, "stage": stage}
        for key in ("C", "s", "C_nw", "s_nw"):
            row[key] = torch.tensor([c[key] for c in curves]).mean(0).tolist()
        for key in ("C_off", "s_off"):
            row[key] = sum(c[key] for c in curves) / len(curves)
        row["resid"] = max(c["resid"] for c in curves)
        row["swap"] = row["s"][0]
        rows.append(row)
    return rows


def _diag_picture(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """``e1diag-<step>.png``: of each row (kind, stage), a panel of $C$ and
    one of $s$ over $\\tau$ (one pixel each), with the shift (blue) and
    without the wrap (orange); grey the line of 0 (of $C$) or 1/2 (of $s$),
    green $s_{\\mathrm{off}}$. Each panel spans its curves' range."""
    height, gap = 96, 4
    width = len(rows[0]["C"])
    panels = []
    for r in rows:
        for key, ref in (("C", 0.0), ("s", 0.5)):
            canvas = torch.full((height, width, 3), 255, dtype=torch.uint8)
            curves = [torch.tensor(r[key]), torch.tensor(r[f"{key}_nw"])]
            lines = [ref] + ([r["s_off"]] if key == "s" else [])
            lo = min(min(float(c.min()) for c in curves), *lines)
            hi = max(max(float(c.max()) for c in curves), *lines)
            span = hi - lo if hi > lo else 1.0

            def ys(v: Tensor, lo: float = lo, span: float = span) -> Tensor:
                return (1 - (v - lo) / span) * (height - 1)

            for value, color in zip(
                lines, ((160, 160, 160), (0, 150, 0)), strict=False
            ):
                xy = torch.tensor([[0.0, 0.0], [width - 1.0, 0.0]])
                xy[:, 1] = ys(torch.tensor(value))
                draw_polyline_(canvas, xy, color)
            for c, color in zip(curves, ((30, 60, 200), (230, 120, 0)), strict=True):
                xy = torch.stack([torch.arange(len(c)).double(), ys(c.double())], 1)
                draw_polyline_(canvas, xy, color)
            panels.append(canvas)
            panels.append(torch.full((gap, width, 3), 200, dtype=torch.uint8))
    write_png(path, torch.cat(panels[:-1]))


# -- E1f: the readouts fit on the predictions ---------------------------------------------------


QUARTER_NAMES = ("tl", "tr", "bl", "br")  # the quarters Q_<q> (sem_eval's order)
E1F_K = (4, 8, 12)
E1F_STAGES = ("pred", "pool", "pred_k3", "vis_color", "oracle", "mean")
_ROW_HALF = {"tl": "H_t", "tr": "H_t", "bl": "H_b", "br": "H_b"}
_COL_HALF = {"tl": "H_l", "tr": "H_r", "bl": "H_l", "br": "H_r"}


def e1f_masks() -> dict[tuple[str, int], Tensor]:
    """The coarse masks ``(16,)`` of E1f by (quarter, $k$): $k$ = 4 the
    quarter $Q_q$, 8 its row's half, 12 that and its column's half (the
    diagonal quarter alone visible)."""
    out = {}
    for q in QUARTER_NAMES:
        row, col = EVAL_MASKS[_ROW_HALF[q]], EVAL_MASKS[_COL_HALF[q]]
        out[(q, 4)] = EVAL_MASKS[f"Q_{q}"]
        out[(q, 8)] = row
        out[(q, 12)] = row | col
    return out


def e1f_inputs(z_hat: Tensor) -> dict[str, Tensor]:
    r"""The inputs of the stages ``pred``, ``pool`` and ``pred_k3`` of the
    predictor's outputs at a quarter's positions $\hat t_p$ ``(n, P, d)``
    (in order of position): them flat ``(n, P d)``, their mean over the
    positions ``(n, d)``, and them all replaced by that mean ``(n, P d)``.

    For a K3 family $\hat t_p = A_p u$, ``pred`` is $A u$ ($A$ the $A_p$
    stacked) and ``pool`` $\bar A u$: with $\bar A$ invertible (and well
    conditioned) both are one-to-one linear functions of $u$, of the same
    column space, so a readout fit on either predicts the same. A singular
    $\bar A$ breaks this (``pool`` loses what ``pred`` keeps; e1f's rows
    give ``pool``'s condition number)."""
    pool = z_hat.mean(1)
    return {
        "pred": z_hat.flatten(1),
        "pool": pool,
        "pred_k3": pool[:, None].expand_as(z_hat).flatten(1),
    }


def e1f_fits(
    x: Mapping[str, Mapping[str, Tensor]], y_fit: Tensor, y_cal: Tensor
) -> dict[str, Ridge]:
    r"""The readouts of E1f's stages ``x`` (stage, then ``"fit"`` and
    ``"calib"``: the inputs) to ``y``: each stage's scale of
    ``ALPHA_SCALES`` chosen by ``centered_r2`` on calib, but ``pred_k3``'s,
    $P c$ for ``pool``'s $c$ ($P$ the positions, ``pred_k3``'s width over
    ``pool``'s), at which the two are the same fit.

    With ``pool`` $X$ ``(n, d)`` and ``pred_k3`` $X' = X J$, $J = (I_d,
    \dots, I_d)$ ``(d, P d)`` ($J J^	op = P I_d$): $\mathrm{tr}(X'^	op X')
    = P\,\mathrm{tr}(X^	op X)$, so a scale $c'$ is $lpha' = c'
    \mathrm{tr}(X^	op X) / d$; and by the push-through identity $X'
    (X'^	op X' + lpha' I)^{-1} X'^	op = X (X^	op X + rac{lpha'}{P}
    I)^{-1} X^	op$. At $c' = P c$, $lpha' / P$ is ``pool``'s $lpha$:
    the predictions are the same up to rounding. (At the same $c$, the
    penalty on $u$ would be $1/P$ of ``pool``'s, a scale off the grid.)"""
    fits: dict[str, Ridge] = {}
    for stage, xs in x.items():
        if stage != "pred_k3":
            fits[stage] = select_ridge(
                xs["fit"], y_fit, xs["calib"], y_cal, score=centered_r2
            )
    if "pred_k3" in x:
        xk, xp = x["pred_k3"]["fit"], x["pool"]["fit"]
        positions = xk.shape[1] // xp.shape[1]
        fits["pred_k3"] = fit_ridge(xk, y_fit, positions * fits["pool"].scale)
    return fits


def _cond(x: Tensor) -> float:
    """The condition number of the covariance of ``x`` ``(n, D)``."""
    return float(torch.linalg.cond(torch.atleast_2d(torch.cov(x.double().T))))


@torch.no_grad()
def e1f_rows(
    mods: JEPAModules,
    stacks: Mapping[str, Tensor],
    targets: Targets,
    *,
    expand: int,
    k_run: int,
    dev: torch.device,
) -> list[dict[str, Any]]:
    r"""E1f's rows (``plans/jepa.md``, 「当てはめ直す補完のゲート」) of the JEPA
    ``mods`` (masks spread by ``expand``): the stacks of ``fit`` and of the
    split's ``calib``, ``sel``, ``final``; the targets ``quad/<q>`` (their
    ``fit`` and ``grid``, the split in ``meta["sets_split"]``). Of each
    $k$ of ``E1F_K`` and stage of ``E1F_STAGES``: of each quarter $q$ under
    ``e1f_masks()[(q, k)]``, a ridge from the stage's inputs (``vis_color``
    the visible coarse cells' mean colors, ``oracle`` the target side's
    tokens at $Q_q$) to ``quad/<q>`` (``e1f_fits``: its scale chosen by
    ``centered_r2`` on calib, ``pred_k3``'s tied to ``pool``'s); the items'
    $e_i$ and $v_i$ of sel and final summed over the quarters into one $m$.
    ``mean`` is the constant ($m$ exactly 0). ``scale`` and ``alpha`` by
    quarter; ``pool``'s ``cond`` by quarter, its inputs' covariance's
    condition number on fit (a near-singular $\bar A$ makes ``pred`` beat
    ``pool`` even of a K3 family). ``id_easy_sel``: the quarters'
    predictions side by side among sel's."""
    split = targets.meta["sets_split"]
    idx = {s: torch.tensor(split[s]["idx"]) for s in ("calib", "sel", "final")}
    sets = ("fit", "calib", "sel", "final")
    size = stacks["fit"].shape[-1]
    tok = {s: _apply(mods.tokens, stacks[s], dev) for s in sets}
    color = ColorCode(size // COARSE)
    cells = {s: _apply(color.tokens, stacks[s], dev) for s in sets}  # (N, 16, 3n)
    masks = e1f_masks()
    sums: dict[tuple[int, str], dict[str, Any]] = {}
    preds: dict[tuple[int, str], dict[str, Tensor]] = {}
    for (q, k), coarse in masks.items():
        fine = expand_masks(coarse[None], expand)[0].to(dev)
        hide = fine.nonzero().flatten()
        quarter = expand_masks(EVAL_MASKS[f"Q_{q}"][None], expand)[0].to(dev)
        at = torch.isin(hide, quarter.nonzero().flatten())
        visible = (~coarse).nonzero().flatten().to(dev)
        key = f"quad/{q}"
        y = {"fit": targets.y(key, "fit")}
        grid = targets.y(key, "grid")
        y |= {s: grid[idx[s]] for s in ("calib", "sel", "final")}
        inputs: dict[str, dict[str, Tensor]] = {}
        for s in sets:
            k_d = _apply(_predicted(mods, fine), stacks[s], dev)
            z_hat = k_d.reshape(len(k_d), len(hide), -1)[:, at]
            got = e1f_inputs(z_hat)
            got["vis_color"] = cells[s][:, visible].flatten(1)
            got["oracle"] = tok[s][:, quarter].flatten(1)
            for stage, x in got.items():
                inputs.setdefault(stage, {})[s] = x
        fits = e1f_fits(inputs, y["fit"].to(dev), y["calib"].to(dev))
        for stage in E1F_STAGES:
            acc = sums.setdefault((k, stage), {"alpha": {}, "scale": {}})
            if stage == "mean":
                p = {s: torch.zeros(len(y[s]), 1, dtype=torch.float64) for s in sets}
                acc["alpha"][q] = acc["scale"][q] = None
            else:
                g = fits[stage]
                p = {s: predict(g, inputs[stage][s]).cpu() for s in sets}
                acc["alpha"][q], acc["scale"][q] = g.alpha, g.scale
            if stage == "pool":
                acc.setdefault("cond", {})[q] = _cond(inputs[stage]["fit"])
            for s in ("calib", "sel", "final"):
                e, v = centered_terms(p[s], y[s])
                acc[f"e_{s}"] = acc.get(f"e_{s}", 0) + e
                acc[f"v_{s}"] = acc.get(f"v_{s}", 0) + v
            preds.setdefault((k, stage), {})[q] = p["sel"]
    center = torch.cat([targets.y(f"quad/{q}", "grid") for q in QUARTER_NAMES], 1)
    truth = center[idx["sel"]]
    rows = []
    for (k, stage), acc in sums.items():
        row: dict[str, Any] = {"k": k, "stage": stage, "in_dist": k == k_run}
        row["alpha"], row["scale"] = acc["alpha"], acc["scale"]
        if "cond" in acc:
            row["cond"] = acc["cond"]
        for s in ("calib", "sel", "final"):
            row[f"m_{s}"] = float(1 - acc[f"e_{s}"].sum() / acc[f"v_{s}"].sum())
        for s in ("sel", "final"):
            row[f"e_{s}"] = acc[f"e_{s}"].tolist()
            row[f"v_{s}"] = acc[f"v_{s}"].tolist()
        row["id_easy_sel"] = None
        if stage != "mean":
            side = torch.cat([preds[(k, stage)][q] for q in QUARTER_NAMES], 1)
            found = identification(side, truth, center=center.mean(0))
            row["id_easy_sel"] = found["top1"]
        rows.append(row)
    return rows


# -- E1e and E2 ------------------------------------------------------------------------


def _readout_rows(
    x: Mapping[str, Tensor],
    targets: Targets,
    keys: Mapping[str, str],
    sets: FillSets,
    fconf: FillConf,
    labels: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """The rows of a readout of the inputs ``x`` (``fit``, ``calib``,
    ``cand``) to each kind's targets ``keys[kind]``."""
    red = Reducer.fit(x["fit"], fconf.readout_dim)
    x_fit, x_cal, x_cand = red(x["fit"]), red(x["calib"]), red(x["cand"])
    rows = []
    for kind, key in keys.items():
        g = _readout(x_fit, x_cal, targets, key)
        row = dict(labels) | {"target": kind, "alpha": g.alpha}
        rows.append(row | score(predict(g, x_cand), targets, key, sets))
    return rows


def _sets_apply(
    fn: Callable[[Tensor], Tensor], sets: FillSets, dev: torch.device
) -> dict[str, Tensor]:
    """``fn`` of the stacks of ``fit``, ``calib`` and the candidates
    (``cand``)."""
    stacks = {"fit": sets.fit, "calib": sets.calib, "cand": sets.cand_stacks}
    return {s: _apply(fn, x, dev) for s, x in stacks.items()}


def _target_inputs(targets: Targets, key: str, sets: FillSets) -> dict[str, Tensor]:
    """A target's values as a readout's inputs (``ext``)."""
    return {
        "fit": targets.y(key, "fit"),
        "calib": targets.y(key, "calib"),
        "cand": targets.y(key, "grid")[sets.cand],
    }


type Inputs = Callable[[Tensor], Tensor] | Mapping[str, Tensor]


def _inputs_of(got: Inputs, sets: FillSets, dev: torch.device) -> Mapping[str, Tensor]:
    """The inputs of each set: ``got`` itself, or its function of the
    stacks applied (``_sets_apply``)."""
    return _sets_apply(got, sets, dev) if callable(got) else got


@torch.no_grad()
def e1e_rows(
    inputs: Callable[[str], Inputs] | None,
    code: str,
    sets: FillSets,
    targets: Targets,
    fconf: FillConf,
    *,
    k_run: int | None,
    dev: torch.device,
) -> list[dict[str, Any]]:
    """E1e's rows of the code ``code``: ``inputs(mask)`` is a function of the
    stacks (the code's of the visible part) or the inputs of each set
    (``fit``, ``calib``, ``cand``); None for ``ext``, the visible region's
    targets of each kind to the hidden one's of the same kind."""
    rows: list[dict[str, Any]] = []
    for m in E1_MASKS:
        in_dist = k_run is not None and int(EVAL_MASKS[m].sum()) == k_run
        labels = {"mask": m, "in_dist": in_dist, "code": code}
        if inputs is not None:
            # no name holds a mask's inputs past its call: the next mask's
            # are made after these are freed (arm F's Q: about 6 GB)
            keys = {kind: f"U/{m}/{kind}" for kind in KINDS}
            rows += _readout_rows(
                _inputs_of(inputs(m), sets, dev), targets, keys, sets, fconf, labels
            )
            continue
        for kind in KINDS:
            x = _target_inputs(targets, f"V/{m}/{kind}", sets)
            keys = {kind: f"U/{m}/{kind}"}
            rows += _readout_rows(x, targets, keys, sets, fconf, labels)
    return rows + _means(rows, "mask", MEANS, ("target", "code"))


@torch.no_grad()
def e2_rows(
    code: str,
    x: Mapping[str, Tensor] | None,
    sets: FillSets,
    targets: Targets,
    fconf: FillConf,
) -> list[dict[str, Any]]:
    """E2's rows of the code ``code`` of the inputs ``x`` (``fit``,
    ``calib``, ``cand``); with ``x`` None: ``ext`` (the whole frame's
    targets of each kind to the band's of the same kind), ``interp`` and
    ``mean`` (the predictions themselves)."""
    rows: list[dict[str, Any]] = []
    for band in BANDS:
        for kind in KINDS:
            key = f"B/{band}/{kind}"
            labels = {"band": band, "code": code}
            if x is not None or code == "ext":
                inp = _target_inputs(targets, f"C/{kind}", sets) if x is None else x
                rows += _readout_rows(inp, targets, {kind: key}, sets, fconf, labels)
                continue
            if code == "interp":
                y_hat = targets.y(f"BI/{band}/{kind}", "grid")[sets.cand]
            else:
                fit = targets.y(key, "fit")
                y_hat = fit.mean(0, keepdim=True).expand(len(sets.cand), -1)
            row = labels | {"target": kind, "alpha": None}
            rows.append(row | score(y_hat, targets, key, sets))
    return rows + _means(rows, "band", BAND_MEANS, ("target", "code"))


# -- E1g ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Decoder:
    """G's decoder of a code (``gen_eval``'s ``ckpt/dec-….pt``): the net
    and its standardizer."""

    net: FlowUNet
    mean: Tensor
    sd: float

    @classmethod
    def load(cls, path: Path, dev: torch.device) -> Decoder:
        d = torch.load(path, map_location="cpu", weights_only=True)
        net = FlowUNet(d["code_dim"], **d["net_kw"])
        net.load_state_dict(d["net"])
        return cls(net.to(dev).eval(), d["mean"].to(dev), float(d["sd"]))


@torch.no_grad()
def _filled(
    mods: JEPAModules, x: Tensor, mask: Tensor, z_mean: Tensor
) -> dict[str, Tensor]:
    """The codes ``(B, T d)`` of the stacks ``x`` whose hidden positions of
    the fine ``mask`` are filled (``E1G_STAGES`` but ``full``; module
    docstring), the target side's ``full`` too."""
    images = early_fusion(x)
    full = mods.proj(mods.encoder(images))  # (B, T, d)
    keep, hide = _hide(mask, len(x))
    context = mods.encoder(images, keep)
    ctx = mods.proj(context)
    filled = {
        "oracle": full.gather(1, hide[..., None].expand(-1, -1, full.shape[2])),
        "pred": mods.predictor(context, keep, hide),
        "mean": z_mean[hide],
    }
    out = {"full": full.flatten(1)}
    for stage, hidden in filled.items():
        z = torch.empty_like(full)
        z.scatter_(1, keep[..., None].expand_as(ctx), ctx)
        z.scatter_(1, hide[..., None].expand_as(hidden), hidden)
        out[stage] = z.flatten(1)
    return out


@torch.no_grad()
def e1g_rows(
    mods: JEPAModules,
    sets: FillSets,
    targets: Targets,
    dec: Decoder,
    features: Features,
    fconf: FillConf,
    gconf: GenConf,
    *,
    expand: int,
    to_code: Callable[[Tensor], Tensor],
    dev: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Tensor]]:
    """E1g's rows (module docstring) and the drawn images u8 ``(n, 3, S,
    S)`` by ``"<mask>/<stage>"`` of ``PICTURE_MASKS`` (and ``full``):
    ``to_code`` the run's ``z`` of the flat tokens (``zpca``'s PCA for an
    arm M), G's ``gconf`` its samples and sample steps."""
    n, size = len(sets.cand), sets.grid_last.shape[-1]
    noise = torch.randn(
        n,
        gconf.samples,
        3,
        size,
        size,
        generator=torch.Generator().manual_seed(SAMPLE_SEED),
    )[:, 0]
    z_mean = _apply(mods.tokens, sets.fit, dev).mean(0)  # (T, d)

    def draw(z: Tensor, i: int) -> Tensor:
        c = (to_code(z) - dec.mean) / dec.sd
        x = euler_sample(dec.net, noise[i : i + len(z)].to(dev), c, gconf.sample_steps)
        return ((x.clamp(-1, 1) + 1) / 2 * 255).round().to(torch.uint8).cpu()

    drawn: dict[str, list[Tensor]] = {}
    for i in range(0, n, gconf.sample_batch):
        x = sets.cand_stacks[i : i + gconf.sample_batch].to(dev)
        for m in E1_MASKS:
            codes = _filled(mods, x, _fine(m, expand, dev), z_mean)
            if m == E1_MASKS[0]:
                drawn.setdefault("full", []).append(draw(codes["full"], i))
            for stage in E1G_STAGES[1:]:
                drawn.setdefault(f"{m}/{stage}", []).append(draw(codes[stage], i))
    images = {k: torch.cat(v) for k, v in drawn.items()}
    rows = []
    for m in E1_MASKS:
        hidden = pixel_mask(EVAL_MASKS[m], size)
        for stage in E1G_STAGES:
            img = images["full" if stage == "full" else f"{m}/{stage}"]
            for kind in KINDS:
                row = {"target": kind, "mask": m, "stage": stage}
                key = f"U/{m}/{kind}"
                raw = _region_raw(kind, features, img, hidden, fconf.color_block)
                row |= score(targets.spaces[key](raw), targets, key, sets)
                vis_key = f"V/{m}/{kind}"
                raw = _region_raw(kind, features, img, ~hidden, fconf.color_block)
                vis = score(targets.spaces[vis_key](raw), targets, vis_key, sets)
                row |= {"vis_easy": vis["id_easy"], "vis_hits": vis["hits_easy"]}
                rows.append(row)
    shown = {
        k: v
        for k, v in images.items()
        if k == "full" or k.split("/")[0] in PICTURE_MASKS
    }
    return rows, shown


def _e1g_picture(
    path: Path, sets: FillSets, images: Mapping[str, Tensor], mask: str, fconf: FillConf
) -> None:
    """``g-<step>-<mask>.png``: ``picture_rows`` candidates evenly spaced;
    the true last frame, ``full``, ``oracle``, ``pred``, ``mean``."""
    n = len(sets.cand)
    rows = torch.arange(fconf.picture_rows) * (n // fconf.picture_rows)
    cols = [sets.grid_last[sets.cand[rows]]]
    cols += [images["full" if s == "full" else f"{mask}/{s}"][rows] for s in E1G_STAGES]
    grid = torch.stack(cols, 1).float() / 255
    write_png(path, tile(grid, fconf.picture_scale))


# -- the jobs -------------------------------------------------------------------------


def _conf_of(run_dir: Path, device: str | None) -> RunConf:
    conf = from_dict(yaml.safe_load((run_dir / "config.yaml").read_text()))
    return conf if device is None else replace(conf, device=device)


def _yard_conf(device: str | None) -> RunConf:
    d = v16_config()
    d["device"] = device or "cuda"
    return from_dict(d)


def _write_rows(
    path: Path, rows: list[dict[str, Any]], stamp: Mapping[str, Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, [dict(stamp) | r for r in rows])


class _Context:
    """The sets, targets and record of an evaluation of ``conf`` into
    ``out_dir`` (``OUT`` and ``OUT_SIDE`` there)."""

    def __init__(
        self,
        conf: RunConf,
        out_dir: Path,
        *,
        fconf: FillConf,
        root: Path | None,
        cache: Path | None,
        spacing: Mapping[str, float] | None,
        span: float = SPAN,
    ) -> None:
        self.conf, self.fconf = conf, fconf
        self.dev = torch.device(conf.device)
        self.determinism = configure_determinism(conf.device)
        root = tours_root(conf.data.tours.data_version) if root is None else root
        self.root = root
        self.fill, self.side = out_dir / OUT, out_dir / OUT_SIDE
        self.extra: dict[str, Any] = {}  # of the sem split: its name and targets
        self.sets = fill_sets(conf, root, fconf, span=span, spacing=spacing)
        self.key = targets_key(self.sets, fconf)
        base = _default_base() if cache is None else cache
        self.targets = Targets.load(cache_path(base, self.key))
        if self.targets.meta["sets"] != json.loads(json.dumps(dict(self.sets.ident))):
            raise ValueError(f"the targets {self.key} are not of these sets")

    def depends(self) -> dict[str, Any]:
        """What the rows depend on beside the checkpoint: the settings, the
        targets' key, the seeds (``z_w``'s $W$ too) and the candidates."""
        return json.loads(
            json.dumps(
                {
                    "fill": asdict(self.fconf),
                    "targets_key": self.key,
                    "seeds": {
                        "readout_pca": READOUT_PCA_SEED,
                        "target_pca": TARGET_PCA_SEED,
                        "w": W_SEED,
                        "w_dim": W_DIM,
                        "w_chunk": W_CHUNK,
                    },
                    "cand": self.sets.cand.tolist(),
                    "n_hard": len(self.sets.hard_q),
                }
                | self.extra
            )
        )

    def check(self, out: Path) -> bool:
        """Whether ``out/config.json`` is there (its rows of these settings).

        Raises:
            ValueError: The ``config.json`` there is of other settings.
        """
        path = out / "config.json"
        if not path.exists():
            return False
        depends = self.depends()
        old = json.loads(path.read_text())
        if {k: old.get(k) for k in depends} != depends:
            raise ValueError(f"{path} is of other settings: remove it first")
        return True

    def start(self, out: Path, extra: Mapping[str, Any]) -> None:
        """``out/config.json`` (if absent: the first job's record).

        Raises:
            ValueError: The ``config.json`` there is of other settings.
        """
        if self.check(out):
            return
        path = out / "config.json"
        depends = self.depends()
        out.mkdir(parents=True, exist_ok=True)
        write_json(
            path,
            depends
            | {
                "targets": self.targets.meta,
                "determinism": self.determinism,
                "provenance": {
                    k: v for k, v in of_diagnosis({}).items() if k != "settings"
                },
            }
            | dict(extra),
        )


def evaluate_targets(
    cache: Path | None = None,
    *,
    conf: RunConf | None = None,
    fconf: FillConf = FillConf(),
    root: Path | None = None,
    wide: Path | None = None,
    features: Features | None = None,
    inception: Path | None = None,
    spacing: Mapping[str, Any] | None = None,
    span: float = SPAN,
) -> Path:
    """The targets' cache (module docstring) of ``conf`` (default
    ``v16_config`` on CUDA) under ``cache`` (default ``$MUJOU_EXP_DATA``),
    E2's too with the wide data ``wide``; parts there are kept; ``span``
    the fit set's (tests). Returns its directory."""
    conf = _yard_conf(None) if conf is None else conf
    configure_determinism(conf.device)
    root = tours_root(conf.data.tours.data_version) if root is None else root
    record: dict[str, Any] = {}
    if features is None:
        features, record = inception_features(inception, conf.device)
    base = _default_base() if cache is None else cache
    sets = fill_sets(conf, root, fconf, span=span, spacing=spacing)
    path = cache_path(base, targets_key(sets, fconf))
    have = Targets.load(path) if (path / "targets.pt").is_file() else None
    if have and record and have.meta["inception"].get("sha256") != record["sha256"]:
        raise ValueError(f"the targets at {path} are of another Inception")
    got = build_targets(
        sets,
        features,
        fconf,
        wide=None if wide is None else WideFrames(wide),
        have=have,
        record=record,
    )
    got.save(path)
    return path


def _run_settings(run_dir: Path) -> tuple[dict[str, Any], bool, int, int]:
    settings = json.loads((run_dir / "jepa.json").read_text())
    arm_m = settings.get("sigreg_on", "flat") == "mean"
    return settings, arm_m, settings.get("expand", 1), settings["k"]


def evaluate_run(
    run_dir: Path,
    *,
    steps: Sequence[int],
    parts: Sequence[str] = PARTS,
    fconf: FillConf = FillConf(),
    root: Path | None = None,
    cache: Path | None = None,
    device: str | None = None,
    spacing: Mapping[str, Any] | None = None,
    zpca_dim: int = ZPCA_DIM,
    split: str | None = None,
    sem: Path | None = None,
    span: float = SPAN,
) -> Path:
    """E1, E1e and E2 (``parts``) of the checkpoints of ``steps`` of the
    JEPA run in ``run_dir`` (module docstring); with ``split`` ``"sem"``,
    E1 and E1f of ``SEM_PARTS`` on the split of the targets ``sem``
    (``evaluate_sem_split``, into ``run_dir / OUT_SEM``). Returns the
    directory of the rows.

    Raises:
        ValueError: A part is not of ``PARTS``, or a step has no
            checkpoint.
    """
    if split is not None:
        if sem is None:
            raise ValueError(f"the split {split!r} needs its targets (--sem)")
        return evaluate_sem_split(
            run_dir, sem, steps=steps, parts=parts, fconf=fconf, root=root,
            cache=cache, device=device, spacing=spacing, split=split, span=span,
        )  # fmt: skip
    unknown = [p for p in parts if p not in PARTS]
    if unknown:
        raise ValueError(f"the parts are of {PARTS}, not {unknown}")
    conf = _conf_of(run_dir, device)
    ckpts = run_ckpts(run_dir)
    missing = [s for s in steps if s not in ckpts]
    if missing:
        raise ValueError(f"no checkpoint of steps {missing} (of {list(ckpts)})")
    ctx = _Context(
        conf, run_dir, fconf=fconf, root=root, cache=cache, spacing=spacing, span=span
    )
    settings, arm_m, expand, k_run = _run_settings(run_dir)
    mods = build_run_jepa(conf, settings)
    w_code = RandProjCode(mods)
    sets, targets, dev = ctx.sets, ctx.targets, ctx.dev
    ctx.check(ctx.fill)  # rows there are of these settings, or refused
    ctx.check(ctx.side)
    extra = {"run": run_dir.name, "k": k_run, "expand": expand, "arm_m": arm_m}
    for step in steps:
        prefix = f"{step:06d}"
        stamp = {"run": run_dir.name, "step": step, "t": float(ckpts[step].stem)}
        todo = [
            p
            for p in parts
            if p != "e2" and not (ctx.fill / "rows" / f"{prefix}-{p}.json").exists()
        ]
        side = [
            c
            for c in gen_eval.RUN_CODES
            if not (ctx.side / "rows" / f"{prefix}-{c}.json").exists()
        ]
        side = side if "e2" in parts else []
        if not todo and not side:
            continue
        ckpt.load_eval(ckpts[step], mods.modules())
        if todo:
            ctx.start(ctx.fill, extra)
        if "e1" in todo:
            rows = e1_rows(
                mods, sets, targets, fconf, expand=expand, k_run=k_run, dev=dev
            )
            _write_rows(ctx.fill / "rows" / f"{prefix}-e1.json", rows, stamp)
        if "e1e" in todo:
            rows = e1e_rows(
                lambda m: _context(mods, _fine(m, expand, dev)),
                "h_V",
                sets,
                targets,
                fconf,
                k_run=k_run,
                dev=dev,
            )
            _write_rows(ctx.fill / "rows" / f"{prefix}-e1e.json", rows, stamp)
        if side:
            if not targets.has("B/"):
                raise ValueError(
                    f"the targets {ctx.key} have no E2: make them with --wide"
                )
            ctx.start(ctx.side, extra)
            codes = step_codes(
                mods,
                side,
                conf=conf,
                arm_m=arm_m,
                fit=lambda: sets.fit,
                w_code=w_code,
                zpca_dim=zpca_dim,
            )
            for name, code in codes.items():
                rows = e2_rows(
                    name, _sets_apply(code.mu, sets, dev), sets, targets, fconf
                )
                _write_rows(ctx.side / "rows" / f"{prefix}-{name}.json", rows, stamp)
    return ctx.fill


def sem_split_sets(
    sets: FillSets, zurich: Zurich, meta: Mapping[str, Any], fconf: FillConf
) -> tuple[FillSets, dict[str, Tensor]]:
    """The sets of the split of ``sem_eval``'s targets (their ``meta``) and
    its indices of the grid by set: ``calib`` its calib stacks (no longer
    the calib windows, which overlap the grid), the candidates those of
    sel or final (chance $1/n_{\\mathrm{cand}}$), the hard ones those whose
    neighbours are of their own segment.

    Raises:
        ValueError: The targets are not of these sets.
    """
    for k in ("fit_ks", "grid_starts"):
        if meta["sets"][k] != json.loads(json.dumps(sets.ident[k])):
            raise ValueError(f"sem_eval's targets are of other sets ({k})")
    split = meta["sets_split"]
    idx = {s: torch.tensor(split[s]["idx"]) for s in ("calib", "sel", "final")}
    segment = torch.tensor(meta["segment"])
    scored = torch.isin(sets.cand, torch.cat([idx["sel"], idx["final"]]))
    cand = sets.cand[scored]
    hard_q, among = hard_sets(cand, len(segment), fconf.hard_reach, fconf.hard_stride)
    seg = segment[among]
    own = (seg == seg[:, :1]).all(1) & (seg[:, 0] >= 0)
    ident = dict(sets.ident) | {"split": "sem", "calib_idx": idx["calib"].tolist()}
    out = replace(
        sets,
        calib=zurich.stacks(idx["calib"]),
        cand=cand,
        cand_stacks=sets.cand_stacks[scored],
        hard_q=hard_q[own],
        hard_among=among[own],
        ident=ident,
    )
    return out, idx


def sem_split_targets(
    fill: Targets, sem: Targets, calib: Tensor
) -> tuple[Targets, dict[str, tuple[str, ...]]]:
    """The targets of E1 on the sem split: ``color`` of ``fill`` and, of the
    quarters, ``quad`` (``sem``'s ``quad/<q>``), their ``calib`` the grid's
    at ``calib``; and the kinds of each mask (``incep`` of 32 px left out:
    not even the whole view's tokens read it)."""
    values: dict[str, dict[str, Tensor]] = {}
    for m in E1_MASKS:
        u = fill.values[f"U/{m}/color"]
        values[f"U/{m}/color"] = {"fit": u["fit"], "calib": u["grid"][calib]} | {
            "grid": u["grid"]
        }
        values[f"I/{m}/color"] = fill.values[f"I/{m}/color"]
    for q in QUARTER_NAMES:
        v = sem.values[f"quad/{q}"]
        values[f"U/Q_{q}/quad"] = {
            "fit": v["fit"],
            "calib": v["grid"][calib],
            "grid": v["grid"],
        }
    kinds = {
        m: ("color", "quad") if m.startswith("Q_") else ("color",) for m in E1_MASKS
    }
    return Targets({}, values, dict(fill.meta)), kinds


def evaluate_sem_split(
    run_dir: Path,
    sem: Path,
    *,
    steps: Sequence[int],
    parts: Sequence[str] = SEM_PARTS,
    fconf: FillConf = FillConf(),
    root: Path | None = None,
    cache: Path | None = None,
    device: str | None = None,
    spacing: Mapping[str, Any] | None = None,
    split: str = "sem",
    span: float = SPAN,
) -> Path:
    """E1 (its targets ``color`` and ``quad``, with the K3 stages and
    ``n_cand``) and E1f (``e1f_rows``) of ``parts`` of the checkpoints of
    ``steps`` of the JEPA run in ``run_dir``, on the split of ``sem_eval``'s
    targets at ``sem``, into ``run_dir / OUT_SEM`` (its ``config.json`` of
    the split's candidates). Returns ``run_dir / OUT_SEM``.

    Raises:
        ValueError: ``split`` is not ``"sem"``, a part is not of
            ``SEM_PARTS``, a step has no checkpoint, or the targets are not
            of these sets.
    """
    if split != "sem":
        raise ValueError(f"the split is 'sem', not {split!r}")
    unknown = [p for p in parts if p not in SEM_PARTS]
    if unknown:
        raise ValueError(
            f"the parts of the sem split are of {SEM_PARTS}, not {unknown}"
        )
    conf = _conf_of(run_dir, device)
    ckpts = run_ckpts(run_dir)
    missing = [s for s in steps if s not in ckpts]
    if missing:
        raise ValueError(f"no checkpoint of steps {missing} (of {list(ckpts)})")
    ctx = _Context(
        conf, run_dir, fconf=fconf, root=root, cache=cache, spacing=spacing, span=span
    )
    sem_t = Targets.load(sem)
    ctx.extra = {"split": "sem", "sem": sem.name}
    zurich = Zurich(conf, ctx.root, fconf.grid_step)
    sets, idx = sem_split_sets(ctx.sets, zurich, sem_t.meta, fconf)
    ctx.sets = sets  # depends: the split's candidates
    out = run_dir / OUT_SEM
    targets, kinds = sem_split_targets(ctx.targets, sem_t, idx["calib"])
    stacks = {"fit": sets.fit, "calib": sets.calib}
    stacks |= {s: zurich.stacks(idx[s]) for s in ("sel", "final")}
    settings, _, expand, k_run = _run_settings(run_dir)
    mods = build_run_jepa(conf, settings)
    ctx.check(out)
    extra = {"run": run_dir.name, "k": k_run, "expand": expand}
    for step in steps:
        prefix = f"{step:06d}"
        todo = [p for p in parts if not (out / "rows" / f"{prefix}-{p}.json").exists()]
        if not todo:
            continue
        stamp = {"run": run_dir.name, "step": step, "t": float(ckpts[step].stem)}
        ckpt.load_eval(ckpts[step], mods.modules())
        ctx.start(out, extra)
        if "e1" in todo:
            rows = e1_rows(
                mods, sets, targets, fconf, expand=expand, k_run=k_run,
                dev=ctx.dev, kinds=kinds, k3=True,
            )  # fmt: skip
            rows = [r | {"n_cand": len(sets.cand)} for r in rows]
            _write_rows(out / "rows" / f"{prefix}-e1.json", rows, stamp)
        if "e1f" in todo:
            rows = e1f_rows(
                mods, stacks, sem_t, expand=expand, k_run=k_run, dev=ctx.dev
            )
            _write_rows(out / "rows" / f"{prefix}-e1f.json", rows, stamp)
    return out


def evaluate_diag(
    run_dir: Path,
    *,
    steps: Sequence[int],
    fconf: FillConf = FillConf(),
    root: Path | None = None,
    cache: Path | None = None,
    device: str | None = None,
    spacing: Mapping[str, Any] | None = None,
    span: float = SPAN,
) -> Path:
    """The diagnosis of ``shuf`` (``e1diag_rows``) of the checkpoints of
    ``steps`` of the JEPA run in ``run_dir``, on E1's sets and targets as
    they are: ``rows/<step>-e1diag.json`` and ``e1diag-<step>.png``.
    Returns ``run_dir / OUT``.

    Raises:
        ValueError: A step has no checkpoint.
    """
    conf = _conf_of(run_dir, device)
    ckpts = run_ckpts(run_dir)
    missing = [s for s in steps if s not in ckpts]
    if missing:
        raise ValueError(f"no checkpoint of steps {missing} (of {list(ckpts)})")
    ctx = _Context(
        conf, run_dir, fconf=fconf, root=root, cache=cache, spacing=spacing, span=span
    )
    settings, arm_m, expand, k_run = _run_settings(run_dir)
    mods = build_run_jepa(conf, settings)
    ctx.check(ctx.fill)
    for step in steps:
        prefix = f"{step:06d}"
        path = ctx.fill / "rows" / f"{prefix}-e1diag.json"
        if path.exists():
            continue
        ckpt.load_eval(ckpts[step], mods.modules())
        ctx.start(
            ctx.fill,
            {"run": run_dir.name, "k": k_run, "expand": expand, "arm_m": arm_m},
        )
        rows = e1diag_rows(
            mods, ctx.sets, ctx.targets, fconf, expand=expand, dev=ctx.dev
        )
        stamp = {"run": run_dir.name, "step": step, "t": float(ckpts[step].stem)}
        _write_rows(path, rows, stamp)
        _diag_picture(ctx.fill / f"e1diag-{prefix}.png", rows)
    return ctx.fill


def evaluate_gen(
    run_dir: Path,
    *,
    steps: Sequence[int],
    fconf: FillConf = FillConf(),
    root: Path | None = None,
    cache: Path | None = None,
    device: str | None = None,
    features: Features | None = None,
    inception: Path | None = None,
    spacing: Mapping[str, Any] | None = None,
    zpca_dim: int = ZPCA_DIM,
) -> Path:
    """E1g (module docstring) of the checkpoints of ``steps`` of the JEPA
    run in ``run_dir``, by G's decoders of its ``z``
    (``posthoc_gen/ckpt/dec-<step>-z-d0.pt``) and ``features`` (default
    FID's Inception of ``inception``, the targets'). Returns ``run_dir /
    OUT``.

    Raises:
        ValueError: A step has no checkpoint, G's candidates are not these,
            or the Inception is not the targets'.
        FileNotFoundError: A step has no decoder.
    """
    conf = _conf_of(run_dir, device)
    ckpts = run_ckpts(run_dir)
    missing = [s for s in steps if s not in ckpts]
    if missing:
        raise ValueError(f"no checkpoint of steps {missing} (of {list(ckpts)})")
    ctx = _Context(conf, run_dir, fconf=fconf, root=root, cache=cache, spacing=spacing)
    ctx.check(ctx.fill)
    if features is None:
        features, record = inception_features(inception, conf.device)
        # by the sha256 alone: the weights' path differs by host
        if record["sha256"] != ctx.targets.meta["inception"].get("sha256"):
            raise ValueError("the Inception is not the targets'")
    gen_dir = run_dir / gen_eval.OUT
    g = json.loads((gen_dir / "config.json").read_text())
    if g["cand"] != ctx.sets.cand.tolist():
        raise ValueError("G's candidates are not these")
    known = {k: v for k, v in g["gen"].items() if k in GenConf.__dataclass_fields__}
    gconf = GenConf(**known)
    settings, arm_m, expand, _ = _run_settings(run_dir)
    mods = build_run_jepa(conf, settings)
    for step in steps:
        prefix = f"{step:06d}"
        path = ctx.fill / "rows" / f"{prefix}-e1g.json"
        if path.exists():
            continue
        dec_path = gen_dir / "ckpt" / f"dec-{prefix}-z-d0.pt"
        if not dec_path.is_file():
            raise FileNotFoundError(f"no decoder {dec_path}: run gen_eval first")
        ckpt.load_eval(ckpts[step], mods.modules())
        to_code: Callable[[Tensor], Tensor] = lambda z: z  # noqa: E731
        if arm_m:  # G's zpca (gen_eval.step_codes): the same PCA of the same z
            gen = torch.Generator().manual_seed(conf.seed + PCA_SEED)
            mean, comps = pca.fit_pca(
                _apply(mods.mu, ctx.sets.fit, ctx.dev), zpca_dim, gen=gen
            )
            to_code = lambda z: pca.encode(z, mean, comps)  # noqa: E731
        ctx.start(ctx.fill, {"run": run_dir.name, "expand": expand, "arm_m": arm_m})
        rows, images = e1g_rows(
            mods,
            ctx.sets,
            ctx.targets,
            Decoder.load(dec_path, ctx.dev),
            features,
            fconf,
            gconf,
            expand=expand,
            to_code=to_code,
            dev=ctx.dev,
        )
        stamp = {"run": run_dir.name, "step": step, "t": float(ckpts[step].stem)}
        _write_rows(path, rows, stamp | {"decoder": dec_path.name})
        for m in PICTURE_MASKS:
            _e1g_picture(ctx.fill / f"g-{prefix}-{m}.png", ctx.sets, images, m, fconf)
    return ctx.fill


def evaluate_yardsticks(
    out_dir: Path,
    *,
    parts: Sequence[str] = ("e1e", "e2"),
    codes: Sequence[str] = YARD_E2,
    conf: RunConf | None = None,
    v16: Path | None = None,
    v17: Path | None = None,
    m_run: Path | None = None,
    fconf: FillConf = FillConf(),
    root: Path | None = None,
    cache: Path | None = None,
    spacing: Mapping[str, Any] | None = None,
    pca_components: int = 1024,
    zpca_dim: int = ZPCA_DIM,
) -> Path:
    """The yardsticks of E1e (``color_V``, ``pca_V``, ``ext``; module
    docstring) and of E2 (``codes`` of ``YARD_E2``) of ``conf`` (default
    ``v16_config`` on CUDA) into ``out_dir``. Returns ``out_dir / OUT``.

    Raises:
        ValueError: A code is not of ``YARD_E2``, or its checkpoint or run
            is not given.
    """
    unknown = [c for c in codes if c not in YARD_E2]
    if unknown:
        raise ValueError(f"E2's yardsticks are of {YARD_E2}, not {unknown}")
    given = {"v16": v16, "v17": v17, "init_m": m_run}
    lacking = [c for c in codes if c in given and given[c] is None]
    if "e2" in parts and lacking:
        raise ValueError(f"the yardsticks {lacking} need their checkpoint or run")
    conf = _yard_conf(None) if conf is None else conf
    ctx = _Context(conf, out_dir, fconf=fconf, root=root, cache=cache, spacing=spacing)
    ctx.check(ctx.fill)
    ctx.check(ctx.side)
    sets, targets, dev = ctx.sets, ctx.targets, ctx.dev
    stamp: dict[str, Any] = {}
    if "e1e" in parts:
        path = ctx.fill / "rows" / f"{YARD_PREFIX}-e1e.json"
        if not path.exists():
            ctx.start(ctx.fill, {"yardsticks": list(YARD_E1E)})
            rows = []
            size = sets.grid_last.shape[-1]
            for name in YARD_E1E:
                rows += e1e_rows(
                    None
                    if name == "ext"
                    else lambda m, name=name: _yard_e1e_inputs(name, m, size, dev),
                    name,
                    sets,
                    targets,
                    fconf,
                    k_run=None,
                    dev=dev,
                )
            _write_rows(path, rows, stamp)
    if "e2" in parts:
        ctx.start(ctx.side, {"yardsticks": list(codes)})
        for name in codes:
            path = ctx.side / "rows" / f"{YARD_PREFIX}-{name}.json"
            if path.exists():
                continue
            x = _yard_e2_inputs(name, conf, sets, dev, given, pca_components, zpca_dim)
            _write_rows(path, e2_rows(name, x, sets, targets, fconf), stamp)
    return ctx.fill


def _yard_e1e_inputs(
    name: str, mask: str, size: int, dev: torch.device
) -> Callable[[Tensor], Tensor]:
    """The inputs of E1e's yardstick ``name`` under ``mask`` of frames of
    ``size``: ``color_V`` the visible coarse cells' mean colors, ``pca_V``
    the pixels with the hidden ones 0 (PCA'd by the readout)."""
    hidden = EVAL_MASKS[mask]
    if name == "color_V":
        color = ColorCode(size // COARSE)
        visible = (~hidden).nonzero().flatten().to(dev)
        return lambda x: color.tokens(x)[:, visible].flatten(1)
    keep = (~pixel_mask(hidden, size)).to(dev)
    return lambda x: pixels(x * keep)


def _yard_e2_inputs(
    name: str,
    conf: RunConf,
    sets: FillSets,
    dev: torch.device,
    given: Mapping[str, Path | None],
    pca_components: int,
    zpca_dim: int,
) -> dict[str, Tensor] | None:
    """The inputs of E2's yardstick ``name`` (None for ``ext``, ``interp``
    and ``mean``: ``e2_rows``)."""
    if name in ("ext", "interp", "mean"):
        return None
    gen = torch.Generator().manual_seed(conf.seed + PCA_SEED)
    code: Code
    if name == "color":
        code = ColorCode(conf.model.patch)
    elif name == "pca":
        code = PCACode.fit(
            pixels, sets.fit, pca_components, gen=gen, device=conf.device
        )
    elif name in ("v16", "v17"):
        path = given[name]
        assert path is not None
        code = vae_at(path, conf.device)
    else:  # init_m
        m_run = given["init_m"]
        assert m_run is not None
        m_conf = _conf_of(m_run, conf.device)
        mods = build_run_jepa(m_conf, json.loads((m_run / "jepa.json").read_text()))
        m_gen = torch.Generator().manual_seed(m_conf.seed + PCA_SEED)
        code = PCACode.fit(mods.mu, sets.fit, zpca_dim, gen=m_gen, device=conf.device)
    return _sets_apply(code.mu, sets, dev)


# -- the command line ------------------------------------------------------------------


def _ints(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split(","))


def _names(text: str) -> tuple[str, ...]:
    return tuple(text.split(","))


def main(argv: Sequence[str] | None = None) -> Path | list[str]:
    """``python -m exp.vision.fill_eval`` (module docstring). Returns the
    output (``videos``: the videos E2's wide data must hold)."""
    ap = argparse.ArgumentParser(
        prog="python -m exp.vision.fill_eval",
        description="V18c's E1, E1e, E1g and E2: the features of the unobserved parts.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    tg = sub.add_parser("targets", help="the targets' cache (once)")
    tg.add_argument("--wide", type=Path, help="the wide data (E2)")
    vid = sub.add_parser("videos", help="the videos E2's wide data must hold")
    run = sub.add_parser("run", help="E1, E1e and E2 of a JEPA run")
    run.add_argument("run_dir", type=Path)
    run.add_argument("--steps", type=_ints, required=True)
    run.add_argument("--parts", type=_names, default=None)
    run.add_argument("--split", choices=("sem",), help="sem_eval's split (E1, E1f)")
    run.add_argument("--sem", type=Path, help="sem_eval's targets (--split sem)")
    run.add_argument(
        "--target", choices=("scene",), help="E1f of scene_eval's labels (e1f only)"
    )
    diag = sub.add_parser("diag", help="E1's diagnosis of shuf (old checkpoints)")
    diag.add_argument("run_dir", type=Path)
    diag.add_argument("--steps", type=_ints, required=True)
    gen = sub.add_parser("gen", help="E1g of a JEPA run (after gen_eval)")
    gen.add_argument("run_dir", type=Path)
    gen.add_argument("--steps", type=_ints, required=True)
    yard = sub.add_parser("yardsticks", help="E1e's and E2's yardsticks")
    yard.add_argument("--out", type=Path, required=True)
    yard.add_argument("--parts", type=_names, default=("e1e", "e2"))
    yard.add_argument("--codes", type=_names, default=YARD_E2)
    yard.add_argument("--v16", type=Path, help="a VAE checkpoint")
    yard.add_argument("--v17", type=Path, help="a VAE checkpoint")
    yard.add_argument("--m-run", type=Path, help="arm M's run (init_m)")
    for p in (tg, vid, run, gen, yard, diag):
        p.add_argument("--root", type=Path, help="the data (default: tours_root)")
        p.add_argument("--cache", type=Path, help="(default: $MUJOU_EXP_DATA)")
        p.add_argument("--device", help="(default: the run's, or cuda)")
    for p in (tg, gen):
        p.add_argument("--inception", type=Path, help="FID's weights")
    args = ap.parse_args(argv)
    if args.cmd == "run" and args.target == "scene":
        if args.parts not in (None, ("e1f",)) or args.split is not None:
            ap.error("--target scene is E1f alone, without --split")
        # scene_eval imports this module: imported here, not at the top
        from exp.vision import scene_eval

        return scene_eval.evaluate_fill(
            args.run_dir,
            steps=args.steps,
            root=args.root,
            cache=args.cache,
            device=args.device,
        )
    if args.cmd == "run":
        default = PARTS if args.split is None else SEM_PARTS
        return evaluate_run(
            args.run_dir,
            steps=args.steps,
            parts=default if args.parts is None else args.parts,
            root=args.root,
            cache=args.cache,
            device=args.device,
            split=args.split,
            sem=args.sem,
        )
    if args.cmd == "diag":
        return evaluate_diag(
            args.run_dir,
            steps=args.steps,
            root=args.root,
            cache=args.cache,
            device=args.device,
        )
    if args.cmd == "gen":
        return evaluate_gen(
            args.run_dir,
            steps=args.steps,
            root=args.root,
            cache=args.cache,
            device=args.device,
            inception=args.inception,
        )
    conf = _yard_conf(args.device)
    if args.cmd == "targets":
        return evaluate_targets(
            args.cache,
            conf=conf,
            root=args.root,
            wide=args.wide,
            inception=args.inception,
        )
    if args.cmd == "videos":
        root = (
            tours_root(conf.data.tours.data_version) if args.root is None else args.root
        )
        sets = fill_sets(conf, root, FillConf())
        assert sets.places is not None
        return sorted({v for s in SETS for v, _ in sets.places[s]})
    return evaluate_yardsticks(
        args.out,
        parts=args.parts,
        codes=args.codes,
        conf=conf,
        v16=args.v16,
        v17=args.v17,
        m_run=args.m_run,
        root=args.root,
        cache=args.cache,
    )


if __name__ == "__main__":
    print(main())
