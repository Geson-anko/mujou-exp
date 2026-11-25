r"""V18c's main measure $m$ of the coarse scene categories (``plans/jepa.md``,
「V18c: 主の指標（場面のカテゴリ、2026-11-25）」; README 「V18c: 場面のカテゴリで
主の指標を作る」): how well a linear readout of a code tells the coarse kind
of scene of a stack within a video, across the city walks.

The grid (``scene_grid``): of each video of ``SceneConf.videos`` (the 9 city
walks, Wildlife left out), the stacks starting every ``stride`` frames (2 s)
of the prepared 64 px data; a stack's place is its last frame.

The labels (``build_labels``, once into a cache ``v18c-scene-<key>``): each
place's 320 px frame (``exp.lib.tours.hires``, its place checked first by
``sem_eval.check_places``) through Places365's ResNet-50 (``exp.lib.
places365``), its 365 probabilities summed into the 16 groups of level 2;
the label is the largest group. Of the kinds ``full`` (the whole frame: the
main target), ``quad/<q>`` (each 160 px quarter: E1f's), ``up64`` (the 64
px last frame, recorded) and ``nuis/<j>`` (the frame dimmed, its white
balance off, blurred by $\sigma$ = 5 px: recorded); and ``places64``, the
network's features before its last layer of the 64 px last frame (a
yardstick). Of ``full`` and each quarter, the groups kept (``kept_groups``:
a share of ``keep_share`` or more over all, and of ``keep_video_share`` or
more in ``keep_videos`` videos or more), the block length $L$
(``block_length``, of ``full``) and the support (``support``: each kept
group in ``min_blocks`` blocks or more); V-c1 and V-E1'5 are of these.

A code's row (``score``): its codes at the kept stacks, ``readout.
nested_cv`` by video to the kept groups (weighted by class), scored by
``readout.scene_m`` ($m = \mathrm{BA} - B$), with the listed scores (BA,
$B$, by video, Zurich alone, the other 8, the classes' AUC); its
predictions ``pred`` over the whole grid (-1 at the stacks left out) and,
for the codes the judgement needs, the jackknife's ``jack_pred`` ``(9,
N)``. ``mean`` predicts group 0 everywhere ($m$ exactly 0).

Into ``<out>/`` (``frames.json``, ``audit/``, ``yardsticks/rows``,
``checks/rows``) and each run's ``posthoc_scene/`` and
``posthoc_scene_fill/`` (``config.json``, ``rows/<step>-<code>.json``); a job
whose file is there is skipped, after the ``config.json`` there is checked::

    python -m exp.vision.scene_eval frames --out runs/v18c-scene/frames.json
    python -m exp.vision.scene_eval labels --hires <dir> --weights <places365>
    python -m exp.vision.scene_eval audit-sheets --out runs/v18c-scene/audit
    python -m exp.vision.scene_eval yardsticks --out runs/v18c-scene/yardsticks \
        --v16 <ckpt> --v17 <ckpt> --init-runs <run s0>,<run s1>
    python -m exp.vision.scene_eval checks --out runs/v18c-scene/checks
    python -m exp.vision.scene_eval run <run_dir> --steps 0,2000,6000,...,60000
    python -m exp.vision.scene_eval fill <run_dir> --steps 0,6000,60000

(``fill`` is also ``fill_eval run <run_dir> --parts e1f --target scene``.)
"""

import argparse
import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor
from torch.nn import functional as F

from exp.lib import ckpt, places365
from exp.lib.metrics import contiguous_blocks
from exp.lib.png import tile, write_png
from exp.lib.provenance import configure_determinism, of_diagnosis
from exp.lib.readout import (
    CONST,
    SCENE_SCALES,
    balanced_accuracy,
    class_auc,
    nested_cv,
    nested_cv_jackknife,
    scene_m,
    video_chance,
)
from exp.lib.regions import COARSE, EVAL_MASKS
from exp.lib.rundir import write_json
from exp.lib.tours.hires import HiresFrames
from exp.lib.tours.stacks import DATA_ENV, tours_root
from exp.lib.video_io import U8Reader
from exp.vision.compose import from_dict
from exp.vision.eval import PCA_SEED
from exp.vision.fill_eval import (
    E1F_K,
    QUARTER_NAMES,
    _predicted,
    _run_settings,
    e1f_inputs,
    e1f_masks,
)
from exp.vision.gen_eval import Features, inception_features
from exp.vision.jepa import JEPAModules, expand_masks
from exp.vision.jepa_diag import grid_starts, run_ckpts
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
from exp.vision.sem_eval import (
    BRIGHT,
    K3Code,
    SemConf,
    blur,
    check_places,
    quarters,
    sem_sets,
    white_balance,
)

__all__ = [
    "AUDIT_N",
    "AUDIT_N_QUAD",
    "AUDIT_SEED",
    "BOOT",
    "BOOT_SEED",
    "CACHE_PREFIX",
    "DELTA_STAR",
    "JACK_STEPS",
    "N_RAND",
    "OUT",
    "OUT_FILL",
    "RAND_SEED",
    "RUN_CODES",
    "SIDE_STEPS",
    "SYN_DIM",
    "SYN_SEED",
    "VIDEOS",
    "YARDSTICKS",
    "Grid",
    "SceneConf",
    "SceneLabels",
    "audit_sample",
    "block_length",
    "build_labels",
    "e1f_scene_rows",
    "evaluate_audit",
    "evaluate_checks",
    "evaluate_fill",
    "evaluate_frames",
    "evaluate_labels",
    "evaluate_run",
    "evaluate_yardsticks",
    "kept_groups",
    "main",
    "random_labels",
    "scene_grid",
    "score",
    "support",
    "synthetic_codes",
]

OUT = "posthoc_scene"
OUT_FILL = "posthoc_scene_fill"
CACHE_PREFIX = "v18c-scene-"
VIDEOS = (
    "Amsterdam",
    "Bangkok",
    "Chiang_Mai",
    "Istanbul",
    "Kuala_Lampur",
    "Singapore",
    "Stockholm",
    "Venice",
    "Zurich",
)
NUIS = ("bright", "white", "blur")  # sem_eval's nuisances
BLUR_PX = 5.0  # the blur's sigma at the hires size (1 px at 64 px)
BOOT, BOOT_SEED = 2000, 101  # the bootstrap's (judge_scene), recorded here
AUDIT_SEED, AUDIT_N, AUDIT_N_QUAD = 109, 30, 20
RAND_SEED, N_RAND = 113, 20  # V-c4
SYN_SEED, SYN_DIM = 127, 1024  # V-c5
DELTA_STAR = 0.03  # the least difference to tell (README)
SYN_Q = (0.0, 0.25, None, 0.5)  # V-c5's q; None: 0.25 + dq
RUN_CODES = ("z", "z_k3", "h_mean")
SIDE_STEPS = (0, 6000, 60000)  # the steps of z_k3, h_mean and the jackknife
JACK_STEPS = SIDE_STEPS
YARDSTICKS = (
    "mean",
    "label",
    "video",
    "color",
    "color8",
    "pca16",
    "pca64",
    "pca256",
    "pca1024",
    "v16",
    "v17",
    "init_c",
    "inc64",
    "places64",
)
JACK_YARDS = ("init_c", "color", "color8", "inc64")
YARD_PREFIX = "yardstick"
E1F_STAGES = ("pred", "pool", "pred_k3", "vis_color", "oracle", "mean")
E1F_JACK = {("pred", 8), ("pool", 8), ("vis_color", 8), ("oracle", 4)}


@dataclass(frozen=True)
class SceneConf:
    """The settings (module docstring; ``plans/jepa.md``), by default the
    plan's."""

    videos: tuple[str, ...] = VIDEOS
    stride: int = 20  # frames between the grid's stacks (2 s)
    keep_share: float = 0.05
    keep_videos: int = 5
    keep_video_share: float = 0.01
    min_blocks: int = 40
    l_min: int = 16
    max_group: float = 0.7  # V-c1
    max_dropped: float = 0.3  # V-c1
    batch: int = 64  # stacks or frames at once


# -- the grid ------------------------------------------------------------------------


@dataclass(frozen=True)
class Grid:
    """The grid's stacks: each one's video ``video`` ``(N,)`` (its index in
    ``videos``), first frame ``starts`` ``(N,)`` and place (the video and
    its last frame), the videos in order, then the starts."""

    videos: tuple[str, ...]
    video: Tensor
    starts: Tensor
    places: list[tuple[str, int]]
    frames: int  # n, the stack
    stride: int  # video frames between a stack's frames

    @property
    def span(self) -> int:
        return (self.frames - 1) * self.stride

    def rows(self, v: int) -> Tensor:
        """The grid's rows of video ``v``."""
        return (self.video == v).nonzero().flatten()


def scene_grid(conf: RunConf, root: Path, sconf: SceneConf) -> Grid:
    """The grid (module docstring) of ``conf``'s data at ``root``.

    Raises:
        ValueError: A video of ``sconf.videos`` is not there.
    """
    t = conf.data.tours
    missing = [v for v in sconf.videos if not (root / v).is_dir()]
    if missing:
        raise ValueError(f"no videos {missing} under {root}")
    span = (t.frames - 1) * t.stride
    video, starts, places = [], [], []
    for i, v in enumerate(sconf.videos):
        s = grid_starts(len(U8Reader(root / v)), t.frames, t.stride, sconf.stride)
        video.append(torch.full((len(s),), i, dtype=torch.long))
        starts.append(s)
        places += [(v, int(f) + span) for f in s]
    return Grid(
        sconf.videos, torch.cat(video), torch.cat(starts), places, t.frames, t.stride
    )


@torch.no_grad()
def grid_apply(
    fn: Callable[[Tensor], Tensor],
    grid: Grid,
    root: Path,
    dev: torch.device,
    batch: int,
    *,
    last: bool = False,
) -> Tensor:
    """``fn`` of the grid's stacks u8 ``(b, n, 3, S, S)`` (or, with
    ``last``, of their last frames ``(b, 3, S, S)``), ``batch`` at a time on
    ``dev``, float32 on the CPU ``(N, ·)``; a video's frames are read once."""
    offsets = torch.arange(grid.frames) * grid.stride
    out = []
    for v, name in enumerate(grid.videos):
        reader = U8Reader(root / name)
        frames = reader[0 : len(reader)]
        starts = grid.starts[grid.rows(v)]
        for s in starts.split(batch):
            x = frames[s + grid.span] if last else frames[s[:, None] + offsets]
            out.append(fn(x.to(dev)).float().cpu())
    return torch.cat(out)


# -- the labels: groups kept, block length, support -----------------------------------


def kept_groups(
    y: Tensor, video: Tensor, groups: int, sconf: SceneConf
) -> tuple[list[int], dict[str, Any]]:
    """The groups kept (module docstring) of the labels ``y`` ``(N,)`` in
    the videos ``video`` ``(N,)`` (of ``len(sconf.videos)``), and the
    record: each group's share, its videos of ``keep_video_share`` or more,
    the counts ``(groups, videos)`` and each group's largest video share."""
    nv = len(sconf.videos)
    counts = torch.zeros(groups, nv, dtype=torch.long)
    counts.index_put_((y, video), torch.ones_like(y), accumulate=True)
    share = counts.sum(1).double() / len(y)
    per_video = counts.double() / counts.sum(0).clamp(min=1)
    videos = (per_video >= sconf.keep_video_share).sum(1)
    keep = ((share >= sconf.keep_share) & (videos >= sconf.keep_videos)).nonzero()
    kept = keep.flatten().tolist()
    most = counts.max(1).values.double() / counts.sum(1).clamp(min=1)
    return kept, {
        "share": share.tolist(),
        "videos_over": videos.tolist(),
        "counts": counts.tolist(),
        "top_video_share": most.tolist(),
    }


def block_length(t: Tensor, video: Tensor, k: int, l_min: int) -> dict[str, Any]:
    r"""The integrated time $\tau$ of the kept labels ``t`` ``(N,)`` (-1 left
    out; the grid in order within each video ``video``) and $L = \min(\max(
    \lceil 2 \tau \rceil, l_{\min}), L_{\max})$, $L_{\max} = \lfloor
    n_{\min} / 8 \rfloor$ ($n_{\min}$ the kept stacks of the video with the
    fewest): with $x_t$ the one-hot of $t$ less its video's mean (0 at a
    stack left out), $\rho(\ell) = \sum_v \sum_t \langle x_t, x_{t + \ell}
    \rangle / \sum_v \sum_t \lVert x_t \rVert^2$ over pairs within a video,
    and $\tau = 1 + 2 \sum_{\ell \le \ell^*} \rho(\ell)$, $\ell^*$ the lag
    before $\rho$ is first at most 0."""
    on = t >= 0
    x = torch.zeros(len(t), k, dtype=torch.float64)
    x[on] = F.one_hot(t[on], k).double()
    vids = torch.unique(video).tolist()
    parts = []
    for v in vids:
        r = (video == v).nonzero().flatten()
        xv, ov = x[r], on[r]
        xv[ov] -= xv[ov].mean(0)
        parts.append(xv)
    den = sum(float(p.square().sum()) for p in parts)
    tau = 1.0
    longest = max(len(p) for p in parts)
    for lag in range(1, longest):
        num = sum(float((p[:-lag] * p[lag:]).sum()) for p in parts if len(p) > lag)
        rho = num / den
        if rho <= 0:
            break
        tau += 2 * rho
    n_min = min(int(on[video == v].sum()) for v in vids)
    l_max = n_min // 8
    length = min(max(math.ceil(2 * tau), l_min), l_max)
    return {"tau": tau, "L": length, "L_max": l_max, "L_at_max": length == l_max}


def support(t: Tensor, blocks: Tensor, k: int) -> list[int]:
    """The number of blocks ``(N,)`` each kept class of ``t`` ``(N,)`` (-1
    left out) appears in."""
    on = t >= 0
    return [int(torch.unique(blocks[on & (t == c)]).numel()) for c in range(k)]


def _analysis(
    y: Tensor, video: Tensor, groups: int, sconf: SceneConf, length: int | None
) -> dict[str, Any]:
    """Of the labels ``y``: the groups kept, the kept labels ``t`` (-1 left
    out), $L$ (of these labels when ``length`` is None) and the support,
    and the checks of V-c1 (or V-E1'5)."""
    kept, record = kept_groups(y, video, groups, sconf)
    t = torch.full_like(y, -1)
    for j, g in enumerate(kept):
        t[y == g] = j
    k = len(kept)
    out: dict[str, Any] = {"kept": kept, "K": k, **record}
    if k == 0:
        return out | {"t": t.tolist(), "passed": False}
    blk = block_length(t, video, k, sconf.l_min)
    out |= blk if length is None else {}
    ell: int = blk["L"] if length is None else length
    blocks = contiguous_blocks(video, ell)
    sup = support(t, blocks, k)
    shares = torch.tensor(record["share"])
    dropped = float((t < 0).double().mean())
    biggest = float(shares[kept].max())  # of all the stacks
    checks = {
        "K_ge_3": k >= 3,
        "max_group": biggest,
        "max_group_ok": biggest <= sconf.max_group,
        "dropped": dropped,
        "dropped_ok": dropped <= sconf.max_dropped,
        "support": sup,
        "support_ok": min(sup) >= sconf.min_blocks,
    }
    passed = all(
        v for key, v in checks.items() if key.endswith("ok") or key == "K_ge_3"
    )
    return out | {"t": t.tolist(), "checks": checks, "passed": passed}


# -- the labels' cache -------------------------------------------------------------------


def _nuisances(x: Tensor) -> dict[str, Tensor]:
    """The nuisances (``NUIS``) of the hires u8 frames ``x``: sem_eval's."""
    return {
        "bright": (x.double() * BRIGHT).round().to(torch.uint8),
        "white": white_balance(x),
        "blur": blur(x, BLUR_PX),
    }


def labels_key(grid: Grid, sconf: SceneConf, hires_name: str) -> str:
    """The first 12 hex digits of the sha256 of the network's files, the
    hires data's name, the grid's places and the settings: the cache's name
    ``v18c-scene-<key>``."""
    text = json.dumps(
        {
            "places365": [
                places365.RESNET50_SHA256,
                places365.HIERARCHY_SHA256,
                places365.CATEGORIES_SHA256,
            ],
            "hires": hires_name,
            "places": grid.places,
            "sconf": _settings(asdict(sconf)),
            "nuis": {"bright": BRIGHT, "blur_px": BLUR_PX},
        },
        sort_keys=True,
    )
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _settings(d: Mapping[str, Any]) -> dict[str, Any]:
    """The settings the values depend on: all but ``batch``, as JSON has them."""
    return json.loads(json.dumps({k: v for k, v in d.items() if k != "batch"}))


def build_labels(
    grid: Grid,
    hires: HiresFrames,
    net: torch.nn.Module,
    sconf: SceneConf,
    *,
    root: Path,
    held_out: str,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """The groups' probabilities ``(N, 16)`` of each kind, ``places64``
    ``(N, 2048)`` and the meta (module docstring) of the grid's places from
    ``hires`` by ``net`` (Places365's), each batch's places checked first
    against the prepared frames at ``root`` (``sem_eval.check_places``).

    Raises:
        ValueError: A place is not of its stack.
    """
    member = places365.membership()
    kinds = ("full", *(f"quad/{q}" for q in QUARTER_NAMES), "up64")
    kinds += tuple(f"nuis/{j}" for j in NUIS)
    probs: dict[str, list[Tensor]] = {k: [] for k in kinds}
    feats: list[Tensor] = []
    worst, margin = 0.0, math.inf
    for v, name in enumerate(grid.videos):
        frames = U8Reader(root / name)
        prepared = {name: frames}  # a video's places are of it alone
        for rows in grid.rows(v).split(sconf.batch):
            places = [grid.places[i] for i in rows.tolist()]
            last = frames[grid.starts[rows] + grid.span]
            x = hires.frames(places)
            factor = x.shape[-1] // last.shape[-1]
            small = F.avg_pool2d(x.double(), factor)
            got = check_places("grid", places, last, small, prepared)
            worst, margin = max(worst, got["max_diff"]), min(margin, got["b_margin"])
            images = {"full": x, "up64": last}
            images |= {f"quad/{q}": c.contiguous() for q, c in quarters(x).items()}
            images |= {f"nuis/{j}": c for j, c in _nuisances(x).items()}
            for kind in kinds:
                pre = places365.preprocess(images[kind])
                probs[kind].append(
                    places365.group_probs(places365.logits(net, pre), member).float()
                )
            feats.append(places365.penultimate(net, places365.preprocess(last)).cpu())
    out = {k: torch.cat(v).cpu() for k, v in probs.items()}
    out["places64"] = torch.cat(feats)
    meta = _labels_meta(out, grid, sconf, held_out)
    meta |= {
        "places_max_diff": worst,
        "places_b_margin": margin,
        "places_checked": True,
        "groups": places365.group_names(),
        "places365": {
            "resnet50_sha256": places365.RESNET50_SHA256,
            "hierarchy_sha256": places365.HIERARCHY_SHA256,
            "categories_sha256": places365.CATEGORIES_SHA256,
        },
        "hires": {"name": hires.root.name},
    }
    return out, meta


def _labels_meta(
    probs: Mapping[str, Tensor], grid: Grid, sconf: SceneConf, held_out: str
) -> dict[str, Any]:
    """The kept groups, $L$, the support and the checks of ``full`` and each
    quarter (``_analysis``; the quarters at ``full``'s $L$), and the
    record: the labels' changes under each nuisance and 2 s on, and the 64
    px labels' agreement, by group of ``full``."""
    groups = probs["full"].shape[1]
    y = {k: p.argmax(1) for k, p in probs.items()}
    full = _analysis(y["full"], grid.video, groups, sconf, None)
    if "L" not in full:
        full["L"] = None
    meta: dict[str, Any] = {
        "sconf": asdict(sconf),
        "videos": list(grid.videos),
        "held_out": held_out,
        "video": grid.video.tolist(),
        "places": grid.places,
        "full": full,
    }
    for q in QUARTER_NAMES:
        meta[f"quad/{q}"] = _analysis(
            y[f"quad/{q}"], grid.video, groups, sconf, full["L"] or 1
        )
    yf = y["full"]

    def by_group(changed: Tensor, on: Tensor) -> dict[str, float | None]:
        return {
            str(g): (float(changed[on & (yf == g)].double().mean())
                     if bool((on & (yf == g)).any()) else None)
            for g in full["kept"]
        }  # fmt: skip

    everywhere = torch.ones_like(yf, dtype=torch.bool)
    nxt = torch.zeros_like(everywhere)
    nxt[:-1] = grid.video[1:] == grid.video[:-1]
    moved = torch.zeros_like(everywhere)
    moved[:-1] = yf[1:] != yf[:-1]
    meta["record"] = {
        "nuis": {j: by_group(y[f"nuis/{j}"] != yf, everywhere) for j in NUIS},
        "next_2s": by_group(moved, nxt),
        "up64_agree": float((y["up64"] == yf).double().mean()),
        "up64_recall": by_group(y["up64"] == yf, everywhere),
        "quad_agree": {
            q: float((y[f"quad/{q}"] == yf).double().mean()) for q in QUARTER_NAMES
        },
    }
    return meta


class SceneLabels:
    """The labels' cache (module docstring): ``probs`` by kind, ``meta``
    (``labels.json``), the kept labels ``t(kind)`` ``(N,)`` (-1 left out),
    the grid's videos ``video`` ``(N,)`` and its rows of the held-out video."""

    def __init__(self, probs: dict[str, Tensor], meta: dict[str, Any]) -> None:
        self.probs, self.meta = probs, meta
        self.video = torch.tensor(meta["video"])

    def t(self, kind: str = "full") -> Tensor:
        return torch.tensor(self.meta[kind]["t"])

    def k(self, kind: str = "full") -> int:
        return int(self.meta[kind]["K"])

    @property
    def held_out(self) -> int:
        return self.meta["videos"].index(self.meta["held_out"])

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.probs, path / "labels.pt")
        write_json(path / "labels.json", self.meta)

    @classmethod
    def load(cls, path: Path) -> SceneLabels:
        """The cache at ``path``.

        Raises:
            FileNotFoundError: None there (``scene_eval labels`` makes it).
        """
        if not (path / "labels.pt").is_file():
            raise FileNotFoundError(f"no labels at {path}: make them by `labels`")
        probs = torch.load(path / "labels.pt", weights_only=True)
        return cls(probs, json.loads((path / "labels.json").read_text()))


def _default_base() -> Path:
    repo = Path(__file__).resolve().parents[2]
    return Path(os.environ.get(DATA_ENV, repo / ".fleet" / "data"))


def find_labels(base: Path, grid: Grid, sconf: SceneConf) -> Path:
    """The one cache under ``base`` of this grid and settings.

    Raises:
        FileNotFoundError: None.
        ValueError: More than one (name it by --labels).
    """
    found = []
    for d in sorted(base.glob(f"{CACHE_PREFIX}*")):
        path = d / "labels.json"
        if not path.is_file():
            continue
        meta = json.loads(path.read_text())
        same_places = meta["places"] == json.loads(json.dumps(grid.places))
        if same_places and _settings(meta["sconf"]) == _settings(asdict(sconf)):
            found.append(d)
    if not found:
        raise FileNotFoundError(f"no labels under {base}: make them by `labels`")
    if len(found) > 1:
        raise ValueError(f"several labels of this grid: {found}; pass one by --labels")
    return found[0]


# -- the scores ------------------------------------------------------------------------


def _scale_name(c: float) -> float | str:
    return "const" if c == CONST else c


def score(
    x: Tensor | None,
    t: Tensor,
    video: Tensor,
    k: int,
    held_out: int,
    *,
    jack: bool = False,
    fixed: Mapping[int, float] | None = None,
) -> tuple[dict[str, Any], dict[int, float] | None]:
    """A code's row (module docstring) of its codes ``x`` ``(N, D)`` at the
    grid's stacks (None: ``mean``) to the kept labels ``t`` ``(N,)`` (-1
    left out) of ``k`` classes, ``held_out`` the video id of Zurich; and the
    scales chosen by video (None for ``mean``). ``fixed``: the scales given
    (``readout.nested_cv``)."""
    on = t >= 0
    tk, vk = t[on], video[on]
    pred = torch.full_like(t, -1)
    chosen: dict[int, float] | None = None
    row: dict[str, Any] = {}
    if x is None:
        pred[on] = 0
        scores = torch.zeros(int(on.sum()), k, dtype=torch.float64)
        row |= {"dim": 0, "scale_by_video": None, "const_folds": None}
    else:
        res = nested_cv(x[on.to(x.device)], tk, vk, k, fixed=fixed)
        pred[on], scores, chosen = res.pred, res.scores, res.scale_by_video
        row |= {
            "dim": x.shape[1],
            "scale_by_video": {str(v): _scale_name(c) for v, c in chosen.items()},
            "const_folds": sum(c == CONST for c in chosen.values()),
            "edge_folds": sum(c == SCENE_SCALES[0] for c in chosen.values()),
        }
        if jack:
            jp = torch.full((len(chosen), len(t)), -1, dtype=torch.long)
            jp[:, on] = nested_cv_jackknife(x[on.to(x.device)], tk, vk, k, chosen)
            row["jack_pred"] = jp.tolist()
    pk = pred[on]
    vids = torch.unique(vk).tolist()

    def sub(on_sub: Tensor, ba: bool = False) -> float | None:
        """$m$ (or BA) of a subset, its classes of 10 rows or more (None if
        none is)."""
        p, y, v = pk[on_sub], tk[on_sub], vk[on_sub]
        got = (
            balanced_accuracy(p, y, k, min_count=10)
            if ba
            else scene_m(p, y, v, k, min_count=10)
        )
        return None if math.isnan(got) else got

    row |= {
        "m": scene_m(pk, tk, vk, k),
        "ba": balanced_accuracy(pk, tk, k),
        "chance": video_chance(pk, tk, vk, k),
        "m_by_video": {str(v): sub(vk == v) for v in vids},
        "ba_by_video": {str(v): sub(vk == v, ba=True) for v in vids},
        "m_zurich": sub(vk == held_out),
        "m_seen8": sub(vk != held_out),
        "auc": class_auc(scores, tk, vk, k),
        "pred": pred.tolist(),
    }
    return row, chosen


def random_labels(t: Tensor, k: int, draws: int, seed: int) -> list[Tensor]:
    """V-c4's labels: ``draws`` times, each kept stack's label drawn anew
    from the kept labels' shares, independent of the inputs, the videos and
    the times (-1 kept where ``t`` has it)."""
    g = torch.Generator().manual_seed(seed)
    on = t >= 0
    share = torch.bincount(t[on], minlength=k).double()
    out = []
    for _ in range(draws):
        r = torch.full_like(t, -1)
        r[on] = torch.multinomial(share, int(on.sum()), replacement=True, generator=g)
        out.append(r)
    return out


def synthetic_codes(
    t: Tensor, video: Tensor, k: int, *, dim: int = SYN_DIM, seed: int = SYN_SEED
) -> dict[str, tuple[float, Tensor, Tensor]]:
    r"""V-c5's codes by name ``syn_<q>``: $q$, the signal $S_q$ ``(N,)`` and
    the code $[\mathrm{onehot}(S_q), \varepsilon]$ ``(N, dim)``. $U_i$,
    $R_i$ and $\varepsilon_i$ are drawn once from ``seed`` (the same for
    every $q$); $S_{q,i} = t_i$ if $U_i < q$, else $R_i$; $q$ of ``SYN_Q``,
    its third $0.25 + \Delta q$, $\Delta q = \Delta_* / (1 - B(\mathrm{label}))$
    (the population's $m^*(S_q) = q (1 - B(\mathrm{label}))$). -1 stays at
    the stacks left out."""
    g = torch.Generator().manual_seed(seed)
    n = len(t)
    u = torch.rand(n, generator=g, dtype=torch.float64)
    r = torch.randint(k, (n,), generator=g)
    eps = torch.randn(n, dim - k, generator=g)
    on = t >= 0
    b = video_chance(t[on], t[on], video[on], k)
    dq = DELTA_STAR / (1 - b)
    out = {}
    for q in SYN_Q:
        qq = 0.25 + dq if q is None else q
        s = torch.where(u < qq, t, r)
        s[~on] = -1
        code = torch.cat([F.one_hot(s.clamp(min=0), k).float(), eps], 1)
        out[f"syn_{qq:.4f}"] = (qq, s, code)
    return out


# -- the jobs ------------------------------------------------------------------------------


def _conf_of(run_dir: Path, device: str | None) -> RunConf:
    conf = from_dict(yaml.safe_load((run_dir / "config.yaml").read_text()))
    return conf if device is None else replace(conf, device=device)


def _yard_conf(device: str | None) -> RunConf:
    d = v16_config()
    d["device"] = device or "cuda"
    return from_dict(d)


def _max_mem(dev: torch.device) -> float | None:
    if dev.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(dev) / 2**30


class _Context:
    """The grid, the labels and the record of an evaluation of ``conf``."""

    def __init__(
        self,
        conf: RunConf,
        out: Path,
        *,
        sconf: SceneConf,
        root: Path | None,
        labels: Path | None,
        cache: Path | None,
    ) -> None:
        self.conf, self.sconf, self.out = conf, sconf, out
        self.dev = torch.device(conf.device)
        self.determinism = configure_determinism(conf.device)
        self.root = tours_root(conf.data.tours.data_version) if root is None else root
        self.grid = scene_grid(conf, self.root, sconf)
        if labels is None:
            labels = find_labels(
                _default_base() if cache is None else cache, self.grid, sconf
            )
        self.path = labels
        # the pixels' PCA of pca<k>: sem_eval's fit set and size (tests: smaller)
        self.spacing: Mapping[str, Any] | None = None
        self.span, self.pca_max = SPAN, SemConf().pca_max
        self.labels = SceneLabels.load(labels)
        if self.labels.meta["places"] != json.loads(json.dumps(self.grid.places)):
            raise ValueError(f"the labels {labels} are not of this grid")
        if _settings(self.labels.meta["sconf"]) != _settings(asdict(sconf)):
            raise ValueError(f"the labels {labels} are of other settings")

    def depends(self) -> dict[str, Any]:
        """What the rows depend on beside the codes."""
        full = self.labels.meta["full"]
        d = {
            "sconf": _settings(asdict(self.sconf)),
            "labels": self.path.name,
            "K": full["K"],
            "kept": full["kept"],
            "L": full["L"],
            "scales": list(SCENE_SCALES),
            "seeds": {"pixels_pca": PCA_SEED, "rand": RAND_SEED, "syn": SYN_SEED},
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
        provenance = {k: v for k, v in of_diagnosis({}).items() if k != "settings"}
        write_json(
            path,
            depends
            | {"determinism": self.determinism, "provenance": provenance}
            | dict(extra),
        )

    def write(self, name: str, row: Mapping[str, Any], t0: float) -> None:
        """The row ``row`` into ``rows/<name>.json``."""
        out = dict(row) | {"sec": time.time() - t0, "max_mem": _max_mem(self.dev)}
        out["key"] = self.depends()["key"]
        path = self.out / "rows" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, out)

    def score(
        self, x: Tensor | None, *, jack: bool = False, t: Tensor | None = None
    ) -> dict[str, Any]:
        lab = self.labels
        tt = lab.t() if t is None else t
        xd = None if x is None else x.to(self.dev)
        row, _ = score(xd, tt, lab.video, lab.k(), lab.held_out, jack=jack)
        return row

    def apply(self, fn: Callable[[Tensor], Tensor], *, last: bool = False) -> Tensor:
        return grid_apply(
            fn, self.grid, self.root, self.dev, self.sconf.batch, last=last
        )


def evaluate_frames(
    out: Path,
    *,
    conf: RunConf | None = None,
    sconf: SceneConf = SceneConf(),
    root: Path | None = None,
) -> Path:
    """``out``: each video's places (the last frames the labels need),
    sorted, once each."""
    conf = _yard_conf("cpu") if conf is None else conf
    root = tours_root(conf.data.tours.data_version) if root is None else root
    grid = scene_grid(conf, root, sconf)
    lists: dict[str, set[int]] = {}
    for v, f in grid.places:
        lists.setdefault(v, set()).add(f)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, {v: sorted(f) for v, f in sorted(lists.items())})
    return out


def evaluate_labels(
    hires: Path,
    weights: Path,
    cache: Path | None = None,
    *,
    conf: RunConf | None = None,
    sconf: SceneConf = SceneConf(),
    root: Path | None = None,
    net: torch.nn.Module | None = None,
) -> Path:
    """The labels' cache (module docstring) of the grid of ``conf`` (default
    ``v16_config`` on CUDA) from the hires data ``hires`` by Places365's
    weights ``weights`` (or ``net``), under ``cache`` (default
    ``$MUJOU_EXP_DATA``). Returns its directory (an existing one is kept)."""
    conf = _yard_conf(None) if conf is None else conf
    configure_determinism(conf.device)
    root = tours_root(conf.data.tours.data_version) if root is None else root
    grid = scene_grid(conf, root, sconf)
    base = _default_base() if cache is None else cache
    path = base / f"{CACHE_PREFIX}{labels_key(grid, sconf, hires.name)}"
    if (path / "labels.pt").is_file():
        return path
    if net is None:
        net = places365.load_places365(weights, conf.device)
    probs, meta = build_labels(
        grid,
        HiresFrames(hires),
        net,
        sconf,
        root=root,
        held_out=conf.data.tours.held_out,
    )
    meta["key"] = path.name
    meta["provenance"] = {k: v for k, v in of_diagnosis({}).items() if k != "settings"}
    SceneLabels(probs, meta).save(path)
    return path


def audit_sample(
    t: Tensor, video: Tensor, k: int, n: int, gen: torch.Generator
) -> list[list[int]]:
    """Of each kept class of ``t`` ``(M,)`` (-1 left out): ``n`` of its rows
    drawn without replacement, in proportion to the videos' counts of it
    (``video`` ``(M,)``; the remainders to the largest fractions, then the
    first videos), at random within each video (``gen``)."""
    out = []
    for c in range(k):
        rows = (t == c).nonzero().flatten()
        vids, counts = torch.unique(video[rows], return_counts=True)
        want = min(n, len(rows))
        exact = counts.double() * want / len(rows)
        take = exact.floor().long()
        rest = want - int(take.sum())
        order = sorted(range(len(vids)), key=lambda i: (-float(exact[i] - take[i]), i))
        for i in order[:rest]:
            take[i] += 1
        picked = []
        for i, v in enumerate(vids.tolist()):
            own = rows[video[rows] == v]
            picked += own[
                torch.randperm(len(own), generator=gen)[: int(take[i])]
            ].tolist()
        out.append(sorted(picked))
    return out


def _sheets(images: Tensor, path: Path, stem: str) -> list[str]:
    """``images`` u8 ``(n, 3, S, S)`` as 4 x 4 sheets ``<stem>-<j>.png``
    (numbered 1 to 16 in reading order); their names."""
    names = []
    for j, start in enumerate(range(0, len(images), 16)):
        chunk = images[start : start + 16].double() / 255
        pad = 16 - len(chunk)
        if pad:
            chunk = torch.cat([chunk, torch.full((pad, *chunk.shape[1:]), 0.5)])
        name = f"{stem}-{j:02d}.png"
        write_png(path / name, tile(chunk.view(4, 4, *chunk.shape[1:])))
        names.append(name)
    return names


def evaluate_audit(
    out: Path,
    hires: Path,
    *,
    conf: RunConf | None = None,
    sconf: SceneConf = SceneConf(),
    root: Path | None = None,
    labels: Path | None = None,
    cache: Path | None = None,
) -> Path:
    """V-c2's and V-E1'6's sheets (``plans/jepa.md``, 「監査」) into ``out``:
    of each kept group of ``full``, ``AUDIT_N`` frames of its label (and of
    each kept group of the quarters, ``AUDIT_N_QUAD`` quarters, the four
    pooled) by video (``audit_sample``, ``AUDIT_SEED``), in a random order,
    at 320 px (a quarter enlarged) and, apart, at 64 px enlarged;
    ``key.json`` (each number's place, quarter and teacher's group: not for
    the judge), ``groups.json`` (the groups and their classes: for the
    judge) and ``view-<g>.png`` (16 of each group, the visual check)."""
    conf = _yard_conf("cpu") if conf is None else conf
    ctx = _Context(conf, out, sconf=sconf, root=root, labels=labels, cache=cache)
    lab, grid = ctx.labels, ctx.grid
    data = HiresFrames(hires)
    gen = torch.Generator().manual_seed(AUDIT_SEED)
    names = lab.meta["groups"]
    member = places365.membership()
    classes = places365.class_names()
    items: list[dict[str, Any]] = []
    full = lab.meta["full"]
    for c, rows in enumerate(audit_sample(lab.t(), lab.video, lab.k(), AUDIT_N, gen)):
        items += [{"row": r, "quarter": None, "group": full["kept"][c]} for r in rows]
    pooled_t, pooled_v, pooled_at = [], [], []
    for q in QUARTER_NAMES:
        mq = lab.meta[f"quad/{q}"]
        tq = torch.tensor([mq["kept"][i] if i >= 0 else -1 for i in mq["t"]])
        pooled_t.append(tq)
        pooled_v.append(lab.video)
        pooled_at += [(r, q) for r in range(len(tq))]
    tp, vp = torch.cat(pooled_t), torch.cat(pooled_v)
    quad_groups = sorted(
        {g for q in QUARTER_NAMES for g in lab.meta[f"quad/{q}"]["kept"]}
    )
    tp_idx = torch.full_like(tp, -1)
    for j, g in enumerate(quad_groups):
        tp_idx[tp == g] = j
    for j, rows in enumerate(
        audit_sample(tp_idx, vp, len(quad_groups), AUDIT_N_QUAD, gen)
    ):
        for r in rows:
            row, q = pooled_at[r]
            items.append({"row": row, "quarter": q, "group": quad_groups[j]})
    order = torch.randperm(len(items), generator=gen).tolist()
    items = [items[i] for i in order]
    big, small = [], []
    for it in items:
        place = grid.places[it["row"]]
        x = data.frames([place])[0]
        if it["quarter"] is not None:
            x = quarters(x[None])[it["quarter"]][0]
            x = x.repeat_interleave(2, 1).repeat_interleave(2, 2)
        big.append(x)
        v, f = place
        lo = U8Reader(ctx.root / v)[f]
        if it["quarter"] is not None:
            lo = quarters(lo[None])[it["quarter"]][0]
        factor = x.shape[-1] // lo.shape[-1]
        small.append(lo.repeat_interleave(factor, 1).repeat_interleave(factor, 2))
    out.mkdir(parents=True, exist_ok=True)
    sheets = _sheets(torch.stack(big), out, "sheet320")
    sheets64 = _sheets(torch.stack(small), out, "sheet64")
    key = [
        {"number": i + 1, "sheet": sheets[i // 16], "sheet64": sheets64[i // 16],
         "place": list(grid.places[it["row"]]), "quarter": it["quarter"],
         "teacher": names[it["group"]]}
        for i, it in enumerate(items)
    ]  # fmt: skip
    write_json(out / "key.json", key)
    kept = sorted(set(full["kept"]) | set(quad_groups))
    groups = {
        names[g]: [classes[c] for c in member[:, g].nonzero().flatten().tolist()]
        for g in kept
    }
    write_json(out / "groups.json", groups)
    for c, g in enumerate(full["kept"]):
        rows = (lab.t() == c).nonzero().flatten()
        pick = rows[torch.randperm(len(rows), generator=gen)[:16]]
        x = data.frames([grid.places[r] for r in pick.tolist()])
        write_png(
            out / f"view-{g:02d}.png",
            tile(_pad16(x).view(4, 4, *x.shape[1:])),
        )
    return out


def _pad16(x: Tensor) -> Tensor:
    """``x`` u8 ``(n, 3, S, S)`` in [0, 1], padded with grey to 16."""
    y = x.double() / 255
    if len(y) < 16:
        y = torch.cat([y, torch.full((16 - len(y), *y.shape[1:]), 0.5)])
    return y


def _yard_codes(
    name: str,
    ctx: _Context,
    given: Mapping[str, Path | None],
    pca: dict[str, Tensor],
    *,
    features: Features | None,
) -> Tensor | None:
    """The codes ``(N, D)`` of the yardstick ``name`` at the grid (None for
    ``mean``); the pixels' PCA kept in ``pca`` for the next ``pca<k>``."""
    lab, conf = ctx.labels, ctx.conf
    if name == "mean":
        return None
    if name == "label":
        t = lab.t()
        return F.one_hot(t.clamp(min=0), lab.k()).float()
    if name == "video":
        return F.one_hot(lab.video, len(ctx.grid.videos)).float()
    if name == "places64":
        return lab.probs["places64"].float()
    if name == "inc64":
        assert features is not None
        return ctx.apply(features, last=True)
    if name in ("color", "color8"):
        return ctx.apply(ColorCode(16 if name == "color" else 8).mu)
    if name.startswith("pca"):
        if "all" not in pca:
            sem = sem_sets(
                conf, ctx.root, SemConf(), span=ctx.span, spacing=ctx.spacing
            )
            gen = torch.Generator().manual_seed(conf.seed + PCA_SEED)
            code = PCACode.fit(
                pixels, sem.fit, ctx.pca_max, gen=gen, device=conf.device
            )
            pca["all"] = ctx.apply(code.mu)
        return pca["all"][:, : int(name[3:])]
    if name in ("v16", "v17"):
        path = given[name]
        assert path is not None
        return ctx.apply(vae_at(path, conf.device).mu)
    raise ValueError(f"no yardstick {name!r}")


def evaluate_yardsticks(
    out_dir: Path,
    *,
    codes: Sequence[str] = YARDSTICKS,
    conf: RunConf | None = None,
    v16: Path | None = None,
    v17: Path | None = None,
    init_runs: Sequence[Path] = (),
    sconf: SceneConf = SceneConf(),
    root: Path | None = None,
    labels: Path | None = None,
    cache: Path | None = None,
    features: Features | None = None,
    inception: Path | None = None,
    spacing: Mapping[str, Any] | None = None,
    span: float = SPAN,
    pca_max: int = SemConf().pca_max,
) -> Path:
    """The yardsticks ``codes`` (``YARDSTICKS``) into ``out_dir / rows``;
    ``init_c`` of each run of ``init_runs`` (its step 0, a row
    ``init_c-<run>`` each); the pixels' PCA of ``pca<k>`` is sem_eval's
    (``spacing``, ``span``, ``pca_max``: smaller for tests). Returns
    ``out_dir``.

    Raises:
        ValueError: A code is not of ``YARDSTICKS``, or its checkpoint or
            runs are not given.
    """
    unknown = [c for c in codes if c not in YARDSTICKS]
    if unknown:
        raise ValueError(f"the yardsticks are of {YARDSTICKS}, not {unknown}")
    given = {"v16": v16, "v17": v17}
    lacking = [c for c in codes if c in given and given[c] is None]
    if "init_c" in codes and not init_runs:
        lacking.append("init_c")
    if lacking:
        raise ValueError(f"the yardsticks {lacking} need their checkpoint or runs")
    conf = _yard_conf(None) if conf is None else conf
    ctx = _Context(conf, out_dir, sconf=sconf, root=root, labels=labels, cache=cache)
    ctx.spacing, ctx.span, ctx.pca_max = spacing, span, pca_max
    if "inc64" in codes and features is None:
        features, _ = inception_features(inception, conf.device)
    pca: dict[str, Tensor] = {}
    jobs = [(c, None) for c in codes if c != "init_c"]
    if "init_c" in codes:
        jobs += [(f"init_c-{r.name}", r) for r in init_runs]
    for name, run in jobs:
        path = ctx.out / "rows" / f"{YARD_PREFIX}-{name}.json"
        if path.exists():
            continue
        ctx.start({"yardsticks": list(codes)})
        t0 = time.time()
        if run is not None:
            r_conf = _conf_of(run, conf.device)
            mods = build_run_jepa(r_conf, json.loads((run / "jepa.json").read_text()))
            x: Tensor | None = ctx.apply(mods.mu)
            code = "init_c"
        else:
            x = _yard_codes(name, ctx, given, pca, features=features)
            code = name
        stamp = {
            "step": None,
            "code": code,
            "source": None if run is None else str(run.resolve()),
        }
        jack = code in JACK_YARDS
        ctx.write(f"{YARD_PREFIX}-{name}", stamp | ctx.score(x, jack=jack), t0)
    return ctx.out


def evaluate_checks(
    out_dir: Path,
    *,
    conf: RunConf | None = None,
    sconf: SceneConf = SceneConf(),
    root: Path | None = None,
    labels: Path | None = None,
    cache: Path | None = None,
    features: Features | None = None,
    inception: Path | None = None,
    draws: int = N_RAND,
) -> Path:
    """V-c4 (``rand-<code>-<i>``: ``color`` and ``inc64`` to ``draws``
    random labels, ``random_labels``) and V-c5 (``syn_<q>``: the synthetic
    codes, ``synthetic_codes``, with their true $m^*(S_q)$ and the
    jackknife) into ``out_dir / rows``. Returns ``out_dir``."""
    conf = _yard_conf(None) if conf is None else conf
    ctx = _Context(conf, out_dir, sconf=sconf, root=root, labels=labels, cache=cache)
    lab = ctx.labels
    t, k, video = lab.t(), lab.k(), lab.video
    on = t >= 0
    for name, (q, s, code) in synthetic_codes(t, video, k).items():
        if (ctx.out / "rows" / f"{name}.json").exists():
            continue
        ctx.start({"draws": draws})
        t0 = time.time()
        truth = scene_m(s[on], t[on], video[on], k)
        ctx.write(name, {"code": name, "q": q, "m_true": truth}
                  | ctx.score(code, jack=True), t0)  # fmt: skip
    rand = random_labels(t, k, draws, RAND_SEED)
    for base in ("color", "inc64"):
        todo = [i for i in range(draws)
                if not (ctx.out / "rows" / f"rand-{base}-{i:02d}.json").exists()]  # fmt: skip
        if not todo:
            continue
        ctx.start({"draws": draws})
        if base == "inc64" and features is None:
            features, _ = inception_features(inception, conf.device)
        x = _yard_codes(base, ctx, {}, {}, features=features)
        for i in todo:
            t0 = time.time()
            row = ctx.score(x, t=rand[i])
            row.pop("pred")
            ctx.write(f"rand-{base}-{i:02d}", {"code": base, "draw": i} | row, t0)
    return ctx.out


def _run_codes(mods: JEPAModules, code: str) -> Callable[[Tensor], Tensor]:
    if code == "z":
        return mods.mu
    if code == "z_k3":
        return K3Code(mods).mu
    return TokenCode(mods, "h_mean").mu


def evaluate_run(
    run_dir: Path,
    *,
    steps: Sequence[int],
    codes: Sequence[str] = RUN_CODES,
    sconf: SceneConf = SceneConf(),
    root: Path | None = None,
    labels: Path | None = None,
    cache: Path | None = None,
    device: str | None = None,
) -> Path:
    """The rows of the JEPA run in ``run_dir`` (module docstring): ``z`` of
    the checkpoints of ``steps``, ``z_k3`` and ``h_mean`` of those of
    ``SIDE_STEPS``, the jackknife of ``z`` at ``JACK_STEPS``. Returns
    ``run_dir / OUT``.

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
        conf, run_dir / OUT, sconf=sconf, root=root, labels=labels, cache=cache
    )
    mods = build_run_jepa(conf, json.loads((run_dir / "jepa.json").read_text()))
    for step in steps:
        prefix = f"{step:06d}"
        todo = [
            c
            for c in codes
            if (c == "z" or step in SIDE_STEPS)
            and not (ctx.out / "rows" / f"{prefix}-{c}.json").exists()
        ]
        if not todo:
            continue
        ckpt.load_eval(ckpts[step], mods.modules())
        ctx.start({"run": run_dir.name})
        for code in todo:
            t0 = time.time()
            if ctx.dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats(ctx.dev)
            x = ctx.apply(_run_codes(mods, code))
            jack = code == "z" and step in JACK_STEPS
            stamp = {"step": step, "t": float(ckpts[step].stem), "code": code}
            ctx.write(f"{prefix}-{code}", stamp | ctx.score(x, jack=jack), t0)
    return ctx.out


# -- E1f of the scene categories ----------------------------------------------------------


def _cond(x: Tensor) -> float:
    """The condition number of the covariance of ``x`` ``(n, D)``."""
    return float(torch.linalg.cond(torch.atleast_2d(torch.cov(x.double().T))))


def e1f_scene_rows(
    stages: Mapping[tuple[str, int], Mapping[str, Tensor]],
    labels: SceneLabels,
    *,
    k_run: int,
) -> list[dict[str, Any]]:
    r"""E1f's rows of the scene categories (``plans/jepa.md``, 「``e1f`` の的
    （四半分のラベル）」) of the inputs ``stages[(q, k)][stage]`` ``(N,
    ·)`` at the grid's stacks: of each $k$ of ``E1F_K`` and stage of
    ``E1F_STAGES``, each quarter's ``score`` to its own kept labels
    (``quad/<q>``), ``pred_k3`` at $P$ times ``pool``'s chosen scale by
    video ($P$ its width over ``pool``'s: the same fit, ``fill_eval.
    e1f_fits``; ``CONST`` stays), ``mean`` the constant; $m$ the mean over
    the quarters. Each quarter's ``pred`` (and for ``E1F_JACK``, its
    ``jack_pred``) and ``pool``'s ``cond`` by quarter."""
    rows = []
    held = labels.held_out
    for k in E1F_K:
        pool_scales: dict[str, dict[int, float]] = {}
        for stage in E1F_STAGES:  # pool before pred_k3
            row: dict[str, Any] = {"k": k, "stage": stage, "in_dist": k == k_run}
            per: dict[str, dict[str, Any]] = {}
            for q in QUARTER_NAMES:
                kind = f"quad/{q}"
                t, kk = labels.t(kind), labels.k(kind)
                x = None if stage == "mean" else stages[(q, k)][stage]
                fixed = None
                if stage == "pred_k3":
                    p = (
                        stages[(q, k)][stage].shape[1]
                        // stages[(q, k)]["pool"].shape[1]
                    )
                    fixed = {
                        v: c if c == CONST else p * c for v, c in pool_scales[q].items()
                    }
                jack = (stage, k) in E1F_JACK
                got, chosen = score(
                    x, t, labels.video, kk, held, jack=jack, fixed=fixed
                )
                if stage == "pool":
                    assert x is not None and chosen is not None
                    pool_scales[q] = chosen
                    got["cond"] = _cond(x[t >= 0])
                per[q] = got
            row["m"] = sum(per[q]["m"] for q in QUARTER_NAMES) / len(QUARTER_NAMES)
            row["ba"] = sum(per[q]["ba"] for q in QUARTER_NAMES) / len(QUARTER_NAMES)
            row["quarters"] = per
            rows.append(row)
    return rows


@torch.no_grad()
def _e1f_inputs(
    mods: JEPAModules, ctx: _Context, expand: int
) -> dict[tuple[str, int], dict[str, Tensor]]:
    """E1f's inputs at the grid's stacks (``fill_eval.e1f_rows``'): of each
    quarter $q$ and $k$, ``pred``, ``pool``, ``pred_k3`` (the predictor's
    outputs at $Q_q$), ``vis_color`` (the visible coarse cells' mean
    colours) and ``oracle`` (the target side's tokens at $Q_q$)."""
    size = ctx.conf.data.tours.size
    color = ColorCode(size // COARSE)
    tok = ctx.apply(lambda x: mods.tokens(x).flatten(1))  # (N, T d)
    cells = ctx.apply(lambda x: color.tokens(x).flatten(1))
    cells = cells.view(len(cells), COARSE * COARSE, -1)
    dev = ctx.dev
    out: dict[tuple[str, int], dict[str, Tensor]] = {}
    for (q, k), coarse in e1f_masks().items():
        fine = expand_masks(coarse[None], expand)[0].to(dev)
        hide = fine.nonzero().flatten()
        quarter = expand_masks(EVAL_MASKS[f"Q_{q}"][None], expand)[0].to(dev)
        at = torch.isin(hide, quarter.nonzero().flatten()).cpu()
        visible = (~coarse).nonzero().flatten()
        k_d = ctx.apply(_predicted(mods, fine))
        z_hat = k_d.reshape(len(k_d), len(hide), -1)[:, at]
        got = e1f_inputs(z_hat)
        got["vis_color"] = cells[:, visible].flatten(1)
        tokens = tok.view(len(tok), quarter.numel(), -1)
        got["oracle"] = tokens[:, quarter.cpu()].flatten(1)
        out[(q, k)] = got
    return out


def evaluate_fill(
    run_dir: Path,
    *,
    steps: Sequence[int],
    sconf: SceneConf = SceneConf(),
    root: Path | None = None,
    labels: Path | None = None,
    cache: Path | None = None,
    device: str | None = None,
) -> Path:
    """E1f of the scene categories (``e1f_scene_rows``) of the checkpoints
    of ``steps`` of the JEPA run in ``run_dir`` into ``run_dir /
    OUT_FILL``. Returns it.

    Raises:
        ValueError: A step has no checkpoint.
    """
    conf = _conf_of(run_dir, device)
    ckpts = run_ckpts(run_dir)
    missing = [s for s in steps if s not in ckpts]
    if missing:
        raise ValueError(f"no checkpoint of steps {missing} (of {list(ckpts)})")
    ctx = _Context(
        conf, run_dir / OUT_FILL, sconf=sconf, root=root, labels=labels, cache=cache
    )
    settings, _, expand, k_run = _run_settings(run_dir)
    mods = build_run_jepa(conf, settings)
    for step in steps:
        name = f"{step:06d}-e1f"
        if (ctx.out / "rows" / f"{name}.json").exists():
            continue
        ckpt.load_eval(ckpts[step], mods.modules())
        ctx.start({"run": run_dir.name, "k": k_run, "expand": expand})
        t0 = time.time()
        rows = e1f_scene_rows(_e1f_inputs(mods, ctx, expand), ctx.labels, k_run=k_run)
        stamp = {"run": run_dir.name, "step": step, "t": float(ckpts[step].stem)}
        ctx.write(name, stamp | {"rows": rows}, t0)
    return ctx.out


# -- the command line ------------------------------------------------------------------------


def _ints(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split(","))


def _names(text: str) -> tuple[str, ...]:
    return tuple(text.split(","))


def _paths(text: str) -> tuple[Path, ...]:
    return tuple(Path(x) for x in text.split(","))


def main(argv: Sequence[str] | None = None) -> Path:
    """``python -m exp.vision.scene_eval`` (module docstring). Returns the
    output."""
    ap = argparse.ArgumentParser(
        prog="python -m exp.vision.scene_eval",
        description="V18c's main measure m: the coarse scene categories.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    fr = sub.add_parser("frames", help="the places the hires data must hold")
    fr.add_argument("--out", type=Path, required=True)
    lb = sub.add_parser("labels", help="the labels' cache (once)")
    lb.add_argument("--hires", type=Path, required=True)
    lb.add_argument("--weights", type=Path, required=True, help="Places365's")
    au = sub.add_parser("audit-sheets", help="V-c2's sheets")
    au.add_argument("--out", type=Path, required=True)
    au.add_argument("--hires", type=Path, required=True)
    yard = sub.add_parser("yardsticks", help="the yardsticks' rows")
    yard.add_argument("--out", type=Path, required=True)
    yard.add_argument("--codes", type=_names, default=YARDSTICKS)
    yard.add_argument("--v16", type=Path, help="a VAE checkpoint")
    yard.add_argument("--v17", type=Path, help="a VAE checkpoint")
    yard.add_argument("--init-runs", type=_paths, default=(), help="long runs")
    chk = sub.add_parser("checks", help="V-c4's and V-c5's rows")
    chk.add_argument("--out", type=Path, required=True)
    run = sub.add_parser("run", help="the rows of a JEPA run's checkpoints")
    run.add_argument("run_dir", type=Path)
    run.add_argument("--steps", type=_ints, required=True)
    run.add_argument("--codes", type=_names, default=RUN_CODES)
    fill = sub.add_parser("fill", help="E1f of the scene categories")
    fill.add_argument("run_dir", type=Path)
    fill.add_argument("--steps", type=_ints, required=True)
    for p in (fr, lb, au, yard, chk, run, fill):
        p.add_argument("--root", type=Path, help="the data (default: tours_root)")
        p.add_argument("--device", help="(default: the run's, or cuda)")
    for p in (lb, au, yard, chk, run, fill):
        p.add_argument("--cache", type=Path, help="(default: $MUJOU_EXP_DATA)")
    for p in (au, yard, chk, run, fill):
        p.add_argument("--labels", type=Path, help="the labels (default: found)")
    for p in (yard, chk):
        p.add_argument("--inception", type=Path, help="FID's weights")
    args = ap.parse_args(argv)
    common = {"root": args.root}
    if args.cmd in ("run", "fill"):
        fn = evaluate_run if args.cmd == "run" else evaluate_fill
        extra = {"codes": args.codes} if args.cmd == "run" else {}
        return fn(
            args.run_dir, steps=args.steps, labels=args.labels, cache=args.cache,
            device=args.device, **common, **extra,
        )  # fmt: skip
    conf = _yard_conf(args.device)
    if args.cmd == "frames":
        return evaluate_frames(args.out, conf=conf, **common)
    if args.cmd == "labels":
        return evaluate_labels(
            args.hires, args.weights, args.cache, conf=conf, **common
        )
    if args.cmd == "audit-sheets":
        return evaluate_audit(
            args.out, args.hires, conf=conf, labels=args.labels, cache=args.cache,
            **common,
        )  # fmt: skip
    if args.cmd == "checks":
        return evaluate_checks(
            args.out, conf=conf, labels=args.labels, cache=args.cache,
            inception=args.inception, **common,
        )  # fmt: skip
    return evaluate_yardsticks(
        args.out, codes=args.codes, conf=conf, v16=args.v16, v17=args.v17,
        init_runs=args.init_runs, labels=args.labels, cache=args.cache,
        inception=args.inception, **common,
    )  # fmt: skip


if __name__ == "__main__":
    print(main())
