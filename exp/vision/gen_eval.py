r"""V18c's G (``plans/jepa.md``, 「G: 生成で測る」; README 「V18c: 生成と
Inception で JEPA の符号を測り直す」): draw images from a code by a light
flow-matching decoder learned on it afterwards (``exp.lib.flow_decoder``),
and judge whether the drawings tell their own image apart from the others
by the features of FID's InceptionV3 (``exp.lib.inception``).

The data, the same for every code:

- the held-out video's grid (``Zurich``): the stacks whose first frames are
  ``GenConf.grid_step`` frames apart (1 s); of its $G$ stacks, the
  ``n_cand`` candidates $c_i = \lfloor i (G - 1) / (n - 1) \rfloor$
  (``candidates``), and for ``hard_top1`` the 33 grid stacks $c_i + 2m$, $m
  = 0, \pm 1, \dots, \pm 16$ of each candidate far enough from the ends
  (``hard_sets``);
- the reference set $R$ of FID: ``ref_count`` last frames spread evenly
  over the video (``spread``);
- the training set: ``set_size`` stacks drawn uniformly with replacement
  from the training videos (``iid.Units`` of ``"all"``, ``GEN_DATA_SEED``,
  not of a run's seed); a code's decoder learns its codes and their last
  frames (``training_set``).

A job, of a code and a decoder seed $j$ (``GenEvaluator.jobs``):
standardize the codes (``eval.Standardizer`` of the training set's; not
``uncond``'s), learn ``FlowUNet`` (its weights, batches, $t$ and noise of
``DEC_SEED`` + ``DEC_SEED_STEP`` $j$) for ``fit_steps``, draw ``samples``
images of each candidate by Euler's method of ``sample_steps`` from the
noise of ``SAMPLE_SEED`` (the same for every code), then the row:
``top1``, ``top5``, ``top1_se``, ``top1_adj``, ``pick_max`` (the largest
share of the picks on one candidate: a hub), ``hits`` (by
``exp.lib.metrics.identification`` of the drawn images' features against
the candidates' true last frames, both less the mean of $R$'s), ``hard_*``
of the same among the 33 neighbours, ``feat_dist`` $=
\frac{1}{nK}\sum_{i,j} \lVert g_{ij} - r_i \rVert^2$, ``feat_ratio`` (over
the mean of $\lVert r_i - r_l \rVert^2$, $i \ne l$), ``feat_i``, ``fid_j``
(of each sample $j$, the $n$ images drawn, one of each candidate, against
$R$) and ``fid`` their mean (not the $nK$ images at once: the closer a
code's $K$ samples of a candidate, the fewer distinct images, and the
larger FID's bias of a finite sample, about $1/N$; apart, every code has
$n$), ``fm_zurich`` (the flow's loss on the
candidates at the fixed times $(m + 0.5) / 16$ and the noise of
``FM_SEED``), ``fit_loss`` and ``fit_loss_first`` (the means of the last and
first ``loss_window`` steps), timings (``sec_encode`` of the job's codes
at once: one pass over the training set encodes them all) and memory. The
picture
``<prefix>-<code>-d<j>.png`` (16 candidates; their true frame, 4 samples,
the mean and 4 x the sd of the samples) and the decoder
``ckpt/dec-<prefix>-<code>-d<j>.pt`` (with its standardizer, for E1g).

Codes: of a run's checkpoint ``z`` (the code of V18's and V18b's
evaluation: ``JEPAModules.mu``, or ``zpca`` of an arm M) and ``z_w``
(``jepa_eval.RandProjCode``); the yardsticks ``color``, ``pca``, ``v16``,
``v17``, ``init_m`` (arm M's ``zpca`` at step 0) and ``uncond`` (0 in
``UNCOND_DIM``, not standardized). ``refs`` writes ``refs.json``:
``real_shift`` (the candidates found by the true last frame of the grid's
next stack, 1 s on), ``fid_cand`` (the candidates' $n$ true last frames
against $R$: the floor of ``fid``) and ``fid_train`` (the mean over $K$
groups of $n$ of the training set's last frames against $R$, as ``fid``:
the shift from the training videos to Zurich).

Into ``<run_dir>/posthoc_gen/`` (or ``--out-name``; yardsticks and refs:
``<out>/posthoc_gen/``): ``config.json`` (``GenEvaluator.settings``, its
``key`` the sha256 of what the rows depend on, and the parameter counts of
every code dimension), ``rows/<prefix>-<code>-d<j>.json`` (``<prefix>`` the
step in 6 digits, or ``yardstick``; each with the ``key``; a job is skipped
where its row is, and a ``config.json`` or a row of another ``key`` is
refused), the pictures, ``ckpt/``::

    python -m exp.vision.gen_eval run <run_dir> --steps 0,6000,20000,60000 \
        --codes z,z_w [--dec-seeds 0] [--fit-steps N] [--inception <path>] \
        [--out-name posthoc_gen_trial]
    python -m exp.vision.gen_eval yardsticks --out runs/v18c-gen/yardsticks \
        --v16 <ckpt> --v17 <ckpt> --m-run <run_dir> \
        [--codes color,pca,v16,v17,init_m,uncond]
    python -m exp.vision.gen_eval refs --out runs/v18c-gen/yardsticks
"""

import argparse
import hashlib
import json
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor, nn

from exp.lib import ckpt
from exp.lib.flow_decoder import FlowUNet, euler_sample, fit_flow, flow_loss
from exp.lib.inception import fid as fid_of, load_fid_inception, sha256, weights_path
from exp.lib.metrics import identification
from exp.lib.png import tile, write_png
from exp.lib.provenance import configure_determinism, of_diagnosis
from exp.lib.rundir import write_json
from exp.lib.tours.stacks import tours_root
from exp.lib.video_io import U8Reader
from exp.vision.compose import from_dict
from exp.vision.eval import BATCH, PCA_SEED, Standardizer
from exp.vision.iid import Units
from exp.vision.jepa import JEPAModules
from exp.vision.jepa_diag import GRID_STEP, grid_starts, run_ckpts
from exp.vision.jepa_eval import (
    PCA_COMPONENTS,
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
from exp.vision.schema import RunConf
from exp.vision.stream_eval import Code, StreamEvaluator, spanned

__all__ = [
    "DEC_SEED",
    "DEC_SEED_STEP",
    "FM_SEED",
    "GEN_DATA_SEED",
    "OUT",
    "RUN_CODES",
    "SAMPLE_SEED",
    "UNCOND_DIM",
    "YARDSTICKS",
    "YARD_PREFIX",
    "GenConf",
    "GenEvaluator",
    "ZeroCode",
    "Zurich",
    "candidates",
    "evaluate_refs",
    "evaluate_run",
    "evaluate_yardsticks",
    "hard_sets",
    "inception_features",
    "main",
    "run_codes",
    "spread",
    "step_codes",
    "training_set",
]

OUT = "posthoc_gen"
GEN_DATA_SEED = 53  # the training set's draws: not added to a run's seed
DEC_SEED, DEC_SEED_STEP = 67, 1000  # decoder j: DEC_SEED + DEC_SEED_STEP j
SAMPLE_SEED = 71  # the noise of the samples, the same for every code
FM_SEED = 73  # the noise of fm_zurich
UNCOND_DIM = 1024
RUN_CODES = ("z", "z_w")
YARD_PREFIX = "yardstick"  # the prefix of the yardsticks' rows and pictures
YARDSTICKS = ("color", "pca", "v16", "v17", "init_m", "uncond")

type Features = Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class GenConf:
    """The settings of G (module docstring; ``plans/jepa.md``), by default
    the plan's. ``net`` passes ``FlowUNet``'s keywords (its defaults are the
    plan's net)."""

    n_cand: int = 512
    samples: int = 8  # K: images drawn of each candidate
    sample_steps: int = 100
    set_size: int = 131072  # 2^17 training stacks
    fit_steps: int = 10000  # provisional: set by the trial (plans/jepa.md)
    fit_batch: int = 128
    fit_lr: float = 2e-4
    warmup: int = 500
    loss_window: int = 1000  # fit_loss: the mean of the last (V-G3: first)
    log_every: int = 200
    ref_count: int = 4096  # R, FID's reference set
    grid_step: int = GRID_STEP  # frames between the grid's stacks (1 s)
    hard_reach: int = 16  # m = +-1 .. +-hard_reach of hard_top1's neighbours
    hard_stride: int = 2  # grid stacks between them (2 s)
    fm_points: int = 16
    picture_rows: int = 16
    picture_samples: int = 4
    picture_scale: int = 2
    sample_batch: int = 256  # images through the net at once when drawing
    net: Mapping[str, Any] = field(default_factory=dict)


def candidates(g: int, n: int) -> Tensor:
    r"""The grid indices $c_i = \lfloor i (g - 1) / (n - 1) \rfloor$, $i < n$
    ``(n,)``: evenly spread from the first to the last.

    Raises:
        ValueError: ``n`` is not in ``[2, g]``.
    """
    if not 2 <= n <= g:
        raise ValueError(f"{n} candidates of a grid of {g}")
    return torch.arange(n) * (g - 1) // (n - 1)


def hard_sets(cand: Tensor, g: int, reach: int, stride: int) -> tuple[Tensor, Tensor]:
    r"""The candidates of ``hard_top1``: their positions ``(q,)`` in
    ``cand`` (those whose neighbours $c_i + s m$, $s$ = ``stride``, $\lvert m
    \rvert \le$ ``reach``, are all on the grid of ``g``) and those grid
    indices ``(q, 2 reach + 1)``, $c_i$ first."""
    offsets = torch.arange(1, reach + 1) * stride
    offsets = torch.cat([torch.zeros(1, dtype=torch.long), -offsets.flip(0), offsets])
    far = reach * stride
    keep = ((cand >= far) & (cand <= g - 1 - far)).nonzero().flatten()
    return keep, cand[keep][:, None] + offsets


def spread(count: int, frames: int, span: int) -> Tensor:
    r"""The last frames ``(count,)`` of ``count`` stacks of ``span`` + 1
    frames spread evenly over a video of ``frames``: the first $\lfloor i
    (F - 1 - \text{span}) / (\text{count} - 1) \rfloor$, the last ``span``
    after it."""
    first = torch.arange(count) * (frames - 1 - span) // (count - 1)
    return first + span


class ZeroCode:
    """``uncond``: 0 ``(B, dim)`` for every stack."""

    def __init__(self, dim: int = UNCOND_DIM) -> None:
        self._dim = dim

    def modules(self) -> Mapping[str, nn.Module]:
        return {}

    def mu(self, x: Tensor) -> Tensor:
        return x.new_zeros(len(x), self._dim, dtype=torch.float32)


class Zurich:
    """The held-out video of ``conf`` at ``root`` (all its frames, u8 on the
    CPU) and its grid of stacks every ``step`` frames (``jepa_diag.
    grid_starts``)."""

    def __init__(self, conf: RunConf, root: Path, step: int) -> None:
        t = conf.data.tours
        reader = U8Reader(root / t.held_out)
        self.frames = reader[0 : len(reader)]  # (F, 3, S, S)
        self.starts = grid_starts(len(reader), t.frames, t.stride, step)
        self.span = (t.frames - 1) * t.stride  # the last frame after the first
        self._offsets = torch.arange(t.frames) * t.stride

    def __len__(self) -> int:
        return len(self.starts)

    def stacks(self, idx: Tensor) -> Tensor:
        """The grid's stacks ``idx`` u8 ``(b, n, 3, S, S)``."""
        return self.frames[self.starts[idx][:, None] + self._offsets]

    def last(self, idx: Tensor) -> Tensor:
        """The last frames u8 ``(b, 3, S, S)`` of the grid's stacks ``idx``."""
        return self.frames[self.starts[idx] + self.span]


@torch.no_grad()
def _encode(code: Code, x: Tensor, dev: torch.device) -> Tensor:
    """``code.mu`` of the stacks u8 ``x`` (CPU), ``BATCH`` at a time,
    without autograd."""
    return torch.cat([code.mu(c.to(dev)).float() for c in x.split(BATCH)])


def _set_batches(conf: RunConf, root: Path, size: int) -> Iterator[Tensor]:
    """The training set's stacks u8 ``(b, n, 3, S, S)``, ``BATCH`` at a time
    in order (module docstring)."""
    units = Units(conf, root, "all", seed=GEN_DATA_SEED)
    for i in range(0, size, BATCH):
        yield units.draw(min(BATCH, size - i))


@torch.no_grad()
def training_set(
    conf: RunConf, root: Path, codes: Mapping[str, Code], size: int
) -> tuple[dict[str, Tensor], Tensor]:
    """The codes ``(size, D)`` (on ``conf.device``) of each of ``codes`` and
    the last frames u8 ``(size, 3, S, S)`` (CPU) of the training set's
    ``size`` stacks (module docstring), in one pass."""
    dev = torch.device(conf.device)
    z: dict[str, list[Tensor]] = {name: [] for name in codes}
    last: list[Tensor] = []
    for x in _set_batches(conf, root, size):
        xd = x.to(dev)
        for name, code in codes.items():
            z[name].append(code.mu(xd).float())
        last.append(x[:, -1])
    return {k: torch.cat(v) for k, v in z.items()}, torch.cat(last)


def _features(features: Features, frames: Tensor) -> Tensor:
    """The features (float64, CPU) of the u8 frames ``(N, 3, S, S)``,
    ``BATCH`` at a time."""
    return torch.cat([features(c).double().cpu() for c in frames.split(BATCH)])


def _max_mem(dev: torch.device) -> float | None:
    if dev.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(dev) / 2**30


class GenEvaluator:
    """G's data and jobs (module docstring) of ``conf`` (its data, seed and
    device) at ``root``, into ``out``.

    Args:
        conf: The config (of the run, or ``v16_config`` for the yardsticks).
        root: The prepared Walking Tours.
        out: The output directory (``.../posthoc_gen``).
        features: u8 frames ``(B, 3, S, S)`` (on the device) to their
            features ``(B, F)``: FID's Inception in use.
        gconf: G's settings.
        inception: The record of the features (the weights' path and
            sha256) for ``config.json``.
        span: Seconds of the probes' span, whose fit stacks fit the PCAs.
        spacing: ``StreamEvaluator``'s spacings by name (tests).
        pca_components, zpca_dim: The PCAs' components of ``pca`` and
            ``zpca`` (for ``depends``; the codes are made by the caller).
    """

    def __init__(
        self,
        conf: RunConf,
        root: Path,
        out: Path,
        *,
        features: Features,
        gconf: GenConf = GenConf(),
        inception: Mapping[str, Any] | None = None,
        span: float = SPAN,
        spacing: Mapping[str, float] | None = None,
        pca_components: int = PCA_COMPONENTS,
        zpca_dim: int = ZPCA_DIM,
    ) -> None:
        self.conf, self._root, self.out, self.gconf = conf, root, out, gconf
        self._pca = {
            "seed": self.conf.seed + PCA_SEED,
            "components": pca_components,
            "zpca_dim": zpca_dim,
        }
        self._dev = torch.device(conf.device)
        self._features = features
        self._view = spanned(conf, span)
        self._spacing: dict[str, Any] = dict(spacing or {})
        self._fit: Tensor | None = None
        self.determinism = configure_determinism(conf.device)
        g = gconf
        self.zurich = z = Zurich(conf, root, g.grid_step)
        self.cand = candidates(len(z), g.n_cand)
        self.hard_q, self.hard_among = hard_sets(
            self.cand, len(z), g.hard_reach, g.hard_stride
        )
        self.ref = spread(g.ref_count, len(z.frames), z.span)
        self.grid_feats = _features(features, z.last(torch.arange(len(z))))
        self.ref_feats = _features(features, z.frames[self.ref])
        self.center = self.ref_feats.mean(0)
        real = self.grid_feats[self.cand]
        self._real_spread = float(
            torch.cdist(real, real).square().sum() / (len(real) * (len(real) - 1))
        )
        self._inception = dict(inception or {})
        out.mkdir(parents=True, exist_ok=True)

    # -- the data --------------------------------------------------------------

    @property
    def fit_stacks(self) -> Tensor:
        """The probes' fit stacks (``StreamEvaluator.fit_stacks`` of the span
        ``SPAN``), which the PCAs fit (``zpca``, ``pca``)."""
        if self._fit is None:
            ev = StreamEvaluator(
                self._view,
                self.out.parent,
                self._root,
                self.out,
                c=1,
                **self._spacing,
            )
            self._fit = ev.fit_stacks
        return self._fit

    def depends(self) -> dict[str, Any]:
        """What the rows depend on beside the code: G's settings, the data
        (``conf.data.tours``: the training set, the grid and the
        candidates), the seeds (``z_w``'s $W$ and the PCAs' too), the
        candidates, $R$ and the Inception's sha256 (not its path, which
        differs by host)."""
        g = self.gconf
        return json.loads(
            json.dumps(
                {
                    "gen": asdict(g) | {"net": dict(g.net)},
                    "tours": asdict(self.conf.data.tours),
                    "pca": self._pca,
                    "seeds": {
                        "gen_data": GEN_DATA_SEED,
                        "dec": DEC_SEED,
                        "dec_step": DEC_SEED_STEP,
                        "sample": SAMPLE_SEED,
                        "fm": FM_SEED,
                        "w": W_SEED,
                        "w_dim": W_DIM,
                        "w_chunk": W_CHUNK,
                    },
                    "grid": len(self.zurich),
                    "frames": len(self.zurich.frames),
                    "cand": self.cand.tolist(),
                    "n_hard": len(self.hard_q),
                    "ref": self.ref.tolist(),
                    "inception_sha256": self._inception.get("sha256"),
                }
            )
        )

    @property
    def key(self) -> str:
        """The first 12 hex digits of the sha256 of ``depends()``: in
        ``config.json`` and in every row."""
        text = json.dumps(self.depends(), sort_keys=True)
        return hashlib.sha256(text.encode()).hexdigest()[:12]

    def settings(self) -> dict[str, Any]:
        """``config.json`` but the parameter counts: ``depends()``, its
        ``key``, the Inception's record and the machine's."""
        return self.depends() | {
            "key": self.key,
            "inception": self._inception,
            "determinism": self.determinism,
            "provenance": {
                k: v for k, v in of_diagnosis({}).items() if k != "settings"
            },
        }

    def _start(self, code_dims: Mapping[str, int]) -> None:
        """``config.json`` (the first job's record) with the parameter count
        of each code's dimension, those of later jobs' new dimensions added.

        Raises:
            ValueError: The ``config.json`` there is of other settings.
        """
        path = self.out / "config.json"
        old = json.loads(path.read_text()) if path.exists() else None
        if old is not None and old.get("key") != self.key:
            raise ValueError(
                f"{path} is of other settings (key {old.get('key')}, not "
                f"{self.key}): write elsewhere (--out-name) or remove it"
            )
        have: dict[str, int] = {} if old is None else old.get("params", {})
        dims = sorted({d for d in code_dims.values() if str(d) not in have})
        if not dims and old is not None:
            return
        with torch.random.fork_rng(devices=[]):  # the global RNG left as it was
            params = have | {
                str(d): sum(
                    p.numel() for p in FlowUNet(d, **self.gconf.net).parameters()
                )
                for d in dims
            }
        base = self.settings() if old is None else old
        write_json(path, base | {"params": params})

    # -- the jobs ---------------------------------------------------------------

    def row_path(self, prefix: str, name: str, j: int) -> Path:
        return self.out / "rows" / f"{prefix}-{name}-d{j}.json"

    def done(self, prefix: str, name: str, j: int) -> bool:
        """Whether the job's row is there.

        Raises:
            ValueError: The row there is of other settings (its ``key``).
        """
        path = self.row_path(prefix, name, j)
        if not path.exists():
            return False
        key = json.loads(path.read_text()).get("key")
        if key != self.key:
            raise ValueError(
                f"{path} is of other settings (key {key}, not {self.key}): "
                "write elsewhere (--out-name) or remove it"
            )
        return True

    def jobs(
        self,
        codes: Mapping[str, Code],
        *,
        prefix: str,
        stamp: Mapping[str, Any],
        dec_seeds: Sequence[int] = (0,),
        plain: Sequence[str] = ("uncond",),
    ) -> list[Path]:
        """The jobs of ``codes`` and the decoder seeds ``dec_seeds`` (module
        docstring) whose rows are not there, with ``stamp`` in each row; the
        codes of ``plain`` are not standardized. Returns the rows written."""
        pending = [
            (name, j)
            for name in codes
            for j in dec_seeds
            if not self.done(prefix, name, j)
        ]
        if not pending:
            return []
        self._start({})  # refuses another config.json before the codes' time
        t0 = time.perf_counter()
        needed = {name: codes[name] for name in dict.fromkeys(n for n, _ in pending)}
        z_set, frames = training_set(self.conf, self._root, needed, self.gconf.set_size)
        with torch.no_grad():
            stacks = self.zurich.stacks(self.cand)
            z_cand = {n: _encode(c, stacks, self._dev) for n, c in needed.items()}
        sec_encode = time.perf_counter() - t0
        self._start({n: z.shape[1] for n, z in z_set.items()})
        written = []
        for name, j in pending:
            std = None if name in plain else Standardizer(z_set[name])
            row = self._job(
                name, j, z_set[name], frames, z_cand[name], std, prefix=prefix
            )
            row = dict(stamp) | row | {"sec_encode": sec_encode, "key": self.key}
            path = self.row_path(prefix, name, j)
            path.parent.mkdir(parents=True, exist_ok=True)
            write_json(path, row)
            written.append(path)
        return written

    def _job(
        self,
        name: str,
        j: int,
        z_set: Tensor,
        frames: Tensor,
        z_cand: Tensor,
        std: Standardizer | None,
        *,
        prefix: str,
    ) -> dict[str, Any]:
        g, dev = self.gconf, self._dev
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
        seed = DEC_SEED + DEC_SEED_STEP * j
        c_set = z_set if std is None else std(z_set)
        c_cand = z_cand if std is None else std(z_cand)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            net = FlowUNet(c_set.shape[1], **g.net)
        net = net.to(dev)
        t0 = time.perf_counter()
        losses = fit_flow(
            net,
            c_set,
            frames,
            steps=g.fit_steps,
            batch=g.fit_batch,
            lr=g.fit_lr,
            warmup=g.warmup,
            gen=torch.Generator().manual_seed(seed),
        )
        sec_fit = time.perf_counter() - t0
        t0 = time.perf_counter()
        drawn = self._draw(net, c_cand)  # (n, K, 3, S, S) u8
        feats = _features(self._features, drawn.flatten(0, 1))
        feats = feats.reshape(len(drawn), g.samples, -1)
        sec_sample = time.perf_counter() - t0
        name_of = f"{prefix}-{name}-d{j}"
        self._picture(name_of, drawn)
        self._save(name_of, net, std, c_set.shape[1])
        w = min(g.loss_window, len(losses))
        return (
            {
                "code": name,
                "dim": c_set.shape[1],
                "dec_seed": j,
                "fit_steps": g.fit_steps,
                "fit_batch": g.fit_batch,
                "fit_loss": float(losses[-w:].mean()),
                "fit_loss_first": float(losses[:w].mean()),
                "fit_log": _log(losses, g.log_every),
                "fm_zurich": self._fm(net, c_cand),
            }
            | self._scores(feats)
            | {
                "sec_fit": sec_fit,
                "sec_sample": sec_sample,
                "max_mem": _max_mem(dev),
            }
        )

    def _scores(self, feats: Tensor) -> dict[str, Any]:
        """The scores of the drawn images' features ``(n, K, F)`` (module
        docstring)."""
        real = self.grid_feats[self.cand]
        easy = identification(feats, real, center=self.center)
        hard = identification(
            feats[self.hard_q],
            self.grid_feats,
            center=self.center,
            truth=self.cand[self.hard_q],
            among=self.hard_among,
        )
        dist = (feats - real[:, None]).square().sum(2)  # (n, K)
        feat_i = dist.mean(1)
        # each sample j apart: n images of n candidates for every code (module
        # docstring)
        fid_j = [fid_of(feats[:, j], self.ref_feats) for j in range(feats.shape[1])]
        return {
            "top1": easy["top1"],
            "top5": easy["top5"],
            "top1_se": easy["top1_se"],
            "top1_adj": easy["top1_adj"],
            "pick_max": easy["pick_max"],
            "hits": easy["hits"],
            "hard_top1": hard["top1"],
            "hard_top1_se": hard["top1_se"],
            "hard_hits": hard["hits"],
            "n_hard": len(self.hard_q),
            "feat_dist": float(feat_i.mean()),
            "feat_ratio": float(feat_i.mean()) / self._real_spread,
            "feat_i": feat_i.tolist(),
            "fid": sum(fid_j) / len(fid_j),
            "fid_j": fid_j,
        }

    @torch.no_grad()
    def _draw(self, net: nn.Module, c_cand: Tensor) -> Tensor:
        """``samples`` images u8 ``(n, K, 3, S, S)`` of each candidate's code
        from the noise of ``SAMPLE_SEED``: Euler's ``sample_steps``, clipped
        to $[-1, 1]$, $\\mathrm{round}((x + 1) / 2 \\cdot 255)$."""
        g = self.gconf
        n, k = len(c_cand), g.samples
        shape = self.zurich.frames.shape[1:]
        noise = torch.randn(
            n, k, *shape, generator=torch.Generator().manual_seed(SAMPLE_SEED)
        ).flatten(0, 1)
        codes = c_cand.repeat_interleave(k, 0)
        out = []
        for i in range(0, n * k, g.sample_batch):
            x0 = noise[i : i + g.sample_batch].to(self._dev)
            x = euler_sample(net, x0, codes[i : i + g.sample_batch], g.sample_steps)
            u8 = ((x.clamp(-1, 1) + 1) / 2 * 255).round().to(torch.uint8)
            out.append(u8.cpu())
        return torch.cat(out).reshape(n, k, *shape)

    @torch.no_grad()
    def _fm(self, net: nn.Module, c_cand: Tensor) -> float:
        """``fm_zurich``: the flow's loss on the candidates' last frames at
        $t_m = (m + 0.5) / P$, $m < P$ = ``fm_points``, of the noise of
        ``FM_SEED``."""
        g = self.gconf
        gen = torch.Generator().manual_seed(FM_SEED)
        x1 = self.zurich.last(self.cand).to(self._dev).float() * (2 / 255) - 1
        total = 0.0
        for m in range(g.fm_points):
            x0 = torch.randn(x1.shape, generator=gen).to(self._dev)
            t = torch.full((len(x1),), (m + 0.5) / g.fm_points, device=self._dev)
            for i in range(0, len(x1), g.sample_batch):
                s = slice(i, i + g.sample_batch)
                part = flow_loss(net, x1[s], c_cand[s], t=t[s], x0=x0[s])
                total += float(part) * len(x1[s])
        return total / (g.fm_points * len(x1))

    def _picture(self, name: str, drawn: Tensor) -> None:
        """``<name>.png``: of ``picture_rows`` candidates evenly spaced, the
        true last frame, ``picture_samples`` samples, their mean and 4 x their
        sd."""
        g = self.gconf
        rows = torch.arange(g.picture_rows) * (len(drawn) // g.picture_rows)
        real = self.zurich.last(self.cand[rows]).float() / 255  # (r, 3, S, S)
        x = drawn[rows].float() / 255  # (r, K, 3, S, S)
        cols = [real[:, None], x[:, : g.picture_samples]]
        cols += [x.mean(1, keepdim=True), (4 * x.std(1, keepdim=True)).clamp(0, 1)]
        write_png(self.out / f"{name}.png", tile(torch.cat(cols, 1), g.picture_scale))

    def _save(
        self, name: str, net: nn.Module, std: Standardizer | None, d: int
    ) -> None:
        """The decoder and its standardizer into ``ckpt/dec-<name>.pt``."""
        path = self.out / "ckpt" / f"dec-{name}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "net": {k: v.cpu() for k, v in net.state_dict().items()},
                "net_kw": dict(self.gconf.net),
                "code_dim": d,
                "mean": None if std is None else std.mean.cpu(),
                "sd": None if std is None else std.sd,
            },
            path,
        )

    # -- the references of real frames -----------------------------------------

    def refs(self) -> dict[str, Any]:
        """``real_shift``, ``fid_cand`` and ``fid_train`` (module
        docstring)."""
        g, z = self.gconf, self.zurich
        nxt = torch.where(self.cand + 1 < len(z), self.cand + 1, self.cand - 1)
        shift = identification(
            self.grid_feats[nxt], self.grid_feats[self.cand], center=self.center
        )
        n = len(self.cand)
        _, frames = training_set(self.conf, self._root, {}, n * g.samples)
        train = _features(self._features, frames).reshape(g.samples, n, -1)
        fid_train = [fid_of(f, self.ref_feats) for f in train]
        return {
            "real_shift_top1": shift["top1"],
            "real_shift_top5": shift["top5"],
            "real_shift_hits": shift["hits"],
            "fid_cand": fid_of(self.grid_feats[self.cand], self.ref_feats),
            "fid_train": sum(fid_train) / len(fid_train),
            "fid_train_j": fid_train,
            "key": self.key,
        }


def _log(losses: Tensor, every: int) -> list[float]:
    """The means of the losses over each ``every`` steps."""
    return [float(c.mean()) for c in losses.split(every)]


# -- the codes ---------------------------------------------------------------------


def _conf_of(run_dir: Path, device: str | None) -> RunConf:
    conf = from_dict(yaml.safe_load((run_dir / "config.yaml").read_text()))
    return conf if device is None else replace(conf, device=device)


def step_codes(
    mods: JEPAModules,
    names: Sequence[str],
    *,
    conf: RunConf,
    arm_m: bool,
    fit: Callable[[], Tensor],
    w_code: RandProjCode,
    zpca_dim: int = ZPCA_DIM,
) -> dict[str, Code]:
    """The codes ``names`` (``RUN_CODES``) of the JEPA ``mods`` as loaded:
    ``z_w`` is ``w_code``; ``z`` the modules, or of an arm M (``arm_m``)
    ``zpca``, the PCA of ``zpca_dim`` of their $z$ on ``fit()`` from
    ``conf.seed + eval.PCA_SEED`` (``jepa_eval.arm_codes``'s)."""
    codes: dict[str, Code] = {}
    for name in names:
        if name == "z_w":
            codes[name] = w_code
        elif arm_m:
            gen = torch.Generator().manual_seed(conf.seed + PCA_SEED)
            codes[name] = PCACode.fit(
                mods.mu, fit(), zpca_dim, gen=gen, device=conf.device
            )
        else:
            codes[name] = mods
    return codes


def run_codes(
    run_dir: Path,
    conf: RunConf,
    names: Sequence[str],
    fit: Callable[[], Tensor],
    zpca_dim: int = ZPCA_DIM,
) -> tuple[Callable[[int], dict[str, Code]], dict[int, Path]]:
    """The codes ``names`` (``RUN_CODES``) of the JEPA run in ``run_dir``
    (``conf`` its config) as a function of a step that loads that
    checkpoint (``step_codes``), and the checkpoints by step.

    Raises:
        ValueError: A name is not of ``RUN_CODES``.
    """
    unknown = [n for n in names if n not in RUN_CODES]
    if unknown:
        raise ValueError(f"a run's codes are of {RUN_CODES}, not {unknown}")
    settings = json.loads((run_dir / "jepa.json").read_text())
    mods = build_run_jepa(conf, settings)
    w_code = RandProjCode(mods)  # W made once for the run
    arm_m = settings.get("sigreg_on", "flat") == "mean"
    ckpts = run_ckpts(run_dir)

    def at(step: int) -> dict[str, Code]:
        ckpt.load_eval(ckpts[step], mods.modules())
        return step_codes(
            mods,
            names,
            conf=conf,
            arm_m=arm_m,
            fit=fit,
            w_code=w_code,
            zpca_dim=zpca_dim,
        )

    return at, ckpts


def inception_features(
    path: Path | None, device: str
) -> tuple[Features, dict[str, Any]]:
    """FID's Inception of the weights ``path`` (default ``weights_path()``)
    on ``device`` and its record (the path, the sha256 and torchvision's
    version)."""
    path = weights_path() if path is None else path
    net = load_fid_inception(path, device)
    record = {"path": str(path), "sha256": sha256(path)}
    return net, record | {"torchvision": version("torchvision")}


def evaluate_run(
    run_dir: Path,
    *,
    steps: Sequence[int],
    codes: Sequence[str] = RUN_CODES,
    dec_seeds: Sequence[int] = (0,),
    gconf: GenConf = GenConf(),
    root: Path | None = None,
    device: str | None = None,
    features: Features | None = None,
    inception: Path | None = None,
    spacing: Mapping[str, float] | None = None,
    zpca_dim: int = ZPCA_DIM,
    out_name: str = OUT,
) -> Path:
    """G's jobs of the ``codes`` (``RUN_CODES``) of the checkpoints of
    ``steps`` of the JEPA run in ``run_dir`` into ``run_dir / out_name``
    (module docstring; another name for a trial of other settings), on
    ``device`` (default the run's), with ``features`` (default FID's
    Inception of ``inception``). Returns the directory."""
    conf = _conf_of(run_dir, device)
    root = tours_root(conf.data.tours.data_version) if root is None else root
    record: dict[str, Any] = {}
    if features is None:
        features, record = inception_features(inception, conf.device)
    ev = GenEvaluator(
        conf,
        root,
        run_dir / out_name,
        features=features,
        gconf=gconf,
        inception=record,
        spacing=spacing,
        zpca_dim=zpca_dim,
    )
    at, ckpts = run_codes(run_dir, conf, codes, lambda: ev.fit_stacks, zpca_dim)
    missing = [s for s in steps if s not in ckpts]
    if missing:
        raise ValueError(f"no checkpoint of steps {missing} (of {list(ckpts)})")
    for step in steps:
        prefix = f"{step:06d}"
        if all(ev.done(prefix, n, j) for n in codes for j in dec_seeds):
            continue
        found = at(step)
        stamp = {"run": run_dir.name, "step": step, "t": float(ckpts[step].stem)}
        ev.jobs(found, prefix=prefix, stamp=stamp, dec_seeds=dec_seeds)
    return ev.out


def evaluate_yardsticks(
    out_dir: Path,
    *,
    codes: Sequence[str] = YARDSTICKS,
    conf: RunConf | None = None,
    v16: Path | None = None,
    v17: Path | None = None,
    m_run: Path | None = None,
    dec_seeds: Sequence[int] = (0,),
    gconf: GenConf = GenConf(),
    root: Path | None = None,
    features: Features | None = None,
    inception: Path | None = None,
    spacing: Mapping[str, float] | None = None,
    pca_components: int = PCA_COMPONENTS,
    zpca_dim: int = ZPCA_DIM,
) -> Path:
    """G's jobs of the yardsticks ``codes`` (``YARDSTICKS``; module
    docstring) of ``conf`` (default ``v16_config`` on CUDA) into ``out_dir /
    OUT``: ``v16`` and ``v17`` the VAE checkpoints, ``init_m`` of the arm M
    run ``m_run`` at its step 0. Returns the directory.

    Raises:
        ValueError: A name is not of ``YARDSTICKS``, or its checkpoint or
            run is not given.
    """
    unknown = [n for n in codes if n not in YARDSTICKS]
    if unknown:
        raise ValueError(f"the yardsticks are of {YARDSTICKS}, not {unknown}")
    given = {"v16": v16, "v17": v17, "init_m": m_run}
    lacking = [n for n in codes if n in given and given[n] is None]
    if lacking:
        raise ValueError(f"the yardsticks {lacking} need their checkpoint or run")
    if conf is None:
        d = v16_config()
        d["device"] = "cuda"
        conf = from_dict(d)
    root = tours_root(conf.data.tours.data_version) if root is None else root
    record: dict[str, Any] = {}
    if features is None:
        features, record = inception_features(inception, conf.device)
    ev = GenEvaluator(
        conf,
        root,
        out_dir / OUT,
        features=features,
        gconf=gconf,
        inception=record,
        spacing=spacing,
        pca_components=pca_components,
        zpca_dim=zpca_dim,
    )
    found: dict[str, Code] = {}
    for name in codes:
        if all(ev.done(YARD_PREFIX, name, j) for j in dec_seeds):
            continue
        gen = torch.Generator().manual_seed(conf.seed + PCA_SEED)
        if name == "color":
            found[name] = ColorCode(conf.model.patch)
        elif name == "pca":
            found[name] = PCACode.fit(
                pixels, ev.fit_stacks, pca_components, gen=gen, device=conf.device
            )
        elif name in ("v16", "v17"):
            path = given[name]
            assert path is not None
            found[name] = vae_at(path, conf.device)
        elif name == "init_m":
            assert m_run is not None
            m_conf = _conf_of(m_run, conf.device)
            mods = build_run_jepa(m_conf, json.loads((m_run / "jepa.json").read_text()))
            m_gen = torch.Generator().manual_seed(m_conf.seed + PCA_SEED)
            found[name] = PCACode.fit(
                mods.mu, ev.fit_stacks, zpca_dim, gen=m_gen, device=conf.device
            )
        else:
            found[name] = ZeroCode()
    for name, code in found.items():  # one at a time: one code's set in memory
        ev.jobs({name: code}, prefix=YARD_PREFIX, stamp={}, dec_seeds=dec_seeds)
    return ev.out


def evaluate_refs(
    out_dir: Path,
    *,
    conf: RunConf | None = None,
    gconf: GenConf = GenConf(),
    root: Path | None = None,
    features: Features | None = None,
    inception: Path | None = None,
) -> Path:
    """``refs.json`` (module docstring) into ``out_dir / OUT``. Returns its
    path."""
    if conf is None:
        d = v16_config()
        d["device"] = "cuda"
        conf = from_dict(d)
    root = tours_root(conf.data.tours.data_version) if root is None else root
    record: dict[str, Any] = {}
    if features is None:
        features, record = inception_features(inception, conf.device)
    ev = GenEvaluator(
        conf, root, out_dir / OUT, features=features, gconf=gconf, inception=record
    )
    path = ev.out / "refs.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, ev.refs() | {"settings": ev.settings()})
    return path


# -- the command line ------------------------------------------------------------


def _ints(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split(","))


def _names(text: str) -> tuple[str, ...]:
    return tuple(text.split(","))


def main(argv: Sequence[str] | None = None) -> Path:
    """``python -m exp.vision.gen_eval`` (module docstring). Returns the
    output."""
    ap = argparse.ArgumentParser(
        prog="python -m exp.vision.gen_eval",
        description="V18c's G: draw from codes by a flow decoder, judge by Inception.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="a JEPA run's checkpoints")
    run.add_argument("run_dir", type=Path)
    run.add_argument("--steps", type=_ints, required=True)
    run.add_argument("--codes", type=_names, default=RUN_CODES)
    run.add_argument("--out-name", default=OUT, help="(a trial: not OUT)")
    yard = sub.add_parser("yardsticks", help="the yardsticks")
    yard.add_argument("--out", type=Path, required=True)
    yard.add_argument("--codes", type=_names, default=YARDSTICKS)
    yard.add_argument("--v16", type=Path, help="a VAE checkpoint")
    yard.add_argument("--v17", type=Path, help="a VAE checkpoint")
    yard.add_argument("--m-run", type=Path, help="arm M's run (init_m)")
    refs = sub.add_parser("refs", help="the references of real frames")
    refs.add_argument("--out", type=Path, required=True)
    for p in (run, yard, refs):
        p.add_argument("--root", type=Path, help="the data (default: tours_root)")
        p.add_argument("--inception", type=Path, help="FID's weights")
        p.add_argument("--fit-steps", type=int, default=GenConf.fit_steps)
        p.add_argument("--fit-batch", type=int, default=GenConf.fit_batch)
        p.add_argument("--device", help="(default: the run's, or cuda)")
    for p in (run, yard):
        p.add_argument("--dec-seeds", type=_ints, default=(0,))
    args = ap.parse_args(argv)
    gconf = GenConf(fit_steps=args.fit_steps, fit_batch=args.fit_batch)
    if args.cmd == "run":
        return evaluate_run(
            args.run_dir,
            steps=args.steps,
            codes=args.codes,
            dec_seeds=args.dec_seeds,
            gconf=gconf,
            root=args.root,
            device=args.device,
            inception=args.inception,
            out_name=args.out_name,
        )
    d = v16_config()
    d["device"] = args.device or "cuda"
    conf = from_dict(d)
    if args.cmd == "yardsticks":
        return evaluate_yardsticks(
            args.out,
            codes=args.codes,
            conf=conf,
            v16=args.v16,
            v17=args.v17,
            m_run=args.m_run,
            dec_seeds=args.dec_seeds,
            gconf=gconf,
            root=args.root,
            inception=args.inception,
        )
    return evaluate_refs(
        args.out, conf=conf, gconf=gconf, root=args.root, inception=args.inception
    )


if __name__ == "__main__":
    print(main())
