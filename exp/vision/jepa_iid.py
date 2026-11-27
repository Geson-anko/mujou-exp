r"""V18, V18b, V18c and V18d: the JEPA (``exp.vision.jepa``) learned by
backpropagation and AdamW on batches of stacks drawn at random
(``plans/jepa.md``, 「学習」 of V18, V18b and V18c; ``plans/v18d.md``), the
JEPA's counterpart of V16 (``exp.vision.iid``).

The model is ``build_jepa`` of the run's config (``--config``, a
``config.yaml``; by default V16's, ``v16_config``), its ``model.patch`` set
by ``--patch`` (V18b's 4, V18c's 8; the grid $g$ = size / patch, $T = g^2$),
$d$ = ``--token-dim``, $P$ of ``--proj`` (V18c's ``fixed`` $R$) and
``--checkpoint-blocks`` (``jepa.TokenEncoder``); with ``--sigreg-on mix``
also the fixed $W$ of ``jepa.LATENT_DIM`` rows (V18c's). A step
draws ``--batch`` stacks uniformly with replacement from every unit of the
training videos (``iid.Units`` of ``"all"``, from ``seed + iid.DRAW_SEED``),
a mask of $F_k$ of the coarse grid for each ($k$ = ``--mask-ratio`` x
$G^2$, ``jepa.mask_family``; from ``seed + MASK_SEED``) spread to the grid
$g = s G$ (``jepa.expand_masks``; $s^2 k$ positions) and ``--directions``
unit directions of $\mathbb{R}^{D_S}$ (from ``seed + DIRECTION_SEED``, the
same for the batch; $D_S = T d$ of the flat $z$, $d$ of $\bar z$ or
``LATENT_DIM`` of $W \mathrm{vec}(z)$ by ``--sigreg-on``), and descends $(1 -
\lambda) L_{\mathrm{pred}} + \lambda S$ (``jepa.jepa_terms``, $\lambda$ =
``--lam``). The defaults (the config's patch, $d$ = 64, ``learned``,
``flat``) are V18's run, the same computation. $R$ and $W$ are buffers: no
optimizer holds them and the checkpoints keep them.

AdamW ($\beta$ = 0.9, 0.999, eps 1e-8, no weight decay, no clipping), a
group per learning-rate multiplier of ``jepa_mup_lr`` (muP; all 1 in SP); the
learning rate is ``iid.lr_at`` (linear over ``--warmup`` steps, then a cosine
to 0 at ``--steps``).

V18d (``--st K``, the spatio-temporal completion; the default 0 is the
computation above): a step draws ``--batch`` clips of $K$ stacks
(``iid.Units.draw_clips``, the first stacks from ``seed + DRAW_SEED``) at
lags of ``iid.draw_lags`` on ``--st-lags`` (from ``seed + DELTA_SEED``),
with ``--st-perm`` each stack drawn again in its role's units (from ``seed +
PERM_SEED``, ST-perm), their masks of ``jepa.st_masks`` (half tubes of
$F_k$, half whole times hidden; from ``seed + MASK_SEED``), and descends
``jepa.st_terms``. ``--st 1`` is V18c's step through V18d's code, bit for
bit. The rows add ``sigreg_times`` (each time's term), ``pred_now`` (the
tubes' error), and for $K > 1$ ``pred_time`` (the time side's),
``pred_masks`` (of each row of ``jepa.time_masks``), ``pred_bands`` (the
time side's of the lags in each band of ``ST_BANDS``, ``None`` of none) and
``overlap`` (the share of clips two of whose stacks share a frame); the
z statistics are of every stack.

The foundation (``--arm``, ``plans/foundation.md`` 3; ``train_fnd``): the
arm ``T0``, ``S1`` or ``Ss`` of ``fnd_jepa.ARMS`` learns ``fnd_jepa.build_fnd``
($d$ = ``fnd_jepa.TOKEN_DIM``, the square $W$; the config's patch, 8 at 64
px) on ``--batch`` stacks of the spacings ``fnd_jepa.draw_spacings`` (on
``--s-max``, from ``seed + SPACING_SEED``) drawn by ``Units.draw_spaced``
(from ``seed + DRAW_SEED``), their masks ``fnd_jepa.draw_masks`` (from
``seed + MASK_SEED``), descending ``fnd_jepa.fnd_terms``. Its rows add
``pred_short`` and ``pred_long`` (the mean error of each kind's stacks),
``t_var`` (that of the tokens), ``time_r2`` (``fnd_jepa.time_r2`` of the
batch's tokens) and ``hidden`` (the share of the tokens hidden); ``z_var`` is
of $z$. At each checkpoint ``val.jsonl`` gets a row of
``fnd_val.val_rows`` of the validation video (``step`` and ``sec`` added).

Time, as V16: a checkpoint after $s$ steps has seen ``frame`` $= s \cdot$
``batch`` stacks, $t$ = ``frame`` $\cdot$ ``time.frame`` (12.8 s a step at
batch 128; of V18d's clips too, so a step's checkpoint has V18c's name). The checkpoints are at the steps of ``ckpt_steps``.

``<out>/``: ``config.yaml`` (the run's, ``device`` set), ``jepa.json`` (these
settings and ``provenance.of_diagnosis``'s record of the code and the host),
``train.jsonl`` (a row every ``--log-every`` steps: the means
since the last row of ``loss``, ``pred``, ``sigreg``, ``grad_norm`` and the
batch's ``z_var`` (the mean over the coordinates of the flat tokens of
their variance), ``rho_in`` (``jepa.rho_in``) and ``zbar_var`` (that of $\bar
z$), of a ``mix`` run ``mix_var`` (that of $W \mathrm{vec}(z)$); ``lr``,
``sec`` since the start, and on CUDA ``max_mem``, the peak of the memory
allocated, GiB) and
``ckpt/<t>.pt`` (``program`` ``"jepa"``; ``encoder``, ``proj``,
``predictor``, and ``mix`` of a ``mix`` run). A non-finite mean stops the run
(``FloatingPointError``)::

    python -m exp.vision.jepa_iid --mask-ratio 0.5 --lam 0.05 --lr 1e-3 \
        --steps 6000 --out runs/v18-jepa/jepa-r0.5-lam0.05-lr1e-3-s0
    python -m exp.vision.jepa_iid --patch 8 --token-dim 16 --proj fixed \
        --sigreg-on mix --mask-ratio 0.5 --lam 0.8 --lr 3e-5 --steps 6000 \
        --out <run_dir>
    python -m exp.vision.jepa_iid --patch 8 --token-dim 16 --proj fixed \
        --sigreg-on mix --mask-ratio 0.5 --lam 0.8 --lr 3e-5 --steps 6000 \
        --st 4 [--st-perm] --out <run_dir>
    python -m exp.vision.jepa_iid --patch 8 --arm Ss --lam 0.8 --lr 1e-4 \
        --steps 6000 --out <run_dir>
"""

import argparse
import json
import math
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor

from exp.lib import ckpt
from exp.lib.provenance import configure_determinism, of_diagnosis
from exp.lib.tours.stacks import tours_root
from exp.vision import fnd_jepa
from exp.vision.compose import from_dict, to_dict
from exp.vision.iid import DRAW_SEED, Units, draw_lags, lr_at
from exp.vision.jepa import (
    COARSE_GRID,
    LATENT_DIM,
    PRED_BLOCKS,
    PRED_WIDTH,
    PREDICTOR_SEED,
    PROJ_KINDS,
    PROJ_SEED,
    R_SEED,
    SIGREG_ON,
    TOKEN_DIM,
    W_SEED,
    JEPAModules,
    build_jepa,
    directions,
    draw_masks,
    expand_masks,
    hidden_count,
    jepa_mup_lr,
    jepa_terms,
    mask_expansion,
    mask_family,
    rho_in,
    sigreg_points,
    st_masks,
    st_terms,
    time_masks,
)
from exp.vision.schema import ModelConf, OptimConf, RunConf
from exp.vision.schema.model import HEAD_DIM

__all__ = [
    "DELTA_SEED",
    "DIRECTIONS",
    "DIRECTION_SEED",
    "LONG_CKPT",
    "MASK_SEED",
    "PERM_SEED",
    "RUNS",
    "SHORT_CKPT",
    "SPACING_SEED",
    "ST_BANDS",
    "ST_LAGS",
    "V16_RUN",
    "ckpt_steps",
    "main",
    "train",
    "train_fnd",
    "v16_config",
]

MASK_SEED = 29  # added to the seed: the masks
DIRECTION_SEED = 31  # added to the seed: SIGReg's directions
DIRECTIONS = 1024  # M
DELTA_SEED = 89  # added to the seed: V18d's lags
PERM_SEED = 59  # added to the seed: ST-perm's stacks
SPACING_SEED = 97  # added to the seed: the foundation's spacings
ST_LAGS = (8, 512)  # V18d's lags, in stacks (0.8 to 51.2 s)
ST_BANDS = (8, 32, 128, 512)  # the edges of the rows' bands of lags
SHORT_CKPT = (0, 1000, 2000, 4000, 6000)  # steps of the checkpoints of a short run
LONG_CKPT = (0, 2000, 6000, 10000, 20000, 30000, 40000, 50000, 60000)
RUNS = Path(__file__).resolve().parents[2] / "runs"
V16_RUN = "v0-base-b2006f0a-s0"  # V14's run whose config V16 learned


def ckpt_steps(steps: int) -> list[int]:
    """The steps of the checkpoints of a run of ``steps``: those of
    ``SHORT_CKPT`` (up to 6000 steps) or ``LONG_CKPT`` (more) up to
    ``steps``, and ``steps``."""
    plan = SHORT_CKPT if steps <= SHORT_CKPT[-1] else LONG_CKPT
    return sorted({s for s in plan if s <= steps} | {steps})


def v16_config(runs: Path = RUNS) -> dict[str, Any]:
    """The config (a dict of ``to_dict``) of ``V16_RUN``: its ``config.yaml``
    under ``runs/<batch>/`` if one is there, else the schema's defaults with
    the ViT of width 1024, muP and ``scale_decoder`` (V14's run), its
    optimizer ``rmsprop`` (muP's check; the JEPA's AdamW ignores it)."""
    found = sorted(runs.glob(f"*/{V16_RUN}/config.yaml"))
    if found:
        return yaml.safe_load(found[0].read_text())
    model = ModelConf(width=1024, scale_decoder=True, param="mup")
    conf = RunConf(
        name="v0/base", tags=["v0"], model=model, optim=OptimConf(kind="rmsprop")
    )
    return to_dict(conf)


def _optimizer(mods: JEPAModules, conf: RunConf, lr: float) -> torch.optim.AdamW:
    """AdamW without weight decay, a group per learning-rate multiplier."""
    mult = jepa_mup_lr(mods, conf.model.width) if conf.model.param == "mup" else {}
    by: dict[float, list[torch.nn.Parameter]] = {}
    for p in mods.parameters():
        by.setdefault(mult.get(p, 1.0), []).append(p)
    groups = [{"params": ps, "lr": lr * m, "mult": m} for m, ps in by.items()]
    return torch.optim.AdamW(groups, lr=lr, weight_decay=0.0)


class _SideMeans:
    """V18d's means of a row (module docstring) since the last, on ``dev``,
    of clips of ``k`` stacks (none for ``k`` = 0)."""

    def __init__(self, k: int, dev: torch.device) -> None:
        self._k = k
        masks = len(time_masks(k)) if k else 0
        self._sums = {
            "sigreg_times": torch.zeros(k, dtype=torch.float64, device=dev),
            "now": torch.zeros(2, dtype=torch.float64, device=dev),
            "time": torch.zeros(2, dtype=torch.float64, device=dev),
            "masks": torch.zeros(2, masks, dtype=torch.float64, device=dev),
            "bands": torch.zeros(2, len(ST_BANDS) - 1, dtype=torch.float64, device=dev),
            "overlap": torch.zeros(2, dtype=torch.float64, device=dev),
        }
        self._steps = 0

    def add(
        self, terms: dict[str, Tensor], kind: Tensor, n: Tensor, near: Tensor
    ) -> None:
        """A step's ``st_terms`` of the clips' ``kind`` ``(B,)``
        (``st_masks``), lags ``n`` and ``near`` (sharing a frame) ``(B,)``."""
        e = self._sums
        err = terms["clip_err"].double()
        e["sigreg_times"] += terms["sigreg_times"].double()
        tube, timed = kind < 0, kind >= 0
        e["now"] += torch.stack([err[tube].sum(), tube.sum().double()])
        e["time"] += torch.stack([err[timed].sum(), timed.sum().double()])
        one = torch.ones_like(err)
        j = kind[timed]
        e["masks"][0].index_add_(0, j, err[timed])
        e["masks"][1].index_add_(0, j, one[timed])
        edges = torch.tensor(ST_BANDS, device=n.device)
        band = (torch.searchsorted(edges, n, right=True) - 1).clamp(0, len(edges) - 2)
        e["bands"][0].index_add_(0, band[timed], err[timed])
        e["bands"][1].index_add_(0, band[timed], one[timed])
        e["overlap"] += torch.stack([near.double().sum(), one.sum()])
        self._steps += 1

    def pop(self) -> dict[str, Any]:
        """The row's means (empty for ``k`` = 0), the sums cleared."""
        if not self._k:
            return {}
        e = {name: v.tolist() for name, v in self._sums.items()}

        def mean(pair: list[float]) -> float | None:
            return pair[0] / pair[1] if pair[1] else None

        rec: dict[str, Any] = {
            "sigreg_times": [v / self._steps for v in e["sigreg_times"]],
            "pred_now": mean(e["now"]),
        }
        if self._k > 1:
            rec["pred_time"] = mean(e["time"])
            rec["pred_masks"] = [mean(list(p)) for p in zip(*e["masks"], strict=True)]
            rec["pred_bands"] = [mean(list(p)) for p in zip(*e["bands"], strict=True)]
            rec["overlap"] = mean(e["overlap"])
        for v in self._sums.values():
            v.zero_()
        self._steps = 0
        return rec


def train(
    conf: RunConf,
    out: Path,
    *,
    root: Path,
    mask_ratio: float,
    lam: float,
    lr: float,
    steps: int,
    batch: int,
    warmup: int,
    directions_count: int = DIRECTIONS,
    log_every: int = 100,
    token_dim: int = TOKEN_DIM,
    sigreg_on: str = "flat",
    checkpoint_blocks: bool = False,
    proj: str = "learned",
    st: int = 0,
    st_perm: bool = False,
    st_lags: tuple[int, int] = ST_LAGS,
) -> None:
    """Learn ``build_jepa(conf, token_dim=token_dim, proj=proj)`` (with $W$
    of ``LATENT_DIM`` rows for ``sigreg_on`` ``"mix"``) as the module
    docstring says, into ``out``; with ``st`` $K$ > 0 by V18d's clips of
    $K$ stacks at lags on ``st_lags`` (``st_perm``: ST-perm's).

    Raises:
        ValueError: ``mask_ratio`` hides no whole number of the coarse grid's
            positions in ``[1, G^2 - 1]`` (``hidden_count``), the grid is no
            whole multiple of the coarse one (``mask_expansion``), ``lam`` is
            not in ``[0, 1]``, ``sigreg_on`` not of ``SIGREG_ON`` or ``proj``
            not of ``PROJ_KINDS``, ``st`` is neither 0, 1 nor even, ``st_perm``
            is without ``st`` > 1, the batch is odd for ``st`` > 1, or
            ``st_lags`` is not ``1 <= lo <= hi``.
        FloatingPointError: A mean of ``train.jsonl`` is not finite.
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1], not {lam}")
    if sigreg_on not in SIGREG_ON:
        raise ValueError(f"sigreg_on is one of {SIGREG_ON}, not {sigreg_on!r}")
    if proj not in PROJ_KINDS:
        raise ValueError(f"proj is one of {PROJ_KINDS}, not {proj!r}")
    if st:
        time_masks(st)  # raises for an odd st > 1
    if st_perm and st < 2:
        raise ValueError("st_perm is of clips of st > 1 stacks")
    if st > 1 and batch % 2:
        raise ValueError(f"the clips of two sides are even, not {batch}")
    lo, hi = st_lags
    if not 1 <= lo <= hi:
        raise ValueError(f"st_lags must be 1 <= lo <= hi, not {st_lags}")
    dev = torch.device(conf.device)
    grid = conf.data.tours.size // conf.model.patch
    s = mask_expansion(grid)
    k = hidden_count(mask_ratio, COARSE_GRID * COARSE_GRID)
    family = mask_family(k, COARSE_GRID).to(dev)
    tokens_count = grid * grid
    latent = LATENT_DIM if sigreg_on == "mix" else None
    dims = {"flat": tokens_count * token_dim, "mean": token_dim, "mix": LATENT_DIM}
    d = dims[sigreg_on]  # D_S
    stamps = ckpt_steps(steps)
    seed = conf.seed
    rows = time_masks(st).tolist() if st else []
    hidden_times = [[i + 1 for i, h in enumerate(r) if h] for r in rows]  # from 1
    settings: dict[str, Any] = {
        "mask_ratio": mask_ratio,
        "k": k,
        "tokens": tokens_count,
        "family": len(family),
        "patch": conf.model.patch,
        "grid": grid,
        "coarse_grid": COARSE_GRID,
        "expand": s,
        "k_fine": s * s * k,
        "lam": lam,
        "token_dim": token_dim,
        "proj": proj,
        "latent_dim": latent,
        "sigreg_on": sigreg_on,
        "sigreg_dim": d,
        "checkpoint_blocks": checkpoint_blocks,
        "directions": directions_count,
        "predictor": {
            "width": PRED_WIDTH,
            "blocks": PRED_BLOCKS,
            "heads": PRED_WIDTH // HEAD_DIM,
        },
        "st": st,
        "st_perm": st_perm,
        "st_lags": list(st_lags) if st else None,
        "st_bands": list(ST_BANDS) if st > 1 else None,
        "time_masks": hidden_times if st else None,
        "lr": lr,
        "steps": steps,
        "batch": batch,
        "warmup": warmup,
        "ckpt_steps": stamps,
        "seeds": {
            "init": seed,
            "proj": seed + PROJ_SEED if proj == "learned" else None,
            "r": R_SEED if proj == "fixed" else None,
            "w": W_SEED if latent is not None else None,
            "predictor": seed + PREDICTOR_SEED,
            "draw": seed + DRAW_SEED,
            "mask": seed + MASK_SEED,
            "direction": seed + DIRECTION_SEED,
            "delta": seed + DELTA_SEED if st else None,
            "perm": seed + PERM_SEED if st_perm else None,
        },
        "optimizer": "adamw(betas=(0.9, 0.999), eps=1e-8, weight_decay=0)",
        "determinism": configure_determinism(conf.device),
    }
    # what it ran on: the snapshot's CODE.json (its mujou commit) from a
    # host's code/<snap>, the host, the command line and the environment
    prov = of_diagnosis({})
    settings["provenance"] = {k: v for k, v in prov.items() if k != "settings"}
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(to_dict(conf), sort_keys=False))
    mods = build_jepa(
        conf,
        seed=seed,
        token_dim=token_dim,
        checkpoint_blocks=checkpoint_blocks,
        proj=proj,
        latent_dim=latent,
    )
    opt = _optimizer(mods, conf, lr)
    params = mods.parameters()
    units = Units(conf, root, "all", seed=seed + DRAW_SEED)
    settings["units"] = units.count
    (out / "jepa.json").write_text(json.dumps(settings, indent=1))
    mask_gen = torch.Generator().manual_seed(seed + MASK_SEED)
    dir_gen = torch.Generator(device=dev).manual_seed(seed + DIRECTION_SEED)
    delta_gen = torch.Generator().manual_seed(seed + DELTA_SEED)
    perm_gen = torch.Generator().manual_seed(seed + PERM_SEED) if st_perm else None
    span = (conf.data.tours.frames - 1) * conf.data.tours.stride

    def draw() -> tuple[Tensor, Tensor | None, Tensor | None]:
        """The next batch: ``batch`` stacks, or V18d's clips, their lags and
        the shares of frames (``overlap``) of each."""
        if not st:
            return units.draw(batch), None, None
        n = draw_lags(batch, lo, hi, delta_gen)
        x, ends = units.draw_clips(n, st, perm_gen)
        gap = (ends[:, :, None] - ends[:, None, :]).abs()  # (B, K, K)
        near = (gap <= span) & ~torch.eye(st, dtype=torch.bool)
        return x, n, near.flatten(1).any(1)

    def save(step: int) -> None:
        frame = step * batch
        ckpt.save_eval(
            out,
            program="jepa",
            group="iid",
            t=round(frame * conf.time.frame, 6),
            frame=frame,
            modules=mods.modules(),
        )

    keys = ("loss", "pred", "sigreg", "grad_norm", "z_var", "rho_in", "zbar_var")
    keys += ("mix_var",) if mods.mix is not None else ()
    acc = torch.zeros(len(keys), dtype=torch.float64, device=dev)
    side = _SideMeans(st, dev)
    start, n = time.monotonic(), 0
    log = (out / "train.jsonl").open("w")
    with ThreadPoolExecutor(1) as pool:
        nxt = pool.submit(draw)
        for step in range(steps):
            if step in stamps:
                save(step)
            x, lags, near = nxt.result()
            x = x.to(dev, non_blocking=True)
            nxt = pool.submit(draw)
            kind = None
            if st:
                mask, kind = st_masks(family, batch, st, s, mask_gen)
            else:
                mask = expand_masks(draw_masks(family, batch, mask_gen), s)
            dirs = directions(directions_count, d, dir_gen)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, lr=lr, warmup=warmup, steps=steps) * g["mult"]
            if lags is None or near is None or kind is None:
                terms = jepa_terms(mods, x, mask, dirs, lam, sigreg_on)
            else:
                terms = st_terms(mods, x, lags.to(dev), mask, dirs, lam, sigreg_on)
                side.add(terms, kind, lags.to(dev), near.to(dev))
            opt.zero_grad(set_to_none=True)
            terms["loss"].backward()
            norm = torch.nn.utils.clip_grad_norm_(params, math.inf)
            opt.step()
            tokens = terms["tokens"]
            row = [terms[k].detach() for k in ("loss", "pred", "sigreg")]
            row += [norm.detach(), tokens.flatten(1).var(0).mean(), rho_in(tokens)]
            row += [sigreg_points(tokens, "mean").var(0).mean()]
            if mods.mix is not None:
                with torch.no_grad():
                    row += [mods.mix(tokens.flatten(1)).var(0).mean()]
            acc += torch.stack([r.double() for r in row])
            n += 1
            if (step + 1) % log_every == 0 or step + 1 == steps:
                means = (acc / n).tolist()
                if not all(map(math.isfinite, means)):
                    raise FloatingPointError(f"non-finite loss at step {step}")
                rec = {"step": step + 1, **dict(zip(keys, means, strict=True))}
                rec |= side.pop()
                rec |= {
                    "lr": lr_at(step, lr=lr, warmup=warmup, steps=steps),
                    "sec": time.monotonic() - start,
                }
                if dev.type == "cuda":
                    rec["max_mem"] = torch.cuda.max_memory_allocated(dev) / 2**30
                log.write(json.dumps(rec) + "\n")
                log.flush()
                acc.zero_()
                n = 0
    save(steps)
    log.close()


def train_fnd(
    conf: RunConf,
    out: Path,
    *,
    root: Path,
    arm: str,
    lam: float,
    lr: float,
    steps: int,
    batch: int,
    warmup: int,
    s_max: int = fnd_jepa.S_MAX,
    directions_count: int = DIRECTIONS,
    log_every: int = 100,
    checkpoint_blocks: bool = False,
    val_count: int = 512,
    held_count: int = 1024,
) -> None:
    """Learn the foundation's ``arm`` (module docstring) into ``out``: the
    files of ``train``, its rows, and ``val.jsonl`` of ``val_count`` and
    ``held_count`` stacks (``fnd_val.val_set``; ``fnd_val.VAL_COUNT`` and
    ``HELD_COUNT`` by default).

    Raises:
        ValueError: ``arm`` is not of ``fnd_jepa.ARMS``, ``lam`` is not in
            ``[0, 1]``, ``s_max`` is below 1, or as ``fnd_jepa.build_fnd``.
        FloatingPointError: A mean of ``train.jsonl`` is not finite.
    """
    # fnd_val reads jepa_eval, which imports this module: imported here
    from exp.vision import fnd_val

    if arm not in fnd_jepa.ARMS:
        raise ValueError(f"arm is one of {list(fnd_jepa.ARMS)}, not {arm!r}")
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1], not {lam}")
    if s_max < 1:
        raise ValueError(f"s_max must be at least 1, not {s_max}")
    the = fnd_jepa.ARMS[arm]
    dev = torch.device(conf.device)
    t, m = conf.data.tours, conf.model
    tubelet, token_dim = fnd_jepa.TUBELET, fnd_jepa.TOKEN_DIM
    times, grid = t.frames // tubelet, t.size // m.patch
    tokens_count = times * grid * grid
    d = tokens_count * token_dim  # D, W square
    heads = m.head_count
    stamps = ckpt_steps(steps)
    seed = conf.seed
    settings: dict[str, Any] = {
        "arm": arm,
        "timed": the.timed,
        "scaled": the.scaled,
        "s_max": s_max if the.scaled else 1,
        "tubelet": tubelet,
        "patch": m.patch,
        "times": times,
        "grid": grid,
        "tokens": tokens_count,
        "token_dim": token_dim,
        "proj": "fixed",
        "latent_dim": d,
        "sigreg_on": "mix",
        "masks": {
            "kinds": [
                {"name": k.name, "blocks": k.blocks, "scale": k.scale}
                for k in fnd_jepa.KINDS
            ],
            "aspect": list(fnd_jepa.ASPECT),
            "split": "first half short, the rest long",
        },
        "lam": lam,
        "checkpoint_blocks": checkpoint_blocks,
        "directions": directions_count,
        "predictor": {
            "width": PRED_WIDTH,
            "blocks": PRED_BLOCKS,
            "heads": PRED_WIDTH // HEAD_DIM,
        },
        "lr": lr,
        "steps": steps,
        "batch": batch,
        "warmup": warmup,
        "ckpt_steps": stamps,
        "val": {"count": val_count, "held": held_count, "seed": fnd_val.VAL_SEED},
        "seeds": {
            "init": seed,
            "embed": seed + fnd_jepa.EMBED_SEED,
            "r": R_SEED,
            "w": W_SEED,
            "predictor": seed + PREDICTOR_SEED,
            "draw": seed + DRAW_SEED,
            "spacing": seed + SPACING_SEED if the.scaled else None,
            "mask": seed + MASK_SEED,
            "direction": seed + DIRECTION_SEED,
        },
        "optimizer": "adamw(betas=(0.9, 0.999), eps=1e-8, weight_decay=0)",
        "determinism": configure_determinism(conf.device),
    }
    prov = of_diagnosis({})
    settings["provenance"] = {k: v for k, v in prov.items() if k != "settings"}
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(to_dict(conf), sort_keys=False))
    mods = fnd_jepa.build_fnd(conf, seed=seed, token_dim=token_dim, tubelet=tubelet)
    opt = _optimizer(mods, conf, lr)
    params = mods.parameters()
    units = Units(conf, root, "all", seed=seed + DRAW_SEED)
    settings["units"] = units.count
    vs = fnd_val.val_set(
        conf, root, the, s_max=s_max, times=times, count=val_count, held=held_count
    )
    settings["val"]["video"] = vs.video
    (out / "jepa.json").write_text(json.dumps(settings, indent=1))
    space_gen = torch.Generator().manual_seed(seed + SPACING_SEED)
    mask_gen = torch.Generator().manual_seed(seed + MASK_SEED)
    dir_gen = torch.Generator(device=dev).manual_seed(seed + DIRECTION_SEED)

    def draw() -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """The next batch: the spacings, the stacks, their masks and kinds."""
        s = fnd_jepa.draw_spacings(batch, the, s_max, space_gen)
        x = units.draw_spaced(s)
        hidden, kind = fnd_jepa.draw_masks(batch, times, grid, the.timed, mask_gen)
        return s, x, hidden, kind

    start = time.monotonic()
    val_log = (out / "val.jsonl").open("w")

    def save(step: int) -> None:
        frame = step * batch
        ckpt.save_eval(
            out,
            program="fnd",
            group="iid",
            t=round(frame * conf.time.frame, 6),
            frame=frame,
            modules=mods.modules(),
        )
        row = fnd_val.val_rows(mods, vs, heads=heads, times=times, scaled=the.scaled)
        rec = {"step": step, **row, "sec": time.monotonic() - start}
        val_log.write(json.dumps(rec) + "\n")
        val_log.flush()

    keys = ("loss", "pred", "sigreg", "grad_norm", "t_var", "z_var", "time_r2")
    keys += ("hidden", "pred_short", "pred_long")
    acc = torch.zeros(len(keys), dtype=torch.float64, device=dev)
    n = 0
    log = (out / "train.jsonl").open("w")
    with ThreadPoolExecutor(1) as pool:
        nxt = pool.submit(draw)
        for step in range(steps):
            if step in stamps:
                save(step)
            s, x, hidden, kind = nxt.result()
            x, hidden = x.to(dev, non_blocking=True), hidden.to(dev)
            short = (kind == 0).to(dev)
            nxt = pool.submit(draw)
            ell = s.to(dev).double().log2() if the.scaled else None
            dirs = directions(directions_count, d, dir_gen)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, lr=lr, warmup=warmup, steps=steps) * g["mult"]
            terms = fnd_jepa.fnd_terms(
                mods,
                x,
                hidden,
                dirs,
                lam,
                heads=heads,
                times=times,
                ell=ell,
                again=checkpoint_blocks,
            )
            opt.zero_grad(set_to_none=True)
            terms["loss"].backward()
            norm = torch.nn.utils.clip_grad_norm_(params, math.inf)
            opt.step()
            tok, per = terms["tokens"], terms["per"]
            row = [terms[k].detach() for k in ("loss", "pred", "sigreg")]
            row += [norm.detach(), tok.flatten(1).var(0).mean()]
            row += [terms["z"].var(0).mean(), fnd_jepa.time_r2(tok, times)]
            row += [hidden.float().mean(), per[short].mean(), per[~short].mean()]
            acc += torch.stack([r.double() for r in row])
            n += 1
            if (step + 1) % log_every == 0 or step + 1 == steps:
                means = (acc / n).tolist()
                if not all(map(math.isfinite, means)):
                    raise FloatingPointError(f"non-finite loss at step {step}")
                rec = {"step": step + 1, **dict(zip(keys, means, strict=True))}
                rec |= {
                    "lr": lr_at(step, lr=lr, warmup=warmup, steps=steps),
                    "sec": time.monotonic() - start,
                }
                if dev.type == "cuda":
                    rec["max_mem"] = torch.cuda.max_memory_allocated(dev) / 2**30
                log.write(json.dumps(rec) + "\n")
                log.flush()
                acc.zero_()
                n = 0
    save(steps)
    log.close()
    val_log.close()


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument(
        "--config", type=Path, help="a run's config.yaml (default: v16_config)"
    )
    ap.add_argument(
        "--mask-ratio", type=float, help="r: k = r T (required but of --arm)"
    )
    ap.add_argument("--lam", type=float, required=True, help="SIGReg's weight")
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--directions", type=int, default=DIRECTIONS, help="M")
    ap.add_argument("--patch", type=int, help="instead of the config's model.patch")
    ap.add_argument("--token-dim", type=int, default=TOKEN_DIM, help="d")
    ap.add_argument(
        "--sigreg-on", choices=SIGREG_ON, default="flat", help="SIGReg's data point"
    )
    ap.add_argument(
        "--proj", choices=PROJ_KINDS, default="learned", help="P: learned or fixed R"
    )
    ap.add_argument(
        "--checkpoint-blocks",
        action="store_true",
        help="compute the encoder's blocks again in the backward (less memory)",
    )
    ap.add_argument(
        "--st", type=int, default=0, help="V18d: K stacks a clip (0: V18c's step)"
    )
    ap.add_argument("--st-perm", action="store_true", help="V18d's ST-perm")
    ap.add_argument(
        "--st-lags",
        type=_lags,
        default=ST_LAGS,
        help="lo:hi, V18d's lags in stacks (log-uniform)",
    )
    ap.add_argument(
        "--arm",
        choices=list(fnd_jepa.ARMS),
        help="the foundation's arm (train_fnd; the masks are its own)",
    )
    ap.add_argument(
        "--s-max", type=int, default=fnd_jepa.S_MAX, help="Ss's largest spacing"
    )
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--seed", type=int, help="instead of the config's")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--root", type=Path, help="the data (default: tours_root)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    d = v16_config() if args.config is None else yaml.safe_load(args.config.read_text())
    d["device"] = args.device
    if args.seed is not None:
        d["seed"] = args.seed
    if args.patch is not None:
        d["model"]["patch"] = args.patch
    conf = from_dict(d)
    root = tours_root(conf.data.tours.data_version) if args.root is None else args.root
    if args.arm is not None:
        train_fnd(
            conf,
            args.out,
            root=root,
            arm=args.arm,
            lam=args.lam,
            lr=args.lr,
            steps=args.steps,
            batch=args.batch,
            warmup=args.warmup,
            s_max=args.s_max,
            directions_count=args.directions,
            log_every=args.log_every,
            checkpoint_blocks=args.checkpoint_blocks,
        )
        return
    if args.mask_ratio is None:
        ap.error("--mask-ratio is required but of --arm")
    train(
        conf,
        args.out,
        root=root,
        mask_ratio=args.mask_ratio,
        lam=args.lam,
        lr=args.lr,
        steps=args.steps,
        batch=args.batch,
        warmup=args.warmup,
        directions_count=args.directions,
        log_every=args.log_every,
        token_dim=args.token_dim,
        sigreg_on=args.sigreg_on,
        checkpoint_blocks=args.checkpoint_blocks,
        proj=args.proj,
        st=args.st,
        st_perm=args.st_perm,
        st_lags=args.st_lags,
    )


def _lags(text: str) -> tuple[int, int]:
    """``lo:hi`` as ``(lo, hi)``."""
    lo, hi = text.split(":")
    return int(lo), int(hi)


if __name__ == "__main__":
    main()
