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

V18d (``--time-arm`` other than ``none``, the default being V18c's
computation): a step draws ``--batch`` lags of ``iid.draw_lags`` on
``--time-lags`` (from ``seed + DELTA_SEED``) and a triple of stacks at each
(``iid.Units.draw_triplets``, the centres from ``seed + DRAW_SEED``; of
``STperm`` the targets from ``seed + PERM_SEED``), and descends
``jepa.jepa_time_terms`` of the arm (``--time-weight`` $w$ of ``ST`` and
``STperm``): the masks, the directions and SIGReg are the centres'. The time
predictor (``jepa.TimePredictor`` of ``TIME_KNOTS``, from 0) is in an
AdamW group of its own, of the peak ``--time-lr`` and no muP multiplier;
``grad_norm`` is of the other parameters, ``tpred_grad_norm`` its own. The
rows add ``time`` ($L_T$), ``time_bands`` (the mean of $L_T$ of the lags
between neighbouring knots, since the last row) and ``roles`` (the last
step's audit of the codes, ``_role_audit``); the checkpoints add ``tpred``.

Time, as V16: a checkpoint after $s$ steps has seen ``frame`` $= s \cdot$
``batch`` stacks, $t$ = ``frame`` $\cdot$ ``time.frame`` (12.8 s a step at
batch 128). The checkpoints are at the steps of ``ckpt_steps``.

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
        --time-arm ST --time-weight 0.25 --out <run_dir>
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
    TIME_ARMS,
    TIME_KNOTS,
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
    jepa_time_terms,
    mask_expansion,
    mask_family,
    rho_in,
    sigreg_points,
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
    "TIME_LAGS",
    "TIME_LR",
    "V16_RUN",
    "ckpt_steps",
    "main",
    "train",
    "v16_config",
]

MASK_SEED = 29  # added to the seed: the masks
DIRECTION_SEED = 31  # added to the seed: SIGReg's directions
DIRECTIONS = 1024  # M
DELTA_SEED = 89  # added to the seed: V18d's lags
PERM_SEED = 59  # added to the seed: S+T-perm's targets
TIME_LR = 1e-3  # the peak learning rate of V18d's time predictor
TIME_LAGS = (8, 512)  # V18d's lags, in stacks (0.8 to 51.2 s)
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


def _optimizer(
    mods: JEPAModules, conf: RunConf, lr: float, time_lr: float = TIME_LR
) -> torch.optim.AdamW:
    """AdamW without weight decay, a group per learning-rate multiplier (of
    the peak ``lr``), and V18d's ``tpred`` in a group of its own of the peak
    ``time_lr`` (no multiplier). A group's ``peak`` is its peak learning
    rate before the multiplier."""
    mult = jepa_mup_lr(mods, conf.model.width) if conf.model.param == "mup" else {}
    by: dict[float, list[torch.nn.Parameter]] = {}
    for p in _model_parameters(mods):
        by.setdefault(mult.get(p, 1.0), []).append(p)
    groups = [
        {"params": ps, "lr": lr * m, "mult": m, "peak": lr} for m, ps in by.items()
    ]
    if mods.tpred is not None:
        ps = list(mods.tpred.parameters())
        groups.append({"params": ps, "lr": time_lr, "mult": 1.0, "peak": time_lr})
    return torch.optim.AdamW(groups, lr=lr, weight_decay=0.0)


def _model_parameters(mods: JEPAModules) -> list[torch.nn.Parameter]:
    """The learned parameters but the time predictor's (V18c's)."""
    tpred = set() if mods.tpred is None else {id(p) for p in mods.tpred.parameters()}
    return [p for p in mods.parameters() if id(p) not in tpred]


def _set_lr(opt: torch.optim.Optimizer, step: int, *, warmup: int, steps: int) -> None:
    """Each group's learning rate at ``step``: ``lr_at`` of its peak, times its
    multiplier."""
    for g in opt.param_groups:
        g["lr"] = lr_at(step, lr=g["peak"], warmup=warmup, steps=steps) * g["mult"]


def _bands(n: Tensor, knots: Sequence[int]) -> Tensor:
    """The band ``(B,)`` of each lag: $j$ for ``knots[j]`` <= n < ``knots[j +
    1]``, the last band closed (and a lag outside the knots in its end's)."""
    k = torch.tensor(knots, device=n.device)
    return (torch.searchsorted(k, n, right=True) - 1).clamp(0, len(knots) - 2)


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
    time_arm: str = "none",
    time_weight: float | None = None,
    time_lr: float = TIME_LR,
    time_lags: tuple[int, int] = TIME_LAGS,
) -> None:
    """Learn ``build_jepa(conf, token_dim=token_dim, proj=proj)`` (with $W$
    of ``LATENT_DIM`` rows for ``sigreg_on`` ``"mix"``) as the module
    docstring says, into ``out``; with a ``time_arm`` other than ``"none"``,
    V18d's arm (``jepa_time_terms``; ``time_weight`` $w$ of ``"ST"`` and
    ``"STperm"``) on ``batch`` triples a step, at lags of ``draw_lags`` on
    ``time_lags`` (from ``seed + DELTA_SEED``), the time predictor learned at
    the peak ``time_lr``.

    Raises:
        ValueError: ``mask_ratio`` hides no whole number of the coarse grid's
            positions in ``[1, G^2 - 1]`` (``hidden_count``), the grid is no
            whole multiple of the coarse one (``mask_expansion``), ``lam`` is
            not in ``[0, 1]``, ``sigreg_on`` not of ``SIGREG_ON``, ``proj``
            not of ``PROJ_KINDS`` or ``time_arm`` not of ``TIME_ARMS``,
            ``time_weight`` is missing for ``"ST"`` or ``"STperm"`` or given
            for another arm, or ``time_lags`` is not ``1 <= lo <= hi``.
        FloatingPointError: A mean of ``train.jsonl`` is not finite.
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1], not {lam}")
    if sigreg_on not in SIGREG_ON:
        raise ValueError(f"sigreg_on is one of {SIGREG_ON}, not {sigreg_on!r}")
    if proj not in PROJ_KINDS:
        raise ValueError(f"proj is one of {PROJ_KINDS}, not {proj!r}")
    if time_arm not in TIME_ARMS:
        raise ValueError(f"time_arm is one of {TIME_ARMS}, not {time_arm!r}")
    if (time_weight is None) != (time_arm in ("none", "T")):
        raise ValueError(
            f"time_weight is of the arms ST and STperm only, not {time_arm!r}"
        )
    lo, hi = time_lags
    if not 1 <= lo <= hi:
        raise ValueError(f"time_lags must be 1 <= lo <= hi, not {time_lags}")
    timed = time_arm != "none"
    knots = TIME_KNOTS if timed else None
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
        "time_arm": time_arm,
        "time_weight": time_weight,
        "time_lr": time_lr if timed else None,
        "time_lags": list(time_lags) if timed else None,
        "time_knots": None if knots is None else list(knots),
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
            "delta": seed + DELTA_SEED if timed else None,
            "perm": seed + PERM_SEED if time_arm == "STperm" else None,
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
        time_knots=knots,
    )
    opt = _optimizer(mods, conf, lr, time_lr)
    params = _model_parameters(mods)
    units = Units(conf, root, "all", seed=seed + DRAW_SEED)
    settings["units"] = units.count
    (out / "jepa.json").write_text(json.dumps(settings, indent=1))
    mask_gen = torch.Generator().manual_seed(seed + MASK_SEED)
    dir_gen = torch.Generator(device=dev).manual_seed(seed + DIRECTION_SEED)
    delta_gen = torch.Generator().manual_seed(seed + DELTA_SEED)
    perm_gen = torch.Generator().manual_seed(seed + PERM_SEED)

    def draw() -> tuple[Tensor, Tensor | None]:
        """The next batch: ``batch`` stacks, or of a time arm triples and
        their lags."""
        if not timed:
            return units.draw(batch), None
        n = draw_lags(batch, lo, hi, delta_gen)
        return units.draw_triplets(n, perm_gen if time_arm == "STperm" else None), n

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

    terms_keys = ("loss", "sigreg") if time_arm == "T" else ("loss", "pred", "sigreg")
    terms_keys += ("time",) if timed else ()
    keys = (*terms_keys, "grad_norm")
    keys += ("tpred_grad_norm",) if timed else ()
    keys += ("z_var", "rho_in", "zbar_var")
    keys += ("mix_var",) if mods.mix is not None else ()
    acc = torch.zeros(len(keys), dtype=torch.float64, device=dev)
    bands = len(TIME_KNOTS) - 1
    band_sum = torch.zeros(bands, dtype=torch.float64, device=dev)
    band_count = torch.zeros(bands, dtype=torch.float64, device=dev)
    start, n_acc = time.monotonic(), 0
    log = (out / "train.jsonl").open("w")
    with ThreadPoolExecutor(1) as pool:
        nxt = pool.submit(draw)
        for step in range(steps):
            if step in stamps:
                save(step)
            x, n = nxt.result()
            x = x.to(dev, non_blocking=True)
            nxt = pool.submit(draw)
            mask = expand_masks(draw_masks(family, batch, mask_gen), s)
            dirs = directions(directions_count, d, dir_gen)
            _set_lr(opt, step, warmup=warmup, steps=steps)
            if n is None:
                terms = jepa_terms(mods, x, mask, dirs, lam, sigreg_on)
            else:
                n = n.to(dev)
                terms = jepa_time_terms(
                    mods,
                    x,
                    n,
                    mask,
                    dirs,
                    lam,
                    sigreg_on,
                    arm=time_arm,
                    weight=0.0 if time_weight is None else time_weight,
                )
            opt.zero_grad(set_to_none=True)
            terms["loss"].backward()
            norm = torch.nn.utils.clip_grad_norm_(params, math.inf)
            tokens = terms["tokens"]
            row = [terms[k].detach() for k in terms_keys]
            row += [norm.detach()]
            if mods.tpred is not None:
                tp = list(mods.tpred.parameters())
                row += [torch.nn.utils.clip_grad_norm_(tp, math.inf).detach()]
            opt.step()
            row += [tokens.flatten(1).var(0).mean(), rho_in(tokens)]
            row += [sigreg_points(tokens, "mean").var(0).mean()]
            if mods.mix is not None:
                with torch.no_grad():
                    row += [mods.mix(tokens.flatten(1)).var(0).mean()]
            acc += torch.stack([r.double() for r in row])
            band = None
            if n is not None:
                band = _bands(n, TIME_KNOTS)
                band_sum.index_add_(0, band, terms["time_err"].double())
                band_count.index_add_(0, band, torch.ones_like(band_sum[band]))
            n_acc += 1
            if (step + 1) % log_every == 0 or step + 1 == steps:
                means = (acc / n_acc).tolist()
                if not all(map(math.isfinite, means)):
                    raise FloatingPointError(f"non-finite loss at step {step}")
                rec: dict[str, Any] = {
                    "step": step + 1,
                    **dict(zip(keys, means, strict=True)),
                }
                if band is not None:
                    counts = band_count.tolist()
                    sums = band_sum.tolist()
                    rec["time_bands"] = [
                        a / c if c else None for a, c in zip(sums, counts, strict=True)
                    ]
                    rec["roles"] = _role_audit(terms["codes"], band, bands)
                    band_sum.zero_()
                    band_count.zero_()
                rec |= {
                    "lr": lr_at(step, lr=lr, warmup=warmup, steps=steps),
                    "sec": time.monotonic() - start,
                }
                if dev.type == "cuda":
                    rec["max_mem"] = torch.cuda.max_memory_allocated(dev) / 2**30
                log.write(json.dumps(rec) + "\n")
                log.flush()
                acc.zero_()
                n_acc = 0
    save(steps)
    log.close()


def _role_audit(codes: Tensor, band: Tensor, bands: int) -> dict[str, Any]:
    """Of each role ($z_-$, $z_0$, $z_+$) of the step's ``codes`` ``(B, 3,
    D)`` and each band of ``band`` ``(B,)``: ``mean_sq``, the mean over the
    coordinates of the square of their mean, and ``var``, the mean of their
    variance (``None`` of a band of fewer than 2 triples; plans/v18d.md,
    「役割と帯の監査」)."""
    audit: dict[str, Any] = {}
    for r, name in enumerate(("minus", "centre", "plus")):
        mean_sq: list[float | None] = []
        var: list[float | None] = []
        for j in range(bands):
            z = codes[band == j, r].double()  # (B_j, D)
            if len(z) < 2:
                mean_sq.append(None)
                var.append(None)
                continue
            mean_sq.append(float(z.mean(0).square().mean()))
            var.append(float(z.var(0).mean()))
        audit[name] = {"mean_sq": mean_sq, "var": var}
    return audit


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument(
        "--config", type=Path, help="a run's config.yaml (default: v16_config)"
    )
    ap.add_argument("--mask-ratio", type=float, required=True, help="r: k = r T")
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
        "--time-arm",
        choices=TIME_ARMS,
        default="none",
        help="V18d's arm (none: V18c's loss)",
    )
    ap.add_argument("--time-weight", type=float, help="w of the arms ST and STperm")
    ap.add_argument("--time-lr", type=float, default=TIME_LR, help="tpred's peak lr")
    ap.add_argument(
        "--time-lags",
        type=_lags,
        default=TIME_LAGS,
        help="lo:hi, the lags in stacks (log-uniform)",
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
        time_arm=args.time_arm,
        time_weight=args.time_weight,
        time_lr=args.time_lr,
        time_lags=args.time_lags,
    )


def _lags(text: str) -> tuple[int, int]:
    """``lo:hi`` as ``(lo, hi)``."""
    lo, hi = text.split(":")
    return int(lo), int(hi)


if __name__ == "__main__":
    main()
