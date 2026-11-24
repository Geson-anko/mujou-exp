r"""V18, V18b and V18c: the JEPA (``exp.vision.jepa``) learned by
backpropagation and AdamW on batches of stacks drawn at random
(``plans/jepa.md``, 「学習」 of V18, V18b and V18c), the JEPA's counterpart of
V16 (``exp.vision.iid``).

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

from exp.lib import ckpt
from exp.lib.provenance import configure_determinism, of_diagnosis
from exp.lib.tours.stacks import tours_root
from exp.vision.compose import from_dict, to_dict
from exp.vision.iid import DRAW_SEED, Units, lr_at
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
)
from exp.vision.schema import ModelConf, OptimConf, RunConf
from exp.vision.schema.model import HEAD_DIM

__all__ = [
    "DIRECTIONS",
    "DIRECTION_SEED",
    "LONG_CKPT",
    "MASK_SEED",
    "RUNS",
    "SHORT_CKPT",
    "V16_RUN",
    "ckpt_steps",
    "main",
    "train",
    "v16_config",
]

MASK_SEED = 29  # added to the seed: the masks
DIRECTION_SEED = 31  # added to the seed: SIGReg's directions
DIRECTIONS = 1024  # M
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
) -> None:
    """Learn ``build_jepa(conf, token_dim=token_dim, proj=proj)`` (with $W$
    of ``LATENT_DIM`` rows for ``sigreg_on`` ``"mix"``) as the module
    docstring says, into ``out``.

    Raises:
        ValueError: ``mask_ratio`` hides no whole number of the coarse grid's
            positions in ``[1, G^2 - 1]`` (``hidden_count``), the grid is no
            whole multiple of the coarse one (``mask_expansion``), ``lam`` is
            not in ``[0, 1]``, ``sigreg_on`` not of ``SIGREG_ON`` or ``proj``
            not of ``PROJ_KINDS``.
        FloatingPointError: A mean of ``train.jsonl`` is not finite.
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1], not {lam}")
    if sigreg_on not in SIGREG_ON:
        raise ValueError(f"sigreg_on is one of {SIGREG_ON}, not {sigreg_on!r}")
    if proj not in PROJ_KINDS:
        raise ValueError(f"proj is one of {PROJ_KINDS}, not {proj!r}")
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
    start, n = time.monotonic(), 0
    log = (out / "train.jsonl").open("w")
    with ThreadPoolExecutor(1) as pool:
        nxt = pool.submit(units.draw, batch)
        for step in range(steps):
            if step in stamps:
                save(step)
            x = nxt.result().to(dev, non_blocking=True)
            nxt = pool.submit(units.draw, batch)
            mask = expand_masks(draw_masks(family, batch, mask_gen), s)
            dirs = directions(directions_count, d, dir_gen)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, lr=lr, warmup=warmup, steps=steps) * g["mult"]
            terms = jepa_terms(mods, x, mask, dirs, lam, sigreg_on)
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
    )


if __name__ == "__main__":
    main()
