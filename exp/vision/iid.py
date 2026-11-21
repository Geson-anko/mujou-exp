r"""The VAE of a stream run learned the plain way (V16): by backpropagation
and AdamW on batches of stacks drawn at random, not from the stream by
PC-ALM. How good the same model gets without the stream's constraints.

The model is ``build_vae`` of the run's config (``--config``, a
``config.yaml``; its seed, its parametrization): with ``model.param`` =
``"mup"`` each parameter's learning rate is ``--lr`` times its ``mup_lr``.
A step draws ``--batch`` stacks of the stream's units (``Units``: stream 0's
$k$-th stack, as ``eval.stacks_at``; those it marks not ready or reset drawn
again) uniformly with replacement from

- ``"span"``: the units of the run's span, $k < $ ``measure.duration`` /
  ``time.frame``: the stacks the stream run learned from;
- ``"all"``: every unit of the loop of the training videos (all but
  ``data.tours.held_out``),

and descends the mean negative ELBO of the stacks, the stream run's
objective: $-\log N(y; \hat y, \sigma^2 I) + \mathrm{KL}(q \Vert N(0, I))$
of $y = (x + u) / 256$, $z = \mu + e^{\ell/2} \odot \varepsilon$
(``exp.lib.elbo``). The learning rate rises linearly over ``--warmup``
steps, then decays by a cosine to 0 at ``--steps``; the gradient's norm is
clipped at ``--clip`` (by default not: the loss is a sum over a stack's
values, so the norm is of order 1e5 and Adam's step does not depend on it).

Time. A checkpoint after $s$ steps is stamped as if a stream had shown the
same stacks one a frame: ``frame`` $= s \cdot$ ``batch``, $t$ = ``frame``
$\cdot$ ``time.frame``. The checkpoints are at the steps whose $t$ is in
``CKPT_T`` and at the last step, so ``stream_eval`` and ``z_use`` read the
run as they read a stream run (their groups by age mean nothing here; span
and after do), and $t$ = 1800 s is the stacks a stream run sees in 1800
s.

``<out>/``: ``config.yaml`` (the run's, ``device`` set), ``iid.json`` (these
settings), ``train.jsonl`` (a row every ``--log-every`` steps: the means
since the last row of ``loss``, ``nll``, ``kl``, ``sq`` $\lVert y - \hat
y\rVert^2$ a stack and ``grad_norm``, the gradient's norm before clipping;
``log_sigma``, ``lr``, ``sec`` since the start) and ``ckpt/<t>.pt``::

    python -m exp.vision.iid --config runs/<b>/<run>/config.yaml --data span \
        --lr 3e-4 --steps 60000 --out runs/v16-iid/<name>
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
from exp.lib.elbo import dequantize, gauss_kl, gauss_nll
from exp.lib.provenance import configure_determinism
from exp.lib.tours.stacks import StackedTourFrames, tours_root
from exp.lib.tours.walking_tours import videos
from exp.lib.video_io import U8Reader
from exp.vision.compose import from_dict, to_dict
from exp.vision.models import VAEModules, build_vae, early_fusion, mup_lr
from exp.vision.schema.run import RunConf

__all__ = ["CKPT_T", "DATA", "Units", "lr_at", "main", "neg_elbo", "train"]

DATA = ("span", "all")
# the stamps (s) of the checkpoints: stream_eval.PROBE_AT, then on to 1e6
CKPT_T = (0.0, 60.0, 300.0, 600.0, 1200.0, 1800.0, 6e3, 1.8e4, 6e4, 1.8e5, 6e5, 1e6)
NOISE_SEED = 13  # added to the seed: u, eps (program.NOISE_SEED)
DRAW_SEED = 17  # added to the seed: the units drawn


class Units:
    """Draws of the stacks of stream 0 of the run ``conf`` at its units of
    ``data`` (module docstring), u8 ``(B, n, 3, S, S)`` on the CPU.

    Raises:
        ValueError: ``data`` is not of ``DATA``.
    """

    def __init__(self, conf: RunConf, root: Path, data: str, *, seed: int) -> None:
        t = conf.data.tours
        if data == "span":
            self.count = round(conf.measure.duration / conf.time.frame)
        elif data == "all":
            names = [v for v in videos(root) if v != t.held_out]
            self.count = sum(len(U8Reader(root / v)) for v in names)  # L
        else:
            raise ValueError(f"data must be one of {DATA}, not {data!r}")
        self._stream = StackedTourFrames(
            root,
            n=1,
            held_out=t.held_out,
            frames=t.frames,
            stride=t.stride,
            advance=t.advance,
        )
        self._gen = torch.Generator().manual_seed(seed)

    def draw(self, b: int) -> Tensor:
        """``b`` stacks of units drawn uniformly, with replacement."""
        out: list[Tensor] = []
        while len(out) < b:
            k = int(torch.randint(self.count, (1,), generator=self._gen))
            self._stream.load_state_dict({"k": k})
            x, ready, reset = self._stream.next()
            if ready[0] and not reset[0]:
                out.append(x[0])
        return torch.stack(out)


def neg_elbo(mods: VAEModules, x: Tensor, u: Tensor, eps: Tensor) -> dict[str, Tensor]:
    r"""The mean over the stacks ``x`` u8 ``(B, n, 3, S, S)`` of $-\log N(y;
    \hat y, \sigma^2 I)$ (``nll``), KL (``kl``), their sum (``loss``) and
    $\lVert y - \hat y \rVert^2$ (``sq``), with $y$ = ``dequantize(x, u)``
    and $z = \mu + e^{\ell/2} \odot$ ``eps`` ``(B, D)``."""
    mu, logvar = mods.posterior(x)
    z = mu + (0.5 * logvar).exp() * eps
    y_hat = mods.decoder(z)  # (B, 3n, S, S)
    y = early_fusion(dequantize(x, u))
    nll = gauss_nll(y, y_hat, mods.log_sigma).mean()
    kl = gauss_kl(mu, logvar).sum(1).mean()
    sq = (y - y_hat).square().flatten(1).sum(1).mean()
    return {"loss": nll + kl, "nll": nll, "kl": kl, "sq": sq}


def lr_at(step: int, *, lr: float, warmup: int, steps: int) -> float:
    """The learning rate of step ``step`` (from 0; module docstring)."""
    if step < warmup:
        return lr * (step + 1) / warmup
    frac = (step - warmup) / max(1, steps - warmup)
    return lr * 0.5 * (1 + math.cos(math.pi * frac))


def _optimizer(mods: VAEModules, conf: RunConf, lr: float) -> torch.optim.AdamW:
    """AdamW without weight decay, a group per learning-rate multiplier."""
    params = mods.parameters()
    mult = mup_lr(mods, conf.model) if conf.model.param == "mup" else {}
    by: dict[float, list[torch.nn.Parameter]] = {}
    for p in params:
        by.setdefault(mult.get(p, 1.0), []).append(p)
    groups = [{"params": ps, "lr": lr * m, "mult": m} for m, ps in by.items()]
    return torch.optim.AdamW(groups, lr=lr, weight_decay=0.0)


def train(
    conf: RunConf,
    out: Path,
    *,
    root: Path,
    data: str,
    lr: float,
    steps: int,
    batch: int,
    warmup: int,
    clip: float,
    log_every: int,
) -> None:
    """Learn ``build_vae(conf)`` as the module docstring says, into ``out``."""
    dev = torch.device(conf.device)
    settings: dict[str, Any] = {
        "data": data,
        "lr": lr,
        "steps": steps,
        "batch": batch,
        "warmup": warmup,
        "clip": clip,
        "optimizer": "adamw(betas=(0.9, 0.999), eps=1e-8, weight_decay=0)",
        "determinism": configure_determinism(conf.device),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(to_dict(conf), sort_keys=False))
    mods = build_vae(conf, seed=conf.seed)
    opt = _optimizer(mods, conf, lr)
    params = mods.parameters()
    units = Units(conf, root, data, seed=conf.seed + DRAW_SEED)
    settings["units"] = units.count
    (out / "iid.json").write_text(json.dumps(settings, indent=1))
    noise = torch.Generator(device=dev).manual_seed(conf.seed + NOISE_SEED)
    stamps = {
        s
        for t in CKPT_T
        if math.isclose(
            (s := round(t / conf.time.frame / batch)) * batch, t / conf.time.frame
        )
    }

    def save(step: int) -> None:
        frame = step * batch
        ckpt.save_eval(
            out,
            program="vae",
            group="iid",
            t=round(frame * conf.time.frame, 6),
            frame=frame,
            modules=mods.modules(),
        )

    acc = torch.zeros(5, dtype=torch.float64, device=dev)
    start, n = time.monotonic(), 0
    log = (out / "train.jsonl").open("w")
    with ThreadPoolExecutor(1) as pool:
        nxt = pool.submit(units.draw, batch)
        for step in range(steps):
            if step in stamps:
                save(step)
            x = nxt.result().to(dev, non_blocking=True)
            nxt = pool.submit(units.draw, batch)
            u = torch.rand(x.shape, generator=noise, device=dev)
            eps = torch.randn(len(x), mods.latent, generator=noise, device=dev)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, lr=lr, warmup=warmup, steps=steps) * g["mult"]
            terms = neg_elbo(mods, x, u, eps)
            opt.zero_grad(set_to_none=True)
            terms["loss"].backward()
            norm = torch.nn.utils.clip_grad_norm_(params, clip)
            opt.step()
            row = [terms[k].detach() for k in ("loss", "nll", "kl", "sq")]
            acc += torch.stack([*row, norm.detach()]).double()
            n += 1
            if (step + 1) % log_every == 0 or step + 1 == steps:
                means = (acc / n).tolist()
                if not all(map(math.isfinite, means)):
                    raise FloatingPointError(f"non-finite loss at step {step}")
                keys = ("loss", "nll", "kl", "sq", "grad_norm")
                rec = {"step": step + 1, **dict(zip(keys, means, strict=True))}
                rec |= {
                    "log_sigma": float(mods.log_sigma.detach()),
                    "lr": lr_at(step, lr=lr, warmup=warmup, steps=steps),
                    "sec": time.monotonic() - start,
                }
                log.write(json.dumps(rec) + "\n")
                log.flush()
                acc.zero_()
                n = 0
    save(steps)
    log.close()


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--config", type=Path, required=True, help="a run's config.yaml")
    ap.add_argument("--data", choices=DATA, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--clip", type=float, default=math.inf)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--seed", type=int, help="instead of the config's")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--root", type=Path, help="the data (default: tours_root)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    d = yaml.safe_load(args.config.read_text())
    d["device"] = args.device
    if args.seed is not None:
        d["seed"] = args.seed
    conf = from_dict(d)
    root = tours_root(conf.data.tours.data_version) if args.root is None else args.root
    train(
        conf,
        args.out,
        root=root,
        data=args.data,
        lr=args.lr,
        steps=args.steps,
        batch=args.batch,
        warmup=args.warmup,
        clip=args.clip,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
