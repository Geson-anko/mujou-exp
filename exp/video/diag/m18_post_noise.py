r"""DIAGNOSIS ONLY (M18 stage 4): how much of a difference of the posthoc
``recon_blur`` between two codes is the posthoc decoder's own fit?

The posthoc (``measure.encoder_eval``) learns a decoder of the last frame
from the code by 2000 steps of Adam (``posthoc.fit_decoder``, from the
generator of ``seed + DECODER_SEED``) and measures its ``recon_blur`` on
val. In stage 4 the codes of PC-ALM and of BP at the same seed agree to
1e-5 or better by ``dyn_1/copy_1`` and ``stack_mse``, yet their
``recon_blur`` differ by up to 0.014. This measures, at given checkpoints,

- ``diffs``: per pair of checkpoints, the difference of their codes on the
  training stacks and on val's first stacks (``code_diff``);
- ``fits``: per checkpoint and per decoder seed (offset ``k``, the
  generator of ``seed + DECODER_SEED + SEED_STEP k``; ``k`` = 0 is the
  posthoc's own), ``recon``, ``recon_blur`` and ``recon_fit`` as the
  posthoc computes them, so that ``k`` = 0 reproduces the posthoc's lines
  (on the same device type) and the other seeds give the fit's spread on
  one code;
- ``repeat``: the first checkpoint's fit at ``k`` = 0 once more (is the fit
  deterministic?);
- ``jitter``: the first checkpoint's code plus Gaussian noise of RMS ``rel``
  times the code's RMS std (``jittered``, the same on the training and val
  codes, from the generator of ``JITTER_SEED``), fit at ``k`` = 0 (is the
  fit sensitive to a difference of the codes of the size measured?).

The numerics are the runs' (``provenance.configure_determinism``: no TF32
in matmuls and cuDNN, the deterministic algorithms), as the posthoc's; a
first version left cuDNN's TF32 and the nondeterministic algorithms on, and
its ``repeat`` differed from its own first fit. Needs the data and the
checkpoints (run it on a host). Run from the
repository root: ``python -m exp.video.diag.m18_post_noise --out F --ckpt
NAME=PATH [--ckpt ...] [--seeds K] [--jitter REL ...]
+experiment=m18/vae_t128 seed=S device=cuda``.
"""

import argparse
import json
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from exp.lib import ckpt
from exp.lib.blur_error import blur_recon
from exp.lib.provenance import configure_determinism
from exp.video.compose import compose
from exp.video.measure.encoder_eval import (
    DECODER_LR,
    EncoderEval,
    _Code,  # pyright: ignore[reportPrivateUsage]
    _Standardized,  # pyright: ignore[reportPrivateUsage]
)
from exp.video.measure.posthoc import DECODER_SEED, as_float, fit_decoder
from exp.video.programs.vision import tours_root
from exp.video.vae.modules import AEModules, VAEModules, build_modules

__all__ = ["code_diff", "jittered"]

SEED_STEP = 1000  # between the decoder seeds of the fits
JITTER_SEED = 7


def code_diff(a: Tensor, b: Tensor) -> dict[str, float]:
    r"""Of codes ``a`` and ``b`` ``(n, D)`` of the same stacks:
    $\lVert a - b\rVert / \lVert b\rVert$ (``rel``), the same over the
    spread of ``b`` about its mean code (``rel_spread``, the scale the
    posthoc decoder sees), and the largest $|a - b|$ over the RMS std of
    ``b``'s components (``max_over_sd``), in float64."""
    a, b = a.double(), b.double()
    d = (a - b).norm()
    spread = (b - b.mean(0)).norm()
    sd = b.var(0).mean().sqrt()
    return {
        "rel": float(d / b.norm()),
        "rel_spread": float(d / spread),
        "max_over_sd": float((a - b).abs().max() / sd),
    }


def jittered(z: Tensor, rel: float, sd: float, gen: torch.Generator) -> Tensor:
    """``z`` plus Gaussian noise of std ``rel * sd`` drawn from ``gen`` (a
    CPU generator; the noise is moved to ``z``'s device), in ``z``'s
    dtype."""
    noise = torch.randn(z.shape, generator=gen, dtype=torch.float64)
    return (z.double() + rel * sd * noise.to(z.device)).to(z.dtype)


def _flat_code(mods: AEModules | VAEModules) -> Callable[[Tensor], Tensor]:
    """The code the VAE's posthoc evaluates: $\\mu$ flattened, ``(n, D)``."""
    return lambda s: mods.code(s).flatten(1)


def _fit(
    ev: EncoderEval,
    seed: int,
    z_train: Tensor,
    z_val: Tensor,
    f0: Tensor,
) -> dict[str, float]:
    """``recon``, ``recon_blur`` (val) and ``recon_fit`` of a decoder fit on
    ``z_train`` ``(n, D)`` from the generator of ``seed``, as
    ``EncoderEval._evaluate`` computes them; ``z_val`` ``(m, D)`` the codes
    of val's first stacks, ``f0`` their last frames."""
    gen = torch.Generator().manual_seed(seed)
    dec = _Standardized(z_train)
    dec.dec = fit_decoder(
        dec.standardize(z_train),
        ev._targets,  # pyright: ignore[reportPrivateUsage]
        steps=ev._steps,  # pyright: ignore[reportPrivateUsage]
        lr=DECODER_LR,
        gen=gen,
    )
    mean = ev._mean  # pyright: ignore[reportPrivateUsage]
    x_hat = ev._decoded(dec, z_val)  # pyright: ignore[reportPrivateUsage]
    base = float(((mean.expand_as(f0) - f0) ** 2).mean())
    return {
        "recon": float(((x_hat - f0) ** 2).mean()) / base,
        "recon_blur": blur_recon(x_hat, f0, mean),
        "recon_fit": ev._fit_error(dec, z_train),  # pyright: ignore[reportPrivateUsage]
    }


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m exp.video.diag.m18_post_noise")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ckpt", action="append", default=[], help="NAME=PATH")
    ap.add_argument("--seeds", type=int, default=8, help="decoder seeds a code")
    ap.add_argument("--jitter", type=float, action="append", default=[])
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args(argv)
    conf = compose(a.overrides)
    assert conf.data.tours is not None and conf.vision is not None
    configure_determinism(conf.device)  # the posthoc's numerics (k = 0 reproduces it)
    root = tours_root(conf.data.tours)
    with tempfile.TemporaryDirectory() as tmp:
        ev = EncoderEval(None, conf, Path(tmp), root=root)
    windows = ev._measures.windows["val"]  # pyright: ignore[reportPrivateUsage]
    dev = torch.device(conf.device)
    f0 = as_float(windows[:, conf.vision.frames - 1].to(dev))
    codes: dict[str, tuple[Tensor, Tensor]] = {}
    for spec in a.ckpt:
        name, path = spec.split("=", 1)
        mods = build_modules(conf, seed=conf.seed)
        ckpt.load_eval(Path(path), mods.named())
        code = _Code(name, _flat_code(mods))
        z_train, _ = ev._encode(code, ev._train)  # pyright: ignore[reportPrivateUsage]
        z_val = ev._split_z(code, "val")[:, 0]  # pyright: ignore[reportPrivateUsage]
        codes[name] = (z_train.flatten(0, 1), z_val)
    names = list(codes)
    result: dict[str, Any] = {"seed": conf.seed, "diffs": {}, "fits": {}}
    for i, x in enumerate(names):
        for y in names[i + 1 :]:
            result["diffs"][f"{x}-{y}"] = {
                "train": code_diff(codes[x][0], codes[y][0]),
                "val": code_diff(codes[x][1], codes[y][1]),
            }
    print(json.dumps(result["diffs"]))
    base_seed = conf.seed + DECODER_SEED
    for name, (zt, zv) in codes.items():
        rows = []
        for k in range(a.seeds):
            r = _fit(ev, base_seed + SEED_STEP * k, zt, zv, f0) | {"k": k}
            print(name, json.dumps(r))
            rows.append(r)
        result["fits"][name] = rows
    first = names[0]
    zt, zv = codes[first]
    result["repeat"] = _fit(ev, base_seed, zt, zv, f0)
    print("repeat", json.dumps(result["repeat"]))
    sd = float(zt.double().var(0).mean().sqrt())
    result["jitter"] = []
    for rel in a.jitter:
        gen = torch.Generator().manual_seed(JITTER_SEED)
        jt, jv = jittered(zt, rel, sd, gen), jittered(zv, rel, sd, gen)
        r = _fit(ev, base_seed, jt, jv, f0) | {
            "rel": rel,
            "diff_train": code_diff(jt, zt),
            "diff_val": code_diff(jv, zv),
        }
        print("jitter", json.dumps(r))
        result["jitter"].append(r)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
