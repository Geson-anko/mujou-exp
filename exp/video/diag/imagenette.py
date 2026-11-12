r"""DIAGNOSIS ONLY (M17, 2026-11-12): the sanity benchmark of the recipe
(``memory/agents/main/m17-recipe.md``, section 3), LeJEPA's minimal setting on
Imagenette with labels to read, to tell a fault of our parts (the ViT, SIGReg,
the optimizer) from one of the objective on Walking Tours. Not a program of
the runs, and nothing of the runs imports it.

- ``prep``: the JPEGs of Imagenette (``imagenette2-160``, from
  ``https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz``,
  extracted: ``<src>/{train,val}/<class>/*.JPEG``) to ``<out>/{train,val}.pt``,
  each ``{"x": u8 (n, 3, S, S), "y": int64 (n,), "classes": [...]}``: the
  shorter side resized to ``S`` (128) and the centre cropped, by ffmpeg.
- ``train``: our ViT of M17 (``vit_m17``, ``frames`` 1: 64 tokens of 16 x 16
  patches) learned by BP on the train split, ``--steps`` steps of ``--batch``
  images, by ``StepAdamW`` (linear warm-up, cosine), with one of
  - ``lejepa`` (stage S2): ``--views`` views of each image, each a random
    resized crop (scale 0.08 to 1, aspect 3/4 to 4/3) flipped at random with
    LeJEPA's colour jitter (``augment``), embedded $u_v$ = projector(mean of
    the tokens); $L = (1 - \lambda) \frac1V \sum_v \frac1d \lVert \bar u -
    u_v \rVert^2 + \lambda \frac1V \sum_v S(\{u_{n,v}\}_n)$, $\bar u$ the mean
    over the views (LeJEPA's minimal, all views global);
  - ``masked`` (stage S3): the packed layout of M17 on the image (no
    augmentation): V-JEPA's two context views of the whole image, with the
    loss ``jepa.loss.jepa_loss_packed`` of ``--objective``: ``tokens`` (the
    masked tokens' prediction, SIGReg of $z$ and of the tokens less their
    image's mean) or ``z`` (the completion of $u_g$ from each view's $u_v$,
    SIGReg of $u$ per view), never with an EMA.

  Every ``--every`` steps and at the start (the untrained ViT) it writes a
  line of ``<out>/log.jsonl``: the loss and the linear probe's accuracy on
  val (``probe``: ridge regression of the one-hot labels from the
  standardized mean of the tokens of the centre view, its penalty chosen on
  a fifth of train held out; argmax).

Run: ``uv run python -m exp.video.diag.imagenette prep --src DIR --out DIR``,
then ``... train --data DIR --out DIR [--mode lejepa|masked] [--device cuda]``.
The pass mark of the recipe: the probe gains at least 30 points over the
untrained ViT.
"""

import argparse
import json
import math
import subprocess
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imageio_ffmpeg
import torch
from torch import Tensor, nn

from exp.lib.optim import StepAdamW
from exp.video.compose import compose
from exp.video.jepa.loss import jepa_loss_packed, weighted_sigreg
from exp.video.jepa.masks import multiblock_masks, pack_views, spatial_keep
from exp.video.jepa.nodes import draw_directions, draw_packed_directions
from exp.video.jepa.vit import Projector, VisionModules, build_vision
from exp.video.schema.run import RunConf

__all__ = [
    "RIDGES",
    "SIDE",
    "augment",
    "decode_jpeg",
    "lejepa_loss",
    "main",
    "probe_accuracy",
    "vision_conf",
]

SIDE = 128
RIDGES = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)  # of the probe, times n


def decode_jpeg(path: Path, side: int = SIDE) -> Tensor:
    """The image at ``path`` as u8 ``(3, side, side)``: its shorter side
    resized to ``side``, the centre cropped (ffmpeg)."""
    vf = f"scale={side}:{side}:force_original_aspect_ratio=increase,crop={side}:{side}"
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error", "-i"]
    cmd += [str(path), "-vf", vf, "-frames:v", "1", "-f", "rawvideo"]
    cmd += ["-pix_fmt", "rgb24", "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    if len(out) != 3 * side * side:
        raise RuntimeError(f"ffmpeg gave {len(out)} bytes of {path}")
    hwc = torch.frombuffer(bytearray(out), dtype=torch.uint8)
    return hwc.view(side, side, 3).permute(2, 0, 1).contiguous()


def prep(src: Path, out: Path, *, side: int = SIDE, workers: int = 8) -> None:
    """``<out>/{train,val}.pt`` of the extracted Imagenette at ``src``."""
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        classes = sorted(p.name for p in (src / split).iterdir() if p.is_dir())
        files = [
            (f, c)
            for c, name in enumerate(classes)
            for f in sorted((src / split / name).iterdir())
            if f.suffix.lower() in (".jpeg", ".jpg", ".png")
        ]
        with ThreadPoolExecutor(workers) as pool:
            images = list(pool.map(lambda fc: decode_jpeg(fc[0], side), files))
        x = torch.stack(images)
        y = torch.tensor([c for _, c in files])
        torch.save({"x": x, "y": y, "classes": classes}, out / f"{split}.pt")
        print(json.dumps({"split": split, "images": len(x), "classes": classes}))


def vision_conf(overrides: Sequence[str] = ()) -> RunConf:
    """The M17 preset (``m17/wt_n16``) cut to an image: ``frames`` 1 (64
    tokens of 16 x 16 patches), the pixels of Imagenette, then
    ``overrides``."""
    return compose(
        [
            "+experiment=m17/wt_n16",
            "learner=bp",
            "vision.frames=1",
            "vision.tubelet=1",
            "vision.pixel_mean=0.45",
            "vision.pixel_std=0.27",
            *overrides,
        ]
    )


def augment(x: Tensor, g: torch.Generator, *, side: int = SIDE) -> Tensor:
    """One view of each image of ``x`` (float ``(n, 3, H, W)`` in ``[0, 1]``,
    on the generator's device): a random resized crop (area 0.08 to 1 of the
    image, aspect 3/4 to 4/3) to ``side``, a horizontal flip of probability
    1/2, LeJEPA's colour jitter (brightness, contrast and saturation by
    factors in [0.6, 1.4], [0.6, 1.4], [0.8, 1.2]) of probability 0.8 and a
    grey scale of probability 0.2. In ``[0, 1]``."""
    n, _, h, w = x.shape
    dev = x.device

    def rand(*shape: int) -> Tensor:
        return torch.rand(*shape, generator=g, device=dev)

    area = 0.08 + 0.92 * rand(n)
    aspect = torch.exp(math.log(3 / 4) + math.log(16 / 9) * rand(n))
    cw = torch.sqrt(area * aspect).clamp(max=1.0)  # of the width
    ch = torch.sqrt(area / aspect).clamp(max=1.0)
    cx = (rand(n) * 2 - 1) * (1 - cw)  # centres in [-1, 1]
    cy = (rand(n) * 2 - 1) * (1 - ch)
    flip = torch.where(rand(n) < 0.5, -1.0, 1.0)
    theta = torch.zeros(n, 2, 3, device=dev)
    theta[:, 0, 0], theta[:, 0, 2] = cw * flip, cx
    theta[:, 1, 1], theta[:, 1, 2] = ch, cy
    grid = nn.functional.affine_grid(theta, [n, 3, side, side], align_corners=False)
    v = nn.functional.grid_sample(x, grid, mode="bilinear", align_corners=False)
    jitter = (rand(n) < 0.8).view(n, 1, 1, 1)
    b = (0.6 + 0.8 * rand(n)).view(n, 1, 1, 1)
    c = (0.6 + 0.8 * rand(n)).view(n, 1, 1, 1)
    s = (0.8 + 0.4 * rand(n)).view(n, 1, 1, 1)
    j = v * b
    j = (j - j.mean((1, 2, 3), keepdim=True)) * c + j.mean((1, 2, 3), keepdim=True)
    jg = (0.299 * j[:, :1] + 0.587 * j[:, 1:2] + 0.114 * j[:, 2:]).expand_as(j)
    j = jg + (j - jg) * s
    v = torch.where(jitter, j, v)
    grey = (0.299 * v[:, :1] + 0.587 * v[:, 1:2] + 0.114 * v[:, 2:]).expand_as(v)
    v = torch.where((rand(n) < 0.2).view(n, 1, 1, 1), grey, v)
    return v.clamp(0, 1)


def lejepa_loss(u: Tensor, directions: Tensor, lam: float) -> tuple[Tensor, Tensor]:
    r"""LeJEPA's loss of the embeddings ``u`` ``(V, n, d)`` of ``V`` views of
    ``n`` images (module docstring) and its SIGReg term, along
    ``directions`` ``(M, d)``."""
    centre = u.mean(0, keepdim=True)
    pred = ((centre - u) ** 2).mean()
    ones = torch.ones(u.shape[1], device=u.device, dtype=u.dtype)
    s = torch.stack([weighted_sigreg(v, ones, directions) for v in u]).mean()
    return (1 - lam) * pred + lam * s, s


def _features(mods: VisionModules, x: Tensor, batch: int, device: str) -> Tensor:
    """The mean of the tokens ``(n, D)`` of the u8 images ``x`` ``(n, 3, S,
    S)``."""
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xi = x[i : i + batch].to(device)[:, None]  # (b, 1, 3, S, S)
            out.append(mods.tokens(xi).mean(1).double().cpu())
    return torch.cat(out)


def probe_accuracy(
    f_train: Tensor, y_train: Tensor, f_val: Tensor, y_val: Tensor, classes: int
) -> float:
    """The accuracy on val of the ridge regression of the one-hot labels from
    the standardized features (with a bias), its penalty (``RIDGES`` times
    the samples) chosen by the accuracy on the last fifth of train, then fit
    on all of train."""
    mu, sd = f_train.mean(0), f_train.std(0).clamp(min=1e-8)

    def design(f: Tensor) -> Tensor:
        z = (f.double() - mu) / sd
        return torch.cat([z, torch.ones(len(z), 1, dtype=z.dtype)], 1)

    def fit(x: Tensor, y: Tensor, alpha: float) -> Tensor:
        t = nn.functional.one_hot(y, classes).double()
        a = x.T @ x + alpha * len(x) * torch.eye(x.shape[1], dtype=x.dtype)
        return torch.linalg.solve(a, x.T @ t)

    x, xv = design(f_train), design(f_val)
    cut = len(x) * 4 // 5
    perm = torch.randperm(len(x), generator=torch.Generator().manual_seed(0))
    fit_i, hold_i = perm[:cut], perm[cut:]
    scores = []
    for alpha in RIDGES:
        w = fit(x[fit_i], y_train[fit_i], alpha)
        acc = (x[hold_i] @ w).argmax(1).eq(y_train[hold_i]).double().mean()
        scores.append(float(acc))
    best = RIDGES[max(range(len(RIDGES)), key=lambda i: scores[i])]
    w = fit(x, y_train, best)
    return float((xv @ w).argmax(1).eq(y_val).double().mean())


def _masked_loss(
    mods: VisionModules, conf: RunConf, x: Tensor, g: torch.Generator
) -> tuple[Tensor, Tensor]:
    """The loss of the packed layout (module docstring, ``masked``) of the
    images ``x`` float ``(n, 3, S, S)`` and its SIGReg term."""
    v = conf.vision
    assert v is not None
    n = len(x)
    keeps = [spatial_keep(mods.grid, m) for m in v.context_masks]
    masks = [
        multiblock_masks(n, mods.grid, m, keep=k, generator=g)
        for m, k in zip(v.context_masks, keeps, strict=True)
    ]
    h, y = pack_views(x[:, None], masks, keeps=keeps, patch=v.patch, tubelet=1)
    dz, dt = draw_packed_directions(
        v.sigreg_directions, v.z_width, v.dim, tokens=True, generator=g
    )
    du = draw_directions(v.sigreg_directions, v.u_width, g)
    t = jepa_loss_packed(
        mods.encoder(h, mods.segments),
        y,
        torch.ones(n, dtype=torch.bool, device=x.device),
        dz,
        dt,
        mods.predictor,
        keeps=list(mods.segments[1:]),
        proj=mods.proj,
        lam=v.sigreg,
        sg=False,
        sigreg_on=v.sigreg_on,
        pred_loss=v.pred_loss,
        token_sigreg=v.token_sigreg,
        token_weight=v.token_weight,
        z_weight=v.z_weight,
        projector=mods.projector,
        zpred=mods.zpred,
        directions_u=du,
    )
    return t.loss, t.sigreg


_OBJECTIVES = {
    "tokens": ["vision.sigreg_on=both", "vision.token_sigreg=clip"],
    "z": [
        "vision.token_weight=0.0",
        "vision.z_weight=1.0",
        "vision.projector=[512,512,64]",
        "vision.sigreg_on=views",
    ],
}


def train(args: argparse.Namespace) -> None:
    """The ``train`` command (module docstring)."""
    dev = args.device
    torch.manual_seed(args.seed)
    data = {s: torch.load(args.data / f"{s}.pt") for s in ("train", "val")}
    classes = len(data["train"]["classes"])
    extra = _OBJECTIVES[args.objective] if args.mode == "masked" else []
    conf = vision_conf([f"seed={args.seed}", f"device={dev}", *extra, *args.set])
    v = conf.vision
    assert v is not None
    mods = build_vision(conf, seed=args.seed)
    params = [*mods.encoder.parameters()]
    head: Projector | None = None
    if args.mode == "lejepa":
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(args.seed + 1)
            head = Projector(v.dim, args.projector).to(dev)
        params += head.parameters()
    else:
        params += mods.predictor.parameters()
        for m in (mods.proj, mods.projector, mods.zpred):
            if m is not None:
                params += m.parameters()
    opt = StepAdamW(
        lr=args.lr,
        weight_decay=args.wd,
        warmup_steps=args.warmup,
        decay="cosine",
        decay_steps=args.steps,
        min_lr=args.min_lr,
    ).bind(params)
    g = torch.Generator(device=dev).manual_seed(args.seed + 2)
    gi = torch.Generator().manual_seed(args.seed + 3)
    x_train, y_train = data["train"]["x"], data["train"]["y"]
    args.out.mkdir(parents=True, exist_ok=True)
    log = (args.out / "log.jsonl").open("a")

    def record(step: int, loss: float, sig: float, t0: float) -> None:
        f_tr = _features(mods, x_train, 256, dev)
        f_va = _features(mods, data["val"]["x"], 256, dev)
        acc = probe_accuracy(f_tr, y_train, f_va, data["val"]["y"], classes)
        line = {
            "step": step,
            "loss": loss,
            "sigreg": sig,
            "probe": acc,
            "wall": round(time.time() - t0, 1),
            "mode": args.mode,
            "objective": args.objective if args.mode == "masked" else None,
        }
        log.write(json.dumps(line) + "\n")
        log.flush()
        print(json.dumps(line), flush=True)

    t0 = time.time()
    record(0, math.nan, math.nan, t0)
    sums = [0.0, 0.0, 0]
    for step in range(1, args.steps + 1):
        idx = torch.randint(len(x_train), (args.batch,), generator=gi)
        x = x_train[idx].to(dev).float() / 255
        with torch.enable_grad():
            if head is not None:
                views = torch.stack([augment(x, g) for _ in range(args.views)])
                tok = mods.tokens(views.flatten(0, 1)[:, None])  # (V n, T, D)
                u = head(tok.mean(1)).unflatten(0, (args.views, args.batch))
                dirs = draw_directions(v.sigreg_directions, u.shape[-1], g)
                loss, sig = lejepa_loss(u, dirs, args.lam)
            else:
                loss, sig = _masked_loss(mods, conf, x, g)
            grads = torch.autograd.grad(loss, params, materialize_grads=True)
        opt.step_(list(grads), 1.0)
        sums[0] += float(loss.detach())
        sums[1] += float(sig.detach())
        sums[2] += 1
        if step % args.every == 0 or step == args.steps:
            record(step, sums[0] / sums[2], sums[1] / sums[2], t0)
            sums = [0.0, 0.0, 0]
    log.close()


def main(argv: Sequence[str] | None = None) -> None:
    """The command line (module docstring)."""
    p = argparse.ArgumentParser(description="The sanity benchmark (diagnosis).")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prep")
    a.add_argument("--src", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--workers", type=int, default=8)
    t = sub.add_parser("train")
    t.add_argument("--data", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--mode", choices=("lejepa", "masked"), default="lejepa")
    t.add_argument("--objective", choices=tuple(_OBJECTIVES), default="tokens")
    t.add_argument("--device", default="cuda")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--steps", type=int, default=20000)  # ~ 540 epochs of 9469
    t.add_argument("--batch", type=int, default=256)
    t.add_argument("--views", type=int, default=4)
    t.add_argument("--lam", type=float, default=0.02)
    t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--min_lr", type=float, default=1e-3)  # of lr
    t.add_argument("--wd", type=float, default=0.05)
    t.add_argument("--warmup", type=int, default=400)
    t.add_argument("--every", type=int, default=1000)
    t.add_argument("--projector", type=int, nargs="+", default=[2048, 2048, 16])
    t.add_argument("--set", nargs="*", default=[], help="config overrides")
    args = p.parse_args(argv)
    if args.cmd == "prep":
        prep(args.src, args.out, workers=args.workers)
    else:
        train(args)


if __name__ == "__main__":
    main()
