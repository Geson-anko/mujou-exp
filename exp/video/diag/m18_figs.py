r"""DIAGNOSIS ONLY (M18 stage 1): the pictures of the autoencoder's
reconstructions, untrained and trained, in one image per run.

- ``init-own``: the untrained autoencoder's own picture, as the posthoc's
  ``<t>-own.png`` draws a checkpoint's (``eval.AEEval``): the true last frames
  of the first ``SHOWN`` held-out stacks of val, then the last frame of
  $g(f(x))$ in the evaluation's units (``own_rows``). The initial weights are
  those of ``build_ae(conf, seed=conf.seed)``, as in the run. Needs the data
  (run it on a host).
- ``figure``: one image of the rows (``ROWS``) cut from the pictures the
  posthoc and the yardsticks wrote, each two rows (the true frames, then the
  decoded): the true frames, the system's decoder untrained (``init-own``)
  and at checkpoint ``t``, the decoder learned afterwards on the untrained
  ViT (the yardsticks' ``init.png``), on the mean color of each tubelet
  (``tube.png``) and on the code at ``t`` (``<t>.png``). The true rows of the
  pictures must agree (the same windows of val); it raises if they do not.
- ``stats``: of a ``figure``, per row of ``ROWS``, the chroma of the mean
  colours of the ``PATCH`` x ``PATCH`` patches (``chroma``: the mean absolute
  deviation of a colour from its grey, 0 for grey) and the correlation of the
  patches' mean brightness with the true frames'.

Run from the repository root: ``python -m exp.video.diag.m18_figs init-own
--out x.png +experiment=m18/ae_t128 seed=100 device=cuda``, ``python -m
exp.video.diag.m18_figs figure --post P --base B --init-own x.png --t 01800.0
--out y.png``, ``python -m exp.video.diag.m18_figs stats y.png``.
"""

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

import torch
from torch import Tensor

from exp.lib.png import read_png, tile, write_png
from exp.video.compose import compose
from exp.video.measure.encoder_eval import SHOWN
from exp.video.measure.posthoc import as_float
from exp.video.programs.vision import tours_root
from exp.video.vae.loss import LEVELS
from exp.video.vae.measures import held_out_stacks
from exp.video.vae.modules import build_ae

PATCH = 32  # the ViT's patch: the decoder writes a frame a patch at a time
ROWS = (
    "入力",
    "システムのデコーダ（学習前）",
    "システムのデコーダ（t）",
    "後から学ぶデコーダ：学習前の ViT",
    "後から学ぶデコーダ：tube（平均の色）",
    "後から学ぶデコーダ：符号（t）",
)


@torch.no_grad()
def own_rows(autoencoder: Callable[[Tensor], Tensor], stacks: Tensor) -> Tensor:
    r"""The true last frames and those of $g(f(x))$ ``(2, n, C, H, W)`` in
    ``[0, 1]`` of the u8 stacks ``(n, N, C, H, W)`` (``autoencoder``: $x
    \mapsto \hat y$, ``AEModules``): $\hat y \cdot 256 / 255$, clamped, as
    ``eval.AEEval`` draws them."""
    own = (autoencoder(stacks)[:, -1] * (LEVELS / 255)).clamp(0, 1)
    return torch.stack([as_float(stacks[:, -1]), own])


def picture(rows: Tensor) -> Tensor:
    """The image of frames ``(rows, cols, C, H, W)``, scaled as the posthoc
    scales its pictures."""
    return tile(rows, scale=max(1, 96 // rows.shape[-1]))


def split_rows(image: Tensor, n: int) -> list[Tensor]:
    """The ``n`` rows of frames of an image of ``n`` rows (``tile``)."""
    h = image.shape[0]
    if h % n:
        raise ValueError(f"an image of height {h} is not {n} rows")
    return list(image.split(h // n))


def figure(post: Path, base: Path, init_own: Path, t: str) -> Tensor:
    """The image of ``ROWS`` (module docstring) of the posthoc directory
    ``post`` and the yardsticks' ``base`` (``post/decode`` of each run).

    Raises:
        ValueError: The true rows of the pictures differ.
    """
    pics = [
        init_own,
        post / f"{t}-own.png",
        base / "init.png",
        base / "tube.png",
        post / f"{t}.png",
    ]
    pairs = [split_rows(read_png(p), 2) for p in pics]
    truth = pairs[0][0]
    for p, (true, _) in zip(pics, pairs, strict=True):
        if not torch.equal(true, truth):
            raise ValueError(f"the true frames of {p} are not those of {init_own}")
    return torch.cat([truth, *(decoded for _, decoded in pairs)])


def frames_of(row: Tensor, size: int) -> Tensor:
    """The frames ``(cols, size, size, 3)`` in ``[0, 1]`` of a row of
    ``tile`` at scale 1 (the 1-pixel gaps removed)."""
    cols = row.shape[1] // (size + 2)
    x = row.float().reshape(size + 2, cols, size + 2, 3)[1:-1, :, 1:-1]
    return x.permute(1, 0, 2, 3) / 255


def patch_means(frames: Tensor, patch: int = PATCH) -> Tensor:
    """The mean colour ``(n, H / p, W / p, 3)`` of each ``patch`` x ``patch``
    patch of the frames ``(n, H, W, 3)``."""
    n, h, w, c = frames.shape
    x = frames.reshape(n, h // patch, patch, w // patch, patch, c)
    return x.mean(dim=(2, 4))


def chroma(colours: Tensor) -> float:
    """The mean absolute deviation of each colour ``(..., 3)`` from its grey
    (the mean of its channels): 0 for grey."""
    return float((colours - colours.mean(-1, keepdim=True)).abs().mean())


def stats(image: Tensor, size: int) -> list[tuple[str, float, float]]:
    """Per row of ``ROWS`` of a ``figure`` of frames ``size`` x ``size``: the
    ``chroma`` of its patches' mean colours and the correlation of their
    mean brightness with the true frames'."""
    rows = [patch_means(frames_of(r, size)) for r in split_rows(image, len(ROWS))]
    truth = rows[0].mean(-1).flatten()
    out = []
    for name, m in zip(ROWS, rows, strict=True):
        r = torch.corrcoef(torch.stack([truth, m.mean(-1).flatten()]))[0, 1]
        out.append((name, chroma(m), float(r)))
    return out


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m exp.video.diag.m18_figs")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init-own", help="the untrained autoencoder's picture")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("overrides", nargs="*")
    f = sub.add_parser("figure", help="the rows of one run in one image")
    f.add_argument("--post", type=Path, required=True)
    f.add_argument("--base", type=Path, required=True)
    f.add_argument("--init-own", type=Path, required=True)
    f.add_argument("--t", default="01800.0")
    f.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("stats", help="colour and brightness of a figure's patches")
    s.add_argument("figure", type=Path)
    s.add_argument("--size", type=int, default=128)
    a = ap.parse_args(argv)
    if a.cmd == "init-own":
        conf = compose(a.overrides)
        assert conf.data.tours is not None
        stacks = held_out_stacks(conf, tours_root(conf.data.tours))["val"][:SHOWN]
        mods = build_ae(conf, seed=conf.seed)
        dev = torch.device(conf.device)
        write_png(a.out, picture(own_rows(mods, stacks.to(dev)).cpu()))
    elif a.cmd == "figure":
        write_png(a.out, figure(a.post, a.base, a.init_own, a.t))
    else:
        for name, c, r in stats(read_png(a.figure), a.size):
            print(f"{name}\tchroma {c:.4f}\tbrightness corr {r:.3f}")


if __name__ == "__main__":
    main()
