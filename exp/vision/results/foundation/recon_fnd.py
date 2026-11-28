r"""The foundation's qualitative reconstruction (``plans/foundation.md``
4.3): each code's linear readout (``fnd_guard.content_sums``'s ridge, the
same $\alpha$ chosen on ``val``) of the observation (the stack's last frame
averaged to 32 x 32) on Zurich's stacks, laid side by side as one image:
the first row the observations, then a row of each code.

``readout`` is the map of ``content_sums``; ``grid`` lays the rows out.
Written before any image::

    python -m exp.vision.results.foundation.recon_fnd --out recon.png \
        --count 8 --code pca=runs/fnd-f0-eval --code T0=runs/fnd-train-eval:L-T0-...
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor

from exp.lib.linear_dyn import ALPHA_SCALES, fit_scales
from exp.vision.fnd_dyn import load_sets

__all__ = ["grid", "main", "readout"]


def readout(
    w_train: Tensor, y_train: Tensor, w_val: Tensor, y_val: Tensor,
    scales: Sequence[float] = ALPHA_SCALES,
) -> tuple[Tensor, Tensor]:  # fmt: skip
    """The ridge ``(A, b)`` from codes to observations of the scale of the
    least squared error on ``val`` (the first on a tie): ``content_sums``'s
    choice."""
    fits = fit_scales(w_train.double(), y_train.double(), scales)
    errs = [float((w_val.double() @ a.T + b - y_val.double()).square().sum())
            for _, a, b in fits]  # fmt: skip
    _, a, b = fits[min(range(len(fits)), key=lambda i: (errs[i], i))]
    return a, b


def grid(
    rows: Sequence[Tensor], size: int, scale: int = 2, gap: int = 2
) -> Image.Image:
    """The rows of images ``(m, 3 size^2)`` in [0, 1] (clipped), each
    enlarged ``scale`` times, ``gap`` white pixels apart."""
    m = rows[0].shape[0]
    cell = size * scale
    h, w = len(rows) * (cell + gap) - gap, m * (cell + gap) - gap
    out = torch.ones(3, h, w)
    for i, r in enumerate(rows):
        img = r.float().clamp(0, 1).reshape(m, 3, size, size)
        img = img.repeat_interleave(scale, -1).repeat_interleave(scale, -2)
        for j in range(m):
            y0, x0 = i * (cell + gap), j * (cell + gap)
            out[:, y0 : y0 + cell, x0 : x0 + cell] = img[j]
    arr = (out * 255).round().byte().permute(1, 2, 0).numpy()
    return Image.fromarray(arr)


def main(argv: Sequence[str] | None = None) -> Path:
    p = argparse.ArgumentParser(
        prog="python -m exp.vision.results.foundation.recon_fnd"
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--count", type=int, default=8)
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--code", action="append", required=True,
                   help="LABEL=DIR[:NAME] (NAME defaults to LABEL)")  # fmt: skip
    a = p.parse_args(argv)
    rows: list[Tensor] = []
    labels: list[str] = []
    for spec in a.code:
        label, _, rest = spec.partition("=")
        root, _, name = rest.partition(":")
        sets, _ = load_sets(Path(root) / "codes" / f"{name or label}.pt")
        obs = torch.load(Path(root) / "obs.pt")
        a_, b_ = readout(sets["train"].w, obs["train"].float(), sets["val"].w,
                         obs["val"].float())  # fmt: skip
        y = obs["zurich"].float()
        pick = torch.linspace(0, len(y) - 1, a.count).round().long()
        if not rows:
            rows.append(y[pick])
            labels.append("obs")
        rows.append((sets["zurich"].w[pick].double() @ a_.T + b_).float())
        labels.append(label)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    grid(rows, a.size).save(a.out)
    a.out.with_suffix(".txt").write_text("\n".join(labels) + "\n")
    return a.out


if __name__ == "__main__":
    main()
