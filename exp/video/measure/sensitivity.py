r"""The sensitivity of a code to changes of its input that carry no
information (M17, 2026-11-13): does the representation amplify nuisance?

A SIGReg of the data points of one stream pushes the codes of nearby moments
apart; the cheapest differences between two stacks 0.1 s apart may be
nuisance (the quantization noise, a sub-pixel shift). For stacks $x_i$ and a
perturbation $x_i'$ of each,

$$
\mathrm{sens} = \frac{E_i \lVert f(x_i') - f(x_i) \rVert^2}
    {E_{i \ne j} \lVert f(x_i) - f(x_j) \rVert^2},
$$

the change of the code over the spread of the codes of different stacks (the
normalization of the drift of ``diag.stream_stats``: 1 when the perturbed
code is as far from the code as another stack's is). The perturbations:

- ``dither``: $x' = x + u$, $u \sim U(-\frac12, \frac12) / 255$ independent
  per pixel: the quantization error of the u8 frames, the values that the
  same u8 observation stands for;
- ``shift``: every frame translated by one pixel along the width (to the
  right), its first column replicated.

``<kind>_pix`` is the same ratio of the pixels themselves ($f$ the identity on
the stack in $[0, 1]$): a code whose ``sens`` is far above its input's
amplifies a change the input barely has. The ratio does not change under an
orthogonal map of the code (both terms are squared distances).
"""

from collections.abc import Callable

import torch
from torch import Tensor

__all__ = ["KINDS", "dither", "ratio", "sensitivities", "shift"]

KINDS = ("dither", "shift")


def dither(x: Tensor, gen: torch.Generator) -> Tensor:
    """``x`` (floats in ``[0, 1]``) plus $U(-\\frac12, \\frac12) / 255$ per
    value, drawn from ``gen`` (on the CPU, so that every device draws the
    same)."""
    u = torch.rand(x.shape, generator=gen, dtype=torch.float32) - 0.5
    return x + u.to(x) / 255


def shift(x: Tensor) -> Tensor:
    """The frames of ``x`` ``(..., H, W)`` translated by one pixel to the
    right, the first column replicated."""
    return torch.cat([x[..., :1], x[..., :-1]], -1)


def ratio(clean: Tensor, moved: Tensor) -> float:
    """$E_i \\lVert m_i - c_i \\rVert^2 / E_{i \\ne j} \\lVert c_i - c_j
    \\rVert^2$ over the rows of ``clean`` and ``moved`` ``(n, D)``, $E_{i \\ne
    j} \\lVert c_i - c_j \\rVert^2 = \\frac{2n}{n-1} E_i \\lVert c_i - \\bar c
    \\rVert^2$."""
    c, m = clean.double(), moved.double()
    n = len(c)
    apart = 2 * n / (n - 1) * float(((c - c.mean(0)) ** 2).sum(1).mean())
    return float(((m - c) ** 2).sum(1).mean()) / apart


def sensitivities(
    code: Callable[[Tensor], Tensor],
    stacks: Tensor,
    *,
    gen: torch.Generator,
    batch: int,
    device: torch.device,
) -> dict[str, float]:
    """``sens_<kind>`` of ``code`` (floats in ``[0, 1]`` ``(n, N, C, H, W)``
    on ``device`` to codes ``(n, D)``) and ``sens_<kind>_pix`` of the pixels,
    over the stacks ``stacks`` (u8, on any device), ``batch`` at a time, the
    dither drawn from ``gen`` batch by batch (module docstring)."""
    codes: dict[str, list[Tensor]] = {"clean": [], **{k: [] for k in KINDS}}
    moved: dict[str, float] = dict.fromkeys(KINDS, 0.0)  # sum |x'_i - x_i|^2
    total, square = None, 0.0  # sum_i x_i, sum_i |x_i|^2
    with torch.no_grad():
        for part in stacks.split(batch):
            x = part.to(device).float() / 255
            perturbed = {"dither": dither(x, gen), "shift": shift(x)}
            codes["clean"].append(code(x).double())
            for k, xp in perturbed.items():
                codes[k].append(code(xp).double())
                moved[k] += float(((xp.double() - x.double()) ** 2).sum())
            flat = x.flatten(1).double()
            s = flat.sum(0)
            total = s if total is None else total + s
            square += float((flat**2).sum())
    assert total is not None
    n = len(stacks)
    mean = total / n
    apart = 2 * n / (n - 1) * (square / n - float((mean**2).sum()))
    clean = torch.cat(codes["clean"])
    out = {}
    for k in KINDS:
        out[f"sens_{k}"] = ratio(clean, torch.cat(codes[k]))
        out[f"sens_{k}_pix"] = moved[k] / n / apart
    return out
