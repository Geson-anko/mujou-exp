r"""The JEPA loss of a frame, shared by ``JEPAHead`` (PC-ALM), ``JEPABP`` and
the evaluation.

For stream $b$ with the target view's tokens $s_b$ and the context views'
predicted tokens $\hat s_{b,v}$ ($v = 1 \dots V - 1$), with the masks
$m_{b,v,t}$ of the context views,

$$
L^{\mathrm{pred}}_b = \frac{1}{V - 1} \sum_v
    \frac{\sum_t m_{b,v,t}\, \frac1D \lVert \hat s_{b,v,t} - s_{b,t} \rVert^2}
         {\sum_t m_{b,v,t}},
$$

and with the pooled $z_b$ of the target view, over the streams $M$ of the
frame that have a chunk (``valid``) out of $B$,

$$
L = \frac1B \sum_{b \in M} \left[ (1 - \lambda) L^{\mathrm{pred}}_b
    + \lambda\, S + \mu\, L^{\mathrm{rec}}_b \right],
$$

with $L^{\mathrm{rec}}_b = \frac1P \lVert g(z_b) - (x_b - m) / \sigma
\rVert^2$ the reconstruction of the target view's last frame $x_b$ ($P$
pixels, standardized by the data's $m$ and $\sigma$, the Embed's) by the
decoder $g$ of ``decoder`` (the term is absent, $\mu = 0$, without one).

$S$ the SIGReg statistic (``exp.lib.sigreg.sigreg``, along ``directions``)
of what ``sigreg_on`` names: ``"z"`` $S_z$, of the $|M|$ samples $z_M$
together; ``"tokens"`` $S_t$, of the $|M| T$ patch tokens of the target views
together, divided by $T$ (the Epps-Pulley statistic grows with the samples,
so this puts it on the scale of $S_z$); ``"both"`` $(S_z + S_t) / 2$. With all
streams valid it is $(1 - \lambda) \bar L^{\mathrm{pred}} + \lambda S$
(LeJEPA). Mind that $S$ grows with $|M|$: the same $\lambda$ is a stronger
SIGReg with more streams (as ``exp.lib.sigreg.SIGReg``). With ``sg`` the
target tokens are constants of $L^{\mathrm{pred}}$ (the SIGReg term still
reaches the target view, and so does the reconstruction: through $z$). There
is no target encoder: the target view goes through the same weights.

The loss is computed for every stream, valid or not, and the invalid ones are
weighted out (no shape depends on ``valid``, so nothing reads the device
back).
"""

import math
from collections.abc import Callable
from typing import NamedTuple

import torch
from torch import Tensor

from exp.lib.sigreg import T_POINTS

__all__ = [
    "JEPATerms",
    "Reconstruction",
    "jepa_loss",
    "pool",
    "prediction_errors",
    "split_y",
    "weighted_sigreg",
]


def pool(tokens: Tensor, *, z_pool: str, extra: int) -> Tensor:
    """$z$ ``(n, D)`` of the tokens ``(n, E + T, D)``: the class token
    (``"cls"``), the mean of the ``extra`` register tokens (``"registers"``) or
    of the patch tokens (``"mean"``)."""
    match z_pool:
        case "cls":
            return tokens[:, 0]
        case "registers":
            return tokens[:, :extra].mean(1)
        case _:
            return tokens[:, extra:].mean(1)


def split_y(
    y: Tensor, streams: int, views: int, shape: tuple[int, int, int] | None
) -> tuple[Tensor, Tensor | None]:
    """The masks of the context views ``(B, V - 1, T)`` and the target views'
    last frames ``(B, C, H, W)`` of the ``y`` ``(B V, T [+ C H W])`` of the
    ``Chunker``: with ``shape`` ``(C, H, W)`` the last ``C H W`` entries of
    each row are the frame (of view 0; zeros in the others), with None there
    is no frame."""
    y = y.view(streams, views, -1)
    if shape is None:
        return y[:, 1:], None
    t = y.shape[-1] - math.prod(shape)
    return y[:, 1:, :t], y[:, 0, t:].unflatten(1, shape)


def prediction_errors(
    tokens: Tensor,
    masks: Tensor,
    predictor: Callable[[Tensor], Tensor],
    *,
    extra: int,
    sg: bool,
) -> Tensor:
    r"""$L^{\mathrm{pred}}_b$ ``(B,)`` of the module docstring, from the tokens
    of the views ``(B, V, E + T, D)`` (view 0 the target) and the masks of the
    context views ``(B, V - 1, T)``."""
    b, v = tokens.shape[:2]
    target = tokens[:, 0, extra:]  # (B, T, D)
    if sg:
        target = target.detach()
    pred = predictor(tokens[:, 1:].flatten(0, 1)).unflatten(0, (b, v - 1))
    err = ((pred[:, :, extra:] - target[:, None]) ** 2).mean(-1)  # (B, V-1, T)
    m = masks.to(err.dtype)
    per_view = (err * m).sum(-1) / m.sum(-1).clamp(min=1)
    return per_view.mean(-1)


_T_POINTS_ON: dict[tuple[torch.device, torch.dtype], Tensor] = {}


def _t_points(x: Tensor) -> Tensor:
    """``T_POINTS`` on the device and in the dtype of ``x``, copied once: a copy
    from the host in every call could not be replayed from a CUDA graph."""
    key = (x.device, x.dtype)
    t = _T_POINTS_ON.get(key)
    if t is None:
        t = _T_POINTS_ON[key] = T_POINTS.to(x)
    return t


def weighted_sigreg(x: Tensor, weights: Tensor, directions: Tensor) -> Tensor:
    r"""``exp.lib.sigreg.sigreg(x[weights > 0], ..., directions)`` for 0/1
    ``weights`` ``(n,)`` of the samples ``x`` ``(n, D)``, without the dynamic
    shape: the Epps-Pulley statistic $n_w \int \lvert \hat\varphi_w(t) -
    e^{-t^2/2} \rvert^2 e^{-t^2/2} dt$ with the weighted empirical
    characteristic function $\hat\varphi_w = \sum_i w_i e^{i t a^\top x_i} /
    n_w$, $n_w = \sum_i w_i$, averaged over the directions $a$. 0 when no
    weight is 1; a sample of weight 0 gets a gradient of exactly 0."""
    t = _t_points(x)
    w = weights.to(x)
    n = w.sum()
    tx = (x @ directions.T).unsqueeze(-1) * t  # (n, M, T)
    wv = w.view(-1, 1, 1)
    re = (torch.cos(tx) * wv).sum(0) / n.clamp(min=1)  # (M, T)
    im = (torch.sin(tx) * wv).sum(0) / n.clamp(min=1)
    phi = torch.exp(-(t**2) / 2)
    err = ((re - phi) ** 2 + im**2) * phi
    return (n * torch.trapezoid(err, t, dim=-1)).mean()


class Reconstruction(NamedTuple):
    """What the reconstruction term of ``jepa_loss`` needs: the weight ``mu``,
    the ``decoder`` ``(n, D) -> (n, C, H, W)`` (in standardized pixels), the
    pixel constants ``mean`` and ``std`` of the data and the target views'
    last frames ``frames`` ``(B, C, H, W)`` (raw pixels)."""

    mu: float
    decoder: Callable[[Tensor], Tensor]
    mean: float
    std: float
    frames: Tensor


class JEPATerms(NamedTuple):
    """The loss $L$ and its parts of one frame (0-d tensors), and $z$."""

    loss: Tensor
    pred: Tensor  # the mean of L_pred over the valid streams (0 with none)
    sigreg: Tensor  # S
    valid: Tensor  # |M| as a float
    z: Tensor  # (B, D), of every stream
    recon: Tensor  # the mean of L_rec over the valid streams (0 with none or off)


def jepa_loss(
    tokens: Tensor,
    masks: Tensor,
    valid: Tensor,
    directions: Tensor,
    predictor: Callable[[Tensor], Tensor],
    *,
    extra: int,
    z_pool: str,
    lam: float,
    sg: bool,
    sigreg_on: str = "z",
    recon: Reconstruction | None = None,
) -> JEPATerms:
    """$L$ of the module docstring.

    Args:
        tokens: The tokens of the views, ``(B, V, E + T, D)``.
        masks: The masks of the context views, ``(B, V - 1, T)`` (bool or 0/1).
        valid: The streams that have a chunk, ``(B,)`` bool.
        directions: The unit directions of SIGReg, ``(M, D)``.
        predictor: ``(n, E + T, D) -> (n, E + T, D)``.
        extra: The tokens before the patch tokens.
        z_pool: How $z$ is pooled (``pool``).
        lam: $\\lambda$.
        sg: Stop the gradient of the prediction at the target tokens.
        sigreg_on: What SIGReg sees: ``"z"``, ``"tokens"`` or ``"both"``.
        recon: The reconstruction term (None: off).
    """
    b, _, _, d = tokens.shape
    z = pool(tokens[:, 0], z_pool=z_pool, extra=extra)
    w = valid.to(tokens.dtype)
    n = w.sum()
    pred = prediction_errors(tokens, masks, predictor, extra=extra, sg=sg)
    pred_sum = torch.where(valid, pred, 0.0).sum()
    parts = []
    if sigreg_on in ("z", "both"):
        parts.append(weighted_sigreg(z, w, directions))
    if sigreg_on in ("tokens", "both"):
        patches = tokens[:, 0, extra:]  # (B, T, D)
        t = patches.shape[1]
        s_t = weighted_sigreg(
            patches.reshape(-1, d), w.repeat_interleave(t), directions
        )
        parts.append(s_t / t)
    if not parts:
        raise ValueError(
            f"sigreg_on must be 'z', 'tokens' or 'both', not {sigreg_on!r}"
        )
    s = torch.stack(parts).mean()
    total = (1 - lam) * pred_sum + lam * n * s
    rec = torch.zeros_like(s)
    if recon is not None:
        target = (recon.frames.to(z) - recon.mean) / recon.std
        err = ((recon.decoder(z) - target) ** 2).flatten(1).mean(1)  # (B,)
        rec_sum = torch.where(valid, err, 0.0).sum()
        total = total + recon.mu * rec_sum
        rec = rec_sum / n.clamp(min=1)
    return JEPATerms(total / b, pred_sum / n.clamp(min=1), s, n, z, rec)
