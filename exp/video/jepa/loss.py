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

The packed layout (M17, ``jepa_loss_packed``) has the same $L$ with $\mu = 0$
(no reconstruction), each context view predicting its $M_v = T - K_v$ masked
tokens from its $K_v$ visible ones (``vit.PackedPredictor``), and the error of
a token $\ell$ either $\frac1D \lVert\cdot\rVert^2$ (``"l2"``, LeJEPA) or
$\frac1D \lVert\cdot\rVert_1$ (``"l1"``, V-JEPA):
$L^{\mathrm{pred}}_b = \frac{1}{V - 1} \sum_v \frac{1}{M_v}
\sum_{t \in \mathrm{masked}_v} \ell(\hat s_{b,v,t} - s_{b,t})$. $z_b$ is the
mean of the target tokens, through the projection when there is one.

The SIGReg of the tokens of the packed layout ($S_t$, with ``sigreg_on``
``"tokens"`` or ``"both"``) is one of ``TOKEN_SIGREG``: ``"scaled"`` (as
above, the statistic of the $|M| T$ target tokens divided by $T$), ``"raw"``
(not divided) or ``"clip"`` (not divided, of the tokens less their stack's
mean token $s_{b,t} - \frac1T \sum_{t'} s_{b,t'}$). The statistic of
Gaussian samples is $O(1)$ however many there are, and that of a constant
grows with their number, so ``"scaled"`` makes the penalty of tokens that
collapse to a constant per stack at most $S_z$: below what the prediction
gains by the collapse, which with $z$ alone is a (near) optimum of $L$ (M17
audit, 2026-11-12). ``"clip"`` sees only the spread within the stacks, which
such a collapse removes: its penalty grows as $|M| T$.

The space-time completion of the packed layout (M17, 2026-11-12; LeJEPA's
Eq. 6-9 with one global view) embeds every view: $u_{b,0} = h(z_b)$ of the
target view (the whole stack) and $u_{b,v} = h(z_{b,v})$ of each context
view, $z_{b,v}$ the mean of its own $K_v$ tokens through the projection, $h$
the ``projector`` (identity without one). A context view sees part of the
stack (outside its blocks, and only the past with a ``future``), and its
embedding is to complete that of the whole,

$$
L^{z}_b = \frac{1}{V - 1} \sum_v \frac{1}{d_u}
    \lVert u_{b,0} - q(u_{b,v}, v) \rVert^2,
$$

$q$ the ``zpred`` (identity without one). The gradient reaches $u_{b,0}$ too
(LeJEPA has no stop-gradient; ``sg`` stops it). The loss is

$$
L = \frac1B \sum_{b \in M} \left[ (1 - \lambda)
    (\beta L^{\mathrm{pred}}_b + \alpha L^{z}_b) + \lambda S \right]
$$

with $\beta$ ``token_weight`` (0: no token is predicted) and $\alpha$
``z_weight``: $\beta = 1$, $\alpha = 0$ is the loss above. ``sigreg_on``
``"views"`` makes $S$ LeJEPA's $\frac1V \sum_v S(\{u_{b,v}\}_{b \in M})$
(``directions_u``), ``"views_tokens"`` the mean of it and $S_t$. LeJEPA's
Eq. 8 also counts the global view in its mean ($\frac1V$, a term of 0): it
is this $L^z$ times $(V - 1) / V$.
"""

import math
from collections.abc import Callable, Sequence
from typing import NamedTuple

import torch
from torch import Tensor, nn

from exp.lib.sigreg import T_POINTS
from exp.video.schema.vision import TOKEN_SIGREG

__all__ = [
    "TOKEN_SIGREG",
    "JEPATerms",
    "Reconstruction",
    "jepa_loss",
    "jepa_loss_packed",
    "view_ids",
    "pool",
    "prediction_errors",
    "split_y",
    "token_sigreg_of",
    "view_embeddings",
    "weighted_sigreg",
    "z_errors",
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
    # the mean of L_z over the valid streams (packed; None or 0 when off)
    zpred: Tensor | None = None


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


def token_sigreg_of(target: Tensor, w: Tensor, directions: Tensor, kind: str) -> Tensor:
    r"""$S_t$ of the target tokens ``(B, T, D)`` of the streams of 0/1 weights
    ``w`` ``(B,)``, along ``directions`` ``(M, D)``: ``kind`` of
    ``TOKEN_SIGREG`` (module docstring)."""
    b, t, d = target.shape
    if kind not in TOKEN_SIGREG:
        raise ValueError(f"token_sigreg must be one of {TOKEN_SIGREG}, not {kind!r}")
    x = target - target.mean(1, keepdim=True) if kind == "clip" else target
    s = weighted_sigreg(x.reshape(-1, d), w.repeat_interleave(t), directions)
    return s / t if kind == "scaled" else s


def view_ids(y_view: Tensor, keep: int) -> tuple[Tensor, Tensor]:
    """The ids of the visible tokens ``(n, K)`` and of the masked ones ``(n, T
    - K)`` of a context view, each in increasing order, from its row of ``y``
    ``(n, T)`` (1 where masked) with ``keep`` = $K$ visible tokens."""
    order = y_view.sort(dim=1, stable=True).indices
    return order[:, :keep], order[:, keep:]


def view_embeddings(
    tokens: Tensor,
    keeps: Sequence[int],
    t: int,
    proj: nn.Module | None,
    projector: Callable[[Tensor], Tensor] | None,
) -> Tensor:
    r"""$u$ ``(B, V, d_u)`` of every view of the packed tokens ``(B, L, D)``
    (the target view's $T$ = ``t``, then the $K_v$ = ``keeps`` of each context
    view): the mean of the view's tokens, through ``proj`` (to $z$) and
    ``projector`` when there are; view 0 the target's ($u_g$, of $z$)."""
    pooled, at = [tokens[:, :t].mean(1)], t
    for k in keeps:
        pooled.append(tokens[:, at : at + k].mean(1))
        at += k
    e = torch.stack(pooled, 1)  # (B, V, D)
    if proj is not None:
        e = proj(e)
    return e if projector is None else projector(e)


def z_errors(u: Tensor, zpred: Callable[[Tensor, int], Tensor] | None) -> Tensor:
    r"""$rac1{d_u} \lVert u_g - q(u_v, v)
    Vert^2$ ``(B, V - 1)`` of each
        context view $v$ of $u$ ``(B, V, d_u)`` (``view_embeddings``), $q$ the
        ``zpred`` (None: $q(u_v, v) = u_v$)."""
    out = []
    for v in range(1, u.shape[1]):
        q = u[:, v] if zpred is None else zpred(u[:, v], v - 1)
        out.append(((u[:, 0] - q) ** 2).mean(1))
    return torch.stack(out, 1)


def jepa_loss_packed(
    tokens: Tensor,
    y: Tensor,
    valid: Tensor,
    directions_z: Tensor,
    directions_t: Tensor | None,
    predictor: Callable[[Tensor, Tensor, Tensor], Tensor],
    *,
    keeps: Sequence[int],
    proj: nn.Module | None,
    lam: float,
    sg: bool,
    sigreg_on: str,
    pred_loss: str,
    token_sigreg: str = "scaled",
    token_weight: float = 1.0,
    z_weight: float = 0.0,
    projector: Callable[[Tensor], Tensor] | None = None,
    zpred: Callable[[Tensor, int], Tensor] | None = None,
    directions_u: Tensor | None = None,
) -> JEPATerms:
    r"""$L$ of the packed layout (module docstring).

    Args:
        tokens: The packed tokens ``(B, L, D)``: the target view's $T$, then
            the $K_v$ of each context view (``masks.pack_views``).
        y: The masks of the context views ``(B, (V - 1) T)``, 1 where masked.
        valid: The streams that have a stack, ``(B,)`` bool.
        directions_z: The directions of SIGReg on $z$, ``(M, D_z)``.
        directions_t: Those on the target tokens, ``(M, D)`` (with
            ``sigreg_on`` ``"tokens"`` or ``"both"``).
        predictor: ``(ctx (n, K, D), ctx_ids (n, K), tgt_ids (n, M)) -> (n, M,
            D)`` (``vit.PackedPredictor``).
        keeps: $K_v$ of each context view.
        proj: The map of the pooled tokens to $z$ (None: $z$ is the mean).
        lam: $\lambda$.
        sg: Stop the gradient of the prediction at the target tokens (and of
            the completion of $u_g$ at $u_g$).
        sigreg_on: What SIGReg sees (``schema.vision.SIGREG_ON``).
        pred_loss: ``"l2"`` or ``"l1"``.
        token_sigreg: The SIGReg of the tokens (``TOKEN_SIGREG``).
        token_weight: $\beta$, the weight of the prediction of the tokens (0:
            the predictor is not called).
        z_weight: $\alpha$, the weight of the completion of $u_g$.
        projector: $u = h(z)$ (None: $u = z$).
        zpred: $q(u_v, v)$ (None: $u_v$).
        directions_u: The directions of SIGReg on $u$, ``(M, d_u)`` (with
            ``sigreg_on`` ``"views"`` or ``"views_tokens"``).
    """
    b, _, d = tokens.shape
    views = len(keeps)
    t = y.shape[1] // views
    target = tokens[:, :t]  # (B, T, D)
    z = target.mean(1)
    if proj is not None:
        z = proj(z)
    w = valid.to(tokens.dtype)
    n = w.sum()
    goal = target.detach() if sg else target
    pred_sum = torch.zeros_like(n)
    if token_weight > 0:
        per_view, at = [], t
        for v, k in enumerate(keeps):
            ctx_ids, tgt_ids = view_ids(y[:, v * t : (v + 1) * t], k)
            pred = predictor(tokens[:, at : at + k], ctx_ids, tgt_ids)  # (B, M, D)
            at += k
            s_t = goal.gather(1, tgt_ids[..., None].expand(-1, -1, d))
            diff = pred - s_t
            err = diff.abs() if pred_loss == "l1" else diff**2
            per_view.append(err.mean((1, 2)))  # (B,): 1/M sum_t 1/D l
        pred_b = torch.stack(per_view, 1).mean(1)
        pred_sum = torch.where(valid, pred_b, 0.0).sum()
    views_on = sigreg_on in ("views", "views_tokens")
    z_sum = torch.zeros_like(n)
    u = None
    if z_weight > 0 or views_on:
        u = view_embeddings(tokens, keeps, t, proj, projector)  # (B, V, d_u)
    if z_weight > 0:
        assert u is not None
        centre = u[:, :1].detach() if sg else u[:, :1]
        z_b = z_errors(torch.cat([centre, u[:, 1:]], 1), zpred).mean(1)
        z_sum = torch.where(valid, z_b, 0.0).sum()
    parts = []
    if sigreg_on in ("z", "both"):
        parts.append(weighted_sigreg(z, w, directions_z))
    if sigreg_on in ("tokens", "both", "views_tokens"):
        parts.append(
            token_sigreg_of(
                target,
                w,
                directions_z if directions_t is None else directions_t,
                token_sigreg,
            )
        )
    if views_on:
        assert u is not None
        dirs = directions_z if directions_u is None else directions_u
        per = [weighted_sigreg(u[:, v], w, dirs) for v in range(u.shape[1])]
        parts.append(torch.stack(per).mean())
    if not parts:
        raise ValueError(f"sigreg_on {sigreg_on!r} is not of SIGREG_ON")
    s = torch.stack(parts).mean()
    inner = pred_sum if token_weight == 1 else token_weight * pred_sum
    if z_weight > 0:
        inner = inner + z_weight * z_sum
    total = (1 - lam) * inner + lam * n * s
    return JEPATerms(
        total / b,
        pred_sum / n.clamp(min=1),
        s,
        n,
        z,
        torch.zeros_like(s),
        z_sum / n.clamp(min=1),
    )
