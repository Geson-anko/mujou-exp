r"""The foundation's learning (``plans/foundation.md`` 3; F1 to F3): the JEPA
of one stack completed in space and time, of tubelet tokens, the masks of
V-JEPA's multi-block spread over the stack's times, the spaced stacks, the
loss, the numbers of the validation video and the code $z$ of a checkpoint.
``jepa_iid --arm`` learns it.

**The tokens.** A stack of $n$ = 8 frames $x_\tau, x_{\tau + s}, \dots,
x_{\tau + 7 s}$ ($s$ the spacing, in frames of 0.1 s) is cut into tubelets of
``TUBELET`` = 2 consecutive frames x 8 x 8 pixels (``jepa.TubeletEmbed``):
the grid $\Omega = T \times P$ of $T$ = 4 times x $P$ = 8 x 8 positions, 256
tokens, time-major ($\omega = j P + p$), each of the fixed position
``jepa.sincos_3d``. The encoder is V18c's ViT (``build_fnd``: ``build_jepa``
with the tubelet embedding in place of the patch embedding) attending across
times and positions, no recurrence; the context side passes only the visible
tokens (``encode`` with ``keep``). The tokens are $t_\omega = R h_\omega \in
\mathbb{R}^{16}$ ($R$ V18c's fixed 16 x 1024) and the code of a stack is $z =
W \mathrm{vec}(t) \in \mathbb{R}^{4096}$, $W$ the fixed square 4096 x 4096
(``jepa.W_SEED``): one $z$ a stack.

**The masks** (``draw_masks``): V-JEPA's multi-block on $\Omega$. A stack's
mask is the union $U$ of the blocks of one ``BlockKind``: ``SHORT`` (8
blocks of 15 % of the 8 x 8 positions) or ``LONG`` (2 blocks of 70 %), the
first half of a batch short and the rest long (V-JEPA applies both to every
clip, of equal weight). A stack draws one aspect ratio $a$ uniform on
``ASPECT`` = [0.75, 1.5] (V-JEPA draws it once per batch: the same law of a
stack), the blocks' sides $h = \min(\mathrm{round}(\sqrt{n a}), g)$, $w =
\min(\mathrm{round}(\sqrt{n / a}), g)$ of $n = \lfloor g^2 \cdot
\mathrm{scale} \rfloor$ positions, and each block its top-left uniformly
where it fits in the $g \times g$ grid. Of a timed arm each block also draws
its length uniformly on $\{1, \dots, T\}$ and its first time uniformly where
it fits (the law is the same under the reversal and the translations of the
times); else it spans every time (a tube). The context is $V = \Omega
\setminus U$; a mask of an empty context is drawn again. The count hidden is
not fixed. With every block over every time the masks are V-JEPA's; at $T$ =
1 a timed arm draws no time and its masks are the image multi-block's, the
same draws as the untimed.

**The arms** (``ARMS``): ``T0`` (every block over every time, $s$ = 1),
``S1`` (the blocks' lengths drawn, $s$ = 1) and ``Ss`` (Sσ: the same, $\log
s$ uniform on $[0, \log s_{\max}]$, $s$ rounded to the nearest whole number:
``iid.draw_lags``). $s$ enters the predictor only, as $\psi(\ell)$ of $\ell =
\log_2 s$ added to its positions (``jepa.Predictor.st_positions``); ``T0``
and ``S1`` add none.

**The loss** (``fnd_terms``): $L = (1 - \lambda) \frac1B \sum_b \frac{1}{|U_b|
d} \sum_{\omega \in U_b} \lVert \hat t_{b,\omega} - t_{b,\omega} \rVert^2 +
\lambda\,\mathrm{SIGReg}(\{z_b\})$, no stop-gradient, no EMA. The attention
is ``attend``'s: ``models.Block``'s computation by
``scaled_dot_product_attention``, the padding of the context side masked out
of the keys.

The validation video's numbers are ``fnd_val``'s.

**The code of a checkpoint** (``FndCode``, ``z_fn``): $z$ ``(B, 4096)`` and
the tokens $t$ ``(B, 256, 16)`` of stacks u8 ``(B, 8, 3, 64, 64)``, the
target side's (every token seen).
"""

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from exp.lib import ckpt
from exp.lib.randproj import gaussian_projection
from exp.lib.sigreg import sigreg
from exp.vision.compose import from_dict
from exp.vision.iid import draw_lags
from exp.vision.jepa import (
    W_SEED,
    JEPAModules,
    Mix,
    Predictor,
    TokenEncoder,
    TubeletEmbed,
    build_jepa,
    sincos_3d,
    take,
)
from exp.vision.models import Block, early_fusion
from exp.vision.schema.model import HEAD_DIM
from exp.vision.schema.run import RunConf

__all__ = [
    "ARMS",
    "ASPECT",
    "EMBED_SEED",
    "EVAL_BATCH",
    "KINDS",
    "LONG",
    "SHORT",
    "S_MAX",
    "TOKEN_DIM",
    "TUBELET",
    "Arm",
    "BlockKind",
    "FndCode",
    "attend",
    "block_sides",
    "build_fnd",
    "context_index",
    "complete",
    "draw_blocks",
    "draw_masks",
    "draw_spacings",
    "encode",
    "fnd_terms",
    "load_fnd",
    "targets",
    "time_r2",
    "uniform_int",
    "z_fn",
]

TUBELET = 2  # frames of a tubelet
TOKEN_DIM = 16  # d (V18c's); D = |Omega| d (4096: W square)
S_MAX = 128  # the largest spacing of Ss (the stack's width about 90 s)
EMBED_SEED = 71  # added to the seed: the tubelet embedding's initial weights
ASPECT = (0.75, 1.5)  # V-JEPA's aspect ratios of a block


@dataclass(frozen=True)
class BlockKind:
    """A kind of V-JEPA's multi-block: ``blocks`` blocks, each of ``scale``
    of the spatial grid's positions."""

    name: str
    blocks: int
    scale: float


SHORT = BlockKind("short", 8, 0.15)  # V-JEPA's short range
LONG = BlockKind("long", 2, 0.7)  # V-JEPA's long range
KINDS = (SHORT, LONG)


@dataclass(frozen=True)
class Arm:
    """An arm of the foundation: ``timed`` (the blocks' lengths drawn) and
    ``scaled`` ($\\log s$ uniform, $\\psi(\\ell)$ to the predictor)."""

    name: str
    timed: bool
    scaled: bool


ARMS = {
    "T0": Arm("T0", timed=False, scaled=False),
    "S1": Arm("S1", timed=True, scaled=False),
    "Ss": Arm("Ss", timed=True, scaled=True),
}

EVAL_BATCH = 128  # stacks a time through the JEPA


# -- masks ----------------------------------------------------------------------


def block_sides(kind: BlockKind, grid: int, aspect: Tensor) -> tuple[Tensor, Tensor]:
    r"""The sides $h$, $w$ ``(B,)`` int64 of the blocks of ``kind`` on the
    ``grid`` x ``grid`` positions at the aspect ratios ``aspect`` ``(B,)``
    (V-JEPA's): $n = \lfloor g^2\,\mathrm{scale} \rfloor$, $h = \min(
    \mathrm{round}(\sqrt{n a}), g)$, $w = \min(\mathrm{round}(\sqrt{n / a}),
    g)$."""
    n = int(grid * grid * kind.scale)
    a = aspect.double()
    h = (n * a).sqrt().round().clamp(max=grid).long()
    w = (n / a).sqrt().round().clamp(max=grid).long()
    return h, w


def uniform_int(high: Tensor, gen: torch.Generator) -> Tensor:
    """Whole numbers uniform on ``[0, high)`` (``high`` int64 ``(...)``, each
    at least 1), one float64 draw each."""
    u = torch.rand(high.shape, generator=gen, dtype=torch.float64)
    return (u * high).floor().long().clamp(max=high - 1)


def draw_blocks(
    b: int, kind: BlockKind, times: int, grid: int, timed: bool, gen: torch.Generator
) -> Tensor:
    """``b`` masks of ``kind`` (module docstring) on the grid of ``times`` x
    ``grid`` x ``grid``, bool ``(b, times, grid, grid)`` (True hidden): per
    mask its aspect ratio, then per block its length and first time (of
    ``timed`` with ``times`` > 1; else every time, no draw), its top and its
    left, each in turn of ``gen``; the masks hiding every token drawn again
    (after the others, in order)."""
    lo, hi = ASPECT
    aspect = lo + torch.rand(b, generator=gen, dtype=torch.float64) * (hi - lo)
    h, w = block_sides(kind, grid, aspect)  # (b,)
    k = kind.blocks
    if timed and times > 1:
        length = 1 + uniform_int(torch.full((b, k), times), gen)
        first = uniform_int(times - length + 1, gen)
    else:
        length = torch.full((b, k), times)
        first = torch.zeros(b, k, dtype=torch.long)
    top = uniform_int((grid - h + 1)[:, None].expand(b, k), gen)
    left = uniform_int((grid - w + 1)[:, None].expand(b, k), gen)
    j = torch.arange(times)[None, None, :]  # (1, 1, T)
    r = torch.arange(grid)[None, None, :]  # (1, 1, g)
    in_t = (j >= first[..., None]) & (j < (first + length)[..., None])  # (b, k, T)
    in_r = (r >= top[..., None]) & (r < (top + h[:, None])[..., None])  # (b, k, g)
    in_c = (r >= left[..., None]) & (r < (left + w[:, None])[..., None])
    blocks = in_t[..., None, None] & in_r[:, :, None, :, None] & in_c[:, :, None, None]
    mask = blocks.any(1)  # (b, T, g, g)
    full = mask.flatten(1).all(1)
    if full.any():
        mask[full] = draw_blocks(int(full.sum()), kind, times, grid, timed, gen)
    return mask


def draw_masks(
    b: int, times: int, grid: int, timed: bool, gen: torch.Generator
) -> tuple[Tensor, Tensor]:
    """The masks of a batch of ``b`` stacks (module docstring), bool ``(b,
    times grid^2)`` time-major (True hidden), and each one's kind ``(b,)``
    (its index in ``KINDS``): the first ``b - b // 2`` of ``SHORT``, then
    ``LONG``, drawn in that order by ``gen``."""
    counts = (b - b // 2, b // 2)
    masks = [
        draw_blocks(n, kind, times, grid, timed, gen).flatten(1)
        for kind, n in zip(KINDS, counts, strict=True)
    ]
    kind = torch.cat([torch.full((n,), i) for i, n in enumerate(counts)])
    return torch.cat(masks), kind


def draw_spacings(b: int, arm: Arm, s_max: int, gen: torch.Generator) -> Tensor:
    """The spacings ``(b,)`` int64 of ``b`` stacks of ``arm``: 1 (no draw),
    or of ``scaled`` ``iid.draw_lags(b, 1, s_max, gen)`` (the rounded
    log-uniform)."""
    if not arm.scaled:
        return torch.ones(b, dtype=torch.long)
    return draw_lags(b, 1, s_max, gen)


# -- modules --------------------------------------------------------------------


def build_fnd(
    conf: RunConf, *, seed: int, token_dim: int = TOKEN_DIM, tubelet: int = TUBELET
) -> JEPAModules:
    r"""The foundation's JEPA of ``conf`` (module docstring) on
    ``conf.device``: ``build_jepa`` of the fixed $R$ (d = ``token_dim``),
    its encoder's patch embedding replaced by a ``TubeletEmbed`` of
    ``tubelet`` frames drawn from ``seed + EMBED_SEED`` (LeCun's truncated
    normal, zero bias; ``models._init_vit``'s rule, the input of muP), and
    the square $W$ ``(D, D)`` of ``W_SEED``, $D = |\Omega|\,d$. The blocks
    and the predictor are V18c's of ``seed``.

    Raises:
        ValueError: The stack is no whole number of tubelets, or as
            ``build_jepa``.
    """
    m, t = conf.model, conf.data.tours
    mods = build_jepa(conf, seed=seed, token_dim=token_dim, proj="fixed")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + EMBED_SEED)
        embed = TubeletEmbed(
            frames=t.frames, tubelet=tubelet, patch=m.patch, size=t.size, dim=m.width
        )
        std = 1 / math.sqrt(embed.proj.in_features)
        nn.init.trunc_normal_(embed.proj.weight, std=std, a=-2 * std, b=2 * std)
        nn.init.zeros_(embed.proj.bias)
    dev = torch.device(conf.device)
    encoder = mods.encoder
    assert isinstance(encoder, TokenEncoder)  # build_jepa's
    encoder.vit[0].embed = embed.to(dev)
    tokens = (t.frames // tubelet) * (t.size // m.patch) ** 2
    d = tokens * token_dim
    mods.mix = Mix(gaussian_projection(d, d, W_SEED)).to(dev)
    return mods


def attend(block: Block, x: Tensor, heads: int, valid: Tensor | None = None) -> Tensor:
    """``block``'s computation of ``x`` ``(B, L, w)`` (``heads`` heads) with
    the attention of ``scaled_dot_product_attention``; with ``valid`` ``(B,
    L)`` (True a token) the others are no key of any query (padding)."""
    n, length, d = x.shape
    q, k, v = (
        block.qkv(block.norm1(x))
        .reshape(n, length, 3, heads, d // heads)
        .permute(2, 0, 3, 1, 4)
    )  # each (n, heads, L, d_head)
    mask = None if valid is None else valid[:, None, None, :]
    a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    x = x + block.proj(a.transpose(1, 2).reshape(n, length, d))
    return x + block.fc2(F.gelu(block.fc1(block.norm2(x))))


def encode(
    mods: JEPAModules,
    images: Tensor,
    heads: int,
    keep: Tensor | None = None,
    valid: Tensor | None = None,
    *,
    again: bool = False,
) -> Tensor:
    """The encoder's tokens $h$ of the early-fused stacks ``images`` ``(B, 3n,
    H, W)``: every token ``(B, |Ω|, w)``, or with ``keep`` ``(B, V)`` (and
    ``valid``, its padding; ``context_index``) those of the positions
    ``keep`` only ``(B, V, w)``, dropped after the embedding. With ``again``
    each block is computed again in the backward (``checkpoint``)."""
    encoder = mods.encoder
    assert isinstance(encoder, TokenEncoder)  # build_fnd's
    parts = encoder.vit
    embed, norm = parts[0].embed, parts[-1].norm
    assert embed is not None and norm is not None
    x = embed(images)
    if keep is not None:
        x = take(x, keep)
    again = again and torch.is_grad_enabled()
    for part in parts:
        for block in part.blocks:
            assert isinstance(block, Block)
            if again:
                x = checkpoint(attend, block, x, heads, valid, use_reentrant=False)
            else:
                x = attend(block, x, heads, valid)
    return norm(x)


def context_index(hidden: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Of the masks ``hidden`` ``(B, |Ω|)`` (True hidden): ``keep`` ``(B,
    V)`` each stack's visible positions in increasing order, then (to $V$,
    the most visible of a stack) hidden ones as padding; ``valid`` ``(B,
    V)``, True on the visible; and ``slot`` ``(B, |Ω|)``, each visible
    position's index in ``keep`` (0 of a hidden one)."""
    visible = ~hidden
    count = visible.sum(1)  # (B,)
    v = int(count.max())
    order = torch.sort(hidden.to(torch.uint8), dim=1, stable=True).indices
    keep = order[:, :v]
    valid = torch.arange(v, device=hidden.device)[None] < count[:, None]
    slot = (visible.long().cumsum(1) - 1).clamp(min=0)
    return keep, valid, slot


def complete(
    mods: JEPAModules,
    context: Tensor,
    slot: Tensor,
    hidden: Tensor,
    times: int,
    ell: Tensor | None = None,
) -> Tensor:
    r"""The predictor's $\hat t$ ``(B, |Ω|, d)`` at every position (read at
    the hidden ones) from the context side's tokens ``context`` ``(B, V,
    w)`` (``context_index``'s ``keep``, ``slot``) of the masks ``hidden``
    ``(B, |Ω|)``: ``embed`` of a visible token or ``mask_token`` at a hidden
    one, plus the positions ``sincos_3d`` of the ``times`` x $g$ x $g$ grid
    (with the scales ``ell`` ``(B,)``, ``Predictor.st_positions``: plus
    $\psi(\ell)$), through the blocks (no padding: $V \cup U = \Omega$),
    ``norm`` and ``out``."""
    pred = mods.predictor
    assert isinstance(pred, Predictor)  # build_jepa's
    dim = pred.mask_token.shape[0]
    grid = math.isqrt(hidden.shape[1] // times)
    if ell is None:
        table = sincos_3d(times, grid, dim).to(context)  # (|Ω|, dim)
    else:
        table = pred.st_positions(times, ell)  # (B, |Ω|, dim)
    c = pred.embed(take(context, slot))  # (B, |Ω|, dim)
    x = torch.where(hidden[..., None], pred.mask_token, c) + table
    for block in pred.blocks:
        assert isinstance(block, Block)
        x = attend(block, x, dim // HEAD_DIM)
    return pred.out(pred.norm(x))


def fnd_terms(
    mods: JEPAModules,
    x: Tensor,
    hidden: Tensor,
    dirs: Tensor,
    lam: float,
    *,
    heads: int,
    times: int,
    ell: Tensor | None = None,
    again: bool = False,
) -> dict[str, Tensor]:
    r"""The loss of the stacks ``x`` u8 ``(B, n, 3, H, W)`` under the masks
    ``hidden`` ``(B, |Ω|)`` (``draw_masks``) and SIGReg's directions ``dirs``
    ``(M, D)`` (module docstring): ``loss`` $L$, ``pred`` (the mean over the
    stacks of each one's mean squared error over its hidden tokens and their
    coordinates), ``sigreg`` $S$ of $z$; detached, each stack's error
    ``per`` ``(B,)``, the target side's tokens ``tokens`` $t$ ``(B, |Ω|,
    d)`` and the codes ``z`` ``(B, D)``. ``ell`` the scales of
    ``complete``; ``again`` of ``encode``."""
    mix = mods.mix
    assert mix is not None  # build_fnd's
    images = early_fusion(x)
    t = mods.proj(encode(mods, images, heads, again=again))  # (B, |Ω|, d)
    z = mix(t.flatten(1))  # (B, D)
    keep, valid, slot = context_index(hidden)
    context = encode(mods, images, heads, keep, valid, again=again)
    t_hat = complete(mods, context, slot, hidden, times, ell)
    err = (t_hat - t).square().mean(-1)  # (B, |Ω|)
    per = (err * hidden).sum(1) / hidden.sum(1)
    pred = per.mean()
    s = sigreg(z, z[:0], dirs)
    return {
        "loss": (1 - lam) * pred + lam * s,
        "pred": pred,
        "sigreg": s,
        "per": per.detach(),
        "tokens": t.detach(),
        "z": z.detach(),
    }


def time_r2(tokens: Tensor, times: int) -> Tensor:
    r"""``time_r2`` of the tokens ``(N, |Ω|, d)`` (time-major, ``times``
    times): with $u_{n,j,p} = t_{n,j,p} - \langle t_{j,p} \rangle$ (the mean
    over the $N$ stacks) and $\bar u_{n,p} = \frac1T \sum_j u_{n,j,p}$, the
    share of the variance the mean over the times explains, $T \sum \lVert
    \bar u_{n,p} \rVert^2 / \sum \lVert u_{n,j,p} \rVert^2$: 1 of tokens
    constant along the times, about $1/T$ of tokens independent across the
    times of equal variance. A 0-d float64 tensor."""
    u = tokens.double() - tokens.double().mean(0)
    u = u.unflatten(1, (times, -1))  # (N, T, P, d)
    return times * u.mean(1).square().sum() / u.square().sum()


@torch.no_grad()
def targets(mods: JEPAModules, x: Tensor, heads: int) -> tuple[Tensor, Tensor]:
    """The tokens $t$ ``(N, |Ω|, d)`` and the codes $z$ ``(N, D)`` of the
    stacks ``x`` u8 (on any device), ``EVAL_BATCH`` at a time."""
    mix = mods.mix
    assert mix is not None
    dev = mods.parameters()[0].device
    ts, zs = [], []
    for chunk in x.split(EVAL_BATCH):
        t = mods.proj(encode(mods, early_fusion(chunk.to(dev)), heads))
        ts.append(t)
        zs.append(mix(t.flatten(1)))
    return torch.cat(ts), torch.cat(zs)


# -- the code of a checkpoint ---------------------------------------------------


def load_fnd(path: Path, device: str) -> tuple[RunConf, dict[str, Any], JEPAModules]:
    """The run's config (on ``device``), its ``jepa.json`` and its JEPA at
    the checkpoint ``path`` (``<run>/ckpt/<t>.pt``)."""
    run = path.parent.parent
    d = yaml.safe_load((run / "config.yaml").read_text())
    d["device"] = device
    conf = from_dict(d)
    settings = json.loads((run / "jepa.json").read_text())
    mods = build_fnd(
        conf,
        seed=conf.seed,
        token_dim=settings["token_dim"],
        tubelet=settings["tubelet"],
    )
    ckpt.load_eval(path, mods.modules())
    return conf, settings, mods


class FndCode:
    """The codes of the JEPA ``mods`` (``load_fnd``): ``mu`` the code $z$
    ``(B, D)`` and ``tokens`` $t$ ``(B, |Ω|, d)`` of stacks u8 ``(B, n, 3,
    S, S)`` (on any device; ``EVAL_BATCH`` at a time, no gradient)."""

    def __init__(self, mods: JEPAModules, heads: int) -> None:
        self._mods, self._heads = mods, heads

    def mu(self, x: Tensor) -> Tensor:
        return targets(self._mods, x, self._heads)[1]

    def tokens(self, x: Tensor) -> Tensor:
        return targets(self._mods, x, self._heads)[0]


def z_fn(arg: str, device: str) -> Callable[[Tensor], Tensor]:
    """``fnd_dyn``'s factory of a code: the map from stacks u8 to $z$ of the
    checkpoint ``arg`` (``<run>/ckpt/<t>.pt``; ``ckpt/00000.0.pt`` the step
    0) on ``device``."""
    conf, _, mods = load_fnd(Path(arg), device)
    return FndCode(mods, conf.model.head_count).mu
