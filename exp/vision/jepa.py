r"""The JEPA of V18, V18b and V18c (``plans/jepa.md``, 「V18」「V18b」「V18c」)
and V18d's spatio-temporal completion (``plans/v18d.md``): the masks, the
modules and the loss, in LeJEPA's form (one encoder, no EMA target, no
stop-gradient).

The encoder is the ViT of ``build_vae`` (the same steps and seed, so its
blocks start as the VAE's) without the head: a stack enters by
``early_fusion`` and leaves as $T = g^2$ tokens $h_p$ of the width $w$ after
the last ``LayerNorm`` ($g$ = size / patch; 4 x 4 at 64 px and patch 16,
V18's; 16 x 16 at patch 4, V18b's; 8 x 8 at patch 8, V18c's). The
projection $P$ (shared by the positions; $d$ = ``TOKEN_DIM`` = 64 by default)
maps them to $z_p = P h_p$, and $z = (z_1, \dots, z_T)$ flattened (row-major
positions, $\mathrm{vec}(z)_{dp + i} = z_{p,i}$) is the code of a stack, $D =
T d$ (1024 in V18). $P$ is learned (``Linear(w, d)``, ``"learned"``, V18's)
or the fixed random $R$ (``FixedProj``, ``"fixed"``, V18c's: $d \times w$ of
``exp.lib.randproj.gaussian_projection`` of ``R_SEED``, standard normal over
$\sqrt w$, never learned). V18c's code is $z = W \mathrm{vec}(t)$ of its
tokens $t_p = R h_p$ by the fixed random $W$ (``Mix``, ``LATENT_DIM`` x $T d$
of ``W_SEED``, standard normal over $\sqrt{T d}$): its size $D$ =
``LATENT_DIM`` depends on neither $T$ nor $d$, and an isotropic
$\mathrm{vec}(t)$ of unit variance gives $z$ of unit variance on average.
$R$ and $W$ are buffers of the checkpoints, not parameters, and their seeds
are not the run's.

Masks. The family $F_k$ (``mask_family``) is every union of one or two
rectangles of sides 1..G on the ``COARSE_GRID`` x ``COARSE_GRID`` grid ($G$ =
4, V18's) seen as a torus (its edges joined) that covers exactly $k$
positions. It is closed under the torus's translations, so under the uniform
draw on it (``draw_masks``) each position is hidden with probability exactly
$k / G^2$. On a finer grid $g = s G$ a drawn mask is spread to it
(``expand_masks``), each coarse position to its $s \times s$ block: the
pixels hidden are V18's, each fine position hidden with probability $k /
G^2$, $s^2 k$ positions a mask. With early fusion a hidden position is hidden
in every frame of the stack (V-JEPA's tube).

The two sides. The target side passes all $T$ tokens through the encoder;
the context side passes the visible $T - k$ only: they are dropped after the
patch embedding and its positions (V-JEPA), so attention runs among the
visible tokens and the hidden patches' pixels do not reach it. The
predictor (``Predictor``, width ``PRED_WIDTH``, ``PRED_BLOCKS`` pre-norm
blocks) maps the context to the width 256 by ``embed`` and adds its
positions, puts at each hidden position the learned ``mask_token`` plus its
position, runs the blocks over the $T$ tokens, and maps the hidden ones by a
``LayerNorm`` and ``out`` to $\hat z_p \in \mathbb{R}^d$.

The loss (``jepa_terms``): $L = (1 - \lambda) L_{\mathrm{pred}} + \lambda S$,
$L_{\mathrm{pred}} = \frac{1}{B k d} \sum_n \sum_{p \in M_n} \lVert \hat
z_{n,p} - z_{n,p} \rVert^2$ ($k$ the positions a mask hides), $S$ =
``exp.lib.sigreg.sigreg`` of the batch's data points along unit directions
(no buffer): the flat $z$ ``(B, T d)`` (``"flat"``, V18's), the mean token
$\bar z = \frac1T \sum_p z_p$ ``(B, d)`` (``"mean"``, V18b's arm M) or $W
\mathrm{vec}(z)$ ``(B, LATENT_DIM)`` (``"mix"``, V18c's; ``sigreg_points``).
The gradient reaches the target $z$ too, through $R$ and $W$ for V18c's.

V18d (``st_terms``): a clip is $K$ stacks of one video $n$ stacks apart,
its grid $\Omega$ the $K T$ tokens. A mask (``st_masks``) is a tube, a mask
of $F_k$ hidden at every time, or hides $K / 2$ whole times
(``time_masks``); either hides half of $\Omega$. The encoder stays a
function of one stack; only the predictor, of the positions $e_P(p) +
\phi(u_i) + \psi(\ell)$ (``Predictor.st_positions``), sees across times.
SIGReg is the mean of its terms of each time's $B$ points.

muP (``jepa_mup_roles``, ``jepa_mup_lr``): the encoder's roles are the
VAE's; a learned $P$ and ``embed`` map the encoder's width to a fixed size
(output, $k_w^{-1}$ with $k_w$ = width / ``BASE_WIDTH``); the predictor's
blocks and ``out`` are of the fixed width 256 ($k$ = 1). $R$ and $W$ have no
role (no parameters).
"""

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import override

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from exp.lib.randproj import gaussian_projection
from exp.lib.sigreg import sigreg
from exp.vision.models import (
    INIT_STD,
    LR_EXPONENT,
    Block,
    PatchEmbed,
    ViTPart,
    build_vae,
    early_fusion,
    sincos_2d,
)
from exp.vision.schema.model import BASE_WIDTH, HEAD_DIM
from exp.vision.schema.run import RunConf

__all__ = [
    "COARSE_GRID",
    "LATENT_DIM",
    "PREDICTOR_SEED",
    "PRED_BLOCKS",
    "PRED_WIDTH",
    "PROJ_KINDS",
    "PROJ_SEED",
    "R_SEED",
    "W_SEED",
    "SIGREG_ON",
    "ST_LAG_BASE",
    "TOKEN_DIM",
    "FixedProj",
    "JEPAModules",
    "Mix",
    "Predictor",
    "TokenEncoder",
    "TubeletEmbed",
    "build_jepa",
    "directions",
    "draw_masks",
    "expand_masks",
    "hidden_count",
    "jepa_mup_lr",
    "jepa_mup_roles",
    "jepa_terms",
    "mask_expansion",
    "mask_family",
    "mask_index",
    "predict",
    "rho_in",
    "sigreg_points",
    "sincos_1d",
    "sincos_3d",
    "st_masks",
    "st_predict",
    "st_terms",
    "take",
    "time_masks",
]

TOKEN_DIM = 64  # d: the code of a token (V18's; V18b's arm F takes 4)
COARSE_GRID = 4  # G: the grid of the masks' family (V18's, patch 16 at 64 px)
SIGREG_ON = ("flat", "mean", "mix")  # SIGReg's data point: z, z_bar or W vec(z)
PROJ_KINDS = ("learned", "fixed")  # P: Linear(w, d) (V18's) or the fixed R (V18c's)
LATENT_DIM = 1024  # D of W vec(t) (V18c's), whatever T and d
R_SEED = 67  # R's own seed (not added to the run's)
W_SEED = 61  # W's own seed, of jepa_eval's z_w too (plans/jepa.md, V18c)
PRED_WIDTH, PRED_BLOCKS = 256, 4  # the predictor's width and blocks
PROJ_SEED = 21  # added to the seed: the projection's initial weights
PREDICTOR_SEED = 23  # added to the seed: the predictor's initial weights
ST_LAG_BASE = 8  # V18d: the scale's position is log2(n / ST_LAG_BASE)


# -- masks ----------------------------------------------------------------------


def hidden_count(ratio: float, tokens: int) -> int:
    """$k$ = ``ratio`` x ``tokens``, the positions a mask hides.

    Raises:
        ValueError: It is not a whole number in ``[1, tokens - 1]``.
    """
    k = round(ratio * tokens)
    if abs(k - ratio * tokens) > 1e-9 or not 1 <= k < tokens:
        raise ValueError(
            f"a mask ratio of {ratio} hides {ratio * tokens} of {tokens} tokens, "
            "not a whole number in [1, tokens - 1]"
        )
    return k


def _torus_rects(grid: int) -> set[int]:
    """The distinct rectangles of sides 1..``grid`` on the ``grid`` x ``grid``
    torus, as bit masks of the row-major positions."""
    rects: set[int] = set()
    for i in range(grid):
        for j in range(grid):
            for h in range(1, grid + 1):
                for w in range(1, grid + 1):
                    bits = 0
                    for a in range(h):
                        for b in range(w):
                            bits |= 1 << ((i + a) % grid * grid + (j + b) % grid)
                    rects.add(bits)
    return rects


def mask_family(k: int, grid: int) -> Tensor:
    r"""$F_k$ (module docstring): the distinct unions of one or two torus
    rectangles that cover exactly ``k`` of the ``grid``\ :sup:`2` positions,
    bool ``(|F_k|, grid^2)`` (True hidden), in increasing order of their bit
    masks."""
    rects = _torus_rects(grid)
    family = sorted({a | b for a in rects for b in rects if (a | b).bit_count() == k})
    t = grid * grid
    return torch.tensor([[bits >> p & 1 for p in range(t)] for bits in family]).bool()


def draw_masks(family: Tensor, b: int, gen: torch.Generator) -> Tensor:
    """``b`` masks drawn uniformly and independently from ``family`` ``(F,
    T)``: bool ``(b, T)``, on the family's device (``gen`` a CPU
    generator)."""
    idx = torch.randint(len(family), (b,), generator=gen)
    return family[idx.to(family.device)]


def mask_expansion(grid: int) -> int:
    """$s$ = ``grid`` / ``COARSE_GRID``, the fine positions a side of a
    coarse one.

    Raises:
        ValueError: ``grid`` is not a whole multiple of ``COARSE_GRID``.
    """
    s = grid // COARSE_GRID
    if s < 1 or s * COARSE_GRID != grid:
        raise ValueError(
            f"a grid of {grid} is not a whole multiple of the masks' {COARSE_GRID}"
        )
    return s


def expand_masks(mask: Tensor, s: int) -> Tensor:
    r"""The masks ``(B, G^2)`` of the coarse grid ($G$ = ``COARSE_GRID``)
    spread to the grid $g = s G$: the fine position $p = g i + j$ takes the
    coarse $G \lfloor i / s \rfloor + \lfloor j / s \rfloor$'s, ``(B, g^2)``
    (the identity for ``s`` = 1)."""
    b, g = len(mask), COARSE_GRID
    grids = mask.reshape(b, g, g)
    return grids.repeat_interleave(s, 1).repeat_interleave(s, 2).flatten(1)


def mask_index(mask: Tensor) -> tuple[Tensor, Tensor]:
    """The visible and the hidden positions of the masks ``(B, T)`` (each
    hiding the same $k$), in increasing order: ``(B, T - k)`` and ``(B,
    k)``."""
    b = len(mask)
    keep = (~mask).nonzero()[:, 1].reshape(b, -1)
    hide = mask.nonzero()[:, 1].reshape(b, -1)
    return keep, hide


def time_masks(k: int) -> Tensor:
    r"""V18d's time side (plans/v18d.md, 「マスクの族」): every $J \subset \{1,
    \dots, k\}$ of $\lvert J \rvert = k / 2$, bool ``(C(k, k/2), k)`` (True
    the hidden times), in the order of ``itertools.combinations``; empty ``(0,
    1)`` for ``k`` = 1 (V18c's: no time side).

    Raises:
        ValueError: ``k`` is neither 1 nor a positive even number.
    """
    if k != 1 and (k < 2 or k % 2):
        raise ValueError(f"the clips' times are 1 or even, not {k}")
    if k == 1:
        return torch.zeros(0, 1, dtype=torch.bool)
    rows = [
        [i in j for i in range(k)] for j in itertools.combinations(range(k), k // 2)
    ]
    return torch.tensor(rows)


def st_masks(
    family: Tensor, b: int, k: int, s: int, gen: torch.Generator
) -> tuple[Tensor, Tensor]:
    r"""The masks of ``b`` clips of ``k`` times (plans/v18d.md, 「側の選び方」),
    bool ``(b, k, T)`` on the fine grid ($T = s^2$ ``family.shape[1]``), and
    each clip's ``kind`` ``(b,)``: -1 for a tube, else its row of
    ``time_masks(k)``. The first ``b / 2`` clips (all ``b`` for ``k`` = 1)
    are tubes: a mask of ``draw_masks(family, ., gen)`` spread by
    ``expand_masks`` and hidden at every time; the others hide whole times,
    a row of ``time_masks(k)`` drawn uniformly by ``gen`` (after the tubes).
    For ``k`` = 1 the draw and the masks are V18c's.

    Raises:
        ValueError: ``k`` is not of ``time_masks``, or ``k`` > 1 and ``b`` is
            odd.
    """
    times = time_masks(k).to(family.device)
    if k > 1 and b % 2:
        raise ValueError(f"the clips of two sides are even, not {b}")
    tubes = b if k == 1 else b // 2
    tube = expand_masks(draw_masks(family, tubes, gen), s)  # (b_t, T)
    count = tube.shape[1]
    out = [tube[:, None].expand(-1, k, -1)]
    kind = [torch.full((tubes,), -1, dtype=torch.long)]
    if b > tubes:
        j = torch.randint(len(times), (b - tubes,), generator=gen)
        out.append(times[j.to(family.device)][:, :, None].expand(-1, -1, count))
        kind.append(j)
    return torch.cat(out), torch.cat(kind).to(family.device)


def take(tokens: Tensor, idx: Tensor) -> Tensor:
    """The tokens ``(B, T, w)`` at the positions ``idx`` ``(B, m)``: ``(B, m,
    w)``."""
    return tokens.gather(1, idx[..., None].expand(-1, -1, tokens.shape[-1]))


# -- modules --------------------------------------------------------------------


class TokenEncoder(nn.Module):
    """The ViT of ``build_vae`` without its head (``parts``, the
    ``ViTPart``\\ s whose last keeps its ``LayerNorm``): images ``(B, 3n, H,
    W)`` to the tokens ``(B, T, w)``, or with ``keep`` ``(B, V)`` to those of
    the visible positions only ``(B, V, w)``, dropped after the patch
    embedding. With ``checkpoint_blocks`` a block's activations are not kept
    for the backward but computed again by the same operations
    (``torch.utils.checkpoint``): the same values, less memory."""

    def __init__(
        self, parts: Sequence[ViTPart], *, checkpoint_blocks: bool = False
    ) -> None:
        super().__init__()
        self.parts = nn.ModuleList(parts)
        self._vit = list(parts)  # the same modules, as ViTPart
        self._checkpoint = checkpoint_blocks

    @property
    def vit(self) -> list[ViTPart]:
        """The ``ViTPart``\\ s, in order (the same modules as ``parts``)."""
        return self._vit

    @override
    def forward(self, images: Tensor, keep: Tensor | None = None) -> Tensor:
        embed, norm = self._vit[0].embed, self._vit[-1].norm
        assert embed is not None and norm is not None
        x = embed(images)
        if keep is not None:
            x = take(x, keep)
        again = self._checkpoint and torch.is_grad_enabled()
        for part in self._vit:
            for block in part.blocks:
                x = checkpoint(block, x, use_reentrant=False) if again else block(x)
        return norm(x)


def sincos_1d(x: Tensor, dim: int) -> Tensor:
    r"""The fixed position of the reals ``x`` ``(...,)``, float64 ``(...,
    dim)``: $\sin$ then $\cos$ of $x \omega_j$, $\omega_j = 10000^{-j /
    (dim / 2)}$ ($j < dim / 2$; ``sincos_2d``'s frequencies of one axis
    spread over the whole width). $x \mapsto -x$ flips the sines only, and
    $x \mapsto x + a$ rotates each pair $(\sin, \cos)$ by $a \omega_j$."""
    half = dim // 2
    omega = 1.0 / 10000 ** (torch.arange(half, dtype=torch.float64) / half)
    a = x.double()[..., None] * omega.to(x.device)
    return torch.cat([a.sin(), a.cos()], -1)


def sincos_3d(times: int, grid: int, dim: int) -> Tensor:
    r"""The fixed positions of the tubelets' grid $\Omega$ of ``times`` x
    ``grid`` x ``grid`` (the foundation's, ``plans/foundation.md`` 3),
    ``(times grid^2, dim)`` float32 time-major ($\omega = j\,g^2 + p$):
    $e(j, p) = e_P(p) + \phi(u_j)$, $e_P$ = ``sincos_2d((grid, grid), dim)``
    and $\phi$ = ``sincos_1d`` of the centred time $u_j = j - (\mathrm{times}
    - 1)/2$, each over the whole width (``Predictor.st_positions`` without
    $\psi$: the same sum, so adding $\psi$ gives its table to the bit)."""
    pos = sincos_2d((grid, grid), dim)  # (P, dim)
    u = torch.arange(times, dtype=torch.float64) - (times - 1) / 2
    phi = sincos_1d(u, dim).to(pos)  # (times, dim)
    return (pos[None] + phi[:, None]).flatten(0, 1)


class TubeletEmbed(PatchEmbed):
    r"""The patch embedding of tubelets (``plans/foundation.md`` 3): the
    early-fused stacks ``(B, 3 n, H, W)`` (``early_fusion``) as the tokens
    ``(B, T P, w)`` of boxes of ``tubelet`` consecutive frames x ``patch``
    x ``patch`` pixels, $T = n$ / ``tubelet`` times x $P = (H / p)^2$
    positions, time-major ($\omega = j P + p$, ``sincos_3d``'s order). A box
    is flattened as ``PatchEmbed`` flattens a patch of the early-fused
    channels (channel $3 f + c$ of its frame $f$, then the row, then the
    column), mapped by ``proj`` (``PatchEmbed``'s of ``3 tubelet`` channels:
    ``Linear(3 tubelet p^2, w)``) plus its fixed position ``sincos_3d(T, H /
    p, w)``. With ``tubelet`` = $n$ the linear part is ``PatchEmbed``'s of
    the same weight.

    Raises:
        ValueError: The stacks are no whole number of tubelets.
    """

    def __init__(
        self, *, frames: int, tubelet: int, patch: int, size: int, dim: int
    ) -> None:
        if frames % tubelet or size % patch:
            raise ValueError(
                f"{frames} frames of {size} px are no whole tubelets of "
                f"{tubelet} x {patch} x {patch}"
            )
        super().__init__(channels=3 * tubelet, patch=patch, size=size, dim=dim)
        self._tubelet = tubelet
        self.pos = sincos_3d(frames // tubelet, size // patch, dim)

    @override
    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        a, p = self._tubelet, self._patch
        t = c // (3 * a)
        x = x.reshape(b, t, 3 * a, h // p, p, w // p, p).permute(0, 1, 3, 5, 2, 4, 6)
        return (
            self.proj(x.reshape(b, t * (h // p) * (w // p), 3 * a * p * p)) + self.pos
        )


class Predictor(nn.Module):
    """The predictor (module docstring): ``embed`` (width ``width_in`` to
    ``dim``), ``mask_token``, the fixed positions ``sincos_2d((grid, grid),
    dim)``, ``depth`` ``Block``\\ s of ``dim / HEAD_DIM`` heads, ``norm`` and
    ``out`` (``dim`` to ``out_dim``)."""

    def __init__(
        self,
        *,
        width_in: int,
        grid: int,
        dim: int = PRED_WIDTH,
        depth: int = PRED_BLOCKS,
        out_dim: int = TOKEN_DIM,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(width_in, dim)
        self.mask_token = nn.Parameter(torch.zeros(dim))
        self.blocks = nn.ModuleList([Block(dim, dim // HEAD_DIM) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, out_dim)
        self.register_buffer("pos", sincos_2d((grid, grid), dim), persistent=False)
        self.pos: Tensor

    @override
    def forward(
        self,
        context: Tensor,
        keep: Tensor,
        hide: Tensor,
        ell: Tensor | None = None,
        times: int = 1,
    ) -> Tensor:
        """$\\hat z$ ``(B, k, out_dim)`` at the positions ``hide`` ``(B, k)``
        from the context side's tokens ``context`` ``(B, V, width_in)`` at
        ``keep`` ``(B, V)``. Without ``ell`` the positions are those of one
        stack (V18c's); with V18d's scales ``ell`` ``(B,)`` a position $i T +
        p$ of the clip's ``times`` stacks is $e(i, p) = e_P(p) + \\phi(u_i) +
        \\psi(\\ell)$ (``st_positions``)."""
        if ell is None:
            pos_c, pos_m = self.pos[keep], self.pos[hide]
        else:
            table = self.st_positions(times, ell)  # (B, times T, dim)
            pos_c, pos_m = take(table, keep), take(table, hide)
        c = self.embed(context) + pos_c  # (B, V, dim)
        m = self.mask_token + pos_m  # (B, k, dim)
        x = torch.cat([c, m], 1)
        for block in self.blocks:
            x = block(x)
        return self.out(self.norm(x[:, keep.shape[1] :]))

    def st_positions(self, times: int, ell: Tensor) -> Tensor:
        r"""V18d's positions ``(B, times T, dim)`` (plans/v18d.md, 「位置の符号」)
        of the clips of the scales ``ell`` ``(B,)``: $e_P(p) + \phi(u_i) +
        \psi(\ell)$ at $i T + p$, $e_P$ the stack's ``pos``, $\phi$ and
        $\psi$ ``sincos_1d`` of the time $u_i = i - (\mathrm{times} - 1)/2$
        (from 0) and of $\ell$; fixed, no parameter."""
        u = torch.arange(times, dtype=torch.float64) - (times - 1) / 2
        phi = sincos_1d(u, self.pos.shape[1]).to(self.pos)  # (times, dim)
        psi = sincos_1d(ell.double(), self.pos.shape[1]).to(self.pos)  # (B, dim)
        table = self.pos[None, None] + phi[None, :, None] + psi[:, None, None]
        return table.flatten(1, 2)


class FixedProj(nn.Module):
    r"""The fixed projection $R$ ``weight`` ``(d, w)`` of the tokens, a
    buffer (not learned): $t = h R^\top$, ``(..., w)`` to ``(..., d)``."""

    def __init__(self, weight: Tensor) -> None:
        super().__init__()
        self.register_buffer("weight", weight)
        self.weight: Tensor

    @override
    def forward(self, h: Tensor) -> Tensor:
        return h @ self.weight.T


class Mix(nn.Module):
    r"""The fixed map $W$ ``weight`` ``(D, T d)`` of the flat tokens, a buffer
    (not learned): $z = \mathrm{vec}(t) W^\top$, ``(B, T d)`` to ``(B,
    D)``."""

    def __init__(self, weight: Tensor) -> None:
        super().__init__()
        self.register_buffer("weight", weight)
        self.weight: Tensor

    @override
    def forward(self, v: Tensor) -> Tensor:
        return v @ self.weight.T


def _trunc_normal_(w: Tensor, std: float) -> None:
    nn.init.trunc_normal_(w, std=std, a=-2 * std, b=2 * std)


def _init_predictor(p: Predictor) -> None:
    """``_init_vit``'s rule: LeCun for ``embed`` and ``out``, 0.02 for the
    blocks' matrices and the mask token, zero biases, unit norms."""
    lecun = {id(p.embed), id(p.out)}
    for m in p.modules():
        if isinstance(m, nn.Linear):
            std = 1 / math.sqrt(m.in_features) if id(m) in lecun else INIT_STD
            _trunc_normal_(m.weight, std)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    _trunc_normal_(p.mask_token, INIT_STD)


@dataclass
class JEPAModules:
    """The JEPA: ``encoder`` (images to tokens ``(B, T, w)``, with ``keep``
    the visible ones), ``proj`` (a token to $z_p$: a learned ``Linear`` or
    the fixed ``FixedProj``), ``predictor`` (``Predictor``'s call) and, of
    V18c's, ``mix`` (``Mix``, the flat tokens to the code)."""

    encoder: nn.Module
    proj: nn.Module
    predictor: nn.Module
    mix: nn.Module | None = None

    def modules(self) -> dict[str, nn.Module]:
        """The modules by the names of the checkpoints (``mix`` if any)."""
        mods = {"encoder": self.encoder, "proj": self.proj, "predictor": self.predictor}
        return mods if self.mix is None else mods | {"mix": self.mix}

    def parameters(self) -> list[nn.Parameter]:
        r"""The learned parameters, the encoder's first ($R$ and $W$ are
        buffers, not of them)."""
        return [p for m in self.modules().values() for p in m.parameters()]

    def tokens(self, stacks: Tensor) -> Tensor:
        """$z_p$ ``(B, T, d)`` of the stacks ``(B, n, 3, H, W)`` (u8, or floats
        in ``[0, 1]``): the target side."""
        return self.proj(self.encoder(early_fusion(stacks)))

    def mu(self, stacks: Tensor) -> Tensor:
        r"""The code of the stacks (the evaluations'): the flat $z$ ``(B, T
        d)``, or with ``mix`` $W \mathrm{vec}(z)$ ``(B, D)``."""
        flat = self.tokens(stacks).flatten(1)
        return flat if self.mix is None else self.mix(flat)


def build_jepa(
    conf: RunConf,
    *,
    seed: int,
    token_dim: int = TOKEN_DIM,
    checkpoint_blocks: bool = False,
    proj: str = "learned",
    latent_dim: int | None = None,
) -> JEPAModules:
    r"""The JEPA of ``conf.model`` (a ViT) on ``conf.device``, $d$ =
    ``token_dim``: the encoder of ``build_vae(conf, seed=seed)`` (muP's
    initialization included) without its head (``TokenEncoder`` of
    ``checkpoint_blocks``); $P$ of ``proj`` (``PROJ_KINDS``): ``"learned"``
    drawn from ``seed + PROJ_SEED``, ``"fixed"`` the $R$ of ``R_SEED``
    (module docstring, whatever ``seed``); the predictor drawn from ``seed +
    PREDICTOR_SEED`` (on the CPU, the global RNG left as it was); with
    ``latent_dim`` $D$, ``mix`` $W$ ``(D, T d)`` of ``W_SEED``. In muP the
    output matrices of a learned $P$ and ``embed`` are drawn of LeCun's std
    times $k_w^{-1/2}$. The encoder depends on neither ``token_dim`` nor
    ``proj``, nor the predictor on ``proj``.

    Raises:
        ValueError: ``model.arch`` is not ``"vit"``, or ``proj`` is not of
            ``PROJ_KINDS``.
    """
    m, t = conf.model, conf.data.tours
    if m.arch != "vit":
        raise ValueError(f"the JEPA's encoder is the ViT, not {m.arch!r}")
    if proj not in PROJ_KINDS:
        raise ValueError(f"proj is one of {PROJ_KINDS}, not {proj!r}")
    parts = list(build_vae(conf, seed=seed).encoder.parts)
    vit = [p for p in parts if isinstance(p, ViTPart)]
    vit[-1].head = None
    grid = t.size // m.patch
    with torch.random.fork_rng(devices=[]):
        projection: nn.Module
        if proj == "learned":
            torch.manual_seed(seed + PROJ_SEED)
            projection = nn.Linear(m.width, token_dim)
            _trunc_normal_(projection.weight, 1 / math.sqrt(m.width))
            nn.init.zeros_(projection.bias)
        else:
            projection = FixedProj(gaussian_projection(token_dim, m.width, R_SEED))
        torch.manual_seed(seed + PREDICTOR_SEED)
        predictor = Predictor(width_in=m.width, grid=grid, out_dim=token_dim)
        _init_predictor(predictor)
    if m.param == "mup":
        k = m.width / BASE_WIDTH
        with torch.no_grad():
            if isinstance(projection, nn.Linear):
                projection.weight.mul_(k**-0.5)
            predictor.embed.weight.mul_(k**-0.5)
    dev = torch.device(conf.device)
    encoder = TokenEncoder(vit, checkpoint_blocks=checkpoint_blocks)
    mix = None
    if latent_dim is not None:
        w = gaussian_projection(latent_dim, grid * grid * token_dim, W_SEED)
        mix = Mix(w).to(dev)
    return JEPAModules(encoder, projection.to(dev), predictor.to(dev), mix)


# -- muP ------------------------------------------------------------------------


def jepa_mup_roles(mods: JEPAModules) -> dict[Tensor, str]:
    """The role in muP of each parameter (``exp.vision.models.ROLES``): the
    patch embedding's matrix (``PatchEmbed``'s, a ``TubeletEmbed``'s too) is
    ``"input"``; a learned $P$'s, ``embed``'s and ``out``'s ``"output"``;
    the other matrices ``"hidden"``; the biases, the norms and the mask
    token ``"vector"``.

    Raises:
        ValueError: The modules are not ``build_jepa``'s.
    """
    pred = mods.predictor
    embeds = [m for m in mods.encoder.modules() if isinstance(m, PatchEmbed)]
    if not isinstance(pred, Predictor) or len(embeds) != 1:
        raise ValueError("muP's roles are of build_jepa's modules")
    inputs = {id(embeds[0].proj.weight)}
    outputs = {id(pred.embed.weight), id(pred.out.weight)}
    if isinstance(mods.proj, nn.Linear):
        outputs.add(id(mods.proj.weight))
    matrices = {
        id(m.weight)
        for module in mods.modules().values()
        for m in module.modules()
        if isinstance(m, nn.Linear)
    }
    return {
        p: "input"
        if id(p) in inputs
        else "output"
        if id(p) in outputs
        else "hidden"
        if id(p) in matrices
        else "vector"
        for p in mods.parameters()
    }


def jepa_mup_lr(mods: JEPAModules, width: int) -> dict[Tensor, float]:
    r"""The learning rate's multiplier of each parameter in muP: $k^e$ of
    its role's exponent $e$ (Adam's: $-1$ for the hidden and the output
    matrices, 0 for the inputs and the vectors), $k$ = ``width`` /
    ``BASE_WIDTH`` for the encoder, $P$ and ``embed`` (of the encoder's
    width), $k$ = ``PRED_WIDTH`` / ``BASE_WIDTH`` = 1 for the rest of the
    predictor."""
    roles = jepa_mup_roles(mods)
    pred = mods.predictor
    assert isinstance(pred, Predictor)  # jepa_mup_roles checked it
    of_width = {id(p) for p in (*mods.encoder.parameters(), *mods.proj.parameters())}
    of_width |= {id(p) for p in pred.embed.parameters()}
    k_w, k_p = width / BASE_WIDTH, PRED_WIDTH / BASE_WIDTH
    return {
        p: (k_w if id(p) in of_width else k_p) ** LR_EXPONENT[role]
        for p, role in roles.items()
    }


# -- the loss -------------------------------------------------------------------


def directions(m: int, d: int, gen: torch.Generator) -> Tensor:
    """``m`` directions ``(m, d)`` uniform on the unit sphere (standard
    normal rows made of length 1), on ``gen``'s device."""
    a = torch.randn(m, d, generator=gen, device=gen.device)
    return a / torch.linalg.vector_norm(a, dim=1, keepdim=True)


def predict(mods: JEPAModules, images: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    """$\\hat z$ ``(B, k, d)`` of the context side of ``images`` ``(B, 3n, H,
    W)`` under ``mask`` ``(B, T)``, and the hidden positions ``(B, k)``."""
    keep, hide = mask_index(mask)
    return mods.predictor(mods.encoder(images, keep), keep, hide), hide


def sigreg_points(tokens: Tensor, on: str, mix: nn.Module | None = None) -> Tensor:
    r"""SIGReg's data points of the tokens $z_p$ ``(B, T, d)``: the flat $z$
    ``(B, T d)`` (``on`` ``"flat"``), the mean token $\bar z = \frac1T
    \sum_p z_p$ ``(B, d)`` (``"mean"``) or $W \mathrm{vec}(z)$ ``(B, D)``
    by ``mix`` (``"mix"``).

    Raises:
        ValueError: ``on`` is not of ``SIGREG_ON``, or is ``"mix"`` without
            ``mix``.
    """
    if on == "flat":
        return tokens.flatten(1)
    if on == "mean":
        return tokens.mean(1)
    if on == "mix":
        if mix is None:
            raise ValueError("sigreg_on 'mix' needs the JEPA's mix W")
        return mix(tokens.flatten(1))
    raise ValueError(f"sigreg_on is one of {SIGREG_ON}, not {on!r}")


def jepa_terms(
    mods: JEPAModules,
    x: Tensor,
    mask: Tensor,
    dirs: Tensor,
    lam: float,
    sigreg_on: str = "flat",
) -> dict[str, Tensor]:
    r"""The loss of the stacks ``x`` u8 ``(B, n, 3, H, W)`` under ``mask``
    ``(B, T)`` (each hiding $k$) and the directions ``dirs`` ``(M, D_S)`` of
    SIGReg's data points (``sigreg_points`` of ``sigreg_on`` and
    ``mods.mix``; module docstring): ``loss`` $L$, ``pred``
    $L_{\mathrm{pred}}$, ``sigreg`` $S$, and the target side's ``tokens``
    $z_p$ ``(B, T, d)`` (detached)."""
    images = early_fusion(x)
    tokens = mods.proj(mods.encoder(images))  # (B, T, d), the target side
    z_hat, hide = predict(mods, images, mask)
    pred = (z_hat - take(tokens, hide)).square().mean()
    z = sigreg_points(tokens, sigreg_on, mods.mix)  # (B, D_S)
    s = sigreg(z, z[:0], dirs)
    return {
        "loss": (1 - lam) * pred + lam * s,
        "pred": pred,
        "sigreg": s,
        "tokens": tokens.detach(),
    }


def st_predict(
    mods: JEPAModules, x: Tensor, n: Tensor, mask: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    r"""V18d's completion (plans/v18d.md, 「目的の式」) of the clips ``x`` u8
    ``(B, K, N, 3, H, W)`` of ``K`` stacks $n$ = ``n`` ``(B,)`` apart under
    ``mask`` ``(B, K, T)`` (``st_masks``; each clip hiding the same count
    $m$): $\hat t$ ``(B, m, d)`` at the hidden positions ``(B, m)`` of the
    grid $\Omega$ ($i T + p$, increasing), and the target side's tokens $t$
    ``(B K, T, d)``.

    The target side runs every stack once: $h$ ``(B K, T, w)``, $t = P h$.
    The context of a stack is $h$ itself when it is wholly visible (the same
    function of the same weights as a second call), none when wholly hidden,
    and else the encoder's run on its visible positions only. The predictor
    takes a clip's visible tokens and its hidden positions over $\Omega$, of
    V18d's positions of $\ell = \log_2(n / $ ``ST_LAG_BASE``$)$ (V18c's for
    $K$ = 1)."""
    b, k, count = mask.shape
    images = early_fusion(x.flatten(0, 1))  # (B K, 3N, H, W)
    h = mods.encoder(images)  # (B K, T, w)
    tokens = mods.proj(h)  # (B K, T, d)
    flat = mask.flatten(0, 1)  # (B K, T)
    hidden = flat.sum(1)
    part = ((hidden > 0) & (hidden < count)).nonzero()[:, 0]
    ctx = h
    if len(part):
        keep_p, _ = mask_index(flat[part])
        seen = mods.encoder(images[part], keep_p)  # (P, V_i, w)
        ctx = h.index_put((part[:, None], keep_p), seen)
    keep, hide = mask_index(mask.flatten(1))  # (B, K T - m), (B, m)
    context = take(ctx.unflatten(0, (b, k)).flatten(1, 2), keep)
    ell = None if k == 1 else (n.double() / ST_LAG_BASE).log2()
    return mods.predictor(context, keep, hide, ell, k), hide, tokens


def st_terms(
    mods: JEPAModules,
    x: Tensor,
    n: Tensor,
    mask: Tensor,
    dirs: Tensor,
    lam: float,
    sigreg_on: str = "flat",
) -> dict[str, Tensor]:
    r"""V18d's loss $L = (1 - \lambda) L_{\mathrm{ST}} + \lambda S$
    (plans/v18d.md, 「損失」「SIGReg」) of ``st_predict``'s clips: ``pred``
    $L_{\mathrm{ST}}$ is the mean squared error of $\hat t$ over the hidden
    tokens and their coordinates; ``sigreg`` $S = \frac1K \sum_i$ SIGReg of
    the $B$ points of time $i$ (``sigreg_points`` of ``sigreg_on``, the same
    ``dirs``), its terms in ``sigreg_times`` ``(K,)``. Also ``clip_err``
    ``(B,)``, each clip's mean squared error, and the target side's
    ``tokens`` ``(B K, T, d)``, detached. For $K$ = 1 and tubes, V18c's
    ``jepa_terms`` bit for bit."""
    b, k = mask.shape[:2]
    z_hat, hide, tokens = st_predict(mods, x, n, mask)
    per = tokens.unflatten(0, (b, k))  # (B, K, T, d)
    err = (z_hat - take(per.flatten(1, 2), hide)).square()
    pred = err.mean()
    terms = []
    for i in range(k):
        z = sigreg_points(per[:, i], sigreg_on, mods.mix)  # (B, D_S)
        terms.append(sigreg(z, z[:0], dirs))
    each = torch.stack(terms)
    s = each.mean()
    return {
        "loss": (1 - lam) * pred + lam * s,
        "pred": pred,
        "sigreg": s,
        "sigreg_times": each.detach(),
        "clip_err": err.detach().mean((1, 2)),
        "tokens": tokens.detach(),
    }


def rho_in(tokens: Tensor) -> Tensor:
    r"""$\rho_{\mathrm{in}}$ of the tokens ``(N, T, d)``: with $u_{n,p} =
    z_{n,p} - \bar z_p$ (the mean over the stacks) and $\bar u_n$ the mean of
    stack $n$'s, $\sum \lVert u_{n,p} - \bar u_n \rVert^2 / \sum \lVert
    u_{n,p} \rVert^2$; 0 when a stack's tokens are all the same, $(T - 1)/T$
    for tokens independent of equal variance. A 0-d float64 tensor."""
    u = tokens.double() - tokens.double().mean(0)
    within = u - u.mean(1, keepdim=True)
    return within.square().sum() / u.square().sum()
