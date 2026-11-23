r"""The JEPA of V18 and V18b (``plans/jepa.md``, 「V18」「V18b」): the masks,
the modules and the loss, in LeJEPA's form (one encoder, no EMA target, no
stop-gradient).

The encoder is the ViT of ``build_vae`` (the same steps and seed, so its
blocks start as the VAE's) without the head: a stack enters by
``early_fusion`` and leaves as $T = g^2$ tokens $h_p$ of the width $w$ after
the last ``LayerNorm`` ($g$ = size / patch; 4 x 4 at 64 px and patch 16,
V18's; 16 x 16 at patch 4, V18b's). The projection $P$ (``Linear(w, d)``,
shared by the positions; $d$ = ``TOKEN_DIM`` = 64 by default) maps them to
$z_p = P h_p$, and $z = (z_1, \dots, z_T)$ flattened (row-major positions) is
the code of a stack, $D = T d$ (1024 in V18).

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
(no buffer): the flat $z$ ``(B, T d)`` (``"flat"``, V18's) or the mean token
$\bar z = \frac1T \sum_p z_p$ ``(B, d)`` (``"mean"``, V18b's arm M;
``sigreg_points``). The gradient reaches the target $z$ too.

muP (``jepa_mup_roles``, ``jepa_mup_lr``): the encoder's roles are the
VAE's; $P$ and ``embed`` map the encoder's width to a fixed size (output,
$k_w^{-1}$ with $k_w$ = width / ``BASE_WIDTH``); the predictor's blocks and
``out`` are of the fixed width 256 ($k$ = 1).
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import override

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

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
    "PREDICTOR_SEED",
    "PRED_BLOCKS",
    "PRED_WIDTH",
    "PROJ_SEED",
    "SIGREG_ON",
    "TOKEN_DIM",
    "JEPAModules",
    "Predictor",
    "TokenEncoder",
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
    "take",
]

TOKEN_DIM = 64  # d: the code of a token (V18's; V18b's arm F takes 4)
COARSE_GRID = 4  # G: the grid of the masks' family (V18's, patch 16 at 64 px)
SIGREG_ON = ("flat", "mean")  # SIGReg's data point: z (V18's) or z_bar
PRED_WIDTH, PRED_BLOCKS = 256, 4  # the predictor's width and blocks
PROJ_SEED = 21  # added to the seed: the projection's initial weights
PREDICTOR_SEED = 23  # added to the seed: the predictor's initial weights


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
    def forward(self, context: Tensor, keep: Tensor, hide: Tensor) -> Tensor:
        """$\\hat z$ ``(B, k, out_dim)`` at the positions ``hide`` ``(B, k)``
        from the context side's tokens ``context`` ``(B, V, width_in)`` at
        ``keep`` ``(B, V)``."""
        c = self.embed(context) + self.pos[keep]  # (B, V, dim)
        m = self.mask_token + self.pos[hide]  # (B, k, dim)
        x = torch.cat([c, m], 1)
        for block in self.blocks:
            x = block(x)
        return self.out(self.norm(x[:, keep.shape[1] :]))


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
    the visible ones), ``proj`` (a token to $z_p$) and ``predictor``
    (``Predictor``'s call)."""

    encoder: nn.Module
    proj: nn.Linear
    predictor: nn.Module

    def modules(self) -> dict[str, nn.Module]:
        """The learned modules by the names of the checkpoints."""
        return {"encoder": self.encoder, "proj": self.proj, "predictor": self.predictor}

    def parameters(self) -> list[nn.Parameter]:
        """The learned parameters, the encoder's first."""
        return [p for m in self.modules().values() for p in m.parameters()]

    def tokens(self, stacks: Tensor) -> Tensor:
        """$z_p$ ``(B, T, d)`` of the stacks ``(B, n, 3, H, W)`` (u8, or floats
        in ``[0, 1]``): the target side."""
        return self.proj(self.encoder(early_fusion(stacks)))

    def mu(self, stacks: Tensor) -> Tensor:
        """The code $z$ ``(B, T d)`` of the stacks (the evaluations')."""
        return self.tokens(stacks).flatten(1)


def build_jepa(
    conf: RunConf,
    *,
    seed: int,
    token_dim: int = TOKEN_DIM,
    checkpoint_blocks: bool = False,
) -> JEPAModules:
    r"""The JEPA of ``conf.model`` (a ViT) on ``conf.device``, $d$ =
    ``token_dim``: the encoder of ``build_vae(conf, seed=seed)`` (muP's
    initialization included) without its head (``TokenEncoder`` of
    ``checkpoint_blocks``); $P$ drawn from ``seed + PROJ_SEED`` and the
    predictor from ``seed + PREDICTOR_SEED`` (on the CPU, the global RNG left
    as it was). In muP the output matrices $P$ and ``embed`` are drawn of
    LeCun's std times $k_w^{-1/2}$. The encoder does not depend on
    ``token_dim``.

    Raises:
        ValueError: ``model.arch`` is not ``"vit"``.
    """
    m, t = conf.model, conf.data.tours
    if m.arch != "vit":
        raise ValueError(f"the JEPA's encoder is the ViT, not {m.arch!r}")
    parts = list(build_vae(conf, seed=seed).encoder.parts)
    vit = [p for p in parts if isinstance(p, ViTPart)]
    vit[-1].head = None
    grid = t.size // m.patch
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + PROJ_SEED)
        proj = nn.Linear(m.width, token_dim)
        _trunc_normal_(proj.weight, 1 / math.sqrt(m.width))
        nn.init.zeros_(proj.bias)
        torch.manual_seed(seed + PREDICTOR_SEED)
        predictor = Predictor(width_in=m.width, grid=grid, out_dim=token_dim)
        _init_predictor(predictor)
    if m.param == "mup":
        k = m.width / BASE_WIDTH
        with torch.no_grad():
            proj.weight.mul_(k**-0.5)
            predictor.embed.weight.mul_(k**-0.5)
    dev = torch.device(conf.device)
    encoder = TokenEncoder(vit, checkpoint_blocks=checkpoint_blocks)
    return JEPAModules(encoder, proj.to(dev), predictor.to(dev))


# -- muP ------------------------------------------------------------------------


def jepa_mup_roles(mods: JEPAModules) -> dict[Tensor, str]:
    """The role in muP of each parameter (``exp.vision.models.ROLES``): the
    patch embedding's matrix is ``"input"``; $P$'s, ``embed``'s and ``out``'s
    ``"output"``; the other matrices ``"hidden"``; the biases, the norms and
    the mask token ``"vector"``.

    Raises:
        ValueError: The modules are not ``build_jepa``'s.
    """
    pred = mods.predictor
    embeds = [m for m in mods.encoder.modules() if isinstance(m, PatchEmbed)]
    if not isinstance(pred, Predictor) or len(embeds) != 1:
        raise ValueError("muP's roles are of build_jepa's modules")
    inputs = {id(embeds[0].proj.weight)}
    outputs = {id(mods.proj.weight), id(pred.embed.weight), id(pred.out.weight)}
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


def sigreg_points(tokens: Tensor, on: str) -> Tensor:
    r"""SIGReg's data points of the tokens $z_p$ ``(B, T, d)``: the flat $z$
    ``(B, T d)`` (``on`` ``"flat"``) or the mean token $\bar z = \frac1T
    \sum_p z_p$ ``(B, d)`` (``"mean"``).

    Raises:
        ValueError: ``on`` is not of ``SIGREG_ON``.
    """
    if on == "flat":
        return tokens.flatten(1)
    if on == "mean":
        return tokens.mean(1)
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
    SIGReg's data points (``sigreg_points`` of ``sigreg_on``; module
    docstring): ``loss`` $L$, ``pred`` $L_{\mathrm{pred}}$, ``sigreg`` $S$,
    and the target side's ``tokens`` $z_p$ ``(B, T, d)`` (detached)."""
    images = early_fusion(x)
    tokens = mods.proj(mods.encoder(images))  # (B, T, d), the target side
    z_hat, hide = predict(mods, images, mask)
    pred = (z_hat - take(tokens, hide)).square().mean()
    z = sigreg_points(tokens, sigreg_on)  # (B, D_S)
    s = sigreg(z, z[:0], dirs)
    return {
        "loss": (1 - lam) * pred + lam * s,
        "pred": pred,
        "sigreg": s,
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
