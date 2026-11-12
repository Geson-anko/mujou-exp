"""The settings of the vision (M15, M16, M17)."""

from dataclasses import dataclass, field

from exp.video.schema._check import check_choice

__all__ = [
    "LAYOUTS",
    "MAX_Z_DIM",
    "POOLS",
    "PRED_LOSSES",
    "RECON_DECODERS",
    "SIGREG_ON",
    "TOKEN_SIGREG",
    "MaskConf",
    "VisionConf",
]

# How z_k is read from the target view's tokens: the class token, the mean of
# the patch tokens, or the mean of ``registers`` register tokens.
POOLS = ("cls", "mean", "registers")
# What SIGReg sees of the target view: z, its patch tokens, or both.
SIGREG_ON = ("z", "tokens", "both")
# The decoder of the reconstruction: an MLP from z to all the pixels, or a
# spatial broadcast decoder (one small MLP of z and the pixel's coordinates).
RECON_DECODERS = ("mlp", "broadcast")
# How a chunk reaches the encoder: "pixels" (M15, M16), every view a whole chunk
# with the masked patches at 0 and the mask as a channel; "packed" (M17, as
# V-JEPA), one sequence per stream of the target view's tokens and the visible
# tokens of each context view, attention within each view only.
LAYOUTS = ("pixels", "packed")
# The prediction error of a token: squared (LeJEPA) or absolute (V-JEPA).
PRED_LOSSES = ("l2", "l1")
# The SIGReg of the target tokens of the packed layout (jepa.loss): of the
# tokens divided by T, not divided, or of the tokens less their stack's mean.
TOKEN_SIGREG = ("scaled", "raw", "clip")
MAX_Z_DIM = 1024  # the largest z (M17: about 2 % of the 128 x 128 x 3 input)


@dataclass(kw_only=True)
class MaskConf:
    """A context view of the ``"packed"`` layout (V-JEPA's multi-block mask):
    the union of ``blocks`` rectangles of the spatial grid, each of ``scale``
    of the grid's area and of aspect ratio (height / width) log-uniform in
    ``aspect``, masking the same patches in every frame (a tube). The visible
    patches are made exactly ``jepa.masks.expected_keep``, all outside the
    blocks (``jepa.masks``)."""

    blocks: int
    scale: float
    aspect: list[float] = field(default_factory=lambda: [0.75, 1.5])

    def __post_init__(self) -> None:
        if self.blocks < 1 or not 0 < self.scale <= 1:
            raise ValueError(
                f"a context mask needs blocks >= 1 and scale in (0, 1], not "
                f"{self.blocks} and {self.scale}"
            )
        if len(self.aspect) != 2 or not 0 < self.aspect[0] <= self.aspect[1]:
            raise ValueError(
                f"aspect must be [lo, hi] with 0 < lo <= hi: {self.aspect}"
            )


@dataclass(kw_only=True)
class VisionConf:
    r"""A ViT on chunks of the last ``frames`` frames, learned by a JEPA loss
    $L = (1 - \lambda) L_{pred} + \lambda\,\mathrm{SIGReg} + \mu L_{rec}$
    with $\lambda$ = ``sigreg`` and $\mu$ = ``recon``. The defaults are the
    tiny ViT (``vit_t``).

    $L_{rec}$ (off at ``recon`` 0, which builds no decoder and changes nothing)
    is the mean squared error per pixel of a decoder of $z$ (an MLP of one
    hidden layer of ``recon_hidden`` units, ``recon_decoder`` "mlp"; or the
    spatial broadcast decoder of Watters et al. 2019, "broadcast": z tiled
    over the pixels with their coordinates, through one MLP of two hidden
    layers of ``recon_hidden`` units applied per pixel) to the target view's
    last frame of the chunk, in the standardized pixels. The prediction of masked tokens
    alone drops what it cannot predict (the balls); the reconstruction is meant
    to keep them in $z$ (a hypothesis to test, not a result; ``jepa.loss``).

    Each chunk is shown as ``views`` views: view 0 the target (no mask), the
    others the context, with about ``mask_ratio`` of the patches masked in
    ``mask_blocks`` blocks (tubes through the frames for ``frames > 1``).
    The encoder sees the pixels standardized by ``pixel_mean`` and
    ``pixel_std`` (constants of the data, set by the experiment).

    The ``"packed"`` layout (M17; ``jepa.masks.pack_views``) stacks ``frames``
    frames ``stride`` video frames apart, cuts them into tubelets of
    ``tubelet`` frames x ``patch`` x ``patch`` pixels (one token each, 3-D
    sine-cosine positions), and has ``views - 1`` context views of
    ``context_masks`` whose masked tokens are dropped from the encoder's input
    (``mask_ratio`` and ``mask_blocks`` are unused). $z$ is the mean of the
    target view's tokens, then a linear map to ``z_dim`` (None: no map, $z$
    of the width ``dim``). ``pred_loss`` is the prediction error of a token,
    ``token_sigreg`` the SIGReg of the target tokens (``jepa.loss``).
    """

    patch: int = 4  # pixels per side of a patch
    dim: int = 128
    depth: int = 6  # transformer blocks of the encoder
    heads: int = 4
    nodes: int = 2  # the encoder's PC-ALM nodes, depth / nodes blocks each
    pred_depth: int = 2  # transformer blocks of the predictor
    frames: int = 1  # N
    views: int = 2
    mask_ratio: float = 0.5
    mask_blocks: int = 4
    sg: bool = False  # the target view gets no credit from the prediction
    sigreg: float = 0.05  # lambda; stronger with more streams (loss.jepa_loss)
    sigreg_directions: int = 256  # drawn anew each frame
    sigreg_on: str = "z"  # SIGREG_ON
    recon: float = 0.0  # mu, the weight of the pixel reconstruction from z
    recon_decoder: str = "mlp"  # RECON_DECODERS
    # the decoder's hidden units, with recon > 0; None: 256 for "mlp", 64 for
    # "broadcast" (it is applied to every pixel)
    recon_hidden: int | None = None
    pixel_mean: float = 0.0
    pixel_std: float = 1.0
    z_pool: str = "mean"
    registers: int = 4  # with z_pool "registers"
    # The joint program (M16) only: the encoder is the last checkpoint of the
    # M15 run the job names as its "parent" and never learns (the control).
    frozen: bool = False
    layout: str = "pixels"  # LAYOUTS
    tubelet: int = 0  # frames per token, "packed" only (0: all the frames)
    stride: int = 1  # video frames between two stacked frames, "packed" only
    z_dim: int | None = None  # "packed" only; None: dim, no projection
    context_masks: list[MaskConf] = field(default_factory=list)  # "packed" only
    pred_loss: str = "l2"  # PRED_LOSSES, "packed" only
    # TOKEN_SIGREG, "packed" only: the SIGReg of the target tokens (with
    # sigreg_on "tokens" or "both")
    token_sigreg: str = "scaled"

    def __post_init__(self) -> None:
        check_choice("vision.z_pool", self.z_pool, POOLS)
        check_choice("vision.sigreg_on", self.sigreg_on, SIGREG_ON)
        check_choice("vision.recon_decoder", self.recon_decoder, RECON_DECODERS)
        if not self.pixel_std > 0:
            raise ValueError(f"vision.pixel_std must be > 0, not {self.pixel_std}")
        if self.nodes < 1:
            raise ValueError(f"vision.nodes must be at least 1, not {self.nodes}")
        if self.depth % self.nodes:
            raise ValueError(
                f"depth {self.depth} must split evenly into {self.nodes} nodes"
            )
        if self.dim % self.heads or self.dim % 4:
            raise ValueError(
                f"vision.dim {self.dim} must be a multiple of 4 (the sine-cosine "
                f"positions) and of the heads {self.heads}"
            )
        if self.frames < 1 or self.views < 2 or self.mask_blocks < 1:
            raise ValueError(
                "vision needs frames >= 1, views >= 2 and mask_blocks >= 1, not "
                f"{self.frames}, {self.views} and {self.mask_blocks}"
            )
        if not 0 < self.mask_ratio < 1:
            raise ValueError(
                f"vision.mask_ratio must be in (0, 1), not {self.mask_ratio}"
            )
        if not self.recon >= 0:
            raise ValueError(f"vision.recon must be >= 0, not {self.recon}")
        if self.recon_hidden is not None and self.recon_hidden < 1:
            raise ValueError(
                f"vision.recon_hidden must be at least 1, not {self.recon_hidden}"
            )
        if not 0 <= self.sigreg <= 1:
            raise ValueError(f"vision.sigreg must be in [0, 1], not {self.sigreg}")
        check_choice("vision.layout", self.layout, LAYOUTS)
        check_choice("vision.pred_loss", self.pred_loss, PRED_LOSSES)
        check_choice("vision.token_sigreg", self.token_sigreg, TOKEN_SIGREG)
        if self.layout != "packed" and self.token_sigreg != "scaled":
            raise ValueError("vision.token_sigreg is of the packed layout only")
        if self.layout == "packed":
            self._check_packed()

    def _check_packed(self) -> None:
        if self.tubelet < 1 or self.frames % self.tubelet:
            raise ValueError(
                f"vision.tubelet {self.tubelet} must divide vision.frames "
                f"{self.frames} (packed)"
            )
        if self.stride < 1:
            raise ValueError(f"vision.stride must be at least 1, not {self.stride}")
        if self.views != 1 + len(self.context_masks):
            raise ValueError(
                f"vision.views {self.views} must be 1 + the {len(self.context_masks)}"
                " context_masks (packed)"
            )
        if self.z_pool != "mean":
            raise ValueError(f"packed pools the mean only, not {self.z_pool!r}")
        if self.dim % 16:
            raise ValueError(
                f"vision.dim {self.dim} must be a multiple of 16 (the 3-D positions)"
            )
        if self.z_dim is not None and not 1 <= self.z_dim <= MAX_Z_DIM:
            raise ValueError(f"vision.z_dim must be in [1, {MAX_Z_DIM}]: {self.z_dim}")
        if self.projected and self.sigreg_on == "tokens":
            raise ValueError(
                "sigreg_on 'tokens' with a projection (z_dim != dim) gives the "
                "projection no gradient: use 'z' or 'both'"
            )

    @property
    def projected(self) -> bool:
        """Whether $z$ is a linear map of the pooled tokens (``z_dim`` set and
        not ``dim``)."""
        return self.z_dim is not None and self.z_dim != self.dim

    @property
    def z_width(self) -> int:
        """The dimension of $z$: ``z_dim`` or ``dim``."""
        return self.z_dim if self.z_dim is not None else self.dim

    @property
    def decoder_hidden(self) -> int:
        """``recon_hidden``, or the default of ``recon_decoder``."""
        if self.recon_hidden is not None:
            return self.recon_hidden
        return 64 if self.recon_decoder == "broadcast" else 256
