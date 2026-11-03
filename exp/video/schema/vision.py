"""The settings of the vision (M15, M16)."""

from dataclasses import dataclass

from exp.video.schema._check import check_choice

__all__ = ["POOLS", "RECON_DECODERS", "SIGREG_ON", "VisionConf"]

# How z_k is read from the target view's tokens: the class token, the mean of
# the patch tokens, or the mean of ``registers`` register tokens.
POOLS = ("cls", "mean", "registers")
# What SIGReg sees of the target view: z, its patch tokens, or both.
SIGREG_ON = ("z", "tokens", "both")
# The decoder of the reconstruction: an MLP from z to all the pixels, or a
# spatial broadcast decoder (one small MLP of z and the pixel's coordinates).
RECON_DECODERS = ("mlp", "broadcast")


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

    @property
    def decoder_hidden(self) -> int:
        """``recon_hidden``, or the default of ``recon_decoder``."""
        if self.recon_hidden is not None:
            return self.recon_hidden
        return 64 if self.recon_decoder == "broadcast" else 256
