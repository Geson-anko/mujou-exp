"""The settings of the vision (M15, M16)."""

from dataclasses import dataclass

from exp.video.schema._check import check_choice

__all__ = ["POOLS", "SIGREG_ON", "VisionConf"]

# How z_k is read from the target view's tokens: the class token, the mean of
# the patch tokens, or the mean of ``registers`` register tokens.
POOLS = ("cls", "mean", "registers")
# What SIGReg sees of the target view: z, its patch tokens, or both.
SIGREG_ON = ("z", "tokens", "both")


@dataclass(kw_only=True)
class VisionConf:
    r"""A ViT on chunks of the last ``frames`` frames, learned by a JEPA loss
    $L = (1 - \lambda) L_{pred} + \lambda\,\mathrm{SIGReg}$ with
    $\lambda$ = ``sigreg``. The defaults are the tiny ViT (``vit_t``).

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
        if not 0 <= self.sigreg <= 1:
            raise ValueError(f"vision.sigreg must be in [0, 1], not {self.sigreg}")
