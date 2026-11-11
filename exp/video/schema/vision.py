"""The settings of the vision (M15, M16)."""

from dataclasses import dataclass

from exp.video.schema._check import check_choice

__all__ = ["POOLS", "VisionConf"]

# How z_k is read from the target view's tokens: the class token, the mean of
# the patch tokens, or the mean of ``registers`` register tokens.
POOLS = ("cls", "mean", "registers")


@dataclass(kw_only=True)
class VisionConf:
    r"""A ViT on chunks of the last ``frames`` frames, learned by a JEPA loss
    $L = (1 - \lambda) L_{pred} + \lambda\,\mathrm{SIGReg}$ with
    $\lambda$ = ``sigreg``. The defaults are the tiny ViT (``vit_t``).

    Each chunk is shown as ``views`` views: view 0 the target (no mask), the
    others the context, with ``mask_ratio`` of the patches masked in
    ``mask_blocks`` blocks (tubes through the frames for ``frames > 1``).
    """

    patch: int = 4  # pixels per side of a patch
    dim: int = 192
    depth: int = 12  # transformer blocks of the encoder
    heads: int = 3
    nodes: int = 3  # the encoder's PC-ALM nodes, depth / nodes blocks each
    pred_depth: int = 4  # transformer blocks of the predictor
    frames: int = 1  # N
    views: int = 2
    mask_ratio: float = 0.5
    mask_blocks: int = 4
    sg: bool = False  # the target view gets no credit from the prediction
    sigreg: float = 0.05  # lambda
    sigreg_directions: int = 256  # drawn anew each frame
    z_pool: str = "mean"
    registers: int = 4  # with z_pool "registers"

    def __post_init__(self) -> None:
        check_choice("vision.z_pool", self.z_pool, POOLS)
        if self.nodes < 1:
            raise ValueError(f"vision.nodes must be at least 1, not {self.nodes}")
        if self.depth % self.nodes:
            raise ValueError(
                f"depth {self.depth} must split evenly into {self.nodes} nodes"
            )
