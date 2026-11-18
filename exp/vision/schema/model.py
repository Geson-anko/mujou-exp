"""The settings of the VAE's modules (V0: ``exp/vision/plans/v0-foundation.md``)."""

from dataclasses import dataclass, field

from exp.vision.schema._check import check_choice

__all__ = ["ARCHS", "BLOCKS", "ENC_NODES", "DEC_NODES", "SIZES", "ModelConf"]

ARCHS = ("vit", "resnet")
SIZES = ("S", "M", "L")
BLOCKS = {"S": 1, "M": 2, "L": 4}  # blocks per node of the encoder
# The node counts do not depend on the arch or the size (a credit travels one
# node a step in the Jacobi run, so a count that changed would confound the
# effect of the size).
ENC_NODES = 4
DEC_NODES = 2


@dataclass(kw_only=True)
class ModelConf:
    """The encoder (``arch``, ``size``: blocks per node ``BLOCKS[size]``), the
    latent of ``latent`` dimensions and the decoder."""

    arch: str = "vit"
    size: str = "S"
    latent: int = 256  # D
    enc_nodes: int = ENC_NODES  # fixed
    dec_nodes: int = DEC_NODES  # fixed
    # ViT
    width: int = 256
    heads: int = 4
    patch: int = 16
    # ResNet: the channels after the stem and each of the 3 further stages
    # (one per encoder node), chosen so that "M" has the ViT-M encoder's
    # parameters within 10 % at 8 frames of 64 px (exp.vision.models.param_table:
    # 9.44 M against 9.99 M); S and L use the same channels.
    resnet_channels: list[int] = field(default_factory=lambda: [48, 96, 192, 384])
    # The decoder: the Linear's channels at 4 x 4, then those after each of the
    # 4 doublings (4 -> 64).
    decoder_channels: list[int] = field(default_factory=lambda: [256, 128, 64, 32, 32])
    # The likelihood's sigma^2 at the start: M18's ML variance of the stacks
    # about the mean stack (Walking Tours, 128 px; the 64 px value is close).
    sigma0_sq: float = 0.0743

    def __post_init__(self) -> None:
        check_choice("model.arch", self.arch, ARCHS)
        check_choice("model.size", self.size, SIZES)
        if self.enc_nodes != ENC_NODES or self.dec_nodes != DEC_NODES:
            raise ValueError(
                f"the node counts are fixed at {ENC_NODES} and {DEC_NODES}, not "
                f"{self.enc_nodes} and {self.dec_nodes}"
            )
        if not self.sigma0_sq > 0:
            raise ValueError(f"model.sigma0_sq must be > 0, not {self.sigma0_sq}")
        if self.latent < 1 or self.width < 1 or self.heads < 1 or self.patch < 1:
            raise ValueError("model.latent, width, heads and patch must be positive")
        if self.width % self.heads:
            raise ValueError(
                f"model.width {self.width} is not a multiple of heads {self.heads}"
            )
        if len(self.resnet_channels) != 4 or min(self.resnet_channels) < 1:
            raise ValueError("model.resnet_channels needs 4 positive channels")
        if len(self.decoder_channels) != 5 or min(self.decoder_channels) < 1:
            raise ValueError("model.decoder_channels needs 5 positive channels")

    @property
    def blocks(self) -> int:
        """Blocks per encoder node."""
        return BLOCKS[self.size]
