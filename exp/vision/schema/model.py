"""The settings of the VAE's modules (V0: ``exp/vision/plans/v0-foundation.md``)."""

from dataclasses import dataclass, field

from exp.vision.schema._check import check_choice

__all__ = [
    "ARCHS",
    "BASE_WIDTH",
    "BLOCKS",
    "DEC_NODES",
    "ENC_NODES",
    "HEAD_DIM",
    "PARAMS",
    "SIZES",
    "ModelConf",
]

ARCHS = ("vit", "resnet")
SIZES = ("S", "M", "L")
BLOCKS = {"S": 1, "M": 2, "L": 4}  # blocks per node of the encoder
# The node counts do not depend on the arch or the size (a credit travels one
# node a step in the Jacobi run, so a count that changed would confound the
# effect of the size).
ENC_NODES = 4
DEC_NODES = 2
HEAD_DIM = 64  # of the ViT's heads when model.heads is not given
# The parametrizations: "sp", the initialization and learning rates as they
# are; "mup", muP of Adam's rules about BASE_WIDTH (exp.vision.models.mup_lr).
PARAMS = ("sp", "mup")
BASE_WIDTH = 256  # the width at which mup is sp


@dataclass(kw_only=True)
class ModelConf:
    """The encoder (``arch``, ``size``: blocks per node ``BLOCKS[size]``), the
    latent of ``latent`` dimensions and the decoder.

    The ViT has ``heads`` heads, by default (``None``) ``width / HEAD_DIM``
    (``head_count``). With ``scale_decoder`` the decoder's
    channels are ``decoder_channels`` times ``width / BASE_WIDTH``
    (``decoder_widths``). ``param`` is the parametrization (``PARAMS``):
    ``"mup"`` is of the ViT only, with heads of ``HEAD_DIM``, and not with
    OGD (``RunConf``)."""

    arch: str = "vit"
    size: str = "S"
    latent: int = 256  # D
    enc_nodes: int = ENC_NODES  # fixed
    dec_nodes: int = DEC_NODES  # fixed
    # ViT
    width: int = 256
    heads: int | None = None  # None: width / HEAD_DIM
    patch: int = 16
    # ResNet: the channels after the stem and each of the 3 further stages
    # (one per encoder node), chosen so that "M" has the ViT-M encoder's
    # parameters within 10 % at 8 frames of 64 px (exp.vision.models.param_table:
    # 9.44 M against 9.99 M); S and L use the same channels.
    resnet_channels: list[int] = field(default_factory=lambda: [48, 96, 192, 384])
    # The decoder: the Linear's channels at 4 x 4, then those after each of the
    # 4 doublings (4 -> 64).
    decoder_channels: list[int] = field(default_factory=lambda: [256, 128, 64, 32, 32])
    # The decoder's channels times width / BASE_WIDTH (V14).
    scale_decoder: bool = False
    param: str = "sp"  # of PARAMS
    # The likelihood's sigma^2 at the start: M18's ML variance of the stacks
    # about the mean stack (Walking Tours, 128 px; the 64 px value is close).
    sigma0_sq: float = 0.0743
    # Whether ln sigma is learned; when not, sigma^2 stays at sigma0_sq, which
    # then fixes the balance of the reconstruction against the KL (as the beta
    # = 2 sigma^2 of a beta-VAE at sigma = 1).
    learn_sigma: bool = True

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
        check_choice("model.param", self.param, PARAMS)
        if self.heads is None and self.width % HEAD_DIM:
            raise ValueError(
                f"model.width {self.width} is not a multiple of {HEAD_DIM}: "
                "give model.heads"
            )
        heads = self.head_count
        if self.latent < 1 or self.width < 1 or heads < 1 or self.patch < 1:
            raise ValueError("model.latent, width, heads and patch must be positive")
        if self.width % heads:
            raise ValueError(
                f"model.width {self.width} is not a multiple of heads {heads}"
            )
        if self.param == "mup" and (
            self.arch != "vit" or self.width != heads * HEAD_DIM
        ):
            raise ValueError(
                f"model.param mup is of the vit with heads of {HEAD_DIM}, not "
                f"{self.arch} of width {self.width} and {heads} heads"
            )
        if len(self.resnet_channels) != 4 or min(self.resnet_channels) < 1:
            raise ValueError("model.resnet_channels needs 4 positive channels")
        if len(self.decoder_channels) != 5 or min(self.decoder_channels) < 1:
            raise ValueError("model.decoder_channels needs 5 positive channels")
        if self.scale_decoder and any(
            c * self.width % BASE_WIDTH for c in self.decoder_channels
        ):
            raise ValueError(
                f"model.decoder_channels {self.decoder_channels} times width "
                f"{self.width} / {BASE_WIDTH} are not whole"
            )

    @property
    def head_count(self) -> int:
        """The ViT's heads: ``heads``, or ``width / HEAD_DIM`` without it."""
        return self.width // HEAD_DIM if self.heads is None else self.heads

    @property
    def decoder_widths(self) -> list[int]:
        """The decoder's channels: ``decoder_channels``, times ``width /
        BASE_WIDTH`` with ``scale_decoder``."""
        if not self.scale_decoder:
            return list(self.decoder_channels)
        return [c * self.width // BASE_WIDTH for c in self.decoder_channels]

    @property
    def blocks(self) -> int:
        """Blocks per encoder node."""
        return BLOCKS[self.size]
