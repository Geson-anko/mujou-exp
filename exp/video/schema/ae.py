"""The settings of the autoencoder and the VAE of M18 (``program="ae"`` and
``"vae"``): the decoder that learns with the encoder of ``vision`` inside the
system, and the VAE's initial likelihood."""

from dataclasses import dataclass

__all__ = ["AEConf"]


@dataclass(kw_only=True)
class AEConf:
    """The decoder of the autoencoder (``exp.video.vae.modules.TokenDecoder``):
    the encoder of ``vision`` turned around, ``decoder_depth`` transformer
    blocks of the encoder's width and heads (M18: 2, one node of the encoder,
    lighter than its 8 so that the reconstruction measures the code rather
    than the decoder's capacity; plans/m18-vae-stream.md)."""

    decoder_depth: int = 2
    # The VAE's sigma^2 at the start (program "vae", exp.video.vae.modules
    # .build_vae): sigma_0^2, the ML variance of the mean stack's Gaussian of
    # the dequantized training stacks (plans/m18-vae-stream.md). None for the
    # autoencoder, which has no sigma.
    sigma0_sq: float | None = None

    def __post_init__(self) -> None:
        if self.decoder_depth < 0:
            raise ValueError(
                f"ae.decoder_depth must be at least 0, not {self.decoder_depth}"
            )
        if self.sigma0_sq is not None and not self.sigma0_sq > 0:
            raise ValueError(f"ae.sigma0_sq must be > 0, not {self.sigma0_sq}")
