"""The settings of the autoencoder of M18 (``program="ae"``): the decoder that
learns with the encoder of ``vision`` inside the system."""

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

    def __post_init__(self) -> None:
        if self.decoder_depth < 0:
            raise ValueError(
                f"ae.decoder_depth must be at least 0, not {self.decoder_depth}"
            )
