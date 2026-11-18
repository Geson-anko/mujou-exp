"""The settings of the stream: the Walking Tours stacks."""

from dataclasses import dataclass, field

__all__ = ["DataConf", "ToursConf"]


@dataclass(kw_only=True)
class ToursConf:
    """The prepared Walking Tours (``exp.lib.tours.walking_tours``) of
    ``size`` x ``size`` at ``fps``, shown as stacks of ``frames`` frames
    ``stride`` video frames apart, ``advance`` video frames further at each
    frame of the run (``time.frame``); ``held_out`` is the video kept for
    evaluation only. ``data_version`` is read by ``exp.lib.provenance``."""

    data_version: str = "ad6e56b5995e"
    held_out: str = "Zurich"
    size: int = 64
    fps: float = 10.0
    frames: int = 8  # n, the stack
    stride: int = 1
    advance: int = 1

    def __post_init__(self) -> None:
        for name in ("size", "frames", "stride", "advance"):
            if getattr(self, name) < 1:
                raise ValueError(f"data.tours.{name} must be at least 1")
        if not self.fps > 0:
            raise ValueError(f"data.tours.fps must be > 0, not {self.fps}")


@dataclass(kw_only=True)
class DataConf:
    """``streams`` streams side by side (the batch B) over the Walking Tours."""

    streams: int = 1
    tours: ToursConf = field(default_factory=ToursConf)

    def __post_init__(self) -> None:
        if self.streams < 1:
            raise ValueError(f"streams must be at least 1, not {self.streams}")
