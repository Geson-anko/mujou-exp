"""Small configs of the packed layout (M17) and a small finalized Walking Tours
for the tests."""

import math
from pathlib import Path
from typing import Any

import torch

from exp.lib.video_io import U8Writer
from exp.video.data.walking_tours import finalize
from exp.video.schema import (
    DataConf,
    LearnerConf,
    MeasureConf,
    OptimConf,
    RunConf,
    TimeConf,
    ToursConf,
    VisionConf,
)
from exp.video.schema.vision import MaskConf

SIDE = 16  # 4 x 4 patches of 4
SMALL: dict[str, Any] = {
    "layout": "packed",
    "patch": 4,
    "dim": 16,
    "depth": 2,
    "heads": 2,
    "nodes": 2,
    "pred_depth": 1,
    "frames": 4,
    "tubelet": 2,
    "views": 3,
    "sigreg_directions": 16,
}


def masks() -> list[MaskConf]:
    return [MaskConf(blocks=2, scale=0.2), MaskConf(blocks=1, scale=0.5)]


def _with_masks() -> dict[str, Any]:
    return {"context_masks": masks()}


def packed_conf(
    learner: str = "pc",
    *,
    streams: int = 2,
    learn_interval: float | None = math.inf,
    data_version: str = "none",
    **vision: Any,
) -> RunConf:
    """A run of the packed layout on 16 x 16 frames of Walking Tours (held out
    ``H``), the small ViT of ``SMALL`` updated by ``vision``."""
    return RunConf(
        name="test",
        program="vision",
        learner=LearnerConf(kind=learner),
        data=DataConf(
            streams=streams,
            balls=None,
            tours=ToursConf(data_version=data_version, held_out="H", size=SIDE),
        ),
        vision=VisionConf(**(_with_masks() | SMALL | vision)),
        optim_vision=OptimConf(kind="sgd", lr=0.1, warmup=0.0),
        time=TimeConf(learn_interval=learn_interval),
        measure=MeasureConf(horizons=[1, 2], steps=2, clips=2, windows=2),
    )


def stacks(n: int, frames: int = 4, seed: int = 0) -> torch.Tensor:
    """Random u8 stacks ``(n, frames, 3, SIDE, SIDE)``."""
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (n, frames, 3, SIDE, SIDE), generator=g).to(
        torch.uint8
    )


def smooth_tours(root: Path, lengths: dict[str, int], size: int = SIDE) -> Path:
    """Finalized data of videos whose frames drift smoothly: frame j of a video
    is a fixed random image of the video rolled by j // 2 pixels, plus j."""
    g = torch.Generator().manual_seed(0)
    for name, n in sorted(lengths.items()):
        base = torch.randint(0, 200, (3, size, size), generator=g)
        meta = {"video": name, "fps": 30.0, "size": size}
        with U8Writer(root / name, (3, size, size), meta) as w:
            for j in range(n):
                frame = base.roll(j // 2, 2) + j % 50
                w.append(frame.clamp(0, 255).to(torch.uint8))
    return finalize(root, expected=list(lengths))


def tiny(version: str) -> list[str]:
    """The overrides of a run of ``m17/wt_n16`` (BP) of a second on 16 x 16
    frames of the data ``version`` (held out ``H``) with the small ViT."""
    return [
        "+experiment=m17/wt_n16",
        f"data.tours.data_version={version}",
        "data.tours.held_out=H",
        "data.tours.size=16",
        "data.streams=2",
        "seconds=1",
        "measure.clips=2",
        "measure.windows=2",
        "measure.every=0.5",
        "measure.steps=2",
        "measure.horizons=[1,2]",
        "ctl.ckpt_every=0.5",
        "vision.patch=4",
        "vision.dim=16",
        "vision.depth=2",
        "vision.nodes=2",
        "vision.heads=2",
        "vision.pred_depth=1",
        "vision.frames=4",
        "vision.stride=2",
        "vision.sigreg_directions=16",
        "learner=bp",
    ]
