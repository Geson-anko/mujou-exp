"""The settings of the evaluation."""

from dataclasses import dataclass, field

__all__ = ["MeasureConf"]


@dataclass(kw_only=True)
class MeasureConf:
    """The evaluation: on each window of the clips, ``observed`` frames are
    shown (teacher forced) and the model predicts open loop up to the largest
    of ``horizons`` frames ahead."""

    observed: int = 16
    horizons: list[int] = field(default_factory=lambda: [1, 5, 10, 20])
    clips: int = 256  # per split
    windows: int = 4  # per clip
    every: float = 60.0  # seconds of virtual time between two evaluations
    # The vision of the "packed" layout (M17): the stacks after the first of an
    # evaluation window (one each vision.stride frames), the largest horizon of
    # the linear dynamics afterwards (task=posthoc).
    steps: int = 10
    # The dynamics program: the run diverged when roll10 exceeds diverge times
    # mean10. (The joint program stops on non-finite values only.)
    diverge: float = 10.0
    # The thresholds of a confirmation preset (the same values as the README),
    # by metric name; empty: nothing to judge.
    criteria: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.horizons or min(self.horizons) < 1:
            raise ValueError(f"horizons must be positive, not {self.horizons}")
        if self.steps < 0:
            raise ValueError(f"measure.steps must be >= 0, not {self.steps}")
