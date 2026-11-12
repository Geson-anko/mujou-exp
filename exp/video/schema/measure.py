"""The settings of the evaluation."""

from dataclasses import dataclass, field

__all__ = ["ENCODER_OBJECTIVES", "POSTHOC_CODES", "MeasureConf"]

# The objectives of the posthoc evaluation of the packed layout (M17,
# measure.encoder_eval): "recon_dyn", the worse of the reconstruction and the
# linear dynamics against the PCA; "completion" (provisional, 2026-11-12: the
# formula is not decided yet), the mean completion error on val.
ENCODER_OBJECTIVES = ("recon_dyn", "completion")
# The code of a stack in the posthoc evaluation of the packed layout (M17,
# measure.encoder_eval): "z", the pooled z; "tokens", the target tokens of the
# stack flattened (T x the token width; vision.token_dim, 2026-11-12).
POSTHOC_CODES = ("z", "tokens")


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
    # The objective of the posthoc evaluation of the packed layout
    # (ENCODER_OBJECTIVES).
    encoder_objective: str = "recon_dyn"
    posthoc_code: str = "z"  # POSTHOC_CODES

    def __post_init__(self) -> None:
        if not self.horizons or min(self.horizons) < 1:
            raise ValueError(f"horizons must be positive, not {self.horizons}")
        if self.steps < 0:
            raise ValueError(f"measure.steps must be >= 0, not {self.steps}")
        if self.encoder_objective not in ENCODER_OBJECTIVES:
            raise ValueError(
                f"measure.encoder_objective must be one of {ENCODER_OBJECTIVES}, "
                f"not {self.encoder_objective!r}"
            )
        if self.posthoc_code not in POSTHOC_CODES:
            raise ValueError(
                f"measure.posthoc_code must be one of {POSTHOC_CODES}, not "
                f"{self.posthoc_code!r}"
            )
