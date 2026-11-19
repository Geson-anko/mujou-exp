"""The settings of the evaluation."""

from dataclasses import dataclass, field

__all__ = ["MeasureConf"]


@dataclass(kw_only=True)
class MeasureConf:
    """The evaluation of V0: the held-out stacks of the video ``data.tours.
    held_out`` (``clips`` x ``windows`` per split; calib, val and test are the
    three equal parts in time of ``walking_tours.eval_windows``), the
    checkpoints (``ckpt_times``, seconds of virtual time; 0 is the
    initialization; with ``ckpt_every`` > 0 they are instead every
    ``ckpt_every`` seconds from 0 to ``duration``) and the posthoc evaluation of each (``recon_blur`` with
    ``blur_sigma`` pixels, ``posthoc_seeds`` seeds of the decoder it learns),
    and with ``grad_stats`` the statistics of every node's gradients."""

    duration: float = 1800.0  # the length of the run
    ckpt_times: list[float] = field(
        default_factory=lambda: [0.0, 60.0, 300.0, 600.0, 1200.0, 1800.0]
    )
    # > 0: checkpoints every this many seconds from 0 to duration, in place of
    # ckpt_times (the matrix of python -m exp.vision.stream_eval).
    ckpt_every: float = 0.0
    # Seconds between two evaluations on val during a run, by the program's
    # measures (exp.vision.program.Measures). V0's PROGRAM has none: its
    # online runs write no evals.jsonl rows, and the numbers come from the
    # posthoc evaluation of the checkpoints (python -m exp.vision.eval).
    every: float = 60.0
    clips: int = 256  # per split
    windows: int = 4  # per clip
    blur_sigma: float = 1.0  # pixels (64 px; 128 px used 2)
    posthoc_seeds: int = 4
    # The thresholds of a confirmation preset by metric name; empty: nothing to
    # judge (judged by the loop on the last online row of "test", else "val":
    # passed when below, not judged without the value; so, as every, not for
    # V0's posthoc numbers).
    criteria: dict[str, float] = field(default_factory=dict)
    # Wrap every node's optimizer in exp.lib.grad_stats.GradStats: grad.jsonl
    # (a row a second per node). The learning is the same bit for bit.
    grad_stats: bool = False

    def __post_init__(self) -> None:
        if not self.duration > 0:
            raise ValueError(f"measure.duration must be > 0, not {self.duration}")
        if self.ckpt_every < 0:
            raise ValueError(f"measure.ckpt_every must be >= 0, not {self.ckpt_every}")
        if self.ckpt_every > 0:
            count = round(self.duration / self.ckpt_every)
            if abs(count * self.ckpt_every - self.duration) > 1e-9 * self.duration:
                raise ValueError(
                    f"measure.duration {self.duration} is not a multiple of "
                    f"measure.ckpt_every {self.ckpt_every}"
                )
            self.ckpt_times = [i * self.ckpt_every for i in range(count + 1)]
        times = self.ckpt_times
        if times != sorted(set(times)) or (times and times[0] < 0):
            raise ValueError(f"measure.ckpt_times must increase from >= 0: {times}")
        if times and times[-1] > self.duration:
            raise ValueError(
                f"measure.ckpt_times {times} go beyond measure.duration {self.duration}"
            )
        if self.clips < 3 or self.windows < 1 or self.posthoc_seeds < 1:
            raise ValueError("measure.clips >= 3, windows >= 1, posthoc_seeds >= 1")
