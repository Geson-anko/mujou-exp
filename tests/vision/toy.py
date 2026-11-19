"""A toy ``loop.Program`` whose numbers are known in advance: unit ``u`` adds
``u`` to a sum, a line of ``stream.jsonl`` is the mean of the units since the
last line, and an evaluation is ``{"v": sum}``. It needs no mujou and no data."""

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from torch import nn

from exp.vision.loop import Schedule, every

__all__ = ["PROGRAM", "Toy", "ToyProgram", "schedule_of"]


def schedule_of(conf) -> Schedule:
    """The schedule a program derives from ``conf`` (``RunConf``)."""
    frame, ctl, m = conf.time.frame, conf.ctl, conf.measure
    return Schedule(
        units=round(m.duration / frame),
        seconds=frame,
        log=every(ctl.log_every, frame),
        evaluate=every(m.every, frame),
        ckpt_at=tuple(round(t / frame) for t in m.ckpt_times),
        resume=every(ctl.resume_every, frame),
        test=ctl.eval_test,
    )


class Toy:
    """A ``loop.Built`` of plain numbers (module docstring)."""

    def __init__(
        self, schedule: Schedule, resume: Mapping[str, Any] | None, fail: dict
    ) -> None:
        self._schedule = schedule
        self.fail = fail
        self.u, self.total, self.since, self.n = 0, 0.0, 0.0, 0
        if resume is not None:
            h = resume["host"]
            self.u, self.total, self.since, self.n = (
                h["u"],
                h["total"],
                h["since"],
                h["n"],
            )
        self.lin = nn.Linear(1, 1)

    @property
    def schedule(self) -> Schedule:
        return self._schedule

    def advance(self) -> None:
        self.u += 1
        if self.fail.get("raise_at") == self.u:
            raise self.fail.get("error", RuntimeError("the toy failed"))
        self.total += self.u
        self.since += self.u
        self.n += 1

    def log(self) -> dict[str, float | None]:
        x = self.since / self.n
        if self.fail.get("nan_at") == self.u:
            x = math.nan
        self.since, self.n = 0.0, 0
        return {"x": x}

    def evaluate(self, split: str) -> dict[str, float]:
        row = {"v": self.total + (0.5 if split == "test" else 0.0), "gap": math.nan}
        if split == "val":
            row["objective"] = -self.total
        return row

    def diverged(self, row: Mapping[str, float]) -> str | None:
        limit = self.fail.get("diverge_above")
        return None if limit is None or row["v"] <= limit else f"v {row['v']} > {limit}"

    def objective(
        self, row: Mapping[str, Any] | None, *, diverged: bool
    ) -> float | None:
        return 99.0 if diverged else (None if row is None else row["objective"])

    def modules(self) -> dict[str, nn.Module]:
        return {"lin": self.lin}

    def visualize(self, out: Path) -> None:
        out.mkdir(parents=True)
        (out / "seen.txt").write_text(f"{self.u}\n")

    def rows(self) -> dict[str, list[dict[str, Any]]]:
        return {}

    def resume_state(self) -> tuple[Mapping[str, Any], dict[str, Any]]:
        return {}, {"u": self.u, "total": self.total, "since": self.since, "n": self.n}

    def final(self) -> dict[str, dict[str, Any]]:
        return {"extra": {"u": self.u}}

    def close(self) -> None:
        pass


class ToyProgram:
    """The ``loop.Program`` of ``Toy``. ``schedule`` is the fixed one of the
    loop tests; ``None`` derives it from the config as a program does (``every``
    of ``ctl`` and ``measure``, ``ckpt_at`` from ``measure.ckpt_times``)."""

    def __init__(self, stop_at: int | None = None, **fail) -> None:
        self.stop_at = stop_at
        self.fail = fail
        self.schedule: Schedule | None = Schedule(
            units=12,
            unit="frame",
            seconds=0.5,
            log=2,
            evaluate=4,
            ckpt=6,
            viz=6,
            resume=4,
            test=True,
        )

    def build(self, conf, run_dir: Path, resume) -> Toy:
        toy = Toy(self.schedule or schedule_of(conf), resume, self.fail)
        if self.stop_at is not None:
            advance, stop_at = toy.advance, self.stop_at

            def stopping() -> None:
                advance()
                if toy.u == stop_at:
                    (run_dir / "stop.request").touch()

            toy.advance = stopping  # type: ignore[method-assign]
        return toy


PROGRAM = ToyProgram()
PROGRAM.schedule = None
