r"""The task ``"posthoc"`` of the vision program (design 8.5): the decoder
learned afterwards (``exp.video.measure.posthoc``) of each evaluation
checkpoint of a finished vision run, the parent, as a job of the fleet::

    python -m exp.video.main --job <run_dir>/job.json --attempt <n>

for a job of ``program="vision"`` or ``"joint"`` (M16: the lines also have the
image keys of the dynamics' rollouts and of the dream, ``measure.joint``), ``task="posthoc"``, whose top-level
``"parent"`` names the parent's run directory: ``"<batch>/<run_id>"``,
relative to the runs' root of the worker (``<run_dir>/../..``), or an
absolute path. The parent must be on this host (submit the job ``after``
the parent and on its ``host``, while its ``ckpt/`` is here). An optional
``"decode": {"frames", "steps"}`` replaces ``FRAMES`` and ``STEPS``. The
device is the job's ``config.device``.

The loop drives it a checkpoint a unit (``unit="ckpt"``, so that the
heartbeat moves after each and a stop resumes at the next): it writes the
files of ``measure.posthoc`` into ``post/decode/`` of the job's own run
directory, which the worker copies to the NAS with the rest; the job's
``provenance/`` and ``CODE.json`` are of the code that decoded, and
``post/decode/config.json`` names the parent's ``run_id``, ``group`` and
``code``. ``result.json``'s ``final`` has ``"decode"``: the last line of each
split. There is no objective.

For a parent of the packed layout (M17) the job runs ``measure.encoder_eval``
instead: unit 0 the yardsticks (the pixel PCA and the ViT before learning,
``baselines.jsonl``), then each checkpoint (``"decode": {"ckpts": "last"}``
evaluates the last one only; ``"windows"`` replaces ``WINDOWS``), and
``final`` has ``"decode"`` (the last line of each split, the val one with the
objective) and ``"baselines"``. ``task="baselines"`` of the vision program is
unit 0 alone, of the job's own config (no parent).

By hand, on this machine, into ``<run>/post/decode/`` of the parent itself::

    python -m exp.video.programs.posthoc --run <run_dir> --device cpu
"""

import argparse
import json
import os
import sys
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from torch import Tensor, nn

from exp.lib.provenance import assert_free_threaded, configure_determinism
from exp.lib.rundir import EnvError, Exit
from exp.video.loop import Schedule
from exp.video.measure.encoder_eval import WINDOWS, EncoderEval
from exp.video.measure.joint import DREAM, ball_windows
from exp.video.measure.posthoc import FRAMES, STEPS, Decoding, Parent
from exp.video.programs.vision import STRIDE, Measures, tours_root
from exp.video.schema.run import RunConf

__all__ = ["SPLITS", "Posthoc", "decoding", "main", "parent_dir"]

SPLITS = ("val", "test")  # the evaluation chunks decoded (Measures)
_SETTINGS = {"frames", "steps", "windows", "ckpts"}


def parent_dir(job: Mapping[str, Any], run_dir: Path) -> Path:
    """The parent's run directory of the posthoc job ``job`` in ``run_dir``
    (module docstring).

    Raises:
        ValueError: The job has no ``"parent"``.
    """
    name = job.get("parent")
    if not isinstance(name, str) or not name:
        raise ValueError(f"the posthoc job in {run_dir} names no 'parent' run")
    path = Path(name)
    return path if path.is_absolute() else run_dir.parents[1] / path


def decoding(
    parent: Parent,
    out: Path,
    *,
    frames: int = FRAMES,
    steps: int = STEPS,
    done: int = 0,
) -> Decoding:
    """The ``Decoding`` of ``parent`` into ``out``, of the evaluation chunks
    of ``SPLITS`` of the parent (``Measures``) and, for Walking Tours, its
    data's local copy (``tours_root``); for a parent of the program
    ``"joint"``, also its evaluation windows and those of the dream
    (``measure.joint``, horizon ``DREAM``).

    Raises:
        EnvError: The run streams Walking Tours whose data is not here.
    """
    conf = parent.conf
    if conf.vision is not None and conf.vision.layout == "packed":
        raise ValueError("a run of the packed layout is evaluated by encoder_eval")
    sets = Measures(conf).sets
    evals: dict[str, Tensor] = {s: sets[s].chunks for s in SPLITS}
    t = conf.data.tours
    root = None if t is None else tours_root(t)
    windows = dreams = None  # a joint run rolls out from windows of the balls
    if conf.program == "joint":
        windows = {s: ball_windows(conf, s, stride=STRIDE) for s in SPLITS}
        dreams = {
            s: ball_windows(conf, s, stride=STRIDE, horizon=DREAM) for s in SPLITS
        }
    return Decoding(
        parent,
        out,
        evals=evals,
        root=root,
        windows=windows,
        dreams=dreams,
        frames=frames,
        steps=steps,
        done=done,
    )


class Posthoc:
    """The task ``"posthoc"`` (module docstring), a ``loop.Built``.

    Raises:
        ValueError: The job names no parent, the parent is not a vision or joint
            run with checkpoints, or ``"decode"`` has other keys.
        EnvError: The parent streams Walking Tours whose data is not here.
    """

    def __init__(
        self, conf: RunConf, run_dir: Path, resume: Mapping[str, Any] | None
    ) -> None:
        job = json.loads((run_dir / "job.json").read_text())
        settings = dict(job.get("decode") or {})
        if not set(settings) <= _SETTINGS:
            raise ValueError(
                f"decode takes {sorted(_SETTINGS)}, not {sorted(settings)}"
            )
        done = 0 if resume is None else int(resume["frame"])
        out = run_dir / "post" / "decode"
        steps = int(settings.get("steps", STEPS))
        self._decoding: Decoding | EncoderEval
        parent = None
        if conf.task != "baselines":
            parent = Parent.read(parent_dir(job, run_dir), device=conf.device)
        packed = (parent.conf if parent else conf).vision
        if parent is None or (packed is not None and packed.layout == "packed"):
            base = conf if parent is None else parent.conf
            t = base.data.tours
            if t is None:
                raise ValueError(f"the run {base.name!r} does not stream Walking Tours")
            self._decoding = EncoderEval(
                parent,
                base,
                out,
                root=tours_root(t),
                windows=int(settings.get("windows", WINDOWS)),
                steps=steps,
                ckpts=str(settings.get("ckpts", "all")),
                done=done,
            )
        else:
            self._decoding = decoding(
                parent,
                out,
                frames=int(settings.get("frames", FRAMES)),
                steps=steps,
                done=done,
            )
        self._last: list[dict[str, Any]] = []
        self._schedule = Schedule(units=len(self._decoding), unit="ckpt")

    @property
    def schedule(self) -> Schedule:
        return self._schedule

    def advance(self) -> None:
        self._last = self._decoding.step()

    def log(self) -> dict[str, float | None]:
        return {}

    def evaluate(self, split: str) -> dict[str, float]:
        raise RuntimeError("the posthoc task has no evaluations")

    def diverged(self, row: Mapping[str, float]) -> str | None:
        return None

    def objective(
        self, row: Mapping[str, Any] | None, *, diverged: bool
    ) -> float | None:
        return None

    def modules(self) -> dict[str, nn.Module]:
        return {}

    def visualize(self, out: Path) -> None:
        pass

    def resume_state(self) -> tuple[Mapping[str, Any], dict[str, Any]]:
        return {}, {}  # the loop's frame is the checkpoints done

    def final(self) -> dict[str, dict[str, Any]]:
        if isinstance(self._decoding, EncoderEval):
            return self._decoding.final()
        if not self._last:  # resumed after the last checkpoint
            lines = (
                (self._decoding.out / "metrics.jsonl")
                .read_text()
                .splitlines()[-len(SPLITS) :]
            )
            self._last = [json.loads(x) for x in lines]
        return {"decode": {line["split"]: line for line in self._last}}

    def close(self) -> None:
        pass


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m exp.video.programs.posthoc --run DIR --device D``: decode
    the run in ``DIR`` into ``DIR/post/decode/`` (module docstring), print
    the lines and return the exit code of the run contract (``Exit``):
    ``ENV`` for an ``EnvError`` (the GIL, the threads, the device or the
    data), ``ERROR`` for any other exception. ``OMP_NUM_THREADS`` is set to 1
    unless it is set."""
    ap = argparse.ArgumentParser(
        prog="python -m exp.video.programs.posthoc",
        description="Decode the z of each evaluation checkpoint of a vision or joint run.",
    )
    ap.add_argument("--run", type=Path, required=True, help="the run's directory")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--frames", type=int, default=FRAMES, help="chunks to fit on")
    ap.add_argument("--steps", type=int, default=STEPS, help="of each decoder")
    args = ap.parse_args(argv)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        assert_free_threaded("import")
        configure_determinism(args.device)
        parent = Parent.read(args.run, device=args.device)
        out = args.run / "post" / "decode"
        v, t = parent.conf.vision, parent.conf.data.tours
        d: Decoding | EncoderEval
        if v is not None and v.layout == "packed" and t is not None:
            d = EncoderEval(
                parent, parent.conf, out, root=tours_root(t), steps=args.steps
            )
        else:
            d = decoding(parent, out, frames=args.frames, steps=args.steps)
        lines = [line for _ in range(len(d)) for line in d.step()]
        assert_free_threaded("posthoc")
    except EnvError as e:
        print(f"posthoc: {e}", file=sys.stderr)
        return Exit.ENV
    except Exception:
        traceback.print_exc()
        return Exit.ERROR
    for line in lines:
        print(json.dumps(line))
    return Exit.OK


if __name__ == "__main__":
    sys.exit(main())
