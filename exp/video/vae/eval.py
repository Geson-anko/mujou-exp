r"""The evaluation of the autoencoder's checkpoints afterwards (M18,
``task=posthoc`` of a run of the program ``"ae"``, and ``task=baselines``):
M17's evaluation of a code (``measure.encoder_eval``), with the autoencoder's
own decoder beside it.

The units, a step each (``AEEval.step``), into the directory ``out``:

- unit 0, the yardsticks: those of ``EncoderEval`` (the pixel PCA of the last
  frame and the ViT before it learns, ``baselines.jsonl``; the run's encoder
  before it learns is that ViT: ``modules.build_ae``), and the mean color of
  each tubelet (``measure.mean_color.tubelet_means``, ``tube``,
  $D$ = 384), a reference shown beside U and judged by nothing
  (``refs.jsonl``; M17 found that U's reconstruction asks for the mean
  color only, results/m17.md);
- unit ``1 + i``: checkpoint ``i`` of the run (all of them, or the last
  only, ``ckpts``), a line of val and of test in ``metrics.jsonl``: the
  lines of ``EncoderEval.evaluate_code`` of the code ``"ae"`` (the stack's
  $T$ tokens flattened, $D = T d$: the decoder learned afterwards, the
  linear dynamics, ``rank``, ``z_spread``, the sensitivities; no
  restoration and no completion, which need a predictor), and of the
  autoencoder itself on the held-out stacks of ``measures.held_out_stacks``
  (the first stack of each of the same windows): ``stack_mse`` and ``loss``
  (``measures.stack_errors``), and ``own_recon``, ``own_recon_blur``, the
  last frame of $g(f(x))$ against the true one as ``recon`` and
  ``recon_blur`` measure the decoder learned afterwards (over the error of
  the mean last frame of the training stacks); the pictures of
  ``evaluate_code`` and ``<t>-own.png`` (the true last frames of ``SHOWN``
  windows of val, then those of $g$; ``<t>.png`` has those of the decoder
  learned afterwards).

The target of the autoencoder is $y = x / 256$ and the evaluation's frames are
$x / 255$: $g$'s frames are taken to the evaluation's units by $256 / 255$.
"""

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from exp.lib import ckpt
from exp.lib.blur_error import blur_recon
from exp.lib.png import tile, write_png
from exp.lib.rundir import atomic_open, write_json
from exp.video.measure.encoder_eval import CKPTS, SHOWN, WINDOWS, EncoderEval
from exp.video.measure.mean_color import tubelet_means
from exp.video.measure.posthoc import STEPS, Parent, as_float
from exp.video.schema.run import RunConf
from exp.video.vae.loss import LEVELS
from exp.video.vae.measures import (
    EVAL_BATCH,
    SPLITS,
    held_out_stacks,
    stack_errors,
    training_moments,
)
from exp.video.vae.modules import build_ae

__all__ = ["AEEval"]

_SCALE = LEVELS / 255  # the autoencoder's units to the evaluation's


class AEEval:
    """The evaluation of the checkpoints of ``parent`` (module docstring),
    the interface of ``EncoderEval``: ``len``, ``step``, ``final``.

    Args:
        parent: The run, or None (``task=baselines``: unit 0 alone).
        conf: The run's config.
        out: The directory of the files.
        root: The local copy of the data.
        windows: The training windows of ``EncoderEval`` (its decoders and
            yardsticks); the mean stack $m$ of ``stack_mse`` is the run's
            (``training_moments`` of ``WINDOWS``) whatever it is.
        steps: The steps of each decoder's fit.
        ckpts: ``"all"`` or ``"last"``.
        done: The units done already.
    """

    def __init__(
        self,
        parent: Parent | None,
        conf: RunConf,
        out: Path,
        *,
        root: Path,
        windows: int = WINDOWS,
        steps: int = STEPS,
        ckpts: str = "all",
        done: int = 0,
    ) -> None:
        if ckpts not in CKPTS:
            raise ValueError(f"ckpts must be one of {CKPTS}, not {ckpts!r}")
        paths = [] if parent is None else list(parent.ckpts)
        self._ckpts = paths[-1:] if ckpts == "last" else paths
        if not 0 <= done <= len(self):
            raise ValueError(f"done {done} of {len(self)} units")
        self.parent, self.conf, self.out, self.done = parent, conf, out, done
        metrics = out / "metrics.jsonl"
        kept = max(done - 1, 0) * len(SPLITS)
        old = metrics.read_text().splitlines(True)[:kept] if metrics.exists() else []
        # the yardsticks of unit 0 (it cuts metrics.jsonl: its own units stop at 0)
        self._eval = EncoderEval(
            None, conf, out, root=root, windows=windows, steps=steps, done=min(done, 1)
        )
        with atomic_open(metrics) as f:
            f.write("".join(old).encode())
        if done == 0:
            config = json.loads((out / "config.json").read_text())
            write_json(out / "config.json", config | self._config(ckpts))
        self._dev = torch.device(conf.device)
        self._mods = build_ae(conf, seed=conf.seed)
        self._stacks = held_out_stacks(conf, root)
        moments = training_moments(conf, root)  # the run's m, whatever windows
        self._mean = moments.mean.float().to(self._dev)
        self._mean_last = (moments.mean[-1] * _SCALE).float().to(self._dev)[None]
        self._last: list[dict[str, Any]] = []

    def __len__(self) -> int:
        """The units: the yardsticks, then the checkpoints."""
        return 1 + len(self._ckpts)

    def _config(self, ckpts: str) -> dict[str, Any]:
        job = None if self.parent is None else self.parent.job
        return {
            "ckpts": [p.name for p in self._ckpts],
            "which": ckpts,
            "parent": None
            if job is None
            else {
                "run_id": job.get("run_id"),
                "group": job["group"],
                "code": job.get("code"),
            },
        }

    def step(self) -> list[dict[str, Any]]:
        """Do unit ``done``; returns its lines."""
        if self.done == 0:
            lines = self._eval.step()
            v = self.conf.vision
            assert v is not None
            refs = self._eval.evaluate_code(
                "tube", lambda s: tubelet_means(as_float(s), v.patch, v.tubelet)
            )
            _write(self.out / "refs.jsonl", refs)
            lines = [*lines, *refs]
        else:
            path = self._ckpts[self.done - 1]
            assert self.parent is not None
            head = ckpt.load_eval(
                path, self._mods.named(), group=self.parent.job["group"]
            )
            mods = self._mods
            lines = self._eval.evaluate_code(
                path.stem, lambda s: mods.code(s).flatten(1)
            )
            for line in lines:
                line |= {"t": head["t"], "frame": head["frame"], "code": "ae"}
                line |= self._own(line["split"], path.stem)
            _append(self.out / "metrics.jsonl", lines)
            self._last = lines
        self.done += 1
        return lines

    def _own(self, split: str, name: str) -> dict[str, float]:
        """The keys of the autoencoder itself on the held-out stacks of
        ``split`` (module docstring); the picture of val."""
        stacks = self._stacks[split]
        mse, loss = stack_errors(self._mods, stacks, self._mean)
        with torch.no_grad():
            own = torch.cat(
                [
                    self._mods(c.to(self._dev))[:, -1] * _SCALE
                    for c in stacks.split(EVAL_BATCH)
                ]
            ).clamp(0, 1)
        f0 = as_float(stacks[:, -1].to(self._dev))
        mean = self._mean_last.expand_as(f0)
        base = float(((mean - f0) ** 2).mean())
        if split == "val":  # beside <name>.png of evaluate_code
            rows = torch.stack([f0[:SHOWN], own[:SHOWN]]).cpu()
            scale = max(1, 96 // f0.shape[-1])
            write_png(self.out / f"{name}-own.png", tile(rows, scale=scale))
        return {
            "stack_mse": mse,
            "loss": loss,
            "own_recon": float(((own - f0) ** 2).mean()) / base,
            "own_recon_blur": blur_recon(own, f0, self._mean_last),
        }

    def final(self) -> dict[str, Any]:
        """``result.json``'s ``final``: the last checkpoint's line of each
        split (``decode``), the yardsticks (``baselines``) and the
        references (``refs``)."""
        last = self._last
        if not last and self._ckpts:
            raw = (self.out / "metrics.jsonl").read_text().splitlines()
            last = [json.loads(x) for x in raw[-len(SPLITS) :]]
        refs = [
            json.loads(x) for x in (self.out / "refs.jsonl").read_text().splitlines()
        ]
        return {
            "decode": {line["split"]: line for line in last},
            "baselines": self._eval.baselines(),
            "refs": {line["split"]: line for line in refs},
        }


def _clean(x: Any) -> Any:
    return None if isinstance(x, float) and not math.isfinite(x) else x


def _lines(lines: Sequence[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps({k: _clean(x) for k, x in ln.items()}) + "\n" for ln in lines
    )


def _append(path: Path, lines: Sequence[Mapping[str, Any]]) -> None:
    """Append ``lines`` to the JSON lines ``path`` (non-finite values as null)."""
    with path.open("a") as f:
        f.write(_lines(lines))


def _write(path: Path, lines: Sequence[Mapping[str, Any]]) -> None:
    """Write ``lines`` as the JSON lines ``path``."""
    with atomic_open(path) as f:
        f.write(_lines(lines).encode())
