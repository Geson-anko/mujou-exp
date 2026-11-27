r"""V18d's selection numbers on the dyn/copy validation video (plans/v18d.md,
「学習の実装」 and results/v18d/README.md, 「短い run の判定」): every
number the short runs' judgement and the long run's choice read, none of
Zurich.

The validation video is the second of the training videos permuted by
``DynConf.split_seed`` (``dyn_eval.split_videos``; a training video of the
JEPA, as V18c's data are kept). Its whole 128 s blocks give the stacks of
``dyn_eval.layout_of`` with ``max_lag`` = ``MAX_LAG``: the common centres
$u$ whose triples $(u - n, u, u + n)$ lie in the block for every lag of
``LAGS``.

- **The time loss** (``time_loss_rel``; of the replaced version's time
  predictor, which the runs no longer have, so ``evaluate_run`` no longer
  writes it; V18d's completion loss is ``plans/v18d.md``'s ``st_val``): of
  the checkpoint's codes $z$ of the video's stacks and its time predictor
  $f$, at each lag $n$ of ``LAGS`` $\hat L^{\mathrm{rel}}_T(n) = \sum_u
  \lVert f(z_u, z_{u-n}; n) - z_{u+n}\rVert^2 / \sum_u \lVert z_{u+n} -
  \bar z \rVert^2$ over the common centres ($\bar z$ the mean of the
  video's stacks: the predictor's $R_{\mathrm{clim}}$), and ``time_val``
  their trapezoid mean in $\log n$ (``log_mean``).
- **The collapse numbers** (``ValStacks``): ``jepa_eval.CodeStats`` with
  the video's stacks in place of Zurich's: ``held`` ``STATS_COUNT`` stacks
  evenly over the even blocks (the val windows' role: ``rank``,
  ``token_r2``, ``pred``), ``calib`` as many over the odd blocks (the calib
  windows' role: ``pred_ratio_calib``'s means); ``token_r2``'s fit stacks
  stay the probes' (``StreamEvaluator.fit_stacks``, V18c's), and
  ``sigreg_heldin`` is of the training units as V18c's. ``rank`` is
  ``effective_rank`` of the codes of ``held``.

``evaluate_run`` writes ``<run>/posthoc_v18d/select.jsonl`` (a row per
step); ``evaluate_yard`` the yardsticks of the thresholds (V18's ``color``
of patch 16, ``color8``, V18c's ``init_c``) to ``<out>/select.jsonl``::

    python -m exp.vision.time_val run <run_dir> [--steps 6000]
    python -m exp.vision.time_val yard --config <v18c run> --out <dir>
"""

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from exp.lib import ckpt
from exp.lib.metrics import effective_rank
from exp.lib.provenance import configure_determinism
from exp.lib.rundir import write_json
from exp.lib.tours.stacks import tours_root
from exp.lib.tours.walking_tours import videos
from exp.lib.video_io import U8Reader
from exp.vision.dyn_eval import (
    DynConf,
    SetLayout,
    layout_of,
    split_videos,
)
from exp.vision.eval import BATCH
from exp.vision.geom_eval import _conf_of
from exp.vision.jepa import LATENT_DIM, JEPAModules, build_jepa
from exp.vision.jepa_diag import run_ckpts
from exp.vision.jepa_eval import SPAN, CodeStats, ColorCode, build_run_jepa
from exp.vision.schema import RunConf
from exp.vision.stream_eval import Code, StreamEvaluator, append_jsonl, spanned

__all__ = [
    "LAGS",
    "MAX_LAG",
    "OUT",
    "STATS_COUNT",
    "ValStacks",
    "evaluate_run",
    "evaluate_yard",
    "even_pick",
    "log_mean",
    "main",
    "stacks_of",
    "time_loss_rel",
    "val_layout",
]

OUT = "posthoc_v18d"
LAGS = (8, 11, 16, 23, 32, 45, 64, 91, 128, 181, 256, 362, 512)  # stacks (0.1 s)
MAX_LAG = 512
STATS_COUNT = 1024  # the held and calib stacks (Zurich's val windows: 1024)
CENTRE_CHUNK = 4096  # centres a time through the predictor

type Predict = Callable[[Tensor, Tensor, Tensor], Tensor]


def val_layout(
    conf: RunConf, root: Path, dconf: DynConf = DynConf()
) -> tuple[tuple[str, ...], SetLayout]:
    """The validation video's name(s) and its layout (module docstring)."""
    t = conf.data.tours
    names = [v for v in videos(root) if v != t.held_out]
    by = {v: len(U8Reader(root / v)) for v in names}
    val = split_videos(names, dconf.split_seed)["val"]
    span = (t.frames - 1) * t.stride
    return val, layout_of([by[v] for v in val], span, dconf.block, MAX_LAG)


@torch.no_grad()
def time_loss_rel(
    z: Tensor, centres: Tensor, predict: Predict, lags: Sequence[int] = LAGS
) -> list[float]:
    r"""$\hat L^{\mathrm{rel}}_T(n)$ (module docstring) of each lag of
    ``lags``: ``predict(z0, zm, n)`` of the codes ``z`` ``(N, D)`` at the
    ``centres`` ``(C,)`` (indices into ``z``, each $u \pm n$ in it), $\bar
    z$ the mean of ``z``; sums in float64."""
    zbar = z.double().mean(0)
    out = []
    for n in lags:
        num = den = 0.0
        for c in centres.split(CENTRE_CHUNK):
            z0, zm, zp = z[c], z[c - n], z[c + n]
            lag = torch.full((len(c),), n, device=z.device)
            num += float((predict(z0, zm, lag).double() - zp.double()).square().sum())
            den += float((zp.double() - zbar).square().sum())
        out.append(num / den)
    return out


def log_mean(values: Sequence[float], lags: Sequence[int] = LAGS) -> float:
    """The trapezoid mean of ``values`` over $\\log n$ of ``lags``."""
    x = torch.tensor(lags, dtype=torch.float64).log()
    y = torch.tensor(values, dtype=torch.float64)
    return float(torch.trapezoid(y, x) / (x[-1] - x[0]))


def stacks_of(
    root: Path,
    names: Sequence[str],
    layout: SetLayout,
    which: Tensor,
    frames: int,
    stride: int,
    block: int,
) -> Tensor:
    """The stacks u8 ``(len(which), frames, C, H, W)`` of the layout's
    indices ``which`` (its videos ``names``, its blocks of ``block``
    frames).

    Raises:
        ValueError: A stack's frames leave its block.
    """
    offsets = torch.arange(frames) * stride
    per = block - (frames - 1) * stride
    readers = {v: U8Reader(root / n) for v, n in enumerate(names)}
    out = []
    for i in which.tolist():
        v, f = int(layout.video[i]), int(layout.frame[i])
        if f % block >= per:
            raise ValueError(f"stack {i} leaves its block")
        out.append(readers[v][f + offsets])
    return torch.stack(out)


def even_pick(pool: Tensor, count: int) -> Tensor:
    """``count`` of ``pool`` evenly spread (its ends included)."""
    return pool[torch.linspace(0, len(pool) - 1, count).round().long()]


@dataclass(frozen=True)
class ValStacks:
    """The stacks of ``CodeStats`` (module docstring): ``zurich_stacks`` the
    validation video's held stacks, ``zurich_calib_stacks`` its calib
    stacks, ``fit_stacks`` the probes' (named as ``StreamEvaluator``'s
    properties, which ``CodeStats`` reads)."""

    fit_stacks: Tensor
    zurich_stacks: Tensor
    zurich_calib_stacks: Tensor

    @classmethod
    def make(
        cls,
        root: Path,
        names: Sequence[str],
        layout: SetLayout,
        fit: Tensor,
        *,
        frames: int,
        stride: int,
        block: int,
        count: int = STATS_COUNT,
    ) -> ValStacks:
        """Of the layout: ``count`` stacks evenly over the even blocks
        (held) and as many over the odd blocks (calib)."""
        sets = [
            stacks_of(root, names, layout, even_pick(torch.nonzero(layout.block % 2 == r)
                      .flatten(), count), frames, stride, block)
            for r in (0, 1)
        ]  # fmt: skip
        return cls(fit, *sets)


@torch.no_grad()
def _stats_rows(
    code: Code, stats: CodeStats, vs: ValStacks, dev: torch.device
) -> dict[str, float]:
    """``CodeStats.of`` with its zurich keys renamed ``val`` and ``rank``."""
    row = {k.replace("zurich", "val"): v for k, v in stats.of(code).items()}
    z = torch.cat([code.mu(c.to(dev)).float() for c in vs.zurich_stacks.split(BATCH)])
    row["rank"] = effective_rank(z)
    return row


def _setup(
    conf: RunConf, root: Path, out: Path
) -> tuple[ValStacks, CodeStats, tuple[str, ...], SetLayout]:
    view = spanned(conf, SPAN)
    ev = StreamEvaluator(view, out, root, out / "tmp", c=1)
    names, layout = val_layout(conf, root)
    t = conf.data.tours
    vs = ValStacks.make(root, names, layout, ev.fit_stacks, frames=t.frames,
                        stride=t.stride, block=DynConf().block)  # fmt: skip
    stats = CodeStats(view, root, vs)
    return vs, stats, names, layout


def evaluate_run(
    run_dir: Path,
    *,
    steps: Sequence[int] | None = None,
    root: Path | None = None,
    device: str | None = None,
) -> Path:
    """The rows (module docstring) of the run's checkpoints of ``steps``
    (default the last) into ``run_dir / OUT / "select.jsonl"`` (appended).
    Returns the file.

    Raises:
        ValueError: A step has no checkpoint, or the run has no time
            predictor.
    """
    conf = _conf_of(run_dir, device)
    settings = json.loads((run_dir / "jepa.json").read_text())
    ckpts = run_ckpts(run_dir)
    todo = [max(ckpts)] if steps is None else list(steps)
    missing = [s for s in todo if s not in ckpts]
    if missing:
        raise ValueError(f"no checkpoint of steps {missing} (of {list(ckpts)})")
    determinism = configure_determinism(conf.device)
    dev = torch.device(conf.device)
    root = tours_root(conf.data.tours.data_version) if root is None else root
    out = run_dir / OUT
    out.mkdir(parents=True, exist_ok=True)
    vs, stats, names, layout = _setup(conf, root, out)
    dconf = DynConf()
    write_json(
        out / "config.json",
        {
            "val_videos": list(names),
            "lags": list(LAGS),
            "max_lag": MAX_LAG,
            "centres": len(layout.centres),
            "stacks": len(layout.video),
            "count": STATS_COUNT,
            "split_seed": dconf.split_seed,
            "determinism": determinism,
        },  # fmt: skip
    )
    mods = build_run_jepa(conf, settings)
    for step in todo:
        head = ckpt.load_eval(ckpts[step], mods.modules())
        row: dict[str, Any] = {"step": step, "t": head["t"], "code": "z"}
        row |= _stats_rows(mods, stats, vs, dev)
        row |= stats.pred(mods, settings["k"], settings.get("expand", 1))
        append_jsonl(out / "select.jsonl", [row])
    return out / "select.jsonl"


def evaluate_yard(conf: RunConf, out: Path, *, root: Path | None = None) -> Path:
    """The yardsticks' rows (module docstring; ``init_c`` built as V18c's
    from ``conf``'s seed) into ``out / "select.jsonl"``. Returns the file."""
    determinism = configure_determinism(conf.device)
    dev = torch.device(conf.device)
    root = tours_root(conf.data.tours.data_version) if root is None else root
    out.mkdir(parents=True, exist_ok=True)
    vs, stats, names, _ = _setup(conf, root, out)
    init_c: JEPAModules = build_jepa(
        conf, seed=conf.seed, token_dim=16, proj="fixed", latent_dim=LATENT_DIM
    )
    codes: dict[str, Code] = {
        "color": ColorCode(16),
        f"color{conf.model.patch}": ColorCode(conf.model.patch),
        "init_c": init_c,
    }
    write_json(out / "config.json", {"val_videos": list(names), "codes": list(codes),
                                     "count": STATS_COUNT, "determinism": determinism})  # fmt: skip
    (out / "select.jsonl").write_text("")
    for name, code in codes.items():
        append_jsonl(
            out / "select.jsonl", [{"code": name} | _stats_rows(code, stats, vs, dev)]
        )
    return out / "select.jsonl"


def main(argv: Sequence[str] | None = None) -> Path:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = p.add_subparsers(dest="job", required=True)
    run = sub.add_parser("run")
    run.add_argument("run_dir", type=Path)
    run.add_argument("--steps", type=lambda s: [int(x) for x in s.split(",")])
    run.add_argument("--device", default=None)
    run.add_argument("--root", type=Path, default=None)
    yard = sub.add_parser("yard")
    yard.add_argument("--config", type=Path, required=True, help="a V18c run")
    yard.add_argument("--out", type=Path, required=True)
    yard.add_argument("--device", default=None)
    yard.add_argument("--root", type=Path, default=None)
    a = p.parse_args(argv)
    if a.job == "run":
        return evaluate_run(a.run_dir, steps=a.steps, root=a.root, device=a.device)
    return evaluate_yard(_conf_of(a.config, a.device), a.out, root=a.root)


if __name__ == "__main__":
    print(main())
