r"""The evaluation of a run's checkpoints on its own stream (V8): lifelong
learning is judged on everything the stream showed and will show, not on a
held-out video alone. A checkpoint after $F$ frames has learned from the
stream's units $k < F$ (``stacks_at``: stream 0, stack $k$ is the one of its
$k$-th ``next()``); a unit $k < F$ is past (a loss that grew back there is
forgetting), $k \ge F$ future (a low loss there is generalization).

Positions (``stream_positions``): every ``PAST_EVERY`` seconds over the run's
span ``[0, measure.duration)`` and every ``FUTURE_EVERY`` seconds over the
``FUTURE`` seconds after it, the units the stream marks not ready or reset
left out.

The matrix. For every checkpoint $T$ (``ckpt/<t>.pt``) and position $k$,
$e_{T,k} = \lVert x_k/255 - g(\mu(x_k)) \rVert^2$, the system's squared error
on the stack $x_k$; $b_k = \lVert x_k/255 - m \rVert^2$ with $m$ the mean
stack of the positions in the run's span. A group $G$ of positions has the
ratio $\sum_G e / \sum_G b$ (1 is the mean stack, 0 a perfect
reconstruction). The groups of a checkpoint after $F$ frames of $\Delta$
seconds (``groups``): by the age $(F - k)\Delta$ of a past unit ``now`` (0,
10], ``past_10_60`` (10, 60], ``past_60_300`` (60, 300], ``past_300_`` > 300
seconds, all of them ``past``; by the lead $(k - F)\Delta$ of a future unit
``next_0_10`` [0, 10), ``next_10_60``, ``next_60_300``, ``next_300_``, all of
them ``future``. Whatever the checkpoint, ``span`` is the positions in the
run's span and ``after`` those after it, so that two checkpoints compare on
the same stacks; ``zurich`` is the ratio on the first stack of each of the
held-out video's val windows (``eval.eval_splits``), $b$ of the same $m$,
the stacks no run learns from.

Prequential. ``stream.jsonl``'s ``sq`` of second $s$ is the mean $\lVert y -
\hat y \rVert^2$ of the frames of $(s - 1, s]$, made before the weights
learned from them, of $y = (x + u)/256$ and the sampled $z$; in the units of
$x/255$ it is $p = (256/255)^2\,\mathrm{sq} - P/(12 \cdot 255^2)$ ($u$
uniform, $P$ the values of a stack; ``prequential``). A past group's
``pre_<group>`` is $\sum_G p / \sum_G b$ (the error when the stream showed
those stacks), so ``<group> - pre_<group>`` is how much was forgotten since.

The probes, at the checkpoints of ``PROBE_AT`` and the last one. The run's span is cut
into ``BLOCK``-second blocks; pairs of consecutive stacks $(x_k, x_{k+1})$
every ``PROBE_EVERY`` seconds of the odd blocks fit, of the even blocks
evaluate (``span``), with those every ``FUTURE_EVERY`` seconds after the span
(``after``) and the held-out video's val windows (``zurich``,
``eval.eval_splits``). Per group: ``recon_blur`` (and its
``_sd``, ``_k``) of ``measure.posthoc_seeds`` decoders of the standardized
$z$ fit to the fit pairs' last frames (``eval.fit_decoder``), against the
mean last frame of the fit pairs, blurred by ``measure.blur_sigma`` pixels;
``dyn_1/copy_1`` (``linear_dyn``, fit on the fit pairs, its ridge chosen on
the held-out calib windows); ``rank`` of the first codes. The fit set is the
same for every checkpoint, so the probes compare checkpoints, not data.

A run of the layout ``"chunk"`` is read as the single stream it walks
(``eval.stream_view``): every unit, time and frame above is of that stream,
a checkpoint at $t$ after $F$ frames of a run of $B$ streams being at $tB$
after $FB$ units. ``span`` (seconds) evaluates the span ``[0, span)`` in
place of the run's (of its view), so that a run longer than its loop of
videos is read on stacks it saw first in order.

``evaluate_stream`` writes ``<run_dir>/posthoc_stream/``: ``positions.json``
(``k``, ``t`` seconds, ``base`` $b_k$, ``pre`` $p_k$ or null),
``matrix.jsonl`` (a row per checkpoint: ``t``, ``frame``, ``err`` $e_{T,k}$
in the order of the positions), ``summary.jsonl`` (a row per checkpoint: the
ratio and ``n_<group>`` of each group and ``pre_<group>`` of the past ones),
``probes.jsonl`` (a row per probed checkpoint and group) and the pictures
``<t>-span.png`` and ``<t>-after.png`` of the probed checkpoints (``SHOWN``
positions evenly spaced: the true last frames, the system's, the decoder's
of seed 0)::

    python -m exp.vision.stream_eval <run_dir> [--root <data>] [--fit-steps S] \
        [--span SECONDS] [--probe-at T,T,...]
"""

import argparse
import json
import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor

from exp.lib import ckpt, linear_dyn
from exp.lib.blur_error import blur_recon
from exp.lib.metrics import effective_rank
from exp.lib.png import tile, write_png
from exp.lib.provenance import configure_determinism
from exp.lib.rundir import atomic_open, write_json
from exp.lib.tours.stacks import tours_root
from exp.vision.compose import from_dict
from exp.vision.eval import (
    BATCH,
    DECODER_LR,
    DECODER_SEED,
    FIT_STEPS,
    HORIZON,
    SCALE,
    SEED_STEP,
    SHOWN,
    Standardizer,
    VaeCode,
    eval_splits,
    fit_decoder,
    stacks_at,
    stream_view,
)
from exp.vision.schema import RunConf

__all__ = [
    "BLOCK",
    "FUTURE",
    "FUTURE_EVERY",
    "OUT",
    "PAST_EVERY",
    "PROBE_AT",
    "PROBE_EVERY",
    "StreamEvaluator",
    "evaluate_stream",
    "groups",
    "main",
    "prequential",
    "spanned",
    "stream_positions",
]

PAST_EVERY, FUTURE_EVERY, FUTURE = 5.0, 10.0, 3000.0  # seconds
PROBE_EVERY, BLOCK = 0.5, 10.0  # seconds
PROBE_AT = (0.0, 60.0, 300.0, 600.0, 1200.0, 1800.0)  # the probed checkpoints
OUT = "posthoc_stream"
# (name, low, high] of the age of a past unit, [low, high) of the lead of a
# future one, in seconds
PAST_GROUPS = (
    ("now", 0.0, 10.0),
    ("past_10_60", 10.0, 60.0),
    ("past_60_300", 60.0, 300.0),
    ("past_300_", 300.0, math.inf),
)
FUTURE_GROUPS = (
    ("next_0_10", 0.0, 10.0),
    ("next_10_60", 10.0, 60.0),
    ("next_60_300", 60.0, 300.0),
    ("next_300_", 300.0, math.inf),
)


def spanned(conf: RunConf, span: float | None) -> RunConf:
    """``conf`` whose span is ``[0, span)`` (``measure.duration`` = ``span``,
    its checkpoints left out), or ``conf`` itself for None."""
    if span is None:
        return conf
    measure = replace(conf.measure, duration=span, ckpt_times=[], ckpt_every=0.0)
    return replace(conf, measure=measure)


def _units(conf: RunConf, seconds: float) -> int:
    """``seconds`` in frames of ``time.frame``.

    Raises:
        ValueError: Not a whole number of frames.
    """
    p = seconds / conf.time.frame
    if abs(p - round(p)) > 1e-6:
        raise ValueError(f"{seconds} s is not a whole number of {conf.time.frame} s")
    return round(p)


def stream_positions(
    conf: RunConf,
    *,
    past_every: float = PAST_EVERY,
    future_every: float = FUTURE_EVERY,
    future: float = FUTURE,
) -> list[int]:
    """The units of the matrix before the readiness check (module docstring):
    every ``past_every`` seconds over the run's span, then every
    ``future_every`` seconds over the ``future`` seconds after it."""
    d = _units(conf, conf.measure.duration)
    p, q, e = (_units(conf, s) for s in (past_every, future_every, future))
    return [*range(0, d, p), *range(d, d + e, q)]


def groups(ks: Sequence[int], frame: int, dt: float) -> dict[str, list[int]]:
    """The indices into ``ks`` of each group (module docstring) of a
    checkpoint after ``frame`` frames of ``dt`` seconds; ``past`` and
    ``future`` are all of each side."""
    out: dict[str, list[int]] = {g[0]: [] for g in (*PAST_GROUPS, *FUTURE_GROUPS)}
    out["past"], out["future"] = [], []
    eps = 1e-9 * dt
    for i, k in enumerate(ks):
        if k < frame:
            age = (frame - k) * dt
            out["past"].append(i)
            name = next(g for g, lo, hi in PAST_GROUPS if lo + eps < age <= hi + eps)
        else:
            lead = (k - frame) * dt
            out["future"].append(i)
            name = next(g for g, lo, hi in FUTURE_GROUPS if lo - eps <= lead < hi - eps)
        out[name].append(i)
    return out


def prequential(
    run_dir: Path, conf: RunConf, ks: Sequence[int], c: int = 1
) -> list[float | None]:
    """$p_k$ (module docstring) of each unit of ``ks``: of the row of
    ``stream.jsonl`` whose second $s$ holds it ($s = \\lfloor \\lfloor k/c
    \\rfloor \\Delta / \\mathrm{log\\_every} \\rfloor + 1$ logs: the run
    showed unit $k$ of its view at its frame $\\lfloor k/c \\rfloor$,
    ``eval.stream_view``), null after the run's span or without the row's
    ``sq``."""
    every = _units(conf, conf.ctl.log_every)
    t = conf.data.tours
    p = t.frames * 3 * t.size * t.size
    sq: dict[int, float] = {}
    path = run_dir / "stream.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r.get("sq") is not None:
                sq[round(r["t"] / conf.ctl.log_every)] = r["sq"]
    out: list[float | None] = []
    for k in ks:
        v = sq.get(k // c // every + 1)
        out.append(None if v is None else (256 / 255) ** 2 * v - p / (12 * 255**2))
    return out


def _ratio(num: Sequence[float], den: Sequence[float], idx: Sequence[int]) -> float:
    """$\\sum_{i} \\mathrm{num}_i / \\sum_i \\mathrm{den}_i$ over ``idx``,
    NaN of none."""
    if not idx:
        return math.nan
    return sum(num[i] for i in idx) / sum(den[i] for i in idx)


class StreamEvaluator:
    """The matrix rows, summaries and probes (module docstring) of the run
    ``conf`` (in ``run_dir``) on the data at ``root``; pictures into ``out``.

    Args:
        conf: The run's config.
        run_dir: The run (its ``stream.jsonl`` for ``prequential``).
        root: The prepared Walking Tours.
        out: The directory of the pictures.
        fit_steps: The steps of each decoder's fit.
        past_every, future_every, future, probe_every, block: as the
            module's constants (seconds).
        c: The stacks of ``conf`` a frame of the run showed (``conf`` is
            the run's ``eval.stream_view``; 1 for a spread run).
    """

    def __init__(
        self,
        conf: RunConf,
        run_dir: Path,
        root: Path,
        out: Path,
        *,
        fit_steps: int = FIT_STEPS,
        past_every: float = PAST_EVERY,
        future_every: float = FUTURE_EVERY,
        future: float = FUTURE,
        probe_every: float = PROBE_EVERY,
        block: float = BLOCK,
        c: int = 1,
    ) -> None:
        self._conf, self._out, self._fit_steps = conf, out, fit_steps
        self._dev = torch.device(conf.device)
        self._dt = conf.time.frame
        self._span = span = _units(conf, conf.measure.duration)
        ks = stream_positions(
            conf, past_every=past_every, future_every=future_every, future=future
        )
        self.ks, self._x = stacks_at(conf, root, ks)  # u8, CPU
        inside = [i for i, k in enumerate(self.ks) if k < span]
        self._mean_stack = (
            self._x[inside].double().mean(0).div(255).float().to(self._dev)
        )
        self.base = [
            float(((xc.to(self._dev).float() / 255 - self._mean_stack) ** 2).sum())
            for xc in self._x.split(1)
        ]
        self.pre = prequential(run_dir, conf, self.ks, c)
        # the probes' pairs: odd blocks fit, even blocks and after the span
        # evaluate
        p, b = _units(conf, probe_every), _units(conf, block)
        in_span = range(0, span - 1, p)
        self._fit = self._pairs(root, [k for k in in_span if (k // b) % 2 == 1])
        self._held = self._pairs(root, [k for k in in_span if (k // b) % 2 == 0])
        after = range(span, span + _units(conf, future), _units(conf, future_every))
        self._after = self._pairs(root, list(after))
        self._zurich = eval_splits(conf, root)
        n = conf.data.tours.frames
        self._zx = self._zurich["val"][:, :n]  # the first stack of each window
        self.zurich_base = self._sq_errors(lambda _: self._mean_stack, self._zx)
        last = self._fit[1][:, 0, -1]  # (m, C, H, W) u8: the fit pairs' last frames
        self._targets = last.to(self._dev)
        self._mean = last.double().mean(0).div(255).float().to(self._dev)
        out.mkdir(parents=True, exist_ok=True)

    def _pairs(self, root: Path, ks: Sequence[int]) -> tuple[list[int], Tensor]:
        """The ``k`` of ``ks`` whose stacks $x_k$ and $x_{k+1}$ are both kept
        by ``stacks_at`` and those pairs u8 ``(m, 2, n, C, H, W)``."""
        k0, x0 = stacks_at(self._conf, root, ks)
        k1, x1 = stacks_at(self._conf, root, [k + 1 for k in k0])
        nxt = {k: i for i, k in enumerate(k1)}
        keep = [i for i, k in enumerate(k0) if k + 1 in nxt]
        if not keep:
            n, s = self._conf.data.tours.frames, self._conf.data.tours.size
            return [], torch.empty(0, 2, n, 3, s, s, dtype=torch.uint8)
        x = torch.stack([x0[keep], x1[[nxt[k0[i] + 1] for i in keep]]], 1)
        return [k0[i] for i in keep], x

    def settings(self) -> dict[str, Any]:
        """What the rows depend on beside the run's config."""
        m = self._conf.measure
        return {
            "positions": len(self.ks),
            "fit_pairs": len(self._fit[0]),
            "held_pairs": len(self._held[0]),
            "after_pairs": len(self._after[0]),
            "fit_steps": self._fit_steps,
            "lr": DECODER_LR,
            "decoder_seeds": [
                self._conf.seed + DECODER_SEED + SEED_STEP * k
                for k in range(m.posthoc_seeds)
            ],
            "blur_sigma": m.blur_sigma,
        }

    @torch.no_grad()
    def _sq_errors(self, recon: Callable[[Tensor], Tensor], x: Tensor) -> list[float]:
        """$\\lVert x/255 - \\mathrm{recon}(x)\\rVert^2$ of each stack of ``x``
        u8, ``recon`` of a batch of them on the device."""
        out: list[float] = []
        for xc in x.split(BATCH):
            xd = xc.to(self._dev)
            d = (recon(xd) - xd.float() / 255) ** 2
            out += d.double().flatten(1).sum(1).tolist()
        return out

    def errors(self, code: VaeCode) -> list[float]:
        """$e_{T,k}$ of each position for the VAE ``code``."""
        return self._sq_errors(lambda x: code.decode(code.mu(x)), self._x)

    def zurich(self, code: VaeCode) -> float:
        """The ratio ``zurich`` (module docstring) of the VAE ``code``."""
        err = self._sq_errors(lambda x: code.decode(code.mu(x)), self._zx)
        return sum(err) / sum(self.zurich_base)

    def summary(self, err: Sequence[float], frame: int) -> dict[str, Any]:
        """The ratio and count of each group, and ``pre_`` of the past ones,
        of the errors ``err`` of a checkpoint after ``frame`` frames."""
        row: dict[str, Any] = {}
        pre_ok = [i for i, p in enumerate(self.pre) if p is not None]
        pre = [0.0 if p is None else p for p in self.pre]
        for name, idx in groups(self.ks, frame, self._dt).items():
            row[name] = _ratio(err, self.base, idx)
            row[f"n_{name}"] = len(idx)
            if name == "now" or name.startswith("past"):
                ok = sorted(set(idx) & set(pre_ok))
                row[f"pre_{name}"] = _ratio(pre, self.base, ok)
        for name, inside in (("span", True), ("after", False)):
            idx = [i for i, k in enumerate(self.ks) if (k < self._span) == inside]
            row[name], row[f"n_{name}"] = _ratio(err, self.base, idx), len(idx)
        return row

    @torch.no_grad()
    def probes(self, code: VaeCode, name: str) -> list[dict[str, Any]]:
        """The rows of the groups span, after and zurich (module docstring)
        of a checkpoint, and its pictures ``<name>-span.png`` and
        ``<name>-after.png``."""
        conf = self._conf
        z_zur = {s: self._codes(code, self._zurich_pairs(s)) for s in ("calib", "val")}
        d = z_zur["calib"].shape[-1]
        z_fit = self._codes(code, self._fit[1], d)  # (m, 2, D)
        z_held = self._codes(code, self._held[1], d)
        z_after = self._codes(code, self._after[1], d)
        sets = {
            "span": (z_held, self._held[1]),
            "after": (z_after, self._after[1]),
            "zurich": (z_zur["val"], self._zurich_pairs("val")),
        }
        flat = z_fit[:, 0]
        std = Standardizer(flat)
        blur: dict[str, list[float]] = {s: [] for s in sets}
        dec0 = None
        for k in range(conf.measure.posthoc_seeds):
            seed = conf.seed + DECODER_SEED + SEED_STEP * k
            dec = fit_decoder(
                std(flat),
                self._targets,
                steps=self._fit_steps,
                lr=DECODER_LR,
                gen=torch.Generator().manual_seed(seed),
            )
            dec0 = dec if k == 0 else dec0
            for s, (z, x) in sets.items():
                if len(z):
                    f = x[:, 0, -1].to(self._dev).float() / 255
                    x_hat = _decoded(dec, std(z[:, 0]))
                    blur[s].append(
                        blur_recon(x_hat, f, self._mean, conf.measure.blur_sigma)
                    )
        assert dec0 is not None
        z0, z1 = z_fit[:, 0], z_fit[:, 1]
        alpha = linear_dyn.select_alpha(z0, z1, z_zur["calib"])
        a, b = linear_dyn.fit(z0, z1, alpha=alpha)
        rows = []
        for s, (z, _) in sets.items():
            row: dict[str, Any] = {"group": s, "pairs": len(z), "alpha": alpha}
            if len(z) > 1:
                row |= {
                    "recon_blur": statistics.fmean(blur[s]),
                    "recon_blur_sd": _sd(blur[s]),
                    "recon_blur_k": blur[s],
                    "rank": effective_rank(z[:, 0]),
                }
                row |= linear_dyn.errors(a, b, z, [HORIZON], train=z0)
            rows.append(row)
        for s in ("span", "after"):
            z, x = sets[s]
            if len(z):
                self._picture(
                    f"{name}-{s}", code, z, x, lambda c: _decoded(dec0, std(c))
                )
        return rows

    def _zurich_pairs(self, split: str) -> Tensor:
        """The held-out windows of ``split`` as pairs u8 ``(m, 2, n, C, H,
        W)`` (their stacks 0 and 1)."""
        w, n = self._zurich[split], self._conf.data.tours.frames
        return torch.stack([w[:, :n], w[:, 1 : n + 1]], 1)

    def _codes(self, code: VaeCode, pairs: Tensor, d: int = 0) -> Tensor:
        """The codes ``(m, 2, D)`` of the pairs u8 ``(m, 2, n, C, H, W)``;
        ``(0, 2, d)`` of none."""
        if not len(pairs):
            return torch.empty(0, 2, d, device=self._dev)
        flat = pairs.flatten(0, 1)
        zs = [code.mu(c.to(self._dev)).float() for c in flat.split(BATCH)]
        return torch.cat(zs).unflatten(0, (len(pairs), 2))

    def _picture(
        self,
        name: str,
        code: VaeCode,
        z: Tensor,
        x: Tensor,
        post_of: Callable[[Tensor], Tensor],
    ) -> None:
        """``<name>.png`` of ``SHOWN`` of the pairs evenly spaced: the true
        last frames, the system's and the decoder's of seed 0."""
        idx = [i * len(z) // SHOWN for i in range(min(SHOWN, len(z)))]
        true = x[idx, 0, -1].to(self._dev).float() / 255
        own = code.decode(z[idx, 0])[:, -1].clamp(0, 1)
        grid = torch.stack([true, own, post_of(z[idx, 0])]).cpu()
        write_png(self._out / f"{name}.png", tile(grid, scale=SCALE))


def _decoded(dec: torch.nn.Module, z: Tensor) -> Tensor:
    return torch.cat([dec(c) for c in z.split(BATCH)]).clamp(0, 1)


def _sd(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else math.nan


def evaluate_stream(
    run_dir: Path,
    build: Callable[[RunConf], VaeCode],
    *,
    root: Path | None = None,
    out: Path | None = None,
    fit_steps: int = FIT_STEPS,
    spacing: Mapping[str, float] | None = None,
    span: float | None = None,
    probe_at: Sequence[float] = PROBE_AT,
) -> Path:
    """Evaluate every checkpoint ``ckpt/<t>.pt`` of the run in ``run_dir``
    (its ``config.yaml``; the VAE of ``build(conf)``, on ``conf.device``)
    in the order of ``t``, under ``provenance.configure_determinism``, into
    ``out`` (``run_dir / OUT``; module docstring), on its ``eval.stream_view``
    whose span is ``[0, span)`` (``spanned``); the probes only at the
    checkpoints of ``probe_at`` (seconds of the view) and the last one.
    ``spacing`` passes the spacings of ``StreamEvaluator`` by name. Returns
    ``out``.

    Raises:
        FileNotFoundError: The run has no checkpoint.
    """
    run = from_dict(yaml.safe_load((run_dir / "config.yaml").read_text()))
    view, c = stream_view(run)
    conf = spanned(view, span)
    determinism = configure_determinism(conf.device)
    paths = sorted(run_dir.glob("ckpt/*.pt"), key=lambda p: float(p.stem))
    if not paths:
        raise FileNotFoundError(f"no checkpoint under {run_dir / 'ckpt'}")
    times = [round(float(p.stem) * c, 6) for p in paths]  # of the view
    probed = [
        t for t in times if t == times[-1] or any(abs(t - a) < 1e-6 for a in probe_at)
    ]
    root = tours_root(conf.data.tours.data_version) if root is None else root
    out = run_dir / OUT if out is None else out
    ev = StreamEvaluator(
        conf, run_dir, root, out, fit_steps=fit_steps, c=c, **(spacing or {})
    )
    write_json(
        out / "config.json",
        ev.settings()
        | {
            "probe_at": probed,
            "ckpts": [p.name for p in paths],
            "stacks_per_frame": c,
            "span": conf.measure.duration,
            "determinism": determinism,
        },
    )
    write_json(
        out / "positions.json",
        {
            "k": ev.ks,
            "t": [k * conf.time.frame for k in ev.ks],
            "base": ev.base,
            "pre": ev.pre,
        },
    )
    for f in ("matrix.jsonl", "summary.jsonl", "probes.jsonl"):
        _write(out / f, [])
    code = build(run)
    for path, at in zip(paths, times, strict=True):
        head = ckpt.load_eval(path, code.modules())
        t, frame = round(head["t"] * c, 6), int(head["frame"]) * c
        err = ev.errors(code)
        stamp = {"t": t, "frame": frame}
        _append(out / "matrix.jsonl", [stamp | {"err": err}])
        row = ev.summary(err, frame) | {"zurich": ev.zurich(code)}
        _append(out / "summary.jsonl", [stamp | row])
        if at in probed:
            rows = ev.probes(code, path.stem)
            _append(out / "probes.jsonl", [stamp | r for r in rows])
    return out


def _lines(rows: Sequence[Mapping[str, Any]]) -> str:
    """JSON lines of ``rows``, non-finite floats as null."""

    def clean(x: Any) -> Any:
        if isinstance(x, list):
            return [clean(v) for v in x]  # pyright: ignore[reportUnknownVariableType]
        return None if isinstance(x, float) and not math.isfinite(x) else x

    return "".join(json.dumps({k: clean(v) for k, v in r.items()}) + "\n" for r in rows)


def _write(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with atomic_open(path) as f:
        f.write(_lines(rows).encode())


def _append(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("a") as f:
        f.write(_lines(rows))


def main(argv: Sequence[str] | None = None) -> Path:
    """``python -m exp.vision.stream_eval``: ``evaluate_stream`` of a run of
    the program ``"vae"``. Returns its ``posthoc_stream/``."""
    ap = argparse.ArgumentParser(
        prog="python -m exp.vision.stream_eval",
        description="Evaluate the checkpoints of a vae run on its stream.",
    )
    ap.add_argument("run_dir", type=Path, help="the run's directory")
    ap.add_argument("--root", type=Path, help="the data (default: tours_root)")
    ap.add_argument("--fit-steps", type=int, default=FIT_STEPS, help="per decoder")
    ap.add_argument("--span", type=float, help="seconds of the span (of the view)")
    ap.add_argument(
        "--probe-at",
        type=_seconds,
        default=PROBE_AT,
        help="comma-separated seconds of the view (default: PROBE_AT)",
    )
    args = ap.parse_args(argv)
    from exp.vision.models import build_vae  # the model, only for the shell

    return evaluate_stream(
        args.run_dir,
        lambda conf: build_vae(conf, seed=conf.seed),
        root=args.root,
        fit_steps=args.fit_steps,
        span=args.span,
        probe_at=args.probe_at,
    )


def _seconds(text: str) -> tuple[float, ...]:
    """``"1800,6000"`` as ``(1800.0, 6000.0)``."""
    return tuple(float(x) for x in text.split(","))


if __name__ == "__main__":
    print(main())
