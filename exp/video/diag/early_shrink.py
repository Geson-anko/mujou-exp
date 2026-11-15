r"""DIAGNOSIS ONLY (M17, 2026-11-14): why the held-out ``tok_within`` and
``pred`` of every run of ``m17-opt-v1`` fall within the first 60 s, even at
AdamW 1e-6. Not a program of the runs, and nothing of the runs imports it.

Every number comes from ``PackedMeasures.evaluate`` (the function that made
``baselines.json`` and every row of ``evals.jsonl``) on the val split, so the
time-0 values and the later ones are of one function on the same windows.

``live``: the run's own program (``VisionProgram.build`` of the run's
config, ``seconds`` and ``time.learn_interval`` replaced) driven frame by
frame as ``loop`` drives it (``advance``, then ``evaluate("val")`` every
``every`` seconds). With ``--learn off`` (``time.learn_interval = inf``:
``JEPABP`` takes no step) every row must equal the time-0 row: the
decisive test of the measurement. Each row also has ``weight_change`` of
the live weights from those at time 0; with ``--attribute``, a last row has
the ``attribution`` of the weights at the end.

``ckpt``: the run's initial weights (``build_vision`` with the run's seed, as
``programs.vision.rebuild``) and its evaluation checkpoints, the init row
against ``baselines.json``, then for each checkpoint its row,
``weight_change`` and, with ``--attribute``, ``attribution``.

``weight_change`` of a tensor $\theta$ from $\theta_0$, $\Delta = \theta -
\theta_0$: ``rel`` $= \lVert\Delta\rVert_F / \lVert\theta_0\rVert_F$,
``max_abs`` $= \max|\Delta|$, and for a matrix ``rel_op`` $=
\sigma_{\max}(\Delta) / \sigma_{\max}(\theta_0)$. A step that moves every
element of an $m \times n$ matrix by $\pm\eta$ in the signs of a rank-one
pattern has $\lVert\Delta\rVert_F = \sigma_{\max}(\Delta) = \eta\sqrt{mn}$:
Adam's steps of about $\eta$ an element are not small against weights of std
0.02 when they are coherent.

``output_change`` of a linear map $W x + b$ on the inputs $x$ it gets from
the val stacks: $\lVert \Delta W x + \Delta b\rVert_F / \lVert W_0 x +
b_0\rVert_F$ over the tokens. The weights' change is small or large by what
it does to the map's output: a step of $\pm\eta$ an element in the signs of
$\delta x^\top$ moves the output on an input like $x$ by about $\eta
\lVert x\rVert_1$ per output, while LeCun's $W_0$ (std $1/\sqrt n$) gives
about $\lVert x\rVert_2 / \sqrt n$: for inputs of $n$ elements of one size
the ratio is $\eta n$, large for the patch embedding's $n = \tau C p^2$.

``gain_bound`` (at time 0, of each linear map's inputs from the val
stacks): the most the ratio of ``output_change`` to ``rel`` can be, and the
inputs' effective dimension (``gain_bound``'s docstring).

``step_bound`` (at time 0, ``live`` only): each linear map's
``output_change`` per unit of a step of every weight element by $a / n$ and
every bias element by $a$ (``step_bound``'s docstring), with $n$ the map's
fan-in under ``optim_vision.scale`` ``fan_in`` and 1 otherwise: Adam's
steps of about $\eta$ an element a frame give $a \approx \eta F$ after $F$
frames, so ``output_change`` $/ (\eta F)$ is at most about ``step_bound``,
with equality for coherent signs on inputs on one line.

``attribution``: for each module that holds parameters, the row with that
module's tensors back at $\theta_0$ and every other at the checkpoint
(``revert``), and the row with only that module at the checkpoint
(``only``).

Run on a GPU host with the data (``$MUJOU_EXP_DATA``)::

    python -m exp.video.diag.early_shrink live <run_dir> <out> \
        --seconds 60 --every 10 [--learn off] [--attribute]
    python -m exp.video.diag.early_shrink ckpt <run_dir> <out> \
        [--ckpt 00300.0 ...] [--attribute]

``<run_dir>`` holds the run's ``job.json`` (and ``baselines.json``,
``ckpt/``); ``<out>`` gets ``rows.jsonl``.
"""

import argparse
import dataclasses
import json
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from exp.lib import ckpt
from exp.video.compose import from_dict
from exp.video.jepa.vit import build_vision
from exp.video.programs.vision import PACKED_BATCH, PackedMeasures, VisionProgram
from exp.video.schema.run import RunConf

KEYS = ("loss", "pred", "sigreg", "tok_within", "tok_between", "tok_temporal")

Params = dict[str, Tensor]


def params_of(modules: Mapping[str, nn.Module]) -> Params:
    """The parameters of ``modules`` by ``<module>.<name>`` (live tensors)."""
    return {
        f"{m}.{k}": p for m, mod in modules.items() for k, p in mod.named_parameters()
    }


def weight_change(init: Params, now: Params) -> dict[str, dict[str, float]]:
    """``rel``, ``max_abs`` and (matrices) ``rel_op`` of each tensor of
    ``now`` from ``init`` (module docstring); ``rel`` is ``inf`` for a
    tensor that was 0 and moved, ``nan`` for one still 0."""
    out = {}
    for name, p0 in init.items():
        a, d = p0.double(), now[name].detach().double() - p0.double()
        base, moved = float(a.norm()), float(d.norm())
        row = {
            "rel": moved / base if base > 0 else (math.inf if moved else math.nan),
            "max_abs": float(d.abs().max()) if d.numel() else 0.0,
        }
        if a.dim() == 2:
            s0 = float(torch.linalg.matrix_norm(a, ord=2))
            row["rel_op"] = float(torch.linalg.matrix_norm(d, ord=2)) / s0
        out[name] = row
    return out


def linear_inputs(
    modules: Mapping[str, nn.Module], forward: Callable[[], object]
) -> dict[str, Tensor]:
    """The inputs ``(k, n)`` each ``nn.Linear`` of ``modules`` gets in
    ``forward()`` (its rows over every call), by the name of its weight
    without ``.weight``; a map that gets none is left out."""
    seen: dict[str, list[Tensor]] = {}
    hooks = []
    for m, mod in modules.items():
        for sub, lin in mod.named_modules():
            if isinstance(lin, nn.Linear):
                name = f"{m}.{sub}" if sub else m
                seen[name] = []
                hooks.append(
                    lin.register_forward_hook(
                        lambda _m, x, _y, name=name: seen[name].append(x[0].detach())
                    )
                )
    try:
        with torch.no_grad():
            forward()
    finally:
        for h in hooks:
            h.remove()
    return {
        k: torch.cat([x.flatten(0, -2) for x in xs]) for k, xs in seen.items() if xs
    }


def output_change(
    modules: Mapping[str, nn.Module], init: Params, forward: Callable[[], object]
) -> dict[str, float]:
    """``output_change`` (module docstring) of each ``nn.Linear`` of
    ``modules`` on the inputs it gets in ``forward()`` (``linear_inputs``);
    ``init`` holds the weights at the start."""
    now = params_of(modules)
    out = {}
    for name, x in linear_inputs(modules, forward).items():
        x = x.double()
        w0, b0 = init[f"{name}.weight"].double(), init[f"{name}.bias"].double()
        w = now[f"{name}.weight"].detach().double()
        b = now[f"{name}.bias"].detach().double()
        moved = float((x @ (w - w0).T + (b - b0)).norm())
        out[name] = moved / float((x @ w0.T + b0).norm())
    return out


def gain_bound(x: Tensor) -> dict[str, float]:
    r"""How much more the output of a linear map on the inputs ``x``
    ``(k, n)`` can move, relatively, than its weights: for $W_0$ of i.i.d.
    entries $\lVert x W_0^\top\rVert_F \approx \lVert W_0\rVert_F
    \lVert x\rVert_F / \sqrt n$, and $\lVert x \Delta W^\top\rVert_F \le
    \lVert\Delta W\rVert_F\,\sigma_{\max}(x)$, so the output's relative
    change over the weights' is at most ``bound`` $= \sqrt n\,
    \sigma_{\max}(x) / \lVert x\rVert_F = \sqrt{n \lambda_{\max} /
    \sum_i \lambda_i}$ ($\lambda$ the eigenvalues of $x^\top x$; 1 for
    isotropic inputs, $\sqrt n$ for inputs on one line), reached by a change
    along the inputs' top direction. Also ``participation`` $(\sum
    \lambda)^2 / \sum \lambda^2$, the inputs' effective dimension, and
    ``n``."""
    lam = torch.linalg.eigvalsh(x.double().T @ x.double())
    total = float(lam.sum())
    return {
        "n": x.shape[1],
        "bound": math.sqrt(x.shape[1] * float(lam.max()) / total),
        "participation": total**2 / float((lam**2).sum()),
    }


def step_bound(x: Tensor, w0: Tensor, b0: Tensor, n: int) -> float:
    r"""The ``output_change`` on the inputs ``x`` ``(k, n_in)`` of a map
    $W_0 x + b_0$ ($W_0$ ``(m, n_in)``) per unit $a$ of a change with
    $|\Delta W_{ij}| \le a / n$ and $|\Delta b_i| \le a$: $|(\Delta W x_t +
    \Delta b)_i| \le a (\lVert x_t\rVert_1 / n + 1)$, so ``output_change``
    $\le a\,\sqrt{m \sum_t (\lVert x_t\rVert_1 / n + 1)^2} / \lVert x
    W_0^\top + b_0\rVert_F$, the value returned. Equality for one input and
    the sign step $\Delta W = (a / n)\,s\,\mathrm{sign}(x)^\top$, $\Delta b =
    a\,s$ ($s = \pm 1$)."""
    x = x.double()
    per = x.abs().sum(1) / n + 1
    base = float((x @ w0.double().T + b0.double()).norm())
    return math.sqrt(w0.shape[0]) * float(per.norm()) / base


def owners(names: Sequence[str]) -> dict[str, list[str]]:
    """The tensors ``names`` grouped by the module that holds them (the name
    without its last part), in order."""
    out: dict[str, list[str]] = {}
    for name in names:
        out.setdefault(name.rpartition(".")[0], []).append(name)
    return out


@contextmanager
def swapped(params: Params, values: Mapping[str, Tensor]) -> Iterator[None]:
    """``params`` named in ``values`` hold ``values`` inside the block, and
    their own values again after it (bit for bit)."""
    kept = {k: params[k].detach().clone() for k in values}
    with torch.no_grad():
        for k, v in values.items():
            params[k].copy_(v)
    try:
        yield
    finally:
        with torch.no_grad():
            for k, v in kept.items():
                params[k].copy_(v)


def attribution(
    params: Params, init: Params, measure: Callable[[], dict[str, float]]
) -> dict[str, dict[str, dict[str, float]]]:
    """For each module of ``params``, ``measure()`` with its tensors at
    ``init`` (``revert``) and with every other tensor at ``init`` (``only``)."""
    out = {}
    for owner, names in owners(list(params)).items():
        mine = set(names)
        with swapped(params, {k: init[k] for k in names}):
            revert = measure()
        with swapped(params, {k: v for k, v in init.items() if k not in mine}):
            only = measure()
        out[owner] = {"revert": revert, "only": only}
    return out


def same_value(a: float, b: float) -> bool:
    """``a`` and ``b`` are the same float (two NaNs are)."""
    return a == b or (math.isnan(a) and math.isnan(b))


def pick(row: Mapping[str, float]) -> dict[str, float]:
    """The ``KEYS`` of an evaluation row."""
    return {k: row[k] for k in KEYS if k in row}


def run_conf(run_dir: Path) -> RunConf:
    """The config of the run in ``run_dir`` (its ``job.json``), on CUDA."""
    job = json.loads((run_dir / "job.json").read_text())
    return dataclasses.replace(from_dict(job["config"]), device="cuda")


def _write(out: Path, row: Mapping[str, Any]) -> None:
    with (out / "rows.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    big = ("weight_change", "output_change", "gain_bound", "step_bound", "attribution")
    print(json.dumps({k: v for k, v in row.items() if k not in big}))


def live(args: argparse.Namespace) -> None:
    conf = run_conf(args.run_dir)
    learn = math.inf if args.learn == "off" else conf.time.learn_interval
    conf = dataclasses.replace(
        conf,
        seconds=args.seconds,
        time=dataclasses.replace(conf.time, learn_interval=learn),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    built = VisionProgram().build(conf, args.out, None)
    try:
        params = params_of(built.modules())
        init = {k: p.detach().clone() for k, p in params.items()}
        s = built.schedule
        every = round(args.every / s.seconds)
        online: Any = built  # the vision's _Online: its modules and measures
        stacks = online.measures.windows["val"][:PACKED_BATCH, : online.mods.chunk[0]]
        stacks = stacks.to(conf.device)
        row0 = built.evaluate("val")
        inputs = linear_inputs(built.modules(), lambda: online.mods.tokens(stacks))
        fan_in = conf.optim_vision is not None and conf.optim_vision.scale == "fan_in"
        _write(
            args.out,
            {
                "t": 0.0,
                "frame": 0,
                "learn": args.learn,
                **pick(row0),
                "gain_bound": {k: gain_bound(x) for k, x in inputs.items()},
                "step_bound": {
                    k: step_bound(
                        x,
                        init[f"{k}.weight"],
                        init[f"{k}.bias"],
                        x.shape[1] if fan_in else 1,
                    )
                    for k, x in inputs.items()
                },
            },
        )
        for u in range(1, s.units + 1):
            built.advance()
            if u % every and u != s.units:
                continue
            row = built.evaluate("val")
            same = {k: same_value(row[k], row0[k]) for k in row0 if k in row}
            _write(
                args.out,
                {
                    "t": s.t(u),
                    "frame": u,
                    "learn": args.learn,
                    **pick(row),
                    "stream": built.log(),
                    "equal_to_t0": all(same.values()),
                    "differ": sorted(k for k, ok in same.items() if not ok),
                    "weight_change": weight_change(init, params),
                    "output_change": output_change(
                        built.modules(), init, lambda: online.mods.tokens(stacks)
                    ),
                },
            )
        if args.attribute:
            measure = lambda: pick(built.evaluate("val"))  # noqa: E731
            _write(args.out, {"t": s.t(s.units), **attribution(params, init, measure)})
    finally:
        built.close()


def checkpoints(args: argparse.Namespace) -> None:
    conf = run_conf(args.run_dir)
    args.out.mkdir(parents=True, exist_ok=True)
    measures = PackedMeasures(conf)
    mods = build_vision(conf, seed=conf.seed)  # programs.vision.rebuild's
    params = params_of(mods.named())
    init = {k: p.detach().clone() for k, p in params.items()}
    stacks = measures.windows["val"][:PACKED_BATCH, : mods.chunk[0]].to(conf.device)

    def measure() -> dict[str, float]:
        return pick(measures.evaluate(mods, "val"))

    def forward() -> Tensor:
        return mods.tokens(stacks)

    base = json.loads((args.run_dir / "baselines.json").read_text())["val"]
    row0 = measure()
    _write(
        args.out,
        {
            "ckpt": "init",
            **row0,
            "baselines": {k: base[f"init_{k}"] for k in row0},
            "gain_bound": {
                k: gain_bound(x)
                for k, x in linear_inputs(mods.named(), forward).items()
            },
        },
    )
    for name in args.ckpt:
        header = ckpt.load_eval(args.run_dir / "ckpt" / f"{name}.pt", mods.named())
        row: dict[str, Any] = {"ckpt": name, "t": header["t"], **measure()}
        row["weight_change"] = weight_change(init, params)
        row["output_change"] = output_change(mods.named(), init, forward)
        if args.attribute:
            row["attribution"] = attribution(params, init, measure)
        _write(args.out, row)


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="the early shrink of tok_within (M17)")
    sub = p.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("live")
    a.add_argument("run_dir", type=Path)
    a.add_argument("out", type=Path)
    a.add_argument("--seconds", type=float, default=60.0)
    a.add_argument("--every", type=float, default=10.0)
    a.add_argument("--learn", choices=("on", "off"), default="on")
    a.add_argument("--attribute", action="store_true")
    c = sub.add_parser("ckpt")
    c.add_argument("run_dir", type=Path)
    c.add_argument("out", type=Path)
    c.add_argument("--ckpt", nargs="*", default=["00300.0"])
    c.add_argument("--attribute", action="store_true")
    args = p.parse_args(argv)
    {"live": live, "ckpt": checkpoints}[args.mode](args)


if __name__ == "__main__":
    main()
