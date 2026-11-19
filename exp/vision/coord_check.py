r"""The coord check of V14 (``plans/optim-study.md``): does the change of each
node's output in the first frames depend on the width? In muP it should not;
in SP it should grow with the width::

    MUJOU_EXP_DATA=... uv run python -m exp.vision.coord_check --device cuda \
        [--widths 128,256,512,1024] [--frames 0,10,50,100] [--out coord.json] \
        [override ...]

For each parametrization (``--params``, ``sp`` and ``mup``) and width it
learns the program's ``Online`` of ``+experiment=<--experiment>
task=bench optim=rmsprop optim.tau=1e4 model.width=<w>
model.scale_decoder=true model.param=<p>`` and the overrides from the start
of the stream (``--random``: synthetic stacks, no data). At each frame of
``--frames`` (0: the initialization) it feeds the stacks of the first frame
(the probe) through the node modules as they are (``node_outputs``) and
prints per node the RMS of the output and the RMS of its change from frame 0
(``delta``). Then, per parametrization, node and frame after 0, the slope of
$\log \text{delta}$ (and of $\log \text{rms}$) against $\log$ width
(``loglog_slope``): about 0 in muP. The rows and the slopes are also written
to ``--out`` as JSON.
"""

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from exp.vision.bench import RandomStacks
from exp.vision.compose import compose
from exp.vision.models import VAEModules, early_fusion
from exp.vision.program import Online, tour_streams
from exp.vision.schema.run import RunConf

__all__ = ["NODES", "coord_rows", "loglog_slope", "main", "node_outputs"]

NODES = ("enc1", "enc2", "enc3", "enc4", "dec1", "dec2", "out")


def node_outputs(mods: VAEModules, stacks: Tensor) -> dict[str, Tensor]:
    r"""The output of each node module (``NODES``) for ``stacks`` u8 ``(B, n,
    3, H, W)``, the decoder fed $z = \mu$."""
    out: dict[str, Tensor] = {}
    h = early_fusion(stacks).to(mods.log_sigma.dtype)
    for i, part in enumerate(mods.encoder.parts):
        h = part(h)
        out[f"enc{i + 1}"] = h
    z = h[:, : mods.latent]
    for i, part in enumerate(mods.decoder.parts):
        z = part(z)
        out[f"dec{i + 1}"] = z
    out["out"] = mods.decoder.readout(z)
    return out


def _rms(x: Tensor) -> float:
    return float(x.double().square().mean().sqrt())


def coord_rows(
    conf: RunConf, online: Online, probe: Tensor, frames: Sequence[int]
) -> list[dict[str, Any]]:
    """The rows ``{"param", "width", "frame", "node", "rms", "delta"}`` of
    ``online`` (fresh, of ``conf``) at each of ``frames``, of the ``probe``
    stacks. It does not close ``online``."""
    mods = online.net.mods
    with torch.no_grad():
        start = node_outputs(mods, probe)
    rows, done = [], 0
    for frame in sorted(frames):
        while done < frame:
            online.advance()
            online.rows()  # gradient statistics, if any: dropped
            done += 1
        with torch.no_grad():
            now = node_outputs(mods, probe)
        for node in NODES:
            rows.append(
                {
                    "param": conf.model.param,
                    "width": conf.model.width,
                    "frame": frame,
                    "node": node,
                    "rms": _rms(now[node]),
                    "delta": _rms(now[node] - start[node]),
                }
            )
    return rows


def loglog_slope(widths: Sequence[float], values: Sequence[float]) -> float | None:
    r"""The least-squares slope of $\log$ ``values`` against $\log$
    ``widths`` (the exponent of a power law), or ``None`` when a value is not
    finite and > 0."""
    if not all(math.isfinite(v) and v > 0 for v in values):
        return None
    xs = [math.log(w) for w in widths]
    ys = [math.log(v) for v in values]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sxx


def _slopes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per parametrization, node and frame after 0, the slopes over the
    widths of ``delta`` and ``rms``."""
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for r in rows:
        if r["frame"] > 0:
            groups.setdefault((r["param"], r["node"], r["frame"]), []).append(r)
    out = []
    for (param, node, frame), rs in groups.items():
        rs = sorted(rs, key=lambda r: r["width"])
        widths = [r["width"] for r in rs]
        out.append(
            {
                "param": param,
                "node": node,
                "frame": frame,
                "delta_slope": loglog_slope(widths, [r["delta"] for r in rs]),
                "rms_slope": loglog_slope(widths, [r["rms"] for r in rs]),
            }
        )
    return out


def _fmt(x: float | None) -> str:
    return "     -" if x is None else f"{x:+.2f}"


def main(argv: Sequence[str] | None = None) -> dict[str, list[dict[str, Any]]]:
    ap = argparse.ArgumentParser(
        prog="python -m exp.vision.coord_check",
        description="The RMS of each node's output and its change by width.",
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--experiment", default="v0/base")
    ap.add_argument("--widths", default="128,256,512,1024")
    ap.add_argument("--frames", default="0,10,50,100")
    ap.add_argument("--params", default="sp,mup")
    ap.add_argument("--random", action="store_true", help="synthetic stacks")
    ap.add_argument("--out", type=Path)
    ap.add_argument("overrides", nargs="*", help="hydra overrides of every run")
    args = ap.parse_args(argv)
    widths = [int(w) for w in args.widths.split(",")]
    frames = [int(f) for f in args.frames.split(",")]
    rows: list[dict[str, Any]] = []
    print("param width node  frame       rms     delta", flush=True)
    for param in args.params.split(","):
        for width in widths:
            conf = compose(
                [
                    f"+experiment={args.experiment}",
                    "task=bench",
                    "optim=rmsprop",
                    "optim.tau=1e4",
                    f"model.width={width}",
                    "model.scale_decoder=true",
                    f"model.param={param}",
                    f"device={args.device}",
                    *args.overrides,
                ]
            )
            stream = RandomStacks if args.random else tour_streams
            probe = stream(conf).next()[0].to(conf.device)
            online = Online(conf, None, stream(conf))
            try:
                got = coord_rows(conf, online, probe, frames)
            finally:
                online.close()
            for r in got:
                print(
                    f"{param:5} {width:5} {r['node']:5} {r['frame']:5} "
                    f"{r['rms']:9.3e} {r['delta']:9.3e}",
                    flush=True,
                )
            rows += got
    slopes = _slopes(rows)
    print("param node  frame  delta_slope  rms_slope")
    for s in slopes:
        print(
            f"{s['param']:5} {s['node']:5} {s['frame']:5}  "
            f"{_fmt(s['delta_slope']):>11}  {_fmt(s['rms_slope']):>9}"
        )
    result = {"rows": rows, "slopes": slopes}
    if args.out is not None:
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    main()
