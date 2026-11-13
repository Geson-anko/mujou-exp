r"""DIAGNOSIS ONLY (M17, 2026-11-13): two numbers needed before the runs of
the SIGReg of the data points on one stream (``m17/ssig_*``): the lowest
$\lambda$ at which tokens constant within a stack stop being cheaper than iid
tokens, and the speed of the two ways of storing the buffer.

- ``bound``: the representation of a stack is its $T$ target tokens of $d$
  each, flattened ($x \in \mathbb{R}^{Td}$), and SIGReg sees $n$ of them
  (the current stack and the buffer). Tokens constant within each stack,
  $s_{b,t} = c_b$ with $c_b \sim N(0, k I_d)$ iid over the stacks, make the
  prediction of the masked tokens exact (the mean of the context is $c_b$).
  Along a unit direction $a = (a_1, \dots, a_T)$ their projection is
  $a^\top x_b = s_a^\top c_b \sim N(0, k \lVert s_a \rVert^2)$, $s_a = \sum_t
  a_t$; for $a$ uniform on the sphere, $\lVert s_a \rVert^2 = T \lVert P a
  \rVert^2$ ($P$ the projection on the $d$ directions of the repeated
  token), of mean 1 and relative variance about $2/d$: the larger $d$, the
  closer to $N(0, 1)$ every projection of the collapse looks. For $n$ iid
  samples of $N(0, v)$, the expectation of the statistic of
  ``exp.lib.sigreg`` (the trapezoid rule on ``T_POINTS``) is exactly

  $$E\,S(v) = \int \left[(1 - e^{-v t^2}) + n (e^{-v t^2 / 2} -
  e^{-t^2/2})^2\right] e^{-t^2/2}\, dt$$

  (``expected_s``; $E |\hat\varphi|^2 = 1/n + (1 - 1/n) e^{-v t^2}$). The
  collapse costs $\lambda S_c$ with $S_c = \min_k E_a E\,S(k \lVert s_a
  \rVert^2)$ (the encoder picks the scale), against $(1 - \lambda) + \lambda
  S_0$, $S_0 = E\,S(1)$, of iid $N(0, 1)$ tokens predicted by their mean 0
  (an error of 1 per dimension). It is the dearer of the two only for

  $$\lambda > \lambda^* = \frac{1}{1 + S_c - S_0}.$$

  ``bound`` gives $\lambda^*$ for the arms $(T, d)$ of ``m17/ssig_*`` at
  $n$ = 257 (the buffer of 256 and the current stack). It compares two
  points of the loss, not its minimum: below $\lambda^*$ the collapse is
  cheaper than iid tokens; above, the loss still may have other cheap
  solutions. The iid tokens are held at $v = 1$ there; the encoder may pick
  their scale as well, of cost $(1 - \lambda) v + \lambda E\,S(v)$ (the
  error of predicting $N(0, v)$ by 0 is $v$), and the collapse is dearer than
  the cheapest of them only for

  $$\lambda > \lambda^*_v = \min_v \frac{v}{v + S_c - E\,S(v)}$$

  (``lam_star_scaled``, on the same grid of scales; the collapse is dearer
  than iid tokens of variance $v$ iff $\lambda$ exceeds the term of $v$),
  $\lambda^*_v \le \lambda^*$, the term of $v = 1$. Either bound rules out
  one solution only, the code of $d$ numbers per stack ($m = d$ below): tokens
  $s_{b,t} = W_t c_b$ of a latent $c_b \in \mathbb{R}^m$, $m > d$, that the
  visible tokens determine also predict exactly, and their projections
  $(\sum_t W_t^\top a_t)^\top c_b$ have a spread over the directions of
  relative variance about $2/m$: as $m$ grows their statistic falls to
  $S_0$, below that of iid tokens for every $\lambda < 1$. Whether such a
  code is a collapse depends on $m$, which only the learned code tells
  (its rank).
- ``bench``: the learning steps per second of ``JEPABP`` with the buffer full
  (random rows: the cost does not depend on the values), the stored
  representations (``z``) against the stored inputs encoded anew at each
  step (``input``), on random u8 stacks of the preset's shape, after
  ``warmup`` frames. A step is a frame of the run (``time.frame`` 1/30 s of
  virtual time, in which the stream advances 0.1 s): ``JEPABP`` evaluates its
  loss once in a frame and, ``time.learn_interval`` (25 ms) being shorter
  than a frame, learns once in it (``mujou.pc.clock.LearnClock`` learns 25
  ms after the input's write and starts again at the next frame), so a step
  is also one learning step; with ``input`` each one encodes the whole buffer
  anew. A run of 1800 s is 54000 frames, whatever its speed. With ``--preset
  gkl`` (``m17/gkl_*``, the Gaussian moment matching of the data points,
  ``exp.lib.gauss_kl``) it measures the stored representations only (the
  law refuses the stored inputs); each step's insertion then makes the
  buffer's moments anew (its scatter, Cholesky factor and precision).

Run: ``python -m exp.video.diag.ssig bound`` and ``python -m
exp.video.diag.ssig bench --arm t128 --device cpu [--preset gkl]`` (``cpu``
or ``cuda[:i]``, as ``RunConf.device``).
"""

import argparse
import json
import math
import time
from collections.abc import Sequence
from typing import Any

import mujou.pc as pc
import mujou.testing
import torch
from torch import Tensor

from exp.lib.sigreg import T_POINTS
from exp.video.compose import compose
from exp.video.jepa.nets import VisionBP, build_vision_net

__all__ = [
    "ARMS",
    "SCALES",
    "bench",
    "bound",
    "expected_s",
    "main",
    "spread_of_sums",
]

ARMS = {"t512": (512, 2), "t128": (128, 8), "t64": (64, 16)}  # (T, d), D = 1024
# the scales of the collapse and of the iid tokens: 481 points, 1/8 to 8, log
SCALES = torch.exp(torch.linspace(math.log(1 / 8), math.log(8), 481))


def expected_s(v: Tensor, n: int, t: Tensor = T_POINTS) -> Tensor:
    """$E\\,S$ of ``n`` iid $N(0, v)$ samples along one direction, of each
    variance of ``v`` (module docstring), by the trapezoid rule on ``t`` as
    ``exp.lib.sigreg.epps_pulley``."""
    t = t.to(torch.float64)
    v = v.to(torch.float64)[..., None]
    w = torch.exp(-(t**2) / 2)
    per = (1 - torch.exp(-v * t**2)) + n * (torch.exp(-v * t**2 / 2) - w) ** 2
    return torch.trapezoid(per * w, t, dim=-1)


def spread_of_sums(t: int, d: int, draws: int, gen: torch.Generator) -> Tensor:
    """$\\lVert s_a \\rVert^2$, $s_a = \\sum_t a_t$, of ``draws`` directions
    $a$ uniform on the unit sphere of $\\mathbb{R}^{T d}$ ``(draws,)``."""
    a = torch.randn(draws, t, d, generator=gen, dtype=torch.float64)
    a = a / a.flatten(1).norm(dim=1)[:, None, None]
    return (a.sum(1) ** 2).sum(1)


def bound(t: int, d: int, n: int, *, draws: int = 20000, seed: int = 0) -> dict:
    """$\\lambda^*$ and $\\lambda^*_v$ of the arm $(T, d)$ at ``n`` samples
    (module docstring): the mean over ``draws`` directions, the scales on
    ``SCALES``."""
    s2 = spread_of_sums(t, d, draws, torch.Generator().manual_seed(seed))
    costs = torch.stack([expected_s(k * s2, n).mean() for k in SCALES])
    i = int(costs.argmin())
    s_c, s_0 = float(costs[i]), float(expected_s(torch.tensor(1.0), n))
    v = SCALES.to(torch.float64)
    terms = v / (v + s_c - expected_s(v, n))  # lambda at which iid of v ties
    j = int(torch.where(terms > 0, terms, math.inf).argmin())
    return {
        "T": t,
        "d": d,
        "n": n,
        "s_collapse": s_c,
        "scale": float(SCALES[i]),
        "s_iid": s_0,
        "spread_var": float(s2.var()),
        "lam_star": 1 / (1 + s_c - s_0),
        "scale_iid": float(v[j]),
        "lam_star_scaled": float(terms[j]),
    }


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    elif dev.type == "mps":
        torch.mps.synchronize()


def bench(
    overrides: Sequence[str], *, store: str, frames: int, warmup: int
) -> dict[str, Any]:
    """The steps per second of ``JEPABP`` of the run ``overrides`` with
    ``vision.datapoint_store`` = ``store`` and its buffer full (module
    docstring)."""
    conf = compose(
        [*overrides, "learner=bp", "data.streams=1", f"vision.datapoint_store={store}"]
    )
    net = build_vision_net(conf)
    assert isinstance(net, VisionBP)
    points = net.vision.points
    if points is None:
        raise ValueError("the run has no buffer of data points")
    dev = torch.device(conf.device)
    g = torch.Generator().manual_seed(0)
    if store == "input":
        rows = torch.randint(0, 256, points.rows.shape, generator=g).to(torch.uint8)
    else:
        rows = torch.randn(points.rows.shape, generator=g)
    points.rows.copy_(rows.to(dev))
    points.count = len(rows)
    v, tours = conf.vision, conf.data.tours
    assert v is not None and tours is not None
    shape = (1, v.frames, 3, tours.size, tours.size)
    tau = pc.TimeConstants(h=conf.time.h, lam=conf.time.lam, r=conf.time.r)
    updates = round(conf.time.frame / tau.period)
    ones = torch.ones(1, dtype=torch.bool, device=dev)
    started = 0.0
    with mujou.testing.run(net) as run:
        for k in range(warmup + frames):
            if k == warmup:
                _sync(dev)
                started = time.perf_counter()
            x = torch.randint(0, 256, shape, generator=g).to(torch.uint8).to(dev)
            sample = pc.msg.Sample(
                x=x, y=torch.zeros(1, 1, device=dev), mask=ones, reset=ones & (k == 0)
            )
            run.write({net.prev: sample})
            for _ in range(updates):
                run.step(tau.period)
        _sync(dev)
    seconds = time.perf_counter() - started
    assert points.count == len(points.rows)
    return {
        "store": store,
        "buffer": len(points.rows),
        "device": conf.device,
        "frames": frames,
        "seconds": seconds,
        "steps_per_s": frames / seconds,
    }


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="SIGReg of the data points (diagnosis).")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bound")
    b.add_argument("--n", type=int, default=257)
    c = sub.add_parser("bench")
    c.add_argument("--arm", choices=sorted(ARMS), default="t128")
    c.add_argument("--preset", choices=["ssig", "gkl"], default="ssig")
    c.add_argument("--device", default="cpu")
    c.add_argument("--frames", type=int, default=20)
    c.add_argument("--warmup", type=int, default=3)
    c.add_argument("overrides", nargs="*")
    args = p.parse_args(argv)
    if args.cmd == "bound":
        out: Any = [bound(t, d, args.n) for t, d in ARMS.values()]
    else:
        over = [f"+experiment=m17/{args.preset}_{args.arm}", f"device={args.device}"]
        over += args.overrides
        stores = ("z", "input") if args.preset == "ssig" else ("z",)
        out = [
            bench(over, store=s, frames=args.frames, warmup=args.warmup) for s in stores
        ]
        if len(out) == 2:
            out.append({"ratio": out[1]["steps_per_s"] / out[0]["steps_per_s"]})
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
