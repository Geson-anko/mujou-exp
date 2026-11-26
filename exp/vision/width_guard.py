r"""The guard of the linear horizon by the flow decoder (``plans/jepa.md``,
「線形の幅（2026-11-26）」の「拡散のデコーダのガード」; README, the same
section's 「拡散のデコーダのガード」, registered before any number): decode the
code's extrapolation $2z_t - z_{t-\Delta}$ (the main) and midpoint
$\tfrac12(z_{t-\Delta} + z_{t+\Delta})$ (the side), and read whether the
drawings pick the right moment among the candidates near in time, by
Inception's features (``metrics.identification``), not by a distance or FID.
A description: the guard's verdict is ``results/v18c/judge_width.py``'s.

**The decoder** (``fit_decoder``): G's (``gen_eval``) ``FlowUNet`` of the
same settings (``GenConf``; the seed of decoder $j$ ``DEC_SEED +
DEC_SEED_STEP`` $j$), learned on G's training set (131072 stacks of the
training videos, ``gen_eval.training_set``), whose condition is the very
coordinates the width is measured in: the code's place whitened by its
reference (``midpoint.Whitener``, PCA64; C8), then ``eval.Standardizer`` of
the training set's (an affine map: the midpoint and the extrapolation
commute with it). Saved as G's ``ckpt/dec-width-<code>-d<j>.pt``.

**The anchors** (``guard_anchors``): of each lag of ``GuardConf.lags``, the
``in`` centres $t$ whose ten stacks $t - 4\Delta .. t + 5\Delta$ lie in one
128 s block, ``anchors`` of them evenly spread (all if fewer). The
candidates (``width_eval.slots``): of the extrapolation the last frames of
$t + \Delta + j\Delta$, of the midpoint of $t + j\Delta$, $j = -4 .. 4$, the
true one in the middle (chance 1/9; a candidate may cross a cut).

**The drawings** (``Guard.code_rows``): ``samples`` images of each anchor
from the noise of ``noise_seed`` (the same for every condition and code), of
the conditions ``past`` $z_{t-\Delta}$, ``now`` $z_t$, ``future``
$z_{t+\Delta}$, ``ext``, ``mid``, ``shuffle`` (the future of another anchor of
the same block: what a decoder hits by the scene alone; C9) and ``pos``
($z_{t+\Delta}$ moved by ``ext``'s distance to its nearest training code in a
random direction of ``pos_seed``: a positive control of the drift off the
codes; C10. The distance is to the decoder's training conditions, the codes
it learned on, not the reference's. In $k = 64$ dimensions a random
direction is nearly orthogonal to the way to the nearest code, so ``pos``
lies about $\sqrt{d_{\mathrm{fut}}^2 + d_{\mathrm{ext}}^2}$ from it,
farther out than ``ext``: its own distance is recorded beside). Each
drawing's features less the mean of $R$ (G's reference
frames) pick one candidate by the nearest cosine; a row holds each
condition's hits on its target (``future``, ``now``, ``ext``, ``shuffle``,
``pos`` on the extrapolation's; ``past``, ``now``, ``future``, ``mid`` on
the midpoint's), ``pick_max`` and the same of the local features (2 x 2 of
Inception's last maps; C11), with the anchors' blocks for the bootstrap
(``diff_se``).

**The controls** (``Guard.control_rows``, of no decoder): the pixels' mean
image $\tfrac12(x_{t-\Delta} + x_{t+\Delta})$ and the features' mean on the
midpoint's target; the features' $2\phi(x_t) - \phi(x_{t-\Delta})$, the
pixels' $\mathrm{clip}(2x_t - x_{t-\Delta})$ and the real frame 0.1 s after
the target on the extrapolation's.

**The cheats** (V-h2, ``cheat_codes``): of ``pca``'s whitened coordinates
$w$, (i) their mean over the past window $[u - w, u]$ of ``cheat_windows``
stacks, (ii) the fastest ``cheat_drops`` directions (by $s_i$ at
``cheat_lag`` on the training intervals, videos other than Zurich) set to
the reference's mean 0; strength 0 the identity. Their training set
(``train_intervals``): ``cheat_count`` runs of ``cheat_length`` consecutive
stacks of the training videos (seed ``cheat_seed``), each run's middle half
the targets (the earlier stacks feed the past window).

::

    python -m exp.vision.width_guard yard --config <v18c run> --sets runs/v18c-tlin \
        --out runs/v18c-width/guard --v17 <ckpt> --dec runs/v18c-gen/yardsticks/posthoc_gen \
        --fit pca:0,1,2 --fit v17:0
    python -m exp.vision.width_guard cheat --config <v18c run> --sets runs/v18c-tlin \
        --out runs/v18c-width/guard/cheat [--only base,mean-8]
    python -m exp.vision.width_guard run <run_dir> --steps 60000 --sets runs/v18c-tlin
"""

import argparse
import json
import math
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from exp.lib import ckpt, pca
from exp.lib.flow_decoder import FlowUNet, euler_sample, fit_flow
from exp.lib.inception import load_fid_inception, sha256, weights_path
from exp.lib.metrics import identification, ratio_resamples
from exp.lib.midpoint import Whitener
from exp.lib.png import tile, write_png
from exp.lib.provenance import configure_determinism
from exp.lib.tours.stacks import StackedTourFrames
from exp.lib.tours.walking_tours import videos
from exp.lib.video_io import U8Reader
from exp.vision.eval import Standardizer
from exp.vision.fill_eval import FillConf
from exp.vision.gen_eval import DEC_SEED, DEC_SEED_STEP, GenConf, spread, training_set
from exp.vision.geom_eval import (  # the geom job's own helpers, reused as they are
    GeomConf,
    JepaPlaces,
    VaeTokenCode,
    _conf_of,
    _feature_code,
    _jepa_of,
    _start,
    _write,
    encode_places,
)
from exp.vision.jepa_diag import run_ckpts
from exp.vision.jepa_eval import PCA_SEED, PCACode, pixels, vae_at
from exp.vision.models import VAEModules
from exp.vision.width_eval import (
    OUT as WIDTH_OUT,
    WidthConf,
    WidthSets,
    conditions,
    counts_of,
    lin_sums,
    sets_and_ref,
    slots,
    white_rows,
)

__all__ = [
    "OUT",
    "Guard",
    "GuardConf",
    "WhiteCode",
    "cheat_codes",
    "diff_se",
    "evaluate_cheat",
    "evaluate_run",
    "evaluate_yard",
    "fit_decoder",
    "guard_anchors",
    "load_decoder",
    "main",
    "same_k",
    "save_decoder",
    "train_intervals",
    "winning_lags",
]

OUT = "posthoc_width_guard"
EXT_CONDS = ("future", "now", "ext", "shuffle", "pos")
MID_CONDS = ("past", "now", "future", "mid")

type Row = dict[str, Any]
type Features = Callable[[Tensor], Tensor]
type Places = Callable[[Tensor], dict[str, Tensor]]


@dataclass(frozen=True)
class GuardConf:
    """The settings (module docstring; ``plans/jepa.md``), by default the
    plan's: lags and windows in stacks of 0.1 s."""

    lags: tuple[int, ...] = (8, 16, 32, 64)
    anchors: int = 512
    samples: int = 2
    reach: int = 4  # candidates j = -reach .. reach
    noise_seed: int = 79
    shuffle_seed: int = 83
    pos_seed: int = 89
    picture_every: int = 64
    picture_scale: int = 1
    cells: int = 2  # the local features' cells a side
    k: int = 64
    cheat_windows: tuple[int, ...] = (8, 32, 64, 128)
    cheat_drops: tuple[int, ...] = (16, 32, 48, 56)
    cheat_lag: int = 8
    cheat_count: int = 512
    cheat_length: int = 512
    cheat_seed: int = 1401


def same_k(guard: GuardConf, gconf: GeomConf) -> None:
    """The guard's whitening is the width's (C8).

    Raises:
        ValueError: ``guard.k`` is not ``gconf.k``.
    """
    if guard.k != gconf.k:
        raise ValueError(f"the guard's k {guard.k} is not the width's {gconf.k}")


def guard_gen() -> GenConf:
    """G's settings with the plan's batch of 64 (``posthoc_gen``'s runs)."""
    return replace(GenConf(), fit_batch=64)


# -- the anchors and the decoder --------------------------------------------------------


def guard_anchors(ws: WidthSets, lag: int, count: int, reach: int) -> Tensor:
    """The anchors ``(A,)`` of ``lag`` (module docstring): the ``in``
    centres whose stacks $t - \\text{reach}\\,n .. t + (\\text{reach} +
    1)\\,n$ lie in one block, ``count`` evenly spread."""
    _, u = ws.centres(lag)
    lo, hi = u - reach * lag, u + (reach + 1) * lag
    ok = (lo >= 0) & (hi < ws.n)
    u, lo, hi = u[ok], lo[ok], hi[ok]
    u = u[ws.frame_block[lo] == ws.frame_block[hi + ws.span]]
    if len(u) <= count:
        return u
    return u[torch.arange(count) * (len(u) - 1) // (count - 1)]


def fit_decoder(
    c_set: Tensor, frames: Tensor, gen: GenConf, j: int, dev: torch.device
) -> tuple[nn.Module, Tensor]:
    """G's decoder $j$ (its seeds: ``gen_eval.GenEvaluator._job``'s) learned
    on the conditions ``c_set`` ``(N, D)`` and the last frames u8 ``frames``
    ``(N, 3, S, S)``. Returns the net (eval) and the losses."""
    seed = DEC_SEED + DEC_SEED_STEP * j
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        net = FlowUNet(c_set.shape[1], **gen.net)
    net = net.to(dev)
    losses = fit_flow(net, c_set.to(dev), frames, steps=gen.fit_steps, batch=gen.fit_batch,
                      lr=gen.fit_lr, warmup=gen.warmup,
                      gen=torch.Generator().manual_seed(seed))  # fmt: skip
    return net, losses


def save_decoder(
    path: Path, net: nn.Module, std: Standardizer | None, gen: GenConf, code_dim: int
) -> None:
    """``gen_eval``'s format: the net, its keywords, the code's dimension and
    the standardizer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "net": {k: v.cpu() for k, v in net.state_dict().items()},
            "net_kw": dict(gen.net),
            "code_dim": code_dim,
            "mean": None if std is None else std.mean.cpu(),
            "sd": None if std is None else std.sd,
        },
        path,
    )


def load_decoder(
    path: Path, dev: torch.device
) -> tuple[nn.Module, Tensor | None, float | None]:
    """A decoder of ``gen_eval``'s format (``dec-*.pt``) on ``dev``, its
    standardizer's mean and sd (None of ``uncond``)."""
    d = torch.load(path, map_location="cpu", weights_only=False)
    net = FlowUNet(int(d["code_dim"]), **d["net_kw"])
    net.load_state_dict(d["net"])
    return net.to(dev).eval(), d["mean"], d["sd"]


# -- the codes -----------------------------------------------------------------------------


class WhiteCode:
    """A place of a code whitened by its reference (``Whitener``) as a
    ``stream_eval.Code``: ``mu`` the whitened coordinates (float32), of the
    place ``place`` of ``places`` and the optional linear ``pre`` (arm M's
    ``zpca``) before the whitening."""

    def __init__(
        self,
        places: Places,
        place: str,
        whitener: Whitener,
        pre: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        self._places, self._place, self._wh, self._pre = places, place, whitener, pre

    def modules(self) -> Mapping[str, nn.Module]:
        return {}

    def raw(self, x: Tensor) -> Tensor:
        """The place (after ``pre``) of the stacks ``x``."""
        z = self._places(x)[self._place]
        return z if self._pre is None else self._pre(z)

    def white(self, raw: Tensor) -> Tensor:
        """The whitened ``raw`` ``(n, k)`` float64."""
        return self._wh.white(raw.cpu())

    @torch.no_grad()
    def mu(self, x: Tensor) -> Tensor:
        return self._wh.white(self.raw(x).cpu()).float().to(x.device)


def cheat_codes(
    w: Tensor, s: Tensor, conf: GuardConf
) -> dict[str, Callable[[Tensor], Tensor]]:
    r"""The cheats' maps of whitened coordinates ``(..., L, k)`` (time on
    dim -2) by name: ``base`` the identity, ``mean-<w>`` the past window's
    mean, ``drop-<r>`` the ``r`` directions of the largest $s_i$ (``s``
    ``(k,)``) set to 0. ``w`` only names the dimension."""
    del w
    order = torch.argsort(s, descending=True)
    out: dict[str, Callable[[Tensor], Tensor]] = {"base": lambda x: x}
    for width in conf.cheat_windows:
        out[f"mean-{width}"] = lambda x, width=width: _past_mean(x, width)
    for r in conf.cheat_drops:
        keep = torch.ones(len(s), dtype=torch.float64)
        keep[order[:r]] = 0
        out[f"drop-{r}"] = lambda x, keep=keep: x * keep.to(x.dtype)
    return out


def _past_mean(x: Tensor, width: int) -> Tensor:
    """The mean over $[u - w, u]$ (fewer at the start) along dim -2."""
    length = x.shape[-2]
    zero = torch.zeros_like(x[..., :1, :])
    cs = torch.cat([zero, x.cumsum(-2)], -2)  # (..., L + 1, k)
    u = torch.arange(length)
    lo = (u - width).clamp(min=0)
    count = (u + 1 - lo).to(x.dtype)[:, None]
    return (cs[..., u + 1, :] - cs[..., lo, :]) / count


def lag_speeds(w: Tensor, lag: int) -> Tensor:
    r"""$s_i = \overline{(w_{u+n,i} - w_{u,i})^2} / (2 \mathrm{Var}_i)$
    ``(k,)`` of runs ``(R, L, k)`` at ``lag`` (pairs within a run)."""
    d = (w[:, lag:] - w[:, :-lag]).square().mean((0, 1))
    var = w.reshape(-1, w.shape[-1]).var(0, unbiased=False)
    return d / (2 * var)


def train_intervals(
    conf: Any,
    root: Path,
    conf_g: GuardConf,
    code: Callable[[Tensor], Tensor],
    dev: torch.device,
) -> tuple[Tensor, Tensor]:
    """The cheats' training set (module docstring): the codes ``code`` of
    every stack of the runs ``(R, L, k)`` float64 and the last frames u8 of
    the runs' middle halves ``(R L / 2, 3, S, S)``, the runs drawn uniformly
    among those of consecutive stacks (one frame apart) of one video."""
    t = conf.data.tours
    stream = StackedTourFrames(root, n=1, held_out=t.held_out, frames=t.frames,
                               stride=t.stride, advance=1)  # fmt: skip
    total = sum(len(U8Reader(root / v)) for v in videos(root) if v != t.held_out)
    gen = torch.Generator().manual_seed(conf_g.cheat_seed)
    length, a = conf_g.cheat_length, conf_g.cheat_length // 4
    codes, last = [], []
    while len(codes) < conf_g.cheat_count:
        k = int(torch.randint(total, (1,), generator=gen))
        stream.load_state_dict({"k": k})
        stacks, ok = [], True
        for i in range(length):
            x, ready, reset = stream.next()
            if not bool(ready[0]) or (i > 0 and bool(reset[0])):
                ok = False
                break
            stacks.append(x[0])
        if not ok:
            continue
        run = torch.stack(stacks)  # (L, n, 3, S, S)
        with torch.no_grad():
            z = torch.cat([code(c.to(dev)).double().cpu() for c in run.split(64)])
        codes.append(z)
        last.append(run[a : length - a, -1])
    return torch.stack(codes), torch.cat(last)


# -- the guard ----------------------------------------------------------------------------


def _identify(pred: Tensor, cand: Tensor, center: Tensor) -> tuple[Tensor, Tensor]:
    """The hits ``(A,)`` (the share of $K$) of the features ``pred`` ``(A, K,
    F)`` among each anchor's candidates ``cand`` ``(A, C, F)`` (the true one
    in the middle) and the picked places ``(A, K)``."""
    a, c, f = cand.shape
    real = cand.reshape(a * c, f)
    truth = torch.arange(a) * c + c // 2
    among = torch.arange(a)[:, None] * c + torch.arange(c)
    got = identification(pred, real, center=center, truth=truth, among=among)
    p = pred.double() - center.double()
    r = cand.double() - center.double()
    p = p / torch.linalg.vector_norm(p, dim=-1, keepdim=True).clamp(min=1e-300)
    r = r / torch.linalg.vector_norm(r, dim=-1, keepdim=True).clamp(min=1e-300)
    picks = torch.einsum("akf,acf->akc", p, r).argmax(2)
    return torch.tensor(got["hits"], dtype=torch.float64), picks


def _pick_max(picks: Tensor, c: int) -> float:
    return float(torch.bincount(picks.flatten(), minlength=c).max()) / picks.numel()


def diff_se(a: Tensor, b: Tensor, blocks: Tensor, gconf: GeomConf) -> float:
    r"""$\mathrm{SE}^*$ of the mean of the paired differences ``a - b``
    ``(A,)`` of anchors of the ``blocks`` ``(A,)``: the larger of the block
    bootstraps of the blocks and of their adjacent pairs (``geom_eval``'s
    draws, ``boot`` of ``boot_seed``)."""
    nb = int(blocks.max()) + 1
    d = (a - b).double()
    num = torch.zeros(nb, dtype=torch.float64).index_add_(0, blocks, d)
    den = torch.zeros(nb, dtype=torch.float64).index_add_(0, blocks, torch.ones_like(d))
    sds = []
    for n_, d_ in ((num, den), (_pair(num), _pair(den))):
        r = ratio_resamples(n_[None], d_[None], counts_of(len(n_), gconf))[:, 0]
        r = r[torch.isfinite(r)]
        sds.append(float(r.std()) if len(r) >= 2 else math.nan)
    return math.nan if any(math.isnan(x) for x in sds) else max(sds)


def _pair(b: Tensor) -> Tensor:
    n = len(b)
    to = torch.arange(n).div(2, rounding_mode="floor").clamp(max=max(n // 2 - 1, 0))
    return torch.zeros(max(n // 2, 1), dtype=b.dtype).index_add_(0, to, b)


@dataclass
class _Lag:
    lag: int
    t: Tensor  # (A,) anchors
    sl: dict[str, Tensor]
    ext: Tensor  # (A, C, F) candidates' features
    mid: Tensor
    ext_local: Tensor | None
    mid_local: Tensor | None
    noise: Tensor  # (A, K, 3, S, S)
    blocks: Tensor  # (A,)
    other: Tensor  # (A,) another anchor of the same block (itself if alone)


class Guard:
    """The guard's data (module docstring) of Zurich's ``frames`` u8 ``(F, 3,
    S, S)`` and the sets ``ws``, with ``features`` (u8 frames to features)
    and optionally ``local``; ``ref_count`` the frames of $R$."""

    def __init__(
        self,
        frames: Tensor,
        ws: WidthSets,
        *,
        features: Features,
        local: Features | None,
        conf: GuardConf,
        gen: GenConf,
        dev: torch.device,
    ) -> None:
        self._frames, self._ws, self.conf, self._gen, self._dev = (
            frames,
            ws,
            conf,
            gen,
            dev,
        )
        self._features, self._local = features, local
        r = spread(gen.ref_count, len(frames), ws.span)
        self.center = self._feats(frames[r]).mean(0)
        self.center_local = (
            None if local is None else self._feats(frames[r], local).mean(0)
        )
        noise_gen = torch.Generator().manual_seed(conf.noise_seed)
        shuffle_gen = torch.Generator().manual_seed(conf.shuffle_seed)
        shape = frames.shape[1:]
        self.lags: list[_Lag] = []
        for lag in conf.lags:
            t = guard_anchors(ws, lag, conf.anchors, conf.reach)
            sl = slots(t, lag, conf.reach)
            ext, mid = self._last(sl["ext"]), self._last(sl["mid"])
            noise = torch.randn(len(t), conf.samples, *shape, generator=noise_gen)
            blocks = ws.blocks[t]
            other = torch.arange(len(t))
            for b in blocks.unique().tolist():
                idx = (blocks == b).nonzero().flatten()
                if len(idx) > 1:
                    perm = torch.randperm(len(idx) - 1, generator=shuffle_gen) + 1
                    other[idx] = idx[(torch.arange(len(idx)) + perm[0]) % len(idx)]
            self.lags.append(_Lag(
                lag, t, sl, self._feats(ext), self._feats(mid),
                None if local is None else self._feats(ext, local),
                None if local is None else self._feats(mid, local),
                noise, blocks, other,
            ))  # fmt: skip

    def _last(self, stacks: Tensor) -> Tensor:
        """The last frames u8 ``(..., 3, S, S)`` of the stacks ``stacks``."""
        return self._frames[stacks + self._ws.span]

    def _feats(self, frames: Tensor, fn: Features | None = None) -> Tensor:
        """The features (float64, CPU) of u8 frames ``(..., 3, S, S)``."""
        fn = self._features if fn is None else fn
        flat = frames.reshape(-1, *frames.shape[-3:])
        out = torch.cat([fn(c.to(self._dev)).double().cpu() for c in flat.split(256)])
        return out.reshape(*frames.shape[:-3], -1)

    @torch.no_grad()
    def _draw(self, net: nn.Module, cond: Tensor, noise: Tensor) -> Tensor:
        """Images u8 ``(A, K, 3, S, S)`` of the conditions ``(A, D)``."""
        a, k = noise.shape[:2]
        x0 = noise.flatten(0, 1)
        c = cond.float().repeat_interleave(k, 0)
        out = []
        for i in range(0, len(x0), self._gen.sample_batch):
            s = slice(i, i + self._gen.sample_batch)
            x = euler_sample(net, x0[s].to(self._dev), c[s].to(self._dev),
                             self._gen.sample_steps)  # fmt: skip
            out.append(((x.clamp(-1, 1) + 1) / 2 * 255).round().to(torch.uint8).cpu())
        return torch.cat(out).reshape(a, k, *noise.shape[2:])

    def _read(self, lg: _Lag, drawn: Tensor, target: str) -> dict[str, Any]:
        """The hits of ``drawn`` among the candidates of ``target`` (``ext``
        of ``slots(..)["ext"]``, ``mid`` of ``["mid"]``; the true one the
        middle, the place ``width_eval.TARGETS[target]``)."""
        cand = lg.ext if target == "ext" else lg.mid
        hits, picks = _identify(self._feats(drawn), cand, self.center)
        out: dict[str, Any] = {"hits": hits.tolist(), "share": float(hits.mean()),
                               "pick_max": _pick_max(picks, cand.shape[1])}  # fmt: skip
        if self._local is not None:
            loc = lg.ext_local if target == "ext" else lg.mid_local
            assert loc is not None and self.center_local is not None
            lh, _ = _identify(self._feats(drawn, self._local), loc, self.center_local)
            out["local_hits"] = lh.tolist()
            out["local_share"] = float(lh.mean())
        return out

    def code_rows(
        self,
        net: nn.Module,
        cond: Tensor,
        train: Tensor | None,
        labels: Mapping[str, Any],
        picture: Path | None = None,
    ) -> list[Row]:
        r"""The rows of each lag (module docstring) of the decoder ``net`` and
        the conditions ``cond`` ``(n, D)`` of every stack of Zurich (already
        standardized); ``train`` ``(N, D)`` the training conditions (the
        nearest code of ``pos``; None: no ``pos``); with ``picture`` a PNG of
        each lag ``<picture>-<lag>.png``."""
        rows = []
        gen_pos = torch.Generator().manual_seed(self.conf.pos_seed)
        for lg in self.lags:
            conds = conditions(cond.double(), lg.sl)
            fut = conds["future"]
            conds["shuffle"] = fut[lg.other]
            nn_dist: dict[str, float] = {}
            if train is not None:
                d_ext = _nearest(conds["ext"], train.double(), self._dev)
                d_fut = _nearest(fut, train.double(), self._dev)
                u = torch.randn(fut.shape, generator=gen_pos, dtype=torch.float64)
                u = u / torch.linalg.vector_norm(u, dim=1, keepdim=True)
                conds["pos"] = fut + d_ext[:, None] * u
                d_pos = _nearest(conds["pos"], train.double(), self._dev)
                nn_dist = {"ext": float(d_ext.mean()), "future": float(d_fut.mean()),
                           "pos": float(d_pos.mean())}  # fmt: skip
            drawn = {name: self._draw(net, v, lg.noise) for name, v in conds.items()}
            read: dict[str, Any] = {"ext": {}, "mid": {}}
            for name in EXT_CONDS:
                if name in drawn:
                    read["ext"][name] = self._read(lg, drawn[name], "ext")
            for name in MID_CONDS:
                read["mid"][name] = self._read(lg, drawn[name], "mid")
            rows.append(dict(labels) | {
                "part": "guard", "lag": lg.lag, "n_anchors": len(lg.t),
                "anchors": lg.t.tolist(), "blocks": lg.blocks.tolist(),
                "read": read, "nn": nn_dist,
                "n_alone": int((lg.other == torch.arange(len(lg.t))).sum()),
            })  # fmt: skip
            if picture is not None:
                self._picture(Path(f"{picture}-{lg.lag}.png"), lg, drawn)
        return rows

    def control_rows(self) -> list[Row]:
        """The controls of no decoder (module docstring) of each lag."""
        rows = []
        for lg in self.lags:
            past, now = self._last(lg.sl["past"]), self._last(lg.sl["now"])
            fut = self._last(lg.sl["future"])
            fp, fn_, ff = self._feats(past), self._feats(now), self._feats(fut)
            pix_mean = ((past.float() + fut.float()) / 2).round().to(torch.uint8)
            pix_ext = (
                (2 * now.float() - past.float()).clamp(0, 255).round().to(torch.uint8)
            )
            nxt = self._last((lg.sl["future"] + 1).clamp(max=self._ws.n - 1))
            preds = {
                ("mid", "pixel_mean"): self._feats(pix_mean),
                ("mid", "feature_mean"): (fp + ff) / 2,
                ("ext", "feature_ext"): 2 * fn_ - fp,
                ("ext", "pixel_ext"): self._feats(pix_ext),
                ("ext", "real_next"): self._feats(nxt),
            }
            read: dict[str, Any] = {"ext": {}, "mid": {}}
            for (target, name), f in preds.items():
                cand = lg.ext if target == "ext" else lg.mid
                hits, picks = _identify(f[:, None], cand, self.center)
                read[target][name] = {"hits": hits.tolist(), "share": float(hits.mean()),
                                      "pick_max": _pick_max(picks, cand.shape[1])}  # fmt: skip
            rows.append({"part": "control", "lag": lg.lag, "n_anchors": len(lg.t),
                         "blocks": lg.blocks.tolist(), "read": read})  # fmt: skip
        return rows

    def _picture(self, path: Path, lg: _Lag, drawn: Mapping[str, Tensor]) -> None:
        """Of the anchors 0, ``picture_every``, ...: the real $x_{t-\\Delta}$,
        $x_t$, $x_{t+\\Delta}$, $D(z_t)$, $D(z_{t+\\Delta})$, the
        extrapolation's and the midpoint's drawings and the pixels' mean."""
        idx = torch.arange(0, len(lg.t), self.conf.picture_every)
        past, now = self._last(lg.sl["past"][idx]), self._last(lg.sl["now"][idx])
        fut = self._last(lg.sl["future"][idx])
        cols = [past, now, fut, drawn["now"][idx, 0], drawn["future"][idx, 0],
                drawn["ext"][idx, 0], drawn["mid"][idx, 0],
                ((past.float() + fut.float()) / 2).round().to(torch.uint8)]  # fmt: skip
        grid = torch.stack(cols, 1).float() / 255
        path.parent.mkdir(parents=True, exist_ok=True)
        write_png(path, tile(grid, self.conf.picture_scale))


@torch.no_grad()
def _nearest(x: Tensor, ref: Tensor, dev: torch.device) -> Tensor:
    """The distance ``(A,)`` of each row of ``x`` to its nearest row of
    ``ref``."""
    xd = x.to(dev)
    best = torch.full((len(x),), math.inf, dtype=torch.float64, device=dev)
    for c in ref.split(8192):
        best = torch.minimum(best, torch.cdist(xd, c.to(dev)).min(1).values)
    return best.cpu()


def winning_lags(rows: Sequence[Row], lags: Sequence[int], **labels: Any) -> list[int]:
    r"""The guard's lags where the point $p_{1/4} \ge 1/2$ (the ``common``
    rows of ``labels``, which name one whitening: ``place``, ``k``, ``eps``,
    ``ref``): the set $L$."""
    ls = lin_sums(rows, **labels)
    got = []
    for lag in lags:
        i = int((ls.lags == lag).nonzero()[0])
        if float(ls.win["25"][i].sum() / ls.n[i].sum()) >= 0.5:
            got.append(lag)
    return got


# -- the jobs ------------------------------------------------------------------------------


def _decoder_for(
    path: Path,
    code: WhiteCode | None,
    conf: Any,
    root: Path,
    gen: GenConf,
    j: int,
    dev: torch.device,
    data: tuple[Tensor, Tensor] | None = None,
) -> tuple[nn.Module, Standardizer, Tensor]:
    """The decoder $j$ at ``path`` (learned and saved there if absent) of
    the code ``code`` (a ``Code``) on G's training set, or on ``data``
    (conditions, frames); its standardizer and standardized training
    conditions."""
    if data is None:
        assert code is not None
        z, frames = training_set(conf, root, {"c": code}, gen.set_size)
        c_raw = z["c"].double().cpu()
    else:
        c_raw, frames = data
    std = Standardizer(c_raw)
    c_set = std(c_raw)
    if path.exists():
        net, _, _ = load_decoder(path, dev)
    else:
        net, _ = fit_decoder(c_set, frames, gen, j, dev)
        save_decoder(path, net, std, gen, c_set.shape[1])
    return net, std, c_set


def _features_of(
    features: Features | None,
    local: Features | None,
    inception: Path | None,
    device: str,
    cells: int,
) -> tuple[Features, Features | None, dict[str, Any]]:
    """The given features, or FID's Inception (pool3 and ``FIDInception.
    local``) of the weights ``inception`` and its record."""
    if features is not None:
        return features, local, {}
    path = weights_path() if inception is None else inception
    net = load_fid_inception(path, device)
    record = {"path": str(path), "sha256": sha256(path)}
    return net, lambda x: net.local(x, cells), record


def evaluate_yard(
    conf: Any,
    sets_dir: Path,
    out: Path,
    *,
    fit: Mapping[str, Sequence[int]],
    v17: Path | None = None,
    dec: Path | None = None,
    root: Path | None = None,
    features: Features | None = None,
    local: Features | None = None,
    inception: Path | None = None,
    guard: GuardConf = GuardConf(),
    gen: GenConf | None = None,
    gconf: GeomConf = GeomConf(),
    wconf: WidthConf = WidthConf(),
    fconf: FillConf = FillConf(),
    spacing: Mapping[str, Any] | None = None,
    span: float | None = None,
) -> Path:
    """V-h3 into ``out``: the controls (``controls.json``), the rows of the
    decoders of ``fit`` (``pca`` and ``v17``, each its seeds; PCA64 of its
    own place, learned into ``ckpt/`` if absent) ``<code>/rows-d<j>.json``
    and pictures, and of ``uncond`` the existing decoder of G's yardsticks
    in ``dec`` (``ckpt/dec-yardstick-uncond-d0.pt``; the chance's row).
    Returns ``out``.

    Raises:
        ValueError: A code of ``fit`` is not ``pca`` or ``v17``, ``v17``
            is fitted without its checkpoint, or the guard's ``k`` is not
            the width's (``same_k``).
    """
    unknown = [c for c in fit if c not in ("pca", "v17")]
    if unknown or ("v17" in fit and v17 is None):
        raise ValueError(f"fit pca or v17 (with its checkpoint), not {list(fit)}")
    same_k(guard, gconf)
    gen = guard_gen() if gen is None else gen
    determinism = configure_determinism(conf.device)
    dev = torch.device(conf.device)
    ws, zurich, ref, root = sets_and_ref(conf, sets_dir, root, gconf, wconf, fconf,
                                         spacing, span)  # fmt: skip
    feats, loc, record = _features_of(features, local, inception, conf.device,
                                      guard.cells)  # fmt: skip
    _start(out, {"guard": asdict(guard), "gen": asdict(gen) | {"net": dict(gen.net)},
                 "key": ws.key, "inception": record.get("sha256")}, determinism)  # fmt: skip
    g = Guard(
        zurich.frames, ws, features=feats, local=loc, conf=guard, gen=gen, dev=dev
    )
    _write(out / "controls.json", g.control_rows(), {"key": ws.key})
    for name, seeds in fit.items():
        code = _yard_code(name, conf, ref, v17, guard, gconf, dev)
        ev = _encode(code, zurich, ws.n, dev)
        for j in seeds:
            path = out / "ckpt" / f"dec-width-{name}-d{j}.pt"
            net, std, c_set = _decoder_for(path, code, conf, root, gen, j, dev)
            rows = g.code_rows(net, std(ev).double(), c_set, {"code": name, "dec_seed": j,
                               "cond": "pca64"}, out / name / f"d{j}")  # fmt: skip
            _write(out / name / f"rows-d{j}.json", rows, {"key": ws.key})
    if dec is not None:
        path = dec / "ckpt" / "dec-yardstick-uncond-d0.pt"
        net, _, _ = load_decoder(path, dev)
        dim = int(torch.load(path, map_location="cpu", weights_only=False)["code_dim"])
        zeros = torch.zeros(ws.n, dim, dtype=torch.float64)
        rows = g.code_rows(net, zeros, None, {"code": "uncond", "dec_seed": 0,
                           "cond": "none", "source": str(path)})  # fmt: skip
        _write(out / "uncond" / "rows-d0.json", rows, {"key": ws.key})
    return out


@torch.no_grad()
def _encode(code: WhiteCode, zurich: Any, n: int, dev: torch.device) -> Tensor:
    """The whitened code ``(n, k)`` float64 of Zurich's ``n`` stacks."""
    raw = encode_places(lambda x: {"z": code.raw(x)}, zurich.stacks, n, dev)["z"]
    return code.white(raw)


def _yard_code(
    name: str,
    conf: Any,
    ref: Tensor,
    v17: Path | None,
    guard: GuardConf,
    gconf: GeomConf,
    dev: torch.device,
) -> WhiteCode:
    """``pca`` (``geom_eval``'s yardstick) or ``v17``'s $\\mu$, whitened by
    the reference."""
    if name == "pca":
        gen = torch.Generator().manual_seed(conf.seed + PCA_SEED)
        pc = PCACode.fit(pixels, ref, gconf.pca_dim, gen=gen, device=conf.device)
        places: Places = _feature_code(pc.mu)
    else:
        assert v17 is not None
        vae = vae_at(v17, conf.device)
        assert isinstance(vae, VAEModules)
        places = VaeTokenCode(vae).places
    z_ref = encode_places(places, lambda idx: ref[idx], len(ref), dev)["own"]
    return WhiteCode(places, "own", Whitener.fit(z_ref, guard.k, c_eps=gconf.c_eps))


def evaluate_cheat(
    conf: Any,
    sets_dir: Path,
    out: Path,
    *,
    root: Path | None = None,
    features: Features | None = None,
    local: Features | None = None,
    inception: Path | None = None,
    guard: GuardConf = GuardConf(),
    gen: GenConf | None = None,
    gconf: GeomConf = GeomConf(),
    wconf: WidthConf = WidthConf(),
    fconf: FillConf = FillConf(),
    spacing: Mapping[str, Any] | None = None,
    span: float | None = None,
    only: Collection[str] | None = None,
) -> Path:
    """V-h2 into ``out``: of each cheat of ``pca`` (``cheat_codes``) its
    width's rows on Zurich ``<name>/width.json`` (the whitened coordinates
    as they are: no whitening anew) and its decoder's guard rows
    ``<name>/rows-d0.json``, the decoders learned on the training intervals;
    the speeds $s_i$ ``speeds.json``; with ``only`` those cheats alone (each
    one's files the same to the byte: its decoder's seed and the guard's
    noise are of no other cheat). Returns ``out``.

    Raises:
        ValueError: A name of ``only`` is no cheat.
    """
    same_k(guard, gconf)
    names = list(cheat_codes(torch.zeros(0), torch.zeros(guard.k), guard))
    bad = [n for n in only or () if n not in names]
    if bad:
        raise ValueError(f"no cheats {bad} (of {names})")
    gen = guard_gen() if gen is None else gen
    determinism = configure_determinism(conf.device)
    dev = torch.device(conf.device)
    ws, zurich, ref, root = sets_and_ref(conf, sets_dir, root, gconf, wconf, fconf,
                                         spacing, span)  # fmt: skip
    feats, loc, record = _features_of(features, local, inception, conf.device,
                                      guard.cells)  # fmt: skip
    _start(out, {"guard": asdict(guard), "gen": asdict(gen) | {"net": dict(gen.net)},
                 "key": ws.key, "inception": record.get("sha256")}, determinism)  # fmt: skip
    code = _yard_code("pca", conf, ref, None, guard, gconf, dev)
    w_eval = _encode(code, zurich, ws.n, dev)
    w_train, frames = train_intervals(conf, root, guard, code.mu, dev)
    s = lag_speeds(w_train, guard.cheat_lag)
    _write(out / "speeds.json", [{"part": "speeds", "s": s.tolist()}], {"key": ws.key})
    a = guard.cheat_length // 4
    g = Guard(
        zurich.frames, ws, features=feats, local=loc, conf=guard, gen=gen, dev=dev
    )
    for name, fn in cheat_codes(w_eval, s, guard).items():
        if only is not None and name not in only:
            continue
        x = fn(w_eval)
        rows = [{"code": name, "place": "own", "k": guard.k, "ref": "cheat", "key": ws.key}
                | r for r in white_rows(x, ws, wconf, full=False)]  # fmt: skip
        _write(out / name / "width.json", rows, {"key": ws.key})
        c_train = fn(w_train)[:, a : guard.cheat_length - a].reshape(-1, guard.k)
        path = out / "ckpt" / f"dec-width-{name}-d0.pt"
        net, std, c_set = _decoder_for(path, None, conf, root, gen, 0, dev,
                                       data=(c_train, frames))  # fmt: skip
        guard_rows = g.code_rows(net, std(x).double(), c_set,
                                 {"code": name, "dec_seed": 0, "cond": "pca64"},
                                 out / name / "d0")  # fmt: skip
        _write(out / name / "rows-d0.json", guard_rows, {"key": ws.key})
    return out


def evaluate_run(
    run_dir: Path,
    *,
    steps: Sequence[int],
    sets_dir: Path,
    root: Path | None = None,
    device: str | None = None,
    features: Features | None = None,
    local: Features | None = None,
    inception: Path | None = None,
    guard: GuardConf = GuardConf(),
    gen: GenConf | None = None,
    gconf: GeomConf = GeomConf(),
    wconf: WidthConf = WidthConf(),
    fconf: FillConf = FillConf(),
    spacing: Mapping[str, Any] | None = None,
    span: float | None = None,
) -> Path:
    """The exploration's guard of a JEPA run's checkpoints of ``steps`` into
    ``run_dir / OUT``: of a step whose width (``posthoc_width``, its own
    place) wins at a guard lag, the decoder of its PCA64 (learned if absent)
    and its rows ``rows/<step>.json``; else a row of no lag. With G's
    decoder of the same map (``posthoc_gen/ckpt/dec-<step>-z-d0.pt``, of a
    run whose own latent is its $z$) its rows as a description
    (``cond`` ``full``). Returns the directory.

    Raises:
        ValueError: A step has no checkpoint or no width rows.
    """
    conf = _conf_of(run_dir, device)
    ckpts = run_ckpts(run_dir)
    missing = [s for s in steps if s not in ckpts]
    if missing:
        raise ValueError(f"no checkpoint of steps {missing} (of {list(ckpts)})")
    same_k(guard, gconf)
    gen = guard_gen() if gen is None else gen
    determinism = configure_determinism(conf.device)
    dev = torch.device(conf.device)
    ws, zurich, ref, root = sets_and_ref(conf, sets_dir, root, gconf, wconf, fconf,
                                         spacing, span)  # fmt: skip
    feats, loc, record = _features_of(features, local, inception, conf.device,
                                      guard.cells)  # fmt: skip
    out = run_dir / OUT
    _start(out, {"guard": asdict(guard), "gen": asdict(gen) | {"net": dict(gen.net)},
                 "key": ws.key, "inception": record.get("sha256")}, determinism)  # fmt: skip
    mods, own = _jepa_of(run_dir, conf.device)
    g: Guard | None = None
    for step in steps:
        path = out / "rows" / f"{step:06d}.json"
        if path.exists():
            continue
        width = run_dir / WIDTH_OUT / "rows" / f"{step:06d}.json"
        if not width.exists():
            raise ValueError(f"no width rows of step {step}: {width}")
        lags = winning_lags(json.loads(width.read_text()), guard.lags, place="own",
                            k=gconf.k, eps=gconf.c_eps, ref="all")  # fmt: skip
        stamp = {"run": run_dir.name, "step": step, "t": float(ckpts[step].stem)}
        if not lags:
            _write(
                path, [{"part": "no_lag", "code": run_dir.name, "step": step}], stamp
            )
            continue
        ckpt.load_eval(ckpts[step], mods.modules())
        places = JepaPlaces(mods).places
        z_ref = encode_places(places, lambda idx: ref[idx], len(ref), dev)
        pre = None
        z_ref_own = z_ref["own"]
        if own == "zpca":  # geom_eval._zpca's map, fit on the reference
            pre = _zpca_map(z_ref["own"], conf.seed, gconf.pca_dim)
            z_ref_own = pre(z_ref_own)
        code = WhiteCode(
            places, "own", Whitener.fit(z_ref_own, guard.k, c_eps=gconf.c_eps), pre
        )
        ev = _encode(code, zurich, ws.n, dev)
        if g is None:
            g = Guard(
                zurich.frames,
                ws,
                features=feats,
                local=loc,
                conf=guard,
                gen=gen,
                dev=dev,
            )
        dec_path = out / "ckpt" / f"dec-width-{step:06d}-d0.pt"
        net, std, c_set = _decoder_for(dec_path, code, conf, root, gen, 0, dev)
        rows = g.code_rows(net, std(ev).double(), c_set,
                           {"code": run_dir.name, "step": step, "dec_seed": 0,
                            "cond": "pca64", "lags_won": lags},
                           out / f"{step:06d}")  # fmt: skip
        old = run_dir / "posthoc_gen" / "ckpt" / f"dec-{step:06d}-z-d0.pt"
        if old.exists() and own == "z":
            net_f, mean, sd = load_decoder(old, dev)
            raw = encode_places(places, zurich.stacks, ws.n, dev)["own"].double()
            assert mean is not None and sd is not None
            full = (raw - mean.double()) / sd
            rows += g.code_rows(net_f, full, None, {"code": run_dir.name, "step": step,
                                "dec_seed": 0, "cond": "full", "source": str(old)})  # fmt: skip
        _write(path, rows, stamp)
    return out


def _zpca_map(z_ref: Tensor, seed: int, dim: int) -> Callable[[Tensor], Tensor]:
    """``geom_eval._zpca``'s map (fit on the reference) as a function."""
    g = torch.Generator().manual_seed(seed + PCA_SEED)
    mean, comps = pca.fit_pca(z_ref, dim, gen=g)
    return lambda z: pca.encode(z.cpu(), mean, comps)


def _fits(items: Sequence[str]) -> dict[str, tuple[int, ...]]:
    out = {}
    for item in items:
        name, _, seeds = item.partition(":")
        out[name] = tuple(int(s) for s in seeds.split(",") if s) or (0,)
    return out


def main(argv: Sequence[str] | None = None) -> Path:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = p.add_subparsers(dest="job", required=True)
    yard = sub.add_parser("yard")
    yard.add_argument("--fit", action="append", default=[], help="code:seeds")
    yard.add_argument("--v17", type=Path)
    yard.add_argument("--dec", type=Path, help="G's yardsticks' posthoc_gen (uncond)")
    cheat = sub.add_parser("cheat")
    cheat.add_argument("--only", type=lambda t: tuple(x for x in t.split(",") if x),
                       default=None, help="only these cheats, comma-separated")  # fmt: skip
    for q in (yard, cheat):
        q.add_argument("--config", type=Path, required=True)
        q.add_argument("--out", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("run_dir", type=Path)
    run.add_argument("--steps", type=lambda t: tuple(int(x) for x in t.split(",")),
                     required=True)  # fmt: skip
    for q in (yard, cheat, run):
        q.add_argument("--sets", type=Path, required=True)
        q.add_argument("--device", default=None)
        q.add_argument("--root", type=Path, default=None)
        q.add_argument("--inception", type=Path, default=None)
    a = p.parse_args(argv)
    if a.job == "run":
        return evaluate_run(a.run_dir, steps=a.steps, sets_dir=a.sets, root=a.root,
                            device=a.device, inception=a.inception)  # fmt: skip
    conf = _conf_of(a.config, a.device)
    if a.job == "yard":
        return evaluate_yard(conf, a.sets, a.out, fit=_fits(a.fit), v17=a.v17, dec=a.dec,
                             root=a.root, inception=a.inception)  # fmt: skip
    return evaluate_cheat(conf, a.sets, a.out, root=a.root, inception=a.inception,
                          only=a.only)  # fmt: skip


if __name__ == "__main__":
    main()
