r"""The guard of the linear horizon rebuilt on the observation's content
(``plans/jepa.md``, 「線形の幅のガードの作り直し（2026-11-26、V19 の値を見る
前）」; README, 「線形の幅のガードの作り直し（回す前に決めたこと、
2026-11-26）」): how much of the observation of a time its code of the same
time gives back, beside the width. The quantities, their intervals and the
front's verdict are ``exp.lib.fidelity``'s; the widths are
``width_eval``'s; the verdicts of the real data are
``results/v18c/judge_content.py``'s.

**The real data** (``evaluate_content``, V-h4b): of each code, the
whitened PCA64 the width reads (``pca``'s ``base`` and its cheats of
``width_guard.cheat_codes``, made anew from ``pca``'s codes; ``rot``:
``pca``'s times a random orthogonal matrix of ``rot_seed``; the
yardsticks ``color``, ``mean8``, the step 0 of the five JEPA runs and V16's
VAE, each whitened by the reference as the yard's width), the readouts of
``ContentConf.seeds`` learned on the cheats' training set
(``width_guard.interval_runs``: the runs' middle halves, videos other than
Zurich) from the code of each stack to its last frame averaged to
``obs`` x ``obs`` (``fidelity.pool_obs``), and on Zurich at the common
centres (``WidthSets.common``, the blocks of the width's triples) the
blocks' sums of $E$ and $E^d$ (``fidelity.content_sums``): rows ``content``
of ``readout`` ``mlp`` (the guard) and ``lin`` (least squares; a record),
of ``obs`` ``pix`` and, with ``local``, ``local`` (Inception's 2 x 2 local
features of the last frame; a record), and ``obs_ext`` (a record: at the
lags ``ext_lags`` the errors of $g(c_t)$ (copy), $g(2c_t - c_{t-\Delta})$
(the extrapolation) and $g(c_{t+\Delta})$ (the ceiling) on $y_{t+\Delta}$
and their denominator) into ``<out>/<code>/content.json``.

**The synthetic checks** (``check_rows``, V-h4a): the realizations of the
plan's processes (``positive``, ``negative``; ``SynthConf``) on V-h0's sets
(``width_eval.synthetic_sets``; the codes of unit variance, so the
population's whitening is the identity), each code's readouts (``SynthConf.
mlp``, the seeds of ``readout_seeds``) learned on a training realization of
the same seed's generator (``train_n`` times past a warm-up of the largest
window), and the front's verdict (``front_of``) of the plan's pairs:

- V-h4a1: ``slow`` against ``fast`` (both of one phase): on the long
  realization ``kept``; of each realization recorded
- V-h4a2: each cheat of ``mixed`` (``drop`` its OU set to 0, ``mean-<w>``
  its past mean) against it: on the long realization at least one cheat's
  width clears $+\Delta_*$ wholly, and every such one is ``dropped``; of the
  realizations, those where a cheat is ``kept`` at most ``max_false``
- V-h4a3: ``noisy`` (``mixed`` with Gaussian noise of ``calib_noise`` on
  its OU coordinates, $\psi = -\psi_*$ exactly in the population) against
  ``mixed``: the realizations where it is comparable and $\psi$'s lower end
  clears $-\psi_*$ at most ``max_false`` (the plan's 「下端 $> -\psi_*$」, of
  $\psi$ alone: $\psi^d$ of this noise lies far lower, so the both's count
  would test nothing; recorded beside); the power (recorded): ``mixed`` of
  each realization against ``mixed`` of the V-h4a2 realization of the same
  index (independent, $\psi = 0$)

$\sigma_{\mathrm{seed}}$ of each process: of the long realization, the
per-seed $\psi_s$ of the code rotated (``slow-rot``, ``mixed-rot``) against
the code. The long units and the realizations run apart
(``check_parts``) and merge to the rows of one run (``merge_checks``)::

    python -m exp.vision.content_eval checks --out runs/v18c-width/content/checks
    ... checks --out <part> --only V-h4a1-long | --only V-h4a1,V-h4a2,V-h4a3 --seeds 2101:2111
    ... merge-checks --out runs/v18c-width/content/checks <part> <part> ..
    python -m exp.vision.content_eval real --config <v18c run> --sets runs/v18c-tlin \
        --out runs/v18c-width/content --codes base,rot,mean-8 [--local] \
        [--init <..> --init-f <..> --init-m <..> --init-c <..> --init-c1 <..> --init-v16 <..>]
"""

import argparse
import json
import math
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from exp.lib.fidelity import (
    ContentSums,
    MLPConf,
    comparable,
    content_sums,
    fit_linear,
    fit_mlp,
    front_verdict,
    pool_obs,
    predict,
    predict_linear,
    ratio_interval,
    seed_sigma,
)
from exp.lib.midpoint import Whitener
from exp.lib.provenance import configure_determinism
from exp.lib.synth_series import ou
from exp.vision.fill_eval import FillConf
from exp.vision.geom_eval import (  # the geom job's own helpers, reused as they are
    GeomConf,
    JepaPlaces,
    VaeTokenCode,
    _conf_of,
    _feature_code,
    _first_vae,
    _init_of,
    _mean8,
    _start,
    _write,
    encode_places,
)
from exp.vision.jepa_eval import PCA_SEED, ColorCode, PCACode, vae_at
from exp.vision.models import VAEModules
from exp.vision.results.v18c.judge_width import pair_ratio
from exp.vision.width_eval import (
    YARD_INITS,
    LinSums,
    WidthConf,
    WidthSets,
    _cat,
    _common_sums,
    counts_of,
    point,
    sets_and_ref,
    synthetic_sets,
)
from exp.vision.width_guard import (
    GuardConf,
    WhiteCode,
    _encode,
    _features_of,
    _past_mean,
    _yard_code,
    _zpca_map,
    cheat_codes,
    interval_runs,
    lag_speeds,
    same_k,
)

__all__ = [
    "LONG_UNITS",
    "SEEDED",
    "YARD",
    "ContentConf",
    "SynthConf",
    "calib_noise",
    "check_parts",
    "check_rows",
    "content_rows",
    "evaluate_checks",
    "evaluate_content",
    "front_of",
    "interval_codes",
    "main",
    "merge_checks",
    "merge_checks_into",
    "negative",
    "orthogonal",
    "positive",
]

type Row = dict[str, Any]
type Places = Callable[[Tensor], dict[str, Tensor]]

_F64 = torch.float64
YARD = ("color", "mean8", *YARD_INITS, "init_v16")  # the yardsticks beside pca (base)


@dataclass(frozen=True)
class ContentConf:
    """The real data's settings (``plans/jepa.md``), by default the plan's."""

    obs: int = 32
    seeds: tuple[int, ...] = (3001, 3002, 3003)
    mlp: MLPConf = MLPConf()
    rot_seed: int = 3101
    ext_lags: tuple[int, ...] = (8, 16, 32, 64)
    cells: int = 2  # the local features' cells a side


def orthogonal(d: int, seed: int) -> Tensor:
    """A random orthogonal ``(d, d)`` float64 (Haar: the QR of a Gaussian,
    the signs of R's diagonal moved into Q) of ``seed``."""
    g = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=_F64))
    return q * torch.sign(torch.diagonal(r))


def front_of(
    width: tuple[float | None, float | None],
    a: ContentSums,
    b: ContentSums,
    counts: Tensor,
    counts2: Tensor,
    sigmas: tuple[float, float],
    delta_star: float,
    psi_star: float,
) -> dict[str, Any]:
    r"""The front's verdict of $a$ (claimed wide) against $b$: the width's
    ratio interval ``width``, $\psi$ and $\psi^d$ (``ratio_interval`` of
    $\sigma_{\mathrm{seed}}$ ``sigmas`` of $E$ and of $E^d$), whether $b$'s
    $E$ and $E^d$ are comparable, and ``fidelity.front_verdict``."""
    psi = ratio_interval(a, b, counts, counts2, sigmas[0])
    psi_d = ratio_interval(a, b, counts, counts2, sigmas[1], change=True)
    comp = [comparable(b, counts, counts2), comparable(b, counts, counts2, change=True)]
    ok = all(bool(c["ok"]) for c in comp)
    return {"verdict": front_verdict(width, psi, psi_d, ok, delta_star, psi_star),
            "width": list(width), "psi": psi, "psi_d": psi_d, "comparable": comp,
            "E": [a.value(), b.value()], "Ed": [a.value(True), b.value(True)]}  # fmt: skip


# -- the readouts of a code ----------------------------------------------------------------


def _anchors(ws: WidthSets) -> tuple[Tensor, Tensor]:
    """The common centres and their blocks (the width's triples')."""
    t = ws.common
    return t, ws.blocks[t - ws.max_lag]


def _obs_ext(
    net: Any, x: Tensor, y: Tensor, t: Tensor, mean: Tensor, lags: Sequence[int],
    dev: torch.device, lin: Tensor | None = None,
) -> dict[str, dict[str, float]]:  # fmt: skip
    """The record of the extrapolation in the observation (module
    docstring) at each lag: the sums of the errors of ``copy``, ``ext``,
    ``ceil`` on $y_{t+\\Delta}$ and of $\\lVert y_{t+\\Delta} - \\bar y
    \\rVert^2$ (``tot``)."""
    out = {}
    x = x.to(_F64)

    def g(c: Tensor) -> Tensor:
        return (predict_linear(lin, c) if lin is not None else predict(net, c, dev)).to(
            _F64
        )

    for lag in lags:
        fut = y[t + lag].to(_F64)
        conds = {"copy": x[t], "ext": 2 * x[t] - x[t - lag], "ceil": x[t + lag]}
        row = {k: float((fut - g(c)).square().sum()) for k, c in conds.items()}
        out[str(lag)] = row | {"tot": float((fut - mean.to(_F64)).square().sum())}
    return out


def content_rows(
    labels: Mapping[str, Any],
    x_train: Tensor,
    y_train: Tensor,
    x_eval: Tensor,
    y_eval: Tensor,
    ws: WidthSets,
    conf: ContentConf,
    dev: torch.device,
    *,
    lin: bool = True,
    ext: bool = True,
) -> list[Row]:
    r"""The rows of a code (module docstring) of its training codes and
    observations ``x_train`` ``(N, D)``, ``y_train`` ``(N, K)`` and Zurich's
    ``x_eval`` ``(n, D)``, ``y_eval`` ``(n, K)`` (every stack): ``content``
    of the readouts ``mlp`` of ``conf.seeds`` (each seed's training losses
    at its end) and, with ``lin``, ``lin``; with ``ext`` the ``obs_ext``
    record of each seed (and of ``lin``)."""
    t, blocks = _anchors(ws)
    nb = ws.n_blocks
    mean = y_train.to(_F64).mean(0)
    preds, last, exts = [], [], []
    for seed in conf.seeds:
        net, losses = fit_mlp(x_train, y_train, conf.mlp, seed, dev)
        preds.append(predict(net, x_eval, dev))
        last.append(losses[-1] if losses else None)
        if ext:
            exts.append(_obs_ext(net, x_eval, y_eval, t, mean, conf.ext_lags, dev))
    cs = content_sums(y_eval, preds, mean, t, blocks, nb)
    lab = dict(labels)
    rows: list[Row] = [lab | {"part": "content", "readout": "mlp", "seeds": list(conf.seeds),
                              "loss_last": last, "n_anchors": len(t)} | cs.to_row()]  # fmt: skip
    if ext:
        rows.append(lab | {"part": "obs_ext", "readout": "mlp", "seeds": list(conf.seeds),
                           "lags": exts})  # fmt: skip
    if lin:
        b = fit_linear(x_train, y_train)
        p = predict_linear(b, x_eval)
        cl = content_sums(y_eval, [p], mean, t, blocks, nb)
        rows.append(lab | {"part": "content", "readout": "lin", "seeds": [None],
                           "n_anchors": len(t)} | cl.to_row())  # fmt: skip
        if ext:
            rows.append(lab | {"part": "obs_ext", "readout": "lin", "seeds": [None],
                               "lags": [_obs_ext(None, x_eval, y_eval, t, mean,
                                                 conf.ext_lags, dev, b)]})  # fmt: skip
    return rows


# -- the real data (V-h4b) ---------------------------------------------------------------


def interval_codes(
    conf: Any,
    root: Path,
    guard: GuardConf,
    codes: Mapping[str, Callable[[Tensor], Tensor]],
    dev: torch.device,
) -> tuple[dict[str, Tensor], Tensor]:
    """The cheats' training set of several codes in one pass
    (``width_guard.train_intervals``' of each): each code's ``(R, L, D)``
    float64 and the last frames u8 of the runs' middle halves."""
    length, a = guard.cheat_length, guard.cheat_length // 4
    got: dict[str, list[Tensor]] = {name: [] for name in codes}
    last = []
    for run in interval_runs(conf, root, guard):
        with torch.no_grad():
            for name, code in codes.items():
                got[name].append(
                    torch.cat([code(c.to(dev)).double().cpu() for c in run.split(64)])
                )
        last.append(run[a : length - a, -1])
    return {k: torch.stack(v) for k, v in got.items()}, torch.cat(last)


def _yard_white(
    name: str,
    conf: Any,
    ref: Tensor,
    inits: Mapping[str, Path],
    init_v16: Path | None,
    guard: GuardConf,
    gconf: GeomConf,
    dev: torch.device,
) -> WhiteCode:
    """A yardstick of ``YARD`` whitened by the reference as the yard's width
    (``width_eval.evaluate_yard``: its place ``own``, a JEPA's ``zpca`` of
    its run's seed)."""
    pre = None
    if name == "color":
        places: Places = _feature_code(ColorCode(16).mu)
    elif name == "mean8":
        gen = torch.Generator().manual_seed(conf.seed + PCA_SEED)
        pc = PCACode.fit(_mean8, ref, gconf.pca_dim, gen=gen, device=conf.device)
        places = _feature_code(pc.mu)
    elif name in YARD_INITS:
        mods, own, _ = _init_of(inits[name], conf.device)
        places = JepaPlaces(mods).places
        if own == "zpca":
            z = encode_places(places, lambda idx: ref[idx], len(ref), dev)["own"]
            pre = _zpca_map(z, _conf_of(inits[name], None).seed, gconf.pca_dim)
    else:
        assert init_v16 is not None
        vae = vae_at(_first_vae(init_v16), conf.device)
        assert isinstance(vae, VAEModules)
        places = VaeTokenCode(vae).places
    z_ref = encode_places(places, lambda idx: ref[idx], len(ref), dev)["own"]
    if pre is not None:
        z_ref = pre(z_ref)
    return WhiteCode(
        places, "own", Whitener.fit(z_ref, guard.k, c_eps=gconf.c_eps), pre
    )


def content_names(guard: GuardConf) -> tuple[str, ...]:
    """The codes of the real data: ``pca``'s cheats (``base`` first),
    ``rot`` and the yardsticks."""
    cheats = tuple(cheat_codes(torch.zeros(0), torch.zeros(guard.k), guard))
    return (*cheats, "rot", *YARD)


def evaluate_content(
    conf: Any,
    sets_dir: Path,
    out: Path,
    *,
    codes: Collection[str] | None = None,
    inits: Mapping[str, Path] | None = None,
    init_v16: Path | None = None,
    root: Path | None = None,
    local: bool = False,
    local_features: Callable[[Tensor], Tensor] | None = None,
    inception: Path | None = None,
    guard: GuardConf = GuardConf(),
    cconf: ContentConf = ContentConf(),
    gconf: GeomConf = GeomConf(),
    wconf: WidthConf = WidthConf(),
    fconf: FillConf = FillConf(),
    spacing: Mapping[str, Any] | None = None,
    span: float | None = None,
) -> Path:
    """V-h4b's rows (module docstring) of ``codes`` (of ``content_names``;
    None all) into ``out / <code> / content.json``, each code's file the same
    to the byte whichever others run beside it. With ``local`` also the
    local features' rows (``local_features`` of u8 frames, else
    Inception's of ``inception``). Returns ``out``.

    Raises:
        ValueError: A code is unknown, a yardstick's run is not given, or the
            guard's ``k`` is not the width's.
    """
    known = content_names(guard)
    todo = known if codes is None else tuple(c for c in known if c in codes)
    bad = [c for c in codes or () if c not in known]
    if bad:
        raise ValueError(f"no codes {bad} (of {list(known)})")
    inits = dict(inits or {})
    missing = [n for n in todo if (n in YARD_INITS and n not in inits)
               or (n == "init_v16" and init_v16 is None)]  # fmt: skip
    if missing:
        raise ValueError(f"the codes {missing} need their runs")
    same_k(guard, gconf)
    determinism = configure_determinism(conf.device)
    dev = torch.device(conf.device)
    ws, zurich, ref, root = sets_and_ref(conf, sets_dir, root, gconf, wconf, fconf,
                                         spacing, span)  # fmt: skip
    record: dict[str, Any] = {}
    if local and local_features is None:
        _, local_features, record = _features_of(None, None, inception, conf.device,
                                                 cconf.cells)  # fmt: skip
    _start(out, {"guard": asdict(guard), "content": asdict(cconf), "key": ws.key,
                 "local": local, "inception": record.get("sha256")}, determinism)  # fmt: skip
    pca = _yard_code("pca", conf, ref, None, guard, gconf, dev)
    yard = {n: _yard_white(n, conf, ref, inits, init_v16, guard, gconf, dev)
            for n in todo if n in YARD}  # fmt: skip
    train, frames = interval_codes(conf, root, guard,
                                   {"pca": pca.mu} | {n: c.mu for n, c in yard.items()},
                                   dev)  # fmt: skip
    a, length = guard.cheat_length // 4, guard.cheat_length
    last = zurich.frames[ws.span : ws.span + ws.n]
    obs = {"pix": (pool_obs(frames, cconf.obs), pool_obs(last, cconf.obs))}
    if local:
        assert local_features is not None
        obs["local"] = (
            _feats(local_features, frames, dev),
            _feats(local_features, last, dev),
        )
    w_eval = _encode(pca, zurich, ws.n, dev)
    w_train = train["pca"]
    s = lag_speeds(w_train, guard.cheat_lag)
    fns = dict(cheat_codes(w_eval, s, guard))
    q = orthogonal(guard.k, cconf.rot_seed)
    fns["rot"] = lambda x: x @ q
    for name in todo:
        if name in fns:
            x_eval, x_run = fns[name](w_eval), fns[name](w_train)
        else:
            x_eval, x_run = _encode(yard[name], zurich, ws.n, dev), train[name]
        x_train = x_run[:, a : length - a].reshape(-1, x_run.shape[-1])
        rows: list[Row] = []
        for kind, (y_train, y_eval) in obs.items():
            rows += content_rows({"code": name, "obs": kind}, x_train, y_train, x_eval,
                                 y_eval, ws, cconf, dev, lin=True, ext=kind == "pix")  # fmt: skip
        _write(out / name / "content.json", rows, {"key": ws.key})
    return out


def _feats(fn: Callable[[Tensor], Tensor], frames: Tensor, dev: torch.device) -> Tensor:
    """The features (float32, CPU) of u8 frames ``(N, 3, S, S)``."""
    return torch.cat([fn(c.to(dev)).float().cpu() for c in frames.split(256)])


# -- the synthetic checks (V-h4a) -------------------------------------------------------


@dataclass(frozen=True)
class SynthConf:
    r"""The settings of V-h4a (``plans/jepa.md``): V-h0's length and the
    plan's processes (periods and time constants in steps of 0.1 s), seeds
    and thresholds."""

    n: int = 38989
    span: int = 7
    period: float = 300.0  # 30 s
    fast_ks: tuple[int, int] = (4, 5)
    noise: float = 0.1  # the observation's noise, of each dimension's variance
    ou_tau: float = 3.0  # 0.3 s
    f_dims: int = 2
    windows: tuple[int, ...] = (8, 32, 64, 128)
    long_seed: int = 2100
    long_parts: int = 10
    seeds: tuple[int, int] = (2101, 2301)
    cal_seeds: tuple[int, int] = (2301, 2501)
    train_n: int = 131072
    readout_seeds: tuple[int, ...] = (3001, 3002, 3003)
    mlp: MLPConf = MLPConf(hidden=256)
    rot_seed: int = 3101
    psi_star: float = 0.1
    delta_star: float = 0.5
    max_false: int = 13


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _circle(theta: Tensor, ks: Sequence[int]) -> Tensor:
    r"""$(\cos k\theta, \sin k\theta)$ of each $k$ ``(n, 2 |ks|)``."""
    return torch.cat(
        [torch.stack([(k * theta).cos(), (k * theta).sin()], 1) for k in ks], 1
    )


def _phase(n: int, period: float, gen: torch.Generator) -> Tensor:
    r"""$\theta_t = 2\pi t / T + \phi_0$, $\phi_0$ uniform."""
    phi = float(torch.rand(1, generator=gen, dtype=_F64)) * 2 * math.pi
    return 2 * math.pi * torch.arange(n, dtype=_F64) / period + phi


def positive(
    n: int, sc: SynthConf, gen: torch.Generator
) -> tuple[Tensor, dict[str, Tensor]]:
    r"""The positive process (``plans/jepa.md``): $y = (\cos k\theta, \sin
    k\theta)_{k = 1, 4, 5} + \varepsilon$ (each dimension's noise of
    variance ``noise`` times its 1/2) and the codes ``slow`` $\sqrt2(\cos
    \theta, \sin\theta)$ and ``fast`` $\sqrt2(\cos 4\theta, \sin 4\theta,
    \cos 5\theta, \sin 5\theta)$ (each of the other: $\theta = 5\theta -
    4\theta$)."""
    theta = _phase(n, sc.period, gen)
    y = _circle(theta, (1, *sc.fast_ks))
    y = y + math.sqrt(sc.noise / 2) * torch.randn(y.shape, generator=gen, dtype=_F64)
    root2 = math.sqrt(2)
    return y, {"slow": root2 * _circle(theta, (1,)),
               "fast": root2 * _circle(theta, sc.fast_ks)}  # fmt: skip


def _obs_var(sc: SynthConf) -> Tensor:
    return torch.tensor([0.5, 0.5] + [1.0] * sc.f_dims, dtype=_F64)


def negative(n: int, sc: SynthConf, gen: torch.Generator) -> tuple[Tensor, Tensor]:
    r"""The negative process: $y = (\cos\theta, \sin\theta, f_t) +
    \varepsilon$ ($f$ an OU of ``ou_tau`` and unit variance in ``f_dims``
    columns; the noise of variance ``noise`` times each dimension's) and the
    code ``mixed`` $(\sqrt2\cos\theta, \sqrt2\sin\theta, f_t)$."""
    theta = _phase(n, sc.period, gen)
    f = ou(n, sc.f_dims, math.exp(-1 / sc.ou_tau), gen)
    eps = torch.randn(n, 2 + sc.f_dims, generator=gen, dtype=_F64)
    y = torch.cat([_circle(theta, (1,)), f], 1) + (sc.noise * _obs_var(sc)).sqrt() * eps
    return y, torch.cat([math.sqrt(2) * _circle(theta, (1,)), f], 1)


def negative_cheats(mixed: Tensor, sc: SynthConf) -> dict[str, Tensor]:
    """``drop`` (the OU set to its mean 0: a function of the same time) and
    ``mean-<w>`` (the past window's mean, ``width_guard._past_mean``)."""
    drop = mixed.clone()
    drop[:, 2:] = 0
    return {"drop": drop} | {f"mean-{w}": _past_mean(mixed, w) for w in sc.windows}


def calib_noise(sc: SynthConf) -> float:
    r"""The variance $\sigma^2$ of the Gaussian noise on ``mixed``'s OU
    coordinates whose best readout's $E$ is $2^{-\psi_*}$ of ``mixed``'s:
    the rotation's explained variance $R = 1$ (it is kept), the OU's $F =$
    ``f_dims`` shrunk to $F / (1 + \sigma^2)$ ($E[f \mid f + n] = f / (1 +
    \sigma^2)$ of unit Gaussians), the noise's the same in both, so $(R + F /
    (1 + \sigma^2)) / (R + F) = 2^{-\psi_*}$."""
    r, f = 1.0, float(sc.f_dims)
    rho = 2**-sc.psi_star
    return f / (rho * (r + f) - r) - 1


def _noisy(mixed: Tensor, sigma2: float, gen: torch.Generator) -> Tensor:
    x = mixed.clone()
    x[:, 2:] += math.sqrt(sigma2) * torch.randn(
        x[:, 2:].shape, generator=gen, dtype=_F64
    )
    return x


def _fit_sums(
    x_train: Tensor,
    y_train: Tensor,
    evals: Sequence[tuple[Tensor, Tensor]],
    ws: WidthSets,
    sc: SynthConf,
    dev: torch.device,
) -> ContentSums:
    """The sums of the readouts of ``readout_seeds`` on each segment of
    ``evals`` (codes, observations), its blocks side by side."""
    t, blocks = _anchors(ws)
    mean = y_train.mean(0)
    nets = [fit_mlp(x_train, y_train, sc.mlp, s, dev)[0] for s in sc.readout_seeds]
    parts = [content_sums(y, [predict(net, x, dev) for net in nets], mean, t, blocks,
                          ws.n_blocks) for x, y in evals]  # fmt: skip
    return ContentSums.cat(parts)


def _widths(xs: Sequence[Tensor], ws: WidthSets, wconf: WidthConf) -> LinSums:
    return _cat([_common_sums(x, ws, wconf) for x in xs])


def _pair(
    la: LinSums, lb: LinSums, wconf: WidthConf, gconf: GeomConf
) -> dict[str, Any]:
    pr = pair_ratio(la, lb, wconf, gconf)
    return {
        "mean1": list(pr["mean1"]),
        "cls": pr["mean1_cls"],
        "windowed": pr["windowed"],
    }


def _mean1(ls: LinSums, wconf: WidthConf) -> float:
    return point(ls, wconf)["25"]["mean1"]


def _a1(sc: SynthConf, ws: WidthSets, gconf: GeomConf, wconf: WidthConf, seed: int,
        parts: int, dev: torch.device) -> dict[str, Any]:  # fmt: skip
    """A realization of the positive process (``parts`` segments; the long
    one with the rotated ``slow``): the width's pair and each code's sums."""
    gen = _gen(seed)
    segs = [positive(sc.n, sc, gen) for _ in range(parts)]
    y_tr, c_tr = positive(sc.train_n, sc, gen)
    names = ["slow", "fast"]
    trains, evals = dict(c_tr), {k: [c[k] for _, c in segs] for k in names}
    if parts > 1:
        q = orthogonal(2, sc.rot_seed)
        trains["slow-rot"] = c_tr["slow"] @ q
        evals["slow-rot"] = [x @ q for x in evals["slow"]]
    content = {k: _fit_sums(trains[k], y_tr, list(zip(evals[k], [y for y, _ in segs], strict=True)),
                            ws, sc, dev).to_row() for k in evals}  # fmt: skip
    ls = {k: _widths(evals[k], ws, wconf) for k in names}
    return {"seed": seed, "width": _pair(ls["slow"], ls["fast"], wconf, gconf),
            "mean1": {k: _mean1(v, wconf) for k, v in ls.items()}, "content": content}  # fmt: skip


def _negative_set(n: int, sc: SynthConf, gen: torch.Generator, warm: int
                  ) -> tuple[Tensor, dict[str, Tensor]]:  # fmt: skip
    """A negative series of ``n`` times past ``warm`` (the past means' warm-up,
    dropped): its observations and codes (``mixed`` and its cheats)."""
    y, mixed = negative(n + warm, sc, gen)
    codes = {"mixed": mixed} | negative_cheats(mixed, sc)
    return y[warm:], {k: v[warm:] for k, v in codes.items()}


def _a2(sc: SynthConf, ws: WidthSets, gconf: GeomConf, wconf: WidthConf, seed: int,
        parts: int, dev: torch.device) -> dict[str, Any]:  # fmt: skip
    """A realization of the negative process (the long one with the rotated
    ``mixed``): each cheat's width's pair against ``mixed`` and each code's
    sums. The segments start at time 0 (a past mean of fewer at the start;
    the anchors lie past the largest window)."""
    gen = _gen(seed)
    segs = [negative(sc.n, sc, gen) for _ in range(parts)]
    y_tr, c_tr = _negative_set(sc.train_n, sc, gen, max(sc.windows))
    evals = {"mixed": [m for _, m in segs]}
    cheats_of = [negative_cheats(m, sc) for _, m in segs]
    for k in cheats_of[0]:
        evals[k] = [c[k] for c in cheats_of]
    if parts > 1:
        q = orthogonal(2 + sc.f_dims, sc.rot_seed)
        c_tr["mixed-rot"] = c_tr["mixed"] @ q
        evals["mixed-rot"] = [x @ q for x in evals["mixed"]]
    ys = [y for y, _ in segs]
    content = {k: _fit_sums(c_tr[k], y_tr, list(zip(v, ys, strict=True)), ws, sc,
                            dev).to_row() for k, v in evals.items()}  # fmt: skip
    cheats = [k for k in evals if k not in ("mixed", "mixed-rot")]
    ls = {k: _widths(evals[k], ws, wconf) for k in ["mixed", *cheats]}
    return {"seed": seed,
            "width": {k: _pair(ls[k], ls["mixed"], wconf, gconf) for k in cheats},
            "mean1": {k: _mean1(v, wconf) for k, v in ls.items()}, "content": content}  # fmt: skip


def _a3(sc: SynthConf, ws: WidthSets, seed: int, dev: torch.device) -> dict[str, Any]:
    """A realization of the calibration: ``mixed`` and ``noisy``'s sums."""
    gen = _gen(seed)
    y, mixed = negative(sc.n, sc, gen)
    y_tr, m_tr = negative(sc.train_n, sc, gen)
    s2 = calib_noise(sc)
    noisy, n_tr = _noisy(mixed, s2, gen), _noisy(m_tr, s2, gen)
    content = {"mixed": _fit_sums(m_tr, y_tr, [(mixed, y)], ws, sc, dev).to_row(),
               "noisy": _fit_sums(n_tr, y_tr, [(noisy, y)], ws, sc, dev).to_row()}  # fmt: skip
    return {"seed": seed, "sigma2": s2, "content": content}


LONG_UNITS = ("V-h4a1-long", "V-h4a2-long")
SEEDED = ("V-h4a1", "V-h4a2", "V-h4a3")


def check_units() -> tuple[str, ...]:
    return LONG_UNITS + SEEDED


def _seeds_of(unit: str, sc: SynthConf) -> range:
    return range(*(sc.cal_seeds if unit == "V-h4a3" else sc.seeds))


def check_parts(
    sc: SynthConf,
    gconf: GeomConf,
    wconf: WidthConf,
    units: Collection[str],
    seeds: tuple[int, int] | None = None,
    device: str = "cpu",
) -> list[Row]:
    """The rows of the ``units`` (``check_units``) of one process for
    ``merge_checks``: of a long unit ``{"part": "long", "unit", "value"}``,
    of each realization of a seeded unit whose seed lies in ``seeds``
    (``lo``, ``hi``: $lo \\le$ seed $< hi$; None: all) ``{"part": "draw",
    "unit", "value"}``.

    Raises:
        ValueError: A unit is not one of ``check_units``.
    """
    bad = [u for u in units if u not in check_units()]
    if bad:
        raise ValueError(f"no check units {bad} (of {list(check_units())})")
    ws = synthetic_sets(sc.n, sc.span, gconf.block, wconf.max_lag)
    dev = torch.device(device)
    lo, hi = (-math.inf, math.inf) if seeds is None else seeds
    rows: list[Row] = []
    for u in check_units():
        if u not in units:
            continue
        if u == "V-h4a1-long":
            v = _a1(sc, ws, gconf, wconf, sc.long_seed, sc.long_parts, dev)
            rows.append({"part": "long", "unit": u, "value": v})
        elif u == "V-h4a2-long":
            v = _a2(sc, ws, gconf, wconf, sc.long_seed, sc.long_parts, dev)
            rows.append({"part": "long", "unit": u, "value": v})
        else:
            for s in _seeds_of(u, sc):
                if not lo <= s < hi:
                    continue
                if u == "V-h4a1":
                    v = _a1(sc, ws, gconf, wconf, s, 1, dev)
                elif u == "V-h4a2":
                    v = _a2(sc, ws, gconf, wconf, s, 1, dev)
                else:
                    v = _a3(sc, ws, s, dev)
                rows.append({"part": "draw", "unit": u, "value": v})
    return rows


def _sums(v: Mapping[str, Any], name: str) -> ContentSums:
    return ContentSums.of_row(v["content"][name])


def _sigmas(a: ContentSums, b: ContentSums) -> tuple[float, float]:
    return seed_sigma(a, b), seed_sigma(a, b, change=True)


def _check(name: str, ok: bool, **values: Any) -> Row:
    return {"part": "check", "name": name, "pass": bool(ok)} | values


def _record(name: str, **values: Any) -> Row:
    return {"part": "record", "name": name} | values


def merge_checks(
    parts: Sequence[Row], sc: SynthConf, gconf: GeomConf = GeomConf()
) -> list[Row]:
    """V-h4a's rows (module docstring) of the parts' rows (any split, any
    order): of each check ``part`` ``check`` (a failure stops) and the
    records, $\\sigma_{\\mathrm{seed}}$ of each process among them.

    Raises:
        ValueError: A unit or a realization is missing or given twice.
    """
    long: dict[str, Any] = {}
    got: dict[str, dict[int, Any]] = {u: {} for u in SEEDED}
    for r in parts:
        u = r["unit"]
        if r["part"] == "long":
            if u in long:
                raise ValueError(f"{u} given twice")
            long[u] = r["value"]
        else:
            s = r["value"]["seed"]
            if s in got[u]:
                raise ValueError(f"{u} {s} given twice")
            got[u][s] = r["value"]
    if len(_seeds_of("V-h4a3", sc)) != len(_seeds_of("V-h4a2", sc)):
        raise ValueError("V-h4a3's power pairs each realization with V-h4a2's")
    missing = [u for u in LONG_UNITS if u not in long]
    missing += [f"{u} {s}" for u in SEEDED for s in _seeds_of(u, sc) if s not in got[u]]
    if missing:
        raise ValueError(f"missing {missing}")
    l1, l2 = long["V-h4a1-long"], long["V-h4a2-long"]
    sig_pos = _sigmas(_sums(l1, "slow-rot"), _sums(l1, "slow"))
    sig_neg = _sigmas(_sums(l2, "mixed-rot"), _sums(l2, "mixed"))
    nb_long = len(l1["content"]["slow"]["tot_b"])
    nb = len(got["V-h4a1"][sc.seeds[0]]["content"]["slow"]["tot_b"])
    c_long = (counts_of(nb_long, gconf), counts_of(nb_long // 2, gconf))
    c_one = (counts_of(nb, gconf), counts_of(nb // 2, gconf))
    star = (sc.delta_star, sc.psi_star)
    rows: list[Row] = [_record("V-h4a-sigma", positive=list(sig_pos), negative=list(sig_neg),
                               calib_sigma2=calib_noise(sc))]  # fmt: skip
    # V-h4a1: the positive control
    f1 = front_of(tuple(l1["width"]["mean1"]), _sums(l1, "slow"), _sums(l1, "fast"),
                  *c_long, sig_pos, *star)  # fmt: skip
    rows.append(_check("V-h4a1", f1["verdict"] == "kept", front=f1, mean1=l1["mean1"]))
    kept1 = [front_of(tuple(v["width"]["mean1"]), _sums(v, "slow"), _sums(v, "fast"),
                      *c_one, sig_pos, *star)["verdict"] == "kept"
             for _, v in sorted(got["V-h4a1"].items())]  # fmt: skip
    rows.append(_record("V-h4a1-share", kept=sum(kept1), n=len(kept1)))
    # V-h4a2: the negative control
    fronts = {k: front_of(tuple(w["mean1"]), _sums(l2, k), _sums(l2, "mixed"), *c_long,
                          sig_neg, *star) for k, w in l2["width"].items()}  # fmt: skip
    claim = [k for k, f in fronts.items() if f["verdict"] != "width"]
    ok2 = bool(claim) and all(fronts[k]["verdict"] == "dropped" for k in claim)
    rows.append(_check("V-h4a2", ok2, claim=claim, fronts=fronts, mean1=l2["mean1"]))
    false2 = []
    for s, v in sorted(got["V-h4a2"].items()):
        verdicts = {k: front_of(tuple(w["mean1"]), _sums(v, k), _sums(v, "mixed"), *c_one,
                                sig_neg, *star)["verdict"] for k, w in v["width"].items()}  # fmt: skip
        if "kept" in verdicts.values():
            false2.append({"seed": s, "verdicts": verdicts})
    rows.append(_check("V-h4a2-count", len(false2) <= sc.max_false, kept=len(false2),
                       n=len(got["V-h4a2"]), which=false2))  # fmt: skip
    # V-h4a3: the margin's calibration, and the power of the same code
    shift = sc.cal_seeds[0] - sc.seeds[0]
    passed, both, power, power_both = 0, 0, 0, 0
    for s, v in sorted(got["V-h4a3"].items()):
        a, b = _sums(v, "noisy"), _sums(v, "mixed")
        psi = ratio_interval(a, b, *c_one, sig_neg[0])
        psi_d = ratio_interval(a, b, *c_one, sig_neg[1], change=True)
        comp = all(bool(comparable(b, *c_one, change=ch)["ok"]) for ch in (False, True))
        lo_ok = comp and _lo(psi) > -sc.psi_star
        passed += lo_ok
        both += lo_ok and _lo(psi_d) > -sc.psi_star
        other = _sums(got["V-h4a2"][s - shift], "mixed")
        p = ratio_interval(b, other, *c_one, sig_neg[0])
        p_d = ratio_interval(b, other, *c_one, sig_neg[1], change=True)
        p_ok = _lo(p) > -sc.psi_star
        power += p_ok
        power_both += p_ok and _lo(p_d) > -sc.psi_star
    n3 = len(got["V-h4a3"])
    rows.append(_check("V-h4a3", passed <= sc.max_false, kept=passed, n=n3,
                       kept_both=both))  # fmt: skip
    rows.append(_record("V-h4a3-power", kept=power, kept_both=power_both, n=n3))
    return rows


def _lo(v: Mapping[str, float | None]) -> float:
    x = v["lo"]
    return -math.inf if x is None else x


def check_rows(
    sc: SynthConf = SynthConf(),
    gconf: GeomConf = GeomConf(),
    wconf: WidthConf = WidthConf(),
    device: str = "cpu",
) -> list[Row]:
    """V-h4a in one process (``check_parts`` of every unit, merged)."""
    return merge_checks(check_parts(sc, gconf, wconf, check_units(), None, device), sc,
                        gconf)  # fmt: skip


def _check_config(sc: SynthConf, gconf: GeomConf, wconf: WidthConf) -> dict[str, Any]:
    return {"synth": asdict(sc), "geom": asdict(gconf), "width": asdict(wconf)}


def evaluate_checks(
    out: Path,
    sc: SynthConf = SynthConf(),
    gconf: GeomConf = GeomConf(),
    wconf: WidthConf = WidthConf(),
    *,
    units: Collection[str] | None = None,
    seeds: tuple[int, int] | None = None,
    device: str = "cpu",
) -> Path:
    """V-h4a's parts into ``out`` (``config.json``, ``rows.json``); with
    neither ``units`` nor ``seeds`` the whole, merged."""
    determinism = configure_determinism(device)
    _start(out, _check_config(sc, gconf, wconf) | {"device": device}, determinism)
    if units is None and seeds is None:
        rows = check_rows(sc, gconf, wconf, device)
    else:
        rows = check_parts(sc, gconf, wconf, check_units() if units is None else units,
                           seeds, device)  # fmt: skip
    _write(out / "rows.json", rows, {})
    return out


def merge_checks_into(
    out: Path,
    parts: Sequence[Path],
    sc: SynthConf = SynthConf(),
    gconf: GeomConf = GeomConf(),
    wconf: WidthConf = WidthConf(),
) -> Path:
    """V-h4a's ``rows.json`` in ``out`` of the parts' directories.

    Raises:
        ValueError: A part is of other settings, or a unit or a realization
            is missing or given twice.
    """
    want = json.loads(json.dumps(_check_config(sc, gconf, wconf)))
    rows: list[Row] = []
    devices = set()
    for d in parts:
        conf = json.loads((d / "config.json").read_text())
        if {k: conf.get(k) for k in want} != want:
            raise ValueError(f"{d} is of other settings")
        devices.add(conf.get("device"))
        rows += json.loads((d / "rows.json").read_text())
    _start(out, _check_config(sc, gconf, wconf) | {"devices": sorted(map(str, devices))},
           configure_determinism("cpu"))  # fmt: skip
    _write(out / "rows.json", merge_checks(rows, sc, gconf), {})
    return out


# -- the command line -------------------------------------------------------------------


def _names(text: str) -> tuple[str, ...]:
    return tuple(x for x in text.split(",") if x)


def _range(text: str) -> tuple[int, int]:
    lo, _, hi = text.partition(":")
    return int(lo), int(hi)


def main(argv: Sequence[str] | None = None) -> Path:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = p.add_subparsers(dest="job", required=True)
    checks = sub.add_parser("checks")
    checks.add_argument("--out", type=Path, required=True)
    checks.add_argument(
        "--only", type=_names, default=None, help="units, comma-separated"
    )
    checks.add_argument("--seeds", type=_range, default=None, help="realizations lo:hi")
    checks.add_argument("--device", default="cpu")
    merge = sub.add_parser("merge-checks")
    merge.add_argument("--out", type=Path, required=True)
    merge.add_argument("parts", type=Path, nargs="+")
    real = sub.add_parser("real")
    real.add_argument("--config", type=Path, required=True)
    real.add_argument("--sets", type=Path, required=True)
    real.add_argument("--out", type=Path, required=True)
    real.add_argument("--codes", type=_names, default=None, help="comma-separated")
    real.add_argument("--local", action="store_true")
    real.add_argument("--inception", type=Path, default=None)
    real.add_argument("--device", default=None)
    real.add_argument("--root", type=Path, default=None)
    for name in (*YARD_INITS, "init_v16"):
        real.add_argument(f"--{name.replace('_', '-')}", type=Path, default=None)
    a = p.parse_args(argv)
    if a.job == "checks":
        return evaluate_checks(a.out, units=a.only, seeds=a.seeds, device=a.device)
    if a.job == "merge-checks":
        return merge_checks_into(a.out, a.parts)
    conf = _conf_of(a.config, a.device)
    inits = {n: getattr(a, n) for n in YARD_INITS if getattr(a, n) is not None}
    return evaluate_content(conf, a.sets, a.out, codes=a.codes, inits=inits,
                            init_v16=a.init_v16, root=a.root, local=a.local,
                            inception=a.inception)  # fmt: skip


if __name__ == "__main__":
    main()
