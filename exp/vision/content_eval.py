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
and their denominator) into ``<out>/<code>/content.json``. Records beside
(the review's, ``plans/jepa.md``'s proposition 1 and $E^d$): ``err_rho1``,
the level error's lag-1 autocorrelation on Zurich, and rows ``fit`` of the
same readouts' sums on the training set (``train``) and on
``ContentConf.holdout_count`` runs of the training videos overlapping none
of its runs (``holdout``; ``pix`` only).

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
  index (independent, $\psi = 0$); ``V-h4a3-d`` (recorded): ``noisy-d``
  (the noise of ``calib_noise_d``, $\psi^d = -\psi_*$ exactly) against
  ``mixed``, the realizations where it is comparable and $\psi^d$'s lower
  end clears $-\psi_*$

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

**v3** (``plans/jepa.md``, 「線形の幅の内容のガード v3」; README, 「線形の幅の
内容のガード v3（回す前に決めたこと、2026-11-26）」): of each code
``select_rows`` chooses the level's readout among ``SelectConf``'s grid
(``fidelity.fit_ridge`` of 4 $\alpha$, the MLPs of 2 widths stopped every
1000 steps) by the error on validation runs of the training videos
(``RunSet``; at least ``val_gap`` stacks from the training and the
record's runs, ``width_guard.slot_draws``), and the change's among the same
learned from the pairs $(c_t, c_{t-1})$ to $d_t$ and the chosen level's
difference; the rows keep the sums of the chosen (``sel``), the least
squares (``lin``, ``lin-diff``) and v2's MLP (``v2``) on the training,
validation and record runs and on Zurich. The verdicts use the contrast
$D = E_a - 2^{-\psi_*} E_b$ (``pair3``, ``front3``). V-h5a
(``process3``: the training series of the real data's length and runs, the
scene $s$ in the codes, its detail $n$ and the fast $g$ not) runs as
V-h4a (``checks3``, ``merge-checks3``); V-h5b (``evaluate_content3``)
writes ``<out>/<code>/content3.json`` and the manifest
``<out>/intervals.json``::

    python -m exp.vision.content_eval checks3 --out <part> --only V-h5a2-long
    ... checks3 --out <part> --only V-h5a1,V-h5a2,V-h5a3 --seeds 2101:2111
    ... merge-checks3 --out runs/v18c-width/content3/checks <part> <part> ..
    python -m exp.vision.content_eval real3 --config <v18c run> --sets runs/v18c-tlin \
        --out runs/v18c-width/content3 --codes base,rot [--init <..> ..]
"""

import argparse
import json
import math
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from functools import cache, partial
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from exp.lib.fidelity import (
    ContentSums,
    MLPConf,
    Predictor,
    candidate_names,
    comparable,
    content_sums,
    contrast_interval,
    contrast_seed_sd,
    contrast_verdict,
    diff_predictor,
    error_lag1,
    fit_linear,
    fit_mlp,
    fit_mlp_path,
    fit_ridge,
    front_verdict,
    kind_of,
    linear_predictor,
    noninferior,
    pair_inputs,
    pool_obs,
    predict,
    predict_linear,
    ratio_interval,
    run_errors,
    seed_sigma,
    select_candidate,
    snapshot_predictor,
    stack_seeds,
)
from exp.lib.metrics import block_counts
from exp.lib.midpoint import Whitener
from exp.lib.provenance import configure_determinism
from exp.lib.synth_series import gauss_kernel, kernel_acf, ou, smooth, white
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
    interval_draws,
    lag_speeds,
    same_k,
    slot_draws,
    slot_starts,
)

__all__ = [
    "LONG3",
    "LONG_UNITS",
    "SEEDED",
    "SEEDED3",
    "YARD",
    "ContentConf",
    "ContentConf3",
    "RunSet",
    "SelectConf",
    "Synth3Conf",
    "SynthConf",
    "calib3",
    "calib3_d",
    "calib_noise",
    "calib_noise_d",
    "change_explained",
    "change_parts3",
    "cheats3",
    "check_parts",
    "check_parts3",
    "check_rows",
    "check_units3",
    "chosen3",
    "columns3",
    "content_rows",
    "diff_ed3",
    "draw_starts",
    "evaluate_checks",
    "evaluate_checks3",
    "evaluate_content",
    "evaluate_content3",
    "front3",
    "front_of",
    "interval_codes",
    "main",
    "merge_checks",
    "merge_checks3",
    "merge_checks3_into",
    "merge_checks_into",
    "negative",
    "orthogonal",
    "pair3",
    "pair_ed3",
    "parts3",
    "positive",
    "process3",
    "select_rows",
    "starts3",
    "sums3",
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
    holdout_seed: int = 3201  # the held-out runs of the training videos (a record)
    holdout_count: int = 64


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
    g: Callable[[Tensor], Tensor], x: Tensor, y: Tensor, t: Tensor, mean: Tensor,
    lags: Sequence[int],
) -> dict[str, dict[str, float]]:  # fmt: skip
    """The record of the extrapolation in the observation (module
    docstring) of the readout ``g`` at each lag: the sums of the errors of
    ``copy``, ``ext``, ``ceil`` on $y_{t+\\Delta}$ and of $\\lVert
    y_{t+\\Delta} - \\bar y \\rVert^2$ (``tot``)."""
    out = {}
    x = x.to(_F64)
    for lag in lags:
        fut = y[t + lag].to(_F64)
        conds = {"copy": x[t], "ext": 2 * x[t] - x[t - lag], "ceil": x[t + lag]}
        row = {k: float((fut - g(c).to(_F64)).square().sum()) for k, c in conds.items()}
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
    fits: Mapping[str, tuple[Tensor, Tensor, int]] | None = None,
) -> list[Row]:
    r"""The rows of a code (module docstring) of its training codes and
    observations ``x_train`` ``(N, D)``, ``y_train`` ``(N, K)`` and Zurich's
    ``x_eval`` ``(n, D)``, ``y_eval`` ``(n, K)`` (every stack): ``content``
    of the readouts ``mlp`` of ``conf.seeds`` (each seed's training losses
    at its end and its level error's lag-1 autocorrelation, ``err_rho1``)
    and, with ``lin``, ``lin``; with ``ext`` the ``obs_ext`` record of each
    seed (and of ``lin``); of each of ``fits`` (a name to the codes, the
    observations and the length $l$ of the runs laid end to end; a record)
    the ``fit`` rows of the readouts' sums there (``set`` the name; the
    anchors every time but each run's first, a block each run)."""
    t, blocks = _anchors(ws)
    nb = ws.n_blocks
    mean = y_train.to(_F64).mean(0)
    preds, last, exts = [], [], []
    fit_sums: dict[str, list[ContentSums]] = {k: [] for k in fits or {}}
    for seed in conf.seeds:
        net, losses = fit_mlp(x_train, y_train, conf.mlp, seed, dev)
        preds.append(predict(net, x_eval, dev))
        last.append(losses[-1] if losses else None)
        if ext:
            exts.append(_obs_ext(partial(predict, net, dev=dev), x_eval, y_eval, t, mean,
                                 conf.ext_lags))  # fmt: skip
        for k, (xs, ys, run) in (fits or {}).items():
            fit_sums[k].append(_run_sums(ys, predict(net, xs, dev), mean, run))
    cs = content_sums(y_eval, preds, mean, t, blocks, nb)
    lab = dict(labels)
    rows: list[Row] = [lab | {"part": "content", "readout": "mlp", "seeds": list(conf.seeds),
                              "loss_last": last, "n_anchors": len(t),
                              "err_rho1": error_lag1(y_eval, preds, t)} | cs.to_row()]  # fmt: skip
    for k, parts in fit_sums.items():
        rows.append(lab | {"part": "fit", "set": k, "readout": "mlp",
                           "seeds": list(conf.seeds)} | stack_seeds(parts).to_row())  # fmt: skip
    if ext:
        rows.append(lab | {"part": "obs_ext", "readout": "mlp", "seeds": list(conf.seeds),
                           "lags": exts})  # fmt: skip
    if lin:
        b = fit_linear(x_train, y_train)
        p = predict_linear(b, x_eval)
        cl = content_sums(y_eval, [p], mean, t, blocks, nb)
        rows.append(lab | {"part": "content", "readout": "lin", "seeds": [None],
                           "n_anchors": len(t)} | cl.to_row())  # fmt: skip
        for k, (xs, ys, run) in (fits or {}).items():
            rows.append(lab | {"part": "fit", "set": k, "readout": "lin", "seeds": [None]}
                        | _run_sums(ys, predict_linear(b, xs), mean, run).to_row())  # fmt: skip
        if ext:
            rows.append(lab | {"part": "obs_ext", "readout": "lin", "seeds": [None],
                               "lags": [_obs_ext(linear_predictor(b), x_eval, y_eval,
                                                 t, mean, conf.ext_lags)]})  # fmt: skip
    return rows


def _run_sums(y: Tensor, pred: Tensor, mean: Tensor, run: int) -> ContentSums:
    """The sums of one readout on runs of length ``run`` laid end to end:
    the anchors every time but each run's first, a block each run."""
    idx = torch.arange(len(y))
    t = idx[idx % run != 0]
    return content_sums(y, [pred], mean, t, t // run, len(y) // run)


# -- the real data (V-h4b) ---------------------------------------------------------------


def interval_codes(
    conf: Any,
    root: Path,
    guard: GuardConf,
    codes: Mapping[str, Callable[[Tensor], Tensor]],
    dev: torch.device,
    *,
    holdout: tuple[int, int, Collection[int]] | None = None,
    gap: int | None = None,
) -> tuple[dict[str, Tensor], Tensor, list[int]]:
    """The cheats' training set of several codes in one pass
    (``width_guard.train_intervals``' of each): each code's ``(R, L, D)``
    float64, the last frames u8 of the runs' middle halves and the runs'
    starts. With ``holdout`` (seed, count, the training runs' starts) the
    same of ``count`` runs of the training videos overlapping none of the
    training runs nor each other (``width_guard.interval_draws``; with
    ``gap`` v3's validation runs, ``width_guard.slot_draws``: at least
    ``gap`` stacks from the given starts' runs)."""
    length, a = guard.cheat_length, guard.cheat_length // 4
    got: dict[str, list[Tensor]] = {name: [] for name in codes}
    last, starts = [], []
    if holdout is None:
        draws = interval_draws(conf, root, length, guard.cheat_count, guard.cheat_seed)
    elif gap is None:
        seed, count, avoid = holdout
        draws = interval_draws(
            conf, root, length, count, seed, avoid, tries=1000 * count
        )
    else:
        seed, count, avoid = holdout
        draws = slot_draws(conf, root, length, count, seed, avoid, gap)
    for k, run in draws:
        with torch.no_grad():
            for name, code in codes.items():
                got[name].append(
                    torch.cat([code(c.to(dev)).double().cpu() for c in run.split(64)])
                )
        last.append(run[a : length - a, -1])
        starts.append(k)
    return {k: torch.stack(v) for k, v in got.items()}, torch.cat(last), starts


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
    mus = {"pca": pca.mu} | {n: c.mu for n, c in yard.items()}
    train, frames, starts = interval_codes(conf, root, guard, mus, dev)
    hold, h_frames, _ = interval_codes(conf, root, guard, mus, dev,
                                       holdout=(cconf.holdout_seed, cconf.holdout_count,
                                                starts))  # fmt: skip
    a, length = guard.cheat_length // 4, guard.cheat_length
    last = zurich.frames[ws.span : ws.span + ws.n]
    obs = {"pix": (pool_obs(frames, cconf.obs), pool_obs(last, cconf.obs))}
    y_hold = pool_obs(h_frames, cconf.obs)
    if local:
        assert local_features is not None
        obs["local"] = (
            _feats(local_features, frames, dev),
            _feats(local_features, last, dev),
        )
    w_eval = _encode(pca, zurich, ws.n, dev)
    w_train, w_hold = train["pca"], hold["pca"]
    s = lag_speeds(w_train, guard.cheat_lag)
    fns = dict(cheat_codes(w_eval, s, guard))
    q = orthogonal(guard.k, cconf.rot_seed)
    fns["rot"] = lambda x: x @ q
    for name in todo:
        if name in fns:
            x_eval, x_run, x_h = (
                fns[name](w_eval),
                fns[name](w_train),
                fns[name](w_hold),
            )
        else:
            x_eval, x_run, x_h = (
                _encode(yard[name], zurich, ws.n, dev),
                train[name],
                hold[name],
            )
        x_train = x_run[:, a : length - a].reshape(-1, x_run.shape[-1])
        x_h = x_h[:, a : length - a].reshape(-1, x_h.shape[-1])
        rows: list[Row] = []
        for kind, (y_train, y_eval) in obs.items():
            fits = None if kind != "pix" else {"train": (x_train, y_train, length - 2 * a),
                                               "holdout": (x_h, y_hold, length - 2 * a)}  # fmt: skip
            rows += content_rows({"code": name, "obs": kind}, x_train, y_train, x_eval,
                                 y_eval, ws, cconf, dev, lin=True, ext=kind == "pix",
                                 fits=fits)  # fmt: skip
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


def change_explained(sc: SynthConf, sigma2: float) -> float:
    r"""The population's $E^d$ of the best readout of ``mixed`` with
    Gaussian noise of variance $\sigma^2$ on its OU coordinates (``_a3``):
    with $a = 1 / (1 + \sigma^2)$ and $\phi = e^{-1/\tau}$ the OU's lag-1
    correlation, each OU coordinate's change error $(1 - a)(f_t - f_{t-1})
    - a(n_t - n_{t-1}) + (\varepsilon_t - \varepsilon_{t-1})$ has the
    variance $2(1 - a)(1 - \phi + a\phi) + 2\nu$ ($\nu$ ``noise``; $2a^2
    \sigma^2 = 2a(1 - a)$), the rotation's only its noise's $2\nu / 2$ a
    coordinate; the change's power is $2 - 2\cos(2\pi / T)$ of the
    rotation, $2(1 - \phi)$ an OU coordinate, and the noise's
    $2\nu(1 + F)$ ($F$ ``f_dims``)."""
    nu, f = sc.noise, float(sc.f_dims)
    phi = math.exp(-1 / sc.ou_tau)
    a = 1 / (1 + sigma2)
    den = (
        2 - 2 * math.cos(2 * math.pi / sc.period) + 2 * f * (1 - phi) + 2 * nu * (1 + f)
    )
    num = 2 * nu * (1 + f) + 2 * f * (1 - a) * (1 - phi + a * phi)
    return 1 - num / den


def calib_noise_d(sc: SynthConf) -> float:
    r"""The variance $\sigma^2$ of the Gaussian noise on ``mixed``'s OU
    coordinates whose best readout's $E^d$ (``change_explained``) is
    $2^{-\psi_*}$ of ``mixed``'s: with $u = 1 - a$ the error's excess $2F
    u(1 - u\phi) = (1 - 2^{-\psi_*})(D - N_0)$ ($D$ the change's power,
    $N_0$ the noise's error), the smaller root $u = (1 - \sqrt{1 - 4\phi
    H}) / (2\phi)$ of $\phi u^2 - u + H = 0$, $H = (1 - 2^{-\psi_*})(D -
    N_0) / (2F)$ (the excess rises with $\sigma^2$ up to $u = 1/(2\phi)$),
    and $\sigma^2 = u / (1 - u)$."""
    nu, f = sc.noise, float(sc.f_dims)
    phi = math.exp(-1 / sc.ou_tau)
    den = (
        2 - 2 * math.cos(2 * math.pi / sc.period) + 2 * f * (1 - phi) + 2 * nu * (1 + f)
    )
    h = (1 - 2**-sc.psi_star) * (den - 2 * nu * (1 + f)) / (2 * f)
    u = (1 - math.sqrt(1 - 4 * phi * h)) / (2 * phi)
    return u / (1 - u)


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
    r"""A realization of the calibration: ``mixed``, ``noisy`` ($\psi =
    -\psi_*$) and ``noisy-d``'s ($\psi^d = -\psi_*$; its noise drawn after
    ``noisy``'s, so ``noisy`` is as without it) sums."""
    gen = _gen(seed)
    y, mixed = negative(sc.n, sc, gen)
    y_tr, m_tr = negative(sc.train_n, sc, gen)
    s2, s2d = calib_noise(sc), calib_noise_d(sc)
    noisy, n_tr = _noisy(mixed, s2, gen), _noisy(m_tr, s2, gen)
    noisy_d, nd_tr = _noisy(mixed, s2d, gen), _noisy(m_tr, s2d, gen)
    content = {"mixed": _fit_sums(m_tr, y_tr, [(mixed, y)], ws, sc, dev).to_row(),
               "noisy": _fit_sums(n_tr, y_tr, [(noisy, y)], ws, sc, dev).to_row(),
               "noisy-d": _fit_sums(nd_tr, y_tr, [(noisy_d, y)], ws, sc, dev).to_row()}  # fmt: skip
    return {"seed": seed, "sigma2": s2, "sigma2_d": s2d, "content": content}


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
    s2, s2d = calib_noise(sc), calib_noise_d(sc)
    rows: list[Row] = [_record("V-h4a-sigma", positive=list(sig_pos), negative=list(sig_neg),
                               calib_sigma2=s2, calib_sigma2_d=s2d,
                               calib_psi_d=math.log2(change_explained(sc, s2)
                                                     / change_explained(sc, 0.0)))]  # fmt: skip
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
    passed, both, power, power_both, passed_d = 0, 0, 0, 0, 0
    for s, v in sorted(got["V-h4a3"].items()):
        a, b = _sums(v, "noisy"), _sums(v, "mixed")
        psi = ratio_interval(a, b, *c_one, sig_neg[0])
        psi_d = ratio_interval(a, b, *c_one, sig_neg[1], change=True)
        comp = all(bool(comparable(b, *c_one, change=ch)["ok"]) for ch in (False, True))
        lo_ok = comp and _lo(psi) > -sc.psi_star
        passed += lo_ok
        both += lo_ok and _lo(psi_d) > -sc.psi_star
        dd = ratio_interval(_sums(v, "noisy-d"), b, *c_one, sig_neg[1], change=True)
        passed_d += comp and _lo(dd) > -sc.psi_star
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
    rows.append(_record("V-h4a3-d", kept=passed_d, n=n3, max_false=sc.max_false))
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


# -- v3: the readout chosen on held-out runs (plans/jepa.md, 「線形の幅の内容のガード v3」) ----


NI_DRAWS, NI_SEED = 2000, 102  # V-h5a4, V-h5b4: the record's runs drawn (plans/jepa.md)


@dataclass(frozen=True)
class SelectConf:
    """v3's grid of readouts and its choice (``plans/jepa.md``), by default
    the real data's: ``fidelity.fit_ridge`` of each ``alphas``, the MLP of
    each ``widths`` (``mlp`` otherwise) stopped at every ``every`` steps,
    of the ``seeds`` (a candidate's error the mean of its seeds'); v2's
    readout (a record) the ``v2_width`` at the last step; the record of the
    validation's gap to the runner-up of another kind drawn ``se_draws``
    times of ``se_seed``."""

    alphas: tuple[float, ...] = (0.0, 0.01, 0.1, 1.0)
    widths: tuple[int, ...] = (256, 1024)
    every: int = 1000
    mlp: MLPConf = MLPConf()
    seeds: tuple[int, ...] = (3001, 3002, 3003)
    v2_width: int = 1024
    se_draws: int = 2000
    se_seed: int = 103

    def order(self) -> list[str]:
        return candidate_names(self.alphas, self.widths, self.mlp.steps, self.every)

    @property
    def v2(self) -> str:
        return f"mlp-{self.v2_width}-{self.mlp.steps}"


@dataclass(frozen=True)
class RunSet:
    """Runs' middle halves: the codes ``x`` ``(R, L, D)`` and the
    observations ``y`` ``(R, L, K)`` of the same stacks."""

    x: Tensor
    y: Tensor

    def flat(self) -> tuple[Tensor, Tensor]:
        return self.x.flatten(0, 1), self.y.flatten(0, 1)

    def pairs(self) -> tuple[Tensor, Tensor]:
        r"""The pair inputs $(c_t, c_{t-1})$ and the changes $y_t - y_{t-1}$
        of every time ``(R L, 2D)``, ``(R L, K)`` (each run's first a dummy:
        no anchor reads it)."""
        prev = torch.cat([self.y[:, :1], self.y[:, :-1]], 1)
        return pair_inputs(self.x).flatten(0, 1), (self.y - prev).flatten(0, 1)

    def runs(self) -> Tensor:
        return torch.arange(len(self.x)).repeat_interleave(self.x.shape[1])

    def inner(self) -> Tensor:
        """The rows past each run's first (the anchors of $E^d$)."""
        return torch.arange(self.x.shape[0] * self.x.shape[1]) % self.x.shape[1] != 0


def _candidates(x: Tensor, y: Tensor, sc: SelectConf, dev: torch.device
                ) -> dict[str, list[Predictor]]:  # fmt: skip
    """Each candidate's predictors (one a seed) learned on ``x``, ``y``."""
    out = {f"lin-{a:g}": [linear_predictor(fit_ridge(x, y, a))] for a in sc.alphas}
    for h in sc.widths:
        conf = replace(sc.mlp, hidden=h)
        paths = [fit_mlp_path(x, y, conf, s, dev, sc.every) for s in sc.seeds]
        for i, k in enumerate(range(sc.every, conf.steps + 1, sc.every)):
            out[f"mlp-{h}-{k}"] = [snapshot_predictor(net, snaps[i], dev)
                                   for net, snaps in paths]  # fmt: skip
    return out


def _choice(cands: Mapping[str, Sequence[Predictor]], x: Tensor, y: Tensor, runs: Tensor,
            n_runs: int, order: Sequence[str], sc: SelectConf) -> dict[str, Any]:  # fmt: skip
    """The choice by the validation runs' error: ``selected``, each
    candidate's ``scores`` (the seeds' mean of the summed squared error),
    the best of another kind (``runner``) and its gap with the gap's
    standard deviation over draws of the runs (``gap_se``; a record)."""
    per = {n: torch.stack([run_errors(g(x), y, runs, n_runs) for g in fns]).mean(0)
           for n, fns in cands.items()}  # fmt: skip
    scores = {n: float(v.sum()) for n, v in per.items()}
    sel = select_candidate(scores, order)
    others = [n for n in order if kind_of(n) != kind_of(sel)]
    row: dict[str, Any] = {"selected": sel, "scores": scores, "runner": None}
    if others:
        run = select_candidate(scores, others)
        c = block_counts(n_runs, sc.se_draws, _gen(sc.se_seed))
        gap = c @ (per[run] - per[sel])
        row |= {
            "runner": run,
            "gap": scores[run] - scores[sel],
            "gap_se": float(gap.std()),
        }
    return row


def select_rows(
    labels: Mapping[str, Any],
    train: RunSet,
    val: RunSet,
    record: RunSet,
    evals: Sequence[tuple[Tensor, Tensor]],
    ws: WidthSets,
    sc: SelectConf,
    dev: torch.device,
    *,
    sets: Collection[str] = ("train", "val", "record", "eval"),
    readouts: Collection[str] = ("sel", "lin", "lin-diff", "v2"),
    components: Mapping[str, Sequence[int]] | None = None,
    ext_lags: Sequence[int] = (),
) -> list[Row]:
    r"""v3's rows of a code (``plans/jepa.md``): the level's readout chosen
    among ``sc``'s candidates learned on ``train``'s times by ``val``'s
    error, the change's among the same learned from the pairs $(c_t,
    c_{t-1})$ to $d_t$ (each run's first time left out) and the chosen
    level's difference (``diff``, the last in the ties) by ``val``'s error
    of $d$ (rows ``select``); of each of ``readouts`` (``sel`` the chosen
    two, ``lin`` the least squares of both, ``lin-diff`` the level's least
    squares and its difference, ``v2`` v2's MLP and its difference) the sums
    of $E$, $E^d$ (rows ``sums``) on each of ``sets``: the runs (anchors
    past each run's first, a block each run; $\bar y$ ``train``'s mean) and
    ``eval`` (the segments ``evals`` of codes and observations at the
    common centres of ``ws``, their blocks side by side); with
    ``components`` (a name to the observation's columns) the same of those
    columns (rows ``comp``); with ``ext_lags`` the extrapolation in the
    observation of the chosen level's seeds on ``evals[0]`` (``obs_ext``)."""
    d = train.x.shape[-1]
    xl, yl = train.flat()
    mean = yl.to(_F64).mean(0)
    order = sc.order()
    xv, yv = val.flat()
    rv, nv, iv = val.runs(), len(val.x), val.inner()
    level = _candidates(xl, yl, sc, dev)
    lc = _choice(level, xv, yv, rv, nv, order, sc)
    lsel = lc["selected"]
    level = {k: v for k, v in level.items() if k in (lsel, "lin-0", sc.v2)}
    xp, dp = train.pairs()
    inner = train.inner()
    change = _candidates(xp[inner], dp[inner], sc, dev)
    change["diff"] = [diff_predictor(g, d) for g in level[lsel]]
    xpv, dpv = val.pairs()
    cc = _choice(change, xpv[iv], dpv[iv], rv[iv], nv, [*order, "diff"], sc)
    csel = cc["selected"]
    lab = dict(labels)
    rows: list[Row] = [lab | {"part": "select", "target": "level"} | lc,
                       lab | {"part": "select", "target": "change"} | cc]  # fmt: skip
    both = {"sel": (lsel, level[lsel], csel, change[csel]),
            "lin": ("lin-0", level["lin-0"], "lin-0", change["lin-0"]),
            "lin-diff": ("lin-0", level["lin-0"], "diff",
                         [diff_predictor(g, d) for g in level["lin-0"]]),
            "v2": (sc.v2, level[sc.v2], "diff", [diff_predictor(g, d) for g in level[sc.v2]])}  # fmt: skip
    t, blocks = _anchors(ws)
    runsets = {"train": train, "val": val, "record": record}
    for name in readouts:
        ln, gl, cn, gc = both[name]
        head = lab | {"readout": name, "level": ln, "change": cn}
        for s in sets:
            if s == "eval":
                got = [_seg_sums(x, y, gl, gc, mean, t, blocks, ws.n_blocks, components)
                       for x, y in evals]  # fmt: skip
                whole = ContentSums.cat([g[0] for g in got])
                comps = {k: ContentSums.cat([g[1][k] for g in got]) for k in got[0][1]}
            else:
                whole, comps = _run_set_sums(runsets[s], gl, gc, mean, components)
            rows.append(head | {"part": "sums", "set": s} | whole.to_row())
            rows += [head | {"part": "comp", "set": s, "comp": k} | v.to_row()
                     for k, v in comps.items()]  # fmt: skip
    if ext_lags:
        x, y = evals[0]
        exts = [_obs_ext(g, x, y, t, mean, ext_lags) for g in level[lsel]]
        rows.append(
            lab | {"part": "obs_ext", "readout": "sel", "level": lsel, "lags": exts}
        )
    return rows


def _sums_of(
    y: Tensor,
    x: Tensor,
    xp: Tensor,
    gl: Sequence[Predictor],
    gc: Sequence[Predictor],
    mean: Tensor,
    t: Tensor,
    blocks: Tensor,
    nb: int,
    components: Mapping[str, Sequence[int]] | None,
) -> tuple[ContentSums, dict[str, ContentSums]]:
    """The sums of the level's predictors ``gl`` on ``x`` and the change's
    ``gc`` on the pairs ``xp``, one seed at a time (two predictions held at
    once): the seeds paired in order, a single one (a linear readout)
    repeated beside the other's (its rows the same, so its value and its
    seeds' spread are a single one's)."""
    n = max(len(gl), len(gc))
    parts: list[ContentSums] = []
    comp: dict[str, list[ContentSums]] = {k: [] for k in components or {}}
    for i in range(n):
        p, q = gl[i % len(gl)](x), gc[i % len(gc)](xp)
        parts.append(content_sums(y, [p], mean, t, blocks, nb, dpreds=[q]))
        for k, cols in (components or {}).items():
            c = torch.tensor(list(cols))
            comp[k].append(content_sums(y[:, c], [p[:, c]], mean[c], t, blocks, nb,
                                        dpreds=[q[:, c]]))  # fmt: skip
    return stack_seeds(parts), {k: stack_seeds(v) for k, v in comp.items()}


def _seg_sums(x: Tensor, y: Tensor, gl: Sequence[Predictor], gc: Sequence[Predictor],
              mean: Tensor, t: Tensor, blocks: Tensor, nb: int,
              components: Mapping[str, Sequence[int]] | None
              ) -> tuple[ContentSums, dict[str, ContentSums]]:  # fmt: skip
    return _sums_of(y, x, pair_inputs(x), gl, gc, mean, t, blocks, nb, components)


def _run_set_sums(rs: RunSet, gl: Sequence[Predictor], gc: Sequence[Predictor], mean: Tensor,
                  components: Mapping[str, Sequence[int]] | None
                  ) -> tuple[ContentSums, dict[str, ContentSums]]:  # fmt: skip
    x, y = rs.flat()
    xp, _ = rs.pairs()
    length = rs.x.shape[1]
    idx = torch.arange(len(y))
    t = idx[idx % length != 0]
    return _sums_of(y, x, xp, gl, gc, mean, t, t // length, len(rs.x), components)


def draw_starts(
    total: int,
    length: int,
    count: int,
    seed: int,
    avoid: Sequence[int] | None = None,
    tries: int | None = None,
) -> list[int]:
    """``width_guard.interval_draws``' rule on one series of ``total``
    times (its only cut the end): ``count`` starts drawn uniformly by a
    generator of ``seed``, one rejected where its run would pass the end
    and, with ``avoid`` (None: runs may overlap), where it would overlap one
    of ``avoid`` or one drawn before.

    Raises:
        ValueError: ``count`` starts are not found within ``tries`` draws.
    """
    gen = torch.Generator().manual_seed(seed)
    fixed = None if avoid is None else torch.tensor(list(avoid), dtype=torch.long)
    got: list[int] = []
    drawn = 0
    while len(got) < count:
        if tries is not None and drawn >= tries:
            raise ValueError(f"{len(got)} of {count} runs in {tries} draws")
        drawn += 1
        k = int(torch.randint(total, (1,), generator=gen))
        if fixed is not None and _clash(k, fixed, got, length):
            continue
        if k + length > total:
            continue
        got.append(k)
    return got


def _clash(k: int, fixed: Tensor, new: Sequence[int], length: int) -> bool:
    if len(fixed) and bool(((fixed - k).abs() < length).any()):
        return True
    return any(abs(k - j) < length for j in new)


@dataclass(frozen=True)
class Synth3Conf:
    r"""V-h5a's settings (``plans/jepa.md``; times in steps of 0.1 s): the
    evaluation's segments as V-h4a's; the training series of ``total``
    times and its runs (``run`` long, the middle halves used) drawn as the
    real data's (``draw_starts``); the parts' shares of the observation's
    variance (``parts3``) and the readouts' grid ``select``."""

    n: int = 38989
    span: int = 7
    period: float = 300.0  # 30 s
    fast_ks: tuple[int, int] = (4, 5)
    f_dims: int = 2
    f_tau: float = 30.0  # 3 s
    scene: int = 16  # s
    detail: int = 16  # n
    fast: int = 16  # g
    ell: float = 600.0  # 60 s
    g_tau: float = 3.0  # 0.3 s
    kept: float = 0.74
    core: float = 0.30
    eps: float = 0.01
    r: float = 0.113
    total: int = 479350
    run: int = 512
    train_count: int = 512
    train_seed: int = 1401
    record_count: int = 64
    record_seed: int = 3201
    val_count: int = 128
    val_seed: int = 3301
    gap: int = 256
    windows: tuple[int, ...] = (8, 32, 64, 128)
    long_seed: int = 2100
    long_parts: int = 10
    seeds: tuple[int, int] = (2101, 2301)
    cal_seeds: tuple[int, int] = (2301, 2501)
    select: SelectConf = SelectConf(widths=(64, 256), v2_width=256)
    rot_seed: int = 3101
    psi_star: float = 0.1
    delta_star: float = 0.5
    max_false: int = 13


def _rho1_smooth(sc: Synth3Conf) -> float:
    k = gauss_kernel(sc.ell)
    return float(kernel_acf(k, torch.tensor([1]))[0])


def parts3(sc: Synth3Conf, positive: bool) -> dict[str, tuple[float, float]]:
    r"""Each part's share of the observation's variance (the sum of its
    columns') and its change factor $2(1 - \rho_1)$ (a step's change's
    variance over the level's): the core (positive: the circles ``k1``,
    ``k4``, ``k5`` of ``core`` / 3 each, $2(1 - \cos 2\pi k / T)$;
    negative: ``circle`` and ``f``, an OU of ``f_tau``, $2(1 - e^{-1 /
    \tau})$, of ``core`` / 2 each), the scene ``s`` (``kept`` - ``core``)
    and its detail ``n`` (the squared exponential's $\rho_1$ of
    ``kernel_acf``), the fast content ``g`` (an OU of ``g_tau``) and the
    white ``eps``: ``g`` and ``n`` solved in closed form so that the shares
    sum to 1 and the change's to ``r`` (the change's variance over the
    level's)."""
    sm = 2 * (1 - _rho1_smooth(sc))
    if positive:
        core = {f"k{k}": (sc.core / 3, 2 * (1 - math.cos(2 * math.pi * k / sc.period)))
                for k in (1, *sc.fast_ks)}  # fmt: skip
    else:
        core = {"circle": (sc.core / 2, 2 * (1 - math.cos(2 * math.pi / sc.period))),
                "f": (sc.core / 2, 2 * (1 - math.exp(-1 / sc.f_tau)))}  # fmt: skip
    kg = 2 * (1 - math.exp(-1 / sc.g_tau))
    known = sum(s * f for s, f in core.values()) + (sc.kept - sc.core) * sm + 2 * sc.eps
    rest = 1 - sc.kept - sc.eps
    g = (sc.r - known - rest * sm) / (kg - sm)
    return core | {"s": (sc.kept - sc.core, sm), "n": (rest - g, sm), "g": (g, kg),
                   "eps": (sc.eps, 2.0)}  # fmt: skip


def columns3(sc: Synth3Conf, positive: bool) -> dict[str, list[int]]:
    """The observation's columns of the core, ``s``, ``n`` and ``g``."""
    c = 6 if positive else 2 + sc.f_dims
    out, at = {}, 0
    for name, w in (("core", c), ("s", sc.scene), ("n", sc.detail), ("g", sc.fast)):
        out[name] = list(range(at, at + w))
        at += w
    return out


def process3(
    n: int, sc: Synth3Conf, positive: bool, gen: torch.Generator
) -> tuple[Tensor, dict[str, Tensor]]:
    r"""A series of V-h5a (``plans/jepa.md``): the observation $y$ (the
    core's columns, then $s$, $n$, $g$, each part's columns of equal variance
    summing to its share of ``parts3``, plus the white noise) and the codes
    (positive: ``slow`` $(\sqrt2(\cos\theta, \sin\theta), s)$ and ``fast``
    $(\sqrt2(\cos 4\theta, \ldots, \sin 5\theta), s)$; negative: ``mixed``
    $(\sqrt2\cos\theta, \sqrt2\sin\theta, f, s)$), every code column of
    unit variance. $s$ and $n$ are ``smooth`` of ``gauss_kernel(ell)``
    ($\rho(\Delta) = e^{-\Delta^2 / (2\ell^2)}$), $f$ and $g$ ``ou``."""
    p = parts3(sc, positive)
    theta = _phase(n, sc.period, gen)
    root2 = math.sqrt(2)
    if positive:
        ks = (1, *sc.fast_ks)
        circ = root2 * _circle(theta, ks)
        core = torch.cat([math.sqrt(p[f"k{k}"][0] / 2) * circ[:, 2 * i : 2 * i + 2]
                          for i, k in enumerate(ks)], 1)  # fmt: skip
        codes = {"slow": circ[:, :2], "fast": circ[:, 2:]}
    else:
        circ = root2 * _circle(theta, (1,))
        f = ou(n, sc.f_dims, math.exp(-1 / sc.f_tau), gen)
        core = torch.cat([math.sqrt(p["circle"][0] / 2) * circ,
                          math.sqrt(p["f"][0] / sc.f_dims) * f], 1)  # fmt: skip
        codes = {"mixed": torch.cat([circ, f], 1)}
    kern = gauss_kernel(sc.ell)
    s = smooth(n, sc.scene, kern, gen)
    detail = smooth(n, sc.detail, kern, gen)
    g = ou(n, sc.fast, math.exp(-1 / sc.g_tau), gen)
    y = torch.cat([core, math.sqrt(p["s"][0] / sc.scene) * s,
                   math.sqrt(p["n"][0] / sc.detail) * detail,
                   math.sqrt(p["g"][0] / sc.fast) * g], 1)  # fmt: skip
    y = y + math.sqrt(p["eps"][0] / y.shape[1]) * white(n, y.shape[1], gen)
    return y, {k: torch.cat([v, s], 1) for k, v in codes.items()}


def cheats3(sc: Synth3Conf) -> dict[str, Callable[[Tensor], Tensor]]:
    """``mixed``'s cheats as functions of codes ``(..., L, D)``: ``drop``
    ($f$ set to its mean 0) and ``mean-<w>`` (the past mean along dim -2)."""
    fs = slice(2, 2 + sc.f_dims)

    def drop(x: Tensor) -> Tensor:
        out = x.clone()
        out[..., fs] = 0
        return out

    out: dict[str, Callable[[Tensor], Tensor]] = {"drop": drop}
    for w in sc.windows:
        out[f"mean-{w}"] = partial(_past_mean, width=w)
    return out


@cache
def starts3(sc: Synth3Conf) -> dict[str, list[int]]:
    """The runs' starts of V-h5a's training series (``plans/jepa.md``):
    ``train`` (may overlap) and ``record`` (overlapping neither) of
    ``draw_starts``, ``val`` (at least ``gap`` from both, overlapping no
    other) of ``width_guard.slot_starts``, as the real data's."""
    tr = draw_starts(sc.total, sc.run, sc.train_count, sc.train_seed)
    rec = draw_starts(sc.total, sc.run, sc.record_count, sc.record_seed, tr,
                      1000 * sc.record_count)  # fmt: skip
    ok = torch.arange(sc.total) + sc.run <= sc.total
    for j in tr + rec:
        ok[max(0, j - sc.run - sc.gap + 1) : j + sc.run + sc.gap] = False
    val = slot_starts(ok, sc.run, sc.val_count, sc.val_seed)
    return {"train": tr, "record": rec, "val": val}


def _runs3(full: Tensor, starts: Sequence[int], sc: Synth3Conf,
           fn: Callable[[Tensor], Tensor] | None = None) -> Tensor:  # fmt: skip
    """The runs of ``full`` at ``starts`` ``(R, run / 2, ·)``: ``fn`` applied
    to each whole run, then its middle half."""
    a = sc.run // 4
    runs = torch.stack([full[k : k + sc.run] for k in starts])
    if fn is not None:
        runs = fn(runs)
    return runs[:, a : sc.run - a]


def _code3(
    sc: Synth3Conf,
    ws: WidthSets,
    y_full: Tensor,
    x_full: Tensor,
    fn: Callable[[Tensor], Tensor] | None,
    segs: Sequence[tuple[Tensor, Tensor]],
    dev: torch.device,
    long: bool,
    components: Mapping[str, Sequence[int]] | None = None,
) -> list[Row]:
    """``select_rows`` of a code: its training series ``x_full`` (``fn``
    applied to each run) and the segments ``segs`` (codes, observations;
    ``fn`` applied from each segment's start). A realization of the long
    unit keeps every readout and set, another ``sel`` on ``eval``."""
    st = starts3(sc)
    rs = {k: RunSet(_runs3(x_full, st[k], sc, fn), _runs3(y_full, st[k], sc))
          for k in ("train", "val", "record")}  # fmt: skip
    evals = [(x if fn is None else fn(x), y) for x, y in segs]
    kw: dict[str, Any] = {} if long else {"sets": ("eval",), "readouts": ("sel",)}
    return select_rows({}, rs["train"], rs["val"], rs["record"], evals, ws, sc.select, dev,
                       components=components, **kw)  # fmt: skip


def _a1_3(sc: Synth3Conf, ws: WidthSets, gconf: GeomConf, wconf: WidthConf, seed: int,
          parts: int, dev: torch.device) -> dict[str, Any]:  # fmt: skip
    """A positive realization (the long one with ``slow`` rotated)."""
    gen = _gen(seed)
    segs = [process3(sc.n, sc, True, gen) for _ in range(parts)]
    y_tr, full = process3(sc.total, sc, True, gen)
    xs = {k: [c[k] for _, c in segs] for k in ("slow", "fast")}
    if parts > 1:
        q = orthogonal(full["slow"].shape[1], sc.rot_seed)
        full["slow-rot"] = full["slow"] @ q
        xs["slow-rot"] = [x @ q for x in xs["slow"]]
    ys = [y for y, _ in segs]
    content = {k: _code3(sc, ws, y_tr, full[k], None, list(zip(v, ys, strict=True)), dev,
                         parts > 1) for k, v in xs.items()}  # fmt: skip
    ls = {k: _widths(xs[k], ws, wconf) for k in ("slow", "fast")}
    return {"seed": seed, "width": _pair(ls["slow"], ls["fast"], wconf, gconf),
            "mean1": {k: _mean1(v, wconf) for k, v in ls.items()}, "content": content}  # fmt: skip


def _noisy3(x: Tensor, sigma2: float, sc: Synth3Conf, gen: torch.Generator) -> Tensor:
    """``x`` with independent Gaussian noise of ``sigma2`` on ``f``."""
    out = x.clone()
    fs = slice(2, 2 + sc.f_dims)
    out[:, fs] += math.sqrt(sigma2) * torch.randn(len(x), sc.f_dims, generator=gen,
                                                  dtype=_F64)  # fmt: skip
    return out


def _a2_3(sc: Synth3Conf, ws: WidthSets, gconf: GeomConf, wconf: WidthConf, seed: int,
          parts: int, dev: torch.device) -> dict[str, Any]:  # fmt: skip
    """A negative realization: ``mixed`` and its cheats (the long one also
    ``mixed`` rotated and the calibration's ``noisy``, ``noisy-d``, whose
    noises are drawn last)."""
    gen = _gen(seed)
    segs = [process3(sc.n, sc, False, gen) for _ in range(parts)]
    y_tr, full = process3(sc.total, sc, False, gen)
    ys = [y for y, _ in segs]
    mixed = [c["mixed"] for _, c in segs]
    codes: dict[str, tuple[Tensor, Callable[[Tensor], Tensor] | None, list[Tensor]]] = {
        "mixed": (full["mixed"], None, mixed)
    }
    for k, fn in cheats3(sc).items():
        codes[k] = (full["mixed"], fn, mixed)
    long = parts > 1
    if long:
        q = orthogonal(full["mixed"].shape[1], sc.rot_seed)
        codes["mixed-rot"] = (full["mixed"] @ q, None, [x @ q for x in mixed])
        for k, s2 in (("noisy", calib3(sc)), ("noisy-d", calib3_d(sc))):
            codes[k] = (_noisy3(full["mixed"], s2, sc, gen), None,
                        [_noisy3(x, s2, sc, gen) for x in mixed])  # fmt: skip
    comps = columns3(sc, False)
    content = {k: _code3(sc, ws, y_tr, xf, fn, list(zip(xs, ys, strict=True)), dev, long,
                         comps if long and k == "mixed" else None)
               for k, (xf, fn, xs) in codes.items()}  # fmt: skip
    fns = cheats3(sc)
    cheats = list(fns)
    ls = {k: _widths([fns[k](x) if k in fns else x for x in mixed], ws, wconf)
          for k in ["mixed", *cheats]}  # fmt: skip
    return {"seed": seed,
            "width": {k: _pair(ls[k], ls["mixed"], wconf, gconf) for k in cheats},
            "mean1": {k: _mean1(v, wconf) for k, v in ls.items()}, "content": content}  # fmt: skip


def _a3_3(
    sc: Synth3Conf, ws: WidthSets, seed: int, dev: torch.device
) -> dict[str, Any]:
    """A calibration's realization: ``mixed``, ``noisy`` ($\\psi = -\\psi_*$)
    and ``noisy-d`` ($\\psi^d = -\\psi_*$)."""
    gen = _gen(seed)
    y, c = process3(sc.n, sc, False, gen)
    y_tr, full = process3(sc.total, sc, False, gen)
    m, mf = c["mixed"], full["mixed"]
    codes = {"mixed": (mf, m)}
    for k, s2 in (("noisy", calib3(sc)), ("noisy-d", calib3_d(sc))):
        codes[k] = (_noisy3(mf, s2, sc, gen), _noisy3(m, s2, sc, gen))
    content = {k: _code3(sc, ws, y_tr, xf, None, [(x, y)], dev, False)
               for k, (xf, x) in codes.items()}  # fmt: skip
    return {"seed": seed, "content": content}


def change_parts3(sc: Synth3Conf, positive: bool) -> dict[str, float]:
    """Each part's share of the change's variance (``parts3``: share times
    factor; they sum to ``r``)."""
    return {k: s * f for k, (s, f) in parts3(sc, positive).items()}


def calib3(sc: Synth3Conf) -> float:
    r"""The noise $\sigma^2$ on ``mixed``'s $f$ of $\psi = -\psi_*$ in the
    population: $E[f \mid u] = u / (1 + \sigma^2)$ ($u = f + $ noise)
    loses $f$'s share times $\sigma^2 / (1 + \sigma^2)$ of ``kept``, so
    $\sigma^2 / (1 + \sigma^2) = \text{kept}(1 - 2^{-\psi_*}) / f$."""
    x = sc.kept * (1 - 2**-sc.psi_star) / parts3(sc, False)["f"][0]
    return x / (1 - x)


def pair_ed3(sc: Synth3Conf, sigma2: float) -> float:
    r"""The population's $E^d$ of the best pair readout of ``mixed`` with
    noise $\sigma^2$ on $f$: of $f$'s change $d^f = f_t - f_{t-1}$ (on the
    $(1, -1)$ direction, where $u$'s variance is $2(1 + \sigma^2 - \phi)$)
    the share $(1 - \phi) / (1 + \sigma^2 - \phi)$ explained; the circle's
    and $s$'s change wholly, $n$'s, $g$'s and the noise's not at all."""
    c = change_parts3(sc, False)
    phi = math.exp(-1 / sc.f_tau)
    q = (1 - phi) / (1 + sigma2 - phi)
    return (c["circle"] + c["s"] + c["f"] * q) / sum(c.values())


def diff_ed3(sc: Synth3Conf, sigma2: float) -> float:
    r"""The population's $E^d$ of the best level readout's difference of
    ``mixed`` with noise $\sigma^2$ on $f$: $a(u_t - u_{t-1})$, $a = 1 /
    (1 + \sigma^2)$, errs on $f$'s change by the share $[(1 - a)^2(1 - \phi)
    + a^2 \sigma^2] / (1 - \phi)$ of it; the rest as ``pair_ed3``."""
    c = change_parts3(sc, False)
    phi = math.exp(-1 / sc.f_tau)
    a = 1 / (1 + sigma2)
    err = ((1 - a) ** 2 * (1 - phi) + a * a * sigma2) / (1 - phi)
    return (c["circle"] + c["s"] + c["f"] * (1 - err)) / sum(c.values())


def calib3_d(sc: Synth3Conf) -> float:
    r"""The noise $\sigma^2$ on ``mixed``'s $f$ of $\psi^d = -\psi_*$ in
    the population (``pair_ed3``): $q = 1 - (1 - 2^{-\psi_*}) E^d_0 / f$
    ($E^d_0$ the change explained without noise, $f$ its share of $f$),
    $\sigma^2 = (1 - \phi) / q - (1 - \phi)$."""
    c = change_parts3(sc, False)
    phi = math.exp(-1 / sc.f_tau)
    q = 1 - (1 - 2**-sc.psi_star) * (c["circle"] + c["s"] + c["f"]) / c["f"]
    return (1 - phi) / q - (1 - phi)


LONG3 = ("V-h5a1-long", "V-h5a2-long")
SEEDED3 = ("V-h5a1", "V-h5a2", "V-h5a3")


def check_units3() -> tuple[str, ...]:
    return LONG3 + SEEDED3


def _seeds_of3(unit: str, sc: Synth3Conf) -> range:
    return range(*(sc.cal_seeds if unit == "V-h5a3" else sc.seeds))


def check_parts3(
    sc: Synth3Conf,
    gconf: GeomConf,
    wconf: WidthConf,
    units: Collection[str],
    seeds: tuple[int, int] | None = None,
    device: str = "cpu",
) -> list[Row]:
    """V-h5a's rows of ``units`` (``check_units3``) for ``merge_checks3``,
    as ``check_parts``.

    Raises:
        ValueError: A unit is not one of ``check_units3``.
    """
    bad = [u for u in units if u not in check_units3()]
    if bad:
        raise ValueError(f"no check units {bad} (of {list(check_units3())})")
    ws = synthetic_sets(sc.n, sc.span, gconf.block, wconf.max_lag)
    dev = torch.device(device)
    lo, hi = (-math.inf, math.inf) if seeds is None else seeds
    rows: list[Row] = []
    for u in check_units3():
        if u not in units:
            continue
        if u in LONG3:
            fn = _a1_3 if u == "V-h5a1-long" else _a2_3
            v = fn(sc, ws, gconf, wconf, sc.long_seed, sc.long_parts, dev)
            rows.append({"part": "long", "unit": u, "value": v})
            continue
        for s in _seeds_of3(u, sc):
            if not lo <= s < hi:
                continue
            if u == "V-h5a1":
                v = _a1_3(sc, ws, gconf, wconf, s, 1, dev)
            elif u == "V-h5a2":
                v = _a2_3(sc, ws, gconf, wconf, s, 1, dev)
            else:
                v = _a3_3(sc, ws, s, dev)
            rows.append({"part": "draw", "unit": u, "value": v})
    return rows


def sums3(rows: Sequence[Row], readout: str = "sel", set_: str = "eval",
          comp: str | None = None) -> ContentSums:  # fmt: skip
    """The sums of ``readout`` on ``set_`` (of the component ``comp``) of
    a code's ``select_rows``.

    Raises:
        KeyError: No such row.
    """
    for r in rows:
        if r.get("readout") == readout and r.get("set") == set_ and r.get("comp") == comp \
                and r["part"] in ("sums", "comp"):  # fmt: skip
            return ContentSums.of_row(r)
    raise KeyError((readout, set_, comp))


def chosen3(rows: Sequence[Row]) -> dict[str, str]:
    """The chosen candidate of the level and of the change."""
    return {r["target"]: r["selected"] for r in rows if r["part"] == "select"}


def pair3(
    a: ContentSums,
    b: ContentSums,
    counts: Tensor,
    counts2: Tensor,
    sig_rot: tuple[float, float],
    psi_star: float,
) -> dict[str, Any]:
    r"""v3's pair $(a, b)$: $\sigma = \max(\sigma_{\mathrm{rot}}, s_{ab})$ of
    $D$ and of $D^d$ (``fidelity.contrast_seed_sd``; ``s_ab`` recorded), the intervals of $D$,
    $D^d$, whether $b$'s $E$, $E^d$ are comparable and $\psi$, $\psi^d$'s
    points (a display)."""
    own = (
        contrast_seed_sd(a, b, psi_star),
        contrast_seed_sd(a, b, psi_star, change=True),
    )
    sig = (max(sig_rot[0], own[0]), max(sig_rot[1], own[1]))
    d = contrast_interval(a, b, counts, counts2, sig[0], psi_star)
    dd = contrast_interval(a, b, counts, counts2, sig[1], psi_star, change=True)
    comp = [comparable(b, counts, counts2), comparable(b, counts, counts2, change=True)]
    psi = [
        ratio_interval(a, b, counts, counts2, 0.0, change=ch)["point"]
        for ch in (False, True)
    ]
    return {"D": d, "Dd": dd, "sigma": list(sig), "s_ab": list(own), "comparable": comp,
            "ok": all(bool(c["ok"]) for c in comp), "psi": psi,
            "E": [a.value(), b.value()], "Ed": [a.value(True), b.value(True)]}  # fmt: skip


def front3(
    width: tuple[float | None, float | None],
    a: ContentSums,
    b: ContentSums,
    counts: Tensor,
    counts2: Tensor,
    sig_rot: tuple[float, float],
    delta_star: float,
    psi_star: float,
) -> dict[str, Any]:
    """v3's front verdict (``fidelity.contrast_verdict``) of ``pair3``."""
    p = pair3(a, b, counts, counts2, sig_rot, psi_star)
    return p | {"width": list(width),
                "verdict": contrast_verdict(width, p["D"], p["Dd"], p["ok"], delta_star)}  # fmt: skip


def _se_change(s: ContentSums, counts: Tensor, counts2: Tensor) -> float:
    return max(
        float(s.draws(counts, True).std()), float(s.paired().draws(counts2, True).std())
    )


def merge_checks3(
    parts: Sequence[Row], sc: Synth3Conf, gconf: GeomConf = GeomConf()
) -> list[Row]:
    """V-h5a's rows (``plans/jepa.md``) of the parts' rows (any split, any
    order): each check (``part`` ``check``; a failure stops) and the
    records.

    Raises:
        ValueError: A unit or a realization is missing or given twice.
    """
    long: dict[str, Any] = {}
    got: dict[str, dict[int, Any]] = {u: {} for u in SEEDED3}
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
    if len(_seeds_of3("V-h5a3", sc)) != len(_seeds_of3("V-h5a2", sc)):
        raise ValueError("V-h5a3's power pairs each realization with V-h5a2's")
    missing = [u for u in LONG3 if u not in long]
    missing += [
        f"{u} {s}" for u in SEEDED3 for s in _seeds_of3(u, sc) if s not in got[u]
    ]
    if missing:
        raise ValueError(f"missing {missing}")
    psi, delta = sc.psi_star, sc.delta_star
    l1, l2 = long["V-h5a1-long"], long["V-h5a2-long"]

    def s_(v: Mapping[str, Any], name: str, readout: str = "sel", set_: str = "eval",
           comp: str | None = None) -> ContentSums:  # fmt: skip
        return sums3(v["content"][name], readout, set_, comp)

    def sig(v: Mapping[str, Any], a: str, b: str) -> tuple[float, float]:
        return (contrast_seed_sd(s_(v, a), s_(v, b), psi),
                contrast_seed_sd(s_(v, a), s_(v, b), psi, change=True))  # fmt: skip

    sig_pos, sig_neg = sig(l1, "slow-rot", "slow"), sig(l2, "mixed-rot", "mixed")
    nb_long = len(s_(l1, "slow").tot)
    nb = len(s_(next(iter(got["V-h5a1"].values())), "slow").tot)
    c_long = (counts_of(nb_long, gconf), counts_of(nb_long // 2, gconf))
    c_one = (counts_of(nb, gconf), counts_of(nb // 2, gconf))
    s2, s2d = calib3(sc), calib3_d(sc)
    rows: list[Row] = [_record("V-h5a-sigma", positive=list(sig_pos), negative=list(sig_neg),
                               calib_sigma2=s2, calib_sigma2_d=s2d,
                               diff_ed=diff_ed3(sc, s2), pair_ed=pair_ed3(sc, s2))]  # fmt: skip
    # V-h5a0: v2's rule fails on the synthetic by memorizing the runs
    mx = l2["content"]["mixed"]
    e_v2, e_lin = sums3(mx, "v2", "val").value(), sums3(mx, "lin", "val").value()
    n_tr = sums3(mx, "v2", "train", "n").value()
    n_val = sums3(mx, "v2", "val", "n").value()
    comps = {f"{ro}/{st}/{k}": [sums3(mx, ro, st, k).value(), sums3(mx, ro, st, k).value(True)]
             for ro in ("sel", "lin", "v2") for st in ("train", "val", "record", "eval")
             for k in columns3(sc, False)}  # fmt: skip
    rows.append(_check("V-h5a0", e_v2 < e_lin and n_tr > 0 > n_val, E_val_v2=e_v2,
                       E_val_lin=e_lin, E_n_train=n_tr, E_n_val=n_val, components=comps))  # fmt: skip
    # V-h5a0-d: the mechanism of the difference's error, by closed forms
    nz = l2["content"]["noisy"]
    got_d = {}
    for ro, want in (("lin-diff", diff_ed3(sc, s2)), ("lin", pair_ed3(sc, s2))):
        s = sums3(nz, ro, "eval")
        se = _se_change(s, *c_long)
        got_d[ro] = {"Ed": s.value(True), "want": want, "se": se,
                     "ok": abs(s.value(True) - want) <= 2 * se}  # fmt: skip
    rows.append(_check("V-h5a0-d", all(v["ok"] for v in got_d.values()), values=got_d))
    # V-h5a4: the chosen readout not worse than the least squares; comparable
    rec = block_counts(sc.record_count, NI_DRAWS, _gen(NI_SEED))
    ni: dict[str, Any] = {}
    for v, b in ((l1, "fast"), (l2, "mixed")):
        lin_b = s_(v, b, "lin", "record")
        for name, rws in v["content"].items():
            sel, lin = sums3(rws, "sel", "record"), sums3(rws, "lin", "record")
            ni[name] = [noninferior(sel, lin, (1 - 2**-psi) * lin_b.value(ch), rec, change=ch)
                        for ch in (False, True)]  # fmt: skip
    comp4 = {n: [comparable(s_(v, n), *c_long, change=ch) for ch in (False, True)]
             for v, n in ((l1, "slow"), (l1, "fast"), (l2, "mixed"))}  # fmt: skip
    ok4 = all(bool(x["ok"]) for v in ni.values() for x in v) and \
        all(bool(x["ok"]) for v in comp4.values() for x in v)  # fmt: skip
    rows.append(_check("V-h5a4", ok4, noninferior=ni, comparable=comp4))
    # V-h5a1: the positive control
    f1 = front3(tuple(l1["width"]["mean1"]), s_(l1, "slow"), s_(l1, "fast"), *c_long, sig_pos,
                delta, psi)  # fmt: skip
    rows.append(_check("V-h5a1", f1["verdict"] == "kept", front=f1, mean1=l1["mean1"]))
    kept1 = [front3(tuple(v["width"]["mean1"]), s_(v, "slow"), s_(v, "fast"), *c_one, sig_pos,
                    delta, psi)["verdict"] == "kept" for _, v in sorted(got["V-h5a1"].items())]  # fmt: skip
    rows.append(_record("V-h5a1-share", kept=sum(kept1), n=len(kept1)))
    # V-h5a2: the negative control
    fronts = {k: front3(tuple(w["mean1"]), s_(l2, k), s_(l2, "mixed"), *c_long, sig_neg, delta,
                        psi) for k, w in l2["width"].items()}  # fmt: skip
    claim = [k for k, f in fronts.items() if f["verdict"] != "width"]
    ok2 = bool(claim) and all(fronts[k]["verdict"] == "dropped" for k in claim)
    rows.append(_check("V-h5a2", ok2, claim=claim, fronts=fronts, mean1=l2["mean1"]))
    false2 = []
    for s, v in sorted(got["V-h5a2"].items()):
        verdicts = {k: front3(tuple(w["mean1"]), s_(v, k), s_(v, "mixed"), *c_one, sig_neg,
                              delta, psi)["verdict"] for k, w in v["width"].items()}  # fmt: skip
        if "kept" in verdicts.values():
            false2.append({"seed": s, "verdicts": verdicts})
    rows.append(_check("V-h5a2-count", len(false2) <= sc.max_false, kept=len(false2),
                       n=len(got["V-h5a2"]), which=false2))  # fmt: skip
    # V-h5a3, V-h5a3-d: the margins' calibration; the power; the choices
    shift = sc.cal_seeds[0] - sc.seeds[0]
    n3 = {"kept": 0, "kept_both": 0, "kept_d": 0, "power": 0, "power_both": 0}
    freq: dict[str, dict[str, int]] = {}
    for s, v in sorted(got["V-h5a3"].items()):
        p = pair3(s_(v, "noisy"), s_(v, "mixed"), *c_one, sig_neg, psi)
        n3["kept"] += p["ok"] and p["D"]["lo"] > 0
        n3["kept_both"] += p["ok"] and p["D"]["lo"] > 0 and p["Dd"]["lo"] > 0
        pd = pair3(s_(v, "noisy-d"), s_(v, "mixed"), *c_one, sig_neg, psi)
        n3["kept_d"] += pd["ok"] and pd["Dd"]["lo"] > 0
        pw = pair3(
            s_(v, "mixed"), s_(got["V-h5a2"][s - shift], "mixed"), *c_one, sig_neg, psi
        )
        n3["power"] += pw["ok"] and pw["D"]["lo"] > 0
        n3["power_both"] += pw["ok"] and pw["D"]["lo"] > 0 and pw["Dd"]["lo"] > 0
        for name, rws in v["content"].items():
            for target, cand in chosen3(rws).items():
                key = f"{name}/{target}/{cand}"
                freq.setdefault(name, {})
                freq[name][key] = freq[name].get(key, 0) + 1
    n = len(got["V-h5a3"])
    rows.append(_check("V-h5a3", n3["kept"] <= sc.max_false, kept=n3["kept"], n=n,
                       kept_both=n3["kept_both"]))  # fmt: skip
    rows.append(
        _check("V-h5a3-d", n3["kept_d"] <= sc.max_false, kept=n3["kept_d"], n=n)
    )
    rows.append(
        _record("V-h5a3-power", kept=n3["power"], kept_both=n3["power_both"], n=n)
    )
    rows.append(_record("V-h5a3-chosen", counts=freq))
    rows.append(_record("V-h5a-long-chosen", chosen={
        name: chosen3(rws) for v in (l1, l2) for name, rws in v["content"].items()}))  # fmt: skip
    return rows


def _check_config3(sc: Synth3Conf, gconf: GeomConf, wconf: WidthConf) -> dict[str, Any]:
    return {"synth3": asdict(sc), "geom": asdict(gconf), "width": asdict(wconf)}


def evaluate_checks3(
    out: Path,
    sc: Synth3Conf = Synth3Conf(),
    gconf: GeomConf = GeomConf(),
    wconf: WidthConf = WidthConf(),
    *,
    units: Collection[str] | None = None,
    seeds: tuple[int, int] | None = None,
    device: str = "cpu",
) -> Path:
    """V-h5a's parts into ``out`` (``config.json``, ``rows.json``); with
    neither ``units`` nor ``seeds`` the whole, merged."""
    determinism = configure_determinism(device)
    _start(out, _check_config3(sc, gconf, wconf) | {"device": device}, determinism)
    parts = check_parts3(sc, gconf, wconf, check_units3() if units is None else units,
                         seeds, device)  # fmt: skip
    rows = merge_checks3(parts, sc, gconf) if units is None and seeds is None else parts
    _write(out / "rows.json", rows, {})
    return out


def merge_checks3_into(
    out: Path,
    parts: Sequence[Path],
    sc: Synth3Conf = Synth3Conf(),
    gconf: GeomConf = GeomConf(),
    wconf: WidthConf = WidthConf(),
) -> Path:
    """V-h5a's ``rows.json`` in ``out`` of the parts' directories.

    Raises:
        ValueError: A part is of other settings, or a unit or a realization
            is missing or given twice.
    """
    want = json.loads(json.dumps(_check_config3(sc, gconf, wconf)))
    rows: list[Row] = []
    devices = set()
    for d in parts:
        conf = json.loads((d / "config.json").read_text())
        if {k: conf.get(k) for k in want} != want:
            raise ValueError(f"{d} is of other settings")
        devices.add(conf.get("device"))
        rows += json.loads((d / "rows.json").read_text())
    _start(out, _check_config3(sc, gconf, wconf) | {"devices": sorted(map(str, devices))},
           configure_determinism("cpu"))  # fmt: skip
    _write(out / "rows.json", merge_checks3(rows, sc, gconf), {})
    return out


# -- v3 on the real data (V-h5b) -----------------------------------------------------------


@dataclass(frozen=True)
class ContentConf3:
    """V-h5b's settings (``plans/jepa.md``), by default the plan's: the
    observation, the readouts' grid, ``rot``'s seed, the extrapolation's
    lags (a record) and the runs of the training videos beside the
    training set: the record's (overlapping none of the training runs nor
    each other) and the validation's (at least ``val_gap`` stacks from
    both, overlapping no other)."""

    obs: int = 32
    select: SelectConf = SelectConf()
    rot_seed: int = 3101
    ext_lags: tuple[int, ...] = (8, 16, 32, 64)
    record_seed: int = 3201
    record_count: int = 64
    val_seed: int = 3301
    val_count: int = 128
    val_gap: int = 256


def evaluate_content3(
    conf: Any,
    sets_dir: Path,
    out: Path,
    *,
    codes: Collection[str] | None = None,
    inits: Mapping[str, Path] | None = None,
    init_v16: Path | None = None,
    root: Path | None = None,
    guard: GuardConf = GuardConf(),
    cconf: ContentConf3 = ContentConf3(),
    gconf: GeomConf = GeomConf(),
    wconf: WidthConf = WidthConf(),
    fconf: FillConf = FillConf(),
    spacing: Mapping[str, Any] | None = None,
    span: float | None = None,
) -> Path:
    """V-h5b's rows (``select_rows``; the codes of ``content_names``, None
    all) into ``out / <code> / content3.json`` and the runs' starts into
    ``out / intervals.json`` (the manifest), each file the same to the byte
    whichever codes run beside. Returns ``out``.

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
    _start(out, {"guard": asdict(guard), "content3": asdict(cconf), "key": ws.key},
           determinism)  # fmt: skip
    pca = _yard_code("pca", conf, ref, None, guard, gconf, dev)
    yard = {n: _yard_white(n, conf, ref, inits, init_v16, guard, gconf, dev)
            for n in todo if n in YARD}  # fmt: skip
    mus = {"pca": pca.mu} | {n: c.mu for n, c in yard.items()}
    train, frames, starts = interval_codes(conf, root, guard, mus, dev)
    rec, r_frames, r_starts = interval_codes(
        conf, root, guard, mus, dev,
        holdout=(cconf.record_seed, cconf.record_count, starts))  # fmt: skip
    val, v_frames, v_starts = interval_codes(
        conf, root, guard, mus, dev,
        holdout=(cconf.val_seed, cconf.val_count, starts + r_starts), gap=cconf.val_gap)  # fmt: skip
    _write(out / "intervals.json", [{"set": k, "starts": v} for k, v in
                                    (("train", starts), ("record", r_starts), ("val", v_starts))],
           {"key": ws.key, "length": guard.cheat_length})  # fmt: skip
    a, length = guard.cheat_length // 4, guard.cheat_length
    half = length - 2 * a

    def obs_runs(f: Tensor) -> Tensor:
        y = pool_obs(f, cconf.obs)
        return y.reshape(-1, half, y.shape[-1])

    ys = {
        "train": obs_runs(frames),
        "record": obs_runs(r_frames),
        "val": obs_runs(v_frames),
    }
    y_eval = pool_obs(zurich.frames[ws.span : ws.span + ws.n], cconf.obs)
    w_eval = _encode(pca, zurich, ws.n, dev)
    xs = {"train": train, "record": rec, "val": val}
    s = lag_speeds(train["pca"], guard.cheat_lag)
    fns = dict(cheat_codes(w_eval, s, guard))
    q = orthogonal(guard.k, cconf.rot_seed)
    fns["rot"] = lambda x: x @ q
    for name in todo:
        if name in fns:
            x_eval = fns[name](w_eval)
            runs = {k: fns[name](v["pca"]) for k, v in xs.items()}
        else:
            x_eval = _encode(yard[name], zurich, ws.n, dev)
            runs = {k: v[name] for k, v in xs.items()}
        rs = {k: RunSet(v[:, a : length - a], ys[k]) for k, v in runs.items()}
        rows = select_rows({"code": name, "obs": "pix"}, rs["train"], rs["val"], rs["record"],
                           [(x_eval, y_eval)], ws, cconf.select, dev,
                           ext_lags=cconf.ext_lags)  # fmt: skip
        _write(out / name / "content3.json", rows, {"key": ws.key})
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
    checks3 = sub.add_parser("checks3")
    checks3.add_argument("--out", type=Path, required=True)
    checks3.add_argument(
        "--only", type=_names, default=None, help="units, comma-separated"
    )
    checks3.add_argument(
        "--seeds", type=_range, default=None, help="realizations lo:hi"
    )
    checks3.add_argument("--device", default="cpu")
    merge3 = sub.add_parser("merge-checks3")
    merge3.add_argument("--out", type=Path, required=True)
    merge3.add_argument("parts", type=Path, nargs="+")
    real3 = sub.add_parser("real3")
    for arg in ("--config", "--sets", "--out"):
        real3.add_argument(arg, type=Path, required=True)
    real3.add_argument("--codes", type=_names, default=None, help="comma-separated")
    real3.add_argument("--device", default=None)
    real3.add_argument("--root", type=Path, default=None)
    for name in (*YARD_INITS, "init_v16"):
        real3.add_argument(f"--{name.replace('_', '-')}", type=Path, default=None)
    a = p.parse_args(argv)
    if a.job == "checks":
        return evaluate_checks(a.out, units=a.only, seeds=a.seeds, device=a.device)
    if a.job == "merge-checks":
        return merge_checks_into(a.out, a.parts)
    if a.job == "checks3":
        return evaluate_checks3(a.out, units=a.only, seeds=a.seeds, device=a.device)
    if a.job == "merge-checks3":
        return merge_checks3_into(a.out, a.parts)
    conf = _conf_of(a.config, a.device)
    inits = {n: getattr(a, n) for n in YARD_INITS if getattr(a, n) is not None}
    if a.job == "real3":
        return evaluate_content3(conf, a.sets, a.out, codes=a.codes, inits=inits,
                                 init_v16=a.init_v16, root=a.root)  # fmt: skip
    return evaluate_content(conf, a.sets, a.out, codes=a.codes, inits=inits,
                            init_v16=a.init_v16, root=a.root, local=a.local,
                            inception=a.inception)  # fmt: skip


if __name__ == "__main__":
    main()
