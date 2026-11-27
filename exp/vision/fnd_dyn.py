r"""The foundation's main evaluation (``plans/foundation.md``, 4.1; F0): the
future and the past of a code by a minGRU that reads the history
(``exp.lib.gru_dyn``), against the copy and the mean, with the two-point
linear of dyn/copy beside as the lower bound. The decisions are those of
2026-11-27 (``results/foundation/README.md``); the verdicts are
``results/foundation/judge_fnd.py``'s.

**The sequences.** The stacks that do not overlap: of a video of $F$
frames the stacks of ``data.tours.frames`` frames ``stride`` apart whose
first frames are $0, S, 2S, ..$ ($S$ = frames x stride: 0.8 s at $s = 1$),
every one whose last frame is in the video (``stack_starts``). A code's
sequence is its codes of them, whitened by its reference (``midpoint.
Whitener`` of ``GeomConf.k`` = 64 components fit on ``fill_eval``'s
reference stacks; ``Whitener.white``). The training videos (all but Zurich)
are split as dyn/copy's (``dyn_eval.split_videos``, seed 4101): one the
record (``rec``), one the validation (``val``: the early stopping), the
rest the training set (``train``). Zurich is the evaluation (``zurich``).
The anchors are ``gru_dyn.seq_set``'s of the margin $\max(W - 1,
\Delta_{\max})$, on Zurich only those with no cut (``geom_eval``'s, through
``WidthSets.before``) between the first frame of the stack $t -
\Delta_{\max}$ and the last of $t + \Delta_{\max}$ (``cut_keep``); the
blocks are 128 s (160 steps), and on Zurich also those shifted by 64 s.

**The codes** (``code_fn``): ``pca`` (the PCA of the stacks' pixels,
``jepa_eval.PCACode`` of ``conf.seed + PCA_SEED``), ``mean8`` (the PCA of
the stack's mean image), a VAE's $\mu$ (``vae``: ``v16``, ``v17``), and any
other code through a factory ``module:function`` called as ``function(arg,
device)`` that returns the map from stacks u8 ``(B, n, 3, S, S)`` on the
device to codes ``(B, D)`` (``places``: the JEPA's $z \in \mathbb{R}^{4096}$
and its step 0, once they exist). ``rot`` is ``pca``'s whitened codes
times a random orthogonal matrix (``ContentConf.rot_seed``): the spread of
the minGRU's training on one code's content (``sigma_rot``).

**The units** (each its own job): ``encode`` (a code's whitened sequences
into ``<out>/codes/<code>.pt``), ``gru`` (a code, a count of layers, a
direction and a seed: ``fit_gru`` on ``train`` stopped on ``val``, the
blocks' sums of ``val``, ``rec`` and ``zurich``), ``lin`` (the two-point
linear of every lag in both directions, ``dynpred.fit_linear``), and the
synthetic checks ``check`` (``CHECKS``, below)::

    python -m exp.vision.fnd_dyn encode --config <v18c run> --sets runs/v18c-tlin \
        --out runs/fnd-dyn --code pca [--vae v16 <ckpt>] [--places init <mod:fn> <arg>]
    python -m exp.vision.fnd_dyn rot --out runs/fnd-dyn
    python -m exp.vision.fnd_dyn gru --out runs/fnd-dyn --code pca --layers 1 \
        --dir future --seed 6001 [--device cuda]
    python -m exp.vision.fnd_dyn lin --out runs/fnd-dyn --code pca
    python -m exp.vision.fnd_dyn check --out runs/fnd-dyn/checks --name kalman --layers 1

**The checks** (``check_unit``; F0's synthetic checks, theory's oracles)
on series of ``CheckConf``: one training, one validation and one test
series, each of its own generator. Of each lag the test error $\hat E_\Delta
= \frac{1}{A}\sum_t \lVert \hat w_{t+\Delta} - w_{t+\Delta} \rVert^2$, its
standard error $\mathrm{SE}$ by batch means (the test anchors cut into
blocks of ``block`` steps, far longer than the processes' memory), and the
bounds of theory:

- below, $\hat E_\Delta \ge E^*_\Delta - z\,\mathrm{SE}$: of a Gaussian
  process no predictor from the window has an expected error below $E^*$
- above, $\hat E_\Delta \le E^*_\Delta (1 + p / (n k)) + z\,\mathrm{SE}$:
  the expected excess of a least-squares fit of $p$ parameters (all the
  predictor's, an upper count for one lag's head) on $n k$ scored values
  ($n$ training times of $k$ coordinates), $p / (nk)$ of the error (the
  asymptotic excess of a correct model)

$E^*$ is the least error of the window the predictor sees
(``gru_dyn.best_window``; of a Gaussian process, any predictor's); the
Kalman filter's (``synth_series.kalman_errors``, the whole past) is
recorded beside.

``ou``: a fully observed AR(1) $w_{t+1} = F w_t + \eta_t$ ($F$ of spectral
norm ``rho``), $E^* = \mathrm{tr}(\Sigma - F^\Delta \Sigma F^{\Delta\top})$.
``kalman``: AR(1) states $x_{i,t+1} = \phi_i x_{i,t} + \eta_{i,t}$ of unit
variance ($Q = \mathrm{diag}(1 - \phi_i^2)$, $\phi_i$ evenly over
``phis``), seen mixed by a random orthogonal $H$ with noise of variance
``noise``: as $H^\top \nu$ is white too, each state's Kalman filter reads
its own coordinate $v_t = (H^\top w_t)_i$, $\hat x_{i,t} = a_i \hat
x_{i,t-1} + K_i v_t$ of decay $a_i = \phi_i (1 - K_i) \in (0, 1)$: a
minGRU's unit of a constant gate, so the optimum $\hat w_{t+\Delta} = H
\Phi^\Delta \hat x_t$ lies in the family and a gap to it is the
training's, not the architecture's (value-seen fix, 2026-11-27: damped
rotations, whose filter decays oscillate and so lie outside a minGRU's
positive real decays, kept the minGRU 4 % above the oracle at 0.8 s);
besides, the window loses nothing measurable against
the Kalman filter ($E^*_\Delta - E^{\mathrm{K}}_\Delta \le z\,\mathrm{SE}$),
and the error is below the least of the two-point linear
(``dynpred.best_linear``) at every lag where the two oracles differ by more
than the tolerance ($E^{(2)}_\Delta - E^*_\Delta (1 + p/(nk)) > 2 z
\mathrm{SE}$; at least one such lag): the minGRU uses the history. ``reverse``: a
time-reversible process (a symmetric $F$, $Q = I$: $\Gamma(\ell) =
\Gamma(\ell)^\top$) partly observed; the future and the past are learned
apart and both meet the bounds, and $|\hat E^{\mathrm{f}}_\Delta - \hat
E^{\mathrm{p}}_\Delta| \le z\,\mathrm{SE}_{\mathrm{f-p}} + E^*_\Delta\,p/(nk)$
(the paired blocks' difference). ``iid``: white noise, $E^* = k$ at every
lag, so $R_{\mathrm{clim}} \approx 1$ (and $R_{\mathrm{copy}} \approx
1/2$).
"""

import argparse
import importlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from exp.lib.dynpred import best_linear, fit_linear
from exp.lib.gru_dyn import (
    GRUConf,
    SeqSet,
    best_window,
    block_terms,
    fit_gru,
    gru_sums,
    seq_set,
)
from exp.lib.midpoint import Whitener
from exp.lib.provenance import configure_determinism
from exp.lib.synth_series import kalman_errors, lgssm, lgssm_gamma, white
from exp.lib.tours.walking_tours import videos
from exp.lib.video_io import U8Reader
from exp.vision.content_eval import ContentConf, orthogonal
from exp.vision.dyn_eval import split_videos
from exp.vision.eval import BATCH, PCA_SEED, stream_view
from exp.vision.fill_eval import FillConf
from exp.vision.geom_eval import (
    GeomConf,
    VaeTokenCode,
    _conf_of,
    _mean8,
    _start,
    _write,
)
from exp.vision.jepa_eval import PCACode, pixels, vae_at
from exp.vision.models import VAEModules
from exp.vision.width_eval import WidthConf, sets_and_ref

__all__ = [
    "CHECKS",
    "DIRS",
    "ROLES",
    "CheckConf",
    "FndConf",
    "check_model",
    "check_unit",
    "code_fn",
    "cut_keep",
    "encode_stacks",
    "gru_rows",
    "lin_rows",
    "load_sets",
    "main",
    "rot_sets",
    "save_sets",
    "stack_starts",
]

ROLES = ("train", "val", "rec")
SCORED = ("val", "rec", "zurich")
DIRS = ("future", "past")
CHECKS = ("ou", "kalman", "reverse", "iid")
CODES = ("pca", "mean8")

type Row = dict[str, Any]
type CodeFn = Callable[[Tensor], Tensor]

_F64 = torch.float64


@dataclass(frozen=True)
class FndConf:
    """The settings (module docstring), by default the decided ones: steps
    of 0.8 s."""

    split_seed: int = 4101
    block: int = 160  # 128 s
    seeds: tuple[int, ...] = (6001, 6002, 6003)
    layers: tuple[int, ...] = (1, 2)
    rot_seed: int = ContentConf().rot_seed
    gru: GRUConf = field(default_factory=GRUConf)
    read: int = 256  # stacks read from a video at a time


# -- the sequences ------------------------------------------------------------------


def stack_starts(n_frames: int, frames: int, stride: int) -> Tensor:
    """The first frames ``(m,)`` of the stacks of ``frames`` frames
    ``stride`` apart that do not overlap, in a video of ``n_frames``."""
    span = (frames - 1) * stride
    if n_frames <= span:
        return torch.zeros(0, dtype=torch.long)
    return torch.arange(0, n_frames - span, frames * stride)


@torch.no_grad()
def encode_stacks(
    fn: CodeFn,
    frames_of: Callable[[int, int], Tensor],
    starts: Tensor,
    frames: int,
    stride: int,
    dev: torch.device,
    post: Callable[[Tensor], Tensor],
    read: int,
) -> Tensor:
    """``post`` of the codes ``fn`` ``(m, D')`` of the stacks of ``starts``
    (``frames_of(a, b)`` the frames ``a .. b - 1`` u8 ``(b - a, 3, S, S)``),
    ``read`` stacks' frames read at a time, ``BATCH`` stacks encoded at a
    time on ``dev``; ``post`` row-wise, applied to each batch (a wide code
    is never held whole)."""
    offsets = torch.arange(frames) * stride
    out = []
    for c in starts.split(read):
        a, b = int(c[0]), int(c[-1]) + int(offsets[-1]) + 1
        clip = frames_of(a, b)
        x = clip[(c - a)[:, None] + offsets]  # (r, n, 3, S, S)
        for xb in x.split(BATCH):
            out.append(post(fn(xb.to(dev)).double().cpu()).float())
    return torch.cat(out) if out else torch.zeros(0)


def cut_keep(before: Tensor, starts: Tensor, span: int, margin: int) -> Tensor:
    """Of each stack ``(m,)`` whether no cut lies between the first frame
    of the stack ``margin`` before it and the last frame of the one
    ``margin`` after it (``before`` ``(F,)`` the count of cuts before each
    frame; stacks without both are False)."""
    m = len(starts)
    keep = torch.zeros(m, dtype=torch.bool)
    if m <= 2 * margin:
        return keep
    t = torch.arange(margin, m - margin)
    first = before[starts[t - margin]]
    last = before[starts[t + margin] + span]
    keep[t] = last == first
    return keep


def save_sets(path: Path, sets: Mapping[str, SeqSet], meta: Mapping[str, Any]) -> None:
    """The sets and ``meta`` into ``path`` (``torch.save``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {name: {**s.__dict__} for name, s in sets.items()}
    torch.save({"sets": data, "meta": dict(meta)}, path)


def load_sets(path: Path) -> tuple[dict[str, SeqSet], dict[str, Any]]:
    """The sets and the meta of ``save_sets``."""
    data = torch.load(path, weights_only=False)
    return {k: SeqSet(**v) for k, v in data["sets"].items()}, data["meta"]


def rot_sets(sets: Mapping[str, SeqSet], seed: int) -> dict[str, SeqSet]:
    """The sets of the codes times the orthogonal matrix of ``seed``
    (``content_eval.orthogonal``)."""
    k = next(iter(sets.values())).w.shape[1]
    q = orthogonal(k, seed).float()
    return {n: replace(s, w=s.w @ q) for n, s in sets.items()}


# -- the codes ----------------------------------------------------------------------


def _factory(spec: str) -> Callable[[str, str], CodeFn]:
    mod, _, name = spec.partition(":")
    return getattr(importlib.import_module(mod), name)


def code_fn(
    kind: str, conf: Any, ref: Tensor, *, arg: str = "", gconf: GeomConf = GeomConf()
) -> CodeFn:
    """The map from stacks u8 to codes of ``kind`` (module docstring):
    ``pca``, ``mean8``, ``vae`` (of the checkpoint ``arg``) or ``places``
    (``arg`` = ``"module:function=<its arg>"``).

    Raises:
        ValueError: ``kind`` is unknown, or a VAE's stacks are not
            ``conf``'s.
    """
    if kind in ("pca", "mean8"):
        gen = torch.Generator().manual_seed(conf.seed + PCA_SEED)
        feats = pixels if kind == "pca" else _mean8
        return PCACode.fit(feats, ref, gconf.pca_dim, gen=gen, device=conf.device).mu
    if kind == "vae":
        path = Path(arg)
        view, _ = stream_view(_conf_of(path.parent.parent, conf.device))
        mine, its = conf.data.tours, view.data.tours
        if (mine.frames, mine.stride) != (its.frames, its.stride):
            raise ValueError(
                f"the VAE's stacks ({its.frames}, {its.stride}) are not the "
                f"sets' ({mine.frames}, {mine.stride})"
            )
        mods = vae_at(path, conf.device)
        assert isinstance(mods, VAEModules)
        places = VaeTokenCode(mods).places
        return lambda x: places(x)["own"]
    if kind == "places":
        spec, _, a = arg.partition("=")
        return _factory(spec)(a, conf.device)
    raise ValueError(f"no code kind {kind!r} (pca, mean8, vae, places)")


def encode_code(
    name: str,
    kind: str,
    conf: Any,
    sets_dir: Path,
    out: Path,
    *,
    arg: str = "",
    root: Path | None = None,
    fconf: FndConf = FndConf(),
    gconf: GeomConf = GeomConf(),
) -> Path:
    """The whitened sequences of the code ``name`` (of ``kind``, ``arg``)
    into ``out / "codes" / f"{name}.pt"`` (module docstring; the sets of
    ``conf``'s data, Zurich's cuts of ``sets_dir``). Returns the file."""
    dev = torch.device(conf.device)
    ws, zurich, ref, root = sets_and_ref(
        conf, sets_dir, root, gconf, WidthConf(), FillConf(), None, None
    )
    fn = code_fn(kind, conf, ref, arg=arg, gconf=gconf)
    t = conf.data.tours
    with torch.no_grad():
        raw_ref = torch.cat([fn(c.to(dev)).double().cpu() for c in ref.split(BATCH)])
    wh = Whitener.fit(raw_ref, gconf.k, c_eps=gconf.c_eps)
    target = out / "codes" / f"{name}.pt"
    meta: dict[str, Any] = {"code": name, "kind": kind, "arg": arg,
                            "spec": wh.spec(), "geom": asdict(gconf),
                            "fnd": json.loads(json.dumps(asdict(fconf)))}  # fmt: skip
    if not wh.scoreable:
        save_sets(target, {}, meta | {"unmeasurable": wh.why})
        return target
    margin = fconf.gru.margin
    names = [v for v in videos(root) if v != t.held_out]
    roles = split_videos(names, fconf.split_seed)
    sets: dict[str, SeqSet] = {}
    for role in ROLES:
        ws_, lengths = [], []
        for v in roles[role]:
            reader = U8Reader(root / v)
            starts = stack_starts(len(reader), t.frames, t.stride)
            ws_.append(encode_stacks(fn, lambda a, b, r=reader: r[a:b], starts, t.frames,
                                     t.stride, dev, wh.white, fconf.read))  # fmt: skip
            lengths.append(len(starts))
        sets[role] = seq_set(torch.cat(ws_), lengths, margin, fconf.block,
                             shifted=role == "rec")  # fmt: skip
    frames = zurich.frames
    starts = stack_starts(len(frames), t.frames, t.stride)
    wz = encode_stacks(fn, lambda a, b: frames[a:b], starts, t.frames, t.stride, dev,
                       wh.white, fconf.read)  # fmt: skip
    keep = cut_keep(ws.before, starts, (t.frames - 1) * t.stride, margin)
    sets["zurich"] = seq_set(
        wz, [len(wz)], margin, fconf.block, keep=keep, shifted=True
    )
    meta |= {"roles": {r: list(roles[r]) for r in ROLES}, "key": ws.key,
             "frames": t.frames, "stride": t.stride}  # fmt: skip
    save_sets(target, sets, meta)
    return target


# -- the rows -----------------------------------------------------------------------


def _set_row(name: str, sums: Mapping[str, Tensor]) -> dict[str, Any]:
    """Blocks' sums of ``rec`` and ``zurich``, totals of the others."""
    if name in ("rec", "zurich"):
        return {k: v.tolist() for k, v in sums.items()}
    return {k.removesuffix("_b"): v.sum(-1).tolist() for k, v in sums.items()
            if not k.endswith("_s")}  # fmt: skip


def _directed(sets: Mapping[str, SeqSet], direction: str) -> dict[str, SeqSet]:
    if direction not in DIRS:
        raise ValueError(f"no direction {direction!r} (of {DIRS})")
    return (
        dict(sets)
        if direction == "future"
        else {n: s.reversed() for n, s in sets.items()}
    )


def gru_rows(
    sets: Mapping[str, SeqSet], direction: str, layers: int, seed: int,
    conf: GRUConf, dev: torch.device,
) -> Row:  # fmt: skip
    """The row of one minGRU (module docstring): its validation curve, the
    best epoch and the blocks' sums of ``SCORED``."""
    on = _directed(sets, direction)
    c = replace(conf, layers=layers)
    model, fit = fit_gru(on["train"], on["val"], c, seed, dev)
    mean = on["train"].w.double().mean(0)
    row: Row = {"part": "gru", "dir": direction, "layers": layers, "seed": seed,
                "lags": list(c.lags), "val_curve": list(fit.val), "best": fit.best,
                "epochs": fit.epochs, "params": sum(p.numel() for p in model.parameters())}  # fmt: skip
    row["sets"] = {n: _set_row(n, gru_sums(model, on[n], mean, c, dev))
                   for n in SCORED if n in on}  # fmt: skip
    return row


def lin_rows(sets: Mapping[str, SeqSet], conf: GRUConf) -> list[Row]:
    """The two-point linear (``dynpred.fit_linear``; $\\alpha$ chosen on
    ``val``) of every lag, a row of each direction: the chosen $\\alpha$ and
    the blocks' sums of ``SCORED``."""
    rows: list[Row] = []
    for direction in DIRS:
        on = _directed(sets, direction)
        tr, va = on["train"].triples(), on["val"].triples()
        mean = tr.w.mean(0)
        fits = [fit_linear(tr, va, lag) for lag in conf.lags]
        row: Row = {"part": "lin", "dir": direction, "lags": list(conf.lags),
                    "alpha": [f.alpha for f in fits], "scale": [f.scale for f in fits],
                    "sets": {}}  # fmt: skip
        for n in SCORED:
            if n not in on:
                continue
            s = on[n].triples()
            terms: dict[str, list[Tensor]] = {"num": [], "copy": [], "clim": []}
            for f, lag in zip(fits, conf.lags, strict=True):
                pred, y = f.predict(s)
                now, _, fut = s.at(lag)
                terms["num"].append((pred - y).square().sum(1))
                terms["copy"].append((now - fut).square().sum(1))
                terms["clim"].append((mean - fut).square().sum(1))
            sums = block_terms({k: torch.stack(v) for k, v in terms.items()}, on[n])
            row["sets"][n] = _set_row(n, sums)
        rows.append(row)
    return rows


# -- the checks ---------------------------------------------------------------------


@dataclass(frozen=True)
class CheckConf:
    """The synthetic checks' settings (module docstring)."""

    k: int = 64
    n_train: int = 200_000
    n_val: int = 20_000
    n_test: int = 100_000
    block: int = 500
    z: float = 4.0
    seed: int = 7100
    rho: float = 0.8  # ou: F's spectral norm
    noise: float = 1.0  # kalman, reverse: the observation noise's variance
    phis: tuple[float, float] = (0.9, 0.99)  # kalman: the states' AR(1)
    eigs: tuple[float, float] = (-0.5, 0.9)  # reverse: F's eigenvalues


type Model = tuple[Tensor, Tensor, Tensor, Tensor]  # F, Q, H, R


def check_model(name: str, cc: CheckConf) -> Model:
    """$F, Q, H, R$ of the check ``name`` (module docstring)."""
    k = cc.k
    eye = torch.eye(k, dtype=_F64)
    gen = torch.Generator().manual_seed(cc.seed)
    if name == "ou":
        a = white(k, k, gen)
        return cc.rho * a / torch.linalg.matrix_norm(a, 2), eye, eye, 0 * eye
    if name == "iid":
        return 0 * eye, eye, eye, 0 * eye
    if name == "kalman":
        phi = torch.linspace(cc.phis[0], cc.phis[1], k, dtype=_F64)
        h = orthogonal(k, cc.seed).double()
        return torch.diag(phi), torch.diag(1 - phi**2), h, cc.noise * eye
    if name == "reverse":
        m = 2 * k
        u = orthogonal(m, cc.seed)
        lam = torch.linspace(cc.eigs[0], cc.eigs[1], m, dtype=_F64)
        f = u @ torch.diag(lam) @ u.T
        h = torch.eye(m, dtype=_F64)[:k]
        return (f + f.T) / 2, torch.eye(m, dtype=_F64), h, cc.noise * eye
    raise ValueError(f"no check {name!r} (of {CHECKS})")


def _series(model: Model, n: int, seed: int, cc: CheckConf, margin: int) -> SeqSet:
    w = lgssm(n, *model, torch.Generator().manual_seed(seed)).float()
    return seq_set(w, [n], margin, cc.block)


def _errors(sums: Mapping[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
    """$\\hat E$ ``(L,)``, the blocks' means ``(L, B)`` and $\\mathrm{SE}$
    by batch means."""
    per = sums["num_b"] / sums["n_b"]
    e = sums["num_b"].sum(1) / sums["n_b"].sum()
    return e, per, per.std(1) / math.sqrt(per.shape[1])


def check_unit(
    name: str, conf: GRUConf, cc: CheckConf, seed: int, dev: torch.device
) -> Row:
    """The check ``name`` (module docstring) of the minGRU of ``conf`` and
    ``seed``: its errors, the oracles, the bounds and ``ok``."""
    model = check_model(name, cc)
    m = conf.margin
    on = {"train": _series(model, cc.n_train, cc.seed + 1, cc, m),
          "val": _series(model, cc.n_val, cc.seed + 2, cc, m),
          "test": _series(model, cc.n_test, cc.seed + 3, cc, m)}  # fmt: skip
    star = best_window(lambda lag: lgssm_gamma(lag, *model), conf.window, conf.lags)
    kal = kalman_errors(*model, conf.lags)
    row: Row = {"part": "check", "check": name, "layers": conf.layers, "width": conf.width,
                "seed": seed, "lags": list(conf.lags), "oracle": star.tolist(),
                "kalman": kal.tolist()}  # fmt: skip
    oks = []
    errs: dict[str, tuple[Tensor, Tensor, Tensor]] = {}
    for direction in DIRS if name == "reverse" else ("future",):
        s = _directed(on, direction)
        net, fit = fit_gru(s["train"], s["val"], conf, seed, dev)
        mean = s["train"].w.double().mean(0)
        sums = gru_sums(net, s["test"], mean, conf, dev)
        e, per, se = _errors(sums)
        errs[direction] = (e, per, se)
        p = sum(q.numel() for q in net.parameters())
        excess = p / (len(s["train"].anchors) * cc.k)
        lo, hi = star - cc.z * se, star * (1 + excess) + cc.z * se
        ok = bool(((e >= lo) & (e <= hi)).all())
        oks.append(ok)
        row[direction] = {"err": e.tolist(), "se": se.tolist(), "lo": lo.tolist(),
                          "hi": hi.tolist(), "ok": ok, "params": p, "excess": excess,
                          "best": fit.best, "epochs": fit.epochs, "val": list(fit.val),
                          "r_copy": (sums["num_b"].sum(1) / sums["copy_b"].sum(1)).tolist(),
                          "r_clim": (sums["num_b"].sum(1) / sums["clim_b"].sum(1)).tolist()}  # fmt: skip
        if name == "kalman":
            near = bool(
                (star - kal <= cc.z * se).all()
            )  # the window forgets nothing seen
            oks.append(near)
            row["window_gap"] = {"gap": (star - kal).tolist(), "ok": near}
            g = [[lgssm_gamma(j * lag, *model) for j in (0, 1, 2)] for lag in conf.lags]
            two = torch.tensor([best_linear(*gs) for gs in g], dtype=_F64)
            gap = two - star * (1 + excess) > 2 * cc.z * se
            beats = e + cc.z * se < two
            ok2 = bool(gap.any()) and bool(beats[gap].all())
            oks.append(ok2)
            row["two_point"] = {"oracle": two.tolist(), "gap": gap.tolist(),
                                "beats": beats.tolist(), "ok": ok2}  # fmt: skip
    if name == "reverse":
        (ef, pf, _), (ep, pp, _) = errs["future"], errs["past"]
        d = pf - pp
        se_d = d.std(1) / math.sqrt(d.shape[1])
        p = row["future"]["params"]
        tol = cc.z * se_d + star * p / (len(on["train"].anchors) * cc.k)
        ok3 = bool(((ef - ep).abs() <= tol).all())
        oks.append(ok3)
        row["symmetry"] = {"diff": (ef - ep).tolist(), "tol": tol.tolist(), "ok": ok3}
    row["ok"] = all(oks)
    return row


# -- the CLI ------------------------------------------------------------------------


def _codes_path(out: Path, code: str) -> Path:
    return out / "codes" / f"{code}.pt"


def main(argv: Sequence[str] | None = None) -> Path:
    p = argparse.ArgumentParser(prog="python -m exp.vision.fnd_dyn")
    sub = p.add_subparsers(dest="cmd", required=True)
    enc = sub.add_parser("encode")
    enc.add_argument("--config", type=Path, required=True)
    enc.add_argument("--sets", type=Path, required=True)
    enc.add_argument("--out", type=Path, required=True)
    enc.add_argument("--code", choices=CODES)
    enc.add_argument("--vae", nargs=2, metavar=("NAME", "CKPT"))
    enc.add_argument("--places", nargs=3, metavar=("NAME", "MODULE:FUNCTION", "ARG"))
    enc.add_argument("--root", type=Path)
    enc.add_argument("--device")
    rot = sub.add_parser("rot")
    rot.add_argument("--out", type=Path, required=True)
    g = sub.add_parser("gru")
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--code", required=True)
    g.add_argument("--layers", type=int, required=True)
    g.add_argument("--dir", choices=DIRS, required=True)
    g.add_argument("--seed", type=int, required=True)
    g.add_argument("--device", default="cpu")
    lin = sub.add_parser("lin")
    lin.add_argument("--out", type=Path, required=True)
    lin.add_argument("--code", required=True)
    ch = sub.add_parser("check")
    ch.add_argument("--out", type=Path, required=True)
    ch.add_argument("--name", choices=CHECKS, required=True)
    ch.add_argument("--layers", type=int, required=True)
    ch.add_argument("--seed", type=int, default=FndConf().seeds[0])
    ch.add_argument("--device", default="cpu")
    a = p.parse_args(argv)
    fconf = FndConf()
    if a.cmd == "encode":
        conf = _conf_of(a.config, a.device)
        configure_determinism(conf.device)
        given = [x for x in (a.code, a.vae, a.places) if x]
        if len(given) != 1:
            raise SystemExit("give one of --code, --vae, --places")
        if a.code:
            name, kind, arg = a.code, a.code, ""
        elif a.vae:
            (name, arg), kind = a.vae, "vae"
        else:
            name, spec, arg0 = a.places
            kind, arg = "places", f"{spec}={arg0}"
        return encode_code(name, kind, conf, a.sets, a.out, arg=arg, root=a.root,
                           fconf=fconf)  # fmt: skip
    if a.cmd == "rot":
        sets, meta = load_sets(_codes_path(a.out, "pca"))
        target = _codes_path(a.out, "rot")
        save_sets(target, rot_sets(sets, fconf.rot_seed),
                  meta | {"code": "rot", "rot_seed": fconf.rot_seed})  # fmt: skip
        return target
    if a.cmd == "gru":
        determinism = configure_determinism(a.device)
        sets, meta = load_sets(_codes_path(a.out, a.code))
        target = a.out / "gru" / a.code / f"L{a.layers}-{a.dir}-s{a.seed}.json"
        if "unmeasurable" in meta:
            _write(target, [{"part": "unmeasurable", "why": meta["unmeasurable"]}], {})
            return target
        _start(
            a.out / "gru", {"fnd": json.loads(json.dumps(asdict(fconf)))}, determinism
        )
        row = gru_rows(sets, a.dir, a.layers, a.seed, fconf.gru, torch.device(a.device))
        _write(target, [row], {"code": a.code, "key": meta.get("key")})
        return target
    if a.cmd == "lin":
        configure_determinism("cpu")
        sets, meta = load_sets(_codes_path(a.out, a.code))
        target = a.out / "lin" / f"{a.code}.json"
        rows = ([{"part": "unmeasurable", "why": meta["unmeasurable"]}]
                if "unmeasurable" in meta else lin_rows(sets, fconf.gru))  # fmt: skip
        _write(target, rows, {"code": a.code, "key": meta.get("key")})
        return target
    determinism = configure_determinism(a.device)
    cc = CheckConf()
    _start(a.out, {"check": asdict(cc), "fnd": json.loads(json.dumps(asdict(fconf)))},
           determinism)  # fmt: skip
    conf = replace(fconf.gru, layers=a.layers)
    row = check_unit(a.name, conf, cc, a.seed, torch.device(a.device))
    target = a.out / f"{a.name}-L{a.layers}-s{a.seed}.json"
    _write(target, [row], {})
    return target


if __name__ == "__main__":
    main()
