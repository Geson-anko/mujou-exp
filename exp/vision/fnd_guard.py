r"""The foundation's present and guards (``plans/foundation.md``, 4.2 and
4.3; F0): the completion inside a stack by half the view, the collapse
along time (``time_r2``) and the content a code keeps under a linear
readout ($E^{\mathrm{lin}}$, $E^{d,\mathrm{lin}}$). The JEPA's codes come
later; here are the measures, their inputs and their oracles.

**The tubelets** (``tubelets``): the stack's $n = 8$ frames cut into boxes
of 2 frames x 8 x 8 pixels: a grid of 4 times x 8 x 8 positions (row-major),
each token the box's pixels in [0, 1] flattened (frame, channel, row,
column): ``(B, 4, 64, 384)``. The token of time $j$ holds the frames $2j$
and $2j + 1$.

**time_r2** (``time_r2``): of tokens $t_{n,j,p}$ ``(N, T, P, d)``, less
their mean over the samples $\bar t_{j,p}$, $u = t - \bar t$, the share of
their variance that the mean over time explains, $T \sum \lVert \tfrac1T
\sum_j u_{n,j,p} \rVert^2 / \sum \lVert u_{n,j,p} \rVert^2$: 1 of tokens
constant along time (also with an offset of their time: centred per
$(j, p)$), $1/T$ of tokens independent over time of equal variance. Its
threshold (``time_r2_threshold``) is $\max$(the untrained code's, the
pixel tubelets') + 0.05, fixed before a learned code's value is seen.

**The half view** (``half_rows``; Fable's proposal, 6.2): the grid cut in
two along one axis, a code of each half ($e_H$, ``HALVES``: the spatial
halves ``left``, ``right``, ``top``, ``bottom`` are the present, ``early``
-> ``late`` the future and ``late`` -> ``early`` the past inside the stack).
Of an ordered pair $(a, b)$: one ``midpoint.Whitener`` of ``k``
components fit on the reference's codes of both halves, each less its own
mean there (the same for both directions; a half's origin is its own, as
the encoder knows the times), the means $\mu_a$, $\mu_b$ of the training set, the ridge
$\hat w_b = M w_a + c$ (``linear_dyn.fit_scales``, $\alpha$ the least
numerator on ``val``), and on the evaluation the blocks' sums of the
numerator $\lVert \hat w_b - w_b \rVert^2$, the copy corrected by the means
$\lVert (w_a - \mu_a) - (w_b - \mu_b) \rVert^2$ and the mean's $\lVert
\mu_b - w_b \rVert^2$. Of a field whose times are an AR(1) of correlation
$r$ (each position and dimension apart), ``early`` -> ``late`` has in the
raw coordinates $R_{\mathrm{clim}} = 1 - (r^2 + r^4)/2$ and $R_{\mathrm{
copy}} = (2 + r^2)/4$; the whitened ones are those of the Schur complement
read in the half's whitening. ``pixel_halves`` is the yardstick's code of
a half (its pixels); a learned code's comes of its encoder with the half's
tokens kept (``half_keep``).

**The real data** (the CLI; the sets of ``fnd_dyn``: the training
videos' non-overlapping stacks by role, Zurich's with its 128 s blocks):
``now`` the half view of a code (``pca``: each axis's halves' pixels by one
PCA of ``GeomConf.pca_dim`` fit on both halves of the reference,
``PCACode``'s seed; ``places``: a factory ``module:function`` called as
``function(arg, device)`` that returns the map ``(stacks, half) -> codes``),
``obs`` the observations of ``fnd_dyn``'s sets (once), ``content`` a code's
readout (its whitened sequences of ``fnd_dyn encode``), ``time-r2`` the
pixel tubelets' ``time_r2`` on the reference, or with ``--places`` a code's
tokens' (``function(arg, device)`` the map stacks -> tokens ``(B, T, P,
d)``). The JEPA's checkpoints enter by ``jepa_tokens`` (its tokens $t$ of
every position, ``fnd_jepa.FndCode.tokens``) and ``jepa_halves`` (a half's
code: $\mathrm{vec}$ of the tokens $t = R h$ of the half's positions, the
encoder seeing only them, ``half_keep``)::

    python -m exp.vision.fnd_guard now --config <v18c run> --sets runs/v18c-tlin \
        --out runs/fnd-dyn --code pca [--places init <mod:fn> <arg>]
    python -m exp.vision.fnd_guard obs --config <v18c run> --sets runs/v18c-tlin --out runs/fnd-dyn
    python -m exp.vision.fnd_guard content --out runs/fnd-dyn --code pca
    python -m exp.vision.fnd_guard time-r2 --config <v18c run> --sets runs/v18c-tlin \
        --out runs/fnd-dyn [--places init exp.vision.fnd_guard:jepa_tokens <ckpt>]

**The content** (``content_sums``): a readout $g(w) = A w + c$ (ridge,
$\alpha$ of ``ALPHA_SCALES`` the least level error on ``val``) from a
stack's whitened code to its observation $y$ (the last frame averaged to
32 x 32, ``fidelity.pool_obs``), and on the evaluation the blocks' sums of
$\lVert y - g(w) \rVert^2$, $\lVert y - \bar y \rVert^2$ ($\bar y$ the
training set's), and of the changes between consecutive stacks of a
sequence $d_t = y_t - y_{t-1}$, $\hat d_t = g(w_t) - g(w_{t-1})$: $\lVert d
- \hat d \rVert^2$ and $\lVert d \rVert^2$. ``fidelity_of`` gives $E^{
\mathrm{lin}} = 1 - \sum \lVert y - g \rVert^2 / \sum \lVert y - \bar y
\rVert^2$ and $E^{d,\mathrm{lin}} = 1 - \sum \lVert d - \hat d \rVert^2 /
\sum \lVert d \rVert^2$. The non-inferiority (``noninferior``) is of the
errors $U = 1 - E$: $U_a \le 1.10 \max(U_{\mathrm{pca}}, U_{\mathrm{init}})$
(the decided margin 1.10; a ratio above 1 only bounds an error).
"""

import argparse
import importlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from exp.lib import pca
from exp.lib.dynpred import block_sums
from exp.lib.fidelity import pool_obs
from exp.lib.gru_dyn import step_blocks
from exp.lib.linear_dyn import ALPHA_SCALES, fit_scales
from exp.lib.midpoint import Whitener
from exp.lib.provenance import configure_determinism
from exp.lib.tours.walking_tours import videos
from exp.lib.video_io import U8Reader
from exp.vision.dyn_eval import split_videos
from exp.vision.eval import BATCH, PCA_SEED
from exp.vision.fill_eval import FillConf
from exp.vision.fnd_dyn import FndConf, encode_stacks, load_sets, stack_starts
from exp.vision.fnd_jepa import EVAL_BATCH, FndCode, encode, load_fnd
from exp.vision.geom_eval import GeomConf, _conf_of, _start, _write
from exp.vision.jepa import JEPAModules
from exp.vision.models import early_fusion
from exp.vision.width_eval import WidthConf, sets_and_ref

__all__ = [
    "HALVES",
    "PAIRS",
    "ContentSet",
    "HalfSets",
    "content_sums",
    "fidelity_of",
    "half_keep",
    "half_rows",
    "jepa_halves",
    "jepa_tokens",
    "main",
    "noninferior",
    "pixel_halves",
    "time_r2",
    "time_r2_threshold",
    "tubelets",
]

HALVES = ("left", "right", "top", "bottom", "early", "late")
PAIRS: dict[str, tuple[tuple[str, str], ...]] = {
    "now": (("left", "right"), ("right", "left"), ("top", "bottom"), ("bottom", "top")),
    "future": (("early", "late"),),
    "past": (("late", "early"),),
}
NONINFERIOR = 1.10
TIME_R2_MARGIN = 0.05

type Row = dict[str, Any]

_F64 = torch.float64


# -- the tokens and time_r2 ---------------------------------------------------------


def tubelets(x: Tensor, frames: int = 2, patch: int = 8) -> Tensor:
    """The tubelets ``(B, n / frames, (S / patch)^2, frames 3 patch^2)``
    float32 in [0, 1] of the stacks u8 ``x`` ``(B, n, 3, S, S)`` (module
    docstring)."""
    b, n, c, s, _ = x.shape
    g = s // patch
    y = x.float().div(255).reshape(b, n // frames, frames, c, g, patch, g, patch)
    # (B, T, f, c, gy, py, gx, px) -> (B, T, gy, gx, f, c, py, px)
    y = y.permute(0, 1, 4, 6, 2, 3, 5, 7)
    return y.reshape(b, n // frames, g * g, frames * c * patch * patch)


def time_r2(tokens: Tensor) -> float:
    """``time_r2`` of the tokens ``(N, T, P, d)`` (module docstring)."""
    u = tokens.double() - tokens.double().mean(0)
    t = u.shape[1]
    explained = t * u.mean(1).square().sum()
    return float(explained / u.square().sum())


def time_r2_threshold(
    init: float, pixel: float, margin: float = TIME_R2_MARGIN
) -> float:
    """The threshold of ``time_r2``: $\\max$(``init``, ``pixel``) +
    ``margin``."""
    return max(init, pixel) + margin


# -- the half view ------------------------------------------------------------------


def half_keep(half: str, grid: tuple[int, int, int] = (4, 8, 8)) -> Tensor:
    """The tokens of ``half`` ``(T, H, W)`` bool of the grid (times, rows,
    columns): the encoder's keep of a half's code.

    Raises:
        ValueError: ``half`` is not of ``HALVES``.
    """
    t, h, w = grid
    keep = torch.zeros(grid, dtype=torch.bool)
    cut = {"left": (slice(None), slice(None), slice(0, w // 2)),
           "right": (slice(None), slice(None), slice(w // 2, None)),
           "top": (slice(None), slice(0, h // 2), slice(None)),
           "bottom": (slice(None), slice(h // 2, None), slice(None)),
           "early": (slice(0, t // 2), slice(None), slice(None)),
           "late": (slice(t // 2, None), slice(None), slice(None))}  # fmt: skip
    if half not in cut:
        raise ValueError(f"no half {half!r} (of {HALVES})")
    keep[cut[half]] = True
    return keep


def pixel_halves(x: Tensor, half: str) -> Tensor:
    """The pixels ``(B, n 3 S S / 2)`` float32 in [0, 1] of ``half`` of the
    stacks u8 ``x`` ``(B, n, 3, S, S)`` (the frames ``early``: the first
    half of the stack)."""
    n, s = x.shape[1], x.shape[-1]
    parts = {"left": x[..., : s // 2], "right": x[..., s // 2 :],
             "top": x[..., : s // 2, :], "bottom": x[..., s // 2 :, :],
             "early": x[:, : n // 2], "late": x[:, n // 2 :]}  # fmt: skip
    if half not in parts:
        raise ValueError(f"no half {half!r} (of {HALVES})")
    return parts[half].flatten(1).float() / 255


@dataclass(frozen=True)
class HalfSets:
    """The halves' codes ``(N, D)`` by set (``ref``, ``train``, ``val``,
    ``eval``) and half, aligned by stack, the evaluation's blocks
    ``(N_eval,)`` of ``n_blocks`` (and optionally shifted ``blocks_s``)."""

    codes: Mapping[str, Mapping[str, Tensor]]
    blocks: Tensor
    n_blocks: int
    blocks_s: Tensor | None = None
    n_blocks_s: int = 0


def _sums(terms: Mapping[str, Tensor], h: HalfSets) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    ones = torch.ones(len(h.blocks), dtype=_F64)
    for blocks, nb, tail in (
        (h.blocks, h.n_blocks, ""),
        (h.blocks_s, h.n_blocks_s, "_s"),
    ):
        if blocks is None:
            continue
        for name, v in terms.items():
            out[f"{name}_b{tail}"] = block_sums(v, blocks, nb).tolist()
        out[f"n_b{tail}"] = block_sums(ones, blocks, nb).tolist()
    return out


def half_rows(
    h: HalfSets,
    pairs: Sequence[tuple[str, str]],
    k: int,
    *,
    c_eps: float = 1e-6,
    scales: Sequence[float] = ALPHA_SCALES,
) -> list[Row]:
    """The rows of the ordered ``pairs`` (module docstring): of each, the
    chosen $\\alpha$ and the evaluation's blocks' sums ``num_b``,
    ``copy_b``, ``clim_b``, ``n_b`` (and ``*_s``); a pair whose whitening
    cannot score is ``unmeasurable``."""
    rows: list[Row] = []
    c = h.codes
    for a, b in pairs:
        ref = {x: c["ref"][x].double() for x in (a, b)}
        m_ref = {x: ref[x].mean(0) for x in (a, b)}
        pooled = torch.cat([ref[a] - m_ref[a], ref[b] - m_ref[b]])
        wh = Whitener.fit(pooled, k, c_eps=c_eps)
        if not wh.scoreable:
            rows.append({"part": "unmeasurable", "src": a, "dst": b, "why": wh.why})
            continue
        w = {s: {x: wh.white(c[s][x].double() - m_ref[x]) for x in (a, b)}
             for s in ("train", "val", "eval")}  # fmt: skip
        mu_a, mu_b = w["train"][a].mean(0), w["train"][b].mean(0)
        fits = fit_scales(w["train"][a], w["train"][b], scales)
        val = [
            float((w["val"][a] @ m.T + v - w["val"][b]).square().sum())
            for _, m, v in fits
        ]
        best = min(range(len(fits)), key=lambda i: (val[i], i))
        alpha, m, v = fits[best]
        xa, xb = w["eval"][a], w["eval"][b]
        terms = {"num": (xa @ m.T + v - xb).square().sum(1),
                 "copy": ((xa - mu_a) - (xb - mu_b)).square().sum(1),
                 "clim": (mu_b - xb).square().sum(1)}  # fmt: skip
        rows.append({"part": "half", "src": a, "dst": b, "alpha": alpha,
                     "scale": scales[best], "val_scales": val} | _sums(terms, h))  # fmt: skip
    return rows


# -- the content --------------------------------------------------------------------


@dataclass(frozen=True)
class ContentSet:
    """A set's whitened codes ``w`` ``(N, k)``, observations ``y`` ``(N,
    m)`` and sequences ``bounds`` ``(V + 1,)`` (consecutive rows of a
    sequence are consecutive stacks); of the evaluation, each row's block
    ``blocks`` ``(N,)`` of ``n_blocks``."""

    w: Tensor
    y: Tensor
    bounds: Tensor
    blocks: Tensor | None = None
    n_blocks: int = 0

    def pairs(self) -> Tensor:
        """The rows $t$ ``(n,)`` whose $t - 1$ is of their sequence."""
        idx = torch.arange(len(self.w))
        first = torch.zeros(len(self.w), dtype=torch.bool)
        first[self.bounds[:-1][self.bounds[:-1] < len(self.w)]] = True
        return idx[~first]


def content_sums(
    train: ContentSet, val: ContentSet, ev: ContentSet,
    scales: Sequence[float] = ALPHA_SCALES,
) -> Row:  # fmt: skip
    """The readout's rows (module docstring): the chosen $\\alpha$, the
    level errors of ``val`` at every scale, and the evaluation's blocks'
    sums ``err_b``, ``den_b``, ``derr_b``, ``dden_b``, ``n_b``, ``dn_b``.

    Raises:
        ValueError: The evaluation has no blocks.
    """
    if ev.blocks is None:
        raise ValueError("the evaluation's rows need their blocks")
    fits = fit_scales(train.w.double(), train.y.double(), scales)
    errs = [float((val.w.double() @ a.T + b - val.y.double()).square().sum())
            for _, a, b in fits]  # fmt: skip
    best = min(range(len(fits)), key=lambda i: (errs[i], i))
    alpha, a, b = fits[best]
    y = ev.y.double()
    g = ev.w.double() @ a.T + b
    ybar = train.y.double().mean(0)
    t = ev.pairs()
    d, dhat = y[t] - y[t - 1], g[t] - g[t - 1]
    nb = ev.n_blocks
    blk = ev.blocks
    return {"part": "content", "alpha": alpha, "scale": scales[best], "val_scales": errs,
            "err_b": block_sums((y - g).square().sum(1), blk, nb).tolist(),
            "den_b": block_sums((y - ybar).square().sum(1), blk, nb).tolist(),
            "n_b": block_sums(torch.ones(len(y), dtype=_F64), blk, nb).tolist(),
            "derr_b": block_sums((d - dhat).square().sum(1), blk[t], nb).tolist(),
            "dden_b": block_sums(d.square().sum(1), blk[t], nb).tolist(),
            "dn_b": block_sums(torch.ones(len(t), dtype=_F64), blk[t], nb).tolist()}  # fmt: skip


def fidelity_of(row: Mapping[str, Any]) -> tuple[float, float]:
    """$E^{\\mathrm{lin}}$ and $E^{d,\\mathrm{lin}}$ of a ``content_sums``
    row."""
    e = 1 - sum(row["err_b"]) / sum(row["den_b"])
    ed = 1 - sum(row["derr_b"]) / sum(row["dden_b"])
    return e, ed


def noninferior(e: float, refs: Sequence[float], margin: float = NONINFERIOR) -> bool:
    """Whether the error $1 - E$ of ``e`` is at most ``margin`` times the
    largest of the references' (module docstring)."""
    return 1 - e <= margin * max(1 - r for r in refs)


# -- the real data ------------------------------------------------------------------


type HalfFn = Callable[[Tensor, str], Tensor]


def _pixel_half_fn(ref: Tensor, d: int, seed: int, dev: torch.device) -> HalfFn:
    """``pca``'s half code: each axis's PCA of ``d`` components of both
    halves' pixels of the reference."""
    maps: dict[str, tuple[Tensor, Tensor]] = {}
    for a, b in (("left", "right"), ("top", "bottom"), ("early", "late")):
        x = torch.cat([torch.cat([pixel_halves(c.to(dev), h).cpu() for c in ref.split(BATCH)])
                       for h in (a, b)])  # fmt: skip
        fit = pca.fit_pca(x, d, gen=torch.Generator().manual_seed(seed))
        maps[a] = maps[b] = (fit[0].to(dev), fit[1].to(dev))
    return lambda x, half: pca.encode(pixel_halves(x, half), *maps[half])


def _jepa(arg: str, device: str) -> tuple[JEPAModules, int, tuple[int, int, int]]:
    """The checkpoint ``arg``'s JEPA, its heads and its grid (times, rows,
    columns)."""
    conf, settings, mods = load_fnd(Path(arg), device)
    t = conf.data.tours
    g = t.size // conf.model.patch
    return mods, conf.model.head_count, (t.frames // settings["tubelet"], g, g)


def jepa_tokens(arg: str, device: str) -> Callable[[Tensor], Tensor]:
    """The tokens $t$ ``(B, T, P, d)`` of the JEPA's checkpoint ``arg`` (all
    positions seen; ``fnd_jepa.FndCode.tokens``, time-major)."""
    mods, heads, (times, _, _) = _jepa(arg, device)
    code = FndCode(mods, heads)
    return lambda x: code.tokens(x).unflatten(1, (times, -1))


def jepa_halves(arg: str, device: str) -> HalfFn:
    r"""The half code of the JEPA's checkpoint ``arg``: the encoder sees the
    half's positions only (``half_keep``) and the code is $\mathrm{vec}$ of
    their tokens $t = R h$ ``(B, |half| d)``."""
    mods, heads, grid = _jepa(arg, device)
    dev = torch.device(device)
    keeps = {
        h: half_keep(h, grid).flatten().nonzero().squeeze(1).to(dev) for h in HALVES
    }

    @torch.no_grad()
    def fn(x: Tensor, half: str) -> Tensor:
        out = []
        for c in x.split(EVAL_BATCH):
            keep = keeps[half].expand(len(c), -1)
            out.append(
                mods.proj(encode(mods, early_fusion(c.to(dev)), heads, keep)).flatten(1)
            )
        return torch.cat(out)

    return fn


def _role_stacks(
    conf: Any, root: Path, names: Sequence[str]
) -> list[tuple[Callable[[int, int], Tensor], Tensor]]:
    t = conf.data.tours
    out = []
    for v in names:
        r = U8Reader(root / v)
        out.append(
            ((lambda a, b, r=r: r[a:b]), stack_starts(len(r), t.frames, t.stride))
        )
    return out


@torch.no_grad()
def half_sets(fn: HalfFn, conf: Any, sets_dir: Path, root: Path | None,
              fconf: FndConf = FndConf(), gconf: GeomConf = GeomConf()) -> HalfSets:  # fmt: skip
    """The halves' codes of ``fn`` on the reference, the training videos'
    roles ``train`` and ``val`` and Zurich (``eval``, its 128 s blocks and
    the shifted ones; module docstring)."""
    dev = torch.device(conf.device)
    _, zurich, ref, root = sets_and_ref(conf, sets_dir, root, gconf, WidthConf(),
                                        FillConf(), None, None)  # fmt: skip
    t = conf.data.tours
    names = [v for v in videos(root) if v != t.held_out]
    roles = split_videos(names, fconf.split_seed)
    codes: dict[str, dict[str, Tensor]] = {
        "ref": {},
        "train": {},
        "val": {},
        "eval": {},
    }
    frames = zurich.frames
    zs = stack_starts(len(frames), t.frames, t.stride)
    for half in HALVES:
        codes["ref"][half] = torch.cat([fn(c.to(dev), half).double().cpu()
                                        for c in ref.split(BATCH)]).float()  # fmt: skip
        for role in ("train", "val"):
            parts = [encode_stacks(lambda x, h=half: fn(x, h), get, st, t.frames, t.stride, dev,
                                   lambda z: z, fconf.read)
                     for get, st in _role_stacks(conf, root, roles[role])]  # fmt: skip
            codes[role][half] = torch.cat(parts)
        codes["eval"][half] = encode_stacks(lambda x, h=half: fn(x, h),
                                            lambda a, b: frames[a:b], zs, t.frames,
                                            t.stride, dev, lambda z: z, fconf.read)  # fmt: skip
    blocks = step_blocks([len(zs)], fconf.block, 0)
    blocks_s = step_blocks([len(zs)], fconf.block, fconf.block // 2)
    return HalfSets(
        codes, blocks, int(blocks.max()) + 1, blocks_s, int(blocks_s.max()) + 1
    )


def obs_sets(conf: Any, sets_dir: Path, root: Path | None, size: int,
             fconf: FndConf = FndConf(), gconf: GeomConf = GeomConf()) -> dict[str, Tensor]:  # fmt: skip
    """The observations (the last frame of each stack averaged to ``size``
    x ``size``, ``fidelity.pool_obs``) ``(N, 3 size^2)`` float16 of
    ``fnd_dyn``'s sets, in their rows' order."""
    _, zurich, _, root = sets_and_ref(conf, sets_dir, root, gconf, WidthConf(),
                                      FillConf(), None, None)  # fmt: skip
    t = conf.data.tours
    span = (t.frames - 1) * t.stride
    names = [v for v in videos(root) if v != t.held_out]
    roles = split_videos(names, fconf.split_seed)
    out: dict[str, Tensor] = {}
    for role in ("train", "val", "rec"):
        parts = []
        for get, st in _role_stacks(conf, root, roles[role]):
            for c in st.split(fconf.read):
                parts.append(
                    pool_obs(
                        get(int(c[0]) + span, int(c[-1]) + span + 1)[c - c[0]], size
                    ).half()
                )
        out[role] = torch.cat(parts)
    zs = stack_starts(len(zurich.frames), t.frames, t.stride)
    out["zurich"] = pool_obs(zurich.frames[zs + span], size).half()
    return out


def _content_set(s: Any, y: Tensor, blocks: bool) -> ContentSet:
    if not blocks:
        return ContentSet(s.w, y.float(), s.bounds)
    lengths = (s.bounds[1:] - s.bounds[:-1]).tolist()
    b = step_blocks(lengths, FndConf().block, 0)  # the anchors' cut
    return ContentSet(s.w, y.float(), s.bounds, b, int(b.max()) + 1 if len(b) else 0)


def _factory(spec: str) -> Callable[[str, str], Any]:
    mod, _, name = spec.partition(":")
    return getattr(importlib.import_module(mod), name)


def main(argv: Sequence[str] | None = None) -> Path:
    p = argparse.ArgumentParser(prog="python -m exp.vision.fnd_guard")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("now", "obs", "time-r2"):
        q = sub.add_parser(name)
        q.add_argument("--config", type=Path, required=True)
        q.add_argument("--sets", type=Path, required=True)
        q.add_argument("--out", type=Path, required=True)
        q.add_argument("--root", type=Path)
        q.add_argument("--device")
        if name == "now":
            q.add_argument("--code", choices=("pca",))
        if name != "obs":
            q.add_argument(
                "--places", nargs=3, metavar=("NAME", "MODULE:FUNCTION", "ARG")
            )
    c = sub.add_parser("content")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--code", required=True)
    a = p.parse_args(argv)
    gconf = GeomConf()
    if a.cmd == "content":
        configure_determinism("cpu")
        sets, meta = load_sets(a.out / "codes" / f"{a.code}.pt")
        target = a.out / "content" / f"{a.code}.json"
        if "unmeasurable" in meta:
            _write(target, [{"part": "unmeasurable", "why": meta["unmeasurable"]}], {})
            return target
        obs = torch.load(a.out / "obs.pt")
        row = content_sums(_content_set(sets["train"], obs["train"], False),
                           _content_set(sets["val"], obs["val"], False),
                           _content_set(sets["zurich"], obs["zurich"], True))  # fmt: skip
        _write(target, [row], {"code": a.code, "key": meta.get("key")})
        return target
    conf = _conf_of(a.config, a.device)
    determinism = configure_determinism(conf.device)
    dev = torch.device(conf.device)
    if a.cmd == "obs":
        target = a.out / "obs.pt"
        torch.save(obs_sets(conf, a.sets, a.root, 32), target)
        return target
    if a.cmd == "time-r2":
        _, _, ref, _ = sets_and_ref(conf, a.sets, a.root, gconf, WidthConf(), FillConf(),
                                    None, None)  # fmt: skip
        name, tok_fn = "pixel", tubelets
        if a.places:
            name, spec, arg = a.places
            tok_fn = _factory(spec)(arg, conf.device)
        with torch.no_grad():
            tok = torch.cat([tok_fn(x.to(dev)).cpu() for x in ref.split(BATCH)])
        target = a.out / "guard" / f"time_r2_{name}.json"
        _write(target, [{"part": "time_r2", "code": name, "value": time_r2(tok),
                         "n": len(tok)}], {})  # fmt: skip
        return target
    if a.code:
        _, _, ref, _ = sets_and_ref(conf, a.sets, a.root, gconf, WidthConf(), FillConf(),
                                    None, None)  # fmt: skip
        name = a.code
        fn = _pixel_half_fn(ref, gconf.pca_dim, conf.seed + PCA_SEED, dev)
    else:
        name, spec, arg = a.places
        fn = _factory(spec)(arg, conf.device)
    _start(a.out / "now", {"geom": asdict(gconf)}, determinism)
    h = half_sets(fn, conf, a.sets, a.root, gconf=gconf)
    rows = [r | {"axis": axis} for axis, pairs in PAIRS.items()
            for r in half_rows(h, pairs, gconf.k, c_eps=gconf.c_eps)]  # fmt: skip
    target = a.out / "now" / f"{name}.json"
    _write(target, rows, {"code": name})
    return target


if __name__ == "__main__":
    main()
