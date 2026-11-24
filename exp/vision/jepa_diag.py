r"""The diagnosis of V18's failure (README, 「V18」の「診断（回す前に決めたこと）」),
on the checkpoints of the runs, without learning again. Three commands:

- ``tokens`` (question 1): the probes of ``stream_eval.StreamEvaluator``
  (``recon_blur``, ``dyn_1/copy_1``, ``rank`` of span, after and zurich,
  ``jepa_eval.SPAN`` by default) of a run's checkpoints through the codes
  of ``jepa_eval.TokenCode``: by default (``POOLS``) ``z`` (the code of
  V18's evaluation), and of $h$, the encoder's tokens after its last
  ``LayerNorm`` ($T \times w$, the input of the projection $P$; LeJEPA
  evaluates there), ``h_mean`` (their mean over the positions, $w$) and
  ``h_flat`` (flattened in row-major positions, $T w$); or ``h_pool``
  (V18b's patch 4: the means of $h$ over each block of the coarse grid,
  $16 w$ as ``h_flat`` of patch 16) and ``zbar``. Into
  ``<run_dir>/posthoc_diag/``: ``tokens.json``, ``tokens.jsonl`` (a row per
  checkpoint, code and group) and the pictures ``<t>-<code>-
  {zurich,span}.png``.
- ``sigreg`` (question 3): $S$ (``exp.lib.sigreg.sigreg``) of $z$ along the
  directions of ``jepa_eval.CodeStats`` on stacks of the held-out video in
  groups of ``n`` of the same samples: the stacks of a grid every
  ``GRID_STEP`` video frames over the whole video (``ZurichGrid``), cut
  into ``partition``'s groups, ``consec`` (``n`` neighbours) or ``spaced``
  (every $P$-th, $P$ = grid // ``n``); beside them, $S$ of the val stacks of
  ``CodeStats`` (``val``) and of the held-in batches (``heldin``, ``n`` =
  128 and their concatenations to 1024). Each group's row has ``pred``, the
  expectation of $S$ (``exp.lib.sigreg.expected_epps_pulley``) of jointly
  Gaussian samples of each direction's mean, variance (over the grid's
  stacks of the groups) and autocovariance at the lags of the pairs
  (``autocov``, ``lag_pairs``), and ``pred_iid``, the same without the
  correlation: ``pred - pred_iid`` is the bias of the correlated samples
  (Fable's $\frac1N \sum_{i \ne j} G(\rho_{ij})$ for a unit marginal). Into
  ``<run_dir>/posthoc_diag/``: ``sigreg.json``, ``sigreg.jsonl`` (a row per
  checkpoint and set).
- ``imagenette`` (question 2): M17's linear probe (``metrics.
  probe_accuracy``) of Imagenette (``exp/video/diag/imagenette.py``'s
  ``prep``, ``<data>/{train,val}.pt``) of each code, an image shrunk to
  ``IMAGE_SIDE`` (``shrink``) and repeated into a still stack of the run's
  $n$ frames (``still_stacks``): ``IMAGENETTE_POOLS`` of the runs'
  checkpoints by their SIGReg's data point (``h_mean`` and ``z``, of arm M
  ``h_mean`` and ``zbar``; or ``--codes``, of ``TokenCode``'s pools and
  ``z_w``, ``jepa_eval.RandProjCode``), ``color`` (``jepa_eval.ColorCode``) and the
  $\mu$ of VAE checkpoints. Into ``<out>/``: ``imagenette.json``, ``imagenette.jsonl`` (a
  row per code)::

    python -m exp.vision.jepa_diag tokens <run_dir> [--steps 0,6000,20000,60000] \
        [--codes z,h_mean,h_flat] [--span 6000]
    python -m exp.vision.jepa_diag sigreg <run_dir> [--steps ...]
    python -m exp.vision.jepa_diag imagenette --data <dir> --out <dir> \
        [--runs <run_dir> ...] [--steps ...] [--v16 <ckpt>] [--v17 <ckpt>] \
        [--codes h_mean,z,z_w]
"""

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor, nn

from exp.lib import ckpt
from exp.lib.metrics import probe_accuracy
from exp.lib.provenance import configure_determinism, of_diagnosis
from exp.lib.rundir import write_json
from exp.lib.sigreg import expected_epps_pulley, sigreg
from exp.lib.tours.stacks import tours_root
from exp.lib.video_io import U8Reader
from exp.vision.compose import from_dict
from exp.vision.eval import BATCH, FIT_STEPS, eval_splits
from exp.vision.iid import Units
from exp.vision.jepa import JEPAModules, directions
from exp.vision.jepa_eval import (
    EVAL_DIRECTION_SEED,
    EVAL_DIRECTIONS,
    HELDIN_BATCH,
    HELDIN_BATCHES,
    HELDIN_SEED,
    PICTURES,
    SPAN,
    ColorCode,
    RandProjCode,
    TokenCode,
    build_run_jepa,
    vae_at,
)
from exp.vision.jepa_iid import v16_config
from exp.vision.schema import RunConf
from exp.vision.stream_eval import Code, StreamEvaluator, append_jsonl, spanned

__all__ = [
    "DIAG_STEPS",
    "GRID_STEP",
    "GROUP_SIZES",
    "IMAGENETTE_POOLS",
    "IMAGE_SIDE",
    "LAYOUTS",
    "OUT",
    "POOLS",
    "TokenCode",
    "ZurichGrid",
    "autocov",
    "evaluate_imagenette",
    "evaluate_sigreg",
    "evaluate_tokens",
    "grid_starts",
    "lag_pairs",
    "main",
    "partition",
    "run_ckpts",
    "shrink",
    "still_stacks",
]

OUT = "posthoc_diag"
POOLS = ("z", "h_mean", "h_flat")  # the default codes of tokens
IMAGENETTE_POOLS = {"flat": ("h_mean", "z"), "mean": ("h_mean", "zbar")}
DIAG_STEPS = (0, 6000, 20000, 60000)  # the default steps of tokens
GRID_STEP = 10  # video frames between the grid's stacks: 1 s at 10 fps
GROUP_SIZES = (128, 1024)  # n of sigreg's groups: the training's batch, val's
LAYOUTS = ("consec", "spaced")
IMAGE_SIDE = 64


# -- codes and checkpoints -------------------------------------------------------


def _conf(run_dir: Path) -> RunConf:
    return from_dict(yaml.safe_load((run_dir / "config.yaml").read_text()))


def _settings_of(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "jepa.json").read_text())


def _jepa(run_dir: Path, conf: RunConf) -> JEPAModules:
    """The JEPA of the run in ``run_dir`` of ``conf`` (its $d$)."""
    return build_run_jepa(conf, _settings_of(run_dir))


def run_ckpts(run_dir: Path) -> dict[int, Path]:
    """The checkpoints of the JEPA run in ``run_dir`` by their step (``t``
    over ``time.frame`` frames, over ``jepa.json``'s batch), in order."""
    conf = _conf(run_dir)
    batch = _settings_of(run_dir)["batch"]
    paths = sorted(run_dir.glob("ckpt/*.pt"), key=lambda p: float(p.stem))
    return {round(float(p.stem) / conf.time.frame) // batch: p for p in paths}


def _pick(
    ckpts: Mapping[int, Path], steps: Sequence[int] | None, default: Sequence[int]
) -> list[int]:
    """``steps`` (each a checkpoint's), or those of ``default`` there (all
    the checkpoints for an empty ``default``).

    Raises:
        ValueError: A step of ``steps`` has no checkpoint, or none is chosen.
    """
    if steps is not None:
        missing = [s for s in steps if s not in ckpts]
        if missing:
            raise ValueError(f"no checkpoint of steps {missing} (of {list(ckpts)})")
        return list(steps)
    chosen = [s for s in ckpts if not default or s in default]
    if not chosen:
        raise ValueError(f"no checkpoint of steps {list(default)} (of {list(ckpts)})")
    return chosen


def _encode(fn: Callable[[Tensor], Tensor], x: Tensor, dev: torch.device) -> Tensor:
    """``fn`` of the stacks u8 ``x`` (CPU), ``BATCH`` at a time on ``dev``."""
    return torch.cat([fn(c.to(dev)).float() for c in x.split(BATCH)])


def _settings(**kw: Any) -> dict[str, Any]:
    """``kw`` with the provenance of the diagnosis (``of_diagnosis``)."""
    prov = of_diagnosis({})
    return kw | {"provenance": {k: v for k, v in prov.items() if k != "settings"}}


# -- question 1: the probes of h ------------------------------------------------


def evaluate_tokens(
    run_dir: Path,
    *,
    root: Path | None = None,
    span: float = SPAN,
    steps: Sequence[int] | None = None,
    pools: Sequence[str] = POOLS,
    fit_steps: int = FIT_STEPS,
    spacing: Mapping[str, float] | None = None,
) -> Path:
    """The probes of ``pools`` (``TokenCode``) of the checkpoints of
    ``steps`` (by default those of ``DIAG_STEPS``) of the JEPA run in
    ``run_dir`` on the span ``[0, span)``, as ``jepa_eval.evaluate_run``'s,
    into ``run_dir / OUT`` (module docstring). Returns the directory."""
    conf = _conf(run_dir)
    view = spanned(conf, span)
    determinism = configure_determinism(conf.device)
    ckpts = run_ckpts(run_dir)
    chosen = _pick(ckpts, steps, DIAG_STEPS)
    root = tours_root(conf.data.tours.data_version) if root is None else root
    out = run_dir / OUT
    ev = StreamEvaluator(
        view, run_dir, root, out, fit_steps=fit_steps, c=1, **(spacing or {})
    )
    settings = ev.settings() | {"span": view.measure.duration, "steps": chosen}
    settings |= {"codes": list(pools), "determinism": determinism}
    write_json(out / "tokens.json", _settings(**settings))
    (out / "tokens.jsonl").write_text("")
    mods = _jepa(run_dir, conf)
    for step in chosen:
        path = ckpts[step]
        head = ckpt.load_eval(path, mods.modules())
        for pool in pools:
            rows = ev.probes(TokenCode(mods, pool), f"{path.stem}-{pool}", PICTURES)
            stamp = {"t": head["t"], "step": step, "code": pool}
            append_jsonl(out / "tokens.jsonl", [stamp | r for r in rows])
    return out


# -- question 3: SIGReg of correlated and of spaced stacks ----------------------


def grid_starts(frames: int, n: int, stride: int, step: int) -> Tensor:
    """The first frames ``0, step, 2 step, ...`` of the stacks of ``n``
    frames ``stride`` apart that fit in a video of ``frames`` frames."""
    return torch.arange(0, frames - (n - 1) * stride, step)


def partition(g: int, n: int, layout: str) -> Tensor:
    """$P$ = ``g`` // ``n`` groups of ``n`` of the indices $0 .. Pn - 1$,
    each once, ``(P, n)``: ``"consec"`` the runs $jn .. jn + n - 1$,
    ``"spaced"`` every $P$-th from $j$.

    Raises:
        ValueError: ``g`` < ``n``, or ``layout`` is not of ``LAYOUTS``.
    """
    p = g // n
    if p < 1 or layout not in LAYOUTS:
        raise ValueError(f"no {layout!r} groups of {n} of {g} (of {LAYOUTS})")
    idx = torch.arange(p * n)
    return idx.view(p, n) if layout == "consec" else idx.view(n, p).T.contiguous()


def lag_pairs(groups: Tensor) -> tuple[Tensor, Tensor]:
    """The distinct lags $\\lvert i - j \\rvert$ of the pairs of indices
    within a row of ``groups`` ``(P, n)``, in increasing order ``(L,)``, and
    the ordered pairs $i \\ne j$ of each a group (their mean over the rows)
    ``(L,)`` float64."""
    n = groups.shape[1]
    iu = torch.triu_indices(n, n, 1)
    lags = (groups[:, iu[0]] - groups[:, iu[1]]).abs().flatten()
    lags, counts = lags.unique(return_counts=True)
    return lags, 2 * counts.double() / len(groups)


def autocov(x: Tensor, lags: Tensor) -> Tensor:
    r"""$c(l) = \frac{1}{G - l} \sum_{g < G - l} (x_g - \bar x)(x_{g+l} -
    \bar x)$ of each column of ``x`` ``(G, M)`` ($\bar x$ its mean) at the
    ``lags`` ``(L,)`` (each below $G$): ``(L, M)`` float64, by the FFT."""
    g = len(x)
    u = x.double() - x.double().mean(0)
    nfft = 1 << (2 * g - 1).bit_length()  # no wrap-around of the lags
    f = torch.fft.rfft(u, nfft, dim=0)
    r = torch.fft.irfft(f.real.square() + f.imag.square(), nfft, dim=0)[:g]
    return r[lags] / (g - lags).double()[:, None].to(r.device)


class ZurichGrid:
    """The stacks of the held-out video of ``conf`` at ``root`` whose first
    frames are ``grid_starts`` of ``step``: ``codes`` of a code, ``(G,
    D)``."""

    def __init__(self, conf: RunConf, root: Path, step: int = GRID_STEP) -> None:
        t = conf.data.tours
        reader = U8Reader(root / t.held_out)
        self._frames = reader[0 : len(reader)]  # (F, 3, S, S) u8, CPU
        self.starts = grid_starts(len(reader), t.frames, t.stride, step)
        self._offsets = torch.arange(t.frames) * t.stride
        self.step = step

    def __len__(self) -> int:
        return len(self.starts)

    def stacks(self, idx: Tensor) -> Tensor:
        """The grid's stacks ``idx`` u8 ``(b, n, 3, S, S)``."""
        return self._frames[self.starts[idx][:, None] + self._offsets]

    @torch.no_grad()
    def codes(self, mu: Callable[[Tensor], Tensor], dev: torch.device) -> Tensor:
        return torch.cat(
            [
                mu(self.stacks(i).to(dev)).float()
                for i in torch.arange(len(self)).split(BATCH)
            ]
        )


def _predict(
    x: Tensor, n: int, lags: Tensor | None = None, pairs: Tensor | None = None
) -> float:
    """``pred`` of groups of ``n`` of the projections ``x`` ``(G, M)`` (in
    time order; ``lags`` ``(L,)`` in their indices, ``pairs`` of each a
    group, ``lag_pairs``), or without ``lags`` ``pred_iid`` (module
    docstring): the mean over the directions."""
    m, s2 = x.mean(0), x.var(0, unbiased=False)
    if lags is None or pairs is None:
        c, p = x[:0], x.new_zeros(0)
    else:
        c, p = autocov(x, lags), pairs.to(x)
    return float(expected_epps_pulley(m, s2, c, p, n).mean())


class _Sigreg:
    """The rows of ``sigreg`` (module docstring) of a code for the run
    ``conf`` on the data at ``root``."""

    def __init__(
        self,
        conf: RunConf,
        root: Path,
        *,
        step: int,
        sizes: Sequence[int],
        heldin: tuple[int, int],
    ) -> None:
        self._conf, self._dev = conf, torch.device(conf.device)
        self._sizes = sizes
        self.grid = ZurichGrid(conf, root, step)
        self._val = eval_splits(conf, root)["val"][:, : conf.data.tours.frames]
        # the held-in batches of CodeStats: the same seed, the same draws
        units = Units(conf, root, "all", seed=conf.seed + HELDIN_SEED)
        self._heldin = [units.draw(heldin[1]) for _ in range(heldin[0])]

    def _dirs(self, d: int) -> Tensor:
        gen = torch.Generator().manual_seed(self._conf.seed + EVAL_DIRECTION_SEED)
        return directions(EVAL_DIRECTIONS, d, gen).to(self._dev)

    @torch.no_grad()
    def rows(self, mu: Callable[[Tensor], Tensor]) -> list[dict[str, Any]]:
        z = self.grid.codes(mu, self._dev)  # (G, D)
        a = self._dirs(z.shape[1])
        rows: list[dict[str, Any]] = []

        def row(name: str, n: int, groups: list[Tensor], **kw: Any) -> None:
            s = torch.tensor([float(sigreg(zg, zg[:0], a)) for zg in groups])
            sd = float(s.std()) if len(s) > 1 else float("nan")
            rows.append(
                {"set": name, "n": n, "groups": len(groups)}
                | {"sigreg": float(s.mean()), "sigreg_sd": sd}
                | kw
            )

        for n in self._sizes:
            if len(z) // n < 1:
                continue
            used = z[: len(z) // n * n]  # the grid's stacks of the groups
            x = used.double() @ a.double().T  # (P n, M)
            for layout in LAYOUTS:
                groups = partition(len(z), n, layout)
                lags, pairs = lag_pairs(groups)
                lags = lags.to(x.device)
                rho = autocov(x, lags[:1])[0] / x.var(0, unbiased=False)
                row(
                    layout,
                    n,
                    [used[g.to(used.device)] for g in groups],
                    spacing=int(lags[0]) * self.grid.step,  # video frames
                    rho=float(rho.mean()),  # of the neighbours, over the a
                    pred=_predict(x, n, lags, pairs),
                    pred_iid=_predict(x, n),
                )

        def proj(c: Tensor) -> Tensor:
            return c.double() @ a.double().T

        zv = _encode(mu, self._val, self._dev)
        row("val", len(zv), [zv], pred_iid=_predict(proj(zv), len(zv)))
        zh = [_encode(mu, x, self._dev) for x in self._heldin]
        every, b = proj(torch.cat(zh)), len(zh[0])
        row("heldin", b, zh, pred_iid=_predict(every, b))
        for n in self._sizes:
            k = n // b  # held-in batches a group
            if n > b and k * b == n and len(zh) >= k:
                cut = range(0, len(zh) - k + 1, k)
                row(
                    "heldin",
                    n,
                    [torch.cat(zh[i : i + k]) for i in cut],
                    pred_iid=_predict(every, n),
                )
        return rows


def evaluate_sigreg(
    run_dir: Path,
    *,
    root: Path | None = None,
    steps: Sequence[int] | None = None,
    step: int = GRID_STEP,
    sizes: Sequence[int] = GROUP_SIZES,
    heldin: tuple[int, int] = (HELDIN_BATCHES, HELDIN_BATCH),
) -> Path:
    """The rows of ``sigreg`` (module docstring) of the $z$ of the
    checkpoints of ``steps`` (by default all) of the JEPA run in
    ``run_dir``, the grid every ``step`` video frames, groups of ``sizes``,
    into ``run_dir / OUT``. Returns the directory."""
    conf = _conf(run_dir)
    determinism = configure_determinism(conf.device)
    ckpts = run_ckpts(run_dir)
    chosen = _pick(ckpts, steps, ())
    root = tours_root(conf.data.tours.data_version) if root is None else root
    out = run_dir / OUT
    out.mkdir(parents=True, exist_ok=True)
    diag = _Sigreg(conf, root, step=step, sizes=sizes, heldin=heldin)
    write_json(
        out / "sigreg.json",
        _settings(
            steps=chosen,
            grid_step=step,
            grid=len(diag.grid),
            sizes=list(sizes),
            heldin=list(heldin),
            eval_directions=EVAL_DIRECTIONS,
            determinism=determinism,
        ),
    )
    (out / "sigreg.jsonl").write_text("")
    mods = _jepa(run_dir, conf)
    for s in chosen:
        head = ckpt.load_eval(ckpts[s], mods.modules())
        stamp = {"t": head["t"], "step": s}
        append_jsonl(out / "sigreg.jsonl", [stamp | r for r in diag.rows(mods.mu)])
    return out


# -- question 2: Imagenette ------------------------------------------------------


def shrink(images: Tensor, side: int = IMAGE_SIDE) -> Tensor:
    """The u8 images ``(B, 3, S, S)`` at ``side`` u8 ``(B, 3, side,
    side)``: the mean of each $f \\times f$ block, $f$ = S / ``side``,
    rounded.

    Raises:
        ValueError: The images are not square of a multiple of ``side``.
    """
    s = images.shape[-1]
    f = s // side
    if images.shape[-2] != s or f * side != s:
        raise ValueError(f"images of {tuple(images.shape[-2:])} to {side}")
    if f == 1:
        return images
    return torch.cat(
        [
            nn.functional.avg_pool2d(c.float(), f).round().to(torch.uint8)
            for c in images.split(1024)
        ]
    )


def still_stacks(images: Tensor, n: int) -> Tensor:
    """The images ``(B, 3, H, W)`` each repeated into a stack of ``n``
    frames ``(B, n, 3, H, W)``."""
    return images[:, None].repeat(1, n, 1, 1, 1)


def evaluate_imagenette(
    data: Path,
    out: Path,
    *,
    conf: RunConf,
    runs: Sequence[Path] = (),
    steps: Sequence[int] | None = None,
    v16: Path | None = None,
    v17: Path | None = None,
    side: int = IMAGE_SIDE,
    codes: Sequence[str] | None = None,
) -> Path:
    """The linear probe of Imagenette at ``data`` (module docstring) of
    ``color`` (of ``conf``'s patch), ``v16`` and ``v17`` given, and the
    ``codes`` (``TokenCode``'s pools or ``"z_w"``; by default
    ``IMAGENETTE_POOLS`` of each run's SIGReg's data point) of the
    checkpoints of ``steps`` (by default all) of each of ``runs``, on
    ``conf.device``, into ``out``. Returns ``out``."""
    dev = torch.device(conf.device)
    determinism = configure_determinism(conf.device)
    split = {s: torch.load(data / f"{s}.pt") for s in ("train", "val")}
    classes = len(split["train"]["classes"])
    images = {s: shrink(v["x"], side) for s, v in split.items()}
    n = conf.data.tours.frames
    out.mkdir(parents=True, exist_ok=True)
    write_json(
        out / "imagenette.json",
        _settings(
            data=str(data),
            side=side,
            frames=n,
            images={s: len(x) for s, x in images.items()},
            classes=classes,
            runs=[str(r) for r in runs],
            v16=None if v16 is None else str(v16),
            v17=None if v17 is None else str(v17),
            determinism=determinism,
        ),
    )
    (out / "imagenette.jsonl").write_text("")

    def probe(code: Code) -> float:
        f = {
            s: _encode(lambda c: code.mu(still_stacks(c, n)), x, dev).double().cpu()
            for s, x in images.items()
        }
        ys = {s: v["y"] for s, v in split.items()}
        return probe_accuracy(f["train"], ys["train"], f["val"], ys["val"], classes)

    def write(row: dict[str, Any], code: Code) -> None:
        with torch.no_grad():
            acc = probe(code)
        append_jsonl(out / "imagenette.jsonl", [row | {"acc": acc}])

    write({"code": "color"}, ColorCode(conf.model.patch))
    for name, path in (("v16", v16), ("v17", v17)):
        if path is not None:
            write({"code": name}, vae_at(path, conf.device))
    for run in runs:
        rc = replace(_conf(run), device=conf.device)
        mods = _jepa(run, rc)
        pools = IMAGENETTE_POOLS[_settings_of(run).get("sigreg_on", "flat")]
        pools = pools if codes is None else codes
        ckpts = run_ckpts(run)
        w_code = RandProjCode(mods)  # W made once for the run's checkpoints
        for s in _pick(ckpts, steps, ()):
            head = ckpt.load_eval(ckpts[s], mods.modules())
            for pool in pools:
                row = {"run": run.name, "step": s, "t": head["t"], "code": pool}
                write(row, w_code if pool == "z_w" else TokenCode(mods, pool))
    return out


# -- the command line ------------------------------------------------------------


def _ints(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split(","))


def main(argv: Sequence[str] | None = None) -> Path:
    """``python -m exp.vision.jepa_diag`` (module docstring). Returns the
    output directory."""
    ap = argparse.ArgumentParser(
        prog="python -m exp.vision.jepa_diag",
        description="Diagnose V18's JEPA runs on their checkpoints.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    tok = sub.add_parser("tokens", help="question 1: the probes of h")
    tok.add_argument("run_dir", type=Path)
    tok.add_argument("--codes", type=lambda s: tuple(s.split(",")), default=POOLS)
    tok.add_argument("--span", type=float, default=SPAN)
    tok.add_argument("--fit-steps", type=int, default=FIT_STEPS, help="per decoder")
    sig = sub.add_parser("sigreg", help="question 3: SIGReg of spaced stacks")
    sig.add_argument("run_dir", type=Path)
    sig.add_argument("--grid-step", type=int, default=GRID_STEP, help="frames")
    sig.add_argument("--sizes", type=_ints, default=GROUP_SIZES)
    img = sub.add_parser("imagenette", help="question 2: the linear probe")
    img.add_argument("--data", type=Path, required=True, help="prep's train/val.pt")
    img.add_argument("--out", type=Path, required=True)
    img.add_argument("--runs", type=Path, nargs="*", default=[])
    img.add_argument("--v16", type=Path, help="a VAE checkpoint")
    img.add_argument("--v17", type=Path, help="a VAE checkpoint")
    img.add_argument("--config", type=Path, help="(default: v16_config)")
    img.add_argument("--device", default="cuda")
    img.add_argument(
        "--codes",
        type=lambda s: tuple(s.split(",")),
        help="comma-separated codes of the runs (default: IMAGENETTE_POOLS)",
    )
    for p in (tok, sig, img):
        p.add_argument("--steps", type=_ints, help="comma-separated steps")
    for p in (tok, sig):
        p.add_argument("--root", type=Path, help="the data (default: tours_root)")
    args = ap.parse_args(argv)
    if args.cmd == "tokens":
        return evaluate_tokens(
            args.run_dir,
            root=args.root,
            span=args.span,
            steps=args.steps,
            pools=args.codes,
            fit_steps=args.fit_steps,
        )
    if args.cmd == "sigreg":
        return evaluate_sigreg(
            args.run_dir,
            root=args.root,
            steps=args.steps,
            step=args.grid_step,
            sizes=args.sizes,
        )
    d = v16_config() if args.config is None else yaml.safe_load(args.config.read_text())
    d["device"] = args.device
    return evaluate_imagenette(
        args.data,
        args.out,
        conf=from_dict(d),
        runs=args.runs,
        steps=args.steps,
        v16=args.v16,
        v17=args.v17,
        codes=args.codes,
    )


if __name__ == "__main__":
    print(main())
