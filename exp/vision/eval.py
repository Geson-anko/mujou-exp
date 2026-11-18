r"""The evaluation of a run's checkpoints afterwards (V0, plans/v0-foundation.md
「評価」): with the VAE held, a decoder of its code learned on the training
videos, the linear dynamics on the code, and the VAE's own decoder, measured
on the held-out video.

The model is seen through ``VaeCode`` only: the code $z = \mu(x)$ of a batch
of stacks and the system's reconstruction $\hat x = g(\mu)$ of the stacks.

Data. The training windows are ``stream_windows``: ``windows`` windows of $n
+$ ``steps`` frames (``data.tours.stride`` apart) evenly spaced over the loop
of the training videos (all but ``data.tours.held_out``), none across two
videos; window ``j`` has ``steps + 1`` stacks, stack ``k`` its frames ``k ..
k + n - 1``. The held-out windows are ``eval_splits``: ``3 m`` windows of $n
+$ ``HORIZON`` frames of the held-out video (``walking_tours.eval_windows``
of ``TOURS_SEED``, non-overlapping, in time order), $m$ = ``measure.clips``
x ``measure.windows``, cut into thirds in time order: calib, val, test.

A row of a split (val; test too with ``ctl.eval_test``) has

- ``stack_mse`` (the VAE only): $\sum_i \lVert x_i/255 - g(\mu(x_i))
  \rVert^2 / \sum_i \lVert x_i/255 - m \rVert^2$ over the first stack $x_i$
  of each window, $m$ the mean stack of the training stacks (1 is the mean
  stack, 0 a perfect reconstruction);
- ``recon`` and ``recon_blur``, the mean over ``measure.posthoc_seeds``
  decoders learned afterwards (``recon_sd``, ``recon_blur_sd`` their sample
  standard deviation, ``recon_k``, ``recon_blur_k`` each seed's): a
  ``decoder`` of $z$ standardized by one scalar (``_Standardizer``) fit to
  the last frames of the training stacks (``fit_decoder``, ``DECODER_LR``,
  seed $k$ from the generator of ``seed + DECODER_SEED + SEED_STEP k``),
  decoding the first stack's $z$ of each window; ``recon`` its squared error
  against the stack's last frame over that of the mean image (the mean of the
  training stacks' last frames), ``recon_blur`` the same after a Gaussian
  blur of ``measure.blur_sigma`` pixels (``blur_error.blur_recon``);
- ``dyn_1``, ``copy_1``, ``dyn_1/copy_1``, ``dynw_1``, ``copyw_1``
  (``linear_dyn.errors``): $z_{k+1} = A z_k + b$ fit on the consecutive
  training stacks, its ridge ``alpha`` chosen on the calib windows
  (``linear_dyn.select_alpha``), on the split's windows;
- ``rank``: the effective rank of the first stacks' $z$
  (``metrics.effective_rank``).

The yardstick (``Evaluator.baseline``, ``code`` ``"pca"``): the pixel PCA of
$D$ = ``model.latent`` components of the training stacks' last frames
(``exp.lib.pca``, from the generator of ``seed + PCA_SEED``) as the code of a
stack's last frame, through the same rows but ``stack_mse``, with
``recon_lin`` and ``recon_lin_blur`` of its linear decoder.

Pictures of val (``SHOWN`` windows, ``SCALE`` times larger): ``<name>.png``,
rows the true last frames, the system's (the VAE's $g(\mu)$, the PCA's
linear decoder), and those of the decoder learned afterwards with seed 0.

``evaluate_run`` evaluates every ``ckpt/<t>.pt`` of a run directory into
``posthoc/`` (``baselines.jsonl``, ``evals.jsonl``, a row per checkpoint and
split, the pictures and ``config.json``) under
``provenance.configure_determinism``; from the shell, for a run of the
program ``"vae"`` (``exp.vision.models.build_vae``)::

    python -m exp.vision.eval <run_dir> [--root <data>] [--windows W] [--fit-steps S]

``check_splits`` tells before a run whether the held-out video has the
windows of its splits.
"""

import argparse
import bisect
import json
import math
import statistics
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import torch
import yaml
from torch import Tensor, nn

from exp.lib import ckpt, linear_dyn, pca
from exp.lib.blur_error import blur_recon
from exp.lib.metrics import effective_rank
from exp.lib.optim import adam
from exp.lib.png import tile, write_png
from exp.lib.provenance import configure_determinism
from exp.lib.rundir import atomic_open, write_json
from exp.lib.tours.stacks import tours_root
from exp.lib.tours.walking_tours import eval_windows, videos
from exp.lib.video_io import U8Reader
from exp.vision.compose import from_dict
from exp.vision.schema import RunConf

__all__ = [
    "BATCH",
    "DECODER_LR",
    "DECODER_SEED",
    "FIT_BATCH",
    "FIT_STEPS",
    "HORIZON",
    "PCA_SEED",
    "POSTHOC",
    "SCALE",
    "SEED_STEP",
    "SHOWN",
    "SPLITS",
    "STREAM_STEPS",
    "TOURS_SEED",
    "WINDOWS",
    "Evaluator",
    "VaeCode",
    "check_splits",
    "decoder",
    "eval_splits",
    "evaluate_run",
    "fit_decoder",
    "main",
    "stream_windows",
]

TOURS_SEED = 4242  # the windows of the held-out video (as exp/video)
DECODER_SEED = 3  # added to the run's seed: the decoders learned afterwards
PCA_SEED = 4  # added to the run's seed: the randomized SVD of the yardstick
SEED_STEP = 1000  # between the seeds of two decoders of one code
# At 1e-3 the decoder's fit died to a constant image on some codes (M17); 3e-4
# fit them all.
DECODER_LR = 3e-4
FIT_STEPS, FIT_BATCH = 2000, 256  # of each decoder's fit
WINDOWS, STREAM_STEPS = 2048, 8  # the training windows and their stacks after the first
HORIZON = 1  # stacks after the first in a held-out window: dyn_1
BATCH = 256  # stacks encoded or decoded at once
SHOWN, SCALE = 10, 2  # val windows in a picture, and its upscale
SPLITS = ("calib", "val", "test")
POSTHOC = "posthoc"  # the directory of evaluate_run in the run's
_BASE = 32  # the channels of the decoder's last transposed convolution


class VaeCode(Protocol):
    """The VAE as the evaluation sees it: its modules on the run's device
    (``modules`` names them as the run's checkpoints do, for
    ``ckpt.load_eval``), its code and its reconstruction. Both are called
    under ``torch.no_grad`` and must not draw random numbers."""

    def modules(self) -> Mapping[str, nn.Module]:
        """The modules whose weights an evaluation checkpoint holds."""
        ...

    def mu(self, x: Tensor, /) -> Tensor:
        """The code $\\mu$ ``(B, D)`` of the stacks ``x`` u8 ``(B, n, 3, H,
        W)`` (on the modules' device)."""
        ...

    def decode(self, mu: Tensor, /) -> Tensor:
        """The system's reconstruction ``(B, n, 3, H, W)`` of the stacks in
        the units of ``x / 255`` (in ``[0, 1]``) from the code ``mu`` ``(B,
        D)``."""
        ...


# -- data ------------------------------------------------------------------------


def eval_splits(conf: RunConf, root: Path) -> dict[str, Tensor]:
    """The held-out windows u8 ``(m, n + HORIZON, 3, S, S)`` of calib, val
    and test (module docstring), on the CPU.

    Raises:
        ValueError: The held-out video has too few slots for them
            (``eval_windows``), or its frames are not ``data.tours.size``.
    """
    t, m = conf.data.tours, conf.measure.clips * conf.measure.windows
    u8 = eval_windows(
        root,
        held_out=t.held_out,
        clips=len(SPLITS) * m,
        frames=t.frames,
        stride=t.stride,
        steps=HORIZON,
        seed=TOURS_SEED,
    )
    if u8.shape[-1] != t.size:
        raise ValueError(f"the data is of {u8.shape[-1]} pixels, not {t.size}")
    return {s: u8[i * m : (i + 1) * m] for i, s in enumerate(SPLITS)}


def check_splits(conf: RunConf, root: Path) -> None:
    """Check, without reading the frames, that the held-out video at ``root``
    has the ``3 m`` windows of ``eval_splits`` (as ``eval_windows`` counts its
    slots of ``(n + HORIZON - 1) stride + 1`` frames).

    Raises:
        ValueError: It has fewer slots.
    """
    t, m = conf.data.tours, conf.measure.clips * conf.measure.windows
    span = (t.frames + HORIZON - 1) * t.stride + 1
    frames = len(U8Reader(root / t.held_out))
    if len(SPLITS) * m > frames // span:
        raise ValueError(
            f"{t.held_out} has {frames} frames, {frames // span} windows of "
            f"{span}: fewer than the {len(SPLITS)} x {m} of the splits; lower "
            "measure.clips or measure.windows"
        )


def stream_windows(
    conf: RunConf, root: Path, *, windows: int = WINDOWS, steps: int = STREAM_STEPS
) -> Tensor:
    """About ``windows`` training windows u8 ``(w, n + steps, 3, S, S)``
    (module docstring): with the training videos joined in name order into a
    loop of $L$ frames, window ``j`` starts at frame $\\lfloor j L /
    \\mathrm{windows} \\rfloor$ and its frame ``i`` is ``i stride`` frames
    later; a window that would cross the end of its video is left out.

    Raises:
        ValueError: No window fits in a video.
    """
    got = list(_iter_stream_windows(conf, root, windows=windows, steps=steps))
    if not got:
        raise ValueError(f"no training window of {conf.data.tours.frames + steps}")
    return torch.stack(got)


def _iter_stream_windows(
    conf: RunConf, root: Path, *, windows: int, steps: int
) -> Iterator[Tensor]:
    t = conf.data.tours
    readers = [U8Reader(root / v) for v in videos(root) if v != t.held_out]
    lengths = [len(r) for r in readers]
    firsts = [sum(lengths[:j]) for j in range(len(lengths))]
    loop = sum(lengths)
    span = (t.frames + steps - 1) * t.stride + 1
    for j in range(windows):
        g = j * loop // windows
        a = bisect.bisect_right(firsts, g) - 1
        local = g - firsts[a]
        if local + span <= lengths[a]:
            yield readers[a][local : local + span : t.stride]


def _stacks(windows: Tensor, n: int) -> Tensor:
    """The stacks ``(w, K, n, C, H, W)`` of the windows ``(w, n + K - 1, C, H,
    W)``: stack ``k`` is frames ``k .. k + n - 1``."""
    k = windows.shape[1] - n + 1
    return torch.stack([windows[:, i : i + n] for i in range(k)], 1)


def _mean_stack(windows: Tensor, n: int) -> Tensor:
    """The mean ``(n, C, H, W)`` float64, in ``x / 255``, of the stacks of
    the windows u8 ``(w, n + K - 1, C, H, W)``: slot ``s`` of the ``K``
    stacks of a window is its frames ``s .. s + K - 1``, a difference of
    cumulative sums (no copy of the stacks)."""
    k = windows.shape[1] - n + 1
    total = torch.zeros(n, *windows.shape[2:], dtype=torch.float64)
    for chunk in windows.split(64):
        c = chunk.double().sum(0).cumsum(0)  # (n + K - 1, C, H, W)
        c = torch.cat([torch.zeros_like(c[:1]), c])
        total += c[k : k + n] - c[:n]
    return total / (len(windows) * k * 255)


# -- the decoder learned afterwards ----------------------------------------------


def decoder(d_z: int, shape: tuple[int, int, int]) -> nn.Module:
    """$z$ ``(B, d_z)`` to frames ``(B, C, H, W)`` for ``shape = (C, H, W)``
    with ``H = W`` a power of 2 from 8: a linear map to 4 x 4 maps of ``32 *
    2^u`` channels, then ``u = log2(H / 4)`` transposed convolutions that
    double the side and halve the channels (ReLU before each), then a 3 x 3
    convolution to ``C`` channels (exp/video's ``posthoc.decoder``).

    Raises:
        ValueError: ``H`` is not such a side, or ``W != H``.
    """
    c, h, w = shape
    if h != w or h < 8 or h & (h - 1):
        raise ValueError(f"frames of a side that is a power of 2 from 8, not {h}x{w}")
    ups = int(math.log2(h // 4))
    width = _BASE * 2**ups
    layers: list[nn.Module] = [
        nn.Linear(d_z, width * 16),
        nn.Unflatten(1, (width, 4, 4)),
    ]
    for _ in range(ups):
        layers += [nn.ReLU(), nn.ConvTranspose2d(width, width // 2, 4, 2, 1)]
        width //= 2
    layers += [nn.ReLU(), nn.Conv2d(width, c, 3, padding=1)]
    return nn.Sequential(*layers)


def fit_decoder(
    z: Tensor,
    images: Tensor,
    *,
    steps: int = FIT_STEPS,
    batch: int = FIT_BATCH,
    lr: float = DECODER_LR,
    gen: torch.Generator,
) -> nn.Module:
    """A ``decoder`` of ``z`` ``(N, D)`` to ``images`` u8 ``(N, C, H, W)``
    (read as ``x / 255``; one device), fit by ``steps`` steps of
    ``adam(lr)`` on the mean squared error of ``batch`` pairs drawn with
    replacement.

    ``gen`` (a CPU generator) draws the seed of the initial weights (built on
    the CPU, so they do not depend on the device; the global RNG is left as
    it was), then the batches. The gradient is autograd's within the decoder
    alone: it is a measurement, outside the run's network.
    """
    seed = int(torch.randint(2**62, (1,), generator=gen))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        dec = decoder(z.shape[1], (images.shape[1], images.shape[2], images.shape[3]))
    dec = dec.to(z.device)
    params = list(dec.parameters())
    optimizer = adam(lr).bind(params)
    z, images = z.detach(), images.detach()
    for _ in range(steps):
        idx = torch.randint(len(z), (batch,), generator=gen).to(z.device)
        with torch.enable_grad():
            loss = ((dec(z[idx]) - images[idx].float() / 255) ** 2).mean()
            grads = torch.autograd.grad(loss, params)
        optimizer.step_(grads, 1.0)
    return dec


class _Standardizer:
    r"""$(z - \bar z) / s$: $\bar z$ the mean of each component of the
    training codes ``z`` ``(N, D)``, $s$ one scalar for all of them, the root
    mean square of their stds. One scale, not one per component: the PCA's
    last components, each scaled up to std 1, drowned the first and the
    decoder's fit died to a constant image (M17)."""

    def __init__(self, z: Tensor) -> None:
        z = z.float()
        self._mean = z.mean(0)
        self._sd = max(float(z.var(0).mean().sqrt()), 1e-12)

    def __call__(self, z: Tensor) -> Tensor:
        return (z.float() - self._mean) / self._sd


def _decoded(dec: nn.Module, z: Tensor) -> Tensor:
    """The frames ``(B, C, H, W)`` of ``dec`` of ``z`` ``(B, D)``, a batch at
    a time, clamped to ``[0, 1]``."""
    return torch.cat([dec(c) for c in z.split(BATCH)]).clamp(0, 1)


def _mse(a: Tensor, b: Tensor) -> float:
    return float(((a - b) ** 2).double().mean())


def _sd(values: Sequence[float]) -> float:
    """The sample standard deviation, NaN of one value."""
    return statistics.stdev(values) if len(values) > 1 else math.nan


# -- the evaluation ----------------------------------------------------------------


class Evaluator:
    """The rows of a code (module docstring) of the run ``conf`` on the data
    at ``root``; its pictures into ``out``.

    Args:
        conf: The run's config.
        root: The prepared Walking Tours.
        out: The directory of the pictures.
        windows: The training windows (``stream_windows``).
        steps: Their stacks after the first.
        fit_steps: The steps of each decoder's fit.
    """

    def __init__(
        self,
        conf: RunConf,
        root: Path,
        out: Path,
        *,
        windows: int = WINDOWS,
        steps: int = STREAM_STEPS,
        fit_steps: int = FIT_STEPS,
    ) -> None:
        self._conf, self._out, self._fit_steps = conf, out, fit_steps
        self._dev = torch.device(conf.device)
        self._n = n = conf.data.tours.frames
        self._measured = ("val", "test") if conf.ctl.eval_test else ("val",)
        self._windows = eval_splits(conf, root)  # u8, CPU
        # The first stack's last frame of each held-out window, in [0, 1].
        self._f0 = {
            s: self._windows[s][:, n - 1].to(self._dev).float() / 255
            for s in self._measured
        }
        self._train = stream_windows(conf, root, windows=windows, steps=steps)
        # The last frames (window, stack) of the training stacks, u8.
        self._targets = self._train[:, n - 1 :].flatten(0, 1).to(self._dev)
        self._mean_stack = _mean_stack(self._train, n).float().to(self._dev)
        self._mean = self._mean_stack[-1]  # (C, H, W): the mean last frame
        out.mkdir(parents=True, exist_ok=True)

    def settings(self) -> dict[str, Any]:
        """What the rows depend on beside the run's config."""
        m = self._conf.measure
        return {
            "train_windows": len(self._train),
            "train_stacks": len(self._targets),
            "split_windows": len(self._windows["val"]),
            "tours_seed": TOURS_SEED,
            "horizon": HORIZON,
            "fit_steps": self._fit_steps,
            "fit_batch": FIT_BATCH,
            "lr": DECODER_LR,
            "decoder_seeds": [
                self._conf.seed + DECODER_SEED + SEED_STEP * k
                for k in range(m.posthoc_seeds)
            ],
            "pca_seed": self._conf.seed + PCA_SEED,
            "blur_sigma": m.blur_sigma,
            "splits": list(self._measured),
        }

    @torch.no_grad()
    def evaluate(self, code: VaeCode, name: str) -> list[dict[str, Any]]:
        """The VAE's rows (``code`` ``"vae"``) and its picture ``<name>.png``."""
        n = self._n
        rows, zs, post = self._rows(code.mu)
        own = None
        for row in rows:
            split = row["split"]
            z0 = zs[split][:, 0]
            first = self._windows[split][:, :n]
            err = base = 0.0
            for zc, xc in zip(z0.split(BATCH), first.split(BATCH), strict=True):
                x = xc.to(self._dev).float() / 255
                x_hat = code.decode(zc)
                err += float(((x_hat - x) ** 2).double().sum())
                base += float(((self._mean_stack - x) ** 2).double().sum())
                if split == "val" and own is None:
                    own = x_hat[:SHOWN, -1].clamp(0, 1)
            row["stack_mse"] = err / base
        assert own is not None
        self._picture(name, own, post)
        return [{"code": "vae", **r} for r in rows]

    @torch.no_grad()
    def baseline(self) -> list[dict[str, Any]]:
        """The PCA's rows (``code`` ``"pca"``) and its picture ``pca.png``."""
        conf = self._conf
        gen = torch.Generator().manual_seed(conf.seed + PCA_SEED)
        flat = self._targets.flatten(1).float() / 255
        mean, comps = pca.fit_pca(flat, conf.model.latent, gen=gen)
        del flat

        def z_of(x: Tensor) -> Tensor:
            return pca.encode(x[:, -1].flatten(1).float() / 255, mean, comps)

        rows, zs, post = self._rows(z_of)
        sigma = conf.measure.blur_sigma
        lins = {}
        for row in rows:
            f0 = self._f0[row["split"]]
            lin = pca.decode(zs[row["split"]][:, 0], mean, comps)
            lin = lin.view_as(f0).clamp(0, 1)
            row["recon_lin"] = _mse(lin, f0) / _mse(self._mean.expand_as(f0), f0)
            row["recon_lin_blur"] = blur_recon(lin, f0, self._mean, sigma)
            lins[row["split"]] = lin
        self._picture("pca", lins["val"][:SHOWN], post)
        return [{"code": "pca", **r} for r in rows]

    def _codes(self, z_of: Callable[[Tensor], Tensor], windows: Tensor) -> Tensor:
        """The codes ``(w, K, D)`` on the device of the stacks of the windows
        u8 ``(w, n + K - 1, C, H, W)`` (CPU), a batch of windows at a time."""
        w, k = len(windows), windows.shape[1] - self._n + 1
        per = max(1, BATCH // k)
        zs = []
        for i in range(0, w, per):
            x = _stacks(windows[i : i + per].to(self._dev), self._n).flatten(0, 1)
            zs.append(z_of(x).float())
        return torch.cat(zs).unflatten(0, (w, k))

    def _rows(
        self, z_of: Callable[[Tensor], Tensor]
    ) -> tuple[list[dict[str, Any]], dict[str, Tensor], Tensor]:
        """The rows of the code ``z_of`` but ``stack_mse`` (module docstring),
        its codes ``(m, 1 + HORIZON, D)`` of each split, and the first
        ``SHOWN`` frames of val of the decoder of seed 0."""
        conf = self._conf
        z_train = self._codes(z_of, self._train)  # (w, K, D)
        zs = {s: self._codes(z_of, w) for s, w in self._windows.items()}
        flat = z_train.flatten(0, 1)
        std = _Standardizer(flat)
        recon: dict[str, list[float]] = {s: [] for s in self._measured}
        blur: dict[str, list[float]] = {s: [] for s in self._measured}
        post = None
        for k in range(conf.measure.posthoc_seeds):
            seed = conf.seed + DECODER_SEED + SEED_STEP * k
            dec = fit_decoder(
                std(flat),
                self._targets,
                steps=self._fit_steps,
                lr=DECODER_LR,
                gen=torch.Generator().manual_seed(seed),
            )
            for s in self._measured:
                f0 = self._f0[s]
                x_hat = _decoded(dec, std(zs[s][:, 0]))
                recon[s].append(_mse(x_hat, f0) / _mse(self._mean.expand_as(f0), f0))
                blur[s].append(
                    blur_recon(x_hat, f0, self._mean, conf.measure.blur_sigma)
                )
                if s == "val" and k == 0:
                    post = x_hat[:SHOWN]
        assert post is not None
        z0, z1 = z_train[:, :-1].flatten(0, 1), z_train[:, 1:].flatten(0, 1)
        alpha = linear_dyn.select_alpha(z0, z1, zs["calib"])
        a, b = linear_dyn.fit(z0, z1, alpha=alpha)
        rows = []
        for s in self._measured:
            row: dict[str, Any] = {
                "split": s,
                "recon": statistics.fmean(recon[s]),
                "recon_sd": _sd(recon[s]),
                "recon_blur": statistics.fmean(blur[s]),
                "recon_blur_sd": _sd(blur[s]),
                "recon_k": recon[s],
                "recon_blur_k": blur[s],
            }
            row |= linear_dyn.errors(a, b, zs[s], [HORIZON], train=z0)
            row["alpha"] = alpha
            row["rank"] = effective_rank(zs[s][:, 0])
            rows.append(row)
        return rows, zs, post

    def _picture(self, name: str, own: Tensor, post: Tensor) -> None:
        """``<name>.png``: the true last frames of val, ``own`` and ``post``
        ``(SHOWN, C, H, W)``."""
        true = self._f0["val"][:SHOWN]
        grid = torch.stack([true, own, post]).cpu()
        write_png(self._out / f"{name}.png", tile(grid, scale=SCALE))


def evaluate_run(
    run_dir: Path,
    build: Callable[[RunConf], VaeCode],
    *,
    root: Path | None = None,
    out: Path | None = None,
    windows: int = WINDOWS,
    steps: int = STREAM_STEPS,
    fit_steps: int = FIT_STEPS,
) -> Path:
    """Evaluate every checkpoint ``ckpt/<t>.pt`` of the run in ``run_dir``
    (its ``config.yaml``; the VAE of ``build(conf)``, on ``conf.device``)
    in the order of ``t``, under ``provenance.configure_determinism``, into
    ``out`` (``run_dir / POSTHOC``; module docstring): ``config.json``,
    ``baselines.jsonl`` (the PCA), ``evals.jsonl`` (a row per checkpoint and
    split, with its ``t`` and ``frame``, appended as each is done) and the
    pictures. Returns ``out``.

    Args:
        run_dir: The run.
        build: The VAE of a config, its weights to be loaded.
        root: The prepared Walking Tours (``tours_root`` of the run's data).
        out: The directory of the files.
        windows: The training windows (``stream_windows``).
        steps: Their stacks after the first.
        fit_steps: The steps of each decoder's fit.

    Raises:
        FileNotFoundError: The run has no checkpoint.
    """
    raw = yaml.safe_load((run_dir / "config.yaml").read_text())
    conf = from_dict(raw)
    determinism = configure_determinism(conf.device)
    paths = sorted(run_dir.glob("ckpt/*.pt"), key=lambda p: float(p.stem))
    if not paths:
        raise FileNotFoundError(f"no checkpoint under {run_dir / 'ckpt'}")
    t = conf.data.tours
    root = tours_root(t.data_version) if root is None else root
    out = run_dir / POSTHOC if out is None else out
    ev = Evaluator(conf, root, out, windows=windows, steps=steps, fit_steps=fit_steps)
    write_json(
        out / "config.json",
        ev.settings() | {"ckpts": [p.name for p in paths], "determinism": determinism},
    )
    _write(out / "baselines.jsonl", ev.baseline())
    _write(out / "evals.jsonl", [])
    code = build(conf)
    for path in paths:
        head = ckpt.load_eval(path, code.modules())
        rows = ev.evaluate(code, path.stem)
        _append(
            out / "evals.jsonl",
            [{"t": head["t"], "frame": head["frame"], **r} for r in rows],
        )
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
    """``python -m exp.vision.eval``: ``evaluate_run`` of a run of the
    program ``"vae"`` (module docstring). Returns its ``posthoc/``."""
    ap = argparse.ArgumentParser(
        prog="python -m exp.vision.eval",
        description="Evaluate the checkpoints of a vae run into <run_dir>/posthoc.",
    )
    ap.add_argument("run_dir", type=Path, help="the run's directory")
    ap.add_argument("--root", type=Path, help="the data (default: tours_root)")
    ap.add_argument("--windows", type=int, default=WINDOWS, help="training windows")
    ap.add_argument("--fit-steps", type=int, default=FIT_STEPS, help="per decoder")
    args = ap.parse_args(argv)
    from exp.vision.models import build_vae  # the model, only for the shell

    return evaluate_run(
        args.run_dir,
        lambda conf: build_vae(conf, seed=conf.seed),
        root=args.root,
        windows=args.windows,
        fit_steps=args.fit_steps,
    )


if __name__ == "__main__":
    print(main())
