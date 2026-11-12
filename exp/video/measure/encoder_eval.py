r"""The evaluation of an encoder of the packed layout afterwards (M17,
``task=posthoc`` and ``task=baselines``): with the encoder held, a decoder of
$z$, the linear dynamics on $z$ and a token decoder are learned on the
training videos and measured on the held-out one.

The units, a step each (``EncoderEval.step``):

- unit 0, the yardsticks (``baselines.jsonl``, ``pca*.png``, ``init*.png``):
  the pixel PCA of $D$ components of the last frame of each stack (``code``
  ``"pca"``, ``exp.lib.pca`` fit on the training stacks' last frames from the
  generator of ``seed + PCA_SEED``) and the ViT before it learns (``"init"``,
  ``vit.build_vision`` of the run's seed), through the evaluation below;
- unit ``1 + i``: checkpoint ``i`` of the run (``"vit"``; all of them, or the
  last only, ``ckpts``), its lines in ``metrics.jsonl`` and its pictures.

The code of a stack is $z$ (``measure.posthoc_code`` ``"z"``) or, with
``"tokens"`` (the tokens are the representation, ``vision.token_dim``), the
stack's $T$ target tokens flattened, $D = T d$ numbers: what the decoder of
$z$, the linear dynamics, ``rank``, ``z_std_min`` and ``z_spread`` below see
as $z$, for the ViT before it learns too; the PCA is then of $D$ components,
or of the most the training frames allow ($n - 1$ of $n$ frames), written
as ``pca_rank`` in its lines when fewer than $D$.

The evaluation of a code. The training windows are ``stream_windows``:
``windows`` windows of ``N + STREAM_STEPS`` frames ``vision.stride`` apart
from the loop of the training videos (none across two videos), each
``STREAM_STEPS + 1`` stacks; every stack has a $z$ and a last frame. A
``posthoc.decoder`` learns the last frames from $z$ centred on the training
stacks and divided by the RMS of its components' stds (``_Standardized``:
the PCA's components and a ViT's $z$ reach the decoder at the same scale;
``posthoc.fit_decoder``, from the generator of ``seed + DECODER_SEED``); ``exp.lib.linear_dyn`` fits
$z_{k+1} = A z_k + b$ on the consecutive stacks of the windows, its ridge
chosen on the calib windows of the held-out video (``linear_dyn.select_alpha``);
for a ViT, a token decoder (``restore.fit_token_decoder``, from the generator
after the decoder's fit) learns the last frame's patches from the tokens of
the last time slot. On the windows of val and test (``PackedMeasures``,
``measure.steps`` stacks after the first), a line has

- ``recon``: the squared error per pixel of the decoded $z$ of the first stack
  against its last frame, over the same of the mean image (the pixel mean of
  the training stacks' last frames: 1 is the mean image); ``recon_fit`` the
  same on the training stacks; ``recon_lin`` (pca) the linear PCA's;
  ``recon_blur`` and ``recon_lin_blur`` the same after a Gaussian blur of the
  error (``blur_error.blur_recon``, $\sigma$ = 2 px: a shift by one pixel
  costs little), the error M17 judges by;
- ``dyn_h``, ``copy_h``, ``dyn_h/copy_h``, and ``dynw_h``, ``copyw_h`` in
  the coordinates whitened by the training codes' covariance
  (``linear_dyn.errors``: every direction weighs the same, where ``dyn_h``
  favours a code whose variance changes slowly, as the PCA's) for
  ``measure.horizons``; ``alpha`` the ridge;
- ``img_dyn_h``: the decoded rollout $g(\hat z_h)$ against the true frame
  $h$ stacks ahead, over the mean image's error, and ``img_dyn_h/copy``, over
  the error of the first stack's last frame (the copy);
- ``restore_<view>_mask`` and ``restore_<view>_all`` (ViT): the restoration of
  each context view (``restore.restore``, the split's fixed masks) on the
  masked pixels and on the whole frame, over the mean image's error on the
  same pixels; ``restore_target_all`` the token decoder on the target view's
  tokens (its ceiling). The views are ``short`` and ``long`` for two
  context views, else ``c0``, ``c1``, ...;
- ``rank``, ``z_std_min``, ``z_spread`` of the first stacks' $z$;
- (ViT) the completion errors of the first stacks (``measure.completion``,
  ``PackedMeasures.completion_rows``): ``comp_<kind>``, ``comp_<kind>_vis``,
  ``comp_<kind>_mean``, ``comp_<kind>_pos`` (the reference's mean target
  token at each position: near ``comp_<kind>``, tokens of the position
  alone) and ``ncomp_*`` per mask kind (``short``, ``long``, ``future``), the
  line of val also of calib (``calib_comp_*``) and of the held-in stacks of
  the training videos (``fit_comp_*``);
- ``objective`` (val, ``vit``, minimized), by ``measure.encoder_objective``.
  ``"completion"`` (provisional): $\log$ of the mean of ``comp_<kind>`` over
  the kinds, at most ``PENALTY``, ``FLOOR`` below, ``PENALTY`` when $z$
  collapsed or a value is missing or not finite. ``"recon_dyn"`` (the
  default): $J = \log \max(\mathrm{recon} /
  \mathrm{recon}_{PCA},\ r_1 / r_{1, PCA})$ with $r_1$ = ``dyn_1/copy_1``
  and $\mathrm{recon}_{PCA}$ the smaller of the PCA's ``recon`` and
  ``recon_lin`` (its best known decoder), at
  most ``PENALTY`` ($\log 10$, sweep's ``DIVERGED`` too: a code ten times
  worse than the PCA ranks with one that collapsed or diverged, never
  below), a ratio below ``FLOOR`` counted as ``FLOOR``; ``PENALTY`` when $z$
  collapsed (``z_spread`` below ``COLLAPSE``) or a value is missing or not
  finite. The dynamics term is over the copy: a ViT codes a stack of $N$
  frames, of which consecutive stacks share $N - 1$, and the PCA the last
  frame alone, so any code of the stacks is smooth and its ``dyn_1`` small
  by construction (the ViT before it learns: 1/23 of the PCA's); the copy's
  error cancels that overlap.

Pictures of val (``SHOWN`` windows): ``<name>.png`` (the true last frames,
the decoded), ``<name>-restore.png`` (the true frames, the short context, its
restoration, the long context, its restoration, the target ceiling: a row
each) and ``<name>-dyn.png`` (``DYN_SHOWN`` windows, three rows each: the true
frames, $g(\hat z_h)$, $g(z_h)$, at $h = 0$ and each horizon), ``<name>`` the
checkpoint's ``<t>``, ``init`` or ``pca``.
"""

import bisect
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, override

import torch
from torch import Tensor, nn

from exp.lib import ckpt, linear_dyn, pca
from exp.lib.blur_error import blur_recon
from exp.lib.png import tile, write_png
from exp.lib.rundir import atomic_open, write_json
from exp.lib.video_io import U8Reader
from exp.video.data.walking_tours import videos
from exp.video.jepa.vit import VisionModules, build_vision
from exp.video.measure.objective import FLOOR, PENALTY
from exp.video.measure.posthoc import (
    BATCH,
    DECODER_SEED,
    FIT_BATCH,
    STEPS,
    Parent,
    as_float,
    fit_decoder,
)
from exp.video.measure.restore import (
    fit_token_decoder,
    last_slot,
    patches_of,
    picture_of,
    restore,
)
from exp.video.programs.vision import COLLAPSE, PackedMeasures, z_stats
from exp.video.schema.run import RunConf

__all__ = [
    "CKPTS",
    "DECODER_LR",
    "DYN_SHOWN",
    "PCA_SEED",
    "SHOWN",
    "STREAM_STEPS",
    "WINDOWS",
    "EncoderEval",
    "stream_windows",
]

PCA_SEED = 4  # added to the run's seed (DECODER_SEED 3)
# The rate of the decoder of z: at posthoc.LR (1e-3) its fit of 128 x 128
# frames died to a constant image on some codes (recon_fit 1.07 on a learned
# ViT, 2026-11-12); 3e-4 fit both that one (0.66) and the PCA-256 (0.15).
DECODER_LR = 3e-4
WINDOWS, STREAM_STEPS = 2048, 8  # the training windows and their stacks after the first
SHOWN, DYN_SHOWN = 8, 4  # windows of val in the pictures
CKPTS = ("all", "last")
SPLITS = ("val", "test")


def stream_windows(
    conf: RunConf, root: Path, *, windows: int = WINDOWS, steps: int = STREAM_STEPS
) -> Tensor:
    """About ``windows`` windows u8 ``(n, N + steps, C, S, S)`` of the
    training videos of ``conf`` (all but ``data.tours.held_out``, joined in
    name order into a loop of $L$ frames as the run's stream): window ``j``
    starts at frame $\\lfloor j L / \\mathrm{windows} \\rfloor$ of the loop and
    its frame ``i`` is ``i vision.stride`` frames later; a window that would
    cross the end of its video is left out."""
    t, v = conf.data.tours, conf.vision
    assert t is not None and v is not None
    readers = [U8Reader(root / name) for name in videos(root) if name != t.held_out]
    lengths = [len(r) for r in readers]
    firsts = [sum(lengths[:j]) for j in range(len(lengths))]
    loop = sum(lengths)
    span = (v.frames + steps - 1) * v.stride + 1
    out = []
    for j in range(windows):
        g = j * loop // windows
        a = bisect.bisect_right(firsts, g) - 1
        local = g - firsts[a]
        if local + span <= lengths[a]:
            out.append(readers[a][local : local + span : v.stride])
    return torch.stack(out)


def _stacks(windows: Tensor, n: int) -> Tensor:
    """The stacks ``(w, K, N, C, H, W)`` of the windows ``(w, N + K - 1, C, H,
    W)``: stack ``k`` is frames ``k .. k + N - 1``."""
    k = windows.shape[1] - n + 1
    return torch.stack([windows[:, i : i + n] for i in range(k)], 1)


def _mse(a: Tensor, b: Tensor) -> float:
    return float(((a - b) ** 2).mean())


def _masked_mse(a: Tensor, b: Tensor, m: Tensor) -> float:
    """The squared error per value of ``a`` and ``b`` ``(n, C, H, W)`` on the
    pixels ``m`` ``(n, H, W)``."""
    w = m[:, None].to(a)
    return float((((a - b) ** 2) * w).sum() / (w.sum() * a.shape[1]).clamp(min=1))


class _Code:
    """A code of the stacks to evaluate: ``z`` of the stacks ``(n, N, C, H,
    W)`` u8, and, for a ViT, its modules (the tokens, the restoration)."""

    def __init__(
        self,
        name: str,
        z: Any,
        mods: VisionModules | None = None,
        linear: tuple[Tensor, Tensor] | None = None,
    ) -> None:
        self.name, self._z, self.mods, self.linear = name, z, mods, linear

    @torch.no_grad()
    def z(self, stacks: Tensor) -> Tensor:
        return self._z(stacks)


class _Standardized(nn.Module):
    r"""``dec`` of $(z - \mu) / \sigma$: $\mu$ the mean of each component of
    the codes ``z`` ``(n, D)``, $\sigma$ one scalar for all of them, the root
    mean square of their stds (the codes reach the decoder at an RMS std of
    1). One scale, not one per component: the PCA's last components, each
    scaled up to std 1, drowned the first and the decoder's fit died to a
    constant image (``recon`` 1.09 against ``recon_lin`` 0.16, 2026-11-12)."""

    def __init__(self, z: Tensor) -> None:
        super().__init__()
        z = z.float()
        rms = float(z.var(0).mean().sqrt())
        self.mu: Tensor = z.mean(0)  # on the device of z, as the decoder
        self.sd: float = max(rms, 1e-12)
        self.dec: nn.Module | None = None

    def standardize(self, z: Tensor) -> Tensor:
        return (z.float() - self.mu) / self.sd

    @override
    def forward(self, z: Tensor) -> Tensor:
        assert self.dec is not None
        return self.dec(self.standardize(z))


class EncoderEval:
    """The evaluation of the checkpoints of ``parent`` (module docstring) into
    the directory ``out``, a unit per ``step``, from unit ``done`` (0: a fresh
    start, which writes ``config.json``; the lines of the units before ``done``
    are kept). With ``parent`` None (``task=baselines``) there is unit 0 only,
    of ``conf``.

    Args:
        parent: The run, or None.
        conf: The run's config (that of ``parent``, or of the baselines job).
        out: The directory of the files.
        root: The local copy of the data.
        windows: The training windows (``stream_windows``).
        steps: The steps of each decoder's fit.
        ckpts: ``"all"`` or ``"last"``: the checkpoints evaluated.
        done: The units done already.
    """

    def __init__(
        self,
        parent: Parent | None,
        conf: RunConf,
        out: Path,
        *,
        root: Path,
        windows: int = WINDOWS,
        steps: int = STEPS,
        ckpts: str = "all",
        done: int = 0,
    ) -> None:
        if ckpts not in CKPTS:
            raise ValueError(f"ckpts must be one of {CKPTS}, not {ckpts!r}")
        v = conf.vision
        if v is None or v.layout != "packed":
            raise ValueError(
                f"the run {conf.name!r} has no vision of the packed layout"
            )
        if max(conf.measure.horizons) > conf.measure.steps:
            raise ValueError(
                f"measure.horizons {conf.measure.horizons} beyond measure.steps "
                f"{conf.measure.steps}"
            )
        self.parent, self.conf, self.out = parent, conf, out
        paths = [] if parent is None else list(parent.ckpts)
        self._ckpts = paths[-1:] if ckpts == "last" else paths
        if not 0 <= done <= len(self):
            raise ValueError(f"done {done} of {len(self)} units")
        self.done, self._steps = done, steps
        self._dev = torch.device(conf.device)
        self._measures = PackedMeasures(conf)
        self._views = self._measures.views
        self._train = stream_windows(conf, root, windows=windows)  # u8, CPU
        # The last frames of the training stacks (window, stack), u8 on the device.
        self._targets = self._train[:, v.frames - 1 :].flatten(0, 1).to(self._dev)
        sums = torch.stack([as_float(c).sum(0) for c in self._targets.split(BATCH)])
        self._mean = (sums.sum(0) / len(self._targets))[None]  # (1, C, H, W)
        self._mods = build_vision(conf, seed=conf.seed)
        self._tokens = conf.measure.posthoc_code == "tokens"  # module docstring
        self._calib: dict[str, Tensor] = {}
        out.mkdir(parents=True, exist_ok=True)
        if done == 0:
            write_json(out / "config.json", self._config(windows, ckpts))
        kept = {"baselines.jsonl": 2 * len(SPLITS), "metrics.jsonl": 0}
        kept["metrics.jsonl"] = max(done - 1, 0) * len(SPLITS)
        for name, lines in kept.items():
            path = out / name
            old = path.read_text().splitlines(True) if done and path.exists() else []
            with atomic_open(path) as f:
                f.write("".join(old[:lines]).encode())
        self._last: list[dict[str, Any]] = []

    def __len__(self) -> int:
        """The units: the yardsticks, then the checkpoints."""
        return 1 + len(self._ckpts)

    def _config(self, windows: int, ckpts: str) -> dict[str, Any]:
        job = None if self.parent is None else self.parent.job
        return {
            "windows": windows,
            "stacks": len(self._targets),
            "steps": self._steps,
            "batch": FIT_BATCH,
            "lr": DECODER_LR,
            "seed": self.conf.seed + DECODER_SEED,
            "pca_seed": self.conf.seed + PCA_SEED,
            "device": self.conf.device,
            "ckpts": [p.name for p in self._ckpts],
            "which": ckpts,
            "parent": None
            if job is None
            else {
                "run_id": job.get("run_id"),
                "group": job["group"],
                "code": job.get("code"),
            },
        }

    # -- the units ----------------------------------------------------------------

    def step(self) -> list[dict[str, Any]]:
        """Do unit ``done``; returns its lines."""
        if self.done == 0:
            lines = self._yardsticks()
            _append(self.out / "baselines.jsonl", lines)
        else:
            path = self._ckpts[self.done - 1]
            assert self.parent is not None
            head = ckpt.load_eval(
                path, self._mods.named(), group=self.parent.job["group"]
            )
            code = _Code("vit", self._mods.encode, self._mods)
            lines = self._evaluate(code, path.stem, t=head["t"], frame=head["frame"])
            pcas = self.baselines()
            for line in lines:
                if line["split"] == "val":
                    line["objective"] = (
                        _completion_objective(line, self._measures.kinds)
                        if self.conf.measure.encoder_objective == "completion"
                        else _objective(line, pcas["val"]["pca"])
                    )
            _append(self.out / "metrics.jsonl", lines)
            self._last = lines
        self.done += 1
        return lines

    def baselines(self) -> dict[str, dict[str, dict[str, Any]]]:
        """The lines of unit 0 by split and code (read from ``baselines.jsonl``)."""
        out: dict[str, dict[str, dict[str, Any]]] = {}
        for raw in (self.out / "baselines.jsonl").read_text().splitlines():
            line = json.loads(raw)
            out.setdefault(line["split"], {})[line["code"]] = line
        return out

    def final(self) -> dict[str, Any]:
        """``result.json``'s ``final``: the last ``vit`` line of each split
        (``decode``; empty without checkpoints) and the yardsticks
        (``baselines``)."""
        last = self._last
        if not last and len(self._ckpts):
            raw = (self.out / "metrics.jsonl").read_text().splitlines()
            last = [json.loads(x) for x in raw[-len(SPLITS) :]]
        return {
            "decode": {line["split"]: line for line in last},
            "baselines": self.baselines(),
        }

    def _yardsticks(self) -> list[dict[str, Any]]:
        conf = self.conf
        v = conf.vision
        assert v is not None
        gen = torch.Generator().manual_seed(conf.seed + PCA_SEED)
        flat = as_float(self._targets).flatten(1)
        d = self._mods.tokens_per_view * v.token_width if self._tokens else v.z_width
        rank = min(d, len(flat) - 1) if self._tokens else d  # module docstring
        mean, comps = pca.fit_pca(flat, rank, gen=gen)
        del flat

        def pca_z(stacks: Tensor) -> Tensor:
            return pca.encode(as_float(stacks[:, -1]).flatten(1), mean, comps)

        lines = self._evaluate(_Code("pca", pca_z, linear=(mean, comps)), "pca")
        if rank < d:
            for line in lines:
                line["pca_rank"] = rank
        init = build_vision(conf, seed=conf.seed)
        lines += self._evaluate(_Code("init", init.encode, init), "init")
        return lines

    # -- one code -----------------------------------------------------------------

    def _encode(self, code: _Code, windows: Tensor) -> tuple[Tensor, Tensor | None]:
        """The code ``(w, K, D)`` of the stacks of the windows ``(w, N + K - 1,
        C, H, W)`` (CPU u8; ``_stacks``, made on the device a batch of windows
        at a time; a ViT's $z$ or its flattened tokens, module docstring) and,
        for a ViT, the tokens of the last time slot ``(w K, gh gw, d)``."""
        v = self.conf.vision
        assert v is not None
        w, k = len(windows), windows.shape[1] - v.frames + 1
        per = max(1, BATCH // k)
        zs, toks = [], []
        with torch.no_grad():
            for i in range(0, w, per):
                x = _stacks(windows[i : i + per].to(self._dev), v.frames).flatten(0, 1)
                if code.mods is None:
                    zs.append(code.z(x))
                    continue
                tokens = code.mods.tokens(x)
                zs.append(tokens.flatten(1) if self._tokens else code.mods.z_of(tokens))
                toks.append(last_slot(code.mods, tokens))
        z = torch.cat(zs).unflatten(0, (w, k))
        return z, torch.cat(toks) if toks else None

    def _fit_error(self, dec: nn.Module, z: Tensor) -> float:
        """``recon_fit``: the decoded ``z`` of the training stacks against
        their last frames, over the mean image's error, a batch at a time."""
        err = base = 0.0
        with torch.no_grad():
            for zc, tc in zip(z.split(BATCH), self._targets.split(BATCH), strict=True):
                t = as_float(tc)
                err += float(((dec(zc).clamp(0, 1) - t) ** 2).sum())
                base += float(((self._mean - t) ** 2).sum())
        return err / base

    def _decoded(self, dec: nn.Module, z: Tensor) -> Tensor:
        with torch.no_grad():
            return torch.cat([dec(c) for c in z.float().split(BATCH)]).clamp(0, 1)

    def _evaluate(
        self, code: _Code, name: str, *, t: float = 0.0, frame: int = 0
    ) -> list[dict[str, Any]]:
        conf, v = self.conf, self.conf.vision
        assert v is not None
        horizons = conf.measure.horizons
        z_train, tok_train = self._encode(code, self._train)
        gen = torch.Generator().manual_seed(conf.seed + DECODER_SEED)
        dec = _Standardized(z_train.flatten(0, 1))
        dec.dec = fit_decoder(
            dec.standardize(z_train.flatten(0, 1)),
            self._targets,
            steps=self._steps,
            lr=DECODER_LR,
            gen=gen,
        )
        mean_img = self._mean
        recon_fit = self._fit_error(dec, z_train.flatten(0, 1))
        tok_dec = None
        if tok_train is not None:
            patches = patches_of(self._targets, v.patch)
            tok_dec = fit_token_decoder(
                tok_train.flatten(0, 1), patches, steps=self._steps, gen=gen
            )
            del patches, tok_train
        calib = self._split_z(code, "calib")
        z0, z1 = z_train[:, :-1].flatten(0, 1), z_train[:, 1:].flatten(0, 1)
        del z_train  # copied into z0, z1
        if z0.is_cuda:
            # The ridge needs a few float64 D x D matrices (2 GiB each at the
            # D = 16384 of 32-dimensional tokens), more than the allocator's
            # cache left free after the PCA and the decoders (7 GiB in use,
            # 18 GiB reserved, measured): return the cache to the device.
            torch.cuda.empty_cache()
        alpha = linear_dyn.select_alpha(z0, z1, calib)
        a, b = linear_dyn.fit(z0, z1, alpha=alpha)
        comp = (
            {}
            if code.mods is None
            else self._measures.completion_rows(code.mods, list(SPLITS))
        )
        lines = []
        for split in SPLITS:
            zs = self._split_z(code, split)
            windows = self._measures.windows[split]  # (n, N + H, C, H, W) u8
            frames = as_float(
                windows[:, v.frames - 1 :].to(self._dev)
            )  # (n, 1 + H, ...)
            f0 = frames[:, 0]
            base = _mse(mean_img.expand_as(f0), f0)
            x_hat = self._decoded(dec, zs[:, 0])
            line: dict[str, Any] = {
                "t": t,
                "frame": frame,
                "split": split,
                "code": code.name,
                "recon": _mse(x_hat, f0) / base,
                "recon_blur": blur_recon(x_hat, f0, mean_img),
                "recon_fit": recon_fit,
            }
            if code.linear is not None:
                lin = pca.decode(zs[:, 0], *code.linear).view_as(f0).clamp(0, 1)
                line["recon_lin"] = _mse(lin, f0) / base
                line["recon_lin_blur"] = blur_recon(lin, f0, mean_img)
            line |= linear_dyn.errors(a, b, zs, horizons, train=z0)
            pred = linear_dyn.rollout(a, b, zs[:, 0], max(horizons)).float()
            dyn_imgs = {}
            for h in horizons:
                g = self._decoded(dec, pred[:, h - 1])
                fh = frames[:, h]
                err = _mse(g, fh)
                line[f"img_dyn_{h}"] = err / _mse(mean_img.expand_as(fh), fh)
                line[f"img_dyn_{h}/copy"] = err / _mse(f0, fh)
                dyn_imgs[h] = g
            restored: list[Tensor] = []
            if code.mods is not None and tok_dec is not None:
                restored = self._restore(code.mods, split, tok_dec, f0, line)
            line |= z_stats(zs[:, 0].float())
            line |= comp.get(split, {})
            line["alpha"] = alpha
            lines.append(line)
            if split == "val":
                self._pictures(name, dec, zs, frames, x_hat, dyn_imgs, restored)
        return lines

    def _split_z(self, code: _Code, split: str) -> Tensor:
        """$z$ ``(n, 1 + H, D)`` of every stack of the windows of ``split``."""
        z, _ = self._encode(code, self._measures.windows[split])
        return z

    def _restore(
        self,
        mods: VisionModules,
        split: str,
        tok_dec: nn.Module,
        f0: Tensor,
        line: dict[str, Any],
    ) -> list[Tensor]:
        """The restoration keys of ``line`` (module docstring) and the
        pictures' rows of val: per view the context and its restoration, then
        the target ceiling."""
        v = self.conf.vision
        assert v is not None
        windows = self._measures.windows[split][:, : v.frames]
        rows: list[Tensor] = []
        mean = self._mean.expand_as(f0)
        for view, m, keep, future in zip(
            self._views,
            self._measures.masks[split],
            self._measures.keeps,
            self._measures.futures,
            strict=True,
        ):
            imgs, pix = [], []
            for i in range(0, len(windows), BATCH):
                x = windows[i : i + BATCH].to(self._dev)
                img, px = restore(
                    mods,
                    x,
                    m[i : i + BATCH].to(self._dev),
                    tok_dec,
                    keep=keep,
                    future=future,
                )
                imgs.append(img.clamp(0, 1))
                pix.append(px)
            img, px = torch.cat(imgs), torch.cat(pix)
            line[f"restore_{view}_mask"] = _masked_mse(img, f0, px) / _masked_mse(
                mean, f0, px
            )
            line[f"restore_{view}_all"] = _mse(img, f0) / _mse(mean, f0)
            rows += [f0[:SHOWN] * (~px[:SHOWN])[:, None].to(f0), img[:SHOWN]]
        target = []
        with torch.no_grad():
            for i in range(0, len(windows), BATCH):
                x = windows[i : i + BATCH].to(self._dev)
                toks = last_slot(mods, mods.tokens(x))
                target.append(
                    picture_of(tok_dec(toks.flatten(0, 1)), len(x), mods.grid)
                )
        ceiling = torch.cat(target).clamp(0, 1)
        line["restore_target_all"] = _mse(ceiling, f0) / _mse(mean, f0)
        return [*rows, ceiling[:SHOWN]]

    def _pictures(
        self,
        name: str,
        dec: nn.Module,
        zs: Tensor,
        frames: Tensor,
        x_hat: Tensor,
        dyn_imgs: Mapping[int, Tensor],
        restored: Sequence[Tensor],
    ) -> None:
        f0 = frames[:, 0]
        _png(self.out / f"{name}.png", torch.stack([f0[:SHOWN], x_hat[:SHOWN]]))
        if restored:
            _png(self.out / f"{name}-restore.png", torch.stack([f0[:SHOWN], *restored]))
        hs = [0, *self.conf.measure.horizons]
        rows = []
        for w in range(min(DYN_SHOWN, len(f0))):
            true = torch.stack([frames[w, h] for h in hs])
            pred = torch.stack([x_hat[w] if h == 0 else dyn_imgs[h][w] for h in hs])
            seen = self._decoded(dec, torch.stack([zs[w, h] for h in hs]))
            rows += [true, pred, seen]
        _png(self.out / f"{name}-dyn.png", torch.stack(rows))


def _objective(line: Mapping[str, Any], pca_line: Mapping[str, Any]) -> float:
    """$J$ of a val line of the ViT against the PCA's (module docstring)."""
    spread = line.get("z_spread")
    if spread is None or not spread >= COLLAPSE:
        return PENALTY
    keys = ("recon", "dyn_1/copy_1")
    got = [line.get(k) for k in keys] + [pca_line.get(k) for k in keys]
    values = [float(x) for x in got if x is not None]
    if len(values) < len(got) or not all(math.isfinite(x) for x in values):
        return PENALTY
    recon, dyn, recon_pca, dyn_pca = values
    lin = pca_line.get("recon_lin")  # the PCA's best decoder: Eckart-Young's
    if lin is not None and math.isfinite(float(lin)):
        recon_pca = min(recon_pca, float(lin))
    if not (recon_pca > 0 and dyn_pca > 0 and recon >= 0 and dyn >= 0):
        return PENALTY
    j = math.log(max(recon / recon_pca, dyn / dyn_pca, FLOOR))
    return min(j, PENALTY)


def _completion_objective(line: Mapping[str, Any], kinds: Sequence[str]) -> float:
    """The ``"completion"`` objective of a val line of the ViT (module
    docstring): the log of the mean ``comp_<kind>`` of ``kinds``."""
    spread = line.get("z_spread")
    if spread is None or not spread >= COLLAPSE:
        return PENALTY
    got = [line.get(f"comp_{k}") for k in kinds]
    values = [float(x) for x in got if x is not None]
    if not values or len(values) < len(got):
        return PENALTY
    if not all(math.isfinite(x) and x >= 0 for x in values):
        return PENALTY
    return min(math.log(max(sum(values) / len(values), FLOOR)), PENALTY)


def _png(path: Path, frames: Tensor) -> None:
    grid = frames.cpu()
    write_png(path, tile(grid, scale=max(1, 96 // grid.shape[-1])))


def _append(path: Path, lines: Sequence[Mapping[str, Any]]) -> None:
    """Append ``lines`` to the JSON lines ``path`` (non-finite values as null)."""

    def clean(x: Any) -> Any:
        return None if isinstance(x, float) and not math.isfinite(x) else x

    with path.open("a") as f:
        f.write(
            "".join(
                json.dumps({k: clean(x) for k, x in ln.items()}) + "\n" for ln in lines
            )
        )
