r"""The evaluation of the joint learning (M16): the dynamics' rollouts in the
latent space of the encoder, and, with a decoder learned afterwards
(``measure.posthoc``), in images.

``Windows`` are fixed windows of frames of the balls: ``observed + horizon``
chunks of ``frames`` (N) frames, each a frame further than the one before, so
a window is ``N - 1 + observed + horizon`` frames of a clip. Chunk $j$ of a
window ends at its frame $N - 1 + j$ and has the representation $z_j$ of the
encoder as it is. The dynamics is shown $z_0 \dots z_{O-1}$ ($O$ =
``observed``, teacher forced from the state 0, as in training) and rolls out
open loop: its $h$-th prediction (``measure.dynamics.rollout``) is of $z_{O +
h - 1}$, the frame $N - 1 + O + h - 1$ of the window.

``latent_errors`` has, for each horizon ``h`` (a mean over the windows and
the dimensions of the squared error):

- ``roll{h}``: the rollout against $z_{O+h-1}$;
- ``copy{h}``: the last observed $z_{O-1}$ again;
- ``mean{h}``: the mean $z$ of the windows (over the windows and all their
  $O + H$ chunks), a constant that has seen the targets, so a hard bar;
- ``roll{h}/copy{h}``, ``roll{h}/mean{h}``; ``nonfinite``: the windows with a
  prediction that is not finite.

A chunk shares $N - 1$ frames with its neighbour, so with $N > 1$ the copy of
$z$ is very strong: the judgement that counts is in images (``image_rows``):

- ``img_roll{h}``: the mean squared error per pixel of the predicted $z_{O+h-1}$
  decoded, against the true frame of the chunk;
- ``pix_copy{h}``: that of the last observed frame again;
- ``pix_mean{h}``: that of the mean of the true frames at ``h`` over the
  windows (a constant that has seen the targets);
- ``img_recon``: that of the decoded observed $z_{O-1}$ against its frame (what
  the decoder alone loses).

``image_rows`` has those keys under ``prefix`` ``"img_"`` (the decoder learned
afterwards); the decoder of the run's own reconstruction term is scored under
``"rec_"`` (``rec_roll{h}``, ``rec_recon``), so the two never share a key.

The dream (``dream``, a required evaluation of every joint run since
2026-11-11) runs the dynamics open loop ``DREAM`` frames (5 s) on windows of
that horizon (``ball_windows(..., horizon=DREAM)``), shown either the
``observed`` chunks as above (``dream``) or only the last of them, from the
state 0 (``dream1``), both predicting the same frames. In images, for $h$ in
``DREAM_HORIZONS``:

- ``dream_img{h}``, ``dream1_img{h}``: as ``img_roll{h}``;
- ``dream_copy{h}``, ``dream_mean{h}``: as ``pix_copy{h}`` and ``pix_mean{h}``;
- ``dream_lead``, ``dream1_lead``: the frames ahead before the decoded
  prediction is first no better than the mean image (``DREAM`` if never);
- ``dream_nonfinite``, ``dream1_nonfinite``: as ``nonfinite``.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from exp.video.data.clips import make_clips
from exp.video.data.physics import render
from exp.video.dyn.modules import DynamicsModules, build_dynamics
from exp.video.jepa.vit import VisionModules
from exp.video.measure.dynamics import ratio, rollout
from exp.video.schema.run import RunConf

__all__ = [
    "BATCH",
    "DREAM",
    "DREAM_EVERY",
    "DREAM_HORIZONS",
    "DREAM_SHOWN",
    "DYNAMICS_SEED",
    "SHOWN",
    "Windows",
    "ball_windows",
    "decode",
    "dream",
    "image_rows",
    "joint_dynamics",
    "latent_errors",
]

BATCH = 256  # chunks encoded or decoded at once
SHOWN = 4  # windows in a picture
DYNAMICS_SEED = 1  # added to the run's seed: the vision has the seed itself
DREAM = 150  # frames a dream runs open loop (5 s at 30 fps)
DREAM_HORIZONS = (1, 5, 10, 20, 30, 60, 150)  # the dream's keys (module docstring)
DREAM_EVERY = 10  # frames between two columns of a dream's picture
DREAM_SHOWN = 3  # windows in a dream's picture


def joint_dynamics(conf: RunConf) -> DynamicsModules:
    """The dynamics of the joint run ``conf`` at its initial weights: the
    modules of ``build_dynamics`` on $z$ of the vision (``vision.dim``) with
    the seed ``conf.seed + DYNAMICS_SEED``, so that the encoder's initial
    weights are those of the vision program of the same seed.

    Raises:
        ValueError: ``conf`` has no dynamics or no vision.
    """
    if conf.dynamics is None or conf.vision is None:
        raise ValueError(f"the run {conf.name!r} needs dynamics and vision")
    d = conf.vision.dim
    return build_dynamics(
        conf.dynamics, d, d, seed=conf.seed + DYNAMICS_SEED, device=conf.device
    )


@dataclass(frozen=True)
class Windows:
    """The frames ``images`` ``(n, N - 1 + observed + horizon, C, H, W)`` of
    ``n`` windows (module docstring)."""

    images: Tensor
    frames: int
    observed: int
    horizon: int

    @property
    def chunks(self) -> int:
        """The chunks of a window, ``observed + horizon``."""
        return self.observed + self.horizon

    def frame(self, h: int) -> Tensor:
        """The true frame ``(n, C, H, W)`` of ``z_{O+h-1}`` (``h`` = 0: the last
        observed chunk's)."""
        return self.images[:, self.frames - 1 + self.observed + h - 1]

    @torch.no_grad()
    def z(self, mods: VisionModules, chunks: int | None = None) -> Tensor:
        """$z$ ``(n, c, D)`` of the first ``c = chunks`` (all, ``observed +
        horizon``, if ``None``) chunks of every window at the weights of
        ``mods`` as they are."""
        n = len(self.images)
        c = self.chunks if chunks is None else chunks
        flat = torch.arange(n * c, device=self.images.device)
        window, j = flat // c, flat % c
        span = torch.arange(self.frames, device=self.images.device)
        out = []
        for i in range(0, len(flat), BATCH):
            w, k = window[i : i + BATCH], j[i : i + BATCH]
            out.append(mods.encode(self.images[w[:, None], k[:, None] + span]))
        return torch.cat(out).view(n, c, -1)


def ball_windows(
    conf: RunConf, split: str, *, stride: int, horizon: int | None = None
) -> Windows:
    """The windows of ``split`` of the clips of balls of ``conf``
    (``measure.clips`` clips of ``measure.windows`` windows ``stride`` frames
    apart, as ``programs.vision.Measures`` has its chunks), rendered, on the
    device of the run, of ``horizon`` (``max(measure.horizons)`` if ``None``)
    frames past the observed.

    Raises:
        ValueError: ``conf`` has no vision or no balls.
    """
    balls, v, m = conf.data.balls, conf.vision, conf.measure
    if balls is None or v is None:
        raise ValueError(f"the run {conf.name!r} needs balls and a vision")
    n = v.frames
    horizon = max(m.horizons) if horizon is None else horizon
    clips = make_clips(
        balls,
        split=split,
        clips=m.clips,
        length=n - 1 + m.observed + horizon + stride * (m.windows - 1),
        observed=n - 1 + m.observed,
        horizon=horizon,
        stride=stride,
        frame=conf.time.frame,
    )
    span = n - 1 + m.observed + horizon
    idx = clips.starts[:, 1:] + torch.arange(span)  # (windows, span)
    states = clips.states[clips.starts[:, :1], idx]  # (windows, span, 4K)
    images = render(states.flatten(0, 1), balls.size, balls.radii).float()
    images = images.unflatten(0, (len(idx), span)).to(conf.device)
    return Windows(images, n, m.observed, horizon)


def _sq(pred: Tensor, target: Tensor) -> Tensor:
    return ((pred - target) ** 2).flatten(1).mean(-1)  # (n,)


@torch.no_grad()
def latent_errors(
    dyn: DynamicsModules,
    z: Tensor,
    *,
    observed: int,
    horizons: Sequence[int],
    residual: bool,
) -> tuple[dict[str, float], Tensor]:
    """The keys of the module docstring for the windows' $z$ ``(n, O + H, D)``
    and the rollout of ``dyn``; also returns the rollout ``(n, H, D)``."""
    horizon = max(horizons)
    pred = rollout(dyn, z, residual=residual, observed=observed, horizon=horizon)
    target = z[:, observed : observed + horizon]
    mean = z.mean((0, 1))
    out: dict[str, float] = {}
    for h in horizons:
        t = target[:, h - 1]
        roll = float(_sq(pred[:, h - 1], t).double().mean())
        copy = float(_sq(z[:, observed - 1], t).double().mean())
        avg = float(_sq(mean.expand_as(t), t).double().mean())
        out[f"roll{h}"], out[f"copy{h}"], out[f"mean{h}"] = roll, copy, avg
        out[f"roll{h}/copy{h}"] = ratio(roll, copy)
        out[f"roll{h}/mean{h}"] = ratio(roll, avg)
    out["nonfinite"] = int((~torch.isfinite(pred).flatten(1).all(1)).sum())
    return out, pred


@torch.no_grad()
def decode(dec: nn.Module, z: Tensor) -> Tensor:
    return torch.cat([dec(x) for x in z.split(BATCH)])


@torch.no_grad()
def image_rows(
    dec: nn.Module,
    w: Windows,
    z: Tensor,
    pred: Tensor,
    horizons: Sequence[int],
    *,
    prefix: str = "img_",
) -> tuple[dict[str, float], Tensor]:
    """The image keys of the module docstring (``{prefix}roll{h}`` and
    ``{prefix}recon``; ``pix_*`` have no prefix) of the windows ``w``, from the
    decoder ``dec`` (on the device of ``z``), their $z$ ``(n, O + H, D)`` and
    the rollout ``pred`` ``(n, H, D)``; also a picture's frames ``(2 SHOWN,
    1 + len(horizons), C, H, W)``: for each of the first windows a row of the
    true frames (the last observed one, then those of the horizons) and a row
    of the decoded ones (``z`` of the last observed, then the rollout's).
    """
    steps = [0, *horizons]
    decoded = {
        h: decode(dec, z[:, w.observed - 1] if h == 0 else pred[:, h - 1])
        for h in steps
    }
    out: dict[str, float] = {}
    last = w.frame(0)
    out[f"{prefix}recon"] = float(((decoded[0] - last) ** 2).double().mean())
    for h in horizons:
        truth = w.frame(h)
        out[f"{prefix}roll{h}"] = float(((decoded[h] - truth) ** 2).double().mean())
        out[f"pix_copy{h}"] = float(((last - truth) ** 2).double().mean())
        mean = truth.mean(0, keepdim=True)
        out[f"pix_mean{h}"] = float(((mean - truth) ** 2).double().mean())
    k = min(SHOWN, len(z))
    rows = []
    for i in range(k):
        rows.append(torch.stack([w.frame(h)[i] for h in steps]))
        rows.append(torch.stack([decoded[h][i] for h in steps]))
    return out, torch.stack(rows)


def _mse(a: Tensor, b: Tensor) -> float:
    return float(((a - b) ** 2).double().mean())


@torch.no_grad()
def dream(
    dyn: DynamicsModules,
    dec: nn.Module,
    w: Windows,
    z: Tensor,
    *,
    residual: bool,
) -> tuple[dict[str, float], dict[str, list[float]], Tensor]:
    """The dream (module docstring) of ``dyn`` on the windows ``w`` (of
    horizon ``DREAM`` for the evaluation) from $z$ ``(n, observed, D)`` of
    their observed chunks, decoded by ``dec``.

    Returns:
        The keys of the module docstring; the curves of the mean squared
        error per pixel at every frame ahead 1 .. H (``"dream"``,
        ``"dream1"``, ``"copy"``, ``"mean"``, each ``H`` values); and a
        picture ``(3 k, 1 + H // DREAM_EVERY, C, H, W)`` of the first ``k =
        DREAM_SHOWN`` windows: for each, a row of the true frames, one of
        ``dream`` and one of ``dream1``, at the last observed frame and then
        every ``DREAM_EVERY`` frames (the first column of the dreams is the
        decoded $z$ of the last observed chunk).
    """
    o, horizon = w.observed, w.horizon
    preds = {
        "dream": rollout(dyn, z, residual=residual, observed=o, horizon=horizon),
        "dream1": rollout(
            dyn, z[:, o - 1 :], residual=residual, observed=1, horizon=horizon
        ),
    }
    k = min(DREAM_SHOWN, len(z))
    last = w.frame(0)
    recon = decode(dec, z[:k, o - 1])
    columns: dict[str, list[Tensor]] = {
        "truth": [last[:k]],
        **{key: [recon] for key in preds},
    }
    curves: dict[str, list[float]] = {key: [] for key in (*preds, "copy", "mean")}
    for h in range(1, horizon + 1):
        truth = w.frame(h)
        curves["copy"].append(_mse(last, truth))
        curves["mean"].append(_mse(truth.mean(0, keepdim=True), truth))
        shown = h % DREAM_EVERY == 0
        if shown:
            columns["truth"].append(truth[:k])
        for key, pred in preds.items():
            x = decode(dec, pred[:, h - 1])
            curves[key].append(_mse(x, truth))
            if shown:
                columns[key].append(x[:k])
    out: dict[str, float] = {}
    for h in (h for h in DREAM_HORIZONS if h <= horizon):
        for key in preds:
            out[f"{key}_img{h}"] = curves[key][h - 1]
        out[f"dream_copy{h}"] = curves["copy"][h - 1]
        out[f"dream_mean{h}"] = curves["mean"][h - 1]
    for key, pred in preds.items():
        below = [a < b for a, b in zip(curves[key], curves["mean"], strict=True)]
        out[f"{key}_lead"] = below.index(False) if False in below else horizon
        out[f"{key}_nonfinite"] = int((~torch.isfinite(pred).flatten(1).all(1)).sum())
    rows = {key: torch.stack(c, 1) for key, c in columns.items()}  # (k, cols, ...)
    picture = torch.stack([rows[key] for key in ("truth", *preds)], 1)
    return out, curves, picture.flatten(0, 1)
