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
    "DYNAMICS_SEED",
    "SHOWN",
    "Windows",
    "ball_windows",
    "decode",
    "image_rows",
    "joint_dynamics",
    "latent_errors",
    "latent_row",
]

BATCH = 256  # chunks encoded or decoded at once
SHOWN = 4  # windows in a picture
DYNAMICS_SEED = 1  # added to the run's seed: the vision has the seed itself


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
    def z(self, mods: VisionModules) -> Tensor:
        """$z$ ``(n, observed + horizon, D)`` of every chunk of every window at
        the weights of ``mods`` as they are."""
        n = len(self.images)
        c = self.chunks
        flat = torch.arange(n * c, device=self.images.device)
        window, j = flat // c, flat % c
        span = torch.arange(self.frames, device=self.images.device)
        out = []
        for i in range(0, len(flat), BATCH):
            w, k = window[i : i + BATCH], j[i : i + BATCH]
            out.append(mods.encode(self.images[w[:, None], k[:, None] + span]))
        return torch.cat(out).view(n, c, -1)


def ball_windows(conf: RunConf, split: str, *, stride: int) -> Windows:
    """The windows of ``split`` of the clips of balls of ``conf``
    (``measure.clips`` clips of ``measure.windows`` windows ``stride`` frames
    apart, as ``programs.vision.Measures`` has its chunks), rendered, on the
    device of the run.

    Raises:
        ValueError: ``conf`` has no vision or no balls.
    """
    balls, v, m = conf.data.balls, conf.vision, conf.measure
    if balls is None or v is None:
        raise ValueError(f"the run {conf.name!r} needs balls and a vision")
    n, horizon = v.frames, max(m.horizons)
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
def latent_row(
    mods: VisionModules,
    dyn: DynamicsModules,
    w: Windows,
    horizons: Sequence[int],
    *,
    residual: bool,
) -> dict[str, float]:
    """``latent_errors`` of the windows ``w`` at the weights of ``mods`` and
    ``dyn`` as they are."""
    row, _ = latent_errors(
        dyn, w.z(mods), observed=w.observed, horizons=horizons, residual=residual
    )
    return row


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
) -> tuple[dict[str, float], Tensor]:
    """The image keys of the module docstring of the windows ``w``, from the
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
    out["img_recon"] = float(((decoded[0] - last) ** 2).double().mean())
    for h in horizons:
        truth = w.frame(h)
        out[f"img_roll{h}"] = float(((decoded[h] - truth) ** 2).double().mean())
        out[f"pix_copy{h}"] = float(((last - truth) ** 2).double().mean())
        mean = truth.mean(0, keepdim=True)
        out[f"pix_mean{h}"] = float(((mean - truth) ** 2).double().mean())
    k = min(SHOWN, len(z))
    rows = []
    for i in range(k):
        rows.append(torch.stack([w.frame(h)[i] for h in steps]))
        rows.append(torch.stack([decoded[h][i] for h in steps]))
    return out, torch.stack(rows)
