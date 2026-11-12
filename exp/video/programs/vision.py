r"""The program of M15: the vision alone (``exp.video.jepa``), for the tasks

- ``"online"``: the vision (PC-ALM or BP, ``conf.learner``) learns from ``B``
  streams of frames (``BallFrames`` of the balls, or ``TourFrames`` of Walking
  Tours) frame by frame; every ``measure.every`` seconds it is evaluated on
  fixed chunks (``Measures``);
- ``"bench"``: the same without the evaluations, to time it;
- ``"posthoc"``: the decoder learned afterwards of each evaluation checkpoint
  of a finished run (``exp.video.programs.posthoc``);
- ``"baselines"`` (the packed layout): the yardsticks of the posthoc
  evaluation alone, of the job's config (``measure.encoder_eval``).

A frame: the next frame of each stream written as a ``pc.msg.Sample`` (``x``
the images, ``reset`` where a stream starts anew), and the terms of the JEPA
loss the head sends first for it read back (``Host.show``; they are of the
weights before anything learned from the frame), added to the
sums of ``stream.jsonl`` (``loss``, ``pred``, ``sigreg``, and ``recon`` with
``vision.recon`` > 0: the means over the frames that have a full chunk). Frame 0 is not counted:
the head of PC-ALM first sees the zeros sent before it, at the same time 0.

The evaluation reads the live modules (no copy) between the frames. Each split
(``"calib"``, ``"val"``, ``"test"``) has ``measure.clips`` x
``measure.windows`` chunks of $N$ frames and fixed context masks: for the
balls, the chunks end every ``STRIDE`` frames of clips of new balls drawn from
the split's seed (``data.clips``); for Walking Tours, chunks of the held-out
video (``walking_tours.eval_clips``) in time order, the first third calib, the
second val, the last test. A row of val or test has the JEPA loss of the
chunks (``loss``, ``pred``, ``sigreg``, and ``recon`` with ``vision.recon`` > 0:
``loss.jepa_loss`` of batches of
``data.streams`` chunks, as in a frame of the stream, along fixed directions,
averaged over the batches, so that they compare with ``stream.jsonl``), the
effective rank of $z$ (``rank``), the least standard deviation of a dimension
of $z$ (``z_std_min``), the mean standard deviation of a dimension over the
root mean square of $z$ (``z_spread``: 0 on a collapse, however large the
common part of $z$), the $R^2$ of the probe from $z$ to the grey 8 x 8
thumbnail of the last frame (``thumb``, ``measure.vision.thumbnail_probe``)
and, for the balls, of the probe from $z$ to each ball's position
(``probe_x{k}``, ``probe_y{k}``, their least ``probe_pos_min``) and, with
$N > 1$, velocity (``probe_vx{k}``, ``probe_vy{k}``, ``probe_vel_min``)
(``measure.vision.ball_probe``). The probes fit on the chunks of calib (their
ridge chosen by cross-validation within calib) and score on the split's.
``baselines.json`` has the same of the vision before it learns (``init_*``)
and the ball probes from the pixels of the chunks (``pixels_*``), each frame
shrunk to ``THUMB`` x ``THUMB`` by the mean of its blocks: the shrunk frames
are the higher bar (on the val of ``balls_n1`` ``probe_pos_min`` 0.088 against
0.029 of the full frames, which have more pixels than calib has chunks).

The objective (val rows): $1 - $ ``probe_pos_min`` for the balls, $1 -$
``thumb`` for Walking Tours, at most ``PENALTY``; ``PENALTY`` when a value is
not finite or $z$ collapsed (``z_spread`` below ``COLLAPSE``, where a probe of
the standardized $z$ would read the rounding).

The packed layout (M17, ``vision.layout="packed"``) streams stacks of
``vision.frames`` frames ``vision.stride`` video frames apart
(``StackedTourFrames``, ``data.tours.advance`` video frames per frame of the
run) and is evaluated by ``PackedMeasures``: the windows of
``walking_tours.eval_windows`` (``measure.steps`` stacks after the first, for
the linear dynamics of ``task=posthoc``), the first third calib, the second
val, the last test, with fixed context masks per split. A row has ``loss``,
``pred``, ``sigreg`` (``loss.jepa_loss_packed`` of the first stack of each
window), ``rank``, ``z_std_min``, ``z_spread``, and the spread of the
target tokens within the stacks against that between them (the collapse of
the tokens within a stack, which ``z_spread`` does not see; ``token_spread``):
``tok_within`` (the mean squared deviation of a token from its stack's mean
token, per dimension), ``tok_between`` (the variance of the stacks' mean
tokens over the split, per dimension) and, with more than one time slot,
``tok_temporal`` (the same deviation from the mean over the slots of the
token's spatial position); no probe and no
``objective`` (the decoders and the dynamics are learned afterwards, and
``task=posthoc`` writes the objective). It also has the completion errors of
the split (``measure.completion``, ``PackedMeasures.completion``):
``comp_<kind>``, ``comp_<kind>_vis``, ``comp_<kind>_mean`` per mask kind,
and the same normalized within the stacks ``ncomp_*``
(``short``, ``long``, ``future``); the row of val also those of calib
(``calib_comp_*``) and of the held-in stacks of the training videos
(``fit_comp_*``), to see the gap.

The prepared Walking Tours are read from ``$MUJOU_EXP_DATA/<data_version>``
(``DATA_ENV``; by default ``.fleet/data`` of this checkout).
"""

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujou.pc as pc
import mujou.testing
import torch
from mujou.testing import ManualRun
from torch import Tensor, nn

from exp.lib.host import Host
from exp.lib.metrics import effective_rank
from exp.lib.png import tile, write_png
from exp.lib.rundir import EnvError, write_json
from exp.video.data.clips import SPLIT_SEEDS, make_clips
from exp.video.data.frames import (
    BallFrames,
    FrameStreams,
    StackedTourFrames,
    StackStreams,
    TourFrames,
)
from exp.video.data.physics import render
from exp.video.data.walking_tours import eval_clips, eval_windows
from exp.video.jepa.loss import Reconstruction, jepa_loss, jepa_loss_packed
from exp.video.jepa.masks import (
    block_masks,
    expected_keep,
    make_views,
    multiblock_masks,
    pack_views,
)
from exp.video.jepa.nets import build_vision_net, vision_modules
from exp.video.jepa.nodes import draw_directions, draw_packed_directions
from exp.video.jepa.vit import VisionModules, build_vision
from exp.video.loop import Built, Schedule, every
from exp.video.measure.completion import (
    FUTURE,
    completion,
    future_slots,
    future_tokens,
    mean_token,
    tube_tokens,
)
from exp.video.measure.objective import PENALTY
from exp.video.measure.vision import THUMB, ball_probe, thumbnail_probe
from exp.video.schema.data import ToursConf
from exp.video.schema.run import RunConf
from exp.video.schema.vision import VisionConf

__all__ = [
    "COLLAPSE",
    "DATA_ENV",
    "PROGRAM",
    "STRIDE",
    "Measures",
    "PackedMeasures",
    "VisionProgram",
    "rebuild",
    "spread_stats",
    "token_spread",
    "tours_root",
    "vision_objective",
    "z_stats",
]

DATA_ENV = "MUJOU_EXP_DATA"
STRIDE = 30  # frames between the ends of two chunks of a clip of balls
EVAL_BATCH = 256  # chunks encoded at once for the probes' calibration
COLLAPSE = 1e-3  # the least z_spread of a representation that did not collapse
_SPLITS = ("val", "test")
_TOURS_SEED = 4242  # the chunks of the held-out video
HELD_IN_SEED = 2718  # the context masks of the held-in stacks (packed)
_REPO = Path(__file__).resolve().parents[3]


def tours_root(t: ToursConf) -> Path:
    """The local copy of the prepared data ``t.data_version``:
    ``$MUJOU_EXP_DATA/<data_version>``, or ``.fleet/data/<data_version>`` of
    this checkout.

    Raises:
        EnvError: It has no ``index.json`` (the data is not there).
    """
    base = Path(os.environ.get(DATA_ENV, _REPO / ".fleet" / "data"))
    root = base / t.data_version
    if not (root / "index.json").is_file():
        raise EnvError(
            f"no prepared Walking Tours at {root}: copy the data {t.data_version} "
            f"there or set {DATA_ENV} to its parent"
        )
    return root


def vision_objective(row: Mapping[str, Any]) -> float | None:
    """The objective of a val row (module docstring); None for a row of the
    packed layout (no probe: ``task=posthoc`` has the objective)."""
    if "probe_pos_min" not in row and "thumb" not in row:
        return None
    spread = row.get("z_spread")
    if spread is None or not spread >= COLLAPSE:
        return PENALTY
    score = row["probe_pos_min"] if "probe_pos_min" in row else row["thumb"]
    if score is None or not math.isfinite(1 - score):
        return PENALTY
    return min(1 - score, PENALTY)


@dataclass(frozen=True)
class _Chunks:
    """The fixed chunks of a split: ``chunks`` ``(n, N, C, H, W)``, the
    context masks ``(n, V - 1, gh, gw)``, the directions of SIGReg ``(M, D)``
    and, for the balls, the states at each chunk's last frame ``(n, 4K)``."""

    chunks: Tensor
    masks: Tensor
    directions: Tensor
    states: Tensor | None


def _ball_chunks(conf: RunConf, split: str) -> tuple[Tensor, Tensor]:
    """The chunks of the balls of ``split`` and their last states."""
    balls, v, m = conf.data.balls, conf.vision, conf.measure
    assert balls is not None and v is not None
    n = v.frames
    clips = make_clips(
        balls,
        split=split,
        clips=m.clips,
        length=n + STRIDE * (m.windows - 1),
        observed=n,
        horizon=0,
        stride=STRIDE,
        frame=conf.time.frame,
    )
    idx = clips.starts[:, 1:] + torch.arange(n)  # (windows, N)
    states = clips.states[clips.starts[:, :1], idx]  # (windows, N, 4K)
    images = render(states.flatten(0, 1), balls.size, balls.radii).float()
    chunks = images.unflatten(0, (len(idx), n))
    return chunks, states[:, -1].float()


class Measures:
    """The fixed chunks of each split (module docstring) and the evaluation
    of a vision on them.

    Args:
        conf: The run; its vision, data, ``measure`` and device.
    """

    def __init__(self, conf: RunConf) -> None:
        v = conf.vision
        if v is None:
            raise ValueError(f"the run {conf.name!r} has no vision")
        self.v: VisionConf = v
        self.balls = conf.data.balls
        self._batch = conf.data.streams  # B: the chunks of a frame's loss
        dev = torch.device(conf.device)
        if self.balls is not None:
            data: dict[str, tuple[Tensor, Tensor | None]] = {
                s: _ball_chunks(conf, s) for s in ("calib", *_SPLITS)
            }
        else:
            t = conf.data.tours
            assert t is not None
            n = conf.measure.clips * conf.measure.windows
            u8 = eval_clips(
                tours_root(t),
                held_out=t.held_out,
                clips=3 * n,
                length=v.frames,
                seed=_TOURS_SEED,
            )
            if u8.shape[-1] != t.size:
                raise ValueError(f"the data is of {u8.shape[-1]} pixels, not {t.size}")
            frames = u8.float() / 255  # in time order: calib, val, test
            data = {
                s: (frames[i * n : (i + 1) * n], None)
                for i, s in enumerate(("calib", *_SPLITS))
            }
        self.sets: dict[str, _Chunks] = {}
        grid_side = data["val"][0].shape[-1] // v.patch
        for split, (chunks, states) in data.items():
            g = torch.Generator().manual_seed(SPLIT_SEEDS[split])
            masks = block_masks(
                len(chunks) * (v.views - 1),
                (grid_side, grid_side),
                ratio=v.mask_ratio,
                blocks=v.mask_blocks,
                generator=g,
            ).unflatten(0, (len(chunks), v.views - 1))
            directions = draw_directions(v.sigreg_directions, v.dim, g)
            self.sets[split] = _Chunks(
                chunks.to(dev),
                masks.to(dev),
                directions.to(dev),
                None if states is None else states.to(dev),
            )
        self._calib: tuple[int, Tensor] | None = None  # (version, z of calib)

    # -- the evaluation ---------------------------------------------------------

    @torch.no_grad()
    def evaluate(
        self, mods: VisionModules, split: str, *, version: int
    ) -> dict[str, float]:
        """The row of ``split`` (module docstring) of ``mods`` as they are;
        ``version`` names the weights (the frame), so that the calibration's
        $z$ is encoded once for both splits."""
        s, v, enc = self.sets[split], self.v, mods.encoder
        terms, zs = [], []
        dec = mods.decoder
        every = torch.ones(self._batch, dtype=torch.bool, device=s.chunks.device)
        for i in range(0, len(s.chunks), self._batch):
            chunks = s.chunks[i : i + self._batch]
            b = len(chunks)
            h, y = make_views(chunks, s.masks[i : i + b], mods.patch)
            recon = None
            if dec is not None:
                recon = Reconstruction(
                    v.recon, dec, v.pixel_mean, v.pixel_std, chunks[:, -1]
                )
            t = jepa_loss(
                enc(h).unflatten(0, (b, v.views)),
                y.unflatten(0, (b, v.views))[:, 1:],
                every[:b],
                s.directions,
                mods.predictor,
                extra=enc.extra,
                z_pool=enc.z_pool,
                lam=v.sigreg,
                sg=False,
                sigreg_on=v.sigreg_on,
                recon=recon,
            )
            terms.append(torch.stack([t.loss, t.pred, t.sigreg, t.recon]))
            zs.append(t.z)
        loss, pred, sig, rec = torch.stack(terms).mean(0).tolist()
        z = torch.cat(zs)
        row = {
            "loss": loss,
            "pred": pred,
            "sigreg": sig,
            **({"recon": rec} if dec is not None else {}),
            **z_stats(z),
        }
        return row | self._probes(self._calib_z(mods, version), z, s)

    def _calib_z(self, mods: VisionModules, version: int) -> Tensor:
        if self._calib is None or self._calib[0] != version:
            chunks = self.sets["calib"].chunks
            z = torch.cat([mods.encode(c) for c in chunks.split(EVAL_BATCH)])
            self._calib = (version, z)
        return self._calib[1]

    def _probes(self, fit: Tensor, x: Tensor, s: _Chunks) -> dict[str, float]:
        """``thumb`` and the ball probes from the features ``x`` ``(n, F)`` of
        the chunks of ``s``, fit on the features ``fit`` ``(n', F)`` of the
        calibration's chunks."""
        calib = self.sets["calib"]
        both = torch.cat([fit, x])
        half = torch.arange(len(both), device=both.device) < len(fit)
        last = torch.cat([calib.chunks[:, -1], s.chunks[:, -1]])  # (n' + n, C, H, W)
        out = {"thumb": thumbnail_probe(both, last, half)}
        return out | self._ball_probe(fit, x, s)

    def _ball_probe(self, fit: Tensor, x: Tensor, s: _Chunks) -> dict[str, float]:
        if self.balls is None:
            return {}
        assert s.states is not None and self.sets["calib"].states is not None
        both = torch.cat([fit, x])
        half = torch.arange(len(both), device=both.device) < len(fit)
        states = torch.cat([self.sets["calib"].states, s.states])
        k = len(self.balls.radii)
        return ball_probe(both, states, k, half, velocity=self.v.frames > 1)

    def baselines(self, mods: VisionModules) -> dict[str, dict[str, float]]:
        """``init_*``: the rows of ``mods`` as they are (before learning);
        ``pixels_*``: the ball probes from the shrunk pixels (the balls only;
        module docstring)."""
        out: dict[str, dict[str, float]] = {}
        fit = _shrunk(self.sets["calib"].chunks)
        for split in _SPLITS:
            row = {
                f"init_{k}": x
                for k, x in self.evaluate(mods, split, version=-1).items()
            }
            s = self.sets[split]
            pixels = self._ball_probe(fit, _shrunk(s.chunks), s)
            row |= {f"pixels_{k}": x for k, x in pixels.items()}
            out[split] = row
        self._calib = None
        return out

    def visualize(self, out: Path, patch: int) -> None:
        """``views.png``: 4 chunks of val, a row each: the $N$ frames of the
        target view, then the last frame of each context view (masked 0)."""
        s, v = self.sets["val"], self.v
        chunks, masks = s.chunks[:4], s.masks[:4]
        h, _ = make_views(chunks, masks, patch)
        h = h.unflatten(0, (len(chunks), v.views))[:, :, :, :-1]  # no mask channel
        grid = torch.cat([h[:, 0], h[:, 1:, -1]], 1).cpu()  # (4, N + V - 1, C, H, W)
        out.mkdir(parents=True, exist_ok=True)
        write_png(out / "views.png", tile(grid, scale=3))


def token_spread(target: Tensor, slots: int) -> tuple[Tensor, Tensor, Tensor]:
    """The spread of the target tokens ``(n, T, D)`` of ``slots`` time slots
    (in the id order, slot first) within each stack: the mean over the tokens
    and the dimensions of the squared deviation from the stack's mean token
    ``(n,)``, the same from the mean over the slots of the token's spatial
    position ``(n,)``, and the stack's mean token ``(n, D)``."""
    n, t, d = target.shape
    mean = target.mean(1, keepdim=True)
    within = ((target - mean) ** 2).mean((1, 2))
    tv = target.view(n, slots, t // slots, d)
    temporal = ((tv - tv.mean(1, keepdim=True)) ** 2).mean((1, 2, 3))
    return within, temporal, mean[:, 0]


def spread_stats(
    within: Tensor, temporal: Tensor, means: Tensor, slots: int
) -> dict[str, float]:
    """``tok_within``, ``tok_between`` and (more than one slot)
    ``tok_temporal`` of the ``token_spread`` of all the stacks of a split
    (module docstring)."""
    out = {
        "tok_within": float(within.mean()),
        "tok_between": float(means.var(0, unbiased=False).mean()),
    }
    if slots > 1:
        out["tok_temporal"] = float(temporal.mean())
    return out


def z_stats(z: Tensor) -> dict[str, float]:
    """``rank``, ``z_std_min`` and ``z_spread`` of $z$ ``(n, D)`` (module
    docstring)."""
    rms = float(z.pow(2).mean().sqrt())
    return {
        "rank": effective_rank(z),
        "z_std_min": float(z.std(0).min()),
        "z_spread": float(z.std(0).mean()) / rms if rms > 0 else 0.0,
    }


class PackedMeasures:
    """The evaluation of the packed layout (module docstring): the windows of
    each split ``windows`` (u8 ``(n, N + steps, C, H, W)``, on the CPU), the
    masks of each context view ``masks`` (``(n, gh, gw)`` each) and the
    directions of SIGReg of each split; and, for the completion errors, the
    held-in stacks ``held_in`` (u8 ``(n', N, C, H, W)``, ``n`` windows of
    ``encoder_eval.stream_windows`` but those across two videos) with their
    masks ``held_in_masks`` (drawn from ``HELD_IN_SEED``), and the mask kinds
    ``kinds`` (``views``, the names of the context views, then ``future``
    when a stack has more than one time slot).

    Args:
        conf: The run; its vision, Walking Tours, ``measure`` and device.

    Raises:
        ValueError: The run has no vision, or does not stream Walking Tours.
    """

    def __init__(self, conf: RunConf) -> None:
        v, t = conf.vision, conf.data.tours
        if v is None or t is None:
            raise ValueError(f"the run {conf.name!r} needs a vision on Walking Tours")
        self.v: VisionConf = v
        self._batch = conf.data.streams
        self._dev = torch.device(conf.device)
        n = conf.measure.clips * conf.measure.windows
        u8 = eval_windows(
            tours_root(t),
            held_out=t.held_out,
            clips=3 * n,
            frames=v.frames,
            stride=v.stride,
            steps=conf.measure.steps,
            seed=_TOURS_SEED,
        )
        if u8.shape[-1] != t.size:
            raise ValueError(f"the data is of {u8.shape[-1]} pixels, not {t.size}")
        self.windows = {
            s: u8[i * n : (i + 1) * n] for i, s in enumerate(("calib", *_SPLITS))
        }
        side = t.size // v.patch
        self.grid = (side, side)
        self.keeps = [expected_keep(self.grid, m) for m in v.context_masks]
        self.masks: dict[str, list[Tensor]] = {}
        self._directions: dict[str, tuple[Tensor, Tensor | None]] = {}
        two = v.sigreg_on in ("tokens", "both") and v.projected
        for split in self.windows:
            g = torch.Generator().manual_seed(SPLIT_SEEDS[split])
            self.masks[split] = [
                multiblock_masks(n, self.grid, m, keep=k, generator=g)
                for m, k in zip(v.context_masks, self.keeps, strict=True)
            ]
            dz, dt = draw_packed_directions(
                v.sigreg_directions, v.z_width, v.dim, tokens=two, generator=g
            )
            self._directions[split] = (
                dz.to(self._dev),
                None if dt is None else dt.to(self._dev),
            )
        c = len(v.context_masks)
        self.views = ["short", "long"] if c == 2 else [f"c{i}" for i in range(c)]
        self._slots = v.frames // v.tubelet
        self._future = future_slots(self._slots)
        self.kinds = [*self.views, *((FUTURE,) if self._future else ())]
        # imported here: it imports this module
        from exp.video.measure.encoder_eval import stream_windows

        held = stream_windows(conf, tours_root(t), windows=n, steps=0)
        self.held_in = held[:, : v.frames]
        g = torch.Generator().manual_seed(HELD_IN_SEED)
        self.held_in_masks = [
            multiblock_masks(len(held), self.grid, m, keep=k, generator=g)
            for m, k in zip(v.context_masks, self.keeps, strict=True)
        ]

    @torch.no_grad()
    def evaluate(
        self, mods: VisionModules, split: str, *, version: int = 0
    ) -> dict[str, float]:
        """The row of ``split`` of ``mods`` as they are (``version`` unused:
        no calibration)."""
        v, dev = self.v, self._dev
        windows, masks = self.windows[split], self.masks[split]
        dz, dt = self._directions[split]
        terms, zs, spreads = [], [], []
        for i in range(0, len(windows), self._batch):
            chunks = windows[i : i + self._batch, : v.frames].to(dev)
            b = len(chunks)
            views = [m[i : i + b].to(dev) for m in masks]
            h, y = pack_views(
                chunks, views, keeps=self.keeps, patch=v.patch, tubelet=v.tubelet
            )
            tokens = mods.encoder(h, mods.segments)
            spreads.append(token_spread(tokens[:, : mods.segments[0]], self._slots))
            t = jepa_loss_packed(
                tokens,
                y,
                torch.ones(b, dtype=torch.bool, device=dev),
                dz,
                dt,
                mods.predictor,
                keeps=list(mods.segments[1:]),
                proj=mods.proj,
                lam=v.sigreg,
                sg=False,
                sigreg_on=v.sigreg_on,
                pred_loss=v.pred_loss,
            )
            terms.append(torch.stack([t.loss, t.pred, t.sigreg]))
            zs.append(t.z)
        loss, pred, sig = torch.stack(terms).mean(0).tolist()
        row = {"loss": loss, "pred": pred, "sigreg": sig}
        row |= z_stats(torch.cat(zs))
        within, temporal, means = (torch.cat(x) for x in zip(*spreads, strict=True))
        row |= spread_stats(within, temporal, means, self._slots)
        return row | self.completion_rows(mods, [split])[split]

    def _kinds(self, split: str) -> dict[str, tuple[Tensor, int]]:
        """The token masks ``(n, T)`` and the visible tokens of a row of each
        mask kind on the stacks of ``split`` (``"fit"``: ``held_in``)."""
        masks = self.held_in_masks if split == "fit" else self.masks[split]
        slots, per = self._slots, self.grid[0] * self.grid[1]
        out = {
            view: (tube_tokens(m, slots), k * slots)
            for view, m, k in zip(self.views, masks, self.keeps, strict=True)
        }
        if self._future:
            grid3 = (slots, *self.grid)
            out[FUTURE] = (
                future_tokens(len(masks[0]), grid3, self._future),
                (slots - self._future) * per,
            )
        return out

    @torch.no_grad()
    def completion(
        self, mods: VisionModules, splits: list[str]
    ) -> dict[str, dict[str, float]]:
        """The completion errors (``measure.completion.completion``) of
        ``mods`` on the held-in stacks (``"fit"``, whose mean target token is
        the ``_mean`` predictor of every split) and on the first stack of
        each window of ``splits``, by split."""
        v, b, dev = self.v, self._batch, self._dev
        mean = mean_token(mods, self.held_in, batch=b, device=dev)
        sets = {"fit": self.held_in}
        sets |= {s: self.windows[s][:, : v.frames] for s in splits}
        return {
            s: completion(mods, x, self._kinds(s), mean_token=mean, batch=b, device=dev)
            for s, x in sets.items()
        }

    def completion_rows(
        self, mods: VisionModules, splits: list[str]
    ) -> dict[str, dict[str, float]]:
        """The completion keys of the rows of ``splits`` (of ``"val"`` and
        ``"test"``): those of the split, and in the row of val also those of
        calib (``calib_``) and of the held-in stacks (``fit_``)."""
        comp = self.completion(
            mods, [*splits, *(("calib",) if "val" in splits else ())]
        )
        out = {s: dict(comp[s]) for s in splits}
        if "val" in out:
            out["val"] |= {f"calib_{k}": x for k, x in comp["calib"].items()}
            out["val"] |= {f"fit_{k}": x for k, x in comp["fit"].items()}
        return out

    def baselines(self, mods: VisionModules) -> dict[str, dict[str, float]]:
        """``init_*``: the rows of ``mods`` as they are (before learning)."""
        return {
            s: {f"init_{k}": x for k, x in self.evaluate(mods, s).items()}
            for s in _SPLITS
        }

    def visualize(self, out: Path, patch: int) -> None:
        """``views.png``: 4 windows of val, a row each: the first stack's $N$
        frames, then its last frame as each context view shows it (the
        masked patches 0)."""
        v = self.v
        chunks = self.windows["val"][:4, : v.frames].float() / 255
        last = chunks[:, -1]  # (4, C, H, W)
        shown = []
        for m in self.masks["val"]:
            px = m[:4].repeat_interleave(patch, 1).repeat_interleave(patch, 2)
            shown.append(last * (~px)[:, None].float())
        grid = torch.cat([chunks, torch.stack(shown, 1)], 1)
        out.mkdir(parents=True, exist_ok=True)
        write_png(out / "views.png", tile(grid, scale=1))


def _shrunk(chunks: Tensor) -> Tensor:
    """The frames of ``chunks`` ``(n, N, C, H, W)`` shrunk to ``THUMB`` x
    ``THUMB`` by the mean of each block, flattened: ``(n, N C THUMB^2)``."""
    frames = nn.functional.adaptive_avg_pool2d(chunks.flatten(0, 1), THUMB)
    return frames.unflatten(0, chunks.shape[:2]).flatten(1)


def _clean(d: Mapping[str, float]) -> dict[str, float | None]:
    return {k: (x if math.isfinite(x) else None) for k, x in d.items()}


def _frames(conf: RunConf) -> FrameStreams | StackStreams:
    b, frame, v = conf.data.streams, conf.time.frame, conf.vision
    assert v is not None
    if conf.data.balls is not None:
        if v.layout == "packed":
            raise ValueError("vision.layout 'packed' streams Walking Tours only")
        return BallFrames.of(conf.data.balls, b, frame=frame, seed=conf.seed)
    assert conf.data.tours is not None
    t = conf.data.tours
    if v.layout == "packed":
        return StackedTourFrames(
            tours_root(t),
            n=b,
            held_out=t.held_out,
            frames=v.frames,
            stride=v.stride,
            advance=t.advance,
        )
    if t.advance != 1:
        raise ValueError(f"data.tours.advance {t.advance} needs vision.layout 'packed'")
    return TourFrames(tours_root(t), n=b, held_out=t.held_out)


class _Online:
    """The tasks ``"online"`` and ``"bench"`` (no evaluations)."""

    def __init__(
        self, conf: RunConf, run_dir: Path, resume: Mapping[str, Any] | None
    ) -> None:
        v = conf.vision
        if v is None:
            raise ValueError(f"the run {conf.name!r} has no vision")
        if v.frozen:
            raise ValueError("vision.frozen is for the joint program (M16)")
        if conf.data.balls is not None and conf.data.balls.obs != "frames":
            raise ValueError(
                f"the vision sees images: data.balls.obs must be 'frames', not "
                f"{conf.data.balls.obs!r}"
            )
        self.conf = conf
        self._recon = v.recon > 0
        bench = conf.task == "bench"
        self.net = build_vision_net(conf)
        self.mods = vision_modules(self.net)
        self._packed = v.layout == "packed"
        self.measures: Measures | PackedMeasures | None = None
        if not bench:
            self.measures = PackedMeasures(conf) if self._packed else Measures(conf)
        if self.measures is not None:  # of the initial weights, before a resume
            base = self.measures.baselines(self.mods)
            write_json(
                run_dir / "baselines.json", {s: _clean(b) for s, b in base.items()}
            )
        self.streams = _frames(conf)
        frame, b = conf.time.frame, conf.data.streams
        dev = torch.device(conf.device)
        self._y = torch.zeros(b, 1, device=dev)
        self._mask = torch.ones(b, dtype=torch.bool, device=dev)
        # The sums since the last log: loss, pred, sigreg, recon, and the frames, in
        # host memory, where Host.show reads the stats (as in programs.dynamics).
        self._acc = torch.zeros(5, dtype=torch.float64)
        state = None if resume is None else resume["mujou"]
        self._run: ManualRun = mujou.testing.run(self.net, state=state)
        self.host = Host(self._run, frame=frame, period=self.net.tau.period)
        if resume is not None:
            h = resume["host"]
            self.host.load_state_dict(h["host"])
            self.streams.load_state_dict(h["streams"])
            self._acc.copy_(h["acc"])
        ctl, m = conf.ctl, conf.measure
        self._schedule = Schedule(
            units=round(conf.seconds / frame),
            unit="frame",
            seconds=frame,
            log=every(ctl.log_every, frame),
            evaluate=0 if bench else every(m.every, frame),
            ckpt=every(ctl.ckpt_every, frame),
            viz=0 if bench else every(ctl.viz_every, frame),
            resume=every(ctl.resume_every, frame),
            test=ctl.eval_test and not bench,
        )

    @property
    def schedule(self) -> Schedule:
        return self._schedule

    def advance(self) -> None:
        k = self.host.frames
        images, extra, reset = self.streams.next()
        dev = self._y.device
        # A stack's ready (packed) is its mask; a frame of the balls or the
        # tours is always shown.
        mask = extra.to(dev) if self._packed else self._mask
        sample = pc.msg.Sample(
            x=images.to(dev), y=self._y, mask=mask, reset=reset.to(dev)
        )
        p, _ = self.host.show({self.net.prev: sample}, self.net.stats)
        if k == 0:
            return
        counted = p["n"] > 0
        terms = torch.stack(
            [p["loss"], p["pred"], p["sigreg"], p["recon"], torch.ones_like(p["n"])]
        )
        self._acc += torch.where(counted, terms.double(), 0.0)

    def log(self) -> dict[str, float | None]:
        loss, pred, sig, rec, n = self._acc.tolist()
        self._acc.zero_()
        names = ("loss", "pred", "sigreg", *(("recon",) if self._recon else ()))
        if not n:
            return dict.fromkeys(names)
        sums = {"loss": loss, "pred": pred, "sigreg": sig, "recon": rec}
        return {k: sums[k] / n for k in names}

    def evaluate(self, split: str) -> dict[str, float]:
        assert self.measures is not None  # no evaluations in a bench
        row = self.measures.evaluate(self.mods, split, version=self.host.frames)
        objective = vision_objective(row) if split == "val" else None
        if objective is not None:
            row["objective"] = objective
        return row

    def diverged(self, row: Mapping[str, float]) -> str | None:
        keys = ("loss", "pred", "sigreg", *(("recon",) if self._recon else ()))
        bad = [k for k in keys if not math.isfinite(row[k])]
        return f"{', '.join(bad)} not finite" if bad else None

    def objective(
        self, row: Mapping[str, Any] | None, *, diverged: bool
    ) -> float | None:
        if diverged:
            return PENALTY
        return None if row is None else vision_objective(row)

    def modules(self) -> dict[str, nn.Module]:
        return self.mods.named()

    def visualize(self, out: Path) -> None:
        if self.measures is not None:
            self.measures.visualize(out, self.mods.patch)

    def resume_state(self) -> tuple[Mapping[str, Any], dict[str, Any]]:
        host = {
            "host": self.host.state_dict(),
            "streams": self.streams.state_dict(),
            "acc": self._acc.cpu().clone(),
        }
        return self._run.state_dict(), host

    def final(self) -> dict[str, dict[str, Any]]:
        return {}

    def close(self) -> None:
        self._run.shutdown()


class VisionProgram:
    """The program ``"vision"`` (module docstring), a ``loop.Program``."""

    def build(
        self, conf: RunConf, run_dir: Path, resume: Mapping[str, Any] | None
    ) -> Built:
        match conf.task:
            case "online" | "bench":
                return _Online(conf, run_dir, resume)
            case "posthoc" | "baselines":  # imported here: it imports this module
                from exp.video.programs.posthoc import Posthoc

                return Posthoc(conf, run_dir, resume)
            case task:
                raise ValueError(f"the vision program has no task {task!r}")


def rebuild(conf: RunConf) -> dict[str, nn.Module]:
    """Fresh modules of ``conf`` by the names of the evaluation checkpoint, to
    load one into (``exp.lib.ckpt.load_eval``)."""
    return build_vision(conf, seed=conf.seed).named()


PROGRAM = VisionProgram()
