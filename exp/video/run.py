"""One run of the video experiment.

uv run python -m exp.video.run --config <path>
"""

import argparse
import copy
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import mujou.pc as pc
import mujou.testing
import torch
from mujou.testing import ManualRun
from torch import nn

from exp.lib.host import Host, RealtimeHost
from exp.lib.linear_stream import LinearStream
from exp.lib.record import REPO_ROOT, Recorder
from exp.video.balls import Frame, Physics, Streams, Video, eval_clips, ordered
from exp.video.bp import WorldBPNetwork, bp_chain
from exp.video.config import Config, make_optimizer
from exp.video.evaluate import Snapshot, evaluate
from exp.video.model import Network
from exp.video.nodes import open_mask
from mujou import TensorSpec


def run_linear(config: Config, recorder: Recorder) -> None:
    """Stage 0: a linear regression (or an MLP of ``model.hidden``) on the
    drifting stream, by PC-ALM or BP (``config.bp``).

    Each frame records the prequential squared error ``loss`` and, for the
    linear model, ``dist`` = ||w - w_*||; each ``eval_interval`` seconds,
    ``iid_loss`` on samples of the marginal distribution.
    """
    d, t = config.data, config.time
    torch.manual_seed(config.seed)  # the initial weights
    modules: list[nn.Module]
    if config.model.hidden == 0:
        modules = [nn.Linear(d.d_x, 1)]
    else:
        h = config.model.hidden
        modules = [nn.Sequential(nn.Linear(d.d_x, h), nn.Tanh()), nn.Linear(h, 1)]
    tau = pc.TimeConstants(h=t.h, lam=t.lam, r=t.r)
    make = bp_chain if config.bp else pc.Chain
    net = make(
        *modules,
        x=TensorSpec((d.d_x,)),
        y=TensorSpec((1,)),
        loss=pc.loss.MSELoss(),
        tau=tau,
        optimizer=make_optimizer(config.optim),
        learn_interval=t.learn_interval,
    )
    forward = nn.Sequential(*modules)  # the same modules, to evaluate
    stream = LinearStream(
        d=d.d_x,
        tau_x=d.tau_x,
        frame=t.frame,
        noise=d.noise,
        iid=d.iid,
        seed=config.seed,
    )
    test_x, test_y = stream.sample_iid(1024, seed=config.seed + 1000)
    frames = round(d.seconds / t.frame)
    every = max(1, round(config.eval_interval / t.frame))
    yes = torch.tensor(True)
    with mujou.testing.run(net) as run:
        host = Host(run, frame=t.frame, period=tau.period)
        for k in range(frames):
            x, y = stream.next()
            sample = pc.msg.Sample(x=x, y=y, mask=yes, reset=torch.tensor(k == 0))
            p, written = host.show({net.prev: sample}, net.next)
            values = {"frame": k, "loss": float(((p["y_hat"] - y) ** 2).mean())}
            if config.model.hidden == 0:
                w = next(modules[0].parameters()).detach()[0]
                values["dist"] = float(torch.linalg.vector_norm(w - stream.w))
            if (k + 1) % every == 0:
                with torch.no_grad():
                    values["iid_loss"] = float(((forward(test_x) - test_y) ** 2).mean())
            recorder.log(float(written), **values)


def snapshot(net: Network | WorldBPNetwork) -> Snapshot:
    """Copies of the modules of ``net``, to evaluate."""
    if isinstance(net, WorldBPNetwork):
        w = net.world
        return Snapshot(
            copy.deepcopy(w.encoder),
            copy.deepcopy(w.step),
            copy.deepcopy(w.readout),
            w.residual,
        )
    return Snapshot.of(net)


def save_checkpoint(snap: Snapshot, run_dir: Path, seconds: float) -> Path:
    """Write ``snap`` to ``<run_dir>/ckpt/<seconds>.pt`` and return the path."""
    path = run_dir / "ckpt" / f"{round(seconds):05d}.pt"
    path.parent.mkdir(exist_ok=True)
    torch.save(snap, path)
    return path


def load_checkpoint(path: Path, device: str = "cpu") -> Snapshot:
    """The ``Snapshot`` that ``save_checkpoint`` wrote (pickled modules: load
    only checkpoints of your own runs)."""
    return torch.load(path, map_location=device, weights_only=False)


def source(config: Config) -> tuple[Any, Frame]:
    """The stream of frames (``next()`` gives a ``Frame``) and the clips to
    evaluate on."""
    d, t = config.data, config.time
    if d.kind == "egodex":
        from exp.video.egodex import EgoDexStream, egodex_clips

        return EgoDexStream(d.path), egodex_clips(d.path)
    physics = Physics(period=d.physics_period)
    kwargs: dict[str, Any] = dict(
        size=d.size,
        balls=d.balls,
        frame=t.frame,
        physics=physics,
        episode=(d.episode_min, d.episode_max),
    )
    video = (
        Video(seed=config.seed, **kwargs)
        if d.streams == 1
        else Streams(d.streams, seed=config.seed, spread=d.spread, **kwargs)
    )
    clips = eval_clips(size=d.size, balls=d.balls, frame=t.frame, physics=physics)
    return video, clips


def run_video(config: Config, recorder: Recorder) -> None:
    """Stages 1-3: the world model on the synthetic video or EgoDex, by PC-ALM
    or BP (``config.bp``).

    Each frame records the prequential ``error`` (1/D ||z_hat - z||^2, on the
    frames with a prediction from the observed z; ``open_error`` on those from
    the model's own, with ``model.open_loop``) and ``z_rms`` (the root mean
    square of z); with ``data.streams`` = B > 1 streams, each is the mean over
    the streams that have such a prediction in the frame (and ``z_rms`` over
    all of them); each ``eval_interval`` seconds, the metrics of
    ``evaluate`` on the clips and, for PC-ALM, the correlation ``rho_<node>_<lag>``
    of each node's gradients over the interval; each ``checkpoint_interval``
    seconds of frames, a checkpoint (``save_checkpoint``). The frames come in
    ``data.order`` (``balls.ordered``); a replayed frame records only its
    ``replay_error``, and the metrics count the live frames.
    """
    d, t = config.data, config.time
    video, clips = source(config)
    dev = config.device
    clips = clips._replace(
        image=clips.image.to(dev),
        state=clips.state.to(dev),
        params=clips.params.to(dev),
    )
    net = WorldBPNetwork(config) if config.bp else Network(config, correlation=True)
    d_state = config.model.d_s * config.model.dyn_nodes
    frames = round(d.seconds / t.frame)
    every = max(1, round(config.eval_interval / t.frame))
    ckpt_every = round(config.checkpoint_interval / t.frame)  # 0: none
    shown: Iterator[tuple[Frame, bool]]
    if isinstance(video, Streams):
        shown = ((video.next(), True) for _ in range(frames))
    else:
        shown = ordered(
            video,
            frames=frames,
            order=d.order,
            clip=round(d.clip / t.frame),
            replay=d.replay,
            seed=config.seed + 1,
        )
    k = 0  # the live frames shown
    count = torch.zeros(d.batch, dtype=torch.int64)  # predictions of each video
    if config.realtime:
        context = mujou.run(net)
    else:
        context = mujou.testing.run(net)
    with context as run:
        if isinstance(run, ManualRun):
            host: Host | RealtimeHost = Host(run, frame=t.frame, period=net.tau.period)
        else:
            host = RealtimeHost(run, frame=t.frame)
        for f, live in shown:
            reset = torch.as_tensor(f.reset)
            sample = pc.msg.Sample(
                x=f.image, y=torch.zeros(*d.batch, 1), mask=~reset, reset=reset
            )
            p, written = host.show({net.prev: sample}, net.next)
            if not live:
                if p is not None and bool(p["mask"]):
                    error = float(((p["y_hat"] - p["y"]) ** 2).mean())
                    recorder.log(float(written), replay_error=error)
                continue
            values: dict[str, float | int] = {"frame": k}
            count = (count + 1).masked_fill(reset, 0)  # of each video so far
            m = config.model
            opened = (count > 1) & open_mask(count, m.open_loop, m.open_cycle)
            if p is None:  # real time: no prediction of the frame in time
                values["missed"] = int((~reset).sum())
            else:
                has = p["mask"].cpu()
                error = ((p["y_hat"] - p["y"]) ** 2).mean(-1).cpu()
                for key, which in (
                    ("error", has & ~opened),
                    ("open_error", has & opened),
                ):
                    if bool(which.any()):
                        values[key] = float(error[which].mean())
            if p is not None:  # the root mean square of z_k: 1 for N(0, I)
                values["z_rms"] = float(p["y"].pow(2).mean().sqrt())
            # The modules are read between steps only: not while a real-time run runs.
            if (k + 1) % every == 0 and not config.realtime:
                values.update(evaluate(snapshot(net), clips, d_state))
                if isinstance(net, Network):
                    for name, corr in net.correlations.items():
                        for lag, rho in corr.take().items():
                            values[f"rho_{name}_{lag}"] = rho
            if ckpt_every and (k + 1) % ckpt_every == 0 and not config.realtime:
                save_checkpoint(snapshot(net), recorder.dir, (k + 1) * t.frame)
            recorder.log(float(written), **values)
            k += 1
    if config.realtime:  # once the run has stopped
        recorder.log(d.seconds, **evaluate(snapshot(net), clips, d_state))


def main(argv: list[str] | None = None) -> Path:
    """Run the stream of the settings, and return the directory of the run."""
    parser = argparse.ArgumentParser(description="One run of the video experiment.")
    parser.add_argument(
        "--config", type=Path, help="settings (JSON); the defaults if left out"
    )
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "runs", help="where runs go"
    )
    parser.add_argument("--dry", action="store_true", help="write config.json only")
    args = parser.parse_args(argv)
    config = Config() if args.config is None else Config.load(args.config)
    torch.set_num_threads(1)
    with Recorder(config, args.out) as recorder:
        if not args.dry and config.data.kind == "linear":
            run_linear(config, recorder)
        elif not args.dry and config.data.kind in ("video", "egodex"):
            run_video(config, recorder)
        return recorder.dir


if __name__ == "__main__":
    print(main())
