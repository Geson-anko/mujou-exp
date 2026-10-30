"""One run of the continuous-stream experiment.

uv run python -m experiments.continuous_stream.run --config <path>
"""

import argparse
import copy
from pathlib import Path

import mujou.pc as pc
import mujou.testing
import torch
from mujou.testing import ManualRun
from torch import nn

from experiments.continuous_stream.bp import WorldBPNetwork, bp_chain
from experiments.continuous_stream.config import Config
from experiments.continuous_stream.evaluate import Snapshot, evaluate
from experiments.continuous_stream.host import Host, RealtimeHost
from experiments.continuous_stream.model import Network
from experiments.continuous_stream.optim import from_config
from experiments.continuous_stream.record import REPO_ROOT, Recorder
from experiments.continuous_stream.stream0 import LinearStream
from experiments.continuous_stream.video import Physics, Video, eval_clips, ordered
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
        optimizer=from_config(config.optim),
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
            copy.deepcopy(w.encoder), copy.deepcopy(w.step), copy.deepcopy(w.readout)
        )
    return Snapshot.of(net)


def run_video(config: Config, recorder: Recorder) -> None:
    """Stages 1-2: the world model on the synthetic video, by PC-ALM or BP
    (``config.bp``).

    Each frame records the prequential ``error`` (1/D ||z_hat - z||^2, on the
    frames with a prediction); each ``eval_interval`` seconds, the metrics of
    ``evaluate`` on the clips and, for PC-ALM, the correlation ``rho_<node>_<lag>``
    of each node's gradients over the interval. The frames come in
    ``data.order`` (``video.ordered``); a replayed frame records only its
    ``replay_error``, and the metrics count the live frames.
    """
    d, t = config.data, config.time
    physics = Physics(period=d.physics_period)
    video = Video(
        size=d.size,
        balls=d.balls,
        frame=t.frame,
        physics=physics,
        episode=(d.episode_min, d.episode_max),
        seed=config.seed,
    )
    clips = eval_clips(size=d.size, balls=d.balls, frame=t.frame, physics=physics)
    net = WorldBPNetwork(config) if config.bp else Network(config, correlation=True)
    frames = round(d.seconds / t.frame)
    every = max(1, round(config.eval_interval / t.frame))
    shown = ordered(
        video,
        frames=frames,
        order=d.order,
        clip=round(d.clip / t.frame),
        replay=d.replay,
        seed=config.seed + 1,
    )
    k = 0  # the live frames shown
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
            sample = pc.msg.Sample(
                x=f.image,
                y=torch.zeros(1),
                mask=torch.tensor(not f.reset),
                reset=torch.tensor(f.reset),
            )
            p, written = host.show({net.prev: sample}, net.next)
            if not live:
                if p is not None and bool(p["mask"]):
                    error = float(((p["y_hat"] - p["y"]) ** 2).mean())
                    recorder.log(float(written), replay_error=error)
                continue
            values: dict[str, float | int] = {"frame": k}
            if p is None:  # real time: no prediction of the frame in time
                values["missed"] = int(not f.reset)
            elif bool(p["mask"]):
                values["error"] = float(((p["y_hat"] - p["y"]) ** 2).mean())
            # The modules are read between steps only: not while a real-time run runs.
            if (k + 1) % every == 0 and not config.realtime:
                values.update(evaluate(snapshot(net), clips, config.model.d_s))
                if isinstance(net, Network):
                    for name, corr in net.correlations.items():
                        for lag, rho in corr.take().items():
                            values[f"rho_{name}_{lag}"] = rho
            recorder.log(float(written), **values)
            k += 1
    if config.realtime:  # once the run has stopped
        recorder.log(d.seconds, **evaluate(snapshot(net), clips, config.model.d_s))


def main(argv: list[str] | None = None) -> Path:
    """Run the stream of the settings, and return the directory of the run."""
    parser = argparse.ArgumentParser(
        description="One run of the continuous-stream experiment."
    )
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
        elif not args.dry and config.data.kind == "video":
            run_video(config, recorder)
        return recorder.dir


if __name__ == "__main__":
    print(main())
