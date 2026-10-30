"""One run of the continuous-stream experiment.

uv run python -m experiments.continuous_stream.run --config <path>
"""

import argparse
from pathlib import Path

import mujou.pc as pc
import mujou.testing
import torch
from torch import nn

from experiments.continuous_stream.bp import bp_chain
from experiments.continuous_stream.config import Config
from experiments.continuous_stream.host import Host
from experiments.continuous_stream.optim import from_config
from experiments.continuous_stream.record import REPO_ROOT, Recorder
from experiments.continuous_stream.stream0 import LinearStream
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
        return recorder.dir


if __name__ == "__main__":
    print(main())
