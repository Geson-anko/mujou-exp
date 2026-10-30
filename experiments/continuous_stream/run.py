"""One run of the continuous-stream experiment.

uv run python -m experiments.continuous_stream.run --config <path>
"""

import argparse
from pathlib import Path

from experiments.continuous_stream.config import Config
from experiments.continuous_stream.record import REPO_ROOT, Recorder


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
    args = parser.parse_args(argv)
    config = Config() if args.config is None else Config.load(args.config)
    with Recorder(config, args.out) as recorder:
        return recorder.dir


if __name__ == "__main__":
    print(main())
