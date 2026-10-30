# mujou-exp

Experiments with [mujou](https://github.com/Geson-anko/mujou), the asynchronous distributed deep learning library for continuous learning.

mujou is a git submodule (`mujou/`) installed in editable mode. When an experiment needs something mujou lacks, add it in the submodule, send a pull request to mujou, and keep experimenting on top of the change.

## Setup

```sh
git clone --recurse-submodules https://github.com/Geson-anko/mujou-exp.git
cd mujou-exp
just setup   # git submodule update --init, uv venv --clear, uv sync, pre-commit install
```

`uv sync` builds mujou's native module (Rust, via maturin), so it needs [uv](https://docs.astral.sh/uv/), [just](https://just.systems/) 1.58 or later and [rustup](https://rustup.rs/). The dev container has all of them.

## Dev container

`.devcontainer/` is the same environment as mujou's: Ubuntu with uv, just, gh, the Rust toolchain and Claude Code. Open the checkout in VS Code and choose "Reopen in Container", or run any command through the [Dev Container CLI](https://github.com/devcontainers/cli):

```sh
just container just run
```

The container uses its own virtualenv, `.venv-devcontainer/`, so that it does not collide with the host's `.venv/`.

## Commands

| Recipe              | What it does                                                            |
| ------------------- | ----------------------------------------------------------------------- |
| `just setup`        | Fetch the submodule, create the virtualenv, install the pre-commit hook |
| `just format`       | Run the pre-commit hooks on all files                                   |
| `just test`         | pytest of this project                                                  |
| `just test-fast`    | pytest without the `slow` tests                                         |
| `just type`         | pyright over `experiments/` and `tests/`                                |
| `just run`          | format → test → type. **Run it before committing**                      |
| `just mujou-status` | State of the submodule; fails if its commit is not on origin            |
| `just mujou-update` | Move the submodule to the latest `origin/main`                          |
| `just container …`  | Run a command in the dev container                                      |

## Working with mujou

See [CLAUDE.md](CLAUDE.md) and the skill `mujou-upstream` (`.claude/skills/mujou-upstream/SKILL.md`).
