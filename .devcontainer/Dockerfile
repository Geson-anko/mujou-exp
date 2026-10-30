FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04

# Development tools. gh comes from GitHub's apt repository (Ubuntu's is old).
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        -o /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        > /etc/apt/sources.list.d/github-cli.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends gh git tmux ripgrep jq \
    && rm -rf /var/lib/apt/lists/*

# uv manages Python (3.14t from .python-version) and the virtualenv.
COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /uvx /usr/local/bin/

# just >= 1.58 is required for the `[windows] set shell` syntax in the justfile.
RUN curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh \
    | bash -s -- --tag 1.58.0 --to /usr/local/bin

# Rust toolchain for the native core (mujou._native).
USER vscode
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --no-modify-path --profile minimal --component rustfmt,clippy
ENV PATH="/home/vscode/.cargo/bin:${PATH}"

# Claude Code (native installer, into ~/.local/bin).
RUN curl -fsSL https://claude.ai/install.sh | bash
ENV PATH="/home/vscode/.local/bin:${PATH}"
