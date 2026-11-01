#!/usr/bin/env bash
# Point .devcontainer/.mounts/egodex at the host's EgoDex directory, which
# devcontainer.json bind-mounts read-only. Runs on the host (initializeCommand).
# The directory comes from EGODEX_DIR in the environment or in .env; without
# it, the link points at an empty directory so the container still starts.
set -euo pipefail
cd "$(dirname "$0")/.."

dir="${EGODEX_DIR:-}"
if [ -z "$dir" ] && [ -f .env ]; then
    dir="$(sed -n 's/^[[:space:]]*EGODEX_DIR=//p' .env | tail -n 1)"
    dir="${dir%\"}"; dir="${dir#\"}"; dir="${dir%\'}"; dir="${dir#\'}"
    dir="${dir/#\~/$HOME}"
fi

mkdir -p .devcontainer/.mounts/empty
if [ -z "$dir" ]; then
    target="$PWD/.devcontainer/.mounts/empty"
elif [ -d "$dir" ]; then
    target="$(cd "$dir" && pwd -P)"
else
    echo "EGODEX_DIR is not a directory: $dir" >&2
    exit 1
fi
ln -sfn "$target" .devcontainer/.mounts/egodex
