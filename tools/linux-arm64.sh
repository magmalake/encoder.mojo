#!/bin/sh
# Run a pixi task on linux-aarch64 (the Hetzner CAX architecture) in a fresh
# container. Usage: tools/linux-arm64.sh [task] [env]
#   tools/linux-arm64.sh test          # the test suite
#   tools/linux-arm64.sh smoke-cpu     # the max-core smoke test
set -eu
task=${1:-test}
env=${2:-default}
docker run --rm --platform linux/arm64 -v "$PWD":/src:ro -w /work \
  -e TASK="$task" -e ENV="$env" ghcr.io/prefix-dev/pixi:latest sh -c '
    set -e
    apt-get update -qq >/dev/null && apt-get install -y -qq curl >/dev/null
    cp -r /src/pixi.toml /src/pixi.lock /src/src /src/tests /src/tools /src/smoke /src/bench .
    pixi install -e "$ENV" >/dev/null
    echo "cores: $(nproc)"
    pixi run -e "$ENV" "$TASK" 2>&1 | grep -v -e "warning" -e "^\s*$"'
