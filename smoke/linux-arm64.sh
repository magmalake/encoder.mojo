#!/bin/sh
# Run the CPU smoke test on linux-aarch64 in a container (the Hetzner CAX
# architecture). Usage: smoke/linux-arm64.sh [env]   (env: default|nightly)
set -eu
env=${1:-default}
docker run --rm --platform linux/arm64 -v "$PWD":/src:ro -w /work \
  -e ENV="$env" ghcr.io/prefix-dev/pixi:latest sh -c '
    set -e
    cp /src/pixi.toml . && cp -r /src/smoke . && mkdir build
    pixi install -e "$ENV" >/dev/null
    if ! pixi run -e "$ENV" mojo build -O3 smoke/cpu.mojo -o build/smoke-cpu >build.log 2>&1; then
      grep -v "warning" build.log | grep -E "error" -A4 | head -40
      exit 1
    fi
    echo "binary: $(du -h build/smoke-cpu | cut -f1), cores: $(nproc)"
    ./build/smoke-cpu
    echo "runtime libraries from the pixi env:"
    ldd build/smoke-cpu | grep /work/.pixi | while read -r name _ path _; do
      echo "  $name $(du -h "$path" | cut -f1)"
    done'
