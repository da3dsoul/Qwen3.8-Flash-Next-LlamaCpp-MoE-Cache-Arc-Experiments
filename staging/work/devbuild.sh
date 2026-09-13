#!/bin/bash
# Incremental dev build loop (host side).
#
# The project Dockerfile rebuilds llama.cpp from scratch on every source edit,
# which costs ~15 min per iteration. This copies only the edited files into a
# long-lived container started from the Dockerfile's `build` stage
# (qwen4exp-moe-cache:sycl-buildenv), runs an incremental cmake build there, and
# copies the fresh .so / binaries back to staging/devbin. Run the model with
# staging/devbin bind-mounted over /app to test.
#
# Usage: staging/work/devbuild.sh <file-relative-to-src/llama.cpp> [...]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONTAINER=gdnbuild

if [ "$#" -eq 0 ]; then
    echo "usage: $0 <file-relative-to-src/llama.cpp> [...]" >&2
    exit 1
fi

# the build container is long-lived but may be stopped between sessions;
# recreate it from the build-stage image if it is gone
if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER"; then
        docker start "$CONTAINER" >/dev/null
    else
        echo "creating $CONTAINER (needs qwen4exp-moe-cache:sycl-buildenv:"
        echo "  docker build --target build -t qwen4exp-moe-cache:sycl-buildenv -f docker/Dockerfile.sycl-dev .)"
        docker run -d --name "$CONTAINER" qwen4exp-moe-cache:sycl-buildenv sleep infinity >/dev/null
    fi
fi

for f in "$@"; do
    echo "copy $f"
    docker cp "$ROOT/src/llama.cpp/$f" "$CONTAINER:/app/$f"
done

docker exec "$CONTAINER" cmake --build build --config Release -j16

mkdir -p "$ROOT/staging/devbin"
docker cp "$CONTAINER:/app/build/bin/." "$ROOT/staging/devbin/"
echo "devbin updated: $ROOT/staging/devbin"
