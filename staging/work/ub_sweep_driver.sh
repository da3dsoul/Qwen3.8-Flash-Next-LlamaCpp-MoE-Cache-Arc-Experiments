#!/bin/bash
# Host-side driver: sequential -ub sweep inside one llm-test-sycl container.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

docker compose -f docker/docker-compose.yml run --rm \
    --entrypoint /usr/bin/env \
    -v "$ROOT/staging/devbin:/devbin" \
    llm-test-sycl bash -c '
      for ub in 512 1024 2048 4096; do
        /work/ub_sweep.sh "$ub"
        echo "@@@ SEP"
      done
    ' > logs/ub-sweep-32k.log 2>&1
echo "sweep done rc=$?"
