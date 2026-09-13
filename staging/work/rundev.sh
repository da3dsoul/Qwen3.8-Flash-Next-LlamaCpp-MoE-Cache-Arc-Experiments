#!/bin/bash
# Run a command in the SYCL test container against a chosen devbin.
#
# Do NOT add SYCL_CACHE_PERSISTENT / NEO_CACHE_PERSISTENT here. Tried on 2026-09-12 to amortise
# kernel compilation across runs; the box then crashed mid-write and every later run died with a
# general protection fault in glibc's EVEX strcmp on a non-canonical pointer, on binaries that had
# worked minutes earlier. Removing the two cache dirs fixed it immediately.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="$1"; shift
exec docker compose -f "$ROOT/docker/docker-compose.yml" run --rm \
    --entrypoint /usr/bin/env \
    -v "$ROOT/$BIN:/devbin" \
    llm-test-sycl bash -c "$*"
