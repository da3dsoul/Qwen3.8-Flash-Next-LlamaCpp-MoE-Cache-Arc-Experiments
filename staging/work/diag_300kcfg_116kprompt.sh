#!/bin/bash
# Is the 300K degeneracy caused by DEPTH (past n_ctx_train, YaRN or not) or by
# the 300K CONFIG (-c 311296 / -ncmoe 32 / YaRN active)?
# Decisive arm: the KNOWN-GOOD 116K prompt (116,277 tokens, comfortably under
# n_ctx_train 262,144) run through the EXACT 300K server config.
#   coherent -> config is fine, the problem is depth past native context
#   garbage  -> the config itself is broken, and the 300K numbers mean nothing
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.."
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$ROOT/logs/long-context-300kcfg-116kprompt"
mkdir -p "$OUT"; cd "$ROOT"
docker compose -f docker/docker-compose.yml run --rm --service-ports \
    --entrypoint /usr/bin/env -v "$ROOT/staging/devbin:/devbin" \
    -e NCTX=311296 -e NCMOE=32 -e UB=2048 \
    llm-test-sycl bash -c "/work/bench300k/serve.sh" > "$OUT/server.log" 2>&1 &
SW=$!
for i in $(seq 1 1200); do curl -sf http://localhost:8090/health >/dev/null 2>&1 && break; sleep 1; done
python3 "$ROOT/staging/work/bench300k/run_bench.py" localhost:8090 \
    "$ROOT/staging/work/bench120k/full_prompt.txt" "$OUT/run1" 200 2>&1 | tee "$OUT/client.log"
docker stop qwen4exp-test-sycl >/dev/null 2>&1; wait $SW 2>/dev/null; echo done
