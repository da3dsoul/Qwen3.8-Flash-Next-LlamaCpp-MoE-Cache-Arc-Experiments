#!/bin/bash
# Re-run the 116K real-task long-context benchmark with the radix-select TOP_K
# fix in /devbin, against the same prompt and the same server config as
# logs/long-context-120k-benchmark-report.md (the 68.21 tok/s prefill baseline).
#
# Server runs in the llm-test-sycl container with --service-ports (8090 -> 8080);
# the client (run_bench.py) runs on the host, exactly as the baseline pass did.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$ROOT/logs/long-context-120k-after"
MAXTOK="${MAXTOK:-2000}"
mkdir -p "$OUT"

cd "$ROOT"
docker compose -f docker/docker-compose.yml run --rm --service-ports \
    --entrypoint /usr/bin/env \
    -v "$ROOT/staging/devbin:/devbin" \
    llm-test-sycl bash -c '/work/bench120k/serve.sh' > "$OUT/server.log" 2>&1 &
SERVER_WAIT=$!

echo "waiting for server health..."
for i in $(seq 1 900); do
    if curl -sf http://localhost:8090/health >/dev/null 2>&1; then
        echo "server up after ${i}s"
        break
    fi
    sleep 1
done

curl -sf http://localhost:8090/health || { echo "server never came up"; }

python3 "$ROOT/staging/work/bench120k/run_bench.py" \
    localhost:8090 \
    "$ROOT/staging/work/bench120k/full_prompt.txt" \
    "$OUT/run1" \
    "$MAXTOK" 2>&1 | tee "$OUT/client.log"

docker stop qwen4exp-test-sycl >/dev/null 2>&1
wait $SERVER_WAIT 2>/dev/null
echo "done"
