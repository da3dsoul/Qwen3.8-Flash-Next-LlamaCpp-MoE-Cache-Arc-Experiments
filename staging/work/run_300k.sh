#!/bin/bash
# The real ~300K-token long-context benchmark: same prompt generator, same
# client, same server-side timers as logs/long-context-120k-*, at 305,750
# prompt tokens with YaRN active. Preserves the 116K artifacts untouched.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.."
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$ROOT/logs/long-context-300k"
MAXTOK="${MAXTOK:-1000}"
mkdir -p "$OUT"
cd "$ROOT"

docker compose -f docker/docker-compose.yml run --rm --service-ports \
    --entrypoint /usr/bin/env \
    -v "$ROOT/staging/devbin:/devbin" \
    -e "NCTX=${NCTX:-311296}" -e "NCMOE=${NCMOE:-32}" -e "UB=${UB:-2048}" \
    llm-test-sycl bash -c "/work/bench300k/serve.sh" > "$OUT/server.log" 2>&1 &
SERVER_WAIT=$!

echo "waiting for server health..."
for i in $(seq 1 1200); do
    curl -sf http://localhost:8090/health >/dev/null 2>&1 && { echo "server up after ${i}s"; break; }
    sleep 1
done
curl -sf http://localhost:8090/health || echo "server never came up"

python3 "$ROOT/staging/work/bench300k/run_bench.py" \
    localhost:8090 \
    "$ROOT/staging/work/bench300k/full_prompt.txt" \
    "$OUT/run1" \
    "$MAXTOK" 2>&1 | tee "$OUT/client.log"

docker stop qwen4exp-test-sycl >/dev/null 2>&1
wait $SERVER_WAIT 2>/dev/null
echo "done"
