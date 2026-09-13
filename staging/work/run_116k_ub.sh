#!/bin/bash
# Re-run the 116K real-task long-context benchmark at a chosen -ub, against
# the same prompt, same server config and same client as
# logs/long-context-120k-after/ (the 112.73 tok/s prefill / 17 min 11 s TTFT
# post-top-k-fix baseline at -ub 512).
#
# UB=<n>  ubatch size to test (default 2048)
# OUT dir is logs/long-context-120k-ub<UB>/
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UB="${UB:-2048}"
NCTX="${NCTX:-122880}"
NCMOE="${NCMOE:-24}"
OUT="$ROOT/logs/long-context-120k-ub${UB}"
MAXTOK="${MAXTOK:-2000}"
mkdir -p "$OUT"

cd "$ROOT"
docker compose -f docker/docker-compose.yml run --rm --service-ports \
    --entrypoint /usr/bin/env \
    -v "$ROOT/staging/devbin:/devbin" \
    -e "NCTX=$NCTX" \
    llm-test-sycl bash -c "/work/bench120k/serve.sh -ub $UB -ncmoe $NCMOE" > "$OUT/server.log" 2>&1 &
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
