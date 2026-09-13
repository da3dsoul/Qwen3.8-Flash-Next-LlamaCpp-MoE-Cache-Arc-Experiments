#!/bin/bash
# Driver for the --context-tiers validation. Starts the server from
# tier_switch_serve.sh, runs the multi-turn conversation in
# tier_switch_client.py, and keeps the server log for the switch trace.
#
# Note: `docker compose run` ignores container_name (PLAN.md 2026-09-11), so
# the container id from `run -d` is captured and stopped explicitly.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${OUT:-$ROOT/logs/context-tiers}"
DEVBIN="${DEVBIN:-$ROOT/staging/devbin-tiers}"
mkdir -p "$OUT"
cd "$ROOT"

# a slice of a real benchmark prompt, sized to cross the tier boundary under test
# (3.52 chars/token for this generator, measured in PLAN.md)
FILLER_CHARS="${FILLER_CHARS:-18000}"
FILLER_SRC="${FILLER_SRC:-$ROOT/staging/work/bench120k/prompt32k.txt}"
FILLER="$ROOT/staging/work/tier_filler-$FILLER_CHARS.txt"
if [ ! -f "$FILLER" ] || [ "$(stat -c %s "$FILLER")" -lt "$FILLER_CHARS" ]; then
    head -c "$FILLER_CHARS" "$FILLER_SRC" > "$FILLER"
fi

CID=$(docker compose -f docker/docker-compose.yml run -d --service-ports \
    --entrypoint /usr/bin/env \
    -v "$DEVBIN:/devbin" \
    -e "TIERS=${TIERS:-4096:512:24,32768:2048:26}" \
    -e "RESERVE=${RESERVE:-256}" \
    -e "DOWNTURNS=${DOWNTURNS:-2}" \
    -e "LZM=${LZM--lzm off}" -e "LM=${LM--lm none}" \
    -e "THREADS=${THREADS:-6}" \
    -e "EXTRA=${EXTRA-}" \
    llm-test-sycl bash -c "/work/${SERVE:-tier_switch_serve.sh}")
echo "server container: $CID"

trap 'docker stop "$CID" >/dev/null 2>&1' EXIT

echo "waiting for server health..."
for i in $(seq 1 1800); do
    curl -sf http://localhost:8090/health >/dev/null 2>&1 && { echo "server up after ${i}s"; break; }
    sleep 1
done
curl -sf http://localhost:8090/health || { echo "server never came up"; docker logs "$CID" > "$OUT/server.log" 2>&1; exit 1; }

# a reload must give the host tensors back, so watch box-wide memory across it
( while true; do
    echo "$(date +%H:%M:%S) $(free -m | awk '/^Mem:/{print "used_mib="$3" available_mib="$7}')"
    sleep 10
  done ) > "$OUT/host-mem.log" 2>&1 &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null; docker stop "$CID" >/dev/null 2>&1' EXIT

PROFILE="${PROFILE:-}" python3 "$ROOT/staging/work/tier_switch_client.py" \
    localhost:8090 "$OUT/run" "$FILLER" 2>&1 | tee "$OUT/client.log"

kill $SAMPLER 2>/dev/null

docker logs "$CID" > "$OUT/server.log" 2>&1
echo "--- tier switch trace ---"
grep -n "context tier\|load_model: initializing\|model loaded\|n_ctx_slot" "$OUT/server.log" | tail -40
