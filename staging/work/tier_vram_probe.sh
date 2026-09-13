#!/bin/bash
# VRAM probe for a --context-tiers entry: load the model at the tier and read the
# allocator's own buffer sizes under -lv 4, then stop. One tier per run.
#
# -lzm auto / -lm auto: PLAN.md (2026-09-11) established that -lzm does not change
# VRAM, and the load is much faster, which is what makes this probe cheap.
# -fit off so nothing silently reshapes the tier we are trying to check.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${OUT:-$ROOT/logs/context-tiers}"
DEVBIN="${DEVBIN:-$ROOT/staging/devbin-tiers}"
TIER="${TIER:-262144:2048:30}"
TAG="${TAG:-$(echo "$TIER" | tr ':' '-')}"
mkdir -p "$OUT"
cd "$ROOT"

CID=$(docker compose -f docker/docker-compose.yml run -d \
    --entrypoint /usr/bin/env \
    -v "$DEVBIN:/devbin" \
    llm-test-sycl bash -c "
      export LD_LIBRARY_PATH=/devbin:/opt/intel/oneapi/2025.3/lib:/opt/intel/oneapi/compiler/2025.3/lib
      /devbin/llama-server \
        -m /models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf \
        -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 \
        --context-tiers '$TIER' \
        -fit off -np 1 -lv 4 \
        --host 0.0.0.0 --port 8080")
echo "probe container: $CID (tier $TIER)"

for i in $(seq 1 900); do
    docker logs "$CID" 2>&1 | grep -q "model loaded\|error\|failed\|OUT_OF" && break
    sleep 2
done
sleep 3

docker logs "$CID" > "$OUT/vram-probe-$TAG.log" 2>&1
docker stop "$CID" >/dev/null 2>&1

grep -E "buffer size|model loaded|OUT_OF|error|failed|n_ctx_slot|context tier" "$OUT/vram-probe-$TAG.log" | tail -30
