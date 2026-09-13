#!/bin/bash
# Single-variable test: the 116K prompt at the 300K config with YaRN REMOVED.
# Everything else identical to logs/long-context-300kcfg-116kprompt (which was
# garbage): -c 311296 -ncmoe 32 -ub 2048 -lm none -lzm off, same prompt.
#   coherent -> YaRN is the culprit, and it is harmful even BELOW n_ctx_train
#   garbage  -> the fault is -ncmoe 32 or -c 311296, not the rope config
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$ROOT/logs/long-context-300kcfg-noyarn"; mkdir -p "$OUT"; cd "$ROOT"
docker compose -f docker/docker-compose.yml run --rm --service-ports -d \
    --entrypoint /usr/bin/env -v "$ROOT/staging/devbin:/devbin" \
    llm-test-sycl bash -c 'export LD_LIBRARY_PATH=/devbin:/opt/intel/oneapi/2025.3/lib:/opt/intel/oneapi/compiler/2025.3/lib;
      exec /devbin/llama-server -m /models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf \
      -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -ncmoe 32 -ub 2048 -c 311296 \
      -lm none -lzm off -fit off -np 1 --host 0.0.0.0 --port 8080' > "$OUT/cid.txt" 2>&1
CID=$(tail -1 "$OUT/cid.txt")
for i in $(seq 1 1200); do curl -sf http://localhost:8090/health >/dev/null 2>&1 && break; sleep 1; done
python3 "$ROOT/staging/work/bench300k/run_bench.py" localhost:8090 \
    "$ROOT/staging/work/bench120k/full_prompt.txt" "$OUT/run1" 200 2>&1 | tee "$OUT/client.log"
docker logs "$CID" > "$OUT/server.log" 2>&1
docker stop "$CID" >/dev/null 2>&1
echo done
