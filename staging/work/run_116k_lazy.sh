#!/bin/bash
# 116K real-task benchmark with arbitrary extra server flags, for the
# `-lm none/mlock` null-result investigation (docs/research/11 correction).
#
# Why this exists: `-lm none` and `-lm mlock` both failed to move decode, and
# the reason is that NEITHER of them covers the tensor that is actually being
# demand-paged. `qwen4exp` marks `per_layer_token_embd` TENSOR_READ_LAZY
# (src/models/qwen4exp.cpp:193); it is 27,465 MiB in this GGUF, and
# `llama_model_loader::load_all_data` keeps an mmap for lazy tensors
# regardless of --load-mode (src/llama-model-loader.cpp:1404-1405) while mlock
# deliberately skips them ("locking a lazy tensor would fault all of it in,
# which is what lazy avoids", :1653-1654). The flag that covers it is
# `-lzm off` / `--lazy-mode off`.
#
# EXTRA="..."  extra args appended to serve.sh
# OUTNAME=...  logs/<OUTNAME>/
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UB="${UB:-2048}"
NCTX="${NCTX:-122880}"
NCMOE="${NCMOE:-24}"
EXTRA="${EXTRA:-}"
OUT="$ROOT/logs/${OUTNAME:-long-context-120k-lazyoff}"
MAXTOK="${MAXTOK:-400}"
PROMPT="${PROMPT:-$ROOT/staging/work/bench120k/full_prompt.txt}"
mkdir -p "$OUT"

cd "$ROOT"
docker compose -f docker/docker-compose.yml run --rm --service-ports \
    --entrypoint /usr/bin/env \
    -v "$ROOT/staging/devbin:/devbin" \
    -e "NCTX=$NCTX" \
    llm-test-sycl bash -c "/work/bench120k/serve.sh -ub $UB -ncmoe $NCMOE $EXTRA" > "$OUT/server.log" 2>&1 &
SERVER_WAIT=$!

echo "waiting for server health..."
for i in $(seq 1 1800); do
    if curl -sf http://localhost:8090/health >/dev/null 2>&1; then
        echo "server up after ${i}s"
        break
    fi
    sleep 1
done

curl -sf http://localhost:8090/health || { echo "server never came up"; }

python3 "$ROOT/staging/work/bench120k/run_bench.py" \
    localhost:8090 \
    "$PROMPT" \
    "$OUT/run1" \
    "$MAXTOK" 2>&1 | tee "$OUT/client.log"

docker stop qwen4exp-test-sycl >/dev/null 2>&1
wait $SERVER_WAIT 2>/dev/null
echo "done"
