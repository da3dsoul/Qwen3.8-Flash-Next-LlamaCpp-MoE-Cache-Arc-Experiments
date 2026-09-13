#!/bin/bash
# 116K real-task run with QSA sparse flash attention on, plus the backend contract check
# (GGML_SYCL_FA_SPARSE_CHECK=1 aborts if any mask row carries more finite entries than the
# n_kv_max the model promised). Same prompt/config/timers as logs/long-context-120k-lazyoff/,
# so the numbers and the saved reasoning trace compare directly.
#
# `docker compose run` ignores container_name, so capture the id and stop that -- the
# `docker stop qwen4exp-test-sycl` the older drivers use has been silently failing.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UB="${UB:-2048}"
NCTX="${NCTX:-122880}"
NCMOE="${NCMOE:-24}"
EXTRA="${EXTRA:--lm none -lzm off}"
BIN="${BIN:-$ROOT/staging/devbin-sparse}"
OUT="$ROOT/logs/${OUTNAME:-long-context-120k-sparse}"
MAXTOK="${MAXTOK:-400}"
PROMPT="${PROMPT:-$ROOT/staging/work/bench120k/full_prompt.txt}"
mkdir -p "$OUT"

cd "$ROOT"
CID=$(docker compose -f docker/docker-compose.yml run -d --service-ports \
    --entrypoint /usr/bin/env \
    -v "$BIN:/devbin" \
    -e "NCTX=$NCTX" \
    -e "GGML_SYCL_FA_SPARSE_CHECK=${SPARSE_CHECK:-1}" \
    ${SPARSE_OFF:+-e LLAMA_QSA_NO_SPARSE_FA=1} \
    llm-test-sycl bash -c "/work/bench120k/serve.sh -ub $UB -ncmoe $NCMOE $EXTRA")

echo "server container $CID"

for i in $(seq 1 1800); do
    if curl -sf http://localhost:8090/health >/dev/null 2>&1; then
        echo "server up after ${i}s"
        break
    fi
    sleep 1
done

curl -sf http://localhost:8090/health || echo "server never came up"

python3 "$ROOT/staging/work/bench120k/run_bench.py" \
    localhost:8090 \
    "$PROMPT" \
    "$OUT/run1" \
    "$MAXTOK" 2>&1 | tee "$OUT/client.log"

docker logs "$CID" > "$OUT/server.log" 2>&1
docker rm -f "$CID" >/dev/null 2>&1
echo "done"
