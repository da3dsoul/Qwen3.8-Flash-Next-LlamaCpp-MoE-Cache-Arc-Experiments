#!/bin/bash
# Direct reproduction attempt of the run this repo's YaRN verdict rests on.
#
# logs/long-context-300kcfg-116kprompt/ = llama-server, --rope-scaling yarn
# --rope-scale 1.1875 --yarn-orig-ctx 262144, -c 311296 -ncmoe 32 -ub 2048
# -lm none -lzm off, the 116,277-token prompt -> "b" then 199 "/".
# logs/long-context-300k/ = same server, the 305,801-token prompt -> 999 "/".
#
# Today's llama-completion arms at 305,759 tokens with the SAME rope flags are
# fully coherent (staging/work/yarn-gen-yarn-305k.txt), so either the collapse
# was specific to llama-server, or it did not survive the day's changes to
# build_qsa_top_k -- the function that holds the indexer's own RoPE application
# and that gained block-granularity top-k at 21:45 and a pooled-key cache at
# 23:13, both AFTER the 21:19 degenerate run whose binary was staging/devbin as
# it stood then. This arm holds the tool and the config fixed and changes only
# the binary (staging/devbin is now the 01:59 build).
#
# Two sub-arms, because the server's n_ctx_slot differs between them and that is
# the one server-side thing YaRN changes:
#   yarn      -> n_ctx_train rewritten to 311,296, n_ctx_slot = 311296
#   nearunity -> same slot lift, rope numerically unscaled (control)
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# kernel-level gate first: test-backend-ops sweeps ext_factor = {0, 0.7465} x
# freq_scale = {1, 1.4245} x attn_factor over GGML_ROPE_TYPE_IMROPE (what
# qwen4exp uses) at partial n_dims, against the CPU reference. A pass means the
# SYCL rope kernel is not where any YaRN divergence lives.
echo "### ROPE op gate $(date +%H:%M:%S)"
docker compose -f docker/docker-compose.yml run --rm --entrypoint /usr/bin/env \
    -v "$ROOT/staging/devbin:/devbin" llm-test-sycl bash -c \
    'LD_LIBRARY_PATH=/devbin:/opt/intel/oneapi/2025.3/lib:/opt/intel/oneapi/compiler/2025.3/lib /devbin/test-backend-ops test -b SYCL0 -o ROPE' 2>&1 | tail -6

for arm in yarn; do
    OUT="$ROOT/logs/yarn/server-repro-$arm"; mkdir -p "$OUT"
    case "$arm" in
        yarn)      R="--rope-scaling yarn --rope-scale 1.1875 --yarn-orig-ctx 262144" ;;
        nearunity) R="--rope-scaling yarn --rope-scale 1.0001 --yarn-orig-ctx 311296" ;;
    esac
    echo "### arm=$arm $(date +%H:%M:%S)  rope='$R'"
    docker compose -f docker/docker-compose.yml run --rm --service-ports -d \
        --entrypoint /usr/bin/env -v "$ROOT/staging/devbin:/devbin" \
        llm-test-sycl bash -c "export LD_LIBRARY_PATH=/devbin:/opt/intel/oneapi/2025.3/lib:/opt/intel/oneapi/compiler/2025.3/lib;
          exec /devbin/llama-server -m /models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf \
          -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -ncmoe 32 -ub 2048 -c 311296 \
          -lzm off -fit off -np 1 $R --host 0.0.0.0 --port 8080" > "$OUT/cid.txt" 2>&1
    CID=$(tail -1 "$OUT/cid.txt")
    for i in $(seq 1 1500); do curl -sf http://localhost:8090/health >/dev/null 2>&1 && break; sleep 1; done
    python3 "$ROOT/staging/work/bench300k/run_bench.py" localhost:8090 \
        "$ROOT/staging/work/bench120k/full_prompt.txt" "$OUT/run1" 200 2>&1 | tee "$OUT/client.log"
    docker logs "$CID" > "$OUT/server.log" 2>&1
    docker stop "$CID" >/dev/null 2>&1
    # capture the id from `run -d` and stop THAT, per the 300K report's section 4.1.3
    echo "### arm=$arm stopped, waiting for memory release $(date +%H:%M:%S)"
    for i in $(seq 1 60); do
        docker ps --format '{{.ID}}' | grep -q "${CID:0:12}" || break
        sleep 5
    done
    sleep 20
    free -h | sed -n 2p
    echo "--- reasoning, first 700 chars ---"; head -c 700 "$OUT/run1.reasoning.txt"; echo
done
echo "### done $(date +%H:%M:%S)"
