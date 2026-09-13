#!/bin/bash
# Interleaved A/B of the shipped image binaries (baseline) against the
# incrementally built ones in /devbin (SYCL large-k TOP_K fix). One container,
# alternating runs, so both sides see the same shared-box contention over the
# same window. llama-bench cannot be used here: it rejects the custom
# SYCL_MoE_Cached buffer type that -ot needs.
set -uo pipefail
M=/models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
OT='\.ffn_(gate|up|down)_exps\.weight=SYCL_MoE_Cached'
PROMPT="Write a short story about a robot who discovers music for the first time."

run() { # $1 = label, $2 = binary
    echo "@@@ RUN $1"
    if [ "$2" = "base" ]; then
        echo "$PROMPT" | llama-cli -m "$M" -ngl 99 -fa 1 -ot "$OT" -st -n 200 --temp 0 --seed 42
    else
        echo "$PROMPT" | LD_LIBRARY_PATH=/devbin:${LD_LIBRARY_PATH:-} \
            /devbin/llama-cli -m "$M" -ngl 99 -fa 1 -ot "$OT" -st -n 200 --temp 0 --seed 42
    fi
    echo "@@@ END $1"
}

for i in 1 2; do
    run "baseline-$i" base
    run "fixed-$i" dev
done
