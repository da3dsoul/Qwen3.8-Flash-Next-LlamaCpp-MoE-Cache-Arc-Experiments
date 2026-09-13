#!/bin/bash
# Interleaved A/B of the shipped image binaries (baseline) against /devbin
# (hyper-connection fusions). One container, alternating runs, so both sides see
# the same shared-box contention over the same window.
#
# The prompt is a counting task on purpose: run-to-run token divergence changes
# which MoE experts get touched, and that moves throughput far more than the
# change under test. A count keeps the token stream nearly the same every run.
#
# ROUNDS controls how many base/dev pairs to run (default 3).
set -uo pipefail
M=/models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
OT='\.ffn_(gate|up|down)_exps\.weight=SYCL_MoE_Cached'
PROMPT="Count from one to two hundred."
NGEN="${NGEN:-300}"

run() { # $1 = label, $2 = base|dev
    echo "@@@ RUN $1"
    if [ "$2" = "base" ]; then
        echo "$PROMPT" | llama-cli -m "$M" -ngl 99 -fa 1 -ot "$OT" \
            -st -n "$NGEN" --temp 0 --seed 42 --no-reasoning-preserve
    else
        echo "$PROMPT" | LD_LIBRARY_PATH=/devbin:${LD_LIBRARY_PATH:-} \
            /devbin/llama-cli -m "$M" -ngl 99 -fa 1 -ot "$OT" \
            -st -n "$NGEN" --temp 0 --seed 42 --no-reasoning-preserve
    fi
    echo "@@@ END $1"
}

for i in $(seq 1 "${ROUNDS:-3}"); do
    run "baseline-$i" base
    run "fused-$i"    dev
done
