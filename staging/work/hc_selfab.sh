#!/bin/bash
# One binary against itself: GGML_SYCL_HC_FUSION=0 disables only the two
# hyper-connection matchers, so nothing else differs between the two sides -
# not the build, not the other fusions, not the model load.
#
# ABBA ordering (off, on, on, off) cancels any drift over the session.
# MODE=2 also prints the host-dispatch / GPU-wait split per side.
set -uo pipefail
M=/models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
OT='\.ffn_(gate|up|down)_exps\.weight=SYCL_MoE_Cached'
export LD_LIBRARY_PATH="/devbin:${LD_LIBRARY_PATH:-}"
export GGML_SYCL_OP_PROFILE="${MODE:-0}"
export GGML_SYCL_OP_PROFILE_WINDOW="${WINDOW:-100}"

run() {
    echo "@@@ RUN hc_fusion=$1"
    echo "Count from one to fifty." | GGML_SYCL_HC_FUSION="$1" \
        /devbin/llama-cli -m "$M" -ngl 99 -fa 1 -ot "$OT" \
            -st -n 300 --temp 0 --seed 42 --no-reasoning-preserve 2>&1
    echo "@@@ END hc_fusion=$1"
}

for i in $(seq 1 "${ROUNDS:-2}"); do
    run 0; run 1; run 1; run 0
done
