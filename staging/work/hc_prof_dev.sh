#!/bin/bash
# Whole-graph op profile of the /devbin build, both profiler modes in one
# container so the 77 GB model is only pulled through the page cache once.
# Mode 1 = per-node GPU time (queue drained around every node, distorted).
# Mode 2 = host dispatch time vs tail wait (undistorted).
set -uo pipefail
M=/models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
OT='\.ffn_(gate|up|down)_exps\.weight=SYCL_MoE_Cached'
export LD_LIBRARY_PATH="/devbin:${LD_LIBRARY_PATH:-}"
export GGML_SYCL_OP_PROFILE_WINDOW="${WINDOW:-100}"

for mode in 1 2; do
    echo "@@@ OP_PROFILE MODE $mode"
    echo "Count from one to fifty." | GGML_SYCL_OP_PROFILE=$mode \
        /devbin/llama-cli -m "$M" -ngl 99 -fa 1 -ot "$OT" \
            -st -n 300 --temp 0 --seed 42 --no-reasoning-preserve 2>&1
    echo "@@@ END MODE $mode"
done
