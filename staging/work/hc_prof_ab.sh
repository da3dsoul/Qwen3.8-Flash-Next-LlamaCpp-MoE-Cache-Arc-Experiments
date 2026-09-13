#!/bin/bash
# Back-to-back op-profile of the image binaries (baseline) and /devbin (fused), in one
# container. End-to-end t/s on this box swings by +-30% with background load, but the
# profiler's per-op totals over a fixed 100-token window are a controlled comparison:
# same prompt, same op call counts, minutes apart instead of hours.
#
# MODE picks the profiler mode (1 = per-node GPU time, 2 = host dispatch vs tail wait).
set -uo pipefail
M=/models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
OT='\.ffn_(gate|up|down)_exps\.weight=SYCL_MoE_Cached'
export GGML_SYCL_OP_PROFILE="${MODE:-1}"
export GGML_SYCL_OP_PROFILE_WINDOW="${WINDOW:-100}"

for side in base dev base dev; do
    echo "@@@ PROF $side"
    if [ "$side" = base ]; then
        echo "Count from one to fifty." | \
            llama-cli -m "$M" -ngl 99 -fa 1 -ot "$OT" \
                -st -n 300 --temp 0 --seed 42 --no-reasoning-preserve 2>&1
    else
        echo "Count from one to fifty." | LD_LIBRARY_PATH=/devbin:${LD_LIBRARY_PATH:-} \
            /devbin/llama-cli -m "$M" -ngl 99 -fa 1 -ot "$OT" \
                -st -n 300 --temp 0 --seed 42 --no-reasoning-preserve 2>&1
    fi
    echo "@@@ END $side"
done
