#!/bin/bash
# IQ3_XXS Flash-Next under the whole-graph per-op profiler
# (GGML_SYCL_OP_PROFILE=1, set by the caller). Short prompt on purpose so the
# 400-split profiler windows land in steady-state decode rather than prefill.
# Timings here are NOT production timings: the profiler drains the queue around
# every node.
set -euo pipefail
echo "Count from one to fifty." | \
  llama-cli -m /models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf \
    -ngl 99 -fa 1 \
    -ot "\.ffn_(gate|up|down)_exps\.weight=SYCL_MoE_Cached" \
    -st -n 300 --temp 0 --seed 42 --no-reasoning-preserve 2>&1
