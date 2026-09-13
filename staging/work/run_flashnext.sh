#!/bin/bash
# -ncmoe (static per-layer CPU/GPU split, stock upstream flag) measured
# 19.7-25.3 t/s on this box's B70 vs. the custom cache's tuned 23.1 t/s
# (256 slots/tensor) -- see PLAN.md's 2026-09-11 update. Adopted as the
# interim baseline: simpler, zero maintenance, matches or beats the cache.
# The cache path is still real/correct/validated, just not the default here
# -- swap the last three lines for:
#   -ot "\.ffn_(gate|up|down)_exps\.weight=SYCL_MoE_Cached"
# to test it instead (tune via GGML_SYCL_MOE_CACHE_SIZE, default 64, 256 is
# the measured sweet spot before OOM on this box's 30GB B70).
set -euo pipefail
echo "Write a short story about a robot who discovers music for the first time." | \
  llama-cli -m /models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf \
    -ngl 99 -fa 1 -ncmoe 20 \
    -st -n 200 --temp 0 --seed 42
