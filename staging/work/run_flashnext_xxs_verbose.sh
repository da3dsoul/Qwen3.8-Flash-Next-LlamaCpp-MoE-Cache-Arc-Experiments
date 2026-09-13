#!/bin/bash
# IQ3_XXS Flash-Next with INFO-level logging (-lv 4) so llama_context's
# resolve_fused_ops probe lines are visible. Short generation: this run is for
# the log, not for timing. NOTE: llama.cpp core LLAMA_LOG_INFO maps to
# verbosity 4 (see common_log_get_verbosity), so -lv 3 is not enough.
set -euo pipefail
echo "Write a short story about a robot who discovers music for the first time." | \
  llama-cli -m /models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf \
    -ngl 99 -fa 1 \
    -ot "\.ffn_(gate|up|down)_exps\.weight=SYCL_MoE_Cached" \
    -st -n 200 --temp 0 --seed 42 -lv 4
