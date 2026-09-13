#!/bin/bash
# Does skipping FA collapse MoE expert routing? Compare cache hit rate.
set -uo pipefail
M=/models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
P="Write a short story about a robot who discovers music for the first time."
COMMON=(-m "$M" -ngl 99 -fa 1 -ot "\.ffn_(gate|up|down)_exps\.weight=SYCL_MoE_Cached" -st -n 200 --temp 0 --seed 42)
strip() { tr '\r' '\n' | grep -vE '^[\\|/-]+$'; }

echo "########## A: FA normal ##########"
GGML_SYCL_MOE_PROFILE=1 llama-cli "${COMMON[@]}" <<< "$P" 2>&1 | strip | grep -E "MOE_PROFILE ops|Generation:" | tail -6

echo "########## B: FA skipped ##########"
GGML_SYCL_MOE_PROFILE=1 GGML_SYCL_FA_SKIP=1 llama-cli "${COMMON[@]}" --ignore-eos <<< "$P" 2>&1 | strip | grep -E "MOE_PROFILE ops|Generation:" | tail -6
