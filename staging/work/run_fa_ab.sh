#!/bin/bash
# Four runs in one container so the model page cache stays warm after the first.
#  RUN 0: short verbose run, only to log which FA kernel decode dispatches to.
#  RUN 1: clean -n 200 baseline t/s.
#  RUN 2: same, but with the FA kernel skipped entirely. Output is garbage on
#         purpose; the t/s difference vs RUN 1 is flash attention's real share
#         of a token.
#  RUN 3: whole-graph per-op profile, for the relative split only.
set -uo pipefail
M=/models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
P="Write a short story about a robot who discovers music for the first time."
COMMON=(-m "$M" -ngl 99 -fa 1 -ot "\.ffn_(gate|up|down)_exps\.weight=SYCL_MoE_Cached" -st --temp 0 --seed 42)
strip() { tr '\r' '\n' | grep -vE '^[\\|/-]+$'; }

echo "########## RUN 0: dispatch probe (-n 4, verbose) ##########"
GGML_SYCL_MKL_FA_DEBUG=1 llama-cli "${COMMON[@]}" -n 4 -v <<< "$P" 2>&1 | strip \
  | grep -E "FA-DISP|n_ctx |n_ctx_per_seq|Generation:" | head -30

echo "########## RUN 1: baseline -n 200 ##########"
llama-cli "${COMMON[@]}" -n 200 <<< "$P" 2>&1 | strip | tail -14

echo "########## RUN 2: GGML_SYCL_FA_SKIP=1 -n 200 (timing only) ##########"
GGML_SYCL_FA_SKIP=1 llama-cli "${COMMON[@]}" -n 200 --ignore-eos <<< "$P" 2>&1 | strip | tail -5

echo "########## RUN 3: GGML_SYCL_OP_PROFILE=1 -n 60 (relative split only) ##########"
GGML_SYCL_OP_PROFILE=1 llama-cli "${COMMON[@]}" -n 60 <<< "$P" 2>&1 | strip \
  | grep -E "OP_PROFILE|ms  \(|Generation:" | tail -60
