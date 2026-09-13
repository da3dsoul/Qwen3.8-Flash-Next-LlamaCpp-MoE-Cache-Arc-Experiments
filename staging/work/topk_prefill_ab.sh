#!/bin/bash
# Before/after prefill A/B for the SYCL large-k TOP_K path (PR #28670 radix select
# vs. this project's earlier full-argsort fallback). Run inside the llm-test-sycl
# container with both /devbin (radix) and /devbin-before (argsort) bind-mounted.
#
# $1 = "before" | "after"
set -uo pipefail

WHICH="$1"
if [ "$WHICH" = "before" ]; then
    BIN=/devbin-before
else
    BIN=/devbin
fi

M=/models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
P=/work/bench120k/prompt32k.txt

export LD_LIBRARY_PATH="$BIN:/opt/intel/oneapi/2025.3/lib:/opt/intel/oneapi/compiler/2025.3/lib"
export GGML_SYCL_OP_PROFILE=2
export GGML_SYCL_OP_PROFILE_WINDOW=1

echo "@@@ RUN $WHICH ($BIN)"
"$BIN/llama-completion" -m "$M" -f "$P" -no-cnv \
    -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -ncmoe 24 -c 65536 -ub 512 \
    -n 8 --temp 0 --seed 42 -st --no-warmup
echo "@@@ END $WHICH"
