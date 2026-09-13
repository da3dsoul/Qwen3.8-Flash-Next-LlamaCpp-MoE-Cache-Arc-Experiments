#!/bin/bash
# Before/after A/B for the block-granularity QSA top-k (build_qsa_top_k).
# Same binary for both arms: LLAMA_QSA_CELL_TOPK=1 forces the old per-cell path.
#
# $1 = "before" | "after"
set -uo pipefail

WHICH="$1"
if [ "$WHICH" = "before" ]; then
    export LLAMA_QSA_CELL_TOPK=1
fi

BIN=/devbin
M=/models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
P=/work/bench120k/prompt32k.txt

export LD_LIBRARY_PATH="$BIN:/opt/intel/oneapi/2025.3/lib:/opt/intel/oneapi/compiler/2025.3/lib"

echo "@@@ RUN $WHICH (LLAMA_QSA_CELL_TOPK=${LLAMA_QSA_CELL_TOPK:-unset})"
"$BIN/llama-completion" -m "$M" -f "$P" -no-cnv \
    -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -ncmoe 24 -c 65536 -ub 2048 \
    -lm none -lzm off \
    -n 48 --temp 0 --seed 42 -st --no-warmup
echo "@@@ END $WHICH"
