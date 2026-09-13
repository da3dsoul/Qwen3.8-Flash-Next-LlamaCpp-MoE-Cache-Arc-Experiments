#!/bin/bash
# -ub (ubatch) sweep for prefill throughput on the 32K stand-in prompt.
# Same tree/binaries as staging/work/topk_prefill_ab.sh's "after" arm (radix
# select top_k, PR #28670), same config as the 116K benchmark except -c/-ub.
# Run inside the llm-test-sycl container with /devbin bind-mounted.
#
# $1 = ub value ; $2 = optional -b value (defaults to max(2048, ub))
set -uo pipefail

UB="$1"
B="${2:-}"
if [ -z "$B" ]; then
    if [ "$UB" -gt 2048 ]; then B="$UB"; else B=2048; fi
fi

M=/models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
P=/work/bench120k/prompt32k.txt

export LD_LIBRARY_PATH="/devbin:/opt/intel/oneapi/2025.3/lib:/opt/intel/oneapi/compiler/2025.3/lib"

echo "@@@ RUN ub=$UB b=$B"
"/devbin/llama-completion" -m "$M" -f "$P" -no-cnv \
    -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -ncmoe 24 -c 65536 \
    -b "$B" -ub "$UB" \
    -n 8 --temp 0 --seed 42 -st --no-warmup
echo "@@@ END ub=$UB b=$B"
