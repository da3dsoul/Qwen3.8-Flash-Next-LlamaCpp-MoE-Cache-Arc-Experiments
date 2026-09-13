#!/bin/bash
# Phase 2 of the -ub sweep:
#  (a) trailing -ub 512 re-control at -c 65536, to test whether the leading
#      512 arm's 97.72 tok/s (vs. the 12:46 A/B's 128.60) was drift.
#  (b) -ub 2048 and -ub 1024 at -c 163840 -- the REAL 116K benchmark context
#      size -- to find out whether the larger compute buffer still fits
#      alongside a 163840-token q4_0 KV cache at -ncmoe 24 (PLAN.md measured
#      only 2,836 MiB spare there at -ub 512).
# Prompt stays the 32K stand-in throughout, so each arm is ~80-330 s.
set -uo pipefail

M=/models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
P=/work/bench120k/prompt32k.txt
export LD_LIBRARY_PATH="/devbin:/opt/intel/oneapi/2025.3/lib:/opt/intel/oneapi/compiler/2025.3/lib"

run() {  # $1=ub $2=b $3=ctx $4=ncmoe
    echo "@@@ RUN ub=$1 b=$2 ctx=$3 ncmoe=$4"
    /devbin/llama-completion -m "$M" -f "$P" -no-cnv \
        -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -ncmoe "$4" -c "$3" \
        -b "$2" -ub "$1" \
        -n 8 --temp 0 --seed 42 -st --no-warmup -lv 4 2>&1 \
      | grep -viE '^\s*(#|def |class |[a-z_A-Z]+\.append|errors_list|import |from |return |if |for |while |try|except|print\(|pass$|else)' \
      | grep -vE '^\s{2,}'
    echo "@@@ END ub=$1 b=$2 ctx=$3 ncmoe=$4 rc=$?"
    echo "@@@ SEP"
}

run 512  2048 65536  24
run 2048 2048 163840 24
run 1024 2048 163840 24
