#!/bin/bash
# Host-side profile of the per-ubatch QSA grouping and input fill (LLAMA_QSA_HOST_PROF=1).
# The profiler ends a window when the depth changes, so one process covers every -d in one load.
# NOTE: --no-warmup must NOT be used here -- without it the 32 measured tokens also pay the SYCL
# kernel JIT and the first arm came back at 0.90 tok/s instead of ~22.
set -uo pipefail
B=${B:-/devbin}
M=/models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
export LD_LIBRARY_PATH="$B:/opt/intel/oneapi/2025.3/lib:/opt/intel/oneapi/compiler/2025.3/lib"
export LLAMA_QSA_HOST_PROF=1

COMMON="-m $M -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -lzm off -lm none -p 0 -n 32 -r 1"

O=${O:-/work/qsa-hostprof.log}
: > "$O"

echo "### $* $(date +%H:%M:%S)" >> "$O"
"$B/llama-bench" $COMMON "$@" 2>&1 | grep -E 'qsa host prof|^\| ' >> "$O"
echo "### rc=$? $(date +%H:%M:%S)" >> "$O"
