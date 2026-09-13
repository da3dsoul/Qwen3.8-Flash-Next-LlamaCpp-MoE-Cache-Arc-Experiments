#!/bin/bash
# GPU (-ngl 99) determinism check, leg 2. Same prompt/config as determinism_cpu.sh
# (prompt1500, -c 2048 -ub 512 -n 64, temp 0, seed 42), run twice, to compare against the
# CPU-only leg. Per the topk-radix.cpp atomic-tie-break hypothesis, expect drift here where
# the CPU-only leg was byte-identical.
set -uo pipefail
M=/models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
export LD_LIBRARY_PATH=/devbin:/opt/intel/oneapi/2025.3/lib:/opt/intel/oneapi/compiler/2025.3/lib

for i in 1 2; do
  echo "### run$i $(date +%H:%M:%S)"
  /devbin/llama-completion -m "$M" -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -lzm off -ncmoe 38 \
      -c 2048 -ub 512 -n 64 --temp 0 --seed 42 \
      -f /work/bench120k/prompt1500.txt \
      > "/work/determinism-gpu-run$i.txt" 2> "/work/determinism-gpu-run$i.err"
  echo "### run$i rc=$? $(date +%H:%M:%S)"
done
echo "### done $(date +%H:%M:%S)"
