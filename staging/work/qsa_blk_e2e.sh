#!/bin/bash
set -uo pipefail
NEW=/devbin
M=/models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
export LD_LIBRARY_PATH="$NEW:/opt/intel/oneapi/2025.3/lib:/opt/intel/oneapi/compiler/2025.3/lib"
C="-ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -ncmoe 24 --temp 0 --seed 42 -st --no-warmup"

echo "=== A. 32k prefill A/B, arms interleaved (warm cache, trailing control) ==="
for pass in 1 2; do
  for arm in cell blk; do
    [ "$arm" = cell ] && export LLAMA_QSA_CELL_TOPK=1 || unset LLAMA_QSA_CELL_TOPK
    "$NEW/llama-completion" -m "$M" -no-cnv $C -f /work/bench120k/prompt32k.txt \
       -c 65536 -ub 2048 -n 32 > /work/e-$arm-$pass.txt 2>/work/e-$arm-$pass.err
    echo -n "  pass$pass $arm exit=$? : "
    grep -oE 'prompt eval time =.*' /work/e-$arm-$pass.err | head -1
  done
done
echo "  blk pass1 vs pass2 output: $(cmp -s /work/e-blk-1.txt /work/e-blk-2.txt && echo SAME || echo DIFFER)"
echo "  cell pass1 vs pass2 output: $(cmp -s /work/e-cell-1.txt /work/e-cell-2.txt && echo SAME || echo DIFFER)"
echo "--- blk continuation (last 400 chars) ---"; tail -c 400 /work/e-blk-1.txt; echo

echo "=== B. perplexity at depth, selection active ==="
for arm in cell blk; do
  [ "$arm" = cell ] && export LLAMA_QSA_CELL_TOPK=1 || unset LLAMA_QSA_CELL_TOPK
  echo -n "  $arm : "
  "$NEW/llama-perplexity" -m "$M" -f /work/bench120k/prompt32k.txt \
     -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -ncmoe 24 -c 8192 -ub 2048 --chunks 3 2>&1 \
     | grep -E 'Final estimate' | tail -1
done
