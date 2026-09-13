#!/bin/bash
# End-to-end decode A/B, same instrument and config as docs/research/12 sect 8.
# Writes to /work so the result survives a dead client. d32768: the modelled
# effect there is ~1.5%, against llama-bench's own +/-0.46% at that depth.
set -uo pipefail
NEW=/devbin
M=/models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
export LD_LIBRARY_PATH="$NEW:/opt/intel/oneapi/2025.3/lib:/opt/intel/oneapi/compiler/2025.3/lib"
OUT=/work/qsa-blk-bench.log
: > "$OUT"
for arm in blk cell blk cell; do
  [ "$arm" = cell ] && export LLAMA_QSA_CELL_TOPK=1 || unset LLAMA_QSA_CELL_TOPK
  echo "### arm=$arm $(date +%H:%M:%S)" >> "$OUT"
  "$NEW/llama-bench" -m "$M" -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -ncmoe 24 -ub 2048 \
     -lzm off -p 0 -n 32 -d 32768 -r 3 2>/dev/null >> "$OUT"
  echo "### rc=$?" >> "$OUT"
done
