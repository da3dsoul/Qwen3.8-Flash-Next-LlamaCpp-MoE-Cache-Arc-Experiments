#!/bin/bash
# Decode A/B for QSA sparse flash attention, same binary, arms interleaved behind a discarded
# warm-up arm (the leading-arm page-cache effect this project has hit repeatedly).
# Section A: the depths the pooled-key cache was measured at, same config, so the rows compare
# directly to docs/research/12 section 10.3. Section B: the real 300K target, -ncmoe 32 per the
# VRAM probe in docs/research/12 section 6.
set -uo pipefail
B=/devbin
M=/models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
export LD_LIBRARY_PATH="$B:/opt/intel/oneapi/2025.3/lib:/opt/intel/oneapi/compiler/2025.3/lib"

COMMON="-m $M -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -ub 2048 -lzm off -lm none -p 0 -n 32"

O=${O:-/work/qsa-sparse-bench.log}
: > "$O"

echo "### warm-up (discarded) $(date +%H:%M:%S)" >> "$O"
"$B/llama-bench" $COMMON -ncmoe 24 -d 2048 -r 1 2>/dev/null >> "$O"

for arm in dense sparse dense; do
  case "$arm" in
    dense) export LLAMA_QSA_NO_SPARSE_FA=1 ;;
    *)     unset LLAMA_QSA_NO_SPARSE_FA ;;
  esac
  echo "### A arm=$arm ncmoe=24 $(date +%H:%M:%S)" >> "$O"
  "$B/llama-bench" $COMMON -ncmoe 24 -d 8192,32768,65536,118016 -r 2 2>/dev/null >> "$O"
  echo "### rc=$? $(date +%H:%M:%S)" >> "$O"
done

for arm in dense sparse; do
  case "$arm" in
    dense) export LLAMA_QSA_NO_SPARSE_FA=1 ;;
    *)     unset LLAMA_QSA_NO_SPARSE_FA ;;
  esac
  echo "### B arm=$arm ncmoe=32 d=307200 $(date +%H:%M:%S)" >> "$O"
  "$B/llama-bench" $COMMON -ncmoe 32 -d 307200 -r 1 2>/dev/null >> "$O"
  echo "### rc=$? $(date +%H:%M:%S)" >> "$O"
done
unset LLAMA_QSA_NO_SPARSE_FA
echo "### done $(date +%H:%M:%S)" >> "$O"
