#!/bin/bash
# 1. perplexity on the same corpus/config as the block-top-k gate (qsa_blk_e2e.sh sect B),
#    so the numbers compare directly to PLAN.md's 2.6174 / 2.6028.
# 2. decode at the real 300K target. -ncmoe 32 is what docs/research/12's VRAM probe found
#    300K needs at -ub 2048; the pooled-key buffer takes 456 of the 1472 MiB it left spare.
set -uo pipefail
B=/devbin
M=/models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
export LD_LIBRARY_PATH="$B:/opt/intel/oneapi/2025.3/lib:/opt/intel/oneapi/compiler/2025.3/lib"

P=/work/qsa-pool-ppl32k.log
: > "$P"
for arm in warm pool plain pool plain; do
  case "$arm" in plain) export LLAMA_QSA_NO_POOL_CACHE=1 ;; *) unset LLAMA_QSA_NO_POOL_CACHE ;; esac
  echo -n "### arm=$arm $(date +%H:%M:%S) : " >> "$P"
  "$B/llama-perplexity" -m "$M" -f /work/bench120k/prompt32k.txt -ngl 99 -fa 1 \
     -ctk q4_0 -ctv q4_0 -ncmoe 24 -c 8192 -ub 2048 -lzm off --chunks 3 2>&1 \
     | grep -E 'Final estimate' | tail -1 >> "$P"
done
unset LLAMA_QSA_NO_POOL_CACHE

O=/work/qsa-pool-bench-300k.log
: > "$O"
for arm in pool plain; do
  case "$arm" in plain) export LLAMA_QSA_NO_POOL_CACHE=1 ;; *) unset LLAMA_QSA_NO_POOL_CACHE ;; esac
  echo "### arm=$arm $(date +%H:%M:%S)" >> "$O"
  "$B/llama-bench" -m "$M" -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -ncmoe 32 -ub 2048 \
     -lzm off -lm none -p 0 -n 32 -d 307200 -r 1 2>/dev/null >> "$O"
  echo "### rc=$? $(date +%H:%M:%S)" >> "$O"
done
