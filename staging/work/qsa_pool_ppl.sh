#!/bin/bash
# Perplexity A/B for the pooled-key cache, same binary, same corpus.
# Matches the block-top-k gate in qsa_blk_e2e.sh (sect B) so the numbers compare
# to PLAN.md's 2.6174 / 2.6028, but over more chunks for a real error bar.
set -uo pipefail
B=/devbin
M=/models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
export LD_LIBRARY_PATH="$B:/opt/intel/oneapi/2025.3/lib:/opt/intel/oneapi/compiler/2025.3/lib"
OUT=${OUT:-/work/qsa-pool-ppl.log}
CORPUS=${CORPUS:-/work/bench120k/full_prompt.txt}
CHUNKS=${CHUNKS:-10}
: > "$OUT"
for arm in pool plain pool plain; do
  case "$arm" in
    plain) export LLAMA_QSA_NO_POOL_CACHE=1 ;;
    *)     unset LLAMA_QSA_NO_POOL_CACHE ;;
  esac
  echo "### arm=$arm $(date +%H:%M:%S)" >> "$OUT"
  "$B/llama-perplexity" -m "$M" -f "$CORPUS" -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 \
     -ncmoe 24 -c 8192 -ub 2048 -lzm off --chunks "$CHUNKS" 2>&1 \
     | grep -E "^\[[0-9]+\]|Final estimate" >> "$OUT"
  echo "### rc=$?" >> "$OUT"
done
