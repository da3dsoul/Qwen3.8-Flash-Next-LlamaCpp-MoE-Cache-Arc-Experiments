#!/bin/bash
# End-to-end decode depth curve, pooled-key cache on vs off, same binary.
# Same instrument and config as docs/research/12 sect 8. Writes to /work so the
# result survives a dead client.
set -uo pipefail
B=/devbin
M=/models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
export LD_LIBRARY_PATH="$B:/opt/intel/oneapi/2025.3/lib:/opt/intel/oneapi/compiler/2025.3/lib"
OUT=${OUT:-/work/qsa-pool-bench.log}
DEPTHS=${DEPTHS:-0,8192,32768,65536,118016}
NCMOE=${NCMOE:-24}
CTX=${CTX:-}
: > "$OUT"
for arm in warm pool plain pool plain; do
  case "$arm" in
    plain) export LLAMA_QSA_NO_POOL_CACHE=1 ;;
    *)     unset LLAMA_QSA_NO_POOL_CACHE ;;
  esac
  echo "### arm=$arm $(date +%H:%M:%S)" >> "$OUT"
  "$B/llama-bench" -m "$M" -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -ncmoe "$NCMOE" -ub 2048 \
     -lzm off -lm none -p 0 -n 32 -d "$DEPTHS" -r 2 ${CTX:+-c $CTX} 2>/dev/null >> "$OUT"
  echo "### rc=$?" >> "$OUT"
done
