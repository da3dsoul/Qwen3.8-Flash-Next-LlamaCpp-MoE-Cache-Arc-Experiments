#!/bin/bash
# Real decode-throughput-vs-depth curve, post -lzm off. The question this
# answers is the SLOPE, not a single depth: how much does decode degrade per
# token of context on the recommended production config?
set -uo pipefail
export LD_LIBRARY_PATH="/devbin:/opt/intel/oneapi/2025.3/lib:/opt/intel/oneapi/compiler/2025.3/lib"
exec /devbin/llama-bench \
  -m /models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf \
  -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -ncmoe 24 -ub 2048 \
  -lm none -lzm off \
  -p 0 -n 32 -d 0,2048,8192,32768,65536,118016 -r 2 -o md
