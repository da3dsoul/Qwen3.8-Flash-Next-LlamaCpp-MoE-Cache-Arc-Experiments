#!/bin/bash
# Same-config short-context control for the 600K arm: -ncmoe 38 -ub 1024 costs floor, not slope,
# and 12.42 tok/s at 614400 cannot be read against a -ncmoe 24 short-context number.
set -uo pipefail
B=/devbin
M=/models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
export LD_LIBRARY_PATH="$B:/opt/intel/oneapi/2025.3/lib:/opt/intel/oneapi/compiler/2025.3/lib"
O=${O:-/work/qsa-sparse-600k-ctl.log}
: > "$O"
echo "### ncmoe=38 ub=1024 shallow control $(date +%H:%M:%S)" >> "$O"
"$B/llama-bench" -m "$M" -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -ncmoe 38 -ub 1024 \
   -lzm off -lm none -p 0 -n 32 -d 8192,32768 -r 2 2>&1 | tail -12 >> "$O"
echo "### rc=$? $(date +%H:%M:%S)" >> "$O"
