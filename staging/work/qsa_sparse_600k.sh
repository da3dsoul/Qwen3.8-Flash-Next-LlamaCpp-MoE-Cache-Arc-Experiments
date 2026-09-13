#!/bin/bash
# 600K decode with sparse FA. -ub 1024 because -ub 2048's compute buffer is ~16 GiB at this depth
# (docs/research/12 sect 6's 0.0267 MiB/token), and -ncmoe 38 for the model side. -ub does not
# change decode throughput (the n_tokens=1 graph), only the compute buffer and prefill.
set -uo pipefail
B=/devbin
M=/models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
export LD_LIBRARY_PATH="$B:/opt/intel/oneapi/2025.3/lib:/opt/intel/oneapi/compiler/2025.3/lib"
O=${O:-/work/qsa-sparse-600k.log}
: > "$O"
echo "### sparse ncmoe=38 ub=1024 d=614400 $(date +%H:%M:%S)" >> "$O"
"$B/llama-bench" -m "$M" -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -ncmoe 38 -ub 1024 \
   -lzm off -lm none -p 0 -n 32 -d 614400 -r 1 2>&1 | tail -20 >> "$O"
echo "### rc=$? $(date +%H:%M:%S)" >> "$O"
