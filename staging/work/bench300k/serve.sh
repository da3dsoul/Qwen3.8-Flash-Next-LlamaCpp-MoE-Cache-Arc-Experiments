#!/bin/bash
# llama-server for the ~300K-token real-task long-context benchmark.
#
# Differences from bench120k/serve.sh, all forced by the depth:
#  -c 311296   : 305,750 prompt tokens + generation, and 311296/262144 = 1.1875
#                exactly, which is the YaRN factor below.
#  YaRN on     : 311,296 > n_ctx_train 262,144. The GGUF ships no rope-scaling
#                metadata, so without these flags llama.cpp extrapolates RoPE
#                unscaled and only warns -- silent quality loss, not a crash.
#  -ncmoe 32   : at -ub 2048 the compute buffer at this depth is 8,716 MiB
#                (measured, logs/decode-scaling/vram-probe-300k.log) against
#                3,692 at 122,880. That costs 8 -ncmoe steps. -ncmoe 31 lands
#                exactly on the measured 31,835 MiB envelope; 32 leaves
#                1,472 MiB spare.
# Everything else is the recommended production config verbatim.
set -euo pipefail
export LD_LIBRARY_PATH="/devbin:/opt/intel/oneapi/2025.3/lib:/opt/intel/oneapi/compiler/2025.3/lib"

exec /devbin/llama-server \
  -m /models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf \
  -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -ncmoe "${NCMOE:-32}" -ub "${UB:-2048}" \
  -c "${NCTX:-311296}" \
  --rope-scaling yarn --rope-scale "${SCALE:-1.1875}" --yarn-orig-ctx 262144 \
  -lm none -lzm off \
  -fit off \
  -np 1 \
  --host 0.0.0.0 --port 8080 \
  "$@"
