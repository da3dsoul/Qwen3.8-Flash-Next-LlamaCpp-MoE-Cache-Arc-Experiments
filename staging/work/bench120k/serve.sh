#!/bin/bash
# llama-server for the 116K-token real-task long-context benchmark.
# Config is docs/research/08-long-context-vram-budget.md's recommended
# long-context placement, verbatim: -ngl 99 -fa 1 -ctk/-ctv q4_0 -ncmoe 24.
# -fit off so llama.cpp does NOT silently reshape -c/-ncmoe to fit -- we want
# to know whether the recommended config actually holds, not whether the
# auto-fitter can rescue it.
set -euo pipefail
export LD_LIBRARY_PATH="/devbin:/opt/intel/oneapi/2025.3/lib:/opt/intel/oneapi/compiler/2025.3/lib"

exec /devbin/llama-server \
  -m /models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf \
  -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -ncmoe 24 \
  -c "${NCTX:-163840}" \
  -fit off \
  -np 1 \
  --host 0.0.0.0 --port 8080 \
  "$@"
