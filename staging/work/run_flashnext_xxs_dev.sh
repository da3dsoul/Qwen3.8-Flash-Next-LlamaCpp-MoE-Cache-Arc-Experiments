#!/bin/bash
# Flash-Next IQ3_XXS run against the incrementally built binaries in
# staging/devbin (mounted at /devbin), not the ones baked into the image.
# Extra llama-cli args can be passed on the command line.
set -euo pipefail
export LD_LIBRARY_PATH="/devbin:${LD_LIBRARY_PATH:-}"
echo "${PROMPT:-Write a short story about a robot who discovers music for the first time.}" | \
  /devbin/llama-cli -m /models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf \
    -ngl 99 -fa 1 \
    -ot "\.ffn_(gate|up|down)_exps\.weight=SYCL_MoE_Cached" \
    -st -n "${NGEN:-200}" --temp 0 --seed 42 "$@"
