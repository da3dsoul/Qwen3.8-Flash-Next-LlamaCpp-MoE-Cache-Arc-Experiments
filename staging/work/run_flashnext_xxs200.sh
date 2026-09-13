#!/bin/bash
set -euo pipefail
echo "Write a short story about a robot who discovers music for the first time." | \
  llama-cli -m /models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf \
    -ngl 99 -fa 1 \
    -ot "\.ffn_(gate|up|down)_exps\.weight=SYCL_MoE_Cached" \
    -st -n 200 --temp 0 --seed 42
