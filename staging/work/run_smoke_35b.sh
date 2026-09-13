#!/bin/bash
set -euo pipefail
echo "Write a short story about a robot who discovers music for the first time." | \
  llama-cli -m /models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
    -ngl 99 -fa 1 \
    -ot "\.ffn_(gate|up|down)_exps\.weight=SYCL_MoE_Cached" \
    -st -n 200 --temp 0 --seed 42
