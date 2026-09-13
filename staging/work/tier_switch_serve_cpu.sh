#!/bin/bash
# Same --context-tiers validation as tier_switch_serve.sh, but on the 21 GiB
# Qwen3.6-35B-A3B MoE (also an -ncmoe model) with no GPU offload and few
# threads, so the tier policy and the reload path can be exercised while the
# Arc card is busy with another workstream. Cost/VRAM numbers must come from
# the real model; this arm only proves the mechanism.
set -euo pipefail
export LD_LIBRARY_PATH="/devbin:/opt/intel/oneapi/2025.3/lib:/opt/intel/oneapi/compiler/2025.3/lib"

exec /devbin/llama-server \
  -m /models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  -ngl 0 --device none \
  --context-tiers "${TIERS:-4096:512:4,32768:2048:8}" \
  --context-tier-reserve "${RESERVE:-256}" \
  --context-tier-down-turns "${DOWNTURNS:-2}" \
  -t "${THREADS:-6}" \
  -fit off \
  -np 1 \
  --host 0.0.0.0 --port 8080 \
  "$@"
