#!/bin/bash
# llama-server for validating a THIRD --context-tiers tier whose n_ctx exceeds
# n_ctx_train (262144), combined with the doc-15 "nearunity" YaRN flag that
# lifts the server's own n_ctx_slot cap for exactly that case
# (docs/research/13 addendum, 2026-09-13). Never tested together before.
#
# CPU-only, small model (Qwen3.6-35B-A3B, also n_ctx_train=262144, verified),
# so a reload costs ~5s instead of ~5min and the whole test is safe to run
# alongside anything else on the box -- see tier_switch_run.sh's arm-1
# precedent (docs/research/13 §9.9), extended here with a third tier that
# deliberately crosses the training-context ceiling.
set -euo pipefail
export LD_LIBRARY_PATH="/devbin:/opt/intel/oneapi/2025.3/lib:/opt/intel/oneapi/compiler/2025.3/lib"

exec /devbin/llama-server \
  -m /models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  -ngl 0 --device none -t 4 \
  --context-tiers "${TIERS:-1024:512:4,8192:1024:8,280000:1024:12}" \
  --context-tier-reserve "${RESERVE:-64}" \
  --context-tier-down-turns "${DOWNTURNS:-100}" \
  --rope-scaling yarn --rope-scale 1.0001 --yarn-orig-ctx "${YARN_ORIG_CTX:-280000}" \
  --sleep-idle-seconds "${SLEEP_IDLE:-6}" \
  -fit off -np 1 \
  --host 0.0.0.0 --port 8080
