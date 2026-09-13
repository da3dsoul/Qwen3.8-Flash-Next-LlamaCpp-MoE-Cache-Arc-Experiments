#!/bin/bash
# llama-server for the --context-tiers validation.
#
# Tier boundaries are deliberately small (TIERS below) so a real tier crossing
# costs one model reload and a few thousand tokens of reprefill instead of the
# 20 min a 300K prompt costs. The reload path is identical either way -- it is
# destroy() + load_model() with a different (n_ctx, n_ubatch, n_cpu_moe) -- and
# the reload cost this measures IS the production one, because the load is
# dominated by -lzm off's eager PLE table read, which does not depend on -c.
#
# Everything else is the recommended production config verbatim
# (PLAN.md 2026-09-11 "Recommended production config"), minus -c/-ub/-ncmoe,
# which the tier table now owns.
set -euo pipefail
export LD_LIBRARY_PATH="/devbin:/opt/intel/oneapi/2025.3/lib:/opt/intel/oneapi/compiler/2025.3/lib"

exec /devbin/llama-server \
  -m /models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf \
  -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 \
  --context-tiers "${TIERS:-4096:512:24,32768:2048:26}" \
  --context-tier-reserve "${RESERVE:-256}" \
  --context-tier-down-turns "${DOWNTURNS:-2}" \
  ${LZM--lzm off} ${LM--lm none} \
  -fit off \
  -np 1 \
  ${EXTRA-} \
  --host 0.0.0.0 --port 8080 \
  "$@"
