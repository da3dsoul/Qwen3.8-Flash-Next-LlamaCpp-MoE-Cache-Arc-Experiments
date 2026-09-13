#!/bin/bash
# Decode-throughput-vs-depth curve for the long-context decode gap investigation
# (docs/research/11). Pure `llama_decode` cost -- llama-bench never touches the
# sampler chain -- at the config PLAN.md's latest update recommends.
#
# Arm order matters: d2048 appears FIRST and LAST. The leading arm is the
# throwaway the `-ub` sweep's page-cache confound taught this project to
# discard; the trailing one is the control that says whether the run drifted.
# If the two d2048 numbers disagree by more than a few percent, the run is
# contaminated and none of the middle arms are usable.
set -uo pipefail
export LD_LIBRARY_PATH="/devbin:/opt/intel/oneapi/2025.3/lib:/opt/intel/oneapi/compiler/2025.3/lib"

MODEL=/models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf

exec /devbin/llama-bench \
  -m "$MODEL" \
  -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -ncmoe 24 -ub 2048 \
  -p 0 -n 32 \
  -d "${DEPTHS:-2048,8192,32768,65536,2048}" \
  -r 1 \
  "$@"
