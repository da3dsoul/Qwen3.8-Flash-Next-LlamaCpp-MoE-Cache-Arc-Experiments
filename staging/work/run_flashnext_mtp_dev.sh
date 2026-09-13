#!/bin/bash
# Best known config as of 2026-09-11 (see PLAN.md's 2026-09-11 update):
# -ncmoe static split + MTP speculative decoding, ~26-29 t/s vs ~23-25 t/s
# without MTP. MTP support isn't in the pinned image yet (implemented today,
# not committed) -- runs against staging/devbin, built via devbuild.sh.
# Needs the downloaded draft head at
# staging/models/Qwen3.8-Flash-Next-GGUF/MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf
# (the self-contained variant -- the "shared-" one isn't supported, see
# PLAN.md). Do NOT swap in UD-Q3_K_XL here: MTP+Q3_K_XL is confirmed broken
# (3.2 t/s, not just untuned -- see PLAN.md's 2026-09-11 update) until
# root-caused. Extra llama-cli args can be passed on the command line.
set -euo pipefail
export LD_LIBRARY_PATH="/devbin:/opt/intel/oneapi/2025.3/lib:/opt/intel/oneapi/compiler/2025.3/lib:${LD_LIBRARY_PATH:-}"
echo "${PROMPT:-Write a short story about a robot who discovers music for the first time.}" | \
  /devbin/llama-cli -m /models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf \
    -md /models/Qwen3.8-Flash-Next-GGUF/MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf \
    --spec-type draft-mtp --spec-draft-n-max 2 \
    -ngl 99 -fa 1 -ncmoe 26 \
    -st -n "${NGEN:-200}" --temp 0 --seed 42 "$@"
