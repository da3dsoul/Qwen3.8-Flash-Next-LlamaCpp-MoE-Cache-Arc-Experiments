#!/bin/bash
# Same model and flags as run_flashnext.sh, but through llama-completion instead of
# llama-cli. llama-completion prints llama_perf_context_print, which splits the wall
# clock into prompt eval / eval and reports how many graphs were reused - the numbers
# llama-cli's TUI does not show.
set -uo pipefail
M=/models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
OT='\.ffn_(gate|up|down)_exps\.weight=SYCL_MoE_Cached'
BIN="${BIN:-/devbin/llama-completion}"
export LD_LIBRARY_PATH="/devbin:${LD_LIBRARY_PATH:-}"

"$BIN" -m "$M" -ngl 99 -fa 1 -ot "$OT" -no-cnv \
    -p "Write a short story about a robot who discovers music for the first time." \
    -n "${NGEN:-200}" --temp 0 --seed 42 2>&1
