#!/bin/bash
# Phase 0 of docs/research/14-usage-based-static-expert-placement.md section 6:
# collect a per-(layer, expert) hit histogram at -ncmoe 48 (every expert tensor
# on the CPU path, so every routed id passes through the instrumented loop in
# ggml/src/ggml-cpu/ggml-cpu.c). Two prompts (full_prompt.txt, chat_prompt.txt)
# for cross-prompt stability (doc 14 section 9 item 2); for each prompt, one
# n_predict=1 run (prefill-only histogram) and one n_predict=300 run
# (prefill+decode combined) -- diffed later to get decode-only counts.
#
# -lzm auto (not -lzm off): this run does not care about decode speed, and
# -lzm off would pin ~27 GiB of PLE table in host RAM on top of the ~46 GiB
# of expert tensors (doc 14 section 6 step 1).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
O="${O:-$ROOT/staging/work/moe-profile}"
mkdir -p "$O"

CFG="-ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -ncmoe 48 -c 122880 -ub 2048 -lzm auto -fit off -st --no-warmup --temp 0 --seed 42"

run_one() {
    local tag="$1" prompt="$2" n_predict="$3"
    echo "### $tag  n_predict=$n_predict  $(date +%H:%M:%S)"
    docker run --rm \
        --device /dev/dri/renderD129:/dev/dri/renderD129 \
        --device /dev/dri/card0:/dev/dri/card0 \
        --group-add 44 --group-add 136 \
        -e NEOReadDebugKeys=1 -e EnableSharedSystemUsmSupport=0 \
        -e EnableImplicitMigrationOnFaultableHardware=0 -e GGML_SYCL_USM_SYSTEM=0 \
        -e GGML_MOE_EXPERT_PROFILE=/work/moe-profile/${tag}.csv \
        -e LD_LIBRARY_PATH="/devbin:/opt/intel/oneapi/2025.3/lib:/opt/intel/oneapi/compiler/2025.3/lib" \
        --memory=90g --memory-swap=90g --ipc=host \
        -v "$ROOT/staging/models:/models:ro" \
        -v "$ROOT/staging/work:/work" \
        -v "$ROOT/staging/devbin:/devbin:ro" \
        -w /work \
        --entrypoint /devbin/llama-completion \
        qwen4exp-moe-cache:sycl-pinned \
        -m /models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf \
            $CFG -no-cnv -f "/work/bench120k/$prompt" -n "$n_predict" \
        > "$O/${tag}.log" 2>&1
    echo "###   rc=$?  $(date +%H:%M:%S)"
    wc -l "$O/${tag}.csv" 2>&1 || echo "  NO CSV PRODUCED for $tag"
}

for p in full_prompt.txt chat_prompt.txt; do
    tag="${p%.txt}"
    run_one "${tag}-prefill" "$p" 1
    run_one "${tag}-combined" "$p" 300
done

echo "### all runs done $(date +%H:%M:%S)"
