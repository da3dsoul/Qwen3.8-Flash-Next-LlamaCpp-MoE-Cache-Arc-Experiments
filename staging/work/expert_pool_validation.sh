#!/bin/bash
# Production-scale validation of the Phase 1 static expert pool
# (docs/research/14) against the real 116K-token chat_prompt, comparing:
#   baseline  - the project's standing production config, -ncmoe 24
#   pool      - -ncmoe 48 (everything CPU) + the static pool sized to the
#               same expert VRAM as -ncmoe 24 (23,100 MiB, docs/research/14
#               section 1.3's expert_vram(24) = 46200 - 962.5*24)
#   floor     - -ncmoe 48 alone, no pool: the "do nothing" CPU floor
# All three: same prompt, same -c/-ub, --ignore-eos so decode length is exact
# and comparable, -lzm auto throughout (this run cares about decode speed,
# not TTFT, and auto keeps host RAM bounded across all three).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
O="${O:-$ROOT/staging/work/expert-pool-validation}"
mkdir -p "$O"

M=/models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
CFG="-ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -c 122880 -ub 2048 -lzm auto -fit off -st --no-warmup --temp 0 --seed 42"

# run_one <tag> <n_predict> <extra docker -e args...> -- <extra llama-completion args...>
run_one() {
    local tag="$1"; local n_predict="$2"; shift 2
    local -a env_args=()
    while [ "$1" != "--" ]; do env_args+=("$1"); shift; done
    shift # consume --

    echo "### $tag  n_predict=$n_predict  $(date +%H:%M:%S)"
    docker run --rm \
        --device /dev/dri/renderD129:/dev/dri/renderD129 \
        --device /dev/dri/card0:/dev/dri/card0 \
        --group-add 44 --group-add 136 \
        -e NEOReadDebugKeys=1 -e EnableSharedSystemUsmSupport=0 \
        -e EnableImplicitMigrationOnFaultableHardware=0 -e GGML_SYCL_USM_SYSTEM=0 \
        -e LD_LIBRARY_PATH="/devbin:/opt/intel/oneapi/2025.3/lib:/opt/intel/oneapi/compiler/2025.3/lib" \
        "${env_args[@]}" \
        --memory=90g --memory-swap=90g --ipc=host \
        -v "$ROOT/staging/models:/models:ro" \
        -v "$ROOT/staging/work:/work" \
        -v "$ROOT/staging/devbin:/devbin:ro" \
        -w /work \
        --entrypoint /devbin/llama-completion \
        qwen4exp-moe-cache:sycl-pinned \
        -m "$M" $CFG -no-cnv -f /work/bench120k/chat_prompt.txt -n "$n_predict" --ignore-eos "$@" \
        > "$O/${tag}.log" 2>&1
    echo "###   rc=$?  $(date +%H:%M:%S)"
    grep -E "common_perf_print|EXPERT_POOL: enabled|MMID_HYBRID: [0-9]" "$O/${tag}.log"
}

run_one baseline-ncmoe24 300 -- -ncmoe 24
run_one pool-ncmoe48-23100mib 300 \
    -e GGML_SYCL_EXPERT_PROFILE=/work/moe-profile/decode_profile.csv \
    -e GGML_SYCL_EXPERT_VRAM_MIB=23100 \
    -- -ncmoe 48
run_one floor-ncmoe48-nopool 100 -- -ncmoe 48

echo "### all runs done $(date +%H:%M:%S)"
