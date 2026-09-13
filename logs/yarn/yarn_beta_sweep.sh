#!/bin/bash
# PLAN.md has carried "none of --yarn-attn-factor / --yarn-beta-fast /
# --yarn-beta-slow was swept" as an open item since the 300K run. This closes it.
#
# beta_fast/beta_slow set which frequency bands YaRN interpolates, via
# ggml_rope_yarn_corr_dims(n_dims=64, n_ctx_orig=262144, freq_base=1e7, ...).
# Computed for this GGUF (scratchpad/yarn_math.py):
#   beta 32/1   (default) -> corr_dims [14,22] : pairs >=22 fully interpolated
#   beta 64/2             -> corr_dims [12,20] : MORE bands interpolated
#   beta  4/0.125         -> corr_dims [18,26] : FEWER bands interpolated
#   ext_factor 0          -> no ramp at all, freq_scale on every band (linear)
# So the four arms are ordered by how much of the rotary space gets rescaled,
# and the question is whether any of them beats simply not scaling.
#
# Depth 32,768 because that is where the shallow ladder already measured a
# large, unambiguous YaRN penalty (43.10 -> 82.62) on this exact corpus and
# config, so a beta setting that helps has something visible to recover.
set -uo pipefail
B=/devbin
M=/models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
export LD_LIBRARY_PATH="$B:/opt/intel/oneapi/2025.3/lib:/opt/intel/oneapi/compiler/2025.3/lib"
O=${O:-/work/yarn-beta-sweep.log}
: > "$O"
COMMON="-m $M -f /work/bench120k/full_prompt.txt -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -ncmoe 24 -ub 2048 -lzm off --chunks 1 -c 32768"

run() {
    local tag="$1"; shift
    echo -n "### $tag $(date +%H:%M:%S) : " >> "$O"
    timeout 3600 "$B/llama-perplexity" $COMMON "$@" > "/work/yarn-beta-$tag.out" 2>&1
    grep -E 'Final estimate' "/work/yarn-beta-$tag.out" | tail -1 >> "$O" || echo "(none)" >> "$O"
    echo "###   rc=$? $(date +%H:%M:%S)" >> "$O"
}

Y="--rope-scaling yarn --rope-scale 1.1875 --yarn-orig-ctx 262144"
run off                                                        # reference, repeat of the ladder
run yarn-default   $Y
run yarn-beta64-2  $Y --yarn-beta-fast 64 --yarn-beta-slow 2
run yarn-beta4-125 $Y --yarn-beta-fast 4  --yarn-beta-slow 0.125
run linear         --rope-scaling linear --rope-scale 1.1875
echo "### done $(date +%H:%M:%S)" >> "$O"
