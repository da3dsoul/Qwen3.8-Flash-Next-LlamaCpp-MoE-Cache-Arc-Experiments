#!/bin/bash
# Does `--rope-scaling yarn` actually register, or is it silently ignored?
#
# The load banner prints hparams' TRAINED rope scaling ("rope scaling = linear")
# and llama_context prints only freq_base/freq_scale -- neither prints the
# effective rope_scaling_type or yarn_ext_factor, so the banner alone cannot
# tell YaRN-active from YaRN-ignored. Three arms at identical everything else,
# greedy, same seed:
#   A  no rope flags                      -> freq_scale 1,      ext_factor 0
#   B  --rope-scaling linear --rope-scale -> freq_scale 0.84..., ext_factor 0
#   C  --rope-scaling yarn   --rope-scale -> freq_scale 0.84..., ext_factor 1
# B vs A proves --rope-scale took effect. C vs B proves the *type* took effect,
# i.e. that ext_factor reached the rope kernel -- the only thing that separates
# them is yarn_ext_factor, since freq_scale is identical.
set -uo pipefail
export LD_LIBRARY_PATH="/devbin:/opt/intel/oneapi/2025.3/lib:/opt/intel/oneapi/compiler/2025.3/lib"
M=/models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
P="List the first eight prime numbers, then explain in one sentence why 1 is not prime."
COMMON=(-m "$M" -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -ncmoe 24 -ub 512 -c 8192
        -fit off --temp 0 --seed 1 -n 48 -st -v)

run() {
    local tag="$1"; shift
    echo "########## ARM $tag : $* ##########"
    /devbin/llama-cli "${COMMON[@]}" "$@" -p "$P" 2>&1 \
        | tr '\r' '\n' | grep -vE '^[\|/\\-]+$' \
        | grep -E 'freq_base_train|freq_base|freq_scale|n_ctx_orig_yarn|rope scaling|rope type|yarn|n_ctx_seq|^[A-Za-z0-9]' \
        | tail -40
    echo
}

run A
run B --rope-scaling linear --rope-scale 1.1875 --yarn-orig-ctx 262144
run C --rope-scaling yarn   --rope-scale 1.1875 --yarn-orig-ctx 262144
