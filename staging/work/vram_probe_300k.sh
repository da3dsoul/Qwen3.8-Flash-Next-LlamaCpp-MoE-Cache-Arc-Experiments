#!/bin/bash
# Allocator check at the real 300K+ target depth, with YaRN on.
#
# Method is the one established earlier today: load with a trivial prompt, read
# the buffer lines, and decide from the numbers -- never commit an expensive
# real-prompt run to a config whose `SYCL0 compute buffer size` has not been
# read against the card's measured ~31.9 GiB usable envelope first. -fit off so
# llama.cpp cannot silently reshape -c/-ncmoe to make it fit.
#
# -lzm auto here on purpose: the PLE table lands in SYCL_Host either way (both
# logs/long-context-120k-ub2048 and -lazyoff show SYCL0 model buffer =
# 26916.16 MiB at -ncmoe 24), so load mode does not move the VRAM answer and
# `auto` loads in ~60 s instead of ~5 min.
set -uo pipefail
export LD_LIBRARY_PATH="/devbin:/opt/intel/oneapi/2025.3/lib:/opt/intel/oneapi/compiler/2025.3/lib"
M=/models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
NCTX="${NCTX:-311296}"
# 311296 / 262144 = 1.1875 exactly
SCALE="${SCALE:-1.1875}"

probe() {
    local ub="$1" ncmoe="$2"
    echo "######## -c $NCTX -ub $ub -ncmoe $ncmoe  (yarn x$SCALE) ########"
    timeout 900 /devbin/llama-cli -m "$M" \
        -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -ncmoe "$ncmoe" -ub "$ub" \
        -c "$NCTX" -fit off \
        --rope-scaling yarn --rope-scale "$SCALE" --yarn-orig-ctx 262144 \
        --temp 0 --seed 1 -n 4 -no-cnv -p "Hello." 2>&1 \
      | grep -vE '^[\|/\\-]+$' \
      | grep -E 'rope|freq_|n_ctx_orig|n_ctx_seq|n_ctx  |n_ubatch|model buffer size|llama_kv_cache: size|RS buffer|compute buffer size|graph nodes|graph splits|out of|OUT_OF|error|failed|abort|GGML_ASSERT'
    echo
}

probe 2048 32
probe 2048 33
probe 1024 28
