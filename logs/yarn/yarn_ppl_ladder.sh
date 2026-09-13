#!/bin/bash
# Perplexity depth ladder, YaRN on vs off, on the real 116K refactor corpus.
#
# Purpose: the project's standing conclusion is "YaRN garbles this GGUF, even
# below n_ctx_train". That was established from one greedy generation. This
# measures it quantitatively and as a function of DEPTH, which is the thing the
# single-point test could not separate: static YaRN perturbs theta in
# proportion to position, so if the damage is positional it must grow with
# depth and be near-zero at shallow depth.
#
# One chunk per invocation (--chunks 1) so the chunk size IS the depth: every
# scored token sits at position < -c, and the deepest ones sit just under it.
# mmap (no -lm none) deliberately: 52.8 GiB of page cache instead of 52.8 GiB
# of pinned USM, which keeps the box safe across many sequential arms. Load
# mode cannot change the logits.
set -uo pipefail
B=/devbin
M=/models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
export LD_LIBRARY_PATH="$B:/opt/intel/oneapi/2025.3/lib:/opt/intel/oneapi/compiler/2025.3/lib"
O=${O:-/work/yarn-ppl-ladder.log}
: > "$O"

COMMON="-m $M -f /work/bench120k/full_prompt.txt -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -ncmoe 24 -ub 2048 -lzm off --chunks 1"

run() {
    local tag="$1"; shift
    echo -n "### $tag $(date +%H:%M:%S) : " >> "$O"
    timeout 3600 "$B/llama-perplexity" $COMMON "$@" > /tmp/ppl.out 2>&1
    local rc=$?
    grep -E 'Final estimate' /tmp/ppl.out | tail -1 >> "$O" || true
    echo "###   rc=$rc  $(grep -cE 'Final estimate' /tmp/ppl.out) est  $(date +%H:%M:%S)" >> "$O"
    grep -iE 'GGML_ASSERT|error|abort' /tmp/ppl.out | head -3 >> "$O"
    cp /tmp/ppl.out "/work/yarn-ppl-$tag.out"
}

# interleave the two arms at each depth so any drift shows up as a within-depth
# inconsistency rather than as a fake trend
for depth in 8192 32768 98304; do
    # --rope-scale is n_ctx / n_ctx_orig; keep it fixed at the project's 1.1875
    # so the ONLY variable across depths is position, not the scaling factor.
    run "off-$depth"  -c "$depth"
    run "yarn-$depth" -c "$depth" --rope-scaling yarn --rope-scale 1.1875 --yarn-orig-ctx 262144
done
echo "### done $(date +%H:%M:%S)" >> "$O"
