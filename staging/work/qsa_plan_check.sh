#!/bin/bash
# Correctness gate for the set_input_qsa host-term fix.
#
# LLAMA_QSA_PLAN_CHECK=1 builds the general grouping into a second stream struct and the general
# per-block bias row into a scratch buffer, on the same cells in the same call, and asserts both
# agree with the single-sequence shortcuts. The oracle has to be in-call: this stack's forward
# pass is not reproducible across processes (docs/research/12 section 10.2), so nothing that spans
# two runs can be used. A GGML_ASSERT here aborts the process, so "ran to completion" is the pass.
#
# Drivers: llama-bench and llama-perplexity only. llama-cli, llama-completion and
# llama-batched-bench all crash on this model in this fork *before* any of this change
# (reproduced on staging/devbin-sparse, the 13:50 build that predates it).
#
# mmap + lazy PLE deliberately (no -lm none -lzm off): which cells group into which block does not
# depend on where the weights live, and this keeps host RAM at ~10 GiB instead of ~52.
set -uo pipefail
B=${B:-/devbin}
M=/models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
export LD_LIBRARY_PATH="$B:/opt/intel/oneapi/2025.3/lib:/opt/intel/oneapi/compiler/2025.3/lib"
export LLAMA_QSA_PLAN_CHECK=1

COMMON="-m $M -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -ncmoe 24 -lzm off"

O=${O:-/work/qsa-plan-check.log}
: > "$O"

run() {
    local tag="$1"; shift
    echo "### $tag $(date +%H:%M:%S)" >> "$O"
    "$@" > /tmp/chk.out 2>&1
    local rc=$?
    grep -iE 'GGML_ASSERT|disagree|abort|error' /tmp/chk.out | head -5 >> "$O"
    grep -E '^\| .*(t/s|[0-9])' /tmp/chk.out | tail -4 >> "$O"
    grep -E 'Final estimate|\[1\]' /tmp/chk.out | tail -2 >> "$O"
    echo "### $tag rc=$rc $(date +%H:%M:%S)" >> "$O"
}

# A: an empty cache (n_bid == 0, spare-block-only bias row) and a shallow one
run "A empty+shallow cache"  "$B/llama-bench" $COMMON -ub 512 -p 0 -n 16 -d 0,2048 -r 1

# B: prefill ubatches at n_tps = 512 and 2048, where the bias check is O(n_tps * n_blocks)
run "B prefill ubatches"     "$B/llama-bench" $COMMON -ub 512  -p 4096 -n 16 -r 1
run "B2 prefill ub 2048"     "$B/llama-bench" $COMMON -ub 2048 -p 8192 -n 16 -r 1

# C: deeper decode, and a depth that is not a multiple of the compress ratio times the ubatch
run "C decode at depth"      "$B/llama-bench" $COMMON -ub 512 -p 0 -n 32 -d 8191,16384 -r 1

# D: the per-cell selection, so cell_blk and blk_of are built on both sides and compared too
export LLAMA_QSA_CELL_TOPK=1
run "D per-cell selection" "$B/llama-bench" $COMMON -ub 512 -p 0 -n 32 -d 8192 -r 1
unset LLAMA_QSA_CELL_TOPK

# E: perplexity - many chunks through one context, so cells are cleared and refilled between them,
#    which is the pooled-key cache's invalidation path and the plan's worst case for block ids
run "E perplexity 3 chunks"  "$B/llama-perplexity" $COMMON -ub 512 -c 4096 -f /work/bench120k/prompt32k.txt --chunks 3

echo "### done $(date +%H:%M:%S)" >> "$O"
