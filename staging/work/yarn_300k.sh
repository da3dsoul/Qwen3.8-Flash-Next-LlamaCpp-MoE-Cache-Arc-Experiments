#!/bin/bash
# Quality at REAL long context, past the model's native 262,144-token ceiling.
#
# Two questions, one corpus (bench300k, 305,801 real tokens from the sibling
# project's generator at n=400, the same prompt the 300K timing run used):
#
#  1. What does UNSCALED RoPE extrapolation actually cost?  Every 300K/600K
#     speed number in this repo was taken by just setting a big -c and letting
#     llama.cpp extrapolate past n_ctx_train with a warning. Nobody measured
#     the output. Arms `off-*` answer that: perplexity over one chunk whose
#     size IS the depth, at 98,304 (under the native ceiling, control) and
#     294,912 (past it).
#  2. Does YaRN help when it is finally applied at a depth that actually needs
#     it?  The project's "YaRN destroys this GGUF" finding came from a test at
#     116K -- below n_ctx_orig -- so it could not distinguish "YaRN is wrong
#     for this checkpoint" from "YaRN was applied where no scaling was needed".
#     Arm `yarn-294912` closes that: same depth, same corpus, --rope-scale
#     1.125 = 294912/262144 exactly, --yarn-orig-ctx 262144.
#
# -ncmoe 32 -ub 2048 is the measured 300K envelope (docs/research/12): compute
# buffer 8,280 MiB at 294,912 + 19,216 model + 2,673 KV + 456 pooled = 30,625
# of the card's measured ~31,900 MiB.  -lm none -lzm off is the production load
# mode and keeps the 61 GiB host-side expert set out of the page cache.
set -uo pipefail
B=/devbin
M=/models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
export LD_LIBRARY_PATH="$B:/opt/intel/oneapi/2025.3/lib:/opt/intel/oneapi/compiler/2025.3/lib"
O=${O:-/work/yarn-300k.log}
: > "$O"

CFG="-ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -ncmoe 32 -ub 2048 -lm none -lzm off"

ppl() {
    local tag="$1"; shift
    echo "### ppl $tag $(date +%H:%M:%S)" >> "$O"
    timeout 7200 "$B/llama-perplexity" -m "$M" -f /work/bench300k/full_prompt.txt \
        $CFG --chunks 1 "$@" > "/work/yarn300k-$tag.out" 2>&1
    echo "###   rc=$? $(date +%H:%M:%S)" >> "$O"
    grep -E 'Final estimate|^\[1\]|n_ctx_seq|training context' "/work/yarn300k-$tag.out" | tail -4 >> "$O"
}

cmp_gen() {
    local tag="$1"; shift
    echo "### gen $tag $(date +%H:%M:%S)" >> "$O"
    timeout 7200 "$B/llama-completion" -m "$M" -no-cnv $CFG \
        -f /work/bench300k/chat_prompt.txt -c 311296 -n 250 --temp 0 --seed 42 \
        -st --no-warmup "$@" > "/work/yarn300k-$tag.txt" 2>"/work/yarn300k-$tag.err"
    echo "###   rc=$? $(date +%H:%M:%S)" >> "$O"
    grep -E 'prompt eval time|eval time|n_ctx =' "/work/yarn300k-$tag.err" | tail -4 >> "$O"
    echo "--- first 600 chars of continuation ---" >> "$O"
    head -c 600 "/work/yarn300k-$tag.txt" >> "$O"; echo >> "$O"
}

# control first: same corpus, same binary, a depth the model was trained for
ppl off-98304  -c 98304
# the headline: unscaled extrapolation 12.5% past n_ctx_train
ppl off-294912 -c 294912
# YaRN, finally applied at a depth that needs it
ppl yarn-294912 -c 294912 --rope-scaling yarn --rope-scale 1.125 --yarn-orig-ctx 262144
# human-readable: does it still do the task at 305,814 tokens with no rope flags
cmp_gen off-305k

echo "### done $(date +%H:%M:%S)" >> "$O"
