#!/bin/bash
# Long-context RoPE validation past the model's native 262,144-token ceiling.
#
# Arms, in the order that puts the cheapest/highest-value result first:
#  0. slotcap  - can llama-server be made to accept a >n_ctx_train prompt with
#                a rope reparametrisation that is numerically a no-op?
#  1. gen-off  - 305,814-token chat prompt, -c 311296, NO rope flags: the
#                honest human-readable answer for unscaled extrapolation, which
#                is what every 300K/600K speed number in this repo actually ran.
#  2. ppl-262144-off  - perplexity exactly AT the native ceiling (control).
#  3. ppl-294912-off  - perplexity 12.5% PAST it, still unscaled. (2) vs (3) is
#                the cost of extrapolation with depth and sparsity budget held
#                as close to equal as this harness allows.
#  4. ppl-294912-yarn - YaRN at a depth that genuinely exceeds n_ctx_orig, with
#                --rope-scale 1.125 = 294912/262144 exactly. This is the arm the
#                brief asked for: the repo's "YaRN destroys this GGUF" finding
#                came from a 116K test, below n_ctx_orig, and could not rule out
#                "YaRN was applied where no scaling was needed".
#
# llama-perplexity refuses a chunk unless the corpus tokenizes to >= 2*n_ctx
# (measured the hard way: the 116K corpus could not do -c 98304), so arms 2-4
# use a fresh ~611K-token corpus from the sibling generator at n=800.
#
# mmap deliberately (-lzm off, NO -lm none): at -ncmoe 32 the host side is
# ~61 GiB, and page cache is reclaimable where pinned USM is not. The box killed
# background processes for memory during the shallow ladder; load mode cannot
# change a single logit, only the wall clock.
set -uo pipefail
B=/devbin
M=/models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
export LD_LIBRARY_PATH="$B:/opt/intel/oneapi/2025.3/lib:/opt/intel/oneapi/compiler/2025.3/lib"
O=${O:-/work/yarn-deep.log}
: > "$O"
CFG="-ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -ncmoe 32 -ub 2048 -lzm off -fit off"

say() { echo "$@" >> "$O"; }

say "### corpus tokenization $(date +%H:%M:%S)"
"$B/llama-tokenize" -m "$M" -f /work/bench600k/full_prompt.txt --show-count 2>&1 | tail -2 >> "$O"

# ---- arm 0: the slot cap -------------------------------------------------
for arm in plain nearunity; do
    EXTRA=""
    [ "$arm" = nearunity ] && EXTRA="--rope-scaling yarn --rope-scale 1.0001 --yarn-orig-ctx 311296"
    say "### slotcap/$arm  '$EXTRA'  $(date +%H:%M:%S)"
    timeout 1200 "$B/llama-server" -m "$M" $CFG -c 311296 -np 1 \
        --host 127.0.0.1 --port 8099 $EXTRA > "/work/yarn-slotcap-$arm.log" 2>&1 &
    pid=$!
    for i in $(seq 1 1100); do
        grep -qE 'n_ctx_slot =' "/work/yarn-slotcap-$arm.log" && break
        kill -0 $pid 2>/dev/null || break
        sleep 1
    done
    grep -E 'n_ctx_slot =|exceeds the training context|re-adjusting|n_ctx_train adjusted|freq_scale' \
        "/work/yarn-slotcap-$arm.log" >> "$O"
    kill $pid 2>/dev/null; wait $pid 2>/dev/null
    say "### slotcap/$arm done $(date +%H:%M:%S)"
done

# ---- arm 1: readable generation at real depth, unscaled ------------------
say "### gen-off-305k $(date +%H:%M:%S)"
timeout 7200 "$B/llama-completion" -m "$M" -no-cnv $CFG -c 311296 \
    -f /work/bench300k/chat_prompt.txt -n 250 --temp 0 --seed 42 -st --no-warmup \
    > /work/yarn-gen-off-305k.txt 2>/work/yarn-gen-off-305k.err
say "###   rc=$? $(date +%H:%M:%S)"
grep -E 'prompt eval time|^.*eval time|n_ctx =|training context' /work/yarn-gen-off-305k.err | tail -5 >> "$O"
say "--- continuation, first 1200 chars ---"
head -c 1200 /work/yarn-gen-off-305k.txt >> "$O"; say ""

# ---- arms 2-4: perplexity straddling the native ceiling -----------------
ppl() {
    local tag="$1"; shift
    say "### $tag $(date +%H:%M:%S)"
    timeout 7200 "$B/llama-perplexity" -m "$M" -f /work/bench600k/full_prompt.txt \
        $CFG --chunks 1 "$@" > "/work/yarn-$tag.out" 2>&1
    say "###   rc=$? $(date +%H:%M:%S)"
    grep -E 'Final estimate|^\[1\]|you need at least|exceeds|training context' "/work/yarn-$tag.out" | tail -4 >> "$O"
}
ppl ppl-262144-off  -c 262144
ppl ppl-294912-off  -c 294912
ppl ppl-294912-yarn -c 294912 --rope-scaling yarn --rope-scale 1.125 --yarn-orig-ctx 262144

say "### done $(date +%H:%M:%S)"
