#!/bin/bash
# Replaces yarn_deep.sh's perplexity arms, which cannot exist: llama-perplexity
# reserves n_ctx * n_vocab floats of HOST memory for one chunk
# (tools/perplexity/perplexity.cpp:514), i.e. 262144 * 151936 * 4 B = 159 GB at
# -c 262144. It dies with std::bad_alloc after tokenizing (measured, rc=134).
# Perplexity is therefore capped at roughly -c 98304 on this model regardless of
# VRAM, so it cannot be used to measure anything past the native ceiling.
#
# The currency for depth instead is a real greedy generation on the 305,759-token
# prompt, single-variable against yarn_deep.sh's coherent no-rope-flags arm
# (same prompt, same -c 311296, same --temp 0 --seed 42, same binary, same
# -ncmoe 32 -ub 2048 -lzm off).
#
#  A. gen-yarn-305k      -- YaRN at a depth that genuinely exceeds n_ctx_orig.
#     This is the arm the brief asked for: the repo's "YaRN destroys this GGUF"
#     verdict came from a 116K test, BELOW n_ctx_orig 262,144, so it could not
#     separate "wrong for this checkpoint" from "applied where nothing needed
#     scaling". 305,759 > 262,144, so here YaRN is doing the job it exists for.
#  B. beta sweep         -- PLAN.md's standing open item.
#  C. gen-nearunity-305k -- the near-unity reparametrisation that lifts
#     llama-server's n_ctx_slot cap from 262,144 to 311,296 has to leave the
#     output alone, or it is not a workaround.
set -uo pipefail
B=/devbin
M=/models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
export LD_LIBRARY_PATH="$B:/opt/intel/oneapi/2025.3/lib:/opt/intel/oneapi/compiler/2025.3/lib"
O=${O:-/work/yarn-deep2.log}
: > "$O"
CFG="-ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -ncmoe 32 -ub 2048 -lzm off -fit off"

gen() {
    local tag="$1"; shift
    echo "### gen-$tag $(date +%H:%M:%S)" >> "$O"
    timeout 7200 "$B/llama-completion" -m "$M" -no-cnv $CFG -c 311296 \
        -f /work/bench300k/chat_prompt.txt -n 250 --temp 0 --seed 42 -st --no-warmup "$@" \
        > "/work/yarn-gen-$tag.txt" 2>"/work/yarn-gen-$tag.err"
    echo "###   rc=$? $(date +%H:%M:%S)" >> "$O"
    grep -E 'prompt eval time|eval time' "/work/yarn-gen-$tag.err" | tail -3 >> "$O"
    echo "--- continuation (text after the final 'assistant' marker) ---" >> "$O"
    python3 - "$tag" >> "$O" <<'PY'
import sys
t=sys.argv[1]
s=open(f"/work/yarn-gen-{t}.txt", errors="replace").read()
m="assistant\n"
i=s.rindex(m) if m in s else 0
print(s[i+len(m):][:1400])
PY
}

gen yarn-305k --rope-scaling yarn --rope-scale 1.1875 --yarn-orig-ctx 262144

/work/yarn_beta_sweep.sh
cat /work/yarn-beta-sweep.log >> "$O"

gen nearunity-305k --rope-scaling yarn --rope-scale 1.0001 --yarn-orig-ctx 311296

echo "### done $(date +%H:%M:%S)" >> "$O"
