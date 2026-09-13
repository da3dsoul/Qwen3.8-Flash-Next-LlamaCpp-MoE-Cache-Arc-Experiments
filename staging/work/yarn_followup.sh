#!/bin/bash
# Two follow-ups, cheapest first.
#
# A. The beta sweep PLAN.md has carried as an open item since the 300K run.
# B. Validate the near-unity rope reparametrisation that lifts llama-server's
#    n_ctx_slot cap from 262,144 to 311,296 (proved in yarn_deep.sh arm 0):
#    it has to produce output indistinguishable from no rope flags at all, or
#    it is not a workaround, just a smaller version of the same damage. Same
#    prompt, same seed, same everything as yarn_deep.sh's gen-off-305k.
set -uo pipefail
B=/devbin
M=/models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
export LD_LIBRARY_PATH="$B:/opt/intel/oneapi/2025.3/lib:/opt/intel/oneapi/compiler/2025.3/lib"
O=${O:-/work/yarn-followup.log}
: > "$O"

/work/yarn_beta_sweep.sh
cat /work/yarn-beta-sweep.log >> "$O"

CFG="-ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -ncmoe 32 -ub 2048 -lzm off -fit off"
echo "### gen-nearunity-305k $(date +%H:%M:%S)" >> "$O"
timeout 7200 "$B/llama-completion" -m "$M" -no-cnv $CFG -c 311296 \
    -f /work/bench300k/chat_prompt.txt -n 250 --temp 0 --seed 42 -st --no-warmup \
    --rope-scaling yarn --rope-scale 1.0001 --yarn-orig-ctx 311296 \
    > /work/yarn-gen-nearunity-305k.txt 2>/work/yarn-gen-nearunity-305k.err
echo "###   rc=$? $(date +%H:%M:%S)" >> "$O"
grep -E 'prompt eval time|eval time' /work/yarn-gen-nearunity-305k.err | tail -3 >> "$O"
echo "--- continuation, first 1200 chars ---" >> "$O"
head -c 1200 /work/yarn-gen-nearunity-305k.txt >> "$O"; echo >> "$O"
echo "### done $(date +%H:%M:%S)" >> "$O"
