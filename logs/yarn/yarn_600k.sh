#!/bin/bash
# The one gap docs/research/14 left open: quality at 600K via unscaled
# extrapolation. 614,400 is 2.34x the native 262,144 ceiling, and every theta
# shift doc 14 tracks is linear in position, so the 305,759-token result does
# not transfer.
#
# Config is the one already proven to fit at this depth (PLAN.md's 600K row,
# logs/qsa-sparse-600k.log): -ncmoe 38 -ub 1024 -c 614400. mmap, not -lm none:
# at -ncmoe 38 the host side is ~67 GiB against ~76 GiB available, and page
# cache is reclaimable where pinned USM is not.
#
# Corpus: sibling generator at n=800, 612,680 tokens verified with this model's
# own tokenizer, wrapped in the model's chat template. 612,693 + 250 generated
# sits inside 614,400.
set -uo pipefail
B=/devbin
M=/models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
export LD_LIBRARY_PATH="$B:/opt/intel/oneapi/2025.3/lib:/opt/intel/oneapi/compiler/2025.3/lib"
O=${O:-/work/yarn-600k.log}
: > "$O"
echo "### gen-off-600k $(date +%H:%M:%S)" >> "$O"
timeout 14400 "$B/llama-completion" -m "$M" -no-cnv \
    -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -ncmoe 38 -ub 1024 -c 614400 -lzm off -fit off \
    -f /work/bench600k/chat_prompt.txt -n 250 --temp 0 --seed 42 -st --no-warmup \
    > /work/yarn-gen-off-600k.txt 2>/work/yarn-gen-off-600k.err
echo "###   rc=$? $(date +%H:%M:%S)" >> "$O"
grep -E 'prompt eval time|eval time|n_ctx =|training context|error|OUT_OF' /work/yarn-gen-off-600k.err | tail -6 >> "$O"
echo "--- continuation ---" >> "$O"
python3 - >> "$O" <<'PY'
s=open("/work/yarn-gen-off-600k.txt",errors="replace").read()
m="assistant\n"
i=s.rindex(m) if m in s else 0
print(s[i+len(m):][:1500])
PY
echo "### done $(date +%H:%M:%S)" >> "$O"
