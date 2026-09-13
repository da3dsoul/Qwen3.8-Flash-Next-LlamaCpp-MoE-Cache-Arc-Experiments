#!/bin/bash
# Same-binary A/B for the set_input_qsa host-term fix, with its host-side profile, at the config
# 600K actually needs (-ub 1024 -ncmoe 38 per PLAN.md's 600K bullet).
#
# Both arms run sequentially in ONE container, one llama-bench process each: two model loads can
# never overlap (the box was taken down once today by two concurrent ~60 GiB loads), and each arm
# covers every depth from a single load, which matters because a 614 400-token prefill is ~40 min.
#
# -lzm off without -lm none on purpose: PLAN.md measured `-lm auto -lzm off` decode at 10.90 vs
# `-lm none -lzm off`'s 11.146 tok/s (same within noise) while using ~30 GiB of host RAM instead
# of ~66. Decode is what this change moves, so the cheaper-on-RAM arm is the honest one to use
# here; it costs prefill time, not the number being measured.
#
# ARM=slow -> LLAMA_QSA_SLOW_PLAN=1: the general grouping, the per-block bias loop, and the
# cell_blk/blk_of columns a block-granularity selection never reads. ARM=fast -> the new path.
set -uo pipefail
B=${B:-/devbin}
M=/models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
export LD_LIBRARY_PATH="$B:/opt/intel/oneapi/2025.3/lib:/opt/intel/oneapi/compiler/2025.3/lib"
export LLAMA_QSA_HOST_PROF=1

NCMOE=${NCMOE:-38}
UB=${UB:-1024}
DEPTHS=${DEPTHS:-8192,32768,614400}
ARMS=${ARMS:-"slow fast"}

O=${O:-/work/qsa-plan-ab.log}
: > "$O"

for arm in $ARMS; do
    if [ "$arm" = slow ]; then export LLAMA_QSA_SLOW_PLAN=1; else unset LLAMA_QSA_SLOW_PLAN; fi

    echo "### arm=$arm ncmoe=$NCMOE ub=$UB d=$DEPTHS $(date +%H:%M:%S)" >> "$O"

    "$B/llama-bench" -m "$M" -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -lzm off \
        -ub "$UB" -ncmoe "$NCMOE" -p 0 -n 32 -r 1 -d "$DEPTHS" 2>&1 \
        | stdbuf -oL grep -E --line-buffered 'qsa host prof|^\| qwen|error|Error|abort|ASSERT' >> "$O"

    echo "### arm=$arm rc=${PIPESTATUS[0]} $(date +%H:%M:%S)" >> "$O"
done

echo "### done $(date +%H:%M:%S)" >> "$O"
