#!/bin/bash
# ABBA-ordered A/B: base, dev, dev, base, ... so a monotone drift over the session
# (page cache filling, thermal, whatever) cancels between the two sides instead of
# always favouring whichever side runs second.
# Same prompt and token budget as the op-profile runs, so the two measurements line up.
set -uo pipefail
M=/models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
OT='\.ffn_(gate|up|down)_exps\.weight=SYCL_MoE_Cached'
PROMPT="Count from one to fifty."

run() {
    echo "@@@ RUN $1"
    if [ "$1" = base ]; then
        echo "$PROMPT" | llama-cli -m "$M" -ngl 99 -fa 1 -ot "$OT" \
            -st -n 300 --temp 0 --seed 42 --no-reasoning-preserve
    else
        echo "$PROMPT" | LD_LIBRARY_PATH=/devbin:${LD_LIBRARY_PATH:-} \
            /devbin/llama-cli -m "$M" -ngl 99 -fa 1 -ot "$OT" \
            -st -n 300 --temp 0 --seed 42 --no-reasoning-preserve
    fi
    echo "@@@ END $1"
}

for i in $(seq 1 "${ROUNDS:-2}"); do
    run base; run dev; run dev; run base
done
