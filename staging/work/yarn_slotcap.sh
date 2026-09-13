#!/bin/bash
# llama-server caps a slot at n_ctx_train (server-context.cpp:4199-4207), so a
# 300K prompt is refused however large -c is. The ONLY thing that lifts the cap
# is llama-context.cpp:3782-3786, which rewrites hparams.n_ctx_train to
# n_ctx_orig_yarn / rope_freq_scale -- but only when the scaling type is YARN
# *and* rope_freq_scale differs from the trained value. This repo's 300K server
# run bought that lift with --rope-scale 1.1875, which destroys the output.
#
# Test: buy the same lift with a factor so close to 1 that the RoPE it applies
# is numerically indistinguishable from no scaling at all. At --rope-scale
# 1.0001 the interpolated bands are multiplied by 0.99990 and mscale is
# 1.00001; the largest theta shift anywhere at position 311,296 is 0.002 rad,
# against the ~1.0 rad that --rope-scale 1.1875 puts on the same band.
#
# Pass = the banner reports n_ctx_slot = 311296 rather than 262144.
set -uo pipefail
B=/devbin
M=/models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
export LD_LIBRARY_PATH="$B:/opt/intel/oneapi/2025.3/lib:/opt/intel/oneapi/compiler/2025.3/lib"
O=${O:-/work/yarn-slotcap.log}
: > "$O"

for arm in plain nearunity; do
    EXTRA=""
    [ "$arm" = nearunity ] && EXTRA="--rope-scaling yarn --rope-scale 1.0001 --yarn-orig-ctx 311296"
    echo "### arm=$arm  extra='$EXTRA'  $(date +%H:%M:%S)" >> "$O"
    timeout 900 "$B/llama-server" -m "$M" -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 \
        -ncmoe 32 -ub 2048 -c 311296 -lzm off -fit off -np 1 \
        --host 127.0.0.1 --port 8099 $EXTRA > /tmp/srv-$arm.log 2>&1 &
    pid=$!
    for i in $(seq 1 780); do
        grep -qE 'n_ctx_slot =' /tmp/srv-$arm.log && break
        kill -0 $pid 2>/dev/null || break
        sleep 1
    done
    grep -E 'n_ctx_slot =|training context|re-adjusting|n_ctx_train adjusted|n_ctx_seq' /tmp/srv-$arm.log >> "$O"
    kill $pid 2>/dev/null; wait $pid 2>/dev/null
    echo "### arm=$arm done $(date +%H:%M:%S)" >> "$O"
done
