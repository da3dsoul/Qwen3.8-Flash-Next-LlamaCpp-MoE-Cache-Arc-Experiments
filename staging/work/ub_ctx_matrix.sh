#!/bin/bash
# Isolate the -ub 2048 / -c 163840 hang found in ub_phase2.sh: at that pairing
# llama_context construction spins one core forever (RSS frozen, 0 progress,
# killed after 13.5 min), while -ub 2048 / -c 65536 reserves in ~8 s.
#
# Each arm is given a hard 240 s timeout -- a healthy reserve takes <15 s, and
# prefill on the 32K stand-in takes 80-330 s, so 240 s cleanly separates
# "hung in reserve" (timeout with no `generate:` line) from "ran".
# Output is unbuffered (stdbuf) so a hang is visible at the exact log line.
set -uo pipefail

M=/models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
P=/work/bench120k/prompt32k.txt
export LD_LIBRARY_PATH="/devbin:/opt/intel/oneapi/2025.3/lib:/opt/intel/oneapi/compiler/2025.3/lib"

# reserve-only probe: -n 0 so we pay context construction but not prefill
probe() {  # $1=ub $2=ctx $3=ncmoe $4=extra
    echo "@@@ PROBE ub=$1 ctx=$2 ncmoe=$3 extra='${4:-}'"
    timeout -s KILL 240 stdbuf -oL -eL /devbin/llama-completion -m "$M" \
        -p "hello" -no-cnv \
        -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -ncmoe "$3" -c "$2" \
        -b 2048 -ub "$1" ${4:-} \
        -n 1 --temp 0 --seed 42 --no-warmup -lv 4 2>&1 \
      | grep -E 'buffer size|graph splits|n_ctx |generate:|SYCL error|Exception|abort|error'
    echo "@@@ PROBE_END ub=$1 ctx=$2 ncmoe=$3 rc=${PIPESTATUS[0]}"
    echo "@@@ SEP"
}

probe 2048 163840 24 "-fit off"
probe 2048 163840 26 "-fit off"
probe 2048 131072 24 "-fit off"
probe 2048 122880 24 "-fit off"
probe 1024 163840 24 "-fit off"
