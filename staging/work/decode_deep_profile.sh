#!/bin/bash
# Deep-depth decode profile for the `-lm none/mlock` null-result investigation.
#
# The point doc 11 never measured: a `llama-bench` decode token at the REAL
# benchmark depth. Doc 11 built its component model from d2048/d8192 only --
# its d32768/d65536 arms were aborted -- and then compared the extrapolation
# against an `llama-server` number. That comparison cannot distinguish
# "model compute is more expensive than the isolated-op model says" from
# "the serving path costs more at depth". llama-bench never touches the
# sampler chain, so running it AT d118016 separates the two directly.
#
# `-lm none` (not mlock): the container has `ulimit -l` = 8 MiB and no
# CAP_IPC_LOCK, so `--mlock` cannot lock the 52.8 GiB CPU-resident expert set
# at all -- llama_mlock::impl::raw_lock (src/llama-mmap.cpp:689-719) warns and
# returns false. `-lm none` is the only arm that genuinely removes mmap paging.
set -uo pipefail
export LD_LIBRARY_PATH="/devbin:/opt/intel/oneapi/2025.3/lib:/opt/intel/oneapi/compiler/2025.3/lib"

MODEL=/models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf

exec /devbin/llama-bench \
  -m "$MODEL" \
  -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -ncmoe 24 -ub 2048 \
  -lm "${LM:-none}" \
  -p 0 -n "${NGEN:-32}" \
  -d "${DEPTHS:-2048,32768,118016}" \
  -r 1 \
  "$@"
