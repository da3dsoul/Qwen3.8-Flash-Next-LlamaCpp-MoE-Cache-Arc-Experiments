#!/bin/bash
# Exact-equivalence A/B of the QSA pooled-key cache, one binary, two arms.
set -uo pipefail
B=/devbin
M=/models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
export LD_LIBRARY_PATH="$B:/opt/intel/oneapi/2025.3/lib:/opt/intel/oneapi/compiler/2025.3/lib"

NP=${NP:-4000}
ND=${ND:-24}
NC=${NC:-8192}
UB=${UB:-512}

run() {
  local tag=$1; shift
  echo "### arm=$tag $(date +%H:%M:%S)"
  "$B/qsa_pool_equiv" "$M" "$NP" "$ND" "$NC" "$UB" 24 2>/dev/null
  echo "### rc=$?"
}

: > /work/qsa-pool-equiv-cached.log
: > /work/qsa-pool-equiv-plain.log
: > /work/qsa-pool-equiv-cached2.log

unset LLAMA_QSA_NO_POOL_CACHE
run cached  >> /work/qsa-pool-equiv-cached.log
export LLAMA_QSA_NO_POOL_CACHE=1
run plain   >> /work/qsa-pool-equiv-plain.log
unset LLAMA_QSA_NO_POOL_CACHE
run cached2 >> /work/qsa-pool-equiv-cached2.log
