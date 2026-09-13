#!/bin/bash
# CPU-only (-ngl 0) determinism check, leg 1 of 2. Per the forked code-reading pass
# (session 2026-09-12), ggml/src/ggml-sycl/topk-radix.cpp's tie-break at the top-k boundary
# uses a local atomic fetch_add whose winner is a GPU scheduling race -- CPU top-k
# (ggml-cpu/ops.cpp cmp_top_k, std::partial_sort, no parallel reduction) should not have this
# problem. Same prompt/config docs/research/12 section 9.5 already showed SYCL output drift on
# (prompt1500, -c 2048 -ub 512 -n 64), run twice here with -ngl 0. Expect byte-identical if the
# hypothesis is right.
set -uo pipefail
M=/models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
export LD_LIBRARY_PATH=/devbin:/opt/intel/oneapi/2025.3/lib:/opt/intel/oneapi/compiler/2025.3/lib

for i in 1 2; do
  echo "### run$i $(date +%H:%M:%S)"
  /devbin/llama-completion -m "$M" -ngl 0 -fa 1 -ctk q4_0 -ctv q4_0 -lzm off \
      -c 2048 -ub 512 -n 64 --temp 0 --seed 42 \
      -f /work/bench120k/prompt1500.txt \
      > "/work/determinism-cpu-run$i.txt" 2> "/work/determinism-cpu-run$i.err"
  echo "### run$i rc=$? $(date +%H:%M:%S)"
done
echo "### done $(date +%H:%M:%S)"
