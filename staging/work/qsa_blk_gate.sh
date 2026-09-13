#!/bin/bash
set -uo pipefail
NEW=/devbin; OLD=/devbin-before
M=/models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
ONEAPI=/opt/intel/oneapi/2025.3/lib:/opt/intel/oneapi/compiler/2025.3/lib
run() { local bin=$1 out=$2; shift 2
  LD_LIBRARY_PATH="$bin:$ONEAPI" "$bin/llama-completion" -m "$M" -no-cnv \
    -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -ncmoe 24 --temp 0 --seed 42 -st --no-warmup "$@" \
    > "$out" 2>"$out.err"; echo "  $out exit=$? bytes=$(wc -c <"$out")"; }

echo "=== 1. op equivalence vs CPU ==="
LD_LIBRARY_PATH="$NEW:$ONEAPI" "$NEW/test-backend-ops" test -b SYCL0 \
  -o QSA_INDEXER,QSA_INDEXER_BLK,QSA_INDEXER_CACHED,QSA_INDEXER_CACHED_BLK 2>&1 | grep -E 'QSA_INDEXER|passed|FAIL'

echo "=== 2. n_kv <= width: the two paths must select every cell -> identical output ==="
SH="-f /work/bench120k/prompt1500.txt -c 2048 -ub 512 -n 64"
LLAMA_QSA_CELL_TOPK=1 run "$NEW" /work/g-shallow-cell.txt $SH
run "$NEW" /work/g-shallow-blk.txt $SH
run "$OLD" /work/g-shallow-oldtree.txt $SH
cmp -s /work/g-shallow-cell.txt /work/g-shallow-blk.txt \
  && echo "  cell vs block : IDENTICAL" || { echo "  cell vs block : DIFFER"; diff /work/g-shallow-cell.txt /work/g-shallow-blk.txt | head -6; }
cmp -s /work/g-shallow-oldtree.txt /work/g-shallow-cell.txt \
  && echo "  old tree vs new tree, per-cell path : IDENTICAL" || { echo "  old tree vs new tree, per-cell path : DIFFER"; diff /work/g-shallow-oldtree.txt /work/g-shallow-cell.txt | head -6; }
