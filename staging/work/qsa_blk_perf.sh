#!/bin/bash
# Isolated cost of the QSA indexer chain, per-cell vs block-granularity selection.
# QSA_INDEXER / QSA_INDEXER_BLK are the two arms; _CACHED is the same pair with the
# pooled-key derivation hoisted out, i.e. what the change is worth once the pooled
# cache of docs/research/12 5.1 exists.
set -uo pipefail

BIN=${BIN:-/devbin}
export LD_LIBRARY_PATH="$BIN:/opt/intel/oneapi/2025.3/lib:/opt/intel/oneapi/compiler/2025.3/lib"

OPS=${OPS:-QSA_INDEXER,QSA_INDEXER_BLK,QSA_INDEXER_CACHED,QSA_INDEXER_CACHED_BLK}
"$BIN/test-backend-ops" perf -b SYCL0 -o "$OPS"
echo "@@@ test-backend-ops exit=$?"
