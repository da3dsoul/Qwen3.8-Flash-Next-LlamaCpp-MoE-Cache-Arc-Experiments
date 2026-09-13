#!/bin/bash
# Host-side driver: wait for exclusive GPU, then run the QSA block-top-k perf and
# correctness batteries. /devbin = the new tree, /devbin-before = this morning's
# build (unchanged model code), used to show set_input_qsa is neutral.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# require the GPU to be free for 60 consecutive seconds before grabbing it
free=0
while [ "$free" -lt 3 ]; do
    # match the process NAME, not the command line: a -f pattern also matches this script
    if pgrep -x 'llama-server|llama-cli|llama-completion|llama-bench|llama-perplexity|test-backend-ops' >/dev/null 2>&1; then
        free=0
    else
        free=$((free+1))
    fi
    sleep 20
done
echo "### GPU acquired $(date)"
mkdir -p logs/qsa-blocktopk

DC="docker compose -f docker/docker-compose.yml run --rm --entrypoint /usr/bin/env"
MNT="-v $PWD/staging/devbin-blk:/devbin -v $PWD/staging/devbin:/devbin-before"

# correctness gate first: it is cheap, and a failure makes the perf sweep pointless
echo "### optest $(date)"
timeout 1800 $DC $MNT llm-test-sycl bash -c \
    'LD_LIBRARY_PATH=/devbin:/opt/intel/oneapi/2025.3/lib:/opt/intel/oneapi/compiler/2025.3/lib \
     /devbin/test-backend-ops test -b SYCL0 -o QSA_INDEXER,QSA_INDEXER_BLK,QSA_INDEXER_CACHED,QSA_INDEXER_CACHED_BLK' \
    > logs/qsa-blocktopk/optest.log 2>&1
rc=$?
echo "### optest rc=$rc $(date)"
tail -3 logs/qsa-blocktopk/optest.log
if [ "$rc" -ne 0 ]; then echo "### ABORT: op-level correctness failed"; exit 1; fi

echo "### perf $(date)"
timeout 2400 $DC $MNT llm-test-sycl bash -c '/work/qsa_blk_perf.sh' \
    > logs/qsa-blocktopk/perf.log 2>&1
echo "### perf rc=$? $(date)"

echo "### gate $(date)"
timeout 3600 $DC $MNT llm-test-sycl bash -c '/work/qsa_blk_gate.sh' \
    > logs/qsa-blocktopk/gate.log 2>&1
echo "### gate rc=$? $(date)"

echo "### e2e $(date)"
timeout 3600 $DC $MNT llm-test-sycl bash -c '/work/qsa_blk_e2e.sh' \
    > logs/qsa-blocktopk/e2e.log 2>&1
echo "### e2e rc=$? $(date)"
