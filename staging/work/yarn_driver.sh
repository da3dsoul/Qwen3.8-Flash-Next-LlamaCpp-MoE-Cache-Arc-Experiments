#!/bin/bash
# Host driver: wait for exclusive GPU (another agent's perplexity gate was live
# when this was queued), confirm RAM headroom, then run the YaRN ppl ladder.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

free=0
while [ "$free" -lt 3 ]; do
    if pgrep -x 'llama-server|llama-cli|llama-completion|llama-bench|llama-perplexity|test-backend-ops' >/dev/null 2>&1; then
        free=0
    else
        free=$((free+1))
    fi
    sleep 20
done
echo "### GPU acquired $(date)"
free -h | tee logs/yarn/box-state-pre.txt
avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
echo "### available GiB = $avail"
if [ "$avail" -lt 25 ]; then echo "### ABORT: not enough RAM headroom"; exit 1; fi

docker compose -f docker/docker-compose.yml run --rm --entrypoint /usr/bin/env \
    -v "$PWD/staging/devbin:/devbin" llm-test-sycl \
    bash -c 'O=/work/yarn-ppl-ladder.log /work/yarn_ppl_ladder.sh' \
    > logs/yarn/ladder-container.log 2>&1
echo "### ladder rc=$? $(date)"
cp staging/work/yarn-ppl-ladder.log logs/yarn/ 2>/dev/null
cat staging/work/yarn-ppl-ladder.log 2>/dev/null
