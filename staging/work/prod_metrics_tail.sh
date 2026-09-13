#!/bin/bash
# Keeps prod_metrics_tail.py attached to llm-b70's log stream, reconnecting
# whenever `docker logs -f` exits (container restart/recreation -- expected
# behavior any time --sleep-idle-seconds sleeps+wakes the process itself
# does NOT restart the container, so this mainly covers `docker compose up`
# recreations and host reboots, not normal sleep/wake cycles).
#
# Not a systemd unit -- deliberately lightweight per the user's own choice.
# Start: nohup staging/work/prod_metrics_tail.sh > logs/prod-metrics/tail.log 2>&1 &
# Stop:  pkill -f prod_metrics_tail.sh; pkill -f prod_metrics_tail.py
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$ROOT/logs/prod-metrics/requests.ndjson"
mkdir -p "$(dirname "$OUT")"

while true; do
    echo "$(date -Is) attaching to llm-b70 logs"
    docker logs -f --timestamps llm-b70 2>&1 | python3 "$ROOT/staging/work/prod_metrics_tail.py" "$OUT"
    echo "$(date -Is) log stream ended, retrying in 5s"
    sleep 5
done
