#!/bin/bash
# Host-side sampler run alongside a long-context benchmark.
# Records, every INTERVAL seconds: the llama process's RSS / major+minor fault
# counters / scheduler state / utime+stime, plus bcache0+md0 %util and the box's
# overall CPU idle. This is the evidence needed to decide whether a slow decode
# token is paging (majflt climbing), host CPU work (utime climbing), or neither.
set -uo pipefail
INTERVAL="${INTERVAL:-5}"
OUT="${1:-/dev/stdout}"

prev_maj=0; prev_min=0; prev_ut=0; prev_st=0; prev_t=0
{
echo "# ts pid rss_gib state majflt/s minflt/s utime%cpu stime%cpu bcache0_util md0_util idle%"
while true; do
  pid=$(pgrep -f '/devbin/llama-(bench|server|cli)' | head -1)
  now=$(date +%s.%N)
  if [ -n "$pid" ] && [ -r "/proc/$pid/stat" ]; then
    read -r -a S < "/proc/$pid/stat"
    # fields (1-indexed): 3 state, 10 minflt, 12 majflt, 14 utime, 15 stime, 24 rss(pages)
    st=${S[2]}; mnf=${S[9]}; mjf=${S[11]}; ut=${S[13]}; stm=${S[14]}; rss=${S[23]}
    rssg=$(awk -v r="$rss" 'BEGIN{printf "%.2f", r*4096/1073741824}')
    if [ "$prev_t" != "0" ]; then
      dt=$(awk -v a="$now" -v b="$prev_t" 'BEGIN{print a-b}')
      mjs=$(awk -v a="$mjf" -v b="$prev_maj" -v d="$dt" 'BEGIN{printf "%.1f",(a-b)/d}')
      mns=$(awk -v a="$mnf" -v b="$prev_min" -v d="$dt" 'BEGIN{printf "%.0f",(a-b)/d}')
      utp=$(awk -v a="$ut" -v b="$prev_ut" -v d="$dt" 'BEGIN{printf "%.0f",100*(a-b)/100/d}')
      stp=$(awk -v a="$stm" -v b="$prev_st" -v d="$dt" 'BEGIN{printf "%.0f",100*(a-b)/100/d}')
    else
      mjs=-; mns=-; utp=-; stp=-
    fi
    prev_maj=$mjf; prev_min=$mnf; prev_ut=$ut; prev_st=$stm; prev_t=$now
  else
    pid=-; rssg=-; st=-; mjs=-; mns=-; utp=-; stp=-
    prev_t=0
  fi
  io=$(iostat -x 1 2 2>/dev/null | awk '/^bcache0/{b=$NF} /^md0/{m=$NF} END{print b, m}')
  idle=$(top -bn1 | awk '/^%Cpu/{print $8}')
  echo "$(date +%H:%M:%S) $pid $rssg $st $mjs $mns $utp $stp $io $idle"
  sleep "$INTERVAL"
done
} >> "$OUT" 2>&1
