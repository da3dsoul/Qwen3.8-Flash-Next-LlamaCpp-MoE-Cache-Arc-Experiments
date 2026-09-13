#!/bin/bash
# Host-side thermal/clock/throttle sampler for the run-to-run speed-variance investigation
# (session 2026-09-12, general infra question raised during YaRN benchmarking, not YaRN-specific).
#
# Prior investigation (docs/research/12 section 11.8) found d614400 decode spans 17.27-1.76 tok/s
# across identical-config arms, with a "session-order effect" (second arm of a campaign always
# slower) left unexplained -- own paging and neighbour I/O were both ruled out, but nothing in the
# project ever sampled GPU/CPU thermal or throttle state during a run. This does that.
#
# GT0 throttle reasons (xe driver, /sys/class/drm/card0/device/tile0/gt0/freq0/throttle) are ground
# truth, not inference: reason_pl1/pl2/pl4 = power-limit throttling, reason_thermal/prochot/
# vr_thermalert = thermal throttling, reason_ratl = running-average power limit. All zero at idle
# (verified before this probe was written).
set -uo pipefail
INTERVAL="${INTERVAL:-3}"
OUT="${1:-/dev/stdout}"
GT=/sys/class/drm/card0/device/tile0/gt0/freq0

{
echo "# ts cpu_tctl gpu_pkg_C gpu_vram_C gpu_act_freq gpu_cur_freq gpu_throttle_status gpu_throttle_reasons load1"
while true; do
  ts=$(date +%H:%M:%S)
  read -r tctl gpu_pkg gpu_vram <<< "$(sensors 2>/dev/null | awk -v RS="" -v FS="\n" '
    { if ($1 ~ /^xe-pci-0300/) { for (i=1;i<=NF;i++) {
          if ($i ~ /^pkg:.*°C/) { split($i,a," "); p=a[2] }
          if ($i ~ /^vram:.*°C/) { split($i,a," "); v=a[2] } } }
      if ($1 ~ /^k10temp/) { for (i=1;i<=NF;i++) { if ($i ~ /^Tctl:/) { split($i,a," "); t=a[2] } } }
    }
    END { print t, p, v }')"
  act=$(cat "$GT/act_freq" 2>/dev/null || echo -)
  cur=$(cat "$GT/cur_freq" 2>/dev/null || echo -)
  tstatus=$(cat "$GT/throttle/status" 2>/dev/null || echo -)
  treasons=$(cat "$GT/throttle/reasons" 2>/dev/null || echo -)
  load1=$(awk '{print $1}' /proc/loadavg)
  echo "$ts $tctl $gpu_pkg $gpu_vram $act $cur $tstatus $treasons $load1"
  sleep "$INTERVAL"
done
} >> "$OUT" 2>&1
