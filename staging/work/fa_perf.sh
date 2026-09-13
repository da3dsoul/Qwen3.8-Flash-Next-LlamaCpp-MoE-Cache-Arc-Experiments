#!/bin/bash
# Fast FLASH_ATTN_EXT perf loop: no model load, real SYCL timing.
set -uo pipefail
FILT="${1:-hsk=256,hsv=256,nh=2,nr23=\\[12,1\\]}"
test-backend-ops perf -o FLASH_ATTN_EXT -p "$FILT" -b SYCL0
