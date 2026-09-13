#!/bin/bash
# Is the SYCL rope kernel itself wrong under YaRN, or is YaRN just wrong for
# this checkpoint?  test-backend-ops' ROPE battery already sweeps
# ext_factor = {0, 0.7465} x freq_scale = {1, 1.4245} x attn_factor and covers
# GGML_ROPE_TYPE_IMROPE (what qwen4exp uses) at partial n_dims (20 and 32 of a
# 128-wide row), against the CPU reference.  A pass means the divergence is not
# in the kernel.
set -uo pipefail
B=/devbin
export LD_LIBRARY_PATH="$B:/opt/intel/oneapi/2025.3/lib:/opt/intel/oneapi/compiler/2025.3/lib"
"$B/test-backend-ops" test -b SYCL0 -o ROPE 2>&1 | tail -25
