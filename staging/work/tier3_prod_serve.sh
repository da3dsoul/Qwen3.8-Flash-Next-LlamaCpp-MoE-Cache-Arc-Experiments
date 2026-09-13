#!/bin/bash
# Real production --context-tiers crossing, redoing docs/research/13 §9.10
# arm 2b (which never completed -- the box crashed on a sequencing error, a
# concurrent llama-bench also loading at -ncmoe 32 -lm none while this arm's
# own reload was in flight, not a technical limitation of the tier mechanism
# itself). Recommended table from §9.3/§11, extended with the reinstated
# 500K tier from §11 and the --yarn-orig-ctx margin fix from §12 (set above
# the tier's actual -c, not equal to it, to avoid the padding-vs-cap
# collision §12 found).
#
# mmap, not -lm none -lzm off, deliberately: §11's own arm (yarn_600k.sh)
# proved mmap safe at -ncmoe 38 (~67 GiB host) against less headroom than
# this box has now, and this test's tier 2 is -ncmoe 32 (lighter). Trades
# away worst-case-cold reload timing for host-RAM safety, which is the
# right trade after the arm 2b crash.
set -euo pipefail
export LD_LIBRARY_PATH="/devbin:/opt/intel/oneapi/2025.3/lib:/opt/intel/oneapi/compiler/2025.3/lib"

exec /devbin/llama-server \
  -m /models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf \
  -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 \
  --context-tiers "${TIERS:-122880:2048:24,262144:2048:30,512000:1024:32}" \
  --context-tier-reserve "${RESERVE:-4096}" \
  --context-tier-down-turns "${DOWNTURNS:-100}" \
  --rope-scaling yarn --rope-scale 1.0001 --yarn-orig-ctx "${YARN_ORIG_CTX:-513000}" \
  -lzm off -fit off -np 1 \
  --host 0.0.0.0 --port 8080
