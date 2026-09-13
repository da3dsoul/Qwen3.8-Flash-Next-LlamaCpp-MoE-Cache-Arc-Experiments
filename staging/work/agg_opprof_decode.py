#!/usr/bin/env python3
"""Aggregate the DECODE portion of a GGML_SYCL_OP_PROFILE run, by op and call site.

`agg_opprof.py` sums a whole run, which for a `-d <depth>` benchmark is
dominated by the depth prefill -- one prefill ubatch's FLASH_ATTN_EXT over 2048
queries swamps 32 decode tokens' worth of single-query calls. This script
isolates the decode tail instead.

How the boundary is found, exactly: the model has 12 full-attention layers, so
every decode token emits exactly 12 FLASH_ATTN_EXT calls. Walking the per-split
reports backwards and stopping once 12*n_gen of them have been counted lands on
the first split of the decode phase without needing to know how many scheduler
splits a token costs. That is why the profiler must be run with
GGML_SYCL_OP_PROFILE_WINDOW=1 (one report per split): a coarser window can
straddle the prefill/decode boundary and there is then no way to unmix it.

Usage: agg_opprof_decode.py <log> <n_gen> [n_fa_layers]
"""
import re
import sys
from collections import defaultdict

SECTION = re.compile(r"^\s+(by op|by call site) \((\d+) buckets, (\d+) calls, ([\d.]+) ms\):")
ROW = re.compile(r"^\s{4}(.+?)\s{2,}([\d.]+) ms\s+\([\d.]+%\)\s+(\d+) calls")

log = sys.argv[1]
n_gen = int(sys.argv[2])
n_fa_layers = int(sys.argv[3]) if len(sys.argv) > 3 else 12

# ---- pass 1: cut the log into per-report blocks -------------------------------
blocks = []
cur = None
for line in open(log, errors="replace"):
    if line.startswith("OP_PROFILE"):
        cur = []
        blocks.append(cur)
    elif cur is not None:
        cur.append(line)
print(f"{len(blocks)} profiler reports in {log}")


def parse(block):
    """-> {section: {bucket: [ms, calls]}}"""
    out = {"by op": {}, "by call site": {}}
    sec = None
    for line in block:
        m = SECTION.match(line)
        if m:
            sec = m.group(1)
            continue
        if sec:
            m = ROW.match(line)
            if m:
                out[sec][m.group(1).strip()] = [float(m.group(2)), int(m.group(3))]
            else:
                sec = None
    return out


parsed = [parse(b) for b in blocks]

# ---- pass 2: walk back until 12*n_gen FLASH_ATTN_EXT calls are accounted for ---
want = n_fa_layers * n_gen
seen = 0
start = len(parsed)
for i in range(len(parsed) - 1, -1, -1):
    fa = parsed[i]["by op"].get("FLASH_ATTN_EXT")
    if fa:
        seen += fa[1]
    start = i
    if seen >= want:
        break
print(f"decode window = reports [{start}:{len(parsed)}]  "
      f"({len(parsed)-start} splits, {seen} FLASH_ATTN_EXT calls, wanted {want})")
if seen < want:
    print("WARNING: never reached the expected FA call count -- boundary is not trustworthy")

# ---- aggregate ---------------------------------------------------------------
for sec in ("by op", "by call site"):
    agg = defaultdict(lambda: [0.0, 0])
    for p in parsed[start:]:
        for k, v in p[sec].items():
            agg[k][0] += v[0]
            agg[k][1] += v[1]
    tot = sum(v[0] for v in agg.values())
    print(f"\n== {sec} == {tot:.1f} ms over {n_gen} decode tokens "
          f"= {tot/n_gen:.2f} ms/token, {len(agg)} buckets")
    print(f"  {'bucket':<46} {'ms/token':>9} {'share':>7} {'calls/token':>12}")
    for k, v in sorted(agg.items(), key=lambda kv: -kv[1][0])[:30]:
        print(f"  {k:<46} {v[0]/n_gen:9.3f} {100*v[0]/tot if tot else 0:6.1f}% "
              f"{v[1]/n_gen:12.1f}")
