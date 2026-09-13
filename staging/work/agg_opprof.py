#!/usr/bin/env python3
"""Aggregate GGML_SYCL_OP_PROFILE=2 GGML_SYCL_OP_PROFILE_WINDOW=1 output.

The profiler prints one windowed report per scheduler split; with WINDOW=1 that
is thousands of small reports. This sums them back into one table per section
("by op" / "by call site") so a whole prefill can be attributed in one place.
"""
import re
import sys
from collections import defaultdict

SECTION = re.compile(r"^\s+(by op|by call site) \((\d+) buckets, (\d+) calls, ([\d.]+) ms\):")
ROW = re.compile(r"^\s{4}(.+?)\s{2,}([\d.]+) ms\s+\([\d.]+%\)\s+(\d+) calls")
SUBMIT = re.compile(r"^\s+host dispatch loop ([\d.]+) ms, tail wait ([\d.]+) ms")

sections = {"by op": defaultdict(lambda: [0.0, 0]), "by call site": defaultdict(lambda: [0.0, 0])}
cur = None
submit = 0.0
wait = 0.0
splits = 0

for line in open(sys.argv[1], errors="replace"):
    if line.startswith("OP_PROFILE"):
        splits += 1
        cur = None
        continue
    m = SUBMIT.match(line)
    if m:
        submit += float(m.group(1))
        wait += float(m.group(2))
        continue
    m = SECTION.match(line)
    if m:
        cur = m.group(1)
        continue
    if cur:
        m = ROW.match(line)
        if m:
            e = sections[cur][m.group(1).strip()]
            e[0] += float(m.group(2))
            e[1] += int(m.group(3))
        else:
            cur = None

print(f"splits={splits}  host dispatch loop={submit/1000:.2f} s  tail wait={wait/1000:.2f} s")
for name, d in sections.items():
    tot = sum(v[0] for v in d.values())
    print(f"\n== {name} == total {tot/1000:.2f} s over {len(d)} buckets")
    for k, v in sorted(d.items(), key=lambda kv: -kv[1][0])[:20]:
        print(f"  {k:<48} {v[0]/1000:9.2f} s  ({100*v[0]/tot if tot else 0:5.1f}%)  {v[1]} calls")
