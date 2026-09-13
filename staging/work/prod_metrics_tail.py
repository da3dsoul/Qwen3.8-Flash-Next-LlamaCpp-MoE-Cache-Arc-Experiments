#!/usr/bin/env python3
"""Tail llm-b70's own per-request timing log lines and emit one NDJSON
record per completed request. Read-only -- parses the print_timing/release
lines llama-server already prints (tools/server/server-context.cpp
print_timings(), server-common.h SLT_INF), no server-side change needed.

Usage: docker logs -f --timestamps llm-b70 | prod_metrics_tail.py OUTFILE
"""
import json
import re
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "requests.ndjson"

TS_RE      = re.compile(r'^(\S+) (.*)$')
TIER_RE    = re.compile(r'switching context tier \d+ -> (\d+)')
PROMPT_RE  = re.compile(r'task (\d+) \| prompt eval time = +([\d.]+) ms / +(\d+) tokens \([^,]+, *([\d.]+) tokens per second\)')
EVAL_RE    = re.compile(r'task (\d+) \| +eval time = +([\d.]+) ms / +(\d+) tokens \([^,]+, *([\d.]+) tokens per second\)')
RELEASE_RE = re.compile(r'task (\d+) \| stop processing: n_tokens = (\d+), truncated = (\d+)')

tier = 0
pending = {}

with open(OUT, "a", buffering=1) as out:
    for raw in sys.stdin:
        m = TS_RE.match(raw.rstrip("\n"))
        if not m:
            continue
        docker_ts, line = m.groups()

        m = TIER_RE.search(line)
        if m:
            tier = int(m.group(1))
            continue

        m = PROMPT_RE.search(line)
        if m:
            task_id = int(m.group(1))
            rec = pending.setdefault(task_id, {})
            rec["prompt_ms"]          = float(m.group(2))
            rec["prompt_tokens_new"]  = int(m.group(3))
            rec["prompt_tok_s"]       = float(m.group(4))
            continue

        m = EVAL_RE.search(line)
        if m:
            task_id = int(m.group(1))
            rec = pending.setdefault(task_id, {})
            rec["predicted_ms"]     = float(m.group(2))
            rec["predicted_tokens"] = int(m.group(3))
            rec["predicted_tok_s"]  = float(m.group(4))
            continue

        m = RELEASE_RE.search(line)
        if m:
            task_id = int(m.group(1))
            n_tokens_final = int(m.group(2))
            truncated = bool(int(m.group(3)))
            rec = pending.pop(task_id, None)
            if not rec:
                # embeddings/rerank/errored request -- no timing lines to report
                continue
            cached = n_tokens_final - rec.get("prompt_tokens_new", 0) - rec.get("predicted_tokens", 0)
            out.write(json.dumps({
                "ts": docker_ts,
                "task_id": task_id,
                "tier": tier,
                "prompt_tokens_new": rec.get("prompt_tokens_new"),
                "prompt_tok_s": rec.get("prompt_tok_s"),
                "predicted_tokens": rec.get("predicted_tokens"),
                "predicted_tok_s": rec.get("predicted_tok_s"),
                "n_tokens_final": n_tokens_final,
                "cached_tokens": max(cached, 0),
                "truncated": truncated,
            }) + "\n")
            continue
