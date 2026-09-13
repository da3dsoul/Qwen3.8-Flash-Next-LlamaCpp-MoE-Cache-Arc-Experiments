#!/usr/bin/env python3
"""116K-token real-task long-context benchmark against llama-server.

Methodology mirrors the sibling project's kvarn-120k-benchmark-report.md:
 - real messy-Python refactor prompt, not synthetic filler
 - temperature 0
 - headline numbers come from llama-server's OWN `timings` block
   (prompt_ms / prompt_per_second / predicted_ms / predicted_per_second),
   not client wall-clock; client wall-clock is recorded only as a cross-check
 - full output + reasoning trace + finish_reason + token counts saved to disk
 - /slots + /metrics captured before and after for a pre/post diff
"""
import json
import sys
import time
import urllib.request

ADDR = sys.argv[1] if len(sys.argv) > 1 else "localhost:8090"
PROMPT_FILE = sys.argv[2]
OUT_PREFIX = sys.argv[3]
MAX_TOKENS = int(sys.argv[4]) if len(sys.argv) > 4 else 30000


def get(path):
    try:
        with urllib.request.urlopen(f"http://{ADDR}{path}", timeout=30) as r:
            return r.read().decode()
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"


prompt = open(PROMPT_FILE).read()
print(f"prompt chars: {len(prompt)}", flush=True)

open(f"{OUT_PREFIX}.slots.pre.json", "w").write(get("/slots"))
open(f"{OUT_PREFIX}.metrics.pre.txt", "w").write(get("/metrics"))

payload = {
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0,
    "max_tokens": MAX_TOKENS,
    "stream": True,
    "cache_prompt": False,
    "stream_options": {"include_usage": True},
}

req = urllib.request.Request(
    f"http://{ADDR}/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)

content_parts = []
reasoning_parts = []
finish_reason = None
timings = None
usage = None
chunk_times = []

t0 = time.perf_counter()
ttft = None
n_chunks = 0

with urllib.request.urlopen(req, timeout=36000) as resp:
    raw = open(f"{OUT_PREFIX}.stream.jsonl", "w")
    for line in resp:
        line = line.decode("utf-8").strip()
        if not line or not line.startswith("data: "):
            continue
        body = line[6:]
        if body == "[DONE]":
            break
        raw.write(body + "\n")
        try:
            d = json.loads(body)
        except json.JSONDecodeError:
            continue
        now = time.perf_counter()
        for ch in d.get("choices", []) or []:
            delta = ch.get("delta") or {}
            c = delta.get("content")
            r = delta.get("reasoning_content")
            if c:
                if ttft is None:
                    ttft = now - t0
                content_parts.append(c)
            if r:
                if ttft is None:
                    ttft = now - t0
                reasoning_parts.append(r)
            if ch.get("finish_reason"):
                finish_reason = ch["finish_reason"]
        if (c or r):
            n_chunks += 1
            chunk_times.append(now - t0)
            if n_chunks % 250 == 0:
                print(
                    f"  [{now - t0:8.1f}s] {n_chunks} chunks "
                    f"({len(''.join(reasoning_parts))} reasoning chars, "
                    f"{len(''.join(content_parts))} content chars)",
                    flush=True,
                )
        if d.get("timings"):
            timings = d["timings"]
        if d.get("usage"):
            usage = d["usage"]
    raw.close()

wall = time.perf_counter() - t0

open(f"{OUT_PREFIX}.slots.post.json", "w").write(get("/slots"))
open(f"{OUT_PREFIX}.metrics.post.txt", "w").write(get("/metrics"))

content = "".join(content_parts)
reasoning = "".join(reasoning_parts)
open(f"{OUT_PREFIX}.content.txt", "w").write(content)
open(f"{OUT_PREFIX}.reasoning.txt", "w").write(reasoning)

result = {
    "server_timings": timings,
    "usage": usage,
    "finish_reason": finish_reason,
    "client_wall_s": wall,
    "client_ttft_s": ttft,
    "content_chars": len(content),
    "reasoning_chars": len(reasoning),
    "n_stream_chunks": n_chunks,
    "max_tokens_requested": MAX_TOKENS,
}
open(f"{OUT_PREFIX}.result.json", "w").write(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
