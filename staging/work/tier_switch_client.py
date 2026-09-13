#!/usr/bin/env python3
"""Multi-turn client for the --context-tiers validation.

Drives a real growing conversation against /v1/chat/completions, which is the
shape the tier policy is designed for: the client resends the whole
conversation every turn, so the server knows the exact depth of the turn
before it decodes anything.

Per turn it records the server's OWN timings block (prompt_ms /
prompt_per_second / predicted_per_second), the token counts, and the reply, so
a tier switch shows up as (a) a gap in wall clock between the request and the
first token and (b) a full re-prefill instead of a cached prefix.

Usage: tier_switch_client.py ADDR OUT_PREFIX FILLER_FILE
"""
import json
import os
import sys
import time
import urllib.request

ADDR = sys.argv[1]
OUT_PREFIX = sys.argv[2]
FILLER_FILE = sys.argv[3]

filler = open(FILLER_FILE).read()

# turns: (label, user message, max_tokens)
# the growing conversation: two short turns, one that crosses the tier
# boundary, one more deep turn, then short turns again
SHORT_Q = [
    "In one short sentence: what is the capital of France?",
    "And in one short sentence: what is the capital of Japan?",
    "One short sentence: name a primary color.",
    "One short sentence: what is 2 + 2?",
]

# reset = start a fresh conversation, which is the only way a growing chat ever
# gets shorter again (a client resends the whole history every turn)
TURNS = [
    ("t1-short", SHORT_Q[0], 48, False),
    ("t2-short", SHORT_Q[1], 48, False),
    ("t3-cross", "Here is a Python module.\n\n" + filler +
        "\n\nIn at most three sentences, say what this module is mostly made of.", 128, False),
    ("t4-deep", "In one sentence, roughly how many similar function definitions did you see?", 64, False),
    ("t5-new-chat", SHORT_Q[2], 48, True),
    ("t6-new-chat", SHORT_Q[3], 48, True),
]

# PROFILE=big: only the turns that matter at production depth, the cheap ones
# are covered by the small-tier arm
if os.environ.get("PROFILE") == "big":
    TURNS = [t for t in TURNS if t[0] in ("t1-short", "t3-cross", "t4-deep")]

messages = []
log = []

for label, content, max_tokens, reset in TURNS:
    if reset:
        messages = []
    messages.append({"role": "user", "content": content})

    payload = {
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
        "cache_prompt": True,
    }

    req = urllib.request.Request(
        f"http://{ADDR}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )

    print(f"--- {label}: sending {len(content)} chars, history {len(messages)} messages", flush=True)
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=7200) as r:
            body = json.loads(r.read().decode())
    except Exception as e:  # noqa: BLE001
        print(f"{label}: REQUEST FAILED: {e}", flush=True)
        log.append({"turn": label, "error": str(e)})
        break
    wall = time.perf_counter() - t0

    if "error" in body:
        print(f"{label}: SERVER ERROR: {body['error']}", flush=True)
        log.append({"turn": label, "server_error": body["error"], "wall_s": wall})
        break

    choice = body["choices"][0]
    reply = choice["message"].get("content") or ""
    timings = body.get("timings", {})
    usage = body.get("usage", {})

    messages.append({"role": "assistant", "content": reply})

    entry = {
        "turn": label,
        "wall_s": round(wall, 2),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "prompt_n": timings.get("prompt_n"),
        "prompt_ms": timings.get("prompt_ms"),
        "prompt_per_second": timings.get("prompt_per_second"),
        "predicted_per_second": timings.get("predicted_per_second"),
        "finish_reason": choice.get("finish_reason"),
        "reply": reply,
    }
    log.append(entry)
    print(json.dumps({k: v for k, v in entry.items() if k != "reply"}), flush=True)
    print(f"    reply: {reply[:200]!r}", flush=True)

with open(f"{OUT_PREFIX}.turns.json", "w") as f:
    json.dump(log, f, indent=2)

print("done", flush=True)
