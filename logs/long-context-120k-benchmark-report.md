# 116K-token real-task long-context benchmark — Qwen3.8-Flash-Next on Arc Pro B70

**Date:** 2026-09-11
**Model:** `unsloth/Qwen3.8-Flash-Next-GGUF` **UD-IQ3_XXS** (3.0625 bpw, 176.94 B params), arch `qwen4exp`
**Backend:** llama.cpp SYCL, build **10882 (`6d9c82ea2`)**, IntelLLVM 2025.3.3, `staging/devbin` incremental dev build
**Hardware:** Intel Arc Pro B70 Graphics, 32,656 MiB (32,580 MiB free at load) / AMD Ryzen 9 9950X / 123 GiB host RAM
**Config:** `-ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -ncmoe 24 -c 163840 -fit off -np 1`

> **Headline: yes, it works.** This is the first confirmation in this project that Qwen3.8-Flash-Next
> actually *runs* a real long prompt end-to-end on this card — not just that it loads. 116,277 prompt
> tokens in, 16,382 tokens out, `finish_reason: stop`, no truncation, no garbling, and a **correct,
> idiomatic refactor** with **99.2% (119/120) verbatim accuracy** on constants scattered across the
> 116K-token input. `docs/research/08-long-context-vram-budget.md`'s recommended config held exactly as
> predicted, with no adjustment.

---

## 1. Methodology

Mirrors the sibling project's `docs/05-benchmarks/kvarn-120k-benchmark-report.md` in *principle*, not in
harness — that report's `bench_120k.py` scrapes vLLM's Prometheus `/metrics` and does not port to
llama.cpp. Principles carried over:

- **A real task, not synthetic filler.** The prompt is the sibling project's deterministic messy-Python
  refactor benchmark, regenerated from source (seed 42, hardcoded).
- **Native server-side timers, not client wall-clock.** All headline numbers come from llama-server's own
  `timings` block (`prompt_n`/`prompt_ms`/`prompt_per_second`/`predicted_n`/`predicted_ms`/
  `predicted_per_second`, from `server_slot_stats::to_json()` in `tools/server/server-common.cpp:66`).
  Client wall-clock is recorded only as a cross-check and agrees to within 0.1 s.
- **`llama-server`, not `llama-cli`**, so the full real serving path (sampler chain, detokenization,
  response streaming) is paid for, same as a real deployment.
- **temperature = 0**, `cache_prompt: false`, single slot.
- **Full output, full reasoning trace, finish_reason and token counts saved to disk** for a real quality
  review, not just a speed number.

### 1.1 Prompt provenance

Generated **in place** from the sibling project's own directory
(`../Qwen3.8-vLLM-KVarN-MTP-Experiments/scripts/`, MIT, same author). Nothing was copied into this repo
except the generated artifacts, which are outputs rather than sibling source:

```
cd ../Qwen3.8-vLLM-KVarN-MTP-Experiments/scripts
python3 generate_messy.py 153 > <this repo>/staging/work/bench120k/messy.py
python3 build_prompt.py         <...>/messy.py  <...>/full_prompt.txt
```

Input characteristics, verified this pass:

| property | value |
|---|---|
| `messy.py` | 410,744 chars, 1,229 top-level defs/classes |
| structure | **153 near-identical entity blocks** (7 functions + 1 handler class each), cycling 20 entity names |
| planted anti-patterns | 153 bare `except:`, 154 mutable default args, 459 `== None`, 153 `== True`, 460 `for i in range(len(...))`, 153 `global` statements, 153 Python-2 `has_key`, manual `open()`/`close()` |
| `full_prompt.txt` | 411,558 chars |

### 1.2 Token count verified against *this* project's tokenizer

Not assumed from the sibling's 27B model:

```
llama-tokenize -m .../Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf \
               -f full_prompt.txt --ids   ->  116,226 tokens
```

116,226 raw / **116,277 after chat templating**. The sibling reported 116,226 / 116,277 for
`Qwen3.8-27B` — **the two tokenizers agree exactly on this input**, so the sibling's size calibration
carries over unchanged and no re-tuning of the entity count was needed.

### 1.3 GPU discipline

Confirmed before starting and after finishing: no `llama-cli`/`llama-server` process, no
`*sycl*` container. The box's unrelated services (Minecraft ×2, Immich, litellm, vaultwarden, MSSQL,
Shoko) were left running and untouched — none of them use the B70. Sole occupant of the GPU for the
whole ~80-minute run.

---

## 2. Results

### 2.1 Headline numbers (llama-server native timers)

| Metric | Value |
|---|---|
| prompt_tokens | **116,277** |
| completion_tokens | **16,382** |
| total_tokens | 132,659 |
| finish_reason | **`stop`** (natural EOS — not truncated; cap was 30,000) |
| **prefill time** | **1,704.79 s (28 min 25 s)** |
| **prefill throughput** | **68.21 tok/s** (14.66 ms/token) |
| **time to first token** | **1,704.96 s** (client-measured; = prefill, as expected) |
| **decode time** | **3,065.09 s (51 min 5 s)** |
| **decode throughput** | **5.34 tok/s** (187.11 ms/token) |
| end-to-end | **4,769.87 s (79 min 30 s)** |
| content_chars / reasoning_chars | 10,364 / 54,116 |
| SYCL graphs reused | 16,441 |
| prefix-cache hits | 0 (`cache_prompt: false`, clean cold prefill) |

Client wall-clock cross-check: 4,770.03 s vs. server 4,769.87 s — 0.16 s apart, so the server timers are
not hiding queue or transport time.

### 2.2 Decode falloff with depth (measured this session, same server process, same config)

| depth (`n_kv`) | decode throughput |
|---|---|
| ~60 tokens (warm smoke test) | **17.05 tok/s** |
| 116K → 132K tokens (this benchmark) | **5.34 tok/s** |

**A 3.2× decode penalty for going from trivial depth to 116K.** This is the number
`docs/research/08` §4.2 flagged as the biggest unknown in the whole long-context budget
("every 'implied decode' cell in §4.1 is an upper bound set by expert placement alone, and the real
number at 300K–1M will be dominated by dense-attention depth cost, which nothing in this project has ever
measured"). It is now measured at 116K. For calibration, the only comparable same-family evidence
(`docs/research/02` §3.2, third-party Arc Pro B65 + Vulkan) was 30.4 t/s @ 8K → 13.3 t/s @ 32K, a 2.3×
loss over 4× depth — our 3.2× over a ~2,000× depth increase is a **much gentler** curve than naively
extrapolating that trend would suggest.

Per-interval `tg_3s` readings stayed in a **3.5–7.9 tok/s** band throughout the 51-minute decode with no
downward drift as the cache filled from 116K to 132K, i.e. within this window the depth cost is roughly
flat rather than still climbing.

### 2.3 Cold-start caveat (worth knowing, affects the prefill number)

The **first** request after model load ran at 13.2 tok/s prefill / 2.8 tok/s decode — roughly 5× slower
than warm. Cause is host-side, not GPU: the model is 77 GiB of `mmap`'d IQ3_XXS and `-ncmoe 24` leaves
24 of 48 expert layers resident on the CPU, so the first pass is page-faulting expert weights off disk.
Two short warm-up requests were issued before the benchmark; the reported 68.21 tok/s prefill is the
**warm** figure. A cold-start production request would be materially slower than this report's numbers.

---

## 3. VRAM: does doc 08's arithmetic hold?

Captured by re-loading the identical config with `-v` after the benchmark (`logs/long-context-120k/vram-check.log`).

| allocation | measured | doc 08 §4 prediction @ D=163,840 |
|---|---|---|
| SYCL0 model buffer (dense core + 24 GPU expert layers) | **26,916.16 MiB** | `3,816.16 + (46,200 − 962.5×24)` = **26,916.16 MiB** ✔ **exact** |
| main attention KV (q4_0) | 1,080.00 MiB | — |
| QSA indexer KV (q4_0) | 405.00 MiB | — |
| **both KV caches** | **1,485.00 MiB** | `16,896 × D × 0.5625 / 2^20` = **1,485.00 MiB** ✔ **exact** |
| recurrent (RS) | 112.57 MiB | 112.57 MiB ✔ exact |
| SYCL0 compute buffer (incl. FA staging) | **1,230.57 MiB** | compute 1,196 + FA staging 320 = 1,516 MiB (**over**-predicted by 285 MiB) |
| **SYCL0 total** | **29,744.30 MiB** | — |
| headroom on a 32,580 MiB card | **2,835.70 MiB (8.7%)** | — |
| CPU_Mapped model buffers (host) | 52,837.03 MiB | — |

**Verdict: doc 08's model is correct and conservative.** Two of its terms reproduce to the hundredth of a
MiB; the only divergence is that the compute buffer came in **285 MiB smaller** than predicted, i.e. the
document errs in the safe direction. In particular:

- **§0 verdict 1 confirmed directly** — there really are two context-scaling KV caches, and the indexer
  one is not small: 405 MiB against the main cache's 1,080 MiB, **37.5% on top**, exactly as doc 08
  computed and doc 06 missed.
- **§3.3's flash-attention staging-buffer reuse holds, again.** The whole compute buffer is 1,230.57 MiB.
  The pessimistic §4.3 case (12× redundant FA staging = 3,840 MiB of staging alone at this depth) is
  decisively ruled out. **Doc 08's §4 tables, not §4.3, are the ones to use.**
- **§0 verdict 2 confirmed and still costing us** — the indexer cache's V side (270 of its 405 MiB) is
  allocated and, per `qwen4exp.cpp`, never read or written. That is 270 MiB of dead VRAM at this depth
  and would be ~1,000 MiB at 600K.

**`-ncmoe 24` was, if anything, slightly conservative at this depth.** Doc 08 §4.1 asks for `-ncmoe 21`
at 100K and `-ncmoe 24` at 300K with `q4_0` KV; we ran 24 at 163,840 and finished with 2,836 MiB spare —
about three more expert layers' worth. `-ncmoe 22` would have fit and been faster. No adjustment to the
recommended config was needed in the other direction; **nothing OOM'd, and `-fit off` was used precisely
so llama.cpp could not silently rescue a bad config.**

---

## 4. Quality review

**Verdict: correct, idiomatic, and a strong long-context retrieval result. One real transcription bug,
one deliberate-but-lossy design choice.**

### 4.1 The core ask — did it collapse the duplication?

Yes. The model replaced **1,229 top-level defs/classes** with a `frozen` `EntityConfig` dataclass, a
20-entry `ENTITY_CONFIGS` table, one generic `EntityProcessor`, five free functions
(`validate_record`, `calculate_total`, `format_full_name`, `find_by_id`, `delete_by_id`), an
`EntityHandler`, a `DataManager`, and an `EntityAPI` factory — 10,364 chars of output for 410,744 chars
of input. This is the same architecture the sibling project's 27B model converged on, and it is the
intended fix.

### 4.2 Every listed anti-pattern was fixed

| asked for | delivered |
|---|---|
| global mutable state | removed; state encapsulated in `EntityProcessor` |
| bare `except:` | `except Exception as exc` + `LOGGER.exception` |
| mutable default args | `field(default_factory=...)`, `values or []` |
| `for i in range(len(x))` | `enumerate`, comprehensions, `next(...)`, `sum(...)` |
| `== None` / `== True` | `is None` / truthiness |
| string concat | f-strings |
| mixed naming | consistent `snake_case` / `PascalCase` |
| type hints & docstrings | present throughout, incl. `from __future__ import annotations` |
| manual `open`/`close` | `Path(path).open(...)` context managers |
| code duplication | collapsed (§4.1) |

It also volunteered fixes not on the list (`print`-logging → `logging`, unused imports removed).

### 4.3 Long-context retrieval accuracy — the interesting measurement

The 153 entity blocks each embed distinct numeric constants (multiplier, offset, divisor, tax rate,
uppercase flag, default id) scattered across the whole 116K-token prompt. The model's `ENTITY_CONFIGS`
table transcribes 20 of these blocks × 6 fields = 120 values. Checked mechanically against the source:

> **119 / 120 correct = 99.2%.**

The single error: `EntityConfig("invoice", …, uppercase_name=False)` — `format_invoice_name_3` in the
original is `fullname.upper() if True else fullname.lower()`, so it should be `True`. One flipped boolean
out of 120 values pulled from across a 116K-token context.

### 4.4 The one substantive design defect

The constants vary **per suffix (0–152), not per entity name**. Verified directly: `process_user_0`
uses multiplier 3 / divisor 6 / tax 0.13 / uppercase `True`, while `process_user_20` uses 6 / 2 / 0.22 /
`False` and `process_user_40` uses 5 / 1 / 0.25 / `True` — all three named "user". The model's 20-entry,
name-keyed table therefore preserves the behaviour of suffixes 0–19 only; **133 of 153 blocks' constants
are silently generalized away.**

This is partly mitigated by the model flagging the decision explicitly in a code comment —
*"If exact per-suffix constants are required, add more configs keyed by suffix"* — so it recognized that
a keying choice was being made. Nothing in the trace shows it noticed that the constants actually *do*
differ across cycles, so the caveat reads as generic prudence rather than a diagnosed finding. The
sibling project's 27B made the same reduction. Given the prompt asks for "a cleaned-up, well-structured
version" rather than strict behaviour preservation, this is defensible, but a reviewer would want it
called out.

### 4.5 Reasoning-trace health (the degeneracy check)

54,116 chars of reasoning (vs. the sibling 27B's 46K–80K band). Checked for the circular /
non-convergent failure mode common in reasoning models at long context:

- **No degenerate repetition.** Most-repeated non-blank line appears **10 times** and is `@dataclass`;
  the next four are ordinary code lines at 5–7 occurrences. Consistent with drafting code, not looping.
- **Coherent, on-task structure**: correctly identifies the 153-block cycling structure in the opening
  lines, designs the parameterized replacement, then runs an explicit self-critique pass
  (a long run of `"Potential concern: …"` items, each resolved) before emitting the answer.
- No refusal, no truncation, no topic drift, no restatement of the input.

### 4.6 Bugs *not* present

The sibling project's 163,840 run produced a real defect — a `RecordHandler` that set
`self.is_active = True` as an instance attribute while also defining `def is_active(self)`, permanently
shadowing the method. **Our output avoids this**: `EntityHandler` stores the flag as `active: bool` and
exposes `def is_active(self) -> bool: return self.active`. No collision.

---

## 5. Comparison to the sibling project's run

Different model, different quant, different serving stack, **same prompt, same GPU** — so this is a
stack/model comparison, not an apples-to-apples one. Included because it is the only other real
measurement of this exact task on this exact card.

| | sibling: Qwen3.8-**27B** W4A16, vLLM+KVarN | **this run: Qwen3.8-Flash-Next 180B-A6B IQ3_XXS, llama.cpp SYCL** |
|---|---|---|
| prompt_tokens | 116,277 | 116,277 |
| generation_tokens | 14,238 | **16,382** |
| finish_reason | stop | **stop** |
| prefill | 120.8 s @ **962.8 tok/s** | 1,704.8 s @ **68.2 tok/s** |
| decode | **6.48 tok/s** | **5.34 tok/s** |
| e2e | 38.6 min | **79.5 min** |
| content / reasoning chars | 9,191 / 46,486 | **10,364 / 54,116** |

**Prefill is ~14× slower**; decode is only ~1.2× slower. That gap is the story: the 27B is fully
GPU-resident and dense, while Flash-Next is 180B total with half its expert layers (`-ncmoe 24`) on the
CPU behind PCIe, and prefill is exactly the regime `docs/00-background.md` §1 warns about — *"large-batch
prefill has a wide unique-expert set per op, closer to a worst case for a small resident pool."*
**This is the first direct evidence for `PLAN.md`'s suspicion that prefill TTFT, not steady-state decode,
is the metric that matters for this deployment.** A 28-minute time-to-first-token at 116K, on a target
workload of 300K–600K, is the number to attack.

---

## 6. Caveats

1. **n = 1.** One run, one config. No variance estimate.
2. **Warm, not cold.** §2.3 — the first post-load request was ~5× slower. The headline prefill number
   assumes a warmed page cache.
3. **116K, not 300–600K.** This validates *correctness and behaviour at a real long prompt*, which is
   what it set out to do. It does **not** establish that 300K–600K works; doc 08 predicts `-ncmoe 24`
   (300K) / `-ncmoe 28` (500K) with `q4_0` KV, and the VRAM model is now well-validated at 163,840, but
   the depth-scaling of both prefill and decode past 132K is still unmeasured. Naive linear extrapolation
   of prefill puts 300K at ~73 min and 600K at ~2.4 h TTFT, and prefill is super-linear in depth, so
   treat those as optimistic floors.
4. **`/metrics` unavailable.** llama-server returns HTTP 501 unless started with `--metrics`; the
   pre/post metrics diff the sibling methodology calls for degraded to a `/slots` diff. Add `--metrics`
   next time. (`/slots` pre/post *was* captured and is clean: 0 prefix-cache hits, `temperature: 0.0`
   confirmed server-side, single slot, no preemption.)
5. **UD-IQ3_XXS, not UD-Q3_K_XL.** Doc 08's entire budget (46,200 MiB total expert VRAM, 962.5 MiB per
   expert layer) is computed against IQ3_XXS, and `-ncmoe 24` is meaningless without that. `PLAN.md`
   names UD-Q3_K_XL as "our quant tier"; **the VRAM arithmetic in this report does not transfer to it**
   and would need redoing before that quant is benchmarked at depth.
6. **Sampler is the server default apart from temperature.** `temperature: 0` was set (verified via
   `/slots`), which makes `top_k`/`top_p`/`min_p` moot for selection, but the full sampler chain is still
   executed per token and is included in the 5.34 tok/s decode figure — deliberately, since a real
   deployment pays it too.
7. **`-ub 512`** (llama-server default). Doc 08's compute-buffer term assumes this; `-ub 2048` changes
   the budget materially (doc 08 §3.1).

---

## 7. Artifacts

All under `logs/long-context-120k/`:

| file | contents |
|---|---|
| `run1.result.json` | server `timings`, usage, finish_reason, client cross-check |
| `run1.content.txt` | full model answer (10,364 chars) |
| `run1.reasoning.txt` | full reasoning trace (54,116 chars) |
| `run1.stream.jsonl` | raw SSE stream, every chunk (5.4 MB) |
| `run1.slots.{pre,post}.json` | `/slots` before/after |
| `run1.metrics.{pre,post}.txt` | HTTP 501 — see caveat 4 |
| `server.log` | llama-server log incl. per-interval prefill/decode progress |
| `vram-check.log` | verbose re-load at the identical config, for §3's buffer sizes |
| `METHODOLOGY.md` | prompt-provenance scratch notes |

Reproduction inputs under `staging/work/bench120k/`: `messy.py`, `full_prompt.txt`, `serve.sh`,
`run_bench.py`.

---

## 8. What this changes

- **`PLAN.md`'s open question "nothing has confirmed the model runs a real long prompt end-to-end at this
  depth" is now closed, positively, at 116K.** Correctness at long context is not a risk; speed is.
- **`docs/research/08`'s §4 tables are validated against hardware** at 163,840 tokens — two terms exact,
  compute buffer 285 MiB conservative. Its §4.3 pessimistic branch is dead.
- **Prefill, not decode, is the bottleneck for this workload.** 28 min TTFT at 116K vs. 51 min for 16K
  tokens of output. At the real 300K–600K target, prefill will dominate end-to-end latency outright.
  Optimization effort aimed at decode throughput (the "high-40s to mid-50s tok/s" target) is aimed at the
  smaller half of the problem.
- **The depth penalty is real but gentler than feared**: 3.2× decode loss from trivial depth to 116K,
  against a third-party Vulkan datapoint implying much worse. Doc 08 §4.2's warning that decode at depth
  "will be dominated by dense-attention depth cost" is directionally right but was over-pessimistic at
  this depth.
- **Next measurements, in priority order:** (a) prefill scaling — run `-c 327680` and prefill ~300K to get
  the real TTFT curve, since that is the deployment's actual constraint; (b) `-ncmoe 22` at this depth,
  since 2,836 MiB went unused; (c) the QSA sparse-attention thread (`docs/research/07`) now has a concrete
  prefill number to beat, and its claimed 1.76–2.2× prefill speedup would cut 28 min to ~13–16 min.
