# 16 — Long-context prefill (TTFT): where the time goes now, and what to do about it

**Status: research only. No code was written, no benchmark was run, the production server was not touched and
the GPU was not used.** Everything below comes from reading the tree, re-analysing logs already committed to
this repo, `gh` queries against upstream, and a small amount of web search. Date: 2026-09-27.

Triggered by the brief: prefill, not decode, is this deployment's binding metric at 116K-600K tokens
(`PLAN.md:109-119`), and the question was what is consuming the marginal prefill time as depth grows, and
which fixes are worth their cost.

Sources actually read for this document:

| Source | How obtained |
|---|---|
| `PLAN.md` (CRITICAL block `:9-133`, every 2026-09-11 → 2026-09-19 update) | direct read |
| `docs/research/02` §2.3, §3.0, §4.4; `05` §3.2, §10.4-10.5; `07` §3.3-4.1; `09` (all); `10` §4.3 + addendum; `11` §7 via `PLAN.md`; `12` §9.4; `13` §3.3, §7, §12 | direct read |
| `logs/long-context-120k-benchmark-report.md`, `logs/long-context-300k-benchmark-report.md` §4.2 | direct read |
| Every `print_timing` progress line in `logs/long-context-120k*/server.log`, `logs/long-context-300k*/server.log`, `logs/tier3-prod/server.log` | re-fitted, script below (§1.2) |
| `logs/prod-metrics/requests.ndjson` (127 production requests, 2026-09-13) | re-analysed (§4.1) |
| `logs/topk-ab-{before,after}.log` (the `GGML_SYCL_OP_PROFILE=2` 32K prefill profile) | re-aggregated (§2.4) |
| Vendored tree `src/llama.cpp/` (= pinned `6d9c82ea2` + `patches/llama.cpp.patch`): `ggml-sycl.cpp`, `fattn.cpp`, `fattn-onednn.cpp`, `lightning-indexer.cpp`, `ggml-backend.cpp`, `llama-kv-cache.cpp`, `llama-model{,-loader}.cpp`, `src/models/qwen4exp.cpp`, `tools/server/server-context.cpp`, `common/{arg.cpp,common.h}` | direct reads, lines cited |
| Upstream PRs #28501, #28670, #28770, #28796, #29245, #28985, #28414, #28092, #29030, #29298, #25222 | `gh pr view` / `gh pr diff` |
| `/sys/bus/pci/devices/*/{current,max}_link_{speed,width}`, DMI board name | read-only sysfs |
| B70 spec sheet (INT8 TOPS), FlashMLA sparse-prefill README | web search |

Conventions as in docs 09-15: `file:LINE` means read directly; **inference** marks a connection drawn between
separately established facts; **estimate** marks arithmetic, with its assumptions shown. Paths are relative to
`src/llama.cpp/` unless they start with `logs/`, `docs/`, `staging/` or `PLAN.md`.

---

## 0. Verdict in one paragraph

**Two premises in the brief are out of date, and correcting them reorders the priorities.** (a) The radix
top-k is not a prerequisite still to do: #28670 was vendored on 2026-09-11 (`PLAN.md:134-209`) and merged
upstream on 2026-09-14; a 32K prefill profile puts `TOP_K` at 0.06% (`docs/research/12` §9.4). (b) The
"1.84x marginal-throughput decay from 22K to 141K" in `logs/long-context-300k-benchmark-report.md` §4.2 was
measured on a binary that predates block-granularity top-k and the pooled indexer-key cache. **Re-fitted from
every committed server log, the depth slope has since fallen by about 35%** (0.0104-0.0119 → 0.0073-0.0083
ms per token per 1,000 tokens of depth, §1.2), so on today's code per-token cost grows ~1.4x over that span,
not 1.84x. **The consequence that matters: prefill time is now dominated by a depth-*independent* per-ubatch
cost, not by the depth slope.** At 116K the slope is ~18% of TTFT, at 300K (the 512K tier, `-ub 1024`)
~22%, and it only approaches half of TTFT near 600K (§1.3). The largest items, in order of evidence × cost:
**(1) conversations the server has already prefilled are being prefilled again after every tier switch and
every sleep/wake reload** — in the one day of production metrics committed here, that was **~56% of all
prefilled tokens** (§4.1), and fixing it is a server-only change of tens of lines; **(2) production runs
the CPU-resident experts from pageable mmap memory**, and the only clean A/B in this repo measured the
pinned-host arrangement at **1.56x faster prefill** at 116K (394 vs 252 tok/s, `PLAN.md:681-696`) — a small
loader change could get that back without pinning the 28.8 GB PLE table (§4.2); **(3) upstream #29245 — a
grouped XMX MoE GEMM for SYCL, by the author of the radix top-k, measured +34% prompt throughput on this
exact model** — attacks the per-expert launch loop that every prefill ubatch pays 73,728 times (§2.2, §4.3).
Sparse *prefill* attention remains the structurally correct fix for the slope but is the most expensive item
and, on today's numbers, worth ≤10-20% at the depths this deployment actually serves (§4.6). **The single
most important open unknown is how the ~2 ms/token depth-independent intercept splits between PCIe
streaming of the CPU-resident experts and the SYCL per-expert `MUL_MAT_ID` loop** (§6.1): it decides whether
the transfer-side levers or the compute-side lever is the bigger one, and no measurement in this repo can
separate them (§2.4 explains why the one profile that looks like it does, doesn't).

---

## 1. What prefill actually costs today, re-derived from committed logs

### 1.1 The model: a flat intercept plus a linear depth slope

`llama-server` prints a progress line per ubatch (`slot print_timing: ... n_tokens = N ... t = T s`). The
difference between consecutive lines is the marginal cost of one ubatch at a known depth. Fitting per-token
marginal cost against depth gives two numbers per run: an **intercept** (ms/token at zero depth — every cost
that does not grow with `n_kv`) and a **slope** (ms/token added per 1,000 tokens of depth — every cost that
does). The fit script is 30 lines of Python over the committed logs (kept in the scratchpad, not the repo;
it reads only the `print_timing` lines, fits depth ≥ 8K, one fit per request).

### 1.2 Results, every long run in `logs/`

| run (log dir) | binary era | config (relevant) | intercept ms/tok | slope ms/tok per 1K | tokens |
|---|---|---|---:|---:|---:|
| `long-context-120k-lazyoff` | 09-11, per-cell top-k, no pool cache | `-ub 2048 -ncmoe 24 -lm none -lzm off` | 1.83 | **0.0119** | 116,273 |
| `long-context-300kcfg-noyarn` | 09-11, same era | `-ub 2048 -ncmoe 32 -lm none -lzm off` | 1.95 | **0.0119** | 116,273 |
| `long-context-300k` | 09-11, same era | `-ub 2048 -ncmoe 32 -lm none -lzm off` | 2.33 | **0.0104** | 305,797 |
| `long-context-120k-auto-lazyoff` | 09-11, same era | `-ub 2048 -ncmoe 24` **`-lm auto`** `-lzm off` | 2.13 | 0.0118 | 116,273 |
| `long-context-120k-sparse` | 09-12, block top-k + pool cache | `-ub 2048 -ncmoe 24 -lm none -lzm off` | 1.94 | **0.0075** | 116,273 |
| `long-context-120k-ncmoe25` | 09-13 | `-ub 2048 -ncmoe 25 -lm none -lzm off` | 1.96 | **0.0073** | 116,273 |
| `long-context-120k-garudias` | 09-14 | (storage move) | 2.16 | 0.0083 | 116,273 |
| `tier3-prod`, tier 1 | 09-13, mmap | `-ub 2048 -ncmoe 30`, mmap | 2.48 | 0.0077 | 149,545 |
| `tier3-prod`, tier 2 | 09-13, mmap | **`-ub 1024`** `-ncmoe 32`, mmap | **4.26** | 0.0082 | 303,843 |

(The earlier `-ub 512` runs — `long-context-120k`, `-after` — and the `-lzm auto` run `-ub2048` are omitted:
their intercepts are 4-10 ms/token and dominated by effects that have since been fixed, top-k and PLE paging.)

Three readings:

1. **The slope dropped ~35% between 09-11 and 09-12** (0.0119 → 0.0073-0.0075), which is when block-granularity
   top-k and the pooled indexer-key cache landed (`PLAN.md:924-1034`, `:1101-1192`). Doc 12 §9.3 had measured
   the block top-k at 1.46x/2.31x/2.52x on the chain at *prefill* shape and could not resolve it end to end
   (`PLAN.md:1027-1034`); this is that effect, resolved as a slope rather than as a single noisy ratio.
   **Inference** — the two changes landed together, so their shares are not separable from these logs.
2. **The 300K report's "1.84x from 22K to 141K" is a pre-fix number.** With today's slope, marginal cost from
   22K to 141K rises from ~2.12 to ~2.99 ms/token, **~1.41x**.
3. **The slope barely moves with `-ub` or `-ncmoe`; the intercept moves a lot.** Tier 2's `-ub 1024` doubles
   the intercept (4.26 vs ~2.0-2.5) at an almost identical slope — as expected if the intercept is a fixed
   per-ubatch cost amortised over `n_ubatch` tokens and the slope is per-(query, key) work.

### 1.3 How much of TTFT each term is, at the depths this deployment serves

**Estimate**, a cold prompt of `N` tokens has mean depth `N/2`, so TTFT ≈ `N × (intercept + slope × N/2000)`:

| prompt | tier (per `README.md` table) | intercept | slope term (mean) | TTFT est. | slope share | measured |
|---|---|---:|---:|---:|---:|---|
| 116K | 0, `-ub 2048 -ncmoe 25`, `-lm none` | 1.96 | 0.42 | 4.6 min | **18%** | 4 min 38.6 s (`PLAN.md:2052-2056`) |
| 150K | 1, `-ub 2048 -ncmoe 30`, mmap | 2.48 | 0.58 | 7.6 min | 19% | 8 min 53 s (`logs/tier3-prod/server.log:120`) |
| 300K | 2, `-ub 1024 -ncmoe 32`, mmap | 4.26 | 1.23 | 27.5 min | **22%** | 28 min 14 s (`logs/tier3-prod/server.log:323`) |
| 600K | 2 | 4.26 | 2.46 | 67 min | 37% | 73 min 51 s at `-ncmoe 38` (`PLAN.md:1716-1719`) |

The measured column is there as a sanity check on the arithmetic, not as a new result. **At every depth up to
~300K, at least three quarters of TTFT is the depth-independent intercept.** Anything that only attacks the
slope (sparse prefill attention, the KQ-mask build, the indexer) is capped at the slope share above.

---

## 2. What is inside the intercept

### 2.1 CPU-resident experts are streamed to the GPU every ubatch

This part was established by `docs/research/05` §3.2 and is re-verified here because it is load-bearing:

- `-ncmoe N` overrides expert tensors to `ggml_backend_cpu_buffer_type()` (`common/common.h:1157-1165`).
- At prefill the scheduler offloads the `MUL_MAT_ID` anyway, because `ggml_backend_sycl_device_offload_op`
  returns `get_op_batch_size(op) >= 32` and `get_op_batch_size` is `ne[2]` (tokens) for `MUL_MAT_ID`
  (`ggml/src/ggml-sycl/ggml-sycl.cpp:7421-7438`, threshold `:7820`).
- The scheduler then copies the expert weights to the device, **per split, synchronously**: it waits for the
  split backend, reads the routing ids back to the host with a blocking `ggml_backend_synchronize`, and issues
  run-length-merged `tensor_set_async` copies of the used experts (`ggml/src/ggml-backend.cpp:1684-1776`).
- At `-ub 2048` with 10-of-512 routing, a layer sees 20,480 routed slots over 512 experts; the chance any one
  expert is unused is ~`e^-40`. **Every ubatch therefore copies every CPU-resident expert tensor in full.**
  At `-ncmoe 25` that is ~25 × 1.01 GB ≈ 25 GB per ubatch (962.5 MiB per `-ncmoe` step,
  `logs/long-context-300k-benchmark-report.md` §2.2).

Measured link rate: **21.9 GB/s from pinned host** (`docs/research/05` §10.4). The link is PCIe 4.0 x16
because the board is: the card's upstream port reports `max 32.0 GT/s x16` but the AMD root port
`0000:00:01.1` reports `max 16.0 GT/s x16` (B650 EAGLE AX). **Estimate:** 25 GB / 21.9 GB/s ≈ 1.15 s of a
~4.0 s ubatch (1.96 ms × 2048) — ~29%, *if* nothing overlaps it. The one in-repo cross-check is the
`-ncmoe 24` vs `-ncmoe 32` pair from the same binary (intercept 1.83 vs 1.95 ms/token, §1.2): +0.12 ms/token
for 8 layers is ~32 ms per layer per ubatch, i.e. a layer's 1.01 GB at ~31 GB/s — the right order, a little
faster than line rate, and each intercept carries a few percent of fit noise (n=1 per arm). So: **PCIe
streaming is plausibly 20-30% of the intercept at `-lm none`; it is not most of it.**

### 2.2 Every prefill `MUL_MAT_ID` is a host-driven loop of one GEMM per expert

`ggml_sycl_mul_mat_id` at prefill (`ne12 > 1`) goes to the legacy path (`ggml-sycl.cpp:5325-5541`):

1. copy `ids` to the host and **`stream->wait()`** — drain the whole in-order queue (`:5328-5332`);
2. counting-sort the routed rows on the host (`:5462-5463`), upload the row map, gather `src1` rows;
3. **loop over all `n_as` = 512 experts, calling `ggml_sycl_mul_mat` once per expert with ~40 rows**
   (`:5493-5520`).

MMQ is disabled on SYCL outright (`ggml_sycl_supports_mmq` returns `false`, `:4040-4044`), and 40 rows is
over the MMVQ batch limit of 8 (`common.hpp:170`), so each expert takes `ggml_sycl_op_mul_mat_sycl`:
**dequantise the whole expert to f16, convert the activations to f16, then a oneDNN GEMM**
(`:2950-2985`). Per ubatch that is 48 layers × 3 `MUL_MAT_ID` × 512 experts = **73,728 expert GEMMs, each
with its own dequant kernel writing ~3.3 MB of f16** (640 × 2560 × 2 B) — **estimate** ~240 GB of f16
written and re-read per ubatch, plus ~300K kernel submissions. Neither number depends on depth or on `-ub`
(40 rows at `-ub 2048`, 10 at `-ub 512`: same launch count, same dequant traffic). This, not only PCIe, is
why `-ub 512 → 2048` returned 3.19x (`PLAN.md:277-280`): **inference** — `PLAN.md:343-347` attributed the
whole `-ub` win to "sweeping the CPU-resident expert bank"; the per-expert loop is also a fixed per-ubatch
cost, and it runs for all 48 layers, GPU-resident ones included. The `-ub` sweep cannot tell the two apart.

### 2.3 The MoE cache: still no prefill path, and there is no prefill path worth building

Confirmed in the current code: a `SYCL_MoE_Cached` tensor at `ne12 > 1` falls through to the per-expert loop
reading `src0` through the pinned host pointer (`ggml-sycl.cpp:5357-5364` comment, branch at `:5447`), i.e.
zero-copy PCIe reads — the pattern `docs/research/05` §10.5 measured at **~3.1 GB/s cold** against 21.9 GB/s
for bulk copies. `docs/research/13` §3.3 stands. **Is it fixable? Only to parity.** At `-ub 2048` every
expert is touched every ubatch (§2.1), so the working set is the whole bank; the best a resident pool can do
in prefill is behave like `slots/512` of a layer being GPU-resident, which is exactly what `-ncmoe` already
gives, and the best way to service the rest is a bulk DMA — which is what op-offload already does. A prefill
mode for the cache would be a rewrite of op-offload with extra state. **Fundamental, not an omission.**

### 2.4 Why the existing prefill profile cannot split §2.1 from §2.2

`docs/research/12` §9.4 cites a 32K prefill call-site profile as "126 s of GPU time against `MUL_MAT_ID`'s
87%". Re-aggregating `logs/topk-ab-after.log` reproduces the arithmetic (`MUL_MAT_ID` iq2_s + iq4_nl +
iq3_s = 69.1% by op; the three `ffn_moe_*` call sites 69.0%; `FLASH_ATTN_EXT` 20.9%, of which 25.7 s is one
first-call JIT). But it is **mode 2**, which "time[s] only the host-side dispatch call, never sync"
(`ggml-sycl.cpp:6499-6500`), and the prefill `MUL_MAT_ID` path contains a blocking `stream->wait()`
(`:5332`). **So every `MUL_MAT_ID` in a mode-2 profile is charged with all GPU work enqueued before it** —
attention, the indexer, the dense layers. The 87% (or 69%) is "time spent blocked in `MUL_MAT_ID`", not
"GPU time of `MUL_MAT_ID`". The scheduler's expert copies (§2.1) happen outside `graph_compute` and are not
in the profile at all: 126 s of dispatch plus 43.6 s of tail wait against a 248 s prompt eval leaves ~78 s
unattributed. **Nothing in this repo separates PCIe streaming, the per-expert GEMM loop, and the rest.**
This is §6.1.

---

## 3. What is inside the slope

Today's slope is **~0.0075 ms per token per 1,000 tokens of depth**, i.e. **~15.4 ms per 2,048-token ubatch
per 1,000 tokens of depth.** Candidate terms, each **estimated** for 1,000 tokens of depth at `n_tps = 2048`
over the 12 QSA layers, against that 15.4 ms:

| term | where | per-1K-depth work | est. ms | share |
|---|---|---|---:|---:|
| Dense FA (oneDNN SDPA, §3.1) | `fattn.cpp:135` → `fattn-onednn.cpp` | 2048 × 1000 × 12 × 24,576 FLOP = 0.60 TFLOP | remainder | **~60-80%** |
| Indexer score + ReLU + head sum + bias | `qwen4exp.cpp:934-953` | `[n_kv/4, 4, n_tps]` f32 written, rewritten, summed: ~62 MB/layer | 1.5-2 | ~10-13% |
| Host KQ-mask fill + upload | `llama-kv-cache.cpp:1601-1700` | 2048 rows × `std::copy` of `n_kv` f16 = 4.1 MB + 4.1 MB H2D, single thread | 0.6-1.0 | ~4-7% |
| QSA mask build (`fill` + `set_rows` + `add`) | `qwen4exp.cpp:1043-1066` | ~16 MB/layer | ~0.5 | ~3% |
| `set_input_qsa` block bias + upload | `llama-memory-hybrid-idx.cpp` | `[n_blocks, n_tps]` f32 = 2 MB | 0.2-0.4 | ~2% |
| q4_0 → f16 KV staging for SDPA | `fattn-onednn.cpp` dequant | 24 MB | <0.1 | <1% |

**The KQ-mask build question from `docs/research/13` §7 item 1 — "the cheapest thing in this document to
settle", never measured — comes out small at prefill by this arithmetic: ~3% of the slope on the GPU side,
~5% counting the host-side fill it depends on.** At decode its share was the open question; at prefill it is
not where the time is. This is an estimate, not a measurement, and it assumes ~400 GB/s effective for the
SYCL `fill`/`add` kernels.

### 3.1 FA: which kernel, and how fast it would have to be

`fattn.cpp:135-139` takes `BEST_FATTN_KERNEL_ONEDNN` for `Q->ne[1] >= 32` whenever
`ggml_sycl_flash_attn_ext_onednn_supported` passes; for our shape (q4_0 K/V dequantised to f16 first, f16
`[n_kv, n_tps, 1, 1]` mask, no sinks/softcap) it does (`fattn-onednn.cpp:30-110`), and `GGML_SYCL_DNN` is ON
by default (`ggml/CMakeLists.txt:255`). **Inference** — no log line records the chosen kernel; the 25.7 s
first-call cost in `logs/topk-ab-after.log` is consistent with a oneDNN graph compile plus JIT.

If the whole slope were FA it would be running at **~39 TFLOP/s** (0.60 TFLOP / 15.4 ms); with the other
rows removed, ~50. The B70 is specified at 367 INT8 TOPS; at the usual 2:1 INT8:FP16 XMX ratio that is
~183 dense FP16 TFLOP/s (**inference** — Intel does not publish the FP16 figure). So SDPA is at roughly
20-30% of peak on a head-256, GQA-12 shape with a dense f16 mask broadcast across 24 heads
(`fattn-onednn.cpp:176`, `msk = {1,1,1,q,seq}`) — a mask that is 99% `-INF` at depth under QSA. There is
headroom in the kernel, but the structural waste is computing 100% of `n_kv` when 2,051 cells per query
survive.

### 3.2 Two things checked and ruled out

- **oneDNN re-compiling the SDPA partition every ubatch.** The partition cache is keyed on `seq` = `n_kv`
  (`fattn-onednn.cpp:403-410`), which changes every prefill ubatch, so every ubatch *does* build a new
  partition. But the mode-2 profile's `FLASH_ATTN_EXT` host time, minus the one 25.7 s first call, is ~0.64 s
  over 839 calls in a 62-ubatch prefill — **≤~10 ms per ubatch, <0.5% of prefill.** Not a lever. (The cache
  is also unbounded — up to `n_ctx/256` entries — which is a leak-shaped curiosity, not a speed problem.)
- **Top-k.** 0.06% of the 32K profile (`docs/research/12` §9.4); block-granularity top-k shrank it further.

---

## 4. Candidate fixes, ranked by plausible impact against cost and risk

### 4.1 Stop re-prefilling conversations after reloads — **highest value per line of code**

**Evidence, from production.** `logs/prod-metrics/requests.ndjson` (127 requests, 2026-09-13) prefilled
415,348 new tokens in total. Three requests account for 276K of them; two are **the same conversation being
prefilled again from zero**:

| time | tier | new tokens | cached | prefill tok/s | what happened (**inference**) |
|---|---|---:|---:|---:|---|
| 19:26:45 | 0 | 3,196 | 102,410 | 322 | conversation at 105,752 tokens |
| **19:39:05** | **1** | **109,990** | **0** | 232 | tier 0 → 1 switch: prompt cache discarded, whole conversation re-prefilled (~7.9 min) |
| 19:49:53 | 1 | 2,356 | 120,037 | 228 | normal follow-up, final 123,032 |
| **23:23:12** | 1 | **123,606** | **0** | 250 | 3.5 h idle → sleep, then wake reload: re-prefilled (~8.2 min) |

That is **~234K of 415K prefilled tokens (~56%) and ~16 minutes of TTFT spent recomputing KV state the
server had already computed**. The mechanism is in the code: `destroy()` drops the context
(`tools/server/server-context.cpp:952-966`), sleep calls it (`:971-985`), tier switches go through the same
reload, and `load_model` recreates the host-RAM prompt cache from scratch
(`prompt_cache = std::make_unique<server_prompt_cache>(...)`, `:1541`) — so the one structure that could
have carried the conversation across the reload is also thrown away. `docs/research/13` §9.5 already
recorded "the prompt cache does not survive, so the triggering turn reprefills its whole conversation";
what is new here is its measured share of real traffic.

**Fix.** Before `destroy()` on a tier switch or sleep, `prompt_save` each non-empty slot into
`prompt_cache` (the method exists, `:300-324`, and uses `llama_state_seq_get_data_ext`); on the `is_resume`
path (`:1188`) keep the existing `prompt_cache` instead of recreating it. The next request then restores
through the existing `prompt_load` path (`:1819-1835`). State size, **estimate** from doc 08's 9,504 B/token
for both q4_0 caches: ~1.2 GB at 123K, ~2.9 GB at 300K, plus 112.6 MiB recurrent state — inside the default
`--cache-ram 8192` MiB (`common/common.h:644`), restored at memcpy + H2D speed (seconds) instead of 8-28
minutes of prefill. Upstream #28092 (`server: add --cache-disk`, open, +1706/-71, "supports hybrid/SWA
context checkpoints, restores automatically across process restarts") is the heavier, disk-backed version
of the same idea and is worth watching.

**Cost/risk.** ~30-80 LOC, server only, zero kernel code. Risks, all checkable without production:
(a) `docs/research/13` §7 item 3 — whether `llama_state_seq_{get,set}_data` round-trips this model's three
state kinds — is still formally open; `llama_memory_hybrid_idx::state_write/state_read` exist
(`src/llama-memory-hybrid-idx.cpp:443-480`), but a restore into a context with a *different* `n_ctx`,
`n_ubatch` and `-ncmoe` has never been exercised; (b) the pooled indexer-key cache must be invalidated or
fingerprint-rebuilt after a `state_read` — the existing `LLAMA_QSA_POOL_CHECK=1` in-graph oracle
(`PLAN.md:1139-1147`) is the right gate; (c) host RAM: the saved state is additional anonymous memory on a
box where `free -h`'s `available` is the binding resource (standing rule, `PLAN.md:1506-1515`). The
mechanism can be validated on the small CPU arm `docs/research/13` §9.9 used, with no GPU at all.

**Caveat.** One day, 127 requests; how often reloads fire depends on the idle timeout and on how often
conversations cross 122,880. The share could be much lower on other days. The per-event saving (8+ minutes
per reloaded 120K conversation, ~28 minutes at 300K) is not in doubt.

### 4.2 Put the CPU-resident experts back in pinned host memory — **likely 1.3-1.6x, needs one check first**

**Evidence.** The one clean A/B: same 116K prompt and config, `-lm none -lzm off` **394.14 tok/s** vs
`-lm auto -lzm off` **252.08 tok/s** — 1.56x on prefill, decode unchanged (`PLAN.md:681-696`). Under
`-lm none` the host tensors land in `SYCL_Host` pinned USM; under mmap they are `CPU_Mapped` pageable pages
(`PLAN.md:659-663`, `logs/long-context-300k-benchmark-report.md` §2.1). Production then moved to
`-lm mmap+mlock --mlock-experts-only` (`PLAN.md:2148-2222`) for RAM reasons, and the only production prefill
numbers in the repo sit on the slow side of that A/B: **228-250 tok/s** at 110-124K
(`logs/prod-metrics/requests.ndjson`, 09-13, which predates the mlock change) against 394-418 in the
`-lm none` benchmarks.

**Mechanism (inference, not measured).** §2.1's copies are `tensor_set_async` from the expert tensor's host
buffer. From pinned USM they are DMA at ~21.9 GB/s; from pageable memory the runtime has to stage them. The
arithmetic fits: the A/B's ubatch times are 5.20 s vs 8.13 s, and moving ~25 GB is 1.2 s at the pinned rate,
so the pageable path would be ~6 GB/s. **`mlock` does not change this**: it keeps pages resident, it does not
register them with Level Zero, so mlocked mmap pages are still "pageable" from the copy engine's point of
view. That last step is the unverified one.

**The check before any code:** a `-lm mmap+mlock --mlock-experts-only` arm of the existing
`staging/work/run_116k_lazy.sh` 116K benchmark, compared against `logs/long-context-120k-ncmoe25/`. If it
lands near 252, this is real; if near 417, mlock already fixed it and this item is closed.

**Fix, if the check says so.** The goal is the `-lm none` placement for the ~25-33 GB of expert tensors only,
leaving the 28.8 GB PLE table on mmap as production does today — the same resident-RAM footprint as
`--mlock-experts-only`. Two small obstacles, both in code read here: `-ot` cannot name the host buffer type
(only device and "extra" bufts are registered, `common/arg.cpp:252-280`), and the loader converts a host
buft back to the CPU buft whenever mmap is on (`src/llama-model-loader.cpp:1262-1270`). The upstream comment
at `src/llama-model.cpp:1046-1051` says in as many words that host buffers exist to make offloaded
large-batch ops cheaper. **~20-50 LOC** (a `--pin-experts`-style flag, or honour an explicit host-buft
override under mmap), low-medium risk. Costs to price: pinned USM is invisible in `RssAnon`
(`PLAN.md:659-663`), and copying 25-33 GB into USM at load makes every reload slower (the full `-lm none
-lzm off` load was 219 s, `PLAN.md:1450-1451`) — which is another reason to do §4.1 first.

### 4.3 Vendor upstream #29245, grouped MoE XMX GEMM — **+34-50% measured upstream, medium effort**

`gh pr view 29245`: "sycl: add grouped MoE XMX GEMM", **OPEN**, +775/-16, author `cwriter` (who also wrote
#28670, the radix top-k this project already vendored). It replaces exactly §2.2's one-GEMM-per-expert loop:
"Build a host-side schedule of active expert tiles and launch as one grouped kernel instead of one GEMM per
expert", dequantising IQ weights into f16 tiles in local memory (`joint_matrix`, XMX) instead of writing f16
out and reading it back. Measured by the author, **on this model**: `Qwen3.8-Flash-Next UD-IQ4_XS`, prompt
274.1 → 368.3 tok/s (**+34%**), decode +1%; on a B60 with `Qwen3-30B-A3B UD-IQ3_XXS`, pp2048 677 → 1007
(**+49%**). It now covers IQ4_NL, IQ3_S, IQ4_XS, IQ3_XXS, IQ2_XXS/XS/S, IQ1_S/M
(`GGML_SYCL_XMX_GATHER_TYPES` bitmask, default all on) — which includes the `iq2_s`, `iq3_s` and `iq4_nl`
expert types the 32K profile shows this project's `UD-IQ3_XXS` actually dispatches
(`logs/topk-ab-after.log`, §2.4).

**What it cannot do:** anything about §2.1's PCIe copies; it speeds up the GPU side of every `MUL_MAT_ID`,
CPU-sourced or not. **Risks:** open PR, one reviewer (arthw) could not reproduce a gain on a different MoE
and asked for the method; it computes in f16 on XMX "trad[ing] some precision for speed relative to the
per-expert library GEMM" (its own SYCL.md text); and it edits `ggml-sycl.cpp`'s `mul_mat_id` dispatch, which
this project has also modified (fused decode paths, MoE-cache branches) — expect a hand-merge like #28670's.
Correctness gate: `test-backend-ops -o MUL_MAT_ID` plus the env var set to 0 as a same-binary control.
**Effort: ~a day including the merge; the measurement is one 116K run.**

### 4.4 Shrink the `n_kv × n_ubatch` compute buffer so the deep tier can run `-ub 2048`

**Why it matters.** Tier 2's intercept is 4.26 ms/token at `-ub 1024` vs 1.96-2.48 at `-ub 2048` (§1.2), and
at 300K that intercept is ~78% of TTFT (§1.3). The tier is at `-ub 1024` because the compute buffer grows as
`n_kv × n_ubatch`: 8,716 MiB at 311K/`-ub 2048` (`logs/long-context-300k-benchmark-report.md` §2.3), ~16 GiB
at 614K (`PLAN.md:1288-1289`). **Estimate**, that is ~13.6 bytes per (`kv`, ubatch-token), and the graph
explains it: the f16 KQ mask input (2 B), `kq_mask_all` (2 B) and the `ggml_add` result (2 B) in
`build_attn_qsa` (`qwen4exp.cpp:1043-1066`), plus the indexer's `[n_kv/4, 4 heads, n_tps]` f32 score (4 B)
and its ReLU copy (4 B) (`:934-937`).

**Fix.** (a) Replace `:934-953` with `ggml_lightning_indexer` — it exists, has a SYCL kernel
(`ggml/src/ggml-sycl/lightning-indexer.cpp`), and computes `Σ_h w_h·relu(q_h·k)` straight to `[n_kv, n_batch]`
(`ggml/include/ggml.h:2662-2678`), with `w_h = 1` and the per-block bias as its mask — deleting ~8 of the
13.6 B and several `n_kv × n_tps` passes. `docs/research/11` / `PLAN.md:556-561` already proposed this fusion
for decode. (b) Build the QSA mask in place (fill, `set_rows`, `ggml_add_inplace`) for another 2 B.
**Estimate:** ~13.6 → ~4.5 B per (`kv`, token), enough for `-ub 2048` at the 512K tier on today's
arithmetic, or 3-5 more GPU-resident expert layers at tier 1. **Expected value:** tier 2 intercept toward
tier 1's ~2.5 ms/token → **~1.4-1.6x TTFT for 300K+ prompts.** It also trims ~10-15% off the slope (§3).

**Risks.** The op takes an **f16** mask (`ggml.h:2667`); the block bias uses `1e9` to force-select recent
blocks, which overflows f16 to `+inf` — harmless for top-k ordering but must be checked against the tie
semantics `PLAN.md:163-176` documented. Needs a `test-backend-ops` equivalence arm (the `test_qsa_indexer`
harness exists) and one allocator probe (`-lv 4`, `sched_reserve: SYCL0 compute buffer size`, ~4 minutes, no
prompt) before any tier table changes. **~100-200 LOC, medium.**

### 4.5 Overlap the expert copies with compute (port of upstream #28414) — **≤ the §2.1 share**

#28414 (`--prefetch-experts-slots`, OPEN, +327) issues the next split's expert uploads one split ahead on a
second backend instance into rotating staging buffers, prefill-only, gated on
`ids->ne[0]*ids->ne[1] >= 2*n_expert` — exactly our regime. Measured on CUDA: 42K-token TTFT −11% and −22%.
It needs `async` + `events` device caps and was written for CUDA streams; a SYCL port means a second queue
and cross-queue events, and this project already has a recorded unresolved non-determinism from an async
cross-queue barrier (`docs/research/13` §7 item 5). A reviewer reproduced **silent wrong output (runs of
`/`) on `qwen4exp`** in a multi-GPU layer-split configuration — not this box's topology, but the same model.
Ceiling here: the ~20-30% PCIe share of §2.1, less whatever §4.2 already recovers. **Medium-large effort,
medium-high risk; after §4.1-4.4.**

### 4.6 Sparse prefill attention — **structurally right, worth ≤10-20% where this deployment lives**

Status upstream: CUDA has it for qwen4 (#28770, merged 2026-09-20: union of the selected cells per
`ncols1 = 8` query group, enabled at `n_kv ≥ 2 × ncols1 × n_kv_max`, i.e. ~32K), measured **1.08x at 10K →
1.26x at 100K** pp2048 on a DGX Spark with a fully resident IQ1_S model. SYCL's sparse FA (#28796, merged
2026-09-25, and this project's own `fattn-sparse.cpp`) is **decode-only by construction** —
"single-token decode only; prefill amortises the scan already" (`gh pr diff 28796`). Vulkan has none. So the
read side does not exist for prefill on any Intel backend.

**What it would buy here:** at most the FA share of the slope — **~11-15% of TTFT at 116K, ~13-18% at 300K,
~25-30% at 600K** (§1.3 × §3). Upstream's 1.26x at 100K came from a model whose intercept is a fraction of
ours (964 tok/s shallow); with our heavier intercept the same kernel win is a smaller TTFT win. **How:**
`docs/research/10` §4.3's objection (256 oneDNN SDPA calls per ubatch) still holds for a CUDA-shaped port.
The shape that fits oneDNN is a per-query-tile *union* gather — compact the union of blocks selected by, say,
64-256 consecutive queries into a contiguous K/V + mask and hand SDPA an effective `seq` — whose value
depends entirely on how much adjacent queries' 513-block selections overlap, which nobody has measured (§6.3).
A custom XMX kernel doing per-query gathers (the FlashMLA sparse-prefill approach, 640 TFLOP/s on H800) is
the other route and is a project. **Defer until §4.1-4.4 have moved the intercept**; the slope's share grows
as the intercept shrinks, which is when this becomes worth its cost (the same "fix the floor and the slope
becomes the problem" dynamic `PLAN.md:791-800` described for decode).

### 4.7 Not recommended

- **Switching prefill to Vulkan.** #28501 (512-expert `count_experts` fix) **merged 2026-09-18**, so Vulkan's
  reported +16-19% MoE prefill is now available there; Vulkan also has the fused `topk_radix_qsa`. But Vulkan
  has no sparse FA at all, so a split deployment would give up this project's measured ~4.5x decode win at
  116K (`PLAN.md:1351-1352`) for the decode half, and the one Vulkan prefill artifact in `logs/`
  (`bench-flashnext-vulkan.log`: `-ncmoe 48`, pp4096 14.43 tok/s, tg 0.53) is uncited and not comparable to
  anything. `docs/research/09` §8's one-off same-config Vulkan comparison was never run. If anyone wants the
  question closed, that measurement is still the way.
- **Chasing the KQ-mask build or `set_input_qsa` for prefill.** ~5% and ~2% of the slope by §3's arithmetic.
- **A MoE-cache prefill path.** §2.3.
- **Hardware, for the record:** a PCIe 5.0 board would roughly double §2.1's streaming rate (card is
  Gen5-capable, root port is not). Not a software recommendation.

### 4.8 Summary

| # | change | est. effect on TTFT | effort | risk | gate before building |
|---|---|---|---|---|---|
| 4.1 | carry prompt cache across tier switch / sleep-wake | removes 8-28 min per reloaded long conversation; ~56% of prefilled tokens on the logged day | ~30-80 LOC, server | low-med | CPU-arm state round-trip + `LLAMA_QSA_POOL_CHECK` |
| 4.2 | experts in pinned `SYCL_Host`, PLE stays mmap | up to 1.56x at tiers 0/1 if prod is on the slow path | ~20-50 LOC, loader | low-med | one `mmap+mlock` 116K arm |
| 4.3 | vendor #29245 grouped MoE XMX GEMM | +34-49% (upstream-measured) | ~1 day incl. merge | med | `test-backend-ops -o MUL_MAT_ID`, env-var A/B |
| 4.4 | lightning-indexer fusion + in-place mask → `-ub 2048` at 512K tier | ~1.4-1.6x for 300K+ prompts | ~100-200 LOC | med | equivalence arm + one allocator probe |
| 4.5 | port #28414 copy/compute overlap | ≤20-30% of intercept, less after 4.2 | M-L | med-high | after 4.2 |
| 4.6 | sparse prefill FA | ≤11-18% at 116-300K, ~25-30% at 600K | L | high | §6.3 overlap measurement |

None of these compound perfectly — 4.2 and 4.5 attack the same bytes, and 4.3/4.4 shrink the intercept that
makes 4.6 small — but 4.1 is orthogonal to all of them.

---

## 5. What was not done, and why

No GPU work of any kind: the card serves production traffic and the brief forbade heavy benchmarks. No
production metrics after 2026-09-13 are committed here, so nothing in this document is measured on the
current `mmap+mlock` production config. The fit in §1 uses n=1 runs per configuration, and `PLAN.md` records
leading-arm and run-to-run effects of several percent on prefill (`PLAN.md:1027-1034`); the slope change in
§1.2 (−35%) is far outside that, the per-row intercepts are not.

---

## 6. Open unknowns, stated plainly

Ordered by how much they would change §4.

1. **How the depth-independent intercept (~4 s per 2,048-token ubatch at tier 0) splits between PCIe expert
   streaming (§2.1), the per-expert `MUL_MAT_ID` loop (§2.2), and everything else.** This decides whether
   §4.2/§4.5 or §4.3 is the bigger lever, and it is ≥75% of TTFT at every depth up to 300K. The existing
   profile cannot answer it (§2.4). Cheapest decisive measurements, none needing a long prompt:
   `test-backend-ops perf -o MUL_MAT_ID` at the real prefill shape (512 experts, 10 used, 2048 tokens,
   2560×640, the model's `iq2_s`/`iq4_nl` types) × 144 gives the GPU side; a `GGML_SYCL_OP_PROFILE=1`
   (drain-around-every-node) window over a single 2,048-token ubatch gives the per-node split at the price of
   a distorted run; and a two-point `-ncmoe` pair at `-ub 2048` on today's binary gives the streaming term.
2. **Whether production's `mmap+mlock` config really pays the pageable-copy penalty.** §4.2's check: one 116K
   arm. Inferred from mechanism plus pre-mlock production numbers only.
3. **How much adjacent queries' QSA selections overlap.** Determines whether a union-gather sparse prefill FA
   (§4.6) is a 5x or a 1.2x on the attention term. Measurable offline by dumping `indexer_top_k_blocks` for
   one ubatch at depth and computing the union size per 64/128/256-query tile — a tensor dump, not a
   benchmark.
4. **Whether `llama_state_seq` restore works across a reload into a different tier's `n_ctx`/`n_ubatch`/
   `-ncmoe`** — §4.1's only real risk, and `docs/research/13` §7 item 3 restated for this purpose.
5. **The FA vs indexer vs host split of the slope (§3).** Arithmetic only; `test-backend-ops perf -o
   FLASH_ATTN_EXT` at `Q->ne[1] = 2048`, head 256, GQA 12, q4_0, `n_kv = 32K..300K` would pin FA's share
   directly (the existing qwen4exp perf loop only covers the decode shape).
6. **PLE page faults during prefill in production.** `--mlock-experts-only` deliberately leaves the PLE table
   evictable, and `mincore()` showed it 59.6% resident (`PLAN.md:2206-2210`). Prefill touches 16 rows × 2,048
   tokens per ubatch; at 40% eviction that is ~13K potential faults per ubatch — **~0.1 s if they resolve in
   parallel on the dedicated NVMe, up to ~1 s if they serialise.** Unmeasured; a `majflt/s` sample from
   `/proc/<pid>/stat` during one real production prefill is non-invasive and settles it. Upstream #29030
   (direct-read gather for lazy PLE rows, +65-121% pp on Strix Halo with lazy mode) is related but targets
   `-lzm on`, which production does not use.

---

## 7. Corrections to fold back into other documents

1. **`logs/long-context-300k-benchmark-report.md` §4.2** ("falls 1.84x from 22K to 141K"): pre-dates block
   top-k and the pooled-key cache; on today's binary the same span is ~1.41x (§1.2 here).
2. **`docs/research/12` §9.4** ("126 s of GPU time against `MUL_MAT_ID`'s 87%"): the profile is mode 2
   (host dispatch time) and the prefill `MUL_MAT_ID` path blocks on `stream->wait()` (`ggml-sycl.cpp:5332`),
   so that share includes all GPU work enqueued before each call. It is not a GPU-time attribution (§2.4).
3. **`PLAN.md:343-347`** attributes the `-ub` win to "the fixed per-ubatch cost of sweeping the CPU-resident
   expert bank — weight streaming". The per-expert GEMM loop (§2.2) is an equally fixed per-ubatch cost that
   also covers the 23-24 GPU-resident layers; the `-ub` sweep cannot separate them.
4. **The brief's framing** that the SYCL radix top-k is an unimplemented prerequisite: vendored 2026-09-11,
   merged upstream 2026-09-14 (#28670). And the QSA sparse compute path is not a no-op on SYCL for decode any
   more (`fattn-sparse.cpp`, `PLAN.md:1194-1352`; upstream #28796) — it is a no-op for prefill only.
