# Where the 3.1x Long-Context Decode Gap Actually Goes

**Status: measured. Verdict: the QSA indexer is real but small — 18.29 ms of a 240 ms decode token at 116K,
which closes only ~10% of the gap `docs/research/10` left open. ~146 ms/token, 61% of the token, is not in
the SYCL graph at all. No production code was changed.**

> **READ §7 FIRST (correction, same day, later). §3.3 and §5 of this document are wrong about *which tensor*
> is being paged and therefore about *which flag* fixes it.** §5's `--load-mode none` recommendation was
> acted on and delivered nothing; so did `-lm mlock`. The paged tensor is not the CPU-resident experts — it
> is `per_layer_token_embd`, a **27,465 MiB `TENSOR_READ_LAZY` PLE n-gram hash table** that keeps an mmap
> regardless of `--load-mode` and that mlock deliberately skips, and which every token hits with **16
> uniformly-random row gathers** (`src/models/qwen4exp.cpp:192-193, 1294-1306, 1381`). The right flag is
> **`-lzm off` / `--lazy-mode off`**, and it is measured: **decode 240.01 -> 89.72 ms/token (4.167 -> 11.146
> tok/s, 2.68x), prefill 240.15 -> 394.14 tok/s (1.64x), `tg_3s` interval spread 3.09x -> 1.03x, major
> faults 80-100/s -> 0.** §2's component model survives intact and is now **closed to 4.1%** (predicted
> 93.52 ms/token, measured 89.72); §3.1's separately-estimated "~16 ms serving path" does not survive and
> must not be added on top. §1, §2 and §4 stand as written.

Date: 2026-09-11 (late, after `docs/research/10-sycl-sparse-attention-scoping.md`). This document answers
doc 10 §3.4's open question ("the cheapest next measurement in the project and it has never been run") and
supersedes its §3.3 candidate list. Doc 10's §1, §2 and §4 stand unchanged and were re-verified here.

All repo paths are relative to
`/media/da3dsoul/Golias/AIProjects/Qwen3.8-Flash-Next-LlamaCpp-MoE-Cache-Arc-Experiments/src/llama.cpp/`
unless stated otherwise. Tree state: `HEAD = 6d9c82ea2`, working tree dirty (this project's uncommitted
SYCL MoE-cache / MTP / top-k work).

Statements joining facts established separately are flagged **inference**. Arithmetic extrapolating a
measured datapoint is flagged **estimate** and shows its assumptions.

---

## 0. Verdict in one paragraph

A decode token at `n_kv = 118016` was decomposed into measured components. Two are depth-proportional and
both are GPU work: `FLASH_ATTN_EXT` at **34.77 ms/token** (re-measured, within 1% of doc 10) and the QSA
lightning-indexer chain at **18.29 ms/token** (measured here for the first time). The rest of the graph is
depth-independent at **40.46 ms/token**. That model predicts **93.52 ms/token (10.69 tok/s)** at 116K, and
it is not a guess: it reproduces this project's own `llama-bench` numbers at `d2048` and `d8192` to **0.5%**
(§2.4), and it lands within 14% of the **fastest 3-second intervals the real 116K benchmark itself
recorded** (§3.2). The 116K benchmark's *median* was 240.01 ms/token. So **~146.5 ms/token — 61% of a real
decode token — is spent outside the SYCL graph**, it varies **3.09x within one run at constant depth**
(§3.2), and the mechanism was caught live on this box during this pass: the array holding the model was at
**99.9% utilization** from an unrelated 10-day training job while llama.cpp page-faulted its 52.8 GiB of
CPU-resident expert weights at **0.36 MB/s** (§3.3). **The unexplained remainder is host-side expert-weight
paging, not the indexer.** Doc 10's leading hypothesis was measured and is mostly wrong: it guessed
25-100 ms/token for the indexer; the answer is 18.29. Recommendation in §5: fix the paging exposure
(`--load-mode none`) before any more kernel work, because it is worth **~2.2x on its own**, whereas a
perfect sparse FA *and* a perfect indexer together are worth only **1.28x** while the paging term is still
there.

---

## 1. What was measured, and how

| Measurement | Tool | Log |
|---|---|---|
| Decode throughput vs. depth, pure `llama_decode` | `llama-bench -p 0 -n 32 -d ... -r 1` | `logs/decode-profile/bench-depth-curve-ABORTED.log` |
| `FLASH_ATTN_EXT` at production decode shape | `test-backend-ops perf -o FLASH_ATTN_EXT` | `logs/decode-profile/fa-perf-recheck.log` |
| QSA indexer chain at production decode shape | `test-backend-ops perf -o QSA_INDEXER` (**new case**) | `logs/decode-profile/qsa-indexer-perf.log` |
| QSA indexer chain vs. CPU reference | `test-backend-ops test -o QSA_INDEXER` | `logs/decode-profile/qsa-indexer-correctness.log` |
| In-model decode composition, host/GPU split | `GGML_SYCL_OP_PROFILE=2 GGML_SYCL_OP_PROFILE_WINDOW=1` | `logs/decode-profile/opprof-mode2-d2048-d8192.log` |
| Host I/O contention during all of the above | `iostat -x`, `vmstat`, `/proc/<pid>/{io,stat,status}` | `logs/decode-profile/host-io-sampler.log` |

Config throughout is `PLAN.md`'s current recommendation: `-ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -ncmoe 24
-ub 2048`, model `UD-IQ3_XXS`.

### 1.1 The only source change: one test case

`tests/test-backend-ops.cpp:6632-6806` adds `test_qsa_indexer`, which rebuilds the whole of
`llama_model_qwen4exp::graph::build_qsa_top_k` (`src/models/qwen4exp.cpp:721-870`) op for op at a chosen
`n_kv`, with `n_tps = n_stream = 1` (decode) and `width = 2051`. Perf cases at `tests/test-backend-ops.cpp:11342-11343`,
correctness cases at `:10568-10569`. **No production file was touched**; `src/models/qwen4exp.cpp:943-946`
is still exactly as doc 10 left it.

Three details of that test matter for believing its numbers:

1. **The indexer K cache is quantized, because it really is.** `llama_memory_hybrid_idx` hands the indexer
   cache the same `type_k` as the attention cache (`src/llama-memory-hybrid-idx.cpp:60-63`), and that
   `type_k` is `params.type_k` (`src/llama-model.cpp:2585-2603`). So under `-ctk q4_0` the 118016-row
   `ggml_get_rows` at `src/models/qwen4exp.cpp:789` is a *dequantizing* gather. Both KV types were measured.
2. **The chain is timed as one graph.** `eval_perf` amortizes by appending copies of the **final node only**
   (`tests/test-backend-ops.cpp:1624-1627`), which for a whole-graph case divides one chain evaluation by
   thousands of duplicated `top_k` conts — the first run of this case reported a flat 2.1 us at every depth
   for exactly that reason. Overriding `op_size` above `eval_perf`'s 32 GiB target forces `n_runs == 1`, so
   `us/run` is one honest evaluation of the chain.
3. **It is verified against the CPU reference, not just run.** `test-backend-ops test -b SYCL0 -o QSA_INDEXER`:
   **2/2 pass at `max_err = 0.0`**, in both `q4_0` and `f16`, comparing the selected index *set* (the chain
   ends in a `top_k`, whose order is unspecified). That is what makes the replica a measurement of the
   model's computation rather than of a lookalike graph.

### 1.2 Caveat carried from doc 10, restated

These are isolated-op timings used as a proxy for in-model cost, with the same caveat doc 10 stated: nothing
else competes for cache or bandwidth. The cross-check that makes the proxy usable is §2.4 — the component
model built from them reproduces whole-model `llama-bench` numbers at two depths.

---

## 2. The decode token, decomposed

### 2.1 `FLASH_ATTN_EXT`, re-measured

`test-backend-ops perf -b SYCL0 -o FLASH_ATTN_EXT -p "hsk=256,hsv=256,nh=2,nr23=[12,1]"`, `nb = 1`:

| `n_kv` | `q4_0` us/call | `f16` us/call | `q4_0`, x12 layers |
|---:|---:|---:|---:|
| 32 768 | 809.72 | 511.93 | 9.72 ms |
| 65 536 | 1 619.44 | 1 012.28 | 19.43 ms |
| **118 016** | **2 897.77** | **1 785.81** | **34.77 ms** |

Doc 10 measured 2 927.28 us at 118016/`q4_0`; this pass gets 2 897.77, **1.0% apart**. The environment is
stable and the two documents' numbers are directly comparable. The `q4_0` re-dequant tax doc 10 identified
is **13.34 ms/token** at 118016 (`(2897.77 - 1785.81) x 12`).

### 2.2 The QSA indexer chain, measured for the first time

`test-backend-ops perf -b SYCL0 -o QSA_INDEXER`, one full `build_qsa_top_k` per call:

| `n_kv` | `q4_0` us/call | `f16` us/call | `q4_0`, x12 layers |
|---:|---:|---:|---:|
| 8 192 | 157.23 | 164.85 | 1.89 ms |
| 16 384 | 243.43 | 266.97 | 2.92 ms |
| 32 768 | 427.79 | 476.75 | 5.13 ms |
| 65 536 | 824.42 | 923.17 | 9.89 ms |
| **118 016** | **1 524.24** | **1 711.11** | **18.29 ms** |

Three readings:

1. **Linear in `n_kv` above ~16k** — 3.60x the depth from 32768 to 118016 costs 3.56x, so extrapolation
   between measured points is safe.
2. **Doc 10's estimate was too high.** Its §3.4 bracket was "25-100 ms/token" and it flagged that this
   "straddles the missing 163 ms". The measured number is **18.29 ms/token**, below the bottom of that
   bracket. The bracket's error was the efficiency assumption, not the traffic arithmetic.
3. **`q4_0` makes the indexer 11-12% *faster*, the opposite of its effect on FA.** The gather at `:789`
   reads 72 B per row quantized against 256 B dense, and nothing downstream re-expands the cache. So
   `-ctk q4_0` is unambiguously right for the indexer, while on `FLASH_ATTN_EXT` it is a 1.62x tax. Worth
   recording because the two pull in opposite directions and only the FA half was previously known.

### 2.3 Depth-proportional total

At `n_kv = 118016`, per decode token: **34.77 ms FA + 18.29 ms indexer = 53.06 ms** of depth-proportional
GPU work. The indexer is **34.5% of that pair**, i.e. real, and materially larger than doc 10 could assume
when it wrote "sparse FA cannot remove one microsecond of it" — that statement is still true, and now
carries a price tag.

### 2.4 The component model, and the two points that validate it

Taking the measured `d2048` token and subtracting the two depth-proportional terms at that depth gives a
depth-independent floor:

```
floor = 42.27 ms (measured, tg32 @ d2048, 23.66 tok/s)
      -  0.67 ms (FA @ 2048, 12 layers, q4_0)
      -  1.14 ms (indexer @ 2048, 12 layers, q4_0, interpolated from the 8192/16384 slope)
      = 40.46 ms/token
```

**Validation at `d8192`** (the model was not fitted to this point):

```
model @ 8192 = 40.46 + 2.59 (FA) + 1.89 (indexer) = 44.94 ms/token = 22.25 tok/s
measured                                                            22.13 tok/s   -> 0.5% error
```

**Extrapolation to the 116K benchmark's depth:**

```
model @ 118016 = 40.46 + 34.77 + 18.29 = 93.52 ms/token = 10.69 tok/s
```

Doc 10's equivalent model gave 76.9 ms/token because it had no indexer term. **The indexer accounts for
16.6 ms of doc 10's 163.1 ms of unexplained cost — 10.2% of it.** The other 89.8% remains.

### 2.5 What the depth-independent 40.46 ms is made of

`GGML_SYCL_OP_PROFILE=2` (host-dispatch timing, no queue drain, so the run is only ~5.5% slower than
un-profiled) with `WINDOW=1`, aggregated over an exactly-bounded decode window. The window boundary is
found by walking reports backwards until `12 x n_gen = 384` `FLASH_ATTN_EXT` calls have been counted, which
lands on the first decode split without needing to know splits-per-token
(`staging/work/agg_opprof_decode.py`):

| term | `d2048` | `d8192` | delta |
|---|---:|---:|---:|
| host SYCL dispatch loop | 12.57 ms | 12.98 ms | +0.41 |
| GPU tail wait | 13.53 ms | 14.52 ms | **+0.99** |
| outside `graph_compute` (CPU-backend MoE + host graph/input) | 18.64 ms | 18.24 ms | -0.40 |
| **token** | **44.74 ms** | **45.75 ms** | **+1.01** |

Three things fall out:

1. **The whole depth slope lands in GPU tail wait** (+0.99 of +1.01). Host dispatch and the
   outside-`graph_compute` term are flat with depth, exactly as a correct model requires. This is
   independent confirmation that the depth-proportional cost is GPU kernel time, i.e. FA plus the indexer,
   and that there is no fourth depth-proportional term hiding anywhere.
2. **`-ncmoe 24`'s CPU-resident MoE is ~18.2-18.6 ms/token and depth-independent.** It is invisible to the
   SYCL profiler by construction (the 24 GPU-resident expert layers show as `MUL_MAT_ID iq2_s`/`iq4_nl` at
   only 0.38 ms/token), so it is measured here as the residual between the token and `graph_compute`. This
   is the useful sanity baseline the brief asked for, and it holds.
3. **A decode token dispatches ~2,800 SYCL nodes, and hyper-connections are the single largest group.** The
   top ten call sites are all `hc_*`/`hcc_*` (`hc_down`, `hc_up`, `hc_rms`, `hc_norm`, `hcc_repeat`,
   `hc_inject`, `hc_cont`, `hc_gate`, `hc_down_silu`, `hc_down_scale`), each at **96.8 calls/token** —
   ~970 nodes/token, ~4.0 ms of the attributed host dispatch. The indexer's own call sites
   (`indexer_score_tokens`, `indexer_top_k`, ...) are far down the list at shallow depth, as the arithmetic
   predicts. **Noted, not pursued**: `hyper_connect.cpp` is out of scope for this pass, and fused
   `GGML_OP_DSV4_HC_PRE`/`_COMB`/`_POST` ops already exist in the SYCL backend
   (`ggml/src/ggml-sycl/ggml-sycl.cpp:5673-5681`), so why decode still emits the unfused sites is a question
   for whoever owns that code, not for this document.

---

## 3. The 146 ms that is not in the graph

### 3.1 The size of it

```
measured, llama-server @ 116K, server-side timers over 2000 tokens : 240.01 ms/token (4.166 tok/s)
model (§2.4)                                                       :  93.52 ms/token (10.69 tok/s)
residual                                                           : 146.49 ms/token -- 61% of the token
```

(`logs/long-context-120k-ub2048/run1.stream.jsonl`: `predicted_n = 2000`, `predicted_ms = 479778.926`.)

Part of that residual is the serving path, and it is small: the 116K benchmark report's own §2.2 measured
`llama-server` at **17.05 tok/s (58.7 ms/token) at trivial depth** against `llama-bench`'s 42.27 ms at
`d2048`, i.e. the full sampler chain plus detokenization plus SSE streaming costs about **16 ms/token**.
That leaves ~130 ms/token with no graph-side and no serving-side explanation.

### 3.2 The strongest evidence is inside the benchmark this project already ran

`llama-server` logs a `tg_3s` decode rate every three seconds. Across the three 116K runs, at constant
depth and constant config:

| run | intervals | min tok/s | median | max tok/s | best interval ms/token | spread |
|---|---:|---:|---:|---:|---:|---:|
| `logs/long-context-120k/server.log` (baseline) | 973 | 1.49 | 4.85 | **10.53** | **95.0** | **7.07x** |
| `logs/long-context-120k-after/server.log` (top-k fix) | 144 | 2.85 | 3.95 | 7.61 | 131.4 | 2.67x |
| `logs/long-context-120k-ub2048/server.log` (current) | 145 | 3.05 | 3.93 | **9.41** | **106.3** | **3.09x** |

Two things follow, and they are the load-bearing result of this document:

1. **The best intervals match the model.** The `-ub 2048` run's fastest three seconds ran at 106.3 ms/token
   against a model prediction of 93.52; the baseline run's fastest ran at 95.0. **The graph cost model is
   not just arithmetic — the real run reaches it, repeatedly, in its own best windows.**
2. **The spread is 3.09x to 7.07x within a single steady-state run at fixed `n_kv`.** GPU kernel time at a
   fixed shape does not vary 3-7x between adjacent three-second windows. A host-side stall backed by a
   shared disk does. The median being 2.4x worse than the best interval is the signature of a cost that is
   *sometimes absent*, which no term in §2 can be.

### 3.3 The mechanism, caught live during this pass

The `d32768` and `d65536` arms of the depth sweep were started and **abandoned**, and the reason is the
finding:

- `bcache0` at **99.9% util**, `md0` at **99.8% util** (`iostat -x`), on the array the model file lives on
  (`/dev/bcache0` -> `md0` + `nvme1n1`, `staging/models` is on it).
- The load was external: `unbooru-tagger-training` (pid 1040967, **10 days 3 h uptime**, `read_bytes =
  26.16 TB` lifetime) reading at a sustained **80-96 MB/s** from that same array. This is the same process
  doc 10 §3.1 had to abandon runs for, and **it has been running continuously since before every 116K
  benchmark this project has ever recorded.**
- Our `llama-bench` got the leftovers: main thread in `D` state, **83 major faults/s**, `r_await` 15-16 ms,
  RSS growing at **0.36 MB/s** and stuck at 24.8 GiB against the ~52.8 GiB of `CPU_Mapped` expert tensors
  that `-ncmoe 24` requires. At that rate the arm would have needed ~22 hours.
- Host memory: `free` = ~1 GiB, `buff/cache` = 97 GiB, **swap fully exhausted at 9.2/9.3 GiB**. So the
  52.8 GiB expert set is held only in evictable page cache, competing directly with an unrelated process
  streaming ~90 MB/s through that same cache.

**Why this explains the residual (inference, but quantified).** Every decode token routes to a fresh,
randomly-selected 10-of-512 expert set in each of the 24 CPU-resident layers, so the effective working set
is the whole 52.8 GiB — there is no locality to exploit. At the ~13 ms per major fault measured above,
**146.5 ms/token is about 11 faults per token, i.e. ~45 KB/token**, or ~90 MB over the 2000-token benchmark
window. That is a small enough number to be invisible in any aggregate the project has collected, and it is
entirely consistent with both the 3-7x interval spread (§3.2) and the observed array saturation.

### 3.4 What was ruled out

- **The QSA indexer** — measured at 18.29 ms/token, §2.2. Accounts for 10.2% of the gap, not the bulk.
- **A fourth depth-proportional graph term** — §2.5 shows the entire `d2048`->`d8192` slope lands in GPU
  tail wait, and §2.4's model reproduces `d8192` to 0.5%. There is no room for one.
- **CPU-resident MoE compute** — 18.2-18.6 ms/token and flat with depth (§2.5). It is a real cost but it
  cannot grow into 146 ms by itself. (It is, however, exactly the thing that pays the page-fault bill.)
- **VRAM pressure** — already ruled out by doc 10 §3.3 across three configs with very different headroom.
- **The serving path** — ~16 ms/token, §3.1.

### 3.5 One redundancy found, measured, and deliberately not fixed

`src/models/qwen4exp.cpp:851-857` casts the KQ mask from f16 to f32 inside `build_qsa_top_k`. `kq_mask`
comes from `inp->get_kq_mask()` (`:970`), which returns the single `self_kq_mask_cnv` tensor shared by every
layer (`src/llama-graph.h:321,340`), and ggml does not common-subexpression graph nodes. So a decode token
performs **12 identical 118016-element f16->f32 casts**, one per QSA layer, where one would do — hoistable
into the existing per-ratio `qsa_inps` cache (`src/models/qwen4exp.cpp:753-775`) with no correctness
ambiguity.

**It is not worth doing.** The cast moves 708 KB; eleven redundant copies are ~8 MB/token, which even at
10% of this card's bandwidth is well under 0.2 ms/token — under 0.1% of the token. (The mode-1 profiler
reports this op as the chain's most expensive, at 2.67 ms/call, but mode-1 per-op absolutes are unusable
here: mode 1 drains the queue around every node, and the chain's ~30 nodes sum to 11.16 ms against a true
1.52 ms, so a single op reported above the whole chain's real cost is arithmetically impossible.) Recording
it so the next reader does not re-derive it and mistake it for a lever.

---

## 4. Corrections to `docs/research/10`

| doc 10 claim | status |
|---|---|
| §3.4 "the larger depth-proportional term is almost certainly the QSA indexer chain", est. 25-100 ms/token | **Wrong.** Measured 18.29 ms/token, below the bracket, and smaller than FA's 34.77. |
| §0/§5 "there is a 3.1x discrepancy... not attention" | **Correct, and now attributed**: 10.2% indexer, ~10% serving path, ~80% host expert paging. |
| §3.3 candidate 2, "host page-cache residency of the 52.8 GiB CPU-resident expert set" | **Promoted to the leading explanation**, with a live mechanism observation (§3.3). |
| §2.2 `FLASH_ATTN_EXT` 2927.28 us at 118016 `q4_0` | **Reproduced** at 2897.77 us, 1.0% apart. |
| §3.1 standing note to check `vmstat`/`D`-state before trusting a long-context number | **Reaffirmed the hard way**; it cost this pass two arms as well. |
| §5 revisit condition (a): "the call-site profile shows `FLASH_ATTN_EXT` is the largest depth-proportional decode term" | **Met.** FA (34.77) > indexer (18.29), and nothing else is depth-proportional. Condition (b) is still open. |

---

## 5. Recommendation

**Do the cheap host fix first; it is worth more than both kernel projects combined.**

1. **Take expert-weight paging off the table, then re-measure.** `--load-mode none` (equivalently
   `--no-mmap`) makes the 52.8 GiB CPU-resident expert set anonymous RAM that cannot be evicted by an
   unrelated process, instead of page cache on a shared array. `PLAN.md` already lists this as "still
   untested and cheap", and the loader warns about the mmap/CPU-override combination on every run of this
   config. **Check headroom before trying it**: the box has 123 GiB with ~27 GiB in other services and swap
   currently exhausted, and the container is capped at `mem_limit: 90g`. Then re-run the 116K decode on a
   quiet array. **Expected: 240 -> ~110 ms/token, i.e. ~2.2x**, which is larger than anything in §2 and
   costs no code.
2. **Re-measure `d32768`/`d65536` with `llama-bench` on a quiet array** to close doc 10 §5's condition (b)
   properly. This pass could not, twice over. The component model (§2.4) predicts **18.1 tok/s at `d32768`
   and 14.3 at `d65536`**; if a quiet box reproduces those, the model is closed and the graph work below is
   correctly prioritized.
3. **Only then, graph work — and in this measured order at 116K:**
   - `FLASH_ATTN_EXT`, 34.77 ms/token. Doc 10 §4.2's Design B is unchanged and recovers ~33 ms of it.
   - The QSA indexer, 18.29 ms/token. **The natural target here is fusion, not a new kernel**:
     `ggml_lightning_indexer` (`ggml/include/ggml.h:2673`) already exists, is already implemented in this
     SYCL backend (`ggml/src/ggml-sycl/lightning-indexer.cpp`, dispatched at
     `ggml/src/ggml-sycl/ggml-sycl.cpp:5682-5684`), and is already used by five other DSA-style models
     (`src/models/{deepseek4,deepseek32,dots3note,glm-dsa,hy-v4}.cpp`). It fuses exactly the per-head dot
     product + relu + weighted head sum + mask add that `src/models/qwen4exp.cpp:825-857` builds from ~8
     primitive ops. qwen4exp's variant differs (it scores mean-pooled *blocks* and then expands to cells,
     and its head weights are uniform), so this is not a drop-in — but it is adapting existing, tested
     infrastructure rather than writing a new kernel, which is the cheaper and lower-risk half of the pair.
   - The ~2,800 nodes/token dispatch cost, ~12.6 ms/token, of which hyper-connections are ~970 nodes
     (§2.5.3). Out of scope here, but it is the third-largest term and it is pure launch overhead.

**Ceiling check, so the ordering is defensible.** Decomposing the measured 240.01 ms/token:

```
 40.46 ms  floor      (CPU-resident MoE + SYCL dispatch + host graph work, depth-independent)
 34.77 ms  FA         @ n_kv 118016, q4_0
 18.29 ms  indexer    @ n_kv 118016, q4_0
~16    ms  serving    (sampler chain + detokenize + SSE, from the 17.05 tok/s trivial-depth figure)
~130   ms  paging     (residual)
------
240.01 ms/token
```

| do this | ms/token | speedup |
|---|---:|---:|
| nothing (today) | 240.01 | 1.00x |
| kernel work only (perfect sparse FA **and** perfect indexer) | ~187 | **1.28x** |
| paging fix only (`--load-mode none`, quiet array) | ~110 | **2.18x** |
| both | ~57 | **4.2x** |

**Kernel work alone buys 1.28x; the flag buys 2.18x.** Doing the kernel work first would spend ~400-700
lines of correctness-critical SYCL chasing the smaller of two co-located costs — the exact mistake doc 10 §5
caught, one level up. Doing both is worth 4.2x, and the cheap half must come first so that the expensive
half can be measured at all.

---

## 6. Measurement hygiene note, fourth time in this project

`ps aux | grep llama` and `docker ps | grep sycl` were both clean before every run in this pass, and the
GPU was genuinely idle. It was not enough, again. The check that mattered was `iostat -x` on the device
holding the model, plus `/proc/<pid>/stat` for `D` state and major-fault rate. **For this box, a
long-context number is only trustworthy if the model's array is below ~50% utilization and the process's
RSS has reached ~53 GiB and stopped growing.** `staging/work/decode_depth_profile.sh` puts a `d2048` arm
first and last for exactly this reason; the leading arm reproduced doc 10's 23.54 tok/s at 23.66, which is
what established the shallow arms as usable, and the trailing control was never reached.

---

## 7. CORRECTION (2026-09-11, same day, later): §3.3 named the wrong tensor and §5 therefore named the wrong flag

**Status of this section: measured. §5's recommendation (`--load-mode none`, "worth ~2.2x") was acted on and
produced nothing, twice. The reason is that neither `-lm none` nor `-lm mlock` covers the tensor that is
actually being demand-paged. The residual §3 measured is real and its fault arithmetic was right; the
tensor it attributed the faults to was wrong. The flag that does cover it is `-lzm off` /
`--lazy-mode off`.**

### 7.1 The null result that forced this correction

`-lm none` and, separately, `-lm mlock` were run against the same 116K prompt at the same config
(`-ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -ncmoe 24 -ub 2048 -c 122880`). **Neither moved decode**: both landed
at ~4.3-4.8 tok/s against the `auto`/mmap baseline's 4.17, i.e. inside this benchmark's own band. §5
predicted 240 -> ~110 ms/token.

### 7.2 Why `-lm mlock` was a no-op, at two independent levels

1. **It could not lock anything.** The test container reports `ulimit -l` = **8192 kB = 8 MiB**, and Docker's
   default capability set excludes `CAP_IPC_LOCK`, so `mlock()` over a ~50 GiB buffer fails outright.
   `llama_mlock::impl::raw_lock` (`src/llama-mmap.cpp:689-719`) handles that by emitting
   `warning: failed to mlock ... Try increasing RLIMIT_MEMLOCK` and returning `false` — a soft degrade, so
   the run proceeds looking normal.
2. **Even a working mlock deliberately skips the tensor in question.** `load_all_data` locks mapped tensors
   only when they are *not* lazy (`src/llama-model-loader.cpp:1653-1654`), with the comment *"locking a lazy
   tensor would fault all of it in, which is what lazy avoids"*.

`-lm mlock` also sets `use_mmap = false` (`src/llama-model-loader.cpp:559` — only `mmap`, `mmap+mlock` and
`auto` enable mmap), so the expert weights were already unevictable anonymous/pinned host memory in a
cgroup with `memswap_limit == mem_limit` (swap disabled for the container). **Under both arms, the
page-cache eviction mechanism §3.3 described was already impossible for the expert weights — and decode did
not move.** That alone falsifies the expert-paging attribution.

### 7.3 What is actually being paged: the lazy PLE n-gram table, not the experts

§3.3 wrote "the ~52.8 GiB of `CPU_Mapped` expert tensors that `-ncmoe 24` requires
(19,240.51 + 6,130.57 + 27,465.95 MiB)". **The third of those three buffers is not experts.** From this
project's own `logs/long-context-120k/vram-check.log`, two lines above the buffer sizes it quoted:

```
load_tensors: loading model tensors, this can take a while... (load_mode = mmap)
add: tensor per_layer_token_embd.weight (size = 27465 MiB) lazy read enabled
load_tensors:   CPU_Mapped model buffer size = 19240.51 MiB
load_tensors:   CPU_Mapped model buffer size =  6130.57 MiB
load_tensors:        SYCL0 model buffer size = 26916.16 MiB
load_tensors:   CPU_Mapped model buffer size = 27465.95 MiB   <-- the lazy PLE table
```

So the real split at `-ncmoe 24` is **25,371 MiB of CPU-resident experts + 27,466 MiB of lazily-mmapped
per-layer-token-embedding table**, not 52.8 GiB of experts.

`qwen4exp` creates that table with `TENSOR_READ_LAZY` (`src/models/qwen4exp.cpp:192-193`), and
`llama_model_loader::lazy_read::add` keeps it lazy because it is over the 4 GiB `auto` threshold
(`src/llama-model-loader.cpp:1086-1091`). Two consequences, both code-level:

- **Lazy tensors keep an mmap no matter what `--load-mode` says.**
  `init_mappings` is entered on `if (use_mmap || lazy.any())`, carrying the comment *"read_lazy also
  requires mmap; this condition make sure it's usable even when --load-mode is not set to mmap"*
  (`src/llama-model-loader.cpp:1403-1405`).
- **`-lm none` makes it strictly worse than `auto`, not neutral.** The same function computes
  `prefetch_size = prefetch && use_mmap ? -1 : 0` (`:1422`). With `-lm none`, `use_mmap == false`, so the
  lazy mapping is created with **no `MADV_WILLNEED` advice at all**, where `auto` at least advises the whole
  file.

**The access pattern is the worst case for demand paging, by construction.** The GGUF carries
`ple.ngram_size = 3`, `ple.heads_per_ngram = 8` → `ple_n_heads = (3-1)*8 = **16`, and
`ple.head_vocab_sizes` = 16 x ~20,000,003 → **~320 M rows over 27,465 MiB ≈ 90 bytes/row**. Per token,
`llm_graph_input_ple::set_input` computes 16 indices as
`idx = mixed % ple_head_vocab_sizes[h] + ple_head_offsets[h]`, where `mixed` is a multiplicative-XOR **hash**
of the n-gram (`src/models/qwen4exp.cpp:1294-1306`), and `:1381` gathers them:
`ggml_get_rows(ctx0, model.per_layer_tok_embd, rows)`. The source comment at `:1220` says it outright:
*"PLE n-gram hash embedding: each token gathers `ple_n_heads` rows of a shared table."*

So **every token touches 16 uniformly-random 4 KiB pages of a 28.8 GB file to consume 16 x 90 = 1,440
bytes — a 45x read amplification, with zero locality available even on repeated text, because the index is
a hash.** There is no working set to warm: the hot set over a 116K prompt is 116,277 x 16 ≈ 1.86 M distinct
pages ≈ 7.4 GiB of page traffic for 167 MB of data.

**This reconciles §3's arithmetic rather than overturning it.** §3.3 computed the residual as "~11 faults
per token, i.e. ~45 KB/token". Sixteen hash-random row gathers per token is **~11 effective faults/token
after partial cache hits, ~64 KiB of page traffic** — the same number, now with a mechanism that is per-token
by construction instead of an inferred property of expert routing.

### 7.4 The live evidence, both directions

Measured on a box confirmed idle first (`nproc` 32; `top` 90.2% idle with the loudest tenants
`rustdesk` 110% + `java` 70% + ~40% of miscellany ≈ 220% of 3200% = **6.9% of capacity**; `free`
**available 95 GiB of 123 GiB**; `numactl --hardware` reports **1 NUMA node**, so the NUMA hypothesis is
structurally impossible on this hardware and was dropped):

| condition | process state | majflt/s | CPU | model array |
|---|---|---:|---|---|
| `-lm none`, lazy still on (`llama-bench -d 32768`) | **D** (uninterruptible) | **80-100** | ~0% | `bcache0` 100% util |
| `-lzm off` (116K `llama-server` prefill) | **R** (running) | **0.0** | 70% user + 30% sys | `bcache0` ~70% (ambient) |

The `-lm none` arm sat in `D` state at 80-100 major faults/s for **8 minutes** with its CPU time frozen,
reproducing §3.3's "RSS growing at 0.36 MB/s" to the decimal — which retires §3.3's reading of that number
as expert weights still loading. It was the PLE table faulting, and `-lm none` had just removed its only
prefetch advice.

**Disk-stall as a *first-order* explanation is also dead, and the arithmetic is the reason.** Sequentially
this array does ~470 MB/s (measured during model load: `bcache0` 469,972 kB/s). The PLE pattern gets
**73 MB/s at 156 IOPS with `r_await` 13.2 ms**, because ~90-byte random reads spread over 28.8 GB miss the
NVMe cache and land on `md0` (the HDD array, 98.5% util, `r_await` 15.6 ms). So the array is not slow and
external contention is not the driver — **the access pattern is**, and the fix is to stop issuing it rather
than to make the disk faster.

### 7.5 Measured effect of the right flag: `-lzm off`

`-lzm off` (`--lazy-mode off`, "always read the whole tensor up front") folds the PLE table into resident
host memory. Confirmed from the load banner of `logs/long-context-120k-lazyoff/server.log` — note the
**absence** of any `lazy read enabled` line and the single fused host buffer:

```
load_tensors: loading model tensors, this can take a while... (load_mode = none)
load_tensors:        SYCL0 model buffer size = 26916.16 MiB
load_tensors:    SYCL_Host model buffer size = 51238.27 MiB     (= 23,772 experts + 27,466 PLE)
```

(Incidentally this resolves a puzzle from the `-lm none` arm: without mmap the host weights land in
`SYCL_Host` **pinned USM**, which is not accounted in the process's `RssAnon` — the earlier run showed
`RssAnon` = 767 MB while holding ~50 GiB of weights. Judge residency from the load banner and box-wide
`used`, not from `/proc/<pid>/status`.)

**Prefill, same prompt / config / server-side timers as `logs/long-context-120k-ub2048/`:**

| | baseline (`-lm auto`, lazy on) | `-lm none -lzm off` |
|---|---:|---:|
| first chunk | 42 tok in 16.41 s, then 102.0 tok/s marginal | **2,090 tok in 5.31 s (393.4 tok/s)** |
| marginal rate, 16K-35K | 226-338 tok/s | **470-525 tok/s** |
| cumulative @ ~10K | 181.8 tok/s | **478.5 tok/s** |

The warm-up penalty this project has been discarding as "the first arm" (PLAN.md's page-cache rule, and the
116K report's "first post-load request measured ~5x slower") **is the PLE table faulting in, and `-lzm off`
removes it**, not just amortises it.

### 7.6 Decode: the headline, and §2.4's cost model is now closed

Same prompt (116,277 tokens), same `-ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -ncmoe 24 -ub 2048 -c 122880`, same
client, same server-side `timings` block as `logs/long-context-120k-ub2048/`. Artifacts in
`logs/long-context-120k-lazyoff/`; driver `staging/work/run_116k_lazy.sh`, host sampler
`staging/work/host_sampler.sh`.

| | baseline (`-lm auto`, lazy on) | `-lm none -lzm off` | ratio |
|---|---:|---:|---:|
| prefill | 240.15 tok/s | **394.14 tok/s** | **1.64x** |
| TTFT | 484.19 s (8 min 04 s) | **295.01 s (4 min 55 s)** | **1.64x** |
| **decode** | **240.01 ms/token (4.167 tok/s)** | **89.72 ms/token (11.146 tok/s)** | **2.68x** |
| `tg_3s` spread over the decode window | 3.09x (min 3.05, p50 3.93, max 9.41, n=145) | **1.03x (min 10.98, p50 11.25, max 11.29, n=9)** | — |
| major faults/s during decode | 80-100 | **0.0** | — |

Three things follow, and together they close this document.

1. **§5's magnitude was right and its flag was wrong.** It predicted "240 -> ~110 ms/token, i.e. ~2.2x" from
   removing the paging exposure. Removing it via the flag that actually covers the paged tensor gives
   **89.72 ms/token, 2.68x** — better than predicted.
2. **§2.4's component model is closed to 4.1%, and the separately-estimated serving term was not real.**
   The model predicted `40.46 (floor) + 34.77 (FA) + 18.29 (indexer) = 93.52 ms/token`; the real server now
   measures **89.72**. So the graph model explains essentially the whole token, and §3.1's "~16 ms/token
   serving path" must **not** be added on top of it — that figure came from comparing `llama-server` at
   trivial depth (17.05 tok/s) against `llama-bench`, and was itself mostly PLE fault time rather than
   sampler/detokenize/SSE cost.
3. **§3.2's interval spread was the load-bearing evidence, and it was the right instrument pointed at the
   right phenomenon.** "GPU kernel time at a fixed shape does not vary 3-7x between adjacent three-second
   windows. A host-side stall backed by a shared disk does." Correct — and with the stall removed the spread
   collapses from **3.09x to 1.03x**. A cost that is "sometimes absent" is exactly what a hash-indexed
   gather against a partially-cached 28.8 GB table produces.

**Revised ceiling table, replacing §5's.** The paging term is gone, so the kernel work is no longer hiding
behind a larger co-located cost — it is now the whole remaining token:

```
 40.46 ms  floor      (CPU-resident MoE + SYCL dispatch + host graph work, depth-independent)
 34.77 ms  FA         @ n_kv 118016, q4_0
 18.29 ms  indexer    @ n_kv 118016, q4_0
------
 93.52 ms  predicted   vs 89.72 ms measured  (-4.1%)
```

| do this | ms/token | speedup vs. today's 89.72 |
|---|---:|---:|
| nothing (post-`-lzm off`) | 89.72 | 1.00x |
| doc 10 §4.2 Design B (sparse FA, recovers ~33.6) | ~56 | **1.60x** |
| + the indexer (fusion onto `ggml_lightning_indexer`) | ~38 | **2.37x** |

**So §5's ordering is now inverted in the good direction: the kernel work it deferred is worth 2.37x, not
1.28x, because the term that was dwarfing it is fixed.** FA remains the larger half and Design B remains
unchanged.

### 7.7 Cost of the fix, and what is still open

- **RAM.** `-lzm off` moves 27,466 MiB from lazy mmap into resident host memory: the host model buffer goes
  **23,772 -> 51,238 MiB**. Box-wide `used` went 27 -> 77 GiB, `available` 95 -> 45 GiB. That fits here, but
  it is a real new line item for `docs/research/08`'s 300-600K budget, which never counted the PLE table as
  resident.
- **Load time.** 59 s (mmap) -> **~4.9 min** (`-lm none -lzm off`, reading 76.32 GiB at ~300-470 MB/s).
  Paid once per server start.
- **Not separated: `-lm none` vs `-lzm off`.** Both were set in this run. `-lm none` is known-neutral on its
  own (§7.1) and is code-provably neutral-to-harmful for the lazy tensor (§7.3), so the win is attributable
  to `-lzm off` — but a clean `-lm auto -lzm off` arm was not run and should be, since `auto` would keep
  mmap for the experts and avoid ~24 GiB of the anonymous-RAM cost.
- **n=1, and a 400-token decode window** against the baseline's 2,000. The `tg_3s` spread of 1.03x over 9
  intervals makes a longer window unlikely to change the number, but it is not the same window.
- **Prefill did not improve as much as decode** (1.64x vs 2.68x), which is expected: a prefill ubatch gathers
  2048 x 16 = 32,768 PLE rows in one op across 16 threads, so the faults issue concurrently and the array's
  queue depth hides much of the latency. Decode gathers 16 rows with nothing to overlap them against.
- **`MUL_MAT_ID iq2_s` shows 13-14 ms of *host dispatch* time per call in mode 2**, which never syncs. That
  means the SYCL `MUL_MAT_ID` path host-serializes (queue drain inside the op). Noted, not chased — it is
  inside the 40.46 ms floor, not a new term, but it is a lead for whoever attacks the floor.
- **`set_input_qsa` is O(n_kv) host work per ubatch** and carries its own TODO saying so ("about 865 us at
  33k context", `src/llama-memory-hybrid-idx.cpp:301-302`) — ~3.1 ms/token at 118016, once per unique
  compress ratio, and this model has exactly one (`attention.compress_ratios` is `[0,0,0,4]` repeating, so
  all 12 QSA layers share `r = 4` and `qsa_inps` holds a single entry). Small, real, inside the floor —
  but note it is depth-proportional host work that §2.5's "flat with depth" reading could not see at
  d2048/d8192, where the same scan costs ~54 us.

### 7.8 Measurement hygiene, fifth time, and the rule this changes

§6's rule ("a long-context number is only trustworthy if the model's array is below ~50% utilization and the
process's RSS has reached ~53 GiB and stopped growing") is **wrong in both halves and should be retired**:

- **RSS is not the right gauge.** Without mmap the host weights land in `SYCL_Host` pinned USM, which does
  not appear in `/proc/<pid>/status` `RssAnon` at all — a run holding ~50 GiB of weights showed `RssAnon` =
  767 MB. Judge residency from the `load_tensors:` banner and box-wide `used`.
- **Array utilization is a symptom, not a cause.** `bcache0` read 100% util in the bad runs because *we*
  were issuing 90-byte random reads into it, not because a neighbour was saturating it. Sequentially the
  same array does ~470 MB/s.

**The replacement rule is one line: before trusting any long-context number on this model, check the load
banner for `lazy read enabled` and check `majflt/s` on the process. If the first is present and the second
is non-zero during decode, the number is measuring disk, not the model.**

---

## FOLLOW-ON (2026-09-11, later): the component model was extended into curves, and it holds

**See `docs/research/12-decode-depth-scaling.md`.** §2's component model (floor + FA + indexer) was built
at 116K and validated at two shallow depths. It has now been measured at nine depths out to
`n_kv = 614400` and end-to-end at six, and it survives:

- `llama-bench` on the recommended config measures **26.48 / 26.03 / 24.63 / 19.43 / 15.26 / 11.22 tok/s**
  at `d0 / d2048 / d8192 / d32768 / d65536 / d118016`. The model reproduces every one of the four deepest
  to within **0.8%**, and the `d118016` point agrees with the real server benchmark's 11.146 tok/s.
- **§2.5's depth-independent floor is confirmed depth-independent**: the implied floor is
  **35.8-37.5 ms/token** across that whole range, with no trend. Every millisecond between `d0` and 116K is
  accounted for by the FA and indexer curves.
- §2.1 and §2.2's single-depth numbers reproduce on a different day to **0.5%** and **0.3%**.
- §2.2's side finding that `q4_0` makes the indexer *faster* is explained by doc 12 §2's byte model: the
  gather at `qwen4exp.cpp:789` reads 72 B/row instead of 256, and that gather is inside the 94.3% of
  indexer traffic that is pure recomputation of block-pooled keys.
- The `test_qsa_indexer` case this document added has gained a `pooled_cached` arm (doc 12 §4.1) that
  measures what the chain would cost with those keys cached: **1 528.56 -> 217.51 us at `n_kv = 118016`.**

The one thing doc 12 changes about this document's conclusions is emphasis. §5's costed table is framed
around multipliers at 116K; under a slope requirement the multipliers matter less than the fact that both
remaining terms are **linear in depth**, and that the `-lzm off` win — real and large — did nothing to the
slope.
