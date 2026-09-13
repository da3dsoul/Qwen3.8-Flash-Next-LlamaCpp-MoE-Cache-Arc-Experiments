# Decode Cost vs. Context Depth — the Slope, Not the Point

**Date: 2026-09-11 (late).** Measurements on this box's Arc Pro B70, GPU idle and exclusively
held for every number below.

> **Read section 9 and section 10 before acting on section 4-6.** Section 9 (2026-09-11) implements
> section 5.3's block top-k. Section 10 (2026-09-12) implements section 5.1's pooled-key cache and
> **retires section 9.4's program table**: the real implementation reaches 7.4x on the indexer
> term, not the 40.8x the isolated arm implied, and lands at **+15.0% at 300K / +30.5% at 600K**
> once sparse FA is added, not +3.3% / +6.1%.

All source changes in sections 1-8 are **test-only**, in
`tests/test-backend-ops.cpp`: extra depths in the two existing qwen4exp perf loops, and a
`pooled_cached` variant of `test_qsa_indexer` (§4.1) registered in both the perf and the
correctness suites. No model, kernel or memory-module code was touched.

One caveat on the end-to-end numbers, which does **not** touch §1-§4: the 300K configuration used
for `logs/long-context-300k/` turned out to produce degenerate output (cause isolated to
`--rope-scaling yarn`, see `logs/long-context-300k-benchmark-report.md` §4.1.1-4.1.3). All
component curves here were measured by `test-backend-ops` with no model loaded and are unaffected,
as is the `llama-bench` curve in §8, which ran without rope scaling.

Binaries: `staging/devbin` as built at 20:36, i.e. the same `libllama` (09:46) and
`libggml-sycl` (13:50) that today's 116K `-lzm off` run used, so every number here is directly
comparable to `logs/long-context-120k-lazyoff/`. A separate workstream began editing
`src/models/qwen4exp.cpp` and `src/llama-memory-hybrid-idx.cpp` at 20:48, after all of these
measurements were taken; nothing here reflects those edits.

## 0. Why this document exists

Every performance number this project owns is a **point** measurement. `docs/research/10` measured
`FLASH_ATTN_EXT` at one depth and declined sparse attention on the strength of the resulting single
multiplier. `docs/research/11` built a component model at one depth (116K) and validated it at two
shallow ones. The `-lzm off` fix took 116K decode from 4.17 to 11.15 tok/s and the project treated
that as the answer.

It is not the answer, because the requirement is a **slope**: decode throughput must stay within
**5% at 300K** and **25% at 600K** of its short-context value. Against that bar, 11.15 tok/s at
116K is a *failure*, not a win — short-context decode on the same config is 22-24 tok/s, so the
current stack loses ~half its speed by 100K. This document measures the slope of every
depth-proportional term, decomposes the dominant one down to bytes moved, and says what would have
to change to meet the bar.

**Verdicts up front:**

1. **Both depth-proportional terms are exactly linear in `n_kv`, and neither is small.** Dense
   `FLASH_ATTN_EXT` costs **0.0243 us per KV token per layer**; the QSA lightning-indexer chain
   costs **0.0131 us**, i.e. **54% as much as attention itself**. Over 12 QSA layers the combined
   slope is **0.448 ms per 1,000 tokens of depth**. §1.
2. **The indexer is the reason sparse attention alone cannot meet the bar.** With a perfect sparse
   FA and today's indexer, 300K decode is still **+120%** slower than 8K. `docs/research/10`'s
   "declined" verdict was right about Design B's value in isolation and wrong about the framing:
   the two fixes are **not substitutes, they are a matched pair**, and either one alone leaves a
   linear term in place. §4.
3. **94.3% of the indexer's cost is recomputing something that did not change.** The chain is
   memory-bandwidth-bound at a flat **284-297 GB/s across a 19x depth range**, and a byte-level
   decomposition of `build_qsa_top_k` shows **3,536 of 3,749 B/token/layer** is the re-derivation
   of block-pooled indexer keys — a function of the KV cache and block positions *only*, identical
   between consecutive tokens except for the single block that just closed. §2, §3.
4. **The prize of caching those keys is measured, not assumed, and it is 7-8x rather than the
   17.6x the byte model implied.** A `pooled_cached` arm of the same test graph -- correctness-gated
   against the CPU reference at `max_err = 0.0` -- runs the chain with the block keys handed in
   ready: **1 528.56 -> 217.51 us at 118K, 4 043.51 -> 534.41 at 300K.** The shortfall against the
   byte model is itself informative: the residual chain only reaches **122-131 GB/s** against the
   streaming pool's 285, so it is launch-bound and has roughly another 2x in fusion. §4.1.
5. **The bar is reachable, but only with the whole program, and 300K is the binding end.** Sparse FA
   plus the pooled-key cache lands at **+14.9% at 300K / +29.8% at 600K** -- a 24x slope improvement
   that still misses both bars. Adding block-level top-k and an f16 pooled cache gives
   **+7.3% / +14.6%**: 600K inside, 300K 2.3 points outside. Fusing the residual chain (estimated)
   gives **+3.7% / +7.3%**, inside both. Nothing requires retraining or a model change. §5, §6.
6. **"Flat" is the wrong frame, and 300K is harder than 600K.** No attention architecture has
   depth-independent decode; the target is a slope small enough to hide under the per-token floor.
   Because the bar is *relative*, a ~37 ms floor allows only 1.9 ms of depth cost at 300K but 9.3 ms
   at 600K, so every candidate design clears 600K before it clears 300K. §4.3.
7. **A caution that falls out of the same arithmetic: the MoE-cache workstream makes this harder.**
   Shrinking the floor raises the *relative* degradation for a fixed slope. §7.

---

## 1. The two curves

`test-backend-ops perf`, SYCL0, production decode shape. FA: `hsk=hsv=256, nh=2, nr23=[12,1],
nb=1`, `-ctk/-ctv q4_0`. Indexer: the whole of `build_qsa_top_k` as one graph (`test_qsa_indexer`,
added by `docs/research/11`), `type_k=q4_0`, `n_tps=n_stream=1`, `width=2051`. Both timed per call,
i.e. per QSA layer; a decode token pays each 12 times.

| `n_kv` | FA `q4_0` us/call | FA ms/token | indexer `q4_0` us/call | indexer ms/token | indexer as % of FA |
|---|---|---|---|---|---|
| 2 048 | 55.47 | 0.67 | — (n_kv < width) | — | — |
| 4 096 | 100.25 | 1.20 | 108.37 | 1.30 | 108% |
| 8 192 | 178.97 | 2.15 | 152.35 | 1.83 | 85% |
| 16 384 | 398.66 | 4.78 | 243.64 | 2.92 | 61% |
| 32 768 | 807.59 | 9.69 | 432.72 | 5.19 | 54% |
| 65 536 | 1 614.66 | 19.38 | 827.27 | 9.93 | 51% |
| **118 016** | **2 913.43** | **34.96** | **1 528.56** | **18.34** | **52%** |
| 200 704 | 4 917.78 | 59.01 | 2 647.11 | 31.77 | 54% |
| **307 200** | **7 488.28** | **89.86** | **4 043.51** | **48.52** | **54%** |
| **614 400** | **14 964.10** | **179.57** | **8 011.00** | **96.13** | **54%** |

Logs: `logs/decode-scaling/fa-depth-curve.log`, `logs/decode-scaling/qsa-indexer-depth-curve.log`.

Reproduction check against `docs/research/11` §2.1-2.2 (different day, same tree): FA at 118 016
**2 913.43 vs 2 897.77 us** (0.5%), indexer **1 528.56 vs 1 524.24 us** (0.3%).

Least-squares fits over the `n_kv >= 32768` points:

```
FLASH_ATTN_EXT   0.024317 us per KV token per layer   (+25.7 us fixed)
QSA indexer      0.013068 us per KV token per layer   (-0.5 us fixed)
combined, x12    0.448 ms per 1,000 tokens of depth
```

Both are straight lines to within 1% over a 19x range. **There is no knee, no plateau, and no
sub-linear regime anywhere in the range that matters.** The f16 KV variants are on the same lines
with different constants (FA 1.63x cheaper, indexer 1.13x *more* expensive — the two knobs pull
opposite ways, as `docs/research/11` §2.2 found).

### 1.1 The sparse-restricted reference point

QSA's budget is `indexer_top_k + r - 1 = 2051` cells, which a real sparse FA would pad to ~2 304.
The `n_kv = 2048` row above is therefore a direct proxy for what one **bounded** FA call costs:
**55.47 us, i.e. 0.67 ms/token, flat with depth**. This is a proxy measured on the *dense* kernel at
a small `n_kv`, not a measurement of a sparse kernel — SYCL still ignores the sparse hint entirely
(`docs/research/10` §2), so no sparse kernel exists here to time. It is the right proxy because
Design B's whole construction is "hand the unmodified kernel a compacted K/V of `width` rows".

---

## 2. The indexer is memory-bound, and the model that proves it

`build_qsa_top_k` (`src/models/qwen4exp.cpp:721-869`) was read op by op and every node's traffic
counted, with `n_kv = D`, `idx_dim = 128`, `r = 4`, `n_blocks = D/4`, one token, one stream.

**Group A — derives block-pooled keys. Function of the KV cache contents and the block positions
only.**

| node | what | bytes/layer/token |
|---|---|---|
| `ggml_get_rows(k_all, blk_cells)` `:789` | gather all `D` cell keys out of the `q4_0` indexer cache | 72·D read + 4·D index + 512·D f32 write = **588·D** |
| 4x `ggml_cont(ggml_view_3d(...))` `:794-797` | the `r` strided member slices | **1 024·D** |
| 3x `ggml_add` `:798` | sum the slices | **1 152·D** |
| `ggml_scale` `:800` | divide by `r` | **256·D** |
| `build_norm` RMS `:805` | normalize pooled keys | **256·D** |
| `ggml_rope_multi` `:808` | rotate pooled keys by `blk_pos` | **260·D** |
| | **subtotal** | **3 536·D** |

**Group B — depends on this token's query. Must run every token.**

| node | what | bytes/layer/token |
|---|---|---|
| `ggml_mul_mat(pooled, q)` `:820` | score every block against 4 query heads | **132·D** |
| `ggml_relu` + 4-head sum + block bias `:823-836` | | **23·D** |
| `ggml_get_rows` + 2 permute-`cont` `:839-841` | expand `D/4` block scores to `D` cells | **28·D** |
| mask cast + `ggml_add` `:843-848` | add the f16 KQ mask over all `D` cells | **18·D** |
| `ggml_top_k(expanded, 2051)` `:853` | | **12·D** |
| | **subtotal** | **213·D** |

Total **3 749·D bytes per layer per token**, of which **Group A is 94.3%**.

Against the measured times, that is a flat effective bandwidth:

| `n_kv` | modelled MB | measured us | effective GB/s |
|---|---|---|---|
| 8 192 | 30.7 | 152.35 | 202 |
| 16 384 | 61.4 | 243.64 | 252 |
| 32 768 | 122.8 | 432.72 | 284 |
| 65 536 | 245.7 | 827.27 | 297 |
| 118 016 | 442.4 | 1 528.56 | **290** |
| 200 704 | 752.4 | 2 647.11 | **284** |
| 307 200 | 1 151.7 | 4 043.51 | **285** |
| 614 400 | 2 303.4 | 8 011.00 | **288** |

**284-297 GB/s, flat from 32K to 614K.** Below 32K the curve rolls off into launch overhead, which
is why the chain looks super-linear at the shallow end. Predicted vs. measured chain time at
118 016: **18.63 vs 18.34 ms/token** (1.6%); at 307 200: **48.49 vs 48.52** (0.06%).

> **The indexer is not compute-bound, not launch-bound, and not doing anything clever. It is a
> streaming read of the entire indexer KV cache, re-pooled from scratch, on every QSA layer of every
> decode token.** At 307 200 tokens that is **1.15 GB per layer, 13.8 GB per token, 12 times per
> token at ~285 GB/s.**

A per-node SYCL profile was also taken (`GGML_SYCL_OP_PROFILE=1`,
`logs/decode-scaling/qsa-indexer-opprofile.log`) and agrees on the *shape* — `CONT` 39%, `GET_ROWS`
24%, `ADD` 18% at `n_kv = 307200`, i.e. Group A — but mode 1 drains the queue around every node, so
its absolute per-node numbers are floor-limited by sync latency and are not used quantitatively
here. The byte model is the quantitative instrument; the profile only corroborates which nodes it
points at.

---

## 3. So: is the indexer itself the hidden scaling problem?

**Yes, in the sense that matters, and no, in the sense that was feared.**

- It is **not** bigger than attention — it is 54% of dense FA, exactly as `docs/research/11`
  measured, and `docs/research/10`'s 25-100 ms/token estimate stays retired.
- But it is **linear with a slope 54% of attention's**, and that is fatal to the bar *by itself*.
  Removing dense attention entirely and leaving the indexer alone still leaves a term that grows
  0.157 ms per 1,000 tokens — **48.5 ms/token at 300K**, against a floor of ~36 ms.
- The mechanism is the worst possible one for caching **as written** and the best possible one for
  caching **in principle**: the expensive 94.3% depends on nothing that changes between tokens. It
  is recomputed only because the graph is rebuilt per ubatch and has nowhere to keep it.

---

## 4. Redoing `docs/research/10`'s verdict as curves

`docs/research/10` §4.2 declined Design B (sparse FA by gather-dequant into a compacted `width`-row
K/V buffer) because at 116K it was worth ~1.60x after the `-lzm off` fix — real, but not obviously
worth ~360 LOC of MEDIUM-risk SYCL. Under a slope requirement the same numbers read differently,
and *also* say the opposite of what a single-point reading would suggest about doing it alone.

Model: `ms/token = floor + FA(D) + indexer(D)`, floor measured (§8), FA and indexer measured at
every depth in the table (§1), the pooled-key-cache arm measured directly (§4.1). Script:
`staging/work/decode_scaling_model.py`.

| scenario | 8K | 32K | 65K | 118K | 200K | 307K | 614K |
|---|---|---|---|---|---|---|---|
| **today** | 24.8 t/s | 19.5 (+27.1%) | 15.2 (+62.9%) | 11.2 (+122.4%) | 7.9 (+215.4%) | 5.7 (**+333.6%**) | 3.2 (**+674.4%**) |
| sparse FA only (doc 10 Design B) | 25.8 | 23.7 (+8.7%) | 21.3 (+20.9%) | 18.1 (+42.6%) | 14.5 (+77.1%) | 11.7 (**+120.3%**) | 7.5 (**+243.0%**) |
| indexer pooled-key cache only | 25.5 | 21.1 (+20.6%) | 17.4 (+46.5%) | 13.5 (+88.3%) | 10.0 (+153.8%) | 7.5 (**+237.9%**) | 4.4 (**+480.9%**) |
| **both** | 26.5 | 26.1 (+1.4%) | 25.8 (+2.7%) | 25.3 (+4.9%) | 24.2 (+9.3%) | 23.0 (**+14.9%**) | 20.4 (**+29.8%**) |
| **both + block top-k + f16 pool** | 26.8 | 26.6 (+0.7%) | 26.4 (+1.3%) | 26.1 (+2.4%) | 25.6 (+4.5%) | 24.9 (**+7.3%**) | 23.4 (**+14.6%**) |
| ... + fused residual chain (**est.**) | 26.9 | 26.8 (+0.4%) | 26.7 (+0.7%) | 26.6 (+1.2%) | 26.3 (+2.3%) | 25.9 (**+3.7%**) | 25.1 (**+7.3%**) |

(Percentages are slowdown in ms/token vs. the same row's 8K column. Bar: <=5% at 300K, <=25% at
600K. Only the last row contains an unmeasured factor; every other cell is built from components
measured at that exact depth.)

### 4.1 The pooled-key cache arm is measured, not assumed

`test_qsa_indexer` gained a `pooled_cached` flag: the same chain with the block-key derivation
(Group A) replaced by a ready input tensor, i.e. exactly the cost the chain would have if those keys
were cached and updated incrementally. It is registered as a correctness case too and **passes
against the CPU reference at `max_err = 0.0`**, which is what makes the Group A / Group B split a
verified factorization of the graph rather than a reading of it.

| `n_kv` | full chain us | pooled-cached us | reduction | Group B effective GB/s |
|---|---|---|---|---|
| 4 096 | 108.37 | 55.29 | 1.96x | 15.8 |
| 8 192 | 152.35 | 64.34 | 2.37x | 27.1 |
| 32 768 | 432.72 | 109.09 | 3.97x | 64.0 |
| 65 536 | 827.27 | 148.57 | 5.57x | 94.0 |
| 118 016 | 1 528.56 | 217.51 | **7.03x** | 115.6 |
| 200 704 | 2 647.11 | 355.40 | 7.45x | 120.3 |
| **307 200** | 4 043.51 | **534.41** | **7.57x** | 122.4 |
| **614 400** | 8 011.00 | **1 000.96** | **8.00x** | 130.7 |

**Slope drops 8.5x** (0.013068 -> 0.0015334 us per KV token per layer). Log:
`logs/decode-scaling/qsa-indexer-cached.log`.

**This measurement corrected the estimate it was run to check, and in the pessimistic direction.**
§2's byte model said Group B is 213 of 3 749 B/token/layer, so a naive reading predicted a **17.6x**
reduction. The measured reduction is **7-8x**, because Group B does not achieve the streaming
chain's bandwidth: **122-131 GB/s against 285 GB/s**, on tensors 4x smaller spread over ~9 separate
small kernels. §6's flagged soft spot was real and is now closed with a number. It is also the
origin of the "fused residual chain" row above: an op running at 43% of the bandwidth its sibling
achieves, on the same data, in the same graph, is launch- and latency-bound, and fusing it is
ordinary work with roughly 2x in it.

### 4.2 What this changes about doc 10's recommendation

- Its verdict "leave `qwen4exp.cpp:943-946` as it is" remains correct *as a standalone decision*:
  sparse FA alone comes to **+120% at 300K against a 5% bar**, i.e. 24x the allowed excess.
  Building only Design B would not deliver what is being asked for, and doc 10 was right not to
  build it on a single-point multiplier.
- But its framing — kernel work as an optional multiplier — is wrong under a slope requirement.
  **Design B is a necessary half of the only combination that meets the bar.** Neither half alone
  gets within 20x of it. Together they reach **+14.9% at 300K**, which is still 3x the allowance;
  the two cheap in-graph follow-ons (§5.3) bring it to **+7.3%**, 1.5x the allowance; the fusion
  step clears it.
- Its Design A vs. Design B comparison is unaffected and B is still the right one: it needs no
  edits to the FA kernel, and it frees the 242 MiB f16 staging buffer whose re-dequantization of
  the whole cache per call is itself **38.7% of FA's `q4_0` cost** — reconfirmed by §1's own pair at
  118 016 (f16 1 785.77 us vs `q4_0` 2 913.43, 1.63x), so doc 10 §2.2's 1.6x tax reproduces.

### 4.3 300K is the binding constraint, not 600K

Counter-intuitively the deeper target is the easier one. The bar is relative, so at a ~37 ms floor
"5% at 300K" allows **1.9 ms** of depth cost while "25% at 600K" allows **9.3 ms** — the allowance
grows 5x while the cost only doubles. Every row of §4's table crosses the 600K bar before it crosses
the 300K one. **Any plan should be sized against 300K**, and a design that lands at "600K is fine,
300K is marginal" is behaving exactly as this arithmetic predicts rather than failing oddly.

---

## 5. What "close to flat" actually requires

Three changes, in dependency order. All three are inference-time only — no retraining, no model
surgery, no change to what the model computes (modulo top-k tie-breaking, which this project already
accepts and documented at `PLAN.md`'s top-k update).

### 5.1 Incremental pooled-key cache for the indexer — the biggest win, and backend-agnostic

Keep the post-norm, post-RoPE pooled block keys (`indexer_k`, `qwen4exp.cpp:810`) in a persistent
per-layer buffer sized `ceil(n_ctx/r)` instead of rebuilding them from the whole cache every token.
Per token only the block that just closed needs writing, plus the trailing partial block recomputed
(1 block, `r = 4` rows). Group A's 3 536·D collapses to O(r); Group B's 213·D remains.

- Cost: **236 MiB VRAM at 300K, 472 MiB at 600K** (f32; half that at f16), 12 layers x `idx_dim`
  128 x `n_ctx/4` blocks.
- Value, **measured** (§4.1, not modelled): indexer **48.5 -> 6.4 ms/token at 300K**,
  **96.1 -> 12.0 at 600K**; slope 8.5x shallower.
- Where: `llama_memory_hybrid_idx` gains a third cache next to the attention and indexer caches
  (`src/llama-memory-hybrid-idx.cpp:47-63`), `build_qsa_top_k` reads it instead of re-pooling, and
  the graph writes only new blocks. **The per-layer ratio is uniform on this model, which makes the
  shape trivial**: the GGUF's `qwen4exp.attention.compress_ratios` is
  `[0,0,0,4, 0,0,0,4, ...]` over 48 layers (`logs/long-context-120k-lazyoff/server.log`, kv 34) —
  `r = 4` on exactly the 12 QSA layers and 0 on the 36 gated-DeltaNet layers, so one buffer of
  `n_ctx/4` blocks per QSA layer, no per-ratio multiplicity. (`build_qsa_top_k` reads
  `hparams.dsv4_compress_ratios[il]` per layer and the existing `qsa_inps` map already keys inputs
  by `r`, so a non-uniform model would still be expressible.)
- Risk: **MEDIUM-HIGH.** It is a memory-module change, and correctness depends on getting
  invalidation right (sequence removal, defrag, the partial trailing block, multi-stream block
  membership). It is gated by an existing exact-equivalence test: `test_qsa_indexer` already
  demands the CPU reference's answer at `max_err = 0.0`.
- Estimate **350-500 LOC**. **Not SYCL-specific** — CUDA, Metal and CPU all pay this same
  recomputation, so this is the one piece of the program that is plausibly an upstream contribution
  rather than a local patch.

### 5.2 Sparse FA, `docs/research/10`'s Design B

Unchanged from doc 10 §4.2: gather-dequant only the `width = 2051` selected rows (padded to 2 304)
into a compact K/V buffer and hand the unmodified TILE/oneDNN kernel an effective `ne11`.
**~360 LOC SYCL, MEDIUM risk, zero FA-kernel edits.** Value against §1's curve: FA
**89.9 -> 0.67 ms/token at 300K** and flat thereafter; it also deletes the whole-cache dequant
staging pass, which is why the win exceeds the naive `2304/307200` ratio.

Prerequisite already satisfied: `qwen4exp.cpp:943-946`'s hint is a no-op on every backend we run,
so nothing needs un-commenting until the kernel side exists.

### 5.3 Two cheap follow-ons, worth 7.6 points at 300K

Both live entirely inside `build_qsa_top_k` and together cut Group B from 213·D to ~103.5·D, which
takes 300K from **+14.9% to +7.3%** and 600K from **+29.8% to +14.6%** (§4). For ~150 LOC they are
the best value in the program:

- **Top-k over blocks, not cells.** The code's own comment at `:838` says the budget is whole
  blocks; it nevertheless expands `D/4` block scores to `D` cells (`:839-841`), adds a `D`-wide
  mask, and runs `ggml_top_k` over `D`. Doing the top-k at block granularity (`width/r = 513`
  blocks) and expanding only the winners removes 4x of the tail. The `D`-wide mask add is the one
  subtlety: at decode with causal attention the block bias already carries the visibility test
  (`blk_bias`, `:750-754`), so the per-cell mask only matters for the trailing partial block.
  ~100-150 LOC, MEDIUM risk (it changes which cells tie-break in, same class as the radix top-k
  change already accepted).
- **f16 pooled-key cache.** Halves the 132·D score matmul read (132 -> 66) and halves 5.1's VRAM.
  ~20 LOC.

### 5.4 A fourth step, only if 300K must be strictly inside 5%: fuse the residual chain

§4.1 measured the query-dependent half at **122-131 GB/s** where the streaming pool it replaces
achieves **285**. Same data, same graph, 43% of the bandwidth — because after 5.1 the chain is ~9
small kernels on `n_blocks`-sized tensors and is launch- and latency-bound, not bandwidth-bound.
Fusing score -> relu -> head-sum -> bias -> (block) top-k into one kernel should recover ~2x, which
takes 300K to **+3.7%** and 600K to **+7.3%**. **This is the only step in the program whose value is
an engineering judgement rather than a measurement**, and it is also the one that becomes measurable
for free the moment 5.1 lands (the `pooled_cached` arm already isolates exactly this chain).

### 5.5 What is explicitly *not* needed

- **Retraining or a different model.** None of the above changes the mathematics; 5.1 is pure
  memoization, 5.2 is exact by permutation-invariance of softmax over the finite-mask column set
  (doc 10 §4.2), 5.3 changes tie-breaking only.
- **Reducing the floor.** The floor is not the problem for this bar - it is what makes the slope
  survivable at all; see §7.
- **MTP or the MoE cache.** Both optimize the floor, not the slope.

---

## 6. Honest assessment

**The bar is achievable, but only with the whole program, and 300K is the hard end of it.**

The question is *not* whether the curve can be made asymptotically flat — it cannot, by any
attention architecture — but whether it sits inside the envelope **within the 300K-600K range this
deployment actually uses**. Everything below is that concrete test, at those two depths.

- No single change gets close. Sparse FA alone: **+120% at 300K**. Indexer cache alone: **+238%**.
- 5.1 + 5.2 together: **+14.9% at 300K, +29.8% at 600K.** That is a **24x** improvement on the
  slope and it still misses both bars. Worth knowing before anyone builds half of it and expects
  the target.
- 5.1 + 5.2 + 5.3: **+7.3% at 300K, +14.6% at 600K.** **600K is inside the bar. 300K is 2.3 points
  outside a 5% ask** — close enough that the honest statement is "at the bar, within the
  uncertainty of the floor", not "met".
- Fusing the residual chain (§4.1, estimated 2x) takes it to **+3.7% / +7.3%**, inside both. This is
  the only step in the program that is not backed by a direct measurement.

Call it **~800-1,000 LOC across three changes** (one MEDIUM-HIGH, two MEDIUM), plus a fourth
fusion step of similar size if 300K must be strictly inside 5%. All of it is gated by existing
exact-equivalence tests, none of it needs retraining, and one piece (5.1) is backend-agnostic and
plausibly upstreamable.

Where the estimate is soft, in order of how much it could move:

1. **The residual chain's bandwidth — flagged as the soft spot, then measured, and it cost 2
   points.** Group B runs at 122-131 GB/s, not the 285 GB/s the streaming pool achieves, so the
   pooled-key cache is worth 7-8x rather than the byte model's 17.6x. That is already folded into
   §4's table. What remains uncertain is only the *fusion* row: 2x is an engineering judgement from
   the bandwidth gap, not a measurement.
2. **The floor's own depth-independence past 116K.** §8 measures it flat from `d0` to `d118016`
   (35.8-37.5 ms). Structurally it should keep holding — the 36 gated-DeltaNet layers carry a
   fixed-size recurrent state, MoE routing is per-token, and the per-token kernel-launch count does
   not vary with depth — but past 116K it is an extrapolation. The 300K end-to-end run
   (`logs/long-context-300k/`) measured **173.37 ms/token** against this model's **174.7**, which
   would be a 0.8% confirmation except that **that run's output was degenerate** and degeneracy
   flatters the floor by collapsing MoE routing onto one expert set
   (`logs/long-context-300k-benchmark-report.md` §4.1.4 measures the artifact at ~13 ms/token). On
   the *healthy* `-ncmoe 32` floor of 39.5 ms the model predicts 177.9 ms at 300K. Either way the
   floor is depth-flat to within a few ms out to 300K, but the end-to-end 300K point should be
   re-measured on a working rope configuration before it is quoted.
3. **5.2's sparse proxy.** 0.67 ms/token comes from the dense kernel at `n_kv = 2048`. A real
   compaction pass adds a gather over 2 304 rows per layer; doc 10 §4.2 put the recovered fraction
   at 33.6 of 35.13 ms at 116K, i.e. ~4% overhead, which is inside the margin here.
4. **Interaction between 5.1 and VRAM.** The pooled-key cache wants 236-472 MiB at 300-600K, and
   `logs/decode-scaling/vram-probe-300k.log` shows 300K already needs `-ncmoe 32` at `-ub 2048`,
   with 1 472 MiB spare. It fits, but it costs roughly one further `-ncmoe` step at 600K, which
   raises the floor slightly — which, per §4.3, actually helps the percentages.

---

## 7. The tension with the MoE-cache thesis, stated plainly

The floor is what makes the slope survivable. A larger floor *improves* the percentages in §4's
table for a fixed slope, because the bar is relative. The MoE expert cache — this project's founding
thesis — exists to shrink the floor.

Concretely, with 5.1 + 5.2 + 5.3 done (the "both + block top-k + f16 pool" row):

| floor | 8K | 300K | 600K |
|---|---|---|---|
| **36.3 ms** (measured today, `-ncmoe 24`) | 37.35 ms | 40.09 ms (**+7.3%**) | 42.81 ms (**+14.6%**) |
| **20 ms** (a resident-expert target, from `PLAN.md`'s 28-32 tok/s projection) | 21.04 ms | 23.78 ms (**+13.0%**) | 26.50 ms (**+26.0%**) |

The identical, unchanged slope goes from clearing the 600K bar with 10 points to spare to missing
it. Nothing about the attention or indexer work got worse; the model just got faster at 8K.

That is not an argument against either workstream. It is an argument for tracking them against the
same bar: **the slope work has to land with, or ahead of, any large reduction in the floor**, and
"5% at 300K" gets harder every time the model gets faster at short context. The user's dense-model
analogy is exactly this effect in reverse: a dense model looks flat because its per-token weight
read is enormous relative to its KV traffic, not because its attention is O(1).

---

## 8. The end-to-end curve, measured, and the floor

`llama-bench -p 0 -n 32 -d 0,2048,8192,32768,65536,118016 -r 2` on the full recommended config
(`-ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -ncmoe 24 -ub 2048 -lm none -lzm off`), GPU exclusively held.
Driver `staging/work/decode_depth_scaling.sh`, log
`logs/decode-scaling/llama-bench-depth-curve.log`.

| depth | measured tg32 (tok/s) | ms/token | FA (§1) | indexer (§1) | **implied floor** | model at floor 36.31 |
|---|---|---|---|---|---|---|
| 0 | 26.48 ± 0.45 | 37.76 | 0.31 | — | 37.45 | — |
| 2 048 | 26.03 ± 0.04 | 38.42 | 0.91 | 0.32 | 37.20 | — |
| 8 192 | 24.63 ± 0.02 | 40.60 | 2.15 | 1.83 | **36.63** | 40.29 (**-0.8%**) |
| 32 768 | 19.43 ± 0.09 | 51.47 | 9.69 | 5.19 | **36.59** | 51.19 (**-0.5%**) |
| 65 536 | 15.26 ± 0.04 | 65.53 | 19.38 | 9.93 | **36.23** | 65.62 (**+0.1%**) |
| **118 016** | **11.22 ± 0.01** | **89.13** | 34.96 | 18.34 | **35.82** | 89.61 (**+0.5%**) |

Independent cross-check at 118 016 from the real server benchmark rather than `llama-bench`:
`logs/long-context-120k-lazyoff/` measured **11.146 tok/s = 89.72 ms/token**, against `llama-bench`'s
**11.22** and the model's **89.61**. **Six depths, two tools, one model, worst error 0.8%.**

**The floor is 35.8-37.5 ms/token and does not move with depth.** That is the single most
load-bearing fact in this document, and it is measured rather than argued: every ms of the
difference between 37.76 ms at `d0` and 89.72 ms at 116K is accounted for by §1's two curves, with
nothing left over. Structurally it should be flat — the 36 gated-DeltaNet layers carry a
fixed-size recurrent state, MoE routing is per-token, and the per-token kernel-launch count does
not vary with depth — and §1's fits confirm it empirically to 116K.

### 8.1 What the user's bar looks like against this curve

Reference depth 8 192 (measured 24.63 tok/s). Bar: ≤5% slowdown at 300K, ≤25% at 600K.

| depth | measured / modelled ms/token | slowdown vs. 8K |
|---|---|---|
| 8 192 | 40.60 (measured) | — |
| 32 768 | 51.47 (measured) | **+26.8%** |
| 65 536 | 65.53 (measured) | **+61.4%** |
| 118 016 | 89.72 (measured, server) | **+121%** |
| 307 200 | 175.0 (modelled, components measured at this exact depth) | **+331%** |
| 614 400 | 312.3 (modelled, components measured at this exact depth) | **+669%** |

**The bar is missed at 32K, let alone 300K.** The "half the speed at 100K vs 10K" pattern is exactly
right: 24.63 -> 11.15 tok/s is a 2.21x loss between 8K and 116K. Note that the 307 200 and 614 400
rows are *not* extrapolations of a fit — `FLASH_ATTN_EXT` and the indexer chain were both measured at
those exact `n_kv` values (§1); only the floor is carried over from the shallow measurements.

---

## 9. ADDENDUM (2026-09-11, later): §5.3's first follow-on is implemented, and §5.3 was wrong about what makes it safe

Block-granularity top-k is in the tree (`src/models/qwen4exp.cpp`, `src/llama-memory-hybrid-idx.cpp`),
behind `LLAMA_QSA_CELL_TOPK=1` for a same-binary A/B. §5.3 called it "~100-150 LOC, MEDIUM risk (it
changes which cells tie-break in)". The LOC estimate held. **The risk analysis did not**: §5.3's
one named subtlety was the wrong one, and two hazards it did not name would each have silently
broken the model.

### 9.1 What §5.3 got wrong

§5.3: *"at decode with causal attention the block bias already carries the visibility test
(`blk_bias`), so the per-cell mask only matters for the trailing partial block."* True at decode.
False at prefill, and it misses the tail entirely.

1. **The incomplete tail is unreachable through `blk_cells`.** Cells no full block covers -- the
   trailing `< r` cells of a sequence, which at decode **include the token's own key** -- all point
   at one spare block through `cell_blk`, and the per-cell path reaches them by expanding that
   block's score. But `blk_cells`, the only block -> cell map a block-granularity selection has, was
   zero-filled for that spare block (`llama-memory-hybrid-idx.cpp`, `std::fill(cur_blk_cells, ...,
   0)`). A block-level top-k would have selected the spare block and gathered **cell 0 four times**,
   dropping the most recent 1-3 tokens from attention on every decode step. Fixed by packing the
   unpooled cells into the spare blocks from `dead_bid` up -- `n_blocks = ceil(n_kv/r)` always has
   room, since `n_kv = n_bid*r + U + E` gives `n_blocks >= n_bid + ceil(U/r)` -- and giving every
   used spare block the tail bias.
2. **Whole future blocks carry `+1e9`, not `-inf`.** The per-block bias marked every block at or
   past the query's `tail_start` as always-visible and left the per-cell mask to drop the future
   ones (the code says so: *"the caller adds the attention mask, which drops empty, foreign and
   future cells"*). At decode nothing is in the future, so this is invisible. At prefill a
   `-ub 2048` ubatch writes **all** its cells before the graph runs, so the first token of a ubatch
   sees **~511 whole blocks at `+1e9`** and would have spent its entire 513-block budget on them,
   then had all of it masked out -- attending to almost nothing. Fixed by making the bias an exact
   block-level visibility test: `bid_idx > q` -> `-inf`, straddles the query -> `1e9`, fully visible
   -> `0`. Provably a no-op for the per-cell path, because those cells were `-inf` after the mask
   add anyway.

A third, smaller one: when nothing is unpooled the spare block still took the `1e9` tail bias, so
one of the 513 blocks was spent on an all-zero `blk_cells` row. Now `n_spare = ceil(n_tail/r)`, with
a floor of 1 only when `n_bid == 0` (a sequence shorter than `r` owns no full block, and an all
`-inf` row is a nan).

**Generalisable lesson: a block-granularity selection has no per-cell mask downstream of it, so
every visibility test the per-cell mask used to carry has to be re-expressed at block granularity
first.** That is the real content of this change; the graph edit is nine lines.

### 9.2 Correctness: an index-level equivalence proof, not an output inspection

`staging/work/qsa_blk_index_equiv.py` reimplements `set_input_qsa`'s grouping, bias and `blk_cells`
and both selection tails, then compares the **attended** cell set (the selection intersected with
the per-cell mask `build_attn_qsa` applies afterwards). `ggml_top_k`'s order is unspecified, so
every tie class is resolved both ways to bound what the per-cell path could legally pick.

| regime | result |
|---|---|
| budget covers the context (`n_kv <= width`), 13 lengths x 6 query positions | **exact set equality**, both tie directions |
| deep, block scores distinct | symmetric difference **<= r = 4 cells of 2051 (0.2%)**, and it is entirely the whole-block rounding: the per-cell path spilled `r-1` cells into a partial 514th block, the block path cuts on a block boundary. At the **decode** shape (`q` = the last position) the block path is a strict **superset**, 2052 vs 2051; mid-prefill it is short by 2-4 |
| deep, with the real `ggml_relu` tie mass (40% of blocks at exactly 0.0) | additional **symmetric** reshuffling inside the zero-score tie class (e.g. 194 out / 191 in) -- the class this project already accepted for the radix top-k change |
| unified cache, 2-4 interleaved sequences | **no structural loss**: the spare-block packing reaches every sequence's tail, which the old `idx%r` slot assignment could not have |

So the change is **not** set-equivalent, and the difference is bounded and one-directional in the
untied case: at most `r` cells out of 2051, always at the bottom of the ranking. That is closer to
the reference than the code it replaces -- DeepSeek's design selects `indexer_top_k/r` whole blocks
plus the tail, which is exactly `n_top = ceil(width/r) = 513`; the old `width = indexer_top_k + r -
1 = 2051` was a cell-granularity approximation of that and over-selected by up to `r-1`.

### 9.3 Value, measured in isolation

`test-backend-ops perf -b SYCL0`, same instrument, same shapes and same box as §1. Log:
`logs/qsa-blocktopk/perf2.log`. The per-cell arm reproduces §1's slope to **0.08%**
(0.013078 vs 0.013068 us per KV token per layer) on a different day, which is what makes the
comparison trustworthy.

**Decode shape (`n_tps = 1`), us/call for one QSA layer:**

| `n_kv` | per-cell | block | x | + pooled cache | + pooled cache and block | x |
|---:|---:|---:|---:|---:|---:|---:|
| 8 192 | 151.93 | 130.56 | 1.16 | 63.99 | 45.63 | 1.40 |
| 32 768 | 432.12 | 369.10 | 1.17 | 113.42 | 52.41 | 2.16 |
| 65 536 | 825.88 | 731.05 | 1.13 | 145.10 | 60.44 | 2.40 |
| **118 016** | **1 536.17** | **1 385.75** | **1.11** | **211.67** | **78.45** | **2.70** |
| 200 704 | 2 653.70 | 2 430.65 | 1.09 | 358.01 | 118.19 | 3.03 |
| **307 200** | **4 060.84** | **3 677.91** | **1.10** | **534.11** | **148.53** | **3.60** |
| **614 400** | **8 014.61** | **7 277.08** | **1.10** | **1 002.94** | **236.26** | **4.25** |

**Prefill shape (`n_tps = 512`)** -- the per-cell segment scales with `n_tps` and Group A does not,
so this is where the change is worth the most per call:

| `n_kv` | per-cell | block | x |
|---:|---:|---:|---:|
| 8 192 | 1 083.40 | 740.27 | **1.46** |
| 32 768 | 4 299.76 | 1 861.75 | **2.31** |
| 65 536 | 8 489.00 | 3 368.07 | **2.52** |

**Least-squares slopes over `n_kv >= 32768`, us per KV token per layer:**

```
per-cell                 0.013078      (reference)
block                    0.011905       1.10x shallower
pooled cache             0.001558       8.39x
pooled cache + block     0.000320      40.84x
```

**The slope line is the result.** On its own the change is worth 1.09-1.17x on the chain at decode
and 1.5-2.5x at prefill shape. On top of 5.1 it removes **80% of the slope that 5.1 leaves behind**
-- 8.39x becomes 40.84x -- because what 5.1 cannot touch is exactly the `n_kv`-wide per-cell tail
this removes.

### 9.4 The program table, with no estimated factor left in the row that matters

`staging/work/decode_scaling_model.py`, now carrying the measured `IDX_BLK_US` and
`IDX_CACHED_BLK_US` arms. The component model still reproduces the four measured end-to-end depths
to within 0.8%.

| scenario | 32K | 118K | 307K | 614K |
|---|---|---|---|---|
| today | +27.1% | +122.4% | +333.6% | +674.4% |
| block top-k only (measured) | +26.0% | +119.6% | +325.5% | +657.5% |
| sparse FA + pooled-key cache ("both") | +1.4% | +4.9% | **+14.9%** | **+29.8%** |
| **both + block top-k (measured)** | +0.2% | +1.0% | **+3.3%** | **+6.1%** |
| ... + f16 pooled cache (est.) | +0.2% | +0.8% | +1.7% | +4.6% |
| ... + fused residual chain (est.) | +0.1% | +0.4% | +1.2% | +2.3% |

**This changes §5/§6's conclusion.** §6 said the bar needed 5.1 + 5.2 + 5.3 *and* a fourth, wholly
estimated fusion step ("the only unmeasured factor in the table") to put 300K strictly inside 5%.
With 5.3's first half now implemented and measured, **5.1 + 5.2 + block top-k clears both bars on
measured numbers alone: +3.3% at 300K and +6.1% at 600K.** The f16 pooled cache and the fusion step
stop being requirements and become headroom. §6's "~800-1,000 LOC across three changes" stands;
what changes is that the third change is now cheaper than it looked and is already done.

**Standalone value today is still small, and that should not be oversold.** 1.11x on the chain at
118K is 18.43 -> 16.63 ms of an 89.7 ms decode token, i.e. **2.0%**; at 307 200 it is 4.59 ms of
175.0, **2.6%**. On prefill the chain is a much larger multiple per call but a much smaller share of
the whole -- a `GGML_SYCL_OP_PROFILE=2` call-site profile of a 32K prefill
(`logs/topk-ab-after.log`, aggregated) puts `TOP_K` at **0.06%** and all `GET_ROWS` at **0.28%** of
126 s of GPU time against `MUL_MAT_ID`'s 87%, and the arithmetic from the table above predicts about
**1.4%** of a 32K prefill. Neither is resolvable end to end (see 9.5). **The reason to keep this
change is the slope row, not today's number.**

### 9.5 A measurement finding that is bigger than this change: greedy output on this stack is not run-to-run reproducible

The plan for this change was a before/after output-equivalence check at a depth where the budget
covers the whole context, where both selections provably pick every visible cell, so the greedy
continuation had to be byte-identical. It was not. **The control run explains why: the same binary,
same arm, same `--temp 0 --seed 42`, same prompt, run twice, produces different output.**

```
prompt1500, -c 2048 -ub 512 -n 64, /devbin-blk, two runs each
  per-cell arm : run1 5217 bytes, run2 5223 bytes   -> DIFFER
  block arm    : run1 5223 bytes, run2 5223 bytes   -> same
32K prompt, -c 65536 -ub 2048 -n 32, two passes each
  per-cell arm : DIFFER      block arm : DIFFER
```

Perplexity moves too: the per-cell arm measured **2.5914** in one session and **2.6174** in the next
on the identical corpus and config, a 0.026 spread.

The pre-change tree was used as a third arm and lands inside the same spread rather than outside it:
`staging/devbin` (this morning's `libllama`) and `staging/devbin-blk` carry a **byte-identical**
`libggml-sycl.so`, so they differ only in model code -- and in one session the **block** arm on the
new tree reproduced the pre-change tree byte for byte (5 217 bytes each) while the new tree's own
per-cell arm did not (5 223). Which arm looks like the outlier is a property of the run, not of the
change.

**Consequences, in order of importance:**

1. **Byte-identical model output is not a valid correctness oracle on this stack, and several of
   today's A/B write-ups lean on it.** `PLAN.md`'s top-k update reasoned from "the 32K A/B produced
   different greedy continuations from the same seed" to a conclusion about top-k tie-breaking. That
   conclusion may still be right, but the observation does not support it on its own -- the same
   thing happens with no change at all.
2. **A quality claim needs an average over many tokens and an error bar.** The arms here are
   **PPL 2.6174 +/- 0.060 (per-cell) vs 2.6028 +/- 0.060 (block)** on the same corpus in the same
   session: the block path is nominally *lower*, the gap (0.015) is a quarter of the per-arm
   run-to-run spread (0.026) and a quarter of the error bar. That is the real quality result.
3. **Root cause is not established here and is out of this change's scope.** It is not the QSA
   selection: the block arm was stable across two runs at the shallow shape while the per-cell arm
   was not, and both drift at 32K. The candidates worth checking are the CPU-side `MUL_MAT_ID` at
   `-ncmoe 24` (thread-count-dependent float reduction order) and the SYCL reductions. **Worth its
   own pass**: until it is closed, no A/B on this box can claim bit-identity, and the cheapest check
   is a single run repeated with `--threads 1`.

### 9.6 What was and was not measurable end to end

A 32K prefill A/B was run with the arms interleaved. It is **not** able to resolve a ~1.4% effect:
the per-cell arm alone measured **306.84 / 373.38 / 410.23 tok/s** across three runs of an identical
config (the leading-arm page-cache effect this document's `-ub` sweep already documented, plus
ordinary variance), against the block arm's **405.82 / 399.76**. Reporting the trailing-arm ratio
(+7.1%) would be reporting noise. The isolated `test-backend-ops` measurement in 9.3 is the right
instrument for an effect this size, and the component model that turns it into an end-to-end number
is validated to 0.8% in 9.4.

**The end-to-end decode A/B did land, and it confirms the isolated measurement.** `llama-bench -p 0 -n 32
-d 32768 -r 3` on the recommended config, four arms interleaved blk/cell/blk/cell (log
`logs/qsa-blocktopk/qsa-blk-bench.log`):

```
block    19.78 +/- 0.04    19.69 +/- 0.11     mean 19.735 t/s = 50.671 ms/token
per-cell 19.35 +/- 0.04    19.52 +/- 0.06     mean 19.435 t/s = 51.454 ms/token
```

**+1.54%, and every block arm beats every per-cell arm** (min 19.69 > max 19.52), which is the honest form
of the claim: the between-pass spread within an arm reaches 0.9%, so a single pass would not have been
decisive and the interleaving is what makes it usable. The saving is **0.782 ms/token against 0.756 ms
predicted** from 9.3's isolated chain measurement -- agreement to **3.4%**, which is the real result here:
it is the `test-backend-ops` -> end-to-end chain of inference in 9.4 being validated a fifth time, so the
300K/600K rows of that table rest on a link that has now been checked directly for this change.

Two operational notes. An earlier attempt at this same A/B (`llama-bench`, the instrument §8 used) was lost once to
a **false low-memory alarm**: the agent harness killed the shell wrappers for "running low on memory" while
`free` reported **available 91 GiB of 123** -- only the `free` column was low, with 72 GiB of it reclaimable
page cache. That is the same `free`-vs-`available` error this project has now made in several forms. The
real damage was indirect: killing the wrapper orphaned the `docker compose` client, so `llama-bench` kept
running and holding the card with nowhere to write. **Long GPU runs should write to a file under `/work`
rather than relying on the client's stdout**, which is what `staging/work/qsa_blk_bench.sh` now does.

The first `test-backend-ops perf` invocation died with **rc=137** 39 s in,
at the moment the previous tenant's `llama-server` container was being torn down. It reproduced
cleanly on a re-run (`logs/qsa-blocktopk/perf2.log`, exit 0) with identical numbers where the two
overlap (e.g. `n_kv = 32768` per-cell 428.18 vs 432.12, 0.9%). Treat a 137 during a container
handover as a handover artifact, but always re-run rather than stitching two partial sweeps.

---

## 10. ADDENDUM (2026-09-12): §5.1 is implemented. The chain win holds at 7.4x; the slope win is 1.39x, and §9.4's table was optimistic

The incremental pooled-key cache is in the tree
(`src/llama-memory-hybrid-idx.{h,cpp}`, `src/models/qwen4exp.cpp`), behind
`LLAMA_QSA_NO_POOL_CACHE=1` for a same-binary A/B. This is the first entry in this document
written from a real implementation rather than from a `test-backend-ops` arm, and the two
disagree in a way that matters.

### 10.1 What was built

The post-norm, post-RoPE pooled block keys live in one F32 buffer per QSA layer,
`[indexer_head_size, n_ctx/r]`, allocated next to the indexer KV cache. Per ubatch the graph
pools, norms and ropes only a fixed window of blocks and writes them back with `ggml_set_rows`;
the scorer then reads a **view of that write**, so the dependency is in the graph rather than in
build order. The window is `ceil(n_tps/r) + 2*n_seq_max + 2` blocks - 5 at decode with one
sequence, 514 at a `-ub 2048` prefill ubatch - against `n_ctx/r` = 76 800 blocks at 300K.

Two pieces carry the correctness:

- **The grouping moved out of `set_input_qsa` into `qsa_prepare`**, memoized and dropped by
  `apply()`, so the graph build (which needs the window size before `set_input` runs, because it
  is a graph shape) and `set_input_qsa` share exactly one O(`n_kv`) pass per ubatch, as before.
- **A two-part validity test.** A fingerprint of every cached block - its position bucket and its
  first cell - is compared prefix-wise each ubatch, which catches block ids shifting; and every
  operation that can free, move or re-seat a cell (`clear`, `seq_rm`, `seq_cp`, `seq_keep`,
  `seq_add`, `seq_div`, `state_read`, `state_drop`) drops the whole cache, which catches the case
  the fingerprint cannot see: a freed cell refilled by a different token at the same position.
  When the window is too small for what has to be rebuilt the plan falls back to rebuilding every
  block, which is exactly today's cost for that one ubatch.

Scope deliberately left out: the cache is off for `n_stream > 1` (block ids are per stream), for
ranked mrope cells (an insert re-ranks everything), and for a model with more than one compress
ratio. Each falls back to the old path. `seq_add` turns out to be unreachable on this model
anyway - `llama_kv_cache` refuses it for `n_pos_per_embd() != 1`, and this model is mrope.

One limitation that is correct but not fast, worth knowing before this meets a busy server: on a
**unified cache with several interleaved sequences**, a block completing at a position bucket that
already holds another sequence's block shifts the ids of everything after it, the fingerprint sees
it, and that ubatch rebuilds every block. Single-sequence long context - this deployment - never
hits it. Making ids stable under that would mean not compacting them, which costs buffer rows;
it was not done.

VRAM: `1 536 * n_ctx` bytes, i.e. **180 MiB at 122 880, 456 MiB at 307 200, 900 MiB at 614 400**.
The 300K probe in §6 left 1 472 MiB spare at `-ncmoe 32`, so it fits with ~1 GiB over.

### 10.2 Correctness: an in-graph oracle, because a cross-run one does not exist

**The first attempt at an A/B failed, and the reason is more important than the change.** Two runs
of the **same** arm, same binary, same fixed token stream, produce **different indexer tensors** -
not just different final output. Hashing `indexer_k`, `indexer_score` and `indexer_top_k_blocks`
over 12 layers x 40 graph invocations, cached-vs-cached mismatched in 396 of 480 comparisons,
exactly as cached-vs-plain did. **§9.5 found greedy output is not reproducible on this stack; it
is worse than that - the forward pass itself is not reproducible.** Any comparison that spans two
processes is measuring that noise, whatever it is nominally testing.

So the oracle was moved inside one graph. `LLAMA_QSA_POOL_CHECK=1` builds the from-scratch
derivation of the same keys alongside the cached ones, on the same inputs, in the same graph, and
a harness (`staging/work/qsa_pool_equiv.cpp`) compares them row by row.

| run | result |
|---|---|
| 4 003-token prompt (8 ubatches) + 32 decode steps, 12 layers | **480/480 comparisons bit-identical over every block the bias can select** |
| where any row differs at all | `first_diff_row` is **exactly** `ceil(n_cells/r)`, the first block past the live region, in every case, and the differing count is exactly `n_blocks - ceil(n_cells/r)` |
| same, with a mid-run `seq_rm` of the last 100 positions and a refill | **fully identical from the invalidating step onward**, tail included |

The rows that differ are the ones `set_input_qsa` writes `-INFINITY` for, so no selection can
reach them. This is the exact-equivalence bar §5.1 asked for, and it is met rather than
approximated: the cache is pure memoization and the measurement says so bit for bit.

`test-backend-ops test -b SYCL0 -o QSA_INDEXER`: **2/2 backends pass**, unchanged (those cases are
test-only graphs and the change does not touch them). `-o SET_ROWS`, the op the cache writes
through, **aborts on this backend** at `ggml-sycl/set_rows.cpp:550`, "Unsupported tensor type:
src0 f16 src1 i64 dst tq1_0" -- **pre-existing**: it reproduces identically on `staging/devbin-blk`,
the tree before this change, and the ternary dst types it dies on are not ones this cache uses.
Recorded so nobody attributes it here.

Perplexity, `logs/qsa-pool/`:

| corpus | pool | plain |
|---|---|---|
| 116K refactor benchmark, `-c 8192`, 10 chunks, geometric mean over chunks 2-10 | **1.0235 / 1.0235** | **1.0235 / 1.0236** |
| same, per-chunk spread across all four arms, chunks 2-10 | **0.015 - 0.103% across all four arms** | |

Chunk 1 swings 13.62 - 29.38 across the four arms with no relation to which arm it is (the two
`pool` arms are the highest *and* the lowest of the four), which is the same non-reproducibility
as above showing up where the model is uncertain. Cumulative 10-chunk estimates are dominated by
it and should not be quoted on their own.

**The 3-chunk gate that §9.2 used was also run, and it has no discriminating power at all.** Same
corpus and config as `staging/work/qsa_blk_e2e.sh` section B (`prompt32k.txt`, `-c 8192 -ub 2048
--chunks 3`), five interleaved arms: **2.5424 (warm-up), 2.6985 / 2.9168 (pool), 3.5860 / 2.5986
(plain)**. The *within-arm* spread on `plain` alone is **0.987 absolute**. §9.2 reported the
block-top-k gate as `2.6174 +/- 0.060` vs `2.6028 +/- 0.060` from one run each on this exact
configuration; the run-to-run spread measured here is **~70x the gap that comparison rested on**.
The 116K ten-chunk per-chunk comparison above is the one to trust, and the three-chunk gate should
not be used again.

### 10.3 Performance: the chain win reproduces, the slope win does not

`llama-bench -p 0 -n 32 -r 2`, `-ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -ncmoe 24 -ub 2048 -lzm off
-lm none`, arms interleaved behind a discarded warm-up arm, GPU exclusively held. Logs
`logs/qsa-pool/bench-shallow.log`, `bench-deep.log`.

| depth | plain (tok/s) | pool (tok/s) | x |
|---|---|---|---|
| 0 | 26.53 | 26.27 | 0.99 |
| 8 192 | 24.70 | 24.86 | 1.01 |
| 32 768 | 19.77 | 21.00 | **1.06** |
| 65 536 | 15.47 | 17.27 | **1.12** |
| 118 016 | 11.36 | 13.46 | **1.18** |

Least squares over `d >= 8192`, and the residuals are within 0.9%:

```
plain   floor 36.58 ms/token   slope 0.4338 ms per 1 000 tokens of depth
pool    floor 37.56 ms/token   slope 0.3111 ms per 1 000 tokens of depth   ->  1.39x shallower
```

Subtracting §1's measured `FLASH_ATTN_EXT` slope (0.2918 ms/1k over 12 layers) leaves the
indexer-plus-host term:

```
plain   0.1420 ms/1k     (sect 9.3's block-top-k arm predicts 0.1429  -- 0.6% agreement)
pool    0.0193 ms/1k     (sect 9.3's pooled+block arm predicts 0.0038 -- 5.1x optimistic)
                         -> measured reduction 7.4x, not the modelled 40.8x
```

**The component model predicts the code it was built from to 0.6% and its own proposal to 5x.**
That asymmetry is the finding. §9.3's arms are `test-backend-ops` graphs: they contain the GPU
chain and nothing else. A real implementation also pays, every token, the part of
`set_input_qsa` that stays O(`n_kv`) - the grouping scan, the per-token O(`n_blocks`) block bias,
and the `cell_blk` / `blk_cells` / `bias` uploads. That is ~0.0155 ms/1k, i.e. **~4.8 ms of a
133 ms token at 300K**, and it is now the second-largest depth-proportional term in the decoder.
It is ordinary work to fix - the grouping is incremental for exactly the same reason the pooled
keys are - but it was invisible to the instrument that sized this change.

The cache also costs ~1.0 ms/token of fixed overhead (the 5-block window, three extra input
uploads and the `set_rows`, 12 times per token), so it **breaks even around 6-8K of depth** and is
a small loss below that. At `d0` it measures 26.27 vs 26.53.

### 10.4 Where this leaves the target, stated plainly

**`FLASH_ATTN_EXT` is now 93.8% of the remaining slope.** Extrapolating the measured fit (both
terms are linear and §1 measured FA at these exact depths):

| depth | plain | **pool** | x | |
|---|---|---|---|---|
| 118 016 | 87.8 ms (11.36 t/s) | **74.3 ms (13.46 t/s)** | 1.18 | measured, `-ncmoe 24` |
| 200 704 | 123.7 ms (8.09 t/s) | **100.0 ms (10.00 t/s)** | 1.24 | fit |
| **307 200** | **175.4 ms (5.70 t/s)** | **136.1 ms (7.35 t/s)** | **1.29** | **measured, `-ncmoe 32`** |
| **614 400** | 303.1 ms (3.30 t/s) | **228.7 ms (4.37 t/s)** | **1.33** | fit, `-ncmoe 24` floor |

**The 300K row is a real measurement, and it validates the fit.** `llama-bench -d 307200 -r 1
-ncmoe 32 -ub 2048` (300K needs `-ncmoe 32`, per the VRAM probe in section 6, and the pooled-key
buffer takes 456 of the 1 472 MiB it left spare): **7.35 vs 5.70 tok/s**, against the
`-ncmoe 24` fit's prediction of 7.51 / 5.89 / 1.28x -- **2.2% / 3.2% / 0.8% agreement across an
`-ncmoe` change and a 2.6x extrapolation in depth.** Log `logs/qsa-pool/bench-300k.log`. The
614 400 row stays a projection and carries the `-ncmoe 24` floor, so it is if anything optimistic:
600K needs a higher `-ncmoe` still.

**This fix alone does not come close to the bar, and was never going to.** §4.2 said the pooled-key
cache and sparse FA are "a matched pair, not substitutes"; that was a projection, and it is now a
measurement. **7.35 tok/s measured at 300K** and ~4.4 projected at 600K are against a target of 17-25 tok/s at
600K.

What the same measured fit says **if** `docs/research/10`'s Design B lands on top of this
(FA collapses to a bounded ~0.67 ms/token, flat with depth):

| depth | ms/token | tok/s | vs 8K |
|---|---|---|---|
| 8 192 | 38.38 | 26.05 | - |
| 118 016 | 40.50 | 24.69 | +5.5% |
| **307 200** | **44.16** | **22.65** | **+15.0%** |
| **614 400** | **50.09** | **19.97** | **+30.5%** |

Anchoring the 300K row on the **measured** `-ncmoe 32` token instead of the `-ncmoe 24` fit
(136.1 ms, of which section 1 puts 89.6 ms in `FLASH_ATTN_EXT`) gives **47.2 ms, 21.2 tok/s** --
the same conclusion one `-ncmoe` step lower. 600K would need a higher `-ncmoe` again, so read
19.97 as an upper bound.

**The user's absolute bar is met by the pair; this document's relative bar is not.** 20.0 tok/s at
600K sits inside the stated 17-25 band, and 22.7 at 300K is comfortable. But +15.0% at 300K and
+30.5% at 600K are three times §9.4's "+3.3% / +6.1%", for the single reason in §10.3: the
residual host term is 5x what the isolated arm implied. §9.4's table should be read as retired and
this one used instead.

Two things follow, in order:

1. **`docs/research/10`'s Design B is the next task and is now the whole remaining problem.** It is
   93.8% of the slope that is left, it is unbuilt, and nothing else in the program moves it.
2. **The residual host term (~0.0155 ms/1k) is the cheap follow-on**, worth ~3.6 points at 300K and
   ~7 at 600K on the table above, and it is the same "only the tail changes" argument applied to
   `qsa_prepare` rather than to the keys. The f16 pooled cache (§5.3) and the fused residual chain
   (§5.4) remain available behind it.

## 11. ADDENDUM (2026-09-12, later): §10.3's residual host term is measured and fixed (2.17x, worth 2.76 ms of a ~58 ms 600K token). It is not the 600K shortfall, and the deep end-to-end instrument turns out not to resolve either

§10.3 named "the part of `set_input_qsa` that stays O(`n_kv`) every token" as the second-largest
depth-proportional term in the decoder and sized it at **~0.0155 ms per 1 000 tokens of depth**
from the gap between its own component model and a real implementation. `PLAN.md`'s sparse-FA
update then made it the named cause of the 600K miss, and
`13-dynamic-context-aware-expert-placement.md` §5.4 priced removing it at **600K -> 19.7 tok/s**
on the assumption that the *whole* `-ncmoe 38` excess slope was this term.

That term has now been instrumented directly, at the config 600K actually needs, and fixed.
Three results, in descending order of how much they should change anyone's plans:

1. **The deep end-to-end instrument does not work.** The unfixed code measures 17.27 *and* 11.50
   tok/s at `d614400` from the identical command (§11.8). Nothing that moves a decode token by a
   few percent can be A/B'd at that depth on this box today, and several numbers this project has
   quoted from single deep runs inherit that spread.
2. **The term is real, is 1.9x smaller than §10.3 estimated, and is now 2.17x cheaper** — but it
   is **31.9%** of the residual depth slope, not the whole of it, so §5.4's 19.7 tok/s projection
   does not follow from removing it.
3. **The size and the attribution were both wrong, in opposite directions**: smaller than
   estimated as a slope, larger than estimated as a share of `set_input_qsa` itself, and not
   located where the phrase "grouping scan" suggests (§11.3).

### 11.1 The instrument

`LLAMA_QSA_HOST_PROF=1` wraps eleven phases of `qsa_prepare` and `set_input_qsa` in
`ggml_time_us()` scopes and accumulates only over decode ubatches (`n_tokens == 1`), ending a
window when the depth changes so one `llama-bench` process covers every `-d`. This is host CPU
time inside named code, so unlike an end-to-end `tok/s` it is immune to the page-cache and
leading-arm effects this document has been bitten by four times. Driver
`staging/work/qsa_hostprof.sh` / `qsa_plan_ab.sh`, log `staging/work/qsa-plan-ab.log`.

Config: `-ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -ub 1024 -ncmoe 38 -lzm off`, i.e. `PLAN.md`'s 600K
arm. `-lm none` is deliberately **not** set — see §11.5, which is the largest finding here.

### 11.2 The measured term, and it is 1.9x smaller than §10.3's estimate

ms/token, one QSA input set shared by all 12 QSA layers, 32 decode ubatches per window:

| phase | `n_kv` 8 448 | 33 024 | 614 656 |
|---|---|---|---|
| `prep.alloc` (three n_kv-wide `assign`s) | 0.0047 | 0.0149 | 0.2384 |
| **`prep.group_cells`** (the per-cell scan) | 0.0421 | 0.1460 | **2.6656** |
| `prep.blk_enum` (per block) | 0.0053 | 0.0171 | 0.3289 |
| **`prep.cell_loop`** (second per-cell pass) | 0.0154 | 0.0444 | **0.8507** |
| `prep.fingerprint` | 0.0016 | 0.0038 | 0.0566 |
| `set.cpy_cell_blk` | 0.0016 | 0.0053 | 0.1360 |
| `set.cpy_blk_cells` | 0.0008 | 0.0034 | 0.1232 |
| **`set.bias`** (per-token, per block) | 0.0097 | 0.0308 | **0.6544** |
| `set.pool_cpy` + `set.upd` | 0.0008 | 0.0008 | 0.0012 |
| **total** | **0.0846** | **0.2691** | **5.058** |

Linear in `n_kv` to within 10% over a 73x range (0.00751 ms/1k between the first two points,
0.00824 between the last two). So the term is **0.00823 ms per 1 000 tokens of depth, not
0.0155** — §10.3's figure was inferred from a residual between two models and is 1.9x high.

At 614 400 that is **5.06-5.20 ms/token** (two runs an hour apart, agreeing to 2.8%), i.e.
**~4.8-8.9% of a 600K token** depending on which of the two irreconcilable end-to-end numbers in
§11.8 the denominator comes from, and **31.9% of the token's whole residual depth slope**
(0.02584 ms/1k, from the faster arm's `d32768` and `d614400` points).

### 11.3 Why it costs what it costs

Three mechanisms, all visible in the table:

1. **`set.bias` walks the cells' sequence sets.** The per-token loop tests
   `cells.seq_has(bid_cell[b], seq_id)` once per block, and `llama_kv_cells::seq` is a
   `std::bitset<LLAMA_MAX_SEQ>` — **32 bytes per cell, 19.7 MiB at 614 400** — indexed by a
   block's first cell, i.e. a stride-128B walk of the whole array every token. That is why the
   phase grows 67x between 33 024 and 614 656 while `n_kv` grows only 18.6x: at the shallow point
   the array is in L2.
2. **Six n_kv-wide buffers are written per token**, three of them (`cell_blk`, `blk_of`,
   `cell_grp`) only so that a later pass can read them back, and two of those are read by nothing
   at all in the production configuration (§11.4).
3. **`ratio` is a runtime value, so `idx/r` and `idx%r` are 64-bit divides.** The grouping does
   both per cell per pass. Measured cost of one extra divide per cell at 614 656: **0.61 ms**
   (~4.5 cycles/cell), which is how the first version of the fix below made `cell_loop` *worse*.

### 11.4 The fix

`src/llama-memory-hybrid-idx.{h,cpp}` only, behind `LLAMA_QSA_SLOW_PLAN=1` for a same-binary
A/B. No kernel, MoE-cache, `topk-radix.cpp`, `fattn-sparse.*`, `gated_delta_net.cpp` or
`hyper_connect.cpp` code was touched. Four changes, all of them removing work rather than
parallelising it (the scan is sequential host code that runs *before* `graph_compute`, so it does
not in fact contend with the CPU-resident expert GEMM the way `PLAN.md` supposed):

- **Bucketed grouping.** A block is keyed on (sequence set, position bucket). When the stream's
  cells hold at most one sequence — every single-sequence long-context session, and every stream
  of a split cache — the list search per bucket can only ever match its one node, so the group
  *is* the bucket. The group tables become `n_blocks`-indexed arrays, and the `n_kv`-wide
  `cell_grp` column and the list walk disappear. This is the same algorithm, not an
  approximation.
- **Drop the two dead columns.** `cell_blk` is read only by the per-cell selection and `blk_of`
  only by the per-cell bias; the block-granularity selection that landed in §9 leaves both
  unread. They are now not allocated, not filled and not copied — three n_kv-wide passes gone.
- **Run-length bias.** With bucketed grouping `bid_idx` is strictly ascending, and when the only
  sequence in the cells is the query's the visibility test holds for every pooled block. The row
  is then three runs — visible, always-visible tail, invisible — which two binary searches and
  three `std::fill`s write. 19.7 MiB of strided bitset reads become a 0.6 MiB memset.
- **Shift/mask for a power-of-two ratio**, and the grouping scratch moved into the memoized plan
  so a decode step reuses capacity instead of faulting in several MiB per token.

The last of those was not an afterthought, and the instrument is what caught it: the first
version traded a sequential array read for one extra divide per cell and made `cell_loop`
*worse*, 0.851 -> 1.465 ms. That measures one divide per cell at this depth at **0.61 ms**
(~4.5 cycles/cell). Without the profiler it would have shipped as a regression in the largest
phase. Per-phase result at `n_kv = 614 656`, ms/token:

| phase | unfixed | fixed, v1 | **fixed, final** | x |
|---|---|---|---|---|
| `prep.alloc` | 0.2384 | 0.0771 | **0.0762** | 3.13 |
| `prep.group_cells` | 2.6656 | 1.5626 | **0.8294** | **3.21** |
| `prep.blk_enum` | 0.3289 | 0.2725 | **0.2354** | 1.40 |
| `prep.cell_loop` | 0.8507 | 1.4654 | **1.0092** | **0.84** |
| `prep.fingerprint` | 0.0566 | 0.0686 | **0.0560** | 1.01 |
| `set.cpy_cell_blk` | 0.1360 | 0.0000 | **0.0000** | gone |
| `set.cpy_blk_cells` | 0.1232 | 0.1363 | **0.1360** | 0.91 |
| `set.bias` | 0.6544 | 0.0240 | **0.0216** | **30.3** |
| `set.pool_cpy` + `set.upd` | 0.0012 | 0.0008 | **0.0010** | 1.2 |
| **total** | **5.055** | **3.607** | **2.365** | **2.14** |

Against the mean of the two baseline runs (5.128) that is **2.17x, saving 2.76 ms/token**, and the
host-term slope goes **0.00823 -> 0.00407 ms/1k**. `prep.cell_loop` is still **1.19x worse** than
the general path and is the remaining headroom: for a sequentially-filled single-sequence cache the
block id equals the bucket, so the compaction is the identity and the `blk_cells` write could fold
into the first pass, deleting the second `n_kv` pass entirely.

### 11.5 A lead that looks bigger than the fix: dropping `-lm none` may be worth ~1.4x at 600K

`PLAN.md`'s 600K arm measured **12.42 tok/s (80.52 ms/token)** at `-ub 1024 -ncmoe 38 -lzm off
-lm none`. The same command with **`-lm none` dropped** — mmap for the expert tensors, everything
else identical — measures **17.27 tok/s (57.91 ms/token)** on the slow arm here, i.e. the
un-fixed code. That is **1.39x from one flag**, and it re-prices the whole 600K question:

| | `-lm none -lzm off` | `-lzm off` only |
|---|---|---|
| d614400 | 80.52 ms (12.42 tok/s) | **57.91 ms (17.27 tok/s)**, and 86.96 ms (11.50) on a repeat |
| shallow control | 44.0-44.35 ms | 42.88 ms (d32768) |
| residual depth slope | 0.0602 ms/1k | **0.02584 ms/1k** (from the 17.27 arm) |

**Read that second `-lzm off` entry before using this table.** §11.8 shows the same command
measures 17.27 *and* 11.50 tok/s at this depth, so the 1.39x this section claims for the flag is
inside the box's own spread and should be treated as a lead to re-measure, not a result. What is
solid is that the flag is worth **more** than the host-term fix, not how much more.

So the "5.1x steeper slope at `-ncmoe 38`" that `PLAN.md` attributed to host-thread contention is
**2.3x of configuration and only 1.15x of depth**: at `-lzm off` alone the slope is 0.0258 against
the `-ncmoe 24` arm's 0.0119, not 0.0602. `PLAN.md`'s §"600K was measured too" hypothesis named
the right file and the wrong magnitude.

Mechanism, stated as a hypothesis rather than a measurement: `-lm none` puts ~66 GiB of expert
and PLE tensors in pinned `SYCL_Host` USM. That is unevictable, it pushes the box to ~100 GiB of
123 with swap already exhausted, and host USM is not necessarily cached as writeback for CPU
reads — any of which would slow the CPU-resident expert GEMM that *is* the depth-independent
floor. `docs/research/11` §7 recommended `-lm none` on a **prefill** measurement (394 vs 252
tok/s), which still holds; nothing had re-checked it against decode at `-ncmoe 38`.

### 11.6 Correctness: an in-*call* oracle, which is stronger than the in-graph one

§10.2 established that nothing spanning two processes can be used on this stack. This change can
do better than the in-graph oracle that finding forced: the fast path and the general path are
both pure host functions of the same `llama_kv_cells`, so `LLAMA_QSA_PLAN_CHECK=1` runs **both,
on the same cells, inside the same call**, and asserts they agree.

- The per-stream builder was refactored into one lambda parameterised on `bucketed`. Check mode
  builds a second `llama_qsa_plan::stream` with `bucketed = false` and `GGML_ASSERT`s equality of
  all twelve fields: `n_bid`, `dead_bid`, `n_spare`, `ranked`, `cell_blk`, `blk_cells`, `blk_of`,
  `bid_idx`, `bid_cell`, `bid_slot0`, `rank`, `order`.
- The run-length bias row is compared against the general per-block loop written into a scratch
  buffer, by `memcmp` over `n_blocks` floats — bit-identical, not within-tolerance. The values
  are only `0.0f`, `1e9f` and `-INFINITY`, so `memcmp` is exact and well-defined.

A failing `GGML_ASSERT` aborts the process, so "ran to completion" is the pass.
`staging/work/qsa_plan_check.sh`, log `staging/work/qsa-plan-check.log`, **6/6 sections rc=0**:
an empty cache (`d0`, where `n_bid == 0` and only the spare block carries bias) and `d2048`;
prefill ubatches at `n_tps = 512` and `2048`, where whole blocks sit past the query and the
`blk_sel` future-block test is load-bearing; `d8191` (deliberately not a multiple of
`ratio * n_ubatch`) and `d16384`; `LLAMA_QSA_CELL_TOPK=1`, which is the only configuration in
which `cell_blk` and `blk_of` are built at all, so those two fields are compared there rather
than trivially empty on both sides; and three `llama-perplexity` chunks through one context,
which clears and refills the cells between chunks and so exercises the pooled-key cache's
invalidation path.

**Perplexity is applicable to this change, unlike to `docs/research/10`'s Design B**, because
`set_input_qsa` runs on prefill ubatches too — and it ran clean every time. It also reproduced
§10.2's verdict on the 3-chunk gate for free: **three runs of the *same* arm gave 1.4859, 1.4153
and 1.5432** (a 9% spread), so the gate is a smoke test here and not evidence. The bit-exact
in-call assert is the gate.

Two gaps, recorded rather than papered over. **The mrope-ranked path is not covered** by any arm:
it needs 2-D image positions, which this text model never produces. It is handled identically by
both paths (ranking is orthogonal to bucketing) and the in-call assert would catch it if it ever
ran. And **`llama-cli`, `llama-completion` and `llama-batched-bench` all crash on this model in
this fork before this change** — reproduced on `staging/devbin-sparse`, the build that predates
it — so the multi-sequence general path has no driver to exercise it here; it is unchanged code
reached only when `n_seq_present >= 2`.

### 11.7 Two measurement findings that outlive this change

**`bcache` never caches this model, and that is a knob not a capacity problem.**
`/sys/block/bcache0/bcache/sequential_cutoff` is **4.0M**, so any read above 4 MiB bypasses the
cache device -- and the GGUF parts are 49.5 GB and 32.4 GB, read sequentially at every load.
Measured while a load was in flight: `bypassed` **20.8 G** total and **512 M in five minutes**,
`cache_hit_ratio` **67% / 58%**, and `cache_available_percent` **97** on a ~2 TB cache
(`nbuckets` 3,815,456). So repeated runs of the same model do **not** get faster, which is the
opposite of the natural expectation, and the reason is the cutoff rather than eviction.
`dirty_data` is **19.2 G** against `writeback_percent = 10`, so writeback competes with our reads
on `md0` as well. `echo 0 > sequential_cutoff` (root) would put the model on the NVMe tier; at
5-45 minutes of model read per measurement run, that is worth more to this project's iteration
speed than most of the code changes in this document.

**"The array is at 98% util" is still not a cause, and this pass got it wrong before checking.**
`bcache0` 98.1% / `md0` 97% during the anomalous arm looked exactly like the neighbour-I/O
confound `docs/research/11` §3.3 diagnosed. Measuring the neighbours instead of assuming them
refuted it: **`deluged` 352 KB/s read and zero writes, `unbooru-tagger` 0 KB/s** -- its 490 GB is
cumulative over five hours since boot. The utilization was our own model load. §7.6's replacement
rule ("check the load banner and `majflt/s`; utilization is a symptom") is the one that works, and
our process took 3 major faults in the decode minute, i.e. it was not paging.

### 11.8 The result that limits every other number here: `d614400` decode is not reproducible to 1.5x

The fix was A/B'd end to end with `LLAMA_QSA_SLOW_PLAN=1`, arms in both orders across two
campaigns. Four arms at `d614400`, identical command every time
(`staging/work/qsa-plan-ab.log`, `qsa-plan-ab-600k.log`):

| arm | order | tok/s | host term |
|---|---|---|---|
| unfixed | campaign 1, leading | **17.27** | 5.058 ms |
| unfixed | campaign 2, trailing | **11.50** | 5.198 ms |
| fixed (pre shift/mask) | campaign 1, trailing | 14.22 | 3.610 ms |
| fixed (final) | campaign 2, leading | **1.76** | 2.368 ms |

**The unfixed code alone spans 17.27 to 11.50 tok/s — 1.50x — while its own host term reproduces
to 2.8%.** So the end-to-end instrument at this depth has a spread several times larger than the
4.8% this change is worth; the 1.76 arm is that instability in the extreme rather than a code
effect (the in-call oracle proves that arm's plan outputs bit-identical to the general path, so it
cannot have altered one GPU node); and no ratio in this table is quotable. This is the same
verdict §9.6 reached for the block-top-k *prefill* A/B, now reached for *decode* at 600K.

Consequences, and they are bigger than this change:

- **The 2.76 ms/token saving is quoted from the host-time instrument, not from a tok/s ratio.**
  That instrument reproduces to 2.8% across an hour and a rebuild, which is what makes it usable.
- **Every single-run `d614400` number in this project should be treated as ±25%**, including
  `PLAN.md`'s 12.42 and this document's 17.27. The earlier deep points at `d118016` and below were
  taken with `-r 2`/interleaving and are on firmer ground.
- **A trustworthy deep A/B needs arms in both orders with repeats, at ~45-55 min of prefill per
  arm.** That is 4-6 hours for one comparison, which is why it was not completed here, and it is
  the open item this change leaves behind.
- The cause of the spread is **not established**. It is not our own paging (3 major faults in the
  decode minute), and it is not neighbour I/O (§11.7). What is left is a session-order effect: the
  second arm of a campaign was slower in both campaigns, regardless of which code it ran.
