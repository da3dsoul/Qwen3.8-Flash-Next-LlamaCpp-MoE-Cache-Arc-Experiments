# Hybrid CPU/GPU `MUL_MAT_ID` (upstream RFC #24528) — SYCL/Arc Port Scoping

**Status: scoping only. No implementation code was written; no benchmarks were run.**
Date: 2026-09-11. Motivated directly by `PLAN.md`'s "Update (2026-09-11)" section (`PLAN.md:114-243`), which
records that both our custom SYCL cache (23.1 tok/s) and stock `-ncmoe` (19.7-25.3 tok/s) plateau far under the
47 tok/s same-model reference, and names this RFC's design as "the largest un-tried lever" (`PLAN.md:242-243`).

Sources actually read for this document (not summarized secondhand):

| Source | How obtained |
|---|---|
| RFC body, full text | `gh api repos/ggml-org/llama.cpp/discussions/24528` (`.body`) |
| RFC thread, all 33 top-level comments + replies | `gh api graphql` on the discussion |
| Titles/states of every prior attempt the RFC cites | `gh api repos/ggml-org/llama.cpp/issues/{N}` and `.../discussions/{N}` |
| `leloch/llama.cpp@moe-cache-v2-pr` file-by-file diff vs. master | `gh api .../compare/master...leloch:llama.cpp:moe-cache-v2-pr` |
| `ggml/src/ggml-backend-moe-cache.h` (the API table), full 109 lines | `gh api .../contents/...?ref=moe-cache-v2-pr` |
| `ggml/src/ggml-cpu/ggml-cpu.c` hook diff (445 added lines), first ~230 lines of patch | same compare API, `.files[].patch` |
| Our tree: `ggml-cpu.c`, `ggml-backend.cpp`, `ggml-cpu.cpp`, `traits.h`, `ggml-sycl.cpp`, `moe-cache.cpp/.hpp`, `dpct/helper.hpp`, `common.hpp`, `mmvq.hpp`, `common/arg.cpp`, `common/common.h`, `src/models/qwen4exp.cpp` | direct reads, line ranges cited below |

Everything cited as `file:LINE-LINE` was read directly. Statements that connect facts established in separate
places, rather than quoting one place, are flagged **inference**.

All repo paths below are relative to
`/media/da3dsoul/Golias/AIProjects/Qwen3.8-Flash-Next-LlamaCpp-MoE-Cache-Arc-Experiments/src/llama.cpp/` unless
stated otherwise.

---

## 0. One-paragraph summary of the finding

The RFC's design does **not** require a scheduler change, does **not** require a new buffer type, and does
**not** require the existing SYCL MoE cache. It slots into a ~40-line window inside
`ggml_compute_forward_mul_mat_id` (`ggml/src/ggml-cpu/ggml-cpu.c:1640-1731`) that our tree already has,
under a configuration (`-ncmoe`) our project already ships as its production baseline. The single genuinely
hard question is whether a SYCL dispatch-and-wait round trip, issued from the CPU threadpool's thread 0,
costs less than the CPU work it displaces — 78 times per token at our current `-ncmoe 26` split. That is a
measurable question, and it is measurable with a ~200-line spike that does not touch the MoE cache at all.

---

## 1. What the RFC actually proposes

### 1.1 The design, in the RFC's own words

RFC #24528, "RFC: MoE expert cache, VRAM caching of hot CPU-resident experts with hybrid hit/miss execution",
opened 2026-06-12 by `leloch` in the *Ideas* category, 68 comments, last updated 2026-09-09. Never merged; it
is an RFC, and the PR it follows up (`#24524`, "cuda: MoE expert cache, adaptive VRAM caching of CPU-resident
experts") was **closed 10 minutes after opening** (created `2026-06-12T15:39:38Z`, closed `2026-06-12T15:49:58Z`)
as too large to review.

The core proposal, quoted verbatim from the body:

> That persistent buffer, plus an inverted execution model: **MUL_MAT_ID stays on the CPU.** Inside the CPU
> kernel, thread 0 dispatches one batched matvec over the cached (hit) rows on the GPU while the other threads
> compute the miss rows as they normally would. Misses cost nothing extra, hit rate is pure upside, worst case
> degrades to the vanilla CPU path.

And the reason it exists at all, quoting the "Problem" section:

> Prior attempts (#20757, #21609, #21614, #21620, #23170) all moved MUL_MAT_ID to the GPU and tried to make the
> weight copies cheaper. That puts every cache miss on the critical path as a synchronous PCIe transfer:
> @batot1 measured ~3× decode regression from forced decode offload without residency (#20757), and the Metal
> slot-pool experiment in the same thread was 2× slower than vanilla even at 97-99% hit rate, purely from
> per-layer sync points. #23170's post-mortem concluded the staging-buffer approach is a no-op when made
> correct, and that the fix is a separate persistent expert-cache buffer with explicit expert→slot bookkeeping.

Fill policy, verbatim:

> Fill is decode-only (prompt routing is far flatter and thrashes the cache); decode routing is skewed enough
> to make this work — measured on Qwen3.5-122B, the top 10% of experts take ~80% of hits (independent
> corroboration in #20757: Gini ≈ 0.76, ~99% simulated hit rate at 69% expert budget).

Measured benefit, verbatim table (4× RTX 3090, EPYC 7R13 48c, 8-ch DDR4-3200 256GB, `llama-bench tg300`):

| Model | Regime | cache | vanilla | gain |
|---|---|---|---|---|
| GLM-5.1 754B IQ2_M (220 GB) | zero-config | 17.49 | 13.96 | **+25%** |
| Qwen3.5 397B Q3_K_XL (167 GB) | zero-config | 30.25 | 28.18 | +7% |
| Qwen3.5 122B (fits in VRAM) | zero-config | 74.98 | 74.98 | parity (cache dormant) |
| 13 models / 9 architectures | forced `-ngl 99 -ncmoe 99` | | | +10%…+57%, 16/16 ≥ parity |

Maintenance burden, verbatim:

> ~1,700 lines in one new CUDA file pair (`moe-cache.cu/.cuh`); a backend-agnostic function-pointer API table
> (`ggml-backend-moe-cache.h`); small null-checked hooks in `ggml-cpu.c`/`ops.cpp` (mul_mat_id + GLU) and
> `ggml-backend.cpp` (scheduler offer/redirect); a fit rule + `--moe-cache` flag in common. Non-CUDA builds see
> a zero-initialized table and no-op hooks; `--moe-cache 0` restores stock behavior exactly.

The minimal-core suggestion, verbatim:

> A **minimal core** (plain per-node dispatch only: no fused gate+up+SwiGLU, no GPU-resident handoff, no
> hot-set persistence) is roughly half the code and keeps most of the gain; I can measure each layer's exact
> contribution on this hardware.

### 1.2 The RFC body is stale relative to its own thread — three things changed after publication

The body still describes an "EWMA, 4 strikes at +5%" baseline-sampled bail-out under "Heuristics that
maintainers didn't choose." In a 2026-08-06 comment announcing the rebased `moe-cache-v2-pr` branch, `leloch`
states verbatim in a fixes table:

> | Bail-out/trim didn't restore baseline | **EWMA bail-out deleted.** Eligibility decided up front from
> device capability; trim is OOM-only. |

Two further rows from that same table matter to us:

> | 1080 Ti regression | Fenced by policy: `auto` needs ≥2 cc 8.0 devices, forced needs cc 7.0. sm_61 creates
> no session - expected result is hard-off parity. |

> | Silent bypass above max batch (@noonghunna) | One actionable log line, emitted once, when a node exceeds
> the batch or row bound. **All modes now default to 8-token eligibility**, so `on` + speculation no longer
> silently degrades. |

**Consequences for us.** (a) The admission-policy surface the RFC's own "Maintenance burden" section warns
about is smaller in v2 than the body implies — one less heuristic to port. (b) The eligibility gate is
expressed as a *CUDA compute capability*, which has no meaning on an Arc Pro B70; a SYCL port needs its own
device-capability fence and has no reference for what it should test. (c) The "8-token eligibility" bound is
the thing that interacts with MTP (§7).

### 1.3 The thread's most load-bearing external result: the reconciliation rule

`mfethe1`, 2026-09-02, consolidating the thread's contradictory datapoints (quoted verbatim):

> **The reconciliation rule.** Compare the cache budget to the model's expert working set at equal total VRAM:
>
> - **Budget ≥ working set** → resident layers strictly dominate. Our matched-budget A/B (single 5060 Ti 16 GB,
>   Qwen3-30B-A3B Q4_K_M, pp512/tg128, ABBA ×3): resident 149.6 / 35.4 t/s vs cache 64.0 / 19.7 t/s at ~12.5 GiB
>   — despite the cache arm reaching 85.3% hit rate with zero fill failures.
> - **Budget < working set** → the cross-layer cache wins, and the margin grows the deeper below WS you are.

**This corrects a misattribution carried in our own planning docs.** `PLAN.md:148-154` quotes
`docs/00-background.md` §1 as recording: "Lost at matched VRAM budget in one external report (llama.cpp
discussion #24528, hybrid CPU/GPU design, 19.7 vs 35.4 tok/s vs. plain resident layers)". Those are exactly
`mfethe1`'s numbers — but they are a measurement of **leloch's own branch**, i.e. of the RFC design itself,
in the *budget ≥ working set* regime (Qwen3-30B-A3B Q4_K_M is 17.28 GiB total against a ~12.5 GiB budget).
They are not a measurement of a different design that the RFC's approach avoids. The honest reading is:
**the 19.7-vs-35.4 result is the RFC design losing in a regime we are not in.**

Our regime, per `docs/research/02` §5 as summarized at `PLAN.md:383-386`: "no quant fits fully in 32 GB VRAM
— smallest published quant's expert set alone is 39.85 GB", with the dense core only 3.9-5.5 GB. At
UD-IQ3_XXS with ~30 GB usable, the budget is roughly half the expert working set. **Inference:** that places
us on the `budget < working set` side of `mfethe1`'s rule, which is the side where the thread's evidence says
the cache design wins — but not deeply below it, so the predicted margin is at the modest end, not the +57%
end. `mfethe1`'s companion sweep (2026-09-03) found the cache arm "budget-insensitive" at 18.7-19.0 t/s flat
across 10→15 GiB, i.e. extra budget bought nothing until it approached the full working set — a shape that,
if it reproduces, predicts our slot-pool sizing knob will not be the lever either.

### 1.4 Prior attempts the RFC cites — all verified by title and state

| Ref | Title | Kind / state |
|---|---|---|
| #20757 | Feature Request: Two-tier GPU+RAM expert cache for MoE offload (pluggable eviction policy) | issue, closed 2026-04-08, 19 comments |
| #21609 | expert-cache: N-slot LFRU cache with FATE prefetch for MoE weight offloading | PR, closed after **7 minutes** |
| #21614 | ggml : persistent expert cache for `--n-cpu-moe` (RFC #20757) | PR, closed after ~80 minutes |
| #21620 | ggml : dedup expert cache for MoE CPU offload | PR, closed after ~28 minutes |
| #23170 | ggml: treat experts as cache residents during MoE offloading | PR, closed 2026-05-25 (9 days open) |
| #22584 | Profiling Experts on MoE and strategic placement | discussion (Ideas), 1 comment |
| #19030 | Batch-aware weight prefetching for offloaded inference / JIT weight prefetch | discussion (Ideas), 9 comments |
| #21067 | ggml: allow prefetching tensor overrides | PR, **still open**, 22 comments |
| #22183 | MoE offload to second (slower) GPU rather than to CPU | discussion (Ideas), 12 comments |
| #24524 | cuda: MoE expert cache, adaptive VRAM caching of CPU-resident experts | PR, closed after **10 minutes** |

The pattern worth internalizing: **four separate PRs in this line were closed within 80 minutes of opening.**
This is not a space where upstream is waiting for a good implementation — it is a space where upstream has
repeatedly declined the whole category. That is an argument against investing in upstreamability, not against
the technique itself for our own use.

---

## 2. The reference implementation's actual shape

`gh api .../compare/master...leloch:llama.cpp:moe-cache-v2-pr` returns 28 changed files. The ones that define
the architecture:

| File | +/- | Role |
|---|---|---|
| `ggml/src/ggml-backend-moe-cache.h` | +109 | **added** — the backend-agnostic function-pointer table |
| `ggml/src/ggml-cuda/moe-cache.cu` | +3145 | **added** — the CUDA provider |
| `ggml/src/ggml-cuda/moe-cache.cuh` | +19 | **added** — 19 lines, i.e. essentially just a registration decl |
| `ggml/src/ggml-cpu/ggml-cpu.c` | +445 / -3 | the CPU hook (mul_mat_id + the fused gate/up/GLU detector) |
| `ggml/src/ggml-backend.cpp` | +130 / -2 | scheduler session enter/leave + `ggml_backend_sched_set_moe_cache` |
| `ggml/src/ggml-cuda/mmvq.cu` / `.cuh` | +253/-40, +19 | the batched hit-row matvec entry point |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | +21 | registers the provider into the table |
| `ggml/src/ggml-backend-reg.cpp` | +3 | table init |
| `common/fit.cpp` / `fit.h` | +463/-43, +62 | VRAM fit rule |
| `common/arg.cpp`/`common.cpp`/`common.h` | +38, +29, +14 | `--moe-cache` flag |
| `src/llama-context.cpp`, `llama-cparams.h`, `llama-ext.h`, `llama-model.cpp`, `include/llama.h` | +100, +3, +13, +36, +10 | session lifetime plumbing |
| `tests/test-moe-cache.cpp`, `test-moe-cache-fit.cpp`, `test-arg-parser.cpp` | +3154, +170, +126 | tests |
| `tools/llama-bench`, `tools/server/server-context.cpp`, `tools/fit-params` | +307/-35, +306/-29, +13/-1 | tooling |
| `docs/backend/CUDA-MOE-CACHE.md` | +261 | docs |

Note the RFC body says "~1,700 lines in one new CUDA file pair"; the actual v2 pair is **3,164 lines**
(`moe-cache.cu` 3,145 + `.cuh` 19), plus a 3,154-line test. The body's number is either the pre-v2 count or an
understatement. **Treat "~1,700" as not reflecting the branch that exists.** There is also a newer
`v3-expert-cache` branch on the same fork whose provider is smaller (`expert-cache.cu` +2045,
`ggml-backend-expert-cache.h` +103, `ggml-cpu.c` +70/-1, `ops.cpp` +13, `ggml-backend.cpp` +97) — this looks
like the "minimal core" the RFC offered, and is the closer model for what we would build.

### 2.1 The API contract, read in full

`ggml-backend-moe-cache.h` (109 lines, read in full). The per-node protocol, quoting the header's own comments:

```c
    // Begin one CPU MUL_MAT_ID node. Returns an opaque plan, or NULL when the stock CPU path should handle the complete node.
    void * (*begin)(const char * tensor_name, const void * host_base, size_t expert_size,
                    int64_t n_in, int64_t n_out, int wtype, int64_t n_expert,
                    int64_t n_tokens, int64_t n_rows);

    // Mark cache hits and enqueue bounded demand fills for misses. A nonnegative slot index means that the row may be omitted from CPU work only if dispatch subsequently succeeds.
    int (*plan)(void * node, const int32_t * ids, int n_ids, int32_t * slot_idx);

    // Dispatch all planned hit rows. Returns 1 only after the complete GPU operation has been accepted.
    // On 0, the caller must restore every row to the normal CPU mapping before worker threads start.
    int (*dispatch)(void * node, int wtype, int64_t n_in, int64_t n_out, int n_hits,
                    const int32_t * slot_idx, const float * const * act_rows);

    // Copy GPU results into dst_rows. On 0, the caller must recompute every skipped row on the CPU.
    int (*collect)(void * node, int n_hits, float * const * dst_rows, int64_t n_out);

    // Releases slot pins and all per-node ownership. Must be called exactly once for every non-NULL begin result, on every success or failure path.
    void (*end)(void * node);
```

Three details that are load-bearing and easy to miss:

1. **`act_rows` is `const float * const *` and `dst_rows` is `float * const *`** — arrays of *host* row
   pointers, F32 in and F32 out. The provider owns all staging, quantization and scatter/gather. §4.4 explains
   why the F32-in choice is forced rather than stylistic.
2. **`dispatch` is synchronous in its contract** — "Returns 1 only after the complete GPU operation has been
   accepted", and on failure the caller unwinds "before worker threads start". So dispatch happens in the
   serial `ith == 0` prologue, before the threadpool barrier, not concurrently with it.
3. The config struct carries `int32_t overlap_cpu_rows;` with the comment "`-1` selects the provider policy;
   `0..8` is a fixed total per node" — i.e. the reference design found it worth *also* computing a few hit rows
   on the CPU to soak up latency. **Inference:** that knob's existence is evidence the dispatch round trip is
   not free even on CUDA, and is a direct hint about where our SYCL risk will show up.

The table is a single global (`GGML_API struct ggml_moe_cache_api ggml_moe_cache;`) with
`ggml_moe_cache_unregister(const void * owner)` and
`ggml_backend_sched_set_moe_cache(sched, mode, budget_mib)`. Non-registering backends see a zero-initialized
table; every call site null-checks the pointers.

### 2.2 The CPU hook, read from the actual patch

The `ggml-cpu.c` patch renames `ggml_compute_forward_mul_mat_id` to `..._impl` and adds three parameters
(`uint64_t row_mask, bool use_row_mask, bool allow_moe_cache`), keeping the old name as a thin wrapper. The
substantive additions, in order:

- **`#define MOE_CACHE_MAX_TOPK 64`** and a set of `[MOE_CACHE_MAX_TOPK]` stack arrays declared right after
  `n_as`, with the comment "MoE expert cache state is used by thread 0 only."
- Inside the existing `if (ith == 0) {` block, **before** the row-grouping loop: eligibility test
  (`ggml_moe_cache.begin && .plan && .dispatch && .collect && .end && src0->op == GGML_OP_NONE &&
  ggml_backend_buffer_is_host(src0_buffer) && usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
  src1->type == GGML_TYPE_F32`), then `begin(...)`, then a flat read of every routed id into
  `expert_ids[]`, then `plan(node, expert_ids, n_ids * ids->ne[1], moe_cache_slot_idx)`.
- **Inside the row-grouping loop**, a `continue` that diverts hit rows away from the existing bookkeeping:
  ```c
  if (moe_cache_node && moe_cache_slot_idx[iid1*n_ids + id] >= 0) {
      ...
      moe_cache_acts[moe_cache_n_hits] = (const float *) ((const char *) src1->data + i11*nb11 + iid1*nb12);
      moe_cache_rows[moe_cache_n_hits] = (float *) ((char *) dst->data + iid1*nb2 + id*nb1);
      moe_cache_n_hits++;
      continue;
  }
  MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = (struct mmid_row_mapping) {id, iid1};
  matrix_row_counts[i02] += 1;
  ```
- Still at `ith == 0`, still before the barrier: `dispatch(...)`; on a `0` return, every diverted row is pushed
  back into `MMID_MATRIX_ROW`/`matrix_row_counts` and `end()` is called.
- After the main `for (cur_a...)` loop: `if (ith == 0 && moe_cache_node)` → `collect(...)`; on a `0` return,
  each hit row is recomputed by calling `ggml_compute_forward_mul_mat_id_one_chunk` directly with
  `ir1_start = row, ir1_end = row + 1`. Then `end()`.
- `ggml_graph_plan` gains `work_size += MOE_CACHE_FUSED_WORK_SIZE;`.

**This is the single most important structural finding in this document.** The hook does not modify the
per-thread chunking logic at all. It removes hit rows from `matrix_row_counts` / `MMID_MATRIX_ROW` *before*
the chunking ever reads them. Every `nchunk0`/`nchunk1`/`dr0`/`dr1`/`atomic_current_chunk` computation
downstream simply sees a smaller `cne1` for the affected experts — often zero, which the existing
`if (cne1 == 0) continue;` already handles. There is **no** new synchronization, **no** new barrier, and **no**
change to `ggml_compute_forward_mul_mat_id_one_chunk`.

---

## 3. Mapping the design onto our tree

### 3.1 `ggml_compute_forward_mul_mat_id` — the hook window

`ggml/src/ggml-cpu/ggml-cpu.c:1542-1732` (the task brief's "~line 1542" is exact for current HEAD, commit
`6d9c82ea2`). Structure as read:

| Region | Lines | What it does |
|---|---|---|
| signature | `1542-1544` | `(const struct ggml_compute_params * params, struct ggml_tensor * dst)` |
| `ith`/`nth` | `1552-1553` | |
| `n_ids = ids->ne[0]`, `n_as = ne02` | `1573-1574` | 10 and 512 for qwen4exp |
| `wdata` carve-up: `matrix_row_counts[n_as]`, `matrix_rows[n_as][ids->ne[0]*ids->ne[1]]`, `atomic_current_chunk[n_as]` | `1576-1601` | via `incr_ptr_aligned` (`1534-1540`) |
| **all-threads** `from_float` of `src1` into `wdata` | `1603-1638` | each thread does a slice of `ne10` (`1629-1630`) |
| **`if (ith == 0)`**: zero `matrix_row_counts`, build `MMID_MATRIX_ROW` | `1640-1655` | **← `begin`/`plan`/divert/`dispatch` go here** |
| reset per-expert chunk counters | `1657-1661` | |
| **`ggml_barrier(params->threadpool)`** | `1663` | the only barrier in the function |
| `for (cur_a = 0; cur_a < n_as; ++cur_a)` | `1665-1731` | skip on `cne1 == 0` (`1668-1670`); optional IQ-panel path (`1672-1677`); chunking (`1683-1708`); work-stealing loop (`1709-1730`) |
| (function ends) | `1732` | **← `collect`/`end` go here** |

Two facts about this window that constrain the port:

**(a) Thread 0 cannot read `params->wdata`.** The `from_float` loop at `1603-1638` is executed by *all*
threads over disjoint `ne10` slices, and the only barrier is at `1663` — *after* the `ith == 0` block. So when
thread 0 runs the prologue, other threads' quantization slices may still be in flight. This is exactly why the
reference implementation passes `src1->data` (F32) and not `wdata`:
`moe_cache_acts[...] = (const float *)((const char *) src1->data + i11*nb11 + iid1*nb12)`. Any SYCL port must
do the same — **the GPU side gets F32 activations and must quantize them itself** (or use an F32 path).

**(b) Thread 0 is the caller's own thread, in both threadpool modes.** With OpenMP,
`ggml-cpu.c:3419-3438` runs `#pragma omp parallel` and dispatches `ggml_graph_compute_thread(&threadpool->workers[ith])`
with `ith = omp_get_thread_num()`, so `ith == 0` is the encountering thread. Without OpenMP,
`ggml-cpu.c:3450-3453` kicks workers `1..n-1` and then comments `// This is a work thread too` before calling
`ggml_graph_compute_thread(&threadpool->workers[0])` on the calling thread. **In both builds, `ith == 0` is the
same OS thread that called `ggml_backend_sched_graph_compute`** — and therefore the same thread that ran SYCL
initialization. §4.2 explains why that specific fact defuses the largest SYCL correctness hazard.

Other threads reaching `ggml_barrier` at `ggml-cpu.c:3157` (the per-node barrier in
`ggml_graph_compute_thread`, `3101-3174`) while thread 0 blocks on a GPU event is functionally correct: the
node's `dst` is not read until after that barrier. The cost is that the node takes
`max(GPU_hit_path, CPU_miss_path)` plus the dispatch/collect latency, not their sum.

### 3.2 The scheduler — what breaks, and the surprise

The brief asks what scheduler assumption a row-split op breaks. The answer, from a full read of
`ggml/src/ggml-backend.cpp`:

**The assumption is `#define tensor_backend_id(tensor) sched->hv_tensor_backend_ids[hash_id(tensor)]`
(`ggml-backend.cpp:844`)** — a single `int` lvalue per tensor, written in every assignment pass
(`:1090, 1099, 1137, 1158, 1179, 1195, 1217, 1268, 1278, 1306, 1320, 1335, 1519, 1552, 1580, 2085`).
`struct ggml_backend_sched_split` (`:775-784`) is `{ backend_id; i_start; i_end; inputs; n_inputs; ... }` — a
backend plus a contiguous node range, materialized by `ggml_graph_view(graph, i_start, i_end)` at `:1460` and
handed wholesale to `ggml_backend_graph_compute_async(split_backend, &split->graph)` at `:1792`. There is no
per-node callback, no sub-op fan-out, and the CPU backend holds no handle to any other backend. Confirmed: no
precedent anywhere in the file for one node being executed by two backends.

**But the RFC's design never asks the scheduler for that.** The node stays 100% assigned to the CPU backend;
the GPU participation happens *inside* the CPU backend's own `graph_compute`, invisible to the scheduler. The
`ggml-backend.cpp` changes in the reference branch (+130/-2) are session lifetime
(`session_create`/`enter`/`leave`) and `ggml_backend_sched_set_moe_cache`, not assignment logic — consistent
with the header's own comment, "The scheduler owns one cache session."

**And here is the part worth pausing on: our current `-ncmoe` production baseline already puts these nodes on
the CPU backend at decode, for free.** The chain:

1. `-ncmoe N` (`common/arg.cpp:2782-2790`) calls `llm_add_n_cpu_ffn_overrides` (`common/common.h:1143-1150`),
   which pushes `{blk\.<i>\.ffn_(up|down|gate)_exps\., ggml_backend_cpu_buffer_type()}` for `i < N`.
2. `ggml_backend_sched_backend_id_from_cur` (`ggml-backend.cpp:921-985`) reaches the weight branch at
   `:951-982` and calls `ggml_backend_sched_backend_from_buffer` (`:888-908`), whose loop at `:895-900` returns
   the first backend with both `supports_buft` and `supports_op`. The SYCL backend rejects the CPU buft —
   `ggml_backend_sycl_device_supports_buft` (`ggml-sycl.cpp:7370-7389`) returns `false` at `:7383-7385` unless
   `get_name == ggml_backend_sycl_buffer_type_get_name`. So `src_backend_id` comes back as the CPU
   (`n_backends - 1`, guaranteed last by the assert at `ggml-backend.cpp:1850`).
3. The offload override at `:970-977` is the only escape, and it requires
   `ggml_backend_offload_op(SYCL, node)` → `ggml_backend_sycl_device_offload_op` (`ggml-sycl.cpp:7405-7408`)
   → `get_op_batch_size(op) >= op_offload_min_batch_size`. For `GGML_OP_MUL_MAT_ID`,
   `get_op_batch_size` returns `op->ne[2]` (`ggml-sycl.cpp:7397-7399`) — the token count — and the default
   `min_batch_size` is **32** (`ggml-sycl.cpp:7790`, `GGML_OP_OFFLOAD_MIN_BATCH`). At decode (`ne[2] == 1`)
   this is false.
4. Pass 2 explicitly refuses to expand into the CPU backend (`:1126-1127` comment, breaks at `:1139-1141`,
   `:1160-1162`). Pass 3's upgrade branch (`:1240-1262`) requires
   `sched->bufts[b] == sched->bufts[*node_backend_id]` at `:1243` — the SYCL device buft is not the CPU buft,
   so no upgrade fires.

**Conclusion: at decode, with `-ncmoe`, `GGML_OP_MUL_MAT_ID` over CPU-placed experts already executes in
`ggml_compute_forward_mul_mat_id` on the CPU backend, with zero scheduler modification.** That is precisely
the RFC's `-ngl 99 -ncmoe 99` benchmark arm, and precisely the function the hook goes into. The scheduler work
in our port is **zero for the decode path**, which is the only path the RFC's design claims to improve.

For completeness, the prefill path (`ne[2] >= 32`) *does* offload to SYCL, and it takes the used-experts-only
copy at `ggml-backend.cpp:1690-1774` — which reads the ids back to host with a blocking
`ggml_backend_synchronize(ids_backend)` at `:1721` after `ggml_backend_tensor_get_async` at `:1720`, then
issues run-length-merged `ggml_backend_tensor_set_async` copies at `:1744-1749`. **Inference:** this is stock
upstream already paying "a synchronous readback plus a PCIe copy per MoE node" on the prefill path — the exact
cost class the RFC criticizes — and the RFC's decode-only design leaves it entirely untouched. Our measured
prefill behavior under `-ncmoe` is therefore unaffected either way, which is a point in favor: the thread's
consistent finding is that every cache design loses prefill (`mfethe1`: "Prefill almost never benefits. Every
cache arm in this thread loses prefill (ours by 2.34×, SharkWipf's by ~2×)").

### 3.3 There is already a CPU-backend op-interception mechanism, and nobody in the RFC thread used it

`ggml/src/ggml-cpu/traits.h:11-33` defines:

```c
// return true if op part of extra "accelerator"
bool ggml_cpu_extra_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op);
bool ggml_cpu_extra_work_size(int n_threads, const struct ggml_tensor * op, size_t * size);
```
```cpp
namespace ggml::cpu {
class tensor_traits { virtual bool work_size(...); virtual bool compute_forward(ggml_compute_params *, ggml_tensor *) = 0; };
class extra_buffer_type { virtual bool supports_op(ggml_backend_dev_t, const ggml_tensor *) = 0;
                          virtual tensor_traits * get_tensor_traits(const ggml_tensor *) = 0; };
}
```

`ggml_compute_forward` calls `ggml_cpu_extra_compute_forward(params, tensor)` and returns early on `true` at
`ggml-cpu.c:1744-1746` — i.e. **an extra buffer type can already claim an entire op and compute it itself,
with the full `ggml_compute_params` including `ith`/`nth`/`threadpool`.** Registration is
`ggml_backend_cpu_get_extra_buffer_types()` (`ggml-cpu.cpp:42-74`); the scheduler sees them via
`ggml_backend_cpu_device_supports_buft` (`ggml-cpu.cpp:483`:
`return ggml_backend_buft_is_host(buft) || ggml_backend_cpu_is_extra_buffer_type(buft);`) and
`ggml-cpu.cpp:436-437`. Current users are AMX, KleidiAI, RISC-V SpaceMiT and CPU-repack (`:46-68`).

There is also `ggml_cpu_try_fuse_ops(cgraph, node_n, params, cplan)` at `ggml-cpu.c:3067-3099`, called from
`ggml_graph_compute_thread` at `:3143-3145`, which today only fuses `RMS_NORM + MUL` (`:3079-3096`). It
receives the whole `cgraph` and the node index — **exactly the shape the RFC's fused gate+up+SwiGLU detector
needs**, and exactly what the reference implementation's `ggml_moe_cache_can_fuse(cgraph, node_n, fusion)`
re-implements alongside it.

**Recommendation:** if a spike graduates to a real implementation, prefer an `extra_buffer_type` +
`tensor_traits::compute_forward` over a raw global function-pointer table, and extend `ggml_cpu_try_fuse_ops`
rather than adding a parallel fusion path. Both are already-upstream extension points with existing consumers.
This is a genuine simplification over the reference design and is the single largest reduction available in
the "hooks the CPU hot path" maintenance cost the RFC calls "the largest structural cost."

**Caveat (inference, not verified by running anything):** `tensor_traits::compute_forward` replaces the whole
op, so a trait-based implementation must *reimplement* `ggml_compute_forward_mul_mat_id`'s body rather than
patch into it, duplicating ~90 lines of chunking logic. The in-place patch keeps one copy of that logic at the
cost of touching the stock function. That is a real trade-off and should be decided with the code in front of
you, not here.

### 3.4 What of our existing SYCL cache is reusable

`ggml/src/ggml-sycl/moe-cache.cpp` (1005 lines) + `moe-cache.hpp` (207 lines), plus the hook sites in
`ggml-sycl.cpp`. Component by component:

**Reusable essentially as-is:**

| Component | Location | Why it transfers |
|---|---|---|
| `ggml_sycl_mul_mat_vec_q_id` | declared `ggml-sycl/mmvq.hpp:29-41`, defined `mmvq.cpp:2774` | This *is* the batched hit-row matvec the RFC's `dispatch` needs. Signature `(src0_type, vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used, expert_weight_stride, dst_row_stride, src1_row_stride, stream)` — `vx_base` = pool base, `ids_dev` = per-row slot indices, `dst_row_stride` = output stride. Already quant-type-generic, already driven off a remapped slot table, already used exactly this way at `ggml-sycl.cpp:5180-5185`. **No modification needed.** |
| The VRAM slot pool | `moe-cache.cpp:172-174` (`pool`, `slot_stride`, `n_slots`), allocated `:220` | A flat `n_slots * slot_stride` `sycl::malloc_device` region with uniform stride. Exactly the "persistent buffer." |
| The device-side gather kernel | `moe_submit_gather` host wrapper `:498-514`, kernel `:516-553`, alignment dispatch `moe_submit_gather_dispatch:557-585` | Bulk coalesced pinned-host→VRAM copy at 16/8/4/1-byte word granularity, grid-strided over `(miss, chunk)` blocks. This is the fill path. The `83×` measurement table justifying it is at `:446-450`. **Keep.** |
| The decayed-LFU admission kernel | `moe_plan_effective_frequency:404-407`; the plan kernel `:754-974` with its `reduce_over_group(minimum<uint64_t>)` at `:919-920` and the pack/unpack helpers `:424-429` | This is hard-won, hardware-debugged code. The nine `local_accessor`s at `:762-795` each carry a comment documenting a real bug found on the B70 — the protected-expert list at `:767-777`, the local slot mirror at `:778-790`, the slot-pinning race at `:791-795` / `:934-945`. **Do not rewrite this.** |
| `quantize_row_q8_1_sycl` | used at `ggml-sycl.cpp:5155-5156` | Device-side F32→Q8_1 for the activation rows. Needed because `dispatch` receives F32 (§2.1). |
| `GGML_SYCL_MOE_CACHE_SIZE` / profiler | `moe-cache.cpp:271-282`, `:605-694` | Sizing knob and the per-token profiler (`profile_tick(144)`, `:649-675`). |

**Needs modification:**

| Component | Location | Change |
|---|---|---|
| `ggml_sycl_moe_cache_plan_and_gather` | `moe-cache.cpp:696-1005` | Today it takes `ids_dev` as a **device** pointer (the router's own output, already in VRAM) and returns a **device** pointer to `remapped_ids`. Under the RFC design the ids arrive as a **host** `int32_t*` from `plan()`, and the slot indices must be returned to the **host** so `ggml-cpu.c` can decide which rows to divert. That inverts the data flow: a small H2D of the ids and a small D2H of the slot table, or — better — move the admission decision to the host and keep only the gather on device. **This is the single largest code change on the SYCL side.** |
| Cache lifetime / registry | `moe-cache.cpp:284-309` (`g_moe_caches` keyed by `const ggml_tensor *`) | Still works — `begin()` receives `src0->name` and `host_base`; keying on the host base pointer is equivalent and matches the reference API. But the registry is a process-global with a mutex (`:284`) and is not per-`ggml_backend_sycl_context`; it must now be reachable from CPU-backend code, which is a plumbing change, not a design change. |
| Queue acquisition | every entry point takes `sycl::queue &` (`moe-cache.hpp:65, 74, 83, 99, 180`), resolved by the caller as `ctx.stream()` (`ggml-sycl.cpp:5138`) | There is no `ggml_backend_sycl_context` on the CPU side. A queue must be captured at registration time and stored. §4 is entirely about this. |
| Cold-tier device-0 hardcoding | `moe-cache.cpp:90, 106, 149` | Uses `dpct::dev_mgr::instance().get_device(0).default_queue()` unconditionally. Fine on a single-GPU box, wrong in general. |

**Made obsolete by this design:**

| Component | Location | Why |
|---|---|---|
| **The whole `SYCL_MoE_Cached` buffer type** | `moe-cache.cpp:82-163`, registered at `ggml-sycl.cpp:7380-7382` and `:7748-7750` | The RFC hook's eligibility test requires `ggml_backend_buffer_is_host(src0_buffer)`. Our buft deliberately returns `false` from `is_host` (`moe-cache.cpp:131-135`) specifically so the *SYCL* backend claims the node instead of the CPU backend. That is the exact opposite of what the new design needs. See §3.5 for the replacement. |
| The Phase-3a host-LRU path | `moe-cache.cpp:176-179` (fields), `:311-348` (`acquire`), and its caller `ggml-sycl.cpp:5325-5398` | Superseded. Also currently broken-by-design in one respect: `ggml_sycl_moe_cache_barrier` (`:350-372`) **blocks the host** with `e.wait()` at `:370` despite `moe-cache.hpp:77-83` documenting it as a non-blocking device barrier; the code comment at `:351-364` records that the `ext_oneapi_submit_barrier` version was reverted after producing non-deterministic greedy decode. Header and implementation disagree today. |
| `ggml_sycl_mul_mat_id_device_planned_fused` | `ggml-sycl.cpp:5123-5186` | Its guard conditions and structure become the model for the new `dispatch`, but the function itself is a SYCL-backend-side interception that no longer fires once the node lives on the CPU backend. |
| The ids readback drain | `ggml-sycl.cpp:5296-5302` (`stream->memcpy(...)` + `stream->wait()`) | The comment at `:5103-5105` records this "was found to drain the ENTIRE in-order queue on every one of ~144 MoE tensor ops per token, dominating all of Phase 3a/3b's cost." Under the new design the ids are already on the host. |

**Net:** roughly the gather kernel, the plan kernel, the pool, and `ggml_sycl_mul_mat_vec_q_id` survive — call
it 600-700 of the 1005 lines — and the buffer type plus the legacy path (roughly 300 lines) go away.

### 3.5 The cold tier: a free win hiding in `ggml_backend_sycl_host_buffer_type`

`-ncmoe` places experts in `ggml_backend_cpu_buffer_type()` (`common/common.h:1140, 1148`) — plain
`malloc`'d memory. A Level Zero device kernel cannot dereference that, so the existing gather kernel
(`moe-cache.cpp:535`, `src = cold_base + eid*row_bytes`, read directly by the device) would not work against
it. The fill would have to degrade to `queue::memcpy` from pageable host memory. And enabling system USM is
not available to us: `PLAN.md:247-251` makes `GGML_SYCL_USM_SYSTEM=1` an absolute prohibition, tied to the
sibling project's two host-wide OOM kills.

But `ggml_backend_sycl_host_buffer_type()` (`ggml-sycl.cpp:1601-1617`) is:

- backed by `sycl::malloc_host` (`ggml_backend_sycl_host_malloc`, `:1540-1556`, allocation at `:1546`, with
  the comment "USM host memory is page-locked and device-accessible by construction"), enabled by default —
  `g_ggml_sycl_enable_host_pinned_mem = 1` at `:120`, settable via `GGML_SYCL_ENABLE_HOST_PINNED_MEM` (`:379`);
- **and reports `is_host` = the CPU buffer type's `is_host`** (`:1610`:
  `/* .is_host = */ ggml_backend_cpu_buffer_type()->iface.is_host,`), i.e. **true**;
- with a 2 GiB max-size cap already implemented behind `GGML_SYCL_HOST_PINNED_MEM_2G` (`:1586-1598`,
  `:1591-1592`), which is `PLAN.md:252-256`'s required ≤2 GiB chunking rule, off by default (`:121`).

**Inference (connecting `ggml-sycl.cpp:1610` with `ggml-backend.cpp:895-900` and `ggml-cpu.cpp:483`):**
placing the `-ncmoe` expert tensors in `SYCL_Host` instead of `CPU` would (a) still land the `MUL_MAT_ID` node
on the CPU backend, because SYCL's `supports_buft` rejects it by name (`ggml-sycl.cpp:7383-7385`) while the
CPU backend accepts any host buft (`ggml-cpu.cpp:483`); (b) satisfy the hook's
`ggml_backend_buffer_is_host(src0_buffer)` test; and (c) give the gather kernel a directly-dereferenceable
pinned source with no copy. That is the CUDA fork's `cudaHostGetDevicePointer` zero-copy alias
(`docs/research/03` Topic 4, `device_alias` at `moe-cache.cu:4512-4568`) obtained for free, because SYCL USM
host allocations are device-accessible by construction.

**This needs verification before being relied on.** Two concrete checks: (1) `SYCL_Host` is not currently
nameable from `-ot` — `parse_tensor_buffer_overrides` (`common/arg.cpp:252-303`) builds its name map from each
device's default buft (`:258-261`) plus its extra bufts (`:269-280`), and `SYCL_Host` is neither; adding it to
`ggml_backend_sycl_moe_cached_get_extra_bufts` (`moe-cache.cpp:159-163`) is a one-line change. (2) The pinned
allocation goes through `ggml_backend_cpu_buffer_from_ptr` with an overridden `free_buffer`
(`ggml-sycl.cpp:1579-1581`, comment: "FIXME: this is a hack to avoid having to implement a new buffer type"),
so whether ~20-26 GB of expert weights actually allocate as pinned USM without hitting the driver's pinned-memory
limits is an empirical question — and `docs/research/02` §3.3's still-open issue #25812
(`UR_RESULT_ERROR_OUT_OF_HOST_MEMORY` when offloading MoE experts to Arc GPUs) is adjacent enough that it
should be assumed to bite until proven otherwise. `PLAN.md:526-532` already flags exactly this.

---

## 4. The headline risk: SYCL dispatch-and-wait from a CPU worker thread

This section is the one the brief asked not to hand-wave, so it is the longest.

### 4.1 What CUDA gets for free here

The reference design's `dispatch` is contractually synchronous ("Returns 1 only after the complete GPU
operation has been accepted"), and `collect` must land results in host `dst_rows` before the CPU threads'
work is joined. On CUDA that is: `cudaMemcpyAsync` H2D on a private stream from any host thread, a kernel
launch on that stream, `cudaMemcpyAsync` D2H, and a `cudaStreamSynchronize` or event poll — all from a
non-primary thread, on a stream independent of whatever the rest of the process is doing, with a launch cost
in the single-digit microseconds. None of those four properties is automatic in SYCL.

### 4.2 Thread identity — the risk that turns out to be defused

`dpct::dev_mgr::current_device_id()` (`ggml/src/ggml-sycl/dpct/helper.hpp:877-884`) is **keyed on the calling
thread**:

```cpp
unsigned int current_device_id() const {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    auto it = _thread2dev_map.find(get_tid());
    if (it != _thread2dev_map.end()) return it->second;
    return DEFAULT_DEVICE_ID;
}
```

and `select_device(id)` (`:889-894`) writes `_thread2dev_map[get_tid()] = id`. So **any call to
`dpct::get_current_device()` / `dpct::get_default_queue()` (`:1108-1111`, `:2080-2083`, `:2090-2093`) from a
thread that never called `select_device` silently resolves to `DEFAULT_DEVICE_ID`, not the backend's selected
device — no error, no warning.** On a multi-GPU box that is a silent wrong-device bug.

Two mitigations, and they stack:

1. Per §3.1(b), `ith == 0` is the same OS thread that ran SYCL init, so its TLS mapping is already correct.
   **This is why the hook must be thread-0-only, not "any thread that gets there first."** That is also what
   the reference implementation does ("MoE cache state is used by thread 0 only").
2. Regardless, the port should never call the thread-local path. Capture an explicit `sycl::queue *` at
   registration time and store it in the provider's state. Our `moe-cache.cpp` already takes queues as
   explicit `sycl::queue &` parameters everywhere (`moe-cache.hpp:65, 74, 83, 99, 180`) — the right pattern is
   already established; only `moe_cached_pinned_malloc`/`free` (`moe-cache.cpp:90, 106`) and
   `ggml_backend_sycl_host_malloc` (`ggml-sycl.cpp:1545`, `:1563`) use the singleton, and both are allocation-
   time, not per-token.

**SYCL 2020 makes `sycl::queue` itself thread-safe for submission**, so the cross-thread submit is legal.
The risk here is not "SYCL forbids it"; it is the dpct TLS trap and the queue-choice problem below.

### 4.3 There is exactly one queue per device, and it is in-order

`ggml_backend_sycl_context::stream(device, stream)` (`ggml/src/ggml-sycl/common.hpp:350-355`):

```cpp
queue_ptr stream(int device, int stream) {
    if (qptrs[device][stream] == nullptr) {
        qptrs[device][stream] = &(dpct::get_device(device).default_queue());
    }
    return qptrs[device][stream];
}
```

Despite `qptrs[GGML_SYCL_MAX_DEVICES][GGML_SYCL_MAX_STREAMS]` (`common.hpp:342`) with
`GGML_SYCL_MAX_STREAMS 8` (`presets.hpp:16`), **every index returns the same object** —
`dpct::get_device(device).default_queue()`, which is `in_order_queue()` (`dpct/helper.hpp:727-731`), created
by `init_queues()` (`:793-797`) as `create_queue_impl(true, sycl::property::queue::in_order())`. So
`ggml-sycl` has exactly **one in-order queue per device**, and `PLAN.md:557-558` already recorded this ("no
second queue exists anywhere in `ggml-sycl` today").

Consequences:

- **Do not submit the hit-row matvec onto `ctx.stream()`.** It is in-order and shared with the SYCL backend's
  own graph execution; a submit from thread 0 would serialize behind whatever the SYCL split last enqueued,
  and would also perturb that split's ordering.
- A private queue is required. `sycl_moe_cache`'s constructor already shows the pattern:
  `copy_queue(compute_q.get_context(), compute_q.get_device())` at `moe-cache.cpp:218` — same context, same
  device, new queue. `PLAN.md:558-560` flags the cross-context silent-no-op trap; this construction avoids it.
  **However:** `moe-cache.cpp:218` constructs it **without** `property::queue::in_order`, while the whole
  design's correctness argument elsewhere (`moe-cache.cpp:736-740`, `:999-1004`) rests on in-order semantics.
  That inconsistency exists in code we already ship and should be fixed regardless of this port.
- A private queue also means the hit-row work genuinely overlaps the CPU miss work rather than queueing
  behind graph work — which is the entire performance premise.

### 4.4 What one `dispatch`/`collect` round trip actually costs

The minimum sequence per hooked node, given the API's F32-in/F32-out contract (§2.1) and the constraint that
thread 0 cannot read `wdata` (§3.1(a)):

1. Gather the `n_hits` F32 activation row pointers into one contiguous staging buffer (CPU memcpy, tens of KB).
2. H2D that buffer (`queue::memcpy`).
3. `quantize_row_q8_1_sycl` (`ggml-sycl.cpp:5155`) — one kernel.
4. Optionally the plan+gather kernels for misses, if fills are enqueued this node.
5. `ggml_sycl_mul_mat_vec_q_id` (`mmvq.hpp:29-41`) — one kernel.
6. D2H the `n_hits × ne01` F32 results.
7. Host wait on the final event; scatter results into `dst_rows` (CPU memcpy).

That is 2 transfers + 2-4 kernels + **one host-visible wait**, per hooked node.

**Volume is not the problem.** With `n_expert_used = 10` (per `PLAN.md`'s architecture summary) the per-node
traffic is `10 × ne00 × 4` bytes in and `10 × ne01 × 4` bytes out — tens of kilobytes. Even at 144 nodes/token
and 47 tok/s this is a low-hundreds-of-MB/s PCIe load. **Latency is the problem.**

**The arithmetic, explicitly labelled inference.** qwen4exp has 48 layers × 3 expert tensors
(`ffn_gate_exps`/`ffn_up_exps`/`ffn_down_exps`, created at `src/models/qwen4exp.cpp:255-256`) = **144
`MUL_MAT_ID` nodes per token** — independently corroborated by our own code's
`ggml_sycl_moe_cache_profile_tick(144)` (`ggml-sycl.cpp:5176`) and the `n_mmid % 1440` report at `:5260`.
At our current best config `-ncmoe 26` (`PLAN.md:207-208`), **26 × 3 = 78 of those nodes are CPU-resident**
and would be hooked. Against a 47 tok/s target (21.3 ms/token), 78 round trips at an assumed *x* µs of
non-overlapped host-visible latency cost `78x` µs:

| assumed per-node round-trip latency | cost/token | % of a 21.3 ms budget | % of today's ~36 ms/token (28 tok/s) |
|---|---|---|---|
| 10 µs | 0.78 ms | 3.7% | 2.2% |
| 30 µs | 2.34 ms | 11.0% | 6.5% |
| 60 µs | 4.68 ms | 22.0% | 13.0% |
| 150 µs | 11.7 ms | 55% | 33% |

**We have one in-tree anchor for what to expect.** `moe-cache.cpp:726-746` records a measured kernel-launch
cost of "3-4 µs" for the plan kernel on this hardware. That is a *submit* cost, not a submit-plus-host-wait
round trip, so it is a floor, not an estimate. The number we do not have — and cannot get without running
something — is the cost of `event.wait()` on a Level Zero out-of-order queue from a thread that also has
useful work queued behind it. **This is the single number that decides the project.** Everything else in this
document is tractable.

Three things make the honest estimate better than the table's pessimistic rows suggest:

- The wait is not pure loss: thread 0 is waiting *while `nth - 1` other threads compute miss rows*. The node's
  cost is `max(GPU, CPU_misses) + latency`, not `GPU + CPU`. At a 70% hit rate the CPU still has 3 of 10 rows
  to grind through at RAM bandwidth, which is not fast.
- The `overlap_cpu_rows` knob in the reference API (§2.1) exists precisely to trade a few hit rows back to the
  CPU to cover the latency — a tuning lever that already has a reference implementation to copy.
- Fusing gate+up+SwiGLU (the RFC's own full-design item) collapses 3 nodes/layer to 1, cutting round trips
  from 78 to 26 per token. That is a ~3× reduction in the dominant risk term, and it means the fused path is
  not a nice-to-have optimization on SYCL — **it is the mitigation for the primary risk**, which is a
  different conclusion than the RFC's own "minimal core drops the fusion" framing.

### 4.5 A second, subtler hazard: reentrancy into the SYCL backend

The CPU split runs inside `ggml_backend_sched_compute_splits` (`ggml-backend.cpp:1643-1839`) between SYCL
splits. The SYCL backend's `graph_compute` (`ggml-sycl.cpp:6740-6799`) does **not** unconditionally wait — the
non-profiling branch at `:6794-6796` just calls `ggml_backend_sycl_graph_compute_impl` and returns; only
`ggml_backend_sycl_synchronize` (`:6099-6103`) does `stream->wait()`, and the scheduler calls it at specific
points (`ggml-backend.cpp:1658-1666, 1677-1688, 1702, 1721, 1779-1783`). So when thread 0 enters our hook,
the SYCL default queue may well still have in-flight work. Using a private queue (§4.3) is what keeps that
from mattering — but note that the private queue shares the *device*, so the GPU is genuinely contended, and
the hit-row matvec competes with the resident layers' own work. On a 4×3090 box with 3 idle GPUs that
contention does not exist; on our single B70 it does. **Inference: this is a reason to expect a smaller
relative gain than the RFC's 4-GPU numbers, independent of everything else.** `leloch`'s own reply to the
1080 Ti regression report points the same way: "Ideally you want enough VRAM spare for cache (after placing
ALL dense layers on a GPU - otherwise there is no point)."

Additionally, `moe-cache.cpp` carries unsynchronized process globals written from the dispatch path —
`g_moe_last_events[2]` / `g_moe_last_event_count` (`:488-489`) and `g_prof` (`:618`). Those are benign while
only one thread ever calls in, and become a data race the moment that assumption is relaxed. Keep the
thread-0-only invariant explicit and asserted.

### 4.6 What a spike would have to measure, and what it would not have to build

A validation spike does **not** need the cache, the admission policy, the fill path, the buffer type, or any
scheduler change. It needs:

1. A private in-order `sycl::queue` built from `ctx.stream()`'s context+device, stored in a global, captured
   once at SYCL backend init.
2. A `dispatch`-shaped function that takes `n` F32 row pointers and `n` expert indices, H2D's them, runs
   `quantize_row_q8_1_sycl` + `ggml_sycl_mul_mat_vec_q_id` against **the original `src0->data` directly** (no
   cache, no slot table — `expert_weight_stride = src0->nb[2]`, `ids_dev` = the raw expert ids), D2H's the
   result, waits, and scatters.
3. The hook in `ggml_compute_forward_mul_mat_id` as described in §2.2, with a `GGML_SYCL_MMID_HYBRID_FRACTION`
   env knob deciding what fraction of routed rows go to the GPU.

That is roughly 200-300 lines and requires the `SYCL_Host` cold-tier check from §3.5 (so the kernel can read
`src0->data`). It answers, directly: **what does one round trip cost, and at what row-split fraction does
decode throughput peak?** A fraction of 0 must reproduce today's `-ncmoe` number exactly; a fraction of 1.0
measures the all-GPU-rows extreme. If the curve is flat or inverted, the full design cannot help and the
project stops having spent a few days. If it has a clear interior maximum, the cache is what converts "some
rows are cheap" into "most rows are cheap," and the full build is justified by data.

---

## 5. File-by-file scope, our tree, SYCL not CUDA

Mirroring the RFC's own maintenance-burden breakdown, but for what we would actually touch.

### Minimal core (the spike, plus enough to be real)

| File | Change | Est. LOC | Risk |
|---|---|---|---|
| `ggml/src/ggml-cpu/ggml-cpu.c` | Hook in `ggml_compute_forward_mul_mat_id` (`:1640-1732`): `begin`/`plan`/divert/`dispatch` in the `ith==0` prologue, `collect`/`end` after the `cur_a` loop, full unwind on both failure paths. Plus `MOE_CACHE_MAX_TOPK` and the `row_mask` parameters if the fused path is wanted later. | ~120 | **Medium.** Touches the hottest CPU function in the build. Mitigated by the reference patch being readable and by the divert-before-grouping structure leaving the chunking untouched. |
| `ggml/src/ggml-cpu/traits.h` + `ggml-cpu.cpp` **or** a new `ggml/src/ggml-backend-mmid-hybrid.h` | Either an `extra_buffer_type`/`tensor_traits` registration (§3.3) or a null-checked function-pointer table. | ~60-110 | Low. |
| `ggml/src/ggml-sycl/ggml-sycl.cpp` | Register the provider at backend init; capture and own the private queue; add `SYCL_Host` to the `-ot`-nameable extra bufts. | ~60 | Low. |
| `ggml/src/ggml-sycl/moe-cache.cpp` (or a new `mmid-hybrid.cpp`) | The provider: staging buffers, H2D/quantize/matvec/D2H, the host wait, the scatter. Reuses `ggml_sycl_mul_mat_vec_q_id` unmodified. | ~250 | **High — this is §4.** |
| `common/common.h` + `common/arg.cpp` | A `-ncmoe`-equivalent that targets `SYCL_Host` rather than `CPU`, or just document `-ot`. | ~20 | Low. |
| **Total minimal core** | | **~500-560** | |

### Full design (additional)

| File | Change | Est. LOC | Risk |
|---|---|---|---|
| `ggml/src/ggml-sycl/moe-cache.cpp` | Invert `plan_and_gather` (`:696-1005`) to a host-ids-in / host-slots-out contract; keep the gather kernel (`:498-585`) and the plan kernel's scoring (`:754-974`); delete the buffer type (`:82-163`) and the Phase-3a LRU (`:176-179`, `:311-372`). | ~300 changed, ~300 deleted | Medium. |
| `ggml/src/ggml-cpu/ggml-cpu.c` + `ops.cpp` | Fused `gate` + `up` + GLU detection. Extend `ggml_cpu_try_fuse_ops` (`:3067-3099`) rather than adding a parallel fusion path; the GLU CPU kernels are `ggml_compute_forward_swiglu` (`ops.cpp:3297-3317`) and friends. | ~200 | Medium. Per §4.4 this is the **primary latency mitigation on SYCL**, not an optional extra. |
| `ggml/src/ggml-sycl/mmvq.cpp` | A fused gate+up+SwiGLU expert GEMV. Note `ggml_sycl_mul_mat_vec_q_glu_reorder` already exists (`mmvq.hpp:62-64`) for the *dense* FFN case — a starting point, not a drop-in. | ~150 | Medium. |
| `ggml/src/ggml-sycl/ggml-sycl.cpp` | A device-capability eligibility fence with no CUDA analogue to copy (§1.2). | ~40 | Low code, **unknown policy**. |
| `common/`, `src/llama-context.cpp` | Session lifetime, a real CLI flag, a VRAM fit rule. | ~200 | Low. |
| `tests/` | `test-backend-ops` coverage plus a model-free dispatch selftest. | ~200 | Low. |
| **Total full design** | | **~1,100-1,400 on top** | |

Not in scope and explicitly excluded, matching `PLAN.md:258-262`'s standing descope: graph capture/replay,
hot-set persistence to `~/.cache`, GPU-resident handoff, an L2 pinned tier (`PLAN.md:511-514` already rules
this out — no `cudaHostRegister` equivalent exists in SYCL).

---

## 6. Is the minimal core worth doing first? Yes, but not the RFC's version of it

The RFC's minimal core is "plain per-node dispatch only: no fused gate+up+SwiGLU, no GPU-resident handoff, no
hot-set persistence... roughly half the code." On CUDA that is right, because per-node dispatch is cheap there.

**On SYCL the ordering should differ.** Per §4.4, per-node dispatch latency is the dominant risk, and fusion is
what reduces the node count 3×. So:

- **Step 1 (spike, §4.6, ~200-300 lines, no cache at all): measure the round trip.** This is the go/no-go and
  it is cheap. Do not build anything else until this number exists.
- **Step 2 (minimal core, ~500 lines): per-node dispatch + a trivial static admission policy** (e.g. "the top
  N experts by a warm-up counter, never evicted") rather than the full decayed-LFU. `SharkWipf`'s thread
  comment (2026-09-03) is directly relevant: "if you just count experts every token, you get 90%-98% of the
  final shape within the first 256 tokens... After that, you don't need to count at every generated token
  anymore." A static hot set sidesteps eviction races entirely and is the thing `Volunteer-1`'s
  2026-09-03 comment argues for on cold-start grounds.
- **Step 3: fusion**, if and only if step 1's number says latency is the binding constraint.
- **Step 4: the real admission policy**, reusing our existing plan kernel (`moe-cache.cpp:754-974`).

---

## 7. Interaction with the MTP work landed 2026-09-11

`PLAN.md:197-224` records that `--spec-type draft-mtp` now works for qwen4exp, with the best known
config being `IQ3_XXS + -ncmoe 26 + --spec-draft-n-max 2` at 26-29 tok/s and 92.9-94.9% acceptance, and
records that `--spec-draft-cpu-moe` was tried and was a **net loss** (20.6 vs 26-29 tok/s).

Four concrete interactions, in descending order of how much they worry me:

**(1) The RFC's own thread contains a measured, severe interaction with exactly our configuration.**
`noonghunna`, 2026-08-09, verbatim:

> | DSpark drafter alone (GPU-resident) | 33.6 |
> | DSpark (GPU) + cache | 33.4 — **cache never engages** |
> | DSpark on CPU (`-devd none`) + cache | 46.8 — **composes** |

and, in the same comment:

> With a GPU-resident DSpark, the cache allocates its pools and then never sees traffic: counters read hits=0
> with ~12 lookups total across thousands of decode tokens, and no bypass message ever prints (the verify nodes
> are ~6 tokens, under the max_batch=8 gate — whatever skips them is upstream of that check).

Our MTP draft head is GPU-resident by construction — `PLAN.md:218-224` says pushing it to CPU was measured as
a net loss, so the composing configuration is the one we already rejected on independent grounds. **Open
question, unresolved: whether `noonghunna`'s engagement blocker is specific to llama.cpp's
`server_prepare_shared_draft_de...` path (which the comment names) and therefore absent from our
`--spec-type draft-mtp` graph, or whether it is a general property of GPU-resident drafters.** The comment's
own admission that "whatever skips them is upstream of that check" means even the author did not identify the
mechanism. This should be treated as a real possibility that the hybrid design contributes ~0% under our best
known production config, and it is cheap to test early: the spike in §4.6 can be run with and without
`-md`, and a zero delta under `-md` is the diagnostic.

**(2) The row-count cap excludes larger verify batches, but not ours.** `MOE_CACHE_MAX_TOPK` is 64 and the
gate is `moe_cache_n_rows = ids->ne[0] * ids->ne[1] <= 64`. For qwen4exp, `ids->ne[0] = n_expert_used = 10`,
so the bound is **6 tokens per ubatch**. A `--spec-draft-n-max 2` verify batch is 3 tokens → 30 rows → in
scope. `--spec-draft-n-max 4` is 5 tokens → 50 rows → in scope. `--spec-draft-n-max 6` would be 7 tokens → 70
rows → **silently out of scope**, falling back to the pure CPU path. `leloch`'s "All modes now default to
8-token eligibility" is the same bound stated in tokens. **This is a real, hard, quantified ceiling on how far
speculative depth can be pushed if the hybrid path is adopted**, and it is worth knowing before tuning
`--spec-draft-n-max` upward.

**(3) A verify batch dilutes the hit rate.** At `--spec-draft-n-max 2` the verify step routes the *union* of
3 tokens' expert sets — up to 30 distinct experts per tensor instead of 10. **Inference:** for a fixed slot
pool that lowers the per-node hit rate and raises fill pressure on precisely the step that dominates
wall-clock under MTP. The RFC's decode-only fill policy was measured on non-speculative decode; whether it
holds at ubatch 3 is untested by anyone in the thread.

**(4) The draft block itself goes through the same path.** `src/models/qwen4exp.cpp:607` calls
`build_layer_ffn(cur, il)` inside the NextN block, so the draft head is a full 512-expert MoE layer
(`PLAN.md:219-220`: "3.3 GB resident"). At `-ncmoe 26` it is GPU-resident and therefore *not* hooked. Fine —
but it means a draft step's cost is unaffected by this work, so the achievable end-to-end speedup is bounded
by Amdahl against a draft step that already costs real time. **This is not a correctness risk, but it caps the
upside.**

**No correctness risk is identified.** Speculative decoding is fail-safe by construction
(`PLAN.md:193-195`), and the hybrid path's every failure mode falls back to bit-identical CPU compute. But
note `docs/research/02` §3.5 as cited at `PLAN.md:650-654`: PR #23174 fixed "MTP on SYCL gives garbled output
after a few tokens" as recently as 2026-05-22, so SYCL+MTP has an independent history of silent correctness
bugs. Any benchmark of this combination needs the full Phase 4 correctness battery, not a speed comparison.

---

## 8. Open questions

Ordered by how much they would change the answer.

1. **What does one SYCL dispatch-and-host-wait round trip cost from a CPU worker thread on the B70?** Nothing
   in tree or thread answers this. The only anchor is "3-4 µs" for a bare kernel submit
   (`moe-cache.cpp:726-746`). Resolved by the §4.6 spike.
2. **Does the hybrid path engage at all under `--spec-type draft-mtp` with a GPU-resident draft head?** See
   §7(1). Resolved by running the spike with and without `-md`.
3. **Can ~20-26 GB of expert weights actually allocate through `ggml_backend_sycl_host_buffer_type()` as
   pinned USM on this box without tripping issue #25812?** See §3.5. Resolved by a load-only test — no
   inference needed.
4. **What should the SYCL eligibility fence test, given CUDA's `min_compute_capability` has no analogue?**
   Candidates: `aspect::usm_host_allocations`, max-mem-alloc-size, device global-memory bandwidth. No
   reference exists; this is genuinely our problem to invent.
5. **Does an interior optimum in row-split fraction exist, or is the curve monotone?** The whole design assumes
   a hit row on the GPU is cheaper than the same row on the CPU. On a 608 GB/s B70 (`PLAN.md:78-80`) versus
   whatever this box's host RAM bandwidth is, that is likelier than not — but it is an assumption.
6. **Does prefill regress?** The RFC's design is decode-only and leaves the `ne[2] >= 32` offload path
   untouched, so in principle no. But `mfethe1`'s and `SharkWipf`'s measured prefill losses were on a design
   that also claimed to be decode-only, which suggests something else is going on in the reference
   implementation that this document has not identified. Worth a specific prefill A/B in any spike.
7. **Should the hook be an in-place patch or an `extra_buffer_type`?** §3.3. Decide with the code open.

---

## 9. Recommendation

**Lean go — but only to a bounded spike, not to the design.**

The case for:

- The architectural fit with our tree is unusually good, and better than the brief assumed. The design needs
  **no scheduler change** and **no new buffer type**, because `-ncmoe` already puts these exact nodes on the
  CPU backend at decode (§3.2). The hook window is ~40 lines in a function we have read in full, it does not
  disturb the chunking logic (§2.2), and the CPU backend already has two upstream extension points
  (`ggml_cpu_extra_compute_forward`, `ggml_cpu_try_fuse_ops`) that the reference implementation did not use
  and that would make our version cleaner than theirs (§3.3).
- The single most valuable piece of our existing SYCL work — `ggml_sycl_mul_mat_vec_q_id` — is *exactly* the
  kernel the new design's `dispatch` needs, unmodified (§3.4). The gather and plan kernels, which carry the
  three hard hardware bugs this project already paid for (device-wide-grid gather, same-call eviction race,
  sub-group-size hangs), survive too.
- `ggml_backend_sycl_host_buffer_type()` already provides a pinned, device-dereferenceable, `is_host == true`
  cold tier with a 2 GiB chunking cap built in (§3.5) — the SYCL equivalent of the CUDA fork's zero-copy alias,
  for approximately one line of wiring, and without going anywhere near the forbidden `GGML_SYCL_USM_SYSTEM`.
- The `19.7 vs 35.4` result that `docs/00-background.md` recorded as evidence against this design category is,
  on reading the source, a measurement of **this design** in the `budget ≥ working set` regime — which is not
  our regime (§1.3). That is a meaningful correction in the design's favor.
- The team is demonstrably not SYCL-naive. Reading `moe-cache.cpp:762-795` and `:934-957`, the comments record
  three distinct hardware-only bugs found and fixed. The skills this port needs are the skills this project
  already built.

The case for bounding it hard:

- **The round-trip latency question is unanswered and is decisive.** 78 host-visible GPU waits per token at
  `-ncmoe 26`, against a 21.3 ms budget. At 30 µs it is an 11% tax that the design should easily repay; at 150
  µs it is fatal. Nothing in our tree, the RFC, or the 68-comment thread contains this number for SYCL on
  Level Zero, because nobody has built this on SYCL. CUDA's cheap any-thread stream dispatch is exactly the
  property SYCL does not obviously hand us (§4.1-4.4).
- On a single contended GPU the hit-row matvec competes with the resident layers; the RFC's headline numbers
  came from a box with three spare GPUs (§4.5).
- Four PRs in this line were closed within 80 minutes upstream (§1.4). There is no upstreaming payoff here.
  Whatever we build, we maintain.
- And the honest prior from the thread's own best-controlled experiment is that our measured position is
  mid-table, not at the +57% end (§1.3).

**Concretely: spend a few days on the §4.6 spike — a private queue, a raw `dispatch`-shaped function against
uncached `src0->data`, the `ggml-cpu.c` hook, and an env-var row-split fraction, with `-ot ...=SYCL_Host` for
the cold tier. Roughly 200-300 lines, touching no cache code.** It produces three numbers that nothing else
can: the per-node round-trip cost, the throughput-vs-split-fraction curve, and whether the path engages at all
under `--spec-type draft-mtp`. Commit to the ~1,600-line full design only if the spike shows an interior
optimum that beats today's 26-29 tok/s, and treat gate+up+SwiGLU fusion as part of the *core* rather than a
later refinement, because on SYCL it is the primary mitigation for the primary risk (§4.4, §6).

If the spike says no, we will have spent a few days to close out `PLAN.md`'s "largest un-tried lever" with
data instead of leaving it open, and `-ncmoe 26 + --spec-type draft-mtp` remains the production baseline it
already is.

---

## 10. Spike results (measured)

**Status: §4.6's Step-1 spike is built and measured.** Date 2026-09-11, same session as the rest of this
document, same box (Arc Pro B70 ~30 GB, Ryzen 9 9950X 16c/32t, 123 GB DDR5, single GPU). Nothing is
committed; the code below lives in the working tree only. Every number in this section came out of the
instrumentation described in §10.2, not from a stopwatch on a wall clock.

### 10.1 What was built

About 660 lines, of which 580 are two new files; the three existing files take 92 lines between them:

| File | Change | Lines |
|---|---|---|
| `ggml/src/ggml-mmid-hybrid.h` | **new.** The hook table: `dispatch(src0, n_rows, act_rows, expert_ids, dst_rows)` and `collect()`, plus `GGML_MMID_HYBRID_MAX_ROWS 64`. Same shape as the reference fork's `ggml-backend-moe-cache.h` (§2.1) minus `begin`/`plan`/`end`, which have nothing to do without a cache. | 65 |
| `ggml/src/ggml-backend.cpp` | `struct ggml_mmid_hybrid_api ggml_mmid_hybrid = { nullptr, nullptr };` in ggml-base, so ggml-cpu can call it and ggml-sycl can fill it in without either linking the other. Confirmed in the built objects: `B` in `libggml-base.so`, `U` in `libggml-cpu-*.so` and `libggml-sycl.so`. | +10 |
| `ggml/src/ggml-sycl/mmid-hybrid.{cpp,hpp}` | **new.** The provider: private in-order queue, staging buffers, H2D, `quantize_row_q8_1_sycl<quantize_q8_1>`, `ggml_sycl_mul_mat_vec_q_id` against `src0->data` with `expert_weight_stride = src0->nb[2]` and the raw expert ids, D2H, host wait, scatter. Plus all of §10.2's instrumentation. | 501 |
| `ggml/src/ggml-sycl/ggml-sycl.cpp` | `ggml_sycl_mmid_hybrid_register()` at the end of `ggml_check_sycl()`, i.e. on the thread that ran SYCL init. | +6 |
| `ggml/src/ggml-cpu/ggml-cpu.c` | The hook, exactly where §2.2 and §3.1 said: divert in the `ith == 0` prologue **before** the row-grouping loop, `collect()` after the `cur_a` loop. Plus `ggml_mmid_hybrid_fraction()`, a `static float` cache of `GGML_SYCL_MMID_HYBRID_FRACTION`. | +76 |

Three things in §4.6's list turned out differently than the doc expected, and they are corrections to the
document above, not just notes:

**(a) The private queue construction is right, but `moe-cache.cpp:218` is not the latent bug §4.3 calls it.**
The new queue is `sycl::queue(compute_q.get_context(), compute_q.get_device(), {sycl::property::queue::in_order()})`
built from `dpct::dev_mgr::instance().get_device(0).default_queue()` at init, stored in a file-static, and never
re-resolved per call. §4.3 says `moe-cache.cpp:218`'s omission of `in_order` is an inconsistency "in code we
already ship" that "should be fixed regardless." On reading the uses, it is not: the in-order argument at
`moe-cache.cpp:736-740` and `:999-1004` is about `compute_q` (the backend's `ctx.stream()`, which is already
in-order by construction, `dpct/helper.hpp:727-731`), not about `copy_queue`. `copy_queue` is used at `:342`
for a batch of **independent** host->device miss copies whose events are collected explicitly
(`miss_events.push_back`), at `:385` for an `ext_oneapi_submit_barrier`, and otherwise only for `sycl::free`.
Adding `in_order` there would serialize the one thing that is deliberately concurrent. **`moe-cache.cpp` was
left untouched**, as the brief also required; §4.3's third bullet should be read as "do not copy the omission
into new code," which is what was done.

**(b) §3.5's cold tier is free, but only with mmap off, and it needs no `-ot` change at all.**
§3.5 predicted that `SYCL_Host` placement would need "a one-line change" to
`ggml_backend_sycl_moe_cached_get_extra_bufts` so `-ot` could name it. That is unnecessary. With plain
`-ncmoe 26`, `llama_model_loader`'s override branch (`src/llama-model-loader.cpp:1235`) already calls
`select_weight_buft(..., buft_list_cpu)` whenever the override buft is `ggml_backend_cpu_buffer_type()`, and
`make_cpu_buft_list` (`src/llama-model.cpp:1032`, its own comment: "CPU: ACCEL -> GPU host -> CPU extra -> CPU")
puts `ggml_backend_dev_host_buffer_type(sycl_dev)` **ahead of** the plain CPU buft. Confirmed in a `-v` run:
234 `buffer type overridden to SYCL_Host` lines, i.e. all 78 CPU-resident expert tensors across the three
load passes.

But the *selected* buft is not the *materialized* one. With mmap on (the default, and the command every prior
number in `PLAN.md` used), those tensors are served out of the file mapping and report as `CPU_Mapped`, and
the spike's own check says so out loud:

```
MMID_HYBRID: blk.0.ffn_gate_exps.weight is not device accessible (usm kind 3), declining.
```

`usm kind 3` is `sycl::usm::alloc::unknown`. The fix is `-lm none`; with it the same tensors allocate through
`ggml_backend_sycl_host_malloc` (`ggml-sycl.cpp:1540-1556`) and the check passes. **This answers open question
3: ~21 GiB of expert weights allocate as pinned `sycl::malloc_host` on this box without tripping #25812**
(no `GGML_SYCL_HOST_PINNED_MEM_2G` needed; `mem_limit: 90g` on the container was not reached). It also means
every fraction > 0 measurement below carries `-lm none`, so §10.3 reports a matched `-lm none` baseline
alongside the standard mmap one.

**(c) The provider, not `ggml-cpu.c`, decides device-accessibility.** `ggml_backend_buffer_is_host()` is true
for both a `malloc`'d CPU buffer and a pinned `SYCL_Host` one, and only the former is unreadable by a kernel.
The C hook therefore only screens on the reference fork's own cheap conditions (`src0->op == GGML_OP_NONE`,
`ggml_backend_buffer_is_host`, `usage == WEIGHTS`, F32 in/out, `ids->ne[1]*n_ids <= 64`) and the provider makes
the final call with `sycl::get_pointer_type(src0->data, ctx)`, memoized per weight base pointer. A decline is
a complete decline: nothing is written, no row is diverted, and the stock CPU path runs.

### 10.2 The instrumentation

Two independent measurements, both on the private queue from CPU thread 0, both reported to stderr:

1. **Per-node accounting**, windowed over `GGML_SYCL_MMID_HYBRID_WINDOW` dispatches (default 780 = 10 decode
   tokens at 78 hooked nodes each). The node span is split into `gather+submit` (host time to memcpy the F32
   rows into the staging buffer and enqueue the whole chain), the gap in which thread 0 runs its share of the
   CPU miss rows, `wait` (host time blocked in `event::wait()` inside `collect`), and `scatter`. The four
   segments are contiguous and sum exactly to the span, so the true submit-to-host-visible-completion time is
   `gap + wait`.
2. **A calibration triplet**, run once at the first hooked node and then every 780 dispatches during decode.
   It times the same chain three ways: `bare` (two tiny memcpys and a host wait, no kernel - the floor cost of
   one round trip), `host w` (the real chain against `src0->data` in pinned host memory), and `vram w` (the
   same chain against a device-resident mirror of the routed expert slices). The mirror is 5-13 MB, is never
   used to produce output, and exists only so the round trip can be timed with the weight read taken out of
   it - it is a stopwatch, not a cache. `vram w` is the number the full cached design would pay on a hit.

`GGML_SYCL_MMID_HYBRID_FRACTION` is parsed once via a `static float` in `ggml-cpu.c` and a `static const float`
in the provider. At `0` (or unset) the provider never registers, `ggml_mmid_hybrid.dispatch` stays `NULL`, and
the branch in `ggml_compute_forward_mul_mat_id` is `ith == 0 && NULL != NULL` - i.e. one predictable
never-taken test per node on one thread. It is not byte-for-byte the old function (an `int hybrid_n = 0`, that
test, a `hybrid_skip` test inside the grouping loop, and an `if (hybrid_n > 0)` at the end remain), but nothing
allocates, no queue is created, and no SYCL symbol is touched. §10.3's first row is the measured check.

### 10.3 Throughput vs. row-split fraction

Command as specified (`-ngl 99 -fa 1 -ncmoe 26 -st -n 300 --temp 0 --seed 42`, prompt
`Count from one to fifty.`, UD-IQ3_XXS), one run at a time, no other `llama-cli` or SYCL container active.
The box was not otherwise quiet: an unrelated tagger-training process (~1 core) and a Minecraft server
(~0.75 core) ran throughout, as they did for `PLAN.md`'s own numbers.

| `GGML_SYCL_MMID_HYBRID_FRACTION` | GPU rows per node (of 10) | decode t/s | prompt t/s |
|---|---|---|---|
| unset, mmap (the standard command) | 0 | 19.1, 24.4 | 1.4 (cold page cache), 24.2 |
| 0, `-lm none` | 0 | 24.0 | 25.7 |
| 0.1, `-lm none` | 1 | 17.5 | 20.5 |
| 0.25, `-lm none` | 3 | 12.2 | 23.7 |
| 0.5, `-lm none` | 5 | 8.7 | 22.7 |
| 1.0, `-lm none` | 10 | 5.2 | 20.4 |

**The curve is inverted, monotonically, with no interior maximum.** Every row the GPU takes off the CPU makes
decode slower, and the loss is close to linear in the number of rows diverted.

Two controls on that table. First, the no-op check: fraction 0 reproduces the band `PLAN.md:210-213` records
for this configuration (23.6, 24.5 without `-md`; 19.7-25.3 for stock `-ncmoe`), and `-lm none` costs nothing
measurable relative to mmap. Second, correctness: the generated text is **byte-identical at every fraction,
including 1.0**, where all 10 routed rows of all 78 hooked nodes are computed on the GPU - 189 characters of
greedy output, `md5 625719b63796`, identical to the fraction-0 runs. No argmax flipped, so the
FMA-contraction class of difference this project hit in `hyper_connect.cpp` did not need addressing here.
`test-backend-ops` does **not** cover this path: its `test_mul_mat_id` weights are not tagged
`GGML_BACKEND_BUFFER_USAGE_WEIGHTS`, so the hook's eligibility test excludes them. Output identity is the
correctness evidence, not the op test.

Prompt throughput is unchanged within noise, as §3.2 predicted: the hook cannot fire on a prefill ubatch,
because `ids->ne[1]*n_ids` exceeds `GGML_MMID_HYBRID_MAX_ROWS 64` above 6 tokens, and anything at 32 tokens or
more is offloaded to SYCL before the CPU backend ever sees it.

### 10.4 The round trip, isolated: the number §8's question 1 asked for

Calibration triplet, `GGML_SYCL_MMID_HYBRID_FRACTION=1.0`, 20 rows (the first hooked node routes 2 tokens x 10
experts), 200 iterations at the start of decode and 30 iterations per window in steady state:

| chain | at start of decode, min / mean | in steady-state decode, min / mean |
|---|---|---|
| `bare` - submit + host wait, no kernel | 15.1 / 17.1 us | **6.2-7.0 / 7.3-8.4 us** |
| full chain, weights in pinned host | 474.3 / 485.8 us | 476.6-478.5 / 477.1-481.9 us |
| full chain, weights in VRAM | 78.3 / 80.1 us | 78.6-82.4 / 80.6-83.6 us |

And the same triplet across the sweep, which varies the row count (fraction x 20), start-of-decode minima:

| rows | `bare` | host weights | VRAM weights |
|---|---|---|---|
| 2 | 15.9 | 62.8 | 18.8 |
| 5 | 15.6 | 133.0 | 30.5 |
| 10 | 15.6 | 247.6 | 47.5 |
| 20 | 15.5 | 474.5 | 78.4 |

Both weight paths are cleanly linear in rows:

- **VRAM weights: 12.2 us fixed + 3.31 us per row.**
- **Pinned host weights: 17.0 us fixed + 22.9 us per row.** At ~512 KiB per expert slice that is 21.9 GB/s,
  i.e. practical PCIe 4.0 x16 peak. The kernel is doing exactly what it should; the link is the limit.

**Answer to §8 question 1: one SYCL dispatch-and-host-wait round trip from CPU thread 0 on this stack costs
about 15 us cold and 6-8 us once the machine is in steady-state decode.** That is *below* the most optimistic
row of §4.4's table. At 78 hooked nodes per token it is 0.5-0.6 ms/token, about 2.8% of a 21.3 ms budget for
47 t/s and 1.4% of today's ~42 ms token. The in-tree anchor (`moe-cache.cpp:726-746`, "3-4 us" for a bare
kernel submit) was a good floor: the host-visible wait adds roughly 3-4 us on top of it, not 100.

**§4.5's contention hazard did not materialize.** The doc predicted that sharing one B70 between the private
queue and the backend's own resident-layer work would inflate these numbers relative to the RFC's 4-GPU box.
It does not: the steady-state calibration is taken while 22 GPU-resident layers are executing every token, and
the host-weight chain (477-482 us) and the VRAM chain (80-84 us) are statistically indistinguishable from the
same chains at the start of decode, while `bare` actually gets *faster* (15 us -> 7 us) once the CPU is hot.
This is a genuine and useful negative result: on this hardware, a private in-order queue submitting small
MoE GEMVs alongside the graph's own work does not measurably queue behind it.

### 10.5 Where the loss actually comes from

Per-node accounting, steady-state windows only (first window dropped; means over ~22 windows of 780
dispatches each):

| fraction | GPU rows | node span | gather+submit | thread-0 CPU gap | host wait | scatter |
|---|---|---|---|---|---|---|
| 0.1 | 1 | 310 us | 12.1 | 170 | 127 | 0.4 |
| 0.25 | 3 | 666 us | 9.5 | 127 | 529 | 0.9 |
| 0.5 | 5 | 1062 us | 10.6 | 105 | 944 | 1.5 |
| 1.0 | 10 | 2068 us | 10.2 | 4 | 2051 | 2.9 |

Read the CPU gap column bottom-up and it gives the CPU side of the same node for free. Fitting the three
fractions that leave CPU work (9, 7 and 5 rows -> 170, 127 and 105 us): **~16.3 us per row plus ~24 us of
barrier and quantize overhead, so a full 10-row node costs the whole 32-thread pool about 186 us.** Across 78
hooked nodes that is 14.5 ms of a 41.7 ms token at 24 t/s - 35% of the token, which is the size of the prize
this whole design is chasing.

Now the GPU side. At fraction 1.0 the CPU gap collapses to 4 us, so the node span *is* the round trip:
**2068 us for 10 rows**, against a CPU cost of 186 us for the same 10 rows. An 11x loss, and that is the
entire explanation of §10.3's curve.

The uncomfortable part is that 2068 us is **8.4x worse than the 247.6 us the calibration measures for the
same 10 rows against the same pinned host memory on the same queue at the same moment.** The difference
between the two is not contention (§10.4 rules that out) and not row count. It is that the calibration re-reads
*the same* expert slices 30 times in a row, while the real path reads 10 *newly routed* slices out of a
256-450 MiB tensor and never touches them again. Effective PCIe read rate: **21.9 GB/s hot, ~3.1 GB/s cold**
(78 nodes x ~6.4 MB per node / 161 ms of GPU time per token at 5.2 t/s). Candidate causes, none verified:
IOMMU/device-TLB misses on first touch of each 512-900 KiB slice (~1600 fresh 4 KiB pages per node, ~125k per
token) with no huge-page backing for the `sycl::malloc_host` region; or Level Zero host-USM DMA setup per
distinct region. **This is the single biggest unexplained factor in the spike and it deserves a follow-up**,
because it is also the fill bandwidth any cached version of this design would get: at 3 GB/s a 4 GB cache
takes over a second to fill, and every demand-filled miss is expensive.

### 10.6 Verdict

Answering §4.6's own framing directly, and splitting it, because the single question it poses turns out to
contain two with opposite answers.

**The spike's own curve is inverted, not flat and not humped, and the uncached hybrid path is dead.** Reading
CPU-resident expert weights across PCIe costs 22.9 us/row hot and roughly 200 us/row in the access pattern
decode actually produces, against 16.3 us/row for the CPU doing the same work out of DDR5. There is no
fraction at which that trade wins, and §10.3 measured it at five points to be sure. The RFC's `-ngl 99
-ncmoe 99` benchmark arm, ported literally and without a cache, cannot help here.

**But the number this document said "decides the project" came back favorable, and it is the opposite of what
§4.4 feared.** The dispatch-and-host-wait round trip is 6-8 us in steady state, not 30 or 150; the private
in-order queue is not contended by the backend's own work; and with the weights already on the device the
whole chain is 12.2 us + 3.31 us/row, i.e. **~45 us for a 10-row node against the CPU's 186 us, a 4.1x
per-node win.** Every structural worry in §4.1-4.5 - cross-thread submit, the dpct TLS trap, queue choice,
reentrancy, device contention - either did not bite or was cheap to avoid. §4.4's conclusion that
gate+up+SwiGLU fusion is "the primary mitigation for the primary risk" on SYCL is **not supported**: at 8 us
per round trip, cutting 78 round trips to 26 saves ~0.4 ms/token, which is noise. Fusion is back to being an
ordinary optimization.

So the honest reading is: **the cache is not one of several things that makes this design work - it is the
only thing.** §4.6 anticipated exactly this ("the cache is what converts 'some rows are cheap' into 'most rows
are cheap'"), and the spike has now put numbers on both halves.

What those numbers project for Step 2. This is a projection, so the model is stated in full: per node, at hit
rate `h`, the hybrid structure costs `max(GPU(10h), CPU(10(1-h))) + 8` us, where `GPU(k) = 12.2 + 3.31k`
(§10.4's VRAM fit), `CPU(k) = 24 + 16.3k` (§10.5's fit, and `4` us at `k = 0` as measured), and the `+8` is
§10.4's steady-state round trip. Today's node is `CPU(10) = 187` us and today's token is 41.7 ms at 24.0 t/s:

| hit rate | projected node cost | saving/token over 78 nodes | projected decode t/s |
|---|---|---|---|
| 0.5 | 114 us | 5.7 ms | ~27.8 |
| 0.65 | 89 us | 7.6 ms | ~29.3 |
| 0.8 | 65 us | 9.5 ms | ~31.0 |
| 0.9 | 50 us | 10.6 ms | ~32.2 |
| 1.0 | 53 us | 10.4 ms | ~31.9 |

(The curve turns over at `h = 0.89`, where the GPU's own work stops being hidden behind the CPU's miss rows.
Above that, more hits buy nothing.) Against the 47 t/s target these are **+16% to +34%, landing at 28-32 t/s**
- real, matching the low-to-middle band of the RFC's own +10%..+57%, and still short of 47. Two hard
constraints bound `h` from above, and both are already measured facts about this box rather than open
questions:

1. **VRAM.** At `-ngl 99 -ncmoe 26` the SYCL device already holds 24,991 MiB of weights plus ~2 GB of compute
   buffers on a ~30 GB card. The expert working set of the 78 hooked tensors is 26 x (256 + 256 + 450) MiB =
   ~24.4 GiB. A cache of at most ~4 GB is under 17% of the working set - `mfethe1`'s `budget < working set`
   side (§1.3), which is the side the cache is supposed to win on, but far enough below it that `h` in the
   0.5-0.65 range is the realistic expectation, not 0.8.
2. **Fill cost.** §10.5's ~3 GB/s cold host-to-device rate is what any demand fill will run at. A cache that
   converges to a static hot set pays this once; one that keeps evicting pays it forever. `SharkWipf`'s
   thread comment (§6) about the hot set being 90-98% settled within 256 tokens is now load-bearing, not a
   nice-to-have.

**Recommendation: conditional go on Step 2, at reduced ambition.** The dispatch mechanism is proven and costs
almost nothing, which is what the spike existed to find out. But the projected ceiling is ~28-32 t/s against a
47 t/s target, on top of a design this project has *already* built once in a different shape and measured at
23.1 t/s (`PLAN.md:118`). Step 2 is worth doing only with a hard prior commitment to kill it if it does not
clear ~28 t/s, and it should reuse the existing slot pool, gather kernel and plan kernel (§3.4) rather than
build new ones. Before writing any of it, two cheap things are worth doing first because they could change the
answer outright:

- **Root-cause the 8.4x cold/hot PCIe gap (§10.5).** If it is IOMMU page-walk cost and huge pages fix it, the
  *uncached* path goes from 200 us/row to 23 us/row, which is still worse than the CPU's 16.3 but close enough
  that the whole picture shifts - and it would also make fills 8x cheaper, which lifts the achievable `h`.
- **Re-run the sweep under `--spec-type draft-mtp`** to close §8 question 2. This spike was run without `-md`;
  §7(1)'s "cache never engages with a GPU-resident drafter" report is still untested here, and the production
  config is the MTP one.

Neither of those needs new code. The spike's own code should stay in the tree, unreverted and behind its
`FRACTION=0` default, because §10.4's calibration triplet is the only instrument this project has for asking
"what does a GPU round trip cost right now," and Step 2 will need it on every run.
