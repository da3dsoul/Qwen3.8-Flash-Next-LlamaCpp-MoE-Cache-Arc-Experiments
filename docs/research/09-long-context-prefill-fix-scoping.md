# 09 — Long-context prefill / TTFT: scoping the fix

**Status: research only. No code was written, no benchmark was run. One `git apply --check` (no-op, read-only)
and four `gh` queries were the only things executed beyond file reads; the GPU was verified idle and never
used.**
Date: 2026-09-11. Triggered by `logs/long-context-120k-benchmark-report.md`: a real 116,277-token prompt ran
end-to-end at **68.21 tok/s prefill (28 min 25 s TTFT)** vs. **5.34 tok/s decode**, making prefill — not decode
— the metric that governs this deployment at the real 300K–600K target.

Sources actually read for this document (not summarized secondhand):

| Source | How obtained |
|---|---|
| `PLAN.md:1-133` (the CRITICAL block and both Update blocks) | direct read |
| `docs/research/07-qsa-sparse-attention-scoping.md` (all 594 lines) | direct read |
| `docs/research/02-model-support-status.md` §§ exec-summary, 2.3, 2.4, 3.2, 3.3 | direct read |
| `logs/long-context-120k-benchmark-report.md` §§1.2–2.3, 4 | direct read |
| `logs/phase3-vram-check.log:49-50, 142-182` (model shape, `n_ff_exp`) | direct read |
| Our tree's **Vulkan** backend: `ggml-vulkan.cpp` (pattern/fusion/dispatch/supports_op sites), `vulkan-shaders/topk_radix_select.comp` (all 144 lines), `vulkan-shaders/count_experts.comp` | direct reads, line ranges cited below |
| Our tree's **SYCL** backend: `ggml-sycl.cpp` (argsort, top_k, supports_op, op-profiler), `fattn.cpp` | direct reads |
| Our tree's model graph: `src/models/qwen4exp.cpp:721-946` | direct read |
| **Upstream PR #28670** body, file list, labels, state, and full diff | `gh pr view 28670 --repo ggml-org/llama.cpp --json ...` and `gh pr diff 28670` |
| Upstream PR #28501 state | `gh pr view 28501 --repo ggml-org/llama.cpp --json state,mergedAt` |
| Patch applicability against our tree | `git apply --check --include=<path> /tmp/pr28670.diff` (read-only, applies nothing) |

Everything cited as `file:LINE-LINE` was read directly. Statements that connect facts established in separate
places, rather than quoting one place, are flagged **inference**. Arithmetic that extrapolates a measured or
structural fact is flagged **estimate** and shows its assumptions.

All repo paths below are relative to
`/media/da3dsoul/Golias/AIProjects/Qwen3.8-Flash-Next-LlamaCpp-MoE-Cache-Arc-Experiments/src/llama.cpp/`
unless stated otherwise. Tree state at time of writing: `HEAD = 6d9c82ea2`, working tree dirty (this project's
own SYCL MoE-cache / MTP / top-k work is uncommitted).

---

## 0. Verdict in one paragraph

**The framing in the task — "Option A: write a radix-select top_k for SYCL" vs. "Option B: switch to
Vulkan" — is out of date by two days.** Upstream PR **#28670, "sycl: rfc: Use radix select for top_k"**
(opened 2026-09-09, **OPEN**, labelled **`merge ready`**, last updated 2026-09-11T14:20Z) already contains
that kernel: 531 new lines in a self-contained new file `ggml/src/ggml-sycl/topk-radix.cpp`, written
explicitly because *"Running qwen3.8-flash-next on SYCL causes CPU offloads for the TOP_K operation"*, and
benchmarked by its author on **Arc Pro B60 (BMG-G21) — Battlemage, our card's family** — on this exact model.
So Option A's expensive half is already written by someone else. Four of its five files apply **cleanly** to
our tree; only `ggml-sycl.cpp` conflicts, in two hunks, both trivially, because this project's own local top_k
work already lifted the `k <= 32` guard it needs lifted. **Option A collapses from "~400-700 lines of novel
kernel work" (doc 07 §4.2) to "vendor one upstream file and hand-merge ten lines."** Meanwhile the top-k is,
by a memory-traffic floor, the single largest *backend-fixable* term in prefill — and it grows as
`O(N² log²N)`, so it goes from ≥437 s at 116K to **≥15,899 s (4.4 h) at 600K, at peak device bandwidth**, i.e.
it alone makes the real target unreachable. Vulkan (Option B) fixes the same term, and fixes it slightly
better (it has a model-specific fused kernel, `pipeline_topk_radix_qsa`, verified below), but it does **not**
have sparse flash-attention either (verified: zero occurrences of `n_kv_max` anywhere under
`ggml/src/ggml-vulkan/`), it still carries the open 512-expert `mul_mat_id` regression (#28501, still open),
and switching to it discards every line of this project's SYCL MoE-cache, MTP and hybrid-`MUL_MAT_ID` work —
which matters because the *other* large prefill term, CPU-resident MoE, is exactly what that work targets and
neither backend fixes. **Recommendation: Option A, specifically "land #28670 locally, then re-measure." Not a
backend switch, not a dual-backend split. Sparse FA (doc 07's subject) stays deferred — it is the third-largest
term, not the first.** The one thing worth spending GPU time on for Option B is a *measurement*, not a
migration: the Vulkan image is already built and pinned to the same commit
(`docker/Dockerfile.vulkan`, image `qwen4exp-moe-cache:vulkan-pinned`, 3.01 GB, present locally), so a
side-by-side prefill at moderate depth costs an hour, not a project.

---

## 1. Reframing: where the 1,705 s actually goes

Doc 07 and `PLAN.md`'s latest update both treat "QSA sparse attention" as one thing. It is two things with
very different cost curves, and at 116K they are three orders of magnitude apart. This section separates them
and adds the third term nobody has costed.

### 1.1 The measured anchor

`logs/long-context-120k-benchmark-report.md:85-97` and `:299`:

| | |
|---|---|
| prompt_tokens | 116,277 |
| prefill time | **1,704.79 s** (68.21 tok/s, 14.66 ms/token) |
| config | `-ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -ncmoe 24 -c 163840 -fit off`, **`-ub 512`** (llama-server default) |

Model shape, from the load banner (`logs/phase3-vram-check.log:144-182`, `:49-50`):
`n_layer = 48`, `n_embd = 2560`, `n_head = 24`, `n_head_kv = 2`, `n_embd_head_k/v = 256`,
`n_expert = 512`, `n_expert_used = 10`, `expert_feed_forward_length = 640`,
`expert_shared_feed_forward_length = 640`. Twelve of the 48 layers are full-attention/QSA
(`docs/research/07` §1.2: `compress_ratios` is `4` at indices 3, 7, …, 47).

### 1.2 The three candidate terms

**Term 1 — `ggml_top_k(2051)` on the QSA indexer.** `src/models/qwen4exp.cpp:846-863` gathers a per-token
score row of width `n_kv`, adds the f16 causal mask, and calls `ggml_top_k` on it, once per QSA layer per
ubatch. In our tree that lands in `ggml_sycl_op_top_k` (`ggml/src/ggml-sycl/ggml-sycl.cpp:3111-3160`), which
for `k > 32` (`SYCL_TOP_K_MAX_SCAN_MERGE_K`, `:2620`) does a **full argsort of the entire row** and keeps the
leading `k`. `argsort_f32_i32_sycl` (`:2389-2440`) picks between an in-SLM bitonic (one kernel) and, when
`next_power_of_2(ncols) * 4 > smpbo`, a **global-memory multi-pass bitonic** (`:2397-2430`) issuing
`log2(ncols_pad) · (log2(ncols_pad)+1) / 2` separate kernel launches over `nrows × ncols_pad` work-items each
(`argsort_f32_i32_global_pass`, `:2314-2360`). With the usual 64 KiB Battlemage `smpbo` the global path
engages from `n_kv > 16384`.

**Term 2 — dense flash attention over the full KV.** The QSA top-k restriction is applied as `-INF` mask
entries (`src/models/qwen4exp.cpp:911-937`) and then *executed densely*, because line 946 passes `0` instead
of the sparse hint and because no SYCL FA path reads `op_params[4]` anyway (doc 07 §3.1, §2.5). So prefill
attention pays the full `O(N²)` cost with zero benefit from the indexer's selection.

**Term 3 — CPU-resident MoE experts.** `-ncmoe 24` puts 24 of 48 layers' expert weights on the host. With
`-ub 512` and 10-of-512 routing, **inference**: a 512-token ubatch activates ~5,120 expert slots per layer,
i.e. essentially all 512 experts, so every CPU layer must sweep its whole expert bank once per ubatch.

### 1.3 Cost model and numbers

All three are **estimates**, computed from the structural facts above; the assumptions are stated so they can
be checked or corrected by a measurement (§8).

*Term 1 (bitonic argsort traffic).* Per global pass, each of `nrows × ncols_pad` work-items either exits early
(`ixj <= col`, `:2333`) or performs 2 coalesced `idx` reads, 2 scattered `x` reads and up to 2 `idx` writes
(`:2341-2352`) — ~16 B per element averaged over the early-exit half. Summed over every ubatch of a cold
prefill (`n_kv` grows 512 → N, `ncols_pad = next_pow2(n_kv)`, `nrows = 512`, ×12 QSA layers):

*Term 2 (dense FA FLOPs).* `2 · 2 · n_q · n_kv · d_head · n_head` per layer per ubatch, summed causally,
×12 layers; divided by an assumed 40 TFLOPS effective XMX f16 throughput.

*Term 3 (MoE FLOPs).* `3 · n_embd · n_ff_exp · 2 · (n_expert_used + 1)` per token per layer × 48 layers; half
of it on CPU at `-ncmoe 24`; divided by an assumed 200–400 GFLOPS effective for IQ3_XXS GEMM on the box's
Ryzen 9 9950X (16 cores / 32 threads).

| | 116,277 tok (measured total: **1,705 s**) | 300,000 tok | 600,000 tok |
|---|---:|---:|---:|
| **Term 1** — bitonic top-k, traffic | 265.6 TB | 2,173 TB | 9,666 TB |
| **Term 1** — time **floor** @ 608 GB/s peak | **≥ 437 s** | **≥ 3,575 s** | **≥ 15,899 s (4.4 h)** |
| **Term 1** — kernel launches | 334,512 | 1,079,952 | 2,451,552 |
| **Term 1** — same, with a radix select (≈6 sweeps) | ~3 s | ~21 s | ~85 s |
| **Term 2** — dense FA | 2,018 TFLOP → ~50 s | 13,297 TFLOP → ~332 s | 53,141 TFLOP → ~1,329 s |
| **Term 2** — same, if sparse FA existed (width 2051) | ~1 s | ~2 s | ~5 s |
| **Term 3** — CPU MoE half | 302 TFLOP → 754–1,509 s | 779 TFLOP → 1,946–3,893 s | 1,557 TFLOP → 3,893–7,786 s |

Three readings of that table, in order of importance:

1. **The 116K row is self-consistent with the measurement.** Term 1's floor (437 s) plus Term 3's midpoint
   (~900 s) plus Term 2 (~50 s) lands near the measured 1,705 s without needing a fourth term. That does not
   *prove* the split — §8 says how to settle it — but it means the model is not wildly wrong.
2. **Term 1 is the term that kills the real target.** It grows as `O(N² log²N)` — worse than everything else
   — and at 600K its *floor at peak device bandwidth* is 4.4 hours. No amount of `-ncmoe`, `-ub` or MoE-cache
   tuning touches it. It is also the one term that a backend fix reduces by ~100×, not ~2×.
3. **Sparse attention — the subject of doc 07 and of `PLAN.md`'s latest update — is Term 2, the *smallest*
   of the three.** At 116K it is worth ~50 s of 1,705 s. Even at 600K, after Term 1 is fixed, it ranks behind
   CPU MoE. `PLAN.md:121-132` calls QSA sparse attention "the clear highest-leverage next step"; that is
   **half right and the wrong half is load-bearing**. The high-leverage piece is the indexer's `top_k`, which
   doc 07 §4.3 already identified correctly ("on SYCL, long-context QSA is bottlenecked by the top-k, not by
   dense attention") and which is paid *unconditionally, today, whether or not the sparse flag is ever
   enabled*. See §9 for the correction to fold back.

**Caveat, stated rather than buried.** Term 1's "floor at peak bandwidth" is a floor in one direction and an
overstatement in another: real bitonic passes will not hit 608 GB/s (the `x[idx[…]]` read at `:2352` is a
scattered gather), which pushes the number up; but the early-exit half of the work-items cost a dispatch slot
rather than 16 B of traffic, which pushes it down. The 116K row's consistency with the measured total suggests
the two roughly cancel. The *ratio* between Term 1 and its radix replacement (~100×) is far more robust than
either absolute number, because both are the same traffic model with different pass counts.

---

## 2. Option A, re-scoped: PR #28670 already wrote the kernel

### 2.1 What the PR is — verified, not relayed

`gh pr view 28670 --repo ggml-org/llama.cpp --json title,state,createdAt,updatedAt,mergedAt,isDraft,author,files,additions,deletions,labels,body`:

| field | value |
|---|---|
| title | `sycl: rfc: Use radix select for top_k` |
| author | `cwriter` |
| state | **OPEN**, not a draft, `mergedAt: null` |
| created / updated | 2026-09-09T20:25:13Z / **2026-09-11T14:20:37Z** (today) |
| labels | `testing`, `ggml`, **`merge ready`** ("A maintainer can use this label to indicate that they consider the changes final and ready to merge"), `SYCL` |
| diff | **+587 / −3** across 5 files |

Files:

| file | +/− |
|---|---|
| `ggml/src/ggml-sycl/topk-radix.cpp` | **+531 / −0** (new) |
| `ggml/src/ggml-sycl/topk-radix.hpp` | +24 / −0 (new) |
| `ggml/src/ggml-sycl/ggml-sycl.cpp` | +7 / −3 |
| `ggml/src/ggml-sycl/backend.hpp` | +1 / −0 |
| `tests/test-backend-ops.cpp` | +24 / −0 |

The PR body names our model and our problem in its first line:

> Running qwen3.8-flash-next on SYCL causes CPU offloads for the TOP_K operation due to a guard condition to
> k <= 32. However, qwen3.8-flash-next requires a TOP_K of K == 2048.

and its hardware disclosure is exactly our family:

> I tested only on BMG G21 (Arc Pro B60); testing on Alchemist would be very much appreciated.

(Our card is BMG-G31 / Arc Pro B70 — same Xe2 Battlemage generation, wider. Its self-described tuning constant
`SYCL_TOP_K_RADIX_GROUPS_PER_NSM` is annotated *"tuned on Xe2 and has not been measured on Xe-HPG"* — i.e. the
tuning was done on our architecture, and the *untested* case is the older one, not ours.)

Its own measured numbers, from the body. The `ne=[131072,1]` row is literally our decode-time indexer shape at
131K context:

| shape | k | before | after | speedup |
|---|---|---:|---:|---:|
| `ne=[131072,1]` (**qwen4exp tg**) | 2048 | 208.36 µs | **41.82 µs** | 4.98× |
| `ne=[200000,1]` | 400 | 358.61 µs | 46.10 µs | 7.78× |
| `ne=[151936,1]` (sampler) | 40 | 281.68 µs | 42.33 µs | 6.65× |
| `ne=[65000,16]` | 32 | 11,016.34 µs | 92.79 µs | **118.7×** |
| `ne=[1024,8192]` | 32 | 49,716.84 µs | 1,158.91 µs | 42.9× |

and end-to-end decode on the model, plus a perplexity check:

| depth | master | +radix | improvement |
|---|---:|---:|---:|
| 0 | 18.77 | 19.80 | +5.5 % |
| 8,192 | 15.73 | 17.52 | +11.4 % |
| 32,768 | 11.35 | 13.09 | +15.3 % |
| 131,072 | 5.29 | 5.86 | +10.8 % |

`ppl -c 512 --chunks 200`: 4.2954 → 4.2980 (+0.0026). `-c 81920 --chunks 3`: 3.8119 → 3.8249 (+0.0130).

**Two caveats on those numbers, which matter for us.**

1. **The "before" column is not our baseline.** Upstream master's `ggml_sycl_op_top_k` asserts `k <= 32` and
   `supports_op` refuses `k > 32`, so the PR's "before" is the existing *scan-merge* kernel with the guard
   relaxed — not this project's local *argsort* fallback (`ggml-sycl.cpp:3137-3160`). The two degrade
   differently. The one row that is structurally closest to a many-row, wide-`ncols` case (`ne=[65000,16]`,
   118.7×) happens to land right on §1.3's modelled ~100–135× ratio for our prefill shape, which is a useful
   cross-check but not the same measurement. **Inference:** our speedup could be larger or smaller than 4.98×;
   the honest claim is "same order, not the same number."
2. **Every real-model number in the PR is decode (`tg`).** There is no prefill number in it at all. Our
   problem is prefill. §1.3 argues the prefill win is much larger than the decode win, because prefill runs
   `nrows = 512` rows through the same sort — but that is our arithmetic, not the PR's measurement.

### 2.2 What the algorithm actually is, and why it suits prefill

`topk-radix.cpp` was read directly from the diff. It is the same algorithm as Vulkan's
`vulkan-shaders/topk_radix_select.comp` (all 144 lines read): map float→order-preserving uint
(`top_k_radix_key`, inverting the sign bit or the whole word), then four 8-bit radix passes, each histogramming
only the candidates matching the prefix fixed so far, then a two-phase emit (`> threshold`, then ties up to
`k`). SLM cost is `256 buckets × 8 private copies + 5` words ≈ 8.2 KiB — **fixed, independent of `ncols` and of
`k`**, which is precisely the property the current scan-merge kernel lacks (its own comment at
`ggml-sycl.cpp:2638` explains the SLM blow-up, and doc 02 §2.3 computes ~2.1 MB at `k = 2051`).

Two structural properties matter for our workload specifically:

- **No global scratch in the prefill regime.** `top_k_radix_split_groups` returns `1` when
  `nrows >= 2 · nsm`, so with `nrows = 512` (our `-ub`) every row gets one work-group, all state lives in SLM,
  and **the kernel allocates nothing in VRAM**. The multi-group split path (with its `256+6`-word global state
  buffer) only engages for `nrows = 1` — decode. Compare the current path, which allocates
  `nrows × ncols × 4` for `sorted_alloc` (`ggml-sycl.cpp:3138`) *plus* `nrows × ncols_pad × 4` for
  `idx_padded_alloc` (`:2398`).
- **It is `O(ncols)` per pass with ≤ 4 passes plus an emit**, against the bitonic's
  `log2(ncols_pad)·(log2+1)/2` = 153–190 passes at our depths.

VRAM consequence, **estimate**, `-ncmoe 24` (the benchmark had 2,836 MiB spare at 116K,
`PLAN.md:106-107`):

| depth | `-ub` | current SYCL argsort scratch | Vulkan QSA fusion scratch | **#28670 radix** |
|---:|---:|---:|---:|---:|
| 262,144 | 512 | 1.00 GiB | 0.50 GiB | **0** |
| 614,400 | 512 | 3.17 GiB | 1.17 GiB | **0** |
| 614,400 | 2,048 | 12.69 GiB | 4.69 GiB | **0** |

That last row is the one to notice. **Inference:** raising `-ub` is the natural lever against Term 3 (it
amortises the CPU expert-bank sweep over 4× more tokens) and leaves Term 1's traffic unchanged (the
`Σ nrows·ncols` sum is `≈ N²/2` regardless of `-ub`) — but today it is blocked by scratch that scales with
`-ub`. **The radix kernel unblocks the `-ub` lever for free.** This is a second, independent reason to land it,
and it is not something doc 07 or doc 02 anticipated.

### 2.3 What landing it costs us — checked, not guessed

`git apply --check --include=<path> /tmp/pr28670.diff`, run per file against our tree:

| file | applies cleanly? |
|---|---|
| `ggml/src/ggml-sycl/topk-radix.cpp` | ✅ (new file) |
| `ggml/src/ggml-sycl/topk-radix.hpp` | ✅ (new file) |
| `ggml/src/ggml-sycl/backend.hpp` | ✅ |
| `tests/test-backend-ops.cpp` | ✅ |
| `ggml/src/ggml-sycl/ggml-sycl.cpp` | ❌ `patch failed at 3098` |

The one conflict is expected and trivial, and it conflicts because **we already did half the work.** The PR's
two hunks are:

```
-    GGML_ASSERT(k > 0 && k <= 32);
+    GGML_ASSERT(k > 0);
     GGML_ASSERT(k <= ncols);
-    top_k_f32_sycl(ctx, src0_dd, dst_dd, ncols, nrows, k, main_stream);
+    if (k <= SYCL_TOP_K_SCAN_MERGE_MAX_K) {
+        top_k_f32_sycl(...);
+    } else {
+        ggml_sycl_top_k_radix(ctx, src0_dd, dst_dd, ncols, nrows, k, main_stream);
+    }
```
and, in `supports_op`:
```
-                k > 0 && k <= 32;
+                k > 0 && k <= src0->ne[0];
```

Our tree's `supports_op` **already reads** `k > 0 && k <= src0->ne[0]` (`ggml-sycl.cpp:7315`) — that hunk is
already applied. And our `ggml_sycl_op_top_k` already has the `if (k > SYCL_TOP_K_MAX_SCAN_MERGE_K)` branch
(`:3137`); the change is to replace its argsort body with the `ggml_sycl_top_k_radix` call, plus reconcile the
constant's name (`SYCL_TOP_K_MAX_SCAN_MERGE_K` here vs. `SYCL_TOP_K_SCAN_MERGE_MAX_K` there). No build-system
change is needed: `ggml/src/ggml-sycl/CMakeLists.txt:26-27` globs `*.hpp` / `*.cpp`, so the new file compiles
automatically.

**Effort estimate: ~600 lines vendored verbatim, ~10 lines hand-merged, 0 lines of new algorithm.**

| step | effort | risk |
|---|---|---|
| Vendor `topk-radix.{cpp,hpp}` + `backend.hpp` + test hunks | ~15 min | none |
| Hand-merge the two `ggml-sycl.cpp` hunks onto our local top_k branch | ~30 min | low |
| Rebuild `qwen4exp-moe-cache:sycl-pinned` | ~40 min wall | none |
| `test-backend-ops -b SYCL0 -o TOP_K` (the PR adds 24 lines of new cases) | ~10 min GPU | **low-medium** — untested on BMG-G31; the PR's own device-tuning constants were fit on BMG-G21 |
| Re-run the 116K benchmark for a before/after TTFT | ~1 h GPU | none |
| **Total** | **~half a day, of which ~1.5 h is GPU** | **low** |

Residual risks worth naming: (a) the PR is **not merged**, so upstream may still change it — but `merge ready`
plus an active 2026-09-11 update means the design is settled, and vendoring a not-yet-merged patch is something
this project already does routinely (`docker/Dockerfile.sycl` pins a commit for exactly this reason);
(b) the author flags *"may have issues with NaN values"* — our input is a ReLU-rectified score sum plus an f16
causal mask (`src/models/qwen4exp.cpp:827-858`), so the only non-finite values are `-INF` mask entries, which
the key mapping orders correctly; a NaN would require a NaN in the indexer projections, which would already be
breaking the model. **Inference**, not verified by running; `test-backend-ops` is where to check it.

### 2.4 The other half of Option A — sparse FA wiring — stays deferred

Doc 07 §3.1 established that **no** SYCL FA path reads `op_params[4]`, on either the oneDNN/MKL prefill path or
the vec/tile decode path, and §4.2 scoped the work at ~400-700 lines across 3-4 files. Nothing read for this
document changes that, and two things sharpen it:

1. **Re-verified: the prefill path has nowhere to put a gather.** `ggml/src/ggml-sycl/fattn.cpp:106-180` selects
   `BEST_FATTN_KERNEL_ONEDNN` or `BEST_FATTN_KERNEL_MKL` for our prefill shapes (`gqa_ratio = 12`,
   `Q->ne[0] = 256`, `Q->ne[1] = 512 ≥ 32`, `K->ne[1] ≥ 1024`; the MKL comment at `:141-146` notes it converts
   non-F16 K/V to F16, so our `q4_0` cache still takes it). Both are library GEMM calls. Injecting a sparse
   index gather means either physically compacting K/V rows into a scratch buffer first — which at width 2051
   × 12 layers × 512 queries is its own bandwidth problem — or abandoning XMX for QSA layers.
2. **Term 2 is the smallest of the three costs** (§1.3). At 600K, post-radix, it is ~1,329 s against CPU MoE's
   3,893-7,786 s. Optimising it first would again be optimising the smaller term — the same mistake doc 07
   warned about, one rung up the ladder.

**Verdict unchanged from doc 07 §5: leave `src/models/qwen4exp.cpp:943-946` as it is.** Revisit only after
§8's profile shows FA is a double-digit share of post-radix prefill.

---

## 3. Option B, verified: what Vulkan actually has

### 3.1 `pipeline_topk_radix_qsa` is real, and it is more than doc 02 claimed

Doc 02 §2.3 reported this secondhand. Verified directly in our tree:

- **Declaration and comment**, `ggml/src/ggml-vulkan/ggml-vulkan.cpp:1114`:
  ```cpp
  vk_pipeline pipeline_topk_radix_qsa; // qwen4 QSA indexer fusion (f16 mask)
  ```
- **It is a specialisation of the generic radix shader, not a separate shader** — `:6140-6141` creates both
  `topk_radix_f32` and `topk_radix_qsa` from the same `topk_radix_select_f32_data`, differing only in
  specialisation constant 1 (`{BLOCK_SIZE, 0}` vs. `{BLOCK_SIZE, 1}`), which is
  `layout(constant_id = 1) const int QSA` in `vulkan-shaders/topk_radix_select.comp:9`.
- **It is a seven-op graph fusion, not just a top-k.** `ggml-vulkan.cpp:668-681` defines
  `topk_qsa_pattern { GET_ROWS, PERMUTE, CONT, CPY, RESHAPE, ADD, TOP_K }` with an explicit edge list, under
  the comment *"qwen4 QSA indexer: gather per-block scores to cells + add f16 mask (cast+reshape) + top-k,
  fused into one radix-select. The cast/reshape are elided; the raw f16 mask is read in-shader."*
  `ggml_vk_can_fuse_topk_qsa` (`:17920-17985`) validates use-counts, types and the exact tensor layout;
  `ggml_vk_topk_qsa` (`:14680-14727`) dispatches it.
- **The pattern matches our graph exactly.** `src/models/qwen4exp.cpp:846-863` emits, in order:
  `ggml_get_rows(cont(permute(score)), cell_blk)` → `ggml_cont(ggml_permute(expanded))` →
  `ggml_cast(kq_mask, F32)` → `ggml_reshape_3d` → `ggml_add` → `ggml_top_k(expanded, width)`. The `ggml_cast`
  branch is gated on `blk_bias` (`:748-752`), which is true for us (`causal_attn = 1`, no ALiBi, mask shape
  matches). So the f16-mask variant — the one the fusion is written for — is the variant our graph produces.

So the claim in doc 02 line 228 is not only true, it undersells it: upstream Vulkan has a kernel written for
**this model's indexer specifically**, which also elides the f16→f32 mask cast and the intermediate
`permute`/`cont` copies that SYCL will still materialise even with #28670 landed.

**But**: the fusion memoises the gather into `prealloc_x`, sized `n_kv × nrows × 4` (`:14705-14709`) — the same
`n_kv × n_ubatch × 4` term the Arc Pro B65 reporter blamed for the 7× cliff (doc 02 §3.2). #28670's SYCL
kernel needs **zero** scratch at `nrows = 512` (§2.2). On the VRAM axis specifically, SYCL-plus-#28670 is the
*better* of the two.

### 3.2 Vulkan does **not** have sparse flash attention

Whole-tree grep: `grep -rn 'n_kv_max' ggml/src/ggml-vulkan/` → **zero matches.** Vulkan's FA never reads
`op_params[4]`, exactly like SYCL and CPU (doc 07 §3.1 said the same from the four-file name grep; this is the
directory-scoped confirmation). So Term 2 is unfixed on Vulkan too. **Migrating to Vulkan buys Term 1 and
nothing else on the attention side.**

### 3.3 Vulkan op coverage for `qwen4exp` — no new blockers found, one known regression

Checked the ops doc 02 §2.2 flagged as bounded on Vulkan, against our tree and our model's real shapes:

| op | Vulkan gate | our value | ok? |
|---|---|---|---|
| `GGML_OP_GATED_DELTA_NET` | `S_v ∈ {16,32,64,128}` (`ggml-vulkan.cpp:12262-12274`) | `ssm.state_size = 128` | ✅ |
| `GGML_OP_CUMSUM` | F32, two pipelines split at `ne[0] ≤ 512` (`:12159-12167`) | GDN chunk 64 | ✅ |
| `GGML_OP_SOLVE_TRI` | F32, pipeline built per `(ne0, ne1)` state (`:12168-12180`) | 64×64 | ✅ |
| `GGML_OP_TOP_K` | `"large k falls back to radix-select"` (`:19737-19749`) | k = 2051 | ✅ |

So there is no Vulkan-side analogue of SYCL's top_k wall. There is, however, one live Vulkan-specific
regression on exactly our model: **PR #28501 ("vulkan: raise the hoisted row-id limit for mul_mat_id from 256
to 512 experts") is still OPEN** (`gh pr view 28501 --json state,mergedAt` → `OPEN`, `mergedAt: null`, last
updated 2026-09-07). `vulkan-shaders/count_experts.comp:26` still hard-codes `#define BLOCK_SIZE 256` and
sizes its shared arrays with it (`:33-35`), so for our 512-expert model the row-id hoisting is disabled. The PR
reports **+16-19 % prefill** on Strix Halo when fixed — i.e. Vulkan is currently leaving a prefill-specific
double-digit percentage on the table for this exact architecture, and it is the *MoE* path, our Term 3.

### 3.4 The B65 7× cliff, re-read

Doc 02 line 304, third-party, Arc Pro B65 32 GB, Windows driver:

> raising `-c` past ~30K on this B65 made *every* prompt ~7× slower — a 10K-token prompt went from 110 t/s
> prefill (`-c 30720`) to 16.8 t/s (`-c 36864`)… The QSA top-k scratch (`ggml_vk_topk_radix_qsa`,
> `n_kv × n_ubatch × 4` bytes) plus the KV grows with `-c`, and once the working set exceeds VRAM the Windows
> driver spills silently over PCIe rather than failing. Moving four more expert layers to the CPU fixed it:
> `-c 49152 --n-cpu-moe 29` = 118.6 t/s prefill.

Now that §3.1 has read the code, this report is fully explained rather than merely plausible: `prealloc_x` at
`ggml-vulkan.cpp:14705` is literally `n_kv × nrows × 4`, and their `-ub 4096` (doc 02 §3.2 quotes
`-b 4096 -ub 4096`) makes it 8× heavier than our `-ub 512` would. At `-c 36864, -ub 4096` that is 576 MiB of
scratch on top of the KV — on a card that was already near its limit at `--n-cpu-moe 25`.

**Three things follow, and they cut both ways.**

- The cliff's *cause* is VRAM exhaustion, and VRAM exhaustion is card- and config-specific. Their B65 ran
  `-ub 4096`; our benchmark ran `-ub 512` and finished with 2,836 MiB spare (`PLAN.md:106-107`). **Inference:**
  we are unlikely to hit it at our current settings, and doc 08's budget already accounts for the compute
  buffer (measured `SYCL0 compute buffer size = 1949.57 MiB` at `-c 262144 -ub 512`, `PLAN.md:66-68`).
- But the silent-degradation *failure mode* is real and is a genuine argument against Vulkan for
  production-shaped long-context work: a config that overshoots does not error, it gets 7× slower, and you only
  find out by benchmarking. SYCL's failure mode at the same point has historically been a hard
  `UR_RESULT_ERROR_OUT_OF_HOST_MEMORY` (doc 02 §3.3, #25812) — unpleasant, but loud.
- **The cliff is not a reason to prefer SYCL-as-it-is-today**, because SYCL's *current* argsort scratch is
  2.7× larger than Vulkan's fused scratch at the same depth (§2.2 table). It is a reason to prefer
  SYCL-with-#28670, which needs none.

### 3.5 What a Vulkan migration would actually cost

The build is not the cost. `docker/Dockerfile.vulkan` already exists, pinned to the same
`LLAMA_COMMIT=6d9c82ea2bb34e277c0664b8dd3434bfb4dcfb27` as the SYCL image "because this repo's Phase 0.2
explicitly wants a same-commit SYCL-vs-Vulkan comparison"; `docker-compose.yml:89-104` wires up
`llm-test-vulkan` on port 8091 with the same model mount; and the image
`qwen4exp-moe-cache:vulkan-pinned` (3.01 GB) **is already built on this box.** Running a Vulkan benchmark
costs GPU time and nothing else.

The cost is everything this project has built on SYCL that does not exist on Vulkan:

- the SYCL MoE expert cache (`ggml/src/ggml-sycl/moe-cache.cpp/.hpp`), the project's entire reason for existing;
- the MTP implementation and its measured acceptance rates (`PLAN.md:210-224`);
- the hybrid CPU/GPU `MUL_MAT_ID` spike and its measured 4.1×-per-row compute advantage
  (`docs/research/05` §10, `PLAN.md:80-91`);
- the local top_k fallback, the `GGML_SYCL_OP_PROFILE` instrumentation (`ggml-sycl.cpp:6453-6600`), and the
  `GGML_SYCL_MOE_PROFILE` stage profiler that all of the above were tuned with.

Critically, **the MoE cache is the only thing on this project's roadmap that addresses Term 3** — the largest
remaining prefill term once Term 1 is fixed. Migrating to Vulkan trades a 100× win on Term 1 (obtainable on
SYCL for half a day of work) for the loss of the only lever on a 3,900-7,800 s term. That is a bad trade.

---

## 4. The dual-backend question, answered directly

The task asks whether to run SYCL for short-context/decode work and Vulkan for long-context/prefill serving.
**Recommendation: no, not as a standing arrangement — but yes as a one-off measurement.**

Against a standing split:

1. **It does not partition cleanly.** The thing that makes long context expensive on SYCL (Term 1) is fixed by
   half a day of vendoring. The thing that makes it expensive on *both* (Term 3) is fixed only by work that
   exists on SYCL alone. After #28670 lands there is no term where Vulkan is structurally ahead except the
   fusion's elision of two intermediate copies — worth something, not worth a fork in the project.
2. **It doubles the bug surface on a model nobody upstream validates for either backend.** Doc 02 finding 5:
   *"the support PR claims testing on CPU, CUDA and Metal only. SYCL/Vulkan support is inherited from generic
   op coverage, not verified."* Both backends needed a 2026-09-09 fix to run this model at all (#28476 SYCL,
   #28592 Vulkan). Running two means finding and fixing bugs in two.
3. **Vulkan carries its own open prefill regression on this exact model** (#28501, §3.3), which nobody in this
   project would be positioned to work around.
4. **Two deployments means two VRAM budgets.** Doc 08's entire `-ncmoe`/`-c` table was computed and then
   verified to the hundredth of a MiB against SYCL's allocator (`PLAN.md:103-107`). None of it transfers.

For the one-off measurement, see §8 — it is cheap and it closes the largest open question in this document.

---

## 5. The term neither option fixes

**Term 3 — CPU-resident MoE at `-ncmoe 24` — is, on this document's arithmetic, comparable to or larger than
Term 1 at 116K and the largest single term at 600K after Term 1 is fixed.** Nothing in Option A or Option B
touches it. The levers that do:

1. **Raise `-ub`.** Free, zero code, immediately available *after* #28670 removes the scratch that scales with
   it (§2.2). `-ub 2048` cuts the CPU expert-bank sweep count 4×. **Estimate**, untested; also raises the FA
   staging buffer, which doc 08 §4 already models.
2. **Lower `-ncmoe`.** The 116K run finished with 2,836 MiB spare and `PLAN.md:106-107` notes `-ncmoe 22`
   would have fit. Each layer moved to GPU removes ~1/24 of Term 3. At 300-600K the KV grows and this lever
   shrinks — doc 08 §4 puts `-ncmoe 24` at 300K and `-ncmoe 28` at 500K.
3. **The MoE expert cache** — the project's original thesis, still unmeasured against prefill. Note
   `docs/00-background.md` §1's warning, restated at `PLAN.md:36-41`: the CUDA fork's own cache *regresses*
   prefill 14-66% because "large-batch prefill has a wide unique-expert set per op, closer to a worst case for
   a small resident pool." With `-ub 512` and 512 experts, **inference**: essentially every expert is touched
   every ubatch, so a resident pool smaller than the full bank has a ~0% hit rate during prefill. **The expert
   cache may be a decode-only win.** That is a first-order question for the project's thesis and it is not
   answered anywhere in `docs/`.

None of this changes the recommendation — Term 1 is still first, because it is the one that scales worst and
the one whose fix is already written — but it should change what comes *second*, and it is a stronger candidate
for second place than sparse FA.

---

## 6. Recommendation

**Option A, narrowed: vendor PR #28670 into our SYCL build, then re-measure. Do not migrate to Vulkan. Do not
run a dual-backend split. Keep sparse FA deferred.**

Sequenced:

| # | action | effort | GPU | expected effect |
|---|---|---|---|---|
| 1 | Vendor #28670 (§2.3), rebuild, `test-backend-ops -b SYCL0 -o TOP_K` | ~1.5 h | ~15 min | correctness gate |
| 2 | **Profile a 32-64K prefill** with `GGML_SYCL_OP_PROFILE=2` (§8) — before *and* after step 1 | ~1 h | ~30 min | **settles §1.3's split; this is the measurement everything else is guessing at** |
| 3 | Re-run the 116K benchmark, same config, for a before/after TTFT | ~1 h | ~1 h | the headline number |
| 4 | If step 2 confirms Term 1 dominated: sweep `-ub 512 / 1024 / 2048` at 116K | ~1 h | ~2 h | attacks Term 3, unblocked by step 1 |
| 5 | One Vulkan run at the same depth/config on the already-built image (§8) | ~30 min | ~1 h | closes the Option-B question with a number |
| 6 | Only then: push to 300K (needs `-c 327680`; see the caveat below) | — | ~2-4 h | first real datapoint at target depth |

**Effort for the decision-relevant part: about one working day, of which ~3 hours is GPU.** That is an order
of magnitude less than either doc 07's sparse-FA scope (~400-700 lines of novel kernel work) or a Vulkan
migration, and it is the step that unblocks knowing which of those, if either, is worth doing next.

**A prerequisite nobody has checked.** `qwen4exp.context_length = 262144` (doc 07 §1.2) and the load banner
reports `rope scaling = linear`, `n_ctx_orig_yarn = 262144` (`logs/phase3-vram-check.log:174-177`). **The
600K target is 2.3× past the model's trained context and the GGUF does not ship a YaRN configuration for it.**
`PLAN.md:9` assumes a "300K-1M" YaRN ceiling; nothing in `docs/` verifies that the flags to reach it are wired
for `qwen4exp`, or what quality costs. 300K is only 1.14× past native and is the safer first target. This is
an independent risk to the whole 600K goal and should be checked before any further optimisation is aimed at
that depth.

---

## 7. The biggest remaining unknown, and the smallest experiment that closes it

**Unknown: what fraction of the measured 1,705 s prefill is Term 1 (top-k) versus Term 3 (CPU MoE).** §1.3
puts Term 1's floor at 437 s and Term 3's range at 754-1,509 s — overlapping bands that sum to roughly the
measured total either way. The recommendation does not hinge on it (Term 1 wins on *scaling* regardless: it is
the only term growing as `N² log²N`, and its fix is nearly free), but *everything after step 1* does.

**The experiment, in full:**

```
GGML_SYCL_OP_PROFILE=2 GGML_SYCL_OP_PROFILE_WINDOW=1 \
  llama-server -m <UD-IQ3_XXS> -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 \
               -ncmoe 24 -c 65536 -ub 512
# then one ~32K-token prompt (truncate the existing 116K benchmark prompt)
```

Why this shape:
- **Mode 2, not mode 1.** The profiler's own comment (`ggml-sycl.cpp:6456-6459`) says mode 1 drains the queue
  around every node and "the run is far slower than production"; mode 2 times only host-side dispatch and
  never syncs. For attributing a prefill that is already 28 minutes long, mode 1 is unusable.
- **`WINDOW=1`.** `sycl_op_prof_report` (`:6569-6586`) is windowed in *scheduler splits*, defaulting to 400,
  with the explicit note that "the first splits of a run are warmup and prefill — one `FLASH_ATTN_EXT` over
  the whole prompt swamped a cumulative average." For a prefill-only question, report every split.
- **The by-call-site table is the payoff.** The profiler keys a second map on the `cb()` tensor name
  (`:6468-6470`), and `build_qsa_top_k` names its tensors `indexer_score_tokens` and `indexer_top_k`
  (`src/models/qwen4exp.cpp:858, 867`). So the report attributes time directly to `indexer_top_k-3`,
  `indexer_top_k-7`, … — Term 1, named, with no inference required.
- **32K, not 116K.** 32K is comfortably past the 16,384 SLM→global-bitonic threshold (§1.2), so the expensive
  regime is engaged, and the run takes minutes rather than half an hour.

**GPU cost: ~30 minutes.** Run it before and after step 1 and the before/after difference *is* Term 1,
measured.

**A second, near-free experiment worth bundling** (this is the honest answer to "is the B65 7× cliff
ours too?"): the Vulkan image is already built and same-commit (§3.5). Run the identical 32K prompt against
`llm-test-vulkan` (port 8091) at the same `-c`/`-ncmoe`/`-ub`, then again at `-ub 2048`. That produces, in
about an hour of GPU: a real Vulkan prefill number on *our* B70 at a depth we care about (doc 02's only Vulkan
datapoint is a *decode* number on a *B65* at 8-32K), and a direct test of whether the scratch-driven cliff
reproduces on a card with more VRAM headroom. **Both runs need the GPU exclusively — check
`ps aux | grep llama` and `docker ps | grep sycl` first; this box has one GPU and concurrent use has corrupted
measurements repeatedly.**

---

## 8. Corrections to fold back into other documents

1. **`PLAN.md:121-132`** says QSA sparse attention "directly targets prefill cost at exactly this depth … and
   is now the clear highest-leverage next step." **Half wrong, and the wrong half matters.** The indexer's
   `ggml_top_k` — paid unconditionally today, whether or not the sparse flag is ever enabled — is the
   high-leverage piece (§1.3, Term 1). The sparse *attention* restriction is Term 2, the smallest of the three
   prefill costs at every depth modelled. Doc 07 §4.3 got this right and should be the citation, not doc 07's
   headline verdict.
2. **`docs/research/02` §2.3 mitigation 2** ("Port a radix-select top_k to SYCL. … Probably the highest-value
   single upstream contribution this project could make") is now **superseded**: someone else made it, on
   2026-09-09, on Battlemage, for this model, and it is `merge ready` (§2.1). The action item is "vendor and
   validate," not "write."
3. **`docs/research/02` §2.3 mitigation 1** ("Use the Vulkan backend instead of SYCL … the lowest-effort path
   to fully-GPU-resident QSA on Arc") was correct when written and is now **not** the lowest-effort path —
   #28670 is, by roughly a factor of ten, and it keeps the MoE cache.
4. **`docs/research/07` §4.3's kernel-launch table** assumed `nrows = 1` (decode) and `n_ubatch = 2048`. At the
   config actually used (`-ub 512`) and summed over a whole cold prefill the numbers are far larger:
   **334,512 launches at 116K, 2.45 million at 600K** (§1.3). Its qualitative conclusion holds; its magnitude
   was an order of magnitude too small for the prefill case.
5. **Doc 02 §2.3's Vulkan claim is verified and, if anything, understated** — `pipeline_topk_radix_qsa` is not
   just a radix kernel but a seven-op graph fusion matching our model's exact indexer chain
   (`ggml-vulkan.cpp:668-681`, `:14680-14727`, `:17920-17985`). Conversely, doc 02 does not record that
   **Vulkan has no sparse FA either** (§3.2) — worth adding, since it is the main thing a reader would assume
   a switch buys.
