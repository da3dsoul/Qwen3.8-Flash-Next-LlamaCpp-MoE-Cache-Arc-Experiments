# CUDA Fork MoE Cache — Structural Outline

Source: `GenerelSchwerz/llama.cpp` fork, `ggml/src/ggml-cuda/moe-cache.cu` (11,402 lines) and `moe-cache.cuh` (1,120 lines), at
`/media/da3dsoul/Golias/AIProjects/Qwen3.8-Flash-Next-LlamaCpp-MoE-Cache-Arc-Experiments/src/moe-cache-fork/ggml/src/ggml-cuda/`.
Cross-checked against `tests/test-moe-cache.cpp` (12,466 lines, targeted reads only — see the last section).

This is a navigational map, not a paraphrase of every line — jump to the `### name (file:LINE-LINE)` heading you need.
Every line range below was read directly (not guessed); where two passes disagreed by a handful of lines the
more precisely-verified number is used. A few items are flagged explicitly as **inference** (connecting facts
established in separate parts of the file) rather than something one chunk stated outright — look for the
"synthesis" callouts.

**Important naming trap:** this file's `ggml_cuda_moe_graph_*` types (`ggml_cuda_moe_graph_plan`,
`ggml_cuda_moe_graph_execution`, `begin_graph_dispatch`, etc.) refer to **ggml's own `ggml_cgraph` compute
graph** — a cached, structurally-keyed dispatch plan for repeated calls with the same graph shape. **The CUDA
Graphs API (`cudaGraphCreate`/`cudaGraphLaunch`/`cudaGraphAddDependencies`/etc.) is never called from
`moe-cache.cu` itself** (confirmed by whole-file grep — zero matches in the .cu/.cuh pair). All device work
*inside this file* is issued as ordinary kernel launches and `cudaMemcpyAsync`/`cudaMemsetAsync` calls on
ordinary streams, ordered via `cudaEvent_t` + `cudaStreamWaitEvent`. There is also no persistent-kernel/
device-side-launch/queue pattern — every kernel is launched from the host, once per call, in the ordinary CUDA
execution model.
**Correction/nuance (verified directly against the test source, not just grep):** the CUDA Graphs API *is*
genuinely exercised, but only from the **caller's** side, in `tests/test-moe-cache.cpp:5786-5841`
(`test_grouped_graph_replay_lifecycle`-style test). That test wraps a `GGML_CUDA_MOE_GRAPH_DISPATCH_CAPTURE`
dispatch call in a real `cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed)` /
`cudaStreamEndCapture(&captured_graph)` / `cudaGraphInstantiate(&captured_instance, captured_graph, ...)` /
`cudaGraphLaunch(captured_instance, stream)` sequence, walks the captured node list asserting no
`cudaGraphNodeTypeWaitEvent`/`cudaGraphNodeTypeEventRecord` nodes got captured (i.e. the CAPTURE-mode dispatch
path in `activate_graph_resources`/`begin_graph_dispatch` is deliberately written to avoid emitting
event-wait/event-record nodes into the capture — those would be illegal or semantically wrong inside a replayed
graph), then later does a second dispatch in `GGML_CUDA_MOE_GRAPH_DISPATCH_REPLAY` mode and calls
`cudaGraphLaunch(captured_instance, stream)` **again** on the *same* pre-captured graph object, confirming
capture-once/replay-many actually works end to end against this file's dispatch-mode state machine. So: **zero
CUDA Graph API calls inside `moe-cache.cu`, but the file's `DISPATCH_CAPTURE`/`DISPATCH_REPLAY` modes are
real, load-bearing plumbing for an external CUDA Graph capture/replay done by the caller, and this is proven
end-to-end by the test suite** — not merely aspirational naming. This is directly relevant to the SYCL port: the
equivalent of "capture once, replay many, zero per-token kernel-launch overhead" is achievable via SYCL's graph
extension (`sycl_ext_oneapi_graph`) only if the per-token dispatch path is similarly kept free of anything that
can't be captured (no host-visible event waits inside the captured region, no host branching on device data).

---

## 0. Overall architecture & data flow

Two **independent, coexisting** cache-acquisition paths run in the same process, sharing (when possible) the
same VRAM, plus one shared "cold tier" buffer type and one optional pinned-host staging tier:

- **The "legacy" path** — one `ggml_cuda_moe_cache` object per weight *tensor* (e.g. one for `ffn_gate_exps`,
  one for `ffn_up_exps`, one for `ffn_down_exps`). Fully **host-driven**: a plain C++ function
  (`ggml_cuda_moe_cache_acquire_locked`) does a hash-map lookup, a host-side LRU eviction scan, an optional L2
  promotion-source lookup, and issues one explicit `cudaMemcpyAsync` H2D per miss. This is architecturally the
  closest analogue to a naive/first-cut SYCL port.
- **The "grouped" path** — one shared per-candidate-group device resource. Admission (which expert gets which
  slot, which resident expert gets evicted) is decided **entirely on-device**: a single-block kernel
  (`moe_grouped_plan_decode`) reads the already-device-resident routing-ids tensor and, via warp/block-level
  min-reductions over per-slot decayed-frequency+age scores, computes eviction/admission decisions and writes a
  compact device-resident "plan" (miss list: expert→slot pairs, plus a full per-route slot remap). A second
  kernel (`moe_grouped_gather_decode`), launched immediately after on the **same stream with zero host
  round-trip in between**, reads that plan and performs the actual weight-row copy: a grid-stride,
  16-byte-word-granularity **bulk coalesced copy** per missed expert, from `banks[].source` to `banks[].data`.
  Critically, `banks[].source` is not a raw pageable host pointer — it is a **UVA/zero-copy device alias of
  pinned host memory**, obtained once (at resource-build time, host-side, no kernel work) via
  `cudaHostGetDevicePointer` (see `device_alias`). So the gather kernel's "H2D copy" is really an ordinary GPU
  kernel doing wide, contiguous, coalesced *global-memory loads* against a PCIe-mapped pinned-host address —
  fundamentally different from the naive SYCL port's scattered, fine-grained, per-element host reads used
  directly for arithmetic. **No compute kernel in the grouped path ever reads pageable/unregistered host memory
  directly for per-element math; every byte a compute kernel touches is either already in VRAM or reached
  through a pinned+UVA alias designed for exactly this access pattern.**

Both paths draw their raw expert-weight bytes from the same **cold tier**: a `ggml_backend_buffer_type_t`
(name `"CUDA_MoE_Cached"`) that wraps *either* (a) freshly `cudaMallocHost`-allocated pinned RAM, *or* (b) an
existing pointer into the model file's `mmap()` mapping, tracked in a residency-instrumentation registry. Which
of the two is actually used for a given model's expert tensors is decided by **llama.cpp's model loader**,
outside this file — both code paths exist and are exercised by the test suite side-by-side.

An optional **L2 tier** (`moe_cache_l2`) is a small, separately-budgeted **pinned-host** staging cache (its pool
field is literally named `slot_pool_h`), keyed by raw host source pointer. It is consulted **only by the legacy
path**, and **only when the legacy cache's source is mmap'd** (`cache->source_is_mmap`) — i.e. it exists
specifically to convert a "cold, possibly-nonresident, non-pinned mmap page" into "a resident pinned buffer a
fast `cudaMemcpyAsync` can source from," before the legacy path's H2D copy runs. The grouped/device-kernel path
never touches L2 — it solves the same underlying problem (mmap'd cold data isn't zero-copy-mappable) a different
way, by requiring `device_alias`'s `cudaPointerGetAttributes` classification to succeed, which (see the L2
section below) most plausibly means **the fully-grouped fast path is only reachable for cold-tier sources that
are genuinely CUDA-registered pinned memory** — an unregistered raw `mmap()` region falls back to the legacy
path, where L2 exists precisely to soften that case.

```
                       ┌─────────────────────────────────────────────────────────┐
                       │  COLD TIER  (ggml_backend_buffer_type "CUDA_MoE_Cached") │
                       │  (a) cudaMallocHost pinned RAM   -- OR --                │
                       │  (b) wrapped pointer into mmap'd GGUF file               │
                       │      (tracked in mmap residency-instrumentation table)   │
                       └───────────────┬───────────────────────┬─────────────────┘
                                        │                       │
                     LEGACY PATH        │                       │   GROUPED PATH
            (per-tensor, host-driven)   │                       │  (per-group, on-device)
                                        ▼                       ▼
                  ┌────────────────────────────┐   ┌──────────────────────────────────┐
                  │ optional L2 staging tier    │   │ device_alias(): cudaPointerGet-   │
                  │ (moe_cache_l2, PINNED HOST, │   │ Attributes + cudaHostGetDevice-   │
                  │ keyed by host ptr, engaged  │   │ Pointer -> UVA device alias of    │
                  │ only when source_is_mmap)   │   │ pinned host memory (no copy)      │
                  └──────────────┬───────────────┘   └──────────────┬────────────────────┘
                                 │ cudaMemcpyAsync                  │ gather kernel reads
                                 │ HostToDevice                     │ the alias directly —
                                 │ (host acquire_locked)            │ bulk coalesced uint4
                                 ▼                                  ▼   loads, no memcpy call
                  ┌────────────────────────────┐   ┌──────────────────────────────────┐
                  │ HOT SLOT POOL (per tensor)  │   │ HOT SLOT POOL (per group,         │
                  │ cudaMalloc'd VRAM,          │   │ own cudaMalloc, can be borrowed   │
                  │ host-driven LRU eviction    │   │ by legacy path — pool sharing)    │
                  │                              │   │ admission decided ON-DEVICE by    │
                  │                              │   │ moe_grouped_plan_decode kernel,   │
                  │                              │   │ consumed same-stream, zero host   │
                  │                              │   │ round trip, by moe_grouped_       │
                  │                              │   │ gather_decode kernel              │
                  └──────────────┬───────────────┘   └──────────────┬────────────────────┘
                                 │                                  │
                                 └───────────────┬──────────────────┘
                                                  ▼
                                   GEMM / expert-compute kernels
                             (only ever read from VRAM — hot pool)
```

Synchronization in both paths is **exclusively** `cudaEvent_t` + `cudaStreamWaitEvent` (device-side, never
host-blocking) on the hot path; genuine host-blocking calls (`cudaStreamSynchronize`/`cudaEventSynchronize`)
occur only in pool-growth, teardown, and error/fallback branches — see Topic 6.

---

## Topic 1 — The Cold Tier / Buffer Type (pinned host memory)

The cold tier is implemented as a custom `ggml_backend_buffer_type_t` named `"CUDA_MoE_Cached"`, with **two
distinct, coexisting construction paths** feeding the same buffer type. Which path a given model's expert
tensors go through is decided entirely by llama.cpp's model loader (outside this file); the test suite
exercises both side-by-side and asserts they produce identical grouped-decode results.

### ggml_cuda_moe_cached_pinned_malloc (moe-cache.cu:10998-11011)
```cpp
static void * ggml_cuda_moe_cached_pinned_malloc(size_t size) {
    if (getenv("GGML_CUDA_NO_PINNED") != nullptr) { return nullptr; }
    void * ptr = nullptr;
    cudaError_t err = cudaMallocHost((void **) &ptr, size);
    if (err != cudaSuccess) { /* log DEBUG, clear sticky error */ return nullptr; }
    return ptr;
}
```
The **only** CUDA allocation call in the whole buffer-type section. Uses plain **`cudaMallocHost`** — functionally
`cudaHostAlloc(ptr, size, cudaHostAllocDefault)`, i.e. **flags = 0**. No `cudaHostAllocMapped`, no
`cudaHostAllocPortable`. Escapable via the `GGML_CUDA_NO_PINNED` env var (forces `nullptr` unconditionally, for
testing the CPU-fallback path) and gracefully returns `nullptr` on genuine allocation failure (sticky CUDA error
cleared via `cudaGetLastError()`, logged at DEBUG level) — callers treat both cases identically.

### ggml_backend_cuda_moe_cached_buffer_type_alloc_buffer (moe-cache.cu:11013-11028)
```cpp
void * ptr = ggml_cuda_moe_cached_pinned_malloc(size);
if (ptr == nullptr) {
    // costs PCIe bandwidth on cache miss but keeps the model loadable
    return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
}
ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
buffer->buft = buft;
buffer->iface.free_buffer = ggml_backend_cuda_moe_cached_buffer_free_buffer;
return buffer;
```
This is the **standard ggml `alloc_buffer` vtable slot** — the function ggml's own generic allocator calls
whenever it wants a *new* buffer of this type (e.g. via `ggml_backend_alloc_ctx_tensors_from_buft` during model
load). Path: `cudaMallocHost` → wrap with `ggml_backend_cpu_buffer_from_ptr` → stamp `buft`/`free_buffer`. On
pinned-alloc failure it falls back to a **plain CPU buffer**, never to mmap. Does **not** touch the mmap
registry. **This is unambiguously path (a): dedicated `cudaMallocHost`-pinned memory.**

### ggml_backend_cuda_moe_cached_buffer_from_host_ptr (moe-cache.cu:11064-11075)
```cpp
ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
buffer->buft = ggml_backend_cuda_moe_cached_buffer_type();
buffer->iface.free_buffer = ggml_backend_cuda_moe_cached_mmap_buffer_free_buffer;
moe_cache_register_mmap_range(ptr, size);
return buffer;
```
A **second, non-allocating** public entry point: wraps an **already-existing** host pointer (almost certainly
llama.cpp's own `mmap()`'d GGUF-file region, handed in by the loader) — **zero CUDA API calls**. It sets a
*different* `free_buffer` (calls `moe_cache_unregister_mmap_range`, never `cudaFreeHost` — decisive evidence the
pointer wasn't `cudaMallocHost`'d) and registers `[ptr, ptr+size)` in the process-global mmap-residency
registry. **This is path (b): a thin wrapper around externally-owned, presumably-mmap'd memory**, confirmed real
and exercised (see Test File Confirmation, item 5), not vestigial.

### ggml_backend_cuda_moe_cached_buffer_free_buffer / _mmap_buffer_free_buffer (moe-cache.cu:10990-10998)
Two distinct `free_buffer` callbacks exist *specifically* because the two construction paths need different
teardown: `cudaFreeHost(buffer->context)` for the pinned-malloc path (10990-10992) vs.
`moe_cache_unregister_mmap_range(buffer->context, buffer->size)` — no CUDA free call at all, no `munmap()`
either (ownership of the mapping stays with whoever created it) — for the wrapped-pointer path (10994-10998).

### ggml_backend_cuda_moe_cached_buffer_type_is_host (moe-cache.cu:11036-11039) — non-obvious, important
```cpp
static bool ggml_backend_cuda_moe_cached_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return false;
}
```
Preceded by an explicit comment (11030-11035), quoted in full:
> `// is_host MUST return false (or be NULL) for this buffer type, even though`
> `// the data is technically in pinned host memory. If is_host returns true,`
> `// ggml's scheduler treats tensors here as CPU-backend-resident and routes`
> `// mul_mat_id ops to the CPU backend, completely bypassing our dispatch hook.`
> `// We rely on CUDA reading the pinned mapping directly via cudaMemcpyAsync`
> `// for the H2D copy in the dispatch hook.`
This is a deliberate override: ggml's scheduler would otherwise special-case a "host" buffer type and run
`mul_mat_id` on the CPU backend directly, silently bypassing the entire custom MoE-cache dispatch hook. Forcing
`is_host=false` (for **both** cold-tier flavors — the comment's "the pinned mapping" is generic) keeps every
MoE-cached tensor routed through this file's dispatch/cache machinery. Note the comment describes the generic
`cudaMemcpyAsync` path (the legacy path's mechanism); it does not mention the grouped path's `device_alias`
zero-copy mechanism at all, nor the performance cost of `cudaMemcpyAsync` from genuinely non-pinned mmap memory.

### ggml_backend_cuda_moe_cached_buffer_type / ggml_backend_buft_is_cuda_moe_cached (moe-cache.cu:11041-11062)
Meyers-singleton buffer-type struct (borrows `get_alignment`/`get_alloc_size` from the CPU buffer type's iface);
`ggml_backend_buft_is_cuda_moe_cached` is a function-pointer identity check (`buft->iface.get_name ==
ggml_backend_cuda_moe_cached_buffer_type_name`) used throughout the file to recognize this buffer type without
RTTI.

### Summary: two kinds of host memory, one uniform buffer type
`is_host` and `ggml_backend_buft_is_cuda_moe_cached` return identical results regardless of which construction
path built the buffer — from ggml's perspective the two are indistinguishable. Internally, though, they diverge
sharply: (a) `cudaMallocHost`-backed buffers are genuinely CUDA-pinned, and (per the `device_alias`
finding in Topic 4) are the memory class the grouped path's `cudaHostGetDevicePointer`-based zero-copy alias can
actually succeed against; (b) mmap-wrapped buffers are ordinary OS-paged memory from CUDA's point of view unless
something elsewhere registers them with the driver (no `cudaHostRegister` call appears anywhere in this file),
so `cudaPointerGetAttributes` on such a pointer would not classify it as `cudaMemoryTypeHost` — **inference**:
this is most plausibly why the mmap case is expected to fall back to the legacy path (where `cudaMemcpyAsync`
against pageable memory still works, just without pinned-DMA speed, and L2 exists to buy back that speed).

### Instrumentation: mmap registry & page-fault/residency tracking (moe-cache.cu:330-415, 137-328) — supporting material, not the allocator itself
- `struct moe_cache_mmap_range { begin, end }` / `moe_cache_mmap_registry { mutex, vector<range> }` (330-338);
  `get_mmap_registry()` (340-343, Meyers singleton); `moe_cache_register_mmap_range`/`_unregister_mmap_range`
  (345-371, push/erase under mutex); `moe_cache_is_mmap_range` (373-388, linear-scan containment check). This
  registry is what lets other code (`cache->source_is_mmap`, set at legacy-cache-build time via
  `moe_cache_is_mmap_range(tensor->data, ...)` at moe-cache.cu:5932) and the grouped path's
  `group_source_mapped`/`registered_mmap_bank_count` (4340-4347, 4570-4602) classify a given tensor's backing
  pointer at runtime.
- `moe_cache_mm_sample_mincore` (137-208, two overloads, Linux-only real impl) and the `/proc/self/stat`/
  `/proc/vmstat` snapshot-delta helpers (210-328) are pure **telemetry**: they periodically sample `mincore()`
  over registered mmap ranges to estimate what fraction of "cold tier" pages are actually resident in the OS
  page cache vs. paged out, and track process-wide minor/major fault and swap counters. None of this feeds any
  control-flow decision — it only feeds the `moe_cache_mm_stats` reported via
  `ggml_cuda_moe_cache_mm_stats`/`moe_cache_log_telemetry`.

---

## Topic 2 — The L2 Staging Tier

**Verdict: L2 is a genuine two-stage promotion path (cold mmap'd host → L2 pinned-host staging → hot VRAM
pool), but L2 itself lives in *pinned host memory*, not VRAM — and it is used only by the legacy path, and only
when the legacy cache's source is mmap'd.**

### moe_cache_l2 struct (moe-cache.cu:390-414)
Fields: `slot_size_bytes`, `n_slots`, `void * slot_pool_h` (note the `_h` suffix — **host** pointer), parallel
per-slot vectors `slot_to_host`/`last_used`/`is_protected`, a reverse map `host_to_slot`, an `access_counter`
LRU clock, and hit/miss/fill/eviction counters (aggregate + per-phase `[2]`).

### moe_cache_l2_init (moe-cache.cu:416-450)
**(a) Memory type — pinned HOST, not VRAM.** One allocation for the whole pool:
`cudaMallocHost(&ptr, n_slots * slot_size_bytes)` (line 425). Respects `GGML_CUDA_NO_PINNED`. Returns `false`
on failure (caller sets `cache->l2_alloc_failed = true` and permanently skips L2 for that cache instance).

### moe_cache_l2_select_slot (moe-cache.cu:480-507)
Three-tier victim selection among the fixed pool of `n_slots` L2 slots: (1) first empty slot, else (2) LRU among
slots not currently `is_protected` (protected = "a caller currently holds a pointer into this slot, don't evict
out from under it"), else (3) a fallback plain-global-LRU ignoring the protected flag so the function always
returns *something*.

### moe_cache_l2_acquire (moe-cache.cu:509-561)
End-to-end trace:
- **Hit** (517-524): mark the slot `is_protected=1`, bump its LRU timestamp, return a pointer into
  `slot_pool_h` — no CUDA call at all on the hit path.
- **Miss**: pick a victim (`moe_cache_l2_select_slot`). If the victim currently holds different data (an
  eviction), **line 535**: `cudaStreamSynchronize(copy_stream)` — a genuine host-blocking call, but only on this
  specific eviction sub-path — it fences against a still-in-flight async `cudaMemcpyAsync` that might still be
  reading *from* this pinned slot (it was the source of an earlier H2D copy into VRAM) before the code
  overwrites it.
- **The fill itself** (line ~546-547): a plain **CPU-side, synchronous `memcpy(dst, host_src, byte_count)`** —
  not a CUDA call at all. This is exactly where an OS page fault (and, if the mmap'd page is genuinely
  nonresident, a disk read) would be paid, on the CPU, before anything touches the GPU.
- Returns a pointer into `slot_pool_h` (host memory) either way.

### moe_cache_l2_free (moe-cache.cu:452-478)
`cudaFreeHost(l2.slot_pool_h)` plus full field reset.

### ggml_cuda_moe_cache_l2_source (moe-cache.cu:9544-9563) — the connector to the copy path
```cpp
if (!cache->source_is_mmap || cache->l2_target_slots <= 0 ||
        cache->l2_budget_bytes == 0 || cache->l2_alloc_failed) {
    return host_src;                    // L2 bypassed entirely
}
if (cache->l2.slot_pool_h == nullptr) {
    if (!moe_cache_l2_init(...)) { cache->l2_alloc_failed = true; return host_src; }
}
const void * l2_src = moe_cache_l2_acquire(cache->l2, host_src, byte_count, is_decode, copy_stream);
return l2_src ? l2_src : host_src;
```
This is the function that decides, **per miss**, whether the upcoming H2D copy sources from L2 or falls straight
through to the cold pointer. It is engaged **only** when `source_is_mmap` is true (i.e. this legacy cache's
weight tensor lives in a registered mmap range) and an L2 budget was actually configured; otherwise it's a pure
pass-through. Its return value becomes the literal `copy_src` used by `ggml_cuda_moe_cache_acquire_locked`
(moe-cache.cu:9760) a few lines before the real `cudaMemcpyAsync`. Note the copy is **always**
`cudaMemcpyHostToDevice` (moe-cache.cu:9786) regardless of whether the source is L2 or cold — never
`cudaMemcpyDeviceToDevice` — reinforcing that L2 is host memory, not a second VRAM tier.

### Is L2 used by the grouped path? No.
`moe_cache_l2_acquire`/`ggml_cuda_moe_cache_l2_source` are called only from the legacy cache's
`acquire_locked`/`copy_to_staging`/`prepare_split_staging` functions. The grouped path's `device_alias`
(Topic 4) solves the "mmap isn't fast-DMA-able" problem structurally differently — by requiring the source to
already be CUDA-registered pinned memory so a zero-copy alias can be formed, rather than by staging a copy.
**Synthesis (inference, not directly stated in one place):** this strongly suggests a mmap-backed cold tier is
only reachable via the *legacy* dispatch path (where L2 exists to compensate for its lack of true pinning),
while the *grouped* fast path is only viable for `cudaMallocHost`-backed cold tier tensors. This would be a very
important finding for the SYCL port if confirmed by tracing `group_source_mapped`'s actual call sites and
llama.cpp's loader behavior — flagged here as the most valuable next research step.

### L2 config & stats (brief)
`ggml_backend_cuda_moe_set_l2_pinned_cache_size` / `_get_l2_pinned_cache_size` (moe-cache.cu:11111-11119) — the
only public tuning knob for L2, an atomic `size_t` in bytes; **the name itself** ("pinned_cache_size") is
independent corroborating evidence for the host-memory conclusion. `ggml_cuda_moe_cache_l2_stats`
(10364-10390) reports `budget_bytes`/`slots`/`used_bytes` plus L2's own hit/miss/fill/eviction counters, gated
on `cache->l2.slot_pool_h != nullptr` (lazily initialized).

---

## Topic 3 — Admission / Eviction: on-device warp-reduction (grouped path)

**Confirmed: the eviction/admission decision is computed entirely on-device, via a genuine cooperative
warp-level (and, for larger slot counts, two-level warp-then-block) shuffle-based min-reduction — no host
involvement in the decision itself.**

### moe_grouped_effective_frequency (moe-cache.cu:2786-2795) `__device__`
```cpp
if (stored_epoch > current_epoch) return UINT32_MAX;         // corruption sentinel
const uint64_t elapsed = current_epoch - stored_epoch;
return elapsed < 32 ? frequency >> elapsed : 0;
```
A **decayed-LFU** score: raw per-expert hit count (`expert_frequency[]`), halved once per "epoch" elapsed since
last touch (an epoch = 16 decode-plan calls, via `MOE_GROUPED_FREQUENCY_EPOCH_SHIFT = 4`), floored to 0 once
32+ epochs (512+ calls) have passed. This is the eviction-priority key — lower = more evictable.

### moe_grouped_warp_min / moe_grouped_warp_min_slot / moe_grouped_lane_mask_lt (moe-cache.cu:2797-2830) `__device__`
`moe_grouped_warp_min` (2797-2810): a textbook **butterfly `__shfl_xor_sync` reduction** over `(frequency, age,
slot)` triples held by reference in each lane's registers, offset halving `16,8,4,2,1`, comparator = strict
lexicographic min(frequency) → min(age, i.e. LRU tie-break) → min(slot, deterministic final tie-break). After
the loop every lane holds the identical winning triple — a genuine cooperative 32-thread reduction in 5 rounds,
entirely in registers, zero shared/global memory round-trips, zero host involvement.
`moe_grouped_warp_min_slot` (2812-2825) is a faster single-warp-only variant that skips carrying `slot` through
the shuffles at all (lane id *is* the slot identity in that fast path) and uses `__ballot_sync` +
find-first-set to pick the lowest-lane winner among ties. `moe_grouped_lane_mask_lt` (2828-2830) is a small
warp-aggregated-atomics helper used in the route-dedup path.

### moe_grouped_plan_decode (moe-cache.cu:2856-3207) `__global__` — THE ADMISSION KERNEL
Single-block kernel (`if (blockIdx.x != 0) return;`). Phases, in order:
1. **Commit prior plan** (2897-2934) — the *previous* call's already-computed decisions (still sitting in the
   plan buffer from last time) are applied to the persistent `slot_for_expert`/`expert_for_slot`/`last_used`/
   `expert_frequency` tables *first*, before this call's new decisions are computed. The plan buffer is
   deliberately reused/pipelined across calls, one call in arrears.
2. **Epoch/clock bookkeeping** (2937-2966) — advances a global atomic `device_step` counter (→ `frequency_epoch`)
   and, if a device-clock is in use, atomically reserves a `[clock_begin, clock_end)` range via CAS.
3. **Route ingestion** (2971-3106) — reads `ids[]` (the actual device-resident router top-k output — the only
   "real" input data), deduplicates the routed expert set via a warp-fast path (`__match_any_sync`/
   `__ballot_sync`, single-token decode) or a general `atomicMin`-based path (larger batches).
4. **Direct-mapped residency lookup** — `slot_for_expert[expert]`/`expert_for_slot[slot]` (`int32_t` arrays,
   `-1` = empty), an O(1) array index, not a hash table.
5. **Eviction-victim selection per miss** (3124-3183) — for each miss, in sequence, the whole block cooperatively
   scans free candidate slots, computes local `(frequency,age,slot)` minima via `moe_grouped_effective_frequency`,
   and reduces via `moe_grouped_warp_min`/`_slot`; the winning slot is claimed immediately so the next miss in
   the same call can't pick it again.
6. **Finalization** (3185-3206) — writes `remapped_ids[route] = unique_slots[...]` for every one of the token's
   routes (this is the slot-index table the downstream GEMM kernels actually consume), records `plan->n_misses`/
   `n_unique`, and flips `plan->status = MOE_GROUPED_PLAN_READY`.

Every array touched — `ids`, `slot_for_expert`, `expert_for_slot`, `last_used`, `expert_frequency`,
`expert_frequency_epoch`, `device_step`, `device_clock`, the whole `plan` buffer — is a device pointer. The only
host-originated values are two plain scalar clock bounds, never dereferenced as pointers.

---

## Topic 4 — How a cache MISS is actually serviced in the grouped compute path (the crux)

**Answer: no general compute kernel in the grouped path ever dereferences pageable/unregistered host memory for
per-element math. Every miss's weight row is either (a) copied host→VRAM via ordinary `cudaMemcpyAsync` before
any compute touches it — this is what the legacy path does — or, in the grouped path, (b) read by a dedicated
bulk-copy kernel (`moe_grouped_gather_decode`) performing wide, contiguous, coalesced loads against a
UVA/zero-copy device alias of *pinned* host memory. The actual GEMM/expert-compute kernels that follow only ever
read from the hot VRAM slot pool — they never see the cold-tier pointer at all.**

### moe_grouped_gather_decode (moe-cache.cu:3209-3273) `__global__` — THE MISS-SERVICING KERNEL (grouped path)
```cpp
template<bool debug_transfers>
static __global__ void moe_grouped_gather_decode(
        const moe_grouped_device_bank * banks, uint32_t n_banks, size_t words_per_miss,
        const moe_grouped_device_auxiliary * auxiliaries, uint32_t n_auxiliaries,
        size_t auxiliary_values_per_miss, uint32_t plan_capacity,
        const moe_grouped_decode_plan * plan, uint64_t * transfer_counters);
```
`moe_grouped_device_bank { const char * source; char * data; size_t expert_stride; }` — a plain descriptor pair.
Access pattern: `total_words = words_per_miss * plan->n_misses`; each thread computes a grid-stride sequence of
contiguous `uint4` (16-byte) word indices, decomposes into `(miss, bank_word)` via the plan's miss list, and does
`destination[bank_word] = source[bank_word]` — **one coalesced, contiguous, thread-per-16-byte-word bulk row
copy per missed expert**, the architectural opposite of a scattered per-element gather feeding immediate math.
The `bool debug_transfers` template parameter (resolved by reading both call sites, moe-cache.cu:6473/6478) is
**purely a telemetry on/off switch** (atomically tallies transfer-count/byte-count into `transfer_counters` when
`true`) — it is *not* a legacy/grouped or decode/prefill mode switch.

`descriptor.source` is what makes this safe/fast: see `device_alias` next.

### device_alias (moe-cache.cu:4512-4568) `static` — the load-bearing gate
```cpp
cudaPointerGetAttributes(&attributes, buffer_base);
if (attributes.type == cudaMemoryTypeHost) {
    cudaHostGetDevicePointer(&alias_base, buffer_base, 0);   // UVA zero-copy alias, NO copy
    if (host_alias) *host_alias = true;
} else if (allow_device && attributes.type == cudaMemoryTypeDevice && attributes.device == device) {
    alias_base = buffer_base;                                 // already resident, use directly
} else {
    return false;                                             // not aliasable at all
}
```
This is the function that resolves a candidate weight bank's source pointer into something a kernel can
dereference, **without ever issuing a copy**. For host-pinned memory it calls **`cudaHostGetDevicePointer`** —
the canonical CUDA zero-copy/UVA API — obtaining a device-visible address that, when a kernel loads from it,
transparently performs a PCIe-mapped read. Weight-bank sources are resolved with `allow_device=false`
(moe-cache.cu:4761, inside `make_device_resource`) — i.e. **bank sources are required to be host-pinned
memory**; an already-device-resident tensor is explicitly *not* accepted as a weight-bank source on this path.
`cudaPointerGetAttributes` only classifies memory the CUDA driver already knows about (registered via
`cudaHostAlloc`/`cudaMallocHost`/`cudaHostRegister`); a raw, unregistered `mmap()` pointer would not be reported
as `cudaMemoryTypeHost` (**inference** — no chunk found a `cudaHostRegister` call anywhere in this file), which
is the basis for the Topic 2 synthesis that mmap-backed cold-tier tensors most likely can't use the grouped fast
path at all.

### make_device_resource (moe-cache.cu:4701-4889) `const` — pool allocation & descriptor staging
This is the function that actually `cudaMalloc`s the grouped path's VRAM, but it is important to be precise
about what it copies and what it doesn't:
- **cudaMalloc's its own dedicated VRAM slot pool per bank** (line ~4774, size = `expert_stride * n_slots`) — a
  separate allocation owned by this specific `grouped_device_resource`, sized by *cache capacity* (`n_slots`),
  not by total expert count. This is **not** a call into the legacy `ggml_cuda_moe_cache_acquire`/shared-pool
  API — grouped resources manage their own allocations directly (though see Topic 7 for the "legacy cache
  borrows VRAM from a grouped resource" pool-sharing case, the reverse direction).
- Resolves each bank's *source* pointer via `device_alias(..., allow_device=false, ...)` (line 4761) — no copy.
- **Uploads only small descriptor arrays** host→device: `cudaMemcpyAsync(result->device_banks,
  device_banks.data(), n_banks * sizeof(moe_grouped_device_bank), cudaMemcpyHostToDevice, compute_stream)`
  (4855-4857) — this moves `{source ptr, dest ptr, stride}` triples, **never the actual weight payload**. The
  weight payload is fetched later, entirely inside `moe_grouped_gather_decode`.
- The one genuine bulk weight-payload H2D copy in this function is for **small, always-resident "prefill
  bias" auxiliaries only** (`cudaMemcpyAsync(result->prefill_auxiliary_data[i], ..., byte_extent, ...)`,
  4868-4870) — a deliberately-eager-resident small-tensor optimization, unrelated to the main per-expert weight
  banks.
- Creates the resource's completion event (`cudaEventCreateWithFlags(&result->completion, cudaEventDisableTiming)`,
  4810) and records it (`cudaEventRecord(result->completion, compute_stream)`, 4881) once all setup is enqueued.
  **Every `cudaStreamSynchronize` call in this function (4838, 4863, 4873, 4882) sits strictly inside an
  error-cleanup branch** — none execute on the success path.

### Contrast — ggml_cuda_moe_cache_acquire_locked (moe-cache.cu:9661-9812) — the LEGACY path's miss servicing
The legacy path *does* do a genuine bulk `cudaMemcpyAsync` of the full expert row on every miss
(`cudaMemcpyAsync(dst, copy_src, byte_count, cudaMemcpyHostToDevice, copy_stream)`, line 9786), where `dst` is a
VRAM hot-pool slot and `copy_src` is either the raw cold-tier pointer or an L2-staged pinned pointer (Topic 2).
Victim selection here is a plain host-side LRU scan (not frequency-weighted — `expert_access_counts` is
telemetry-only and is never consulted by the eviction logic), confirming the two paths use genuinely different
admission *strategies* even though they share the same underlying "never let compute read cold memory directly"
guarantee.

---

## Topic 5 — What decides which experts to copy for a token, without a host-blocking ids readback

**Answer: the routing-ids readback the naive approach would need simply never happens, because both the
decision *and* the copy-issuing kernel are on-device and stream-ordered — there is no cross-token lookahead
mechanism; the trick is that a single token's own admission-kernel output feeds its own gather-kernel input
purely via CUDA's same-stream in-order execution guarantee.**

### prepare_decode (moe-cache.cu:6327-6507) — the per-decode-step grouped-path orchestrator
After ensuring the candidate group's VRAM resource exists (via `acquire_group_resources`/`make_device_resource`,
called earlier in this same function if needed) and reserving a clock window, it launches:
```cpp
moe_grouped_plan_decode<<<1, plan_threads, 0, compute_stream>>>(
    static_cast<const int32_t *>(ids->data), n_routes, top_k, row_stride, ... , device.plan);
CUDA_CHECK(cudaGetLastError());                       // async error check only, never blocks
// ... (host-side grid-size arithmetic, no device calls)
moe_grouped_gather_decode<true/false><<<transfer_blocks, MOE_GROUPED_TRANSFER_THREADS, 0, compute_stream>>>(
    device.device_banks, ..., device.plan, transfer_counters);
CUDA_CHECK(cudaGetLastError());
```
Both kernels launch on the **same `compute_stream`**, back-to-back, with **zero host synchronization of any
kind between them** — no `cudaStreamSynchronize`, no `cudaEventSynchronize`, no blocking `cudaMemcpy`. `ids`
(the current token's already-computed router output) is passed straight through as a device pointer, never
read back to host. `moe_grouped_plan_decode` writes its miss list directly into `device.plan` (a device
allocation); `moe_grouped_gather_decode` reads that same buffer as its own kernel argument a few microseconds
of stream-time later — the dependency is enforced *purely* by CUDA's guarantee that operations on one stream
execute in launch order, not by any CPU-visible signal. This is the direct, definitive answer to the "how does
a device-computed decision turn into a host-schedulable copy" question: **it doesn't need to** — the copy isn't
host-schedulable in the traditional sense (no host-side addresses/byte-counts are computed per miss); it's a
second kernel that reads the first kernel's device-resident output as its own input.

### finish_decode (moe-cache.cu:6509-6528)
Called after the caller has also enqueued whatever downstream compute consumes the gathered banks (on the same
stream). Records the resource's `completion` event: `cudaEventRecord(resource->device->completion,
compute_stream)` (line 6521) — again async, no host block — marking the point future cross-stream consumers must
wait behind (Topic 6).

### Is there genuine cross-token lookahead anywhere?
Two adjacent mechanisms exist but neither is a "prefetch based on future routing":
- `prefetch_legacy_siblings` (moe-cache.cu:6081-6135) — given a legacy lease already acquired for **this
  token's already-known** expert ids on one tensor (e.g. `down_proj`), it eagerly warms the *sibling* tensors
  (`gate_proj`/`up_proj`) for the *same, already-decided* experts — a same-token, cross-tensor warm-up, not a
  cross-token prediction.
- `ggml_backend_cuda_moe_prefetch_experts` (moe-cache.cu:11152-11166) — a public API shaped exactly for
  per-token runtime prefetch (`eids`/`n_eids`/`is_decode` parameters), but its body in this revision is a
  **complete no-op** (every parameter `GGML_UNUSED`, empty body) — flagged explicitly as unwired/disabled, not
  evidence of an active lookahead scheme.

---

## Topic 6 — Synchronization: the complete site inventory

**No CUDA Graphs API call appears inside `moe-cache.cu`/`.cuh` themselves (confirmed by whole-file grep of
`cudaGraph*`/`cuGraph*` — zero matches). All ordering *within this file* is plain `cudaEvent_t` +
`cudaStreamWaitEvent` (device-side, never blocks the host thread) on the hot path. Genuine host-blocking
`cudaStreamSynchronize`/`cudaEventSynchronize` calls exist only in pool-growth, teardown, and error/fallback
branches — never in per-token steady-state operation. See the intro section above for the important nuance that
the file's `CAPTURE`/`REPLAY` dispatch modes ARE real CUDA-Graph-capture plumbing, proven by a test that drives
them through an actual `cudaStreamBeginCapture`/`cudaGraphInstantiate`/`cudaGraphLaunch` cycle — the CUDA Graphs
API itself is just never called from inside this translation unit.**

### Non-blocking cross-stream ordering (the mechanism answering "how is H2D-before-compute enforced without host blocking")
Three named, per-cache/per-resource `cudaEvent_t`s recur throughout the file, all created with
`cudaEventDisableTiming` (ordering only, no timing overhead):

**Grouped path** — `grouped_device_resource::completion` (declared moe-cache.cu:3784-3872 struct region):
  - Recorded: after H2D descriptor upload + prefill-bias copy in `make_device_resource` (4881); after
    `finish_decode` (6521); after `cold_reset_grouped_resource`'s memsets (4927); defensively in
    `retire_failed_group_resource` on the failure path (4249); after `finish_prefill_add_id` (8677).
  - Waited on (`cudaStreamWaitEvent`, always with the pattern "only if `completion_stream != this_stream`"):
    `cold_reset_grouped_resource` entry (4903); `prepare_decode`'s cross-stream check (6454);
    `activate_graph_resources` (8143 — the one wait that applies to a whole grouped-dispatch execution at once);
    `prefill_add_id_source` (8634).

**Legacy path** — three events created in `ggml_cuda_moe_cache_init_with_pool` (9308/9320/9333):
  - `compute_done` — recorded by `ggml_cuda_moe_cache_record_compute_locked` (~9479) whenever a caller marks
    "a compute kernel has now read cache slots on this stream"; waited on by the *copy* stream before any
    subsequent write to a slot (`acquire_locked` 9779, `copy_to_staging` 9882, `prepare_split_staging` 9972-9975)
    — a Write-After-Read hazard guard.
  - `stage_done` — recorded on `copy_stream` after a staging batch is enqueued (`copy_to_staging` 9909,
    `prepare_split_staging` 10104); waited on by `compute_stream` before consuming staged data
    (`copy_to_staging` 9910, `prepare_split_staging` non-overlap path 10106, `finish_split_staging` 10130).
  - `handoff_done` — the legacy↔grouped bridge event, see Topic 7.

### Genuine host-blocking calls (all outside the hot path)
- `moe_cache_l2_acquire`, line 535 — `cudaStreamSynchronize(copy_stream)`, only when evicting an L2 slot whose
  prior contents might still be mid-flight as a `cudaMemcpyAsync` source.
- `make_device_resource`, lines 4838/4863/4873/4882 — all four strictly inside error-cleanup branches, never on
  the success path.
- `finish_prefill_add_id`, line 8679 — `cudaStreamSynchronize`, only as a best-effort cleanup when
  `cudaEventRecord` itself already failed (the function still returns failure either way).
- `ggml_cuda_moe_cache_grow_pool`, lines 10191 (`cudaEventSynchronize(compute_done)`) and 10197
  (`cudaStreamSynchronize(copy_stream)`) — **the one deliberate, unavoidable exception**: reallocating the VRAM
  pool means `cudaFree`ing the old allocation, which is unsafe while any in-flight copy or compute kernel might
  still touch it, so the function must fully drain before freeing.
- `ggml_cuda_moe_cache_free` (teardown), lines 9409/9412 — same reasoning, at destruction time.
- `ggml_cuda_moe_cache_init_with_pool`'s one-time `cuStreamWriteValue32` capability probe, line 9276 — executed
  once at cache construction, not per-token.
- `grouped_device_resource`'s destructor — `cudaEventSynchronize(completion)` before `cudaEventDestroy`, again
  only at teardown.
- Test-only helpers (`trailing_padding_zero_for_test`, `device_slot_for_expert_for_test`) do synchronous
  `cudaMemcpy` D2H reads — fine, they're not part of the runtime hot path.

**Is it ever a pure device-side dependency with NO host blocking call at all?** Yes — this is the *normal* case
for every per-token operation in both paths. The specific functions that set up such a dependency with zero
host blocking: `activate_graph_resources` (moe-cache.cu:8143), `prepare_decode` (6454), `finish_decode` (6521,
recording only), `cold_reset_grouped_resource` (4903/4927), `prefill_add_id_source`/`finish_prefill_add_id`
(8634/8677), `ggml_cuda_moe_cache_acquire_locked` (9779/9786), `ggml_cuda_moe_cache_copy_to_staging`
(9882/9909-9910), `ggml_cuda_moe_cache_prepare_split_staging` (9972-9975, plus the `cuStreamWriteValue32`
semaphore-based path described in Topic 8), `ggml_cuda_moe_cache_finish_split_staging` (10130),
`ggml_cuda_moe_cache_prepare_legacy`/`_handoff_grouped` (9451, 9465-9468).

---

## Topic 7 — Legacy vs. Grouped dispatch and the handoff between them

### ggml_cuda_moe_graph_execution::rejects_cached_mmid (moe-cache.cu:3458-3467)
```cpp
return cached && plan_ != nullptr && !legacy && plan_->find(node) == nullptr;
```
Detects a `GGML_OP_MUL_MAT_ID` node whose weight source lives in a CUDA-MoE-cached buffer, where the *current
non-legacy* plan's node hash table doesn't know about this specific node — i.e. "this graph touches
expert-cached weights the cached plan didn't certify," forcing a rebuild rather than trusting a stale plan.

### ggml_cuda_moe_graph_execution::requires_dispatch (moe-cache.cu:3555-3558)
```cpp
return plan_ != nullptr && (n_groups_ != 0 ||
    (plan_->outcome_ == GGML_CUDA_MOE_GRAPH_OUTCOME_ERROR && plan_->coverage_diagnostics_.cached_mmid != 0));
```
Distinguishes "nothing to do" from "something needs handling, but the grouped fast path isn't usable" — the
second clause specifically catches the case where compilation failed (`ERROR`) but the graph still genuinely
contains expert-cached mmid nodes, so a fallback (legacy) path must still run.

### ggml_cuda_moe_cache_prepare_legacy (moe-cache.cu:9446-9458) / ggml_cuda_moe_cache_handoff_grouped (moe-cache.cu:9460-9472)
The actual bidirectional event handshake for a **VRAM pool shared** between the two paths
(`!cache->owns_slot_pool` gates both — this only applies when the legacy cache is *borrowing* a grouped
resource's pool, see `acquire_legacy_cache` below):
- `prepare_legacy` (grouped→legacy handoff): if a `grouped_done` event was supplied,
  `cudaStreamWaitEvent(cache->copy_stream, grouped_done, 0)` (9451) — the legacy copy stream won't start until
  prior grouped work on that pool has completed — then flushes the legacy cache's own bookkeeping
  (`ggml_cuda_moe_cache_clear_slots_locked`).
- `handoff_grouped` (legacy→grouped handoff): `cudaEventRecord(cache->handoff_done, cache->copy_stream)` (9465),
  `cudaStreamWaitEvent(grouped_stream, cache->handoff_done, 0)` (9466), and conditionally also
  `cudaStreamWaitEvent(grouped_stream, cache->compute_done, 0)` (9467-9468) — the grouped stream won't proceed
  until all prior legacy copy-stream *and* legacy compute-consumption work is done.
Both directions are entirely `cudaStreamWaitEvent`-based — no host block anywhere in either function.

### acquire_legacy_cache (moe-cache.cu:5738-6067) — legacy cache-object creation, and where pool-sharing is decided
Entry point for the legacy per-tensor path. If the requested tensor is also a registered bank in the *grouped*
candidate table, it calls `acquire_group_resources_impl`/`make_device_resource` to obtain (or build) that
group's VRAM allocation and **borrows** a slice of it as the legacy cache's `slot_pool_d`
(`ggml_cuda_moe_cache_init_with_pool` called with a non-null `slot_pool_d` + the grouped resource's `completion`
event as `wait_event`) — this is the mechanism letting several legacy per-tensor caches share one group's VRAM
instead of each independently `cudaMalloc`ing. If not registered/borrowable, it builds a fully independent pool.

### probe (moe-cache.cu:6173-6240) — read-only path classifier
A non-mutating dry-run: given up to two observed `(ids, weight)` bank pairs, verifies they resolve to a single
registered candidate group with matching roles, **without** touching `impl_->resources` or building anything —
used by callers to check feasibility before committing to either path.

### begin_graph_dispatch (moe-cache.cu:8208-8546) — the authority/ownership state machine
**Not** where kernels launch. It decides, per semantic group, whether that group's live *authority*
(`GGML_CUDA_MOE_GROUP_AUTHORITY_GROUPED` vs. `_LEGACY`) should change, and if so orchestrates the transition —
a host-side `std::mutex`+`std::condition_variable` barrier-wait for quiescence (`impl_->resource_cv.wait(...)`,
9404-ish — a CPU/host synchronization primitive, **not** a CUDA call) before applying the
`prepare_legacy`/`handoff_grouped`/`cold_reset_grouped_resource` delegate calls described above. **Zero direct
CUDA API calls in this function's own body** — everything CUDA-facing is delegated. The fast/common case (no
authority change needed) skips the barrier entirely and just arms each group's dispatch state.
**What triggers falling back from grouped to legacy**, gathered from `compile_graph_plan`/`bind_graph_plan`
(Topic "everything else") and confirmed by the test suite: a dormant candidate layout, an active LoRA adapter, a
tensor override, incomplete/uncertified mmid coverage, incoherent stream assignment across a grouped group's
readers (`has_coherent_grouped_streams() == false`), or the candidate table simply having zero accepted
groups (grouped candidates disabled entirely) — any of these causes `compile_graph_plan` to classify the
plan's outcome as `PREFILL_LEGACY`/`DECODE_LEGACY`/`ERROR` instead of `DECODE_GROUPED`, which
`begin_graph_dispatch` then honors when computing each group's `desired` authority.

---

## Topic 8 — Split / overlapped staging

**"Split" = dividing a batch of cache misses into multiple sequential "waves"; "overlap" = letting compute begin
consuming an early wave via a device-polled semaphore word before later waves' copies have finished — i.e.
software pipelining across the miss batch within one miss-servicing call, gated on driver/device support for
raw CUDA stream-memory-ops.**

### ggml_cuda_moe_cache_copy_to_staging (moe-cache.cu:9856-9912)
The simpler, non-split sibling: gathers N host sources (a mix of already-hot-resident VRAM slots and
not-yet-resident cold sources) into **one contiguous caller-provided destination buffer** via a run-length-merge
loop that batches physically-adjacent same-kind sources into single `cudaMemcpyAsync` calls — `HostToDevice` for
cold sources, `DeviceToDevice` for already-resident ones (line 9900-9905; the D2D case is only possible because
"resident" here means a hot-pool VRAM slot, not an L2 pinned-host slot — further confirming L2 lives in host
memory). One `stage_done` event covers the whole batch.

### ggml_cuda_moe_cache_prepare_split_staging (moe-cache.cu:9914-10111) — the actual "split" logic
Used specifically when the miss count exceeds the hot pool's own slot capacity
(`n_host_srcs > cache->n_slots`, line 9946). Key mechanics:
- `overlap = (stage_ready != nullptr) && cache->stream_mem_ops_supported` (9950) — opt-in by the caller,
  gated on a one-time device capability probe done at cache-init time.
- Fills as much of the miss set as fits directly into existing hot-pool capacity first (reusing
  `acquire_locked`'s LRU machinery with `use_l2=false, wait_for_compute=false`, 9983-9999), pinning everything
  acquired so it survives the rest of the operation.
- Computes wave size/count (`wave_size = max(n_slots, ceil(n_misses / (stage_ready_capacity-1)))`), then per
  wave: merges contiguous sources, issues either a single `cudaMemcpyBatchAsync` (CUDA ≥ 12.8, with
  `cudaMemcpyFlagPreferOverlapWithCompute` when `overlap`, lines 10074-10091) or a fallback per-run
  `cudaMemcpyAsync` loop (10065-10070, older CUDA/HIP/MUSA), then writes a **per-wave ready semaphore** via the
  raw driver primitive `cuStreamWriteValue32(copy_stream, &stage_ready[wave+1], 1, ...)` (10098-10099) —
  letting a device-side consumer polling `stage_ready[wave+1]` start on that wave's data immediately, without
  waiting for later waves.
- A final `cudaEventRecord(stage_done, copy_stream)` (10104) always happens; in the **non**-overlap case it's
  immediately followed by `cudaStreamWaitEvent(compute_stream, stage_done, 0)` (10106) — in the overlap case the
  caller is expected to rely on the per-wave semaphores instead of this blanket wait.

### ggml_cuda_moe_cache_can_overlap_staging (moe-cache.cu:10113-10118)
One-line predicate: `cache->stream_mem_ops_supported` — the device/driver capability, probed once via a real
`cuStreamWriteValue32` + blocking `cudaStreamSynchronize` at cache-init time (moe-cache.cu:9261-9295), not
re-checked per call.

### ggml_cuda_moe_cache_finish_split_staging (moe-cache.cu:10120-10132)
`cudaStreamWaitEvent(compute_stream, cache->stage_done, 0)` (line 10130) — the mandatory final catch-all wait
that guarantees the *entire* operation (all waves including the last) is complete, regardless of whether the
caller already consumed early waves via the semaphore mechanism.

### ggml_cuda_moe_cache_release_split_slots (moe-cache.cu:10134-10161)
Records `compute_done` on the compute stream (so future copy-stream work knows to wait behind this
consumption), then unpins every slot in the caller's array. No direct memcpy/event calls beyond that recording.

---

## Everything else (brief)

### Graph plan caching & compile/bind (ggml_cgraph structural cache, NOT CUDA Graphs)
- `compile_graph_plan` (moe-cache.cu:6530-7281, the single largest function in the file, ~750 lines) — pure
  CPU-side graph-topology/metadata bookkeeping (zero CUDA calls anywhere in it). Walks the current `ggml_cgraph`
  twice: pass 1 builds a cached-mmid-node inventory/fingerprint; a pure-prefill graph short-circuits to
  `PREFILL_LEGACY`; pass 2 (decode graphs only) validates per-group geometry/capability/route proofs and
  classifies each group's eligibility via a strict priority chain, aggregating into
  `PREFILL_LEGACY`/`DECODE_GROUPED`/`DECODE_LEGACY`/`ERROR`.
- `graph_mmid_inventory_matches` (7283-7312) / `graph_group_witness_matches` (7314-7571) — cheap re-verification
  of a previously-compiled plan against the *current* graph, without a full rescan; failure here is what forces
  a fresh `compile_graph_plan` call.
- `bind_graph_plan` (7573-7772) — the actual cheap-reuse path, binding a validated cached plan into a fresh
  `ggml_cuda_moe_graph_execution` for this call.
- `prepare_graph_execution` (7774-7849) — top-level: try `bind_graph_plan` (reuse) first, else
  `compile_graph_plan` (fresh build).
- `ggml_cuda_moe_graph_plan`/`ggml_cuda_moe_graph_execution` classes (3285-3574) — a fixed-capacity
  (`NODE_TABLE_SIZE=4096`) open-addressing hash table keyed by `ggml_tensor*` pointer identity
  (`ggml_cuda_moe_graph_node_hash`, 3334-3340), letting the plan answer "have I seen this exact node before, and
  with what group/bank/slot binding" in O(1) — the mechanism underlying all of the above reuse logic.
- `graph_resource_fingerprint_locked`/`graph_resource_fingerprint` (7860-8075) — a running FNV-1a-style hash
  over every device pointer/generation counter a bound execution currently references, used by
  `activate_graph_resources` to cheaply detect whether the underlying VRAM resources drifted (reallocated,
  evicted, moved) since the fingerprint was last computed.
- `activate_graph_resources` (8077-8206) — re-validates the fingerprint, then either primes the device clock
  kernel for CUDA-graph-capture-mode dispatch, or rebinds live transaction/bank/auxiliary pointers for
  replay-mode dispatch; the one `cudaStreamWaitEvent` at line 8143 covers the whole execution's cross-stream
  ordering in one shot.
- `prefill_add_id_source`/`finish_prefill_add_id` (8548-8690) — prefill-phase-only helper letting an `ADD_ID`
  graph node read a prefill-resident bias tensor directly by index (bypassing the per-slot cache indirection
  used during decode), with its own completion-event handshake.
- `prepare_graph_group`/`finish_graph_group`/`finish_graph_dispatch`/`end_group_call`/`shutdown` (8692-9013) —
  per-group and per-execution lifecycle bookkeeping around the grouped dispatch state machine; `shutdown`
  implements a single-owner drain-then-teardown protocol using the same mutex/condition-variable pattern as
  `begin_graph_dispatch`'s barrier wait.

### Candidate registry / tensor discovery & validation (moe-cache.cu:832-2696)
A graph-scanning subsystem, entirely separate from the byte-movement mechanics above: it decides **what**
tensors in the current `ggml_cgraph` structurally qualify as a cacheable routed-expert weight group (matching
up/gate/down projection layouts, quantization scale/bias auxiliaries, capability witnesses, execution
certificates) before any of the caching machinery is allowed to touch them. Its main entry point,
`moe_candidate_build_v2` (2450-2679), scans an entire snapshot/graph and populates the `moe_candidate_table`
(reverse-mapped by tensor pointer) that every other function in the file consults via `find_weight`/`get_group`/
`get_bank`/`find_down_group`. Dozens of small validators feed into it (role/layout classification, structural
group validation, capability-witness computation, execution-phase/geometry derivation, route-proof validation) —
low control-flow relevance to the byte-movement/synchronization questions, high relevance to "how does the cache
decide what's eligible at all."

### `impl` struct internals (moe-cache.cu:3738-5180) — grouped_context private state
`struct ggml_cuda_moe_grouped_context::impl` holds essentially all grouped-path mutable state. Notable nested
types: `grouped_device_resource` (3784-3872, the actual per-group VRAM/bookkeeping bundle — device pointers for
weight/auxiliary banks, the admission tables `slot_for_expert`/`expert_for_slot`/`last_used`/`expert_frequency`/
`expert_frequency_epoch`, the device clock, the `plan` buffer, and the `completion` event); `grouped_resource`
(3874-3883, owns a `unique_ptr<grouped_device_resource>` plus dirty/building flags); `resource_build_input`
(3885-3898, host-side staging struct for building a resource); `group_authority_record` (3900-3905, the
legacy/grouped authority + `admission_closed` flag per group — this is what
`admission_closed_for_test` reads); `graph_coverage_record` (3907-3913, CUDA-graph-capture coverage tracking).
Key methods: `capture_resource_input` (4349-4390, snapshots candidate-table data before dropping the table
lock), `make_descriptor`/`make_grouped_resource` (host-only metadata assembly, no CUDA calls),
`descriptor_matches`/`resource_matches_table` (staleness checks), `group_source_mapped`/`decode_eligible`
(alias-ability/layout gate checks before a grouped acquire is attempted), `acquire_group_resources_impl`
(4639-4699, the metadata-cache-hit-or-build orchestrator — VRAM allocation itself is deferred to
`make_device_resource`, called separately under a `building_device` guard), `refresh_group_resource`
(4990-5074+, a reactive — not predictive — incremental-vs-full-rebuild refresh path, triggered only by internal
clock-counter overflow, not by any lookahead signal), `certify_graph_coverage`/`recover_graph_coverage`
(5236-5336, mmid-fingerprint-based coverage registration for CUDA-graph-capture replay validation).

### RAII leases (brief)
`ggml_cuda_moe_group_call_lease` (3582-3642), `ggml_cuda_moe_legacy_operation_lease` (3644-3672),
`ggml_cuda_moe_legacy_cache_lease` (3674-3721) — move-only scope guards releasing their respective resource
(`end_group_call`/`end_legacy_operation`/`release_legacy_cache`) on destruction. `legacy_cache_lease::get()`
returning a raw `ggml_cuda_moe_cache*` is the bridge type letting grouped-context code reach the older simpler
cache struct.

### Cache construction/teardown & the `ggml_cuda_moe_cache` struct (moe-cache.cu:9060-9444)
`struct ggml_cuda_moe_cache` (9060-9148) — the legacy per-tensor cache's full field layout (VRAM pool, the three
named events, per-slot LRU/pin/hit-count vectors, the nested `moe_cache_l2 l2` member, expert-access telemetry
vectors). `ggml_cuda_moe_cache_init_with_pool` (9170-9378) is the real constructor: `cudaMalloc`s the pool (only
if `slot_pool_d` wasn't supplied by a caller sharing a grouped resource's pool), creates a dedicated
`copy_stream` (`cudaStreamNonBlocking`), probes `cuStreamWriteValue32` support once, and creates the three
events. No pinned-host allocation happens here at all — the cold-tier host memory is always someone else's
(the buffer-type's) allocation. `ggml_cuda_moe_cache_free` (9393-9431) tears down with an explicit
sync-then-destroy sequence (the one legitimate teardown-time host block).

### Stats/telemetry/logging (very brief — pure reporting, no control-flow effect)
Dozens of small `moe_cache_*`/`ggml_cuda_moe_*` functions (roughly moe-cache.cu:10331-10985, 10812-10985,
11174-11402) aggregate and format hit/miss/eviction/H2D-byte/mincore-residency/expert-reuse-distance statistics
for periodic `GGML_LOG` dumps (`moe_cache_log_telemetry`, 11174-11367, the largest of these at ~195 lines).
None of them participate in any cache-hit/miss/admission decision.

### mmap / page-fault instrumentation (moe-cache.cu:137-328, 330-388)
`moe_cache_mm_sample_mincore` (periodic `mincore()` sampling of registered mmap ranges) and `/proc/self/stat`/
`/proc/vmstat`-derived minor/major-fault and swap-activity snapshots — pure diagnostics answering "is the cold
tier's mmap'd data actually resident right now," feeding only the `moe_cache_mm_stats` telemetry struct.

### Dead/stub functions found in this revision (notable — do not assume these do anything)
`ggml_cuda_moe_cache_get_or_create` (11079-11088, unconditionally returns `nullptr`, comment: "dispatch hook now
uses the per-tensor variant"); `ggml_cuda_moe_cache_free_all` (11090-11092, empty body);
`ggml_backend_cuda_moe_observe_expert_tensor`/`_reset_expert_size_observation` (11131-11145, no-ops);
`ggml_backend_cuda_moe_preallocate_pools`/`_preallocate_pool` (11147-11172, no-ops, superseded-by-lazy-alloc
comments); **`ggml_backend_cuda_moe_prefetch_experts` (11152-11166, no-op — see Topic 5)**.

### Test-only hooks (one-line mentions only)
`admission_closed_for_test`, `has_device_resource_for_test`, `device_slot_for_expert_for_test` (does a real
synchronous D2H `cudaMemcpy`), `device_bank_data_for_test`, `prefill_auxiliary_ordering_for_test`,
`set_prefill_resident_budget_for_test`, `device_resource_complete_for_test`, `graph_clock_active_for_test`,
`legacy_backing_count_for_test`, `fail_borrowed_cache_init_after_probe_for_test` (simulates a legacy-cache-init
failure immediately after a successful borrowed-resource probe), `poison_split_staging_for_test`/
`split_staging_poison_calls_for_test` (forces the next N `prepare_split_staging` calls to fail), plus a dozen
more — all under moe-cache.cu:5336-5633, each a one-liner reading or forcing some piece of internal state for
unit tests.

---

## moe-cache.cuh — Header / API Surface

Reference-style listing (field names only, not explained) — see the two big classes below for full method
tables. **No `#define` constants or the `moe_grouped_decode_plan`/`moe_grouped_plan_status` types are declared
in this header** — they're defined in the .cu file itself (Topic 3/`Plan Buffer Layout Helpers`).

### ggml_cuda_moe_graph_span (moe-cache.cuh:34-37)
`{ begin, end }` — byte-range for graph-node overlap checks (companion: `ggml_cuda_moe_graph_span_bounds`,
`ggml_cuda_moe_graph_spans_overlap`).

### Candidate enums (moe-cache.cuh:68-102)
`ggml_cuda_moe_candidate_rejection` (17 values: `INVALID_ABI`…`GENERATION_EXHAUSTED`),
`ggml_cuda_moe_candidate_encoding` (`PLAIN`/`NVFP4_COMPOUND`), `ggml_cuda_moe_candidate_movement`
(`SLOT_BOUND`/`PERMANENT_CANDIDATE`), `ggml_cuda_moe_candidate_index_mode` (bitflags:
`ORIGINAL_DIRECT`/`GROUP_SLOT_DIRECT`/`ORIGINAL_SOURCE_MAP`).

### ggml_cuda_moe_candidate_registry_state (moe-cache.cuh:104-114)
`generation, logical_signature, slot_bound_bytes, permanent_candidate_bytes, n_slots, n_groups, n_weights,
accepted, rejection`.

### ggml_cuda_moe_candidate_bank_info (moe-cache.cuh:116-129)
`generation, byte_extent, expert_stride, tensor, source_data, group_index, role, type, source_flags, encoding,
movement, index_modes`.

### ggml_cuda_moe_candidate_group_key (moe-cache.cuh:131-134)
`{ generation, group_index }`.

### ggml_cuda_moe_candidate_group_info (moe-cache.cuh:136-145)
`key, down, layout, domain, semantic_group_index, flags, n_banks, n_slots`.

### ggml_cuda_moe_candidate_probe_bank/input/result (moe-cache.cuh:147-165)
`probe_bank{weight, ids, scale, bias, expected_role}`; `probe_input{banks[2], expected_generation, n_banks,
exact_auxiliaries}`; `probe_result{key, roles[2]}`.

### ggml_cuda_moe_grouped_acquisition / _transaction (moe-cache.cuh:167-175)
`grouped_acquisition{candidate, resource_generation}`; `grouped_transaction{acquisition, transaction_token}`.

### ggml_cuda_moe_legacy_acquisition (moe-cache.cuh:177-187)
`owner, tensor, candidate_generation, authority_epoch, group_authority_epoch, group_index, role, n_slots,
registered_source`.

### ggml_cuda_moe_legacy_operation_lease / legacy_cache_lease (classes, moe-cache.cuh:189-230)
RAII, move-only — see "RAII leases" above for behavior.

### ggml_cuda_moe_ids_signature (moe-cache.cuh:232-239)
`tensor, data, buffer, ne[GGML_MAX_DIMS], nb[GGML_MAX_DIMS], type` — identity fingerprint of a routing-ids
tensor.

### ggml_cuda_moe_complete_group_key (moe-cache.cuh:241-247)
`candidate, ids, execution_semantic_key, layout, n_banks` — the full key type `prepare_decode` takes.

### ggml_cuda_moe_grouped_decode_result (enum, moe-cache.cuh:249-253)
`FALLBACK, READY, ERROR`.

### ggml_cuda_moe_grouped_decode_bank / decode_acquisition (moe-cache.cuh:255-274)
`decode_bank{tensor, data, bank_index, role, type}`; `decode_acquisition{transaction, remapped_ids,
banks[MAX_BANKS], auxiliary_tensors[3], auxiliary_data[3], auxiliary_roles[3], layout, n_banks, n_slots,
n_auxiliary_shadows}` — the out-parameter struct `prepare_decode` fills.

### ggml_cuda_moe_group_authority (enum, moe-cache.cuh:276-279)
`LEGACY, GROUPED`.

### ggml_cuda_moe_graph_outcome (enum, moe-cache.cuh:281-286)
`PREFILL_LEGACY, DECODE_GROUPED, DECODE_LEGACY, ERROR`.

### ggml_cuda_moe_group_call_lease (class, moe-cache.cuh:288-314)
RAII authority token — see "RAII leases" above.

### ggml_cuda_moe_graph_group_state (enum, moe-cache.cuh:316-322)
`WHOLE_LEGACY, GROUPED_ARMED, GROUPED_ACTIVE, GROUPED_REPLAY, FINISHED`.

### ggml_cuda_moe_graph_dispatch_mode (enum, moe-cache.cuh:324-329)
`LEGACY, DIRECT, CAPTURE, REPLAY`.

### ggml_cuda_moe_graph_capability_witness (moe-cache.cuh:331-363)
`tensor, source_data, byte_extent, expert_stride, source_ne/nb, grouped_ne/nb, n_tokens, n_experts, smpbo,
device, cc, warp_size, role, source_type, source_flags, input_type, output_type, phase, mapping, row_semantics,
consumer, reason, equivalence_reason, top_k, n_rows, n_routes, row_stride, n_slots, use_mmq`.

### ggml_cuda_moe_graph_group_dispatch (moe-cache.cuh:365-382)
`key, authority, transaction, capabilities, first_reader, last_reader, remapped_ids, bank_data[MAX_BANKS],
auxiliary_tensors[3], auxiliary_data[3], auxiliary_roles[3], stream, state, n_slots, n_auxiliary_shadows,
defer_completion`.

### ggml_cuda_moe_graph_binding (moe-cache.cuh:386-391)
`key, role, bank_index, slot_index`.

### ggml_cuda_moe_grouped_debug_telemetry / legacy_debug_telemetry (moe-cache.cuh:399-428)
Grouped: `registered, covered, plan_calls, plan_compiles, plan_reuses, calls, ready(_min/_max),
completed(_min/_max), admitted_banks, fallback, rollback, prepare_error, finish_error, h2d_banks, h2d_bytes`.
Legacy: `ops, staged_ops, split_staged_ops, overflow_ops, unique_experts_max, ids_cache_hits`.

### ggml_cuda_moe_graph_coverage_reason (enum) / coverage_diagnostics (moe-cache.cuh:436-467)
12 reason values (`REGISTERED`…`INCOMPLETE`) plus `coverage_diagnostics{cached_mmid, manifest_version,
counts[REASON_COUNT], first_source/node_index/group_index/bank_index/role/status/layout/domain/flags/
group_flags/rejection[...]}`.

### ggml_cuda_moe_graph_plan (class, moe-cache.cuh:469-667)
Public: `ggml_cuda_moe_graph_plan(); uint32_t size() const; uint64_t registry_generation() const; uint64_t
graph_uid() const; int32_t graph_node_count() const; ggml_cuda_moe_graph_outcome outcome() const; bool
has_certified_complete_mmid_inventory() const; const ggml_cuda_moe_graph_coverage_diagnostics &
coverage_diagnostics() const;` (private: `reset()`, `insert(...)`, `find(...)` — node hash table,
`NODE_TABLE_SIZE=4096`). `static_assert(sizeof(...) <= 128*1024)`.

### ggml_cuda_moe_graph_execution (class, moe-cache.cuh:671-705)
Public: ctor/dtor, `find`, `rejects_cached_mmid`, `find_group`, `find_authority`, `resolve_streams`,
`has_stream_grouped_candidate`, `has_coherent_grouped_streams`, `requires_dispatch`,
`has_prefill_resident_witnesses`, `outcome`, `dispatch_mode`, `size` (private: `reset`, `retain`).

### ggml_cuda_moe_grouped_resource_info / grouped_bank_descriptor (moe-cache.cuh:707-734)
`resource_info{acquisition, down, layout, n_slots, n_banks, transaction_active}`;
`bank_descriptor{tensor, buffer, buft, source_data, buffer_base, buffer_size, data_offset, byte_extent,
expert_stride, alignment, ne[], nb[], role, type, encoding, movement, index_modes}`.

### ggml_cuda_moe_grouped_context (class, moe-cache.cuh:738-921) — MAIN CLASS
Single field: `std::unique_ptr<impl> impl_` (all real state lives in the .cu-file-private `impl` struct). Public
methods (implemented across the whole .cu file — see the line-numbered sections above for each):
`replace()` (v1/v2 candidate snapshot ingestion), `state`, `find_down_group[_key]`, `find_weight`, `get_group`,
`get_bank`, `probe`, `acquire_group_resources`, `begin/end_group_transaction`, `get_group_resources`,
`get_group_resource_bank`, `begin_legacy_operation`, `acquire_legacy_cache`, `prefetch_legacy_siblings`,
`record_legacy_op`, `log_and_reset_legacy_stats` (static), `certify_graph_coverage`, `recover_graph_coverage`,
`compile_graph_plan`, `bind_graph_plan`, `prepare_graph_execution`, `graph_resource_fingerprint`,
`activate_graph_resources`, `begin_graph_dispatch` (two overloads), `prepare_graph_group`, `finish_graph_group`,
`finish_graph_dispatch`, `prefill_add_id_source`, `finish_prefill_add_id`, `prepare_decode`, `finish_decode`,
`shutdown`. Private: ~25 `*_for_test` accessors plus `graph_mmid_inventory_matches`,
`graph_group_witness_matches`, `graph_resource_fingerprint_locked`, `end_group_call`, `end_legacy_operation`,
`release_legacy_cache`, and the `impl` pimpl struct itself.

### Free test-hook functions & closing comment (moe-cache.cuh:923-963)
`ggml_cuda_moe_grouped_context_for_test`, `ggml_cuda_moe_ids_cache_count_for_test`,
`ggml_cuda_graph_capture_state_query_for_test`, `ggml_cuda_moe_execution_semantic_key`,
`ggml_cuda_moe_take_split_staging_poison_for_test`, `ggml_backend_cuda_moe_candidate_replace_v1/v2`.

### Public extern "C" API (moe-cache.cuh:965-1120) — quick reference
`ggml_cuda_moe_cache_init/_free/_acquire/_release_slots/_copy_to_staging/_prepare_split_staging/
_trailing_padding_bytes_for_test/_trailing_padding_zero_for_test/_can_overlap_staging/_finish_split_staging/
_release_split_slots/_record_op_stats/_grow_pool/_slot_ptr/_slot_size_bytes/_n_slots/_copy_stream/_mark_used/
_stats/_reset_stats/_get_or_create (deprecated stub)/_free_all`, plus the `ggml_backend_cuda_moe_*` runtime
tuning-knob declarations (`set/get_cache_slots`, `set/get_l2_pinned_cache_size`, `set/get_debug_mm`,
`observe_expert_tensor`, `reset_expert_size_observation`, `preallocate_pools/_pool`, `prefetch_experts`,
`log_and_reset_stats`).

---

## Test file confirmation notes (tests/test-moe-cache.cpp)

Targeted grep+read only (12,466 lines, not read in full); confirms/refines several conclusions above.

### 1. Admission / eviction policy — frequency-weighted-LRU, confirmed against a host reference model
`test_active_grouped_q4k_eviction_refill_case` (test-moe-cache.cpp:6578-6669) maintains a host-side shadow model
mirroring the device's real per-expert `frequency`/`frequency_epoch` counters and per-slot `last_used` clock
(gated by `GGML_CUDA_MOE_FREQUENCY` env var), and asserts the real device eviction choice (via
`device_slot_for_expert_for_test`) matches: lowest decayed frequency, tie-break oldest `last_used`, tie-break
lowest slot index — exactly the `moe_grouped_warp_min` comparator order.

### 2. Cache hit/miss servicing & whether a compute kernel touches host memory directly — directly confirms Topic 4
No test asserts the exact `device_alias`/`cudaHostGetDevicePointer` mechanism by name (that's internal), but
`test_active_grouped_materialization_eligibility` (7873-7968, see item 5) confirms functional correctness of the
grouped path against both a pinned and an mmap-backed source, and no test anywhere constructs a scenario where a
compute kernel reads directly from an unregistered/pageable pointer.

### 3. Split / overlapped staging — confirms genuinely asynchronous, non-blocking behavior
`poison_split_staging_for_test(3)` (10067-10087, 12301-12446) forces the next 3 `prepare_split_staging` calls to
fail; the test asserts overall correctness is preserved and the poison-call counter correctly drains to 0 —
i.e. the caller degrades/retries gracefully under injected staging failures rather than corrupting cache state.

### 4. Legacy vs grouped dispatch & fallback triggers
`test_cached_mmid_fusion_decline` (9231-9288) disables grouped candidates entirely (0 accepted) and asserts
every MoE op falls to the legacy path (`active_grouped_legacy_op_count == 2 * cached_weights.size()`).
`requires_dispatch()`/`rejects_cached_mmid()` are exercised together (1322-1394, 4455-4558, 8035-8191) for
dormant-layout, active-LoRA, tensor-override, and incomplete-coverage scenarios — matching the coverage-reason
enum and `compile_graph_plan`'s outcome classification described in Topic 7.
`resolve_streams`/`has_coherent_grouped_streams` (3273-3321): a graph resolved across two different streams
yields `!has_coherent_grouped_streams()`, forcing `GGML_CUDA_MOE_GRAPH_DISPATCH_LEGACY`.
`prepare_decode(...) == GGML_CUDA_MOE_GROUPED_DECODE_FALLBACK` is asserted for a stale group key (11176-11303)
and after a failed background maintenance rebuild (11588-11716).

### 5. mmap / pinned / cached-buffer / is_host — directly confirms Topic 1
`test_active_grouped_materialization_eligibility` (7873-7968) runs the **same** grouped-decode scenario twice:
once against the real `ggml_backend_cuda_moe_cached_buffer_type()` (genuinely `cudaMallocHost`-pinned) and once
against a synthetic `file_mmap_cached_buffer_type()` (4764-4827, wraps `mmap(...MAP_SHARED...)` on a tmpfile,
`is_host` hardcoded false), asserting numerically identical output (`pinned_output == expected &&
pageable_output == expected`) — confirming both cold-tier flavors are real, both are supported, and both
produce correct grouped-decode results. It also asserts legacy-cache acquisition is refused once grouped device
resources already exist for that tensor (pool-sharing exclusivity).

### 6. Test-only hooks (behavior they simulate)
`admission_closed_for_test` — polls until a group's device-side admission window has closed, before asserting
final cache state. `fail_borrowed_cache_init_after_probe_for_test` — forces the very next legacy-cache lazy-init
to fail immediately *after* a successful borrowed-resource probe, verifying the device slab's payload is left
untouched and a retry succeeds cleanly. `poison_split_staging_for_test` — see item 3.

### 7. L2 tier — real but easy-to-miss test coverage (correction: not zero)
A case-insensitive text grep for the literal string `l2` across the 12,466-line test file returns zero matches,
which earlier passes over this document took as "no test coverage." That is wrong: `test_owner_legacy_cache`
calls `context->prefetch_legacy_siblings(lease, experts, n_experts, /*use_l2=*/true, /*is_decode=*/true)` twice
(test-moe-cache.cpp:10956 and :11011) — `use_l2` is passed as a **positional literal `true`**, so it is
functionally invisible to a text search for "l2". This genuinely exercises the L2-enabled path end to end
(`prefetch_legacy_siblings` -> `acquire_legacy_cache`/`ggml_cuda_moe_cache_acquire` -> `moe_cache_l2_source` ->
`moe_cache_l2_acquire`) and the test's own hit/miss assertions (10957-10961) hold under it. Verified directly
against the source for this correction (not re-derived from a sub-agent's claim). The conclusions in Topic 2
above are therefore corroborated by a real test path, not resting purely on reading `moe-cache.cu` alone —
though no test asserts L2-*specific* telemetry (hit/miss/fill counters) or exercises the `source_is_mmap=false`
(L2-bypassed) vs. `true` (L2-engaged) distinction explicitly by name, so that finer-grained behavior is still
only confirmed by direct source reading.
