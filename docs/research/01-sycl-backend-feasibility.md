# SYCL backend feasibility: porting the CUDA MoE expert cache to `ggml-sycl`

Research date: **2026-09-09**. Question: for each CUDA-specific mechanism in
`GenerelSchwerz/llama.cpp`'s `ggml/src/ggml-cuda/moe-cache.cu`, does an
equivalent exist in llama.cpp's SYCL backend, and is `ggml-sycl` structurally
symmetric enough with `ggml-cuda` to receive a port?

## Method and evidence grade

Primary evidence is a **local checkout of upstream `ggml-org/llama.cpp` at
`master`**, commit `22397c31a00e78f55ae556c41fc78b717c5911bd` (2026-09-09).
Line numbers below are against that commit and will drift; the symbol names
will not. Every file path is relative to the repo root and readable at
`https://github.com/ggml-org/llama.cpp/blob/master/<path>`.

Two labels are used throughout:

- **[CONFIRMED]** — read directly out of that source tree (or out of a cited
  upstream issue/PR).
- **[INFERRED]** — my reasoning on top of confirmed facts. Treat as a
  hypothesis to validate, not a finding.

The fork's own `moe-cache.cu` was **not** read for this document (the brief
described it; it is ~11,400 lines and lives outside this checkout). Claims
about what the fork does are taken from `docs/00-background.md` in this repo
and are therefore second-hand.

## Summary

**`ggml-sycl` is structurally symmetric enough with `ggml-cuda` to receive this
port.** Nothing is hard-blocked. The abstraction the whole mechanism depends on
— pluggable buffer types plus a single `mul_mat_id` chokepoint — is fully
implemented, and `ggml-sycl` additionally has a *fused, device-side-IDs* MoE
decode path (`mul_mat_vec_q_moe`) that is a better insertion point than CUDA's.

Five things worth knowing before reading further:

1. **The riskiest component is graph replay, and it is less blocked than
   expected.** SYCL graphs exist in `ggml-sycl` today, but are hard-disabled
   for every model containing `GGML_OP_MUL_MAT_ID`. The fix is an **open,
   39-line upstream PR ([#25089](https://github.com/ggml-org/llama.cpp/pull/25089))
   already validated on an Arc Pro B70 with a Qwen MoE**, stalled on a rebase,
   not on any objection. §5.4
2. **Two hard design constraints from SYCL's memory model**, both cheap to
   design in and painful to retrofit: pinned host allocations must be **≤2 GiB
   each** (above that, copy/compute overlap is lost), and there is **no
   `cudaHostRegister` equivalent** — the cold tier must be allocated, not
   pinned-in-place, so the fork's mmap-backed L2 tier has no analogue. §2.3–2.4
3. **The OOM hazard is opt-in and localised.** The implicit-migration path is
   `GGML_SYCL_USM_SYSTEM` (default **0**), documented as experimental and
   naming Battlemage explicitly. `sycl::malloc_host` — the mechanism this
   project actually wants — is a categorically different, explicit one, and is
   on by default. Leave the one env var alone. §2.5
4. **`ggml-sycl` never constructs a queue of its own**; all eight nominal
   "streams" alias one in-order queue. A dedicated copy queue is genuinely new
   code and **must share the compute queue's `sycl::context`** or its copies
   silently no-op. §3.3
5. **The backend is actively maintained** (121 commits in five months, level
   with CUDA's 142 and Vulkan's 120, and accelerating), but carries a
   substantial open reliability backlog concentrated on **Arc Pro B70 /
   Battlemage** — including a still-open correctness regression on hybrid
   SSM+MoE architectures such as `qwen3next`. §6.3

**Recommended first actions**, in order, all cheap: probe whether the B70
reports `aspect::ext_oneapi_graph`; rebase #25089 and measure whether graph
capture alone moves decode throughput at all; and — before any of it — settle
the two questions this document cannot answer, namely whether the chosen quant
already fits in 32 GB (in which case the cache costs performance) and whether
issue #24168 blocks the target architecture outright.

---

## 1. Buffer-type / dispatch parity

### 1.1 The buffer-type abstraction is genuinely symmetric

**[CONFIRMED]** `ggml-sycl` implements the full `ggml_backend_buffer_type_i` /
`ggml_backend_buffer_i` vtable pair exactly as `ggml-cuda` does, and it
implements **three** buffer types, not one:

| Buffer type | `ggml-sycl` symbol | File / line |
|---|---|---|
| Device | `ggml_backend_sycl_buffer_type(int device)` | `ggml/src/ggml-sycl/ggml-sycl.cpp:1030` |
| Split (multi-GPU by rows) | `ggml_backend_sycl_split_buffer_type(const float *)` | `ggml/src/ggml-sycl/ggml-sycl.cpp:1466` |
| Pinned host | `ggml_backend_sycl_host_buffer_type(void)` | `ggml/src/ggml-sycl/ggml-sycl.cpp:1583` |

All three are exported in the public header `ggml/include/ggml-sycl.h`. The
device buffer-type interface is at `ggml-sycl.cpp:1021-1028`:

```c
static const ggml_backend_buffer_type_i ggml_backend_sycl_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_sycl_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_sycl_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_sycl_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_sycl_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_sycl_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};
```

Compare `ggml/src/ggml-cuda/ggml-cuda.cu:927-934` — same six slots, same
ordering, same `NULL`-means-default convention. **The abstraction is not
CUDA-shaped-with-a-SYCL-veneer; it is the shared ggml contract and ggml-sycl
implements it properly.** The buffer vtable
(`ggml_backend_sycl_buffer_interface`, `ggml-sycl.cpp:897`) likewise carries
`init_tensor`, `set_tensor`, `get_tensor`, `cpy_tensor`, `clear`,
`memset_tensor`, and `reset` — including `init_tensor`
(`ggml-sycl.cpp:617`), which is the hook the CUDA fork would use to attach
per-tensor cache metadata via `tensor->extra`.

**[CONFIRMED]** `ggml_tensor_extra_gpu` already exists in
`ggml/src/ggml-sycl/common.hpp:317-323` with `data_device[]`, per-device/
per-stream `events[][]`, and an `optimized_feature` sub-struct — i.e. the
per-tensor side-channel a cache needs is established practice here, and the
split buffer type already uses it.

### 1.2 The gotcha: buffer-type identity is compared by function pointer

**[CONFIRMED]** Both backends identify "is this buffer mine?" by comparing the
`get_name` function pointer, not by any registry:

- `ggml-sycl.cpp:594-596` — `ggml_backend_buffer_is_sycl()` →
  `buffer->buft->iface.get_name == ggml_backend_sycl_buffer_type_get_name`
- `ggml-sycl.cpp:1405-1407` — same pattern for the split buft
- `ggml-sycl.cpp:6667-6674` — `ggml_backend_sycl_device_supports_buft()`
  **returns `false` outright** for any buft whose `get_name` differs
- `ggml-cuda.cu:879-881` — `ggml_backend_buft_is_cuda()`, identical pattern

Consequence for a MoE-cache buffer type: a naively-added
`ggml_backend_sycl_moe_cache_buffer_type` with its own `get_name` will be
rejected by `supports_buft`, so the scheduler will never assign expert tensors
to it, and `cpy_tensor` (`ggml-sycl.cpp:773-780`) will not recognise it as a
SYCL buffer. **Three call sites minimum must be widened.** This is *not* a
SYCL deficiency — the CUDA fork faces exactly the same three-site problem
against `ggml_backend_buft_is_cuda` / `ggml_backend_cuda_device_supports_buft`
— so whatever the fork did there transfers.

**[CONFIRMED]** One extra SYCL-only tripwire: `ggml-sycl.cpp:5941`, inside an
`#ifndef NDEBUG` block in `ggml_backend_sycl_graph_compute_impl`, asserts

```c
assert(node->buffer->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device));
```

for every node and every source. A custom buft trips this in debug builds.
`ggml-cuda` has no equivalent assert. Trivial to fix; easy to lose an evening
to if unexpected.

### 1.3 The dispatch chokepoint exists and is structurally the same

**[CONFIRMED]** `ggml_sycl_mul_mat_id(ggml_backend_sycl_context &, ggml_tensor * dst)`
at `ggml/src/ggml-sycl/ggml-sycl.cpp:5077` is reached from exactly one place —
`case GGML_OP_MUL_MAT_ID:` at `ggml-sycl.cpp:5561-5565` inside
`ggml_sycl_compute_forward`. That is the single interception point the fork's
`ggml_cuda_mul_mat_id` diff corresponds to
(`ggml-cuda.cu:1902`, dispatched at `ggml-cuda.cu:2254-2255`).

The two functions are near line-for-line analogues: same
`src0`/`src1`/`ids = dst->src[2]` layout, same `ne12 == 1` fast-path test,
same fallback that copies `ids` to host and loops per-expert with stack copies
of the tensor descriptors.

**[CONFIRMED — and the most important structural finding in this section]**
`ggml-sycl` has a **fused MoE token-generation fast path** that keeps the
router IDs on-device:

- `ggml_sycl_mul_mat_id_mmvq_fused()` — `ggml-sycl.cpp:4976-5034`, tried first
  at `ggml-sycl.cpp:5092-5096` whenever `ne12 == 1`
- it quantizes `src1` to Q8_1 and calls `ggml_sycl_mul_mat_vec_q_id()` or
  `ggml_sycl_mul_mat_vec_q_id_reorder()` (`ggml/src/ggml-sycl/mmvq.hpp:29`
  and `:46`), passing `ids_dev` as a **device** `const int32_t *`
- **no host readback, no `queue::wait()` on this path**

The actual kernel, `mul_mat_vec_q_moe` at
`ggml/src/ggml-sycl/mmvq.cpp:2704-2751`, contains the exact line a slot-pool
indirection would replace:

```c
const int expert_idx = item_ct1.get_group(1);
const int i02        = ids_dev[expert_idx];
const char * vx = (const char *) vx_base + (size_t) i02 * expert_weight_stride;
```

**[INFERRED]** The minimal cache hook is therefore: replace
`vx_base + i02 * expert_weight_stride` with a lookup into a device-resident
slot table (`slot_base + slot_of[i02] * slot_stride`, or a plain
`const void * const * expert_ptr` array). One kernel line, plus a second
kernel that maintains `slot_of[]`. The `expert_weight_stride` parameter is
already threaded through the whole call chain
(`mmvq.hpp:38`, `mmvq.hpp:55`), so the plumbing for a per-expert stride exists.

Note the reorder variant: `opt_for_reorder_id()` (`ggml-sycl.cpp:4999`) lazily
rewrites Q4_K expert weights into a per-expert SoA layout in place, and
`mmvq.hpp:44` documents that "each expert slice (stride `expert_weight_stride`
== `src0->nb[2]`) is a self-contained reorder/SoA layout." **[INFERRED]** That
self-containedness is exactly what a slab cache needs — an expert slab is
relocatable without fixing up cross-expert references — but it also means the
reorder pass and the cache must agree about when reordering happens (reorder
the host-resident cold copy once at load, not per admission).

### 1.4 Verdict — buffer-type / dispatch parity

**PORTABLE.** The abstraction is backend-symmetric in practice, all three
buffer types exist, `tensor->extra` exists, and the single `mul_mat_id`
chokepoint exists with a device-side-IDs fast path that is arguably a *better*
insertion point than CUDA's. Budget for widening three identity checks and one
debug assert.

---

## 2. Pinned host memory approach

### 2.1 What ggml-sycl uses: `sycl::malloc_host` USM, and it is the default

**[CONFIRMED]** `ggml-sycl.cpp:1522-1538`:

```c
//host pinned memory
static void * ggml_backend_sycl_host_malloc(size_t size) {
    ...
    // USM host memory is page-locked and device-accessible by construction
    auto & q = dpct::dev_mgr::instance().get_device(0).default_queue();
    ptr = sycl::malloc_host(size, q, sycl::property_list{});
```

This is the direct `cudaMallocHost` analogue. It is **on by default**:
`g_ggml_sycl_enable_host_pinned_mem = 1` (`ggml-sycl.cpp:112`), overridable via
`GGML_SYCL_ENABLE_HOST_PINNED_MEM` (`ggml-sycl.cpp:362`). When disabled the
host buffer type falls back to plain `aligned_alloc` (`ggml-sycl.cpp:1553`).

**[CONFIRMED]** There is also a **pooled** pinned-host allocator:
`ggml_sycl_pool_host` (`ggml-sycl.cpp:~1860-1935`), reachable via
`ggml_backend_sycl_context::host_pool(int device)`
(`common.hpp:466-471`), which recycles `sycl::malloc_host` blocks. Precedent
for a managed pinned-host tier already exists in-tree.

### 2.2 Precedent for a kernel reading host memory directly

**[CONFIRMED]** `sycl_reorder_temp_buffer` (`ggml-sycl.cpp:4084-4120`) falls
back to `sycl::malloc_host` when device allocation fails, and then **runs the
reorder kernel directly against that host pointer over PCIe**:

```c
#ifdef GGML_SYCL_HOST_MEM_FALLBACK
    if (!ptr) {
        ptr = sycl::malloc_host(size, *stream);
        ...
// Device access to host memory requires Linux kernel 6.8+ (Ubuntu 26.04+).
```

Documented as a build flag in `docs/backend/SYCL.md` (`GGML_SYCL_HOST_MEM_FALLBACK`,
ON by default): *"Allow host memory fallback when device memory is full during
quantized weight reorder. Enables inference to continue at reduced speed
(reading over PCIe) instead of failing. Requires Linux kernel 6.8+."*

**[INFERRED — and this is the single most useful architectural consequence in
this document]** Because a SYCL kernel can dereference a `sycl::malloc_host`
pointer, the CUDA fork's device-side admission planner can be ported as a
**device-side gather kernel** rather than a host-orchestrated copy: a kernel
reads `ids_dev`, decides admission, and *itself* copies the needed slab bytes
from host USM into the VRAM slot pool. That removes the host round-trip
entirely, needs no queue-level `memcpy` whose arguments depend on device
state, and — critically for §5 — is a plain kernel launch, which is legal
inside a recorded SYCL graph in a way a data-dependent `queue::memcpy` is not.
Whether a hand-rolled gather kernel reaches the same PCIe throughput as the
copy engine driving `queue::memcpy` is an **open perf question** (§8).

### 2.3 The asymmetry: no `cudaHostRegister` equivalent

**[CONFIRMED]** `ggml-cuda` exposes host *registration* — pinning memory it did
not allocate:

- `ggml-cuda.cu:4830-4836`: `ggml_backend_cuda_register_host_buffer()` →
  `cudaHostRegister(buffer, size, cudaHostRegisterPortable | cudaHostRegisterReadOnly)`
- exported via `get_proc_address` at `ggml-cuda.cu:5681-5685`

`ggml-sycl` explicitly does not, and says so twice:

- `ggml-sycl.cpp:7044-7046`: `// SYCL doesn't support registering host memory, left here for reference`
- `ggml/include/ggml-sycl.h`: the two API declarations are present but
  commented out with the same note.

**Consequence:** the cold expert tier must be **allocated** through
`sycl::malloc_host` up front; you cannot `mmap` the GGUF and then pin it in
place. **[INFERRED]** This maps onto the fork's own flag semantics cleanly —
`--load-mode none` (no mmap, cold experts in host RAM) is already the mode the
fork *requires* for its grouped-decode fast path, and `--load-mode mmap` is
already documented as disabling grouped decode. So the SYCL limitation
forecloses a mode the fast path had already foreclosed. The fork's
`--moe-expert-cache-l2-pinned-mb` (an mmap-only pinned L2) has **no clean SYCL
analogue** and should be considered out of scope.

### 2.4 The >2 GiB cliff

**[CONFIRMED]** `docs/backend/SYCL.md`, runtime env-var table, entry
`GGML_SYCL_HOST_PINNED_MEM_2G`:

> *"Limit the max memory allocation to be no more than 2GB when enable host
> pinned memory. **USM allocations above 2 GiB take the relaxed/large-allocation
> path, which serializes H2D copies with compute and prevents copy/compute
> overlap.**"*

The corresponding code is `ggml_backend_sycl_host_buffer_type_get_max_size`
(`ggml-sycl.cpp:1568-1581`), clamping to 2 GiB when
`g_ggml_sycl_host_pinned_mem_2g` is set (default `0`, i.e. **unclamped** —
`ggml-sycl.cpp:113`, `:365`).

**[INFERRED]** For a MoE cache whose whole value proposition is overlapping
PCIe admission with compute, this is a hard design constraint: **the cold
expert tier must be chunked into `sycl::malloc_host` allocations of ≤2 GiB
each.** A ~125B-parameter expert stack is tens of GiB, so this means dozens of
allocations and a slab→(allocation, offset) map. Cheap to design in from the
start, painful to retrofit. Note also the related
`UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS` env var documented in the same table
for *device* allocations >4 GiB.

### 2.5 Implicit migration — the OOM-incident hazard, and where it lives

This is the item `docs/00-background.md` §3 flags as the project's single most
avoidable failure mode. The good news is that the dangerous path is **opt-in
and off by default**, and it is clearly localised.

**[CONFIRMED]** `check_usm_system()` (`ggml-sycl.cpp:926-935`) and the branch
it guards in `ggml_backend_sycl_buffer_type_alloc_buffer`
(`ggml-sycl.cpp:962-978`):

```c
bool use_usm_system = g_ggml_sycl_usm_system && size >= ((size_t)4 * MEM_SIZE_1G);
...
if (use_usm_system) {
    GGML_SYCL_DEBUG("[SYCL] allocating %zu Bytes with USM system\n", size);
    dev_ptr = (void *)aligned_malloc_host(alignment, aligned_size);   // plain aligned_alloc!
```

That is: when `GGML_SYCL_USM_SYSTEM=1`, a **device** buffer ≥4 GiB is
satisfied with an ordinary host `aligned_alloc` and the driver is left to
fault/migrate pages on access. `g_ggml_sycl_usm_system` defaults to `0`
(`ggml-sycl.cpp:111`) and is read from `GGML_SYCL_USM_SYSTEM`
(`ggml-sycl.cpp:361`); the device must also advertise
`sycl::aspect::usm_system_allocations` (`ggml-sycl.cpp:176`).

**[CONFIRMED]** `docs/backend/SYCL.md` describes it as:

> *"Enable **experimental** support for USM system allocations for large GPU
> buffers. This requires enough host memory for model weights and caches, an
> **Intel Xe2+ GPU such as BMG or newer** and supported on Linux only, with
> `CONFIG_DRM_XE_GPUSVM` enabled."*

**This names Battlemage explicitly.** It is precisely the implicit-migration
mechanism behind the prior project's two host-wide OOM kills.

**Standing rule for this project [INFERRED, but strongly indicated]:** never
set `GGML_SYCL_USM_SYSTEM=1`, and keep the sibling project's
`EnableSharedSystemUsmSupport=0` / `EnableImplicitMigrationOnFaultableHardware=0`
/ `NEOReadDebugKeys=1` in force. Nothing the MoE cache needs requires it —
`sycl::malloc_host` USM (§2.1) is *explicit* host allocation with
*explicit* copies and is a categorically different mechanism. Add a startup
assertion that `GGML_SYCL_USM_SYSTEM` is unset, and check the backend's own
banner line (`ggml-sycl.cpp:470`, `GGML_SYCL_USM_SYSTEM: %d`) in every log.

### 2.6 Verdict — pinned host memory

**PORTABLE, with two design constraints baked in from day one.**
`sycl::malloc_host` is a true `cudaMallocHost` equivalent, is on by default, is
already pooled in-tree, and is already read directly by kernels. The
constraints: **≤2 GiB per pinned allocation** (else copy/compute overlap is
lost), and **no host registration** (allocate-then-fill, never pin-an-mmap).
The implicit-migration hazard is real, is named for Battlemage in the docs, and
is entirely avoidable by leaving one env var alone.

---

## 3. Async copy pattern (dedicated copy stream + readiness events)

### 3.1 The primitives all exist

**[CONFIRMED]**

| CUDA | SYCL, as used in `ggml-sycl` | Where |
|---|---|---|
| `cudaStream_t` | `sycl::queue` (`dpct::queue_ptr`) | `common.hpp:341, 348-358` |
| `cudaMemcpyAsync` | `queue::memcpy(...)` returning `sycl::event` | `ggml-sycl.cpp:4130-4133` |
| `cudaEventRecord` | `queue::ext_oneapi_submit_barrier()` → `sycl::event` | `ggml-sycl.cpp:6105-6109` |
| `cudaStreamWaitEvent` | `queue::ext_oneapi_submit_barrier({event, ...})` | `ggml-sycl.cpp:3436-3437`, `:3558` |
| `cudaEventSynchronize` | `sycl::event::wait()` / `wait_and_throw()` | `ggml-sycl.cpp:6120-6124` |
| `cudaMallocAsync` / `cudaFreeAsync` | `syclex::async_malloc` / `async_free` | `ggml-sycl.cpp:4056`, `:4072` |

The backend even exports these through the generic ggml event API:
`ggml_backend_sycl_event_record` (`ggml-sycl.cpp:6100`),
`ggml_backend_sycl_event_wait` (`ggml-sycl.cpp:6117`),
`ggml_backend_sycl_device_event_new` (`ggml-sycl.cpp:6699`).

### 3.2 Existing async-overlap precedent in-tree

**[CONFIRMED]** Three real precedents:

1. **Copy-then-kernel with event dependency, the reorder path.**
   `reorder_qw_q4_0` (`ggml-sycl.cpp:4121-4157`) and its nine siblings do
   `copy_event = stream->memcpy(tmp_buf, data_device, size)` and then launch
   the reorder kernel, waiting on the event **only when async memory ops are
   unavailable**:
   ```c
   sycl::event copy_event;
   SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
   if (!g_ggml_sycl_use_async_mem_op) { copy_event.wait(); }
   ```
2. **Cross-queue event fencing, the multi-GPU split-matmul path.**
   `ggml-sycl.cpp:3411-3415` records a barrier event on the main device's
   queue into `src0_extra->events[ctx.device][0]`; `ggml-sycl.cpp:3435-3437`
   has other devices' queues wait on it via
   `stream->ext_oneapi_submit_barrier({*src0_extra->events[ctx.device][0]})`;
   `ggml-sycl.cpp:3553-3560` joins back. This is a complete
   record/wait/join triangle, i.e. exactly the shape a copy-queue + compute-
   queue handshake needs.
3. **Kernel `depends_on()` a memcpy event**, the tensor-parallel allreduce —
   documented at `ggml-sycl.cpp:6793-6795`: *"the kernel `depends_on()` its
   corresponding memcpy event so it doesn't read partial data."*

### 3.3 The real gap: there is only ever **one** queue

**[CONFIRMED]** This is the substantive difference from CUDA.
`ggml_backend_sycl_context::stream(int device, int stream)`
(`common.hpp:348-353`):

```c
queue_ptr stream(int device, int stream) {
    if (qptrs[device][stream] == nullptr) {
        qptrs[device][stream] = &(dpct::get_device(device).default_queue());
    }
    return qptrs[device][stream];
}
```

The `stream` index is **ignored**. All `GGML_SYCL_MAX_STREAMS == 8`
(`presets.hpp:16`) slots alias the same object. And that object is a single
**in-order** queue: `dpct`'s `default_queue()` returns `in_order_queue()`
(`dpct/helper.hpp:727-731`), constructed with
`sycl::property::queue::in_order()` (`helper.hpp:753-757`, `:793-796`).

**[CONFIRMED]** `grep 'sycl::queue('` over `ggml/src/ggml-sycl/*.cpp|*.hpp`
returns nothing outside `dpct/helper.hpp` — **the backend never constructs a
queue of its own.** So a dedicated copy queue is genuinely new code, not a
configuration change.

**[CONFIRMED]** …and it comes with a trap the codebase has already been bitten
by. `ggml-sycl.cpp:6961-6964`:

> *"COMM-D2D-FIX: the two devices are in SEPARATE SYCL contexts, so a raw
> `q->memcpy` of a peer USM pointer is a **silent no-op**."*

**[INFERRED]** Therefore the copy queue **must** be constructed from the same
`sycl::context` as the compute queue:
`sycl::queue(ctx.stream()->get_context(), ctx.stream()->get_device(), sycl::property::queue::in_order())`.
Getting this wrong yields silently wrong results, not an error. Also note the
in-order property means intra-queue ordering is free but *inter*-queue ordering
must go through `ext_oneapi_submit_barrier({event})` explicitly.

### 3.4 Verdict — async copy + readiness signalling

**NEEDS NEW CODE (small, well-precedented).** Every primitive exists and is
already used in-tree; what does not exist is a second queue. Roughly: one
queue construction (sharing the compute context), a `sycl::event` per
in-flight admission, and `ext_oneapi_submit_barrier({events})` on the compute
queue before the `mul_mat_id` that consumes them. Copy/compute overlap is
additionally contingent on the ≤2 GiB pinned-allocation rule from §2.4.

---

## 4. Sub-group reduction rewrite (`__shfl_xor_sync`)

This is the least risky item on the list.

### 4.1 Direct 1:1 primitive, used everywhere already

**[CONFIRMED]** `dpct::permute_sub_group_by_xor(sub_group, value, mask [, width])`
is the direct `__shfl_xor_sync` analogue and is the backbone of every reduction
helper in `ggml/src/ggml-sycl/common.hpp:477-620`:
`warp_reduce_sum` (float, float2, half2, int), `warp_reduce_max`,
`warp_reduce_all`, `warp_reduce_any`. The canonical loop
(`common.hpp:478-484`) is byte-for-byte the CUDA idiom:

```c
for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
    x += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), x, mask);
}
```

Standard SYCL group algorithms are used too where they fit:
`sycl::reduce_over_group` (`common.hpp:500-503`), `sycl::all_of_group`
(`common.hpp:562`), `sycl::any_of_group` (`common.hpp:588`).

**[CONFIRMED]** 13 files under `ggml/src/ggml-sycl/` use sub-group primitives,
including `mmvq.cpp`, `dmmv.cpp`, `norm.cpp`, `fattn-vec.hpp`, `cumsum.cpp`,
`quantize.hpp`, `lightning-indexer.cpp`, and — most relevantly —
`topk-moe.cpp`.

### 4.2 Sub-group width is **16**, not 32 — and that is a real porting hazard

**[CONFIRMED]** `ggml/src/ggml-sycl/CMakeLists.txt:166`:

```cmake
if (GGML_SYCL_TARGET STREQUAL "INTEL")
    add_compile_definitions(GGML_SYCL_WARP_SIZE=16)
```

`presets.hpp:19` then does `#define WARP_SIZE GGML_SYCL_WARP_SIZE`. The only
other branch sets 32 and is preceded by `message(FATAL_ERROR ...)`
(`CMakeLists.txt:180-182`) — i.e. **on the supported (Intel) target,
`WARP_SIZE` is unconditionally 16.** `common.hpp:225` documents the intent:
*"For Intel GPU, 16 is better in most cases. Some OP support 32 only."*
A second constant `QK_WARP_SIZE 32` (`presets.hpp:75`) exists for the ops that
genuinely need 32.

Kernels declare the width they need with the SYCL attribute
`[[sycl::reqd_sub_group_size(WARP_SIZE)]]` — e.g.
`mmvq.cpp:2766`, `topk-moe.cpp:216`, and ~30 sites in `cpy.cpp`.

**[INFERRED]** Battlemage/Xe2 hardware supports sub-group sizes 8/16/32 (the
EU SIMD width is 16 on Xe2-HPG, and the compiler can emit SIMD32 via
`reqd_sub_group_size(32)`), but I did **not** confirm the Xe2-specific
supported-widths list from Intel documentation in this pass — see §8. What is
confirmed is only that llama.cpp *chooses* 16 for all Intel targets. Any
ported CUDA reduction that hard-codes 32 lanes, a `0xffffffff` lane mask, or
`mask` starting at 16 must be re-derived against `WARP_SIZE`. Note
`warp_reduce_all`/`warp_reduce_any` in `common.hpp:559-596` still contain
`0xffffffff` lane-mask arithmetic inherited from the CUDA original — evidence
that this class of bug survives translation.

### 4.3 An entire CUDA MoE kernel has already been ported this way

**[CONFIRMED]** `ggml/src/ggml-sycl/topk-moe.cpp:10-13`:

> *"SYCL port of `ggml-cuda/topk-moe.cu`. The kernel is a translation of the
> CUDA no-bias, no-PDL path of `topk_moe_cuda`; the fusion-detection helpers
> below are ported near-verbatim from `ggml-cuda.cu` (pure graph / pointer
> inspection, backend-agnostic)."*

and at `:340`, `:483`, and `ggml-sycl.cpp:5853`, three more
`// ported from ggml_cuda_*` annotations. Its `softmax_warp_inplace`
(`topk-moe.cpp:31-63`) is a sub-group softmax built from `warp_reduce_max` /
`warp_reduce_sum`.

**This is the strongest single feasibility signal in the whole document**: the
MoE *router* — the piece of CUDA logic nearest in kind to the fork's admission
planner — has already been ported CUDA→SYCL in this codebase, by contributors
who documented the method, and it lives on. A device-side frequency-decay LRU
planner is the same class of work.

### 4.4 Verdict — sub-group reductions

**PORTABLE (lowest-risk item).** `dpct::permute_sub_group_by_xor` is a drop-in
for `__shfl_xor_sync`; `reqd_sub_group_size` replaces implicit warp semantics;
the codebase has a documented CUDA→SYCL MoE-kernel port to copy the style
from. The one real cost is auditing every reduction for the 32→16 lane-count
change.

---

## 5. Graph capture / replay — the biggest open risk in this port

### 5.1 SYCL graph support *does* exist in ggml-sycl (correcting the prior assumption)

**[CONFIRMED]** The prior research note that flagged this as the big unknown —
and my own starting assumption — was wrong in the direction of pessimism about
*existence*, and right in the direction of pessimism about *usability*.

`ggml-sycl` implements CUDA-graph-equivalent record/replay via the
`sycl_ext_oneapi_graph` extension:

- Build option, **ON by default**: `ggml/CMakeLists.txt:252` —
  `option(GGML_SYCL_GRAPH "ggml: enable graphs in the SYCL backend" ON)`;
  wired at `ggml/src/ggml-sycl/CMakeLists.txt:185-188`.
- Documented in `docs/backend/SYCL.md:780`, linking
  `https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_graph.asciidoc`.
- Executable graph cached on the context: `common.hpp:462-464` —
  `std::unique_ptr<sycl_ex::command_graph<sycl_ex::graph_state::executable>> exec_graph`.
- Full record → finalize → update → replay cycle in
  `ggml_backend_sycl_graph_compute` (`ggml-sycl.cpp:6051-6091`):
  `command_graph model_sycl_graph(*stream, {property::graph::assume_buffer_outlives_graph{}})`,
  `begin_recording` / `end_recording`,
  `finalize({property::graph::updatable{}})`,
  `exec_graph->update(model_sycl_graph)` with a re-finalize on exception, and
  replay via `stream->ext_oneapi_graph(*exec_graph)`.
- Capability gating on device aspects (`ggml-sycl.cpp:6059`, `:6072`):
  `sycl::aspect::ext_oneapi_limited_graph` for basic record/replay,
  `sycl::aspect::ext_oneapi_graph` for **updatable** graphs.

### 5.2 …but it is off by default at runtime

**[CONFIRMED]** `ggml-sycl.cpp:335`:

```c
g_ggml_sycl_enable_graph = ggml_sycl_get_env("GGML_SYCL_ENABLE_GRAPH", 0);
```

and `docs/backend/SYCL.md`, runtime table:

> `GGML_SYCL_ENABLE_GRAPH` — *"Enable running computations through SYCL Graphs
> feature. **Disabled by default because SYCL Graph is still on development, no
> better performance.**"*

That is the maintainers' own assessment, in-tree: compiled in, opt-in,
currently not a win.

### 5.3 …and MoE models are excluded outright

**[CONFIRMED]** This is the finding that matters most for this project.
`check_graph_compatibility()` (`ggml-sycl.cpp:6006-6048`) refuses to use a
graph at all if the ggml cgraph contains certain ops:

```c
case GGML_OP_CONCAT:
    // ggml_sycl_op_concat() does a blocking host wait after memcpy operations,
    // but wait() can't be called on the events returned by a queue recording to a graph.
    [[fallthrough]];
case GGML_OP_MUL_MAT_ID:
    // ggml_sycl_mul_mat_id() does a blocking host wait on the sycl queue after
    // submitting a memcpy operation, but wait() can't be called on a queue that
    // is recording to a graph.
    GGML_LOG_INFO("%s: disabling SYCL graphs due to unsupported node type %s\n", ...);
    return false;
```

**Every MoE model contains `GGML_OP_MUL_MAT_ID`. Therefore, today, SYCL graphs
are unconditionally disabled for every MoE model in llama.cpp** — including
Qwen3.8-Flash-Next. The "certified decode path with graph replay" half of the
fork's design has, as of this commit, *no working substrate on SYCL*.

Two further exclusions from the same function: multi-device is refused
(`ggml-sycl.cpp:6009-6013`, *"A `sycl_ex::command_graph` object can only be
created for a single device"* — irrelevant to a single-B70 target), and
`GGML_OP_MUL_MAT` is refused unless `SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC` is
available (`ggml-sycl.cpp:6034-6045`), because the weight-reorder path calls
`sycl::malloc`/`free` and `wait()`, none of which are legal while recording.

### 5.4 The exclusion is over-broad — CUDA already fixed it, and **an open PR fixes it for SYCL on this exact card**

**[CONFIRMED]** `ggml-cuda` hits the same underlying problem and solves it
*conditionally* rather than by op type. `ggml-cuda.cu:1871-1899` defines
`ggml_cuda_mul_mat_id_needs_sync(dst, cc)`, which returns `false` (no sync
needed) whenever the op will take one of the vectorised fast paths; and
`ggml_cuda_graph_check_compability` (`ggml-cuda.cu:2551-2574`) disables graphs
only when that predicate is true:

```c
// [TAG_MUL_MAT_ID_CUDA_GRAPHS]
if (node->op == GGML_OP_MUL_MAT_ID) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (ggml_cuda_mul_mat_id_needs_sync(node, cc)) {
        // the mul_mat_id fallback path synchronizes the stream, so we cannot use CUDA graphs
        // ref: https://github.com/ggml-org/llama.cpp/pull/18958
        use_cuda_graph = false;
```

Both sites are tagged `[TAG_MUL_MAT_ID_CUDA_GRAPHS]`, and
`ggml_cuda_mul_mat_id` itself asserts at `ggml-cuda.cu:1944` that the
sync-requiring fallback is never reached under capture. That refinement landed
as PR [#26802](https://github.com/ggml-org/llama.cpp/pull/26802), merged
**2026-08-11**, tightening the earlier blanket check from #18958.

**[CONFIRMED, via the GitHub API]** The SYCL history is:

- PR [#13587](https://github.com/ggml-org/llama.cpp/pull/13587) — *"SYCL: Avoid
  using SYCL-Graph for unsupported nodes"*, by `EwanC` (Codeplay), merged
  **2025-05-22**. This is the **origin** of the `CONCAT` / `MUL_MAT_ID`
  exclusion. Its rationale, verbatim, is `test-backend-ops` throwing "from the
  blocking waits during queue recording", and it says it was modelled on
  `ggml-cuda`'s `check_node_graph_compatibility_and_refresh_copy_ops`. It was
  a conservative op-type blacklist from the start.
- PR [**#25089**](https://github.com/ggml-org/llama.cpp/pull/25089) —
  *"sycl: fix `check_graph_compatibility()` to allow graphs for MoE decode
  (CONCAT dim!=3, MUL_MAT_ID fused path)"*, by `Captain-Tripps`, **opened
  2026-06-28, still open, last touched 2026-07-28**. **+28 / −11 lines, one
  file.** Its argument is exactly the one §1.3 arrives at independently: the
  non-fused prefill path (`ne12 > 1`) copies expert IDs to host and must stay
  excluded, but `ggml_sycl_mul_mat_id_mmvq_fused()` (`ne12 == 1`, FP32 `src1`)
  "runs `ggml_sycl_mul_mat_vec_q_id()` entirely on GPU with no host wait —
  graph-compatible." Same for `CONCAT` with `dim != 3`.

  **Its test plan was run on an Intel Arc Pro B70 (Xe2/Battlemage) with
  qwen3.6-35B-A3B Q4_K_M** — the same card as this project, on a Qwen MoE.
  Reported: graph capture succeeded, decode output correct, 3678-token
  sustained decode clean over 207 s, prefill unaffected.

  It also answers the scratch-allocation worry directly: *"`ggml_sycl_pool_vmm`
  uses a fixed base address with LIFO linear allocation. `src1_q8_alloc` in the
  fused path always gets `pool_addr+0`, so addresses are stable across graph
  replays when `g_ggml_sycl_use_async_mem_op` is set."*

  **Why it is stalled: rebase, not rejection.** Maintainers (`ggerganov`,
  `NeoZhangJianyu`, `arthw`) asked for a rebase on 2026-07-07 and again on
  2026-07-28; `arthw` noted on 2026-07-08 there was "no code conflict with the
  base". No technical objection was raised. The env var was renamed
  (`GGML_SYCL_DISABLE_GRAPH` → `GGML_SYCL_ENABLE_GRAPH`) in the interim, which
  is presumably part of why it needs one. The bot also flagged AI-generated
  content and the author disclosed Claude Code assistance.

**Assessment.** This changes the §5 outlook materially: the prerequisite is not
research, it is a 39-line diff that someone has already written, already
validated on an Arc Pro B70, and that is waiting on a rebase. **[INFERRED]**
The pragmatic path is to rebase #25089 onto current `master` locally, use it as
a build-time patch for this project, and — if it holds up — offer the rebase
upstream. That is hours of work, not a research programme. It does not remove
the deeper risk in §5.2 (graphs are still off by default and the maintainers
still say they give "no better performance"), and it does not by itself provide
the expert-slab re-pointing described in §5.5–5.6.

Related and worth reading alongside it: issue
[#24810](https://github.com/ggml-org/llama.cpp/issues/24810), which #25089 grew
out of — the SYCL backend does not detect device loss and spins forever when a
Battlemage GPU resets. That is a live hazard for any long-running experiment on
this card, independent of the MoE cache.

The `async_malloc`/`async_free` machinery the fused path depends on under
capture (`ggml-sycl.cpp:4051-4079`, gated on
`SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC`) was made independently controllable by PR
[#22153](https://github.com/ggml-org/llama.cpp/pull/22153), merged
**2026-05-19**, adding `GGML_SYCL_USE_ASYNC_MEM_OP` (default 1);
`g_ggml_sycl_use_async_mem_op` is additionally forced on whenever graphs are
enabled (`ggml-sycl.cpp:485`).

### 5.5 Re-pointing kernel arguments between replays: supported by the spec, not by ggml-sycl

This is the mechanism the MoE cache actually needs — after an eviction, the
`mul_mat_id` kernel must read a *different* device address for an expert's
weights, without re-recording the graph.

**[CONFIRMED, from the extension spec at
`https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_graph.asciidoc`]**

- Status: **experimental**, with the standard "shipping software products
  should not rely on APIs defined in this specification" disclaimer. Feature
  macro `SYCL_EXT_ONEAPI_GRAPH` (currently `2`).
- Two update APIs exist:
  - **Individual node update** — a `dynamic_parameter<ValueT>` is registered to
    a node's kernel argument via `handler::set_arg()`; `dynamic_parameter::update(v)`
    then `command_graph<executable>::update(node)` commits it. The spec states
    the underlying type may be *"an accessor, **a pointer to a USM
    allocation**, scalar passed by value, or a raw byte representation."*
    **So yes — swapping which USM address a kernel reads its weights from is an
    explicitly supported update.** Graph topology/edges are *not* re-derived;
    the caller owns data-dependency correctness.
  - **Whole-graph update** — `command_graph<executable>::update(const command_graph<modifiable>&)`,
    replacing all node parameters from a topologically identical source graph.
- Both require `property::graph::updatable` at `finalize()` **and**
  `aspect::ext_oneapi_graph` (the full aspect, not `ext_oneapi_limited_graph`).
- USM `memcpy` **is** a legal graph node — `node_type` includes `memcpy`,
  `memset`, `memfill`, `prefetch`, `memadvise`, `async_malloc`, `async_free`,
  `host_task`, `kernel`.
- Prohibited while recording, confirming the in-tree comments: *"A host-side
  wait on a queue in the recording state is an error and will throw
  synchronously with error code `invalid`"*; likewise `event::wait()`,
  `event::get_info<command_execution_status>()`, and `get_profiling_info()`.
  Host tasks *are* legal nodes but may block whole-graph submission.
  `sycl::malloc_device`/`free` are host-synchronous and outside the submission
  model — `async_malloc`/`async_free` is the sanctioned in-graph replacement
  (**[INFERRED]** from the spec's type table and example placement; not stated
  as a verbatim prohibition).
- A separate, weaker capture mode exists: `property::graph::enable_native_recording`
  delegates capture to the backend's own native graph API. It forbids host
  tasks, `async_malloc`/`async_free`, and buffers, and **`update()` throws
  `feature_not_supported`** — native-recorded graphs are not updatable. Do not
  use it for this.

**[CONFIRMED]** `ggml-sycl` uses **whole-graph update only**
(`exec_graph->update(model_sycl_graph)`, `ggml-sycl.cpp:6080`). It exposes no
per-node `dynamic_parameter`. So the capability this project needs exists in
the extension and is *not* reachable through the existing
`ggml_backend_sycl_graph_compute` path — wiring it up is new work on top of
#25089, not something to inherit.

**[CONFIRMED]** Driver backing on Battlemage looks present. `intel/llvm`'s
Level Zero adapters implement the Unified Runtime `command_buffer` (the SYCL
graph backend) on raw L0 command lists, and create it with
`ze_mutable_command_list_exp_desc_t` in the `pNext` chain when updatable,
asserting `ZeMutableCmdListExt.Supported`
(`unified-runtime/source/adapters/level_zero/command_buffer.cpp` and its `v2/`
counterpart). Support is probed at runtime via
`zeDriverGetExtensionFunctionAddress(..., "zeCommandListGetNextCommandIdExp")`
and `"zeCommandListUpdateMutableCommandsExp"`
(`.../level_zero/common/platform.cpp`). And `intel/compute-runtime` ships
**Battlemage-specific mutable-command-list code**:
`level_zero/core/source/xe2_hpg_core/bmg/mutable_cmdlist_bmg.cpp`. So the
`ZE_extension_mutable_command_list` chain that `aspect::ext_oneapi_graph`
depends on is implemented for BMG.

**[INFERRED]** That makes it *likely* the B70 reports the full
`aspect::ext_oneapi_graph`. It is still a one-line runtime probe on the actual
card and should be run before designing around it (§8 Q1).

For completeness on the Level Zero layer: plain `zeCommandListCreate` produces
an open list that is appended to, `zeCommandListClose`d, and submitted with
`zeCommandQueueExecuteCommandLists` — the record-once/replay-many primitive.
`zeCommandListCreateImmediate` is the opposite (low-latency streaming, must
*not* be passed to `ExecuteCommandLists`). Separately, compute-runtime also
ships an undocumented driver-experimental native graph extension
(`zeGraphCreateExt`, `zeCommandListBeginGraphCaptureExt`, …) that backs SYCL's
`enable_native_recording`; it is absent from the public `oneapi-src/level-zero-spec`
repo and, per above, is not updatable.

### 5.6 A stable-address alternative that may sidestep graph updates entirely

**[CONFIRMED]** `ggml-sycl` uses the `sycl_ext_oneapi_virtual_mem` extension.
`ggml_sycl_pool_vmm` (`ggml-sycl.cpp:1741-1840`) calls
`sycl::ext::oneapi::experimental::reserve_virtual_mem()`, constructs
`physical_mem` objects, `phys->map(pool_addr + pool_size, reserve_size, address_access_mode::read_write)`,
and `unmap()`s in the destructor. It is selected by default
(`GGML_SYCL_ENABLE_VMM` defaults to 1, `ggml-sycl.cpp:102`, `:342`) whenever
the device advertises `sycl::aspect::ext_oneapi_virtual_mem`
(`ggml-sycl.cpp:152`). This is the direct `cuMemAddressReserve`/`cuMemMap`
analogue.

**[INFERRED]** That opens a third design, avoiding node updates altogether.
Instead of re-pointing kernel arguments per decode step, reserve one fixed
virtual address range for the slot pool and **remap physical pages underneath a
constant VA** on eviction/admission. Kernel arguments never change, so a
recorded graph stays valid across expert swaps with no `update()` call and no
dependence on the full `aspect::ext_oneapi_graph`. Note PR #25089's own
argument leans on a weaker version of the same property — the VMM pool's
fixed base plus LIFO allocation is what makes the fused path's scratch buffer
address stable across replays. Two things must be checked before relying on the
stronger form (§8): whether `unmap`/`map` is legal while an executable graph
referencing that VA exists, and what the remap latency actually is on the L0
driver — a per-decode-step `unmap`+`map` pair could easily cost more than the
copy it was meant to optimise.

**[INFERRED]** A fourth option deserves mention because it may dominate all of
the above: keep the kernel argument a *constant* pointer to a small device-side
**slot table** (`const void * const * expert_ptr`, or `int32_t slot_of[n_expert]`),
and have the admission kernel write into that table. The graph then never needs
updating, no VMM remapping is involved, and the indirection costs one extra
load per work-group. This is the design §1.3 and §2.2 point at, and it is the
one that requires the least from the graph extension — only that a plain kernel
launch replays, which is `aspect::ext_oneapi_limited_graph` territory.

### 5.7 Verdict — graph capture/replay

**Downgraded from "blocked" to "needs new code, on top of a stalled upstream
PR" — but still the riskiest component and still the right thing to descope
from milestone one.**

What changed relative to the initial read:

- The substrate exists (`sycl_ext_oneapi_graph`, wired, ON at build time), and
  its update API **does** support re-pointing a USM-pointer kernel argument.
- The L0 mutable-command-list chain that backs updatable graphs **is**
  implemented for Battlemage in `compute-runtime`.
- The MoE exclusion is over-broad, and the fix is a **39-line open PR
  (#25089), already validated on an Arc Pro B70 with a Qwen MoE**, stalled on a
  rebase rather than on any technical objection.

What has not changed:

- Graphs remain **off by default at runtime**, with the maintainers' own note
  that they give *"no better performance"* — so the payoff is unproven on this
  backend even once capture works.
- `ggml-sycl` uses **whole-graph update only**; per-node `dynamic_parameter`
  re-pointing is new work regardless.
- The extension is **experimental** by its own declaration, on a driver stack
  that already has an open "does not detect device loss, spins forever" bug on
  this exact card (#24810).

**[INFERRED] Recommended sequencing:** treat graph replay as a **Phase 3
stretch goal, explicitly descoped from the first milestone.** The fork's own
description separates a "legacy path" (host readback of router IDs, plain LRU)
from the "grouped decode fast path" (device-side planner + graph capture). The
legacy path needs no graphs at all, and even the device-side planner (§2.2)
needs no graphs — only the capture/replay layer on top does. Build and measure
in that order; if the cache does not win *without* graph replay, graph replay
is unlikely to rescue it, and if it does win, the graph work has a measured
baseline to justify itself against. Cheap early action, independent of the
cache: rebase #25089 locally, set `GGML_SYCL_ENABLE_GRAPH=1`, and measure
whether graph capture alone moves decode throughput on the B70 at all. That
single number decides how much §5 is worth.

---

## 6. Backend maturity assessment

### 6.1 Commit velocity: comparable to CUDA and Vulkan

**[CONFIRMED]** Measured on the local checkout, window 2026-04-22 → 2026-09-09
(the full depth of the fetched history, 2001 commits of `master`):

| Backend | Commits touching it in window |
|---|---|
| `ggml/src/ggml-cuda` | 142 |
| `ggml/src/ggml-sycl` | **121** |
| `ggml/src/ggml-vulkan` | 120 |

Monthly for `ggml-sycl`: 2026-05 → 16, 06 → 26, 07 → 25, 08 → **38**, 09 (9
days) → 11. Activity is **rising**, not decaying. Top authors in window: Neo
Zhang (41, Intel), Titaniumtown (17), an "Intel AI Get-to Market" account (7),
Todd Malsbary (6), Alexey Kopytko (6).

Independent corroboration: 152 commits to `ggml/src/ggml-sycl/` in 2026 to
date and ~160 merged PRs with "sycl" in the title created since 2026-01-01;
the most recent at research time was PR **#28476, "SYCL: Add IQ type handling
for MoE"**, merged **2026-09-09** — the same day as this research.

**[INFERRED]** Codeplay's direct upstream involvement appears to have gone
quiet: contributors historically associated with it (Svetlozar Georgiev,
Romain Biessy) show 0 merged llama.cpp PRs in 2026, and there are no 2026
issue/PR mentions of "Codeplay". Intel headcount plus a community long tail is
carrying the backend.

### 6.2 MoE / `MUL_MAT_ID` support is real and actively extended

**[CONFIRMED]**

- `docs/ops.md` (upstream's generated op-support matrix) marks
  `MUL_MAT_ID` as fully supported (✅) on SYCL, level with CUDA and Vulkan.
- `docs/backend/SYCL.md` lists "Fused MoE" among 2026.04–05 feature additions.
- In-tree, beyond `ggml_sycl_mul_mat_id` itself: `topk-moe.cpp` (the fused
  router), `ggml_sycl_mul_mat_vec_q_id` / `..._id_reorder` (fused MoE GEMV,
  `mmvq.cpp:2774`, `:2984`), `opt_for_reorder_id` (per-expert SoA relayout),
  `add-id.cpp`, and a fusion pass matching
  `{MUL_MAT_ID, ADD_ID, MUL_MAT_ID, ADD_ID, GLU}` on the CUDA side
  (`ggml-cuda.cu:3187`) with SYCL analogues in `fusion.cpp`.
- SYCL-specific engineering for this exact silicon exists: PR #26689
  *"sycl: use TILE for quantized KV decode on BMG"* (merged 2026-08-28).

### 6.3 Open reliability backlog, much of it on this exact card

**[CONFIRMED]** The `SYCL` GitHub label is stale (0 open issues; maintainers
ask for a `[SYCL]` title prefix instead), so the backlog only shows up by title
search. Open as of 2026-09:

| Issue | Date | Summary |
|---|---|---|
| [#26581](https://github.com/ggml-org/llama.cpp/issues/26581) | 2026-09-04 | Decode attention memory-latency-bound on **Arc Pro B70** (BMG-G31, 32 GB, 608 GB/s); ~21–25 ns per KV position per layer; *identical on Vulkan and SYCL* |
| [#27198](https://github.com/ggml-org/llama.cpp/issues/27198) | 2026-09-01 | `--split-mode tensor` crashes in `dev2dev_memcpy` (DEVICE_LOST) on **dual Arc Pro B70** |
| [#25692](https://github.com/ggml-org/llama.cpp/issues/25692) | 2026-09-01 | GPU hang (Xe CCS engine reset) with FA + quantized KV under sustained server load, **Arc Pro B70** |
| [#24810](https://github.com/ggml-org/llama.cpp/issues/24810) | (see §5.4) | SYCL backend does not detect device loss — **spins forever when a Battlemage GPU resets**. This is the investigation PR #25089 grew out of, and a live hazard for any long-running experiment on this card |
| [#28193](https://github.com/ggml-org/llama.cpp/issues/28193) | 2026-09-02 | `test-backend-ops FLASH_ATTN_EXT` failures on **Arc B70**, permuted q8_0 KV |
| [#26206](https://github.com/ggml-org/llama.cpp/issues/26206) | 2026-09-03 | Gemma 4 12B garbled output on large prompts, **Arc Pro B70** (Xe2) |
| [#28515](https://github.com/ggml-org/llama.cpp/issues/28515) | 2026-09-07 | `ggml_backend_sycl_device_get_memory` aborts, "failed to get device memory size", **Arc Pro B60 (Battlemage)** |
| [#24168](https://github.com/ggml-org/llama.cpp/issues/24168) | 2026-06-05 → 09-04 | Gibberish/crash regression on hybrid (SSM+MoE) archs — **qwen3next / qwen35** — on Arc Pro B60; pure-MoE Qwen3-Coder-30B-A3B unaffected |
| [#25812](https://github.com/ggml-org/llama.cpp/issues/25812) | 2026-07-17 | `UR_RESULT_ERROR_OUT_OF_HOST_MEMORY` offloading GLM-5.2 MoE experts; crashes in `reorder_qw_q4_k_moe` |
| [#25973](https://github.com/ggml-org/llama.cpp/issues/25973) | 2026-08-29 | Performance regression on newer oneAPI versions |
| [#26010](https://github.com/ggml-org/llama.cpp/issues/26010) | 2026-09-05 | SYCL ~2.9× slower than Vulkan for token generation (Arc iGPU) |

**#24168 is the single most alarming entry for this project**: it is a
still-open correctness regression specifically on **hybrid SSM+MoE
architectures including `qwen3next`** — the closest published relative of
Qwen3.8-Flash-Next's Gated-DeltaNet-plus-MoE design — on Battlemage hardware.
Cross-check this against `docs/research/02-model-support-status.md` before
anything else.

**[CONFIRMED]** `docs/backend/SYCL.md`'s "Verified devices" table lists only
`Arc B580` for the B-series; **Arc Pro B60/B70 and Battlemage/BMG-G31 appear
nowhere in it**, despite the volume of B70 bug reports above. The doc's "Known
Issues" section lists exactly two items (`split-mode:[row]` unsupported;
no AOT build) and reflects none of the backlog.

### 6.4 Institutional signals, both directions

**[CONFIRMED, negative]** PR
[#23705](https://github.com/ggml-org/llama.cpp/pull/23705) — *"ci: reduce
(disable SYCL and CANN builds/releases)"*, merged **2026-05-26** by the project
lead: *"The SYCL builds alone consume more than 1/3 of the total 10GB cache
that we have. I don't think it's reasonable, so disabling them for now."*
SYCL binaries were absent from official releases until PR
[#27385](https://github.com/ggml-org/llama.cpp/pull/27385) restored them on
**2026-08-19** — roughly three months.

**[CONFIRMED, negative]** `docs/backend/SYCL.md` news for 2026.02: support for
NVIDIA and AMD GPUs via SYCL was **removed** ("the oneAPI plugin for Nvidia &
AMD GPU is unavailable"). The backend is Intel-only in practice.

**[CONFIRMED, competitive]** An open Intel-Xe-targeted Vulkan optimisation
series — PRs #24404 / #24406 / #24407 under mega-PR #24408 — adds
"Intel Xe flash attention optimization kernels (Xe-LPG Plus / Xe2 / Xe3)" and
GEMM/group-GEMM optimisations, plus merged #27471 (Intel coopmat1 matmul
pipelines) and #27925 (Vulkan `mul_mat_id` K-padding, MoE-relevant).
**[INFERRED]** Some of that effort looks Intel-affiliated by author handle
(e.g. `jxia4intel`), though profile `company` fields are unset. Net reading:
Vulkan is receiving serious Intel-specific attention in exactly the areas SYCL
users complain about.

**[CONFIRMED, positive]** Against all of that: 121 commits in five months,
rising monthly, MoE work landing the day of this research, BMG-specific kernel
tuning in-tree, and a documented, repeated practice of porting CUDA kernels
into `ggml-sycl` verbatim (§4.3).

### 6.5 Verdict — maturity

**Actively maintained; a reasonable but not comfortable bet.** `ggml-sycl` is
not lagging on *velocity* — it tracks CUDA and Vulkan closely and is
accelerating. It lags on *reliability* (a long open backlog concentrated on
Battlemage / Arc Pro B-series), on *documentation currency* (the hardware
table does not know this card exists), and on *institutional priority*
(a three-month release-CI outage; Intel's own Vulkan investment). The specific
risk to this project is not "the backend will be abandoned" but "you will spend
a meaningful fraction of your time debugging pre-existing SYCL bugs on B70
that have nothing to do with the MoE cache" — and, worse, that the hybrid-arch
correctness regression #24168 blocks the target model independently of any of
this.

---

## 7. Bottom-line feasibility verdict per component

| # | CUDA mechanism | SYCL equivalent | Verdict |
|---|---|---|---|
| 1a | Pluggable `ggml_backend_buffer_type` | Fully implemented, 3 buft variants, `tensor->extra` present | **Portable** |
| 1b | Single `mul_mat_id` interception site | `ggml_sycl_mul_mat_id` (`ggml-sycl.cpp:5077`), one dispatch case | **Portable** |
| 1c | Slot indirection in the expert GEMV | `mul_mat_vec_q_moe` (`mmvq.cpp:2704`) — one line to redirect | **Portable** |
| 1d | Custom buft accepted by the backend | Blocked by 3 `get_name`-pointer identity checks + 1 debug assert | **Needs new code** (small; same in CUDA) |
| 2a | `cudaMallocHost` pinned host tier | `sycl::malloc_host`, default-on, already pooled | **Portable** |
| 2b | Kernel reads host-resident weights | Precedent: `GGML_SYCL_HOST_MEM_FALLBACK` reorder path | **Portable** (perf unverified) |
| 2c | `cudaHostRegister` on an mmap'd file | **Does not exist in SYCL** | **Blocked** — use `--load-mode none` only; drop the mmap L2 tier |
| 2d | Arbitrarily large pinned allocation | >2 GiB kills copy/compute overlap | **Needs new code** — chunk the cold tier ≤2 GiB |
| 3a | Dedicated `cudaStream_t` copy stream | Every primitive exists; **no second queue exists anywhere in-tree** | **Needs new code** (small, must share `sycl::context`) |
| 3b | `cudaEvent_t` readiness signalling | `sycl::event` + `ext_oneapi_submit_barrier({e})`, used in-tree | **Portable** |
| 4 | `__shfl_xor_sync` warp reductions | `dpct::permute_sub_group_by_xor`, `reqd_sub_group_size` | **Portable** (audit for 32→**16** lanes) |
| 5a | CUDA graph capture/replay substrate | `sycl_ext_oneapi_graph`, wired, build-ON; L0 mutable-cmdlist present for BMG | **Portable in principle** (extension is *experimental*) |
| 5b | Graph replay **for a MoE decode step** | Hard-disabled by op type (`ggml-sycl.cpp:6025-6031`); **open PR #25089 fixes it, +28/−11, validated on an Arc Pro B70** | **Needs new code** — rebase and carry #25089 |
| 5c | Re-point an expert's weight pointer between replays | Spec supports `dynamic_parameter<T*>`; **ggml-sycl exposes whole-graph update only** | **Needs new code** (or avoid entirely via 5e) |
| 5d | Graph enabled by default | `GGML_SYCL_ENABLE_GRAPH` defaults 0, *"no better performance"* | **Unproven** — measure before investing |
| 5e | Avoiding 5c: constant-arg slot table, or VMM remap under a fixed VA | Device-side `slot_of[]`/pointer table; or `reserve_virtual_mem` + `map`/`unmap` | **Portable** (5e is the recommended design) |
| 6 | Device-side admission planner | Ordinary SYCL kernel; `topk-moe.cpp` is a ported-from-CUDA precedent | **Portable** |
| 7 | Fixed-size VRAM slot pool | `sycl_ext_oneapi_virtual_mem` in `ggml_sycl_pool_vmm`, default-on | **Portable** |

**Overall.** Nothing is hard-blocked. Everything outside §5 is portable or
needs a small, well-precedented amount of new code, and the one true blocker
found in §5 turned out to have a 39-line open upstream PR against it that was
already tested on this exact GPU with a Qwen MoE model. The remaining §5 risk
is not "can it be made to capture" but "is capture worth anything on this
backend" — the maintainers' own default says not yet.

The project is feasible **if scoped to land the cache without graph replay
first**: cached buffer type → pinned host tier → copy queue → device-side
planner, all of which stand alone, with graph replay as a measured add-on.

**Two risks larger than any of the above, both outside this document's scope:**
(a) whether the chosen quant of Qwen3.8-Flash-Next's expert stack already fits
in 32 GB, in which case the cache is worth **negative** performance (`docs/00-background.md`
§1 records **-6.5%** for a fully-resident model); and (b) open issue
[#24168](https://github.com/ggml-org/llama.cpp/issues/24168), an unresolved
correctness regression on hybrid SSM+MoE architectures including `qwen3next`,
on Battlemage. Neither is a SYCL-porting question, and either can end the
project before a line of cache code is written.

---

## 8. Open questions / gaps

Confirmed-from-source unless noted; these are the things this pass could
**not** settle.

Questions 1, 2, 5 and 6 from the first draft of this document were resolved
during research and now live in §5.4–§5.5. What remains:

**Graph questions still open.**
1. Does an Arc Pro B70 on our L0 driver advertise `aspect::ext_oneapi_graph`
   (updatable) or only `aspect::ext_oneapi_limited_graph`? The
   `compute-runtime` BMG mutable-command-list source says it should be the
   former, but that is inference — **it is a one-line runtime probe on the
   actual card and should be the very first thing run.** A "limited only"
   result forces the constant-argument design (§5.6 option four).
2. Does PR [#25089](https://github.com/ggml-org/llama.cpp/pull/25089) still
   apply after rebase onto current `master`, given the
   `GGML_SYCL_DISABLE_GRAPH` → `GGML_SYCL_ENABLE_GRAPH` rename and whatever
   else moved since 2026-06-28? And does its B70 result reproduce here?
3. With #25089 applied and `GGML_SYCL_ENABLE_GRAPH=1`, **does graph capture
   alone change decode throughput on the B70 for a MoE model?** The
   maintainers say "no better performance" generally. This single measurement
   determines whether any further §5 work is justified.
4. Is `SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC` present in the oneAPI toolchain
   version we will build with? Without it, graphs are refused for plain
   `GGML_OP_MUL_MAT` too (`ggml-sycl.cpp:6034-6045`), so nothing works.
5. Is `unmap`/`map` on the VMM pool legal while an executable graph referencing
   that virtual address exists (§5.6)? Not addressed by the extension spec text
   reviewed.

**Hardware/driver facts to confirm on the actual card.**
7. Xe2/Battlemage supported **sub-group widths**. `WARP_SIZE=16` is what
   llama.cpp *chooses* for all Intel targets (`CMakeLists.txt:166` —
   confirmed); the hardware's supported set (8/16/32?) was **not** confirmed
   from Intel documentation in this pass. Matters only if a ported kernel wants
   `reqd_sub_group_size(32)`.
8. Measured PCIe H2D bandwidth of a **kernel-driven gather from
   `sycl::malloc_host`** (§2.2) versus `queue::memcpy` on a dedicated queue.
   The whole device-side-planner design leans on the former being acceptable.
9. Actual latency of `unmap` + `map` on the VMM pool (§5.6). If it is
   milliseconds, stable-address remapping is dead as a per-step mechanism and
   the constant-argument slot table is the only viable design.
10. Does the ≤2 GiB pinned-allocation rule (§2.4) reflect a hard Level Zero
    limit or a tunable? `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS` exists for the
    >4 GiB *device* case; the >2 GiB *host* case's mechanism was not traced.

**Codebase questions.**
11. Does the `mmvq_fused` path's `ggml_sycl_pool_alloc` scratch allocation
    (`ggml-sycl.cpp:5004`) actually become graph-legal under
    `async_malloc`, or does the VMM pool bypass the async path? Determines
    whether §5.4's predicate PR is sufficient on its own.
12. How does the lazy Q4_K reorder (`opt_for_reorder_id`,
    `ggml-sycl.cpp:4999`) interact with slabs that move between host and
    device? Reorder must happen once on the cold copy, not per admission.
13. Not read in this pass: the fork's own `moe-cache.cu`/`.cuh`. Everything
    said here about *what the fork does* is second-hand from
    `docs/00-background.md`. Before committing to a port, read the fork's
    `ggml_cuda_mul_mat_id` diff specifically — it is the part with a direct
    SYCL counterpart.
14. `ggml-sycl` has a `memtrace.cpp` / `GGML_SYCL_MEMTRACE` facility
    (`docs/backend/SYCL.md` runtime table). Worth wiring the cache's
    allocations into it rather than inventing separate instrumentation.
