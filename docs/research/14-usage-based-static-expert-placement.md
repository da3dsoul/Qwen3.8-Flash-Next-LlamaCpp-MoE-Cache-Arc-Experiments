# Usage-Based Static Per-Expert Placement — Scoping

Date: 2026-09-12. Status: **scoped, NOT implemented. The recommendation is a
40-LOC measurement first, and a conditional yes to the build only if that
measurement comes back a particular way.** No GPU work was done for this pass
(another workstream was running `qsa_plan_ab.sh` on the card throughout — `ps`
confirmed `llama-bench -ncmoe 38 -d 8192,32768,614400` live, `free -h`
`available` 82 GiB with 40 GiB already committed to it).

The proposal under examination:

> Instead of `-ncmoe` offloading whole MoE layers indiscriminately to CPU,
> determine which *individual experts* (across all layers) are used most
> frequently, keep those GPU-resident and offload the least-used ones — a
> finer-grained but still **static** placement, decided once at load and never
> migrated at runtime.

Everything below was read directly in the working tree
(`src/llama.cpp`, HEAD `6d9c82ea2`, plus this project's uncommitted work).
Line ranges are cited, not guessed; claims that connect facts from separate
files are flagged **inference**.

Companion documents this one depends on and does not repeat:
`05-hybrid-cpu-gpu-mul-mat-id-scoping.md` §3.2/§10.3-§10.6 (the scheduler
mechanics and, critically, the measured per-node cost fits),
`13-dynamic-context-aware-expert-placement.md` §3/§4/§9 (why live migration is
dead and what `--context-tiers` actually is), `08-long-context-vram-budget.md`
§2.3 (the expert-VRAM closed form), `12-decode-depth-scaling.md` §5 (the
floor/slope tension), `10-sycl-sparse-attention-scoping.md` ADDENDUM (the
current decode decomposition), and `00-background.md` §1 (the reference fork).

---

## 0. One-paragraph summary of the finding

**This is a genuinely different proposal from the one doc 13 rejected, it
avoids all three of that document's blockers, and most of the infrastructure it
needs already exists in this tree and has already been measured. But its entire
value reduces to a single unmeasured number, and the arithmetic that shows this
is exact rather than hand-waved.** Four findings. (1) **Per-expert placement
cannot be expressed as a buffer-type override, at any granularity** — the
override is a `std::regex_search` against the tensor *name*
(`src/llama-model-loader.cpp:1226-1252`), and all 512 experts of a layer are one
3D tensor (`src/llama-model.cpp:3222-3228`), so this needs op-level dispatch, not
plumbing (§2.1-§2.2). (2) **But the op-level dispatch is already built.** The
hybrid `MUL_MAT_ID` hook from doc 05's spike is live in this tree
(`ggml/src/ggml-mmid-hybrid.h`, `ggml/src/ggml-cpu/ggml-cpu.c:1659-1698`,
`ggml/src/ggml-sycl/mmid-hybrid.cpp`), it already splits a node's routed rows
between CPU and GPU per row, it already produced **byte-identical output at
every split fraction including 1.0** (doc 05 §10.3), and its GPU chain already
takes the weight base and per-expert stride as arguments
(`mmid-hybrid.cpp:393-394`). What the spike lacked is exactly what this proposal
supplies: a reason for the diverted rows' weights to be *already in VRAM*, which
is the difference between its measured 22.9 us/row and 3.31 us/row (§2.4). (3)
**The value is very nearly just the non-uniformity of the routing
distribution.** At residency fraction `f` and hit rate `h`, host-side expert
work is proportional to `(1−h)` where today's `-ncmoe` gives `(1−f)`, and the
two are equal when `h = f` — so **at uniform routing this proposal is worth
approximately nothing** (same CPU rows, twice the nodes, 144 dispatches), and
essentially every millisecond of gain is bought by `h − f` (§3.1). The one
exception, and it is not a bet on skew: the hook overlaps GPU expert work
*inside* a CPU-backend node, where today's alternating sched splits run
sequentially (`ggml/src/ggml-backend.cpp:775-784`). Break-even is `h ≈ 0.46` if
that overlap holds and `h ≈ 0.67` if it does not; the ceiling is **1.12x-1.35x**
decode at `h ≈ 0.89`, and 1.11-1.31x at 300K/600K where `-ncmoe` is higher
(§3.2-§3.3). (4) **The skew premise is
not measured for this model, and the number everyone is citing is about a
different model in a different project.** "Top 10% of experts take ~80% of hits"
is RFC #24528's body about **Qwen3.5-122B**
(`05-hybrid-cpu-gpu-mul-mat-id-scoping.md:69-71`), not this fork's measurement
and not `qwen4exp`; and `expert_frequency` in this project's own cache is
**never incremented on a hit** (`moe-cache.cpp:931-932` is its only write), so no
per-expert profile exists anywhere in this repo (§4). Recommendation: build the
histogram first — ~40 LOC on the CPU path, no GPU, no kernel, reusable inside
any existing benchmark (§6). Effort table in §8.

---

## 1. What the proposal is, restated against the code

### 1.1 Today's granule is one named tensor, and that is a hard property of the override mechanism

`-ncmoe N` is sugar over `-ot`. `common/arg.cpp:2828-2837` does nothing but call
`llm_add_n_cpu_ffn_overrides` (`common/common.h:1156-1162`), which pushes `N`
regex strings `blk\.<i>\.ffn_(up|down|gate|gate_up)_(ch|)exps`
(`common/common.h:1144`, `:1148-1150`) paired with
`ggml_backend_cpu_buffer_type()` into `common_params::tensor_buft_overrides`.

Those reach the loader and are consumed at
`src/llama-model-loader.cpp:1226-1252`. The whole of the matching logic is:

```cpp
        if (tensor_buft_overrides) {
            std::string tensor_name = tn.str();
            for (const auto * overrides = tensor_buft_overrides; overrides->pattern != nullptr; ++overrides) {
                std::regex pattern(overrides->pattern);
                if (std::regex_search(tensor_name, pattern)) {
```

**The key is a string — the tensor's name — and the value is one
`ggml_backend_buffer_type_t` for the whole tensor.** There is no offset, no row
range, no sub-tensor addressing anywhere in the structure
(`llama_model_tensor_buft_override` is `{ pattern, buft }`). Sub-tensor
granularity is not "unimplemented" here; it is *unrepresentable*, and no amount
of new plumbing in this path changes that without a different data model.

Downstream, the chosen `buft` selects which `ggml_context` the tensor is created
in, buffers are allocated one per (context, buft), and placement is therefore
fixed before any memory exists — doc 13 §1.1 establishes this chain and it is
unchanged.

### 1.2 All 512 experts are one tensor, and the experts are its outermost dimension

`src/models/qwen4exp.cpp:254-256` creates the router and the down-projection and
delegates gate/up:

```cpp
        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", il), { n_embd, n_expert }, flags);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, flags);
        create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, flags);
```

and `llama_model_base::create_tensor_gate_up_exps`
(`src/llama-model.cpp:3222-3228`) creates `{n_embd, n_ff_exp, n_expert}` for
gate and up separately when no fused `ffn_gate_up_exps` is present in the GGUF.

For this model (`logs/long-context-120k-lazyoff/server.log:113`, `:136-137`, and
the GGUF KVs at `:37-42`): **`n_embd = 2560`, `n_ff_exp = 640`,
`n_expert = 512`, `n_expert_used = 10`, 48 layers.**

Two consequences, and the first one is the single most favourable structural
fact in this document:

- **Expert `e` is the contiguous byte range `[e·nb[2], (e+1)·nb[2])` of the
  tensor.** Experts are `ne[2]`, the outermost dimension, so a per-expert split
  is a *contiguous* split — unlike the row-splits `ggml_backend_sycl_split_buffer_type`
  does (`ggml/src/ggml-sycl/ggml-sycl.cpp:1181-1183`), which slice `ne1` and
  need row padding (`:1240-1245`). Nothing needs to be strided, gathered or
  re-laid-out to move one expert.
- **The slab is 1.88 MiB and it is the natural dial.** `08-long-context-vram-budget.md`
  §2.3's `expert_vram(N) = 46,200 − 962.5·N` MiB gives 962.5 MiB per layer for
  its three expert tensors, so per (layer, expert) the granule is
  962.5 / 512 = **1.8799 MiB**, and the model has 48 × 512 = **24,576** such
  slabs. Against `-ncmoe`'s 962.5 MiB steps (doc 08 `:719-728`) and the existing
  cache's 90.58 MiB slot steps, this is a **512x finer dial than the production
  knob**, which is worth stating because it is what makes a budget-driven
  placement exact rather than rounded.

Two details of this GGUF that a pool design has to respect, read from the
loader's own debug output (`logs/phase3-vram-check.log:278-285`):

```
create_tensor: loading tensor blk.0.ffn_down_exps.weight
tensor blk.0.ffn_down_exps.weight (450 MiB iq4_nl) buffer type overridden to SYCL_MoE_Cached
create_tensor: loading tensor blk.0.ffn_gate_up_exps.weight
create_tensor: loading tensor blk.0.ffn_gate_exps.weight
tensor blk.0.ffn_gate_exps.weight (256 MiB iq2_s) buffer type overridden to SYCL_MoE_Cached
create_tensor: loading tensor blk.0.ffn_up_exps.weight
tensor blk.0.ffn_up_exps.weight (256 MiB iq2_s) buffer type overridden to SYCL_MoE_Cached
```

- **There are three expert tensors per layer, not two**, so **144
  `MUL_MAT_ID` nodes per token** — the `ffn_gate_up_exps` line is only the
  `TENSOR_NOT_REQUIRED` probe at `src/llama-model.cpp:3223` returning `nullptr`,
  not a fused tensor. This confirms the profiler's hardcoded `144`
  (`ggml-sycl.cpp:5196`, default `moe-cache.cpp:650-652`) and doc 05 §10.5's 78
  hooked nodes at `-ncmoe 26` (26 × 3). Worth stating because the fused variant
  exists in the codebase and reading the probe line as a hit halves the node
  count and the round-trip budget.
- **The three tensors are at different quants** — 450 MiB `iq4_nl` down,
  256 MiB `iq2_s` gate and up; 450 + 256 + 256 = 962 MiB, reproducing doc 08's
  962.5 to 0.05%. So per-expert slabs are 0.879 / 0.500 / 0.500 MiB and a pool
  must keep **per-tensor strides**, not one model-wide maximum — padding to the
  largest would waste 43% of the gate/up pools. The existing cache already does
  exactly this (`slot_stride = src0->nb[2]` per tensor, `moe-cache.cpp:302`,
  with the rationale at `:295-301`), which is one more reason §2.5 models the
  static pool on it.

### 1.3 So the proposal in its precise form

Three knobs, of which today's configuration is the degenerate case:

| knob | today | proposal |
|---|---|---|
| whole GPU-resident layers | `48 − N` (`-ncmoe N`) | optional, can be 0 |
| GPU-resident individual experts | 0 | `S` slabs of 1.8799 MiB, chosen by measured usage |
| everything else | host | host |

`S = 0` **is** `-ncmoe`. That is a useful property: the proposal strictly
generalizes the production knob, so it can be built to fall back to it exactly.

---

## 2. Does this need new infrastructure? Mostly no, and that is the surprise

### 2.1 Nothing in the loader path can do it (§1.1), and there is no "partially-CPU-backed buffer"

Checked directly rather than assumed. `ggml_backend_buffer_i`
(`ggml/src/ggml-backend-impl.h:46-70`) has no notion of a buffer whose address
range is partly device and partly host; `get_base` returns one pointer. The one
in-tree mechanism that backs a single tensor with several allocations is the
SYCL split buffer type, and it is instructive precisely because of how it has to
cheat: `ggml_backend_sycl_split_buffer_get_base` returns
**`(void *)0x1000` — "a dummy address and never dereferenced"**
(`ggml/src/ggml-sycl/ggml-sycl.cpp:1206-1211`), with the real per-device
pointers hidden in `tensor->extra`
(`ggml_tensor_extra_gpu`, populated `:1214-1260`). Every op that can see such a
tensor has to know about it, which is why plain matmul asserts against it for
`dst`/`src1` (`ggml-sycl.cpp:3351-3352`) — and **`MUL_MAT_ID` does not support
it at all** (no split handling anywhere in `ggml_sycl_mul_mat_id`,
`ggml-sycl.cpp:5250-5531`).

**Conclusion: a per-expert-split *buffer type* is the wrong shape.** It would
require teaching the loader sub-tensor placement, teaching `ggml_gallocr` and
`ggml_backend_sched` about a tensor with two homes, and teaching `MUL_MAT_ID`
about `tensor->extra` — all of which the split buft precedent shows costs a lot
and buys nothing here, because the only op that ever reads these tensors is
`MUL_MAT_ID`.

### 2.1b There *is* one mechanism in the tree that can express an expert-axis split — and it is still not the answer

Worth recording because it is the best candidate for "does something support
this already," and because the reasons it fails are specific rather than
general. `ggml/src/ggml-backend-meta.cpp` — the meta backend behind
`--split-mode tensor` (`src/llama.cpp:193-220`) — splits one logical tensor
across N devices along an arbitrary axis, declared per tensor name by
`llama_meta_device_get_split_state` (`src/llama-model.cpp:371-730`). Its split
axes include `GGML_BACKEND_SPLIT_AXIS_2`
(`ggml/include/ggml-backend.h:362-375`), it creates a separate `ggml_tensor` per
device for one logical tensor (`ggml-backend-meta.cpp:1180-1230`), it scatters
one GGUF tensor's bytes across those buffers at load
(`:1482-1504`), `GGML_OP_MUL_MAT_ID` is in its propagation table
(`:920-923`), and `handle_mul_mat` explicitly accepts an axis-≥2 split of
`src0` against a mirrored `src1` — `ggml-backend-meta.cpp:600-604`:

```cpp
        // batched matmul with the batches split across devices and a replicated activation
        if (src_ss[0].axis >= GGML_BACKEND_SPLIT_AXIS_2 && src_ss[0].axis < GGML_MAX_DIMS &&
                src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
```

Three independent reasons it does not solve this proposal, each checked:

1. **CPU is excluded from the device list by construction.** `src/llama.cpp:196-200`
   skips any device whose buffer type is `ggml_backend_cpu_buffer_type()` "for
   tensor parallelism". The mechanism is for GPU↔GPU, and a CPU/GPU expert split
   is the entire point here.
2. **Nothing rebases the router ids, so the shape is permitted but not
   correct.** `ggml-backend-meta.cpp:557` asserts the ids input is mirrored:
   `GGML_ASSERT(tensor->src[2] == nullptr || src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);`
   Every device therefore receives the global ids `0..511` while holding a
   locally-truncated `ne[2]`, and `MUL_MAT_ID`'s `base + i02*nb02` arithmetic
   (§2.2) would index the wrong expert or past the end. An expert-axis split of
   an `*_exps` tensor is expressible in the split-state algebra and wrong at
   execution.
3. **The policy never produces it anyway.** For expert tensors the table returns
   `AXIS_1` for gate/up/gate_up and `AXIS_0` for down
   (`src/llama-model.cpp:561-572`) — splits *within* each expert, not across
   experts. `AXIS_2` for an `*_exps` tensor appears nowhere.

So: the "one tensor, many buffers, sliced on the expert axis, scattered at load"
machinery exists, and if this proposal is ever wanted for a multi-GPU box it is
the right place to look. For a CPU/GPU split on one card it is not, and the
id-rebasing gap (2) is the same gap §2.3 identifies in the kernel signature —
**the blocker is per-expert id/base resolution at the op, in both designs.**

### 2.2 The right shape is op-level dispatch, which is what `moe-cache.cpp` already is

The existing SYCL cache is the in-tree proof that op-level per-expert dispatch is
the cheap way to do this, and reading how it does it is the fastest route to a
static design. Its mechanism, in one line: **keep the tensor's placement
completely static and hand the unmodified GEMV a different base pointer and a
remapped index array.**

- Hot tier: one `sycl::malloc_device` pool per cached tensor
  (`moe-cache.cpp:220`), `slot_stride = src0->nb[2]` — exactly one expert
  (`moe-cache.cpp:302`, field `:173`), `n_slots` from
  `GGML_SYCL_MOE_CACHE_SIZE`, default **64** (`moe-cache.cpp:271-282`).
- Indirection: device `int32[n_experts] slot_table` (expert → slot or −1) and
  its reverse `expert_for_slot` (`moe-cache.cpp:199-213`, allocated `:220-232`).
- Dispatch: the plan kernel rewrites the router ids into *slot* indices
  (`moe-cache.cpp:964-969`), and the op handler calls the ordinary GEMV with
  three substituted arguments — `ggml-sycl.cpp:5200-5205` passes
  `pool_base`, `remapped_ids` and `slot_stride` where it would pass
  `src0->data`, `ids` and `nb02`. The kernel itself is untouched: its arithmetic
  is `vx_base + i02*expert_weight_stride` (`mmvq.cpp:2713-2715`), which now
  indexes slots. Rationale stated outright at `moe-cache.hpp:102-112`.

So the cache "solves" per-expert placement already — dynamically. **A static
version is the same mechanism with the policy deleted**: allocate the pool once,
fill `slot_table` once from a profile, never run the plan kernel, never evict,
never copy during inference. Everything doc 13 §4 found fatal to live migration
(no buft mutation primitive, stale graph reuse, `cpy_tensor_async` NULL'd at
`ggml-sycl.cpp:6861-6863`, and the twice-corrupted async barrier recorded at
`moe-cache.cpp:350-372`) is structurally absent, because **nothing moves after
load**. That is the proposal's central claim and the code supports it.

### 2.3 The one thing the cache's mechanism cannot do, and why it matters

The fused GEMV takes **one** weight base pointer. A *fully*-resident cache never
needs two (a miss is resolved by copying into the pool first, then everything is
in the pool). A *statically partial* placement needs two by construction:
resident experts from the pool, non-resident experts from wherever they live.

Three ways out, in increasing cost:

1. **Two GEMV calls per node** over disjoint route subsets (pool base + slot
   ids; cold base + expert ids). Needs a device-side partition of the routes —
   one tiny kernel, no changes to `mmvq.cpp`.
2. **Per-route byte offsets** instead of slot indices, with `stride = 1`. Zero
   kernel changes in principle, but `i02` is `int` and is cast `(size_t)`
   (`mmvq.cpp:2713-2715`), so a cold base below the pool base becomes a
   catastrophic unsigned offset. **Not recommended** — it is a correctness
   landmine for a one-line saving.
3. **One extra argument** to `ggml_sycl_mul_mat_vec_q_id` (a second base, or a
   per-route base array). Smallest and clearest, but it edits a shared kernel.

**And there is a fourth option that avoids the question entirely, which is what
§2.4 recommends.**

### 2.4 The piece that changes the estimate: doc 05's hybrid `MUL_MAT_ID` hook is live in this tree

This was not in the brief and it is the most consequential thing found this
pass. The spike doc 05 §10 measured is **present, uncommitted, in the working
tree**: `ggml/src/ggml-mmid-hybrid.h` (65 lines),
`ggml/src/ggml-sycl/mmid-hybrid.{cpp,hpp}` (487 + 14),
`ggml/src/ggml-backend.cpp:2515` (the zero-initialized table), registration at
`ggml/src/ggml-sycl/ggml-sycl.cpp:528-529`, and the CPU-side hook at
`ggml/src/ggml-cpu/ggml-cpu.c:1546-1555`, `:1659-1698`, `:1700-1725`,
`:1803-1807`.

What it does today, per `MUL_MAT_ID` node whose weights are host-resident: CPU
thread 0 picks a *fraction* of the node's routed rows, hands them to the GPU,
the other threads compute the remaining rows as usual, and thread 0 host-waits
after doing its own share — so the node costs `max(GPU, CPU_misses) + latency`.
The row selection is the whole of `ggml-cpu.c:1672-1691`:

```cpp
            int n_gpu = (int) (ggml_mmid_hybrid_fraction()*(float) n_routed + 0.5f);
            ...
                int k = 0;
                for (int64_t iid1 = 0; iid1 < ids->ne[1] && k < n_gpu; ++iid1) {
                    for (int id = 0; id < n_ids && k < n_gpu; ++id) {
                        eids[k] = *(const int32_t *) ((const char *) ids->data + iid1*ids->nb[1] + id*ids->nb[0]);
```

and the CPU side then skips exactly those rows (`:1705-1715`).

**A usage-based static placement is this hook with one predicate changed**: take
the row if `resident(src0, eids[k])`, instead of taking the first `n_gpu` rows.
And the GPU chain is already parameterized correctly —
`mmid-hybrid.cpp:393-394` submits

```cpp
    sycl::event ev = submit_chain(s, src0->type, src0->data, src0->nb[2], s->d_ids,
                                  n_rows, ncols, nrows_out, &ok);
```

i.e. `(weight base, per-expert stride, device id array)`. Pointing those at a
static VRAM pool, its `slot_stride`, and slot ids is a **three-argument
change**. §2.3's two-base problem does not arise: non-resident rows never reach
the GPU, they stay on the CPU path that already handles them.

Three further properties of the hook that matter:

- **It is already correctness-evidenced on this model.** Doc 05 §10.3: output
  **byte-identical at every fraction including 1.0**, where all 10 routed rows
  of all 78 hooked nodes ran on the GPU — 189 characters, `md5 625719b63796`.
  Given `PLAN.md:1004-1018`'s later finding that greedy output on this stack is
  *not* generally run-to-run reproducible, byte-identity here is a strong result
  rather than a weak one, and it was obtained before that finding, on the same
  binary, across fractions rather than across runs.
- **It is structurally decode-only**, which resolves §5's prefill question for
  free: `GGML_MMID_HYBRID_MAX_ROWS` is 64 (`ggml-mmid-hybrid.h:34`) and the
  eligibility test requires `n_routed = ids->ne[1]*n_ids <= 64`
  (`ggml-cpu.c:1664-1670`), i.e. **at most 6 tokens per ubatch** for
  `n_expert_used = 10`. A prefill ubatch cannot enter it.
- **Its per-node cost model is measured, on this hardware, in both regimes** —
  and this is what makes §3's arithmetic real rather than a projection
  (doc 05 §10.4-§10.5):

  | fit | value |
  |---|---|
  | GPU, weights **in VRAM** | `12.2 us + 3.31 us/row` |
  | GPU, weights in pinned host | `17.0 us + 22.9 us/row` (21.9 GB/s, PCIe-limited) |
  | GPU, weights in pinned host, *real* decode access pattern | **~200 us/row** (~3.1 GB/s, cause unexplained, doc 05 §10.5) |
  | CPU, 32 threads, DDR5 | `24 us + 16.3 us/row` |
  | dispatch + host wait, steady state | **6-8 us** |

  Doc 05 §10.6's verdict — "the cache is not one of several things that makes
  this design work, it is the only thing" — is precisely this proposal's
  premise. The spike's curve was monotonically *inverted* because every diverted
  row paid 22.9-200 us of PCIe instead of 3.31 us of VRAM. **A static
  usage-based pool is the missing half of the spike.**

### 2.5 The resulting design, in one paragraph

Place all 48 layers' expert tensors on the host (`-ncmoe 48`, ordinary CPU
buffer type, so prefill keeps today's exact behaviour — §5). Allocate one static
`sycl::malloc_device` pool sized to the expert VRAM budget, and fill it once at
load from a precomputed `(layer, expert) → slot` table derived from a usage
profile; the fill is a plain `q.memcpy` per slab from the already-loaded host
tensor, done once, synchronously, before the first graph is built. At decode the
existing hybrid hook diverts exactly the resident rows to the GPU against that
pool and leaves the rest to the CPU threads. **No admission, no eviction, no
plan kernel, no cross-queue barrier, no tensor re-pointing, no async anything.**

---

## 3. The decisive arithmetic: the value is very nearly `h − f`

This section is the reason to read the document. Notation: `f` = fraction of all
24,576 expert slabs held in VRAM; `h` = fraction of *routed* expert rows at
decode that hit the resident set. Uniform routing means `h = f`.

### 3.1 Host-side expert work, both schemes, per token

Routed rows per token = 48 layers × 10 = **480**.

- **Today, `-ncmoe N`**: `10·N` host rows. At `N = 24`, **240 rows**, and the
  expert tensors of those layers occupy `(48−N)/48` of expert VRAM, i.e.
  `f = 1 − N/48 = 0.5`.
- **Proposal**: `480·(1−h)` host rows at residency `f`.

Set them equal: `480(1−h) = 480(1−f)` ⟺ `h = f`.

> **Under uniform routing, per-expert placement moves exactly zero CPU work off
> the host.** It cannot: the same number of routed rows miss VRAM either way.
> What it *adds* is 144 hooked nodes per token instead of `3N`, each paying the
> 6-8 us round trip, and it gives up the one thing `-ncmoe` does well — a whole
> node's 10 experts computed as one clean 32-thread CPU GEMM with no dispatch at
> all.

That is the honest framing, and it is the opposite of how the idea reads
intuitively. **Almost every millisecond this proposal saves is bought with
`h − f`.** The one gain that is not is the intra-node CPU/GPU overlap (§3.2),
which is a property of the hook rather than of the placement and would be
available to a fraction-based split too — doc 05's spike could not collect it
only because its diverted rows paid PCIe instead of reading VRAM.

### 3.2 Priced against the measured decode token

Anchors, all measured in this repo: 8K decode at the recommended config is
**25.36 tok/s = 39.43 ms/token** (`PLAN.md:1269`, sparse-FA arm); each
host-resident expert *layer* costs **+0.4 ms/token** (`PLAN.md:897-899`); and
doc 05 §10.5's per-node fits are `CPU(k) = 24 + 16.3k` us,
`GPU(k) = 12.2 + 3.31k` us (VRAM weights), with an 8 us dispatch-and-wait.

**Reconciling the two cost anchors, because they look contradictory and are
not.** Doc 05's fit makes a 10-row CPU node 187 us, so `-ncmoe 24`'s 72 CPU
nodes are **13.46 ms/token**. But `PLAN.md:897-899`'s +0.4 ms per host-resident
*layer* implies only 133 us per node. Both hold: 0.4 ms/layer is the **marginal**
cost of moving a layer from GPU to CPU, so a GPU-native expert node costs
187 − 133 = **~54 us**, and `-ncmoe 24`'s total expert cost is
72 × 187 + 72 × 54 = **17.35 ms** of a 39.43 ms token. **Inference**, but it is
the only reading under which both measurements are simultaneously true, and the
54 us figure is plausible for a resident 10-expert GEMV on this card.

Per-node cost under the hook is doc 05 §10.6's own model,
`max(GPU(10h), CPU(10(1−h))) + 8` us — a `max`, not a sum, because the GPU work
is submitted before the CPU threads run their miss rows and host-waited after
(`ggml-cpu.c:1693`, then `:1803-1807`). Doc 05 §10.5 verified the overlap
directly: the thread-0 CPU gap collapses to 4 us at fraction 1.0.

| `h` | CPU side | GPU side | per node | ×144 nodes |
|---|---|---|---|---|
| 0.5 (uniform at `f=0.5`) | 105.5 | 28.8 | 113.5 us | 16.34 ms |
| 0.6 | 89.2 | 31.9 | 97.2 us | 14.00 ms |
| 0.7 | 72.9 | 35.4 | 80.9 us | 11.65 ms |
| **0.8** | 56.6 | 38.7 | **64.6 us** | **9.30 ms** |
| **0.89** (crossover) | 41.8 | 41.7 | **49.8 us** | **7.17 ms** |
| 1.0 | 4 | 45.3 | 53.3 us | 7.68 ms |

Note the curve has an **interior minimum at `h ≈ 0.89`** and gets slightly worse
above it: past the crossover the GPU side binds, and the hook's GPU path is not
free — it copies activations host→device and results back per node, so it never
reaches the ~54 us of a natively resident node. **Pushing `h` past ~0.9 buys
nothing; if VRAM allowed full residency you would not use this mechanism at
all.**

Against today's 17.35 ms of total expert cost in a 39.43 ms token, and bounded
two ways by whether the intra-node CPU/GPU overlap actually materializes:

| `h` | **overlap holds** (`max + 8`) | | **overlap fails** (`CPU + GPU + 8`) | |
|---|---|---|---|---|
| | expert cost | token → tok/s | expert cost | token → tok/s |
| 0.5 | 16.34 ms | 41.07 → 24.4 (**0.96x**) | 20.48 ms | 42.56 → 23.5 (**0.93x**) |
| 0.7 | 11.65 | 33.73 → 29.6 (**1.17x**) | 16.75 | 38.83 → 25.8 (**1.02x**) |
| **0.8** | **9.30** | **31.38 → 31.9 (1.26x)** | 14.88 | 36.96 → 27.1 (**1.07x**) |
| **0.89** | **7.17** | **29.25 → 34.2 (1.35x)** | 13.18 | 35.26 → 28.4 (**1.12x**) |
| 1.0 | 7.68 | 29.76 → 33.6 (1.32x) | 12.03 | 34.11 → 29.3 (1.16x) |

**The overlap is the design's whole structural trick and doc 05 measured it, so
the left pair is the expectation and the right pair is the floor.** Two
supporting facts. First, the hook submits GPU work before the CPU threads run
their miss rows (`ggml-cpu.c:1693`) and host-waits after
(`:1803-1807`), and doc 05 §10.5's thread-0 CPU gap collapsing to 4 us at
fraction 1.0 is direct evidence the two really do proceed concurrently. Second,
today's arrangement provably does *not* overlap: a `ggml_backend_sched` split is
a backend plus a contiguous node range (`ggml/src/ggml-backend.cpp:775-784`) and
splits are executed in order, so `-ncmoe 24`'s alternating CPU and SYCL splits
are sequential. **Moving the GPU expert work inside a CPU-backend node is
therefore a gain that does not depend on skew at all** — it is the one part of
this proposal that is not a bet on §4's unmeasured number. The risk on the right
column is that 144 dispatches per token behave worse than doc 05's 78 (§9 item
5).

Two consequences worth stating plainly, because the intuitive pitch ("keep the
hot experts on the GPU") sounds unbounded and is not:

- **Break-even is `h ≈ 0.46` if the overlap holds and `h ≈ 0.67` if it does
  not.** At `f = 0.5`, uniform routing (`h = 0.5`) is within ±5% of today on
  both columns — i.e. **at uniform routing this proposal is worth approximately
  nothing**, which is §3.1's arithmetic showing up in measured units.
- **The decode ceiling is 1.12x-1.35x**, reached at `h ≈ 0.89`, capped by the
  17.35 ms of expert cost that exists today.

### 3.3 The larger prize is VRAM, and it lands on a *measured* gap

The table above holds `f = 0.5` (the same expert VRAM as `-ncmoe 24`). The more
interesting question is the other direction: **at what `f` can a given `h` be
held?** Because expert VRAM is the residual term in doc 13 §1.3's budget, and
doc 10's ADDENDUM measured what that residual costs:

> Of the 8.0 ms/token gap between the 8K and 300K sparse arms, **~3.5 ms is
> depth** and **~4.5 ms is VRAM** — 300K only fits at `-ncmoe 32`, eight more
> expert layers on the host (`PLAN.md:1282-1286`).

`-ncmoe 32` is `f = 0.333`; the 600K arm's `-ncmoe 38` is `f = 0.208`. The key
structural point: **under this proposal the expert cost is a function of `h`
only, not of `f`** — `f` merely determines whether a given `h` is attainable. So
the deeper the configuration, the more the proposal displaces:

| depth | today | `f` today | routed rows on host today | today's expert cost | at `h=0.8` (overlap / no overlap) |
|---|---|---|---|---|---|
| 8K | `-ncmoe 24` | 0.500 | 240 of 480 | 13.46 + 3.89 = **17.35 ms** | 9.30 / 14.88 ms |
| 300K | `-ncmoe 32` | 0.333 | 320 of 480 | 17.95 + 2.59 = **20.54 ms** | 9.30 / 14.88 ms |
| 600K | `-ncmoe 38` (`-ub 1024`) | 0.208 | 380 of 480 | 21.32 + 1.62 = **22.94 ms** | 9.30 / 14.88 ms |

Projected end to end, at `h = 0.8`:

| depth | measured today | overlap holds | overlap fails |
|---|---|---|---|
| 8K | 39.43 ms, **25.36 tok/s** | 31.38 → **31.9** (1.26x) | 36.96 → **27.1** (1.07x) |
| 300K | 47.44 ms, **21.08 tok/s** | 36.20 → **27.6** (1.31x) | 41.78 → **23.9** (1.14x) |
| 600K | 80.52 ms, **12.42 tok/s** | 66.88 → **15.0** (1.20x) | 72.46 → **13.8** (1.11x) |

**Inference, and it is the strongest argument for the proposal:** if a hit rate
near 0.8 survives down to `f ≈ 0.21-0.33`, the expert term stops scaling with
depth-driven VRAM pressure altogether, which removes the ~4.5 ms/token doc 10
measured as the *VRAM* half of the 300K decode gap — the term that makes 300K
miss the relative bar for a reason that has nothing to do with depth. And at
600K, where `PLAN.md:1297-1300` names the culprit as "38 layers of CPU expert
GEMM on the same cores as `set_input_qsa`'s single-threaded O(`n_kv`) grouping
scan", cutting host-resident routed rows 380 → 96 attacks the *contention* half
of that hypothesis from the other side — an upside the table above does not
count, because that contention has never been isolated (doc 13 §7 item 1).
**600K still misses the 17 tok/s floor on these projections.** That should be
said before anyone treats this as the 600K fix; it is a 1.11-1.20x on an arm
that needs 1.37x.

**This is the material difference from doc 13's proposal.** That one delivered a
measured 1.00x at 600K and ~5-6% elsewhere (doc 13 §9.4). This one targets a
term that has been decomposed and measured, at every depth, and the same
mechanism improves both the floor and the VRAM budget at once.

### 3.4 The floor/slope tension, restated honestly

`PLAN.md:791-800` and `12-decode-depth-scaling.md:429-435` record a standing
warning: lowering the floor makes the *relative* depth bar harder. It applies
here and should not be buried. With the sparse-FA slope of
**0.01174 ms/1,000 tokens** (`PLAN.md:1283-1284`) and a floor cut 39.43 → 31.38
ms at `h = 0.8`, the pure depth term at 300K moves from +8.9% to +11.2%.

But §3.3's table shows the *configuration* term shrinking much further than the
floor does, and the relative arithmetic comes out ahead: 300K today is
47.44 / 39.43 = **+20.3%** against the 8K arm; at `h = 0.8` with overlap it is
36.20 / 31.38 = **+15.4%**, and without overlap 41.78 / 36.96 = **+13.0%**. So
the net relative position **improves**, which is the first candidate in this
project for which a floor reduction has not made the bar harder — because it
removes a depth-correlated term (`-ncmoe` rising with depth) at the same time as
the floor. Still outside a 5% ask at 300K; better, not solved. And it depends
entirely on `h` holding at low `f`, which is §4's unmeasured quantity.

---

## 4. The premise is not measured — and the number being cited is about a different model

This is the load-bearing weakness, and the brief's framing of it needs
correcting in two places.

### 4.1 "Top 10% of experts take ~80% of hits" is not this fork's measurement, and not this model

The figure traces to exactly one place in this repo,
`05-hybrid-cpu-gpu-mul-mat-id-scoping.md:69-71`, quoting **RFC #24528's body
verbatim**:

> Fill is decode-only (prompt routing is far flatter and thrashes the cache);
> decode routing is skewed enough to make this work — measured on
> **Qwen3.5-122B**, the top 10% of experts take ~80% of hits (independent
> corroboration in #20757: Gini ≈ 0.76, ~99% simulated hit rate at 69% expert
> budget).

Three corrections. (a) It is **Qwen3.5-122B**, not Qwen3.8-Flash-Next; this
model has 512 experts per layer with 10 used, a far larger and sparser routing
space. (b) It is **RFC #24528 (`leloch`)**, a different project from the
`GenerelSchwerz/llama.cpp` fork that `docs/research/03` maps and that
`00-background.md` §1 describes — the brief attributed it to the latter. (c) The
`00-background.md` §1 reading of the *fork's* own results points the other way:

> This requires either a skewed expert-selection distribution or (more simply,
> and apparently what dominates in the wiki's own 16GB recipes) a cache that's
> *large relative to the live expert working set* — "dynamic residency beats
> static placement" is doing most of the work, not exotic hot-expert
> exploitation. (`docs/00-background.md:35-38`)

That sentence is a direct warning about this proposal: if the fork's wins come
from residency-versus-placement rather than from skew, a *static* hot set
inherits none of them. Note also the RFC's own parenthesis — "prompt routing is
far flatter" — which is §5's subject.

### 4.2 This project's own "no locality" claim was retracted, so it is not evidence either way

`PLAN.md:517-522` argued the opposite of skew:

> since every decode token routes to a fresh random 10-of-512 expert set in each
> of 24 CPU layers, the working set is the whole 52.8 GiB with no locality

But that reasoning was **withdrawn the same day**: the paging it explained was
`per_layer_token_embd`, not experts — `PLAN.md:626-629` records "reproducing doc
11 §3.3's '0.36 MB/s' to the decimal, **which retires its reading of that number
as expert weights still loading**", and §7 of doc 11 is flagged as
read-before-§3.3. So the "no locality" statement is a retracted inference, not a
measurement. **Expert routing skew for `qwen4exp` is genuinely unknown in this
repo, in both directions.**

One real datum, and it is suggestive rather than conclusive:
`PLAN.md:893-899` found that **degenerate output flattered decode by ~13
ms/token** at `-ncmoe 32` "by routing every step to the same 10-of-512 experts
in each CPU layer, collapsing the host working set." That is a measurement that
extreme skew is worth ~13 ms/token on this hardware — an upper bound on this
proposal's prize obtained by accident, and it brackets §3.2's 9.6 ms nicely.

### 4.3 There is no per-expert counter anywhere, including in this project's own cache

The brief hoped the existing cache's "LRU/frequency-decay admission policy"
could be reused as a data source. **It cannot, because that policy does not
actually count hits.** `expert_frequency` is a device `uint32[n_experts]`
(`moe-cache.cpp:202`, allocated `:223`), the decay function is
`moe-cache.cpp:396-407`, and its **only** write outside zero-init is on
admission — `moe-cache.cpp:931-932`:

```cpp
                        expert_frequency[eid]      = 1;
                        expert_epoch[eid]          = step;
```

Hits take the `continue` at `moe-cache.cpp:856` and touch neither. Doc 13 §3.2
found this and called the resulting policy "neither LRU nor LFU"; the relevant
consequence here is narrower and worse: **the array is not a usage histogram and
never was.** The only host readback of any cache state is
`ggml_sycl_moe_cache_debug_n_misses` (`moe-cache.cpp:677-686`), aggregate; the
`MOE_PROFILE` state (`moe-cache.cpp:605-616`) is aggregate hit-rate only.

### 4.4 Nor can the router be read as a proxy

Checked, because it would have been free. `qwen4exp` creates only
`ffn_gate_inp` `{n_embd, n_expert}` and `ffn_gate_inp_shexp` `{n_embd}`
(`src/models/qwen4exp.cpp:254`, `:258`) — **there is no `ffn_exp_probs_b`
router-bias tensor** in this architecture's tensor list, and the load banner
reports `n_expert_groups = 0`
(`logs/long-context-120k-lazyoff/server.log:138`), so there is no
group-routing prior to exploit either. The only static signal available is the
router weight matrix itself, whose per-expert column norms are a weak and
untested proxy for selection frequency. **Option (b) of the brief has no tensor
to read.**

---

## 5. The prefill objection: inherited as neutrality, not as harm

Doc 13 §3.3's objection, quoted: the cache "deliberately sidesteps buft mutation
… and prefill bypasses it entirely … **This is disqualifying on its own**"
because TTFT was the binding metric.

### 5.1 Why prefill bypasses the cache today, exactly

Every cache entry point is gated on `ne12 == 1`, and `ne12` **is** the token
count: `ggml.c:3403` asserts `ids->ne[1] == b->ne[2]` for `MUL_MAT_ID`, and
`llm_graph_context::build_lora_mm_id` (`src/llama-graph.cpp:1550`) passes
`b->ne[2] == n_tokens`. The gates:

- `ggml-sycl.cpp:5299` — device-planned path: `if (ne12 == 1 && src0_cached && !force_legacy)`
- `ggml-sycl.cpp:5149` — the same test again inside it: `if (ne12 != 1) return false;`
- `ggml-sycl.cpp:5347` — legacy cache path: `if (ne12 == 1 && src0_cached) {`

**The threshold is literally one token, and it is not env-tunable.** What prefill
does instead is the `else` at `ggml-sycl.cpp:5437-5531`: counting-sort the routed
rows by expert, then one batched `ggml_sycl_mul_mat` per expert with
`src0_row.data = src0_original + i02*nb02` (`ggml-sycl.cpp:5492`) — i.e. the GPU
kernel dereferences the **pinned-host USM pointer** over PCIe, no staging copy,
cache untouched. The rationale is explicit at `ggml-sycl.cpp:5347-5354`:

```
        // Legacy (host round-trip) cache path -- decode only for now; see
        // PLAN.md Phase 3a. Prefill (the ne12 > 1 branch below) still falls
        // through unmodified when the buffer is cached: it reads src0
        // through the pinned-host pointer directly, which is correct (SYCL
        // USM host allocations are device-dereferenceable) just unoptimized,
        // matching the sibling project's own finding that a cache regresses
        // large-batch prefill -- deliberately not attempting that here.
```

So the bypass is a deliberate decision about *the dynamic cache*, made because a
wide unique-expert set per prefill op is the worst case for a small rotating
pool (`00-background.md:75-77`: the fork's own "regresses prefill 14-66%"), not
a property of per-expert placement as such.

### 5.2 Does a static placement inherit it? Yes — but as *exactly today's
### behaviour*, which is a different thing from doc 13's objection

§2.5's design keeps expert tensors in the **ordinary CPU buffer type**, which is
what `-ncmoe` already gives them, and the hybrid hook cannot fire above 6 tokens
per ubatch (`ggml-mmid-hybrid.h:34`, `ggml-cpu.c:1664-1670`). Therefore at
prefill the graph takes **bit-for-bit the path it takes today under `-ncmoe`**:
the scheduler's used-experts-only copy at `ggml/src/ggml-backend.cpp:1690-1774`
(doc 05 §3.2 documents this path, including its blocking `ggml_backend_synchronize`
at `:1721`), unchanged, with the static pool simply unused.

That splits doc 13's objection cleanly:

- **The "inert on prefill" half is inherited.** A static hot set does nothing for
  TTFT by default.
- **The "suspected negative on prefill" half is not.** There is no admission, no
  thrash, and no change of code path, so prefill cannot regress *from the
  mechanism*. The only prefill cost is the configuration change — going from
  `-ncmoe 24` to `-ncmoe 48` puts twice as many expert layers on the host
  path. `PLAN.md:900-901` measured the analogous step: `-ncmoe` 24 → 32 cost
  prefill **−4.7%** at `-ub 2048`. **Inference:** 24 → 48 is three times that
  step, so a **−10% to −15% prefill** expectation is the honest planning figure,
  and it is bounded, measurable in one run, and avoidable by keeping some whole
  layers GPU-resident (§1.3's `S = 0` generalization means the split between
  whole layers and hot slabs is itself a tunable).
- **And there is a clean later extension that turns prefill positive**, which
  doc 13's proposal had no analogue for: make the scheduler's used-experts-only
  copy (`ggml-backend.cpp:1690-1774`) skip experts already resident in the
  static pool. That is a strict reduction in PCIe traffic, needs no policy, and
  is only sound because the placement never changes. Out of scope for a first
  build; worth recording as the thing that makes the mechanism pay on both
  metrics.

**So: the objection is reduced from "disqualifying" to "a −10-15% TTFT line item
with a known fix, against a 1.12-1.35x decode and a VRAM saving."** Whether that
trade is acceptable depends on whether TTFT is still the binding metric —
`PLAN.md:1341` has 116K TTFT at 4 min 37 s and `PLAN.md:855` has 300K at 19 min
57 s, so it is still large, and this trade should be presented to the user rather
than assumed.

---

## 6. How to obtain the profile — one recommendation, not a menu

**Recommended: (a), an instrumented real-workload run, collected on the CPU
path, and baked into the load as a file.** Reasoning, then the recipe.

Why not (b), the router's own statistics: §4.4 — this architecture ships no
router bias tensor and no expert groups, so there is nothing to read.

Why not (c), periodic recomputation from recent logged usage: it buys nothing
this design needs and costs the property that makes it safe. Placement is
consumed at `load_tensors` time (doc 13 §1.1), so "recompute periodically" means
"reload the model periodically", which is the `--context-tiers` machinery at
**219 s per switch** (`PLAN.md:1450`, `13 §9.5`) — and doc 13 §9.5's breakeven
arithmetic already showed a reload needs ~91,000 decoded tokens to repay a
2.4 ms/token gain. A profile refresh would have to repay the same 219 s for a
much smaller delta. Recompute **offline**, adopt at the next restart.

Why (a) is also cheap, and this is the part that makes it the answer: **the
profile can be collected with no GPU code, no kernel work, and no risk to any
measured path.** The CPU `MUL_MAT_ID` already reads every routed id on the host,
in a serial thread-0 block, for its own row grouping —
`ggml/src/ggml-cpu/ggml-cpu.c:1707-1723`:

```cpp
        for (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            for (int id = 0; id < n_ids; ++id) {
                ...
                const int32_t i02 = *(const int32_t *) ((const char *) ids->data + iid1*ids->nb[1] + id*ids->nb[0]);
```

An env-gated `++counts[tensor][i02]` there, plus an `atexit` dump keyed by
`src0->name` (which carries the layer, `blk.N.ffn_gate_exps.weight`), is the
whole instrumentation. **~40 LOC in one file, zero effect when the env var is
unset, and it runs inside any existing benchmark or server run.**

Recipe, and the RAM arithmetic matters given `PLAN.md:1493-1515`:

1. Run with **`-ncmoe 48`** so every expert tensor is on the CPU path and every
   routed id passes through the instrumented loop. **Use `-lzm auto`, not
   `-lzm off`**, for this run only: `-lzm off` pins the 27,466 MiB PLE table in
   host RAM (`PLAN.md:645-648`), and the profiling run does not care about
   decode speed. Host requirement is then ~46 GiB of expert tensors against
   `free -h`'s `available`, not the ~72 GiB `-ncmoe 48 -lzm off` would want on a
   123 GiB box with ~35 GiB already committed.
2. Use the real workloads, not a synthetic prompt — this project already has
   them: the 116K refactor prompt (`staging/work/bench300k/` has the 300K
   variant) whose routing is the actual production distribution. Collect
   **prefill and decode separately**, because RFC #24528 states outright that
   "prompt routing is far flatter", and a placement optimized on a
   prefill-dominated histogram would be optimized for the wrong phase. The
   decode histogram is the one that matters (§5.2: prefill does not use the
   pool).
3. Report, per (layer, expert): hit count; and derive the two curves that decide
   everything — **`h(f)`** (hit rate as a function of residency fraction, the
   cumulative-share curve) and its stability across two different prompts. The
   second is the real risk: a placement tuned to one document that does not
   transfer is worthless.

**The go/no-go is then read straight off §3.2's table**: `h(0.5) < 0.67` is a
no-build (it is below break-even even if the overlap holds only partly);
`h(0.33) ≥ 0.8` makes §3.3's case at every depth; anything between is a
judgement call against the prefill cost in §5.2.

Baking it in: a plain text/JSON file of `(layer, tensor, expert) → rank`, passed
as a new CLI flag, read once before `load_tensors`. It is data, not code, so it
can be regenerated without rebuilding, versioned against the GGUF hash, and
diffed between workloads.

---

## 7. Composition with what landed today

Checked rather than assumed, as the brief asked.

### 7.1 Orthogonal to all four of today's fixes

- **Block-granularity QSA top-k** (`src/models/qwen4exp.cpp`,
  `src/llama-memory-hybrid-idx.cpp`; `PLAN.md:924-1034`) — operates on the
  indexer/attention path. No expert tensors, no `MUL_MAT_ID`.
- **Pooled indexer-key caching** (`src/llama-memory-hybrid-idx.{h,cpp}`;
  `PLAN.md:1101-1192`) — its only interaction is VRAM: `1,536·n_ctx` bytes, 180 /
  456 / 900 MiB at the three depths. It *competes* with the expert pool for the
  same budget, which §3.3 already accounts for by working in `f` rather than in
  absolute MiB.
- **Sparse attention restriction** (`ggml/src/ggml-sycl/fattn-sparse.{cpp,hpp}`;
  `PLAN.md:1194-1352`) — gated on `Q->ne[1] == 1`, i.e. the same decode-only
  regime, but on a disjoint op (`FLASH_ATTN_EXT`). The one real interaction is
  §3.4's floor/slope tension, and it runs in the favourable direction here.
- **The `-ub 2048` / `-lzm off` / `-lm none` config** — `-lzm off` is about
  `per_layer_token_embd` and "does not touch VRAM" (`PLAN.md:838-840`); `-lm
  none` governs mmap for the `-ncmoe` overrides. Both compose, but note
  §6's recipe deliberately uses `-lzm auto` for the *profiling* run only.

**No fix landed today assumes `-ncmoe`'s whole-layer granularity.** Grep-checked:
the only consumers of `LLM_FFN_EXPS_REGEX` are `common/arg.cpp:2835` and
`:4188` (the draft model) and `common/common.h:1153`, plus
`tools/server/server-context.cpp:1007` and `:1049` — the tier code, §7.2.

### 7.2 It composes with `--context-tiers`, and the composition is the right shape

This was raised mid-task and it holds up. `ctx_tier_set`
(`tools/server/server-context.cpp:1039-1057`) is exactly one place, and it does
three things: mutate `n_ctx`/`n_ubatch`/`n_batch`, then

```cpp
        overrides = ctx_tier_overrides;
        llm_add_n_cpu_ffn_overrides(next.n_cpu_moe, LLM_FFN_EXPS_REGEX, overrides);
```

i.e. it rebuilds the `-ncmoe` override list from scratch per tier, on top of the
user's preserved non-`-ncmoe` overrides (separated at `:1001-1013`). `-fit` is
already force-disabled because it writes the same array (`:996-999`).

So a per-tier expert *budget* drops straight in: `common_context_tier`
(`common/common.h`) gains a slab-count or MiB field, `ctx_tier_set` passes it
alongside the `n_cpu_moe` it already computes, and the switch's existing
`destroy()` + `load_model()` rebuilds every buffer anyway — **so the static pool
is re-sized for free by the reload, with no new lifecycle and no new hazard.**
Tier 0 would pick its hot set at `f ≈ 0.5`, tier 1 at `f ≈ 0.33`, from one
profile. And because §1.3 makes `-ncmoe` the `S = 0` case, a tier table can mix
the two forms during bring-up.

Two things this composition *improves* rather than merely tolerating:

- Doc 13 §9.3's tier-1 row is tight — 31,482 MiB computed, 533 MiB under the
  proven-good 32,015, with the compute buffer the one estimated term and a
  stated falsification at 8,001 MiB. A finer expert dial (1.88 MiB steps instead
  of 962.5) removes that squeeze: the budget can be met exactly instead of
  rounded, which doc 08 `:719-728` measured as up to 580 MiB of waste at the
  deepest `-ncmoe` configs.
- Doc 13 §9.4's honest verdict was "~5-6%, not a throughput fix." Its prize
  shrank because the deepest servable tier is only 6 `-ncmoe` steps from tier 0.
  A per-expert placement makes each tier *individually* faster (§3.2), so the
  tier feature's value and this one's add rather than overlap. **They are
  independent and worth having together**, which is how the user framed it.

**One caveat to flag:** `ctx_tier_init` warns but does not correct for
`-np > 1` with a non-unified cache (`:1015-1018`), and doc 10's ADDENDUM notes
sparse FA falls back to dense for a unified cache batching N sequences. A static
expert pool is indifferent to both — the hook is per-node, not per-sequence — so
it does not add to that pile.

---

## 8. Effort and risk, file by file

Assumes §2.5's design (static pool + the existing hybrid hook), not a new buffer
type. Phases are deliberately ordered so that each one produces a decision.

### Phase 0 — the measurement (do this first, and possibly only this)

| file | change | LOC | risk |
|---|---|---|---|
| `ggml/src/ggml-cpu/ggml-cpu.c` | env-gated per-(tensor, expert) counter in the existing thread-0 id loop (`:1707-1723`) + `atexit` dump | 40-60 | **LOW** |
| `staging/work/` | histogram analysis: `h(f)` curve, cross-prompt stability | 80-120 (Python, not shipped) | LOW |

**~40-60 LOC of C, no GPU, no kernel, no model code, one benchmark run.**
Answers the only question that matters. Note the host-RAM recipe in §6 — this is
the one step that needs a `-ncmoe 48` load, and `PLAN.md:1493-1515`'s standing
rule applies in full.

### Phase 1 — the static pool and the hook predicate (only if Phase 0 says yes)

| file | change | LOC | risk |
|---|---|---|---|
| new `ggml/src/ggml-sycl/expert-pool.{cpp,hpp}` | `sycl::malloc_device` pool, `(tensor, expert) → slot` table, one-time fill from host tensor data, profile-file parse; modelled on `moe-cache.cpp:172-232`, `:289-309` **with the policy deleted** | 200-280 | MEDIUM |
| `ggml/src/ggml-mmid-hybrid.h` | one API entry (`resident(src0, eid)` or let `dispatch` report which rows it accepted) | 15-25 | LOW |
| `ggml/src/ggml-cpu/ggml-cpu.c` | replace the fraction predicate (`:1672-1691`) and the fixed `hybrid_skip` count (`:1705-1715`) with a per-row residency test — the skip must become a per-row mask, not a prefix count | 50-80 | **MEDIUM** |
| `ggml/src/ggml-sycl/mmid-hybrid.cpp` | point `submit_chain` at `(pool_base, slot_stride, slot_ids)` instead of `(src0->data, src0->nb[2], expert_ids)` (`:393-394`); drop the host-USM usability check (`:355-372`) for resident rows | 60-90 | LOW |
| `common/arg.cpp`, `common/common.h` | `--expert-profile FILE`, `--expert-vram-mib N` | 40-60 | LOW |
| `ggml/src/ggml-sycl/CMakeLists.txt` | new source — **note the glob has no `CONFIGURE_DEPENDS`** (`PLAN.md:151-154`), force `cmake .` or the library links stale | 1 | LOW |

**~370-540 LOC, MEDIUM overall.** The single riskiest line is the
`hybrid_skip` prefix-count → mask change: today's code relies on the GPU having
taken *the first `k`* routed rows in iteration order
(`ggml-cpu.c:1709-1711`'s comment says so explicitly), and a residency predicate
breaks that invariant. Getting it wrong silently double-computes or drops routed
rows — which is exactly the class of bug that `moe-cache.cpp:767-775` and `:934-942`
record as "found the hard way (garbled/degenerate greedy-decode output on real
hardware)."

Correctness plan, and it must not use a cross-run diff (`PLAN.md:1129-1138`: the
forward pass itself is not reproducible): the oracle is **an in-graph / in-node
comparison** — run the node both ways within one call under an env flag and
compare `dst` rows, the pattern `LLAMA_QSA_POOL_CHECK` and
`GGML_SYCL_FA_SPARSE_CHECK` already established. Plus a cheap invariant assert
that every routed row was computed exactly once, which catches the mask bug
directly. And `--expert-vram-mib 0` must reproduce `-ncmoe` exactly, which is a
real regression gate rather than a smoke test.

### Phase 2 — optional, later

| item | LOC | risk |
|---|---|---|
| per-tier expert budget in `--context-tiers` (§7.2) | 30-50 | LOW |
| skip resident experts in the scheduler's prefill copy (`ggml-backend.cpp:1690-1774`) — turns prefill from neutral to positive (§5.2) | 80-150 | MEDIUM-HIGH |

**Is this novel kernel work? No.** That is the headline of this section. No new
GEMM, no new gather kernel, no change to `mmvq.cpp`, no async cross-queue
anything. It is a static allocation, a lookup table, a changed predicate in an
existing measured hook, and one retargeted function call. The hard part is not
the kernels; it is Phase 0's number and the row-accounting invariant.

---

## 9. Open questions, ordered by how much they would change the above

1. **What is `h(f)` for this model on this workload?** Everything in §3 is a
   function of it and it has never been measured (§4). Phase 0, ~40 LOC, no GPU.
2. **Does the placement transfer between workloads?** A hot set tuned on the
   116K refactor prompt that does not hold on a different document makes the
   whole idea a per-request optimization, which a static design cannot be. Two
   prompts, same instrumentation, one comparison.
3. **How flat is prefill routing really?** RFC #24528 asserts it; nothing here
   has checked it for `qwen4exp` at `-ub 2048`. It decides whether Phase 2's
   prefill extension is worth anything.
4. **What does `-ncmoe 48` actually cost prefill?** §5.2 extrapolates −10-15%
   from a measured −4.7% at 24 → 32. One run at `-ub 2048`, no code, and it
   bounds the whole trade.
5. **Does the hybrid hook's 6-8 us round trip hold at 144 nodes/token?** Doc 05
   measured it at 78 nodes under `-ncmoe 26`. 144 nodes is 1.15 ms of fixed
   dispatch cost, and more importantly §3.2's two columns differ entirely on
   whether the CPU/GPU overlap survives at that rate — which moves break-even
   from `h ≈ 0.46` to `h ≈ 0.67` and the ceiling from 1.35x to 1.12x. **This is
   the second most decisive unknown after item 1**, and doc 05's spike can
   measure it as it stands: run it at `-ncmoe 48` and read the per-node span
   against the CPU gap in its own instrumentation (doc 05 §10.5's table).
6. **Is `sycl::malloc_device` of ~11-23 GiB in one region even allocatable?** The
   existing cache allocates per tensor (`moe-cache.cpp:220`), 144 small pools,
   which is probably the right shape here too, but the 2 GiB cap the cached buft
   advertises (`moe-cache.cpp:80`, `:126-129`) is a hint that large single
   allocations have been a problem on this stack.
7. ~~The rope/quality gap past 262,144 is still open~~ (`PLAN.md:872-891`, doc
   13 §9.2). §3.3's 600K projection is throughput on output whose quality at
   that depth has never been validated, and the server cannot even accept such a
   prompt. Same sequencing error doc 13 flagged; it applies to this proposal's
   deepest claims too. **Closed 2026-09-12 by `15-yarn-and-long-context-rope.md`
   — see §11 below.** Unscaled extrapolation is coherent and accurate at
   305,759 and 612,689 tokens (doc 15 §4), and `llama-server` can now accept a
   >262,144-token prompt via a one-flag workaround (doc 15 §6.2) that does not
   touch quality. §3.3's 600K throughput number is therefore on output that is
   both servable and validated as good, not merely projected.

---

## 10. Recommendation

**Do Phase 0. Decide from its number. Do not build Phase 1 on the strength of a
figure measured on a different model.**

In order:

1. **Build the histogram** (§6, §8 Phase 0). ~40-60 LOC on the CPU path, no GPU
   code, no kernel, runs inside an existing benchmark, and it produces `h(f)`
   plus a cross-prompt stability check. This is the cheapest decisive
   measurement available in this project right now and it has never been taken —
   the same shape of gap doc 11 closed by finally profiling instead of
   estimating.
2. **Read §3.2's table against `h(0.5)` and §3.3's against `h(0.33)`.**
   `h(0.5) < 0.67` → the proposal is a *loss* or a wash and should be recorded
   as closed, with the arithmetic in §3.1 as the reason (uniform routing moves
   zero CPU work). `h(0.33) ≥ 0.8` → build Phase 1, because that combination is
   worth **1.07-1.26x at 8K and 1.14-1.31x at 300K** and removes the
   ~4.5 ms/token VRAM term doc 10 measured as half the 300K gap.
3. **Take the second cheap measurement while you are there** (§9 item 5): re-run
   doc 05's existing spike at `-ncmoe 48` with `GGML_SYCL_MMID_HYBRID_FRACTION`
   and read its own per-node instrumentation (doc 05 §10.5's node-span /
   CPU-gap / host-wait columns) at 144 nodes per token. That is the difference
   between §3.2's two columns — break-even `h ≈ 0.46` versus `h ≈ 0.67`, ceiling
   1.35x versus 1.12x — and it needs **zero new code**, only GPU time on a quiet
   box. Order it after item 1 only because item 1 needs no GPU at all.
4. **If built, build it as §2.5's design** — static pool plus the existing
   hybrid hook — and not as a buffer type (§2.1), not via the meta backend
   (§2.1b), and not as a modification of `moe-cache.cpp` (which stays untouched;
   its dynamic policy is not wanted here and its defects, doc 13 §3.2, are not
   worth inheriting). **And carry §7.2's per-tier budget into the design from
   the start** rather than retrofitting it: `ctx_tier_set`
   (`tools/server/server-context.cpp:1039-1057`) already rebuilds the placement
   per tier and the switch's reload rebuilds every buffer, so making the tier
   struct carry an expert-VRAM budget alongside its `n_cpu_moe` is 30-50 LOC and
   removes doc 13 §9.3's 533 MiB tier-1 squeeze as a side effect. The two
   features are independent and compose; neither substitutes for the other.
5. **Hold Phase 2's prefill extension back**, and price the `-ncmoe 48` prefill
   cost (§9 item 4) before committing, because TTFT is still minutes.
6. **Nothing here changes doc 13 §8's ordering of the items ahead of it.** The
   O(`n_kv`) host term in `set_input_qsa` is still the named next task and still
   the thing standing between 600K and the bar — although note §3.3's
   observation that cutting CPU expert rows 380 → 96 attacks the *contention*
   half of `PLAN.md:1297-1300`'s hypothesis from the other side, so the two
   tasks are complementary rather than competing.

The one-line version: **this is the right shape of idea, the infrastructure is
already 70% built and already measured, doc 13's three blockers genuinely do not
apply, and the prefill objection reduces from "disqualifying" to a bounded
−10-15% line item — but the entire prize is the amount by which expert usage is
non-uniform, that number is unknown for this model, the number everyone quotes
is about a different one, and it costs 40 lines and one benchmark run to find
out.**

---

## 11. Addendum, 2026-09-12 (later same day): §9 item 7 closed

`15-yarn-and-long-context-rope.md` closes the rope/quality gap this document's
§9 item 7 named. Full detail in `13-dynamic-context-aware-expert-placement.md`
§10 (written first, since doc 13's deepest claims depended on it more
directly); the short version for this document: unscaled RoPE extrapolation is
coherent and accurate at 305,759 and 612,689 tokens, "YaRN destroys this model"
was a stale-binary artifact rather than a real defect, and `llama-server` can
serve a >262,144-token prompt today via a one-flag workaround that does not
touch quality. §3.3's 600K projection in this document is therefore priced
against a depth that is both servable and validated as good.

**One dependency this document has that doc 13's addendum flags as unresolved:**
doc 13 §9.3's tier table and §9.4's value pricing (which this document's §5.2
and recommendation item 4 both cite — the 533 MiB tier-1 squeeze and the
600K `-ncmoe 38` arm) were derived under §9.2's now-reversed "no 600K tier"
premise and have not been re-derived with the third tier reinstated. Nothing
in *this* document's own arithmetic depends on which way that comes out — §3.3
prices the existing measured 600K arm directly, not doc 13's tier table — but
treat any future cross-reference to doc 13 §9.3/§9.4 as pointing at stale
numbers until that re-derivation happens.
