# Dynamic, Context-Aware CPU/GPU Expert Placement — Scoping

Date: 2026-09-12. Status: **scoped, NOT implemented, and the recommendation is
"not now, and not the version that was asked for."**

The proposal under examination, verbatim from the user:

> "Can we dynamically move experts to CPU (and back) based on needed context
> length? I imagine we do a prediction on the tokenization, then determine if
> we need the VRAM, and asynchronously move the experts."

Everything below was read directly in the working tree at
`/media/da3dsoul/Golias/AIProjects/Qwen3.8-Flash-Next-LlamaCpp-MoE-Cache-Arc-Experiments`
(llama.cpp checkout at `src/llama.cpp`, HEAD `6d9c82ea2`). Line ranges are
cited, not guessed. Where a claim connects facts established in separate files
rather than something one file states outright it is flagged **inference**.

Companion documents, all of which this one depends on and none of which it
repeats: `08-long-context-vram-budget.md` (the budget model),
`12-decode-depth-scaling.md` (the slope model and the host-term measurement),
`05-hybrid-cpu-gpu-mul-mat-id-scoping.md` §3.2/§3.4/§4 (the scheduler and SYCL
queue mechanics), `03-cuda-fork-moe-cache-outline.md` (the reference design),
and `PLAN.md`'s 2026-09-12 sparse-attention addendum (the 600K measurement this
proposal is trying to fix).

---

## 0. One-paragraph summary of the finding

**The proposal is implementable, but it does not fix the problem it was aimed
at, and the mechanism it assumes exists does not exist.** Three findings, in
descending order of importance. (1) **Dynamic expert placement cannot improve
600K at all** — at 600K the deep configuration *is* the configuration, so
`-ncmoe 38` is paid either way; what dynamic placement buys is that *shallower*
requests in a 600K-capable server stop paying the worst-case tax, worth
**1.12x at 8K and ~1.32x at 300K, and exactly 1.00x at 600K** (§5.3). The
600K bar miss (12.42 tok/s against a 17-25 band) has a different, already-named,
already-measured cause — the single-threaded O(`n_kv`) host term in
`set_input_qsa` — and fixing that one term plausibly closes 600K outright
(§5.4). (2) **ggml has no mechanism to move a loaded tensor between buffer
types, and the graph-reuse path would not notice if you forced one** — both
confirmed by exhaustive grep and by the two hard asserts at
`ggml/src/ggml-backend.cpp:2128-2141`; SYCL additionally has `cpy_tensor_async`
NULL'd out (`ggml/src/ggml-sycl/ggml-sycl.cpp:6861-6863`), so the word
"asynchronously" in the proposal has no available implementation on this
backend (§4). (3) **The existing MoE cache does not solve this problem and never
did** — it deliberately *sidesteps* buffer-type mutation by keeping placement
static and moving only slot *contents*, which is why it works; and its per-token
dynamism is orthogonal to a per-session VRAM budget (§3). The one genuinely
cheap version that does exist is a **discrete-tier reconfiguration through the
server's already-working sleep/wake model-reload path**
(`tools/server/server-context.cpp:938-1050`), which needs **zero ggml changes**
(§6, Option A). Recommendation and effort table in §8.

---

## 1. What the proposal is actually asking for, restated against the budget

### 1.1 `-ncmoe` is a load-time-only partition of expert *layers*

`-ncmoe N` is pure sugar over `-ot`. `common/arg.cpp:2780-2790` does nothing but
append to `common_params::tensor_buft_overrides` (`common/common.h:349`) via
`llm_add_n_cpu_ffn_overrides` (`common/common.h:1138-1150`), pushing N regexes
`blk\.<i>\.ffn_(up|down|gate|gate_up)_(ch|)exps` → `ggml_backend_cpu_buffer_type()`
for `i` in `0..N-1`. Per `08-long-context-vram-budget.md:302-304`, it pins the
`ffn_{gate,up,down}_exps` tensors of layers 0…N−1 to CPU and touches nothing
else (`ffn_*_shexp`, the shared expert, does not match `(ch|)exps`).

The overrides reach `llama_model_loader::create_tensor`
(`src/llama-model-loader.cpp:1109-1111`) and are consumed at
`src/llama-model-loader.cpp:1225-1254`, where the matched `buft` selects *which
`ggml_context` the tensor is created in* (`ctx_for_buft`,
`src/llama-model-loader.cpp:1115-1135`). Buffers are then allocated one per
(context, buft) at `src/llama-model.cpp:1777` and marked
`GGML_BACKEND_BUFFER_USAGE_WEIGHTS` at `src/llama-model.cpp:1791-1795`.

**So placement is decided before any memory exists, and the granule is one
layer's three expert tensors — 962.50 MiB for 47 of 48 layers, 1,137.50 MiB for
layer 2** (`08-long-context-vram-budget.md:312-321`, appendix `:777-779`).
Closed form, valid for N ≥ 3 (`:333-336`):

```
expert_vram(N) = 46,200 − 962.5·N        MiB
```

### 1.2 The corrected VRAM budget, including today's new term

`08-long-context-vram-budget.md:517-523` gives the master formula. Two
corrections are needed before using it for this analysis:

- **The FA f16 staging buffer is not a separate term** — it is reserved inside
  the compute buffer at `sched_reserve` time. `PLAN.md:1314-1316` established
  this while measuring Design B: "the f16 staging buffer is reserved at
  `sched_reserve` time against the `n_tokens = n_ubatch` worst case … the
  compute buffer is byte-identical with and without the change." Doc 08 §4.1
  counts it separately, which double-counts against a measured compute buffer.
- **A new depth-proportional VRAM term landed today**: the QSA pooled indexer-key
  cache, `1,536 · n_ctx` bytes = 456 MiB at 307,200 and 900 MiB at 614,400
  (`PLAN.md:1117-1120`).

Reconciled formula, all MiB, `D` = **allocated** `-c`:

```
VRAM(D, ub, N) =   3,928.73                      dense core 3,816.16 + recurrent 112.57
                 + 9,504·D / 2^20                KV, q4_0, both caches
                 + 1,536·D / 2^20                pooled indexer-key cache (new 2026-09-12)
                 + compute(D, ub)                mask/score scratch + FA f16 staging
                 + 46,200 − 962.5·N              GPU-resident experts
```

**Checked against the one config where every term was measured**
(`PLAN.md:824-837`, `-c 311296 -ub 2048 -ncmoe 32`, `logs/decode-scaling/vram-probe-300k.log`):
19,216.16 (model buffer at `-ncmoe 32`) + 2,821 (KV, computed
311,296 × 9,504 / 2²⁰ against a measured 2,052.00 + 769.50) + 8,716.28 (measured
compute buffer) + 112.57 = **30,866 MiB against the probe's implied 30,930**
(32,402 usable less its reported 1,472 spare). **Agreement to 64 MiB, 0.2%.**
That is what makes the rest of this section's arithmetic usable.

Compute-buffer slope, from the same measurement: **0.02800 MiB/token at
`-ub 2048`** (8,716.28 / 311,296), against `PLAN.md:836-837`'s independently
fitted ~0.0267. Halving `-ub` roughly halves it
(`08-long-context-vram-budget.md:425-439`: the term scales as
`n_kv × n_ubatch`).

### 1.3 Which term actually forces `-ncmoe` up — and it is not the KV cache

Decomposing the two measured production configs and the deep one:

| term | 116K tier<br>`-c 122880 -ub 2048` | 300K tier<br>`-c 311296 -ub 2048` | 600K tier<br>`-c 614400 -ub 1024` |
|---|---|---|---|
| dense core + recurrent | 3,929 | 3,929 | 3,929 |
| KV (q4_0, both caches) | 1,114 | 2,821 | 5,570 |
| pooled indexer keys | 180 | 456 | 900 |
| **compute buffer** | **3,692** | **8,716** | **~8,960** |
| non-expert subtotal | 8,915 | 15,922 | 19,359 |
| left for experts (of 32,402) | 23,487 | 16,480 | 13,043 |
| **implied minimum `-ncmoe`** | **24** | **31** | **35** |
| **config actually used** | 24 | 32 | **38** |

(The two `-ub 2048` compute buffers are measured — 3,692.28 at `-c 122880`
(`PLAN.md:308`) and 8,716.28 at `-c 311296` (`PLAN.md:829-831`). The pooled-key
figures are the ones `PLAN.md:1118-1120` states directly (180 / 456 / 900). Only
the 600K compute buffer is extrapolated, from the measured `-ub 1024 -c 163840`
point of 2,391.14 (`PLAN.md:307`) → 0.01459 MiB/token. `PLAN.md:1288-1289`
corroborates the `-ub` scaling independently: it reports `-ub 2048` at 614,400 as
"~16 GiB", against 0.028 × 614,400 = 17,203 MiB. The `-ncmoe 35` conclusion is
insensitive to this term — taking half the `-ub 2048` slope instead (8,602 MiB)
still gives 35.)

Two things fall out of that table, and both matter more than the proposal does.

**(a) The compute buffer is the largest non-expert consumer at every depth, and
it is sized against *allocated* `-c`, not actual `n_kv`.** This is not
inference — it is two lines of code. The reserve path builds its graph from
`memory->init_full()` (`src/llama-context.cpp:610`), whose
`llama_kv_cache_context` constructor sets `n_kv = kv->get_size()`
(`src/llama-kv-cache.cpp:2672`) — the whole allocated context. The *real* decode
graph instead takes `n_kv = kv->get_n_kv(sinfos[i_cur])`
(`src/llama-kv-cache.cpp:2726`), which is `GGML_PAD(cells.used_max_p1(), 256)`
clamped to `cells.size()` (`src/llama-kv-cache.cpp:1250-1263`) — the *actual*
used depth. That is the mechanism behind `PLAN.md:316-320`'s standing
observation that "`-c` is a pure VRAM knob that does not touch throughput
(attention cost follows actual `n_kv`, not allocated context)." **A session
sitting at 8K in a 600K-configured server is holding ~8.6 GiB of compute buffer
it is not using, and that is 9 `-ncmoe` steps.**

**(b) `-ncmoe 38` at 600K looks 2-3 steps more conservative than the budget
requires.** The table's own arithmetic says 35 fits with the pooled cache
included and ~888 MiB spare; 38 was chosen in `PLAN.md:1287-1294` as the first
value that worked and was never swept downward. At `PLAN.md:897-899`'s measured
+0.4 ms/token per host-resident expert layer, three steps is 1.2 ms of an 80.52
ms token — **~1.5%, free, hours of work, no code.** Flagged here because it is
strictly cheaper than anything else in this document (§8, item 1).

### 1.4 Restating the proposal correctly

The proposal says "move experts based on needed context length." The budget says
expert VRAM is the *residual* — it is whatever the compute buffer, KV cache and
pooled-key cache leave over. So:

> **Dynamic expert placement is not a knob. It is the release valve on a knob.
> The knob is allocated `-c` (through the compute buffer), and expert placement
> is what has to give when that knob turns.**

Any design that moves experts without also making the compute-buffer/KV
reservation depth-adaptive gains nothing, because the reservation is what pins
`-ncmoe` high. Any design that makes the reservation adaptive *needs* expert
migration, or it OOMs the moment a session grows. The two halves are a matched
pair, in the same sense `12-decode-depth-scaling.md:406-412` used for sparse FA
and the pooled-key cache.

---

## 2. Prediction feasibility — easier than expected, and the easy part is genuinely easy

Asked: is the session's eventual length knowable in advance, or must the policy
be reactive? Answered against the actual harness
(`tools/server/`, the post-refactor multi-file server — slot logic is in
`server-context.cpp`, not `server.cpp`).

### 2.1 Per-turn prompt length is known exactly, before any forward pass

Tokenization happens on the **HTTP handler thread**, before the task is even
queued: `tools/server/server-context.cpp:4293-4300` calls
`tokenize_input_prompts`, and the count is readable the moment the tokens land
on the task at `:4314` via `server_task::n_tokens()`
(`tools/server/server-task.h:182-184`). Tasks are posted at `:4341`. The first
`llama_decode` is much later and on a different thread
(`tools/server/server-context.cpp:3678-3687`).

Independently re-confirmed on the compute thread before any batch is filled:
`update_slots()` logs `task.n_tokens` at `SLOT_STATE_STARTED`
(`tools/server/server-context.cpp:3147-3148`) and the hard length gates run
right there (`:3196-3204` non-splittable, `:3206-3214` splittable —
`if (slot.task->n_tokens() >= slot.n_ctx) send_error(... ERROR_TYPE_EXCEED_CONTEXT_SIZE)`).

**Verdict: trivially available, at two distinct points, both strictly before the
first forward pass.** There is even a decode-free counting endpoint
(`tools/server/server-context.cpp:5513-5535`).

### 2.2 The multi-turn worry does not apply, because the server is stateless

This was the sub-question flagged as the hard one, and the code settles it in the
easy direction. **`llama-server` has no notion of a session.** Every
`/v1/chat/completions` request must carry the entire `messages` array
(`tools/server/server-common.cpp:1185-1196` throws if absent), the template is
applied to the whole list producing one flat prompt
(`tools/server/server-common.cpp:1337-1341`), and server-side conversation state
is explicitly refused — `tools/server/server-chat.cpp:10-12` rejects
`previous_response_id` with "llama.cpp does not support 'previous_response_id'."

The KV-reuse optimization is applied *after* full tokenization, purely as a
cache hit: `n_past = slot.prompt.tokens.get_common_prefix(input_tokens)` at
`tools/server/server-context.cpp:3216-3219`, trimming at `:3409-3414`.

**So the client resends the whole conversation on every turn, and the new total
length is therefore exact and known upfront at every turn.** A growing
multi-turn session is not a blind ramp — it is a sequence of requests each of
which announces its own full depth before the model runs. The
predictive-vs-reactive dichotomy in the brief collapses: **the policy can be
predictive per turn, which is strictly better than reactive, and it needs no new
client protocol.**

### 2.3 What is *not* knowable, and why it does not matter here

The eventual *generated* length. `n_predict` (`tools/server/server-task.h:61`)
is populated from `max_tokens` / `max_completion_tokens` by a declarative schema
alias (`tools/server/server-schema.cpp:44-48`) and lands on
`slot.n_predict_max` at `tools/server/server-context.cpp:1818`. But
`set_hard_limits(-1, INT32_MAX)` means **it is never clamped against `n_ctx`,
and the default is −1, unlimited** (`common/common.h:449`). Anthropic-shaped
requests do get a 4096 default injected (`tools/server/server-chat.cpp:571-576`).

**Inference:** against a 300-600K prompt, an unbounded generation is a rounding
error on the tier decision — even 16K of output is 2.6% of 600K. A tier policy
can safely use `n_prompt_tokens + max(n_predict, default_reserve)` with a
default reserve of a few thousand tokens and be right essentially always. This
is not a blocker.

### 2.4 Where reconfiguration could safely run

`update_slots()` opens with an explicit all-idle early return —
`tools/server/server-context.cpp:2807-2823`:

```cpp
        // check if all slots are idle
        { bool all_idle = true;
          for (auto & slot : slots) { if (slot.is_processing()) { all_idle = false; break; } }
          if (all_idle) { SRV_TRC("%s", "all slots are idle\n"); metrics_flush_idle(); return; } }
```

Main-loop thread, every slot `SLOT_STATE_IDLE`, nothing decoding, nothing
batched, followed by a blocking wait (`tools/server/server-queue.cpp:319-360`).
**This is the reconfiguration window, and it is the only one with that
guarantee** — `slot.release()` (`tools/server/server-context.cpp:544-566`) is
called from inside the slot-iteration loop (`:3193`, `:3203`, `:3213`, `:3504`)
and from `pre_decode()` (`:2920`, `:2931`), so its `callback_on_reset` /
`callback_on_release` hooks (`:362-363`, wired `:1302-1311`) do **not** imply no
other slot is mid-batch.

The context-shift path is the other candidate, and it is the closest thing this
harness has to "compaction": `pre_decode()` at
`tools/server/server-context.cpp:2909-2972`, with the left-shift-and-discard core
at `:2937-2971` (`seq_rm` + `seq_add` by `n_discard`, `slot.truncated = true`).
It runs on the main loop thread before `batch.clear()` at `:2974` and before any
decode. Note it is **off by default** (`common/common.h:571`,
`bool ctx_shift = false;`); with it off, overflow just stops generation with
`STOP_TYPE_LIMIT` (`tools/server/server-context.cpp:1885-1893`). There is no
summarization-style compaction in this harness at all — the "compaction viable
at 500K" the user referred to would be a *client-side* behaviour, which reaches
the server as an ordinary shorter prompt and is therefore handled by §2.2's
per-turn prediction for free.

---

## 3. The existing MoE cache: what it does, and why it is not the mechanism this needs

Read in full: `ggml/src/ggml-sycl/moe-cache.hpp` (207 lines) and
`ggml/src/ggml-sycl/moe-cache.cpp` (1,005 lines), plus the integration sites in
`ggml/src/ggml-sycl/ggml-sycl.cpp`. `05-hybrid-cpu-gpu-mul-mat-id-scoping.md`
§3.4 already inventories it component-by-component and is not repeated here;
what follows is only what bears on *this* proposal.

### 3.1 It does not move tensors. That is the whole point of it.

The brief's framing — "the existing cache does per-expert CPU/GPU movement
today, just per-token rather than per-session, so the safety question may already
be answered by its existence" — is **not what the code does**, and the
distinction is the most important architectural fact in this document.

The cache never changes any tensor's `buffer`, `buft`, or (on the live path)
`data`. Expert weights live **permanently** in the `SYCL_MoE_Cached` pinned-host
buffer type (`moe-cache.cpp:82-163`, cold pointer read as
`const char * cold_base = (const char *) src0->data;` at `moe-cache.cpp:747`).
A fixed-size VRAM slot pool (`moe-cache.cpp:172-174`: `void * pool`,
`size_t slot_stride`, `int n_slots`, allocated at `:220`) holds *copies*. The
op handler then hands the **unmodified** GEMV a different base pointer and a
remapped index array — `ggml-sycl.cpp:5200-5205` calls
`ggml_sycl_mul_mat_vec_q_id(src0->type, pool_base, src1_ddq, remapped_ids, …,
/*expert_weight_stride=*/ slot_stride, …)`, and the kernel's own
`vx_base + ids_dev[i]*stride` arithmetic indexes *slots* instead of experts.
Rationale is stated outright at `moe-cache.hpp:102-112`.

The only place `->data` is rewritten is the Phase-3a fallback, and it writes a
**stack copy** of the tensor struct, not the graph's tensor:
`ggml_tensor src0_row = *src0;` at `ggml-sycl.cpp:5324`, then
`src0_row.data = expert_slot_ptr[i02];` at `:5408`.

**So the cache is evidence that the *hard* problem is avoidable, not that it is
solved.** It answers "can expert bytes move between host and device mid-flight"
(yes) while saying nothing about "can a tensor's backend assignment change"
(§4 answers that: no).

### 3.2 Its policy trigger is per-op, and the policy is measurably broken

Admission/eviction is decided on-device by a single-work-group plan kernel
(`moe-cache.cpp:754-974`, one WG by design per the rationale at `:726-734`),
scored by a decayed-LFU function at `moe-cache.cpp:404-407`:

```cpp
static inline uint32_t moe_plan_effective_frequency(uint32_t frequency, uint32_t touched_step, uint32_t now_step) {
    const uint32_t elapsed = now_step - touched_step;
    return elapsed < 32 ? (frequency >> elapsed) : 0;
}
```

Cadence is **per cached MoE tensor per token** — `device_step` increments once
per `plan_and_gather` call, ~144 calls/token for this model
(`ggml-sycl.cpp:5196`, `moe-cache.cpp:653`).

**Found while reading, and it should be recorded before anyone builds on this
policy: `expert_frequency` is never incremented on a cache hit.** Hits take the
`continue` at `moe-cache.cpp:856` without touching frequency or epoch, and the
only write is `expert_frequency[eid] = 1;` on admission (`moe-cache.cpp:931`).
Every resident expert therefore scores `1 >> elapsed`, i.e. 0 after a single
step, so all occupied slots tie at 0 and the victim is whichever slot wins the
numeric min of the packed candidate (`moe-cache.cpp:424-429`, `:894-920`) —
**in practice lowest-slot-index-first, which is neither LRU nor LFU.** Two
further defects in the same neighbourhood: comments assume 96 slots
(`moe-cache.cpp:477`, `:786`) while the configured default is 64
(`moe-cache.cpp:271-282`, `GGML_SYCL_MOE_CACHE_SIZE`, and no script in
`staging/` or `docker/` exports it); and the registry
(`moe-cache.cpp:284-285`, `g_moe_caches` keyed by tensor pointer) is **never
erased**, so every cached tensor's VRAM pool is held for process lifetime with
no accounting against the SYCL pool.

### 3.3 It is off by default, and prefill bypasses it entirely

**Off by default.** Zero conditional compilation in `moe-cache.cpp`; the enable
switch is *tensor placement*. Nothing routes tensors to `SYCL_MoE_Cached`
automatically — the buft is merely offered via
`ggml_backend_sycl_moe_cached_get_extra_bufts` (`moe-cache.cpp:159-163`,
registered `ggml-sycl.cpp:7769`) and must be named by
`-ot "\.ffn_(gate|up|down)_exps\.weight=SYCL_MoE_Cached"`. Every runner script
that uses it does so explicitly, and it is deliberately **commented out** in
`staging/work/run_flashnext.sh:8-9` — consistent with `PLAN.md:1445-1451`'s
standing decision that "`-ncmoe` is the interim production baseline, not the
custom cache."

**And the decisive point for this deployment: prefill does not go through the
cache at all.** `ne12 > 1` falls through uncached at `ggml-sycl.cpp:5437+`,
reading `src0` straight through the pinned host pointer — deliberate, per the
comment at `ggml-sycl.cpp:5348-5354`. `05-hybrid-cpu-gpu-mul-mat-id-scoping.md`
§10.3 measured that read-over-PCIe shape as a **monotone loss** (tok/s falls as
more rows go to GPU), and its §3.2 quotes the upstream thread's consistent
finding: "Prefill almost never benefits. Every cache arm in this thread loses
prefill."

**This is disqualifying on its own.** TTFT is this deployment's dominant metric
— 4 min 37 s at 116K and 19 min 57 s at 300K (`PLAN.md:855-856`, `:1340-1341`)
against decode tokens costing tens of milliseconds. A mechanism that is inert
on prefill and suspected-negative on it cannot be the vehicle for a
long-context optimization here.

### 3.4 Where it *would* be the right mechanism, if the rest were fixed

Stated for completeness, because it is genuinely the elegant answer to the
narrow question and should not be rediscovered:

**If you wanted a continuously-variable, context-driven expert VRAM budget, the
cache's slot pool is the correct dial and `-ncmoe` is not.** The slot pool is a
single `sycl::malloc_device` region per tensor with uniform stride
(`moe-cache.cpp:220`, `:302`) whose size is read from an env var once at cache
creation (`moe-cache.cpp:271-282`). Making it resizable is one `sycl::free` +
`malloc_device` + re-`fill` of `slot_table` / `expert_for_slot` at an idle
point, per tensor — call it 100-150 LOC on top of code that already exists. And
the granularity is better by an order of magnitude:
`08-long-context-vram-budget.md:719-728` measured it — "`-ncmoe` quantizes
expert VRAM into **962.5 MiB steps**; the cache tunes it in **90.58 MiB steps**
(one slot index)," with `-ncmoe`'s rounding wasting up to 580 MiB at the deepest
configs.

What stands between that and a usable feature: the prefill hole (§3.3), the
degenerate policy (§3.2), the fact that the cache measured 23.1 tok/s against
`-ncmoe`'s 19.7-25.3 when it was retired (`PLAN.md:1396-1427`), and — the part
that has gotten *worse*, not better, since that decision —
`12-decode-depth-scaling.md:429-435`'s finding that a **lower** floor makes the
relative depth bar **harder**: at a 36.3 ms floor, 300K/600K sit at +7.3%/+14.6%;
at a resident-expert floor of ~20 ms the identical unchanged slope gives
+13.0%/+26.0%. Re-reading `PLAN.md:1445-1451`'s reasoning as the brief asked:
**it still applies, and for a stronger reason than when it was written.** The
question then was "does the cache beat a static split at short context." The
question now is "does a cache that is inert on prefill and lowers the floor help
a deployment whose binding metric is prefill and whose target is a relative
slope." Both answers are no.

---

## 4. The safety question: can a loaded tensor change backend mid-session?

This is item 3 of the brief and the answer is **no, three independent ways**, all
grep-verified rather than reasoned.

### 4.1 There is no move/migrate/resize primitive anywhere in ggml

`ggml_backend_buffer_i` (`ggml/src/ggml-backend-impl.h:46-70`) has
`free_buffer`, `get_base`, `init_tensor`, `memset_tensor`, `set_tensor`,
`get_tensor`, `set_tensor_2d`, `get_tensor_2d`, `cpy_tensor`, `clear`, `reset`.
**No resize, no realloc, no move.** Grep for
`ggml_backend_{move,migrate,relocate,reassign}` across `ggml/` and `src/`:
**zero hits.**

The two public entry points that assign `tensor->buffer` both hard-assert it is
currently NULL — `ggml/src/ggml-backend.cpp:2128-2141`:

```cpp
enum ggml_status ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, void * addr) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->data == NULL);
    ...
```

(same shape at `ggml_backend_view_init`, `:2116-2126`). Every other write to
`tensor->buffer` in the tree is a fresh-tensor initializer, a load-time dummy
(`src/llama-model.cpp:1773`, `:2174`;
`src/llama-model-loader.cpp:1050-1054`), a KV-cache init dummy
(`src/llama-kv-cache.cpp:282`, `src/llama-memory-hybrid-idx.cpp:174`), or inside
the meta/tensor-parallel backend's own construction.

Model weight buffers are owned by
`pimpl->ctxs_bufs` (`src/llama-model.cpp:1162`) and freed only when the
`llama_model` is destroyed (`src/llama-model.cpp:1191-1195`, `:2790-2792`).
**There is no partial-reload path.** Changing a buft means destroying the
context, destroying the model, and re-running
`llama_model_load_from_file` → `load_tensors`.

### 4.2 Graph reuse means a forced buft change would be silently stale

Nothing *forbids* writing `t->buffer` at runtime — the asserts only guard the
alloc helpers. But the scheduler would not see it.

`llama_context::process_ubatch` (`src/llama-context.cpp:1333-1382`) reuses the
previous graph whenever `res->can_reuse(gparams)`, and only otherwise does
`res->reset(); ggml_backend_sched_reset(...); gf = model.build_graph(...);
ggml_backend_sched_alloc_graph(...)`. `llm_graph_result::can_reuse`
(`src/llama-graph.cpp:1406-1436`) delegates to `llm_graph_params::allow_reuse`
(`src/llama-graph.h:814+`), which compares **only** ubatch shape
(`n_tokens`, `n_seq_tokens`, `n_seqs`, `n_seqs_unq`, `equal_seqs()`,
token-vs-embd), `n_outputs`, sampler set and seq ids. **Nothing about buffers,
bufts, or `data` pointers.**

And the backend assignment that *would* have to change is computed only in
`ggml_backend_sched_split_graph`, which reads
`tensor->buffer->buft` through `ggml_backend_sched_backend_from_buffer`
(`ggml/src/ggml-backend.cpp:889-909`) and the weight branch of
`ggml_backend_sched_backend_id_from_cur` (`:922-984`). `split_graph` is called
only from `ggml_backend_sched_alloc_graph` (`:1994`) and the two reserve paths
(`:1964`, `:1975`) — **not** from
`ggml_backend_sched_graph_compute_async`, which skips it entirely when
`sched->is_alloc` is already true (`:2011-2024`).

**So in steady-state single-token decode the graph is built once and the splits
persist for every subsequent token.** Mutating a weight's buffer mid-session
leaves stale sched splits, stale `hv_tensor_copies` staging decisions, and stale
`ggml_gallocr` node allocations — and `ggml-backend.cpp:939-943` will
`GGML_ABORT("pre-allocated tensor (%s) in a buffer (%s) that cannot run the
operation (%s)")` if it ever does re-derive and disagrees. The escape hatches
(`graph_reuse_disable`, `src/llama-context.h:378-379`, env
`LLAMA_GRAPH_REUSE_DISABLE`, read at `src/llama-context.cpp:279-283`; or calling
`ggml_backend_sched_reset` manually) both cost the reuse optimization on every
token, which at ~2,800 dispatched SYCL nodes per decode token
(`PLAN.md:530-531`) is not a small thing to give up.

### 4.3 "Asynchronously" has no implementation on this backend

The proposal's word is "asynchronously." The available surface does not support
it.

`ggml_backend_tensor_copy_async` exists (`ggml/include/ggml-backend.h:116`, def
`ggml/src/ggml-backend.cpp:512-531`) but takes **no event or queue parameter** —
ordering is implicit in the backend's stream — and when the backend hook returns
false or is NULL it degrades to a **full double synchronize plus blocking copy**
(`ggml-backend.cpp:527-530`).

**SYCL NULLs the hook out.** `ggml/src/ggml-sycl/ggml-sycl.cpp:6855-6874`:

```cpp
    /* .cpy_tensor_async        = */ NULL, // ggml_backend_sycl_cpy_tensor_async,
                                           // // TODO: update for the new
                                           // interface
```

A dead implementation sits unreferenced at `ggml-sycl.cpp:6089-6117`. Events
*are* implemented (`event_record`/`event_wait` at `:6822-6853`, device-side
`event_new`/`event_free`/`event_synchronize` at `:7430-7475`, wired `:7490-7492`)
— but `event_wait` is a **host-blocking** `sycl_event->wait()`
(`:6845-6853`), not a device-side cross-queue dependency. This is exactly why
the existing cache bypasses the ggml event API entirely in favour of a raw
`ext_oneapi_submit_barrier` (`moe-cache.hpp:65-90`).

`05-hybrid-cpu-gpu-mul-mat-id-scoping.md` §4.3 adds the structural constraint:
`ggml-sycl` has **exactly one in-order queue per device** —
`ggml_backend_sycl_context::stream()` (`ggml/src/ggml-sycl/common.hpp:350-355`)
returns `dpct::get_device(device).default_queue()` for *every* index despite the
`[MAX_DEVICES][MAX_STREAMS]` array. A background transfer therefore needs a
private queue, built from the same `sycl::context` — the pattern exists
(`moe-cache.cpp:218`) and so does the documented cross-context silent-no-op trap
that construction avoids (`PLAN.md:558-560`).

### 4.4 The hazard that should stop anyone from writing "asynchronously" lightly

`moe-cache.cpp:350-372` is a function whose header
(`moe-cache.hpp:79-83`) documents a non-blocking cross-queue barrier and whose
body is a host-blocking `e.wait()` loop. The implementation comment at
`moe-cache.cpp:351-364` explains why, and it is the single most relevant piece
of prior art in the repo to this proposal:

> `// REVERTED to a host-blocking wait -- an ext_oneapi_submit_barrier(events)`
> `// version of this (compute queue depends on the copy events without the host`
> `// blocking) was tried and twice produced non-deterministic greedy-decode`
> `// output (temp 0, fixed seed, identical repeated runs diverging)` …
> `// Root cause not found in the time available` … `Revisit with a proper SYCL`
> `// event/queue trace before attempting this again.`

**This project has already attempted asynchronous cross-queue movement of
expert weights, twice, and both times it silently corrupted output for reasons
that were never found.** That is the direct precedent for the proposal's third
clause. It does not make the idea impossible; it does mean any effort estimate
that treats "asynchronously" as a detail is wrong.

For what it is worth, the *synchronous* same-call safety problem was solved, and
the solutions are worth reading before rebuilding them — three distinct guards:
the protected-expert list (declared `moe-cache.cpp:776-777`, comment
`:767-775` — "must never be evicted by THIS call's own miss resolution below …
found the hard way (garbled/degenerate greedy-decode output on real
hardware)"), the `local_slot_pinned` mirror for later misses in the same call
(`:795`, comment `:792-794`, set `:943-945`), and the
acquire-all-before-dispatch-any discipline on the legacy path
(`ggml-sycl.cpp:5365-5391`). Plus the cross-token
`mark_reads_submitted`/`wait_for_prior_reads` pair (`moe-cache.cpp:374-386`,
rationale `moe-cache.hpp:85-100`, whose own comment records that "an early
version without this produced different greedy-decoded output across repeated
identical runs").

### 4.5 One thing that *is* free: the compute buffer already grows on demand

The counterweight to all of the above. `ggml_gallocr_reserve_n_impl`
(`ggml/src/ggml-alloc.c:825`, the relevant loop at `:905-945`) sets
`realloc = true` only when `new_chunk_size > cur_chunk_size`, then does
`ggml_vbuffer_free` **before** `ggml_vbuffer_alloc` (`:935-939`) — so growth
neither doubles peak VRAM nor needs new code. And `llama_context::sched_reserve`
(`src/llama-context.cpp:582-588`) is already called on every `decode()`
(`:1740`) and `encode()` (`:1448`), gated on a `sched_need_reserve` flag
(`src/llama-context.h:346`) that is already set from a dozen sites
(`src/llama-context.cpp:1186`, `:1202`, `:1232`, `:1251`, `:1260`, `:1270`,
`:1290`, `:1329`, `:3418`).

**Inference, and it is the most actionable technical finding in this
document:** a grow-on-demand compute buffer needs (a) the reserve graph to use a
high-water `n_kv` instead of `kv->get_size()` — one expression at
`src/llama-kv-cache.cpp:2672` — and (b) a policy setting `sched_need_reserve`
when the high-water mark crosses a band. Both small. **What ggml-alloc will
*not* do is shrink** — `realloc` is never set on a size decrease, so de-escalating
after a compaction would not return VRAM without a new force-shrink path.

---

## 5. Does this fix the thing it was aimed at? No.

### 5.1 What the 600K measurement actually says

`PLAN.md:1287-1306`, measured today: `-ub 1024 -ncmoe 38` at `d614400` gives
**12.42 tok/s**, against a same-config shallow control of **22.55 (d8192) /
22.93 (d32768)** — a residual depth slope of **0.0602 ms per 1,000 tokens,
5.1x the 0.0119** the `-ncmoe 24` arm measures over four depths. And it is
provably not attention: sparse FA measures **0.94 ms/token flat at
`n_kv = 614,400`**, under 1 ms of the 36.5 ms/token that depth costs.

`PLAN.md:1297-1300` names the cause: "A slope that grows 5x when only `-ncmoe`
changes is the signature of a host-side term — `-ncmoe 38` puts 38 layers of CPU
expert GEMM on the same cores as `set_input_qsa`'s single-threaded O(`n_kv`)
grouping scan and block bias," which
`12-decode-depth-scaling.md:841-848` independently measured at **0.0155 ms/1k**
and called "the second-largest depth-proportional term in the decoder."

**A correction to the brief's framing, worth recording:** doc 12 does not use
the word "contention" anywhere (grep-verified, zero hits), and it models the
`-ncmoe` CPU expert workload as **depth-independent** — it lands in the
measured-flat 35.8-37.5 ms/token floor
(`12-decode-depth-scaling.md:465-470`), and an `-ncmoe` 24→32 change left doc
12's depth fit intact to 0.8-3.2% (`:866-870`). The contention reading is
`PLAN.md:1297-1300`'s hypothesis for the 600K arm specifically, and it is a
hypothesis — the two candidates it names (the `set_input_qsa` host term, and the
never-isolated KQ-mask build in `build_attn_qsa`, `:1301-1304`) are both
un-disentangled.

### 5.2 The structural reason dynamic placement cannot help 600K

At 600K, the deep tier *is* the operating point. A session that has actually
reached 614,400 tokens needs the 614,400-token KV cache (5,570 MiB), the
614,400-token pooled-key cache (900 MiB) and a compute buffer sized for its real
`n_kv` — so it lands on `-ncmoe 35-38` no matter how it got there. Dynamic
placement changes *when* you pay that, not *whether*.

**So the 600K number under a perfect implementation of this proposal is 12.42
tok/s: unchanged.** That is the finding.

### 5.3 What it *is* worth, priced honestly

The value is the converse: a server that must be *capable* of 600K currently has
to be *configured* for 600K, and every shallower request pays for it. Dynamic
tiering removes that tax.

Using the measured arms and `PLAN.md:897-899`'s +0.4 ms/token per host-resident
expert layer:

| actual depth | static 600K config<br>(`-ub 1024 -ncmoe 38 -c 614400`) | dynamic, best tier for the depth | gain |
|---|---|---|---|
| ~8K | **22.55** tok/s (measured) | **25.36** tok/s (measured, 8K tier) | **1.12x** |
| 32K | **22.93** (measured) | **25.43** (measured) | 1.11x |
| 116K | ~20.4 (44.0 + 0.0602×118 = 51.1 ms) | **24.56** (measured) | 1.20x |
| 300K | ~16.0 (44.0 + 0.0602×307 = 62.5 ms) | **21.08** (measured, 300K tier) | **1.32x** |
| 600K | **12.42** (measured) | **12.42** (same config) | **1.00x** |

(The 116K and 300K static-config rows are the only projected numbers in the
table, from the 600K arm's own measured control and slope; every other cell is a
measured arm from `PLAN.md:1267-1273` and `:1279-1286`.)

A cross-check on the model: at `-ncmoe 38` the shallow control is 44.35 ms/token;
subtracting 14 layers × 0.4 = 5.6 ms gives 38.75 ms = 25.8 tok/s, against the
`-ncmoe 24` tier's measured 25.36-26.5. **Agreement to ~2%**, which is what makes
the projected rows worth quoting.

So: **a real 1.1-1.3x for everything short of the ceiling, and nothing at the
ceiling.** That is a serving-flexibility win, not a long-context fix, and it
should be sold as exactly that.

### 5.4 What would fix 600K, and it is cheaper than this

If the whole 0.0602 − 0.0119 = **0.0483 ms/1k** excess slope at `-ncmoe 38` is
the `set_input_qsa` host term, removing it takes 600K from 80.52 ms to
80.52 − 0.0483 × 614.4 = **50.8 ms/token = 19.7 tok/s — inside the stated
17-25 band.** Even if only the 0.0155 ms/1k doc 12 measured directly is
recoverable, that is 9.5 ms of 80.52 → 13.9 tok/s, and combined with §1.3(b)'s
free `-ncmoe` sweep it closes a meaningful fraction of a 4.6 tok/s gap.

`PLAN.md:1307-1311` already names this as the next task and already has the
argument for how to fix it: "the same 'only the tail changes' argument applies to
it" — i.e. the grouping scan is incremental for exactly the same reason the
pooled block keys were (`PLAN.md:1111-1120`), and that change is landed,
proven exact 480/480 (`PLAN.md:1139-1146`) and measured. **The mechanism is
known, the precedent is in-tree and one day old, and the prize is larger than
the proposal's.**

---

## 6. Three designs, smallest first

### Option A — discrete tiers through the existing sleep/wake reload path (**recommended, if anything**)

The key enabler was found while investigating §2: **the server already tears
down and rebuilds the model and context in-process, while running, with all
slots idle and the queue lock held.** `handle_sleeping_state`
(`tools/server/server-context.cpp:954-1005`) calls `destroy()`
(`:938-953`) and `load_model(params_base)` (`:1006+`), wired as a queue callback
at `:1415` and driven from `tools/server/server-queue.cpp:325-350`.

**That is a working, upstream, tested precedent for changing `-c`, `-ub` and
`-ncmoe` at runtime — by reloading with a different `common_params` — and it
needs zero ggml changes.** It sidesteps every constraint in §4: no tensor is
moved, no buft is mutated, no graph is stale, nothing is asynchronous.

Design:

1. A tier table — 2 or 3 `(n_ctx, n_ubatch, n_cpu_moe)` triples from §1.3, e.g.
   `(122880, 2048, 24)` / `(311296, 2048, 32)` / `(614400, 1024, 35)` — the
   first two being exactly this project's two measured production configs.
2. A policy function keyed on the exactly-known per-turn prompt length from
   §2.1 (`task.n_tokens()` at `tools/server/server-context.cpp:4314`, or
   `:3147` on the compute thread), plus a generation reserve per §2.3.
3. A trigger at the all-idle gate (`tools/server/server-context.cpp:2807-2823`),
   escalating when a queued task needs a deeper tier and de-escalating on a
   timer or after N consecutive shallow turns.
4. Execute by the sleep/wake pattern: `destroy()` + `load_model(mutated_params)`.

Cost per transition: a full model reload — **59 s with `-lzm auto`, ~4.9 min
with the recommended `-lm none -lzm off`** (`PLAN.md:646-648`) — plus loss of
the prompt cache, so the next turn reprefills from scratch instead of from its
common prefix. Against a 300K TTFT of 19 min 57 s (`PLAN.md:855`) an escalation
is a ~25% surcharge on one turn, paid once, buying ~1.32x decode for the rest of
the session. De-escalation is the cheap direction and can be deferred to genuine
idle.

Files and rough LOC (all in `tools/server/`, **no ggml, no kernel, no model
code**):

| file | change | LOC | risk |
|---|---|---|---|
| `server-context.cpp` | tier table, policy fn, all-idle trigger, params mutation + reload call | 150-250 | LOW |
| `server-task.h` / `server-schema.cpp` | optional explicit client hint (`context_budget`) | 20-40 | LOW |
| `arg.cpp` / `common.h` | `--context-tiers` CLI plumbing | 40-80 | LOW |

**~250-350 LOC, LOW-MEDIUM risk, 2-4 days.** The correctness argument is
trivial — it is a reload, so either the new configuration works or the server
fails to come up, exactly as today.

**Optional extension, and the whole risk lives here:** preserve the KV cache
across the transition with `llama_state_seq_get_data` / `llama_state_seq_set_data`
(`include/llama.h:874`, `:884`) to avoid the reprefill. At 300K that is ~3.3 GiB
round-tripped through host RAM, a few seconds. But it must round-trip **three
different kinds of state at once** — the 12 conventional KV layers, the 36
gated-DeltaNet recurrent states (112.57 MiB, always f32,
`08-long-context-vram-budget.md:194-216`), and the QSA pooled indexer-key cache
that landed **today**, whose lifecycle is guarded by a per-block fingerprint plus
drop-on-any-reseat (`PLAN.md:1121-1128`) and whose state serialization is almost
certainly not implemented. Add 150-250 LOC and move to **HIGH risk**; do it as a
second step, gated on a working Option A, or not at all.

### Option B — the literal proposal: continuous per-expert CPU↔GPU migration

Blocked on four things, all established in §4:

- No ggml primitive re-points a loaded tensor (§4.1). Requires new
  buffer-type-level infrastructure — a "move tensor between bufts" operation
  with re-read-from-source, plus buffer lifetime accounting that
  `pimpl->ctxs_bufs` (`src/llama-model.cpp:1162`) does not currently support.
- Graph reuse would not see the change (§4.2). Requires either a new
  invalidation hook into `llm_graph_result::can_reuse`
  (`src/llama-graph.cpp:1406-1436`) or giving up reuse on ~2,800 nodes/token.
- SYCL has no async tensor copy and host-blocking events (§4.3).
- The async variant of exactly this data movement has already produced
  unexplained non-determinism twice in this repo (§4.4).

Rough scope: ~400 LOC of new ggml-backend infrastructure, ~200 in
`llama-model`/`llama-context`, ~150 SYCL, ~200 server, plus a correctness
harness that cannot use cross-run comparison at all (`PLAN.md:1129-1138`: "the
forward pass itself is not reproducible", so "no future correctness work here
should use a cross-run diff" — the oracle has to live inside one graph, as the
pooled-key cache's and Design B's did).

**~950-1,100 LOC, HIGH risk, 3-6 weeks, and per §5.2 it delivers 1.00x at
600K.** Not recommended.

### Option C — context-driven slot budget on the existing MoE cache

Architecturally the prettiest (§3.4): resize the slot pool instead of retyping
tensors, at 90.58 MiB granularity instead of 962.5 MiB. The resize itself is
100-150 LOC.

But it requires first making the cache competitive again: a prefill path that
does not exist (§3.3, and prefill is the binding metric), a policy that is
currently degenerate (§3.2), plus the registry/allocation-checking defects. And
`12-decode-depth-scaling.md:429-435`'s floor/slope tension means a *successful*
residency improvement makes the relative depth bar harder, not easier
(`PLAN.md:791-800`: "The slope work has to land with, or ahead of, any large
reduction in the floor").

**~600-800 LOC, HIGH risk, 3-5 weeks, with a real chance of regressing TTFT.**
Not recommended now. Revisit only if (a) prefill stops being the binding metric,
and (b) the slope program of `12-decode-depth-scaling.md` §5 is fully landed so
a lower floor is affordable.

---

## 7. Open technical unknowns, stated plainly

Ordered by how much they would change the above.

1. **Is the 600K excess slope really `set_input_qsa`?** `PLAN.md:1297-1304`
   offers two candidates and isolates neither; the second (the KQ-mask build in
   `build_attn_qsa` — `ggml_fill(-INF)` + `ggml_set_rows` + `ggml_add` over the
   full `n_kv`-wide f16 mask, 12x per token) has never been measured by anything
   in this project. **§5.4's whole case rests on this attribution.** It is also
   the cheapest thing in this document to settle — a host-side timer around
   `set_input_qsa` over a bounded decode window at two depths.
2. **Does `-ncmoe 38` at 600K sweep down?** §1.3(b) says the budget allows 35.
   Untested. Free, hours, and it changes the 600K baseline every other number is
   measured against.
3. **Does `llama_state_seq_{get,set}_data` round-trip this model's hybrid state
   at all?** Three state kinds, one of which is one day old (§6, Option A
   extension). Unknown, and it is the difference between Option A being 2-4 days
   and being 2-3 weeks. Testable cheaply at 8K before committing to anything.
4. **Does `ggml_gallocr` handle a *shrinking* reserve?** §4.5 says it does not
   realloc on a size decrease, so de-escalation would not return VRAM. This is
   only load-bearing for a compute-buffer-banding design, not for Option A
   (which reallocates everything anyway).
5. **Why did the async cross-queue barrier produce non-deterministic output?**
   `moe-cache.cpp:351-364` records the failure and the absence of a root cause.
   Unresolved, and it gates anything with "asynchronously" in it.
6. **There is still no validated rope strategy past 262,144.**
   `PLAN.md:872-891`: YaRN "destroys this model … below the trained context
   length, not just past it," and plain unscaled extrapolation past `n_ctx_train`
   is untested for quality. **Every 600K number in this repo, including the
   12.42 tok/s this proposal targets, is a throughput measurement on output
   whose quality at that depth has never been validated.** Building placement
   machinery for a depth the model may not be usable at is a sequencing error.
   **Closed same day by `15-yarn-and-long-context-rope.md` — see §10.**

---

## 8. Recommendation

**Do not build the proposal as stated. Do not build anything here yet.**

In order:

1. **Sweep `-ncmoe` down at 600K** (§1.3(b), §7 item 2). Hours, no code, worth
   ~1.5%, and it corrects the baseline.
2. **Isolate and fix the O(`n_kv`) host term in `set_input_qsa`** (§5.4, §7
   item 1). This is already `PLAN.md:1307-1311`'s named next task, the
   incremental-update precedent is one day old and proven exact, and if the
   attribution holds it takes 600K to ~19.7 tok/s — **inside the bar, which is
   the actual goal, and which dynamic placement cannot reach by construction.**
3. ~~Close the rope/quality gap at 600K~~ (§7 item 6) before optimizing that
   depth further. **Closed 2026-09-12 — see §10.**
4. **Only then, and only if the deployment genuinely needs one server capable of
   both 8K and 600K:** build **Option A** — discrete tiers through the existing
   sleep/wake reload path. ~250-350 LOC, LOW-MEDIUM risk, 2-4 days, zero ggml
   changes, worth a measured **1.1-1.3x for every request short of the
   ceiling**. Hold the `llama_state_seq_*` KV-preservation extension as a
   separate, later, HIGH-risk step.
5. **Options B and C: no.** B is 3-6 weeks of new ggml infrastructure for 1.00x
   at the depth that motivated it. C requires first rebuilding a mechanism that
   was retired for good reasons that have since gotten stronger, and it risks
   TTFT, which is this deployment's binding metric.

The one-line version: **the question "can we move experts based on context
length" has a cheap answer (reload into a different tier at an idle point) and an
expensive answer (per-expert migration ggml cannot express), and neither answers
the question that was actually being asked, which is why 600K is 12.42 tok/s.**

---

## 9. ADDENDUM (2026-09-12): Option A is implemented. What it is worth is a
## capability, not a speed-up.

Date: 2026-09-12, same day as the scoping above. Status: **built; mechanism
validated end to end; one cost/VRAM measurement at a production tier boundary
still open (§9.10).** Section 8 item 4's Option A -- discrete tiers through the server's
existing sleep/wake reload path -- was implemented as
`--context-tiers CTX:UB:NCMOE,...` in `llama-server`. Everything below either
measures a claim made above or corrects one; the scoping's §§1-8 are otherwise
unchanged.

Source changes, all server/CLI, **zero ggml, zero model, zero kernel code**:

| file | change | LOC |
|---|---|---|
| `tools/server/server-context.cpp` | tier state, policy, reload, the tokenize lock | 190 |
| `common/arg.cpp` | `--context-tiers`, `--context-tier-reserve`, `--context-tier-down-turns` | 47 |
| `common/common.h` | `common_context_tier`, three `common_params` fields | 13 |

**~250 LOC, inside §6's 250-350 estimate.** The three `-c`/`-ub`/`-ncmoe` knobs
are all reachable by mutating `common_params` in place and calling the existing
`load_model()`, so no new plumbing was needed, and §6's optional client
`context_budget` hint (20-40 LOC in `server-task.h` / `server-schema.cpp`) proved
unnecessary -- §2.1's `task.n_tokens()` is enough. What the estimate did not
include is the vocab-lifetime lock of §9.7, which is not optional.

### 9.1 Two corrections to §6's design, both found in the code

**(a) The all-idle gate at `server-context.cpp:2807-2823` cannot be the
escalation trigger, and §6 item 3 is wrong to name it.** By the time
`update_slots()` runs, `process_new_tasks()` has already handed the arriving task
to a slot through `launch_slot_with_task()` (`:1820` sets
`slot.state = SLOT_STATE_STARTED`), so at the all-idle gate either there is no
task to read the depth from, or the slots are not idle. Worse, a reload there
would run `slots.clear()` (`load_model`, `:1238`) on a slot that owns the only
copy of the in-flight task, losing the request.

The trigger is instead **`process_single_task()`, immediately before
`get_available_slot()`** (`:2538`). That point has the same safety property the
all-idle gate was chosen for, from two independent guarantees: the function
declines every non-metrics task while the queue is yielding
(`:2376-2380`), so it never runs during an encode/decode; and the policy itself
re-checks `slot.is_processing()` for every slot and defers the task rather than
reloading if any slot is busy. It also still holds the task, so the depth is
exactly §2.1's `task.n_tokens()`.

**(b) `load_model()` must be entered on its `is_resume` path or the server
crashes on the second switch.** `load_model` calls `init()` when
`!is_resume` (`:1389-1391`), and `init()` re-registers the queue callbacks
including `on_sleeping_state`, which *appends* rather than replaces
(`server-queue.h:135-137`). A second registration means `handle_sleeping_state`
is called twice per transition, and its own `GGML_ASSERT(sleeping != new_state)`
(`:955`) then fires. The switch therefore sets a `ctx_tier_switching` flag and
`is_resume` becomes `sleeping || ctx_tier_switching`.

### 9.2 A hard ceiling §6's tier table walks straight into: the server cannot
### serve past `n_ctx_train` at all, so there is no 600K tier

§6 item 1 proposes `(122880, 2048, 24) / (311296, 2048, 32) / (614400, 1024, 35)`.
**The second and third of those cannot be served by `llama-server`.**

`n_ctx_slot()` (`server-context.cpp:4027-4035`) returns
`min(llama_n_ctx_seq(ctx), llama_model_n_ctx_train(model))`, and the prompt gate
at `:3207` rejects any request with `n_tokens >= slot.n_ctx`
(`ERROR_TYPE_EXCEED_CONTEXT_SIZE`). For this GGUF `n_ctx_train` is **262,144**,
so a 300K prompt is refused however large `-c` is.

The one thing that lifts it is custom YaRN scaling: `llama_init_from_model`
rewrites `hparams.n_ctx_train = n_ctx_orig_yarn / rope_freq_scale`
(`src/llama-context.cpp:3782-3786`) when the requested scaling differs from the
trained one. **That is why `logs/long-context-300k/server.log` reports
`n_ctx_slot = 311296` and why the 305,801-token server run was accepted at all**
-- it is the YaRN flags, not `-c 311296`, that permitted it. And
`PLAN.md:872-891` measured that those same flags destroy this model's output
below the trained length, let alone past it.

So: **every 300K and 600K number in this repo comes from `llama-bench`, which has
no such gate, or from the one YaRN server run whose output was degenerate.**
Until the rope/quality gap of §7 item 6 is closed, the deepest *servable* tier is
`-c 262144`. §6's third tier is retired, and this also settles §5.3's value table
in the negative: the 600K column is not merely 1.00x, it does not exist.

> **Superseded, 2026-09-12, see §10.** The "output was degenerate" premise above
> was a stale-binary artifact, not a real YaRN defect (`15-yarn-and-long-context-
> rope.md` §4.2 re-runs the identical failing arm on the current tree and gets
> 974 characters of coherent reasoning instead of `b` + 199 `/`). And the slot
> cap itself is liftable without touching quality: `--rope-scaling yarn
> --rope-scale 1.0001 --yarn-orig-ctx <C>` (doc 15 §6.2) satisfies
> `llama-context.cpp:3782-3786`'s gate with a theta shift four orders of
> magnitude smaller than the destructive-looking 1.1875 setting, validated
> coherent on output at 305,759 tokens (doc 15 §6.3). **`llama-server` can serve
> past `n_ctx_train` today, with this one flag, at unscaled-extrapolation
> quality.** §6's third tier is not retired after all; §9.3's table and §9.4's
> pricing below need re-deriving with it reinstated (not done here — see §10).

### 9.3 The tier table, and why it is two tiers and not three

Derived from `08-long-context-vram-budget.md` -- its `expert_vram(N) = 46,200 -
962.5N` (§2.3) and its 9,504 B/token `q4_0` KV (§1.7) -- via §1.2's reconciled
formula, at `-ub 2048` throughout, **including the pooled indexer-key cache that
landed 2026-09-12** (1,536 B/token) which doc 08 predates:

| tier | `-c` | `-ncmoe` | dense+rec | KV | pooled | compute | non-expert | experts | total |
|---|---|---|---|---|---|---|---|---|---|
| **0** | 122,880 | **24** | 3,928.73 | 1,113.75 | 180.0 | 3,692.28 *m* | 8,914.76 | 23,100 | **32,015** |
| (mid) | 163,840 | 26 | 3,928.73 | 1,485.00 | 240.0 | 4,972.28 *m* | 10,626.01 | 21,175 | **31,801** |
| **1** | 262,144 | **30** | 3,928.73 | 2,376.00 | 384.0 | 7,468.3 *i* | 14,157.01 | 17,325 | **31,482** |

(MiB. *m* = measured, `PLAN.md:308` and `:311`. *i* = interpolated.)

**Tier 0's 32,015 is the number to compare everything against, and it is proven,
not computed**: it is `PLAN.md:308`'s measured 31,834.77 plus the 180 MiB pooled
cache, and today's own 116K sparse server run
(`logs/long-context-120k-sparse/server.log`, `n_ctx_slot = 122880`, pooled cache
on) served a 116,277-token prompt in exactly that configuration. So **32,015 MiB
is known to fit this card**, which is a firmer bound than `PLAN.md:313`'s
31,835-32,101 bracket or the allocator's own 32,402 free-VRAM report.

**Tier 1 therefore has 533 MiB of margin against a configuration that is already
in production**, and that is the whole argument for `-ncmoe 30`: `-ncmoe 29`
totals 32,445, above both the proven figure and the allocator's 32,402. The one
estimated term is the compute buffer, interpolated between the **two nearest**
measured points (4,972.28 at 163,840 and 8,716.28 at 311,296, slope 0.025391
MiB/token) rather than across the whole range -- worth doing carefully, because
the segment below 163,840 is steeper (0.031250), so a single two-point fit from
122,880 under-predicts 163,840's real buffer by 3.9%. **Falsification: tier 1
fails only if the real compute buffer exceeds 8,001 MiB, i.e. 7.1% above the
estimate** (§9.10).

**Recommended table: `--context-tiers 122880:2048:24,262144:2048:30`.** The
middle row is real and fits, but a growing conversation crosses *every* boundary
it passes, and each crossing costs a full reload (§9.5) to save 4 host expert
layers, about 4% of a decode token. Two tiers means one reload per session, not
two.

### 9.4 What it is worth, priced against the measured floor rather than §5.3's
### 600K config

§5.3 priced tiering at 1.11-1.32x by comparing every depth against the
`-ncmoe 38 -ub 1024` 600K configuration. With §9.2's ceiling that comparison is
void: the deepest servable config is `-ncmoe 30 -ub 2048`, only **6** host expert
layers above tier 0, not 14, and at the same `-ub`.

At `PLAN.md:897-899`'s measured +0.4 ms/token per host-resident expert layer, and
tier 0's measured 25.36 tok/s (39.43 ms) at 8K (`PLAN.md:1267-1273`):

| actual depth | static deep tier | best tier for the depth | gain |
|---|---|---|---|
| 8K | 41.83 ms = **23.91** tok/s | 39.43 ms = **25.36** tok/s | **1.06x** |
| 118K | ~42.9 ms = **23.3** tok/s | 40.7 ms = **24.56** tok/s | **1.05x** |
| 262K | same config | same config | **1.00x** |

Prefill moves by about the same amount in the same direction:
`PLAN.md:900-901` measured -4.7% for `-ncmoe` 24 -> 32, so ~-3.5% for 24 -> 30.

**So the honest value is ~5-6% for short requests, not 1.1-1.3x.** The reason is
this project's own success: sparse FA (`PLAN.md` 2026-09-12) made decode flat with
depth, so a tier's cost is now almost entirely its `-ncmoe`, and the servable
depth range only spans 6 `-ncmoe` steps.

### 9.5 Measured cost of a switch, and the breakeven that follows

A switch is `destroy()` + `load_model()`, so its cost is a full model load, and
that cost depends far more on the load flags than on the tier:

| what | measured | where |
|---|---|---|
| reload, 21 GiB model, mmap, warm page cache | **5.44 s** and **5.27 s** | arm 1, `logs/context-tiers/cpu-arm/` |
| first load, production model and flags (`-lm none -lzm off`, `-ncmoe 24`, `-c 122880`) | **219 s** | arm 2a, `logs/context-tiers/prod-arm/` |

**219 s is the number to plan with**, and it is the load this project already
priced at "59 s with `-lzm auto`, ~4.9 min with `-lm none -lzm off`"
(`PLAN.md:646-648`) -- 3 min 39 s here, with the file warm in page cache. The
reload is the same call, and with `-lm none` there is no mmap to keep the
previous load's pages resident, so a switch should be read as **~3.5-5 minutes**.
Arm 1's 5 s reloads are not the production number; they are what shows that the
*path* costs nothing beyond the load itself.

On top of the reload, the switch throws away the prompt cache, so the triggering
turn reprefills its whole conversation. At arm 2a's measured 480.92 tok/s
(32K, tier 0) that is **~4.5 min for a 130K conversation** and ~9 min for a
250K one. A first-time long prompt would have paid that prefill anyway; a
*growing* conversation that was already cached pays it twice.

**The breakeven, and it is why `--context-tier-down-turns` defaults to 64.**
Going *down* a tier is a pure optimization: it buys 6 host expert layers x
0.4 ms = 2.4 ms per decoded token (§9.4). Against a 219 s reload that is

```
219 s / 2.4 ms per token = ~91,000 decoded tokens before the reload pays for itself
```

-- about 45 turns of 2,000 generated tokens, or 180 turns of 500. Hence 64 as a
default, and hence the advice not to lower it. Going *up* a tier is not an
optimization at all: without it the request is refused outright
(`ERROR_TYPE_EXCEED_CONTEXT_SIZE`), so its 219 s buys a capability, and nothing
in the breakeven arithmetic applies to it.


### 9.6 What the implementation does, in order

1. `--context-tiers CTX:UB:NCMOE,...` is parsed in `common/arg.cpp` into
   `common_params::context_tiers`, rejecting a tier list that is not in
   increasing context order.
2. On the **first** `load_model()`, `ctx_tier_init()` records the user's non-`-ncmoe`
   `-ot` overrides, disables `-fit` (it rewrites the same
   `tensor_buft_overrides` array the tier owns), logs the table, and applies
   **tier 0** -- so the server always comes up in the cheapest configuration,
   whatever `-c`/`-ub`/`-ncmoe` were passed.
3. Each arriving completion/infill/embedding/rerank task goes through
   `ctx_tier_update()` before a slot is chosen. The tier is
   `ctx_tier_for(task.n_tokens() + reserve)`, where reserve is the request's own
   `max_tokens` when it set one and `--context-tier-reserve` (4096) otherwise.
4. Same tier: nothing happens, which is the common case and costs one integer
   compare. Deeper tier: switch. Shallower tier: switch only after
   `--context-tier-down-turns` consecutive shallow requests (§9.5 explains the
   default of 64).
5. A switch mutates `params_base` -- `n_ctx`, `n_ubatch`, `n_batch` and a rebuilt
   override list from `llm_add_n_cpu_ffn_overrides(n_cpu_moe, ...)` -- then runs
   the sleep path's own two calls, `destroy()` and `load_model(params_base)`.
   Nothing else in the reload is new code.
6. If the new tier fails to load, the previous tier is reloaded and the request
   fails on the ordinary context-size path instead of taking the server down.

### 9.7 The one hazard this added, and how it is closed

`destroy()` frees the model but leaves `server_context_impl::vocab` pointing into
it until `load_model` reassigns it (`:1114`). HTTP threads tokenize against that
`vocab` **off the main loop**, so a reload can free the vocab under a
tokenizer. Upstream's sleep path has the same window -- it is narrow there
because sleep needs an idle timeout to fire -- but a tier switch happens exactly
when traffic arrives, which is when the window is widest.

Closed with a `std::shared_mutex` (`server_context_impl::mutex_model`): the three
HTTP tokenize sites take it shared, the reload takes it exclusively. The shared
sections are scoped to the tokenize call only and never span a `post_tasks()` or
a result wait, which is what keeps the exclusive side from deadlocking against a
request that is waiting on the main loop. `/tokenize`, `/detokenize` and
`/apply-template` are not covered (they take a vocab argument rather than reading
the field) and are left as pre-existing exposure.

### 9.8 Known limitations, stated rather than hidden

- **The prompt cache does not survive a switch.** `load_model` rebuilds `slots`
  and `prompt_cache`, so the turn that triggers the switch reprefills its whole
  conversation. This is §6's "optional extension" boundary and it stays out of
  scope: preserving KV across the reload needs `llama_state_seq_*` to round-trip
  12 conventional KV layers, 36 gated-DeltaNet recurrent states and the
  one-day-old QSA pooled indexer-key cache (§7 item 3).
- **`/props` reports the startup tier.** `server_routes::meta` is snapshotted
  once before the HTTP server accepts requests, so `slot_n_ctx` there is tier 0's
  even after an escalation. A client that sizes its request from `/props` sees a
  smaller context than the server can currently serve.
- **`max_tokens` participates in tier selection.** A client that asks for an
  enormous generation budget on a short prompt forces a deeper tier. Honoring it
  is deliberate -- the alternative is a generation that stops early at
  `STOP_TYPE_LIMIT` with `--context-shift` off -- but it makes the policy
  sensitive to a field clients set carelessly.
- **`-np > 1` tiers are per KV pool, not per slot.** With a non-unified cache the
  per-slot depth is `n_ctx / n_parallel`, so the tier boundaries would have to be
  divided by `n_parallel` to mean what they say. The server warns; it does not
  correct.
- **Waking from `--sleep` keeps the current tier**, because the wake reload
  happens before any task is visible. Deliberate: choosing tier 0 on wake would
  cost a second reload whenever the waking request is a deep one.

### 9.9 Validation, arm 1: the mechanism, on a small model with no GPU

The tier policy and the reload path were validated first on the 21 GiB
`Qwen3.6-35B-A3B` MoE with `-ngl 0 --device none -t 4`, which exercises exactly
the same code (`destroy()` + `load_model()` with mutated `n_ctx` / `n_ubatch` /
overrides) at a 5 s reload instead of a 5 min one, and let it run while the Arc
card was busy with another workstream. Tiers `1024:512:4,8192:1024:8`,
`--context-tier-reserve 64`, `--context-tier-down-turns 2`, a real six-turn
`/v1/chat/completions` conversation through
`staging/work/tier_switch_client.py`. Artifacts: `logs/context-tiers/cpu-arm/`.

| turn | prompt tokens | tier | reloaded | `prompt_n` (tokens actually prefilled) |
|---|---|---|---|---|
| 1, short | 22 | 0 | no | 22 |
| 2, short | 49 | 0 | no | 31 (cached prefix) |
| 3, +5,395 chars | 1,539 | **0 -> 1** | **yes, 5.44 s** | 1,539 (full reprefill) |
| 4, short follow-up | 1,567 | 1 | no | 32 (cached prefix) |
| 5, new conversation | 19 | 1 (held) | no | 19 |
| 6, new conversation | 22 | **1 -> 0** | **yes, 5.27 s** | 22 |

**Every claim the task asked for is in that table.** Exactly **2 switches for 2
boundary crossings** (`grep -c "switching context tier" = 2`), no spurious
reload on turns 2, 4 or 5. The server came up on tier 0 (`n_ctx_slot = 1024`)
with no `-c` given, and reported `n_ctx_slot = 8192` after the escalation and
`1024` again after the de-escalation, so the reload really did take the new
configuration. Turn 5 shows the hysteresis holding the deep tier for one shallow
request and turn 6 shows it releasing on the second. Turn 4 is the coherence
evidence: immediately after the reload and full reprefill, the model answered
from the document it had just re-read ("one for user operations and one for
order operations"), and turn 4's `prompt_n = 32` proves the prompt cache is
working normally again on the other side of the switch. No assert, abort, error
or warning in the server log, and box-wide memory across the two reloads went
45.7 -> 46.6 -> 42.4 GiB with no monotone growth, i.e. the reload gives the host
tensors back.

### 9.10 Validation, arm 2: the real model at the recommended tier boundaries

The second arm runs the recommended table itself --
`--context-tiers 122880:2048:24,262144:2048:30` -- on the production model and
the full production flags (`-ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -lm none -lzm off
-fit off -np 1`). It was planned in two parts: a session that stays inside tier 0
(**2a, done**) and a session that crosses into tier 1 with a real ~130K-token
document (**2b, not done** -- see below; the box crashed during its load). So the
switch cost and tier 1's VRAM row remain part-measured and part-computed, and
which is which is stated explicitly. Artifacts:
`logs/context-tiers/prod-arm/`, driver `staging/work/tier_switch_run.sh` with
`PROFILE=big`.

**Arm 2a, the static-config control.** The first production run used a ~32K
crossing document by mistake (the filler was sliced from the 32K benchmark
prompt, not the 300K one), which at these boundaries sits inside tier 0 -- so it
became the control the task asked for: a session that stays in one tier for its
whole life. Recorded because a control is worth as much as the crossing.

| turn | prompt tokens | `prompt_n` | prefill tok/s | decode tok/s | reloaded |
|---|---|---|---|---|---|
| 1, short | 64 | 64 | 50.4 | 22.14 | no |
| 2, +113,939 chars | 32,019 | 31,959 | **480.92** | **25.56** | no |
| 3, short follow-up | 32,047 | 32 | 43.5 | 26.12 | no |

**Zero switches over the whole session** (`grep -c "switching context tier" = 0`),
and the throughput is this project's own static-config band: 480.92 tok/s
prefill at 32K against `PLAN.md:279`'s 408.39 for the same `-ub 2048 -ncmoe 24`
on a different day, and 25.56-26.12 tok/s decode against the sparse-FA
addendum's measured 25.07-25.41 at 8K. So **`--context-tiers` on a session that
never crosses a boundary is the static configuration**, which is what the code
says it should be: `ctx_tier_update()` returns on an integer compare and the only
other always-on addition is an uncontended shared-lock acquire per HTTP request.
Turn 1's reply was correct ("The capital of France is Paris.") and turn 3 reused
the cached prefix (`prompt_n = 32`).

**Arm 2b, the real crossing at production boundaries, did NOT run, and the
reason is a serious operational incident rather than a technical obstacle.** The
crossing arm was launched at 11:32:22 with a ~130K-token document, and **the box
crashed and rebooted** during its model load. Cause, reconstructed from this
run's own memory sampler (`logs/context-tiers/prod-arm/host-mem.log`) and
recorded here because it is the largest operational hazard this project has hit:

- Arm 2a held the box at **87.9 GiB used / 38.4 GiB available** at 11:31:45 --
  its own ~53 GiB of pinned host tensors (`-lm none -lzm off` at `-ncmoe 24`)
  plus ~35 GiB of the box's unrelated services.
- Arm 2b was started **37 s after `docker stop`** was issued to arm 2a's
  container, with no check that the process had actually exited and returned its
  53 GiB. `docker stop` is asynchronous (SIGTERM, then SIGKILL after a grace
  period), so a second 53 GiB load was begun against 38.4 GiB of headroom.
- A concurrent workstream on the same box was also doing model loads at
  `-ncmoe 32` (~88 GiB with these flags), so the overcommit was very likely
  worse than one arm's worth. **But arm 2b's own arithmetic -- 53 GiB wanted,
  38.4 GiB available -- was already sufficient to overcommit the machine, so
  this is a sequencing error on this side, not a scheduling accident.**

**Standing rule this adds, and it is stronger than the GPU-idle check every
other run in this repo used.** `ps aux | grep llama` plus `docker ps | grep sycl`
is not enough, and neither is "the GPU is idle": with `-lm none -lzm off` the
binding resource is **host RAM**, one load is ~53 GiB at `-ncmoe 24` and ~88 GiB
at `-ncmoe 32`, and the box has 123 GiB with ~35 GiB already committed to
unrelated services (Immich, Minecraft, Vaultwarden). Before any load: check
`free -h`'s `available` column against the config's real requirement, and after
stopping a previous run **wait for the process to be gone and the memory to come
back**, not for `docker stop` to return.

**What arm 2b would have settled, and what is therefore still open.** Two
things, both of which the rest of section 9 marks as computed rather than
measured: (1) the switch cost at a production tier boundary end to end -- §9.5's
219 s is a *first load* at tier 0 with the same flags, which is the same call the
switch makes, but not the same measurement; and (2) **tier 1's VRAM row, the one
estimated number in §9.3's table.** For `-ncmoe 30` at `-c 262144 -ub 2048` to
fail, the compute buffer would have to exceed **8,001 MiB against the 7,468.3**
interpolated between the two nearest measured points -- **7.1% high** -- which
would also mean the buffer's slope stopped falling with depth, against three
measured points that say it falls (0.031250 MiB/token below 163,840, 0.025391
above). That is the falsifiable prediction; a single load at that tier under
`-lv 4`, reading `sched_reserve: SYCL0 compute buffer size`, settles it in about
four minutes of GPU time and no code.

### 9.11 Verdict

**Built, and worth keeping -- for the capability, not the speed.** One
`llama-server` can now come up in the fast 122,880-token configuration and
reconfigure itself to serve a request up to the model's 262,144-token ceiling,
which it would otherwise have refused. That is the whole of the value:

- **Escalation is a capability.** Without it the deep request is rejected; with
  it, it costs ~3.5-5 min of reload plus a reprefill the request would largely
  have paid anyway. Nothing else in this repo offers that, and the alternative --
  running the deep configuration permanently -- taxes every short request.
- **De-escalation is a speed optimization with a ~91,000-decoded-token
  breakeven** (§9.5), which is why it is gated at 64 consecutive shallow
  requests by default. Do not lower it.
- **The steady-state prize is ~5-6%, not §5.3's 1.11-1.32x** (§9.4), because
  §9.2's ceiling removes the deep tiers that made §5.3's table look good and
  because sparse attention already flattened decode against depth.

So section 8's recommendation stands with its sequencing reversed in one place:
item 4's "only if the deployment genuinely needs one server capable of both" is
exactly the right condition, and it is now a 250-LOC feature rather than a
proposal. **The three items ahead of it in section 8 are still ahead of it**, and
item 3 (the rope/quality gap past 262,144) has become more pointed rather than
less: §9.2 shows the server cannot even *accept* a prompt past `n_ctx_train`
without the YaRN flags this plan measured as destructive, so no amount of
placement machinery reaches 300K until that is solved.

---

## 10. Addendum, 2026-09-12 (later same day): the rope/quality gap this
## document's deepest claims all gated on is closed

`15-yarn-and-long-context-rope.md`, written after this document, resolves §7
item 6 and reverses §9.2 in the direction that matters for this proposal.
Summary, not a repeat of doc 15's own detail:

1. **"YaRN destroys this model" is retired.** The degenerate run this document
   and `PLAN.md:872-891` built on was measured on `staging/devbin` as of
   2026-09-11 21:19 — before block-granularity QSA top-k and the pooled
   indexer-key cache landed. Doc 15 §4.2 re-runs the *exact same* config,
   prompt, and tool on the current tree and gets 974 characters of coherent,
   on-task output instead of `b` + 199 `/`. The correct verdict, replacing
   §9.2's, is doc 15 §3: YaRN costs 1.2-2.9x perplexity at depth and buys
   nothing unscaled extrapolation doesn't already give — a quality tax, not a
   collapse.
2. **Unscaled extrapolation past `n_ctx_train` is coherent and accurate, not
   just untested.** Doc 15 §4 measured it directly at 305,759 tokens (16.6%
   past the 262,144 ceiling) and at 612,689 tokens (2.34x the ceiling): correct
   on every checkable structural and quantitative detail, including aggregate
   statistics (parameter ranges, global-variable names) that require attending
   across the whole prompt rather than a local window. **The 21.08 tok/s at
   300K and 12.42 tok/s at 600K this document cites throughout are throughput
   on output someone would actually want**, not on an unvalidated degenerate
   completion as §7 item 6 and §9.2 both assumed.
3. **§9.2's "there is no 600K tier" no longer holds.** The slot-cap gate
   (`n_ctx_slot() = min(n_ctx_seq, n_ctx_train)`) is still real, but doc 15
   §6.2 found a one-flag workaround that lifts it without a destructive rope
   perturbation: `--rope-scaling yarn --rope-scale 1.0001 --yarn-orig-ctx <C>`
   satisfies the gate at a theta shift four orders of magnitude smaller than
   the 1.1875 setting this document assumed was the only way to buy the slot,
   and doc 15 §6.3 validates it coherent on output at 305,759 tokens.
   `llama-server` can serve a >262,144-token prompt today, with this flag, at
   unscaled-extrapolation quality — no patch required.

**What this changes here, and what it does not.** §7 item 6 and §8 item 3 are
closed. §9.2's retirement of the third tier is reversed — §6's
`(614400, 1024, 35)` row is servable after all, via the `nearunity` flag on the
deep tier's reload. **This document does not re-derive §9.3's tier table or
§9.4's value pricing with the third tier reinstated** — both were built to
work around §9.2's ceiling, and redoing them (VRAM budget for a real three-tier
`--context-tiers`, and re-pricing against the true reachable range rather than
the two-tier `-ncmoe 30` ceiling) is a real, undone piece of analysis, not a
one-line fix. Until that is done, treat §9.3's "two tiers, not three" and
§9.4's "~5-6%, not 1.1-1.3x" as **stale, not current** — the honest value could
land anywhere between those two figures and §5.3's original 1.11-1.32x, and
nobody has re-run the arithmetic to find out which. §5's structural point about
600K (the deep tier *is* the operating point, so dynamic placement cannot beat
1.00x there) is unaffected by any of this — it never depended on §9.2.

**Also unaffected:** the async-vs-between-turns decision (§4, §8) and
`--context-tiers`'s implementation and hysteresis (§9.1, §9.5-9.11) are
independent of the rope finding and stand as validated.

---

## 11. Addendum, 2026-09-13: §10's undone re-derivation, done — a real 500K
## tier, priced

§10 flagged §9.3's tier table and §9.4's value pricing as stale once the
600K-class tier was un-retired, and left the re-derivation undone. This closes
it for a 500K (not 600K) tier specifically, since that is the depth the
deployment's own compaction point targets (`PLAN.md`'s "300K-600K, compaction
viable around 500K").

**VRAM at `D = 512,000`, measured not estimated.** §9.3's KV and pooled-cache
terms are linear in depth and reproduce exactly at every previously-measured
point (KV: `2,376.00 / 262,144 = 0.009064 MiB/token`, matching §5.2's cited
5,570 MiB at 614,400 to the decimal; pooled: `384.0 / 262,144 = 0.0014648
MiB/token`, matching §5.2's cited 900 MiB likewise) — trusted to extrapolate.
The one term that was not previously measured at `-ub 1024` in this depth
range is the compute buffer, and it is the one that decides the answer, so it
was measured directly: two `docker compose run --rm` load-only probes
(`llama-completion`, trivial prompt, `-c 512000 -ub 1024 -ctk q4_0 -ctv q4_0
-lzm off -fit off -n 1 -lv 4`, reading `sched_reserve`), `-ncmoe 38` then
`-ncmoe 32`, both against production flags on UD-IQ3_XXS. Logs:
`logs/500k-probe/ncmoe{38,32}.log`.

```
SYCL0 compute buffer size = 6,131.19 MiB   (identical at both -ncmoe values —
                                             only the graph-split count differs,
                                             98 vs. 116 splits at bs=1024)
```

| `-ncmoe` | dense+rec | KV | pooled | compute | experts (`46,200 − 962.5N`) | total | margin vs. 32,015 |
|---|---|---|---|---|---|---|---|
| 38 | 3,928.73 | 4,640.77 | 750.0 | 6,131.19 | 9,625.0 | 25,075.69 | 6,939.31 |
| **32** | 3,928.73 | 4,640.77 | 750.0 | 6,131.19 | 15,400.0 | **30,850.69** | **1,164.31** |
| 31 (not tested, arithmetic only) | — | — | — | — | 16,362.5 | 31,813.19 | 201.81 — thinner than the 266 MiB margin that caused the `-ncmoe 24`/116K OOM regression; not recommended |

**`-ncmoe 32` is the 500K tier** — both values load cleanly (no OOM, no
assert; box RAM stayed flat at 72-73 GiB available across both loads, no
swap growth), and 32 is the smaller of the two with a safe margin. This
replaces doc 08's stale `-ncmoe 28` estimate (predates `-ub 1024`, the pooled
indexer cache, and this measured compute buffer) and §6's original
`(614400, 1024, 35)` sketch row for the nearby 600K case.

**Re-priced against §9.4's own measured baseline** (25.36 tok/s / 39.43 ms at
tier 0's 8K, +0.4 ms/token per CPU-resident expert layer — same method §9.4
used, extended to the reinstated third tier, 8 layers above tier 0's 24
rather than tier-1's 6 or the retracted 600K sketch's 14):

| session phase | static `-ncmoe 32` the whole time | dynamic (best tier for depth) | gain |
|---|---|---|---|
| shallow, < 122,880 tok | 42.63 ms = 23.46 tok/s | 39.43 ms = **25.36 tok/s** | **1.08x** |
| mid, 122,880-262,144 | 23.46 tok/s (same config) | 41.83 ms = **23.91 tok/s** | 1.02x |
| deep, 262,144-512,000 | 23.46 tok/s | 23.46 tok/s (same config) | **1.00x — converges, per §5.2** |

**Verdict for a workload where sessions grow within themselves (this
deployment's stated pattern): dynamic tiering wins, not by a large margin but
for free.** The static 500K config pays its ~8% tax on *every* turn of *every*
session — including the many that never leave tier 0 — plus a larger
permanent host-RAM/VRAM footprint. The tiered config pays that tax only while
actually shallow, gets a one-time reload-plus-reprefill charge (§9.5's
measured 219 s at production flags, ×2 at most — once per boundary an
individual session actually crosses) only from sessions that grow deep, and
is identical to the static config's throughput once a session is actually at
depth. This generalizes §9.11's "capability, not speed-up" verdict rather
than overturning it: the honest number for a reinstated 500K tier is ~1.08x
for the shallow majority of traffic and 1.00x at the depth that motivated
building this at all — closer to §9.4's already-conservative 5-6% figure than
to §5.3's original pre-§9.2 1.11-1.32x table, because a 500K tier only spans 8
`-ncmoe` steps above tier 0, not 14.

**Not done here, left open:** extending `--context-tiers`' actual table/code
to a third row (`512000:1024:32`) and threading the nearunity YaRN flag
(`--rope-scaling yarn --rope-scale 1.0001 --yarn-orig-ctx 512000`, doc 15 §6.2
— required for `llama-server` specifically, not for the `llama-completion`
probes here) through so the tier is actually servable past `n_ctx_train`; and
the real crossing benchmark itself (doc 13 §9.10's arm 2b, which never
completed — the box crashed on a sequencing error, not a technical
limitation). Both are implementation, not open questions — the VRAM budget
and the value pricing above are now settled inputs to that work, not
unknowns.

---

## 12. Addendum, 2026-09-13 (later same day): the 3-tier + nearunity
## combination, mechanism-tested cheaply, one real gotcha found

§11 left two things as "not done here": wiring a third `--context-tiers` row
past `n_ctx_train` with the nearunity flag threaded through, and the real
crossing benchmark. This closes the first (mechanism only, not production
scale) and surfaces one actionable finding before the expensive real run.

**Method: the same CPU-only trick §9.9's arm 1 used**, extended to three
tiers where the third deliberately exceeds `n_ctx_train` — cheap (5-6 s
reloads) because it never touches the GPU, so it was safe to run without
disturbing anything else on the box. `Qwen3.6-35B-A3B` (same
`n_ctx_train = 262144` as the real model, confirmed by banner), `-ngl 0
--device none -t 4`, `--context-tiers 1024:512:4,8192:1024:8,280000:1024:12
--rope-scaling yarn --rope-scale 1.0001 --yarn-orig-ctx 280000`. Two growing
conversations run against the same live server (`staging/work/
tier3_nearunity_serve.sh`, logs in `logs/tier3-nearunity/`): the first
crosses tier 0 → 1 with a 1,747-token filler, the second (fresh conversation,
server already parked at tier 1) crosses tier 1 → 2 with a 17,959-token
filler — enough to force the escalation without paying for anything close to
a real 280K-token prefill.

**It works.** Exactly 2 reloads for 2 crossings, no spurious reloads on the
shallow turns in between, and the post-crossing reply was coherent and
specifically grounded in the crossed content ("~24 distinct entity types,
roughly 150-170 similar function definitions" — correct for the planted
document). No crash, no assert, no OOM.

**One real gotcha, not fatal but worth fixing before the production table
ships.** The requested `-c 280000` was padded internally to `n_ctx_seq =
280064` before the training-context comparison ran, which came in 36 tokens
*above* the nearunity-lifted ceiling (`280000 / 0.9999 = 280028`) — so the
server logged `the slot context (280064) exceeds the training context of the
model (280028) - capping` and silently served 280,028 instead of 280,000.
Not a failure (it capped and continued, didn't abort), but it means
**`--yarn-orig-ctx` needs headroom above the tier's actual `-c`, not an exact
match** — set it a few hundred to a few thousand tokens above the deepest
tier's context, e.g. `--yarn-orig-ctx 513000` against a `512000:1024:32`
tier, not `512000` against `512000` as doc 15's own single-tier methodology
would suggest by direct analogy.

**Still open, and now the only thing left:** the real crossing at production
scale and depth (§9.10 arm 2b, which never completed — the box crashed on a
sequencing error, not a technical limitation) — this addendum's mechanism
test de-risks that run but does not substitute for it.

**Update, same day: the real crossing ran, both hops, clean.** §9.10 arm 2b's
job — done this time. Pre-flight this time actually followed §9.10's own
"standing rule": confirmed zero other `llama`/SYCL processes before starting
(the crash arm had a concurrent `llama-bench` load racing it) and used mmap
rather than `-lm none -lzm off` (§11's reasoning: less host RAM at risk, and
already proven safe at a heavier `-ncmoe` in `yarn_600k.sh`). Table:
`122880:2048:24,262144:2048:30,512000:1024:32`, `--yarn-orig-ctx 513000` (the
§12-gotcha-fixed margin, not `512000`). Real growing conversations, not
synthetic fillers: a 149,549-token slice of the sibling generator's
refactor-benchmark corpus for the first crossing, the full 303,913-token
corpus for the second. Logs: `logs/tier3-prod/`.

| crossing | trigger | reload time | `n_ctx_slot` after | capped? |
|---|---|---|---|---|
| 0 → 1 | 149,549-token request | **151.84 s** | 262,144 | no (under `n_ctx_train`) |
| 1 → 2 | 305,891-token buffered depth (303,913 prompt + reserve) | **88.75 s** | **512,000** | **no** — §12's margin fix held |

Both well under §9.5's 219 s worst-case-cold estimate (warm page cache from
this session's own earlier reads of the same GGUF). **Host RAM never dropped
below 69.8 GiB available across either reload** (`logs/tier3-prod/
host-mem.log`, sampled every 15 s) — nothing close to the crash arm's 38.4
GiB. Post-crossing-1 reply was coherent and specifically grounded in the
149K-token document's real content, matching §9.10 arm 1's coherence
standard.

**One incident, attributable and non-fatal.** The client process driving the
second crossing (a 28+ minute prefill) was killed by this session's own tool
sandbox for "running low on memory" at **99.98% through the prefill**
(303,843 of 303,913 tokens, server-side `print_timing` log) — the *client*'s
sandbox, not the target box: `free -h` on the host showed 70 GiB available
at the moment of the kill, and the server (a separate container, unaffected)
finished the reload to tier 2 cleanly regardless (`n_ctx_slot = 512000`,
confirmed uncapped) and answered a follow-up request normally afterward. So
this is a tooling artifact of how this validation was driven, not a finding
about `--context-tiers`, `nearunity`, or this box — recorded because it
explains why no full completion+reply exists for the second crossing's
triggering request, only the reload and a confirmed-healthy server on the
other side of it.

**Verdict: doc 13's `--context-tiers` mechanism, extended to a third tier
past `n_ctx_train` with the nearunity YaRN unlock, is validated at real
production scale on both hops.** Combined with §11's pricing, this closes
the whole thread §10 opened: the 500K tier is real, it is safely reachable
from a cold 8K-context boot through two ordinary in-place reloads, and
nothing about adding it destabilizes the box when run one load at a time
with the pre-flight checks §9.10 already prescribed.

---

## 13. Scoping, 2026-09-13: idle-time tier de-escalation

**The gap.** `--context-tiers` as built reacts only to incoming requests.
Grep-verified: `ctx_tier_down_turns` (`server-context.cpp:946`) is touched
only inside `ctx_tier_update()` (`:1058-1130`), which only runs when a
`server_task` arrives. There is no timer anywhere in that path. So a session
that escalates to a deep tier and then simply stops sending requests sits at
that tier's full VRAM/host-RAM footprint **forever** — de-escalation needs
`--context-tier-down-turns` (default 64) *consecutive shallow requests to
actually arrive*, and idle time with zero requests never advances that
counter. Separately, `--sleep-idle-seconds` (`common/common.h:665`, default
`-1` = off) is a real timer, but it does something different: after N idle
seconds it fully tears the model down (`handle_sleeping_state`,
`server-context.cpp:969-983`, calling `destroy()`) and reloads on the next
request — but reloads at **whatever tier was active when it went to sleep**,
not tier 0 (`ctx_tier_set()` is never called on the wake path), so it doesn't
substitute for de-escalation either. Neither knob does "idle → cheap."

**Where the hook already exists.** `server_queue::start_loop()`
(`server-queue.cpp:278-365`) runs a `while(true)` loop that, whenever the
task queue is empty, wakes once a second (`max_wait_time =
std::chrono::seconds(1)`, `:322-360`) purely to re-check
`should_sleep()` (`:287-296`) against `time_last_task` — a timestamp already
tracked and already excluding time spent inside `update_slots()` (the
"shift instead of reset" comment at `:311`). This is the natural, already-
proven insertion point for a second, independent idle check.

### 13.1 Three designs, smallest first

**A — zero code, operational workaround.** Down-turns is purely
request-count-based, so an external keepalive (a cron or sidecar issuing a
trivially short completion request every idle interval) already produces
de-escalation after 64 of them, using code that exists today. Free, but
fragile (the interval has to be tuned against `--context-tier-down-turns`,
it adds constant request/log noise, and it needs an operator to run and
maintain a second process). Worth naming as the thing to do *right now* with
zero engineering, not as a real fix.

**B (recommended) — piggyback the existing sleep transition.** Change
`handle_sleeping_state(true)` (`server-context.cpp:969-983`, the "entering
sleeping" branch, immediately before its `destroy()` call) to also call
`ctx_tier_set(0)` when `ctx_tiers.size() >= 2`. Since `destroy()` +
`load_model(params_base)` already runs on wake regardless, this costs
**one call, no new timer, no new config surface** — the existing
`--sleep-idle-seconds` becomes the de-escalation trigger too, and it fixes
§9.8's documented limitation ("waking from `--sleep` keeps the current
tier") as a side effect rather than a separate task. Estimated **10-15
LOC**, LOW risk — it reuses the exact reload path both `--context-tiers` and
`--sleep` already exercise individually; the only genuinely new thing is
calling `ctx_tier_set()` from a code path that has never called it before,
and that function's whole body is three field assignments plus an override-
list rebuild (`:1036-1052`), not new state machinery.

The real limitation: it conflates two timeouts a deployment might want
decoupled — "stop paying the deep-tier tax" (cheap-ish, keeps the server hot)
vs. "free the GPU entirely" (expensive to undo). A deployment that wants to
de-escalate quickly but not fully unload for a long while cannot express
that with Option B alone.

**C — a genuinely separate idle-tier timer.** Clone `should_sleep()`'s shape:
a second config field (`context_tier_idle_seconds` in `common/common.h`,
alongside `sleep_idle_seconds` at `:665`; a `--context-tier-idle-seconds`
flag in `common/arg.cpp` next to `--sleep-idle-seconds` at `:3863-3869`), a
second threshold check in `start_loop()`'s idle-wait loop (`server-
queue.cpp:322-360`) alongside `should_sleep()`, reusing the same
`time_last_task` the sleep check already reads — no second timestamp needed.
On trigger, invoke a new callback (a second `std::vector<std::function<void()>>`
parallel to `callback_sleeping_state`, or reuse that vector's registration
pattern via a new `on_idle_tier_timeout()` method on `server_queue`,
mirroring `on_sleeping_state()` at `server-queue.h:135-136`) that
server-context.cpp wires in `init()` next to the existing
`queue_tasks.on_sleeping_state(...)` registration (`:1580-1582`) to call
`ctx_tier_set(0)` + `destroy()` + `load_model(params_base)` under
`mutex_model` — the same three calls `ctx_tier_update()` already makes
(`:1103-1106`), just invoked from a timer instead of a request. Estimated
**60-90 LOC** across `server-queue.h/.cpp`, `server-context.cpp`,
`common/common.h`, `common/arg.cpp`. LOW-MEDIUM risk: the reload machinery
itself is fully validated (§9, §12); the new surface is the second timer and
its wiring, structurally identical to the first one already in the tree.

### 13.2 Open questions, ordered by how much they'd change the design

1. **Does an idle-triggered switch need the "all slots idle" guard
   `ctx_tier_update()` uses** (`server-context.cpp:1084-1090`,
   `slot.is_processing()`)? Almost certainly not — the trigger only fires
   from inside the queue-empty branch of `start_loop()`'s own loop, which by
   construction cannot observe a busy slot at that instant (per the same
   argument `should_sleep()` already relies on for the identical sleep case).
   Worth a one-line assertion rather than a real check, not a new hazard.
2. **Should de-escalation jump straight to tier 0, or step down one tier at
   a time?** Recommend straight to tier 0 for the idle case (unlike the
   request-driven down-turns path, there is no next request to size against,
   and zero traffic for the whole idle window is a stronger signal than "64
   shallow requests happened to arrive in a row"). Keeps Option B trivial and
   Option C's callback body simple.
3. **Default value.** `--sleep-idle-seconds` defaults to off (`-1`); a new
   idle-tier timeout should default off too, consistent with this project's
   "off by default, opt in" posture for every reload-triggering flag so far
   (`--context-tiers` itself defaults to unset/unchanged behavior). Suggest
   documenting a starting value (a few minutes) rather than picking one here
   — it trades reload frequency against how long a deployment is willing to
   keep the deep tier's VRAM/host-RAM pinned against an idle session, which
   is a deployment-specific judgment call, not a technical one.
4. **Interaction with `--np > 1`.** `ctx_tier_update()` already warns and
   does not correct for `n_parallel > 1` with a non-unified cache
   (`:1013-1017`, §9.8). An idle-tier timer inherits the same caveat
   unchanged — not a new problem, just one that would need re-checking if
   this project ever moves past `-np 1`.

### 13.3 Recommendation

**Do B now, keep C as a later option if a deployment actually needs the two
timeouts decoupled.** B is a ~10-15 LOC change to code already proven twice
over (§9's mechanism validation, §12's production validation), fixes an
existing documented wart as a free side effect, and directly answers "does
it drop back down while idle" with "yes, on the next `--sleep-idle-seconds`
wake, once this lands" rather than "no." Not implemented here — this section
is scoping only, per the request that produced it.

**Built and validated, same day.** 5-line change in `handle_sleeping_state`
(`tools/server/server-context.cpp`, entering-sleep branch, immediately
before `destroy()`): `if (ctx_tiers.size() >= 2 && ctx_tier_cur != 0) {
ctx_tier_set(0); }`, logged. Rebuilt clean via `staging/work/devbuild.sh
tools/server/server-context.cpp`. Verified with the CPU-only mechanism rig
(`staging/work/tier3_deescalate_serve.sh`, `--sleep-idle-seconds 6` added to
§12's tiers-plus-nearunity setup): escalated 0 -> 1 (`n_ctx_slot 1024 ->
8192`), went idle, the server log showed `sleeping: also resetting context
tier 1 -> 0`, and the next request's reload confirmed `n_ctx_slot = 1024` --
woke at tier 0, not tier 1. Logs: `logs/tier3-deescalate/`. Not yet
re-verified against the real production model/tiers (the CPU rig proves the
logic; a production run would only reconfirm the already-validated reload
path, per §9's and §12's own reasoning for why the cheap rig suffices).

---

## 14. Addendum, 2026-09-13 (later still): §13's own fix caused a real
## production double-reload, and the "obvious" repair is unsafe -- reverted
## to a simpler one

**The regression, with real evidence.** Deployed to the live `llm-b70`
production container (the sibling `coding-agent` repo's `docker-compose.
qwen4exp-moe.override.yml`), §13's fix produced exactly the failure mode its
own tradeoff analysis predicted was possible but didn't quantify: a
conversation already at tier 1 (123,606 tokens) went idle, slept (correctly
logging `sleeping: also resetting context tier 1 -> 0`), and when the *same*
conversation resumed, paid **two full reloads back to back** -- wake into
tier 0 (`load_model` 23:08:03 -> `n_ctx_slot = 122880` ready 23:11:42, **3m
39s**), immediately followed by `ctx_tier_upd: switching context tier 0 -> 1
(request is 123606 tokens + 16384 reserved)` and a second reload to tier 1
(ready 23:14:28, **2m 45s** more). **6m 25s of pure reload before the first
prompt token processed**, for a request one reload straight into tier 1
would have served. Real user-reported symptom: "starting a new context takes
a really long time, even if the prefill amount is low" -- the delay was
never prefill, it was two model loads.

**The fix that looks obvious does not work, and this section explains why in
enough detail that nobody re-attempts it without re-deriving the same
finding.** The design considered: don't reset to tier 0 at sleep entry; defer
the reload past wake-exit instead; force exactly one reload, sized by the
first real task's own depth, from inside `ctx_tier_update()`. This requires
`wait_until_no_sleep()` (`server-queue.cpp:115-128`, called from
`server_res_generator`'s constructor, `server-context.cpp:4409-4415`, which
every HTTP handler that touches `ctx_server` constructs before doing
anything else) to return *before* the model is reloaded.

That breaks a real invariant several call sites depend on without
re-checking it. `handle_completions_impl` (`server-context.cpp:4471-4481`,
and the same shape at `:5068`, `:5617`) takes only a **shared** lock on
`mutex_model` and reads `ctx_server.vocab` immediately after
`create_response()` (which is what calls `wait_until_no_sleep()`) returns --
the shared lock protects against a *concurrent* reload racing the read, not
against reading a pointer that was never valid in the first place. In the
current (working) design this is safe because `handle_sleeping_state(false)`
calls `load_model(params_base)` **synchronously, before** flipping `sleeping
= false` -- so by the time `wait_until_no_sleep()`'s condition
(`!sleeping`) is satisfied, `vocab` (set at `server-context.cpp:1286`,
inside `load_model()`) is already valid. `destroy()`
(`server-context.cpp:936-942`) frees `model_tgt` but never resets `vocab`
itself, so it becomes a **dangling pointer**, not a null one, the moment
`destroy()` runs. Deferring the reload past `wait_until_no_sleep()`'s return
means every one of those shared-lock tokenize sites would read that dangling
pointer on the HTTP thread, with no writer holding the exclusive lock at
that moment to serialize against -- a real use-after-free, not a race
that occasionally loses. `/tokenize`, `/detokenize` and `/apply-template`
already have a version of this gap today (§9.7's "pre-existing exposure" --
they never call `wait_until_no_sleep()` at all, so they can read a stale
`vocab` for the entire sleep duration already); the deferred-reload design
would have extended the same class of gap to every completion-shaped
endpoint too, for the length of "however long until the next real request,"
which is unbounded. Not attempted.

**What shipped instead: the smaller, safe half of the fix.** Delete the
`ctx_tier_set(0)` call §13 added at sleep entry (`server-context.cpp`,
`handle_sleeping_state`, the `if (new_state)` branch) and nothing else. Wake
still eagerly calls `load_model(params_base)` before flipping `sleeping`
(unchanged, still holds the invariant above) -- but since nothing reset the
tier at sleep time, it reloads straight back into whichever tier the session
was actually at, in one shot. This fully fixes the reported bug (a resumed
session gets exactly one reload, sized correctly) at the honest, disclosed
cost of reintroducing part of §9.8's original limitation: a genuinely new,
shallow session that happens to be the *first* request after a long sleep
now wakes into whatever tier was last active rather than the cheap default,
until 64 consecutive shallow requests bring it back down. Net effect vs. the
pre-§13 baseline: strictly better (same reload-into-last-tier behavior on
wake, same as before §13 ever shipped) -- §13's own contribution is fully
retracted, not replaced by something new. A real fix for the shallow-first
case exists in principle (thread the request's known token count through to
`wait_until_no_sleep()` so wake reloads directly into the tier *that
specific request* needs, preserving the "always loaded before returning"
invariant) but requires plumbing depth information through every
`server_res_generator`-constructing call site across the server, which is
real surgery, not a five-line fix -- left as a scoped-but-undone option, not
attempted here.

**Validated, CPU-only rig, both real-world cases -- not just the one that
broke.** `staging/work/tier3_deescalate_serve.sh`, same setup as §13
(`--sleep-idle-seconds 6`, tiers `1024:512:4,8192:1024:8,280000:1024:12`).

| case | scenario | reloads | result |
|---|---|---|---|
| A | escalate to tier 1, sleep, **same-depth** request resumes | **1** | wakes straight into `n_ctx_slot = 8192` -- no tier-0 stop, no second switch |
| B | escalate to tier 1, sleep, **short/shallow** request arrives first | **1** | wakes into `n_ctx_slot = 8192` (the stale tier, not 0) -- correct response, no crash, the disclosed tradeoff, not a bug |

Rebuilt via `staging/work/devbuild.sh tools/server/server-context.cpp`,
clean. Logs: `logs/tier3-wakefix/`. **Not yet redeployed to the live
production container** -- the fix is validated on the cheap rig only, per
this project's own established practice of proving a change cheaply before
touching the box that's actually serving traffic; redeploying `llm-b70` is a
separate step.

---

## 15. Addendum, 2026-09-13 (later still): §14's fallback was correctly
## rejected -- a real fix, not another compromise

**§14's fallback traded one bug for a smaller one, and that's not good
enough.** Reloading into whatever tier was last active fixes the reported
case (a deep session resumes) but reintroduces exactly the cost the original
`--sleep-idle-seconds` de-escalation was built to remove: a genuinely new,
short session that happens to arrive first after a long sleep now pays a
full reload into an unnecessarily deep tier, every time, until 64 consecutive
shallow requests bring it back down. Correctly called out as not acceptable
as a final answer.

**The opening §14 didn't have: a cheap depth signal that exists *before* the
model-liveness gate, without touching it.** `wait_until_no_sleep()`
(`server-queue.cpp:115-128`) still returns only once a fully loaded model is
guaranteed -- §14's use-after-free finding stands, unchanged, and nothing
here reopens it. What's new is recognizing that every completions-shaped
route handler already holds its **raw, unparsed** `req.body` before it ever
calls `create_response()` (which is what triggers the wait) -- no
tokenization, no even JSON-parsing required to get a usable depth estimate,
just `req.body.size()`. That estimate can ride along into the wake decision
without moving where the model becomes valid at all.

**Design, as implemented:**

1. `handle_completions_impl` (`server-context.cpp:4446`, its own
   `create_response()` call at `:4459`) computes `depth_hint =
   data.dump().size() / 3.5` (this project's own measured chars/token, doc
   13 §12) from the already-parsed `data` it receives as a parameter --
   turned out to be **dead code for every real request**, see the false
   start below.
2. `create_response()` (`server-context.h:182` declaration,
   `server-context.cpp:4724` definition) and `server_res_generator`'s
   constructor (`:4419`) both grew an `int32_t depth_hint = -1` parameter,
   threaded straight into `wait_until_no_sleep(depth_hint)`.
3. `server_queue` gained `wake_depth_hint` (`server-queue.h`, next to
   `req_stop_sleeping`), set under `mutex_tasks` by `wait_until_no_sleep()`
   at the same point it sets `req_stop_sleeping = true` -- first caller
   during a given sleep wins the hint, same as it already wins the wake
   trigger, no new arbitration needed. `callback_sleeping_state` widened
   from `std::function<void(bool)>` to `std::function<void(bool, int32_t)>`
   (one registration site each in `server-context.cpp`'s `init()` and
   `server_routes`'s constructor -- both updated, the second ignores the
   hint, it only needs the boolean) so `start_loop()`
   (`server-queue.cpp:322-350`) can hand the hint to the wake callback: `cb(true,
   -1)` entering sleep (no hint exists yet), `callback_sleeping_state[i-1](false,
   depth_hint)` on the way out, reading `wake_depth_hint` once and resetting
   it to -1 immediately after, under the same lock.
4. `handle_sleeping_state(bool new_state, int32_t depth_hint)`
   (`server-context.cpp:968`): on wake, if `ctx_tiers.size() >= 2 &&
   depth_hint >= 0`, call `ctx_tier_set(ctx_tier_for(depth_hint))` before
   `load_model(params_base)` -- skipped, falling through to §14's unchanged
   fallback (reload into whatever `ctx_tier_cur` already is), whenever no
   caller offered a hint.

**A false start worth recording, not hiding: the first version of this did
not work, and the reason is itself informative.** Point 1 above --
computing the hint inside `handle_completions_impl` -- compiled clean and
looked right, but every one of the ~7 route lambdas that call it
(`post_completions`, `post_completions_oai`, `post_chat_completions`,
`post_infill`, `post_responses_oai`, `post_transcriptions_oai`,
`post_anthropic_messages`, `server-context.cpp:5030-5251`) already call
`create_response()` themselves, **first**, per this file's own standing
convention ("IMPORTANT: all lambda functions must start with
`create_response()`", `:4850`) -- and that first, hint-less call is the one
that actually triggers `wait_until_no_sleep()`. By the time
`handle_completions_impl`'s own hinted call runs, `sleeping` is already
`false` and `wait_until_no_sleep()`'s fast path (`if (!sleeping) return;`)
skips the hint entirely -- confirmed the hard way, first live test of the
"shallow request after a deep sleep" case still woke into the stale deep
tier, unchanged from §14. **Fix: moved the estimate one level up**, to each
of those 7 outer lambdas' own `create_response()` call, computed from
`req.body.size() / 3.5` (available even before JSON parsing, cheaper than
point 1's `data.dump()`). `handle_completions_impl`'s own hinted call stays
as harmless defense in depth for any future caller that reaches it directly.
Scope held to completions/infill-shaped endpoints only, per the original
directive -- `post_control`, embeddings, slots, health, metrics, tokenize,
etc. all keep calling plain `create_response()` and fall through to the
unchanged §14 fallback, which is the correct behavior for requests with no
real prompt to estimate from.

**Validated, CPU-only rig, all three cases with a log trace for each --
including the one that was missing.**
`staging/work/tier3_deescalate_serve.sh`, same setup as §13/§14
(`--sleep-idle-seconds 6`, tiers `1024:512:4,8192:1024:8,280000:1024:12`).

| case | scenario | reloads | log evidence |
|---|---|---|---|
| A | deep session resumes after sleep | **1** | straight to `n_ctx_slot = 8192`, no `switching context tier` line follows |
| B | no-hint endpoint wakes the server (`POST /props`) | **1** | no `waking: sizing reload` line (hint absent, as designed) -- falls back to stale `n_ctx_slot = 8192`, matching §14's already-validated behavior, unchanged |
| C | **new, shallow completion request arrives first** | **1** | `waking: sizing reload from tier 1 to 0 for a 37-token hint`, then `n_ctx_slot = 1024` directly -- no stop at tier 1 |

Case C is the one the earlier report was held to, and it's the one that was
actually broken by §14's fallback: a 19-token request (`req.body.size()` 130
bytes / 3.5 = 37) now wakes the server directly into the cheap tier instead
of the stale deep one, with the same single-reload cost either way. Rebuilt
clean via `staging/work/devbuild.sh tools/server/server-queue.h
tools/server/server-queue.cpp tools/server/server-context.h
tools/server/server-context.cpp`. Logs: `logs/tier3-wakefix2/`.

**Known imprecision, not a correctness issue.** `req.body.size() / 3.5`
overshoots the real token count by however much JSON structure (field
names, message-role wrappers, tool definitions) inflates the raw byte count
over the actual prompt text -- this can only push the estimate **up**, which
can only pick a tier *at least as deep* as the request needs, never too
shallow. The failure mode this design cannot produce is the one that
matters (waking into a tier too small for the request); the one it can
produce (waking one tier deeper than strictly necessary, on a borderline
request) costs at most one extra tier's reload margin, not correctness.

**Not yet redeployed to the live production container**, same posture as
every change in this file -- validated on the cheap rig, `llm-b70`
redeployment is a separate, later step.
