# Plan: Qwen3.8-Flash-Next MoE expert cache on Arc Pro B70

Status: **planning complete, nothing implemented.** Grounded in
`docs/00-background.md` and the two research docs in `docs/research/`
(`01-sycl-backend-feasibility.md`, `02-model-support-status.md`), both dated
2026-09-09. Re-read those before starting — this plan cites but does not
repeat their evidence.

## Why this project is a better bet than it looked at first

Two facts, confirmed after this plan was scoped, changed the shape of the
project for the better:

1. **Architecture support is upstream, not fork-only.** `qwen4exp` (Qwen3.8-
   Flash-Next's real architecture name in llama.cpp) is fully merged into
   `ggml-org/llama.cpp` master, and the `moe-cache` fork tracks master closely
   (a 2-line diff against `llama-arch.cpp`). There is no fork-vs-upstream
   architecture split to reconcile.
2. **Almost nothing about the SYCL port is actually blocked.** Of the ~14
   CUDA→SYCL component questions in `docs/research/01`, all but one are
   "portable" or "needs a small, well-precedented amount of new code." The one
   real blocker — SYCL graphs are hard-disabled for any MoE model — already
   has a 39-line open upstream PR (#25089) fixing it, **already validated on
   an Arc Pro B70 with a Qwen MoE model**, stalled only on a maintainer
   rebase request.

The honest risks are elsewhere: an **unresolved correctness regression**
(#24168) on hybrid SSM+MoE architectures — Qwen3.8-Flash-Next's exact
category — on Battlemage; a `GGML_OP_TOP_K` hard cap (`k≤32` vs. the model's
`k≈2051`) that forces part of every full-attention layer onto the CPU on
SYCL specifically (Vulkan doesn't have this limit); and a real, if
comfortably-bounded, host-RAM budget. None of these are cache-design
problems — they're bring-up problems that would exist even without the
cache, which is exactly why they're front-loaded into Phase 0/1 below.

## Ground rules carried from `docs/00-background.md`

- **Never set `GGML_SYCL_USM_SYSTEM=1`.** That's the implicit-migration path
  named for Battlemage that caused the sibling project's two host-wide OOM
  kills. Everything this project needs (`sycl::malloc_host`) is a different,
  explicit mechanism and does not require it. Assert it's unset at every
  startup; check the `GGML_SYCL_USM_SYSTEM: %d` banner line in every log.
- **Explicit, bounded, developer-controlled host allocation only.** Chunk the
  cold-expert pinned tier into ≤2 GiB `sycl::malloc_host` allocations (a
  Level Zero requirement for copy/compute overlap, not a style choice — see
  `docs/research/01` §2.4) and size the total against measured host RAM
  before running anything, not after.
- **Descope graph capture/replay from the first working version.** It's the
  riskiest, least-precedented component, it's off by default upstream with
  the maintainers' own "no better performance" note, and the legacy
  (non-graph) cache path doesn't need it at all. Land and measure the cache
  without it first; treat graph replay as a Phase 3 add-on whose value is
  measured in isolation before investing further.

---

## Phase 0 — Go/no-go gates before writing any cache code

Everything here is either already answered by the research docs or is a
cheap, targeted check. **Do not start Phase 1 until every item below has an
answer.** This phase can kill the project outright, and that's the point of
doing it first.

### 0.1 Reproduce (or rule out) issue #24168 — the single highest-priority check

`docs/research/01` §6.3: an open, still-unresolved llama.cpp issue reports
gibberish/crash output specifically on **hybrid SSM+MoE architectures
including `qwen3next`** on **Arc Pro B60** (Battlemage sibling to our B70).
`qwen3next` is the architectural predecessor Qwen3.8-Flash-Next's Gated-
DeltaNet-plus-MoE design is closest to. Pure-MoE models (no SSM component)
are reported unaffected — so this is specifically about the hybrid
combination, which is exactly what we're building on.

**Action:** build vanilla upstream llama.cpp SYCL, no cache patch, and run
the smallest available `qwen3next`-family GGUF (or a small Qwen3.8-Flash-Next
quant if one exists small enough to sanity-check quickly) on the actual B70.
Compare output against the CPU backend on an identical prompt/seed.

**Go/no-go:** if this reproduces and the upstream issue thread offers no
workaround, **stop here.** A caching layer cannot fix a correctness bug one
level below it. If it doesn't reproduce, or reproduces only under conditions
we can avoid (specific op fusion flags, a specific quant, etc.), proceed and
document the exact avoidance condition — Phase 1 must not silently drift back
into it.

### 0.2 SYCL vs. Vulkan backend decision (`GGML_OP_TOP_K`)

`docs/research/02` §2.3, §6: Qwen Sparse Attention needs `ggml_top_k(width≈2051)`
on all 12 full-attention layers. SYCL hard-caps `k≤32` (a shared-local-memory
limit, not a tunable — `docs/research/02` quotes the kernel's own comment
explaining why); Vulkan already has radix-select and handles arbitrary `k`.
On SYCL this forces a CPU round-trip per full-attention layer per ubatch —
cheap at decode (~1.5 MiB/token), *crippling* at prefill (~3 GiB/ubatch at
`n_ubatch=2048`, per `docs/research/02` §2.3's estimate).

Three options, cost order:
1. **Disable QSA entirely.** Free (a GGUF without indexer tensors runs dense
   attention). QSA currently provides no speed benefit on *any* backend today
   anyway (`docs/research/02` §2.1 — the sparse compute path is TODO'd out
   upstream; only the mask is sparse). Costs quality only, not throughput.
   **Confirm first whether a CLI flag actually exposes this** —
   `docs/research/02` §6 flags this as unverified; check `common/arg.cpp`.
2. **Target Vulkan instead of SYCL for the whole project.** Sidesteps the
   `top_k` problem outright, at the cost of every §1–§6 SYCL-specific finding
   in `docs/research/01` needing a Vulkan re-derivation (different
   abstraction: Vulkan compute shaders, not SYCL kernels — a different, not
   necessarily smaller, porting effort, and `docs/research/01` was not
   written against Vulkan). Also note `docs/research/01`'s issue #26581: full
   decode attention is memory-latency-bound and *identical* on Vulkan and
   SYCL on this exact card — so Vulkan is not a free performance win outside
   the `top_k` question.
3. **Port a radix-select `top_k` to SYCL**, using Vulkan's
   `topk_radix_select.comp` as a reference shader. Self-contained, the
   highest-value single upstream contribution this project could make, but
   it's new kernel work with no existing precedent to copy from (unlike the
   MoE router port in §4.3 of `docs/research/01`).

**Action:** measure the actual `GGML_OP_TOP_K` CPU-fallback cost on SYCL with
`llama-bench` (reads the `layer %d is assigned to device ... (usually due to
missing support)` diagnostic already built into llama.cpp —
`docs/research/01`'s auto-fallback note, §2.5 equivalent) before choosing.
If prefill is genuinely crippled and QSA can't be cheaply disabled, this
decides SYCL vs. Vulkan **before** any cache-porting work starts, since it
determines which backend that work targets.

**Go/no-go:** pick one of the three options and write down why. Don't let
this stay open into Phase 1.

### 0.3 Quant choice and expert-residency sizing

`docs/research/02` §5 already answers this precisely — **no quant fits fully
in 32 GB VRAM** (smallest published quant's expert set alone is 39.85 GB),
but the *dense* core (everything except experts and the PLE table) is only
3.9–5.5 GB, leaving a genuinely large slot pool against a 40–77 GB expert
working set. This is the favorable regime the cache is built for — the
opposite of the sibling project's dense-model case.

**Action:** pick a starting quant. Doc 02 §5.6 recommends **UD-Q2_K_XL or
UD-IQ3_XXS** (~49–53% expert residency at 32 k context) as the starting
point, with UD-IQ1_S (~61% residency) worth a later A/B once correctness is
established. Avoid UD-Q4_K_XL and above initially (29.6% residency, and
penalized by the `moe_intermediate_size=640` K-quant-incompatibility gotcha
in doc 02 §5.5).

### 0.4 Host RAM budget — verify before assuming

`docs/research/02` §5.1 confirms the 28.8 GB PLE table is, **by default**,
kept in host memory / mmap'd from the GGUF, not force-loaded. If that stays
true under whatever load flags the cache setup ultimately requires, the PLE
table costs page-cache pressure, not a hard RAM allocation — its actual
access footprint is tiny (~2.7 KB/token, per the video-findings research in
the sibling project). **This is not yet verified for our exact flag
combination** — the fork requires `--load-mode none` for the cache's cold
expert tier, and it's unconfirmed whether that flag is global (forcing the
PLE table to fully load too) or per-tensor-class.

**Action:** confirm `--load-mode none`'s scope before sizing host RAM. If it
only affects `ffn_*_exps`, host RAM budget ≈ cold-expert tier alone (~20–26
GB at the Phase 0.3 quant choice) — comfortable on most homelab boxes. If it
forces the PLE table to fully load too, budget ≈ 28.8 GB + cold-expert tier
(~50–55 GB) — confirm the actual box has that much free before proceeding,
and treat "PLE forced into RAM" as a design smell worth fixing upstream
rather than living with.

### 0.5 Cheap hardware probes to run once, early

- Does the B70 report `aspect::ext_oneapi_graph` (updatable) or only
  `aspect::ext_oneapi_limited_graph`? One-line runtime probe
  (`docs/research/01` §8 Q1). Doesn't block Phase 0–2, but decides which of
  the two graph-replay designs (§5.6 constant-arg slot table vs. VMM remap)
  is even available when Phase 3's stretch goal is attempted.
- Confirm `SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC` is present in our oneAPI
  toolchain version (`docs/research/01` §8 Q4) — without it, SYCL graphs are
  refused even for plain matmul, independent of the MoE-specific fix.

### Phase 0 exit criteria

- [ ] 0.1: #24168 does not reproduce, or reproduces only under an avoidable
      condition (documented).
- [ ] 0.2: backend decided (SYCL, or Vulkan, or SYCL+QSA-disabled) with a
      measured `top_k`-fallback cost backing the decision.
- [ ] 0.3: starting quant chosen.
- [ ] 0.4: host RAM budget computed against the actual box's free RAM, with
      `--load-mode` scope confirmed.
- [ ] 0.5: graph-aspect and async-alloc probes run and recorded (informs
      Phase 3, not blocking).

**If 0.1 fails outright: stop. This is the one gate that can end the project
before Phase 1 regardless of everything else.**

---

## Phase 1 — Base inference, no cache, establish the baseline to beat

Goal: get Qwen3.8-Flash-Next producing correct output on the chosen backend
(SYCL or Vulkan per 0.2), using upstream's existing **static** expert
placement (`--n-cpu-moe` / `-ot ...=CPU`) — no cache code written yet. This
is the baseline the eventual cache must beat; per `docs/00-background.md` and
`docs/research/03` (sibling project), that's the correct baseline, not "no
offload at all" (which doesn't fit regardless).

**Tasks:**
- Build llama.cpp against the Phase 0.2 backend decision, at the Phase 0.3
  quant, with `--n-cpu-moe` set to keep experts on CPU and the dense
  core + PLE table placed per upstream defaults.
- Correctness: fixed-prompt, fixed-seed output compared against the CPU
  backend (or a CUDA reference run if one is reachable) — not bit-exact
  (different backends won't be), but coherent and free of the #24168 garbling
  pattern. Re-run the 0.1 test model here too as a regression check now that
  more of the graph is active.
- Benchmark: prefill and decode tok/s at a couple of context lengths, using
  `llama-bench`. This number is the one everything in Phase 3 is measured
  against.
- Resolve the QSA CLI-flag question from 0.2 concretely if not already done.
- If targeting SYCL: capture the `layer %d is assigned to device ...`
  fallback log and get a real number for the `top_k` CPU round-trip cost
  (turns 0.2's estimate into a measurement).

**Go/no-go:** is baseline correctness clean, and is the static-CPU-offload
baseline already fast enough that the cache's added complexity isn't
justified? (Unlikely given the video's own reported gap between static
placement and the cache, but check — the cache is not free to build, and
this project should stop here rather than proceed on faith if the baseline
already looks acceptable.)

---

## Phase 2 — Cached buffer type: pinned host tier + VRAM slot pool

Goal: a working `ggml_backend_buffer_type` that holds expert tensors with
cold copies in chunked `sycl::malloc_host` pinned memory and hot copies in a
fixed-size VRAM slot pool — **no dispatch/admission logic yet.** Validate
with direct tensor read/write tests, not full model inference.

Grounded in `docs/research/01` §1–§2, portable/needs-new-code items 1a–2d:

- New buffer type implementing the standard vtable (§1.1 — `ggml-sycl`
  already has three buffer types to copy the shape from: device, split,
  host). Use `tensor->extra` (`ggml_tensor_extra_gpu`, already present in
  `common.hpp`) for per-tensor cache metadata, following the split-buffer-type
  precedent.
- Widen the **three** `get_name`-function-pointer identity checks plus the
  one SYCL-only debug assert (§1.2) so the scheduler and `cpy_tensor`
  recognize the new buffer type. This is the same three-site problem the
  CUDA fork already solved — read what it did there first (see Phase-2 note
  below) rather than re-deriving it.
- Pinned host tier: `sycl::malloc_host`, chunked into ≤2 GiB allocations
  (§2.4), with a slab→(allocation, offset) map. No `cudaHostRegister`
  equivalent exists (§2.3) — allocate-and-fill, never try to pin an mmap in
  place; this also means the fork's mmap-only L2 tier
  (`--moe-expert-cache-l2-pinned-mb`) has no SYCL analogue and is out of
  scope.
- VRAM slot pool: fixed-size, uniform stride = largest expert slab (mirrors
  the CUDA fork's design per `docs/00-background.md` §1), backed by ordinary
  device buffer-type allocation to start (§7 in the feasibility doc's table
  notes `sycl_ext_oneapi_virtual_mem` / `ggml_sycl_pool_vmm` as an available
  but optional refinement — not required for a first working version).
- **Before writing this phase's code, actually read the fork's
  `ggml_cuda_mul_mat_id` diff and the buffer-type portion of
  `moe-cache.cu`/`.cuh`.** `docs/research/01` was explicitly written without
  reading that source (its author's own open item §8 Q13) — it reasoned from
  `docs/00-background.md`'s secondhand description. Closing that gap first
  will save real time here.

**Validation:** write/read known expert slabs through the new buffer type
end-to-end (host↔device), confirm eviction/re-admission round-trips
byte-correctly, before touching the dispatch path.

---

## Phase 3 — Dispatch, admission, and (as a stretch goal) graph replay

Goal: correct, then fast, end-to-end decode through the cache. Two
sub-phases, deliberately sequential — legacy path first, so there's always a
working, measurable fallback before attempting the higher-effort fast path.

### 3a. Legacy path (host round-trip, plain LRU)

- Hook `ggml_sycl_mul_mat_id` (`docs/research/01` §1.3 — the confirmed single
  chokepoint, structurally a near-line-for-line analogue of the CUDA fork's
  interception site) to recognize the cache buffer type, read router IDs back
  to host, dedup unique experts per op, and do plain LRU admission/eviction
  against the Phase 2 buffer type.
- Dedicated copy queue: `docs/research/01` §3 — this needs genuinely new code
  (no second queue exists anywhere in `ggml-sycl` today), constructed from
  the **same `sycl::context`** as the compute queue (§3.3's documented
  cross-context silent-no-op trap — this is the single easiest mistake to
  make in this phase and the hardest to notice, since it fails silently
  rather than erroring). `sycl::event` + `ext_oneapi_submit_barrier({event})`
  for readiness signalling (§3.1, both already used in-tree elsewhere —
  follow the reorder-path precedent at `ggml-sycl.cpp:4121-4157`).
- **Correctness gate before moving on:** full-model decode through the cache
  vs. the Phase 1 static-placement baseline, same prompt/seed, checked for
  output match (not just "doesn't crash"). Also re-run the #24168 regression
  check one more time — the cache changes memory-residency patterns for
  exactly the tensor class that bug concerns.
- **Benchmark gate:** does the legacy path already beat Phase 1's baseline?
  If not, something is wrong before adding more complexity — don't proceed
  to the fast path on the assumption it'll fix a regression the legacy path
  itself has.

### 3b. Device-side "grouped decode" fast path

- Use `ggml_sycl_mul_mat_id_mmvq_fused()` / `mul_mat_vec_q_moe`
  (`docs/research/01` §1.3 — confirmed to already keep router IDs on-device,
  a better insertion point than CUDA's own). The redirect is described as
  literally one line: `vx_base + i02*expert_weight_stride` becomes a lookup
  into a device-resident slot table.
- Frequency-decay admission planner as an ordinary SYCL kernel using
  `dpct::permute_sub_group_by_xor` sub-group reductions (`docs/research/01`
  §4 — lowest-risk component, and `topk-moe.cpp` is a documented, in-tree
  CUDA→SYCL port of the MoE *router* to copy the method from). **Audit every
  reduction for the 32→16 lane-width change** (`WARP_SIZE=16` on all Intel
  targets, not 32 — the single most likely silent-bug source in this
  sub-phase per §4.2).
- Watch the reorder-timing interaction flagged in `docs/research/01` §8 Q12:
  the lazy Q4_K per-expert SoA reorder must happen once on the cold host
  copy, not on every admission — get this wrong and every cache hit pays a
  reorder cost that should have been amortized at load time.

### 3c. Graph capture/replay — explicit stretch goal, measured before investing further

Per `docs/research/01` §5's own recommended sequencing:

1. Rebase PR #25089 (already validated on an Arc Pro B70 with a Qwen MoE
   model — `docs/research/01` §5.4) locally onto current master.
2. Set `GGML_SYCL_ENABLE_GRAPH=1` and **measure whether graph capture alone
   changes decode throughput** on 3a/3b's already-working cache. This one
   number decides whether the rest of this sub-phase is worth doing at all —
   the upstream maintainers' own default assessment is "no better
   performance," and that should be trusted until our own measurement says
   otherwise.
3. Only if step 2 shows a real win: implement expert re-pointing between
   replays using the **constant-argument slot-table design**
   (`docs/research/01` §5.6, option four — a fixed device-side `slot_of[]`
   table the admission kernel writes into, so the graph's kernel arguments
   never change and only `aspect::ext_oneapi_limited_graph` is required, not
   the stronger updatable-graph capability). This is the design the research
   explicitly recommends over per-node `dynamic_parameter` updates (not
   exposed by `ggml-sycl` today, §5.5) or VMM remapping under a fixed virtual
   address (unverified `unmap`/`map` legality and latency while a graph is
   live, §5.6).

If 0.1's #24168 avoidance condition or any correctness gate from 3a breaks
under graph capture specifically, that's a signal to abandon 3c rather than
debug around it — this sub-phase is explicitly the lowest-priority, most
speculative part of the whole project.

---

## Phase 4 — Correctness and benchmark gates before calling this done

Adopting the sibling project's gate discipline (`docs/00-background.md`
references `../coding-agent/docs/`'s "Gate 1/2/3" correctness-battery
convention) — this phase is where that discipline pays for itself, since a
memory-residency cache is exactly the kind of feature that can look fast and
be silently wrong.

- **Correctness battery:** fixed prompt/seed decode at several context
  lengths, cache cold vs. warm, deliberately forced eviction storms (fill the
  slot pool past capacity, confirm output is still correct, not just
  non-crashing). Compare against the Phase 1 static-placement baseline as the
  correctness oracle, not against "does it look plausible."
- **Host-RAM-pressure gate, run deliberately conservatively the first time:**
  start well under the Phase 0.4 computed budget, monitor system memory in
  real time, and ramp up only after confirming there's no repeat of the
  sibling project's implicit-migration OOM pattern. Confirm
  `GGML_SYCL_USM_SYSTEM` stays `0` and the ≤2 GiB pinned-chunk rule is
  actually being honored by the allocator, not just assumed.
- **Speculative-decoding interaction, explicitly re-tested, not assumed:**
  `docs/00-background.md` §1 records that the CUDA fork's own report found
  the expert cache **fights** GPU-resident MTP-style draft heads (a verify
  batch's union-of-experts read pattern can bypass/thrash the cache), with
  the one reported working combination elsewhere using a CPU-side drafter.
  Unsloth's `qwen4exp/mtp` branch (unverified contents per
  `docs/research/02` §6 item 7) is the natural thing to test this against
  when it's reached — but treat cache+MTP as a combination to *measure*, not
  assume works, and be prepared for the honest answer to be "pick one, not
  both," same as the sibling project's dense-model analysis flagged as the
  most load-bearing risk for that (inapplicable) case.
- **Prefill regression check:** the CUDA fork's own numbers show 14-66%
  prefill regression under the cache (`docs/00-background.md` §1). Confirm
  whether the SYCL port shows the same pattern, and whether it's acceptable
  given this deployment's actual workload (long-context prefill matters a
  lot if this is meant to serve real conversations, not just decode
  benchmarks).
- **Final benchmark comparison:** Phase 1 baseline vs. Phase 3a (legacy
  cache) vs. Phase 3b (fast path) vs. Phase 3c (+graph, if pursued), at the
  Phase 0.3 quant and at least one higher-residency quant (e.g. UD-IQ1_S) as
  a sensitivity check on the doc 02 §5.4 residency-vs-quant table.

**Only after this phase clears** does it make sense to write up results and
decide whether to invest in upstreaming any of it (the SYCL `top_k` port
from 0.2, or a rebase-and-resubmit of #25089, are both plausible standalone
contributions independent of whether the full cache project succeeds).
