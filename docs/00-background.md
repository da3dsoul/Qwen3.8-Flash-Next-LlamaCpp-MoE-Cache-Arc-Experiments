# Background

Condensed from a prior research thread in the sibling `coding-agent` repo
(`../coding-agent/docs/06-moe-cache-research/`, 2026-09-09), re-grounded here
for a project where the technique is actually applicable. Read that source
folder for full citations/transcripts; this doc keeps only what this project
needs.

## 1. The mechanism: llama.cpp MoE expert cache (CUDA-only, source fork)

Fork: `GenerelSchwerz/llama.cpp`, branch `moe-cache`. Implementation:
`ggml/src/ggml-cuda/moe-cache.cu`/`.cuh` (~11,400 lines) + a small diff to
`ggml-cuda.cu`'s `ggml_cuda_mul_mat_id` dispatch site.

**What it does:** a new `ggml_backend_buffer_type` intercepts a model's
`ffn_*_exps` tensors (the routed MoE expert weights). Cold experts live in
CPU **pinned** host memory; a fixed-size **VRAM slot pool** (`N` slots *per
expert tensor*, uniform stride = the largest expert slab in the model) holds
the hot ones. On a routing miss, the missing expert's slab is copied
VRAM-ward on a dedicated copy stream (events for readiness) and the LRU (or
frequency-decayed) victim is evicted. Two execution paths:

- **Legacy path**: router IDs are read back to host, unique experts per op
  are deduped, plain LRU eviction. Slower — the fork instruments this cost
  explicitly.
- **"Grouped decode" fast path**: the admission planner runs *as a CUDA
  kernel on-device* (warp-shuffle reductions, `cudaMalloc`'d per-expert
  frequency-counter state that halves every 16 decode steps), eliminating
  the host round-trip, plus CUDA graph capture/replay for the certified
  decode path and grouped admission across an expert's gate/up/down/scale
  banks in one decision.

**Why it wins when it wins:** expert matmuls move from CPU (DDR bandwidth,
~50-90 GB/s) to GPU (~500-900 GB/s), PCIe paid only on cache misses. This
requires either a skewed expert-selection distribution or (more simply, and
apparently what dominates in the wiki's own 16GB recipes) a cache that's
*large relative to the live expert working set* — "dynamic residency beats
static placement" is doing most of the work, not exotic hot-expert exploitation.

**Flags** (fork adds these; full table in the source research doc):

| Flag | Meaning |
|---|---|
| `--moe-expert-cache-size N` | N expert slabs per tensor kept resident in VRAM, LRU eviction. `0` = off (default). **Overrides `--n-cpu-moe`/`--cpu-moe` when enabled — mutually exclusive strategies, not stacked.** |
| `--moe-expert-cache-l2-pinned-mb N` | mmap-only pinned-host L2 budget (MiB) |
| `--load-mode none` | no mmap; cold experts in host RAM; **required** for the grouped-decode fast path and for combining with speculative drafting |
| `--load-mode mmap` | cold experts read from file-backed mmap; **disables grouped decode** |
| `-fit off` | mandatory — vLLM-style auto-fit doesn't know about cache pools |
| `-ngl all` | offload everything; the cached buffer type intercepts experts regardless |
| `GGML_CUDA_MOE_FREQUENCY=0` | force pure LRU instead of frequency-decay |

VRAM cost ≈ `slots × slab_stride × cached_tensor_count` + pool overhead.
Upstream llama.cpp's own `--n-cpu-moe`/`-ot ...=CPU` is the baseline this
beats — that's **static** placement (a tensor lives on CPU for the whole
run); the fork is **dynamic** residency + GPU compute for hot experts.

**Known failure modes / limits** (all confirmed, not speculative):

- **CUDA-only.** No SYCL/Vulkan/CPU-backend implementation exists anywhere.
  There is a design to study, not code to port as-is. This is exactly the
  gap `docs/research/01-sycl-backend-feasibility.md` in this repo exists to
  close.
- **Only helps when experts must spill.** A model that already fits fully in
  VRAM measured **-6.5%** with the cache turned on (pure overhead, no
  spillage to amortize). Concretely: **check "does the chosen quant of
  Qwen3.8-Flash-Next's ~125B-parameter expert stack fit in 32GB?" before
  investing further** — if it does at some quant level, the cache is not
  worth building for that quant, same as it wasn't worth building for the
  dense Qwen3.8-27B model in the other project.
- **Fights speculative/draft decoding.** A verify batch's expert reads touch
  the union of experts routed to by every draft token (K× the unique-expert
  set of a single token), which can bypass/thrash the cache. The one
  reported *working* combination elsewhere used a **CPU-side** drafter, not
  a GPU-resident one.
- **Regresses prefill 14-66%** in the fork's own numbers — large-batch
  prefill has a wide unique-expert set per op, closer to a worst case for a
  small resident pool.
- **Lost at matched VRAM budget** in one external report (llama.cpp
  discussion #24528, hybrid CPU/GPU design, 19.7 vs 35.4 tok/s vs. plain
  resident layers) — the fork's all-GPU-execution design is claimed to be
  what avoids that specific failure, but that claim is the fork's own and
  wasn't independently re-verified.

## 2. The model: Qwen3.8-Flash-Next

Alibaba, released 2026-08-26, a Qwen4-architecture preview. Two independent
components that a viral video conflated as one:

1. **The MoE FFN stack** — ~125B parameters, ~6B activated per token. This is
   what the expert cache (§1) could apply to. Uses Gated DeltaNet linear
   attention + Qwen Sparse Attention (both need SYCL kernel coverage
   independent of the expert-cache question — see
   `docs/research/02-model-support-status.md`).
2. **A ~51B-parameter n-gram/PLE (per-layer-embedding) lookup table** — not a
   neural network, a deterministic hash-keyed lookup (current + preceding
   tokens → embedding rows), effectively free compute, cheap to keep on SSD
   via mmap since access is small and prefetchable. In llama.cpp: GGUF keys
   `%s.ple.ngram_size`, `%s.ple.heads_per_ngram`, `%s.ple.layers`,
   `%s.ple.head_vocab_sizes`; tensors `blk.%d.ple_key`, `blk.%d.ple_value`,
   `blk.%d.ple_conv1d`. **A completely different tensor family from
   `ffn_*_exps` — `--moe-expert-cache-size` never touches it, and it needs no
   porting work of its own beyond "does the PLE lookup op run on SYCL"**
   (also a `docs/research/02` question).

125B + 51B ≈ the "177-billion-parameter model" framing. The GGUF conversion
demoed publicly is `unsloth/Qwen3.8-Flash-Next-GGUF`, quantized `UD-IQ3_XXS`
(82GB file) — that quant was chosen to fit a 12GB-VRAM/64GB-host-RAM demo
box, not because it's the natural choice for our hardware.

## 3. Target hardware and known driver risk

Intel Arc Pro B70, **32GB VRAM** (Battlemage/Xe2-HPG), single GPU — same
physical card used by the sibling `Qwen3.8-vLLM-KVarN-MTP-Experiments`
project (that project's dense model uses 17.8GB of it; this project is a
**separate, standalone llama.cpp deployment on the same card**, not expected
to run concurrently with the vLLM deployment).

**Real, already-paid-for lesson from that sibling project, directly relevant
here**: Intel's compute-runtime can *silently* migrate GPU allocations into
host RAM under memory pressure (implicit USM migration). On this exact
machine that behavior caused **two host-wide OOM kills and GPU hangs
requiring a physical reboot**, and the other project now hard-disables it
(`EnableSharedSystemUsmSupport=0`, `EnableImplicitMigrationOnFaultableHardware=0`,
`NEOReadDebugKeys=1`) as a standing safety requirement.

This does **not** rule out the expert cache's pinned-host-memory design —
that design is *explicit, bounded, developer-controlled* host allocation +
async copy, which is a fundamentally different risk profile from the driver
silently deciding to spill on its own. But it means: **never** develop or
test this against a config that allows implicit migration, size the pinned
host pool explicitly and conservatively, and treat "what happens when the
pinned pool + KV cache + everything else exceeds host RAM" as a correctness
gate to clear early, not late — a repeat of that incident on this box is the
single most avoidable failure mode available to this project.

## 4. Why this project exists (vs. the sibling project's "doesn't apply" verdict)

The sibling `Qwen3.8-vLLM-KVarN-MTP-Experiments` investigation concluded the
expert cache doesn't apply *there* because: the served model is dense (no
experts, nothing to cache), it already fits fully in VRAM, it fights the MTP
speculative decoding that deployment depends on, and vLLM has no hook point
equivalent to ggml's pluggable buffer types (porting would mean partially
reimplementing vLLM's MoE weight-loading path, not writing a patch).

Two of those four objections don't apply here by construction: Qwen3.8-Flash-
Next genuinely has ~125B params of routed experts (unlike the dense sibling
model), and llama.cpp *does* have the ggml buffer-type hook point the fork's
mechanism depends on (unlike vLLM) — the open question is only whether
`ggml-sycl` implements that hook point with enough parity to receive a port,
which is what `docs/research/01-sycl-backend-feasibility.md` establishes.
The other two objections — "does it already fit in VRAM" and "does it fight
speculative decoding" — are **not** automatically resolved and must be
re-checked for this model/quant/hardware combination specifically before
committing to build anything; see `PLAN.md` Phase 0.
