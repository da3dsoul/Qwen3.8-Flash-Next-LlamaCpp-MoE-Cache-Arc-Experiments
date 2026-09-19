# Qwen3.8-Flash-Next MoE Expert Cache — llama.cpp SYCL Port (Arc Pro B70)

This project started as an attempt to port a CUDA-only MoE expert-cache
design (host-pinned cold experts + a fixed VRAM slot pool for hot ones) to
llama.cpp's SYCL backend, to run Qwen3.8-Flash-Next (the `qwen4exp`
architecture — a hybrid Gated-DeltaNet/MoE/hyper-connection model) well on an
Intel Arc Pro B70. Partway through, real usage data made it clear the actual
deployment need was serving **very long context (300K-600K+ tokens)**, not
maximizing short-prompt decode throughput — and the project's center of
gravity shifted accordingly. Both halves are documented here, including the
negative result.

Everything is logged chronologically and in detail in **[`PLAN.md`](PLAN.md)**
— that is the primary source of truth for this project, not this README.
This file is an entry point.

## Headline findings

- **The custom SYCL MoE expert cache was built, works, and is correct** —
  but it only ties, not beats, plain `-ncmoe` static CPU/GPU expert
  placement (a stock upstream flag, zero custom code). This is a real
  negative result and the project's own course-correction because of it:
  `-ncmoe` became the production default, and effort moved to what actually
  mattered. The cache code is still in the tree and still valid; see
  `docs/research/13` §3 for exactly why it stopped being the answer.
- **MTP speculative decoding was implemented from scratch** for `qwen4exp`
  (no upstream support existed for this architecture's hyper-connection
  output head) and gives a real, if modest, decode speedup.
- **The project's actual value ended up being long-context serving.** Three
  results compound:
  - Sparse QSA flash attention got decode flat with depth instead of
    collapsing — a measured ~4-5x cumulative win at 116K tokens across the
    fixes that got there (`docs/research/10`, `docs/research/11`,
    `docs/research/12`).
  - A YaRN/RoPE false alarm got resolved: "YaRN destroys this model" turned
    out to be a stale-binary artifact, and **unscaled RoPE extrapolation past
    the model's trained context length is coherent and quantitatively
    accurate at 305K and 612K tokens** — no patch needed
    (`docs/research/15`).
  - A dynamic `--context-tiers` mechanism was designed, built, and validated
    **end to end at real production scale** — real ~150K- and ~300K-token
    requests, real in-place model reloads, on the real Arc B70, with an
    idle-timeout de-escalation fix layered on top so a session that goes
    quiet drops back to the cheap tier instead of sitting pinned at whatever
    depth it last reached (`docs/research/13` §9-§13).
- **It's deployed.** This build now runs as the production `local-smart`
  backend behind this box's LiteLLM router, serving real traffic. That
  deployment's config lives in the sibling `coding-agent` repo (out of scope
  here) — this repo is the research, the port, and the validated artifacts
  it's built from.

## Repo layout

| Path | What it is |
|---|---|
| `PLAN.md` | The full, chronological project log. Every phase, every dated update, every course-correction, in order. Start here for *how* a conclusion was reached; the docs below are where individual questions got answered in depth. |
| `docs/00-background.md` | Ground rules and the CUDA-fork mechanism this project ports from, condensed from a prior research thread. |
| `docs/research/*.md` | 14 deep-dive scoping/investigation docs, each answering one specific technical question (index below). |
| `docker/` | Dockerfiles for the SYCL/Vulkan runtime and build environments, plus the standalone kernel-equivalence test harnesses used during development. |
| `patches/llama.cpp.patch` | **The actual code.** A unified diff of every change made against the pinned upstream llama.cpp checkout — see "The code" below. |
| `staging/work/*.sh` `*.py` | The real benchmark/validation drivers this project's numbers came from — reload/tier-crossing tests, YaRN quality probes, per-depth decode-scaling sweeps, the production metrics tailer, etc. Reproducible, not one-off scratch. |
| `staging/work/bench{120k,300k,600k}/` | The actual benchmark corpora (real, not synthetic-filler, ~116K/~300K/~612K-token prompts) every long-context number in this repo was measured against. |
| `logs/` | Benchmark reports (`.md`) and a curated set of representative/cited raw logs backing specific claims — not a full dump (see below). |

Not committed here (see `.gitignore`): the vendored `src/llama.cpp/` and
`src/moe-cache-fork/` checkouts (~1.2G of upstream source), GGUF model
weights (~209G), compiled `staging/devbin*/` binaries, and the bulk of raw
debug/profiling logs that aren't cited as evidence anywhere or aren't small
enough to be worth the noise. The curated logs kept here total under 25 MB.

### The code

This project's actual deliverable — the SYCL MoE expert cache, sparse QSA
flash attention, `--context-tiers`, the idle de-escalation fix, MTP support
for `qwen4exp`, and everything else — lives as **local modifications on top
of a pinned upstream llama.cpp checkout**, not as a standalone codebase. That
vendored tree isn't committed (see above), so the changes are captured as
`patches/llama.cpp.patch`: a single diff covering 31 modified files and 13
new files (mostly under `ggml/src/ggml-sycl/` — `moe-cache.{cpp,hpp}`,
`fattn-sparse.{cpp,hpp}`, `hyper_connect.{cpp,hpp}`, `topk-radix.{cpp,hpp}`,
`mmid-hybrid.{cpp,hpp}`, `expert-pool.{cpp,hpp}` — plus the server/arch/model
plumbing that wires them in).

To reproduce: check out `ggml-org/llama.cpp` at commit `6d9c82ea2` (pinned
2026-09-09, the commit both the Vulkan `ggml_vk_fill` fix and the SYCL
IQ-quant silent-fallback fix landed at or after — see `PLAN.md`'s Phase 0.0
for why that date matters), apply the patch, and build via
`docker/Dockerfile.sycl` or `staging/work/devbuild.sh`'s incremental-build
pattern.

## docs/research index

Each of these answers one specific question, with its own verdict/summary
section near the top (usually `## 0. ...`) — read that first, go deeper only
if you need the evidence.

| Doc | Answers |
|---|---|
| `01-sycl-backend-feasibility.md` | Can the CUDA MoE-cache design be ported to `ggml-sycl` at all? (Yes — structurally symmetric, one real gap: `GGML_OP_TOP_K`.) |
| `02-model-support-status.md` | What does `qwen4exp` actually need from llama.cpp, and what's already upstream vs. fork-only? |
| `03-cuda-fork-moe-cache-outline.md` | Structural outline of the CUDA fork's own `moe-cache.cu`/`.cuh` implementation being ported from. |
| `05-hybrid-cpu-gpu-mul-mat-id-scoping.md` | Scoping a hybrid CPU/GPU `MUL_MAT_ID` dispatch (upstream RFC #24528) as an alternative/complement to the cache. |
| `06-vllm-migration-and-kvarn-viability.md` | Is migrating to vLLM + the sibling project's KVarN port worth it instead of this llama.cpp effort? |
| `07-qsa-sparse-attention-scoping.md` | Should Qwen Sparse Attention (the "lightning indexer") be pursued on this backend, and when? |
| `08-long-context-vram-budget.md` | The joint KV-cache + MoE-expert-placement VRAM budget across context depth, computed rather than cited. |
| `09-long-context-prefill-fix-scoping.md` | Scoping the fix for long-context prefill/TTFT. |
| `10-sycl-sparse-attention-scoping.md` | Measuring and scoping QSA sparse attention specifically on SYCL. |
| `11-long-context-decode-profile.md` | Where a 116K-token decode step's time actually goes, decomposed component by component. |
| `12-decode-depth-scaling.md` | The *slope* of decode cost vs. context depth, not just a point measurement — what has to change to meet a 300K/600K throughput bar. |
| `13-dynamic-context-aware-expert-placement.md` | The largest doc — scopes, builds, and validates `--context-tiers` (dynamic context-length-aware expert placement), the 500K tier, the nearunity YaRN unlock, and idle-based de-escalation, ending in a real production-scale validation run. |
| `14-usage-based-static-expert-placement.md` | A complementary idea: static placement informed by real per-expert usage skew rather than context length. Scoped, not built. |
| `15-yarn-and-long-context-rope.md` | Resolves the YaRN/RoPE question directly: is scaled or unscaled extrapolation the right call past the trained context length? |

## How to reproduce / use

- **Rebuild the binary:** pin the commit noted above, apply
  `patches/llama.cpp.patch`, build with `docker/Dockerfile.sycl` (full clean
  build) or `staging/work/devbuild.sh <file>` (incremental, ~seconds, needs a
  running build-stage container).
- **Re-run a specific benchmark:** most claims in `PLAN.md` and
  `docs/research/*.md` cite the exact driver script under `staging/work/`
  that produced them (e.g. `tier_switch_run.sh`, `qsa_sparse_bench.sh`,
  `yarn_600k.sh`) — these are real, runnable scripts, not pseudocode.
- **Long-context corpora:** `staging/work/bench120k/`, `bench300k/`,
  `bench600k/` hold the actual prompt files (and their generator scripts)
  used for every 116K/300K/600K-token measurement in this repo.
- **The tier mechanism specifically:** `--context-tiers CTX:UB:NCMOE,...`
  (see `docs/research/13` §9.6 for the exact CLI surface) is the single
  highest-value artifact here if you just want the production config —
  the recommended table, per `docs/research/13` §11 and `PLAN.md`'s
  2026-09-13 VRAM-margin correction, is
  `122880:2048:25,262144:2048:30,512000:1024:32` (tier 1's `-ncmoe` moved
  24 -> 25 after a later addition ate the config's VRAM margin; see
  `PLAN.md` for the measured before/after), with
  `--rope-scaling yarn --rope-scale 1.0001 --yarn-orig-ctx 513000` to unlock
  the third tier past the model's native context ceiling.
- **Protecting decode from silent page-cache eviction:** `-lzm off` alone
  only guarantees tensors are read at load time, not that they stay
  resident — under host memory pressure from other tenants they can get
  evicted afterward, reintroducing a page fault mid-decode (`PLAN.md`,
  2026-09-15 update, measured live via `mincore()`). Add
  `-lm mmap+mlock --mlock-experts-only` (container needs `CAP_IPC_LOCK` and
  an unlimited/raised `memlock` ulimit) to pin just the `--n-cpu-moe` expert
  tensors — ~25-33 GiB depending on tier, not the full model — leaving the
  28.8 GB PLE table on ordinary evictable page cache, since its sparse
  per-token access tolerates the occasional cheap fault.

## Current status / what's open

This is a research repo, not a finished product — real open items, most
recorded as explicit next steps in their own docs:

- **The O(`n_kv`) host-side term in `set_input_qsa` this bullet used to name
  as the largest remaining lever has already been fixed, and it was smaller
  than first estimated.** Bucketed grouping + dropped dead columns +
  run-length bias + shift/mask for the power-of-two ratio measured 5.13 →
  2.37 ms/token at `n_kv = 614,656` (2.17x, reproduced to 2.8% across two
  runs), gated by an in-call correctness oracle against the general
  algorithm (6/6 sections passed) (`PLAN.md`, 2026-09-12 later update).
  But the ~19.7 tok/s projection this bullet used to cite doesn't hold: the
  same update found the term is only 31.9% of the excess depth slope, not
  all of it. The bigger lever that update also found — dropping `-lm none`
  for mmap, measured 17.27 vs 12.42 tok/s on the *unfixed* code — is now
  moot for a different reason, not because it was wrong: production moved
  off `-lm none` entirely to `-lm mmap+mlock --mlock-experts-only`
  (`PLAN.md`, 2026-09-15 update). What's actually still open: the fix
  combined with the current mlock config was never cleanly isolated
  end-to-end — 600K decode has a measured ~1.5x run-to-run spread on
  unmodified code (`PLAN.md`, 2026-09-12 later update), so a single current
  tok/s number for 600K would be quoting noise. The one number that
  reproduces is the host-instrument measurement above (2.76 ms/token saved,
  4.8% of the faster arm's token time).
- **MTP + K-quant pathology: leading hypothesis tested and falsified, cause
  still open.** MTP speculative decoding paired with a UD-Q3_K_XL target
  measures ~7x worse than the target alone at matched VRAM (`PLAN.md`,
  2026-09-11 update). A code-reading pass proposed a specific mechanism
  (`ggml/src/ggml-cpu/iqp.cpp`'s batched `mul_mat_id` fast path is I-quant-only
  and only activates at ≥8 rows/expert, a threshold only MTP's verify batches
  reach) — but inspecting the actual production GGUF (`gguf-py` against the
  real file, not assumed from its name) found **zero `Q3_K` tensors in
  `UD-Q3_K_XL` at all**: `ffn_gate_exps`/`ffn_up_exps` are `IQ3_XXS` (already
  fast-path-eligible) and `ffn_down_exps` is `IQ4_NL`/`Q8_0`, identical to the
  working `UD-IQ3_XXS` quant's own `down_exps` type. `down_exps` is also
  permanently ineligible for the fast path regardless of type — its `ne[0]`
  is 640 (`moe_intermediate_size`), and the path hard-requires
  `ne[0] % 256 == 0` (`ggml/src/ggml-cpu/iqp.cpp:1135`) — but that's equally
  true in both the working and broken quant, so it can't explain the
  difference either. **The panel-path theory does not hold for this model**;
  the real cause of the 7x cliff is unresolved, and the other two originally-named
  hypotheses (KV-cache type mismatch; near-VRAM-ceiling thrashing) remain the
  live suspects. Confirming either now needs a real GPU-loaded MTP+Q3_K_XL
  decode measurement on production hardware — not yet attempted, since the
  GPU is currently occupied by live production traffic (`PLAN.md`, 2026-09-19
  update). Separately: `Q3_K` support was still added to the fast path
  (`ggml/src/ggml-cpu/iqp.cpp`, `iqp_decode_q3_K`), correctness-validated via
  `test-backend-ops`'s dual-path check (888/888 `MUL_MAT_ID` tests pass) — a
  real, tested addition, but it measures ~1.00x (no speedup) on this CPU and
  doesn't apply to this deployment's files regardless, since no `Q3_K` tensors
  exist in them.
- **`docs/research/13` §13 Option C** (a fully decoupled idle-tier timer,
  independent of `--sleep-idle-seconds`) was scoped but not built — Option B
  (piggybacking the existing sleep timeout) was built and validated instead,
  and is what's actually deployed.
- **`docs/research/14`'s usage-based static placement** was scoped as a
  promising, cheap-to-check complementary idea (estimated ~1.1-1.3x, ~70% of
  the infrastructure already exists) but the one number that would decide
  it — how skewed real expert usage actually is for this model — was never
  measured.
- Several docs flag their own measurements as single-run (n=1) or otherwise
  under-replicated where a real hardware/timing measurement was involved;
  each says so explicitly rather than presenting a point estimate as more
  certain than it is. Check a doc's own "what's left open" section before
  treating a number as load-bearing.
