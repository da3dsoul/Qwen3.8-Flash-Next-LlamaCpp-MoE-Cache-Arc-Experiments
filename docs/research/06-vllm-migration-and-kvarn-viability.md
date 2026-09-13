# vLLM Migration and KVarN Reuse — Evidence-Based Verdict

Date: 2026-09-11. Scope: three questions, in dependency order.

1. Is the sibling project's **KVarN** work (`/media/da3dsoul/Golias/AIProjects/Qwen3.8-vLLM-KVarN-MTP-Experiments`)
   reusable for *this* project's problem — the **MoE expert cache**? (§1)
2. Does **vLLM support Qwen3.8-Flash-Next's actual architecture** — and should this project migrate to it? (§2, §3)
3. Separately: is KVarN's technique transferable for **extending usable context via YaRN**, which is a
   genuinely different problem (KV-cache memory footprint, not expert-weight placement)? (§4)

Method, same rules as `03-cuda-fork-moe-cache-outline.md`: every claim below is either (a) a file read
directly this pass with a `path:LINE` citation, (b) a GitHub object fetched live via `gh` this pass with its
number and timestamp, or (c) arithmetic on those. Anything that is a connection *I* drew rather than something
a source stated outright is flagged **inference**. Third-party measurements posted by strangers on GitHub are
labelled as such every time they appear — they are evidence, not verified results.

**Nothing was run.** No GPU work, no vLLM launch, no benchmark, no code modified.

---

## 0. The verdicts, up front

**KVarN is not reusable for the MoE-cache problem, and the near-misses are near-misses.** KVarN is a
4-bit-key/2-bit-value **KV-cache quantization** scheme for stretching context length, applied to a **dense**
27B model. It has no MoE component, no expert-weight residency component, and no CPU/GPU compute-dispatch
component. The preliminary read in the task framing is confirmed in full. What *is* genuinely reusable is a
large body of **B70-specific hardware-tuning knowledge** and a well-argued **SYCL-vs-Triton-on-Xe2
verdict** — knowledge, not code. §1.

**KVarN is also not the lever for long context here, for a different and more interesting reason.**
llama.cpp already ships 4-bit KV quantization (`q4_0`/`q4_1`/`iq4_nl`), and this model's KV cache is
**2.7× lighter per token than the sibling's** because only 12 of 48 layers have one and it has 2 KV heads,
not 4 — so `-ctk q4_0 -ctv q4_0` puts a full 1M-token YaRN context at ~6.75 GiB. KVarN's extra bit-shaving
would save ~1.8 GiB of that. The real long-context blocker is not memory at all: **QSA sparse attention is
implemented but deliberately disabled** in this repo's `qwen4exp.cpp:943-946`, so attention runs dense over
the whole context. §4.

**vLLM supports this architecture — and my first answer to this was wrong.** vLLM merged
`Qwen3.8-Flash-Next` upstream as the `qwen4_exp` model package on **2026-08-31** (PR
[#53896](https://github.com/vllm-project/vllm/pull/53896), +19,425/−451 across 124 files). The local clone at
`/media/da3dsoul/Golias/AIProjects/vllm` is pinned at `v0.28.1rc0-122-g9d0fe9bac8`, dated **2026-08-30** —
**one day before the merge** — so a thorough, correct, case-insensitive grep of the local clone returns a
clean zero for `qwen4exp` and is *completely misleading*. This is the single most important methodological
finding in this document: the local clone was ~12 days stale and that was enough to invert the answer. §2.1.

But the architecture being supported does **not** make migration the right move, because of the layer below
it:

- vLLM **hard-refuses to load this model on Intel XPU**. `vllm/models/qwen4_exp/__init__.py:30-31` raises
  `NotImplementedError("Qwen4Exp currently supports CUDA and ROCm only")`. §2.3.
- The XPU port exists only as an **unreviewed draft PR** ([#55068](https://github.com/vllm-project/vllm/pull/55068),
  opened 2026-09-03, last touched 2026-09-04, zero reviews). §2.4.
- vLLM's answer to "the model does not fit in VRAM" is **UVA zero-copy weight offload — which is a permanent
  cache miss, not a cache.** It is the *inverse* of this project's entire thesis. §2.7.

**Recommendation: stay on llama.cpp. Do not migrate now. Re-evaluate when PR #55068 merges.** §5.

---

## 1. KVarN reuse verdict

### 1.1 What KVarN actually is (read, not assumed)

From the sibling project's own README
(`/media/da3dsoul/Golias/AIProjects/Qwen3.8-vLLM-KVarN-MTP-Experiments/README.md:1-12`):

> using **KVarN** (4-bit-key / 2-bit-value KV-cache quantization, ported here from CUDA to Intel Arc) to push
> far beyond the model's native 262K-token context — up to **~550K tokens** — by shrinking the KV cache
> footprint instead of the model

The measured tradeoff curve (`README.md:46-58`), all on the same Arc Pro B70, same ~116K-token task:

| Configuration | Decode | Context ceiling |
|---|---|---|
| Flash / fp8 (vLLM stock) | 34.8–39.0 tok/s | ~196K |
| KVarN | 12.92 tok/s | ~550K |
| KVarN + MTP depth 1 | 21.62 tok/s | ~404K |
| KVarN + MTP depth 2 | 23.70 tok/s | ~400K |

The served model is `dbirks/Qwen3.8-27B-W4A16-AutoRound`.

### 1.2 The model is dense — the sibling project verified this itself, against its own earlier assumption

This is not my inference; the sibling project caught its own mis-framing and wrote it up. From
`docs/experiments/gemm-xpu-w4a16-decode-plan.md:74-100`, under the heading "Correction first: the model is
dense, not MoE":

> The served checkpoint's config says otherwise, and the trace agrees: […] `hidden_size=5120`,
> `intermediate_size=17408`, `num_hidden_layers=64`, **no** `num_experts` / `moe_*` / router keys of any kind.
> […] The trace shows exactly one FFN up GEMM and one FFN down GEMM per layer per token, identical across all
> 64 layers — dense MLP, no expert routing, no grouped-GEMM kernels, no router GEMM.

and its consequence, stated in that project's own words (`:95-100`):

> Consequences: the planned "MoE-expert vs dense GEMM" split is moot, and the whole family of MoE-shaped
> concerns (small-M per-expert GEMMs, `XPUExpertsWNA16`, the Cutlass Xe2 grouped GEMM […]) is real, exists
> upstream, and is simply **not in this model's execution path**.

### 1.3 The repo-wide MoE / expert / offload grep: essentially a clean negative

A case-insensitive grep for `moe|expert|cpu_offload|cpu-offload|n-cpu|--cpu-moe` across every `.md`, `.py`,
`.yml`, `Dockerfile*` and `.env*` in the sibling repo returns, after discarding vendored vLLM overlay files
that merely *inherit* upstream MoE plumbing:

- `docs/experiments/gemm-xpu-w4a16-decode-plan.md` — the correction quoted above.
- `docs/experiments/cpu-offload-long-context-plan.md` — **CPU-offloaded long-context *attention* (KV), not
  weights.** Verdict in that project's own index (`docs/experiments/README.md:44`): **rejected** as the
  primary approach, "Research-only: no code changed, GPU untouched."
- `vllm-xpu-patches/overlay/v1/worker/gpu_model_runner.py:65-67, 82-87, 602-606` — imports of upstream
  vLLM's `RoutedExpertsCapturer` / `MixtureOfExperts` / EPLB state. These are inherited from the base file
  the overlay replaces wholesale; the project wrote none of it and the dense model never reaches it.

Grep for `hyper.?connection|lightning.?index|qwen4exp|flash.?next` across the entire sibling repo: **zero
hits.** The project never touched any feature this project's model needs.

**Verdict: there is no MoE-expert-offload or CPU/GPU hybrid compute mechanism in the sibling project, because
its model never needed one.** The four-way objection recorded in this project's own
`docs/00-background.md:159-166` ("the served model is dense […] it already fits fully in VRAM") holds up
exactly as written.

### 1.4 The one algorithmic near-miss worth naming — Sinkhorn — and why it is a near-miss, not a hit

The task framing asked whether "Sinkhorn-based routing/balancing" in
`vllm-xpu-patches/overlay/v1/attention/ops/triton_kvarn_sinkhorn.py` might transfer to MoE routing. It does
not, but the coincidence is real enough to be worth writing down so nobody re-investigates it.

KVarN's Sinkhorn is a **store-side numerical conditioner for quantization**, not a router. From its own
docstring (`triton_kvarn_sinkhorn.py:3-12`):

> Triton fused log-domain iterative variance-normalization for KVarN. […] same 16 alternating col/row
> std-normalization passes, same best-so-far tracking via the imbalance metric

And it is definitively off the decode hot path — the sibling project established this independently
(`docs/experiments/kvarn-decode-kernel-alternatives-B-C.md`, §2.3):

> **NOT Sinkhorn.** Confirmed independently this pass: `_sinkhorn_pack_kv` is called only from `_flush_tail`
> […] i.e. **store-side only, once per 128-token tile per layer** […] no decode kernel — Triton or SYCL —
> needs to touch it.

Separately, and by genuine coincidence, **Sinkhorn also appears in the hyper-connections machinery this
project's model actually uses** — `vllm-xpu-kernels`'s `csrc/xpu/mhc/mhc_interface.h:11-21` declares
`mhc_pre(..., double hc_sinkhorn_eps, int64_t sinkhorn_repeat)` as "RMS-norm projection → sigmoid/Sinkhorn
gating → weighted reduction". Same mathematical primitive (iterative row/column balancing), entirely
different role and entirely different tensor shapes. **Inference:** nothing in KVarN's Sinkhorn kernel is a
useful starting point for a hyper-connection kernel; the shared word is a false cognate. Flagging it so it
does not get re-discovered as a "lead".

### 1.5 What genuinely *is* reusable: B70 hardware-tuning knowledge

This is the real payload of the sibling project for this one, and it is substantial. All of it is *knowledge
about this exact card*, not code.

**Triton-on-XPU hazards** (`docs/01-kvarn-port/kvarn-xpu-port-plan.md:384-410`, "Category (d) — Triton
kernel-level issues"):

- `maxnreg` is **silently dropped** by Intel's Triton backend — `XPUBackend.parse_options` filters unknown
  keys, `XPUOptions` has no `maxnreg` field, so autotune configs differing only in `maxnreg` collapse into
  duplicates and waste compiles without erroring (`:398`).
- `num_warps` means **sub-groups, not warps** — every "num_warps=8 was empirically faster" comment inherited
  from CUDA source is a CUDA fact, not a portable one (`:399`).
- Register/GRF budgets on Xe are far smaller than a CUDA SM's, so any kernel holding a large tile live
  **will** spill to scratch (`:401`).

**Measured tuning outcomes on the B70** (`kvarn-xpu-port-plan.md:1131-1170`):

- The winning Triton config at head-dim 256 was `BLOCK_N=16, num_warps=2` for *every* shape;
  `num_warps ≥ 4` measured 2–6× worse. `BLOCK_N=128` was dead code (SLM overflow, 139520 > 131072, silently
  pruned).
- `grf_mode` is the Xe analogue of `maxnreg`, worth 3–9%, and **must be a launch kwarg** —
  `triton.Config(grf_mode=...)` raises `TypeError` on `triton 3.7.2+xpu`. Valid values are
  `"auto"/"default"/"128"/"256"`; `"large"`/`"small"` are rejected at runtime.
- A **fatal, unprunable** failure mode: under `grf auto`, spilly configs die at binary-load time with
  `total scratch space exceeds HW supported limit: 274304 > 262144`, which the autotuner cannot catch
  (it only handles `OutOfResources`) — the process dies.
- Occupancy saturates past ~256 concurrent programs on this card; more splitting cannot help beyond that.

**The SYCL-vs-Triton verdict, with measured bandwidth numbers**
(`docs/experiments/kvarn-decode-kernel-alternatives-B-C.md:42-107`). This is the most directly relevant
document in the sibling project to *this* project's possible future kernel work:

- The fast kernels on this box are **CUTLASS SYCL-TLA (cute)**, not hand-rolled SYCL: Xe 2D block-copy atoms
  with explicit software prefetch (`make_block_2d_copy_A/B`, `make_block_2d_prefetch`), occupancy controlled
  via kernel properties `sub_group_size<sg_size>, grf_size<256>`, built through a CMake FetchContent of
  `sycl-tla` with `icpx -fsycl -fsycl-targets=spir64_gen` and a build-time config-variant matrix.
- Measured on this card: paged-FA runs at **315–547 GB/s**, oneDNN GEMMs at **539–549 GB/s**, and **every
  Triton kernel measured here was slow** (KVarN fused decode at 10–13 GB/s). The document is scrupulous that
  this is "consistent with 'the layout was wrong'" but does not prove Triton *cannot* go fast on this stack.
- Its conclusion on writing new SYCL: a hand-written fused SYCL kernel was estimated at **4–8 engineer-weeks**
  vs. 2.5–4 for the Triton equivalent, and **rejected as a first move** — "the diagnosed problem is the
  *packing layout*, not the language." The stated honest counterweight is that the project has real Triton
  practice and **zero in-house CUTLASS-Xe/cute authoring experience**.
- Build-loop cost is real and documented: a full `vllm-xpu-kernels` config rebuild took **~8.5 hours** at
  `MAX_JOBS=6`, with host-OOM during builds as a recurring trap.

**Operational safety knowledge**, already carried into this project's `PLAN.md:245-262` ground rules but
worth re-stating as sourced: Intel's compute-runtime silently spills VRAM-exhausted allocations into host RAM
uncharged to any cgroup, which caused two host-wide OOM kills and a GPU hang on this exact box. The fix is
three env vars — `NEOReadDebugKeys=1`, `EnableSharedSystemUsmSupport=0`,
`EnableImplicitMigrationOnFaultableHardware=0` — set unconditionally in the sibling's compose file
(`KNOWN-ISSUES.md:209-233`).

**Verdict on §1: KVarN itself — no. The B70 tuning corpus — yes, and it is the main reason the sibling
project matters to this one.**

---

## 2. Does vLLM support Qwen3.8-Flash-Next? — the gating question

### 2.1 Correction first: the local vLLM clone is stale in a way that inverts the answer

`/media/da3dsoul/Golias/AIProjects/vllm` is a real clone of `https://github.com/vllm-project/vllm.git`, on
`main`, at commit `9d0fe9bac89d0bda98f977770f5ae88b386e981a`, `git describe` = **`v0.28.1rc0-122-g9d0fe9bac8`**,
dated **2026-08-30**.

A thorough, case-insensitive, repo-wide grep of that clone for `qwen4exp|qwen4_exp|Qwen4Exp|flash.?next`
returns **zero hits**, in `vllm/model_executor/models/`, in `registry.py`, and in
`vllm/transformers_utils/configs/` alike. That result is correct *for that commit* and **wrong about
reality**, because upstream merged the model on **2026-08-31** — the day after. Verified live:

```
$ gh pr view 53896 --repo vllm-project/vllm --json number,title,state,mergedAt,additions,changedFiles
{"number":53896,"title":"[Model] Support Qwen3.8-Flash-Next","state":"MERGED",
 "mergedAt":"2026-08-31T05:57:56Z","additions":19425,"deletions":451,"changedFiles":124}
```

**Methodological note worth keeping:** for a model released 2026-08-24 into a repo that merges hundreds of PRs
a week, a two-week-old clone is not a usable source of truth for "does vLLM support X". Check upstream via
`gh api` for any support question about a model newer than the clone.

### 2.2 What upstream actually has, verified against `main` this pass

The implementation is **not** in `vllm/model_executor/models/` — it lives in a newer per-vendor package tree,
which is why a search confined to the conventional models directory would miss it even on a current clone.
Fetched live from `main`:

```
$ gh api repos/vllm-project/vllm/contents/vllm/models/qwen4_exp --jq '.[] | "\(.type)\t\(.size)\t\(.name)"'
file    1724    __init__.py
dir     0       amd
dir     0       common
file    9552    config.py
dir     0       nvidia
```

`nvidia/` contains `model.py` (41,465 B), `ngram_embedding.py` (35,120 B), `mtp.py` (18,014 B),
`indexer_qsa.py` (17,002 B), `qsa.py` (16,741 B), `ple_layer.py` (15,994 B), `low_latency_gemm.py`,
`hyperconnection.py`, `model_state.py`, and an `ops/` subdirectory. `common/` contains `qsa_cache.py`
(34,172 B), `hyperconnection.py`, `ple.py`.

Registry entries on `main` (`vllm/model_executor/models/registry.py`, fetched live):

| line | architecture string | target |
|---|---|---|
| 114-116 | `Qwen4ExpForCausalLM` | `vllm.models.qwen4_exp` |
| 602-604 | `Qwen4ExpForConditionalGeneration` | `vllm.models.qwen4_exp` |
| 697 | `Qwen4ExpMTP` | `vllm.models.qwen4_exp` |

These match exactly the architecture strings this project's own converter registers —
`src/llama.cpp/conversion/qwen4exp.py:16`:
`@ModelBase.register("Qwen4ExpForConditionalGeneration", "Qwen4ExpForCausalLM")`, with
`@ModelBase.example("Qwen/Qwen3.8-Flash-Next")` on the next line. Same model, same names.

**Every architectural feature is implemented, by name.** Cross-checking this project's own GGUF metadata
against upstream's package layout:

| feature | this project's evidence | upstream vLLM |
|---|---|---|
| 48 layers, 512 experts, 10 routed | `logs/` load banner: `n_layer = 48`, `n_expert = 512`, `n_expert_used = 10` | `vllm/models/qwen4_exp/config.py`, `Qwen4ExpTextConfig(Qwen3NextConfig)` |
| hybrid full-attn + gated DeltaNet | `src/llama.cpp/src/models/qwen4exp.cpp:33-42` (SSM hparams) | shared `vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn`, same module Qwen3-Next uses |
| hyper-connections | `logs/`: `qwen4exp.hyper_connection.count = 4`, `.low_rank = 320`; tensors `hc_attn_*`/`hc_ffn_*`/`hc_head_*` at `qwen4exp.cpp:164-218` | `qwen4_exp/common/hyperconnection.py` + `nvidia/hyperconnection.py` + `nvidia/ops/hc.py` |
| lightning-indexer sparse attention ("QSA") | `logs/`: `attention.indexer.head_count = 4`, `.key_length = 128`, `.top_k = 2048`; `qwen4exp.cpp:56-61` | `qwen4_exp/common/qsa_cache.py`, `nvidia/qsa.py`, `nvidia/indexer_qsa.py` |
| PLE n-gram hash embeddings | `conversion/qwen4exp.py:37-51, 68-70` | `qwen4_exp/common/ple.py`, `nvidia/ple_layer.py`, `nvidia/ngram_embedding.py` |

**Terminology note that matters for future searching:** Qwen and vLLM call the sparse attention **QSA (Qwen
Sparse Attention)**, not "lightning indexer" — that name belongs to DeepSeek's DSA. Searching vLLM for
"lightning indexer" returns only DeepSeek-V4/V3.2 and MiniMax-M3 hits and produces a false negative on this
model.

For calibration on "how big is adding an architecture": Qwen3-Next landed via
[#24526](https://github.com/vllm-project/vllm/pull/24526) at +2,476/−61 across 29 files. `qwen4_exp` was
**~8× that** (+19,425/−451, 124 files). The delta is exactly QSA, hyper-connections, and the PLE table.

### 2.3 The gate: vLLM refuses to load this model on Intel XPU

This is the finding that decides the migration question. Fetched live from `main`,
`vllm/models/qwen4_exp/__init__.py`:

```python
22: def __getattr__(name: str) -> Any:
        if name in {"Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration", "Qwen4ExpMTP"}:
            from vllm.platforms import current_platform
30:         if current_platform.is_xpu() or current_platform.is_tpu():
31:             raise NotImplementedError("Qwen4Exp currently supports CUDA and ROCm only")
```

Two lines. The dispatch below them offers `amd/` for ROCm and `nvidia/` otherwise. There is no XPU branch on
`main`.

Note *where* the refusal sits: inside the package's `__getattr__`, so it fires **before**
`ModelRegistry.register_model` or a `vllm.general_plugins` entry point can be consulted. An out-of-tree XPU
implementation cannot be supplied through vLLM's documented extension path; it can only be injected by
patching the installed package at runtime. That point is made by a TPU contributor in
[issue #54595](https://github.com/vllm-project/vllm/issues/54595) (comment 2026-09-08), not by me.

### 2.4 The XPU port exists — as an unreviewed draft

Verified live:

```
$ gh pr view 55068 --repo vllm-project/vllm
number 55068 | "add xpu support for qwen3.8-next" | author xiaolong-intel
state OPEN | isDraft true | createdAt 2026-09-03 | updatedAt 2026-09-04
additions 5045 | deletions 3 | changedFiles 14 | reviews 0 | comments 1
```

Its 14 files are a complete `vllm/models/qwen4_exp/xpu/` tree mirroring `nvidia/` — `model.py`,
`ple_layer.py`, `qsa.py`, `ops/qsa.py`, `ops/hc.py`, `ops/ple.py`, `indexer_qsa.py`, `hyperconnection.py`,
`mtp.py`, `model_state.py`, `low_latency_gemm.py` — plus the `__init__.py` gate change.

**Status as of today: opened 8 days ago, untouched for 7, zero reviews, no description.** It is authored by
Intel (`xiaolong-intel`), which is the best available signal that it is real work rather than a drive-by, but
it is not merged and there is no public timeline.

### 2.5 Somebody has already run this model on an Arc Pro B70 — and published what broke

[Issue #54595](https://github.com/vllm-project/vllm/issues/54595), "Qwen4Exp is gated off on XPU, but most of
what the gate hides is not XPU-specific", opened 2026-08-31 by `TSUMUGI-XE`, open, 4 comments, last activity
2026-09-09. **This is a third-party report; I have not reproduced any of it.** Quoting the body:

> I removed those lines and got the model serving on Intel Arc Pro B70. What the gate hides is **14 things,
> one of which is a missing implementation**

Their taxonomy: 1 gate, **1 genuinely missing implementation** (#6, QSA block selection — a CUDA-only custom
op `torch.ops._C.cooperative_topk`/`persistent_topk`, for which they supply a slower pure-PyTorch fallback),
and 12 naming/declaration/config mismatches (`mxfp4_pack_quantized` absent from the compressed-tensors
activation allowlist; checkpoint spells `weight_scale_inv` where code reads `weight_scale`; Triton paths
gated on `tensor.is_cuda`; XPU sets `use_static_cuda_launcher` where Inductor reads
`use_static_triton_launcher`; and so on). Their summary line:

> The XPU kernels were already there (`XPUExpertsMxFp4`, `torch.ops._xpu_C.fp4_gemm`) and the linear/sparse
> attention paths are Triton, so those ported at zero cost.

Their measurement, as posted:

> **Measured**: 2× Arc Pro B70, 117.2 GiB MXFP4+FP8, experts offloaded — 12.4 tok/s single, 41.9 peak (TP=2),
> GPU busy 4–5% (host-bound).

They also note a **hard blocker on unpatched `main`**: without the unmerged
[PR #54129](https://github.com/vllm-project/vllm/pull/54129) ("Support disk-backed (mmap) PLE table for
Qwen3.8-Flash-Next", open, not merged, +10,077/16 files), the model "fails earlier, in
`vocab_parallel_embedding.create_weights`, trying to allocate 95.37 GiB". The ~51B-parameter PLE table this
project's `docs/00-background.md:117-126` describes — and which llama.cpp handles for free via mmap — is
**not yet handled by vLLM `main` at all**.

A second, independent XPU bring-up by the same reporter via GGUF
(`vllm-gguf-plugin` 0.0.5) reports retuning two CUDA-tuned Triton constants —
`TRITON_BLOCK_N = 128`, `TRITON_NUM_WARPS = 2` — took single-stream decode from **1.29 to 7.47 tok/s**. That
is the same lesson the sibling KVarN project learned independently (§1.5: CUDA-derived `num_warps` values are
not portable to Xe).

One more B70-specific hazard, filed against the kernel package:
[vllm-xpu-kernels#559](https://github.com/vllm-project/vllm-xpu-kernels/issues/559) (opened 2026-08-31, open,
**zero replies**) — `moe_gather` device page fault on Arc Pro B70 (BMG) at the `TOPK=10` instantiation.
**TOPK=10 is exactly this model's routing width.** Unaddressed for 11 days.

### 2.6 How mature is vLLM's Intel-XPU MoE support in general?

Genuinely mature for what it covers, and I checked the kernel source directly rather than trusting the vLLM
side. In `/media/da3dsoul/Golias/AIProjects/vllm-xpu-kernels` (a clean clone of
`vllm-project/vllm-xpu-kernels`, HEAD 2026-08-28):

- `csrc/moe/` is ~5,600 lines of real SYCL: `topk.cpp` (997), `grouped_topk.cpp` (942),
  `moe_align_sum_kernels.cpp` (1,549), `topk_softplus_sqrt_kernels.cpp` (817), `remap_hidden_states.cpp`
  (567), `moe_gather.cpp` (204), `init_expert_map.cpp` (95), `reorder_mxfp_scales.cpp` (234).
- `csrc/xpu/grouped_gemm/xe_2/` holds a CUTLASS-Xe2 per-expert grouped GEMM with an explicit MoE collective
  (`collective/gemm/moe_gemm_array_cooperative.hpp`, `moe_tile_scheduler.hpp`, `moe_array_mma.hpp`).
- `csrc/xpu/mhc/` holds **registered SYCL hyper-connection kernels** — `mhc_pre`, `mhc_post`,
  `hc_head_fused`, `mhc_fused_post_pre` (`mhc_interface.h:11-53`, bound at
  `csrc/xpu/torch_bindings.cpp:144-168`). On the vLLM side these are reached through
  `vllm/model_executor/layers/mhc.py`, which has a `forward_xpu` for each of the four ops calling
  `torch.ops._xpu_C.mhc_pre` / `mhc_post` / `hc_head_fused` / `mhc_fused_post_pre`
  (`mhc.py:167-182, 258-265, 357-374, 547-566`).

Two important qualifications, because the `mhc` find is tempting to over-read:

1. **In the local clone, the MHC layer is consumed only by DeepSeek-V4** —
   `vllm/models/deepseek_v4/xpu/dspark.py:115-116, 172-175` and `.../xpu/model.py:1051, 1138`. No Qwen file
   calls it. Upstream, `qwen4_exp` carries *its own* `hyperconnection.py` in `common/`, `nvidia/` and `amd/`
   plus its own `ops/hc.py`, and does not route through `layers/mhc.py`. So these SYCL kernels are **adjacent
   prior art, not a drop-in** for a qwen4_exp XPU port.
2. Every one of `csrc/moe/`'s kernels is **activation-side** — routing (`topk`, `grouped_topk`), token
   permutation (`moe_align_block_size`, `remap_hidden_states`), output combination (`moe_gather`, `moe_sum`)
   — plus `init_expert_map(expert_map, num_experts, ep_rank, ep_size)`, which shards experts across
   *ranks*. **There is no weight-residency, hot-set, or eviction kernel anywhere in the package.**

### 2.7 vLLM's answer to "the model does not fit in VRAM" — and why it is the opposite of this project's thesis

This is the part that decides the question even if PR #55068 merged tomorrow. The hard constraint is
unchanged: `staging/models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/` is **~76.4 GB** and `UD-Q3_K_XL/` is
**~83.8 GB**, against a 32 GB card (≈30 GB usable per `PLAN.md:146`). The third-party MXFP4+FP8 run cited
above was **117.2 GiB**.

vLLM has exactly two weight-offload mechanisms, both in the local clone at
`/media/da3dsoul/Golias/AIProjects/vllm/vllm/config/offload.py` (`UVAOffloadConfig:16`,
`PrefetchOffloadConfig:48`, `OffloadConfig:80`, `offload_backend: Literal["auto","uva","prefetch"]` at `:83`).

**(a) `--cpu-offload-gb` → UVA backend. Zero-copy PCIe reads, forever. Not a cache.**
`vllm/model_executor/offloader/uva.py:110-116` does `p.data.to("cpu")` → `.pin_memory()` →
`p.data = get_accelerator_view_from_cpu_tensor(cpu_data)`. The weight then lives permanently in pinned host
RAM and the GPU reads it over PCIe *inside the kernel*, every forward pass. The config docstring says so
plainly (`vllm/config/offload.py:29-31`):

> part of the model is loaded from CPU memory to GPU memory on the fly in each model forward pass. This uses
> UVA (Unified Virtual Addressing) for zero-copy access.

**This genuinely works on XPU** — `vllm/utils/torch_utils.py:904-914` dispatches
`current_platform.is_xpu()` to `torch.ops._C.get_xpu_view_from_cpu_tensor`, implemented in
`vllm-xpu-kernels/csrc/xpu_view.cpp:92` and bound at `csrc/torch_bindings.cpp:222-227`.

It is also **targetable at MoE experts specifically**: `cpu_offload_params` does exact dotted-segment
matching, and the docstring's own worked example is `"experts"` matching `mlp.experts.w2_weight`
(`vllm/config/offload.py:35-45`).

**The architectural comparison, stated precisely.** UVA offload is structurally *the same mechanism* as the
CUDA fork's `device_alias` zero-copy path documented in
`docs/research/03-cuda-fork-moe-cache-outline.md` Topic 4 — pinned host memory, `cudaHostGetDevicePointer`'s
XPU equivalent, kernels reading a PCIe-mapped address directly. What it is missing is the *entire rest of
that design*: there is no VRAM hot-slot pool, no admission kernel, no eviction policy, no frequency
tracking. **vLLM's offload is the fork's cache with the cache removed — permanent 100% miss rate.**
(Inference, but a narrow one: the mechanism match is verified in source on both sides, and the absence of any
residency state is verified by grep — `expert.*offload|offload.*expert` returns zero hits across the whole
vLLM tree.)

There is measured third-party evidence for exactly how much that costs, on exactly this model.
[Issue #54593](https://github.com/vllm-project/vllm/issues/54593), "`--cpu-offload-gb` evicts in declaration
order, which for MoE offloads the hottest weights first" (opened 2026-08-31 by `TSUMUGI-XE`, open). Their
byte ledger for Qwen3.8-Flash-Next:

| weight class | on disk | read per token |
|---|---|---|
| dense (attention, router, norms, shared expert) | 4.80 GiB | **4.80 GiB — all of it** |
| experts | 62.11 GiB | 1.22 GiB (top-10 of 512) |
| n-gram (PLE) table | 47.75 GiB | ~0 (gather only) |

and their measurement, **one card, `--cpu-offload-gb 47`**:

```
default order                     4.26 tok/s     4.5  GiB/token over PCIe
--cpu-offload-params experts      9.46 tok/s     0.93 GiB/token
```

Two things follow. First, this **confirms rather than merely asserts** that UVA reads only the routed expert
rows (1.22 GiB/token for top-10-of-512, not 62 GiB) — so the mechanism is efficient per-access and simply
never caches. Second, and bluntly: **9.46 tok/s on a single card with expert offload, against this project's
current measured 26–29 tok/s on llama.cpp** (`PLAN.md:240-243`, IQ3_XXS + `-ncmoe 26` + MTP depth 2).

Comparison caveats, stated rather than buried: different quantization (MXFP4+FP8 ~117 GiB vs. IQ3_XXS
~76 GB), a patched vLLM, an unspecified host, and a third-party report I have not reproduced. It is not a
controlled A/B. It is, however, the only same-model single-card vLLM number that exists, and it is **~3×
below** where llama.cpp already is.

**(b) `prefetch` backend.** `vllm/model_executor/offloader/prefetch.py` does real async H2D into a
`StaticBufferPool`, but selects layers **positionally** (`:191`,
`if module_index % self.group_size >= self.group_size - self.num_in_group`), is not routing-aware (grep for
`topk|router|routed|expert` inside `prefetch.py`: **zero hits**), re-copies every step, and is hard-wired to
`torch.cuda.Stream` (`:156`, and `:268, 275, 279, 288, 313, 397, 527`) with no `current_platform` dispatch —
**it cannot run on XPU at all.**

**(c) EPLB / expert parallelism.** Real, but it rebalances experts **between GPUs**, never GPU↔CPU
(`vllm/distributed/eplb/rebalance_execute.py`), and is likewise `torch.cuda.Stream`-hardcoded.

**(d) A hot/cold MoE expert residency cache.** Does not exist. `expert.*offload|offload.*expert` across the
whole tree: zero hits.

**This updates — and mostly vindicates — this project's own prior verdict** at
`docs/00-background.md:164-166` ("vLLM has no hook point equivalent to ggml's pluggable buffer types; porting
would mean partially reimplementing vLLM's MoE weight-loading path"). The correction is that vLLM *does* now
have a coarse hook — `cpu_offload_params` can select expert tensors, and UVA gives you the zero-copy read —
but it is a **static, whole-tensor, no-residency** selection. Building this project's cache inside vLLM would
still mean writing the hot-slot pool, the admission kernel, and the eviction policy from scratch, against a
weight-loading path with no concept of a slot.

---

## 3. Honest effort estimate: "add qwen4exp to vLLM-XPU" vs. "keep pushing llama.cpp SYCL"

The framing in the task ("what would adding qwen4exp to vLLM take") turns out to be the wrong question — the
model is already added. The real question is **"what would getting it to run *well* on a single Arc Pro B70
take"**, and that decomposes into four stacked items, none optional:

| # | item | who owns it | state today |
|---|---|---|---|
| 1 | XPU model implementation | Intel, PR #55068 | draft, 0 reviews, 7 days stale |
| 2 | QSA block-selection op on non-CUDA | nobody upstream | slow pure-PyTorch fallback exists in #54595 |
| 3 | PLE table not held in VRAM | PR #54129 | open, unmerged; without it, **95.37 GiB alloc failure at load** |
| 4 | ~46–85 GB of weights on a 30 GB card | UVA offload | works, but 100% miss rate; 9.46 tok/s reported |
| 5 | `moe_gather` faults at TOPK=10 on B70 | vllm-xpu-kernels#559 | open, zero replies, 11 days |

Items 1 and 3 are **someone else's merge decisions**, on someone else's schedule, with no public timeline.
Item 4 is an architectural mismatch that merging PRs will not fix: even with a perfect XPU port, the single-
card configuration is the one vLLM is worst at, and the one llama.cpp's `-ncmoe` static split and the CUDA
fork's cache are built for. Item 2 is a real kernel to write, and §1.5's own verdict says a new Xe kernel is
4–8 engineer-weeks if written in SYCL, 2.5–4 in Triton.

Against that, the llama.cpp path's cost structure today (`PLAN.md:114-243`): the SYCL cache port is **done and
correct**, MTP speculative decoding **landed and works** at 92–95% acceptance, the production config is
IQ3_XXS + `-ncmoe 26` + `--spec-draft-n-max 2` at **26–29 tok/s**, and the largest untried lever is
identified and specific — the llama.cpp discussion #24528 RFC's CPU-driven hybrid `MUL_MAT_ID` design, whose
own reported numbers (4×3090, 13 models, +10%..+57%, 16/16 ≥ parity) are the strongest evidence anyone has
produced that the hybrid-placement failure mode is fixable rather than fundamental.

**Estimate:** migrating to vLLM is **not a bounded engineering task this project can schedule**, because two
of its five blockers are upstream merge decisions and a third (item 4) is unfixable within vLLM's design.
Continuing on llama.cpp is a bounded task with a named next lever. The asymmetry is not close.

### The one thing vLLM would genuinely buy: concurrency

This deserves to be said plainly, because it is the user's actual stated motivation and it is correct.
The sibling `/media/da3dsoul/Golias/AIProjects/qwen38-27b-b70-vllm/README.md:46-47` measured, on this exact
card, for a dense 27B that *fits in VRAM*:

> Measured warm decode | **~27.3 tok/s** single stream
> Measured concurrency | 1 → 4 simultaneous requests, near-flat wall time (~10.4–10.6 s), **~65.7 tok/s aggregate**

That is a real 2.4× aggregate win from continuous batching that llama.cpp will not match. But note the
single-stream number: **27.3 tok/s, for a model that fits entirely in VRAM** — essentially identical to this
project's current 26–29 tok/s for a model 4× larger that does not. vLLM's advantage here is **throughput
under load, not latency**. If the workload is one interactive stream, migrating buys nothing and costs the
five blockers above.

---

## 4. KVarN for context extension via YaRN, specifically

**This is a different problem from §1** and deserves its own verdict. §1 asks whether KVarN helps place
**MoE expert weights**; this section asks whether KVarN's technique helps serve **262K–1M tokens of context**
on a 32 GB card. They are unrelated memory pressures and the answers differ in their reasoning, though not in
their conclusion.

Framing caveat up front: **nothing in this project has ever been tested at long context.** Every benchmark in
`PLAN.md:114-243` used a ~10-token prompt (`Count from one to fifty.`) with `-n 200-300`, purely to measure
decode throughput. The stated target (`PLAN.md:102-103`, "high-40s to mid-50s tok/s decode") names no context
requirement at all. **Whether long-context serving is actually a goal for this deployment is an open
question for the user, not an assumption this document makes.** Everything below is conditional on it being
one.

### 4.1 What llama.cpp already offers — it goes to 4 bits, but not to 2

`src/llama.cpp/common/arg.cpp:323-333` enumerates the complete set of KV-cache types, and it is short:

```cpp
const std::vector<ggml_type> kv_cache_types = {
    GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16,
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_IQ4_NL,
    GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
};
```

Selected per side via `-ctk`/`-ctv` (`arg.cpp:2453-2469`), with independent draft-model equivalents
`-ctkd`/`-ctvd` (`arg.cpp:4096-4112`) — which matters, because this project's production config runs MTP.

So the floor is **4-bit**, in blocks of 32 with an fp16 scale: `q4_0` costs 18 bytes per 32 elements = **4.5
bits/element**. KVarN's `kvarn_k4v2_g128` is 4-bit K + **2-bit** V. Reading its own cache-layout constant —
`triton_kvarn_decode.py:19-21`, "`KVarNConfig.tile_bytes_aligned` bytes (26880 for kvarn_k4v2_g128 @ head_dim
256)" — a 128-token × 256-dim K+V tile at 4+2 bits would be 24,576 bytes of payload, so the 26,880 figure
implies ~2,304 bytes of scale/metadata overhead, i.e. **~3.3 bits/element** all-in.

**The gap llama.cpp has versus KVarN is therefore 4.5 → 3.3 bits/element, a 1.36× ratio — not the 4–8×
that "KV quantization" suggests in the abstract**, because llama.cpp is already doing most of it.

### 4.2 This model's KV cache is much smaller than the sibling's — the numbers, verified

Qwen3.8-Flash-Next is hybrid, and aggressively so. From a real model-load banner in this repo's `logs/`:
`n_layer = 48`, `n_head_kv = 2`, `n_embd_head_k = n_embd_head_v = 256`, `n_embd_k_gqa = n_embd_v_gqa = 512`,
`n_ctx_train = 262144`. Only every 4th layer has a KV cache — the loader log shows
`llama_kv_cache: layer 0: filtered / layer 1: filtered / layer 2: filtered / layer 3: dev = SYCL0`, i.e.
**12 of 48 layers**, matching `docs/research/02-model-support-status.md` §4.2 ("36 Gated DeltaNet + 12
full-attention (indices 3,7,…,47)").

Per-token cost: 12 layers × (512 K + 512 V) elements × 2 bytes = **24,576 B/token = 24 KiB/token at f16**.
This is verified, not derived: the same log shows `llama_kv_cache: SYCL0 KV buffer size = 96.00 MiB` at
`n_ctx_slot = 4096`, and 12 × 1024 × 4096 × 2 = 100,663,296 B = exactly 96.00 MiB.

Contrast the sibling's dense 27B, whose own analysis
(`docs/experiments/cpu-offload-long-context-plan.md`, §1) puts it at **64 KiB/token at bf16** — 16
full-attention layers × 4 KV heads × 256. **Flash-Next's KV cache is 2.67× lighter per token.** That single
fact is most of why KVarN mattered there and matters much less here.

Projecting to the context lengths in question (main KV cache only):

| `-ctk`/`-ctv` | bits/elem | B/token | @262,144 (native) | @1,048,576 (YaRN ×4) |
|---|---|---|---|---|
| `f16` (default) | 16 | 24.0 KiB | 6.00 GiB | 24.0 GiB |
| `q8_0` | 8.5 | 12.75 KiB | 3.19 GiB | 12.75 GiB |
| **`q4_0`** | **4.5** | **6.75 KiB** | **1.69 GiB** | **6.75 GiB** |
| *KVarN k4v2, hypothetically ported* | *~3.3* | *~4.9 KiB* | *~1.23 GiB* | *~4.9 GiB* |

**The whole KVarN prize at 1M tokens is ~1.8 GiB on a 30 GiB card** — against a model whose weights are
76–84 GB and whose expert placement is the actual binding constraint. (Arithmetic on measured constants; the
KVarN row is a projection onto this model's shape, not a measurement of anything.)

Two things this table omits, both flagged rather than hidden: (a) the **QSA indexer cache** is a second,
separately-allocated tier — `src/llama-kv-cache-msa.cpp:39-49` builds it with the same `type_k`/`type_v` as
the main cache, one key head of `indexer_head_size` (128) per sparse layer — so it scales with the same
flags but adds a smaller increment I have not computed precisely; (b) the 36 Gated-DeltaNet layers hold a
recurrent state that is **constant in context length**, not growing, and so does not participate in this
table at all.

### 4.3 The SYCL catch: quantizing the KV cache does not fully buy back VRAM

This is the finding most likely to be missed. On the SYCL backend, a non-`F16` KV cache is **widened back to
F16 in a scratch buffer before every flash-attention call**.

`ggml/src/ggml-sycl/fattn.cpp:431-453`:

```cpp
const bool tile_needs_K = K->type != GGML_TYPE_F16;
const bool tile_needs_V = V->type != GGML_TYPE_F16;
...
if (tile_needs_K) { need_K = std::max(need_K, (size_t) ggml_nelements(K)); }
if (tile_needs_V) { need_V = std::max(need_V, (size_t) ggml_nelements(V)); }
```

Those counts are reserved as `sycl::half`s (`ggml_sycl_fattn_reserve_halves`, `fattn.cpp:398-406`) and
appended to the FA node's own allocation so the graph allocator sees them
(`ggml/src/ggml-sycl/ggml-sycl.cpp:1021-1025`, comment: *"Reserve the additional scratch so it's visible to
the graph allocator"*). The kernel-selection comments say the same thing in prose — oneDNN is "native F16 and
**dequant**+non-F16" (`fattn.cpp:132`), and the MKL path "converts non-F16 K/V to F16 via `to_fp16_sycl`
before GEMM, so quantized, F16, BF16, and F32 caches all benefit from XMX acceleration"
(`fattn.cpp:140-142`).

`ggml_nelements(K)` for a decode call is the **whole KV cache view**, not a tile. At 1M tokens that is
12 × 1024 × 1,048,576 elements across the graph's 12 FA nodes, reserved at 2 bytes each.

**Inference, and the one thing here I would measure before trusting:** ggml's graph allocator reuses buffers
across nodes with disjoint lifetimes, and the 12 FA nodes are sequential, so in practice this should collapse
to roughly *one* such scratch region (~2 GiB at 1M tokens for K+V), not twelve. If that reuse holds, `q4_0`
at 1M still nets out ahead — ~6.75 GiB persistent + ~2 GiB transient ≈ 8.75 GiB peak, versus 24 GiB for
`f16`. If it does not hold, quantized KV on SYCL is actively counterproductive at long context. **This is a
one-command check** (`-c 262144 -ctk q4_0 -ctv q4_0` and read the buffer-size lines) and it should be run
before any long-context work is planned.

There is also a per-call **dequantization cost** on the decode hot path that f16 does not pay. Unmeasured
here; also cheap to measure.

### 4.4 What "way faster than other attempts" actually meant — read carefully, it does not transfer

The motivating quote deserves a precise answer, because the comparison it refers to is not the one it sounds
like.

**KVarN was measured as the *slowest* of the sibling project's four configurations, not the fastest.** From
`README.md:46-51`: stock vLLM Flash/fp8 **34.8–39.0 tok/s**, KVarN **12.92**, KVarN+MTP depth-1 **21.62**,
depth-2 **23.70**. The benchmark methodology
(`docs/05-benchmarks/kvarn-three-way-deployment-comparison.md:17-40`) makes the baseline explicit: the
"Flash" row is `--kv-cache-dtype fp8 --attention-backend FLASH_ATTN` — i.e. **8-bit KV, vLLM's own stock
path**. KVarN bought **context** (196K → 550K), and **paid decode speed for it**. That is the trade the
sibling README states in its own words (`README.md:60-63`): "None of these four are strictly 'the best' —
they're a real speed-vs-context tradeoff curve."

So "way faster than other attempts" is best read as referring to **KVarN's own optimization trajectory**, and
on that reading it is correct and impressive. `README.md:29-33`: "KVarN's decode path was measured at as
little as **6.3 tok/s** before this project's fixes […] landing at **up to 23.7 tok/s**". That is a 3.8×
improvement over *the same technique as originally ported*, achieved by finding one large routing bug, real
kernel tuning wins, and a separate speculative-decoding bug. It is **not** a claim that KVarN beats other
long-context techniques on speed.

The one genuinely *different* long-context technique the sibling evaluated was CPU-offloaded attention
(`docs/experiments/cpu-offload-long-context-plan.md`), and its verdict is recorded as **rejected — but on
reach, not speed** (`docs/experiments/README.md:44`): "plausible on paper (reaching the model's full native
262K context at roughly half decode speed, no quantization loss on the offloaded KV) but KVarN's approach
reaches much further (~550K) for a lower engineering cost." Its own projection was ~14–17 tok/s at 262K —
*comparable to or better than* KVarN's 12.92 at 550K.

**Conclusion on point 3: the comparison is entirely within vLLM, against vLLM's own fp8 path, and it went
against KVarN on speed. It implies nothing about llama.cpp's long-context path being slow.**

### 4.5 The actual long-context blocker in llama.cpp, and it is not the KV cache

While checking the above I found something that matters more than any of it. In this repo's own checkout,
`src/llama.cpp/src/models/qwen4exp.cpp:943-946`:

```cpp
// TODO: enable sparse attention when we are ready
//ggml_tensor * cur = build_attn_mha(q, k, v, nullptr, kq_mask_top_k, nullptr, nullptr, top_k->ne[0], kq_scale, il);
ggml_tensor * cur = build_attn_mha(q, k, v, nullptr, kq_mask_top_k, nullptr, nullptr, 0, kq_scale, il);
```

The whole QSA pipeline is built and correct — the indexer scores (`:780-869`), `ggml_top_k` selects a width
of `indexer_top_k + compress_ratio - 1` = **2,051** (`:861-863`), and the selection is folded into the KQ
mask via `ggml_set_rows` (`:930`). But the sparse-width argument is passed as **`0` instead of
`top_k->ne[0]`**, and the function's own header comment says so plainly (`:872`): *"Dense GQA self-attention
restricted to the cells that top_k names."*

**So attention is semantically sparse and computationally dense.** QSA's entire purpose is that attention
cost stays flat at ~2,048 keys regardless of context depth; as shipped here, it is O(n_kv) per token *plus*
the cost of a 2,051-wide `ggml_top_k` over all n_kv, on every one of the 12 full-attention layers, every
token.

One piece of good news found in the same pass, which corrects an earlier concern in this project's own
planning: **the `GGML_OP_TOP_K` `k ≤ 32` cap that `PLAN.md:29-31` and `PLAN.md:310-342` treated as a
backend-choice-forcing blocker is gone from the current source.**
`ggml/src/ggml-sycl/ggml-sycl.cpp:7302-7310` now accepts any `k > 0 && k <= src0->ne[0]`, consistent with
`PLAN.md:117` listing "TOP_K CPU-fallback fixed" among this project's landed Phase-3 work. The `k≈2051`
select runs on SYCL.

### 4.6 Verdict on the YaRN/long-context question

**Porting KVarN to llama.cpp-SYCL is not the right move, and it is not close.** Ordered by how much each
point matters:

1. **llama.cpp already does 4-bit KV.** The entire delta is 4.5 → ~3.3 bits/element.
2. **This model's KV cache is small.** 24 KiB/token, 12 of 48 layers, 2 KV heads. `q4_0` puts a **full 1M-token
   YaRN context at ~6.75 GiB.** The KVarN prize is ~1.8 GiB of that.
3. **The binding constraint at long context is still the weights.** 76–84 GB of model against ~30 GB of
   VRAM. Shaving 1.8 GiB off the KV cache buys roughly one more `-ncmoe` layer's worth of headroom.
4. **Attention is dense (§4.5).** Until that `0` becomes `top_k->ne[0]`, long-context decode cost grows
   linearly with depth no matter how the KV cache is stored. This is a **one-line change plus validation**,
   against a **4–8 engineer-week** SYCL kernel port (§1.5's own estimate for new Xe kernel work) — and it
   attacks the dominant term rather than a secondary one.
5. **The port cost is real.** KVarN is two Triton kernels plus a large surrounding backend
   (`triton_kvarn_decode.py` is 1,458 lines, `kvarn_attn.py` larger still, and the sibling project's port
   consumed months and produced a silent-corruption bug that took a four-gate byte-level investigation to
   root-cause — `KNOWN-ISSUES.md:1-75`). llama.cpp has no equivalent of vLLM's pluggable attention-backend
   registry; a custom KV layout would mean a new `ggml_type` plus SYCL FA support for it.

**Does this change the vLLM verdict?** Marginally, and in vLLM's favour — if and only if 400K+ context is a
real requirement. The sibling project has KVarN **working and validated on this exact card**, reaching ~550K
tokens, which llama.cpp cannot reach at any setting. But that is for a *dense 27B that fits in VRAM*; there
is no evidence anyone has combined KVarN with a 76 GB MoE model, and §2's five blockers all still apply. It
is a point on the scale, not a thumb on it.

**Open question for the user, which this document deliberately does not answer:** is long context actually a
goal? If the answer is "yes, 262K+", then the priority order is (a) measure the §4.3 scratch-buffer question,
(b) enable real sparse attention per §4.5, (c) run `-ctk q4_0 -ctv q4_0` at depth and see where it lands —
and *only then* ask whether the last 1.8 GiB is worth a kernel port. If the answer is "no, this is an
interactive short-prompt deployment", then §4 is moot and `PLAN.md`'s decode-throughput focus is correct as
written.

---

## 5. Transferable value from the sibling projects, regardless of the vLLM answer

Independent of §2's verdict, three things are worth extracting now:

1. **The B70 Triton/SYCL tuning corpus** (§1.5). Specifically: `BLOCK_N=16, num_warps=2` as the Xe2 starting
   point at large head dims; `grf_mode` as a launch kwarg; the unprunable scratch-overflow crash;
   `maxnreg` silently dropped; occupancy saturating past ~256 programs; and the measured bandwidth ceilings
   (315–547 GB/s FA, 539–549 GB/s oneDNN, 10–13 GB/s for an untuned Triton gather). If this project ever
   writes a new Xe kernel, this is the prior art, and the third-party 1.29 → 7.47 tok/s retune in #54595 is
   independent confirmation that the same class of constant is mis-set upstream too.
2. **"vLLM on this exact B70" is a well-trodden but scarred path.**
   `/media/da3dsoul/Golias/AIProjects/qwen38-27b-b70-vllm` shows what "smooth" looks like: a stock
   `vllm/vllm-openai-xpu:v0.28.0` image with **thirteen files replaced** and four re-derived upstream patches
   (`README.md:284-300`), an overlay pinned to one vLLM version whose upgrade procedure is a manual
   file-by-file diff (`README.md:723-762`), and an explicit maintenance warning that `pip install -U vllm`
   silently reverts it. That is the *validated, supported* model. This project's model would start harder.
3. **A directly relevant, already-paid-for lesson about vLLM's offloader.** From that same README
   (`:533-540`), about the vision tower:
   > know that `--cpu-offload-gb` will **not** help. vLLM's offloader is installed only in `make_layers()`
   > […] Setting the flag spends the budget on language layers instead, putting PCIe in the decode loop.

   Same mechanism, same failure shape, found independently a fortnight before issue #54593 said the same
   thing about MoE expert ordering.

Also worth recording: `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound` exists on HuggingFace (published
2026-08-28). Intel quantizing this model, plus Intel authoring PR #55068, is the strongest available signal
that XPU support is intended rather than incidental.

---

## 6. Recommendation

**Stay on llama.cpp. Do not migrate to vLLM now. Do not port KVarN.**

Not because vLLM lacks the architecture — it has it, fully, since 2026-08-31 — but because:

- vLLM **refuses to load it on XPU today** (`qwen4_exp/__init__.py:30-31`), and the fix is an unreviewed
  draft PR owned by someone else.
- vLLM's answer to "the model doesn't fit in 32 GB" is **permanent zero-copy PCIe reads with no residency
  cache**, which is the exact opposite of this project's thesis, and the only same-model single-card number
  in existence (9.46 tok/s, third-party) is ~3× below where llama.cpp already is.
- The PLE table — 47.75 GiB that llama.cpp mmaps for free — currently **fails at load** on vLLM `main`,
  pending unmerged PR #54129.
- vLLM's real advantage is **aggregate throughput under concurrency**, which the measured single-stream
  numbers on this card (27.3 tok/s for a model that fits) show is not a single-stream latency win.

And do not port KVarN to llama.cpp either, because llama.cpp's existing `-ctk q4_0 -ctv q4_0` already
captures most of the benefit for *this* model's unusually light KV cache, and the dominant long-context cost
is dense attention (§4.5), not KV bytes.

### What would change this recommendation

In rough order of how much each would move it:

1. **PR #55068 merges** and a released `vllm-xpu` image loads `Qwen4ExpForConditionalGeneration` without
   patching. This is the gate; nothing below matters until it opens. *Watch:
   `gh pr view 55068 --repo vllm-project/vllm`.*
2. **PR #54129 (disk-backed PLE) merges**, removing the 95.37 GiB load-time allocation. Without this a single
   box cannot load the model at all under vLLM.
3. **Someone lands a real MoE expert-residency cache in vLLM** — i.e. `cpu_offload_params` gains a VRAM
   hot-slot pool with admission/eviction rather than pure UVA. Today this does not exist anywhere in the
   tree. If it appeared, vLLM would become the better home for this project's *entire* idea, and the right
   move would be to port the design there rather than maintain a llama.cpp fork. *Watch: issue #54593 and
   whatever it turns into.*
4. **The workload changes from one interactive stream to genuine concurrency.** vLLM's 2.4× aggregate
   advantage at 4 concurrent requests is real and llama.cpp will not match it. If this deployment ever needs
   to serve several users at once, re-run this analysis with that as the primary metric — the conclusion
   could legitimately flip even with items 1–3 only partly resolved.
5. **Long context becomes a stated requirement beyond what §4.6 step (c) reaches.** If `-ctk q4_0
   -ctv q4_0` plus real sparse attention still falls short of the needed depth, vLLM gains a genuine
   advantage — the sibling project has KVarN validated to ~550K on this exact card, and llama.cpp has no
   path to that. This is the only line of argument in this document that meaningfully favours vLLM on
   technical merit rather than on concurrency.
6. **vllm-xpu-kernels#559 (`moe_gather` page fault at TOPK=10 on B70) gets fixed.** Minor relative to the
   above, but it is unaddressed, it is this exact card, and it is this exact routing width.

Meanwhile, two llama.cpp levers are actually available, both on code this project controls:

- **For decode throughput (the stated target):** the one `PLAN.md:158-164` already names — the
  discussion-#24528 CPU-driven hybrid `MUL_MAT_ID` design, which keeps `MUL_MAT_ID` on the CPU and dispatches
  a GPU batched matvec over cache-*hit* rows while other threads compute miss rows concurrently, rather than
  paying a synchronous PCIe copy per miss.
- **For long context, if it turns out to be a goal:** flip `qwen4exp.cpp:946`'s sparse-width argument from
  `0` to `top_k->ne[0]` and validate. One line, attacking the dominant term, against the 4–8 engineer-weeks a
  KVarN-style kernel port would cost for a ~1.8 GiB memory saving.
