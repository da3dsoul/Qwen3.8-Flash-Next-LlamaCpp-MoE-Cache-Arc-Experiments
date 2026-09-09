# 02 — Qwen3.8-Flash-Next on llama.cpp: support status & technical requirements

**Research date:** 2026-09-09
**Target hardware:** Intel Arc Pro B70, 32 GB VRAM (Battlemage / Xe2-HPG), llama.cpp SYCL backend
**Question driving this:** can we run `unsloth/Qwen3.8-Flash-Next-GGUF` on SYCL, and does a MoE expert cache (port of `GenerelSchwerz/llama.cpp` `moe-cache`) make sense here?

**Method note.** Everything below marked *Confirmed* was read directly from raw source files on `raw.githubusercontent.com`, from the GitHub tree API, or parsed out of the actual GGUF binary headers on Hugging Face (via HTTP range requests over the real published files). Nothing in the *Confirmed* rows is from a summary, a blog post, or secondhand paraphrase. Rows marked *Inferred* are my arithmetic or reasoning on top of those primary facts, and say so.

---

## Executive summary

Six findings dominate the rest of this document:

1. **Architecture support is fully upstream, not fork-only.** `ggml-org/llama.cpp` master has `LLM_ARCH_QWEN4EXP` (`"qwen4exp"`), a dedicated `src/models/qwen4exp.cpp`, and `conversion/qwen4exp.py` that explicitly names `Qwen/Qwen3.8-Flash-Next`. A SYCL port does **not** need to be based on a fork to get the architecture. Better still, the `GenerelSchwerz` `moe-cache` branch is ~2 diff lines from upstream master, so the expert cache and the architecture already coexist on one branch.
2. **qwen4exp introduced no new ggml ops at all**, and every op it does need already has a SYCL kernel. There is no missing-kernel wall. The PLE n-gram hash is not a GPU op — it is host C++ that emits row indices for a plain `get_rows`.
3. **There is exactly one concrete SYCL gap, and it is `GGML_OP_TOP_K`.** SYCL hard-caps `k <= 32`. Qwen Sparse Attention needs `k = 2051`. Vulkan handles this via radix-select; SYCL cannot. This forces a CPU fallback on 12 of 48 layers, every ubatch.
4. **It already runs on Intel Arc — on both backends, including Battlemage.** SYCL: confirmed on 3× Arc Pro B60 at our exact quant (UD-IQ3_XXS), 17.9 t/s tg512. Vulkan: confirmed on Arc Pro B65 32 GB (essentially our card) at UD-Q3_K_XL, 30.4 t/s @ 8 k with `--n-cpu-moe 25`. **But both required fixes that landed 2026-09-09 — pin the build to that date or later.**
5. **Nobody upstream validates non-CUDA for this architecture.** The support PR claims testing on CPU, CUDA and Metal only. SYCL/Vulkan support is inherited from generic op coverage, not verified. Expect to find our own bugs.
6. **The 32 GB card story is much better than the "82 GB file" suggests.** The file is dominated by a 28.8 GB PLE lookup table that llama.cpp *automatically* assigns to host memory and never uploads. The actual GPU-resident dense core is only **3.9–5.5 GB**. That leaves ~22 GB of VRAM as an expert slot pool against a 40–77 GB expert working set — precisely the regime an MoE expert cache is built for, and the B65 datapoint above shows a crude static version of this (`--n-cpu-moe`) already working.

**Recommended starting posture:** build from master ≥ 2026-09-09; bring up on **Vulkan first** (better op coverage for QSA, and the closest published datapoint is a 32 GB Battlemage card on Vulkan); benchmark SYCL against it; decide the expert-cache port target from that measurement rather than in advance.

---

## 1. Where architecture support lives (upstream vs fork)

### 1.1 Upstream `ggml-org/llama.cpp` master — full support. *Confirmed.*

The architecture is enumerated as `qwen4exp`:

> `src/llama-arch.cpp:43` — `{ LLM_ARCH_QWEN4EXP, "qwen4exp" },`
> https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-arch.cpp

Neighbouring Qwen entries at lines 32–43 include `qwen3next`, `qwen35`, `qwen35moe` — the predecessor architectures that share the Gated DeltaNet path.

The PLE metadata keys named in the project brief are all present, at `src/llama-arch.cpp:302–310`:

```
{ LLM_KV_PLE_LAYERS,            "%s.ple.layers"            },
{ LLM_KV_PLE_NGRAM_SIZE,        "%s.ple.ngram_size"        },
{ LLM_KV_PLE_HEADS_PER_NGRAM,   "%s.ple.heads_per_ngram"   },
{ LLM_KV_PLE_CONV_KERNEL,       "%s.ple.conv_kernel"       },
{ LLM_KV_PLE_LAYER_MULTIPLIERS, "%s.ple.layer_multipliers" },
{ LLM_KV_PLE_HEAD_OFFSETS,      "%s.ple.head_offsets"      },
{ LLM_KV_PLE_HEAD_VOCAB_SIZES,  "%s.ple.head_vocab_sizes"  },
{ LLM_KV_PLE_EOS_TOKEN_ID,      "%s.ple.eos_token_id"      },
{ LLM_KV_PLE_IMAGE_TOKEN_ID,    "%s.ple.image_token_id"    },
```

and the PLE tensors at `src/llama-arch.cpp:542–547`: `blk.%d.ple_key`, `blk.%d.ple_value`, `blk.%d.ple_norm_key`, `blk.%d.ple_norm_query`, `blk.%d.ple_norm_conv`, `blk.%d.ple_conv1d`.

**Note on repo layout:** upstream has been refactored since the versions most write-ups describe. Per-architecture graph code now lives in `src/models/<arch>.cpp`, and `convert_hf_to_gguf.py` is now a thin CLI wrapper over a `conversion/` package. So:

- Graph implementation: **`src/models/qwen4exp.cpp`** (1283 lines)
  https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/models/qwen4exp.cpp
- Conversion: **`conversion/qwen4exp.py`** (not `convert_hf_to_gguf.py`)
  https://raw.githubusercontent.com/ggml-org/llama.cpp/master/conversion/qwen4exp.py
- Shared linear-attention helper: **`src/models/delta-net-base.cpp`** (606 lines)

`conversion/qwen4exp.py` names this exact model in its registration decorators — this is the single clearest confirmation that upstream targets Qwen3.8-Flash-Next specifically, not merely a related architecture:

```python
@ModelBase.register("Qwen4ExpForConditionalGeneration", "Qwen4ExpForCausalLM")
@ModelBase.example("Qwen/Qwen3.8-Flash-Next")
class Qwen4ExpTextModel(_Qwen35MRopeMixin, _LinearAttentionVReorderBase):
    """Qwen3.8-Flash-Next.

    Shares the Qwen3.5 gated delta net and interleaved mrope, and adds three things:
    hyper-connections in place of every layer norm, QSA sparse attention on the full
    attention layers, and PLE n-gram hash embeddings on a single layer.
    """
    model_arch = gguf.MODEL_ARCH.QWEN4EXP
```

That docstring is also the most authoritative one-paragraph statement of what is architecturally new here.

### 1.2 `GenerelSchwerz/llama.cpp`, branch `moe-cache` — also has it, because it tracks upstream closely. *Confirmed.*

- Branch head: `078a36bcdedb96e5ec72b774bdb46228b3aa8e55`, committed **2026-09-09T14:57:14Z** (same day as this research), message `server : add experimental batched CUDA decode overlap (#79)`.
  https://api.github.com/repos/GenerelSchwerz/llama.cpp/branches/moe-cache
- `src/models/qwen4exp.cpp` → HTTP 200 on that branch.
- `ggml/src/ggml-sycl/gated_delta_net.cpp` → HTTP 200 on that branch.
- I diffed the fork's `src/llama-arch.cpp` against upstream master's: **2 diff lines total** (the fork is missing one `case LLM_ARCH_KIMI_K3:` label). Byte sizes 82131 vs 82162.

**This is a significant de-risking result.** The concern raised in the brief — "a SYCL port would need to be based on whichever fork actually has the architecture, not vanilla upstream" — does not bite. The `moe-cache` fork *is* essentially vanilla upstream plus the MoE cache. There is no fork-vs-upstream architecture split to reconcile: qwen4exp support and the moe-cache feature already coexist on one branch, and that branch is being rebased on upstream continuously.

### 1.3 Unsloth's fork

Unsloth ships an MTP (multi-token prediction) draft-head branch separate from the main GGUFs:

> `git clone --branch qwen4exp/mtp https://github.com/danielhanchen/llama.cpp`
> — https://unsloth.ai/docs/models/qwen3.8-next

*Confirmed from Unsloth docs (fetched page), not from the branch itself — I did not verify that branch's contents.* Note the branch name is `qwen4exp/mtp`, which corroborates that even Unsloth uses the upstream `qwen4exp` naming. The MTP head is deliberately **not** in the main conversion path — `conversion/qwen4exp.py` sets `supports_mtp_export = False` and `no_mtp = True`, with the comment *"the MTP block is a separate draft head; vLLM drops it too"*. MTP is therefore an optional speed add-on, not part of base support.

---

## 2. New ops required and their backend coverage

### 2.0 Headline: qwen4exp added *no* new ggml ops

Per PR [#27742](https://github.com/ggml-org/llama.cpp/pull/27742) (the upstream support PR): *"`git diff master --stat -- ggml/` is empty: no new ggml op"*. Everything qwen4exp needs was already in ggml before it landed. The ops that actually matter for this model are `GATED_DELTA_NET`, `TOP_K`, `SET_ROWS`, `MUL_MAT_ID`, `ROPE` (interleaved mrope), `FLASH_ATTN_EXT`, `GET_ROWS`, `FILL`, `SOLVE_TRI`, `TRI`, `CUMSUM`.

### 2.1 The three "scary" features are less scary than expected

**PLE / n-gram lookup: no GPU op exists, and none is needed.** *Confirmed.*

This is the most important correction to the framing in the brief. The hash is **computed on the host in C++**, in `llm_graph_input_ple::set_input()` (`src/models/qwen4exp.cpp:1052`). It walks the preceding tokens out of the KV cells, applies a rolling multiply-XOR hash, mods by each head's vocab size, adds that head's offset, and uploads the resulting **I32 row indices**:

```cpp
for (int64_t n = 2; n <= n_gram; ++n) {
    uint64_t mixed = (uint64_t) ctx[0] * hp.ple_layer_multipliers[0];
    for (int64_t j = 1; j < n; ++j) {
        mixed ^= (uint64_t) ctx[j] * hp.ple_layer_multipliers[j];
    }
    const int64_t base = (n - 2) * per_gram;
    for (int64_t g = 0; g < per_gram; ++g) {
        const int64_t h_i = base + g;
        idx[i * n_heads + h_i] =
            (int32_t) (mixed % hp.ple_head_vocab_sizes[h_i] + hp.ple_head_offsets[h_i]);
    }
}
ggml_backend_tensor_set(rows, idx.data(), 0, idx.size()*ggml_element_size(rows));
```

The only GPU-side operation is then a plain gather (`src/models/qwen4exp.cpp:1186`):

```cpp
ggml_tensor * emb = ggml_get_rows(ctx0, model.per_layer_tok_embd, rows);
```

**Consequence:** the PLE lookup is 100 % backend-agnostic. It costs a SYCL port nothing. It is also why the 28.8 GB table can live on the CPU / be mmap'd from SSD without a graph penalty — a `get_rows` against a host tensor is a memcpy of 16 rows × 160 values per token.

The PLE's dilated depthwise conv is likewise composed from standard ops, not a custom kernel. The source explicitly avoids `ggml_conv_1d_dw` (*"ggml_conv_1d_dw is documented as unreliable"*) and decomposes it into shifted views + `ggml_mul` + `ggml_add`. Note that `src/llama-arch.cpp:775` maps `LLM_TENSOR_PLE_CONV1D` to `GGML_OP_SSM_CONV`, but that mapping is only used for buffer-type assignment — the real graph never calls `ggml_ssm_conv` for PLE.

**Gated DeltaNet: a real fused op, `GGML_OP_GATED_DELTA_NET`, and SYCL has it.** *Confirmed.*

Declared in `ggml/include/ggml.h:2652`. Your Mamba/RWKV precedent intuition was right — it sits in the enum right next to `GGML_OP_RWKV_WKV6`, `GGML_OP_RWKV_WKV7` and `GGML_OP_GATED_LINEAR_ATTN`, and is implemented per-backend the same way those are.

`src/models/delta-net-base.cpp` offers three paths, selected at runtime (`delta-net-base.cpp:425`):

```cpp
if (n_seq_tokens == 1) {
    if (cparams.fused_gdn_ar) return build_delta_net_fused(...);
    return build_delta_net_autoregressive(...);
}
if (cparams.fused_gdn_ch)   return build_delta_net_fused(...);
return build_delta_net_chunking(...);
```

- `build_delta_net_fused` → one `ggml_gated_delta_net()` call.
- `build_delta_net_chunking` → the decomposed WY/UT-transform path, which is what pulls in `ggml_cumsum`, `ggml_tri` and `ggml_solve_tri`. Chunk size `CS = 64` (`delta-net-base.cpp:61`).

**Qwen Sparse Attention: no dedicated sparse-attention op is used today.** *Confirmed.* QSA is built as (a) an indexer that scores KV blocks, (b) a `ggml_top_k` over those scores, (c) construction of a masked KQ mask via `ggml_fill` + `ggml_set_rows`, then (d) **ordinary dense attention with that mask**. The genuinely sparse kernel is not wired up yet — `src/models/qwen4exp.cpp:747`:

```cpp
// TODO: enable sparse attention when we are ready
// ref: https://github.com/ggml-org/llama.cpp/pull/27970
//ggml_tensor * cur = build_attn_mha(q, k, v, nullptr, kq_mask_top_k, nullptr, nullptr, top_k->ne[0], kq_scale, il);
ggml_tensor * cur = build_attn_mha(q, k, v, nullptr, kq_mask_top_k, nullptr, nullptr, 0, kq_scale, il);
```

So today QSA costs full dense-attention compute and only saves via masking. That means the sparsity is currently a **quality** feature, not a speed feature, on every backend including CUDA.

### 2.2 Backend coverage table

Sources: SYCL `supports_op` at `ggml/src/ggml-sycl/ggml-sycl.cpp:6376–6660`; CUDA `supports_op` at `ggml/src/ggml-cuda/ggml-cuda.cu:5451–5511`; Vulkan `supports_op` at `ggml/src/ggml-vulkan/ggml-vulkan.cpp:19737–19920`; plus file presence from the repo tree API. All *Confirmed*.

⚠️ **Correction to a natural assumption.** `GGML_OP_LIGHTNING_INDEXER` and `GGML_OP_DSV4_HC_{PRE,COMB,POST}` exist in ggml and have SYCL kernels — but **qwen4exp does not use them**. I grepped `src/models/qwen4exp.cpp` for `ggml_lightning_indexer` / `ggml_dsv4_hc*`: zero direct calls, and `src/llama-graph.cpp` contains no fusion pass that would rewrite the primitive pattern into them. The architecture builds hyper-connections from `ggml_rms_norm`/`ggml_mul`/`ggml_sigmoid`/`ggml_scale`/`ggml_repeat_4d`/`ggml_add` in its own `build_hc_mix`/`build_hc_combine` (`qwen4exp.cpp:266,314`), and the indexer from `ggml_mul_mat`/`ggml_relu`/manual head sum. Those ops belong to DeepSeek V4 (`dsv4`); qwen4exp only borrows the `dsv4_hc_mult` hparam *name*. Corroborated by PR #27742, which states *"`git diff master --stat -- ggml/` is empty: no new ggml op"*. Rows below are marked accordingly.

| Op | Used by qwen4exp? | CUDA | SYCL | Vulkan | CPU | Verdict for Arc/SYCL |
|---|---|---|---|---|---|---|
| `GGML_OP_GATED_DELTA_NET` | **Yes** — 36 of 48 layers | yes | **yes** — `ggml-sycl/gated_delta_net.cpp`; `supports_op` returns unconditional `true` | yes, but requires `S_v ∈ {16,32,64,128}` | yes | **OK.** Model's `S_v = 128`, inside Vulkan's set and unconstrained on SYCL. |
| `GGML_OP_MUL_MAT_ID` | **Yes** — 512-expert MoE, every layer | yes | yes | yes | yes | **OK now**, but see §3.3 — had a real B70 prefill correctness bug (#25455) and an IQ-quant performance cliff (#28476) both fixed only recently. |
| `GGML_OP_SOLVE_TRI` | Yes — GDN chunked (WY transform) | yes — unconditional `true` | **yes**, but `N ≤ 64 && K ≤ 64` (`ggml-sycl/solve_tri.hpp:5-6`) | yes, bounded | yes | **OK, but only just.** GDN chunk size is exactly 64, so it sits *at* the limit. Any future chunk-size increase breaks SYCL. |
| `GGML_OP_LIGHTNING_INDEXER` | **No** (DeepSeek V4 only) | yes | yes — `ggml-sycl/lightning-indexer.cpp`; SYCL is well-covered here | yes (added #27453, 2026-08-27) | yes | Not on this model's path. |
| `GGML_OP_DSV4_HC_PRE`/`_COMB`/`_POST` | **No** (DeepSeek V4 only) | yes | yes — `ggml-sycl/dsv4-hc.cpp`, F32-only | yes | yes | Not on this model's path. |
| `GGML_OP_TRI` | GDN chunked decay masks | yes | **yes** — F32 + contiguous | yes | yes | OK. |
| `GGML_OP_CUMSUM` | GDN chunked gate cumsum | yes | **yes** — unconditional `true` | yes, **only if** `subgroup_arithmetic && subgroup_require_full_support` | yes | OK on SYCL. |
| `GGML_OP_FILL` | QSA mask construction | yes | **yes** — unconditional `true` | yes (F32/F16) | yes | OK. |
| `GGML_OP_SET_ROWS` | QSA mask unmasking | yes | **yes** — F32/F16/BF16 src, I32/I64 idx | yes | yes | OK. |
| `GGML_OP_TOP_K` | **QSA block selection** | yes — unbounded with CUB, else `src0->ne[0] ≤ 1024` | **`k > 0 && k <= 32` — HARD LIMIT** | **yes, unbounded** — falls back to radix-select | yes | **❌ THE ONE REAL GAP.** See §2.3. |
| `GGML_OP_GET_ROWS` | PLE table gather, indexer expansion | yes | yes | yes | yes | OK. |
| `GGML_OP_SSM_CONV` | GDN short conv | yes | yes (F32) | yes | yes | OK. |
| `GGML_OP_FLASH_ATTN_EXT` | 12 full-attention layers | yes | yes — see §2.4 | yes | n/a | OK; head_dim 256 is explicitly handled. |
| PLE n-gram hash | PLE row index computation | **host C++** | **host C++** | **host C++** | **host C++** | Not a backend concern at all. |

### 2.3 The one real gap: `GGML_OP_TOP_K` with k ≈ 2051 on SYCL

*Confirmed, and this is the single most actionable finding in the document.*

SYCL, `ggml/src/ggml-sycl/ggml-sycl.cpp:6599`:

```cpp
case GGML_OP_TOP_K: {
    const ggml_tensor * src0 = op->src[0];
    const int k = op->ne[0];
    return src0 && op->type == GGML_TYPE_I32 && src0->type == GGML_TYPE_F32 &&
           ggml_is_contiguous(src0) && k > 0 && k <= 32;
}
```

and the kernel asserts it again at `ggml-sycl.cpp:3101`: `GGML_ASSERT(k > 0 && k <= 32);`

What QSA asks for, `src/models/qwen4exp.cpp:664`:

```cpp
// the reference returns indexer_top_k + compress_ratio - 1: whole blocks plus the tail
const int64_t width = std::min<int64_t>(n_kv, (int64_t) hparams.indexer_top_k + r - 1);
ggml_tensor * top_k = ggml_cont(ctx0, ggml_top_k(ctx0, expanded, width));
```

With the model's real values, read from the shipped GGUF: `indexer.top_k = 2048`, `compress_ratio = 4`, so **`width = min(n_kv, 2051)`**. `ggml_top_k` sets `result->ne[0] = k` (`ggml/src/ggml.c:5465`), so `op->ne[0] = 2051`. That is 64× over the SYCL ceiling. Even at tiny contexts, `width = n_kv`, which exceeds 32 past the first 32 tokens.

**Why the limit exists** — the SYCL kernel's own comment (`ggml-sycl.cpp:2594`) is unusually explicit and worth quoting, because it tells us the fix is not a one-line constant bump:

> `split_block trades parallelism against SLM residency. Its cost is (split_block + 1) * k * 8 bytes of SLM per group, so at the k <= 32 ceiling 128 lanes need about 33 KB, which leaves a single resident group per Xe-core. Revisit if the supported k ever grows.`

At `k = 2051` that formula wants ~2.1 MB of shared local memory per work-group. The scan-merge algorithm simply does not scale; a different algorithm is required.

**What actually happens at runtime:** `supports_op` returns false, so `ggml_backend_sched` assigns that single node to the CPU backend, and the graph is split around it. It will **run correctly** — this is a performance bug, not a correctness bug. But it forces a device→host→device round trip of the `expanded` score tensor on each of the 12 full-attention layers, per ubatch:

- Decode (`n_tokens = 1`), 32 k context: 32768 × 4 B = 128 KiB per layer, ~1.5 MiB per token round-tripped across 12 sync points. *Inferred* — noticeable decode latency, probably tolerable.
- Prefill (`n_ubatch = 2048`), 32 k context: 32768 × 2048 × 4 B = **256 MiB per layer**, ~3 GiB per ubatch. *Inferred* — this would be crippling.

**Mitigations, in rough order of effort** (all *Inferred*, none verified by running):
1. **Use the Vulkan backend instead of SYCL on the Arc.** Vulkan already has `topk_radix_select.comp` and `topk_nary_search.comp` and its `supports_op` explicitly says *"large k falls back to radix-select"* (`ggml-vulkan.cpp:19742`). This is the lowest-effort path to a fully-GPU-resident QSA on Arc — at the cost of the MoE-cache port then targeting Vulkan rather than SYCL.
2. **Port a radix-select top_k to SYCL.** Self-contained, well-scoped work with a working Vulkan reference shader to translate. Probably the highest-value single upstream contribution this project could make.
3. **Disable QSA.** `src/models/qwen4exp.cpp:771` — `const bool qsa = mctx_hyb->get_idx() != nullptr && hparams.dsv4_compress_ratios[il] > 0;` — so a GGUF converted without indexer tensors runs dense. Since QSA currently provides no speed benefit anyway (§2.1), losing it costs only quality, not throughput. **Not yet verified whether a CLI flag exposes this**; see Open questions.

### 2.4 Flash attention with `head_dim = 256` on SYCL — looks fine

*Confirmed.* This was a plausible second blocker (head_dim 256 is unusually large) but it checks out. `ggml/src/ggml-sycl/fattn.cpp:155` gates the XMX/MKL prefill path on:

```cpp
gqa_ratio >= 2 && Q->ne[0] >= 64 && Q->ne[0] <= 512 && Q->ne[0] % 64 == 0 && Q->ne[0] == V->ne[0] ...
```

The model gives `gqa_ratio = 24/2 = 12`, `head_dim = 256` (256 % 64 == 0, within [64,512]), and K/V head sizes match. The file even carries a dedicated `D == 256` dispatch counter (`fattn.cpp:296`). oneDNN and TILE/VEC paths remain as fallbacks.

### 2.5 Graceful auto-fallback exists

*Confirmed.* `src/llama-context.cpp:232–242, 555–580` sets up probes that reserve a graph, check whether each fused op actually landed on the intended device, and disable it if not:

```cpp
cparams.fused_gdn_ar = true;  cparams.fused_gdn_ch = true;  cparams.auto_fgdn = false;
cparams.fused_lid    = true;  cparams.auto_flid = false;
cparams.fused_dsv4_hc_pre = true; ... cparams.auto_fhc = true;
```

with a diagnostic that is going to be extremely useful during bring-up:

> `layer %d is assigned to device %s but %s is assigned to device %s (usually due to missing support)`

So a first SYCL run should be expected to *work*, and to tell us in the log exactly which ops fell off the GPU.

---

## 3. Non-CUDA backend track record

**Short version: yes, people have run this model on both SYCL and Vulkan on Intel Arc, including on Battlemage B60/B65/B70 specifically — but only on builds from ~2026-09-09 onward.** Before that date both backends had a model-breaking or performance-destroying bug.

### 3.0 Minimum viable build date

*Confirmed.* Two fixes landed **2026-09-09** and both are mandatory:

- **Vulkan**: [#28247](https://github.com/ggml-org/llama.cpp/issues/28247) — hard abort running Qwen3.8-Flash-Next on Arc A770, `GGML_ASSERT(wg0 <= maxComputeWorkGroupCount[0] ...) failed` at `ggml-vulkan.cpp:8290`. The failing dispatch is `ggml_vk_fill`, whose workgroup count scales as `n_ubatch × kv_len / 512` and aborts past 65535 (i.e. `ub × kv ≈ 33 M`). **Reproduced on Arc Pro B65 32 GB and on Arc Pro B70.** Fixed by [#28592 "vulkan: Convert FILL to distribute workgroups in 2D"](https://github.com/ggml-org/llama.cpp/pull/28592).
- **SYCL**: [#28476 "SYCL: Add IQ type handling for MoE"](https://github.com/ggml-org/llama.cpp/pull/28476), merged 2026-09-09 — without it, IQ-quantized MoE weights miss `mul_mat_vec_q_moe` and silently host-serialize.

**Action: pin the build to ≥ 2026-09-09.** This is the single most important operational takeaway in the document, and it explains any older "it doesn't work on Arc" report you may encounter.

### 3.1 SYCL — confirmed working on Arc Pro B60

*Confirmed.* PR #28476's author benchmarked exactly our model and quant — `unsloth/qwen3.8-flash-next:UD-IQ3_XXS` on **3× Arc Pro B60**:

> "Running unsloth/qwen3.8-flash-next:UD-IQ3_XXS on 3x Arc Pro B60 is slower than Qwen3.8-27B on Q4 and Q6. Analysis reveals this to be done due to missing IQ quant support in `mul_mat_vec_q_moe`, causing a host serialization and therefore horrific performance."

| test | master (before) | with #28476 |
|---|---|---|
| pp131072 | 152.92 | 153.12 |
| tg512 | 11.17 | **17.92 (+60 %)** |
| tg2048 | 11.59 | **17.24 (+49 %)** |
| tg128 @ d131072 | 4.76 | 5.27 |

This is the only end-to-end SYCL run of qwen4exp found. It is a genuine existence proof, on Battlemage, at our quant.

**Open SYCL bug — avoid tensor-parallel:** [#28100 "Eval bug: qwen 3.8 flash next tensor parallel on sycl"](https://github.com/ggml-org/llama.cpp/issues/28100), 2026-08-31, **still open**. 2× Arc B580: `--split-mode tensor` → `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`, preceded by `llama_params_fit is not implemented for SPLIT_MODE_TENSOR, abort`. Intel maintainer **arthw** advised `--split-mode layer` plus `--cpu-moe`. Not relevant to a single-B70 build, but relevant if we ever add a second card.

### 3.2 Vulkan — confirmed working on Arc Pro B65 32 GB and B70

*Confirmed.* Post-#28592, on **Arc Pro B65 32 GB (Battlemage BMG-G31)** — effectively our card — running `Qwen3.8-Flash-Next UD-Q3_K_XL` with `-b 4096 -ub 4096 -ngl 99 --n-cpu-moe 25 -fa 1`:

| depth | tg |
|---|---|
| `-d 8192` | 30.4 t/s |
| `-d 16384` | 28.3 t/s |
| `-d 32768` | 13.3 t/s |

Note `--n-cpu-moe 25` — they pushed 25 of 48 layers' experts to the CPU, which is the crude static analogue of what a real expert cache would do dynamically. **This is our baseline to beat.**

The same reporter's tuning note is directly load-bearing for §5's VRAM budget:

> "raising `-c` past ~30K on this B65 made *every* prompt ~7× slower — a 10K-token prompt went from 110 t/s prefill (`-c 30720`) to 16.8 t/s (`-c 36864`)... The QSA top-k scratch (`ggml_vk_topk_radix_qsa`, `n_kv × n_ubatch × 4` bytes) plus the KV grows with `-c`, and once the working set exceeds VRAM the Windows driver spills silently over PCIe rather than failing. Moving four more expert layers to the CPU fixed it: `-c 49152 --n-cpu-moe 29` = 118.6 t/s prefill."

Two lessons: (a) **the QSA top-k scratch is `n_kv × n_ubatch × 4` bytes and scales with *allocated* `-c`, not context in use** — at `-c 32768, -ub 4096` that is 512 MiB of scratch, which I did not account for in §5.4; (b) overshooting VRAM degrades silently rather than erroring.

### 3.3 Intel-specific bugs worth knowing about

*Confirmed.* Most relevant first:

- [#25455 "SYCL: MUL_MAT_ID prefill path produces wrong results on **Intel Arc Pro B70 (Battlemage G31)**"](https://github.com/ggml-org/llama.cpp/issues/25455) — opened 2026-07-08, **closed 2026-08-30**. `test-backend-ops -b SYCL0 -o MUL_MAT_ID` failed 28/792 cases, all in the `ne12 > 1` (batched/prefill) path, errors 0.3–1.9 against a 0.0005 tolerance. Decode unaffected. The reporter's comparison is pointed: *"the same model on the Vulkan backend of the same GPU produces fully correct, coherent output at similar speed, so the corruption is SYCL-specific."* This is literally our GPU and the op that all 512-expert MoE routing goes through. Fixed, but re-run `test-backend-ops -o MUL_MAT_ID` on our build as a smoke test.
- [#26581](https://github.com/ggml-org/llama.cpp/issues/26581) (**open**) — decode attention is memory-latency-bound on Arc Pro B70: a constant ~21–25 ns per KV position per attention layer per token, **identical on Vulkan and SYCL**, invariant to KV row width, quant, and build. A hardware/kernel-parallelism ceiling rather than a bug — sets realistic long-context decode expectations for us regardless of backend or expert cache.
- [#25812](https://github.com/ggml-org/llama.cpp/issues/25812) (**open**) — `UR_RESULT_ERROR_OUT_OF_HOST_MEMORY` when offloading MoE experts to Arc GPUs; works with all experts on CPU. Suggests an oversized host/USM staging allocation per expert tensor. **Directly adjacent to expert-cache work** — worth reading before designing the SYCL staging path.
- [#28515](https://github.com/ggml-org/llama.cpp/issues/28515) (**open**, 2026-09-06) — `ggml_backend_sycl_device_get_memory` aborts with "failed to get device memory size" on Arc Pro B60 with the `xe` driver + compute runtime 26.31.39395.13. A driver-version trap.
- [#22885](https://github.com/ggml-org/llama.cpp/issues/22885) — Qwen3-Next layer-split broken on dual Arc Pro B70; root-caused by the reporter to an **intel-compute-runtime (NEO) regression** between 25.40.35563.10 and 26.05.37020.3, *not* llama.cpp. A reminder that NEO version matters independently of llama.cpp version.
- Vulkan MoE, [#28501](https://github.com/ggml-org/llama.cpp/pull/28501) (**open**) — `count_experts.comp` sizes shared arrays with `BLOCK_SIZE` (256), disabling row-id hoisting for models with **>256 experts**. Qwen3.8-Flash-Next has **512**, so it is hitting this. The PR reports +16–19 % prefill on Strix Halo. Free performance for us once merged.

### 3.4 Predecessor architectures — the pattern that motivated the kernels

*Confirmed.* Useful because it shows the GDN path on SYCL has a history of being fixed reactively:

- [#20423](https://github.com/ggml-org/llama.cpp/issues/20423) "SYCL: Qwen3.5 produces gibberish and crashes on Intel Arc A770" — the issue that prompted the SYCL GDN kernel.
- [#24168](https://github.com/ggml-org/llama.cpp/issues/24168) (2026-06-05, **still open**) — 2× Arc Pro B60: `qwen3next` crashes at `ggml_sycl_op_mul_mat` during warmup on SSM/GDN weights; `qwen35` emits number-spam. Explicitly notes *"Pure attention MoE models (qwen3moe arch, no SSM layers) are completely unaffected"* — i.e. the linear-attention path is where SYCL bugs concentrate.

### 3.5 SYCL kernel provenance

*Confirmed.* Who built the SYCL support and when:

| Op(s) | PR | Merged | Notes |
|---|---|---|---|
| `GATED_DELTA_NET` | [#20455](https://github.com/ggml-org/llama.cpp/pull/20455) by **arthw** (Intel) | 2026-03-14 | Fixes #20423. Arc A770 pp 90.8 → 339.2 t/s, tg 10.5 → 11.7 |
| GDN `K>1` (MTP) fix | [#23174](https://github.com/ggml-org/llama.cpp/pull/23174) | 2026-05-22 | *"Without this patch, MTP on SYCL gives garbled output after a few tokens"* — relevant if we pursue MTP |
| `FILL`, `CUMSUM`, `DIAG`, `SOLVE_TRI`, `SSM_SCAN`, `GATED_DELTA_NET` | [#22149](https://github.com/ggml-org/llama.cpp/pull/22149) by Intel aicss-genai | 2026-05-07 | States `SOLVE_TRI` is *"constrained to N ≤ SYCL_SOLVE_TRI_MAX_N, K ≤ SYCL_SOLVE_TRI_MAX_K"* |
| `LIGHTNING_INDEXER`, `DSV4_HC_*` | [#26568](https://github.com/ggml-org/llama.cpp/pull/26568) by **arthw** | 2026-08-07 | Fixes [#26549](https://github.com/ggml-org/llama.cpp/issues/26549), filed against Arc Pro B70. Not on qwen4exp's path. |
| Battlemage AOT / Q5_K / PAD / oneMKL | [#22066](https://github.com/ggml-org/llama.cpp/pull/22066) | **not merged** (split up) | The *only* PR explicitly benchmarked on **Arc Pro B70 (BMG-G31)** |

Also: [#27877 "context: disable non-fused GDN and LID ops"](https://github.com/ggml-org/llama.cpp/pull/27877) by **ggerganov**, merged 2026-08-28 — *"These ops have good backend coverage now (Metal, CUDA, Vulkan, SYCL)... For backends that don't yet support these ops, there is CPU fallback."* This is upstream stating SYCL is a first-class target for the GDN family.

### 3.6 What the upstream support PR claims about backends

*Confirmed.* [#27742 "model: add Qwen3.8-Flash-Next (qwen4exp)"](https://github.com/ggml-org/llama.cpp/pull/27742) by **danielhanchen** (Unsloth), opened 2026-08-26, **merged 2026-08-27**.

**Backend validation claimed: CPU, CUDA and Metal only.** `test-llama-archs -a qwen4exp` "OK on CPU (0.00e+00), CUDA (8.44e-08) and Metal". **No SYCL, Vulkan or ROCm validation is claimed.** The author additionally flags the test as weak: *"its synthesised model carries no PLE tensors, so the PLE `set_input` never runs, and it is insensitive to the GDN fused-QKV segmentation convention."*

So SYCL/Vulkan support for this architecture is **incidental** (inherited from generic op coverage) rather than **validated**. That is consistent with everything above: it works, but nobody upstream is testing it.

### 3.7 Status of sparse attention (PR #27970)

*Confirmed.* [#27970 "CUDA + ggml: add sparse-fa for DSV4/GLM"](https://github.com/ggml-org/llama.cpp/pull/27970) by **am17an**, merged **2026-09-02**. It adds an API hint for max live KV entries per token (`ggml_flash_attn_ext_set_n_kv_max`) rather than sparse indices, and says *"Further sparse attention methods like QSA(qwen4) can be enrolled at a later stage."*

qwen4exp has **not** been enrolled — `qwen4exp.cpp:747` still passes `0`. Two follow-ups:

- [#28349](https://github.com/ggml-org/llama.cpp/pull/28349) would flip that line on. **Closed unmerged 2026-09-04.** Critical line for us: *"Its gate falls back to dense below 4096 KV, and **CPU and Vulkan ignore the hint**."* SYCL likewise has no sparse-FA path. So even when enabled, this benefits CUDA/Metal only.
- [#28213](https://github.com/ggml-org/llama.cpp/pull/28213) (**open**) — gathers selected K/V into a compact buffer for decode instead. +50 % at 130 k on dual A6000. **Backend-agnostic**, so this is the one that would eventually help SYCL/Vulkan. Worth tracking.

### 3.8 Other backends, briefly

*Confirmed.* **Metal**: [#28098 "metal: add sparse FA"](https://github.com/ggml-org/llama.cpp/pull/28098) merged; #28349 benched Qwen3.8-Flash-Next IQ4_XS on M5 Max at pp2048@d131072 340 → 589 t/s. **ROCm/HIP** carries the most qwen4exp bug traffic ([#27856](https://github.com/ggml-org/llama.cpp/issues/27856), [#27797](https://github.com/ggml-org/llama.cpp/issues/27797), [#28266](https://github.com/ggml-org/llama.cpp/issues/28266), [#28201](https://github.com/ggml-org/llama.cpp/issues/28201), [#27865](https://github.com/ggml-org/llama.cpp/issues/27865) — the last being TOP_K "invalid configuration argument", the same op that bites SYCL). **CPU-only** works; #27742's own correctness table shows `qwen4exp` OK on CPU at 0.00e+00.

### 3.9 Lower-confidence / adjacent signal

*Relayed, not independently verified.*

- [steveseguin/b70-optimization-lab](https://github.com/steveseguin/b70-optimization-lab/blob/main/results/qwen38-flash-next-fp8-b70/README.md) — 4× Arc Pro B70, **vLLM XPU (not llama.cpp)**, FP8, TP4/EP4. Decode 5.22 → 34.50 t/s across a series of optimizations; MTP1 37.83 t/s. Reports 8 K and 16 K contexts **unstable/nondeterministic at all MTP depths** with B70 engine resets. Author's own framing: "research-screened, not deployment-qualified." Different stack, but the only other Flash-Next-on-B70 datapoint that exists.
- [PMZFX/intel-arc-pro-b70-benchmarks](https://github.com/PMZFX/intel-arc-pro-b70-benchmarks) — Qwen3-Coder-Next 80B-A3B (`qwen3next`) at 43.4 t/s across 2× B70 on SYCL.
- [Unsloth's own GGUF speed thread](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/discussions/3) — **no Intel/SYCL/oneAPI entries at all.**
- **intel/llm-scaler** and **ipex-llm** releases: newest Qwen support named is Qwen3.6-27B / 3.6-35B-A3B / Qwen3-Coder-Next. **No `qwen4_exp` / Flash-Next / Gated DeltaNet / QSA mention.** Intel's own stack has not caught up to this architecture.
- **Reddit: unknown.** `reddit.com` was blocked for the search tooling and ~15 query variants produced no organic results. Do not read this as "nothing was said there."

### 3.10 A documentation trap

*Confirmed.* **`docs/ops.md` and `docs/ops/*.csv` in the llama.cpp repo are stale — do not trust them.** Regeneration dates: `CUDA.csv` 2025-12-09, `CPU.csv` 2026-03-10, `SYCL.csv` 2026-08-17, `Vulkan.csv` 2026-08-26. They show CUDA ❌ for `GATED_DELTA_NET`/`LIGHTNING_INDEXER`/`TOP_K` and Vulkan ❌ for `LIGHTNING_INDEXER`/`DSV4_HC`, all wrong at master. Every coverage claim in §2.2 was read from `supports_op` in the actual backend source instead, for this reason.

### 3.11 What the source itself implies

Independent of any bug report, the source establishes these as *Confirmed*:

- SYCL kernels for `gated_delta_net`, `lightning-indexer`, `dsv4-hc`, `cumsum`, `fill`, `solve_tri`, `tri`, `set_rows` all exist as dedicated files in `ggml/src/ggml-sycl/`. Somebody deliberately built out this architecture family for SYCL; these are not accidental.
- Vulkan has the matching shaders (`gated_delta_net.comp`, `lightning_indexer.comp`, `dsv4_hc_{pre,comb,post}.comp`, `solve_tri.comp`, `tri.comp`, `cumsum*.comp`, `topk_radix_select.comp`).
- The `moe-cache` fork carries all of the above unchanged, since it tracks upstream.

The presence of `ggml-et`, `ggml-hexagon`, `ggml-opencl`, `ggml-openvino`, `ggml-webgpu` and `ggml-metal` implementations of `solve_tri` in the tree suggests this op family got a broad cross-backend rollout, which is a good sign for non-CUDA maturity generally.

### 3.12 Vendor guidance on non-CUDA

*Confirmed from https://unsloth.ai/docs/models/qwen3.8-next:* Unsloth documents CUDA (`-DGGML_CUDA=ON`) and Apple Metal (`-DGGML_CUDA=OFF`, Metal on by default). **They do not mention Vulkan, SYCL, or Intel Arc at all.** Absence of guidance, not evidence of breakage — but combined with §3.6 (upstream validated only CPU/CUDA/Metal) and §3.9 (Intel's own llm-scaler has no qwen4exp support), it means we are early, with no vendor-blessed recipe on either side.

---

## 4. Architecture numbers

Two fully independent sources agree on every shared field, which is a strong validity check:

- **(A)** GGUF KV metadata parsed directly out of `unsloth/Qwen3.8-Flash-Next-GGUF` `UD-IQ3_XXS` shard headers, read over HTTP range requests against the real published files.
- **(B)** `https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/config.json` (rev `de4b8e4d`), all hyperparameters under `text_config`; plus safetensors headers.

All *Confirmed* unless noted.

### 4.1 Identity

| Field | Value |
|---|---|
| GGUF `general.architecture` | `qwen4exp` |
| GGUF `general.name` / `general.description` | `Qwen3.8 Flash Next` / *"A Preview of the Qwen4 Architecture"* |
| GGUF `general.size_label` | `512x56B` |
| HF `architectures` | `["Qwen4ExpForConditionalGeneration"]`, `model_type: qwen4_exp` |
| Multimodal | **Yes.** Vision tower = unmodified Qwen3-VL ViT, 448.9 M params, `depth 27`, `hidden 1152`, `patch 16`. `mmproj-F16.gguf` is 0.90 GB. |

### 4.2 Layers and attention

| Field | Value | Source |
|---|---|---|
| `block_count` / `num_hidden_layers` | **48** | A + B |
| `embedding_length` / `hidden_size` | **2560** | A + B |
| Residual stream width | **10240** (= 4 hyper-connection branches × 2560) | *Inferred*, confirmed by `[10240]` tensor shapes |
| `full_attention_interval` | **4** | A + B |
| Layer pattern | `[linear, linear, linear, full] × 12` → **36 Gated DeltaNet + 12 full-attention** (indices 3,7,…,47) | A (`attention.compress_ratios = [0,0,0,4,...]`) + B (`layer_types`) |
| `attention.head_count` | **24** | A + B |
| `attention.head_count_kv` | **2** (GQA ratio 12) | A + B |
| `attention.key_length` / `value_length` / `head_dim` | **256** | A + B |
| `rope.freq_base` | **10 000 000** | A + B |
| `rope.dimension_count` | **64** (partial rotary factor 0.25) | A + B |
| `rope.dimension_sections` (mrope) | `[11, 11, 10, 0]` | A |
| `context_length` | **262144** native; 1 M via YaRN `factor 4.0` (README only, not in config.json) | A + B |
| `vocab_size` | **248320**, `tie_word_embeddings: false` | B |

### 4.3 MoE — the numbers that size the expert cache

| Field | Value | Source |
|---|---|---|
| `expert_count` | **512 routed experts per layer** | A + B |
| `expert_used_count` (top-k) | **10** | A + B |
| `expert_feed_forward_length` | **640** | A + B |
| Shared expert | **1, always on**, `expert_shared_feed_forward_length = 640` | A + B |
| Dense early layers | **None — all 48 layers are MoE** | *Inferred* from `mlp.experts.*` present on layers 0–47 |
| **Total expert slots in the model** | **48 × 512 = 24 576** | *Inferred* |
| One expert = | gate `[2560,640]` + up `[2560,640]` + down `[640,2560]` = **4 915 200 params** | A + B |

### 4.4 Qwen Sparse Attention (indexer)

| Field | Value |
|---|---|
| `indexer.head_count` | **4** (`indexer_kv_heads: 1`, MQA) |
| `indexer.key_length` | **128** |
| **`indexer.top_k`** | **2048** ← *this is what breaks SYCL top_k* |
| `indexer_compress_ratio` | **4** (block size 4 → 512 blocks) |

### 4.5 Gated DeltaNet

| Field | Value |
|---|---|
| `linear_num_value_heads` (`ssm.time_step_rank`) | **48** |
| `linear_num_key_heads` (`ssm.group_count`) | **16** |
| `linear_key_head_dim` / `linear_value_head_dim` (`ssm.state_size`) | **128** |
| `linear_conv_kernel_dim` (`ssm.conv_kernel`) | **4** |
| `ssm.inner_size` | **6144** |
| GDN chunk size in llama.cpp | **64** (`delta-net-base.cpp:61`) |

### 4.6 PLE / n-gram table

| Field | Value |
|---|---|
| `ple.layers` | **[1]** — a single layer carries PLE |
| `ple.ngram_size` | **3** (bigrams + trigrams) |
| `ple.heads_per_ngram` | **8** → **16 total heads** |
| `ple.conv_kernel` | **4** |
| `embedding_length_per_layer_input` (row dim) | **160** |
| `ple.head_vocab_sizes` | 16 near-20M primes: `20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077, 20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171` |
| Sum of vocab sizes | 320 001 446, padded to **320 001 536 rows** |
| GGUF tensor | `per_layer_token_embd.weight`, shape **[160, 320001536]** |
| **Total PLE params** | **51 200 245 760 (51.20 B)** — matches the "51 B" claim exactly |
| `ple.layer_multipliers` | `[23703573157769, 20109073645365, 8052911324071]` (45-bit rolling-hash constants, kept as I64 KV, never as a tensor) |
| `ple.eos_token_id` / `ple.image_token_id` | 248044 / 248056 |

⚠️ **Off-by-one gotcha.** `config.json` says `ple_layer_ids: [2]`, but the actual weights are at `model.language_model.layers.1.ple.*` and the GGUF says `ple.layers = [1]`. `conversion/qwen4exp.py` handles this explicitly — *"ple_layer_ids is 1-based in the HF config"* → `ple_layers = [i - 1 for i in hp["ple_layer_ids"]]`. Use **1**.

### 4.7 Hyper-connections

`hyper_connection.count = 4`, `hyper_connection.low_rank = 320`, 640 624 640 params total. These replace every layernorm, and are why the residual stream is 10240 wide rather than 2560.

### 4.8 Parameter census (measured from all 131 safetensors headers)

| Component | Params | Share |
|---|---:|---:|
| **MoE routed experts** | 120 795 955 200 | **67.11 %** |
| **PLE n-gram table** | 51 200 245 760 | **28.44 %** |
| MTP draft head | 2 607 150 848 | 1.45 % |
| Gated DeltaNet | 2 086 510 464 | 1.16 % |
| Hyper-connections | 640 624 640 | 0.36 % |
| embed_tokens | 635 699 200 | 0.35 % |
| lm_head | 635 699 200 | 0.35 % |
| Full attention | 597 694 464 | 0.33 % |
| Vision tower | 448 931 056 | 0.25 % |
| Shared experts | 236 052 480 | 0.13 % |
| MoE routers | 62 914 560 | 0.03 % |
| PLE projections | 32 839 715 | 0.02 % |
| QSA indexers | 19 663 872 | 0.01 % |
| **Total** | **179 999 981 459 (180.0 B)** | |

Cross-check: ×2 bytes = 359 999 962 918, versus the HF index's declared `total_size` 359 999 963 128. *Confirmed consistent.*

**Reconciling the "125B" claim** (*Inferred*): 180.0 B − 51.2 B PLE − 2.6 B MTP = **126.2 B**, i.e. the marketing figure excludes the PLE table and the draft head. Activated per token ≈ 10 × 4 915 200 × 48 = 2.359 B routed + 0.236 B shared + 2.087 B GDN + 0.598 B attn + 0.641 B HC ≈ **6.0 B**, matching the "6B activated" claim.

**The headline structural fact for this project: 67 % of the model is routed experts, and 28 % is a lookup table that never touches the GPU.**

---

## 5. Quantization options for a 32 GB-VRAM target

### 5.1 The key structural insight

*Confirmed.* `src/llama-arch.cpp:905` classifies the PLE table as an **input** tensor:

```cpp
{LLM_TENSOR_PER_LAYER_TOKEN_EMBD, {LLM_TENSOR_LAYER_INPUT, GGML_OP_GET_ROWS}},
```

and the table's own header comment (`src/llama-arch.cpp:706`) states: *"input layers are usually assigned to CPU/host buffer types"*.

**So llama.cpp already keeps the 28.8 GB PLE table in host memory by default, and mmaps it from the GGUF on disk.** Unsloth confirms the intent independently: *"you can also offload the PLE / Ngram layer to SSD and use mmap which allows less usage of CPU and GPU VRAM"* (https://unsloth.ai/docs/models/qwen3.8-next).

This means the "82 GB model on a 32 GB card" framing is misleading. The right way to read the file size is **three separate budgets**.

### 5.2 Measured three-way split per quant

*Confirmed.* I parsed the GGUF tensor headers of every candidate quant directly (HTTP range reads over the published shards, all 1224 tensors accounted for) and summed by category:

| Quant | File total | PLE table (→ host/SSD) | MoE experts (→ cache) | Dense core (→ VRAM) | Per-expert slot |
|---|---:|---:|---:|---:|---:|
| **UD-IQ1_S** | 72.54 GB | 28.80 GB | 39.85 GB | **3.89 GB** | 1.49 MiB |
| UD-IQ1_M | 74.54 GB | 28.80 GB | *(not parsed)* | — | — |
| **UD-Q2_K_XL** | 78.86 GB | 28.80 GB | 46.08 GB | **3.97 GB** | 1.78 MiB |
| **UD-IQ3_XXS** ← the video's | 81.95 GB | 28.80 GB | 48.63 GB | **4.52 GB** | 1.88 MiB |
| **UD-Q3_K_XL** | 89.98 GB | 28.80 GB | 55.82 GB | **5.35 GB** | 2.08 MiB |
| **UD-IQ4_XS** | 93.67 GB | 28.80 GB | 59.52 GB | **5.35 GB** | 2.22 MiB |
| **UD-Q4_K_XL** | 111.32 GB | 28.80 GB | 77.02 GB | **5.51 GB** | 2.93 MiB |
| UD-Q5_K_XL | 158.29 GB | 28.80 GB | — | — | — |
| UD-Q6_K_XL | 169.17 GB | 28.80 GB | — | — | — |
| Q8_0 | 188.23 GB | — | — | — | — |
| BF16 | 354.03 GB | 95.37 GB | 225.00 GB | — | 9.375 MiB |

**Note the constant column.** Unsloth ships `per_layer_token_embd.weight` as **IQ4_NL at exactly 28.80 GB in every quant from IQ1_S through Q4_K_XL**. They never quantize the PLE table below IQ4_NL. It is a fixed floor on host RAM (or SSD, via mmap), and it does not shrink no matter how far down you push the quant.

**Consequence:** picking a smaller quant buys you *nothing* on the PLE budget and *everything* on the expert budget. Choosing IQ1_S over Q4_K_XL cuts experts 77 → 40 GB but leaves 28.8 GB of PLE untouched.

### 5.3 Answering the brief's question directly

> *"what quant levels would fit more entirely in VRAM outright"*

**None.** The smallest published quant is UD-IQ1_S at 72.54 GB. Even ignoring the PLE table entirely, the smallest expert working set (39.85 GB at IQ1_S) still exceeds 32 GB VRAM on its own. There is no quant at which this model is VRAM-resident on a 32 GB card. *Confirmed.*

**But that is the wrong question for this project**, because the dense core — everything that isn't an expert or the PLE table — is only **3.89–5.51 GB** and fits trivially. The real question is expert residency, which is exactly what the MoE cache addresses.

### 5.4 VRAM budget and expert slot-pool sizing

Working memory besides weights (*Confirmed* shapes, *Inferred* arithmetic):

| Item | Size |
|---|---|
| KV cache, 12 full-attn layers, 2 KV heads × 256 dim | **24.00 KiB/token** F16 → 0.19 GiB @8k, **0.75 GiB @32k**, 3.00 GiB @128k, 6.00 GiB @262k |
| same, Q8_0 KV | 12.75 KiB/token → 0.40 GiB @32k |
| same, Q4_0 KV | 6.75 KiB/token → 0.21 GiB @32k |
| GDN recurrent state, 36 layers × 128×128×48 F32 | **0.105 GiB per sequence, context-independent** |
| GDN + PLE conv states | ~4.6 MiB per sequence |
| Indexer K cache, 12 layers × 1 head × 128 | 3.00 KiB/token → 0.094 GiB @32k |
| **QSA top-k scratch** — `n_kv × n_ubatch × 4` B | **scales with *allocated* `-c`, not context in use**: 512 MiB at `-c 32768, -ub 4096`; 128 MiB at `-ub 1024` |

The recurrent state being context-independent is a nice property — long contexts cost only the 12 full-attention layers' KV, not all 48.

⚠️ **The QSA scratch row is the one that bites**, and it is the correction the field report in §3.2 forced. It is charged against *allocated* context, so raising `-c` costs VRAM even if prompts are short — and on Windows, overshooting spills silently over PCIe rather than erroring (7× prefill loss observed on a B65 going from `-c 30720` to `-c 36864`). **Keep `-ub` modest (1024–2048) and `-c` no larger than actually needed**, or the slot pool gets eaten by scratch.

**Slot-pool projection** at 32 GB VRAM, 32 k context, F16 KV, assuming ~0.7 GiB driver reserve and ~1.8 GiB compute buffers (*Inferred* — needs measurement):

| Quant | Fixed VRAM | Expert slot pool | Slots (of 24 576) | Expert residency |
|---|---:|---:|---:|---:|
| UD-IQ1_S | ~6.6 GiB | **22.73 GiB** | 15 620 | **61.2 %** |
| UD-Q2_K_XL | ~6.7 GiB | 22.65 GiB | 13 032 | **52.8 %** |
| UD-IQ3_XXS | ~7.2 GiB | 22.14 GiB | 12 060 | **48.9 %** |
| UD-Q3_K_XL | ~8.0 GiB | 21.37 GiB | 10 520 | **41.1 %** |
| UD-IQ4_XS | ~8.0 GiB | 21.37 GiB | 9 857 | **38.6 %** |
| UD-Q4_K_XL | ~8.1 GiB | 21.22 GiB | 7 416 | **29.6 %** |

**This is a genuinely favourable regime for an expert cache.** With only 10 of 512 experts active per layer per token and 30–60 % of slots resident, hit rates should be well above the residency fraction if there is any routing locality at all. Contrast this with a dense model, where caching buys nothing.

### 5.5 Quality guidance

*Confirmed from https://unsloth.ai/docs/models/qwen3.8-next:* Unsloth's own memory table is **RAM + VRAM combined**, not VRAM: 1-bit 75 GB, 2-bit 79 GB, 3-bit 90 GB, 4-bit 96–114 GB, 5-bit 163 GB, 8-bit 200 GB, BF16 355 GB.

They recommend **1-bit as the starting point**, reporting **UD-IQ1_M at 79.691 top-1 %** accuracy (vs an 80 % reference) at 74.5 GB — i.e. *"79 % smaller than BF16"* with ~1 point of top-1 loss. Their Dynamic GGUF v2 quants are per-tensor mixed, which is why "1-bit" is not uniformly 1-bit.

That mixing is visible in my parse — at UD-IQ3_XXS: `ffn_down_exps` is IQ4_NL on all 48 layers while `ffn_gate_exps`/`ffn_up_exps` are IQ2_S (47 layers) / IQ3_S (1 layer). The down-projection is deliberately kept at higher precision.

⚠️ **K-quant gotcha** (*Inferred*, from observed dtypes): `moe_intermediate_size = 640` is not divisible by 256, so `ffn_down_exps` (inner dim 640) **cannot use K-quants**. In UD-Q4_K_XL, `ffn_gate_exps`/`ffn_up_exps` are Q4_K/Q5_K but `ffn_down_exps` falls back to **Q5_1/Q8_0**. That is why the real per-expert size (2.93 MiB) is ~11 % above a nominal Q4_K expert (2.64 MiB). Anyone hand-rolling a quant for slot-pool efficiency needs to know this.

### 5.6 Recommendation

*Inferred.*

- **Start with UD-Q2_K_XL or UD-IQ3_XXS.** IQ3_XXS is the video's known-good configuration and gives ~49 % expert residency; Q2_K_XL buys ~4 more points of residency for a modest quality cost.
- **UD-IQ1_S** is the aggressive option at 61 % residency — worth benchmarking, since Unsloth claims 1-bit holds up unusually well here.
- **Avoid Q4_K_XL and above** unless the expert cache proves so effective that residency stops mattering; at 29.6 % it is doing the most work for the least benefit, and it is penalised by the 640-divisibility issue above.
- **Host RAM requirement is roughly `28.8 GB (PLE) + (experts − VRAM pool)`**, unless the PLE is mmap'd from SSD, in which case it is `experts − pool` plus page cache. Confirm the box's RAM before choosing.

---

## 6. Open questions / gaps

**Blocking-ish, needs answering before committing to SYCL:**

1. **Is there a CLI flag to disable QSA / the indexer?** §2.3's mitigation 3 depends on it. I confirmed the *graph-level* condition (`qsa = get_idx() != nullptr && compress_ratios[il] > 0`) but did not find a user-facing flag. Check `common/arg.cpp` for indexer/QSA options, and whether `--no-indexer`-style handling exists.
2. **Measure the actual cost of the `top_k` CPU fallback.** Run `llama-bench` on SYCL and read the `layer %d is assigned to device ... (usually due to missing support)` warnings from §2.5. This converts §2.3's inferred prefill estimate into a real number and decides SYCL-vs-Vulkan.
3. **SYCL vs Vulkan on the B70 for this model.** Vulkan wins on `top_k` outright. Does it lose enough elsewhere (matmul throughput, XMX utilisation, MoE `MUL_MAT_ID`) to matter? This is a bake-off, and it also determines which backend the MoE-cache port should target — a decision worth making *before* writing porting code.

**Unverified in this pass:**

4. **§3 is under-evidenced.** The GitHub-issue and Reddit search for real-world non-CUDA reports did not land in time for this document. Specifically still open: any qwen4exp/Qwen3-Next SYCL or Vulkan bug reports; which PRs added the SYCL kernels and what limitations their descriptions note; the qwen4exp support PR and its stated backend coverage; and the status of **PR ggml-org/llama.cpp#27970** (referenced in-source as the gate on real sparse attention).
5. **`SYCL_SOLVE_TRI_MAX_N/K = 64` exactly matches the GDN chunk size of 64.** Confirmed to fit today, but it is a zero-margin coincidence. Worth a comment upstream, and worth re-checking after any llama.cpp update.
6. **Compute-buffer size on SYCL is estimated, not measured** (~1.8 GiB assumed in §5.4). With `n_ubatch = 2048` and a 10240-wide residual stream, this could be materially larger and would eat directly into the slot pool. Measure it.
7. **Unsloth's `qwen4exp/mtp` branch contents unverified** — MTP would add ~2.6 B params and claims 1.3–1.7× decode speedup, but interacts with both the expert cache and the recurrent-state rollback slots (`[TAG_RECURRENT_ROLLBACK_SPLITS]` in `qwen4exp.cpp`). Treat as a later phase.
8. **Vision tower on SYCL untested.** Not needed for a text-only bring-up; `mmproj` is a separate 0.90 GB file, so it can simply be omitted initially.

**Notable for the MoE-cache port specifically:**

9. The expert tensors are **fused 3-D tensors** (`ffn_down_exps [640, 2560, 512]`, `ffn_gate_exps`/`ffn_up_exps` `[2560, 640, 512]`), not per-expert tensors. Any slot pool must slice into a 3-D tensor along `ne[2]`, and the CUDA `moe-cache.cu` presumably already does this — worth confirming that assumption carries over, since it shapes the SYCL memory-management design.
10. **512 experts/layer × 48 layers = 24 576 slots** with ~1.5–2.9 MiB granularity is a much finer-grained pool than typical MoE models (which have 8–128 experts). Check that the cache's metadata structures scale to 24 k entries.

---

## Appendix: source index

**Upstream llama.cpp (`ggml-org/llama.cpp`, branch `master`, read 2026-09-09)**
- `src/llama-arch.cpp` — arch enum L43, PLE KVs L302-310, PLE tensors L542-547, tensor-info/buffer-type map L712+, `PER_LAYER_TOKEN_EMBD` L905
- `src/models/qwen4exp.cpp` — QSA top-k L525-673, `build_attn_qsa` L678-760 (sparse TODO L747), QSA gate L771, PLE host hash L1052-1112, `build_inp_ple` L1171, `build_ple` L1192
- `src/models/delta-net-base.cpp` — chunked path L16+ (CS L61), fused path L375+, dispatch L425-447
- `src/llama-context.cpp` — fused-op defaults L232-242, probe/resolve L515-580
- `ggml/include/ggml.h` — op enum L500-600, `ggml_gated_delta_net` L2652
- `ggml/src/ggml.c` — `ggml_top_k` L5459-5471
- `ggml/src/ggml-sycl/ggml-sycl.cpp` — top_k kernel comment L2594, `ggml_sycl_op_top_k` L3083-3105, compute dispatch L5382-5691, `supports_op` L6376-6660
- `ggml/src/ggml-sycl/solve_tri.hpp` — `SYCL_SOLVE_TRI_MAX_N/K` L5-6
- `ggml/src/ggml-sycl/fattn.cpp` — kernel selection L106+, MKL gate L155
- `ggml/src/ggml-cuda/ggml-cuda.cu` — `supports_op` L5451-5511
- `ggml/src/ggml-vulkan/ggml-vulkan.cpp` — `supports_op` L19737-19920 (top_k radix note L19742)
- `conversion/qwen4exp.py` — full file
- Tree listing: `https://api.github.com/repos/ggml-org/llama.cpp/git/trees/master?recursive=1` (3939 paths)

**Fork**
- `https://api.github.com/repos/GenerelSchwerz/llama.cpp/branches/moe-cache` — head `078a36bc`, 2026-09-09
- `https://raw.githubusercontent.com/GenerelSchwerz/llama.cpp/moe-cache/src/{llama-arch.cpp,models/qwen4exp.cpp}`

**Hugging Face**
- `https://huggingface.co/api/models/unsloth/Qwen3.8-Flash-Next-GGUF?blobs=true` — 60 files, sizes
- GGUF headers parsed via HTTP range reads on `https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main/{UD-IQ1_S,UD-Q2_K_XL,UD-IQ3_XXS,UD-Q3_K_XL,UD-IQ4_XS,UD-Q4_K_XL}/...gguf`
- `https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/config.json` (rev `de4b8e4d`) + model card + safetensors headers
- `https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4` — NVFP4 expert layout

**Vendor docs**
- `https://unsloth.ai/docs/models/qwen3.8-next`
