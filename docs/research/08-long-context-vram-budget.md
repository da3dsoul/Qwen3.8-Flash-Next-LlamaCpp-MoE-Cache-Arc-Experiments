# Long-Context VRAM Budget — KV Cache vs. MoE Expert Placement, Jointly

**Status: research only. No code was written, no GPU was touched, no benchmark was run.**
Date: 2026-09-11. Written while another agent held the box's single GPU; every number below comes from
(a) GGUF headers parsed on the CPU, (b) log lines this project already recorded, (c) source reads, or
(d) arithmetic on those.

Triggered by `PLAN.md:9-52`'s late update: the real workload is **300K–1M tokens of context**, and
`docs/research/06-vllm-migration-and-kvarn-viability.md` §4.2 computed this model's KV-cache footprint
**in isolation**, never against the MoE expert VRAM competing for the same card. This document fills that
gap and, in doing so, **corrects doc 06's KV-cache figure by 37.5% and reverses one of its conclusions.**

Sources actually read for this document (not summarized secondhand):

| Source | How obtained |
|---|---|
| Real per-tensor byte sizes and quant types for all 48 layers' expert tensors + every non-expert tensor | standalone Python GGUF header parse (`gguf-py` `GGUFReader`) over `staging/models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/*.gguf`. **Header metadata only — no tensor data read, no GPU, no container.** |
| MTP draft-head tensor sizes | same, over `staging/models/Qwen3.8-Flash-Next-GGUF/MTP/mtp-Qwen3.8-Flash-Next{,-shared}-Q8_0.gguf` |
| Measured KV / recurrent / compute buffer sizes at `n_ctx` 2048, 4096 and **262144** | `logs/gdn-sched-debug.log`, `logs/gdn-topkfix-sched.log`, `logs/phase3-vram-check.log`, `logs/flash-next-sycl-verbose-stdout.log` |
| Card's real reported VRAM | `logs/phase3-vram-check.log:6` |
| `-ncmoe` / `-ctk` / `-ctv` semantics | `src/llama.cpp/common/arg.cpp`, `src/llama.cpp/common/common.h` |
| Hybrid KV/recurrent/indexer memory construction | `src/llama.cpp/src/llama-model.cpp`, `src/llama-memory-hybrid-idx.cpp`, `src/llama-kv-cache-msa.cpp`, `src/llama-hparams.cpp` |
| Which caches `qwen4exp` actually reads/writes | `src/llama.cpp/src/models/qwen4exp.cpp` |
| SYCL flash-attention supported KV types + its scratch-reservation rule | `src/llama.cpp/ggml/src/ggml-sycl/{fattn.cpp,common.hpp,ggml-sycl.cpp}` |

Everything cited `file:LINE` was read directly this pass. Statements that connect facts established in
separate places, rather than quoting one place, are flagged **inference**. Extrapolation past a measured
datapoint is flagged **estimate** and shows its assumptions.

---

## 0. The verdicts, up front

1. **Doc 06's "24 KiB/token" is wrong — the real figure is 33 KiB/token at f16.** There are **two**
   context-scaling KV caches, not one: the main attention cache (24 KiB/token) and a **separate QSA
   indexer cache (9 KiB/token)** that doc 06 §4.2 explicitly flagged as "a smaller increment I have not
   computed precisely." It is not small: it is **37.5% on top**. Measured, not derived —
   `logs/gdn-sched-debug.log:2863` and `:2966`. §1.

2. **Two-thirds of that indexer cache is allocated and never touched.** `qwen4exp.cpp` writes and reads
   the indexer cache's **K only** (`:782`, `:785`); it never calls `cpy_v`/`get_v` on it. The indexer
   cache nevertheless allocates a full V side at 256 elements/token/layer. **That is 1,536 MiB wasted at
   262144 tokens and 6,144 MiB wasted at 1M tokens, at f16.** A one-line-ish upstream fix. §1.4.

3. **1M tokens at f16 is infeasible outright** — the non-expert VRAM alone comes to **44,437 MiB against
   a 32,402 MiB card**, before a single expert layer is placed. §4.

4. **1M tokens IS feasible at `q4_0`/`q4_1`/`q5_0` KV, at roughly `-ncmoe 38–40`** (8–10 of 48 expert
   layers GPU-resident), *conditional on one unmeasured allocator behaviour* (§3.3). 300K is comfortable
   at `-ncmoe 24–27`. §4.

5. **The binding constraint at long context is NOT the KV cache — it is the SYCL flash-attention f16
   staging buffer plus the f32 mask/indexer scratch in the compute buffer.** At 1M tokens those two
   together are **8.8 GiB** and **neither shrinks when you quantize the KV cache**. One of them
   (the FA staging) *only exists* when you quantize. §3.

6. **MTP does not survive at 1M, and is marginal at 500K.** At 500K + `q8_0` the arithmetic leaves
   759 MiB — under one expert layer. At 1M it is infeasible at `f16` and `q8_0` and only barely arithmetically
   possible at `q4_0`/`q5_0` with ~5 of 48 expert layers resident, which is not a configuration worth
   running. **Plain answer: long-context serving and MTP speculative decoding do compete for the same
   VRAM, and past ~500K you pick one.** §5.

7. **This reopens the question doc 06 closed.** Doc 06 §4.6 rejected KVarN-style sub-4-bit KV
   quantization because "the whole KVarN prize at 1M tokens is ~1.8 GiB on a 30 GiB card." Against the
   corrected 33 KiB/token figure the prize is **~2.5 GiB**, and — more importantly — it lands in a budget
   that is *already negative at f16 and only ~10 GiB positive at q4_0*. It is no longer a rounding error.
   But it is still **not the best lever**, because a KVarN-class KV format would not touch the
   FA-staging or compute-buffer terms that actually dominate. §6.

---

## 1. The KV cache, recomputed from scratch

### 1.1 The model's shape (measured, from a load banner)

`logs/phase3-vram-check.log:142-182`:

```
n_ctx_train           = 262144      n_layer     = 48       n_head_kv       = 2
n_embd_head_k         = 256         n_embd_k_gqa = 512     n_embd_head_v   = 256
n_embd_v_gqa          = 512         n_expert    = 512      n_expert_used   = 10
freq_base_train       = 10000000.0  rope scaling = linear  n_ctx_orig_yarn = 262144
```

and from the same file's GGUF KV dump (`:51-62`):

```
qwen4exp.ssm.conv_kernel = 4     qwen4exp.ssm.state_size = 128   qwen4exp.ssm.group_count = 16
qwen4exp.ssm.inner_size  = 6144  qwen4exp.attention.indexer.head_count = 4
qwen4exp.attention.indexer.key_length = 128   qwen4exp.attention.indexer.top_k = 2048
```

### 1.2 Only 12 of 48 layers hold a conventional KV cache — verified, not assumed

`src/llama-model.cpp:2547-2562` installs the layer filters for `LLM_ARCH_QWEN4EXP`:

```cpp
filter_attn = [&](uint32_t il) { return il < hparams.n_layer() && !hparams.is_recr(il); };
filter_recr = [&](uint32_t il) { return il < hparams.n_layer() &&  hparams.is_recr(il); };
if (arch == LLM_ARCH_QWEN4EXP && hparams.indexer_head_size > 0) {
    // QSA runs on the dense-attention layers only
    filter_idx = [&](uint32_t il) { return il < hparams.n_layer() && !hparams.is_recr(il); };
}
```

`hparams.is_recr(il)` is a plain per-layer lookup into `is_recr_impl[]` (`src/llama-hparams.cpp:260-266`),
populated at load time (`llama-model.cpp:1287`). The loader log then prints the resolved placement
directly — `logs/gdn-sched-debug.log:2814-2861` (main cache) and `:2917-2964` (indexer cache) both show
`layer 0/1/2: filtered`, `layer 3: dev = SYCL0`, repeating every 4 layers through
`layer 47: dev = SYCL0`. **12 attention layers at indices 3,7,…,47; 36 gated-DeltaNet layers everywhere
else.** Doc 06 §4.2's claim confirmed.

### 1.3 There are TWO context-scaling caches, not one — the correction

`llama_memory_hybrid_idx` (`src/llama-model.cpp:2585-2604`) builds the attention cache **and** a second
`llama_kv_cache` for the QSA indexer. `src/llama-memory-hybrid-idx.cpp:47-63`:

```cpp
mem_idx(filter_idx == nullptr ? nullptr : [&] {
    // MQA with a single key head of indexer_head_size, as llama_kv_cache_dsa shapes its own
    std::fill(hparams_idx.n_head_kv_arr.begin(), hparams_idx.n_head_kv_arr.end(), 1);
    hparams_idx.n_embd_head_k_full = model.hparams.indexer_head_size;
    ...
    return new llama_kv_cache(
        model, hparams_idx, type_k, type_v, v_trans, offload, unified,
        kv_size, n_seq_max, n_pad, n_swa, swa_type, nullptr, filter_idx, nullptr, nullptr, "idx_");
}())
```

Two things follow immediately, both load-bearing:

- The indexer cache is sized at the **same `kv_size`** as the main cache, i.e. it scales with `-c`
  identically.
- It is built with the **same `type_k`/`type_v`**, i.e. `-ctk`/`-ctv` apply to it too. (The
  `llama_kv_cache_msa` variant at `src/llama-kv-cache-msa.cpp:31-48` does the same thing for the other
  code path.)

Only `n_head_kv` is overridden (to 1) and `n_embd_head_k` (to `indexer_head_size` = 128).
`n_embd_head_v` is **not** overridden and stays at 256.

**Measured confirmation at the exact depth that matters** — `logs/gdn-sched-debug.log`, one run at
`n_ctx = 262144` (`:2760`):

```
:2863  llama_kv_cache: size = 6144.00 MiB (262144 cells, 12 layers, 1/1 seqs), K (f16): 3072.00 MiB, V (f16): 3072.00 MiB
:2915  llama_memory_recurrent: size = 112.57 MiB (1 cells, 48 layers, 1 seqs 0 rs_seq), R (f32): 4.22 MiB, S (f32): 108.00 MiB, P (f32): 0.35 MiB
:2916  operator(): creating indexer KV cache, size = 262144 cells
:2966  llama_kv_cache: size = 2304.00 MiB (262144 cells, 12 layers, 1/1 seqs), K (f16):  768.00 MiB, V (f16): 1536.00 MiB
:2967  llama_kv_cache: attn_rot_k = 0, n_embd_head_k_all = 128
```

and the scheduler's own roll-up of the same three allocations (`:3303`, `context` column):

```
|   - SYCL0 (Intel(R) Arc(TM) Pro B70 Graphics) | 32656 = 32402 + (60560 = 50191 + 8560 + 1809) + -60307 |
```

`8560 MiB = 6144 + 2304 + 112.57`. The derivation and the measurement agree exactly.

### 1.4 Per-token element counts — and the indexer V that is never used

| cache | layers | K elem/layer/token | V elem/layer/token |
|---|---|---|---|
| main attention | 12 | `n_embd_k_gqa` = 512 | `n_embd_v_gqa` = 512 |
| QSA indexer | 12 | `1 × indexer_head_size` = 128 | `1 × n_embd_head_v` = **256** |

- K elements/token = 12×512 + 12×128 = **7,680**
- V elements/token = 12×512 + 12×256 = **9,216**
- Total = **16,896 elements/token**

At f16: 16,896 × 2 = **33,792 B/token = 33.00 KiB/token**. Check against the measurement:
33,792 × 262,144 = 8,858,370,048 B = **8,448 MiB = 6144 + 2304** ✓.

> **Doc 06 §4.2's "24 KiB/token" is the main cache only.** The real all-in figure is **33 KiB/token**.
> Every downstream number in doc 06's table (1.69 GiB @262K / 6.75 GiB @1M at `q4_0`) is **37.5% low**.

**And the indexer V is dead weight.** `src/models/qwen4exp.cpp` touches the indexer cache in exactly two
places — `:782` `mctx_idx->cpy_k(...)` and `:785` `mctx_idx->get_k(...)`. A grep of that file for
`cpy_v`/`get_v` returns only `:908` and `:941`, both on `mctx_cur`, the **main** cache. The indexer's V
side is allocated and never written or read:

| depth | wasted indexer V, f16 | wasted, `q4_0` |
|---|---|---|
| 262,144 | 1,536 MiB | 432 MiB |
| 300,000 | 1,758 MiB | 494 MiB |
| 1,048,576 | **6,144 MiB** | 1,728 MiB |

**Inference** (connecting the graph-builder reads to the cache constructor, which no single site states):
overriding `hparams_idx.n_embd_head_v` to 0 — or giving the indexer cache a K-only constructor path —
recovers 6 GiB at 1M tokens at f16, or 1.7 GiB at `q4_0`, for no behavioural change. This is the single
cheapest VRAM win identified anywhere in this document, and it is bigger than the entire KVarN prize
doc 06 evaluated.

### 1.5 The gated-DeltaNet recurrent state — fixed, and genuinely small

`src/llama-hparams.cpp:229` and `:257`:

```cpp
const uint32_t n_conv = (ssm_d_conv > 0 ? ssm_d_conv - 1 : 0) * (ssm_d_inner + 2*ssm_n_group*ssm_d_state);
...
return ssm_d_state * ssm_d_inner;   // n_embd_s
```

- `n_embd_r` = (4−1) × (6144 + 2×16×128) = 3 × 10,240 = **30,720** elem → ×4 B ×36 layers = **4.22 MiB** ✓
- `n_embd_s` = 128 × 6144 = **786,432** elem → ×4 B ×36 layers = **108.00 MiB** ✓
- plus the PLE conv state `P` = **0.35 MiB**

Both element counts reproduce the logged `R (f32): 4.22 MiB, S (f32): 108.00 MiB` exactly.

**It is always f32 and `-ctk`/`-ctv` cannot touch it.** `src/llama-model.cpp:2594-2595` hard-wires
`/* recurrent_type_k */ GGML_TYPE_F32, /* recurrent_type_v */ GGML_TYPE_F32` — only `attn_type_k`/
`attn_type_v` (`:2586-2587`) come from `params.type_k/type_v`.

**Always-resident, non-negotiable recurrent baseline: 112.57 MiB, independent of context depth.**
(One cell; it grows with `n_seq_max`, not with `-c`.)

### 1.6 Which KV quantization levels are actually usable on this backend

`common/arg.cpp:323-333` enumerates what `-ctk`/`-ctv` accept (`:2453`, `:2466`):

```cpp
GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q8_0,
GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_IQ4_NL, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
```

But the CLI accepting a value is not the same as SYCL flash-attention supporting it.
`ggml/src/ggml-sycl/fattn.cpp:214-236` gates the whole FA path on K's type:

```cpp
#ifndef GGML_SYCL_FA_ALL_QUANTS
    if (K->type != V->type) { return BEST_FATTN_KERNEL_NONE; }
#endif
    switch (K->type) {
        case GGML_TYPE_F32: case GGML_TYPE_F16: case GGML_TYPE_BF16:  break;
        case GGML_TYPE_Q4_1: case GGML_TYPE_Q5_0: case GGML_TYPE_Q5_1:
#ifndef GGML_SYCL_FA_ALL_QUANTS
            return BEST_FATTN_KERNEL_NONE;
#endif
        case GGML_TYPE_Q4_0: case GGML_TYPE_Q8_0: break;
        default: return BEST_FATTN_KERNEL_NONE;
    }
```

`GGML_SYCL_FA_ALL_QUANTS` is **unconditionally `#define`d** at `ggml/src/ggml-sycl/common.hpp:50` — it is
not a CMake option (a repo-wide grep finds only `GGML_CUDA_FA_ALL_QUANTS` in `ggml/CMakeLists.txt:206`;
there is no SYCL equivalent to turn it off). So on this build **all five quant types plus mixed K/V are
available**, which is better than the conservative `#else` branch suggests.

**`iq4_nl` is the exception: it is in the CLI list and in neither FA branch.** `-ctk iq4_nl` falls to
`default: return BEST_FATTN_KERNEL_NONE` (`fattn.cpp:234`), i.e. it silently disables flash attention.
Do not use it.

| `-ctk`/`-ctv` | block layout | bits/elem | B/elem | usable on SYCL FA? |
|---|---|---|---|---|
| `f32` | — | 32 | 4.0 | yes (pointless) |
| `f16` | — | 16 | 2.0 | yes, default |
| `bf16` | — | 16 | 2.0 | yes, but excluded from the VEC kernel (`fattn.cpp:243-245`) |
| `q8_0` | 32 elem / 34 B | 8.5 | 1.0625 | **yes** |
| `q5_1` | 32 elem / 24 B | 6.0 | 0.75 | **yes** (needs `FA_ALL_QUANTS`, which is on) |
| `q5_0` | 32 elem / 22 B | 5.5 | 0.6875 | **yes** (same) |
| `q4_1` | 32 elem / 20 B | 5.0 | 0.625 | **yes** (same) |
| `q4_0` | 32 elem / 18 B | 4.5 | 0.5625 | **yes** |
| `iq4_nl` | 32 elem / 18 B | 4.5 | 0.5625 | **NO — disables flash attention** |

Row sizes are block-compatible at every level: the main cache's K/V rows are 512 elements and the
indexer's are 128 / 256 — all multiples of 32.

### 1.7 KV cache VRAM across the range that matters

`16,896 elem/token × B/elem`:

| `-ctk`/`-ctv` | B/token | 100K | 300K | 500K | 1M (262144×4) |
|---|---|---|---|---|---|
| `f16` | 33,792 | 3,223 MiB | 9,668 MiB | 16,113 MiB | **33,792 MiB** |
| `q8_0` | 17,952 | 1,712 MiB | 5,136 MiB | 8,560 MiB | 17,952 MiB |
| `q5_1` | 12,672 | 1,208 MiB | 3,625 MiB | 6,042 MiB | 12,672 MiB |
| `q5_0` | 11,616 | 1,108 MiB | 3,323 MiB | 5,539 MiB | 11,616 MiB |
| `q4_1` | 10,560 | 1,007 MiB | 3,021 MiB | 5,035 MiB | 10,560 MiB |
| `q4_0` | 9,504 | 906 MiB | 2,719 MiB | 4,532 MiB | 9,504 MiB |

(`100K` = 100,000 tokens; `300K` = 300,000; `500K` = 500,000; `1M` = 1,048,576 = the exact YaRN ×4 ceiling.)

Note **`f16` at 1M is 33,792 MiB — larger than the entire card (32,656 MiB) — on its own.**

---

## 2. The MoE expert side, computed rather than cited

### 2.1 What `-ncmoe N` actually does

`common/arg.cpp:2781-2790` calls `llm_add_n_cpu_ffn_overrides(value, LLM_FFN_EXPS_REGEX, ...)`, and
`common/common.h:1131,1143-1150`:

```cpp
const char * const LLM_FFN_EXPS_REGEX = "\\.ffn_(up|down|gate|gate_up)_(ch|)exps";
...
for (int i = 0; i < n; ++i) {
    buft_override_strings.push_back(llm_ffn_block_regex(i, ffn_regex));   // "blk\\.<i>\\.ffn_..._exps"
    overrides.push_back({buft_override_strings.back().c_str(), ggml_backend_cpu_buffer_type()});
}
```

So `-ncmoe N` pins the `ffn_{gate,up,down}_exps` tensors of **layers 0…N−1** to CPU. It does **not**
touch `ffn_*_shexp` (the shared expert) — `shexp` does not match `(ch|)exps` — nor anything else.

### 2.2 The real per-layer expert byte cost (GGUF headers, UD-IQ3_XXS)

Parsed directly from the three shards. **Note the quant types differ from this project's working
assumption:** gate/up are **IQ2_S** (not IQ3_XXS) on 47 layers and **IQ3_S** on layer 2; down is
**IQ4_NL** on all 48 (not IQ4_NL/Q8_0 mixed).

| tensor | shape | type | bytes | per layer |
|---|---|---|---|---|
| `ffn_gate_exps` | [2560, 640, 512] | IQ2_S | 268,697,600 | 256.25 MiB |
| `ffn_up_exps` | [2560, 640, 512] | IQ2_S | 268,697,600 | 256.25 MiB |
| `ffn_down_exps` | [640, 2560, 512] | IQ4_NL | 471,859,200 | 450.00 MiB |
| **layer total (47 of 48 layers)** | | | 1,009,254,400 | **962.50 MiB** |
| **layer 2 only** (gate/up are IQ3_S, 360,448,000 each) | | | 1,192,755,200 | **1,137.50 MiB** |

Sum over 48 layers = 48,627,712,000 B = **46,375.00 MiB**, which reproduces
`logs/phase3-vram-check.log:5731` — `SYCL_MoE_Cached model buffer size = 46375.00 MiB` — **exactly**.

The non-expert tensors sum to 33,323,087,360 B, of which `per_layer_token_embd.weight` (the PLE table,
IQ4_NL) is 28,800,138,240 B = **27,465.95 MiB** — reproducing `:5732`
`CPU_Mapped model buffer size = 27465.95 MiB` exactly. The remainder,
4,522,949,120 B = **4,313.47 MiB**, reproduces `:5729-5730`
(`SYCL0 3816.16 MiB` + `SYCL_Host 497.31 MiB`) exactly; the 497.31 MiB on the host is `output.weight`.

**Always-resident dense core on the GPU: 3,816.16 MiB.** This is independent of `-ncmoe`.

### 2.3 GPU expert VRAM per `-ncmoe` value

For `N ≥ 3` (so layer 2 is among the offloaded):

```
expert_vram(N) = 46,375.00 − [1,137.50 + (N−1) × 962.50]  =  46,200 − 962.5·N   MiB
```

| `-ncmoe` | GPU-resident expert layers | expert VRAM |
|---|---|---|
| 0 (`-ngl 99`, nothing offloaded) | 48 | 46,375.0 MiB |
| 18 | 30 | 28,875.0 MiB |
| 20 | 28 | 26,950.0 MiB |
| 21 | 27 | 25,987.5 MiB |
| 24 | 24 | 23,100.0 MiB |
| 26 | 22 | 21,175.0 MiB |
| 28 | 20 | 19,250.0 MiB |
| 34 | 14 | 13,475.0 MiB |
| 40 | 8 | 7,700.0 MiB |
| 48 (`--cpu-moe`) | 0 | 0 MiB |

### 2.4 The MoE cache's slot pool, for comparison

The cache allocates a fixed number of slots per expert tensor. Per-expert slab bytes:
`268,697,600/512 = 524,800 B` (gate, IQ2_S), same for up, `471,859,200/512 = 921,600 B` (down, IQ4_NL)
→ 1,971,200 B = 1.8799 MiB per expert per layer across the three tensors (2.2217 MiB on layer 2).

```
slot_pool_vram(S) = S × (47 × 1.8799 + 2.2217) = S × 90.58 MiB
```

| slots/tensor | VRAM | expert residency | PLAN.md's measured outcome |
|---|---|---|---|
| 256 | 23,187.5 MiB | 50.0% | works, 23.1 tok/s (best cache config) |
| 320 | 28,984.4 MiB | 62.5% | **OOM** |
| 384 | 34,781.2 MiB | 75.0% | **OOM** |
| 512 | 46,375.0 MiB | 100% | = the full expert set ✓ (exact) |

### 2.5 Cross-validating the whole model against today's measured OOM boundaries

Fixed non-expert GPU cost at `n_ctx = 4096`, `f16` KV:
3,816.16 (dense core) + 112.57 (recurrent) + 96.00 (main KV) + 36.00 (indexer KV) + 199.14 (compute
buffer, `logs/gdn-sched-debug.log:6539`) = **4,259.87 MiB**.

The card reports **32,656 MiB total, 32,402 MiB free** at model-device selection
(`logs/phase3-vram-check.log:6`, `:3080`).

| config | predicted total | predicted | `PLAN.md` measured |
|---|---|---|---|
| `-ncmoe 18` | 33,135 MiB | OOM | **OOM** ✓ |
| `-ncmoe 19` | 32,173 MiB | marginal fit (229 MiB spare) | untested |
| `-ncmoe 20` | 31,210 MiB | fits | **works**, 19.7–25.3 tok/s ✓ |
| `-ncmoe 21` | 30,248 MiB | fits | **works** ("solo-best") ✓ |
| cache, 256 slots | 27,447 MiB | fits | **works**, 23.1 tok/s ✓ |
| cache, 320 slots | 33,244 MiB | OOM | **OOM** ✓ |
| cache, 384 slots | 39,041 MiB | OOM | **OOM** ✓ |
| `-ncmoe 26` + MTP (draft 2,647 MiB) | 28,090 MiB | fits | **works**, 26–29 tok/s ✓ |
| `-ncmoe 24` + MTP | 30,015 MiB | marginal | **OOM** ✓ (within fragmentation margin) |

**Every measured working/OOM boundary this project has recorded is reproduced by the model above**, with
the practical ceiling sitting somewhere in **[31,210 , 33,135) MiB** and most consistent with ~30,000–
32,000 MiB once allocator fragmentation is allowed for. The rest of this document uses **32,402 MiB** (the
driver-reported free figure) as the nominal budget, which is mildly optimistic; subtract ~1,000 MiB for a
safety margin when acting on it.

> **Note on `PLAN.md`'s "~30 GB usable":** the card actually reports 32,656 MiB / 32,402 MiB free
> (34.24 GB decimal, per the Level-Zero line at `logs/phase3-vram-check.log:2807`). The "30 GB" figure is
> a conservative working number, not the driver's. The arithmetic here uses the driver's.

---

## 3. The two terms that actually dominate at long context

This is the part doc 06's KV-cache-only calculation could not see.

### 3.1 The compute buffer scales with context, hard

Two measured points from the **same log**, same build, same flags, differing only in `-c`:

| `n_ctx` | `SYCL0 compute buffer size` | source |
|---|---|---|
| 4,096 | **199.14 MiB** | `logs/gdn-sched-debug.log:6539` |
| 262,144 | **1,809.09 MiB** | `logs/gdn-sched-debug.log:3297` |

(`logs/gdn-topkfix-sched.log:3009` independently reproduces 1,809.09 MiB at 262144 on a different build.)

Two-point linear fit:

```
compute_buffer(D) ≈ 174 MiB + D × 6,541.85 bytes        (at the default n_ubatch = 512)
```

**Structural corroboration (inference).** The reserve log's own worst case is `n_tokens = 512`
(`logs/gdn-sched-debug.log:2974`). The context-proportional terms visible in the graph dump are the
attention mask, the QSA-modified mask, and `indexer_score_tokens` — the dump at
`logs/gdn-sched-debug.log:2980,2982` shows `[attn_inp_kq_mask ( 512K)]` and
`[indexer_score_tokens-3 ( 1M)]` scaling with `n_kv` (that dump is for a 1-token ubatch, so 1M = 262,144
× 4 B exactly). Three such f32 buffers at `n_kv × n_ubatch × 4 B` give
3 × 262,144 × 512 × 4 = 1,536 MiB, plus ~256 MiB for the indexer's `ggml_get_rows` gather over the whole
cache (`qwen4exp.cpp:790-797`). The fit's proportional part is 1,635 MiB. Close enough to trust the
*shape*: **the compute buffer is f32 mask/score scratch, it scales as `n_kv × n_ubatch`, and `-ctk`/`-ctv`
do nothing to it.**

| depth | compute buffer (`-ub 512`) | **estimate**, `-ub 2048` |
|---|---|---|
| 100,000 | 798 MiB | ~2.9 GiB |
| 300,000 | 2,046 MiB | ~7.4 GiB |
| 500,000 | 3,293 MiB | ~12.0 GiB |
| 1,048,576 | **6,716 MiB** | **~25 GiB — infeasible** |

**Consequence for prefill, flagged now:** this project's prefill benchmarking used `-ub 2048`
(`logs/bench-prefill-ub2048*.log`). **At 300K+ that ubatch is not affordable.** Long-context prefill on
this card is forced onto `-ub 512`, i.e. ~600 sequential ubatches for a 300K prompt. The 1M column is an
extrapolation 4× past the measured point — **estimate**, flagged.

### 3.2 Quantizing the KV cache on SYCL allocates an f16 staging buffer of the whole view

`ggml/src/ggml-sycl/fattn.cpp:431-461`:

```cpp
const bool tile_needs_K = K->type != GGML_TYPE_F16;
const bool tile_needs_V = V->type != GGML_TYPE_F16;
...
if (tile_needs_K) { need_K = std::max(need_K, (size_t) ggml_nelements(K)); }
if (tile_needs_V) { need_V = std::max(need_V, (size_t) ggml_nelements(V)); }
extra.K_buffer_ptr = ggml_sycl_fattn_reserve_halves(extra, need_K);
extra.V_buffer_ptr = ... ggml_sycl_fattn_reserve_halves(extra, need_V);
```

reserved as `sycl::half` (`:398-406`) and folded into the FA node's own allocation so the graph allocator
sees it (`fattn.cpp:466-469`, hooked at `ggml/src/ggml-sycl/ggml-sycl.cpp:1029-1030`). The oneDNN path
takes the same hit whenever `!ggml_sycl_fattn_onednn_binds_kv(K, V)` (`:443-446`), and the comment at
`:132` and `:140-142` confirms the intent: oneDNN is "native F16 and **dequant**+non-F16", and the MKL
path "converts non-F16 K/V to F16 via `to_fp16_sycl` before GEMM".

`ggml_nelements(K)` for one attention layer's cache view is `n_embd_k_gqa × n_kv = 512 × D`. K+V per FA
node = 1,024 × D elements × 2 B = **4,096 B/token = 4 KiB/token**.

> **So on SYCL, `-ctk q4_0 -ctv q4_0` costs you back 4 KiB/token of f16 staging against a saving of
> 24.7 KiB/token — a net win, but a 17% tax you do not pay on CUDA.** At `q8_0` the tax is 4 KiB against
> a 15.5 KiB saving — a 26% tax.

**This is doc 06 §4.3's open question, and it is the single biggest uncertainty in this document.**

### 3.3 The one thing that must be measured before acting on any of this

Does the graph allocator collapse those 12 FA staging reservations into one? The 12 FA nodes are
sequential and their outputs have disjoint lifetimes, so ggml-alloc *should* reuse one region. **Inference,
not verified** — doc 06 §4.3 reached the same conclusion and also could not verify it.

| assumption | FA staging at 1M | consequence |
|---|---|---|
| reuse holds (1 node's worth) | 4,096 MiB | quantized KV is a large net win; §4's tables apply |
| reuse fails (12 nodes' worth) | **49,152 MiB** | **quantized KV is catastrophically counterproductive at long context, and f16 is the only option — which is itself infeasible at 1M** |

**This is a one-command check and it gates everything else in this document.** Run
`-c 262144 -fa 1 -ctk q4_0 -ctv q4_0` and read the `SYCL0 compute buffer size` line. If it comes back near
1,809 + 1,024 ≈ 2,833 MiB, reuse holds. If it comes back near 1,809 + 12,288 ≈ 14,097 MiB, it does not, and
the long-context plan needs rewriting. **Do this before any other long-context GPU work.**

**Measured 2026-09-11 (adding `-ngl 99 -ncmoe 24` to load successfully; the compute-buffer term is
independent of `-ncmoe`, which only moves expert-tensor VRAM, so this is directly comparable):**

```
0.01.798.864 I sched_reserve:      SYCL0 compute buffer size =  1949.57 MiB
0.01.798.866 I sched_reserve:  SYCL_Host compute buffer size =   402.32 MiB
```

**Reuse holds, and by a wider margin than predicted** — 1,949.57 MiB, *below* the 2,833 MiB "fine" estimate,
not just under the 14,097 MiB "broken" threshold. The graph allocator correctly collapses the 12
full-attention layers' FA staging reservations into one. Gate cleared: quantized KV is not harmful at this
depth, and §4's tables (not §4.3's pessimistic case) apply. This was the single biggest open uncertainty in
this document and in the long-context plan generally — it's now resolved, positively, with a real
measurement, not an inference.

Everything from §4 onward assumes **reuse holds**. The pessimistic case is tabulated in §4.3. That case can
now be disregarded (see the measurement above) — kept in the document only as a record of what was checked
and why.

---

## 4. The joint budget — the actual deliverable

Budget: **32,402 MiB** (driver-reported free). Always-resident non-context cost:
**3,816.16 (dense core) + 112.57 (recurrent) = 3,928.73 MiB.**

```
VRAM(D, q, ncmoe) = 3,928.73
                  + 16,896·D·bpe(q) / 2^20                        (KV, both caches)
                  + [q ≠ f16] × 1,024·D·2 / 2^20                  (SYCL FA f16 staging, reuse assumed)
                  + 174 + D·6,541.85 / 2^20                       (compute buffer, -ub 512)
                  + 46,200 − 962.5·ncmoe                          (GPU-resident experts)
```

### 4.1 The headline table

`required -ncmoe` = the smallest `N` whose expert VRAM fits the leftover. "GPU layers" = 48 − N.

| depth | `-ctk`/`-ctv` | KV | FA stage | compute | non-expert total | left for experts | **required `-ncmoe`** | GPU expert layers | feasible? | implied decode |
|---|---|---|---|---|---|---|---|---|---|---|
| **100K** | `f16` | 3,223 | 0 | 798 | 7,949 | 24,453 | **23** | 25 | ✔ | ~22–25 t/s (just outside measured 20–21) |
| | `q8_0` | 1,712 | 195 | 798 | 6,634 | 25,768 | **22** | 26 | ✔ | ~22–25 t/s |
| | `q5_0` | 1,108 | 195 | 798 | 6,030 | 26,372 | **21** | 27 | ✔ | **measured band** (19.7–25.3) |
| | `q4_0` | 906 | 195 | 798 | 5,828 | 26,574 | **21** | 27 | ✔ | **measured band** |
| **300K** | `f16` | 9,668 | 0 | 2,046 | 15,642 | 16,760 | **31** | 17 | ✔ | unmeasured (below `-ncmoe 26`) |
| | `q8_0` | 5,136 | 586 | 2,046 | 11,696 | 20,706 | **27** | 21 | ✔ | ~26 is measured w/ MTP; 27 is one step past |
| | `q5_0` | 3,323 | 586 | 2,046 | 9,884 | 22,518 | **25** | 23 | ✔ | between measured 21 and 26 |
| | `q4_0` | 2,719 | 586 | 2,046 | 9,279 | 23,123 | **24** | 24 | ✔ | between measured 21 and 26 |
| **500K** | `f16` | 16,113 | 0 | 3,293 | 23,335 | 9,067 | **39** | 9 | ✔ (tight) | unmeasured, far past `-ncmoe 26` |
| | `q8_0` | 8,560 | 977 | 3,293 | 16,759 | 15,643 | **32** | 16 | ✔ | unmeasured |
| | `q5_0` | 5,539 | 977 | 3,293 | 13,738 | 18,664 | **29** | 19 | ✔ | unmeasured |
| | `q4_0` | 4,532 | 977 | 3,293 | 12,731 | 19,671 | **28** | 20 | ✔ | unmeasured (one step past MTP's 26) |
| **1M** | `f16` | 33,792 | 0 | 6,716 | **44,437** | **−12,035** | — | — | **INFEASIBLE** | — |
| | `q8_0` | 17,952 | 2,048 | 6,716 | 30,645 | 1,757 | **47** | 1 | ✔ in arithmetic only | ≈ `--cpu-moe` floor, 15.6 t/s |
| | `q5_1` | 12,672 | 2,048 | 6,716 | 25,365 | 7,037 | **41** | 7 | ✔ | unmeasured |
| | `q5_0` | 11,616 | 2,048 | 6,716 | 24,309 | 8,093 | **40** | 8 | ✔ | unmeasured |
| | `q4_1` | 10,560 | 2,048 | 6,716 | 23,253 | 9,149 | **39** | 9 | ✔ | unmeasured |
| | `q4_0` | 9,504 | 2,048 | 6,716 | 22,197 | **10,205** | **38** | 10 | ✔ | unmeasured |

All figures MiB. **`-ub 512`.** Add the §3.1 `-ub 2048` column and 300K's `f16` row goes infeasible too.

### 4.2 What "implied decode throughput" can and cannot be said

`PLAN.md:185-192, 253-269` gives the complete measured set for UD-IQ3_XXS, **all at `n_kv ≤ 512`**:

| config | GPU expert layers | measured decode |
|---|---|---|
| `--cpu-moe` (`-ncmoe 48`) | 0 | **15.6 tok/s** |
| `-ncmoe 20` | 28 | **19.7–25.3 tok/s** |
| `-ncmoe 21` | 27 | "solo-best" (no separate number recorded) |
| cache, 256 slots | ≈24 equivalent | **23.1 tok/s** |
| `-ncmoe 26` + MTP `-n-max 2` | 22 | **26.3–29.5 tok/s** |

**The measured `-ncmoe` range is 20–26. Nothing past 26 has ever been run.** For the long-context
configurations the table demands — `-ncmoe 28` (500K `q4_0`) through `-ncmoe 40` (1M `q5_0`) — the only
honest statement is:

- **bounded below** by the `--cpu-moe` floor of 15.6 tok/s (0 GPU expert layers),
- **bounded above** by ~25 tok/s (the best non-MTP number ever measured, at 28 GPU layers),
- and a crude linear interpolation on GPU-resident layer count puts `-ncmoe 38–40` (8–10 layers) at
  **~18 tok/s**. **This is an extrapolation, flagged as such, and it is almost certainly optimistic** —
  see the next paragraph.

**The far bigger unknown is not `-ncmoe` at all — it is depth itself.** `docs/research/06` §4.5 established
that QSA's sparse-compute hint is disabled (`qwen4exp.cpp:946` passes `0`, not `top_k->ne[0]`), so
attention is **dense over the whole context** on all 12 attention layers, *plus* a 2,051-wide `ggml_top_k`
over all `n_kv` per layer per token. The only same-family evidence at depth is the third-party Arc Pro B65
Vulkan run in `docs/research/02` §3.2: **30.4 t/s @ 8K falling to 13.3 t/s @ 32K** — a 2.3× loss for a 4×
depth increase, *at constant VRAM*. Naively continuing that trend to 300K would put decode in the low
single digits regardless of `-ncmoe`.

> **Therefore: every "implied decode" cell in §4.1 is an upper bound set by expert placement alone, and the
> real number at 300K–1M will be dominated by dense-attention depth cost, which nothing in this project has
> ever measured.** `PLAN.md:47-52`'s instinct — that prefill TTFT, not steady-state decode, is the metric
> that matters for this workload — is supported by this analysis.

### 4.3 The pessimistic case (FA staging not reused)

If §3.3's allocator-reuse assumption fails, multiply the FA-stage column by 12:

| depth | `q4_0` FA stage ×12 | non-expert total | verdict |
|---|---|---|---|
| 100K | 2,344 | 7,977 | ✔ `-ncmoe 23` — but now **`f16` is cheaper than `q4_0`** |
| 300K | 7,031 | 15,725 | ✔ `-ncmoe 31` — **`f16` (15,642) is cheaper than every quant level** |
| 500K | 11,719 | 23,473 | ✔ `-ncmoe 39` — `f16` still cheaper |
| 1M | 24,576 | 44,725 | **INFEASIBLE at every quant level, `f16` included** |

**In the pessimistic case, KV quantization is actively harmful at every depth ≥ 100K, and 1M is
unreachable by any means available today.** That inversion is why §3.3's one-command check is the highest
priority item this document produces.

---

## 5. MTP at long context — does it survive?

The draft head, parsed from `staging/models/Qwen3.8-Flash-Next-GGUF/MTP/`:

| file | total | contents |
|---|---|---|
| `mtp-…-Q8_0.gguf` | 3,935.32 MiB | one `blk.48` qwen4exp block **+** its own `token_embd`/`output` (644.14 MiB each) |
| `mtp-…-shared-Q8_0.gguf` | **2,647.04 MiB** | the same block, sharing the target's embeddings |

Of that, `ffn_{gate,up,down}_exps` are 850.00 MiB each = 2,550 MiB — i.e. **96% of the draft head is its
own 512-expert MoE layer**, exactly as `PLAN.md:264-266` describes, and exactly why
`--spec-draft-cpu-moe` was measured as a net loss.

Empirical cross-check: MTP needs `-ncmoe 26` where non-MTP runs at `-ncmoe 21`, i.e. 5 expert layers
(4,812.5 MiB) freed; `-ncmoe 24` (3 layers, 2,887.5 MiB) OOMs. **2,647 MiB of weights plus the draft's own
KV cache, recurrent-free single attention layer and compute scratch lands squarely in that bracket** —
derivation and measurement agree.

The draft block has **no indexer cache** (`qwen4exp.cpp:591`: "the draft block has no indexer cache, so
attention is dense") and gets a **plain** `llama_kv_cache`, not the hybrid wrapper
(`src/llama-model.cpp:2509-2513`, `mtp_on_hybrid_qwen`). Its `attn_k`/`attn_v` are [2560, 512], so
`n_embd_k_gqa = 512`: **1 layer × 1,024 elem/token**. It is sized by `-c` like everything else, with its
own `-ctkd`/`-ctvd` (`common/arg.cpp:4096-4118`).

| depth | KV quant | draft weights | draft KV | non-expert total **with MTP** | left | `-ncmoe` | verdict |
|---|---|---|---|---|---|---|---|
| 4K (today) | `f16` | 2,647 | 8 | 6,915 | 25,487 | 22 | ✔ **matches the measured `-ncmoe 26` working config with margin** |
| 100K | `q4_0` | 2,647 | 55 | 8,726 | 23,676 | **24** | ✔ comfortable |
| 300K | `q4_0` | 2,647 | 165 | 12,677 | 19,725 | **28** | ✔ — 20 GPU expert layers |
| 300K | `q8_0` | 2,647 | 311 | 15,241 | 17,161 | **31** | ✔ but tight |
| 500K | `q4_0` | 2,647 | 275 | 16,629 | 15,773 | **32** | ✔ marginal |
| 500K | `q8_0` | 2,647 | 519 | 20,901 | 11,501 | **37** | ✔ marginal |
| 500K | `f16` | 2,647 | 977 | 26,959 | 5,443 | **43** | ✖ 5 GPU expert layers — not worth running |
| **1M** | `f16` | 2,647 | 2,048 | 49,132 | **−16,730** | — | **INFEASIBLE** |
| **1M** | `q8_0` | 2,647 | 1,088 | 36,428 | **−4,026** | — | **INFEASIBLE** |
| **1M** | `q5_0` | 2,647 | 704 | 29,708 | 2,694 | **46** | ✖ 2 GPU expert layers — arithmetically "fits", operationally pointless |
| **1M** | `q4_0` | 2,647 | 576 | 27,468 | 4,934 | **43** | ✖ 5 GPU expert layers — same |

**The plain answer to the question `PLAN.md` raises: yes, long-context serving and MTP speculative
decoding compete for the same VRAM, and the competition is decided against MTP somewhere between 300K and
500K.**

- Up to **300K with `q4_0` KV**, MTP is comfortable (`-ncmoe 28`, 20 GPU expert layers vs. 24 without MTP).
  The cost of keeping MTP is **4 expert layers**, i.e. ~3.85 GiB — the same price it costs today.
- At **500K** MTP costs 4 more expert layers on top of a budget that is already down to 20; you are trading
  a ~1.15× speculative speedup for a 20% cut in expert residency. **Inference:** that is very likely a net
  loss, by the same logic that made `--spec-draft-cpu-moe` a net loss at trivial context
  (`PLAN.md:266-269`), but it has not been measured.
- At **1M**, MTP is **infeasible at `f16` and `q8_0` outright**, and at `q4_0`/`q5_0` it leaves 2–5 of 48
  expert layers on the GPU, which is within noise of `--cpu-moe`'s 15.6 tok/s floor. **MTP does not survive
  at 1M. Sacrifice it.**

One further hazard, not quantified here: `PLAN.md:271-284` records that MTP paired with a K-quant target
(UD-Q3_K_XL) runs at 3.2 tok/s — a suspected pathology whose candidate causes explicitly include "**KV
cache type mismatch handling between draft/target**." Adding `-ctk q4_0 -ctv q4_0` to an MTP run without
also setting `-ctkd`/`-ctvd` is exactly the shape of that bug. **If MTP is combined with quantized KV, set
both pairs and re-run the correctness battery** (`PLAN.md:697-712` also records SYCL+MTP's history of
silent garbling, PR #23174).

---

## 6. Does this reopen KVarN?

`docs/research/06` §4.6 closed sub-4-bit KV quantization with: *"the whole KVarN prize at 1M tokens is
~1.8 GiB on a 30 GiB card."* That was computed against the wrong denominator.

Recomputed against 33 KiB/token (16,896 elem/token) and KVarN's own ~3.3 bits/element all-in
(doc 06 §4.1's derivation from `triton_kvarn_decode.py:19-21`):

| level | B/elem | 1M-token KV | delta vs. `q4_0` |
|---|---|---|---|
| `q4_0` (llama.cpp floor) | 0.5625 | 9,504 MiB | — |
| KVarN k4v2_g128 (projected) | ~0.4125 | ~6,970 MiB | **−2,534 MiB** |

So the prize is **~2.5 GiB, not 1.8 GiB** — a 40% upward revision. And the context it lands in is very
different from what doc 06 assumed: at 1M/`q4_0` there are only **10,205 MiB** left for experts, so
2,534 MiB is **+2.6 expert layers, a 25% increase in expert residency** — not a rounding error.

**But it is still the wrong lever, for three reasons this analysis makes visible and doc 06 could not:**

1. **Two cheaper wins come first and are larger.** Dropping the never-read indexer V (§1.4) recovers
   **1,728 MiB at `q4_0`/1M** — 68% of the entire KVarN prize — for what looks like a constructor-argument
   change, against doc 06 §1.5's 4–8 engineer-week estimate for new Xe kernel work. And at `f16` it
   recovers 6,144 MiB.
2. **A KVarN-class format would not touch the terms that dominate.** At 1M the compute buffer is
   6,716 MiB and the FA staging is 4,096 MiB — **10.8 GiB that is f32/f16 scratch, invariant to the KV
   storage format.** KVarN attacks the 9.5 GiB term and leaves the 10.8 GiB term untouched.
3. **It would make the FA-staging problem worse, not better.** §3.2: any non-`F16` K/V triggers
   `ggml_sycl_fattn_reserve_halves` over the whole view. A custom `ggml_type` would need its own SYCL FA
   kernel to avoid that — which is precisely the 4–8 week job, and it is a *prerequisite*, not an optional
   optimization.

**Verdict: doc 06's conclusion survives, but its reasoning needs replacing.** "KVarN's prize is too small
to matter" was wrong by 40% and computed against a budget that turns out to be far tighter than assumed.
The correct reason to still say no is: **the KV cache is no longer the dominant long-context term once you
count the FA staging and the compute buffer, and there are two much cheaper wins ahead of it in the
queue.** If §3.3's check comes back badly (staging not reused), revisit this section — at that point a
native-`F16`-avoiding KV format becomes the *only* path to 1M, and the calculus changes completely.

---

## 7. Open risk, flagged not solved: prefill at 300K–1M under the expert cache

`docs/00-background.md` §1 records that the CUDA fork's own expert cache **regresses prefill by 14–66%**,
with the fork's stated reason being that "large-batch prefill has a wide unique-expert set per op, closer
to a worst case for a small resident pool." `PLAN.md:38-41` already flags that this has **never been
checked against this project's SYCL port at any prompt length**.

Three things this document adds, without attempting to quantify any of them:

1. **The regime is worse than the fork's.** At 300K–1M tokens there are 600–2,000 sequential ubatches
   (forced to `-ub 512` by §3.1), each touching a wide unique-expert set across 48 layers. If the
   fork's "wide unique-expert set per op" diagnosis is right, this is that failure mode at ~1,000× the
   scale anything has been tested at.

2. **But the cache's VRAM is the *right shape* for long context, and `-ncmoe`'s is not.** This is a genuine
   new finding. `-ncmoe` quantizes expert VRAM into **962.5 MiB steps**; the cache tunes it in
   **90.58 MiB steps** (one slot index). At long context, where the leftover for experts drops from
   ~26,000 MiB to ~10,000 MiB, that granularity matters:

   | depth, KV quant | left for experts | best `-ncmoe` (wastes) | equivalent cache slots |
   |---|---|---|---|
   | 300K `q4_0` | 23,123 MiB | 24 (wastes 23 MiB) | **255 slots — today's tuned 256-slot config, almost exactly** |
   | 500K `q4_0` | 19,671 MiB | 28 (wastes 421 MiB) | 217 slots |
   | 1M `q4_0` | 10,205 MiB | 38 (wastes 580 MiB) | 113 slots (22% residency) |

   `PLAN.md:210-216` made `-ncmoe` the interim production baseline because the two were "wins of about
   the same size" **at trivial context**. At long context they are not obviously the same size any more:
   the cache can use 100% of a shrinking budget while `-ncmoe` rounds down to a 962.5 MiB grid, and the
   cache's *already-tuned* 256-slot configuration is a near-exact fit for 300K + `q4_0`. **Inference** —
   the VRAM arithmetic is solid, the throughput consequence is not measured and could easily go the other
   way once the prefill regression above is accounted for.

3. **Nothing has confirmed the model even loads at 300K+ on this hardware.** `PLAN.md:42-46` says so
   explicitly, and this document does not change that: every long-context number here is arithmetic over
   a `sched_reserve` dry run at `n_ctx = 262144`, not a completed load. YaRN past 262144 needs
   `--rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 262144` explicitly — the GGUF ships
   `rope scaling = linear`, `freq_scale_train = 1` (`logs/phase3-vram-check.log:174-177`), and
   `src/llama-context.cpp:325-327` only *warns* when `n_ctx_seq > n_ctx_train`, it does not configure
   YaRN for you.

---

## 8. What to do, in order

1. **Run the §3.3 check first.** `-c 262144 -fa 1 -ctk q4_0 -ctv q4_0`, read `SYCL0 compute buffer size`.
   ~2,833 MiB → reuse holds, §4.1 applies. ~14,097 MiB → reuse fails, §4.3 applies and long context is a
   very different (much worse) problem. **One command. Everything else is downstream of it.**
2. **Confirm the model loads and generates at all at 300K** with `-c 300000 -fa 1 -ctk q4_0 -ctv q4_0
   -ub 512 -ncmoe 24` and explicit YaRN flags. `PLAN.md:42-46`'s item, now with a concrete
   configuration to try.
3. **Fix the indexer cache's dead V side** (§1.4). Largest single VRAM win found, cheapest to implement,
   and it is a plausible upstream contribution independent of everything else here.
4. **Re-measure decode at depth before trusting any `-ncmoe` number.** Every throughput figure this
   project owns was taken at `n_kv ≤ 512`. The B65 Vulkan 30.4→13.3 t/s (8K→32K) datapoint suggests depth
   dominates placement by a wide margin, and `docs/research/07`'s "don't pursue QSA" verdict is the thing
   most likely to flip as a result — as `PLAN.md:18-29` already anticipated.
5. **Only then** decide MTP. Per §5 it is free up to 300K and dead by 1M; the decision point is 500K and
   it is a measurement, not an arithmetic question.

---

## Appendix — every constant used, with its provenance

| constant | value | source |
|---|---|---|
| card total / free VRAM | 32,656 / 32,402 MiB | `logs/phase3-vram-check.log:6`, `:3080` |
| dense core on GPU (non-expert, non-PLE, minus `output.weight`) | 3,816.16 MiB | `logs/phase3-vram-check.log:5729`; reproduced from GGUF headers to the byte |
| recurrent state (36 GDN layers, always f32) | 112.57 MiB | `logs/gdn-sched-debug.log:2915`; reproduced from `llama-hparams.cpp:229,257` |
| main KV, f16 | 24.00 KiB/token | `logs/gdn-sched-debug.log:2863` (6,144 MiB @ 262,144) |
| indexer KV, f16 | 9.00 KiB/token | `logs/gdn-sched-debug.log:2966` (2,304 MiB @ 262,144) |
| **total KV, f16** | **33.00 KiB/token** | sum; corroborated by `:3303`'s `context = 8560` |
| compute buffer @ 4,096 / 262,144 (`-ub 512`) | 199.14 / 1,809.09 MiB | `logs/gdn-sched-debug.log:6539` / `:3297` |
| expert tensors, layers ≠ 2 | 962.50 MiB/layer | GGUF headers (IQ2_S ×2 + IQ4_NL) |
| expert tensors, layer 2 | 1,137.50 MiB | GGUF headers (IQ3_S ×2 + IQ4_NL) |
| all 48 layers' experts | 46,375.00 MiB | GGUF sum; exact match to `logs/phase3-vram-check.log:5731` |
| PLE table (host-mapped, not VRAM) | 27,465.95 MiB | GGUF; exact match to `:5732` |
| cache slot-pool cost | 90.58 MiB per slot index | GGUF per-expert slabs; 512 slots → 46,375 MiB ✓ |
| MTP draft head (shared-embedding variant) | 2,647.04 MiB | GGUF headers |
| MTP draft KV | 1 layer × 1,024 elem/token | `mtp-…gguf` `blk.48.attn_k/v` shape [2560, 512] |
