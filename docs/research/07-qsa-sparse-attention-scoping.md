# QSA Sparse Attention (the "lightning indexer") for `qwen4exp` on SYCL/Arc — Scoping & Verdict

**Status: research only. No code was written, no GPU was touched, no benchmark was run.**
Date: 2026-09-11. Triggered by the `TODO` at `src/models/qwen4exp.cpp:943-946`, which builds the entire sparse
top-k mask and then deliberately discards the sparse hint by passing `0` to `build_attn_mha`.

Sources actually read for this document (not summarized secondhand):

| Source | How obtained |
|---|---|
| Our GGUF's real metadata (all 67 KV pairs) | standalone Python GGUF header parser, run on the host CPU against `staging/models/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS/…-00001-of-00003.gguf`. **No GPU, no container, no `llama-gguf`.** |
| PR #27970 file list + merge date + body benchmark table | `gh pr view 27970 --repo ggml-org/llama.cpp --json title,mergedAt,files,body` |
| PR #27970 diff (the `qwen4exp.cpp` hunk specifically) | `gh pr diff 27970 --repo ggml-org/llama.cpp` |
| The commit that actually wrote the `TODO` | `git log -S "TODO: enable sparse attention when we are ready" -- src/models/qwen4exp.cpp` → `7bb0fc18f` |
| Our tree: `src/models/qwen4exp.cpp`, `src/llama-graph.cpp/.h`, `src/llama-kv-cache.cpp`, `ggml/include/ggml.h`, `ggml/src/ggml.c`, `ggml/src/ggml-cuda/fattn.cu`, `ggml/src/ggml-cuda/fattn-common.cuh`, `ggml/src/ggml-metal/ggml-metal-ops.cpp`, `ggml/src/ggml-sycl/{fattn.cpp,fattn-common.hpp,fattn-tile.hpp,fattn-onednn.cpp,ggml-sycl.cpp}`, `ggml/src/ggml-cpu/ops.cpp` | direct reads, line ranges cited below |

Everything cited as `file:LINE-LINE` was read directly. Statements that connect facts established in separate
places, rather than quoting one place, are flagged **inference**. Arithmetic that extrapolates a measured
datapoint is flagged **estimate** and shows its assumptions.

All repo paths below are relative to
`/media/da3dsoul/Golias/AIProjects/Qwen3.8-Flash-Next-LlamaCpp-MoE-Cache-Arc-Experiments/src/llama.cpp/`
unless stated otherwise. Tree state at time of writing: `HEAD = 6d9c82ea2`, working tree dirty (this project's
own SYCL MoE-cache / MTP / top-k work is uncommitted — `git status --short` lists 15 modified files including
`src/models/qwen4exp.cpp` and `ggml/src/ggml-sycl/ggml-sycl.cpp`).

---

## 0. Verdict in one paragraph

`indexer_top_k = 2048`, `compress_ratio = 4` for every full-attention layer, so the sparse budget is
`width = min(n_kv, 2051)` and **the sparse restriction is literally a no-op until `n_kv > 2051`** — i.e. until
about 2 049 tokens of context, which this project has never tested (all testing has used ~10-token prompts and
`-n 300`). Flipping `qwen4exp.cpp:946` to pass `top_k->ne[0]` is **safe** — provably a silent no-op on our
stack, because the hint is written into `op_params[4]` of the `FLASH_ATTN_EXT` node
(`ggml/src/ggml.c:5561-5568`) and **no SYCL and no CPU code path ever reads `op_params[4]` for that op** (only
CUDA and Metal do). The mask remains the single source of truth in every backend, so output is bit-identical
either way; this is why the flag is an optimization hint, not a semantic switch. But safe-and-inert is also
worthless: getting any speed out of it requires implementing sparse FA in the SYCL backend, and the payoff only
exists at context depths (≥ 16k, really ≥ 32k) that this project does not currently run — and at exactly those
depths a *different*, larger SYCL problem bites first: QSA's `ggml_top_k(k=2051)` falls back to a full bitonic
argsort whose multi-pass global-memory path fires at roughly `n_kv > 16384` and costs ~120-170 kernel launches
per layer per ubatch. **Recommendation: do not do this now. Leave line 946 as it is.** Revisit only if and when
this project takes on long-context decode as a goal, and even then fix the top-k argsort first.

---

## 1. The real numbers for our model (step 1)

### 1.1 Where the values come from in code

`src/models/qwen4exp.cpp:56-62` reads the indexer hparams straight from the GGUF, with a hard failure if any
of the three is zero:

```cpp
    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);
    qwen4exp_require_nonzero(ml, LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    qwen4exp_require_nonzero(ml, LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    qwen4exp_require_nonzero(ml, LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);
    ml.get_key_or_arr(LLM_KV_ATTENTION_COMPRESS_RATIOS, hparams.dsv4_compress_ratios, hparams.n_layer_all, false);
```

The key strings are `"%s.attention.indexer.top_k"` and `"%s.attention.compress_ratios"`
(`src/llama-arch.cpp:285`, `src/llama-arch.cpp:292`). Which layers are full-attention is derived at
`src/models/qwen4exp.cpp:127-135`:

```cpp
    if (!ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all, false)) {
        uint32_t full_attn_interval = 4;
        ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
        ...
        hparams.is_recr_impl[i] = (i < hparams.n_layer()) && ((i + 1) % full_attn_interval != 0);
    }
```

and QSA engages per layer at `src/models/qwen4exp.cpp:968-970`:

```cpp
    const bool qsa = mctx_hyb != nullptr && mctx_hyb->get_idx() != nullptr && hparams.dsv4_compress_ratios[il] > 0;
    ggml_tensor * top_k = qsa ? build_qsa_top_k(mctx_hyb, cur, inp_pos, inp->get_kq_mask(), sections, il) : nullptr;
```

— so a layer runs QSA iff its `compress_ratio` entry is nonzero, and the `compress_ratios` array is therefore
the authoritative per-layer switch, not `full_attention_interval`.

### 1.2 The values actually in our GGUF

Parsed directly out of `Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf` (shard 1 carries the full KV block;
`split.count = 3`, `split.tensors.count = 1224`):

| Key | Value |
|---|---|
| `general.architecture` | `qwen4exp` |
| `qwen4exp.block_count` | 48 |
| `qwen4exp.full_attention_interval` | 4 |
| **`qwen4exp.attention.indexer.top_k`** | **2048** |
| **`qwen4exp.attention.compress_ratios`** | `[0,0,0,4, 0,0,0,4, 0,0,0,4, 0,0,0,4, 0,0,0,4, 0,0,0,4, 0,0,0,4, 0,0,0,4, 0,0,0,4, 0,0,0,4, 0,0,0,4, 0,0,0,4]` (48 entries) |
| `qwen4exp.attention.indexer.head_count` | 4 |
| `qwen4exp.attention.indexer.key_length` | 128 |
| `qwen4exp.attention.head_count` / `head_count_kv` | 24 / 2 |
| `qwen4exp.attention.key_length` / `value_length` | 256 / 256 |
| `qwen4exp.context_length` | 262144 |
| `qwen4exp.expert_count` / `expert_used_count` | 512 / 10 |

**Correction to the project's prior note.** `PLAN.md:312` and `docs/research/02` §2.3 say QSA needs
`ggml_top_k(width≈2051)`. That number is *right*, but it is the **derived width**, not `indexer_top_k`. The
GGUF's `indexer_top_k` is **2048**; 2051 is `2048 + 4 - 1`. The distinction matters below because the CUDA
gate keys off `n_kv_max` (= the width, 2051), not off `indexer_top_k`. The earlier "k≈2051" figure is now
independently verified against the actual GGUF we run, and both numbers are correct in their own terms.

The `compress_ratios` array puts `4` at indices 3, 7, 11, …, 47 — exactly the 12 layers where
`(il + 1) % 4 == 0`, i.e. exactly the non-recurrent layers, 12 of 48. That is consistent, and it means
**`r = 4` for every QSA layer in this model** (there is no per-layer variation to account for, unlike DSV4,
which uses distinct CSA/HCA ratios — see `src/llama-kv-cache-dsv4.cpp:1276-1284`).

---

## 2. The crossover arithmetic (step 2)

### 2.1 The budget formula

`src/models/qwen4exp.cpp:860-867`:

```cpp
    // the reference returns indexer_top_k + compress_ratio - 1: whole blocks plus the tail
    const int64_t width = std::min<int64_t>(n_kv, (int64_t) hparams.indexer_top_k + r - 1);

    ggml_tensor * top_k = ggml_cont(ctx0, ggml_top_k(ctx0, expanded, width));
    top_k = ggml_reshape_4d(ctx0, top_k, width, n_tps, 1, n_stream);
```

With our values: `width = min(n_kv, 2048 + 4 - 1) = min(n_kv, 2051)`.

The hint the disabled line would pass is `top_k->ne[0]`, which **is** `width` — already clamped to `n_kv`.

### 2.2 What `n_kv` is, in tokens

`src/llama-kv-cache.cpp:1250-1260`:

```cpp
uint32_t llama_kv_cache::get_n_kv(const slot_info & sinfo) const {
    ...
    const uint32_t n_pad_cur = std::max(n_pad, 256u);
    ...
        result = std::max(std::min(cells.size(), std::max(n_pad_cur, GGML_PAD(cells.used_max_p1(), n_pad_cur))), result);
```

So `n_kv` is the used cell high-water mark rounded **up to a multiple of 256**, floored at 256, capped at the
allocated cache size. Concretely: 10 used tokens → `n_kv = 256`; 300 used → `n_kv = 512`; 2 049 used →
`n_kv = 2304`.

### 2.3 The crossover

`width < n_kv` ⟺ `n_kv > 2051` ⟺ `n_kv ≥ 2304` ⟺ **used tokens ≥ 2 049** (and `-c ≥ 2304`).

Below that, `width == n_kv` exactly: the "sparse set" **is** the whole KV cache. This is the crisp statement
of what the earlier quick math suspected, now grounded in the real `indexer_top_k`:

| used tokens | `n_kv` | `width` | fraction of KV the mask keeps |
|---:|---:|---:|---:|
| 10 (our standard test prompt) | 256 | 256 | 100% |
| 300 | 512 | 512 | 100% |
| 2 048 | 2 048 | 2 048 | 100% |
| **2 049** | **2 304** | **2 051** | **89.0%** |
| 4 096 | 4 096 | 2 051 | 50.1% |
| 8 192 | 8 192 | 2 051 | 25.0% |
| 16 384 | 16 384 | 2 051 | 12.5% |
| 32 768 | 32 768 | 2 051 | 6.3% |
| 131 072 | 131 072 | 2 051 | 1.6% |
| 262 144 (model max) | 262 144 | 2 051 | 0.8% |

**The DSV4 curve does not transfer, and it is not conservative in our favour.** The prompt's worry — "maybe our
`compress_ratio`/`indexer_top_k` are small enough that even our short test prompts already exceed the budget" —
is checked and answered: they are not. 2 051 is *large*, and PR #27970's own benchmark table (§3.3 below)
showing ~1.00× below `d32768` is, if anything, *optimistic* relative to us, because the first useful sparsity
ratio (50%) needs 4k of context.

### 2.4 What the flag actually changes downstream — read, not guessed

`src/llama-graph.cpp:2591-2601` is the signature; the parameter is named `n_kv_max`, not "sparse":

```cpp
ggml_tensor * llm_graph_context::build_attn_mha(
         ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
         ggml_tensor * kq_b, ggml_tensor * kq_mask, ggml_tensor * sinks, ggml_tensor * v_mla,
             int64_t   n_kv_max, float kq_scale, int il) const {
```

There are exactly **two** uses of `n_kv_max` in the whole function body, both inside the flash-attention branch
(`src/llama-graph.cpp:2637-2638`):

```cpp
        GGML_ASSERT(n_kv_max >= 0 && n_kv_max <= INT32_MAX);
        ggml_flash_attn_ext_set_n_kv_max(cur, static_cast<int32_t>(n_kv_max));
```

The non-flash-attention `else` branch (`src/llama-graph.cpp:2659-2722`, the `ggml_mul_mat` + `ggml_soft_max_ext`
path) **never mentions `n_kv_max` at all**. So:

- **Without `-fa`, flipping line 946 is a literal no-op in the graph builder.** Nothing is even written.
- **With `-fa`**, the only effect is one `int32` written into the node's op-params:

`ggml/src/ggml.c:5561-5568`:

```cpp
void ggml_flash_attn_ext_set_n_kv_max(struct ggml_tensor * a, int32_t n_kv_max) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_ASSERT(n_kv_max >= 0);
    ggml_set_op_params_i32(a, 4, n_kv_max);
}
```

Slot 4 is otherwise unused for this op: `ggml_flash_attn_ext` writes `{scale, max_bias, logit_softcap}` into
slots 0-2 (`ggml/src/ggml.c:5529-5530`) and `ggml_flash_attn_ext_set_prec` uses slot 3
(`ggml/src/ggml.c:5542-5551`). No aliasing risk.

The API contract is explicit that this is a *hint*, `ggml/include/ggml.h:2505-2509`:

```c
    // Use finite mask entries as a sparse K/V set. Set 0 to disable.
    // n_kv_max must bound the number of finite entries in every mask row.
```

Metal's implementation restates it in the same terms (`ggml/src/ggml-metal/ggml-metal-ops.cpp:2860-2862`):

> `// the mask (src[3]) remains the single source of truth: finite entries are the valid KV positions,`
> `// n_kv_max is only an upper bound on their number per mask row, used to size the index lists`

### 2.5 The "already paying for it" question, answered

The prompt's framing is right and worth stating precisely, because it changes the cost/benefit shape:

The expensive machinery — `build_qsa_top_k` (`src/models/qwen4exp.cpp:721-870`: the indexer K/Q projections,
block pooling, two RoPEs, the score mul_mat, the ReLU-rectified head sum, and `ggml_top_k(width)`) and the
mask construction (`src/models/qwen4exp.cpp:911-937`: `ggml_fill(-INF)` → `ggml_set_rows` of zeros at the
top-k indices → `ggml_add` with the causal mask) — is built **unconditionally** on every QSA layer, today, in
every build, regardless of line 946. `build_layer_attn` calls `build_qsa_top_k` at line 970 before
`build_attn_qsa` is ever entered, and the resulting `kq_mask_top_k` **is already what gets passed to
`build_attn_mha`** at line 946.

**Inference** (connecting `qwen4exp.cpp:946` with `llama-graph.cpp:2632-2638`): the model as it ships today
already computes *numerically sparse* attention — the top-k restriction is applied via `-INF` mask entries —
it just executes it *densely*. Enabling the flag therefore cannot change model output at all; it can only
change how the attention kernel walks the KV cache. That is a materially better safety story than "turning on
a new attention mode", and it is the single most important structural fact in this document.

Corollary: the often-cited justification for leaving QSA off ("it costs us compute for no benefit") is only
half true. We are already paying 100% of the indexer + top-k + mask cost and receiving 0% of the attention
saving. Turning the flag on does not add that cost. It also, on SYCL, does not remove anything (§3).

---

## 3. Backend support status (step 3) — the safety question

### 3.1 Which backends read `op_params[4]` for `FLASH_ATTN_EXT`

Whole-tree grep for `n_kv_max` under `ggml/src/` returns matches in **exactly four files**:

```
ggml/src/ggml.c
ggml/src/ggml-cuda/fattn-common.cuh
ggml/src/ggml-cuda/fattn.cu
ggml/src/ggml-metal/ggml-metal-impl.h
ggml/src/ggml-metal/ggml-metal-ops.cpp
ggml/src/ggml-metal/kernels/fa.metal
```

(`ggml.c` is the setter itself.) **No Vulkan, no OpenCL, no CPU, no SYCL.** That grep is name-based, so it was
cross-checked by reading the op-params accesses directly:

- **SYCL** reads `KQV->op_params` at exactly five sites, and every one of them touches only float slots 0/1/2:
  `ggml/src/ggml-sycl/fattn-common.hpp:1112-1114`, `ggml/src/ggml-sycl/fattn.cpp:125-128`,
  `ggml/src/ggml-sycl/fattn-mkl.cpp:371-373`, `ggml/src/ggml-sycl/fattn-onednn.cpp:96-97` and `:251`,
  `ggml/src/ggml-sycl/fattn-vec.hpp:620`, `ggml/src/ggml-sycl/fattn-tile.hpp:1160` and `:1219`. Slot 4 is
  never touched, on either of this project's two FA paths (the oneDNN/MKL prefill path *or* the vec/tile decode
  path).
- **SYCL's `supports_op`** for `GGML_OP_FLASH_ATTN_EXT` (`ggml/src/ggml-sycl/ggml-sycl.cpp:7354-7355`)
  delegates to `ggml_sycl_flash_attn_ext_supported`, and the kernel selector
  (`ggml/src/ggml-sycl/fattn.cpp:106-274`) branches on head sizes, K/V types, `gqa_ratio`, `max_bias`,
  `logit_softcap` and strides — never on `op_params[4]`. So the hint cannot change *op placement* either.
- **CPU**: `ggml_compute_forward_flash_attn_ext_f16` (`ggml/src/ggml-cpu/ops.cpp:9212-9300`) contains no
  `op_params` / `ggml_get_op_params` access at all in that range.

### 3.2 What PR #27970 actually touched

`gh pr view 27970 --repo ggml-org/llama.cpp --json files` — merged **2026-09-02T14:27:38Z**:

| file | +/− |
|---|---|
| `ggml/include/ggml.h` | +6 −0 |
| `ggml/src/ggml.c` | +9 −0 |
| `ggml/src/ggml-cuda/fattn-common.cuh` | +19 −4 |
| `ggml/src/ggml-cuda/fattn-mma-f16.cuh` | +149 −74 |
| `ggml/src/ggml-cuda/fattn-tile.cuh` | +6 −6 |
| `ggml/src/ggml-cuda/fattn-vec.cuh` | +1 −1 |
| `ggml/src/ggml-cuda/fattn.cu` | +133 −0 |
| `src/llama-graph.cpp` / `.h` | +10 −7 / +1 −0 |
| `src/models/deepseek4.cpp` | +4 −3 |
| `src/models/qwen4exp.cpp` | +1 −1 |
| `tests/test-backend-ops.cpp` | +52 −4 |

**Zero SYCL files, zero Vulkan files, zero CPU files.** The title's "CUDA" scope is accurate. Two further
points from the diff:

1. The `qwen4exp.cpp` change in #27970 was purely mechanical — inserting the new `0` argument into the existing
   call:
   ```
   -    ggml_tensor * cur = build_attn_mha(q, k, v, nullptr, kq_mask_top_k, nullptr, nullptr, kq_scale, il);
   +    ggml_tensor * cur = build_attn_mha(q, k, v, nullptr, kq_mask_top_k, nullptr, nullptr, 0, kq_scale, il);
   ```
   #27970 did **not** write the `TODO`.
2. The `TODO` and the commented-out enabled line were added by a *later* commit,
   **`7bb0fc18f` "metal : add sparse FA (#28098)"** (Georgi Gerganov, 2026-09-03), found with
   `git log -S "TODO: enable sparse attention when we are ready"`. That PR's own commit list contains a commit
   literally titled `qwen4 : enable sparse attention`, yet the merged end state has the call **disabled** with
   the `TODO` above it. **Inference:** the qwen4exp enablement was attempted and then deliberately backed out
   before merge. That is a meaningful signal — the upstream maintainer who *wrote* the Metal sparse kernel chose
   not to ship it for this architecture — but the reason is not recorded in the tree, and this document does
   not speculate about it beyond noting it.
3. Metal's implementation landed in that separate PR (#28098), which is why `ggml-metal` appears in the grep
   above but not in #27970's file list.

### 3.3 PR #27970's own benchmark table, and why it doesn't transfer

From the PR body (`gh pr view 27970 --json body`), model `deepseek4 ?B IQ2_XXS`, "Performance of 2-bit DSV4
quant on DGX spark":

| Test | baseline t/s | sparse-fa t/s | speedup |
|---|---:|---:|---:|
| pp2048 | 563.31 | 563.53 | 1.00 |
| pp2048@d8192 | 502.37 | 520.61 | 1.04 |
| pp2048@d32768 | 389.50 | 461.63 | 1.19 |
| pp2048@d131072 | 204.63 | 315.26 | 1.54 |
| pp2048@d262144 | 122.99 | 216.94 | 1.76 |
| pp2048@d1048576 | 33.99 | 74.90 | 2.20 |
| tg32 | 19.00 | 19.07 | 1.00 |
| tg32@d8192 | 18.18 | 18.26 | 1.00 |
| tg32@d32768 | 17.01 | 17.52 | 1.03 |
| tg32@d131072 | 14.20 | 15.73 | 1.11 |
| tg32@d262144 | 11.68 | 13.94 | 1.19 |
| tg32@d1048576 | 5.31 | 8.30 | 1.56 |

Caveats, stated rather than assumed:

- Different model (`deepseek4`), different architecture family, different `indexer_top_k`/compress ratios,
  different hardware, and **the table's first column is labelled `CPU` for every row** while the PR implements
  CUDA only and no CPU-side `n_kv_max` handling exists in-tree (§3.1). That column label is unexplained; the
  device attribution of these rows should be treated as unverified. What is usable is the *shape* of the
  curve, not the absolute numbers.
- **`tg32` — decode, which is this project's actual target metric — is 1.00-1.03× out to `d32768`.** Even at
  `d131072` it is 1.11×. The 1.5-2.2× numbers are all prefill, at depths ≥ 131k.

### 3.4 What CUDA does with the hint — the gate that validates the crossover reasoning

`ggml/src/ggml-cuda/fattn.cu:108-129`:

```cpp
    const int32_t n_kv_max = ggml_get_op_params_i32(dst, 4);
    return GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc) &&
        mask != nullptr && n_kv_max > 0 && max_bias == 0.0f && logit_softcap == 0.0f &&
        mask->ne[0] == K->ne[1] && mask->ne[1] >= Q->ne[1] && mask->ne[2] == 1 &&
        K->ne[1] >= std::max<int64_t>(4096, 2LL*n_kv_max);
```

That last clause is the important one. Even on the backend that *has* the feature, sparse is refused unless
`n_kv ≥ max(4096, 2 × n_kv_max)`. For us that is `max(4096, 4102) = 4102`, so with 256-padded `n_kv`, CUDA
would take the sparse path only from **`n_kv = 4352`, i.e. ≥ 4 097 tokens of context** — roughly double our
naive crossover of 2 049. Upstream's own gate encodes the same conclusion this document reached from the
formula: below ~2× headroom, the index-compaction pass costs more than the skipped KV rows save.

When engaged, CUDA allocates `n_kv_max * mask_rows` int32s and runs a ballot/popcount compaction kernel
(`ggml/src/ggml-cuda/fattn-common.cuh:1095-1102`, kernel at `ggml/src/ggml-cuda/fattn.cu:10-88`), then
substitutes `n_kv = n_kv_max` for the tiling arithmetic (`ggml/src/ggml-cuda/fattn-common.cuh:1131`). Note the
sparse kernel support lives in `fattn-mma-f16.cuh` only — `fattn-tile.cuh` and `fattn-vec.cuh` got signature
changes only (+6/−6 and +1/−1). **Inference:** CUDA's sparse path exists exclusively on the MMA (tensor-core)
kernel, and CUDA does not do sparse decode on its tile/vec kernels at all.

Metal's gate (`ggml/src/ggml-metal/ggml-metal-ops.cpp:2863-2916`) is looser — it requires only
`n_kv_max > 0`, a mask, `n_kv_max ≤ 4096`, and a supported `(dk, dv)` pair (256/256 **is** in the list) — with
**no minimum `n_kv`**. **Inference:** on Metal, enabling the flag for qwen4exp at short context would engage
the index-compaction kernel with `n_kv_max == n_kv` (a "sparse" set covering 100% of the KV), paying the
compaction cost for zero skipped work. That is a plausible reason the qwen4 enablement was reverted in #28098,
though as noted the tree does not record one.

### 3.5 Answer to the safety question

**If `src/models/qwen4exp.cpp:946` were changed to pass `top_k->ne[0]`, on this project's SYCL/Arc stack:
nothing would happen. It is a safe no-op — not a crash, not silent corruption, and not a speedup.**

Chain of custody for that claim:

1. `top_k->ne[0] == width ≥ 1`, so `GGML_ASSERT(n_kv_max >= 0 && n_kv_max <= INT32_MAX)`
   (`src/llama-graph.cpp:2637`) and `GGML_ASSERT(n_kv_max >= 0)` (`ggml/src/ggml.c:5565`) both pass. No assert
   risk.
2. The API's correctness precondition — "`n_kv_max` must bound the number of finite entries in every mask row"
   (`ggml/include/ggml.h:2506`) — **is satisfied by construction.** `kq_mask_top_k`
   (`src/models/qwen4exp.cpp:914-937`) starts as all `-INF` and un-masks exactly the `width` positions named by
   `top_k`, then adds the causal mask (which can only add more `-INF`, never remove one). So finite entries per
   row ≤ `width` = the hint. The bound holds even if `ggml_top_k` returns duplicate indices.
3. Without `-fa`, the value is never even written (`src/llama-graph.cpp:2615-2616` gates the whole FA branch on
   `cparams.flash_attn && kq_b == nullptr`; the dense branch ignores `n_kv_max`).
4. With `-fa` on SYCL, the value is written to `op_params[4]` and then **never read** by any SYCL FA code path
   (§3.1) — not the oneDNN/MKL prefill path, not the vec/tile decode path — nor by `supports_op`. Attention is
   computed from the mask exactly as it is today.
5. Same for any FA node that lands on the CPU backend (§3.1).
6. Output is therefore **bit-identical**, because the mask was already carrying the sparsity (§2.5).

The one way to make it unsafe would be to hand it a hint that is *not* an upper bound on finite mask entries —
e.g. passing `hparams.indexer_top_k` (2048) instead of `top_k->ne[0]` (2051) — on a backend that implements
the feature. Don't do that. Pass `top_k->ne[0]`, which is what the commented-out line already says.

---

## 4. What SYCL support would cost, and what it would buy

This section is deliberately short, per the brief — the numbers in §1-3 do not justify the depth of
`docs/research/05`.

### 4.1 What it would buy — arithmetic

Decode attention cost on this exact card has a measured anchor: `docs/research/01-sycl-backend-feasibility.md:931`
records upstream issue #26581 (2026-09-04) as measuring decode attention on **Arc Pro B70 (BMG-G31, 608 GB/s)**
at **~21-25 ns per KV position per layer, identical on Vulkan and SYCL**. A second, project-local anchor is the
comment this session added at `ggml/src/ggml-sycl/fattn.cpp:288`: *"The FA kernel itself is ~19 µs/call there"*
— measured on Qwen3.8-Flash-Next at this project's standard short-prompt shape.

Those two are consistent and jointly informative (**inference**): at `n_kv = 256`, 23 ns/position would predict
~5.9 µs, against ~19 µs measured — i.e. at short context the FA kernel is launch/latency-floor-dominated, and
**linear extrapolation from the 19 µs figure would badly overstate long-context FA cost.** Use the marginal
slope instead.

Two bounds for per-token decode FA cost over all 12 QSA layers (**estimate**):

- *Bandwidth floor*: each KV position costs `2 kv_heads × 256 dim × 2 (K+V) × 2 B = 2048 B` per layer, so
  `12 × 2048 = 24 576 B` per position per token; at 608 GB/s that is **40.4 ns/position/token**.
- *Measured latency-bound slope* (#26581, different model geometry — flagged as a transfer, not a measurement
  of ours): `12 × 23 ns = 276 ns/position/token`.

| `n_kv` | dense FA / token (floor … #26581 slope) | sparse FA / token (capped at 2051) | saving |
|---:|---:|---:|---:|
| 2 304 | 0.09 – 0.64 ms | 0.08 – 0.57 ms | ~0.01 – 0.07 ms |
| 8 192 | 0.33 – 2.26 ms | 0.08 – 0.57 ms | 0.25 – 1.7 ms |
| 32 768 | 1.32 – 9.04 ms | 0.08 – 0.57 ms | 1.2 – 8.5 ms |
| 131 072 | 5.30 – 36.2 ms | 0.08 – 0.57 ms | 5.2 – 35.6 ms |
| 262 144 | 10.6 – 72.4 ms | 0.08 – 0.57 ms | 10.5 – 71.8 ms |

Against this project's current best decode of **26-29 tok/s ≈ 34.5-38.5 ms/token** (`PLAN.md:210-224`, MTP
config), that translates to roughly:

- **@8k**: ~1.01-1.05× — noise.
- **@32k**: ~1.03-1.28×.
- **@131k**: ~1.15-2.0× (but see §4.3 — the model plus a 3.6 GB KV cache plus `-ncmoe` placement on a 30 GB
  card is its own problem, and this project already OOMs at `-ncmoe 24`).

The wide bands are honest: the two anchors differ by 6.8×, and closing that gap requires a GPU measurement this
task is explicitly barred from making.

### 4.2 What it would cost — rough scope

To get any of the above, SYCL would need (mirroring CUDA's structure, none of which is copy-pasteable — SYCL
has no `__ballot_sync`; the equivalents are `sycl::ext::oneapi::group_ballot` /
`sycl::exclusive_scan_over_group`):

1. **A mask→index-list compaction kernel**, SYCL port of `flash_attn_mask_to_sparse_indices`
   (`ggml/src/ggml-cuda/fattn.cu:10-88`, ~80 lines). The scratch-buffer plumbing already exists in SYCL — the
   `KV_max` pool alloc and its launch shape at `ggml/src/ggml-sycl/fattn-common.hpp:931, 1017-1041` is the
   right pattern to copy.
2. **Sparse gather inside a kernel.** This is the real work and the real risk. The decode kernel selected for
   our shapes is **TILE** (traced through `ggml/src/ggml-sycl/fattn.cpp:245-273`: head dim 256 and f16 KV pass
   `can_use_vector_kernel`, `Q->ne[1] == 1` with `gqa_opt_applies` true falls through the `VEC` branch to
   `return BEST_FATTN_KERNEL_TILE`). `fattn-tile.hpp` is 1 246 lines, and CUDA never added sparse to *its* tile
   kernel — only to MMA (§3.4) — so there is no upstream reference implementation to mirror for the kernel we
   actually need. Its existing `KV_max` mechanism (`ggml/src/ggml-sycl/fattn-tile.hpp:863`,
   `ggml/src/ggml-sycl/fattn-common.hpp:619-672`) skips a *contiguous suffix* of all-masked KV blocks and is of
   no help here: QSA's top-k selection is scattered, not a suffix. (It is also gated on `Q->ne[1] >= 1024`
   — `ggml/src/ggml-sycl/fattn-common.hpp:1017` — so it never runs at decode anyway.)
3. **Prefill has no hook at all.** Prefill on this stack goes to oneDNN SDPA or MKL
   (`ggml/src/ggml-sycl/fattn.cpp:132-175`), both library GEMM calls with no place to inject an index gather.
   Sparse prefill would mean either physically gathering K/V rows into a compact buffer first, or abandoning
   the XMX-accelerated path for QSA layers — and the prefill numbers are where #27970's table shows the
   biggest wins, so a decode-only SYCL port captures the *smaller* half of the upside.
4. Selection-gate changes in `ggml/src/ggml-sycl/fattn.cpp`, mirroring CUDA's `≥ max(4096, 2·n_kv_max)` rule.

Validation would be cheap: #27970 and #28098 already added `FLASH_ATTN_EXT` sparse cases to
`tests/test-backend-ops.cpp` (+52 lines in #27970), so `test-backend-ops -o FLASH_ATTN_EXT` covers correctness
without any model run.

**Rough scope: ~400-700 lines of new/modified SYCL across 3-4 files, of which the tile-kernel gather is
genuinely novel kernel work with no upstream analogue.** Comparable in size to this project's SYCL MoE-cache
gather work, without that work's motivating measurement.

### 4.3 The blocker that fires first, and is bigger

**This is the most actionable finding in the document, and it is not about sparse FA.**

QSA calls `ggml_top_k(expanded, 2051)` per QSA layer per ubatch (`src/models/qwen4exp.cpp:863`). SYCL's
`ggml_sycl_op_top_k` (`ggml/src/ggml-sycl/ggml-sycl.cpp:3105-3157`) has a fast scan-merge path capped at
`SYCL_TOP_K_MAX_SCAN_MERGE_K = 32` (`ggml/src/ggml-sycl/ggml-sycl.cpp:2613-2614`), and for larger `k` — the
comment names our case explicitly, *"the qwen4exp lightning indexer asks for ~2048"* — falls back to a **full
argsort of the entire row**, keeping the leading `k`:

```cpp
    if (k > SYCL_TOP_K_MAX_SCAN_MERGE_K) {
        ggml_sycl_pool_alloc<int32_t> sorted_alloc(ctx.pool(), (size_t) nrows * (size_t) ncols);
        ...
        argsort_f32_i32_sycl(src0_dd, sorted, (int) ncols, (int) nrows, GGML_SORT_ORDER_DESC, ...);
```

(This local fallback is this project's own uncommitted work — `git diff` shows it as a local modification, and
`PLAN.md:117` records it as "TOP_K CPU-fallback fixed". It replaced a CPU round-trip, so it is strictly an
improvement; the point below is about how it scales, not about the fix.)

`argsort_f32_i32_sycl` (`ggml/src/ggml-sycl/ggml-sycl.cpp:2389-2486`) has two regimes, chosen by whether
`next_power_of_2(ncols) * 4 bytes` fits in `smpbo` (the device's `local_mem_size`,
`ggml/src/ggml-sycl/ggml-sycl.cpp:183`):

- **Fits** → one kernel, one workgroup per row, bitonic sort in SLM.
- **Doesn't fit** (`ggml/src/ggml-sycl/ggml-sycl.cpp:2397-2430`) → a global-memory multi-pass bitonic:
  `log2(ncols_pad) · (log2(ncols_pad)+1) / 2` **separate kernel launches**, plus an `nrows × ncols_pad` int32
  scratch allocation.

With `smpbo = 64 KiB` (the usual Battlemage per-workgroup local-memory limit — **not verified on this box,
since verifying it requires running on the GPU**), the threshold is `ncols_pad ≤ 16384`, i.e. `n_kv ≤ 16384`.
Above that (**estimate**, from the loop bounds at `ggml-sycl.cpp:2409-2410`):

| `n_kv` | `ncols_pad` | bitonic passes | kernel launches per token (× 12 QSA layers) | argsort scratch @ decode (`nrows=1`) | argsort scratch @ prefill (`n_ubatch=2048`) |
|---:|---:|---:|---:|---:|---:|
| 16 384 | 16 384 | SLM path | 12 | 64 KiB | 128 MiB |
| 32 768 | 32 768 | 120 | **1 440** | 128 KiB | 256 MiB |
| 131 072 | 131 072 | 153 | **1 836** | 512 KiB | 1 GiB |
| 262 144 | 262 144 | 171 | **2 052** | 1 MiB | 2 GiB |

At a few microseconds of dispatch overhead each, ~1 440-2 052 extra kernel launches **per decoded token** is
on the order of several to tens of milliseconds — the same magnitude as, or larger than, the entire sparse-FA
saving in §4.1, and it is paid **whether or not the sparse flag is enabled**, because the `ggml_top_k` node
exists either way (§2.5). The prefill scratch column is worse and independently corroborates
`PLAN.md:368-374`'s note (from the Arc Pro B65 report) that QSA top-k scratch scales with allocated `-c` and
that overshooting VRAM degrades *silently* via driver-level PCIe spill rather than erroring.

**Inference, and the practical conclusion of this section:** on SYCL, long-context QSA is bottlenecked by the
top-k, not by dense attention. Implementing sparse FA before fixing the top-k would optimize the smaller of
two costs at the depth where both appear. `PLAN.md:336-340`'s option 3 — port a radix-select `top_k` to SYCL
using Vulkan's `topk_radix_select.comp` as reference — is the higher-value piece of work of the two, and it is
also the one `PLAN.md` already calls "the highest-value single upstream contribution this project could make".

---

## 5. Verdict

**Not worth pursuing now. Leave `src/models/qwen4exp.cpp:943-946` exactly as it is.**

Reasons, in priority order:

1. **The crossover is 2 049 tokens and the useful regime starts around 16-32k.** This project has never
   benchmarked past a ~10-token prompt with `-n 300` — `n_kv = 512` at most, where `width == n_kv` and the
   sparse mask excludes literally nothing. At the depths we actually run, sparse FA has exactly zero upside,
   and upstream's own CUDA gate (`≥ max(4096, 2·n_kv_max)`) refuses to engage there for the same reason.
2. **The stated goal is decode throughput, and decode is where #27970's numbers are weakest** — `tg32` at
   1.00-1.03× out to `d32768` on the architecture it was written for.
3. **SYCL has no implementation, and the kernel we would need has no upstream analogue.** CUDA implemented
   sparse only in its MMA kernel; our decode path is TILE. Prefill (where the wins actually are) runs on
   oneDNN/MKL library GEMMs with nowhere to inject a gather. ~400-700 lines of novel SYCL kernel work for a
   1.0-1.3× decode win in a depth regime we do not run.
4. **A bigger SYCL problem fires first at the same depths** (§4.3): QSA's `ggml_top_k(2051)` degrades into a
   ~120-170-pass global bitonic argsort above `n_kv ≈ 16384`, costing ~1 440-2 052 kernel launches per token at
   32k, paid unconditionally. Fixing that is strictly a prerequisite, strictly higher value, and already on
   `PLAN.md`'s list.

**Conditions under which to revisit**, concretely:

- This project adopts a long-context decode target (≥ 16k, realistically ≥ 32k) as an actual goal rather than
  an untested regime; **and**
- a SYCL radix-select `top_k` has landed (`PLAN.md:336-340`), removing the argsort blocker; **and**
- a measured decode profile at that depth shows `FLASH_ATTN_EXT` is a double-digit percentage of per-token
  time (measurable with the existing `GGML_SYCL_OP_PROFILE`/`GGML_SYCL_OP_PROFILE_WINDOW` instrumentation at
  `ggml/src/ggml-sycl/ggml-sycl.cpp:6569-6586`, and with `test-backend-ops -o FLASH_ATTN_EXT perf` at the real
  shapes — **not** with `GGML_SYCL_FA_SKIP`, for the router-collapse reason documented at
  `ggml/src/ggml-sycl/fattn.cpp:279-293`).

**One correction to fold back into `PLAN.md` regardless of the above.** `PLAN.md:321-324` currently says QSA
"provides no speed benefit on *any* backend today anyway … the sparse compute path is TODO'd out upstream;
only the mask is sparse". The second half is still true **for us**; the first half is now out of date. The
sparse compute path is implemented and merged for **CUDA** (PR #27970, merged 2026-09-02) and **Metal**
(PR #28098, merged 2026-09-03). It is TODO'd out *for the `qwen4exp` architecture specifically*, in the model
graph, on every backend — and that architecture-level `TODO` was written by the Metal PR's own author after a
commit in that same PR had enabled it, which is a signal worth recording even though the tree does not say
why.

**One thing that is genuinely cheap and worth knowing**, if it ever becomes relevant: because the top-k mask
is already applied today (§2.5), flipping line 946 changes model output by exactly nothing on our stack. So
the change is risk-free to carry as a local patch if a future long-context experiment wants it armed for the
day SYCL support exists. It just buys nothing until then, so there is no reason to carry it now.
