# QSA Sparse Attention on SYCL — Scoping, Measurement, and Verdict

**Status: scoped and measured on real hardware. No kernel was written. Verdict: do not ship a sparse-FA
kernel in this pass — `FLASH_ATTN_EXT` is a measured 35 of the 240 ms in a 116K decode token, so the whole
available prize is ~1.16x, and the larger co-located cost (the QSA indexer chain) is one sparse FA provably
cannot touch.**

Date: 2026-09-11 (late). Supersedes `docs/research/07-qsa-sparse-attention-scoping.md` §4-§5 on the *cost*
side (its §1-§3 arithmetic and safety analysis still stand, re-verified below); does not supersede its
verdict so much as replace the reasoning under it with measurements.

This document exists because `docs/research/07` §5 named three conditions for revisiting sparse FA, two of
which have since been met (long-context decode became a real goal; the radix top-k landed) and one of which
had never been checked: *"a measured decode profile at that depth shows `FLASH_ATTN_EXT` is a double-digit
percentage of per-token time."* That third condition is what this pass actually went and measured.

## Sources read directly

| Source | How obtained |
|---|---|
| Upstream PR **#28770** "CUDA: enable sparse fa for qwen4" (OPEN, created 2026-09-11T16:41:11Z) — body, file list, full diff | `gh pr view 28770 --repo ggml-org/llama.cpp --json body,files,title,state,createdAt`; `gh pr diff 28770 --repo ggml-org/llama.cpp` |
| Our tree: `ggml/src/ggml-sycl/{fattn.cpp,fattn-common.hpp,fattn-tile.hpp,fattn-onednn.cpp,fattn-onednn.hpp,fattn-buffers.hpp}`, `src/models/qwen4exp.cpp`, `tests/test-backend-ops.cpp`, `ggml/src/ggml-cuda/fattn-mma-f16.cuh` | direct reads, line ranges cited inline |
| `FLASH_ATTN_EXT` cost at the real production decode shapes, including `q4_0` KV | `test-backend-ops perf -b SYCL0 -o FLASH_ATTN_EXT`, new cases added for this pass — `logs/qsa-scoping/fa-perf.log` |
| Decode throughput vs. KV depth for the real production config | `llama-bench -d 0,2048,8192,32768,65536` — `logs/qsa-scoping/bench-depth.log` |

All repo paths are relative to
`/media/da3dsoul/Golias/AIProjects/Qwen3.8-Flash-Next-LlamaCpp-MoE-Cache-Arc-Experiments/src/llama.cpp/`
unless stated otherwise. Tree state: `HEAD = 6d9c82ea2`, working tree dirty (this project's uncommitted
SYCL MoE-cache / MTP / top-k work).

Statements that join facts established in separate places are flagged **inference**. Arithmetic that
extrapolates a measured datapoint is flagged **estimate** and shows its assumptions.

---

## 0. Verdict in one paragraph

At the real production decode shape (`hsk = hsv = 256`, 2 KV heads, `gqa_ratio = 12`, `Q->ne[1] = 1`,
`q4_0` K and V) the SYCL `FLASH_ATTN_EXT` op costs **2.93 ms per call at `n_kv = 118016`**, i.e. **~35 ms
per decode token** over the model's 12 full-attention layers. A working sparse path would cut that to
roughly **1.2 ms/token** (the `n_kv ≈ 2304` cost, §2.2), saving ~34 ms/token. Against the 116K server
benchmark's measured **240 ms/token (4.17 tok/s)** that is a **~1.16x** decode win — real, but not a fix.
Two further things came out of the measurement and both argue against building now. First, **a 3.1x
discrepancy between our own numbers that is not attention**: `llama-bench` on the same config measures
**23.54 tok/s at `d2048` and 22.93 at `d8192`**, and the isolated FA cost accounts for essentially the whole
gap between those two, which extrapolates the clean-environment 116K figure to **~13 tok/s, not 4.17**
(§3.3). Second, **the larger depth-proportional term is almost certainly the QSA indexer chain, and sparse FA
cannot remove one microsecond of it** — ~20 ops per QSA layer on `[128, 29504]`/`[118016]` tensors, ~240
kernel launches per decode token, built unconditionally whether or not the sparse hint is set, and never
measured (§3.4). **So: profile a decode window by call site at depth (the instrumentation and the `cb()`
names already exist) before writing 360-470 lines of correctness-critical SYCL gather kernel for ~1.16x.**
Sparse FA stays a genuine ~1.78x decode lever *if* the 3.1x gap turns out to be environmental, and §4 scopes
two designs for when it is next picked up — including one (§4.2) that needs **zero edits to the FA kernel
itself**, which the SYCL backend's existing dense-f16 staging pass for quantized KV makes possible and which
CUDA has no analogue for.

---

## 1. What is actually new since `docs/research/07`

### 1.1 PR #28770 exists, is open, and is CUDA-only

`gh pr view 28770 --repo ggml-org/llama.cpp --json files` (state `OPEN`, created 2026-09-11T16:41:11Z):

| file | +/− |
|---|---|
| `ggml/src/ggml-cuda/fattn-common.cuh` | +10 −6 |
| `ggml/src/ggml-cuda/fattn-mma-f16.cuh` | +19 −8 |
| `ggml/src/ggml-cuda/fattn.cu` | +33 −17 |
| `src/models/qwen4exp.cpp` | +1 −4 |
| `tests/test-backend-ops.cpp` | +5 −0 |

Zero SYCL, zero Vulkan, zero CPU files — same as #27970 before it. Its own benchmark table (PR body) is a
DGX-Spark run on `qwen4exp A3B IQ1_S`:

| Test | baseline t/s | sparse t/s | speedup |
|---|---:|---:|---:|
| pp2048@d10000 | 615.70 | 663.17 | 1.08 |
| pp2048@d50000 | 405.18 | 469.03 | 1.16 |
| pp2048@d100000 | 252.81 | 318.28 | 1.26 |
| tg32@d10000 | 22.98 | 23.56 | 1.03 |
| tg32@d20000 | 20.76 | 23.58 | 1.14 |
| tg32@d50000 | 15.88 | 18.72 | 1.18 |
| tg32@d100000 | 12.00 | 14.19 | 1.18 |

**Upstream's own decode claim for this exact architecture is 1.18x at d100000.** That is the number to
hold any SYCL port to, not the 1.5-2.2x figures in #27970's prefill rows. Note also their baseline decode at
d100000 is 12.00 tok/s on a fully-resident model — the same ballpark as this project's *extrapolated clean*
number (§3.3), and 3x our measured server number.

### 1.2 The `qwen4exp.cpp` hunk is unconditional — the gate is entirely backend-side

The model-side change in #28770 is the whole of it (`src/models/qwen4exp.cpp:943-946` in our tree):

```diff
-    // TODO: enable sparse attention when we are ready
-    // ref: https://github.com/ggml-org/llama.cpp/pull/27970
-    //ggml_tensor * cur = build_attn_mha(q, k, v, nullptr, kq_mask_top_k, nullptr, nullptr, top_k->ne[0], kq_scale, il);
-    ggml_tensor * cur = build_attn_mha(q, k, v, nullptr, kq_mask_top_k, nullptr, nullptr, 0, kq_scale, il);
+    ggml_tensor * cur = build_attn_mha(q, k, v, nullptr, kq_mask_top_k, nullptr, nullptr, top_k->ne[0], kq_scale, il);
```

There is **no backend-capability check in the model graph** — the hint is written into `op_params[4]` of the
`FLASH_ATTN_EXT` node unconditionally, and every backend decides for itself whether to honour it. That is
the design `docs/research/07` §2.4-§3.5 described and it is still correct: re-verified for this pass by
whole-tree grep, `n_kv_max` appears under `ggml/src/` only in `ggml.c` (the setter), `ggml-cuda/fattn.cu`,
`ggml-cuda/fattn-common.cuh`, `ggml-metal/{ggml-metal-impl.h,ggml-metal-ops.cpp,kernels/fa.metal}` — and
every one of SYCL's fourteen FA `op_params` accesses touches only float slots 0/1/2
(`ggml-sycl/fattn-mkl.cpp:371-373`, `fattn-onednn.cpp:96-97,251`, `fattn.cpp:125,128`,
`fattn-common.hpp:1112-1114`, `fattn-vec.hpp:620`, `fattn-tile.hpp:1160,1219`).

**So flipping line 946 today remains a provable no-op on SYCL and CPU: same output, same speed.** This pass
deliberately left it as-is, for the reason `docs/research/07` §5 gave — armed-but-inert is indistinguishable
from unchanged, and carrying a dead diff makes the next reader wonder whether it does something.

### 1.3 What the CUDA diff tells us about the real kernel work

Two things, and they matter for the SYCL design in §4.

**First, #28770 is about query-tile *grouping*, not about the gather itself.** The gather already landed in
#27970 and lives only in the MMA kernel (`ggml/src/ggml-cuda/fattn-mma-f16.cuh:370-523`, where
`flash_attn_ext_f16_load_tile` / the mask loader take `const int32_t * indices` and read row
`indices[k_VKQ_0 + i]` instead of row `k_VKQ_0 + i`). What #28770 adds is: one index list per *group of
`ncols1` queries*, holding the union of that group's visible columns, plus a per-group live count:

```
n_kv_max = std::min<int64_t>(K->ne[1], int64_t(ncols1)*n_kv_max_query);
...
KV_max.alloc(size_t(n_kv_max)*n_lists + n_lists);
ggml_cuda_flash_attn_ext_compact_mask(mask, KV_max.ptr, KV_max.ptr + size_t(n_kv_max)*n_lists, Q->ne[1], ncols1, n_kv_max, main_stream);
```

and in the kernel, `kb0_stop` is clamped by that count so the KV loop stops after the gathered columns:

```
kb0_stop = min(kb0_stop, (KV_max[(sequence % ne33)*iter_j + jt] + nbatch_fa - 1) / nbatch_fa);
```

**This is the distinction the brief asked about, and the answer is unambiguous: the saving comes from not
iterating over the skipped KV blocks at all.** The mask is never used to zero scores after a full pass — the
loop bound itself shrinks from `n_kv/nbatch_fa` to `count/nbatch_fa`, and the rows that *are* visited are
fetched by indirection. Any SYCL design that only skips masked contributions post-hoc would be a
correctness-preserving no-op with full cost, and is not worth writing.

**Second, the gate moved and got tighter per query, looser per tile** (`ggml-cuda/fattn.cu:131-140`):

```
const int64_t n_gather = (ncols1 == 1 ? Q->ne[1] : ncols1) * (int64_t) n_kv_max;
return ... && K->ne[1] >= std::max<int64_t>(4096, 2*n_gather);
```

For our model (`n_kv_max = 2051`) at decode, `ncols1 == 1` and `Q->ne[1] == 1`, so `n_gather = 2051` and
the gate opens at `K->ne[1] >= 4102`, i.e. `n_kv >= 4352` with 256-padding — the same threshold
`docs/research/07` §3.4 derived. At prefill `ncols1 == 8`, `n_gather = 16408`, and the gate needs
`n_kv >= 32816` — which is exactly the "For Qwen4 this value is 32768 ctx" line in the PR body.

**Third, and load-bearing for §4: CUDA's sparse decode path is MMA-only, and #28770 has to actively steer
decode *away* from its vector kernel to reach it** (`ggml-cuda/fattn.cu:606-616`, the `!sparse_decode`
clause added to the `BEST_FATTN_KERNEL_VEC` condition). The SYCL analogue of MMA does not exist; our decode
kernel is TILE (§2.1). So there is still no upstream reference implementation for the kernel we would
actually have to modify — `docs/research/07` §4.2 point 2 is unchanged by #28770.

### 1.4 Test coverage upstream added, and what it gives us for free

#28770's `tests/test-backend-ops.cpp` hunk is three eval cases at our architecture's shape:

```cpp
    // sparse attn (qwen4 shape - gqa 12)
    test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {12, 1}, 4096,  1, ..., 512));
    test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {12, 1}, 8192, 64, ..., 512));
    test_cases.emplace_back(new test_flash_attn_ext(256, 256, 1, {12, 2}, 8192, 67, ..., 512));
```

The trailing `512` is `n_kv_max`. This matters more than it looks: `test_flash_attn_ext::initialize_tensors`
(`tests/test-backend-ops.cpp:7694-7707`) switches the mask initializer to `init_tensor_kq_mask_sparse(t,
n_kv_max)` when `n_kv_max > 0`, and the CPU reference backend ignores `op_params[4]` entirely (§1.2). **So
these cases are a genuine equivalence proof, not a smoke test**: a sparse implementation must reproduce the
dense CPU result from a genuinely sparse mask, and any column the compaction kernel drops shows up
immediately as NMSE error. That is the right guard for the "silent corruption" risk class, and it would be
available from day one of a SYCL port.

---

## 2. What decode actually costs on this stack, measured

### 2.1 Which kernel decode uses, and the hidden `q4_0` tax

Traced through `ggml_sycl_get_best_fattn_kernel` (`ggml/src/ggml-sycl/fattn.cpp:106-274`) for the production
decode shape (`Q->ne[0] = 256`, `Q->ne[1] = 1`, `gqa_ratio = 12`, K/V `q4_0`, mask present, no sinks):

- the first oneDNN gate (`fattn.cpp:135-139`) needs `Q->ne[1] >= g_ggml_sycl_fa_onednn_min_q` — fails at decode;
- the MKL gate (`fattn.cpp:155-176`) needs `Q->ne[1] >= 32` — fails;
- the second, "fused-XMX" oneDNN gate (`fattn.cpp:250-252`) calls
  `ggml_sycl_flash_attn_ext_onednn_supported(dst)` with `use_shape_limit` defaulted to `true`
  (`fattn-onednn.hpp:8`), which returns false for `Q->ne[1] < GGML_SYCL_FA_ONEDNN_MIN_Q`
  (`fattn-onednn.cpp:110-113`) — fails;
- K is quantized, so the `Q->ne[1] <= 2` branch at `fattn.cpp:263-270` fires and, because the device arch is
  `intel_gpu_bmg_g31`, returns `BEST_FATTN_KERNEL_TILE`.

**Decode is TILE, at every depth.** (`GGML_SYCL_MKL_FA_DEBUG=1` prints the selected kernel per FA call at
`fattn.cpp:296-326` if this ever needs re-confirming against a live run.) Prefill at `-ub 2048` is oneDNN or
MKL, both library GEMMs.

The TILE kernel is written against `sycl::half2` K/V pointers (`fattn-tile.hpp:661-697, 730-737`) and is
launched with `need_f16_K = need_f16_V = true` (`fattn-tile.hpp:1087-1089` and its siblings). So with
`-ctk q4_0 -ctv q4_0`, `launch_fattn` **dequantizes the entire KV cache into a dense f16 staging buffer on
every single FA call** (`fattn-common.hpp:945-1007`; the buffer is the persistent
`ggml_sycl_fattn_kv_buffers` of `fattn-buffers.hpp:20-47`, so it is reused but still fully rewritten). At
`n_kv = 118016` that is `118016 x 256 x 2 heads x 2 B = 121 MiB` written per tensor per layer per token, K
and V — **242 MiB of pure conversion traffic per layer per decode token, 2.8 GiB per token over 12 layers.**
This cost is inside `FLASH_ATTN_EXT` and was not accounted for anywhere in `docs/research/07`.

### 2.2 `FLASH_ATTN_EXT` cost at the real shapes

New perf cases were added to `tests/test-backend-ops.cpp` for this pass (the existing qwen4exp loop stopped
at `kv = 8192` and only covered f16 KV), covering the production depths in both KV types. Measured on an
idle GPU, `test-backend-ops perf -b SYCL0 -o FLASH_ATTN_EXT`, full log `logs/qsa-scoping/fa-perf.log`:

| `n_kv` | `nb` | KV type | us/run | 12 layers |
|---:|---:|---|---:|---:|
| 2 048 | 1 | f16 | 37.90 | 0.45 ms |
| 4 096 | 1 | f16 | 67.45 | 0.81 ms |
| 8 192 | 1 | f16 | 146.97 | 1.76 ms |
| 16 384 | 1 | f16 | 266.10 | 3.19 ms |
| 16 384 | 1 | **q4_0** | **394.73** | **4.74 ms** |
| 32 768 | 1 | f16 | 513.08 | 6.16 ms |
| 32 768 | 1 | **q4_0** | **807.09** | **9.69 ms** |
| 65 536 | 1 | f16 | 1 012.17 | 12.15 ms |
| 65 536 | 1 | **q4_0** | **1 615.03** | **19.38 ms** |
| 118 016 | 1 | f16 | 1 787.36 | 21.45 ms |
| **118 016** | 1 | **q4_0** | **2 927.28** | **35.13 ms** |

(`118016 = GGML_PAD(116277, 256)`, the `n_kv` of the 116K benchmark prompt.)

**Caveat, stated rather than buried: these are isolated-op timings, used as a proxy for in-model cost.** They
run the op back to back with nothing else competing for cache or bandwidth, so in a real graph the same call
could be faster (overlap) or slower (cache pollution from the surrounding 48 layers). The cross-check that
makes the proxy usable is §3.2: at the one depth pair where a whole-model measurement exists, the isolated
numbers reproduce the model's depth slope. The test does mirror the model's tensor layout — `kv_view = true`
builds K and V as views of a 2x-sized buffer, so `ggml_is_contiguously_allocated(K)` is false and
`launch_fattn` takes the same `to_fp16_nc_sycl` branch (`fattn-common.hpp:958-969`) a real KV cache does.

Three readings:

1. **Cost is linear in `n_kv` above ~4k** (2.93 ms at 118016 vs. 807 us at 32768 — 3.63x the cost for
   3.60x the depth). Below 2k it is launch-latency-floored, exactly as `docs/research/07` §4.1 predicted.
2. **The `q4_0` KV cache costs a flat ~1.6x on the FA op** (1.48x at 16384, 1.57x at 32768, 1.60x at 65536,
   1.64x at 118016) — that is the dequant staging pass of §2.1, and it is **13.7 ms/token at 116K** on its
   own. Not actionable (VRAM is the binding constraint and `q4_0` KV is what makes 116K fit at all) but
   worth knowing it exists.
3. **A sparse path would land at the `width`-row cost, not zero.** `width = 2051`, padded up to 2304 for
   `FATTN_KQ_STRIDE = 256` (`fattn-common.hpp:18`); the f16 2048-row case is 37.90 us, so with the `q4_0`
   1.6x and the added compaction/gather passes, **estimate ~100-150 us/call, ~1.2-1.8 ms/token** — against
   35.13 ms/token dense. **Saving ~33-34 ms per decode token at 116K.**

---

## 3. Where the other ~200 ms of a 116K decode token goes

### 3.1 The depth sweep

`llama-bench -m <UD-IQ3_XXS> -ngl 99 -fa on -ctk q4_0 -ctv q4_0 -ncmoe 24 -ub 2048 -p 0 -n 32 -d ... -r 1`,
idle GPU, full log `logs/qsa-scoping/bench-depth.log`:

| test | t/s | ms/token | usable? |
|---|---:|---:|---|
| `tg32` (d0) | 1.30 | 769 | **no** — cold page cache, the confound `PLAN.md` documents; discarded as the leading arm exactly as the `-ub` sweep's first arm was |
| `tg32 @ d2048` | **23.54** | **42.48** | yes (warm) |
| `tg32 @ d8192` | **22.93** | **43.62** | yes (warm) |
| `tg32 @ d32768` | — | — | **not obtained** |
| `tg32 @ d65536` | — | — | **not obtained** |

**The last two arms were abandoned, deliberately, and the reason is itself a finding.** After 16 minutes the
process sat in uninterruptible sleep (`ps -o stat`: `Dsl`) with its CPU time frozen at `00:05:40` and the box
at **88-91% iowait**, reading 25-40 MB/s continuously. The cause was external: `unbooru-tagger` (26 TB read
lifetime) and `deluged` (2.2 TB) were saturating the array the model file lives on (`md0` 12 MB/s,
`nvme1n1` 19.5 MB/s concurrent with the run), while `free` showed **swap fully exhausted (9/9 GiB)** and only
~9 GiB free against this config's **52.8 GiB of `CPU_Mapped` expert tensors**. Any number produced in that
state would measure the disk, not the GPU. The run was stopped rather than reported. **Standing note for the
next session: check `vmstat`/`iostat` and `ps -eo stat -C llama-*` for `D` state before trusting any
long-context throughput number on this box — GPU idleness (`ps aux | grep llama`) is necessary but not
sufficient.**

### 3.2 FA plausibly accounts for all of the measured depth slope in the range that was measurable

Two independent estimates of the `d2048 → d8192` decode cost increase:

| | ms/token |
|---|---:|
| measured, `llama-bench` (§3.1) | **1.14** |
| predicted from isolated FA cost (§2.2), f16 rows | `(146.97 − 37.90) us x 12 = ` **1.31** |
| predicted from isolated FA cost, with the `q4_0` staging term | ~1.7 |

At `n = 1` repetition on a contended box the measured and f16-predicted figures are indistinguishable, and
the `q4_0`-adjusted prediction overshoots — the honest reading is **FA accounts for all of the depth slope in
this range, to within single-run noise, and nothing else in the graph adds measurably between 2k and 8k.**
The arms that would have tested whether that still holds at 32k and 64k are the ones lost to §3.1.

### 3.3 Two numbers that cannot both be right, and what follows either way

Extrapolating the §3.2 model to the 116K prompt's depth (**estimate**; assumption: FA is the *only*
depth-proportional term, which §3.2 supports at 2-8k and which §3.4 argues is probably false above that):

```
decode @ n_kv 118016  =  42.48 ms (measured @ d2048)
                       -  0.68 ms (FA @ d2048, 12 layers)
                       + 35.13 ms (FA @ 118016, 12 layers, q4_0, MEASURED)
                       = 76.9 ms/token  ->  13.0 tok/s
```

**The 116K server benchmark measured 4.17 tok/s — 240 ms/token, 3.1x slower.** There is ~163 ms/token in the
real run that this model does not explain and that §2.2 proves is not the FA op.

Three candidate causes were considered, and the first is the only one this pass can rule *out*:

1. **VRAM pressure / silent PCIe spill — ruled out as the primary cause.** The `-ub 2048 -c 122880 -ncmoe 24`
   run does sit at 31,835 MiB against a ceiling `PLAN.md` brackets at 31,835-32,101 MiB (≤266 MiB of
   headroom), which is the regime `PLAN.md` documents as degrading silently. **But decode measured 5.34,
   4.22 and 4.17 tok/s across three configs with very different VRAM pressure** (`-ub 512 -c 163840`,
   ~29.7 GiB total, is comfortable) — so the 4-5 tok/s band is not a function of VRAM headroom.
2. **Host page-cache residency of the 52.8 GiB CPU-resident expert set.** Directly observed dominating this
   pass (§3.1) and already on record in `PLAN.md` as worth 24% on prefill. Every decode token touches a
   randomly-selected 10-of-512 expert set in each of 24 CPU-resident layers, so the effective working set is
   the whole 52.8 GiB; if it is not cache-resident, a token pays disk. At the 26 MB/s this box was serving,
   240 ms/token is ~6 MB of faulting — entirely plausible.
3. **The QSA indexer chain, which sparse FA does not touch** — §3.4.

**Either way the conclusion for this task is the same**, and it is the point of this document: the sparse-FA
win is `35.13 - ~1.5 = ~33.6` ms/token, which is **1.78x against the 76.9 ms/token clean-environment model
and 1.16x against the 240 ms/token that was actually measured.** Spending 360-470 lines of
correctness-critical kernel work to collect 1.16x, while a 3.1x discrepancy sits unexplained in the same
measurement, is the wrong order of operations.

### 3.4 The part of long-context decode that sparse FA cannot fix, and that nobody has measured

`docs/research/07` §2.5 established that the indexer + top-k + mask machinery is built **unconditionally** on
every QSA layer, today, regardless of line 946 — so enabling sparse FA neither adds nor removes any of it.
What §07 did not do is count how much of that machinery is `n_kv`-proportional. Reading
`build_qsa_top_k` (`src/models/qwen4exp.cpp:721-870`) at decode (`n_tokens = 1`, `n_stream = 1`,
`n_kv = 118016`, `r = 4`, so `n_blocks = 29504`, `idx_dim = 128`):

| op | site | shape at 118016 |
|---|---|---|
| `ggml_get_rows(k_all, blk_cells)` | `:789` | gathers **118 016 rows** of 128 |
| 4x `ggml_cont(ggml_view_3d(...))` | `:794-799` | four strided copies of `[128, 29504]` |
| 3x `ggml_add` + `ggml_scale` | `:798-800` | `[128, 29504]` |
| `build_norm` (RMS) | `:805` | `[128, 29504]` |
| `ggml_rope_multi` | `:809-811` | `[128, 1, 29504]` |
| `ggml_mul_mat(pooled, q)` | `:825` | `[128, 29504] x [128, 4]` |
| `ggml_relu`, 4x head add | `:828-836` | `[29504, 4]` |
| `ggml_cont(ggml_permute)` x2, `ggml_get_rows` | `:847-849` | `[29504] -> [118016]` |
| `ggml_cast(kq_mask)` + `ggml_add` | `:853-854` | `[118016]` |
| `ggml_top_k(expanded, 2051)` | `:863` | `[118016] -> [2051]` |
| `ggml_fill(-INF)`, `ggml_set_rows`, `ggml_add` | `:914-937` | `[118016]` f16 |

That is **~20 ops per QSA layer, x 12 layers = ~240 kernel launches per decode token**, most of them on
`[128, 29504]` or `[118016]` tensors, several of them strided (the four `ggml_cont` slices walk with stride
`r = 4`) and one of them a 118 016-row `get_rows`. **Estimate**, at 608 GB/s and generous assumptions, the
pure traffic is 3-6 GB/token → 5-10 ms; at the 10-20% of peak that poorly-coalesced strided copies and
tiny-row gathers typically achieve on this card, the same traffic is **25-100 ms/token**. That bracket
straddles the missing 163 ms and it is *unconditional* — sparse FA cannot remove one microsecond of it.

**This is the single cheapest next measurement and it was not possible today.** The instrumentation already
exists and already buckets by graph call site (`ggml/src/ggml-sycl/ggml-sycl.cpp:6506-6524`,
`sycl_op_prof_site_name`), and `build_qsa_top_k` names its tensors via `cb()` — `indexer_k_raw`,
`indexer_k_pooled`, `indexer_k`, `indexer_q`, `indexer_score`, `indexer_score_tokens`, `indexer_top_k`
(`qwen4exp.cpp:780, 801, 813, 821, 839, 858, 867`). So
`GGML_SYCL_OP_PROFILE=1 GGML_SYCL_OP_PROFILE_WINDOW=<small>` on a decode window at 32k depth prints the
indexer's share by name, next to `FLASH_ATTN_EXT`, in one run. Do that before writing any kernel.

---

## 4. Scope, if and when this is picked up

### 4.1 Design A — port CUDA's structure into the TILE kernel

Mirror #27970/#28770: compact the mask into index lists, then make the kernel's K/V/mask row lookups
indirect.

| file | work | LOC |
|---|---|---:|
| new `ggml-sycl/fattn-sparse.{cpp,hpp}` | SYCL port of `flash_attn_mask_to_sparse_indices` (`ggml-cuda/fattn.cu:10-99`). No `__ballot_sync`; use `sycl::ext::oneapi::group_ballot` + `sycl::exclusive_scan_over_group` for the per-sub-group compaction and an SLM running offset across sub-groups. `flash_attn_mask_to_KV_max` (`fattn-common.hpp:618-673`) is the right launch-shape and SLM-reduction pattern to copy. | ~110 |
| `ggml-sycl/fattn-tile.hpp` | add `bool use_sparse` template arg + `const int32_t * indices` kernel arg; thread through `flash_attn_tile` (`:661-1069`) → `flash_attn_tile_iter` (`:414-655`) → `flash_attn_tile_iter_KQ` (`:325-395`) → **both** `flash_attn_tile_load_tile` overloads (`:193-312`), replacing `KV + i*stride_KV` with `KV + indices[k_VKQ_0+i]*stride_KV`; the mask read at `:494`; the V tile load at `:583-584`; and the loop bound at `:863-893` so `k_VKQ_max` becomes the compacted count. | ~180-250 |
| `ggml-sycl/fattn-common.hpp` | sparse branch in `launch_fattn` (`:895-1185`): allocate `n_kv_max*n_lists + n_lists` int32s, launch the compaction, pass the pointer down. CUDA overloads its existing `KV_max` argument for this; SYCL's TILE kernel reads `KV_max` at `fattn-tile.hpp:863`, so the same trick applies. | ~60 |
| `ggml-sycl/fattn.cpp` | `shall_use_sparse(dst)` gate mirroring `ggml-cuda/fattn.cu:109-134` as amended by #28770 — `n_kv_max > 0`, mask present, `max_bias == 0`, `logit_softcap == 0`, `mask->ne[0] == K->ne[1]`, `K->ne[1] >= max(4096, 2*n_gather)`. | ~40 |
| `tests/test-backend-ops.cpp` | adopt #28770's three qwen4 cases + `q4_0` variants at our depths | ~10 |

**Risk: HIGH.** Three specific hazards. (i) `flash_attn_tile_load_tile`'s whole index algebra
(`:202-248`) is built around row `i` being at a fixed stride, and its `oob_check` compares `i < i_sup`;
indirection has to preserve the 16-byte `ggml_sycl_memcpy_1<cpy_nb>` vector copies (rows stay contiguous, so
it can) while every bound check switches from a depth comparison to a count comparison. Getting that wrong
drops or duplicates KV rows and the failure mode is a *quietly* wrong softmax, not a crash. (ii) A
`use_sparse` template argument **doubles the TILE instantiation count**, and those instantiations are
committed source, not generated at build time — `ggml/src/ggml-sycl/template-instances/` holds ten
hand-maintained `fattn-tile-instance-dkq*-dv*.cpp` files (of 46 total), each expanded again by
`launch_fattn_tile_switch_ncols1`/`_ncols2`'s `ncols1 x ncols2` matrix (`fattn-tile.hpp:1071-1212`). This is
already the slowest-compiling part of the SYCL backend. (One mitigation worth considering if Design A is
ever taken: instantiate sparse only for `DKQ == DV == 256`, the sole shape this model needs, exactly as
CUDA's `ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse` whitelists four `(DKQ, DV, ncols1, ncols2)` tuples
rather than all of them.)
(iii) **It leaves ~40% of the win on the table for a `q4_0` cache**: `launch_fattn`'s f16 staging pass
(`:945-1007`) still converts the *whole* cache, so Design A recovers the 21.45 ms/token f16 term but not the
13.7 ms/token dequant term (§2.2).

### 4.2 Design B — gather the sparse rows into the staging buffer the TILE path already builds

This is the design this document recommends, and it exists only because of the §2.1 finding. **The SYCL TILE
path already materializes a dense f16 copy of K and V for a quantized cache.** The cheapest correct place to
apply sparsity is that copy: gather-dequant only the selected rows into a compact buffer and hand the
**unmodified** kernel an effective `ne11 = width_pad`.

| file | work | LOC |
|---|---|---:|
| new `ggml-sycl/fattn-sparse.{cpp,hpp}` | (1) mask → index list + count (shared with Design A, ~110); (2) gather-dequant: row `i` of the compact f16 K/V buffer ← row `indices[i]` of the cache, reusing the per-type row dequant already in `fattn-common.hpp:598-616` (`get_dequantize_V`) (~90); (3) mask gather: compact mask of `width_pad` entries carrying the gathered values, `-INF` past `count` (~40) | ~240 |
| `ggml-sycl/fattn-common.hpp` | in `launch_fattn`, when sparse: substitute `K_data`/`V_data`/mask pointer and an effective `ne11`/`nb11`/`nb12`/`nb13`/`nb21..23`/`nb31`. **Every one of these is already a plain kernel argument** (`:1131-1138`), so nothing inside the kernel changes. `ntiles_KQ`/`parallel_blocks` (`:1069-1096`) recompute off the effective `ne11` for free. | ~80 |
| `ggml-sycl/fattn.cpp` | gate: `n_kv_max > 0 && Q->ne[1] == 1 && Q->ne[3] == 1 && mask && max_bias == 0 && logit_softcap == 0 && K->ne[1] >= max(4096, 2*n_kv_max)` | ~30 |
| `tests/test-backend-ops.cpp` | as above | ~10 |

**Why it is exact.** Softmax over a set of columns is permutation-invariant, and the compacted set is exactly
the finite-mask columns — which the op's own API contract already guarantees `n_kv_max` bounds
(`ggml/include/ggml.h:2505-2509`). Padding rows carry `-INF` and contribute nothing;
`width = 2051 → width_pad = 2304` is a clean multiple of `FATTN_KQ_STRIDE = 256`
(`fattn-common.hpp:18`), which is also what `gqa_opt` needs (`fattn-tile.hpp:1169`). The only difference
from dense is floating-point accumulation order — the same class of difference as any retiling.

**Why it is decode-only, by construction.** One compact K/V buffer can serve many query rows only if they
share one index set. At decode `Q->ne[1] == 1`, so there is exactly one mask row, and the mask is
head-independent (`[n_kv, n_tokens, 1, n_seq]`) — all 24 Q heads share it. For `Q->ne[1] > 1` each row has
its own set; that is precisely what upstream's `ncols1` grouping exists to amortize, and it needs Design A.
Decode is the case this task cares about, so the restriction costs nothing here.

**Bonus.** It also retires the dense f16 staging buffer — **242 MiB of VRAM at `n_kv = 118016`**
(`118016 x 256 x 2 heads x 2 B x 2 tensors`), on a card this config leaves ≤266 MiB of headroom on (§3.3).

**Risk: MEDIUM.** Three new small kernels; **zero edits to the FA kernel itself**; and the equivalence
harness already exists (§1.4). The residual risk is entirely in the compaction kernel dropping a column,
which `init_tensor_kq_mask_sparse` + the CPU reference catches directly.

### 4.3 Prefill: out of scope, for a stronger reason than `docs/research/07` gave

`docs/research/07` §4.2 point 3 said prefill has no hook because oneDNN/MKL are library GEMMs. #28770
sharpens this: its prefill design gathers per group of `ncols1 = 8` queries, so `n_gather = 8 x 2051 = 16408`
(and its gate needs `n_kv >= 32816` — the PR body's "For Qwen4 this value is 32768 ctx"). Reproducing that on
SYCL means **256 separate SDPA calls per `-ub 2048` ubatch** instead of one, each against a 16 408-row
compacted cache — trading a 7.2x traffic reduction for 256x the library-call overhead, and destroying the
batching that makes the XMX path fast in the first place. Combined with `PLAN.md`'s latest result (prefill at
116K is now 240.15 tok/s / 8 min 04 s TTFT, no longer the emergency it was), **prefill sparse FA on SYCL is
its own project and must not be attached to this one.**

### 4.4 Effort/risk summary

| | new/changed LOC | files | novel kernel work | recovers | risk |
|---|---:|---:|---|---|---|
| Design A (CUDA-shaped, kernel gather) | ~400-470 | 6 | compaction + indirect tile loads in a 1 246-line templated header | 21.45 of 35.13 ms/token | **HIGH** |
| Design B (compact-gather staging) | ~360 | 6 | compaction + gather-dequant + mask gather, all standalone | ~33.6 of 35.13 ms/token | **MEDIUM** |
| Prefill (either design) | 500+ | 7+ | restructuring the oneDNN/MKL call pattern | prefill only | **HIGH**, out of scope |

**Build gotcha that applies to both designs**, already paid for once by the top-k work: a *new* source file
under `ggml/src/ggml-sycl/` is picked up by a `*.cpp` glob **without** `CONFIGURE_DEPENDS`
(`ggml/src/ggml-sycl/CMakeLists.txt`), so `staging/work/devbuild.sh` links a stale library and silently omits
it — module libraries tolerate the undefined symbol at link time. Force `cmake .` in the build container's
`build/` first. (`PLAN.md` records this from the #28670 vendoring.)

---

## 5. Verdict

**Do not write the kernel in this pass.** `src/models/qwen4exp.cpp:943-946` stays exactly as it is.

Reasons, in order:

1. **The measured upside against the number this task set out to beat is 1.16x, not a fix.** FA is
   35.13 ms of a 240 ms decode token at 116K (§2.2, §3.3). Upstream's own decode claim for this architecture
   is 1.18x at `d100000` (§1.1), which agrees.
2. **The larger, unconditional half of long-context decode cost is the QSA indexer chain, which sparse FA
   provably cannot touch** (§3.4) — ~240 kernel launches per token on `n_kv`- and `n_blocks`-sized tensors,
   never measured, plausibly 25-100 ms/token. Optimizing FA before measuring that repeats the exact mistake
   `docs/research/07` §4.3 caught last time (fixing the smaller of two co-located costs).
3. **There is a 3.1x discrepancy between two of our own decode measurements that is not attention** (13.0
   tok/s modelled vs. 4.17 measured, §3.3), and the leading candidate — host page-cache residency of the
   52.8 GiB CPU-resident expert set — was observed dominating this box *today* (§3.1) and costs nothing to
   re-test. Chasing 1.16x while 3.1x is unexplained is the wrong order.
4. **The box could not produce a trustworthy long-context number today** (§3.1), so neither an A/B nor a
   perf claim could have been honestly reported even if the kernel had been written. Per the project's
   standing guardrail, the small-signal gate did not pass, so the large run was not started.

### What to do instead, cheapest first

1. **Profile a decode window at 32k depth by call site** — `GGML_SYCL_OP_PROFILE=1
   GGML_SYCL_OP_PROFILE_WINDOW=<small>`, on a *quiet* box. One run answers §3.4 and tells you whether the
   indexer or FA is the bigger term. This is the gate on everything below.
2. **Re-measure 116K decode with the host quiet and the model pre-warmed** (`PLAN.md`'s `cat`-the-model
   procedure), and check `vmstat`/`D`-state first. If decode comes back near 13 tok/s, the "severe decode
   problem" was an environment artifact and sparse FA is worth 1.78x rather than 1.16x — which changes the
   §4 recommendation from "don't" to "Design B is worth it".
3. **Try `-ctk f16 -ctv f16` at a depth that fits.** The `q4_0` KV cache costs a flat ~1.6x on the FA op,
   13.7 ms/token at 116K (§2.2) — a previously unaccounted cost. It is probably not a good trade (VRAM is
   what makes 116K fit) but the number should be on record.
4. **Only then, Design B** (§4.2), gated behind the `test-backend-ops` equivalence cases of §1.4 before any
   model run, and behind a 32K A/B before any 116K run.

### Conditions to revisit

- Step 1 above shows `FLASH_ATTN_EXT` is the largest depth-proportional term at decode, **and**
- step 2 reproduces a long-context decode number that the §3.2 model explains (i.e. the 3.1x gap closes),
  **and**
- the box can hold the CPU-resident expert set in page cache for the duration of an A/B.

### One correction to fold back into `docs/research/07`

§4.1's cost model for decode FA was a bracket of `0.08-0.57 ms` sparse vs. `5.30-36.2 ms` dense at
`n_kv = 131072`, derived from two anchors 6.8x apart. **The measured numbers are 2.93 ms dense at
`n_kv = 118016` per call, 35.13 ms/token over 12 layers** — i.e. the top of that document's dense bracket
was right and its bandwidth floor was 7x too optimistic, and the `#26581` 23 ns/position/layer slope it
called "a transfer, not a measurement of ours" transfers well (276 ns/position/token predicts 32.6 ms/token
against 35.13 measured). §4.1's sparse column was also right. What §4.1 missed entirely is the `q4_0`
staging tax (§2.1-§2.2), which is 39% of the dense cost.

---

## SUPERSEDING NOTE (2026-09-11, later): this document's verdict was measured at one depth, and the requirement turned out to be a slope

**Read `docs/research/12-decode-depth-scaling.md` before acting on the recommendation above.**

This document declined Design B because it was worth ~1.60x at 116K (post-`-lzm off`) and that did not
justify ~360 LOC of MEDIUM-risk SYCL. The requirement has since been stated as a *slope*: decode must stay
within **5% at 300K and 25% at 600K** of short-context speed. Doc 12 re-measured `FLASH_ATTN_EXT` and the
QSA indexer chain at nine depths out to `n_kv = 614400` and the reading changes in both directions:

- **Design B is no longer optional.** Doc 12 §4 shows no combination that meets the bar excludes it. §2's
  "the whole available sparse prize ~1.16x" was a 116K statement; the dense FA term is `0.0243 us per KV
  token per layer`, i.e. **89.9 ms/token at 300K and 179.6 at 600K**, against **0.67 ms flat** if bounded
  to `width = 2051`.
- **But Design B alone still misses the bar by a wide margin — +120% at 300K.** §3.4's suspicion that the
  QSA indexer is the other large depth-proportional term was right in kind even though doc 11 corrected its
  magnitude: the indexer's slope is **54% of dense FA's**, so removing attention entirely leaves a linear
  term that is 48.5 ms/token at 300K. **Design B is half of a matched pair, not a standalone fix.**
- The other half is an incremental cache for the indexer's block-pooled keys, which doc 12 §2 shows is
  **94.3% of the indexer's memory traffic** and measures at **7-8x** when hoisted out (doc 12 §4.1).
- §5's revisit conditions (a) and (b) are both now **met** — FA is the largest depth-proportional term, and
  the `-lzm off` fix closed the 3.1x gap so the model explains the measured token to 0.6%.

The Design A vs. Design B comparison in §4.2 is unaffected and B is still the right one.

**Second superseding note (2026-09-11, later still): "half of a matched pair" is now "one of a matched
triple", and the third piece is already built.** `docs/research/12` §9 implements block-granularity QSA
top-k (`build_qsa_top_k` selects the 513 winning *blocks* instead of expanding block scores to all `n_kv`
cells and running `ggml_top_k(2051)` over the full width). Measured on `test-backend-ops perf`, it removes
**80% of the depth slope that the pooled-key cache leaves behind** (0.001558 -> 0.000320 us per KV token
per layer; 8.39x -> 40.84x shallower than today). That changes what Design B is worth *as part of a
program*, not in isolation:

| combination | 300K | 600K |
|---|---|---|
| Design B alone | +120.3% | +243.0% |
| Design B + pooled-key cache ("the matched pair") | +14.9% | +29.8% |
| **+ block top-k (implemented, measured)** | **+3.3%** | **+6.1%** |

So the pair alone misses both bars and the triple clears both, on measured numbers with no estimated
factor. **This does not change the recommendation to build Design B** -- it strengthens it, because the
other two thirds of the combination are now one-third built and the arithmetic no longer depends on a
speculative fusion step. It also does not change §4.2's Design A vs. B verdict.

One caveat carried over from `docs/research/12` §9.5 that applies to any future A/B here: **greedy output
on this stack is not run-to-run reproducible**, so a Design B validation must use `test-backend-ops`
equivalence (which #28770's own sparse cases give for free) and an averaged quality metric with an error
bar, not a byte-comparison of generated text.

---

## ADDENDUM (2026-09-12): Design B is implemented. Dense FA's depth slope is gone: 26x shallower, and decode is flat to within 3% out to 116K

This document declined to write the kernel (section 5). Both superseding notes above reversed that on
projections; `docs/research/12` section 10 then turned the projection into a measurement and called
Design B "the next task and the whole remaining problem". **It is now built and measured.**

Source: new `ggml/src/ggml-sycl/fattn-sparse.{cpp,hpp}` (411 lines), 43 changed lines in
`ggml/src/ggml-sycl/fattn-common.hpp`, 22 in `src/models/qwen4exp.cpp`, 310 in
`tests/test-backend-ops.cpp`. Behind `LLAMA_QSA_NO_SPARSE_FA=1` (model side, drops the hint) and
`GGML_SYCL_FA_SPARSE=0` (backend side, ignores it) for same-binary A/Bs. No oneDNN/MKL, MoE-cache,
`topk-radix.cpp`, `gated_delta_net.cpp` or `hyper_connect.cpp` code was touched.

### A.1 What was built, and the one place it departs from section 4.2

Section 4.2's structure holds: the compact K/V/mask copy is built in `launch_fattn` and the
**unmodified** TILE/VEC kernel runs with an effective `ne11`. Every stride the kernel needs
(`nb11/12/13`, `nb21/22/23`, `nb31/32/33`) was already a plain kernel argument, and `k_VKQ_max`
falls back to `ne11` when `KV_max` is null (`fattn-tile.hpp:862`), so the loop bound shrinks for
free. Five small kernels, none of them templated:

1. per-tile count of finite mask entries (1 024 columns per work group);
2. prefix over tiles plus an in-group `exclusive_scan_over_group`, writing the ascending column list
   and the per-sequence count;
3. row gather for K, 4. row gather for V, 5. mask gather.

**The departure: section 4.2 costed a per-type gather-dequant kernel (~90 LOC of new dequant code).
That was avoided.** Rows of a KV cache are contiguous and `K->nb[0] == ggml_type_size(K->type)`, so
the gather is a **type-agnostic 32-bit byte copy** into a compact buffer in the cache's *own* type,
and the existing `ggml_get_to_fp16_sycl()` converter then dequantizes that buffer -- `n_rows` tall
instead of `n_kv` tall. The dequantization path is therefore the same tested code the dense staging
pass uses, and the one genuinely new numeric risk in the design is gone. It also covers an f16 cache
with no special case (the copy is the whole job).

Gate (`ggml_sycl_fattn_sparse_rows`, mirroring `ggml-cuda/fattn.cu:109-140`): `n_kv_max > 0`,
`Q->ne[1] == 1`, mask present and f16 with `mask->ne[0] == K->ne[1]` and `mask->ne[2] == 1`, no
sinks, `max_bias == 0`, `logit_softcap == 0`, `Q/K/V/mask` agreeing on `ne[3]`, contiguous rows a
multiple of 4 bytes, and `K->ne[1] >= max(4096, 2*GGML_PAD(n_kv_max, 256))`. Anything else stays
dense. Model side: `build_attn_qsa` now passes `top_k->ne[0]` (2 052 = 513 blocks x r=4 after
today's block-granularity top-k) instead of the hardcoded `0`.

**Two of this document's own claims about Design B did not survive contact with the code**, and both
are recorded rather than buried:

- **The 242 MiB VRAM bonus (section 4.2 "Bonus") does not materialize.** The f16 staging buffer is
  reserved by `ggml_sycl_flash_attn_ext_get_alloc_size()` at `sched_reserve` time, where the
  worst-case ubatch is `n_tokens = n_ubatch = 2048` and sparse therefore does not apply. The
  decode-time buffer is smaller but the *reservation* is unchanged, so the compute buffer is
  byte-identical with and without the change. Shrinking it would need the prefill design, not this one.
- **`Q->ne[1] == 1` is a stricter limit in a server than it looks.** A `--parallel N` deployment with
  split streams still decodes one token per stream (`Q->ne[1] == 1`, `ne[3] == N`) and is covered; a
  *unified* cache batching N sequences into one ubatch gives `Q->ne[1] == N` and silently falls back
  to dense. The 300K single-request target is the covered case.

### A.2 Correctness, three ways, none of them a cross-run comparison

Per `docs/research/12` section 10.2, nothing on this stack is comparable across processes, so no
oracle here uses one.

**(1) `test-backend-ops test -b SYCL0 -o FLASH_ATTN_EXT`: 23/23 sparse cases pass, 0 new failures.**
This is the real equivalence proof. The CPU reference ignores `op_params[4]` and
`init_tensor_kq_mask_sparse` builds a genuinely sparse mask with a *different* finite set per row, so
any column the compaction drops shows up as NMSE against a dense CPU result, at `max_nmse = 5e-4`.
Six qwen4exp-shaped cases were added: head 256 / gqa 12 at `n_kv_max = 2052` in f16 and in the
production `q4_0`, a two-sequence (`nr23 = {12,2}`) variant, budgets of 513 and 2 048, and an
`nb = 64` arm that must take the dense fallback. The pre-existing 17 upstream sparse cases (MLA
shapes, quantized, V-as-subview-of-K, large batch) now exercise this path too and all pass.
The suite's one failure -- `hsk=256 hsv=256 nh=2 nr23=[16,1] kv=1025 nb=64 q8_0 permute=[0,2,1,3]`,
`n_kv_max = 0` -- **reproduces on `staging/devbin-pool`, the 13:50 build that predates this work**
(ERR 1.235 there, 1.352 here; the ERR itself moves run to run). It is a dense bug and it is not this
change's. Logs: `logs/fa-sparse-test.log`, `logs/fa-prechange-test.log`.

**(2) An in-graph oracle inside the real model: 240/240 comparisons pass, worst NMSE 1.7e-6.**
`LLAMA_QSA_SPARSE_CHECK=1` makes `build_attn_qsa` build a *second* `build_attn_mha` with the hint set
to 0, on the same q/k/v and the same `kq_mask_top_k`, in the same graph;
`staging/work/qsa_sparse_equiv.cpp` compares `kqv_out-<il>` against `kqv_out_ref-<il>` row by row
inside one run. Over an 8 000-token prompt (4 ubatches) plus 16 decode steps at `n_kv = 8 192`,
12 layers: **worst NMSE 1.691e-06, worst absolute difference 1.44e-02, nothing over the 5e-4 bar.**
The load-bearing detail is the control: the four **prefill** ubatches take the dense fallback on both
sides and come back **bit-identical (NMSE exactly 0, max abs diff exactly 0)** -- so two dense FA
calls in one graph agree to the bit, which is what makes the 1e-7..1.7e-6 spread on the decode steps
attributable to the restriction's floating-point reassociation rather than to ambient noise.
Log: `logs/qsa-sparse-equiv.log`.

**(3) A backend contract check.** The op contract is that `n_kv_max` bounds the finite entries of
*every* mask row; over that bound the gather silently drops columns. `GGML_SYCL_FA_SPARSE_CHECK=1`
reads the per-sequence counts back and aborts if any exceeds it. It was on for the whole of (2) and
never fired. The argument that it cannot fire: `kq_mask_top_k` is `-INF` everywhere except the rows
`ggml_set_rows` unmasks from `top_k`, so the finite set is a subset of at most `top_k->ne[0]` indices.

**What is NOT covered, stated plainly.** The deeper in-graph run (32 000-token prompt) **could not be
made to fit**: check mode builds a second FA node per layer, which doubles the per-node f16 staging
reservation, and it OOMs above ~16K of context at this VRAM -- twice, at `-ncmoe 24 -c 40960` and
`-ncmoe 32 -c 33792`, both times as the "OOM presents as a spinning process" hazard `PLAN.md`
documents. Depth coverage therefore rests on (1), where the CPU-reference comparison runs at
`n_kv` up to **614 400** (600 compaction tiles), not on (2). And **`llama-perplexity` cannot gate
this change at all**: perplexity is pure prefill (`Q->ne[1] = 2048` per ubatch), the gate rejects it,
and the sparse path never executes -- so a perplexity A/B here measures the known cross-run noise and
nothing else. That is a property of a decode-only design, not an omission.

### A.3 The kernel number, and it is larger than any estimate in this document

`test-backend-ops perf -b SYCL0 -o FLASH_ATTN_EXT`, production decode shape (head 256, 2 KV heads,
gqa 12, `Q->ne[1] = 1`, `q4_0` K/V), idle GPU, full log `logs/fa-sparse-perf.log`:

| `n_kv` | dense us/call | sparse us/call | x | dense, 12 layers | sparse, 12 layers |
|---:|---:|---:|---:|---:|---:|
| 8 192 | 181.34 | **69.51** | 2.6 | 2.18 ms | 0.83 ms |
| 16 384 | 394.95 | **69.58** | 5.7 | 4.74 ms | 0.83 ms |
| 32 768 | 810.78 | **70.09** | 11.6 | 9.73 ms | 0.84 ms |
| 65 536 | 1 622.27 | **71.60** | 22.7 | 19.47 ms | 0.86 ms |
| 118 016 | 2 919.42 | **72.22** | 40.4 | 35.03 ms | 0.87 ms |
| 200 704 | 4 930.89 | **72.41** | 68.1 | 59.17 ms | 0.87 ms |
| 307 200 | 7 507.15 | **73.88** | 101.6 | 90.09 ms | 0.89 ms |
| 614 400 | 14 995.09 | **78.70** | 190.5 | 179.94 ms | 0.94 ms |

Section 2.2's dense `118016` figure was 2 927.28 us; this run measures **2 919.42 us on a different
day, 0.3% apart**, which is what makes the sparse column trustworthy.

Two readings:

1. **The slope, which is the whole point.** Dense is `0.024437 us per KV token per layer`; sparse is
   `1.516e-05` -- **1 612x shallower**. Over 12 layers the depth term goes from `0.2932 ms per 1 000
   tokens of depth` to `0.00018`. The residual is the two-pass mask scan (4 bytes per KV token per
   layer), i.e. 0.05 ms/token at 300K, which is why the sparse column rises 69.5 -> 78.7 us over a
   75x depth range rather than staying exactly flat.
2. **The absolute is above section 1.1's proxy and that was predictable.** `docs/research/12`
   section 1.1 used the dense `n_kv = 2048` row (55.47 us) as the bounded-call proxy; the real thing
   is **69.5-78.7 us**, 25-42% more, because it also pays the mask scan, the gather, the compact
   dequantization and five extra kernel launches per call. 0.83-0.94 ms/token against the proxy's
   0.67. The proxy was the right order of magnitude and slightly optimistic.

### A.4 End-to-end decode, arms interleaved with a trailing control

`llama-bench -p 0 -n 32 -r 2`, `-ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -ncmoe 24 -ub 2048 -lzm off -lm
none`, same binary, `LLAMA_QSA_NO_SPARSE_FA=1` for the dense arm, behind a discarded warm-up arm.
Driver `staging/work/qsa_sparse_bench.sh`, log `staging/work/qsa-sparse-bench.log`.

| depth | dense (lead) | dense (trail) | **sparse** | x |
|---:|---:|---:|---:|---:|
| 8 192 | 25.07 | 25.41 | **25.36** | 1.00 |
| 32 768 | 21.18 | 21.04 | **25.43** | **1.20** |
| 65 536 | 17.39 | 17.47 | **25.49** | **1.46** |
| 118 016 | 13.56 | 13.62 | **24.56** | **1.81** |

(tok/s.) **The trailing dense control reproduces the leading one to 0.4-1.4% at every depth**, which
is what licenses the comparison -- the same interleaving discipline `docs/research/12` section 9.6
needed for a 1.5% effect, here applied to a 1.8x one.

**Decode is now flat with depth to within 3.2%**: 25.36 / 25.43 / 25.49 / 24.56 from 8K to 116K,
against the dense arm's 25.07 -> 13.56 (a 1.85x loss). The dense arm reproduces
`docs/research/12` section 10.3's pooled-key numbers (24.86 / 21.00 / 17.27 / 13.46) to within 1.2%.

### A.5 The 300K target, measured

Same driver, section B: `-ncmoe 32 -d 307200 -r 1` (the VRAM configuration `docs/research/12`
section 6's probe found 300K needs at `-ub 2048`), arms interleaved on one binary.

| depth | config | dense | **sparse** | x |
|---:|---|---:|---:|---:|
| 307 200 | `-ncmoe 32 -ub 2048` | 7.38 tok/s | **21.08 tok/s** | **2.86** |

The dense arm lands on `docs/research/12` section 10.4's pre-registered projection of **7.51 tok/s**
to **1.7%**, at a depth 2.6x past anything that model was fitted from -- which is the check that makes
the sparse number believable rather than a measurement artifact.

**Decomposing the 21.08 honestly, because two different things cost ms here.** 21.08 tok/s is
47.44 ms/token against the `-ncmoe 24` 8K sparse arm's 39.43 ms.

- **Depth costs ~3.5 ms of that.** The measured `-ncmoe 24` sparse slope (39.43 ms at 8 192 ->
  40.72 ms at 118 016, i.e. **0.01174 ms per 1 000 tokens of depth**) puts 307 200 at **42.94 ms
  (23.29 tok/s)** if it could run at `-ncmoe 24`.
- **The other ~4.5 ms is VRAM, not attention.** 300K only fits at `-ncmoe 32`, i.e. eight more
  expert layers streaming from host RAM. `PLAN.md` measured that floor penalty at +0.4 ms per
  host-resident expert layer (36.3 -> 39.5 ms from `-ncmoe 24` to 32); 8 layers x 0.4 = 3.2 ms
  predicted against 4.5 measured.

So the depth-attributable loss from 8K to 300K is now **+8.9%**, and the whole measured loss
including the `-ncmoe` penalty is +20.3%.

**600K was measured, it misses the bar, and the reason is not attention.** `-ub 2048` cannot reach
614 400 at all -- its compute buffer runs ~0.0267 MiB per token of context (`docs/research/12`
section 6), i.e. **~16 GiB** on top of 5.6 GiB of KV and 0.9 GiB of pooled-key cache. At
**`-ub 1024 -ncmoe 38`** it fits and runs (`-ub` changes the compute buffer and prefill, not the
`n_tokens = 1` decode graph, so the decode number is comparable):

| depth | config | **sparse** |
|---:|---|---:|
| 8 192 | `-ncmoe 38 -ub 1024` | 22.55 tok/s |
| 32 768 | `-ncmoe 38 -ub 1024` | 22.93 tok/s |
| **614 400** | `-ncmoe 38 -ub 1024` | **12.42 tok/s** |

**12.42 tok/s is below the user's stated 17-25 tok/s floor at 600K**, and the same-config shallow
control is what makes that interpretable: 22.55/22.93 at 8K-32K means the loss is **+82% in
ms/token** (44.0 -> 80.52 ms), i.e. a **residual depth slope of 0.0602 ms per 1 000 tokens** at this
configuration -- **5.1x the 0.0119 ms/1k least-squares slope the `-ncmoe 24` arm measured over four
depths.**

**That residual is provably not attention.** A.3 measures the sparse FA op at **0.94 ms/token flat
at `n_kv = 614 400`**, so of the 36.5 ms/token that depth costs here, attention is under 1 ms. Two
candidates remain, neither ever isolated:

- **`docs/research/12` section 10.3's O(`n_kv`) host half of `set_input_qsa`** (the grouping scan,
  the per-token O(`n_blocks`) block bias, the `cell_blk`/`blk_cells`/`bias` uploads), measured there
  at 0.0155 ms/1k. **A slope that grows 5x when only `-ncmoe` changes is the signature of a host-side
  term**, because `-ncmoe 38` puts 38 layers of CPU expert GEMM on the same cores, and this work is
  single-threaded. That is the leading hypothesis and it is testable with the profiler this project
  already has.
- **The KQ-mask build in `build_attn_qsa` itself** (`qwen4exp.cpp`: `ggml_fill(-INF)` +
  `ggml_set_rows` + `ggml_add` over the full `n_kv`-wide f16 mask, 12 times per token), which is
  O(`n_kv`) GPU traffic that the sparse FA op does not remove and that no measurement in this project
  has ever separated from the rest.

So: **300K is inside the bar on a real measurement; 600K is not, and what stands between them is no
longer flash attention.** The `-ncmoe 24` slope would put 614 400 at ~46.3 ms (21.6 tok/s) if VRAM
allowed it, which brackets how much of the 600K shortfall is configuration rather than depth.

### A.6 Verdict, replacing section 5

**Build it -- it is built.** Against section 5's four reasons for declining:

1. "The measured upside is 1.16x." It is **1.81x at 116K and 2.86x at 300K** end to end. Section 5's
   1.16x was computed against a 240 ms/token that was 61% host paging; `-lzm off` removed that
   denominator and the pooled-key cache and block top-k removed most of the rest.
2. "The larger co-located cost is the indexer." It was, and it is fixed (`docs/research/12`
   sections 9-10). Sparse FA was correctly sequenced *after* it, not instead of it.
3. "A 3.1x discrepancy sits unexplained." Closed by `docs/research/11` section 7 (`-lzm off`).
4. "The box could not produce a trustworthy number." It did, with a trailing control reproducing the
   leading arm to 0.4-1.4%.

**What is left, in order of value.** (i) The residual slope is now **0.01174 ms/1k**, of which the
mask scan is only ~0.0002 -- so what remains is `docs/research/12` section 10.3's O(`n_kv`) host term
in `set_input_qsa` (the grouping scan, the per-token block bias, and the three input uploads). It is
now **the largest depth-proportional term in the decoder by a wide margin** and the same
"only the tail changes" argument applies to it. (ii) Prefill sparse FA remains out of scope for
section 4.3's unchanged reason. (iii) The `Q->ne[1] == 1` restriction means a unified-cache
multi-sequence decode batch falls back to dense; splitting streams avoids it.

### A.7 The real 116K server run, with the sampler and a quality read

`staging/work/run_116k_sparse.sh`, same prompt (116 277 tokens), same config and same server-side
timers as `logs/long-context-120k-lazyoff/`, `GGML_SYCL_FA_SPARSE_CHECK=1` on for the whole run.
Artifacts in `logs/long-context-120k-sparse/`.

| | baseline (`-lzm off`, dense) | **sparse** |
|---|---:|---:|
| prefill | 394.14 tok/s | 419.70 tok/s |
| **decode** | **11.15 tok/s** | **23.75 tok/s (2.13x)** |
| TTFT | 4 min 55 s | 4 min 37 s |

Prefill moves 6.5%, which is **not** attributable to this change -- the gate rejects prefill, so that
is ordinary run-to-run variance and should be read as unchanged. Decode 2.13x is the result, and it
agrees with the `llama-bench` `d118016` row (24.56) to 3.4% once the sampler chain and streaming are
paid for. Against the original pre-`-lzm off` 5.34 tok/s the cumulative decode improvement at 116K is
**4.45x**.

The `tg_3s` interval spread over the decode window is **22.47-24.72 t/s, 1.10x**, i.e. still the
tight band `-lzm off` produced rather than the 3.09x paging band before it.

**Quality holds and the contract check never fired.** The saved reasoning trace
(`run1.reasoning.txt`, 1 968 chars) correctly reads the prompt's real structure -- the near-identical
per-entity functions (`user_0`, `order_1`, `product_2` ... `transaction_99`), the per-entity
constants (multiplier, addend, divisor, tax rate, cache key default, name case, error label), and
every planted anti-pattern (bare except, mutable defaults, global mutable state, manual index loops,
`== None`) -- and proposes the parameterized `EntityConfig` refactor. No degeneracy. And
`GGML_SYCL_FA_SPARSE_CHECK=1` verified on **every one of 400 decode steps x 12 layers at
`n_kv = 118 016`** that no mask row carried more finite entries than the `n_kv_max` the model
promised, which is the production-depth contract coverage the in-graph run in A.2 could not reach.
