# 19 - Hand-merging upstream #29245 (grouped MoE XMX GEMM) against our SYCL fork

**Status: merge applied to the working tree only, not built, not run.** No GPU work of any kind was done; the
production server was not touched. Date: 2026-09-27.

Triggered by `docs/research/16` §4.3: upstream PR #29245 (`ggml-org/llama.cpp`, author `cwriter`, OPEN as of
this writing) replaces the SYCL `MUL_MAT_ID` per-expert dequant+GEMM loop with a grouped/batched XMX kernel,
measured by its author at **+34% prompt throughput on this exact model** (`Qwen3.8-Flash-Next UD-IQ4_XS`,
274.1 -> 368.3 tok/s) and **+49% on a similar IQ3_XXS MoE on a B60**. Doc 16 estimated the merge at ~1 day,
medium risk, gated on `test-backend-ops -o MUL_MAT_ID` plus an env-var A/B - neither was run here per the
brief (no build, no GPU).

Sources: `gh pr view 29245` / `gh pr diff 29245 --repo ggml-org/llama.cpp` (fetched fresh, not from doc 16's
earlier `gh` query), the PR's three review comments, and direct reads of
`src/llama.cpp/ggml/src/ggml-sycl/{common.hpp,ggml-sycl.cpp,moe-cache.{cpp,hpp}}` as they stand in this
repo's working tree (pinned upstream `6d9c82ea2` + `patches/llama.cpp.patch` + this session's uncommitted
local changes).

## 1. What the PR actually touches

Five files, +775/-16: `docs/backend/SYCL.md` (+1), `ggml/src/ggml-sycl/common.hpp` (+23), two new files
`fused-gemm.{cpp,hpp}` (+633/+80), and `ggml/src/ggml-sycl/ggml-sycl.cpp` (+38/-16). The `ggml-sycl.cpp` diff
touches exactly three spots:

1. `ggml_sycl_op_mul_mat_sycl` (the single-GEMM dispatcher `ggml_sycl_mul_mat` calls for one row range):
   moves the existing `src0_as_f16` dequant-to-f16-then-GEMM block after the `src1_ptr` f16 conversion, and
   inserts one call to `ggml_sycl_fused_dequant_gemm_f16(...)` ahead of it that, when it succeeds, dequantizes
   straight into the GEMM's XMX tiles and returns early - skipping the f16 write-out/read-back entirely for
   narrow-N cases (N <= 64).
2. `ggml_sycl_mul_mat_id`'s prefill branch (`ne12 > 1`, the "many tokens routed to many experts" case): right
   after the existing `k_copy_src1_to_contiguous` gather kernel and before the `for (i02 < n_as)` per-expert
   loop, it inserts one call to `ggml_sycl_grouped_dequant_gemm_f16(...)`. On success it sets `grouped = true`
   and the per-expert loop's condition becomes `i02 < n_as && !grouped`, so the loop becomes dead code (never
   iterates) instead of being deleted - a one-line, easy-to-revert gate.
3. The env var machinery: a new `GGML_SYCL_XMX_GATHER_TYPES` bitmask (one bit per quant type: IQ4_NL, IQ3_S,
   IQ4_XS, IQ3_XXS, IQ2_XXS/XS/S, IQ1_S/M; default all-on, 0 disables both paths above and falls back to the
   pre-PR code unconditionally).

`common.hpp` adds the bitmask enum + its global, a `ggml_sycl_gg_tile { expert, n0, n1 }` host-schedule
struct, and one new `std::vector<ggml_sycl_gg_tile> mmid_tile_schedule_host` member on
`ggml_backend_sycl_context` (same pattern as the existing `mmid_row_mapping_host` right above it - the PR
author clearly modeled this on that field). `fused-gemm.cpp` is the kernel itself: per-quant-type "A stage"
dequant functions (one lane decodes one row's current 16- or 32-wide step into `sycl::half2` registers,
folding the block's scale in), an XMX `joint_matrix` GEMM tiled 8x16x16 with a 4-way K-split summed at the
end, and two host entry points - `ggml_sycl_fused_dequant_gemm_f16` (plain `MUL_MAT`, one weight matrix) and
`ggml_sycl_grouped_dequant_gemm_f16` (the `MUL_MAT_ID` case: the host walks `expert_row_offsets` to build a
list of `(expert, row_range)` tiles up to `GGML_SYCL_FG_MAX_N` (64) rows wide, uploads that schedule, and
launches one grid where work-group `t` reads tile `t`'s expert directly from `src0_base + tile.expert *
expert_stride`).

## 2. Where our fork's local changes sit relative to that, and why the merge came back clean

This project's SYCL fork has three families of local changes in the MoE dispatch path (read directly, not
paraphrased from PLAN.md): a decode-time fused MMVQ path for uncached experts
(`ggml_sycl_mul_mat_id_mmvq_fused`, `ggml-sycl.cpp:5067`), a device-planned fused path for cached experts
backed by the MoE cache (`ggml_sycl_mul_mat_id_device_planned_fused`, `:5153`, which calls into
`ggml_sycl_moe_cache_*` in `moe-cache.cpp`), and the cached-tensor branch inside the legacy per-row loop
(`ggml_sycl_mul_mat_id`, the `if (ne12 == 1 && src0_cached)` block at `:5357-5427`). **All three are gated on
`ne12 == 1`** - i.e. they only ever run for decode (one token, or a small batch of tokens each choosing their
own experts). The PR's grouped kernel is gated the opposite way: it only fires inside the `else` branch that
handles `ne12 > 1` (prefill: many token-positions sharing one op, each routing independently). Diff review
confirms zero line-level overlap between what our fork modified and what the PR modified inside
`ggml_sycl_mul_mat_id` - our changes stop right where the PR's insertion point begins (`}` closing the
`ne12 == 1` branch already sits three lines before the `} else {` the PR's context anchors on). The only
thing shared between old and new code in that function is the `ids_host` readback and `stream->wait()`
above both branches, untouched by either side.

`ggml_sycl_op_mul_mat_sycl` - the second insertion point - has **no local modifications at all**; it reads
identically to stock upstream (confirmed by diffing our tree's copy line-by-line against the PR's "before"
context). `common.hpp`'s insertion points (the `extern int g_ggml_sycl_*` block, the `mmid_row_mapping`
struct, and the `ggml_backend_sycl_context` field list) also matched stock text exactly, just at different
absolute line numbers because of earlier unrelated local insertions above them in the file.

**Net result: this was not a conflict-resolution merge in the normal sense.** I applied the PR's five hunks
by hand at the equivalent (shifted) locations in our tree, copied the two new files across verbatim, and
nothing needed to be reconciled - not because I forced a side to yield, but because the PR's author (also the
radix top-k author) happened to touch a part of `mul_mat_id` (prefill) that is structurally disjoint from the
part this project has spent its effort on (decode, via the MoE cache). Doc 16 §2.3 already established why:
"the MoE cache: still no prefill path, and there is no prefill path worth building" - our fork deliberately
never extended its cache logic into the `ne12 > 1` branch, which is exactly the branch this PR rewrites.

## 3. The one real design question, worked through (not a code conflict, a behavioral one)

Doc 16 §2.3 also established that a `SYCL_MoE_Cached` tensor at prefill "falls through to the per-expert loop
reading `src0` through the pinned host pointer" - i.e. zero-copy PCIe reads directly against USM host memory,
not a device-resident copy. Confirmed again here directly: `moe-cache.cpp:15-29`'s own header comment states
`is_host()` is hardcoded `false` specifically so the scheduler treats the buffer as SYCL-backend-owned and
never routes a generic host->device copy for it before compute; the tensor's `->data` pointer is the pinned
host allocation for the tensor's whole lifetime, dereferenced directly by the compute kernel.

The grouped path I merged in does not distinguish `src0_cached` from ordinary device- or CPU-resident
`src0` at all - and it doesn't need to. `src0_original = (char *) src0->data` is captured once, before either
the old per-expert loop or the new grouped call runs, and both index into it the exact same way
(`src0_original + i02 * nb02` in the old loop, `src0_base + tile.expert * expert_stride` in the new kernel,
same `nb02`/`expert_stride`). Whatever `src0->data` already points to - pinned host USM for a cached tensor,
a `SYCL_Host` pinned buffer for an `-lm none`-loaded CPU-resident expert, or a device buffer the scheduler
already copied there for a normal `-ncmoe` split - is unchanged by this merge; the grouped kernel is a
drop-in replacement for the loop's compute, not for its addressing. **So there is no correctness conflict to
resolve between the MoE cache and the grouped GEMM.** The open question is a performance one, not a
correctness one, and it cuts in a direction doc 16 didn't consider: the grouped kernel's A-stage does
scattered per-lane, per-k-step reads structured for a *dequant-heavy, bandwidth-light* access pattern
(one XMX tile's worth of one row at a time). Doc 16 §10.5 (cited via doc 05) measured that exact
zero-copy-host-read pattern at ~3.1 GB/s cold against SYCL_Host - far below the 21.9 GB/s bulk-copy rate the
old loop's *dequant-then-GEMM* pays for GPU-resident experts. Whether the grouped kernel's finer-grained
reads make that number better or worse against pinned host memory is not something either the PR's author
(who benchmarked on `-lm none`/fully pinned setups per the PR body) or this repo's existing measurements can
answer. **This does not block the merge** - prefill correctness doesn't depend on it, and at `-ub 2048` doc
16 §2.1 established the whole expert bank is touched every ubatch regardless of cache state, so a cached
tensor at prefill is already on the slow path today; the grouped kernel can only match or beat that baseline,
never fall below the old per-expert loop's own already-slow host-read behavior for that same tensor. It is,
however, worth an explicit A/B once built: `-lm none` (everything pinned as one contiguous host buffer, no
MoE-cache buffer type in play) vs. whatever the production `--mlock-experts-only` config exercises, both with
`GGML_SYCL_XMX_GATHER_TYPES` on and off, to see if the grouped kernel's per-row-narrow reads regress the
cached case specifically.

## 4. Shape/type gating sanity-checked against this deployment

- Type coverage: the grouped path's `ggml_sycl_xmx_gather_type_enabled` covers IQ4_NL, IQ3_S, IQ4_XS,
  IQ3_XXS, IQ2_XXS/XS/S, IQ1_S/M. Doc 16 §2.4 (`logs/topk-ab-after.log`) shows this project's `UD-IQ3_XXS`
  build actually dispatches `iq2_s`, `iq3_s`, and `iq4_nl` `MUL_MAT_ID` ops - all three are in the covered
  list, so the type gate should not silently decline every call in practice (unverified without a build).
- Width gate: `ggml_sycl_grouped_dequant_gemm_f16_shape_ok` requires `total_rows <= n_active *
  GGML_SYCL_FG_MAX_N` (64), i.e. the grouped kernel only fires when the average active expert sees <= 64
  routed rows. At `-ub 2048`, 10-of-512 routing: `n_routed_rows = 20,480`; doc 16 §2.1 established that at
  this batch size essentially all 512 experts see nonzero traffic (`~e^-40` chance any one is unused), so
  `n_active ~= 512` and the average is `~40 rows/expert` - comfortably under 64. At smaller `-ub` (512, 1024)
  the same batch-vs-expert-count arithmetic holds even more loosely (fewer total rows, same expert count).
  **I did not find a plausible shape in this deployment's tier table where the grouped path would decline on
  width alone** - it should dispatch on every prefill ubatch this project actually runs. This is arithmetic
  from doc 16, not a new measurement.
- The `K % QK_K == 0` (or `% QK4_NL` for IQ4_NL) requirement: this model's expert `ne[0]` (the FFN
  intermediate/hidden dim feeding these tensors) needs to be a whole multiple of 256 (`QK_K`) for every
  non-IQ4_NL type in the list. I did not re-derive this project's exact tensor shapes from the GGUF metadata
  in this session (out of scope for a no-build merge) - this is a one-line `test-backend-ops` check away and
  should be part of the build/test pass's gate, not assumed here.

## 5. What I deliberately left as-is / did not touch

- Did not modify `patches/llama.cpp.patch` - per the brief, that gets regenerated from the tree separately.
- Did not touch `moe-cache.{cpp,hpp}`, `topk-radix.{cpp,hpp}`, `mmid-hybrid.{cpp,hpp}`, `expert-pool.{cpp,hpp}`,
  or `hyper_connect.{cpp,hpp}` - none of them intersect the two functions this PR changes (confirmed by
  grepping for cross-references from those files into `ggml_sycl_mul_mat_id`'s `ne12 > 1` branch or into
  `ggml_sycl_op_mul_mat_sycl`; there are none).
- Did not add a CMakeLists.txt entry for the two new files - `ggml/src/ggml-sycl/CMakeLists.txt` already
  globs `*.cpp`/`*.hpp` in the directory (confirmed by reading it; this project's own prior additions like
  `moe-cache.cpp` rely on the same glob), so `fused-gemm.{cpp,hpp}` are picked up automatically.
- Did not run `test-backend-ops`, did not build, did not touch the GPU or the production server, per the
  brief.

## 6. What I'm not sure will actually work once built

1. **`std::unordered_map<sycl::device, bool>` in `ggml_sycl_fused_dequant_gemm_f16_device_ok`**
   (`fused-gemm.cpp`) needs `std::hash<sycl::device>` to be defined by the SYCL implementation. I did not
   verify this project's Intel oneAPI/DPC++ version actually provides it - if it doesn't, this is a
   compile-time error, easy to fix (swap to a raw pointer or backend-specific handle as the map key) but not
   something a no-build pass can confirm.
2. **The XMX `joint_matrix` combination query** (`fused_gemm_f16_supported`, checking for an
   `fp16 x fp16 -> fp32` combination at 8x16x16) against this project's actual toolchain/driver on the B70 -
   the PR was authored and measured on Arc Pro B60s; whether the B70's reported `matrix_combinations` satisfy
   the `(max_msize >= 8 || msize == 8)`-shaped OR-conditions the same way is unverified.
3. **Performance on the cached (`SYCL_MoE_Cached`) src0 case at prefill** - see §3. Correctness should hold
   regardless; whether it's a net win, a wash, or a regression against pinned-host reads is genuinely open
   and worth an explicit A/B rather than assuming the author's +34% (measured on `-lm none`, i.e. no MoE
   cache involved at all, since the cache is decode-only) transfers unchanged to this project's mixed
   cached/uncached prefill traffic.
4. **The `K % QK_K == 0` shape gate against this model's actual expert tensor dimensions** (§4) - not
   re-derived here.

## 7. Suggested build/test-pass gate (per doc 16 §4.3, restated)

1. `test-backend-ops -o MUL_MAT_ID` (and, if it exists in this tree's harness, `-o MUL_MAT`) as the
   correctness gate, run twice: once with `GGML_SYCL_XMX_GATHER_TYPES=0` (should reproduce today's numbers
   exactly, since every dispatch declines) and once at the default (all bits set), diffing output.
2. One 116K prefill run (`staging/work/run_116k_lazy.sh` per doc 16 §4.2's own reference) at each setting,
   same env-var A/B, to reproduce a throughput delta before trusting the upstream author's number on this
   box's actual config (`--mlock-experts-only`, not the author's `-lm none`).
3. If §3's cached-case question matters in practice, a third arm: `-lm none -lzm off` (matching the PR
   author's own tested configuration) vs. production's `mmap+mlock`, both with the grouped path on, to see if
   the +34% shows up equally in both.
