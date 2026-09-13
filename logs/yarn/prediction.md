# Pre-registered predictions, written before any arm ran (2026-09-12)

Derived from a from-scratch replication of llama.cpp's `rope_yarn` math with
this GGUF's own metadata (`rope.freq_base = 1e7`, `rope.dimension_count = 64`,
`qwen4exp.context_length = 262144`, no rope-scaling metadata at all), at
`--rope-scale 1.1875 --yarn-orig-ctx 262144` (`staging/work/yarn_math.py`):

- `ggml_rope_yarn_corr_dims(64, 262144, 1e7, 32, 1)` = **[14, 22]** of 32
  channel pairs. So pairs 0-14 are **fully extrapolated** (theta untouched),
  pairs 22-31 **fully interpolated** (theta x 0.842105), 15-21 on the ramp.
- `mscale = 1 + 0.1*ln(1.1875) = 1.01719`, applied to all 64 rotated dims and
  **not** to the 64 (attention, head 256 -> 192) / 64 (indexer, head 128 -> 64)
  unrotated dims. Uniform on q and k, so it cannot reorder anything.
- The perturbation is therefore **exactly proportional to position**. At the
  deepest position of a chunk, the largest absolute theta shift is on pair 15
  (wavelength 12,007 tokens): 0.0197 rad per 1,000 tokens of depth.

**Predictions (falsifiable):**

1. `ppl(yarn, c=8192)` ~= `ppl(off, c=8192)` to within run-to-run noise.
   At pos < 8192 the shift on the worst band is 0.085 rad and on the fully
   interpolated bands 0.02 rad. If YaRN is *visibly* broken here, the damage is
   NOT positional and llama.cpp's YaRN wiring is suspect.
2. `ppl(yarn, c=32768)` degraded but finite; `ppl(yarn, c=98304)` badly
   degraded (pair-15 shift ~1.0 rad = 58 degrees).
3. Consequently **the hypothesis that the 116K YaRN test was invalid because
   116,277 < 262,144 must fail**: static YaRN in llama.cpp does not gate on
   whether the sequence exceeds `n_ctx_orig` (and neither does the reference
   YaRN, which is a reparametrisation, not dynamic NTK), so moving the test
   past 262,144 strictly increases every theta shift above. If prediction 1
   holds and 2 holds, testing "properly" past 262,144 cannot rescue YaRN.
4. `ppl(off, c=294912)` (unscaled extrapolation 12.5% past `n_ctx_train`)
   is the number that actually matters for this project, and nothing in the
   static analysis says it must be bad: unscaled extrapolation leaves every
   trained frequency alone and only asks the two longest-wavelength bands
   (408K and 675K tokens) to report a phase 12.5% beyond what they saw in
   training, which is under a tenth of a turn.

## Ruled out by code reading before any arm ran

- **No main-attention / indexer RoPE mismatch.** All four
  `ggml_rope_multi` call sites in `src/models/qwen4exp.cpp` (Q and K at
  `:1144,1150`, the indexer's pooled keys at `:890` inside `build_pool`, and
  the indexer queries at `:927`) pass the *same* `n_rot`, `sections`,
  `rope_type`, `n_ctx_orig`, `freq_base`, `freq_scale`, `ext_factor`,
  `attn_factor`, `beta_fast`, `beta_slow`. Positions are in the same units on
  both sides: the pooled keys use `bid_idx[b]`, documented and implemented as
  the block's **first token position** (`llama-memory-hybrid-idx.h:45`,
  `.cpp:837` = `pb*r`), not a block ordinal.
- **`dimension_sections = [11,11,10,0]` is irrelevant to YaRN here, and
  irrelevant to text inference entirely.** `rope_multi`'s sections only select
  *which* position row a frequency band reads; the frequency itself is still
  `theta_scale^(iw/2)` over the global dim index, which is exactly what
  `rope_yarn_ramp` indexes. And for a text batch `llama-batch.cpp:781-788`
  broadcasts the same position into all four rows, so imrope degenerates to
  plain NeoX rope and all four sections read the same value.
- **`yarn_attn_factor` is not double-applied.** `llama-context.cpp:196-210`
  sets it to `get_mscale(factor,1)` and then divides by the identical
  expression precisely because the kernel re-applies it, landing at exactly
  1.0 before `hparams.rope_attn_factor`.
