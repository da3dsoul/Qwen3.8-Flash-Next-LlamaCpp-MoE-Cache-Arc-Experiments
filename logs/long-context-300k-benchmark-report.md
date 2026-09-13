# 300K+ Long Context: YaRN Configuration, VRAM Budget, and the Decode-Scaling Verdict

**Date: 2026-09-11 (late).** Follows `logs/long-context-120k-benchmark-report.md` and the three fixes
that landed the same day (radix top-k, `-ub 2048`, `-lzm off`).

This pass started as "push the validated 116K setup out to 300K" and was redirected mid-flight by a
much stricter requirement: **decode throughput must stay within 5% at 300K and 25% at 600K of its
short-context value.** The groundwork below (YaRN, VRAM) is still the prerequisite for running at
that depth at all, but the headline result is a scaling analysis, written up in full in
**`docs/research/12-decode-depth-scaling.md`** — read that for the argument; this file is the
configuration and measurement record.

> **SUPERSEDED IN PART (2026-09-12) — results 1 and 2 below are retracted. See
> `docs/research/15-yarn-and-long-context-rope.md`, and the annotation on the VERDICT box in
> §4.1.1.** The collapse does not reproduce on the next day's binary: the exact arm in result 1,
> re-run, gives 974 characters of correct on-task reasoning. Result 2's premise is also gone —
> 300K *does* have a validated rope strategy, namely **plain unscaled extrapolation, now measured
> coherent at 305 759 and 612 689 tokens**, and the `beta_fast`/`beta_slow` sweep result 2 wanted
> has been run (no setting makes YaRN worth enabling). Results 3 and 4, and all of §1-§3, stand.

**Four results up front:**

1. ~~**`--rope-scaling yarn` destroys this model, below the trained context length as well as past
   it.**~~ **RETRACTED.** One variable, two outcomes on the same 116K prompt and otherwise
   identical config: YaRN on -> 199 repetitions of `/`; YaRN off -> a correct, on-task refactor
   plan. That is what was measured, and the note that the short-prompt A/B could not have caught it
   is still right — but the result does not reproduce on a later binary, so the *cause* was not the
   rope flags. YaRN is still the wrong flag here, for the smaller reason that it costs 1.24x
   perplexity at depth 8 192 and 1.9-2.9x at 32 768 and buys nothing. §1.3, §4.1.1-4.1.3 and
   doc 15.
2. ~~**300K therefore has no validated rope strategy yet**~~ **RESOLVED: it does — no rope flags at
   all.** Everything else it needs was already in place here: a working VRAM budget
   (`-ub 2048 -ncmoe 32 -c 311296`, 1 472 MiB spare, §2.3), a real 305 801-token prompt (§3), and
   drivers. Both options this section listed have since been taken: unscaled extrapolation past
   262 144 is **measured coherent** at 305 759 and 612 689 tokens, and the
   `beta_fast`/`beta_slow`/`attn_factor` sweep found no setting that beats not using YaRN. doc 15
   §3-§5.
3. **The machine runs fine at 300K even though the model did not**: prefill 255.47 tok/s, TTFT
   19 min 57 s, decode 5.768 tok/s, landing 0.8% from a pre-registered component-model prediction.
   §4.1.
4. **Decode misses the 5%/25% bar by a wide margin and already misses it at 32K** (+26.8%). The
   why, and what would fix it, is doc 12. §4.

---

## 1. YaRN — the configuration, and proof it actually registers

### 1.1 The factor

The GGUF ships **no rope-scaling metadata at all** (confirmed earlier today by direct header parse:
only `qwen4exp.context_length = 262144`, `rope.freq_base = 1e7`, `rope.dimension_count = 64`,
`rope.dimension_sections = [11,11,10,0]`). `src/llama-model.cpp` therefore defaults
`rope_scaling_type_train = LINEAR`, `rope_freq_scale_train = 1.0`,
`n_ctx_orig_yarn = n_ctx_train = 262144`, and `src/llama-context.cpp:164-174` sets
`yarn_ext_factor = 0` — **YaRN is entirely off and nothing turns it on implicitly.** Past 262 144
`llama-context.cpp:326` only *warns*; it extrapolates RoPE unscaled, which is the silent
quality-loss failure mode, not a crash.

`--rope-scale N` sets `rope_freq_scale = 1/N` (`common/arg.cpp:2360-2363`), and YaRN's internal
`factor` is `1.0f / cparams.rope_freq_scale` (`llama-context.cpp:180`). So **N is exactly
`n_ctx / n_ctx_orig`**:

| target `-c` | `--rope-scale` | exact? |
|---|---|---|
| 262 144 | 1.0 (i.e. don't) | — |
| 307 200 (300K) | **1.171875** | yes |
| **311 296** | **1.1875** | yes |
| 327 680 (320K) | 1.25 | yes |
| 614 400 (600K) | 2.34375 | yes |
| 1 048 576 (1M) | 4.0 | yes — this is doc 08 §7.3's value, and it is the **1M** value, not a 300-600K one |

Recommended flags for the ~300K target: **`--rope-scaling yarn --rope-scale 1.1875
--yarn-orig-ctx 262144 -c 311296`**. `--yarn-orig-ctx 262144` is redundant (it defaults to
`n_ctx_train`) but is worth passing explicitly so the intent survives a future GGUF that ships
different metadata.

### 1.2 Proof it is not silently ignored

This needed care, because **no log line prints the effective rope-scaling type or
`yarn_ext_factor`**: the load banner's `rope scaling = linear` is `hparams`' *trained* value and is
printed before any `cparams` override, and `llama_context`'s banner prints only `freq_base` and
`freq_scale` (`src/llama-context.cpp:314-315`). A banner reading "linear" therefore proves nothing
either way — exactly the class of trap this project has hit before.

Two independent checks were used instead. (Results in §1.3.)

1. **Source chain, read end to end.** `--rope-scaling yarn` sets
   `params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_YARN`; `llama-context.cpp:164-166` keeps it
   (the `UNSPECIFIED -> hparams` fallback is skipped because it is specified);
   `:172-174` then sets `cparams.yarn_ext_factor = 1.0f` because the type is YARN; that reaches
   `ggml_rope_multi` via `ext_factor` (`src/models/qwen4exp.cpp:1004-1010`), and the SYCL
   `rope_multi` kernel passes it into `rope_yarn` (`ggml/src/ggml-sycl/rope.cpp:20-28, 96, 154`).
   No branch on the path drops it.
2. **Behavioural A/B, `staging/work/rope_yarn_ab.sh`.** Three greedy runs, identical seed, prompt
   and config, differing only in the rope flags: **(A)** none, **(B)** `--rope-scaling linear
   --rope-scale 1.1875`, **(C)** `--rope-scaling yarn --rope-scale 1.1875`. B vs A isolates
   `--rope-scale`; **C vs B isolates the scaling *type* alone**, since `freq_scale` is identical in
   both — the only difference reaching the kernel is `yarn_ext_factor`. If C's output matched B's
   token for token, the type would be a no-op.

### 1.3 Result: YaRN registers, and the type is doing real work

`logs/decode-scaling/rope-yarn-ab.log`, three arms, `-c 8192`, greedy, `--seed 1`, same prompt
("Recite the alphabet backwards from Z to A, then count from 100 down to 90."), 64 tokens.

| arm | flags | `llama_context: freq_scale` | first 64 tokens of reasoning |
|---|---|---|---|
| **A** | none | **1** | "...Need produce final with alphabet backwards and numbers 100 to 90 inclusive. Ensure no extra? Could be simple. Need maybe line breaks. Let's" |
| **B** | `--rope-scaling linear --rope-scale 1.1875` | **0.842105** | "...Need produce final likely just sequence. Need ensure backwards alphabet Z to A then numbers 100 down to 90 inclusive. Could format maybe line. Need be careful" |
| **C** | `--rope-scaling yarn --rope-scale 1.1875` | **0.842105** | "...Need produce final with alphabet backwards and numbers 100 to 90 inclusive likely. Ensure no extra? Could be simple. Need maybe line breaks. Let" |

Both banner values are exactly `1/1.1875 = 0.8421053`, so **`--rope-scale` registers.** And
**B != C at byte level with identical `freq_scale`** — the only difference that reaches the RoPE
kernel between those two arms is `yarn_ext_factor` (0 vs 1), so **the scaling *type* registers too.
YaRN is live, not silently ignored.**

Note that every arm also prints `print_info: rope scaling = linear` and
`n_ctx_orig_yarn = 262144` unchanged, because those are `hparams`' trained values. **Do not use the
load banner to check whether YaRN is on — it always says linear on this GGUF.** Use `freq_scale`
in the `llama_context` block, plus the A/B if the type itself is in question.

A third observation, and a reassuring one: **C is much closer to A than B is.** That is textbook
YaRN behaviour rather than a sign of the flag doing nothing — YaRN interpolates only the
low-frequency RoPE dimensions and leaves the high-frequency ones near-unscaled, so at short
positions it stays close to the unscaled model, whereas pure linear scaling (B) perturbs every
dimension uniformly and drifts further. Which is exactly the property that makes YaRN, and not
`--rope-scaling linear`, the right choice here.

---

## 2. VRAM budget at 300K+, with today's fixes included

### 2.1 `-lzm off` does not touch VRAM

Checked rather than assumed, since `-lzm off` makes the 27.5 GiB `per_layer_token_embd` PLE table
eager. Both 116K runs report the **same** `SYCL0 model buffer size = 26916.16 MiB` at `-ncmoe 24`
(`logs/long-context-120k-ub2048/server.log` with `-lzm auto`,
`logs/long-context-120k-lazyoff/server.log:217` with `-lzm off`); the table lands entirely in
`SYCL_Host model buffer size`, which goes **23 772 -> 51 238 MiB**. So the PLE cost is host RAM
only — but note it is *pinned USM host* memory under `-lm none`, counted against the container's
90 GiB `mem_limit` and invisible in `RssAnon`.

### 2.2 The model the probe is checking against

Reconciled exactly against the measured 116K config
(`logs/long-context-120k-lazyoff/server.log:217-257`: 26 916.16 + 810.00 + 303.75 + 112.57 +
3 692.28 = **31 834.76 MiB**, which is the known-good top of the envelope):

```
VRAM(D, ncmoe, ub) = 3 816.16                       dense core
                   + 46 200 - 962.5 x ncmoe         GPU-resident experts
                   + 112.57                         recurrent (GDN) state
                   + 9 504 x D / 2^20               both KV caches, q4_0
                   + compute_buffer(D, ub)
compute_buffer(D, 2048) ~ 0.031250 x D - 148   MiB  (fit: 122880->3692.28, 163840->4972.28)
compute_buffer(D, 1024) ~ 0.015625 x D - 148   MiB  (163840->2391.14, half-slope, consistent)
```

Card envelope, measured earlier today: **31 835 MiB loads, 32 101 MiB OOMs.** Predictions at
`D = 311 296`:

| `-ub` | `-ncmoe` | model | KV (attn+idx) | RS | compute | total | verdict |
|---|---|---|---|---|---|---|---|
| 2048 | 32 | 19 216 | 2 051 + 769 | 113 | ~9 580 | **31 729** | predicted to fit, tight |
| 2048 | 33 | 18 254 | 2 051 + 769 | 113 | ~9 580 | **30 767** | predicted to fit |
| 1024 | 28 | 22 066 | 2 051 + 769 | 113 | ~4 716 | **29 715** | predicted to fit |

Note how far this has moved from `docs/research/08` §4.1's "300K `q4_0` -> `-ncmoe 24`": that table
was computed at `-ub 512`, where the compute buffer at 300K is ~2 GiB rather than ~9.6 GiB.
**`-ub 2048` costs 8-9 `-ncmoe` steps at this depth.** Doc 08's number was not wrong, it was for a
different ubatch.

### 2.3 Probe result: all three load, and the compute-buffer prediction was 10% conservative

`logs/decode-scaling/vram-probe-300k.log`, `-c 311296` with YaRN, `-fit off`, trivial prompt,
`-lv` on. **All three configurations loaded and generated.**

| `-ub` | `-ncmoe` | SYCL0 model | KV attn | KV idx | RS | compute | SYCL0 self | free | predicted |
|---|---|---|---|---|---|---|---|---|---|
| 2048 | **32** | 19 216.16 | 2 052.00 | 769.50 | 112.57 | **8 716.28** | **30 866** | **1 472** | 31 729 |
| 2048 | 33 | 18 253.66 | 2 052.00 | 769.50 | 112.57 | 8 716.28 | 29 904 | 2 434 | 30 767 |
| 1024 | 28 | 23 066.16 | 2 052.00 | 769.50 | 112.57 | **4 279.14** | 30 279 | 2 059 | 29 715 |

(MiB. `SYCL0 self` and `free` are from `common_memory_breakdown_print`; a further 317 MiB shows as
`unaccounted` in every arm.)

- **KV and model terms reproduce the §2.2 arithmetic exactly** — 2 052.00 + 769.50 = 2 821.50
  predicted and measured, and 19 216.16 / 18 253.66 / 23 066.16 all to the cent. The 9 504 B/token
  and 962.5 MiB/`-ncmoe`-step constants are solid at this depth.
- **The compute buffer is where the prediction was off, and in the safe direction**: 8 716.28 vs
  9 580 predicted at `-ub 2048` (-9%), 4 279.14 vs 4 716 at `-ub 1024` (-9%). The two-point fit from
  the 122 880/163 840 pair over-extrapolates; the real curve is slightly concave. Implied slope over
  the wider baseline is **0.0267 MiB/token** at `-ub 2048`, not 0.03125.
- **Recommendation: `-ub 2048 -ncmoe 32 -c 311296`**, 1 472 MiB spare. `-ncmoe 31` would come to
  31 829 MiB of `self`, which lands *on* the measured 31 835 MiB known-good boundary with the
  32 101 MiB OOM point just above — not worth the risk for one expert layer. `-ub 1024 -ncmoe 28` is
  the alternative if host-side prefill cost matters more than ubatch amortization.
- **Host RAM is the new pressure point, not VRAM.** At `-ncmoe 32` the CPU-side tensors are
  19 240.51 + 14 375.27 + 27 465.95 = **61 081 MiB**, up from ~52 GiB at `-ncmoe 24`, and under
  `-lm none -lzm off` all of it is pinned host USM (invisible in `RssAnon`). That is inside the
  container's 90 GiB `mem_limit` and inside the box's 96 GiB `available`, but the margin is now
  ~30 GiB rather than ~45.
- **`n_ctx_seq (311296) > n_ctx_train (262144) -- possible training context overflow`** is printed,
  as expected. It is a warning, not an error, and it fires regardless of whether YaRN is configured
  — another reason not to read the banner as a YaRN check.

---

## 3. The 300K prompt

Generated with the sibling project's own generator, unchanged, scaled by entity count:

```
python3 ../Qwen3.8-vLLM-KVarN-MTP-Experiments/scripts/generate_messy.py 400 > messy400.py
python3 ../Qwen3.8-vLLM-KVarN-MTP-Experiments/scripts/build_prompt.py messy400.py full_prompt.txt
```

`n = 400` (vs. 153 for the 116K prompt, 2.61x) gives **1 076 019 chars** and, tokenized with this
project's own model and tokenizer (`llama-tokenize --ids` over the UD-IQ3_XXS GGUF),
**305 750 tokens** — 1.9% over the 300K target, which is inside "close to 300K". Staged at
`staging/work/bench300k/{messy400.py,full_prompt.txt}`.

Chars/token is essentially identical to the 116K prompt (3.519 vs. 3.540), which is the expected
result for the same generator and confirms the scaling was proportional rather than accidental.

---

## 4. Decode scaling — the actual deliverable

Full argument and derivations: **`docs/research/12-decode-depth-scaling.md`**. In one table, the
measured decode-vs-depth curve on the recommended 116K config
(`llama-bench -p 0 -n 32 -d ... -r 2`, `-ncmoe 24 -ub 2048 -lm none -lzm off`):

| depth | tok/s | ms/token | slowdown vs. 8K |
|---|---|---|---|
| 0 | 26.48 ± 0.45 | 37.76 | — |
| 2 048 | 26.03 ± 0.04 | 38.42 | — |
| **8 192** | **24.63 ± 0.02** | **40.60** | reference |
| 32 768 | 19.43 ± 0.09 | 51.47 | **+26.8%** |
| 65 536 | 15.26 ± 0.04 | 65.53 | **+61.4%** |
| 118 016 | 11.22 ± 0.01 | 89.13 | **+119.5%** |

Bar: <=5% at 300K, <=25% at 600K. **It is already missed at 32K.** The two components responsible
were measured at nine depths out to `n_kv = 614 400` and are both exactly linear; the projections
and the four-change program that would meet the bar are in doc 12 §1-§6.

### 4.1 The real 300K run

Prediction to test, made before the run: `-ncmoe 32` moves 8 more expert layers to the host than the
116K config, which on this project's own `-ncmoe`/decode datapoints costs roughly 1.1 ms/token/layer
at trivial depth, so the floor should be **~46.5 ms** rather than 36.3. Adding the components
measured at this exact depth (FA 89.86, indexer 48.52) predicts **~184.9 ms/token = ~5.4 tok/s**
decode at 300K.

**Result: the speed numbers land almost exactly on the prediction, and the output is garbage.**

| | measured |
|---|---|
| prompt tokens | **305 801** |
| prefill | **255.47 tok/s** (1 197 022.83 ms) |
| **TTFT** | **1 197.02 s = 19 min 57 s** |
| decode | **5.768 tok/s = 173.37 ms/token** (1 000 tokens) |
| `tg_3s` spread over the decode window | 5.67-5.83, i.e. **1.03x** — no paging, `-lzm off` holding |
| finish_reason | `length` |
| **output quality** | **TOTAL DEGENERACY - 999 tokens of `/`, from the very first token** |

Artifacts in `logs/long-context-300k/`.

**The timing half is a clean success and validates the whole component model at the real target
depth.** Predicted before the run from components measured at exactly this `n_kv`:
`floor 36.3 + FA 89.86 + indexer 48.52 = 174.7 ms/token`. Measured **173.37**, i.e. **0.8% error**,
at a depth 2.6x deeper than anything the model was built from. Two incidental findings fall out:

- **The floor is insensitive to `-ncmoe` as well as to depth.** The implied floor here is
  `173.37 - 89.86 - 48.52 = 35.0 ms` at `-ncmoe 32`, against 36.3 ms at `-ncmoe 24`. Eight more
  expert layers on the host cost nothing measurable at decode (n=1 each, so read this as "within
  noise", not as an improvement). Combined with §4.2's prefill result, `-ncmoe` has become a
  near-free VRAM knob at `-ub 2048`.
- **`-lzm off` holds at 300K.** `tg_3s` varies 1.03x over the decode window and the load banner
  carries no `lazy read enabled`, which is doc 11 §7.6's replacement rule for "is this number
  measuring the model or the disk".

**The quality half is an outright failure, and it is not subtle.** Greedy, `temperature 0`: the
model emits the token `/` 999 times in a row, starting from the *first* generated token. There is no
coherent prefix that degrades - the representation is broken before generation begins. Every
character in `run1.reasoning.txt` is `/`.

**This invalidates the 300K configuration for production use, and it does not invalidate the
timing.** Decode cost per token does not depend on which token is chosen, and the 0.8% agreement
with components measured independently at this depth is itself strong evidence the graph executed
correctly. So `logs/long-context-300k/` should be read as: **the machine runs at 300K, the model
does not.**

### 4.1.1 Isolating the cause

What is already ruled out:

- **Not the YaRN flags as such.** §1.3 arm C ran the identical flags at `-c 8192` and produced
  coherent, on-task output. YaRN does not break generation at positions the model was trained for.
- **Not paging or host-memory pressure.** 0 major faults, `tg_3s` flat to 1.03x, box `available`
  35 GiB throughout.
- **Not an allocator or OOM problem.** §2.3's probe passed and the run completed with 1 472 MiB
  spare.

What remains, and the test that separates them: either **(a) depth past `n_ctx_train` = 262 144**,
where YaRN applied to a model that ships *no* YaRN metadata - i.e. one never trained or finetuned
with it - is a heuristic with no guarantee, or **(b) the 300K configuration itself** (`-ncmoe 32`,
or `-c 311296` interacting with something at scale). The decisive arm is the **known-good 116K
prompt run through the exact 300K server config**: 116 277 tokens is comfortably under 262 144, so
coherent output there means the config is sound and the fault is depth; garbage there means the
config is at fault and the 300K numbers describe a broken setup.

That arm is `staging/work/diag_300kcfg_116kprompt.sh` ->
`logs/long-context-300kcfg-116kprompt/`. **Result: also garbage** - `b` followed by 199 `/`, same
collapse, on a 116 277-token prompt that is comfortably inside the trained context and that this
project has run coherently three times today.

**So it is (b): the configuration, not the depth.** Depth past `n_ctx_train` is exonerated, and the
300K timing numbers describe a setup that was already broken at 116K.

Three variables separate this config from the known-good 116K one
(`-ncmoe 24 -ub 2048 -c 122880 -lm none -lzm off`): `-c 311296`, `-ncmoe 32`, and **YaRN**. Ranking
them:

1. **YaRN, by a wide margin.** §1.3's arm C exonerates YaRN only at *short* prompts, and on
   reflection it could not have done more than that: rope scaling multiplies the *position*, so at
   arm C's ~75-token prompt the difference between scaled and unscaled angles is negligible, while
   at 116K it is enormous. **Arm C proves the flags parse and reach the kernel; it proves nothing
   about their effect at depth.** A model that ships no rope-scaling metadata was never trained or
   finetuned with YaRN, so applying it is an unguaranteed heuristic - and this is evidence that on
   this model it is not merely unnecessary below 262 144, it is **actively destructive**.
2. `-ncmoe 32`. Moves expert tensors between host and device but computes the same weights; a
   catastrophic numerical difference would be a backend bug, not a configuration effect.
3. `-c 311296`. Sizes the KV cache; should be numerically inert.

Single-variable test: the same 116K prompt, the same `-c 311296 -ncmoe 32 -ub 2048`, **YaRN
removed**. `staging/work/diag_yarn_off.sh` -> `logs/long-context-300kcfg-noyarn/`.

**Result: fully coherent.** *"We need answer user's request: produce cleaned-up, well-structured
version of massive legacy Python script, fix listed issues, explain key changes briefly at end...
Need ensure code is modern idiomatic Python, type hints, docstrings, context managers, no
gl[obals]"* - 971 characters of on-task reasoning, correctly identifying the task, the duplication
problem and the requested fixes. Prefill **375.64 tok/s**, decode **10.78 tok/s (92.77 ms/token)**.

> ## RETRACTED (2026-09-12) — see `docs/research/15-yarn-and-long-context-rope.md`
>
> **This verdict does not reproduce and should not be relied on.** The exact arm below --
> `llama-server`, `--rope-scaling yarn --rope-scale 1.1875 --yarn-orig-ctx 262144`,
> `-c 311296 -ncmoe 32 -ub 2048`, the same 116,277-token prompt, the same client at
> `temperature 0` -- was re-run on the next day's binary and produced **974 characters of
> correct, on-task reasoning** naming all 20 entity types, against `b` + 199 `/` here
> (`logs/yarn/server-repro-yarn/`). YaRN at **305,759 tokens**, genuinely past `n_ctx_train`,
> is coherent as well. `test-backend-ops -o ROPE` passes 470/470 on SYCL0 including the
> `ext_factor != 0` IMROPE cases, and all four `ggml_rope_multi` call sites in `qwen4exp.cpp`
> were read and pass identical YaRN parameters, so there is no rope bug of the kind this
> section's follow-up hypotheses proposed (partial RoPE / mrope sections / quantization --
> all three disposed of in doc 15 sections 2.2 and 7).
>
> **The one variable that changed is the binary**: these runs predate the block-granularity
> QSA top-k (21:45) and the pooled indexer-key cache (23:13), both landing inside
> `build_qsa_top_k` -- the function holding the indexer's own RoPE application and the
> `ggml_top_k` that decides which 2,051 of `n_kv` cells attention is allowed to see.
>
> **What survives**: YaRN is still the wrong flag here, for a smaller reason. It costs
> **1.24x perplexity at depth 8,192 and 1.9-2.9x at 32,768** and buys nothing that unscaled
> extrapolation does not already give (doc 15 sections 3 and 4). And the methodological
> caveat this section raised is not just right, it is bigger than stated: the indexer's hard
> top-k makes output quality a *discontinuous* function of the rope parameters, and identical
> configs re-measure 12-33% apart. **The original single-run verdict was not safe, and neither
> is any single-run replacement.**
>
> ## ORIGINAL VERDICT (retracted, kept for the record): `--rope-scaling yarn --rope-scale N` destroys this model, and it does so *below* the
> trained context length, not just past it.
>
> One variable, two outcomes, same prompt and same everything else: **YaRN on -> 199 repetitions of
> `/`. YaRN off -> a correct, on-task refactor plan.** `-ncmoe 32` and `-c 311296` are both
> exonerated. **Do not set `--rope-scaling yarn` on this GGUF.**

**This closes the YaRN gap this plan has carried all day - with a negative result, which is the more
useful kind here.** The flags parse, reach the SYCL `rope_yarn` kernel and change `freq_scale`
exactly as documented in §1; they simply produce a model that cannot read its own context. A GGUF
that ships no rope-scaling metadata was never trained or finetuned with YaRN, and this is a direct
measurement of what that costs. A plausible additional contributor, worth checking before anyone
retries: this model uses **partial RoPE** (`rope.dimension_count = 64` of a 128-wide head) with
**mrope sections `[11,11,10,0]`**, and YaRN's ramp (`beta_fast`/`beta_slow` over
`rope.dimension_count`) may simply not be meaningful under that layout - none of
`--yarn-attn-factor`, `--yarn-beta-fast` or `--yarn-beta-slow` was swept.

**Consequence for the 300K target: there is currently no validated rope strategy past 262 144.**
Everything else needed is now in place and validated (VRAM budget §2.3, a real 305 801-token prompt
§3, harness and drivers), but the positional-encoding question is reopened, not closed. The two
untried options are (i) run unscaled past `n_ctx_train` and measure the quality of plain
extrapolation - llama.cpp only warns, and this is now the *less* damaged-looking of the two paths -
and (ii) sweep YaRN's own parameters rather than accepting the defaults.

### 4.1.4 The corrected floor, and what the degenerate runs cost us

The no-YaRN arm is also the first *healthy* decode measurement at `-ncmoe 32`, which lets §4.1.2's
suspicion be settled rather than left hanging:

| run | `-ncmoe` | output | ms/token | implied floor |
|---|---|---|---|---|
| `logs/long-context-120k-lazyoff/` | 24 | coherent | 89.72 | **36.3** |
| `logs/long-context-300kcfg-noyarn/` | 32 | coherent | **92.77** | **39.5** |
| `logs/long-context-300kcfg-116kprompt/` | 32 | degenerate | 80.10 | 26.8 (**artifact**) |

**The floor rises 3.2 ms for 8 more host-resident expert layers - 0.4 ms/layer, sensible and
small - and the degenerate run's 26.8 ms was indeed an artifact of repeating one token into the
same 10-of-512 experts every step.** §4.1.2's caveat is confirmed and now quantified: degeneracy
flattered that arm by ~13 ms/token.

Re-deriving the 300K prediction on the healthy `-ncmoe 32` floor:
`39.5 + 89.86 + 48.52 = 177.9 ms/token = 5.62 tok/s`, against the degenerate run's measured 5.77.
**So the headline 300K decode figure of ~5.6-5.8 tok/s stands**, but it should be re-measured on a
working configuration before it is quoted as final.

Prefill, same comparison: **375.64 tok/s at `-ncmoe 32`** against 394.14 at `-ncmoe 24`, both cold -
a **4.7% loss for a third more host-resident experts**, confirming §4.2's finding that `-ncmoe` is
nearly free at `-ub 2048`. (The 802.82 tok/s in the degenerate arm was a *warm* second request on an
already-running server, not a comparable number - and a reminder of the project's standing
discard-the-first-arm rule, here operating in reverse.)

### 4.1.2 Two caveats this diagnostic raises about the timing numbers

- **Degenerate output probably flatters the decode numbers.** Emitting the same token every step
  routes to the *same* 10-of-512 experts in every CPU-resident layer on every step, collapsing the
  host expert working set from ~25 GiB of random access to a few megabytes that stay in cache. Both
  degenerate runs should therefore be read as **upper bounds** on healthy-generation decode. The
  116K arm is the direct evidence: **80.10 ms/token at `-ncmoe 32`** against the component model's
  89.6 and the healthy `-ncmoe 24` run's 89.72 - i.e. 11% *faster* while running 8 more expert
  layers on the host, which is the wrong direction for everything except this explanation.
- **That in turn makes the 300K run's 0.8% agreement with the component model partly luck.** The
  model predicted 174.7 ms/token and 173.37 was measured, but the 116K arm at the same `-ncmoe`
  shows the floor can move ~9 ms under degeneracy. The honest statement is that
  **`FLASH_ATTN_EXT` and the indexer curves are measured directly at these depths and stand on
  their own** (§4 and doc 12 §1); the *end-to-end* 300K number is corroborating rather than
  confirming, and wants a re-run once the configuration is fixed.

### 4.1.3 An operational bug found while doing this, affecting every driver in this project

`docker compose run` **ignores `container_name`** from the compose file - it creates
`docker-<project>-<service>-run-<hash>`. So the `docker stop qwen4exp-test-sycl` at the end of
`staging/work/run_116k_*.sh`, `run_300k.sh` and the rest has been **silently failing**, leaving the
server container running after every benchmark. It surfaced here as the diagnostic's own server
failing to bind port 8090 (`failed to set up container networking`) and the client then talking to
the *previous* run's still-live server. That was harmless this time - the still-live server was the
exact config under test - but it is a live footgun: a driver can silently benchmark the previous
run's configuration. **Fix: capture the container id from `docker compose run -d` (as
`diag_yarn_off.sh` now does) and stop that, or `docker compose down`, and always
`docker ps | grep sycl` before trusting a run.**

A third hypothesis worth keeping on the list if (a) is confirmed: **UD-IQ3_XXS** is an aggressive
3.06 bpw quantization, and YaRN's whole mechanism is a fine-grained reweighting of RoPE frequency
bands. A quantization-induced failure would plausibly show up first exactly where the positional
encoding is being asked to extrapolate. Testing that needs UD-Q3_K_XL, which this project has
downloaded but has never run at long context.

### 4.2 Prefill also decays with depth, and by a similar mechanism

Read out of the same run's `print_timing` progress lines (`logs/long-context-300k/server.log`),
which report cumulative tokens/s every 2 048-token ubatch:

| `n_kv` reached | elapsed | cumulative tok/s | **marginal tok/s** |
|---|---|---|---|
| 2 090 | 5.6 s | 373.7 | 373.7 |
| 22 570 | 49.9 s | 452.1 | **461.9** |
| 43 050 | 98.4 s | 437.3 | 422.2 |
| 63 530 | 151.2 s | 420.3 | 388.5 |
| 84 010 | 207.6 s | 404.7 | 363.0 |
| 104 490 | 268.0 s | 389.9 | 339.1 |
| 124 970 | 341.7 s | 365.7 | 277.8 |
| 141 354 | 406.9 s | 347.4 | **251.2** |

Marginal prefill throughput falls **1.84x from 22K to 141K** (461.9 -> 251.2 tok/s), i.e. per-token
prefill cost rises from 2.16 ms to 3.98 ms. Same two `O(n_kv)` terms, just amortized across a
2 048-token ubatch instead of paid by a single token, which is why prefill degrades far more gently
than decode's 2.2x over a similar range. The `-lzm off` + `-ub 2048` wins are intact here.

One incidental result worth recording: **prefill is much less sensitive to `-ncmoe` than expected.**
This run is at `-ncmoe 32` (8 more expert layers on the host than the 116K run's `-ncmoe 24`) and
still reaches **365.7 tok/s cumulative at ~125K**, against the 116K run's 394.14 at 116K — a ~7%
loss for a third more host-resident experts. At `-ub 2048` the per-ubatch expert-bank sweep is
amortized well enough that expert placement is no longer the dominant prefill term.
