# YaRN, and What Unscaled Extrapolation Actually Costs at 300K-600K

Date: 2026-09-12. Binary: `staging/devbin` as of 01:59/02:04 today (the build
that carries radix top-k, block-granularity QSA top-k, the pooled indexer-key
cache and sparse FA). Model: `Qwen3.8-Flash-Next-UD-IQ3_XXS`.

This document answers two questions the project had left open and coupled
together:

1. **Why does `--rope-scaling yarn` produce garbage on this GGUF?**
   `logs/long-context-300k-benchmark-report.md` §4.1.1 concluded, from two
   single-variable server arms, that "`--rope-scaling yarn --rope-scale N`
   destroys this model, and it does so *below* the trained context length".
2. **What does output quality look like at 300K-600K via plain unscaled
   extrapolation** — the path every 300K/600K speed number in this repo
   actually took, by setting a large `-c` and letting llama.cpp warn and
   extrapolate past `n_ctx_train = 262144`?

The short answers, up front, because they reverse a standing verdict:

> **Question 2 first, because it is the one that matters and the answer is
> good.** At **305,759 real tokens — 16.6% past the native ceiling — with no
> rope flags at all, the model is fully coherent and quantitatively accurate.**
> It named all 20 entity types in the generator's own order, identified the
> "400+ near-identical function groups" structure, and recovered the *ranges*
> of every randomised parameter (multiplier 2-9, addend 1-50, divisor 1-7, tax
> 0.10-0.9, cache-key default 0-399) — every one of which matches
> `generate_messy.py` exactly, and none of which can be read off a local
> window, because they are the min/max of 400 independent draws spread through
> a 1.07 MB prompt. **And it holds at the other end of the band: at 612,689
> tokens — 2.34x the native ceiling — it is still coherent**, gets the 800-block
> count right, lists all 20 entities in order, and recalls all six of the
> generator's global variable names from the first ~100 tokens of a 2.15 MB
> prompt. **Unscaled extrapolation across the whole 300K-600K target is not
> degraded. The 21.08 tok/s at 300K and 12.42 tok/s at 600K describe output
> somebody would actually want.**
>
> **Question 1: the collapse does not reproduce, and there is no bug in
> llama.cpp's YaRN.** The same flags the verdict was written against
> (`--rope-scaling yarn --rope-scale 1.1875 --yarn-orig-ctx 262144`) at
> **305,759 tokens are also fully coherent** — and so is **the original failing
> arm re-run verbatim** (same `llama-server`, same config, same 116,277-token
> prompt, only a newer binary): 974 characters of correct on-task reasoning
> where it previously emitted `b` and 199 `/`. The RoPE path was read end to
> end and is correct for this architecture on every point the brief suspected,
> and `test-backend-ops -o ROPE` passes 470/470 on SYCL including the
> `ext_factor != 0` IMROPE cases. What YaRN *does* cost, reproducibly, is
> **quality without collapse**: perplexity 1.24x worse at depth 8,192 and
> 1.9-2.9x worse at 32,768.
>
> **And the finding that subsumes both**: this architecture puts a **hard
> top-k context selection between RoPE and attention**, so quality at depth is
> a *discontinuous, high-variance* function of RoPE parameters. Identical
> configs re-measured 12-33% apart, and the response to `beta_fast`/`beta_slow`
> is non-monotone. **No single-run verdict about RoPE on this model is safe —
> including the original one, and including any in this document that is not
> replicated.**

---

## 1. The hypothesis the brief asked to test first, and why it is wrong on its
own terms but right about the conclusion

The brief's lead hypothesis was that YaRN "is only supposed to matter/activate
once actual sequence length exceeds `yarn_orig_ctx`", so testing it at 116K —
under the native 262,144 — applied "wrong, unnecessary scaling", and llama.cpp
failing to no-op below the threshold would be a narrow bug.

**That is not a bug, and not specific to llama.cpp.** YaRN as published is a
*static reparametrisation* of the rotary frequencies, not dynamic NTK: the
scaled frequencies apply at every position, including position 1. llama.cpp
implements exactly that — `rope_yarn()` branches on `ext_factor != 0`, never on
the current sequence length — and so does the reference. So enabling YaRN at
116K is a legitimate test of what those flags do, and the original arm was not
methodologically void on these grounds.

**The hypothesis was nevertheless pointing at something real, just one step
over.** The 116K arm conflated two things it could not separate: "YaRN is wrong
for this checkpoint" and "YaRN was asked to do something unnecessary here". The
way to separate them is not to hope the code no-ops, it is to run YaRN at a
depth where it genuinely earns its keep. §4 does that. And because static YaRN
perturbs theta *in proportion to position*, moving the test past 262,144 makes
every angle shift strictly larger — so if the collapse were positional, deeper
could only be worse. It is not worse. It is fine.

## 2. The RoPE path, read end to end: nothing the brief suspected is wrong

All of this is static analysis, no GPU, and all of it came back negative —
which is the useful outcome, because it removes four plausible bugs.

**2.1 There is no main-attention / indexer RoPE mismatch.** The brief's
strongest specific suspicion was that `build_qsa_top_k`'s separate RoPE
application on indexer keys might be YaRN-scaled differently from the main
attention layers. It is not. All four `ggml_rope_multi` call sites in
`src/models/qwen4exp.cpp` — Q and K at `:1144` and `:1150`, the indexer's
pooled keys at `:890` inside `build_pool`, and the indexer queries at `:927` —
pass the *same* `n_rot`, `sections`, `rope_type`, `n_ctx_orig`, `freq_base`,
`freq_scale`, `ext_factor`, `attn_factor`, `beta_fast`, `beta_slow`.

Positions are in the same units on both sides too, which was the other way this
could have gone wrong: the pooled keys rope at `bid_idx[b]`, which is
documented and implemented as the block's **first token position**
(`src/llama-memory-hybrid-idx.h:45`, `.cpp:837` = `pb*r`), *not* a block
ordinal. Had it been a block ordinal, the indexer's q and k would have been
4x out of register — invisible at `ext_factor = 0` but not under YaRN.

**2.2 `dimension_sections = [11,11,10,0]` does not interact with YaRN, and does
not matter for text at all.** `rope_multi`'s sections only choose *which
position row* a frequency band reads; the frequency itself is still
`theta_scale^(iw/2)` over the global dim index, which is exactly the index
`rope_yarn_ramp()` uses. So YaRN's ramp lands on frequency bands, and the bands
are globally indexed — sectioning is orthogonal. Moreover, for a **text** batch
`src/llama-batch.cpp:781-788` broadcasts one position into all four rows, so
imrope degenerates to plain NeoX rope and all four sections read the same value.
`n_offs` is 0 for `ggml_rope_multi` (`ggml/src/ggml.c:4313`), so the partial
rope is simply dims `[0,64)` of each head on both sides.

**2.3 `yarn_attn_factor` is not double-applied.** `src/llama-context.cpp:196`
sets it to `get_mscale(factor, 1.0f)` and then `:210` divides by the identical
expression, precisely because the kernel re-applies it — landing at exactly 1.0
before `hparams.rope_attn_factor`. The kernel's own
`mscale *= 1 + 0.1*ln(1/freq_scale)` is the single application.

**2.4 The SYCL kernel matches the reference, and upstream already tests it.**
`ggml/src/ggml-sycl/rope.cpp`'s `rope_yarn` and `rope_yarn_ramp` are
line-for-line the CPU/CUDA math, and `ggml_sycl_op_rope_impl` plumbs
`ext_factor`, `attn_factor`, `beta_fast`, `beta_slow` and `corr_dims` through
without dropping anything. `tests/test-backend-ops.cpp:10410-10454` already
sweeps `ext_factor = {0, 0.7465} x freq_scale = {1, 1.4245} x attn_factor =
{1, 1.4245}` over `GGML_ROPE_TYPE_IMROPE` at partial `n_dims` (20 and 32 of a
128-wide row) against the CPU reference. **Run on this tree:
`test-backend-ops test -b SYCL0 -o ROPE` -> 470/470 pass**
(`logs/yarn/server-repro.log`). So the SYCL rope kernel is provably not where
any YaRN divergence lives.

**2.5 The numbers YaRN actually applies to this GGUF.** Replicated from scratch
(`staging/work/yarn_math.py`) using the GGUF's own metadata (`freq_base = 1e7`,
`rope.dimension_count = 64`, `context_length = 262144`, and — confirmed again by
direct header parse — no rope-scaling metadata whatsoever):

`ggml_rope_yarn_corr_dims(64, 262144, 1e7, 32, 1)` = **[14, 22]** of 32 channel
pairs. Pairs 0-14 fully **extrapolated** (theta untouched), pairs 22-31 fully
**interpolated** (theta x 0.842105 at `--rope-scale 1.1875`), 15-21 on the ramp.
`mscale = 1.01719`. Proportionally this is close to what YaRN does to llama2
(there, [20,45] of 64 pairs), and the interpolation boundary sits at a
wavelength of 408,019 tokens against a 262,144 native context — i.e. YaRN is
interpolating exactly the bands that are longer than the trained context. **The
default betas are not mis-tuned for `freq_base = 1e7`.**

## 3. YaRN's real cost, measured: quality, not collapse

`llama-perplexity` with `--chunks 1`, so the chunk size *is* the depth. The tool
scores the deepest half of its window (`tools/perplexity/perplexity.cpp:542`,
`first = n_ctx/2`), so every scored token sits at depth >= `-c`/2. Corpus:
`staging/work/bench120k/full_prompt.txt` (116,225 tokens). Config
`-ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -ncmoe 24 -ub 2048 -lzm off`, mmap.
Driver `staging/work/yarn_ppl_ladder.sh`, log `logs/yarn/yarn-ppl-ladder.log`.

| `-c` | scored positions | YaRN off | YaRN 1.1875 | penalty |
|---|---|---|---|---|
| 8,192 | 4,096 - 8,191 | **10.6070** +/- 0.560 | **13.1616** +/- 0.728 | **1.24x** |
| 32,768 | 16,384 - 32,767 | **43.1005** +/- 1.484 | **82.6182** +/- 3.034 | **1.92x** |
| 32,768 (re-run, §5) | same | **38.1926** +/- 1.333 | **110.0320** +/- 4.047 | **2.88x** |

Both arms at a given depth score the identical token set, so the within-depth
ratio is clean even though the absolute values are not comparable across depths
(different scored regions, and a `indexer_top_k = 2048` budget that covers a
smaller fraction of context as depth grows).

**Two things follow.** First, the penalty is real and far outside the error
bars: YaRN measurably degrades this checkpoint. Second, it **grows with depth**,
which is the signature of a positional perturbation and is exactly what §2.5's
arithmetic predicts. Note also that the project's existing "YaRN is fine at
short prompts" control (300K report §1.3 arm C, ~75 tokens) is consistent: at
75 tokens the angle shifts are ~0 *and* `n_kv < indexer_top_k`, so there is no
selection to disturb either.

**What does not follow is collapse.** A 1.9-2.9x perplexity increase is a
serious quality loss. It is not 999 repetitions of `/`.

## 4. The arm the brief asked for: YaRN at a depth that genuinely exceeds
`n_ctx_orig`

Perplexity cannot reach past the native ceiling at all, for a reason worth
recording because it will catch the next person:
`tools/perplexity/perplexity.cpp:514` reserves `n_ctx * n_vocab` **floats of
host memory** for one chunk. At `-c 262144` that is 262,144 x 151,936 x 4 B =
**159 GB**, and the tool dies with `std::bad_alloc` after tokenizing (measured,
`rc=134`, `staging/work/yarn-ppl-262144-off.out`). The ceiling is ~`-c 98304`
on a 123 GiB box regardless of VRAM. **Perplexity is structurally unavailable
as a long-context quality metric for this model.**

So the currency is a real greedy generation. Prompt:
`staging/work/bench300k/chat_prompt.txt` — the sibling generator at `n = 400`
wrapped in this model's own chat template, **305,759 tokens**. Config
`-ncmoe 32 -ub 2048 -c 311296 -fa 1 -ctk q4_0 -ctv q4_0 -lzm off` (mmap),
`--temp 0 --seed 42 -n 250`, via `llama-completion` because `llama-server`
refuses the prompt (§5.2). Drivers `staging/work/yarn_deep.sh`,
`yarn_deep2.sh`; artifacts `staging/work/yarn-gen-*.txt`,
`logs/yarn/gen-off-305k.continuation.txt`.

| arm | rope flags | prefill | decode | output |
|---|---|---|---|---|
| `off` | none (unscaled extrapolation) | 253.55 tok/s | 6.57 tok/s | **coherent, most precise** |
| `yarn` | `yarn --rope-scale 1.1875 --yarn-orig-ctx 262144` | 243.94 tok/s | 6.42 tok/s | **coherent** |
| `nearunity` | `yarn --rope-scale 1.0001 --yarn-orig-ctx 311296` | 265.44 tok/s | 12.01 tok/s | **coherent, most precise** |

**All three are coherent.** Every arm correctly identified the task, the
"400+ near-identical function groups" structure, the full function-group
membership (`process_X_N`, `validate_X_N`, `calculate_X_total_N`,
`format_X_name_N`, `get_X_by_id_N`, `update_X_N`, `delete_X_N`, `XHandlerN`),
and **all 20 entity names in `generate_messy.py`'s own declaration order**.

The `off` and `nearunity` arms additionally recovered the *ranges* of the
randomised parameters, and every one is exactly right against the generator
(`generate_messy.py:165-169`):

| model said | generator |
|---|---|
| multiplier "2-9" | `random.randint(2, 9)` |
| addend "1-50" | `random.randint(1, 50)` |
| divisor "1-7" | `random.randint(1, 7)` |
| tax "0.10-0.9" / "0.10-0.90" | `0.{random.randint(5, 25)}` -> literals 0.10 .. 0.9 |
| cache-key default "0-399" | `obj.get('id', {idx})`, idx 0..399 |
| "`if True`/`if False` pattern" | `random.choice(["True","False"])` |

Those ranges are the min and max of 400 independent draws distributed through
the whole prompt. Recovering them is aggregation over the full 305,759 tokens,
not local retrieval. The `yarn` arm named the same seven varying fields and
gave concrete instances (`process_user_0`, `process_order_1`,
`process_product_2`) but did **not** produce the numeric ranges — which is
consistent with §3's "YaRN costs precision, not coherence", though n=1.

### 4.1 600K, unscaled: also coherent, at 2.34x the native ceiling

The angle shifts §2.5 tracks are linear in position, so 305,759 tokens says
nothing about 600K. Measured directly: a fresh **612,689-token** prompt
(sibling generator at `n = 800`, 800 entity blocks verified by
`grep -c "def process_"`, wrapped in the model's chat template; corpus token
count verified with `llama-tokenize` at 612,680 before templating), config
`-ncmoe 38 -ub 1024 -c 614400 -fa 1 -ctk q4_0 -ctv q4_0 -lzm off` — the row
PLAN.md's 600K number already used — **no rope flags**, `--temp 0 --seed 42`.
Driver `staging/work/yarn_600k.sh`, artifacts `logs/yarn/yarn-600k.log`,
`logs/yarn/gen-off-600k.continuation.txt`.

Prefill **138.26 tok/s** (612,689 tokens in 73 min 51 s), decode **4.71
tok/s**. `llama_context` warns `n_ctx_seq (614400) > n_ctx_train (262144)` and
proceeds; the allocator fits at this config as expected.

**The output is coherent and correct on everything checkable:**

- "**~800+ functions that are near-identical copies**" — the corpus has
  **exactly 800** blocks.
- **All 20 entity types, in `generate_messy.py`'s declaration order.**
- The full eight-member group per entity (`process_X`, `validate_X`,
  `calculate_X_total`, `format_X_name`, `get_X_by_id`, `update_X`, `delete_X`,
  "and a Handler class").
- The varying parameters: multiplier, addend, divisor, tax rate, and
  "whether format uses `.upper()` or `.lower()`".
- **Every global by name** — `data`, `DATA2`, `temp`, `counter`,
  `GLOBAL_CACHE`, `errors_list` — which is exactly the generator's `HEADER`
  global set, and those names appear only at the very top of a 2.15 MB prompt,
  i.e. ~612,000 tokens behind the generation point.

It did **not** volunteer the numeric *ranges* the 305,759-token arm recovered.
That is a plausible mild degradation, but it is not evidence of one: the arm is
capped at `-n 250` and spent its budget on a longer structural analysis plus
the issue list. n=1, and per §5 that is all it is.

> **So the honest answer for the whole 300K-600K band, on the unscaled
> extrapolation path every speed number in this repo used: coherent, on-task
> and accurate at both ends.** Not "degraded but usable" — at 305,759 tokens it
> recovered aggregate statistics of the entire prompt, and at 612,689 it
> recalled six variable names from the first 100 tokens of a 2.15 MB input.
> The 21.08 tok/s at 300K and 12.42 tok/s at 600K in PLAN.md describe a model
> that is genuinely working at those depths.

### 4.2 The direct reproduction: the exact failing arm, re-run

The §4 arms change the tool (`llama-completion`) and the depth relative to the
run the verdict was written against. This arm changes neither. It is
`logs/long-context-300kcfg-116kprompt/` again — **`llama-server`,
`--rope-scaling yarn --rope-scale 1.1875 --yarn-orig-ctx 262144`,
`-c 311296 -ncmoe 32 -ub 2048 -lzm off`, the same 116,277-token prompt through
the same `bench300k/run_bench.py` client at `temperature 0`** — with one
variable changed: the binary is `staging/devbin` as of 01:59 today instead of
as of 21:19 yesterday. Driver `staging/work/yarn_server_repro.sh`; artifacts
`logs/yarn/server-repro-yarn/`.

| | original (2026-09-11 21:22) | re-run (2026-09-12 20:10) |
|---|---|---|
| prompt tokens | 116,277 | 116,277 |
| prefill | 802.82 tok/s (warm 2nd request) | 295.48 tok/s |
| decode | 12.48 tok/s (degeneracy artifact) | **16.58 tok/s** |
| output | `b` + 199 `/` | **974 chars of coherent reasoning** |

The re-run's reasoning: *"We need answer user's request: refactor huge legacy
Python script to modern idiomatic Python, fix listed issues... The original is
massive with 150 near-identical blocks for entities: user, order, product,
invoice, customer, shipment, employee, ticket, payment, account, vendor,
warehouse, review, coupon, subscription, session, device, report, category,
transaction repeated cycles. Need parameterize. Need fix global mutable state,
bare except, mutable defaults, manual loops, == None/True, string concat,
namin[g]..."* — correct task, all 20 entities, the duplication diagnosis, and
the requested fix list. ("150" against the corpus's actual 153 blocks is the
only inaccuracy.) This is the same quality as the project's known-good
*no*-YaRN control at this config, which produced 971 characters of very
similar text (`logs/long-context-300kcfg-noyarn/`).

Two residual differences from the original, stated so nobody has to guess:
this arm ran under mmap rather than `-lm none` (load mode moves where weights
live, not what they are), and the original arm's server was a leftover
container rather than a fresh one (the 300K report §4.1.3 established it was
running the identical config, so this is a difference in page-cache warmth,
which the prefill numbers show and which cannot change a logit).

> ### Verdict on question 1
> **`--rope-scaling yarn --rope-scale 1.1875` does not destroy this model.**
> The collapse does not reproduce — not at 305,759 tokens via
> `llama-completion` (§4), and not in the original's own tool, config, prompt
> and depth (§4.2). The 300K report's "destroys this model" verdict should be
> **retired** in favour of "costs 1.2-2.9x perplexity, grows with depth, and
> buys nothing".
>
> **What changed is the binary.** The degenerate runs
> (`logs/long-context-300k/`, `logs/long-context-300kcfg-116kprompt/`) used
> `staging/devbin` as it stood at 21:19 on 2026-09-11 — **before** the
> block-granularity QSA top-k (built 21:45, `staging/devbin-blk`) and the
> pooled indexer-key cache (23:13, `staging/devbin-pool`) landed in
> `build_qsa_top_k`, the function that holds the indexer's own RoPE
> application and its `ggml_top_k`. Given §5.1 — the indexer's hard top-k is
> what turns a small RoPE perturbation into a wholesale change of *which* 2,051
> cells attention sees — a pre-fix selection path is by far the most plausible
> mechanism. **It is not proven**, and §5 is the reason it may never be cleanly
> provable: proving it means rebuilding the 21:19 tree and getting a
> *reproducible* collapse out of a harness whose spread on identical configs is
> 12-33%. Not worth the GPU hours now that the current tree is measured.

> ### Verdict on question 2
> **Unscaled extrapolation to ~306K is coherent and accurate, and is the right
> strategy.** It was the best of the three arms on the one axis that separated
> them. There is no reason to reach for YaRN on this checkpoint at 300K.

## 5. The finding that limits every verdict here, including this document's

**5.1 A hard top-k sits between RoPE and attention, and it makes quality at
depth discontinuous in the RoPE parameters.** `build_qsa_top_k` selects
`width = indexer_top_k + r - 1 = 2051` cells out of `n_kv` from
`relu(q_idx . k_idx)` scores, and `build_attn_qsa` then restricts attention to
exactly those cells. At `n_kv = 305,759` the model attends to **0.7% of its own
context**, chosen by a `ggml_top_k` over scores that RoPE feeds. A small,
smooth change in theta therefore produces a *reshuffled selection*, not a small
change in the attention output.

Three independent observations line up with that, and with nothing else:

- **Identical configs do not reproduce.** `off` at `-c 32768` measured
  **43.1005** in one session and **38.1926** in another (12%); `yarn` at the
  same depth measured **82.6182** and **110.0320** (33%). Same binary, same
  corpus, same flags. This is the same phenomenon `docs/research/12` §9.5
  recorded as "greedy output on this stack is not run-to-run reproducible", and
  the same order of magnitude as the ~1.5x run-to-run spread independently
  observed at 600K depth today on unmodified code.
- **The response to `beta_fast`/`beta_slow` is non-monotone** (§5.3).
- **The original collapse and today's coherent runs are the same flags.**

**5.2 Practical consequence: never conclude anything about RoPE on this model
from one run.** The original "YaRN destroys this model" rested on n=1 per arm.
So does this document's `yarn`-at-305K coherence. The difference is that the
*direction* here is corroborated by the `off` and `nearunity` arms and by the
perplexity ladder; but a reader should treat "YaRN is coherent at 305K" as one
observation, not a law.

**5.3 The `beta_fast`/`beta_slow` sweep PLAN.md has carried as an open item,
closed — with a non-result.** Same corpus and config as §3, `-c 32768`,
`--rope-scale 1.1875`. `corr_dims` computed for each setting.
Log: `staging/work/yarn-deep2.log`, driver `staging/work/yarn_beta_sweep.sh`.

| arm | `corr_dims` (of 32 pairs) | PPL |
|---|---|---|
| no rope flags | — | **38.1926** +/- 1.333 |
| `--yarn-beta-fast 64 --yarn-beta-slow 2` | [12, 20] | **39.0299** +/- 1.339 |
| `--yarn-beta-fast 4 --yarn-beta-slow 0.125` | [18, 26] | 68.2094 +/- 2.474 |
| YaRN defaults (32 / 1) | [14, 22] | 110.0320 +/- 4.047 |
| `--rope-scaling linear --rope-scale 1.1875` | n/a (`ext_factor` 0) | 113.8389 +/- 4.367 |

`linear` being worst is the one expected ordering (it rescales every band,
including the high-frequency ones the model reads locally) and it confirms the
300K report §1.3's reason for preferring `yarn` over `linear` as the flag.
Everything else is not orderable by "how much of the rotary space gets
rescaled": [12,20] rescales *more* bands than the [14,22] default and scores
**2.8x better**, and [18,26] rescales *fewer* and lands in between.

**So: `--yarn-beta-fast 64 --yarn-beta-slow 2` came back indistinguishable from
not scaling at all (39.03 vs 38.19), and that is a lead, not a
recommendation** — n=1, against a harness whose own spread on this exact arm is
12-33%, and with no mechanism that explains the non-monotonicity except §5.1's
discontinuous selection. **And even taken at face value it only *matches*
no-scaling.** There is no YaRN configuration in this sweep that beats simply
not using YaRN, which is what makes the open item closable: the answer is "no
setting of the betas makes YaRN worth turning on here."

## 6. `llama-server` cannot serve past `n_ctx_train` — and a one-flag
workaround that does not touch the model

**6.1 The cap.** `n_ctx_slot()`
(`tools/server/server-context.cpp:4199-4207`) returns
`min(llama_n_ctx_seq(ctx), llama_model_n_ctx_train(model))`, so with
`n_ctx_train = 262144` a 300K prompt is refused however large `-c` is. Measured
directly (`staging/work/yarn-slotcap-plain.log`):

```
srv load_model: the slot context (311296) exceeds the training context of the model (262144) - capping
srv load_model: initializing, n_slots = 1, n_ctx_slot = 262144
```

The only thing in the codebase that lifts it is
`src/llama-context.cpp:3782-3786`, which rewrites
`hparams.n_ctx_train = n_ctx_orig_yarn / rope_freq_scale` — but only when the
scaling type is YARN **and** `rope_freq_scale` differs from the trained value.
That is why this repo's 300K server run had to pass `--rope-scale 1.1875`: it
was buying a slot, not a rope strategy. **Every 300K/600K number in this repo
is therefore either `llama-bench`/`llama-completion` (no gate) or that one
server run.**

**6.2 The workaround: buy the lift at a factor that is numerically nothing.**
`--rope-scaling yarn --rope-scale 1.0001 --yarn-orig-ctx 311296` satisfies the
gate (`rope_freq_scale = 0.99990001 != 1.0`) and rewrites `n_ctx_train` to
311,327. The rope it applies: interpolated bands x 0.99990, `mscale` 1.00001,
and the **largest theta shift anywhere at position 311,296 is 0.002 rad**,
against the ~1.0 rad that `--rope-scale 1.1875` puts on the same band — four
orders of magnitude smaller. Measured
(`staging/work/yarn-slotcap-nearunity.log`):

```
srv load_model: initializing, n_slots = 1, n_ctx_slot = 311296
```

No "capping" warning, no code change.

**6.3 And it is validated on output, not just on the banner.** §4's
`nearunity` arm is the same 305,759-token prompt, same seed, same config as the
no-flags arm, with these flags added: **coherent, and the joint-best of the
three arms on parameter-range recovery.** So the flag combination is a genuine
capability unlock — `llama-server` can serve a >262,144-token prompt today, at
whatever quality unscaled extrapolation gives, with no patch.

Generalised: `--rope-scaling yarn --rope-scale 1.0001 --yarn-orig-ctx <C>`
raises the servable ceiling to `C` for any `C`, since the rope perturbation
does not depend on `C`. Use `-c <C>` to match.

**6.4 The cleaner fix, deliberately not implemented.** The honest fix is for
the server to stop capping at `n_ctx_train` when the operator explicitly asked
for more, the way `llama-bench`, `llama-completion` and `llama_context` itself
do (they warn: `n_ctx_seq (311296) > n_ctx_train (262144) -- possible training
context overflow`). That is a change to
`tools/server/server-context.cpp`, which is owned by other work in flight, and
§6.2 makes it unnecessary for now. Recorded, not done.

## 7. What to do

1. **At 300K, use no rope flags.** If it has to go through `llama-server`, add
   `--rope-scaling yarn --rope-scale 1.0001 --yarn-orig-ctx <-c value>` purely
   to lift the slot cap (§6.2). Do **not** use `--rope-scale 1.1875`.
2. **Retire "YaRN destroys this model"** in favour of "YaRN costs 1.2-2.9x
   perplexity, grows with depth, and buys nothing that unscaled extrapolation
   does not already give" (§3, §4). Retire the "partial RoPE / mrope sections
   may make the ramp meaningless" hypothesis too — §2.2 disposes of it — and
   the "UD-IQ3_XXS quantization may be the cause" one is now moot, since the
   same quant is coherent at 305K with and without YaRN.
3. **600K is now measured too, and it also holds** (§4.1). Do not extrapolate
   past it: 612,689 tokens is 2.34x the native ceiling and is the deepest
   quality measurement this project has.
4. **Stop drawing RoPE conclusions from single runs** (§5). Anything that
   matters here needs the arm repeated, ideally interleaved.
