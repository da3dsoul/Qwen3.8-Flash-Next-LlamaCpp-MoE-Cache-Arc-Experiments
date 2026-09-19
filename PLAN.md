# Plan: Qwen3.8-Flash-Next MoE expert cache on Arc Pro B70

Status: **planning complete, nothing implemented.** Grounded in
`docs/00-background.md` and the two research docs in `docs/research/`
(`01-sycl-backend-feasibility.md`, `02-model-support-status.md`), both dated
2026-09-09. Re-read those before starting — this plan cites but does not
repeat their evidence.

> **CRITICAL, read before anything else (2026-09-11 late update): the real
> target is long context (300K-1M tokens), not just decode throughput at
> trivial depth.** Every benchmark run in this project through this update —
> the cache tuning, the `-ncmoe` sweeps, the MTP implementation and its
> measured acceptance rates, the hybrid `MUL_MAT_ID` spike (if it's landed by
> the time you're reading this) — used a ~10-token prompt with a few hundred
> tokens of decode. That is not this deployment's actual workload. This
> flips at least one just-closed decision and puts several open questions
> back on the table:
> - `docs/research/07-qsa-sparse-attention-scoping.md`'s "don't pursue QSA
>   sparse attention" verdict was correct **for the regime tested** (crossover
>   at `n_kv ≥ 2049`, nothing in this project's testing ever got there) but is
>   very likely wrong for the real target — at 300K-1M tokens, sparse
>   attention's own upstream numbers (PR #27970, cited in that doc) show
>   1.76-2.2x prefill and up to 1.56x decode. It's blocked on two
>   prerequisites, not optional now: SYCL doesn't read the sparse hint at all
>   (CUDA/Metal only), and `ggml_top_k(2051)` degrades to an expensive
>   multi-pass sort above `n_kv ≈ 16384` — **the radix-select top_k fix
>   (`docs/research/02`'s "option 3") is now a hard prerequisite, not a
>   someday-nice-to-have**, since 300K-1M is deep in the regime where the
>   current bitonic top_k would be paying thousands of kernel launches/token.
> - `docs/research/06-vllm-migration-and-kvarn-viability.md`'s "KVarN not
>   worth porting for YaRN context" verdict was computed as KV-cache-size-
>   in-isolation (1M tokens @ q4_0 ≈ 6.75 GiB) — it did **not** account for
>   that VRAM competing against the MoE expert budget (`-ncmoe`/cache) on the
>   same ~30 GB card at the same time. Needs redoing as a joint budget, not
>   two separate numbers that happen to both look affordable alone.
> - The CUDA fork's own documented prefill regression under its cache
>   (`docs/00-background.md` §1: "Regresses prefill 14-66%... large-batch
>   prefill has a wide unique-expert set per op, closer to a worst case for a
>   small resident pool") has never been checked against **this** project's
>   port at all, at any prompt length, let alone at hundreds of thousands of
>   tokens of prefill — this is now a first-order risk, not a footnote.
> - Nothing about this project has confirmed the model even **loads and runs
>   a 300K+ token prompt end-to-end on this hardware today** — context
>   allocation, KV cache sizing, and whether `-c`/YaRN scaling are wired up
>   correctly for `qwen4exp` are all unverified. Establish that before
>   optimizing anything at this depth.
> - Decode throughput (the "high-40s to mid-50s tok/s" target) may not even
>   be the right headline metric for this deployment — prefill
>   time-to-first-token on a 300K-1M token prompt is plausibly more important
>   in practice than steady-state decode speed. Don't keep optimizing the old
>   metric on autopilot; confirm which one actually matters for the real
>   workload.
>
> **Update, same day, refined target:** the user's real contexts run
> **300K-600K tokens**, with compaction viable around 500K — not the full
> 1M YaRN ceiling. This matters: `docs/research/08-long-context-vram-budget.md`
> found 1M is only reachable at `-ncmoe 38-40` (8-10 of 48 expert layers
> GPU-resident) with MTP infeasible outright, but **300K-500K is much more
> comfortable** — `-ncmoe 24` (`q4_0` KV) at 300K, `-ncmoe 28` (`q4_0`) at
> 500K, both with real VRAM headroom, MTP "marginal" rather than infeasible
> at 500K. **The one open uncertainty that gated this whole analysis is now
> resolved, positively, with a real measurement**: whether the graph
> allocator collapses the 12 full-attention layers' flash-attention staging
> buffer into one reused region, or allocates it 12x redundantly (which
> would have made quantized KV *harmful* and 1M unreachable by any means).
> Measured `-c 262144 -fa 1 -ctk q4_0 -ctv q4_0 -ncmoe 24`: `SYCL0 compute
> buffer size = 1949.57 MiB` — *below* the 2,833 MiB "reuse holds" estimate,
> nowhere near the 14,097 MiB "broken" threshold. Reuse holds. Doc 08's §4
> tables (not its pessimistic §4.3) are the ones to use going forward.
> Nothing has yet confirmed the model runs a real long prompt end-to-end at
> this depth, only that it *loads* — that's the next thing to check, ideally
> with the sibling project's real ~116K-token refactor-benchmark prompt
> (`../Qwen3.8-vLLM-KVarN-MTP-Experiments/scripts/generate_messy.py 153` +
> `build_prompt.py`, reusable as-is, no vLLM dependency) rather than a
> synthetic filler prompt — see that sibling project's
> `docs/05-benchmarks/kvarn-120k-benchmark-report.md` for the methodology to
> mirror (native server-side timers, not client wall-clock; full output
> saved for a real quality review, not just a speed number).
>
> Also resolved the same day: the hybrid CPU/GPU `MUL_MAT_ID` spike
> (`docs/research/05-hybrid-cpu-gpu-mul-mat-id-scoping.md` §10) — the naive
> read-from-CPU-weights-over-PCIe version is a clear loss (tok/s drops
> monotonically as more rows go to GPU), but the underlying compute
> comparison favors GPU 4.1x per row *once weights are already resident* —
> confirming the cache isn't an optional enhancement to this design, it's
> the only thing that makes it viable. Projected ceiling with a real cache
> integrated: 28-32 tok/s, better than today's ~24-29 but still short of the
> (now secondary) 47 tok/s decode target. Given the real target is long
> context, not short-prompt decode throughput, this is real but no longer
> the highest-priority thread — establishing long-context correctness and
> throughput comes first.
>
> **Update, same day: the model was actually run end-to-end at real long
> context for the first time, and it changes the priority order again.**
> Used the sibling project's real ~116K-token refactor-benchmark prompt
> (not synthetic filler) against `llama-server`, `-ngl 99 -fa 1 -ctk q4_0
> -ctv q4_0 -ncmoe 24 -c 163840 -fit off`. Full report:
> `logs/long-context-120k-benchmark-report.md`.
>
> **It works, and the quality is genuinely good** — correctly collapsed
> 1,229 near-duplicate definitions into a parameterized structure, fixed
> every requested anti-pattern, transcribed 119/120 (99.2%) of constants
> scattered across the prompt correctly, no degeneracy. Doc 08's VRAM
> arithmetic reproduced to the hundredth of a MiB (model buffer 26,916.16
> predicted vs. measured, both KV caches 1,485.00 vs. measured) and was if
> anything slightly conservative — `-ncmoe 22` would have fit, not just 24,
> 2,836 MiB spare at `-ncmoe 24`.
>
> **But: prefill (TTFT), not decode, is the real bottleneck, and by a wide
> margin.** Measured **68.21 tok/s prefill** (116,277 tokens, 28 min 25 s
> to first token) vs. **5.34 tok/s decode** (16,382 tokens) — decode only
> degraded 3.2x from the short-context baseline (gentler than feared), but
> prefill is ~14x slower than the sibling project's fully-resident 27B
> model on vLLM and dominates end-to-end latency. This is the first hard
> evidence for this document's own earlier suspicion that TTFT, not
> steady-state decode tok/s, is the metric that actually governs this
> deployment. **At the real 300K-600K target, if this scales linearly or
> worse (likely, since attention is currently fully dense — no sparsity),
> TTFT could run well past an hour.**
>
> **This reprioritizes everything above.** The hybrid `MUL_MAT_ID` design
> (28-32 tok/s decode ceiling) and MTP (decode-only) both optimize the
> *smaller* term. `docs/research/07-qsa-sparse-attention-scoping.md`'s QSA
> sparse-attention path — disabled, blocked on SYCL never reading the
> sparse-attention hint (CUDA/Metal only) and `ggml_top_k(2051)`'s
> multi-pass-sort cost cliff above `n_kv≈16384` — directly targets prefill
> cost at exactly this depth (116K is deep past that 16K cliff) and is now
> the clear highest-leverage next step, not a someday item. Caveats on this
> run: n=1, the first post-load request measured ~5x slower than the
> reported numbers (a real warm-up cost worth accounting for in any
> production deployment plan), and this was UD-IQ3_XXS only — doc 08's
> budget was not re-verified against UD-Q3_K_XL.
>
> **Update, same day: the top-k fix is in and measured — 116K TTFT drops
> from 28 min 25 s to 17 min 11 s (prefill 68.21 -> 112.73 tok/s, 1.65x).**
> Upstream PR **#28670** ("sycl: rfc: Use radix select for top_k", `merge
> ready`, written by its author explicitly for this model) was vendored per
> `docs/research/09`'s §2.3 plan. Four of its five files applied with
> `git apply`; `ggml/src/ggml-sycl/ggml-sycl.cpp` was hand-merged as doc 09
> predicted — its `supports_op` hunk was **already present** (this project
> had lifted `k <= 32` to `k <= src0->ne[0]` earlier today), so the only real
> edit was swapping the large-k branch's body from the full-argsort fallback
> to `ggml_sycl_top_k_radix`. **The two changes did not conflict in intent**:
> the relaxed `supports_op` stays exactly as it was and the radix kernel is
> now the implementation behind it. One deliberate divergence from the PR: it
> switches to radix above `k > 8`, we kept this backend's own scan+merge
> ceiling (`k > 32`), because everything at or below 32 already runs the
> split-row scan+merge kernel this project measured and tuned — verified
> unchanged by `test-backend-ops perf` (e.g. `ne=[65000,1],k=32`: 811.5 vs
> 816.4 us, `ne=[200000,16],k=16`: 1381.0 vs 1374.4 us). A build gotcha worth
> recording: `ggml/src/ggml-sycl/CMakeLists.txt` globs `*.cpp` **without**
> `CONFIGURE_DEPENDS`, so `devbuild.sh` links a stale library and silently
> omits a *new* source file (module libraries tolerate the undefined symbol
> at link time); force `cmake .` in the build container's `build/` first.
>
> **Correctness was gated before any performance work, and it holds.**
> `test-backend-ops test -b SYCL0 -o TOP_K`: **525/525 pass**, including the
> large-k cases at `k = 2051` up to `ne=[33024,4]` *and* the deliberately
> tie-heavy `ties=1` variants. That test is a real equivalence proof, not a
> smoke test: for distinct values it demands the **identical index set** as
> the CPU reference at `max_err = 0.0`, and under ties it demands the
> identical multiset of *selected values* plus exactly `k` distinct indices.
> **But model output is not bit-identical, and that is expected rather than a
> bug**: `ggml_top_k` leaves tie-breaking unspecified, and the lightning
> indexer's `ggml_relu` (`src/models/qwen4exp.cpp:828`) flattens every
> non-positive block score to exactly `0.0f`, so the selection threshold at
> `width = 2051` sits inside a large tie set. `build_attn_qsa`'s mask build
> is order-insensitive (`ggml_set_rows` + `ggml_add` with the causal mask,
> `:913-937`) but not tie-insensitive, so two valid top-k implementations
> pick different equally-valid zero-score blocks. Observed directly: the 32K
> A/B produced different greedy continuations from the same seed. Upstream
> measured the quality cost of exactly this at `ppl 4.2954 -> 4.2980`
> (+0.0026). Our own quality check is the 116K run's saved reasoning trace
> (`logs/long-context-120k-after/run1.reasoning.txt`) — coherent, correctly
> recalls the 153 entity blocks and every planted anti-pattern across the
> full 116K prompt, no degeneracy.
>
> **Numbers, three levels, all on an idle GPU with no other llama/vLLM
> process running.** Kernel level (`test-backend-ops perf -o TOP_K`,
> `k = 2048`, argsort -> radix): `ne=[8192,1]` 226.93 -> 11.51 us (**19.7x**),
> `ne=[32768,32]` 1805.44 -> 32.20 us (**56x**), `ne=[131072,32]` 14709.41 ->
> 135.04 us (**109x**) — right on doc 09 §1.3's modelled ~100x. 32K stand-in
> prefill (same source tree, `GGML_SYCL_TOPK_ARGSORT_BASELINE` macro builds
> the old path for a one-tree A/B, `staging/work/topk_prefill_ab.sh`): 80.63
> -> **128.60 tok/s** (395.76 -> 248.14 s), 1.59x. And the headline, the real
> 116K benchmark re-run against `logs/long-context-120k-benchmark-report.md`
> verbatim — same prompt, same `-ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -ncmoe 24
> -c 163840 -fit off`, same 116,277 prompt tokens, same server-side timers:
> **prefill 68.21 -> 112.73 tok/s, TTFT 1,704.79 s -> 1,031.45 s (28 min 25 s
> -> 17 min 11 s, 673 s saved, 1.65x)**. Full artifacts in
> `logs/long-context-120k-after/`. **Decode is neutral, not improved**: 4.22
> tok/s over 2,000 tokens vs. the baseline's 5.34 over 16,382 — not
> comparable as averages (different windows), but the per-interval `tg_3s`
> bands overlap almost completely (3.30-5.32 now vs. 2.91-5.95 then), so the
> honest read is "unchanged within this benchmark's own noise." The PR's own
> +10.8% decode claim at 131K is a `nrows = 1` kernel result and does not
> survive contact with a decode step whose cost is dominated elsewhere.
>
> **This corrects doc 09's cost model in a way that matters for what comes
> next.** Doc 09 §1.3 put Term 1 (top-k) at a `>= 437 s` floor of the
> measured 1,705 s and argued it grows as `O(N^2 log^2 N)`, projecting
> **4.4 h at 600K** and "it alone makes the real target unreachable." The
> measured saving is **673 s of 1,705 s (~40%)** — so the *share* was right,
> slightly better than its floor. **The growth claim was not.** The baseline
> server log's own progress curve (`logs/long-context-120k/server.log`) shows
> marginal prefill throughput *flat* at ~76-80 tok/s from 10K all the way to
> 116K, identical to the 32K run's 80.63 tok/s average — i.e. the bitonic
> top-k was a large constant-ish tax, not a runaway quadratic. The "4.4 h at
> 600K for top-k alone" projection should be treated as retired.
> **Post-fix, the binding term is the one doc 09 §5 named and neither option
> fixed: CPU-resident MoE.** Arithmetic, from the model shape: 48 layers x
> (10+1) experts x 3 matmuls x 2560 x 640 x 2 = **~5.2 GFLOP/token**, half of
> it on the host at `-ncmoe 24`. 2.6 GFLOP/token against a realistic
> 0.2-0.4 TFLOP/s of IQ-quant GEMM on the 9950X caps prefill at roughly
> **75-150 tok/s** — and we measured 112.73, i.e. **already near this
> configuration's arithmetic floor.** No further top-k or attention work
> moves it. The levers that do, in order: (1) **`-ub` 512 -> 2048**, which
> amortises the CPU expert-bank sweep ~4x and which **this fix just
> unblocked** — doc 09 §2.2's scratch table shows the old argsort needed
> 12.69 GiB of pool at `-ub 2048`/614K while the radix kernel allocates
> **zero** at `nrows >= 2*nsm`; (2) lower `-ncmoe` (the 116K run had
> 2,836 MiB spare, `-ncmoe 22` fits); (3) real expert residency, which is the
> MoE-cache thesis and still unmeasured against prefill. For scale on what
> residency is worth: the sibling project's fully-resident 27B does ~963
> tok/s prefill on the same box (report §5), and a fully-GPU-resident
> Flash-Next would have a MoE-term ceiling around 1,100-2,300 tok/s — so
> four-digit prefill is an ordinary number for a resident model and simply
> unreachable while 24 of 48 expert layers stream from host RAM on a 30 GB
> card. Dense attention becomes the next wall only further out (`O(N^2)`:
> ~2,300 tok/s ceiling at 116K but ~450 tok/s at 600K), which is when doc 07's
> QSA sparse attention stops being the smallest term.
>
> **YaRN risk, checked but deliberately not pursued** (the standing priority
> is making 116K solid before 300-600K gets attention). Dumped the GGUF
> header directly: it carries **no rope-scaling metadata at all** — only
> `qwen4exp.context_length = 262144`, `rope.freq_base = 1e7`,
> `rope.dimension_count = 64`, `rope.dimension_sections = [11,11,10,0]`.
> There is no `rope_scaling.type`, no factor, no
> `rope_scaling.original_context_length`. So `src/llama-model.cpp:1313-1332`
> defaults `rope_scaling_type_train = LINEAR`, `rope_freq_scale_train = 1.0`,
> `n_ctx_orig_yarn = n_ctx_train = 262144`; `src/llama-context.cpp:164-174`
> then inherits LINEAR and sets `yarn_ext_factor = 0`, i.e. **YaRN is
> entirely off and nothing activates it implicitly.** Past 262144 llama.cpp
> only *warns* — it extrapolates RoPE unscaled rather than erroring, which is
> the silent-quality-loss failure mode. This **confirms doc 08 §7.3's
> unverified claim, from the GGUF rather than from the load banner**, and adds
> that the machinery does work if asked: the SYCL `rope_multi` kernel (this
> model uses `ggml_rope_multi`, `src/models/qwen4exp.cpp:1004-1010`) does pass
> `ext_factor` into `rope_yarn` (`ggml/src/ggml-sycl/rope.cpp:20-28, 96, 154`),
> so `--rope-scaling yarn --yarn-orig-ctx 262144 --rope-scale N` would take
> effect (N ~= 1.25 for `-c 327680`, ~2.34 for `-c 614400`; doc 08's
> `--rope-scale 4` is the 1M ceiling, not a 300-600K value). **Nothing in this
> project has ever set those flags** — today's 116K run did not need them
> (116,277 < 262,144) and doc 08's recommended configs / `bench120k/serve.sh`
> omit them. Not a problem for the 116K target; a real, unclosed gap for
> 300-600K, and untested for quality at any depth.
>
> **Update, same day: `-ub` 512 -> 2048 was taken and it is the largest
> single win this project has measured — 116K TTFT drops again, 17 min 11 s
> to 8 min 04 s (prefill 112.73 -> 240.15 tok/s, 2.13x).** Pure flag tuning,
> no code change; same working tree, same `staging/devbin` (radix top-k
> confirmed live before measuring: `test-backend-ops perf -o TOP_K`
> reproduced `ne=[131072,32],k=2048` at 134.71 us against the 135.04 us
> recorded above). **The real 116K run**, same prompt and same 116,277 prompt
> tokens as `logs/long-context-120k-after/`, same server-side timers:
> **prefill 112.73 -> 240.15 tok/s, TTFT 1,031.45 s -> 484.19 s (547 s
> saved)**. Decode is again neutral — 4.17 vs 4.22 tok/s over the same
> 2,000-token window, -1.4%, inside this benchmark's own band. Quality holds:
> the saved trace (`logs/long-context-120k-ub2048/run1.reasoning.txt`)
> correctly reads the 153-block structure ("numeric suffixes 0..152") and
> recalls the planted anti-patterns, no degeneracy. Artifacts in
> `logs/long-context-120k-ub2048/`; sweep in `logs/ub-sweep-32k.log`,
> `logs/ub-phase2.log`, `logs/ub-ctx-matrix.log`; scripts
> `staging/work/ub_sweep.sh`, `ub_phase2.sh`, `ub_ctx_matrix.sh`,
> `run_116k_ub.sh`.
>
> **32K stand-in sweep** (`prompt32k.txt`, 31,911 tokens, `-ncmoe 24
> -c 65536`, everything else as the A/B above): `-ub 512` **128.15** tok/s,
> `-ub 1024` **263.40** (2.06x), `-ub 2048` **408.39** (3.19x), `-ub 4096`
> **OOM**. Monotone and nowhere near noise.
>
> **A guardrail catch worth carrying forward as standing practice: the
> sweep's own first arm was wrong by 24% and would have understated the
> win.** Leading `-ub 512` measured 97.72 tok/s, against the 128.60 the 12:46
> A/B had already established for that exact config. Rather than report the
> ratio off a bad baseline, `-ub 512` was re-run as a *trailing* control and
> came back at **128.15** — i.e. the first arm, not the established number,
> was the outlier. Cause: **page cache.** `-ncmoe 24` leaves 52.8 GiB of
> `CPU_Mapped` tensors (19,240.51 + 6,130.57 + 27,465.95 MiB) to be faulted
> from `bcache0` on demand, and the box only has ~98 GiB available; a cold
> first run reads them at 30-58 MB/s (observed one context construction
> single-threaded for >10 min) where a warm one loads in ~8 s. **So: after
> any container start, discard the first arm or pre-warm the model file
> (`logs/warm-pagecache.log`'s 2m23s `cat`), and never A/B across a
> cold/warm boundary.** This is the same class of confound as the stale
> Docker image in the update below — on this box, measurement hygiene has
> now twice been worth more than the thing being measured.
>
> **The VRAM tradeoff `-ub` buys is real, was characterised, and it costs
> `-c` (or `-ncmoe`).** The compute buffer scales with `-ub` *and* with `-c`,
> and the `-ub 2048` buffer no longer fits alongside a 163,840-token KV cache
> at `-ncmoe 24`. Measured (`-lv 4`, `SYCL0` model + KV + RS + compute):
>
> | `-ub` | `-c` | `-ncmoe` | model | KV | RS | compute | total | result |
> |---|---|---|---|---|---|---|---|---|
> | 512 | 65536 | 24 | 26,916.16 | 432.00+162.01 | 112.57 | 1,086.57 | 28,709 | ok |
> | 1024 | 163840 | 24 | 26,916.16 | 1,080.00+405.01 | 112.57 | 2,391.14 | 30,905 | ok |
> | 2048 | 122880 | 24 | 26,916.16 | 810.00+303.76 | 112.57 | 3,692.28 | 31,835 | **ok** |
> | 2048 | 131072 | 24 | 26,916.16 | 864.00+324.01 | 112.57 | 3,884.28 | 32,101 | OOM |
> | 2048 | 163840 | 24 | 26,916.16 | 1,080.00+405.01 | 112.57 | 4,972.28 | 33,486 | OOM |
> | 2048 | 163840 | 26 | 24,991.16 | 1,080.00+405.01 | 112.57 | 4,972.28 | 31,561 | ok |
>
> (MiB.) That brackets this card's usable ceiling at **~31.9 GiB, between
> 31,835 and 32,101 MiB** — a firmer number than the "30GB B70" this plan has
> been quoting, and it retires the "2,836 MiB spare" figure in favour of a
> measured envelope. **Recommendation: `-ub 2048 -ncmoe 24 -c 122880`** —
> 122,880 still covers the 116,277-token prompt plus 2,000 generated with
> room over, and `-c` is a pure VRAM knob that does not touch throughput
> (attention cost follows actual `n_kv`, not allocated context), so this is
> apples-to-apples with the baseline on everything that moves the number.
> `-ncmoe 26` is the alternative if 163,840 of context is actually wanted,
> but it pushes two more expert layers onto the host and so pays back some of
> the win. `-ub 1024 -c 163840 -ncmoe 24` is the conservative fallback and
> still worth 2.06x. **`-ub 4096` is out** — it OOMs even at `-c 65536`.
>
> **Operational hazard, found the hard way: an over-large `-ub` can present
> as a hang rather than an OOM.** At `-ub 2048 -c 163840 -ncmoe 24` the
> SYCL/level-zero allocation fails (`UR_RESULT_ERROR_OUT_OF_RESOURCES`, and
> under `-fit` auto the VMM pool's `alloc` at `ggml-sycl.cpp:1841`) and the
> process then **spins one core indefinitely with RSS frozen** instead of
> aborting — 13 min 30 s observed before it was killed, reproduced under a
> 240 s `timeout`. At `-c 131072` the same overflow exits reasonably with
> `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`. So when raising `-ub`, read the
> `sched_reserve: SYCL0 compute buffer size` line under `-lv 4` against the
> envelope above *before* concluding anything from a stalled run.
>
> **This corrects the cost model in the update above, in the same direction
> that update corrected doc 09.** That update put a **75-150 tok/s**
> arithmetic ceiling on prefill from 2.6 GFLOP/token of host-side MoE at
> `-ncmoe 24`, and read the then-measured 112.73 as "already near this
> configuration's arithmetic floor." Measured 240.15 tok/s at 116K and 408.39
> at 32K **exceed that ceiling by 1.6-2.7x**, so the premise was wrong: the
> binding term at `-ub 512` was never host FLOPs, it was the **fixed
> per-ubatch cost of sweeping the CPU-resident expert bank** — weight
> streaming, not arithmetic. `-ub` amortises it almost exactly as predicted
> (512 -> 2048 is 4x fewer sweeps and returned 3.19x at 32K). The FLOP
> ceiling is still out there but is now roughly 3x further away than the
> number we measure, and **"already near the floor" should be treated as
> retired** — the same way that update retired doc 09's "4.4 h at 600K."
> Of the three levers that update listed, (1) is now spent and delivered more
> than it promised; (2) lower `-ncmoe` has become a *cost* of (1) rather than
> a lever of its own; (3) real expert residency is still unmeasured against
> prefill and is still the MoE-cache thesis.
>
> **What is now the largest remaining prefill term: dense attention.** The
> same config measures 408.39 tok/s at 32K and 240.15 at 116K — a 41% decay
> purely from depth, and the per-interval curve in
> `logs/long-context-120k-ub2048/server.log` climbs to ~252 tok/s by 30K and
> then falls away monotonically. That is the `O(N^2)` term this plan has been
> deferring, and it is no longer hiding behind a larger constant. **It brings
> doc 07's QSA sparse attention forward of where the update above placed
> it** ("only further out... ~450 tok/s at 600K"): at 116K it is already the
> top term, and its two prerequisites are unchanged — SYCL still doesn't read
> the sparse hint, and the radix top-k that the other prerequisite needed is
> now done. Also still untested and cheap: `--load-mode none`, which the
> loader itself warns about on every run of this config ("tensor overrides to
> CPU are used with mmap enabled") and which would take the page-cache
> fragility described above off the table entirely.
>
> **Update, same day (2026-09-11, late): QSA sparse attention on SYCL was
> scoped against real measurements and deliberately NOT implemented. Full
> writeup: `docs/research/10-sycl-sparse-attention-scoping.md`.** The update
> above called dense attention "the largest remaining prefill term" and
> brought doc 07's QSA path forward on that basis; a separate confirmation
> that decode collapses to 4.17-5.34 tok/s at 116K made sparse FA look like
> the decode fix too. **Both framings were checked and the decode one does not
> survive measurement.**
>
> - **Upstream has an open PR for exactly this** — **#28770** "CUDA: enable
>   sparse fa for qwen4" (OPEN, created 16:41 UTC today), continuing #27970.
>   Still **zero SYCL files**, so there is no port to adapt this time. Its
>   `qwen4exp.cpp` hunk is simply the un-commenting of `:943-946` with **no
>   backend-capability gate** — the hint goes into `op_params[4]`
>   unconditionally and each backend decides. Re-verified by whole-tree grep
>   that no SYCL or CPU FA path reads slot 4, so doc 07 §3.5's "provable no-op
>   on our stack" still holds. **Its own decode claim for this architecture is
>   1.18x at d100000**, not the 1.5-2.2x prefill numbers.
> - **The decisive new number, and doc 07 §5's own unchecked revisit
>   condition**: at the real production decode shape (head 256, 2 KV heads,
>   gqa 12, `Q->ne[1]=1`, `q4_0` K/V) `test-backend-ops perf -o
>   FLASH_ATTN_EXT` measures **2.93 ms/call at `n_kv = 118016`** —
>   **35.13 ms/token** over the 12 full-attention layers. Against the measured
>   240 ms/token (4.17 tok/s) at 116K that makes `FLASH_ATTN_EXT`
>   **14.6% of a decode token, and the whole available sparse prize ~1.16x.**
>   New perf cases covering `kv = 16384..118016` in both `f16` and `q4_0` were
>   added to `tests/test-backend-ops.cpp` for this (the existing qwen4exp loop
>   stopped at 8192 and f16 only) — that is the *only* source change this pass
>   made; `qwen4exp.cpp:946` was left exactly as it is.
> - **A previously unaccounted cost, found while doing this: `-ctk q4_0 -ctv
>   q4_0` costs a flat ~1.6x on the FA op itself** (1.48x at 16k rising to
>   1.64x at 118k) = **13.7 ms/token at 116K.** Cause: the SYCL TILE kernel is
>   written against `half2` K/V and `launch_fattn` is called with
>   `need_f16_K = need_f16_V = true`, so it **re-dequantizes the entire KV
>   cache into a dense f16 staging buffer on every single FA call**
>   (`ggml/src/ggml-sycl/fattn-common.hpp:945-1007`) — 242 MiB of conversion
>   traffic per layer per token at 116K, 2.8 GiB/token over 12 layers, plus
>   242 MiB of standing VRAM. Not obviously worth changing (`q4_0` KV is what
>   makes 116K fit) but it should be on record.
> - **There is a 3.1x hole in our own decode numbers that is not attention.**
>   `llama-bench` on the identical config measures **23.54 tok/s @ d2048 and
>   22.93 @ d8192**, and the isolated FA cost accounts for essentially the
>   whole 1.14 ms/token gap between them (f16 prediction 1.31 ms). Extrapolated
>   the same way, 116K should be **~13 tok/s, not 4.17**. VRAM pressure is
>   *not* the explanation (decode measured 5.34 / 4.22 / 4.17 across configs
>   with very different headroom). The two live candidates are host page-cache
>   residency of the 52.8 GiB `CPU_Mapped` expert set and the QSA indexer
>   chain below.
> - **The bigger depth-proportional term is probably the QSA indexer, and
>   sparse FA provably cannot touch it.** Doc 07 §2.5 established the indexer +
>   top-k + mask machinery is built unconditionally regardless of line 946;
>   what nobody counted is that at decode it is **~20 ops per QSA layer on
>   `[128, 29504]` and `[118016]` tensors — ~240 kernel launches per token over
>   12 layers**, several strided (`qwen4exp.cpp:794-799`'s four `ggml_cont`
>   slices walk with stride 4) and one a **118,016-row `ggml_get_rows`**
>   (`:789`). Estimated 25-100 ms/token at realistic efficiency for that access
>   pattern. **This is the cheapest next measurement in the project and it has
>   never been run**: `GGML_SYCL_OP_PROFILE=1` already buckets by graph call
>   site and `build_qsa_top_k` already names its tensors (`indexer_k_raw`,
>   `indexer_k_pooled`, `indexer_k`, `indexer_q`, `indexer_score`,
>   `indexer_score_tokens`, `indexer_top_k`), so one decode window at 32k
>   depth prints the indexer's share next to `FLASH_ATTN_EXT`.
> - **Doc 10 §4 scopes two SYCL designs for when this is picked up, and the
>   cheaper one is not the CUDA-shaped one.** Design A (port #27970/#28770's
>   structure: compaction kernel + indirect K/V/mask row lookups threaded
>   through the 1,246-line templated TILE header) is ~400-470 LOC, HIGH risk,
>   and recovers only 21.45 of the 35.13 ms/token because the f16 staging pass
>   still converts the whole cache. Design B exploits that staging pass
>   instead: **gather-dequant only the `width = 2051` selected rows into a
>   compact buffer (padded to 2304, a clean multiple of `FATTN_KQ_STRIDE`) and
>   hand the *unmodified* kernel an effective `ne11`** — ~360 LOC, MEDIUM risk,
>   **zero edits to the FA kernel**, recovers ~33.6 ms/token, and frees the
>   242 MiB staging buffer. Exact because softmax is permutation-invariant over
>   the finite-mask column set; decode-only by construction (one compact K/V
>   buffer needs one shared index set, which only `Q->ne[1] == 1` gives).
>   **Prefill is explicitly out of scope and got a stronger reason than doc
>   07's**: #28770's prefill design gathers per group of `ncols1 = 8` queries,
>   which on SYCL means 256 separate oneDNN SDPA calls per `-ub 2048` ubatch
>   instead of one — trading 7.2x traffic for 256x call overhead.
> - **Correctness harness, for free, when it is time**: #28770's
>   `test-backend-ops` hunk adds qwen4-shaped sparse cases, and
>   `test_flash_attn_ext` already switches to `init_tensor_kq_mask_sparse()`
>   whenever `n_kv_max > 0` while the CPU reference ignores slot 4 — so those
>   cases are a real equivalence proof against a genuinely sparse mask, and any
>   column a compaction kernel drops shows up as NMSE error rather than as
>   silent quality loss. They were **not** added this pass because on SYCL
>   today they pass trivially and would be dead coverage.
> - **Measurement hygiene, third time today, and it cost the two arms that
>   mattered most.** The `d32768`/`d65536` arms were abandoned after 16 min
>   with the process in uninterruptible sleep (`D` state), CPU time frozen, and
>   the box at **88-91% iowait**: `unbooru-tagger` (26 TB lifetime read) and
>   `deluged` (2.2 TB) were saturating the array the model lives on while swap
>   sat fully exhausted (9/9 GiB) with ~9 GiB free against a 52.8 GiB mapped
>   working set. **`ps aux | grep llama` + `docker ps | grep sycl` is necessary
>   but NOT sufficient before a long-context run on this box — also check
>   `vmstat` iowait and `ps -eo stat` for `D`.** The leading `d0` arm also came
>   back at 1.30 tok/s (cold cache), reproducing the discard-the-first-arm rule
>   from the `-ub` sweep above.
> - **Verdict: leave `src/models/qwen4exp.cpp:943-946` as it is.** Revisit when
>   (a) the call-site profile shows `FLASH_ATTN_EXT` is the largest
>   depth-proportional decode term, (b) a re-measured 116K decode number is one
>   the FA cost model can explain, and (c) the box can hold the expert set in
>   page cache for the length of an A/B. If (b) closes the 3.1x gap, Design B
>   is worth ~1.78x and becomes worth building.
>
> **Update, same day (2026-09-11, later): the 3.1x decode gap was profiled and
> attributed. It is NOT the QSA indexer. 61% of a 116K decode token is spent
> outside the SYCL graph entirely, and the mechanism was caught live. Full
> writeup: `docs/research/11-long-context-decode-profile.md`.** The update above
> named the indexer as the leading hypothesis at an estimated 25-100 ms/token
> and called the call-site profile "the cheapest next measurement in the
> project". That measurement was made. The hypothesis does not survive it.
>
> - **The QSA indexer chain now has a number, and it is 18.29 ms/token at
>   `n_kv = 118016`** — below the bottom of doc 10's bracket, and **half of
>   `FLASH_ATTN_EXT`'s 34.77 ms**. Measured by replicating the whole of
>   `build_qsa_top_k` (`src/models/qwen4exp.cpp:721-870`) as one
>   `test-backend-ops` graph at the real decode shape (`n_tps = n_stream = 1`,
>   `width = 2051`), gated first on **2/2 correctness passes against the CPU
>   reference at `max_err = 0.0`**. Cost is linear in `n_kv` above ~16k
>   (157.23 us/layer at 8192 rising to 1524.24 at 118016). **That is the only
>   source change this pass made, and it is test-only**; `qwen4exp.cpp:943-946`
>   is still untouched. Side finding: **`q4_0` makes the indexer 11-12%
>   *faster*** (the gather at `:789` reads 72 B/row instead of 256 B), the exact
>   opposite of its 1.62x tax on FA — the two knobs pull opposite ways and only
>   the FA half was known.
> - **A component model built from these numbers reproduces this project's own
>   measurements to 0.5%.** Floor (depth-independent) 40.46 ms/token, plus FA,
>   plus indexer. It predicts `d8192` at 22.25 tok/s against **22.13 measured**,
>   and predicts **93.52 ms/token (10.69 tok/s) at 116K**. Doc 10's model said
>   76.9 ms because it had no indexer term — so **the indexer closes 16.6 ms of
>   doc 10's 163 ms hole, 10.2% of it.**
> - **The other ~146 ms/token is not in the graph, and the 116K benchmark's own
>   logs already proved it.** `llama-server` prints a `tg_3s` rate every three
>   seconds; across the three 116K runs, at constant depth and constant config,
>   decode varies **3.09x** (`-ub 2048` run) to **7.07x** (baseline run)
>   interval to interval — and the **fastest intervals hit 106.3 and 95.0
>   ms/token, i.e. the model's 93.52.** GPU kernel time at fixed `n_kv` does not
>   swing 3-7x between adjacent windows; a host stall backed by a shared disk
>   does. The median is 2.4x worse than the best window because the cost is
>   *sometimes absent*.
> - **Mechanism, observed live while trying to run the `d32768` arm.**
>   `bcache0` at **99.9% util** / `md0` at **99.8%**, driven by
>   `unbooru-tagger-training` (**up 10 days**, 26.16 TB lifetime read) at
>   **80-96 MB/s** on the array the model lives on. Our `llama-bench` sat in `D`
>   state at **83 major faults/s**, `r_await` 15-16 ms, RSS stuck at 24.8 GiB of
>   the ~52.8 GiB `CPU_Mapped` expert set, growing at **0.36 MB/s** — ~22 hours
>   to finish one arm. Host `free` ~1 GiB, swap exhausted 9.2/9.3 GiB. Since
>   every decode token routes to a fresh random 10-of-512 expert set in each of
>   24 CPU layers, the working set is the whole 52.8 GiB with no locality;
>   **146 ms/token is ~11 faults/token, ~45 KB/token** — small enough to hide in
>   every aggregate this project has collected. **That training job has been
>   running continuously since before every 116K benchmark in this repo.**
> - **Also measured, and useful: what the depth-independent 40.46 ms is.**
>   `GGML_SYCL_OP_PROFILE=2` over an exactly-bounded 32-token decode window
>   (boundary found by counting back to `12 x n_gen` FA calls —
>   `staging/work/agg_opprof_decode.py`): **12.57 ms host SYCL dispatch, 13.53 ms
>   GPU wait, 18.64 ms outside `graph_compute`** (= CPU-resident MoE + host graph
>   build, and it is **flat with depth**, confirming the `-ncmoe` baseline). The
>   whole `d2048 -> d8192` slope lands in GPU wait (+0.99 of +1.01 ms), so there
>   is **no fourth depth-proportional term**. A decode token dispatches **~2,800
>   SYCL nodes**, and the top ten call sites are all hyper-connections
>   (`hc_down`/`hc_up`/`hc_rms`/... at 96.8 calls/token each, ~970 nodes,
>   ~4.0 ms) — flagged only, `hyper_connect.cpp` is out of scope, but fused
>   `GGML_OP_DSV4_HC_*` ops already exist at `ggml-sycl.cpp:5673-5681` and decode
>   is not using them.
> - **One redundancy found, quantified, and deliberately NOT fixed.**
>   `qwen4exp.cpp:851-857` casts the KQ mask f16->f32 inside `build_qsa_top_k`,
>   but `kq_mask` is the single `self_kq_mask_cnv` shared by all layers
>   (`llama-graph.h:321,340`) and ggml does not CSE nodes — so a token does **12
>   identical 118016-element casts** where one would do, trivially hoistable into
>   the existing `qsa_inps` cache. It moves 708 KB, so eleven redundant copies
>   are **under 0.2 ms/token, <0.1% of the token**. Recorded so nobody re-derives
>   it and mistakes it for a lever. (The mode-1 profiler reports it as the
>   chain's most expensive op at 2.67 ms/call; mode-1 absolutes are unusable
>   here — its ~30 nodes sum to 11.16 ms against a true 1.52 ms, so an op
>   "costing" more than the whole chain is arithmetically impossible.)
> - **Recommendation, and it reverses the priority order again.** Decomposed:
>   40.46 floor + 34.77 FA + 18.29 indexer + ~16 serving + ~130 paging = 240.01
>   ms/token. So: **kernel work alone (perfect sparse FA *and* perfect indexer)
>   is worth 1.28x; `--load-mode none` alone is worth ~2.18x; both are worth
>   4.2x.** Do the flag first — `PLAN.md` above already listed it as "still
>   untested and cheap" and the loader warns about mmap + CPU overrides on every
>   run of this config (check headroom: 123 GiB box, ~27 GiB in other services,
>   swap exhausted, container capped at `mem_limit: 90g`). Then re-measure
>   `d32768`/`d65536` on a quiet array — the model predicts **18.1 and 14.3
>   tok/s**. Only then the graph work, in measured order: FA (doc 10 §4.2 Design
>   B, unchanged), then the indexer — where the target is **fusion, not a new
>   kernel**: `ggml_lightning_indexer` already exists, is already implemented in
>   this SYCL backend (`ggml-sycl/lightning-indexer.cpp`, dispatched at
>   `ggml-sycl.cpp:5682-5684`) and is already used by five other DSA models, and
>   it fuses exactly what `qwen4exp.cpp:825-857` builds from ~8 primitive ops.
> - **Measurement hygiene, fourth time.** `ps aux | grep llama` and `docker ps |
>   grep sycl` were clean before every run and the GPU was genuinely idle. Not
>   enough, again. **The check that mattered was `iostat -x` on the device
>   holding the model plus `/proc/<pid>/stat` for `D` state and major-fault
>   rate.** Standing rule for this box: a long-context number is trustworthy only
>   if the model's array is under ~50% util *and* the process RSS has reached
>   ~53 GiB and stopped growing. The `d2048` leading arm reproduced doc 10's
>   23.54 tok/s at 23.66, which is what made the shallow arms usable; the
>   trailing `d2048` control was never reached.
>
> **Update, same day (2026-09-11, later still): the update above named the right
> cost and the wrong tensor, so it recommended the wrong flag. `-lm none` and
> `-lm mlock` were both tested and both delivered nothing. The tensor actually
> being demand-paged is `per_layer_token_embd`, and the flag that fixes it is
> `-lzm off`. Measured: decode 4.167 -> 11.146 tok/s (2.68x), prefill 240.15 ->
> 394.14 tok/s (1.64x), TTFT 8 min 04 s -> 4 min 55 s.** Correction in
> `docs/research/11-long-context-decode-profile.md` §7 (read its §7 before its
> §3.3/§5); artifacts in `logs/long-context-120k-lazyoff/`, driver
> `staging/work/run_116k_lazy.sh`, host sampler `staging/work/host_sampler.sh`.
>
> - **The `-lm` arms were no-ops for two independently code-verifiable reasons.**
>   `-lm mlock` could not lock anything: the test container has `ulimit -l` =
>   **8 MiB** and no `CAP_IPC_LOCK`, and `llama_mlock::impl::raw_lock`
>   (`src/llama-mmap.cpp:689-719`) warns and returns `false` rather than failing
>   the run. And even a working mlock **deliberately skips lazy tensors**
>   (`src/llama-model-loader.cpp:1653-1654`: *"locking a lazy tensor would fault
>   all of it in, which is what lazy avoids"*). `-lm none`/`-lm mlock` both also
>   set `use_mmap = false` (`:559`), so the expert weights were already
>   unevictable pinned/anonymous host RAM in a swapless cgroup — **the page-cache
>   eviction story doc 11 §3.3 told was already impossible in both arms, and
>   decode still did not move.**
> - **What doc 11 §3.3 called "52.8 GiB of `CPU_Mapped` expert tensors
>   (19,240.51 + 6,130.57 + 27,465.95 MiB)" is two different things.** The real
>   split is **25,371 MiB of CPU-resident experts + 27,466 MiB of lazily-mmapped
>   PLE table**, and the line naming it sits two lines above the numbers doc 11
>   quoted, in this repo's own `logs/long-context-120k/vram-check.log`:
>   `add: tensor per_layer_token_embd.weight (size = 27465 MiB) lazy read enabled`.
> - **Why no `--load-mode` value can help it.** `qwen4exp` marks that tensor
>   `TENSOR_READ_LAZY` (`src/models/qwen4exp.cpp:192-193`) and it is over the
>   4 GiB `-lzm auto` threshold, so `init_mappings` keeps an mmap for it on
>   `if (use_mmap || lazy.any())` — *"read_lazy also requires mmap; this
>   condition make sure it's usable even when --load-mode is not set to mmap"*
>   (`src/llama-model-loader.cpp:1403-1405`). Worse, that same function computes
>   `prefetch_size = prefetch && use_mmap ? -1 : 0` (`:1422`), so **`-lm none`
>   strips the mapping's only `MADV_WILLNEED` advice — it is actively worse than
>   `auto` for this tensor, not neutral.** The loader's own warning that led doc
>   11 here (*"tensor overrides to CPU are used with mmap enabled - consider
>   using --load-mode none"*, `:1239`) fires for `-ncmoe`'s **overrides**, which
>   the PLE table is not.
> - **The access pattern is the worst case for demand paging, by construction,
>   and the arithmetic matches doc 11's own fault count.** GGUF says
>   `ple.ngram_size = 3`, `ple.heads_per_ngram = 8` -> **16 rows per token**, and
>   `ple.head_vocab_sizes` = 16 x ~20,000,003 -> **~320 M rows over 27,465 MiB
>   ≈ 90 B/row**. `llm_graph_input_ple::set_input` derives each index as
>   `mixed % head_vocab_sizes[h] + head_offsets[h]` where `mixed` is a
>   multiplicative-XOR **hash** of the n-gram (`qwen4exp.cpp:1294-1306`), gathered
>   at `:1381`; the source comment at `:1220` says it plainly. So each token
>   touches **16 uniformly-random 4 KiB pages to consume 1,440 bytes — 45x read
>   amplification with zero locality even on repeated text.** Doc 11 §3.3
>   computed the residual as "~11 faults per token, ~45 KB/token" — the same
>   number, now with a per-token mechanism instead of an inferred property of
>   expert routing.
> - **Live before/after, on a box confirmed idle first.** `-lm none` with lazy
>   still on: process in **`D` state at 80-100 majflt/s with CPU time frozen for
>   8 minutes**, `bcache0` 100% util, RSS creeping 0.33 MB/s — reproducing doc 11
>   §3.3's "0.36 MB/s" to the decimal, which retires its reading of that number
>   as expert weights still loading. `-lzm off`: **`R` state, 0.0 majflt/s**, 70%
>   user + 30% sys, `bcache0` back to ambient. **`tg_3s` interval spread over the
>   decode window collapses 3.09x -> 1.03x** (10.98/11.25/11.29 min/p50/max),
>   which is the cleanest confirmation available that the variance doc 11 §3.2
>   called its load-bearing result *was* the paging.
> - **Doc 11 §2's component model survives and is now closed to 4.1%.** It
>   predicted `40.46 floor + 34.77 FA + 18.29 indexer = 93.52 ms/token`; the real
>   server now measures **89.72**. So the graph model explains essentially the
>   whole token and **§3.1's separately-estimated "~16 ms/token serving path" is
>   not additive** — that figure was itself mostly PLE fault time measured at
>   trivial depth.
> - **This re-inverts the priority order, back toward the kernel work.** With the
>   paging term gone, FA (34.77) + indexer (18.29) are **59% of a decode token**,
>   so `docs/research/10` §4.2's Design B is now worth **1.60x** and Design B plus
>   the indexer fusion **2.37x**, against the 1.28x doc 11 §5 computed while the
>   paging term was still in the denominator. Doc 11 §5's "kernel work is the
>   smaller of two co-located costs" no longer holds.
> - **Costs and what is still open.** `-lzm off` moves 27,466 MiB into resident
>   host RAM (host model buffer **23,772 -> 51,238 MiB**; box `available` 95 ->
>   45 GiB) and load time goes **59 s -> ~4.9 min** — a real new line item for
>   `docs/research/08`'s 300-600K budget, which never counted the PLE table as
>   resident. `-lm none` and `-lzm off` were set **together** in this run; the win
>   is attributable to `-lzm off` on the evidence above, but a clean
>   **`-lm auto -lzm off`** arm was not run and should be, since `auto` keeps mmap
>   for the experts and would save ~24 GiB of that RAM. n=1, 400-token decode
>   window against the baseline's 2,000.
> - **Recommended config is now `-ub 2048 -ncmoe 24 -c 122880 -lzm off`**, and the
>   `-lm` knob should be treated as untested-but-irrelevant rather than as the
>   lever doc 11 made it.
> - **Measurement hygiene, fifth time — and this one retires the previous rule
>   rather than adding to it.** The standing rule above ("array under ~50% util
>   *and* process RSS reached ~53 GiB") is wrong in both halves. **RSS is the
>   wrong gauge**: without mmap the host weights land in `SYCL_Host` pinned USM,
>   which never appears in `/proc/<pid>/status` — a run holding ~50 GiB showed
>   `RssAnon` = **767 MB**. Judge residency from the `load_tensors:` banner and
>   box-wide `used`. And **array utilization is a symptom, not a cause**:
>   `bcache0` read 100% util because *we* were issuing 90-byte random reads into
>   it (73 MB/s at 156 IOPS, `r_await` 13.2 ms, missing the NVMe cache and landing
>   on `md0`), not because a neighbour was saturating it — sequentially the same
>   array does ~470 MB/s. **Replacement rule, one line: check the load banner for
>   `lazy read enabled` and check `majflt/s` during decode; if the first is
>   present and the second is non-zero, the number is measuring disk, not the
>   model.**
> - **And the contention check, done properly this time, found nothing — again.**
>   `nproc` **32**; `top` 90.2% idle with the loudest tenants `rustdesk` 110% +
>   `java` 70% + ~40% of miscellany = **~220% of 3200%, i.e. 6.9% of capacity**;
>   `free` **available 95 GiB of 123 GiB** (swap showed 9.3/9.3 GiB used, but that
>   is stale anon from other services and our own process's `VmSwap` was 0).
>   `numactl --hardware` reports **1 NUMA node**, so the NUMA/mismatched-node
>   hypothesis is structurally impossible on this hardware and was dropped without
>   further testing. **Third time ambient contention has been suspected on this
>   box and third time the arithmetic says no.**
>
> **The open `-lm auto -lzm off` arm was run.** Same 116K prompt/config/timers,
> 400-token decode window, `EXTRA="-lzm off"` (no `-lm` override, so `auto`/mmap
> for everything except the now-eager PLE table): **decode 10.90 tok/s** (vs.
> 11.146 for `-lm none -lzm off` — same, within noise, confirming `-lzm off`
> alone is what fixes decode, exactly as predicted) but **prefill only 252.08
> tok/s** (vs. 394.14 for `-lm none -lzm off`) — `-lm none` still earns its
> keep on the *prefill* side, where `-ncmoe`'s CPU-offloaded expert tensors see
> a much wider access pattern than decode's few-experts-per-token routing.
> Host RAM: **~29 GiB used vs. ~52 GiB** for `-lm none -lzm off` — `auto`
> genuinely does save the ~24 GiB doc 11 predicted. **Recommendation: use
> `-lm none -lzm off` (not `auto -lzm off`) given this box's real RAM headroom
> (95 GiB available) comfortably covers the extra ~23 GiB, and prefill/TTFT is
> this deployment's actual bottleneck metric** — trade RAM for the 56% prefill
> win since RAM isn't the scarce resource here. If a future deployment is RAM-
> constrained, `-lm auto -lzm off` is the fallback that keeps the full decode
> win at lower cost.
>
> **Cumulative result of today's three fixes (top-k radix-select + `-ub 2048`
> + `-lzm off`), same 116K benchmark, same original baseline:**
>
> | | Original | Final (`-lm none -lzm off`) |
> |---|---|---|
> | Prefill | 68.21 tok/s | **394.14 tok/s (5.78x)** |
> | Decode | 5.34 tok/s | **11.15-11.15 tok/s (2.09x)** |
> | TTFT | 28 min 25 s | **4 min 55 s (5.8x)** |
>
> Recommended production config: `-ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -ncmoe 24
> -ub 2048 -c 122880 -lm none -lzm off` (adjust `-c` upward for deeper context
> per `docs/research/08`'s tables, same `-ncmoe`/quant guidance otherwise).
>
> **Update, same day (2026-09-11, latest): the target was restated as a
> *slope*, not a throughput number, and against that bar today's 11.15 tok/s
> at 116K is a failure rather than a win. Decode must stay within 5% at 300K
> and 25% at 600K of short-context speed. Full analysis:
> `docs/research/12-decode-depth-scaling.md`; configuration and YaRN/VRAM
> record: `logs/long-context-300k-benchmark-report.md`.** Every performance
> number this project owned was a point measurement; this pass measured the
> curves. All source changes are test-only
> (`tests/test-backend-ops.cpp`: extra depths in the two qwen4exp perf loops,
> plus a `pooled_cached` variant of `test_qsa_indexer` registered in the perf
> *and* correctness suites). No model, kernel or memory-module code touched.
>
> - **The real curve, measured end to end** (`llama-bench -p 0 -n 32 -d ...
>   -r 2` on the recommended config, `logs/decode-scaling/llama-bench-depth-curve.log`):
>   **26.48 / 26.03 / 24.63 / 19.43 / 15.26 / 11.22 tok/s** at
>   `d0 / d2048 / d8192 / d32768 / d65536 / d118016`. So decode loses **half
>   its speed by 116K** (24.63 -> 11.22, 2.19x) and misses the bar **already
>   at 32K (+26.8%)**. The `d118016` point independently agrees with the real
>   server benchmark's 11.146 tok/s.
> - **Both depth-proportional terms are exactly linear in `n_kv`, measured at
>   nine depths out to 614,400** (`test-backend-ops perf`, production decode
>   shape): dense `FLASH_ATTN_EXT` **0.024317 us per KV token per layer**, the
>   QSA lightning-indexer chain **0.013068** — i.e. **the indexer's slope is
>   54% of attention's own**. Over 12 QSA layers the combined slope is
>   **0.448 ms per 1,000 tokens of depth**. No knee, no plateau, straight
>   lines to within 1% over a 19x range. Doc 11's 116K numbers reproduced to
>   0.5% / 0.3% on a different day.
> - **`docs/research/11` §2.5's depth-independent floor is confirmed
>   depth-independent**: the implied remainder is **35.8-37.5 ms/token** from
>   `d0` to 116K with no trend, and the component model
>   `floor + FA(D) + indexer(D)` reproduces all four deep measured points to
>   within **0.8%**. That is what makes the 300K/600K columns projections of
>   *one* term rather than of the whole curve — FA and the indexer were both
>   measured at exactly 307,200 and 614,400.
> - **So yes, the indexer is a hidden scaling problem, and no, it is not
>   bigger than attention.** At 300K it is 48.5 ms/token against FA's 89.9.
>   Its slope alone is fatal to the bar: **a perfect sparse attention with
>   today's indexer still leaves 300K at +120%.** This is the single most
>   important finding for prioritisation — `docs/research/10`'s Design B is
>   **half of a matched pair, not a standalone fix**, and building only it
>   would not deliver the target.
> - **94.3% of the indexer's cost is recomputing something that did not
>   change.** `build_qsa_top_k` (`src/models/qwen4exp.cpp:721-869`) was read
>   op by op and its traffic counted: **3,536 of 3,749 B/token/layer** is the
>   re-derivation of block-pooled indexer keys (the `get_rows` over every
>   cell, the `r` strided `cont` slices, the pooling adds, scale, RMS norm and
>   RoPE), which is a function of the KV cache and the block positions
>   **only** — identical between consecutive tokens except for the one block
>   that just closed. The byte model predicts the measured chain time at
>   118,016 to **1.6%** and at 307,200 to **0.06%**, at a flat effective
>   **284-297 GB/s across 32K-614K**: the chain is not compute-bound or
>   launch-bound, it is a streaming re-read of the whole indexer cache, 12
>   times per token, **13.8 GB/token at 300K**.
> - **The prize was measured rather than assumed, and it is 7-8x, not the
>   byte model's 17.6x.** The new `pooled_cached` arm runs the same chain with
>   the block keys handed in ready (correctness-gated against the CPU
>   reference at `max_err = 0.0`): **1,528.56 -> 217.51 us at 118,016**,
>   **4,043.51 -> 534.41 at 307,200**, **8,011.00 -> 1,000.96 at 614,400**;
>   slope 8.5x shallower. The shortfall is itself the finding: the residual
>   query-dependent half only reaches **122-131 GB/s against 285**, on
>   tensors 4x smaller spread over ~9 small kernels, so it is launch-bound
>   and has roughly another 2x in fusion.
> - **The bar is achievable, and the arithmetic says exactly what it takes.**
>   Sparse FA alone: +120% at 300K. Indexer pooled-key cache alone: +238%.
>   **Both: +14.9% at 300K / +29.8% at 600K** — a 24x slope improvement that
>   still misses both bars, which is worth knowing before anyone builds half
>   the program. **Both plus block-granularity top-k and an f16 pooled cache:
>   +7.3% / +14.6%** — 600K inside the bar, 300K 2.3 points outside a 5% ask.
>   Fusing the residual chain (estimated 2x, the only unmeasured factor in the
>   table) gives **+3.7% / +7.3%**, inside both. Call it **~800-1,000 LOC over
>   three changes** (one MEDIUM-HIGH, two MEDIUM), plus a fourth fusion step
>   if 300K must be strictly inside 5%; all gated by existing
>   exact-equivalence tests, no retraining, and **the largest piece (the
>   pooled-key cache) is backend-agnostic** — CUDA and Metal pay the same
>   recomputation — so it is plausibly upstreamable rather than a local patch.
> - **300K is the binding constraint, not 600K, and that is arithmetic not
>   luck.** The bar is relative, so a ~36 ms floor allows **1.9 ms** of depth
>   cost at 300K but **9.3 ms** at 600K: the allowance grows 5x while the cost
>   only doubles. Every candidate design clears 600K before it clears 300K.
>   **Size any plan against 300K.**
> - **A tension this exposes between the two workstreams, stated plainly.**
>   The floor is what makes the slope survivable, and the MoE expert cache —
>   this plan's founding thesis — exists to shrink it. With 5.1-5.3 done and
>   today's 36.3 ms floor, 300K/600K sit at +7.3%/+14.6%; with a resident-
>   expert floor of ~20 ms the **identical, unchanged slope** gives
>   **+13.0%/+26.0%** and 600K newly fails. Nothing got worse; the model just
>   got faster at 8K. **The slope work has to land with, or ahead of, any
>   large reduction in the floor.** The dense-model intuition is this effect
>   in reverse: a dense model looks flat because its per-token weight read is
>   enormous relative to its KV traffic, not because its attention is O(1).
>
> **The 300K prerequisites were both closed on the way, and the YaRN gap this
> plan has carried since the GGUF header dump is now shut.**
>
> - **`--rope-scale N` is exactly `n_ctx / n_ctx_orig`** (`arg.cpp:2360-2363`
>   sets `rope_freq_scale = 1/N`; `llama-context.cpp:180` takes
>   `factor = 1/rope_freq_scale`), so **1.1875 for `-c 311296`**, 1.171875 for
>   307,200, 2.34375 for 614,400. Doc 08's `--rope-scale 4` is the **1M**
>   value and is wrong for 300-600K.
> - **Verified live, not assumed — and the banner cannot tell you.**
>   `print_info: rope scaling = linear` is `hparams`' *trained* value and
>   prints unchanged in every arm, and `llama_context` prints no scaling type
>   and no `yarn_ext_factor` at all. Three greedy arms at identical seed and
>   prompt (`staging/work/rope_yarn_ab.sh`,
>   `logs/decode-scaling/rope-yarn-ab.log`): A (no flags) `freq_scale = 1`;
>   B (`--rope-scaling linear --rope-scale 1.1875`) `freq_scale = 0.842105`;
>   C (`--rope-scaling yarn ...`) `freq_scale = 0.842105`. **B differs from A**
>   (so `--rope-scale` registers) and **C differs from B at identical
>   `freq_scale`** — the only difference reaching the RoPE kernel there is
>   `yarn_ext_factor` 0 vs 1, so **the scaling type registers too.** C also
>   sits much closer to A than B does, which is textbook YaRN (it interpolates
>   only the low-frequency dimensions) and is precisely why YaRN and not
>   `--rope-scaling linear` is the right flag.
> - **VRAM at 300K, allocator-checked before anything expensive**
>   (`logs/decode-scaling/vram-probe-300k.log`, `-c 311296`, `-fit off`, all
>   three arms loaded and generated): the KV and model terms reproduce doc 08's
>   arithmetic **to the cent** (2,052.00 + 769.50 attn+idx, 19,216.16 model at
>   `-ncmoe 32`), but **`-ub 2048`'s compute buffer at this depth is
>   8,716.28 MiB** against 3,692.28 at 122,880. **That costs 8 `-ncmoe`
>   steps**: doc 08 §4.1's "300K `q4_0` -> `-ncmoe 24`" was computed at
>   `-ub 512` and does not survive `-ub 2048`. **Working config:
>   `-ub 2048 -ncmoe 32 -c 311296`, 1,472 MiB spare**; `-ncmoe 31` lands
>   exactly on the measured 31,835 MiB known-good boundary and is not worth
>   one expert layer. `-ub 1024 -ncmoe 28` is the alternative (4,279.14 MiB
>   compute). My own two-point compute-buffer fit over-predicted by 9% in the
>   safe direction — the real curve is slightly concave, ~0.0267 MiB/token at
>   `-ub 2048`.
> - **`-lzm off` does not touch VRAM** (checked, not assumed): `SYCL0 model
>   buffer size` is 26,916.16 MiB at `-ncmoe 24` with `-lzm auto` *and* with
>   `-lzm off`; the PLE table lands wholly in `SYCL_Host`. **Host RAM is the
>   new pressure point**: at `-ncmoe 32` the CPU-side tensors are
>   **61,081 MiB** (up from ~52 GiB at `-ncmoe 24`), all pinned USM under
>   `-lm none` and invisible in `RssAnon`, leaving ~30 GiB of box headroom
>   rather than ~45.
> - **The 300K prompt is real, not filler**: the sibling generator at `n = 400`
>   (vs. 153 for 116K) gives 1,076,019 chars and, tokenized with this
>   project's own model, **305,750 tokens** — 1.9% over target, at essentially
>   the same chars/token as the 116K prompt (3.519 vs. 3.540). Staged at
>   `staging/work/bench300k/`, driver `staging/work/run_300k.sh`, server config
>   `staging/work/bench300k/serve.sh`.
>
> **The real 300K run happened, and it produced a clean timing result and a
> hard negative on YaRN.** 305,801 real tokens (sibling generator at `n = 400`),
> full recommended config plus `-ncmoe 32 -c 311296` and YaRN:
> **prefill 255.47 tok/s, TTFT 19 min 57 s, decode 5.768 tok/s
> (173.37 ms/token)** — against a pre-registered component-model prediction of
> 174.7 ms/token, **0.8% out**, at a depth 2.6x past anything the model was
> built from. **But the output was total degeneracy: 999 repetitions of `/`,
> from the first token, greedy.** Per the standing guardrail this was
> investigated rather than reported, and the cause is now isolated by two
> single-variable arms:
>
> - The **known-good 116K prompt through the exact 300K config** is *also*
>   garbage (`logs/long-context-300kcfg-116kprompt/`). 116,277 < 262,144, so
>   **depth past `n_ctx_train` is exonerated — the configuration was already
>   broken at 116K.**
> - The **same prompt, same config, YaRN removed** is **fully coherent**
>   (`logs/long-context-300kcfg-noyarn/`, 375.64 tok/s prefill, 10.78 tok/s
>   decode, 971 chars of correct on-task reasoning). `-ncmoe 32` and
>   `-c 311296` are exonerated too.
>
> **Verdict: `--rope-scaling yarn --rope-scale N` destroys this model, and it
> does so *below* the trained context length, not just past it. Do not set it
> on this GGUF.** This closes the YaRN gap this plan has carried all day, with
> a negative result. The flags parse, reach the SYCL `rope_yarn` kernel and
> move `freq_scale` exactly as documented — they simply produce a model that
> cannot read its own context, which is what applying YaRN to a checkpoint
> that ships no rope-scaling metadata (i.e. was never trained with it) buys
> you. **Note the trap in the verification: the short-prompt A/B that proved
> the flags "work" could not have detected this**, because rope scaling
> multiplies *position*, so at a 75-token prompt the effect is negligible.
> Any future rope check must be run at depth. Untried and worth trying before
> giving up: this model uses **partial RoPE** (`dimension_count = 64` of a
> 128-wide head) with mrope sections `[11,11,10,0]`, and YaRN's
> `beta_fast`/`beta_slow` ramp may be meaningless under that layout — none of
> `--yarn-attn-factor`/`--yarn-beta-fast`/`--yarn-beta-slow` was swept.
> **So 300K+ has no validated rope strategy yet**; everything else it needs
> (VRAM budget, a real 305,801-token prompt, harness, drivers) is in place.
> The remaining options are (i) measure plain unscaled extrapolation past
> 262,144, now the less-damaged-looking path, and (ii) sweep YaRN's own
> parameters.
>
> **Two corrections this forced, both worth carrying.** First, **degenerate
> output flatters decode numbers** by routing every step to the same
> 10-of-512 experts in each CPU layer, collapsing the host working set: the
> degenerate 116K arm measured 80.10 ms/token against the healthy 92.77 at the
> same `-ncmoe 32`, a **~13 ms/token artifact**. The healthy floor is
> **36.3 ms at `-ncmoe 24` and 39.5 ms at `-ncmoe 32`** — +0.4 ms per
> host-resident expert layer, which also confirms `-ncmoe` is nearly free at
> `-ub 2048` (prefill 394.14 -> 375.64 tok/s, -4.7%, for a third more host
> experts). Second, **an operational bug affecting every driver in this
> project**: `docker compose run` **ignores `container_name`**, so the
> `docker stop qwen4exp-test-sycl` ending `run_116k_*.sh`, `run_300k.sh` and
> the rest has been **silently failing** and leaving the server up. It
> surfaced here as a diagnostic failing to bind port 8090 and its client
> silently talking to the *previous* run's server. Harmless this time; a live
> footgun in general. Capture the id from `docker compose run -d` and stop
> that, and always `docker ps | grep sycl` before trusting a run.
>
> **Two housekeeping notes.** All numbers above come from `staging/devbin` as
> built at 20:36 — the same `libllama` (09:46) and `libggml-sycl` (13:50) that
> today's `-lzm off` 116K run used, so they are directly comparable to
> `logs/long-context-120k-lazyoff/`. A **separate workstream began editing
> `src/models/qwen4exp.cpp` and `src/llama-memory-hybrid-idx.cpp` at 20:48**,
> after every measurement here was taken; nothing above reflects those edits,
> and `staging/devbin` should be rebuilt before mixing the two. And no
> result in this pass needed a contention explanation, but the box state was
> recorded anyway per standing practice: `nproc` **32**, `free` **available
> 96 GiB of 123**, **0 major faults/s** and no `lazy read enabled` in the
> load banner (doc 11 §7.6's replacement rule), and no other `llama*` process
> or `sycl` container during any measurement. Every component number
> reproduces a prior day's measurement to under 1%, which is the real check.
>
> **Update, same day (2026-09-11, latest): `docs/research/12` 5.3's first
> follow-on -- block-granularity QSA top-k -- is implemented, validated and
> measured. It is worth ~2% on its own today, and it is what takes the
> `docs/research/12` program from "misses both bars" to "clears both bars on
> measured numbers alone". Full writeup: `docs/research/12` section 9.**
> Source changes: `src/models/qwen4exp.cpp` (nine lines) and
> `src/llama-memory-hybrid-idx.cpp`, both behind `LLAMA_QSA_CELL_TOPK=1` for a
> same-binary A/B; plus a `block_topk` arm on `test_qsa_indexer` in
> `tests/test-backend-ops.cpp`.
>
> - **What it does.** `build_qsa_top_k` already pooled KV into blocks of
>   `compress_ratio = 4` and scored only the `n_kv/4` blocks -- matching
>   DeepSeek's block-compressed indexer -- and then **expanded the block scores
>   back to all `n_kv` cells, added the `n_kv`-wide f16 KQ mask, and ran
>   `ggml_top_k(k = 2051)` over the full width**, throwing the compression away
>   immediately before the expensive op. It now runs `ggml_top_k` on the
>   `n_kv/4`-wide block scores at `k = ceil(width/r) = 513` and expands only the
>   winners through `blk_cells`. That is the reference's own semantics: DeepSeek
>   selects `indexer_top_k/r` whole blocks plus the tail, and the old
>   `width = indexer_top_k + r - 1` was a cell-granularity approximation of it.
> - **It is NOT the "graph-level reordering, lower risk than a kernel" the
>   scoping called it, and two hazards had to be fixed for it to be sound.**
>   A block-granularity selection has no per-cell mask downstream, so every
>   visibility test the per-cell mask carried had to be re-expressed at block
>   granularity first. (1) The **incomplete tail was unreachable**: cells no
>   full block covers -- at decode, *the token's own key* -- are reachable only
>   through `blk_cells`, which was zero-filled for the spare block, so a block
>   top-k would have gathered cell 0 four times and dropped the most recent 1-3
>   tokens from attention on every decode step. (2) **Whole future blocks
>   carried `+1e9`, not `-inf`**, because the code left the per-cell mask to
>   drop them; invisible at decode, catastrophic at prefill, where a `-ub 2048`
>   ubatch writes all its cells before the graph runs and the ubatch's first
>   token would have spent its entire 513-block budget on future blocks. A
>   third, found only because the first run crashed: with the fast path,
>   `cell_blk` becomes an input **no graph node reads**, so ggml-alloc never
>   allocates it and `set_input_qsa` dereferences a null buffer
>   (`ggml-backend.cpp:206: GGML_ASSERT(buffer)`). Kept in the graph with one
>   `ggml_build_forward_expand`.
> - **Correctness, four ways, and the strongest one needed no GPU.**
>   `staging/work/qsa_blk_index_equiv.py` reimplements `set_input_qsa`'s
>   grouping, bias and `blk_cells` and both selection tails, then compares the
>   **attended** cell set, resolving every tie class both ways:
>   **exact set equality** when the budget covers the context; symmetric
>   difference **<= r = 4 cells of 2051** when block scores are distinct (pure
>   whole-block rounding -- a strict *superset* at the decode shape); symmetric
>   reshuffling only inside the `ggml_relu` zero-score tie class; and **no
>   structural loss with 2-4 interleaved sequences in a unified cache**, which
>   the old `idx%r` slot assignment could not have given. On the GPU,
>   `test-backend-ops test -b SYCL0 -o QSA_INDEXER_BLK`: **5/5 pass** against
>   the CPU reference at `max_err = 0.0`, including an `n_tps = 4, n_stream = 2`
>   shape. Quality: **PPL 2.6174 +/- 0.060 (per-cell) vs 2.6028 +/- 0.060
>   (block)** on the same corpus in the same session -- nominally *lower*, and
>   a quarter of the same arm's own run-to-run spread.
> - **Measured value, isolated (`test-backend-ops perf`, `logs/qsa-blocktopk/`).**
>   The per-cell arm reproduces doc 12 section 1's slope to **0.08%** across
>   days, which is what makes the rest comparable. Chain cost per QSA layer:
>   **1.16x at 8K falling to 1.10x at 614K** at decode shape, but **1.46x /
>   2.31x / 2.52x at 8K / 32K / 64K at prefill shape** (`n_tps = 512`), because
>   the per-cell segment scales with `n_tps` and the block-pooling half does
>   not. On top of `docs/research/12` 5.1's pooled-key cache it is
>   **1.40x -> 4.25x, growing with depth**.
> - **The slope line is the result.** us per KV token per layer, `n_kv >=
>   32768`: per-cell **0.013078**, block **0.011905** (1.10x), pooled cache
>   **0.001558** (8.39x), **pooled cache + block 0.000320 (40.84x)**. So this
>   removes **80% of the slope the pooled-key cache leaves behind** -- which is
>   exactly right, because what 5.1 cannot touch is the `n_kv`-wide per-cell
>   tail this deletes.
> - **And that flips doc 12 section 6.** With the measured arms in
>   `staging/work/decode_scaling_model.py`: sparse FA + pooled-key cache alone
>   is **+14.9% at 300K / +29.8% at 600K** (misses both bars); **plus block
>   top-k it is +3.3% / +6.1% -- inside both bars on measured numbers only.**
>   Doc 12 had needed a fourth, wholly estimated fusion step to get there. The
>   f16 pooled cache and the fusion step are now headroom, not requirements.
> - **Standalone value today is ~2% and should not be oversold.** 1.11x on the
>   chain at 118K is 18.43 -> 16.63 ms of an 89.7 ms decode token (**2.0%**);
>   at 307,200 it is **2.6%**. On prefill the chain wins more per call but is a
>   far smaller share of the whole: a call-site profile of a 32K prefill puts
>   `TOP_K` at **0.06%** and all `GET_ROWS` at **0.28%** of 126 s of GPU time
>   against `MUL_MAT_ID`'s 87%, and the arithmetic predicts **~1.4%**. **The
>   reason to keep the change is the slope row, not today's number.**
> - **A measurement finding that outlives this change: greedy output on this
>   stack is not run-to-run reproducible.** Same binary, same arm, same
>   `--temp 0 --seed 42`, same prompt, run twice: **5 217 vs 5 223 bytes**. Both
>   arms drift at 32K, and perplexity of one arm moved 2.5914 -> 2.6174 between
>   sessions. **Byte-identical output is therefore not a valid oracle on this
>   box, and today's earlier A/B write-ups that lean on it -- including the
>   top-k update above -- are not supported by that observation alone** (their
>   conclusions may still hold; the evidence does not). `libggml-sycl.so` is
>   byte-identical between `staging/devbin` and this build, so the model code is
>   the only difference, and in one session the *block* arm reproduced the
>   pre-change tree byte for byte while the new tree's own per-cell arm did not.
>   Root cause is not established and is out of this change's scope; the
>   candidates are CPU-side `MUL_MAT_ID` at `-ncmoe 24` and SYCL reductions, and
>   the cheapest next check is a repeat run with `--threads 1`. **This is worth
>   its own pass before any further bit-identity claim.**
> - **End-to-end decode does resolve it, and it matches the prediction.**
>   `llama-bench -p 0 -n 32 -d 32768 -r 3`, recommended config, four arms
>   interleaved: block **19.78 / 19.69 tok/s**, per-cell **19.35 / 19.52** --
>   **+1.54%**, with **every block arm above every per-cell arm** (19.69 >
>   19.52). Saving **0.782 ms/token against 0.756 ms predicted** from the
>   isolated chain measurement, agreement to **3.4%**. The within-arm spread
>   reaches 0.9%, so the interleaving is what makes a 1.5% effect usable; a
>   single pass would not have been.
> - **End-to-end *prefill* could not resolve it, and that is reported rather
>   than papered over.** A 32K prefill A/B with interleaved arms gave the per-cell
>   arm **306.84 / 373.38 / 410.23 tok/s** across three identical-config runs
>   against the block arm's **405.82 / 399.76**; quoting the trailing-arm ratio
>   (+7.1%) would be quoting noise. The box was idle and checked
>   (`nproc` 32, `available` 100 GiB of 123, no other `llama*` process or `sycl`
>   container) -- the spread is the known leading-arm page-cache effect plus
>   ordinary variance, not contention.

## Why this project is a better bet than it looked at first

Two facts, confirmed after this plan was scoped, changed the shape of the
project for the better:

1. **Architecture support is upstream, not fork-only.** `qwen4exp` (Qwen3.8-
   Flash-Next's real architecture name in llama.cpp) is fully merged into
   `ggml-org/llama.cpp` master, and the `moe-cache` fork tracks master closely
   (a 2-line diff against `llama-arch.cpp`). There is no fork-vs-upstream
   architecture split to reconcile.
2. **Almost nothing about the SYCL port is actually blocked.** Of the ~14
   CUDA→SYCL component questions in `docs/research/01`, all but one are
   "portable" or "needs a small, well-precedented amount of new code." The one
   real blocker — SYCL graphs are hard-disabled for any MoE model — already
   has a 39-line open upstream PR (#25089) fixing it, **already validated on
   an Arc Pro B70 with a Qwen MoE model**, stalled only on a maintainer
   rebase request.

The honest risks are elsewhere: an **unresolved correctness regression**
(#24168) on hybrid SSM+MoE architectures — Qwen3.8-Flash-Next's exact
category — on Battlemage; a `GGML_OP_TOP_K` hard cap (`k≤32` vs. the model's
`k≈2051`) that forces part of every full-attention layer onto the CPU on
SYCL specifically (Vulkan doesn't have this limit); and a real, if
comfortably-bounded, host-RAM budget. None of these are cache-design
problems — they're bring-up problems that would exist even without the
cache, which is exactly why they're front-loaded into Phase 0/1 below.

**Update (post-write, `docs/research/02` §3 landed after this plan's first
draft):** this model has actually been run end-to-end on Arc hardware,
including on our exact card class. Two things follow immediately. First,
**both backends had a model-breaking or performance-destroying bug on this
exact architecture until fixes that both landed 2026-09-09** — Vulkan
hard-aborted on Arc Pro B65/B70 (`ggml_vk_fill` workgroup-count overflow,
fixed by #28592) and SYCL silently host-serialized every IQ-quantized MoE
weight (fixed by #28476). **Pin the build to ≥ 2026-09-09 — this is now a
Phase 0 action item, not a nice-to-have.** Second, there is now a real
published no-cache baseline on a near-identical card: **Arc Pro B65 32 GB,
Vulkan, UD-Q3_K_XL, `--n-cpu-moe 25` (static placement, no cache), 30.4 t/s
decode @ 8k depth falling to 13.3 t/s @ 32k** (`docs/research/02` §3.2) — a
*floor*, not a target; it's the number the cache has to beat, same as the
sibling project's static-`-ot`-CPU baseline. Full detail, including a
VRAM-scratch-sizing correction this plan's first draft missed, is in
`docs/research/02` §3.

### Performance target: correcting an early mis-read against the wrong model

The GenerelSchwerz wiki's 16GB-VRAM-Setup page (an RTX 5070 Ti) reports
111-160+ tok/s decode with the cache enabled — but **that number is for
Qwen3.6-35B-A3B (35B total, ~3B active/token), not Qwen3.8-Flash-Next** (180B
total, ~6B active/token, 512 experts/layer). Roughly double the active
compute per token means that comparison overstates what our actual target
model should do, even on identical hardware. **The wiki does have a
same-model run, and it's the number this plan should actually target**
(`../coding-agent/docs/06-moe-cache-research/03-llamacpp-moe-expert-cache-findings.md`,
"The run that matches the video's model" — not yet copied into this
project's `docs/research/`):

> `unsloth/Qwen3.8-Flash-Next-GGUF` UD-Q3_K_XL (our quant tier, per §0.3),
> RTX 5070 Ti 16 GB, `--moe-expert-cache-size 80`, `-c 12288`: **decode
> 47.00 tok/s**, prefill 95.31 tok/s. Explicitly caveated by the wiki itself
> as "one measured run, not an average," no speculative decoding, and — this
> matters — **memory-pressured**: the 83.81 GiB model doesn't fit the box's
> 64 GB RAM either, so the run used lazy mmap with free RAM down to ~2.29
> GiB by the end.
>
> **Update, next day (2026-09-12): `docs/research/12` 5.1 -- the incremental
> pooled-key cache for the QSA indexer -- is implemented for real, proven exact,
> and measured. The indexer chain win holds (7.4x). The end-to-end slope win is
> only 1.39x, because dense `FLASH_ATTN_EXT` is now 93.8% of what is left.
> Full writeup: `docs/research/12` section 10.** Source changes:
> `src/llama-memory-hybrid-idx.{h,cpp}` and `src/models/qwen4exp.cpp`, behind
> `LLAMA_QSA_NO_POOL_CACHE=1` for a same-binary A/B. No kernel, MoE-cache,
> `topk-radix.cpp`, `gated_delta_net.cpp`, `hyper_connect.cpp` or MTP code was
> touched.
>
> - **What it does.** The post-norm, post-RoPE pooled block keys now live in one
>   F32 buffer per QSA layer next to the indexer KV cache, and the graph pools,
>   norms and ropes only the blocks that changed -- **5 blocks at decode against
>   76 800 at 300K** -- writing them back with `ggml_set_rows` and reading a
>   *view of that write*, so the ordering is a graph dependency rather than a
>   build-order convention. This is what DeepSeek-V4's own paper specifies
>   ("computed once for each historical KV entry and stored in GPU memory") and
>   what this code diverged from. Cost: `1 536 * n_ctx` bytes = **180 MiB at
>   122 880, 456 MiB at 307 200, 900 MiB at 614 400**, inside the 1 472 MiB the
>   300K VRAM probe left spare.
> - **The correctness risk was the lifecycle, and it is covered two ways.** A
>   per-block fingerprint (position bucket + first cell) is compared prefix-wise
>   each ubatch, which catches block ids shifting; and every operation that can
>   free, move or re-seat a cell drops the cache outright, which catches what the
>   fingerprint cannot see -- a freed cell refilled by a different token at the
>   same position. Too-large a rebuild falls back to rebuilding every block, i.e.
>   exactly today's cost for that one ubatch. Off for `n_stream > 1`, ranked
>   mrope cells, and multi-ratio models.
> - **A measurement finding that invalidates a method used earlier today, and it
>   is bigger than the change.** The obvious A/B -- run both arms, diff the
>   indexer tensors -- does not work, because **running the *same* arm twice
>   produces different indexer tensors**. Same binary, same fixed token stream,
>   no sampling: cached-vs-cached mismatched in 396 of 480 tensor comparisons,
>   exactly as cached-vs-plain did. The update above recorded that greedy
>   *output* is not reproducible on this stack; **the forward pass itself is not
>   reproducible**, so any cross-process comparison is measuring that noise
>   whatever it is nominally testing. **No future correctness work here should
>   use a cross-run diff.**
> - **So the oracle went inside one graph, and the change is exact.**
>   `LLAMA_QSA_POOL_CHECK=1` builds the from-scratch derivation of the same keys
>   alongside the cached ones, same inputs, same graph;
>   `staging/work/qsa_pool_equiv.cpp` compares row by row. **480/480 comparisons
>   bit-identical over every block the bias can select** (4 003-token prompt, 8
>   prefill ubatches, 32 decode steps, 12 layers). Where rows differ at all,
>   `first_diff_row` is *exactly* `ceil(n_cells/r)` -- the first block past the
>   live region, which `set_input_qsa` writes `-inf` for -- in every single case.
>   A mid-run `seq_rm` + refill comes back **fully identical from the
>   invalidating step onward**. Perplexity on the 116K corpus, 10 chunks, four
>   interleaved arms: **geometric mean over chunks 2-10 of 1.0235 / 1.0235
>   (pool) against 1.0235 / 1.0236 (plain)**, per-chunk spread 0.015-0.103%.
>   Chunk 1 swings 13.62-29.38 with no relation to arm (the two pool arms are
>   both the highest *and* the lowest of the four) -- the same
>   non-reproducibility, showing up where the model is uncertain. **The 3-chunk
>   32K gate the block-top-k update used was run too, and it has no
>   discriminating power**: five interleaved arms gave 2.5424 (warm-up),
>   2.6985 / 2.9168 (pool), 3.5860 / 2.5986 (plain) -- a **within-arm spread of
>   0.987 absolute on `plain` alone, ~70x the 0.015 gap that update's
>   `2.6174 vs 2.6028` rested on**. That gate should not be used again.
> - **Performance: the chain win reproduces, the program table does not.**
>   `llama-bench`, recommended config, arms interleaved behind a discarded
>   warm-up: `d32768` **19.77 -> 21.00**, `d65536` **15.47 -> 17.27**, `d118016`
>   **11.36 -> 13.46 tok/s**. Fitted slope **0.4338 -> 0.3111 ms per 1 000 tokens
>   of depth, 1.39x**, residuals within 0.9%. Subtracting `docs/research/12`
>   section 1's measured FA slope leaves the indexer-plus-host term at **0.1420
>   -> 0.0193 ms/1k, 7.4x** -- inside section 4.1's measured 7-8x bracket.
> - **But section 9.3's isolated arm predicted 0.0038, i.e. 40.8x, and that does
>   not survive a real implementation.** The component model reproduces the code
>   it was built from to **0.6%** and its own proposal to **5x**. The gap is the
>   part of `set_input_qsa` that stays O(`n_kv`) every token -- the grouping scan,
>   the per-token O(`n_blocks`) bias, and the `cell_blk`/`blk_cells`/`bias`
>   uploads -- which a `test-backend-ops` graph does not contain. It is
>   **~4.8 ms of a 133 ms token at 300K** and is now the second-largest
>   depth-proportional term in the decoder. The cache also costs ~1.0 ms/token
>   fixed, so it **breaks even around 6-8K** and is a ~1% loss below that.
> - **Where this leaves the target, stated plainly: this fix alone does not meet
>   the bar, and was never going to.** **300K was measured directly, not
>   extrapolated** (`-d 307200 -r 1 -ncmoe 32 -ub 2048`, which is what 300K needs
>   per `docs/research/12` section 6's VRAM probe; the pooled buffer takes 456 of
>   the 1 472 MiB it left spare): **5.70 -> 7.35 tok/s, 1.29x**, against the
>   `-ncmoe 24` fit's 5.89 / 7.51 / 1.28x -- **2-3% agreement across an `-ncmoe`
>   change and a 2.6x extrapolation in depth**, which is what makes the 600K
>   projection (**3.30 -> 4.37 tok/s, 1.33x**, still at the `-ncmoe 24` floor and
>   so an upper bound) worth quoting. Against a stated target of 17-25 tok/s at
>   600K. `docs/research/12` section 4.2 called the
>   pooled-key cache and sparse FA "a matched pair, not substitutes"; that was a
>   projection and it is now a measurement. **`docs/research/10`'s Design B is
>   the next task and is the whole remaining problem**: it is 93.8% of the slope
>   that is left, it is unbuilt, and nothing else in the program moves it. With
>   it on top, the same measured fit gives **22.65 tok/s at 300K (+15.0% vs 8K)
>   and 19.97 at 600K (+30.5%)** -- inside the user's absolute 17-25 tok/s band,
>   but three times section 9.4's "+3.3% / +6.1%" relative claim, for the one
>   reason above. Treat section 9.4's table as retired.

> **Update, same day (2026-09-12): `docs/research/10`'s Design B -- real sparse
> attention restriction in the SYCL decode path -- is implemented, validated
> three ways and measured. Decode at 300K goes 7.38 -> 21.08 tok/s (2.86x), and
> decode is now flat with depth to within 3.2% out to 116K. This is the piece
> the update above called "the whole remaining problem". Full writeup:
> `docs/research/10` ADDENDUM (2026-09-12).** Source changes: new
> `ggml/src/ggml-sycl/fattn-sparse.{cpp,hpp}` (411 lines), 43 lines in
> `ggml/src/ggml-sycl/fattn-common.hpp`, 22 in `src/models/qwen4exp.cpp`, 310 in
> `tests/test-backend-ops.cpp`. Behind `LLAMA_QSA_NO_SPARSE_FA=1` (model side)
> and `GGML_SYCL_FA_SPARSE=0` (backend side) for same-binary A/Bs. No oneDNN /
> MKL / MoE-cache / `topk-radix.cpp` / `gated_delta_net.cpp` /
> `hyper_connect.cpp` code was touched.
>
> - **What it does, and it is real compute avoidance, not a post-hoc mask.** The
>   finite-mask columns are compacted into an index list, the selected K and V
>   rows are gathered into a compact buffer, and the **unmodified** TILE kernel
>   runs with an effective `ne11 = GGML_PAD(n_kv_max, 256) = 2304` instead of
>   `n_kv`. The KV loop bound itself shrinks (`k_VKQ_max` falls back to `ne11`
>   when `KV_max` is null), so the skipped blocks are never loaded and never
>   computed against. Decode only (`Q->ne[1] == 1`): one compact copy needs one
>   shared column set.
> - **The design departs from `docs/research/10` section 4.2 in the one place
>   that carried the numeric risk.** That scoping costed a per-KV-type
>   gather-dequant kernel. Instead the gather is a **type-agnostic 32-bit byte
>   copy** of whole cache rows into a compact buffer *in the cache's own type*,
>   and the existing `ggml_get_to_fp16_sycl()` converter then dequantizes that
>   buffer -- `n_rows` tall rather than `n_kv` tall. No new dequantization code
>   exists, and an f16 cache needs no special case.
> - **Correctness, three ways, and none of them a cross-run comparison** (per
>   the finding in the update above that the forward pass itself is not
>   reproducible across processes). (1) `test-backend-ops test -b SYCL0 -o
>   FLASH_ATTN_EXT`: **23/23 sparse cases pass, zero new failures** -- the CPU
>   reference ignores the hint and `init_tensor_kq_mask_sparse` builds a
>   genuinely sparse mask, so a dropped column shows as NMSE against a dense CPU
>   result. Six qwen4exp-shaped cases were added (head 256 / gqa 12, f16 and
>   production `q4_0`, two-sequence, budgets 513 / 2048 / 2052, plus a dense
>   fallback arm). (2) An **in-graph oracle in the real model**:
>   `LLAMA_QSA_SPARSE_CHECK=1` builds a second `build_attn_mha` with the hint
>   off on the same mask in the same graph, and
>   `staging/work/qsa_sparse_equiv.cpp` compares them row by row --
>   **240/240 comparisons pass, worst NMSE 1.7e-06 against a 5e-4 bar**. The
>   control is what makes it load-bearing: the prefill ubatches take the dense
>   fallback on both sides and come back **bit-identical (NMSE exactly 0)**, so
>   two dense FA calls in one graph agree to the bit and the decode spread is
>   attributable to the restriction's floating-point reassociation. (3) A
>   backend contract check (`GGML_SYCL_FA_SPARSE_CHECK=1`) that aborts if any
>   mask row carries more finite entries than the `n_kv_max` the model promised;
>   on for the whole of (2) and never fired.
> - **One pre-existing failure, proven pre-existing.** The FA suite's single
>   FAIL (`hsk=256 nh=2 nr23=[16,1] kv=1025 nb=64 q8_0 permute=[0,2,1,3]`,
>   `n_kv_max = 0`) **reproduces on `staging/devbin-pool`, the 13:50 build that
>   predates this work entirely**. It is a dense bug, not this change's.
> - **`llama-perplexity` cannot gate this change, and that is structural rather
>   than an omission.** Perplexity is pure prefill (`Q->ne[1] = 2048` per
>   ubatch); the gate rejects it and the sparse path never executes, so a
>   perplexity A/B here would measure only the known cross-run noise. The
>   deeper in-graph run also could not be made to fit -- check mode builds a
>   second FA node per layer and so doubles the f16 staging reservation, OOMing
>   above ~16K of context. **Depth coverage therefore rests on
>   `test-backend-ops`, where the CPU-reference comparison runs at `n_kv` up to
>   614 400 (600 compaction tiles).**
> - **The kernel number, and it is larger than any estimate this project made.**
>   `test-backend-ops perf`, production decode shape, `q4_0` KV: dense
>   **7 507.15 us/call at `n_kv = 307 200` against sparse 73.88 (101.6x)**, and
>   **14 995.09 vs 78.70 at 614 400 (190.5x)**. The dense `118 016` row
>   reproduces `docs/research/10` section 2.2 to **0.3% on a different day**.
>   The slope goes `0.024437` -> `1.516e-05` us per KV token per layer,
>   **1 612x shallower**; the residual is the two-pass mask scan. Over 12 layers
>   FA is now **0.83-0.94 ms/token, flat**, against 90.1 ms dense at 300K.
> - **End to end, arms interleaved behind a discarded warm-up, with a trailing
>   control that reproduces the leading arm to 0.4-1.4%** (`-ngl 99 -fa 1 -ctk
>   q4_0 -ctv q4_0 -ub 2048 -lzm off -lm none`, `staging/work/qsa_sparse_bench.sh`):
>
> | depth | `-ncmoe` | dense | **sparse** | x |
> |---|---|---|---|---|
> | 8 192 | 24 | 25.07 / 25.41 | **25.36** | 1.00 |
> | 32 768 | 24 | 21.18 / 21.04 | **25.43** | **1.20** |
> | 65 536 | 24 | 17.39 / 17.47 | **25.49** | **1.46** |
> | 118 016 | 24 | 13.56 / 13.62 | **24.56** | **1.81** |
> | **307 200** | 32 | **7.38** | **21.08** | **2.86** |
>
> - **The dense 300K arm lands on the update above's pre-registered projection
>   of 7.51 tok/s to 1.7%**, at a depth 2.6x past anything that model was fitted
>   from. That is what makes the sparse column believable rather than an
>   artifact.
> - **Against the user's bar, both ends.** Short context is **25.4-26.5 tok/s**,
>   inside the ideal 25-30 band. **300K is 21.08 tok/s measured, inside the
>   stated 17-25 tok/s floor. 600K is 12.42 tok/s measured, below it** (see the
>   600K bullet below for why, and it is not attention). Decomposed: of the 8.0 ms/token gap from the 8K arm,
>   **~3.5 ms is depth** (the measured sparse slope, 0.01174 ms per 1 000
>   tokens, so **+8.9%** from 8K to 300K) and **~4.5 ms is VRAM** -- 300K only
>   fits at `-ncmoe 32`, eight more expert layers on the host, which `PLAN.md`
>   already priced at +0.4 ms per layer.
> - **600K was measured too, it misses the bar, and what stands in the way is no
>   longer attention.** `-ub 2048` cannot reach 614 400 (its compute buffer is
>   ~16 GiB there), but **`-ub 1024 -ncmoe 38`** fits and gives **12.42 tok/s at
>   d614400**, against a **same-config shallow control of 22.55 (d8192) / 22.93
>   (d32768)**. That control is what makes the number interpretable: the loss is
>   **+82% in ms/token** (44.0 -> 80.52), i.e. a residual depth slope of
>   **0.0602 ms per 1 000 tokens -- 5.1x the 0.0119 the `-ncmoe 24` arm measures
>   over four depths.** **12.42 is below the stated 17-25 tok/s floor at 600K.**
>   It is provably not attention: the sparse FA op measures **0.94 ms/token flat
>   at `n_kv = 614 400`**, so under 1 ms of the 36.5 ms/token that depth costs
>   here. **A slope that grows 5x when only `-ncmoe` changes is the signature of
>   a host-side term** -- `-ncmoe 38` puts 38 layers of CPU expert GEMM on the
>   same cores as `set_input_qsa`'s single-threaded O(`n_kv`) grouping scan and
>   block bias (`docs/research/12` section 10.3 measured that at 0.0155 ms/1k).
>   The second candidate, never isolated by anything in this project, is the
>   **KQ-mask build in `build_attn_qsa` itself** (`ggml_fill(-INF)` +
>   `ggml_set_rows` + `ggml_add` over the whole `n_kv`-wide f16 mask, 12 times
>   per token). Both are the next task; neither is sparse FA's. At `-ncmoe 24`'s
>   slope 614 400 would sit at ~46.3 ms (21.6 tok/s), which brackets how much of
>   the 600K shortfall is configuration rather than depth.
> - **What is now the largest depth-proportional term.** The residual slope is
>   **0.01174 ms per 1 000 tokens**, of which the mask scan is ~0.0002. The rest
>   is `docs/research/12` section 10.3's O(`n_kv`) host term in `set_input_qsa`
>   (the grouping scan, the per-token block bias, the input uploads). It is the
>   next target and the same "only the tail changes" argument applies to it.
> - **Two claims of `docs/research/10` section 4.2 that did not survive the
>   code, recorded rather than buried.** The promised 242 MiB VRAM saving does
>   **not** materialize: the f16 staging buffer is reserved at `sched_reserve`
>   time against the `n_tokens = n_ubatch` worst case, where sparse does not
>   apply, so the compute buffer is byte-identical with and without the change.
>   And `Q->ne[1] == 1` is stricter than it reads in a server: split streams
>   (`--parallel N`) are covered, a unified cache batching N sequences into one
>   ubatch is not and falls back to dense.
> - **An operational hazard that bit this pass and will bite again: the
>   `gdnbuild` build container is shared between agents.** Another agent's
>   `staging/work/devbuild.sh` run at 01:59 picked up source files this pass had
>   copied into `/app` and published the result to `staging/devbin`, so for a
>   period `staging/devbin` held a half-tested build of *this* change that its
>   owner did not know about. It was resynced to the finished build. **Copy into
>   the container and build only immediately before measuring, keep results in
>   a per-change `staging/devbin-<name>`, and treat `staging/devbin` as shared
>   mutable state.**
> - **Box state, per standing practice** (no result here needed a contention
>   explanation): `nproc` **32**, load average 3.5, the loudest tenants
>   `unbooru-tagger` 100% + `java` 64% = **~5% of 3200%**, `free` **available
>   29 GiB of 123** (the `-ncmoe 32 -lm none -lzm off` arm pins ~61 GiB of host
>   tensors by design), no other `llama*` process or `sycl` container during any
>   measurement.
>
> **The real 116K server run confirms it with the sampler in the loop, and the
> quality holds.** Same prompt (116 277 tokens), config and server-side timers as
> `logs/long-context-120k-lazyoff/`, `GGML_SYCL_FA_SPARSE_CHECK=1` on throughout;
> artifacts in `logs/long-context-120k-sparse/`, driver
> `staging/work/run_116k_sparse.sh`. **Decode 11.15 -> 23.75 tok/s (2.13x)**,
> TTFT 4 min 55 s -> 4 min 37 s. Prefill reads 394.14 -> 419.70 tok/s, which is
> **not** this change (the gate rejects prefill) and should be read as unchanged
> run-to-run variance. The `tg_3s` interval spread stays tight at **1.10x**
> (22.47-24.72). The saved reasoning trace correctly reads the 116K prompt's real
> structure -- the per-entity function families, the per-entity constants and
> every planted anti-pattern -- and proposes the parameterized refactor, no
> degeneracy. And the contract check verified on **400 decode steps x 12 layers
> at `n_kv = 118 016`** that no mask row exceeded the promised bound, which is the
> production-depth coverage the in-graph oracle could not reach.
>
> **Cumulative decode at 116K across the four fixes now stands at 5.34 -> 23.75
> tok/s (4.45x)**, on the same benchmark and the same original baseline.

> **Update, same day (2026-09-12): context-length-aware expert placement is
> implemented -- as `--context-tiers`, the discrete-tier reload
> `docs/research/13` recommended, not the mid-inference tensor migration that
> document rejected. It works, and the honest verdict is that it buys a
> capability, not a speed-up: ~5-6% for short requests, against a switch cost of
> one full model reload plus a full reprefill. Full writeup:
> `docs/research/13` section 9.** Source changes are server/CLI only --
> `tools/server/server-context.cpp` (190), `common/arg.cpp` (47),
> `common/common.h` (13) -- **~250 LOC, zero ggml, zero model, zero kernel code**,
> inside doc 13's own 250-350 estimate. Nothing in `topk-radix.cpp`,
> `moe-cache.cpp`, `fattn-sparse.*`, `gated_delta_net.cpp`, `hyper_connect.cpp`,
> `llama-memory-hybrid-idx.cpp` or the QSA indexer was touched.
>
> - **What it does.** `--context-tiers CTX:UB:NCMOE,...` gives the server a list
>   of placements. It starts on the first one, and before each arriving request
>   gets a slot it computes the tier that request's own exact token count needs
>   (`task.n_tokens()` plus the request's `max_tokens`, or
>   `--context-tier-reserve`) and, if that differs from the loaded tier, reloads
>   the model through the **existing** sleep/wake path -- `destroy()` +
>   `load_model(params_base)` with `n_ctx`, `n_ubatch` and the `-ncmoe` override
>   list mutated. No tensor moves, no buffer type changes, nothing is
>   asynchronous, so none of doc 13 section 4's three blockers apply.
> - **Two things doc 13's design sketch got wrong, both found in the code.** Its
>   trigger point (the all-idle gate at `server-context.cpp:2807`) cannot work:
>   by then `launch_slot_with_task` has already put the arriving task in a slot,
>   and a reload there runs `slots.clear()` on the only copy of the in-flight
>   request. The trigger is `process_single_task` just before
>   `get_available_slot`, which has the same "nothing is decoding" guarantee
>   (completions are declined while the queue yields) plus an explicit
>   `is_processing()` check over every slot, and still holds the task. And
>   `load_model` must be entered on its `is_resume` path, or `init()` re-registers
>   the sleeping-state callback and the **second** switch trips
>   `GGML_ASSERT(sleeping != new_state)`.
> - **A hard ceiling that deletes doc 13's own 300K and 600K tiers: this server
>   cannot serve past `n_ctx_train` at all.** `n_ctx_slot()`
>   (`server-context.cpp:4027`) is `min(n_ctx_seq, n_ctx_train)` and the prompt
>   gate at `:3207` rejects `n_tokens >= slot.n_ctx`, so with `n_ctx_train =
>   262144` a 300K prompt is refused whatever `-c` says. The **only** thing that
>   lifted it in this repo's own 300K server run is YaRN --
>   `llama_init_from_model` rewrites `hparams.n_ctx_train = n_ctx_orig_yarn /
>   rope_freq_scale` (`src/llama-context.cpp:3782-3786`), which is why
>   `logs/long-context-300k/server.log` reports `n_ctx_slot = 311296` -- and this
>   plan already measured that those flags destroy the model's output. **So every
>   300K/600K number in this repo is either `llama-bench` (no gate) or the one
>   degenerate YaRN server run**, and the deepest servable tier today is
>   `-c 262144`.
> - **Tier table, every row checked against a measured config.** Recommended:
>   **`--context-tiers 122880:2048:24,262144:2048:30`**. Tier 0 is the production
>   config this plan already recommends and is proven by today's own 116K sparse
>   server run (`logs/long-context-120k-sparse/`, `n_ctx_slot = 122880`), pooled
>   indexer cache included: 26,916.16 model + 1,113.75 KV + 112.57 RS + 3,692.28
>   compute + 180.0 pooled = **32,015 MiB, works** -- a firmer bound than this
>   plan's own 31,835-32,101 bracket, because it is a configuration that served a
>   116,277-token prompt rather than an allocator report. Tier 1 computes to
>   3,928.73 dense+recurrent + 2,376.00 KV + 384.0 pooled + 7,468.3 compute +
>   17,325 experts = **31,482 MiB, 533 MiB *below* the configuration already in
>   production**, which is what makes `-ncmoe 30` the right value (`-ncmoe 29`
>   would be 32,445, above both that figure and the allocator's 32,402). The only
>   estimated term is the compute buffer, interpolated between the **two nearest**
>   measured `-ub 2048` points (4,972.28 at 163,840 and 8,716.28 at 311,296,
>   slope 0.025391 MiB/token) -- worth doing carefully, because the segment below
>   163,840 is steeper (0.031250), so a two-point fit anchored at 122,880
>   under-predicts 163,840's own measured buffer by 3.9%. A third tier at
>   `163840:2048:26` (31,801 MiB today) is real and
>   fits (measured, this plan's `-ub` table) but is **not** recommended: a growing
>   conversation crosses every boundary it passes, so the middle tier costs a
>   second five-minute reload to save four host expert layers (~4%).
> - **And that is the whole value, honestly priced: ~5-6%, not doc 13 section
>   5.3's 1.11-1.32x.** That table compared every depth against the 600K
>   configuration (`-ncmoe 38 -ub 1024`); with the ceiling above, the deepest
>   servable config is only **6** `-ncmoe` steps above tier 0, at the same `-ub`.
>   At this plan's measured +0.4 ms/token per host-resident expert layer, running
>   the deep tier for everything costs 2.4 ms on a 39.43 ms token: **23.91 vs
>   25.36 tok/s at 8K, 1.06x**, and ~3.5% on prefill. The reason the prize shrank
>   is this project's own sparse-FA win -- decode is flat with depth now, so a
>   tier's cost is almost entirely its `-ncmoe`.
> - **Validated in two arms, and the mechanism arm needed no GPU at all.** Arm 1
>   ran the whole six-turn conversation on the 21 GiB `Qwen3.6-35B-A3B` MoE with
>   `-ngl 0 --device none -t 4`, which exercises the identical reload path at a
>   5 s reload instead of a 5 min one -- and could therefore run while the Arc
>   card was busy with another workstream. Tiers `1024:512:4,8192:1024:8`,
>   reserve 64, down-turns 2 (`logs/context-tiers/cpu-arm/`,
>   `staging/work/tier_switch_run.sh`): **exactly 2 switches for 2 boundary
>   crossings and not one spurious reload.** The server came up on tier 0
>   (`n_ctx_slot = 1024`) with no `-c` given; turn 3 (1,539 tokens) escalated in
>   **5.44 s** and reported `n_ctx_slot = 8192`; turn 4 answered *from the
>   document it had just re-read* ("one for user operations and one for order
>   operations") with `prompt_n = 32`, i.e. **coherent output and a working prompt
>   cache on the far side of the switch**; turn 5 held the deep tier for one
>   shallow request and turn 6 released it in **5.27 s**. No assert, abort or
>   error, and box memory across both reloads went 45.7 -> 46.6 -> 42.4 GiB with
>   no monotone growth, so a reload gives the host tensors back.
> - **Arm 2 ran the recommended table on the production model and flags, and its
>   control result is the one the task asked for: a session that never crosses a
>   boundary is the static configuration.** `--context-tiers
>   122880:2048:24,262144:2048:30`, `-ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -lm none
>   -lzm off -fit off -np 1` (`logs/context-tiers/prod-arm/`). Load **219 s**
>   (3 min 39 s, the `-lm none -lzm off` cost this plan priced at ~4.9 min, warm
>   page cache). Three turns, **zero switches**: 64 tokens at 22.14 tok/s decode,
>   then a real 32,019-token document at **480.92 tok/s prefill / 25.56 tok/s
>   decode**, then a follow-up on the cached prefix (`prompt_n = 32`) at 26.12
>   tok/s. Both numbers sit in this plan's own static-config band (408.39 tok/s
>   prefill at 32K on 2026-09-11; 25.07-25.41 tok/s decode at 8K in the sparse-FA
>   addendum), i.e. **tiering costs nothing when it does not fire** -- which the
>   code predicts, since the policy returns on an integer compare.
> - **Cost of a switch, and the breakeven that decides how it should be used.** A
>   switch is `destroy()` + `load_model()`, so it costs a full load: **5.3-5.4 s**
>   on the 21 GiB arm (mmap, warm), and **~219 s / 3.5-5 min** at production
>   flags, where `-lm none` means there is no mmap to keep the previous load
>   resident. On top of that the prompt cache does not survive, so the triggering
>   turn reprefills its whole conversation -- at the measured 480.92 tok/s that is
>   ~4.5 min for a 130K conversation. **Against a 2.4 ms/token decode gain (6 host
>   expert layers x this plan's measured +0.4 ms), a 219 s reload needs ~91,000
>   decoded tokens to pay for itself** -- roughly 45 turns of 2,000 tokens. So
>   `--context-tier-down-turns` defaults to **64** and should not be lowered:
>   going *down* a tier is a speed bet that almost never pays, while going *up* is
>   not an optimization at all -- without it the request is refused
>   (`ERROR_TYPE_EXCEED_CONTEXT_SIZE`), so its cost buys a capability.
> - **Verdict: keep it, for the capability.** One server can now start in the fast
>   config and reconfigure itself up to the model's 262,144-token ceiling instead
>   of refusing the request or being permanently configured deep. The steady-state
>   prize is the ~5-6% above, so **this is not a throughput fix and should not be
>   sold as one.** Doc 13 section 8's ordering survives: the three items ahead of
>   this one are still ahead of it, and its item 3 (no validated rope strategy
>   past 262,144) is now *more* pointed -- the server cannot even accept a prompt
>   past `n_ctx_train` without the YaRN flags this plan measured as destructive,
>   so no placement machinery reaches 300K until that is closed.
> - **One open item, deliberately left open: the real crossing at production
>   boundaries (arm 2b) never completed, because the box crashed.** Still
>   unmeasured: an end-to-end switch at a production tier boundary, and tier 1's
>   VRAM row -- the one estimated number in the table above. **The falsifiable
>   prediction**: `-ncmoe 30` at `-c 262144 -ub 2048` fails only if the compute
>   buffer exceeds **8,001 MiB against the 7,468 interpolated** between the two
>   nearest measured points (7.1% high), which would need the buffer's slope to
>   stop falling with depth against three measured points that say it falls
>   (0.031250 MiB/token below 163,840, 0.025391 above). One load under `-lv 4`,
>   reading `sched_reserve: SYCL0 compute buffer size`, settles it in ~4 minutes
>   and no code.
>
> **Operational hazard, and it is the most serious one this project has hit: the
> box crashed and rebooted, and the sequencing error was mine.** Arm 2b was
> started **37 s after `docker stop`** was issued to arm 2a's container, with no
> check that the process had exited and given back its **~53 GiB** of pinned host
> tensors. Arm 2a's own sampler recorded the box at **87.9 GiB used / 38.4 GiB
> available** immediately before (`logs/context-tiers/prod-arm/host-mem.log`), so
> a second 53 GiB load was begun against 38.4 GiB of headroom. A concurrent
> workstream was loading at `-ncmoe 32` (~88 GiB) as well, which made it worse,
> **but one arm's own arithmetic was already sufficient to overcommit the machine,
> so this is not a scheduling accident.** The box runs unrelated services for the
> user (Immich, Minecraft, Vaultwarden), so a physical reboot is real damage, not
> a lost benchmark.
>
> **Standing rule this adds, and it supersedes the GPU-idle check every run in
> this repo has used.** `ps aux | grep llama` + `docker ps | grep sycl` + "the GPU
> is idle" is **not sufficient**. With `-lm none -lzm off` the binding resource is
> **host RAM**: one load is **~53 GiB at `-ncmoe 24`** and **~88 GiB at
> `-ncmoe 32`**, on a 123 GiB box with ~35 GiB already committed elsewhere -- so
> **two concurrent loads cannot fit and will take the machine down.** Before any
> load: check `free -h`'s `available` against the config's real requirement, and
> after stopping a previous run **wait for the process to be gone and the memory
> to come back** rather than for `docker stop` to return. Sixth time measurement
> hygiene has mattered on this box, and the first time it cost more than a number.

> **Update, same day (2026-09-12, later): the O(`n_kv`) host term in `set_input_qsa`
> was instrumented directly and fixed -- 5.13 -> 2.37 ms/token at `n_kv = 614 656`
> (2.17x, baseline reproduced to 2.8% across two runs). But it is 8.9% of a 600K
> token, not the 5.1x excess slope this document blamed it for; and the headline
> finding is worse than that -- **`d614400` decode on this box is not reproducible
> to better than 1.50x, measured on the *unfixed* code**, so no end-to-end A/B of a
> ~5% effect is possible there at all. Full writeup: `docs/research/12`
> section 11.** Source
> changes: `src/llama-memory-hybrid-idx.{h,cpp}` only, behind
> `LLAMA_QSA_SLOW_PLAN=1` for a same-binary A/B, plus `LLAMA_QSA_PLAN_CHECK=1` for
> the correctness oracle. No kernel, MoE-cache, `topk-radix.cpp`,
> `fattn-sparse.*`, `gated_delta_net.cpp` or `hyper_connect.cpp` code was touched.
>
> - **The term now has a number, and this document's own estimate for it was 1.9x
>   high.** `LLAMA_QSA_HOST_PROF=1` times eleven phases of `qsa_prepare` /
>   `set_input_qsa` with `ggml_time_us()` and accumulates over decode ubatches
>   only, ending a window when the depth changes. At `-ub 1024 -ncmoe 38 -lzm off`:
>   **0.0846 ms/token at `n_kv = 8 448`, 0.2691 at 33 024, 5.058 at 614 656** --
>   linear to within 10% over a 73x range, so **0.00823 ms per 1 000 tokens of
>   depth, against `docs/research/12` section 10.3's inferred 0.0155.** Because it
>   is host CPU time inside named code it is immune to the page-cache and
>   leading-arm effects that have spoiled four end-to-end A/Bs in this project.
> - **Where the 5.06 ms goes, and none of it is what "grouping scan" suggests.**
>   `group_cells` 2.67, `cell_loop` 0.85, **the per-token block bias 0.65**,
>   `blk_enum` 0.33, the three n_kv-wide `assign`s 0.24, the `cell_blk` memcpy 0.14.
>   The bias is the interesting one: it tests `cells.seq_has(bid_cell[b], seq_id)`
>   per block, and `llama_kv_cells::seq` is a `std::bitset<256>` -- **32 bytes per
>   cell, 19.7 MiB at 614 400** -- so the test is a stride-128B walk of the whole
>   array every token. That phase grows **67x** between 33 024 and 614 656 while
>   `n_kv` grows 18.6x, because at the shallow point the array is in L2.
> - **Four changes, all of them deleting work rather than parallelising it**, and
>   the premise that the scan "competes for CPU threads with the CPU-resident
>   expert workload" does not survive reading the call order: `set_input` runs to
>   completion *before* `graph_compute`, so the two are additive, not concurrent.
>   (1) **Bucketed grouping**: a block is keyed on (sequence set, position bucket),
>   so when the stream's cells hold at most one sequence the per-bucket list search
>   can only match its one node and the group *is* the bucket -- the `n_kv`-wide
>   `cell_grp` column and the list walk disappear. Same algorithm, not an
>   approximation. (2) **Drop two dead columns**: `cell_blk` is read only by the
>   per-cell selection and `blk_of` only by the per-cell bias, and the
>   block-granularity selection that landed yesterday leaves both unread -- three
>   `n_kv`-wide passes gone, including a 2.5 MB memcpy per token. (3)
>   **Run-length bias**: bucketed grouping makes `bid_idx` strictly ascending, so
>   with one sequence the row is three runs, written by two binary searches and
>   three `std::fill`s. **19.7 MiB of strided bitset reads become a 0.6 MiB
>   memset: 0.654 -> 0.022 ms, 30x.** (4) **Shift/mask for a power-of-two ratio**,
>   plus the scratch moved into the memoized plan.
> - **(4) was not an afterthought and the profile is what caught it.** `ratio` is a
>   runtime value, so `idx/r` and `idx%r` are 64-bit divides, and the grouping does
>   both per cell per pass. The first version of the fix traded a sequential array
>   read for one extra divide per cell and **made `cell_loop` worse, 0.851 ->
>   1.465 ms** -- measured cost of one divide per cell at this depth, **0.61 ms**
>   (~4.5 cycles/cell). With shift/mask the same phase is 1.009 and `group_cells`
>   goes 1.563 -> 0.829. Without the instrument this would have shipped as a
>   regression in the largest phase.
> - **Net: 5.128 -> 2.368 ms/token at `n_kv = 614 656`, 2.17x** (baseline is the
>   mean of two runs that agree to 2.8%), with per-phase
>   `group_cells` 3.21x, `alloc` 3.13x, `bias` 30.3x, `cell_blk` copy gone. The
>   host-term slope goes **0.00823 -> 0.00407 ms/1k**. `cell_loop` is still 1.19x
>   *worse* than the general path and is the remaining headroom: folding the
>   `blk_cells` write into the first pass would delete it, since for a
>   sequentially-filled single-sequence cache the block id equals the bucket and the
>   compaction is the identity.
> - **Correctness: an in-*call* oracle, which is stronger than the in-graph one the
>   pooled-key cache had to use.** Both paths are pure host functions of the same
>   `llama_kv_cells`, so `LLAMA_QSA_PLAN_CHECK=1` builds a second
>   `llama_qsa_plan::stream` with the general algorithm **on the same cells inside
>   the same call** and `GGML_ASSERT`s equality of all twelve fields (`n_bid`,
>   `dead_bid`, `n_spare`, `ranked`, `cell_blk`, `blk_cells`, `blk_of`, `bid_idx`,
>   `bid_cell`, `bid_slot0`, `rank`, `order`); the run-length bias row is compared
>   against the general per-block loop by `memcmp` over `n_blocks` floats, which is
>   exact here because the only values are `0.0f`, `1e9f` and `-INFINITY`. A failing
>   assert aborts, so "ran to completion" is the pass. **6/6 sections rc=0**
>   (`staging/work/qsa_plan_check.sh`): empty cache at `d0` (where `n_bid == 0` and
>   only the spare block carries bias) and `d2048`; prefill ubatches at `n_tps = 512`
>   and `2048`, where whole blocks sit past the query and the future-block test is
>   load-bearing; `d8191` (deliberately not a multiple of `ratio * n_ubatch`) and
>   `d16384`; `LLAMA_QSA_CELL_TOPK=1`, the only configuration where `cell_blk` and
>   `blk_of` exist at all, so those two are really compared rather than empty on
>   both sides; and three `llama-perplexity` chunks through one context, which
>   clears and refills cells and so exercises the pooled-key invalidation path.
> - **Perplexity *is* a valid gate for this change, unlike for sparse FA**, because
>   `set_input_qsa` runs on prefill ubatches too. It ran clean (**1.4153 +/- 0.0245**
>   over 3 chunks) but per `docs/research/12` section 10.2 the 3-chunk gate has no
>   discriminating power -- reproduced here for free, since **three runs of the same
>   arm gave 1.4859, 1.4153 and 1.5432** -- so it is a smoke test and the bit-exact
>   assert is the gate. Two gaps recorded rather than buried: **the mrope-ranked path has no
>   driver** (it needs 2-D image positions this text model never produces; ranking
>   is orthogonal to bucketing and the assert would catch it if it ran), and
>   **`llama-cli`, `llama-completion` and `llama-batched-bench` all crash on this
>   model in this fork before this change** -- reproduced on `staging/devbin-sparse`,
>   the build that predates it -- so the multi-sequence general path has no driver
>   either; it is unchanged code reached only when two sequences share one cache.
> - **The finding that is bigger than the fix: `-lm none` costs 600K 1.39x.** The
>   600K bullet above measured **12.42 tok/s (80.52 ms/token)** at `-ub 1024
>   -ncmoe 38 -lzm off -lm none`. The identical command with **`-lm none` dropped**
>   (mmap for the expert tensors, everything else the same) measures **17.27 tok/s
>   (57.91 ms/token)** on the *un-fixed* code, with a **residual depth slope of
>   0.0258 ms/1k against that bullet's 0.0602**. So the "5.1x steeper slope at
>   `-ncmoe 38`, the signature of a host-side term" reading was **2.3x
>   configuration and 1.15x depth**: this document named the right file and the
>   wrong magnitude, and `docs/research/13` section 5.4's "removing it takes 600K
>   to 19.7 tok/s" rested on attributing the whole excess to it. The host term is
>   **31.9% of the residual slope**, not all of it. Mechanism for the flag, offered
>   as hypothesis not measurement: `-lm none` pins ~66 GiB of expert and PLE
>   tensors in `SYCL_Host` USM, which is unevictable, pushes the box to ~100 GiB of
>   123 with swap exhausted, and is not necessarily cached writeback for CPU reads.
>   `docs/research/11` section 7 recommended `-lm none` off a **prefill**
>   measurement (394 vs 252 tok/s), which still stands; nothing had re-checked it
>   against decode at `-ncmoe 38`.
> - **Against the user's bar at 600K, stated with the spread rather than
>   cherry-picked: `-ub 1024 -ncmoe 38 -lzm off` measures 11.50-17.27 tok/s
>   un-fixed, straddling the bottom of the stated 17-25 band, and this change adds
>   2.76 ms/token (4.8% of the faster arm's token) on top.** The honest reading is
>   that **600K is at the edge of the band and this fix does not decide it** --
>   dropping `-lm none` moves it far more (PLAN's 12.42 arm vs 17.27 here) and even
>   that is inside the box's own 1.50x spread. `docs/research/13` section 5.4's
>   "removing the host term takes 600K to 19.7 tok/s" rested on attributing the
>   entire excess slope to it; the term is **31.9%** of that slope. Short
>   context is unchanged: the term is 0.085 ms of a 46 ms token at `d8192`
>   (0.2%), and `d8192` / `d32768` measured 21.56 / 23.32 (slow) against 22.85 /
>   22.62 (fast), i.e. inside this configuration's own 8% leading-arm spread and
>   with no regression.
> - **The deep end-to-end A/B does not resolve, and a trailing control proves that
>   rather than leaving it as a suspicion.** Four `d614400` arms were run at the
>   identical command. **The unfixed code alone measures 17.27 tok/s and 11.50
>   tok/s -- a 1.50x spread between two runs of the same binary and flags -- while
>   its own host term reproduces to 2.8% (5.058 and 5.198 ms/token).** The fixed
>   arms gave 14.22 (before the shift/mask fix) and 1.76 (final). So the
>   end-to-end instrument at this depth has a spread several times larger than the
>   4.8% this change is worth, the 1.76 outlier is the same instability in the
>   extreme rather than a code effect, and **quoting any of these ratios would be
>   quoting noise** -- the same conclusion the block-top-k prefill A/B reached
>   above. The 2.76 ms/token saving is therefore quoted from the host-time
>   instrument, which is the one measurement that reproduces.
> - **The cause of that spread is not established, and the first explanation
>   offered for it was wrong.** `bcache0` at 98.1% util and `md0` at 97% looked
>   exactly like the documented neighbour-I/O confound, but measuring the
>   neighbours instead of assuming them killed it: **`deluged` was reading
>   352 KB/s and writing nothing, and `unbooru-tagger` was at 0 KB/s** (its 490 GB
>   is cumulative over five hours since boot). The utilization was **our own model
>   load**. That is `docs/research/11` section 7.6's own rule -- "array utilization
>   is a symptom, not a cause" -- ignored for a third time in this project, and a
>   user caught it. Our process also took **3 major faults in the decode minute**,
>   so it was not paging. What remains unexplained is a session-order effect: the
>   second arm of a campaign was slower in both campaigns regardless of which code
>   it ran.
> - **A standing-practice finding that outlives this change, and it explains why
>   every run in this project pays for its model twice.**
>   `/sys/block/bcache0/bcache/sequential_cutoff` is **4.0M**, so bcache
>   **bypasses** any read above 4 MiB -- and the GGUF parts are 49.5 GB and
>   32.4 GB read sequentially. The model is therefore **never cached on the NVMe
>   tier at all** (`bypassed` 20.8 G total, 512 M in five minutes) while
>   `cache_available_percent` is **97** on a ~2 TB cache device: capacity was never
>   the constraint, the cutoff was. `dirty_data` is **19.2 G** against
>   `writeback_percent = 10`, so writeback also competes with our reads on `md0`.
>   `echo 0 > sequential_cutoff` (root) would put the 76 GiB model on the cache
>   tier and is plausibly worth minutes per run on every measurement this project
>   still has to make.
> - **Box state, per standing practice.** `nproc` **32**, `free` **available
>   77-86 GiB of 123** throughout, **0-12 major faults/s** during the measured
>   windows, no other `llama*` process or `sycl` container during any measurement.
>   Deliberately **not** `-lm none` for these runs: two concurrent ~60 GiB loads
>   took the whole box down earlier in the day, so every run here uses mmap for the
>   experts (~30 GiB) and the one arm that needed the pinned config is quoted from
>   this document's existing measurement rather than re-run.
> - **One self-inflicted hazard worth recording.** Setting
>   `SYCL_CACHE_PERSISTENT=1` / `NEO_CACHE_PERSISTENT=1` with a mounted cache
>   directory, to amortise kernel compilation across runs, left a corrupt cache
>   entry when the box went down mid-write -- after which **every** run, including
>   binaries that had worked minutes earlier, died with a general protection fault
>   in glibc's EVEX `strcmp` on a non-canonical pointer, ~7 s in and before the
>   first table row. Deleting the two cache directories fixed it immediately.
>   `staging/work/rundev.sh` now carries the warning. Do not persist the Intel
>   compiler cache on this box.

> **Update, same day (2026-09-12, later): the "YaRN destroys this GGUF" verdict is
> RETRACTED, and -- separately and more importantly -- unscaled RoPE
> extrapolation at 300K is measured and it is GOOD. Full writeup:
> `docs/research/15-yarn-and-long-context-rope.md`; the superseded verdict in
> `logs/long-context-300k-benchmark-report.md` section 4.1.1 is annotated in
> place.** No source changes at all: this pass is measurement plus a
> read-through of the RoPE path. Artifacts in `logs/yarn/`, drivers
> `staging/work/yarn_{ppl_ladder,deep,deep2,beta_sweep,server_repro}.sh`.
>
> - **The headline this plan actually needed: every 300K/600K speed number in
>   this repo was taken on the unscaled-extrapolation path, and nobody had
>   looked at the output. Now measured, and it is not degraded.** A real
>   **305,759-token** prompt (sibling generator at `n = 400`, this model's own
>   chat template), `-c 311296 -ncmoe 32 -ub 2048`, **no rope flags**, greedy:
>   coherent, on-task, and *quantitatively* right. It named all **20** entity
>   types in the generator's own declaration order, identified the "400+
>   near-identical function groups" structure and its eight-member function
>   group, and recovered the **ranges** of every randomised parameter --
>   multiplier 2-9, addend 1-50, divisor 1-7, tax 0.10-0.9, cache-key default
>   0-399 -- each exactly matching `generate_messy.py:165-169`. Those ranges are
>   the min/max of 400 independent draws spread through 1.07 MB of prompt, so
>   they cannot be read off a local window. **Today's 21.08 tok/s at 300K
>   corresponds to output somebody would actually want.**
> - **And 600K holds too, measured, at 2.34x the native ceiling.** A fresh
>   **612,689-token** prompt (generator at `n = 800`, 800 blocks verified),
>   `-ncmoe 38 -ub 1024 -c 614400 -lzm off`, no rope flags: prefill **138.26
>   tok/s** (73 min 51 s), decode **4.71 tok/s**, and the output is coherent --
>   "~800+ near-identical copies" (exactly 800), all 20 entities in generator
>   order, the full eight-member function group, and **every one of the six
>   generator globals by name** (`data`, `DATA2`, `temp`, `counter`,
>   `GLOBAL_CACHE`, `errors_list`), which appear only in the first ~100 tokens
>   of a 2.15 MB prompt. It did not volunteer the numeric *ranges* the 305K arm
>   did, which may be mild degradation or may just be the `-n 250` budget going
>   to a longer structural analysis; n=1. **So the whole 300K-600K band is
>   validated on the unscaled path, and the 12.42 tok/s at 600K is a real
>   number about a working model.**
> - **The YaRN collapse does not reproduce, and the retraction is based on the
>   original arm itself, not a lookalike.** `logs/long-context-300kcfg-116kprompt/`
>   re-run verbatim -- same `llama-server`, same `--rope-scaling yarn
>   --rope-scale 1.1875 --yarn-orig-ctx 262144`, same `-c 311296 -ncmoe 32
>   -ub 2048`, same 116,277-token prompt, same client at `temperature 0`, one
>   variable changed (the binary) -- returns **974 characters of correct on-task
>   reasoning naming all 20 entities**, against `b` + 199 `/` before
>   (`logs/yarn/server-repro-yarn/`, decode 16.58 tok/s). YaRN at 305,759 tokens
>   is coherent too. **The one variable is the binary**: the degenerate runs
>   predate the block-granularity QSA top-k (21:45) and the pooled indexer-key
>   cache (23:13), both of which landed inside `build_qsa_top_k` -- the function
>   that holds the indexer's own RoPE application *and* the `ggml_top_k` that
>   decides which 2,051 of `n_kv` cells attention may see. Most plausible
>   mechanism by a wide margin; **not proven**, and per the next bullet probably
>   not cleanly provable.
> - **The rope path is correct, and three standing hypotheses are retired by
>   reading rather than guessing.** All four `ggml_rope_multi` call sites in
>   `src/models/qwen4exp.cpp` (Q/K `:1144,1150`; the indexer's pooled keys `:890`;
>   the indexer queries `:927`) pass **identical** `n_rot`, `sections`,
>   `n_ctx_orig`, `freq_base`, `freq_scale`, `ext_factor`, `attn_factor`,
>   `beta_fast`, `beta_slow` -- so there is **no main-attention/indexer YaRN
>   mismatch** -- and the pooled keys rope at `bid_idx[b]`, which is the block's
>   **first token position** (`llama-memory-hybrid-idx.cpp:837` = `pb*r`), not a
>   block ordinal, so q and k are in the same units. **`dimension_sections
>   [11,11,10,0]` is a non-issue**: sections only choose which *position row* a
>   band reads, the frequency is still `theta_scale^(iw/2)` over the global dim
>   index that `rope_yarn_ramp` uses -- and for a text batch
>   `llama-batch.cpp:781-788` broadcasts one position into all four rows, so
>   imrope degenerates to plain NeoX rope. `yarn_attn_factor` is **not**
>   double-applied (`llama-context.cpp:196-210` divides out exactly what the
>   kernel re-applies). And `test-backend-ops test -b SYCL0 -o ROPE` is
>   **470/470**, which includes `ext_factor = 0.7465` IMROPE cases at partial
>   `n_dims`. The quantization hypothesis is moot: the same UD-IQ3_XXS is
>   coherent at 305K with and without YaRN.
> - **What YaRN does cost, and the beta sweep this plan has carried as open,
>   closed.** Perplexity, `--chunks 1` so chunk size *is* depth (the tool scores
>   the deepest half, `perplexity.cpp:542`): **8,192 -> 10.61 off vs 13.16 YaRN
>   (1.24x); 32,768 -> 43.10 vs 82.62 (1.92x)**, re-measured 38.19 vs 110.03
>   (2.88x). Real, far outside the error bars, growing with depth -- exactly what
>   the arithmetic predicts, since static YaRN perturbs theta in proportion to
>   position. Betas at `-c 32768`: off **38.19**, `beta 64/2` (`corr_dims
>   [12,20]`) **39.03**, `beta 4/0.125` ([18,26]) **68.21**, defaults 32/1
>   ([14,22]) **110.03**, `--rope-scaling linear` **113.84**. `linear` worst is
>   the only expected ordering. **`beta 64/2` came back indistinguishable from
>   not scaling -- a lead, not a recommendation** (n=1, non-monotone, and it only
>   *matches* no-scaling). **The open item closes as: no setting of the betas
>   makes YaRN worth turning on here.**
> - **The finding that limits all of the above, and every future rope claim on
>   this model.** `build_qsa_top_k` picks **2,051 of `n_kv`** cells -- 0.7% of
>   context at 305K -- via a hard `ggml_top_k` over `relu(q.k)` scores that RoPE
>   feeds, and `build_attn_qsa` restricts attention to exactly those. So a small,
>   smooth theta change produces a *reshuffled context selection*, not a small
>   output change. Consistent with all three of: identical configs re-measuring
>   **12-33%** apart, a **non-monotone** response to beta, and the same flags
>   collapsing one day and being fine the next. This is the same effect
>   `docs/research/12` section 9.5 recorded and the same order as the ~1.5x
>   `d614400` spread measured today on unmodified code. **Standing rule: no
>   single-run verdict about RoPE on this model is safe -- including the original
>   one, and including any here that is not replicated.**
> - **A capability unlock with no code change, which this plan's tier work needs:
>   `llama-server` can serve past `n_ctx_train` today.** `n_ctx_slot()`
>   (`server-context.cpp:4199-4207`) caps at `n_ctx_train`, and the only lift is
>   `llama-context.cpp:3782-3786`, which requires scaling type YARN *and* a
>   changed `rope_freq_scale` -- which is the only reason the 300K server run ever
>   passed `--rope-scale 1.1875`. **`--rope-scaling yarn --rope-scale 1.0001
>   --yarn-orig-ctx <C>` satisfies the gate while applying a rope perturbation
>   four orders of magnitude smaller** (interpolated bands x 0.99990, `mscale`
>   1.00001, largest theta shift at position 311,296 **0.002 rad** against
>   ~1.0 rad for 1.1875). Measured: `n_ctx_slot = 311296`, no capping warning
>   (`staging/work/yarn-slotcap-nearunity.log`) -- **and validated on output, not
>   just the banner**: the same 305,759-token arm with these flags is coherent and
>   joint-best on parameter-range recovery. This retires "the deepest servable
>   tier today is `-c 262144`" from the context-tier update above. The cleaner fix
>   -- stop capping when the operator asked for more, as `llama-bench` and
>   `llama-completion` already do -- is a `tools/server/server-context.cpp` change
>   owned by other work and was **deliberately not made**.
> - **A tooling ceiling worth recording before someone re-derives it:
>   perplexity cannot measure past the native context on this model at all.**
>   `perplexity.cpp:514` reserves `n_ctx * n_vocab` **host** floats per chunk =
>   **159 GB** at `-c 262144`; it dies `std::bad_alloc` after tokenizing (rc=134,
>   measured). The ceiling is ~`-c 98304` on a 123 GiB box regardless of VRAM, and
>   the corpus must also tokenize to `>= 2*n_ctx`. Deep quality work has to use
>   generations, not ppl.
> - **What is left open.** (i) Nothing here is replicated, and per the variance
>   bullet above that is the standing weakness -- the 305K and 600K arms are n=1
>   each. (ii) Whether YaRN *past* 600K (`--rope-scale 2.34375`, where the angle
>   shifts are 2x those measured at 305K) is still benign is untested, and
>   irrelevant unless someone wants YaRN, which nothing here recommends.
>   (iii) The 600K corpus is now staged and tokenizer-verified at
>   `staging/work/bench600k/` (612,680 tokens) if a repeat or a YaRN arm is
>   wanted.

**Honest math on whether our B70 should beat 47 tok/s, not just assumed:**
two effects pull in opposite directions, and "more VRAM → faster" alone
isn't a complete argument.

- *Against us*: raw memory bandwidth. `docs/research/01` §6.3 (issue #26581)
  puts the B70 at **608 GB/s**; the RTX 5070 Ti is **896 GB/s** — the B70 has
  only **~68%** of the bandwidth. Decode on this class of workload is
  bandwidth-bound (issue #26581 shows B70 decode attention is already
  memory-latency-bound, identically on both SYCL and Vulkan), so *per cache
  hit*, the B70 is the slower card, not the faster one. This is the flaw in
  "more VRAM, slower card, therefore ≥50 by basic math" as originally
  stated — the bandwidth deficit is real and has to be argued past, not
  asserted away.
- *For us*: cache hit rate, and avoiding the 5070 Ti run's specific penalty.
  32 GB vs. 16 GB roughly doubles the achievable slot pool (`docs/research/02`
  §5.4 projects ~41% residency at UD-Q3_K_XL on our card vs. the 5070 Ti
  run's much tighter ~14.4 GB total footprint on a 16 GB card at cache-size
  80). A materially higher hit rate directly reduces how often the (slower)
  B70 has to pay a PCIe miss at all. Separately — and this may matter more
  than the slot-pool math — **the 5070 Ti run was RAM-constrained, not just
  VRAM-constrained**: 64 GB RAM was nearly exhausted, forcing lazy mmap for
  a model that doesn't fit, which is its own performance tax independent of
  the GPU. If our box has meaningfully more system RAM (open Phase 0.4
  question — check this now, it's doing double duty as both a hard
  requirement and a plausible source of headroom over the reference run),
  we could avoid that specific penalty entirely, which the 5070 Ti run did
  not get to do.

**Working target for this plan: high-40s to mid-50s tok/s decode is a
defensible goal to design and measure against, not a guaranteed floor.**
Treat 47 tok/s (same model, worse VRAM, better bandwidth, RAM-pressured) as
the reference point every Phase 1/3/4 benchmark should be reported against,
replacing the no-cache B65 baseline as the headline comparison (that number
stays useful as the "did the cache do anything at all" floor, just not as
the ambition). If Phase 3's actual measurement lands meaningfully under 47,
that's a real signal worth investigating (host RAM shortfall reproducing
the 5070 Ti's mmap penalty, a SYCL/Vulkan-vs-CUDA implementation gap, or a
lower achieved hit rate than projected) rather than something to wave off as
"expected for a slower card."

## Update (2026-09-11): measured results land meaningfully under 47, course correction

Phase 3's SYCL cache port is done and correct (device-wide-grid gather fix,
same-call eviction race fixed, TOP_K CPU-fallback fixed, hyper-connection
kernel fusion) but decode tops out at **23.1 tok/s at 256 slots/tensor**
(UD-IQ3_XXS, `Count from one to fifty.` prompt, `-n 300 --temp 0`) — well
under the 47 tok/s target, and 320/384/512 slots all OOM
(`UR_RESULT_ERROR_OUT_OF_RESOURCES`) before reaching it. Per this plan's own
instruction above, that gap gets investigated rather than waved off:

- **Resolved: the UD-Q3_K_XL confound was a stale Docker image, not a real
  bug.** `qwen4exp-moe-cache:sycl-pinned`'s baked-in binary gets 6.5-6.9 tok/s
  on UD-Q3_K_XL + `-ncmoe 25` (matching the old "~150ms/token" folklore that
  made `run_flashnext.sh` switch to UD-IQ3_XXS) -- but a fresh `devbuild.sh`
  build from the *current* checked-out source gets **21.9-23.4 tok/s on the
  identical command**, matching UD-IQ3_XXS's numbers. Whatever the pinned
  image's bug was, it's not present in current source (plausibly one of the
  2026-09-09 fixes above, never independently confirmed since testing moved
  to IQ3_XXS instead of re-checking). **Practical takeaway: rebuild the
  pinned image, or just always use `staging/devbin` for benchmarking** --
  `docker/Dockerfile.sycl`'s baked image is not a reliable measurement
  baseline right now and shouldn't be trusted without rebuilding first.
  UD-Q3_K_XL needs a lower `-ncmoe` ceiling than IQ3_XXS at equal VRAM
  (bigger quant, `-ncmoe 23` OOMs, `-ncmoe 25` is the confirmed-working
  floor) but otherwise both quants land in the same 22-25 tok/s band on this
  card -- quant choice is not the lever that closes the gap to 47.
- **A zero-code static-placement baseline matches or beats the tuned cache.**
  `-ngl 99 -fa 1 -ncmoe 20` (stock upstream flag, no cache, no custom SYCL
  code at all) measured 19.7-25.3 tok/s across repeated runs (noisy, but
  centered above the cache's 23.1) — `-ncmoe 18` and below OOM on this box's
  30GB B70. Plain `--cpu-moe` (everything on CPU, the naive floor) measured
  15.6 tok/s, so the cache and `-ncmoe` are both real wins over doing
  nothing — the finding is that they're *wins of about the same size*, not
  that the cache is failing outright.
- **This is not a surprise the hard way — `docs/00-background.md` §1 already
  flagged it as a known risk before this project started implementation:**
  "Lost at matched VRAM budget in one external report (llama.cpp discussion
  #24528, hybrid CPU/GPU design, 19.7 vs 35.4 tok/s vs. plain resident
  layers) — the fork's all-GPU-execution design is claimed to be what avoids
  that specific failure, but that claim is the fork's own and wasn't
  independently re-verified." Today's numbers are the independent
  re-verification, and they land closer to "the risk is real" than "the
  fork's claim holds": re-fetched directly from GitHub (`gh api
  repos/ggml-org/llama.cpp/discussions/24528`, not just cited secondhand) —
  the RFC's proposed fix keeps `MUL_MAT_ID` on the CPU and has thread 0
  dispatch a GPU batched matvec over cache-*hit* rows while other threads
  compute miss rows concurrently for free, instead of our (and the CUDA
  fork's) design of a synchronous PCIe copy per miss. Never merged upstream,
  but its own measured numbers (4x3090, 13 models, +10%..+57%, 16/16 ≥
  parity) are the best evidence anyone's produced that this failure mode is
  fixable, not fundamental.
- **Decision: `-ncmoe` is the interim production baseline, not the custom
  cache.** Simpler, zero maintenance burden, matches or beats the tuned
  cache today. The cache code (`ggml-sycl/moe-cache.cpp`) is not being
  deleted — it's real, correct, validated work and may still win at a
  different quant/VRAM-budget point once the Q3_K_XL confound above is
  resolved — but it's no longer the thing `run_flashnext.sh`-style scripts
  should default to.
- **New avenue opened, not yet in the original plan: MTP speculative
  decoding.** Unlike the sibling vLLM project's "fights MTP" objection
  (`docs/00-background.md` §4, about a *dense* model on vLLM — doesn't
  transfer here) and unlike the CUDA fork's own "fights speculative
  decoding" caveat (`docs/00-background.md` §1 — specific to a *dynamic*
  per-token GPU-resident cache thrashing under a verify batch's
  union-of-experts read pattern), neither objection applies to a *static*
  `-ncmoe` placement: there's no runtime admission/eviction for a draft
  verify batch to thrash. unsloth ships a real MTP draft-head GGUF for this
  exact model (`unsloth/Qwen3.8-Flash-Next-GGUF/MTP/`, one full qwen4exp
  transformer block, ~4B params) reporting +50% decode elsewhere (23.5→35.7
  tok/s, ~85% acceptance) — but **qwen4exp has zero NextN/MTP support in
  this codebase** (confirmed: `grep nextn src/models/qwen4exp.cpp` returns
  nothing, unlike `qwen3next.cpp`/`deepseek2.cpp`/etc.), because its final
  output head is a 3-tensor low-rank hyper-connection combine
  (`hc_head_norm`/`down`/`up`) instead of the plain `output_norm` every
  existing NextN template assumes. Implementing this (new
  `LLM_TENSOR_NEXTN_HC_HEAD_*` constants, `llama_layer_nextn` struct
  extension, tensor loading, and a draft-graph builder mirroring
  `qwen3next.cpp`'s pattern with the hc-head substituted in) is in progress
  as of this update. Speculative decoding is fail-safe on correctness by
  construction (the base model's verify step rejects any mismatched draft
  token), so a subtly-wrong draft graph costs acceptance rate, not
  correctness — lower risk to attempt than it looks.

**MTP implementation landed the same day, working.** `--spec-type draft-mtp`
now builds and runs for qwen4exp (see the implementation notes above this
line for the tensor/graph details — not repeated here). The one genuine
architecture-specific call: the trunk-to-draft-head hand-off uses the *wide*
hyper-connection residual (`[n_embd, hc, T]`, 10240 floats/token), not the
mixed `[n_embd]` output, confirmed by cross-referencing `deepseek4.cpp`'s
same pattern for its own HC architecture and validated by a 92-95%
acceptance rate. Normal (non-MTP) loading confirmed unaffected (every change
gated behind `mtp_flags`/`n_layer_nextn`, inert otherwise). Not committed.

Measured (IQ3_XXS, `-ncmoe 26` — the tightest split that fits the draft
head alongside the target; `-ncmoe 24` OOMs):

| config | decode tok/s | acceptance |
|---|---|---|
| baseline, no `-md` | 23.6, 24.5 | — |
| `--spec-draft-n-max 2` | 26.3-29.5 | 92.9-94.9% |
| `--spec-draft-n-max 4` | 26.6 | 91.3% |

~1.15x end-to-end — real, but capped because the draft block is itself a
full 512-expert MoE layer (3.3 GB resident), so a draft step isn't cheap
relative to a heavily CPU-offloaded target step. Tried `--spec-draft-cpu-moe`
(push the draft's own experts to CPU, freeing VRAM to lower the target's
`-ncmoe` back toward its solo-best 21) expecting a net win — **it's a net
loss**, 20.6 tok/s at `-ncmoe 24` vs. the 26-29 tok/s baseline: running the
draft's own MoE serially on CPU costs more per draft step than 2 extra
GPU-resident target layers buy back. Reverted; `-ncmoe 26` with no
`-cmoed` stays the best known MTP config.

**MTP + UD-Q3_K_XL: don't use this combination as-is.** Now that the
Q3_K_XL confound above is resolved (stale image, not a real bug), the
natural next check was MTP on the actual 47 tok/s reference quant. Result:
`-ncmoe 26`/`27` OOM (Q3_K_XL's bigger per-layer footprint plus the draft
head no longer fits where IQ3_XXS did), and the first value that loads
without OOM, `-ncmoe 28`, runs at **3.2 tok/s** — roughly 7x worse than
Q3_K_XL alone at `-ncmoe 25` (21.9-23.4 tok/s), far more than 3 extra
CPU-offloaded layers should cost. This is not a tuning problem, it's a
likely real pathology specific to pairing a Q8_0 MTP draft head with a
K-quant (not I-quant) target — worth root-causing later (candidates: KV
cache type mismatch handling between draft/target, near-VRAM-ceiling
thrashing distinct from a clean OOM, or a gap in the new MTP code path
that was only validated against an IQ3_XXS target) but not chased further
this session given IQ3_XXS+MTP already works and the payoff is uncertain.
**Current best known production config remains IQ3_XXS + `-ncmoe 26` +
`--spec-type draft-mtp --spec-draft-n-max 2`, ~26-29 tok/s** — still well
under the 47 tok/s target; the RFC's CPU-driven hybrid `MUL_MAT_ID` design
(above) remains the largest un-tried lever.

## Update (2026-09-13): production `-ncmoe 24` config now OOMs at 116K — VRAM margin regression, new recommended `-ub`

While validating a usage-based static expert-placement mechanism
(`docs/research/14`), the standing recommended production config from
above — `-ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -ncmoe 24 -ub 2048 -c 122880` —
was re-run against the real 116,235-token `chat_prompt.txt` and failed with
`UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` partway through prefill (`Error OP
MUL_MAT`), not at load.

**Root cause: the VRAM margin at this config was always razor-thin, and the
QSA pooled indexer-key cache ate what was left of it.** The table earlier in
this document measured this exact config at 31,835 MiB, marked **ok**,
against a card ceiling bracketed between 31,835 MiB (ok) and 32,101 MiB
(OOM) — a **~266 MiB margin**. That measurement predates the QSA pooled
indexer-key cache (`docs/research/13`/`14` date it 2026-09-12), which adds
`1,536 × n_ctx / 2^20` MiB — **~180 MiB at n_ctx = 122,880** — and is baked
into the model's own hyperparameters (`hparams.dsv4_compress_ratios[il]`,
`src/llama-memory-hybrid-idx.cpp:175`), with no flag to disable it. 180 of
266 MiB alone is close; whatever else has landed since (hyper-connection
fusion, sparse attention scratch) was evidently enough to push it over.

Confirmed empirically, not just by arithmetic: `-ncmoe 24` works fine at 4K
(15.07 tok/s) and at 32K (slow, but completes); only the full 116K depth at
`-ub 2048` OOMs. Dropping `-ub` shrinks the compute-buffer term (this
document's own model, above: it scales with `n_kv × n_ubatch`) enough to
restore headroom — both `-ub 512` and `-ub 1024` complete cleanly at the
same `-c 122880 -ncmoe 24`, same prompt.

**New recommended production config: `-ub 1024` in place of `-ub 2048`,
everything else unchanged** —
`-ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -ncmoe 24 -ub 1024 -c 122880 -lm none
-lzm off`. This is the same shape as this document's own already-documented
"conservative fallback" (previously paired with `-c 163840`; `-c 122880` is
kept here since that is the tier the 116K workload actually needs). **Not a
full re-benchmark**: the one decode number measured while confirming this
(0.71 tok/s over 19 tokens, `-lzm auto`) is too small a sample and the wrong
`-lzm` setting to quote as the new production throughput — re-measure
properly (`-lm none -lzm off`, a real decode run) before updating any
throughput claims elsewhere in this document against it.

`staging/work/bench120k/serve.sh` does not hardcode `-ub` (defaults to 512)
and is unaffected; this regression only bites configs that explicitly pass
`-ub 2048`.

## Update (2026-09-13, later): `-ub 1024` fallback retracted — `-ncmoe 25` restores `-ub 2048` cleanly, measured

The `-ub 1024` fallback above was never re-benchmarked at 116K and was always
the conservative option, not the preferred one — it gives up roughly half of
the `-ub` lever (the sweep earlier in this document put `-ub 1024` at 2.06x
over baseline against `-ub 2048`'s 3.19x at the 32K stand-in shape). Since
the regression's root cause is a ~266 MiB-or-less VRAM overrun, not something
structural to `-ub 2048` itself, the cheaper fix is to free that margin
directly: move one more expert layer to the host.

**`-ncmoe 24 -> 25` was tested against the same real 116,277-token prompt,
same server-side timers as every other 116K number in this document**
(`-ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -ncmoe 25 -ub 2048 -c 122880 -lm none
-lzm off`, `staging/work/run_116k_lazy.sh`, `NCMOE=25`). Clean start, no OOM,
`n_ctx_slot = 122880` as expected:

| | Result |
|---|---|
| Prefill | **417.54 tok/s** |
| Decode | 23.82 tok/s |
| TTFT | 4 min 38.6 s |

This lands inside the pre-regression `-ncmoe 24 -ub 2048` band (394.14-419.70
tok/s measured across the `-lzm off` and sparse-FA runs above), i.e. **the
one extra CPU-resident expert layer's cost is within this benchmark's own
run-to-run noise** — the `-ub 2048` lever is fully recovered, not partially.
Decode (23.82 tok/s) also matches the sparse-FA-era numbers, confirming
nothing else regressed. Logs: `logs/long-context-120k-ncmoe25/`.

**New recommended production config, superseding the `-ub 1024` fallback
above: `-ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -ncmoe 25 -ub 2048 -c 122880 -lm
none -lzm off`.** `staging/work/bench120k/serve.sh`'s hardcoded default and
the sibling `coding-agent` repo's production `--context-tiers` tier-1 value
(`docker-compose.qwen4exp-moe.override.yml`) were both updated to `25` to
match (see that repo's own history for the production-side change). The
`-ub 1024` fallback is not deleted from this document — it remains a valid,
more-conservative option if a future VRAM regression eats more than one
layer's margin (~962 MiB, from this document's own `-ncmoe 24` vs `26`
table) — but it is no longer the recommended default.

Not yet re-tested: whether `-ncmoe 25` also has enough margin for the 262K
and 512K tiers' own `-ub`/`-ncmoe` pairs (`262144:2048:30`, `512000:1024:32`)
— those were not touched by today's regression report and were not
re-verified here. `docs/research/13`'s own tier-recommendation tables are
left as originally written (per this document's own practice of appending
corrections rather than editing historical measurements); this entry is the
correction to fold back if that doc is revisited.

## Update (2026-09-15, later): `-np 2 -kvu` deployed -- 2 concurrent requests off one dynamically-shared KV pool

Follow-up to the two updates above, prompted by a real user question: two
concurrent connections to `llm-b70` (`-np 1` at the time) queued the second
behind the first rather than erroring -- correct behavior for a single-slot
server, but raised the question of whether 2 concurrent requests could be
served without regressing the single-request context ceiling this whole
`--context-tiers` mechanism exists to protect.

**The naive approach (`-np 2`, `kv_unified` left at its default `false`)
was rejected on VRAM math alone, not tested.** With a statically pre-split
KV pool, every tier's `n_ctx` halves per slot regardless of whether the
other slot is in use (61,440/131,072/256,000 vs the documented
122,880/262,144/512,000 floors) -- a real regression to the already-proven
116K-token capability. The alternative, doubling `-c` to preserve full
depth per slot, was VRAM-infeasible even optimistically (~515 MiB margin
against a card that already OOM'd once on a similar margin) and outright
over budget pessimistically, computed from `docs/research/08`'s VRAM
formula before touching a GPU.

**`-kvu` (`--kv-unified`) changes the trade entirely.** One shared KV pool
sized at the full tier's `n_ctx`, allocated dynamically per sequence
instead of pre-split -- a single deep conversation can still use the whole
tier if the other slot is idle/shallow; two active conversations share the
pool rather than each being hard-capped at half. The formula predicted
this would cost about the same VRAM margin as today's `-np 1`, *if* the
graph allocator's buffer-reuse (already confirmed, doc 08 §3.3, across one
sequence's 12 sequential attention layers) also extends across two
sequences batched together in the same step -- previously unverified in
either direction.

**Verified live before touching production**, via the same isolated
diagnostic-instance pattern used for the earlier phase-timing work (`-np 2
-kvu`, tier-0 config, side port, `-lv 4`): it loaded clean, and the real
allocator numbers came back *better* than the optimistic estimate --
`SYCL0 model buffer size = 25,953.66 MiB` + `SYCL0 compute buffer size =
3,452.38 MiB` = ~29.4 GiB against the 32,402 MiB card, ~3 GiB margin (vs.
`-np 1`'s ~1.7 GiB today). The allocator's reuse does extend across
concurrently-batched sequences -- confirmed, not assumed. Two real
concurrent requests (one ~2,600-token prompt + a trivial one, then a
balanced pair of ~520-token prompts) both completed correctly, `200 OK`,
with per-request decode landing around 8-9.5 tok/s while both slots were
active, against a ~23-25 tok/s solo baseline. **Concurrency costs
per-request decode throughput, roughly to a third, not correctness or
depth** -- the expected, honest trade of sharing one GPU's decode
bandwidth across two active streams, not a bug.

**Deployed**: `coding-agent/docker-compose.qwen4exp-moe.override.yml`'s
`llm-b70` command changed `-np 1` to `-np 2 -kvu`. Production reload
confirmed the same live numbers (`n_slots = 2, n_ctx_slot = 122880,
kv_unified = 'true'` -- note `n_ctx_slot` reports the *full* tier value now,
not a half-split), and two real concurrent chat-completion requests against
the actual production server both returned `200 OK` in ~5.55s each,
started and finishing together.

**What's still open**: only tested at tier 0 (122,880); tiers 1/2's
concurrent-load VRAM margin is computed from the same formula but not
independently re-measured live, for the same reason as the mlock
tier 1/2 gap noted above -- token cost of forcing a real deep-context
request on production. The `--context-tier-down-turns`/idle-deescalation
logic already requires *all* slots idle before a tier switch
(`ctx_tier_update` iterates every slot), so this generalizes structurally,
just not independently measured.

## Update (2026-09-15): decode-time eviction risk found live in production; scoped `--mlock-experts-only` patch added

Follow-up to the 2026-09-14 storage-move update (user question: can startup be
faster without hurting decode). Live measurement on the running,
non-sleeping `llm-b70` (`docker restart`, warm-vs-cold, `mincore()` against
the real GGUF) found the actual reload cost splits into a small disk-bound
piece (~4-46 s, now smaller since the readahead bump) and a much larger
non-disk piece before `threadpool init` even runs (~18-72 s across repeated
restarts, most plausibly SYCL/Level-Zero device init and/or the loader's
per-tensor `ggml_backend_dev_supports_op` probing in `select_weight_buft` --
not root-caused further, flagged as a real open item below).

**The bigger find**: `-lzm off` + mmap only guarantees every tensor is read
*at load time* -- nothing pins those pages afterward. `mincore()` against the
live, fully-loaded, non-sleeping production GGUF showed only 81.3% of the
model resident (66.7 / 82.0 GB), and critically, the CPU-resident MoE expert
tensors placed by `--n-cpu-moe` (the ones `mul_mat_id` reads directly from
the mmap on *every* decode step) were only 79.5% resident in the largest
split. This box also runs vLLM/mssql/shoko/unbooru/Minecraft, and Linux is
free to reclaim clean file-backed pages under their memory pressure with no
warning. That silently reintroduces the exact page-fault-during-decode
pattern `-lzm off` was chosen to prevent (§ the `-lzm off` 2.68x win, this
doc's earlier entries) -- it just does it gradually, off any timer, instead
of via lazy tensors. `-lm mlock`'s earlier "no effect" result (this doc's
first `-lzm off` entry) was the container's 8 MiB `RLIMIT_MEMLOCK` / missing
`CAP_IPC_LOCK` silently no-opping it, not mlock being unhelpful --
re-confirmed live (`/proc/<pid>/limits` showed `Max locked memory: 8388608`
before the fix below).

Naively fixing this with plain `-lm mmap+mlock` would lock the *whole*
mmap-backed CPU set, which turned out to be ~54-61 GB depending on
context tier (PLE table 28.8 GB + `--n-cpu-moe` experts, 25.4/30.5/32.5 GB
at tiers 0/1/2 -- computed from actual GGUF tensor metadata via the vendored
`gguf-py`, not estimated) -- not the naively-assumed 82 GB (whole-file
residency conflates the harmless GPU-uploaded portion, which is dead weight
in host RAM once copied to VRAM and was never going to be read from the
mmap again, with the genuinely at-risk CPU-resident portion). Even 54-61 GB
didn't fit this box's real headroom (swap already in use at the time of
this measurement), so the fix built here is narrower: **`--mlock-experts-only`**
(new flag, `common/arg.cpp` + `src/llama-mmap.{h,cpp}` + `src/llama-model-loader.{h,cpp}`
+ `src/llama-model.cpp` + `include/llama.h` + `common/common.{h,cpp}`) locks
only tensors matching the MoE-expert pattern via a new `llama_mlock::lock_range()`
(a standalone page-aligned range lock, independent of the existing
`grow_to()`'s monotonic-region assumption -- the two must not be mixed on
the same instance), skipping the PLE table entirely. The PLE table's access
is a small per-token gather (~2.7 KB/token, `docs/research/02`), not a bulk
sequential read -- losing a page of it to eviction costs one cheap
microsecond-scale fault, nothing like the multi-GB lazy-tensor case already
measured and rejected.

**Validated live** on `llm-b70` (`-lm mmap+mlock --mlock-experts-only`,
container given `cap_add: IPC_LOCK` + `ulimits.memlock: -1` in the sibling
`coding-agent` repo's compose override): `/proc/<pid>/status` showed
`VmLck: 24819500 kB` (~23.7 GiB) after a tier-0 load against a computed
25.4 GB target (close enough to be page-alignment/quant-padding noise, not
a bug); `mincore()` re-check showed the CPU-experts bucket at 100.0%
resident in both splits (was 79.5%/95.6%) while the PLE table stayed
partially evicted (59.6%) as designed; a real chat completion through the
new binary produced coherent output. Load itself still succeeded and
produced correct output with no mlock-failure warnings, confirming
`CAP_IPC_LOCK` + the raised ulimit actually took effect (`Max locked
memory: unlimited` in `/proc/<pid>/limits`, vs 8 MiB before).

**What's still open** (as of this entry -- see the immediate follow-up
update below for the root cause): the ~18-72 s pre-`threadpool init` cost
is not root-caused (SYCL device init vs. per-tensor `select_weight_buft`
probing are candidates, not confirmed) -- next step is temporary timing
instrumentation around both, then a real fix (likely memoizing
`ggml_backend_dev_supports_op` results per (buffer-type, op-shape) instead
of calling it fresh per tensor, since this model has thousands of
near-identical expert tensors). Also not yet re-verified: mlock survives a
`--context-tiers` switch or sleep/wake correctly at tiers 1/2 (code path is
identical to the tier-0 case validated here -- `ctx_tier_set()` still calls
the same `destroy()` + `load_model()` -- but not exercised live at deeper
tiers due to the token cost of forcing one on production).

## Update (2026-09-15, same day): the pre-threadpool-init cost root-caused -- it's SYCL kernel JIT, not backend probing, and it's already cacheable

Direct follow-up to the "what's still open" item above. Added temporary
phase-timing instrumentation (`ggml_time_ms()` deltas gated behind `-lv 4+`,
`[phase]` log lines) around every step of `llama_model_load` (`src/llama.cpp`),
`load_tensors` (`src/llama-model.cpp`), and `common_init_from_params`
(`common/common.cpp`) -- left in the tree, harmless at the default verbosity.

**The per-tensor `select_weight_buft` hypothesis was wrong**: tensor
creation + buft probing for all ~1224 tensors measured **50 ms**, not
seconds. Ruled out.

**The real breakdown**, measured live via a standalone diagnostic instance
(`llm-b70-diag`, same binary/flags/tier-0 config as production, isolated on
port 8081 so it didn't touch the running server):

| Phase | Time |
|---|---|
| loader construct + hparams/vocab | ~0.2 s |
| create tensors (buft probe) | ~0.05 s |
| init mappings (mmap + `MAP_POPULATE` + mlock the CPU experts) | ~15 s |
| create backend buffers | ~0.5-7 s (noisy) |
| load all data (GPU upload + mlock syscalls) | ~14-15 s |
| `llama_init_from_model` (KV cache + scheduler) | ~0.2-0.6 s |
| threadpool init | ~0 ms |
| **first `llama_decode()` call** | **~39-40 s, every single time, on a fresh container** |

That last row is the dominant cost, bigger than the disk-IO-bound mmap
phase and the tensor-copy phase combined. It is **not** the optional
`--no-warmup`-gated warmup pass -- tested directly: passing `--no-warmup`
removed the "warming up..." log line but the ~40 s gap didn't move, because
`common_context_can_seq_rm()` (`common/common.cpp:1589`, called
unconditionally after `common_init_from_params` returns, to decide the
server's KV-cache-trimming strategy) does its own unconditional
`llama_decode()` of 2 tokens to probe the memory backend's capabilities.
Whichever decode call happens to run first -- the optional warmup's or this
mandatory probe's -- pays the cost; skipping one just moves it to the
other. **This is Intel NEO/Level-Zero JIT-compiling every SPIR-V kernel the
model's first forward pass touches into native ISA**, a well-known
oneAPI/compute-runtime cost that has nothing to do with model size or disk
I/O.

**It's already cached, just not persistently.** Intel's compute runtime
caches compiled kernels on disk by default at `~/.cache/neo_compiler_cache`
(confirmed present in the running container, ~19 MB, 14-15 subdirectories)
-- but that path lives in the container's writable layer, not a bind mount,
so it survives a plain `docker restart` or a `--context-tiers`/sleep-wake
reload (same container, same process) but is wiped by a full recreate
(`docker compose up` after a compose/image change). This lines up exactly
with the earlier confusing variance: measurements taken via
`docker compose run --rm` diagnostic containers (fresh cache every time)
always paid the full ~40 s; measurements taken via `docker restart` on the
persistent production container (cache already warm from an earlier run)
sometimes didn't -- they weren't measuring the same thing.

**Fix applied**: bind-mounted the cache directory in the sibling
`coding-agent` repo's compose override --
`./models/b70-neo-cache:/root/.cache/neo_compiler_cache` -- so it survives
recreates too. Zero code change, zero decode-time effect (it's a compiled-
kernel cache, not model data).

**Validated live, end to end, on real reloads of the actual production
container** (not the diagnostic instance):
- A cold container recreate (cache directory freshly created, one-time
  unavoidable cost): **~60-67 s** to "model loaded".
- A second recreate immediately after, cache now warm on the host bind
  mount: **~20.8 s** to "model loaded" -- confirms the fix.
- A genuine sleep-then-wake cycle on the same (already-warm) container,
  timed as a real end-user request (`--sleep-idle-seconds 10` for the
  test, `curl` timing the full round trip including the reload):
  **21.6 s total**, `http_code=200`, correct output. This is the number
  that answers the original question -- the scenario the user actually
  meant by "startup" (loading the model on first message after idle
  unload) was never paying the ~40 s JIT cost in the first place, because
  sleep/wake reuses the same container/process; only a full redeploy was.

**Net, for the actual sleep/wake-triggered reload the user asked about**:
~21-28 s end to end, down from an unmeasured, uncharacterized "multi-minute"
assumption at the start of this investigation, with every remaining second
now attributed to a specific, understood phase rather than a mystery.

**What's still open**: the "create backend buffers" phase's 0.5-7 s
noisiness isn't explained (device memory allocation variance, not
investigated further); mlock at context tiers 1/2 still unverified live for
the same reason as before (token cost of forcing a real deep-context
request on production).

## Update (2026-09-14): model storage moved off the shared array onto a dedicated NVMe

Digging into why a model load takes multiple minutes (user question, not a
regression) found that `docker stats`' "230GB read for a 77GB model" figure
was cgroup `io.stat` triple-counting across storage-stack layers (the real
logical figure was ~99GB for one load, a modest ~29% overshoot) -- but the
per-device breakdown was real and pointed at genuine contention: ~32% of
that load's reads missed this box's bcache NVMe cache tier and hit the slow
RAID5 backing array directly, plausibly via bcache's default 2ms
`congested_read_threshold_us` tripping under concurrent load from other
tenants sharing that array (`unbooru-tagger`, `deluged`, `sqlservr` --
already-known contenders per this project's earlier measurement-hygiene
lessons). Raising that threshold to 20ms (persisted via a
`bcache-congestion-threshold.service` systemd unit on the host) was tried as
a cheap first fix; a before/after load-time test showed a real 22% drop
(272s -> 213s) but the `nvme`-vs-`md0` read split moved the *wrong* direction
between the two runs, meaning the win is more likely attributable to
page-cache-warmth variance between runs than to the threshold change --
inconclusive, not a confirmed fix.

**Decision: dedicated storage instead of tuning around shared-resource
contention.** A new 4TB NVMe (Lexar NM790) was installed, partitioned (GPT),
formatted (ext4, `-m 0`), and mounted at `/media/da3dsoul/Garudias` (named
per this box's standing `/media/da3dsoul/<name>` convention for extra
drives, matching `Golias`/`Levias` -- not `/mnt/`, corrected after an
initial `/mnt/nvme4tb` mount was relabeled and remounted in place, no
re-copy needed since the filesystem itself didn't change). Production
(`llm-b70` + `router`) was stopped, all 209 GiB under this project's
`staging/models/` (all three models there, not just Qwen3.8-Flash-Next --
the drive is for more than this project) were `rsync`'d over and verified
byte-identical (`du -sb` matched exactly: 223,905,719,398 bytes both sides)
before the originals were deleted and replaced with a symlink
(`staging/models -> /media/da3dsoul/Garudias/models`) for anything that
still looks in the old spot. Configs updated to the real new path directly
(not relying on the symlink): this project's `docker/docker-compose.yml`
(`llm-test-sycl`/`llm-test-vulkan` volume mounts) and the sibling
`coding-agent` repo's `docker-compose.qwen4exp-moe.override.yml` (`llm-b70`'s
`/models` mount). Production brought back up afterward, same config
otherwise (`-ncmoe 25 -ub 2048 -c 122880 -lzm off`, mmap enabled).

- **Never set `GGML_SYCL_USM_SYSTEM=1`.** That's the implicit-migration path
  named for Battlemage that caused the sibling project's two host-wide OOM
  kills. Everything this project needs (`sycl::malloc_host`) is a different,
  explicit mechanism and does not require it. Assert it's unset at every
  startup; check the `GGML_SYCL_USM_SYSTEM: %d` banner line in every log.
- **Explicit, bounded, developer-controlled host allocation only.** Chunk the
  cold-expert pinned tier into ≤2 GiB `sycl::malloc_host` allocations (a
  Level Zero requirement for copy/compute overlap, not a style choice — see
  `docs/research/01` §2.4) and size the total against measured host RAM
  before running anything, not after.
- **Descope graph capture/replay from the first working version.** It's the
  riskiest, least-precedented component, it's off by default upstream with
  the maintainers' own "no better performance" note, and the legacy
  (non-graph) cache path doesn't need it at all. Land and measure the cache
  without it first; treat graph replay as a Phase 3 add-on whose value is
  measured in isolation before investing further.

## Update (2026-09-19): MTP + K-quant pathology root-caused -- IQP fast path is I-quant-only, never fires for K-quants

Follow-up to "**MTP + UD-Q3_K_XL: don't use this combination as-is**" above
(2026-09-11 update), which flagged Q3_K_XL+MTP at 3.2 tok/s vs. 21.9-23.4
tok/s for the target alone (~7x) and named three untested hypotheses.
Root-caused this session by code reading only, against the `src/llama.cpp`
checkout -- no live run yet, see caveat below.

**Mechanism.** `ggml/src/ggml-cpu/iqp.cpp` implements a batched, panel-gemm
fast path for `mul_mat_id` (the MoE expert-routing matmul), gated behind a
minimum-rows-per-expert threshold: `GGML_IQP_MIN_BATCH_ID = 8`, checked via
`ggml_cpu_iqp_mul_mat_id_min_batch(cne1)` at
`ggml/src/ggml-cpu/ggml-cpu.c:1855`. The type list this fast path supports
(`IQP_TYPE_LIST` macro, `iqp.cpp` ~line 494) is hardcoded to I-quants only --
`IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_S, IQ1_M, IQ4_XS` -- the Q3_K
family is not in this list.

In plain single-token decode, `cne1` (rows landing on one expert within a
single `mul_mat_id` node) is always 1, far under the threshold -- this fast
path is already dormant for ordinary decode regardless of quant type. It
only has a chance to fire when MTP's multi-token verify batch routes several
tokens to the same expert within one op. Net effect: IQ3_XXS+MTP gets a real
batched-gemm speedup on CPU-offloaded expert layers during verify batches
(plausibly explaining why that combo nets a positive ~1.15x above, rather
than a loss); Q3_K_XL+MTP gets none of it and pays the full scalar per-row
generic path for every CPU-offloaded expert row in the verify batch. This
reads as a structural asymmetry between quant families under one specific
fast path, not a tuning problem.

**The other two named hypotheses**:
- KV-cache type mismatch handling between draft/target -- checked directly
  against `common/speculative.cpp:1324-1438`; no quant-conditional branching
  found there. **Ruled out.**
- Near-VRAM-ceiling silent thrashing distinct from a clean OOM -- this
  project's two known mechanisms for that (`GGML_SYCL_USM_SYSTEM` implicit
  migration, and direct `zeMemAllocDevice` allocation bypassing the usual
  path) are both already disabled/mitigated per `docker/Dockerfile.sycl*`.
  **Largely ruled out**, though an undocumented third variant can't be fully
  excluded without a live GPU measurement -- **not fully ruled out**.

**Caveat.** This is a code-reading finding, not a live measurement. It
explains a *missing* speedup for the CPU-offloaded portion of the verify
batch under MTP, directionally consistent with the ~7x cliff measured in the
2026-09-11 update, but reading the code alone doesn't certify the full
magnitude of that cliff. A CPU-only benchmark to confirm the magnitude is
in progress as a separate piece of work this session -- no number yet, none
should be quoted until it lands. Also in progress in the same session: an
attempt to extend `IQP_TYPE_LIST` to actually support Q3_K, so that
Q3_K_XL+MTP stops paying this penalty -- outcome not yet known.

## Update (2026-09-19, later): the IQP-panel-path theory above is FALSIFIED for
## this model -- UD-Q3_K_XL contains zero Q3_K tensors. Q3_K support was still
## added and validated, but it fixes nothing here. Cliff cause reopened.

Direct follow-up to the entry immediately above, same day. The CPU-only
benchmark and the `IQP_TYPE_LIST` extension both landed, and the benchmark's
first step -- checking what quant types the real file actually uses, rather
than assuming from its name -- overturned the theory before the speedup
measurement even mattered.

**The premise was wrong.** Inspected `/media/da3dsoul/Garudias/models/
Qwen3.8-Flash-Next-GGUF/UD-Q3_K_XL/*.gguf` directly via the vendored
`gguf-py` (independently re-verified by the orchestrating session, not just
the subagent's claim). Full tensor-type histogram across all 3 shards:
`Q6_K:1, Q8_0:502, F32:557, IQ4_NL:44, IQ3_XXS:94, IQ4_XS:2, BF16:24`.
**Not one `Q3_K` block anywhere in the file.** "UD-Q3_K_XL" is Unsloth's
recipe-tier name, not a literal description of every tensor's type. The
actual routed-expert tensors: `ffn_gate_exps`/`ffn_up_exps` are `IQ3_XXS`
(47 of 48 layers) or `IQ4_XS` (1 layer) -- both already in `IQP_TYPE_LIST`,
already fast-path-eligible, before today's change. `ffn_down_exps` is
`IQ4_NL` (43 layers) or `Q8_0` (5 layers) -- and cross-checking the *working*
`UD-IQ3_XXS` quant found its own `down_exps` is **also `IQ4_NL`**, identical
between the working and broken configs. There is no K-quant/I-quant split
between these two quants' actual tensors to explain a differential at all.

**`down_exps` is structurally excluded from the fast path regardless of type,
in both quants equally.** `ggml/src/ggml-cpu/iqp.cpp:1135` hard-requires
`src0->ne[0] % QK_K (256) == 0` for fast-path eligibility. `down_exps` has
`ne[0] = 640` (`moe_intermediate_size`), the same "K-quant gotcha" shape
`docs/research/02` §5.5 already flagged for a different reason (K-quants
themselves being ineligible at this width) -- but the constraint is on the
panel's fixed 256-wide superblock layout, not on K-quant-ness specifically,
and it applies identically whether the model ships with `Q3_K`, `IQ4_NL`, or
anything else at that tensor. Since it's equally excluded in the working
IQ3_XXS+MTP config, it cannot be the source of the 7x asymmetry.

**Q3_K support was implemented and correctness-validated anyway** (the user
asked for this regardless of whether it would fix the cliff). `iqp_decode_q3_K()`
added to `ggml/src/ggml-cpu/iqp.cpp` (57 lines, one file), reusing
`block_iqp_x8` exactly like the existing I-quant grids since Q3_K's
16-groups-of-16 signed-6-bit-scale layout maps onto it directly (unlike
Q4_K/Q5_K, which carry an extra per-group `min` term the current symmetric
panel design has no slot for). Validated two ways: (1) a temporary
`GGML_IQP_VERIFY` build asserting the new decode reproduces
`ggml_get_type_traits(Q3_K)->to_float()` bit-for-bit inside the same call --
caught a real bug on the first attempt (hmask byte offset incorrectly
advancing with the 128-wide half, when only `qs` should; `hmask`'s 32 bytes
are reused across both halves, distinguished only by which bit is tested),
fixed, then 0/272 mismatches; (2) `test-backend-ops`'s existing dual-CPU-backend
in-process check (`ggml_backend_cpu_set_use_ref(true)` forces the generic
scalar path, compared against the panel path within one `eval()` call, same
mechanism this project's own `LLAMA_QSA_PLAN_CHECK` oracle follows for
exactly the reason that cross-run diffs aren't valid here) -- **1314/1314
`MUL_MAT` and 888/888 `MUL_MAT_ID` tests pass**, including every panel-eligible
type plus the new `Q3_K`, at the suite's existing nmse < 5e-4 tolerance
(Q3_K passed comfortably, not near the edge). Built via
`staging/work/devbuild.sh ggml/src/ggml-cpu/iqp.cpp`.

**Measured speedup: ~1.00x, i.e. none**, for Q3_K specifically and, more
surprisingly, for the *existing* IQ3_XXS panel path too. A standalone
harness (single `MUL_MAT_ID` expert, k=2560/m=640 matching real
`ffn_gate_exps`) swept `cne1` = 1/3/5/8/16/32/64 at 1 and 8 threads, A/B'd via
the existing `GGML_NO_IQ_PANEL` env escape hatch. Panel-vs-generic ratio was
~1.00-1.02x across the board on this box's CPU (AMD Ryzen 9 9950X, Zen4/5,
AVX-512 VNNI) -- statistical noise, not a speedup, for a mechanism this
project's own earlier reading assumed was a real win. Likely explanation,
offered as hypothesis: ggml's existing scalar `vec_dot_*_q8_K` kernels are
already VNNI-accelerated on this CPU, leaving little redundant-decode
overhead for panel batching to amortize -- the win this path was designed
for may be real on weaker-SIMD hardware but doesn't show up here, on *any*
panel-eligible type, not just the new one.

**Net: the panel-path theory is dead for this deployment, on two independent
grounds** -- the tensors it would need to apply to don't exist in this file,
and even where the mechanism is eligible it measures no benefit on this CPU
at all. Cause of the 3.2 vs ~22 tok/s MTP+Q3_K_XL cliff is **reopened**; the
two hypotheses from the entry above (KV-cache type mismatch, near-VRAM-ceiling
thrashing) are the remaining live suspects, plus a fourth not previously
separated out: a gap in the MTP code path itself that only IQ3_XXS has ever
exercised. Settling any of these needs a real GPU-loaded MTP+Q3_K_XL decode
run at `-ncmoe 28` on the actual B70 -- not attempted, deliberately, since the
card is currently fully occupied by live production traffic and stopping it
is a decision for the user, not something to do inside an investigation
task.

The `Q3_K` panel-path addition itself is kept in the tree regardless -- it is
correct, tested, and a generically useful capability for any future
model/quant that actually ships real `Q3_K` experts at a `ne[0]` divisible by
256, independent of whether it helps this project's own model.

---

## Phase 0 — Go/no-go gates before writing any cache code

Everything here is either already answered by the research docs or is a
cheap, targeted check. **Do not start Phase 1 until every item below has an
answer.** This phase can kill the project outright, and that's the point of
doing it first.

### 0.1 Reproduce (or rule out) issue #24168 — the single highest-priority check

`docs/research/01` §6.3: an open, still-unresolved llama.cpp issue reports
gibberish/crash output specifically on **hybrid SSM+MoE architectures
including `qwen3next`** on **Arc Pro B60** (Battlemage sibling to our B70).
`qwen3next` is the architectural predecessor Qwen3.8-Flash-Next's Gated-
DeltaNet-plus-MoE design is closest to. Pure-MoE models (no SSM component)
are reported unaffected — so this is specifically about the hybrid
combination, which is exactly what we're building on.

**Action:** build vanilla upstream llama.cpp SYCL, no cache patch, and run
the smallest available `qwen3next`-family GGUF (or a small Qwen3.8-Flash-Next
quant if one exists small enough to sanity-check quickly) on the actual B70.
Compare output against the CPU backend on an identical prompt/seed.

**Go/no-go:** if this reproduces and the upstream issue thread offers no
workaround, **stop here.** A caching layer cannot fix a correctness bug one
level below it. If it doesn't reproduce, or reproduces only under conditions
we can avoid (specific op fusion flags, a specific quant, etc.), proceed and
document the exact avoidance condition — Phase 1 must not silently drift back
into it.

### 0.0 Pin the build — non-negotiable, do this before anything else in Phase 0

`docs/research/02` §3.0: **both** SYCL and Vulkan had a bug that breaks this
exact model on this exact card class, and both fixes landed the same day as
the research (2026-09-09). Vulkan hard-aborted on Arc Pro B65/B70
(`ggml_vk_fill` workgroup-count overflow past `n_ubatch × kv_len ≈ 33M`,
fixed by #28592). SYCL silently fell back to host-serialized execution for
every IQ-quantized MoE weight — not a crash, a silent 40-60% throughput
loss (fixed by #28476, measured tg512 11.17 → 17.92 t/s on 3× Arc Pro B60).
**Any build older than 2026-09-09 will look broken or badly slow for
reasons that have nothing to do with this project.** Pin to a commit at or
after that date before running anything else in this phase, and note the
exact commit hash in every subsequent benchmark record — this is the kind
of fact that's easy to lose track of and re-debug from scratch later.

### 0.2 SYCL vs. Vulkan backend decision (`GGML_OP_TOP_K`)

`docs/research/02` §2.3, §6: Qwen Sparse Attention needs `ggml_top_k(width≈2051)`
on all 12 full-attention layers. SYCL hard-caps `k≤32` (a shared-local-memory
limit, not a tunable — `docs/research/02` quotes the kernel's own comment
explaining why); Vulkan already has radix-select and handles arbitrary `k`.
On SYCL this forces a CPU round-trip per full-attention layer per ubatch —
cheap at decode (~1.5 MiB/token), *crippling* at prefill (~3 GiB/ubatch at
`n_ubatch=2048`, per `docs/research/02` §2.3's estimate).

Three options, cost order:
1. **Disable QSA entirely.** Free (a GGUF without indexer tensors runs dense
   attention). QSA currently provides no speed benefit on *any* backend today
   anyway (`docs/research/02` §2.1 — the sparse compute path is TODO'd out
   upstream; only the mask is sparse). Costs quality only, not throughput.
   **Confirm first whether a CLI flag actually exposes this** —
   `docs/research/02` §6 flags this as unverified; check `common/arg.cpp`.
   **Update 2026-09-11, see `docs/research/07-qsa-sparse-attention-scoping.md`
   for the precise version of this claim**: the sparse compute path (the
   `op_params[4]` hint `qwen4exp.cpp:946` currently zeroes) is real and
   CUDA/Metal do implement it, but SYCL and CPU never read that slot — so on
   *our* backend specifically it's correctly a true no-op either way, safe to
   leave disabled. Separately, the crossover depth where sparsity would even
   start mattering is `n_kv ≥ 2049` (`indexer_top_k=2048`,
   `compress_ratio=4` on all 12 full-attention layers, verified against the
   actual GGUF) — every benchmark this project has run so far used `n_kv ≤
   512`, so this was never going to show up regardless of backend support.
   And even if SYCL gained support, `ggml_top_k(2051)`'s own cost (option 3
   below) becomes the binding constraint at the same depths this would start
   to matter — fix that first if long context ever becomes a goal, not this.
2. **Target Vulkan instead of SYCL for the whole project.** Sidesteps the
   `top_k` problem outright, at the cost of every §1–§6 SYCL-specific finding
   in `docs/research/01` needing a Vulkan re-derivation (different
   abstraction: Vulkan compute shaders, not SYCL kernels — a different, not
   necessarily smaller, porting effort, and `docs/research/01` was not
   written against Vulkan). Also note `docs/research/01`'s issue #26581: full
   decode attention is memory-latency-bound and *identical* on Vulkan and
   SYCL on this exact card — so Vulkan is not a free performance win outside
   the `top_k` question.
3. **Port a radix-select `top_k` to SYCL**, using Vulkan's
   `topk_radix_select.comp` as a reference shader. Self-contained, the
   highest-value single upstream contribution this project could make, but
   it's new kernel work with no existing precedent to copy from (unlike the
   MoE router port in §4.3 of `docs/research/01`).

**Action:** measure the actual `GGML_OP_TOP_K` CPU-fallback cost on SYCL with
`llama-bench` (reads the `layer %d is assigned to device ... (usually due to
missing support)` diagnostic already built into llama.cpp —
`docs/research/01`'s auto-fallback note, §2.5 equivalent) before choosing.
If prefill is genuinely crippled and QSA can't be cheaply disabled, this
decides SYCL vs. Vulkan **before** any cache-porting work starts, since it
determines which backend that work targets.

**This decision now has real data behind it, not just architectural
reasoning** (`docs/research/02` §3.1–§3.2, both post-2026-09-09 fixes):
Vulkan on an Arc Pro B65 32 GB (near-identical card) gets **30.4 t/s decode
@ 8k depth** on UD-Q3_K_XL with static `--n-cpu-moe 25`, falling to 13.3 t/s
@ 32k; SYCL on 3× Arc Pro B60 gets **17.92 t/s tg512** on UD-IQ3_XXS (our
video-matched quant). These aren't directly comparable (different quants,
different core counts, different context depths), but they're both real,
they're both post-fix, and Vulkan avoids the `top_k` problem structurally
rather than working around it. `docs/research/02`'s own recommended
posture: **prototype on Vulkan first, benchmark SYCL against it once
working, and pick the port target from measurement rather than
architecture alone.** This plan adopts that recommendation as the default
unless Phase 0/1's own measurements say otherwise — but note
`docs/research/01` (the SYCL-specific feasibility doc) was written
assuming a SYCL target, so choosing Vulkan means re-deriving that
document's §1–§6 against Vulkan's compute-shader model before Phase 2, not
just swapping a build flag.

**Also note for VRAM budgeting (0.3/0.5 below), from the same B65 report**:
the QSA top-k scratch buffer (`n_kv × n_ubatch × 4` bytes on Vulkan,
`ggml_vk_topk_radix_qsa`) scales with **allocated** `-c`, not context
actually in use — 512 MiB at `-c 32768, -ub 4096` — and overshooting VRAM
degrades **silently** (driver-level PCIe spill) rather than erroring. The
B65 reporter's fix was moving more expert layers to CPU
(`-c 49152 --n-cpu-moe 29`). Size the slot pool against this scratch cost
explicitly, not just against weights.

**Go/no-go:** pick one of the three options and write down why, backed by
the real numbers above rather than architecture alone. Don't let this stay
open into Phase 1.

### 0.3 Quant choice and expert-residency sizing

`docs/research/02` §5 already answers this precisely — **no quant fits fully
in 32 GB VRAM** (smallest published quant's expert set alone is 39.85 GB),
but the *dense* core (everything except experts and the PLE table) is only
3.9–5.5 GB, leaving a genuinely large slot pool against a 40–77 GB expert
working set. This is the favorable regime the cache is built for — the
opposite of the sibling project's dense-model case.

**Action:** pick a starting quant. Doc 02 §5.6 recommends **UD-Q2_K_XL or
UD-IQ3_XXS** (~49–53% expert residency at 32 k context) as the starting
point, with UD-IQ1_S (~61% residency) worth a later A/B once correctness is
established. Avoid UD-Q4_K_XL and above initially (29.6% residency, and
penalized by the `moe_intermediate_size=640` K-quant-incompatibility gotcha
in doc 02 §5.5).

### 0.4 Host RAM budget — verify before assuming

`docs/research/02` §5.1 confirms the 28.8 GB PLE table is, **by default**,
kept in host memory / mmap'd from the GGUF, not force-loaded. If that stays
true under whatever load flags the cache setup ultimately requires, the PLE
table costs page-cache pressure, not a hard RAM allocation — its actual
access footprint is tiny (~2.7 KB/token, per the video-findings research in
the sibling project). **This is not yet verified for our exact flag
combination** — the fork requires `--load-mode none` for the cache's cold
expert tier, and it's unconfirmed whether that flag is global (forcing the
PLE table to fully load too) or per-tensor-class.

**Action:** confirm `--load-mode none`'s scope before sizing host RAM. If it
only affects `ffn_*_exps`, host RAM budget ≈ cold-expert tier alone (~20–26
GB at the Phase 0.3 quant choice) — comfortable on most homelab boxes. If it
forces the PLE table to fully load too, budget ≈ 28.8 GB + cold-expert tier
(~50–55 GB) — confirm the actual box has that much free before proceeding,
and treat "PLE forced into RAM" as a design smell worth fixing upstream
rather than living with.

### 0.5 Cheap hardware probes to run once, early

- Does the B70 report `aspect::ext_oneapi_graph` (updatable) or only
  `aspect::ext_oneapi_limited_graph`? One-line runtime probe
  (`docs/research/01` §8 Q1). Doesn't block Phase 0–2, but decides which of
  the two graph-replay designs (§5.6 constant-arg slot table vs. VMM remap)
  is even available when Phase 3's stretch goal is attempted.
- Confirm `SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC` is present in our oneAPI
  toolchain version (`docs/research/01` §8 Q4) — without it, SYCL graphs are
  refused even for plain matmul, independent of the MoE-specific fix.

### Phase 0 exit criteria

- [ ] 0.0: build pinned to a commit ≥ 2026-09-09 (includes both #28592
      Vulkan fix and #28476 SYCL fix), commit hash recorded.
- [ ] 0.1: #24168 does not reproduce, or reproduces only under an avoidable
      condition (documented).
- [ ] 0.2: backend decided (SYCL, or Vulkan, or SYCL+QSA-disabled) with a
      measured `top_k`-fallback cost backing the decision.
- [ ] 0.3: starting quant chosen.
- [ ] 0.4: host RAM budget computed against the actual box's free RAM, with
      `--load-mode` scope confirmed.
- [ ] 0.5: graph-aspect and async-alloc probes run and recorded (informs
      Phase 3, not blocking).

**If 0.1 fails outright: stop. This is the one gate that can end the project
before Phase 1 regardless of everything else.**

---

## Phase 1 — Base inference, no cache, establish the baseline to beat

Goal: get Qwen3.8-Flash-Next producing correct output on the chosen backend
(SYCL or Vulkan per 0.2), using upstream's existing **static** expert
placement (`--n-cpu-moe` / `-ot ...=CPU`) — no cache code written yet. This
is the baseline the eventual cache must beat; per `docs/00-background.md` and
`docs/research/03` (sibling project), that's the correct baseline, not "no
offload at all" (which doesn't fit regardless).

**Tasks:**
- Build llama.cpp against the Phase 0.2 backend decision, at the Phase 0.3
  quant, with `--n-cpu-moe` set to keep experts on CPU and the dense
  core + PLE table placed per upstream defaults.
- Correctness: fixed-prompt, fixed-seed output compared against the CPU
  backend (or a CUDA reference run if one is reachable) — not bit-exact
  (different backends won't be), but coherent and free of the #24168 garbling
  pattern. Re-run the 0.1 test model here too as a regression check now that
  more of the graph is active.
- Benchmark: prefill and decode tok/s at a couple of context lengths, using
  `llama-bench`. This number is the one everything in Phase 3 is measured
  against.
- Resolve the QSA CLI-flag question from 0.2 concretely if not already done.
- If targeting SYCL: capture the `layer %d is assigned to device ...`
  fallback log and get a real number for the `top_k` CPU round-trip cost
  (turns 0.2's estimate into a measurement).
- **Regression smoke test, SYCL specifically**: issue #25455
  (`docs/research/02` §3.3) reported `MUL_MAT_ID` producing wrong prefill
  output on **Arc Pro B70** — literally our card, and literally the op every
  expert routing decision goes through, both with and without the cache.
  Closed 2026-08-30, but "closed" isn't the same as "verified on our exact
  build" — run `test-backend-ops -b SYCL0 -o MUL_MAT_ID` as part of Phase 1
  bring-up and treat any failure as a hard blocker, not a known issue to
  route around.

**Go/no-go:** is baseline correctness clean, and is the static-CPU-offload
baseline already fast enough that the cache's added complexity isn't
justified? (Unlikely given the video's own reported gap between static
placement and the cache, but check — the cache is not free to build, and
this project should stop here rather than proceed on faith if the baseline
already looks acceptable.)

---

## Phase 2 — Cached buffer type: pinned host tier + VRAM slot pool

Goal: a working `ggml_backend_buffer_type` that holds expert tensors with
cold copies in chunked `sycl::malloc_host` pinned memory and hot copies in a
fixed-size VRAM slot pool — **no dispatch/admission logic yet.** Validate
with direct tensor read/write tests, not full model inference.

Grounded in `docs/research/01` §1–§2, portable/needs-new-code items 1a–2d:

- New buffer type implementing the standard vtable (§1.1 — `ggml-sycl`
  already has three buffer types to copy the shape from: device, split,
  host). Use `tensor->extra` (`ggml_tensor_extra_gpu`, already present in
  `common.hpp`) for per-tensor cache metadata, following the split-buffer-type
  precedent.
- Widen the **three** `get_name`-function-pointer identity checks plus the
  one SYCL-only debug assert (§1.2) so the scheduler and `cpy_tensor`
  recognize the new buffer type. This is the same three-site problem the
  CUDA fork already solved — read what it did there first (see Phase-2 note
  below) rather than re-deriving it.
- Pinned host tier: `sycl::malloc_host`, chunked into ≤2 GiB allocations
  (§2.4), with a slab→(allocation, offset) map. No `cudaHostRegister`
  equivalent exists (§2.3) — allocate-and-fill, never try to pin an mmap in
  place; this also means the fork's mmap-only L2 tier
  (`--moe-expert-cache-l2-pinned-mb`) has no SYCL analogue and is out of
  scope.
- VRAM slot pool: fixed-size, uniform stride = largest expert slab (mirrors
  the CUDA fork's design per `docs/00-background.md` §1), backed by ordinary
  device buffer-type allocation to start (§7 in the feasibility doc's table
  notes `sycl_ext_oneapi_virtual_mem` / `ggml_sycl_pool_vmm` as an available
  but optional refinement — not required for a first working version).
- **Before writing this phase's code, actually read the fork's
  `ggml_cuda_mul_mat_id` diff and the buffer-type portion of
  `moe-cache.cu`/`.cuh`.** `docs/research/01` was explicitly written without
  reading that source (its author's own open item §8 Q13) — it reasoned from
  `docs/00-background.md`'s secondhand description. Closing that gap first
  will save real time here.
- **Also read issue #25812 first** (`docs/research/02` §3.3): a still-open
  `UR_RESULT_ERROR_OUT_OF_HOST_MEMORY` crash specifically when offloading
  MoE experts to Arc GPUs, which the reporter worked around by keeping all
  experts on CPU — i.e. by never exercising the exact host↔device staging
  path this phase is about to build. Whatever oversized allocation pattern
  triggers that bug is directly adjacent to this phase's pinned-host tier
  design; understand it before writing the allocator, not after hitting it.

**Validation:** write/read known expert slabs through the new buffer type
end-to-end (host↔device), confirm eviction/re-admission round-trips
byte-correctly, before touching the dispatch path.

---

## Phase 3 — Dispatch, admission, and (as a stretch goal) graph replay

Goal: correct, then fast, end-to-end decode through the cache. Two
sub-phases, deliberately sequential — legacy path first, so there's always a
working, measurable fallback before attempting the higher-effort fast path.

### 3a. Legacy path (host round-trip, plain LRU)

- Hook `ggml_sycl_mul_mat_id` (`docs/research/01` §1.3 — the confirmed single
  chokepoint, structurally a near-line-for-line analogue of the CUDA fork's
  interception site) to recognize the cache buffer type, read router IDs back
  to host, dedup unique experts per op, and do plain LRU admission/eviction
  against the Phase 2 buffer type.
- Dedicated copy queue: `docs/research/01` §3 — this needs genuinely new code
  (no second queue exists anywhere in `ggml-sycl` today), constructed from
  the **same `sycl::context`** as the compute queue (§3.3's documented
  cross-context silent-no-op trap — this is the single easiest mistake to
  make in this phase and the hardest to notice, since it fails silently
  rather than erroring). `sycl::event` + `ext_oneapi_submit_barrier({event})`
  for readiness signalling (§3.1, both already used in-tree elsewhere —
  follow the reorder-path precedent at `ggml-sycl.cpp:4121-4157`).
- **Correctness gate before moving on:** full-model decode through the cache
  vs. the Phase 1 static-placement baseline, same prompt/seed, checked for
  output match (not just "doesn't crash"). Also re-run the #24168 regression
  check one more time — the cache changes memory-residency patterns for
  exactly the tensor class that bug concerns.
- **Benchmark gate:** does the legacy path already beat Phase 1's baseline?
  If not, something is wrong before adding more complexity — don't proceed
  to the fast path on the assumption it'll fix a regression the legacy path
  itself has.

### 3b. Device-side "grouped decode" fast path

- Use `ggml_sycl_mul_mat_id_mmvq_fused()` / `mul_mat_vec_q_moe`
  (`docs/research/01` §1.3 — confirmed to already keep router IDs on-device,
  a better insertion point than CUDA's own). The redirect is described as
  literally one line: `vx_base + i02*expert_weight_stride` becomes a lookup
  into a device-resident slot table.
- Frequency-decay admission planner as an ordinary SYCL kernel using
  `dpct::permute_sub_group_by_xor` sub-group reductions (`docs/research/01`
  §4 — lowest-risk component, and `topk-moe.cpp` is a documented, in-tree
  CUDA→SYCL port of the MoE *router* to copy the method from). **Audit every
  reduction for the 32→16 lane-width change** (`WARP_SIZE=16` on all Intel
  targets, not 32 — the single most likely silent-bug source in this
  sub-phase per §4.2).
- Watch the reorder-timing interaction flagged in `docs/research/01` §8 Q12:
  the lazy Q4_K per-expert SoA reorder must happen once on the cold host
  copy, not on every admission — get this wrong and every cache hit pays a
  reorder cost that should have been amortized at load time.

### 3c. Graph capture/replay — explicit stretch goal, measured before investing further

Per `docs/research/01` §5's own recommended sequencing:

1. Rebase PR #25089 (already validated on an Arc Pro B70 with a Qwen MoE
   model — `docs/research/01` §5.4) locally onto current master.
2. Set `GGML_SYCL_ENABLE_GRAPH=1` and **measure whether graph capture alone
   changes decode throughput** on 3a/3b's already-working cache. This one
   number decides whether the rest of this sub-phase is worth doing at all —
   the upstream maintainers' own default assessment is "no better
   performance," and that should be trusted until our own measurement says
   otherwise.
3. Only if step 2 shows a real win: implement expert re-pointing between
   replays using the **constant-argument slot-table design**
   (`docs/research/01` §5.6, option four — a fixed device-side `slot_of[]`
   table the admission kernel writes into, so the graph's kernel arguments
   never change and only `aspect::ext_oneapi_limited_graph` is required, not
   the stronger updatable-graph capability). This is the design the research
   explicitly recommends over per-node `dynamic_parameter` updates (not
   exposed by `ggml-sycl` today, §5.5) or VMM remapping under a fixed virtual
   address (unverified `unmap`/`map` legality and latency while a graph is
   live, §5.6).

If 0.1's #24168 avoidance condition or any correctness gate from 3a breaks
under graph capture specifically, that's a signal to abandon 3c rather than
debug around it — this sub-phase is explicitly the lowest-priority, most
speculative part of the whole project.

---

## Phase 4 — Correctness and benchmark gates before calling this done

Adopting the sibling project's gate discipline (`docs/00-background.md`
references `../coding-agent/docs/`'s "Gate 1/2/3" correctness-battery
convention) — this phase is where that discipline pays for itself, since a
memory-residency cache is exactly the kind of feature that can look fast and
be silently wrong.

- **Correctness battery:** fixed prompt/seed decode at several context
  lengths, cache cold vs. warm, deliberately forced eviction storms (fill the
  slot pool past capacity, confirm output is still correct, not just
  non-crashing). Compare against the Phase 1 static-placement baseline as the
  correctness oracle, not against "does it look plausible."
- **Host-RAM-pressure gate, run deliberately conservatively the first time:**
  start well under the Phase 0.4 computed budget, monitor system memory in
  real time, and ramp up only after confirming there's no repeat of the
  sibling project's implicit-migration OOM pattern. Confirm
  `GGML_SYCL_USM_SYSTEM` stays `0` and the ≤2 GiB pinned-chunk rule is
  actually being honored by the allocator, not just assumed.
- **Speculative-decoding interaction, explicitly re-tested, not assumed:**
  `docs/00-background.md` §1 records that the CUDA fork's own report found
  the expert cache **fights** GPU-resident MTP-style draft heads (a verify
  batch's union-of-experts read pattern can bypass/thrash the cache), with
  the one reported working combination elsewhere using a CPU-side drafter.
  Unsloth's `qwen4exp/mtp` branch (unverified contents per
  `docs/research/02` §6 item 7) is the natural thing to test this against
  when it's reached — but treat cache+MTP as a combination to *measure*, not
  assume works, and be prepared for the honest answer to be "pick one, not
  both," same as the sibling project's dense-model analysis flagged as the
  most load-bearing risk for that (inapplicable) case. If targeting SYCL:
  note `docs/research/02` §3.5 — PR #23174 fixed "MTP on SYCL gives garbled
  output after a few tokens" as recently as 2026-05-22, i.e. SYCL+MTP has a
  known history of silent correctness bugs on this exact backend
  independent of the cache. Re-run the Phase 4 correctness battery, not just
  a speed comparison, whenever MTP is added to the mix.
- **Track, don't block on**: Vulkan-specific PR #28501 (`docs/research/02`
  §3.3, open) fixes a `count_experts.comp` shared-array sizing bug that
  disables an optimization for any model with >256 experts — Qwen3.8-Flash-
  Next's 512 hits this today. Reports +16-19% free prefill once merged. Not
  worth blocking Phase 4 on, but worth re-benchmarking against once it
  lands if Vulkan ends up the chosen backend.
- **Prefill regression check:** the CUDA fork's own numbers show 14-66%
  prefill regression under the cache (`docs/00-background.md` §1). Confirm
  whether the SYCL port shows the same pattern, and whether it's acceptable
  given this deployment's actual workload (long-context prefill matters a
  lot if this is meant to serve real conversations, not just decode
  benchmarks).
- **Final benchmark comparison:** Phase 1 baseline vs. Phase 3a (legacy
  cache) vs. Phase 3b (fast path) vs. Phase 3c (+graph, if pursued), at the
  Phase 0.3 quant and at least one higher-residency quant (e.g. UD-IQ1_S) as
  a sensitivity check on the doc 02 §5.4 residency-vs-quant table. Report
  every number against the **47 tok/s same-model reference point** from the
  intro's "Performance target" section (RTX 5070 Ti, UD-Q3_K_XL, cache-80,
  RAM-pressured), not just against the Phase 1 no-cache floor — that's the
  comparison that actually answers "did this project succeed," and a result
  meaningfully below it should be root-caused (host RAM headroom, hit rate,
  backend overhead) rather than attributed to "slower card" without
  checking.

**Only after this phase clears** does it make sense to write up results and
decide whether to invest in upstreaming any of it (the SYCL `top_k` port
from 0.2, or a rebase-and-resubmit of #25089, are both plausible standalone
contributions independent of whether the full cache project succeeds).
