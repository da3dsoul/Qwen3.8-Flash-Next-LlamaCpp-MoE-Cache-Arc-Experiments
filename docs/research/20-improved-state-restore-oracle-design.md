# 20 - An improved state-restore correctness oracle: exact replay, a corruption ladder, same-launch twins

Date: 2026-09-27. Status: **design plus a CPU-only prototype. No GPU was used, no GPU time was
requested, and the production `llm-b70` server was not touched.** The prototype extends
`staging/work/state_restore_equiv.cpp` and was run only against the synthetic random-weight `qwen4exp`
model on the CPU-only `kvstate-arm0-build` tree (`GGML_SYCL=OFF`). **That checks the harness
mechanics. It says nothing about detection power on the B70's real SYCL noise, which remains unverified
until the GPU pass in §8.**

Triggered by the O3 result in Arm 2 of doc 18 (`logs/pcache/arm2/01-harness-8k.log`,
`02-harness-120k.log`): O3 failed its literal pass criterion because the `shift` mutation control came
out *less* divergent than the `fresh` noise floor at both 8K and 116K.

Inputs:

- `18-conversation-state-persistence-plan.md` §4.2 (O1-O5), §5 Arm 2.
- The Arm 2 logs above, and `logs/pcache/arm0/04-state_restore_equiv.log`.
- `PLAN.md:1004-1014` (greedy output is not reproducible) and `PLAN.md:1129-1147` (the forward pass is
  not reproducible across runs; O4's in-graph design).
- Direct reads of the save/restore code in `src/llama.cpp/src/`: `llama-kv-cache.cpp`,
  `llama-memory-recurrent.cpp`, `llama-memory-hybrid-idx.cpp`, `llama-memory-hybrid.cpp`,
  `llama-context.cpp`, `llama-batch.cpp`, `models/qwen4exp.cpp`.

Conventions follow docs 16-19. `file:LINE` means the line was read directly. **Inference** marks a
conclusion drawn by combining separately read facts. **Estimate** marks arithmetic. Code paths are
relative to `src/llama.cpp/` unless they start with `docs/`, `staging/`, `logs/` or `PLAN.md`. Harness
line numbers refer to `staging/work/state_restore_equiv.cpp` as it stands after this doc's additions.

---

## 0. Summary

**Tonight's O3 had detection power. Its pass criterion compared against the wrong null.**

- `rest` and `orig` come from the same prefill. `rest` restores the exact bytes `orig` decoded from.
- `fresh` contains a second, independent prefill, and this backend's run-to-run noise lives there.
- `rest` came out **bit-identical** to `orig` over all 64 positions at 8K and at 116K. So the correct
  noise floor for a restore test on this build is **zero**, not `fresh`.
- Against a zero floor, `shift` was detected at every position at both depths. Its minimum per-position
  KL is 0.0010 at 8K and 0.0042 at 116K, both strictly above 0. Top-1 flips: 9/64 and 12/64.
- `shift` being smaller than `fresh` is expected and irrelevant (§1.4). A 1-position RoPE offset touches
  only the 12 attention layers. A fresh prefill perturbs the accumulated state of all 48 layers.

**What O3 never tested.** It never tested the bugs this feature can actually have:

- corrupted recurrent state content;
- a cell-range off-by-one in the KV or indexer copy;
- a restore into a fragmented cell pool;
- a restore into a different sequence id.

It had one weak positive control and no way to localise a failure.

**Recommendation: build O3x, backed by the twin and O1x, and keep O5 as the behavioural backstop.**

1. **O3x, the exact-replay oracle** (primary, runs at 8K and 116K):
   - The gate is `orig` vs `rest`, bit-for-bit.
   - A second restore arm, `rest2`, is the replay-determinism control that licenses the zero floor.
   - A **ladder of 8 blob-level corruption arms**, each modelling a specific bug class in this fork's
     `state_write`/`state_read`, must every one be non-identical. Each arm has a landing check proving
     the corruption is what ended up in memory.
   - An optional **per-layer fingerprint** extends O4's in-graph idea to every layer. It hashes
     `l_last-<il>`, `indexer_top_k-<il>` and `result_output` per decode step. It localises the first
     divergent layer and catches corruptions the logits round away.
2. **The twin oracle** (secondary, 8K and 32K), plus **O1x**:
   - A live sequence, a restored copy in a *deliberately fragmented* cell pool, and a corrupted copy are
     all decoded in **one `llama_decode` call per step**, and their rows are diffed.
   - This covers the multi-run scatter-read and indexer-mirror restore paths. Arm 2 never exercised them
     because it always restored into an empty pool.
   - O1x is the deterministic, forward-pass-free byte check on the same fragmented restore.
3. **The statistical fallback** is used only when replay turns out not to be bit-deterministic.
   - It scores against the *replay* null, which is cheap: seconds per sample at 116K.
   - The *recompute* null (`fresh`, 7 minutes per sample at 116K) is kept only for the separate
     cross-tier question. That null can never detect anything smaller than prefill noise, whatever N is.
4. **O5x**, the semantic oracle, extends O5 into a localisation test with facts at 8 depths and targeted
   span corruptions. It is interpretable, but coarse. It complements the gate and does not replace it.

**Prototype status.** O3x, twin, O1x and the fingerprint are implemented and run on CPU. Results are in
§7 and `logs/pcache/doc20-cpu/`.

- Replay is bit-exact on CPU.
- All 8 corruption arms land exactly: the landing re-save differs from the mutated blob by 0 bytes.
- Each arm is detected, **except where the toy model's own scale makes it genuinely invisible**, and the
  gate reports that as `BLIND` instead of passing it.
- One arm was invisible in the logits but caught by the fingerprint.
- O1x and twin pass on a fragmented restore.
- The original O1-O4 output is byte-identical to Arm 0's.

**GPU power is unverified.**

---

## 1. What tonight's O3 actually measured

### 1.1 How the harness ran (correcting one framing)

The harness does **not** run the four arms as four processes. They run in one process and one context,
one after another: `orig`, then `seq_rm` + restore + `rest`, then `seq_rm` + reprefill + `fresh`, then
`seq_rm` + restore + `shift` (`state_restore_equiv.cpp:1128-1154`). So only `fresh` recomputes the
prefill. Every other arm decodes from the bytes of the one prefill that `orig` also used.

Two more facts matter:

- **The continuation `S` was synthetic in both Arm 2 runs:** an arithmetic sequence of token ids
  (`state_restore_equiv.cpp:1054`).
- **The 8K run's prompt was synthetic too.** The log says `# using synthetic deterministic prompt
  tokens, depth=8192`, and only the 116K run tokenized a real prompt file. On gibberish the model's
  next-token distribution is flat, which makes argmax flips cheap. That is part of why `fresh` flipped
  9-13 of 64 top-1s.

### 1.2 The data

From `logs/pcache/arm2/01-harness-8k.log` and `02-harness-120k.log`:

| arm vs `orig` | 8K KL median / min / max | 8K top-1 | 116K KL median / min / max | 116K top-1 |
|---|---|---|---|---|
| `rest` | **0 / 0 / 0** | 64/64 | **0 / 0 / 0** | 64/64 |
| `fresh` | 0.0056 / 0.0018 / 0.118 | 55/64 | 0.0156 / 0.0048 / 0.494 | 51/64 |
| `shift` | 0.0037 / **0.0010** / 0.116 | 55/64 | 0.0134 / **0.0042** / 0.609 | 52/64 |

### 1.3 The wrong null

The O3 criterion (`state_restore_equiv.cpp:1178-1179`) asks whether `rest` is within 2x of `fresh`, and
whether `shift` is more than 5x `fresh`. That treats `fresh` as the noise that `rest` must stay inside
and `shift` must escape.

But the noise `fresh` measures is **prefill recompute noise**, and `rest` involves no recompute. The
data says so directly: two different decode passes, `orig` and `rest`, over byte-identical memory gave
bit-identical logits at 64 positions at each depth.

**Inference:** on this build, in one process, single-token decode over identical memory bytes is
replay-deterministic. The run-to-run irreproducibility recorded at `PLAN.md:1004-1014` and
`PLAN.md:1129-1136` enters through the multi-token prefill. Its root cause is still open. `PLAN.md`
names CPU-side `MUL_MAT_ID` and SYCL reductions as candidates. O3x does not depend on the cause.

The right question for a restore is therefore "is this bit-identical?", and the right positive control is
"does a known corruption make it not bit-identical?". **By that reading, tonight's `shift` was detected
at 64/64 positions at both depths**, because its minimum KL is above zero.

The one caveat is that tonight measured replay determinism only *implicitly*. `rest == orig` could in
principle be two errors cancelling exactly, which is implausible. O3x adds an explicit `rest2` control so
the zero floor is established in every run, not assumed (§3.2).

### 1.4 Why `shift` < `fresh` is expected

- **Three quarters of the layers ignore position.** `qwen4exp` routes every `is_recr(il)` layer to the
  Gated-DeltaNet builder and the rest to attention (`models/qwen4exp.cpp:455-458`). The production
  model's KV cache reports `12 layers` (`01-harness-8k.log`: `llama_kv_cache: size = 57.38 MiB ( 8704
  cells, 12 layers ...`). So 36 of 48 layers have no RoPE at all.
- **In the 12 that do, shifting all of `S` by +1 is weak.** It keeps every relative distance *within*
  `S`. It changes the distance to each prefix token by exactly 1 out of 8K-116K. Only the high-frequency
  RoPE bands notice.
- **`fresh` is different in kind.** It replaces the whole prefix state, including 36 layers of recurrent
  state accumulated over the entire prompt, with an independently rounded copy.

So a small, real perturbation was compared against a large, harmless one. No cross-run statistic can make
`shift` escape `fresh` on these numbers, for any number of samples (§5.2).

### 1.5 What tonight did not establish

1. **Power against state-content corruption.** The only positive control was a position mutation of the
   continuation, and it never touched the restored bytes.
2. **Replay determinism as a controlled fact** (§1.3).
3. **The fragmented and cross-sequence restore paths.** Every restore was into an emptied pool, into the
   same sequence id. So `find_slot` returned one contiguous block, and two things never ran on the B70:
   - the multi-run branch of `llama_kv_cache::state_read_data` (`llama-kv-cache.cpp:2511-2527`);
   - the indexer adopting a non-trivial attention layout (`llama-kv-cache.cpp:2394-2416`,
     `llama-memory-hybrid-idx.cpp:462-478`).

   Production runs `-np 2 -kvu` (doc 18 §5 Arm 2), so a restore beside another live slot is the normal
   case, not an edge case.
4. **Cross-tier restores.** A restore under a different `n_ctx`, `n_ubatch` or `-ncmoe` is a different
   graph, so bit-identity is impossible by construction. That question stays with the server-level O5
   and the recompute-null fallback (§5).
5. **Sparse-attention visibility.** A corrupted early cell that the QSA top-k never selects cannot move
   the logits at all (§3.4).

---

## 2. Design principles

1. **Every run carries its own negative and positive controls, on the same scale, in the same process.**
   A threshold calibrated on a different day or backend is not a control.
2. **The null for a restore test is replay noise, not recompute noise.** When the backend offers exact
   replay, use it. Fall back to statistics only when it demonstrably does not.
3. **Positive controls must be the bugs this code can have.** Inject them at the blob level, the bytes
   `state_read` consumes. Then check that they landed.
4. **Look inside the graph, not only at the logits.** This is O4's lesson (`PLAN.md:1139-1147`): named
   intermediate tensors are both more sensitive and diagnostic.

---

## 3. The recommended primary oracle: O3x (exact replay plus a corruption ladder)

### 3.1 Procedure

All steps run in one process and one context (`state_restore_equiv.cpp`, the `want_o3x` block):

1. **Prefill `P` as `P[:-1]`, save blob `A_prev`, decode `P[-1]`, save blob `A`.** The split costs
   nothing extra. `A_prev` is the "one token stale" state that the `rs_stale1` corruption splices from.
2. **`orig`:** teacher-force `S` from the live state and record per-position logits.
3. **Replay arms `rest0 .. rest{N-1}`** (`--replay-n`, default 2): `seq_rm`, restore `A`, teacher-force
   `S`.
4. **Optional `fresh0 .. fresh{M-1}`** (`--fresh-n`, default 0): reprefill, then teacher-force. These are
   informational only (§5).
5. **Corruption arms:** `shift`, kept for continuity, plus each blob mutation in §3.3. For each mutated
   blob `M`:
   - `seq_rm`, then restore `M`.
   - **Landing check:** re-save and require the result to equal `M` byte for byte. This proves the
     corruption is what is resident, and that the reader neither refused it nor repaired it.
   - Teacher-force `S`.
   - If the reader *rejects* `M` (restore returns short or throws), record that as a loud detection,
     which is the good outcome.
6. **With `--fingerprint`:** `cb_eval` hashes (FNV-1a) every `l_last-<il>`, `indexer_top_k-<il>` and
   `result_output` at each decode step, for every arm (`dispatch_cb_eval`). `l_last` is the per-layer
   output name (`models/qwen4exp.cpp:486`). `indexer_top_k` is the sparse selection
   (`models/qwen4exp.cpp:996`).

The blob layout parser (`parse_blob`) walks exactly what the writers emit, in this order:

- the 8-byte context header: `io_magic` and the source `seq_id` (`llama-context.cpp:3089-3090`);
- the attention KV section (`llama-kv-cache.cpp:2053-2333`);
- the recurrent section, including the header-less PLE conv rows (`llama-memory-recurrent.cpp:766-990`,
  PLE at `:926`);
- the indexer KV section, a pure suffix (`llama-memory-hybrid-idx.cpp:443-454`).

Two properties are not self-describing: whether cells carry the 12-byte `llama_kv_cell_ext`, and whether
a section has V payloads. So every combination is tried. The parse is accepted only if it consumes the
blob to the last byte, with every per-cell `n_seq_id` and every type/row header plausible. On the
synthetic model, that is exactly one parse (§7).

### 3.2 Verdicts

| condition | verdict |
|---|---|
| every `rest_r` identical to `rest0` (logits, and fingerprints if on), `rest0` identical to `orig`, every applicable corruption arm non-identical or rejected | **PASS** |
| replay deterministic, but `rest0` ≠ `orig` | **FAIL**: the restored sequence does not reproduce the live one |
| replay deterministic, `rest0` = `orig`, but some corruption arm identical | **PASS-WITHOUT-FULL-POWER**: the gate fails for power, and the `BLIND` arms are named |
| some `rest_r` ≠ `rest0` | **INCONCLUSIVE** for the exact tier: re-scored against the replay-noise null (§5) |

With `--fingerprint`, a corruption arm counts as detected if *either* the logits or any fingerprinted
tensor differs. The report says `DETECTED(fingerprint only)` when only the fingerprint saw it. The
exact-tier controls (`rest` vs `orig`, `rest_r` vs `rest0`) then also require fingerprint identity.

### 3.3 The corruption ladder, and the bug each arm stands for

Each arm is a byte edit of the valid blob `A`. It is the blob a buggy `state_write` would have produced,
or, equivalently, the memory a buggy `state_read` would have left.

| arm | mutation | real bug class it stands for | code it guards |
|---|---|---|---|
| `rs_stale1` | recurrent R, S and PLE rows replaced by those from `A_prev` (one token stale) | the writer reading a rollback snapshot plane instead of the current one; a save taken before the last ubatch's state landed | `cell_id = rs_idx_cur * size + ...` (`llama-memory-recurrent.cpp:803`), "logical current state may live in a rollback snapshot plane" (`:919`, `:952`); `set_rs_idx(seq_id, 0)` on read (`:874`) |
| `rs_zero_layer` | the middle GDN layer's S state zeroed | a layer skipped by the write or read loop (`r_l[il] == nullptr` / `s_l[il] == nullptr` skips) | `llama-memory-recurrent.cpp:906-970` (write), `:1108-1177` (read) |
| `rs_garbage` | every GDN layer's S overwritten with O(1) garbage | wrong H2D source pointer, or an uninitialised destination buffer | `io.read_tensor(s_l[il], head * s_size_row, ...)` in `state_read_data` |
| `conv_zero` | every GDN layer's conv history R, plus PLE rows, zeroed | R or PLE dropped while S travels | PLE "has to travel with the first" (`llama-memory-recurrent.cpp:926`) |
| `kv_rowshift_early` | attention K and V of an early cell span each take the next cell's row | off-by-one in a cell-range offset or scatter run | runs built at `llama-kv-cache.cpp:2511-2527`; `range.first * k_size_row` in `state_write_data` |
| `kv_swap_layers` | attention K of layers 0 and 1 swapped | layer-order mismatch between writer and reader | the `for (const auto & layer : layers)` loops in `state_write_data` / `state_read_data` |
| `idx_rowshift_early` | indexer K of the early span shifted one cell | the indexer drifting from the attention layout: the failure `TAG_HYBRID_IDX_SINFO` exists to prevent | `llama-memory-hybrid-idx.cpp:462-478`, `llama-kv-cache.cpp:2394-2416` |
| `idx_zero` | indexer K all zero | indexer section lost or restored stale while attention and recurrent restore fine | the suffix ordering `[TAG_HYBRID_IDX_STATE]` (`llama-memory-hybrid-idx.cpp:446-452`) |
| `shift` | continuation fed at `pos + 1` | the server's `n_past` bookkeeping off by one after a restore (O2 catches the memory side, not this) | kept from O3 |

**Deliberately not included:**

- **Corrupting per-cell `pos` metadata.** A uniform offset on the last position is O2's job already. K is
  stored post-RoPE, so a `pos` swap between neighbours in one QSA block changes nothing observable. A
  wider interior `pos` corruption would change the masking and the QSA bucketing, and is a reasonable
  later addition. It was left out because what `find_slot` and the `DEBUG CHECK` asserts
  (`llama-kv-cache.cpp:2441-2447`) do with non-monotone positions was not traced here.
- **Truncating the indexer suffix.** The reader throws, and `state_drop` clears all three caches
  (`llama-memory-hybrid-idx.cpp:480-485`), which is loud.
- **A pending-rollback save with MTP active** (`n_rs_seq > 0`). This is the real version of `rs_stale1`.
  It needs a speculative-decode step before the save, which this harness does not drive. It is listed as
  a gap in §9.

### 3.4 Sparse visibility, and why `S` should be a question about the early span

At depth, attention is sparse: the indexer selects blocks (`models/qwen4exp.cpp:930-996`). A corruption
in cells the top-k never selects during the 64-token window **cannot** move the logits. That is not a
false negative of the oracle, since the model's output genuinely does not depend on those cells yet. It
is still a latent bug that fires on the first question that does select them.

The CPU prototype shows the effect even at depth 3,000:

- `kv_rowshift_early` over cells 64-127 first changes `l_last-1` at **step 10**, and changes the logits
  only at step 18 (`logs/pcache/doc20-cpu/02-all-d3000.log`).
- Moved to cells 2000-2255, it never changes the logits in 32 steps. It is caught only by the fingerprint
  at step 3 (`03-...-nofp.log` vs `04-...-fp.log`).

Mitigations:

1. **Make `S` a real-text question about facts planted in the early span**, and aim the `*_early`
   corruptions at those facts' cells (`--cont-file`, `--early-c0`, `--early-n`). Then the continuation
   *must* select the corrupted blocks to answer. That is the O5 insight, applied inside the exact oracle.
2. **Gate on the fingerprint**, which sees the selection itself (`indexer_top_k`) and every layer's
   output.
3. **Keep O1/O1x**, which check every byte regardless of selection.

### 3.5 Why the fingerprint belongs in the gate, and its cost

The O4 lesson generalises. Two arms over byte-identical memory with replay-deterministic decode produce
identical intermediate tensors, so any named tensor is a valid exact comparison. And the intermediates
are strictly more sensitive than the logits: the final RMS norm and output projection can round a tiny
hidden-state change back to identical floats, which is what happened in `03`/`04` above.

The fingerprint also localises failures:

- `idx_rowshift_early` first shows up in `indexer_top_k-1`, which points at the indexer.
- `rs_stale1` and `conv_zero` first show up in `l_last-0`, the GDN layer.
- `kv_rowshift_early` first shows up in `l_last-1`, the attention layer.

On a GPU failure, that turns "O3x failed" into "layer N's recurrent state is wrong".

**Cost (estimate, unverified).** Each `ask == true` tensor forces a graph split and a D2H copy. On the
production model that is about 48 `l_last` + 12 `indexer_top_k` + 1 tensor per step. Arm 2 already runs
O4's `cb_eval` in the same harness, so the mechanism is known to work on SYCL. The slowdown is not
measured. Budget up to 3x decode time for the fingerprinted run. It is 64 steps, so at 116K about
6 s becomes about 20 s per arm.

Numerics with the fingerprint on may differ from numerics with it off, because the extra splits can
change fusion. That is harmless, because every arm in one run shares the setting. But logits from a
fingerprinted run must not be compared with an unfingerprinted one.

---

## 4. Same-launch twins and O1x: covering the restore paths O3x cannot reach

### 4.1 Does the API support it? Yes (read, not assumed)

- **Multi-sequence batches.** A `llama_batch` carries a `seq_id` per token and a per-token `logits`
  flag. `llama_get_logits_ith(ctx, i)` indexes by batch position. For hybrid memory with a unified cache,
  `init_batch` uses the non-sequential `split_equal` (`llama-memory-hybrid-idx.cpp:301-309`), which puts
  one token from each of several sequences in one ubatch. This is the path the server's parallel slots
  already take.
- **Restore into a different sequence id.** Both readers take `dest_seq_id`.
  - The KV reader discards the saved seq id and uses the destination (`llama-kv-cache.cpp:2377-2385`).
  - The recurrent reader calls `find_slot` for the destination (`llama-memory-recurrent.cpp:992-1028`).
- **QSA with several sequences in one stream.** Blocks are keyed on (sequence set, bucket) precisely so
  that a unified cache does not pool two sequences together (`llama-memory-hybrid-idx.cpp:595-596`).
  - With more than one sequence present, the single-sequence bucketed shortcut is off and the general
    grouping runs (`:634`, `:946`).
  - The pooled-key cache stays enabled, since its condition is one stream and one ratio (`:970-971`).

  The twin therefore exercises the *multi-slot* QSA path, which is what `-np 2` production runs anyway.

### 4.2 Why same-launch is not automatically bit-exact on GPU, and how the twin calibrates itself

The live sequence and its restored twin occupy *different cell indices*. In one launch, each query row
reduces over its own unmasked cells.

- **On CPU,** masked entries contribute exact zeros and the unmasked ones are summed in the same relative
  order. The prototype got bit-identity (§7).
- **On SYCL,** a split-KV flash-attention kernel partitions the KV axis by cell index. If the two copies'
  cells fall differently across partitions, the partial sums are grouped differently and can round
  differently.

  **Inference:** choosing depth, hole and restore offsets as multiples of the kernel's KV tile makes
  identical grouping likely. That is not guaranteed.

So the twin has two regimes, and reports which one it is in:

- **Exact:** restored = live bit-for-bit. Any difference in the corrupted row is a detection.
- **Tolerance:** if restored ≠ live, the corrupted row must clear the good row's *maximum* per-position
  KL. The report prints the margin, corrupt KL-median over good KL-max.

  Because the good and corrupted rows come out of the same kernel launches, their noise is correlated.
  **Inference:** that noise should sit orders of magnitude below the recompute noise that sank O3, since
  no prefill is repeated. Only the attention partition grouping differs.

### 4.3 Fragmentation

Before the restore, the harness decodes two scratch sequences of `hole` tokens (seq 2, then seq 3), then
removes seq 2. `seq_rm` moves `head` back to the freed cell (`llama-kv-cache.cpp:417-419`). The restore's
`find_slot(ubatch, false)` (`llama-kv-cache.cpp:2419`) then takes the hole first and continues past seq 3.

That produces a two-run cell set. It drives:

- the multi-run scatter read (`llama-kv-cache.cpp:2511-2527`);
- a non-contiguous layout mirrored into the indexer (`llama-kv-cache.cpp:2394-2416`), with its "not
  free" guard (`:2408-2416`).

**Caveat:** that the cells really are non-contiguous follows from this construction and the code. It is
not observed, because the public API exposes no cell indices.

### 4.4 O1x

O1x runs on the same fragmented restore:

1. Save seq 0 → `A`.
2. Restore into seq 1 → re-save → `B`.
3. Normalise the header seq id and every per-cell seq id to 0 (`normalize_seq_ids`).
4. Require `A == B`.

It is deterministic and needs no forward pass. It catches any placement bug in the fragmented read path
that the writer then faithfully reports: data put in the wrong cells, or data not written. It cannot
catch a symmetric write/read bug, which the logits oracles cover.

### 4.5 Memory cost

- **At 8K,** the Arm 2 blob is 196 MB (`01-harness-8k.log`), and three resident copies are well under
  1 GB.
- **At 116K (estimate):**
  - Attention KV plus indexer is about 1.09 GB at 120,320 cells (`02-harness-120k.log`: 793 + 297 MiB).
  - Recurrent state is about 123 MB (196 MB minus the 8K KV share).
  - The twin needs about 3x the cells: roughly 3.3 GB of KV and indexer, comparable to the 300K tier's
    footprint.

**Recommendation:** run the twin at 8K and 32K on the tier-0 config. Run it at 116K only under the 300K
tier's `-ncmoe`, and only if that window has room.

---

## 5. The statistical fallback, made rigorous (only if exact replay fails)

### 5.1 Two different nulls answer two different questions

| null | how sampled | cost per sample at 116K (estimate) | question it answers |
|---|---|---|---|
| **replay** | restore `A` again and teacher-force (`--replay-n`) | about 0.4 s H2D (1.2 GB at the measured 6.84 GB/s) + 64 × about 90 ms (`PLAN.md:997-999`: 89.7 ms/token at 118K) ≈ **7 s** | is this restore equivalent to the live state, on a backend whose decode is not replay-exact? |
| **recompute** | reprefill `P` and teacher-force (`--fresh-n`) | about 7 min (116K at about 275 tok/s; Arm 2's 116K harness took about 16 min for load + 2 prefills + 256 decode steps, per the log mtimes) | is a *cross-tier* restore, which is a different graph, equivalent to recomputing in that tier? |

If `rest2` ≠ `rest0`, the harness falls back to the replay null automatically: every corruption must
exceed every replay sample. The recompute null never gates same-context restores. It is printed as a
percentile for information only.

### 5.2 The hard limit of the recompute null

The recompute null's detection floor is the prefill noise itself. **More samples sharpen the estimate of
that floor. They do not lower it.** Any corruption whose effect is smaller than an honest reprefill's is
undetectable against it, whatever N is. Tonight's `shift` is exactly that case.

So the recompute null can only ever catch gross bugs, like `rs_garbage` or `kv_swap_layers`. That makes
it acceptable only for the cross-tier question, where nothing better exists. O5/O5x must carry that
question too.

### 5.3 Statistic and N

- **Per-position rank, not a median ratio.** For each position `i`, rank the arm's KL among the N null
  samples' KL at `i`. Under exchangeability, the rank is uniform on `{0..N}`, so P(the arm exceeds all N)
  is 1/(N+1) per position.
- **Report two numbers:**
  - the count of positions exceeding every null sample (expected 64/(N+1) under H0);
  - the mean rank.
- **Calibrate the null from the null samples themselves.** Positions are correlated along the
  continuation, so a binomial model overstates significance. Instead, compute the same statistic
  leave-one-out for each null sample, and require the arm to exceed every leave-one-out value. This
  permutation-style calibration costs no extra samples.
- **N.** N = 19 gives a one-sided 5% level for "exceeds all" at a single position.
  - Replay null at 116K: 19 × 7 s ≈ **2.2 min**, so always affordable.
  - Recompute null at 8K: prefill about 20-30 s (estimate from doc 16's 250-400 tok/s), so 19 samples ≈
    **10 min**.
  - Recompute null at 116K: 19 × 7 min ≈ 2.2 h, too costly. Use N = 4 as a coarse sanity check (about
    30 min), and say so in the report.

The prototype implements the replay-null fallback and the recompute-null percentile. The per-position
rank statistic and the leave-one-out calibration are **design only**, needed only if §8 finds replay
non-deterministic.

---

## 6. O5x: the semantic oracle as a complement

O5 worked: 15/15 restored against 15/15 fresh (`logs/pcache/arm2/arm2_step2_report.json`). It is
outcome-based, so it sidesteps noise entirely. Its weakness is that it has no demonstrated power against
subtle corruption. It proves recall survives, not that nothing is wrong. O5x adds localisation:

- **Plant K = 8 facts** at 1, 5, 10, 25, 50, 75, 90 and 99% of the depth. Each is a distinctive name ↔
  code pair that appears nowhere else.
- **Probe all 8 after the restore,** 5 trials each at temperature 0.7, scored as proportions. Control: the
  same probes against the uncorrupted restore.
- **Targeted corruption arms.** For each fact `j`, apply `kv_rowshift` (a wider, ±1-cell shift over the
  fact's cells, all attention layers, plus the indexer rows) to *that fact's span only*.
  - Pass: probe `j` drops (for example, from ≥4/5 to ≤1/5) while the other 7 stay at control.
  - This proves the probe set has spatial resolution, so a clean pass on the good restore means something.

Caveats:

- It needs the harness to generate text and tokenize, so it runs on the real model only, not on the
  vocab-less synthetic fixture.
- Whether an attention-only span corruption defeats recall, when 36 GDN layers also carry a compressed
  summary, is an **open empirical question**. The design measures it rather than assuming it.
- **Cost (estimate):** 8 probes × 5 trials × about 40 generated tokens × 90 ms ≈ 2.4 min per arm at
  116K. With a control and 8 targeted arms, about 22 min plus one prefill.

**Role:** a behavioural backstop and the only oracle for cross-tier restores in the server. It stays
coarse. It is not the gate for same-context restores.

---

## 7. What was prototyped, and what the CPU runs showed

**Implemented** in `staging/work/state_restore_equiv.cpp`. This is an extension, not a rewrite. The O1-O4
code is unchanged, and its output on the Arm 0 invocation is byte-identical to
`logs/pcache/arm0/04-state_restore_equiv.log` (`logs/pcache/doc20-cpu/06-regression-o1-o4-d512.log`).
New in this change:

- oracles `o3x`, `twin` and `o1x`;
- the blob parser and the 8 blob mutations;
- landing checks;
- `--fingerprint`;
- flags `--replay-n`, `--fresh-n`, `--corrupt`, `--twin-corrupt`, `--frag-hole`, `--cont-file`,
  `--early-c0` and `--early-n`;
- `SRE_DUMP_BLOB=path`, a debug dump of `A`.

**Design only:** O5x, the per-position rank statistic and leave-one-out calibration (§5.3), and an MTP
pending-rollback save arm.

**CPU runs** (synthetic `qwen4exp-moe.gguf`, 2 layers: 1 GDN with PLE, 1 attention with an indexer;
n_vocab 128; 4 threads; all in `logs/pcache/doc20-cpu/`):

- **The parser found exactly one consistent layout.** At depth 3,000 it reads: attention 3,000 cells with
  ext, 1 K and 1 V layer; recurrent 1 cell with R 9,216 B, one PLE row and S 131,072 B; indexer 3,000
  cells (`02-all-d3000.log`).
- **Exact tier:** `rest`, `rest1` and `rest2` are bit-identical to `orig`, logits and fingerprints, in
  every run.
  - CPU `fresh` is exact too, so on CPU the fallback percentiles are degenerate. That is expected; CPU is
    deterministic.
- **Every mutation landed exactly:** landing re-save diff 0 in every arm.
- **Detection at depth 3,000:** all 8 blob arms plus `shift` detected
  (`02-all-d3000.log` → `PASS: restore bit-exact, replay deterministic, every corruption arm detected`).
  `kv_swap_layers` is `n/a` because the fixture has one attention layer.
- **An honest BLIND at depth 512.** `rs_zero_layer` was identical in logits *and* fingerprint
  (`01-o3x-d512.log`). The cause was checked:
  - the synthetic S state's magnitude is |S| ≤ 5e-10, mean 1.4e-11 (read from the dumped blob), so its
    contribution sits below float resolution downstream;
  - `rs_garbage` (O(1) values) *is* detected, so the restored S is read;
  - at depth 3,000, where S has grown, `rs_zero_layer` is detected.

  The gate reported `PASS-WITHOUT-FULL-POWER` and exit code 1, which is exactly the behaviour wanted when
  a control has no power. **On the real model, S is not near zero, so this particular blindness should
  not carry over. That is an inference, and the §8 pass checks it.**
- **Fingerprint-only detection:** `kv_rowshift_early` at cells 2000-2255 gave bit-identical logits for 32
  steps, but the fingerprint caught `l_last-1` at step 3 (`03-...-nofp.log`: `BLIND` against
  `04-...-fp.log`: `DETECTED(fingerprint only)`).
- **Twin and O1x** on a fragmented restore (hole 128 at depth 512, hole 750 at depth 3,000):
  - O1x: 0 differing bytes after seq-id normalisation.
  - Twin: restored row bit-identical to live, with the corrupted row detected for `kv_rowshift_early`,
    `idx_zero`, `rs_garbage` and `rs_stale1` (`05-twin-o1x-*.log`, `02-all-d3000.log`).
- **Wall time:** the whole `02` run, with every oracle, 4 fresh arms and the fingerprint, took 3.7 s.

**What CPU cannot tell us:**

- whether SYCL decode is replay-exact in the same context (tonight's `rest == orig` says very probably,
  at least without the fingerprint);
- whether the twin is exact or needs the tolerance regime on SYCL;
- the real model's sensitivity to each corruption at 116K under sparse attention.

---

## 8. What a future GPU validation pass needs to do (owner-scheduled)

This follows doc 18 Arm 2's discipline:

- one batched stop of `llm-b70`;
- no other llama or SYCL process running;
- wait for the actual process exit and check `free -h` before every load (`MEMORY.md`);
- production flags (`-ctk q4_0 -ctv q4_0 -fa 1 -kvu`, harness `--ncmoe 25`).

Rebuild the harness against the SYCL tree the way `staging/work/kvstate-arm2-harness/` was built. Make
the prompt real text with 8 planted facts, used for both P and a question continuation `S` (`--cont-file`),
and point `--early-c0/--early-n` at the first fact's cells.

| step | command sketch | budget (estimate) | what it establishes |
|---|---|---|---|
| 1 | `--oracles o3x --depth 8192 --replay-n 5 --fresh-n 19` | prefill about 30 s + 5 replay + 9 corruption arms at about 3 s each + 19 × about 35 s fresh ≈ **12 min** | replay determinism on SYCL, stated explicitly; per-arm sensitivity at 8K; the recompute null as a real distribution, for the cross-tier record |
| 2 | step 1 plus `--fingerprint`, `--fresh-n 0` | ≈ 3 min | fingerprint replay-exactness on SYCL; where each corruption first appears |
| 3 | `--oracles o3x --depth 120000 --replay-n 5 --fingerprint` | 1 prefill ≈ 7-8 min + 14 arms × about 7-20 s ≈ **12 min** | the gate at real depth, with sparse attention |
| 4 | `--oracles twin,o1x --depth 8192`, once per `--twin-corrupt` in {`rs_stale1`, `kv_rowshift_early`, `idx_rowshift_early`}, then once at 32K | ≈ 5 min at 8K + about 8 min at 32K | the fragmented/cross-seq restore paths on SYCL; whether the twin is exact or in the tolerance regime |
| 5 (optional) | step 3 with `--fresh-n 4` | + about 30 min | a coarse recompute null at depth, for the cross-tier record only |

Steps 1-4 total **about 40 minutes**, inside one window.

**The redesigned oracle is validated on the B70 if:**

1. **Replay is exact.** `rest_r == rest0 == orig` at 8K and 116K, logits and fingerprints. If it is not,
   the oracle drops to the replay-null fallback. Then the finding to report is that same-context decode
   is not deterministic on SYCL, and the next job is §5.3's rank statistic.
2. **Every applicable corruption arm is `DETECTED`, or rejected loudly, at both depths.** Any `BLIND` arm
   at 116K is reported with its fingerprint. A `BLIND` `*_early` arm with the fact-question continuation
   would mean the early-span selection assumption is wrong, and O1/O1x become that region's only guard.
3. **Twin:** restored = live, or, in the tolerance regime, every corruption row clears the good row's KL
   maximum, with the margin reported.
4. **O1x:** 0 bytes on SYCL.

**What would count against this design:**

- replay non-determinism in the same context;
- a corruption from §3.3 that is `BLIND` in logits *and* fingerprint at 116K despite the targeted
  continuation;
- a twin good row that differs from live by as much as a corruption row does.

Any one of these means the backend's noise, or the sparse selection, defeats even same-process
comparison for that region. The report must then say so instead of rounding it to a pass.

---

## 9. Limits and open items

- **Cross-tier restores stay unprovable at the exact level.** A different `n_ubatch` or `-ncmoe` means a
  different graph. O5/O5x and the recompute-null percentile are the only evidence there, and §5.2 bounds
  what that can show.
- **MTP pending-rollback saves** (`n_rs_seq > 0`, `llama-memory-recurrent.cpp:781-803`) are the real
  version of `rs_stale1`, and they are untested. An arm that runs one speculative step with rollback
  before saving needs the MTP context in the harness. That is future work.
- **The blob parser assumes `n_stream == 1`** (`-kvu`, as production and the harness run). A
  multi-stream blob is rejected rather than misparsed.
- **The fragmentation is by construction, not observed** (§4.3).
- **Replay determinism may depend on things this harness holds fixed:** the same context, the same
  ubatch shape, and single-token decode. The server's restores happen under its own batching. O3x
  answers "are the restored bytes right". Whether the server feeds them correctly is O2's and O5's job.
- **Whether the fingerprint's extra graph splits change SYCL kernel selection enough to break replay
  exactness** is unknown. Step 2 of §8 measures it, and O3x without the fingerprint still stands on its
  own.
