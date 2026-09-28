# Conversation State Persistence Across Reloads and Restarts — Implementation Plan

Date: 2026-09-27. Status: **plan only. No code was written, nothing was built,
no GPU or model was loaded, and the production `llm-b70` server was not
touched.** This document turns doc 17's recommendation (Option A, §6.2) into
stages and tasks a coding agent can pick up one at a time.

Inputs:

- `16-prefill-speedup-scoping.md` §4.1: about 56% of prefilled tokens on the one
  logged day were reprefills after a reload.
- `17-kv-state-persistence-scoping.md`, all of it: the three state kinds
  already serialize, the size and I/O numbers, and the validation arms.
- `13-dynamic-context-aware-expert-placement.md` §6 Option A, §9.8, §13-§15:
  the tier and sleep reload paths these hooks go into.
- The production compose file,
  `../coding-agent/docker-compose.qwen4exp-moe.override.yml`, read-only.
- `gh pr view/diff 28092` and `gh pr view 26004`, as of today.

All code citations are to the **live fork tree `src/llama.cpp/`**: pinned
`6d9c82ea2` plus this project's uncommitted changes, which is what
`patches/llama.cpp.patch` is generated from. Paths are relative to
`src/llama.cpp/` unless they start with `docs/`, `staging/`, `logs/`,
`docker/` or `PLAN.md`.

`src/moe-cache-fork/` is a different, unrelated reference tree. Nothing in
this plan reads from it or touches it.

Conventions are those of docs 16-17:

- `file:LINE` means the line was read directly.
- **Inference** marks a conclusion drawn by combining separately read facts.
- **Estimate** marks arithmetic.

---

## 0. Summary

**What gets built.** A server-only feature. The server's existing host-RAM
prompt cache (`server_prompt_cache`) gets a disk tier on the Garudias NVMe.

- Every live slot and every RAM cache entry is written to disk:
  - before the model is torn down for a tier switch or a sleep, and
  - on SIGTERM.
- On every `load_model()`, whether a fresh start, a wake or a tier switch, the
  server scans that directory, but reads only the token lists.
- The existing LCP matcher picks an entry for the next request. Only then is
  that entry's blob read from disk, lazily.
- The unit that gets persisted is the same `server_prompt_cache_state` the
  server already uses: tokens, context checkpoints and the full
  `llama_state_seq_get_data` blob. That is also what makes upstream PR #26004
  unnecessary here (§4.3).

**Staging.**

1. **Stage 0: validation harnesses, no feature code.** Arm 0, plus a new
   libllama-level oracle harness.
2. **Stage 1: carry the prompt cache in RAM across an in-process reload.**
   About 150 LOC. It proves the restore mechanics with the smallest possible
   diff, and it stays as the fallback when no disk directory is configured.
3. **Stages 2-3: disk format, lazy restore, startup scan, SIGTERM save.**
4. **Arm 1: server mechanics, CPU-only, on `Qwen3.6-35B-A3B`.**
5. **Arm 2: one owner-scheduled production window on the B70.**
6. **Stage 4: deploy behind a default-off flag.**
7. **Stage 5 (deferred): crash survival.**

Nothing touches `llm-b70` until Arms 0 and 1 pass. Nothing is deployed until
Arm 2 passes.

**Size.** About 960 LOC of server and common code, plus about 770 LOC of test
harness and scripts (§7). That is larger than doc 17's 250-400 LOC for three
reasons:

- lazy disk entries (needed so a restart doesn't have to fit every saved
  conversation into `--cache-ram`),
- the alignment guard, and
- the unified-KV ordering fix in §4.4.

**The two hard risks, answered from the code:**

- **Restoring into a different tier is valid.** `llama_state_seq_set_data`
  checks nothing that `n_ctx`, `n_ubatch` or `-ncmoe` change. It fails *safely*
  (a full reprefill) when the cells don't fit. It **silently accepts** a
  mismatch in RoPE parameters, so those must go in the file fingerprint (§4.1).
- **Correctness at depth on SYCL.** The gate is built from four oracles, and
  none of them is a text diff (§4.2):
  - byte idempotence,
  - a position-alignment invariant,
  - a logits divergence measured against a same-process noise floor, with a
    deliberate off-by-one mutation to prove the metric can detect it,
  - an early-context needle question, scored as a count against a
    fresh-prefill control.

**The biggest new risk found here** (it is not in docs 16 or 17): with
production's `-np 2 -kvu`, a restore runs **before** the server evicts the other
idle slot's KV (§4.4).

- The ordering: `get_available_slot` calls `prompt_load` at
  `server-context.cpp:1831`. The idle-slot save-and-clear runs later, at
  `:2624-2640`, after the task has launched.
- So a second persisted conversation is restored while the first one still
  holds its cells.
- If the two together exceed the tier's pool, `find_slot` fails and the server
  silently reprefills the whole conversation. The feature would appear to work
  in every single-conversation test and then miss in real two-conversation use.
- This is a latent upstream-shaped bug in the RAM cache today. Persistence makes
  it common, because every wake brings back more than one conversation.
- Fix: about 40 LOC, in Task T4.

---

## 1. What exists today, re-read for this plan

### 1.1 The server's prompt cache is already the right unit

- `server_prompt` (`tools/server/server-task.h:566-586`) is `server_tokens` plus
  `std::list<common_prompt_checkpoint>`.
- `server_prompt_cache_state` (`:597-610`) adds `server_prompt_data {main,
  drft}` (`:588-595`), which holds the raw blobs.
- `server_prompt_cache` (`:612-635`) is a `std::list` of those, limited by bytes
  (`--cache-ram`) and by tokens (`limit_tokens = n_ctx`, set at construction).

The four operations:

- **`server_slot::prompt_save`** (`server-context.cpp:300-324`). It sizes the
  blob with `llama_state_seq_get_size_ext(..., FLAGS_NONE)`, calls
  `prompt_cache.alloc`, then `llama_state_seq_get_data_ext` into the new
  vector.
- **`server_prompt_cache::alloc`** (`server-task.cpp:1711-1791`) does, in order:
  1. Skip if an existing entry already contains the whole prompt
     (`:1713-1720`).
  2. Skip if the entry alone exceeds `limit_size` (`:1731-1735`).
  3. Erase entries that are strict prefixes of this one (`:1738-1748`).
  4. Evict the oldest entries to make room (`:1750-1758`).
  5. Copy tokens and **checkpoints** (`:1779-1788`).
- **`server_prompt_cache::load`** (`:1793-1868`):
  1. Pick the best entry by `f_keep` and `f_sim` (`:1804-1823`, skipping
     `f_keep < 0.25`).
  2. `llama_state_seq_set_data_ext(ctx_tgt, blob, size, id_slot, 0)` (`:1832`).
  3. On success, **move** the prompt (tokens plus checkpoints) into the slot and
     erase the entry (`:1862-1864`).
  4. On a size mismatch, return `false` and keep the entry (`:1833-1837`).
- **Caller** (`server-context.cpp:1818-1839`). This runs only on the LRU
  selection path, or when LCP selection would drop more than half the slot
  (`:1788-1790`). On a failed load it calls `prompt_clear()` (`:1831-1833`),
  which gives a full reprefill.

### 1.2 Where reloads happen

| event | code | slots idle? | lock held |
|---|---|---|---|
| tier switch | `ctx_tier_update`, `server-context.cpp:1076-1147`; `destroy()` at `:1122`, `load_model` at `:1123`, fallback reload `:1125-1132` | yes, checked `:1101-1106` | `mutex_model` exclusive (`:1109`) |
| sleep entry | `handle_sleeping_state(true)`, `:971-985`; `destroy()` at `:985` | yes (only fires on an empty queue) | `mutex_tasks`: the callback is invoked under it (`server-queue.cpp:325-334`) |
| wake | `handle_sleeping_state(false)`, `:986-999`; tier from `depth_hint` `:988-995`, `load_model` `:996` | yes | `mutex_tasks` (`server-queue.cpp:343-350`) |
| fresh start | `server.cpp:477` → `load_model`; `is_resume == false` → `init()` (`server-context.cpp:1571-1573`) | n/a | none |
| SIGTERM | `server.cpp:491-495` → `ctx_server.terminate()` unblocks `start_loop()` (`server-context.cpp:4352-4355`); then `clean_up()` (`server.cpp:547`) | loop stopped; a slot may have been mid-task | none |

`load_model` rebuilds `slots` (`:1420`, `:1436-1496`). Unconditionally, it also
**recreates `prompt_cache`** (`:1533-1541`). That line is the whole of doc 13
§9.8's limitation.

### 1.3 Production configuration that matters here

From `../coding-agent/docker-compose.qwen4exp-moe.override.yml`, read-only:

- `-np 2 -kvu -ctk q4_0 -ctv q4_0 -fa 1`
- `--context-tiers 122880:2048:25,262144:2048:30,512000:1024:32`
- `--rope-scaling yarn --rope-scale 1.0001 --yarn-orig-ctx
  ${B70_YARN_ORIG_CTX:-513000}`: the nearunity unlock from doc 13 §12
- `--sleep-idle-seconds 3600`
- `mem_limit: 90g`
- `/models` mounted `:ro`
- no `stop_grace_period`, so Docker's default of 10 s applies
- no `--cache-ram`, `--ctx-checkpoints` or `--cache-idle-slots` flags, so the
  defaults apply: 8,192 MiB, 32 checkpoints, min step 8,192, idle-slot caching
  **on** (`common/common.h:640-644`)

Context checkpoints **are active** for this model:

- `do_checkpoint` requires `ctx_tgt_seq_rm_type` to be `FULL` or `RS`
  (`server-context.cpp:3648-3661`).
- `common_context_can_seq_rm` returns `FULL` when a partial `seq_rm` fails
  (`common/common.cpp:1589-1626`), which the recurrent half of a hybrid always
  does.

### 1.4 Upstream #28092 (`--cache-disk`) and why this plan does not vendor it

`gh pr view 28092`: OPEN, +1706/-71, updated 2026-09-24. No reviews. By its own
disclosure, written by Codex and Claude. It implements this same feature for
upstream: a disk-backed prompt cache, checkpoints, restore across restarts, and
LRU by disk size.

Its cache key (`gh pr diff 28092`, `server_prompt_cache_key`) includes
`n_ctx_tgt`, `n_batch_tgt`, `n_ubatch_tgt` and `n_parallel`. **On this
deployment that makes every `--context-tiers` tier a different cache**: a
conversation saved at tier 1 would be invisible after a wake into tier 0 or
tier 2. §4.1 shows none of those four fields affects whether a restore is
valid.

It also replaces the RAM cache with an mmap-backed disk cache, instead of
adding a tier under it. And it rewrites `server-task.cpp` by about 1,000 lines.
That would be a painful merge against this fork's own `server-context.cpp`
changes (tiers, the wake hint, `mutex_model`).

**Recommendation:** build the smaller design below. It borrows #28092's
fingerprint field list (it covers RoPE, LoRA and control vectors, which this
plan also needs) minus the four fields above. Re-evaluate if #28092 merges
upstream. It is the natural thing to rebase onto at the next upstream bump.

---

## 2. What gets saved, and when

### 2.1 The persisted unit

One **entry** is one `server_prompt_cache_state`:

| field | source | size at 300K (estimate, doc 17 §4.1) |
|---|---|---|
| tokens | `server_prompt::tokens`, serialized with `server_tokens::serialize()` (`tools/server/server-common.h:213`) | 1.2 MB |
| checkpoints: only the newest `--prompt-cache-disk-ckpts` (default 4) | `common_prompt_checkpoint` (`common/common.h:1182-1230`): `n_tokens`, `pos_min`, `pos_max`, `data_tgt`, `data_dft`, `data_spec` | about 112.6 MiB each (recurrent state only, `PARTIAL_ONLY`) |
| main blob | `llama_state_seq_get_data_ext(ctx_tgt, …, FLAGS_NONE)`. Covers the attention KV, the recurrent state including the PLE row, and the indexer KV (doc 17 §2.2) | 2.77 GiB |
| draft blob | same, for `ctx_dft` (empty in production) | 0 |
| metadata | §3.2 header | < 4 KiB |

The pooled QSA key cache is not saved. The fork invalidates it on every
`state_read` (`src/llama-memory-hybrid-idx.cpp:457`) and rebuilds it on the
next ubatch (doc 17 §2.3).

**Why only the newest 4 checkpoints.** The defaults allow up to 32 per slot,
about 3.5 GiB at depth (`common/common.h:641-643`). Doc 17 §2.6 found that 237
of 242 production LCP selections are strict extensions, which need no
checkpoint at all. The newest few cover:

- the `[TAG_PROMPT_LOGITS]` identical-prompt case (`server-context.cpp:3590-3595`),
- the SIGTERM-mid-generation retry case (§2.3), and
- a last-turn edit.

### 2.2 Save triggers

All saves run on the main loop thread, which is the only thread allowed to
touch `llama_context`.

| # | trigger | hook point | what is written |
|---|---|---|---|
| S1 | tier switch | `ctx_tier_update`, **between `ctx_tier_set(tier)` (`:1118`) and `destroy()` (`:1122`)** | every non-empty slot, plus every RAM cache entry |
| S2 | sleep entry | `handle_sleeping_state(true)`, **immediately before `destroy()` (`:985`)** | same |
| S3 | graceful shutdown | `server_context::start_loop()` (`:4352-4355`), **after `queue_tasks.start_loop(...)` returns**, only if `!impl->sleeping` | same. If the process is sleeping, S2 already persisted everything and the context is gone. |
| S4 | RAM-cache eviction (disk mode only) | `server_prompt_cache::update()` (`server-task.cpp:1870-1892`) and `alloc()`'s make-room loop (`:1750-1758`) | the evicted entry, written instead of dropped, then it becomes disk-only |
| S5 (Stage 5, deferred) | idle persist, for crash survival | a second idle timer in `server_queue::start_loop()` (`server-queue.cpp:322-360`), the doc 13 §13 Option C pattern | idle non-empty slots, once each, after N seconds idle |

**Why S3 goes in `start_loop()`, not in the signal handler.** The handler runs
on a signal context or an HTTP thread (`server.cpp:28-37`, `:491-495`). It must
not touch the context. `start_loop()` returns on the main thread after the
queue loop exits (`server-queue.cpp:299-301`, `:337-340`), and at that point no
decode is in flight. Every `llama_state_seq_get_data_ext` call synchronizes
first anyway (`src/llama-context.cpp:4203-4206`).

A second SIGTERM or SIGINT still hard-exits (`server.cpp:29-33`). The atomic
write in §3.3 means that leaves at most a stray `.tmp` file, never a corrupt
entry.

**Lock caveat.** S1 runs under `mutex_model` held exclusively. S2 runs under
`mutex_tasks`, because the sleep callback is invoked with the queue mutex held
(`server-queue.cpp:325-334`). So for the length of the save:

- HTTP threads that tokenize (S1) or post a task (S2) block for the save
  duration, about 1-2 s per 300K entry by doc 17 §4.3.
- Nothing deadlocks. The save does not wait on either of them.

This is acceptable at a reload boundary, which already costs minutes.

**A slot that is mid-task at SIGTERM.** Persist it only if the alignment guard
in §2.4 passes. A client whose response was cut off will usually retry the same
prompt, which is a strict *prefix* of the saved tokens (prompt plus partial
generation). Resuming then needs a checkpoint at or before the end of the
prompt. The server already creates one about 4 tokens before the end of every
prompt (`server-context.cpp:3752-3765`), and it is among the newest 4 that get
persisted. **Inference**: that retry then costs about 4 tokens of prefill, not
the whole conversation.

### 2.3 Spill versus carry at S1 and S2

- **Stage 1, no disk configured: carry.** `prompt_save` every non-empty slot
  into the RAM cache, then keep the `prompt_cache` object across the reload
  (§5, Stage 1). The cost is up to `--cache-ram` (8 GiB) of anonymous memory
  held *during* the model reload. That is inside the 90 GiB container limit,
  but see the reload-RAM incident rule in `MEMORY.md`. The Stage 1 validation
  records `free -h` across the reload.
- **Stage 2+, disk configured: spill.**
  1. Write each live slot straight to disk. This does **not** go through
     `alloc()`: at 600K, a 5.42 GiB blob plus 32 checkpoints exceeds the
     8 GiB `--cache-ram` limit and `alloc()` would silently skip it
     (`server-task.cpp:1731-1735`).
  2. Write every RAM entry to disk.
  3. Drop all RAM blobs, keeping only the token lists as disk-only index
     entries.
  4. Then `destroy()`.

  The reload then runs with no prompt-cache RAM held. The peak is one entry's
  blob at a time, before `destroy()`: at most 5.42 GiB at 600K, on top of the
  still-loaded model. Stage 5 can remove that peak by streaming (§6).

### 2.4 The alignment guard, applied on every save and every restore

This is the specific failure behind sglang#39830 and arXiv 2609.15030 (doc 17
§5): the restored recurrent state and the server's token count disagree by one
position. For this hybrid model, the public API can detect that without any
forward pass:

- `llama_memory_hybrid::seq_pos_min` is `max(attn_min, recr_min)`, and
  `seq_pos_max` is `min(attn_max, recr_max)` (`src/llama-memory-hybrid.cpp:172-180`).
- The recurrent cache holds exactly one position per sequence, the last one.
- So, when aligned: `pos_min == pos_max == n_tokens - 1`.
  - If the recurrent state is one position **behind**, `pos_max` is
    `n_tokens - 2`.
  - If it is one **ahead**, `pos_min > pos_max`.
  - Both are caught by one check (**inference**, from the two definitions
    above).

The guard is a `bool slot_state_aligned(const server_slot &)`:

- When `ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL` (hybrid or
  recurrent), require
  `llama_memory_seq_pos_min(mem, id) == llama_memory_seq_pos_max(mem, id) ==
  slot.prompt.tokens.pos_next() - 1`.
- Otherwise, require `pos_max == pos_next() - 1`.

Where it runs:

- **Save side:** do not persist a slot that fails it, and log `SLT_WRN`.
- **Restore side:** run it right after `prompt_cache.load` succeeds. On failure,
  `prompt_clear()` (full reprefill) and **delete the file**. A blob that
  restores misaligned once will do it again.

**What it cannot catch:** a blob whose position labels are right but whose
*contents* are wrong, for example a SYCL H2D bug. The oracles in §4.2 cover
that case.

**To verify in Arm 1, not assumed:** that the invariant actually holds for an
idle slot after generation stops. Doc 17 §2.4 read that a sampled token enters
`prompt.tokens` only when it is added to a batch (`server-context.cpp:531-542`),
so the final unsampled token is in neither. Arm 1 step 2 logs both sides on
every save.

---

## 3. Where it is saved

### 3.1 Location

- Host: `/media/da3dsoul/Garudias/llm-state/qwen4exp-b70/`. This is a new
  directory on the dedicated NVMe: 3.5 TiB free, 4.4-4.9 GB/s read and
  2.4-4.0 GB/s write, per doc 17 §4.2.
- It is a sibling of `models/`, not inside it, because `/models` is mounted
  read-only.
- It is **not** on Golias (bcache/RAID5), the root QLC disk, or tmpfs (doc 17
  §4.2).
- Container mount: `- /media/da3dsoul/Garudias/llm-state/qwen4exp-b70:/state`
  (rw).
- Flag: `--prompt-cache-dir /state`.
- The container runs as root, so the files will be root-owned. Clean-up from
  the host needs `sudo`, or a `find -delete` inside the container.

### 3.2 Layout and naming

```
/state/
  <fp>/                      fp = first 16 hex chars of the fingerprint hash (below)
    fingerprint.txt          the full plain-text fingerprint, for humans and for mismatch logs
    <tokhash>.pcs            one entry. tokhash = 16 hex chars of a 64-bit hash
                             (FNV-1a or std::hash) over the token-id bytes
    .<tokhash>.pcs.tmp.<pid> in-flight write, never read
```

**The fingerprint** is a plain-text `key=value\n` string, hashed. It contains
every field that changes what a blob *means* but that `state_read` does not
check itself (§4.1):

- Model identity: for each GGUF shard, the path, size and mtime (production
  loads a 3-shard GGUF), plus `llama_model_n_params`, `llama_model_size` and
  the `general.architecture` metadata value.
- Cache layout: `cache_type_k`, `cache_type_v`, `flash_attn_type`,
  `kv_unified`, `swa_full`, `no_kv_offload`.
- **RoPE: `rope_scaling_type`, `rope_freq_base`, `rope_freq_scale`,
  `yarn_ext_factor`, `yarn_attn_factor`, `yarn_beta_fast`, `yarn_beta_slow`,
  `yarn_orig_ctx`**, written with `std::hexfloat` as #28092 does.
- LoRA adapter paths and scales, and control-vector files and strengths.
  Simpler still: **refuse to persist at all when any LoRA or control vector is
  loaded**, and log it.
- The draft model path, or empty.
- Versions: `LLAMA_STATE_SEQ_VERSION` (`include/llama.h:49`), `llama_commit()`
  (`common/build-info.h:7`), and a new `PCS_FORMAT_VERSION = 1`.
- **A hand-bumped `PCS_COMPAT_EPOCH`.** This fork carries uncommitted memory
  and model-code changes, so `llama_commit()` does not identify the fork's KV
  semantics. The rule: bump the epoch in any change to
  `src/llama-memory-*.cpp`, `src/llama-kv-cache.cpp` or
  `src/models/qwen4exp.cpp` that changes what a cached row means. Put that rule
  in the constant's comment.

**Deliberately excluded:** `n_ctx`, `n_batch`, `n_ubatch`, `n_parallel`,
`-ncmoe`/`tensor_buft_overrides`, `mlock`/`mmap`, and threads. §4.1 shows none
of them changes blob validity. This is exactly the #28092 difference in §1.4.

**Mismatch behaviour.** A different fingerprint gives a different `<fp>/`
directory, so old entries simply aren't found. The disk-budget garbage
collector (§3.4) deletes stale `<fp>/` directories oldest-first. Old-model
entries therefore can never restore, and the operator never has to clean up by
hand.

**Multiple slots.**

- **Files are keyed by content, not by slot index.** A blob restores into
  whatever slot id it is given: the KV cache discards the saved seq id
  (`src/llama-kv-cache.cpp:2382-2386`), and the recurrent cache writes
  `n_seq_id = 0` for single-sequence saves (`src/llama-memory-recurrent.cpp:883`).
  So slot 0's conversation can resume in slot 1.
- Two slots holding identical token lists produce the same name. The content is
  interchangeable, and `rename` is atomic, so the last writer wins harmlessly.
- All writes happen on the main thread one after another, so there is no
  concurrent-writer case to handle.

### 3.3 File format `.pcs`, version 1

All integers are little-endian.

```
u32  magic 'LPCS'           u32 format_version = 1
u32  header_len             bytes of UTF-8 JSON header:
       { "fp": "<full fingerprint text hash>", "n_tokens": N, "n_ckpt": C,
         "main_size": M, "drft_size": D, "created": unix_s, "last_used": unix_s,
         "saved_by": "sleep|tier|shutdown|evict", "pos_min": p0, "pos_max": p1 }
u64  tokens_bytes           server_tokens::serialize() output
     [tokens]
repeat C times (oldest first, same order as server_prompt::checkpoints):
  i64 n_tokens  i32 pos_min  i32 pos_max
  u64 len_tgt [data_tgt]  u64 len_dft [data_dft]  u64 len_spec [data_spec]
u64  M  [main blob]         verbatim llama_state_seq_get_data_ext output (io_magic + seq_id + memory stream)
u64  D  [draft blob]
u64  total_len              must equal file size
u32  trailer magic 'SCPL'
```

Rules:

- **Tokens come before the checkpoints and blobs.** A startup scan then reads
  only `header + tokens`: about 2.4 MB per 600K entry.
- **Writing:**
  1. `fopen(".tmp", "wb")`, with a large `setvbuf`.
  2. Write every field.
  3. `fflush`, then `fsync(fileno)`.
  4. `rename` onto the final name.
  5. `fsync` the directory fd.

  The fsyncs are what upstream's file path lacks (doc 17 §2.1). **Estimate:**
  at the measured 2.4 GB/s buffered-plus-fsync rate, 1.2 s for 300K and 2.3 s
  for 600K.
- **Validation on read:**
  - magic, format version, fingerprint equal to the directory's, `total_len`
    equal to the file size, trailer present;
  - `server_tokens::validate(ctx_tgt)` (`server-common.h:238`);
  - `n_tokens <= slot n_ctx`.

  Any failure deletes the file and logs a warning.
- **No content checksum in v1.** Truncation is the realistic failure mode, and
  the length plus trailer catch it. A 64-bit hash over 2.7 GiB would add about
  0.3-1 s per save (**estimate**, at 3-10 GB/s). If one is added later, put it
  over the header and tokens only.

### 3.4 Budget and garbage collection

- `--prompt-cache-disk-mib N`, default 65,536 (64 GiB): about 23 entries at
  300K. Garudias has 3.5 TiB free.
- Enforced after every spill, as LRU by header `last_used`:
  1. entries in stale `<fp>/` directories go first,
  2. then the oldest entries in the current directory.
- `--prompt-cache-disk-min-tokens N`, default 4,096. Don't persist short
  prompts: they reprefill in seconds and would otherwise fill the directory
  with thousands of files.
- `alloc()`'s "erase entries that are strict prefixes of this one" rule
  (`server-task.cpp:1737-1748`) is extended to **delete the file** of any
  erased disk-backed entry. Without that, a conversation that grows by 10K per
  turn would leave a trail of near-identical files.

---

## 4. What gets restored, and the hard risks

### 4.1 Hard risk 1: is restoring into a different tier valid?

This was read end to end, not assumed.
`llama_state_seq_set_data_ext` (`src/llama-context.cpp:4206-4209`) calls
`synchronize()`, then `state_seq_set_data` (`:3099-3136`). That reads
`io_magic` and the saved seq id, then calls `memory->state_read(io, seq_id)`.
For `qwen4exp` that is `llama_memory_hybrid_idx::state_read`
(`src/llama-memory-hybrid-idx.cpp:456-488`): attention KV, then recurrent, then
the indexer KV (which mirrors the attention cache's cell layout). Any exception
drops the sequence from all three caches (`:481-487`, `state_drop` `:490-506`).

What each stage checks:

| check | where | depends on n_ctx / n_ubatch / -ncmoe? |
|---|---|---|
| `n_stream` equal | `llama-kv-cache.cpp:2151-2155` | **No.** `n_stream` is 1 under `-kvu`, and `n_seq_max` otherwise. It depends on `kv_unified` and `-np`, which are in the fingerprint anyway. |
| enough free cells: `find_slot(ubatch, cont=false)` for `cell_count` cells | `:2418-2424` | **Yes, capacity only.** Cells need not be contiguous. A blob with more cells than the new pool has free fails, and gets `state_drop`, a `false` return, `prompt_clear`, and a full reprefill. **Safe, never corrupt.** |
| layer count, `v_trans` (`-fa`), K/V type and row size per layer | `:2529-2612` | No. Types are in the fingerprint anyway. |
| recurrent: one cell per sequence, found in a pool of `n_seq_max` cells, not `n_ctx` | `llama-memory-recurrent.cpp:992-1033` | **No.** The recurrent pool size is independent of context length. |
| recurrent layer count, r/s/p types and row sizes | `:1094-1135` | No |
| indexer: must land exactly on the attention cells | `llama-kv-cache.cpp:2393-2417` | No |
| positions | read from the blob into each cell (`:2388`, `:1017`) | No. Positions travel with the data. |

Three things are **not checked**:

- **RoPE parameters.** The K cache holds post-RoPE keys, so a restore under a
  different `freq_scale` or `yarn_orig_ctx` pairs old keys with differently
  rotated queries. That produces fluent-but-wrong output, silently.
  - Concretely: `rope_freq_scale` and `n_ctx_orig_yarn` come straight from the
    flags (`src/llama-context.cpp:133-138`). They are **not** derived from
    `n_ctx`, so a tier switch keeps them fixed (**inference** from that code,
    together with the compose file passing them globally).
  - But `B70_YARN_ORIG_CTX` is an environment knob. An operator who changes it
    between a save and a restore would get silent corruption without the
    fingerprint.
- **Model identity:** doc 17 §2.1.
- **LoRA and control vectors:** they change the KV contents.

All three are in the fingerprint (§3.2).

**Answer.** Restoring into a different tier's context is valid as long as the
cells fit. The plan needs no normalization or padding.

- The tier policy already makes them fit in the common case.
  `ctx_tier_update` runs **before** `get_available_slot` (`:2581-2586`) and
  sizes the tier by the incoming request's own token count plus the reserve
  (`:1083-1084`).
- A strict extension is always at least as long as the saved entry, so the
  tier chosen for the request is always big enough for the entry.
- On wake, the tier comes from `req.body.size()/3.5` (doc 13 §15), which only
  over-estimates.
- So doc 17 §6.2's suggestion to make the wake `depth_hint` consider persisted
  entries is **not needed**. The waking request carries the conversation it
  resumes.

**The one exception is `-kvu` occupancy by the other slot.** See §4.4.

**Fresh process start.** The server always comes up at tier 0 (`ctx_tier_init`
calls `ctx_tier_set(0)`, `:1042`). Resuming a 150K conversation after a restart
therefore costs, **estimate**:

- one tier switch, 88-152 s measured in doc 13 §12, plus
- the restore, about 1 s,

instead of a tier switch plus about 8 minutes of prefill. Starting at the tier
of the newest persisted entry would save the switch, but it would repeat doc 13
§14's double-reload trap for a shallow first request. **Not recommended.**

### 4.2 Hard risk 2: the correctness gate at depth on SYCL

**Constraint.** This stack's forward pass is not reproducible even with
teacher-forced fixed tokens: "cached-vs-cached mismatched in 396 of 480 tensor
comparisons" (`PLAN.md:1129-1136`), and greedy output differs between runs
(`PLAN.md:1004-1014`). So a text diff proves nothing, and neither does a
cross-process tensor diff. The gate is four oracles, each of which has
detection power a text diff lacks.

**O1. Byte idempotence. Deterministic, no forward pass.**

1. Save blob `A`.
2. `seq_rm`, then `set_data(A)`.
3. Save blob `B`.
4. Require `A == B` byte for byte.

This exercises the D2H and H2D paths on SYCL, and the indexer cell mirroring.
It is upstream Test 8 (`tests/test-save-load-state.cpp:452-505`) run on the real
backend and real model. It catches serialization and copy bugs. It cannot catch
a blob that is internally consistent but wrong.

**O2. Position alignment. Deterministic, no forward pass.** The guard from
§2.4, logged on every save and every restore. It catches the sglang#39830 and
LMCache class of bug at the bookkeeping layer.

**O3. Logits divergence against a same-process noise floor, with a mutation
control.** This is the numerical oracle. It is a new libllama harness,
`staging/work/state_restore_equiv.cpp`, written in the style of
`staging/work/qsa_pool_equiv.cpp`. All in one process and one context:

1. Prefill a fixed prompt `P` (8K tokens in Arm 2). Save blob `A`.
2. **Arm `orig`:** teacher-force a fixed continuation `S` (64 tokens: filler
   text, then a question about `P`). Record the logits at every position of
   `S`.
3. **Arm `rest`:** `seq_rm`, then `set_data(A)`, then teacher-force the same
   `S`. Record the logits.
4. **Arm `fresh`:** `seq_rm`, prefill `P` from scratch, then teacher-force `S`.
   Record the logits. This is the noise floor: the same computation run again,
   with no restore involved.
5. **Arm `shift`, the mutation control:** `seq_rm`, then `set_data(A)`, then
   teacher-force `S` **starting one position late** (`pos_next + 1`). This
   deliberately recreates the off-by-one failure.
6. For each arm, compute per-position KL divergence and top-1 agreement against
   `orig`.

**Pass:** `rest` stays within the `fresh` distribution: median KL at most about
2x `fresh`'s median, and top-1 agreement at least `fresh`'s minus 2 of 64. And
`shift` must be clearly separated from both.

**If `shift` is not clearly separable from `fresh`, the oracle has no power at
this depth,** and a pass on `rest` means nothing. That outcome must be reported,
not rounded to a pass.

(**Estimate, unverified**: a one-position shift perturbs every attention score
through RoPE and misaligns the recurrent update, so it should be far outside
run-to-run noise. The mutation arm exists precisely so that this claim is
measured, not assumed.)

**O4. QSA pooled-key exactness after a restore.** Set `LLAMA_QSA_POOL_CHECK=1`
(`src/models/qwen4exp.cpp:744-748`) inside the O3 harness. It builds the
from-scratch pooled keys as `indexer_k_ref` next to the cached `indexer_k` in
the same graph (`:911-916`). The harness's `cb_eval` compares the two for every
QSA layer on the first 4 ubatches after the restore.

This is the in-graph oracle that proved the pool cache exact
(`PLAN.md:1139-1147`), pointed at the one lifecycle event the pool cache has
never seen: a whole-sequence `state_read`. This env var does nothing through
`llama-server` on its own. It needs the harness.

**O5. End-to-end semantic check through the server. Scored as a count, not a
diff.**

- An 8K-token document with **three planted facts in the first 1,024 tokens**:
  distinctive names and numbers that appear nowhere later.
- After a persist and a restore, ask three questions in fresh strict-extension
  turns. Run 5 trials each at `temperature 0.7` with different seeds, so it is
  a proportion, not one sample.
- Control: the same questions against a fresh-prefilled server.
- **Pass:** the restored accuracy is within 1 of the fresh accuracy out of 15,
  and the fresh accuracy is at least 12/15. If the control itself can't
  answer, the question set is too hard and has no power.
- This is the sglang#39830 test shape: 0/10 against 10/10 was the failure
  signature there.

Where the oracles run:

| oracle | Arm 0 (CPU, synthetic `qwen4exp`) | Arm 1 (CPU, `Qwen3.6-35B-A3B`, server) | Arm 2 (B70, real model) |
|---|---|---|---|
| O1 | Test 8 + harness | through `.pcs` files: save, restore, save, `cmp` of main-blob regions | harness + server |
| O2 | harness | server logs, every save and restore | server logs |
| O3 | harness (checks the harness itself works; synthetic logits are meaningless as a quality signal) | harness on Qwen3.6, CPU | **harness at 8K, then at about 120K** |
| O4 | harness | n/a (no QSA) | harness |
| O5 | n/a | server | server |

### 4.3 Is upstream PR #26004 a prerequisite? No.

`gh pr view 26004`: OPEN, "server : preserve context checkpoints across slot
save/restore", updated 2026-09-26.

- It fixes the `/slots` save format, which calls `llama_state_seq_save_file`
  with tokens only (`server-context.cpp:2753-2768`). The restore side then
  does `slot->prompt.clear()` (`:2827`), which wipes the checkpoints.
- This plan never uses `/slots` save or restore. It persists
  `server_prompt.checkpoints` in its own format (§3.3), and restores them by
  moving the whole `server_prompt` into the slot, exactly as the RAM path does
  (`server-task.cpp:1862`).
- Checkpoints *are* relevant to this model (§1.3), but this plan carries them
  by construction.

Leave `/slots` alone. Stay aware that #26004 and #28092 both edit
`server-task.cpp` and `server-context.cpp` near the code this plan touches, so
the next upstream bump will need a hand merge.

### 4.4 The unified-KV restore ordering: a new risk, and its fix

**The mechanism**, every step read in this tree:

1. With `-kvu`, both slots share one cell pool of `n_ctx` cells for the tier.
2. `get_available_slot` (`server-context.cpp:1730-1842`) picks a slot. On the
   LRU path it calls `ret->prompt_save(...)` and then
   **`ret->prompt_load(...)` at `:1831`**. That restore's `find_slot` needs
   `cell_count` free cells.
3. The idle slot that is **not** being used still holds its conversation's
   cells at that moment. Idle-slot caching saves and **clears** idle slots only
   *after* the new task has launched (`:2619-2640`, `[TAG_IDLE_SLOT_CLEAR]`).
4. So restoring conversation B while conversation A is resident needs
   `n_A + n_B <= n_ctx`.
5. Example at tier 1 (262,144): A = 150K, B = 123K. The restore fails and
   `prompt_clear()` runs (`:1831-1833`). B then **reprefills all 123K tokens**,
   about 8 minutes. A is saved and cleared right afterwards anyway.

**Why it is worse with persistence.** Today this needs two deep conversations in
the RAM cache at once, which the 8 GiB cap rarely allows. After a wake with
disk persistence, it is the normal case: every persisted conversation is
available, and whichever resumes second runs into the first.

**Fix.** Task T4, about 40 LOC. In `get_available_slot`'s `update_cache`
branch, **before** `prompt_load`:

1. Identify the entry `load` would pick. Split `server_prompt_cache::load` into
   `find_best(tokens) -> iterator` and `restore(iterator, …)`, so the pick is
   visible before the restore.
2. Compute the need: `need = best->n_tokens`, and
   `free = n_ctx - Σ(other slots' prompt.n_tokens())`.
3. If `params_base.kv_unified && need > free`, run the existing idle-slot
   routine early, for the *other* idle slots only: `prompt_save` (or spill),
   then `prompt_clear()`, the same body as `:2625-2639`.

The end state is identical to what happens 40 lines later today. Only the
order changes, and only when it matters.

**Test.** Arm 1 step 6 exists specifically to exercise this: two conversations
whose sum exceeds the tier pool.

**Related, and left alone:** #28139 (the `f_keep = NaN` for a pinned empty slot,
doc 17 §5) at `:1780`. It is only reachable with an explicit `id_slot`, which
LiteLLM does not send.

### 4.5 Restore flow after any `load_model()`

1. `load_model` (`:1533-1541`):
   - No disk dir configured: on `is_resume`, **keep** the existing
     `prompt_cache`, and set `limit_tokens = n_ctx` for the new tier.
   - Disk dir configured: recreate the cache, then call
     `prompt_cache->scan_disk(dir, fp)`.
   - The scan reads each `.pcs` file's header and tokens, validates them, and
     pushes a **disk-only** `server_prompt_cache_state`: tokens, empty data,
     `disk_path` set, `n_bytes_disk` recorded. Checkpoints stay on disk.
   - The scan cost is bounded by the disk budget: about 23 × 2.4 MB of reads.
2. A request arrives. `ctx_tier_update` may reload again; §4.1 explains why the
   tier fits.
3. `get_available_slot` runs. After a reload every slot is empty, so the LRU
   path is taken and `update_cache` is true. `prompt_save` on an empty slot is a
   no-op (`:301-303`).
4. `find_best` uses the existing scoring, unchanged. Disk-only entries compete
   on tokens alone.
5. §4.4 pre-clear, if needed.
6. `restore(it)`. If the entry is disk-only:
   1. Read the checkpoints and main blob from the file into the entry's
      vectors.
   2. `llama_state_seq_set_data_ext`.
   3. On `n != size`: free the vectors, keep the entry disk-only, return
      `false` (full reprefill, as today).
   4. On success: `prompt = clone(entry.prompt)`. **Clone, not move** (today's
      `:1862` moves), because the disk-only index entry is kept.
   5. Free the RAM vectors, and set `last_used` in the file header. The header
      is rewritten in place, since it's fixed-size JSON padded to 512 bytes.
7. §2.4 alignment guard.
8. `update_slots` runs as today. `n_past = LCP` (`:3407`). A strict extension
   needs no checkpoint (doc 17 §2.4). Anything else searches the restored
   checkpoints (`:3537-3565`).

**Why the disk file is kept after a successful restore.** If the process
crashes before the next save, the older prefix on disk is still a valid restore
point. The next save of that conversation is a strict extension, so `alloc`'s
prefix rule (§3.4) deletes the old file.

**Logging, one `SRV_INF` line per event:**

- `pcache: persisted <tokhash> n_tokens=… bytes=… t_ms=… reason=…`
- `pcache: restored <tokhash> from disk n_tokens=… bytes=… t_read_ms=… t_set_ms=…`
- `pcache: restore failed (…), falling back to prefill`

These are what production monitoring greps for (§5, Stage 4). The existing
save and load lines are `SRV_TRC`, which is invisible at production log level
(doc 17 §2.6).

---

## 5. Staged rollout and validation gates

Every stage below is validated off to the side of production:

- **CPU-only**, `--device none` or no `/dev/dri`, in the `llm-test-sycl` compose
  service or `qwen4exp-moe-cache:sycl-pinned`, with `staging/devbin`
  bind-mounted.
- **Built with** `staging/work/devbuild.sh` (incremental, in the `gdnbuild`
  container).

Contention rules from `MEMORY.md` apply before every CPU arm: check `nproc`,
`free -h` (`available`), and the other tenants on the box. Artifacts go to
`logs/pcache/<arm>/`.

### Stage 0: harnesses and Arm 0. No feature code, no production risk.

- **T1:** write `staging/work/state_restore_equiv.cpp` (O1-O4 of §4.2) and its
  build line.
- **Arm 0** (doc 17 §7 command, unchanged):
  - `test-llama-archs -a qwen4exp` and `test-save-load-state`, `-dev none
    -ngl 0 -t 1`, in a container **with no `/dev/dri`**.
  - Add `test-state-restore-fragmented`, which is already in `staging/devbin`.
    It covers the `-kvu` interleaved-cell case (doc 17 §8 item 3) at the byte
    level.
  - Then run the T1 harness on the synthetic model. That checks the harness
    runs and that O1, O2 and O4 hold. O3 on synthetic weights only checks the
    plumbing, including that the `shift` arm separates.
  - **Status at time of writing:** a `staging/work/kvstate-arm0-build/`
    directory (CPU build, 16:38 today) and an empty `staging/work/kvstate/synth/`
    exist, so someone appears to be running Arm 0 in parallel. Their result was
    **not read and is not relied on** here.
- **Gate G0:**
  - Test 8 passes.
  - The fragmented test passes.
  - Harness O1 is byte-identical and O2 holds.
  - Harness O4 is bit-identical over every bias-selectable block. That is the
    same criterion as `PLAN.md:1139-1147`: rows past `ceil(n_cells/r)` are
    excluded.
  - The `shift` arm separates from `fresh`.

### Stage 1: carry the prompt cache in RAM across an in-process reload. No disk.

- Tasks: **T3** (alignment guard), **T4** (unified-KV pre-clear), **T5**
  (`persist_before_reload()` in RAM mode, the S1/S2 hooks, and keeping
  `prompt_cache` on `is_resume`).
- Scope: `server-context.cpp` only, plus a `server-task.{h,cpp}` split of
  `load()` into `find_best`/`restore` for T4.
- This is doc 16 §4.1 / doc 17 Option A', with the two guards added.
- It fixes the tier-switch and sleep/wake reprefill in-process. It does **not**
  survive a restart.
- Validated on Arm 1 steps 1-3 and 6 (below), with no disk flags.

### Stage 2: disk tier, spill on reload, lazy restore, startup scan

- Tasks: **T6** (flags), **T7** (fingerprint), **T8** (`.pcs`
  writer/reader), **T9** (disk-only entries in `server_prompt_cache`), **T10**
  (scan in `load_model`), **T11** (disk-mode spill in
  `persist_before_reload()` plus the S4 evict-to-disk).
- After this stage, sleep/wake and tier switches go through disk.
- A process restart **after a sleep** also resumes, because S2 already wrote
  everything and the new process scans the directory.

### Stage 3: graceful-shutdown save

- Task: **T12** (the S3 hook in `server_context::start_loop()`).
- Deployment note, applied in Stage 4: `stop_grace_period: 90s` in the compose
  override.
  - **Estimate:** two slots at 600K is about 11.6 GB, or 5-9 s to save
    (doc 17 §4.3). Add up to 8 GiB of RAM-cache entries at the 2.4 GB/s fsync
    rate, about 3.6 s. Add process exit.
  - 90 s is generous on purpose. The cost of the margin is only a slower
    `docker stop`.

### Arm 1: server mechanics, CPU-only, `Qwen3.6-35B-A3B` (after Stages 1-3; Stage 1 alone for steps 1-3 and 6)

Setup:

- Precedent: doc 13 §9.9. A GDN hybrid with no QSA indexer, so it tests the
  server bookkeeping, not `qwen4exp` specifics.
- Flags: `-ngl 0 --device none -t 4 -np 2 -kvu -ctk q4_0 -ctv q4_0 -fa 1`.
- Mirror production's shape: `--context-tiers 8192:512:4,16384:1024:8`,
  `--sleep-idle-seconds 8`, and the same `--rope-scaling yarn --rope-scale 1.0001
  --yarn-orig-ctx 17000` form.
- `--prompt-cache-dir /work/pcache/state --prompt-cache-disk-min-tokens 256`.
- Driven by T14's script and client. Filler comes from
  `staging/work/tier3_filler_6k.txt`.
- Requests go through `/v1/chat/completions`, because that is what LiteLLM
  sends. Record `timings.prompt_n` for each turn.

Steps:

1. **Tier switch.** Conversation A at about 6K tokens (tier 0), then a turn that
   pushes it past 8,192 (switch to tier 1).
   - Expect `prompt_n` about equal to the new turn's delta, **not** the whole
     conversation.
   - Expect a `pcache: restored` line and O2 aligned.
   - Before this plan, the same step reprefilled everything (doc 13 §9.9 turn
     3: `prompt_n` 1,539 of 1,539).
2. **Sleep and wake.** Idle for 8 s. Expect a `persisted … reason=sleep` line.
   Then a next turn with `prompt_n` about the delta. Log `free -h` across the
   reload.
3. **Answer check.** Ask about a fact planted in the first 500 tokens of A.
   That is O5 at small scale, 5 trials against a fresh-server control.
4. **Restart.** `docker stop` (expect `persisted … reason=shutdown` for each
   non-empty slot), then a **new container** with the same flags. Resume A.
   - Expect the scan to find A, a restore from disk, and `prompt_n` about the
     delta.
   - Do O1 by hand: save, restore, save, then `cmp` the main-blob byte ranges of
     the two `.pcs` files.
5. **Stale fingerprint.** Restart with `--yarn-orig-ctx 17001`.
   - Expect a new `<fp>/` directory, no restore, and a full prefill. That
     proves the RoPE field is in the fingerprint.
   - Restart with the original value. Expect the restore to work again.
6. **Two conversations over the pool (§4.4).** Build A to about 10K and B to
   about 9K at tier 1 (16,384 pool). Sleep, then wake.
   - Resume A, then B. **Expect both restores to succeed**: B's restore must
     pre-clear A.
   - Negative control: the same sequence on a build without T4. Expect B to
     reprefill fully. This demonstrates the bug on this tree.
7. **Divergence.** Resume A with its last assistant turn edited. Expect a
   checkpoint restore (`restored context checkpoint` at `-lv 4`) and a partial
   prefill, not a full one.
8. **Truncated file.** Truncate a `.pcs` file by 1 MiB and restart. Expect a
   warning, the file deleted, and a full prefill. No crash.
9. **SIGTERM mid-generation.** Kill during a long generation. On restart, resend
   the same request. Expect `prompt_n` of a few tokens (the checkpoint 4 tokens
   before the end of the prompt, §2.2), or a clean full prefill if the O2 guard
   refused the save. **Either outcome is a pass. A crash or a wrong answer is
   not.**
10. **Harness O3** on Qwen3.6 at 4K depth, CPU: `rest` within `fresh`, `shift`
    separated.

**Gate G1:** every step behaves as stated, and:

- no `GGML_ASSERT` or abort,
- O2 is never violated on a *restore*. A save-side refusal is acceptable, but
  log it and count it.
- host memory shows no monotone growth over 3 or more reload cycles,
- the step 6 negative control reproduces the bug. Otherwise the test isn't
  testing it.

### Arm 2: the real model on the B70, one production window (owner-scheduled)

Batch all of it into **one** stop of `llm-b70` (`MEMORY.md`: batch downtime).
Apply the doc 13 §9.9/§9.10 memory discipline:

- no other `llama` or SYCL process running,
- wait for the actual process exit, and check `free -h` before every load.

Run it **after** G0 and G1. Flags are production's: `-ngl 99 -fa 1 -ctk q4_0
-ctv q4_0 -lzm off -lm mmap+mlock --mlock-experts-only -fit off -np 2 -kvu`,
with the nearunity YaRN flags.

1. **Harness, no server.** O1 + O3 + O4 on the real model at 8K. Also record
   the first real D2H and H2D timings for the blob, which resolves doc 17 §8
   item 2.
2. **Server, cross-tier.** Tiers shrunk to `16384:2048:25,32768:2048:30` so a
   cross-tier restore happens at an 8K-20K depth. This tests the one hard-risk
   question the CPU arm can't: a restore under a different `n_ctx`, `n_ubatch`
   and **`-ncmoe`** on SYCL.
   - Then the Arm 1 steps 1, 2, 4 and 6 equivalents.
   - Plus O5: 3 facts × 5 trials, restored against fresh.
3. **Depth.** If 1 and 2 pass, repeat harness O3 once at about 120K, restoring
   into tier 0.
   - About 5 min of prefill plus about 1 s to restore.
   - The fresh-prefill control arm costs a second 5 minutes. Budget about
     15 min for this step.

**Gate G2:**

- O1 is byte-identical, and O2 holds on every restore.
- O4 is bit-identical over the selectable blocks.
- O3: `rest` within the `fresh` distribution and `shift` separated, at both 8K
  and 120K.
- O5 restored is within 1/15 of fresh.

**If O3 or O5 fails, stop.** Do not deploy. The feature stays behind a
default-off flag, and the finding becomes a bug hunt in the SYCL
`set_tensor`/`get_tensor` path or the hybrid restore.

### Stage 4: production deploy, flag-gated

Only after G2.

- Changes to `../coding-agent/docker-compose.qwen4exp-moe.override.yml` (T13):
  - add the `/state` rw volume,
  - add `--prompt-cache-dir /state`,
  - add `stop_grace_period: 90s`.
- Rebuild `staging/devbin`, regenerate `patches/llama.cpp.patch`, and recreate
  the container once.
- Watch for a week, in `docker logs llm-b70`:
  - the number of `pcache: restored` lines,
  - `prompt_n` on the first turn after each `exiting sleeping state` and each
    `switching context tier`. The pre-change baseline is 142,300 tokens and
    428 s (doc 17 §2.6).
  - any `restore failed`, and any O2 refusal.
- **Rollback:** remove the flag and recreate. The on-disk entries are inert
  without it.

### Stage 5: deferred until Stage 4 has run in production

- **Crash survival (S5).** An idle-persist timer: persist each idle, non-empty,
  changed slot once, after `--prompt-cache-idle-persist-s` (default 120) of
  queue idle. This clones `should_sleep()` (`server-queue.cpp:287-296`) as a
  second callback, the doc 13 §13 Option C pattern. It is much simpler than
  doc 17 §6.4's per-turn write-behind, and it covers the realistic crash
  window: a crash while a user reads a reply. The cost is 1-2 s of main-loop
  stall per 300K save, while idle.
- **Streaming I/O**, to remove the one-entry RAM peak. Write the main blob via
  a `llama_io_write_i` subclass over the `FILE*`, which would need a small
  public API, or via a GGSQ sidecar with `llama_state_seq_save_file` /
  `_load_file` (`src/llama-context.cpp:4212-4232`), which also streams on
  restore.
- **Drop the dead indexer V** from the blob: about 18% smaller (doc 17 §2.3).
  That is libllama code and a format bump.

---

## 6. Open unknowns, stated plainly

1. **SYCL numerical correctness of a restore at depth.** Still unknown. G2
   settles it. Everything before G2 is mechanism.
2. **Whether O3 has power on this stack.** It depends on the `shift` arm
   separating from `fresh` at 8K and 120K. If run-to-run noise at 120K is as
   large as a one-position shift (unlikely, **estimate**, but unmeasured), O5
   becomes the only semantic gate, and O5 is coarse.
3. **The idle-slot alignment invariant** (§2.4). It follows from reading, and
   Arm 1 logs it. If some slot state legitimately violates it, that state is
   refused, not persisted. No correctness risk, possibly fewer saves.
4. **Real D2H and H2D rates** for `ggml_backend_tensor_get/set` into pageable
   memory on this card. Unmeasured. Arm 2 step 1 records them. Doc 17 §4.3
   brackets them, and at most about 2 s at 600K is at stake.
5. **Page-cache effects.** A just-written 5 GiB file sits in the page cache,
   which counts toward the container's memory cgroup until it is reclaimed.
   Page cache is reclaimable, and it is not the anonymous memory that caused
   the reload incident. But `free -h`'s `available` should be watched across
   the first production sleep that spills. Using `posix_fadvise(DONTNEED)`
   after the fsync is a one-line mitigation if needed.
6. **How often `f_keep < 1` resumes need a checkpoint older than the newest
   4.** Doc 17 §8 item 6: the 0.788 and 0.741 rewinds in production. The flag
   is tunable. Count the misses in Stage 4.
7. **The upstream merge collision** with #28092 and #26004 at the next bump.
   This is maintenance cost, not risk.

---

## 7. Task breakdown for coding agents

Each task lists: scope, files, estimated LOC, dependencies, and how to test it
in isolation. LOC counts exclude comments.

| # | title | files | LOC | deps |
|---|---|---|---|---|
| T1 | State-restore oracle harness (O1-O4) | `staging/work/state_restore_equiv.cpp` (new), `staging/work/state_restore_equiv.sh` (new) | 280 + 40 | none |
| T2 | Arm 0 runner + report | `staging/work/pcache_arm0.sh` (new) | 50 | T1 |
| T3 | Alignment guard `slot_state_aligned()` and its save/restore call sites | `tools/server/server-context.cpp` (near `:300-341`, `:1824-1836`) | 40 | none |
| T4 | Unified-KV pre-clear before restore; split `load()` into `find_best`/`restore` | `tools/server/server-task.{h,cpp}` (`:1793-1868`), `tools/server/server-context.cpp` (`:1818-1839`, factor out `:2624-2640`) | 90 | none |
| T5 | RAM carry across reload: `persist_before_reload(reason)`, hooks at `:1118-1122` and `:985`, keep `prompt_cache` on `is_resume` at `:1533-1541` | `tools/server/server-context.cpp` | 60 | T3 |
| T6 | Flags `--prompt-cache-dir`, `--prompt-cache-disk-mib`, `--prompt-cache-disk-ckpts`, `--prompt-cache-disk-min-tokens` | `common/common.h` (near `:640-649`), `common/arg.cpp` (near `:1779-1800`) | 60 | none |
| T7 | Fingerprint builder (§3.2) + `PCS_COMPAT_EPOCH` | `tools/server/server-context.cpp` (static fn), `common/build-info.h` use | 80 | T6 |
| T8 | `.pcs` writer/reader: atomic write, fsync, header, trailer, validation, header-only scan read, in-place `last_used` update | `tools/server/server-task.{h,cpp}` (new `server_prompt_disk` section) | 280 | T6 |
| T9 | Disk-only entries in `server_prompt_cache`: `disk_path`, RAM-only `size()`/`n_tokens()`, lazy load in `restore()`, clone-not-move, prefix-rule file deletion in `alloc()`, evict-to-disk in `update()`, disk budget GC | `tools/server/server-task.{h,cpp}` (`:597-635`, `:1691-1892`) | 200 | T4, T8 |
| T10 | Directory scan in `load_model` | `tools/server/server-context.cpp` (`:1533-1545`) | 50 | T7, T9 |
| T11 | Disk-mode spill in `persist_before_reload`: slot→disk direct (bypassing the `alloc` size limit), RAM entries→disk, drop blobs | `tools/server/server-context.cpp` | 70 | T5, T8, T9 |
| T12 | Shutdown persist in `server_context::start_loop()` (`:4352-4355`), `!sleeping` guard | `tools/server/server-context.{h,cpp}` | 30 | T11 |
| T13 | Deployment: `/state` volume, flag, `stop_grace_period: 90s` | `../coding-agent/docker-compose.qwen4exp-moe.override.yml` | 6 | G2 |
| T14 | Arm 1 driver: server launcher + chat client + 10-step script + O5 scorer | `staging/work/pcache_arm1.sh`, `staging/work/pcache_client.py` (new) | 250 | T1-T12 |
| T15 | Arm 2 runbook script (one production window, pre-flight memory checks, harness + server steps) | `staging/work/pcache_arm2.sh` (new) | 150 | T14, G1 |
| T16 | *(Stage 5, deferred)* Idle-persist timer for crash survival | `tools/server/server-queue.{h,cpp}`, `server-context.cpp`, `common/{common.h,arg.cpp}` | 90 | Stage 4 in prod |
| T17 | *(Stage 5, deferred)* Streaming blob I/O | `server-task.cpp`, possibly `include/llama.h` + `src/llama-context.cpp` | 120 | Stage 4 |

**Totals.**

- Stages 1-3 feature code (T3-T12): about **960 LOC**.
- Harness and scripts (T1, T2, T14, T15): about **770 LOC**.

**Isolation tests per task.**

| task | test |
|---|---|
| T1 | Runs green on the synthetic model (G0). |
| T3 | Log-only first. Run any CPU server conversation at `-lv 3` and check that the guard reports aligned on every `prompt_save`. |
| T4 | Arm 1 step 6, with its negative control. |
| T5 | Arm 1 steps 1-2, disk flags off. |
| T6, T7 | Unit-style. `llama-server --help` shows the flags. Two launches differing only in `--yarn-orig-ctx` produce different `<fp>/` directories. Launches differing only in `-c`/`-ub`/`-ncmoe` produce the same one. |
| T8 | A round-trip of a synthetic `server_prompt_cache_state` (random bytes, tokens, 3 checkpoints) through write → read → compare, plus truncation at 5 offsets, each detected. Can live in `tests/` or as a harness mode. No model needed. |
| T9, T10 | Arm 1 steps 4 and 8. |
| T11, T12 | Arm 1 steps 2, 4 and 9. |

**Order for parallel agents.**

1. T1, T3, T4 and T6 have no dependencies, and can start at once.
2. T7 and T8 follow T6.
3. T5 follows T3.
4. T9 follows T4 and T8.
5. T10 and T11 follow T9.
6. T12 follows T11.
7. T14 needs everything.

Each agent should build with `staging/work/devbuild.sh <files>`, touch only its
listed files, and never run anything with `/dev/dri` mapped.

---

Sources (external): [llama.cpp PR #28092](https://github.com/ggml-org/llama.cpp/pull/28092) ·
[llama.cpp PR #26004](https://github.com/ggml-org/llama.cpp/pull/26004) ·
[llama.cpp#25913](https://github.com/ggml-org/llama.cpp/issues/25913) ·
[llama.cpp#28139](https://github.com/ggml-org/llama.cpp/issues/28139) ·
[sglang#39830](https://github.com/sgl-project/sglang/issues/39830) ·
[arXiv 2609.15030](https://arxiv.org/abs/2609.15030) (all via doc 17 §5, plus `gh pr view/diff` today for the two PRs).
