# Persisting Conversation State to Disk Across Reloads and Restarts — Scoping

Date: 2026-09-27. Status: **scoped, NOT implemented, NOT validated on this
model's real weights.** No GPU or model-loading test was run for this document.
The only things measured were disk throughput (§4.2) and what the production
server's own logs already record (§2.6).

This extends `13-dynamic-context-aware-expert-placement.md`. That document
raised one question and left it open: the "Optional extension" of §6 Option A
(`:738-748`), open unknown §7 item 3 (`:810-813`), and the §9.8 known
limitation (`:1108-1113`). Doc 13 asks whether `llama_state_seq_{get,set}_data`
can carry this model's hybrid state across an **in-process tier switch**, as a
round-trip through host RAM. This document asks the wider question the
deployment actually has: **can a conversation survive a full server restart**
(a sleep/wake reload, a crash, a redeploy, a `docker restart`) by going through
**disk**, so the next turn resumes in seconds instead of re-prefilling the
whole context?

Everything below was read in the working tree at `src/llama.cpp` (upstream HEAD
`6d9c82ea2` with this project's patch applied, i.e. the code in
`staging/devbin`). Line numbers are from that tree. Note that `include/llama.h`
is 5 lines longer than upstream because of this fork's `mlock_experts_only`
field (`:350-354`). So doc 13's `include/llama.h:874`/`:884` are **`:879`/`:889`**
here. A claim that joins facts from separate files, rather than restating one
file, is marked **inference**.

---

## 0. One-paragraph summary

**It is feasible, and the hybrid architecture is not the hard part.** Doc 13
worried about three kinds of state. All of it already serializes today, or does
not need to, and the fork already handles the one that does not.

- The 12 attention layers serialize through `llama_kv_cache`.
- The 36 gated-DeltaNet layers (with the PLE conv row) serialize through the
  generic `llama_memory_recurrent`.
- The QSA **indexer** KV cache is written and read by upstream's own
  `llama_memory_hybrid_idx::state_write/state_read`, which exists and has an
  upstream round-trip test for `qwen4exp`.
- The QSA **pooled-key** cache is indeed not serialized, which confirms doc 13's
  guess. It does not need to be: it is a derived cache, and the fork already
  drops it on every `state_read` (`llama-memory-hybrid-idx.cpp:457`) and rebuilds
  it from the serialized indexer keys on the next ubatch.

The binding gaps are in **`llama-server`**, not in `libllama`:

1. **Nothing saves automatically.** Save only happens through a manual
   `/slots/{id}?action=save` behind `--slot-save-path`, which production does
   not set.
2. **The on-disk slot format drops the context checkpoints.** A hybrid model
   needs these to rewind its recurrent state. So any resume whose next prompt is
   not a *strict* extension of the saved token stream falls back to a full
   reprefill. This is upstream issue #25913, and fix PR #26004 is still unmerged.
3. **Nobody has checked** that the round-trip is *numerically* right on SYCL at
   real depth.

Measured disk rates put a 300K-token state (about 2.77 GiB) at **about 1-2 s to
save and about 1-2 s to restore** on the Garudias NVMe. The reprefill it
replaces is 19 min 57 s at 300K (`PLAN.md:855-856`). The production log records
**428 s** to reprefill 142,300 tokens after an ordinary sleep/wake (§2.6).
A first working version needs no ggml and no model code: server-only, about
250-400 LOC, 3-5 days including validation, MEDIUM risk. It covers surviving
sleep/wake and graceful restarts, for both production slots. Surviving a
**crash** needs a write-behind policy and costs more (§6.4).

---

## 1. What doc 13 claimed, and what this document checks

| doc 13 claim | where | verdict here |
|---|---|---|
| "~3.3 GiB at 300K … a few seconds through host RAM" | `:740-741` | **Size about 16% high.** The serialized state is 2.77 GiB at 300K (§4.1). Doc 13's figure matches the size *including* the pooled-key cache (1,536 B/token), which is not serialized. **"A few seconds" holds even through disk** (§4.3). |
| three kinds of state must round-trip at once | `:741-746` | **True, and all three have serialization code today** (§2.2). |
| QSA pooled cache "state serialization is almost certainly not implemented" | `:746-747` | **Confirmed not implemented, and correctly so.** It is derived state, and it is invalidated on restore (§2.3). |
| "Add 150-250 LOC and move to HIGH risk" | `:747-748` | **The libllama half of that risk mostly goes away.** The risk moves to server bookkeeping: tokens, checkpoints, and which blob goes to which slot (§3). That is the same class of bug that has hit vLLM, SGLang and LMCache on hybrid models (§5). |
| "2-4 days vs 2-3 weeks" hinges on the round-trip working | `:810-813` | The in-RAM tier-switch case is plausibly **under a day of code** on top of the existing host prompt cache (§6.3). It is still not validated. |

---

## 2. What exists in code today

### 2.1 The libllama state API

- `llama_state_seq_get_size/get_data/set_data` (`include/llama.h:874-893`), and
  `_ext` variants with flags (`:925-942`).
  - `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` (`:917`) means "only non-reconstructible
    partial state (SWA / recurrent)". This is what context checkpoints use.
  - `LLAMA_STATE_SEQ_FLAGS_ON_DEVICE` (`:921`) keeps the copy in VRAM.
- Per-sequence files: `llama_state_seq_save_file` / `_load_file` (`:895-909`),
  implemented at `src/llama-context.cpp:3252-3270` / `:3198-3250`.
  - Format: magic `GGSQ`, `LLAMA_STATE_SEQ_VERSION 3` (`include/llama.h:48-49`),
    then a u32 token count, the token ids, then the memory module's
    `state_write` stream.
  - **There is no model identity in a per-sequence file.** The whole-context path
    writes the arch string (`llama-context.cpp:3279-3281`). The per-sequence path
    (`:3318-3324`) writes nothing but the memory stream.
  - The per-layer checks on read are the only guard: KV type and row size, layer
    count, and recurrent type and row size (`llama-kv-cache.cpp:2536-2655`,
    `llama-memory-recurrent.cpp:1094-1126`). So **a blob restores silently into a
    different model with the same shapes**, for example another quant of this
    GGUF. Any persistence layer must key its files on model plus cache-type
    identity itself.
- File I/O: `llama_io_write_file::write_tensor` (`llama-context.cpp:2692-2696`)
  does one synchronous `ggml_backend_tensor_get` per contiguous cell range into
  a pageable `std::vector`, then `fwrite`. Read is the mirror image (`:2717-2721`).
  - No overlap between D2H and disk.
  - **No `fsync`, no temp-file-plus-rename.** A host crash or power loss in the
    middle of a save leaves a truncated file. That file does fail the size
    assertion on load (`:3245-3246`), but it is still a lost save.

### 2.2 Per-memory-type serialization for `qwen4exp`

`qwen4exp` builds a `llama_memory_hybrid_idx` (`src/llama-model.cpp:2601-2609`,
chosen by `needs_mem_idx` at `:2554`). Its write order is attention, then
recurrent, then indexer (`llama-memory-hybrid-idx.cpp:443-454`). The indexer
section goes last "so it is a pure suffix" (`:446`).

| state | owner | serialized by | notes |
|---|---|---|---|
| 12 attention layers, K+V, `q4_0` in production | `llama_kv_cache` | `state_write` `llama-kv-cache.cpp:2053-2121`, data `:2236-2333` | `-fa 1` means `v_trans = false` (`llama-model.cpp:2607`), so every layer is one contiguous range per cell run. That is a handful of large D2H copies, not per-element ones. |
| 36 gated-DeltaNet layers: conv `R`, SSM `S`, PLE conv `P` | `llama_memory_recurrent` | `state_write` `llama-memory-recurrent.cpp:766-845`, data `:897-990` | Generic path. `qwen4exp` reads it through `get_r_l`/`get_s_l` (`src/models/qwen4exp.cpp:1226-1227`). **The PLE conv row travels with `R`** (`:926-935`), so this model's extra recurrent side state is covered. Always f32, 112.57 MiB per sequence (`08-long-context-vram-budget.md:194-216`). |
| QSA indexer cache, 12 layers, one key head | a second `llama_kv_cache` (`mem_idx`) | `llama-memory-hybrid-idx.cpp:448-452` (write) and `:474-479` (read) | **Upstream code**, from `6c84c7d5d` (#27742) and `36b101543` (#27941). On restore it adopts the attention cache's cell layout (`state_read_sinfo`, `:462-469`), so the two caches cannot disagree cell for cell. A half-failed restore drops the sequence from all three caches (`:481-487`, `state_drop` `:490-506`). |
| QSA pooled block keys | `qsa_pool` buffers (fork only) | **not serialized** | See §2.3. |

**Upstream already tests this exact round-trip for this arch.** `36b101543`
added "Test 8: state blob round-trip" (`tests/test-save-load-state.cpp:452-505`).
It works like this:

1. Decode a prompt and save seq 0.
2. Erase it and restore it.
3. Save again and require the two blobs to be byte-identical.

The commit gave the synthetic `qwen4exp` fixture a PLE specifically so the test
"bites" (`tests/test-llama-archs.cpp`, same commit). This is a CPU test on a
synthetic model. It proves the format is complete, not that the restored state
decodes correctly on SYCL.

### 2.3 The pooled indexer-key cache: derived, invalidated, rebuilt

The pooled keys are "a function of the cells and their positions only"
(`llama-memory-hybrid-idx.cpp`, the comment above `init_qsa_pool`). That makes
them a cache over the indexer K cache, which *is* serialized.

The fork calls `qsa_pool_dirty()` (`:275-279`, which clears `qsa_pool_ok` and the
fingerprint) in several places:

- first thing in `state_read` (`:457`)
- in `state_drop` (`:491`)
- in `clear`, and in every `seq_rm/cp/keep/add/div`

After a restore, `qsa_prepare` therefore finds `qsa_pool_ok == false`, so
`n_valid = 0` (`:981-1001`). It then either rebuilds the needed window or, when
that is larger than `n_cap`, **rebuilds every block** (`n_upd = n_blocks`,
`:1014`). The `PLAN.md:1126-1127` addendum describes that as "exactly today's
cost for that one ubatch".

**Inference:** the restored conversation's first ubatch pays one full re-pool of
up to 76,800 blocks at 300K. That is milliseconds to seconds, not minutes.
Correctness rests on the same fingerprint-plus-drop lifecycle that was proven
exact by the in-graph oracle (`PLAN.md:1139-1144`). **Nothing new has to be
serialized for this cache**, and doc 13's "almost certainly not implemented" is
true but harmless.

One cost worth knowing: the indexer cache still allocates and serializes a V it
never writes. That is 12 x 256 elements, or 1,728 B/token at `q4_0`
(`08-long-context-vram-budget.md:150-184`), about **18% of every blob**. A
format that skipped it would be smaller. That is an optimization, not a
correctness issue.

### 2.4 `llama-server`: three existing mechanisms, none of which survives a restart automatically

1. **Host-RAM prompt cache** (`--cache-ram`, default 8,192 MiB,
   `common/common.h:644`).
   - `server_slot::prompt_save` (`tools/server/server-context.cpp:300-324`) calls
     `llama_state_seq_get_data_ext(..., FLAGS_NONE)` into a
     `server_prompt_cache_state`.
   - That state holds **the tokens, the context checkpoints, and the full state
     blob together** (`server-task.cpp:1779-1788`).
   - `server_prompt_cache::load` (`:1793-1870`) picks the entry with the best
     LCP against the incoming prompt and calls `llama_state_seq_set_data_ext`
     (`:1832`).
   - It only runs on the LRU slot-selection path (`server-context.cpp:1817-1836`).
   - **`load_model` recreates it** (`:1541`), so it does not survive a sleep/wake
     or a tier switch. This is doc 13 §9.8's limitation.
2. **Slot save/restore to disk** (`--slot-save-path`, `common/common.h:709`,
   endpoints enabled at `server-context.cpp:4972`).
   - `SERVER_TASK_TYPE_SLOT_SAVE` (`:2733-2782`) calls
     `llama_state_seq_save_file` with the slot's tokens.
   - `SLOT_RESTORE` (`:2783-2847`) loads the file, then does
     `slot->prompt.clear()` (`:2827`), which **discards checkpoints**, and
     reinstalls the tokens.
   - Both run on the main loop thread, so a save or restore of one slot stalls
     the other.
   - It saves `ctx_tgt` only, never `ctx_dft`. That does not matter here:
     production runs no speculative decoding.
3. **Context checkpoints** (`--ctx-checkpoints`, default 32, min step 8,192;
   `common/common.h:641-643`).
   - Each checkpoint is a `PARTIAL_ONLY` blob, meaning recurrent state only:
     about 112.57 MiB per sequence here. It is created near user-turn boundaries
     (`server-context.cpp:3738-3822`).
   - They exist because **a recurrent state cannot be rewound**. When the new
     prompt diverges before the end of the cached tokens, the server looks for a
     checkpoint at or before the divergence (`:3537-3565`). If it finds none, it
     forces `n_past = 0`, a full reprefill (`:3567-3572`).

**When does a resume work without checkpoints?** `pos_min_thold = pos_next -
n_swa - (has_new_tokens ? 0 : 1)` (`:3485`), with `n_swa = 0` for this model
(`:1379-1386`). The hybrid `seq_pos_min` is the max of the attention and
recurrent minimums (`llama-memory-hybrid.cpp:172-175`), which for the recurrent
side is the last position. **Inference** from those three lines:

- A restored slot needs **no** checkpoint when the next prompt is a *strict
  extension* of the saved tokens (`n_past == n_saved < n_new`).
- It needs one whenever the prompt diverges inside the saved tail, or is
  identical to it. The identical case comes from the `[TAG_PROMPT_LOGITS]`
  `n_past--` at `:3590-3595`.

Upstream #25913's repro is the identical-prompt case, which is why it reads as
"slot restore never works on hybrids".

The saved token stream and the memory positions do stay aligned. A sampled
token enters `prompt.tokens` only when it is added to a decode batch
(`server-context.cpp:531-542`), so the stop token is never in either. That is
exactly the invariant whose violation broke hybrid restores in vLLM/LMCache and
SGLang (§5).

### 2.5 `llama-cli` / `llama-completion` `--prompt-cache`

`--prompt-cache FNAME`, `--prompt-cache-all` and `--prompt-cache-ro`
(`common/arg.cpp:1930-1944`) are consumed only by `tools/completion/completion.cpp`
(`:206-219` load, `:945` save). They use the **whole-context**
`llama_state_{load,save}_file`: magic `GGSN`, version 10, which does carry the
arch string.

This is single-session and single-context, and it is not wired into
`tools/server/`. The server's per-slot equivalent is item 2 of §2.4. Nothing in
the CLI path would help the server beyond showing that the whole-context format
also round-trips.

### 2.6 What production already shows (read-only `docker logs llm-b70` / `docker inspect`)

- **Flags:** `-np 2 -kvu -ctk q4_0 -ctv q4_0 --context-tiers
  122880:2048:25,262144:2048:30,512000:1024:32 --sleep-idle-seconds 3600`. No
  `--slot-save-path`, no `--cache-ram` override (so 8 GiB), no speculative
  decoding.
- **The resume pattern is almost always a strict extension.** Of **242** LCP
  slot selections in the log, **237 show `f_keep = 1.000`**. The other five are
  0.999, 0.996, 0.993, 0.788 and 0.741. So the client (via LiteLLM) almost
  always re-sends the saved stream verbatim plus new tokens. A 76,544-token conversation then re-prefilled only
  **279 tokens in 3.1 s** per turn (task 17708). This is the case in §2.4 that
  needs no checkpoint.
- **A sleep/wake throws that away, and the cost is real.** After
  `exiting sleeping state`, the next request re-prefilled **142,300 tokens in
  428.4 s (7 min 8 s)**. The very next turn cost 331 tokens and 3.2 s. The log
  records **16 sleep entries and 10 wakes** across this container's history.
  Each one discards whatever conversation was live.
- **Weak evidence that the host-RAM round-trip already runs on this model and
  this backend.** 20 LRU selections, which is the path that calls `prompt_save`
  and `prompt_load`, produced no `failed to load prompt from cache` warning
  (`server-context.cpp:330`) and no `error loading state`. The actual
  save/load lines are `SRV_TRC`, invisible at production log level. So this
  shows **absence of failure, not proof of a correct restore.**

---

## 3. The actual gaps, ranked

1. **No automatic save/restore trigger** in `handle_sleeping_state`
   (`server-context.cpp:971-1001`), in the tier switch, or in the SIGTERM
   `shutdown_handler` (`tools/server/server.cpp:491-495`). Upstream closed the
   "auto-persist" request (#17107) as not planned.
2. **Checkpoints are not persisted** by the disk path (§2.4 item 2). This is
   upstream #25913, and fix PR #26004 is open as of September 2026.
   - Without it, a restored hybrid slot only resumes on strict extensions.
   - §2.6 says that is the common case here, which makes a checkpoint-less v1
     useful rather than broken.
   - The rare divergence then costs one full reprefill. That is exactly what
     happens every time today.
3. **No validated numerical round-trip on SYCL at depth.** Upstream Test 8 is
   CPU and synthetic. There are two problems with testing it here:
   - This stack's forward pass is not run-to-run reproducible (`PLAN.md:1004-1014`,
     `:1129-1138`), so "generate, restore, generate, diff the text" is not a
     valid oracle.
   - The repo's own CPU determinism pair (`staging/work/determinism-cpu-run{1,2}.txt`)
     also differs.

   The oracles in §7 avoid cross-run text comparison.
4. **File hygiene:**
   - no model identity in the file (§2.1)
   - no fsync or atomic rename (§2.1)
   - the production container has no writable persistent mount (`/models` is
     `ro`)
   - Docker's default 10 s stop grace period is tight for a 600K dual-slot save
     (§4.3)
5. **Multi-conversation bookkeeping.**
   - The `/slots` API is per slot index, not per conversation. The router does
     not pin slots, and production runs 2 slots.
   - The host prompt cache already solves "which blob belongs to this request"
     by LCP matching (`server-task.cpp:1793-1823`). That makes it the natural
     unit to persist (§6.2), not raw slot files.

---

## 4. Sizes and I/O time

### 4.1 Serialized bytes

At `q4_0` (0.5625 B/element), per token:

| part | size |
|---|---|
| attention K | 3,456 B |
| attention V | 3,456 B |
| indexer K | 864 B |
| indexer V (dead, never written) | 1,728 B |
| **total** | **9,504 B/token** (`08-long-context-vram-budget.md:150-184`, `:279`) |

Add **112.57 MiB per sequence** of recurrent state. The pooled-key cache
(1,536 B/token, F32) is **not** in the blob.

| depth | blob (GiB) | without dead indexer V (GiB) | reprefill it replaces |
|---|---|---|---|
| 76,544 (live prod conversation, §2.6) | 0.79 | 0.66 | several minutes (not measured at this depth) |
| 142,300 | 1.37 | 1.14 | **428.4 s**, measured in production after a wake (§2.6) |
| 300,000 | 2.77 | 2.28 | **19 min 57 s** TTFT (`PLAN.md:855-856`) |
| 600,000 | 5.42 | 4.46 | longer still (see doc 13 §5) |

**Optional checkpoint sidecars** (if persisted, §6.2): about 112.6 MiB each. A
76K conversation at the default 8,192-token minimum step holds up to about 9 of
them (about 1 GiB). Persisting only the newest 2-4 is enough for the
"last-turn edited" case.

### 4.2 Measured storage throughput (this box, 2026-09-27)

Production `llm-b70` was asleep and the box's IO pressure was about 4%. The
read used an unused 4.1 GB MTP GGUF. The write was a 4 GiB temp file, deleted
afterwards. Both ran on the Garudias NVMe.

| device | test | result |
|---|---|---|
| Garudias, Lexar NM790 4TB, PCIe 4.0 x4 (`/sys/class/nvme/nvme1`) | 4 GiB sequential read, `iflag=direct`, 64 MiB blocks | **4.4-4.9 GB/s** (two runs) |
| Garudias | 4 GiB sequential write, `oflag=direct` | **4.0 GB/s** |
| Garudias | 4 GiB buffered write, 1 MiB blocks, `conv=fsync` | **2.4 GB/s** |
| Golias, bcache over 6x HDD RAID5 (`md0`) with a CT2000P2 cache SSD **linked at PCIe 3.0 x2** (`nvme2`, 8 GT/s x2, so about 1.97 GB/s ceiling) | not tested: shared array, 89% full | Expect well under 2 GB/s, and HDD-bound once the cache is bypassed. **Do not put state files here.** |
| `/` Crucial P3 Plus 500 GB (QLC) | not tested | Root disk is 73% full. Unsuitable. |
| `/tmp` (tmpfs, 62 G) | n/a | RAM. Survives a process or container restart, not a reboot, and competes with the 64 GiB container cap and mlock'd experts. |

Caveats:

- The write tests used zeros. The NM790's controller is not known to compress,
  but this is a vendor-behaviour assumption.
- Sustained writes past the SLC cache were not tested. 5 GiB is far inside it
  on a 6%-full 4 TB drive.

### 4.3 End-to-end estimate

D2H/H2D for the KV ranges is **unmeasured**. `ggml_backend_tensor_get` goes into
pageable memory on SYCL. The repo's only PCIe numbers are **21.9 GB/s hot** and
**~3.1 GB/s cold** pinned-host reads (`05-hybrid-cpu-gpu-mul-mat-id-scoping.md:1021`,
`:1064`), and those are a different direction and mechanism. Using them as a
bracket, with the upstream code's fully serial D2H-then-write:

| depth | save (D2H + buffered write) | restore (read + H2D) | vs reprefill |
|---|---|---|---|
| 142,300 (1.47 GB) | 0.7-1.1 s | 0.4-0.8 s | 428 s: **about 400-1,000x** |
| 300,000 (2.97 GB) | 1.4-2.2 s | 0.8-1.6 s | 1,197 s: **about 550-1,500x** |
| 600,000 (5.82 GB) | 2.7-4.3 s | 1.6-3.2 s | — |

**"A few seconds" survives the move from RAM to disk.** It holds even at 600K,
even serial, and even at the pessimistic PCIe bracket. The QSA pool rebuild
and the one re-evaluated token come on top, both small (§2.3).

A file that was just written will usually still be in the page cache after a
container restart (not a host reboot), so restore may beat the disk figure. Two
slots at 600K is about 11.6 GB, or 5-9 s to save. **That exceeds Docker's
default 10 s stop grace period once the process exit is added**, so a
SIGTERM-time save needs `stop_grace_period` raised (§6.2).

---

## 5. What the rest of the ecosystem does

The recurrent-state half of this problem is **not unsolved** elsewhere. But
every stack that has solved it has shipped at least one silent-wrong-output bug
in exactly the place this design would add code.

- **vLLM** has a hybrid KV cache manager. Mamba/GDN prefix caching has two
  modes: `all`, which caches state at every `block_size` boundary, and `align`,
  plus a shared-prefix checkpoint option (tracking issue vllm#26201; partial-hit
  support in PR #46384).
  - **LMCache** stores the recurrent state "as an opaque page", so its disk,
    FileSystem and remote backends carry it with no model-specific code. Qwen3.5
    through 3.8 GDN hybrids are listed as validated.
  - LMCache itself warns that cached-vs-fresh generation "is not bit-exact"
    because GDN kernels are not batch-invariant, and that entries must not be
    shared across engines with different attention backends or block sizes.
  - arXiv 2609.15030 (GLM-5.3-Flash + vLLM + LMCache) found transfers that
    "succeed while a hybrid language model resumes from an inconsistent state".
    The restored state covered the full prompt "while the scheduler credited one
    fewer token". After the fix it reached 36/36 agreement.
- **SGLang** has `MambaRadixCache`, plus HiCache host/disk tiers for Mamba
  hybrids through `UnifiedTree`. **Open issue sgl-project/sglang#39830
  (2026-09-16): on Qwen3.8-Flash-Next, this model, a host-tier restore produces
  fluent but wrong answers, 0/10 vs 10/10 for cold and device hits.** The cause
  is a Mamba state restored at the wrong token position, and the report also
  notes that the host pool dropped the PLE side state.
- **llama.cpp upstream** has the pieces but not the automation:
  - `--slot-save-path` with manual `/slots` save/restore exists.
  - Auto-persist (#17107) was closed as not planned.
  - Hook-based tutorials (discussion #20572) and proxy front-ends do
    save-per-turn and restore-per-turn from the client side. They report about
    1 GB to 4.4 GB slot files for a 27B model up to 100K, and restore-side
    prefill falling from over 60 s to about 0.2 s.
  - #25913 documents that checkpoints are lost across save/restore on
    hybrid/recurrent models. PR #26004 appends them to the slot file (`SCKP`
    magic, backward compatible). It is independently confirmed on
    Metal/Vulkan/CUDA/ROCm, with 38.7x on Qwen3.5-35B, and it is **unmerged**.
  - #28139 (open): `f_keep` is computed as 0/0 = NaN when a request is *pinned*
    to an empty slot, which skips the host prompt cache. It is present in this
    tree at `server-context.cpp:1780`. It is only reachable with an explicit
    `id_slot`, which the router does not send.

**What this means here.** The generic mechanism exists. The failure mode to
design against is a **token/state alignment mismatch between the saved blob
and the server's token bookkeeping**. It is silent (fluent wrong text), and it
has been reported on this exact model on another stack. Any validation must
check *meaning* at a depth where the answer depends on the restored context,
not just "it didn't crash" or "prompt_n was small".

---

## 6. Design options

### 6.1 Option 0 — no code: use the existing `/slots` API by hand

Add `--slot-save-path /state/` and a writable Garudias bind mount. Before a
planned restart, `POST /slots/{0,1}?action=save`. After it,
`POST /slots/{i}?action=restore`.

- It works today for strict-extension resumes (§2.4 inference) and loses
  checkpoints.
- The router does not expose `/slots`, so it must be called on `llm-net`
  directly.
- A restart at 64 GiB with mlock can crash the box, so the reload-incident
  discipline applies (`MEMORY.md`, model-reload incident). Wait for the actual
  process exit and check `free -h` between loads.
- Useful as the **validation vehicle** (§7), and as a stopgap for planned
  redeploys only. It does nothing for sleep/wake, tier switches or crashes.

### 6.2 Option A (recommended v1) — persist the host prompt cache's entries across reloads and restarts

The unit to persist is `server_prompt_cache_state`: tokens plus checkpoints plus
the full blob (`server-task.cpp:1779-1788`). It is already the server's own
answer to "which saved state matches this request" (LCP selection, `:1793-1823`),
and it already carries checkpoints, so it closes #25913 by construction.
Upstream's #25913 thread names this unification as one of the two fix options.

The plan:

1. **On sleep entry** (`handle_sleeping_state`, before `destroy()`, `:985`):
   - `prompt_save` every non-empty slot into the prompt cache.
   - Write every cache entry to `--prompt-cache-dir`. Each file holds a header,
     the tokens, the newest N checkpoints (`PARTIAL_ONLY` blobs) and the main
     blob.
   - Write it as temp, then `fsync`, then `rename`.
   - Free the RAM copies. Sleeping is meant to free memory, which is why this
     goes to disk and not to RAM.
2. **On SIGTERM** (`shutdown_handler`): do the same, for a graceful
   `docker stop` or redeploy. Raise compose `stop_grace_period` to about 60 s.
3. **On `load_model`** (`:1541`): after creating `prompt_cache`, repopulate it
   from the directory. The first matching request then restores through the
   existing `prompt_cache.load` path, which lands on an empty slot via LRU. No
   new restore logic and no slot pinning.
4. **Header identity:** model file path, size and mtime (or GGUF hash);
   `type_k`/`type_v`; `LLAMA_STATE_SEQ_VERSION`; `n_embd` per layer; and the
   fork's build id. **Reject on any mismatch.** This must be explicit, because
   the blob checks only shapes (§2.1).
5. **Budget:** `--cache-ram` must hold the entries being restored. Two slots at
   300K is about 5.6 GiB, inside the 8 GiB default. Two at 600K is not.
   Restoring lazily from disk on LCP match would avoid holding everything in
   RAM at once. That is a v1.1 refinement.

Scope: `server-context.cpp`, `server-task.{h,cpp}`, `arg.cpp`/`common.h`. About
**250-400 LOC, 3-5 days including §7, MEDIUM risk.** The risk is not
serialization, which is upstream code already exercised by the prompt cache in
production (§2.6). It is the bookkeeping class of bug in §5, and the
`prompt_cache.load` path's interaction with the tier policy:

- An entry saved at tier 1 (262,144) restored after a wake at tier 0 (122,880)
  does not fit. `state_read_meta`'s `find_slot` fails, the restore throws, and
  the server falls back to a full reprefill. That is safe but slow.
- The wake-time `depth_hint` (`:988-995`) should therefore also consider the
  depth of the best-matching persisted entry. **Inference:** the restore itself
  is independent of `-ub`, `-ncmoe` and `n_ctx`, as long as the cells fit and
  the KV types match.

### 6.3 Option A' — doc 13's in-process tier switch, in RAM

The same mechanism without disk:

1. Before the tier switch's `destroy()`, `prompt_save` every non-empty slot.
2. Skip recreating `prompt_cache` when `load_model` is a resume
   (`is_resume`, `:1188`).

That is about **20-60 LOC**, or it falls out of 6.2 for free. It fixes doc 13
§9.8's first limitation. It still needs §7's validation.

### 6.4 Surviving a crash: write-behind

A crash gives no save opportunity, so state must already be on disk.

**Simple variant:** after each turn's `release()`, rewrite the slot's file
asynchronously. The D2H must happen on the main thread: about 0.3-0.6 s at 300K
by §4.3's bracket, stalling the other slot. The disk write can move to a
worker thread. That is plausible at production's measured cadence (turns tens
of seconds apart, §2.6), but it rewrites 2-5 GiB per turn at depth.

**Incremental variant:** attention and indexer KV rows for positions already
saved never change in this deployment (context shift off, strictly extending
turns). So a per-turn delta is only the new tokens' rows (about 9.5 KB/token)
plus the whole recurrent state (112.57 MiB). That is about 0.1 s per turn.

This needs a new append-friendly format and a range-limited `state_write` in
`llama-kv-cache`. It is libllama code, which the other options avoid.

**About +300-500 LOC, HIGH risk, 1-2 weeks.** Defer until v1 has proven the
round-trip.

---

## 7. Validation plan (NOT run; recommended next step, owner-scheduled)

Each arm uses oracles that avoid cross-run text comparison (§3 item 3).

**Arm 0 — synthetic `qwen4exp`, CPU, no GPU devices, no real model, about a minute.**

This covers all three state kinds (including PLE and the indexer) on this
fork's own binary. The container gets **no `/dev/dri`**, so it cannot touch the
B70.

**Unverified:** that `libggml-sycl` loads cleanly with zero devices. If it
does not, `--device none` inside the normal test container is the fallback,
per doc 13 §9.9.

```bash
S=/media/da3dsoul/Golias/AIProjects/Qwen3.8-Flash-Next-LlamaCpp-MoE-Cache-Arc-Experiments
docker run --rm --entrypoint /usr/bin/env \
  -v "$S/staging/devbin:/devbin:ro" -v "$S/staging/work:/work" \
  qwen4exp-moe-cache:sycl-pinned bash -c '
    export LD_LIBRARY_PATH=/devbin:$LD_LIBRARY_PATH
    mkdir -p /work/kvstate/synth &&
    /devbin/test-llama-archs -a qwen4exp -o /work/kvstate/synth &&
    /devbin/test-save-load-state --models /work/kvstate/synth -dev none -ngl 0 -t 1'
```

Pass criterion: Test 8 (blob idempotence) passes. Tests 3-7 compare generated
text. If they fail while 8 passes, check them at `-t 1` against a
fresh-vs-fresh noise floor before calling it a bug.

**Arm 1 — server mechanics on a small GDN hybrid, CPU, mirroring doc 13 §9.9.**

Use `Qwen3.6-35B-A3B` (a GDN hybrid). It has no QSA indexer, so it tests the
server bookkeeping, not `qwen4exp`. Run it with `--device none`, off to the
side of production:

```bash
cd "$S" && CID=$(docker compose -f docker/docker-compose.yml run -d --service-ports \
  --entrypoint /devbin/llama-server -v "$S/staging/devbin:/devbin" llm-test-sycl \
  -m /models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  -ngl 0 --device none -t 4 -c 16384 -np 1 \
  --slot-save-path /work/kvstate/slots/ --host 0.0.0.0 --port 8080)
```

Steps:

1. Send an 8K-token `/completion` prompt: filler from
   `staging/work/bench120k/prompt32k.txt` (3.52 chars/token, so about 28,800
   chars) plus a question only the filler answers. Set `"return_tokens": true`
   and record the generated token ids.
2. `POST /slots/0?action=save {"filename":"c.bin"}`.
3. `POST /slots/0?action=restore {"filename":"c.bin"}`, then `action=save` to
   `c2.bin`, then `cmp c.bin c2.bin`. This is the blob-idempotence oracle
   through the server.
4. `docker stop "$CID"`. Start a fresh container with the same flags and
   `action=restore c.bin`.
5. Send a **strict extension** as a token array: prompt tokens, then generated
   tokens, then `/tokenize` of a follow-up question about the filler.
   - Expect `timings.prompt_n` to be about the follow-up length.
   - Expect the answer to be correct from context.
6. **Negative control:** a fresh container with no restore and the same
   request. Expect `prompt_n` to be about 8K.
7. **Divergence case:** change one generated token. Expect a full reprefill,
   documenting #25913 on this tree.

**Arm 2 — the real model on the B70.** This needs `llm-b70` stopped and the
§9.9/§9.10 memory discipline (wait for actual process exit, check `free -h`
before the next load). **Only the repo owner should schedule it.**

Run Arm 1's steps with the production flags (`-ngl 99 -fa 1 -ctk q4_0 -ctv
q4_0 -lm none -lzm off -fit off`) at `-c 16384` and 8K depth. Add these
oracles:

- `cmp` of save/restore/save blobs, which exercises the SYCL D2H and H2D paths.
- `LLAMA_QSA_POOL_CHECK=1` on the post-restore turn. This is the existing
  in-graph oracle (`PLAN.md:1139-1144`), and it proves the rebuilt pooled keys
  are exact.
- A context-dependent question whose answer lives in the first 1K tokens.
  Getting that right is what catches a misaligned recurrent state (the
  SGLang#39830 failure).
- Log the save and restore `t_ms` and `n_bytes` from the `/slots` response.
  Those are the first real D2H/H2D numbers for §4.3.

If Arm 2 passes at 8K, repeat step 5 once at about 120K (tier 0) before
building anything.

---

## 8. Open unknowns, stated plainly

1. **Numerical correctness of the SYCL restore** on the real model, at any
   depth. Untested. It is the go/no-go for everything above, and Arm 2 settles
   it.
2. **D2H/H2D throughput** for `ggml_backend_tensor_get/set` into pageable
   memory on this card. Unmeasured, and bracketed at 3.1-21.9 GB/s from a
   different mechanism. It moves §4.3 by at most about 2 s at 600K, and does
   not change the conclusion.
3. **Unified-KV fragmentation with `-np 2 -kvu`.** Two interleaved sequences in
   one stream turn one D2H per layer into one per cell run. Upstream has a
   `test-state-restore-fragmented` binary (in `staging/devbin`), but its
   behaviour at realistic interleave on SYCL is unknown.
4. **Whether the host prompt cache's restore path has already produced correct
   restores in production.** The log proves only the absence of a logged
   failure (§2.6).
5. **Tier interaction.** Restoring an entry saved at one tier after a reload at
   another is safe by inference (§6.2) but untested. De-escalating with a live
   deep entry would force a reprefill.
6. **Checkpoint persistence depth.** How many sidecars are worth writing
   depends on how often a production turn actually diverges. The log suggests
   rarely: 5 of 242 LCP selections have `f_keep` below 1.000, and two of those
   (0.788, 0.741) are large rewinds that a newest-2-4 sidecar policy may not
   cover.

---

## 9. Recommendation

1. **Run Arm 0 now.** It is CPU-only, has no GPU devices, and takes about a
   minute.
2. **Let the owner schedule Arms 1-2.** Arm 1 is CPU, reuses §9.9's precedent,
   and runs in an hour or two. Arm 2 is the real model at 8K and needs a
   production window. Doc 13 §7 item 3 asked exactly this, and Arm 2 answers it
   for both the RAM and the disk case.
3. **If Arm 2 passes, build Option A (§6.2).** It persists prompt-cache entries
   (tokens, checkpoints, blob) on sleep and SIGTERM, reloads them in
   `load_model`, and does a strict identity check. It needs a writable Garudias
   bind mount and `stop_grace_period: 60s`. Server-only, about 250-400 LOC,
   3-5 days, MEDIUM risk. Option A' (the in-RAM tier switch) comes with it.
4. **Defer crash survival (§6.4)** until v1 has run in production.

The single biggest blocker is not the hybrid state. **It is that nobody has yet
shown a restored state produces the *same meaning* on SYCL at depth.** The
serialization code exists for all three state kinds. The disk is fast enough by
two to three orders of magnitude. The one failure the ecosystem keeps hitting,
a misaligned recurrent state that yields fluent wrong answers, has been
reported on this very model on SGLang. It will only be caught by a
context-dependent-answer oracle, never by throughput or `prompt_n`.

---

Sources (external):
[LMCache hybrid models](https://docs.lmcache.ai/mp/hybrid_models.html) ·
[vLLM hybrid KV cache manager](https://docs.vllm.ai/en/stable/design/hybrid_kv_cache_manager/) ·
[vllm#26201](https://github.com/vllm-project/vllm/issues/26201) ·
[arXiv 2609.15030](https://arxiv.org/abs/2609.15030) ·
[SGLang hybrid models blog](https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/) ·
[sglang#39830](https://github.com/sgl-project/sglang/issues/39830) ·
[llama.cpp#17107](https://github.com/ggml-org/llama.cpp/issues/17107) ·
[llama.cpp discussion #20572](https://github.com/ggml-org/llama.cpp/discussions/20572) ·
[llama.cpp#25913](https://github.com/ggml-org/llama.cpp/issues/25913) ·
[llama.cpp PR #26004](https://github.com/ggml-org/llama.cpp/pull/26004) ·
[llama.cpp#28139](https://github.com/ggml-org/llama.cpp/issues/28139) ·
[particula.tech on hybrid reprocessing](https://particula.tech/blog/prompt-reprocessing-swa-hybrid-models-kv-cache)
