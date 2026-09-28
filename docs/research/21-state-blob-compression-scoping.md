# 21 - Streaming compression for the `.pcs` state blob: measured, not assumed

Date: 2026-09-27. Status: **measurement and design only. No server code was changed, no GPU or
model-loading work was done, and the production `llm-b70` server was not touched.** All
compression numbers below are real `zstd`/`lz4` CLI runs (via `zstd --format=lz4`, since a
standalone `lz4` binary is not installed on this box) against real `.pcs` files left over from
today's Stage-0/Arm-1/Arm-2 validation passes for doc 18. CPU-only, single-threaded (`-T1`),
`nice -n 19`, on a box that was running several other tenants at the time (`nproc`=32, `free -h`
`available` was 72-74 GiB throughout, load average 5-8 - checked before and is not a confound
here since every run was single-threaded and the box had comfortable headroom).

Triggered by the user's own question at the end of doc 18: "a streamable compression format may
be worth it, too. That'll need testing." The `.pcs` format (doc 18 §3.3) and its writer/reader
already exist in the tree at `src/llama.cpp/tools/server/server-task.{h,cpp}`, `namespace
server_prompt_disk` (`server-task.h:600-686`, `server-task.cpp:1705-2226`) - this is not a plan
anymore, it is checked-in code, so §4 below cites it directly rather than the doc 18 sketch.

---

## 0. One-paragraph summary

**Not worth it on this deployment's hardware, as currently proposed.** The main state blob
(quantized attention KV + recurrent state + indexer KV) compresses only 7-22% depending on
which region dominates, and the recurrent-state checkpoint blobs - specifically the "always f32"
data the user's question was aimed at - are **empirically incompressible** (94-100% of original
size). Going past `zstd` level 1 buys almost nothing (level 9 is 0.3-1.3 percentage points better
than level 1; level 19 needs 100-180x longer for another 1-2.4 points) and the box's dedicated
Garudias NVMe (4.4-4.9 GB/s read, doc 17 §4.2) is fast enough that single-threaded `zstd` decompress
throughput (1.4-2.0 GB/s measured here) would become the new bottleneck on restore, trading a fast
disk read for a slower CPU-bound one. Compression would also land on the same main-server-thread
save path doc 18 §2.2 already flags as blocking HTTP threads for 1-2s per 300K-token entry, and
would make that worse, not better. The disk-budget GC added for doc 18 already bounds total usage
by a byte cap + LRU + age, and Garudias has 3.5 TiB free (doc 18 §3.1) - so there is no capacity
pressure motivating a 15-20% footprint cut either.

**Conditional case, stated honestly:** if this ever runs against a slower disk (the shared
bcache/RAID5 array, explicitly ruled out for state files in doc 17 §4.2, or any network storage),
or if the disk budget cap is ever set tight enough to matter, `zstd -1` on the main blob only
(never the checkpoint blobs) would be the one configuration worth revisiting - it is the only
point on the measured curve where the ratio gain (about 12-20 points) doesn't cost much CPU time
relative to level 3+.

---

## 1. What was measured, and on what

### 1.1 Real sample blobs, not synthetic data

Per the brief's instruction to avoid touching the GPU or `llm-b70`, no new run was started.
Instead, real `.pcs` files survived today's earlier validation passes for doc 18, found directly
under `staging/work/pcache/`:

| sample | file | fingerprint dir | n_tokens | file size | main blob size | 1 checkpoint (`data_tgt`) size |
|---|---|---|---|---|---|---|
| `main_small` / `small_89m` | `state_o5_control/580e1e96b636ef64/ee6b35ffcf6d0f0b.pcs` | `580e1e96b636ef64` | 4,096 | 89.6 MB | 89.56 MB | (0 checkpoints) |
| `main_mid` / `mid_353m` | `bug1_state/2b7c2e1db62b4aa5/bc0abdcb14ee006d.pcs` | `2b7c2e1db62b4aa5` | 4,128 | 353.2 MB | 89.74 MB | 65.86 MB x4 |
| `main_large` / `large_752m` | `arm2_state_fresh_control/8056c0576a9ab361/8342455380b7f4fe.pcs` | `8056c0576a9ab361` | 17,004 | 752.7 MB | 280.46 MB | 118.04 MB x4 |

**Important caveat, stated plainly:** the fingerprint directories differ between samples, which
means they are **not the same model**. `580e1e96b636ef64` and `2b7c2e1db62b4aa5` are Arm-1-style
CPU validation runs (doc 18's `Qwen3.6-35B-A3B` CPU arm, per the plan's Arm 1 setup); `8056c0576a9ab361`
is an `arm2_state*` directory name, i.e. the real target model. **Only `main_large` is confirmed
to be the real `Qwen3.8-Flash-Next` production model's blob shape** - the other two are a
different (also-hybrid) model at similar shallow depth. This confounds "model" with "depth" in
the small/mid vs. large comparison below; it is called out rather than glossed over. No sample at
production-realistic depth (100K-600K tokens) from the real model survived cleanup or was
available without touching the GPU, so the depth-scaling claim in §3 is an **estimate**, not a
measurement.

The exact `.pcs` binary layout was parsed directly against the real `write_entry`/`read_entry`
code (`server-task.cpp:1711-1712`, `1963-2017`, `2103-2107`) rather than assumed from doc 18's
sketch, to slice out clean regions:

- `MAGIC 'LPCS'(4) + format_version(4) + header_len(4) + JSON header padded to 512 bytes` (`HEADER_PAD_SIZE`, `server-task.h:611`)
- `u64 tokens_len + tokens`
- `n_ckpt x { i64 n_tokens, i32 pos_min, i32 pos_max, u64 len_tgt+data_tgt, u64 len_dft+data_dft, u64 len_spec+data_spec }`
- `u64 main_size + main blob` (attention KV + recurrent state + indexer KV, `FLAGS_NONE`, doc 18 §2.1)
- `u64 drft_size + drft blob` (empty in every sample; no draft model in these runs)
- `u64 total_len + TRAILER 'SCPL'(4)`

`data_tgt` in a checkpoint is the recurrent-state-only serialization (`PARTIAL_ONLY`, doc 18
§2.1), so it is a clean, already-isolated sample of "the recurrent state" without needing to guess
byte ranges inside the combined main blob - this answered §3's per-section question directly
instead of needing a harder isolation effort.

### 1.2 Tools

`zstd` 1.5.7 CLI and `libzstd1`/`libzstd-dev` are already installed; no standalone `lz4` binary is
installed, but `zstd --format=lz4` uses the same installed `liblz4.so.1` and was used as the lz4
comparison point (`fast` = `-1`, `best` = `-12`, lz4's actual level range). All runs: `nice -n 19`,
`-T1` (single thread, to get a clean per-core throughput number uncontaminated by the other
tenants' core usage), compress-then-decompress timed separately with wall-clock (`date +%s.%N`
around each `zstd` invocation, not `zstd -b`'s benchmark mode - that mode's `-b1 -b3 -b9 -b19`
syntax turned out to only honor the last `-b#` flag (`-e#` is the "test up to level" flag), so a
manual compress/decompress/`stat`-size loop was used instead for unambiguous per-level numbers).

---

## 2. Results

### 2.1 Full table (single-threaded, this box, real data)

| region | orig size | method/level | compressed size | ratio (compressed/orig) | compress throughput | decompress throughput |
|---|---|---|---|---|---|---|
| main_small (4,096 tok, CPU-arm model) | 89.56 MB | zstd-1 | 83.49 MB | 93.23% | 683 MB/s | 1564 MB/s |
| | | zstd-3 | 83.25 MB | 92.95% | 524 MB/s | 1392 MB/s |
| | | zstd-9 | 83.22 MB | 92.92% | 350 MB/s | 1593 MB/s |
| | | zstd-19 | 82.67 MB | 92.30% | 4.4 MB/s | 1380 MB/s |
| | | lz4-fast | 89.32 MB | 99.74% | 1359 MB/s | 1797 MB/s |
| | | lz4-best | 88.20 MB | 98.48% | 57 MB/s | 1980 MB/s |
| main_mid (4,128 tok, CPU-arm model) | 89.74 MB | zstd-1 | 83.97 MB | 93.57% | 714 MB/s | 1699 MB/s |
| | | zstd-19 | 83.15 MB | 92.65% | 4.3 MB/s | 1480 MB/s |
| **main_large (17,004 tok, real production model)** | **280.46 MB** | **zstd-1** | **226.20 MB** | **80.65%** | **805 MB/s** | **2011 MB/s** |
| | | zstd-3 | 224.06 MB | 79.88% | 563 MB/s | 1843 MB/s |
| | | zstd-9 | 223.35 MB | 79.63% | 291 MB/s | 1903 MB/s |
| | | zstd-19 | 219.61 MB | 78.30% | 4.4 MB/s | 1473 MB/s |
| | | lz4-fast | 247.43 MB | 88.22% | 1336 MB/s | 2088 MB/s |
| | | lz4-best | 239.77 MB | 85.49% | 57 MB/s | 2039 MB/s |
| ckpt_mid (recurrent state only, `data_tgt`) | 65.86 MB | zstd-1 | 62.03 MB | 94.17% | 702 MB/s | 1643 MB/s |
| | | zstd-19 | 62.05 MB | 94.21% | 4.7 MB/s | 1432 MB/s |
| **ckpt_large (recurrent state only, real model)** | **118.04 MB** | **zstd-1** | **110.63 MB** | **93.72%** | **746 MB/s** | **1715 MB/s** |
| | | zstd-19 | 110.68 MB | 93.76% | 4.3 MB/s | 1457 MB/s |
| | | lz4-fast | 118.04 MB | **100.00%** | 1302 MB/s | 2034 MB/s |
| | | lz4-best | 118.04 MB | **100.00%** | 64 MB/s | 2011 MB/s |

Full CSV with every level (1/3/9/19, both methods, all 5 regions) is in the run's scratchpad;
the table above is the load-bearing subset.

### 2.2 The headline finding: the recurrent state does not compress

The user's question specifically flagged the recurrent state as a candidate ("always f32", per
doc 18, "possibly having more redundant/low-entropy structure than the quantized attention KV").
**Measured, it is the opposite.** The isolated recurrent-state checkpoint blob (`ckpt_large`,
118 MB of real Gated-DeltaNet + PLE-row f32 state) compresses to 93.7-93.8% of its original size
under `zstd` at *any* level from 1 to 19 - level 19, at 100x the CPU cost of level 1, buys nothing
measurable (93.72% vs 93.76%, i.e. it is not even monotonic, which is noise-level). Under `lz4` it
does not compress at all: 100.00%, meaning the compressed output is the same size as the input
(container overhead a wash). This is a real negative result, not a rounding artifact - it held
identically on both the CPU-arm sample (`ckpt_mid`, 94.17-94.22% across all zstd levels) and the
real-model sample.

**Why, most likely:** f32 recurrent state from a trained model's actual activations is highresolution
floating-point data with no format-level redundancy (unlike, say, a sparse or repeated-pattern
buffer). Byte-level compressors exploit repeated byte sequences and low-order-byte redundancy;
32-bit floats produced by real matrix arithmetic have close to full entropy in their mantissa
bits at this precision. This matches the general folk wisdom about "already-dense numeric data
doesn't compress" that the task brief itself warned against assuming - the brief was right to
ask for a measurement instead of trusting either intuition.

### 2.3 The main blob compresses better, and better at more tokens - but the two samples aren't comparable

The main blob (attention KV + recurrent + indexer, mostly `q4_0`-quantized attention data per
production's `-ctk q4_0 -ctv q4_0`) compressed to 92.3-93.6% on the two CPU-arm shallow samples
(4,096-4,128 tokens) and to 78.3-80.65% on the one real-model sample at 17,004 tokens. That is a
meaningfully better ratio at the larger sample, but **the two are different models** (§1.1's
caveat), so this could be a depth effect, a model/architecture effect (e.g. a different head
count or KV layout), or both. Only one clean-model comparison exists in this dataset (the two
CPU-arm samples at 4,096 vs 4,128 tokens, which are practically the same depth and both show
~93% either way), so nothing here isolates depth as the cause with the real model. **This is an
open unknown, not resolved by today's data** - getting a real second real-model sample at depth
would need either a GPU run (out of scope here) or a CPU-only run of the actual
`Qwen3.8-Flash-Next` weights at higher token count, neither of which was in budget for this pass.

Across every region and every model, the shape is the same and is worth stating plainly: **`zstd`
level 1 captures nearly all of the achievable ratio.** Level 9 improves on level 1 by 0.3-1.3
percentage points at 2-2.5x the CPU cost; level 19 improves on level 1 by another 1-2.4 points at
**100-180x** the CPU cost (4.2-4.7 MB/s vs. 683-805 MB/s) - a level 19 pass on the 280 MB
`main_large` sample alone took 60.5 seconds, single-threaded, for 2.35 additional percentage
points of ratio. That is not a usable trade at any depth this project cares about. `lz4` beats no
`zstd` level on ratio and only wins on raw compress speed (1.3-1.4 GB/s vs. zstd-1's 0.68-0.8
GB/s) - not relevant here since even zstd-1 is far faster than the disk and decompress rates that
actually gate the save/restore path (§3).

### 2.4 Estimated whole-blob ratio at production depth

At 300K tokens, doc 18 §2.1 puts the main blob at about 2.77 GiB and up to 4 checkpoints at about
112.6 MiB each (about 450 MiB total) - checkpoint size is driven by per-layer recurrent-state
size, not context length, so it does not grow with depth the way the main blob does (**inference**,
from the checkpoint sizes measured here being close to constant, 65.9-118.0 MB, while token counts
varied 4,096-17,004). Applying this run's measured ratios (main blob 80% at the deepest real
sample as the best available estimate for a large main blob, checkpoints 94% flat) as an
**estimate**:

- Main: 2.77 GiB x 0.80 = 2.22 GiB
- Checkpoints: 0.44 GiB x 0.94 = 0.41 GiB
- Total: 2.63 GiB of 3.21 GiB, i.e. **about 18% smaller overall** at 300K depth.

At the shallower 17K-token sample actually measured, checkpoints (472 MB total) outweigh the main
blob (280 MB) 63/37, so the file-level ratio there is only about 89% (11% smaller) - worse than
the 300K estimate, because the incompressible checkpoints dominate more at shallow depth. Both
numbers are well short of the kind of 2-4x reduction that would make this an obviously worthwhile
feature; they are both closer to "not much."

---

## 3. Why NVMe changes the answer here: decompression throughput vs. disk throughput

Doc 17 §4.2 measured Garudias (the dedicated NVMe this feature already writes to) at
**4.4-4.9 GB/s sequential read** and **2.4 GB/s buffered+fsync write**. This run's single-threaded
`zstd` decompress throughput on real `.pcs` data was **1.4-2.0 GB/s** across every level and every
region (decompress speed does not depend on the compression level used to produce the data, only
on the data itself - consistent across the table). That means on *this* disk, **decompression is
slower than just reading the uncompressed bytes**: reading a 2.77 GiB main blob at 4.4 GB/s takes
about 0.63 s; reading a compressed ~2.2 GiB version at the same 4.4 GB/s and then decompressing
it at, generously, 2.0 GB/s adds about 0.5-1.1 s of pure CPU time on top of a *shorter* disk read,
for a net restore time roughly equal to or worse than doing nothing. This is the opposite of the
usual "compression helps because disk is the bottleneck" argument - it only holds when
decompression throughput exceeds disk throughput, and on this specific dedicated NVMe it does
not. (On the shared bcache/RAID5 array, explicitly disqualified for state files in doc 17 §4.2
at "well under 2 GB/s, HDD-bound," this inequality would flip in compression's favor - noted as
the one condition under which this recommendation would change.)

The save-side cost compounds this. Doc 18 §2.2 already flags every S1/S2/S3 save as running on
the **main server loop thread**, blocking HTTP threads that are tokenizing (S1) or posting a task
(S2) for the save's duration - currently "about 1-2 s per 300K entry" from disk write alone.
Adding `zstd -1` compression at this run's measured 683-805 MB/s single-threaded rate would add
roughly 3.5-4 s of pure CPU time to compress a 2.77 GiB main blob, on the same blocking thread,
for a write that would then be *faster* by only the delta between writing 2.77 GiB and writing
~2.2 GiB at 2.4 GB/s (about 0.24 s). That is a net addition of **several seconds to an
already-acknowledged latency cost**, for a size reduction that (§2.4) does not relieve any real
capacity pressure (3.5 TiB free, byte-cap GC already in place).

---

## 4. If it were built anyway: integration sketch against the real code (not implemented)

Cited directly against the checked-in `namespace server_prompt_disk`
(`src/llama.cpp/tools/server/server-task.{h,cpp}`), not the doc 18 plan-stage sketch:

- **Where to wrap.** `write_entry()`'s main-blob write is exactly two calls,
  `wr_u64((uint64_t) main_size); wr(main_data, main_size);` (`server-task.cpp:2009-2010`). The
  natural change is: compress `main_data`/`main_size` into a buffer first, then write the
  *compressed* length as the on-disk `u64` field (this is already just a length-prefixed blob, so
  the field's meaning becomes "on-disk length" instead of "logical length" - no framing change
  needed) followed by the compressed bytes. Symmetric change in `read_entry()`
  (`server-task.cpp:2103-2104`): read the on-disk length, then decompress into a
  `out_main` buffer pre-sized from a new header field carrying the *logical* size.
- **The streaming argument, specifically.** `main_data`/`main_size` arrive at `write_entry()`
  already fully materialized as a single in-memory buffer (`prompt_save`'s
  `llama_state_seq_get_data_ext` output, per doc 18 §2.1) - so "streaming" here is not needed to
  avoid buffering the *input*. It matters for the **output** side: a one-shot
  `ZSTD_compress()` needs an output buffer sized `ZSTD_compressBound(main_size)` (roughly
  `main_size + main_size/256 + 512`, i.e. essentially a second full-size copy of an
  incompressible-looking blob) allocated *before* compression starts. `ZSTD_compressStream2`
  instead needs only a small (e.g. 1-4 MiB) output buffer, `fwrite`-ing each filled chunk through
  the same `FILE*` `write_entry()` already has open with its 4 MiB `setvbuf` (`server-task.cpp:1934-1935`).
  Given §2.4's blob sizes (2.77-5.42 GiB), avoiding a second same-size heap allocation on the
  main thread is the one genuine argument for the streaming API over one-shot `ZSTD_compress`,
  independent of whether compression is worth doing at all. The same argument applies in reverse
  on read: `ZSTD_decompressStream` can decompress directly into the pre-sized `out_main` vector
  while reading the compressed bytes off disk in chunks, instead of first buffering the whole
  compressed stream separately.
- **Interaction with the atomic-write protocol.** None, structurally. `write_entry()`'s
  `fflush` -> `fsync(fileno)` -> `rename` -> `fsync(dir)` sequence (`server-task.cpp:2019-2050`)
  operates on whatever bytes were handed to `wr()`/`fwrite` - compression is purely a transform
  applied before those calls, so the crash-atomicity story is unchanged. The one thing to get
  right: a `ZSTD_CStream` must be fully flushed (`ZSTD_e_end` on the last `ZSTD_compressStream2`
  call) *before* the existing `fflush(fp)`/`fsync` calls, or a truncated compressed frame would
  read back as valid-looking bytes that fail decompression instead of failing the existing
  length/trailer check - so the existing `if (!ok) { ...; std::filesystem::remove(tmp_path, ec); return false; }`
  path (`server-task.cpp:2029-2033`) needs the stream-finalize call folded into the `wr`-lambda's
  success condition, not added after it.
- **The header-only-scan property is preserved for free.** `read_header()`
  (`server-task.cpp:2132-2173`) never reads through the main blob at all - it reads the header and
  tokens, then seeks straight to `file_size - sizeof(uint64_t) - sizeof(TRAILER)` for the trailer
  (`server-task.cpp:2161`). That seek is agnostic to what is between the tokens and the trailer,
  compressed or not, so compressing only the main/checkpoint blobs (never the header or tokens)
  costs nothing in scan cost - this matches doc 18 §3.3's stated reason tokens come first in the
  file, and requires no change to preserve it.
- **`update_last_used()` is likewise unaffected** (`server-task.cpp:2175-2224`): it rewrites only
  the fixed 512-byte `HEADER_PAD_SIZE` header block in place and never touches the blob region.
- **Version/compat story.** Add a header field, e.g. `"main_codec": "zstd"` (omitted or `"none"`
  for uncompressed), read via `.value("main_codec", std::string("none"))` so an old file missing
  the key naturally decodes as uncompressed - this technically doesn't require bumping
  `FORMAT_VERSION` (`server-task.h:607`), since the JSON header is already a flexible schema and
  the binary framing (`u64 length + bytes`) doesn't change shape. Recommended anyway: bump
  `FORMAT_VERSION` to 2 for explicitness, in the same spirit as this project's hand-bumped
  `PCS_COMPAT_EPOCH` (doc 18 §3.2) - old v1 readers should refuse a v2 file outright rather than
  silently trying to interpret a compressed blob as raw KV bytes if the codec field is ever missed
  by a future edit. Old v1 files continue to read exactly as today; nothing about v1 reading needs
  to change.
- **Main design risk if built:** none of the above is hard to get right in isolation, but the
  *combination* of "compression only pays off above some size threshold" (§2.4, §3) and "the save
  runs on the main server thread" (doc 18 §2.2) means the feature would need a size-gated on/off
  switch (e.g. only compress blobs over some N GiB, or a flag, never unconditional) to avoid
  regressing the common case - and that threshold cannot be justified by anything measured in this
  pass, because the one real deep-model number this project actually needs (compression ratio and
  decompress throughput at 300K+ tokens on the real weights) was not measured here and would need
  either a GPU run or a same-scale CPU run this pass did not have budget for.

---

## 5. Recommendation

**Not worth building now, for this deployment.** Measured on real `.pcs` data from today's
validation runs, not synthetic data:

- The recurrent-state checkpoints - the specific region the user's question was aimed at - do not
  compress at all (93.7-94.2% with zstd at any level, 100% with lz4).
- The main blob compresses 7-22%, with the better end of that range only measured on one
  real-model sample confounded with depth (§2.3) - an open unknown, not a settled number.
- `zstd` level 1 captures essentially all the achievable ratio; levels 9 and 19 are strictly worse
  trades (more CPU for negligible extra ratio).
- The dedicated Garudias NVMe (4.4-4.9 GB/s read) is fast enough that this run's measured
  single-threaded decompress throughput (1.4-2.0 GB/s) would make restores CPU-bound instead of
  disk-bound - a regression on the read path, not an improvement.
- Compression would add several seconds of main-thread CPU time to a save path already flagged as
  a blocking cost (doc 18 §2.2), for a size reduction that doesn't relieve any real capacity
  pressure (3.5 TiB free, GC already bounds growth).

**Conditional trigger to revisit:** if this feature is ever pointed at a slower disk than
Garudias (the explicitly-disqualified shared bcache/RAID5 array, or any network-backed storage),
the decompress-vs-disk-bandwidth inequality in §3 flips, and `zstd -1` on the main blob only
(never the checkpoints, which gain nothing) becomes worth prototyping for real. Absent that,
building this now would trade real, already-accepted latency for a single-digit-percent disk
footprint win the budget doesn't need.

---

Sources (internal): `docs/research/17-kv-state-persistence-scoping.md` §2.1, §4.2 ·
`docs/research/18-conversation-state-persistence-plan.md` §2.1-2.2, §3.1-3.3 · direct reads of
`src/llama.cpp/tools/server/server-task.h:600-686` and `server-task.cpp:1705-2226` (checked-in
`server_prompt_disk` code) · real `.pcs` files under `staging/work/pcache/{state_o5_control,
bug1_state,arm2_state_fresh_control}/` · `zstd` 1.5.7 CLI (`--format=lz4` for the lz4 comparison,
since no standalone `lz4` binary is installed).
