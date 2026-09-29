# Crash recovery tests

`checkpoint_recovery_test` exercises the public Collection API with both plain
indexes and two FTS fields. Each case runs a separate worker, waits for its
`SIGSTOP` checkpoint, then sends `SIGKILL` and reaps it. No destructor or implicit
close is allowed between the checkpoint and termination. Worker errors and
checkpoint timeouts fail the test and include the worker output.

## Coverage

- Acknowledged inserts before explicit flush, and inserts after successful flush.
- Sealing through `create_iterator()`, completed optimize, and automatic segment
  rollover at 1,000 documents.
- Three consecutive opens/replays terminated before close or flush.
- Update, delete, reinsert with the same primary key, and insertion through upsert,
  both before and after explicit flush.
- Deleting every document, flushing, optimizing, and reopening an empty collection.

Every recovered collection is checked against the expected primary keys, complete
field values, document count, deleted keys, exact Flat vector results, filtered
vector results, and (for FTS cases) matches in both text fields. It must also accept
insert/update/delete/upsert, flush, optimize, close, and reopen without changing
those results. Per-document statuses and intermediate mutation results are checked.
A recovered row is updated and verified again after flush/optimize/reopen.

The shared verifier also checks analytical L2 distances and small top-k ordering.
FTS fields have distinct terms, frequencies, and lengths; pristine single-segment
cases additionally check analytical BM25 scores.

`BlockRotationWriteSurvivesProcessKill` and `BlockRotationWriteSurvivesClose`
use a one-byte memory buffer, so every insert after the first rotates the
writing block: the write is logged, the block is checkpointed and its WAL
removed, then the write is applied to the next block. The last acknowledged
write must survive a SIGKILL or a clean close, and a later insert must not
reuse its document ID.

## Durability tests

`manifest_durability_test` and `power_loss_simulation_test` run
`storage_fault_worker` in child processes with the test-only
`storage_shim.c` library preloaded (`DYLD_INSERT_LIBRARIES` on macOS,
`LD_PRELOAD` on Linux; the shim is never linked into zvec). Every shimmed run
proves the shim was loaded through a handshake file, so a stripped or missing
preload fails the test instead of passing vacuously.

| Test | Scenario | Oracle |
| --- | --- | --- |
| `DamagedOnlyManifestFailsClosedWithoutDeleting` | The only manifest is truncated to 0 bytes, 1 byte, half, all but one byte, and to prefixes the decoder accepts. | Open recovers every document or fails; no segment directory or file is removed, and restoring the manifest recovers everything. |
| `DamagedNewestManifestFailsClosedWithOlderOnDisk` | The newest manifest is damaged and an older manifest survived (its unlink was lost), but not its WAL, which the newest checkpoint removed. | Open fails closed and deletes nothing, and recovers once the newest manifest is repaired. A control first checks that the older manifest alone lacks documents, so falling back to it would lose data. |
| `ManifestEnospcTest` | ENOSPC on every write to `manifest.*` during `flush()`. | `flush()` fails; reopening recovers all 128 acknowledged documents. |
| `SimulatedPowerLossTest` | Power loss after `flush()` returned, for insert, flush, block rotation, optimize, create/drop index, and flushed rows followed by rotation, segment rollover or optimize. | Every flushed document is present with correct fields and indexes, and the collection accepts further writes. |
| `AcknowledgementAfterFailedWalFsyncIsTruthful` | One EIO from `fsync` on the WAL during `flush()`, then more inserts and another `flush()`. | Whatever the collection acknowledged survives simulated power loss: all rows if the later `flush()` returned OK, the first half if the failing `flush()` returned OK. Refusing further writes or flushes is also acceptable. |

### Power-loss images

The shim records, in order, every successful barrier below the collection:
the whole file for `fsync`, `fdatasync` and `fcntl(F_FULLFSYNC|F_BARRIERFSYNC)`
(these also write back pages dirtied through mappings), the page-rounded file
range of each `msync(MS_SYNC)` together with the file size, which a range
sync also persists (partial unmaps and remaps are tracked), and
the directory entries for a directory `fsync`. It logs an inode only while it
holds its own buffered read-only descriptor on that inode, opened
independently of the application's (so flags such as `O_DIRECT` do not
apply) and verified with `fstat`; it keeps that descriptor open until exit
and reads contents through it, never through a path. When it cannot open
one, it records nothing, so the data counts as lost. The test likewise keeps every
baseline inode open, so no inode number is reused during a run. After the
worker exits without closing, the test builds crash images from the prepared
baseline (a cleanly closed collection, treated as durable) and those records:

- file data is the baseline contents (else empty) updated by the recorded
  barriers in order; pages written through a mapping and flushed only with
  `MS_ASYNC` are lost;
- an injected `fsync` failure models Linux dropping the dirty pages on a
  writeback error: each byte below the file size at the failure keeps its
  previously durable value (zero where there was none) until a later write
  covers it and a later barrier succeeds; the shim logs writes and truncates
  of such files for this;
- directory entries not covered by a directory `fsync` may or may not have
  persisted: images are built with all such changes applied, none applied,
  and, for each change to a `manifest.*` or `*.wal` entry, that change alone
  applied and alone reverted. A rename is one atomic change.

`PowerLossModelTest` checks the image builder itself on hand-written
records: range syncs set the recorded size (including past a truncated end)
before applying bytes, and bytes dropped by a failed `fsync` stay lost until
rewritten and synced.

Each image must pass the `verify` oracle of the nightly storage tests.

Limits: the model assumes a successful barrier persisted exactly the bytes
the shim read at that moment, and it does not model reordering or torn
writes inside an fsync epoch, partial sector writes, or filesystem-specific
metadata ordering; namespace outcomes are sampled, not enumerated. Contents
are copied right after a barrier returns, so a concurrent write in that
window counts as synced. After an injected `fsync` failure, writes through a
mapping do not restore dropped bytes. On macOS, an `fcntl` command the shim does not know
stops the worker instead of being forwarded with a guessed argument type. Only
barriers issued through libc calls are seen. The Linux dm-log-writes replay in
the nightly storage job therefore remains the release gate; these tests are a
fast, unprivileged approximation that runs on macOS and Linux.

## Run

From the repository root, using an existing configured build directory:

```sh
cmake -S . -B build.release
cmake --build build.release --target checkpoint_recovery_test \
  manifest_durability_test power_loss_simulation_test -j 8
ctest --test-dir build.release \
  -R '^(checkpoint_recovery|manifest_durability|power_loss_simulation)_test$' \
  --output-on-failure
```

`checkpoint_recovery_test` cases are process-crash tests on POSIX platforms
supported by the existing crash recovery suite. They do not simulate power loss,
failed filesystem operations, or termination inside an in-progress
flush/manifest commit/WAL replay. In particular, acknowledged writes surviving
SIGKILL do not establish power-loss durability; the durability tests above and
the nightly block-level replay cover that.

See [nightly storage stability](../../../scripts/stability/README.md) for ENOSPC,
EIO, and block-write replay tests.
