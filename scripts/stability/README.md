# Nightly storage stability

`.github/workflows/nightly_stability.yml` runs daily at 01:30 Asia/Shanghai and
supports manual dispatch on a selected branch. The workflow starts running on a
schedule after it is merged into the default branch. A supplied `seed` reproduces
process-test ordering and power-loss cut selection; `cuts` defaults to eight per
workload. Neither FTS failures nor infrastructure failures are skipped or marked
as expected failures.

## Jobs

| Job | Scope |
| --- | --- |
| Build | Release binaries, Linux preload library, and harness unit tests |
| Recovery | Three shuffled rounds of checkpoint, write, and optimize recovery suites |
| Disk full | Blocks and inodes exhausted separately during insert, flush, and optimize, with plain and FTS schemas |
| I/O errors | One-shot and persistent EIO on write-family calls during insert/optimize, and sync-family calls during flush, with both schemas |
| Power loss | Block-write recording for each operation/schema, then replay of eight sampled prefixes including both operation-boundary marks |

The last three jobs are independent matrix entries with fail-fast disabled. They
run even if the recovery job reports known FTS failures. A failing case is recorded
in JUnit, other cases continue, and the runner ultimately returns a nonzero status.
An operation that unexpectedly succeeds after an injected fault also fails its
case. The audit log must prove the fault actually reached the database's file
operations; filling the disk or merely loading the injector is not sufficient.

## Runner requirements

The default is `ubuntu-24.04`. Storage jobs require an unprivileged test user with
passwordless sudo, loop devices, ext4 mounting, and sufficient local disk space.
Power-loss recording additionally requires the `dm_log_writes` kernel module and
device-mapper access. If the hosted image does not provide these, preflight fails
explicitly. Select a compatible Ubuntu 24.04 x86-64 runner through the repository
variable `STABILITY_RUNNER`, containing a JSON string or label array, for example:

```json
["self-hosted", "Linux", "X64", "storage"]
```

Only newly allocated loop devices and uniquely named mapper targets are used.
Each case has a 128 MiB test filesystem; recording also uses a 256 MiB log image.
The database worker is never run as root. Successful images are removed; failed
images remain in uploaded artifacts. Cancellation attempts cleanup via SIGTERM;
a forcibly killed runner must be recycled or have its owned mounts/devices cleaned
before reuse. No existing host disks should be supplied to these scripts.

The macOS/Windows environment does not implement these Linux storage faults.
`storage_fault_worker` itself is portable to the existing POSIX test platforms,
which allows its protocol and verification to be tested independently.

## Oracle and durability boundary

The worker prepares 64 deterministic documents and successfully flushes/closes
them before fault injection. The operation attempts another 64 documents. For
flush/optimize, those writes occur before the READY checkpoint; insert performs
them after it. The parent arms the fault only after observing READY and SIGSTOP.
Every operation checks per-document statuses and ends with an explicit flush.

After a failure or an intermediate power-loss cut, all original 64 documents must
survive. Additional recovered documents may be present, but their fields and all
indexes must agree. At the completed-operation replay mark, all 128 documents are
required because the worker has acknowledged a successful flush.

Verification checks complete fields, counts, exact vector results, filtered
results, and both FTS fields. It then inserts, updates, deletes, and reinserts a
new primary key, flushes, optimizes, closes, reopens, and checks the results again.
Errors exit without collection destructors so that a cleanup flush cannot repair
the failed operation before the parent observes it.

## Fault model limits

- ENOSPC uses real filesystem exhaustion, not a mocked return value. The preload
  library audits ENOSPC returned from covered database open/write/sync calls.
- EIO interception covers `write`, `pwrite`, `pwrite64`, `fsync`, and `fdatasync`
  in this test process and only paths below the collection directory. It does not
  cover mmap writeback, direct syscalls bypassing libc, rename/unlink, or hardware
  device failures. The library is not linked into production targets.
- Power loss uses Linux `dm-log-writes` and the pinned upstream `replay-log` tool.
  Every cut starts from the same clean baseline; normal filesystem recovery runs
  when the reconstructed image is mounted. Later unmount writes from recording
  are excluded. This samples block-write persistence states, including FLUSH/FUA
  boundaries; it is not physical power cycling or an exhaustive enumeration of
  device reordering/torn writes. The recorded DURABLE mark is not preceded by an
  extra host sync that could hide a missing application flush.

## Reproduction and artifacts

The artifacts include the commit, run ID, seed, operation, injection audit, worker
output, command logs, JUnit, and selected replay limits. Failed disk-full/EIO cases
also retain `after-fault.img` from before database recovery. Failed power-loss
cases retain `baseline.img`, `writes.img`, and `cuts.json`; these reconstruct every
failed cut even though verification modifies the replay image.

Build the native workers using the project's normal CMake configuration:

```sh
cmake --build build --target checkpoint_recovery_test write_recovery_test optimize_recovery_test storage_fault_worker
cc -shared -fPIC -O2 -Wall -Wextra -Werror scripts/stability/io_fault.c -ldl -o build/lib/libstorage_fault.so
STORAGE_FAULT_LIBRARY="$PWD/build/lib/libstorage_fault.so" python3 -m unittest discover -s scripts/stability -v
python3 scripts/stability/run_recovery.py --build build --output artifacts/recovery --seed 759
python3 scripts/stability/run_storage_faults.py --mode disk-full --worker build/bin/storage_fault_worker --injector build/lib/libstorage_fault.so --output artifacts/disk-full
python3 scripts/stability/run_storage_faults.py --mode io-errors --worker build/bin/storage_fault_worker --injector build/lib/libstorage_fault.so --output artifacts/io-errors
python3 scripts/stability/run_storage_faults.py --mode power-loss --worker build/bin/storage_fault_worker --replay-log /absolute/path/to/replay-log --output artifacts/power-loss --seed 759 --cuts 8
```

Output directories must not already exist. The pinned replay-tool revision is
`7b70d8a6863c5de30933d42a7672d35d01d2dc6c`; its build steps are in the workflow.
To reproduce a failed power cut, copy `baseline.img` to a new disposable image,
attach only that copy to a new loop device, and run `replay-log --log writes.img
--replay <new-loop-device> --limit <cut>`. Mount it and invoke `storage_fault_worker
<mount>/collection verify plain|fts 64` (use `128` for the durable cut). Running the
verifier changes the image, so always start each attempt from a fresh copy.
