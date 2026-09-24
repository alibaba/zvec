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

## Run

From the repository root, using an existing configured build directory:

```sh
cmake -S . -B build.release
cmake --build build.release --target checkpoint_recovery_test -j 8
ctest --test-dir build.release -R '^checkpoint_recovery_test$' --output-on-failure
```

These are process-crash tests on POSIX platforms supported by the existing crash
recovery suite. They do not simulate power loss, failed filesystem operations, or
termination inside an in-progress flush/manifest commit/WAL replay. In particular,
acknowledged writes surviving SIGKILL do not establish power-loss durability.

See [nightly storage stability](../../../scripts/stability/README.md) for ENOSPC,
EIO, and block-write replay tests.
