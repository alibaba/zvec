# Scalar filter integration checks

These tools call the public zvec collection API. The ordinary driver links no
diagnostic profiling symbols and can load the clean upstream or earlier
hardcoded comparator library. A second, optional driver reads query-level
internal activation counters from the new JIT-enabled library so a successful
result comparison cannot hide silent fallback.

Build the zvec shared library first, then build these tools independently:

```sh
cmake -S tools/scalar_filter -B build-scalar-tools \
  -DZVEC_BUILD_DIR="$PWD/build-scalar-jit" \
  -DSCALAR_FILTER_BUILD_INTERNAL_STATS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-scalar-tools -j 4
```

Set `SCALAR_FILTER_BUILD_INTERNAL_STATS=OFF` when building against an upstream
or JIT-disabled library. This omits only `scalar_filter_jit_benchmark`.

## Public API correctness

```sh
python3 tools/scalar_filter/run_smoke.py \
  --binary build-scalar-tools/scalar_filter_smoke \
  --lib-dir build-scalar-jit/lib \
  --output-dir build/scalar-filter-smoke-results
```

The output directory must be absent or empty. The runner creates small fixture
collections once, runs fresh processes with `ZVEC_SCALAR_JIT=0` and `1`, and
compares complete JSON records: ordered IDs, scores, every scalar value, and
group membership/order within each group. Group order itself is normalized by
group key. Each returned row is also checked against an independent predicate
and the original source values. NULL source cells must follow zvec's existing
query-output convention: scalar conversion leaves their `Doc` fields unset;
the JSON dump represents those verified NULL cells as `null`.

Fixtures cover:

- 384 persisted rows with a separately added `dataTime` column block.
- 2,304 rows and a maximum of 1,000 rows per segment, exercising multiple
  persisted segments and separately added scalar blocks.
- A persisted tombstone for document `200`, kept without optimization.
- A writable copy with 32 live rows inserted before querying and before flush.
  Persisted snapshots and mutable-store fallback can coexist in these queries.

The writable fixture uses `invert_to_forward_scan_ratio=0`; the read-only
fixtures and performance matrix use `0.5`. This avoids an independently
reproduced upstream schema bug: unsealed inverted indexes report a zero match
ratio, causing persisted and writing segments to choose different recall plans.
The forward callback plan includes internal `LOCAL_ROW_ID` while the inverted
plan omits it; `SegmentNode` assumes a common output schema. The zero threshold
keeps both segments on the forward callback path, allowing the test to exercise
persisted JIT plus mutable-store fallback without weakening result assertions.
The fixture still compares every scalar output and checks every returned row.

Predicates cover changed INT32 thresholds, nullable AND/OR/NOT IN, `IS NULL`,
INT64/DOUBLE/BOOL, string and IN expressions, mixed unsupported predicates,
selective inverted filtering, no filtering, and group-by vector search. There
are 128 result records per fixture per flag. The public correctness suite
proves equivalent behavior; it does not identify which individual expression
used JIT. Unsupported expressions may correctly fall back as a whole. The
current public SQL grammar does not accept general `NOT(expression)`; the
standalone compiler/bridge tests cover that node directly, while this public
suite uses the supported `NOT IN` syntax and checks its NULL semantics.

## Existing 10M workload

The driver reproduces the earlier query-vector generation exactly: normalized
62-dimensional FP32, cosine INT8 HNSW, top 21, ef 300, default seed 868. Reuse
the existing dataset read-only; do not rebuild it for each comparison.

```sh
python3 tools/scalar_filter/run_matrix.py \
  --binary build-scalar-tools/scalar_filter_benchmark \
  --jit-binary build-scalar-tools/scalar_filter_jit_benchmark \
  --path /path/to/random-10m \
  --integrated-lib-dir build-scalar-jit/lib \
  --baseline-lib-dir /path/to/baseline-lib \
  --hardcode-lib-dir /path/to/bound-int32-lib \
  --queries 4096 --pool 1024 --repeats 3 --threads 1,8 \
  --policies fixed,varying --include-unfiltered \
  --output-dir build/scalar-filter-matrix
```

The two older library arguments are optional. `--quality-only` runs exact
result comparisons and a one-query activation check per variant/policy without
the performance matrix. `--dependency-lib-dir DIR` can be repeated when shared
dependencies need an additional loader search path. Both binaries and every
library are SHA-256 recorded with the executed commands and environment.

- `fixed` uses `dataTime > 536870911`, the original approximately 75% matching
  workload.
- `varying` changes the threshold across the same query-vector pool. Thresholds
  lie between approximately 12.5% and 37.5% of INT32_MAX, keeping approximately
  62.5%–87.5% of the uniform source data. With ratio `0.5`, these predicates
  continue exercising forward filtering rather than mixing in selective
  inverted plans. Complete result dumps are compared across all libraries.
- JIT and Arrow variants use the same integrated library, switching only
  `ZVEC_SCALAR_JIT`. The hardcoded variant sets its two original feature flags;
  no profiling API is used.
- The JIT variant must report one successful binding per query and exactly one
  process-wide compilation, with zero additional timed compilations after
  warmup. `--expected-bindings-per-query` can explicitly adjust the expected
  populated-segment count for another dataset. Disabled and unfiltered variants
  must report zero bindings and zero cache lookups.

The default matrix uses 1,024 query vectors, 4,096 measured queries, and 1,024
warmup queries **per client**. All clients share a fixed ticket sequence, so
the measured query multiset is independent of thread scheduling. Variants are
interleaved; processes and performance runs execute sequentially.

`query_api_*` measures only `Collection::query`. QPS and full request latency
include scalar validation and result destruction. Collection opening, warmup,
quality dumps, and final counter aggregation are outside the timed interval.
The optional unfiltered baseline uses different acceptance semantics and is
reported only for the first threshold policy.

The lower-level `scalar_filter_benchmark build` command can create additional
synthetic datasets and refuses to overwrite an existing path. This is not
needed for the existing 10M case.
