# Inverted benchmarks

Benchmarks for zvec's inverted index using synthetic data.

Follow the [shared build setup](../README.md), then build from the repository root:

```bash
cmake --build build --parallel --target inverted_bench inverted_bench_test

build/bin/inverted_bench_test
```

## Run

The Python driver uses only the standard library. List the available profiles and cases:

```bash
python3 benchmarks/inverted/run_benchmarks.py --list
```

Run the INT64 suite, or select one profile and query case:

```bash
python3 benchmarks/inverted/run_benchmarks.py --type int64 --num-docs 100000

python3 benchmarks/inverted/run_benchmarks.py --type array_int32 --num-docs 100000 \
  --profile window_l8 --case all_2_adjacent
```

Choose `--num-docs 100000` or `10000000` explicitly; the value domain stays the same at either scale.
Without `--type`, both types run. `--profile` and `--case` accept comma-separated names;
case filters select only profiles containing those cases. `--list` also includes supplemental profiles,
which run only when selected explicitly.

Each dataset is built once per repetition and shared by its selected query cases.
Build, verification, and each query case run sequentially in separate processes.
Defaults are three repetitions, seed `20261115`, at least 0.2 seconds of warmup and 1 second
of measurement per case. Each interval completes whole query-pool passes. Override them with
`--repetitions`, `--seed`, `--warmup-seconds`, and `--measure-seconds`.

Use `--bin /path/to/inverted_bench` for another build and `--output /path/to/new-directory`
to choose the result location. Existing result directories are never overwritten.

## Data

| Type | Profile | Value domain | Input array length |
| --- | --- | ---: | ---: |
| INT64 | `uniform_k100` | 100 | — |
| INT64 | `uniform_k10000` | 10,000 | — |
| INT64 | `hotspot_k10000` | 10,000 | — |
| INT64 | `zipf_k10000` | 10,000 | — |
| ARRAY_INT32 | `random_l8`, `window_l8` | 1,000 | 8 |
| ARRAY_INT32, supplemental | `random_l1`, `random_l32` | 1,000 | 1, 32 |

Uniform values have balanced frequencies. Hotspot data assigns 80% of non-NULL documents to
10% of the values. Zipf uses exponent 1.0; some domain values may not appear, so reports include
the observed cardinality. Hotness ranks are shuffled across numeric values.
Hot/cold queries sample the highest/lowest observed document frequencies;
query pools can therefore differ between scales and are fingerprinted in reports.

Random arrays select distinct members independently per document. Window arrays use consecutive
members in a cyclic domain to create co-occurrence. Values are shuffled across contiguous document
IDs and inserted in ID order. Special-value positions are shared across distributions of the same scale.

All datasets contain 10% NULLs. Arrays additionally contain 10% empty arrays. Of the remaining
arrays, 20% contain half as many distinct members, each repeated twice: a length-8 input can be
`[a,a,b,b,c,c,d,d]`. Length means input elements, including duplicates; length 1 has no duplicates.

## Results

Results default to `build/results/inverted/<run-id>/`. `run.json` records the configuration
and status; `summary.json` aggregates repetition medians and ranges. Each profile/repetition
retains its index, logs, build and verification reports, and individual query reports.
A failed build, verification, or query stops the run without a successful summary.

- **Build:** insertion time, seal time, their combined elapsed time, and documents/second.
  Input preparation and opening are excluded. Seal includes flushing and compaction.
- **Query:** single-threaded, warm-cache mean nanoseconds/query and QPS, including the index call,
  result count, and result release. Full document-ID verification occurs before timing in another process.
  Reported repetition ranges are not per-request latency percentiles.
- **Disk:** index-directory file bytes after seal and close, bytes/document, and allocated bytes where
  supported. Logs and benchmark reports are outside this directory.
- **Memory:** process RSS at stage boundaries and process peak RSS where available. These include
  runtime and allocator overhead; build memory also includes prepared inputs. Query workers load
  query inputs without the full dataset. Peak RSS includes setup and warmup, and is not index-only memory.

Query reports retain actual hit counts. Range targets refer to non-NULL documents and may be
approximate; ordinary match ratios use all documents. Witness ALL queries draw members from
existing documents to guarantee hits, while global queries may legitimately have empty intersections.
NULL and empty arrays remain distinct, including for negative membership queries.
