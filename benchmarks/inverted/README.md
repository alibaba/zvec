# Inverted benchmarks

Benchmarks for zvec's inverted index using synthetic data.

Follow the [shared build setup](../README.md), then build from the repository root:

```bash
cmake --build build --parallel \
  --target inverted_numeric_bench inverted_array_bench inverted_bench_test

build/bin/inverted_bench_test

python3 -m unittest discover -s benchmarks/inverted -p 'run_benchmarks_test.py'
```

## Code layout

```text
numeric_cases/        # INT32 and INT64: CLI, configuration, data, and query cases
array_numeric_cases/  # ARRAY_INT32: CLI, configuration, data, and query cases
shared/               # Index, timing, verification, reporting, and random helpers
run_benchmarks.py      # Batch execution and result aggregation
```

Start with a suite's header for configuration and entry points, then `main.cc` for its CLI.
Numeric uses one `NumericProfile` for data type, document count, distribution, value range,
value order, and NULL ratio. `BuildNumeric`, `VerifyNumeric`, and `QueryNumeric`
handle the three phases. Array starts at `RunArray`.
The adjacent `*_test.cc` covers correctness and boundary cases.
Each suite owns its workload choices; `shared/` provides helpers without defining distributions
or query cases.

## Run

Each suite has an independent executable:

```bash
build/bin/inverted_numeric_bench --mode=list

build/bin/inverted_array_bench --mode=list
```

The numeric CLI accepts `--data_type=int32` or `int64`; listing without it includes both.
The array CLI always uses ARRAY_INT32. The Python driver uses only the standard library
and can orchestrate either or both executables:

```bash
python3 benchmarks/inverted/run_benchmarks.py --list
```

Select a suite, data types, or an individual case:

```bash
python3 benchmarks/inverted/run_benchmarks.py --type int64 --num-docs 100000

python3 benchmarks/inverted/run_benchmarks.py --type int32,int64 --num-docs 100000 \
  --profile uniform --case eq_hit

python3 benchmarks/inverted/run_benchmarks.py --type array_int32 --num-docs 100000 \
  --profile window_l8 --case all_2_adjacent
```

Numeric accepts `--num-docs 100000` or `10000000`. Array accepts any positive uint32 document count.
The `numeric` suite covers INT32 and INT64; `array` covers ARRAY_INT32.
Numeric runs require an explicit `--type`; use `--type all` to run all three types.
`--type`, `--profile`, and `--case` accept comma-separated names. `--list` also includes supplemental
profiles, which run only when selected explicitly.

Each dataset is built once per repetition and shared by its selected query cases.
Build, verification, and each query case run sequentially in separate processes.
Defaults are three repetitions, seed `20261115`, at least 0.2 seconds of warmup and 1 second
of measurement per case. Each interval completes whole query-pool passes.
`--repetitions` controls repetitions. Numeric fixes seed and timing in its implementation;
`--seed`, `--warmup-seconds`, and `--measure-seconds` overrides apply only to array runs.

Use `--bin-dir /path/to/bin` for another build and `--output /path/to/new-directory`
to choose the result location. Selecting a single suite or its types only requires that suite's
executable. Existing result directories are never overwritten.

Numeric workers take one `--run_dir` and derive `index/`, `build.json`, `verify.json`, and
`query-<case>.json` within it. Build receives the workload; verify and query restore it from
`build.json`. The Python driver handles these worker commands and paths.

## Data

Profiles provide defaults; overrides are resolved before selecting cases and recorded in reports:

```bash
python3 benchmarks/inverted/run_benchmarks.py --type int64 --num-docs 100000 \
  --profile hotspot --value-range zero_to_type_max --value-order ascending \
  --null-ratio 0.5 --hot-value-count 100 --hot-doc-ratio 0.8

python3 benchmarks/inverted/run_benchmarks.py --suite array --num-docs 100000 \
  --profile random_l8 --array-length 16 --empty-ratio 0.05
```

| Applies to | Configuration flags |
| --- | --- |
| Both suites | `--null-ratio` |
| Numeric | `--value-range`, `--value-order`, `--hot-value-count`, `--hot-doc-ratio` |
| Array | `--cardinality`, `--array-length`, `--empty-ratio`, `--duplicate-ratio` |

Select the relevant suite or types when using suite-specific options.

Numeric selects its distribution with `--profile uniform` or `--profile hotspot`.
Both default to range `zero_to_30000`, order `shuffled`, and 10% NULLs.

- **Value range:** `zero_to_30000` means `[0, 30000]`, including both bounds. `zero_to_type_max` means
  `[0, INT32_MAX]` or `[0, INT64_MAX]`. Sampling and statistics do not allocate the full value domain.
- **Distribution:** uniform samples every candidate value with equal probability. Hotspot defaults
  to 100 distinct hot values carrying 80% of non-NULL documents. Hot values are sampled across the
  domain; remaining documents sample cold values outside that set. Each group samples uniformly.
- **Value order:** `shuffled` distributes values randomly across document IDs; `ascending` assigns
  non-NULL values in nondecreasing order. Both use the same values and NULL positions.
- **NULL ratio:** numeric accepts only `0`, `0.1`, or `0.5`, producing zero, `num_docs / 10`, or
  `num_docs / 2` NULLs, rounded down. NULL positions are deterministic and randomly distributed.

Query operands are derived from the generated data. Only observed values are counted internally;
reports retain compact cardinality, frequency, and value-bound summaries with a frequency fingerprint,
rather than a per-value table. Large domains can produce many one-document postings.

| Array profile | Value domain size | Input array length |
| --- | ---: | ---: |
| `random_l8`, `window_l8` | 1,000 | 8 |
| `random_l1`, `random_l32` (supplemental) | 1,000 | 1, 32 |

Random arrays select distinct members independently per document. Window arrays use consecutive
members in a cyclic domain to create co-occurrence. Values are shuffled across contiguous document
IDs and inserted in ID order. Special-value positions are shared across distributions of the same scale.

Array defaults include 10% NULLs and 10% empty arrays. Of the remaining arrays, 20% contain half
as many distinct members, each repeated twice: a length-8 input can be `[a,a,b,b,c,c,d,d]`.
`--duplicate-ratio` applies to non-NULL, nonempty arrays. Length means input elements, including
duplicates; length 1 has no duplicates. Array configurations must retain non-NULL data.

## Results

Results default to `build/results/inverted/<run-id>/`. `run.json` records configuration,
executable paths and hashes, and status; `summary.json` aggregates declared metrics into repetition medians and ranges.
Case-specific details and verification fields are retained alongside these metrics. Each profile/repetition
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
