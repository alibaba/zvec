# Numeric inverted index benchmark

Benchmark zvec's inverted index on reproducible INT32 and INT64 workloads.

## Build and run

Run these commands from the repository root to build the Release executable.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_BENCHMARKS=ON

cmake --build build --parallel --target inverted_numeric_bench
```

Run the Python driver in a terminal to choose workloads interactively.

```bash
python3 benchmarks/inverted/numeric_cases/run_benchmarks.py
```

This example runs one workload with 100,000 INT64 documents and one repetition.

```bash
python3 benchmarks/inverted/numeric_cases/run_benchmarks.py \
  --data_type=int64 --num_docs=100000 --distribution=hotspot_moderate \
  --value_order=shuffled --null_ratio=0.1 --repetitions=1
```

Open the `report.md` path printed at the end. Results default to
`build/results/inverted/numeric/<run-id>/`. Use `--run_dir` to name a new output
directory or `--bin_dir` to point to the directory containing the executable (default `build/bin`).
With a multi-config generator such as Visual Studio, add `--config Release` to the
build command and set `--bin_dir` to the resulting directory, typically `build/bin/Release`.

## Choose a workload

| Python flag | Choices | CLI default |
| --- | --- | --- |
| `--data_type` | `int32`, `int64`, `all` | Required |
| `--num_docs` | `100000`, `10000000` | Required |
| `--distribution` | `uniform`, `hotspot_moderate`, `hotspot_extreme`, `all` | `all` |
| `--value_range` | `0_to_30000`, `0_to_type_max`, `all` | `0_to_30000` |
| `--value_order` | `ascending`, `shuffled`, `all` | `ascending` |
| `--null_ratio` | `0`, `0.1`, `0.5`, `all` | `0` |
| `--repetitions` | Positive integer | `3` |

Each combination is a separate workload. For example, selecting `all` for both
type and distribution runs six workloads when each other flag selects one value.
The value range starts at zero and excludes the upper bound (30,000 or the type
maximum). Value order controls how non-NULL values are assigned to document IDs.
Hotspot distributions concentrate records on a subset of values. The extreme
preset puts a larger fraction of records on fewer values.

## Read the report

Each repetition runs four C++ phases in separate processes. `prepare` generates
the dataset and query inputs from a fixed seed. `build` inserts and optimizes
the index. `verify` compares every matching document ID with a scan of the
dataset. `query` times all applicable queries after verification succeeds.
To run individual phases, see `build/bin/inverted_numeric_bench --help`.

| Report section | What to look for |
| --- | --- |
| Workloads | Configuration and repetitions, followed by observed NULL count, distinct values, value bounds, and documents per value. |
| Build | Mean insert and optimize time across repetitions. Insert timing starts after each batch has been read and decoded. Total is insert plus optimize. Index size is the logical size of files in the index directory. Peak and final RSS cover the build process through optimization, before the index closes. Sizes are in MiB. |
| Queries | Mean time per query in µs and mean matching documents per query, across inputs and repetitions. Each input runs 100 times on one thread. Timing includes search, result counting, and release. The index is opened before timing. |

Uniform workloads test equality hits (`eq_hit`). Hotspot workloads distinguish
the most frequent hot value (`eq_hottest`) from sampled hot and cold values
(`eq_hot`, `eq_cold`). All workloads test equality misses below zero,
inequality against existing values, NULL checks, and paired range queries.
For example, `lt_q11` targets the boundary with about 11% of **non-NULL**
documents below it. `gt_q11` uses the same boundary and counts values above it.

For individual measurements, open `build.json` and `query.json` in a workload's
`repeat-001/` directory. Its `dataset.json` records the query inputs in the same
order as the per-input `match_counts` in `query.json`. The run's top-level
`run.json` records the arguments, binary identity, and status. If a phase fails,
the driver prints an error summary and retains that phase's log.
