# Inverted benchmarks

Benchmarks for zvec's inverted index using synthetic data.

Follow the [shared build setup](../README.md), then build from the repository root:

```bash
cmake --build build --parallel \
  --target inverted_numeric_bench inverted_bench_test

build/bin/inverted_bench_test
```

## Code layout

```text
numeric_cases/        # INT32 and INT64 benchmarks and Python orchestration
shared/               # C++ helpers and Python process, JSON, and metric helpers
```

`numeric_cases/` owns its CLI, configuration, data generation, queries, and `run_benchmarks.py`.
C++ performs the benchmark phases and determines the queries for each workload. The Python
script enumerates workload combinations, runs repetitions, and aggregates reports.
Query cases are read from the build report. Python uses only the standard library.
`shared/` provides helpers without defining workloads or controlling suite execution.

See [Numeric benchmarks](numeric_cases/README.md) for configuration and commands.

Start reviewing a suite's header for configuration and entry points, then `main.cc` for its CLI.
The adjacent `*_test.cc` covers correctness and boundary cases; `run_benchmarks_test.py`
covers that suite's orchestration.

Run the Python tests independently:

```bash
python3 -m unittest discover -s benchmarks/inverted/numeric_cases -p 'run_benchmarks_test.py'

python3 -m unittest discover -s benchmarks/inverted/shared -p 'benchmark_helpers_test.py'
```

## Results

The script writes results under `build/results/inverted/numeric/`.
Use `--output /path/to/new-directory` to choose another location; existing directories
are never overwritten. `--bin-dir /path/to/bin` selects another build.

`run.json` records configuration, executable path and hash, and status. `summary.json`
aggregates declared metrics into repetition medians and ranges, retaining case-specific
and verification fields alongside them. Each profile/repetition retains its index, logs,
build and verification reports, and query results. A failed phase stops the run without
a successful summary.

- **Build:** insertion time, seal time, their combined elapsed time, and documents/second.
  Input preparation and opening are excluded. Seal includes flushing and compaction.
- **Query:** single-threaded, warm-cache mean nanoseconds/query and QPS, including the index call,
  result count, and result release. Full document-ID verification occurs before timing in another
  process. Reported repetition ranges are not per-request latency percentiles.
- **Disk:** index-directory file bytes after seal and close, bytes/document, and allocated bytes
  where supported. Logs and benchmark reports are outside this directory.
- **Memory:** process RSS at stage boundaries and process peak RSS where available. These include
  runtime and allocator overhead; build memory also includes prepared inputs. Query workers load
  query inputs without the full dataset. Peak RSS includes setup and warmup, and is not index-only
  memory.
