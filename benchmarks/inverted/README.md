# Inverted benchmarks

Benchmarks for zvec's inverted index using synthetic data.

Follow the [shared build setup](../README.md), then build from the repository root:

```bash
cmake --build build --parallel --target inverted_numeric_bench
```

## Code layout

```text
numeric_cases/        # INT32 and INT64 benchmarks and Python orchestration
shared/               # C++ helpers and Python process, JSON, and metric helpers
```

`numeric_cases/` owns its CLI, configuration, data generation, queries, and `run_benchmarks.py`.
C++ performs the benchmark phases and determines the queries for each workload. The Python
script enumerates workload combinations, runs prepare, build, verify, and query in separate
processes for each repetition, and aggregates reports.
Workload metadata and query cases are read from `dataset.json`. Python uses only the standard library.
`shared/` provides helpers without defining workloads or controlling suite execution.

See [Numeric benchmarks](numeric_cases/README.md) for configuration and commands.

## Results

The script writes results under `build/results/inverted/numeric/`.
Use `--output /path/to/new-directory` to choose another location; existing directories
are never overwritten. `--bin-dir /path/to/bin` selects another build.

`run.json` records configuration, executable path and hash, and status. `summary.json`
aggregates metrics selected by the numeric script into repetition medians and ranges,
retaining case-specific and verification fields alongside them. Each profile/repetition
retains its dataset, index, logs,
build and verification reports, and query results. A failed phase stops the run without
a successful summary.

Floating-point values in JSON reports use two decimal places; integer counts and query
values remain exact. Small measurements may round to `0.00`.

- **Build:** `insert_seconds` measures the complete insertion loop, including dataset
  reads, batch preparation, and index insert calls. `optimize_seconds` measures finalizing
  the index through `InvertedIndexer::seal()`, including statistical index generation,
  flushing, and compaction. `total_seconds` is their sum. Data generation, dataset writes,
  metadata loading, and index opening/closing are excluded.
- **Query:** single-threaded, warm-cache mean nanoseconds/query and QPS, including the index call,
  result count, and result release. Full document-ID verification occurs before timing in another
  process. Reported repetition ranges are not per-request latency percentiles.
- **Disk:** `index_file_mib` reports the total logical size of files in the index directory
  after optimization and close, in MiB (1 MiB = 1,048,576 bytes).
  Dataset files, worker logs, and benchmark reports are excluded.
- **Memory:** values report whole-process RSS in MiB (1 MiB = 1,048,576 bytes),
  or `null` when unavailable. Build reports `process_memory.peak_rss_mib` (the process
  peak through build completion) and `process_memory.final_rss_mib` (RSS after
  optimization, before closing the index). Build runs in a fresh process after prepare
  exits, so its peak excludes data generation but includes startup, metadata loading,
  read buffers, and allocator overhead. Query reports `rss_mib` and `peak_rss_mib`
  in `memory_before` and `memory_after`; its process peak includes setup, warmup, and
  earlier cases in the same query run.
