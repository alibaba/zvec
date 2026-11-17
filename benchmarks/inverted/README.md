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
aggregates declared metrics into repetition medians and ranges, retaining case-specific
and verification fields alongside them. Each profile/repetition retains its dataset, index, logs,
build and verification reports, and query results. A failed phase stops the run without
a successful summary.

- **Build:** the sum of per-batch insertion times, seal time, their sum (`build_seconds`),
  and documents/second based on that sum. Data generation, dataset reads/writes, input encoding, and
  index opening are excluded. Seal includes flushing and compaction. File reads pause
  insertion between batches and affect caches, so results are not directly comparable
  with the previous continuous in-memory insertion loop.
- **Query:** single-threaded, warm-cache mean nanoseconds/query and QPS, including the index call,
  result count, and result release. Full document-ID verification occurs before timing in another
  process. Reported repetition ranges are not per-request latency percentiles.
- **Disk:** logical bytes of files in the index directory after seal and close, and bytes/document.
  Dataset files, worker logs, and benchmark reports are excluded.
- **Memory:** `rss_mib` and `peak_rss_mib` report process RSS at stage boundaries and process peak
  RSS in MiB (1 MiB = 1,048,576 bytes), or `null` when unavailable. These include
  runtime and allocator overhead. Build runs in a fresh process after prepare exits, so its
  peak RSS excludes data generation. Build snapshots bracket index opening through seal;
  data-file opening and batch-buffer allocation occur on the first batch read, after the
  baseline snapshot. Its peak still includes process startup, metadata loading, and read buffers.
  Query workers load query inputs without the full dataset; their peak includes setup and
  warmup. These are process-level measurements, not index-only memory.
