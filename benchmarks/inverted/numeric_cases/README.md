# Numeric inverted index benchmark

Benchmark zvec's inverted index on reproducible INT32 and INT64 workloads.

From the repository root, build the benchmark in Release mode.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_BENCHMARKS=ON

cmake --build build --config Release --parallel --target inverted_numeric_bench
```

Start the interactive runner in a terminal.

```bash
python3 benchmarks/inverted/numeric_cases/run_benchmarks.py
```

Choose a workload and review the configuration before selecting **Run benchmark**.
If your generator puts the executable in `build/bin/Release`, select **Edit configuration**
and set `bin_dir` to that directory.

When the run finishes, open the printed `report.md` path to see dataset statistics,
index build time and size, memory use, and query results.
See the [example report](example-report.md) for the PR #938 baseline comparison,
covering all 72 workloads at 100K documents and eight selected workloads at 10M
documents, with complete per-case query results.
For scripted runs, add `--help` to the Python command to see the available options.
