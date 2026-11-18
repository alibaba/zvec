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
See the [example report](example-report.md) for a run with 10 million documents per workload.
For scripted runs, add `--help` to the Python command to see the available options.
