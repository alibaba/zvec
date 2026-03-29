# Benchmark Scripts

## Maintained Entry Points

| Script | Status | Purpose |
| --- | --- | --- |
| `benchmark_hnsw_vs_omega.py` | maintained | Generic JSON-driven HNSW vs OMEGA runner. |
| `benchmark_cohere_1m.py` | compatibility wrapper | Preset wrapper around `benchmark_hnsw_vs_omega.py --dataset cohere_1m`. |
| `benchmark_cohere_10m.py` | compatibility wrapper | Preset wrapper around `benchmark_hnsw_vs_omega.py --dataset cohere_10m`. |

## Internal Perf Helpers

| Script | Status | Purpose |
| --- | --- | --- |
| `perf_hnsw_hooks_microbench.sh` | internal | Run `perf stat/record` against `hnsw_hooks_microbench`. |
| `perf_ab_search_core.sh` | internal | Compare HNSW, empty-hooks, and hooks-only OMEGA search paths with `perf`. |
| `gcov.sh` | internal | Coverage helper for local development. |

These helpers assume a prepared benchmark environment and are not part of the
stable user-facing benchmarking interface.
