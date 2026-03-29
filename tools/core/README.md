# Core Tools

This directory mixes product-adjacent command-line tools with internal
benchmark helpers. The table below is the maintenance contract for each group.

## Maintained Tools

These binaries are part of the normal local benchmarking / debugging workflow
 and should keep building with the rest of the tree.

| Tool | Status | Purpose |
| --- | --- | --- |
| `txt2vecs` | maintained | Convert text vectors into zvec binary format. |
| `local_builder` | maintained | Build an index from YAML config. |
| `recall` | maintained | Offline recall evaluation from YAML config. |
| `bench` | maintained | Throughput / latency benchmarking from YAML config. |

## Internal Perf Tools

These tools exist to answer OMEGA and HNSW integration questions. They are
useful for development, but they are not general product entrypoints.

| Tool | Status | Purpose | Typical entrypoint |
| --- | --- | --- | --- |
| `hnsw_hooks_microbench` | internal | Compare raw HNSW, empty hooks, and OMEGA hooks on the same search core. | `scripts/perf_hnsw_hooks_microbench.sh` |
| `omega_predict_microbench` | internal | Measure standalone OMEGA prediction cost outside the full search loop. | Invoke binary directly with a saved model. |

`hnsw_hooks_microbench` assumes a persisted HNSW index and benchmark query set.
It is intended for single-machine profiling, not for end-user benchmarking.

## Compatibility / Reference Tools

These binaries are retained so older YAML-based flows and historical result
reproduction still work, but new work should prefer the maintained entrypoints
above.

| Tool | Status | Purpose |
| --- | --- | --- |
| `local_builder_original` | compatibility | Reference copy of the legacy builder flow. |
| `recall_original` | compatibility | Reference copy of the legacy recall flow. |
| `bench_original` | compatibility | Reference copy of the legacy bench flow. |
| `convert_cohere_parquet.py` | compatibility | Dataset conversion helper for historical Cohere experiments. |

## Notes

- The JSON-driven OMEGA vs HNSW workflow lives under [`scripts/`](../scripts).
- Perf shell wrappers in [`scripts/`](../scripts) are internal-only and assume a
  prepared local environment plus existing benchmark artifacts.
