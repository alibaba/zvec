# OMEGA Runtime Flags

This note classifies the runtime flags currently used by zvec's HNSW/OMEGA
integration. The goal is to distinguish product-path controls from benchmark
and profiling knobs.

## Production / Safety

| Flag | Scope | Purpose |
| --- | --- | --- |
| `ZVEC_OMEGA_DISABLE_MODEL_PREDICTION` | OMEGA search path | Forces the OMEGA path to run without model-driven stopping. Useful as a fallback/debug switch while preserving the hook/control path. |

## Profiling / Per-query stats

| Flag | Scope | Purpose |
| --- | --- | --- |
| `ZVEC_HNSW_LOG_QUERY_STATS` | HNSW streamer | Enables per-query HNSW stats logging. |
| `ZVEC_HNSW_LOG_QUERY_LIMIT` | HNSW streamer | Caps how many HNSW query-stat lines are emitted. |
| `ZVEC_OMEGA_LOG_QUERY_STATS` | OMEGA streamer | Enables per-query OMEGA stats logging. |
| `ZVEC_OMEGA_LOG_QUERY_LIMIT` | OMEGA streamer | Caps how many OMEGA query-stat lines are emitted. |
| `ZVEC_OMEGA_PROFILE_CONTROL_TIMING` | OMEGA / OMEGALib | Enables fine-grained OMEGA control-path timing. This is profiling-only and should stay off for normal benchmark runs. |

## Benchmark-only

| Flag | Scope | Purpose |
| --- | --- | --- |
| `ZVEC_HNSW_ENABLE_EMPTY_HOOKS` | HNSW streamer | Forces HNSW to execute the empty-hook path so hook dispatch overhead can be measured in isolation. |

## Generic logging

| Flag | Scope | Purpose |
| --- | --- | --- |
| `ZVEC_LOG_LEVEL` | Logging | Controls zvec log verbosity. Benchmark scripts commonly set it to `INFO` so query-stat lines are visible. |

## Cleanup notes

- All flags listed above still have active call sites or benchmark usage.
- No remaining runtime env var was removed in this cleanup step because no
  clearly dead env-var knob was found in the current branch.
- Previously removed dead surface in this cleanup phase was limited to unused
  code/API, not to active runtime flags.

