# Numeric inverted-index benchmark report

- Run: `20261126T080510Z-36514707`
- Status: passed
- Started: 2026-11-26T08:05:10.548767+00:00

## Environment

| Platform | Target architecture | Compiler | `__cplusplus` | Hardware threads |
| --- | --- | --- | --- | --- |
| macOS | arm64 | Clang 17.0.0 (clang-1700.4.4.1) | 201703 | 11 |

## Workloads

Repetitions per workload: 3

| Workload | Data type | Distribution | Documents | Value range | Order | NULL |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | int32 | hotspot_moderate | 10,000,000 | 0_to_30000 | shuffled | 50.00% |

### Observed dataset statistics

| Workload | NULL documents | Distinct non-NULL values | Min value | Max value | Documents per non-NULL value (min / p50 / p95 / max) |
| --- | --- | --- | --- | --- | --- |
| 1 | 5,000,000 | 30,000 | 0 | 29,999 | 57 / 94 / 832 / 930 |

## Build

| Workload | Insert (s) | Optimize (s) | Total (s) | Index (MiB) | Peak RSS (MiB) | Final RSS (MiB) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 8.93 | 1.15 | 10.08 | 56.97 | 97.48 | 97.48 |

## Queries

### Workload 1: int32 / hotspot_moderate

0_to_30000 / shuffled / NULL 50.00%

| Case | Mean query time (µs) | Matches per query |
| --- | --- | --- |
| eq_hottest | 11.06 | 930.00 |
| eq_hot | 8.69 | 857.57 |
| eq_cold | 3.65 | 88.57 |
| eq_miss | 0.13 | 0.00 |
| ne_hit | 129.74 | 4,999,142.43 |
| lt_q01 | 468.65 | 50,293.00 |
| gt_q01 | 621.46 | 4,949,619.00 |
| lt_q11 | 2,501.72 | 549,983.00 |
| gt_q11 | 3,597.23 | 4,449,917.00 |
| lt_q33 | 158.85 | 1,649,989.00 |
| gt_q33 | 426.69 | 3,349,927.00 |
| lt_q50 | 115.38 | 2,499,974.00 |
| gt_q50 | 152.78 | 2,499,942.00 |
| lt_q77 | 294.87 | 3,850,042.00 |
| gt_q77 | 398.42 | 1,149,870.00 |
| lt_q90 | 258.16 | 4,499,937.00 |
| gt_q90 | 2,135.69 | 499,251.00 |
| lt_q99 | 593.52 | 4,950,051.00 |
| gt_q99 | 407.07 | 49,867.00 |
| is_null | 0.18 | 5,000,000.00 |
| is_not_null | 108.89 | 5,000,000.00 |
