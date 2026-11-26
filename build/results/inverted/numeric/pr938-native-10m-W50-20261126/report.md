# Numeric inverted-index benchmark report

- Run: `20261126T080915Z-9e86c361`
- Status: passed
- Started: 2026-11-26T08:09:15.995599+00:00

## Environment

| Platform | Target architecture | Compiler | `__cplusplus` | Hardware threads |
| --- | --- | --- | --- | --- |
| macOS | arm64 | Clang 17.0.0 (clang-1700.4.4.1) | 201703 | 11 |

## Workloads

Repetitions per workload: 3

| Workload | Data type | Distribution | Documents | Value range | Order | NULL |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | int64 | hotspot_moderate | 10,000,000 | 0_to_30000 | ascending | 10.00% |

### Observed dataset statistics

| Workload | NULL documents | Distinct non-NULL values | Min value | Max value | Documents per non-NULL value (min / p50 / p95 / max) |
| --- | --- | --- | --- | --- | --- |
| 1 | 1,000,000 | 30,000 | 0 | 29,999 | 116 / 168 / 1,500 / 1,634 |

## Build

| Workload | Insert (s) | Optimize (s) | Total (s) | Index (MiB) | Peak RSS (MiB) | Final RSS (MiB) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 10.35 | 0.22 | 10.57 | 36.08 | 65.54 | 65.54 |

## Queries

### Workload 1: int64 / hotspot_moderate

0_to_30000 / ascending / NULL 10.00%

| Case | Mean query time (µs) | Matches per query |
| --- | --- | --- |
| eq_hottest | 1.44 | 1,634.00 |
| eq_hot | 0.69 | 1,521.43 |
| eq_cold | 0.63 | 159.43 |
| eq_miss | 0.11 | 0.00 |
| ne_hit | 158.24 | 8,998,478.57 |
| lt_q01 | 33.95 | 89,439.00 |
| gt_q01 | 151.31 | 8,909,041.00 |
| lt_q11 | 78.50 | 989,992.00 |
| gt_q11 | 187.25 | 8,009,837.00 |
| lt_q33 | 51.81 | 2,970,055.00 |
| gt_q33 | 196.37 | 6,029,786.00 |
| lt_q50 | 37.68 | 4,500,046.00 |
| gt_q50 | 131.08 | 4,499,765.00 |
| lt_q77 | 108.55 | 6,929,491.00 |
| gt_q77 | 183.24 | 2,068,983.00 |
| lt_q90 | 107.38 | 8,100,081.00 |
| gt_q90 | 141.21 | 899,747.00 |
| lt_q99 | 139.13 | 8,909,988.00 |
| gt_q99 | 45.35 | 89,843.00 |
| is_null | 0.18 | 1,000,000.00 |
| is_not_null | 116.03 | 9,000,000.00 |
