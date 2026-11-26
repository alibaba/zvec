# Numeric inverted-index benchmark report

- Run: `20261126T080347Z-9a6feb05`
- Status: passed
- Started: 2026-11-26T08:03:47.290456+00:00

## Environment

| Platform | Target architecture | Compiler | `__cplusplus` | Hardware threads |
| --- | --- | --- | --- | --- |
| macOS | arm64 | Clang 17.0.0 (clang-1700.4.4.1) | 201703 | 11 |

## Workloads

Repetitions per workload: 3

| Workload | Data type | Distribution | Documents | Value range | Order | NULL |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | int32 | uniform | 10,000,000 | 0_to_type_max | shuffled | 50.00% |

### Observed dataset statistics

| Workload | NULL documents | Distinct non-NULL values | Min value | Max value | Documents per non-NULL value (min / p50 / p95 / max) |
| --- | --- | --- | --- | --- | --- |
| 1 | 5,000,000 | 4,994,152 | 224 | 2,147,482,812 | 1 / 1 / 1 / 3 |

## Build

| Workload | Insert (s) | Optimize (s) | Total (s) | Index (MiB) | Peak RSS (MiB) | Final RSS (MiB) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 6.80 | 14.74 | 21.54 | 121.77 | 140.31 | 140.31 |

## Queries

### Workload 1: int32 / uniform

0_to_type_max / shuffled / NULL 50.00%

| Case | Mean query time (µs) | Matches per query |
| --- | --- | --- |
| eq_hit | 0.84 | 1.00 |
| eq_miss | 0.12 | 0.00 |
| ne_hit | 99.73 | 4,999,999.00 |
| lt_q01 | 395.76 | 50,000.00 |
| gt_q01 | 527.72 | 4,949,999.00 |
| lt_q11 | 1,495.43 | 550,000.00 |
| gt_q11 | 3,462.07 | 4,449,999.00 |
| lt_q33 | 134.40 | 1,650,000.00 |
| gt_q33 | 244.98 | 3,349,999.00 |
| lt_q50 | 33.67 | 2,500,000.00 |
| gt_q50 | 140.02 | 2,499,999.00 |
| lt_q77 | 151.26 | 3,850,000.00 |
| gt_q77 | 252.68 | 1,149,999.00 |
| lt_q90 | 238.04 | 4,500,000.00 |
| gt_q90 | 2,038.61 | 499,999.00 |
| lt_q99 | 570.07 | 4,950,000.00 |
| gt_q99 | 360.03 | 49,999.00 |
| is_null | 0.18 | 5,000,000.00 |
| is_not_null | 100.14 | 5,000,000.00 |
