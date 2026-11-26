# Numeric inverted-index benchmark report

- Run: `20261126T080602Z-38bc5c75`
- Status: passed
- Started: 2026-11-26T08:06:02.754675+00:00

## Environment

| Platform | Target architecture | Compiler | `__cplusplus` | Hardware threads |
| --- | --- | --- | --- | --- |
| macOS | arm64 | Clang 17.0.0 (clang-1700.4.4.1) | 201703 | 11 |

## Workloads

Repetitions per workload: 3

| Workload | Data type | Distribution | Documents | Value range | Order | NULL |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | int64 | uniform | 10,000,000 | 0_to_type_max | ascending | 10.00% |

### Observed dataset statistics

| Workload | NULL documents | Distinct non-NULL values | Min value | Max value | Documents per non-NULL value (min / p50 / p95 / max) |
| --- | --- | --- | --- | --- | --- |
| 1 | 1,000,000 | 9,000,000 | 361,440,557,135 | 9,223,371,150,062,733,241 | 1 / 1 / 1 / 1 |

## Build

| Workload | Insert (s) | Optimize (s) | Total (s) | Index (MiB) | Peak RSS (MiB) | Final RSS (MiB) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 6.20 | 53.80 | 60.00 | 219.04 | 107.26 | 107.26 |

## Queries

### Workload 1: int64 / uniform

0_to_type_max / ascending / NULL 10.00%

| Case | Mean query time (µs) | Matches per query |
| --- | --- | --- |
| eq_hit | 0.78 | 1.00 |
| eq_miss | 0.13 | 0.00 |
| ne_hit | 111.89 | 8,999,999.00 |
| lt_q01 | 11.51 | 90,000.00 |
| gt_q01 | 122.74 | 8,909,999.00 |
| lt_q11 | 32.88 | 990,000.00 |
| gt_q11 | 167.48 | 8,009,999.00 |
| lt_q33 | 36.16 | 2,970,000.00 |
| gt_q33 | 134.83 | 6,029,999.00 |
| lt_q50 | 16.93 | 4,500,000.00 |
| gt_q50 | 105.65 | 4,499,999.00 |
| lt_q77 | 73.92 | 6,930,000.00 |
| gt_q77 | 141.03 | 2,069,999.00 |
| lt_q90 | 90.90 | 8,100,000.00 |
| gt_q90 | 717.46 | 899,999.00 |
| lt_q99 | 125.49 | 8,910,000.00 |
| gt_q99 | 582.19 | 89,999.00 |
| is_null | 0.21 | 1,000,000.00 |
| is_not_null | 107.38 | 9,000,000.00 |
