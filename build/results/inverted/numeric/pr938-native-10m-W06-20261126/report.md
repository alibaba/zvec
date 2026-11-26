# Numeric inverted-index benchmark report

- Run: `20261126T080300Z-aa995cb5`
- Status: passed
- Started: 2026-11-26T08:03:00.050212+00:00

## Environment

| Platform | Target architecture | Compiler | `__cplusplus` | Hardware threads |
| --- | --- | --- | --- | --- |
| macOS | arm64 | Clang 17.0.0 (clang-1700.4.4.1) | 201703 | 11 |

## Workloads

Repetitions per workload: 3

| Workload | Data type | Distribution | Documents | Value range | Order | NULL |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | int32 | uniform | 10,000,000 | 0_to_30000 | shuffled | 50.00% |

### Observed dataset statistics

| Workload | NULL documents | Distinct non-NULL values | Min value | Max value | Documents per non-NULL value (min / p50 / p95 / max) |
| --- | --- | --- | --- | --- | --- |
| 1 | 5,000,000 | 30,000 | 0 | 29,999 | 120 / 166 / 188 / 217 |

## Build

| Workload | Insert (s) | Optimize (s) | Total (s) | Index (MiB) | Peak RSS (MiB) | Final RSS (MiB) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 7.93 | 1.35 | 9.28 | 62.41 | 95.32 | 95.32 |

## Queries

### Workload 1: int32 / uniform

0_to_30000 / shuffled / NULL 50.00%

| Case | Mean query time (µs) | Matches per query |
| --- | --- | --- |
| eq_hit | 5.10 | 169.57 |
| eq_miss | 0.11 | 0.00 |
| ne_hit | 129.93 | 4,999,830.43 |
| lt_q01 | 511.38 | 50,098.00 |
| gt_q01 | 708.14 | 4,949,754.00 |
| lt_q11 | 2,436.07 | 549,925.00 |
| gt_q11 | 3,667.55 | 4,449,901.00 |
| lt_q33 | 185.98 | 1,650,004.00 |
| gt_q33 | 967.99 | 3,349,819.00 |
| lt_q50 | 256.63 | 2,500,079.00 |
| gt_q50 | 173.32 | 2,499,751.00 |
| lt_q77 | 315.34 | 3,850,016.00 |
| gt_q77 | 418.93 | 1,149,810.00 |
| lt_q90 | 318.77 | 4,500,028.00 |
| gt_q90 | 2,303.22 | 499,809.00 |
| lt_q99 | 634.70 | 4,950,051.00 |
| gt_q99 | 404.72 | 49,752.00 |
| is_null | 0.22 | 5,000,000.00 |
| is_not_null | 102.21 | 5,000,000.00 |
