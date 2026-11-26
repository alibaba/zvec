# Numeric inverted-index benchmark report

- Run: `20261126T081002Z-233f3b64`
- Status: passed
- Started: 2026-11-26T08:10:02.065714+00:00

## Environment

| Platform | Target architecture | Compiler | `__cplusplus` | Hardware threads |
| --- | --- | --- | --- | --- |
| macOS | arm64 | Clang 17.0.0 (clang-1700.4.4.1) | 201703 | 11 |

## Workloads

Repetitions per workload: 3

| Workload | Data type | Distribution | Documents | Value range | Order | NULL |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | int64 | hotspot_moderate | 10,000,000 | 0_to_type_max | ascending | 50.00% |

### Observed dataset statistics

| Workload | NULL documents | Distinct non-NULL values | Min value | Max value | Documents per non-NULL value (min / p50 / p95 / max) |
| --- | --- | --- | --- | --- | --- |
| 1 | 5,000,000 | 2,503,000 | 7,718,354,480,777 | 9,223,369,887,276,151,159 | 1 / 1 / 1 / 930 |

## Build

| Workload | Insert (s) | Optimize (s) | Total (s) | Index (MiB) | Peak RSS (MiB) | Final RSS (MiB) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 5.13 | 8.02 | 13.15 | 76.19 | 87.67 | 86.48 |

## Queries

### Workload 1: int64 / hotspot_moderate

0_to_type_max / ascending / NULL 50.00%

| Case | Mean query time (µs) | Matches per query |
| --- | --- | --- |
| eq_hottest | 0.89 | 930.00 |
| eq_hot | 1.42 | 857.57 |
| eq_cold | 0.84 | 1.00 |
| eq_miss | 0.12 | 0.00 |
| ne_hit | 102.04 | 4,999,142.43 |
| lt_q01 | 239.93 | 50,000.00 |
| gt_q01 | 354.97 | 4,949,999.00 |
| lt_q11 | 30.06 | 549,740.00 |
| gt_q11 | 131.04 | 4,449,425.00 |
| lt_q33 | 78.97 | 1,649,873.00 |
| gt_q33 | 412.83 | 3,349,327.00 |
| lt_q50 | 248.24 | 2,500,000.00 |
| gt_q50 | 328.06 | 2,499,999.00 |
| lt_q77 | 171.78 | 3,850,000.00 |
| gt_q77 | 316.03 | 1,149,999.00 |
| lt_q90 | 420.25 | 4,500,000.00 |
| gt_q90 | 449.71 | 499,999.00 |
| lt_q99 | 334.30 | 4,950,000.00 |
| gt_q99 | 217.55 | 49,999.00 |
| is_null | 0.17 | 5,000,000.00 |
| is_not_null | 102.92 | 5,000,000.00 |
