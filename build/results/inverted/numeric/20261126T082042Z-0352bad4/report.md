# Numeric inverted-index benchmark report

- Run: `20261126T082042Z-0352bad4`
- Status: passed
- Started: 2026-11-26T08:20:58.044963+00:00

## Environment

| Platform | Target architecture | Compiler | `__cplusplus` | Hardware threads |
| --- | --- | --- | --- | --- |
| macOS | arm64 | Clang 17.0.0 (clang-1700.4.4.1) | 201703 | 11 |

## Workloads

Repetitions per workload: 3

| Workload | Data type | Distribution | Documents | Value range | Order | NULL |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | int32 | uniform | 100,000 | 0_to_30000 | ascending | 0.00% |
| 2 | int32 | hotspot_moderate | 100,000 | 0_to_30000 | ascending | 0.00% |
| 3 | int32 | hotspot_extreme | 100,000 | 0_to_30000 | ascending | 0.00% |

### Observed dataset statistics

| Workload | NULL documents | Distinct non-NULL values | Min value | Max value | Documents per non-NULL value (min / p50 / p95 / max) |
| --- | --- | --- | --- | --- | --- |
| 1 | 0 | 28,880 | 0 | 29,999 | 1 / 3 / 7 / 12 |
| 2 | 0 | 25,765 | 0 | 29,999 | 1 / 2 / 17 / 31 |
| 3 | 0 | 8,795 | 4 | 29,998 | 1 / 1 / 3 / 343 |

## Build

| Workload | Insert (s) | Optimize (s) | Total (s) | Index (MiB) | Peak RSS (MiB) | Final RSS (MiB) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.09 | 0.30 | 0.38 | 1.18 | 21.51 | 21.51 |
| 2 | 0.09 | 0.28 | 0.37 | 1.03 | 21.26 | 21.26 |
| 3 | 0.11 | 0.16 | 0.27 | 0.60 | 20.35 | 20.35 |

## Queries

### Workload 1: int32 / uniform

0_to_30000 / ascending / NULL 0.00%

| Case | Mean query time (µs) | Matches per query |
| --- | --- | --- |
| eq_hit | 0.68 | 3.71 |
| eq_miss | 0.11 | 0.00 |
| ne_hit | 1.09 | 99,996.29 |
| lt_q01 | 8.23 | 998.00 |
| gt_q01 | 10.47 | 98,995.00 |
| lt_q11 | 17.32 | 11,001.00 |
| gt_q11 | 17.46 | 88,996.00 |
| lt_q33 | 8.53 | 32,999.00 |
| gt_q33 | 17.59 | 66,997.00 |
| lt_q50 | 21.24 | 49,999.00 |
| gt_q50 | 2.94 | 49,996.00 |
| lt_q77 | 17.51 | 77,000.00 |
| gt_q77 | 16.85 | 22,996.00 |
| lt_q90 | 15.74 | 90,002.00 |
| gt_q90 | 15.49 | 9,992.00 |
| lt_q99 | 10.72 | 99,000.00 |
| gt_q99 | 8.79 | 999.00 |
| is_null | 0.06 | 0.00 |
| is_not_null | 0.20 | 100,000.00 |

### Workload 2: int32 / hotspot_moderate

0_to_30000 / ascending / NULL 0.00%

| Case | Mean query time (µs) | Matches per query |
| --- | --- | --- |
| eq_hottest | 0.70 | 31.00 |
| eq_hot | 0.68 | 16.71 |
| eq_cold | 0.75 | 2.43 |
| eq_miss | 0.11 | 0.00 |
| ne_hit | 1.27 | 99,983.29 |
| lt_q01 | 8.03 | 1,005.00 |
| gt_q01 | 10.19 | 98,994.00 |
| lt_q11 | 17.96 | 11,001.00 |
| gt_q11 | 17.04 | 88,997.00 |
| lt_q33 | 9.02 | 33,000.00 |
| gt_q33 | 16.15 | 66,994.00 |
| lt_q50 | 20.22 | 50,000.00 |
| gt_q50 | 3.17 | 49,999.00 |
| lt_q77 | 16.84 | 77,000.00 |
| gt_q77 | 16.63 | 22,979.00 |
| lt_q90 | 15.26 | 89,998.00 |
| gt_q90 | 15.67 | 9,996.00 |
| lt_q99 | 9.89 | 99,004.00 |
| gt_q99 | 7.45 | 995.00 |
| is_null | 0.04 | 0.00 |
| is_not_null | 0.20 | 100,000.00 |

### Workload 3: int32 / hotspot_extreme

0_to_30000 / ascending / NULL 0.00%

| Case | Mean query time (µs) | Matches per query |
| --- | --- | --- |
| eq_hottest | 0.65 | 343.00 |
| eq_hot | 0.74 | 303.43 |
| eq_cold | 0.79 | 1.14 |
| eq_miss | 0.11 | 0.00 |
| ne_hit | 1.73 | 99,696.57 |
| lt_q01 | 8.54 | 964.00 |
| gt_q01 | 10.24 | 98,742.00 |
| lt_q11 | 7.43 | 10,855.00 |
| gt_q11 | 9.44 | 88,853.00 |
| lt_q33 | 8.21 | 32,937.00 |
| gt_q33 | 6.43 | 66,747.00 |
| lt_q50 | 2.03 | 49,980.00 |
| gt_q50 | 0.50 | 49,716.00 |
| lt_q77 | 15.23 | 77,000.00 |
| gt_q77 | 19.20 | 22,999.00 |
| lt_q90 | 15.13 | 90,000.00 |
| gt_q90 | 17.37 | 9,999.00 |
| lt_q99 | 7.69 | 99,079.00 |
| gt_q99 | 10.03 | 920.00 |
| is_null | 0.04 | 0.00 |
| is_not_null | 0.19 | 100,000.00 |
