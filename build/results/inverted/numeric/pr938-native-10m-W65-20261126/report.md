# Numeric inverted-index benchmark report

- Run: `20261126T081101Z-f32dfce7`
- Status: passed
- Started: 2026-11-26T08:11:01.172705+00:00

## Environment

| Platform | Target architecture | Compiler | `__cplusplus` | Hardware threads |
| --- | --- | --- | --- | --- |
| macOS | arm64 | Clang 17.0.0 (clang-1700.4.4.1) | 201703 | 11 |

## Workloads

Repetitions per workload: 3

| Workload | Data type | Distribution | Documents | Value range | Order | NULL |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | int64 | hotspot_extreme | 10,000,000 | 0_to_30000 | shuffled | 10.00% |

### Observed dataset statistics

| Workload | NULL documents | Distinct non-NULL values | Min value | Max value | Documents per non-NULL value (min / p50 / p95 / max) |
| --- | --- | --- | --- | --- | --- |
| 1 | 1,000,000 | 30,000 | 0 | 29,999 | 9 / 30 / 40 / 27,406 |

## Build

| Workload | Insert (s) | Optimize (s) | Total (s) | Index (MiB) | Peak RSS (MiB) | Final RSS (MiB) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 14.47 | 0.58 | 15.05 | 61.23 | 205.18 | 205.18 |

## Queries

### Workload 1: int64 / hotspot_extreme

0_to_30000 / shuffled / NULL 10.00%

| Case | Mean query time (µs) | Matches per query |
| --- | --- | --- |
| eq_hottest | 9.54 | 27,406.00 |
| eq_hot | 9.49 | 27,084.57 |
| eq_cold | 1.91 | 28.71 |
| eq_miss | 0.13 | 0.00 |
| ne_hit | 155.59 | 8,972,915.43 |
| lt_q01 | 638.88 | 84,304.00 |
| gt_q01 | 1,008.09 | 8,888,705.00 |
| lt_q11 | 198.44 | 994,720.00 |
| gt_q11 | 338.61 | 8,005,255.00 |
| lt_q33 | 239.51 | 2,969,994.00 |
| gt_q33 | 543.58 | 6,029,977.00 |
| lt_q50 | 34.52 | 4,506,514.00 |
| gt_q50 | 169.17 | 4,493,455.00 |
| lt_q77 | 201.61 | 6,940,840.00 |
| gt_q77 | 317.84 | 2,059,118.00 |
| lt_q90 | 366.34 | 8,091,577.00 |
| gt_q90 | 363.87 | 881,562.00 |
| lt_q99 | 878.40 | 8,911,728.00 |
| gt_q99 | 698.43 | 88,244.00 |
| is_null | 0.18 | 1,000,000.00 |
| is_not_null | 108.16 | 9,000,000.00 |
