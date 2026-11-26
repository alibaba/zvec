# Numeric inverted-index benchmark report

- Run: `20261126T080210Z-0a1e1eca`
- Status: passed
- Started: 2026-11-26T08:02:10.162935+00:00

## Environment

| Platform | Target architecture | Compiler | `__cplusplus` | Hardware threads |
| --- | --- | --- | --- | --- |
| macOS | arm64 | Clang 17.0.0 (clang-1700.4.4.1) | 201703 | 11 |

## Workloads

Repetitions per workload: 3

| Workload | Data type | Distribution | Documents | Value range | Order | NULL |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | int32 | uniform | 10,000,000 | 0_to_30000 | ascending | 0.00% |

### Observed dataset statistics

| Workload | NULL documents | Distinct non-NULL values | Min value | Max value | Documents per non-NULL value (min / p50 / p95 / max) |
| --- | --- | --- | --- | --- | --- |
| 1 | 0 | 30,000 | 0 | 29,999 | 264 / 333 / 364 / 417 |

## Build

| Workload | Insert (s) | Optimize (s) | Total (s) | Index (MiB) | Peak RSS (MiB) | Final RSS (MiB) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 12.59 | 0.25 | 12.84 | 28.70 | 48.27 | 48.27 |

## Queries

### Workload 1: int32 / uniform

0_to_30000 / ascending / NULL 0.00%

| Case | Mean query time (µs) | Matches per query |
| --- | --- | --- |
| eq_hit | 0.66 | 340.14 |
| eq_miss | 0.11 | 0.00 |
| ne_hit | 9.88 | 9,999,659.86 |
| lt_q01 | 32.11 | 100,109.00 |
| gt_q01 | 40.69 | 9,899,529.00 |
| lt_q11 | 69.45 | 1,099,891.00 |
| gt_q11 | 82.27 | 8,899,775.00 |
| lt_q33 | 45.88 | 3,300,159.00 |
| gt_q33 | 86.42 | 6,699,531.00 |
| lt_q50 | 119.83 | 4,999,849.00 |
| gt_q50 | 27.30 | 4,999,804.00 |
| lt_q77 | 92.13 | 7,700,066.00 |
| gt_q77 | 101.38 | 2,299,593.00 |
| lt_q90 | 65.37 | 8,999,909.00 |
| gt_q90 | 127.21 | 999,770.00 |
| lt_q99 | 43.91 | 9,900,136.00 |
| gt_q99 | 28.64 | 99,550.00 |
| is_null | 0.03 | 0.00 |
| is_not_null | 8.04 | 10,000,000.00 |
