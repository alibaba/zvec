# PR #938: sealed range-prefix benchmark

This comparison measures the range-prefix implementation and its build, storage, memory, and reopen costs.

- Baseline: `8d978b06e1a6f16e0179e5d13bb0653598272113`.
- Measured PR revision: `bdd20e0c88b21094780be14f70091cd2dd280db8`.
- Date: 2026-11-26 (Asia/Shanghai).
- Environment: macOS ARM64, Clang 17.0.0 (clang-1700.4.4.1), 11 hardware threads.
- Status: **80/80 workloads passed; 1,624 paired query cases verified**.

## Scope and method

The 100K-document run covers all 72 combinations of INT32/INT64, uniform/moderate-hotspot/extreme-hotspot distributions, two value domains, ascending/shuffled order, and NULL ratios of 0%, 10%, and 50%. The 10M-document run covers the original eight selected workloads: W01, W06, W12, W18, W44, W50, W57, and W65. Workload IDs follow the Cartesian-product order in `run_benchmarks.py`.

Both versions read identical input files generated with seed 19991230 and build separate fresh indexes once per workload. The build version order alternates by workload. Each version then runs three independent query processes, with alternating version order. Every case executes 100 iterations; hot-query timing uses the final 99 iterations. The tables aggregate the median of the three process samples for each case, weighting each case equally.

Each case reopens its index within the query process. `open_us` measures index open, `first_us` measures the first query after that open, and `hot_us` measures the subsequent iterations. `open + first` uses the median of the paired sum from each repetition. OS file caches were retained. RSS includes the process and allocator-retained memory; the process peak is cumulative across cases. Build time and file size have one sample per version and workload.

The two builds reused the same existing dependencies. The inverted-index production sources and related common implementation files were recompiled from the stated Git revisions, with identical benchmark instrumentation. The measured production source blobs, compiler settings, and binary hashes are recorded in [provenance.json](provenance.json).

Both versions passed full DocID verification against the original records for every query in all 80 workloads. Match counts also agree across both versions and all three query repetitions.

## Results

Query timing aggregates below are sums of per-case medians. Speedup is baseline / PR; a value above 1 means lower PR latency. Percentage change is PR / baseline - 1.

| Documents | Metric | Baseline | PR | Change |
| --- | --- | ---: | ---: | ---: |
| 100,000 | Build time total (s) | 20.97 | 28.04 | +33.7% |
| 100,000 | Seal / optimize total (s) | 14.74 | 21.73 | +47.4% |
| 100,000 | Index files total (MiB) | 74.06 | 87.62 | +18.3% |
| 100,000 | Median build peak RSS (MiB) | 24.68 | 25.57 | +3.6% |
| 100,000 | Maximum build peak RSS (MiB) | 26.78 | 27.73 | +3.5% |
| 100,000 | Maximum query peak RSS (MiB) | 20.53 | 20.81 | +1.4% |
| 100,000 | All hot queries, sum (ms) | 97.37 | 11.27 | 8.64x |
| 100,000 | Range hot queries, sum (ms) | 96.31 | 10.64 | 9.06x |
| 100,000 | Other hot queries, sum (ms) | 1.07 | 0.63 | 1.68x |
| 100,000 | Range index open, sum (ms) | 2,605.22 | 2,489.89 | -4.4% |
| 100,000 | Range first query, sum (ms) | 141.75 | 35.01 | -75.3% |
| 100,000 | Range open + first query, sum (ms) | 2,752.80 | 2,526.27 | -8.2% |
| 10,000,000 | Build time total (s) | 133.00 | 156.83 | +17.9% |
| 10,000,000 | Seal / optimize total (s) | 55.88 | 83.73 | +49.8% |
| 10,000,000 | Index files total (MiB) | 565.22 | 662.39 | +17.2% |
| 10,000,000 | Median build peak RSS (MiB) | 62.34 | 99.27 | +59.2% |
| 10,000,000 | Maximum build peak RSS (MiB) | 166.52 | 202.67 | +21.7% |
| 10,000,000 | Maximum query peak RSS (MiB) | 48.58 | 60.77 | +25.1% |
| 10,000,000 | All hot queries, sum (ms) | 324.40 | 55.04 | 5.89x |
| 10,000,000 | Range hot queries, sum (ms) | 322.43 | 53.27 | 6.05x |
| 10,000,000 | Other hot queries, sum (ms) | 1.96 | 1.77 | 1.11x |
| 10,000,000 | Range index open, sum (ms) | 479.06 | 838.31 | +75.0% |
| 10,000,000 | Range first query, sum (ms) | 381.82 | 91.16 | -76.1% |
| 10,000,000 | Range open + first query, sum (ms) | 866.54 | 930.53 | +7.4% |

Range hot-query latency improves by **9.06x** for the 100K matrix and **6.05x** for the selected 10M workloads. The 10M range open-plus-first-query aggregate increases by **7.4%**. Build time, index size, and memory costs are included above.

## Per-workload results

B/P denotes baseline / PR. Range latency is the sum of the 14 range-case medians for that workload.

| Documents | Workload | Type | Distribution | Value domain | Order | NULL | Build B/P (s) | Files B/P (MiB) | Range hot B/P (us) | Speedup | DocIDs |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 100,000 | W01 | int32 | uniform | 0_to_30000 | ascending | 0% | 0.31 / 0.41 | 1.14 / 1.18 | 1,629.68 / 182.47 | 8.93x | passed |
| 100,000 | W02 | int32 | uniform | 0_to_30000 | ascending | 10% | 0.28 / 0.36 | 1.08 / 1.25 | 1,513.34 / 205.18 | 7.38x | passed |
| 100,000 | W03 | int32 | uniform | 0_to_30000 | ascending | 50% | 0.23 / 0.31 | 0.73 / 0.90 | 1,350.82 / 121.85 | 11.09x | passed |
| 100,000 | W04 | int32 | uniform | 0_to_30000 | shuffled | 0% | 0.38 / 0.57 | 1.24 / 1.51 | 1,270.08 / 207.58 | 6.12x | passed |
| 100,000 | W05 | int32 | uniform | 0_to_30000 | shuffled | 10% | 0.33 / 0.44 | 1.16 / 1.42 | 1,293.58 / 199.36 | 6.49x | passed |
| 100,000 | W06 | int32 | uniform | 0_to_30000 | shuffled | 50% | 0.23 / 0.31 | 0.75 / 0.99 | 1,111.71 / 135.66 | 8.19x | passed |
| 100,000 | W07 | int32 | uniform | 0_to_type_max | ascending | 0% | 0.44 / 0.72 | 2.19 / 2.23 | 2,307.36 / 106.49 | 21.67x | passed |
| 100,000 | W08 | int32 | uniform | 0_to_type_max | ascending | 10% | 0.39 / 0.69 | 1.99 / 2.16 | 2,081.14 / 155.43 | 13.39x | passed |
| 100,000 | W09 | int32 | uniform | 0_to_type_max | ascending | 50% | 0.26 / 0.39 | 1.14 / 1.31 | 1,767.13 / 84.65 | 20.88x | passed |
| 100,000 | W10 | int32 | uniform | 0_to_type_max | shuffled | 0% | 0.48 / 0.95 | 2.20 / 2.47 | 1,732.19 / 138.60 | 12.50x | passed |
| 100,000 | W11 | int32 | uniform | 0_to_type_max | shuffled | 10% | 0.44 / 0.69 | 2.00 / 2.27 | 1,720.04 / 153.49 | 11.21x | passed |
| 100,000 | W12 | int32 | uniform | 0_to_type_max | shuffled | 50% | 0.27 / 0.41 | 1.14 / 1.39 | 1,518.85 / 107.93 | 14.07x | passed |
| 100,000 | W13 | int32 | hotspot_moderate | 0_to_30000 | ascending | 0% | 0.28 / 0.36 | 1.00 / 1.03 | 1,422.20 / 172.34 | 8.25x | passed |
| 100,000 | W14 | int32 | hotspot_moderate | 0_to_30000 | ascending | 10% | 0.28 / 0.34 | 0.94 / 1.11 | 1,374.73 / 199.31 | 6.90x | passed |
| 100,000 | W15 | int32 | hotspot_moderate | 0_to_30000 | ascending | 50% | 0.20 / 0.28 | 0.64 / 0.81 | 1,349.89 / 115.55 | 11.68x | passed |
| 100,000 | W16 | int32 | hotspot_moderate | 0_to_30000 | shuffled | 0% | 0.31 / 0.39 | 1.05 / 1.32 | 1,098.89 / 185.03 | 5.94x | passed |
| 100,000 | W17 | int32 | hotspot_moderate | 0_to_30000 | shuffled | 10% | 0.28 / 0.37 | 0.98 / 1.25 | 1,021.89 / 183.05 | 5.58x | passed |
| 100,000 | W18 | int32 | hotspot_moderate | 0_to_30000 | shuffled | 50% | 0.21 / 0.27 | 0.67 / 0.91 | 1,229.16 / 135.10 | 9.10x | passed |
| 100,000 | W19 | int32 | hotspot_moderate | 0_to_type_max | ascending | 0% | 0.35 / 0.48 | 1.43 / 1.47 | 1,694.69 / 159.66 | 10.61x | passed |
| 100,000 | W20 | int32 | hotspot_moderate | 0_to_type_max | ascending | 10% | 0.31 / 0.45 | 1.32 / 1.48 | 1,620.09 / 190.65 | 8.50x | passed |
| 100,000 | W21 | int32 | hotspot_moderate | 0_to_type_max | ascending | 50% | 0.21 / 0.31 | 0.80 / 0.97 | 1,482.68 / 112.15 | 13.22x | passed |
| 100,000 | W22 | int32 | hotspot_moderate | 0_to_type_max | shuffled | 0% | 0.36 / 0.51 | 1.46 / 1.73 | 1,359.67 / 190.64 | 7.13x | passed |
| 100,000 | W23 | int32 | hotspot_moderate | 0_to_type_max | shuffled | 10% | 0.38 / 0.49 | 1.35 / 1.61 | 1,407.41 / 200.24 | 7.03x | passed |
| 100,000 | W24 | int32 | hotspot_moderate | 0_to_type_max | shuffled | 50% | 0.23 / 0.33 | 0.83 / 1.07 | 1,393.02 / 127.72 | 10.91x | passed |
| 100,000 | W25 | int32 | hotspot_extreme | 0_to_30000 | ascending | 0% | 0.25 / 0.27 | 0.59 / 0.60 | 1,325.27 / 115.93 | 11.43x | passed |
| 100,000 | W26 | int32 | hotspot_extreme | 0_to_30000 | ascending | 10% | 0.21 / 0.28 | 0.55 / 0.70 | 1,200.92 / 139.64 | 8.60x | passed |
| 100,000 | W27 | int32 | hotspot_extreme | 0_to_30000 | ascending | 50% | 0.17 / 0.19 | 0.35 / 0.51 | 757.87 / 77.47 | 9.78x | passed |
| 100,000 | W28 | int32 | hotspot_extreme | 0_to_30000 | shuffled | 0% | 0.26 / 0.30 | 0.59 / 0.84 | 937.17 / 114.88 | 8.16x | passed |
| 100,000 | W29 | int32 | hotspot_extreme | 0_to_30000 | shuffled | 10% | 0.24 / 0.28 | 0.56 / 0.80 | 900.35 / 143.88 | 6.26x | passed |
| 100,000 | W30 | int32 | hotspot_extreme | 0_to_30000 | shuffled | 50% | 0.17 / 0.21 | 0.35 / 0.58 | 689.00 / 102.22 | 6.74x | passed |
| 100,000 | W31 | int32 | hotspot_extreme | 0_to_type_max | ascending | 0% | 0.42 / 0.30 | 0.63 / 0.64 | 1,309.65 / 124.02 | 10.56x | passed |
| 100,000 | W32 | int32 | hotspot_extreme | 0_to_type_max | ascending | 10% | 0.22 / 0.26 | 0.59 / 0.73 | 1,072.40 / 144.61 | 7.42x | passed |
| 100,000 | W33 | int32 | hotspot_extreme | 0_to_type_max | ascending | 50% | 0.16 / 0.19 | 0.36 / 0.52 | 841.93 / 81.92 | 10.28x | passed |
| 100,000 | W34 | int32 | hotspot_extreme | 0_to_type_max | shuffled | 0% | 0.28 / 0.33 | 0.63 / 0.88 | 879.11 / 105.92 | 8.30x | passed |
| 100,000 | W35 | int32 | hotspot_extreme | 0_to_type_max | shuffled | 10% | 0.25 / 0.28 | 0.59 / 0.83 | 726.16 / 145.01 | 5.01x | passed |
| 100,000 | W36 | int32 | hotspot_extreme | 0_to_type_max | shuffled | 50% | 0.18 / 0.20 | 0.37 / 0.60 | 718.36 / 108.80 | 6.60x | passed |
| 100,000 | W37 | int64 | uniform | 0_to_30000 | ascending | 0% | 0.43 / 0.39 | 1.16 / 1.20 | 1,618.25 / 188.69 | 8.58x | passed |
| 100,000 | W38 | int64 | uniform | 0_to_30000 | ascending | 10% | 0.29 / 0.37 | 1.09 / 1.27 | 1,627.49 / 223.53 | 7.28x | passed |
| 100,000 | W39 | int64 | uniform | 0_to_30000 | ascending | 50% | 0.22 / 0.29 | 0.74 / 0.91 | 1,267.76 / 113.03 | 11.22x | passed |
| 100,000 | W40 | int64 | uniform | 0_to_30000 | shuffled | 0% | 0.35 / 0.49 | 1.25 / 1.53 | 1,283.96 / 195.79 | 6.56x | passed |
| 100,000 | W41 | int64 | uniform | 0_to_30000 | shuffled | 10% | 0.31 / 0.41 | 1.17 / 1.44 | 1,226.74 / 201.10 | 6.10x | passed |
| 100,000 | W42 | int64 | uniform | 0_to_30000 | shuffled | 50% | 0.22 / 0.30 | 0.76 / 1.00 | 1,160.75 / 138.56 | 8.38x | passed |
| 100,000 | W43 | int64 | uniform | 0_to_type_max | ascending | 0% | 0.44 / 0.72 | 2.58 / 2.63 | 2,206.23 / 110.60 | 19.95x | passed |
| 100,000 | W44 | int64 | uniform | 0_to_type_max | ascending | 10% | 0.39 / 0.64 | 2.35 / 2.52 | 2,054.54 / 157.14 | 13.07x | passed |
| 100,000 | W45 | int64 | uniform | 0_to_type_max | ascending | 50% | 0.27 / 0.40 | 1.34 / 1.51 | 1,745.12 / 85.83 | 20.33x | passed |
| 100,000 | W46 | int64 | uniform | 0_to_type_max | shuffled | 0% | 0.72 / 0.98 | 2.59 / 2.87 | 1,749.21 / 139.34 | 12.55x | passed |
| 100,000 | W47 | int64 | uniform | 0_to_type_max | shuffled | 10% | 0.43 / 0.75 | 2.35 / 2.63 | 1,739.83 / 157.28 | 11.06x | passed |
| 100,000 | W48 | int64 | uniform | 0_to_type_max | shuffled | 50% | 0.29 / 0.43 | 1.34 / 1.59 | 1,574.43 / 104.96 | 15.00x | passed |
| 100,000 | W49 | int64 | hotspot_moderate | 0_to_30000 | ascending | 0% | 0.28 / 0.37 | 1.01 / 1.05 | 1,435.43 / 172.99 | 8.30x | passed |
| 100,000 | W50 | int64 | hotspot_moderate | 0_to_30000 | ascending | 10% | 0.26 / 0.33 | 0.95 / 1.13 | 1,406.59 / 202.92 | 6.93x | passed |
| 100,000 | W51 | int64 | hotspot_moderate | 0_to_30000 | ascending | 50% | 0.22 / 0.26 | 0.65 / 0.82 | 1,388.59 / 115.57 | 12.02x | passed |
| 100,000 | W52 | int64 | hotspot_moderate | 0_to_30000 | shuffled | 0% | 0.31 / 0.39 | 1.06 / 1.34 | 1,126.11 / 193.35 | 5.82x | passed |
| 100,000 | W53 | int64 | hotspot_moderate | 0_to_30000 | shuffled | 10% | 0.32 / 0.36 | 1.00 / 1.27 | 1,026.87 / 183.04 | 5.61x | passed |
| 100,000 | W54 | int64 | hotspot_moderate | 0_to_30000 | shuffled | 50% | 0.21 / 0.27 | 0.67 / 0.92 | 1,214.51 / 137.96 | 8.80x | passed |
| 100,000 | W55 | int64 | hotspot_moderate | 0_to_type_max | ascending | 0% | 0.33 / 0.50 | 1.64 / 1.69 | 1,635.78 / 184.80 | 8.85x | passed |
| 100,000 | W56 | int64 | hotspot_moderate | 0_to_type_max | ascending | 10% | 0.35 / 0.47 | 1.51 / 1.68 | 1,971.59 / 255.33 | 7.72x | passed |
| 100,000 | W57 | int64 | hotspot_moderate | 0_to_type_max | ascending | 50% | 0.23 / 0.31 | 0.91 / 1.09 | 1,601.16 / 116.21 | 13.78x | passed |
| 100,000 | W58 | int64 | hotspot_moderate | 0_to_type_max | shuffled | 0% | 0.36 / 0.52 | 1.67 / 1.95 | 1,192.69 / 201.12 | 5.93x | passed |
| 100,000 | W59 | int64 | hotspot_moderate | 0_to_type_max | shuffled | 10% | 0.33 / 0.48 | 1.54 / 1.81 | 1,569.35 / 204.27 | 7.68x | passed |
| 100,000 | W60 | int64 | hotspot_moderate | 0_to_type_max | shuffled | 50% | 0.23 / 0.31 | 0.94 / 1.18 | 2,024.60 / 169.02 | 11.98x | passed |
| 100,000 | W61 | int64 | hotspot_extreme | 0_to_30000 | ascending | 0% | 0.27 / 0.29 | 0.60 / 0.61 | 1,320.18 / 121.28 | 10.89x | passed |
| 100,000 | W62 | int64 | hotspot_extreme | 0_to_30000 | ascending | 10% | 0.22 / 0.36 | 0.56 / 0.71 | 1,226.37 / 140.60 | 8.72x | passed |
| 100,000 | W63 | int64 | hotspot_extreme | 0_to_30000 | ascending | 50% | 0.16 / 0.19 | 0.35 / 0.52 | 737.35 / 75.37 | 9.78x | passed |
| 100,000 | W64 | int64 | hotspot_extreme | 0_to_30000 | shuffled | 0% | 0.28 / 0.32 | 0.60 / 0.85 | 961.12 / 113.49 | 8.47x | passed |
| 100,000 | W65 | int64 | hotspot_extreme | 0_to_30000 | shuffled | 10% | 0.30 / 0.30 | 0.56 / 0.81 | 885.46 / 145.04 | 6.10x | passed |
| 100,000 | W66 | int64 | hotspot_extreme | 0_to_30000 | shuffled | 50% | 0.18 / 0.23 | 0.36 / 0.59 | 622.47 / 93.59 | 6.65x | passed |
| 100,000 | W67 | int64 | hotspot_extreme | 0_to_type_max | ascending | 0% | 0.24 / 0.29 | 0.67 / 0.69 | 1,169.21 / 127.79 | 9.15x | passed |
| 100,000 | W68 | int64 | hotspot_extreme | 0_to_type_max | ascending | 10% | 0.23 / 0.28 | 0.62 / 0.77 | 1,478.57 / 169.60 | 8.72x | passed |
| 100,000 | W69 | int64 | hotspot_extreme | 0_to_type_max | ascending | 50% | 0.17 / 0.19 | 0.38 / 0.55 | 834.91 / 86.62 | 9.64x | passed |
| 100,000 | W70 | int64 | hotspot_extreme | 0_to_type_max | shuffled | 0% | 0.35 / 0.35 | 0.68 / 0.92 | 813.45 / 114.40 | 7.11x | passed |
| 100,000 | W71 | int64 | hotspot_extreme | 0_to_type_max | shuffled | 10% | 0.27 / 0.35 | 0.63 / 0.88 | 1,515.05 / 190.46 | 7.95x | passed |
| 100,000 | W72 | int64 | hotspot_extreme | 0_to_type_max | shuffled | 50% | 0.26 / 0.23 | 0.39 / 0.62 | 789.24 / 130.06 | 6.07x | passed |
| 10,000,000 | W01 | int32 | uniform | 0_to_30000 | ascending | 0% | 13.47 / 13.12 | 28.65 / 28.70 | 3,866.73 / 966.55 | 4.00x | passed |
| 10,000,000 | W06 | int32 | uniform | 0_to_30000 | shuffled | 50% | 9.16 / 8.90 | 45.05 / 62.41 | 26,670.40 / 11,660.00 | 2.29x | passed |
| 10,000,000 | W12 | int32 | uniform | 0_to_type_max | shuffled | 50% | 33.04 / 30.63 | 104.39 / 121.77 | 49,148.11 / 15,057.18 | 3.26x | passed |
| 10,000,000 | W18 | int32 | hotspot_moderate | 0_to_30000 | shuffled | 50% | 10.17 / 9.26 | 39.60 / 56.97 | 27,488.21 / 12,593.43 | 2.18x | passed |
| 10,000,000 | W44 | int64 | uniform | 0_to_type_max | ascending | 10% | 33.48 / 58.67 | 210.01 / 219.04 | 118,058.48 / 2,309.91 | 51.11x | passed |
| 10,000,000 | W50 | int64 | hotspot_moderate | 0_to_30000 | ascending | 10% | 10.51 / 10.49 | 27.04 / 36.08 | 3,692.60 / 1,467.49 | 2.52x | passed |
| 10,000,000 | W57 | int64 | hotspot_moderate | 0_to_type_max | ascending | 50% | 9.67 / 11.91 | 67.14 / 76.19 | 64,223.56 / 3,604.24 | 17.82x | passed |
| 10,000,000 | W65 | int64 | hotspot_extreme | 0_to_30000 | shuffled | 10% | 13.50 / 13.85 | 43.34 / 61.23 | 29,286.38 / 5,607.40 | 5.22x | passed |

## Measured range-query slowdowns

Seven range cases have a measured median increase of at least 10% and 10 us. These are follow-up candidates; attributing a root cause requires separate investigation. The three-process ranges for both versions are included in [cases.csv](cases.csv), and each sample is available in [query-samples.csv](query-samples.csv).

| Documents | Workload | Query | Baseline (us) | PR (us) | Change | PR sample range (us) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 10,000,000 | W18 | gt_q11 | 3,304.31 | 3,690.29 | +11.7% | 3,609.85 - 3,708.86 |
| 10,000,000 | W18 | lt_q11 | 2,264.79 | 2,493.46 | +10.1% | 2,423.42 - 2,585.55 |
| 10,000,000 | W18 | gt_q01 | 576.11 | 671.09 | +16.5% | 586.97 - 714.09 |
| 10,000,000 | W18 | lt_q01 | 449.84 | 510.29 | +13.4% | 469.16 - 514.12 |
| 10,000,000 | W18 | gt_q99 | 398.93 | 439.16 | +10.1% | 388.91 - 491.01 |
| 10,000,000 | W01 | gt_q90 | 110.44 | 129.79 | +17.5% | 114.74 - 138.18 |
| 100,000 | W22 | gt_q11 | 38.43 | 49.63 | +29.1% | 46.73 - 49.73 |

## Data and reproduction

- [cases.csv](cases.csv): all 1,624 paired cases, operands, hit counts, per-version medians, RSS, and hot-query sample ranges.
- [builds.csv](builds.csv): all 160 build samples.
- [query-samples.csv](query-samples.csv): all 9,744 individual per-case process samples, including paired open/first timing and RSS.
- [summary.json](summary.json): aggregate results by scale and query category.
- [provenance.json](provenance.json): source revisions, workload selection, verification, instrumentation and binary hashes.
- [instrumentation.patch](instrumentation.patch): identical measurement hooks applied to each benchmark snapshot.

To repeat this comparison, prepare checkouts of both measured revisions, apply `instrumentation.patch` to each, and build `inverted_numeric_bench` in Release mode using the same toolchain and dependencies. Use the existing `prepare`, `build`, `query`, and `verify` binary stages. Share the same generated `data.bin` and `dataset.json` between the version directories, build once per version, then invoke `query` in three separate processes, renaming each `query.json` before the next invocation. Alternate version order as described above, and run `verify` for both indexes.

The stock Python runner rebuilds an index for every repetition; this comparison uses one build and three query processes per version. The original eight 10M workload profiles are listed explicitly in the per-workload table. RSS and first-query results follow the cache and process lifecycle described above.
