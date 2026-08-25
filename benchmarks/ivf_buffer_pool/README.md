# Cohere 1M IVF storage benchmark

Dataset and index:

- 1,000,000 vectors, 768-dimensional FP32, Cosine metric
- IVF with 1,024 inverted lists (`32*32`), row-major blocks
- 100,000-vector training sample, 10 iterations per clustering level
- Index size: 3,095,214,208 bytes

Full-cache query setup:

- Apple M3 Pro, 36 GiB, macOS, 16 query threads, top-k 10
- 1,000 Cohere test queries
- mmap is explicitly memory-warmed at open
- BufferReadStorage uses sequential warmup with a 4,096 MiB pool
- 5-second query warmup followed by a 10-second measurement

Build:

```sh
build/bin/local_builder \
  benchmarks/ivf_buffer_pool/cohere_1m_ivf_fp32_build.yaml
```

Run one point:

```sh
build/bin/ivf_storage_bench \
  benchmarks/ivf_buffer_pool/cohere_1m_ivf_fp32_1024.index \
  /Users/ali.yzf/cohere_test_vector_1000.new.txt \
  buffer 4096 16 16 5 10
```

The optional arguments after the measurement duration are:

```text
[none|sequential] [ACTIVE_QUERIES] [VALIDATE_QUERIES]
[cyclic|centroid80|zipf1|hot90|semantic80] [HOT_QUERIES]
```

For example, this runs a 90/10 workload where 90% of requests use five hot
queries and the remaining 10% use the other 995 queries. Storage population is
demand-driven, so the query warmup—not a full-index preload—determines the
resident working set:

```sh
build/bin/ivf_storage_bench \
  benchmarks/ivf_buffer_pool/cohere_1m_ivf_fp32_1024.index \
  /Users/ali.yzf/cohere_test_vector_1000.new.txt \
  buffer 1024 16 16 10 15 none 1000 0 hot90 5
```

`hot90` is an exact-repeat best case, not the recommended production model.
The primary no-log model is `centroid80`: it deterministically selects 20% of
the IVF centroid domain and sends 80% of requests to queries assigned there.
The benchmark reports the actual share of probes handled by the busiest 20%
of centroids:

```sh
build/bin/ivf_storage_bench \
  benchmarks/ivf_buffer_pool/cohere_1m_ivf_fp32_1024.index \
  /Users/ali.yzf/cohere_test_vector_1000.new.txt \
  buffer 256 4 4 10 15 none 1000 0 centroid80 1000
```

Run the complete four-workload × storage-budget matrix serially (one fresh
process per point, so mmap and Buffer pools never overlap in memory):

```sh
python benchmarks/ivf_buffer_pool/run_workload_matrix.py \
  --binary build/bin/ivf_storage_bench \
  --index benchmarks/ivf_buffer_pool/cohere_1m_ivf_fp32_1024.index \
  --queries /path/to/cohere_test_vector_1000.new.txt \
  --output-dir /tmp/ivf-buffer-matrix
```

The runner defaults to three runs per point. `matrix_summary.csv` reports
median QPS/P99/RSS, QPS variance, and QPS/GiB relative to mmap.

Before publishing a curve, run one separate correctness invocation per storage
mode with `VALIDATE_QUERIES=1000`. mmap and BufferReadStorage produced the same
top-10 key-and-score checksum at nprobe 16: `11931954759916436377`. The matrix
runner intentionally sets validation to zero so correctness reads cannot warm
the timed cache.

## Resident scatter-read result

After adding resident-only page-span reads for row-major IVF posting lists,
BufferReadStorage no longer copies every cross-page batch into a contiguous
scratch allocation. It computes directly from pinned page spans and copies
only an individual vector that straddles a page boundary. Cold and partially
resident batches retain the previous contiguous fallback.

The follow-up full-cache curve is in
`cohere_1m_ivf_fp32_scatter_full_cache.csv`. Across nprobe 4–64, Buffer QPS is
within 1.8%–12.8% of mmap and improves by 54.6%–71.3% over the previous Buffer
implementation. All cache miss, eviction, and bypass counters remained zero.

## Four memory-constrained workload curves

The corrected comparison uses four workloads, four query threads, `nprobe=4`,
a ten-second warmup, and a fifteen-second measurement. Buffer uses direct I/O
and one benchmark process at a time. Raw results are in
`cohere_1m_ivf_fp32_four_workloads.csv`.

- `uniform unique`: all 1,000 queries cyclically, the no-hotspot lower bound.
- `centroid80`: deterministically selects 20% of the 1,024 primary-centroid
  domain as hot, then sends 80% of requests to queries assigned there. The
  selected domain contains 88 active centroids and 206 queries.
- `zipf1`: all 1,000 unique queries sampled with Zipf alpha 1.0.
- `hot90`: 90% of requests repeat five exact queries, the extreme upper bound.

| Workload | Busiest 20% of all centroids | Buffer 256 MiB | Buffer 512 MiB | Buffer 1 GiB | mmap |
|---|---:|---:|---:|---:|---:|
| Uniform unique | 64.0% | 150 QPS | 88 QPS | 90 QPS | 1429 QPS |
| 80/20 centroid skew | 74.1% | 319 QPS | 400 QPS | 464 QPS | 1768 QPS |
| Zipf alpha 1.0 | 82.9% | 388 QPS | 440 QPS | 509 QPS | 1758 QPS |
| 90/10 exact repeat | 96.4% | 756 QPS | 881 QPS | 999 QPS | 2077 QPS |

At 256 MiB, Buffer QPS/GiB relative to mmap is 0.98x for Uniform, 1.72x for
centroid80, 2.06x for Zipf, and 3.40x for exact repeat. This brackets the
product honestly: Buffer needs posting-list locality to improve throughput
density. The recommended centroid80 result is the main evidence; exact repeat
is only an upper bound.

For centroid80, increasing the pool from 256 to 512 MiB improves QPS by 25.5%
and P99 from 46.8 to 30.4 ms. Increasing it again to 1 GiB improves QPS by
15.8% and P99 to 24.9 ms, but QPS/GiB falls below mmap. The useful operating
point on this machine is therefore 256--512 MiB, depending on the latency SLO.

Uniform is deliberately unfavorable and non-monotonic: its 512-MiB point was
repeated at 72 and 88 QPS. With no reuse, admitting random pages can cost more
than rejecting them and using contiguous bypass reads. Do not infer that a
larger Pool always improves throughput.

## Historical memory-constrained result

The older `hot90` rows below used five exact hot queries. They also used a
correlated random sampler: the same random value selected the hot/tail branch
and the query within that branch. That restricted the reachable query set and
overstated locality. The sampler is now fixed and the corrected curve is
reported above, but the old rows below must not be used as product evidence.
Uniform, full-cache, and correctness rows are unaffected by that sampling bug.

The raw capacity curves are in
`cohere_1m_ivf_fp32_memory_constrained.csv`. These runs use one benchmark
process only; they do not apply whole-system memory pressure. A Buffer pool
smaller than the 2.88-GiB index working set is the memory constraint. RSS is
sampled every 50 ms during the measured window, and cache counters are deltas
for that same window.

A separate 20-query correctness check at `nprobe=16` produced the identical
top-10 key-and-score checksum (`7259331970886160525`) for a 256-MiB Buffer pool
and mmap. It is kept outside the timed curve so validation reads do not warm
the measured cache state.

The historical file contains these workloads:

- `hot90`: legacy exact-repeat best case; its old rows are invalid because of
  the correlated sampler described above.
- `uniform`: all 1,000 queries are issued cyclically. This is the adversarial
  no-locality case and defines the lower bound for a cache.
- Full-cache points use sequential storage warmup and define the Buffer
  compute-path upper bound. mmap points use a normal query warmup and define
  the OS page-cache upper bound.

### Representative points

| Workload | Storage | Pool | QPS | P99 | Peak RSS | QPS/GiB |
|---|---|---:|---:|---:|---:|---:|
| hot90, nprobe 4 | Buffer | 256 MiB | 835.8 | 85.0 ms | 417 MiB | 2052.5 |
| hot90, nprobe 4 | mmap | OS managed | 2491.2 | 26.7 ms | 2322 MiB | 1098.6 |
| hot90, nprobe 16 | Buffer | 768 MiB | 387.0 | 154.5 ms | 903 MiB | 439.1 |
| hot90, nprobe 16 | Buffer | 1024 MiB | 487.3 | 122.4 ms | 1161 MiB | 429.9 |
| hot90, nprobe 16 | mmap | OS managed | 959.0 | 44.0 ms | 2858 MiB | 343.7 |
| uniform, nprobe 16 | Buffer | 1024 MiB | 172.7 | 193.3 ms | 1167 MiB | 151.5 |
| uniform, nprobe 16 | Buffer full cache | 3072 MiB | 644.9 | 61.0 ms | 3058 MiB | 215.9 |
| uniform, nprobe 16 | mmap | OS managed | 722.8 | 54.5 ms | 2952 MiB | 250.8 |

The `hot90` ratios in this historical table are retained for audit only and
are superseded by the corrected four-workload result above.

The uniform curve is the boundary condition: without locality, constrained
Buffer cannot match mmap throughput and should be selected for predictable RSS
rather than speed. mmap is also preferable for a single stable hot working set
when its demand-resident pages already fit the memory budget. Buffer is most
useful when the working set is larger than the allowed resident memory, access
is skewed, and a hard application-controlled cache budget matters.

Two `nprobe=16` direct-I/O observations were excluded after contemporaneous
process sampling showed unrelated Git processes at roughly 100% CPU each and
security-scanner processes above 50%. Excluding them avoids presenting a
transient system slowdown as a cache benefit.
