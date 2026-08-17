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
[cyclic|hot90] [HOT_QUERIES]
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

The benchmark also validates all 1,000 queries before measurement. mmap and
BufferReadStorage produced the same top-10 key-and-score checksum at nprobe 16:
`11931954759916436377`.

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

## Memory-constrained result

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

Three workloads intentionally cover both favorable and unfavorable cases:

- `hot90`: 90% of requests use five queries and 10% use the remaining 995.
  This models a skewed production workload whose hot posting lists can be
  reused while the tail remains much larger than the pool.
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

The strongest Buffer result is throughput density under a strict memory
budget, not maximum raw QPS. At `nprobe=4`, the 256-MiB pool uses 82.0% less
peak RSS than mmap and delivers 1.87x its QPS/GiB. At `nprobe=16`, the
768-MiB pool uses 68.4% less peak RSS and delivers 1.28x mmap QPS/GiB. The
1,024-MiB point delivers 1.25x mmap QPS/GiB. Capacity beyond that still raises
raw QPS but has diminishing throughput-per-memory returns.

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
