# Quantized graph search with configurable entry routing

This opt-in core C++ mode searches an immutable HNSW snapshot using
node-centered RaBitQ neighbor blocks and selective full-dimensional FP32
scoring. It supports inline, nonsparse FP32 vectors, cosine or squared L2, and
original dimensions from 1 to 4096. Ordinary HNSW remains the default. Mutable
collection and Python configuration are not extended by this change.

The [Chinese algorithm article](quantized_graph_principles.zh-CN.md) derives the
Cohere pipeline and explains the distinction between approximate navigation
and conditional correctness of refinement within the expanded candidate set.

## Search and entry selection

1. Prepare the full-dimensional query and seeded orthogonal rotation. Dimension
   768 uses `H12 ⊗ H64`; other dimensions are padded to a power of two, at least
   64, and use a Hadamard transform. Padding preserves the original distances.
2. If a model is supplied and routing is enabled, score its 16-way hierarchy
   using a configurable prefix of rotated residual coordinates. This selects
   an actual **layer-zero node**, replacing upper-layer HNSW entry descent.
   Without a router, start from the valid node nearest the corpus centroid.
3. Search the derived degree-32 graph using full-dimensional RaBitQ blocks.
   Preserve existing links for short rows and augment from two-hop candidates;
   diversity-prune longer rows. Bucket ordering and stopping are approximate.
4. For cosine, estimate centers with full-dimensional shifted integer dot
   products, then order expanded candidates by conservative bounds and refine
   against original FP32 vectors in batches of four. For L2, use SQ8 centers
   with a triangle-inequality bound and refine promising nodes during traversal.
   L2 neighbor codes are anchored at the reconstructed center, and navigation
   consistently uses that center's estimated distance.

**Refinement is retained.** Entry projection never replaces the full-dimensional
search/scoring space. Bounds can avoid full distance computations within the
expanded candidate set, subject to valid arithmetic bounds and scan budgets;
projected routing, RaBitQ navigation and adaptive stopping do not guarantee
whole-database recall. A lower-dimensional router is not always faster overall.

The graph arena and optional model are constructed/loaded on open, using eight
prebuild workers. They are not persisted into the original HNSW file. Open time
is excluded from query QPS. Searches may run concurrently; insertion is rejected
while this mode is enabled. Grouped search uses the existing HNSW path.

## Core API

```cpp
using namespace zvec::core_interface;
HNSWIndexParam param(MetricType::kL2sq, 960, 50, 500);
param.data_type = DataType::DT_FP32;
param.use_quantized_graph = true;
// Optional: empty uses the centroid entry, and requires no trained model.
param.quantized_graph_model = "/path/gist480.bin";
// Initialize/open the matching existing snapshot with read_only = true.

HNSWQueryParam query;
query.topk = 10;
query.ef_search = 800;
query.quantized_graph_route_beam = 1;  // 0: disable routing; 1..4: route beam.
query.quantized_graph_margin = -1;    // -1: metric default; >=0: explicit margin.
query.quantized_graph_bucket_width = 0;  // 0: metric default; (0,1]: coefficient.
query.quantized_graph_max_expansions = 0;  // 0: metric default.
```

A nonempty `quantized_graph_model` also enables the mode, preserving existing
callers. The builder exposes `with_use_quantized_graph(bool)` and
`with_quantized_graph_model(path)`; the corresponding JSON fields have the same
names. Low-level streamer keys are `proxima.hnsw.quantized_graph.enabled` and
`proxima.hnsw.quantized_graph.model`. Query options also round-trip through
`IndexFactory::QueryParamSerializeToJson` / `QueryParamDeserializeFromJson`,
so the standard `recall` and `bench` tools can configure them.
There are no production experiment environment variables.

Metric defaults remain explicit presets, not universal recommendations:

| Setting | Cosine | Squared L2 |
| --- | ---: | ---: |
| Relative bucket width | 0.016 | 0.002 |
| Stopping margin | 0.02 | 0.085 |
| Minimum expansions before adaptive stopping | 10 | 10 |
| Expansion cap after enough eligible results | 64 | 256 |
| Route beam when a model is supplied | 1 | 1 |

The cap applies after at least Top K eligible estimates/exact results exist; it
is not an unconditional bound under filters. `ef_search` is not the bucket's
hard capacity or an expansion count. Use beam zero with the same model to
isolate routing cost while keeping the mean and graph construction unchanged.

A RaBitQ-enabled GCC/Clang x86-64 build and AVX512F/BW/DQ/VL/VNNI plus FMA are
required for optimized graph search. Unsupported configurations fail explicitly;
ordinary HNSW remains available. The offline trainer uses native dispatched
math kernels and does not require VNNI.

## Build the entry model in C++

```sh
cmake --build "$BUILD_DIR" --target quantized_graph_router -j 8

# Cohere-compatible 768D cosine; default projection is half the original D.
"$BUILD_DIR/bin/quantized_graph_router" build \
  /path/cohere.index /path/cohere384.bin

# GIST: ordinary L2 k-means; 960 -> padded rotation 1024 -> retain 480.
"$BUILD_DIR/bin/quantized_graph_router" build \
  /path/gist.index /path/gist480.bin 65536 202610261259 \
  --dimension 960 --metric l2 --projected 480
```

The tool reads the source snapshot in read-only mode, checks its layout against
the requested metric/dimension, computes the full-corpus mean and samples up to
65,536 internal node IDs. It runs ten k-means iterations per split, normalizing
centers only for cosine, then rotates, projects, quantizes and publishes the
model. For GIST 480 coordinates are padded to a 512-byte SIMD stride; query tail
codes are zero. A 512- or 1024-coordinate model can be generated with the same
seed and anchors, so it shares the same full-dimensional clustering tree.

Training, conversion and queries require no Python or NumPy. The output
directory must already exist; existing output is never replaced, and partial
output is not published. Training is single-threaded, using native FP32 distance
kernels, FP64 center accumulation, and `std::mt19937_64` initialization. Same
input/seed/build reproduces the model; NumPy PCG64/BLAS or a different SIMD build
need not produce the same tree. Degenerate splits use balanced partitions with
a depth guard. Neither training mode reads queries or ground truth.

For migration or controlled training-input comparisons:

```sh
"$BUILD_DIR/bin/quantized_graph_router" train \
  /path/anchors.bin /path/mean.f32 /path/router.bin 202610261259 \
  --metric cosine --projected 256
```

The little-endian anchor export contains five uint32 values (magic `0x51474159`,
version 1, anchor count, original dimension, graph document count), followed by
packed records: uint32 internal ID, uint64 external key, then D FP32 coordinates.
The mean contains exactly D FP32 values. IDs must be unique/in range; vectors
must be finite, and cosine vectors normalized. External keys never replace
internal IDs. `train` infers D from the export. The historical Cohere mean used
65,536 database samples; `build` uses the full corpus. Use the same supplied
mean when isolating a trainer/representation change from this input difference.
The snapshot builder requires valid, readable vectors at every internal ID;
cosine vectors must be nonzero and normalized.

## Model compatibility

Version 2 models retain the exact legacy 768D/384D cosine representation.
Version 3 adds original dimension, metric, rotation dimension and code stride.
Both store original/rotated means and quantized routing groups. In cosine the
per-center bias is `mean · center`; in L2 it is the full-dimensional squared
residual norm. The dot-product expansion factor uses the **rotation dimension**
divided by retained coordinates. L2 centers are not normalized.

The loader validates dimensions, metric, transform, finite values, forward tree
edges, depth, leaf IDs and complete file length before publishing the model.
**The model belongs to the exact graph's internal ID mapping.** Count and ID
validity do not prove dataset identity; the format contains no cryptographic
graph fingerprint. Keep external hashes in deployment/benchmark manifests.
Rebuild a matching router after changing graph IDs or the dataset.

## Standard recall and bench tools

Use the existing `tools/core/recall` and `tools/core/bench` executables for
comparable tool-level measurements. Build the underlying HNSW snapshot with
`local_builder` when needed; reuse the same snapshot across search ablations.
Changing the graph requires rebuilding a matching entry model.

```sh
cmake --build "$BUILD_DIR" --target local_builder recall bench -j 8
"$BUILD_DIR/bin/recall" /local/experiment.yaml
"$BUILD_DIR/bin/bench" /local/experiment.yaml
```

Example configuration for a GIST snapshot and an existing 256-coordinate model:

```yaml
IndexCommon:
  IndexConfig: '{"index_type":"kHNSW","data_type":"DT_FP32","metric_type":"kL2sq","dimension":960,"m":50,"ef_construction":500,"use_id_map":true,"use_quantized_graph":true,"quantized_graph_model":"/path/gist256.bin"}'
  IndexPath: /path/gist.index
  TopK: '10'
  QueryFile: /path/queries.txt
  QueryFirstSep: ';'
  QuerySecondSep: ' '
  GroundTruthFile: /path/neighbors.txt
  GroundTruthFirstSep: ';'
  GroundTruthSecondSep: ' '
  CompareById: true
  RecallThreadCount: 1
  RecallBatchCount: 0
  RecallGTCount: 10
  RecallLogDir: /local/recall-results
  BenchBatchCount: 1
  BenchIterCount: 1000000000
  BenchSecs: 5
  BenchRounds: 3
  BenchThreadCounts: [1, 8]
QueryConfig:
  QueryParam: '{"index_type":"kHNSW","ef_search":800,"quantized_graph_margin":0.085,"quantized_graph_bucket_width":0.004,"quantized_graph_max_expansions":256,"quantized_graph_route_beam":1}'
  QueryParams:
    - '{"index_type":"kHNSW","ef_search":800,"quantized_graph_margin":0.085,"quantized_graph_bucket_width":0.004,"quantized_graph_max_expansions":256,"quantized_graph_route_beam":1}'
    - '{"index_type":"kHNSW","ef_search":800,"quantized_graph_margin":0.085,"quantized_graph_bucket_width":0.004,"quantized_graph_max_expansions":256,"quantized_graph_route_beam":0}'
```

`QueryParams` is an optional sequence of complete query-parameter JSON strings.
Cases share one opened index and the common external refiner, if configured.
Without it, the existing single `QueryParam` behavior is preserved. Recall logs
for multiple cases go into `case_0`, `case_1`, etc. `BenchRounds` defaults to one;
`BenchThreadCounts` defaults to the existing `BenchThreadCount`. The bench tool
reverses case order on alternate rounds and prints case/round/thread markers.
Its normal query-parameter cloning and shared timing statistics remain in the
measured path, so its QPS should not be mixed with other drivers' measurements.

Queries use `key;float float ...`; ID ground truth uses `key;id id ...`.
For fvecs/ivecs input, convert the file encoding without changing FP32 values,
query order or IDs, and verify the float round trip. Models and measurements
belong outside the source tree.

## Historical diagnostic driver

The following driver is retained for reproducing earlier experiments and
low-level diagnostics; standard-tool QPS should be reported separately.

```sh
cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DBUILD_TOOLS=ON -DBUILD_TESTING=ON \
  -DBUILD_PYTHON_BINDINGS=OFF -DBUILD_C_BINDINGS=OFF
cmake --build "$BUILD_DIR" --target quantized_graph_benchmark \
  quantized_graph_performance_test quantized_graph_router_builder_test -j 8

BENCH_DATASET=gist BENCH_DATA=/path/gist/ BENCH_EFS=800 BENCH_THREADS=1,8 \
BENCH_MARGINS=0.075,0.085,0.095 BENCH_ROUTE_BEAMS=0,1,2 \
BENCH_QUERY_BEGIN=0 BENCH_QUERY_END=500 \
taskset -c 0,2,4,6,8,10,12,14 "$BUILD_DIR/bin/quantized_graph_benchmark" \
  search /path/gist.index /path/gist480.bin 0 3 /local/output/run
```

GIST input files are `gist_query.fvecs` and `gist_groundtruth.ivecs`. For Cohere,
set `BENCH_DATASET=cohere` and supply `cohere_test_vector_1m.1000.new.txt` and
`neighbors.txt`. Both readers validate 1,000 queries and ID-based ground truth.
Use `hnsw` instead of the model for ordinary HNSW, or `qg` for a quantized graph
without an entry model. `SECONDS=-1` evaluates quality only.

Optional grids `BENCH_MARGINS`, `BENCH_ROUTE_BEAMS`, `BENCH_BUCKET_WIDTHS` and
`BENCH_CAPS` run in the same binary and open index; each combination has its own
`_sN` prefix and configuration CSV. Calibration can use queries 0..499; freeze
selected settings before measuring 500..999. Report actual recall alongside QPS
and include controls with routing disabled. Raw outputs, models and reports
belong outside the source tree and are not committed.
