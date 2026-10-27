# Projected-entry SymphonyQG snapshot

This preset extracts the measured Cohere `proj60_384` path from the research
branch. It is an opt-in core C++ search mode for an existing, immutable HNSW
snapshot: 768-dimensional inline FP32 vectors, cosine distance, and a matching
projected-router model. Ordinary HNSW remains the default. Mutable collection
and Python configuration are not extended by this change.

For the mathematical derivation, search pipeline, and limits of the guarantees,
see the [Chinese algorithm article](quantized_graph_principles.zh-CN.md).

## Execution

1. Normalize the query through the existing cosine pipeline. Apply the seeded
   orthogonal 768-dimensional `H12 ⊗ H64` rotation.
2. Use the first 384 rotated coordinates for the database-trained integer router.
   Its 16-way hierarchy selects one **level-zero node**, replacing HNSW's upper
   layer entry descent. This projection only affects entry selection.
3. Search the derived degree-32 graph using full 768-dimensional node-centered
   RaBitQ blocks. Keep existing links for short rows and augment them with
   two-hop candidates. Diversity-prune longer rows.
4. Estimate centers with shifted 8-bit integer dot products and use an adaptive
   bucket frontier (width `0.016 × entry estimate`, margin `0.02`, minimum 10
   expansions, cap 64 after enough eligible results).
5. Order expanded candidates by conservative distance bounds and refine in
   batches of four against the original full-dimensional FP32 vectors.

**Refinement is retained.** Bounds are used to avoid unnecessary final distance
computations within the expanded candidate set. Projected routing, RaBitQ
navigation and adaptive stopping are approximate; they do not guarantee global
recall. The measured operating point is Top10 with `ef_search = 800`.

The graph arena and model are constructed/loaded on open. They are not persisted
into the original HNSW file. Open/prebuild time is excluded from query QPS and
must be budgeted separately. The implementation currently uses eight prebuild
workers. Queries may run concurrently; insertion is rejected while this preset
is enabled. Reopen with an empty model path to modify the original index, then
rebuild its matching router before re-enabling the preset.

## Core API

```cpp
using namespace zvec::core_interface;
HNSWIndexParam param(MetricType::kCosine, 768, 50, 500);
param.data_type = DataType::DT_FP32;
param.quantized_graph_model = "/absolute/path/router384.bin";
auto index = IndexFactory::CreateAndInitIndex(param);
// Open the corresponding existing HNSW snapshot with read_only = true.
// Query with HNSWQueryParam { topk = 10, ef_search = 800 }.
```

The builder also exposes `with_quantized_graph_model(path)`; JSON uses
`quantized_graph_model`. At the low-level streamer interface the key is
`proxima.hnsw.quantized_graph.model`. There are no production `QG_*` environment
variables or compile-time experiment switches.

A RaBitQ-enabled GCC/Clang x86-64 build and AVX512F/BW/DQ/VL/VNNI plus FMA are
required for this preset. Unsupported configurations fail explicitly. Other
platforms can still use ordinary HNSW.

## Model compatibility

### Build a router in C++

With `BUILD_TOOLS=ON`, build the standalone offline trainer:

```sh
cmake --build "$BUILD_DIR" --target quantized_graph_router -j 8
"$BUILD_DIR/bin/quantized_graph_router" build \
  /path/cohere.index /path/router384.bin
```

This reads the existing inline FP32 768D cosine snapshot in read-only mode,
computes its full-corpus mean, samples up to 65,536 internal node IDs, and trains
a 16-way spherical k-means hierarchy with ten iterations per split. It then
rotates/projects the centers to 384 dimensions, quantizes them, and writes the
version-2 model used directly by `quantized_graph_model`. Training, conversion,
and query execution require no Python or NumPy. The source index is unchanged.
The output directory must already exist; an existing output is never replaced.
Incomplete output is not published.

Optional positional arguments are the anchor count (1–65,536, capped at the
document count) and seed (default `202610261259`):

```sh
"$BUILD_DIR/bin/quantized_graph_router" build \
  /path/cohere.index /path/router384.bin 65536 202610261259
```

The tool currently requires readable, nonzero normalized vectors at every
internal ID in the snapshot. It does not add mutable-index training or a Python
collection API. Training is single-threaded and uses native dispatched FP32 dot
products, FP64 center accumulation, and `std::mt19937_64` initialization. Repeated
runs with the same input/seed/build are deterministic; NumPy PCG64/BLAS and other
SIMD/build combinations need not produce the same tree. Degenerate splits use
balanced partitions, with a depth guard to satisfy the loader's depth limit.

For migration checks, the tool also accepts the historical anchor export and
mean, keeping those training inputs fixed:

```sh
"$BUILD_DIR/bin/quantized_graph_router" train \
  /path/anchors.bin /path/mean.f32 /path/router384.bin 202610261259
```

`anchors.bin` is little-endian: five uint32 values (magic `0x51474159`, version 1,
anchor count, dimension 768, graph document count), followed by packed records
of uint32 internal ID, uint64 external key, and 768 FP32 coordinates.
`mean.f32` contains exactly 768 FP32 values. IDs must be unique and in range;
vectors must be finite and normalized. External keys are not substituted for
internal IDs. The caller must ensure these files belong to the intended graph.
The historical mean was estimated from 65,536 database samples, while `build`
computes the full-corpus mean: use `train` when isolating the trainer migration
from this input difference. Neither mode reads queries or ground truth.

### Load and pair the model

The external model is the little-endian projected-router version 2 format:
magic `0x51475259`, projected dimension 384, graph document count, original
768D mean, transform tag `0x53514701`, rotated 768D mean, and 16-way quantized
routing groups. The loader validates shape, finite parameters, tree edges,
leaf IDs, rotated mean, truncation and trailing bytes before building the arena.
No separate mean file is needed.

**The model must belong to the exact graph's internal ID mapping.** Matching
counts and valid leaf IDs alone cannot prove dataset identity; this format does
not contain a cryptographic graph fingerprint. Keep the model and graph together
and verify their hashes in deployment/benchmark manifests. The existing Cohere
model was trained from 65,536 database anchors without query/ground-truth inputs.
Changing its graph or rebuilding insertion order requires a matching model.

GIST is 960D L2 and is deliberately rejected by this Cohere preset. The separate
GIST routing experiment tests a 1024D padded rotation with 480/512/1024 retained
coordinates; its outcome does not silently broaden this API's supported inputs.

## Reproduce the benchmark

```sh
cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DBUILD_TOOLS=ON -DBUILD_TESTING=ON \
  -DBUILD_PYTHON_BINDINGS=OFF -DBUILD_C_BINDINGS=OFF
cmake --build "$BUILD_DIR" --target quantized_graph_benchmark \
  quantized_graph_performance_test -j 8

BENCH_DATA=/path/to/cohere/1m/ BENCH_EFS=800 BENCH_THREADS=1,8 \
BENCH_ROUNDS=3 BENCH_LABEL=projected384 \
taskset -c 0,2,4,6,8,10,12,14 "$BUILD_DIR/bin/quantized_graph_benchmark" \
  search /path/cohere.index /path/router384.bin 0 3 /local/output/run
```

The dataset directory must contain `cohere_test_vector_1m.1000.new.txt` and
`neighbors.txt`. The driver checks all 1,000 queries, writes returned IDs and
hexadecimal FP32 scores, then times synchronous individual searches after
per-thread warmup. `hnsw` in place of the model path selects ordinary HNSW;
`SECONDS=-1` performs correctness evaluation only. All output paths must be
outside the source tree. Raw results and binary models are not committed.
