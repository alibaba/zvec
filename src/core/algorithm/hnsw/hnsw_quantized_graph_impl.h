// Copyright 2025-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include <array>
#include <ailego/math/normalizer.h>
#include <zvec/ailego/utility/float_helper.h>
#include <zvec/core/framework/index_error.h>
#include "hnsw_context.h"
#include "hnsw_quantized_graph.h"
#if RABITQ_SUPPORTED && defined(__x86_64__) && \
    (defined(__GNUC__) || defined(__clang__))
#define ZVEC_PROJECTED_QG_SUPPORTED 1
#include <turbo/distance/avx512/fp16/fp32_query.h>
#include <turbo/quantizer/symqg_quantizer/symqg_quantizer.h>
#include "algorithm/quantized_graph/quantized_graph.h"
#endif
namespace zvec::core {
struct HnswQuantizedGraph::Impl {
  virtual ~Impl() = default;
  virtual int prebuild(const HnswEntity &, size_t, size_t) = 0;
  virtual int search(uint32_t, HnswContext &) const = 0;
  virtual uint32_t entry() const = 0;
};

#if ZVEC_PROJECTED_QG_SUPPORTED
namespace {
struct HnswGraphAdapter {
  using Vector = IndexStorage::MemoryBlock;
  const HnswEntity &entity;
  bool valid(uint32_t id) const {
    return entity.get_key(id) != kInvalidKey;
  }
  int get_vector(uint32_t id, Vector &vector) const {
    return entity.get_vector(id, vector);
  }
  uint64_t key(uint32_t id) const {
    return entity.get_key(id);
  }
  Neighbors neighbors(uint32_t id) const {
    return entity.get_neighbors(0, id);
  }
};

struct HnswFp16GraphAdapter : HnswGraphAdapter {
  struct Vector {
    IndexStorage::MemoryBlock storage;
    std::array<float, 4096> values;
    const void *data() const {
      return values.data();
    }
  };
  size_t dimension;
  int get_vector(uint32_t id, Vector &vector) const {
    const int ret = entity.get_vector(id, vector.storage);
    if (ret != 0) return ret;
    if (!vector.storage.data()) return IndexError_Mismatch;
    ailego::FloatHelper::ToFP32(
        static_cast<const uint16_t *>(vector.storage.data()), dimension,
        vector.values.data());
    return 0;
  }
};

struct HnswQueryAdapter {
  HnswContext &ctx;
  TopkHeap &reset_results() {
    return ctx.search_heap().reset<TopkHeap>(std::max(ctx.topk(), ctx.ef()));
  }
  const QuantizedGraphSearchOptions &quantized_graph_options() const {
    return ctx.quantized_graph_options();
  }
  uint32_t topk() const {
    return ctx.topk();
  }
  const void *query() const {
    return ctx.dist_calculator().query();
  }
  float distance(const void *vector) {
    return ctx.dist_calculator().dist(vector);
  }
  bool excluded(uint32_t id) const {
    return ctx.filter().is_valid() &&
           ctx.filter()(ctx.get_entity().get_key(id));
  }
  bool reach_scan_limit() const {
    return ctx.reach_scan_limit();
  }
  void on_expand() {
    if (ctx.debugging()) ++(*ctx.mutable_stats_get_neighbors());
  }
};
struct HnswFp16QueryAdapter : HnswQueryAdapter {
  const float *values;
  size_t dimension;
  bool cosine;
  const void *query() const {
    return values;
  }
  float distance(const void *vector) {
    ctx.dist_calculator().record_comparison();
    // Widen in registers after the graph has prefetched the storage batch.
    float score;
    if (cosine)
      turbo::avx512::cosine_fp16_fp32_distance_avx512(vector, values, dimension,
                                                      &score);
    else
      turbo::avx512::squared_euclidean_fp16_fp32_distance_avx512(
          vector, values, dimension, &score);
    return score;
  }
};
}  // namespace

#endif
}  // namespace zvec::core
