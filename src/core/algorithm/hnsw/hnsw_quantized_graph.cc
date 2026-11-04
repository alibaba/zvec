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
#include "hnsw_quantized_graph.h"
#include <array>
#include <sstream>
#include <ailego/math/normalizer.h>
#include <zvec/ailego/utility/float_helper.h>
#include <zvec/core/framework/index_error.h>
#include "hnsw_context.h"
#if RABITQ_SUPPORTED && defined(__x86_64__) && \
    (defined(__GNUC__) || defined(__clang__))
#define ZVEC_PROJECTED_QG_SUPPORTED 1
#include <turbo/distance/avx512/fp16/fp32_query.h>
#include <turbo/quantizer/symqg_quantizer/symqg_quantizer.h>
#include "algorithm/quantized_graph/quantized_graph.h"
#include "algorithm/quantized_graph/quantized_graph_router_builder.h"
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

template <bool Fp16>
struct HnswQuantizedGraph::Model final : Impl {
  using Codec = turbo::SymQGBlockQuantizer;
  Model(size_t dimension, size_t max_neighbors, bool cosine,
        size_t stored_vector_bytes, const std::string &model_data)
      : graph(dimension, Codec(dimension, cosine), max_neighbors,
              stored_vector_bytes, cosine, model_data),
        dimension_(dimension),
        cosine_(cosine) {}
  int prebuild(const HnswEntity &entity, size_t count,
               size_t threads) override {
    if constexpr (Fp16)
      return graph.prebuild(HnswFp16GraphAdapter{{entity}, dimension_}, count,
                            threads);
    else
      return graph.prebuild(HnswGraphAdapter{entity}, count, threads);
  }
  int search(uint32_t entry_id, HnswContext &ctx) const override {
    if constexpr (Fp16) {
      std::array<float, 4096> values;
      const float *query_values = ctx.quantized_graph_fp32_query();
      if (!query_values) {
        // Direct streamer callers supply a stored-format FP16 query.
        ailego::FloatHelper::ToFP32(
            static_cast<const uint16_t *>(ctx.dist_calculator().query()),
            dimension_, values.data());
        query_values = values.data();
      } else if (cosine_) {
        std::copy_n(query_values, dimension_, values.data());
        float norm;
        ailego::Normalizer<float>::L2(values.data(), dimension_, &norm);
        query_values = values.data();
      }
      HnswFp16QueryAdapter query{{ctx}, query_values, dimension_, cosine_};
      return graph.search(entry_id, HnswGraphAdapter{ctx.get_entity()}, query);
    } else {
      HnswQueryAdapter query{ctx};
      return graph.search(entry_id, HnswGraphAdapter{ctx.get_entity()}, query);
    }
  }
  uint32_t entry() const override {
    return graph.entry();
  }
  QuantizedGraph<Codec> graph;
  size_t dimension_;
  bool cosine_;
};

#endif
bool HnswQuantizedGraph::supported() {
#if ZVEC_PROJECTED_QG_SUPPORTED
  return __builtin_cpu_supports("avx512f") &&
         __builtin_cpu_supports("avx512bw") &&
         __builtin_cpu_supports("avx512dq") &&
         __builtin_cpu_supports("avx512vl") &&
         __builtin_cpu_supports("avx512vnni") && __builtin_cpu_supports("fma");
#else
  return false;
#endif
}
HnswQuantizedGraph::HnswQuantizedGraph(size_t dimension, size_t max_neighbors,
                                       bool cosine, size_t stored_vector_bytes,
                                       const std::string &model_data,
                                       bool fp16) {
#if ZVEC_PROJECTED_QG_SUPPORTED
  if (supported() && dimension > 0 && dimension <= 4096 &&
      max_neighbors == 32) {
    if (fp16)
      impl_ = std::make_unique<Model<true>>(dimension, max_neighbors, cosine,
                                            stored_vector_bytes, model_data);
    else
      impl_ = std::make_unique<Model<false>>(dimension, max_neighbors, cosine,
                                             stored_vector_bytes, model_data);
  }
#else
  (void)dimension;
  (void)max_neighbors;
  (void)cosine;
  (void)stored_vector_bytes;
  (void)model_data;
  (void)fp16;
#endif
}
int HnswQuantizedGraph::train_router(const HnswEntity &entity, size_t dimension,
                                     bool cosine, size_t anchors,
                                     size_t projected, uint64_t seed,
                                     std::string &data, bool fp16) {
#if ZVEC_PROJECTED_QG_SUPPORTED
  using Builder = QuantizedGraphRouterBuilder;
  if (!supported()) return IndexError_Unsupported;
  if (!anchors || anchors > Builder::kMaxAnchors || !dimension ||
      dimension > 4096 || !entity.doc_cnt() ||
      entity.doc_cnt() > uint32_t(INT32_MAX))
    return IndexError_InvalidArgument;
  try {
    IndexStorage::MemoryBlock block;
    std::array<float, 4096> decoded;
    const auto training = Builder::collect(
        entity.doc_cnt(), dimension, cosine, anchors, [&](uint32_t id) {
          if (entity.get_key(id) == kInvalidKey ||
              entity.get_vector(id, block) != 0 || !block.data())
            return static_cast<const float *>(nullptr);
          if (fp16) {
            ailego::FloatHelper::ToFP32(
                static_cast<const uint16_t *>(block.data()), dimension,
                decoded.data());
            // FP16 rounding may move the norm outside the FP32 training
            // tolerance. Spherical clustering trains on unit directions.
            if (cosine) {
              float norm;
              ailego::Normalizer<float>::L2(decoded.data(), dimension, &norm);
            }
            return static_cast<const float *>(decoded.data());
          }
          return static_cast<const float *>(block.data());
        });
    Builder::Options options;
    options.projected = projected;
    options.seed = seed;
    const auto model = Builder::train(training, options);
    std::ostringstream out(std::ios::out | std::ios::binary);
    model.write(out);
    data = out.str();
    return 0;
  } catch (const std::domain_error &e) {
    LOG_ERROR("Invalid router training data: %s", e.what());
    return IndexError_Mismatch;
  } catch (const std::exception &e) {
    LOG_ERROR("Router training failed: %s", e.what());
    return IndexError_Runtime;
  }
#else
  (void)entity;
  (void)dimension;
  (void)cosine;
  (void)anchors;
  (void)projected;
  (void)seed;
  (void)data;
  (void)fp16;
  return IndexError_Unsupported;
#endif
}
int HnswQuantizedGraph::prebuild(const HnswEntity &entity, size_t count,
                                 size_t threads) {
  return impl_ ? impl_->prebuild(entity, count, threads)
               : IndexError_Unsupported;
}
int HnswQuantizedGraph::search(uint32_t entry_id, HnswContext &ctx) const {
  return impl_ ? impl_->search(entry_id, ctx) : IndexError_Unsupported;
}
uint32_t HnswQuantizedGraph::entry() const {
  return impl_ ? impl_->entry() : UINT32_MAX;
}
HnswQuantizedGraph::~HnswQuantizedGraph() = default;
}  // namespace zvec::core
