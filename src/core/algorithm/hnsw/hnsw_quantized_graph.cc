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
#include <sstream>
#include <zvec/core/framework/index_error.h>
#include "hnsw_context.h"
#if RABITQ_SUPPORTED && defined(__x86_64__) && \
    (defined(__GNUC__) || defined(__clang__))
#define ZVEC_PROJECTED_QG_SUPPORTED 1
#include "algorithm/quantized_graph/quantized_graph.h"
#include "algorithm/quantized_graph/quantized_graph_router_builder.h"
#include "algorithm/quantized_graph/symphony_qg_codec.h"
#endif
namespace zvec::core {
struct HnswQuantizedGraph::Impl {
  virtual ~Impl() = default;
  virtual int prebuild(const HnswEntity &, size_t, size_t) = 0;
  virtual int search(uint32_t, HnswContext &, size_t) const = 0;
  virtual void clear() = 0;
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
}  // namespace

template <class Codec>
struct HnswQuantizedGraph::Model final : Impl {
  Model(size_t dimension, size_t max_neighbors, bool cosine,
        size_t stored_dimension, const std::string &model_path,
        const std::string &model_data)
      : graph(dimension, Codec(dimension, cosine), max_neighbors,
              stored_dimension, cosine, model_path, model_data) {}
  int prebuild(const HnswEntity &entity, size_t count,
               size_t threads) override {
    int ret = graph.prebuild(HnswGraphAdapter{entity}, count, threads);
    return ret;
  }
  int search(uint32_t entry_id, HnswContext &ctx, size_t) const override {
    HnswQueryAdapter query{ctx};
    return graph.search(entry_id, HnswGraphAdapter{ctx.get_entity()}, query);
  }
  void clear() override {
    graph.clear();
  }
  uint32_t entry() const override {
    return graph.entry();
  }
  QuantizedGraph<Codec> graph;
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
                                       bool cosine, size_t stored_dimension,
                                       const std::string &model_path,
                                       const std::string &model_data) {
#if ZVEC_PROJECTED_QG_SUPPORTED
  if (supported() && dimension > 0 && dimension <= 4096 &&
      max_neighbors == 32) {
    impl_ = std::make_unique<Model<SymphonyQGCodec>>(dimension, max_neighbors,
                                                     cosine, stored_dimension,
                                                     model_path, model_data);
  }
#endif
}
int HnswQuantizedGraph::train_router(const HnswEntity &entity, size_t dimension,
                                     bool cosine, size_t anchors,
                                     size_t projected, uint64_t seed,
                                     std::string &data) {
#if ZVEC_PROJECTED_QG_SUPPORTED
  using Builder = QuantizedGraphRouterBuilder;
  if (!supported()) return IndexError_Unsupported;
  if (!anchors || anchors > Builder::kMaxAnchors || !dimension ||
      dimension > 4096 || !entity.doc_cnt() ||
      entity.doc_cnt() > uint32_t(INT32_MAX))
    return IndexError_InvalidArgument;
  try {
    Builder::Data training;
    training.docs = entity.doc_cnt();
    training.cosine = cosine;
    training.mean.assign(dimension, 0);
    anchors = std::min(anchors, size_t(training.docs));
    training.ids.resize(anchors);
    training.vectors.assign(anchors, Builder::Vector(dimension));
    uint64_t stride = 2654435761ULL;
    while (std::gcd(stride, uint64_t(training.docs)) != 1) ++stride;
    std::vector<int32_t> selected(training.docs, -1);
    for (size_t i = 0; i < anchors; ++i) {
      const uint32_t id = (i * stride + 7193) % training.docs;
      training.ids[i] = id;
      selected[id] = i;
    }
    std::vector<double> sum(dimension);
    for (uint32_t id = 0; id < training.docs; ++id) {
      IndexStorage::MemoryBlock block;
      if (entity.get_key(id) == kInvalidKey ||
          entity.get_vector(id, block) != 0 || !block.data())
        return IndexError_Mismatch;
      const auto *v = static_cast<const float *>(block.data());
      double norm = 0;
      for (size_t d = 0; d < dimension; ++d) {
        if (!std::isfinite(v[d])) return IndexError_Mismatch;
        sum[d] += v[d];
        norm += double(v[d]) * v[d];
      }
      if (cosine && std::abs(std::sqrt(norm) - 1.) > 1e-4)
        return IndexError_Mismatch;
      if (selected[id] >= 0)
        std::copy_n(v, dimension, training.vectors[selected[id]].begin());
    }
    for (size_t d = 0; d < dimension; ++d)
      training.mean[d] = float(sum[d] / training.docs);
    Builder::Options options;
    options.projected = projected;
    options.seed = seed;
    const auto model = Builder::train(training, options);
    std::ostringstream out(std::ios::out | std::ios::binary);
    model.write(out);
    data = out.str();
    return 0;
  } catch (const std::exception &e) {
    LOG_ERROR("Router training failed: %s", e.what());
    return IndexError_Runtime;
  }
#else
  return IndexError_Unsupported;
#endif
}
int HnswQuantizedGraph::prebuild(const HnswEntity &entity, size_t count,
                                 size_t threads) {
  return impl_ ? impl_->prebuild(entity, count, threads)
               : IndexError_Unsupported;
}
int HnswQuantizedGraph::search(uint32_t entry_id, HnswContext &ctx) const {
  return search(entry_id, ctx, ctx.get_entity().doc_cnt());
}
int HnswQuantizedGraph::search(uint32_t entry_id, HnswContext &ctx,
                               size_t doc_cnt) const {
  return impl_ ? impl_->search(entry_id, ctx, doc_cnt) : IndexError_Unsupported;
}
void HnswQuantizedGraph::clear() {
  if (impl_) impl_->clear();
}
uint32_t HnswQuantizedGraph::entry() const {
  return impl_ ? impl_->entry() : UINT32_MAX;
}
HnswQuantizedGraph::~HnswQuantizedGraph() = default;
}  // namespace zvec::core
