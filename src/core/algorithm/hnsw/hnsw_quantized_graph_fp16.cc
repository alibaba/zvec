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
#include "hnsw_quantized_graph_impl.h"

namespace zvec::core {
#if ZVEC_PROJECTED_QG_SUPPORTED
template <>
struct HnswQuantizedGraph::Model<true> final : Impl {
  using Codec = turbo::SymQGBlockQuantizer;
  Model(size_t dimension, size_t max_neighbors, bool cosine,
        size_t stored_vector_bytes, const std::string &model_data)
      : graph(dimension, Codec(dimension, cosine), max_neighbors,
              stored_vector_bytes, cosine, model_data),
        dimension_(dimension),
        cosine_(cosine) {}
  int prebuild(const HnswEntity &entity, size_t count,
               size_t threads) override {
    return graph.prebuild(HnswFp16GraphAdapter{{entity}, dimension_}, count,
                          threads);
  }
  int search(uint32_t entry_id, HnswContext &ctx) const override {
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
  }
  uint32_t entry() const override {
    return graph.entry();
  }
  QuantizedGraph<Codec> graph;
  size_t dimension_;
  bool cosine_;
};

// Keep each precision's traversal in its own translation unit so adding a
// storage format does not change the compiler's inlining of the FP32 loop.
std::unique_ptr<HnswQuantizedGraph::Impl> HnswQuantizedGraph::make_fp16(
    size_t dimension, size_t max_neighbors, bool cosine,
    size_t stored_vector_bytes, const std::string &model_data) {
  return std::make_unique<Model<true>>(dimension, max_neighbors, cosine,
                                       stored_vector_bytes, model_data);
}
#endif
}  // namespace zvec::core
