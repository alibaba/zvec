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

#include <zvec/core/framework/index_factory.h>
#include "turbo/quantizer/quantizer.h"

namespace zvec::core {

// Create an independent quantizer for distances over existing training/centroid
// records. These vectors are already converted: do not quantize them again or
// reuse the posting quantizer, which may use a different precision/codebook.
inline turbo::Quantizer::Pointer CreateDistanceQuantizer(
    const IndexMeta &meta) {
  if (meta.meta_type() != IndexMeta::MT_DENSE) return nullptr;
  const auto &metric = meta.metric_name();
  if (metric != "SquaredEuclidean" && metric != "InnerProduct" &&
      metric != "Cosine") {
    return nullptr;
  }
  const bool half = meta.data_type() == IndexMeta::DT_FP16;
  if (!half && meta.data_type() != IndexMeta::DT_FP32) {
    return nullptr;
  }
  const size_t unit = half ? sizeof(ailego::Float16) : sizeof(float);
  size_t dim = meta.dimension();
  if (metric == "Cosine") {
    // Legacy cosine includes the stored float norm in dimension(), whereas
    // Turbo describes the norm separately as extra metadata.
    if (meta.extra_meta_size() == 0) {
      const size_t tail = sizeof(float) / unit;
      if (dim <= tail) return nullptr;
      dim -= tail;
    } else if (meta.extra_meta_size() != sizeof(float)) {
      return nullptr;
    }
  } else if (meta.extra_meta_size() != 0) {
    return nullptr;
  }
  if (dim == 0) return nullptr;
  IndexMeta input;
  input.set_meta(IndexMeta::DT_FP32, dim);
  input.set_metric(metric, meta.metric_revision(), meta.metric_params());
  auto quantizer =
      IndexFactory::CreateQuantizer(half ? "Fp16Quantizer" : "Fp32Quantizer");
  ailego::Params params;
  // Match raw FP16 training arithmetic, including values whose squared
  // differences overflow half precision. Posting quantizers keep their policy.
  if (half) params.set("fp32_accumulation", true);
  if (!quantizer || quantizer->init(input, params) != 0 ||
      quantizer->quantized_datapoint_vector_length() != meta.element_size()) {
    return nullptr;
  }
  return quantizer;
}

}  // namespace zvec::core
