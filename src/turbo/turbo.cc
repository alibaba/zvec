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

#include <ailego/internal/cpu_features.h>
#include <zvec/turbo/turbo.h>
#include "avx512_vnni/record_quantized_int8/cosine.h"
#include "avx512_vnni/record_quantized_int8/squared_euclidean.h"

namespace zvec::turbo {

DistanceFunc get_distance_func(MetricType metric_type, DataType data_type,
                               QuantizeType quantize_type) {
  if (data_type == DataType::kInt8) {
    if (quantize_type == QuantizeType::kDefault) {
      if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512_VNNI) {
        if (metric_type == MetricType::kSquaredEuclidean) {
          return avx512_vnni::squared_euclidean_int8_distance;
        }
        if (metric_type == MetricType::kCosine) {
          return avx512_vnni::cosine_int8_distance;
        }
      }
    }
  }
  return nullptr;
}

BatchDistanceFunc get_batch_distance_func(MetricType metric_type,
                                          DataType data_type,
                                          QuantizeType quantize_type) {
  if (data_type == DataType::kInt8) {
    if (quantize_type == QuantizeType::kDefault) {
      if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512_VNNI) {
        if (metric_type == MetricType::kSquaredEuclidean) {
          return avx512_vnni::squared_euclidean_int8_batch_distance;
        }
        if (metric_type == MetricType::kCosine) {
          return avx512_vnni::cosine_int8_batch_distance;
        }
      }
    }
  }
  return nullptr;
}

QueryPreprocessFunc get_query_preprocess_func(MetricType metric_type,
                                              DataType data_type,
                                              QuantizeType quantize_type) {
  if (data_type == DataType::kInt8) {
    if (quantize_type == QuantizeType::kDefault) {
      if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512_VNNI) {
        if (metric_type == MetricType::kSquaredEuclidean) {
          return avx512_vnni::squared_euclidean_int8_query_preprocess;
        }
        if (metric_type == MetricType::kCosine) {
          return avx512_vnni::cosine_int8_query_preprocess;
        }
      }
    }
  }
  return nullptr;
}

}  // namespace zvec::turbo
