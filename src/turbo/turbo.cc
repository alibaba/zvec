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
#include "avx512_vnni/uniform_int8/quantize.h"
#include "avx512_vnni/uniform_int8/squared_euclidean.h"
#include "scalar/fp32/cosine.h"
#include "scalar/fp32/inner_product.h"
#include "scalar/fp32/squared_euclidean.h"

namespace zvec::turbo {

DistanceFunc get_distance_func(MetricType metric_type, DataType data_type,
                               QuantizeType quantize_type,
                               CpuArchType cpu_arch_type) {
  if (data_type == DataType::kFp32) {
    if (quantize_type == QuantizeType::kDefault ||
        quantize_type == QuantizeType::kFp32) {
      if (metric_type == MetricType::kCosine) {
        return scalar::cosine_fp32_distance;
      }
      if (metric_type == MetricType::kSquaredEuclidean) {
        return scalar::squared_euclidean_fp32_distance;
      }
      if (metric_type == MetricType::kInnerProduct) {
        return scalar::inner_product_fp32_distance;
      }
    }
    return nullptr;
  }
  if (data_type == DataType::kInt8) {
    if (quantize_type == QuantizeType::kDefault) {
      if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512_VNNI &&
          (cpu_arch_type == CpuArchType::kAuto ||
           cpu_arch_type == CpuArchType::kAVX512VNNI)) {
        if (metric_type == MetricType::kSquaredEuclidean) {
          return avx512_vnni::squared_euclidean_int8_distance;
        }
        if (metric_type == MetricType::kCosine) {
          return avx512_vnni::cosine_int8_distance;
        }
      }
    }
    if (quantize_type == QuantizeType::kUniform) {
      if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512_VNNI &&
          (cpu_arch_type == CpuArchType::kAuto ||
           cpu_arch_type == CpuArchType::kAVX512VNNI)) {
        if (metric_type == MetricType::kSquaredEuclidean) {
          return avx512_vnni::uniform_squared_euclidean_int8_distance;
        }
      }
    }
  }
  return nullptr;
}

BatchDistanceFunc get_batch_distance_func(MetricType metric_type,
                                          DataType data_type,
                                          QuantizeType quantize_type,
                                          CpuArchType cpu_arch_type) {
  if (data_type == DataType::kFp32) {
    if (quantize_type == QuantizeType::kDefault ||
        quantize_type == QuantizeType::kFp32) {
      if (metric_type == MetricType::kCosine) {
        return scalar::cosine_fp32_batch_distance;
      }
      if (metric_type == MetricType::kSquaredEuclidean) {
        return scalar::squared_euclidean_fp32_batch_distance;
      }
      if (metric_type == MetricType::kInnerProduct) {
        return scalar::inner_product_fp32_batch_distance;
      }
    }
    return nullptr;
  }
  if (data_type == DataType::kInt8) {
    if (quantize_type == QuantizeType::kDefault) {
      if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512_VNNI &&
          (cpu_arch_type == CpuArchType::kAuto ||
           cpu_arch_type == CpuArchType::kAVX512VNNI)) {
        if (metric_type == MetricType::kSquaredEuclidean) {
          return avx512_vnni::squared_euclidean_int8_batch_distance;
        }
        if (metric_type == MetricType::kCosine) {
          return avx512_vnni::cosine_int8_batch_distance;
        }
      }
    }
    if (quantize_type == QuantizeType::kUniform) {
      if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512_VNNI &&
          (cpu_arch_type == CpuArchType::kAuto ||
           cpu_arch_type == CpuArchType::kAVX512VNNI)) {
        if (metric_type == MetricType::kSquaredEuclidean) {
          return avx512_vnni::uniform_squared_euclidean_int8_batch_distance;
        }
      }
    }
  }

  return nullptr;
}

QueryPreprocessFunc get_query_preprocess_func(MetricType metric_type,
                                              DataType data_type,
                                              QuantizeType quantize_type,
                                              CpuArchType cpu_arch_type) {
  if (data_type == DataType::kInt8) {
    if (quantize_type == QuantizeType::kDefault) {
      if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512_VNNI &&
          (cpu_arch_type == CpuArchType::kAuto ||
           cpu_arch_type == CpuArchType::kAVX512VNNI)) {
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

UniformQuantizeFunc get_uniform_quantize_func(DataType data_type) {
  if (data_type == DataType::kInt8) {
    // Quantize uses AVX-512F (no VNNI required), but we gate on the same
    // AVX512_VNNI flag for now since the kernel lives in the avx512_vnni
    // directory and is compiled with the same march flag.
    if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512_VNNI) {
      return avx512_vnni::uniform_int8_quantize;
    }
  }
  return nullptr;
}

}  // namespace zvec::turbo
