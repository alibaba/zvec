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
#include "scalar/fht/fht.h"
#include "scalar/fp32/cosine.h"
#include "scalar/fp32/inner_product.h"
#include "scalar/fp32/squared_euclidean.h"
#include "scalar/pq_quantizer_int4/pq_distance.h"
#include "scalar/pq_quantizer_int8/pq_distance.h"

#if defined(__SSE2__)
#include "sse/fht/fht.h"
#endif
#if defined(__AVX2__)
#include "avx2/fht/fht.h"
#include "avx2/pq_quantizer_int4/pq_distance.h"
#include "avx2/pq_quantizer_int8/pq_distance.h"
#endif
#if defined(__AVX512F__)
#include "avx512/fht/fht.h"
#include "avx512/pq_quantizer_int4/pq_distance.h"
#include "avx512/pq_quantizer_int8/pq_distance.h"
#endif
#if defined(__ARM_NEON) && defined(__aarch64__)
#include "neon/fht/fht.h"
#endif

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
      if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512_VNNI) {
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
      if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512_VNNI) {
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

FhtKernels get_fht_kernels(CpuArchType cpu_arch_type) {
  // Suppress unused-parameter warning when no SIMD #if blocks are compiled in.
  (void)cpu_arch_type;
  FhtKernels k;
  // Default: scalar fallback for all
  k.flip_sign = scalar::fht_flip_sign;
  k.kacs_walk = scalar::fht_kacs_walk;
  k.inv_kacs_walk = scalar::fht_inv_kacs_walk;
  k.inplace = scalar::fht_inplace;
  k.rescale = scalar::fht_vec_rescale;

#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F &&
      zvec::ailego::internal::CpuFeatures::static_flags_.AVX512DQ &&
      (cpu_arch_type == CpuArchType::kAuto ||
       cpu_arch_type == CpuArchType::kAVX512)) {
    k.flip_sign = avx512::fht_flip_sign_avx512;
    k.kacs_walk = avx512::fht_kacs_walk_avx512;
    k.inv_kacs_walk = avx512::fht_inv_kacs_walk_avx512;
    k.inplace = avx512::fht_inplace_avx512;
    k.rescale = avx512::fht_vec_rescale_avx512;
    return k;
  }
#endif
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2 &&
      (cpu_arch_type == CpuArchType::kAuto ||
       cpu_arch_type == CpuArchType::kAVX2)) {
    k.flip_sign = avx2::fht_flip_sign_avx2;
    k.kacs_walk = avx2::fht_kacs_walk_avx2;
    k.inv_kacs_walk = avx2::fht_inv_kacs_walk_avx2;
    k.inplace = avx2::fht_inplace_avx2;
    k.rescale = avx2::fht_vec_rescale_avx2;
    return k;
  }
#endif
#if defined(__SSE2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.SSE2 &&
      (cpu_arch_type == CpuArchType::kAuto ||
       cpu_arch_type == CpuArchType::kSSE)) {
    k.flip_sign = sse::fht_flip_sign_sse;
    k.kacs_walk = sse::fht_kacs_walk_sse;
    k.inv_kacs_walk = sse::fht_inv_kacs_walk_sse;
    k.rescale = sse::fht_vec_rescale_sse;
    // inplace fallback to scalar (SSE has no fht_inplace)
    return k;
  }
#endif
#if defined(__ARM_NEON) && defined(__aarch64__)
  if (cpu_arch_type == CpuArchType::kAuto ||
      cpu_arch_type == CpuArchType::kNEON) {
    k.flip_sign = neon::fht_flip_sign_neon;
    k.kacs_walk = neon::fht_kacs_walk_neon;
    k.inv_kacs_walk = neon::fht_inv_kacs_walk_neon;
    k.rescale = neon::fht_vec_rescale_neon;
    // inplace fallback to scalar (NEON has no fht_inplace)
    return k;
  }
#endif
  return k;  // scalar
}

PqKernels get_pq_kernels(DataType data_type, QuantizeType quantize_type,
                         CpuArchType cpu_arch_type) {
  // Suppress unused-parameter warning when no SIMD #if blocks are compiled in.
  (void)cpu_arch_type;
  PqKernels k{};
  if (quantize_type == QuantizeType::kPQ) {
    if (data_type == DataType::kInt4) {
      // int4 packed nibble path — scalar by default.
      k.adc_distance = scalar::pq_adc_int4_distance;
      k.sdc_distance = scalar::pq_sdc_int4_distance;
      k.batch_adc_distance = scalar::pq_adc_int4_batch_distance;

#if defined(__AVX512F__)
      if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F &&
          (cpu_arch_type == CpuArchType::kAuto ||
           cpu_arch_type == CpuArchType::kAVX512)) {
        k.adc_distance = avx512::pq_adc_int4_distance_avx512;
        k.sdc_distance = avx512::pq_sdc_int4_distance_avx512;
        k.batch_adc_distance = avx512::pq_adc_int4_batch_distance_avx512;
        return k;
      }
#endif
#if defined(__AVX2__)
      if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2 &&
          (cpu_arch_type == CpuArchType::kAuto ||
           cpu_arch_type == CpuArchType::kAVX2)) {
        k.adc_distance = avx2::pq_adc_int4_distance_avx2;
        k.sdc_distance = avx2::pq_sdc_int4_distance_avx2;
        k.batch_adc_distance = avx2::pq_adc_int4_batch_distance_avx2;
      }
#endif
      return k;
    }
    // Default (kInt8 / fallback): scalar int8 kernels.
    k.adc_distance = scalar::pq_adc_int8_distance;
    k.sdc_distance = scalar::pq_sdc_int8_distance;
    k.batch_adc_distance = scalar::pq_adc_int8_batch_distance;

#if defined(__AVX512F__)
    if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F &&
        (cpu_arch_type == CpuArchType::kAuto ||
         cpu_arch_type == CpuArchType::kAVX512)) {
      k.adc_distance = avx512::pq_adc_int8_distance_avx512;
      k.sdc_distance = avx512::pq_sdc_int8_distance_avx512;
      k.batch_adc_distance = avx512::pq_adc_int8_batch_distance_avx512;
      return k;
    }
#endif
#if defined(__AVX2__)
    if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2 &&
        (cpu_arch_type == CpuArchType::kAuto ||
         cpu_arch_type == CpuArchType::kAVX2)) {
      k.adc_distance = avx2::pq_adc_int8_distance_avx2;
      k.sdc_distance = avx2::pq_sdc_int8_distance_avx2;
      k.batch_adc_distance = avx2::pq_adc_int8_batch_distance_avx2;
    }
#endif
  }
  return k;
}

}  // namespace zvec::turbo