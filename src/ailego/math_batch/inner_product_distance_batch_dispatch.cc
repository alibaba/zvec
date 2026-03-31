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
#include <ailego/math/inner_product_matrix.h>
#include <ailego/utility/math_helper.h>
#include <zvec/ailego/internal/platform.h>
#include <zvec/ailego/math_batch/utils.h>
#include <zvec/ailego/utility/type_helper.h>
#include "inner_product_distance_batch.h"

namespace zvec::ailego::DistanceBatch {

#if defined(__AVX512VNNI__)
void compute_one_to_many_inner_product_avx512_vnni_int8_query_preprocess(
    void *query, size_t dim);

void compute_one_to_many_inner_product_avx512_vnni_int8_1(
    const int8_t *query, const int8_t **ptrs,
    std::array<const int8_t *, 1> &prefetch_ptrs, size_t dimensionality,
    float *results);

void compute_one_to_many_inner_product_avx512_vnni_int8_12(
    const int8_t *query, const int8_t **ptrs,
    std::array<const int8_t *, 12> &prefetch_ptrs, size_t dimensionality,
    float *results);
#endif

#if defined(__AVX512FP16__)
void compute_one_to_many_inner_product_avx512fp16_fp16_1(
    const ailego::Float16 *query, const ailego::Float16 **ptrs,
    std::array<const ailego::Float16 *, 1> &prefetch_ptrs,
    size_t dimensionality, float *results);

void compute_one_to_many_inner_product_avx512fp16_fp16_12(
    const ailego::Float16 *query, const ailego::Float16 **ptrs,
    std::array<const ailego::Float16 *, 12> &prefetch_ptrs,
    size_t dimensionality, float *results);
#endif  //__AVX512FP16__

#if defined(__AVX512F__)
void compute_one_to_many_inner_product_avx512f_fp16_1(
    const ailego::Float16 *query, const ailego::Float16 **ptrs,
    std::array<const ailego::Float16 *, 1> &prefetch_ptrs,
    size_t dimensionality, float *results);

void compute_one_to_many_inner_product_avx512f_fp16_12(
    const ailego::Float16 *query, const ailego::Float16 **ptrs,
    std::array<const ailego::Float16 *, 12> &prefetch_ptrs,
    size_t dimensionality, float *results);
#endif  //__AVX512F__

#if defined(__AVX2__)
void compute_one_to_many_inner_product_avx2_fp32_1(
    const float *query, const float **ptrs,
    std::array<const float *, 1> &prefetch_ptrs, size_t dimensionality,
    float *results);

void compute_one_to_many_inner_product_avx2_fp16_1(
    const ailego::Float16 *query, const ailego::Float16 **ptrs,
    std::array<const ailego::Float16 *, 1> &prefetch_ptrs,
    size_t dimensionality, float *results);

void compute_one_to_many_inner_product_avx2_int8_1(
    const int8_t *query, const int8_t **ptrs,
    std::array<const int8_t *, 1> &prefetch_ptrs, size_t dimensionality,
    float *results);

void compute_one_to_many_inner_product_avx2_fp32_12(
    const float *query, const float **ptrs,
    std::array<const float *, 12> &prefetch_ptrs, size_t dimensionality,
    float *results);

void compute_one_to_many_inner_product_avx2_fp16_12(
    const ailego::Float16 *query, const ailego::Float16 **ptrs,
    std::array<const ailego::Float16 *, 12> &prefetch_ptrs,
    size_t dimensionality, float *results);

void compute_one_to_many_inner_product_avx2_int8_12(
    const int8_t *query, const int8_t **ptrs,
    std::array<const int8_t *, 12> &prefetch_ptrs, size_t dimensionality,
    float *results);
#endif

void InnerProductDistanceBatchImpl<float, 1>::compute_one_to_many(
    const ValueType *query, const ValueType **ptrs,
    std::array<const ValueType *, 1> &prefetch_ptrs, size_t dim, float *sums) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    return compute_one_to_many_inner_product_avx2_fp32_1(
        query, ptrs, prefetch_ptrs, dim, sums);
  }
#endif
  return compute_one_to_many_inner_product_fallback(query, ptrs, prefetch_ptrs,
                                                    dim, sums);
}

void InnerProductDistanceBatchImpl<ailego::Float16, 1>::compute_one_to_many(
    const ailego::Float16 *query, const ailego::Float16 **ptrs,
    std::array<const ailego::Float16 *, 1> &prefetch_ptrs, size_t dim,
    float *sums) {
#if defined(__AVX512FP16__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512_FP16) {
    return compute_one_to_many_inner_product_avx512fp16_fp16_1(
        query, ptrs, prefetch_ptrs, dim, sums);
  }
#endif
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    return compute_one_to_many_inner_product_avx512f_fp16_1(
        query, ptrs, prefetch_ptrs, dim, sums);
  }
#endif
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    return compute_one_to_many_inner_product_avx2_fp16_1(
        query, ptrs, prefetch_ptrs, dim, sums);
  }
#endif
  return compute_one_to_many_inner_product_fallback(query, ptrs, prefetch_ptrs,
                                                    dim, sums);
}

void InnerProductDistanceBatchImpl<int8_t, 1>::compute_one_to_many(
    const int8_t *query, const int8_t **ptrs,
    std::array<const int8_t *, 1> &prefetch_ptrs, size_t dim, float *sums) {
// #if defined(__AVX512BW__) // TODO: this version is problematic
//     return compute_one_to_many_avx512_int8<ValueType, BatchSize>(
//         query, ptrs, prefetch_ptrs, dim, sums);
#if defined(__AVX512VNNI__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512_VNNI) {
    return compute_one_to_many_inner_product_avx512_vnni_int8_1(
        query, ptrs, prefetch_ptrs, dim, sums);
  }
#endif
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    return compute_one_to_many_inner_product_avx2_int8_1(
        query, ptrs, prefetch_ptrs, dim, sums);
  }
#endif
  return compute_one_to_many_inner_product_fallback(query, ptrs, prefetch_ptrs,
                                                    dim, sums);
}

DistanceBatchQueryPreprocessFunc
InnerProductDistanceBatchImpl<int8_t, 1>::GetQueryPreprocessFunc() {
#if defined(__AVX512VNNI__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512_VNNI) {
    return compute_one_to_many_inner_product_avx512_vnni_int8_query_preprocess;
  }
#endif
  return nullptr;
}

void InnerProductDistanceBatchImpl<float, 12>::compute_one_to_many(
    const ValueType *query, const ValueType **ptrs,
    std::array<const ValueType *, 12> &prefetch_ptrs, size_t dim, float *sums) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    return compute_one_to_many_inner_product_avx2_fp32_12(
        query, ptrs, prefetch_ptrs, dim, sums);
  }
#endif
  return compute_one_to_many_inner_product_fallback(query, ptrs, prefetch_ptrs,
                                                    dim, sums);
}

void InnerProductDistanceBatchImpl<ailego::Float16, 12>::compute_one_to_many(
    const ailego::Float16 *query, const ailego::Float16 **ptrs,
    std::array<const ailego::Float16 *, 12> &prefetch_ptrs, size_t dim,
    float *sums) {
#if defined(__AVX512FP16__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512_FP16) {
    return compute_one_to_many_inner_product_avx512fp16_fp16_12(
        query, ptrs, prefetch_ptrs, dim, sums);
  }
#endif
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    return compute_one_to_many_inner_product_avx512f_fp16_12(
        query, ptrs, prefetch_ptrs, dim, sums);
  }
#endif
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    return compute_one_to_many_inner_product_avx2_fp16_12(
        query, ptrs, prefetch_ptrs, dim, sums);
  }
#endif
  return compute_one_to_many_inner_product_fallback(query, ptrs, prefetch_ptrs,
                                                    dim, sums);
}

void InnerProductDistanceBatchImpl<int8_t, 12>::compute_one_to_many(
    const int8_t *query, const int8_t **ptrs,
    std::array<const int8_t *, 12> &prefetch_ptrs, size_t dim, float *sums) {
// #if defined(__AVX512BW__) // TODO: this version is problematic
//     return compute_one_to_many_avx512_int8<ValueType, BatchSize>(
//         query, ptrs, prefetch_ptrs, dim, sums);
#if defined(__AVX512VNNI__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512_VNNI) {
    return compute_one_to_many_inner_product_avx512_vnni_int8_12(
        query, ptrs, prefetch_ptrs, dim, sums);
  }
#endif
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    return compute_one_to_many_inner_product_avx2_int8_12(
        query, ptrs, prefetch_ptrs, dim, sums);
  }
#endif
  return compute_one_to_many_inner_product_fallback(query, ptrs, prefetch_ptrs,
                                                    dim, sums);
}

}  // namespace zvec::ailego::DistanceBatch
