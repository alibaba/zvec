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
#include <ailego/math/euclidean_distance_matrix.h>
#include <ailego/math/inner_product_matrix.h>
#include <ailego/math/mips_euclidean_distance_matrix.h>
#include <ailego/math/norm2_matrix.h>
#include <ailego/math_batch/distance_batch.h>
#include "quantized_integer_metric_matrix.h"

namespace zvec::core {

template <typename T, size_t BatchSize, size_t PrefetchStep>
struct MinusInnerProductDistanceBatchWithScoreUnquantized;

template <typename T, size_t BatchSize, size_t PrefetchStep>
struct CosineMinusInnerProductDistanceBatchWithScoreUnquantized;

template <typename T, size_t BatchSize, size_t PrefetchStep>
struct SquaredEuclideanDistanceBatchWithScoreUnquantized;

template <typename T, size_t BatchSize, size_t PrefetchStep>
struct MipsSquaredEuclideanDistanceBatchWithScoreUnquantized;


template <template <typename, size_t, size_t> class DistanceType,
          typename ValueType, size_t BatchSize, size_t PrefetchStep,
          typename = void>
struct BaseDistanceBatchWithScoreUnquantized {
  static inline void _ComputeBatch(const ValueType **m, const ValueType *q,
                                   size_t num, size_t dim, float *out) {
    for (size_t i = 0; i < num; ++i) {
      DistanceType<ValueType, 1, 1>::Compute(m[i], q, dim, out + i);
    }
  }

  // If Distance has ComputeBatch, use it; otherwise fall back to _ComputeBatch.
  static inline void ComputeBatch(const ValueType **m, const ValueType *q,
                                  size_t num, size_t dim, float *out) {
    // if constexpr (detail::HasComputeBatch<Distance, ValueType>::value) {
    //   return Distance::ComputeBatch(m, q, num, dim, out);
    // }
    if constexpr (std::is_same_v<DistanceType<ValueType, 1, 1>,
                                 CosineMinusInnerProduct<ValueType, 1, 1>>) {
      return CosineMinusInnerProductDistanceBatchWithScoreUnquantized<
          ValueType, BatchSize, PrefetchStep>::ComputeBatch(m, q, num, dim,
                                                            out);
    } else if constexpr (std::is_same_v<DistanceType<ValueType, 1, 1>,
                                        SquaredEuclidean<ValueType, 1, 1>>) {
      return SquaredEuclideanDistanceBatchWithScoreUnquantized<
          ValueType, BatchSize, PrefetchStep>::ComputeBatch(m, q, num, dim,
                                                            out);
    }

    _ComputeBatch(m, q, num, dim, out);
  }
};

//===========================================================
// CosineMinusInnerProductDistanceBatchWithScoreUnquantized
//===========================================================

// Compute CosineMinusInnerProduct for quantized INT8
template <size_t BatchSize, size_t PrefetchStep>
struct CosineMinusInnerProductDistanceBatchWithScoreUnquantized<
    int8_t, BatchSize, PrefetchStep> {
  using ImplType =
      MinusInnerProductDistanceBatchWithScoreUnquantized<int8_t, BatchSize,
                                                         PrefetchStep>;

  static inline void ComputeBatch(const int8_t **vecs, const int8_t *query,
                                  size_t num_vecs, size_t dim, float *results) {
    size_t original_dim = dim - 24;

    ImplType::ComputeBatch(vecs, query, num_vecs, original_dim, results);
  }

  static ailego::DistanceBatch::DistanceBatchQueryPreprocessFunc
  GetQueryPreprocessFunc() {
    return QueryPreprocess;
  }

  static void QueryPreprocess(void *query, size_t dim) {
    if (auto func = ImplType::GetQueryPreprocessFunc(); func != nullptr) {
      return func(query, dim - 24);
    }
  }
};

// Compute CosineMinusInnerProduct for quantized INT4
template <size_t BatchSize, size_t PrefetchStep>
struct CosineMinusInnerProductDistanceBatchWithScoreUnquantized<
    uint8_t, BatchSize, PrefetchStep> {
  static inline void ComputeBatch(const uint8_t **vecs, const uint8_t *query,
                                  size_t num_vecs, size_t dim, float *results) {
    size_t original_dim = dim - 40;
    MinusInnerProductDistanceBatchWithScoreUnquantized<
        uint8_t, BatchSize, PrefetchStep>::ComputeBatch(vecs, query, num_vecs,
                                                        original_dim, results);
  }
};

//===========================================================
// MinusInnerProductDistanceBatchWithScoreUnquantized
//===========================================================

// Compute MinusInnerProduct for quantized INT8
template <size_t BatchSize, size_t PrefetchStep>
struct MinusInnerProductDistanceBatchWithScoreUnquantized<int8_t, BatchSize,
                                                          PrefetchStep> {
  using ImplType =
      ailego::DistanceBatch::InnerProductDistanceBatch<int8_t, BatchSize,
                                                       PrefetchStep>;
  static inline void ComputeBatch(const int8_t **vecs, const int8_t *query,
                                  size_t num_vecs, size_t dim, float *results) {
    const size_t original_dim = dim;
    ImplType::ComputeBatch(vecs, query, num_vecs, original_dim, results);

    const float *q_tail = reinterpret_cast<const float *>(
        reinterpret_cast<const uint8_t *>(query) + original_dim);
    float qa = q_tail[0];
    float qb = q_tail[1];
    float qs = q_tail[2];
    for (size_t i = 0; i < num_vecs; ++i) {
      const float *m_tail = reinterpret_cast<const float *>(
          reinterpret_cast<const uint8_t *>(vecs[i]) + original_dim);
      float ma = m_tail[0];
      float mb = m_tail[1];
      float ms = m_tail[2];
      float &result = results[i];
      if (ImplType::GetQueryPreprocessFunc() != nullptr) {
        int int_sum = reinterpret_cast<const int *>(m_tail)[4];
        result -= 128 * int_sum;
      }
      result = -(ma * qa * result + mb * qa * qs + qb * ma * ms +
                 original_dim * qb * mb);
    }
  }

  static ailego::DistanceBatch::DistanceBatchQueryPreprocessFunc
  GetQueryPreprocessFunc() {
    return ImplType::GetQueryPreprocessFunc();
  }
};

// Compute MinusInnerProduct for quantized INT4
template <size_t BatchSize, size_t PrefetchStep>
struct MinusInnerProductDistanceBatchWithScoreUnquantized<uint8_t, BatchSize,
                                                          PrefetchStep> {
  static inline void ComputeBatch(const uint8_t **vecs, const uint8_t *query,
                                  size_t num_vecs, size_t dim, float *results) {
    const size_t original_dim = dim;
    const size_t original_dim_in_uint8_array = original_dim >> 1;

    ailego::DistanceBatch::InnerProductDistanceBatch<
        uint8_t, BatchSize, PrefetchStep>::ComputeBatch(vecs, query, num_vecs,
                                                        original_dim, results);
    const float *q_tail = reinterpret_cast<const float *>(
        reinterpret_cast<const uint8_t *>(query) + original_dim_in_uint8_array);
    float qa = q_tail[0];
    float qb = q_tail[1];
    float qs = q_tail[2];
    for (size_t i = 0; i < num_vecs; ++i) {
      const float *m_tail = reinterpret_cast<const float *>(
          reinterpret_cast<const uint8_t *>(vecs[i]) +
          original_dim_in_uint8_array);
      float ma = m_tail[0];
      float mb = m_tail[1];
      float ms = m_tail[2];
      float &result = results[i];
      result = -(ma * qa * result + mb * qa * qs + qb * ma * ms +
                 original_dim * qb * mb);
    }
  }
};

//===========================================================
// SquaredEuclideanDistanceBatchWithScoreUnquantized
//===========================================================

// Compute SquaredEuclidean for quantized INT8
template <size_t BatchSize, size_t PrefetchStep>
struct SquaredEuclideanDistanceBatchWithScoreUnquantized<int8_t, BatchSize,
                                                         PrefetchStep> {
  using ImplType =
      ailego::DistanceBatch::InnerProductDistanceBatch<int8_t, BatchSize,
                                                       PrefetchStep>;
  static void ComputeBatch(const int8_t **vecs, const int8_t *query,
                           size_t num_vecs, size_t dim, float *results) {
    const size_t original_dim = dim - 20;
    ailego::DistanceBatch::InnerProductDistanceBatch<
        int8_t, BatchSize, PrefetchStep>::ComputeBatch(vecs, query, num_vecs,
                                                       original_dim, results);

    const float *q_tail = reinterpret_cast<const float *>(
        reinterpret_cast<const uint8_t *>(query) + original_dim);
    float qa = q_tail[0];
    float qb = q_tail[1];
    float qs = q_tail[2];
    float qs2 = q_tail[3];

    const float sum = qa * qs;
    const float sum2 = qa * qa * qs2;
    for (size_t i = 0; i < num_vecs; ++i) {
      const float *m_tail = reinterpret_cast<const float *>(
          reinterpret_cast<const uint8_t *>(vecs[i]) + original_dim);
      float ma = m_tail[0];
      float mb = m_tail[1];
      float ms = m_tail[2];
      float ms2 = m_tail[3];
      float &result = results[i];
      if (ImplType::GetQueryPreprocessFunc() != nullptr) {
        int int8_sum = reinterpret_cast<const int *>(m_tail)[4];
        result -= 128 * int8_sum;
      }
      result = ma * ma * ms2 + sum2 - 2 * ma * qa * result +
               (mb - qb) * (mb - qb) * original_dim +
               2 * (mb - qb) * (ms * ma - sum);
    }
  }

  static ailego::DistanceBatch::DistanceBatchQueryPreprocessFunc
  GetQueryPreprocessFunc() {
    return QueryPreprocess;
  }

  static void QueryPreprocess(void *query, size_t dim) {
    if (auto func = ImplType::GetQueryPreprocessFunc(); func != nullptr) {
      return func(query, dim - 20);
    }
  }
};

// Compute SquaredEuclidean for quantized INT4
template <size_t BatchSize, size_t PrefetchStep>
struct SquaredEuclideanDistanceBatchWithScoreUnquantized<uint8_t, BatchSize,
                                                         PrefetchStep> {
  static void ComputeBatch(const uint8_t **vecs, const uint8_t *query,
                           size_t num_vecs, size_t dim, float *results) {
    const size_t original_dim = dim - 32;
    const size_t original_dim_in_uint8_array = original_dim >> 1;
    ailego::DistanceBatch::InnerProductDistanceBatch<
        uint8_t, BatchSize, PrefetchStep>::ComputeBatch(vecs, query, num_vecs,
                                                        original_dim, results);

    const float *q_tail = reinterpret_cast<const float *>(
        reinterpret_cast<const uint8_t *>(query) + original_dim_in_uint8_array);
    float qa = q_tail[0];
    float qb = q_tail[1];
    float qs = q_tail[2];
    float qs2 = q_tail[3];

    const float sum = qa * qs;
    const float sum2 = qa * qa * qs2;
    for (size_t i = 0; i < num_vecs; ++i) {
      const float *m_tail = reinterpret_cast<const float *>(
          reinterpret_cast<const uint8_t *>(vecs[i]) +
          original_dim_in_uint8_array);
      float ma = m_tail[0];
      float mb = m_tail[1];
      float ms = m_tail[2];
      float ms2 = m_tail[3];
      *results = ma * ma * ms2 + sum2 - 2 * ma * qa * *results +
                 (mb - qb) * (mb - qb) * original_dim +
                 2 * (mb - qb) * (ms * ma - sum);
      ++results;
    }
  }
};


//===========================================================
// MipsSquaredEuclideanDistanceBatchWithScoreUnquantized
//===========================================================

// Compute MipsSquaredEuclidean for quantized INT8
template <size_t BatchSize, size_t PrefetchStep>
struct MipsSquaredEuclideanDistanceBatchWithScoreUnquantized<int8_t, BatchSize,
                                                             PrefetchStep> {
  using ImplType =
      ailego::DistanceBatch::InnerProductDistanceBatch<int8_t, BatchSize,
                                                       PrefetchStep>;
  static void ComputeBatch(const int8_t **vecs, const int8_t *query,
                           size_t num_vecs, size_t dim, float *results) {
    const size_t original_dim = dim - 20;
    ailego::DistanceBatch::InnerProductDistanceBatch<
        int8_t, BatchSize, PrefetchStep>::ComputeBatch(vecs, query, num_vecs,
                                                       original_dim, results);

    const float *q_tail = reinterpret_cast<const float *>(
        reinterpret_cast<const int8_t *>(query) + original_dim);
    float qa = q_tail[0];
    float qb = q_tail[1];
    float qs = q_tail[2];
    float qs2 = q_tail[3];

    const float sum = qa * qs;
    const float sum2 = qa * qa * qs2;
    for (size_t i = 0; i < num_vecs; ++i) {
      const float *m_tail = reinterpret_cast<const float *>(
          reinterpret_cast<const int8_t *>(vecs[i]) + original_dim);
      float ma = m_tail[0];
      float mb = m_tail[1];
      float ms = m_tail[2];
      float ms2 = m_tail[3];
      *results = ma * ma * ms2 + sum2 - 2 * ma * qa * *results +
                 (mb - qb) * (mb - qb) * original_dim +
                 2 * (mb - qb) * (ms * ma - sum);
      ++results;
    }
  }

  static void QueryPreprocess(void *query, size_t dim) {
    if (auto func = ImplType::GetQueryPreprocessFunc(); func != nullptr) {
      return func(query, dim - 20);
    }
  }
};

// Compute SquaredEuclidean for quantized INT4
template <size_t BatchSize, size_t PrefetchStep>
struct MipsSquaredEuclideanDistanceBatchWithScoreUnquantized<uint8_t, BatchSize,
                                                             PrefetchStep> {
  static void ComputeBatch(const uint8_t **vecs, const uint8_t *query,
                           size_t num_vecs, size_t dim, float *results) {
    const size_t original_dim = dim - 32;
    const size_t original_dim_in_uint8_array = original_dim >> 1;
    ailego::DistanceBatch::InnerProductDistanceBatch<
        uint8_t, BatchSize, PrefetchStep>::ComputeBatch(vecs, query, num_vecs,
                                                        original_dim, results);

    const float *q_tail = reinterpret_cast<const float *>(
        reinterpret_cast<const uint8_t *>(query) + original_dim_in_uint8_array);
    float qa = q_tail[0];
    float qb = q_tail[1];
    float qs = q_tail[2];
    float qs2 = q_tail[3];

    const float sum = qa * qs;
    const float sum2 = qa * qa * qs2;
    for (size_t i = 0; i < num_vecs; ++i) {
      const float *m_tail = reinterpret_cast<const float *>(
          reinterpret_cast<const uint8_t *>(vecs[i]) +
          original_dim_in_uint8_array);
      float ma = m_tail[0];
      float mb = m_tail[1];
      float ms = m_tail[2];
      float ms2 = m_tail[3];
      *results = ma * ma * ms2 + sum2 - 2 * ma * qa * *results +
                 (mb - qb) * (mb - qb) * original_dim +
                 2 * (mb - qb) * (ms * ma - sum);
      ++results;
    }
  }
};

}  // namespace zvec::core
