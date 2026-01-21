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
#include <ailego/math/euclidean_distance_matrix.h>
#include <ailego/math/inner_product_matrix.h>
#include <ailego/math/mips_euclidean_distance_matrix.h>
#include <ailego/math/norm2_matrix.h>
#include <ailego/math_batch/distance_batch.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_factory.h>
#include "metric_params.h"


namespace zvec::core {
//===========================================================
// SquaredEuclidean
//===========================================================

template <typename T, size_t M, size_t N>
struct SquaredEuclidean;

// Compute SquaredEuclidean for quantized INT8
template <size_t M, size_t N>
struct SquaredEuclidean<int8_t, M, N> {
  static void Compute(const int8_t *m, const int8_t *q, size_t dim,
                      float *out) {
    const size_t d = dim - 16;
    ailego::InnerProductMatrix<int8_t, M, N>::Compute(m, q, d, out);

    for (size_t i = 0; i < N; ++i) {
      float qa = *reinterpret_cast<const float *>(&q[d * N + i * 4]);
      float qb = *reinterpret_cast<const float *>(&q[(d + 4) * N + i * 4]);
      float qs = *reinterpret_cast<const float *>(&q[(d + 8) * N + i * 4]);
      float qs2 = *reinterpret_cast<const float *>(&q[(d + 12) * N + i * 4]);
      const float sum = qa * qs;
      const float sum2 = qa * qa * qs2;
      for (size_t j = 0; j < M; ++j) {
        float ma = *reinterpret_cast<const float *>(&m[d * M + j * 4]);
        float mb = *reinterpret_cast<const float *>(&m[(d + 4) * M + j * 4]);
        float ms = *reinterpret_cast<const float *>(&m[(d + 8) * M + j * 4]);
        float ms2 = *reinterpret_cast<const float *>(&m[(d + 12) * M + j * 4]);
        *out = ma * ma * ms2 + sum2 - 2 * ma * qa * *out +
               (mb - qb) * (mb - qb) * d + 2 * (mb - qb) * (ms * ma - sum);
        out++;
      }
    }
  }
};

// Compute SquaredEuclidean for quantized INT4
template <size_t M, size_t N>
struct SquaredEuclidean<uint8_t, M, N> {
  static void Compute(const uint8_t *m, const uint8_t *q, size_t dim,
                      float *out) {
    const size_t d = dim - 32;
    const size_t p = d >> 1;  // params
    ailego::InnerProductMatrix<uint8_t, M, N>::Compute(m, q, d, out);

    for (size_t i = 0; i < N; ++i) {
      float qa = *reinterpret_cast<const float *>(&q[p * N + i * 4]);
      float qb = *reinterpret_cast<const float *>(&q[(p + 4) * N + i * 4]);
      float qs = *reinterpret_cast<const float *>(&q[(p + 8) * N + i * 4]);
      float qs2 = *reinterpret_cast<const float *>(&q[(p + 12) * N + i * 4]);
      const float sum = qa * qs;
      const float sum2 = qa * qa * qs2;
      for (size_t j = 0; j < M; ++j) {
        float ma = *reinterpret_cast<const float *>(&m[p * M + j * 4]);
        float mb = *reinterpret_cast<const float *>(&m[(p + 4) * M + j * 4]);
        float ms = *reinterpret_cast<const float *>(&m[(p + 8) * M + j * 4]);
        float ms2 = *reinterpret_cast<const float *>(&m[(p + 12) * M + j * 4]);
        *out = ma * ma * ms2 + sum2 - 2 * ma * qa * *out +
               (mb - qb) * (mb - qb) * d + 2 * (mb - qb) * (ms * ma - sum);
        out++;
      }
    }
  }
};
//===========================================================
// MinusInnerProduct
//===========================================================

template <size_t M, size_t N>
static void MinusInnerProductImplInt8(const int8_t *m, const int8_t *q,
                                      size_t origin_dim, float *out) {
  const size_t d = origin_dim;
  ailego::InnerProductMatrix<int8_t, M, N>::Compute(m, q, d, out);

  for (size_t i = 0; i < N; ++i) {
    float qa = *reinterpret_cast<const float *>(&q[d * N + i * 4]);
    float qb = *reinterpret_cast<const float *>(&q[(d + 4) * N + i * 4]);
    float qs = *reinterpret_cast<const float *>(&q[(d + 8) * N + i * 4]);
    for (size_t j = 0; j < M; ++j) {
      float ma = *reinterpret_cast<const float *>(&m[d * M + j * 4]);
      float mb = *reinterpret_cast<const float *>(&m[(d + 4) * M + j * 4]);
      float ms = *reinterpret_cast<const float *>(&m[(d + 8) * M + j * 4]);
      *out = -(ma * qa * *out + mb * qa * qs + qb * ma * ms + d * qb * mb);
      out++;
    }
  }
}

template <size_t M, size_t N>
static void MinusInnerProductImplUint8(const uint8_t *m, const uint8_t *q,
                                       size_t origin_dim, float *out) {
  const size_t d = origin_dim;
  const size_t p = d >> 1;  // params
  ailego::InnerProductMatrix<uint8_t, M, N>::Compute(m, q, d, out);

  for (size_t i = 0; i < N; ++i) {
    float qa = *reinterpret_cast<const float *>(&q[p * N + i * 4]);
    float qb = *reinterpret_cast<const float *>(&q[(p + 4) * N + i * 4]);
    float qs = *reinterpret_cast<const float *>(&q[(p + 8) * N + i * 4]);
    for (size_t j = 0; j < M; ++j) {
      float ma = *reinterpret_cast<const float *>(&m[p * M + j * 4]);
      float mb = *reinterpret_cast<const float *>(&m[(p + 4) * M + j * 4]);
      float ms = *reinterpret_cast<const float *>(&m[(p + 8) * M + j * 4]);
      *out = -(ma * qa * *out + mb * qa * qs + qb * ma * ms + d * qb * mb);
      out++;
    }
  }
}


template <typename T, size_t M, size_t N>
struct MinusInnerProduct;

// Compute MinusInnerProduct for quantized INT8
template <size_t M, size_t N>
struct MinusInnerProduct<int8_t, M, N> {
  static void Compute(const int8_t *m, const int8_t *q, size_t dim,
                      float *out) {
    const size_t origin_dim = dim - 16;
    MinusInnerProductImplInt8<M, N>(m, q, origin_dim, out);
  }
};

// Compute MinusInnerProduct for quantized INT4
template <size_t M, size_t N>
struct MinusInnerProduct<uint8_t, M, N> {
  static void Compute(const uint8_t *m, const uint8_t *q, size_t dim,
                      float *out) {
    const size_t origin_dim = dim - 32;
    MinusInnerProductImplUint8<M, N>(m, q, origin_dim, out);
  }
};


//===========================================================
// CosineMinusInnerProduct
//===========================================================
template <typename T, size_t M, size_t N>
struct CosineMinusInnerProduct;

// Compute CosineMinusInnerProduct for quantized INT8
template <size_t M, size_t N>
struct CosineMinusInnerProduct<int8_t, M, N> {
  static void Compute(const int8_t *m, const int8_t *q, size_t dim,
                      float *out) {
    const size_t origin_dim = dim - 20;
    MinusInnerProductImplInt8<M, N>(m, q, origin_dim, out);
  }
};

// Compute CosineMinusInnerProduct for quantized INT4
template <size_t M, size_t N>
struct CosineMinusInnerProduct<uint8_t, M, N> {
  static void Compute(const uint8_t *m, const uint8_t *q, size_t dim,
                      float *out) {
    const size_t origin_dim = dim - 40;
    MinusInnerProductImplUint8<M, N>(m, q, origin_dim, out);
  }
};

//===========================================================
// MipsSquaredEuclidean
//===========================================================

template <typename T, size_t M, size_t N>
struct MipsSquaredEuclidean;

// Compute MipsSquaredEuclidean for quantized INT8
template <size_t M, size_t N>
struct MipsSquaredEuclidean<int8_t, M, N> {
  static void Compute(const int8_t *m, const int8_t *q, size_t dim,
                      float *out) {
    const size_t d = dim - 16;
    ailego::InnerProductMatrix<int8_t, M, N>::Compute(m, q, d, out);

    for (size_t i = 0; i < N; ++i) {
      float qa = *reinterpret_cast<const float *>(&q[d * N + i * 4]);
      float qb = *reinterpret_cast<const float *>(&q[(d + 4) * N + i * 4]);
      float qs = *reinterpret_cast<const float *>(&q[(d + 8) * N + i * 4]);
      float qs2 = *reinterpret_cast<const float *>(&q[(d + 12) * N + i * 4]);
      float q2 = qa * qa * qs2 + 2 * qa * qb * qs + d * qb * qb;
      for (size_t j = 0; j < M; ++j) {
        float ma = *reinterpret_cast<const float *>(&m[d * M + j * 4]);
        float mb = *reinterpret_cast<const float *>(&m[(d + 4) * M + j * 4]);
        float ms = *reinterpret_cast<const float *>(&m[(d + 8) * M + j * 4]);
        float ms2 = *reinterpret_cast<const float *>(&m[(d + 12) * M + j * 4]);
        float m2 = ma * ma * ms2 + 2 * ma * mb * ms + d * mb * mb;
        *out = 2.0f - 2.0f *
                          (ma * qa * *out + mb * qa * qs + qb * ma * ms +
                           d * qb * mb) /
                          std::max(q2, m2);
        out++;
      }
    }
  }
};

// Compute MipsSquaredEuclidean for quantized INT4
template <size_t M, size_t N>
struct MipsSquaredEuclidean<uint8_t, M, N> {
  static void Compute(const uint8_t *m, const uint8_t *q, size_t dim,
                      float *out) {
    const size_t d = dim - 32;
    const size_t p = d >> 1;  // params
    ailego::InnerProductMatrix<uint8_t, M, N>::Compute(m, q, d, out);

    for (size_t i = 0; i < N; ++i) {
      float qa = *reinterpret_cast<const float *>(&q[p * N + i * 4]);
      float qb = *reinterpret_cast<const float *>(&q[(p + 4) * N + i * 4]);
      float qs = *reinterpret_cast<const float *>(&q[(p + 8) * N + i * 4]);
      float qs2 = *reinterpret_cast<const float *>(&q[(p + 12) * N + i * 4]);
      float q2 = qa * qa * qs2 + 2 * qa * qb * qs + d * qb * qb;
      for (size_t j = 0; j < M; ++j) {
        float ma = *reinterpret_cast<const float *>(&m[p * M + j * 4]);
        float mb = *reinterpret_cast<const float *>(&m[(p + 4) * M + j * 4]);
        float ms = *reinterpret_cast<const float *>(&m[(p + 8) * M + j * 4]);
        float ms2 = *reinterpret_cast<const float *>(&m[(p + 12) * M + j * 4]);
        float m2 = ma * ma * ms2 + 2 * ma * mb * ms + d * mb * mb;
        *out = 2.0f - 2.0f *
                          (ma * qa * *out + mb * qa * qs + qb * ma * ms +
                           d * qb * mb) /
                          std::max(q2, m2);
        out++;
      }
    }
  }
};

}  // namespace zvec::core