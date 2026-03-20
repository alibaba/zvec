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

#include <ailego/utility/math_helper.h>
#include <zvec/ailego/internal/platform.h>
#include <zvec/ailego/utility/type_helper.h>
#include "distance_utility.h"

namespace zvec {
namespace ailego {

//--------------------------------------------------
// Dense
//--------------------------------------------------
template <typename T>
inline float SquaredEuclideanDistanceScalar(const T *m, const T *q,
                                            size_t dim) {
  ailego_assert(m && q && dim);

  float sum = 0.0;
  for (size_t i = 0; i < dim; ++i) {
    sum += MathHelper::SquaredDifference(m[i], q[i]);
  }

  return sum;
}

template <typename T>
inline float EuclideanDistanceScalar(const T *m, const T *q, size_t dim) {
  ailego_assert(m && q && dim);

  float sum = 0.0;
  for (size_t i = 0; i < dim; ++i) {
    sum += MathHelper::SquaredDifference(m[i], q[i]);
  }

  return std::sqrt(sum);
}

float SquaredEuclideanDistanceInt4Scalar(const uint8_t *m, const uint8_t *q,
                                         size_t dim) {
  ailego_assert(m && q && dim && !(dim & 1));

  float sum = 0.0;
  for (size_t i = 0; i < (dim >> 1); ++i) {
    uint8_t m_val = m[i];
    uint8_t q_val = q[i];
    sum += Int4SquaredDiffTable[((m_val << 4) & 0xf0) | ((q_val >> 0) & 0xf)] +
           Int4SquaredDiffTable[((m_val >> 0) & 0xf0) | ((q_val >> 4) & 0xf)];
  }

  return sum;
}


float EuclideanDistanceInt4Scalar(const uint8_t *m, const uint8_t *q,
                                  size_t dim) {
  ailego_assert(m && q && dim && !(dim & 1));

  float sum = 0.0;
  for (size_t i = 0; i < (dim >> 1); ++i) {
    uint8_t m_val = m[i];
    uint8_t q_val = q[i];
    sum += Int4SquaredDiffTable[((m_val << 4) & 0xf0) | ((q_val >> 0) & 0xf)] +
           Int4SquaredDiffTable[((m_val >> 0) & 0xf0) | ((q_val >> 4) & 0xf)];
  }

  return std::sqrt(sum);
}


float SquaredEuclideanDistanceInt8Scalar(const int8_t *m, const int8_t *q,
                                         size_t dim) {
  return SquaredEuclideanDistanceScalar<int8_t>(m, q, dim);
}

float EuclideanDistanceInt8Scalar(const int8_t *m, const int8_t *q,
                                  size_t dim) {
  return EuclideanDistanceScalar<int8_t>(m, q, dim);
}

float SquaredEuclideanDistanceFp16Scalar(const ailego::Float16 *m,
                                         const ailego::Float16 *q, size_t dim) {
  return SquaredEuclideanDistanceScalar<ailego::Float16>(m, q, dim);
}

float EuclideanDistanceFp16Scalar(const ailego::Float16 *m,
                                  const ailego::Float16 *q, size_t dim) {
  return EuclideanDistanceScalar<ailego::Float16>(m, q, dim);
}

float SquaredEuclideanDistanceFp32Scalar(const float *m, const float *q,
                                         size_t dim) {
  return SquaredEuclideanDistanceScalar<float>(m, q, dim);
}

float EuclideanDistanceFp32Scalar(const float *m, const float *q, size_t dim) {
  return EuclideanDistanceScalar<float>(m, q, dim);
}


}  // namespace ailego
}  // namespace zvec
