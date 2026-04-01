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
#include "inner_product_matrix.h"

namespace zvec {
namespace ailego {
//--------------------------------------------------
// Dense
//--------------------------------------------------
#if defined(__AVX2__)
float InnerProductInt4AVX2(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float MinusInnerProductInt4AVX2(const uint8_t *lhs, const uint8_t *rhs,
                                size_t size);
#endif

#if defined(__SSE4_1__)
float InnerProductInt4SSE(const uint8_t *lhs, const uint8_t *rhs, size_t size);
float MinusInnerProductInt4SSE(const uint8_t *lhs, const uint8_t *rhs,
                               size_t size);
#endif

float InnerProductInt4Scalar(const uint8_t *m, const uint8_t *q, size_t dim);
float MinusInnerProductInt4Scalar(const uint8_t *m, const uint8_t *q,
                                  size_t dim);

//! Compute the distance between matrix and query (INT4, M=1, N=1)
void InnerProductMatrix<uint8_t, 1, 1>::Compute(const uint8_t *m,
                                                const uint8_t *q, size_t dim,
                                                float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    *out = InnerProductInt4AVX2(m, q, dim);
    return;
  }
#endif  // __AVX2__

#if defined(__SSE4_1__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.SSE4_1) {
    *out = InnerProductInt4SSE(m, q, dim);
    return;
  }
#endif  //__SSE4_1__
  *out = InnerProductInt4Scalar(m, q, dim);
}

//! Compute the distance between matrix and query (INT4, M=1, N=1)
void MinusInnerProductMatrix<uint8_t, 1, 1>::Compute(const uint8_t *m,
                                                     const uint8_t *q,
                                                     size_t dim, float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    *out = MinusInnerProductInt4AVX2(m, q, dim);
    return;
  }
#endif  // __AVX2__

#if defined(__SSE4_1__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.SSE4_1) {
    *out = MinusInnerProductInt4SSE(m, q, dim);
    return;
  }
#endif  //__SSE4_1__
  *out = MinusInnerProductInt4Scalar(m, q, dim);
}

}  // namespace ailego
}  // namespace zvec