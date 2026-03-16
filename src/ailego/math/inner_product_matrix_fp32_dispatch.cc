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

#if defined(__ARM_NEON)
float InnerProductNEON(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductNEON(const float *lhs, const float *rhs, size_t size);
#endif

#if defined(__AVX512F__)
float InnerProductAVX512(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductAVX512(const float *lhs, const float *rhs, size_t size);
#endif

#if defined(__AVX__)
float InnerProductAVX(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductAVX(const float *lhs, const float *rhs, size_t size);
#endif

#if defined(__SSE__)
float InnerProductSSE(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductSSE(const float *lhs, const float *rhs, size_t size);
#endif

#if defined(__SSE__) || defined(__ARM_NEON)
//! Compute the distance between matrix and query (FP32, M=1, N=1)
void InnerProductMatrix<float, 1, 1>::Compute(const ValueType *m,
                                              const ValueType *q, size_t dim,
                                              float *out) {
#if defined(__ARM_NEON)
  *out = InnerProductNEON(m, q, dim);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    if (dim > 15) {
      *out = InnerProductAVX512(m, q, dim);
      return;
    }
  }
#endif  // __AVX512F__
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    if (dim > 7) {
      *out = InnerProductAVX(m, q, dim);
      return;
    }
  }
#endif  // __AVX__
  *out = InnerProductSSE(m, q, dim);
#endif  // __ARM_NEON
}

//! Compute the distance between matrix and query (FP32, M=1, N=1)
void MinusInnerProductMatrix<float, 1, 1>::Compute(const ValueType *m,
                                                   const ValueType *q,
                                                   size_t dim, float *out) {
#if defined(__ARM_NEON)
  *out = MinusInnerProductNEON(m, q, dim);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    if (dim > 15) {
      *out = MinusInnerProductAVX512(m, q, dim);
      return;
    }
  }
#endif  // __AVX512F__
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    if (dim > 7) {
      *out = MinusInnerProductAVX(m, q, dim);
      return;
    }
  }
#endif  // __AVX__
  *out = MinusInnerProductSSE(m, q, dim);
#endif  // __ARM_NEON
}

#endif
}  // namespace ailego
}  // namespace zvec
