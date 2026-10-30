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

#include "accelerate/fp32/distance.h"

#if defined(ZVEC_HAVE_ACCELERATE)

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <zvec/ailego/internal/platform.h>
#if defined(AILEGO_ARM64_NEON)
#include "neon/fp32/inner_product.h"
#endif

// Keep the choice of BLAS ABI local to this provider. The new LP64 interface
// is available on macOS 13.3 and newer; it does not require ILP64.
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

namespace zvec::turbo::accelerate {
namespace {

void InnerProductFallback(const void *base, size_t stride_bytes,
                          const void *query, size_t n, size_t dim,
                          float *distances) {
  const char *rows = static_cast<const char *>(base);
  const char *q = static_cast<const char *>(query);
  for (size_t i = 0; i < n; ++i) {
    const char *row = rows + i * stride_bytes;
    float sum = 0.0f;
    for (size_t d = 0; d < dim; ++d) {
      // The public interface permits byte strides. Odd strides can leave
      // rows unaligned, so the fallback must not dereference a float pointer.
      float lhs;
      float rhs;
      std::memcpy(&lhs, row + d * sizeof(float), sizeof(lhs));
      std::memcpy(&rhs, q + d * sizeof(float), sizeof(rhs));
      sum += lhs * rhs;
    }
    distances[i] = -sum;
  }
}

}  // namespace

// Dispatch has already checked the runtime OS. The availability annotation
// lets this implementation call the new CBLAS API when targeting older macOS.
API_AVAILABLE(macos(13.3))
void inner_product_fp32_strided_batch_distance(
    const void *base, size_t stride_bytes, const void *query, size_t n,
    size_t dim, float *distances, const void *const * /*extra_values*/) {
  if (n == 0) {
    return;
  }
  if (dim == 0) {
    // BLAS may return without writing y when the inner dimension is zero.
    std::fill_n(distances, n, 0.0f);
    return;
  }

#if defined(AILEGO_ARM64_NEON)
  // A single row has no matrix reuse to amortize BLAS dispatch. Its stride
  // does not constrain the pairwise kernel because no second row is read.
  if (n == 1 && reinterpret_cast<uintptr_t>(base) % alignof(float) == 0 &&
      reinterpret_cast<uintptr_t>(query) % alignof(float) == 0) {
    neon::inner_product_fp32_distance(base, query, dim, distances);
    return;
  }
#endif

  const size_t lda = stride_bytes / sizeof(float);
  constexpr size_t kBlasMax =
      static_cast<size_t>(std::numeric_limits<__LAPACK_int>::max());
  if (n <= kBlasMax && dim <= kBlasMax && lda <= kBlasMax && lda >= dim &&
      stride_bytes % sizeof(float) == 0 &&
      reinterpret_cast<uintptr_t>(base) % alignof(float) == 0 &&
      reinterpret_cast<uintptr_t>(query) % alignof(float) == 0) {
    cblas_sgemv(CblasRowMajor, CblasNoTrans, static_cast<__LAPACK_int>(n),
                static_cast<__LAPACK_int>(dim), -1.0f,
                static_cast<const float *>(base),
                static_cast<__LAPACK_int>(lda),
                static_cast<const float *>(query), 1, 0.0f, distances, 1);
    return;
  }

  InnerProductFallback(base, stride_bytes, query, n, dim, distances);
}

API_AVAILABLE(macos(13.3))
void cosine_fp32_strided_batch_distance(const void *base, size_t stride_bytes,
                                        const void *query, size_t n, size_t dim,
                                        float *distances,
                                        const void *const *extra_values) {
  inner_product_fp32_strided_batch_distance(base, stride_bytes, query, n, dim,
                                            distances, extra_values);
  for (size_t i = 0; i < n; ++i) {
    distances[i] += 1.0f;
  }
}

}  // namespace zvec::turbo::accelerate

#endif  // defined(ZVEC_HAVE_ACCELERATE)
