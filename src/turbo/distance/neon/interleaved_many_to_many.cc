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
#include "neon/interleaved_many_to_many.h"
#include "matrix.h"

namespace zvec::turbo::neon {
namespace {
#if defined(AILEGO_ARM64_GNU_LIKE)
// Reuse each interleaved candidate component across two independent queries.
// Unlike the legacy matrix entry point, query vectors stay contiguous. Keeping
// only two query accumulators bounds register pressure for 32-row blocks.
template <typename T, size_t Rows, MetricType Metric>
inline InterleavedManyToManyDistanceFunc InterleavedQueryKernel() {
  return [](const void *matrix, const void *const *queries, size_t n,
            size_t dim, float *out) {
    const auto *m = static_cast<const T *>(matrix);
    size_t q = 0;
    for (; q + 1 < n; q += 2) {
      const auto *a = static_cast<const T *>(queries[q]);
      const auto *b = static_cast<const T *>(queries[q + 1]);
      float sums_a[Rows]{}, sums_b[Rows]{};
      for (size_t d = 0; d < dim; ++d) {
        const float qa = float(a[d]), qb = float(b[d]);
        for (size_t i = 0; i < Rows; ++i) {
          const float value = float(m[d * Rows + i]);
          if constexpr (Metric == MetricType::kInnerProduct) {
            sums_a[i] -= value * qa;
            sums_b[i] -= value * qb;
          } else {
            const float da = value - qa, db = value - qb;
            sums_a[i] += da * da;
            sums_b[i] += db * db;
          }
        }
      }
      std::copy_n(sums_a, Rows, out + q * Rows);
      std::copy_n(sums_b, Rows, out + (q + 1) * Rows);
    }
    if (q < n) {
      InterleavedDistanceMatrix<T, Rows, Metric>::template Compute<1>(
          m, static_cast<const T *>(queries[q]), dim, out + q * Rows);
    }
  };
}

template <typename T, MetricType Metric>
inline InterleavedManyToManyDistanceFunc SelectInterleavedQueryKernel(
    size_t rows) {
  switch (rows) {
    case 2:
      return InterleavedQueryKernel<T, 2, Metric>();
    case 4:
      return InterleavedQueryKernel<T, 4, Metric>();
    case 8:
      return InterleavedQueryKernel<T, 8, Metric>();
    case 16:
      return InterleavedQueryKernel<T, 16, Metric>();
    case 32:
      return InterleavedQueryKernel<T, 32, Metric>();
    default:
      return {};
  }
}

template <typename T>
InterleavedManyToManyDistanceFunc select_metric(MetricType metric,
                                                size_t rows) {
  if (metric == MetricType::kInnerProduct) {
    return SelectInterleavedQueryKernel<T, MetricType::kInnerProduct>(rows);
  }
  if (metric == MetricType::kSquaredEuclidean) {
    return SelectInterleavedQueryKernel<T, MetricType::kSquaredEuclidean>(rows);
  }
  return {};
}
#endif
}  // namespace

InterleavedManyToManyDistanceFunc make_interleaved_many_to_many_distance(
    MetricType metric, DataType type, size_t rows) {
#if defined(AILEGO_ARM64_GNU_LIKE)
  if (type == DataType::kFp32) {
    return select_metric<float>(metric, rows);
  }
  if (type == DataType::kFp16) {
    return select_metric<ailego::Float16>(metric, rows);
  }
#else
  (void)metric;
  (void)type;
  (void)rows;
#endif
  return {};
}

}  // namespace zvec::turbo::neon
