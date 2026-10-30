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
#include <cstdlib>
#include <new>
#include <vector>
#include <gtest/gtest.h>
#include <turbo/distance/matrix.h>

namespace {
thread_local bool count_allocations = false;
thread_local size_t allocation_count = 0;
}  // namespace

// Count allocations only while invoking an already-created matrix kernel.
// This catches reintroducing per-call transpose buffers without timing-based
// assertions, which would be unreliable on shared CI runners.
void *operator new(std::size_t size) {
  if (count_allocations) ++allocation_count;
  if (void *p = std::malloc(size ? size : 1)) return p;
  throw std::bad_alloc();
}
void *operator new[](std::size_t size) {
  return ::operator new(size);
}
void operator delete(void *p) noexcept {
  std::free(p);
}
void operator delete[](void *p) noexcept {
  std::free(p);
}
#if defined(__cpp_sized_deallocation)
void operator delete(void *p, std::size_t) noexcept {
  std::free(p);
}
void operator delete[](void *p, std::size_t) noexcept {
  std::free(p);
}
#endif

namespace {
using namespace zvec;

template <typename T>
void CheckMatrices() {
  for (size_t rows : {2U, 4U, 8U, 16U, 32U}) {
    for (size_t columns = 1; columns <= rows; columns *= 2) {
      for (size_t dim : {1U, 3U, 8U, 32U, 128U}) {
        SCOPED_TRACE(::testing::Message()
                     << rows << "x" << columns << " dim=" << dim);
        constexpr bool int4 = std::is_same_v<T, uint8_t>;
        constexpr bool int8 = std::is_same_v<T, int8_t>;
        const size_t stride = int4   ? (dim + 7) / 8 * 4
                              : int8 ? (dim + 3) / 4 * 4
                                     : dim;
        std::vector<T> matrix(rows * stride), queries(columns * stride);
        auto value = [](size_t row, size_t d, int salt) {
          const float raw = float(int((row * 3 + d * 5 + salt) % 16) - 8);
          if constexpr (std::is_integral_v<T>)
            return raw;
          else
            return float(T(raw / 7.0f));
        };
        auto fill = [&](std::vector<T> &data, size_t count, int salt) {
          for (size_t i = 0; i < count; ++i) {
            for (size_t d = 0; d < dim; ++d) {
              const float v = value(i, d, salt);
              if constexpr (int4) {
                data[(d / 8 * count + i) * 4 + d % 8 / 2] |= (int(v) & 15)
                                                             << (4 * (d % 2));
              } else if constexpr (int8) {
                data[(d / 4 * count + i) * 4 + d % 4] = T(v);
              } else {
                data[d * count + i] = T(v);
              }
            }
          }
        };
        fill(matrix, rows, 0);
        fill(queries, columns, 2);
        for (bool ip : {false, true}) {
          auto kernel = ip ? turbo::MakeRawDistanceMatrix<
                                 turbo::MetricType::kInnerProduct>(
                                 rows, columns, turbo::MatrixDataType<T>())
                           : turbo::MakeRawDistanceMatrix<
                                 turbo::MetricType::kSquaredEuclidean>(
                                 rows, columns, turbo::MatrixDataType<T>());
          ASSERT_TRUE(kernel);
          std::vector<float> output(rows * columns);
          allocation_count = 0;
          count_allocations = true;
          kernel(matrix.data(), queries.data(), dim, output.data());
          count_allocations = false;
          EXPECT_EQ(0U, allocation_count);
          for (size_t j = 0; j < columns; ++j) {
            for (size_t i = 0; i < rows; ++i) {
              float expected = 0;
              for (size_t d = 0; d < dim; ++d) {
                const float a = value(i, d, 0), b = value(j, d, 2);
                expected += ip ? -a * b : (a - b) * (a - b);
              }
              EXPECT_NEAR(expected, output[j * rows + i], 1e-5f * dim);
            }
          }
        }
      }
    }
  }
}

TEST(TurboMatrix, DirectInterleavedLayoutsAndNoAllocations) {
  CheckMatrices<float>();
  CheckMatrices<ailego::Float16>();
  CheckMatrices<double>();
  CheckMatrices<int16_t>();
  CheckMatrices<int8_t>();
  CheckMatrices<uint8_t>();
}

TEST(TurboMatrix, KmeansTemplateDoesNotAllocate) {
  float rows[32 * 128]{}, queries[32 * 128]{}, output[32 * 32];
  allocation_count = 0;
  count_allocations = true;
  turbo::SquaredEuclideanDistanceMatrix<float, 32, 32>::Compute(rows, queries,
                                                                128, output);
  turbo::SquaredEuclideanDistanceMatrix<float, 32, 1>::Compute(rows, queries,
                                                               128, output);
  count_allocations = false;
  EXPECT_EQ(0U, allocation_count);
  for (float distance : output) EXPECT_FLOAT_EQ(0.0f, distance);
}
}  // namespace
