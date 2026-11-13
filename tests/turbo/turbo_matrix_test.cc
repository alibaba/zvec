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
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <new>
#include <type_traits>
#include <vector>
#include <ailego/internal/cpu_features.h>
#include <core/metric/metric_params.h>
#include <gtest/gtest.h>
#include <turbo/distance/matrix.h>
#include <zvec/core/framework/index_factory.h>

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
// Factory registration allocates with nothrow new. Route it through the same
// malloc/free pair as the throwing overloads, including under ASan.
void *operator new(std::size_t size, const std::nothrow_t &) noexcept {
  try {
    return ::operator new(size);
  } catch (...) {
    return nullptr;
  }
}
void *operator new[](std::size_t size, const std::nothrow_t &tag) noexcept {
  return ::operator new(size, tag);
}
void operator delete(void *p, const std::nothrow_t &) noexcept {
  std::free(p);
}
void operator delete[](void *p, const std::nothrow_t &) noexcept {
  std::free(p);
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
  for (size_t rows : {1U, 2U, 4U, 8U, 16U, 32U}) {
    for (size_t columns = 1; columns <= rows; columns *= 2) {
      for (size_t dim : {0U, 1U, 3U, 4U, 7U, 8U, 9U, 17U, 32U, 128U, 513U}) {
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
              if constexpr (std::is_same_v<T, uint8_t>) {
                data[(d / 8 * count + i) * 4 + d % 8 / 2] |= (int(v) & 15)
                                                             << (4 * (d % 2));
              } else if constexpr (std::is_same_v<T, int8_t>) {
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

template <typename T>
void CheckJointQueries(turbo::CpuArchType arch = turbo::CpuArchType::kAuto) {
  for (size_t rows : {2U, 4U, 8U, 16U, 32U}) {
    for (bool ip : {false, true}) {
      auto kernel = turbo::get_interleaved_many_to_many_distance_func(
          ip ? turbo::MetricType::kInnerProduct
             : turbo::MetricType::kSquaredEuclidean,
          turbo::MatrixDataType<T>(), rows, arch);
      ASSERT_TRUE(kernel);
      for (size_t dim : {0U, 1U, 7U, 8U, 17U, 128U, 513U}) {
        for (size_t n : {0U, 1U, 2U, 3U, 7U, 8U, 31U, 32U, 35U}) {
          SCOPED_TRACE(::testing::Message()
                       << rows << "x" << n << " dim=" << dim << " ip=" << ip);
          std::vector<T> matrix(rows * dim + 1);
          std::vector<std::vector<T>> queries(n, std::vector<T>(dim + 1));
          std::vector<const void *> pointers;
          for (size_t d = 0; d < dim; ++d) {
            for (size_t i = 0; i < rows; ++i) {
              matrix[1 + d * rows + i] =
                  T(float(int((d * 13 + i * 7) % 31) - 15) / 7);
            }
            for (size_t q = 0; q < n; ++q) {
              queries[q][1 + d] = T(float(int((q * 11 + d * 3) % 29) - 14) / 9);
            }
          }
          for (size_t q = 0; q < n; ++q)
            pointers.push_back(queries[q].data() + 1);
          const auto original_matrix = matrix;
          const auto original_queries = queries;
          std::vector<float> result(rows * n + 2, -12345.0f);
          allocation_count = 0;
          count_allocations = true;
          kernel(matrix.data() + 1, pointers.data(), n, dim, result.data() + 1);
          count_allocations = false;
          EXPECT_EQ(0U, allocation_count);
          EXPECT_EQ(-12345.0f, result.front());
          EXPECT_EQ(-12345.0f, result.back());
          EXPECT_EQ(original_matrix, matrix);
          EXPECT_EQ(original_queries, queries);
          for (size_t q = 0; q < n; ++q) {
            for (size_t i = 0; i < rows; ++i) {
              double expected = 0, magnitude = 1;
              for (size_t d = 0; d < dim; ++d) {
                const double x = float(matrix[1 + d * rows + i]);
                const double y = float(queries[q][1 + d]);
                const double term = ip ? -x * y : (x - y) * (x - y);
                expected += term;
                magnitude += std::abs(term);
              }
              EXPECT_NEAR(expected, result[1 + q * rows + i], 2e-5 * magnitude);
            }
          }
        }
      }
    }
  }
}

TEST(TurboMatrix, JointInterleavedQueriesAndNoAllocations) {
  using namespace turbo;
  if (!get_interleaved_many_to_many_distance_func(MetricType::kInnerProduct,
                                                  DataType::kFp32, 32))
    GTEST_SKIP() << "No compiled joint kernel supported by this CPU";
  CheckJointQueries<float>();
  if (get_interleaved_many_to_many_distance_func(MetricType::kInnerProduct,
                                                 DataType::kFp16, 32))
    CheckJointQueries<ailego::Float16>();
}

TEST(TurboMatrix, Avx2JointInterleavedQueriesAndNoAllocations) {
  using namespace turbo;
  if (!get_interleaved_many_to_many_distance_func(
          MetricType::kInnerProduct, DataType::kFp32, 32, CpuArchType::kAVX2))
    GTEST_SKIP() << "AVX2 + FMA joint kernel unavailable";
  CheckJointQueries<float>(CpuArchType::kAVX2);
  if (get_interleaved_many_to_many_distance_func(
          MetricType::kInnerProduct, DataType::kFp16, 32, CpuArchType::kAVX2))
    CheckJointQueries<ailego::Float16>(CpuArchType::kAVX2);
}

TEST(TurboMatrix, InterleavedDispatchHonorsArchitectureAndLayout) {
  using namespace turbo;
  for (auto type : {DataType::kFp32, DataType::kFp16}) {
    for (auto metric :
         {MetricType::kInnerProduct, MetricType::kSquaredEuclidean}) {
      for (size_t rows : {2U, 4U, 8U, 16U, 32U}) {
        auto automatic =
            get_interleaved_many_to_many_distance_func(metric, type, rows);
        auto neon = get_interleaved_many_to_many_distance_func(
            metric, type, rows, CpuArchType::kNEON);
        auto avx2 = get_interleaved_many_to_many_distance_func(
            metric, type, rows, CpuArchType::kAVX2);
        EXPECT_EQ(bool(neon) || bool(avx2), bool(automatic));
#if defined(AILEGO_ARM64_GNU_LIKE)
        ASSERT_TRUE(neon);
#else
        EXPECT_FALSE(neon);
#endif
        if (neon) neon(nullptr, nullptr, 0, 128, nullptr);
        if (avx2) avx2(nullptr, nullptr, 0, 128, nullptr);
        for (auto arch :
             {CpuArchType::kScalar, CpuArchType::kSSE2, CpuArchType::kAVX512,
              CpuArchType::kAVX512VNNI, CpuArchType::kAVX512FP16}) {
          EXPECT_FALSE(get_interleaved_many_to_many_distance_func(metric, type,
                                                                  rows, arch));
        }
      }
    }
  }
  for (size_t rows : {0U, 1U, 3U, 33U, 64U}) {
    EXPECT_FALSE(get_interleaved_many_to_many_distance_func(
        MetricType::kInnerProduct, DataType::kFp32, rows));
  }
  EXPECT_FALSE(get_interleaved_many_to_many_distance_func(MetricType::kCosine,
                                                          DataType::kFp32, 32));
  EXPECT_FALSE(get_interleaved_many_to_many_distance_func(
      MetricType::kInnerProduct, DataType::kInt8, 32));
  auto &flags = ailego::internal::CpuFeatures::static_flags_;
  const auto saved = flags;
  flags.NEON = false;
  auto neon = get_interleaved_many_to_many_distance_func(
      MetricType::kInnerProduct, DataType::kFp32, 32, CpuArchType::kNEON);
  flags = saved;
  EXPECT_FALSE(neon);
  for (int missing = 0; missing < 3; ++missing) {
    if (missing == 0) flags.AVX2 = false;
    if (missing == 1) flags.FMA = false;
    if (missing == 2) flags.F16C = false;
    auto half = get_interleaved_many_to_many_distance_func(
        MetricType::kInnerProduct, DataType::kFp16, 32, CpuArchType::kAVX2);
    auto full = get_interleaved_many_to_many_distance_func(
        MetricType::kInnerProduct, DataType::kFp32, 32, CpuArchType::kAVX2);
    flags = saved;
    EXPECT_FALSE(half);
    if (missing < 2)
      EXPECT_FALSE(full);
    else
      EXPECT_EQ(bool(full), bool(get_interleaved_many_to_many_distance_func(
                                MetricType::kInnerProduct, DataType::kFp32, 32,
                                CpuArchType::kAVX2)));
  }
}

TEST(TurboMatrix, InterleavedFp16AccumulatesInFp32) {
  using namespace turbo;
  constexpr size_t Rows = 32, Dim = 17, Queries = 3;
  std::vector<ailego::Float16> matrix(Rows * Dim), query(Queries * Dim);
  std::vector<const void *> pointers;
  for (size_t i = 0; i < matrix.size(); ++i)
    matrix[i] = ailego::Float16(i % 3 ? 65504.0f : -65504.0f);
  for (size_t q = 0; q < Queries; ++q) {
    pointers.push_back(query.data() + q * Dim);
    for (size_t d = 0; d < Dim; ++d)
      query[q * Dim + d] = ailego::Float16((q + d) % 2 ? 65504.0f : -65504.0f);
  }
  for (auto metric :
       {MetricType::kInnerProduct, MetricType::kSquaredEuclidean}) {
    auto kernel = get_interleaved_many_to_many_distance_func(
        metric, DataType::kFp16, Rows);
    if (!kernel) GTEST_SKIP() << "No joint FP16 kernel";
    std::vector<float> out(Rows * Queries);
    kernel(matrix.data(), pointers.data(), Queries, Dim, out.data());
    for (size_t q = 0; q < Queries; ++q)
      for (size_t i = 0; i < Rows; ++i) {
        double expected = 0, magnitude = 1;
        for (size_t d = 0; d < Dim; ++d) {
          const double x = float(matrix[d * Rows + i]),
                       y = float(query[q * Dim + d]);
          const double term =
              metric == MetricType::kInnerProduct ? -x * y : (x - y) * (x - y);
          expected += term;
          magnitude += std::abs(term);
        }
        EXPECT_TRUE(std::isfinite(out[q * Rows + i]));
        EXPECT_NEAR(expected, out[q * Rows + i], 2e-5 * magnitude);
      }
  }
}

template <typename T, size_t Rows, turbo::MetricType Metric>
void CheckSingleColumnTails() {
  constexpr bool int4 = std::is_same_v<T, uint8_t>;
  constexpr bool int8 = std::is_same_v<T, int8_t>;
  for (size_t dim : {0U,  1U,  2U,  3U,  4U,   5U,   7U,   8U,   9U,   15U, 16U,
                     17U, 31U, 32U, 33U, 127U, 128U, 129U, 511U, 512U, 513U}) {
    SCOPED_TRACE(::testing::Message() << "rows=" << Rows << " dim=" << dim);
    const size_t stride = int4   ? (dim + 7) / 8 * 4
                          : int8 ? (dim + 3) / 4 * 4
                                 : dim;
    // Poison padded dimensions, and give the query exactly its logical size.
    // ASan can then detect a full-word read past a partial final query word.
    std::vector<T> matrix(Rows * stride, T(7.0f));
    const size_t query_size = int4 ? (dim + 1) / 2 : dim;
    auto query = std::make_unique<T[]>(query_size);
    std::vector<double> expected(Rows, 0);
    std::vector<double> absolute_products(Rows, 0);
    auto value = [](size_t row, size_t d, bool is_query) -> float {
      const size_t index = is_query ? d / 16 : row + d;
      if constexpr (std::is_same_v<T, uint8_t>) {
        return float(int(index % 16) - 8);
      } else if constexpr (std::is_same_v<T, int8_t>) {
        constexpr int values[] = {-128, 127, -1, 0, 1, 64, -64};
        return float(values[index % 7]);
      } else {
        constexpr float values[] = {-65504, 65504, -0.5f, 0, 0.5f, 3, -3};
        return float(T(values[index % 7]));
      }
    };
    for (size_t d = 0; d < dim; ++d) {
      const float q = value(0, d, true);
      if constexpr (int4) {
        query[d / 2] |= (int(q) & 15) << (4 * (d % 2));
      } else {
        query[d] = T(q);
      }
      for (size_t i = 0; i < Rows; ++i) {
        const float m = value(i, d, false);
        if constexpr (int4) {
          const size_t offset = (d / 8 * Rows + i) * 4 + d % 8 / 2;
          const size_t shift = 4 * (d % 2);
          matrix[offset] =
              (matrix[offset] & ~(15 << shift)) | ((int(m) & 15) << shift);
        } else if constexpr (int8) {
          matrix[(d / 4 * Rows + i) * 4 + d % 4] = T(m);
        } else {
          matrix[d * Rows + i] = T(m);
        }
        const double delta = double(m) - q;
        expected[i] += Metric == turbo::MetricType::kSquaredEuclidean
                           ? delta * delta
                           : -double(m) * q;
        absolute_products[i] += std::abs(double(m) * q);
      }
    }
    auto kernel = turbo::MakeRawDistanceMatrix<Metric>(
        Rows, 1, turbo::MatrixDataType<T>());
    ASSERT_TRUE(kernel);
    std::array<float, Rows + 2> dynamic, direct;
    dynamic.fill(-12345);
    direct.fill(-12345);
    allocation_count = 0;
    count_allocations = true;
    kernel(matrix.data(), query.get(), dim, dynamic.data() + 1);
    turbo::DistanceMatrix<T, Rows, 1, Metric>::Compute(
        matrix.data(), query.get(), dim, direct.data() + 1);
    count_allocations = false;
    EXPECT_EQ(0U, allocation_count);
    EXPECT_EQ(-12345, dynamic.front());
    EXPECT_EQ(-12345, dynamic.back());
    EXPECT_EQ(-12345, direct.front());
    EXPECT_EQ(-12345, direct.back());
    for (size_t i = 0; i < Rows; ++i) {
      double tolerance = std::max(1e-5, std::abs(expected[i]) * 1e-5);
      if constexpr (std::is_same_v<T, ailego::Float16> &&
                    Metric == turbo::MetricType::kInnerProduct) {
        // The new FP16 IP cases include cancellation of 65504^2 products.
        // A bound relative to the near-zero answer is invalid: bound FP32
        // summation error by gamma_n * sum(abs(products)) instead. Products
        // of two finite FP16 values are exactly representable in FP32.
        const double nu = dim * std::numeric_limits<float>::epsilon() / 2;
        tolerance = std::max(1e-5, nu / (1 - nu) * absolute_products[i]);
      }
      EXPECT_NEAR(expected[i], dynamic[i + 1], tolerance);
      EXPECT_FLOAT_EQ(dynamic[i + 1], direct[i + 1]);
    }
  }
}

template <size_t Rows>
void CheckSingleColumnTypes() {
  CheckSingleColumnTails<ailego::Float16, Rows,
                         turbo::MetricType::kSquaredEuclidean>();
  CheckSingleColumnTails<ailego::Float16, Rows,
                         turbo::MetricType::kInnerProduct>();
  CheckSingleColumnTails<int8_t, Rows, turbo::MetricType::kInnerProduct>();
  CheckSingleColumnTails<int8_t, Rows, turbo::MetricType::kSquaredEuclidean>();
  CheckSingleColumnTails<uint8_t, Rows, turbo::MetricType::kSquaredEuclidean>();
  CheckSingleColumnTails<uint8_t, Rows, turbo::MetricType::kInnerProduct>();
}

TEST(TurboMatrix, SingleColumnExtremesTailsAndNoAllocations) {
  CheckSingleColumnTypes<1>();
  CheckSingleColumnTypes<2>();
  CheckSingleColumnTypes<4>();
  CheckSingleColumnTypes<8>();
  CheckSingleColumnTypes<16>();
  CheckSingleColumnTypes<32>();
}

TEST(TurboMatrix, Int16ExtremaMatchIndependentReference) {
  for (size_t rows : {1U, 2U, 4U, 8U, 16U, 32U}) {
    for (size_t columns = 1; columns <= rows; columns *= 2) {
      for (size_t dim : {3U, 33U}) {
        std::vector<int16_t> data(rows * dim), query(columns * dim);
        for (size_t d = 0; d < dim; ++d) {
          for (size_t i = 0; i < rows; ++i)
            data[d * rows + i] = (i + d) % 2 ? INT16_MIN : INT16_MAX;
          for (size_t j = 0; j < columns; ++j)
            query[d * columns + j] = (j + d) % 2 ? INT16_MAX : INT16_MIN;
        }
        for (bool ip : {false, true}) {
          auto kernel = ip ? turbo::MakeRawDistanceMatrix<
                                 turbo::MetricType::kInnerProduct>(
                                 rows, columns, turbo::DataType::kInt16)
                           : turbo::MakeRawDistanceMatrix<
                                 turbo::MetricType::kSquaredEuclidean>(
                                 rows, columns, turbo::DataType::kInt16);
          std::vector<float> output(rows * columns);
          kernel(data.data(), query.data(), dim, output.data());
          for (size_t j = 0; j < columns; ++j) {
            for (size_t i = 0; i < rows; ++i) {
              double expected = 0;
              for (size_t d = 0; d < dim; ++d) {
                const double a = data[d * rows + i], b = query[d * columns + j];
                expected += ip ? -a * b : (a - b) * (a - b);
              }
              EXPECT_NEAR(expected, output[j * rows + i],
                          std::max(1.0, std::abs(expected) * 1e-6));
            }
          }
        }
      }
    }
  }
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

TEST(TurboMatrix, RecordAndMipsFastMatricesDoNotAllocate) {
  using namespace zvec::core;
  for (auto type : {IndexMeta::DT_INT8, IndexMeta::DT_INT4}) {
    for (bool record : {false, true}) {
      for (const char *name :
           {"SquaredEuclidean", "InnerProduct", "MipsSquaredEuclidean",
            "NormalizedCosine", "Cosine"}) {
        if (!record && std::strcmp(name, "MipsSquaredEuclidean")) continue;
        const size_t payload = type == IndexMeta::DT_INT4 ? 4 : 8;
        const bool cosine = !std::strcmp(name, "Cosine");
        const size_t tail_bytes =
            record ? (type == IndexMeta::DT_INT4 ? 16 : 20) + (cosine ? 4 : 0)
                   : 0;
        const size_t bytes = payload + tail_bytes;
        const size_t dim = type == IndexMeta::DT_INT4 ? bytes * 2 : bytes;
        ailego::Params params;
        if (record)
          params.set(QUANTIZED_INTEGER_METRIC_ORIGIN_METRIC_NAME, name);
        auto metric =
            IndexFactory::CreateMetric(record ? "QuantizedInteger" : name);
        ASSERT_TRUE(metric);
        ASSERT_EQ(0, metric->init(IndexMeta(type, dim), params));
        std::vector<uint8_t> data(bytes * 32, 0);
        for (size_t i = 0; i < 32; ++i) {
          for (size_t word = 0; word < payload / 4; ++word)
            std::memset(data.data() + (word * 32 + i) * 4,
                        type == IndexMeta::DT_INT4 ? 0x11 : 0x01, 4);
          if (record) {
            const float tail[] = {0.5f, 1.0f, 8.0f, 8.0f};
            for (size_t k = 0; k < 4; ++k)
              std::memcpy(data.data() + ((payload / 4 + k) * 32 + i) * 4,
                          tail + k, 4);
            const int32_t code_sum = 8;
            if (type == IndexMeta::DT_INT8 ||
                (std::strcmp(name, "SquaredEuclidean") &&
                 std::strcmp(name, "MipsSquaredEuclidean"))) {
              const size_t k = type == IndexMeta::DT_INT8 ? 4 : 3;
              std::memcpy(data.data() + ((payload / 4 + k) * 32 + i) * 4,
                          &code_sum, 4);
            }
            if (cosine) {
              const float norm = 1.0f;
              std::memcpy(data.data() + ((bytes / 4 - 1) * 32 + i) * 4, &norm,
                          4);
            }
          }
        }
        std::vector<uint8_t> row(bytes);
        for (size_t word = 0; word < bytes / 4; ++word)
          std::memcpy(row.data() + word * 4, data.data() + word * 32 * 4, 4);
        float expected;
        metric->distance()(row.data(), row.data(), dim, &expected);
        for (size_t columns : {1U, 2U, 4U, 8U, 16U, 32U}) {
          // Multi-query INT4 records retain the faster contiguous SIMD path.
          if (record && type == IndexMeta::DT_INT4 && columns > 1) continue;
          std::vector<uint8_t> query(bytes * columns);
          for (size_t word = 0; word < bytes / 4; ++word)
            for (size_t j = 0; j < columns; ++j)
              std::memcpy(query.data() + (word * columns + j) * 4,
                          data.data() + word * 32 * 4, 4);
          auto matrix = metric->distance_matrix(32, columns);
          ASSERT_TRUE(matrix);
          std::vector<float> output(32 * columns);
          allocation_count = 0;
          count_allocations = true;
          matrix(data.data(), query.data(), dim, output.data());
          count_allocations = false;
          size_t expected_allocations = 0;
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || \
    defined(_M_IX86)
          // These x86 record shapes deliberately use the faster contiguous
          // SIMD adapter, which owns a matrix buffer and a query buffer.
          if (record && (type == IndexMeta::DT_INT4 || columns > 1))
            expected_allocations = 2;
          if (record && type == IndexMeta::DT_INT4 && columns == 1 &&
              (!std::strcmp(name, "SquaredEuclidean") ||
               !std::strcmp(name, "InnerProduct")))
            expected_allocations = 1;  // Batch path only materializes records.
#endif
          EXPECT_EQ(expected_allocations, allocation_count)
              << name << " columns=" << columns;
          for (float value : output) {
            EXPECT_TRUE(std::isfinite(value));
            EXPECT_NEAR(expected, value,
                        1e-5f * std::max(1.0f, std::abs(expected)));
          }
        }
      }
    }
  }
}
}  // namespace
