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
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/utility/float_helper.h>
#include <zvec/turbo/turbo.h>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace zvec::turbo {
namespace {

void CheckFp16(CpuArchType arch) {
  for (auto quantize : {QuantizeType::kRaw, QuantizeType::kFp16}) {
    for (auto metric : {MetricType::kInnerProduct, MetricType::kCosine,
                        MetricType::kSquaredEuclidean}) {
      auto many = get_many_to_many_distance_func(metric, DataType::kFp16,
                                                 quantize, arch);
      auto batch =
          get_batch_distance_func(metric, DataType::kFp16, quantize, arch);
      ASSERT_EQ(bool(many), bool(batch));
      // Explicit ISA requests do not fall back to scalar. Raw FP16 L2 has
      // scalar/NEON/AVX512 entries, but no AVX2 entry (unlike kFp16 L2).
      if (arch == CpuArchType::kAVX2 && quantize == QuantizeType::kRaw &&
          metric == MetricType::kSquaredEuclidean) {
        EXPECT_FALSE(many);
        continue;
      }
      ASSERT_TRUE(many);
      // Empty shapes must not touch even null input/output pointers.
      many(nullptr, nullptr, 0, 5, 128, nullptr, nullptr);
      many(nullptr, nullptr, 5, 0, 128, nullptr, nullptr);
      for (size_t dim : {0, 1, 3, 7, 8, 9, 15, 16, 17, 31, 32, 63, 64, 65, 127,
                         128, 129, 257}) {
        for (auto shape : {std::pair<size_t, size_t>{1, 1},
                           {1, 5},
                           {5, 1},
                           {2, 2},
                           {3, 5},
                           {8, 4},
                           {32, 32},
                           {33, 7}}) {
          const auto m = shape.first, n = shape.second;
          SCOPED_TRACE(testing::Message()
                       << "arch=" << int(arch) << " metric=" << int(metric)
                       << " quantize=" << int(quantize) << " dim=" << dim
                       << " m=" << m << " n=" << n);
          // Offset each vector by one half and pad the allocation. Pointer
          // ordering and row strides must not be inferred by the kernel.
          std::vector<std::vector<ailego::Float16>> storage(
              m + n, std::vector<ailego::Float16>(dim + 2));
          for (size_t k = 0; k < storage.size(); ++k) {
            std::vector<float> values(dim);
            double norm = 0;
            for (size_t d = 0; d < dim; ++d) {
              values[d] = float(int((k * 13 + d * 7) % 31) - 15);
              norm += values[d] * values[d];
            }
            if (norm > 0) {
              for (auto &v : values) v /= std::sqrt(norm);
            }
            for (size_t d = 0; d < dim; ++d)
              storage[k][d + 1] = ailego::Float16(values[d]);
          }
          std::vector<const void *> vectors(m), queries(n);
          for (size_t i = 0; i < m; ++i)
            vectors[i] = storage[m - 1 - i].data() + 1;
          for (size_t j = 0; j < n; ++j)
            queries[j] = storage[m + n - 1 - j].data() + 1;
          std::vector<float> actual(m * n + 2, -12345.0f), expected(m * n);
          many(vectors.data(), queries.data(), m, n, dim, actual.data() + 1,
               nullptr);
          for (size_t j = 0; j < n; ++j) {
            batch(vectors.data(), queries[j], m, dim, expected.data() + j * m,
                  nullptr);
            for (size_t i = 0; i < m; ++i) {
              const auto *a = static_cast<const ailego::Float16 *>(vectors[i]);
              const auto *b = static_cast<const ailego::Float16 *>(queries[j]);
              double reference = metric == MetricType::kCosine ? 1.0 : 0.0;
              for (size_t d = 0; d < dim; ++d) {
                double x = float(a[d]), y = float(b[d]);
                reference += metric == MetricType::kSquaredEuclidean
                                 ? (x - y) * (x - y)
                                 : -x * y;
              }
              // Native-FP16 cosine batch has lower accumulation precision.
              const float tolerance =
                  quantize == QuantizeType::kFp16 ? 3e-3f : 2e-5f;
              EXPECT_NEAR(reference, actual[1 + j * m + i], tolerance);
              EXPECT_NEAR(expected[j * m + i], actual[1 + j * m + i],
                          tolerance);
            }
          }
          EXPECT_EQ(-12345.0f, actual.front());
          EXPECT_EQ(-12345.0f, actual.back());
        }
      }
    }
  }
}

TEST(ManyToManyDistance, Fp16Scalar) {
  CheckFp16(CpuArchType::kScalar);
}
TEST(ManyToManyDistance, Fp16Auto) {
  CheckFp16(CpuArchType::kAuto);
}
TEST(ManyToManyDistance, Fp16Avx2) {
  if (!get_batch_distance_func(MetricType::kInnerProduct, DataType::kFp16,
                               QuantizeType::kRaw, CpuArchType::kAVX2))
    GTEST_SKIP() << "AVX2/F16C unavailable";
  CheckFp16(CpuArchType::kAVX2);
}
TEST(ManyToManyDistance, Fp16Neon) {
  if (!get_batch_distance_func(MetricType::kInnerProduct, DataType::kFp16,
                               QuantizeType::kRaw, CpuArchType::kNEON))
    GTEST_SKIP() << "NEON unavailable";
  CheckFp16(CpuArchType::kNEON);
}

#if defined(__unix__) || defined(__APPLE__)
TEST(ManyToManyDistance, Fp16TailsDoNotReadPastVector) {
  const size_t page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  std::vector<std::unique_ptr<void, std::function<void(void *)>>> mappings;
  for (size_t k = 0; k < 4; ++k) {
    void *base = mmap(nullptr, page_size * 2, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANON, -1, 0);
    ASSERT_NE(MAP_FAILED, base);
    mappings.emplace_back(base,
                          [page_size](void *p) { munmap(p, page_size * 2); });
    ASSERT_EQ(0, mprotect(static_cast<char *>(base) + page_size, page_size,
                          PROT_NONE));
  }
  for (size_t dim : {1, 3, 7, 8, 9, 15, 16, 17, 31, 127, 128, 129}) {
    const void *ptrs[4];
    for (size_t k = 0; k < 4; ++k) {
      auto *data = reinterpret_cast<ailego::Float16 *>(
                       static_cast<char *>(mappings[k].get()) + page_size) -
                   dim;
      for (size_t d = 0; d < dim; ++d) data[d] = ailego::Float16(0.125f);
      ptrs[k] = data;
    }
    for (auto arch : {CpuArchType::kScalar, CpuArchType::kAuto,
                      CpuArchType::kNEON, CpuArchType::kAVX2}) {
      auto many = get_many_to_many_distance_func(
          MetricType::kInnerProduct, DataType::kFp16, QuantizeType::kRaw, arch);
      if (!many) continue;
      float out[4];
      many(ptrs, ptrs + 2, 2, 2, dim, out, nullptr);
      for (float value : out) EXPECT_FLOAT_EQ(-0.015625f * dim, value);
    }
  }
}
#endif

TEST(ManyToManyDistance, Fp32FallbackAndUnsupportedDispatch) {
  float a[] = {1, 2, 3}, b[] = {-3, 1, 2}, c[] = {2, -1, 4};
  const void *vectors[] = {a, c, b};
  const void *queries[] = {b, a};
  for (auto arch : {CpuArchType::kScalar, CpuArchType::kAuto}) {
    for (auto metric :
         {MetricType::kSquaredEuclidean, MetricType::kInnerProduct}) {
      auto many = get_many_to_many_distance_func(metric, DataType::kFp32,
                                                 QuantizeType::kRaw, arch);
      auto batch = get_batch_distance_func(metric, DataType::kFp32,
                                           QuantizeType::kRaw, arch);
      ASSERT_TRUE(many);
      ASSERT_TRUE(batch);
      float actual[6], expected[6];
      many(vectors, queries, 3, 2, 3, actual, nullptr);
      for (size_t j = 0; j < 2; ++j) {
        batch(vectors, queries[j], 3, 3, expected + j * 3, nullptr);
      }
      for (size_t i = 0; i < 6; ++i) EXPECT_FLOAT_EQ(expected[i], actual[i]);
      many(nullptr, nullptr, 0, 3, 3, nullptr, nullptr);
      many(nullptr, nullptr, 3, 0, 3, nullptr, nullptr);
    }
  }
  EXPECT_FALSE(
      get_many_to_many_distance_func(MetricType::kInnerProduct, DataType::kFp16,
                                     QuantizeType::kRaw, CpuArchType::kSVE2));
}

TEST(ManyToManyDistance, RecordInt8PreprocessAndMetadataFallback) {
  // Record layout: INT8 codes, four FP32 metadata values, then an INT32
  // code sum used to undo the VNNI query's +128 shift.
  constexpr size_t dim = 16, encoded_size = dim + 20, m = 5, n = 3;
  std::vector<std::vector<float>> records(m + n,
                                          std::vector<float>(encoded_size / 4));
  for (size_t k = 0; k < records.size(); ++k) {
    auto *codes = reinterpret_cast<int8_t *>(records[k].data());
    float sum = 0, squares = 0;
    for (size_t d = 0; d < dim; ++d) {
      codes[d] = static_cast<int8_t>(int((k * 7 + d * 3) % 23) - 11);
      sum += codes[d];
      squares += codes[d] * codes[d];
    }
    float *tail = records[k].data() + dim / 4;
    tail[0] = 0.25f;
    tail[1] = 0.5f;
    tail[2] = sum;
    tail[3] = squares;
    const int32_t code_sum = static_cast<int32_t>(sum);
    std::memcpy(tail + 4, &code_sum, sizeof(code_sum));
  }
  for (auto metric :
       {MetricType::kInnerProduct, MetricType::kSquaredEuclidean}) {
    auto scalar = get_distance_func(
        metric, DataType::kInt8, QuantizeType::kRecord, CpuArchType::kScalar);
    ASSERT_TRUE(scalar);
    for (auto arch :
         {CpuArchType::kScalar, CpuArchType::kAuto, CpuArchType::kAVX512VNNI}) {
      SCOPED_TRACE(testing::Message()
                   << "arch=" << int(arch) << " metric=" << int(metric));
      auto kernels = get_distance_kernels(metric, DataType::kInt8,
                                          QuantizeType::kRecord, arch);
      auto many = get_many_to_many_distance_func(metric, DataType::kInt8,
                                                 QuantizeType::kRecord, arch);
      ASSERT_EQ(bool(kernels.batch), bool(many));
      if (!many) continue;
      auto prepared = records;
      const void *vectors[m], *queries[n], *extra[m];
      for (size_t i = 0; i < m; ++i) {
        vectors[i] = records[i].data();
        extra[i] = records[i].data() + dim / 4;
      }
      for (size_t j = 0; j < n; ++j) {
        if (kernels.preprocess)
          kernels.preprocess(prepared[m + j].data(), encoded_size);
        queries[j] = prepared[m + j].data();
      }
      float actual[m * n], expected[m * n];
      for (bool split : {false, true}) {
        many(vectors, queries, m, n, encoded_size, actual,
             split ? extra : nullptr);
        for (size_t j = 0; j < n; ++j) {
          kernels.batch(vectors, queries[j], m, encoded_size, expected + j * m,
                        split ? extra : nullptr);
          for (size_t i = 0; i < m; ++i) {
            float reference;
            scalar(vectors[i], records[m + j].data(), encoded_size, &reference);
            EXPECT_NEAR(reference, actual[j * m + i], 1e-4f);
            EXPECT_FLOAT_EQ(expected[j * m + i], actual[j * m + i]);
          }
        }
      }
    }
  }
}

}  // namespace
}  // namespace zvec::turbo
