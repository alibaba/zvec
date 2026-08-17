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
#include <cstring>
#include <tuple>
#include <vector>
#include <ailego/internal/cpu_features.h>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/turbo/turbo.h>

namespace zvec::core {
namespace {

enum class Encoding { Uint7, Uint8, RecordInt8 };

struct Record {
  std::vector<int8_t> encoded;
  std::vector<double> decoded;
};

Record MakeRecord(Encoding encoding, size_t dimension, uint32_t seed,
                  bool include_int8_min) {
  const size_t tail = encoding == Encoding::Uint7   ? 0
                      : encoding == Encoding::Uint8 ? sizeof(uint32_t)
                                                    : 20;
  Record record{std::vector<int8_t>(dimension + tail),
                std::vector<double>(dimension)};
  const float scale = 0.03125f * (seed % 4 + 1);
  const float bias = 0.5f * (static_cast<int>(seed % 7) - 3);
  int32_t sum = 0;
  uint32_t norm = 0;
  for (size_t d = 0; d < dimension; ++d) {
    // Include endpoint-only and mixed records, not just small positive codes.
    int code = seed % 3 == 0 ? (d % 2 == 0 ? 0 : 255)
                             : (d * 71 + seed * 53 + d * seed * 17) % 256;
    if (encoding == Encoding::Uint7) code /= 2;
    if (encoding == Encoding::RecordInt8) {
      code -= 128;
      // RecordQuantizer emits [-127, 127]; reserve -128 for explicit VNNI.
      if (!include_int8_min && code == -128) code = -127;
    }
    record.encoded[d] =
        static_cast<int8_t>(encoding == Encoding::Uint8 ? code - 128 : code);
    record.decoded[d] = encoding == Encoding::RecordInt8
                            ? static_cast<double>(scale) * code + bias
                            : code;
    sum += code;
    norm += code * code;
  }
  if (encoding == Encoding::Uint8) {
    std::memcpy(record.encoded.data() + dimension, &norm, sizeof(norm));
  } else if (encoding == Encoding::RecordInt8) {
    const float values[] = {scale, bias, static_cast<float>(sum),
                            static_cast<float>(norm)};
    std::memcpy(record.encoded.data() + dimension, values, sizeof(values));
    std::memcpy(record.encoded.data() + dimension + sizeof(values), &sum,
                sizeof(sum));
  }
  return record;
}

// The explicit VNNI cases skip on other CPUs; passing fallback cases is not
// evidence that the Ice Lake kernels ran.
class VamanaQuantizedBatchTest
    : public testing::TestWithParam<std::tuple<Encoding, bool>> {};

TEST_P(VamanaQuantizedBatchTest, BatchesAndPrefetchBoundariesMatchReference) {
  const auto [encoding, explicit_vnni] = GetParam();
  if (explicit_vnni &&
      !ailego::internal::CpuFeatures::static_flags_.AVX512_VNNI) {
    GTEST_SKIP() << "Requires an AVX-512 VNNI CPU; run this case on Ice Lake";
  }
  constexpr size_t kCount = 33;
  for (const size_t dimension : {1U, 63U, 64U, 65U, 127U, 128U, 129U, 255U,
                                 256U, 257U, 959U, 960U, 961U}) {
    SCOPED_TRACE(testing::Message() << "dimension=" << dimension);
    auto query = MakeRecord(encoding, dimension, 37, explicit_vnni);
    std::vector<Record> records;
    for (uint32_t i = 0; i < kCount; ++i) {
      records.push_back(MakeRecord(encoding, dimension, i + 1, explicit_vnni));
    }
    const char *name = encoding == Encoding::Uint7   ? "UniformUint7"
                       : encoding == Encoding::Uint8 ? "UniformUint8"
                                                     : "QuantizedInteger";
    const char *key =
        encoding == Encoding::Uint7
            ? "proxima.uniform_uint7.metric.origin_metric_name"
        : encoding == Encoding::Uint8
            ? "proxima.uniform_uint8.metric.origin_metric_name"
            : "proxima.quantized_integer.metric.origin_metric_name";
    auto metric = IndexFactory::CreateMetric(name);
    ASSERT_TRUE(metric);
    ailego::Params params;
    params.set(key, std::string("SquaredEuclidean"));
    ASSERT_EQ(0, metric->init(IndexMeta(IndexMeta::DataType::DT_INT8,
                                        query.encoded.size()),
                              params));
    auto batch = metric->batch_distance();
    auto preprocess = metric->get_query_preprocess_func();
    if (explicit_vnni) {
      const auto kind =
          encoding == Encoding::Uint7   ? turbo::QuantizeType::kUniform
          : encoding == Encoding::Uint8 ? turbo::QuantizeType::kUniformUint8
                                        : turbo::QuantizeType::kRecord;
      const auto kernels = turbo::get_distance_kernels(
          turbo::MetricType::kSquaredEuclidean, turbo::DataType::kInt8, kind,
          turbo::CpuArchType::kAVX512VNNI);
      ASSERT_TRUE(kernels.batch) << "VNNI hardware present but kernel missing";
      batch = kernels.batch;
      preprocess = kernels.preprocess;
    }
    ASSERT_TRUE(batch);
    if (preprocess) preprocess(query.encoded.data(), query.encoded.size());
    const auto prepared_query = query.encoded;

    std::array<const void *, kCount> pointers{};
    std::array<const void *, kCount> extras{};
    std::array<uint32_t, kCount> norms{};
    std::array<float, kCount> expected{};
    for (size_t i = 0; i < kCount; ++i) {
      double sum = 0;
      for (size_t d = 0; d < dimension; ++d) {
        const double diff = records[i].decoded[d] - query.decoded[d];
        sum += diff * diff;
      }
      expected[i] = static_cast<float>(sum);
      if (encoding == Encoding::Uint8) {
        std::memcpy(&norms[i], records[i].encoded.data() + dimension, 4);
        // Model contiguous storage: no inline norm in the vector allocation.
        std::vector<int8_t> body(records[i].encoded.begin(),
                                 records[i].encoded.begin() + dimension);
        records[i].encoded.swap(body);
        extras[i] = &norms[i];
      }
      pointers[i] = records[i].encoded.data();
    }
    // 0..33 covers 2/4-way batches, multiple batches, 4/8-candidate look-ahead,
    // both sides of each boundary, and every final remainder.
    for (size_t count = 0; count <= kCount; ++count) {
      SCOPED_TRACE(testing::Message() << "count=" << count);
      constexpr float kCanary = -12345.0f;
      std::vector<float> actual(count + 2, kCanary);
      // Exact-sized pointer arrays also expose look-ahead pointer reads past
      // count under ASan, even though a prefetch instruction itself may not
      // fault.
      std::vector<const void *> batch_ptrs(pointers.begin(),
                                           pointers.begin() + count);
      std::vector<const void *> batch_extras(extras.begin(),
                                             extras.begin() + count);
      batch(batch_ptrs.data(), prepared_query.data(), count,
            prepared_query.size(), actual.data() + 1,
            encoding == Encoding::Uint8 ? batch_extras.data() : nullptr);
      EXPECT_EQ(kCanary, actual.front());
      EXPECT_EQ(kCanary, actual.back());
      for (size_t i = 0; i < count; ++i) {
        if (encoding == Encoding::RecordInt8) {
          EXPECT_NEAR(expected[i], actual[i + 1],
                      1e-4f * std::max(1.0f, expected[i]));
        } else {
          EXPECT_EQ(expected[i], actual[i + 1]);
        }
      }
      EXPECT_EQ(prepared_query, query.encoded);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Dispatch, VamanaQuantizedBatchTest,
                         testing::Combine(testing::Values(Encoding::Uint7,
                                                          Encoding::Uint8,
                                                          Encoding::RecordInt8),
                                          testing::Values(false, true)));

}  // namespace
}  // namespace zvec::core
