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
#include <random>
#include <string>
#include <vector>
#include <ailego/internal/cpu_features.h>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/turbo/turbo.h>
#include "metric/metric_params.h"

namespace zvec::core {
namespace {

constexpr size_t kTailBytes = 4 * sizeof(float) + sizeof(int32_t);

void SetTail(std::vector<int8_t> *record, size_t dimension, float scale,
             float bias) {
  int32_t code_sum = 0;
  float sum = 0.0f;
  float sum_squared = 0.0f;
  for (size_t i = 0; i < dimension; ++i) {
    const int code = static_cast<int>((*record)[i]);
    code_sum += code;
    sum += static_cast<float>(code);
    sum_squared += static_cast<float>(code * code);
  }
  const float tail[4] = {scale, bias, sum, sum_squared};
  std::memcpy(record->data() + dimension, tail, sizeof(tail));
  std::memcpy(record->data() + dimension + sizeof(tail), &code_sum,
              sizeof(code_sum));
}

IndexMetric::Pointer CreateRecordInt8Metric(size_t dimension) {
  auto metric = IndexFactory::CreateMetric("QuantizedInteger");
  if (!metric) {
    return nullptr;
  }

  IndexMeta meta(IndexMeta::DataType::DT_INT8, dimension + kTailBytes);
  ailego::Params params;
  params.set(QUANTIZED_INTEGER_METRIC_ORIGIN_METRIC_NAME,
             std::string("SquaredEuclidean"));
  return metric->init(meta, params) == 0 ? metric : nullptr;
}

float ReferenceSquaredDistance(const std::vector<int8_t> &record,
                               const std::vector<int8_t> &query,
                               size_t dimension) {
  float record_params[2];
  float query_params[2];
  std::memcpy(record_params, record.data() + dimension, sizeof(record_params));
  std::memcpy(query_params, query.data() + dimension, sizeof(query_params));
  double result = 0.0;
  for (size_t d = 0; d < dimension; ++d) {
    const double lhs =
        static_cast<double>(record_params[0]) * record[d] + record_params[1];
    const double rhs =
        static_cast<double>(query_params[0]) * query[d] + query_params[1];
    const double delta = lhs - rhs;
    result += delta * delta;
  }
  return static_cast<float>(result);
}

void CheckStoredPairsAndBatchRemainders(bool explicit_vnni) {
  if (explicit_vnni &&
      !ailego::internal::CpuFeatures::static_flags_.AVX512_VNNI) {
    GTEST_SKIP() << "Requires an AVX-512 VNNI CPU";
  }

  // RecordQuantizer maps records to [-127, 127]. The legacy SSE/AVX2
  // stored-pair kernel relies on that range for its sign/abs arithmetic.
  // Exercise the additional -128 boundary only through explicit VNNI kernels.
  const int min_code = explicit_vnni ? -128 : -127;
  std::mt19937 generator(20260825);
  std::uniform_int_distribution<int> code_distribution(min_code, 127);
  std::uniform_real_distribution<float> scale_distribution(0.01f, 0.2f);
  std::uniform_real_distribution<float> bias_distribution(-2.0f, 2.0f);

  for (size_t dimension :
       {1UL, 31UL, 63UL, 64UL, 65UL, 127UL, 128UL, 129UL, 960UL, 1024UL}) {
    SCOPED_TRACE(testing::Message() << "dimension=" << dimension
                                    << ", explicit_vnni=" << explicit_vnni);
    auto metric = CreateRecordInt8Metric(dimension);
    ASSERT_NE(nullptr, metric);
    auto distance = metric->distance();
    auto batch_distance = metric->batch_distance();
    auto preprocess = metric->get_query_preprocess_func();
    if (explicit_vnni) {
      const auto kernels = turbo::get_distance_kernels(
          turbo::MetricType::kSquaredEuclidean, turbo::DataType::kInt8,
          turbo::QuantizeType::kRecord, turbo::CpuArchType::kAVX512VNNI);
      distance = kernels.dist;
      batch_distance = kernels.batch;
      preprocess = kernels.preprocess;
    }
    ASSERT_TRUE(static_cast<bool>(distance));
    ASSERT_TRUE(static_cast<bool>(batch_distance));
    ASSERT_TRUE(static_cast<bool>(preprocess));

    // Cover repeated VNNI batches (2/4) and legacy batches (12), plus tails.
    constexpr size_t kVectorCount = 25;
    const size_t encoded_dimension = dimension + kTailBytes;
    std::vector<int8_t> query(encoded_dimension, 0);
    std::vector<std::vector<int8_t>> records(
        kVectorCount, std::vector<int8_t>(encoded_dimension, 0));
    std::vector<const void *> vectors(kVectorCount);
    std::vector<float> expected(kVectorCount);
    for (size_t d = 0; d < dimension; ++d) {
      query[d] = static_cast<int8_t>(code_distribution(generator));
    }
    query[0] = static_cast<int8_t>(min_code);
    if (dimension > 1) query[dimension - 1] = 127;
    SetTail(&query, dimension, scale_distribution(generator),
            bias_distribution(generator));
    for (size_t i = 0; i < kVectorCount; ++i) {
      for (size_t d = 0; d < dimension; ++d) {
        records[i][d] = static_cast<int8_t>(code_distribution(generator));
      }
      records[i][0] = static_cast<int8_t>(i % 2 == 0 ? min_code : 127);
      if (dimension > 1) {
        records[i][dimension - 1] =
            static_cast<int8_t>(i % 2 == 0 ? 127 : min_code);
      }
      SetTail(&records[i], dimension, scale_distribution(generator),
              bias_distribution(generator));
      vectors[i] = records[i].data();
      expected[i] = ReferenceSquaredDistance(records[i], query, dimension);
      float stored_distance = 0.0f;
      distance(records[i].data(), query.data(), encoded_dimension,
               &stored_distance);
      EXPECT_NEAR(expected[i], stored_distance,
                  1e-4f * std::max(1.0f, std::fabs(expected[i])))
          << "stored pair, vector=" << i;
    }

    std::vector<int8_t> prepared_query = query;
    preprocess(prepared_query.data(), encoded_dimension);
    for (size_t count = 1; count <= kVectorCount; ++count) {
      std::vector<float> actual(count, 0.0f);
      batch_distance(vectors.data(), prepared_query.data(), count,
                     encoded_dimension, actual.data(), nullptr);
      for (size_t i = 0; i < count; ++i) {
        EXPECT_NEAR(expected[i], actual[i],
                    1e-4f * std::max(1.0f, std::fabs(expected[i])))
            << "dimension=" << dimension << ", count=" << count
            << ", vector=" << i;
      }
    }
  }
}

TEST(RecordQuantizedInt8Metric,
     StoredPairAndEveryBatchRemainderMatchReference) {
  CheckStoredPairsAndBatchRemainders(false);
}

TEST(RecordQuantizedInt8Metric, VnniStoredPairsAndBatchesIncludeInt8Min) {
  CheckStoredPairsAndBatchRemainders(true);
}

}  // namespace
}  // namespace zvec::core
