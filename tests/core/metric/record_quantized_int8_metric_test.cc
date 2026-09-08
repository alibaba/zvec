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
#include <gtest/gtest.h>
#include <zvec/core/framework/index_factory.h>
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

TEST(RecordQuantizedInt8Metric,
     StoredPairAndEveryBatchRemainderMatchExactly) {
  std::mt19937 generator(20260825);
  std::uniform_int_distribution<int> code_distribution(-128, 127);
  std::uniform_real_distribution<float> scale_distribution(0.01f, 0.2f);
  std::uniform_real_distribution<float> bias_distribution(-2.0f, 2.0f);

  for (size_t dimension :
       {1UL, 31UL, 63UL, 64UL, 65UL, 127UL, 128UL, 129UL, 960UL, 1024UL}) {
    auto metric = CreateRecordInt8Metric(dimension);
    ASSERT_NE(nullptr, metric);
    auto distance = metric->distance();
    auto batch_distance = metric->batch_distance();
    auto preprocess = metric->get_query_preprocess_func();
    ASSERT_TRUE(static_cast<bool>(distance));
    ASSERT_TRUE(static_cast<bool>(batch_distance));
    ASSERT_TRUE(static_cast<bool>(preprocess));

    constexpr size_t kVectorCount = 11;
    const size_t encoded_dimension = dimension + kTailBytes;
    std::vector<int8_t> query(encoded_dimension, 0);
    std::vector<std::vector<int8_t>> records(
        kVectorCount, std::vector<int8_t>(encoded_dimension, 0));
    std::vector<const void *> vectors(kVectorCount);
    std::vector<float> expected(kVectorCount);
    for (size_t d = 0; d < dimension; ++d) {
      query[d] = static_cast<int8_t>(code_distribution(generator));
    }
    SetTail(&query, dimension, scale_distribution(generator),
            bias_distribution(generator));
    for (size_t i = 0; i < kVectorCount; ++i) {
      for (size_t d = 0; d < dimension; ++d) {
        records[i][d] = static_cast<int8_t>(code_distribution(generator));
      }
      SetTail(&records[i], dimension, scale_distribution(generator),
              bias_distribution(generator));
      vectors[i] = records[i].data();
      distance(records[i].data(), query.data(), encoded_dimension,
               &expected[i]);
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

}  // namespace
}  // namespace zvec::core
