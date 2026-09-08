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
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/interface/index_param.h>
#include <zvec/turbo/turbo.h>
#include "metric/metric_params.h"

namespace zvec::core {
namespace {

constexpr size_t kTailBytes = sizeof(uint32_t);

std::vector<int8_t> EncodeRecord(const std::vector<uint8_t> &codes) {
  std::vector<int8_t> encoded(codes.size() + kTailBytes, 0);
  int64_t sum_squared = 0;
  for (size_t i = 0; i < codes.size(); ++i) {
    encoded[i] = static_cast<int8_t>(static_cast<int>(codes[i]) - 128);
    sum_squared += static_cast<int>(codes[i]) * codes[i];
  }
  const uint32_t tail = static_cast<uint32_t>(sum_squared);
  std::memcpy(encoded.data() + codes.size(), &tail, sizeof(tail));
  return encoded;
}

std::vector<int8_t> EncodeQuery(const std::vector<uint8_t> &codes) {
  return EncodeRecord(codes);
}

uint32_t ReadTail(const std::vector<int8_t> &encoded,
                  size_t original_dimension) {
  uint32_t tail = 0;
  std::memcpy(&tail, encoded.data() + original_dimension, sizeof(tail));
  return tail;
}

int32_t ReadQueryCorrection(const std::vector<int8_t> &encoded,
                            size_t original_dimension) {
  int32_t correction = 0;
  std::memcpy(&correction, encoded.data() + original_dimension,
              sizeof(correction));
  return correction;
}

int64_t SquaredL2(const std::vector<uint8_t> &lhs,
                  const std::vector<uint8_t> &rhs) {
  int64_t result = 0;
  for (size_t i = 0; i < lhs.size(); ++i) {
    const int difference = static_cast<int>(lhs[i]) - rhs[i];
    result += static_cast<int64_t>(difference) * difference;
  }
  return result;
}

std::vector<int8_t> PrepareQuery(const IndexMetric::Pointer &query_metric,
                                 const std::vector<int8_t> &query) {
  auto prepared = query;
  const auto preprocess = query_metric->get_query_preprocess_func();
  EXPECT_TRUE(preprocess);
  if (preprocess) {
    preprocess(prepared.data(), prepared.size());
  }
  return prepared;
}

IndexMetric::Pointer CreateMetric(size_t original_dimension) {
  auto metric = IndexFactory::CreateMetric("UniformUint8");
  if (!metric) {
    return nullptr;
  }
  IndexMeta meta(IndexMeta::DataType::DT_INT8, original_dimension + kTailBytes);
  ailego::Params params;
  params.set(UNIFORM_UINT8_METRIC_ORIGIN_METRIC_NAME,
             std::string("SquaredEuclidean"));
  return metric->init(meta, params) == 0 ? metric : nullptr;
}

TEST(UniformUint8Metric, UsesExactBuildAndQueryDistance) {
  const std::vector<uint8_t> first_codes{0, 1, 127, 128, 254, 255};
  const std::vector<uint8_t> second_codes{255, 128, 127, 1, 0, 254};
  const std::vector<uint8_t> query_codes{17, 255, 3, 200, 128, 0};
  const auto first = EncodeRecord(first_codes);
  const auto second = EncodeRecord(second_codes);
  const auto query = EncodeQuery(query_codes);

  auto metric = CreateMetric(first_codes.size());
  ASSERT_TRUE(metric);
  ASSERT_TRUE(metric->distance());
  ASSERT_TRUE(metric->batch_distance());
  ASSERT_TRUE(metric->get_query_preprocess_func());
  auto query_metric = metric->query_metric();
  ASSERT_TRUE(query_metric);
  ASSERT_TRUE(query_metric->distance());
  ASSERT_TRUE(query_metric->distance_matrix(1, 1));

  float distance = 0.0f;
  metric->distance()(first.data(), second.data(), first.size(), &distance);
  EXPECT_FLOAT_EQ(static_cast<float>(SquaredL2(first_codes, second_codes)),
                  distance);

  const auto prepared_stored_query = PrepareQuery(metric, second);
  const void *stored_vectors[] = {first.data()};
  const void *stored_extra_values[] = {first.data() + first_codes.size()};
  metric->batch_distance()(stored_vectors, prepared_stored_query.data(), 1,
                           first.size(), &distance, stored_extra_values);
  EXPECT_FLOAT_EQ(static_cast<float>(SquaredL2(first_codes, second_codes)),
                  distance);

  query_metric->distance_matrix(1, 1)(first.data(), query.data(), first.size(),
                                      &distance);
  EXPECT_FLOAT_EQ(static_cast<float>(SquaredL2(first_codes, query_codes)),
                  distance);

  const auto prepared_query = PrepareQuery(query_metric, query);
  int64_t expected_correction = 0;
  for (uint8_t code : query_codes) {
    expected_correction +=
        static_cast<int64_t>(code) * code - 256 * static_cast<int>(code);
  }
  EXPECT_EQ(expected_correction,
            ReadQueryCorrection(prepared_query, query_codes.size()));
  query_metric->distance()(first.data(), prepared_query.data(), first.size(),
                           &distance);
  EXPECT_FLOAT_EQ(static_cast<float>(SquaredL2(first_codes, query_codes)),
                  distance);
}

TEST(UniformUint8Metric, QueryBatchMatchesScalarAcrossKernelBoundaries) {
  constexpr size_t kVectorCount = 7;
  for (const size_t dimension :
       {1UL, 15UL, 16UL, 31UL, 63UL, 64UL, 65UL, 127UL, 128UL, 129UL, 1024UL}) {
    std::vector<uint8_t> query_codes(dimension);
    std::vector<std::vector<uint8_t>> vector_codes(
        kVectorCount, std::vector<uint8_t>(dimension));
    for (size_t d = 0; d < dimension; ++d) {
      query_codes[d] = static_cast<uint8_t>((d * 131 + 17) & 0xff);
      for (size_t i = 0; i < kVectorCount; ++i) {
        vector_codes[i][d] =
            static_cast<uint8_t>((d * (29 + i * 18) + i * 53) & 0xff);
      }
    }

    const auto query = EncodeQuery(query_codes);
    std::vector<std::vector<int8_t>> vectors;
    std::vector<const void *> vector_pointers;
    std::vector<const void *> extra_value_pointers;
    vectors.reserve(kVectorCount);
    vector_pointers.reserve(kVectorCount);
    extra_value_pointers.reserve(kVectorCount);
    for (const auto &codes : vector_codes) {
      vectors.push_back(EncodeRecord(codes));
      vector_pointers.push_back(vectors.back().data());
      extra_value_pointers.push_back(vectors.back().data() + dimension);
    }

    auto metric = CreateMetric(dimension);
    ASSERT_TRUE(metric);
    auto query_metric = metric->query_metric();
    ASSERT_TRUE(query_metric);
    ASSERT_TRUE(query_metric->batch_distance());
    // Streamer contexts retain the build metric while swapping in the query
    // distance functions, so preprocessing must also be available there.
    const auto prepared_query = PrepareQuery(metric, query);

    std::vector<float> distances(kVectorCount);
    query_metric->batch_distance()(
        vector_pointers.data(), prepared_query.data(), kVectorCount,
        dimension + kTailBytes, distances.data(), extra_value_pointers.data());
    for (size_t i = 0; i < kVectorCount; ++i) {
      EXPECT_FLOAT_EQ(
          static_cast<float>(SquaredL2(vector_codes[i], query_codes)),
          distances[i])
          << "dimension=" << dimension << ", vector=" << i;
    }
  }
}

TEST(UniformUint8Metric, QueryBatchConsumesSeparateExtraValues) {
  constexpr size_t kVectorCount = 5;
  constexpr size_t kDimension = 129;
  std::vector<uint8_t> query_codes(kDimension);
  std::vector<std::vector<uint8_t>> record_codes(
      kVectorCount, std::vector<uint8_t>(kDimension));
  for (size_t d = 0; d < kDimension; ++d) {
    query_codes[d] = static_cast<uint8_t>((d * 37 + 11) & 0xff);
    for (size_t i = 0; i < kVectorCount; ++i) {
      record_codes[i][d] =
          static_cast<uint8_t>((d * (19 + i * 12) + i * 41) & 0xff);
    }
  }

  auto metric = CreateMetric(kDimension);
  ASSERT_TRUE(metric);
  EXPECT_EQ(kTailBytes, metric->extra_values_size_per_vector());
  auto query_metric = metric->query_metric();
  ASSERT_TRUE(query_metric);
  EXPECT_EQ(kTailBytes, query_metric->extra_values_size_per_vector());
  auto batch_distance = query_metric->batch_distance();
  ASSERT_TRUE(batch_distance);

  const auto prepared_query =
      PrepareQuery(query_metric, EncodeQuery(query_codes));
  std::vector<std::vector<int8_t>> vector_bodies;
  std::vector<uint32_t> extra_values(kVectorCount);
  std::vector<const void *> vector_pointers;
  std::vector<const void *> extra_value_pointers;
  vector_bodies.reserve(kVectorCount);
  vector_pointers.reserve(kVectorCount);
  extra_value_pointers.reserve(kVectorCount);
  for (size_t i = 0; i < kVectorCount; ++i) {
    const auto encoded = EncodeRecord(record_codes[i]);
    vector_bodies.emplace_back(encoded.begin(), encoded.begin() + kDimension);
    std::memcpy(&extra_values[i], encoded.data() + kDimension,
                sizeof(extra_values[i]));
    vector_pointers.push_back(vector_bodies.back().data());
    extra_value_pointers.push_back(&extra_values[i]);
  }

  std::vector<float> distances(kVectorCount);
  batch_distance(vector_pointers.data(), prepared_query.data(), kVectorCount,
                 kDimension + kTailBytes, distances.data(),
                 extra_value_pointers.data());
  for (size_t i = 0; i < kVectorCount; ++i) {
    EXPECT_FLOAT_EQ(static_cast<float>(SquaredL2(record_codes[i], query_codes)),
                    distances[i]);
  }
}

TEST(UniformUint8Metric, QueryPreprocessConvertsCanonicalLayout) {
  constexpr size_t kDimension = MAX_DIMENSION;
  const std::vector<uint8_t> query_codes(kDimension, uint8_t{255});
  auto query = EncodeQuery(query_codes);

  auto metric = CreateMetric(kDimension);
  ASSERT_TRUE(metric);
  const auto preprocess = metric->get_query_preprocess_func();
  ASSERT_TRUE(preprocess);

  preprocess(query.data(), query.size());
  const auto *raw_query = reinterpret_cast<const uint8_t *>(query.data());
  for (size_t i = 0; i < kDimension; ++i) {
    ASSERT_EQ(query_codes[i], raw_query[i]) << "dimension offset=" << i;
  }
  EXPECT_EQ(-static_cast<int64_t>(kDimension) * 255,
            ReadQueryCorrection(query, kDimension));
}

TEST(UniformUint8Metric, TurboBatchCallWritesEveryDistanceWhenAvailable) {
  constexpr size_t kVectorCount = 3;
  const std::vector<uint8_t> query_codes{17, 255, 3, 200, 128, 0};
  const std::vector<std::vector<uint8_t>> record_codes{
      {0, 1, 127, 128, 254, 255},
      {255, 128, 127, 1, 0, 254},
      {33, 66, 99, 132, 165, 198},
  };
  const auto query = EncodeQuery(query_codes);

  std::vector<std::vector<int8_t>> records;
  std::vector<const void *> record_pointers;
  std::vector<const void *> extra_value_pointers;
  records.reserve(kVectorCount);
  record_pointers.reserve(kVectorCount);
  extra_value_pointers.reserve(kVectorCount);
  for (const auto &codes : record_codes) {
    records.push_back(EncodeRecord(codes));
    record_pointers.push_back(records.back().data());
    extra_value_pointers.push_back(records.back().data() + query_codes.size());
  }

  std::vector<float> distances(kVectorCount,
                               std::numeric_limits<float>::quiet_NaN());
  auto batch_distance = zvec::turbo::get_batch_distance_func(
      zvec::turbo::MetricType::kSquaredEuclidean, zvec::turbo::DataType::kInt8,
      zvec::turbo::QuantizeType::kUniformUint8);
  auto preprocess = zvec::turbo::get_query_preprocess_func(
      zvec::turbo::MetricType::kSquaredEuclidean, zvec::turbo::DataType::kInt8,
      zvec::turbo::QuantizeType::kUniformUint8);
  if (!batch_distance || !preprocess) {
    GTEST_SKIP() << "AVX512-VNNI is not available on this CPU";
  }
  auto prepared_query = query;
  preprocess(prepared_query.data(), prepared_query.size());
  batch_distance(record_pointers.data(), prepared_query.data(), kVectorCount,
                 query_codes.size() + kTailBytes, distances.data(),
                 extra_value_pointers.data());

  for (size_t i = 0; i < kVectorCount; ++i) {
    EXPECT_FLOAT_EQ(static_cast<float>(SquaredL2(record_codes[i], query_codes)),
                    distances[i]);
  }
}

TEST(UniformUint8Metric,
     ExactQueryDistanceSupportsUint32RangeAtMaximumDimension) {
  constexpr size_t kDimension = MAX_DIMENSION;

  auto metric = CreateMetric(kDimension);
  ASSERT_TRUE(metric);
  auto query_metric = metric->query_metric();
  ASSERT_TRUE(query_metric);
  ASSERT_TRUE(query_metric->distance());
  ASSERT_TRUE(query_metric->batch_distance());

  const auto verify = [&](uint8_t record_code, uint8_t query_code) {
    constexpr size_t kVectorCount = 4;
    const std::vector<uint8_t> record_codes(kDimension, record_code);
    const std::vector<uint8_t> query_codes(kDimension, query_code);
    const auto record = EncodeRecord(record_codes);
    const auto query = EncodeQuery(query_codes);
    const int64_t expected = SquaredL2(record_codes, query_codes);
    EXPECT_EQ(static_cast<int64_t>(kDimension) * 255 * 255, expected);
    ASSERT_GT(expected, (std::numeric_limits<int32_t>::max)());
    ASSERT_LE(static_cast<uint64_t>(expected),
              (std::numeric_limits<uint32_t>::max)());

    const auto prepared_query = PrepareQuery(query_metric, query);
    float scalar_distance = 0.0f;
    query_metric->distance()(record.data(), prepared_query.data(),
                             kDimension + kTailBytes, &scalar_distance);
    EXPECT_FLOAT_EQ(static_cast<float>(expected), scalar_distance);

    const void *records[kVectorCount] = {record.data(), record.data(),
                                         record.data(), record.data()};
    const void *extra_values[kVectorCount] = {
        record.data() + kDimension, record.data() + kDimension,
        record.data() + kDimension, record.data() + kDimension};
    float batch_distances[kVectorCount] = {};
    query_metric->batch_distance()(records, prepared_query.data(), kVectorCount,
                                   kDimension + kTailBytes, batch_distances,
                                   extra_values);
    for (float distance : batch_distances) {
      EXPECT_FLOAT_EQ(static_cast<float>(expected), distance);
    }
  };

  verify(/*record_code=*/255, /*query_code=*/0);
  verify(/*record_code=*/0, /*query_code=*/255);

  const auto full_range_record =
      EncodeRecord(std::vector<uint8_t>(kDimension, uint8_t{255}));
  EXPECT_EQ(static_cast<uint64_t>(kDimension) * 255 * 255,
            ReadTail(full_range_record, kDimension));
}

TEST(UniformUint8Metric, ExactQueryDistanceFallsBackAboveTurboDimensionLimit) {
  constexpr size_t kDimension = MAX_DIMENSION + 4096;

  auto metric = CreateMetric(kDimension);
  ASSERT_TRUE(metric);
  auto query_metric = metric->query_metric();
  ASSERT_TRUE(query_metric);
  ASSERT_TRUE(query_metric->distance());
  ASSERT_TRUE(query_metric->batch_distance());

  const auto verify = [&](uint8_t record_code, uint8_t query_code) {
    constexpr size_t kVectorCount = 4;
    const std::vector<uint8_t> record_codes(kDimension, record_code);
    const std::vector<uint8_t> query_codes(kDimension, query_code);
    const auto record = EncodeRecord(record_codes);
    const auto query = EncodeQuery(query_codes);
    const int64_t expected = SquaredL2(record_codes, query_codes);
    ASSERT_GT(expected, (std::numeric_limits<uint32_t>::max)());

    const auto prepared_query = PrepareQuery(query_metric, query);
    float scalar_distance = 0.0f;
    query_metric->distance()(record.data(), prepared_query.data(),
                             kDimension + kTailBytes, &scalar_distance);
    EXPECT_FLOAT_EQ(static_cast<float>(expected), scalar_distance);

    const void *records[kVectorCount] = {record.data(), record.data(),
                                         record.data(), record.data()};
    const void *extra_values[kVectorCount] = {
        record.data() + kDimension, record.data() + kDimension,
        record.data() + kDimension, record.data() + kDimension};
    float distances[kVectorCount] = {};
    query_metric->batch_distance()(records, prepared_query.data(), kVectorCount,
                                   kDimension + kTailBytes, distances,
                                   extra_values);
    for (float distance : distances) {
      EXPECT_FLOAT_EQ(static_cast<float>(expected), distance);
    }
  };

  verify(/*record_code=*/0, /*query_code=*/255);
  verify(/*record_code=*/255, /*query_code=*/0);
}

TEST(UniformUint8Metric, RejectsEncodedDimensionWithoutVectorData) {
  for (const uint32_t encoded_dimension :
       {0U, static_cast<uint32_t>(kTailBytes)}) {
    auto metric = IndexFactory::CreateMetric("UniformUint8");
    ASSERT_TRUE(metric);
    IndexMeta meta(IndexMeta::DataType::DT_INT8, encoded_dimension);
    ailego::Params params;
    params.set(UNIFORM_UINT8_METRIC_ORIGIN_METRIC_NAME,
               std::string("SquaredEuclidean"));
    EXPECT_NE(0, metric->init(meta, params))
        << "encoded_dimension=" << encoded_dimension;
  }
}

TEST(UniformUint8Metric, BuildDistanceIsExactForFullRangeAndLargeDimensions) {
  std::mt19937 generator(20260716);
  std::uniform_int_distribution<int> byte_distribution(0, 255);

  for (const size_t dimension : {1UL, 31UL, 32UL, 33UL, 127UL, 128UL, 129UL,
                                 1024UL, 65536UL, 1048577UL}) {
    std::vector<uint8_t> lhs_codes(dimension);
    std::vector<uint8_t> rhs_codes(dimension);
    for (size_t i = 0; i < dimension; ++i) {
      lhs_codes[i] = static_cast<uint8_t>(byte_distribution(generator));
      rhs_codes[i] = static_cast<uint8_t>(byte_distribution(generator));
    }
    auto lhs = EncodeRecord(lhs_codes);
    auto rhs = EncodeRecord(rhs_codes);

    auto metric = CreateMetric(dimension);
    ASSERT_TRUE(metric);
    float distance = 0.0f;
    metric->distance()(lhs.data(), rhs.data(), dimension + kTailBytes,
                       &distance);
    EXPECT_FLOAT_EQ(static_cast<float>(SquaredL2(lhs_codes, rhs_codes)),
                    distance)
        << "dimension=" << dimension;

    std::fill(lhs_codes.begin(), lhs_codes.end(), uint8_t{0});
    std::fill(rhs_codes.begin(), rhs_codes.end(), uint8_t{255});
    lhs = EncodeRecord(lhs_codes);
    rhs = EncodeRecord(rhs_codes);
    metric->distance()(lhs.data(), rhs.data(), dimension + kTailBytes,
                       &distance);
    EXPECT_FLOAT_EQ(static_cast<float>(SquaredL2(lhs_codes, rhs_codes)),
                    distance)
        << "extreme dimension=" << dimension;
  }
}

TEST(UniformUint8Metric, RejectsUnsupportedOriginMetric) {
  auto metric = IndexFactory::CreateMetric("UniformUint8");
  ASSERT_TRUE(metric);
  IndexMeta meta(IndexMeta::DataType::DT_INT8, 16 + kTailBytes);
  ailego::Params params;
  params.set(UNIFORM_UINT8_METRIC_ORIGIN_METRIC_NAME,
             std::string("InnerProduct"));
  EXPECT_NE(0, metric->init(meta, params));
}

}  // namespace
}  // namespace zvec::core
