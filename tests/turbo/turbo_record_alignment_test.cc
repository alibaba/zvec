// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/turbo/turbo.h>
#include "quantizer/record_quantizer.h"

namespace zvec::turbo {
namespace {

TEST(RecordAlignment, QuantizerPreservesPackedLayoutAtEveryOffset) {
  for (bool int4 : {false, true}) {
    for (bool l2 : {false, true}) {
      for (size_t dim : {2U, 6U, 18U, 32U, 130U}) {
        const auto type =
            int4 ? core::IndexMeta::DT_INT4 : core::IndexMeta::DT_INT8;
        const size_t codes = int4 ? dim / 2 : dim;
        std::vector<float> input(dim), decoded(dim);
        std::vector<uint8_t> expected(codes + (int4 ? 16 : 20), 0);
        for (size_t d = 0; d < dim; ++d) input[d] = d % 2 ? 1.0f : -1.0f;
        for (size_t d = 0; d < codes; ++d)
          expected[d] = int4 ? 0x78 : static_cast<uint8_t>(d % 2 ? 127 : -127);
        const float tail[] = {
            int4 ? 2.0f / 15 : 1.0f / 127, int4 ? 1.0f / 15 : -0.0f,
            int4 ? -float(dim) / 2 : 0,
            int4 ? (l2 ? 113.0f * dim / 2 : 0) : 16129.0f * dim};
        std::memcpy(expected.data() + codes, tail, sizeof(tail));
        for (size_t offset = 0; offset < 8; ++offset) {
          SCOPED_TRACE(testing::Message()
                       << "int4=" << int4 << " l2=" << l2 << " dim=" << dim
                       << " offset=" << offset);
          std::vector<uint8_t> storage(16 + offset + expected.size() + 16,
                                       0xa5);
          auto *record = storage.data() + 16 + offset;
          core::RecordQuantizer::quantize_record(input.data(), dim, type, l2,
                                                 record);
          EXPECT_EQ(0, std::memcmp(expected.data(), record, expected.size()));
          core::RecordQuantizer::unquantize_record(record, dim, type,
                                                   decoded.data());
          for (size_t d = 0; d < dim; ++d)
            EXPECT_NEAR(input[d], decoded[d], 2e-7f);
          EXPECT_TRUE(std::all_of(storage.begin(),
                                  storage.begin() + 16 + offset,
                                  [](uint8_t v) { return v == 0xa5; }));
          EXPECT_TRUE(std::all_of(storage.end() - 16, storage.end(),
                                  [](uint8_t v) { return v == 0xa5; }));
        }
      }
    }
  }
}

class RecordAlignmentBackendTest : public testing::TestWithParam<CpuArchType> {
};

TEST_P(RecordAlignmentBackendTest, SingleBatchAndManyMatchDecodedValues) {
  bool supported = false;
  for (auto type : {DataType::kInt8, DataType::kInt4}) {
    const bool int4 = type == DataType::kInt4;
    for (auto metric : {MetricType::kInnerProduct,
                        MetricType::kSquaredEuclidean, MetricType::kCosine}) {
      const auto kernels =
          get_distance_kernels(metric, type, QuantizeType::kRecord, GetParam());
      if (!kernels.dist || !kernels.batch) continue;
      supported = true;
      auto many = get_many_to_many_distance_func(
          metric, type, QuantizeType::kRecord, GetParam());
      ASSERT_TRUE(many);
      for (size_t dim : {2U, 18U, 32U, 66U, 128U}) {
        const size_t codes = int4 ? dim / 2 : dim;
        const size_t tail =
            (int4 ? 16 : 20) + (metric == MetricType::kCosine ? 4 : 0);
        const size_t bytes = codes + tail, units = int4 ? bytes * 2 : bytes;
        for (size_t offset : {0U, 1U, 2U, 3U, 7U}) {
          SCOPED_TRACE(testing::Message()
                       << "arch=" << int(GetParam()) << " type=" << int(type)
                       << " metric=" << int(metric) << " dim=" << dim
                       << " offset=" << offset);
          constexpr size_t Records = 32, Queries = 3;
          std::vector<std::vector<uint8_t>> storage(
              Records + Queries, std::vector<uint8_t>(offset + bytes + 8, 0));
          std::vector<std::vector<double>> decoded(Records + Queries,
                                                   std::vector<double>(dim));
          for (size_t row = 0; row < storage.size(); ++row) {
            auto *data = storage[row].data() + offset;
            const float scale = (1 + row % 3) * 0.25f,
                        bias = (int(row % 5) - 2) * 0.5f;
            int32_t sum = 0;
            float squares = 0;
            for (size_t d = 0; d < dim; ++d) {
              const int code =
                  int((row * 7 + d * 3) % (int4 ? 16 : 255)) - (int4 ? 8 : 127);
              if (int4)
                data[d / 2] |= (code & 15) << (4 * (d % 2));
              else
                data[d] = static_cast<uint8_t>(code);
              sum += code;
              squares += code * code;
              decoded[row][d] = scale * code + bias;
            }
            const float params[] = {scale, bias, float(sum), squares};
            std::memcpy(data + codes, params, sizeof(params));
            if (!int4) std::memcpy(data + codes + 16, &sum, sizeof(sum));
          }
          const auto original = storage;
          const void *vectors[Records], *queries[Queries], *extras[Records];
          for (size_t i = 0; i < Records; ++i) {
            vectors[i] = storage[i].data() + offset;
            extras[i] = storage[i].data() + offset + codes;
          }
          auto prepared = storage;
          for (size_t q = 0; q < Queries; ++q) {
            auto *query = prepared[Records + q].data() + offset;
            if (kernels.preprocess) kernels.preprocess(query, units);
            queries[q] = query;
          }
          for (size_t count : {1U, 3U, 4U, 5U, 9U, 32U}) {
            std::vector<float> batch(count + 2, -12345),
                result(count * Queries + 2, -12345);
            many(vectors, queries, count, Queries, units, result.data() + 1,
                 extras);
            for (size_t q = 0; q < Queries; ++q) {
              kernels.batch(vectors, queries[q], count, units, batch.data() + 1,
                            extras);
              for (size_t i = 0; i < count; ++i) {
                double expected = 0, magnitude = 1;
                for (size_t d = 0; d < dim; ++d) {
                  const double x = decoded[i][d], y = decoded[Records + q][d];
                  const double term = metric == MetricType::kSquaredEuclidean
                                          ? (x - y) * (x - y)
                                          : -x * y;
                  expected += term;
                  magnitude += std::abs(term);
                }
                float single;
                kernels.dist(vectors[i], storage[Records + q].data() + offset,
                             units, &single);
                EXPECT_NEAR(expected, single, 2e-5 * magnitude);
                EXPECT_NEAR(expected, batch[1 + i], 2e-5 * magnitude);
                EXPECT_NEAR(expected, result[1 + q * count + i],
                            2e-5 * magnitude);
              }
            }
            EXPECT_EQ(-12345, batch.front());
            EXPECT_EQ(-12345, batch.back());
            EXPECT_EQ(-12345, result.front());
            EXPECT_EQ(-12345, result.back());
          }
          EXPECT_EQ(original, storage);
        }
      }
    }
  }
  if (!supported) GTEST_SKIP() << "Record backend unavailable on this CPU";
}

INSTANTIATE_TEST_SUITE_P(Architectures, RecordAlignmentBackendTest,
                         testing::Values(CpuArchType::kScalar,
                                         CpuArchType::kAuto, CpuArchType::kNEON,
                                         CpuArchType::kSSE2, CpuArchType::kAVX2,
                                         CpuArchType::kAVX512,
                                         CpuArchType::kAVX512VNNI));

}  // namespace
}  // namespace zvec::turbo
