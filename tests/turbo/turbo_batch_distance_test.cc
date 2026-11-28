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
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <turbo/quantizer/fp32_quantizer/fp32_quantizer.h>
#include <zvec/core/framework/index_factory.h>

namespace zvec {
namespace turbo {
namespace {

float ReadFloat(const void *data) {
  float value;
  std::memcpy(&value, data, sizeof(value));
  return value;
}

std::string EncodeFloat(float value) {
  return std::string(reinterpret_cast<const char *>(&value), sizeof(value));
}

using PaddedRow = std::array<float, 5>;

std::vector<PaddedRow> MakeRows(size_t count) {
  std::vector<PaddedRow> rows(count);
  for (size_t i = 0; i < count; ++i) {
    rows[i] = {static_cast<float>(i), -100.0f, -200.0f, 300.0f, 400.0f};
  }
  return rows;
}

TEST(BatchDistance, PointerOrderAndPaddingArePreserved) {
  auto rows = MakeRows(65);
  std::vector<const void *> candidates(rows.size());
  for (size_t i = 0; i < rows.size(); ++i) {
    candidates[i] = rows[(i * 7) % rows.size()].data();
  }
  size_t calls = 0;
  DistanceFunc single = [](const void *, const void *, size_t, float *) {
    ADD_FAILURE() << "An available batch kernel must be used";
  };
  BatchDistanceFunc batch = [&](const void **input, const void *query,
                                size_t num, size_t dim, float *out,
                                const void **extra_values) {
    ++calls;
    EXPECT_EQ(3u, dim);
    EXPECT_EQ(nullptr, extra_values);
    for (size_t i = 0; i < num; ++i) {
      EXPECT_EQ(candidates[i], input[i]);
      out[i] = ReadFloat(input[i]) + ReadFloat(query);
    }
  };
  DistanceImpl distance(single, batch, EncodeFloat(7.0f), 3);
  std::vector<float> out(rows.size() + 2, -1.0f);
  distance.batch(candidates.data(), candidates.size(), out.data() + 1);
  EXPECT_EQ(1u, calls);
  EXPECT_FLOAT_EQ(-1.0f, out.front());
  EXPECT_FLOAT_EQ(-1.0f, out.back());
  for (size_t i = 0; i < candidates.size(); ++i) {
    EXPECT_FLOAT_EQ(ReadFloat(candidates[i]) + 7.0f, out[i + 1]);
  }
}

TEST(BatchDistance, SingleKernelFallback) {
  auto rows = MakeRows(33);
  std::vector<const void *> candidates(rows.size());
  for (size_t i = 0; i < rows.size(); ++i) candidates[i] = rows[i].data();
  size_t single_calls = 0;
  DistanceFunc single = [&](const void *candidate, const void *query,
                            size_t dim, float *out) {
    EXPECT_EQ(3u, dim);
    EXPECT_EQ(candidates[single_calls], candidate);
    *out = ReadFloat(candidate) + ReadFloat(query);
    ++single_calls;
  };
  DistanceImpl distance(single, EncodeFloat(7.0f), 3);
  std::vector<float> out(rows.size());
  distance.batch(candidates.data(), candidates.size(), out.data());
  EXPECT_EQ(rows.size(), single_calls);
  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_FLOAT_EQ(static_cast<float>(i) + 7.0f, out[i]);
  }
  distance.batch(nullptr, 0, nullptr);
  EXPECT_EQ(rows.size(), single_calls);
  DistanceImpl{}.batch(nullptr, 0, nullptr);
}

TEST(BatchDistance, EncodedRowsMatchSinglePairWithReorderedPointers) {
  constexpr uint32_t kDimension = 34;
  constexpr size_t kCount = 65;
  const char *quantizer_names[] = {"Fp32Quantizer", "Fp16Quantizer",
                                   "Int8Quantizer", "Int4Quantizer"};
  const char *metrics[] = {"SquaredEuclidean", "InnerProduct", "Cosine"};
  const size_t counts[] = {1, 15, 16, 17, 31, 32, 33, 65};

  for (const char *quantizer_name : quantizer_names) {
    const float tolerance =
        std::string(quantizer_name) == "Fp16Quantizer" ? 2e-3f : 1e-4f;
    for (const char *metric : metrics) {
      SCOPED_TRACE(testing::Message()
                   << "quantizer=" << quantizer_name << ", metric=" << metric);
      core::IndexMeta meta;
      meta.set_meta(core::IndexMeta::DT_FP32, kDimension);
      meta.set_metric(metric, 0, ailego::Params());
      auto quantizer = core::IndexFactory::CreateQuantizer(quantizer_name);
      ASSERT_TRUE(quantizer);
      ASSERT_EQ(0, quantizer->init(meta, ailego::Params()));

      std::vector<float> raw(kDimension);
      for (size_t d = 0; d < kDimension; ++d) {
        raw[d] = static_cast<float>(static_cast<int>(d % 11) - 5) * 0.125f;
      }
      std::string query;
      core::IndexQueryMeta query_meta;
      ASSERT_EQ(0,
                quantizer->quantize(
                    raw.data(),
                    core::IndexQueryMeta(core::IndexMeta::DT_FP32, kDimension),
                    &query, &query_meta));
      const std::string original_query = query;
      const auto distance = quantizer->distance(query.data(), query_meta);
      ASSERT_TRUE(distance.valid());

      const size_t code_size = quantizer->quantized_datapoint_vector_length();
      // Both IVF's encoded row width (including record/cosine metadata) and
      // an explicitly padded stride must be independent of logical dimension.
      for (size_t stride : {code_size, ((code_size + 15) / 16 + 1) * 16}) {
        SCOPED_TRACE(testing::Message() << "stride=" << stride);
        std::string rows(kCount * stride, '\x5a');
        std::vector<const void *> candidates(kCount);
        for (size_t i = 0; i < kCount; ++i) {
          for (size_t d = 0; d < kDimension; ++d) {
            raw[d] = static_cast<float>(static_cast<int>((i * 7 + d * 3) % 29) -
                                        14) *
                     0.125f;
          }
          quantizer->quantize_data(raw.data(), rows.data() + i * stride);
          candidates[i] = rows.data() + i * stride;
        }
        const std::string original_rows = rows;
        std::reverse(candidates.begin(), candidates.end());
        std::rotate(candidates.begin(), candidates.begin() + 7,
                    candidates.end());

        for (size_t count : counts) {
          SCOPED_TRACE(testing::Message() << "count=" << count);
          std::vector<float> expected(count);
          std::vector<float> actual(count + 2, -1000.0f);
          for (size_t i = 0; i < count; ++i) {
            expected[i] =
                quantizer->calc_distance_dp_query(candidates[i], query.data());
          }
          quantizer->calc_distance_dp_query_batch(
              candidates.data(), static_cast<int>(count), query.data(),
              actual.data() + 1);
          EXPECT_EQ(original_query, query);
          EXPECT_FLOAT_EQ(-1000.0f, actual.front());
          EXPECT_FLOAT_EQ(-1000.0f, actual.back());
          for (size_t i = 0; i < count; ++i) {
            EXPECT_NEAR(expected[i], actual[i + 1],
                        tolerance * std::max(1.0f, std::abs(expected[i])));
          }

          // The handle owns its query and preserves its single-pair
          // contract, including any INT8 preprocessing fallback.
          for (size_t i = 0; i < count; ++i)
            expected[i] = distance(candidates[i]);
          distance.batch(candidates.data(), count, actual.data() + 1);
          EXPECT_EQ(original_query, distance.query_storage());
          EXPECT_FLOAT_EQ(-1000.0f, actual.front());
          EXPECT_FLOAT_EQ(-1000.0f, actual.back());
          for (size_t i = 0; i < count; ++i) {
            EXPECT_NEAR(expected[i], actual[i + 1],
                        tolerance * std::max(1.0f, std::abs(expected[i])));
          }
        }
        EXPECT_EQ(original_rows, rows);
      }
    }
  }
}

}  // namespace
}  // namespace turbo
}  // namespace zvec
