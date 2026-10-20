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

TEST(StridedDistance, NativeCallbackTakesPrecedence) {
  auto rows = MakeRows(65);
  size_t native_calls = 0;
  const void *received_query = nullptr;
  DistanceFunc single = [](const void *, const void *, size_t, float *) {
    ADD_FAILURE() << "Native strided distance must bypass the single kernel";
  };
  BatchDistanceFunc batch = [](const void **, const void *, size_t, size_t,
                               float *, const void **) {
    ADD_FAILURE() << "Native strided distance must bypass the gather kernel";
  };
  StridedBatchDistanceFunc strided =
      [&](const void *base, size_t stride, const void *query, size_t num,
          size_t dim, float *out, const void **extra_values) {
        ++native_calls;
        EXPECT_EQ(rows.data(), base);
        EXPECT_EQ(sizeof(PaddedRow), stride);
        EXPECT_EQ(rows.size(), num);
        EXPECT_EQ(3u, dim);
        EXPECT_EQ(nullptr, extra_values);
        EXPECT_FLOAT_EQ(7.0f, ReadFloat(query));
        received_query = query;
        for (size_t i = 0; i < num; ++i) {
          out[i] = ReadFloat(static_cast<const char *>(base) + i * stride) +
                   ReadFloat(query);
        }
      };
  DistanceImpl distance(single, batch, EncodeFloat(7.0f), 3, strided);
  std::vector<float> out(rows.size());
  distance.batch_strided(rows.data(), sizeof(PaddedRow), rows.size(),
                         out.data());
  EXPECT_EQ(1u, native_calls);
  EXPECT_EQ(distance.query_storage().data(), received_query);
  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_FLOAT_EQ(static_cast<float>(i) + 7.0f, out[i]);
  }

  distance.batch_strided(nullptr, 0, 0, nullptr);
  EXPECT_EQ(1u, native_calls);
}

TEST(StridedDistance, GatherFallbackPreservesStrideAndTail) {
  auto rows = MakeRows(65);
  std::vector<size_t> batch_sizes;
  size_t processed = 0;
  DistanceFunc single = [](const void *, const void *, size_t, float *) {
    ADD_FAILURE() << "An available gather kernel must be used as the fallback";
  };
  BatchDistanceFunc batch = [&](const void **candidates, const void *query,
                                size_t num, size_t dim, float *out,
                                const void **extra_values) {
    batch_sizes.push_back(num);
    EXPECT_EQ(3u, dim);
    EXPECT_EQ(nullptr, extra_values);
    for (size_t i = 0; i < num; ++i) {
      EXPECT_EQ(rows[processed + i].data(), candidates[i]);
      out[i] = ReadFloat(candidates[i]) + ReadFloat(query);
    }
    processed += num;
  };
  DistanceImpl distance(single, batch, EncodeFloat(7.0f), 3);
  std::vector<float> out(rows.size() + 2, -1.0f);
  distance.batch_strided(rows.data(), sizeof(PaddedRow), rows.size(),
                         out.data() + 1);
  EXPECT_EQ((std::vector<size_t>{32, 32, 1}), batch_sizes);
  EXPECT_EQ(rows.size(), processed);
  EXPECT_FLOAT_EQ(-1.0f, out.front());
  EXPECT_FLOAT_EQ(-1.0f, out.back());
  for (size_t i = 0; i < rows.size(); ++i) {
    EXPECT_FLOAT_EQ(static_cast<float>(i) + 7.0f, out[i + 1]);
  }

  distance.batch_strided(nullptr, 0, 0, nullptr);
  EXPECT_EQ(3u, batch_sizes.size());
}

TEST(StridedDistance, SingleKernelFallback) {
  auto rows = MakeRows(33);
  size_t single_calls = 0;
  DistanceFunc single = [&](const void *candidate, const void *query,
                            size_t dim, float *out) {
    EXPECT_EQ(3u, dim);
    EXPECT_EQ(rows[single_calls].data(), candidate);
    *out = ReadFloat(candidate) + ReadFloat(query);
    ++single_calls;
  };
  DistanceImpl distance(single, EncodeFloat(7.0f), 3);
  std::vector<float> out(rows.size());
  distance.batch_strided(rows.data(), sizeof(PaddedRow), rows.size(),
                         out.data());
  EXPECT_EQ(rows.size(), single_calls);
  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_FLOAT_EQ(static_cast<float>(i) + 7.0f, out[i]);
  }

  distance.batch_strided(nullptr, 0, 0, nullptr);
  EXPECT_EQ(rows.size(), single_calls);
  DistanceImpl{}.batch_strided(nullptr, 0, 0, nullptr);
}

class CountingBatchQuantizer : public Fp32Quantizer {
 public:
  float calc_distance_dp_query(const void *, const void *) const override {
    ADD_FAILURE() << "Quantizer fallback must preserve virtual batch dispatch";
    return 0.0f;
  }

  void calc_distance_dp_query_batch(const void *const *candidates, int num,
                                    const void *query,
                                    float *out) const override {
    batch_sizes.push_back(num);
    for (int i = 0; i < num; ++i) {
      out[i] = ReadFloat(candidates[i]) + ReadFloat(query);
    }
  }

  mutable std::vector<int> batch_sizes;
};

TEST(StridedDistance, QuantizerFallbackUsesVirtualGatherBatch) {
  CountingBatchQuantizer quantizer;
  auto rows = MakeRows(65);
  const float query = 7.0f;
  std::vector<float> out(rows.size());
  const Quantizer &base = quantizer;
  base.calc_distance_dp_query_batch_strided(rows.data(), sizeof(PaddedRow),
                                            static_cast<int>(rows.size()),
                                            &query, out.data());
  EXPECT_EQ((std::vector<int>{32, 32, 1}), quantizer.batch_sizes);
  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_FLOAT_EQ(static_cast<float>(i) + query, out[i]);
  }
  base.calc_distance_dp_query_batch_strided(nullptr, 0, 0, nullptr, nullptr);
  EXPECT_EQ(3u, quantizer.batch_sizes.size());
}

TEST(StridedDistance, EncodedRowsMatchExistingBatch) {
  constexpr uint32_t kDimension = 34;
  constexpr size_t kCount = 65;
  const char *quantizer_names[] = {"Fp32Quantizer", "Fp16Quantizer",
                                   "Int8Quantizer", "Int4Quantizer"};
  const char *metrics[] = {"SquaredEuclidean", "InnerProduct", "Cosine"};
  const size_t counts[] = {0, 1, 31, 32, 33, 65};

  for (const char *quantizer_name : quantizer_names) {
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

        for (size_t count : counts) {
          SCOPED_TRACE(testing::Message() << "count=" << count);
          if (count == 0) {
            quantizer->calc_distance_dp_query_batch_strided(nullptr, stride, 0,
                                                            nullptr, nullptr);
            distance.batch_strided(nullptr, stride, 0, nullptr);
            continue;
          }
          std::vector<float> expected(count);
          std::vector<float> actual(count + 2, -1000.0f);
          quantizer->calc_distance_dp_query_batch(
              candidates.data(), static_cast<int>(count), query.data(),
              expected.data());
          quantizer->calc_distance_dp_query_batch_strided(
              rows.data(), stride, static_cast<int>(count), query.data(),
              actual.data() + 1);
          EXPECT_EQ(original_query, query);
          EXPECT_FLOAT_EQ(-1000.0f, actual.front());
          EXPECT_FLOAT_EQ(-1000.0f, actual.back());
          for (size_t i = 0; i < count; ++i) {
            EXPECT_NEAR(expected[i], actual[i + 1],
                        1e-4f * std::max(1.0f, std::abs(expected[i])));
          }

          // DistanceImpl may deliberately use different kernels than the
          // quantizer (e.g. INT8 query preprocessing); compare each path to
          // its own gather batch contract.
          distance.batch(candidates.data(), count, expected.data());
          distance.batch_strided(rows.data(), stride, count, actual.data() + 1);
          EXPECT_EQ(original_query, distance.query_storage());
          EXPECT_FLOAT_EQ(-1000.0f, actual.front());
          EXPECT_FLOAT_EQ(-1000.0f, actual.back());
          for (size_t i = 0; i < count; ++i) {
            EXPECT_NEAR(expected[i], actual[i + 1],
                        1e-4f * std::max(1.0f, std::abs(expected[i])));
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
