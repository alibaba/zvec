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

#include "ivf_distance_calculator.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/utility/float_helper.h>

using namespace zvec::core;
using zvec::ailego::FloatHelper;
using zvec::ailego::Params;

namespace {

// A symmetric metric with observable entry points. Trailing record bytes are
// deliberately unrelated to the score, so a payload-only stride is detectable.
class CountingMetric : public IndexMetric {
 public:
  explicit CountingMetric(bool has_batch = true, size_t extra_values = 0,
                          bool query_specific = false)
      : has_batch_(has_batch),
        extra_values_(extra_values),
        query_specific_(query_specific) {}

  int init(const IndexMeta &, const Params &) override {
    return 0;
  }
  int cleanup() override {
    return 0;
  }
  bool is_matched(const IndexMeta &) const override {
    return true;
  }
  bool is_matched(const IndexMeta &, const IndexQueryMeta &) const override {
    return true;
  }
  const Params &params() const override {
    return params_;
  }
  Pointer query_metric() const override {
    return query_specific_ ? std::make_shared<CountingMetric>() : nullptr;
  }

  MatrixDistance distance() const override {
    return [this](const void *a, const void *b, size_t, float *out) {
      ++pair_calls;
      *out = Read(a) + Read(b);
    };
  }

  MatrixBatchDistance batch_distance() const override {
    ++batch_lookups;
    if (!has_batch_) return nullptr;
    return [this](const void **rows, const void *query, size_t count,
                  size_t dim, float *out, const void **extra) {
      EXPECT_EQ(nullptr, extra);
      EXPECT_EQ(1u, dim);
      batch_counts.push_back(count);
      batch_queries.push_back(query);
      for (size_t i = 0; i < count; ++i) {
        batch_rows.push_back(rows[i]);
        out[i] = Read(rows[i]) + Read(query);
      }
    };
  }

  size_t extra_values_size_per_vector() const override {
    return extra_values_;
  }

  MatrixDistance distance_matrix(size_t m, size_t n) const override {
    return [this, m, n](const void *, const void *, size_t, float *out) {
      ++matrix_calls;
      matrix_rows = m;
      matrix_queries = n;
      std::fill_n(out, m * n, -123.0f);
    };
  }

  void clear_calls() {
    pair_calls = matrix_calls = 0;
    batch_counts.clear();
    batch_queries.clear();
    batch_rows.clear();
  }

  mutable size_t pair_calls{0}, matrix_calls{0}, batch_lookups{0};
  mutable size_t matrix_rows{0}, matrix_queries{0};
  mutable std::vector<size_t> batch_counts;
  mutable std::vector<const void *> batch_queries, batch_rows;

 private:
  static float Read(const void *p) {
    uint16_t value;
    std::memcpy(&value, p, sizeof(value));
    return value;
  }
  bool has_batch_;
  size_t extra_values_;
  bool query_specific_;
  Params params_;
};

IndexMeta CountingMeta(IndexMeta::MajorOrder order = IndexMeta::MO_ROW) {
  IndexMeta meta(IndexMeta::DT_FP16, 1);
  meta.set_metric("Cosine", 0, Params());
  meta.set_extra_meta_size(4);
  meta.set_major_order(order);
  return meta;
}

std::vector<uint16_t> Records(const IndexMeta &meta, size_t count,
                              uint16_t first) {
  const size_t stride = meta.element_size() / sizeof(uint16_t);
  std::vector<uint16_t> records(count * stride, 0xffffu);
  for (size_t i = 0; i < count; ++i) records[i * stride] = first + i;
  return records;
}

}  // namespace

TEST(IVFDistanceCalculator, Fp16RowBatchKeepsRecordStrideAndSingletonTails) {
  const auto meta = CountingMeta();
  ASSERT_EQ(6u, meta.element_size());
  auto metric = std::make_shared<CountingMetric>();
  IVFDistanceCalculator calculator(meta, metric, 32);
  const auto rows = Records(meta, 65, 1);
  const auto query = Records(meta, 1, 1000);
  struct Case {
    size_t count;
    std::vector<size_t> batches;
    size_t pairs;
  };
  for (const auto &test : std::vector<Case>{{0, {}, 0},
                                            {1, {}, 1},
                                            {2, {2}, 0},
                                            {32, {32}, 0},
                                            {33, {32}, 1},
                                            {65, {32, 32}, 1}}) {
    SCOPED_TRACE(test.count);
    metric->clear_calls();
    std::vector<float> out(test.count + 1, -999.0f);
    calculator.query_features_distance(test.count ? query.data() : nullptr,
                                       test.count ? rows.data() : nullptr,
                                       test.count, out.data());
    EXPECT_EQ(test.batches, metric->batch_counts);
    EXPECT_EQ(test.pairs, metric->pair_calls);
    EXPECT_EQ(0u, metric->matrix_calls);
    EXPECT_EQ(1u, metric->batch_lookups);  // Cached once by the calculator.
    EXPECT_FLOAT_EQ(-999.0f, out.back());
    for (size_t i = 0; i < test.count; ++i)
      EXPECT_FLOAT_EQ(1001.0f + i, out[i]);
    for (size_t i = 0; i < metric->batch_rows.size(); ++i)
      EXPECT_EQ(rows.data() + 3 * i, metric->batch_rows[i]);
    for (const void *p : metric->batch_queries) EXPECT_EQ(query.data(), p);
  }
}

TEST(IVFDistanceCalculator, UnavailableOrIneligibleBatchKeepsPairwiseRoute) {
  for (int variant = 0; variant < 4; ++variant) {
    SCOPED_TRACE(variant);
    auto meta = CountingMeta();
    if (variant == 1) meta.set_meta(IndexMeta::DT_FP32, 1);
    auto metric = std::make_shared<CountingMetric>(
        variant != 0, variant == 2 ? 4 : 0, variant == 3);
    IVFDistanceCalculator calculator(meta, metric, 32);
    const auto rows = Records(meta, 33, 1);
    const auto query = Records(meta, 1, 1000);
    std::vector<float> out(34, -999.0f);
    calculator.query_features_distance(query.data(), rows.data(), 33,
                                       out.data());
    EXPECT_EQ(33u, metric->pair_calls);
    EXPECT_TRUE(metric->batch_counts.empty());
    for (size_t i = 0; i < 33; ++i) EXPECT_FLOAT_EQ(1001.0f + i, out[i]);
    EXPECT_FLOAT_EQ(-999.0f, out.back());
  }
}

TEST(IVFDistanceCalculator,
     CentroidQueriesPreserveOutputStridesAndColumnMatrix) {
  for (auto order : {IndexMeta::MO_ROW, IndexMeta::MO_COLUMN}) {
    SCOPED_TRACE(order);
    const auto meta = CountingMeta(order);
    auto metric = std::make_shared<CountingMetric>();
    IVFDistanceCalculator calculator(meta, metric, 32);
    const auto rows = Records(meta, 32, 1);
    const auto queries = Records(meta, 2, 1000);
    std::vector<float> out(65, -999.0f);
    calculator.query_centroids_distance(queries.data(), 2, rows.data(), 3,
                                        out.data());
    EXPECT_EQ((std::vector<size_t>{3, 3}), metric->batch_counts);
    ASSERT_EQ(2u, metric->batch_queries.size());
    EXPECT_EQ(queries.data(), metric->batch_queries[0]);
    EXPECT_EQ(queries.data() + 3, metric->batch_queries[1]);
    EXPECT_EQ(0u, metric->pair_calls);
    EXPECT_EQ(0u, metric->matrix_calls);
    // Row-major centroid output reserves a physical block per query. The
    // partial column-major matrix fallback writes a packed q-by-fnum result.
    const size_t stride = order == IndexMeta::MO_ROW ? 32 : 3;
    for (size_t i = 0; i < out.size(); ++i) {
      if (i < 3)
        EXPECT_FLOAT_EQ(1001.0f + i, out[i]);
      else if (i >= stride && i < stride + 3)
        EXPECT_FLOAT_EQ(1002.0f + i - stride, out[i]);
      else
        EXPECT_FLOAT_EQ(-999.0f, out[i]);
    }
    if (order == IndexMeta::MO_COLUMN) {
      metric->clear_calls();
      calculator.query_features_distance(queries.data(), rows.data(), 32,
                                         out.data());
      EXPECT_EQ(1u, metric->matrix_calls);
      EXPECT_EQ(32u, metric->matrix_rows);
      EXPECT_EQ(1u, metric->matrix_queries);
      EXPECT_EQ(0u, metric->pair_calls);
      EXPECT_TRUE(metric->batch_counts.empty());
      metric->clear_calls();
      calculator.query_centroids_distance(queries.data(), 2, rows.data(), 32,
                                          out.data());
      EXPECT_EQ(1u, metric->matrix_calls);
      EXPECT_EQ(32u, metric->matrix_rows);
      EXPECT_EQ(2u, metric->matrix_queries);
      EXPECT_EQ(0u, metric->pair_calls);
      EXPECT_TRUE(metric->batch_counts.empty());
      EXPECT_FLOAT_EQ(-123.0f, out[63]);
      EXPECT_FLOAT_EQ(-999.0f, out[64]);
    }
  }
}

TEST(IVFDistanceCalculator, NativeFp16CosineIgnoresNormTailAndMatchesFp64) {
  constexpr size_t kCount = 65;
  for (size_t dim : {3u, 256u, 257u}) {
    SCOPED_TRACE(dim);
    const size_t record_slots = dim + 2;
    IndexMeta meta(IndexMeta::DT_FP16, record_slots);
    meta.set_metric("Cosine", 0, Params());
    meta.set_major_order(IndexMeta::MO_ROW);
    auto metric = IndexFactory::CreateMetric("Cosine");
    ASSERT_NE(nullptr, metric);
    ASSERT_EQ(0, metric->init(meta, Params()));
    IVFDistanceCalculator calculator(meta, metric, 32);
    // The leading half slot and odd payload dimension exercise records that
    // need only half alignment, not float/vector alignment.
    std::vector<uint16_t> records(1 + (kCount + 1) * record_slots, 0);
    for (size_t row = 1; row <= kCount; ++row) {
      double norm2 = 0.0;
      for (size_t d = 0; d < dim; ++d) {
        const int value = static_cast<int>((row * 7 + d * 3) % 13) - 6;
        norm2 += value * value;
      }
      for (size_t d = 0; d < dim; ++d) {
        const int value = static_cast<int>((row * 7 + d * 3) % 13) - 6;
        records[1 + row * record_slots + d] =
            FloatHelper::ToFP16(static_cast<float>(value / std::sqrt(norm2)));
      }
    }
    const uint32_t poison_norm = 0x7fff7fffu;
    for (size_t row = 0; row <= kCount; ++row)
      std::memcpy(records.data() + 1 + row * record_slots + dim, &poison_norm,
                  sizeof(poison_norm));
    const auto *rows = records.data() + 1;
    const auto *query = rows + kCount * record_slots;
    const auto pair = metric->distance();
    ASSERT_TRUE(pair);
    for (size_t count : {1u, 32u, 33u, 65u}) {
      SCOPED_TRACE(count);
      std::vector<float> out(count + 1, -999.0f);
      calculator.query_features_distance(query, rows, count, out.data());
      EXPECT_FLOAT_EQ(-999.0f, out.back());
      for (size_t row = 0; row < count; ++row) {
        const auto *values = rows + row * record_slots;
        double expected = 1.0;
        for (size_t d = 0; d < dim; ++d)
          expected -= static_cast<double>(FloatHelper::ToFP32(values[d])) *
                      FloatHelper::ToFP32(query[d]);
        float reference;
        pair(query, values, record_slots, &reference);
        EXPECT_NEAR(expected, out[row], 2.0e-6);
        EXPECT_NEAR(reference, out[row], 2.0e-6);
      }
      EXPECT_FLOAT_EQ(1.0f, out[0]);  // Zero payload, despite the hostile norm.
    }
  }
}
