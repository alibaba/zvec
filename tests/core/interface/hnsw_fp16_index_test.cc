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
#include <random>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/utility/float_helper.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/interface/index.h>
#include <zvec/core/interface/index_factory.h>
#include <zvec/core/interface/index_param_builders.h>
#include "algorithm/hnsw/hnsw_context.h"
#include "algorithm/hnsw/hnsw_streamer.h"
#include "tests/test_util.h"

namespace {

namespace core = zvec::core;
using namespace zvec::core_interface;
using Half = zvec::ailego::Float16;
using Rows = std::vector<std::pair<uint32_t, float>>;

constexpr uint32_t kDimension = 256;
constexpr uint32_t kCount = core::HnswEntity::kDefaultBruteForceThreshold + 1;
constexpr uint32_t kTopK = 200;
constexpr double kScoreTolerance = 2.0e-6;

class HnswFp16IndexTest : public testing::TestWithParam<MetricType> {};

TEST_P(HnswFp16IndexTest, NativeHalfStorageGraphFilterAndReopen) {
  const auto metric_type = GetParam();
  const bool cosine = metric_type == MetricType::kCosine;
  const bool inner_product = metric_type == MetricType::kInnerProduct;
  const std::string path = "hnsw_native_fp16_" +
                           std::to_string(static_cast<int>(metric_type)) +
                           ".index";
  zvec::test_util::RemoveTestFiles(path);

  // Native FP16 input is distinct from FP32 with an FP16 quantizer or a
  // separate FP16 refinement store. Keep all three choices explicit here.
  auto param = HNSWIndexParamBuilder()
                   .with_data_type(DataType::DT_FP16)
                   .with_dimension(kDimension)
                   .with_metric_type(metric_type)
                   .build();
  ASSERT_EQ(50, param->m);
  ASSERT_EQ(500, param->ef_construction);
  ASSERT_TRUE(!param->quantizer_param ||
              param->quantizer_param->type == QuantizerType::kNone);
  auto index = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, index);
  ASSERT_EQ(0, index->open(path, {StorageOptions::StorageType::kMMAP, true}));

  std::mt19937 rng(20261030);
  std::uniform_real_distribution<float> value(-1.0f, 1.0f);
  std::vector<std::vector<Half>> inputs(kCount, std::vector<Half>(kDimension));
  std::vector<std::vector<float>> encoded(kCount,
                                          std::vector<float>(kDimension));
  std::vector<std::string> encoded_bytes(kCount);
  auto reformer = core::IndexFactory::CreateReformer("CosineHalfFloatReformer");
  ASSERT_NE(nullptr, reformer);
  ASSERT_EQ(0, reformer->init(zvec::ailego::Params{}));
  core::IndexQueryMeta input_meta(DataType::DT_FP16, kDimension);
  for (uint32_t id = 0; id < kCount; ++id) {
    const float scale = cosine ? 0.5f + float(id % 7) * 0.25f : 1.0f;
    for (auto &element : inputs[id]) {
      element = value(rng) * scale / 16.0f;
    }
    ASSERT_EQ(0, index->add(VectorData{DenseVector{inputs[id].data()}}, id));
    // Retain the existing normalization/half encoding. The independent
    // distance oracle below accumulates these actual operands in FP64.
    const void *data = inputs[id].data();
    core::IndexQueryMeta transformed_meta;
    if (cosine) {
      ASSERT_EQ(0, reformer->transform(data, input_meta, &encoded_bytes[id],
                                       &transformed_meta));
      ASSERT_EQ(kDimension * sizeof(Half) + sizeof(float),
                encoded_bytes[id].size());
      data = encoded_bytes[id].data();
    } else {
      encoded_bytes[id].assign(static_cast<const char *>(data),
                               kDimension * sizeof(Half));
    }
    zvec::ailego::FloatHelper::ToFP32(static_cast<const uint16_t *>(data),
                                      kDimension, encoded[id].data());
  }
  ASSERT_EQ(0, index->train());

  std::vector<Rows> before_reopen;
  for (int pass = 0; pass < 2; ++pass) {
    const auto &meta = index->index_searcher()->meta();
    EXPECT_EQ(DataType::DT_FP16, meta.data_type());
    EXPECT_EQ(sizeof(Half), meta.unit_size());
    EXPECT_EQ(kDimension * sizeof(Half) + (cosine ? sizeof(float) : 0),
              meta.element_size());
    EXPECT_TRUE(meta.quantizer_name().empty());
    auto streamer =
        std::dynamic_pointer_cast<core::HnswStreamer>(index->index_searcher());
    ASSERT_NE(nullptr, streamer);
    EXPECT_FALSE(streamer->uses_turbo_distance());
    auto context = index->index_searcher()->create_context();
    auto *hnsw_context = dynamic_cast<core::HnswContext *>(context.get());
    ASSERT_NE(nullptr, hnsw_context);
    ASSERT_GT(index->get_doc_count(), hnsw_context->get_bruteforce_threshold());

    auto metric = core::IndexFactory::CreateMetric(meta.metric_name());
    ASSERT_NE(nullptr, metric);
    ASSERT_EQ(0, metric->init(meta, zvec::ailego::Params{}));
    auto one_to_one = metric->distance();
    ASSERT_TRUE(one_to_one);
    size_t search_number = 0;
    for (uint32_t query_id : {37U, 913U}) {
      SCOPED_TRACE(query_id);
      // A filtered request followed by a plain one also checks reused context
      // state; a filter's true result excludes that key.
      for (bool filtered : {false, true, false}) {
        SCOPED_TRACE(filtered);
        std::vector<std::pair<uint32_t, double>> expected;
        std::vector<double> scores(kCount);
        for (uint32_t id = 0; id < kCount; ++id) {
          double distance = cosine ? 1.0 : 0.0;
          for (uint32_t d = 0; d < kDimension; ++d) {
            const double a = encoded[id][d];
            const double b = encoded[query_id][d];
            distance += metric_type == MetricType::kL2sq
                            ? (a - b) * (a - b)
                            : (inner_product ? a * b : -a * b);
          }
          scores[id] = distance;
          if (!filtered || id % 3 != 1) expected.emplace_back(id, distance);
        }
        std::sort(expected.begin(), expected.end(),
                  [inner_product](const auto &a, const auto &b) {
                    return inner_product ? a.second > b.second
                                         : a.second < b.second;
                  });
        for (bool linear : {false, true}) {
          SCOPED_TRACE(linear);
          auto query_param = HNSWQueryParamBuilder()
                                 .with_topk(kTopK)
                                 .with_is_linear(linear)
                                 .build();
          ASSERT_EQ(300U, query_param->ef_search);
          if (filtered) {
            query_param->filter = std::make_shared<IndexFilter>();
            query_param->filter->set([](uint64_t key) { return key % 3 == 1; });
          }
          SearchResult result;
          ASSERT_EQ(
              0, index->search(VectorData{DenseVector{inputs[query_id].data()}},
                               query_param, &result));
          ASSERT_EQ(kTopK, result.doc_list_.size());
          Rows actual;
          std::vector<bool> returned(kCount, false);
          size_t matches = 0;
          for (const auto &doc : result.doc_list_) {
            ASSERT_LT(doc.key(), kCount);
            EXPECT_FALSE(returned[doc.key()]);
            returned[doc.key()] = true;
            if (filtered) EXPECT_NE(1U, doc.key() % 3);
            EXPECT_NEAR(scores[doc.key()], doc.score(), kScoreTolerance);
            float pairwise_score = 0.0f;
            one_to_one(encoded_bytes[doc.key()].data(),
                       encoded_bytes[query_id].data(), meta.dimension(),
                       &pairwise_score);
            if (metric->support_normalize()) metric->normalize(&pairwise_score);
            EXPECT_NEAR(pairwise_score, doc.score(), kScoreTolerance);
            matches += std::any_of(
                expected.begin(), expected.begin() + kTopK,
                [&doc](const auto &row) { return row.first == doc.key(); });
            actual.emplace_back(static_cast<uint32_t>(doc.key()), doc.score());
          }
          if (linear) {
            const double cutoff = expected[kTopK - 1].second;
            for (const auto &row : actual) {
              EXPECT_TRUE(inner_product
                              ? scores[row.first] >= cutoff - kScoreTolerance
                              : scores[row.first] <= cutoff + kScoreTolerance);
            }
            for (const auto &row : expected) {
              const bool strictly_better =
                  inner_product ? row.second > cutoff + kScoreTolerance
                                : row.second < cutoff - kScoreTolerance;
              if (strictly_better) EXPECT_TRUE(returned[row.first]);
            }
          } else {
            EXPECT_GE(matches, kTopK * 9 / 10);
          }
          if (pass == 0) {
            before_reopen.push_back(actual);
          } else {
            EXPECT_EQ(before_reopen[search_number], actual);
          }
          ++search_number;
        }
      }
    }
    VectorDataBuffer fetched;
    ASSERT_EQ(0, index->fetch(37, &fetched));
    const auto &bytes = std::get<DenseVectorBuffer>(fetched.vector_buffer).data;
    ASSERT_EQ(kDimension * sizeof(Half), bytes.size());
    std::vector<float> decoded(kDimension);
    zvec::ailego::FloatHelper::ToFP32(
        reinterpret_cast<const uint16_t *>(bytes.data()), kDimension,
        decoded.data());
    for (uint32_t d = 0; d < kDimension; ++d) {
      EXPECT_NEAR(static_cast<float>(inputs[37][d]), decoded[d],
                  cosine ? 2.0e-4f : 0.0f);
    }
    ASSERT_EQ(0, index->close());
    if (pass == 0) {
      index = IndexFactory::CreateAndInitIndex(*param);
      ASSERT_NE(nullptr, index);
      ASSERT_EQ(0,
                index->open(path, {StorageOptions::StorageType::kMMAP, false}));
    }
  }
  zvec::test_util::RemoveTestFiles(path);
}

INSTANTIATE_TEST_SUITE_P(Metrics, HnswFp16IndexTest,
                         testing::Values(MetricType::kCosine,
                                         MetricType::kInnerProduct,
                                         MetricType::kL2sq));

}  // namespace
