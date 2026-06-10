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

#define _USE_MATH_DEFINES
#include <algorithm>
#include <cmath>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/db/doc.h>
#include <zvec/db/index_params.h>
#include <zvec/db/reranker.h>
#include <zvec/db/type.h>

using namespace zvec;

namespace {

Doc::Ptr MakeDoc(const std::string &id, float score) {
  auto doc = std::make_shared<Doc>();
  doc->set_pk(id);
  doc->set_score(score);
  return doc;
}

FieldSchema::Ptr MakeField(const std::string &name, MetricType metric) {
  return std::make_shared<FieldSchema>(
      name, DataType::VECTOR_FP16, /*dimension=*/4, /*nullable=*/false,
      std::make_shared<HnswIndexParams>(metric));
}

}  // namespace

// ==================== RRF Tests ====================

TEST(RerankRrfTest, BasicRRF) {
  // Two sub-queries, each returning 3 documents with some overlap.
  std::vector<DocPtrList> results;
  results.push_back(
      {MakeDoc("a", 0.9f), MakeDoc("b", 0.8f), MakeDoc("c", 0.7f)});
  results.push_back(
      {MakeDoc("b", 0.95f), MakeDoc("a", 0.85f), MakeDoc("d", 0.75f)});

  auto result =
      reranker::rerank(reranker::RrfParams{/*rank_constant=*/60}, results,
                       /*fields=*/{}, /*topn=*/10);
  ASSERT_TRUE(result.has_value());
  auto &out = result.value();

  // "a" appears at rank 0 in sub-query 0 and rank 1 in sub-query 1:
  //   rrf_score = 1/(60+0+1) + 1/(60+1+1) = 1/61 + 1/62
  // "b" appears at rank 1 in sub-query 0 and rank 0 in sub-query 1:
  //   rrf_score = 1/(60+1+1) + 1/(60+0+1) = 1/62 + 1/61
  // So a and b should have equal scores and occupy the top two slots.
  ASSERT_GE(out.size(), 3u);
  std::set<std::string> top2 = {out[0]->pk(), out[1]->pk()};
  EXPECT_EQ(top2, (std::set<std::string>{"a", "b"}));
  EXPECT_NEAR(out[0]->score(), out[1]->score(), 1e-10);
}

TEST(RerankRrfTest, Topn) {
  std::vector<DocPtrList> results;
  results.push_back(
      {MakeDoc("a", 0.9f), MakeDoc("b", 0.8f), MakeDoc("c", 0.7f)});

  auto result =
      reranker::rerank(reranker::RrfParams{/*rank_constant=*/60}, results,
                       /*fields=*/{}, /*topn=*/2);
  ASSERT_TRUE(result.has_value());
  ASSERT_EQ(result.value().size(), 2u);
}

TEST(RerankRrfTest, SingleField) {
  std::vector<DocPtrList> results;
  results.push_back({MakeDoc("a", 0.9f), MakeDoc("b", 0.8f)});

  auto result =
      reranker::rerank(reranker::RrfParams{/*rank_constant=*/60}, results,
                       /*fields=*/{}, /*topn=*/10);
  ASSERT_TRUE(result.has_value());
  auto &out = result.value();
  ASSERT_EQ(out.size(), 2u);
  // With single sub-query, RRF score for rank 0 > rank 1.
  EXPECT_GT(out[0]->score(), out[1]->score());
}

TEST(RerankRrfTest, EmptyResults) {
  std::vector<DocPtrList> results;
  auto result =
      reranker::rerank(reranker::RrfParams{/*rank_constant=*/60}, results,
                       /*fields=*/{}, /*topn=*/10);
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(result.value().empty());
}

TEST(RerankRrfTest, DefaultParams) {
  // RrfParams (and therefore RerankParams) defaults to rank_constant = 60.
  std::vector<DocPtrList> results;
  results.push_back({MakeDoc("a", 0.9f), MakeDoc("b", 0.8f)});

  auto result =
      reranker::rerank(reranker::RerankParams{}, results, /*fields=*/{},
                       /*topn=*/10);
  ASSERT_TRUE(result.has_value());
  ASSERT_EQ(result.value().size(), 2u);
}

// ==================== Weighted Tests ====================

TEST(RerankWeightedTest, BasicWeighted) {
  std::vector<DocPtrList> results;
  results.push_back({MakeDoc("a", 0.5f), MakeDoc("b", 0.3f)});
  results.push_back({MakeDoc("a", 0.8f), MakeDoc("c", 0.6f)});
  std::vector<FieldSchema::Ptr> fields = {MakeField("vec1", MetricType::L2),
                                          MakeField("vec2", MetricType::L2)};

  auto result =
      reranker::rerank(reranker::WeightedParams{{0.7, 0.3}}, results, fields,
                       /*topn=*/10);
  ASSERT_TRUE(result.has_value());
  auto &out = result.value();
  ASSERT_GE(out.size(), 2u);
  // "a" appears in both sub-queries, should have highest combined score.
  EXPECT_EQ(out[0]->pk(), "a");
}

TEST(RerankWeightedTest, MixedMetrics) {
  std::vector<DocPtrList> results;
  results.push_back({MakeDoc("a", 0.5f)});
  results.push_back({MakeDoc("a", 0.4f)});
  std::vector<FieldSchema::Ptr> fields = {
      MakeField("vec1", MetricType::L2), MakeField("vec2", MetricType::COSINE)};

  auto result =
      reranker::rerank(reranker::WeightedParams{{0.5, 0.5}}, results, fields,
                       /*topn=*/10);
  ASSERT_TRUE(result.has_value());
  auto &out = result.value();
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out[0]->pk(), "a");
  // L2 normalize(0.5) = 1 - 2*atan(0.5)/pi
  // COSINE normalize(0.4) = 1 - 0.4/2 = 0.8
  // weighted = l2_norm * 0.5 + cos_norm * 0.5
  double l2_norm = 1.0 - 2.0 * std::atan(0.5) / M_PI;
  double cos_norm = 1.0 - 0.4 / 2.0;
  double expected = l2_norm * 0.5 + cos_norm * 0.5;
  EXPECT_NEAR(out[0]->score(), expected, 1e-5);
}

TEST(RerankWeightedTest, WeightsCountMismatch) {
  std::vector<DocPtrList> results;
  results.push_back({MakeDoc("a", 0.5f)});
  results.push_back({MakeDoc("b", 0.3f)});
  std::vector<FieldSchema::Ptr> fields = {MakeField("vec1", MetricType::L2),
                                          MakeField("vec2", MetricType::L2)};

  // Only one weight provided for two sub-queries.
  auto result =
      reranker::rerank(reranker::WeightedParams{{1.0}}, results, fields,
                       /*topn=*/10);
  ASSERT_FALSE(result.has_value());
}

TEST(RerankWeightedTest, FieldsCountMismatch) {
  std::vector<DocPtrList> results;
  results.push_back({MakeDoc("a", 0.5f)});
  results.push_back({MakeDoc("b", 0.3f)});
  std::vector<FieldSchema::Ptr> fields = {MakeField("vec1", MetricType::L2)};

  auto result =
      reranker::rerank(reranker::WeightedParams{{0.5, 0.5}}, results, fields,
                       /*topn=*/10);
  ASSERT_FALSE(result.has_value());
}

TEST(RerankWeightedTest, NullFieldError) {
  std::vector<DocPtrList> results;
  results.push_back({MakeDoc("a", 0.5f)});
  std::vector<FieldSchema::Ptr> fields = {nullptr};

  auto result =
      reranker::rerank(reranker::WeightedParams{{1.0}}, results, fields,
                       /*topn=*/10);
  ASSERT_FALSE(result.has_value());
}

TEST(RerankWeightedTest, NormalizeL2) {
  std::vector<DocPtrList> results;
  results.push_back({MakeDoc("a", 0.0f), MakeDoc("b", 1.0f)});
  std::vector<FieldSchema::Ptr> fields = {MakeField("vec1", MetricType::L2)};

  auto result =
      reranker::rerank(reranker::WeightedParams{{1.0}}, results, fields,
                       /*topn=*/10);
  ASSERT_TRUE(result.has_value());
  auto &out = result.value();
  ASSERT_EQ(out.size(), 2u);
  // L2 normalize(0.0) = 1.0, normalize(1.0) in (0, 1)
  EXPECT_NEAR(out[0]->score(), 1.0, 1e-10);
  EXPECT_EQ(out[0]->pk(), "a");
  EXPECT_GT(out[1]->score(), 0.0);
  EXPECT_LT(out[1]->score(), 1.0);
}

TEST(RerankWeightedTest, NormalizeIP) {
  std::vector<DocPtrList> results;
  results.push_back({MakeDoc("a", 0.0f), MakeDoc("b", 1.0f)});
  std::vector<FieldSchema::Ptr> fields = {MakeField("vec1", MetricType::IP)};

  auto result =
      reranker::rerank(reranker::WeightedParams{{1.0}}, results, fields,
                       /*topn=*/10);
  ASSERT_TRUE(result.has_value());
  auto &out = result.value();
  ASSERT_EQ(out.size(), 2u);
  // IP normalize(1.0) > 0.5 > normalize(0.0) = 0.5
  EXPECT_EQ(out[0]->pk(), "b");
  EXPECT_GT(out[0]->score(), 0.5);
  EXPECT_NEAR(out[1]->score(), 0.5, 1e-10);
}

TEST(RerankWeightedTest, NormalizeCosine) {
  std::vector<DocPtrList> results;
  results.push_back(
      {MakeDoc("a", 0.0f), MakeDoc("b", 1.0f), MakeDoc("c", 2.0f)});
  std::vector<FieldSchema::Ptr> fields = {
      MakeField("vec1", MetricType::COSINE)};

  auto result =
      reranker::rerank(reranker::WeightedParams{{1.0}}, results, fields,
                       /*topn=*/10);
  ASSERT_TRUE(result.has_value());
  auto &out = result.value();
  ASSERT_EQ(out.size(), 3u);
  // COSINE normalize(0.0) = 1.0, normalize(1.0) = 0.5, normalize(2.0) = 0.0
  EXPECT_NEAR(out[0]->score(), 1.0, 1e-10);
  EXPECT_NEAR(out[1]->score(), 0.5, 1e-10);
  EXPECT_NEAR(out[2]->score(), 0.0, 1e-10);
}

TEST(RerankWeightedTest, Topn) {
  std::vector<DocPtrList> results;
  results.push_back(
      {MakeDoc("a", 0.1f), MakeDoc("b", 0.2f), MakeDoc("c", 0.3f)});
  std::vector<FieldSchema::Ptr> fields = {MakeField("vec1", MetricType::L2)};

  auto result =
      reranker::rerank(reranker::WeightedParams{{1.0}}, results, fields,
                       /*topn=*/2);
  ASSERT_TRUE(result.has_value());
  ASSERT_EQ(result.value().size(), 2u);
}

// ==================== Callback Tests ====================

TEST(RerankCallbackTest, BasicCallback) {
  // Simple callback that returns docs sorted by score descending, limited to
  // topn.
  reranker::CallbackParams::Callback cb =
      [](const std::vector<DocPtrList> &results,
         const std::vector<FieldSchema::Ptr> & /*fields*/,
         int topn) -> DocPtrList {
    DocPtrList all_docs;
    for (const auto &docs : results) {
      for (const auto &doc : docs) {
        all_docs.push_back(doc);
      }
    }
    std::sort(all_docs.begin(), all_docs.end(),
              [](const Doc::Ptr &a, const Doc::Ptr &b) {
                return a->score() > b->score();
              });
    if (static_cast<int>(all_docs.size()) > topn) {
      all_docs.resize(topn);
    }
    return all_docs;
  };

  std::vector<DocPtrList> results;
  results.push_back({MakeDoc("a", 0.5f), MakeDoc("b", 0.9f)});
  results.push_back({MakeDoc("c", 0.7f)});

  auto result =
      reranker::rerank(reranker::CallbackParams{cb}, results, /*fields=*/{},
                       /*topn=*/10);
  ASSERT_TRUE(result.has_value());
  auto &out = result.value();
  ASSERT_EQ(out.size(), 3u);
  // Should be sorted by score descending.
  EXPECT_EQ(out[0]->pk(), "b");
  EXPECT_EQ(out[1]->pk(), "c");
  EXPECT_EQ(out[2]->pk(), "a");
}

TEST(RerankCallbackTest, EmptyCallbackError) {
  reranker::CallbackParams params;  // callback is empty
  std::vector<DocPtrList> results;
  results.push_back({MakeDoc("a", 0.5f)});

  auto result = reranker::rerank(params, results, /*fields=*/{}, /*topn=*/10);
  ASSERT_FALSE(result.has_value());
}
