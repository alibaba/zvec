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

CollectionSchema::Ptr MakeSchema(
    const std::vector<std::pair<std::string, MetricType>> &fields) {
  auto schema = std::make_shared<CollectionSchema>("test");
  for (const auto &[name, metric] : fields) {
    auto field = std::make_shared<FieldSchema>(
        name, DataType::VECTOR_FP16, /*dimension=*/4, /*nullable=*/false,
        std::make_shared<HnswIndexParams>(metric));
    schema->add_field(field);
  }
  return schema;
}

}  // namespace

// ==================== RrfReranker Tests ====================

TEST(RrfRerankerTest, BasicRRF) {
  RrfReranker reranker(/*rank_constant=*/60);

  // Two vector fields, each returning 3 documents with some overlap
  std::vector<DocPtrList> query_results;
  query_results.push_back(
      {MakeDoc("a", 0.9f), MakeDoc("b", 0.8f), MakeDoc("c", 0.7f)});
  query_results.push_back(
      {MakeDoc("b", 0.95f), MakeDoc("a", 0.85f), MakeDoc("d", 0.75f)});

  auto result = reranker.rerank(query_results, /*topn=*/10);
  ASSERT_TRUE(result.has_value());
  auto &results = result.value();

  // "a" appears at rank 0 in vec1 and rank 1 in vec2:
  //   rrf_score = 1/(60+0+1) + 1/(60+1+1) = 1/61 + 1/62
  // "b" appears at rank 1 in vec1 and rank 0 in vec2:
  //   rrf_score = 1/(60+1+1) + 1/(60+0+1) = 1/62 + 1/61
  // So a and b should have equal scores and be at the top
  ASSERT_GE(results.size(), 3u);

  // "a" and "b" should have the highest RRF scores (equal, order unspecified)
  std::set<std::string> top2{results[0]->pk(), results[1]->pk()};
  EXPECT_EQ(top2, (std::set<std::string>{"a", "b"}));
  // Verify scores are close (a and b have same RRF score)
  EXPECT_NEAR(results[0]->score(), results[1]->score(), 1e-10);
}

TEST(RrfRerankerTest, Topn) {
  RrfReranker reranker(/*rank_constant=*/60);

  std::vector<DocPtrList> query_results;
  query_results.push_back(
      {MakeDoc("a", 0.9f), MakeDoc("b", 0.8f), MakeDoc("c", 0.7f)});

  auto result = reranker.rerank(query_results, /*topn=*/2);
  ASSERT_TRUE(result.has_value());
  ASSERT_EQ(result.value().size(), 2u);
}

TEST(RrfRerankerTest, SingleField) {
  RrfReranker reranker(/*rank_constant=*/60);

  std::vector<DocPtrList> query_results;
  query_results.push_back({MakeDoc("a", 0.9f), MakeDoc("b", 0.8f)});

  auto result = reranker.rerank(query_results);
  ASSERT_TRUE(result.has_value());
  auto &results = result.value();
  ASSERT_EQ(results.size(), 2u);
  // With single field, RRF score for rank 0 > rank 1
  EXPECT_GT(results[0]->score(), results[1]->score());
}

TEST(RrfRerankerTest, EmptyResults) {
  RrfReranker reranker(/*rank_constant=*/60);

  std::vector<DocPtrList> query_results;
  auto result = reranker.rerank(query_results);
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(result.value().empty());
}

// ==================== WeightedReranker Tests ====================

TEST(WeightedRerankerTest, BasicWeighted) {
  auto schema =
      MakeSchema({{"vec1", MetricType::L2}, {"vec2", MetricType::L2}});
  WeightedReranker reranker({0.7, 0.3});
  reranker.bind_schema(schema, {"vec1", "vec2"});

  std::vector<DocPtrList> query_results;
  query_results.push_back({MakeDoc("a", 0.5f), MakeDoc("b", 0.3f)});
  query_results.push_back({MakeDoc("a", 0.8f), MakeDoc("c", 0.6f)});

  auto result = reranker.rerank(query_results);
  ASSERT_TRUE(result.has_value());
  auto &results = result.value();
  ASSERT_GE(results.size(), 2u);
  // "a" appears in both fields, should have highest combined score
  EXPECT_EQ(results[0]->pk(), "a");
}

TEST(WeightedRerankerTest, MixedMetrics) {
  auto schema =
      MakeSchema({{"vec1", MetricType::L2}, {"vec2", MetricType::COSINE}});
  WeightedReranker reranker({0.5, 0.5});
  reranker.bind_schema(schema, {"vec1", "vec2"});

  std::vector<DocPtrList> query_results;
  query_results.push_back({MakeDoc("a", 0.5f)});
  query_results.push_back({MakeDoc("a", 0.4f)});

  auto result = reranker.rerank(query_results);
  ASSERT_TRUE(result.has_value());
  auto &results = result.value();
  ASSERT_EQ(results.size(), 1u);
  EXPECT_EQ(results[0]->pk(), "a");
  // L2 normalize(0.5) = 1 - 2*atan(0.5)/pi ≈ 0.7048
  // COSINE normalize(0.4) = 1 - 0.4/2 = 0.8
  // weighted = 0.7048 * 0.5 + 0.8 * 0.5 ≈ 0.7524
  double l2_norm = 1.0 - 2.0 * std::atan(0.5) / M_PI;
  double cos_norm = 1.0 - 0.4 / 2.0;
  double expected = l2_norm * 0.5 + cos_norm * 0.5;
  EXPECT_NEAR(results[0]->score(), expected, 1e-5);
}

TEST(WeightedRerankerTest, MissingMetricError) {
  auto schema = MakeSchema({{"vec1", MetricType::L2}});
  WeightedReranker reranker;
  // Binding a field that is absent from the schema should fail at rerank time.
  reranker.bind_schema(schema, {"vec1", "vec2"});

  std::vector<DocPtrList> query_results;
  query_results.push_back({MakeDoc("a", 0.5f)});
  query_results.push_back({MakeDoc("b", 0.3f)});
  auto result = reranker.rerank(query_results);
  ASSERT_FALSE(result.has_value());
}

TEST(WeightedRerankerTest, NormalizeL2) {
  auto schema = MakeSchema({{"vec1", MetricType::L2}});
  WeightedReranker reranker;
  reranker.bind_schema(schema, {"vec1"});

  std::vector<DocPtrList> query_results;
  query_results.push_back({MakeDoc("a", 0.0f), MakeDoc("b", 1.0f)});

  auto result = reranker.rerank(query_results);
  ASSERT_TRUE(result.has_value());
  auto &results = result.value();
  ASSERT_EQ(results.size(), 2u);
  // L2 normalize(0.0) = 1.0, normalize(1.0) ∈ (0, 1)
  EXPECT_NEAR(results[0]->score(), 1.0, 1e-10);
  EXPECT_EQ(results[0]->pk(), "a");
  EXPECT_GT(results[1]->score(), 0.0);
  EXPECT_LT(results[1]->score(), 1.0);
}

TEST(WeightedRerankerTest, NormalizeIP) {
  auto schema = MakeSchema({{"vec1", MetricType::IP}});
  WeightedReranker reranker;
  reranker.bind_schema(schema, {"vec1"});

  std::vector<DocPtrList> query_results;
  query_results.push_back({MakeDoc("a", 0.0f), MakeDoc("b", 1.0f)});

  auto result = reranker.rerank(query_results);
  ASSERT_TRUE(result.has_value());
  auto &results = result.value();
  ASSERT_EQ(results.size(), 2u);
  // IP normalize(1.0) > 0.5 > normalize(0.0) = 0.5... but b scores higher
  EXPECT_EQ(results[0]->pk(), "b");
  EXPECT_GT(results[0]->score(), 0.5);
  EXPECT_NEAR(results[1]->score(), 0.5, 1e-10);
}

TEST(WeightedRerankerTest, NormalizeCosine) {
  auto schema = MakeSchema({{"vec1", MetricType::COSINE}});
  WeightedReranker reranker;
  reranker.bind_schema(schema, {"vec1"});

  std::vector<DocPtrList> query_results;
  query_results.push_back(
      {MakeDoc("a", 0.0f), MakeDoc("b", 1.0f), MakeDoc("c", 2.0f)});

  auto result = reranker.rerank(query_results);
  ASSERT_TRUE(result.has_value());
  auto &results = result.value();
  ASSERT_EQ(results.size(), 3u);
  // COSINE normalize(0.0) = 1.0, normalize(1.0) = 0.5, normalize(2.0) = 0.0
  EXPECT_NEAR(results[0]->score(), 1.0, 1e-10);
  EXPECT_NEAR(results[1]->score(), 0.5, 1e-10);
  EXPECT_NEAR(results[2]->score(), 0.0, 1e-10);
}

TEST(WeightedRerankerTest, Topn) {
  auto schema = MakeSchema({{"vec1", MetricType::L2}});
  WeightedReranker reranker;
  reranker.bind_schema(schema, {"vec1"});

  std::vector<DocPtrList> query_results;
  query_results.push_back(
      {MakeDoc("a", 0.1f), MakeDoc("b", 0.2f), MakeDoc("c", 0.3f)});

  auto result = reranker.rerank(query_results, /*topn=*/2);
  ASSERT_TRUE(result.has_value());
  ASSERT_EQ(result.value().size(), 2u);
}


// ==================== CallbackReranker Tests ====================

TEST(CallbackRerankerTest, BasicCallback) {
  // Simple callback that returns docs sorted by score descending, limited to
  // topn
  CallbackReranker::Callback cb =
      [](const std::vector<DocPtrList> &query_results, int topn) -> DocPtrList {
    DocPtrList all_docs;
    for (const auto &docs : query_results) {
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

  CallbackReranker reranker(cb);

  std::vector<DocPtrList> query_results;
  query_results.push_back({MakeDoc("a", 0.5f), MakeDoc("b", 0.9f)});
  query_results.push_back({MakeDoc("c", 0.7f)});

  auto result = reranker.rerank(query_results, /*topn=*/10);
  ASSERT_TRUE(result.has_value());
  auto &results = result.value();
  ASSERT_EQ(results.size(), 3u);
  // Should be sorted by score descending
  EXPECT_EQ(results[0]->pk(), "b");
  EXPECT_EQ(results[1]->pk(), "c");
  EXPECT_EQ(results[2]->pk(), "a");
}
