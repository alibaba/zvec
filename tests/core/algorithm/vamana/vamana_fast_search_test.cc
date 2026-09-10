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

#include <array>
#include <optional>
#include <unordered_map>
#include <vector>
#include <gtest/gtest.h>
#include "tests/test_util.h"
#include "vamana_algorithm.h"

namespace zvec::core {
namespace {

// Exercise the actual contiguous algorithm with an exact hand-written graph.
// Only distance calls are observed: no test branches in the production loop.
class VamanaFastSearchTest : public testing::Test {
 protected:
  void SetUp() override {
    test_util::RemoveTestPath(path_);
  }

  void TearDown() override {
    context_.reset();
    algorithm_.reset();
    if (entity_) entity_->close();
    entity_.reset();
    if (storage_) storage_->close();
    storage_.reset();
    test_util::RemoveTestPath(path_);
  }

  void CreateGraph(const std::vector<float> &values,
                   const std::vector<std::vector<node_id_t>> &rows) {
    ASSERT_EQ(values.size(), rows.size());
    storage_ = IndexFactory::CreateStorage("MMapFileStorage");
    ASSERT_TRUE(storage_);
    ASSERT_EQ(0, storage_->init(ailego::Params()));
    ASSERT_EQ(0, storage_->open(path_, true));
    entity_ = std::make_shared<VamanaContiguousStreamerEntity>(stats_);
    entity_->set_vector_size(sizeof(float) * kDimension);
    entity_->set_max_degree(8);
    entity_->set_search_list_size(128);
    entity_->set_max_occlusion_size(128);
    ASSERT_EQ(0, entity_->init(values.size()));
    ASSERT_EQ(0, entity_->open(storage_, 0, false));
    for (node_id_t i = 0; i < values.size(); ++i) {
      std::array<float, kDimension> vector{};
      vector[0] = values[i];
      node_id_t id = kInvalidNodeId;
      ASSERT_EQ(0, entity_->add_vector(i, vector.data(), &id));
      ASSERT_EQ(i, id);
    }
    for (node_id_t i = 0; i < rows.size(); ++i) {
      std::vector<std::pair<node_id_t, dist_t>> neighbors;
      for (node_id_t id : rows[i]) neighbors.emplace_back(id, 0.0f);
      ASSERT_EQ(0, entity_->update_neighbors(i, neighbors));
    }
    entity_->update_entry_point(0);
    ASSERT_EQ(0, entity_->build_contiguous_memory());
    ASSERT_TRUE(entity_->is_contiguous());
    for (node_id_t i = 0; i < values.size(); ++i) {
      ids_[entity_->get_vector_ptr(i)] = i;
      // Entry-point distance uses the canonical (mmap) accessor.
      ids_[entity_->get_vector(i)] = i;
    }
    metric_ = IndexFactory::CreateMetric("SquaredEuclidean");
    ASSERT_TRUE(metric_);
    ASSERT_EQ(0,
              metric_->init(IndexMeta(IndexMeta::DataType::DT_FP32, kDimension),
                            ailego::Params()));
    algorithm_ =
        std::make_unique<VamanaAlgorithm<VamanaContiguousStreamerEntity>>(
            *entity_);
    MakeContext();
  }

  void MakeContext(std::optional<VisitFilter::Mode> mode = std::nullopt) {
    context_ = std::make_unique<VamanaContext>(kDimension, metric_, entity_);
    if (mode) context_->set_filter_mode(*mode);
    context_->set_max_scan_num(10000);
    ASSERT_EQ(0, context_->init(VamanaContext::kSearcherContext));
    const auto batch = metric_->batch_distance();
    context_->update_dist_calculator_distance(
        metric_->distance(),
        [this, batch](const void **vectors, const void *query, size_t count,
                      size_t dim, float *out, const void **extra) {
          std::vector<node_id_t> row;
          for (size_t i = 0; i < count; ++i) row.push_back(ids_.at(vectors[i]));
          evaluated_.push_back(std::move(row));
          batch(vectors, query, count, dim, out, extra);
        });
  }

  void Search(float value, uint32_t capacity, uint32_t topk = 0) {
    ASSERT_TRUE(context_);
    evaluated_.clear();
    context_->clear();
    context_->set_ef(capacity);
    context_->set_topk(topk ? topk : capacity);
    std::array<float, kDimension> query{};
    query[0] = value;
    context_->reset_query(query.data());
    ASSERT_EQ(0, algorithm_->search(context_.get()));
    ASSERT_FALSE(context_->error());
    context_->topk_to_result();
  }

  std::vector<std::pair<uint64_t, float>> Results() const {
    std::vector<std::pair<uint64_t, float>> result;
    for (const auto &doc : context_->result()) {
      result.emplace_back(doc.key(), doc.score());
    }
    return result;
  }

  static constexpr size_t kDimension = 16;
  const std::string path_{"vamana_fast_search_test.index"};
  IndexStreamer::Stats stats_;
  IndexStorage::Pointer storage_;
  std::shared_ptr<VamanaContiguousStreamerEntity> entity_;
  IndexMetric::Pointer metric_;
  std::unique_ptr<VamanaAlgorithm<VamanaContiguousStreamerEntity>> algorithm_;
  VamanaContext::Pointer context_;
  std::unordered_map<const void *, node_id_t> ids_;
  std::vector<std::vector<node_id_t>> evaluated_;
};

TEST_F(VamanaFastSearchTest, ContextDefaultsToBitmapAndAllowsExplicitOverride) {
  CreateGraph({10, 2}, {{1}, {0}});
  EXPECT_EQ(VisitFilter::BitMap, context_->visit_filter().get_mode());
  for (auto type :
       {VamanaContext::kBuilderContext, VamanaContext::kSearcherContext,
        VamanaContext::kStreamerContext}) {
    SCOPED_TRACE(type);
    VamanaContext context(kDimension, metric_, entity_);
    ASSERT_EQ(0, context.init(type));
    auto &visit = context.visit_filter();
    EXPECT_EQ(VisitFilter::BitMap, visit.get_mode());
    EXPECT_FALSE(visit.visited(1));
    visit.set_visited(1);
    EXPECT_TRUE(visit.visited(1));
    visit.clear();
    EXPECT_FALSE(visit.visited(1));

    VamanaContext explicit_context(kDimension, metric_, entity_);
    explicit_context.set_filter_mode(VisitFilter::ByteMap);
    ASSERT_EQ(0, explicit_context.init(type));
    EXPECT_EQ(VisitFilter::ByteMap, explicit_context.visit_filter().get_mode());
  }
}

TEST_F(VamanaFastSearchTest, LocalOptimumReusesRowThenPoolFindsBetterPoint) {
  CreateGraph({10, 2, 3, 1}, {{1, 2}, {0, 2}, {3}, {}});
  Search(0, 4);
  const std::vector<std::vector<node_id_t>> expected = {
      {0}, {1, 2}, {0, 2}, {3}};
  EXPECT_EQ(expected, evaluated_);
  EXPECT_EQ(6U, context_->get_scan_num());
  const std::vector<std::pair<uint64_t, float>> results = {
      {3, 1}, {1, 4}, {2, 9}, {0, 100}};
  EXPECT_EQ(results, Results());
}

TEST_F(VamanaFastSearchTest, CandidateOutputPreservesCutoffAndSearchTrace) {
  CreateGraph({10, 2, 3, 1}, {{1, 2}, {0, 2}, {3}, {}});
  for (float query : {0.0f, 2.5f, 0.0f}) {
    SCOPED_TRACE(query);
    // 2.5 gives equal distances, including at a candidate cutoff.
    for (uint32_t topk : {1U, 2U, 4U}) {
      SCOPED_TRACE(topk);
      Search(query, 4, topk);
      const auto expected = Results();
      const auto trace = evaluated_;
      std::vector<uint64_t> keys;
      context_->set_candidate_output(&keys);
      Search(query, 4, topk);
      EXPECT_TRUE(context_->result().empty());
      ASSERT_EQ(expected.size(), keys.size());
      for (size_t i = 0; i < keys.size(); ++i) {
        EXPECT_EQ(expected[i].first, keys[i]);
      }
      EXPECT_EQ(trace, evaluated_);
      // Unique distances export directly; ties use the original heap order.
      EXPECT_EQ(query == 0.0f, context_->topk_heap().empty());
      context_->set_candidate_output(nullptr);
      Search(query, 4, topk);
      EXPECT_EQ(expected, Results());
    }
  }
}

TEST_F(VamanaFastSearchTest, HundredStepCapRecomputesLandingRow) {
  constexpr node_id_t kCount = 104;
  std::vector<float> values(kCount);
  std::vector<std::vector<node_id_t>> rows(kCount);
  for (node_id_t i = 0; i < kCount; ++i) {
    values[i] = static_cast<float>(kCount - i);
    if (i > 0) rows[i].push_back(i - 1);
    if (i + 1 < kCount) rows[i].push_back(i + 1);
  }
  CreateGraph(values, rows);
  Search(0, 1);
  ASSERT_EQ(104U, evaluated_.size());
  EXPECT_EQ((std::vector<node_id_t>{98, 100}), evaluated_[100]);
  // Depth 100 stops at node 100, while the cached row belongs to node 99.
  EXPECT_EQ((std::vector<node_id_t>{99, 101}), evaluated_[101]);
  // The pool phase skips the already visited backwards edges.
  EXPECT_EQ((std::vector<node_id_t>{102}), evaluated_[102]);
  EXPECT_EQ((std::vector<node_id_t>{103}), evaluated_[103]);
  EXPECT_EQ(204U, context_->get_scan_num());
  EXPECT_EQ((std::vector<std::pair<uint64_t, float>>{{103, 1}}), Results());
}

TEST_F(VamanaFastSearchTest, EqualDistanceStopsDescentWithoutLosingNeighbors) {
  CreateGraph({2, 2, 1}, {{1}, {2}, {}});
  Search(0, 3);
  EXPECT_EQ(3U, context_->get_scan_num());
  ASSERT_EQ(3U, Results().size());
  EXPECT_EQ(2U, Results()[0].first);
  EXPECT_EQ(1.0f, Results()[0].second);
}

TEST_F(VamanaFastSearchTest, DuplicateNeighborsAndReusedContextMatchFresh) {
  CreateGraph({10, 2, 3, 1}, {{1, 1, 2}, {0, 2, 2, 1}, {3, 3}, {1, 1}});
  Search(0, 4);
  const auto expected = Results();
  ASSERT_EQ(4U, expected.size());
  EXPECT_EQ(9U, context_->get_scan_num());
  const auto first_trace = evaluated_;
  for (uint32_t capacity : {1U, 4U, 2U, 4U}) {
    Search(12, capacity);
    const auto reused_result = Results();
    const auto reused_trace = evaluated_;
    auto reused_context = std::move(context_);
    MakeContext();
    Search(12, capacity);
    EXPECT_EQ(reused_result, Results());
    EXPECT_EQ(reused_trace, evaluated_);
    context_ = std::move(reused_context);
    Search(0, 4);
    EXPECT_EQ(expected, Results());
    EXPECT_EQ(first_trace, evaluated_);
  }
}

TEST_F(VamanaFastSearchTest, EmptyNeighborRowKeepsTheEntryPoint) {
  CreateGraph({0}, {{}});
  Search(0, 1);
  EXPECT_EQ((std::vector<std::vector<node_id_t>>{{0}}), evaluated_);
  EXPECT_EQ((std::vector<std::pair<uint64_t, float>>{{0, 0}}), Results());
}

TEST_F(VamanaFastSearchTest, ExactVisitModesPreserveResultsAndSearchTrace) {
  CreateGraph({10, 2, 3, 1}, {{1, 1, 2}, {0, 2, 2, 1}, {3, 3}, {1, 1}});
  for (float query : {0.0f, 12.0f}) {
    for (uint32_t capacity : {1U, 4U}) {
      MakeContext(VisitFilter::ByteMap);
      Search(query, capacity);
      const auto expected = Results();
      const auto expected_trace = evaluated_;
      MakeContext(VisitFilter::BitMap);
      Search(query, capacity);
      EXPECT_EQ(expected, Results());
      EXPECT_EQ(expected_trace, evaluated_);
    }
  }
}

TEST_F(VamanaFastSearchTest, InvalidVisitFilterFailsSearchExplicitly) {
  CreateGraph({10, 2, 3, 1}, {{1, 2}, {0, 2}, {3}, {}});
  MakeContext(VisitFilter::Default);
  context_->set_topk(4);
  context_->set_ef(4);
  std::array<float, kDimension> query{};
  context_->reset_query(query.data());
  EXPECT_EQ(IndexError_Runtime, algorithm_->search(context_.get()));
  EXPECT_EQ(0U, context_->topk_heap().size());
  EXPECT_TRUE(evaluated_.empty());
}

}  // namespace
}  // namespace zvec::core
