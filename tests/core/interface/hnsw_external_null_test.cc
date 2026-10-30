// Copyright 2026-present the zvec project
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
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/core/interface/index.h>
#include <zvec/core/interface/index_factory.h>
#include <zvec/core/interface/index_param_builders.h>
#include <zvec/core/interface/vector_source.h>
#include "algorithm/hnsw/hnsw_context.h"
#include "tests/test_util.h"

using namespace zvec::core_interface;

namespace {

class NullableVectorSource : public zvec::core::VectorSource {
 public:
  static constexpr uint32_t kDimension = 16;

  explicit NullableVectorSource(uint32_t count)
      : vectors(count), missing(count, false) {
    for (uint32_t id = 0; id < count; ++id) {
      vectors[id][0] = static_cast<float>(id);
    }
  }

  const void *get_vector(uint32_t id) const override {
    return missing.at(id) ? nullptr : vectors.at(id).data();
  }

  std::vector<std::array<float, kDimension>> vectors;
  std::vector<bool> missing;
};

class HnswExternalNullTest
    : public testing::TestWithParam<std::tuple<uint32_t, bool>> {
 protected:
  void SetUp() override {
    source_ = std::make_unique<NullableVectorSource>(std::get<0>(GetParam()));
    const auto *info = testing::UnitTest::GetInstance()->current_test_info();
    directory_ = std::string("hnsw_external_null/") + info->name() + "/";
    zvec::test_util::RemoveTestPath(directory_);
    auto param = HNSWIndexParamBuilder()
                     .with_metric_type(MetricType::kL2sq)
                     .with_data_type(DataType::DT_FP32)
                     .with_dimension(NullableVectorSource::kDimension)
                     .with_is_sparse(false)
                     .with_use_external_vector(true)
                     .build();
    index_ = std::dynamic_pointer_cast<HNSWIndex>(
        IndexFactory::CreateAndInitIndex(*param));
    ASSERT_NE(nullptr, index_);
    ASSERT_EQ(0, index_->open(directory_ + "index",
                              {StorageOptions::StorageType::kMMAP, true}));
    ASSERT_EQ("external", index_->storage_mode());
    for (uint32_t id = 0; id < source_->vectors.size(); ++id) {
      VectorData vector{DenseVector{source_->vectors[id].data()}};
      ASSERT_EQ(0, index_->add_with_source(vector, id, *source_));
    }
    param_ = HNSWQueryParamBuilder()
                 .with_topk(2)
                 .with_ef_search(50)
                 .with_fetch_vector(std::get<1>(GetParam()))
                 .build();
  }

  void TearDown() override {
    if (index_) index_->close();
    zvec::test_util::RemoveTestPath(directory_);
  }

  int search() {
    // The query is independent of the unavailable source records.
    std::array<float, NullableVectorSource::kDimension> query{};
    VectorData vector{DenseVector{query.data()}};
    result_ = SearchResult{};
    return index_->search_with_source(vector, param_, *source_, &result_);
  }

  void expect_results(std::initializer_list<uint32_t> ids) {
    ASSERT_EQ(ids.size(), result_.doc_list_.size());
    size_t offset = 0;
    for (uint32_t id : ids) {
      const auto &doc = result_.doc_list_[offset++];
      EXPECT_EQ(id, doc.key());
      EXPECT_FLOAT_EQ(static_cast<float>(id * id), doc.score());
      if (std::get<1>(GetParam())) {
        EXPECT_EQ(source_->vectors[id].data(), doc.vector());
      }
    }
  }

  std::unique_ptr<NullableVectorSource> source_;
  std::shared_ptr<HNSWIndex> index_;
  HNSWQueryParam::Pointer param_;
  SearchResult result_;
  std::string directory_;
};

TEST_P(HnswExternalNullTest, SkipsMissingNearestAndRecovers) {
  ASSERT_EQ(0, search());
  expect_results({0, 1});
  source_->missing[0] = true;
  testing::internal::CaptureStderr();
  const int ret = search();
  const std::string diagnostic = testing::internal::GetCapturedStderr();
  EXPECT_EQ(0, ret);
  EXPECT_NE(std::string::npos, diagnostic.find("nullptr"));
  expect_results({1, 2});
  source_->missing[0] = false;
  ASSERT_EQ(0, search());
  expect_results({0, 1});
}

TEST_P(HnswExternalNullTest, HandlesMissingEntryPointAndEmptySource) {
  auto context = index_->index_searcher()->create_context();
  auto *hnsw_context = dynamic_cast<zvec::core::HnswContext *>(context.get());
  ASSERT_NE(nullptr, hnsw_context);
  const uint32_t entry = hnsw_context->get_entity().entry_point();
  ASSERT_LT(entry, source_->missing.size());
  source_->missing[entry] = true;
  ASSERT_EQ(0, search());
  if (entry == 0) {
    expect_results({1, 2});
  } else if (entry == 1) {
    expect_results({0, 2});
  } else {
    expect_results({0, 1});
  }
  std::fill(source_->missing.begin(), source_->missing.end(), true);
  ASSERT_EQ(0, search());
  expect_results({});
  source_->missing[2] = false;
  ASSERT_EQ(0, search());
  expect_results({2});
}

TEST_P(HnswExternalNullTest, HonorsLinearAndPrimaryKeySearch) {
  source_->missing[0] = true;
  param_->is_linear = true;
  ASSERT_EQ(0, search());
  expect_results({1, 2});
  param_->is_linear = false;
  param_->bf_pks = std::make_shared<std::vector<uint64_t>>(
      std::initializer_list<uint64_t>{0, 2, 1});
  ASSERT_EQ(0, search());
  expect_results({1, 2});
}

TEST_P(HnswExternalNullTest, SupportsBatchedQueriesAndCandidateKeys) {
  auto runner = index_->index_searcher();
  auto context = runner->create_context();
  auto *ctx = dynamic_cast<zvec::core::HnswContext *>(context.get());
  ASSERT_NE(nullptr, ctx);
  ctx->set_vector_source(source_.get());
  ctx->set_topk(2);
  ctx->set_ef(50);
  ctx->set_fetch_vector(std::get<1>(GetParam()));
  source_->missing[0] = true;
  std::array<std::array<float, NullableVectorSource::kDimension>, 2> queries{};
  // Avoid tying vectors 1 and 3 for the second query in the larger index.
  queries[1][0] = 1.75f;
  zvec::core::IndexQueryMeta meta(zvec::core::IndexMeta::DataType::DT_FP32,
                                  NullableVectorSource::kDimension);
  ASSERT_EQ(0, runner->search_impl(queries.data(), meta, 2, context));
  ASSERT_EQ(2U, context->result(0).size());
  EXPECT_EQ(1U, context->result(0)[0].key());
  EXPECT_EQ(2U, context->result(0)[1].key());
  ASSERT_EQ(2U, context->result(1).size());
  EXPECT_EQ(2U, context->result(1)[0].key());
  EXPECT_FLOAT_EQ(0.0625f, context->result(1)[0].score());
  EXPECT_EQ(1U, context->result(1)[1].key());
  EXPECT_FLOAT_EQ(0.5625f, context->result(1)[1].score());

  std::vector<uint64_t> keys;
  std::vector<float> scores;
  ASSERT_EQ(0, runner->search_candidates_impl(queries.data(), meta, keys,
                                              context, &scores));
  EXPECT_EQ((std::vector<uint64_t>{1, 2}), keys);
  EXPECT_EQ((std::vector<float>{1.0f, 4.0f}), scores);
}

TEST_P(HnswExternalNullTest, PreservesFilterAndGroupBy) {
  source_->missing[0] = true;
  param_->filter = std::make_shared<zvec::core::IndexFilter>();
  param_->filter->set([](uint64_t id) { return id == 1; });
  ASSERT_EQ(0, search());
  ASSERT_FALSE(result_.doc_list_.empty());
  EXPECT_EQ(2U, result_.doc_list_.front().key());
  for (const auto &doc : result_.doc_list_) {
    EXPECT_NE(0U, doc.key());
    EXPECT_NE(1U, doc.key());
  }

  param_->filter.reset();
  param_->group_by_param = std::make_shared<GroupByParam>();
  param_->group_by_param->group_count = 1;
  param_->group_by_param->group_topk = 2;
  param_->group_by_param->group_by = [](uint64_t) { return "group"; };
  ASSERT_EQ(0, search());
  ASSERT_EQ(1U, result_.group_doc_list_.size());
  const auto &docs = result_.group_doc_list_.front().docs();
  ASSERT_EQ(2U, docs.size());
  EXPECT_EQ(1U, docs[0].key());
  EXPECT_FLOAT_EQ(1.0f, docs[0].score());
  EXPECT_EQ(2U, docs[1].key());
  EXPECT_FLOAT_EQ(4.0f, docs[1].score());
}

INSTANTIATE_TEST_SUITE_P(BruteForceAndGraph, HnswExternalNullTest,
                         testing::Combine(testing::Values(3U, 1001U),
                                          testing::Bool()));

}  // namespace
