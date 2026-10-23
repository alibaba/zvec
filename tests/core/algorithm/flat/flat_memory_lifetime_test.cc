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

#include <tuple>
#include <ailego/utility/memory_helper.h>
#include <gtest/gtest.h>
#include <zvec/ailego/container/vector.h>
#include <zvec/core/framework/index_framework.h>
#include "tests/test_util.h"

using namespace zvec::core;
using namespace zvec::ailego;

namespace {

class FlatMemoryLifetimeTest
    : public testing::TestWithParam<std::tuple<const char *, bool>> {
 protected:
  static constexpr size_t kDimension = 8;
  static constexpr size_t kCount = 67;
  const std::string path_ = "flat_memory_lifetime_test/index";
  IndexStorage::Pointer storage_;
  IndexSearcher::Pointer searcher_;

  static uint64_t key(size_t id) {
    return 1000 + id * 7;
  }

  static float value(size_t id, size_t dim) {
    return static_cast<float>(id * 100 + dim);
  }

  void check_vector(const void *data, size_t id) {
    ASSERT_NE(nullptr, data);
    for (size_t d = 0; d < kDimension; ++d) {
      EXPECT_FLOAT_EQ(value(id, d), static_cast<const float *>(data)[d]);
    }
  }

  void SetUp() override {
    zvec::test_util::RemoveTestPath("flat_memory_lifetime_test");
    MemoryLimitPool::get_instance().init(64UL * 1024 * 1024);
    auto holder =
        std::make_shared<MultiPassIndexHolder<IndexMeta::DT_FP32>>(kDimension);
    for (size_t id = 0; id < kCount; ++id) {
      NumericalVector<float> vector(kDimension);
      for (size_t d = 0; d < kDimension; ++d) vector[d] = value(id, d);
      ASSERT_TRUE(holder->emplace(key(id), vector));
    }
    IndexMeta meta(IndexMeta::DT_FP32, kDimension);
    meta.set_metric("SquaredEuclidean", 0, Params());
    meta.set_major_order(std::get<1>(GetParam()) ? IndexMeta::MO_COLUMN
                                                 : IndexMeta::MO_ROW);
    auto builder = IndexFactory::CreateBuilder("FlatBuilder");
    auto dumper = IndexFactory::CreateDumper("FileDumper");
    ASSERT_NE(nullptr, builder);
    ASSERT_NE(nullptr, dumper);
    ASSERT_EQ(0, builder->init(meta, Params()));
    ASSERT_EQ(0, dumper->create(path_));
    ASSERT_EQ(0, IndexBuilder::TrainBuildAndDump(builder, holder, dumper));
    ASSERT_EQ(0, dumper->close());

    storage_ = IndexFactory::CreateStorage(std::get<0>(GetParam()));
    searcher_ = IndexFactory::CreateSearcher("FlatSearcher");
    ASSERT_NE(nullptr, storage_);
    ASSERT_NE(nullptr, searcher_);
    ASSERT_EQ(0, storage_->init(Params()));
    ASSERT_EQ(0, storage_->open(path_, false));
    ASSERT_EQ(0, searcher_->init(Params()));
    ASSERT_EQ(0, searcher_->load(storage_, nullptr));
  }

  void TearDown() override {
    if (searcher_) searcher_->unload();
    searcher_.reset();
    if (storage_) storage_->close();
    storage_.reset();
    zvec::test_util::RemoveTestPath("flat_memory_lifetime_test");
  }
};

TEST_P(FlatMemoryLifetimeTest, LegacyPointerSurvivesTemporaryProvider) {
  check_vector(searcher_->get_vector(key(3)), 3);
  check_vector(searcher_->get_vector(key(kCount - 1)), kCount - 1);
}

TEST_P(FlatMemoryLifetimeTest, CallerBlocksSurviveProviderAndLaterReads) {
  IndexStorage::MemoryBlock first, tail, copy;
  auto provider = searcher_->create_provider();
  ASSERT_NE(nullptr, provider);
  ASSERT_EQ(0, provider->get_vector(key(3), first));
  copy = first;
  ASSERT_EQ(0, provider->get_vector(key(kCount - 1), tail));
  provider.reset();
  first.reset();
  check_vector(copy.data(), 3);
  check_vector(tail.data(), kCount - 1);

  IndexStorage::MemoryBlock direct;
  ASSERT_EQ(0, searcher_->get_vector(key(35), direct));
  for (size_t id = 0; id < kCount; ++id) {
    check_vector(searcher_->get_vector(key(id)), id);
  }
  check_vector(direct.data(), 35);
  check_vector(copy.data(), 3);
  check_vector(tail.data(), kCount - 1);
  EXPECT_NE(0, searcher_->get_vector(1, direct));
  EXPECT_EQ(nullptr, direct.data());
}

TEST_P(FlatMemoryLifetimeTest, GroupResultsOwnVectorsAndUsePrimaryKeys) {
  auto context = searcher_->create_context();
  ASSERT_NE(nullptr, context);
  context->set_group_params(3, 3);
  context->set_group_by([](uint64_t primary_key) {
    return std::to_string(((primary_key - 1000) / 7) % 3);
  });
  context->set_fetch_vector(true);
  NumericalVector<float> query(kDimension);
  for (size_t d = 0; d < kDimension; ++d) query[d] = value(31, d);
  IndexQueryMeta qmeta(IndexMeta::DT_FP32, kDimension);
  ASSERT_EQ(0, searcher_->search_impl(query.data(), qmeta, context));
  auto groups = context->group_result();
  ASSERT_EQ(3u, groups.size());

  // Neither subsequent reads nor destruction of the search context can own the
  // returned documents' storage-backed or transposed vector buffers.
  for (size_t id = 0; id < kCount; ++id) {
    check_vector(searcher_->get_vector(key(id)), id);
  }
  context.reset();
  for (const auto &group : groups) {
    ASSERT_EQ(3u, group.docs().size());
    for (const auto &doc : group.docs()) {
      ASSERT_LT(doc.index(), kCount);
      EXPECT_EQ(key(doc.index()), doc.key());
      EXPECT_EQ(std::to_string(doc.index() % 3), group.group_id());
      check_vector(doc.vector(), doc.index());
    }
  }

  context = searcher_->create_context();
  context->set_group_params(3, 3);
  context->set_group_by([](uint64_t primary_key) {
    return std::to_string(((primary_key - 1000) / 7) % 3);
  });
  context->set_fetch_vector(true);
  std::vector<std::vector<uint64_t>> keys(1);
  for (size_t id = 0; id < kCount; ++id) keys[0].push_back(key(id));
  ASSERT_EQ(0, searcher_->search_bf_by_p_keys_impl(query.data(), keys, qmeta,
                                                   context));
  auto keyed_groups = context->group_result();
  context.reset();
  ASSERT_EQ(groups.size(), keyed_groups.size());
  for (const auto &group : keyed_groups) {
    ASSERT_EQ(3u, group.docs().size());
    for (const auto &doc : group.docs()) {
      ASSERT_LT(doc.index(), kCount);
      EXPECT_EQ(key(doc.index()), doc.key());
      check_vector(doc.vector(), doc.index());
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    Backends, FlatMemoryLifetimeTest,
    testing::Combine(testing::Values("FileReadStorage", "BufferReadStorage",
                                     "MMapFileReadStorage"),
                     testing::Bool()));

}  // namespace
