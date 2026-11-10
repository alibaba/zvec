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

#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <tuple>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/core/interface/index_factory.h>
#include <zvec/core/interface/index_param_builders.h>
#include "algorithm/hnsw/hnsw_params.h"

namespace zvec::core_interface {
namespace {
constexpr uint32_t kDimension = 16;
using Mode = std::tuple<int, bool, bool, bool>;

class NodePresenceTest : public testing::TestWithParam<Mode> {
 protected:
  void SetUp() override {
    ailego::MemoryLimitPool::get_instance().init(2 * 1024ll * 1024ll * 1024ll);
    std::filesystem::remove_all(path_);
    std::filesystem::create_directory(path_);
  }
  void TearDown() override {
    index_.reset();
    std::filesystem::remove_all(path_);
  }
  BaseIndexParam::Pointer param() {
    const auto [kind, mapped, mmap, contiguous] = GetParam();
    if (kind == 0)
      return FlatIndexParamBuilder()
          .with_dimension(kDimension)
          .with_data_type(DataType::DT_FP32)
          .with_use_id_map(mapped)
          .with_use_contiguous_memory(contiguous)
          .build();
    if (kind == 1)
      return HNSWIndexParamBuilder()
          .with_dimension(kDimension)
          .with_data_type(DataType::DT_FP32)
          .with_use_id_map(mapped)
          .with_use_contiguous_memory(contiguous)
          .build();
    return VamanaIndexParamBuilder()
        .with_dimension(kDimension)
        .with_data_type(DataType::DT_FP32)
        .with_use_id_map(mapped)
        .with_use_contiguous_memory(contiguous)
        .build();
  }
  void open(bool create, bool read_only = false) {
    index_ = IndexFactory::CreateAndInitIndex(*param());
    ASSERT_NE(index_, nullptr);
    const auto storage = std::get<2>(GetParam())
                             ? StorageOptions::StorageType::kMMAP
                             : StorageOptions::StorageType::kBufferPool;
    ASSERT_EQ(0, index_->open(path_ + "/index", {storage, create, read_only}));
  }
  void insert(uint32_t id) {
    std::vector<float> vector(kDimension, float(id));
    ASSERT_EQ(0, index_->add({DenseVector{vector.data()}}, id));
  }
  BaseIndexQueryParam::Pointer query_param() {
    BaseIndexQueryParam::Pointer params;
    switch (std::get<0>(GetParam())) {
      case 0:
        params = std::make_shared<FlatQueryParam>();
        break;
      case 1:
        params = std::make_shared<HNSWQueryParam>();
        break;
      default:
        params = std::make_shared<VamanaQueryParam>();
        break;
    }
    params->topk = 10;
    return params;
  }
  void check(const std::set<uint64_t> &expected) {
    ASSERT_NE(index_, nullptr);
    EXPECT_EQ(6U, index_->get_doc_count());
    EXPECT_EQ(6U, index_->get_slot_count());
    EXPECT_EQ(expected.size(), index_->get_vector_count());
    auto provider = index_->create_index_provider();
    ASSERT_NE(provider, nullptr);
    EXPECT_EQ(expected.size(), provider->count());
    EXPECT_EQ(6U, provider->slot_count());
    auto iter = provider->create_iterator();
    std::set<uint64_t> seen;
    for (; iter->is_valid(); iter->next()) {
      EXPECT_NE(iter->data(), nullptr);
      seen.insert(iter->key());
    }
    EXPECT_EQ(0, iter->status());
    EXPECT_EQ(expected, seen);
    for (uint32_t id = 0; id < 8; ++id) {
      VectorDataBuffer buffer;
      auto ret = index_->fetch(id, &buffer);
      EXPECT_EQ(ret == 0, expected.count(id) != 0) << id;
      core::IndexStorage::MemoryBlock block;
      ret = provider->get_vector(id, block);
      EXPECT_EQ(ret == 0, expected.count(id) != 0) << id;
      if (ret == 0) {
        EXPECT_FLOAT_EQ(*static_cast<const float *>(block.data()), id);
      }
      block.reset(nullptr);
      ret = index_->index_searcher()->get_vector_by_id(id, block);
      EXPECT_EQ(ret == 0, expected.count(id) != 0) << id;
    }
    for (const auto key :
         {UINT64_MAX, uint64_t{1} << 32, (uint64_t{1} << 32) + 2}) {
      core::IndexStorage::MemoryBlock block;
      EXPECT_NE(0, provider->get_vector(key, block)) << key;
    }
    std::vector<float> query(kDimension, 0);
    auto params = query_param();
    params->bf_pks =
        std::make_shared<std::vector<uint64_t>>(std::initializer_list<uint64_t>{
            0, 1, 2, 3, 4, 5, 6, 7, UINT64_MAX, (uint64_t{1} << 32) + 2});
    SearchResult result;
    ASSERT_EQ(0, index_->search({DenseVector{query.data()}}, params, &result));
    seen.clear();
    for (const auto &doc : result.doc_list_) {
      seen.insert(doc.key());
      if (expected.count(doc.key())) {
        EXPECT_FLOAT_EQ(doc.score(), kDimension * doc.key() * doc.key());
      }
    }
    EXPECT_EQ(expected, seen);
    params->bf_pks = std::make_shared<std::vector<uint64_t>>(
        std::initializer_list<uint64_t>{0, 3, 4, 7, UINT64_MAX});
    ASSERT_EQ(0, index_->search({DenseVector{query.data()}}, params, &result));
    EXPECT_TRUE(result.doc_list_.empty());
  }
  std::string path_ = "test_node_presence";
  Index::Pointer index_;
};

TEST_P(NodePresenceTest, GapsAreAbsentWithEitherKeyMappingMode) {
  open(true);
  insert(2);
  insert(5);
  std::vector<float> invalid(kDimension, 1);
  EXPECT_NE(0, index_->add({DenseVector{invalid.data()}}, UINT32_MAX));
  check({2, 5});
  ASSERT_EQ(0, index_->flush());
  ASSERT_EQ(0, index_->close());
  open(false);
  check({2, 5});
  ASSERT_EQ(0, index_->close());
  // BufferStorage uses create-if-missing as write intent, preserving an
  // existing file. Reopen writable before filling a previously empty slot.
  open(true);
  // Filling a previous gap changes vector count, not the addressable span.
  insert(1);
  check({1, 2, 5});
  ASSERT_EQ(0, index_->flush());
  ASSERT_EQ(0, index_->close());
  open(false);
  check({1, 2, 5});
  ASSERT_EQ(0, index_->close());
}

INSTANTIATE_TEST_SUITE_P(StorageAndMapping, NodePresenceTest,
                         testing::Combine(testing::Values(0, 1, 2),
                                          testing::Bool(), testing::Bool(),
                                          testing::Bool()));

TEST_P(NodePresenceTest, KeyInsertionHonorsTheSelectedMapping) {
  open(true);
  ASSERT_NE(index_, nullptr);
  auto streamer = index_->index_searcher();
  auto context = streamer->create_context();
  core::IndexQueryMeta meta(core::IndexMeta::DT_FP32, kDimension);
  const bool mapped = std::get<1>(GetParam());
  const std::vector<uint64_t> keys =
      mapped ? std::vector<uint64_t>{(uint64_t{1} << 48) + 17, UINT32_MAX}
             : std::vector<uint64_t>{2, 5};
  std::vector<float> vector(kDimension, 7);
  for (const auto key : keys)
    ASSERT_EQ(0, streamer->add_impl(key, vector.data(), meta, context));
  auto check = [&] {
    auto provider = index_->create_index_provider();
    EXPECT_EQ(2U, provider->count());
    EXPECT_EQ(mapped ? 2U : 6U, provider->slot_count());
    std::set<uint64_t> actual;
    auto iter = provider->create_iterator();
    for (; iter->is_valid(); iter->next()) actual.insert(iter->key());
    EXPECT_EQ(actual, (std::set<uint64_t>(keys.begin(), keys.end())));
    for (const auto key : keys) {
      core::IndexStorage::MemoryBlock block;
      ASSERT_EQ(0, provider->get_vector(key, block));
      EXPECT_FLOAT_EQ(7, *static_cast<const float *>(block.data()));
    }
    core::IndexStorage::MemoryBlock block;
    EXPECT_NE(0, provider->get_vector(UINT64_MAX, block));
    EXPECT_NE(
        0, provider->get_vector(mapped ? 0 : (uint64_t{1} << 32) + 2, block));
    EXPECT_EQ(
        0, index_->index_searcher()->get_vector_by_id(mapped ? 0 : 2, block));
  };
  check();
  if (!mapped) {
    EXPECT_NE(0, streamer->add_impl((uint64_t{1} << 32) + 2, vector.data(),
                                    meta, context));
  }
  EXPECT_NE(0, streamer->add_impl(UINT64_MAX, vector.data(), meta, context));
  ASSERT_EQ(0, index_->flush());
  ASSERT_EQ(0, index_->close());
  open(false);
  check();
  ASSERT_EQ(0, index_->close());
}
TEST_P(NodePresenceTest, MergeOffsetsAndAppendUseSlotSpans) {
  open(true);
  ASSERT_NE(index_, nullptr);
  insert(4);
  const auto storage = std::get<2>(GetParam())
                           ? StorageOptions::StorageType::kMMAP
                           : StorageOptions::StorageType::kBufferPool;
  std::vector<Index::Pointer> sources;
  for (int source_id = 0; source_id < 2; ++source_id) {
    auto source = IndexFactory::CreateAndInitIndex(*param());
    ASSERT_NE(source, nullptr);
    ASSERT_EQ(0, source->open(path_ + "/source" + std::to_string(source_id),
                              {storage, true}));
    const std::vector<uint32_t> ids = source_id == 0
                                          ? std::vector<uint32_t>{2, 5}
                                          : std::vector<uint32_t>{1, 3};
    for (const auto id : ids) {
      std::vector<float> vector(kDimension, float(source_id * 10 + id));
      ASSERT_EQ(0, source->add({DenseVector{vector.data()}}, id));
    }
    sources.push_back(std::move(source));
  }
  // Source 0 spans six slots, so source 1's local key 1 is global row 7.
  // Appending to the target must start after its existing slot 4.
  IndexFilter filter;
  filter.set([](uint64_t id) { return id == 7; });
  ASSERT_EQ(0, index_->merge(sources, filter));
  EXPECT_EQ(8U, index_->get_slot_count());
  EXPECT_EQ(4U, index_->get_vector_count());
  for (const auto &[id, value] : std::vector<std::pair<uint32_t, float>>{
           {4, 4}, {5, 2}, {6, 5}, {7, 13}}) {
    VectorDataBuffer fetched;
    ASSERT_EQ(0, index_->fetch(id, &fetched));
    const auto &data = std::get<DenseVectorBuffer>(fetched.vector_buffer).data;
    float actual;
    std::memcpy(&actual, data.data(), sizeof(actual));
    EXPECT_FLOAT_EQ(value, actual);
  }
  ASSERT_EQ(0, index_->flush());
  ASSERT_EQ(0, index_->close());
  for (auto &source : sources) ASSERT_EQ(0, source->close());
  open(false);
  EXPECT_EQ(8U, index_->get_slot_count());
  EXPECT_EQ(4U, index_->get_vector_count());
  ASSERT_EQ(0, index_->close());
}
TEST_P(NodePresenceTest, NullInsertionPersistsTrailingDocumentIds) {
  open(true);
  insert(0);  // A genuine all-zero vector is still a vector.
  ASSERT_EQ(0, index_->add({NullVector{}}, 1));
  ASSERT_EQ(0, index_->add_null(3));
  EXPECT_EQ(4U, index_->get_doc_count());
  EXPECT_EQ(1U, index_->get_slot_count());
  EXPECT_EQ(1U, index_->get_vector_count());
  EXPECT_NE(0, index_->add_null(0));
  EXPECT_NE(0, index_->add_null(UINT32_MAX));
  ASSERT_EQ(0, index_->close());
  open(false);
  EXPECT_EQ(4U, index_->get_doc_count());
  EXPECT_EQ(1U, index_->get_vector_count());
  for (uint32_t row : {1U, 2U, 3U}) {
    VectorDataBuffer value;
    EXPECT_NE(0, index_->fetch(row, &value));
  }
  std::vector<float> query(kDimension, 0);
  SearchResult result;
  ASSERT_EQ(
      0, index_->search({DenseVector{query.data()}}, query_param(), &result));
  ASSERT_EQ(1U, result.doc_list_.size());
  EXPECT_EQ(0U, result.doc_list_[0].key());
  EXPECT_FLOAT_EQ(0, result.doc_list_[0].score());
  ASSERT_EQ(0, index_->close());
  open(true);
  insert(2);  // Fill an absent vector without changing the document range.
  EXPECT_EQ(4U, index_->get_doc_count());
  EXPECT_EQ(2U, index_->get_vector_count());
  ASSERT_EQ(0, index_->add_null(4));
  ASSERT_EQ(0, index_->close());
  open(false);
  EXPECT_EQ(5U, index_->get_doc_count());
  EXPECT_EQ(2U, index_->get_vector_count());
  ASSERT_EQ(0, index_->close());
  open(false, true);
  EXPECT_NE(0, index_->add({NullVector{}}, 5));
  EXPECT_NE(0, index_->add_null(5));
  EXPECT_EQ(5U, index_->get_doc_count());
  ASSERT_EQ(0, index_->close());
}

TEST_P(NodePresenceTest, AllNullCoreIndexCanReopenAndAcceptAValue) {
  open(true);
  ASSERT_EQ(0, index_->add({NullVector{}}, 0));
  ASSERT_EQ(0, index_->add({NullVector{}}, 5));
  EXPECT_EQ(6U, index_->get_doc_count());
  EXPECT_EQ(0U, index_->get_slot_count());
  EXPECT_EQ(0U, index_->get_vector_count());
  auto iter = index_->create_index_provider()->create_iterator();
  EXPECT_FALSE(iter->is_valid());
  iter.reset();
  ASSERT_EQ(0, index_->close());
  open(false);
  EXPECT_EQ(6U, index_->get_doc_count());
  EXPECT_EQ(0U, index_->get_vector_count());
  std::vector<float> query(kDimension, 0);
  SearchResult result;
  ASSERT_EQ(
      0, index_->search({DenseVector{query.data()}}, query_param(), &result));
  EXPECT_TRUE(result.doc_list_.empty());
  ASSERT_EQ(0, index_->close());
  open(true);
  insert(2);
  EXPECT_EQ(6U, index_->get_doc_count());
  EXPECT_EQ(1U, index_->get_vector_count());
  ASSERT_EQ(0, index_->close());
  open(false);
  EXPECT_EQ(6U, index_->get_doc_count());
  VectorDataBuffer value;
  EXPECT_EQ(0, index_->fetch(2, &value));
  ASSERT_EQ(0, index_->close());
}

TEST_P(NodePresenceTest, MergeFiltersEveryDocumentIncludingNullTails) {
  open(true);
  ASSERT_EQ(0, index_->add_null(0));
  insert(1);
  const auto storage = std::get<2>(GetParam())
                           ? StorageOptions::StorageType::kMMAP
                           : StorageOptions::StorageType::kBufferPool;
  std::vector<Index::Pointer> sources;
  for (int source_id = 0; source_id < 3; ++source_id) {
    auto source = IndexFactory::CreateAndInitIndex(*param());
    ASSERT_NE(source, nullptr);
    ASSERT_EQ(0, source->open(path_ + "/source" + std::to_string(source_id),
                              {storage, true}));
    if (source_id != 1) {
      std::vector<float> value(kDimension, source_id + 10);
      ASSERT_EQ(0, source->add({DenseVector{value.data()}}, 0));
    }
    ASSERT_EQ(0, source->add_null(source_id == 0 ? 3 : 1));
    sources.push_back(std::move(source));
  }
  std::vector<uint64_t> visited;
  IndexFilter filter;
  filter.set([&](uint64_t row) {
    visited.push_back(row);
    return row == 1 || row == 4 || row == 6;
  });
  ASSERT_EQ(0, index_->merge(sources, filter));
  EXPECT_EQ((std::vector<uint64_t>{0, 1, 2, 3, 4, 5, 6, 7}), visited);
  EXPECT_EQ(7U, index_->get_doc_count());
  EXPECT_EQ(2U, index_->get_vector_count());
  auto provider = index_->create_index_provider();
  std::set<uint64_t> keys;
  auto iter = provider->create_iterator();
  for (; iter->is_valid(); iter->next()) keys.insert(iter->key());
  EXPECT_EQ((std::set<uint64_t>{1, 2}), keys);
  iter.reset();
  provider.reset();
  for (auto &source : sources) ASSERT_EQ(0, source->close());
  ASSERT_EQ(0, index_->close());
  open(false);
  EXPECT_EQ(7U, index_->get_doc_count());
  EXPECT_EQ(2U, index_->get_vector_count());
  ASSERT_EQ(0, index_->close());
}

TEST_P(NodePresenceTest, AllNullMergeKeepsOnlyRetainedDocumentRows) {
  open(true);
  ASSERT_EQ(0, index_->add_null(2));
  auto source = IndexFactory::CreateAndInitIndex(*param());
  ASSERT_NE(source, nullptr);
  const auto storage = std::get<2>(GetParam())
                           ? StorageOptions::StorageType::kMMAP
                           : StorageOptions::StorageType::kBufferPool;
  ASSERT_EQ(0, source->open(path_ + "/source", {storage, true}));
  ASSERT_EQ(0, source->add_null(5));
  ASSERT_EQ(0, source->close());
  source = IndexFactory::CreateAndInitIndex(*param());
  ASSERT_EQ(0, source->open(path_ + "/source", {storage, false}));
  std::vector<uint64_t> visited;
  IndexFilter filter;
  filter.set([&](uint64_t row) {
    visited.push_back(row);
    return row == 1 || row == 5;
  });
  ASSERT_EQ(0, index_->merge({source}, filter));
  EXPECT_EQ((std::vector<uint64_t>{0, 1, 2, 3, 4, 5}), visited);
  EXPECT_EQ(7U, index_->get_doc_count());
  EXPECT_EQ(0U, index_->get_slot_count());
  EXPECT_EQ(0U, index_->get_vector_count());
  ASSERT_EQ(0, source->close());
  ASSERT_EQ(0, index_->close());
  open(false);
  EXPECT_EQ(7U, index_->get_doc_count());
  EXPECT_EQ(0U, index_->get_vector_count());
  ASSERT_EQ(0, index_->close());
}

// Sparse indexes use the same independent presence and key-mapping contract.
class SparseNodePresenceTest
    : public testing::TestWithParam<std::tuple<bool, bool, bool>> {
 protected:
  void SetUp() override {
    ailego::MemoryLimitPool::get_instance().init(2 * 1024ll * 1024ll * 1024ll);
    std::filesystem::remove_all(path_);
    std::filesystem::create_directory(path_);
  }
  void TearDown() override {
    index_.reset();
    std::filesystem::remove_all(path_);
  }
  void open(bool create) {
    const auto [graph, mapped, mmap] = GetParam();
    BaseIndexParam::Pointer param;
    if (graph)
      param = HNSWIndexParamBuilder()
                  .with_is_sparse(true)
                  .with_dimension(0)
                  .with_data_type(DataType::DT_FP32)
                  .with_metric_type(MetricType::kInnerProduct)
                  .with_use_id_map(mapped)
                  .build();
    else
      param = FlatIndexParamBuilder()
                  .with_is_sparse(true)
                  .with_dimension(0)
                  .with_data_type(DataType::DT_FP32)
                  .with_metric_type(MetricType::kInnerProduct)
                  .with_use_id_map(mapped)
                  .build();
    index_ = IndexFactory::CreateAndInitIndex(*param);
    ASSERT_NE(index_, nullptr);
    auto storage = mmap ? StorageOptions::StorageType::kMMAP
                        : StorageOptions::StorageType::kBufferPool;
    ASSERT_EQ(0, index_->open(path_ + "/index", {storage, create}));
  }
  void check(const std::set<uint64_t> &expected, size_t slots) {
    EXPECT_EQ(slots, index_->get_slot_count());
    EXPECT_EQ(expected.size(), index_->get_vector_count());
    auto provider = index_->index_searcher()->create_sparse_provider();
    ASSERT_NE(provider, nullptr);
    EXPECT_EQ(slots, provider->slot_count());
    EXPECT_EQ(expected.size(), provider->count());
    std::set<uint64_t> actual;
    auto iter = provider->create_iterator();
    for (; iter->is_valid(); iter->next()) actual.insert(iter->key());
    EXPECT_EQ(expected, actual);
    for (uint64_t key : expected) {
      uint32_t count = 0;
      std::string indices, values;
      ASSERT_EQ(0, provider->get_sparse_vector(key, &count, &indices, &values));
      EXPECT_EQ(1U, count);
      ASSERT_EQ(sizeof(float), values.size());
      float value;
      std::memcpy(&value, values.data(), sizeof(value));
      EXPECT_FLOAT_EQ(7, value);
    }
    for (uint64_t key : std::vector<uint64_t>{0, 1, 3, 4, 6, UINT64_MAX,
                                              (uint64_t{1} << 32) + 2}) {
      uint32_t count;
      std::string indices, values;
      EXPECT_NE(0, provider->get_sparse_vector(key, &count, &indices, &values))
          << key;
      // Index::fetch accepts a physical document ID, unlike provider key
      // lookup.
      if (key <= UINT32_MAX) {
        VectorDataBuffer buffer;
        bool present = slots == 2 ? key < 2 : expected.count(key) != 0;
        EXPECT_EQ(present,
                  index_->fetch(static_cast<uint32_t>(key), &buffer) == 0)
            << key;
      }
    }
    BaseIndexQueryParam::Pointer params;
    if (std::get<0>(GetParam()))
      params = std::make_shared<HNSWQueryParam>();
    else
      params = std::make_shared<FlatQueryParam>();
    params->topk = 10;
    uint32_t dimension = 17;
    float value = 1;
    auto candidates =
        std::make_shared<std::vector<uint64_t>>(std::initializer_list<uint64_t>{
            0, 1, 3, 4, 6, UINT64_MAX, (uint64_t{1} << 32) + 2});
    candidates->insert(candidates->end(), expected.begin(), expected.end());
    for (bool candidate_search : {false, true}) {
      params->bf_pks = candidate_search ? candidates : nullptr;
      SearchResult result;
      ASSERT_EQ(0, index_->search({SparseVector{1, &dimension, &value}}, params,
                                  &result));
      actual.clear();
      for (const auto &doc : result.doc_list_) actual.insert(doc.key());
      EXPECT_EQ(expected, actual);
      params->group_by_param = std::make_shared<GroupByParam>();
      params->group_by_param->group_topk = 1;
      params->group_by_param->group_count = 10;
      params->group_by_param->group_by = [](uint64_t key) {
        return std::to_string(key);
      };
      ASSERT_EQ(0, index_->search({SparseVector{1, &dimension, &value}}, params,
                                  &result));
      actual.clear();
      for (const auto &group : result.group_doc_list_)
        for (const auto &doc : group.docs()) actual.insert(doc.key());
      EXPECT_EQ(expected, actual);
      params->group_by_param.reset();
    }
  }
  std::string path_ = "test_sparse_node_presence";
  Index::Pointer index_;
};

TEST_P(SparseNodePresenceTest, GapsAreAbsentWithEitherMapping) {
  open(true);
  uint32_t dimension = 17;
  float value = 7;
  for (uint32_t id : {2U, 5U})
    ASSERT_EQ(0, index_->add({SparseVector{1, &dimension, &value}}, id));
  EXPECT_NE(0, index_->add({SparseVector{1, &dimension, &value}}, UINT32_MAX));
  check({2, 5}, 6);
  ASSERT_EQ(0, index_->flush());
  ASSERT_EQ(0, index_->close());
  open(false);
  check({2, 5}, 6);
  ASSERT_EQ(0, index_->close());
}

TEST_P(SparseNodePresenceTest, NativeKeyInsertionHonorsMapping) {
  open(true);
  auto streamer = index_->index_searcher();
  auto context = streamer->create_context();
  core::IndexQueryMeta meta(core::IndexMeta::DT_FP32, 0);
  uint32_t dimension = 17;
  float value = 7;
  bool mapped = std::get<1>(GetParam());
  std::set<uint64_t> keys =
      mapped ? std::set<uint64_t>{(uint64_t{1} << 48) + 17, UINT32_MAX}
             : std::set<uint64_t>{2, 5};
  for (auto key : keys)
    ASSERT_EQ(0, streamer->add_impl(key, 1, &dimension, &value, meta, context));
  EXPECT_NE(
      0, streamer->add_impl(UINT64_MAX, 1, &dimension, &value, meta, context));
  if (!mapped) {
    EXPECT_NE(0, streamer->add_impl((uint64_t{1} << 32) + 2, 1, &dimension,
                                    &value, meta, context));
  }
  check(keys, mapped ? 2 : 6);
  ASSERT_EQ(0, index_->flush());
  ASSERT_EQ(0, index_->close());
  open(false);
  check(keys, mapped ? 2 : 6);
  ASSERT_EQ(0, index_->close());
}

TEST_P(SparseNodePresenceTest, NullInsertionPersistsTrailingSparseDocumentIds) {
  open(true);
  ASSERT_EQ(0, index_->add({NullVector{}}, 0));
  uint32_t dimension = 17;
  float value = 7;
  ASSERT_EQ(0, index_->add({SparseVector{1, &dimension, &value}}, 1));
  ASSERT_EQ(0, index_->add({SparseVector{0, nullptr, nullptr}}, 2));
  ASSERT_EQ(0, index_->add_null(4));
  EXPECT_EQ(5U, index_->get_doc_count());
  EXPECT_EQ(2U, index_->get_vector_count());
  ASSERT_EQ(0, index_->close());
  open(false);
  EXPECT_EQ(5U, index_->get_doc_count());
  VectorDataBuffer fetched;
  EXPECT_EQ(0, index_->fetch(1, &fetched));
  ASSERT_EQ(0, index_->fetch(2, &fetched));
  EXPECT_EQ(0U, std::get<SparseVectorBuffer>(fetched.vector_buffer).count);
  EXPECT_NE(0, index_->fetch(4, &fetched));
  ASSERT_EQ(0, index_->close());
}

INSTANTIATE_TEST_SUITE_P(StorageAndMapping, SparseNodePresenceTest,
                         testing::Combine(testing::Bool(), testing::Bool(),
                                          testing::Bool()));

TEST(DocumentRangeTest, BuilderTrainingPersistsNullTails) {
  ailego::MemoryLimitPool::get_instance().init(2 * 1024ll * 1024ll * 1024ll);
  std::vector<BaseIndexParam::Pointer> params{
      IVFIndexParamBuilder()
          .with_dimension(64)
          .with_data_type(DataType::DT_FP32)
          .with_n_list(2)
          .build()};
#if RABITQ_SUPPORTED
  params.push_back(IVFRabitqIndexParamBuilder()
                       .with_dimension(64)
                       .with_data_type(DataType::DT_FP32)
                       .with_n_list(2)
                       .with_total_bits(4)
                       .build());
#endif
#if DISKANN_SUPPORTED
  params.push_back(DiskAnnIndexParamBuilder()
                       .with_dimension(64)
                       .with_data_type(DataType::DT_FP32)
                       .with_max_degree(8)
                       .with_list_size(16)
                       .with_pq_chunk_num(4)
                       .build());
#endif
  const std::string path = "test_builder_null_tail";
  for (const auto &param : params) {
    for (auto storage : {StorageOptions::StorageType::kMMAP,
                         StorageOptions::StorageType::kBufferPool}) {
      SCOPED_TRACE(int(param->index_type));
      SCOPED_TRACE(int(storage));
      std::filesystem::remove_all(path);
      auto index = IndexFactory::CreateAndInitIndex(*param);
      ASSERT_NE(index, nullptr);
      ASSERT_EQ(0, index->open(path, {storage, true}));
      ASSERT_EQ(0, index->add({NullVector{}}, 0));
      for (uint32_t id = 1; id <= 16; ++id) {
        std::vector<float> vector(64, float(id));
        ASSERT_EQ(0, index->add({DenseVector{vector.data()}}, id));
      }
      ASSERT_EQ(0, index->add({NullVector{}}, 22));
      EXPECT_EQ(23U, index->get_doc_count());
      ASSERT_EQ(0, index->train());
      EXPECT_EQ(23U, index->get_doc_count());
      EXPECT_EQ(16U, index->get_vector_count());
      EXPECT_NE(0, index->add_null(23));
      ASSERT_EQ(0, index->close());
      index = IndexFactory::CreateAndInitIndex(*param);
      ASSERT_EQ(0, index->open(path, {storage, false}));
      EXPECT_EQ(23U, index->get_doc_count());
      EXPECT_EQ(16U, index->get_vector_count());
      auto iter = index->create_index_provider()->create_iterator();
      std::set<uint64_t> keys;
      for (; iter->is_valid(); iter->next()) keys.insert(iter->key());
      EXPECT_EQ(0, iter->status());
      EXPECT_EQ(16U, keys.size());
      for (uint32_t id = 1; id <= 16; ++id) EXPECT_EQ(1U, keys.count(id));
      iter.reset();
      VectorDataBuffer value;
      if (param->index_type != IndexType::kIVFRabitq)
        EXPECT_EQ(0, index->fetch(1, &value));
      EXPECT_NE(0, index->fetch(0, &value));
      EXPECT_NE(0, index->fetch(22, &value));
      ASSERT_EQ(0, index->close());
      std::filesystem::remove_all(path);
    }
  }
}

TEST(DocumentRangeTest, CorruptMetadataIsRejected) {
  ailego::MemoryLimitPool::get_instance().init(2 * 1024ll * 1024ll * 1024ll);
  const std::string path = "test_corrupt_document_range";
  auto param = FlatIndexParamBuilder()
                   .with_dimension(kDimension)
                   .with_data_type(DataType::DT_FP32)
                   .build();
  for (auto mode : {StorageOptions::StorageType::kMMAP,
                    StorageOptions::StorageType::kBufferPool}) {
    std::filesystem::remove_all(path);
    auto index = IndexFactory::CreateAndInitIndex(*param);
    ASSERT_NE(index, nullptr);
    ASSERT_EQ(0, index->open(path, {mode, true}));
    ASSERT_EQ(0, index->add_null(4));
    ASSERT_EQ(0, index->close());
    auto storage = core::IndexFactory::CreateStorage("FileReadStorage");
    ASSERT_EQ(0, storage->init({}));
    ASSERT_EQ(0, storage->open(path, false));
    auto segment = storage->get("IndexDocumentRange");
    ASSERT_NE(segment, nullptr);
    ASSERT_EQ(16U, segment->data_size());
    auto offset = segment->data_offset();
    segment.reset();
    ASSERT_EQ(0, storage->close());
    // Change the document end without updating its checksum.
    std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
    ASSERT_TRUE(file);
    file.seekp(offset + 8);
    file.put(6);
    file.close();
    index = IndexFactory::CreateAndInitIndex(*param);
    EXPECT_NE(0, index->open(path, {mode, false}));
    index.reset();
    std::filesystem::remove_all(path);
  }
}
}  // namespace
}  // namespace zvec::core_interface
