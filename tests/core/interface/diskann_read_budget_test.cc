// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#if DISKANN_SUPPORTED

#include <algorithm>
#include <array>
#include <filesystem>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/utility/float_helper.h>
#include <zvec/core/interface/index_factory.h>
#include <zvec/core/interface/index_param_builders.h>
#include "../../ailego/buffer/pool_payload_budget.h"

namespace zvec::core_interface {
namespace {
using StorageType = StorageOptions::StorageType;

class DiskAnnReadBudgetTest : public ::testing::TestWithParam<StorageType> {
 protected:
  static constexpr uint32_t kDimension = 64;
  static constexpr uint32_t kDocs = 512;
  static constexpr size_t kPoolBytes = 16UL << 20;
  static constexpr uint64_t kHighRss = uint64_t{1} << 40;

  void SetUp() override {
    auto &pool = ailego::MemoryLimitPool::get_instance();
    previous_rss_ = pool.process_rss_limit();
    previous_capacity_ = ailego::PoolPayloadBudget::get_instance().capacity();
    ASSERT_EQ(0, ailego::PoolPayloadBudget::get_instance().init(kPoolBytes));
    ASSERT_TRUE(pool.configure_process_rss_limit(0));
    const auto *test = ::testing::UnitTest::GetInstance()->current_test_info();
    std::string name = test->name();
    std::replace(name.begin(), name.end(), '/', '_');
    directory_ = "diskann_read_budget_" + name;
    std::filesystem::remove_all(directory_);
    ASSERT_TRUE(std::filesystem::create_directory(directory_));
  }

  void TearDown() override {
    auto &pool = ailego::MemoryLimitPool::get_instance();
    EXPECT_TRUE(pool.configure_process_rss_limit(previous_rss_));
    EXPECT_EQ(0u, ailego::PoolPayloadBudget::get_instance().used());
    EXPECT_EQ(0, ailego::PoolPayloadBudget::get_instance().init(previous_capacity_));
    std::filesystem::remove_all(directory_);
  }

  template <typename Value>
  void Build(const std::string &path, const BaseIndexParam &param,
             std::vector<Value> *rows, unsigned seed) {
    std::mt19937 random(seed);
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
    rows->resize(kDocs * kDimension);
    for (auto &value : *rows) value = static_cast<Value>(uniform(random));
    auto index = IndexFactory::CreateAndInitIndex(param);
    ASSERT_NE(nullptr, index);
    ASSERT_EQ(0, index->open(path, {StorageType::kMMAP, true}));
    for (uint32_t id = 0; id < kDocs; ++id) {
      ASSERT_EQ(0, index->add(
          VectorData{DenseVector{rows->data() + id * kDimension}}, id));
    }
    ASSERT_EQ(0, index->train());
    ASSERT_EQ(0, index->close());
  }

  BaseIndexParam::Pointer Parameters(bool fp16) {
    return DiskAnnIndexParamBuilder()
        .with_metric_type(MetricType::kL2sq)
        .with_data_type(fp16 ? DataType::DT_FP16 : DataType::DT_FP32)
        .with_dimension(kDimension)
        .with_max_degree(16)
        .with_list_size(48)
        .with_pq_chunk_num(8)
        .build();
  }

  std::filesystem::path directory_;
  size_t previous_capacity_{0};
  uint64_t previous_rss_{0};
};

TEST_P(DiskAnnReadBudgetTest, SearchAndFetchWith4095ByteCache) {
  const auto path = (directory_ / "fp16.index").string();
  const auto param = Parameters(true);
  std::vector<ailego::Float16> rows;
  ASSERT_NO_FATAL_FAILURE(Build(path, *param, &rows, 317));
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  ASSERT_EQ(0, pool.init(4095));
  ASSERT_TRUE(ailego::MemoryLimitPool::get_instance()
                  .configure_process_rss_limit(kHighRss));

  auto index = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, index);
  ASSERT_EQ(0, index->open(path, {GetParam(), false, true}));
  auto query = std::make_shared<DiskAnnQueryParam>();
  query->topk = 1;
  query->list_size = 48;
  query->fetch_vector = true;
  const VectorData input{DenseVector{rows.data()}};
  SearchResult result;
  for (bool linear : {false, true}) {
    query->is_linear = linear;
    ASSERT_EQ(0, index->search(input, query, &result));
    ASSERT_EQ(1u, result.doc_list_.size());
    const auto id = result.doc_list_.front().key();
    if (linear) {
      EXPECT_EQ(0u, id);
    }
    ASSERT_LT(id, kDocs);
    const std::string expected(
        reinterpret_cast<const char *>(rows.data() + id * kDimension),
        kDimension * sizeof(ailego::Float16));
    VectorDataBuffer fetched;
    ASSERT_EQ(0, index->fetch(id, &fetched));
    EXPECT_EQ(expected, std::get<DenseVectorBuffer>(fetched.vector_buffer).data);
    EXPECT_EQ(expected, result.doc_list_.front().vector_string());
    EXPECT_EQ(0u, pool.stats().page_used);
  }
  query->is_linear = false;
  query->bf_pks = std::make_shared<std::vector<uint64_t>>(1, 7);
  ASSERT_EQ(0, index->search(input, query, &result));
  ASSERT_EQ(1u, result.doc_list_.size());
  EXPECT_EQ(7u, result.doc_list_.front().key());
  ASSERT_EQ(0, index->close());
}

TEST_P(DiskAnnReadBudgetTest, SameThreadSwitchRssRefusalAndRetry) {
  const auto param = Parameters(false);
  const auto first_path = (directory_ / "first.index").string();
  const auto second_path = (directory_ / "second.index").string();
  std::vector<float> first_rows, second_rows;
  ASSERT_NO_FATAL_FAILURE(Build(first_path, *param, &first_rows, 17));
  ASSERT_NO_FATAL_FAILURE(Build(second_path, *param, &second_rows, 27));
  auto &rss = ailego::MemoryLimitPool::get_instance();
  ASSERT_TRUE(rss.configure_process_rss_limit(kHighRss));

  auto first = IndexFactory::CreateAndInitIndex(*param);
  auto second = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, first);
  ASSERT_NE(nullptr, second);
  ASSERT_EQ(0, first->open(first_path, {GetParam(), false, true}));
  ASSERT_EQ(0, second->open(second_path, {GetParam(), false, true}));

  auto query = std::make_shared<DiskAnnQueryParam>();
  query->topk = 1;
  query->list_size = 48;
  const VectorData first_input{DenseVector{first_rows.data() + kDimension}};
  const VectorData second_input{DenseVector{second_rows.data() + 2 * kDimension}};

  SearchResult result;
  ASSERT_EQ(0, second->search(second_input, query, &result));
  ASSERT_EQ(1u, result.doc_list_.size());
  const auto second_key = result.doc_list_.front().key();
  ASSERT_EQ(0, first->search(first_input, query, &result));
  ASSERT_EQ(1u, result.doc_list_.size());
  const auto first_key = result.doc_list_.front().key();
  ASSERT_TRUE(rss.configure_process_rss_limit(1));
  ASSERT_GT(rss.process_rss(), rss.process_rss_limit());

  EXPECT_EQ(core::IndexError_NoMemory,
            second->search(second_input, query, &result));
  ASSERT_EQ(1u, result.doc_list_.size());
  EXPECT_EQ(first_key, result.doc_list_.front().key());
  std::array<int64_t, 1> ids{-42};
  std::array<float, 1> scores{-42.0f};
  EXPECT_EQ(core::IndexError_NoMemory,
            second->search_internal_ids(second_input, query, ids.data(), scores.data()));
  EXPECT_EQ(-42, ids.front());
  EXPECT_EQ(-42.0f, scores.front());
  VectorDataBuffer fetched;
  EXPECT_EQ(core::IndexError_NoMemory, second->fetch(2, &fetched));

  ASSERT_TRUE(rss.configure_process_rss_limit(kHighRss));
  ASSERT_EQ(0, second->search(second_input, query, &result));
  ASSERT_EQ(1u, result.doc_list_.size());
  EXPECT_EQ(second_key, result.doc_list_.front().key());
  ASSERT_EQ(0, second->search_internal_ids(second_input, query, ids.data(),
                                            scores.data()));
  EXPECT_EQ(static_cast<int64_t>(second_key), ids.front());
  ASSERT_EQ(0, second->fetch(2, &fetched));
  const auto previous = std::get<DenseVectorBuffer>(fetched.vector_buffer).data;
  ASSERT_FALSE(previous.empty());

  ASSERT_TRUE(rss.configure_process_rss_limit(1));
  EXPECT_EQ(core::IndexError_NoMemory, second->fetch(1, &fetched));
  EXPECT_EQ(previous, std::get<DenseVectorBuffer>(fetched.vector_buffer).data);
  ASSERT_TRUE(rss.configure_process_rss_limit(kHighRss));
  ASSERT_EQ(0, second->fetch(1, &fetched));
  EXPECT_NE(previous, std::get<DenseVectorBuffer>(fetched.vector_buffer).data);
  ASSERT_EQ(0, second->close());
  ASSERT_TRUE(rss.configure_process_rss_limit(1));
  EXPECT_EQ(core::IndexError_NoMemory,
            second->open(second_path, {GetParam(), false, true}));
  ASSERT_TRUE(rss.configure_process_rss_limit(kHighRss));
  ASSERT_EQ(0, second->open(second_path, {GetParam(), false, true}));
  ASSERT_EQ(0, second->search(second_input, query, &result));
  ASSERT_EQ(1u, result.doc_list_.size());
  EXPECT_EQ(second_key, result.doc_list_.front().key());
  ASSERT_EQ(0, second->close());
  ASSERT_EQ(0, first->close());
}

INSTANTIATE_TEST_SUITE_P(Backends, DiskAnnReadBudgetTest,
                         ::testing::Values(StorageType::kMMAP,
                                           StorageType::kBufferPool));
}  // namespace
}  // namespace zvec::core_interface

#endif  // DISKANN_SUPPORTED
