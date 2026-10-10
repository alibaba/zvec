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

#include "interface/indexes/buffered_input.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <limits>
#include <ailego/pattern/scope_guard.h>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/buffer/vector_page_table.h>
#include <zvec/core/interface/index_factory.h>
#include <zvec/core/interface/index_param_builders.h>

namespace zvec::core_interface {
namespace {

class BufferedInputTest : public ::testing::Test {
 protected:
  const std::filesystem::path directory_{"buffered_input_test_data"};
  size_t old_capacity_{0};
  void SetUp() override {
    auto &pool = ailego::MemoryLimitPool::get_instance();
    ASSERT_EQ(pool.used(), 0u);
    old_capacity_ = pool.capacity();
    const size_t metadata =
        ailego::VecBufferPool::metadata_bytes_for_page_count(4096, true);
    ASSERT_EQ(pool.init(metadata + 768 * 1024), 0);
    ASSERT_TRUE(std::filesystem::create_directory(directory_));
  }
  void TearDown() override {
    EXPECT_TRUE(std::filesystem::is_empty(directory_));
    EXPECT_TRUE(std::filesystem::remove(directory_));
    auto &pool = ailego::MemoryLimitPool::get_instance();
    EXPECT_EQ(pool.used(), 0u);
    EXPECT_EQ(pool.init(old_capacity_), 0);
  }
  std::shared_ptr<BufferedInput> make(core::IndexMeta::DataType type,
                                      size_t dim) {
    core::IndexQueryMeta meta;
    meta.set_meta(type, dim);
    return std::make_shared<BufferedInput>(meta,
                                           (directory_ / "input").string());
  }
};

TEST_F(BufferedInputTest, RawPrecisionsSparseKeysAndOverwrite) {
  for (auto type : {core::IndexMeta::DT_FP32, core::IndexMeta::DT_FP16,
                    core::IndexMeta::DT_INT8}) {
    auto input = make(type, 8);
    std::string first(input->element_size(), '\x23');
    std::string second(input->element_size(), '\x42');
    const uint32_t large = std::numeric_limits<uint32_t>::max() - 1;
    ASSERT_EQ(input->add(large, first.data()), 0);
    ASSERT_EQ(input->add(7, first.data()), 0);
    ASSERT_EQ(input->add(large, second.data()), 0);
    const auto charged =
        ailego::MemoryLimitPool::get_instance().external_used();
    for (size_t i = 0; i < 128; ++i) {
      ASSERT_EQ(input->add(large, second.data()), 0);
    }
    EXPECT_EQ(ailego::MemoryLimitPool::get_instance().external_used(), charged);
    ASSERT_EQ(input->count(), 2u);
    std::string value;
    ASSERT_EQ(input->fetch(large, &value), 0);
    EXPECT_EQ(value, second);
    EXPECT_EQ(input->fetch(8, &value), core::IndexError_NoExist);
    EXPECT_EQ(input->fetch(large + 1, &value), core::IndexError_OutOfRange);
    EXPECT_EQ(input->add(8, nullptr), core::IndexError_InvalidArgument);
    ASSERT_EQ(input->prepare(), 0);
    auto iter = input->create_iterator();
    ASSERT_TRUE(iter->is_valid());
    EXPECT_EQ(iter->key(), 7u);
    EXPECT_EQ(
        std::string(static_cast<const char *>(iter->data()), first.size()),
        first);
    iter->next();
    ASSERT_TRUE(iter->is_valid());
    EXPECT_EQ(iter->key(), large);
    EXPECT_EQ(
        std::string(static_cast<const char *>(iter->data()), second.size()),
        second);
    iter->next();
    EXPECT_FALSE(iter->is_valid());
    EXPECT_EQ(iter->status(), 0);
  }
}

TEST_F(BufferedInputTest, MultipleSegmentsExceedPoolAndReadersOwnOneRow) {
  auto input = make(core::IndexMeta::DT_FP32, 1024);
  auto &pool = ailego::MemoryLimitPool::get_instance();
  std::array<float, 1024> row{};
  // Include large-page hosts, where fixed writeback staging is larger.
  const size_t kRows =
      std::max<size_t>(2048, pool.capacity() / sizeof(row) + 1);
  ASSERT_GT(kRows * sizeof(row), pool.capacity());
  for (size_t i = 0; i < kRows; ++i) {
    row.fill(static_cast<float>(i));
    ASSERT_EQ(input->add(static_cast<uint32_t>(i), row.data()), 0) << i;
    ASSERT_LE(pool.used(), pool.capacity());
  }
  ASSERT_EQ(input->prepare(), 0);
  core::OrdinalAccessHolder::Reader::Pointer first, second;
  ASSERT_EQ(input->create_ordinal_reader(&first), 0);
  ASSERT_EQ(input->create_ordinal_reader(&second), 0);
  uint64_t key = 0;
  const void *data = nullptr;
  ASSERT_EQ(first->read(37, &key, &data), 0);
  EXPECT_EQ(key, 37u);
  const auto *held = static_cast<const float *>(data);
  for (size_t i = 0; i < kRows; ++i) {
    ASSERT_EQ(second->read(i, &key, &data), 0) << i;
    ASSERT_EQ(key, i);
    row.fill(static_cast<float>(i));
    ASSERT_EQ(std::memcmp(data, row.data(), sizeof(row)), 0);
  }
  EXPECT_EQ(held[0], 37.0f);
  EXPECT_LT(pool.external_used(), 512u * 1024u);
  first->reset();
  second->reset();
  input.reset();
  // Readers retain the source lifetime, not an unsafe pointer to an index.
  ASSERT_EQ(first->read(0, &key, &data), 0);
  first.reset();
  second.reset();
  EXPECT_EQ(pool.used(), 0u);
}

TEST_F(BufferedInputTest, BudgetFailurePreservesInputAndCanBeRetried) {
  auto input = make(core::IndexMeta::DT_FP32, 8);
  std::array<float, 8> row{};
  ASSERT_EQ(input->add(1, row.data()), 0);
  ASSERT_EQ(input->flush(), 0);
  auto &pool = ailego::MemoryLimitPool::get_instance();
  size_t reservation =
      pool.capacity() - pool.metadata_used() - pool.external_used();
  ASSERT_TRUE(pool.try_charge_external(reservation));
  auto release =
      ailego::ScopeGuard::Make([&]() { pool.release_external(reservation); });
  EXPECT_EQ(input->add(2, row.data()), core::IndexError_NoMemory);
  EXPECT_EQ(input->prepare(), core::IndexError_NoMemory);
  EXPECT_EQ(input->count(), 1u);
  pool.release_external(reservation);
  reservation = 0;
  std::string value;
  ASSERT_EQ(input->fetch(1, &value), 0);
  EXPECT_EQ(std::memcmp(value.data(), row.data(), sizeof(row)), 0);
  ASSERT_EQ(input->add(2, row.data()), 0);
  ASSERT_EQ(input->prepare(), 0);
  EXPECT_EQ(input->count(), 2u);
}

TEST_F(BufferedInputTest, DirectIndexesReleaseUntrainedInputOnClose) {
  for (bool diskann : {false, true}) {
#if !DISKANN_SUPPORTED
    if (diskann) continue;
#endif
    BaseIndexParam::Pointer param;
    if (diskann) {
      param = DiskAnnIndexParamBuilder()
                  .with_dimension(8)
                  .with_data_type(DataType::DT_FP32)
                  .with_metric_type(MetricType::kL2sq)
                  .with_pq_chunk_num(2)
                  .build();
    } else {
      param = IVFIndexParamBuilder()
                  .with_dimension(8)
                  .with_data_type(DataType::DT_FP32)
                  .with_metric_type(MetricType::kL2sq)
                  .with_n_list(2)
                  .build();
    }
    auto index = IndexFactory::CreateAndInitIndex(*param);
    ASSERT_NE(index, nullptr);
    ASSERT_EQ(index->open((directory_ / "target").string(),
                          {StorageOptions::StorageType::kBufferPool, true}),
              0);
    std::array<float, 8> row{};
    const uint32_t large = std::numeric_limits<uint32_t>::max() - 1;
    ASSERT_EQ(index->add(VectorData{DenseVector{row.data()}}, large), 0);
    row.fill(7.0f);
    ASSERT_EQ(index->add(VectorData{DenseVector{row.data()}}, large), 0);
    VectorDataBuffer output;
    ASSERT_EQ(index->fetch(large, &output), 0);
    EXPECT_EQ(std::memcmp(
                  std::get<DenseVectorBuffer>(output.vector_buffer).data.data(),
                  row.data(), sizeof(row)),
              0);
    ASSERT_EQ(index->flush(), 0);
    ASSERT_EQ(index->close(), 0);
    EXPECT_EQ(index->train(), core::IndexError_NoReady);
    EXPECT_EQ(index->add(VectorData{DenseVector{row.data()}}, large),
              core::IndexError_NoReady);
    EXPECT_TRUE(std::filesystem::is_empty(directory_));
    EXPECT_EQ(ailego::MemoryLimitPool::get_instance().used(), 0u);
  }
}

TEST_F(BufferedInputTest, ReaderBudgetErrorIsExplicitAndFreshReaderCanRetry) {
  auto input = make(core::IndexMeta::DT_FP32, 8);
  std::array<float, 8> row{};
  ASSERT_EQ(input->add(1, row.data()), 0);
  ASSERT_EQ(input->prepare(), 0);
  auto &pool = ailego::MemoryLimitPool::get_instance();
  size_t reservation =
      pool.capacity() - pool.metadata_used() - pool.external_used();
  ASSERT_TRUE(pool.try_charge_external(reservation));
  auto release =
      ailego::ScopeGuard::Make([&]() { pool.release_external(reservation); });
  auto iter = input->create_iterator();
  EXPECT_EQ(iter->data(), nullptr);
  EXPECT_EQ(iter->status(), core::IndexError_NoMemory);
  EXPECT_FALSE(iter->is_valid());
  pool.release_external(reservation);
  reservation = 0;
  // A failed iterator stays failed; a new build attempt gets a fresh reader.
  EXPECT_EQ(iter->status(), core::IndexError_NoMemory);
  iter = input->create_iterator();
  ASSERT_NE(iter->data(), nullptr);
  EXPECT_EQ(iter->status(), 0);
  EXPECT_EQ(iter->key(), 1u);
}

}  // namespace
}  // namespace zvec::core_interface
