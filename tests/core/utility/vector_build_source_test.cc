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

#include "utility/vector_build_source.h"
#include <array>
#include <cstring>
#include <filesystem>
#include <limits>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/buffer/vector_page_table.h>
#include <zvec/core/framework/index_provider.h>

namespace zvec::core {
namespace {

template <IndexMeta::DataType Type, class T>
void CheckBorrowedNumericalInput() {
  auto input = std::make_shared<MultiPassIndexProvider<Type>>(3);
  for (uint64_t i = 0; i < 5; ++i) {
    ailego::NumericalVector<T> row(3);
    for (size_t d = 0; d < 3; ++d) row[d] = static_cast<T>(float(i * 3 + d));
    ASSERT_TRUE(input->emplace(i % 2, std::move(row)));
  }
  VectorBuildSource::Pointer source;
  ASSERT_EQ(VectorBuildSource::Create(input, &source), 0);
  EXPECT_EQ(source->count(), 5u);
  OrdinalAccessHolder::Reader::Pointer reader;
  ASSERT_EQ(source->create_ordinal_reader(&reader), 0);
  auto iter = input->create_iterator();
  for (size_t i = 0; i < 5; ++i, iter->next()) {
    uint64_t key = 0;
    const void *data = nullptr;
    ASSERT_EQ(reader->read(i, &key, &data), 0);
    EXPECT_EQ(key, i % 2);
    // Identity, not just value equality: no vector copy and no key lookup
    // that would collapse the two occurrences of a duplicate key.
    EXPECT_EQ(data, iter->data());
    reader->reset();
  }
  iter.reset();
  std::weak_ptr<IndexHolder> weak = input;
  input.reset();
  source.reset();
  EXPECT_FALSE(weak.expired());
  uint64_t key = 0;
  const void *data = nullptr;
  ASSERT_EQ(reader->read(4, &key, &data), 0);
  EXPECT_EQ(static_cast<float>(*static_cast<const T *>(data)), 12.0f);
  EXPECT_EQ(reader->read(5, &key, &data), IndexError_OutOfRange);
  EXPECT_EQ(reader->read(0, nullptr, &data), IndexError_InvalidArgument);
  reader.reset();
  EXPECT_TRUE(weak.expired());
}

TEST(VectorBuildSourceTest, NumericalProvidersBorrowAllPrecisionsByOrdinal) {
  CheckBorrowedNumericalInput<IndexMeta::DT_FP32, float>();
  CheckBorrowedNumericalInput<IndexMeta::DT_FP64, double>();
  CheckBorrowedNumericalInput<IndexMeta::DT_FP16, ailego::Float16>();
  CheckBorrowedNumericalInput<IndexMeta::DT_INT8, int8_t>();
  CheckBorrowedNumericalInput<IndexMeta::DT_INT16, int16_t>();
}

// One-pass source reusing the same row buffer on every data() call. No test
// retains a full input corpus in memory, including the larger-than-pool case.
class StreamingInput : public IndexHolder {
 public:
  size_t rows{17};
  size_t reported_count{17};
  size_t fail_at{std::numeric_limits<size_t>::max()};
  bool null_data{false};
  size_t iterators{0};
  static constexpr size_t kDim = 257;

  size_t count() const override {
    return reported_count;
  }
  size_t dimension() const override {
    return kDim;
  }
  size_t element_size() const override {
    return kDim * sizeof(float);
  }
  IndexMeta::DataType data_type() const override {
    return IndexMeta::DT_FP32;
  }
  bool multipass() const override {
    return false;
  }

  class Iter : public IndexHolder::Iterator {
   public:
    explicit Iter(StreamingInput *input) : input_(input) {}
    const void *data() const override {
      if (input_->null_data) return nullptr;
      row_.fill(static_cast<float>(ordinal_));
      return row_.data();
    }
    uint64_t key() const override {
      return (uint64_t{1} << 40) + ordinal_ % 7;
    }
    bool is_valid() const override {
      return ordinal_ < input_->rows && ordinal_ != input_->fail_at;
    }
    int status() const override {
      return ordinal_ == input_->fail_at ? IndexError_ReadData : 0;
    }
    void next() override {
      ++ordinal_;
    }

   private:
    StreamingInput *input_;
    size_t ordinal_{0};
    mutable std::array<float, kDim> row_{};
  };

  Iterator::Pointer create_iterator() override {
    ++iterators;
    return std::make_unique<Iter>(this);
  }
};

class BuildVectorSpoolTest : public ::testing::Test {
 protected:
  const std::filesystem::path directory_{"build_vector_spool_test_data"};
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
  std::string prefix() const {
    return (directory_ / "input").string();
  }
};

TEST_F(BuildVectorSpoolTest, OnePassSpillsAndReadersKeepIndependentRows) {
  auto input = std::make_shared<StreamingInput>();
  auto &pool = ailego::MemoryLimitPool::get_instance();
  input->rows =
      std::max<size_t>(8192, pool.capacity() / input->element_size() * 2);
  input->reported_count = std::numeric_limits<size_t>::max();
  ASSERT_GT(input->rows * input->element_size(), pool.capacity());
  VectorBuildSource::Pointer source;
  ASSERT_EQ(VectorBuildSource::Create(input, &source, prefix()), 0);
  ASSERT_EQ(source->count(), input->rows);
  EXPECT_EQ(input->iterators, 1u);
  ASSERT_LE(pool.used(), pool.capacity());
  OrdinalAccessHolder::Reader::Pointer first, second;
  ASSERT_EQ(source->create_ordinal_reader(&first), 0);
  ASSERT_EQ(source->create_ordinal_reader(&second), 0);
  uint64_t key = 0;
  const void *data = nullptr;
  ASSERT_EQ(first->read(37, &key, &data), 0);
  const auto *held = static_cast<const float *>(data);
  // Reverse order crosses all chunk/page boundaries and checks duplicate keys.
  for (size_t i = source->count(); i-- > 0;) {
    ASSERT_EQ(second->read(i, &key, &data), 0) << i;
    EXPECT_EQ(key, (uint64_t{1} << 40) + i % 7);
    std::array<float, StreamingInput::kDim> row{};
    row.fill(static_cast<float>(i));
    ASSERT_EQ(std::memcmp(row.data(), data, sizeof(row)), 0) << i;
    ASSERT_LE(pool.used(), pool.capacity());
  }
  EXPECT_EQ(held[0], 37.0f);
  source.reset();
  ASSERT_EQ(second->read(7, &key, &data), 0);
  first->reset();
  ASSERT_EQ(first->read(11, &key, &data), 0);
  EXPECT_EQ(static_cast<const float *>(data)[0], 11.0f);
}

TEST_F(BuildVectorSpoolTest, IterationAndCountErrorsDoNotPublishOrLeak) {
  for (size_t fail_at : {size_t{0}, size_t{8}, size_t{17}}) {
    auto input = std::make_shared<StreamingInput>();
    input->fail_at = fail_at;
    VectorBuildSource::Pointer source;
    EXPECT_EQ(VectorBuildSource::Create(input, &source, prefix()),
              IndexError_ReadData);
    EXPECT_EQ(source, nullptr);
    EXPECT_TRUE(std::filesystem::is_empty(directory_));
  }
  for (size_t count : {size_t{0}, size_t{16}, size_t{18}}) {
    auto input = std::make_shared<StreamingInput>();
    input->reported_count = count;
    VectorBuildSource::Pointer source;
    EXPECT_EQ(VectorBuildSource::Create(input, &source, prefix()),
              IndexError_InvalidLength);
    EXPECT_EQ(source, nullptr);
    EXPECT_TRUE(std::filesystem::is_empty(directory_));
  }
  auto input = std::make_shared<StreamingInput>();
  input->null_data = true;
  VectorBuildSource::Pointer source;
  EXPECT_EQ(VectorBuildSource::Create(input, &source, prefix()),
            IndexError_ReadData);
}

TEST_F(BuildVectorSpoolTest, BudgetFailureNeverFallsBackToUnmanagedCopy) {
  auto &pool = ailego::MemoryLimitPool::get_instance();
  ASSERT_EQ(pool.init(1), 0);
  auto input = std::make_shared<StreamingInput>();
  VectorBuildSource::Pointer source;
  EXPECT_EQ(VectorBuildSource::Create(input, &source, prefix()),
            IndexError_NoMemory);
  EXPECT_EQ(source, nullptr);
}

class FailingOrdinalInput : public StreamingInput, public OrdinalAccessHolder {
 public:
  int error{IndexError_ReadData};
  int create_ordinal_reader(Reader::Pointer *) override {
    return error;
  }
};

TEST_F(BuildVectorSpoolTest, OnlyUnsupportedOrdinalCapabilityMayFallBack) {
  auto input = std::make_shared<FailingOrdinalInput>();
  VectorBuildSource::Pointer source;
  EXPECT_EQ(VectorBuildSource::Create(input, &source, prefix()),
            IndexError_ReadData);
  EXPECT_EQ(input->iterators, 0u);
  EXPECT_EQ(source, nullptr);
  input->error = 0;  // Success with a null reader is a broken provider.
  EXPECT_EQ(VectorBuildSource::Create(input, &source, prefix()),
            IndexError_Runtime);
  EXPECT_EQ(input->iterators, 0u);
  input->error = IndexError_NotImplemented;
  EXPECT_EQ(VectorBuildSource::Create(input, &source, prefix()), 0);
  EXPECT_EQ(input->iterators, 1u);
}

TEST_F(BuildVectorSpoolTest, EmptyInputDoesNotCreateBackingFile) {
  auto input = std::make_shared<StreamingInput>();
  input->rows = input->reported_count = 0;
  VectorBuildSource::Pointer source;
  ASSERT_EQ(VectorBuildSource::Create(input, &source, prefix()), 0);
  EXPECT_EQ(source->count(), 0u);
  EXPECT_TRUE(std::filesystem::is_empty(directory_));
}

TEST_F(BuildVectorSpoolTest, UnconfiguredPoolPreservesInMemoryFallback) {
  auto &pool = ailego::MemoryLimitPool::get_instance();
  ASSERT_EQ(pool.init(0), 0);
  auto input = std::make_shared<StreamingInput>();
  VectorBuildSource::Pointer source;
  ASSERT_EQ(VectorBuildSource::Create(input, &source), 0);
  OrdinalAccessHolder::Reader::Pointer reader;
  ASSERT_EQ(source->create_ordinal_reader(&reader), 0);
  uint64_t key = 0;
  const void *data = nullptr;
  ASSERT_EQ(reader->read(13, &key, &data), 0);
  EXPECT_EQ(static_cast<const float *>(data)[0], 13.0f);
  EXPECT_EQ(pool.used(), 0u);
}

}  // namespace
}  // namespace zvec::core
