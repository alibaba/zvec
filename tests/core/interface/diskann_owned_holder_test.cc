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
#include <cstring>
#include <future>
#include <limits>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_framework.h>
#include "algorithm/diskann/diskann_builder_entity.h"
#include "interface/indexes/holder_builder.h"
#include "utility/owned_vector_holder.h"

#if DISKANN_SUPPORTED
namespace zvec::core_interface {
namespace {

TEST(DiskAnnOwnedHolder, OwnedVectorsSurviveCacheHolderAndIteratorRelease) {
  constexpr size_t kDimension = 16;
  std::vector<std::pair<uint64_t, std::string>> cache;
  for (uint64_t key : {0U, 6U, 42U}) {
    std::vector<float> data(kDimension, static_cast<float>(key));
    cache.emplace_back(key,
                       std::string(reinterpret_cast<const char *>(data.data()),
                                   data.size() * sizeof(float)));
  }
  cache.emplace_back(kInvalidKey, std::string{});
  core::IndexHolder::Pointer holder;
  ASSERT_EQ(0, BuildMultiPassHolder(DataType::DT_FP32, kDimension, cache,
                                    nullptr, &holder));
  ASSERT_EQ(3U, holder->count());
  auto *stable = dynamic_cast<core::StableVectorHolder *>(holder.get());
  ASSERT_NE(nullptr, stable);
  std::shared_ptr<const void> lifetime;
  ASSERT_EQ(core::IndexError_InvalidArgument,
            stable->retain_stable_data(nullptr));
  ASSERT_EQ(0, stable->retain_stable_data(&lifetime));
  ASSERT_NE(nullptr, lifetime);
  std::weak_ptr<core::IndexHolder> owner = holder;
  auto iter = holder->create_iterator();
  cache.clear();
  holder.reset();
  ASSERT_FALSE(owner.expired());
  std::vector<uint64_t> keys;
  std::vector<const void *> pointers;
  for (; iter->is_valid(); iter->next()) {
    keys.push_back(iter->key());
    pointers.push_back(iter->data());
  }
  ASSERT_EQ(0, iter->status());
  ASSERT_EQ((std::vector<uint64_t>{0U, 6U, 42U}), keys);
  iter.reset();
  std::vector<std::future<bool>> readers;
  for (int thread = 0; thread < 4; ++thread) {
    readers.emplace_back(std::async(
        std::launch::async, [&keys, &pointers, dimension = kDimension]() {
          for (size_t i = 0; i < pointers.size(); ++i) {
            auto *data = static_cast<const float *>(pointers[i]);
            for (size_t d = 0; d < dimension; ++d) {
              if (data[d] != static_cast<float>(keys[i])) return false;
            }
          }
          return true;
        }));
  }
  for (auto &reader : readers) EXPECT_TRUE(reader.get());

  core::IndexMeta meta(core::IndexMeta::DT_FP32, kDimension);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  core::DiskAnnBuilderEntity entity;
  ASSERT_EQ(0, entity.init(meta, 8, 10, 0, 4));
  ASSERT_EQ(0, entity.reserve_space(keys.size(), std::move(lifetime)));
  for (size_t i = 0; i < keys.size(); ++i) {
    ASSERT_EQ(0, entity.add_vector(keys[i], pointers[i]));
    EXPECT_EQ(pointers[i], entity.get_vector(i));
  }
  EXPECT_FALSE(owner.expired());
  entity.release_vectors();
  EXPECT_TRUE(owner.expired());
}

TEST(DiskAnnOwnedHolder, SupportsOwnedNumericalTypes) {
  for (DataType type :
       {DataType::DT_FP32, DataType::DT_FP16, DataType::DT_INT8}) {
    SCOPED_TRACE(static_cast<int>(type));
    const size_t bytes = 16 * core::IndexMeta::UnitSizeof(type);
    std::vector<std::pair<uint64_t, std::string>> cache{
        {5U, std::string(bytes, '\x01')}};
    core::IndexHolder::Pointer holder;
    ASSERT_EQ(0, BuildMultiPassHolder(type, 16, cache, nullptr, &holder));
    auto *stable = dynamic_cast<core::StableVectorHolder *>(holder.get());
    ASSERT_NE(nullptr, stable);
    std::shared_ptr<const void> lifetime;
    ASSERT_EQ(0, stable->retain_stable_data(&lifetime));
    auto iter = holder->create_iterator();
    ASSERT_TRUE(iter->is_valid());
    EXPECT_EQ(5U, iter->key());
    EXPECT_EQ(0, std::memcmp(iter->data(), cache[0].second.data(), bytes));
  }
}

TEST(DiskAnnOwnedHolder, MissingSharedOwnerLeavesLifetimeUnchanged) {
  StableMultiPassIndexHolder<DataType::DT_FP32> holder(16);
  std::shared_ptr<const void> lifetime = std::make_shared<int>(7);
  auto previous = lifetime;
  EXPECT_EQ(core::IndexError_NoReady, holder.retain_stable_data(&lifetime));
  EXPECT_EQ(previous, lifetime);
}

TEST(DiskAnnOwnedHolder, LazyConversionDoesNotAdvertiseStableVectors) {
  core::IndexMeta meta(core::IndexMeta::DT_FP32, 16);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  auto converter = core::IndexFactory::CreateConverter("HalfFloatConverter");
  ASSERT_NE(nullptr, converter);
  ASSERT_EQ(0, converter->init(meta, ailego::Params()));
  std::vector<float> vector(16, 1.0F);
  std::vector<std::pair<uint64_t, std::string>> cache{
      {3U, std::string(reinterpret_cast<const char *>(vector.data()),
                       vector.size() * sizeof(float))}};
  core::IndexHolder::Pointer holder;
  ASSERT_EQ(0, BuildMultiPassHolder(DataType::DT_FP32, 16, cache, converter,
                                    &holder));
  EXPECT_EQ(nullptr, dynamic_cast<core::StableVectorHolder *>(holder.get()));
  EXPECT_EQ(DataType::DT_FP16, holder->data_type());
  EXPECT_EQ(1U, holder->count());

  core::OwnedVectorHolder::Pointer snapshot;
  ASSERT_EQ(0, core::OwnedVectorHolder::CopyFrom(holder, &snapshot));
  holder.reset();
  converter.reset();
  auto iter = snapshot->create_iterator();
  ASSERT_TRUE(iter->is_valid());
  EXPECT_EQ(3U, iter->key());
  auto *data = static_cast<const uint16_t *>(iter->data());
  for (size_t d = 0; d < 16; ++d) EXPECT_EQ(0x3c00U, data[d]);
}

// Models providers and decoders that reuse one temporary record buffer.
class ScratchVectorHolder : public core::IndexHolder {
 public:
  enum class Fault { kNone, kKey, kData, kNext };
  class Iterator : public core::IndexHolder::Iterator {
   public:
    Iterator(size_t count, Fault fault) : count_(count), fault_(fault) {}
    bool is_valid() const override {
      return position_ < count_ && status_ == 0;
    }
    uint64_t key() const override {
      if (fault_ == Fault::kKey && position_ == count_ - 1) {
        status_ = core::IndexError_ReadData;
      }
      return position_ * 7 + 3;
    }
    const void *data() const override {
      std::fill(buffer_.begin(), buffer_.end(),
                static_cast<float>(position_ + 1));
      if (fault_ == Fault::kData && position_ == count_ - 1) {
        status_ = core::IndexError_ReadData;
      }
      return buffer_.data();
    }
    void next() override {
      std::fill(buffer_.begin(), buffer_.end(), -1.0F);
      ++position_;
      if (fault_ == Fault::kNext && position_ == count_) {
        status_ = core::IndexError_ReadData;
      }
    }
    int status() const override {
      return status_;
    }

   private:
    size_t count_;
    Fault fault_;
    size_t position_{0};
    mutable int status_{0};
    mutable std::vector<float> buffer_ = std::vector<float>(16);
  };
  size_t count() const override {
    return declared_count;
  }
  size_t dimension() const override {
    return 16;
  }
  DataType data_type() const override {
    return DataType::DT_FP32;
  }
  size_t element_size() const override {
    return 16 * sizeof(float);
  }
  bool multipass() const override {
    return true;
  }
  core::IndexHolder::Iterator::Pointer create_iterator() override {
    return std::make_unique<Iterator>(3, fault);
  }
  size_t declared_count{3};
  Fault fault{Fault::kNone};
};

TEST(DiskAnnOwnedHolder, FallbackCopiesScratchAndPinsContiguousStorage) {
  auto source = std::make_shared<ScratchVectorHolder>();
  std::weak_ptr<ScratchVectorHolder> source_owner = source;
  core::OwnedVectorHolder::Pointer snapshot;
  ASSERT_EQ(0, core::OwnedVectorHolder::CopyFrom(source, &snapshot));
  source.reset();
  EXPECT_TRUE(source_owner.expired());
  ASSERT_EQ(3U, snapshot->count());
  std::shared_ptr<const void> lifetime;
  ASSERT_EQ(0, snapshot->retain_stable_data(&lifetime));
  std::weak_ptr<const void> storage = lifetime;
  auto iter = snapshot->create_iterator();
  snapshot.reset();
  std::vector<const float *> pointers;
  for (uint64_t i = 0; i < 3; ++i, iter->next()) {
    ASSERT_TRUE(iter->is_valid());
    EXPECT_EQ(i * 7 + 3, iter->key());
    pointers.push_back(static_cast<const float *>(iter->data()));
  }
  EXPECT_FALSE(iter->is_valid());
  iter.reset();
  for (size_t i = 0; i < pointers.size(); ++i) {
    EXPECT_EQ(pointers[0] + i * 16, pointers[i]);
    for (size_t d = 0; d < 16; ++d) EXPECT_EQ(float(i + 1), pointers[i][d]);
  }
  lifetime.reset();
  EXPECT_TRUE(storage.expired());
}

TEST(DiskAnnOwnedHolder, FailedSnapshotPreservesOutputAndIteratorErrors) {
  auto source = std::make_shared<ScratchVectorHolder>();
  core::OwnedVectorHolder::Pointer snapshot;
  ASSERT_EQ(0, core::OwnedVectorHolder::CopyFrom(source, &snapshot));
  auto previous = snapshot;
  for (auto fault :
       {ScratchVectorHolder::Fault::kKey, ScratchVectorHolder::Fault::kData,
        ScratchVectorHolder::Fault::kNext}) {
    source->fault = fault;
    EXPECT_EQ(core::IndexError_ReadData,
              core::OwnedVectorHolder::CopyFrom(source, &snapshot));
    EXPECT_EQ(previous, snapshot);
  }
  source->fault = ScratchVectorHolder::Fault::kNone;
  for (size_t count : {2U, 4U}) {
    source->declared_count = count;
    EXPECT_EQ(core::IndexError_Mismatch,
              core::OwnedVectorHolder::CopyFrom(source, &snapshot));
    EXPECT_EQ(previous, snapshot);
  }
  source->declared_count = std::numeric_limits<size_t>::max();
  EXPECT_EQ(core::IndexError_NoMemory,
            core::OwnedVectorHolder::CopyFrom(source, &snapshot));
  EXPECT_EQ(previous, snapshot);
  EXPECT_EQ(core::IndexError_InvalidArgument,
            core::OwnedVectorHolder::CopyFrom(nullptr, &snapshot));
  EXPECT_EQ(previous, snapshot);
  EXPECT_EQ(core::IndexError_InvalidArgument,
            core::OwnedVectorHolder::CopyFrom(source, nullptr));
}

TEST(DiskAnnOwnedHolder, EntityRequiresPinnedVectorStorage) {
  core::IndexMeta meta(core::IndexMeta::DT_FP32, 16);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  core::DiskAnnBuilderEntity entity;
  ASSERT_EQ(0, entity.init(meta, 8, 10, 0, 4));
  auto data = std::make_shared<std::vector<float>>(16, 1.0F);
  EXPECT_EQ(core::IndexError_InvalidArgument, entity.reserve_space(1, {}));
  EXPECT_EQ(core::IndexError_NoReady, entity.add_vector(0, data->data()));
  ASSERT_EQ(0, entity.reserve_space(1, data));
  ASSERT_EQ(0, entity.add_vector(0, data->data()));
  EXPECT_EQ(data->data(), entity.get_vector(0));
  entity.release_vectors();
  EXPECT_EQ(core::IndexError_NoReady, entity.add_vector(1, data->data()));
}

}  // namespace
}  // namespace zvec::core_interface
#endif
