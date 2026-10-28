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

#include <atomic>
#include <future>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_framework.h>
#include "mixed_reducer/merged_provider_index_holder.h"
#include "tests/test_util.h"
#include "utility/stable_vector_holder.h"
#include "diskann_builder.h"
#include "diskann_builder_entity.h"

namespace zvec::core {
namespace {
class DiskAnnMmapViewTest : public testing::Test {
 protected:
  const std::string path = "diskann_mmap_view_test_data/";
  IndexMeta meta{IndexMeta::DT_FP32, 16};

  void SetUp() override {
    test_util::RemoveTestPath(path);
    meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  }
  void TearDown() override {
    test_util::RemoveTestPath(path);
  }

  IndexStreamer::Pointer source(const char *name, const char *storage_name,
                                float offset = 0) {
    auto storage = IndexFactory::CreateStorage(storage_name);
    EXPECT_NE(nullptr, storage);
    EXPECT_EQ(0, storage->init(ailego::Params()));
    EXPECT_EQ(0, storage->open(path + name, true));
    auto streamer = IndexFactory::CreateStreamer("FlatStreamer");
    EXPECT_NE(nullptr, streamer);
    EXPECT_EQ(0, streamer->init(meta, ailego::Params()));
    EXPECT_EQ(0, streamer->open(storage));
    auto context = streamer->create_context();
    // Non-contiguous keys and multiple Flat blocks exercise ordinal mapping.
    for (uint64_t i = 0; i < 70; ++i) {
      std::vector<float> data(16, offset + i);
      EXPECT_EQ(0, streamer->add_impl(i * 3, data.data(),
                                      IndexQueryMeta(IndexMeta::DT_FP32, 16),
                                      context));
    }
    return streamer;
  }

  MergedProviderIndexHolder::Source merged_source(
      IndexStreamer::Pointer owner) {
    MergedProviderIndexHolder::Source result;
    result.owner = std::move(owner);
    result.provider_meta = IndexQueryMeta(IndexMeta::DT_FP32, 16);
    return result;
  }
};

TEST_F(DiskAnnMmapViewTest, MmapPointersSurviveIteratorAndProviderRelease) {
  std::shared_ptr<const void> lifetime;
  std::vector<const void *> vectors;
  std::vector<uint64_t> keys;
  {
    auto owner = source("mmap", "MMapFileStorage");
    auto provider = owner->create_provider();
    auto *stable = dynamic_cast<StableVectorHolder *>(provider.get());
    ASSERT_NE(nullptr, stable);
    ASSERT_EQ(0, stable->retain_stable_data(&lifetime));
    ASSERT_NE(nullptr, lifetime);
    auto iter = provider->create_iterator();
    for (; iter->is_valid(); iter->next()) {
      keys.push_back(iter->key());
      vectors.push_back(iter->data());
      EXPECT_EQ(provider->get_vector(keys.back()), vectors.back());
    }
    ASSERT_EQ(0, iter->status());
    ASSERT_EQ(70U, vectors.size());
  }
  // Only the lifetime retains the mapping after the source is destroyed.
  std::vector<std::future<bool>> readers;
  for (int t = 0; t < 4; ++t) {
    readers.emplace_back(std::async(std::launch::async, [&vectors]() {
      for (size_t i = 0; i < vectors.size(); ++i) {
        const auto *v = static_cast<const float *>(vectors[i]);
        for (size_t d = 0; d < 16; ++d)
          if (v[d] != float(i)) return false;
      }
      return true;
    }));
  }
  for (auto &reader : readers) EXPECT_TRUE(reader.get());

  std::weak_ptr<const void> mapping = lifetime;
  DiskAnnBuilderEntity entity;
  ASSERT_EQ(0, entity.init(meta, 8, 10, 0, 4));
  ASSERT_EQ(0, entity.reserve_space(vectors.size(), std::move(lifetime)));
  for (size_t i = 0; i < vectors.size(); ++i) {
    EXPECT_EQ(i * 3, keys[i]);
    ASSERT_EQ(0, entity.add_vector(keys[i], vectors[i]));
    EXPECT_EQ(vectors[i], entity.get_vector(i));
  }
  EXPECT_FALSE(mapping.expired());
  entity.release_vectors();
  EXPECT_TRUE(mapping.expired());
  EXPECT_EQ(70U, entity.doc_cnt());
  EXPECT_EQ(69U * 3, entity.get_key(69));
}

TEST_F(DiskAnnMmapViewTest, MergeUsesExistingIteratorAndRetainsAllSources) {
  std::shared_ptr<const void> lifetime;
  std::vector<const float *> vectors;
  std::vector<float> values;
  {
    auto a = source("a", "MMapFileStorage");
    auto b = source("b", "MMapFileStorage", 100);
    auto holder = std::make_shared<MergedProviderIndexHolder>(
        IndexQueryMeta(IndexMeta::DT_FP32, 16),
        std::vector<MergedProviderIndexHolder::Source>{merged_source(a),
                                                       merged_source(b)});
    size_t calls = 0;
    IndexFilter filter;
    filter.set([&calls](uint64_t id) {
      ++calls;
      return id % 2 == 0;
    });
    ASSERT_EQ(0, holder->init(filter));
    EXPECT_EQ(140U, calls);
    ASSERT_EQ(0, holder->retain_stable_data(&lifetime));
    ASSERT_NE(nullptr, lifetime);
    EXPECT_EQ(140U, calls);
    auto iter = holder->create_iterator();
    for (; iter->is_valid(); iter->next()) {
      EXPECT_EQ(vectors.size(), iter->key());
      auto *data = static_cast<const float *>(iter->data());
      ASSERT_NE(nullptr, data);
      vectors.push_back(data);
      values.push_back(*data);
    }
    EXPECT_EQ(0, iter->status());
    EXPECT_EQ(holder->count(), vectors.size());
    EXPECT_EQ(140U, calls);
  }
  // Crossing sources releases providers; the combined lifetime retains both.
  ASSERT_FALSE(vectors.empty());
  for (size_t i = 0; i < vectors.size(); ++i) {
    for (size_t d = 0; d < 16; ++d) EXPECT_EQ(values[i], vectors[i][d]);
  }
}

TEST_F(DiskAnnMmapViewTest, MixedStorageFallsBackWithoutChangingLifetime) {
  ailego::MemoryLimitPool::get_instance().init(64 * 1024UL * 1024UL);
  auto a = source("mmap", "MMapFileStorage");
  auto b = source("buffer", "BufferStorage");
  auto holder = std::make_shared<MergedProviderIndexHolder>(
      IndexQueryMeta(IndexMeta::DT_FP32, 16),
      std::vector<MergedProviderIndexHolder::Source>{merged_source(a),
                                                     merged_source(b)});
  ASSERT_EQ(0, holder->init(IndexFilter()));
  auto sentinel = std::make_shared<int>(999);
  std::shared_ptr<const void> lifetime = sentinel;
  ASSERT_EQ(IndexError_NotImplemented, holder->retain_stable_data(&lifetime));
  EXPECT_EQ(sentinel, lifetime);
  EXPECT_EQ(0, holder->status());
  auto iter = holder->create_iterator();
  size_t count = 0;
  for (; iter->is_valid(); iter->next()) {
    ASSERT_NE(nullptr, iter->data());
    ++count;
  }
  EXPECT_EQ(0, iter->status());
  EXPECT_EQ(140U, count);
}

TEST_F(DiskAnnMmapViewTest, CancellationDoesNotChangeLifetime) {
  auto holder = std::make_shared<MergedProviderIndexHolder>(
      IndexQueryMeta(IndexMeta::DT_FP32, 16),
      std::vector<MergedProviderIndexHolder::Source>{
          merged_source(source("cancel", "MMapFileStorage"))});
  std::atomic<bool> stop{false};
  ASSERT_EQ(0, holder->init(IndexFilter(), &stop));
  stop = true;
  auto sentinel = std::make_shared<int>(999);
  std::shared_ptr<const void> lifetime = sentinel;
  EXPECT_EQ(IndexError_Canceled, holder->retain_stable_data(&lifetime));
  EXPECT_EQ(sentinel, lifetime);
  EXPECT_EQ(IndexError_Canceled, holder->status());
}

TEST_F(DiskAnnMmapViewTest, BuilderCopiesMixedStorageForGraphAndDumps) {
  ailego::MemoryLimitPool::get_instance().init(64 * 1024UL * 1024UL);
  auto a = source("a", "MMapFileStorage");
  auto b = source("b", "BufferStorage", 100);
  auto holder = std::make_shared<MergedProviderIndexHolder>(
      IndexQueryMeta(IndexMeta::DT_FP32, 16),
      std::vector<MergedProviderIndexHolder::Source>{merged_source(a),
                                                     merged_source(b)});
  ASSERT_EQ(0, holder->init(IndexFilter()));
  auto builder = IndexFactory::CreateBuilder("DiskAnnBuilder");
  ailego::Params params;
  params.set("zvec.diskann.builder.threads", 2U);
  params.set("zvec.diskann.builder.pq_chunk_num", 4U);
  ASSERT_EQ(0, builder->init(meta, params));
  ASSERT_EQ(0, builder->train(holder));
  ASSERT_EQ(0, builder->build(holder));
  EXPECT_EQ(140U, builder->stats().built_count());
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_EQ(0, dumper->create(path + "copied.index"));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());
  EXPECT_EQ(140U, builder->stats().dumped_count());
  ASSERT_EQ(0, builder->cleanup());
}

class FailedStableHolder : public MultiPassIndexHolder<IndexMeta::DT_FP32>,
                           public StableVectorHolder {
 public:
  FailedStableHolder() : MultiPassIndexHolder<IndexMeta::DT_FP32>(16) {
    for (size_t i = 0; i < 64; ++i) {
      emplace(i, ailego::NumericalVector<float>(16, float(i)));
    }
  }
  int retain_stable_data(std::shared_ptr<const void> *) override {
    return IndexError_ReadData;
  }
};

TEST_F(DiskAnnMmapViewTest, BuilderPropagatesStableReadError) {
  auto holder = std::make_shared<FailedStableHolder>();
  auto builder = IndexFactory::CreateBuilder("DiskAnnBuilder");
  ailego::Params params;
  params.set("zvec.diskann.builder.threads", 1U);
  params.set("zvec.diskann.builder.pq_chunk_num", 4U);
  ASSERT_EQ(0, builder->init(meta, params));
  ASSERT_EQ(0, builder->train(holder));
  EXPECT_EQ(IndexError_ReadData, builder->build(holder));
  EXPECT_EQ(0U, builder->stats().built_count());
}
// Acquiring a lifetime must not hide failures from the existing iterator.
class IteratorReadErrorHolder : public FailedStableHolder {
 public:
  class FailedIterator : public IndexHolder::Iterator {
   public:
    const void *data() const override {
      status_ = IndexError_ReadData;
      return nullptr;
    }
    bool is_valid() const override {
      return status_ == 0;
    }
    uint64_t key() const override {
      return 0;
    }
    void next() override {
      status_ = IndexError_ReadData;
    }
    int status() const override {
      return status_;
    }

   private:
    mutable int status_{0};
  };
  int retain_stable_data(std::shared_ptr<const void> *lifetime) override {
    *lifetime = std::make_shared<int>(0);
    return 0;
  }
  IndexHolder::Iterator::Pointer create_iterator() override {
    if (fail_reads) return std::make_unique<FailedIterator>();
    return FailedStableHolder::create_iterator();
  }
  bool fail_reads{false};
};

TEST_F(DiskAnnMmapViewTest, BuilderPropagatesIteratorReadErrorAfterRetain) {
  auto holder = std::make_shared<IteratorReadErrorHolder>();
  auto builder = IndexFactory::CreateBuilder("DiskAnnBuilder");
  ailego::Params params;
  params.set("zvec.diskann.builder.threads", 1U);
  params.set("zvec.diskann.builder.pq_chunk_num", 4U);
  ASSERT_EQ(0, builder->init(meta, params));
  ASSERT_EQ(0, builder->train(holder));
  holder->fail_reads = true;
  EXPECT_EQ(IndexError_ReadData, builder->build(holder));
  EXPECT_EQ(0U, builder->stats().built_count());
}
}  // namespace
}  // namespace zvec::core
