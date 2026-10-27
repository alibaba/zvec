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
#include "diskann_builder.h"
#include "diskann_builder_entity.h"
#include "mixed_reducer/merged_provider_index_holder.h"
#include "tests/test_util.h"
#include "utility/stable_vector_holder.h"

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
  void TearDown() override { test_util::RemoveTestPath(path); }

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

  MergedProviderIndexHolder::Source merged_source(IndexStreamer::Pointer owner) {
    MergedProviderIndexHolder::Source result;
    result.owner = std::move(owner);
    result.provider_meta = IndexQueryMeta(IndexMeta::DT_FP32, 16);
    return result;
  }
};

TEST_F(DiskAnnMmapViewTest, MmapPointersSurviveIteratorAndProviderRelease) {
  StableVectorView view;
  {
    auto owner = source("mmap", "MMapFileStorage");
    auto provider = owner->create_provider();
    auto *stable = dynamic_cast<StableVectorHolder *>(provider.get());
    ASSERT_NE(nullptr, stable);
    ASSERT_EQ(0, stable->get_stable_vectors(&view));
    ASSERT_EQ(70U, view.vectors.size());
    for (size_t i = 0; i < view.vectors.size(); ++i) {
      EXPECT_EQ(i * 3, view.keys[i]);
      EXPECT_EQ(provider->get_vector(i * 3), view.vectors[i]);
    }
  }
  // The view owns the mapping even after the source streamer is destroyed.
  std::vector<std::future<bool>> readers;
  for (int t = 0; t < 4; ++t) {
    readers.emplace_back(std::async(std::launch::async, [&view]() {
      for (size_t i = 0; i < view.vectors.size(); ++i) {
        const auto *v = static_cast<const float *>(view.vectors[i]);
        for (size_t d = 0; d < 16; ++d) if (v[d] != float(i)) return false;
      }
      return true;
    }));
  }
  for (auto &reader : readers) EXPECT_TRUE(reader.get());

  std::weak_ptr<const void> mapping = view.owners.front();
  const void *first = view.vectors.front();
  DiskAnnBuilderEntity entity;
  ASSERT_EQ(0, entity.init(meta, 8, 10, 0, 4));
  ASSERT_EQ(0, entity.use_vector_view(std::move(view)));
  EXPECT_EQ(first, entity.get_vector(0));
  EXPECT_EQ(69U * 3, entity.get_key(69));
  EXPECT_FALSE(mapping.expired());
  entity.release_vectors();
  EXPECT_TRUE(mapping.expired());
  EXPECT_EQ(70U, entity.doc_cnt());
  EXPECT_EQ(69U * 3, entity.get_key(69));
}

TEST_F(DiskAnnMmapViewTest, MergeUsesCachedFiltersAndDenseIds) {
  auto a = source("a", "MMapFileStorage");
  auto b = source("b", "MMapFileStorage", 100);
  auto holder = std::make_shared<MergedProviderIndexHolder>(
      IndexQueryMeta(IndexMeta::DT_FP32, 16),
      std::vector<MergedProviderIndexHolder::Source>{merged_source(a),
                                                    merged_source(b)});
  size_t calls = 0;
  IndexFilter filter;
  filter.set([&calls](uint64_t id) { ++calls; return id % 2 == 0; });
  ASSERT_EQ(0, holder->init(filter));
  EXPECT_EQ(140U, calls);
  StableVectorView view;
  ASSERT_EQ(0, holder->get_stable_vectors(&view));
  EXPECT_EQ(140U, calls);
  ASSERT_EQ(holder->count(), view.vectors.size());
  auto iter = holder->create_iterator();
  size_t ordinal = 0;
  for (; iter->is_valid(); iter->next(), ++ordinal) {
    ASSERT_LT(ordinal, view.vectors.size());
    EXPECT_EQ(iter->key(), view.keys[ordinal]);
    EXPECT_EQ(ordinal, view.keys[ordinal]);
    EXPECT_EQ(iter->data(), view.vectors[ordinal]);
  }
  EXPECT_EQ(view.vectors.size(), ordinal);
}

TEST_F(DiskAnnMmapViewTest, MixedStorageFallsBackWithoutPartialView) {
  ailego::MemoryLimitPool::get_instance().init(64 * 1024UL * 1024UL);
  auto a = source("mmap", "MMapFileStorage");
  auto b = source("buffer", "BufferStorage");
  auto holder = std::make_shared<MergedProviderIndexHolder>(
      IndexQueryMeta(IndexMeta::DT_FP32, 16),
      std::vector<MergedProviderIndexHolder::Source>{merged_source(a),
                                                    merged_source(b)});
  ASSERT_EQ(0, holder->init(IndexFilter()));
  StableVectorView view;
  view.keys = {999};
  ASSERT_EQ(IndexError_NotImplemented, holder->get_stable_vectors(&view));
  EXPECT_EQ(std::vector<uint64_t>{999}, view.keys);
  EXPECT_TRUE(view.vectors.empty());
  EXPECT_TRUE(view.owners.empty());
  EXPECT_EQ(0, holder->status());
  auto iter = holder->create_iterator();
  size_t count = 0;
  for (; iter->is_valid(); iter->next()) {
    ASSERT_NE(nullptr, iter->data());
    ++count;
  }
  EXPECT_EQ(140U, count);
}

TEST_F(DiskAnnMmapViewTest, CancellationDoesNotPublishPartialView) {
  auto holder = std::make_shared<MergedProviderIndexHolder>(
      IndexQueryMeta(IndexMeta::DT_FP32, 16),
      std::vector<MergedProviderIndexHolder::Source>{
          merged_source(source("cancel", "MMapFileStorage"))});
  std::atomic<bool> stop{false};
  ASSERT_EQ(0, holder->init(IndexFilter(), &stop));
  stop = true;
  StableVectorView view;
  EXPECT_EQ(IndexError_Canceled, holder->get_stable_vectors(&view));
  EXPECT_TRUE(view.vectors.empty());
  EXPECT_EQ(IndexError_Canceled, holder->status());
}

class FailedStableHolder : public MultiPassIndexHolder<IndexMeta::DT_FP32>,
                           public StableVectorHolder {
 public:
  FailedStableHolder() : MultiPassIndexHolder<IndexMeta::DT_FP32>(16) {
    for (size_t i = 0; i < 64; ++i) {
      emplace(i, ailego::NumericalVector<float>(16, float(i)));
    }
  }
  int get_stable_vectors(StableVectorView *) override {
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
}  // namespace
}  // namespace zvec::core
