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

#include <filesystem>
#include <gtest/gtest.h>
#include "algorithm/flat_sparse/flat_sparse_streamer.h"
#include "mixed_reducer/mixed_reducer_params.h"
#include "mixed_reducer/mixed_streamer_reducer.h"

namespace zvec::core {
namespace {

class UnreadableSparseProvider : public IndexSparseProvider {
 public:
  size_t count() const override {
    return 1;
  }
  size_t total_sparse_count() const override {
    return 1;
  }
  IndexMeta::DataType data_type() const override {
    return IndexMeta::DT_FP32;
  }
  const std::string &owner_class() const override {
    static const std::string name = "UnreadableSparseProvider";
    return name;
  }
  int get_sparse_vector(uint64_t, uint32_t *, std::string *,
                        std::string *) const override {
    return IndexError_ReadData;
  }
  Iterator::Pointer create_iterator() override {
    return std::make_unique<FailedIterator>();
  }

 private:
  class FailedIterator : public IndexSparseHolder::Iterator {
   public:
    uint32_t sparse_count() const override {
      return 0;
    }
    const uint32_t *sparse_indices() const override {
      return nullptr;
    }
    const void *sparse_data() const override {
      return nullptr;
    }
    bool is_valid() const override {
      return false;
    }
    int status() const override {
      return IndexError_ReadData;
    }
    uint64_t key() const override {
      return UINT64_MAX;
    }
    void next() override {}
  };
};

class FailingSparseStreamer : public FlatSparseStreamer {
 public:
  explicit FailingSparseStreamer(bool missing) : missing_(missing) {}
  SparseProvider::Pointer create_sparse_provider() const override {
    ++provider_calls;
    return missing_ ? nullptr : std::make_shared<UnreadableSparseProvider>();
  }
  mutable size_t provider_calls{0};

 private:
  bool missing_;
};

TEST(SparseMergeReadErrorTest, StopsAtFirstFailureAndCreatesEachProviderOnce) {
  const std::string path = "test_sparse_merge_read_error";
  IndexMeta meta(IndexMeta::MT_SPARSE, IndexMeta::DT_FP32);
  meta.set_metric("InnerProductSparse", 0, ailego::Params());
  const IndexQueryMeta qmeta(IndexMeta::MT_SPARSE, IndexMeta::DT_FP32);
  for (bool missing : {false, true}) {
    SCOPED_TRACE(missing);
    std::filesystem::remove_all(path);
    auto target = IndexFactory::CreateStreamer("FlatSparseStreamer");
    auto storage = IndexFactory::CreateStorage("MMapFileStorage");
    ASSERT_EQ(0, target->init(meta, ailego::Params()));
    ASSERT_EQ(0, storage->init(ailego::Params()));
    ASSERT_EQ(0, storage->open(path, true));
    ASSERT_EQ(0, target->open(storage));
    auto first = std::make_shared<FailingSparseStreamer>(missing);
    auto later = std::make_shared<FailingSparseStreamer>(false);
    ASSERT_EQ(0, first->init(meta, ailego::Params()));
    ASSERT_EQ(0, later->init(meta, ailego::Params()));
    ailego::ThreadPool pool(2, false);
    MixedStreamerReducer reducer;
    ailego::Params params;
    params.set(PARAM_MIXED_STREAMER_REDUCER_NUM_OF_ADD_THREADS, 1);
    ASSERT_EQ(0, reducer.init(params));
    reducer.set_thread_pool(&pool);
    ASSERT_EQ(0, reducer.set_target_streamer_wiht_info(nullptr, target, nullptr,
                                                       nullptr, qmeta));
    ASSERT_EQ(0, reducer.feed_streamer_with_reformer(first, nullptr));
    ASSERT_EQ(0, reducer.feed_streamer_with_reformer(later, nullptr));
    EXPECT_EQ(missing ? IndexError_Runtime : IndexError_ReadData,
              reducer.reduce(IndexFilter()));
    EXPECT_EQ(1U, first->provider_calls);
    EXPECT_EQ(0U, later->provider_calls);
    EXPECT_EQ(0U, target->create_sparse_provider()->slot_count());
    ASSERT_EQ(0, target->close());
    ASSERT_EQ(0, storage->close());
    std::filesystem::remove_all(path);
  }
}

}  // namespace
}  // namespace zvec::core
