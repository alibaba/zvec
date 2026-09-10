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
#include "utility/search_heap.h"
#include <memory>
#include <stdexcept>
#include <ailego/internal/cpu_features.h>
#include <gtest/gtest.h>

namespace zvec::core {
namespace {

using TopkHeap = SearchHeap::TopkHeap;

TEST(SearchHeap, DispatchUsesConcreteReferencesAndMoveOnlyCallbacks) {
  SearchHeap heap;
  heap.limit(8);
  auto &topk = heap.select<TopkHeap>();
  topk.emplace(7, 1.0f);
  int calls = 0;
  heap.dispatch([&, token = std::make_unique<int>(1)](auto &buffer) {
    ++calls;
    EXPECT_EQ(1, *token);
    EXPECT_TRUE((std::is_same_v<std::decay_t<decltype(buffer)>, TopkHeap>));
    EXPECT_EQ(static_cast<void *>(&topk), static_cast<void *>(&buffer));
  });
  EXPECT_EQ(1, calls);
  heap.with_topk([&](TopkHeap &buffer) { EXPECT_EQ(&topk, &buffer); });
  EXPECT_EQ(&topk, &heap.select<TopkHeap>());
  heap.clear();
  EXPECT_TRUE(topk.empty());
  EXPECT_EQ(8U, topk.limit());
}

class SearchHeapPoolTest : public testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || \
    defined(_M_IX86)
    if (GetParam() && !ailego::internal::CpuFeatures::static_flags_.AVX2) {
      GTEST_SKIP() << "BlockHeap's x86 path requires AVX2";
    }
#endif
    heap_.limit(4);
  }

  void Fill(bool ties = false) {
    heap_.dispatch_pool(GetParam(), [&](auto &pool) {
      pool.reset(4, 4);
      const uint32_t ids[] = {30, 10, 20, 40};
      const float distances[] = {3.0f, 1.0f, ties ? 1.0f : 2.0f, 4.0f};
      pool.push_block(distances, ids, 4);
    });
  }

  void CheckPool() {
    heap_.dispatch([&](auto &buffer) {
      using Heap = std::decay_t<decltype(buffer)>;
      EXPECT_EQ(GetParam(), (std::is_same_v<Heap, BlockHeap>));
      EXPECT_EQ(!GetParam(), (std::is_same_v<Heap, LinearPool<float>>));
    });
  }

  SearchHeap heap_;
};

TEST_P(SearchHeapPoolTest, OutputKeepsPoolAndReusesScratch) {
  Fill();
  const void *pool_address = nullptr;
  heap_.dispatch([&](auto &pool) { pool_address = &pool; });
  const void *scratch_data = nullptr;
  heap_.with_topk([&](TopkHeap &heap) {
    scratch_data = heap.container().data();
    // Scratch changes must not affect the active search result.
    heap.emplace(999, -1.0f);
  });
  heap_.with_topk([&](TopkHeap &heap) {
    EXPECT_EQ(scratch_data, heap.container().data());
    heap.sort();
    ASSERT_EQ(4U, heap.size());
    EXPECT_EQ(10U, heap[0].first);
    EXPECT_FLOAT_EQ(1.0f, heap[0].second);
  });
  CheckPool();
  heap_.dispatch([&](auto &pool) { EXPECT_EQ(pool_address, &pool); });

  heap_.clear();
  heap_.with_topk([](TopkHeap &heap) { EXPECT_TRUE(heap.empty()); });
  Fill();
  CheckPool();
  heap_.dispatch([&](auto &pool) { EXPECT_EQ(pool_address, &pool); });
  heap_.with_topk([&](TopkHeap &heap) {
    EXPECT_EQ(scratch_data, heap.container().data());
    EXPECT_EQ(4U, heap.size());
  });
}

TEST_P(SearchHeapPoolTest, OutputKeepsLegacyHeapOrderingAndCapacity) {
  for (bool ties : {false, true}) {
    Fill(ties);
    for (size_t limit : {1U, 2U, 4U, 8U}) {
      TopkHeap reference(limit);
      heap_.dispatch([&](auto &pool) {
        if constexpr (!std::is_same_v<std::decay_t<decltype(pool)>, TopkHeap>) {
          copy_pool_to_topk(pool, reference);
        }
      });
      reference.sort();
      heap_.limit(limit);
      CheckPool();
      heap_.with_topk([&](TopkHeap &heap) {
        EXPECT_EQ(limit, heap.limit());
        heap.sort();
        ASSERT_EQ(reference.size(), heap.size());
        for (size_t i = 0; i < heap.size(); ++i) {
          EXPECT_EQ(reference[i], heap[i]);
        }
      });
      CheckPool();
    }
  }
}

TEST_P(SearchHeapPoolTest, ExceptionsAndBackendChangesDoNotLeakResults) {
  Fill();
  EXPECT_THROW(heap_.with_topk([](TopkHeap &) {
    throw std::runtime_error("test export failure");
  }),
               std::runtime_error);
  CheckPool();
  heap_.clear();
  auto &topk = heap_.select<TopkHeap>();
  EXPECT_TRUE(topk.empty());
  EXPECT_EQ(4U, topk.limit());
  topk.emplace(77, 0.5f);
  heap_.with_topk([&](TopkHeap &heap) {
    EXPECT_EQ(&topk, &heap);
    ASSERT_EQ(1U, heap.size());
    EXPECT_EQ(77U, heap[0].first);
  });
  Fill();
  CheckPool();
  auto &materialized = heap_.materialize_topk();
  EXPECT_EQ(4U, materialized.size());
  materialized.emplace(88, -1.0f);
  heap_.with_topk([&](TopkHeap &heap) {
    EXPECT_EQ(&materialized, &heap);
    heap.sort();
    EXPECT_EQ(88U, heap[0].first);
  });
}

INSTANTIATE_TEST_SUITE_P(Backends, SearchHeapPoolTest, testing::Bool());

}  // namespace
}  // namespace zvec::core
