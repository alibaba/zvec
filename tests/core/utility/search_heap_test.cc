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

using Candidates = std::vector<std::pair<uint32_t, float>>;

Candidates Export(SearchHeap &heap, size_t count = 100) {
  Candidates result;
  heap.for_each_sorted(count, [&](uint32_t id, float score) {
    result.emplace_back(id, score);
    return true;
  });
  return result;
}

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
  EXPECT_EQ((Candidates{{7, 1.0f}}), Export(heap));
  EXPECT_EQ(&topk, &heap.select<TopkHeap>());
  heap.clear();
  EXPECT_TRUE(topk.empty());
  EXPECT_EQ(8U, topk.limit());
}

TEST(SearchHeap, PrepareTopkClearsContentsAndReusesStorage) {
  SearchHeap heap;
  auto &topk = heap.reset<TopkHeap>(8);
  topk.emplace(7, 1.0f);
  const void *storage = topk.container().data();
  auto &prepared = heap.reset<TopkHeap>(4);
  EXPECT_EQ(&topk, &prepared);
  EXPECT_EQ(storage, prepared.container().data());
  EXPECT_TRUE(prepared.empty());
  EXPECT_EQ(4U, prepared.limit());
  heap.dispatch([&](auto &buffer) {
    EXPECT_TRUE((std::is_same_v<std::decay_t<decltype(buffer)>, TopkHeap>));
    EXPECT_EQ(static_cast<void *>(&prepared), static_cast<void *>(&buffer));
  });
  heap.reset<TopkHeap>(0);
  EXPECT_EQ(1U, topk.limit());
}

TEST(SearchHeap, TopkExportSortsItsOwnStorageWithoutReplacingIt) {
  SearchHeap heap;
  auto &topk = heap.reset<TopkHeap>(4);
  topk.emplace(30, 3.0f);
  topk.emplace(10, 1.0f);
  topk.emplace(20, 2.0f);
  const void *storage = topk.container().data();
  EXPECT_EQ((Candidates{{10, 1.0f}, {20, 2.0f}}), Export(heap, 2));
  EXPECT_EQ(storage, topk.container().data());
  EXPECT_EQ(&topk, &heap.topk());
  EXPECT_EQ(3U, heap.size());
  EXPECT_EQ(10U, topk[0].first);
  EXPECT_EQ(20U, topk[1].first);
  EXPECT_EQ(30U, topk[2].first);
}

TEST(SearchHeap, AutoPoolSelectsCpuBackendAndResetsState) {
  SearchHeap heap;
  const bool use_block = ailego::internal::CpuFeatures::static_flags_.AVX2;
  for (size_t capacity : {4U, 2U, 8U}) {
    auto &topk = heap.reset<TopkHeap>(capacity);
    EXPECT_TRUE(topk.empty());
    topk.emplace(77, 0.5f);
    for (int query = 0; query < 2; ++query) {
      heap.reset_pool(capacity, 4);
      heap.dispatch([&](auto &pool) {
        using Heap = std::decay_t<decltype(pool)>;
        EXPECT_EQ(use_block, (std::is_same_v<Heap, BlockHeap>));
        EXPECT_EQ(!use_block, (std::is_same_v<Heap, LinearPool<float>>));
        EXPECT_EQ(0U, pool.size());
        if constexpr (!std::is_same_v<Heap, TopkHeap>) {
          EXPECT_FALSE(pool.has_next());
          const uint32_t ids[] = {30, 10, 20, 40};
          const float distances[] = {3.0f, 1.0f, 2.0f, 4.0f};
          pool.push_block(distances, ids, 4);
          ASSERT_TRUE(pool.has_next());
          EXPECT_EQ(10U, pool.pop());
        }
      });
      const auto result = Export(heap);
      ASSERT_EQ(std::min(capacity, size_t{4}), result.size());
      EXPECT_EQ(10U, result[0].first);
    }
  }
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

  void ResetPool(size_t capacity) {
    if (GetParam()) {
      heap_.reset<BlockHeap>(capacity, 4);
    } else {
      heap_.reset<LinearPool<float>>(capacity, 4);
    }
  }

  void Fill(bool ties = false) {
    ResetPool(4);
    heap_.dispatch([&](auto &pool) {
      if constexpr (!std::is_same_v<std::decay_t<decltype(pool)>, TopkHeap>) {
        const uint32_t ids[] = {30, 10, 20, 40};
        const float distances[] = {3.0f, 1.0f, ties ? 1.0f : 2.0f, 4.0f};
        pool.push_block(distances, ids, 4);
      } else {
        ADD_FAILURE() << "Expected a prepared search pool";
      }
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

TEST_P(SearchHeapPoolTest, PreparationResetsStateBeforeDispatch) {
  // Exercise Topk -> pool -> same pool -> Topk on a reused owner, without a
  // separate clear or selection in the execution phase.
  for (size_t capacity : {4U, 2U, 8U}) {
    auto &topk = heap_.reset<TopkHeap>(capacity);
    EXPECT_TRUE(topk.empty());
    EXPECT_EQ(capacity, topk.limit());
    topk.emplace(77, 0.5f);
    for (int query = 0; query < 2; ++query) {
      ResetPool(capacity);
      CheckPool();
      heap_.dispatch([&](auto &pool) {
        EXPECT_EQ(0U, pool.size());
        if constexpr (!std::is_same_v<std::decay_t<decltype(pool)>, TopkHeap>) {
          EXPECT_FALSE(pool.has_next());
          const uint32_t ids[] = {30, 10, 20, 40};
          const float distances[] = {3.0f, 1.0f, 2.0f, 4.0f};
          pool.push_block(distances, ids, 4);
          ASSERT_TRUE(pool.has_next());
          EXPECT_EQ(10U, pool.pop());
        }
      });
      const auto result = Export(heap_);
      ASSERT_EQ(std::min(capacity, size_t{4}), result.size());
      EXPECT_EQ(10U, result[0].first);
    }
  }
}

TEST_P(SearchHeapPoolTest, OutputReadsPoolWithoutChangingObjectOrCursor) {
  Fill();
  const void *pool_address = nullptr;
  heap_.dispatch([&](auto &pool) {
    pool_address = &pool;
    if constexpr (!std::is_same_v<std::decay_t<decltype(pool)>, TopkHeap>) {
      EXPECT_EQ(10U, static_cast<uint32_t>(pool.pop()));
    }
  });
  const Candidates expected{{10, 1.0f}, {20, 2.0f}, {30, 3.0f}, {40, 4.0f}};
  EXPECT_EQ(expected, Export(heap_));
  EXPECT_EQ(expected, Export(heap_));
  CheckPool();
  heap_.dispatch([&](auto &pool) {
    EXPECT_EQ(pool_address, &pool);
    if constexpr (!std::is_same_v<std::decay_t<decltype(pool)>, TopkHeap>) {
      EXPECT_EQ(20U, static_cast<uint32_t>(pool.pop()));
    }
  });

  heap_.clear();
  EXPECT_TRUE(Export(heap_).empty());
  Fill();
  CheckPool();
  EXPECT_EQ(expected, Export(heap_));
}

TEST_P(SearchHeapPoolTest, OutputKeepsPoolTieOrderAndHonorsLimits) {
  for (bool ties : {false, true}) {
    Fill(ties);
    Candidates retained;
    heap_.dispatch([&](auto &pool) {
      if constexpr (!std::is_same_v<std::decay_t<decltype(pool)>, TopkHeap>) {
        for (int32_t i = 0; i < pool.size(); ++i) {
          retained.emplace_back(pool.id(i), pool.dist(i));
        }
      }
    });
    for (size_t limit : {1U, 2U, 4U, 8U}) {
      heap_.limit(limit);
      for (size_t count : {0U, 1U, 3U, 8U}) {
        const size_t n = std::min({count, limit, retained.size()});
        EXPECT_EQ(Candidates(retained.begin(), retained.begin() + n),
                  Export(heap_, count));
      }
      CheckPool();
    }
  }
}

TEST_P(SearchHeapPoolTest, ExceptionsAndBackendChangesDoNotLeakResults) {
  Fill();
  EXPECT_THROW(
      heap_.for_each_sorted(4,
                            [](uint32_t, float) -> bool {
                              throw std::runtime_error("test export failure");
                            }),
      std::runtime_error);
  CheckPool();
  heap_.clear();
  auto &topk = heap_.select<TopkHeap>();
  EXPECT_TRUE(topk.empty());
  EXPECT_EQ(4U, topk.limit());
  topk.emplace(77, 0.5f);
  EXPECT_EQ((Candidates{{77, 0.5f}}), Export(heap_));
  Fill();
  CheckPool();
  Candidates result;
  heap_.for_each_sorted(4, [&](uint32_t id, float score) {
    if (score > 2.0f) return false;
    result.emplace_back(id, score);
    return true;
  });
  EXPECT_EQ((Candidates{{10, 1.0f}, {20, 2.0f}}), result);
  CheckPool();
}

INSTANTIATE_TEST_SUITE_P(Backends, SearchHeapPoolTest, testing::Bool());

}  // namespace
}  // namespace zvec::core
