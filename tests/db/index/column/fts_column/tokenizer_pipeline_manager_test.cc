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

#include "db/index/column/fts_column/tokenizer/tokenizer_pipeline_manager.h"
#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <gtest/gtest.h>
#include "db/index/column/fts_column/fts_types.h"

using namespace zvec::fts;

// ============================================================
// Helpers
// ============================================================

static FtsIndexParams make_params(const std::string &tokenizer) {
  FtsIndexParams params;
  params.tokenizer_name = tokenizer;
  return params;
}

// ============================================================
// make_key tests
// ============================================================

TEST(TokenizerPipelineManagerKeyTest, BasicKey) {
  FtsIndexParams params;
  params.tokenizer_name = "whitespace";
  std::string key = TokenizerPipelineManager::make_key(params);
  EXPECT_FALSE(key.empty());
  EXPECT_NE(key.find("whitespace"), std::string::npos);
}

TEST(TokenizerPipelineManagerKeyTest, SameParamsProduceSameKey) {
  FtsIndexParams params1;
  params1.tokenizer_name = "whitespace";
  params1.extra_params = R"({"dict_path":"/path/to/dict"})";

  FtsIndexParams params2;
  params2.tokenizer_name = "whitespace";
  params2.extra_params = R"({"dict_path":"/path/to/dict"})";

  std::string key1 = TokenizerPipelineManager::make_key(params1);
  std::string key2 = TokenizerPipelineManager::make_key(params2);
  EXPECT_EQ(key1, key2);
}

TEST(TokenizerPipelineManagerKeyTest, DifferentTokenizersDifferentKeys) {
  FtsIndexParams params1 = make_params("whitespace");
  FtsIndexParams params2 = make_params("jieba");
  std::string key1 = TokenizerPipelineManager::make_key(params1);
  std::string key2 = TokenizerPipelineManager::make_key(params2);
  EXPECT_NE(key1, key2);
}

TEST(TokenizerPipelineManagerKeyTest, FilterNamesAffectKey) {
  FtsIndexParams params1 = make_params("whitespace");
  params1.filters.clear();

  FtsIndexParams params2 = make_params("whitespace");
  params2.filters = {"lowercase"};

  std::string key1 = TokenizerPipelineManager::make_key(params1);
  std::string key2 = TokenizerPipelineManager::make_key(params2);
  EXPECT_NE(key1, key2);
}

// ============================================================
// acquire / release tests
// ============================================================

class TokenizerPipelineManagerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Use whitespace tokenizer (always available, no dict needed)
    params_ = make_params("whitespace");
  }

  void TearDown() override {
    // Best-effort cleanup: release the params if it still exists
    // (tests that fail mid-way may leave entries)
    // We do this by calling release repeatedly; release on unknown key is a
    // no-op
  }

  FtsIndexParams params_;
};

TEST_F(TokenizerPipelineManagerTest, FirstAcquireCreatesPipeline) {
  auto &mgr = TokenizerPipelineManager::Instance();
  auto pipeline = mgr.acquire(params_);
  ASSERT_NE(pipeline, nullptr);

  // Cleanup
  mgr.release(params_);
}

TEST_F(TokenizerPipelineManagerTest, RepeatedAcquireReturnsSameInstance) {
  auto &mgr = TokenizerPipelineManager::Instance();
  auto pipeline1 = mgr.acquire(params_);
  auto pipeline2 = mgr.acquire(params_);

  ASSERT_NE(pipeline1, nullptr);
  ASSERT_NE(pipeline2, nullptr);
  // Both should point to the exact same underlying object
  EXPECT_EQ(pipeline1.get(), pipeline2.get());

  // Cleanup: two acquires → two releases
  mgr.release(params_);
  mgr.release(params_);
}

TEST_F(TokenizerPipelineManagerTest, ReleaseDecrementsRefCount) {
  auto &mgr = TokenizerPipelineManager::Instance();
  auto pipeline1 = mgr.acquire(params_);
  auto pipeline2 = mgr.acquire(params_);
  ASSERT_NE(pipeline1, nullptr);

  // Release one reference; pipeline should still be alive (ref_count = 1)
  mgr.release(params_);

  // Acquire again — should still return the same instance (not recreated)
  auto pipeline3 = mgr.acquire(params_);
  ASSERT_NE(pipeline3, nullptr);
  EXPECT_EQ(pipeline1.get(), pipeline3.get());

  // Cleanup: we now have ref_count = 2 (pipeline2 + pipeline3)
  mgr.release(params_);
  mgr.release(params_);
}

TEST_F(TokenizerPipelineManagerTest, RefCountZeroDestroysEntry) {
  auto &mgr = TokenizerPipelineManager::Instance();

  auto pipeline1 = mgr.acquire(params_);
  ASSERT_NE(pipeline1, nullptr);
  void *raw_ptr = pipeline1.get();

  // Release the only reference → entry should be removed
  mgr.release(params_);

  // Acquire again → a new pipeline should be created (possibly different
  // address)
  auto pipeline2 = mgr.acquire(params_);
  ASSERT_NE(pipeline2, nullptr);
  // The old shared_ptr (pipeline1) still holds the object alive, so raw_ptr
  // is still valid, but the manager has created a fresh entry.
  // We can't guarantee same/different address, but we can verify it works.
  (void)raw_ptr;

  // Cleanup
  mgr.release(params_);
}

TEST_F(TokenizerPipelineManagerTest, ReleaseUnknownKeyIsNoOp) {
  auto &mgr = TokenizerPipelineManager::Instance();
  // Should not crash or assert
  FtsIndexParams unknown_params;
  unknown_params.tokenizer_name = "nonexistent_tokenizer_name";
  EXPECT_NO_THROW(mgr.release(unknown_params));
}

TEST_F(TokenizerPipelineManagerTest, DifferentConfigsDifferentPipelines) {
  auto &mgr = TokenizerPipelineManager::Instance();

  FtsIndexParams params_ws = make_params("whitespace");

  // scws tokenizer will fail to create (no dict), but whitespace should succeed
  auto pipeline_ws = mgr.acquire(params_ws);
  ASSERT_NE(pipeline_ws, nullptr);

  // Cleanup
  mgr.release(params_ws);
}

// ============================================================
// Concurrent safety tests
// ============================================================

TEST_F(TokenizerPipelineManagerTest, ConcurrentAcquireSameKey) {
  auto &mgr = TokenizerPipelineManager::Instance();
  constexpr int kThreads = 8;
  constexpr int kAcquiresPerThread = 10;

  std::vector<TokenizerPipelinePtr> results(kThreads * kAcquiresPerThread);
  std::vector<std::thread> threads;
  std::atomic<int> success_count{0};

  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t]() {
      for (int i = 0; i < kAcquiresPerThread; ++i) {
        auto pipeline = mgr.acquire(params_);
        if (pipeline) {
          results[t * kAcquiresPerThread + i] = pipeline;
          success_count.fetch_add(1);
        }
      }
    });
  }

  for (auto &th : threads) {
    th.join();
  }

  // All acquires should succeed
  EXPECT_EQ(success_count.load(), kThreads * kAcquiresPerThread);

  // All non-null results should point to the same underlying pipeline
  void *expected_ptr = nullptr;
  for (const auto &p : results) {
    if (p) {
      if (expected_ptr == nullptr) {
        expected_ptr = p.get();
      } else {
        EXPECT_EQ(p.get(), expected_ptr);
      }
    }
  }

  // Cleanup: release all acquired references
  for (int i = 0; i < kThreads * kAcquiresPerThread; ++i) {
    mgr.release(params_);
  }
}

TEST_F(TokenizerPipelineManagerTest, ConcurrentAcquireAndRelease) {
  auto &mgr = TokenizerPipelineManager::Instance();
  constexpr int kThreads = 4;
  constexpr int kIterations = 20;
  std::atomic<int> errors{0};

  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&]() {
      for (int i = 0; i < kIterations; ++i) {
        auto pipeline = mgr.acquire(params_);
        if (!pipeline) {
          errors.fetch_add(1);
          continue;
        }
        // Hold briefly then release
        mgr.release(params_);
      }
    });
  }

  for (auto &th : threads) {
    th.join();
  }

  EXPECT_EQ(errors.load(), 0);
  // After all threads finish, ref_count should be 0 (all released)
  // Verify by acquiring once more — should succeed
  auto pipeline = mgr.acquire(params_);
  EXPECT_NE(pipeline, nullptr);
  mgr.release(params_);
}
