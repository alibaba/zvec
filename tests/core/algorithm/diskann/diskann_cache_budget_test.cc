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

#include "diskann_cache_budget.h"
#include <limits>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_factory.h>
#include "diskann_params.h"

using namespace zvec::core;

TEST(DiskAnnCacheBudgetTest, AccountsForVectorTypeAndCacheBookkeeping) {
  IndexMeta fp32_meta(IndexMeta::DataType::DT_FP32, 768);
  IndexMeta fp16_meta(IndexMeta::DataType::DT_FP16, 768);
  constexpr uint32_t kMaxDegree = 32;

  const uint64_t fp32_payload =
      DiskAnnCacheBudget::PayloadBytesPerNode(fp32_meta, kMaxDegree);
  const uint64_t fp16_payload =
      DiskAnnCacheBudget::PayloadBytesPerNode(fp16_meta, kMaxDegree);

  EXPECT_EQ(fp32_payload, 768u * sizeof(float) + 33u * sizeof(diskann_id_t));
  EXPECT_EQ(fp16_payload, 768u * sizeof(uint16_t) + 33u * sizeof(diskann_id_t));
  EXPECT_GT(DiskAnnCacheBudget::EstimatedBytesPerNode(fp32_meta, kMaxDegree),
            fp32_payload);
  EXPECT_GT(DiskAnnCacheBudget::EstimatedBytesPerNode(fp16_meta, kMaxDegree),
            fp16_payload);
}

TEST(DiskAnnCacheBudgetTest, ResolvesFloorAndExistingTenPercentCap) {
  constexpr uint64_t kBytesPerNode = 2048;
  constexpr uint64_t kDocCount = 10000;

  EXPECT_EQ(DiskAnnCacheBudget::ResolveNodeCount(100 * kBytesPerNode, kDocCount,
                                                 kBytesPerNode),
            100u);
  EXPECT_EQ(DiskAnnCacheBudget::ResolveNodeCount(2000 * kBytesPerNode,
                                                 kDocCount, kBytesPerNode),
            1000u);
  EXPECT_EQ(DiskAnnCacheBudget::ResolveNodeCount(kBytesPerNode - 1, kDocCount,
                                                 kBytesPerNode),
            0u);
  EXPECT_EQ(
      DiskAnnCacheBudget::ResolveNodeCount(kBytesPerNode, 1, kBytesPerNode),
      1u);
  EXPECT_EQ(
      DiskAnnCacheBudget::ResolveNodeCount(kBytesPerNode, 0, kBytesPerNode),
      0u);
  const uint64_t max_addressable_nodes =
      static_cast<uint64_t>(std::numeric_limits<size_t>::max()) / kBytesPerNode;
  const uint32_t expected_large_budget_nodes =
      static_cast<uint32_t>(std::min<uint64_t>(
          max_addressable_nodes, std::numeric_limits<uint32_t>::max()));
  EXPECT_EQ(DiskAnnCacheBudget::ResolveNodeCount(
                std::numeric_limits<uint64_t>::max(),
                std::numeric_limits<uint64_t>::max(), kBytesPerNode),
            expected_large_budget_nodes);
}

TEST(DiskAnnCacheBudgetTest, RejectsConflictingNodeCountAndByteBudget) {
  auto searcher = IndexFactory::CreateSearcher("DiskAnnSearcher");
  ASSERT_NE(searcher, nullptr);

  zvec::ailego::Params params;
  params.set(PARAM_DISKANN_SEARCHER_CACHE_NODE_NUM, 100);
  params.set(PARAM_DISKANN_SEARCHER_CACHE_NODE_BUDGET_BYTES,
             static_cast<uint64_t>(1024 * 1024));
  EXPECT_EQ(IndexError_InvalidArgument, searcher->init(params));
}

TEST(DiskAnnCacheBudgetTest, RejectsNegativeCacheConfiguration) {
  auto searcher = IndexFactory::CreateSearcher("DiskAnnSearcher");
  ASSERT_NE(searcher, nullptr);

  zvec::ailego::Params negative_budget;
  negative_budget.set(PARAM_DISKANN_SEARCHER_CACHE_NODE_BUDGET_BYTES,
                      static_cast<long long>(-1));
  EXPECT_EQ(IndexError_InvalidArgument, searcher->init(negative_budget));

  auto streamer = IndexFactory::CreateStreamer("DiskAnnStreamer");
  ASSERT_NE(streamer, nullptr);
  IndexMeta meta(IndexMeta::DataType::DT_FP32, 8);
  zvec::ailego::Params negative_nodes;
  negative_nodes.set(PARAM_DISKANN_SEARCHER_CACHE_NODE_NUM, -1);
  EXPECT_EQ(IndexError_InvalidArgument, streamer->init(meta, negative_nodes));
}
