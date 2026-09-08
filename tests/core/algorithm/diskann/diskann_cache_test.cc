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

#include <limits>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_factory.h>
#include "diskann_params.h"
#include "diskann_util.h"

using namespace zvec::core;

TEST(DiskAnnCachePreloadTest, BoundsBatchByScratchBufferCapacity) {
  EXPECT_EQ(DiskAnnUtil::cache_load_batch_size(0), 1u);
  EXPECT_EQ(DiskAnnUtil::cache_load_batch_size(1), 128u);
  EXPECT_EQ(DiskAnnUtil::cache_load_batch_size(2), 64u);
  EXPECT_EQ(DiskAnnUtil::cache_load_batch_size(128), 1u);
  EXPECT_EQ(DiskAnnUtil::cache_load_batch_size(129), 1u);
}

TEST(DiskAnnCacheConfigTest, FailedReinitKeepsPreviousValidConfiguration) {
  zvec::ailego::Params valid;
  valid.set(PARAM_DISKANN_SEARCHER_LIST_SIZE, 321);
  valid.set(PARAM_DISKANN_SEARCHER_CACHE_NODE_NUM, 7);

  zvec::ailego::Params invalid;
  invalid.set(PARAM_DISKANN_SEARCHER_CACHE_NODE_NUM, -1);

  auto searcher = IndexFactory::CreateSearcher("DiskAnnSearcher");
  ASSERT_NE(searcher, nullptr);
  ASSERT_EQ(0, searcher->init(valid));
  ASSERT_EQ(IndexError_InvalidArgument, searcher->init(invalid));
  uint32_t list_size = 0;
  long long cache_nodes = 0;
  EXPECT_TRUE(
      searcher->params().get(PARAM_DISKANN_SEARCHER_LIST_SIZE, &list_size));
  EXPECT_TRUE(searcher->params().get(PARAM_DISKANN_SEARCHER_CACHE_NODE_NUM,
                                     &cache_nodes));
  EXPECT_EQ(list_size, 321u);
  EXPECT_EQ(cache_nodes, 7);

  auto streamer = IndexFactory::CreateStreamer("DiskAnnStreamer");
  ASSERT_NE(streamer, nullptr);
  IndexMeta meta(IndexMeta::DataType::DT_FP32, 8);
  IndexMeta invalid_meta(IndexMeta::DataType::DT_FP32, 16);
  ASSERT_EQ(0, streamer->init(meta, valid));
  ASSERT_EQ(IndexError_InvalidArgument, streamer->init(invalid_meta, invalid));
  EXPECT_EQ(streamer->meta().dimension(), 8u);
}

TEST(DiskAnnCacheConfigTest, RejectsOutOfRangeNodeCount) {
  zvec::ailego::Params negative;
  negative.set(PARAM_DISKANN_SEARCHER_CACHE_NODE_NUM, -1);

  zvec::ailego::Params too_large;
  const long long too_many_nodes =
      static_cast<long long>((std::numeric_limits<uint32_t>::max)()) + 1;
  too_large.set(PARAM_DISKANN_SEARCHER_CACHE_NODE_NUM, too_many_nodes);

  auto searcher = IndexFactory::CreateSearcher("DiskAnnSearcher");
  ASSERT_NE(searcher, nullptr);
  EXPECT_EQ(IndexError_InvalidArgument, searcher->init(negative));
  EXPECT_EQ(IndexError_InvalidArgument, searcher->init(too_large));

  auto streamer = IndexFactory::CreateStreamer("DiskAnnStreamer");
  ASSERT_NE(streamer, nullptr);
  IndexMeta meta(IndexMeta::DataType::DT_FP32, 8);
  EXPECT_EQ(IndexError_InvalidArgument, streamer->init(meta, negative));
  EXPECT_EQ(IndexError_InvalidArgument, streamer->init(meta, too_large));
}
