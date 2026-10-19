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

#include "algorithm/streamer_layout.h"
#include <algorithm>
#include <cstring>
#include <map>
#include <vector>
#include <gtest/gtest.h>
#include "algorithm/hnsw/hnsw_chunk.h"
#include "algorithm/hnsw/hnsw_streamer_entity.h"
#include "algorithm/hnsw_sparse/hnsw_sparse_chunk.h"
#include "streamer_layout_test_helper.h"

namespace zvec {
namespace core {
namespace {

int OpenBroker(ChunkBroker &broker, const IndexStorage::Pointer &storage) {
  uint32_t chunk_size = 2 * 1024 * 1024;
  int ret = broker.open(storage, chunk_size, false);
  broker.set_max_chunks_size(32 * 1024 * 1024);
  return ret;
}
int OpenBroker(SparseChunkBroker &broker,
               const IndexStorage::Pointer &storage) {
  return broker.open(storage, 32 * 1024 * 1024, 2 * 1024 * 1024, false);
}

template <typename Broker>
class StreamerLayoutTest : public ::testing::Test {};
using Brokers = ::testing::Types<ChunkBroker, SparseChunkBroker>;
TYPED_TEST_SUITE(StreamerLayoutTest, Brokers);

TYPED_TEST(StreamerLayoutTest, RestoresWriterPageAndPersistsItAcrossGrowth) {
  for (size_t page : {4096u, 16384u, 65536u}) {
    auto storage = std::make_shared<LayoutTestStorage>();
    storage->seed_legacy(page);
    IndexStreamer::Stats stats;
    TypeParam broker(stats);
    ASSERT_EQ(0, OpenBroker(broker, storage));
    EXPECT_EQ(page, broker.layout_page_size());
    EXPECT_EQ(page * 2, broker.align_size(page + 1));
    auto allocation = broker.alloc_chunk(TypeParam::CHUNK_TYPE_NODE, 0, 4097);
    ASSERT_EQ(0, allocation.first);
    ASSERT_NE(nullptr, allocation.second);
    // Physical padding may exceed the writer's logical chunk size.
    EXPECT_GE(allocation.second->capacity(), broker.align_size(4097));
    ASSERT_EQ(0, broker.close());
    uint64_t words[16]{};
    auto meta = storage->get("HnswT2S0");
    ASSERT_EQ(sizeof(words), meta->fetch(0, words, sizeof(words)));
    EXPECT_EQ(kStreamerLayoutV1, words[13]);
    EXPECT_EQ(page, words[14]);
    TypeParam reopened(stats);
    ASSERT_EQ(0, OpenBroker(reopened, storage));
    EXPECT_EQ(page, reopened.layout_page_size());
    EXPECT_EQ(1u, reopened.get_chunk_cnt(TypeParam::CHUNK_TYPE_NODE));
    ASSERT_EQ(0, reopened.close());
  }
}

TYPED_TEST(StreamerLayoutTest, UnreadableMetadataIsNotRecreatedOrFlushed) {
  auto storage = std::make_shared<LayoutTestStorage>();
  storage->seed_legacy(4096);
  storage->unreadable = true;
  IndexStreamer::Stats stats;
  TypeParam broker(stats);
  EXPECT_EQ(IndexError_ReadData, OpenBroker(broker, storage));
  EXPECT_EQ(0, storage->appends);
  EXPECT_EQ(0, broker.close());
  EXPECT_EQ(0, broker.close());
  EXPECT_EQ(0, storage->flushes);
  EXPECT_EQ(IndexError_Uninitialized, broker.flush(0));
  storage->unreadable = false;
  EXPECT_EQ(0, OpenBroker(broker, storage));
  EXPECT_EQ(0, broker.close());
}

TYPED_TEST(StreamerLayoutTest, InvalidLayoutIsRejectedWithoutWritingMetadata) {
  auto storage = std::make_shared<LayoutTestStorage>();
  storage->seed_legacy(4096);
  uint64_t invalid_version = kStreamerLayoutV1 + 1;
  auto meta = storage->segments.at("HnswT2S0");
  meta->write(13 * sizeof(uint64_t), &invalid_version, sizeof(invalid_version));
  auto original = meta->bytes;
  IndexStreamer::Stats stats;
  TypeParam broker(stats);
  EXPECT_EQ(IndexError_InvalidFormat, OpenBroker(broker, storage));
  EXPECT_EQ(0, broker.close());
  EXPECT_EQ(0, storage->flushes);
  EXPECT_EQ(original, meta->bytes);
}

TYPED_TEST(StreamerLayoutTest, NewIndexRecordsLayoutWithoutChangingMetaSize) {
  auto storage = std::make_shared<LayoutTestStorage>();
  IndexStreamer::Stats stats;
  TypeParam broker(stats);
  ASSERT_EQ(0, OpenBroker(broker, storage));
  auto meta = storage->get("HnswT2S0");
  ASSERT_NE(nullptr, meta);
  EXPECT_EQ(128u, meta->data_size());
  ASSERT_EQ(0, broker.close());
  TypeParam reopened(stats);
  ASSERT_EQ(0, OpenBroker(reopened, storage));
  EXPECT_EQ(ailego::MemoryHelper::PageSize(), reopened.layout_page_size());
  EXPECT_EQ(0, reopened.close());
}

TEST(StreamerLayoutGrowth, DenseNeighborsDoNotConsumeReaderPadding) {
  auto storage = std::make_shared<LayoutTestStorage>();
  storage->seed_legacy(4096);
  IndexStreamer::Stats stats;
  auto setup = [](HnswStreamerEntity &entity) {
    entity.set_vector_size(1024 * sizeof(float));
    entity.set_l0_neighbor_cnt(16);
    entity.set_upper_neighbor_cnt(16);
    entity.set_scaling_factor(16);
    entity.set_ef_construction(100);
    return entity.init(10000);
  };
  HnswStreamerEntity entity(stats);
  ASSERT_EQ(0, setup(entity));
  ASSERT_EQ(0, entity.open(storage, 512 * 1024 * 1024, false));
  std::vector<float> vector(1024, 1.0f);
  // Every node has an upper level so this crosses several logical chunks.
  // The 16K reader gives each 69632-byte logical chunk 81920 physical bytes.
  for (node_id_t i = 0; i < 2500; ++i) {
    node_id_t id;
    ASSERT_EQ(0, entity.add_vector(1, i, vector.data(), &id));
    ASSERT_EQ(i, id);
    ASSERT_EQ(0, entity.update_neighbors(1, id, {{id, 0.0f}}));
  }
  entity.update_ep_and_level(0, 1);
  ASSERT_EQ(0, entity.close());
  HnswStreamerEntity reopened(stats);
  ASSERT_EQ(0, setup(reopened));
  ASSERT_EQ(0, reopened.open(storage, 512 * 1024 * 1024, false));
  for (node_id_t i = 0; i < 2500; ++i) {
    auto neighbors = reopened.get_neighbors(1, i);
    ASSERT_EQ(1u, neighbors.size());
    EXPECT_EQ(i, neighbors[0]);
  }
  EXPECT_EQ(0, reopened.close());
}


}  // namespace
}  // namespace core
}  // namespace zvec
