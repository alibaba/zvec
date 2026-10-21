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

#include <gtest/gtest.h>
#include "algorithm/hnsw_sparse/hnsw_sparse_streamer_entity.h"
#include "streamer_layout_test_helper.h"

namespace zvec {
namespace core {
namespace {

int InitSparseLayoutEntity(
    HnswSparseStreamerEntity &entity,
    size_t chunk_size = HnswSparseEntity::kDefaultChunkSize) {
  entity.set_chunk_size(chunk_size);
  entity.set_l0_neighbor_cnt(36);
  entity.set_upper_neighbor_cnt(36);
  entity.set_scaling_factor(84);
  entity.set_ef_construction(100);
  entity.set_sparse_meta_size(sizeof(uint64_t) + sizeof(uint32_t));
  entity.set_sparse_unit_size(sizeof(uint32_t) + sizeof(float));
  // Growth tests put every node on an upper level, exceeding its usual ratio.
  return entity.init(128 * 1024 * 1024, 10000);
}

void SeedLegacySparseLayout(const std::shared_ptr<LayoutTestStorage> &storage,
                            size_t page, uint64_t chunk_size) {
  storage->seed_legacy(page);
  storage->get("HnswT2S0")
      ->write(8 * sizeof(uint64_t), &chunk_size, sizeof(chunk_size));
}

std::string SparseLayoutVector(node_id_t id) {
  std::vector<uint32_t> indices(128);
  std::vector<float> values(128, static_cast<float>(id));
  for (uint32_t i = 0; i < indices.size(); ++i) {
    indices[i] = i;
  }
  std::string data(reinterpret_cast<const char *>(indices.data()),
                   indices.size() * sizeof(uint32_t));
  data.append(reinterpret_cast<const char *>(values.data()),
              values.size() * sizeof(float));
  return data;
}

TEST(StreamerLayoutGrowth, LegacySparseHugePageMetadataUsesLogicalPage) {
  for (size_t page : {4096u, 16384u, 65536u}) {
    SCOPED_TRACE(page);
    auto storage = std::make_shared<LayoutTestStorage>();
    storage->physical_page = 2 * 1024 * 1024;
    SeedLegacySparseLayout(storage, page, 3 * 1024 * 1024);
    // hugetlbfs rounds capacity up; total_size still counts logical bytes.
    storage->segments.at("HnswT2S0")->bytes.resize(storage->physical_page);
    IndexStreamer::Stats stats;
    HnswSparseStreamerEntity entity(stats);
    ASSERT_EQ(0, InitSparseLayoutEntity(entity));
    ASSERT_EQ(0, entity.open(storage, false));
    ASSERT_EQ(0, entity.close());
    uint64_t words[16]{};
    ASSERT_EQ(sizeof(words),
              storage->get("HnswT2S0")->fetch(0, words, sizeof(words)));
    EXPECT_EQ(3 * 1024 * 1024u, words[8]);
    EXPECT_EQ(page, words[14]);
    HnswSparseStreamerEntity reopened(stats);
    ASSERT_EQ(0, InitSparseLayoutEntity(reopened));
    ASSERT_EQ(0, reopened.open(storage, false));
    EXPECT_EQ(0, reopened.close());
  }
}

TEST(StreamerLayoutGrowth, InvalidLegacySparseHugePageLayoutIsNotRewritten) {
  auto storage = std::make_shared<LayoutTestStorage>();
  SeedLegacySparseLayout(storage, 4096, 3 * 1024 * 1024);
  auto meta = storage->segments.at("HnswT2S0");
  meta->bytes.resize(2 * 1024 * 1024);
  uint64_t invalid_total = 4097;
  meta->write(9 * sizeof(uint64_t), &invalid_total, sizeof(invalid_total));
  const auto original = meta->bytes;
  IndexStreamer::Stats stats;
  HnswSparseStreamerEntity entity(stats);
  ASSERT_EQ(0, InitSparseLayoutEntity(entity));
  EXPECT_EQ(IndexError_InvalidFormat, entity.open(storage, false));
  EXPECT_EQ(0, entity.close());
  EXPECT_EQ(0, storage->appends);
  EXPECT_EQ(0, storage->flushes);
  EXPECT_EQ(original, meta->bytes);
  // A rejected file must not prevent reuse with a valid storage.
  auto fresh = std::make_shared<LayoutTestStorage>();
  EXPECT_EQ(0, entity.open(fresh, false));
  EXPECT_EQ(0, entity.close());
}

TEST(StreamerLayoutGrowth, LegacySparseHugePageDataSurvivesGrowthAndReopen) {
  auto storage = std::make_shared<LayoutTestStorage>();
  storage->physical_page = 2 * 1024 * 1024;
  SeedLegacySparseLayout(storage, 4096, 3 * 1024 * 1024);
  IndexStreamer::Stats stats;
  HnswSparseStreamerEntity writer(stats);
  ASSERT_EQ(0, InitSparseLayoutEntity(writer));
  ASSERT_EQ(0, writer.open(storage, false));
  for (node_id_t i = 0; i < 3000; ++i) {
    node_id_t id;
    ASSERT_EQ(0, writer.add_vector(1, i, SparseLayoutVector(i), 128, &id));
    ASSERT_EQ(i, id);
    ASSERT_EQ(0, writer.update_neighbors(1, id, {{id, 0.0f}}));
  }
  writer.update_ep_and_level(0, 1);
  ASSERT_EQ(0, writer.close());
  // Recreate the legacy metadata representation of a populated huge-page file.
  auto meta = storage->segments.at("HnswT2S0");
  meta->bytes.resize(storage->physical_page);
  uint64_t reserved[3]{};
  ASSERT_EQ(sizeof(reserved),
            meta->write(13 * sizeof(uint64_t), reserved, sizeof(reserved)));
  storage->physical_page = 16384;
  HnswSparseStreamerEntity reader(stats);
  ASSERT_EQ(0, InitSparseLayoutEntity(reader));
  ASSERT_EQ(0, reader.open(storage, false));
  for (node_id_t i = 3000; i < 5000; ++i) {
    node_id_t id;
    ASSERT_EQ(0, reader.add_vector(1, i, SparseLayoutVector(i), 128, &id));
    ASSERT_EQ(i, id);
    ASSERT_EQ(0, reader.update_neighbors(1, id, {{id, 0.0f}}));
  }
  ASSERT_EQ(0, reader.close());
  storage->physical_page = 2 * 1024 * 1024;
  HnswSparseStreamerEntity returned(stats);
  ASSERT_EQ(0, InitSparseLayoutEntity(returned));
  ASSERT_EQ(0, returned.open(storage, false));
  EXPECT_EQ(5000u, returned.doc_cnt());
  for (node_id_t i = 0; i < 5000; ++i) {
    auto neighbors = returned.get_neighbors(1, i);
    ASSERT_EQ(1u, neighbors.size());
    EXPECT_EQ(i, neighbors[0]);
    EXPECT_EQ(i, returned.get_key(i));
    IndexStorage::MemoryBlock data;
    ASSERT_EQ(0, returned.get_sparse_data(i, data));
    EXPECT_EQ(SparseLayoutVector(i),
              std::string(static_cast<const char *>(data.data()), 1024));
  }
  EXPECT_EQ(0, returned.close());
}

TEST(StreamerLayoutGrowth, SparseInstanceCreatesNativeLayoutAfterForeignOpen) {
  const size_t local_page = ailego::MemoryHelper::PageSize();
  const size_t foreign_page = local_page == 4096 ? 16384 : 4096;
  auto foreign = std::make_shared<LayoutTestStorage>();
  foreign->physical_page = foreign_page;
  SeedLegacySparseLayout(foreign, foreign_page,
                         (6144 + foreign_page - 1) & ~(foreign_page - 1));
  IndexStreamer::Stats stats;
  HnswSparseStreamerEntity entity(stats);
  ASSERT_EQ(0, InitSparseLayoutEntity(entity, 4096));
  ASSERT_EQ(0, entity.open(foreign, false));
  ASSERT_EQ(0, entity.close());
  auto fresh = std::make_shared<LayoutTestStorage>();
  fresh->physical_page = local_page;
  ASSERT_EQ(0, entity.open(fresh, false));
  node_id_t id;
  ASSERT_EQ(0, entity.add_vector(1, 42, SparseLayoutVector(42), 128, &id));
  ASSERT_EQ(0u, id);
  ASSERT_EQ(0, entity.update_neighbors(1, id, {{id, 0.0f}}));
  entity.update_ep_and_level(id, 1);
  ASSERT_EQ(0, entity.close());
  HnswSparseStreamerEntity reopened(stats);
  ASSERT_EQ(0, InitSparseLayoutEntity(reopened, 4096));
  ASSERT_EQ(0, reopened.open(fresh, false));
  EXPECT_EQ(1u, reopened.doc_cnt());
  EXPECT_EQ(42u, reopened.get_key(0));
  IndexStorage::MemoryBlock data;
  ASSERT_EQ(0, reopened.get_sparse_data(0, data));
  EXPECT_EQ(SparseLayoutVector(42),
            std::string(static_cast<const char *>(data.data()), 1024));
  auto neighbors = reopened.get_neighbors(1, 0);
  ASSERT_EQ(1u, neighbors.size());
  EXPECT_EQ(0u, neighbors[0]);
  EXPECT_EQ(0, reopened.close());
}

TEST(StreamerLayoutGrowth, SparseNeighborsDoNotConsumeReaderPadding) {
  auto storage = std::make_shared<LayoutTestStorage>();
  storage->physical_page = 65536;
  storage->seed_legacy(4096);
  // Legacy sparse metadata stores the already aligned node chunk size.
  uint64_t chunk_size = 3 * 1024 * 1024;
  storage->get("HnswT2S0")
      ->write(8 * sizeof(uint64_t), &chunk_size, sizeof(chunk_size));
  IndexStreamer::Stats stats;
  auto setup = [](HnswSparseStreamerEntity &entity) {
    entity.set_l0_neighbor_cnt(36);
    entity.set_upper_neighbor_cnt(36);
    entity.set_scaling_factor(84);
    entity.set_ef_construction(100);
    entity.set_sparse_meta_size(sizeof(uint64_t) + sizeof(uint32_t));
    entity.set_sparse_unit_size(sizeof(uint32_t) + sizeof(float));
    return entity.init(64 * 1024 * 1024, 10000);
  };
  HnswSparseStreamerEntity entity(stats);
  ASSERT_EQ(0, setup(entity));
  ASSERT_EQ(0, entity.open(storage, false));
  for (node_id_t i = 0; i < 3000; ++i) {
    node_id_t id;
    ASSERT_EQ(0, entity.add_vector(1, i, std::string(), 0, &id));
    ASSERT_EQ(i, id);
    ASSERT_EQ(0, entity.update_neighbors(1, id, {{id, 0.0f}}));
  }
  entity.update_ep_and_level(0, 1);
  ASSERT_EQ(0, entity.close());
  HnswSparseStreamerEntity reopened(stats);
  ASSERT_EQ(0, setup(reopened));
  ASSERT_EQ(0, reopened.open(storage, false));
  for (node_id_t i = 0; i < 3000; ++i) {
    auto neighbors = reopened.get_neighbors(1, i);
    ASSERT_EQ(1u, neighbors.size());
    EXPECT_EQ(i, neighbors[0]);
  }
  EXPECT_EQ(0, reopened.close());
}

}  // namespace
}  // namespace core
}  // namespace zvec
