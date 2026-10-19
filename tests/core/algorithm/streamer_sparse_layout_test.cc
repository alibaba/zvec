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
