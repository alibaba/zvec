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

#include <functional>
#include <gtest/gtest.h>
#include "hnsw_streamer_entity.h"
#include "../streamer_layout_test_helper.h"

namespace zvec::core {
namespace {

// A bulk mmap copy can be observed after its first word is stored. Sample
// the graph at that point instead of relying on a rare thread preemption.
class ObservedNeighborSegment : public LayoutTestSegment {
 public:
  explicit ObservedNeighborSegment(size_t capacity)
      : LayoutTestSegment(capacity) {}

  size_t write(size_t offset, const void *data, size_t len) override {
    if (!observe || len <= sizeof(uint32_t)) {
      auto written = LayoutTestSegment::write(offset, data, len);
      if (observe) observe();
      return written;
    }
    const auto head = LayoutTestSegment::write(offset, data, sizeof(uint32_t));
    observe();
    return head + LayoutTestSegment::write(
                      offset + head, static_cast<const char *>(data) + head,
                      len - head);
  }

  std::function<void()> observe;
};

class ObservedNeighborStorage : public LayoutTestStorage {
 public:
  int append(const std::string &id, size_t size) override {
    int ret = LayoutTestStorage::append(id, size);
    if (ret == 0 && id == "HnswT4S0") {
      segment = std::make_shared<ObservedNeighborSegment>(
          segments.at(id)->capacity());
      segments[id] = segment;
    }
    return ret;
  }
  std::shared_ptr<ObservedNeighborSegment> segment;
};

TEST(HnswNeighborPublication, ReadersNeverSeeUnwrittenUpperNeighborIds) {
  auto storage = std::make_shared<ObservedNeighborStorage>();
  IndexStreamer::Stats stats;
  HnswMmapStreamerEntity entity(stats);
  entity.set_vector_size(sizeof(float));
  entity.set_l0_neighbor_cnt(16);
  entity.set_upper_neighbor_cnt(16);
  entity.set_scaling_factor(16);
  entity.set_ef_construction(100);
  ASSERT_EQ(0, entity.init(10000));
  ASSERT_EQ(0, entity.open(storage, 32 * 1024 * 1024, false));
  float vector = 1.0f;
  for (node_id_t id = 0; id < 3; ++id) {
    ASSERT_EQ(0, entity.add_vector_with_id(id == 0 ? 0 : 1, id, &vector));
  }
  ASSERT_NE(nullptr, storage->segment);
  size_t observations = 0;
  storage->segment->observe = [&]() {
    ++observations;
    auto neighbors = entity.get_neighbors_typed(1, 1);
    // The old empty list and the complete new list are both valid. Node 0
    // has no upper layer and must never be exposed as an upper neighbor.
    ASSERT_LE(neighbors.size(), 1U);
    if (neighbors.size() == 1) EXPECT_EQ(2U, neighbors[0]);
  };
  ASSERT_EQ(0, entity.update_neighbors(1, 1, {{2, 0.0f}}));
  EXPECT_GT(observations, 0U);
  storage->segment->observe = nullptr;
  auto neighbors = entity.get_neighbors_typed(1, 1);
  ASSERT_EQ(1U, neighbors.size());
  EXPECT_EQ(2U, neighbors[0]);
  EXPECT_EQ(0, entity.close());
}
}  // namespace
}  // namespace zvec::core
