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
#include "hnsw_streamer.h"
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

  bool write_batch(const IndexStorage::SegmentData *writes,
                   size_t count) override {
    batch_error_ = batch_failure ? batch_failure() : 0;
    return batch_error_ == 0 && LayoutTestSegment::write_batch(writes, count);
  }

  int write_error() const override {
    return batch_error_ == 0 ? IndexError_WriteData : batch_error_;
  }

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
  std::function<int()> batch_failure;

 private:
  int batch_error_{0};
};

class ObservedNeighborStorage : public LayoutTestStorage {
 public:
  explicit ObservedNeighborStorage(const char *observed_id = "HnswT4S0")
      : observed_id_(observed_id) {}

  int append(const std::string &id, size_t size) override {
    int ret = LayoutTestStorage::append(id, size);
    if (ret == 0 && id == observed_id_) {
      segment = std::make_shared<ObservedNeighborSegment>(
          segments.at(id)->capacity());
      segments[id] = segment;
    }
    return ret;
  }
  std::shared_ptr<ObservedNeighborSegment> segment;

 private:
  const std::string observed_id_;
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

TEST(HnswNeighborPublication, AdmissionFailureFallsBackBeforePublishingCount) {
  auto storage = std::make_shared<ObservedNeighborStorage>();
  IndexStreamer::Stats stats;
  HnswBufferPoolStreamerEntity entity(stats);
  entity.set_vector_size(sizeof(float));
  entity.set_l0_neighbor_cnt(16);
  entity.set_upper_neighbor_cnt(16);
  entity.set_scaling_factor(16);
  entity.set_ef_construction(100);
  ASSERT_EQ(0, entity.init(10000));
  ASSERT_EQ(0, entity.open(storage, 32 * 1024 * 1024, false));
  const float vector = 1.0f;
  for (node_id_t id = 0; id < 3; ++id) {
    ASSERT_EQ(0, entity.add_vector_with_id(1, id, &vector));
  }
  ASSERT_NE(nullptr, storage->segment);
  storage->segment->batch_failure = [] { return IndexError_NoMemory; };
  std::vector<uint32_t> observed_counts;
  storage->segment->observe = [&] {
    const auto neighbors = entity.get_neighbors(1, 0);
    observed_counts.push_back(neighbors.size());
    if (neighbors.size() >= 1) EXPECT_EQ(1U, neighbors[0]);
    if (neighbors.size() >= 2) EXPECT_EQ(2U, neighbors[1]);
  };
  EXPECT_EQ(0, entity.update_neighbors(1, 0, {{1, 0.0f}}));
  entity.add_neighbor(1, 0, 1, 2);
  EXPECT_EQ((std::vector<uint32_t>{0, 1, 1, 2}), observed_counts);
  storage->segment->observe = nullptr;
  EXPECT_EQ(0, entity.close());
}

TEST(HnswNeighborPublication, StreamerReportsForwardAndReverseWriteFailures) {
  for (int failing_batch : {1, 2}) {
    SCOPED_TRACE(failing_batch);
    auto storage = std::make_shared<ObservedNeighborStorage>("HnswT3S0");
    auto streamer = IndexFactory::CreateStreamer("HnswStreamer");
    ASSERT_NE(nullptr, streamer);
    const auto type = IndexMeta::DataType::DT_FP32;
    constexpr size_t kDimension = 4;
    IndexMeta meta(type, kDimension);
    meta.set_metric("SquaredEuclidean", 0, ailego::Params{});
    ASSERT_EQ(0, streamer->init(meta, {}));
    ASSERT_EQ(0, streamer->open(storage));
    auto ctx = streamer->create_context();
    ASSERT_NE(nullptr, ctx);
    IndexQueryMeta qmeta(type, kDimension);
    const float first[kDimension]{1.0f};
    const float second[kDimension]{2.0f};
    ASSERT_EQ(0, streamer->add_impl(0, first, qmeta, ctx));
    ASSERT_NE(nullptr, storage->segment);
    int batches = 0;
    storage->segment->batch_failure = [&] {
      return ++batches == failing_batch ? IndexError_WriteData : 0;
    };
    EXPECT_EQ(IndexError_WriteData, streamer->add_impl(1, second, qmeta, ctx));
    EXPECT_EQ(failing_batch, batches);
    storage->segment->batch_failure = nullptr;
    ctx.reset();
    EXPECT_EQ(0, streamer->close());
  }
}
}  // namespace
}  // namespace zvec::core
