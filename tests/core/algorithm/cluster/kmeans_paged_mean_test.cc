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

#include <ailego/pattern/scope_guard.h>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_framework.h>
#include "cluster/cluster_params.h"
#include "cluster/multi_chunk_cluster.h"
#include "cluster/paged_mean_table.h"
#include "../../../ailego/buffer/pool_payload_budget.h"

namespace zvec::core {
namespace {
class KmeansPagedMeanTest : public ::testing::Test {
 protected:
  size_t previous_{0};
  void SetUp() override {
    auto &pool = ailego::PoolPayloadBudget::get_instance();
    previous_ = pool.capacity();
    ASSERT_EQ(pool.used(), 0u);
    ASSERT_EQ(pool.init(0), 0);
  }
  void TearDown() override {
    auto &pool = ailego::PoolPayloadBudget::get_instance();
    EXPECT_EQ(pool.used(), 0u);
    EXPECT_EQ(pool.init(previous_), 0);
  }
};

TEST_F(KmeansPagedMeanTest,
       ExactStateSurvivesEvictionForNumericAndPackedMeans) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  ASSERT_EQ(pool.init(ailego::VecBufferPool::metadata_bytes_for_page_count(
                          4096, true) +
                      64 * 1024),
            0);
  std::vector<std::shared_ptr<VectorMean>> means{
      std::make_shared<NumericalVectorMean<float>>(16),
      std::make_shared<NumericalVectorMean<ailego::Float16>>(16),
      std::make_shared<NumericalVectorMean<int8_t>>(16),
      std::make_shared<NibbleVectorMean<uint8_t>>(16),
      std::make_shared<BinaryVectorMean>(16)};
  float floats[16];
  ailego::Float16 halves[16];
  int8_t ints[16];
  uint8_t packed[8]{0x23, 0xf1, 0x87, 0x10, 0x56, 0x32, 0x11, 0x81};
  for (size_t i = 0; i < 16; ++i) {
    floats[i] = float(i) / 13.0f;
    halves[i] = floats[i];
    ints[i] = static_cast<int8_t>(i - 8);
  }
  const void *inputs[]{floats, halves, ints, packed, packed};
  const size_t sizes[]{sizeof(floats), sizeof(halves), sizeof(ints), 8, 2};
  for (size_t t = 0; t < means.size(); ++t) {
    auto &mean = *means[t];
    ASSERT_TRUE(mean.plus(inputs[t], sizes[t]));
    ASSERT_TRUE(mean.plus(inputs[t], sizes[t]));
    std::vector<uint8_t> expected(mean.state_size()), scratch;
    mean.save_state(expected.data());
    PagedMeanTable table;
    ASSERT_EQ(table.init(4096, mean), 0);
    for (size_t i = 0; i < 4096; ++i)
      ASSERT_EQ(table.write(i, mean, &scratch), 0);
    for (size_t i = 0; i < 4096; i += 17) {
      mean.reset();
      ASSERT_EQ(table.read(i, &mean, &scratch), 0);
      EXPECT_EQ(scratch, expected);
      EXPECT_EQ(mean.count(), 2u);
    }
    EXPECT_LE(pool.used(), pool.capacity());
  }
}

TEST_F(KmeansPagedMeanTest, FixedCentroidsMatchHeapAndReleasePool) {
  static constexpr size_t kDimension = 137;
  IndexMeta meta(IndexMeta::DT_FP32, kDimension);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  auto features = std::make_shared<CompactIndexFeatures>(meta);
  for (size_t i = 0; i < 513; ++i) {
    float row[kDimension];
    for (size_t d = 0; d < kDimension; ++d)
      row[d] = float((i * 7 + d * 13) % 101) / 17;
    features->emplace(row);
  }
  for (bool batch : {false, true}) {
    IndexCluster::CentroidList reference;
    auto &pool = ailego::PoolPayloadBudget::get_instance();
    const size_t budget_bytes =
        2 * ailego::VecBufferPool::metadata_bytes_for_page_count(4096, true) +
        // Keep 64 KiB for pages after admitting the newly charged worker
        // rows and initialization tile; the means still exceed resident pages.
        64 * 1024 + (kDimension * 40 + 256) + 64 * 1024 +
        // The retained centers now keep their own capacity reservation.
        64 * (meta.element_size() + 64);
    for (size_t budget : {size_t{0}, budget_bytes}) {
      ASSERT_EQ(pool.init(budget), 0);
      auto cluster = IndexFactory::CreateCluster("KmeansCluster");
      ailego::Params params;
      params.set("zvec.kmeans.cluster.max_iterations", 3);
      params.set("zvec.kmeans.cluster.batch", batch);
      ASSERT_EQ(cluster->init(meta, params), 0);
      ASSERT_EQ(cluster->mount(features), 0);
      IndexCluster::CentroidList centers;
      // Exceed the small inline limit so clustering exercises paged means,
      // including accumulator rows that cross page boundaries.
      for (size_t i = 0; i < 64; ++i) {
        centers.emplace_back(features->element(i), meta.element_size());
        centers.back().set_follows(batch ? 3 : 0);
      }
      // A single worker fixes summation order; the table's worker rows are
      // independently exercised by the concurrent build regressions.
      auto threads = std::make_shared<SingleQueueIndexThreads>(1, false);
      ASSERT_EQ(cluster->cluster(threads, centers), 0);
      if (budget)
        EXPECT_GE(pool.used(), centers.size() * meta.element_size());
      else
        EXPECT_EQ(pool.used(), 0u);
      if (!budget) {
        reference = centers;
      } else {
        ASSERT_EQ(centers.size(), reference.size());
        for (size_t i = 0; i < centers.size(); ++i) {
          EXPECT_EQ(centers[i].follows(), reference[i].follows());
          EXPECT_DOUBLE_EQ(centers[i].score(), reference[i].score());
          EXPECT_EQ(std::memcmp(centers[i].vector<float>().data(),
                                reference[i].vector<float>().data(),
                                meta.element_size()),
                    0);
        }
      }
    }
  }
}

TEST_F(KmeansPagedMeanTest,
       LegacyClassificationDenialCanRetryWithoutLostInput) {
  IndexMeta meta(IndexMeta::DT_FP32, 4);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  auto features = std::make_shared<CompactIndexFeatures>(meta);
  const float low[]{0, 0, 0, 0}, high[]{10, 10, 10, 10};
  for (size_t i = 0; i < 100; ++i) features->emplace(i % 2 ? high : low);
  auto cluster = IndexFactory::CreateCluster("KmeansCluster");
  ASSERT_EQ(0, cluster->init(meta, ailego::Params()));
  ASSERT_EQ(0, cluster->mount(features));
  IndexCluster::CentroidList centers;
  centers.emplace_back(low, sizeof(low));
  centers.emplace_back(high, sizeof(high));
  auto threads = std::make_shared<SingleQueueIndexThreads>(2, false);
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  ASSERT_EQ(0, pool.init(1));
  std::vector<uint32_t> labels{99};
  EXPECT_EQ(IndexError_NoMemory, cluster->label(threads, centers, &labels));
  EXPECT_EQ(std::vector<uint32_t>{99}, labels);
  EXPECT_EQ(IndexError_NoMemory, cluster->classify(threads, centers));
  EXPECT_TRUE(centers[0].similars().empty());
  EXPECT_TRUE(centers[1].similars().empty());
  EXPECT_EQ(0u, pool.used());
  ASSERT_EQ(0, pool.init(2 * 1024 * 1024));
  ASSERT_EQ(0, cluster->label(threads, centers, &labels));
  ASSERT_EQ(100u, labels.size());
  for (size_t i = 0; i < labels.size(); ++i) EXPECT_EQ(i % 2, labels[i]);
  ASSERT_EQ(0, cluster->classify(threads, centers));
  EXPECT_EQ(50u, centers[0].similars().size());
  EXPECT_EQ(50u, centers[1].similars().size());
  // Returned pointer lists retain their charges after temporary shards die.
  EXPECT_GE(pool.used(), 100u * sizeof(void *));
  IndexCluster::CentroidList().swap(centers);
  EXPECT_EQ(0u, pool.used());
}

TEST_F(KmeansPagedMeanTest, MultiChunkLabelsReserveBeforeChangingOutput) {
  IndexMeta meta(IndexMeta::DT_FP32, 4);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  auto features = std::make_shared<CompactIndexFeatures>(meta);
  const float low[]{0, 0, 0, 0}, high[]{10, 10, 10, 10};
  for (size_t i = 0; i < 100; ++i) features->emplace(i % 2 ? high : low);
  MultiChunkCluster cluster;
  ailego::Params params;
  params.set("zvec.cluster.multi_chunk_cluster.count", 2);
  params.set("zvec.cluster.multi_chunk_cluster.chunk_count", 2);
  params.set("zvec.cluster.multi_chunk_cluster.thread_count", 2);
  ASSERT_EQ(0, cluster.init(meta, params));
  ASSERT_EQ(0, cluster.mount(features));
  IndexCluster::CentroidList centers;
  for (size_t chunk = 0; chunk < 2; ++chunk) {
    centers.emplace_back(low, 2 * sizeof(float));
    centers.emplace_back(high, 2 * sizeof(float));
  }
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  ASSERT_EQ(0, pool.init(1));
  std::vector<uint32_t> labels{99};
  EXPECT_EQ(IndexError_NoMemory, cluster.label(nullptr, centers, &labels));
  EXPECT_EQ(std::vector<uint32_t>{99}, labels);
  EXPECT_EQ(0u, pool.used());
  ASSERT_EQ(0, pool.init(2 * 1024 * 1024));
  ASSERT_EQ(0, cluster.label(nullptr, centers, &labels));
  ASSERT_EQ(200u, labels.size());
  for (size_t i = 0; i < labels.size(); ++i) EXPECT_EQ((i / 2) % 2, labels[i]);
  EXPECT_EQ(0u, pool.used());
}
}  // namespace
}  // namespace zvec::core


namespace zvec::core {
TEST_F(KmeansPagedMeanTest, ExportedCentersKeepPayloadAndDirectoryCharges) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  ASSERT_EQ(0, pool.init(32 * 1024 * 1024));
  constexpr size_t count = 8, dim = 1536;
  std::vector<float> input(count * dim, 1.0f);
  IndexMeta meta(IndexMeta::DT_FP32, dim);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params{});
  auto features = std::make_shared<CoherentIndexFeatures>(
      meta, input.data(), input.size() * sizeof(float));
  auto threads = std::make_shared<SingleQueueIndexThreads>(1, false);
  for (const char *name : {"KmeansCluster", "OptKmeansCluster"}) {
    SCOPED_TRACE(name);
    auto cluster = IndexFactory::CreateCluster(name);
    ailego::Params params;
    params.set(KMEANS_CLUSTER_MAX_ITERATIONS, 0);
    params.set(OPTKMEANS_CLUSTER_MAX_ITERATIONS, 1);
    ASSERT_EQ(0, cluster->init(meta, params));
    ASSERT_EQ(0, cluster->mount(features));
    cluster->suggest(count);
    IndexCluster::CentroidList centers;
    ASSERT_EQ(0, cluster->cluster(threads, centers));
    ASSERT_EQ(count, centers.size());
    cluster.reset();
    EXPECT_GE(pool.external_used(),
              count * (dim * sizeof(float) +
                       IndexCluster::Centroid::loaded_metadata_bytes()));
    const size_t before_copy = pool.external_used();
    {
      auto copy = centers;
      EXPECT_GT(pool.external_used(), before_copy);
      auto moved = std::move(copy);
      EXPECT_EQ(count, moved.size());
    }
    EXPECT_EQ(before_copy, pool.external_used());
    IndexCluster::CentroidList().swap(centers);
    EXPECT_EQ(0u, pool.external_used());
  }
}

TEST_F(KmeansPagedMeanTest, ClassifyRejectsPressureAndTransfersResultCharges) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  ASSERT_EQ(0, pool.init(32 * 1024 * 1024));
  std::vector<float> input(4096);
  for (size_t i = 0; i < input.size(); ++i) input[i] = float(i % 2);
  IndexMeta meta(IndexMeta::DT_FP32, 1);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params{});
  auto features = std::make_shared<CoherentIndexFeatures>(
      meta, input.data(), input.size() * sizeof(float));
  auto threads = std::make_shared<SingleQueueIndexThreads>(2, false);
  for (const char *name : {"KmeansCluster", "OptKmeansCluster"}) {
    SCOPED_TRACE(name);
    auto cluster = IndexFactory::CreateCluster(name);
    ASSERT_EQ(0, cluster->init(meta, {}));
    ASSERT_EQ(0, cluster->mount(features));
    float left = 0, right = 1;
    IndexCluster::CentroidList centers{
        IndexCluster::Centroid(&left, sizeof(left)),
        IndexCluster::Centroid(&right, sizeof(right))};
    {
      ailego::MemoryReservation pressure;
      pressure.add(pool.capacity() - pool.used());
      EXPECT_EQ(IndexError_NoMemory, cluster->classify(threads, centers));
    }
    ASSERT_EQ(0, cluster->classify(threads, centers));
    EXPECT_EQ(2048u, centers[0].similars().size());
    EXPECT_EQ(2048u, centers[1].similars().size());
    cluster.reset();
    EXPECT_GE(pool.external_used(), input.size() * sizeof(void *));
    IndexCluster::CentroidList().swap(centers);
    EXPECT_EQ(0u, pool.external_used());
  }
}
}  // namespace zvec::core
