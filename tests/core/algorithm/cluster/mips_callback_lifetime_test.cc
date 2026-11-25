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
#include <zvec/core/framework/index_framework.h>
#include "algorithm/cluster/cluster_params.h"
#include "algorithm/cluster/multi_chunk_cluster.h"

using namespace zvec::core;
using namespace zvec::ailego;

TEST(MipsCallbackLifetime, OptKmeansTrainingAndMultiChunkLabels) {
  IndexMeta meta(IndexMeta::DT_FP32, 3);
  meta.set_metric("MipsSquaredEuclidean", 0, Params());
  auto features = std::make_shared<CompactIndexFeatures>(meta);
  const float vectors[][3] = {{3, 0, 0}, {0, 3, 0}};
  for (size_t i = 0; i < 12; ++i) features->emplace(vectors[i % 2]);
  IndexCluster::CentroidList initial;
  for (const auto &vector : vectors)
    initial.emplace_back(vector, sizeof(vector));

  Params params;
  params.set(GENERAL_CLUSTER_COUNT, 2U);
  params.set(GENERAL_THREAD_COUNT, 1U);
  params.set(OPTKMEANS_CLUSTER_MAX_ITERATIONS, 3U);
  auto cluster = IndexFactory::CreateCluster("OptKmeansCluster");
  ASSERT_TRUE(cluster);
  ASSERT_EQ(0, cluster->init(meta, params));
  ASSERT_EQ(0, cluster->mount(features));
  auto centroids = initial;
  // init_distance_func has returned and destroyed its local metric before
  // update_clusters invokes the saved callback.
  ASSERT_EQ(0, cluster->cluster(nullptr, centroids));
  std::vector<uint32_t> labels;
  ASSERT_EQ(0, cluster->label(nullptr, centroids, &labels));
  ASSERT_EQ(12U, labels.size());
  for (size_t i = 0; i < labels.size(); ++i) EXPECT_EQ(i % 2, labels[i]);

  MultiChunkCluster multi;
  params.set("zvec.cluster.multi_chunk_cluster.count", 2U);
  params.set("zvec.cluster.multi_chunk_cluster.chunk_count", 1U);
  params.set("zvec.cluster.multi_chunk_cluster.thread_count", 1U);
  ASSERT_EQ(0, multi.init(meta, params));
  ASSERT_EQ(0, multi.mount(features));
  ASSERT_EQ(0, multi.label(nullptr, initial, &labels));
  ASSERT_EQ(12U, labels.size());
  for (size_t i = 0; i < labels.size(); ++i) EXPECT_EQ(i % 2, labels[i]);
}
