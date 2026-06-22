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

#include "cluster/multi_chunk_cluster.h"
#include <cmath>
#include <random>
#include <ailego/algorithm/kmeans.h>
#include <gtest/gtest.h>
#include <zvec/ailego/container/params.h>
#include "zvec/core/framework/index_framework.h"

using namespace zvec::core;
using namespace zvec::ailego;
using namespace zvec::ailego;
TEST(MultiChunkCluster, General) {
  const uint32_t count = 10000u;
  const uint32_t dimension = 960u;
  const uint32_t chunk_count = 480u;
  const uint32_t cluster_count = 256u;
  // const uint32_t thread_count = 4;
  const uint32_t thread_count = 16;

  IndexMeta index_meta;

  index_meta.set_meta(IndexMeta::DataType::DT_FP32, dimension);
  index_meta.set_metric("SquaredEuclidean", 0, Params());

  std::shared_ptr<CompactIndexFeatures> features(
      new CompactIndexFeatures(index_meta));

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(0.0, 5.0);

  for (uint32_t i = 0; i < count; ++i) {
    std::vector<float> vec(dimension);
    for (size_t j = 0; j < dimension; ++j) {
      vec[j] = dist(gen);
    }
    features->emplace(vec.data());
  }

  // Create a Kmeans cluster
  MultiChunkCluster cluster = MultiChunkCluster();

  Params params;
  params.set("zvec.cluster.multi_chunk_cluster.count", cluster_count);
  params.set("zvec.cluster.multi_chunk_cluster.chunk_count", chunk_count);
  params.set("zvec.cluster.multi_chunk_cluster.thread_count", thread_count);

  ASSERT_EQ(0, cluster.init(index_meta, params));
  ASSERT_EQ(0, cluster.mount(features));

  IndexCluster::CentroidList centroids;
  std::vector<uint32_t> labels;

  ASSERT_EQ(0, cluster.cluster(nullptr, centroids));

  for (size_t chunk = 0; chunk < chunk_count; ++chunk) {
    for (size_t cluster = 0; cluster < cluster_count; ++cluster) {
      size_t idx = chunk * cluster_count + cluster;
      const auto &cent = centroids[idx];
      const auto &vec = cent.vector<float>();

      std::cout << "chunk: " << chunk << ", cluster: " << cluster
                << ", dim: " << vec.size() << ", count: " << cent.follows()
                << " (" << cent.score() << ") { " << vec[0] << "," << vec[1]
                << " }" << std::endl;
      ASSERT_EQ(0u, cent.similars().size());
    }
  }

  ASSERT_EQ(0, cluster.label(nullptr, centroids, &labels));
}

TEST(MultiChunkCluster, TestChunk) {
  const uint32_t count = 10000u;
  const uint32_t dimension = 95;
  const uint32_t chunk_count = 20u;
  const uint32_t cluster_count = 256u;
  // const uint32_t thread_count = 4;
  const uint32_t thread_count = 16;

  IndexMeta index_meta;

  index_meta.set_meta(IndexMeta::DataType::DT_FP32, dimension);
  index_meta.set_metric("SquaredEuclidean", 0, Params());

  std::shared_ptr<CompactIndexFeatures> features(
      new CompactIndexFeatures(index_meta));

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(0.0, 5.0);

  for (uint32_t i = 0; i < count; ++i) {
    std::vector<float> vec(dimension);
    for (size_t j = 0; j < dimension; ++j) {
      vec[j] = dist(gen);
    }
    features->emplace(vec.data());
  }

  // Create a Kmeans cluster
  MultiChunkCluster cluster = MultiChunkCluster();

  Params params;
  params.set("zvec.cluster.multi_chunk_cluster.count", cluster_count);
  params.set("zvec.cluster.multi_chunk_cluster.chunk_count", chunk_count);
  params.set("zvec.cluster.multi_chunk_cluster.thread_count", thread_count);

  ASSERT_EQ(0, cluster.init(index_meta, params));
  ASSERT_EQ(0, cluster.mount(features));

  IndexCluster::CentroidList centroids;
  std::vector<uint32_t> labels;

  ASSERT_EQ(0, cluster.cluster(nullptr, centroids));

  for (size_t chunk = 0; chunk < chunk_count; ++chunk) {
    for (size_t cluster = 0; cluster < cluster_count; ++cluster) {
      size_t idx = chunk * cluster_count + cluster;
      const auto &cent = centroids[idx];
      const auto &vec = cent.vector<float>();

      std::cout << "chunk: " << chunk << ", cluster: " << cluster
                << ", dim: " << vec.size() << ", count: " << cent.follows()
                << " (" << cent.score() << ") { " << vec[0] << "," << vec[1]
                << " }" << std::endl;
      ASSERT_EQ(0u, cent.similars().size());
    }
  }

  ASSERT_EQ(0, cluster.label(nullptr, centroids, &labels));
}

TEST(MultiChunkCluster, General_InnerProduct) {
  const uint32_t count = 50000u;
  const uint32_t dimension = 96u;
  const uint32_t chunk_count = 12u;
  const uint32_t cluster_count = 16u;
  const uint32_t chain_length = 0;
  const uint32_t thread_count = 16;

  IndexMeta index_meta;

  index_meta.set_meta(IndexMeta::DataType::DT_FP32, dimension);
  index_meta.set_metric("InnerProduct", 0, Params());

  std::shared_ptr<CompactIndexFeatures> features(
      new CompactIndexFeatures(index_meta));

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-1.0, 1.0);

  // do normalize
  for (uint32_t i = 0; i < count; ++i) {
    std::vector<float> vec(dimension);

    float norm = 0;
    for (size_t j = 0; j < dimension; ++j) {
      vec[j] = dist(gen);

      norm += vec[j] * vec[j];
    }
    norm = sqrt(norm);

    for (size_t j = 0; j < dimension; ++j) {
      vec[j] /= norm;
    }

    features->emplace(vec.data());
  }

  // Create a Kmeans cluster
  MultiChunkCluster cluster = MultiChunkCluster();

  Params params;
  params.set("zvec.cluster.multi_chunk_cluster.count", cluster_count);
  params.set("zvec.cluster.multi_chunk_cluster.chunk_count", chunk_count);
  params.set("zvec.cluster.multi_chunk_cluster.thread_count", thread_count);
  params.set("zvec.cluster.multi_chunk_cluster.markov_chain_length",
             chain_length);

  ASSERT_EQ(0, cluster.init(index_meta, params));
  ASSERT_EQ(0, cluster.mount(features));

  IndexCluster::CentroidList centroids;
  std::vector<uint32_t> labels;

  ASSERT_EQ(0, cluster.cluster(nullptr, centroids));

  for (size_t chunk = 0; chunk < chunk_count; ++chunk) {
    for (size_t cluster = 0; cluster < cluster_count; ++cluster) {
      size_t idx = chunk * cluster_count + cluster;
      const auto &cent = centroids[idx];
      const auto &vec = cent.vector<float>();

      std::cout << "chunk: " << chunk << ", cluster: " << cluster
                << ", dim: " << vec.size() << ", count: " << cent.follows()
                << " (" << cent.score() << ") { " << vec[0] << ", " << vec[1]
                << ", " << vec[2] << ", ... , " << vec[vec.size() - 2] << ", "
                << vec[vec.size() - 1] << " }" << std::endl;
      ASSERT_EQ(0u, cent.similars().size());
    }
  }

  ASSERT_EQ(0, cluster.label(nullptr, centroids, &labels));
}
