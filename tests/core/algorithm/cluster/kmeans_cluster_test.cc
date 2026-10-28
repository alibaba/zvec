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

#include <algorithm>
#include <cmath>
#include <random>
#include <gtest/gtest.h>
#include <zvec/ailego/container/params.h>
#include "zvec/core/framework/index_framework.h"
#include "zvec/core/framework/index_meta.h"

using namespace zvec::core;
using namespace zvec::ailego;

TEST(KmeansCluster, General) {
  // Prepare index data
  const uint32_t count = 5000u;
  const uint32_t dimension = 33u;

  IndexMeta index_meta;
  index_meta.set_meta(IndexMeta::DataType::DT_FP32, dimension);
  index_meta.set_metric("SquaredEuclidean", 0, zvec::ailego::Params());

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
  // IndexCluster::Pointer cluster = std::make_shared<KmeansCluster>();
  IndexCluster::Pointer cluster = IndexFactory::CreateCluster("KmeansCluster");
  ASSERT_TRUE(!!cluster);

  zvec::ailego::Params params;
  params.set("zvec.general.cluster.count", 1);
  params.set("zvec.kmeans.cluster.count", 56);

  ASSERT_EQ(0, cluster->init(index_meta, params));
  ASSERT_EQ(0, cluster->mount(features));
  cluster->suggest(64u);

  auto threads = std::make_shared<SingleQueueIndexThreads>();

  std::cout << "---------- FIRST ----------\n";
  std::vector<IndexCluster::Centroid> centroids;
  std::vector<uint32_t> labels;
  ASSERT_NE(0, cluster->classify(threads, centroids));
  ASSERT_NE(0, cluster->label(threads, centroids, &labels));
  ASSERT_EQ(0, cluster->cluster(threads, centroids));

  for (const auto &it : centroids) {
    const auto &vec = it.vector<float>();

    std::cout << it.follows() << " (" << it.score() << ") { " << vec[0] << ", "
              << vec[1] << ", " << vec[2] << ", ... , " << vec[vec.size() - 2]
              << ", " << vec[vec.size() - 1] << " }" << std::endl;
    ASSERT_EQ(0u, it.similars().size());
  }

  std::cout << "---------- SECOND ----------\n";
  ASSERT_EQ(0, cluster->cluster(threads, centroids));

  for (const auto &it : centroids) {
    const auto &vec = it.vector<float>();

    std::cout << it.follows() << " (" << it.score() << ") { " << vec[0] << ", "
              << vec[1] << ", " << vec[2] << ", ... , " << vec[vec.size() - 2]
              << ", " << vec[vec.size() - 1] << " }" << std::endl;
    ASSERT_EQ(0u, it.similars().size());
  }

  std::cout << "---------- THIRD ----------\n";
  ASSERT_EQ(0, cluster->cluster(threads, centroids));

  for (const auto &it : centroids) {
    const auto &vec = it.vector<float>();

    std::cout << it.follows() << " (" << it.score() << ") { " << vec[0] << ", "
              << vec[1] << ", " << vec[2] << ", ... , " << vec[vec.size() - 2]
              << ", " << vec[vec.size() - 1] << " }" << std::endl;
    ASSERT_EQ(0u, it.similars().size());
  }

  ASSERT_EQ(0, cluster->classify(threads, centroids));
  ASSERT_EQ(0, cluster->label(threads, centroids, &labels));
}

TEST(KmeansCluster, RefreshesCentroidsBetweenIterations) {
  for (const char *name : {"KmeansCluster", "BatchKmeansCluster"}) {
    for (size_t count : {size_t{10}, size_t{32}, size_t{33}}) {
      SCOPED_TRACE(name);
      SCOPED_TRACE(count);
      IndexMeta meta;
      meta.set_meta(IndexMeta::DT_FP32, 1);
      meta.set_metric("SquaredEuclidean", 0, Params());
      auto features = std::make_shared<CompactIndexFeatures>(meta);
      for (float value : {0.0f, 2.0f, 3.0f, 10.0f}) {
        features->emplace(&value);
      }
      IndexCluster::CentroidList centroids;
      for (size_t i = 0; i < count; ++i) {
        float value = i == 0 ? 0.0f : i == 1 ? 3.0f : 100.0f * i;
        centroids.emplace_back(&value, sizeof(value));
        if (i >= 2) features->emplace(&value);
      }
      Params params;
      params.set("zvec.kmeans.cluster.max_iterations", 2);
      params.set("zvec.kmeans.cluster.epsilon", 0.0);
      auto cluster = IndexFactory::CreateCluster(name);
      ASSERT_TRUE(cluster);
      ASSERT_EQ(0, cluster->init(meta, params));
      ASSERT_EQ(0, cluster->mount(features));
      auto threads = std::make_shared<SingleQueueIndexThreads>(2, false);
      ASSERT_EQ(0, cluster->cluster(threads, centroids));
      ASSERT_EQ(count, centroids.size());
      // The first pass yields 0 and 5. Reusing that pass's input centers in
      // the second pass would keep those values instead of reaching 1/6.5.
      EXPECT_FLOAT_EQ(1.0f, centroids[0].vector<float>()[0]);
      EXPECT_FLOAT_EQ(6.5f, centroids[1].vector<float>()[0]);
      EXPECT_EQ(2u, centroids[0].follows());
      EXPECT_EQ(2u, centroids[1].follows());
      for (size_t i = 2; i < count; ++i) {
        EXPECT_FLOAT_EQ(100.0f * i, centroids[i].vector<float>()[0]);
      }
      std::vector<uint32_t> labels;
      ASSERT_EQ(0, cluster->label(threads, centroids, &labels));
      ASSERT_EQ(features->count(), labels.size());
      EXPECT_EQ((std::vector<uint32_t>{0, 0, 0, 1}),
                (std::vector<uint32_t>(labels.begin(), labels.begin() + 4)));
      for (size_t i = 2; i < count; ++i) EXPECT_EQ(i, labels[i + 2]);
      ASSERT_EQ(0, cluster->classify(threads, centroids));
      EXPECT_EQ(3u, centroids[0].similars().size());
      EXPECT_EQ(1u, centroids[1].similars().size());
    }
  }
}

TEST(KmeansCluster, PreservesCosineRecordsAndRebindsCenterOrder) {
  for (size_t count : {size_t{10}, size_t{32}, size_t{33}}) {
    SCOPED_TRACE(count);
    // Legacy cosine stores 256 normalized components followed by a norm.
    IndexMeta meta;
    meta.set_meta(IndexMeta::DT_FP32, 257);
    meta.set_metric("Cosine", 0, Params());
    auto features = std::make_shared<CompactIndexFeatures>(meta);
    IndexCluster::CentroidList centroids;
    for (size_t i = 0; i < count; ++i) {
      std::vector<float> row(257, 0.0f);
      row[i] = 0.8f;
      row[255] = 0.6f;
      row[256] = 9000.0f + i;
      centroids.emplace_back(row.data(), meta.element_size());
      for (float norm : {10.0f * i + 1.0f, 10.0f * i + 3.0f}) {
        row[256] = norm;
        features->emplace(row.data());
      }
    }
    Params params;
    params.set("zvec.kmeans.cluster.max_iterations", 2);
    params.set("zvec.kmeans.cluster.epsilon", 0.0);
    auto cluster = IndexFactory::CreateCluster("KmeansCluster");
    ASSERT_TRUE(cluster);
    ASSERT_EQ(0, cluster->init(meta, params));
    ASSERT_EQ(0, cluster->mount(features));
    auto threads = std::make_shared<SingleQueueIndexThreads>(2, false);
    ASSERT_EQ(0, cluster->cluster(threads, centroids));
    ASSERT_EQ(count, centroids.size());
    for (size_t i = 0; i < count; ++i) {
      ASSERT_EQ(meta.element_size(), centroids[i].size());
      EXPECT_FLOAT_EQ(0.8f, centroids[i].vector<float>()[i]);
      EXPECT_FLOAT_EQ(0.6f, centroids[i].vector<float>()[255]);
      EXPECT_FLOAT_EQ(10.0f * i + 2.0f, centroids[i].vector<float>()[256]);
    }
    std::swap(centroids.front(), centroids.back());
    std::vector<uint32_t> labels;
    ASSERT_EQ(0, cluster->label(threads, centroids, &labels));
    ASSERT_EQ(features->count(), labels.size());
    for (size_t i = 0; i < count; ++i) {
      const size_t label = i == 0 ? count - 1 : i == count - 1 ? 0 : i;
      EXPECT_EQ(label, labels[2 * i]);
      EXPECT_EQ(label, labels[2 * i + 1]);
    }
    ASSERT_EQ(0, cluster->classify(threads, centroids));
    for (size_t i = 0; i < count; ++i) {
      const auto &centroid = centroids[labels[2 * i]];
      ASSERT_EQ(2u, centroid.follows());
      const auto &similars = centroid.similars();
      ASSERT_EQ(2u, similars.size());
      for (size_t j = 0; j < 2; ++j) {
        EXPECT_NE(similars.end(), std::find(similars.begin(), similars.end(),
                                            features->element(2 * i + j)));
      }
    }
  }
}
