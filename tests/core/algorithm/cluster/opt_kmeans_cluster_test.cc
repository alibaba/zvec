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

#include <cmath>
#include <random>
#include <ailego/algorithm/kmeans.h>
#include <gtest/gtest.h>
#include <zvec/ailego/container/params.h>
#include "zvec/core/framework/index_framework.h"

using namespace zvec::core;
using namespace zvec::ailego;
using namespace zvec::ailego;

TEST(OptKmeansCluster, General) {
  // Prepare index data
  const uint32_t count = 5000u;
  const uint32_t dimension = 33u;

  IndexMeta index_meta;
  index_meta.set_meta(IndexMeta::DataType::DT_FP32, dimension);

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
  IndexCluster::Pointer cluster =
      IndexFactory::CreateCluster("OptKmeansCluster");
  ASSERT_TRUE(!!cluster);

  Params params;
  params.set("proxima.general.cluster.count", 1);
  params.set("proxima.optkmeans.cluster.count", 56);

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

// TEST(OptKmeansCluster, NoEmptyCentroids) {
//   // Prepare index data
//   const uint32_t count = 500u;
//   const uint32_t dimension = 8u;

//   IndexMeta index_meta;
//   index_meta.set_meta(IndexMeta::DataType::DT_FP32, dimension);
//   index_meta.set_metric("SquaredEuclidean", 0, Params());

//   std::shared_ptr<CompactIndexFeatures> features(
//       new CompactIndexFeatures(index_meta));

//   std::random_device rd;
//   std::mt19937 gen(rd());
//   std::uniform_real_distribution<float> dist(0.0, 5.0);

//   for (uint32_t i = 0; i < count; ++i) {
//     std::vector<float> vec(dimension);
//     for (size_t j = 0; j < dimension; ++j) {
//       vec[j] = dist(gen);
//     }
//     features->emplace(vec.data());
//   }

//   // Create a Kmeans cluster
//   IndexCluster::Pointer cluster =
//       IndexFactory::CreateCluster("OptKmeansCluster");
//   ASSERT_TRUE(!!cluster);

//   Params params;
//   ASSERT_EQ(0, cluster->init(index_meta, params));
//   ASSERT_EQ(0, cluster->mount(features));
//   cluster->suggest(20u);

//   auto threads = std::make_shared<SingleQueueIndexThreads>();
//   std::vector<IndexCluster::Centroid> centroids;
//   for (uint32_t i = 0; i < 3; ++i) {
//     std::vector<float> vec(dimension);
//     for (size_t j = 0; j < dimension; ++j) {
//       vec[j] = NAN;
//     }
//     centroids.emplace_back(vec.data(), vec.size() * sizeof(float));
//   }
//   ASSERT_EQ(0, cluster->cluster(threads, centroids));
//   ASSERT_EQ(3u, centroids.size());

//   for (uint32_t i = 0; i < 3; ++i) {
//     std::vector<float> vec(dimension);
//     for (size_t j = 0; j < dimension; ++j) {
//       vec[j] = dist(gen);
//     }
//     centroids.emplace_back(vec.data(), vec.size() * sizeof(float));
//   }
//   ASSERT_EQ(0, cluster->cluster(threads, centroids));
//   ASSERT_EQ(6u, centroids.size());

//   for (uint32_t i = 0; i < 3; ++i) {
//     std::vector<float> vec(dimension);
//     for (size_t j = 0; j < dimension; ++j) {
//       vec[j] = NAN;
//     }
//     centroids.emplace_back(vec.data(), vec.size() * sizeof(float));
//   }
//   ASSERT_EQ(0, cluster->cluster(threads, centroids));
//   ASSERT_EQ(9u, centroids.size());

//   for (uint32_t i = 0; i < 3; ++i) {
//     std::vector<float> vec(dimension);
//     for (size_t j = 0; j < dimension; ++j) {
//       vec[j] = dist(gen);
//     }
//     centroids.emplace_back(vec.data(), vec.size() * sizeof(float));
//   }
//   ASSERT_EQ(0, cluster->cluster(threads, centroids));
//   ASSERT_EQ(12u, centroids.size());

//   for (const auto &it : centroids) {
//     const auto &vec = it.vector<float>();

//     std::cout << it.follows() << " (" << it.score() << ") { " << vec[0] << ",
//     "
//               << vec[1] << ", " << vec[2] << ", ... , " << vec[vec.size() -
//               2]
//               << ", " << vec[vec.size() - 1] << " }" << std::endl;
//   }

//   params.set("proxima.optkmeans.cluster.purge_empty", true);
//   cluster->update(params);

//   ASSERT_EQ(12u, centroids.size());
//   ASSERT_EQ(0, cluster->cluster(threads, centroids));
//   ASSERT_EQ(7u, centroids.size());
//   for (const auto &it : centroids) {
//     const auto &vec = it.vector<float>();

//     std::cout << it.follows() << " (" << it.score() << ") { " << vec[0] << ",
//     "
//               << vec[1] << ", " << vec[2] << ", ... , " << vec[vec.size() -
//               2]
//               << ", " << vec[vec.size() - 1] << " }" << std::endl;
//   }
// }

TEST(OptKmeansCluster, BinaryGeneral) {
  // Prepare index data
  const uint32_t count = 5000u;
  const uint32_t dimension = 1024u;

  IndexMeta index_meta;
  index_meta.set_meta(IndexMeta::DataType::DT_BINARY32, dimension);
  index_meta.set_metric("SquaredEuclidean", 0, Params());

  std::shared_ptr<CompactIndexFeatures> features(
      new CompactIndexFeatures(index_meta));

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(0.0, 1.0);

  for (uint32_t i = 0; i < count; ++i) {
    BinaryVector<uint32_t> vec(dimension);
    for (size_t j = 0; j < dimension; ++j) {
      if (dist(gen) >= 0.5) {
        vec.set(j);
      }
    }
    features->emplace(vec.data());
  }

  std::cout << "---------- FIRST ----------\n";

  // Create a Kmeans cluster
  IndexCluster::Pointer cluster =
      IndexFactory::CreateCluster("OptKmeansCluster");
  ASSERT_TRUE(!!cluster);

  Params params;
  params.set("proxima.general.cluster.count", 1);
  params.set("proxima.optkmeans.cluster.count", 56);

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
    const auto &vec = it.vector<uint32_t>();

    uint mask = 0x1;
    std::cout << it.follows() << " (" << it.score() << ") { "
              << !!(vec[0] & mask) << ", " << !!(vec[0] & (mask << 1)) << ", "
              << !!(vec[0] & (mask << 2)) << ", ... , "
              << !!(vec[0] & (mask << !!(sizeof(uint32_t) - 2))) << ", "
              << !!(vec[0] & (mask << !!(sizeof(uint32_t) - 1))) << " }"
              << std::endl;
    ASSERT_EQ(0u, it.similars().size());
  }

  std::cout << "---------- SECOND ----------\n";
  ASSERT_EQ(0, cluster->cluster(threads, centroids));

  for (const auto &it : centroids) {
    const auto &vec = it.vector<uint32_t>();

    uint mask = 0x1;
    std::cout << it.follows() << " (" << it.score() << ") { "
              << !!(vec[0] & mask) << ", " << !!(vec[0] & (mask << 1)) << ", "
              << !!(vec[0] & (mask << 2)) << ", ... , "
              << !!(vec[0] & (mask << !!(sizeof(uint32_t) - 2))) << ", "
              << !!(vec[0] & (mask << !!(sizeof(uint32_t) - 1))) << " }"
              << std::endl;
    ASSERT_EQ(0u, it.similars().size());
  }

  std::cout << "---------- THIRD ----------\n";
  ASSERT_EQ(0, cluster->cluster(threads, centroids));

  for (const auto &it : centroids) {
    const auto &vec = it.vector<uint32_t>();

    uint mask = 0x1;
    std::cout << it.follows() << " (" << it.score() << ") { "
              << !!(vec[0] & mask) << ", " << !!(vec[0] & (mask << 1)) << ", "
              << !!(vec[0] & (mask << 2)) << ", ... , "
              << !!(vec[0] & (mask << !!(sizeof(uint32_t) - 2))) << ", "
              << !!(vec[0] & (mask << !!(sizeof(uint32_t) - 1))) << " }"
              << std::endl;
    ASSERT_EQ(0u, it.similars().size());
  }

  ASSERT_EQ(0, cluster->classify(threads, centroids));
  ASSERT_EQ(0, cluster->label(threads, centroids, &labels));
}


TEST(OptKmeansCluster, IN4General) {
  // Prepare index data
  const uint32_t count = 5000u;
  const uint32_t dimension = 64u;
  const uint32_t dimension_wrong = 66u;

  IndexMeta index_meta;
  index_meta.set_meta(IndexMeta::DataType::DT_INT4, dimension);
  index_meta.set_metric("SquaredEuclidean", 0, Params());

  IndexMeta index_meta_wrong;
  index_meta_wrong.set_meta(IndexMeta::DataType::DT_INT4, dimension_wrong);
  index_meta_wrong.set_metric("SquaredEuclidean", 0, Params());

  std::shared_ptr<CompactIndexFeatures> features(
      new CompactIndexFeatures(index_meta));

  std::shared_ptr<CompactIndexFeatures> features_wrong(
      new CompactIndexFeatures(index_meta_wrong));

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint8_t> dist(0, UINT8_MAX);

  for (uint32_t i = 0; i < count; ++i) {
    std::vector<uint8_t> vec(dimension / 2);
    std::vector<uint8_t> vec_wrong(dimension_wrong / 2);
    for (size_t j = 0; j < dimension / 2; ++j) {
      vec[j] = dist(gen);
    }
    for (size_t j = 0; j < dimension_wrong / 2; ++j) {
      vec_wrong[j] = dist(gen);
    }
    features->emplace(vec.data());
    features_wrong->emplace(vec_wrong.data());
  }

  // Create a OptKmeans cluster
  IndexCluster::Pointer cluster =
      IndexFactory::CreateCluster("OptKmeansCluster");
  ASSERT_TRUE(!!cluster);

  Params params;
  ASSERT_EQ(0, cluster->init(index_meta_wrong, params));
  ASSERT_NE(0, cluster->mount(features_wrong));

  params.set("proxima.general.cluster.count", 1);
  params.set("proxima.optkmeans.cluster.count", 56);

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


TEST(OptKmeansCluster, IN4Correctness) {
  // Prepare index data
  const uint32_t count = 5000u;
  const uint32_t dimension = 64u;

  IndexMeta index_meta1;
  index_meta1.set_meta(IndexMeta::DataType::DT_INT8, dimension);
  index_meta1.set_metric("SquaredEuclidean", 0, Params());

  IndexMeta index_meta2;
  index_meta2.set_meta(IndexMeta::DataType::DT_INT4, dimension);
  index_meta2.set_metric("SquaredEuclidean", 0, Params());

  std::shared_ptr<CompactIndexFeatures> features1(
      new CompactIndexFeatures(index_meta1));

  std::shared_ptr<CompactIndexFeatures> features2(
      new CompactIndexFeatures(index_meta2));

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(-8, 7);

  // Generate features
  for (size_t i = 0; i < count; ++i) {
    NumericalVector<int8_t> vec1(dimension);
    NibbleVector<int32_t> vec2(dimension);

    for (size_t j = 0; j < dimension; ++j) {
      int8_t val = (int8_t)dist(gen);
      vec1[j] = val;
      vec2.set(j, val);
    }
    features1->emplace(vec1.data());
    features2->emplace(vec2.data());
  }

  // Create a OptKmeans cluster of int8, and cluster only once
  IndexCluster::Pointer cluster_once =
      IndexFactory::CreateCluster("OptKmeansCluster");
  ASSERT_TRUE(!!cluster_once);

  Params params_once;
  params_once.set("proxima.general.cluster.count", 65);
  params_once.set("proxima.optkmeans.cluster.count", 63);
  params_once.set("proxima.optkmeans.cluster.max_iterations", 1);
  // Use KMC2 to init centroids
  params_once.set("proxima.optkmeans.cluster.markov_chain_length", 20);

  ASSERT_EQ(0, cluster_once->init(index_meta1, params_once));
  ASSERT_EQ(0, cluster_once->mount(features1));
  cluster_once->suggest(63);

  auto threads = std::make_shared<SingleQueueIndexThreads>();

  // Cluster once and get centroids
  std::vector<IndexCluster::Centroid> centroids1;
  ASSERT_EQ(0, cluster_once->cluster(threads, centroids1));

  // Use centroids_one as init centroids to both int8 and int4 cluster
  // Create a int8 cluster
  IndexCluster::Pointer cluster_int8 =
      IndexFactory::CreateCluster("OptKmeansCluster");
  ASSERT_TRUE(!!cluster_int8);

  Params params_int8;
  params_int8.set("proxima.general.cluster.count", 65);
  params_int8.set("proxima.optkmeans.cluster.count", 63);

  ASSERT_EQ(0, cluster_int8->init(index_meta1, params_int8));
  ASSERT_EQ(0, cluster_int8->mount(features1));
  cluster_int8->suggest(63u);

  // Create a int4 cluster
  IndexCluster::Pointer cluster_int4 =
      IndexFactory::CreateCluster("OptKmeansCluster");
  ASSERT_TRUE(!!cluster_int4);

  Params params_int4;
  params_int4.set("proxima.general.cluster.count", 65);
  params_int4.set("proxima.optkmeans.cluster.count", 63);

  ASSERT_EQ(0, cluster_int4->init(index_meta2, params_int4));
  ASSERT_EQ(0, cluster_int4->mount(features2));
  cluster_int4->suggest(63u);

  std::vector<IndexCluster::Centroid> centroids2;

  // Use centroids of int8 to init centroids of int4
  for (size_t i = 0; i < centroids1.size(); ++i) {
    NibbleVector<int8_t> nvec;
    nvec.assign(reinterpret_cast<const int8_t *>(centroids1[i].feature()),
                dimension);
    IndexCluster::Centroid curr_centroid;
    curr_centroid.set_score(centroids1[i].score());
    curr_centroid.set_follows(centroids1[i].follows());
    curr_centroid.set_feature(nvec.data(), nvec.dimension() >> 1);
    centroids2.push_back(curr_centroid);
  }

  ASSERT_EQ(0, cluster_int8->cluster(threads, centroids1));
  ASSERT_EQ(0, cluster_int4->cluster(threads, centroids2));

  EXPECT_EQ(centroids1.size(), centroids2.size());
  for (size_t i = 0; i < centroids1.size(); ++i) {
    EXPECT_EQ(centroids1[i].follows(), centroids2[i].follows());
    EXPECT_DOUBLE_EQ(centroids1[i].score(), centroids2[i].score());
  }
}

TEST(OptKmeansCluster, InnerProduct) {
  // Prepare index data
  const uint32_t count = 5000u;
  const uint32_t dimension = 33u;

  IndexMeta index_meta;
  index_meta.set_meta(IndexMeta::DataType::DT_FP32, dimension);
  index_meta.set_metric("InnerProduct", 0, Params());

  std::shared_ptr<CompactIndexFeatures> features(
      new CompactIndexFeatures(index_meta));

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-1.0, 1.0);

  for (uint32_t i = 0; i < count; ++i) {
    std::vector<float> vec(dimension);
    for (size_t j = 0; j < dimension; ++j) {
      vec[j] = dist(gen);
    }
    features->emplace(vec.data());
  }

  // Create a Kmeans cluster
  IndexCluster::Pointer cluster =
      IndexFactory::CreateCluster("OptKmeansCluster");
  ASSERT_TRUE(!!cluster);

  Params params;
  params.set("proxima.general.cluster.count", 1);
  params.set("proxima.optkmeans.cluster.count", 56);

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
