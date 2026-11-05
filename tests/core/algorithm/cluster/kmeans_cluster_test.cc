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

namespace {

constexpr size_t kCosDim = 64u;
constexpr size_t kCosClusters = 8u;

// Cosine converter layout: [unit components | original norm] (FP32 extra dim)
std::vector<float> MakeCosineFeature(std::mt19937 &gen) {
  std::normal_distribution<float> noise(0.0f, 1.0f);
  std::vector<float> vec(kCosDim + 1);
  double accum = 0.0;
  for (size_t j = 0; j < kCosDim; ++j) {
    vec[j] = noise(gen);
    accum += static_cast<double>(vec[j]) * vec[j];
  }
  const float norm = static_cast<float>(std::sqrt(accum));
  for (size_t j = 0; j < kCosDim; ++j) {
    vec[j] /= norm;
  }
  vec[kCosDim] = 4.25f;  // original norm metadata, ignored by the metric
  return vec;
}

IndexMeta MakeCosineMeta() {
  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, kCosDim + 1);
  meta.set_metric("Cosine", 0, zvec::ailego::Params());
  return meta;
}

float EffectiveNorm(const IndexCluster::Centroid &centroid) {
  const auto &vec = centroid.vector<float>();
  double accum = 0.0;
  for (size_t j = 0; j < kCosDim; ++j) {
    accum += static_cast<double>(vec[j]) * vec[j];
  }
  return static_cast<float>(std::sqrt(accum));
}

std::shared_ptr<CompactIndexFeatures> MakeCosineFeatures(
    const IndexMeta &meta, uint32_t count, uint32_t seed,
    std::vector<std::vector<float>> *raw) {
  auto features = std::make_shared<CompactIndexFeatures>(meta);
  std::mt19937 gen(seed);
  raw->reserve(count);
  for (uint32_t i = 0; i < count; ++i) {
    raw->push_back(MakeCosineFeature(gen));
    features->emplace(raw->back().data());
  }
  return features;
}

}  // namespace

TEST(KmeansCluster, CosineCentroidNormalization) {
  const uint32_t count = 2000u;
  IndexMeta index_meta = MakeCosineMeta();

  std::vector<std::vector<float>> raw;
  auto features = MakeCosineFeatures(index_meta, count, 7u, &raw);

  IndexCluster::Pointer cluster = IndexFactory::CreateCluster("KmeansCluster");
  ASSERT_TRUE(!!cluster);

  zvec::ailego::Params params;
  params.set("zvec.kmeans.cluster.count", kCosClusters);
  params.set("zvec.kmeans.cluster.max_iterations", 10);
  params.set("zvec.kmeans.cluster.epsilon", 0.0);

  ASSERT_EQ(0, cluster->init(index_meta, params));
  ASSERT_EQ(0, cluster->mount(features));

  // Fixed initial centroids keep the run deterministic (init_centroids
  // samples with a random device otherwise).
  std::vector<IndexCluster::Centroid> cents;
  for (size_t k = 0; k < kCosClusters; ++k) {
    cents.emplace_back(raw[k].data(), (kCosDim + 1) * sizeof(float));
  }

  auto threads = std::make_shared<SingleQueueIndexThreads>();
  ASSERT_EQ(0, cluster->cluster(threads, cents));

  size_t follows_total = 0u, non_empty = 0u;
  for (const auto &it : cents) {
    const auto &vec = it.vector<float>();
    ASSERT_EQ(kCosDim + 1u, vec.size());
    for (size_t j = 0; j < vec.size(); ++j) {
      ASSERT_TRUE(std::isfinite(vec[j])) << "dim " << j;
    }

    follows_total += it.follows();
    if (it.follows() == 0u) {
      continue;
    }
    ++non_empty;

    // The arithmetic mean of unit features is shorter than 1; every update
    // must re-normalize the metric-visible part ...
    EXPECT_NEAR(1.0f, EffectiveNorm(it), 1e-4f);
    // ... and write the unit norm into the trailing norm metadata.
    EXPECT_EQ(1.0f, vec[kCosDim]);
  }

  EXPECT_EQ(static_cast<size_t>(count), follows_total);
  EXPECT_GT(non_empty, 0u);
}

TEST(KmeansCluster, CosineEmptyClusterKeepsPreviousCentroid) {
  const uint32_t count = 500u;
  IndexMeta index_meta = MakeCosineMeta();

  std::vector<std::vector<float>> raw;
  auto features = MakeCosineFeatures(index_meta, count, 11u, &raw);

  IndexCluster::Pointer cluster = IndexFactory::CreateCluster("KmeansCluster");
  ASSERT_TRUE(!!cluster);

  zvec::ailego::Params params;
  params.set("zvec.kmeans.cluster.count", 2);
  // A single iteration: with identical initial centroids the seeker's
  // strict-less-than keeps every feature in cluster 0, so cluster 1 has a
  // zero-count accumulator when its centroid is updated.
  params.set("zvec.kmeans.cluster.max_iterations", 1);
  params.set("zvec.kmeans.cluster.epsilon", 0.0);

  ASSERT_EQ(0, cluster->init(index_meta, params));
  ASSERT_EQ(0, cluster->mount(features));

  std::vector<IndexCluster::Centroid> cents;
  cents.emplace_back(raw[0].data(), (kCosDim + 1) * sizeof(float));
  cents.emplace_back(raw[0].data(), (kCosDim + 1) * sizeof(float));
  const std::string initial_bytes = cents[1].buffer();

  auto threads = std::make_shared<SingleQueueIndexThreads>();
  ASSERT_EQ(0, cluster->cluster(threads, cents));

  ASSERT_EQ(2u, cents.size());
  ASSERT_EQ(0u, cents[1].follows());
  // An empty cluster must keep its previous feature instead of collapsing
  // the accumulator mean 0/0 to NaN.
  EXPECT_EQ(initial_bytes, cents[1].buffer());
  const auto &kept = cents[1].vector<float>();
  for (size_t j = 0; j < kept.size(); ++j) {
    EXPECT_TRUE(std::isfinite(kept[j])) << "dim " << j;
  }
  EXPECT_NEAR(1.0f, EffectiveNorm(cents[1]), 1e-4f);

  EXPECT_LT(0u, cents[0].follows());
  EXPECT_NEAR(1.0f, EffectiveNorm(cents[0]), 1e-4f);
}

TEST(KmeansCluster, CosineCentroidOrderingDeterministic) {
  const uint32_t count = 1000u;
  IndexMeta index_meta = MakeCosineMeta();

  std::vector<std::vector<float>> raw;
  auto features = MakeCosineFeatures(index_meta, count, 23u, &raw);

  std::vector<IndexCluster::Centroid> initial;
  for (size_t k = 0; k < kCosClusters; ++k) {
    initial.emplace_back(raw[k].data(), (kCosDim + 1) * sizeof(float));
  }

  std::vector<std::vector<IndexCluster::Centroid>> rounds;
  std::vector<std::vector<uint32_t>> labels;
  for (int round = 0; round < 2; ++round) {
    IndexCluster::Pointer cluster =
        IndexFactory::CreateCluster("KmeansCluster");
    ASSERT_TRUE(!!cluster);

    zvec::ailego::Params params;
    params.set("zvec.kmeans.cluster.count", kCosClusters);
    params.set("zvec.kmeans.cluster.max_iterations", 10);
    params.set("zvec.kmeans.cluster.epsilon", 0.0);

    ASSERT_EQ(0, cluster->init(index_meta, params));
    ASSERT_EQ(0, cluster->mount(features));

    std::vector<IndexCluster::Centroid> cents = initial;
    auto threads = std::make_shared<SingleQueueIndexThreads>();
    ASSERT_EQ(0, cluster->cluster(threads, cents));

    std::vector<uint32_t> assigned;
    ASSERT_EQ(0, cluster->label(threads, cents, &assigned));

    rounds.push_back(std::move(cents));
    labels.push_back(std::move(assigned));
  }

  // Same features and initial centroids must give the same centroid order
  // (the inverted-list assignment depends on it), buffers and labels.
  ASSERT_EQ(rounds[0].size(), rounds[1].size());
  for (size_t k = 0; k < rounds[0].size(); ++k) {
    EXPECT_EQ(rounds[0][k].buffer(), rounds[1][k].buffer()) << "centroid " << k;
    EXPECT_EQ(rounds[0][k].follows(), rounds[1][k].follows())
        << "centroid " << k;
  }
  EXPECT_EQ(labels[0], labels[1]);
}
