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
#include <cstring>
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
template <typename T>
void CheckCosineLabels(IndexMeta::DataType type) {
  for (const size_t dim : {256u, 512u}) {
    for (const size_t count : {10u, 32u, 33u}) {
      SCOPED_TRACE(::testing::Message() << "type=" << type << " dim=" << dim
                                        << " centers=" << count);
      // Legacy cosine records contain normalized components and a float norm
      // tail, which must not be treated as another component by Turbo.
      IndexMeta meta;
      meta.set_meta(type, dim + sizeof(float) / IndexMeta::UnitSizeof(type));
      meta.set_metric("Cosine", 0, Params());
      auto features = std::make_shared<CompactIndexFeatures>(meta);
      IndexCluster::CentroidList centroids;
      for (size_t i = 0; i <= count; ++i) {
        std::vector<T> row(meta.dimension(), T(0.0f));
        if (i < count) {
          row[i] = T(1.0f);
        } else {
          // An equal-distance query must keep the first center.
          row[0] = row[1] = T(std::sqrt(0.5f));
        }
        const float norm = 2.0f + static_cast<float>(i);
        std::memcpy(reinterpret_cast<char *>(row.data()) + dim * sizeof(T),
                    &norm, sizeof(norm));
        features->emplace(row.data());
        if (i < count) centroids.emplace_back(row.data(), meta.element_size());
      }
      auto cluster = IndexFactory::CreateCluster("KmeansCluster");
      ASSERT_NE(nullptr, cluster);
      ASSERT_EQ(0, cluster->init(meta, Params()));
      ASSERT_EQ(0, cluster->mount(features));
      auto threads = std::make_shared<SingleQueueIndexThreads>(2u, false);
      std::vector<uint32_t> labels;
      for (unsigned pass = 0; pass < 2; ++pass) {
        ASSERT_EQ(0, cluster->label(threads, centroids, &labels));
        ASSERT_EQ(count + 1, labels.size());
        for (size_t i = 0; i < count; ++i) {
          const size_t expected =
              pass && (i == 0 || i == count - 1) ? count - 1 - i : i;
          EXPECT_EQ(expected, labels[i]);
        }
        EXPECT_EQ(pass ? 1u : 0u, labels.back());
        // A new labeling pass must observe the updated centers.
        std::swap(centroids.front(), centroids.back());
      }
    }
  }
}
}  // namespace

TEST(KmeansCluster, TurboCosineLabelsAcrossBatchBoundary) {
  CheckCosineLabels<float>(IndexMeta::DT_FP32);
  CheckCosineLabels<Float16>(IndexMeta::DT_FP16);
}
