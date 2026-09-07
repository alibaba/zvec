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
#include <limits>
#include <random>
#include <ailego/algorithm/kmeans.h>
#include <gtest/gtest.h>
#include <zvec/ailego/container/params.h>
#include "algorithm/cluster/cluster_params.h"
#include "algorithm/cluster/holder_cluster.h"
#include "zvec/core/framework/index_framework.h"

using namespace zvec::core;
using namespace zvec::ailego;
using namespace zvec::ailego;

namespace {

// Generate vectors on demand in one reused, deliberately unaligned buffer.
// There is no backing full corpus whose pointers the streaming path can retain.
class StreamingTestHolder : public IndexHolder {
 public:
  StreamingTestHolder(size_t dim, size_t count)
      : meta_(IndexMeta::DT_FP32, dim),
        actual_count(count),
        reported_count(count) {}

  class Iterator : public IndexHolder::Iterator {
   public:
    explicit Iterator(StreamingTestHolder *owner)
        : owner_(owner), buffer_(owner->element_size() + 1) {
      ++owner_->live_iterators;
    }
    ~Iterator() override {
      --owner_->live_iterators;
    }
    const void *data() const override {
      ++owner_->reads;
      if (index_ == owner_->null_at) return nullptr;
      for (size_t d = 0; d < owner_->dimension(); ++d) {
        const float value = static_cast<float>((index_ % 3) * 20) +
                            static_cast<float>((index_ / 3) % 7) * 0.125f +
                            static_cast<float>(d) * 0.25f + owner_->offset;
        std::memcpy(buffer_.data() + 1 + d * sizeof(value), &value,
                    sizeof(value));
      }
      return buffer_.data() + 1;
    }
    bool is_valid() const override {
      return index_ < owner_->actual_count;
    }
    uint64_t key() const override {
      return 1000000 - index_;
    }
    void next() override {
      std::fill(buffer_.begin(), buffer_.end(), '\xff');
      ++index_;
    }

   private:
    StreamingTestHolder *owner_;
    mutable std::vector<char> buffer_;
    size_t index_{0};
  };

  size_t count() const override {
    return reported_count;
  }
  size_t dimension() const override {
    return meta_.dimension();
  }
  IndexMeta::DataType data_type() const override {
    return meta_.data_type();
  }
  size_t element_size() const override {
    return meta_.element_size();
  }
  bool multipass() const override {
    return true;
  }
  IndexHolder::Iterator::Pointer create_iterator() override {
    ++iterations;
    if (fail_iterator) return nullptr;
    return IndexHolder::Iterator::Pointer(new Iterator(this));
  }

  IndexMeta meta_;
  size_t actual_count;
  size_t reported_count;
  size_t iterations{0};
  size_t reads{0};
  size_t live_iterators{0};
  size_t null_at{std::numeric_limits<size_t>::max()};
  bool fail_iterator{false};
  float offset{0};
};

CompactIndexFeatures::Pointer Materialize(StreamingTestHolder &holder) {
  auto features = std::make_shared<CompactIndexFeatures>(holder.meta_);
  features->reserve(holder.actual_count);
  for (auto iter = holder.create_iterator(); iter->is_valid(); iter->next()) {
    features->emplace(iter->data());
  }
  return features;
}

void ExpectSameCentroids(const IndexCluster::CentroidList &expected,
                         const IndexCluster::CentroidList &actual) {
  ASSERT_EQ(expected.size(), actual.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(expected[i].buffer(), actual[i].buffer());
    EXPECT_DOUBLE_EQ(expected[i].score(), actual[i].score());
    EXPECT_EQ(expected[i].follows(), actual[i].follows());
    EXPECT_TRUE(actual[i].similars().empty());
  }
}

}  // namespace

TEST(OptKmeansCluster, HolderTrainingMatchesMountedFeatures) {
  auto threads = std::make_shared<SingleQueueIndexThreads>(1, false);
  for (const size_t dim : {1u, 7u, 33u, 128u}) {
    for (const size_t count : {1u, 17u, 259u}) {
      SCOPED_TRACE(::testing::Message() << "dim=" << dim << " count=" << count);
      auto holder = std::make_shared<StreamingTestHolder>(dim, count);
      auto features = Materialize(*holder);
      auto mounted = IndexFactory::CreateCluster("OptKmeansCluster");
      auto streamed = IndexFactory::CreateCluster("OptKmeansCluster");
      ASSERT_NE(nullptr, mounted);
      ASSERT_NE(nullptr, streamed);
      ASSERT_EQ(0, mounted->init(holder->meta_, Params()));
      ASSERT_EQ(0, streamed->init(holder->meta_, Params()));
      ASSERT_EQ(0, mounted->mount(features));
      auto fast = dynamic_cast<HolderCluster *>(streamed.get());
      ASSERT_NE(nullptr, fast);
      // Identical explicit seeds avoid random initialization in this
      // comparison.
      IndexCluster::CentroidList seeds;
      for (size_t i = 0; i < std::min<size_t>(3, count); ++i) {
        seeds.emplace_back(features->element(i), features->element_size());
      }
      auto expected = seeds;
      ASSERT_EQ(0, mounted->cluster(threads, expected));
      const size_t before = holder->iterations;
      for (int repeat = 0; repeat < 2; ++repeat) {
        auto actual = seeds;
        ASSERT_EQ(0, fast->cluster_holder(threads, holder, actual));
        ExpectSameCentroids(expected, actual);
        EXPECT_EQ(0u, holder->live_iterators);
      }
      EXPECT_EQ(before + 2, holder->iterations);
      EXPECT_EQ(count * 3, holder->reads);
      // One-shot input is not implicitly mounted or retained by the cluster.
      auto actual = seeds;
      EXPECT_EQ(IndexError_NoReady, streamed->cluster(threads, actual));
      std::weak_ptr<IndexHolder> input_lifetime = holder;
      holder.reset();
      EXPECT_TRUE(input_lifetime.expired());
    }
  }
}

TEST(OptKmeansCluster, HolderTrainingPreservesExistingMount) {
  auto holder = std::make_shared<StreamingTestHolder>(7, 259);
  auto features = Materialize(*holder);
  auto cluster = IndexFactory::CreateCluster("OptKmeansCluster");
  ASSERT_NE(nullptr, cluster);
  ASSERT_EQ(0, cluster->init(holder->meta_, Params()));
  ASSERT_EQ(0, cluster->mount(features));
  auto fast = dynamic_cast<HolderCluster *>(cluster.get());
  ASSERT_NE(nullptr, fast);
  auto threads = std::make_shared<SingleQueueIndexThreads>(1, false);
  IndexCluster::CentroidList seed{
      IndexCluster::Centroid(features->element(0), features->element_size())};
  auto expected = seed;
  ASSERT_EQ(0, cluster->cluster(threads, expected));
  holder->offset = 1000;
  auto streamed = seed;
  ASSERT_EQ(0, fast->cluster_holder(threads, holder, streamed));
  EXPECT_NE(expected[0].buffer(), streamed[0].buffer());
  auto actual = seed;
  ASSERT_EQ(0, cluster->cluster(threads, actual));
  ExpectSameCentroids(expected, actual);
}

TEST(OptKmeansCluster, HolderTrainingRejectsMalformedInputWithoutRetry) {
  auto holder = std::make_shared<StreamingTestHolder>(7, 17);
  auto cluster = IndexFactory::CreateCluster("OptKmeansCluster");
  ASSERT_NE(nullptr, cluster);
  ASSERT_EQ(0, cluster->init(holder->meta_, Params()));
  cluster->suggest(2);
  auto fast = dynamic_cast<HolderCluster *>(cluster.get());
  ASSERT_NE(nullptr, fast);
  auto threads = std::make_shared<SingleQueueIndexThreads>(1, false);
  IndexCluster::CentroidList cents;
  EXPECT_EQ(IndexError_InvalidArgument,
            fast->cluster_holder(threads, nullptr, cents));
  for (int fault = 0; fault < 4; ++fault) {
    SCOPED_TRACE(fault);
    holder->fail_iterator = fault == 0;
    holder->null_at = fault == 1 ? 3 : std::numeric_limits<size_t>::max();
    holder->reported_count = fault == 2 ? 16 : (fault == 3 ? 18 : 17);
    const size_t before = holder->iterations;
    EXPECT_NE(0, fast->cluster_holder(threads, holder, cents));
    EXPECT_EQ(before + 1, holder->iterations);
    EXPECT_EQ(0u, holder->live_iterators);
    EXPECT_TRUE(cents.empty());
  }
  holder->reported_count = 17;
  ASSERT_EQ(0, fast->cluster_holder(threads, holder, cents));
  EXPECT_EQ(2u, cents.size());
  auto wrong = std::make_shared<StreamingTestHolder>(8, 17);
  EXPECT_EQ(IndexError_Mismatch, fast->cluster_holder(threads, wrong, cents));
  EXPECT_EQ(0u, wrong->iterations);
  auto empty = std::make_shared<StreamingTestHolder>(7, 0);
  EXPECT_EQ(IndexError_InvalidArgument,
            fast->cluster_holder(threads, empty, cents));
}

TEST(OptKmeansCluster, UnsupportedHolderTrainingDoesNotConsumeInput) {
  auto holder = std::make_shared<StreamingTestHolder>(7, 17);
  for (int unsupported = 0; unsupported < 4; ++unsupported) {
    SCOPED_TRACE(unsupported);
    IndexMeta meta = holder->meta_;
    holder->reported_count = 17;
    if (unsupported == 0) meta.set_meta(IndexMeta::DT_FP16, 7);
    if (unsupported == 1) meta.set_metric("InnerProduct", 0, Params());
    if (unsupported == 2) holder->reported_count = static_cast<size_t>(-1);
    if (unsupported == 3) meta.set_meta(IndexMeta::DT_FP64, 7);
    auto cluster = IndexFactory::CreateCluster("OptKmeansCluster");
    ASSERT_NE(nullptr, cluster);
    ASSERT_EQ(0, cluster->init(meta, Params()));
    auto fast = dynamic_cast<HolderCluster *>(cluster.get());
    ASSERT_NE(nullptr, fast);
    IndexCluster::CentroidList cents;
    EXPECT_EQ(IndexError_NotImplemented,
              fast->cluster_holder(nullptr, holder, cents));
    EXPECT_EQ(0u, holder->iterations);
    EXPECT_TRUE(cents.empty());
  }
}

TEST(OptKmeansCluster, TrainerStreamsOrFallsBackAsAppropriate) {
  auto threads = std::make_shared<SingleQueueIndexThreads>(1, false);
  for (int mode = 0; mode < 6; ++mode) {
    SCOPED_TRACE(mode);
    auto holder = std::make_shared<StreamingTestHolder>(7, 259);
    auto trainer = IndexFactory::CreateTrainer("StratifiedClusterTrainer");
    ASSERT_NE(nullptr, trainer);
    Params params;
    params.set(STRATIFIED_TRAINER_CLUSTER_COUNT, mode == 3 ? "2*2" : "2");
    params.set(STRATIFIED_TRAINER_CLASS_NAME,
               mode == 2 ? "KmeansCluster" : "OptKmeansCluster");
    if (mode == 1) params.set(STRATIFIED_TRAINER_SAMPLE_COUNT, 32u);
    if (mode == 4) holder->reported_count = static_cast<size_t>(-1);
    if (mode == 5) holder->meta_.set_metric("InnerProduct", 0, Params());
    ASSERT_EQ(0, trainer->init(holder->meta_, params));
    for (int repeat = 0; repeat < 2; ++repeat) {
      if (mode == 4 && repeat == 1) {
        // Reuse a trainer after fallback materialized an unknown-size holder.
        holder->reported_count = holder->actual_count;
      }
      ASSERT_EQ(0, trainer->train(threads, holder));
      EXPECT_EQ(mode == 1 ? 32u : holder->count(),
                trainer->stats().trained_count());
      EXPECT_EQ(0u, trainer->stats().discarded_count());
      IndexCluster::CentroidList cents;
      ASSERT_EQ(0, IndexCluster::Deserialize(trainer->meta(),
                                             trainer->indexes(), &cents));
      ASSERT_FALSE(cents.empty());
      size_t follows = 0;
      for (const auto &centroid : cents) follows += centroid.follows();
      EXPECT_EQ(mode == 1 ? 32u : holder->actual_count, follows);
    }
    EXPECT_EQ(2u, holder->iterations);
    EXPECT_EQ(0u, holder->live_iterators);
  }
}

TEST(OptKmeansCluster, TrainerPropagatesStreamingReadFailure) {
  auto holder = std::make_shared<StreamingTestHolder>(7, 17);
  holder->reported_count = 18;
  auto trainer = IndexFactory::CreateTrainer("StratifiedClusterTrainer");
  ASSERT_NE(nullptr, trainer);
  Params params;
  params.set(STRATIFIED_TRAINER_CLUSTER_COUNT, "2");
  params.set(STRATIFIED_TRAINER_CLASS_NAME, "OptKmeansCluster");
  ASSERT_EQ(0, trainer->init(holder->meta_, params));
  auto threads = std::make_shared<SingleQueueIndexThreads>(1, false);
  EXPECT_EQ(IndexError_InvalidArgument, trainer->train(threads, holder));
  EXPECT_EQ(1u, holder->iterations);
  EXPECT_EQ(0u, holder->live_iterators);
  holder->reported_count = 17;
  EXPECT_EQ(0, trainer->train(threads, holder));
  EXPECT_EQ(17u, trainer->stats().trained_count());
}

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
  params.set("zvec.general.cluster.count", 1);
  params.set("zvec.optkmeans.cluster.count", 56);

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

//   params.set("zvec.optkmeans.cluster.purge_empty", true);
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
  std::uniform_int_distribution<unsigned short> dist(0, UINT8_MAX);

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

  params.set("zvec.general.cluster.count", 1);
  params.set("zvec.optkmeans.cluster.count", 56);

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
  params_once.set("zvec.general.cluster.count", 65);
  params_once.set("zvec.optkmeans.cluster.count", 63);
  params_once.set("zvec.optkmeans.cluster.max_iterations", 1);
  // Use KMC2 to init centroids
  params_once.set("zvec.optkmeans.cluster.markov_chain_length", 20);

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
  params_int8.set("zvec.general.cluster.count", 65);
  params_int8.set("zvec.optkmeans.cluster.count", 63);

  ASSERT_EQ(0, cluster_int8->init(index_meta1, params_int8));
  ASSERT_EQ(0, cluster_int8->mount(features1));
  cluster_int8->suggest(63u);

  // Create a int4 cluster
  IndexCluster::Pointer cluster_int4 =
      IndexFactory::CreateCluster("OptKmeansCluster");
  ASSERT_TRUE(!!cluster_int4);

  Params params_int4;
  params_int4.set("zvec.general.cluster.count", 65);
  params_int4.set("zvec.optkmeans.cluster.count", 63);

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
  params.set("zvec.general.cluster.count", 1);
  params.set("zvec.optkmeans.cluster.count", 56);

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
