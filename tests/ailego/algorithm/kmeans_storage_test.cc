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

#include <atomic>
#include <cstring>
#include <limits>
#include <string>
#include <ailego/algorithm/kmeans.h>
#include <gtest/gtest.h>

namespace zvec::ailego {
namespace {

class TestMatrixStorage : public LloydClusterMatrixStorage {
 public:
  int read(size_t offset, void *out, size_t bytes) const override {
    std::lock_guard<std::mutex> lock(mutex);
    const size_t call = reads.fetch_add(1);
    if (call >= fail_read_at) return -73;
    if (offset > data.size() || bytes > data.size() - offset) return -74;
    std::memcpy(out, data.data() + offset, bytes);
    return 0;
  }
  int write(size_t offset, const void *input, size_t bytes) override {
    std::lock_guard<std::mutex> lock(mutex);
    if (writes++ >= fail_write_at) return -75;
    largest_write = std::max(largest_write, bytes);
    data.resize(std::max(data.size(), offset + bytes));
    std::memcpy(data.data() + offset, input, bytes);
    return 0;
  }

  std::string data;
  mutable std::mutex mutex;
  mutable std::atomic<size_t> reads{0};
  size_t writes{0};
  size_t largest_write{0};
  size_t fail_read_at{std::numeric_limits<size_t>::max()};
  size_t fail_write_at{std::numeric_limits<size_t>::max()};
};

template <typename Algorithm, typename Fill>
void CheckMatrixEquivalence(Fill fill) {
  using Container = typename Algorithm::ContainerType;
  constexpr size_t dim = 8;
  for (size_t count : {size_t{1}, size_t{17}, size_t{259}}) {
    SCOPED_TRACE(count);
    const size_t k = std::min(count, size_t{65});
    Algorithm memory(k, dim);
    Algorithm external(k, dim);
    auto storage = std::make_shared<TestMatrixStorage>();
    auto center_storage = std::make_shared<TestMatrixStorage>();
    auto accum_storage = std::make_shared<TestMatrixStorage>();
    external.set_feature_matrix_storage(storage);
    external.set_centroid_matrix_storage(center_storage);
    external.set_accumulator_storage(accum_storage);
    memory.feature_matrix_reserve(count);
    external.feature_matrix_reserve(count);
    Container row(dim);
    row.resize(1);
    for (size_t i = 0; i < count; ++i) {
      fill(i % k, row.data());
      memory.append(row.data(), dim);
      external.append(row.data(), dim);
    }
    EXPECT_EQ(0u, external.feature_matrix().bytes());
    EXPECT_EQ(memory.feature_matrix_count(), external.feature_matrix_count());
    EXPECT_EQ(memory.feature_cache().bytes(), external.feature_cache().bytes());
    EXPECT_EQ(memory.feature_matrix().bytes(), storage->data.size());
    EXPECT_LE(storage->largest_write, Algorithm::BatchCount * row.bytes());
    for (size_t i = 0; i < k; ++i) {
      fill(i, row.data());
      memory.mutable_centroids()->append(row.data(), dim);
      external.mutable_centroids()->append(row.data(), dim);
    }
    ThreadPool pool(1, false);
    for (int iteration = 0; iteration < 3; ++iteration) {
      double expected = 0;
      double actual = 0;
      ASSERT_TRUE(memory.cluster_once(pool, &expected));
      ASSERT_TRUE(external.cluster_once(pool, &actual));
      EXPECT_DOUBLE_EQ(expected, actual);
      ASSERT_EQ(memory.centroids().bytes(), external.centroids().bytes());
      EXPECT_EQ(
          0, std::memcmp(memory.centroids().data(), external.centroids().data(),
                         memory.centroids().bytes()));
      EXPECT_EQ(0u, external.centroid_matrix().bytes());
      EXPECT_EQ(memory.centroid_matrix().bytes(), center_storage->data.size());
      EXPECT_LE(center_storage->largest_write,
                Algorithm::BatchCount * row.bytes());
      size_t follows = 0;
      for (const auto &item : external.context().clusters())
        follows += item.count();
      EXPECT_EQ(count, follows);
      EXPECT_GT(accum_storage->writes, 0u);
      EXPECT_LE(accum_storage->largest_write, dim * sizeof(double));
    }
  }
}

template <typename T>
void CheckNumeric() {
  auto fill = [](size_t i, T *out) {
    for (size_t d = 0; d < 8; ++d) {
      // Equal-norm distinct seeds keep all clusters populated for both L2
      // and IP, including integer types (empty numeric clusters use NaN).
      out[d] = static_cast<T>((i & (size_t{1} << d)) ? 1.0f : -1.0f);
    }
  };
  CheckMatrixEquivalence<NumericalKmeans<T, ThreadPool>>(fill);
  CheckMatrixEquivalence<NumericalInnerProductKmeans<T, ThreadPool>>(fill);
}

TEST(KmeansMatrixStorage, NumericFullAndPartialBlocksMatchMemory) {
  CheckNumeric<float>();
  CheckNumeric<Float16>();
  CheckNumeric<int8_t>();
  CheckNumeric<double>();
  CheckNumeric<int16_t>();
}

TEST(KmeansMatrixStorage, PackedInt4FullAndPartialBlocksMatchMemory) {
  auto fill = [](size_t i, uint32_t *out) {
    NibbleVector<int32_t> row(8);
    for (size_t d = 0; d < 8; ++d) {
      row.set(d, (i & (size_t{1} << d)) ? 1 : -1);
    }
    std::memcpy(out, row.data(), 4);
  };
  CheckMatrixEquivalence<NibbleKmeans<int32_t, ThreadPool>>(fill);
  CheckMatrixEquivalence<NibbleInnerProductKmeans<int32_t, ThreadPool>>(fill);
}

using Algorithm = NumericalKmeans<float, ThreadPool>;

void Populate(Algorithm *algorithm) {
  for (size_t i = 0; i < 128; ++i) {
    const float row[] = {float(i % 3), float(i % 5), float(i % 7)};
    algorithm->append(row, 3);
  }
}

TEST(KmeansMatrixStorage, InitializersReadExternalBlocks) {
  ThreadPool pool(2, false);
  for (int mode = 0; mode < 4; ++mode) {
    SCOPED_TRACE(mode);
    Algorithm algorithm(3, 3);
    auto storage = std::make_shared<TestMatrixStorage>();
    algorithm.set_feature_matrix_storage(storage);
    Populate(&algorithm);
    if (mode == 0) {
      algorithm.init_centroids(pool);
    } else {
      Kmc2CentroidsGenerator<Algorithm, ThreadPool> generator;
      generator.set_chain_length(mode == 1 ? 0 : 8);
      generator.set_assumption_free(mode == 3);
      algorithm.init_centroids(pool, generator);
    }
    ASSERT_EQ(0, algorithm.matrix_status());
    ASSERT_EQ(3u, algorithm.centroids().count());
    EXPECT_GT(storage->reads.load(), 0u);
    double cost = 0;
    EXPECT_TRUE(algorithm.cluster_once(pool, &cost));
  }
}

TEST(KmeansMatrixStorage, InitializersPropagateReadFailure) {
  ThreadPool pool(2, false);
  for (int mode = 0; mode < 4; ++mode) {
    for (size_t fail_at : {size_t{0}, size_t{1}}) {
      SCOPED_TRACE(::testing::Message() << mode << ":" << fail_at);
      Algorithm algorithm(3, 3);
      auto storage = std::make_shared<TestMatrixStorage>();
      algorithm.set_feature_matrix_storage(storage);
      Populate(&algorithm);
      storage->fail_read_at = fail_at;
      if (mode == 0) {
        algorithm.init_centroids(pool);
      } else {
        Kmc2CentroidsGenerator<Algorithm, ThreadPool> generator;
        generator.set_chain_length(mode == 1 ? 0 : 8);
        generator.set_assumption_free(mode == 3);
        algorithm.init_centroids(pool, generator);
      }
      EXPECT_EQ(-73, algorithm.matrix_status());
      double untouched = 123;
      EXPECT_FALSE(algorithm.cluster_once(pool, &untouched));
      EXPECT_EQ(123, untouched);
    }
  }
}

TEST(KmeansMatrixStorage, WorkerReadFailureDoesNotPublishPartialCentroids) {
  Algorithm algorithm(1, 3);
  auto storage = std::make_shared<TestMatrixStorage>();
  algorithm.set_feature_matrix_storage(storage);
  Populate(&algorithm);
  const float seed[] = {1, 2, 3};
  algorithm.mutable_centroids()->append(seed, 3);
  storage->fail_read_at = 1;
  ThreadPool pool(3, false);
  double untouched = 123;
  EXPECT_FALSE(algorithm.cluster_once(pool, &untouched));
  EXPECT_EQ(-73, algorithm.matrix_status());
  EXPECT_EQ(123, untouched);
  EXPECT_EQ(0, std::memcmp(seed, algorithm.centroids().data(), sizeof(seed)));
}

TEST(KmeansMatrixStorage, CentroidBackingErrorsDoNotPublishAndResetClearsThem) {
  for (bool read_error : {true, false}) {
    for (size_t fail_at : {size_t{0}, size_t{1}}) {
      Algorithm algorithm(65, 3);
      auto storage = std::make_shared<TestMatrixStorage>();
      algorithm.set_centroid_matrix_storage(storage);
      Populate(&algorithm);
      for (size_t i = 0; i < 65; ++i) {
        const float seed[] = {float(i % 3), float(i % 5), float(i % 7)};
        algorithm.mutable_centroids()->append(seed, 3);
      }
      auto before = algorithm.centroids();
      if (read_error)
        storage->fail_read_at = fail_at;
      else
        storage->fail_write_at = fail_at;
      ThreadPool pool(3, false);
      double untouched = 123;
      EXPECT_FALSE(algorithm.cluster_once(pool, &untouched));
      EXPECT_EQ(read_error ? -73 : -75, algorithm.matrix_status());
      EXPECT_EQ(123, untouched);
      EXPECT_EQ(0, std::memcmp(before.data(), algorithm.centroids().data(),
                               before.bytes()));
      std::weak_ptr<TestMatrixStorage> lifetime = storage;
      storage.reset();
      algorithm.reset(1, 3);
      EXPECT_TRUE(lifetime.expired());
      EXPECT_EQ(0, algorithm.matrix_status());
      Populate(&algorithm);
      EXPECT_TRUE(algorithm.cluster_once(pool, &untouched));
    }
  }
}

TEST(KmeansMatrixStorage, AccumulatorErrorsPropagateWithoutPublishingCost) {
  for (int fault = 0; fault < 3; ++fault) {
    SCOPED_TRACE(fault);
    Algorithm algorithm(1, 3);
    auto storage = std::make_shared<TestMatrixStorage>();
    algorithm.set_accumulator_storage(storage);
    Populate(&algorithm);
    const float seed[]{1, 2, 3};
    algorithm.mutable_centroids()->append(seed, 3);
    if (fault == 0) storage->fail_write_at = 0;
    if (fault == 1) storage->fail_read_at = 0;
    // 127 worker reads, then one read when publishing the new centroid.
    if (fault == 2) storage->fail_read_at = 127;
    ThreadPool pool(1, false);
    double untouched = 123;
    EXPECT_FALSE(algorithm.cluster_once(pool, &untouched));
    EXPECT_EQ(fault == 0 ? -75 : -73, algorithm.matrix_status());
    EXPECT_EQ(123, untouched);
    EXPECT_EQ(0, std::memcmp(seed, algorithm.centroids().data(), sizeof(seed)));
    std::weak_ptr<TestMatrixStorage> lifetime = storage;
    storage.reset();
    algorithm.reset(1, 3);
    EXPECT_TRUE(lifetime.expired());
    Populate(&algorithm);
    EXPECT_TRUE(algorithm.cluster_once(pool, &untouched));
  }
}

TEST(KmeansMatrixStorage, AccumulatorsReuseHeapRowsAndResetEmptyState) {
  NumericalKmeansContext<float> context;
  const float row[]{1, 2, 3};
  context.reset(2, 3);
  const auto *first = &context[0];
  ASSERT_EQ(0, context[0].append(row, 3, 4));
  context.reset(2, 3);
  EXPECT_EQ(first, &context[0]);
  EXPECT_EQ(0u, context[0].count());
  float result[3]{};
  ASSERT_EQ(0, context[0].centroid(result, 3));
  EXPECT_TRUE(std::isnan(result[0]));
  ASSERT_EQ(0, context[0].append(row, 3, 4));
  ASSERT_EQ(0, context[0].centroid(result, 3));
  EXPECT_EQ(0, std::memcmp(row, result, sizeof(row)));
  EXPECT_EQ(4, context[0].cost());
}

TEST(KmeansMatrixStorage, ConcurrentAccumulatorUpdatesMatchSingleWorker) {
  Algorithm reference(1, 3), actual(1, 3);
  auto storage = std::make_shared<TestMatrixStorage>();
  actual.set_accumulator_storage(storage);
  Populate(&reference);
  Populate(&actual);
  const float seed[]{1, 2, 3};
  reference.mutable_centroids()->append(seed, 3);
  actual.mutable_centroids()->append(seed, 3);
  ThreadPool serial(1, false), parallel(4, false);
  for (int round = 0; round < 3; ++round) {
    double expected = 0, value = 0;
    ASSERT_TRUE(reference.cluster_once(serial, &expected));
    ASSERT_TRUE(actual.cluster_once(parallel, &value));
    EXPECT_NEAR(expected, value, 0.0001);
    EXPECT_EQ(128u, actual.context()[0].count());
    EXPECT_EQ(0, std::memcmp(reference.centroids().data(),
                             actual.centroids().data(), sizeof(seed)));
  }
}

TEST(KmeansMatrixStorage, WriteFailureIsStickyAndResetReleasesStorage) {
  Algorithm algorithm(1, 3);
  auto storage = std::make_shared<TestMatrixStorage>();
  storage->fail_write_at = 1;
  algorithm.set_feature_matrix_storage(storage);
  Populate(&algorithm);
  EXPECT_EQ(-75, algorithm.matrix_status());
  EXPECT_EQ(2u, storage->writes);
  EXPECT_EQ(Algorithm::BatchCount, algorithm.feature_matrix_count());
  std::weak_ptr<TestMatrixStorage> lifetime = storage;
  storage.reset();
  algorithm.reset(1, 3);
  EXPECT_TRUE(lifetime.expired());
  EXPECT_EQ(0, algorithm.matrix_status());
  Populate(&algorithm);
  EXPECT_EQ(128u, algorithm.feature_matrix_count());
}

}  // namespace
}  // namespace zvec::ailego
