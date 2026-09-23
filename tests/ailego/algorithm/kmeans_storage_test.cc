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
    const size_t call = reads.fetch_add(1);
    if (call >= fail_read_at) return -73;
    if (offset > data.size() || bytes > data.size() - offset) return -74;
    std::memcpy(out, data.data() + offset, bytes);
    return 0;
  }
  int write(size_t offset, const void *input, size_t bytes) override {
    if (writes++ >= fail_write_at) return -75;
    largest_write = std::max(largest_write, bytes);
    data.resize(offset + bytes);
    std::memcpy(data.data() + offset, input, bytes);
    return 0;
  }

  std::string data;
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
    Algorithm memory(1, dim);
    Algorithm external(1, dim);
    auto storage = std::make_shared<TestMatrixStorage>();
    external.set_feature_matrix_storage(storage);
    memory.feature_matrix_reserve(count);
    external.feature_matrix_reserve(count);
    Container row(dim);
    row.resize(1);
    for (size_t i = 0; i < count; ++i) {
      fill(i, row.data());
      memory.append(row.data(), dim);
      external.append(row.data(), dim);
    }
    EXPECT_EQ(0u, external.feature_matrix().bytes());
    EXPECT_EQ(memory.feature_matrix_count(), external.feature_matrix_count());
    EXPECT_EQ(memory.feature_cache().bytes(), external.feature_cache().bytes());
    EXPECT_EQ(memory.feature_matrix().bytes(), storage->data.size());
    EXPECT_LE(storage->largest_write, Algorithm::BatchCount * row.bytes());
    fill(0, row.data());
    memory.mutable_centroids()->append(row.data(), dim);
    external.mutable_centroids()->append(row.data(), dim);
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
      EXPECT_EQ(count, external.context().clusters()[0].count());
    }
  }
}

template <typename T>
void CheckNumeric() {
  auto fill = [](size_t i, T *out) {
    for (size_t d = 0; d < 8; ++d) out[d] = T(int((i + d) % 5) - 2);
  };
  CheckMatrixEquivalence<NumericalKmeans<T, ThreadPool>>(fill);
  CheckMatrixEquivalence<NumericalInnerProductKmeans<T, ThreadPool>>(fill);
}

TEST(KmeansMatrixStorage, NumericFullAndPartialBlocksMatchMemory) {
  CheckNumeric<float>();
  CheckNumeric<Float16>();
  CheckNumeric<int8_t>();
}

TEST(KmeansMatrixStorage, PackedInt4FullAndPartialBlocksMatchMemory) {
  auto fill = [](size_t i, uint32_t *out) {
    NibbleVector<int32_t> row(8);
    for (size_t d = 0; d < 8; ++d) row.set(d, int((i + d) % 5) - 2);
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
