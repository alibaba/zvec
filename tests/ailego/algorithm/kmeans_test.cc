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

#include <random>
#include <gtest/gtest.h>
#include <zvec/ailego/container/vector.h>
#include <zvec/ailego/parallel/thread_pool.h>

#define protected public
#define private public
#include <ailego/algorithm/kmeans.h>

using namespace zvec;

TEST(NumericalKmeans, FP32_General) {
  const size_t DIMENSION = 20;
  const size_t K_VALUE = 20;
  const size_t COUNT = 20000u;

  ailego::NumericalKmeans<float, ailego::ThreadPool> kmeans;
  kmeans.reset(K_VALUE, DIMENSION);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(0.0, 1.0);

  for (size_t i = 0; i < COUNT; ++i) {
    ailego::FixedVector<float, DIMENSION> vec;
    for (size_t j = 0; j < DIMENSION; ++j) {
      vec[j] = dist(gen);
    }
    kmeans.append(vec.data(), vec.size());
  }

  ailego::ThreadPool pool;
  double prev_sse = 0.0;
  for (size_t i = 0; i < 20; ++i) {
    double sse = 0.0;
    EXPECT_TRUE(kmeans.cluster_once(pool, &sse));
    printf("(%zu) SSE: %f -> %f = %f\n", i, prev_sse, sse, sse - prev_sse);
    prev_sse = sse;
  }

  for (auto &it : kmeans.context().clusters()) {
    printf("%f: %zu\n", it.cost(), it.count());
  }
}

TEST(NumericalKmeans, FP16_General) {
  const size_t DIMENSION = 20;
  const size_t K_VALUE = 20;
  const size_t COUNT = 20000u;

  ailego::NumericalKmeans<ailego::Float16, ailego::ThreadPool> kmeans;
  kmeans.reset(K_VALUE, DIMENSION);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(0.0, 1.0);

  for (size_t i = 0; i < COUNT; ++i) {
    ailego::FixedVector<ailego::Float16, DIMENSION> vec;
    for (size_t j = 0; j < DIMENSION; ++j) {
      vec[j] = dist(gen);
    }
    kmeans.append(vec.data(), vec.size());
  }

  ailego::ThreadPool pool;
  double prev_sse = 0.0;
  for (size_t i = 0; i < 20; ++i) {
    double sse = 0.0;
    EXPECT_TRUE(kmeans.cluster_once(pool, &sse));
    printf("(%zu) SSE: %f -> %f = %f\n", i, prev_sse, sse, sse - prev_sse);
    prev_sse = sse;
  }

  for (auto &it : kmeans.context().clusters()) {
    printf("%f: %zu\n", it.cost(), it.count());
  }
}

TEST(NumericalKmeans, INT8_General) {
  const size_t DIMENSION = 20 * 4;
  const size_t K_VALUE = 20;
  const size_t COUNT = 20000u;

  ailego::NumericalKmeans<int8_t, ailego::ThreadPool> kmeans;
  kmeans.reset(K_VALUE, DIMENSION);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(-127, 127);

  for (size_t i = 0; i < COUNT; ++i) {
    ailego::FixedVector<int8_t, DIMENSION> vec;
    for (size_t j = 0; j < DIMENSION; ++j) {
      vec[j] = (int8_t)dist(gen);
    }
    kmeans.append(vec.data(), vec.size());
  }

  ailego::ThreadPool pool;
  double prev_sse = 0.0;
  for (size_t i = 0; i < 20; ++i) {
    double sse = 0.0;
    EXPECT_TRUE(kmeans.cluster_once(pool, &sse));
    printf("(%zu) SSE: %f -> %f = %f\n", i, prev_sse, sse, sse - prev_sse);
    prev_sse = sse;
  }

  for (auto &it : kmeans.context().clusters()) {
    printf("%f: %zu\n", it.cost(), it.count());
  }
}

TEST(NumericalKmeans, BINARY32_General) {
  const size_t DIMENSION = 16 * 32;
  const size_t K_VALUE = 20;
  const size_t COUNT = 20000u;

  ailego::BinaryKmeans<uint32_t, ailego::ThreadPool> kmeans;
  kmeans.reset(K_VALUE, DIMENSION);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(0.0, 1.0);

  for (size_t i = 0; i < COUNT; ++i) {
    ailego::BinaryVector<uint32_t> vec(DIMENSION);
    for (size_t j = 0; j < DIMENSION; ++j) {
      if (dist(gen) >= 0.5) {
        vec.set(j);
      }
    }
    kmeans.append(vec.data(), vec.dimension());
  }

  ailego::ThreadPool pool;
  double prev_sse = 0.0;
  for (size_t i = 0; i < 20; ++i) {
    double sse = 0.0;
    EXPECT_TRUE(kmeans.cluster_once(pool, &sse));
    printf("(%zu) SSE: %f -> %f = %f\n", i, prev_sse, sse, sse - prev_sse);
    prev_sse = sse;
  }

  for (auto &it : kmeans.context().clusters()) {
    printf("%f: %zu\n", it.cost(), it.count());
  }
}

#if defined(AILEGO_M64)
TEST(NumericalKmeans, BINARY64_General) {
  const size_t DIMENSION = 8 * 64;
  const size_t K_VALUE = 20;
  const size_t COUNT = 20000u;

  ailego::BinaryKmeans<uint64_t, ailego::ThreadPool> kmeans;
  kmeans.reset(K_VALUE, DIMENSION);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(0.0, 1.0);

  for (size_t i = 0; i < COUNT; ++i) {
    ailego::BinaryVector<uint64_t> vec(DIMENSION);
    for (size_t j = 0; j < DIMENSION; ++j) {
      if (dist(gen) >= 0.5) {
        vec.set(j);
      }
    }
    kmeans.append(vec.data(), vec.dimension());
  }

  ailego::ThreadPool pool;
  double prev_sse = 0.0;
  for (size_t i = 0; i < 20; ++i) {
    double sse = 0.0;
    EXPECT_TRUE(kmeans.cluster_once(pool, &sse));
    printf("(%zu) SSE: %f -> %f = %f\n", i, prev_sse, sse, sse - prev_sse);
    prev_sse = sse;
  }

  for (auto &it : kmeans.context().clusters()) {
    printf("%f: %zu\n", it.cost(), it.count());
  }
}
#endif  // AILEGO_M64

TEST(NibbleKmeans, INT4_General) {
  const size_t DIMENSION = 32;
  const size_t K_VALUE = 63;
  const size_t COUNT = 40000u;

  ailego::NumericalKmeans<int8_t, ailego::ThreadPool> kmeans1;
  ailego::NibbleKmeans<int32_t, ailego::ThreadPool> kmeans2;
  kmeans1.reset(K_VALUE, DIMENSION);
  kmeans2.reset(K_VALUE, DIMENSION);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(-8, 7);

  for (size_t i = 0; i < COUNT; ++i) {
    ailego::NumericalVector<int8_t> vec1(DIMENSION);
    ailego::NibbleVector<int32_t> vec2(DIMENSION);

    for (size_t j = 0; j < DIMENSION; ++j) {
      int8_t val = (int8_t)dist(gen);
      vec1[j] = val;
      vec2.set(j, val);
    }
    kmeans1.append(vec1.data(), vec1.size());
    kmeans2.append(vec2.data(), vec2.size());
  }

  ailego::ThreadPool pool;
  {
    const ailego::NumericalKmeans<int8_t, ailego::ThreadPool> &kmeans1_ref =
        kmeans1;
    ailego::Kmc2CentroidsGenerator<decltype(kmeans1_ref), ailego::ThreadPool> g;

    kmeans1.init_centroids(pool);

    g.set_chain_length(20);
    kmeans1.init_centroids(pool, g);

    g.set_assumption_free(true);
    kmeans1.init_centroids(pool, g);

    // Shared centroids
    auto centroids = kmeans1.centroids();
    for (size_t i = 0; i < centroids.count(); ++i) {
      ailego::NibbleVector<int8_t> nvec;
      nvec.assign(centroids[i], centroids.dimension());
      kmeans2.mutable_centroids()->append(
          reinterpret_cast<const uint32_t *>(nvec.data()), nvec.dimension());
    }
  }

  double prev_sse1 = 0.0;
  double prev_sse2 = 0.0;
  for (size_t i = 0; i < 18; ++i) {
    double sse1 = 0.0;
    double sse2 = 0.0;
    EXPECT_TRUE(kmeans1.cluster_once(pool, &sse1));
    EXPECT_TRUE(kmeans2.cluster_once(pool, &sse2));
    printf("1: (%zu) SSE: %f -> %f = %f\n", i, prev_sse1, sse1,
           sse1 - prev_sse1);
    printf("2: (%zu) SSE: %f -> %f = %f\n", i, prev_sse2, sse2,
           sse2 - prev_sse2);
    prev_sse1 = sse1;
    prev_sse2 = sse2;
  }

  auto &cluster1 = kmeans1.context().clusters();
  auto &cluster2 = kmeans2.context().clusters();
  for (size_t i = 0; i < cluster1.size(); ++i) {
    // printf("(%zu) INT8 %f: %zu\n", i, cluster1[i].cost(),
    //        cluster1[i].count());
    // printf("(%zu) INT4 %f: %zu\n", i, cluster2[i].cost(),
    //        cluster2[i].count());

    for (size_t j = 0; j < cluster1[i].accum_.size(); ++j) {
      EXPECT_DOUBLE_EQ(cluster1[i].accum_[j], cluster2[i].accum_[j]);
    }
  }
}

TEST(NumericalKmeans, FP32_General_InnerProduct) {
  const size_t DIMENSION = 20;
  const size_t K_VALUE = 20;
  const size_t COUNT = 20000u;

  ailego::NumericalInnerProductKmeans<float, ailego::ThreadPool> kmeans;
  kmeans.reset(K_VALUE, DIMENSION);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-1.0, 1.0);

  for (size_t i = 0; i < COUNT; ++i) {
    ailego::FixedVector<float, DIMENSION> vec;
    for (size_t j = 0; j < DIMENSION; ++j) {
      vec[j] = dist(gen);
    }
    kmeans.append(vec.data(), vec.size());
  }

  ailego::ThreadPool pool;
  double prev_sse = 0.0;
  for (size_t i = 0; i < 20; ++i) {
    double sse = 0.0;
    EXPECT_TRUE(kmeans.cluster_once(pool, &sse));
    printf("(%zu) SSE: %f -> %f = %f\n", i, prev_sse, sse, sse - prev_sse);
    prev_sse = sse;
  }

  for (auto &it : kmeans.context().clusters()) {
    printf("%f: %zu\n", it.cost(), it.count());
  }
}

TEST(NumericalKmeans, FP16_General_InnerProduct) {
  const size_t DIMENSION = 20;
  const size_t K_VALUE = 20;
  const size_t COUNT = 20000u;

  ailego::NumericalInnerProductKmeans<ailego::Float16, ailego::ThreadPool>
      kmeans;
  kmeans.reset(K_VALUE, DIMENSION);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-1.0, 1.0);

  for (size_t i = 0; i < COUNT; ++i) {
    ailego::FixedVector<ailego::Float16, DIMENSION> vec;
    for (size_t j = 0; j < DIMENSION; ++j) {
      vec[j] = dist(gen);
    }
    kmeans.append(vec.data(), vec.size());
  }

  ailego::ThreadPool pool;
  double prev_sse = 0.0;
  for (size_t i = 0; i < 20; ++i) {
    double sse = 0.0;
    EXPECT_TRUE(kmeans.cluster_once(pool, &sse));
    printf("(%zu) SSE: %f -> %f = %f\n", i, prev_sse, sse, sse - prev_sse);
    prev_sse = sse;
  }

  for (auto &it : kmeans.context().clusters()) {
    printf("%f: %zu\n", it.cost(), it.count());
  }
}

TEST(NumericalKmeans, INT8_General_InnerProduct) {
  const size_t DIMENSION = 20 * 4;
  const size_t K_VALUE = 20;
  const size_t COUNT = 20000u;

  ailego::NumericalInnerProductKmeans<int8_t, ailego::ThreadPool> kmeans;
  kmeans.reset(K_VALUE, DIMENSION);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(-127, 127);

  for (size_t i = 0; i < COUNT; ++i) {
    ailego::FixedVector<int8_t, DIMENSION> vec;
    for (size_t j = 0; j < DIMENSION; ++j) {
      vec[j] = (int8_t)dist(gen);
    }
    kmeans.append(vec.data(), vec.size());
  }

  ailego::ThreadPool pool;
  double prev_sse = 0.0;
  for (size_t i = 0; i < 20; ++i) {
    double sse = 0.0;
    EXPECT_TRUE(kmeans.cluster_once(pool, &sse));
    printf("(%zu) SSE: %f -> %f = %f\n", i, prev_sse, sse, sse - prev_sse);
    prev_sse = sse;
  }

  for (auto &it : kmeans.context().clusters()) {
    printf("%f: %zu\n", it.cost(), it.count());
  }
}

TEST(NumericalKmeans, FP32_General_InnerProduct_Spherical) {
  const size_t DIMENSION = 20;
  const size_t K_VALUE = 20;
  const size_t COUNT = 20000u;

  ailego::NumericalInnerProductKmeans<float, ailego::ThreadPool> kmeans;
  kmeans.reset(K_VALUE, DIMENSION, true);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-1.0, 1.0);

  for (size_t i = 0; i < COUNT; ++i) {
    ailego::FixedVector<float, DIMENSION> vec;
    for (size_t j = 0; j < DIMENSION; ++j) {
      vec[j] = dist(gen);
    }
    kmeans.append(vec.data(), vec.size());
  }

  ailego::ThreadPool pool;
  double prev_sse = 0.0;
  for (size_t i = 0; i < 20; ++i) {
    double sse = 0.0;
    EXPECT_TRUE(kmeans.cluster_once(pool, &sse));
    printf("(%zu) SSE: %f -> %f = %f\n", i, prev_sse, sse, sse - prev_sse);
    prev_sse = sse;
  }

  for (auto &it : kmeans.context().clusters()) {
    printf("%f: %zu\n", it.cost(), it.count());
  }
}
