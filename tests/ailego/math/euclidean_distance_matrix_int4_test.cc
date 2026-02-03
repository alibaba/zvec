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

#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <ailego/container/bitmap.h>
#include <ailego/internal/cpu_features.h>
#include <ailego/math/distance.h>
#include <ailego/utility/matrix_helper.h>
#include <gtest/gtest.h>
#include <zvec/ailego/container/vector.h>
#include <zvec/ailego/utility/time_helper.h>

using namespace zvec;
using namespace zvec::ailego;

static inline const char *IntelIntrinsics(void) {
  return internal::CpuFeatures::Intrinsics();
}

TEST(DistanceMatrix, Euclidean_General) {
  std::mt19937 gen((std::random_device())());
  size_t dimension = (std::uniform_int_distribution<size_t>(1, 64))(gen) << 1;

  std::vector<int8_t> vec1(dimension), query1(dimension);
  std::vector<uint8_t> vec2(dimension >> 1), query2(dimension >> 1);

  std::uniform_int_distribution<int> dist(-8, 7);

  for (size_t k = 0; k < 100; ++k) {
    for (size_t i = 0; i < dimension; i += 2) {
      vec1[i + 0] = (int8_t)dist(gen);
      vec1[i + 1] = (int8_t)dist(gen);
      vec2[i >> 1] =
          ((uint8_t)(vec1[i + 0]) << 4) | ((uint8_t)(vec1[i + 1]) & 0xf);
      EXPECT_EQ(vec1[i + 0] * vec1[i + 1], Int4MulTable[vec2[i >> 1]]);

      query1[i + 0] = (int8_t)dist(gen);
      query1[i + 1] = (int8_t)dist(gen);
      query2[i >> 1] =
          ((uint8_t)(query1[i + 0]) << 4) | ((uint8_t)(query1[i + 1]) & 0xf);
      EXPECT_EQ(query1[i + 0] * query1[i + 1], Int4MulTable[query2[i >> 1]]);
    }

    EXPECT_FLOAT_EQ(
        Distance::SquaredEuclidean(vec1.data(), query1.data(), dimension),
        Distance::SquaredEuclidean(vec2.data(), query2.data(), dimension));
    EXPECT_FLOAT_EQ(Distance::Euclidean(vec1.data(), query1.data(), dimension),
                    Distance::Euclidean(vec2.data(), query2.data(), dimension));
    EXPECT_FLOAT_EQ(std::sqrt(Distance::SquaredEuclidean(
                        vec1.data(), query1.data(), dimension)),
                    Distance::Euclidean(vec2.data(), query2.data(), dimension));
  }
}

template <size_t M, size_t N>
void TestEuclideanMatrix(void) {
  std::mt19937 gen((std::random_device())());

  const size_t batch_size = M;
  const size_t query_size = N;
  size_t dimension = (std::uniform_int_distribution<size_t>(1, 64))(gen) << 3;
  size_t matrix_size = batch_size * (dimension / 2);
  size_t query_matrix_size = query_size * (dimension / 2);

  std::vector<uint8_t> matrix1(matrix_size);
  std::vector<uint8_t> matrix2(matrix_size);
  std::vector<uint8_t> query1(query_matrix_size);
  std::vector<uint8_t> query2(query_matrix_size);
  std::vector<float> result1(batch_size * query_size);
  std::vector<float> result2(batch_size * query_size);

  std::uniform_int_distribution<int> dist(0, 0xff);
  for (size_t i = 0; i < matrix_size; ++i) {
    matrix1[i] = (uint8_t)dist(gen);
  }
  for (size_t i = 0; i < query_matrix_size; ++i) {
    query1[i] = (uint8_t)dist(gen);
  }
  ailego::MatrixHelper::Transpose<uint32_t, batch_size>(
      matrix1.data(), dimension / 8, &matrix2[0]);
  ailego::MatrixHelper::Transpose<uint32_t, query_size>(
      query1.data(), dimension / 8, &query2[0]);

  for (size_t i = 0; i < query_size; ++i) {
    const uint8_t *cur_query = &query1[i * dimension / 2];
    float *query_result = &result1[i * batch_size];

    for (size_t j = 0; j < batch_size; ++j) {
      EuclideanDistanceMatrix<uint8_t, 1, 1>::Compute(
          &matrix1[j * dimension / 2], cur_query, dimension, &query_result[j]);
    }
  }
  EuclideanDistanceMatrix<uint8_t, batch_size, query_size>::Compute(
      &matrix2[0], &query2[0], dimension, &result2[0]);

  for (size_t i = 0; i < batch_size * query_size; ++i) {
    EXPECT_FLOAT_EQ(result1[i], result2[i]);
  }
}

template <size_t M, size_t N>
void TestSquaredEuclideanMatrix(void) {
  std::mt19937 gen((std::random_device())());

  const size_t batch_size = M;
  const size_t query_size = N;
  size_t dimension = (std::uniform_int_distribution<size_t>(1, 64))(gen) << 3;
  size_t matrix_size = batch_size * (dimension / 2);
  size_t query_matrix_size = query_size * (dimension / 2);

  std::vector<uint8_t> matrix1(matrix_size);
  std::vector<uint8_t> matrix2(matrix_size);
  std::vector<uint8_t> query1(query_matrix_size);
  std::vector<uint8_t> query2(query_matrix_size);
  std::vector<float> result1(batch_size * query_size);
  std::vector<float> result2(batch_size * query_size);

  std::uniform_int_distribution<int> dist(0, 0xff);
  for (size_t i = 0; i < matrix_size; ++i) {
    matrix1[i] = (uint8_t)dist(gen);
  }
  for (size_t i = 0; i < query_matrix_size; ++i) {
    query1[i] = (uint8_t)dist(gen);
  }
  ailego::MatrixHelper::Transpose<uint32_t, batch_size>(
      matrix1.data(), dimension / 8, &matrix2[0]);
  ailego::MatrixHelper::Transpose<uint32_t, query_size>(
      query1.data(), dimension / 8, &query2[0]);

  for (size_t i = 0; i < query_size; ++i) {
    const uint8_t *cur_query = &query1[i * dimension / 2];
    float *query_result = &result1[i * batch_size];

    for (size_t j = 0; j < batch_size; ++j) {
      SquaredEuclideanDistanceMatrix<uint8_t, 1, 1>::Compute(
          &matrix1[j * dimension / 2], cur_query, dimension, &query_result[j]);
    }
  }
  SquaredEuclideanDistanceMatrix<uint8_t, batch_size, query_size>::Compute(
      &matrix2[0], &query2[0], dimension, &result2[0]);

  for (size_t i = 0; i < batch_size * query_size; ++i) {
    EXPECT_FLOAT_EQ(result1[i], result2[i]);
  }
}

TEST(DistanceMatrix, Euclidean_1x1) {
  TestEuclideanMatrix<1, 1>();
}

TEST(DistanceMatrix, Euclidean_2x1) {
  TestEuclideanMatrix<2, 1>();
}

TEST(DistanceMatrix, Euclidean_2x2) {
  TestEuclideanMatrix<2, 2>();
}

TEST(DistanceMatrix, Euclidean_3x3) {
  TestEuclideanMatrix<3, 3>();
}

TEST(DistanceMatrix, Euclidean_4x1) {
  TestEuclideanMatrix<4, 1>();
}

TEST(DistanceMatrix, Euclidean_4x2) {
  TestEuclideanMatrix<4, 2>();
}

TEST(DistanceMatrix, Euclidean_4x4) {
  TestEuclideanMatrix<4, 4>();
}

TEST(DistanceMatrix, Euclidean_8x1) {
  TestEuclideanMatrix<8, 1>();
}

TEST(DistanceMatrix, Euclidean_8x2) {
  TestEuclideanMatrix<8, 2>();
}

TEST(DistanceMatrix, Euclidean_8x4) {
  TestEuclideanMatrix<8, 4>();
}

TEST(DistanceMatrix, Euclidean_8x8) {
  TestEuclideanMatrix<8, 8>();
}

TEST(DistanceMatrix, Euclidean_16x1) {
  TestEuclideanMatrix<16, 1>();
}

TEST(DistanceMatrix, Euclidean_16x2) {
  TestEuclideanMatrix<16, 2>();
}

TEST(DistanceMatrix, Euclidean_16x4) {
  TestEuclideanMatrix<16, 4>();
}

TEST(DistanceMatrix, Euclidean_16x8) {
  TestEuclideanMatrix<16, 8>();
}

TEST(DistanceMatrix, Euclidean_16x16) {
  TestEuclideanMatrix<16, 16>();
}

TEST(DistanceMatrix, Euclidean_32x1) {
  TestEuclideanMatrix<32, 1>();
}

TEST(DistanceMatrix, Euclidean_32x2) {
  TestEuclideanMatrix<32, 2>();
}

TEST(DistanceMatrix, Euclidean_32x4) {
  TestEuclideanMatrix<32, 4>();
}

TEST(DistanceMatrix, Euclidean_32x8) {
  TestEuclideanMatrix<32, 8>();
}

TEST(DistanceMatrix, Euclidean_32x16) {
  TestEuclideanMatrix<32, 16>();
}

TEST(DistanceMatrix, Euclidean_32x32) {
  TestEuclideanMatrix<32, 32>();
}

TEST(DistanceMatrix, Euclidean_64x1) {
  TestEuclideanMatrix<64, 1>();
}

TEST(DistanceMatrix, Euclidean_64x2) {
  TestEuclideanMatrix<64, 2>();
}

TEST(DistanceMatrix, Euclidean_64x4) {
  TestEuclideanMatrix<64, 4>();
}

TEST(DistanceMatrix, Euclidean_64x8) {
  TestEuclideanMatrix<64, 8>();
}

TEST(DistanceMatrix, Euclidean_64x16) {
  TestEuclideanMatrix<64, 16>();
}

TEST(DistanceMatrix, Euclidean_64x32) {
  TestEuclideanMatrix<64, 32>();
}

TEST(DistanceMatrix, Euclidean_64x64) {
  TestEuclideanMatrix<64, 128>();
}

TEST(DistanceMatrix, Euclidean_128x1) {
  TestEuclideanMatrix<128, 1>();
}

TEST(DistanceMatrix, Euclidean_128x2) {
  TestEuclideanMatrix<128, 2>();
}

TEST(DistanceMatrix, Euclidean_128x4) {
  TestEuclideanMatrix<128, 4>();
}

TEST(DistanceMatrix, Euclidean_128x8) {
  TestEuclideanMatrix<128, 8>();
}

TEST(DistanceMatrix, Euclidean_128x16) {
  TestEuclideanMatrix<128, 16>();
}

TEST(DistanceMatrix, Euclidean_128x32) {
  TestEuclideanMatrix<128, 32>();
}

TEST(DistanceMatrix, Euclidean_128x64) {
  TestEuclideanMatrix<128, 128>();
}

TEST(DistanceMatrix, Euclidean_128x128) {
  TestEuclideanMatrix<128, 128>();
}

TEST(DistanceMatrix, SquaredEuclidean_1x1) {
  TestSquaredEuclideanMatrix<1, 1>();
}

TEST(DistanceMatrix, SquaredEuclidean_2x1) {
  TestSquaredEuclideanMatrix<2, 1>();
}

TEST(DistanceMatrix, SquaredEuclidean_2x2) {
  TestSquaredEuclideanMatrix<2, 2>();
}

TEST(DistanceMatrix, SquaredEuclidean_3x3) {
  TestSquaredEuclideanMatrix<3, 3>();
}

TEST(DistanceMatrix, SquaredEuclidean_4x1) {
  TestSquaredEuclideanMatrix<4, 1>();
}

TEST(DistanceMatrix, SquaredEuclidean_4x2) {
  TestSquaredEuclideanMatrix<4, 2>();
}

TEST(DistanceMatrix, SquaredEuclidean_4x4) {
  TestSquaredEuclideanMatrix<4, 4>();
}

TEST(DistanceMatrix, SquaredEuclidean_8x1) {
  TestSquaredEuclideanMatrix<8, 1>();
}

TEST(DistanceMatrix, SquaredEuclidean_8x2) {
  TestSquaredEuclideanMatrix<8, 2>();
}

TEST(DistanceMatrix, SquaredEuclidean_8x4) {
  TestSquaredEuclideanMatrix<8, 4>();
}

TEST(DistanceMatrix, SquaredEuclidean_8x8) {
  TestSquaredEuclideanMatrix<8, 8>();
}

TEST(DistanceMatrix, SquaredEuclidean_16x1) {
  TestSquaredEuclideanMatrix<16, 1>();
}

TEST(DistanceMatrix, SquaredEuclidean_16x2) {
  TestSquaredEuclideanMatrix<16, 2>();
}

TEST(DistanceMatrix, SquaredEuclidean_16x4) {
  TestSquaredEuclideanMatrix<16, 4>();
}

TEST(DistanceMatrix, SquaredEuclidean_16x8) {
  TestSquaredEuclideanMatrix<16, 8>();
}

TEST(DistanceMatrix, SquaredEuclidean_16x16) {
  TestSquaredEuclideanMatrix<16, 16>();
}

TEST(DistanceMatrix, SquaredEuclidean_32x1) {
  TestSquaredEuclideanMatrix<32, 1>();
}

TEST(DistanceMatrix, SquaredEuclidean_32x2) {
  TestSquaredEuclideanMatrix<32, 2>();
}

TEST(DistanceMatrix, SquaredEuclidean_32x4) {
  TestSquaredEuclideanMatrix<32, 4>();
}

TEST(DistanceMatrix, SquaredEuclidean_32x8) {
  TestSquaredEuclideanMatrix<32, 8>();
}

TEST(DistanceMatrix, SquaredEuclidean_32x16) {
  TestSquaredEuclideanMatrix<32, 16>();
}

TEST(DistanceMatrix, SquaredEuclidean_32x32) {
  TestSquaredEuclideanMatrix<32, 32>();
}

TEST(DistanceMatrix, SquaredEuclidean_64x1) {
  TestSquaredEuclideanMatrix<64, 1>();
}

TEST(DistanceMatrix, SquaredEuclidean_64x2) {
  TestSquaredEuclideanMatrix<64, 2>();
}

TEST(DistanceMatrix, SquaredEuclidean_64x4) {
  TestSquaredEuclideanMatrix<64, 4>();
}

TEST(DistanceMatrix, SquaredEuclidean_64x8) {
  TestSquaredEuclideanMatrix<64, 8>();
}

TEST(DistanceMatrix, SquaredEuclidean_64x16) {
  TestSquaredEuclideanMatrix<64, 16>();
}

TEST(DistanceMatrix, SquaredEuclidean_64x32) {
  TestSquaredEuclideanMatrix<64, 32>();
}

TEST(DistanceMatrix, SquaredEuclidean_64x64) {
  TestSquaredEuclideanMatrix<64, 128>();
}

TEST(DistanceMatrix, SquaredEuclidean_128x1) {
  TestSquaredEuclideanMatrix<128, 1>();
}

TEST(DistanceMatrix, SquaredEuclidean_128x2) {
  TestSquaredEuclideanMatrix<128, 2>();
}

TEST(DistanceMatrix, SquaredEuclidean_128x4) {
  TestSquaredEuclideanMatrix<128, 4>();
}

TEST(DistanceMatrix, SquaredEuclidean_128x8) {
  TestSquaredEuclideanMatrix<128, 8>();
}

TEST(DistanceMatrix, SquaredEuclidean_128x16) {
  TestSquaredEuclideanMatrix<128, 16>();
}

TEST(DistanceMatrix, SquaredEuclidean_128x32) {
  TestSquaredEuclideanMatrix<128, 32>();
}

TEST(DistanceMatrix, SquaredEuclidean_128x64) {
  TestSquaredEuclideanMatrix<128, 128>();
}

TEST(DistanceMatrix, SquaredEuclidean_128x128) {
  TestSquaredEuclideanMatrix<128, 128>();
}

template <size_t M, size_t N, size_t B, size_t D>
void EuclideanBenchmark(void) {
  const size_t dimension = D;
  const size_t batch_size = M;
  const size_t block_size = B;
  const size_t query_size = N;
  const size_t matrix_size = block_size * batch_size * dimension / 2;
  const size_t query_matrix_size = query_size * dimension / 2;

  std::vector<uint8_t> matrix1(matrix_size);
  std::vector<uint8_t> matrix2(matrix_size);
  std::vector<uint8_t> query1(query_matrix_size);
  std::vector<uint8_t> query2(query_matrix_size);

  std::mt19937 gen((std::random_device())());
  std::uniform_int_distribution<int> dist(0, 0xff);
  for (size_t i = 0; i < matrix_size; ++i) {
    matrix1[i] = (uint8_t)dist(gen);
  }
  for (size_t i = 0; i < query_matrix_size; ++i) {
    query1[i] = (uint8_t)dist(gen);
  }

  for (size_t i = 0; i < block_size; ++i) {
    size_t start_pos = i * batch_size * dimension / 2;
    ailego::MatrixHelper::Transpose<uint32_t, batch_size>(
        &matrix1[start_pos], dimension / 8, &matrix2[start_pos]);
  }
  ailego::MatrixHelper::Transpose<uint32_t, query_size>(
      query1.data(), dimension / 8, &query2[0]);

  ElapsedTime elapsed_time;
  float results[batch_size * query_size];

  std::cout << "# (" << IntelIntrinsics() << ") INT4 " << dimension << "d, "
            << batch_size << " * " << query_size << " * " << block_size
            << std::endl;

  // 1 Batched Euclidean
  elapsed_time.reset();
  for (size_t i = 0; i < block_size; ++i) {
    const uint8_t *matrix_batch = &matrix2[i * batch_size * dimension / 2];

    for (size_t j = 0; j < query_size; ++j) {
      const uint8_t *current_query = &query1[j * dimension / 2];
      float *current_results = &results[j * batch_size];

      EuclideanDistanceMatrix<uint8_t, batch_size, 1>::Compute(
          matrix_batch, current_query, dimension, current_results);
    }
  }
  std::cout << "* 1 Batched Euclidean (us) \t" << elapsed_time.micro_seconds()
            << std::endl;

  // N Batched Euclidean
  elapsed_time.reset();
  for (size_t i = 0; i < block_size; ++i) {
    const uint8_t *matrix_batch = &matrix2[i * batch_size * dimension / 2];

    EuclideanDistanceMatrix<uint8_t, batch_size, query_size>::Compute(
        matrix_batch, &query2[0], dimension, results);
  }
  std::cout << "* N Batched Euclidean (us) \t" << elapsed_time.micro_seconds()
            << std::endl;

  // Unbatched Euclidean
  elapsed_time.reset();
  for (size_t i = 0; i < block_size; ++i) {
    const uint8_t *matrix_batch = &matrix1[i * batch_size * dimension / 2];

    for (size_t j = 0; j < query_size; ++j) {
      const uint8_t *current_query = &query1[j * dimension / 2];
      float *current_results = &results[j * batch_size];

      for (size_t k = 0; k < batch_size; ++k) {
        EuclideanDistanceMatrix<uint8_t, 1, 1>::Compute(
            &matrix_batch[k * dimension / 2], current_query, dimension,
            &current_results[k]);
      }
    }
  }
  std::cout << "* Unbatched Euclidean (us) \t" << elapsed_time.micro_seconds()
            << std::endl;
}

template <size_t M, size_t N, size_t B, size_t D>
void SquaredEuclideanBenchmark(void) {
  const size_t dimension = D;
  const size_t batch_size = M;
  const size_t block_size = B;
  const size_t query_size = N;
  const size_t matrix_size = block_size * batch_size * dimension / 2;
  const size_t query_matrix_size = query_size * dimension / 2;

  std::vector<uint8_t> matrix1(matrix_size);
  std::vector<uint8_t> matrix2(matrix_size);
  std::vector<uint8_t> query1(query_matrix_size);
  std::vector<uint8_t> query2(query_matrix_size);

  std::mt19937 gen((std::random_device())());
  std::uniform_int_distribution<int> dist(0, 0xff);
  for (size_t i = 0; i < matrix_size; ++i) {
    matrix1[i] = (uint8_t)dist(gen);
  }
  for (size_t i = 0; i < query_matrix_size; ++i) {
    query1[i] = (uint8_t)dist(gen);
  }

  for (size_t i = 0; i < block_size; ++i) {
    size_t start_pos = i * batch_size * dimension / 2;
    ailego::MatrixHelper::Transpose<uint32_t, batch_size>(
        &matrix1[start_pos], dimension / 8, &matrix2[start_pos]);
  }
  ailego::MatrixHelper::Transpose<uint32_t, query_size>(
      query1.data(), dimension / 8, &query2[0]);

  ElapsedTime elapsed_time;
  float results[batch_size * query_size];

  std::cout << "# (" << IntelIntrinsics() << ") INT4 " << dimension << "d, "
            << batch_size << " * " << query_size << " * " << block_size
            << std::endl;

  // 1 Batched SquaredEuclidean
  elapsed_time.reset();
  for (size_t i = 0; i < block_size; ++i) {
    const uint8_t *matrix_batch = &matrix2[i * batch_size * dimension / 2];

    for (size_t j = 0; j < query_size; ++j) {
      const uint8_t *current_query = &query1[j * dimension / 2];
      float *current_results = &results[j * batch_size];

      SquaredEuclideanDistanceMatrix<uint8_t, batch_size, 1>::Compute(
          matrix_batch, current_query, dimension, current_results);
    }
  }
  std::cout << "* 1 Batched SquaredEuclidean (us) \t"
            << elapsed_time.micro_seconds() << std::endl;

  // N Batched SquaredEuclidean
  elapsed_time.reset();
  for (size_t i = 0; i < block_size; ++i) {
    const uint8_t *matrix_batch = &matrix2[i * batch_size * dimension / 2];

    SquaredEuclideanDistanceMatrix<uint8_t, batch_size, query_size>::Compute(
        matrix_batch, &query2[0], dimension, results);
  }
  std::cout << "* N Batched SquaredEuclidean (us) \t"
            << elapsed_time.micro_seconds() << std::endl;

  // Unbatched SquaredEuclidean
  elapsed_time.reset();
  for (size_t i = 0; i < block_size; ++i) {
    const uint8_t *matrix_batch = &matrix1[i * batch_size * dimension / 2];

    for (size_t j = 0; j < query_size; ++j) {
      const uint8_t *current_query = &query1[j * dimension / 2];
      float *current_results = &results[j * batch_size];

      for (size_t k = 0; k < batch_size; ++k) {
        SquaredEuclideanDistanceMatrix<uint8_t, 1, 1>::Compute(
            &matrix_batch[k * dimension / 2], current_query, dimension,
            &current_results[k]);
      }
    }
  }
  std::cout << "* Unbatched SquaredEuclidean (us) \t"
            << elapsed_time.micro_seconds() << std::endl;
}

TEST(DistanceMatrix, DISABLED_Euclidean_Benchmark) {
  EuclideanBenchmark<2, 1, 512, 128>();
  EuclideanBenchmark<2, 2, 512, 128>();
  EuclideanBenchmark<4, 1, 512, 128>();
  EuclideanBenchmark<4, 2, 512, 128>();
  EuclideanBenchmark<4, 4, 512, 128>();
  EuclideanBenchmark<8, 1, 512, 128>();
  EuclideanBenchmark<8, 2, 512, 128>();
  EuclideanBenchmark<8, 4, 512, 128>();
  EuclideanBenchmark<8, 8, 512, 128>();
  EuclideanBenchmark<16, 1, 512, 128>();
  EuclideanBenchmark<16, 2, 512, 128>();
  EuclideanBenchmark<16, 4, 512, 128>();
  EuclideanBenchmark<16, 8, 512, 128>();
  EuclideanBenchmark<16, 16, 512, 128>();
  EuclideanBenchmark<32, 1, 512, 128>();
  EuclideanBenchmark<32, 2, 512, 128>();
  EuclideanBenchmark<32, 4, 512, 128>();
  EuclideanBenchmark<32, 8, 512, 128>();
  EuclideanBenchmark<32, 16, 512, 128>();
  EuclideanBenchmark<32, 32, 512, 128>();
  EuclideanBenchmark<64, 1, 512, 128>();
  EuclideanBenchmark<64, 2, 512, 128>();
  EuclideanBenchmark<64, 4, 512, 128>();
  EuclideanBenchmark<64, 8, 512, 128>();
  EuclideanBenchmark<128, 1, 512, 128>();
}

TEST(DistanceMatrix, DISABLED_SquaredEuclidean_Benchmark) {
  SquaredEuclideanBenchmark<2, 1, 512, 128>();
  SquaredEuclideanBenchmark<2, 2, 512, 128>();
  SquaredEuclideanBenchmark<4, 1, 512, 128>();
  SquaredEuclideanBenchmark<4, 2, 512, 128>();
  SquaredEuclideanBenchmark<4, 4, 512, 128>();
  SquaredEuclideanBenchmark<8, 1, 512, 128>();
  SquaredEuclideanBenchmark<8, 2, 512, 128>();
  SquaredEuclideanBenchmark<8, 4, 512, 128>();
  SquaredEuclideanBenchmark<8, 8, 512, 128>();
  SquaredEuclideanBenchmark<16, 1, 512, 128>();
  SquaredEuclideanBenchmark<16, 2, 512, 128>();
  SquaredEuclideanBenchmark<16, 4, 512, 128>();
  SquaredEuclideanBenchmark<16, 8, 512, 128>();
  SquaredEuclideanBenchmark<16, 16, 512, 128>();
  SquaredEuclideanBenchmark<32, 1, 512, 128>();
  SquaredEuclideanBenchmark<32, 2, 512, 128>();
  SquaredEuclideanBenchmark<32, 4, 512, 128>();
  SquaredEuclideanBenchmark<32, 8, 512, 128>();
  SquaredEuclideanBenchmark<32, 16, 512, 128>();
  SquaredEuclideanBenchmark<32, 32, 512, 128>();
  SquaredEuclideanBenchmark<64, 1, 512, 128>();
  SquaredEuclideanBenchmark<64, 2, 512, 128>();
  SquaredEuclideanBenchmark<64, 4, 512, 128>();
  SquaredEuclideanBenchmark<64, 8, 512, 128>();
  SquaredEuclideanBenchmark<128, 1, 512, 128>();
}
