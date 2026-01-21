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

TEST(DistanceMatrix, InnerProduct_General) {
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
        Distance::MinusInnerProduct(vec1.data(), query1.data(), dimension),
        Distance::MinusInnerProduct(vec2.data(), query2.data(), dimension));
    EXPECT_FLOAT_EQ(
        Distance::InnerProduct(vec1.data(), query1.data(), dimension),
        Distance::InnerProduct(vec2.data(), query2.data(), dimension));
    EXPECT_FLOAT_EQ(
        Distance::MinusInnerProduct(vec1.data(), query1.data(), dimension),
        -Distance::InnerProduct(vec2.data(), query2.data(), dimension));
  }
}

template <size_t M, size_t N>
void TestInnerProductMatrix(void) {
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
      InnerProductMatrix<uint8_t, 1, 1>::Compute(
          &matrix1[j * dimension / 2], cur_query, dimension, &query_result[j]);
    }
  }
  InnerProductMatrix<uint8_t, batch_size, query_size>::Compute(
      &matrix2[0], &query2[0], dimension, &result2[0]);

  for (size_t i = 0; i < batch_size * query_size; ++i) {
    EXPECT_FLOAT_EQ(result1[i], result2[i]);
  }
}

template <size_t M, size_t N>
void TestMinusInnerProductMatrix(void) {
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
      MinusInnerProductMatrix<uint8_t, 1, 1>::Compute(
          &matrix1[j * dimension / 2], cur_query, dimension, &query_result[j]);
    }
  }
  MinusInnerProductMatrix<uint8_t, batch_size, query_size>::Compute(
      &matrix2[0], &query2[0], dimension, &result2[0]);

  for (size_t i = 0; i < batch_size * query_size; ++i) {
    EXPECT_FLOAT_EQ(result1[i], result2[i]);
  }
}

TEST(DistanceMatrix, InnerProduct_1x1) {
  TestInnerProductMatrix<1, 1>();
}

TEST(DistanceMatrix, InnerProduct_2x1) {
  TestInnerProductMatrix<2, 1>();
}

TEST(DistanceMatrix, InnerProduct_2x2) {
  TestInnerProductMatrix<2, 2>();
}

TEST(DistanceMatrix, InnerProduct_3x3) {
  TestInnerProductMatrix<3, 3>();
}

TEST(DistanceMatrix, InnerProduct_4x1) {
  TestInnerProductMatrix<4, 1>();
}

TEST(DistanceMatrix, InnerProduct_4x2) {
  TestInnerProductMatrix<4, 2>();
}

TEST(DistanceMatrix, InnerProduct_4x4) {
  TestInnerProductMatrix<4, 4>();
}

TEST(DistanceMatrix, InnerProduct_8x1) {
  TestInnerProductMatrix<8, 1>();
}

TEST(DistanceMatrix, InnerProduct_8x2) {
  TestInnerProductMatrix<8, 2>();
}

TEST(DistanceMatrix, InnerProduct_8x4) {
  TestInnerProductMatrix<8, 4>();
}

TEST(DistanceMatrix, InnerProduct_8x8) {
  TestInnerProductMatrix<8, 8>();
}

TEST(DistanceMatrix, InnerProduct_16x1) {
  TestInnerProductMatrix<16, 1>();
}

TEST(DistanceMatrix, InnerProduct_16x2) {
  TestInnerProductMatrix<16, 2>();
}

TEST(DistanceMatrix, InnerProduct_16x4) {
  TestInnerProductMatrix<16, 4>();
}

TEST(DistanceMatrix, InnerProduct_16x8) {
  TestInnerProductMatrix<16, 8>();
}

TEST(DistanceMatrix, InnerProduct_16x16) {
  TestInnerProductMatrix<16, 16>();
}

TEST(DistanceMatrix, InnerProduct_32x1) {
  TestInnerProductMatrix<32, 1>();
}

TEST(DistanceMatrix, InnerProduct_32x2) {
  TestInnerProductMatrix<32, 2>();
}

TEST(DistanceMatrix, InnerProduct_32x4) {
  TestInnerProductMatrix<32, 4>();
}

TEST(DistanceMatrix, InnerProduct_32x8) {
  TestInnerProductMatrix<32, 8>();
}

TEST(DistanceMatrix, InnerProduct_32x16) {
  TestInnerProductMatrix<32, 16>();
}

TEST(DistanceMatrix, InnerProduct_32x32) {
  TestInnerProductMatrix<32, 32>();
}

TEST(DistanceMatrix, InnerProduct_64x1) {
  TestInnerProductMatrix<64, 1>();
}

TEST(DistanceMatrix, InnerProduct_64x2) {
  TestInnerProductMatrix<64, 2>();
}

TEST(DistanceMatrix, InnerProduct_64x4) {
  TestInnerProductMatrix<64, 4>();
}

TEST(DistanceMatrix, InnerProduct_64x8) {
  TestInnerProductMatrix<64, 8>();
}

TEST(DistanceMatrix, InnerProduct_64x16) {
  TestInnerProductMatrix<64, 16>();
}

TEST(DistanceMatrix, InnerProduct_64x32) {
  TestInnerProductMatrix<64, 32>();
}

TEST(DistanceMatrix, InnerProduct_64x64) {
  TestInnerProductMatrix<64, 128>();
}

TEST(DistanceMatrix, InnerProduct_128x1) {
  TestInnerProductMatrix<128, 1>();
}

TEST(DistanceMatrix, InnerProduct_128x2) {
  TestInnerProductMatrix<128, 2>();
}

TEST(DistanceMatrix, InnerProduct_128x4) {
  TestInnerProductMatrix<128, 4>();
}

TEST(DistanceMatrix, InnerProduct_128x8) {
  TestInnerProductMatrix<128, 8>();
}

TEST(DistanceMatrix, InnerProduct_128x16) {
  TestInnerProductMatrix<128, 16>();
}

TEST(DistanceMatrix, InnerProduct_128x32) {
  TestInnerProductMatrix<128, 32>();
}

TEST(DistanceMatrix, InnerProduct_128x64) {
  TestInnerProductMatrix<128, 128>();
}

TEST(DistanceMatrix, InnerProduct_128x128) {
  TestInnerProductMatrix<128, 128>();
}

TEST(DistanceMatrix, MinusInnerProduct_1x1) {
  TestMinusInnerProductMatrix<1, 1>();
}

TEST(DistanceMatrix, MinusInnerProduct_2x1) {
  TestMinusInnerProductMatrix<2, 1>();
}

TEST(DistanceMatrix, MinusInnerProduct_2x2) {
  TestMinusInnerProductMatrix<2, 2>();
}

TEST(DistanceMatrix, MinusInnerProduct_3x3) {
  TestMinusInnerProductMatrix<3, 3>();
}

TEST(DistanceMatrix, MinusInnerProduct_4x1) {
  TestMinusInnerProductMatrix<4, 1>();
}

TEST(DistanceMatrix, MinusInnerProduct_4x2) {
  TestMinusInnerProductMatrix<4, 2>();
}

TEST(DistanceMatrix, MinusInnerProduct_4x4) {
  TestMinusInnerProductMatrix<4, 4>();
}

TEST(DistanceMatrix, MinusInnerProduct_8x1) {
  TestMinusInnerProductMatrix<8, 1>();
}

TEST(DistanceMatrix, MinusInnerProduct_8x2) {
  TestMinusInnerProductMatrix<8, 2>();
}

TEST(DistanceMatrix, MinusInnerProduct_8x4) {
  TestMinusInnerProductMatrix<8, 4>();
}

TEST(DistanceMatrix, MinusInnerProduct_8x8) {
  TestMinusInnerProductMatrix<8, 8>();
}

TEST(DistanceMatrix, MinusInnerProduct_16x1) {
  TestMinusInnerProductMatrix<16, 1>();
}

TEST(DistanceMatrix, MinusInnerProduct_16x2) {
  TestMinusInnerProductMatrix<16, 2>();
}

TEST(DistanceMatrix, MinusInnerProduct_16x4) {
  TestMinusInnerProductMatrix<16, 4>();
}

TEST(DistanceMatrix, MinusInnerProduct_16x8) {
  TestMinusInnerProductMatrix<16, 8>();
}

TEST(DistanceMatrix, MinusInnerProduct_16x16) {
  TestMinusInnerProductMatrix<16, 16>();
}

TEST(DistanceMatrix, MinusInnerProduct_32x1) {
  TestMinusInnerProductMatrix<32, 1>();
}

TEST(DistanceMatrix, MinusInnerProduct_32x2) {
  TestMinusInnerProductMatrix<32, 2>();
}

TEST(DistanceMatrix, MinusInnerProduct_32x4) {
  TestMinusInnerProductMatrix<32, 4>();
}

TEST(DistanceMatrix, MinusInnerProduct_32x8) {
  TestMinusInnerProductMatrix<32, 8>();
}

TEST(DistanceMatrix, MinusInnerProduct_32x16) {
  TestMinusInnerProductMatrix<32, 16>();
}

TEST(DistanceMatrix, MinusInnerProduct_32x32) {
  TestMinusInnerProductMatrix<32, 32>();
}

TEST(DistanceMatrix, MinusInnerProduct_64x1) {
  TestMinusInnerProductMatrix<64, 1>();
}

TEST(DistanceMatrix, MinusInnerProduct_64x2) {
  TestMinusInnerProductMatrix<64, 2>();
}

TEST(DistanceMatrix, MinusInnerProduct_64x4) {
  TestMinusInnerProductMatrix<64, 4>();
}

TEST(DistanceMatrix, MinusInnerProduct_64x8) {
  TestMinusInnerProductMatrix<64, 8>();
}

TEST(DistanceMatrix, MinusInnerProduct_64x16) {
  TestMinusInnerProductMatrix<64, 16>();
}

TEST(DistanceMatrix, MinusInnerProduct_64x32) {
  TestMinusInnerProductMatrix<64, 32>();
}

TEST(DistanceMatrix, MinusInnerProduct_64x64) {
  TestMinusInnerProductMatrix<64, 128>();
}

TEST(DistanceMatrix, MinusInnerProduct_128x1) {
  TestMinusInnerProductMatrix<128, 1>();
}

TEST(DistanceMatrix, MinusInnerProduct_128x2) {
  TestMinusInnerProductMatrix<128, 2>();
}

TEST(DistanceMatrix, MinusInnerProduct_128x4) {
  TestMinusInnerProductMatrix<128, 4>();
}

TEST(DistanceMatrix, MinusInnerProduct_128x8) {
  TestMinusInnerProductMatrix<128, 8>();
}

TEST(DistanceMatrix, MinusInnerProduct_128x16) {
  TestMinusInnerProductMatrix<128, 16>();
}

TEST(DistanceMatrix, MinusInnerProduct_128x32) {
  TestMinusInnerProductMatrix<128, 32>();
}

TEST(DistanceMatrix, MinusInnerProduct_128x64) {
  TestMinusInnerProductMatrix<128, 128>();
}

TEST(DistanceMatrix, MinusInnerProduct_128x128) {
  TestMinusInnerProductMatrix<128, 128>();
}

template <size_t M, size_t N, size_t B, size_t D>
void InnerProductBenchmark(void) {
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

  // 1 Batched InnerProduct
  elapsed_time.reset();
  for (size_t i = 0; i < block_size; ++i) {
    const uint8_t *matrix_batch = &matrix2[i * batch_size * dimension / 2];

    for (size_t j = 0; j < query_size; ++j) {
      const uint8_t *current_query = &query1[j * dimension / 2];
      float *current_results = &results[j * batch_size];

      InnerProductMatrix<uint8_t, batch_size, 1>::Compute(
          matrix_batch, current_query, dimension, current_results);
    }
  }
  std::cout << "* 1 Batched InnerProduct (us) \t"
            << elapsed_time.micro_seconds() << std::endl;

  // N Batched InnerProduct
  elapsed_time.reset();
  for (size_t i = 0; i < block_size; ++i) {
    const uint8_t *matrix_batch = &matrix2[i * batch_size * dimension / 2];

    InnerProductMatrix<uint8_t, batch_size, query_size>::Compute(
        matrix_batch, &query2[0], dimension, results);
  }
  std::cout << "* N Batched InnerProduct (us) \t"
            << elapsed_time.micro_seconds() << std::endl;

  // Unbatched InnerProduct
  elapsed_time.reset();
  for (size_t i = 0; i < block_size; ++i) {
    const uint8_t *matrix_batch = &matrix1[i * batch_size * dimension / 2];

    for (size_t j = 0; j < query_size; ++j) {
      const uint8_t *current_query = &query1[j * dimension / 2];
      float *current_results = &results[j * batch_size];

      for (size_t k = 0; k < batch_size; ++k) {
        InnerProductMatrix<uint8_t, 1, 1>::Compute(
            &matrix_batch[k * dimension / 2], current_query, dimension,
            &current_results[k]);
      }
    }
  }
  std::cout << "* Unbatched InnerProduct (us) \t"
            << elapsed_time.micro_seconds() << std::endl;
}

TEST(DistanceMatrix, DISABLED_InnerProduct_Benchmark) {
  InnerProductBenchmark<2, 1, 512, 128>();
  InnerProductBenchmark<2, 2, 512, 128>();
  InnerProductBenchmark<4, 1, 512, 128>();
  InnerProductBenchmark<4, 2, 512, 128>();
  InnerProductBenchmark<4, 4, 512, 128>();
  InnerProductBenchmark<8, 1, 512, 128>();
  InnerProductBenchmark<8, 2, 512, 128>();
  InnerProductBenchmark<8, 4, 512, 128>();
  InnerProductBenchmark<8, 8, 512, 128>();
  InnerProductBenchmark<16, 1, 512, 128>();
  InnerProductBenchmark<16, 2, 512, 128>();
  InnerProductBenchmark<16, 4, 512, 128>();
  InnerProductBenchmark<16, 8, 512, 128>();
  InnerProductBenchmark<16, 16, 512, 128>();
  InnerProductBenchmark<32, 1, 512, 128>();
  InnerProductBenchmark<32, 2, 512, 128>();
  InnerProductBenchmark<32, 4, 512, 128>();
  InnerProductBenchmark<32, 8, 512, 128>();
  InnerProductBenchmark<32, 16, 512, 128>();
  InnerProductBenchmark<32, 32, 512, 128>();
  InnerProductBenchmark<64, 1, 512, 128>();
  InnerProductBenchmark<64, 2, 512, 128>();
  InnerProductBenchmark<64, 4, 512, 128>();
  InnerProductBenchmark<64, 8, 512, 128>();
  InnerProductBenchmark<128, 1, 512, 128>();
}
