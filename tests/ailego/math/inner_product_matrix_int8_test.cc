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
#include <gtest/gtest.h>
#include <zvec/ailego/container/vector.h>
#include <zvec/ailego/utility/time_helper.h>

using namespace zvec::ailego;

static inline const char *IntelIntrinsics(void) {
  return internal::CpuFeatures::Intrinsics();
}

static inline void MatrixTranspose(uint32_t *dst, const uint32_t *src, size_t M,
                                   size_t N) {
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < M; ++j) {
      dst[j * N + i] = src[i * M + j];
    }
  }
}

template <size_t N>
static float InnerProductDistance(const FixedVector<int8_t, N> &lhs,
                                  const FixedVector<int8_t, N> &rhs) {
  return Distance::InnerProduct(lhs.data(), rhs.data(), lhs.size());
}

template <size_t N>
static float MinusInnerProductDistance(const FixedVector<int8_t, N> &lhs,
                                       const FixedVector<int8_t, N> &rhs) {
  return Distance::MinusInnerProduct(lhs.data(), rhs.data(), lhs.size());
}

TEST(DistanceMatrix, InnerProduct_General) {
  int8_t a1[] = {0};
  int8_t b1[] = {0};

  int8_t a17[] = {127, -1,  -1,  127, 127, 127, 127, -1, 127,
                  127, 127, 127, 127, 127, -1,  -1,  127};
  int8_t b17[] = {127, -1,  -1,  127, 127, 127, -1, 127, 127,
                  127, 127, 127, 127, 127, -1,  -1, 127};

  int8_t a47[] = {127, 2, 0,    0,    -127, -127, 0,    0,    0,    0,
                  0,   0, -127, -127, 127,  127,  0,    0,    -127, -127,
                  0,   0, 127,  5,    127,  127,  0,    0,    -127, -127,
                  0,   0, -127, 126,  -127, -127, -127, -127, 127,  127,
                  1,   2, 3,    4,    127,  127,  111};
  int8_t b47[] = {-127, 1, 0,    0,   127,  127,  0,   0,    0,   0,
                  0,    0, 127,  127, -127, -127, 0,   0,    127, 127,
                  0,    0, -127, 3,   -127, -127, 0,   0,    127, 127,
                  0,    0, 127,  127, 100,  122,  123, -127, 1,   2,
                  3,    4, -127, 122, -127, -127, -127};

  EXPECT_FLOAT_EQ(0.0f,
                  InnerProductDistance(*FixedVector<int8_t, 1>::Cast(a1),
                                       *FixedVector<int8_t, 1>::Cast(b1)));
  EXPECT_FLOAT_EQ(177169.0f,
                  InnerProductDistance(*FixedVector<int8_t, 17>::Cast(a17),
                                       *FixedVector<int8_t, 17>::Cast(b17)));
  EXPECT_FLOAT_EQ(-299458.0f,
                  InnerProductDistance(*FixedVector<int8_t, 47>::Cast(a47),
                                       *FixedVector<int8_t, 47>::Cast(b47)));
}

TEST(DistanceMatrix, MinusInnerProduct_General) {
  int8_t a1[] = {0};
  int8_t b1[] = {0};

  int8_t a17[] = {127, -1,  -1,  127, 127, 127, 127, -1, 127,
                  127, 127, 127, 127, 127, -1,  -1,  127};
  int8_t b17[] = {127, -1,  -1,  127, 127, 127, -1, 127, 127,
                  127, 127, 127, 127, 127, -1,  -1, 127};

  int8_t a47[] = {127, 2, 0,    0,    -127, -127, 0,    0,    0,    0,
                  0,   0, -127, -127, 127,  127,  0,    0,    -127, -127,
                  0,   0, 127,  5,    127,  127,  0,    0,    -127, -127,
                  0,   0, -127, 126,  -127, -127, -127, -127, 127,  127,
                  1,   2, 3,    4,    127,  127,  111};
  int8_t b47[] = {-127, 1, 0,    0,   127,  127,  0,   0,    0,   0,
                  0,    0, 127,  127, -127, -127, 0,   0,    127, 127,
                  0,    0, -127, 3,   -127, -127, 0,   0,    127, 127,
                  0,    0, 127,  127, 100,  122,  123, -127, 1,   2,
                  3,    4, -127, 122, -127, -127, -127};

  EXPECT_FLOAT_EQ(0.0f,
                  MinusInnerProductDistance(*FixedVector<int8_t, 1>::Cast(a1),
                                            *FixedVector<int8_t, 1>::Cast(b1)));
  EXPECT_FLOAT_EQ(-177169.0f, MinusInnerProductDistance(
                                  *FixedVector<int8_t, 17>::Cast(a17),
                                  *FixedVector<int8_t, 17>::Cast(b17)));
  EXPECT_FLOAT_EQ(299458.0f, MinusInnerProductDistance(
                                 *FixedVector<int8_t, 47>::Cast(a47),
                                 *FixedVector<int8_t, 47>::Cast(b47)));
}

template <size_t M, size_t N>
void TestMinusInnerProductMatrix(void) {
  std::mt19937 gen((std::random_device())());

  const size_t batch_size = M;
  const size_t query_size = N;
  size_t dimension = (std::uniform_int_distribution<size_t>(1, 64))(gen) << 2;
  size_t matrix_size = batch_size * dimension;
  size_t query_matrix_size = query_size * dimension;

  std::vector<int8_t> matrix1(matrix_size);
  std::vector<int8_t> matrix2(matrix_size);
  std::vector<int8_t> query1(query_matrix_size);
  std::vector<int8_t> query2(query_matrix_size);
  std::vector<float> result1(batch_size * query_size);
  std::vector<float> result2(batch_size * query_size);

  std::uniform_int_distribution<int> dist(-127, 127);
  for (size_t i = 0; i < matrix_size; ++i) {
    matrix1[i] = (int8_t)dist(gen);
  }
  for (size_t i = 0; i < query_matrix_size; ++i) {
    query1[i] = (int8_t)dist(gen);
  }
  MatrixTranspose((uint32_t *)(&matrix2[0]), (const uint32_t *)matrix1.data(),
                  dimension / 4, batch_size);
  MatrixTranspose((uint32_t *)(&query2[0]), (const uint32_t *)query1.data(),
                  dimension / 4, query_size);

  for (size_t i = 0; i < query_size; ++i) {
    const int8_t *cur_query = &query1[i * dimension];
    float *query_result = &result1[i * batch_size];

    for (size_t j = 0; j < batch_size; ++j) {
      MinusInnerProductMatrix<int8_t, 1, 1>::Compute(
          &matrix1[j * dimension], cur_query, dimension, &query_result[j]);
    }
  }
  MinusInnerProductMatrix<int8_t, batch_size, query_size>::Compute(
      &matrix2[0], &query2[0], dimension, &result2[0]);

  for (size_t i = 0; i < batch_size * query_size; ++i) {
    EXPECT_FLOAT_EQ(result1[i], result2[i]);
  }
}

template <size_t M, size_t N>
void TestInnerProductMatrix(void) {
  std::mt19937 gen((std::random_device())());

  const size_t batch_size = M;
  const size_t query_size = N;
  size_t dimension = (std::uniform_int_distribution<size_t>(1, 64))(gen) << 2;
  size_t matrix_size = batch_size * dimension;
  size_t query_matrix_size = query_size * dimension;

  std::vector<int8_t> matrix1(matrix_size);
  std::vector<int8_t> matrix2(matrix_size);
  std::vector<int8_t> query1(query_matrix_size);
  std::vector<int8_t> query2(query_matrix_size);
  std::vector<float> result1(batch_size * query_size);
  std::vector<float> result2(batch_size * query_size);

  std::uniform_int_distribution<int> dist(-127, 127);
  for (size_t i = 0; i < matrix_size; ++i) {
    matrix1[i] = (int8_t)dist(gen);
  }
  for (size_t i = 0; i < query_matrix_size; ++i) {
    query1[i] = (int8_t)dist(gen);
  }
  MatrixTranspose((uint32_t *)(&matrix2[0]), (const uint32_t *)matrix1.data(),
                  dimension / 4, batch_size);
  MatrixTranspose((uint32_t *)(&query2[0]), (const uint32_t *)query1.data(),
                  dimension / 4, query_size);

  for (size_t i = 0; i < query_size; ++i) {
    const int8_t *cur_query = &query1[i * dimension];
    float *query_result = &result1[i * batch_size];

    for (size_t j = 0; j < batch_size; ++j) {
      InnerProductMatrix<int8_t, 1, 1>::Compute(
          &matrix1[j * dimension], cur_query, dimension, &query_result[j]);
    }
  }
  InnerProductMatrix<int8_t, batch_size, query_size>::Compute(
      &matrix2[0], &query2[0], dimension, &result2[0]);

  for (size_t i = 0; i < batch_size * query_size; ++i) {
    EXPECT_FLOAT_EQ(result1[i], result2[i]);
  }
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

template <size_t M, size_t N, size_t B, size_t D>
void InnerProductBenchmark(void) {
  const size_t dimension = D;
  const size_t batch_size = M;
  const size_t block_size = B;
  const size_t query_size = N;
  const size_t matrix_size = block_size * batch_size * dimension;
  const size_t query_matrix_size = dimension * query_size;

  std::vector<int8_t> matrix1(matrix_size);
  std::vector<int8_t> matrix2(matrix_size);
  std::vector<int8_t> query1(query_matrix_size);
  std::vector<int8_t> query2(query_matrix_size);

  std::mt19937 gen((std::random_device())());
  std::uniform_int_distribution<int> dist(-127, 127);
  for (size_t i = 0; i < matrix_size; ++i) {
    matrix1[i] = (int8_t)dist(gen);
  }
  for (size_t i = 0; i < query_matrix_size; ++i) {
    query1[i] = (int8_t)dist(gen);
  }

  for (size_t i = 0; i < block_size; ++i) {
    size_t start_pos = i * batch_size * dimension;
    MatrixTranspose((uint32_t *)(&matrix2[start_pos]),
                    (const uint32_t *)(&matrix1[start_pos]), dimension / 4,
                    batch_size);
  }
  MatrixTranspose((uint32_t *)(&query2[0]), (const uint32_t *)query1.data(),
                  dimension / 4, query_size);

  ElapsedTime elapsed_time;
  float results[batch_size * query_size];

  std::cout << "# (" << IntelIntrinsics() << ") INT8 " << dimension << "d, "
            << batch_size << " * " << query_size << " * " << block_size
            << std::endl;

  // 1 Batched InnerProduct
  elapsed_time.reset();
  for (size_t i = 0; i < block_size; ++i) {
    const int8_t *matrix_batch = &matrix2[i * batch_size * dimension];

    for (size_t j = 0; j < query_size; ++j) {
      const int8_t *current_query = &query1[j * dimension];
      float *current_results = &results[j * batch_size];

      InnerProductMatrix<int8_t, batch_size, 1>::Compute(
          matrix_batch, current_query, dimension, current_results);
    }
  }
  std::cout << "* 1 Batched InnerProduct (us) \t"
            << elapsed_time.micro_seconds() << std::endl;

  // N Batched InnerProduct
  elapsed_time.reset();
  for (size_t i = 0; i < block_size; ++i) {
    const int8_t *matrix_batch = &matrix2[i * batch_size * dimension];

    InnerProductMatrix<int8_t, batch_size, query_size>::Compute(
        matrix_batch, &query2[0], dimension, results);
  }
  std::cout << "* N Batched InnerProduct (us) \t"
            << elapsed_time.micro_seconds() << std::endl;

  // Unbatched InnerProduct
  elapsed_time.reset();
  for (size_t i = 0; i < block_size; ++i) {
    const int8_t *matrix_batch = &matrix1[i * batch_size * dimension];

    for (size_t j = 0; j < query_size; ++j) {
      const int8_t *current_query = &query1[j * dimension];
      float *current_results = &results[j * batch_size];

      for (size_t k = 0; k < batch_size; ++k) {
        InnerProductMatrix<int8_t, 1, 1>::Compute(&matrix_batch[k * dimension],
                                                  current_query, dimension,
                                                  &current_results[k]);
      }
    }
  }
  std::cout << "* Unbatched InnerProduct (us) \t"
            << elapsed_time.micro_seconds() << std::endl;
}

template <size_t M, size_t N, size_t B, size_t D>
void MinusInnerProductBenchmark(void) {
  const size_t dimension = D;
  const size_t batch_size = M;
  const size_t block_size = B;
  const size_t query_size = N;
  const size_t matrix_size = block_size * batch_size * dimension;
  const size_t query_matrix_size = dimension * query_size;

  std::vector<int8_t> matrix1(matrix_size);
  std::vector<int8_t> matrix2(matrix_size);
  std::vector<int8_t> query1(query_matrix_size);
  std::vector<int8_t> query2(query_matrix_size);

  std::mt19937 gen((std::random_device())());
  std::uniform_int_distribution<int> dist(-127, 127);
  for (size_t i = 0; i < matrix_size; ++i) {
    matrix1[i] = (int8_t)dist(gen);
  }
  for (size_t i = 0; i < query_matrix_size; ++i) {
    query1[i] = (int8_t)dist(gen);
  }

  for (size_t i = 0; i < block_size; ++i) {
    size_t start_pos = i * batch_size * dimension;
    MatrixTranspose((uint32_t *)(&matrix2[start_pos]),
                    (const uint32_t *)(&matrix1[start_pos]), dimension / 4,
                    batch_size);
  }
  MatrixTranspose((uint32_t *)(&query2[0]), (const uint32_t *)query1.data(),
                  dimension / 4, query_size);

  ElapsedTime elapsed_time;
  float results[batch_size * query_size];

  std::cout << "# (" << IntelIntrinsics() << ") INT8 " << dimension << "d, "
            << batch_size << " * " << query_size << " * " << block_size
            << std::endl;

  // 1 Batched MinusInnerProduct
  elapsed_time.reset();
  for (size_t i = 0; i < block_size; ++i) {
    const int8_t *matrix_batch = &matrix2[i * batch_size * dimension];

    for (size_t j = 0; j < query_size; ++j) {
      const int8_t *current_query = &query1[j * dimension];
      float *current_results = &results[j * batch_size];

      MinusInnerProductMatrix<int8_t, batch_size, 1>::Compute(
          matrix_batch, current_query, dimension, current_results);
    }
  }
  std::cout << "* 1 Batched MinusInnerProduct (us) \t"
            << elapsed_time.micro_seconds() << std::endl;

  // N Batched MinusInnerProduct
  elapsed_time.reset();
  for (size_t i = 0; i < block_size; ++i) {
    const int8_t *matrix_batch = &matrix2[i * batch_size * dimension];

    MinusInnerProductMatrix<int8_t, batch_size, query_size>::Compute(
        matrix_batch, &query2[0], dimension, results);
  }
  std::cout << "* N Batched MinusInnerProduct (us) \t"
            << elapsed_time.micro_seconds() << std::endl;

  // Unbatched MinusInnerProduct
  elapsed_time.reset();
  for (size_t i = 0; i < block_size; ++i) {
    const int8_t *matrix_batch = &matrix1[i * batch_size * dimension];

    for (size_t j = 0; j < query_size; ++j) {
      const int8_t *current_query = &query1[j * dimension];
      float *current_results = &results[j * batch_size];

      for (size_t k = 0; k < batch_size; ++k) {
        MinusInnerProductMatrix<int8_t, 1, 1>::Compute(
            &matrix_batch[k * dimension], current_query, dimension,
            &current_results[k]);
      }
    }
  }
  std::cout << "* Unbatched MinusInnerProduct (us) \t"
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

TEST(DistanceMatrix, DISABLED_MinusInnerProduct_Benchmark) {
  MinusInnerProductBenchmark<2, 1, 512, 128>();
  MinusInnerProductBenchmark<2, 2, 512, 128>();
  MinusInnerProductBenchmark<4, 1, 512, 128>();
  MinusInnerProductBenchmark<4, 2, 512, 128>();
  MinusInnerProductBenchmark<4, 4, 512, 128>();
  MinusInnerProductBenchmark<8, 1, 512, 128>();
  MinusInnerProductBenchmark<8, 2, 512, 128>();
  MinusInnerProductBenchmark<8, 4, 512, 128>();
  MinusInnerProductBenchmark<8, 8, 512, 128>();
  MinusInnerProductBenchmark<16, 1, 512, 128>();
  MinusInnerProductBenchmark<16, 2, 512, 128>();
  MinusInnerProductBenchmark<16, 4, 512, 128>();
  MinusInnerProductBenchmark<16, 8, 512, 128>();
  MinusInnerProductBenchmark<16, 16, 512, 128>();
  MinusInnerProductBenchmark<32, 1, 512, 128>();
  MinusInnerProductBenchmark<32, 2, 512, 128>();
  MinusInnerProductBenchmark<32, 4, 512, 128>();
  MinusInnerProductBenchmark<32, 8, 512, 128>();
  MinusInnerProductBenchmark<32, 16, 512, 128>();
  MinusInnerProductBenchmark<32, 32, 512, 128>();
  MinusInnerProductBenchmark<64, 1, 512, 128>();
  MinusInnerProductBenchmark<64, 2, 512, 128>();
  MinusInnerProductBenchmark<64, 4, 512, 128>();
  MinusInnerProductBenchmark<64, 8, 512, 128>();
  MinusInnerProductBenchmark<128, 1, 512, 128>();
}
