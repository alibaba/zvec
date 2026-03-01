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
#include <ailego/utility/math_helper.h>
#include <gtest/gtest.h>
#include <zvec/ailego/container/vector.h>
#include <zvec/ailego/utility/time_helper.h>

using namespace zvec::ailego;

static inline const char *IntelIntrinsics(void) {
  return internal::CpuFeatures::Intrinsics();
}

static inline void MatrixTranspose(Float16 *dst, const Float16 *src, size_t M,
                                   size_t N) {
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < M; ++j) {
      dst[j * N + i] = src[i * M + j];
    }
  }
}

template <size_t N>
static float EuclideanDistance(const FixedVector<Float16, N> &lhs,
                               const FixedVector<Float16, N> &rhs) {
  return Distance::Euclidean(lhs.data(), rhs.data(), lhs.size());
}

template <size_t N>
static float SquaredEuclideanDistance(const FixedVector<Float16, N> &lhs,
                                      const FixedVector<Float16, N> &rhs) {
  return Distance::SquaredEuclidean(lhs.data(), rhs.data(), lhs.size());
}

TEST(DistanceMatrix, Euclidean_General) {
  FixedVector<Float16, 1> a{0.0f}, b{0.0f};
  EXPECT_FLOAT_EQ(0.0f, EuclideanDistance(a, b));

  FixedVector<Float16, 3> c{1.0f, 2.0f, 3.0f}, d{2.0f, 4.0f, 6.0f};
  EXPECT_FLOAT_EQ(3.741657f, EuclideanDistance(c, d));

  FixedVector<Float16, 11> e{1.0f, 2.0f, 3.0f, 0.2f, 0.3f, 0.1f,
                             5.2f, 2.1f, 7.1f, 6.8f, 1.2f},
      f{2.0f, 4.0f, 6.0f, 0.6f, 0.7f, 0.9f, 1.0f, 2.3f, 3.4f, 4.5f, 6.4f};
  EXPECT_TRUE(
      MathHelper::IsAlmostEqual(8.86837f, EuclideanDistance(e, f), 1000));
}

TEST(DistanceMatrix, SquaredEuclidean_General) {
  FixedVector<Float16, 1> a{0.0f}, b{0.0f};
  EXPECT_FLOAT_EQ(0.0f, SquaredEuclideanDistance(a, b));

  FixedVector<Float16, 2> c{0.0f, 0.1f}, d{0.0f, 0.1f};
  EXPECT_FLOAT_EQ(0.0f, SquaredEuclideanDistance(c, d));

  FixedVector<Float16, 3> e{0.0f, 0.1f, 0.2f}, f{0.0f, 0.1f, 0.2f};
  EXPECT_FLOAT_EQ(0.0f, SquaredEuclideanDistance(e, f));

  FixedVector<Float16, 4> g{0.0f, 0.1f, 0.2f, 0.3f}, h{0.0f, 0.1f, 0.2f, 0.3f};
  EXPECT_FLOAT_EQ(0.0f, SquaredEuclideanDistance(g, h));

  FixedVector<Float16, 5> i{0.0f, 0.1f, 0.2f, 0.3f, 0.4f},
      j{0.0f, 0.1f, 0.2f, 0.3f, 0.4f};
  EXPECT_FLOAT_EQ(0.0f, SquaredEuclideanDistance(i, j));

  FixedVector<Float16, 6> l{0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f},
      k{0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
  EXPECT_FLOAT_EQ(0.0f, SquaredEuclideanDistance(l, k));

  FixedVector<Float16, 7> m{0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f},
      n{0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
  EXPECT_FLOAT_EQ(0.0f, SquaredEuclideanDistance(m, n));

  FixedVector<Float16, 8> o{0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f},
      p{0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f};
  EXPECT_FLOAT_EQ(0.0f, SquaredEuclideanDistance(o, p));

  FixedVector<Float16, 9> q{0.0f, 0.1f, 0.2f, 0.3f, 0.4f,
                            0.5f, 0.6f, 0.7f, 0.8f},
      r{0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
  EXPECT_FLOAT_EQ(0.0f, SquaredEuclideanDistance(q, r));

  FixedVector<Float16, 10> s{0.0f, 0.1f, 0.2f, 0.3f, 0.4f,
                             0.5f, 0.6f, 0.7f, 0.8f, 0.9f},
      t{0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f};
  EXPECT_FLOAT_EQ(0.0f, SquaredEuclideanDistance(s, t));

  FixedVector<Float16, 11> u{0.0f},
      v{0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f};
  EXPECT_TRUE(MathHelper::IsAlmostEqual(3.84983f,
                                        SquaredEuclideanDistance(u, v), 1000));

  FixedVector<Float16, 12> w{0.0f},
      x{0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f};
  EXPECT_TRUE(MathHelper::IsAlmostEqual(5.05897f,
                                        SquaredEuclideanDistance(w, x), 1000));

  FixedVector<Float16, 13> y{0.0f}, z{0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
                                      0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f};
  EXPECT_TRUE(MathHelper::IsAlmostEqual(6.499438f,
                                        SquaredEuclideanDistance(y, z), 1000));

  FixedVector<Float16, 14> x14{0.0f},
      y14{0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
          0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 2.0f};
  EXPECT_TRUE(MathHelper::IsAlmostEqual(
      10.49944f, SquaredEuclideanDistance(x14, y14), 1000));

  FixedVector<Float16, 15> x15{0.0f},
      y15{0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f,
          0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 2.0f, 3.0f};
  EXPECT_TRUE(MathHelper::IsAlmostEqual(
      19.49944f, SquaredEuclideanDistance(x15, y15), 1000));
}

template <size_t M, size_t N>
void TestEuclideanMatrix(void) {
  std::mt19937 gen((std::random_device())());

  const size_t batch_size = M;
  const size_t query_size = N;
  size_t dimension = (std::uniform_int_distribution<size_t>(1, 65))(gen);
  size_t matrix_size = batch_size * dimension;
  size_t query_matrix_size = query_size * dimension;

  std::vector<Float16> matrix1(matrix_size);
  std::vector<Float16> matrix2(matrix_size);
  std::vector<Float16> query1(query_matrix_size);
  std::vector<Float16> query2(query_matrix_size);
  std::vector<float> result1(batch_size * query_size);
  std::vector<float> result2(batch_size * query_size);

  std::uniform_real_distribution<float> dist(0.0, 1.0);
  for (size_t i = 0; i < matrix_size; ++i) {
    matrix1[i] = dist(gen);
  }
  for (size_t i = 0; i < query_matrix_size; ++i) {
    query1[i] = dist(gen);
  }
  MatrixTranspose(&matrix2[0], matrix1.data(), dimension, batch_size);
  MatrixTranspose(&query2[0], query1.data(), dimension, query_size);

  for (size_t i = 0; i < query_size; ++i) {
    const Float16 *cur_query = &query1[i * dimension];
    float *query_result = &result1[i * batch_size];

    for (size_t j = 0; j < batch_size; ++j) {
      EuclideanDistanceMatrix<Float16, 1, 1>::Compute(
          &matrix1[j * dimension], cur_query, dimension, &query_result[j]);
    }
  }
  EuclideanDistanceMatrix<Float16, batch_size, query_size>::Compute(
      &matrix2[0], &query2[0], dimension, &result2[0]);

  for (size_t i = 0; i < batch_size * query_size; ++i) {
    // EXPECT_FLOAT_EQ(result1[i], result2[i]);
    EXPECT_TRUE(MathHelper::IsAlmostEqual(result1[i], result2[i], 10000));
  }
}

template <size_t M, size_t N>
void TestSquaredEuclideanMatrix(void) {
  std::mt19937 gen((std::random_device())());

  const size_t batch_size = M;
  const size_t query_size = N;
  size_t dimension = (std::uniform_int_distribution<size_t>(1, 65))(gen);
  size_t matrix_size = batch_size * dimension;
  size_t query_matrix_size = query_size * dimension;

  std::vector<Float16> matrix1(matrix_size);
  std::vector<Float16> matrix2(matrix_size);
  std::vector<Float16> query1(query_matrix_size);
  std::vector<Float16> query2(query_matrix_size);
  std::vector<float> result1(batch_size * query_size);
  std::vector<float> result2(batch_size * query_size);

  std::uniform_real_distribution<float> dist(0.0, 1.0);
  for (size_t i = 0; i < matrix_size; ++i) {
    matrix1[i] = dist(gen);
  }
  for (size_t i = 0; i < query_matrix_size; ++i) {
    query1[i] = dist(gen);
  }
  MatrixTranspose(&matrix2[0], matrix1.data(), dimension, batch_size);
  MatrixTranspose(&query2[0], query1.data(), dimension, query_size);

  for (size_t i = 0; i < query_size; ++i) {
    const Float16 *cur_query = &query1[i * dimension];
    float *query_result = &result1[i * batch_size];

    for (size_t j = 0; j < batch_size; ++j) {
      SquaredEuclideanDistanceMatrix<Float16, 1, 1>::Compute(
          &matrix1[j * dimension], cur_query, dimension, &query_result[j]);
    }
  }
  SquaredEuclideanDistanceMatrix<Float16, batch_size, query_size>::Compute(
      &matrix2[0], &query2[0], dimension, &result2[0]);

  for (size_t i = 0; i < batch_size * query_size; ++i) {
    // EXPECT_FLOAT_EQ(result1[i], result2[i]);
    EXPECT_TRUE(MathHelper::IsAlmostEqual(result1[i], result2[i], 10000));
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
  TestEuclideanMatrix<64, 64>();
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
  TestEuclideanMatrix<128, 64>();
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
  TestSquaredEuclideanMatrix<64, 64>();
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
  TestSquaredEuclideanMatrix<128, 64>();
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
  const size_t matrix_size = block_size * batch_size * dimension;
  const size_t query_matrix_size = dimension * query_size;

  std::vector<Float16> matrix1(matrix_size);
  std::vector<Float16> matrix2(matrix_size);
  std::vector<Float16> query1(query_matrix_size);
  std::vector<Float16> query2(query_matrix_size);

  std::mt19937 gen((std::random_device())());
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < matrix_size; ++i) {
    matrix1[i] = dist(gen);
  }
  for (size_t i = 0; i < query_matrix_size; ++i) {
    query1[i] = dist(gen);
  }

  for (size_t i = 0; i < block_size; ++i) {
    size_t start_pos = i * batch_size * dimension;
    MatrixTranspose(&matrix2[start_pos], &matrix1[start_pos], dimension,
                    batch_size);
  }
  MatrixTranspose(&query2[0], query1.data(), dimension, query_size);

  ElapsedTime elapsed_time;
  float results[batch_size * query_size];

  std::cout << "# (" << IntelIntrinsics() << ") FP16 " << dimension << "d, "
            << batch_size << " * " << query_size << " * " << block_size
            << std::endl;

  // 1 Batched Euclidean
  elapsed_time.reset();
  for (size_t i = 0; i < block_size; ++i) {
    const Float16 *matrix_batch = &matrix2[i * batch_size * dimension];

    for (size_t j = 0; j < query_size; ++j) {
      const Float16 *current_query = &query1[j * dimension];
      float *current_results = &results[j * batch_size];

      EuclideanDistanceMatrix<Float16, batch_size, 1>::Compute(
          matrix_batch, current_query, dimension, current_results);
    }
  }
  std::cout << "* 1 Batched Euclidean (us) \t" << elapsed_time.micro_seconds()
            << std::endl;

  // N Batched Euclidean
  elapsed_time.reset();
  for (size_t i = 0; i < block_size; ++i) {
    const Float16 *matrix_batch = &matrix2[i * batch_size * dimension];

    EuclideanDistanceMatrix<Float16, batch_size, query_size>::Compute(
        matrix_batch, &query2[0], dimension, results);
  }
  std::cout << "* N Batched Euclidean (us) \t" << elapsed_time.micro_seconds()
            << std::endl;

  // Unbatched Euclidean
  elapsed_time.reset();
  for (size_t i = 0; i < block_size; ++i) {
    const Float16 *matrix_batch = &matrix1[i * batch_size * dimension];

    for (size_t j = 0; j < query_size; ++j) {
      const Float16 *current_query = &query1[j * dimension];
      float *current_results = &results[j * batch_size];

      for (size_t k = 0; k < batch_size; ++k) {
        EuclideanDistanceMatrix<Float16, 1, 1>::Compute(
            &matrix_batch[k * dimension], current_query, dimension,
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
  const size_t matrix_size = block_size * batch_size * dimension;
  const size_t query_matrix_size = dimension * query_size;

  std::vector<Float16> matrix1(matrix_size);
  std::vector<Float16> matrix2(matrix_size);
  std::vector<Float16> query1(query_matrix_size);
  std::vector<Float16> query2(query_matrix_size);

  std::mt19937 gen((std::random_device())());
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < matrix_size; ++i) {
    matrix1[i] = dist(gen);
  }
  for (size_t i = 0; i < query_matrix_size; ++i) {
    query1[i] = dist(gen);
  }

  for (size_t i = 0; i < block_size; ++i) {
    size_t start_pos = i * batch_size * dimension;
    MatrixTranspose(&matrix2[start_pos], &matrix1[start_pos], dimension,
                    batch_size);
  }
  MatrixTranspose(&query2[0], query1.data(), dimension, query_size);

  ElapsedTime elapsed_time;
  float results[batch_size * query_size];

  std::cout << "# (" << IntelIntrinsics() << ") FP16 " << dimension << "d, "
            << batch_size << " * " << query_size << " * " << block_size
            << std::endl;

  // 1 Batched Euclidean
  elapsed_time.reset();
  for (size_t i = 0; i < block_size; ++i) {
    const Float16 *matrix_batch = &matrix2[i * batch_size * dimension];

    for (size_t j = 0; j < query_size; ++j) {
      const Float16 *current_query = &query1[j * dimension];
      float *current_results = &results[j * batch_size];

      SquaredEuclideanDistanceMatrix<Float16, batch_size, 1>::Compute(
          matrix_batch, current_query, dimension, current_results);
    }
  }
  std::cout << "* 1 Batched SquaredEuclidean (us) \t"
            << elapsed_time.micro_seconds() << std::endl;

  // N Batched Euclidean
  elapsed_time.reset();
  for (size_t i = 0; i < block_size; ++i) {
    const Float16 *matrix_batch = &matrix2[i * batch_size * dimension];

    SquaredEuclideanDistanceMatrix<Float16, batch_size, query_size>::Compute(
        matrix_batch, &query2[0], dimension, results);
  }
  std::cout << "* N Batched SquaredEuclidean (us) \t"
            << elapsed_time.micro_seconds() << std::endl;

  // Unbatched Euclidean
  elapsed_time.reset();
  for (size_t i = 0; i < block_size; ++i) {
    const Float16 *matrix_batch = &matrix1[i * batch_size * dimension];

    for (size_t j = 0; j < query_size; ++j) {
      const Float16 *current_query = &query1[j * dimension];
      float *current_results = &results[j * batch_size];

      for (size_t k = 0; k < batch_size; ++k) {
        SquaredEuclideanDistanceMatrix<Float16, 1, 1>::Compute(
            &matrix_batch[k * dimension], current_query, dimension,
            &current_results[k]);
      }
    }
  }
  std::cout << "* Unbatched SquaredEuclidean (us) \t"
            << elapsed_time.micro_seconds() << std::endl;
}

TEST(DistanceMatrix, DISABLED_Euclidean_Benchmark) {
  EuclideanBenchmark<2, 1, 512, 64>();
  EuclideanBenchmark<2, 2, 512, 64>();
  EuclideanBenchmark<4, 1, 512, 64>();
  EuclideanBenchmark<4, 2, 512, 64>();
  EuclideanBenchmark<4, 4, 512, 64>();
  EuclideanBenchmark<8, 1, 512, 64>();
  EuclideanBenchmark<8, 2, 512, 64>();
  EuclideanBenchmark<8, 4, 512, 64>();
  EuclideanBenchmark<8, 8, 512, 64>();
  EuclideanBenchmark<16, 1, 512, 64>();
  EuclideanBenchmark<16, 2, 512, 64>();
  EuclideanBenchmark<16, 4, 512, 64>();
  EuclideanBenchmark<16, 8, 512, 64>();
  EuclideanBenchmark<16, 16, 512, 64>();
  EuclideanBenchmark<32, 1, 512, 64>();
  EuclideanBenchmark<32, 2, 512, 64>();
  EuclideanBenchmark<32, 4, 512, 64>();
  EuclideanBenchmark<32, 8, 512, 64>();
  EuclideanBenchmark<32, 16, 512, 64>();
  EuclideanBenchmark<32, 32, 512, 64>();
  EuclideanBenchmark<64, 1, 512, 64>();
  EuclideanBenchmark<64, 2, 512, 64>();
  EuclideanBenchmark<64, 4, 512, 64>();
  EuclideanBenchmark<64, 8, 512, 64>();
  EuclideanBenchmark<128, 1, 512, 64>();
  EuclideanBenchmark<1, 1, 1024, 256>();
}

TEST(DistanceMatrix, DISABLED_SquaredEuclidean_Benchmark) {
  SquaredEuclideanBenchmark<2, 1, 512, 64>();
  SquaredEuclideanBenchmark<2, 2, 512, 64>();
  SquaredEuclideanBenchmark<4, 1, 512, 64>();
  SquaredEuclideanBenchmark<4, 2, 512, 64>();
  SquaredEuclideanBenchmark<4, 4, 512, 64>();
  SquaredEuclideanBenchmark<8, 1, 512, 64>();
  SquaredEuclideanBenchmark<8, 2, 512, 64>();
  SquaredEuclideanBenchmark<8, 4, 512, 64>();
  SquaredEuclideanBenchmark<8, 8, 512, 64>();
  SquaredEuclideanBenchmark<16, 1, 512, 64>();
  SquaredEuclideanBenchmark<16, 2, 512, 64>();
  SquaredEuclideanBenchmark<16, 4, 512, 64>();
  SquaredEuclideanBenchmark<16, 8, 512, 64>();
  SquaredEuclideanBenchmark<16, 16, 512, 64>();
  SquaredEuclideanBenchmark<32, 1, 512, 64>();
  SquaredEuclideanBenchmark<32, 2, 512, 64>();
  SquaredEuclideanBenchmark<32, 4, 512, 64>();
  SquaredEuclideanBenchmark<32, 8, 512, 64>();
  SquaredEuclideanBenchmark<32, 16, 512, 64>();
  SquaredEuclideanBenchmark<32, 32, 512, 64>();
  SquaredEuclideanBenchmark<64, 1, 512, 64>();
  SquaredEuclideanBenchmark<64, 2, 512, 64>();
  SquaredEuclideanBenchmark<64, 4, 512, 64>();
  SquaredEuclideanBenchmark<64, 8, 512, 64>();
  SquaredEuclideanBenchmark<128, 1, 512, 64>();
  SquaredEuclideanBenchmark<1, 1, 1024, 256>();
}

TEST(DistanceMatrix, DISABLED_Euclidean_BenchmarkSimple) {
  std::mt19937 gen((std::random_device())());

  size_t dimension = 768;
  size_t loop_cnt = 10000LLU;

  std::vector<Float16> data(dimension);
  std::vector<Float16> query(dimension);

  float result;

  std::uniform_real_distribution<float> dist(0.0, 1.0);
  for (size_t i = 0; i < dimension; ++i) {
    data[i] = dist(gen);
  }
  for (size_t i = 0; i < dimension; ++i) {
    query[i] = dist(gen);
  }

  for (size_t i = 0; i < loop_cnt; ++i) {
    EuclideanDistanceMatrix<Float16, 1, 1>::Compute(&data[0], &query[0],
                                                    dimension, &result);
  }
}

TEST(DistanceMatrix, DISABLED_SquaredEuclidean_BenchmarkSimple) {
  std::mt19937 gen((std::random_device())());

  size_t dimension = 768;
  size_t loop_cnt = 10000LLU;

  std::vector<Float16> data(dimension);
  std::vector<Float16> query(dimension);

  float result;

  std::uniform_real_distribution<float> dist(0.0, 1.0);
  for (size_t i = 0; i < dimension; ++i) {
    data[i] = dist(gen);
  }
  for (size_t i = 0; i < dimension; ++i) {
    query[i] = dist(gen);
  }

  for (size_t i = 0; i < loop_cnt; ++i) {
    SquaredEuclideanDistanceMatrix<Float16, 1, 1>::Compute(&data[0], &query[0],
                                                           dimension, &result);
  }
}
