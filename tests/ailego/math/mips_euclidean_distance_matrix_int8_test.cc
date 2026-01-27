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
#include <ailego/math/norm2_matrix.h>
#include <gtest/gtest.h>
#include <zvec/ailego/container/vector.h>
#include <zvec/ailego/utility/time_helper.h>

using namespace zvec;
using namespace zvec::ailego;

static inline const char *IntelIntrinsics(void) {
  return internal::CpuFeatures::Intrinsics();
}

static inline void MatrixTranspose(uint32_t *dst, const uint32_t *src, size_t M,
                                   size_t N) {
  for (size_t n = 0; n < N * M; n++) {
    size_t i = n / N;
    size_t j = n % N;
    dst[n] = src[M * j + i];
  }
}

static float MipsSquaredEuclidean(const int8_t *lhs, const int8_t *rhs,
                                  size_t dim, size_t m_value, float e2) {
  return Distance::MipsSquaredEuclidean(lhs, rhs, dim, m_value, e2);
}

template <size_t N>
static float MipsSquaredEuclidean(const FixedVector<int8_t, N> &lhs,
                                  const FixedVector<int8_t, N> &rhs,
                                  size_t m_value, float e2) {
  return MipsSquaredEuclidean(lhs.data(), rhs.data(), lhs.size(), m_value, e2);
}

static float ConvertAndComputeByMips(const int8_t *lhs, const int8_t *rhs,
                                     size_t dim, size_t m_value, float e2) {
  float squ = 0.0f;
  float lhs_vec[dim + m_value];
  const float eta = std::sqrt(e2);
  for (size_t i = 0; i < dim; ++i) {
    float val = lhs[i] * eta;
    lhs_vec[i] = val;
    squ += val * val;
  }
  for (size_t i = dim; i < dim + m_value; ++i) {
    lhs_vec[i] = 0.5f - squ;
    squ *= squ;
  }
  float rhs_vec[dim + m_value];
  squ = 0.0f;
  for (size_t i = 0; i < dim; ++i) {
    float val = rhs[i] * eta;
    rhs_vec[i] = val;
    squ += val * val;
  }
  for (size_t i = dim; i < dim + m_value; ++i) {
    rhs_vec[i] = 0.5f - squ;
    squ *= squ;
  }
  return ailego::Distance::SquaredEuclidean(lhs_vec, rhs_vec, dim + m_value);
}

template <size_t N>
static float ConvertAndComputeByMips(const FixedVector<int8_t, N> &lhs,
                                     const FixedVector<int8_t, N> &rhs,
                                     size_t m_value, float e2) {
  return ConvertAndComputeByMips(lhs.data(), rhs.data(), lhs.size(), m_value,
                                 e2);
}

TEST(DistanceMatrix, GeneralRepeatedQuadraticInjection) {
  std::mt19937 gen((std::random_device())());
  const size_t m_val = std::uniform_int_distribution<size_t>(1, 4)(gen);
  const float u_val = std::uniform_real_distribution<float>(0.1, 1.0)(gen);
  const float l2_norm =
      std::uniform_real_distribution<float>(1000.0, 1500.0)(gen);
  const float e2 = (u_val / l2_norm) * (u_val / l2_norm);
  const float epsilon = 1e-6;
  const uint32_t dim = std::uniform_int_distribution<uint32_t>(2, 128)(gen);
  const uint32_t count = std::uniform_int_distribution<uint32_t>(1, 1000)(gen);
  std::uniform_int_distribution<int8_t> dist(-127, 127);
  for (size_t i = 0; i < count; ++i) {
    int8_t vec1[dim];
    int8_t vec2[dim];
    for (size_t d = 0; d < dim; ++d) {
      vec1[d] = dist(gen);
      vec2[d] = dist(gen);
    }
    ASSERT_NEAR(ConvertAndComputeByMips(vec1, vec2, dim, m_val, e2),
                MipsSquaredEuclidean(vec1, vec2, dim, m_val, e2), epsilon);
  }
}

TEST(DistanceMatrix, FixedVectorsRepeatedQuadraticInjection) {
  std::mt19937 gen((std::random_device())());
  const size_t m_val = 4;
  const float u_val = 0.68f;
  const float l2_norm = 30.0f;
  const float e2 = (u_val / l2_norm) * (u_val / l2_norm);
  const float epsilon = 1e-5;

  int8_t a[] = {0}, b[] = {0};
  EXPECT_NEAR(0.0f, MipsSquaredEuclidean(a, b, 1, m_val, e2), epsilon);

  int8_t c[] = {0, 1}, d[] = {0, 1};
  EXPECT_NEAR(0.0f, MipsSquaredEuclidean(c, d, 2, m_val, e2), epsilon);

  int8_t e[] = {0, 1, 2}, f[] = {0, 1, 2};
  EXPECT_NEAR(0.0f, MipsSquaredEuclidean(e, f, 3, m_val, e2), epsilon);

  int8_t g[] = {0, 1, 2, 3}, h[] = {0, 1, 2, 3};
  EXPECT_NEAR(0.0f, MipsSquaredEuclidean(g, h, 4, m_val, e2), epsilon);

  int8_t i[] = {0, 1, 2, 3, 4}, j[] = {0, 1, 2, 3, 4};
  EXPECT_NEAR(0.0f, MipsSquaredEuclidean(i, j, 5, m_val, e2), epsilon);

  int8_t l[] = {0, 1, 2, 3, 4, 5}, k[] = {0, 1, 2, 3, 4, 5};
  EXPECT_NEAR(0.0f, MipsSquaredEuclidean(l, k, 6, m_val, e2), epsilon);

  int8_t m[] = {0, 1, 2, 3, 4, 5, 6}, n[] = {0, 1, 2, 3, 4, 5, 6};
  EXPECT_NEAR(0.0f, MipsSquaredEuclidean(m, n, 7, m_val, e2), epsilon);

  int8_t o[] = {0, 1, 2, 3, 4, 5, 6, 7}, p[] = {0, 1, 2, 3, 4, 5, 6, 7};
  EXPECT_NEAR(0.0f, MipsSquaredEuclidean(o, p, 8, m_val, e2), epsilon);

  int8_t q[] = {0, 1, 2, 3, 4, 5, 6, 7, 8}, r[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  EXPECT_NEAR(0.0f, MipsSquaredEuclidean(q, r, 9, m_val, e2), epsilon);

  int8_t s[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
         t[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  EXPECT_NEAR(0.0f, MipsSquaredEuclidean(s, t, 10, m_val, e2), epsilon);

  int8_t u[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         v[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  EXPECT_NEAR(0.2384642f, MipsSquaredEuclidean(u, v, 11, m_val, e2), epsilon);

  int8_t w[12] = {0}, x[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  EXPECT_NEAR(0.3321453f, MipsSquaredEuclidean(w, x, 12, m_val, e2), epsilon);

  int8_t y[13] = {0}, z[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  EXPECT_NEAR(0.4580747f, MipsSquaredEuclidean(y, z, 13, m_val, e2), epsilon);

  int8_t x14[14] = {0}, y14[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20};
  EXPECT_NEAR(0.9224106f, MipsSquaredEuclidean(x14, y14, 14, m_val, e2),
              epsilon);

  int8_t x15[15] = {0},
         y15[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20, 30};
  EXPECT_NEAR(5.0584077f, MipsSquaredEuclidean(x15, y15, 15, m_val, e2),
              epsilon);
}

template <size_t M, size_t N>
void TestSquaredEuclideanMatrixRepeatedQuadraticInjection(void) {
  std::mt19937 gen((std::random_device())());

  const size_t m_val = std::uniform_int_distribution<size_t>(1, 4)(gen);
  const float u_val = std::uniform_real_distribution<float>(0.3, 0.9)(gen);
  const size_t batch_size = M;
  const size_t query_size = N;
  size_t dimension = std::uniform_int_distribution<size_t>(2, 128)(gen) * 4;
  size_t matrix_size = batch_size * dimension;
  size_t query_matrix_size = query_size * dimension;

  std::vector<int8_t> matrix1(matrix_size);
  std::vector<int8_t> matrix2(matrix_size);
  std::vector<int8_t> query1(query_matrix_size);
  std::vector<int8_t> query2(query_matrix_size);
  std::vector<float> result1(batch_size * query_size);
  std::vector<float> result2(batch_size * query_size);

  std::uniform_int_distribution<int8_t> dist(-127, 127);
  for (size_t i = 0; i < matrix_size; ++i) {
    matrix1[i] = dist(gen);
  }
  for (size_t i = 0; i < query_matrix_size; ++i) {
    query1[i] = dist(gen);
  }
  float squared_l2_norm = 0.0f;
  for (size_t i = 0; i < matrix_size; i += dimension) {
    float score;
    SquaredNorm2Matrix<int8_t, 1>::Compute(&matrix1[i], dimension, &score);
    squared_l2_norm = std::max(squared_l2_norm, score);
  }
  for (size_t i = 0; i < query_matrix_size; i += dimension) {
    float score;
    SquaredNorm2Matrix<int8_t, 1>::Compute(&query1[i], dimension, &score);
    squared_l2_norm = std::max(squared_l2_norm, score);
  }
  const float e2 = u_val * u_val / squared_l2_norm;
  MatrixTranspose((uint32_t *)(&matrix2[0]), (const uint32_t *)matrix1.data(),
                  dimension / 4, batch_size);
  MatrixTranspose((uint32_t *)(&query2[0]), (const uint32_t *)query1.data(),
                  dimension / 4, query_size);

  for (size_t i = 0; i < query_size; ++i) {
    const int8_t *cur_query = &query1[i * dimension];
    float *query_result = &result1[i * batch_size];

    for (size_t j = 0; j < batch_size; ++j) {
      MipsSquaredEuclideanDistanceMatrix<int8_t, 1, 1>::Compute(
          &matrix1[j * dimension], cur_query, dimension, m_val, e2,
          &query_result[j]);
    }
  }
  MipsSquaredEuclideanDistanceMatrix<int8_t, batch_size, query_size>::Compute(
      &matrix2[0], &query2[0], dimension, m_val, e2, &result2[0]);

  for (size_t i = 0; i < batch_size * query_size; ++i) {
    EXPECT_NEAR(result1[i], result2[i], 1e-4);
  }
}

TEST(DistanceMatrix, MipsSquaredEuclideanRepeatedQuadraticInjection_1x1) {
  TestSquaredEuclideanMatrixRepeatedQuadraticInjection<1, 1>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanRepeatedQuadraticInjection_2x1) {
  TestSquaredEuclideanMatrixRepeatedQuadraticInjection<2, 1>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanRepeatedQuadraticInjection_2x2) {
  TestSquaredEuclideanMatrixRepeatedQuadraticInjection<2, 2>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanRepeatedQuadraticInjection_3x3) {
  TestSquaredEuclideanMatrixRepeatedQuadraticInjection<3, 3>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanRepeatedQuadraticInjection_4x1) {
  TestSquaredEuclideanMatrixRepeatedQuadraticInjection<4, 1>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanRepeatedQuadraticInjection_4x2) {
  TestSquaredEuclideanMatrixRepeatedQuadraticInjection<4, 2>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanRepeatedQuadraticInjection_4x4) {
  TestSquaredEuclideanMatrixRepeatedQuadraticInjection<4, 4>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanRepeatedQuadraticInjection_8x1) {
  TestSquaredEuclideanMatrixRepeatedQuadraticInjection<8, 1>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanRepeatedQuadraticInjection_8x2) {
  TestSquaredEuclideanMatrixRepeatedQuadraticInjection<8, 2>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanRepeatedQuadraticInjection_8x4) {
  TestSquaredEuclideanMatrixRepeatedQuadraticInjection<8, 4>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanRepeatedQuadraticInjection_8x8) {
  TestSquaredEuclideanMatrixRepeatedQuadraticInjection<8, 8>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanRepeatedQuadraticInjection_16x1) {
  TestSquaredEuclideanMatrixRepeatedQuadraticInjection<16, 1>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanRepeatedQuadraticInjection_16x2) {
  TestSquaredEuclideanMatrixRepeatedQuadraticInjection<16, 2>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanRepeatedQuadraticInjection_16x4) {
  TestSquaredEuclideanMatrixRepeatedQuadraticInjection<16, 4>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanRepeatedQuadraticInjection_16x8) {
  TestSquaredEuclideanMatrixRepeatedQuadraticInjection<16, 8>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanRepeatedQuadraticInjection_16x16) {
  TestSquaredEuclideanMatrixRepeatedQuadraticInjection<16, 16>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanRepeatedQuadraticInjection_32x1) {
  TestSquaredEuclideanMatrixRepeatedQuadraticInjection<32, 1>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanRepeatedQuadraticInjection_32x2) {
  TestSquaredEuclideanMatrixRepeatedQuadraticInjection<32, 2>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanRepeatedQuadraticInjection_32x4) {
  TestSquaredEuclideanMatrixRepeatedQuadraticInjection<32, 4>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanRepeatedQuadraticInjection_32x8) {
  TestSquaredEuclideanMatrixRepeatedQuadraticInjection<32, 8>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanRepeatedQuadraticInjection_32x16) {
  TestSquaredEuclideanMatrixRepeatedQuadraticInjection<32, 16>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanRepeatedQuadraticInjection_32x32) {
  TestSquaredEuclideanMatrixRepeatedQuadraticInjection<32, 32>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanRepeatedQuadraticInjection_64x1) {
  TestSquaredEuclideanMatrixRepeatedQuadraticInjection<64, 1>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanRepeatedQuadraticInjection_64x2) {
  TestSquaredEuclideanMatrixRepeatedQuadraticInjection<64, 2>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanRepeatedQuadraticInjection_64x4) {
  TestSquaredEuclideanMatrixRepeatedQuadraticInjection<64, 4>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanRepeatedQuadraticInjection_64x8) {
  TestSquaredEuclideanMatrixRepeatedQuadraticInjection<64, 8>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanRepeatedQuadraticInjection_64x16) {
  TestSquaredEuclideanMatrixRepeatedQuadraticInjection<64, 16>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanRepeatedQuadraticInjection_64x32) {
  TestSquaredEuclideanMatrixRepeatedQuadraticInjection<64, 32>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanRepeatedQuadraticInjection_64x64) {
  TestSquaredEuclideanMatrixRepeatedQuadraticInjection<64, 64>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanRepeatedQuadraticInjection_128x1) {
  TestSquaredEuclideanMatrixRepeatedQuadraticInjection<128, 1>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanRepeatedQuadraticInjection_128x2) {
  TestSquaredEuclideanMatrixRepeatedQuadraticInjection<128, 2>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanRepeatedQuadraticInjection_128x4) {
  TestSquaredEuclideanMatrixRepeatedQuadraticInjection<128, 4>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanRepeatedQuadraticInjection_128x8) {
  TestSquaredEuclideanMatrixRepeatedQuadraticInjection<128, 8>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanRepeatedQuadraticInjection_128x16) {
  TestSquaredEuclideanMatrixRepeatedQuadraticInjection<128, 16>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanRepeatedQuadraticInjection_128x32) {
  TestSquaredEuclideanMatrixRepeatedQuadraticInjection<128, 32>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanRepeatedQuadraticInjection_128x64) {
  TestSquaredEuclideanMatrixRepeatedQuadraticInjection<128, 64>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanRepeatedQuadraticInjection_128x128) {
  TestSquaredEuclideanMatrixRepeatedQuadraticInjection<128, 128>();
}

template <size_t M, size_t N, size_t B, size_t D>
void MipsRepeatedQuadraticInjectionBenchMark(void) {
  const size_t m_val = 4;
  const float u_val = 0.6;
  const float l2_norm = 1.0f;
  const float e2 = (u_val / l2_norm) * (u_val / l2_norm);
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
  std::uniform_int_distribution<int8_t> dist(-127, 127);
  for (size_t i = 0; i < matrix_size; ++i) {
    matrix1[i] = dist(gen);
  }
  for (size_t i = 0; i < query_matrix_size; ++i) {
    query1[i] = dist(gen);
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

  // 1 Batched Euclidean
  elapsed_time.reset();
  for (size_t i = 0; i < block_size; ++i) {
    const int8_t *matrix_batch = &matrix2[i * batch_size * dimension];

    for (size_t j = 0; j < query_size; ++j) {
      const int8_t *current_query = &query1[j * dimension];
      float *current_results = &results[j * batch_size];

      MipsSquaredEuclideanDistanceMatrix<int8_t, batch_size, 1>::Compute(
          matrix_batch, current_query, dimension, m_val, e2, current_results);
    }
  }
  std::cout
      << "* 1 Batched MipsSquaredEuclidean(RepeatedQuadraticInjection) (us) \t"
      << elapsed_time.micro_seconds() << std::endl;

  // N Batched Euclidean
  elapsed_time.reset();
  for (size_t i = 0; i < block_size; ++i) {
    const int8_t *matrix_batch = &matrix2[i * batch_size * dimension];

    MipsSquaredEuclideanDistanceMatrix<int8_t, batch_size, query_size>::Compute(
        matrix_batch, &query2[0], dimension, m_val, e2, results);
  }
  std::cout
      << "* N Batched MipsSquaredErclidean(RepeatedQuadraticInjection) (us) \t"
      << elapsed_time.micro_seconds() << std::endl;

  // Unbatched Euclidean
  elapsed_time.reset();
  for (size_t i = 0; i < block_size; ++i) {
    const int8_t *matrix_batch = &matrix1[i * batch_size * dimension];

    for (size_t j = 0; j < query_size; ++j) {
      const int8_t *current_query = &query1[j * dimension];
      float *current_results = &results[j * batch_size];

      for (size_t k = 0; k < batch_size; ++k) {
        MipsSquaredEuclideanDistanceMatrix<int8_t, 1, 1>::Compute(
            &matrix_batch[k * dimension], current_query, dimension, m_val, e2,
            &current_results[k]);
      }
    }
  }
  std::cout
      << "* Unbatched MipsSquaredEuclidean(RepeatedQuadraticInjection) (us) \t"
      << elapsed_time.micro_seconds() << std::endl;
}

TEST(DistanceMatrix,
     DISABLED_MipsSquaredEuclideanRepeatedQuadraticInjection_Benchmark) {
  MipsRepeatedQuadraticInjectionBenchMark<2, 1, 512, 64>();
  MipsRepeatedQuadraticInjectionBenchMark<2, 2, 512, 64>();
  MipsRepeatedQuadraticInjectionBenchMark<4, 1, 512, 64>();
  MipsRepeatedQuadraticInjectionBenchMark<4, 2, 512, 64>();
  MipsRepeatedQuadraticInjectionBenchMark<4, 4, 512, 64>();
  MipsRepeatedQuadraticInjectionBenchMark<8, 1, 512, 64>();
  MipsRepeatedQuadraticInjectionBenchMark<8, 2, 512, 64>();
  MipsRepeatedQuadraticInjectionBenchMark<8, 4, 512, 64>();
  MipsRepeatedQuadraticInjectionBenchMark<8, 8, 512, 64>();
  MipsRepeatedQuadraticInjectionBenchMark<16, 1, 512, 64>();
  MipsRepeatedQuadraticInjectionBenchMark<16, 2, 512, 64>();
  MipsRepeatedQuadraticInjectionBenchMark<16, 4, 512, 64>();
  MipsRepeatedQuadraticInjectionBenchMark<16, 8, 512, 64>();
  MipsRepeatedQuadraticInjectionBenchMark<16, 16, 512, 64>();
  MipsRepeatedQuadraticInjectionBenchMark<32, 1, 512, 64>();
  MipsRepeatedQuadraticInjectionBenchMark<32, 2, 512, 64>();
  MipsRepeatedQuadraticInjectionBenchMark<32, 4, 512, 64>();
  MipsRepeatedQuadraticInjectionBenchMark<32, 8, 512, 64>();
  MipsRepeatedQuadraticInjectionBenchMark<32, 16, 512, 64>();
  MipsRepeatedQuadraticInjectionBenchMark<32, 32, 512, 64>();
  MipsRepeatedQuadraticInjectionBenchMark<64, 1, 512, 64>();
  MipsRepeatedQuadraticInjectionBenchMark<64, 2, 512, 64>();
  MipsRepeatedQuadraticInjectionBenchMark<64, 4, 512, 64>();
  MipsRepeatedQuadraticInjectionBenchMark<64, 8, 512, 64>();
  MipsRepeatedQuadraticInjectionBenchMark<128, 1, 512, 64>();
  MipsRepeatedQuadraticInjectionBenchMark<1, 1, 1024, 256>();
}

static float MipsSquaredEuclidean(const int8_t *lhs, const int8_t *rhs,
                                  size_t dim, float e2) {
  return Distance::MipsSquaredEuclidean(lhs, rhs, dim, e2);
}

template <size_t N>
static float MipsSquaredEuclidean(const FixedVector<int8_t, N> &lhs,
                                  const FixedVector<int8_t, N> &rhs, float e2) {
  return MipsSquaredEuclidean(lhs.data(), rhs.data(), lhs.size(), e2);
}

static float ConvertAndComputeByMips(const int8_t *lhs, const int8_t *rhs,
                                     size_t dim, float e2) {
  float squ = 0.0f;
  float lhs_vec[dim + 1];
  const float eta = std::sqrt(e2);
  for (size_t i = 0; i < dim; ++i) {
    float val = lhs[i] * eta;
    lhs_vec[i] = val;
    squ += val * val;
  }
  float norm2;
  ailego::SquaredNorm2Matrix<float, 1>::Compute(lhs_vec, dim, &norm2);
  lhs_vec[dim] = std::sqrt(1 - norm2);

  float rhs_vec[dim + 1];
  squ = 0.0f;
  for (size_t i = 0; i < dim; ++i) {
    float val = rhs[i] * eta;
    rhs_vec[i] = val;
    squ += val * val;
  }
  ailego::SquaredNorm2Matrix<float, 1>::Compute(rhs_vec, dim, &norm2);
  rhs_vec[dim] = std::sqrt(1 - norm2);
  std::cout << "squ: " << squ << std::endl;
  return ailego::Distance::SquaredEuclidean(lhs_vec, rhs_vec, dim + 1);
}

template <size_t N>
static float ConvertAndComputeByMips(const FixedVector<int8_t, N> &lhs,
                                     const FixedVector<int8_t, N> &rhs,
                                     float e2) {
  return ConvertAndComputeByMips(lhs.data(), rhs.data(), lhs.size(), e2);
}

TEST(DistanceMatrix, GeneralSphericalInjection) {
  std::mt19937 gen((std::random_device())());
  const float u_val = std::uniform_real_distribution<float>(0.1, 1.0)(gen);
  const float l2_norm =
      std::uniform_real_distribution<float>(1000.0, 1500.0)(gen);
  const float e2 = (u_val / l2_norm) * (u_val / l2_norm);
  const float epsilon = 1e-6;
  const uint32_t dim = std::uniform_int_distribution<uint32_t>(2, 128)(gen);
  const uint32_t count = std::uniform_int_distribution<uint32_t>(1, 1000)(gen);
  std::uniform_int_distribution<int8_t> dist(-127, 127);
  for (size_t i = 0; i < count; ++i) {
    int8_t vec1[dim];
    int8_t vec2[dim];
    for (size_t d = 0; d < dim; ++d) {
      vec1[d] = dist(gen);
      vec2[d] = dist(gen);
    }
    ASSERT_NEAR(ConvertAndComputeByMips(vec1, vec2, dim, e2),
                MipsSquaredEuclidean(vec1, vec2, dim, e2), epsilon);
  }
}

TEST(DistanceMatrix, FixedVectorsSphericalInjection) {
  std::mt19937 gen((std::random_device())());
  const float u_val = 0.68f;
  const float l2_norm = 100.0f;
  const float e2 = (u_val / l2_norm) * (u_val / l2_norm);
  const float epsilon = 1e-5;

  int8_t a[] = {0}, b[] = {0};
  EXPECT_NEAR(0.0f, MipsSquaredEuclidean(a, b, 1, e2), epsilon);

  int8_t c[] = {0, 1}, d[] = {0, 1};
  EXPECT_NEAR(0.0f, MipsSquaredEuclidean(c, d, 2, e2), epsilon);

  int8_t e[] = {0, 1, 2}, f[] = {0, 1, 2};
  EXPECT_NEAR(0.0f, MipsSquaredEuclidean(e, f, 3, e2), epsilon);

  int8_t g[] = {0, 1, 2, 3}, h[] = {0, 1, 2, 3};
  EXPECT_NEAR(0.0f, MipsSquaredEuclidean(g, h, 4, e2), epsilon);

  int8_t i[] = {0, 1, 2, 3, 4}, j[] = {0, 1, 2, 3, 4};
  EXPECT_NEAR(0.0f, MipsSquaredEuclidean(i, j, 5, e2), epsilon);

  int8_t l[] = {0, 1, 2, 3, 4, 5}, k[] = {0, 1, 2, 3, 4, 5};
  EXPECT_NEAR(0.0f, MipsSquaredEuclidean(l, k, 6, e2), epsilon);

  int8_t m[] = {0, 1, 2, 3, 4, 5, 6}, n[] = {0, 1, 2, 3, 4, 5, 6};
  EXPECT_NEAR(0.0f, MipsSquaredEuclidean(m, n, 7, e2), epsilon);

  int8_t o[] = {0, 1, 2, 3, 4, 5, 6, 7}, p[] = {0, 1, 2, 3, 4, 5, 6, 7};
  EXPECT_NEAR(0.0f, MipsSquaredEuclidean(o, p, 8, e2), epsilon);

  int8_t q[] = {0, 1, 2, 3, 4, 5, 6, 7, 8}, r[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  EXPECT_NEAR(0.0f, MipsSquaredEuclidean(q, r, 9, e2), epsilon);

  int8_t s[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
         t[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  EXPECT_NEAR(0.0f, MipsSquaredEuclidean(s, t, 10, e2), epsilon);

  int8_t u[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         v[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  EXPECT_NEAR(0.0178823452f, MipsSquaredEuclidean(u, v, 11, e2), epsilon);

  int8_t w[12] = {0}, x[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  EXPECT_NEAR(0.0235359258f, MipsSquaredEuclidean(w, x, 12, e2), epsilon);

  int8_t y[13] = {0}, z[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  EXPECT_NEAR(0.0302853006f, MipsSquaredEuclidean(y, z, 13, e2), epsilon);

  int8_t x14[14] = {0}, y14[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
  EXPECT_NEAR(0.0382360629f, MipsSquaredEuclidean(x14, y14, 14, e2), epsilon);

  int8_t x15[15] = {0},
         y15[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15};
  EXPECT_NEAR(0.0488716699f, MipsSquaredEuclidean(x15, y15, 15, e2), epsilon);
}

template <size_t M, size_t N>
void TestSquaredEuclideanMatrixSphericalInjection(void) {
  std::mt19937 gen((std::random_device())());

  const float u_val = std::uniform_real_distribution<float>(0.3, 0.9)(gen);
  const size_t batch_size = M;
  const size_t query_size = N;
  size_t dimension = std::uniform_int_distribution<size_t>(2, 128)(gen) * 4;
  size_t matrix_size = batch_size * dimension;
  size_t query_matrix_size = query_size * dimension;

  std::vector<int8_t> matrix1(matrix_size);
  std::vector<int8_t> matrix2(matrix_size);
  std::vector<int8_t> query1(query_matrix_size);
  std::vector<int8_t> query2(query_matrix_size);
  std::vector<float> result1(batch_size * query_size);
  std::vector<float> result2(batch_size * query_size);

  std::uniform_int_distribution<int8_t> dist(-127, 127);
  for (size_t i = 0; i < matrix_size; ++i) {
    matrix1[i] = dist(gen);
  }
  for (size_t i = 0; i < query_matrix_size; ++i) {
    query1[i] = dist(gen);
  }
  float squared_l2_norm = 0.0f;
  for (size_t i = 0; i < matrix_size; i += dimension) {
    float score;
    SquaredNorm2Matrix<int8_t, 1>::Compute(&matrix1[i], dimension, &score);
    squared_l2_norm = std::max(squared_l2_norm, score);
  }
  for (size_t i = 0; i < query_matrix_size; i += dimension) {
    float score;
    SquaredNorm2Matrix<int8_t, 1>::Compute(&query1[i], dimension, &score);
    squared_l2_norm = std::max(squared_l2_norm, score);
  }
  const float e2 = u_val * u_val / squared_l2_norm;
  MatrixTranspose((uint32_t *)(&matrix2[0]), (const uint32_t *)matrix1.data(),
                  dimension / 4, batch_size);
  MatrixTranspose((uint32_t *)(&query2[0]), (const uint32_t *)query1.data(),
                  dimension / 4, query_size);

  for (size_t i = 0; i < query_size; ++i) {
    const int8_t *cur_query = &query1[i * dimension];
    float *query_result = &result1[i * batch_size];

    for (size_t j = 0; j < batch_size; ++j) {
      MipsSquaredEuclideanDistanceMatrix<int8_t, 1, 1>::Compute(
          &matrix1[j * dimension], cur_query, dimension, e2, &query_result[j]);
    }
  }
  MipsSquaredEuclideanDistanceMatrix<int8_t, batch_size, query_size>::Compute(
      &matrix2[0], &query2[0], dimension, e2, &result2[0]);

  for (size_t i = 0; i < batch_size * query_size; ++i) {
    EXPECT_GE(1e-4, std::abs(result1[i] - result2[i]));
  }
}

TEST(DistanceMatrix, MipsSquaredEuclideanSphericalInjection_1x1) {
  TestSquaredEuclideanMatrixSphericalInjection<1, 1>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanSphericalInjection_2x1) {
  TestSquaredEuclideanMatrixSphericalInjection<2, 1>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanSphericalInjection_2x2) {
  TestSquaredEuclideanMatrixSphericalInjection<2, 2>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanSphericalInjection_3x3) {
  TestSquaredEuclideanMatrixSphericalInjection<3, 3>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanSphericalInjection_4x1) {
  TestSquaredEuclideanMatrixSphericalInjection<4, 1>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanSphericalInjection_4x2) {
  TestSquaredEuclideanMatrixSphericalInjection<4, 2>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanSphericalInjection_4x4) {
  TestSquaredEuclideanMatrixSphericalInjection<4, 4>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanSphericalInjection_8x1) {
  TestSquaredEuclideanMatrixSphericalInjection<8, 1>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanSphericalInjection_8x2) {
  TestSquaredEuclideanMatrixSphericalInjection<8, 2>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanSphericalInjection_8x4) {
  TestSquaredEuclideanMatrixSphericalInjection<8, 4>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanSphericalInjection_8x8) {
  TestSquaredEuclideanMatrixSphericalInjection<8, 8>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanSphericalInjection_16x1) {
  TestSquaredEuclideanMatrixSphericalInjection<16, 1>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanSphericalInjection_16x2) {
  TestSquaredEuclideanMatrixSphericalInjection<16, 2>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanSphericalInjection_16x4) {
  TestSquaredEuclideanMatrixSphericalInjection<16, 4>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanSphericalInjection_16x8) {
  TestSquaredEuclideanMatrixSphericalInjection<16, 8>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanSphericalInjection_16x16) {
  TestSquaredEuclideanMatrixSphericalInjection<16, 16>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanSphericalInjection_32x1) {
  TestSquaredEuclideanMatrixSphericalInjection<32, 1>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanSphericalInjection_32x2) {
  TestSquaredEuclideanMatrixSphericalInjection<32, 2>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanSphericalInjection_32x4) {
  TestSquaredEuclideanMatrixSphericalInjection<32, 4>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanSphericalInjection_32x8) {
  TestSquaredEuclideanMatrixSphericalInjection<32, 8>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanSphericalInjection_32x16) {
  TestSquaredEuclideanMatrixSphericalInjection<32, 16>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanSphericalInjection_32x32) {
  TestSquaredEuclideanMatrixSphericalInjection<32, 32>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanSphericalInjection_64x1) {
  TestSquaredEuclideanMatrixSphericalInjection<64, 1>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanSphericalInjection_64x2) {
  TestSquaredEuclideanMatrixSphericalInjection<64, 2>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanSphericalInjection_64x4) {
  TestSquaredEuclideanMatrixSphericalInjection<64, 4>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanSphericalInjection_64x8) {
  TestSquaredEuclideanMatrixSphericalInjection<64, 8>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanSphericalInjection_64x16) {
  TestSquaredEuclideanMatrixSphericalInjection<64, 16>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanSphericalInjection_64x32) {
  TestSquaredEuclideanMatrixSphericalInjection<64, 32>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanSphericalInjection_64x64) {
  TestSquaredEuclideanMatrixSphericalInjection<64, 64>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanSphericalInjection_128x1) {
  TestSquaredEuclideanMatrixSphericalInjection<128, 1>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanSphericalInjection_128x2) {
  TestSquaredEuclideanMatrixSphericalInjection<128, 2>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanSphericalInjection_128x4) {
  TestSquaredEuclideanMatrixSphericalInjection<128, 4>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanSphericalInjection_128x8) {
  TestSquaredEuclideanMatrixSphericalInjection<128, 8>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanSphericalInjection_128x16) {
  TestSquaredEuclideanMatrixSphericalInjection<128, 16>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanSphericalInjection_128x32) {
  TestSquaredEuclideanMatrixSphericalInjection<128, 32>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanSphericalInjection_128x64) {
  TestSquaredEuclideanMatrixSphericalInjection<128, 64>();
}

TEST(DistanceMatrix, MipsSquaredEuclideanSphericalInjection_128x128) {
  TestSquaredEuclideanMatrixSphericalInjection<128, 128>();
}

template <size_t M, size_t N, size_t B, size_t D>
void MipsSphericalInjectionBenchMark(void) {
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
  std::uniform_int_distribution<int8_t> dist(-127, 127);
  for (size_t i = 0; i < matrix_size; ++i) {
    matrix1[i] = dist(gen);
  }
  for (size_t i = 0; i < query_matrix_size; ++i) {
    query1[i] = dist(gen);
  }

  for (size_t i = 0; i < block_size; ++i) {
    size_t start_pos = i * batch_size * dimension;
    MatrixTranspose((uint32_t *)(&matrix2[start_pos]),
                    (const uint32_t *)(&matrix1[start_pos]), dimension / 4,
                    batch_size);
  }
  MatrixTranspose((uint32_t *)(&query2[0]), (const uint32_t *)query1.data(),
                  dimension / 4, query_size);

  float squared_l2_norm = 0.0f;
  for (size_t i = 0; i < matrix_size; i += dimension) {
    float score;
    SquaredNorm2Matrix<int8_t, 1>::Compute(&matrix1[i], dimension, &score);
    squared_l2_norm = std::max(squared_l2_norm, score);
  }
  for (size_t i = 0; i < query_matrix_size; i += dimension) {
    float score;
    SquaredNorm2Matrix<int8_t, 1>::Compute(&query1[i], dimension, &score);
    squared_l2_norm = std::max(squared_l2_norm, score);
  }
  const float e2 = 0.98f / squared_l2_norm;
  ElapsedTime elapsed_time;
  float results[batch_size * query_size];

  std::cout << "# (" << IntelIntrinsics() << ") INT8 " << dimension << "d, "
            << batch_size << " * " << query_size << " * " << block_size
            << std::endl;

  // 1 Batched Euclidean
  elapsed_time.reset();
  for (size_t i = 0; i < block_size; ++i) {
    const int8_t *matrix_batch = &matrix2[i * batch_size * dimension];

    for (size_t j = 0; j < query_size; ++j) {
      const int8_t *current_query = &query1[j * dimension];
      float *current_results = &results[j * batch_size];

      MipsSquaredEuclideanDistanceMatrix<int8_t, batch_size, 1>::Compute(
          matrix_batch, current_query, dimension, e2, current_results);
    }
  }
  std::cout << "* 1 Batched MipsSquaredEuclidean(SphericalInjection) (us) \t"
            << elapsed_time.micro_seconds() << std::endl;

  // N Batched Euclidean
  elapsed_time.reset();
  for (size_t i = 0; i < block_size; ++i) {
    const int8_t *matrix_batch = &matrix2[i * batch_size * dimension];

    MipsSquaredEuclideanDistanceMatrix<int8_t, batch_size, query_size>::Compute(
        matrix_batch, &query2[0], dimension, e2, results);
  }
  std::cout << "* N Batched MipsSquaredErclidean(SphericalInjection) (us) \t"
            << elapsed_time.micro_seconds() << std::endl;

  // Unbatched Euclidean
  elapsed_time.reset();
  for (size_t i = 0; i < block_size; ++i) {
    const int8_t *matrix_batch = &matrix1[i * batch_size * dimension];

    for (size_t j = 0; j < query_size; ++j) {
      const int8_t *current_query = &query1[j * dimension];
      float *current_results = &results[j * batch_size];

      for (size_t k = 0; k < batch_size; ++k) {
        MipsSquaredEuclideanDistanceMatrix<int8_t, 1, 1>::Compute(
            &matrix_batch[k * dimension], current_query, dimension, e2,
            &current_results[k]);
      }
    }
  }
  std::cout << "* Unbatched MipsSquaredEuclidean(SphericalInjection) (us) \t"
            << elapsed_time.micro_seconds() << std::endl;
}

TEST(DistanceMatrix,
     DISABLED_MipsSquaredEuclideanSphericalInjection_Benchmark) {
  MipsSphericalInjectionBenchMark<2, 1, 512, 64>();
  MipsSphericalInjectionBenchMark<2, 2, 512, 64>();
  MipsSphericalInjectionBenchMark<4, 1, 512, 64>();
  MipsSphericalInjectionBenchMark<4, 2, 512, 64>();
  MipsSphericalInjectionBenchMark<4, 4, 512, 64>();
  MipsSphericalInjectionBenchMark<8, 1, 512, 64>();
  MipsSphericalInjectionBenchMark<8, 2, 512, 64>();
  MipsSphericalInjectionBenchMark<8, 4, 512, 64>();
  MipsSphericalInjectionBenchMark<8, 8, 512, 64>();
  MipsSphericalInjectionBenchMark<16, 1, 512, 64>();
  MipsSphericalInjectionBenchMark<16, 2, 512, 64>();
  MipsSphericalInjectionBenchMark<16, 4, 512, 64>();
  MipsSphericalInjectionBenchMark<16, 8, 512, 64>();
  MipsSphericalInjectionBenchMark<16, 16, 512, 64>();
  MipsSphericalInjectionBenchMark<32, 1, 512, 64>();
  MipsSphericalInjectionBenchMark<32, 2, 512, 64>();
  MipsSphericalInjectionBenchMark<32, 4, 512, 64>();
  MipsSphericalInjectionBenchMark<32, 8, 512, 64>();
  MipsSphericalInjectionBenchMark<32, 16, 512, 64>();
  MipsSphericalInjectionBenchMark<32, 32, 512, 64>();
  MipsSphericalInjectionBenchMark<64, 1, 512, 64>();
  MipsSphericalInjectionBenchMark<64, 2, 512, 64>();
  MipsSphericalInjectionBenchMark<64, 4, 512, 64>();
  MipsSphericalInjectionBenchMark<64, 8, 512, 64>();
  MipsSphericalInjectionBenchMark<128, 1, 512, 64>();
  MipsSphericalInjectionBenchMark<1, 1, 1024, 256>();
}
