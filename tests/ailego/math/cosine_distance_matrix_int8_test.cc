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

#include <string>
#include <ailego/container/bitmap.h>
#include <ailego/internal/cpu_features.h>
#include <ailego/math/distance.h>
#include <ailego/utility/math_helper.h>
#include <gtest/gtest.h>
#include <zvec/ailego/container/vector.h>
#include <zvec/ailego/utility/time_helper.h>

using namespace zvec::ailego;

inline const char *IntelIntrinsics(void) {
  return internal::CpuFeatures::Intrinsics();
}

inline void MatrixTranspose(uint32_t *dst, const uint32_t *src, size_t M,
                            size_t N) {
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < M; ++j) {
      dst[j * N + i] = src[i * M + j];
    }
  }
}

template <size_t N>
static float CosineDistance(const FixedVector<int8_t, N> &lhs,
                            const FixedVector<int8_t, N> &rhs) {
  size_t dimension = lhs.size() + 4;

  float l_norm = 0.0f;
  Norm2Matrix<int8_t, 1>::Compute(lhs.data(), N, &l_norm);

  float r_norm = 0.0f;
  Norm2Matrix<int8_t, 1>::Compute(rhs.data(), N, &r_norm);

  std::string lhs_normed;

  lhs_normed.resize(dimension * sizeof(int8_t));

  int8_t *lhs_buf = reinterpret_cast<int8_t *>(&(lhs_normed[0]));

  for (size_t i = 0; i < N; ++i) {
    lhs_buf[i] = lhs[i] / l_norm;
  }
  ::memcpy(reinterpret_cast<int8_t *>(&(lhs_normed[0])) + N, &l_norm,
           sizeof(float));

  std::string rhs_normed;

  rhs_normed.resize(dimension * sizeof(int8_t));

  int8_t *rhs_buf = reinterpret_cast<int8_t *>(&(rhs_normed[0]));

  for (size_t i = 0; i < N; ++i) {
    rhs_buf[i] = rhs[i] / r_norm;
  }
  ::memcpy(reinterpret_cast<int8_t *>(&(rhs_normed[0])) + N, &r_norm,
           sizeof(float));

  return Distance::Cosine(reinterpret_cast<const int8_t *>(lhs_normed.data()),
                          reinterpret_cast<const int8_t *>(rhs_normed.data()),
                          dimension);
}

#if 0

TEST(DistanceMatrix, Cosine_General) {
  int8_t a8[] = {127, 0, 1, 2, -127, -127, -127, -127};
  int8_t b8[] = {-127, -127, -127, -127, 1, 2, 1, 127};
  int8_t a16[] = {127, 127, 16,   3,   100,  -127, 1,    2,
                  3,   4,   -127, 100, -127, -127, -127, -127};
  int8_t b16[] = {-127, 123, -127, -127, -127, -127, 127, 127,
                  1,    2,   3,    4,    127,  127,  121, 16};
  int8_t a32[] = {127, 127,  0,    0,   -127, -127, 0,    0,    0,    0, 0,
                  0,   -127, -127, 127, 127,  0,    0,    -127, -127, 0, 0,
                  127, 127,  127,  127, 0,    0,    -127, -127, 0,    0};
  int8_t b32[] = {-127, -127, 0,    0,    127,  127, 0,   0,   0,   0, 0,
                  0,    127,  127,  -127, -127, 0,   0,   127, 127, 0, 0,
                  -127, -127, -127, -127, 0,    0,   127, 127, 0,   0};

  int8_t a47[] = {127, 2, 0,    0,    -127, -127, 0,    0,    0,    0,
                  0,   0, -127, -127, 127,  127,  0,    0,    -127, -127,
                  0,   0, 127,  5,    127,  127,  0,    0,    -127, -127,
                  0,   0, -127, 112,  -127, -127, -127, -127, 127,  127,
                  1,   2, 3,    4,    127,  127,  120};
  int8_t b47[] = {-127, 1, 0,    0,   127,  127,  0,   0,    0,   0,
                  0,    0, 127,  127, -127, -127, 0,   0,    127, 127,
                  0,    0, -127, 3,   -127, -127, 0,   0,    127, 127,
                  0,    0, 127,  127, 80,   111,  122, -127, 1,   2,
                  3,    4, -127, 112, -127, -127, -127};

  EXPECT_FLOAT_EQ(1.4109956f,
                  CosineDistance(*FixedVector<int8_t, 8>::Cast(a8),
                                 *FixedVector<int8_t, 8>::Cast(b8)));
  EXPECT_FLOAT_EQ(1.3013078f,
                  CosineDistance(*FixedVector<int8_t, 16>::Cast(a16),
                                 *FixedVector<int8_t, 16>::Cast(b16)));
  EXPECT_FLOAT_EQ(2.0f, CosineDistance(*FixedVector<int8_t, 32>::Cast(a32),
                                       *FixedVector<int8_t, 32>::Cast(b32)));
  EXPECT_FLOAT_EQ(1.7623165f,
                  CosineDistance(*FixedVector<int8_t, 47>::Cast(a47),
                                 *FixedVector<int8_t, 47>::Cast(b47)));
}

template <size_t M, size_t N>
void TestCosineMatrix(void) {
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
      CosineDistanceMatrix<int8_t, 1, 1>::Compute(
          &matrix1[j * dimension], cur_query, dimension, &query_result[j]);
    }
  }
  CosineDistanceMatrix<int8_t, batch_size, query_size>::Compute(
      &matrix2[0], &query2[0], dimension, &result2[0]);

  for (size_t i = 0; i < batch_size * query_size; ++i) {
    EXPECT_FLOAT_EQ(result1[i], result2[i]);
  }
}

TEST(DistanceMatrix, Cosine_1x1) {
  TestCosineMatrix<1, 1>();
}

TEST(DistanceMatrix, Cosine_2x1) {
  TestCosineMatrix<2, 1>();
}

TEST(DistanceMatrix, Cosine_2x2) {
  TestCosineMatrix<2, 2>();
}

TEST(DistanceMatrix, Cosine_3x3) {
  TestCosineMatrix<3, 3>();
}

TEST(DistanceMatrix, Cosine_4x1) {
  TestCosineMatrix<4, 1>();
}

TEST(DistanceMatrix, Cosine_4x2) {
  TestCosineMatrix<4, 2>();
}

TEST(DistanceMatrix, Cosine_4x4) {
  TestCosineMatrix<4, 4>();
}

TEST(DistanceMatrix, Cosine_8x1) {
  TestCosineMatrix<8, 1>();
}

TEST(DistanceMatrix, Cosine_8x2) {
  TestCosineMatrix<8, 2>();
}

TEST(DistanceMatrix, Cosine_8x4) {
  TestCosineMatrix<8, 4>();
}

TEST(DistanceMatrix, Cosine_8x8) {
  TestCosineMatrix<8, 8>();
}

TEST(DistanceMatrix, Cosine_16x1) {
  TestCosineMatrix<16, 1>();
}

TEST(DistanceMatrix, Cosine_16x2) {
  TestCosineMatrix<16, 2>();
}

TEST(DistanceMatrix, Cosine_16x4) {
  TestCosineMatrix<16, 4>();
}

TEST(DistanceMatrix, Cosine_16x8) {
  TestCosineMatrix<16, 8>();
}

TEST(DistanceMatrix, Cosine_16x16) {
  TestCosineMatrix<16, 16>();
}

TEST(DistanceMatrix, Cosine_32x1) {
  TestCosineMatrix<32, 1>();
}

TEST(DistanceMatrix, Cosine_32x2) {
  TestCosineMatrix<32, 2>();
}

TEST(DistanceMatrix, Cosine_32x4) {
  TestCosineMatrix<32, 4>();
}

TEST(DistanceMatrix, Cosine_32x8) {
  TestCosineMatrix<32, 8>();
}

TEST(DistanceMatrix, Cosine_32x16) {
  TestCosineMatrix<32, 16>();
}

TEST(DistanceMatrix, Cosine_32x32) {
  TestCosineMatrix<32, 32>();
}

TEST(DistanceMatrix, Cosine_64x1) {
  TestCosineMatrix<64, 1>();
}

TEST(DistanceMatrix, Cosine_64x2) {
  TestCosineMatrix<64, 2>();
}

TEST(DistanceMatrix, Cosine_64x4) {
  TestCosineMatrix<64, 4>();
}

TEST(DistanceMatrix, Cosine_64x8) {
  TestCosineMatrix<64, 8>();
}

TEST(DistanceMatrix, Cosine_64x16) {
  TestCosineMatrix<64, 16>();
}

TEST(DistanceMatrix, Cosine_64x32) {
  TestCosineMatrix<64, 32>();
}

TEST(DistanceMatrix, Cosine_64x64) {
  TestCosineMatrix<64, 128>();
}

TEST(DistanceMatrix, Cosine_128x1) {
  TestCosineMatrix<128, 1>();
}

TEST(DistanceMatrix, Cosine_128x2) {
  TestCosineMatrix<128, 2>();
}

TEST(DistanceMatrix, Cosine_128x4) {
  TestCosineMatrix<128, 4>();
}

TEST(DistanceMatrix, Cosine_128x8) {
  TestCosineMatrix<128, 8>();
}

TEST(DistanceMatrix, Cosine_128x16) {
  TestCosineMatrix<128, 16>();
}

TEST(DistanceMatrix, Cosine_128x32) {
  TestCosineMatrix<128, 32>();
}

TEST(DistanceMatrix, Cosine_128x64) {
  TestCosineMatrix<128, 128>();
}

TEST(DistanceMatrix, Cosine_128x128) {
  TestCosineMatrix<128, 128>();
}

template <size_t M, size_t N, size_t B, size_t D>
void CosineBenchmark(void) {
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

  // 1 Batched Cosine
  elapsed_time.reset();
  for (size_t i = 0; i < block_size; ++i) {
    const int8_t *matrix_batch = &matrix2[i * batch_size * dimension];

    for (size_t j = 0; j < query_size; ++j) {
      const int8_t *current_query = &query1[j * dimension];
      float *current_results = &results[j * batch_size];

      CosineDistanceMatrix<int8_t, batch_size, 1>::Compute(
          matrix_batch, current_query, dimension, current_results);
    }
  }
  std::cout << "* 1 Batched Cosine (us) \t" << elapsed_time.micro_seconds()
            << std::endl;

  // N Batched Cosine
  elapsed_time.reset();
  for (size_t i = 0; i < block_size; ++i) {
    const int8_t *matrix_batch = &matrix2[i * batch_size * dimension];

    CosineDistanceMatrix<int8_t, batch_size, query_size>::Compute(
        matrix_batch, &query2[0], dimension, results);
  }
  std::cout << "* N Batched Cosine (us) \t" << elapsed_time.micro_seconds()
            << std::endl;

  // Unbatched Cosine
  elapsed_time.reset();
  for (size_t i = 0; i < block_size; ++i) {
    const int8_t *matrix_batch = &matrix1[i * batch_size * dimension];

    for (size_t j = 0; j < query_size; ++j) {
      const int8_t *current_query = &query1[j * dimension];
      float *current_results = &results[j * batch_size];

      for (size_t k = 0; k < batch_size; ++k) {
        CosineDistanceMatrix<int8_t, 1, 1>::Compute(
            &matrix_batch[k * dimension], current_query, dimension,
            &current_results[k]);
      }
    }
  }
  std::cout << "* Unbatched Cosine (us) \t" << elapsed_time.micro_seconds()
            << std::endl;
}

TEST(DistanceMatrix, DISABLED_Cosine_Benchmark) {
  CosineBenchmark<2, 1, 512, 128>();
  CosineBenchmark<2, 2, 512, 128>();
  CosineBenchmark<4, 1, 512, 128>();
  CosineBenchmark<4, 2, 512, 128>();
  CosineBenchmark<4, 4, 512, 128>();
  CosineBenchmark<8, 1, 512, 128>();
  CosineBenchmark<8, 2, 512, 128>();
  CosineBenchmark<8, 4, 512, 128>();
  CosineBenchmark<8, 8, 512, 128>();
  CosineBenchmark<16, 1, 512, 128>();
  CosineBenchmark<16, 2, 512, 128>();
  CosineBenchmark<16, 4, 512, 128>();
  CosineBenchmark<16, 8, 512, 128>();
  CosineBenchmark<16, 16, 512, 128>();
  CosineBenchmark<32, 1, 512, 128>();
  CosineBenchmark<32, 2, 512, 128>();
  CosineBenchmark<32, 4, 512, 128>();
  CosineBenchmark<32, 8, 512, 128>();
  CosineBenchmark<32, 16, 512, 128>();
  CosineBenchmark<32, 32, 512, 128>();
  CosineBenchmark<64, 1, 512, 128>();
  CosineBenchmark<64, 2, 512, 128>();
  CosineBenchmark<64, 4, 512, 128>();
  CosineBenchmark<64, 8, 512, 128>();
  CosineBenchmark<128, 1, 512, 128>();
  CosineBenchmark<1, 1, 1024, 256>();
}

#endif