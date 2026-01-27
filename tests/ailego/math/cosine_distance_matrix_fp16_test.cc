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

inline const char *IntelIntrinsics(void) {
  return internal::CpuFeatures::Intrinsics();
}

inline void MatrixTranspose(Float16 *dst, const Float16 *src, size_t M,
                            size_t N) {
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < M; ++j) {
      dst[j * N + i] = src[i * M + j];
    }
  }
}

template <size_t N>
static float CosineDistance(const FixedVector<Float16, N> &lhs,
                            const FixedVector<Float16, N> &rhs) {
  size_t dimension = lhs.size() + 2;

  float l_norm = 0.0f;
  Norm2Matrix<Float16, 1>::Compute(lhs.data(), N, &l_norm);

  float r_norm = 0.0f;
  Norm2Matrix<Float16, 1>::Compute(rhs.data(), N, &r_norm);

  std::string lhs_normed;

  lhs_normed.resize(dimension * sizeof(uint16_t));

  Float16 *lhs_buf = reinterpret_cast<Float16 *>(&(lhs_normed[0]));

  for (size_t i = 0; i < N; ++i) {
    lhs_buf[i] = lhs[i] / l_norm;
  }
  ::memcpy(reinterpret_cast<uint16_t *>(&(lhs_normed[0])) + N, &l_norm,
           sizeof(float));

  std::string rhs_normed;

  rhs_normed.resize(dimension * sizeof(uint16_t));

  Float16 *rhs_buf = reinterpret_cast<Float16 *>(&(rhs_normed[0]));

  for (size_t i = 0; i < N; ++i) {
    rhs_buf[i] = rhs[i] / r_norm;
  }
  ::memcpy(reinterpret_cast<uint16_t *>(&(rhs_normed[0])) + N, &r_norm,
           sizeof(float));

  return Distance::Cosine(reinterpret_cast<const Float16 *>(lhs_normed.data()),
                          reinterpret_cast<const Float16 *>(rhs_normed.data()),
                          dimension);
}

TEST(DistanceMatrix, Cosine_General) {
  const float epsilon = 1e-3;

  FixedVector<Float16, 2> a{1.0f, 1.0f}, b{1.0f, 1.0f};
  EXPECT_NEAR(0.0f, CosineDistance(a, b), epsilon);

  FixedVector<Float16, 3> c{0.2f, 0.9f, 0.6f}, d{0.3f, 0.5f, 0.7f};
  EXPECT_NEAR(0.072000861f, CosineDistance(c, d), epsilon);

  FixedVector<Float16, 11> e{1.0f, 2.0f, 3.0f, 0.2f, 0.3f, 0.1f,
                             5.2f, 2.1f, 7.1f, 6.8f, 1.2f},
      f{2.0f, 4.0f, 6.0f, 0.6f, 0.7f, 0.9f, 1.0f, 2.3f, 3.4f, 4.5f, 6.4f};
  EXPECT_NEAR(0.28025103f, CosineDistance(e, f), epsilon);

  // FixedVector<Float16, 1> a{0.0f}, b{0.0f};
  // EXPECT_FLOAT_EQ(0.0f, CosineDistance(a, b));

  // FixedVector<Float16, 2> c{0.0f, 0.1f}, d{0.0f, 0.1f};
  // EXPECT_FLOAT_EQ(0.0f, CosineDistance(c, d));

  // FixedVector<Float16, 3> e{0.0f, 0.1f, 0.2f}, f{0.0f, 0.1f, 0.2f};
  // EXPECT_FLOAT_EQ(0.0f, CosineDistance(e, f));

  // FixedVector<Float16, 4> g{0.0f, 0.1f, 0.2f, 0.3f}, h{0.0f, 0.1f, 0.2f,
  // 0.3f}; EXPECT_FLOAT_EQ(0.0f, CosineDistance(g, h));

  // FixedVector<Float16, 5> i{0.0f, 0.1f, 0.2f, 0.3f, 0.4f},
  //     j{0.0f, 0.1f, 0.2f, 0.3f, 0.4f};
  // EXPECT_FLOAT_EQ(0.0f, CosineDistance(i, j));

  // FixedVector<Float16, 6> l{0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f},
  //     k{0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
  // EXPECT_FLOAT_EQ(0.0f, CosineDistance(l, k));

  // FixedVector<Float16, 7> m{0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f},
  //     n{0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
  // EXPECT_FLOAT_EQ(0.0f, CosineDistance(m, n));

  // FixedVector<Float16, 8> o{0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f},
  //     p{0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f};
  // EXPECT_FLOAT_EQ(0.0f, CosineDistance(o, p));

  // FixedVector<Float16, 9> q{0.0f, 0.1f, 0.2f, 0.3f, 0.4f,
  //                           0.5f, 0.6f, 0.7f, 0.8f},
  //     r{0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
  // EXPECT_FLOAT_EQ(0.0f, CosineDistance(q, r));

  // FixedVector<Float16, 10> s{0.0f, 0.1f, 0.2f, 0.3f, 0.4f,
  //                            0.5f, 0.6f, 0.7f, 0.8f, 0.9f},
  //     t{0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f};
  // EXPECT_FLOAT_EQ(0.0f, CosineDistance(s, t));

  // FixedVector<Float16, 11> u{0.0f},
  //     v{0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f};
  // EXPECT_TRUE(MathHelper::IsAlmostEqual(3.84983f, CosineDistance(u, v),
  // 1000));

  // FixedVector<Float16, 12> w{0.0f},
  //     x{0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f,
  //     0.9f, 1.0f, 1.1f};
  // EXPECT_TRUE(MathHelper::IsAlmostEqual(5.05897f, CosineDistance(w, x),
  // 1000));

  // FixedVector<Float16, 13> y{0.0f}, z{0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f,
  // 0.6f,
  //                                     0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f};
  // EXPECT_TRUE(MathHelper::IsAlmostEqual(6.499438f, CosineDistance(y, z),
  // 1000));

  // FixedVector<Float16, 14> x14{0.0f},
  //     y14{0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
  //         0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 2.0f};
  // EXPECT_TRUE(
  //     MathHelper::IsAlmostEqual(10.49944f, CosineDistance(x14, y14), 1000));

  // FixedVector<Float16, 15> x15{0.0f},
  //     y15{0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f,
  //         0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 2.0f, 3.0f};
  // EXPECT_TRUE(
  //     MathHelper::IsAlmostEqual(19.49944f, CosineDistance(x15, y15), 1000));
}

#if 0
template <size_t M, size_t N>
void TestCosineMatrix(void) {
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
      CosineDistanceMatrix<Float16, 1, 1>::Compute(
          &matrix1[j * dimension], cur_query, dimension, &query_result[j]);
    }
  }
  CosineDistanceMatrix<Float16, batch_size, query_size>::Compute(
      &matrix2[0], &query2[0], dimension, &result2[0]);

  for (size_t i = 0; i < batch_size * query_size; ++i) {
    // EXPECT_FLOAT_EQ(result1[i], result2[i]);
    EXPECT_TRUE(MathHelper::IsAlmostEqual(result1[i], result2[i], 10000));
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
  TestCosineMatrix<64, 64>();
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
  TestCosineMatrix<128, 64>();
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

      CosineDistanceMatrix<Float16, batch_size, 1>::Compute(
          matrix_batch, current_query, dimension, current_results);
    }
  }
  std::cout << "* 1 Batched Cosine (us) \t" << elapsed_time.micro_seconds()
            << std::endl;

  // N Batched Euclidean
  elapsed_time.reset();
  for (size_t i = 0; i < block_size; ++i) {
    const Float16 *matrix_batch = &matrix2[i * batch_size * dimension];

    CosineDistanceMatrix<Float16, batch_size, query_size>::Compute(
        matrix_batch, &query2[0], dimension, results);
  }
  std::cout << "* N Batched Cosine (us) \t" << elapsed_time.micro_seconds()
            << std::endl;

  // Unbatched Euclidean
  elapsed_time.reset();
  for (size_t i = 0; i < block_size; ++i) {
    const Float16 *matrix_batch = &matrix1[i * batch_size * dimension];

    for (size_t j = 0; j < query_size; ++j) {
      const Float16 *current_query = &query1[j * dimension];
      float *current_results = &results[j * batch_size];

      for (size_t k = 0; k < batch_size; ++k) {
        CosineDistanceMatrix<Float16, 1, 1>::Compute(
            &matrix_batch[k * dimension], current_query, dimension,
            &current_results[k]);
      }
    }
  }
  std::cout << "* Unbatched Cosine (us) \t" << elapsed_time.micro_seconds()
            << std::endl;
}

TEST(DistanceMatrix, DISABLED_Cosine_Benchmark) {
  CosineBenchmark<2, 1, 512, 64>();
  CosineBenchmark<2, 2, 512, 64>();
  CosineBenchmark<4, 1, 512, 64>();
  CosineBenchmark<4, 2, 512, 64>();
  CosineBenchmark<4, 4, 512, 64>();
  CosineBenchmark<8, 1, 512, 64>();
  CosineBenchmark<8, 2, 512, 64>();
  CosineBenchmark<8, 4, 512, 64>();
  CosineBenchmark<8, 8, 512, 64>();
  CosineBenchmark<16, 1, 512, 64>();
  CosineBenchmark<16, 2, 512, 64>();
  CosineBenchmark<16, 4, 512, 64>();
  CosineBenchmark<16, 8, 512, 64>();
  CosineBenchmark<16, 16, 512, 64>();
  CosineBenchmark<32, 1, 512, 64>();
  CosineBenchmark<32, 2, 512, 64>();
  CosineBenchmark<32, 4, 512, 64>();
  CosineBenchmark<32, 8, 512, 64>();
  CosineBenchmark<32, 16, 512, 64>();
  CosineBenchmark<32, 32, 512, 64>();
  CosineBenchmark<64, 1, 512, 64>();
  CosineBenchmark<64, 2, 512, 64>();
  CosineBenchmark<64, 4, 512, 64>();
  CosineBenchmark<64, 8, 512, 64>();
  CosineBenchmark<128, 1, 512, 64>();
  CosineBenchmark<1, 1, 1024, 256>();
}

#endif
